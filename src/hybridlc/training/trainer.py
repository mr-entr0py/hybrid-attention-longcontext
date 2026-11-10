"""Single-GPU training loop (protocol §6.3, §7, §8, §13.3 G-div / G-OOM, §18 runbook).

``train(cfg, resume=True, allow_dirty=False)`` runs one :class:`RunConfig` to completion:

* model from ``cfg.impl.model_factory(model_cfg, seed_index)`` (fp32 master weights), data from the
  stateless ``StageStream`` / ``TEFStream`` (``get(step, micro)`` is a pure function, so nothing about
  the data has to be saved on resume), loss from ``chunked_ce`` (fp32 logits, z-loss, pad ignored);
* AdamW(fused) with the §7.1 weight-decay groups, global-norm clipping 1.0, bf16 autocast, gradient
  accumulation over ``data.accum`` micro-batches, per-block activation checkpointing when
  ``runtime.activation_checkpointing`` (Stage B / TEF at 32K);
* LR from :func:`schedules.lr_at` as a pure function of the step: Stage A warm-up→constant, Stage B
  linear decay from the branch point S (model *and* AdamW state carried from the parent's
  ``final.pt``), LR-sweep compressed WSD, TEF re-warm→constant;
* atomic full checkpoints every ``checkpoint.every`` steps (keep last 2) plus bf16 milestones and a
  full ``final.pt``; metrics JSONL every 10 steps; a per-step trace (loss / grad-norm bit patterns and
  batch hash) for the VT-13 bitwise comparisons; validation on the LM-2 subset at intervals;
* G-div: NaN/inf loss or grad-norm, or loss > EMA_0.99 + 1.0 for > 50 steps → one automatic restart
  from the last checkpoint (a second divergence marks the run failed, exit code ``EXIT_DIVERGED``);
* G-OOM: a CUDA OOM writes an ``oom`` registry record and exits with ``EXIT_OOM``; the runner
  auto-resumes from the last checkpoint (3 consecutive no-progress OOMs → failed).

Combining micro-batches. ``chunked_ce`` returns a (weighted) *mean* over the non-ignored tokens of one
micro-batch. The step objective is the mean over all loss tokens of the step (or, for the TEF, the
D014 ½/½ weighted mean encoded in ``loss_weights``), so micro-batch m is scaled by W_m / W where
W_m = Σ_i w_i over its non-ignored tokens (w ≡ 1 without loss weights) and W = Σ_m W_m. The z-loss is
a mean over loss tokens and is scaled by n_m / N. All batches of a step are fetched up front (they are
small CPU tensors) so that W and N are known before the first backward.

The trainer draws no random numbers itself (no dropout anywhere, protocol §3.1); the global RNG
states are still saved and restored for completeness.
"""
from __future__ import annotations

import contextlib
import functools
import json
import math
import os
import shutil
import signal
import struct
import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch
from torch import nn

from . import checkpoint as ckpt
from .compat import claim_gpu, env_info, git_commit, git_dirty_paths, triton_cache_hash
from .config import RunConfig, dump_yaml, import_object
from .gdiv import DivergenceError, GDivMonitor
from .optim import build_optimizer, set_lr
from .registry import Registry
from .schedules import lr_at

EXIT_OK = 0
EXIT_OOM = 75        # G-OOM: retry with auto-resume
EXIT_DIVERGED = 76   # G-div: second divergence -> run failed (terminal)
EXIT_REFUSED = 77    # refused to start (dirty tree, config mismatch, missing parent) -> terminal

#: Test-only fault injection (VT-13 kill -9, G-div, G-OOM). JSON in this environment variable, e.g.
#: ``{"kill_at": [137, 1]}`` (SIGKILL after micro-batch 1 of step 137), ``{"kill_in_ckpt": 150}``
#: (SIGKILL while the step-150 checkpoint is being written), ``{"nan_at": 30}``, ``{"oom_at": 30}``.
#: A fault fires once per run: a marker file ``<run_dir>/fault_<name>.fired`` disarms it.
FAULT_ENV = "HYBRIDLC_FAULT"


class RefusedError(RuntimeError):
    """The run must not start (dirty tree without --allow-dirty, missing parent checkpoint, ...)."""


# ----------------------------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------------------------
def float_hex(x: float) -> str:
    """Bit pattern of a float64 (exact, for bitwise trace comparisons)."""
    return struct.pack(">d", float(x)).hex()


class JsonlWriter:
    """Append-only JSONL file, flushed per line (metrics, trace, events)."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(path, "a", buffering=1)

    def write(self, rec: dict[str, Any]) -> None:
        self.f.write(json.dumps(rec, default=_json_default, allow_nan=True) + "\n")

    def close(self) -> None:
        self.f.close()


def _json_default(o: Any) -> Any:
    if isinstance(o, torch.Tensor):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def read_jsonl_last_per_step(path: str | os.PathLike, kind: str | None = None) -> list[dict[str, Any]]:
    """Read a metrics/trace JSONL written across resumes; keep the *last* record per step.

    Steps re-run after a resume (between the last checkpoint and the crash) appear twice; the later
    record is the one that belongs to the surviving trajectory.
    """
    by_step: dict[int, dict[str, Any]] = {}
    p = Path(path)
    if not p.exists():
        return []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:  # a line torn by kill -9
                continue
            if kind is not None and r.get("kind") != kind:
                continue
            if "step" in r:
                by_step[int(r["step"])] = r
    return [by_step[k] for k in sorted(by_step)]


def _maybe_cuda_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


# ----------------------------------------------------------------------------------------------
# faults (tests only)
# ----------------------------------------------------------------------------------------------
class Faults:
    def __init__(self, run_dir: Path):
        raw = os.environ.get(FAULT_ENV)
        self.spec: dict[str, Any] = json.loads(raw) if raw else {}
        self.run_dir = run_dir

    def _armed(self, name: str) -> bool:
        return name in self.spec and not (self.run_dir / f"fault_{name}.fired").exists()

    def _fire(self, name: str) -> None:
        marker = self.run_dir / f"fault_{name}.fired"
        marker.parent.mkdir(parents=True, exist_ok=True)
        with open(marker, "w") as f:
            f.write(str(time.time()))
            f.flush()
            os.fsync(f.fileno())

    def after_micro(self, step: int, micro: int) -> None:
        if self._armed("kill_at") and list(self.spec["kill_at"]) == [step, micro]:
            self._fire("kill_at")
            os.kill(os.getpid(), signal.SIGKILL)

    def loss_override(self, step: int) -> float | None:
        if self._armed("nan_at") and int(self.spec["nan_at"]) == step:
            self._fire("nan_at")
            return float("nan")
        return None

    def maybe_oom(self, step: int) -> None:
        if self._armed("oom_at") and int(self.spec["oom_at"]) == step:
            self._fire("oom_at")
            raise torch.cuda.OutOfMemoryError("injected OOM (test fault)")

    def ckpt_hook(self, step: int) -> Callable[[Path], None] | None:
        if self._armed("kill_in_ckpt") and int(self.spec["kill_in_ckpt"]) == step:
            def hook(_tmp: Path) -> None:
                self._fire("kill_in_ckpt")
                os.kill(os.getpid(), signal.SIGKILL)
            return hook
        return None


# ----------------------------------------------------------------------------------------------
# activation checkpointing and attention statistics
# ----------------------------------------------------------------------------------------------
def _ckpt_forward(orig: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    from torch.utils.checkpoint import checkpoint

    if not torch.is_grad_enabled():
        return orig(*args, **kwargs)
    # No stochastic ops in the model (protocol §3.1: no dropout), so RNG state need not be replayed.
    return checkpoint(orig, *args, use_reentrant=False, preserve_rng_state=False, **kwargs)


def enable_activation_checkpointing(model: nn.Module) -> str:
    """Per-block activation checkpointing (protocol §6.3, Stage B / TEF at 32K).

    Preference order (INTERFACES.md, HybridLM): ``model.set_activation_checkpointing(True)`` if the
    model provides it; otherwise every block of the first ``nn.ModuleList`` attribute named ``blocks``
    or ``layers`` has its ``forward`` wrapped in non-reentrant ``torch.utils.checkpoint``.
    """
    fn = getattr(model, "set_activation_checkpointing", None)
    if callable(fn):
        fn(True)
        return "model.set_activation_checkpointing"
    for attr in ("blocks", "layers"):
        blocks = getattr(model, attr, None)
        if isinstance(blocks, nn.ModuleList) and len(blocks):
            for blk in blocks:
                if not getattr(blk, "_hybridlc_ckpt", False):
                    blk.forward = functools.partial(_ckpt_forward, blk.forward)  # type: ignore[method-assign]
                    blk._hybridlc_ckpt = True  # type: ignore[attr-defined]
            return f"wrapped model.{attr}[*].forward"
    raise RuntimeError("activation checkpointing requested but the model has neither "
                       "set_activation_checkpointing() nor a blocks/layers ModuleList")


class AttnStats:
    """Cheap optional hook (INTERFACES.md): modules with a boolean ``collect_attn_stats`` attribute
    store ``attn_stats = {"max_logit": float, "entropy": float}`` after a forward while it is True."""

    def __init__(self, model: nn.Module, enabled: bool):
        self.mods = [(n, m) for n, m in model.named_modules() if hasattr(m, "collect_attn_stats")] if enabled else []

    def arm(self, on: bool) -> None:
        for _, m in self.mods:
            m.collect_attn_stats = on
            if on:
                m.attn_stats = None

    def read(self) -> dict[str, Any] | None:
        vals = [(n, getattr(m, "attn_stats", None)) for n, m in self.mods]
        vals = [(n, v) for n, v in vals if v]
        if not vals:
            return None
        return {
            "attn_max_logit": max(float(v["max_logit"]) for _, v in vals),
            "attn_entropy": sum(float(v["entropy"]) for _, v in vals) / len(vals),
            "attn_per_layer": {n: {k: float(x) for k, x in v.items()} for n, v in vals},
        }


# ----------------------------------------------------------------------------------------------
# runtime setup
# ----------------------------------------------------------------------------------------------
def setup_runtime(cfg: RunConfig) -> torch.device:
    """Environment variables, determinism mode, backend pins and the GPU memory cap."""
    try:
        from hybridlc.utils.gpu import set_production_env

        set_production_env(overwrite=False)
    except ImportError:  # pragma: no cover - utils always present once landed
        os.environ.setdefault("FLA_FLASH_QLA", "0")
    for k, v in cfg.runtime.env.items():
        os.environ[k] = str(v)
    if cfg.runtime.audit:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if cfg.runtime.cpu_threads:
        torch.set_num_threads(int(cfg.runtime.cpu_threads))
    # Protocol §18: no cuDNN SDPA in training (production and audit).
    if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        torch.backends.cuda.enable_cudnn_sdp(False)
    dev = torch.device(cfg.runtime.device)
    if dev.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("runtime.device is cuda but CUDA is unavailable")
        claim_gpu(cfg.runtime.mem_fraction)
        # TF32 for fp32 matmuls stays at torch's default (off); bf16 autocast covers the hot path.
    return dev


def _autocast(cfg: RunConfig, device: torch.device) -> contextlib.AbstractContextManager:
    if cfg.runtime.precision == "bf16":
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    if cfg.runtime.precision == "fp32":
        return contextlib.nullcontext()
    raise ValueError(f"unknown precision {cfg.runtime.precision!r}")


def build_stream(cfg: RunConfig) -> Any:
    d = cfg.data
    if d.stream == "stage":
        cls = import_object(cfg.impl.stage_stream)
        return cls(d.stage, d.data_seed, d.seq_len, d.micro_batch, d.accum, (d.mix or None), **d.kwargs)
    if d.stream == "tef":
        cls = import_object(cfg.impl.tef_stream)
        return cls(cfg.seed_index, cfg.schedule.steps, seq_len=d.seq_len, micro_batch=d.micro_batch,
                   accum=d.accum, **d.kwargs)
    raise ValueError(f"unknown data.stream {d.stream!r}")


def build_val_stream(cfg: RunConfig) -> Any | None:
    v = cfg.validation
    if v.every <= 0 and v.final_n_batches <= 0:
        return None
    cls = import_object(cfg.impl.stage_stream)
    kw = dict(cfg.data.kwargs) if cfg.data.stream == "stage" else {}
    kw.update(getattr(v, "kwargs", {}) or {})
    return cls(v.stage, v.data_seed, v.seq_len, v.micro_batch, 1, (v.mix or None), **kw)


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out = dict(batch)
    for k in ("input_ids", "labels", "cu_seqlens", "position_ids", "loss_weights"):
        if k in out and isinstance(out[k], torch.Tensor):
            out[k] = out[k].to(device, non_blocking=True)
    return out


# ----------------------------------------------------------------------------------------------
# the trainer
# ----------------------------------------------------------------------------------------------
class Trainer:
    def __init__(self, cfg: RunConfig, allow_dirty: bool = False):
        self.cfg = cfg
        self.allow_dirty = allow_dirty
        self.run_dir = cfg.run_dir
        self.ckpt_dir = cfg.ckpt_dir
        self.registry = Registry(cfg.registry_path)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.faults = Faults(self.run_dir)
        self.sidecar_path = self.run_dir / "attempts.json"   # survives checkpoint reloads
        self.loss_fn = import_object(cfg.impl.loss_fn)
        self.model_factory = import_object(cfg.impl.model_factory)

    # -- sidecar (restart counters that must not be rolled back by a checkpoint load) -------------
    def _sidecar(self) -> dict[str, Any]:
        if self.sidecar_path.exists():
            return json.loads(self.sidecar_path.read_text())
        return {"n_starts": 0, "gdiv_restarts": 0, "lr_override": None, "events": []}

    def _save_sidecar(self, sc: dict[str, Any]) -> None:
        tmp = self.sidecar_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(sc, indent=1))
        os.replace(tmp, self.sidecar_path)

    # -- provenance ---------------------------------------------------------------------------
    def _meta(self, env: dict[str, Any]) -> dict[str, Any]:
        c = self.cfg
        return {
            "run_id": c.run_id, "config": c.to_dict(), "config_hash": c.config_hash(),
            "model_config_hash": c.model_config_hash(), "git_commit": env.get("git_commit") or git_commit(),
            "git_dirty_code_paths": env.get("git_dirty_code_paths", []), "allow_dirty": self.allow_dirty,
            "triton_cache_hash": triton_cache_hash(), "env": env,
        }

    # -- main entry -----------------------------------------------------------------------------
    def run(self, resume: bool = True) -> dict[str, Any]:
        c = self.cfg
        dirty = git_dirty_paths()
        if dirty and not self.allow_dirty:
            self.registry.append(c.run_id, "failed", reason="dirty_tree", dirty=dirty[:50])
            raise RefusedError(f"git tree has uncommitted changes under code paths ({len(dirty)} entries); "
                               "commit them or pass --allow-dirty")
        device = setup_runtime(c)
        env = env_info()
        env["device"] = str(device)
        dump_yaml(c.to_dict(), self.run_dir / "config.resolved.yaml")
        (self.run_dir / "env.json").write_text(json.dumps(env, indent=1, default=str))

        if not resume and ckpt.latest_checkpoint(self.ckpt_dir) is not None:
            # --no-resume on a run that has checkpoints: start over, keeping the old ones aside.
            aside = self.ckpt_dir.with_name(f"{self.ckpt_dir.name}.old{int(time.time())}")
            shutil.move(str(self.ckpt_dir), str(aside))
            self.ckpt_dir.mkdir(parents=True, exist_ok=True)
            self.registry.append(c.run_id, "note", note="--no-resume: previous checkpoints moved aside",
                                 moved_to=str(aside))
        sc = self._sidecar()
        sc["n_starts"] += 1
        self._save_sidecar(sc)
        while True:
            try:
                return self._attempt(device, env, resume=resume, attempt=sc["n_starts"])
            except DivergenceError as e:
                sc = self._sidecar()
                self.registry.append(c.run_id, "diverged", step=e.step, reason=e.reason, loss=e.loss,
                                     ema=e.ema, restarts_so_far=sc["gdiv_restarts"])
                if sc["gdiv_restarts"] >= c.gdiv.max_restarts:
                    self.registry.append(c.run_id, "failed", reason="diverged_twice", step=e.step)
                    raise
                sc["gdiv_restarts"] += 1
                sc["events"].append({"ts": time.time(), "event": "gdiv_restart", "step": e.step, "reason": e.reason})
                if c.gdiv.fallback_lr is not None:
                    # FEAS (D034): restart once from scratch at the fallback LR. The aborted attempt's
                    # checkpoints are moved aside (kept for the audit trail).
                    sc["lr_override"] = float(c.gdiv.fallback_lr)
                    aside = self.ckpt_dir.with_name(self.ckpt_dir.name + f".diverged{sc['gdiv_restarts']}")
                    if self.ckpt_dir.exists():
                        shutil.move(str(self.ckpt_dir), str(aside))
                    self.ckpt_dir.mkdir(parents=True, exist_ok=True)
                    self.registry.append(c.run_id, "restarted", mode="from_scratch", lr=sc["lr_override"],
                                         moved_ckpts_to=str(aside))
                else:
                    self.registry.append(c.run_id, "restarted", mode="last_checkpoint",
                                         ckpt=str(ckpt.latest_checkpoint(self.ckpt_dir)))
                self._save_sidecar(sc)
                resume = True
            except torch.cuda.OutOfMemoryError as e:
                self.registry.append(c.run_id, "oom", error=str(e)[:500], ckpt_step=ckpt.latest_step(self.ckpt_dir),
                                     peak_mem=(torch.cuda.max_memory_allocated() if device.type == "cuda" else None))
                raise

    # -- one attempt (fresh model/optimiser, resumed from the newest state on disk) --------------
    def _attempt(self, device: torch.device, env: dict[str, Any], resume: bool, attempt: int) -> dict[str, Any]:
        c = self.cfg
        import gc

        gc.collect()  # drop a previous (diverged) attempt's model/optimiser before allocating again
        if device.type == "cuda":
            torch.cuda.empty_cache()
        sc = self._sidecar()
        peak_lr = sc["lr_override"] if sc.get("lr_override") is not None else c.schedule.peak_lr
        sched = c.schedule if peak_lr == c.schedule.peak_lr else _replace(c.schedule, peak_lr=peak_lr)

        model = self.model_factory(c.model_config(), c.seed_index)
        model.to(device=device, dtype=torch.float32)
        model.train()
        ac_mode = enable_activation_checkpointing(model) if c.runtime.activation_checkpointing else None
        opt, opt_summary = build_optimizer(model, c.optim, lr=0.0)
        params = [p for g in opt.param_groups for p in g["params"]]
        meta = self._meta(env)

        # ---- state: resume > parent (branch point) > fresh ----
        ckpt.cleanup_tmp(self.ckpt_dir)
        start_step = sched.start_step
        trainer_state: dict[str, Any] = {}
        resumed_from = None
        last = ckpt.latest_checkpoint(self.ckpt_dir) if resume else None
        if last is not None:
            st = ckpt.load(last)
            if st["config_hash"] != c.config_hash():
                raise RefusedError(f"checkpoint {last} has config hash {st['config_hash'][:8]}, run has "
                                   f"{c.config_hash()[:8]}")
            model.load_state_dict(st["model"])
            opt.load_state_dict(st["optimizer"])
            ckpt.set_rng_state(st["rng"])
            start_step = int(st["step"])
            trainer_state = dict(st.get("trainer_state") or {})
            resumed_from = str(last)
            if last.name == "final.pt" and start_step >= sched.steps:
                self.registry.append(c.run_id, "completed", step=start_step, note="already complete",
                                     final_ckpt=str(last))
                return {"status": "completed", "step": start_step, "already_complete": True}
        elif c.parent_ckpt:
            p = Path(c.parent_ckpt)
            if not p.is_file():
                raise RefusedError(f"parent checkpoint {p} missing (parent run {c.parent_run_id} not complete?)")
            st = ckpt.load(p)
            if st.get("model_config_hash") not in (None, c.model_config_hash()):
                raise RefusedError(f"parent {p} was trained with a different model config")
            _check_parent_lr(st, peak_lr, p)
            model.load_state_dict(st["model"])
            opt.load_state_dict(st["optimizer"])  # AdamW moments and step counts carried (§6.3, §7.3)
            resumed_from = f"parent:{p}"
            trainer_state["parent"] = {"run_id": st.get("run_id"), "step": st.get("step"),
                                       "config_hash": st.get("config_hash")}
        # LR is (re)set from the schedule every step; group hyper-parameters other than lr must
        # match the protocol even when a parent optimiser state was loaded.
        for g in opt.param_groups:
            g["weight_decay"] = c.optim.weight_decay if g.get("group_name") == "decay" else 0.0
            g["betas"], g["eps"] = tuple(c.optim.betas), c.optim.eps

        gdiv = GDivMonitor(c.gdiv, trainer_state.get("gdiv"))
        stream = build_stream(c)
        val_stream = build_val_stream(c)
        attn = AttnStats(model, c.logging.attn_stats)

        metrics = JsonlWriter(self.run_dir / "metrics.jsonl")
        trace = JsonlWriter(self.run_dir / "trace.jsonl") if c.logging.trace_every_step else None
        self.registry.append(c.run_id, "running", pid=os.getpid(), attempt=attempt, start_step=start_step,
                             resumed_from=resumed_from, host=os.uname().nodename, allow_dirty=self.allow_dirty,
                             git_commit=meta["git_commit"], config_hash=meta["config_hash"],
                             activation_checkpointing=ac_mode, peak_lr=peak_lr,
                             # run identity for the analysis registry reader (hybridlc.analysis.results)
                             config_name=c.config_name, scale=c.scale, seed_index=c.seed_index,
                             k=c.model_meta.get("k"), stage=c.stage, exp_stage=c.exp_stage, branch=c.branch,
                             parent_run_id=c.parent_run_id,
                             # §6.2 tokenizer hash + data MANIFEST hash per run (verification round 1, DTS-08)
                             tokenizer_sha256=env.get("tokenizer_sha256"),
                             data_manifest_sha256=env.get("data_manifest_sha256"))
        metrics.write({"kind": "start", "step": start_step, "attempt": attempt, "resumed_from": resumed_from,
                       "ts": time.time(), "optimizer": opt_summary, "activation_checkpointing": ac_mode,
                       "n_params": sum(p.numel() for p in model.parameters()), "peak_lr": peak_lr})

        tokens_seen = int(trainer_state.get("tokens_seen", 0))
        loss_tokens_seen = int(trainer_state.get("loss_tokens_seen", 0))
        win = _Window()
        run_peak_mem = int(trainer_state.get("run_peak_mem", 0))
        below_since: float | None = trainer_state.get("below_since")
        alerted = bool(trainer_state.get("throughput_alerted", False))
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        def state_dict() -> dict[str, Any]:
            return {"gdiv": gdiv.state_dict(), "tokens_seen": tokens_seen, "loss_tokens_seen": loss_tokens_seen,
                    "run_peak_mem": run_peak_mem, "attempt": attempt, "peak_lr": peak_lr,
                    "below_since": below_since, "throughput_alerted": alerted,
                    **({"parent": trainer_state["parent"]} if "parent" in trainer_state else {})}

        if c.validation.at_start and start_step == sched.start_step and val_stream is not None:
            self._validate(model, val_stream, device, start_step, metrics, c.validation.n_batches)

        step = start_step
        last_loss: float | None = None
        try:
            for step in range(start_step, sched.steps):
                t0 = time.perf_counter()
                lr = lr_at(step, sched)
                set_lr(opt, lr)
                self.faults.maybe_oom(step)
                batches = [stream.get(step, m) for m in range(c.data.accum)]
                W, N = _step_weights(batches, c.optim.ignore_index)
                W_obj = _objective_norm(batches, W)
                log_step = (step + 1) % c.logging.every == 0 or step == sched.steps - 1
                ce_acc = torch.zeros((), dtype=torch.float32, device=device)
                z_acc = torch.zeros((), dtype=torch.float32, device=device)
                n_seg = 0
                for m, b in enumerate(batches):
                    b = _to_device(b, device)
                    if log_step and m == 0:
                        attn.arm(True)
                    with _autocast(c, device):
                        hidden = model(b["input_ids"], b["cu_seqlens"], b["position_ids"])
                        kw = {"loss_weights": b["loss_weights"]} if "loss_weights" in b else {}
                        loss, z, n = self.loss_fn(hidden, model.lm_head_weight, b["labels"], chunk=c.optim.ce_chunk,
                                                  z_loss_coef=c.optim.z_loss_coef,
                                                  ignore_index=c.optim.ignore_index, **kw)
                    if log_step and m == 0:
                        attn.arm(False)
                    a_m, b_m = b["_w"] / W_obj, b["_n"] / N
                    obj = loss.float() * a_m + z.float() * b_m
                    obj.backward()
                    # logged / G-div loss: the step's weighted mean CE (sum w*CE / sum w), whatever the
                    # objective's normaliser (verification round 3: TEF block-normalised weights)
                    ce_acc += loss.detach().float() * (b["_w"] / W)
                    z_acc += z.detach().float() * b_m
                    n_seg += int(b["cu_seqlens"].numel()) - 1
                    del hidden, loss, z, obj
                    self.faults.after_micro(step, m)
                gnorm = torch.nn.utils.clip_grad_norm_(params, c.optim.grad_clip, foreach=True)
                # one host sync per step (needed by G-div anyway)
                ce_val, z_val, gn_val = (float(v) for v in torch.stack([ce_acc, z_acc, gnorm.float()]).tolist())
                inj = self.faults.loss_override(step)
                if inj is not None:
                    ce_val = inj
                if not math.isfinite(gn_val) and math.isfinite(ce_val):
                    ce_val = float("nan")  # non-finite gradients count as a NaN event (G-div)
                gdiv.update(step, ce_val)  # raises DivergenceError before the bad update is applied
                opt.step()
                opt.zero_grad(set_to_none=True)

                n_in = sum(int(b["input_ids"].numel()) for b in batches)
                tokens_seen += n_in
                loss_tokens_seen += int(N)
                step_hash = _step_hash(batches)
                if trace is not None:
                    trace.write({"step": step, "loss": ce_val, "loss_hex": float_hex(ce_val), "z": z_val,
                                 "gnorm_hex": float_hex(gn_val), "lr": lr, "batch_hash": step_hash})
                dt = time.perf_counter() - t0
                win.add(ce_val, z_val, gn_val, n_in, dt)
                last_loss = ce_val

                done = step + 1
                if log_step:
                    _maybe_cuda_sync(device)
                    mem = {}
                    if device.type == "cuda":
                        peak = torch.cuda.max_memory_allocated(device)
                        run_peak_mem = max(run_peak_mem, peak)
                        mem = {"mem_peak_alloc": peak, "mem_reserved": torch.cuda.memory_reserved(device),
                               "mem_run_peak": run_peak_mem}
                        torch.cuda.reset_peak_memory_stats(device)
                    rec = {"kind": "train", "step": done - 1, "tokens": tokens_seen, "loss": ce_val,
                           "loss_win": win.mean_loss(), "z_loss": z_val, "lr": lr, "grad_norm": gn_val,
                           "tok_s": win.tok_s(), "n_segments": n_seg, "batch_hash": step_hash,
                           "gdiv_ema": gdiv.s.ema, "gdiv_above": gdiv.s.above, "ts": time.time(), **mem}
                    st_attn = attn.read()
                    rec.update(st_attn or {"attn_max_logit": None, "attn_entropy": None})
                    metrics.write(rec)
                    below_since, alerted = self._throughput_guard(win.tok_s(), below_since, alerted, done)
                    win = _Window()
                if c.validation.every > 0 and val_stream is not None and done % c.validation.every == 0 \
                        and done < sched.steps:
                    self._validate(model, val_stream, device, done, metrics, c.validation.n_batches)
                if done in c.checkpoint.milestone_steps and done < sched.steps:
                    ckpt.save_milestone(self.ckpt_dir, done, model, _milestone_meta(meta))
                if c.checkpoint.every > 0 and done % c.checkpoint.every == 0 and done < sched.steps:
                    meta["triton_cache_hash"] = triton_cache_hash()  # the JIT may have added kernels
                    path = ckpt.save_full(self.ckpt_dir, done, model=model, optimizer=opt, meta=meta,
                                          trainer_state=state_dict(), keep_last=c.checkpoint.keep_last,
                                          fault_hook=self.faults.ckpt_hook(done))
                    metrics.write({"kind": "checkpoint", "step": done - 1, "path": str(path), "ts": time.time()})
        finally:
            metrics.f.flush()

        # ---- end of run ----
        done = sched.steps
        final_val = None
        if val_stream is not None and (c.validation.every > 0 or c.validation.final_n_batches > 0):
            nb = c.validation.final_n_batches or c.validation.n_batches
            final_val = self._validate(model, val_stream, device, done, metrics, nb, final=True)
        final_path = None
        meta["triton_cache_hash"] = triton_cache_hash()
        if c.checkpoint.save_final:
            final_path = ckpt.save_full(self.ckpt_dir, done, model=model, optimizer=opt, meta=meta,
                                        trainer_state=state_dict(), keep_last=c.checkpoint.keep_last, final=True,
                                        fault_hook=self.faults.ckpt_hook(done))
        ckpt.save_milestone(self.ckpt_dir, done, model, _milestone_meta(meta), tag="final")
        psha = ckpt.param_sha256(model)
        summary = {"steps_done": done, "final_loss": last_loss,
                   "final_val": final_val, "param_sha256": psha, "tokens_seen": tokens_seen,
                   "run_peak_mem": run_peak_mem, "final_ckpt": str(final_path) if final_path else None}
        metrics.write({"kind": "end", "step": done - 1, "ts": time.time(), **summary})
        summary = {"status": "completed", **summary}
        metrics.close()
        if trace is not None:
            trace.close()
        (self.run_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
        self.registry.append(c.run_id, "completed", **{k: v for k, v in summary.items() if k != "status"})
        return summary

    # -- validation -----------------------------------------------------------------------------
    @torch.no_grad()
    def _validate(self, model: nn.Module, stream: Any, device: torch.device, step: int,
                  metrics: JsonlWriter, n_batches: int, final: bool = False) -> dict[str, Any]:
        """Token-weighted mean CE over the first ``n_batches`` micro-batches of the LM-2 subset.

        The model stays in train mode (no dropout exists) so the same chunked kernels are used as in
        training (fla switches GDN to fused_recurrent for short inputs in eval mode, pitfall P14).
        """
        c = self.cfg
        t0 = time.perf_counter()
        n_max = getattr(stream, "num_steps", lambda: None)()
        nb = n_batches if n_max is None else min(n_batches, n_max)
        ce_sum = torch.zeros((), dtype=torch.float64, device=device)
        n_tok = 0
        for i in range(nb):
            b = _to_device(stream.get(i, 0), device)
            with _autocast(c, device):
                h = model(b["input_ids"], b["cu_seqlens"], b["position_ids"])
                loss, _z, n = self.loss_fn(h, model.lm_head_weight, b["labels"], chunk=c.optim.ce_chunk,
                                           z_loss_coef=0.0, ignore_index=c.optim.ignore_index)
            ni = int(n)
            ce_sum += loss.double() * ni
            n_tok += ni
        val = float(ce_sum.item() / max(n_tok, 1))
        rec = {"kind": "val", "step": step - 1 if step > 0 else 0, "after_steps": step, "val_loss": val,
               "val_tokens": n_tok, "val_batches": nb, "final": final, "secs": time.perf_counter() - t0,
               "ts": time.time()}
        metrics.write(rec)
        return rec

    # -- G-throughput ---------------------------------------------------------------------------
    def _throughput_guard(self, tok_s: float, below_since: float | None, alerted: bool,
                          step: int) -> tuple[float | None, bool]:
        r = self.cfg.runtime
        if r.g0_tok_s is None or tok_s <= 0:
            return below_since, alerted
        if tok_s < (1.0 - r.throughput_tolerance) * r.g0_tok_s:
            below_since = below_since or time.time()
            if not alerted and time.time() - below_since > r.throughput_alert_after_s:
                self.registry.append(self.cfg.run_id, "throughput_alert", step=step, tok_s=tok_s,
                                     g0_tok_s=r.g0_tok_s, below_for_s=time.time() - below_since)
                alerted = True
        else:
            below_since = None
        return below_since, alerted


# ----------------------------------------------------------------------------------------------
# pieces used by the loop
# ----------------------------------------------------------------------------------------------
class _Window:
    def __init__(self) -> None:
        self.n = 0
        self.loss_sum = 0.0
        self.tokens = 0
        self.secs = 0.0
        self.last_loss: float | None = None

    def add(self, loss: float, z: float, gn: float, tokens: int, secs: float) -> None:
        self.n += 1
        self.loss_sum += loss
        self.tokens += tokens
        self.secs += secs
        self.last_loss = loss

    def mean_loss(self) -> float | None:
        return self.loss_sum / self.n if self.n else None

    def tok_s(self) -> float:
        return self.tokens / self.secs if self.secs > 0 else 0.0


def _check_parent_lr(parent_state: dict[str, Any], child_peak_lr: float, path: Path) -> None:
    """Refuse a branch whose peak LR silently ignores an LR restart of its parent (DTS-06).

    After a D034 restart a parent (e.g. a FEAS Stage A) trained at ``lr_override`` (1.4e-3) while
    its config still says the original ``lr`` (2.0e-3). A child that takes its peak LR from the run
    YAML (``peak_lr: run``) would then decay from the stale value, a 43% LR jump at the branch point
    with an unchanged run_id. Detected as: the parent's actual peak (trainer_state) differs from its
    configured peak, and the child's peak equals that stale configured peak. Stages with their own
    fixed peak (TEF 3e-4) differ from the parent's configured peak and pass. The fix is explicit:
    relaunch the child with ``-o schedule.peak_lr=<parent's actual peak>`` (or set the run YAML's
    ``lr``); either gives a new config hash and run_id. (``-o lr=`` is rejected: verification round 2,
    TRAIN-DTS06-REMEDY-NOOP.)
    """
    actual = (parent_state.get("trainer_state") or {}).get("peak_lr")
    configured = ((parent_state.get("config") or {}).get("schedule") or {}).get("peak_lr")
    if actual is None or configured is None:
        return
    if not math.isclose(float(actual), float(configured), rel_tol=1e-9) and \
            math.isclose(float(child_peak_lr), float(configured), rel_tol=1e-9):
        raise RefusedError(
            f"parent {path} trained at peak LR {actual:g} (LR restart) but its config says {configured:g}, "
            f"and this run's peak LR is the stale {child_peak_lr:g}; relaunch with -o schedule.peak_lr={actual:g} "
            f"(or set lr: {actual:g} in the run YAML)")


def _step_weights(batches: list[dict[str, Any]], ignore_index: int) -> tuple[float, int]:
    """Per-micro-batch loss weight mass W_m and loss-token count n_m (stored in the batch dicts)."""
    W, N = 0.0, 0
    for b in batches:
        valid = (b["labels"] != ignore_index) & (b["labels"] >= 0)  # same rule as chunked_ce
        n = int(valid.sum())
        w = float(b["loss_weights"].double()[valid].sum()) if "loss_weights" in b else float(n)
        b["_n"], b["_w"] = n, w
        W += w
        N += n
    if N == 0 or W <= 0:
        raise ValueError("a training step has no loss tokens")
    return W, N


def _objective_norm(batches: list[dict[str, Any]], W: float) -> float:
    """Divisor of the step objective sum(w * CE): the step's weight mass ``W`` (default: exact step
    mean), or the stream's constant ``loss_norm`` when every micro-batch carries one (TEF with
    block-normalised D014 weights, verification round 3: dividing those by the step's own W would
    turn them back into per-step normalised weights)."""
    have = ["loss_norm" in b for b in batches]
    if not any(have):
        return W
    norms = {float(b["loss_norm"]) for b in batches if "loss_norm" in b}
    if not all(have) or len(norms) != 1:
        raise ValueError(f"inconsistent loss_norm across the micro-batches of a step: {sorted(norms)}, "
                         f"present in {sum(have)} of {len(have)}")
    (v,) = norms
    if not v > 0:
        raise ValueError(f"loss_norm must be > 0, got {v}")
    return v


def _step_hash(batches: list[dict[str, Any]]) -> str:
    import hashlib

    h = hashlib.sha256()
    for b in batches:
        h.update(str(b.get("batch_hash", "")).encode())
    return h.hexdigest()


def _milestone_meta(meta: dict[str, Any]) -> dict[str, Any]:
    return {k: meta[k] for k in ("run_id", "config_hash", "model_config_hash", "git_commit", "triton_cache_hash")}


def _replace(obj: Any, **kw: Any) -> Any:
    import dataclasses

    return dataclasses.replace(obj, **kw)


# ----------------------------------------------------------------------------------------------
# public API
# ----------------------------------------------------------------------------------------------
def train(run_cfg: RunConfig, resume: bool = True, allow_dirty: bool = False) -> dict[str, Any]:
    """Run ``run_cfg`` to completion (INTERFACES.md: ``training/trainer.py: train(run_cfg, resume=True)``).

    Raises :class:`RefusedError`, :class:`DivergenceError` (after the one automatic restart) or
    ``torch.cuda.OutOfMemoryError``; ``scripts/train.py`` maps these to exit codes for the runner.
    """
    return Trainer(run_cfg, allow_dirty=allow_dirty).run(resume=resume)


def main_exit_code(fn: Callable[[], Any]) -> int:
    """Run ``fn`` and map the trainer's exceptions to the runner's exit codes."""
    try:
        fn()
        return EXIT_OK
    except RefusedError as e:
        print(f"[train] REFUSED: {e}", file=sys.stderr, flush=True)
        return EXIT_REFUSED
    except DivergenceError as e:
        print(f"[train] FAILED (G-div twice): {e}", file=sys.stderr, flush=True)
        return EXIT_DIVERGED
    except torch.cuda.OutOfMemoryError as e:
        print(f"[train] OOM: {e}", file=sys.stderr, flush=True)
        return EXIT_OOM


__all__ = ["train", "Trainer", "RefusedError", "EXIT_OK", "EXIT_OOM", "EXIT_DIVERGED", "EXIT_REFUSED",
           "enable_activation_checkpointing", "read_jsonl_last_per_step", "float_hex", "FAULT_ENV",
           "main_exit_code"]

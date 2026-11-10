"""Make the pinned Zoology checkout importable and runnable in this venv (protocol §17).

Three things are needed, and nothing else in Zoology's code path is changed:

1. **Import path.** Zoology lives at ``third_party/zoology`` (pinned commit
   ``1ad20d193b6113cae1e8f3c655c300d7b4b3f4bb``) and is installed editable in the venv
   (``uv pip install -e third_party/zoology --no-deps``, run by the orchestrator). We still
   prepend the checkout to ``sys.path`` (a no-op given the editable install) and verify the
   commit, so a stray site-packages copy can never be picked up.

2. **fla 0.6.0 API shim.** Zoology's ``mixers/gated_delta_net.py`` calls
   ``chunk_gated_delta_rule(..., head_first=False, ...)``. fla 0.6.0 removed the
   ``head_first`` argument and *raises* if it is passed at all; ``head_first=False`` was
   the ``[B, T, H, D]`` layout that fla 0.6.0 now assumes unconditionally. The shim drops
   that keyword when it equals ``False`` (and refuses ``True``), so the computation is
   identical. DEVIATION (API compatibility only; no numerical change).

4. **Gradient accumulation (memory).** DEVIATION, see :func:`_patch_microbatch`.
   Zoology computes fp32 logits for the whole 256-row batch; at T=256 that is a 2 GiB tensor
   ``[256, 256, 8192]`` plus same-size log-softmax and gradient buffers, which exceeds the
   0.44 GPU share (and the physically free memory next to CARLA). The 1-epoch smoke run
   OOMed at the first T=256 batch. We split each optimiser batch into micro-batches of at
   most ``MICRO_BATCH`` rows and weight each micro-batch's mean CE by its share of labelled
   positions, so the accumulated gradient equals the full-batch gradient of Zoology's loss
   (``CrossEntropyLoss`` mean over non-ignored targets) up to float summation order. One
   optimiser step per 256-row batch, as in Zoology. Only the dropout RNG stream differs.

5. **Per-epoch checkpoint/resume (robustness).** The GPU drives a display and is shared with
   CARLA; the driver watchdog occasionally kills a kernel launch (``cudaErrorLaunchTimeout``),
   which aborted two attempts of one 17-min run. :func:`_patch_fit_resumable` replaces
   ``Trainer.fit`` by a line-for-line copy that additionally saves model / optimizer /
   scheduler / all RNG states / logger records after every completed epoch and resumes from
   there. Zoology's data order is fixed (``shuffle=False``), so a resumed run follows the same
   trajectory as an uninterrupted one, up to non-deterministic GPU kernels.

3. **wandb off.** The protocol says "wandb disabled". Zoology's ``WandbLogger`` already
   becomes a no-op when ``project_name``/``entity`` are None; we additionally replace it
   with :class:`CaptureLogger`, which records exactly the metrics Zoology would have sent to
   wandb, so results can be written to ``experiments/s1_zoology/``.
"""
from __future__ import annotations

import functools
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# S1_REPO_ROOT lets a staged copy of these scripts run against the real repo (testing only).
REPO_ROOT = Path(os.environ.get("S1_REPO_ROOT", Path(__file__).resolve().parents[2]))
ZOOLOGY_DIR = REPO_ROOT / "third_party" / "zoology"
ZOOLOGY_COMMIT = "1ad20d193b6113cae1e8f3c655c300d7b4b3f4bb"  # protocol §17
DATA_CACHE_DIR = REPO_ROOT / "data" / "zoology"  # protocol §17: cache_dir data/zoology/
RESULTS_DIR = REPO_ROOT / "experiments" / "s1_zoology"
GPU_MEMORY_FRACTION = 0.44  # INTERFACES.md: every GPU process claims 0.44 of the card
# Engineering constant (not an experimental setting): rows per micro-batch in the
# gradient-accumulation shim. 64 rows at T=256 keeps the logits/CE peak near 2 GiB.
MICRO_BATCH = 64

# Zoology's Trainer wraps every step in tqdm and calls set_postfix (which forces a refresh)
# every step; disable the bars to keep nohup logs small (per-epoch metrics are printed by
# run_one.py). tqdm reads TQDM_* env vars when it is imported, so this must run at import
# time of this module, before torch/zoology. Logging only, no effect on the computation.
os.environ.setdefault("TQDM_DISABLE", "1")
# Allocator setting only (reduces fragmentation next to CARLA); no numerical effect.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

_PATCHED = False


def zoology_commit() -> str:
    """Return the HEAD commit of the Zoology checkout (to be recorded with every result)."""
    out = subprocess.run(
        ["git", "-C", str(ZOOLOGY_DIR), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    return out.stdout.strip()


def zoology_dirty() -> bool:
    out = subprocess.run(
        ["git", "-C", str(ZOOLOGY_DIR), "status", "--porcelain"],
        capture_output=True, text=True, check=True,
    )
    return bool(out.stdout.strip())


def claim_gpu(fraction: float = GPU_MEMORY_FRACTION) -> None:
    """Cap this process's share of the shared GPU (CARLA runs alongside)."""
    try:  # prefer the project helper once it exists (INTERFACES.md utils/gpu.py)
        from hybridlc.utils.gpu import claim  # type: ignore

        claim(fraction)
        return
    except Exception:
        pass
    import torch

    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(fraction)


class CaptureLogger:
    """Drop-in replacement for ``zoology.logger.WandbLogger`` that keeps metrics in memory.

    Zoology calls ``log({"train/loss": ..., "epoch": e})`` every step and
    ``log({"epoch": e, "valid/loss": ..., "valid/accuracy": ..., "valid/num_kv_pairs/accuracy-N": ...})``
    once per epoch. We keep every validation record and a per-epoch train-loss mean.
    The *last* validation record is what wandb's run summary would show, which is what
    Zoology's own analysis scripts read (``zoology/analysis/__init__.py``).
    """

    instances: list["CaptureLogger"] = []

    def __init__(self, config: Any) -> None:
        self.config = config
        self.valid: list[dict[str, float]] = []
        self.train_loss_sum: dict[int, float] = {}
        self.train_loss_n: dict[int, int] = {}
        self.extra: dict[str, Any] = {}
        # Timing across resumed attempts: elapsed_offset_s is the compute time of earlier
        # attempts up to their last checkpoint; epoch_end_s are cumulative seconds.
        self.t0 = time.time()
        self.elapsed_offset_s = 0.0
        self.epoch_end_s: list[float] = []
        self.resumed_at_epochs: list[int] = []
        CaptureLogger.instances.append(self)

    def elapsed_s(self) -> float:
        return self.elapsed_offset_s + (time.time() - self.t0)

    def state_dict(self) -> dict[str, Any]:
        return {"valid": self.valid, "train_loss_sum": self.train_loss_sum, "train_loss_n": self.train_loss_n,
                "extra": self.extra, "epoch_end_s": self.epoch_end_s, "elapsed_s": self.elapsed_s(),
                "resumed_at_epochs": self.resumed_at_epochs}

    def load_state_dict(self, st: dict[str, Any], next_epoch: int) -> None:
        self.valid = list(st["valid"])
        self.train_loss_sum = dict(st["train_loss_sum"])
        self.train_loss_n = dict(st["train_loss_n"])
        self.extra = {**st["extra"], **self.extra}
        self.epoch_end_s = list(st["epoch_end_s"])
        self.elapsed_offset_s = float(st["elapsed_s"])
        self.t0 = time.time()
        self.resumed_at_epochs = list(st["resumed_at_epochs"]) + [int(next_epoch)]

    def log_config(self, config: Any) -> None:  # noqa: D401 - zoology API
        pass

    model_ref: Any = None  # last model seen by log_model (R4 post-training diagnostics; never serialised)

    def log_model(self, model: Any, config: Any) -> None:
        CaptureLogger.model_ref = model
        max_seq_len = max(c.input_seq_len for c in config.data.test_configs)
        self.extra["num_parameters"] = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
        try:
            self.extra["state_size"] = int(model.state_size(sequence_length=max_seq_len))
        except Exception as exc:  # state_size is informational only
            self.extra["state_size_error"] = repr(exc)

    def log(self, metrics: dict[str, Any]) -> None:
        if "train/loss" in metrics:
            e = int(metrics["epoch"])
            self.train_loss_sum[e] = self.train_loss_sum.get(e, 0.0) + float(metrics["train/loss"])
            self.train_loss_n[e] = self.train_loss_n.get(e, 0) + 1
        if "valid/accuracy" in metrics:
            self.valid.append({k: float(v) for k, v in metrics.items()})
            self.epoch_end_s.append(self.elapsed_s())
            if self.on_valid is not None:
                self.on_valid(self)

    on_valid = None  # optional callback(logger) after each validation (used for timing / checkpoints)

    def finish(self) -> None:
        pass

    def train_loss_by_epoch(self) -> dict[int, float]:
        return {e: self.train_loss_sum[e] / self.train_loss_n[e] for e in sorted(self.train_loss_sum)}


def _patch_fla_head_first(zgdn_module: Any) -> None:
    """Wrap fla kernels referenced by zoology.mixers.gated_delta_net to drop head_first=False."""

    def wrap(fn):
        @functools.wraps(fn)
        def inner(*args, **kwargs):
            hf = kwargs.pop("head_first", False)
            if hf:
                raise ValueError("head_first=True is not supported by fla 0.6.0; shim refuses it")
            return fn(*args, **kwargs)

        inner.__zoology_compat__ = True  # type: ignore[attr-defined]
        return inner

    for name in ("chunk_gated_delta_rule", "fused_recurrent_gated_delta_rule"):
        fn = getattr(zgdn_module, name)
        if not getattr(fn, "__zoology_compat__", False):
            setattr(zgdn_module, name, wrap(fn))


def _patch_microbatch(ztrain: Any, micro_batch: int) -> None:
    """Replace ``Trainer.train_epoch`` by an exact gradient-accumulation version.

    Mirrors Zoology's ``train_epoch`` (zoology/train.py L118-145 at the pinned commit) step for
    step: ``zero_grad`` -> loss -> auxiliary losses -> ``backward`` -> ``optimizer.step`` ->
    log ``train/loss``. The only change is that the forward/backward runs over row chunks.

    Exactness: Zoology's loss is ``CrossEntropyLoss()`` (mean over targets != -100) on the
    whole batch. With ``n_c`` labelled targets in chunk c and ``N = sum n_c``,
    ``L = sum_c (n_c / N) * mean_c``, so back-propagating ``(n_c / N) * mean_c`` per chunk
    accumulates exactly ``dL/dθ``. Auxiliary losses (none for attention/GDN) are model-level
    and are added once, after the last chunk's forward, like Zoology does.
    """
    import torch

    Trainer = ztrain.Trainer
    if getattr(Trainer.train_epoch, "__zoology_compat__", False):
        return

    def train_epoch(self, epoch_idx: int) -> None:  # noqa: ANN001 - zoology API
        assert self.input_type == "discrete" and self.loss_type == "ce", "shim covers the MQAR (ce) path only"
        self.model.train()
        for inputs, targets, slices in self.train_dataloader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            n_total = int((targets != -100).sum())
            loss_total = torch.zeros((), device=self.device)
            n_chunks = (inputs.shape[0] + micro_batch - 1) // micro_batch
            for c, (x, y) in enumerate(zip(inputs.split(micro_batch), targets.split(micro_batch))):
                loss_c, _ = self.compute_loss(x, y)
                loss_c = loss_c * (int((y != -100).sum()) / n_total)
                if c == n_chunks - 1:
                    aux: list = []
                    self.model.apply(lambda m: aux.append(m.get_auxiliary_loss())
                                     if hasattr(m, "get_auxiliary_loss") else None)
                    if aux:
                        loss_c = loss_c + sum(aux)
                loss_c.backward()
                loss_total = loss_total + loss_c.detach()
            self.optimizer.step()
            self.logger.log({"train/loss": loss_total.item(), "epoch": epoch_idx})

    train_epoch.__zoology_compat__ = True  # type: ignore[attr-defined]
    Trainer.train_epoch = train_epoch


def _atomic_torch_save(obj: Any, path: Path) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        torch.save(obj, fh)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def collapse_abort_due(valid_accs: list[float], threshold: float = 0.05, run_len: int = 3,
                       first_epoch: int = 3) -> bool:
    """Collapse-abort rule (plan §2.0 / amendment §17.6; reading C1 of experiments/repro/R4/prereg.yaml).

    ``valid_accs[e]`` is the overall validation accuracy after 0-based epoch ``e``. True iff the last
    ``run_len`` epochs all have index >= ``first_epoch`` ("after epoch 2") and accuracy < ``threshold``.
    """
    e = len(valid_accs) - 1
    if e - (run_len - 1) < first_epoch:
        return False
    return all(a < threshold for a in valid_accs[-run_len:])


def _patch_fit_resumable(ztrain: Any, ckpt_path: Path, collapse_abort: bool = False) -> None:
    """Replace ``Trainer.fit`` by Zoology's ``fit`` (train.py L178-202) + per-epoch checkpoints.

    The optimisation code is unchanged: AdamW(lr, weight_decay), CosineAnnealingLR(T_max =
    max_epochs, eta_min = 0) stepped once per epoch after validation, early stopping on
    ``early_stopping_metric > early_stopping_threshold`` checked before the scheduler step.
    After each non-final epoch the full state is written atomically to ``ckpt_path``; if that
    file exists when ``fit`` starts, training resumes at the next epoch.
    """
    import random

    import numpy as np
    import torch
    from torch import nn, optim

    def fit(self) -> None:  # noqa: ANN001 - zoology API
        self.model.to(self.device)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate,
                                     weight_decay=self.weight_decay)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.max_epochs, eta_min=0.0)
        start = 0
        if ckpt_path.exists():
            ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            self.model.load_state_dict(ck["model"])
            self.optimizer.load_state_dict(ck["optimizer"])
            self.scheduler.load_state_dict(ck["scheduler"])
            start = int(ck["next_epoch"])
            self.logger.load_state_dict(ck["logger"], next_epoch=start)
            random.setstate(ck["rng"]["python"])
            np.random.set_state(ck["rng"]["numpy"])
            torch.set_rng_state(ck["rng"]["torch"])
            torch.cuda.set_rng_state_all(ck["rng"]["cuda"])
            print(f"[s1] resumed from {ckpt_path} at epoch {start}", flush=True)
        for epoch_idx in range(start, self.max_epochs):
            self.train_epoch(epoch_idx)
            metrics = self.test(epoch_idx)
            if collapse_abort and collapse_abort_due([r["valid/accuracy"] for r in self.logger.valid]):
                self.logger.extra["collapsed"] = True
                self.logger.extra["collapsed_at_epoch"] = int(epoch_idx)
                print(f"[s1] collapse-abort at epoch {epoch_idx}: valid acc < 0.05 for 3 epochs after epoch 2",
                      flush=True)
                break
            if (self.early_stopping_metric is not None) and metrics[
                self.early_stopping_metric
            ] > self.early_stopping_threshold:
                print(
                    f"Early stopping triggered at epoch {epoch_idx} with "
                    f"{self.early_stopping_metric} {metrics[self.early_stopping_metric]} > {self.early_stopping_threshold}"
                )
                break
            self.scheduler.step()
            if epoch_idx + 1 < self.max_epochs:
                _atomic_torch_save({
                    "next_epoch": epoch_idx + 1,
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "scheduler": self.scheduler.state_dict(),
                    "logger": self.logger.state_dict(),
                    "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()},
                }, ckpt_path)

    fit.__zoology_compat__ = True  # type: ignore[attr-defined]
    ztrain.Trainer.fit = fit


def setup_zoology(check_commit: bool = True, micro_batch: int | None = MICRO_BATCH,
                  ckpt_path: Path | None = None, collapse_abort: bool = False) -> None:
    """Put Zoology on sys.path, apply the fla shim and the capture logger. Idempotent."""
    global _PATCHED
    if _PATCHED:
        return
    if check_commit:
        head = zoology_commit()
        if head != ZOOLOGY_COMMIT:
            raise RuntimeError(f"third_party/zoology is at {head}, protocol pins {ZOOLOGY_COMMIT}")
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("WANDB_DISABLED", "true")
    if str(ZOOLOGY_DIR) not in sys.path:
        sys.path.insert(0, str(ZOOLOGY_DIR))

    import zoology.mixers.gated_delta_net as zgdn
    import zoology.train as ztrain

    _patch_fla_head_first(zgdn)
    if micro_batch is not None:
        _patch_microbatch(ztrain, micro_batch)
    if ckpt_path is not None:
        _patch_fit_resumable(ztrain, ckpt_path, collapse_abort=collapse_abort)
    ztrain.WandbLogger = CaptureLogger  # train() instantiates WandbLogger(config)
    _PATCHED = True

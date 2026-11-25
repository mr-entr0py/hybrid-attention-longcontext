"""EXPLORATORY X2 (not a gate, not confirmatory): MQAR placement + longer-context sweep. Same model family as X1, but the
run id gives the full mixer pattern (e.g. mix2-LLA-s123) and two harder test configs are added: 2048/512 and 4096/1024.
Original X1 description: A 6-layer Zoology model
[BaseConv, M1, BaseConv, M2, BaseConv, M3] at d_model 64, where each M is our Gated DeltaNet (bf16 autocast,
as in R5) or our attention (Zoology-matched MHA, as in R4). k = number of attention mixers among M1..M3.
Everything else is Zoology's original d64 GDN TrainConfig (lr 1e-3, 32 epochs, seeds = data seeds).

    python scripts/s1_zoology/run_one_mix.py --run-id mix-k1-s123

Former R5 docstring follows.
Run one R5 config (MQAR d_model = 64, Zoology GDN vs ours under bf16 autocast) and write its result JSON.

Pre-registration: ``experiments/repro/R5/prereg.yaml`` (plan §2.5). Same harness as R4's ``run_one_r4.py``
(Zoology's own ``zoology.train.train``; shims: exact micro-batch accumulation, per-epoch resume; collapse-abort
rule active, reading C1), plus:

* the run is looked up in ``our_mixers.r5_runs(include_o5=True)`` (6 runs + the 4 O5 runs, which the queue only
  launches if the bimodality rule fires);
* the GDN chunk kernel entry points of BOTH arms (``zoology.mixers.gated_delta_net.chunk_gated_delta_rule`` and
  ``hybridlc.models.linear.chunk_gated_delta_rule``) are wrapped by call counters that record the dtypes of
  q / g / beta at the first call (evidence for reading P1; no numerical effect);
* final weights saved to ``experiments/repro/R5/weights/<run_id>.pt``.

    python scripts/s1_zoology/run_one_r5.py --run-id our_gdn_bf16-d64-lr1.0e-03-s123
"""
from __future__ import annotations

# --- R2-LIVE v2 L1 entry-point guard (PI ruling D077): take the project GPU lock, or verify the
# holder's inherited token, BEFORE any import that can create a CUDA context; otherwise exit 78.
import sys as _sys_guard  # noqa: E402
from pathlib import Path as _Path_guard  # noqa: E402
if __name__ == "__main__":
    _sys_guard.path.insert(0, str(_Path_guard(__file__).resolve().parents[2] / "src"))
    from hybridlc.utils.gpu_lock import guard as _gpu_guard  # noqa: E402
    _gpu_guard(_Path_guard(__file__).name)
# ---

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from zoology_compat import (  # noqa: E402
    DATA_CACHE_DIR, MICRO_BATCH, REPO_ROOT, CaptureLogger, claim_gpu, setup_zoology, zoology_commit, zoology_dirty,
)
from run_one import TEST_CONFIG_KEYS, per_config_acc  # noqa: E402
TEST_CONFIG_KEYS.update({512: "T2048_kv512", 1024: "T4096_kv1024"})

R5_DIR = REPO_ROOT / "experiments" / "explore" / "X2_mqar_placement_length"


def _git(*a: str) -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO_ROOT), *a], capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as e:  # pragma: no cover
        return f"error: {e}"


class KernelCounter:
    """Wraps a module-global GDN chunk-kernel pointer; records call count and first-call input dtypes."""

    def __init__(self, module, name: str = "chunk_gated_delta_rule") -> None:
        self.module, self.name, self.orig = module, name, getattr(module, name)
        self.calls, self.first = 0, None

        def wrapped(*a, **k):
            self.calls += 1
            if self.first is None:
                self.first = {key: str(k[key].dtype) for key in ("q", "k", "v", "g", "beta") if key in k
                              and hasattr(k[key], "dtype")}
            return self.orig(*a, **k)

        setattr(module, name, wrapped)

    def record(self) -> dict:
        return {"module": self.module.__name__, "calls": self.calls, "first_call_dtypes": self.first}


def env_info() -> dict:
    import torch
    import triton

    import fla

    return {"torch": torch.__version__, "triton": triton.__version__, "fla": fla.__version__,
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
            "driver": _git_free(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]),
            "python": platform.python_version(), "host": platform.node()}


def _git_free(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception as e:  # pragma: no cover
        return f"error: {e}"


PATTERNS = {0: "LLL", 1: "LAL", 2: "ALA", 3: "AAA"}


def build_mix_config(OM, run_id: str):
    pat = run_id.split("-")[1]
    assert len(pat) == 3 and set(pat) <= {"A", "L"}, pat
    seed = int(run_id.split("-s")[-1])
    cfg = OM._zoology_gdn_d64_base().model_copy(deep=True)
    seq = cfg.model.sequence_mixer.kwargs["configs"]
    conv = seq[0]
    lin = {"name": "our_mixers.OurGdnBf16Zoology", "kwargs": {}}
    att = {"name": "our_mixers.OurFgZoology", "kwargs": {}}
    mixers = [att if ch == "A" else lin for ch in pat]
    cfg.model.sequence_mixer.kwargs["configs"] = [conv, mixers[0], conv, mixers[1], conv, mixers[2]]
    cfg.model.n_layers = 6
    cfg.model.name = f"mix2-{pat}"
    from zoology.data.multiquery_ar import MQARConfig
    base = cfg.data.test_configs[-1]
    for T, kv in ((2048, 512), (4096, 1024)):
        cfg.data.test_configs.append(MQARConfig(vocab_size=base.vocab_size, num_examples=1000, input_seq_len=T,
                                                num_kv_pairs=kv))
    # Long test configs: dense fp32 attention scores at T=4096 need 4 GiB per test batch of 32 (OOM under the
    # 0.44 memory cap), so evaluate with test batch 8. Training batch is unchanged.
    cfg.data.batch_size = (cfg.data.batch_size[0], 8)
    cfg.seed = seed
    cfg.data.seed = seed
    cfg.run_id = run_id
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--out-dir", default=str(R5_DIR / "runs"))
    ap.add_argument("--max-epochs", type=int, default=None, help="SMOKE ONLY (writes <id>.smoke.json, never a result)")
    ap.add_argument("--micro-batch", type=int, default=MICRO_BATCH)
    args = ap.parse_args()

    import torch

    smoke = args.max_epochs is not None
    ckpt = DATA_CACHE_DIR / "ckpt" / (f"{args.run_id}.smoke{args.max_epochs}.pt" if smoke else f"{args.run_id}.pt")
    claim_gpu()
    setup_zoology(micro_batch=args.micro_batch, ckpt_path=ckpt, collapse_abort=True)
    import zoology.train as ztrain

    import our_mixers as OM

    cfg = build_mix_config(OM, args.run_id)
    if smoke:
        cfg.max_epochs = args.max_epochs
    import hybridlc.models.linear as hlin
    import zoology.mixers.gated_delta_net as zgdn

    counters = {"zoology_gdn": KernelCounter(zgdn), "hybridlc_gdn": KernelCounter(hlin)}
    models_sha = {}
    import hashlib
    for fn in ("attention.py", "linear.py", "mixers.py", "config.py"):
        models_sha[fn] = hashlib.sha256((REPO_ROOT / "src" / "hybridlc" / "models" / fn).read_bytes()).hexdigest()[:8]
    t_start = time.time()
    meta_start = {"git_commit": _git("rev-parse", "HEAD"), "git_status_porcelain_src": _git("status", "--porcelain", "src"),
                  "started_unix": t_start, "started_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                  "git_diff_quiet_src_models": _git("diff", "--stat", "HEAD", "--", "src/hybridlc/models") == "",
                  "src_models_sha256_prefix": models_sha}

    def on_valid(logger: CaptureLogger) -> None:
        ep = int(logger.valid[-1]["epoch"])
        pc = per_config_acc(logger.valid[-1])
        print(f"[x2] {args.run_id} epoch {ep} valid/acc={logger.valid[-1]['valid/accuracy']:.4f} "
              f"1024/256={pc.get('T1024_kv256', float('nan')):.4f} t={logger.epoch_end_s[-1]:.0f}s", flush=True)

    CaptureLogger.on_valid = staticmethod(on_valid)  # type: ignore[assignment]
    CaptureLogger.instances.clear()
    torch.cuda.reset_peak_memory_stats()

    ztrain.train(cfg)

    lg = CaptureLogger.instances[-1]
    model = CaptureLogger.model_ref
    final = lg.valid[-1]
    best = max(lg.valid, key=lambda r: r["valid/accuracy"])
    kernels = {k: c.record() for k, c in counters.items()}
    wdir = R5_DIR / "weights"
    wdir.mkdir(parents=True, exist_ok=True)
    wpath = wdir / f"{args.run_id}{'.smoke' if smoke else ''}.pt"
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, wpath)
    seq = {"name": "pattern:" + cfg.model.name}
    mixer = model.backbone.layers[1].sequence_mixer.mixer
    inner = getattr(mixer, "inner", mixer)
    result = {
        "run_id": args.run_id,
        "item": "X2-exploratory", "pattern": args.run_id.split("-")[1], "k_attention": args.run_id.split("-")[1].count("A"),
        "arm": cfg.model.name,
        "mixer": seq["name"],
        "d_model": cfg.model.d_model,
        "lr": float(cfg.learning_rate),
        "seed": cfg.seed,
        "data_seed": cfg.data.seed,
        "smoke": smoke,
        "final_epoch": int(final["epoch"]),
        "epochs_run": len(lg.valid),
        "early_stopped": (len(lg.valid) < cfg.max_epochs) and not lg.extra.get("collapsed", False),
        "collapsed": bool(lg.extra.get("collapsed", False)),
        "final_overall_acc": final["valid/accuracy"],
        "final_per_config_acc": per_config_acc(final),
        "final_valid_loss": final["valid/loss"],
        "best_epoch": int(best["epoch"]),
        "best_overall_acc": best["valid/accuracy"],
        "valid_history": lg.valid,
        "train_loss_by_epoch": lg.train_loss_by_epoch(),
        "epoch_end_times_s": lg.epoch_end_s,
        "wall_time_s": lg.elapsed_s(),
        "resumed_at_epochs": lg.resumed_at_epochs,
        "peak_mem_gib": torch.cuda.max_memory_allocated() / 2**30,
        "micro_batch": args.micro_batch,
        "model_info": lg.extra,
        "mixer_inner": {"class": type(inner).__name__,
                        **{k: getattr(inner, k) for k in ("num_heads", "head_dim", "head_k_dim", "head_v_dim", "expand_v",
                                                           "use_gate", "use_short_conv", "conv_size")
                           if isinstance(getattr(inner, k, None), (int, float, bool))}},
        "autocast_bf16_mixer": bool(getattr(mixer, "autocast_bf16", False)),
        "gdn_kernel_calls": kernels,
        "weights_path": str(wpath.relative_to(REPO_ROOT)),
        "provenance": {**meta_start, "zoology_commit": zoology_commit(), "zoology_dirty": zoology_dirty(),
                       **env_info(), "argv": sys.argv, "ended_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z")},
        "train_config": cfg.model_dump(mode="json"),
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{args.run_id}{'.smoke' if smoke else ''}.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, indent=1, default=str))
    os.replace(tmp, path)
    if ckpt.exists():
        ckpt.unlink()
    print(f"[x2] wrote {path} final_acc={result['final_overall_acc']:.4f} "
          f"1024/256={result['final_per_config_acc'].get('T1024_kv256')} wall={result['wall_time_s']:.0f}s "
          f"collapsed={result['collapsed']} kernels={kernels}", flush=True)


if __name__ == "__main__":
    main()

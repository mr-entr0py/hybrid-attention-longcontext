"""Run one R5 config (MQAR d_model = 64, Zoology GDN vs ours under bf16 autocast) and write its result JSON.

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

R5_DIR = REPO_ROOT / "experiments" / "repro" / "R5"


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

    runs = {c.run_id: c for c in OM.r5_runs(include_o5=True)}
    cfg = runs[args.run_id]
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
        print(f"[r5] {args.run_id} epoch {ep} valid/acc={logger.valid[-1]['valid/accuracy']:.4f} "
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
    seq = cfg.model.sequence_mixer.kwargs["configs"][1]
    mixer = model.backbone.layers[1].sequence_mixer.mixer
    inner = getattr(mixer, "inner", mixer)
    result = {
        "run_id": args.run_id,
        "item": "R5",
        "arm": args.run_id.split("-d64")[0],
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
    print(f"[r5] wrote {path} final_acc={result['final_overall_acc']:.4f} "
          f"1024/256={result['final_per_config_acc'].get('T1024_kv256')} wall={result['wall_time_s']:.0f}s "
          f"collapsed={result['collapsed']} kernels={kernels}", flush=True)


if __name__ == "__main__":
    main()

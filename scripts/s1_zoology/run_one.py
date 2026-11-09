"""Run one S1(a) Zoology MQAR config and write its result JSON (protocol §17).

Usage (one process per run, so each run gets a fresh CUDA context and RNG state):

    python scripts/s1_zoology/run_one.py --run-id attention-d64-lr1.0e-03

The training loop is Zoology's own ``zoology.train.train(config)``; the only substitution is
the logger (see ``zoology_compat.CaptureLogger``). Output:
``experiments/s1_zoology/runs/<run_id>.json`` with per-epoch validation metrics, the final
(= wandb-summary) accuracy overall and per test config, wall time and provenance.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from zoology_compat import (  # noqa: E402
    RESULTS_DIR, CaptureLogger, claim_gpu, setup_zoology, zoology_commit, zoology_dirty,
)
from s1_configs import build_runs, describe_config  # noqa: E402

# Test segments in the original file, keyed by num_kv_pairs (all distinct, so Zoology's
# ``slice_keys=["num_kv_pairs"]`` slices identify each test config uniquely).
TEST_CONFIG_KEYS = {4: "T64_kv4", 8: "T64_kv8", 16: "T64_kv16", 32: "T128_kv32",
                    64: "T256_kv64", 128: "T512_kv128", 256: "T1024_kv256"}


def per_config_acc(valid_record: dict[str, float]) -> dict[str, float]:
    out = {}
    for kv, name in TEST_CONFIG_KEYS.items():
        key = f"valid/num_kv_pairs/accuracy-{kv}"
        if key in valid_record:
            out[name] = valid_record[key]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--out-dir", default=str(RESULTS_DIR / "runs"))
    ap.add_argument("--max-epochs", type=int, default=None,
                    help="ONLY for smoke tests; results written with smoke=True and never used for G-RP")
    args = ap.parse_args()

    import torch

    claim_gpu()
    setup_zoology()
    import zoology.train as ztrain

    runs = {r.run_id: r for r in build_runs()}
    run = runs[args.run_id]
    cfg = run.config
    smoke = args.max_epochs is not None
    if smoke:
        cfg.max_epochs = args.max_epochs

    epoch_times: list[float] = []
    t_start = time.time()

    def on_valid(logger: CaptureLogger) -> None:
        epoch_times.append(time.time() - t_start)
        print(f"[s1] {run.run_id} epoch {int(logger.valid[-1]['epoch'])} "
              f"valid/acc={logger.valid[-1]['valid/accuracy']:.4f} t={epoch_times[-1]:.0f}s", flush=True)

    CaptureLogger.on_valid = staticmethod(on_valid)  # type: ignore[assignment]
    CaptureLogger.instances.clear()
    torch.cuda.reset_peak_memory_stats()

    ztrain.train(cfg)  # Zoology's own training loop (Trainer.fit, AdamW + cosine, early stopping)

    wall = time.time() - t_start
    lg = CaptureLogger.instances[-1]
    final = lg.valid[-1]
    best = max(lg.valid, key=lambda r: r["valid/accuracy"])
    result = {
        "run_id": run.run_id,
        "family": run.family,
        "d_model": run.d_model,
        "lr": run.lr,
        "smoke": smoke,
        "final_epoch": int(final["epoch"]),
        "epochs_run": len(lg.valid),
        "early_stopped": len(lg.valid) < cfg.max_epochs,
        # Primary metric: last validation = wandb summary value read by Zoology's analysis code.
        "final_overall_acc": final["valid/accuracy"],
        "final_per_config_acc": per_config_acc(final),
        "final_valid_loss": final["valid/loss"],
        # Secondary (not used for G-RP): best epoch.
        "best_epoch": int(best["epoch"]),
        "best_overall_acc": best["valid/accuracy"],
        "valid_history": lg.valid,
        "train_loss_by_epoch": lg.train_loss_by_epoch(),
        "epoch_end_times_s": epoch_times,
        "wall_time_s": wall,
        "peak_mem_gib": torch.cuda.max_memory_allocated() / 2**30,
        "model_info": lg.extra,
        "provenance": {
            "zoology_commit": zoology_commit(),
            "zoology_dirty": zoology_dirty(),
            "torch": torch.__version__,
            "fla": __import__("fla").__version__,
            "gpu": torch.cuda.get_device_name(0),
            "python": platform.python_version(),
            "host": platform.node(),
            "started_unix": t_start,
            "argv": sys.argv,
        },
        "train_config": describe_config(cfg),
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{run.run_id}{'.smoke' if smoke else ''}.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, indent=1))
    os.replace(tmp, path)  # atomic: the queue treats an existing JSON as "done"
    print(f"[s1] wrote {path} final_acc={result['final_overall_acc']:.4f} wall={wall:.0f}s", flush=True)


if __name__ == "__main__":
    main()

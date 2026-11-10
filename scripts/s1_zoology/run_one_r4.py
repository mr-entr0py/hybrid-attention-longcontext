"""Run one R4 (S1(b) + R4b) config in Zoology's harness and write its result JSON.

Pre-registration: ``experiments/repro/R4/prereg.yaml`` (plan §2.4). Same harness as R1's ``run_one.py``
(Zoology's own ``zoology.train.train``; shims: exact micro-batch accumulation, per-epoch resume), plus:

* the run is looked up in ``our_mixers.r4_runs()`` (4 G-RP(b) cells, 2 R4b runs, 2 config-A runs);
* the collapse-abort rule of plan §2.0 is active (reading C1);
* ``hybridlc.models.attention._varlen_attn`` is wrapped by a call counter (evidence that the bf16 arm runs
  the FA2 varlen kernel and that fp32 arms do not);
* after training: attention max-logit / entropy on the first test batch of each test config (F_g runs),
  final weights saved to ``experiments/repro/R4/weights/<run_id>.pt``.

    python scripts/s1_zoology/run_one_r4.py --run-id our_gdn-d128-lr1.0e-03
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

R4_DIR = REPO_ROOT / "experiments" / "repro" / "R4"


def _git(*a: str) -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO_ROOT), *a], capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as e:  # pragma: no cover
        return f"error: {e}"


class VarlenCounter:
    """Counts calls of hybridlc.models.attention._varlen_attn (module global used by attn_varlen_train)."""

    def __init__(self) -> None:
        import hybridlc.models.attention as att

        self.att, self.orig = att, att._varlen_attn
        self.calls, self.first_dtype = 0, None
        if self.orig is None:
            return

        def wrapped(q, *a, **k):
            self.calls += 1
            if self.first_dtype is None:
                self.first_dtype = str(q.dtype)
            return self.orig(q, *a, **k)

        att._varlen_attn = wrapped

    def record(self) -> dict:
        return {"available": self.orig is not None, "calls": self.calls, "first_q_dtype": self.first_dtype,
                "FORCE_REFERENCE": bool(getattr(self.att, "FORCE_REFERENCE", False))}


def attn_stats_by_config(model, cfg) -> dict:
    """Max-logit / entropy of our F_g on the first test batch of each test config (eval mode, no grad)."""
    import torch
    from zoology.data.utils import prepare_data

    mixer = model.backbone.layers[1].sequence_mixer.mixer
    inner = getattr(mixer, "inner", None)
    if inner is None or not hasattr(inner, "collect_attn_stats"):
        return {}
    _, test_dl = prepare_data(cfg.data)
    out: dict = {}
    model.eval()
    inner.collect_attn_stats = True
    with torch.no_grad():
        for inputs, _targets, slices in test_dl:
            kv = int(slices[0]["num_kv_pairs"])
            name = TEST_CONFIG_KEYS.get(kv, str(kv))
            if name in out:
                continue
            model(inputs.cuda())
            out[name] = dict(inner.attn_stats or {})
    inner.collect_attn_stats = False
    return out


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
    ap.add_argument("--out-dir", default=str(R4_DIR / "runs"))
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

    runs = {c.run_id: c for c in OM.r4_runs()}
    cfg = runs[args.run_id]
    if smoke:
        cfg.max_epochs = args.max_epochs
    counter = VarlenCounter()
    t_start = time.time()
    meta_start = {"git_commit": _git("rev-parse", "HEAD"), "git_status_porcelain_src": _git("status", "--porcelain", "src"),
                  "started_unix": t_start, "started_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z")}

    def on_valid(logger: CaptureLogger) -> None:
        ep = int(logger.valid[-1]["epoch"])
        print(f"[r4] {args.run_id} epoch {ep} valid/acc={logger.valid[-1]['valid/accuracy']:.4f} "
              f"t={logger.epoch_end_s[-1]:.0f}s varlen_calls={counter.calls}", flush=True)

    CaptureLogger.on_valid = staticmethod(on_valid)  # type: ignore[assignment]
    CaptureLogger.instances.clear()
    torch.cuda.reset_peak_memory_stats()

    ztrain.train(cfg)

    lg = CaptureLogger.instances[-1]
    model = CaptureLogger.model_ref
    final = lg.valid[-1]
    best = max(lg.valid, key=lambda r: r["valid/accuracy"])
    varlen_train = counter.record()
    stats = attn_stats_by_config(model, cfg)
    wdir = R4_DIR / "weights"
    wdir.mkdir(parents=True, exist_ok=True)
    wpath = wdir / f"{args.run_id}{'.smoke' if smoke else ''}.pt"
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, wpath)
    seq = cfg.model.sequence_mixer.kwargs["configs"][1]
    inner = model.backbone.layers[1].sequence_mixer.mixer.inner
    result = {
        "run_id": args.run_id,
        "item": "R4",
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
                        **{k: getattr(inner, k) for k in ("n_heads", "head_dim", "use_qk_norm", "use_rope", "rope_theta",
                                                           "use_out_gate", "num_heads", "head_k_dim", "head_v_dim",
                                                           "use_gate", "use_short_conv") if hasattr(inner, k)}},
        "autocast_bf16_mixer": bool(getattr(model.backbone.layers[1].sequence_mixer.mixer, "autocast_bf16", False)),
        "varlen_attn_training": varlen_train,
        "attn_stats_post_training": stats,
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
    print(f"[r4] wrote {path} final_acc={result['final_overall_acc']:.4f} wall={result['wall_time_s']:.0f}s "
          f"collapsed={result['collapsed']} varlen={varlen_train}", flush=True)


if __name__ == "__main__":
    main()

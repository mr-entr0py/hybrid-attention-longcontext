#!/usr/bin/env python
"""Score finished core TEF checkpoints (decisions D103-D105: UNREGISTERED, EXPLORATORY).

    python scripts/repro/gpu_lock.py run --label CORE-EVAL --wait-exclusive -- python scripts/repro/core_eval.py --seed 0

Suite FD (development namespace; the held-out CS suite is NOT opened, D105), production weights (P0: final.pt
rounded to bf16), no candidate scoring, bf16 single-pass re-score (P2) only (D104). Resumable (done markers in
experiments/core/logs). Uses its own extraction of the pre-FEAS Triton archive.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "repro"))
import feas_smoke as S  # noqa: E402  (shared helpers: cache extraction, registry rows)

OUT = REPO / "experiments" / "core"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--configs", default="C0,C12,C3")
    a = ap.parse_args()
    rows = S.reg_rows(REPO / "experiments" / "registry.jsonl")
    runs = {}
    for cfg in a.configs.split(","):
        name = f"S4_D384_{cfg}_s{a.seed}_B060_T.yaml"
        q = [r for r in rows if r.get("status") == "queued" and Path(str(r.get("config", ""))).name == name]
        assert q, f"{name} not in the registry"
        rid = q[-1]["run_id"]
        ck = REPO / "checkpoints" / rid / "final.pt"
        done = any(r.get("run_id") == rid and r.get("status") == "completed" for r in rows)
        if done and ck.is_file():
            runs[cfg] = (rid, ck)
        else:
            print(f"[core-eval] {cfg} seed {a.seed}: training not complete, skipped", flush=True)
    keep = REPO / ".cache" / "triton-core-eval"
    if not keep.exists():
        shutil.copytree(S.fresh_cache(), keep)
    env = S.base_env(TRITON_CACHE_DIR=str(keep))
    logs = OUT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    n_expected = len(list((REPO / "experiments" / "items" / "dev" / "FD").glob("*.npz")))
    failed = []
    for cfg, (rid, ck) in runs.items():
        done = logs / f"{rid}.done"
        if done.exists():
            continue
        cmd = [S.PY, "scripts/evaluate.py", "--ckpt", str(ck), "--results-root", str(REPO / "experiments" / "results"),
               "--runs-root", str(REPO / "runs"), "--suite", "FD", "--label", "tef1000", "--no-candidates",
               "--rescore", "--rescore-ref", "bf16"]
        t = time.time()
        with open(logs / f"{rid}.log", "w") as f:
            rc = subprocess.run(cmd, cwd=REPO, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
        got = len(list((REPO / "experiments" / "results" / rid / "tef1000" / "FD").glob("*.jsonl")))
        print(f"[core-eval {time.strftime('%H:%M:%S')}] {cfg} s{a.seed} {rid}: rc={rc} cells={got}/{n_expected} "
              f"wall={time.time() - t:.0f}s", flush=True)
        if rc == 0 and got >= n_expected:
            done.write_text(json.dumps({"cmd": cmd[1:], "cells": got, "wall_s": time.time() - t}))
        else:
            failed.append(rid)
    print(f"[core-eval] seed {a.seed} finished; failed: {failed}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

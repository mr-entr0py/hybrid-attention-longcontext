#!/usr/bin/env python
"""VT-13 production-mode resume check with a run-to-run noise-floor control (integrator, S0).

Protocol §16 VT-13: "Production: |Δloss| < 0.005 at resume + 500". Production mode uses
non-deterministic kernels, so two *clean* runs already differ. This driver runs, all with the
same production recipe (D384 C3, seed 901, η = 2e-3, real Stage-A data, 640 steps, checkpoint
every 50):

  a  clean run
  c  clean run (control: measures the inherent run-to-run difference)
  b  kill -9 at step 137 micro 3 (mid-accumulation), then resume (restarts from step 100)

and reports |Δ| of the training-loss window means at resume + 500 (steps 630–639, the window
used by tests/test_vt13_training.py::test_vt13_production_resume) for (a, b), (c, b) and the
control pair (a, c), plus wider windows. Exit code 75 (G-OOM on the shared GPU) is retried
by resuming, exactly as the runner would.

Usage: python scripts/vt13_production_resume.py --out experiments/s0/vt13_production [--steps 640]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TRAIN = REPO / "scripts" / "train.py"
OOM_EXIT = 75


def _env(fault: dict | None) -> dict:
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "PYTHONHASHSEED": "0"}
    env.pop("HYBRIDLC_FAULT", None)
    if fault:
        env["HYBRIDLC_FAULT"] = json.dumps(fault)
    return env


def train(run_yaml: Path, root: Path, overrides: list[str], fault: dict | None = None,
          max_oom_retries: int = 5) -> list[int]:
    """Run scripts/train.py; on exit 75 (OOM) resume up to ``max_oom_retries`` times. Returns all exit codes."""
    cmd = [sys.executable, str(TRAIN), "--config", str(run_yaml), "--allow-dirty",
           "--runs-root", str(root / "runs"), "--ckpt-root", str(root / "ckpt"),
           "--registry", str(root / "registry.jsonl"), *overrides]
    codes = []
    log = (root / "train.log").open("a")
    for attempt in range(max_oom_retries + 1):
        log.write(f"\n=== {time.strftime('%F %T')} attempt {attempt} fault={fault} ===\n")
        log.flush()
        rc = subprocess.run(cmd, env=_env(fault if attempt == 0 else None), stdout=log, stderr=subprocess.STDOUT,
                            cwd=str(REPO)).returncode
        codes.append(rc)
        if rc != OOM_EXIT:
            break
        time.sleep(30)
    log.close()
    return codes


def losses(root: Path) -> dict[int, float]:
    (d,) = [p for p in (root / "runs").iterdir() if p.is_dir() and not p.name.startswith("_")]
    out: dict[int, float] = {}
    for line in (d / "trace.jsonl").read_text().splitlines():
        r = json.loads(line)
        out[int(r["step"])] = float(r["loss"])  # last record per step wins (resumed steps re-logged)
    return out


def wmean(t: dict[int, float], lo: int, hi: int) -> float:
    return sum(t[s] for s in range(lo, hi)) / (hi - lo)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=640)
    ap.add_argument("--parallel-clean", action="store_true", help="run a and c concurrently")
    args = ap.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    run = out / "run.yaml"
    run.write_text("exp_stage: S0\nmodel: configs/model/D384/C3.yaml\ntraining: configs/training/stage_A.yaml\n"
                   "seed_index: 901\nlr: 2.0e-3\n")
    ov = ["-o", f"schedule.steps={args.steps}", "-o", "checkpoint.every=50", "-o", "validation.every=0",
          "-o", "checkpoint.milestone_tokens=[]"]
    roots = {k: out / k for k in ("a", "b", "c")}
    for r in roots.values():
        r.mkdir(exist_ok=True)
    t0 = time.time()
    codes: dict[str, list[int]] = {}
    if args.parallel_clean:
        with ThreadPoolExecutor(2) as ex:
            fa = ex.submit(train, run, roots["a"], ov)
            fc = ex.submit(train, run, roots["c"], ov)
            codes["a"], codes["c"] = fa.result(), fc.result()
    else:
        codes["a"] = train(run, roots["a"], ov)
        codes["c"] = train(run, roots["c"], ov)
    codes["b_kill"] = train(run, roots["b"], ov, fault={"kill_at": [137, 3]}, max_oom_retries=0)
    codes["b_resume"] = train(run, roots["b"], ov)
    wall = time.time() - t0
    t = {k: losses(r) for k, r in roots.items()}
    end = args.steps
    res: dict = {"steps": args.steps, "codes": codes, "wall_s": wall, "windows": {}}
    for lo, hi in ((end - 10, end), (end - 40, end), (end - 140, end)):
        m = {k: wmean(v, lo, hi) for k, v in t.items()}
        res["windows"][f"{lo}-{hi - 1}"] = {
            "mean": m, "d_ab": abs(m["a"] - m["b"]), "d_cb": abs(m["c"] - m["b"]), "d_ac": abs(m["a"] - m["c"])}
    # Resume restarted from the step-100 checkpoint: steps < 100 must match a's prefix only up to production noise.
    res["first_step_diff_ab"] = {s: abs(t["a"][s] - t["b"][s]) for s in (0, 50, 99, 100, 137, 200, 400, end - 1)}
    res["first_step_diff_ac"] = {s: abs(t["a"][s] - t["c"][s]) for s in (0, 50, 99, 100, 137, 200, 400, end - 1)}
    w = res["windows"][f"{end - 10}-{end - 1}"]
    res["criterion"] = {"window": f"{end - 10}-{end - 1}", "threshold": 0.005,
                        "pass_ab": w["d_ab"] < 0.005, "pass_cb": w["d_cb"] < 0.005, "control_ac": w["d_ac"]}
    (out / "result.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

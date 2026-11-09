#!/usr/bin/env python
"""G0 throughput and peak memory (protocol §13.3, VT-20) with the *production* trainer.

Each measurement is a short real training run (``scripts/train.py``, real ``StageStream`` data, the
production Stage-A / Stage-B recipes, bf16, non-deterministic mode) with a GPU monitor sampling
``nvidia-smi`` alongside, so the contention level is recorded with the number:

* Stage A: 2 x 4K micro-batches x 16 accumulation (131,072 tok/step), no checkpointing;
* Stage B: 1 x 32K x 4 accumulation, per-block activation checkpointing (from init: throughput does
  not depend on the weights).

tok/s is the trainer's own 10-step window value (``metrics.jsonl`` ``tok_s``), reported as the median
and the best window after dropping the first window (Triton JIT / allocator warm-up). Peak memory is
the trainer's ``mem_run_peak`` (max allocated) plus the max reserved. Thresholds (§13.3): Stage A >=
95K (D384) / 70K (D512) tok/s for every k; C12 Stage B >= 27K (D384) / 20K (D512); peak <= 7 GiB.

    python scripts/g0_throughput.py --out experiments/s0/g0 [--stage A B] [--scales D384 D512]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

G0_STAGE_A = {"D384": 95_000, "D512": 70_000}
G0_STAGE_B_C12 = {"D384": 27_000, "D512": 20_000}
MEM_CAP_GIB = 7.0


def run_yaml(out: Path, stage: str, scale: str, cfg: str, steps: int) -> Path:
    recipe = "stage_A.yaml" if stage == "A" else "stage_B.yaml"
    p = out / "yaml" / f"{stage}_{scale}_{cfg}.yaml"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        f"exp_stage: S0G0\nmodel: configs/model/{scale}/{cfg}.yaml\ntraining: configs/training/{recipe}\n"
        f"seed_index: 0\nlr: 2.0e-3\noverrides:\n  schedule.steps: {steps}\n  checkpoint.every: 1000000\n"
        f"  checkpoint.save_final: false\n  checkpoint.milestone_tokens: []\n  checkpoint.milestone_steps: []\n"
        f"  validation.every: 0\n  validation.final_n_batches: 0\n")
    return p


def last_rows(path: Path, kind: str) -> list[dict]:
    rows: dict[int, dict] = {}
    for line in path.read_text().splitlines():
        r = json.loads(line)
        if r.get("kind") == kind:
            rows[r["step"]] = r
    return [rows[k] for k in sorted(rows)]


def measure(out: Path, stage: str, scale: str, cfg: str, steps: int, timeout: int) -> dict:
    y = run_yaml(out, stage, scale, cfg, steps)
    root = out / "runs_root"
    env = {**os.environ, "PYTHONHASHSEED": "0"}
    cmd = [sys.executable, str(REPO / "scripts" / "train.py"), "--config", str(y), "--allow-dirty", "--no-resume",
           "--runs-root", str(root / "runs"), "--ckpt-root", str(root / "ckpt"), "--registry", str(root / "registry.jsonl")]
    t0 = time.time()
    log = out / "logs" / f"{stage}_{scale}_{cfg}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w") as lf:
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env, cwd=str(REPO))
        gpu_out = out / "gpu" / f"{stage}_{scale}_{cfg}.jsonl"
        gpu_out.parent.mkdir(parents=True, exist_ok=True)
        mon = subprocess.Popen([sys.executable, str(REPO / "scripts" / "gpu_monitor.py"), "--out", str(gpu_out),
                                "--pid", str(proc.pid), "--interval", "5"], stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, cwd=str(REPO))
        try:
            code = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            code = -999
        try:
            mon.wait(timeout=30)
        except subprocess.TimeoutExpired:
            mon.terminate()
    res = dict(stage=stage, scale=scale, config=cfg, steps=steps, exit_code=code, wall_s=time.time() - t0, log=str(log))
    from hybridlc.training import gpumon

    try:
        res["gpu"] = gpumon.summarize_file(gpu_out)
    except Exception as e:  # noqa: BLE001
        res["gpu_error"] = repr(e)
    runs = [d for d in (root / "runs").iterdir() if d.is_dir() and d.name.startswith(f"S0G0{'' if stage == 'A' else 'B'}-{scale}-{cfg}-")]
    if not runs or code != 0:
        res["status"] = "failed"
        return res
    rows = last_rows(sorted(runs, key=lambda d: d.stat().st_mtime)[-1] / "metrics.jsonl", "train")
    win = [r["tok_s"] for r in rows if r["step"] >= 10 and r.get("tok_s")]  # drop the first (JIT) window
    win.sort()
    res.update(status="ok", tok_s_windows=win, tok_s_median=win[len(win) // 2] if win else None,
               tok_s_best=max(win) if win else None,
               peak_alloc_gib=max(r.get("mem_run_peak") or 0 for r in rows) / 2**30,
               peak_reserved_gib=max(r.get("mem_reserved") or 0 for r in rows) / 2**30,
               final_loss=rows[-1]["loss"] if rows else None)
    if stage == "A":
        res["threshold"] = G0_STAGE_A[scale]
    elif cfg == "C12":
        res["threshold"] = G0_STAGE_B_C12[scale]
    if "threshold" in res and res["tok_s_median"] is not None:
        res["pass_median"] = res["tok_s_median"] >= res["threshold"]
        res["pass_best"] = res["tok_s_best"] >= res["threshold"]
    res["mem_pass"] = res["peak_reserved_gib"] <= MEM_CAP_GIB
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=REPO / "experiments" / "s0" / "g0")
    ap.add_argument("--stage", nargs="+", default=["A", "B"], choices=["A", "B"])
    ap.add_argument("--scales", nargs="+", default=["D384", "D512"])
    ap.add_argument("--configs-A", nargs="+", default=["C0", "C3", "C12"])
    ap.add_argument("--configs-B", nargs="+", default=["C12", "C0"])
    ap.add_argument("--steps-A", type=int, default=40)
    ap.add_argument("--steps-B", type=int, default=20)
    ap.add_argument("--timeout", type=int, default=1800)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    results_path = a.out / "g0_results.jsonl"
    for stage in a.stage:
        for scale in a.scales:
            for cfg in (a.configs_A if stage == "A" else a.configs_B):
                r = measure(a.out, stage, scale, cfg, a.steps_A if stage == "A" else a.steps_B, a.timeout)
                r["ts"] = time.time()
                with open(results_path, "a") as f:
                    f.write(json.dumps(r) + "\n")
                g = r.get("gpu", {})
                print(f"{stage} {scale} {cfg}: {r.get('status')} median {r.get('tok_s_median')} best {r.get('tok_s_best')} "
                      f"thr {r.get('threshold')} peak_alloc {r.get('peak_alloc_gib')} reserved {r.get('peak_reserved_gib')} "
                      f"contended_frac {g.get('contended_frac')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

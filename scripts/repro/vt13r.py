#!/usr/bin/env python
"""VT-13r: bitwise resume exactness of the real model through the runner and the trainer (prereg experiments/prereg/VT-13r.yaml).

Launch (takes the single GPU lock for the whole check; children inherit the token):
    python scripts/repro/gpu_lock.py run --label VT-13r --wait-exclusive -- python scripts/repro/vt13r.py [--steps 200]

Design (D384 C3, audit mode, real Stage-A data, seed index 903, lr 2e-3, 200 steps, checkpoint every 50):
  p1  clean run through scripts/runner.py
  r1  scripts/train.py, SIGKILL at step 137 micro-batch 3 (HYBRIDLC_FAULT kill_at), resumed by scripts/train.py
  r2  runner; at step 137 mb 3 the child is SIGKILLed AND an Xid-8 + RC-watchdog line is injected into the test journal
      (HYBRIDLC_FAULT inject_journal + HYBRIDLC_JOURNAL_SOURCE); the runner must classify infra_failed_attempt and auto-resume
  r3  runner; SIGKILL inside the step-150 checkpoint write (kill_in_ckpt); an ordinary failed attempt, then resume
Criteria: R1 r1/r2/r3 bitwise == p1 (per-step loss/grad-norm hex, batch hashes, final param SHA-256); R2 r2 classified infra and
resumed, r1/r3 hold no infra class; R3 r2's infra attempt is not counted as a failure; r3's failed attempt counts 1 and is reset
by the completed resume. Output: experiments/repro/VT-13r/vt13r_result.json.
This script never touches CUDA itself; it only launches scripts/runner.py and scripts/train.py. Nothing is tuned to pass.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
OUT = REPO / "experiments" / "repro" / "VT-13r"
PY = sys.executable
SEED, LR = 903, "2.0e-3"
XID = ["NVRM: Xid (PCI:0000:02:00): 8, pid=1, name=python, channel 0x00000003",
       "NVRM: krcWatchdog_IMPL: RC watchdog: GPU is probably locked!  Notify Timeout Seconds: 7"]


def log(m: str) -> None:
    print(f"[vt13r {time.strftime('%H:%M:%S')}] {m}", flush=True)


def env_for(root: Path, fault: dict | None = None) -> dict:
    e = dict(os.environ)
    e.update({"TRITON_CACHE_DIR": str(REPO / ".cache" / "triton-diag"), "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONHASHSEED": "0"})
    e.pop("HYBRIDLC_FAULT", None)
    if fault:
        e["HYBRIDLC_FAULT"] = json.dumps(fault)
        e["HYBRIDLC_JOURNAL_SOURCE"] = str(root / "journal_injected.txt")
    return e


def roots(root: Path) -> list[str]:
    return ["--runs-root", str(root / "runs"), "--ckpt-root", str(root / "ckpt"), "--registry", str(root / "registry.jsonl")]


def sh(cmd: list[str], env: dict, logf: Path) -> int:
    log("run: " + " ".join(cmd[2:8]) + " ...")
    with open(logf, "ab") as f:
        return subprocess.run(cmd, cwd=REPO, env=env, stdout=f, stderr=subprocess.STDOUT).returncode


def run_yaml(steps: int) -> Path:
    p = OUT / "run.yaml"
    p.write_text(f"exp_stage: S0\nmodel: configs/model/D384/C3.yaml\ntraining: configs/training/vt13_audit.yaml\nseed_index: {SEED}\nlr: {LR}\n")
    return p


def ov(steps: int) -> list[str]:
    return ["-o", f"schedule.steps={steps}", "-o", "checkpoint.every=50"]


def run_dir_of(root: Path) -> Path:
    ds = [d for d in (root / "runs").glob("*") if (d / "trace.jsonl").exists()]
    assert len(ds) == 1, f"expected exactly one run dir under {root}/runs, found {ds}"
    return ds[0]


def trace_of(rd: Path) -> tuple[list, str | None]:
    rows = {}
    for line in (rd / "trace.jsonl").read_text().splitlines():
        try:
            j = json.loads(line)
            rows[j["step"]] = (j["step"], j["loss_hex"], j["gnorm_hex"], j["batch_hash"])  # last record per step wins (resumes)
        except Exception:
            continue
    psha = None
    for line in (rd / "metrics.jsonl").read_text().splitlines():
        try:
            j = json.loads(line)
        except Exception:
            continue
        if j.get("kind") == "end":
            psha = j.get("param_sha256")
    return [rows[k] for k in sorted(rows)], psha


def registry_of(root: Path) -> tuple[str, list[dict]]:
    from hybridlc.training.registry import Registry
    reg = Registry(root / "registry.jsonl")
    rid = sorted(reg.state())[0]
    return rid, reg.history(rid)


def failure_counts(root: Path) -> dict:
    from hybridlc.training.config import PathsConfig
    from hybridlc.training.queue import Runner, RunnerConfig
    p = PathsConfig(runs_root=str(root / "runs"), ckpt_root=str(root / "ckpt"), registry=str(root / "registry.jsonl"))
    return Runner(RunnerConfig(paths=p, monitor=False)).failure_counts()


def runner_job(name: str, steps: int, fault: dict | None, max_loops: int = 6) -> None:
    root = OUT / name
    root.mkdir(parents=True, exist_ok=True)
    cfg = run_yaml(steps)
    base = [PY, "scripts/runner.py", *roots(root), "--no-monitor"]
    sh(base + ["enqueue", str(cfg), "--tag", name, *ov(steps)], env_for(root), root / "runner.log")
    for i in range(max_loops):  # the runner resumes classified attempts; loop until the run is completed
        sh(base + ["run", "--max-jobs", "1"], env_for(root, fault), root / "runner.log")
        rid, hist = registry_of(root)
        if hist and hist[-1]["status"] in ("completed", "done", "paused", "failed"):
            break
        if any(r["status"] == "completed" for r in hist):
            break


def trainer_job(name: str, steps: int, fault: dict) -> None:
    root = OUT / name
    root.mkdir(parents=True, exist_ok=True)
    cmd = [PY, "scripts/train.py", "--config", str(run_yaml(steps)), "--tag", name, *ov(steps), *roots(root)]
    rc1 = sh(cmd, env_for(root, fault), root / "trainer.log")
    log(f"{name}: first attempt rc={rc1} (expected -9 / 137)")
    sh(cmd, env_for(root), root / "trainer.log")  # resume, no fault


def compare(a, b) -> dict:
    ta, pa = a
    tb, pb = b
    return {"n_steps": [len(ta), len(tb)], "trace_identical": bool(ta) and ta == tb, "param_sha256": [pa, pb], "sha_identical": bool(pa) and pa == pb}


def evaluate(steps: int) -> dict:
    p1 = trace_of(run_dir_of(OUT / "p1"))
    res = {"item": "VT-13r", "steps": steps, "criteria": {}}
    for n in ("r1", "r2", "r3"):
        res[f"{n}_vs_p1"] = compare(trace_of(run_dir_of(OUT / n)), p1)
    res["criteria"]["R1"] = all(res[f"{n}_vs_p1"]["trace_identical"] and res[f"{n}_vs_p1"]["sha_identical"] for n in ("r1", "r2", "r3"))
    infra_rows = {}
    for n in ("r1", "r2", "r3"):
        try:
            rid, hist = registry_of(OUT / n)
        except Exception as e:  # r1 is trainer-only and may have no registry rows
            infra_rows[n] = {"error": repr(e)}
            continue
        ae = [r for r in hist if r["status"] == "attempt_end"]
        infra_rows[n] = {"statuses": [r["status"] for r in hist],
                         "infra_flagged": any(bool(r.get("infra_flagged")) for r in ae),
                         "infra_class": [r.get("infra_class") for r in ae],
                         "n_infra_failed_attempt": sum(1 for r in hist if r["status"] == "infra_failed_attempt"),
                         "n_crashed": sum(1 for r in hist if r["status"] == "crashed")}
    res["registry"] = infra_rows
    r2, r3 = infra_rows["r2"], infra_rows["r3"]
    r2_resumed = r2.get("statuses", []).count("running") >= 2 and "completed" in r2.get("statuses", [])  # last row is attempt_end
    r1_clean = "error" in infra_rows["r1"] or infra_rows["r1"].get("n_infra_failed_attempt", 0) == 0  # trainer-only run: no infra class
    res["criteria"]["R2"] = bool(r2.get("n_infra_failed_attempt", 0) >= 1 and r2_resumed
                                 and r3.get("n_infra_failed_attempt", 0) == 0 and r1_clean)
    fc2, fc3 = failure_counts(OUT / "r2"), failure_counts(OUT / "r3")
    c2 = next(iter(fc2.get("runs", {}).values()), {})  # failure_counts() -> {"runs": {run_id: counts}, "per_k": ...}
    c3 = next(iter(fc3.get("runs", {}).values()), {})
    res["failure_counts"] = {"r2": c2, "r3": c3}
    res["criteria"]["R3"] = bool(c2.get("counted_failures") == 0 and c2.get("infra_failed_attempt") == 1
                                 and r3.get("n_crashed") == 1 and c3.get("counted_failures") == 0)
    res["verdict"] = "PASS" if all(res["criteria"].values()) else "FAIL"
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--evaluate-only", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if not a.evaluate_only:
        runner_job("p1", a.steps, None)
        trainer_job("r1", a.steps, {"kill_at": [137, 3]})
        runner_job("r2", a.steps, {"kill_at": [137, 3], "inject_journal": XID})
        runner_job("r3", a.steps, {"kill_in_ckpt": 150})
    res = evaluate(a.steps)
    (OUT / "vt13r_result.json").write_text(json.dumps(res, indent=1, default=str))
    log(f"verdict {res['verdict']} criteria {res['criteria']}")
    return 0 if res["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())

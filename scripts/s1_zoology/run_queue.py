"""Sequential S1(a) queue with the pre-declared cut rule (protocol §17). Resumable.

Launch (detached, logs under experiments/s1_zoology/logs/):

    nohup python scripts/s1_zoology/run_queue.py > experiments/s1_zoology/logs/queue.log 2>&1 &

Order:
  PHASE1 = attention d64 × 4 LRs, then GDN d128 × 4 LRs.
  After the first run finishes, cut_decision.json records the implied total
  (first run's wall time × 16). If it exceeds 3.0 GPU-h, PHASE2 (attention d128, GDN d256)
  runs only the 2 best LRs per family, chosen on that family's PHASE1 width.
A run is "done" when experiments/s1_zoology/runs/<run_id>.json exists; re-launching skips it.
Each run is a separate subprocess (fresh CUDA context); a failed attempt is retried and resumes
from the run's last per-epoch checkpoint (up to MAX_ATTEMPTS attempts).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import s1_analysis as A  # noqa: E402

REPO = HERE.parents[1]
RES = REPO / "experiments" / "s1_zoology"
RUNS = RES / "runs"
LOGS = RES / "logs"
CUT_FILE = RES / "cut_decision.json"
# Attempts per run. Each attempt resumes from the last per-epoch checkpoint, so a retry after
# a driver watchdog kill (cudaErrorLaunchTimeout on the shared display GPU) loses < 1 epoch.
MAX_ATTEMPTS = int(os.environ.get("S1_MAX_ATTEMPTS", "6"))


def _done(rid: str) -> bool:
    return (RUNS / f"{rid}.json").exists()


def _load(rid: str) -> dict:
    return json.loads((RUNS / f"{rid}.json").read_text())


def _run(rid: str) -> None:
    for attempt in range(1, MAX_ATTEMPTS + 1):
        if _done(rid):
            return
        log = LOGS / f"{rid}.attempt{attempt}.log"
        print(f"[queue] {time.strftime('%F %T')} start {rid} (attempt {attempt}) -> {log}", flush=True)
        with open(log, "w") as fh:
            rc = subprocess.call([sys.executable, str(HERE / "run_one.py"), "--run-id", rid],
                                 stdout=fh, stderr=subprocess.STDOUT, cwd=str(REPO))
        print(f"[queue] {time.strftime('%F %T')} end {rid} rc={rc}", flush=True)
        if rc == 0 and _done(rid):
            return
    print(f"[queue] FAILED {rid} after {MAX_ATTEMPTS} attempts", flush=True)


def main() -> None:
    RUNS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    phase1 = A.phase_run_ids(A.PHASE1)

    for rid in phase1:
        _run(rid)
        if not CUT_FILE.exists() and _done(phase1[0]):
            dec = A.cut_decision(_load(phase1[0])["wall_time_s"])
            dec["first_run_id"] = phase1[0]
            CUT_FILE.write_text(json.dumps(dec, indent=1))
            print(f"[queue] cut decision: {dec}", flush=True)

    dec = json.loads(CUT_FILE.read_text())
    if dec["cut"]:
        results = [_load(r) for r in phase1 if _done(r)]
        keep = {fam: A.best_lrs(results, fam, A.FIRST_WIDTH[fam]) for fam in A.WIDTHS}
        dec["kept_lrs"] = keep
        CUT_FILE.write_text(json.dumps(dec, indent=1))
        phase2 = A.phase_run_ids(A.PHASE2, keep)
    else:
        phase2 = A.phase_run_ids(A.PHASE2)
    print(f"[queue] phase2: {phase2}", flush=True)
    for rid in phase2:
        _run(rid)
    subprocess.call([sys.executable, str(HERE / "summarize.py")], cwd=str(REPO))
    print("[queue] all done", flush=True)


if __name__ == "__main__":
    main()

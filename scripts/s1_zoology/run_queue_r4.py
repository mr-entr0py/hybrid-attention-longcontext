"""Sequential, resumable R4 queue (plan §2.4; prereg experiments/repro/R4/prereg.yaml).

Launch detached:  bash scripts/repro/r4_launch.sh      (log: experiments/repro/R4/logs/queue.log)

* Order: our_mixers.r4_runs() (4 G-RP(b) cells, 2 R4b runs, 2 config-A runs).
* A run is done when experiments/repro/R4/runs/<run_id>.json exists; re-launching skips it.
* Each attempt is a fresh subprocess wrapped by scripts/repro/infra_events.py (takes the single GPU lock
  scripts/repro/gpu_lock.py; tries --wait-exclusive for <= 10 min, then lock only), which writes
  experiments/repro/R4/infra/R4-<run_id>-a<k>.infra.json (kernel Xid/watchdog/OOM lines, GPU samples, concurrent
  processes). Failed attempts resume from the per-epoch checkpoint; at most MAX_ATTEMPTS attempts (plan §2.0).
* Plan §2.0: a gate run whose window contains an infra event is repeated in full (the flagged result is archived
  as <run_id>.flagged<k>.json). An Xid / RC-watchdog event stops the queue (plan §2.2 trigger O4).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
R4 = REPO / "experiments" / "repro" / "R4"
RUNS, LOGS, INFRA = R4 / "runs", R4 / "logs", R4 / "infra"
CKPT_DIR = REPO / "data" / "zoology" / "ckpt"
INFRA_PY = REPO / "scripts" / "repro" / "infra_events.py"
MAX_ATTEMPTS = int(os.environ.get("R4_MAX_ATTEMPTS", "6"))
MAX_FULL_REPEATS = 2
EXCLUSIVE_WAIT_S = float(os.environ.get("R4_EXCLUSIVE_WAIT_S", "600"))


def log(msg: str) -> None:
    print(f"[r4-queue] {time.strftime('%F %T')} {msg}", flush=True)


def run_ids() -> list[str]:
    from zoology_compat import setup_zoology

    setup_zoology(micro_batch=None)
    import our_mixers as OM

    return [c.run_id for c in OM.r4_runs()]


def _done(rid: str) -> bool:
    return (RUNS / f"{rid}.json").exists()


def _next_attempt(rid: str) -> int:
    return 1 + max((int(p.name.rsplit("-a", 1)[1].split(".")[0]) for p in INFRA.glob(f"R4-{rid}-a*.infra.json")),
                   default=0)


def _attempt(rid: str, k: int) -> tuple[int, dict]:
    tag = f"R4-{rid}-a{k}"
    lg = LOGS / f"{rid}.attempt{k}.log"
    base = [sys.executable, str(INFRA_PY), "run", "--tag", tag, "--out-dir", str(INFRA), "--interval", "10"]
    child = ["--", sys.executable, str(HERE / "run_one_r4.py"), "--run-id", rid]
    log(f"start {rid} attempt {k} -> {lg.relative_to(REPO)}")
    with open(lg, "w") as fh:
        rc = subprocess.call(base + ["--wait-exclusive", "--lock-timeout", str(EXCLUSIVE_WAIT_S)] + child,
                             stdout=fh, stderr=subprocess.STDOUT, cwd=str(REPO))
    if rc == 75:  # GPU not project-exclusive within the wait: proceed holding the lock only (prereg I1)
        log(f"{rid} attempt {k}: no exclusive GPU within {EXCLUSIVE_WAIT_S:.0f} s; running with the lock only")
        with open(lg, "a") as fh:
            fh.write(f"\n[r4-queue] exclusive wait timed out; relaunch with lock only\n")
            fh.flush()
            rc = subprocess.call(base + child, stdout=fh, stderr=subprocess.STDOUT, cwd=str(REPO))
    info_path = INFRA / f"{tag}.infra.json"
    info = json.loads(info_path.read_text()) if info_path.exists() else {}
    info.pop("samples", None)
    log(f"end {rid} attempt {k} rc={rc} flagged={info.get('flagged')} events={len(info.get('infra_events', []))} "
        f"concurrent={info.get('gpu', {}).get('compute_apps_seen')}")
    return rc, info


def run(rid: str) -> str:
    """Returns 'done', 'failed' or 'o4'."""
    flagged_any = False
    repeats = len(list(RUNS.glob(f"{rid}.flagged*.json")))
    attempts = 0
    while attempts < MAX_ATTEMPTS:
        if _done(rid) and not flagged_any:
            return "done"
        if _done(rid) and flagged_any:
            if repeats >= MAX_FULL_REPEATS:
                log(f"{rid}: flagged again after {repeats} full repeats; keeping result, reported as flagged")
                return "done"
            repeats += 1
            os.replace(RUNS / f"{rid}.json", RUNS / f"{rid}.flagged{repeats}.json")
            ck = CKPT_DIR / f"{rid}.pt"
            if ck.exists():
                ck.unlink()
            flagged_any = False
            log(f"{rid}: infra event in window -> repeating in full (plan §2.0), repeat {repeats}")
        k = _next_attempt(rid)
        rc, info = _attempt(rid, k)
        attempts += 1
        ev = info.get("infra_events", [])
        if ev:
            flagged_any = True
            if any(e.get("kind") in ("xid", "rc_watchdog") for e in ev):
                log(f"{rid}: Xid / RC-watchdog event in window -> plan §2.2 trigger O4; stopping the queue")
                return "o4"
        if rc == 0 and _done(rid) and not flagged_any:
            return "done"
    if _done(rid):
        return "done"
    log(f"FAILED {rid} after {MAX_ATTEMPTS} attempts")
    return "failed"


def main() -> None:
    for d in (RUNS, LOGS, INFRA):
        d.mkdir(parents=True, exist_ok=True)
    ids = run_ids()
    only = [a for a in sys.argv[1:] if not a.startswith("-")]
    if only:
        ids = [i for i in ids if i in only]
    log(f"queue: {ids}")
    status = {}
    for rid in ids:
        status[rid] = run(rid)
        if status[rid] == "o4":
            break
    log(f"status: {status}")
    subprocess.call([sys.executable, str(HERE / "r4_analysis.py")], cwd=str(REPO))
    log("queue finished")


if __name__ == "__main__":
    main()

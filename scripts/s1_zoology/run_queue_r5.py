"""Sequential, resumable R5 queue (plan §2.5; prereg experiments/repro/R5/prereg.yaml).

Launch detached:  bash scripts/repro/r5_launch.sh      (log: experiments/repro/R5/logs/queue.log)

* Order: our_mixers.r5_runs() (6 runs, interleaved by seed). After they are done, r5_analysis.py evaluates the
  pre-registered rules; iff it reports the bimodality rule (rule 4) fired, the 4 O5 runs (seeds 126, 127, both arms)
  are queued, then the analysis is re-run.
* A run is done when experiments/repro/R5/runs/<run_id>.json exists; re-launching skips it.
* Each attempt is a fresh subprocess wrapped by scripts/repro/infra_events.py (takes the single GPU lock
  scripts/repro/gpu_lock.py; tries --wait-exclusive for <= 10 min, then lock only), which writes
  experiments/repro/R5/infra/R5-<run_id>-a<k>.infra.json (kernel Xid/watchdog/OOM lines, GPU samples, concurrent
  processes). Failed attempts resume from the per-epoch checkpoint; at most MAX_ATTEMPTS attempts (plan §2.0).
* Plan §2.0: a gate run whose window contains an infra event is repeated in full (the flagged result is archived
  as <run_id>.flagged<k>.json). An Xid / RC-watchdog event stops the queue (plan §2.2 trigger O4).
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

import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
R5 = REPO / "experiments" / "repro" / "R5"
RUNS, LOGS, INFRA = R5 / "runs", R5 / "logs", R5 / "infra"
CKPT_DIR = REPO / "data" / "zoology" / "ckpt"
INFRA_PY = REPO / "scripts" / "repro" / "infra_events.py"
MAX_ATTEMPTS = int(os.environ.get("R5_MAX_ATTEMPTS", "6"))
MAX_FULL_REPEATS = int(os.environ.get("R5_MAX_FULL_REPEATS", "3"))
CONTINUE_ON_XID = os.environ.get("R5_CONTINUE_ON_XID", "0") == "1"
EXCLUSIVE_WAIT_S = float(os.environ.get("R5_EXCLUSIVE_WAIT_S", "600"))


def log(msg: str) -> None:
    print(f"[r5-queue] {time.strftime('%F %T')} {msg}", flush=True)


def run_ids(include_o5: bool = False) -> list[str]:
    from zoology_compat import setup_zoology

    setup_zoology(micro_batch=None)
    import our_mixers as OM

    return [c.run_id for c in OM.r5_runs(include_o5=include_o5)]


def _done(rid: str) -> bool:
    return (RUNS / f"{rid}.json").exists()


def _next_attempt(rid: str) -> int:
    return 1 + max((int(p.name.rsplit("-a", 1)[1].split(".")[0]) for p in INFRA.glob(f"R5-{rid}-a*.infra.json")),
                   default=0)


def _attempt(rid: str, k: int) -> tuple[int, dict]:
    tag = f"R5-{rid}-a{k}"
    lg = LOGS / f"{rid}.attempt{k}.log"
    base = [sys.executable, str(INFRA_PY), "run", "--tag", tag, "--out-dir", str(INFRA), "--interval", "10"]
    child = ["--", sys.executable, str(HERE / "run_one_r5.py"), "--run-id", rid]
    log(f"start {rid} attempt {k} -> {lg.relative_to(REPO)}")
    with open(lg, "w") as fh:
        rc = subprocess.call(base + ["--wait-exclusive", "--lock-timeout", str(EXCLUSIVE_WAIT_S)] + child,
                             stdout=fh, stderr=subprocess.STDOUT, cwd=str(REPO))
    if rc == 75:  # GPU not project-exclusive within the wait: proceed holding the lock only (prereg I1)
        log(f"{rid} attempt {k}: no exclusive GPU within {EXCLUSIVE_WAIT_S:.0f} s; running with the lock only")
        with open(lg, "a") as fh:
            fh.write(f"\n[r5-queue] exclusive wait timed out; relaunch with lock only\n")
            fh.flush()
            rc = subprocess.call(base + child, stdout=fh, stderr=subprocess.STDOUT, cwd=str(REPO))
    info_path = INFRA / f"{tag}.infra.json"
    info = json.loads(info_path.read_text()) if info_path.exists() else {}
    info.pop("samples", None)
    log(f"end {rid} attempt {k} rc={rc} flagged={info.get('flagged')} events={len(info.get('infra_events', []))} "
        f"concurrent={info.get('gpu', {}).get('compute_apps_seen')}")
    return rc, info


def _discard_for_full_repeat(rid: str, repeats: int) -> None:
    """Plan §2.0: a gate run whose window contains an infra event is repeated in full. Archive any result JSON
    of the flagged window as <rid>.flagged<k>.json and delete the per-epoch checkpoint so the repeat starts at
    epoch 0 (no resume across an infra event)."""
    if _done(rid):
        os.replace(RUNS / f"{rid}.json", RUNS / f"{rid}.flagged{repeats}.json")
    ck = CKPT_DIR / f"{rid}.pt"
    if ck.exists():
        os.replace(ck, CKPT_DIR / f"{rid}.flagged{repeats}.pt.discarded")


def run(rid: str) -> str:
    """Returns 'done', 'failed', 'flagged' or 'o4'.

    Infra failure without a kernel event (e.g. a Python crash) -> resume from the per-epoch checkpoint (§2.0).
    Any infra event (Xid / RC watchdog / OOM) in an attempt's window -> the run is repeated in full (§2.0). With
    R5_CONTINUE_ON_XID=1 (declared deviation from the R4 queue behaviour, see results_R5.md) the queue then
    continues; otherwise an Xid / RC-watchdog event stops the queue (plan §2.2 trigger O4)."""
    repeats = len(list(RUNS.glob(f"{rid}.flagged*.json"))) + len(list(CKPT_DIR.glob(f"{rid}.flagged*.pt.discarded")))
    attempts = 0
    while attempts < MAX_ATTEMPTS:
        if _done(rid):
            return "done"
        k = _next_attempt(rid)
        rc, info = _attempt(rid, k)
        attempts += 1
        ev = info.get("infra_events", [])
        if ev:
            xid = any(e.get("kind") in ("xid", "rc_watchdog") for e in ev)
            if repeats >= MAX_FULL_REPEATS:
                log(f"{rid}: infra event again after {repeats} full repeats; stopping this run (reported, not imputed)")
                return "flagged"
            repeats += 1
            _discard_for_full_repeat(rid, repeats)
            if xid and not CONTINUE_ON_XID:
                log(f"{rid}: Xid / RC-watchdog event in window -> plan §2.2 trigger O4; stopping the queue")
                return "o4"
            log(f"{rid}: infra event in window (xid/watchdog={xid}) -> repeating in full (plan §2.0), repeat {repeats}")
            continue
        if rc == 0 and _done(rid):
            return "done"
    if _done(rid):
        return "done"
    log(f"FAILED {rid} after {MAX_ATTEMPTS} attempts")
    return "failed"


def _queue(ids: list[str], status: dict) -> bool:
    """Run ids in order; False if the queue must stop (O4 trigger)."""
    for rid in ids:
        status[rid] = run(rid)
        if status[rid] == "o4":
            return False
    return True


def _analysis() -> dict:
    subprocess.call([sys.executable, str(HERE / "r5_analysis.py")], cwd=str(REPO))
    p = R5 / "r5_results.json"
    return json.loads(p.read_text()) if p.exists() else {}


def main() -> None:
    for d in (RUNS, LOGS, INFRA):
        d.mkdir(parents=True, exist_ok=True)
    ids = run_ids()
    only = [a for a in sys.argv[1:] if not a.startswith("-")]
    if only:
        ids = [i for i in run_ids(include_o5=True) if i in only]
    log(f"queue: {ids}")
    status: dict = {}
    ok = _queue(ids, status)
    log(f"status: {status}")
    res = _analysis()
    if ok and not only and res.get("rule4_bimodality", {}).get("fired"):
        o5 = [i for i in run_ids(include_o5=True) if i not in ids]
        log(f"rule 4 (bimodality) fired -> O5 runs (pre-declared): {o5}")
        ok = _queue(o5, status)
        log(f"status: {status}")
        res = _analysis()
    log(f"verdict: {res.get('verdict')}")
    log("queue finished")


if __name__ == "__main__":
    main()

"""Append-only run registry ``experiments/registry.jsonl`` (protocol §18).

Every line is one JSON record ``{ts, run_id, status, ...}``. Status changes are new lines; nothing is
ever rewritten. The current state of a run is its last record. Appends take an exclusive ``flock`` so
the runner, the trainer and the GPU monitor can write concurrently.

Statuses
--------
queued       enqueued by the runner (carries config path, allow_dirty, parent_run_id, config_hash)
running      trainer started / resumed (pid, start step)
completed    trainer finished (final metrics; runner adds contended_frac)
oom          G-OOM: CUDA OOM; retried with auto-resume
crashed      non-zero exit for any other reason; retried with auto-resume
interrupted  runner found a "running" job whose process is gone; retried
diverged     G-div event (informational; the run restarts once automatically)
failed       terminal: second divergence, 3 consecutive no-progress failures, refused start, ...
cancelled    terminal: removed from the queue by the user

PR-27 v2 (PI ruling D084) and R2-LIVE v2 (D077):
infra_failed_attempt  an attempt ended by an infrastructure event (Xid 8 + RC watchdog, Xid 79, host
                      OOM-kill); retried with auto-resume; never counts toward D031's failure count
suspect_sw_failure    an attempt ended with a suspected software fault (Xid 13/31/43, CUDA illegal
                      address/instruction, misaligned address, unknown Xid); counts as a failure; one resume
paused                the queue stops scheduling the run for the PI (10 infra attempts, or a suspect
                      failure recurring at the same optimizer step); never auto-failed
unpaused              the PI released a paused run (retryable again; infra count restarts)
attempt_end           annotation written by the runner after every attempt (infra block, exit code, last
                      completed optimizer step, attempt number), including attempts killed by a signal
"""
from __future__ import annotations

import fcntl
import json
import os
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterator

TERMINAL = {"completed", "failed", "cancelled"}
RETRYABLE = {"queued", "oom", "crashed", "interrupted", "infra_failed_attempt", "suspect_sw_failure", "unpaused"}
#: Neither terminal nor retryable: the run waits for a PI decision (PR-27 v2).
HELD = {"paused"}
# Records that annotate a run without changing its queue state.
ANNOTATIONS = {"diverged", "restarted", "throughput_alert", "note", "checkpoint", "monitor", "attempt_end"}


class Registry:
    def __init__(self, path: str | os.PathLike):
        self.path = Path(path)

    def append(self, run_id: str, status: str, **fields: Any) -> dict[str, Any]:
        rec = {"ts": time.time(), "iso": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "run_id": run_id,
               "status": status, **fields}
        line = json.dumps(rec, default=str, sort_keys=False) + "\n"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                f.write(line)
                f.flush()
                os.fsync(f.fileno())
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
        return rec

    def records(self) -> Iterator[dict[str, Any]]:
        if not self.path.exists():
            return
        with open(self.path) as f:
            fcntl.flock(f, fcntl.LOCK_SH)
            try:
                lines = f.readlines()
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
        for ln in lines:
            ln = ln.strip()
            if not ln:
                continue
            try:
                yield json.loads(ln)
            except json.JSONDecodeError:  # a torn last line cannot happen with flock+fsync, but be lenient
                continue

    def history(self, run_id: str) -> list[dict[str, Any]]:
        return [r for r in self.records() if r.get("run_id") == run_id]

    def state(self) -> "OrderedDict[str, dict[str, Any]]":
        """run_id -> merged view: last queue-state record, plus the queue record's fields.

        Insertion order is the order in which runs were first enqueued (FIFO queue order).
        """
        out: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
        for r in self.records():
            rid = r.get("run_id")
            if rid is None:
                continue
            cur = out.setdefault(rid, {"run_id": rid, "first_ts": r["ts"], "queue": None, "status": None,
                                       "last": None, "n_records": 0})
            cur["n_records"] += 1
            if r["status"] == "queued":
                cur["queue"] = r
            if r["status"] not in ANNOTATIONS:
                cur["status"] = r["status"]
                cur["last"] = r
        return out

    def status(self, run_id: str) -> str | None:
        st = self.state().get(run_id)
        return st["status"] if st else None

#!/usr/bin/env python
"""Single-GPU run queue with auto-resume (protocol §18). The queue lives in experiments/registry.jsonl.

    python scripts/runner.py enqueue <run.yaml> [<run.yaml> ...] [--allow-dirty] [-o key=value] [--tag T]
    python scripts/runner.py start            # detach: systemd-run --user if linger is on, else nohup
    python scripts/runner.py run [--forever]  # foreground loop (what `start` runs)
    python scripts/runner.py status
    python scripts/runner.py stop             # SIGTERM the detached runner (the job resumes later)
    python scripts/runner.py cancel <run_id>
    python scripts/runner.py unpause <run_id> [--reason TEXT]   # PI release of a paused run (PR-27 v2)
    python scripts/runner.py summary          # per-run / per-class / per-k attempt counts (JSON)

GPU lock (R2-LIVE v2, D077): for every CUDA job the runner acquires the project GPU lock
(``hybridlc.utils.gpu_lock``) before the launch and holds it until the child exits; the child inherits
the token, so ``scripts/train.py``'s guard passes. ``--wait-exclusive`` adds the exclusive-GPU wait.
Each attempt ends with an ``attempt_end`` registry row carrying the infra block; classified attempts
follow PR-27 v2 (see ``hybridlc.training.queue``).

Each job runs ``scripts/train.py --config ... --resume`` with its output appended to
``runs/<run_id>/train.log`` and a GPU monitor writing ``runs/<run_id>/gpu.jsonl`` (+ gpu_summary.json,
whose ``contended_frac`` is copied into the registry when the job ends).
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from hybridlc.training.config import PathsConfig  # noqa: E402
from hybridlc.training.queue import Runner, RunnerConfig, linger_enabled  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--registry", default=None)
    ap.add_argument("--runs-root", default=None)
    ap.add_argument("--ckpt-root", default=None)
    ap.add_argument("--min-free-gb", type=float, default=50.0)
    ap.add_argument("--poll", type=float, default=30.0)
    ap.add_argument("--monitor-interval", type=float, default=30.0)
    ap.add_argument("--no-monitor", action="store_true")
    ap.add_argument("--wait-exclusive", action="store_true", help="per-job lock waits for a project-free GPU")
    ap.add_argument("--max-infra-attempts", type=int, default=10)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("enqueue")
    e.add_argument("configs", nargs="+")
    e.add_argument("--allow-dirty", action="store_true")
    e.add_argument("-o", "--override", action="append", default=[])
    e.add_argument("--tag", default=None)
    e.add_argument("--force", action="store_true", help="re-enqueue a completed/terminal run")
    r = sub.add_parser("run")
    r.add_argument("--forever", action="store_true", help="keep polling when the queue is empty")
    r.add_argument("--max-jobs", type=int, default=None)
    sub.add_parser("start")
    sub.add_parser("stop")
    sub.add_parser("status")
    c = sub.add_parser("cancel")
    c.add_argument("run_id")
    u = sub.add_parser("unpause")
    u.add_argument("run_id")
    u.add_argument("--reason", default="PI decision")
    sub.add_parser("summary")
    return ap


def make_runner(a: argparse.Namespace) -> Runner:
    d = PathsConfig()
    paths = PathsConfig(runs_root=a.runs_root or d.runs_root, ckpt_root=a.ckpt_root or d.ckpt_root,
                        registry=a.registry or d.registry)
    return Runner(RunnerConfig(paths=paths, poll=a.poll, min_free_gb=a.min_free_gb,
                               monitor_interval=a.monitor_interval, monitor=not a.no_monitor,
                               wait_exclusive=a.wait_exclusive, max_infra_attempts=a.max_infra_attempts))


def _pidfile(runner: Runner) -> Path:
    return runner.rc.paths.resolve(runner.rc.paths.runs_root) / "runner.pid"


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    runner = make_runner(a)
    if a.cmd == "enqueue":
        for cfg in a.configs:
            runner.enqueue(cfg, overrides=a.override, allow_dirty=a.allow_dirty, tag=a.tag, force=a.force)
        return 0
    if a.cmd == "status":
        print(runner.status_table())
        return 0
    if a.cmd == "cancel":
        runner.cancel(a.run_id)
        return 0
    if a.cmd == "unpause":
        runner.unpause(a.run_id, reason=a.reason)
        return 0
    if a.cmd == "summary":
        import json

        print(json.dumps(runner.failure_counts(), indent=1, default=str))
        return 0
    if a.cmd == "run":
        import fcntl

        lock_path = runner.rc.paths.resolve(runner.rc.paths.runs_root) / "runner.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock = open(lock_path, "w")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("another runner holds runs/runner.lock; exiting", file=sys.stderr)
            return 1
        _pidfile(runner).write_text(str(os.getpid()))
        n = runner.loop(forever=a.forever, max_jobs=a.max_jobs)
        print(f"[runner] exiting after {n} job(s); pending: {runner.pending()}", flush=True)
        return 0
    if a.cmd == "start":
        base = [sys.executable, str(Path(__file__).resolve())]
        for flag, val in (("--registry", a.registry), ("--runs-root", a.runs_root), ("--ckpt-root", a.ckpt_root)):
            if val:
                base += [flag, val]
        base += ["--min-free-gb", str(a.min_free_gb), "--poll", str(a.poll), "run", "--forever"]
        log = runner.rc.paths.resolve(runner.rc.paths.runs_root) / "runner.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        if linger_enabled():
            cmd = ["systemd-run", "--user", "--unit", "hybridlc-runner", "--same-dir",
                   "-p", f"StandardOutput=append:{log}", "-p", f"StandardError=append:{log}", *base]
            subprocess.run(cmd, check=True, cwd=str(REPO))
            print(f"started systemd user unit hybridlc-runner (log {log})")
        else:
            with open(log, "a") as f:
                p = subprocess.Popen(["nohup", *base], stdout=f, stderr=subprocess.STDOUT, cwd=str(REPO),
                                     start_new_session=True)
            print(f"started runner under nohup, pid {p.pid} (log {log})")
        return 0
    if a.cmd == "stop":
        pf = _pidfile(runner)
        if not pf.exists():
            print("no runner.pid")
            return 1
        pid = int(pf.read_text())
        try:
            os.kill(pid, signal.SIGTERM)
            print(f"sent SIGTERM to runner {pid}")
        except ProcessLookupError:
            print(f"runner {pid} not running")
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())

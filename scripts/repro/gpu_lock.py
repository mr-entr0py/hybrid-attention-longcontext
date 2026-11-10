#!/usr/bin/env python
"""Single GPU queue for project jobs (reproduction plan §2.2 R2(a)).

Every project GPU job acquires one exclusive lock before it touches the GPU and holds it until it
exits, so no two project GPU processes run at once. The lock is an ``fcntl.flock`` on
``runs/gpu_queue.lock`` (override: env ``HYBRIDLC_GPU_LOCK``). The kernel drops a flock when its
holder dies, so a crashed job never leaves a stale lock. The current holder is described in
``<lock>.holder.json`` (informational only; the flock is the source of truth).

Queued jobs wait in arrival order only approximately (flock is not FIFO); a job waiting with
``--timeout`` gives up with exit code 75 after that many seconds.

The lock only serialises jobs that use it. Agents that start GPU work without it (for example the
S0 runner's own queue) are visible through ``--wait-exclusive``: the job also waits until no other
process of this repository's venv holds a CUDA context on the GPU (CARLA is foreign, not project,
load and is ignored by that check).

Usage::

    python scripts/repro/gpu_lock.py run --label R2c-profile [--timeout 3600] [--wait-exclusive] -- \
        python scripts/repro/r2_kernel_profile.py ...
    python scripts/repro/gpu_lock.py status          # holder, lock state, project GPU processes

    from gpu_lock import gpu_lock                    # in-process
    with gpu_lock("R3-vt21", timeout=3600, wait_exclusive=True) as info: ...
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterator

REPO = Path(__file__).resolve().parents[2]
DEFAULT_LOCK = REPO / "runs" / "gpu_queue.lock"
VENV_MARKER = str(REPO / ".venv")
EXIT_TIMEOUT = 75


def lock_path() -> Path:
    return Path(os.environ.get("HYBRIDLC_GPU_LOCK", str(DEFAULT_LOCK)))


def holder_path(lp: Path | None = None) -> Path:
    lp = lp or lock_path()
    return lp.with_name(lp.name + ".holder.json")


def project_gpu_processes(exclude: set[int] | None = None) -> list[dict[str, Any]]:
    """Processes with a CUDA context whose executable lives in this repository's venv (or whose
    command line mentions the repository), from ``nvidia-smi --query-compute-apps``."""
    exclude = exclude or set()
    try:
        r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20)
    except Exception as e:  # pragma: no cover
        return [{"error": f"{type(e).__name__}: {e}"}]
    out = []
    for line in r.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        if pid in exclude:
            continue
        try:
            cmd = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except OSError:
            cmd = parts[1]
        try:  # `python -` heredocs launched from the repo show neither path in name/cmdline
            cwd = os.readlink(f"/proc/{pid}/cwd")
        except OSError:
            cwd = ""
        try:
            env = Path(f"/proc/{pid}/environ").read_bytes()
            venv_env = f"VIRTUAL_ENV={VENV_MARKER}".encode() in env
        except OSError:
            venv_env = False
        if (VENV_MARKER in parts[1] or VENV_MARKER in cmd or str(REPO) in cmd
                or cwd == str(REPO) or cwd.startswith(str(REPO) + "/") or venv_env):
            out.append({"pid": pid, "name": parts[1], "used_mib": parts[2], "cmd": cmd[:300]})
    return out


def _own_tree() -> set[int]:
    """This process, its ancestors and its descendants (they are 'us', not foreign jobs)."""
    own, p = set(), os.getpid()
    while p > 1:
        own.add(p)
        try:
            p = int(Path(f"/proc/{p}/stat").read_text().rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            break
    todo = [os.getpid()]
    while todo:
        q = todo.pop()
        for task in Path(f"/proc/{q}/task").glob("*"):
            try:
                kids = [int(c) for c in (task / "children").read_text().split()]
            except (OSError, ValueError):
                kids = []
            for k in kids:
                if k not in own:
                    own.add(k)
                    todo.append(k)
    return own


@contextlib.contextmanager
def gpu_lock(label: str, timeout: float | None = None, poll_s: float = 2.0, wait_exclusive: bool = False,
             exclusive_settle_s: float = 0.0, verbose: bool = True) -> Iterator[dict[str, Any]]:
    """Hold the project GPU lock for the duration of the ``with`` block.

    Yields an info dict (``label, wait_s, acquired_at, waited_for``). Raises ``TimeoutError`` if the
    lock (and, with ``wait_exclusive``, an otherwise project-free GPU) is not obtained in ``timeout`` s.
    """
    lp = lock_path()
    lp.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lp, os.O_RDWR | os.O_CREAT, 0o664)
    t0 = time.time()
    waited_for: list[str] = []
    last_note = 0.0
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if timeout is not None and time.time() - t0 > timeout:
                    raise TimeoutError(f"gpu_lock: '{label}' did not get {lp} within {timeout:.0f} s")
                if time.time() - last_note > 60:
                    h = read_holder(lp)
                    note = f"lock held by {h.get('label')} pid {h.get('pid')}" if h else "lock held"
                    waited_for.append(note)
                    if verbose:
                        print(f"[gpu_lock] {time.strftime('%H:%M:%S')} '{label}' waiting: {note}", file=sys.stderr, flush=True)
                    last_note = time.time()
                time.sleep(poll_s)
        # lock held; optionally also wait for other (lock-less) project GPU processes to finish
        if wait_exclusive:
            last_note = 0.0
            clear_since = None
            while True:
                others = project_gpu_processes(exclude=_own_tree())
                others = [o for o in others if "error" not in o]
                if not others:
                    clear_since = clear_since or time.time()
                    if time.time() - clear_since >= exclusive_settle_s:
                        break
                else:
                    clear_since = None
                    if time.time() - last_note > 60:
                        note = "project GPU processes without the lock: " + ", ".join(
                            f"{o['pid']}:{o['cmd'][:80]}" for o in others)
                        waited_for.append(note)
                        if verbose:
                            print(f"[gpu_lock] {time.strftime('%H:%M:%S')} '{label}' waiting: {note}", file=sys.stderr, flush=True)
                        last_note = time.time()
                if timeout is not None and time.time() - t0 > timeout:
                    raise TimeoutError(f"gpu_lock: '{label}' GPU not project-exclusive within {timeout:.0f} s")
                time.sleep(max(poll_s, 5.0))
        info = {"label": label, "pid": os.getpid(), "host": socket.gethostname(), "acquired_at": time.time(),
                "acquired_at_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "wait_s": time.time() - t0,
                "lock": str(lp), "wait_exclusive": wait_exclusive, "waited_for": waited_for[-10:]}
        hp = holder_path(lp)
        tmp = hp.with_suffix(".tmp")
        tmp.write_text(json.dumps(info, indent=1))
        os.replace(tmp, hp)
        if verbose:
            print(f"[gpu_lock] {time.strftime('%H:%M:%S')} '{label}' acquired after {info['wait_s']:.1f} s", file=sys.stderr, flush=True)
        try:
            yield info
        finally:
            with contextlib.suppress(OSError):
                if read_holder(lp).get("pid") == os.getpid():
                    hp.unlink()
            fcntl.flock(fd, fcntl.LOCK_UN)
            if verbose:
                print(f"[gpu_lock] {time.strftime('%H:%M:%S')} '{label}' released", file=sys.stderr, flush=True)
    finally:
        os.close(fd)


def read_holder(lp: Path | None = None) -> dict[str, Any]:
    try:
        return json.loads(holder_path(lp).read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def is_locked(lp: Path | None = None) -> bool:
    lp = lp or lock_path()
    if not lp.exists():
        return False
    fd = os.open(lp, os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    except BlockingIOError:
        return True
    finally:
        os.close(fd)


def _cmd_run(args: argparse.Namespace) -> int:
    cmd = args.cmd[1:] if args.cmd and args.cmd[0] == "--" else args.cmd
    if not cmd:
        print("gpu_lock run: no command given (use: run --label X -- cmd ...)", file=sys.stderr)
        return 2
    try:
        with gpu_lock(args.label, timeout=args.timeout, wait_exclusive=args.wait_exclusive,
                      exclusive_settle_s=args.settle_s):
            child = subprocess.Popen(cmd)

            def fwd(sig, _frm):  # forward termination so the child never outlives the lock
                with contextlib.suppress(ProcessLookupError):
                    child.send_signal(sig)
            for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
                signal.signal(s, fwd)
            return child.wait()
    except TimeoutError as e:
        print(str(e), file=sys.stderr)
        return EXIT_TIMEOUT


def _cmd_status(_args: argparse.Namespace) -> int:
    st = {"lock": str(lock_path()), "locked": is_locked(), "holder": read_holder(),
          "project_gpu_processes": project_gpu_processes(exclude=_own_tree())}
    print(json.dumps(st, indent=1))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="sub", required=True)
    r = sub.add_parser("run", help="acquire the lock, run a command, release")
    r.add_argument("--label", required=True)
    r.add_argument("--timeout", type=float, default=None)
    r.add_argument("--wait-exclusive", action="store_true",
                   help="also wait until no other project-venv process holds a CUDA context")
    r.add_argument("--settle-s", type=float, default=0.0, help="with --wait-exclusive: required clear time")
    r.add_argument("cmd", nargs=argparse.REMAINDER)
    r.set_defaults(fn=_cmd_run)
    s = sub.add_parser("status")
    s.set_defaults(fn=_cmd_status)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())

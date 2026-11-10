#!/usr/bin/env python
"""Single GPU queue for project jobs (reproduction plan §2.2 R2(a)) — thin CLI.

The logic lives in ``hybridlc.utils.gpu_lock`` (R2-LIVE v2 L1, PI ruling D077): one ``fcntl.flock`` on
``runs/gpu_queue.lock``; the holder exports ``HYBRIDLC_GPU_LOCK_TOKEN`` so the command it runs (and that
command's children) pass every entry-point guard; ``--wait-exclusive`` also waits until no other
project process holds a CUDA context. While held, a 5-s sampler records ``lock_ok`` per project CUDA
process under ``runs/_logs/gpumon_lock/``.

Usage::

    python scripts/repro/gpu_lock.py run --label R2c-profile [--timeout 3600] [--wait-exclusive] -- \
        python scripts/repro/r2_kernel_profile.py ...
    python scripts/repro/gpu_lock.py status          # holder, lock state, project GPU processes

    from gpu_lock import gpu_lock                    # in-process (re-exported)
    with gpu_lock("R3-vt21", timeout=3600, wait_exclusive=True) as info: ...
"""
from __future__ import annotations

import argparse
import contextlib
import json
import signal
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from hybridlc.utils.gpu_lock import (EXIT_NO_LOCK, EXIT_TIMEOUT, gpu_lock, guard, holder_path, is_locked,  # noqa: E402,F401
                                     lock_path, project_gpu_processes, read_holder, status, verify_token)

DEFAULT_LOCK = REPO / "runs" / "gpu_queue.lock"


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
    print(json.dumps(status(), indent=1))
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

#!/usr/bin/env python
"""Per-run infrastructure-event tagging (reproduction plan §2.2 R2(b)) — thin CLI over hybridlc.utils.infra.

For a run window [start, end] this collects

* ``infra_events``: kernel-log lines matching ``NVRM: Xid|RC watchdog|Out of memory|oom-kill`` (the
  plan's pattern), read with ``journalctl -k`` (works without sudo for members of group ``adm`` /
  ``systemd-journal``; ``dmesg`` is not readable here: ``kernel.dmesg_restrict``), plus
  ``systemd-oomd`` kill lines from the system journal (user-space host-OOM killer);
* GPU samples every ``interval`` s: utilisation, memory, SM clock, temperature, power,
  ``clocks_event_reasons.*`` (active bitmask and each flag), and the compute processes;
* the ``nvidia-smi -q -d PERFORMANCE`` clock-event *counters* (µs) at start and end, and their delta;
* host ``MemAvailable`` (min over samples).

A run is ``flagged`` iff it has any ``infra_events`` line. ``journal_ok`` records whether the kernel
journal was readable and contains NVRM lines (if not, an empty event list means "unknown", not "clean").

Use::

    # wrap any command (also takes the single-queue lock unless --no-lock)
    python scripts/repro/infra_events.py run --tag R2c-D384-C12 --out-dir experiments/repro/r2/runs -- python ...
    # scan a past window
    python scripts/repro/infra_events.py scan --since "2026-11-09 17:30" --until "2026-11-09 18:30"

    from infra_events import InfraTagger
    with InfraTagger("experiments/repro/r2/runs/x.infra.json", tag="x") as tagger: ...
"""
from __future__ import annotations

import argparse
import contextlib
import json
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# The logic moved to hybridlc.utils.infra (R2-LIVE v2 L2, PI ruling D077); re-exported for old callers.
from hybridlc.utils.infra import (GRACE_S, KLOG_PATTERN, OOMD_PATTERN, XID_RE, InfraTagger,  # noqa: E402,F401
                                  classify_events, journal_status, kernel_events, parse_line, perf_counters,
                                  query_gpu, registry_fields)
from hybridlc.utils.infra import run_meta as _run_meta  # noqa: E402


def run_meta() -> dict[str, Any]:
    return _run_meta(REPO)


def _cmd_run(args: argparse.Namespace) -> int:
    cmd = args.cmd[1:] if args.cmd and args.cmd[0] == "--" else args.cmd
    if not cmd:
        print("infra_events run: no command", file=sys.stderr)
        return 2
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / f"{args.tag}.infra.json"
    lock_ctx: Any = contextlib.nullcontext({"label": args.tag, "wait_s": 0.0, "lock": None})
    if not args.no_lock:
        from hybridlc.utils.gpu_lock import gpu_lock
        lock_ctx = gpu_lock(args.tag, timeout=args.lock_timeout, wait_exclusive=args.wait_exclusive,
                            exclusive_settle_s=args.settle_s)
    try:
        with lock_ctx as linfo:
            tagger = InfraTagger(meta_path, tag=args.tag, interval=args.interval,
                                 extra={"cmd": cmd, "meta": run_meta(), "lock": linfo})
            with tagger:
                child = subprocess.Popen(cmd)

                def fwd(sig, _frm):
                    with contextlib.suppress(ProcessLookupError):
                        child.send_signal(sig)
                for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
                    signal.signal(s, fwd)
                rc = child.wait()
                tagger.extra["returncode"] = rc
            res = tagger.result
    except TimeoutError as e:
        print(str(e), file=sys.stderr)
        return 75
    print(f"[infra] {args.tag}: rc={rc} flagged={res['flagged']} class={res['classification']} "
          f"events={len(res['infra_events'])} journal_ok={res['journal_ok']} "
          f"slowdown={res['gpu']['slowdown_flag_samples']} -> {meta_path}", flush=True)
    return rc


def _cmd_scan(args: argparse.Namespace) -> int:
    def to_epoch(s: str) -> float:
        return time.mktime(time.strptime(s, "%Y-%m-%d %H:%M:%S" if s.count(":") == 2 else "%Y-%m-%d %H:%M"))
    res = kernel_events(to_epoch(args.since), to_epoch(args.until))
    res["journal"] = journal_status()
    res["window"] = [args.since, args.until]
    txt = json.dumps(res, indent=1)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(txt)
    print(txt)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="sub", required=True)
    r = sub.add_parser("run")
    r.add_argument("--tag", required=True)
    r.add_argument("--out-dir", default=str(REPO / "experiments" / "repro" / "infra"))
    r.add_argument("--interval", type=float, default=5.0)
    r.add_argument("--no-lock", action="store_true")
    r.add_argument("--wait-exclusive", action="store_true")
    r.add_argument("--lock-timeout", type=float, default=None)
    r.add_argument("--settle-s", type=float, default=0.0, help="with --wait-exclusive: required project-free time")
    r.add_argument("cmd", nargs=argparse.REMAINDER)
    r.set_defaults(fn=_cmd_run)
    s = sub.add_parser("scan")
    s.add_argument("--since", required=True)
    s.add_argument("--until", required=True)
    s.add_argument("--out", default=None)
    s.set_defaults(fn=_cmd_scan)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())

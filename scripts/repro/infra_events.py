#!/usr/bin/env python
"""Per-run infrastructure-event tagging (reproduction plan §2.2 R2(b)).

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
import os
import platform
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

KLOG_PATTERN = re.compile(r"NVRM: Xid|RC watchdog|Out of memory|oom-kill")
OOMD_PATTERN = re.compile(r"[Kk]illed|oom")
XID_RE = re.compile(r"Xid \(PCI:[^)]*\): (\d+), pid=(\d+|'<unknown>'), name=([^,]*)")
GRACE_S = 3.0  # kernel messages can lag the event slightly

GPU_FIELDS = [
    "utilization.gpu", "memory.used", "memory.total", "clocks.sm", "clocks.max.sm", "temperature.gpu", "power.draw",
    "clocks_event_reasons.active", "clocks_event_reasons.gpu_idle", "clocks_event_reasons.applications_clocks_setting",
    "clocks_event_reasons.sw_power_cap", "clocks_event_reasons.hw_slowdown",
    "clocks_event_reasons.hw_thermal_slowdown", "clocks_event_reasons.hw_power_brake_slowdown",
    "clocks_event_reasons.sw_thermal_slowdown", "clocks_event_reasons.sync_boost", "display_active",
]
SLOWDOWN_FLAGS = ["clocks_event_reasons.sw_power_cap", "clocks_event_reasons.hw_slowdown",
                  "clocks_event_reasons.hw_thermal_slowdown", "clocks_event_reasons.hw_power_brake_slowdown",
                  "clocks_event_reasons.sw_thermal_slowdown"]


def _iso(t: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t))


def _journal(args: list[str]) -> tuple[bool, list[str], str]:
    try:
        r = subprocess.run(["journalctl", "--no-pager", "-o", "short-iso-precise", *args],
                           capture_output=True, text=True, timeout=60)
    except Exception as e:
        return False, [], f"{type(e).__name__}: {e}"
    ok = r.returncode == 0
    return ok, r.stdout.splitlines(), r.stderr.strip()[:300]


def journal_status() -> dict[str, Any]:
    """Is the kernel journal readable, and does it contain NVRM lines for the current boot?"""
    ok, lines, err = _journal(["-k", "-b", "0", "--grep", "NVRM", "-n", "3"])
    try:
        dm = subprocess.run(["dmesg", "-T"], capture_output=True, text=True, timeout=10)
        dmesg_ok = dm.returncode == 0
        dmesg_err = dm.stderr.strip()[:200]
    except Exception as e:  # pragma: no cover
        dmesg_ok, dmesg_err = False, str(e)
    groups = []
    with contextlib.suppress(Exception):
        groups = subprocess.run(["id", "-Gn"], capture_output=True, text=True, timeout=5).stdout.split()
    return {"journalctl_k_ok": ok and bool(lines), "journalctl_err": err, "nvrm_lines_visible": len(lines),
            "dmesg_ok": dmesg_ok, "dmesg_err": dmesg_err, "groups": groups}


def parse_line(line: str) -> dict[str, Any]:
    rec: dict[str, Any] = {"line": line, "ts": line.split(" ", 1)[0]}
    m = XID_RE.search(line)
    if m:
        rec.update(kind="xid", xid=int(m.group(1)), pid=m.group(2), name=m.group(3).strip())
    elif "RC watchdog" in line:
        rec["kind"] = "rc_watchdog"
    elif "oom-kill" in line or "Out of memory" in line:
        rec["kind"] = "host_oom"
    elif "systemd-oomd" in line:
        rec["kind"] = "systemd_oomd"
    else:
        rec["kind"] = "other"
    return rec


def kernel_events(since: float, until: float) -> dict[str, Any]:
    """Matching kernel-log lines (and systemd-oomd kills) in [since, until + GRACE_S]."""
    s, u = f"@{since:.3f}", f"@{until + GRACE_S:.3f}"
    ok, lines, err = _journal(["-k", "--since", s, "--until", u])
    ev = [parse_line(x) for x in lines if KLOG_PATTERN.search(x)]
    ok2, lines2, err2 = _journal(["-u", "systemd-oomd", "--since", s, "--until", u])
    ev += [parse_line(x) for x in lines2 if "systemd-oomd" in x and OOMD_PATTERN.search(x) and "Started" not in x]
    return {"journal_ok": ok, "journal_err": err if not ok else "", "oomd_journal_ok": ok2, "events": ev}


def query_gpu() -> dict[str, Any]:
    out: dict[str, Any] = {"ts": time.time()}
    try:
        r = subprocess.run(["nvidia-smi", f"--query-gpu={','.join(GPU_FIELDS)}", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=20, check=True)
        for k, v in zip(GPU_FIELDS, r.stdout.strip().splitlines()[0].split(",")):
            v = v.strip()
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = v
        r2 = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20)
        out["compute_apps"] = [ln.strip() for ln in r2.stdout.strip().splitlines() if ln.strip()]
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
    with contextlib.suppress(OSError, ValueError):
        for ln in Path("/proc/meminfo").read_text().splitlines():
            if ln.startswith("MemAvailable:"):
                out["host_mem_available_gib"] = int(ln.split()[1]) / 2**20
    return out


def perf_counters() -> dict[str, float]:
    """Clock-event-reason counters (µs since driver load) from ``nvidia-smi -q -d PERFORMANCE``."""
    out: dict[str, float] = {}
    try:
        r = subprocess.run(["nvidia-smi", "-q", "-d", "PERFORMANCE"], capture_output=True, text=True, timeout=20)
    except Exception:
        return out
    in_counters = False
    for ln in r.stdout.splitlines():
        if "Clocks Event Reasons Counters" in ln:
            in_counters = True
            continue
        if in_counters:
            m = re.match(r"\s+(.+?)\s+:\s+(\d+) us", ln)
            if m:
                out[m.group(1).strip()] = float(m.group(2))
            elif ln.strip() and not ln.startswith("        "):
                in_counters = False
    return out


class InfraTagger:
    """Context manager: GPU sampling thread + kernel-journal scan over the block's window."""

    def __init__(self, out_json: str | os.PathLike | None = None, tag: str = "", interval: float = 5.0,
                 extra: dict[str, Any] | None = None):
        self.out_json = Path(out_json) if out_json else None
        self.tag, self.interval, self.extra = tag, interval, extra or {}
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self.result: dict[str, Any] = {}

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.samples.append(query_gpu())
            self._stop.wait(self.interval)

    def __enter__(self) -> "InfraTagger":
        self.t_start = time.time()
        self.counters_start = perf_counters()
        self.journal = journal_status()
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()
        return self

    def finish(self) -> dict[str, Any]:
        self._stop.set()
        self._th.join(timeout=30)
        self.samples.append(query_gpu())
        t_end = time.time()
        time.sleep(1.0)  # let late kernel messages land; the scan itself adds GRACE_S
        ke = kernel_events(self.t_start, t_end)
        c_end = perf_counters()
        good = [s for s in self.samples if "error" not in s]
        slow = {f: sum(1 for s in good if str(s.get(f, "")).lower() == "active") for f in SLOWDOWN_FLAGS}
        actives = sorted({str(s.get("clocks_event_reasons.active")) for s in good})
        apps = sorted({a for s in good for a in s.get("compute_apps", [])})
        mem = [s["host_mem_available_gib"] for s in good if "host_mem_available_gib" in s]
        self.result = {
            "tag": self.tag, "start": self.t_start, "end": t_end, "start_iso": _iso(self.t_start), "end_iso": _iso(t_end),
            "journal": self.journal, "journal_ok": ke["journal_ok"] and self.journal["journalctl_k_ok"],
            "infra_events": ke["events"], "flagged": bool(ke["events"]),
            "gpu": {"n_samples": len(good), "n_errors": len(self.samples) - len(good),
                    "interval_s": self.interval,
                    "clocks_event_reasons_active_values": actives,
                    "slowdown_flag_samples": slow,
                    "any_slowdown": any(slow.values()),
                    "max_temp_c": max((s.get("temperature.gpu") or 0) for s in good) if good else None,
                    "max_power_w": max((s.get("power.draw") or 0) for s in good) if good else None,
                    "min_sm_clock_mhz": min((s.get("clocks.sm") or 0) for s in good) if good else None,
                    "max_util_pct": max((s.get("utilization.gpu") or 0) for s in good) if good else None,
                    "display_active": sorted({str(s.get("display_active")) for s in good}),
                    "compute_apps_seen": apps},
            "clock_event_counters_us": {"start": self.counters_start, "end": c_end,
                                        "delta": {k: c_end.get(k, 0) - v for k, v in self.counters_start.items()}},
            "host": {"min_mem_available_gib": min(mem) if mem else None},
            **self.extra,
        }
        if self.out_json:
            self.out_json.parent.mkdir(parents=True, exist_ok=True)
            self.out_json.write_text(json.dumps({**self.result, "samples": self.samples}, indent=1))
        return self.result

    def __exit__(self, *exc) -> None:
        self.finish()


def run_meta() -> dict[str, Any]:
    def git(*a: str) -> str:
        try:
            return subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True, timeout=20).stdout.strip()
        except Exception:
            return ""
    drv = ""
    with contextlib.suppress(Exception):
        drv = subprocess.run(["nvidia-smi", "--query-gpu=driver_version,name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20).stdout.strip()
    return {"git_commit": git("rev-parse", "HEAD"), "git_dirty_files": len(git("status", "--porcelain").splitlines()),
            "python": platform.python_version(), "platform": platform.platform(), "driver_gpu": drv,
            "hostname": platform.node()}


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
        from gpu_lock import gpu_lock
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
    print(f"[infra] {args.tag}: rc={rc} flagged={res['flagged']} events={len(res['infra_events'])} "
          f"journal_ok={res['journal_ok']} slowdown={res['gpu']['slowdown_flag_samples']} -> {meta_path}", flush=True)
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

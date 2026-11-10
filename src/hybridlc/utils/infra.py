"""Per-run infrastructure-event tagging and the PR-27 v2 classifier (plan §2.2 R2(b); R2-LIVE v2 L2;
PI rulings D077, D084).

For a run window ``[start, end]`` this collects

* kernel-log lines matching ``NVRM: Xid|RC watchdog|Out of memory|oom-kill`` (``journalctl -k``) plus
  ``systemd-oomd`` kill lines (user-space host-OOM killer);
* GPU samples every ``interval`` s (utilisation, memory, clocks, throttle flags, compute processes),
  with ``lock_ok`` per project CUDA process (holder or descendant of the GPU-lock holder);
* clock-event counters at start and end, and host ``MemAvailable``.

PR-27 v2 classes (per event and per attempt; ``classify_events``):

* ``infra``: Xid 8 with an RC-watchdog line in the same window (or a ``cudaErrorLaunchTimeout`` /
  "launch timed out" CUDA error matched to such a line), Xid 79, host OOM-kill (kernel or
  systemd-oomd). The RC-watchdog line itself is classed ``infra`` as part of that pair.
* ``suspect_sw``: Xid 13, 31, 43; CUDA illegal address / illegal instruction / misaligned address
  without an RC-watchdog line; Xid 8 without an RC-watchdog line; any other Xid.
* ``none``: no event.

Per attempt the class is ``suspect_sw`` if any event is ``suspect_sw``, else ``infra`` if any event is
``infra``, else ``none`` (a suspected software fault is never hidden by a concurrent infra event).

Test hook (R2-LIVE C1, VT-13r, FEAS-SMOKE): env ``HYBRIDLC_JOURNAL_SOURCE`` replaces the system
journal: ``:unreadable:`` (journal_ok = false), ``:empty:`` (readable, no lines), or the path of a file
whose lines are read as journal lines (lines without an ISO timestamp are stamped with the read time).
Every block records ``journal_source`` so an injected window is never mistaken for a real one.
"""
from __future__ import annotations

import contextlib
import json
import os
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable

KLOG_PATTERN = re.compile(r"NVRM: Xid|RC watchdog|Out of memory|oom-kill")
OOMD_PATTERN = re.compile(r"[Kk]illed|oom")
XID_RE = re.compile(r"Xid \(PCI:[^)]*\): (\d+), pid=(\d+|'<unknown>'), name=([^,]*)")
CUDA_FAULT_RE = re.compile(r"illegal memory access|an illegal instruction was encountered|misaligned address|"
                           r"cudaErrorIllegalAddress|cudaErrorIllegalInstruction|cudaErrorMisalignedAddress", re.I)
CUDA_TIMEOUT_RE = re.compile(r"cudaErrorLaunchTimeout|the launch timed out and was terminated", re.I)
GRACE_S = 3.0
JOURNAL_ENV = "HYBRIDLC_JOURNAL_SOURCE"
INFRA_XIDS = {79}
SUSPECT_XIDS = {13, 31, 43}
CLASSES = ("none", "infra", "suspect_sw")
#: registry statuses of the classified attempt outcomes (PR-27 v2)
STATUS_OF_CLASS = {"infra": "infra_failed_attempt", "suspect_sw": "suspect_sw_failure"}

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
#: Foreign compute clients allowed during gate/timed runs (R2-LIVE v2 C2): display/desktop and the idle CARLA env.
ALLOWED_FOREIGN = ("gnome-remote-desktop", "Xorg", "gnome-shell", "carla_env/bin/python")


def _iso(t: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t))


# ------------------------------------------------------------------------------------------ journal
def journal_source() -> str:
    return os.environ.get(JOURNAL_ENV) or "system"


def _ts_of(line: str) -> float | None:
    tok = line.split(" ", 1)[0]
    try:
        from datetime import datetime

        return datetime.fromisoformat(tok).timestamp()
    except ValueError:
        return None


def _injected(args: list[str]) -> tuple[bool, list[str], str]:
    src = os.environ.get(JOURNAL_ENV, "")
    if src == ":unreadable:":
        return False, [], "injected: unreadable"
    if src == ":empty:":
        return True, [], ""
    try:
        raw = Path(src).read_text().splitlines()
    except OSError as e:
        return False, [], f"injected source unreadable: {e}"
    since = until = None
    if "--since" in args:
        since = float(args[args.index("--since") + 1].lstrip("@"))
    if "--until" in args:
        until = float(args[args.index("--until") + 1].lstrip("@"))
    want_oomd = "-u" in args and "systemd-oomd" in args
    out = []
    for ln in raw:
        if not ln.strip():
            continue
        if ("systemd-oomd" in ln) != want_oomd:
            continue
        t = _ts_of(ln)
        if t is None:
            out.append(f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} host kernel: {ln}")
            continue
        if (since is None or t >= since) and (until is None or t <= until):
            out.append(ln)
    if "--grep" in args:
        pat = args[args.index("--grep") + 1]
        out = [x for x in out if pat in x] or ([f"injected NVRM marker"] if pat == "NVRM" else [])
    return True, out, ""


def _journal(args: list[str]) -> tuple[bool, list[str], str]:
    if os.environ.get(JOURNAL_ENV):
        return _injected(args)
    try:
        r = subprocess.run(["journalctl", "--no-pager", "-o", "short-iso-precise", *args],
                           capture_output=True, text=True, timeout=60)
    except Exception as e:
        return False, [], f"{type(e).__name__}: {e}"
    return r.returncode == 0, r.stdout.splitlines(), r.stderr.strip()[:300]


def journal_status() -> dict[str, Any]:
    """Is the kernel journal readable, and does it contain NVRM lines for the current boot?"""
    ok, lines, err = _journal(["-k", "-b", "0", "--grep", "NVRM", "-n", "3"])
    return {"journalctl_k_ok": ok and (bool(lines) or os.environ.get(JOURNAL_ENV, "") not in ("",)),
            "journalctl_err": err, "nvrm_lines_visible": len(lines), "source": journal_source()}


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
    elif CUDA_TIMEOUT_RE.search(line):
        rec["kind"] = "cuda_launch_timeout"
    elif CUDA_FAULT_RE.search(line):
        rec["kind"] = "cuda_fault"
    else:
        rec["kind"] = "other"
    return rec


def kernel_events(since: float, until: float) -> dict[str, Any]:
    """Matching kernel-log lines (and systemd-oomd kills) in [since, until + GRACE_S]."""
    s, u = f"@{since:.3f}", f"@{until + GRACE_S:.3f}"
    ok, lines, err = _journal(["-k", "--since", s, "--until", u])
    ev = [parse_line(x) for x in lines if KLOG_PATTERN.search(x)]
    ok2, lines2, _err2 = _journal(["-u", "systemd-oomd", "--since", s, "--until", u])
    ev += [parse_line(x) for x in lines2 if "systemd-oomd" in x and OOMD_PATTERN.search(x) and "Started" not in x]
    return {"journal_ok": ok, "journal_err": err if not ok else "", "oomd_journal_ok": ok2, "events": ev,
            "journal_source": journal_source()}


# ------------------------------------------------------------------------------------------ classifier
def classify_events(events: list[dict[str, Any]], log_lines: list[str] | None = None) -> tuple[list[dict[str, Any]], str]:
    """PR-27 v2: (events with a ``class`` field, attempt class). ``log_lines`` are the attempt's own
    stderr/log lines; CUDA fault / launch-timeout errors found there are added as events."""
    evs = [dict(e) for e in events]
    for ln in log_lines or []:
        if CUDA_TIMEOUT_RE.search(ln) or CUDA_FAULT_RE.search(ln):
            evs.append(parse_line(ln) | {"source": "attempt_log"})
    rc = any(e.get("kind") == "rc_watchdog" for e in evs)
    for e in evs:
        k = e.get("kind")
        if k == "xid":
            x = int(e["xid"])
            if x == 8:
                e["class"] = "infra" if rc else "suspect_sw"
            elif x in INFRA_XIDS:
                e["class"] = "infra"
            else:  # 13, 31, 43 and every other Xid
                e["class"] = "suspect_sw"
        elif k in ("rc_watchdog", "host_oom", "systemd_oomd"):
            e["class"] = "infra"
        elif k == "cuda_launch_timeout":
            e["class"] = "infra" if rc else "suspect_sw"
        elif k == "cuda_fault":
            e["class"] = "infra" if rc else "suspect_sw"
        else:
            e["class"] = "none"
    cls = "suspect_sw" if any(e["class"] == "suspect_sw" for e in evs) else (
        "infra" if any(e["class"] == "infra" for e in evs) else "none")
    return evs, cls


# ------------------------------------------------------------------------------------------ GPU samples
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
        from hybridlc.utils.gpu_lock import compute_apps, is_project_process, lock_ok, read_holder

        holder = read_holder()
        apps = []
        for a in compute_apps():
            if "error" in a:
                continue
            proj = is_project_process(a["pid"], a["name"])
            apps.append({**a, "project": proj, "lock_ok": lock_ok(a["pid"], holder) if proj else None})
        out["compute_apps"] = apps
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


def _allowed_foreign(name: str) -> bool:
    return any(a in name for a in ALLOWED_FOREIGN)


class InfraTagger:
    """GPU sampling thread + kernel-journal scan over the block's window.

    ``snapshot()`` computes the infra block for ``[start, now]`` without stopping the sampler (used for
    every terminal registry row); ``finish()`` stops it and writes ``out_json`` if given.
    """

    def __init__(self, out_json: str | os.PathLike | None = None, tag: str = "", interval: float = 5.0,
                 extra: dict[str, Any] | None = None, sample: bool = True, own_pids: Callable[[], set[int]] | None = None):
        self.out_json = Path(out_json) if out_json else None
        self.tag, self.interval, self.extra = tag, interval, extra or {}
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._sample = sample
        self.own_pids = own_pids
        self.result: dict[str, Any] = {}
        self.t_start = time.time()
        self._th: threading.Thread | None = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.samples.append(query_gpu())
            self._stop.wait(self.interval)

    def start(self) -> "InfraTagger":
        self.t_start = time.time()
        self.counters_start = perf_counters() if self._sample else {}
        self.journal = journal_status()
        if self._sample:
            self._th = threading.Thread(target=self._loop, daemon=True, name=f"infra-{self.tag}")
            self._th.start()
        return self

    __enter__ = start

    def block(self, t_end: float | None = None, log_lines: list[str] | None = None) -> dict[str, Any]:
        """The R2-LIVE v2 infra block for [start, t_end]."""
        t_end = time.time() if t_end is None else t_end
        ke = kernel_events(self.t_start, t_end)
        evs, cls = classify_events(ke["events"], log_lines)
        good = [s for s in list(self.samples) if "error" not in s]
        own = self.own_pids() if self.own_pids else set()
        apps = {}
        for s in good:
            for a in s.get("compute_apps", []):
                apps.setdefault(a["pid"], a)
        foreign = sorted({a["name"] for a in apps.values() if not a.get("project")})
        unlocked = sorted({(a["pid"], a["name"]) for s in good for a in s.get("compute_apps", [])
                           if a.get("project") and a.get("lock_ok") is False and a["pid"] not in own})
        journal_ok = bool(ke["journal_ok"] and self.journal.get("journalctl_k_ok", False))
        return {"window": [self.t_start, t_end], "window_iso": [_iso(self.t_start), _iso(t_end)], "events": evs,
                "classification": cls, "flagged": bool(evs), "n_events": len(evs), "journal_ok": journal_ok,
                "oomd_ok": bool(ke["oomd_journal_ok"]), "journal_source": ke["journal_source"],
                "foreign_gpu_processes": foreign,
                "foreign_disallowed": sorted(n for n in foreign if not _allowed_foreign(n)),
                "unlocked_project_processes": [{"pid": p, "name": n} for p, n in unlocked],
                "n_gpu_samples": len(good), "sample_interval_s": self.interval,
                "lock_wait_s": float(os.environ.get("HYBRIDLC_GPU_LOCK_WAIT_S", "0") or 0)}

    def snapshot(self, log_lines: list[str] | None = None) -> dict[str, Any]:
        return self.block(log_lines=log_lines)

    def finish(self, log_lines: list[str] | None = None) -> dict[str, Any]:
        self._stop.set()
        if self._th is not None:
            self._th.join(timeout=30)
            self.samples.append(query_gpu())
        t_end = time.time()
        time.sleep(0.5 if self._sample else 0.0)
        blk = self.block(t_end, log_lines)
        good = [s for s in self.samples if "error" not in s]
        slow = {f: sum(1 for s in good if str(s.get(f, "")).lower() == "active") for f in SLOWDOWN_FLAGS}
        mem = [s["host_mem_available_gib"] for s in good if "host_mem_available_gib" in s]
        c_end = perf_counters() if self._sample else {}
        self.result = {
            "tag": self.tag, "start": self.t_start, "end": t_end, "start_iso": _iso(self.t_start), "end_iso": _iso(t_end),
            "journal": self.journal, "journal_ok": blk["journal_ok"], "infra_events": blk["events"],
            "flagged": blk["flagged"], "classification": blk["classification"], "infra": blk,
            "gpu": {"n_samples": len(good), "n_errors": len(self.samples) - len(good), "interval_s": self.interval,
                    "slowdown_flag_samples": slow, "any_slowdown": any(slow.values()),
                    "max_temp_c": max((s.get("temperature.gpu") or 0) for s in good) if good else None,
                    "max_power_w": max((s.get("power.draw") or 0) for s in good) if good else None,
                    "min_sm_clock_mhz": min((s.get("clocks.sm") or 0) for s in good) if good else None,
                    "max_util_pct": max((s.get("utilization.gpu") or 0) for s in good) if good else None,
                    "compute_apps_seen": sorted({f"{a['pid']}:{a['name']}" for s in good for a in s.get("compute_apps", [])})},
            "clock_event_counters_us": {"start": getattr(self, "counters_start", {}), "end": c_end,
                                        "delta": {k: c_end.get(k, 0) - v for k, v in getattr(self, "counters_start", {}).items()}},
            "host": {"min_mem_available_gib": min(mem) if mem else None},
            **self.extra,
        }
        if self.out_json:
            self.out_json.parent.mkdir(parents=True, exist_ok=True)
            self.out_json.write_text(json.dumps({**self.result, "samples": self.samples}, indent=1, default=str))
        return self.result

    def __exit__(self, *exc) -> None:
        self.finish()


def registry_fields(block: dict[str, Any]) -> dict[str, Any]:
    """The infra fields every terminal registry row and attempt_end row carries (R2-LIVE v2 L2)."""
    return {"infra_flagged": bool(block.get("flagged")), "infra_class": block.get("classification", "none"),
            "journal_ok": bool(block.get("journal_ok")), "n_infra_events": int(block.get("n_events", 0))}


def run_meta(repo: Path | None = None) -> dict[str, Any]:
    import platform

    repo = repo or Path(__file__).resolve().parents[3]

    def git(*a: str) -> str:
        try:
            return subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True, timeout=20).stdout.strip()
        except Exception:
            return ""
    drv = ""
    with contextlib.suppress(Exception):
        drv = subprocess.run(["nvidia-smi", "--query-gpu=driver_version,name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20).stdout.strip()
    return {"git_commit": git("rev-parse", "HEAD"), "git_dirty_files": len(git("status", "--porcelain").splitlines()),
            "python": platform.python_version(), "platform": platform.platform(), "driver_gpu": drv,
            "hostname": platform.node()}


__all__ = ["KLOG_PATTERN", "XID_RE", "parse_line", "kernel_events", "journal_status", "classify_events", "InfraTagger",
           "registry_fields", "query_gpu", "perf_counters", "run_meta", "STATUS_OF_CLASS", "CLASSES", "JOURNAL_ENV"]

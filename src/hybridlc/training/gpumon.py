"""GPU contention monitor (protocol §18: nvidia-smi every 30 s → ``contended_frac``).

Each sample records device utilisation, memory, SM/memory clocks, ``clocks_event_reasons`` (active
bitmask plus the throttle flags), power, temperature and per-process SM utilisation / framebuffer
memory (``nvidia-smi pmon``, which — unlike ``--query-compute-apps`` — also lists graphics processes
such as CARLA's UE4 server).

Definitions (fixed here so every run is summarised the same way):

* *own* processes: the trainer PID and its descendants;
* ``foreign_sm_pct``: summed SM utilisation of all other processes in the sample;
* a sample is **contended** iff ``foreign_sm_pct >= contended_sm_threshold`` (default 5 %);
* a sample is **throttled** iff any of sw_power_cap, hw_slowdown, sw/hw thermal slowdown is active;
* ``contended_frac`` = contended samples / samples (likewise ``throttled_frac``).
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

GPU_FIELDS = [
    "utilization.gpu", "memory.used", "memory.total", "clocks.sm", "clocks.mem", "clocks.max.sm",
    "clocks_event_reasons.active", "clocks_event_reasons.sw_power_cap", "clocks_event_reasons.hw_slowdown",
    "clocks_event_reasons.sw_thermal_slowdown", "clocks_event_reasons.hw_thermal_slowdown",
    "power.draw", "temperature.gpu",
]
THROTTLE_FIELDS = GPU_FIELDS[7:11]
CONTENDED_SM_THRESHOLD = 5.0


def _num(v: str) -> float | str | None:
    v = v.strip()
    if v in ("-", "", "[N/A]", "N/A"):
        return None
    try:
        return float(v)
    except ValueError:
        return v


def query_gpu(index: int = 0) -> dict[str, Any]:
    r = subprocess.run(["nvidia-smi", f"--id={index}", f"--query-gpu={','.join(GPU_FIELDS)}",
                        "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20, check=True)
    vals = r.stdout.strip().splitlines()[0].split(",")
    return {k: _num(v) for k, v in zip(GPU_FIELDS, vals)}


def query_pmon(index: int = 0) -> list[dict[str, Any]]:
    """Per-process ``sm`` % and ``fb`` MiB from one ``nvidia-smi pmon`` sample."""
    r = subprocess.run(["nvidia-smi", "pmon", "-i", str(index), "-c", "1", "-s", "um"],
                       capture_output=True, text=True, timeout=30, check=True)
    header: list[str] = []
    procs = []
    for line in r.stdout.splitlines():
        if line.startswith("#"):
            if not header:
                header = line.lstrip("#").split()
            continue
        parts = line.split()
        if not parts or not header:
            continue
        row = dict(zip(header, parts))
        try:
            pid = int(row.get("pid", "-"))
        except ValueError:
            continue
        procs.append({"pid": pid, "type": row.get("type"), "sm": _num(row.get("sm", "-")) or 0.0,
                      "fb_mib": _num(row.get("fb", "-")) or 0.0, "name": " ".join(parts[len(header) - 1:])})
    return procs


def descendants(pid: int) -> set[int]:
    """``pid`` and all its descendants (via /proc/<pid>/task/*/children)."""
    out, todo = set(), [pid]
    while todo:
        p = todo.pop()
        if p in out:
            continue
        out.add(p)
        for task in Path(f"/proc/{p}/task").glob("*"):
            try:
                todo.extend(int(c) for c in (task / "children").read_text().split())
            except (OSError, ValueError):
                pass
    return out


def classify(sample: dict[str, Any], own: set[int], threshold: float = CONTENDED_SM_THRESHOLD) -> dict[str, Any]:
    procs = sample.get("processes", [])
    foreign = [p for p in procs if p["pid"] not in own]
    mine = [p for p in procs if p["pid"] in own]
    fsm = float(sum(p["sm"] or 0.0 for p in foreign))
    throttled = any(str(sample.get(k, "")).strip().lower() == "active" for k in THROTTLE_FIELDS)
    return {"foreign_sm_pct": fsm, "foreign_fb_mib": float(sum(p["fb_mib"] or 0.0 for p in foreign)),
            "own_sm_pct": float(sum(p["sm"] or 0.0 for p in mine)),
            "own_fb_mib": float(sum(p["fb_mib"] or 0.0 for p in mine)),
            "contended": fsm >= threshold, "throttled": throttled}


def sample_once(own_pid: int | None, index: int = 0, threshold: float = CONTENDED_SM_THRESHOLD) -> dict[str, Any]:
    rec: dict[str, Any] = {"ts": time.time()}
    try:
        rec.update(query_gpu(index))
        rec["processes"] = query_pmon(index)
    except Exception as e:  # nvidia-smi hiccup: record and carry on
        rec["error"] = f"{type(e).__name__}: {e}"
        return rec
    own = descendants(own_pid) if own_pid else set()
    rec.update(classify(rec, own, threshold))
    return rec


class Summary:
    def __init__(self) -> None:
        self.n = self.n_cont = self.n_thr = self.n_err = 0
        self.util_sum = self.fsm_sum = 0.0

    def add(self, rec: dict[str, Any]) -> None:
        if "error" in rec:
            self.n_err += 1
            return
        self.n += 1
        self.n_cont += bool(rec["contended"])
        self.n_thr += bool(rec["throttled"])
        self.util_sum += float(rec.get("utilization.gpu") or 0.0)
        self.fsm_sum += rec["foreign_sm_pct"]

    def to_dict(self) -> dict[str, Any]:
        n = max(self.n, 1)
        return {"samples": self.n, "errors": self.n_err, "contended_frac": self.n_cont / n if self.n else None,
                "throttled_frac": self.n_thr / n if self.n else None, "mean_util": self.util_sum / n,
                "mean_foreign_sm_pct": self.fsm_sum / n, "contended_sm_threshold": CONTENDED_SM_THRESHOLD}


def summarize_file(path: str | os.PathLike) -> dict[str, Any]:
    s = Summary()
    p = Path(path)
    if p.exists():
        for line in p.read_text().splitlines():
            try:
                s.add(json.loads(line))
            except (json.JSONDecodeError, KeyError):
                continue
    return s.to_dict()


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # a zombie child still answers kill(0); treat it as dead
    try:
        return Path(f"/proc/{pid}/stat").read_text().split()[2] != "Z"
    except OSError:
        return False


def monitor(out: str | os.PathLike, pid: int | None = None, interval: float = 30.0,
            duration: float | None = None, index: int = 0, threshold: float = CONTENDED_SM_THRESHOLD) -> dict:
    """Sample every ``interval`` s into ``out`` (JSONL) until ``pid`` exits or ``duration`` elapses.
    Writes ``<out stem>_summary.json`` after every sample (atomic replace)."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    summ_path = out.with_name(out.stem + "_summary.json")
    s = Summary()
    t_end = time.time() + duration if duration else None
    with open(out, "a", buffering=1) as f:
        while True:
            t0 = time.time()
            rec = sample_once(pid, index, threshold)
            rec["pid"] = pid
            f.write(json.dumps(rec) + "\n")
            s.add(rec)
            tmp = summ_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(s.to_dict(), indent=1))
            os.replace(tmp, summ_path)
            if pid is not None and not pid_alive(pid):
                break
            if t_end is not None and time.time() >= t_end:
                break
            # sleep in short slices so a finished trainer is noticed quickly
            while time.time() - t0 < interval:
                if pid is not None and not pid_alive(pid):
                    break
                if t_end is not None and time.time() >= t_end:
                    break
                time.sleep(min(1.0, interval))
            if (pid is not None and not pid_alive(pid)) or (t_end is not None and time.time() >= t_end):
                break
    return s.to_dict()


# ----------------------------------------------------------------------------------------------
# R2-LIVE v2 L3: lock compliance sampler (every 5 s while a GPU lock is held)
# ----------------------------------------------------------------------------------------------
LOCK_SAMPLE_INTERVAL = 5.0


def lock_sample(holder_pid: int | None = None, lock_file: str | os.PathLike | None = None) -> dict[str, Any]:
    """One compliance sample: every process holding a CUDA context, whether it belongs to this project,
    and for project processes ``lock_ok`` = it is the lock holder or descends from it (/proc ancestry)."""
    from hybridlc.utils.gpu_lock import compute_apps, descends_from, is_locked, is_project_process, read_holder

    lf = Path(lock_file) if lock_file else None
    holder = read_holder(lf)
    hp = holder.get("pid") if holder else holder_pid
    locked = is_locked(lf)
    rec: dict[str, Any] = {"ts": time.time(), "holder_pid": hp, "holder_label": holder.get("label"), "locked": locked,
                           "procs": []}
    for a in compute_apps():
        if "error" in a:
            rec["error"] = a["error"]
            continue
        proj = is_project_process(a["pid"], a["name"])
        ok = bool(proj and locked and hp and descends_from(a["pid"], int(hp))) if proj else None
        rec["procs"].append({"pid": a["pid"], "name": a["name"], "used_mib": a["used_mib"], "project": proj,
                             "lock_ok": ok})
    rec["violations"] = [p["pid"] for p in rec["procs"] if p["project"] and not p["lock_ok"]]
    return rec


class LockMonitor:
    """Background thread writing :func:`lock_sample` records to a JSONL file every 5 s."""

    def __init__(self, out: str | os.PathLike, holder_pid: int | None = None,
                 lock_file: str | os.PathLike | None = None, interval: float = LOCK_SAMPLE_INTERVAL):
        import threading

        self.out, self.holder_pid, self.lock_file, self.interval = Path(out), holder_pid, lock_file, interval
        self._stop = threading.Event()
        self._th = threading.Thread(target=self._loop, daemon=True, name="gpumon-lock")
        self.n = self.n_viol = 0

    def _loop(self) -> None:
        self.out.parent.mkdir(parents=True, exist_ok=True)
        with open(self.out, "a", buffering=1) as f:
            while True:
                try:
                    rec = lock_sample(self.holder_pid, self.lock_file)
                except Exception as e:  # noqa: BLE001 - never kill the job; record
                    rec = {"ts": time.time(), "error": f"{type(e).__name__}: {e}"}
                self.n += 1
                self.n_viol += bool(rec.get("violations"))
                f.write(json.dumps(rec) + "\n")
                if self._stop.wait(self.interval):
                    break

    def start(self) -> "LockMonitor":
        self._th.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self._th.join(timeout=30)

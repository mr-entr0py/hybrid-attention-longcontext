"""Shared helpers for the R3 battery (VT-21…VT-25): metrics, raw-output writers, run metadata.

Kept independent of ``hybridlc.models.testing`` on purpose (the reference side of a reproduction
should not share helpers with the code under test). Raw outputs go to ``experiments/repro/R3/``.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT_ROOT = REPO / "experiments" / "repro" / "R3"
REPRO_ENTROPY = 31337


def repro_rng(*key: int) -> np.random.Generator:
    """The ``repro`` RNG stream (entropy 31337), keyed per use."""
    return np.random.default_rng(np.random.SeedSequence(REPRO_ENTROPY, spawn_key=tuple(int(k) for k in key)))


def torch_gen(*key: int, device: str = "cpu"):
    import torch

    seed = int(repro_rng(*key).integers(0, 2**62))
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    return g


# ------------------------------------------------------------------ metrics (fp64)
def rel_rms(a, ref) -> float:
    a64, r64 = a.double(), ref.double()
    return float((a64 - r64).pow(2).mean().sqrt() / r64.pow(2).mean().sqrt().clamp_min(1e-300))


def rel_max(a, ref) -> float:
    a64, r64 = a.double(), ref.double()
    return float((a64 - r64).abs().max() / r64.abs().max().clamp_min(1e-300))


def max_abs(a, ref) -> float:
    return float((a.double() - ref.double()).abs().max())


def rms(a) -> float:
    return float(a.double().pow(2).mean().sqrt())


# ------------------------------------------------------------------ metadata
def _sh(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception as e:  # pragma: no cover
        return f"<error {e}>"


def gpu_snapshot() -> dict[str, Any]:
    q = _sh(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,clocks.sm,temperature.gpu,power.draw",
             "--format=csv,noheader,nounits"])
    apps = _sh(["nvidia-smi", "--query-compute-apps=pid,used_memory,process_name", "--format=csv,noheader"])
    return {"gpu": q, "apps": apps.splitlines(), "ts": time.time()}


def kernel_events(since: float, until: float | None = None) -> list[str]:
    """Xid / NVRM / OOM / watchdog lines from the kernel log in [since, until]."""
    s = _dt.datetime.fromtimestamp(since).strftime("%Y-%m-%d %H:%M:%S")
    cmd = ["journalctl", "-k", "--no-pager", "--since", s]
    if until is not None:
        cmd += ["--until", _dt.datetime.fromtimestamp(until + 1).strftime("%Y-%m-%d %H:%M:%S")]
    out = _sh(cmd)
    keys = ("xid", "nvrm", "out of memory", "oom-kill", "watchdog", "rc_watchdog")
    return [ln for ln in out.splitlines() if any(k in ln.lower() for k in keys)]


def run_meta(item: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    from hybridlc.utils.provenance import env_info

    meta = {"item": item, "start_ts": time.time(), "start_iso": _dt.datetime.now().isoformat(timespec="seconds"),
            "env": env_info(), "git_status_short": _sh(["git", "-C", str(REPO), "status", "--short"]).splitlines(),
            "gpu_start": gpu_snapshot(), "pid": os.getpid(), "repro_entropy": REPRO_ENTROPY}
    try:
        meta["zoology_commit"] = _sh(["git", "-C", str(REPO / "third_party" / "zoology"), "rev-parse", "HEAD"])
    except Exception:
        pass
    if extra:
        meta.update(extra)
    return meta


def finish_meta(meta: dict[str, Any]) -> dict[str, Any]:
    meta["end_ts"] = time.time()
    meta["end_iso"] = _dt.datetime.now().isoformat(timespec="seconds")
    meta["gpu_end"] = gpu_snapshot()
    meta["kernel_events"] = kernel_events(meta["start_ts"], meta["end_ts"])
    return meta


class RawWriter:
    """experiments/repro/R3/<sub>/<name>.jsonl (+ <name>_meta.json). Never overwrites: adds .rN."""

    def __init__(self, sub: str, name: str, meta: dict[str, Any]):
        d = OUT_ROOT / sub
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"{name}.jsonl"
        n = 0
        while p.exists():
            n += 1
            p = d / f"{name}.r{n}.jsonl"
        self.path = p
        self.meta_path = p.with_name(p.stem + "_meta.json")
        self.meta = meta
        self.f = open(p, "a", buffering=1)
        self.rows: list[dict[str, Any]] = []

    def write(self, row: dict[str, Any]) -> None:
        self.rows.append(row)
        self.f.write(json.dumps(row, default=float) + "\n")

    def close(self, summary: dict[str, Any] | None = None) -> None:
        self.f.close()
        finish_meta(self.meta)
        if summary is not None:
            self.meta["summary"] = summary
        self.meta["raw_jsonl"] = str(self.path.relative_to(REPO))
        self.meta_path.write_text(json.dumps(self.meta, indent=1, default=str))

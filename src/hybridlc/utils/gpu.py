"""GPU sharing helpers.

The RTX 5080 is shared with a CARLA simulator (6–8 GB, up to 100 % util), so every
process caps itself with ``claim()`` before allocating (protocol §7.1, memory cap).
"""

from __future__ import annotations

import os
import subprocess
from typing import Any

import torch

#: Default per-process memory fraction (≈ 7.0 GiB of 16 GB; protocol §7.1).
DEFAULT_FRACTION: float = 0.44

#: Production environment variables (protocol §18).  ``FLA_FLASH_QLA=0`` keeps train and
#: eval GDN numerics on the same Triton path.
PRODUCTION_ENV: dict[str, str] = {
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "PYTHONHASHSEED": "0",
    "FLA_CACHE_RESULTS": "1",
    "FLA_FLASH_QLA": "0",
}


def set_production_env(overwrite: bool = False) -> dict[str, str]:
    """Set the protocol §18 environment variables (call before importing fla kernels)."""
    for k, v in PRODUCTION_ENV.items():
        if overwrite or k not in os.environ:
            os.environ[k] = v
    return {k: os.environ[k] for k in PRODUCTION_ENV}


def claim(fraction: float = DEFAULT_FRACTION, device: int | str | torch.device = 0) -> torch.device:
    """Cap this process's CUDA caching allocator at ``fraction`` of device memory.

    Returns the ``torch.device`` (or ``cpu`` when CUDA is unavailable).
    """
    os.environ.setdefault("FLA_FLASH_QLA", "0")
    if not torch.cuda.is_available():
        return torch.device("cpu")
    dev = torch.device(device if not isinstance(device, int) else f"cuda:{device}")
    torch.cuda.set_per_process_memory_fraction(float(fraction), dev)
    return dev


_SMI_FIELDS = [
    "utilization.gpu", "memory.used", "memory.total", "clocks.sm", "clocks.mem",
    "clocks_event_reasons.active", "power.draw", "temperature.gpu",
]


def snapshot() -> dict[str, Any]:
    """Query nvidia-smi: utilisation, memory, clocks, clock-event reasons, power, temperature,
    and per-process memory.  Returns ``{"error": ...}`` if nvidia-smi is unavailable."""
    out: dict[str, Any] = {}
    try:
        r = subprocess.run(
            ["nvidia-smi", f"--query-gpu={','.join(_SMI_FIELDS)}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        )
        vals = [v.strip() for v in r.stdout.strip().splitlines()[0].split(",")]
        for k, v in zip(_SMI_FIELDS, vals):
            try:
                out[k] = float(v) if k != "clocks_event_reasons.active" else v
            except ValueError:
                out[k] = v
        r2 = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        )
        procs = []
        for line in r2.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 3:
                procs.append({"pid": int(parts[0]), "name": parts[1], "used_mib": float(parts[2]) if parts[2].replace(".", "").isdigit() else parts[2]})
        out["processes"] = procs
    except Exception as e:  # pragma: no cover - environment dependent
        out["error"] = f"{type(e).__name__}: {e}"
    return out

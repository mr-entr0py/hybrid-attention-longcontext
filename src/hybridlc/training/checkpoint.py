"""Atomic checkpoints (protocol §18).

Layout under ``checkpoints/<run_id>/``::

    step_0000250.pt            full: fp32 model, AdamW state, step, RNG states, trainer state,
    step_0000500.pt            config + config hash, git commit, Triton-cache hash (last 2 kept)
    final.pt                   full checkpoint at the end of the run (branch point S, end of B, ...)
    milestones/step_0000382_bf16.pt   weights-only bf16 snapshots (kept forever)

Writes go to ``<name>.tmp.<pid>`` in the same directory, are flushed and fsync'ed, then
``os.replace``-d onto the final name and the directory is fsync'ed. A crash at any point leaves
either the old set of checkpoints or the new one, never a truncated ``.pt``.
"""
from __future__ import annotations

import os
import random
import re
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

CKPT_RE = re.compile(r"^step_(\d{7,})\.pt$")
FORMAT_VERSION = 1


def _fsync_dir(d: Path) -> None:
    try:
        fd = os.open(d, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_torch_save(obj: Any, path: str | os.PathLike,
                      _fault_hook: Callable[[Path], None] | None = None) -> None:
    """``torch.save`` to ``path`` atomically (tmp -> flush -> fsync -> os.replace -> fsync dir)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with open(tmp, "wb") as f:
        torch.save(obj, f)
        f.flush()
        if _fault_hook is not None:  # VT-13: kill -9 while the checkpoint is being written
            _fault_hook(tmp)
        os.fsync(f.fileno())
    os.replace(tmp, path)
    _fsync_dir(path.parent)


def rng_state() -> dict[str, Any]:
    st = {"torch_cpu": torch.get_rng_state(), "numpy": np.random.get_state(), "python": random.getstate()}
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        st["torch_cuda"] = torch.cuda.get_rng_state_all()
    return st


def set_rng_state(st: dict[str, Any]) -> None:
    torch.set_rng_state(st["torch_cpu"])
    np.random.set_state(st["numpy"])
    random.setstate(st["python"])
    if "torch_cuda" in st and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st["torch_cuda"])


def list_step_checkpoints(ckpt_dir: str | os.PathLike) -> list[tuple[int, Path]]:
    d = Path(ckpt_dir)
    if not d.is_dir():
        return []
    out = []
    for p in d.iterdir():
        m = CKPT_RE.match(p.name)
        if m:
            out.append((int(m.group(1)), p))
    return sorted(out)


def cleanup_tmp(ckpt_dir: str | os.PathLike) -> list[Path]:
    """Remove leftovers of interrupted writes (``*.tmp.<pid>``)."""
    d = Path(ckpt_dir)
    removed = []
    if d.is_dir():
        for p in list(d.iterdir()) + (list((d / "milestones").iterdir()) if (d / "milestones").is_dir() else []):
            if ".tmp." in p.name:
                p.unlink(missing_ok=True)
                removed.append(p)
    return removed


def latest_checkpoint(ckpt_dir: str | os.PathLike) -> Path | None:
    """The newest full checkpoint (``final.pt`` wins over step checkpoints)."""
    d = Path(ckpt_dir)
    if (d / "final.pt").is_file():
        return d / "final.pt"
    lst = list_step_checkpoints(d)
    return lst[-1][1] if lst else None


def save_full(ckpt_dir: str | os.PathLike, step: int, *, model: torch.nn.Module,
              optimizer: torch.optim.Optimizer, meta: dict[str, Any], trainer_state: dict[str, Any],
              keep_last: int = 2, final: bool = False,
              fault_hook: Callable[[Path], None] | None = None) -> Path:
    """Write a full checkpoint for the state *after* ``step`` optimiser updates have been applied."""
    d = Path(ckpt_dir)
    payload = {
        "format_version": FORMAT_VERSION,
        "step": int(step),
        "model": {k: v.detach().clone() if v.is_floating_point() else v for k, v in model.state_dict().items()},
        "optimizer": optimizer.state_dict(),
        "rng": rng_state(),
        "trainer_state": trainer_state,
        "saved_at": time.time(),
        **meta,  # config, config_hash, run_id, git_commit, triton_cache_hash, model_config_hash, env
    }
    # fp32 master weights are what the model holds; assert so nobody saves a cast model by mistake.
    for k, v in payload["model"].items():
        if v.is_floating_point() and v.dtype != torch.float32:
            raise AssertionError(f"full checkpoint expects fp32 master weights, {k} is {v.dtype}")
    path = d / ("final.pt" if final else f"step_{step:07d}.pt")
    atomic_torch_save(payload, path, _fault_hook=fault_hook)
    if not final:
        for _, old in list_step_checkpoints(d)[:-keep_last] if keep_last > 0 else []:
            old.unlink(missing_ok=True)
    return path


def save_milestone(ckpt_dir: str | os.PathLike, step: int, model: torch.nn.Module,
                   meta: dict[str, Any], tag: str = "") -> Path:
    """Weights-only bf16 snapshot (protocol §18 milestones)."""
    sd = {k: (v.detach().to(torch.bfloat16) if v.is_floating_point() else v.detach().clone())
          for k, v in model.state_dict().items()}
    name = f"step_{step:07d}{('_' + tag) if tag else ''}_bf16.pt"
    path = Path(ckpt_dir) / "milestones" / name
    atomic_torch_save({"format_version": FORMAT_VERSION, "step": int(step), "model": sd, "dtype": "bfloat16",
                       **meta}, path)
    return path


def load(path: str | os.PathLike, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    # weights_only=False: the payload contains numpy RNG state and plain python objects we wrote.
    return torch.load(path, map_location=map_location, weights_only=False)


def param_sha256(model: torch.nn.Module) -> str:
    """SHA-256 over all parameters/buffers in state-dict order (VT-13 bitwise comparison)."""
    import hashlib

    h = hashlib.sha256()
    for k, v in model.state_dict().items():
        h.update(k.encode())
        t = v.detach().cpu().contiguous().reshape(-1)
        h.update(str(t.dtype).encode())
        h.update(t.view(torch.uint8).numpy().tobytes() if t.numel() else b"")
    return h.hexdigest()

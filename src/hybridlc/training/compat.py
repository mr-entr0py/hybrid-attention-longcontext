"""Thin adapters onto ``hybridlc.utils`` (owned by another builder).

The training infrastructure is built concurrently with ``hybridlc.utils``. Every helper here first
tries the real contract function from INTERFACES.md and only falls back to a local implementation of
the *same contract* if the real module is not importable yet. The fallbacks are deliberately
minimal; once ``hybridlc.utils`` lands they are never used.

Nothing here touches the global RNG.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

#: Repository root (src/hybridlc/training/compat.py -> repo).
REPO_ROOT = Path(__file__).resolve().parents[3]

#: Paths whose modification makes the tree "dirty" for the purpose of refusing to start a run
#: (protocol §18). The append-only registry under experiments/ and reports/ are excluded on purpose:
#: the runner appends to ``experiments/registry.jsonl`` while jobs run, which must not block the queue.
CODE_PATHS = ("src", "scripts", "configs", "pyproject.toml", "uv.lock")


# ----------------------------------------------------------------------------------------------
# provenance
# ----------------------------------------------------------------------------------------------
def canonical_json(obj: Any) -> str:
    """Canonical JSON used for hashing (sorted keys, no whitespace, UTF-8 kept)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def config_hash(obj: Any) -> str:
    """SHA-256 of canonical JSON. Delegates to ``hybridlc.utils.provenance.config_hash`` if present."""
    try:
        from hybridlc.utils.provenance import config_hash as _ch  # type: ignore

        return _ch(obj)
    except ImportError:
        return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def sha256_file(path: str | os.PathLike, bufsize: int = 1 << 20) -> str:
    try:
        from hybridlc.utils.provenance import sha256_file as _sf  # type: ignore

        return _sf(path)
    except ImportError:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(bufsize):
                h.update(chunk)
        return h.hexdigest()


def _git(*args: str, cwd: Path = REPO_ROOT) -> str:
    try:
        return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True,
                              check=True, timeout=30).stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return ""


def git_commit() -> str:
    return _git("rev-parse", "HEAD") or "unknown"


def git_dirty_paths(paths: tuple[str, ...] = CODE_PATHS) -> list[str]:
    """Modified or untracked files under the code paths (empty list == clean)."""
    existing = [p for p in paths if (REPO_ROOT / p).exists()]
    if not existing:
        return []
    out = _git("status", "--porcelain", "--untracked-files=all", "--", *existing)
    return [line for line in out.splitlines() if line.strip()]


def env_info() -> dict[str, Any]:
    """Versions, git commit, dirty flag, GPU name and driver."""
    try:
        from hybridlc.utils.provenance import env_info as _ei  # type: ignore

        info = dict(_ei())
    except ImportError:
        info = {}
        try:
            import torch

            info["torch"] = torch.__version__
            info["cuda"] = torch.version.cuda
            if torch.cuda.is_available():
                info["gpu_name"] = torch.cuda.get_device_name(0)
        except Exception:  # pragma: no cover - torch always present in this env
            pass
        for mod in ("triton", "fla", "transformers", "tokenizers", "numpy"):
            try:
                info[mod] = __import__(mod).__version__
            except Exception:
                info[mod] = None
        info["driver"] = _nvidia_driver()
        info["git_commit"] = git_commit()
    # The code-path dirty list is what the runbook gate uses; always record it.
    dirty = git_dirty_paths()
    info["git_dirty_code_paths"] = dirty
    info.setdefault("dirty", bool(dirty))
    return info


def _nvidia_driver() -> str | None:
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=20).stdout.strip() or None
    except (subprocess.SubprocessError, FileNotFoundError):
        return None


# ----------------------------------------------------------------------------------------------
# GPU
# ----------------------------------------------------------------------------------------------
def claim_gpu(fraction: float) -> None:
    """Set the per-process CUDA memory fraction (shared GPU with CARLA, protocol §7.1)."""
    try:
        from hybridlc.utils.gpu import claim as _claim  # type: ignore

        _claim(fraction)
        return
    except ImportError:
        pass
    import torch

    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(fraction)


# ----------------------------------------------------------------------------------------------
# Triton cache hash (protocol §18: every full checkpoint records it)
# ----------------------------------------------------------------------------------------------
_TRITON_MEMO: dict[tuple[str, int, int], str] = {}


def triton_cache_hash(cache_dir: str | os.PathLike | None = None) -> str | None:
    """SHA-256 over (relative path, file SHA-256) of every file in the Triton cache.

    Per-file digests are memoised on (path, size, mtime_ns), so repeated calls during a run only hash
    files that the JIT added since the last checkpoint.
    """
    cache_dir = cache_dir or os.environ.get("TRITON_CACHE_DIR") or os.path.expanduser("~/.triton/cache")
    root = Path(cache_dir)
    if not root.is_dir():
        return None
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.name.endswith(".tmp") or ".tmp." in p.name:
            continue
        try:
            st = p.stat()
        except FileNotFoundError:  # removed by a concurrent JIT
            continue
        key = (str(p), st.st_size, st.st_mtime_ns)
        digest = _TRITON_MEMO.get(key)
        if digest is None:
            try:
                digest = sha256_file(p)
            except FileNotFoundError:
                continue
            _TRITON_MEMO[key] = digest
        h.update(str(p.relative_to(root)).encode())
        h.update(b"\0")
        h.update(digest.encode())
        h.update(b"\n")
    return h.hexdigest()


# ----------------------------------------------------------------------------------------------
# seeds (training itself draws no random numbers; these are used only by the test fakes)
# ----------------------------------------------------------------------------------------------
PROJECT_SALT = 271828182
STREAMS = dict(init_shared=0, init_mixer=1, data_A=2, data_B=3, tef=4)


def seed_sequence(seed_index: int, stream_id: int, *extra: int):
    try:
        from hybridlc.utils.seeds import seed_sequence as _ss  # type: ignore

        return _ss(seed_index, stream_id, *extra)
    except ImportError:
        import numpy as np

        return np.random.SeedSequence(entropy=PROJECT_SALT, spawn_key=(seed_index, stream_id, *extra))


def torch_generator(seed_index: int, stream_id: int, *extra: int, device: str = "cpu"):
    try:
        from hybridlc.utils.seeds import torch_generator as _tg  # type: ignore

        return _tg(seed_index, stream_id, *extra, device=device)
    except ImportError:
        import torch

        seed = int(seed_sequence(seed_index, stream_id, *extra).generate_state(1, dtype="uint64")[0])
        g = torch.Generator(device=device)
        g.manual_seed(seed & ((1 << 63) - 1))
        return g

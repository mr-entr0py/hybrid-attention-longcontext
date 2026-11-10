"""Provenance helpers: environment info, canonical config hashes, file hashes."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=10, check=True).stdout.strip()
    except Exception:
        return None


def env_info() -> dict[str, Any]:
    """Versions, git commit, dirty flag, GPU name and driver."""
    info: dict[str, Any] = {"python": platform.python_version(), "platform": platform.platform()}
    for mod in ("torch", "triton", "fla", "transformers", "tokenizers", "numpy"):
        try:
            m = __import__(mod)
            info[mod] = getattr(m, "__version__", "unknown")
        except Exception:
            info[mod] = None
    try:
        import torch

        info["cuda"] = torch.version.cuda
        info["cudnn"] = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_capability"] = list(torch.cuda.get_device_capability(0))
    except Exception:
        pass
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
        info["driver"] = r.stdout.strip().splitlines()[0] if r.returncode == 0 else None
    except Exception:
        info["driver"] = None
    info["git_commit"] = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    info["git_dirty"] = None if status is None else bool(status)
    info["env"] = {k: os.environ.get(k) for k in ("CUBLAS_WORKSPACE_CONFIG", "PYTHONHASHSEED", "TRITON_CACHE_DIR", "FLA_CACHE_RESULTS", "FLA_FLASH_QLA")}
    info["fla_install"] = fla_install_info()
    info.update(data_provenance())
    return info


def fla_install_info() -> dict[str, Any]:
    """Where ``fla`` is imported from and how it was installed (verification round 1,
    FLA-INSTALL-NOT-PINNED-CLONE): PyPI wheel vs an editable/direct-URL install of the pinned clone."""
    out: dict[str, Any] = {}
    try:
        import importlib.metadata as md

        import fla

        out["path"] = str(Path(fla.__file__).resolve().parent)
        dist = md.distribution("flash-linear-attention")
        out["version"] = dist.version
        du = dist.read_text("direct_url.json")
        out["direct_url"] = json.loads(du) if du else None
        out["source"] = "direct_url" if du else "index (PyPI wheel)"
    except Exception as e:  # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {e}"
    return out


def data_provenance() -> dict[str, Any]:
    """Tokenizer SHA-256 (§6.2) and data MANIFEST SHA-256, for the run registry and env.json
    (verification round 1, DTS-08). Missing files give None (unit tests with fake data)."""
    data = REPO_ROOT / "data"
    tok_sha = None
    try:
        tok_sha = (data / "tokenizer" / "tokenizer.sha256").read_text().split()[0].strip()
    except Exception:  # noqa: BLE001
        pass
    man = data / "MANIFEST.json"
    return {"tokenizer_sha256": tok_sha, "data_manifest_sha256": sha256_file(man) if man.is_file() else None}


def _to_jsonable(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _to_jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def canonical_json(obj: Any) -> str:
    """Canonical JSON: sorted keys, no whitespace, dataclasses expanded, ASCII-safe."""
    return json.dumps(_to_jsonable(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def config_hash(obj: Any) -> str:
    """SHA-256 (hex) of the canonical JSON of ``obj``."""
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def sha256_file(path: str | os.PathLike, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_tensor(t: "Any") -> str:
    """SHA-256 of a tensor's raw bytes (CPU, contiguous), including dtype and shape."""
    import torch

    t = t.detach().to("cpu").contiguous()
    h = hashlib.sha256()
    h.update(f"{t.dtype}|{tuple(t.shape)}|".encode())
    h.update(t.view(torch.uint8).numpy().tobytes() if t.numel() else b"")
    return h.hexdigest()

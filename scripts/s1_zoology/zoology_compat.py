"""Make the pinned Zoology checkout importable and runnable in this venv (protocol §17).

Three things are needed, and nothing else in Zoology's code path is changed:

1. **Import path.** Zoology lives at ``third_party/zoology`` (pinned commit
   ``1ad20d193b6113cae1e8f3c655c300d7b4b3f4bb``). Its ``setup.py`` imports torch at build
   time, and executing it (``uv pip install -e ... --no-build-isolation``) was refused by
   this session's permission policy, so we put the checkout on ``sys.path`` instead.
   This is import-equivalent to an editable install (same source files).
   DEVIATION (packaging only; see reports/s1_reproduction.md).

2. **fla 0.6.0 API shim.** Zoology's ``mixers/gated_delta_net.py`` calls
   ``chunk_gated_delta_rule(..., head_first=False, ...)``. fla 0.6.0 removed the
   ``head_first`` argument and *raises* if it is passed at all; ``head_first=False`` was
   the ``[B, T, H, D]`` layout that fla 0.6.0 now assumes unconditionally. The shim drops
   that keyword when it equals ``False`` (and refuses ``True``), so the computation is
   identical. DEVIATION (API compatibility only; no numerical change).

3. **wandb off.** The protocol says "wandb disabled". Zoology's ``WandbLogger`` already
   becomes a no-op when ``project_name``/``entity`` are None; we additionally replace it
   with :class:`CaptureLogger`, which records exactly the metrics Zoology would have sent to
   wandb, so results can be written to ``experiments/s1_zoology/``.
"""
from __future__ import annotations

import functools
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
ZOOLOGY_DIR = REPO_ROOT / "third_party" / "zoology"
ZOOLOGY_COMMIT = "1ad20d193b6113cae1e8f3c655c300d7b4b3f4bb"  # protocol §17
DATA_CACHE_DIR = REPO_ROOT / "data" / "zoology"  # protocol §17: cache_dir data/zoology/
RESULTS_DIR = REPO_ROOT / "experiments" / "s1_zoology"
GPU_MEMORY_FRACTION = 0.44  # INTERFACES.md: every GPU process claims 0.44 of the card

_PATCHED = False


def zoology_commit() -> str:
    """Return the HEAD commit of the Zoology checkout (to be recorded with every result)."""
    out = subprocess.run(
        ["git", "-C", str(ZOOLOGY_DIR), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    return out.stdout.strip()


def zoology_dirty() -> bool:
    out = subprocess.run(
        ["git", "-C", str(ZOOLOGY_DIR), "status", "--porcelain"],
        capture_output=True, text=True, check=True,
    )
    return bool(out.stdout.strip())


def claim_gpu(fraction: float = GPU_MEMORY_FRACTION) -> None:
    """Cap this process's share of the shared GPU (CARLA runs alongside)."""
    try:  # prefer the project helper once it exists (INTERFACES.md utils/gpu.py)
        from hybridlc.utils.gpu import claim  # type: ignore

        claim(fraction)
        return
    except Exception:
        pass
    import torch

    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(fraction)


class CaptureLogger:
    """Drop-in replacement for ``zoology.logger.WandbLogger`` that keeps metrics in memory.

    Zoology calls ``log({"train/loss": ..., "epoch": e})`` every step and
    ``log({"epoch": e, "valid/loss": ..., "valid/accuracy": ..., "valid/num_kv_pairs/accuracy-N": ...})``
    once per epoch. We keep every validation record and a per-epoch train-loss mean.
    The *last* validation record is what wandb's run summary would show, which is what
    Zoology's own analysis scripts read (``zoology/analysis/__init__.py``).
    """

    instances: list["CaptureLogger"] = []

    def __init__(self, config: Any) -> None:
        self.config = config
        self.valid: list[dict[str, float]] = []
        self.train_loss_sum: dict[int, float] = {}
        self.train_loss_n: dict[int, int] = {}
        self.extra: dict[str, Any] = {}
        CaptureLogger.instances.append(self)

    def log_config(self, config: Any) -> None:  # noqa: D401 - zoology API
        pass

    def log_model(self, model: Any, config: Any) -> None:
        max_seq_len = max(c.input_seq_len for c in config.data.test_configs)
        self.extra["num_parameters"] = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
        try:
            self.extra["state_size"] = int(model.state_size(sequence_length=max_seq_len))
        except Exception as exc:  # state_size is informational only
            self.extra["state_size_error"] = repr(exc)

    def log(self, metrics: dict[str, Any]) -> None:
        if "train/loss" in metrics:
            e = int(metrics["epoch"])
            self.train_loss_sum[e] = self.train_loss_sum.get(e, 0.0) + float(metrics["train/loss"])
            self.train_loss_n[e] = self.train_loss_n.get(e, 0) + 1
        if "valid/accuracy" in metrics:
            self.valid.append({k: float(v) for k, v in metrics.items()})
            if self.on_valid is not None:
                self.on_valid(self)

    on_valid = None  # optional callback(logger) after each validation (used for timing / checkpoints)

    def finish(self) -> None:
        pass

    def train_loss_by_epoch(self) -> dict[int, float]:
        return {e: self.train_loss_sum[e] / self.train_loss_n[e] for e in sorted(self.train_loss_sum)}


def _patch_fla_head_first(zgdn_module: Any) -> None:
    """Wrap fla kernels referenced by zoology.mixers.gated_delta_net to drop head_first=False."""

    def wrap(fn):
        @functools.wraps(fn)
        def inner(*args, **kwargs):
            hf = kwargs.pop("head_first", False)
            if hf:
                raise ValueError("head_first=True is not supported by fla 0.6.0; shim refuses it")
            return fn(*args, **kwargs)

        inner.__zoology_compat__ = True  # type: ignore[attr-defined]
        return inner

    for name in ("chunk_gated_delta_rule", "fused_recurrent_gated_delta_rule"):
        fn = getattr(zgdn_module, name)
        if not getattr(fn, "__zoology_compat__", False):
            setattr(zgdn_module, name, wrap(fn))


def setup_zoology(check_commit: bool = True) -> None:
    """Put Zoology on sys.path, apply the fla shim and the capture logger. Idempotent."""
    global _PATCHED
    if _PATCHED:
        return
    if check_commit:
        head = zoology_commit()
        if head != ZOOLOGY_COMMIT:
            raise RuntimeError(f"third_party/zoology is at {head}, protocol pins {ZOOLOGY_COMMIT}")
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("WANDB_DISABLED", "true")
    # Zoology's Trainer wraps every step in tqdm; keep the nohup logs readable.
    os.environ.setdefault("TQDM_MININTERVAL", "60")
    if str(ZOOLOGY_DIR) not in sys.path:
        sys.path.insert(0, str(ZOOLOGY_DIR))

    import zoology.mixers.gated_delta_net as zgdn
    import zoology.train as ztrain

    _patch_fla_head_first(zgdn)
    ztrain.WandbLogger = CaptureLogger  # train() instantiates WandbLogger(config)
    _PATCHED = True

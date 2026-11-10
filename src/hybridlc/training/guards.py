"""Start-of-run policy checks of the trainer (R2-LIVE v2 L1, PI ruling D069 / PR-12 part 1).

``check_cache_and_lock_policy(cfg)`` returns a dict; ``refuse`` (a reason string) makes the trainer
write a ``failed`` row and exit with code 77 before any CUDA call.

* **GPU-lock escape hatch.** ``HYBRIDLC_GPU_UNLOCKED=1`` is refused for every exp_stage S2..S8 run and
  for reproduction gate runs (``HYBRIDLC_GATE_RUN=1``). Otherwise the run is a labelled debug run: its
  registry rows go to ``<run_dir>/debug_registry.jsonl``, never to the study registry.
* **Triton cache (exp_stage S2..S8).** The run's ``TRITON_CACHE_DIR`` must be the production directory
  ``$REPO/.cache/triton``, or a directory whose trainer-style hash (``compat.triton_cache_hash``)
  equals the recorded pre-FEAS hash (registry note ``run_id = pre-FEAS-triton-cache``, field
  ``triton_cache_hash``, in the run's registry or the study registry; TC-PF). The diagnostics cache
  ``$REPO/.cache/triton-diag`` and every other directory are refused.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from .compat import REPO_ROOT, triton_cache_hash

STUDY_STAGE_RE = re.compile(r"^S[2-8]")
PRODUCTION_CACHE = REPO_ROOT / ".cache" / "triton"
DIAG_CACHE = REPO_ROOT / ".cache" / "triton-diag"
PREFEAS_NOTE_ID = "pre-FEAS-triton-cache"
STUDY_REGISTRY = REPO_ROOT / "experiments" / "registry.jsonl"


def is_study_stage(exp_stage: str) -> bool:
    return bool(STUDY_STAGE_RE.match(str(exp_stage or "")))


def recorded_prefeas_hashes(registries: list[Path]) -> list[str]:
    """Every ``triton_cache_hash`` of a ``pre-FEAS-triton-cache`` note in the given registries."""
    out: list[str] = []
    for reg in registries:
        try:
            lines = Path(reg).read_text().splitlines()
        except OSError:
            continue
        for ln in lines:
            if PREFEAS_NOTE_ID not in ln:
                continue
            try:
                r = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if r.get("run_id") == PREFEAS_NOTE_ID and r.get("triton_cache_hash"):
                out.append(str(r["triton_cache_hash"]))
    return out


def triton_cache_policy(exp_stage: str, registry_path: Path, cache_dir: str | None = None) -> dict[str, Any]:
    """``{ok, kind, cache_dir, hash?, reason?}`` for a run of ``exp_stage``."""
    cd = cache_dir or os.environ.get("TRITON_CACHE_DIR") or os.path.expanduser("~/.triton/cache")
    p = Path(cd).resolve()
    rec: dict[str, Any] = {"cache_dir": str(p)}
    if not is_study_stage(exp_stage):
        return {**rec, "ok": True, "kind": "not-checked (exp_stage not S2..S8)"}
    if p == PRODUCTION_CACHE.resolve():
        return {**rec, "ok": True, "kind": "production"}
    if p == DIAG_CACHE.resolve():
        return {**rec, "ok": False, "kind": "diagnostics",
                "reason": f"exp_stage {exp_stage} refused: TRITON_CACHE_DIR is the diagnostics cache {p} "
                          "(study runs use the production dir or an extraction of the pre-FEAS archive; PR-12)"}
    want = recorded_prefeas_hashes([Path(registry_path), STUDY_REGISTRY])
    have = triton_cache_hash(p) if p.is_dir() else None
    if have is not None and have in want:
        return {**rec, "ok": True, "kind": "pre-FEAS extraction", "hash": have}
    return {**rec, "ok": False, "kind": "other", "hash": have,
            "reason": f"exp_stage {exp_stage} refused: TRITON_CACHE_DIR {p} is neither the production dir nor an "
                      f"extraction whose trainer-style hash equals the recorded pre-FEAS hash "
                      f"({'none recorded' if not want else 'hash differs'}; PR-12)"}


def check_cache_and_lock_policy(cfg: Any) -> dict[str, Any]:
    from hybridlc.utils.gpu_lock import UNLOCKED_ENV, verify_token

    out: dict[str, Any] = {}
    study = is_study_stage(cfg.exp_stage)
    gate = os.environ.get("HYBRIDLC_GATE_RUN") == "1"
    if os.environ.get(UNLOCKED_ENV) == "1":
        out["lock_mode"] = "unlocked"
        if study or gate:
            out["refuse"] = (f"{UNLOCKED_ENV}=1 is refused for exp_stage {cfg.exp_stage}"
                             f"{' / reproduction gate runs' if gate else ''} (R2-LIVE v2 L1)")
            return out
        out["debug_registry"] = True
    elif str(cfg.runtime.device).startswith("cuda"):
        try:
            out["lock_mode"] = "inherited" if verify_token() is not None else "none"
        except RuntimeError as e:
            out["lock_mode"] = f"error: {e}"
    else:
        out["lock_mode"] = "cpu"
    pol = triton_cache_policy(cfg.exp_stage, Path(cfg.registry_path))
    out["cache_check"] = {k: v for k, v in pol.items() if k != "reason"}
    if not pol["ok"]:
        out["refuse"] = pol["reason"]
    return out


__all__ = ["check_cache_and_lock_policy", "triton_cache_policy", "recorded_prefeas_hashes", "is_study_stage",
           "PRODUCTION_CACHE", "DIAG_CACHE", "PREFEAS_NOTE_ID"]

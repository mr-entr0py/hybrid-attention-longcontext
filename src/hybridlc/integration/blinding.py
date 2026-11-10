"""Technical DTS-10 blinding (DTS10-AT v2 ``blinding.technical``; PI ruling D082).

While no ``DTS10-commit`` registry note exists, FEAS (exp_stage S2) scores must not be printed or
summarised: ``scripts/evaluate.py`` writes them only to result files, and ``scripts/analyse_results.py``
(and any summary tool) refuses S2 result trees. This guards against inadvertent reading; the files stay
readable, so the procedural rule (no agent opens FEAS scores before the note) still applies.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

from hybridlc.utils.provenance import REPO_ROOT

DTS10_NOTE_ID = "DTS10-commit"
STUDY_REGISTRY = REPO_ROOT / "experiments" / "registry.jsonl"
BLINDED_STAGES = ("S2",)


def is_blinded_stage(exp_stage_or_run_id: str | None) -> bool:
    """True for exp_stage S2 (a run_id starts with its exp_stage label, e.g. ``S2A-D384-C12-s101-...``)."""
    s = str(exp_stage_or_run_id or "")
    return any(s == st or (s.startswith(st) and not s[len(st):len(st) + 1].isdigit()) for st in BLINDED_STAGES)


def dts10_committed(registries: Iterable[str | os.PathLike] | None = None) -> bool:
    """A ``DTS10-commit`` note exists in any of the registries (default: the study registry)."""
    for reg in (registries or [STUDY_REGISTRY]):
        try:
            text = Path(reg).read_text()
        except OSError:
            continue
        for ln in text.splitlines():
            if DTS10_NOTE_ID not in ln:
                continue
            try:
                if json.loads(ln).get("run_id") == DTS10_NOTE_ID:
                    return True
            except json.JSONDecodeError:
                continue
    return False


def blinded(exp_stage_or_run_id: str | None, registries: Iterable[str | os.PathLike] | None = None) -> bool:
    return is_blinded_stage(exp_stage_or_run_id) and not dts10_committed(registries)


__all__ = ["DTS10_NOTE_ID", "is_blinded_stage", "dts10_committed", "blinded"]

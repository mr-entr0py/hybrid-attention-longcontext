"""VT-12 regression, verification round 2 (ABSENT-CANDIDATES-PRESENT-IN-FILLER).

The frozen items draw 'absent' candidate values excluding only the fact values, so a small share of
them occur in the natural filler (e.g. years). Items are frozen and unchanged; instead every scored
query records ``absent_in_context`` (indices of absent-typed candidates present in the context) so
error types can be relabelled in the analysis.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hybridlc.eval.items import load_items
from hybridlc.eval.scorer import _absent_in_context

ITEMS = Path(__file__).resolve().parents[1] / "experiments" / "items"


def test_absent_in_context_unit():
    item = dict(context_ids=np.array([5, 6, 7, 8, 9, 10, 11]))
    q = dict(candidates=[[1, 2], [7, 8], [8, 7], [10, 11], [11, 12]])
    types = ["gold", "context", "absent", "absent", "absent"]
    assert _absent_in_context(item, q, types) == [3]
    assert _absent_in_context(item, q, ["gold", "context", "context", "context", "context"]) is None


def test_frozen_cell_rate_small_but_nonzero():
    f = ITEMS / "dev" / "DG" / "R1_D15_32K.npz"
    if not f.is_file():
        pytest.skip("frozen items not present")
    items, _ = load_items(f)
    n = k = 0
    for it in items:
        for q in it["queries"]:
            types = q["meta"].get("cand_types", [])
            r = _absent_in_context(it, q, types)
            if r is not None:
                n += sum(t == "absent" for t in types)
                k += len(r)
    assert n > 0 and 0 < k / n < 0.01  # measured 23 / 3,200 = 0.72%

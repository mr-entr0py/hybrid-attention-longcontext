"""VT-12 (Tier 0): LM-1 per-source composition (verification round 3, LM1-PG19-BOOK-SHORTFALL).

§9.6 plans LM-1 as up to 300 FWE test-half documents >= 32,768 tokens plus 100 PG19-test books. Only
78 of the 100 PG19-test books are long enough, so the frozen cell holds 201 FWE + 78 PG19 = 279 items
(a recorded DEVIATION). The scorer's result header now reports actual and planned per-source counts.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from hybridlc.eval.scorer import lm_source_counts

REPO = Path(__file__).resolve().parents[1]
LM1 = REPO / "experiments" / "items" / "test" / "Tier0" / "LM-1_all_32K.npz"


def test_lm_source_counts_header_fields():
    rows = [dict(filler_source=dict(kind="fwe")) for _ in range(3)] + [dict(filler_source=dict(kind="pg19"))]
    h = lm_source_counts(rows, "LM-1")
    assert h["source_counts"] == {"fwe": 3, "pg19": 1}
    assert h["source_counts_planned"] == {"fwe": 300, "pg19": 100}
    assert "source_counts_planned" not in lm_source_counts(rows, "LM-4")


def test_frozen_lm1_composition():
    if not LM1.is_file():
        pytest.skip("frozen Tier-0 items not present")
    from hybridlc.eval.items import load_items

    items, _ = load_items(LM1)
    kinds = Counter(it["meta"]["filler_source"]["kind"] for it in items)
    assert kinds == {"fwe": 201, "pg19": 78}, kinds
    assert all(len(it["context_ids"]) == 32768 for it in items)
    rows = [dict(filler_source=it["meta"]["filler_source"]) for it in items]
    assert lm_source_counts(rows, "LM-1")["source_counts"] == {"fwe": 201, "pg19": 78}

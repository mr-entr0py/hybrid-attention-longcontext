"""Regression for verification round 1, AFREQ-RECENCY-FLOOR.

A-freq rows carry ``recency_heuristic_correct`` / ``primacy_heuristic_correct`` (would "answer
the probe that occurs last / first" be right?), so the counting-free floor ~rho/(1+rho) can be
reported next to A-freq scores; other tasks do not get the fields.

Verification round 3, AFREQ-SHORT-WINDOW-FLOOR: rows also carry ``window{W}_count_heuristic``
(W = 256 / 1,024 / 4,096; a counter over the last W context tokens, ties 1/2), which is near ceiling
at T <= 4K (short-T A-freq = local frequency comparison) and degrades at 32K.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hybridlc.eval.devtools import InductionLM, resolve_test_tokenizer
from hybridlc.eval.filler import StoreFiller, dev_text_filler
from hybridlc.eval.generators import generate
from hybridlc.eval.items import load_items, save_items
from hybridlc.eval.rng import eval_rng
from hybridlc.eval.scorer import _afreq_heuristics, score_file
from hybridlc.eval.tokens import token_space

REPO = Path(__file__).resolve().parents[1]
TOKENS_REAL = (REPO / "data" / "tokens" / "pg19" / "test.bin").exists()


def test_rows_carry_heuristic_fields(tmp_path):
    tok = resolve_test_tokenizer()[0]
    ts = token_space(tok)
    fill = StoreFiller(ts, "dev") if TOKENS_REAL else dev_text_filler(ts, "dev")
    lm = InductionLM(ts.vocab_size, order=6)
    for task, lvl in (("A-freq", {"rho": 2.0}), ("R1", {"D": 3})):
        items = generate(task, lvl, 1024, 3, eval_rng("dev", "afreq-floor", task), "dev", tok, fill)
        p = tmp_path / f"{task}.npz"
        save_items(p, items, dict(task=task, level=lvl, T=1024))
        rows = [json.loads(x) for x in score_file(lm, p, tmp_path / f"{task}.jsonl").read_text().splitlines()]
        if task == "A-freq":
            assert all(isinstance(r["recency_heuristic_correct"], bool) for r in rows)
            assert all(isinstance(r["primacy_heuristic_correct"], bool) for r in rows)
            for W in (256, 1024, 4096):
                assert all(r[f"window{W}_count_heuristic"] in (0.0, 0.5, 1.0) for r in rows)
        else:
            assert "recency_heuristic_correct" not in rows[0]
            assert "window1024_count_heuristic" not in rows[0]


@pytest.mark.parametrize("rho", ["4", "2"])
def test_frozen_floor_matches_rho(rho):
    p = REPO / "experiments" / "items" / "test" / "CS" / f"A-freq_rho{rho}_1K.npz"
    if not p.is_file():
        pytest.skip("frozen items not built")
    items, _ = load_items(p)
    h = [_afreq_heuristics(it, q, "A-freq") for it in items for q in it["queries"]]
    rec = float(np.mean([x["recency_heuristic_correct"] for x in h]))
    r = float(rho)
    print(f"rho={rho}: recency floor {rec:.3f} vs rho/(1+rho) {r / (1 + r):.3f}")
    assert abs(rec - r / (1 + r)) < 0.06


# Verifier's windowed-count numbers on the frozen CS items (recency / last-256 / last-1024 / last-4096)
WINDOW_EXPECTED = {
    "A-freq_rho4_1K": (0.802, 0.991, 1.000, 1.000),
    "A-freq_rho4_4K": (0.802, 0.850, 0.991, 1.000),
    "A-freq_rho2_4K": (0.667, 0.659, 0.818, 1.000),
    "A-freq_rho4_32K": (0.791, 0.578, 0.721, 0.926),
    "A-freq_rho2_32K": (0.667, 0.537, 0.609, 0.729),
}


@pytest.mark.parametrize("cell", sorted(WINDOW_EXPECTED))
def test_frozen_window_count_floor(cell):
    p = REPO / "experiments" / "items" / "test" / "CS" / f"{cell}.npz"
    if not p.is_file():
        pytest.skip("frozen items not built")
    items, _ = load_items(p)
    h = [_afreq_heuristics(it, q, "A-freq") for it in items for q in it["queries"]]
    got = (float(np.mean([x["recency_heuristic_correct"] for x in h])),
           *(float(np.mean([x[f"window{W}_count_heuristic"] for x in h])) for W in (256, 1024, 4096)))
    assert np.allclose(got, WINDOW_EXPECTED[cell], atol=0.0015), (cell, got)

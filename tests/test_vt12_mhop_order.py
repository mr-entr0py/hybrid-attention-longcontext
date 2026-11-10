"""VT-12 regression (verification round 1, MHOP-C0-C2-UNSHUFFLED).

M-hop 'shuffled' order must be realised for every placement. In the no-filler C0 control
(F = 0) and the oracle C2 control (every needle at F) all targets coincide, so the order has
to come from an explicit tie-break rank; the S0 code fell back to generator input order
(each chain contiguous and forward, chains in query order), which a purely local heuristic
("take the value fact nearest before the queried head's link") solved for 100% of items.

Checks, per placement in {spread, oracle, nofill}:
* 'shuffled': the heuristic is near chance (≤ 0.45; chance 0.25), chains are not all
  contiguous, and chain order is not the forward order in most items;
* 'forward' / 'reversed': each chain is in the requested order, and chains are interleaved
  in some items (not just concatenated in input order);
* the frozen C0/C2 cells on disk (if present) pass the same heuristic bound.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hybridlc.eval.context import assemble
from hybridlc.eval.devtools import resolve_test_tokenizer
from hybridlc.eval.filler import StoreFiller, dev_text_filler
from hybridlc.eval.generators import ROLE_LINK, ROLE_QUERIED, generate
from hybridlc.eval.items import load_items
from hybridlc.eval.rng import eval_rng
from hybridlc.eval.tokens import token_space

REPO = Path(__file__).resolve().parents[1]
TOKENS_REAL = (REPO / "data" / "tokens" / "pg19" / "test.bin").exists()
K = 1024


@pytest.fixture(scope="module")
def tok():
    return resolve_test_tokenizer()[0]


@pytest.fixture(scope="module")
def filler(tok):
    ts = token_space(tok)
    return StoreFiller(ts, "test") if TOKENS_REAL else dev_text_filler(ts, "test")


def _nearest_before_value_acc(items: list[dict]) -> float:
    """Accuracy of: answer = value of the value fact closest before the head link."""
    hits = []
    for it in items:
        a = it["arrays"]
        fk, fv, fs, fr = (np.asarray(a[k]) for k in ("fact_key", "fact_value", "fact_start", "fact_role"))
        vidx = np.flatnonzero(fr == ROLE_QUERIED)
        lidx = np.flatnonzero(fr == ROLE_LINK)
        for q in it["queries"]:
            st = q["meta"]["needle_start"]  # value fact, link 1..h
            head_pos = st[-1]
            before = [j for j in vidx if fs[j] < head_pos]
            pick = max(before, key=lambda j: fs[j]) if before else vidx[np.argmin(fs[vidx])]
            hits.append(int(fv[pick]) == int(q["meta"]["value"]))
        assert len(lidx) > 0
    return float(np.mean(hits))


def _chain_stats(items: list[dict]) -> tuple[float, float]:
    """(fraction of chains in forward order, fraction of items with every chain contiguous)."""
    fwd, contig = [], []
    for it in items:
        starts_all = sorted(s for q in it["queries"] for s in q["meta"]["needle_start"])
        rank = {s: r for r, s in enumerate(starts_all)}
        all_contig = True
        for q in it["queries"]:
            st = q["meta"]["needle_start"]
            fwd.append(st == sorted(st))
            r = sorted(rank[s] for s in st)
            all_contig &= r == list(range(r[0], r[0] + len(r)))
        contig.append(all_contig)
    return float(np.mean(fwd)), float(np.mean(contig))


@pytest.mark.parametrize("placement,T", [("spread", 4 * K), ("oracle", 4 * K), ("nofill", None)])
def test_shuffled_order_realised(tok, filler, placement, T):
    lvl = {"h": 3, "placement": placement}
    items = generate("M-hop", lvl, T, 60, eval_rng("test", "vt12-mhop-order", placement), "test", tok, filler)
    acc = _nearest_before_value_acc(items)
    fwd, contig = _chain_stats(items)
    print(f"{placement}: nearest-before={acc:.3f} forward-chains={fwd:.3f} all-contiguous={contig:.3f}")
    assert acc <= 0.45, acc  # chance 0.25; the S0 C0/C2 bug gave 1.0
    assert fwd < 0.3, fwd  # P(random order of 4 needles is forward) = 1/24
    assert contig < 0.2, contig


@pytest.mark.parametrize("order", ["forward", "reversed"])
@pytest.mark.parametrize("placement,T", [("spread", 4 * K), ("oracle", 4 * K), ("nofill", None)])
def test_ordered_chains_interleaved(tok, filler, placement, T, order):
    lvl = {"h": 2, "placement": placement, "order": order}
    items = generate("M-hop", lvl, T, 40, eval_rng("test", "vt12-mhop-order", placement, order), "test", tok, filler)
    for it in items:
        for q in it["queries"]:
            st = q["meta"]["needle_start"]
            assert st == (sorted(st) if order == "forward" else sorted(st, reverse=True)), (placement, order, st)
    _, contig = _chain_stats(items)
    assert contig < 0.5, contig  # chains interleaved, not concatenated in input order


def test_assemble_tie_rank(tok, filler):
    """Equal targets are ordered by tie_rank; without it, by input order (old behaviour)."""
    ts = token_space(tok)
    needles = [np.asarray([100 + i] * 3, dtype=np.int32) for i in range(5)]
    fill = np.zeros(0, dtype=np.int32)
    p = assemble(ts, fill, needles, np.zeros(5), do_snap=False)
    assert list(np.argsort(p.starts)) == [0, 1, 2, 3, 4]
    rank = np.asarray([3.0, 0.5, 4.0, 1.0, 2.0])
    p = assemble(ts, fill, needles, np.zeros(5), do_snap=False, tie_rank=rank)
    assert list(np.argsort(p.starts)) == list(np.argsort(rank))
    with pytest.raises(ValueError):
        assemble(ts, fill, needles, np.zeros(5), do_snap=False, tie_rank=rank[:3])


FROZEN = ["CS/M-hop-C0_h1_nofill.npz", "CS/M-hop-C0_h2_nofill.npz", "CS/M-hop-C0_h3_nofill.npz",
          "CS/M-hop-C0_h4_nofill.npz", "MS/M-hop-C2_h2_4K.npz", "MS/M-hop-C2_h2_32K.npz"]


@pytest.mark.parametrize("rel", FROZEN)
def test_frozen_c0_c2_cells_shuffled(rel):
    p = REPO / "experiments" / "items" / "test" / rel
    if not p.is_file():
        pytest.skip("frozen items not built")
    items, _ = load_items(p)
    acc = _nearest_before_value_acc(items)
    fwd, contig = _chain_stats(items)
    print(f"{rel}: nearest-before={acc:.3f} forward={fwd:.3f} contiguous={contig:.3f}")
    assert acc <= 0.40, acc
    if "h1" not in rel:  # h=1: a 2-needle chain is forward with prob 1/2
        assert fwd < 0.3, fwd
    assert contig < 0.2, contig

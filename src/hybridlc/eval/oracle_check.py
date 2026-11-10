"""Independent G-ORACLE checker for retrieval items (VT-24r v2, PI ruling D074 / PR-17).

Rule G-ORACLE (``experiments/prereg/VT-24r.v2.yaml``): for every queried retrieval fact (R1, R-load,
R3, R-dist; every variant) and every answer digit ``d_j`` (j = 1..4), in the token sequence
``context + prompt + d_1 .. d_{j-1}`` the most recent earlier occurrence of the ``order``-token
window (6 by default) that ends immediately before ``d_j`` lies inside the queried fact's own
occurrence (for R3: its latest update).

This module is deliberately a direct token-window search over numpy arrays. It must not import
``hybridlc.eval.devtools``, ``hybridlc.eval.scorer`` or any model / LM class, and it never calls
``score_items`` / ``score_file`` (an import-graph test enforces this): VT-24r's K1 scores the same
items with ``devtools.InductionLM`` + ``scorer.score_file``, so the repair loop (which uses this
checker) and K1 rest on two independent implementations.

Per queried fact and answer digit two predicates are computed:

* ``pred_ok``: an earlier occurrence of the window exists and the token that follows the most
  recent one equals ``d_j`` (the semantics of VT-24's InductionLM oracle);
* ``loc_ok``: rule G-ORACLE: the most recent earlier occurrence (window start and the following
  token) lies inside ``[fact_start, fact_end)``.

``loc_ok`` implies ``pred_ok`` (the fact continues with its own digit); ``pred_ok`` without
``loc_ok`` is a *coincidental match* (another span continues with the same digit). The repair
enforces ``loc_ok``.
"""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np

#: Generator tasks whose queries are key-value retrieval (rule scope).
RETRIEVAL_TASKS: tuple[str, ...] = ("R1", "R-load", "R3", "R-dist")
#: Window length of VT-24's oracle (InductionLM order 6).
ORACLE_ORDER = 6


def most_recent_occurrence(seq: np.ndarray, pos: int, order: int = ORACLE_ORDER) -> int:
    """End index ``s`` of the most recent occurrence of ``seq[pos-order:pos]`` that ends strictly
    before ``pos - 1`` (the window's own end), or -1 if there is none."""
    if pos < order:
        return -1
    seq = np.asarray(seq)
    w = seq[pos - order:pos]
    last = int(w[-1])
    # candidate ends s in [order-1, pos-2] whose token equals the window's last token
    cand = np.flatnonzero(seq[order - 1:pos - 1] == last) + (order - 1)
    for s in cand[::-1]:
        s = int(s)
        if np.array_equal(seq[s - order + 1:s + 1], w):
            return s
    return -1


def check_digit(seq: np.ndarray, pos: int, fact_start: int, fact_end: int,
                order: int = ORACLE_ORDER) -> dict[str, Any]:
    """Predicates for the gold token ``seq[pos]`` against fact span ``[fact_start, fact_end)``."""
    s = most_recent_occurrence(seq, pos, order)
    pred_ok = bool(s >= 0 and int(seq[s + 1]) == int(seq[pos]))
    loc_ok = bool(s >= 0 and s - order + 1 >= fact_start and s + 1 < fact_end)
    return {"pos": int(pos), "occ_end": int(s), "pred_ok": pred_ok, "loc_ok": loc_ok}


def _query_layout(item: dict) -> Iterable[tuple[int, dict, np.ndarray, int]]:
    """Yield (query index, query, token sequence, offset of the first answer token) per query.

    Branched (eval) items: ``context + prompt_q + answer_q``. Sequential (TEF) items: the
    flattened ``context + prompt_1 + answer_1 + ... + prompt_q + answer_q``.
    """
    ctx = np.asarray(item["context_ids"], dtype=np.int64)
    sequential = item.get("meta", {}).get("query_mode") == "sequential"
    prev: list[np.ndarray] = [ctx]
    for qi, q in enumerate(item["queries"]):
        p = np.asarray(q["prompt_ids"], dtype=np.int64)
        a = np.asarray(q["answer_ids"], dtype=np.int64)
        head = np.concatenate(prev + [p]) if sequential else np.concatenate([ctx, p])
        yield qi, q, np.concatenate([head, a]), len(head)
        if sequential:
            prev += [p, a]


def check_item(item: dict, order: int = ORACLE_ORDER) -> list[dict[str, Any]]:
    """Per-query G-ORACLE record of a retrieval item (empty list for other tasks).

    The queried fact's span is the first entry of the query's ``needle_start`` / ``needle_end``
    (the queried fact; for R3 its latest update, followed by the stale writes)."""
    task = item.get("meta", {}).get("task")
    if task not in RETRIEVAL_TASKS:
        return []
    out = []
    for qi, q, seq, a0 in _query_layout(item):
        m = q["meta"]
        fs, fe = int(m["needle_start"][0]), int(m["needle_end"][0])
        digits = [check_digit(seq, a0 + j, fs, fe, order) for j in range(len(q["answer_ids"]))]
        out.append({"query_idx": int(m.get("query_idx", qi)), "fact_start": fs, "fact_end": fe,
                    "digits": digits, "loc_ok": all(d["loc_ok"] for d in digits),
                    "pred_ok": all(d["pred_ok"] for d in digits)})
    return out


def item_ok(item: dict, order: int = ORACLE_ORDER) -> bool:
    """True iff every queried fact of the item satisfies rule G-ORACLE."""
    return all(r["loc_ok"] for r in check_item(item, order))


def summarize(records: list[dict[str, Any]]) -> dict[str, int]:
    """Counts over query records: queries, pred misses, loc misses, coincidental matches,
    and checker-defect cases (loc_ok without pred_ok, which must never happen)."""
    n = len(records)
    pred_miss = sum(not r["pred_ok"] for r in records)
    loc_miss = sum(not r["loc_ok"] for r in records)
    coincid = sum(1 for r in records for d in r["digits"] if d["pred_ok"] and not d["loc_ok"])
    defect = sum(1 for r in records for d in r["digits"] if d["loc_ok"] and not d["pred_ok"])
    return {"queries": n, "pred_miss_queries": pred_miss, "loc_miss_queries": loc_miss,
            "coincidental_digit_matches": coincid, "checker_defect_digits": defect}


__all__ = ["RETRIEVAL_TASKS", "ORACLE_ORDER", "most_recent_occurrence", "check_digit", "check_item", "item_ok",
           "summarize"]

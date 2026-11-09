"""Tier-0 zero-shot items (protocol §9.6): LM-1 (+ LM-3 annotations), LM-2, LM-4.

R0 (cloze passkey) is a cloze task and lives in ``eval.generators.gen_r0``.

All items are ``Item`` dicts with an empty ``queries`` list; the sequence to score is
``context_ids`` and the scoring annotations are in ``meta`` / ``arrays``. They are scored
with ``eval.scorer.score_lm_file`` (one full forward per item, per-token NLL).

* **LM-1**: up to 300 FineWeb-Edu shard-000 *test-half* documents with >= 32,768 tokens
  (count reported in the cell meta) plus 100 PG19-test books, first 32,768 tokens each
  (the leading ``<|endoftext|>`` included, exactly as in training).
* **LM-3** (annotations on the LM-1 items): positions t whose 4-gram ``x[t-3..t]``
  occurred earlier in the same document, the most recent earlier occurrence ending at
  ``t - Delta``, with ``Delta in [64, 32K)``; 4-grams among the top 1% most frequent 4-gram
  *types* of a 100M-token training sample are excluded (``build_lm3_banned``). The loss of
  token ``x[t]`` is attributed to the Delta bucket.
* **LM-2**: the 2,000 x 4K held-out sequences of ``StageStream('LM2')`` (packed like Stage A;
  labels and segment starts stored).
* **LM-4**: ``[EOT] + P + G + P``; P = a 256-token held-out passage (variant a) or 256 random
  tokens from a fixed 4,096-token vocabulary subset (variant b); gap G of natural held-out
  filler, G in {0, 512, 2K, 8K, 30K}; 200 items per gap per variant. Copy gain = loss on
  tokens 9..256 of the first P minus the same on the second P.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from hybridlc.eval.constants import (LM1_MAX_FWE_DOCS, LM1_N_PG19_BOOKS, LM1_SEQ, LM3_DELTA_EDGES, LM3_NGRAM,
                                     LM3_TOP_FRACTION_EXCLUDED, LM4_P_LEN, LM4_RANDOM_VOCAB)
from hybridlc.eval.filler import NATURAL_KINDS, FillerProvider
from hybridlc.eval.rng import child_rng
from hybridlc.eval.tokens import token_space

Item = dict[str, Any]


# --------------------------------------------------------------------------- LM-3 helpers
def ngram_codes(x: np.ndarray, n: int = LM3_NGRAM) -> np.ndarray:
    """uint64 code of the n-gram ending at each position t >= n-1 (16 bits per token; vocab < 65536)."""
    x = np.asarray(x, dtype=np.uint64)
    if len(x) < n:
        return np.zeros(0, np.uint64)
    c = np.zeros(len(x) - n + 1, dtype=np.uint64)
    for j in range(n):
        c = (c << np.uint64(16)) | x[j:len(x) - n + 1 + j]
    return c


def build_lm3_banned(tokens: np.ndarray, top_fraction: float = LM3_TOP_FRACTION_EXCLUDED) -> np.ndarray:
    """Sorted uint64 codes of the top ``top_fraction`` most frequent 4-gram types in ``tokens``.

    Ties at the cut are broken by code (deterministic). ``tokens`` should be a 100M-token
    training sample (protocol §9.6); 4-grams spanning a separator are dropped.
    """
    codes = ngram_codes(tokens)
    x = np.asarray(tokens)
    has_sep = np.zeros(len(codes), bool)
    sep = np.flatnonzero(x == 0)
    for j in range(LM3_NGRAM):
        s = sep - j
        s = s[(s >= 0) & (s < len(codes))]
        has_sep[s] = True
    codes = codes[~has_sep]
    uniq, cnt = np.unique(codes, return_counts=True)
    k = int(np.ceil(top_fraction * len(uniq)))
    order = np.lexsort((uniq, -cnt))[:k]
    return np.sort(uniq[order])


def lm3_annotations(doc: np.ndarray, banned: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """(positions t, Delta) of repeated 4-grams with Delta in [64, 32K) (see module docstring)."""
    codes = ngram_codes(doc)
    if len(codes) == 0:
        return np.zeros(0, np.int32), np.zeros(0, np.int32)
    pos = np.arange(len(codes)) + LM3_NGRAM - 1  # position of the 4-gram's last token
    order = np.lexsort((pos, codes))
    sc = codes[order]
    same = sc[1:] == sc[:-1]
    prev = np.full(len(codes), -1, dtype=np.int64)
    prev[order[1:][same]] = pos[order[:-1][same]]
    delta = np.where(prev >= 0, pos - prev, 0)
    keep = (prev >= 0) & (delta >= LM3_DELTA_EDGES[0]) & (delta < LM3_DELTA_EDGES[-1])
    if banned is not None and len(banned):
        keep &= ~np.isin(codes, banned)
    return pos[keep].astype(np.int32), delta[keep].astype(np.int32)


# --------------------------------------------------------------------------- generators
def gen_lm1(task_level: dict, T: int | None, n: int, rng: np.random.Generator, namespace: str, tok: Any,
            filler: FillerProvider | None = None, banned: np.ndarray | None = None) -> list[Item]:
    """LM-1 items (``n`` is ignored: the protocol fixes up to 300 FWE + 100 PG19)."""
    ts = token_space(tok)
    filler = filler or _store_filler(ts, namespace)
    L = int(T or LM1_SEQ)
    out: list[Item] = []
    fwe, pg = filler.pools["fwe"], filler.pools["pg19"]
    # pool lengths exclude the leading EOT; an item is [EOT] + first L-1 tokens
    elig = np.flatnonzero(fwe.lengths + 1 >= L)
    n_fwe = min(LM1_MAX_FWE_DOCS, len(elig))
    pick = np.sort(child_rng(rng, "fwe").choice(elig, size=n_fwe, replace=False)) if n_fwe else elig[:0]
    books = np.flatnonzero(pg.lengths + 1 >= L)
    n_pg = min(int(task_level.get("n_pg19", LM1_N_PG19_BOOKS)), len(books))
    books = books[:n_pg]
    for src, pool, ids in (("fwe", fwe, pick), ("pg19", pg, books)):
        for d in ids.tolist():
            seq = np.concatenate([[ts.eot_id], pool.doc(d)[:L - 1]]).astype(np.int32)
            pos, delta = lm3_annotations(seq, banned)
            out.append(dict(context_ids=seq, queries=[],
                            meta=dict(task="LM-1", context_id=len(out), T=L, filler_source=dict(kind=src, pool=pool.name, doc=d),
                                      namespace=namespace, lm3_banned=banned is not None),
                            arrays=dict(lm3_pos=pos, lm3_delta=delta)))
    return out


def gen_lm2(task_level: dict, T: int | None, n: int, rng: np.random.Generator, namespace: str, tok: Any,
            filler: FillerProvider | None = None, root: str | Path | None = None) -> list[Item]:
    """LM-2: the fixed 2,000 x 4K held-out sequences of ``StageStream('LM2')`` (rng unused)."""
    from hybridlc.data.loader import StageStream
    from hybridlc.data.sources import TOKENS_DIR

    s = StageStream("LM2", 0, root=root or TOKENS_DIR)
    n = min(int(n), s.n_sequences)
    out: list[Item] = []
    step = 0
    while len(out) < n:
        b = s.get_numpy(step, 0)
        L = s.seq_len
        cu = b["cu_seqlens"]
        for j in range(s.micro_batch):
            if len(out) >= n:
                break
            seg = cu[(cu >= j * L) & (cu < (j + 1) * L)] - j * L
            out.append(dict(context_ids=b["input_ids"][0, j * L:(j + 1) * L].astype(np.int32), queries=[],
                            meta=dict(task="LM-2", context_id=len(out), T=L, namespace=namespace,
                                      filler_source=dict(kind=b["sources"][j])),
                            arrays=dict(labels=b["labels"][0, j * L:(j + 1) * L].astype(np.int32),
                                        seg_starts=seg.astype(np.int32))))
        step += 1
    return out


def lm4_vocab(tok: Any, namespace: str) -> np.ndarray:
    """The fixed 4,096-token subset for LM-4 variant (b): ordinary (non-special) ids, eval RNG."""
    from hybridlc.eval.rng import NAMESPACE_ENTROPY

    ts = token_space(tok)
    special = set(getattr(tok, "all_special_ids", []) or []) | {ts.eot_id}
    ids = np.asarray([i for i in range(ts.vocab_size) if i not in special], dtype=np.int64)
    g = np.random.Generator(np.random.PCG64(np.random.SeedSequence(NAMESPACE_ENTROPY[namespace],
                                                                   spawn_key=(4, 4096))))
    return np.sort(g.choice(ids, size=LM4_RANDOM_VOCAB, replace=False)).astype(np.int32)


def gen_lm4(task_level: dict, T: int | None, n: int, rng: np.random.Generator, namespace: str, tok: Any,
            filler: FillerProvider | None = None) -> list[Item]:
    """LM-4 copy-after-gap; ``task_level = {'variant': 'a'|'b', 'G': gap}``; T = 1 + 2*256 + G."""
    ts = token_space(tok)
    filler = filler or _store_filler(ts, namespace)
    var, G = task_level["variant"], int(task_level["G"])
    P = LM4_P_LEN
    vocab = lm4_vocab(tok, namespace) if var == "b" else None
    out: list[Item] = []
    for i in range(n):
        irng = child_rng(rng, i)
        if var == "a":
            p, psrc = filler.get(NATURAL_KINDS[int(irng.integers(2))], P, irng)
        elif var == "b":
            p, psrc = vocab[irng.integers(len(vocab), size=P)].astype(np.int32), dict(kind="random_vocab4096")
        else:
            raise ValueError(f"LM-4 variant must be a|b, got {var!r}")
        g, gsrc = filler.get(NATURAL_KINDS[int(irng.integers(2))], G, irng)
        seq = np.concatenate([[ts.eot_id], p, g, p]).astype(np.int32)
        out.append(dict(context_ids=seq, queries=[],
                        meta=dict(task="LM-4", context_id=i, T=len(seq), variant=var, G=G, p1_start=1,
                                  p2_start=1 + P + G, p_len=P, filler_source=dict(passage=psrc, gap=gsrc),
                                  namespace=namespace),
                        arrays={}))
    return out


def _store_filler(ts, namespace):
    from hybridlc.eval.filler import StoreFiller

    return StoreFiller(ts, namespace)


TIER0_GENERATORS = {"LM-1": gen_lm1, "LM-2": gen_lm2, "LM-4": gen_lm4}

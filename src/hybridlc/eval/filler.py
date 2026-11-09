"""Filler providers for evaluation contexts and TEF items (protocol §6.1, §7.3, §9.1).

A provider hands out *exactly n* filler tokens of one kind:

* ``"pg19"``: one contiguous span of a single book (no separator inside). The span *ends*
  at a sentence boundary (snapped within +-``SNAP_WINDOW`` tokens), so the query that
  follows the context starts a new sentence.
* ``"fwe"``: FineWeb-Edu documents joined by ``"\\n\\n"`` (the last n tokens of a random
  sequence of whole documents, so the span ends at a document end).
* ``"repetitive"``: RULER's noise sentence block repeated (the last n tokens, ending at the
  end of a block).

Natural filler is "pg19" or "fwe", 50/50 per context; the *generator* draws that choice so
that it is part of the item's seeded randomness, and records the returned ``source`` dict.

Splits (protocol §6.1): namespace ``test`` -> pg19/test + fwe/test (SHA-256-even half of
FineWeb-Edu shard 000); ``dev`` -> pg19/validation + fwe/dev; ``train`` (TEF) ->
pg19/train + fwe/train. ``StoreFiller`` reads them from ``data/tokens``; ``TextFiller``
tokenises plain-text documents (tests / development while ``data/tokens`` is being built).
Filler never contains ``<|endoftext|>`` (asserted).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from hybridlc.eval.constants import FWE_JOINER, REPETITIVE_FILLER, SNAP_WINDOW
from hybridlc.eval.tokens import TokenSpace

#: Corpus splits feeding each namespace (protocol §6.1).
NAMESPACE_SPLITS: dict[str, dict[str, tuple[str, str]]] = {
    "test": {"pg19": ("pg19", "test"), "fwe": ("fwe", "test")},
    "dev": {"pg19": ("pg19", "validation"), "fwe": ("fwe", "dev")},
    "train": {"pg19": ("pg19", "train"), "fwe": ("fwe", "train")},
}
NATURAL_KINDS = ("pg19", "fwe")


class FillerProvider(Protocol):
    """What generators need: ``get(kind, n, rng) -> (tokens int32[n], source dict)``."""

    ts: TokenSpace
    namespace: str

    def get(self, kind: str, n: int, rng: np.random.Generator) -> tuple[np.ndarray, dict]: ...


class DocPool(Protocol):
    """A list of documents (token arrays *without* the leading separator)."""

    name: str
    lengths: np.ndarray  # int64[n_docs]

    def doc(self, i: int) -> np.ndarray: ...


@dataclass
class ArrayPool:
    """In-memory ``DocPool``."""

    name: str
    docs: list[np.ndarray]

    def __post_init__(self) -> None:
        self.lengths = np.asarray([len(d) for d in self.docs], dtype=np.int64)

    def doc(self, i: int) -> np.ndarray:
        return self.docs[i]


class StorePool:
    """``DocPool`` view of ``data/tokens/<corpus>/<split>`` (strips each document's leading EOT)."""

    def __init__(self, corpus: str, split: str, root: str | Path):
        from hybridlc.data.loader import get_store

        self.store = get_store(corpus, split, str(root))
        self.name = f"{corpus}/{split}"
        self.lengths = self.store.lengths - 1

    def doc(self, i: int) -> np.ndarray:
        return np.asarray(self.store.doc(int(i))[1:], dtype=np.int32)


def boundaries(ts: TokenSpace, toks: np.ndarray, edges: bool = True) -> np.ndarray:
    """Sentence-boundary offsets o in 0..len: token o-1 ends a sentence and token o starts with
    whitespace (or o == len). With ``edges``, o = 0 and o = len always count (start / end of filler)."""
    n = len(toks)
    if n == 0:
        return np.zeros(1, np.int64)
    es = ts.ends_sentence[toks]
    nxt = np.r_[ts.starts_space[toks[1:]], True]  # token after position j (end counts)
    b = np.flatnonzero(es & nxt) + 1
    if edges:
        b = np.unique(np.r_[0, b, n])
    return b.astype(np.int64)


def snap(bpos: np.ndarray, o: int, window: int = SNAP_WINDOW, lo: int = 0, hi: int | None = None) -> tuple[int, bool]:
    """Nearest boundary to ``o`` within +-window (and within [lo, hi]); ``(o, False)`` if none."""
    i = int(np.searchsorted(bpos, o))
    best, bd = o, window + 1
    for j in (i - 1, i):
        if 0 <= j < len(bpos):
            b = int(bpos[j])
            if abs(b - o) < bd and b >= lo and (hi is None or b <= hi):
                best, bd = b, abs(b - o)
    return (best, True) if bd <= window else (o, False)


class _PoolFiller:
    """Shared span logic for any pair of DocPools."""

    def __init__(self, ts: TokenSpace, namespace: str, pools: dict[str, DocPool]):
        self.ts, self.namespace, self.pools = ts, namespace, pools
        self._joiner = ts.enc(FWE_JOINER)
        unit = ts.enc(" " + REPETITIVE_FILLER)
        self._rep_unit = unit
        self._long_cache: dict[tuple[str, int], np.ndarray] = {}

    # -- kinds
    def get(self, kind: str, n: int, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
        n = int(n)
        if n < 0:
            raise ValueError(f"negative filler length {n}")
        if kind == "repetitive":
            out, src = self._repetitive(n), dict(kind="repetitive")
        elif kind == "pg19":
            out, src = self._single_span(self.pools["pg19"], n, rng)
        elif kind == "fwe":
            out, src = self._joined(self.pools["fwe"], n, rng)
        else:
            raise ValueError(f"unknown filler kind {kind!r}")
        out = np.asarray(out, dtype=np.int32)
        assert len(out) == n, (kind, len(out), n)
        if (out == self.ts.eot_id).any():
            raise AssertionError("separator token inside filler")
        return out, src

    def _repetitive(self, n: int) -> np.ndarray:
        u = self._rep_unit
        reps = n // len(u) + 2
        return np.tile(u, reps)[len(u) * reps - n:] if n else np.zeros(0, np.int32)

    def _eligible(self, pool: DocPool, n: int) -> np.ndarray:
        key = (pool.name, n)
        e = self._long_cache.get(key)
        if e is None:
            e = np.flatnonzero(pool.lengths >= n)
            self._long_cache[key] = e
        return e

    def _single_span(self, pool: DocPool, n: int, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
        """One span of one document, ending at a sentence boundary when possible."""
        if n == 0:
            return np.zeros(0, np.int32), dict(kind="pg19", pool=pool.name, doc=-1, start=0, end=0)
        elig = self._eligible(pool, n)
        if len(elig) == 0:  # no single document is long enough (tiny test pools only)
            out, src = self._joined(pool, n, rng)
            src.update(kind="pg19", fallback="joined")
            return out, src
        d = int(elig[rng.integers(len(elig))])
        L = int(pool.lengths[d])
        e = int(rng.integers(n, L + 1))
        doc = pool.doc(d)
        lo_w = max(0, e - SNAP_WINDOW - 1)
        hi_w = min(L, e + SNAP_WINDOW + 1)
        bpos = boundaries(self.ts, doc[lo_w:hi_w], edges=False) + lo_w
        if hi_w == L:
            bpos = np.r_[bpos, L]
        e2, ok = snap(bpos, e, lo=n, hi=L)
        return doc[e2 - n:e2], dict(kind="pg19", pool=pool.name, doc=d, start=e2 - n, end=e2, end_snapped=ok)

    def _joined(self, pool: DocPool, n: int, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
        """Last n tokens of random whole documents joined by the joiner."""
        if n == 0:
            return np.zeros(0, np.int32), dict(kind="fwe", pool=pool.name, docs=[])
        parts, docs, tot = [], [], 0
        nd = len(pool.lengths)
        while tot < n:
            d = int(rng.integers(nd))
            if pool.lengths[d] == 0:
                continue
            if parts:
                parts.append(self._joiner)
                tot += len(self._joiner)
            x = pool.doc(d)
            parts.append(x)
            docs.append(d)
            tot += len(x)
        cat = np.concatenate(parts)
        first_off = len(cat) - n  # offset into the first document where the span starts
        return cat[first_off:], dict(kind="fwe", pool=pool.name, docs=docs, first_offset=int(first_off))


class StoreFiller(_PoolFiller):
    """Filler from ``data/tokens`` per the protocol §6.1 split of ``namespace``.

    The token store must have been built with the same tokenizer as ``ts`` (checked via
    ``<split>.meta.json['tokenizer_sha256']`` when present).
    """

    def __init__(self, ts: TokenSpace, namespace: str, root: str | Path | None = None):
        from hybridlc.data.sources import TOKENS_DIR

        root = Path(root or TOKENS_DIR)
        pools = {}
        for kind, (corpus, split) in NAMESPACE_SPLITS[namespace].items():
            pools[kind] = StorePool(corpus, split, root)
            _check_store_tokenizer(root / corpus / f"{split}.meta.json", ts)
        super().__init__(ts, namespace, pools)


class TextFiller(_PoolFiller):
    """Filler from plain-text documents (tests / development stand-in for the held-out stores).

    ``pg19_texts`` should be long documents (book-like), ``fwe_texts`` web-like documents.
    """

    def __init__(self, ts: TokenSpace, namespace: str, pg19_texts: list[str], fwe_texts: list[str]):
        enc = lambda texts: [np.asarray(x, np.int32) for x in  # noqa: E731
                             ts.tok(list(texts), add_special_tokens=False)["input_ids"]]
        pools = {"pg19": ArrayPool(f"text-pg19/{namespace}", enc(pg19_texts)),
                 "fwe": ArrayPool(f"text-fwe/{namespace}", enc(fwe_texts))}
        for p in pools.values():
            for d in p.docs:
                if (d == ts.eot_id).any():
                    raise ValueError("a filler text contains the separator token")
        super().__init__(ts, namespace, pools)


def _check_store_tokenizer(meta_path: Path, ts: TokenSpace) -> None:
    import json

    if not meta_path.is_file():
        return
    meta = json.loads(meta_path.read_text())
    want = meta.get("tokenizer_sha256")
    if want and ts.fingerprint and want != ts.fingerprint:
        raise RuntimeError(f"{meta_path}: token store built with tokenizer {want[:12]}, "
                           f"but the item tokenizer is {ts.fingerprint[:12]}")


def dev_text_filler(ts: TokenSpace, namespace: str) -> TextFiller:
    """A ``TextFiller`` from the C4 test asset (devtools.english_texts), split by namespace.

    Documents are assigned to namespaces by index (train / dev / test thirds) and "books" are
    made by joining 40 consecutive documents with a space, so a 32K span fits in one "book".
    """
    from hybridlc.eval.devtools import english_texts

    docs = english_texts()
    third = {"train": 0, "dev": 1, "test": 2}[namespace]
    mine = [d for i, d in enumerate(docs) if i % 3 == third]
    books = [" ".join(mine[i:i + 40]) for i in range(0, len(mine), 40)]
    return TextFiller(ts, namespace, books, mine)


def describe_source(src: dict[str, Any]) -> str:
    """Short filler-source label for result rows ("pg19", "fwe", "repetitive", "none", "list")."""
    return str(src.get("kind", "none"))

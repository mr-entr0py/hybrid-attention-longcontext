"""Stateless training/validation streams (protocol §6.3-6.4, §7.2, §9.6 LM-2).

``StageStream.get(step, micro)`` is a pure function of
``(data_seed, stage recipe, step, micro)``; nothing has to be saved on resume.

Construction of one micro-batch
-------------------------------
* A *step* holds ``micro_batch * accum`` sequences of ``seq_len`` tokens; global
  sequence slot ``g = (step * accum + micro) * micro_batch + j``.
* Slots are grouped in *mixing windows* of ``window_sequences`` slots
  (Stage A: 640 = 20 steps x 32; Stage B: 20 = 5 steps x 4). In window ``w``
  exactly ``pg19_per_window`` slots (96 / 12) are PG19 windows and the rest are
  packed FineWeb-Edu. Which slots are PG19 is a uniformly random subset drawn
  from ``SeedSequence(PROJECT_SALT, (data_seed, stream, 2, w))``: the
  "deterministic interleave keyed on (data_seed, step)" of §6.3.
* **FineWeb-Edu** slots read consecutive ``seq_len`` slices of one infinite
  stream: the training documents (each ``[EOT] + tokens``) concatenated in a
  random order, re-permuted every epoch (key ``(data_seed, stream, 0, epoch)``).
  The k-th FWE slot of the run is stream tokens ``[k*L, (k+1)*L)``. A document
  that crosses a sequence boundary is cut there; each piece is its own segment
  (§6.4). No tokens are inserted or dropped.
* **PG19** slots are book-bounded windows: every training book (``[EOT] + text``)
  is cut into ``floor(len / L)`` aligned windows of exactly ``L`` tokens (the
  tail shorter than L is unused). Windows are visited in a random order per epoch
  (key ``(data_seed, stream, 1, epoch)``). A window is one segment.
* ``cu_seqlens`` has a boundary at every document start (EOT) and at every
  sequence start; ``position_ids`` restart at 0 at each boundary.
* ``labels[t]`` is the next token *of the source stream* (so the last token of a
  document predicts the next ``<|endoftext|>`` - separators are predicted, §7.1;
  the last token of a sequence predicts its true continuation). Positions with no
  next token (end of a file) get ``IGNORE_INDEX`` (= pad id 1, the ``ignore_index``
  of ``models.loss.chunked_ce``).
* ``batch_hash`` = SHA-256 over (input_ids, labels, cu_seqlens) bytes - identical
  for every k at a given (data_seed, step, micro).

Stages: ``'A'`` (stream data_A), ``'LR'`` (the Stage-A recipe; the LR sweep runs
it at data_seed 100, §7.2), ``'B'`` (stream data_B), ``'LM2'`` (2,000 x 4K
sequences of the FWE held-out *test* half, packed like Stage A's FWE part, in a
fixed order from the eval_test entropy; independent of data_seed).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from hybridlc.data import rng as R
from hybridlc.data.sources import TOKENS_DIR
from hybridlc.data.tokenizer import EOT_ID, PAD_ID

IGNORE_INDEX = PAD_ID  # chunked_ce(ignore_index=1)

#: Protocol-derived stage recipes (§6.3, §7.2, §9.6). Configs may override
#: seq_len / micro_batch / accum / mix, subject to the exactness checks below.
STAGE_DEFAULTS: dict[str, dict] = {
    "A": dict(seq_len=4096, micro_batch=2, accum=16, stream="data_A",
              mix=dict(pg19_per_window=96, window_sequences=640)),
    "LR": dict(seq_len=4096, micro_batch=2, accum=16, stream="data_A",
               mix=dict(pg19_per_window=96, window_sequences=640)),
    "B": dict(seq_len=32768, micro_batch=1, accum=4, stream="data_B",
              mix=dict(pg19_per_window=12, window_sequences=20)),
    "LM2": dict(seq_len=4096, micro_batch=2, accum=1, stream=None,
                mix=None, n_sequences=2000),
}


# --------------------------------------------------------------------------- stores
class TokenStore:
    """Read-only view of ``data/tokens/<corpus>/<split>.{bin,idx.npy}``."""

    def __init__(self, corpus: str, split: str, root: Path | str = TOKENS_DIR):
        self.corpus, self.split = corpus, split
        base = Path(root) / corpus / split
        self.tokens = np.memmap(str(base) + ".bin", dtype=np.uint16, mode="r")
        self.idx = np.load(str(base) + ".idx.npy")
        assert self.idx[0] == 0 and self.idx[-1] == len(self.tokens), "idx/bin mismatch"
        self.lengths = np.diff(self.idx)
        self.n_docs = len(self.lengths)
        self.n_tokens = int(self.idx[-1])

    def doc(self, i: int) -> np.ndarray:
        """Tokens of document i, starting with its ``<|endoftext|>``."""
        return np.asarray(self.tokens[self.idx[i]:self.idx[i + 1]])

    def doc_hashes(self) -> dict[str, np.ndarray]:
        z = np.load(str(Path(self.tokens.filename).with_suffix("")) + ".docs.npz")
        return {k: z[k] for k in z.files}


@lru_cache(maxsize=None)
def get_store(corpus: str, split: str, root: str = str(TOKENS_DIR)) -> TokenStore:
    """Process-wide cached ``TokenStore``."""
    return TokenStore(corpus, split, root)


# --------------------------------------------------------------------------- streams
def _gather(src: np.ndarray, starts: np.ndarray, lens: np.ndarray, out: np.ndarray) -> None:
    """out[concat] = src[starts[i] : starts[i]+lens[i]] for all i."""
    o = 0
    for s, n in zip(starts.tolist(), lens.tolist()):
        out[o:o + n] = src[s:s + n]
        o += n


class PackedStream:
    """Infinite EOS-separated concatenation of a store's documents, re-permuted per epoch.

    ``order_fn(epoch) -> permutation`` supplies the document order.
    """

    def __init__(self, store: TokenStore, order_fn):
        self.store = store
        self.order_fn = order_fn
        self.epoch_tokens = store.n_tokens

    @lru_cache(maxsize=4)
    def _epoch(self, e: int) -> tuple[np.ndarray, np.ndarray]:
        perm = self.order_fn(e)
        pcum = np.zeros(len(perm) + 1, dtype=np.int64)
        np.cumsum(self.store.lengths[perm], out=pcum[1:])
        return perm, pcum

    def take(self, a: int, n: int) -> tuple[np.ndarray, np.ndarray]:
        """Stream tokens ``[a, a+n)`` and the offsets (relative to a) of document starts."""
        out = np.empty(n, dtype=np.uint16)
        starts_rel: list[np.ndarray] = []
        pos = 0
        while pos < n:
            e, off = divmod(a + pos, self.epoch_tokens)
            m = min(n - pos, self.epoch_tokens - off)
            perm, pcum = self._epoch(e)
            i0 = int(np.searchsorted(pcum, off, side="right")) - 1
            i1 = int(np.searchsorted(pcum, off + m, side="left"))
            d_lo = np.maximum(pcum[i0:i1], off)
            d_hi = np.minimum(pcum[i0 + 1:i1 + 1], off + m)
            src = self.store.idx[perm[i0:i1]] + (d_lo - pcum[i0:i1])
            _gather(self.store.tokens, src, d_hi - d_lo, out[pos:pos + m])
            st = pcum[i0:i1]
            st = st[(st >= off) & (st < off + m)] - off + pos
            starts_rel.append(st)
            pos += m
        return out, np.concatenate(starts_rel)


class WindowSet:
    """All aligned, book-bounded ``W``-token windows of a store, visited in per-epoch random order."""

    def __init__(self, store: TokenStore, W: int, order_fn):
        self.store, self.W, self.order_fn = store, W, order_fn
        nwin = store.lengths // W
        book = np.repeat(np.arange(store.n_docs), nwin)
        first = np.concatenate([[0], np.cumsum(nwin)[:-1]])
        within = np.arange(len(book)) - np.repeat(first, nwin)
        self.starts = store.idx[book] + within * W  # file offsets
        self.book = book
        self.n = len(self.starts)
        assert self.n > 0, f"no {W}-token windows in {store.corpus}/{store.split}"

    @lru_cache(maxsize=4)
    def _perm(self, e: int) -> np.ndarray:
        return self.order_fn(e, self.n)

    def window_id(self, k: int) -> int:
        e, r = divmod(k, self.n)
        return int(self._perm(e)[r])

    def take(self, k: int) -> np.ndarray:
        """Tokens ``[s, s+W+1)`` of the k-th visited window (the extra token is the label of the last
        position; IGNORE_INDEX if the file ends)."""
        s = int(self.starts[self.window_id(k)])
        out = np.full(self.W + 1, IGNORE_INDEX, dtype=np.uint16)
        seg = self.store.tokens[s:s + self.W + 1]
        out[:len(seg)] = seg
        return out


# --------------------------------------------------------------------------- StageStream
@dataclass
class _Slot:
    kind: str  # 'fwe' | 'pg19'
    index: int  # k-th slot of that kind in the run


class StageStream:
    """Pure-function batch source; see the module docstring.

    Args:
        stage: 'A' | 'B' | 'LM2' | 'LR'.
        data_seed: seed index (§8.1). Ignored by 'LM2' (fixed held-out set).
        seq_len, micro_batch, accum: sequence length, sequences per micro-batch, micro-batches per step.
        mix: ``dict(pg19_per_window=int, window_sequences=int)``; ``None`` -> protocol default.
        root: token directory (tests point this at a synthetic corpus).
    """

    def __init__(self, stage: str, data_seed: int, seq_len: int | None = None, micro_batch: int | None = None,
                 accum: int | None = None, mix: dict | None = None, root: Path | str = TOKENS_DIR):
        if stage not in STAGE_DEFAULTS:
            raise ValueError(f"unknown stage {stage!r}")
        d = STAGE_DEFAULTS[stage]
        self.stage, self.data_seed = stage, int(data_seed)
        self.seq_len = int(seq_len or d["seq_len"])
        self.micro_batch = int(micro_batch or d["micro_batch"])
        self.accum = int(accum or d["accum"])
        self.root = str(root)
        self.seqs_per_step = self.micro_batch * self.accum
        if stage == "LM2":  # no mixing: every slot is FWE test
            mix = dict(pg19_per_window=0, window_sequences=self.seqs_per_step)
        self.mix = dict(mix or d["mix"])
        W, P = self.mix["window_sequences"], self.mix["pg19_per_window"]
        if W % self.seqs_per_step:
            raise ValueError(f"window_sequences={W} is not a multiple of sequences/step={self.seqs_per_step}; "
                             "mixing would not be exact per window")
        if not 0 <= P <= W:
            raise ValueError("pg19_per_window out of range")
        self.window_steps = W // self.seqs_per_step
        L = self.seq_len
        if stage == "LM2":
            self.n_sequences = d["n_sequences"]
            store = get_store("fwe", "test", self.root)
            self.fwe = PackedStream(store, lambda e: R.eval_test_generator("LM2", L, e).permutation(store.n_docs))
            self.pg19 = None
            return
        sid = R.STREAMS[d["stream"]]
        s = self.data_seed
        self.stream_id = sid
        fstore = get_store("fwe", "train", self.root)
        self.fwe = PackedStream(fstore, lambda e: R.generator(s, sid, 0, e).permutation(fstore.n_docs))
        self.pg19 = None
        if P > 0:
            pstore = get_store("pg19", "train", self.root)
            self.pg19 = WindowSet(pstore, L, lambda e, n: R.generator(s, sid, 1, e).permutation(n))

    # ---- slot assignment
    @lru_cache(maxsize=64)
    def _window_layout(self, w: int) -> tuple[np.ndarray, np.ndarray]:
        W, P = self.mix["window_sequences"], self.mix["pg19_per_window"]
        is_pg = np.zeros(W, dtype=bool)
        if self.stage != "LM2" and P:
            is_pg[R.generator(self.data_seed, self.stream_id, 2, w).choice(W, size=P, replace=False)] = True
        rank = np.where(is_pg, np.cumsum(is_pg) - 1, np.cumsum(~is_pg) - 1)
        return is_pg, rank

    def slot(self, g: int) -> _Slot:
        W, P = self.mix["window_sequences"], self.mix["pg19_per_window"]
        w, r = divmod(g, W)
        is_pg, rank = self._window_layout(w)
        if is_pg[r]:
            return _Slot("pg19", w * P + int(rank[r]))
        return _Slot("fwe", w * (W - P) + int(rank[r]))

    def num_steps(self) -> int | None:
        """Number of steps for finite streams (LM2), else None."""
        if self.stage == "LM2":
            return -(-self.n_sequences // self.seqs_per_step)
        return None

    # ---- batch construction
    def sequence(self, g: int) -> tuple[np.ndarray, np.ndarray, str]:
        """(tokens[L+1] uint16, segment starts within [0, L), source kind) of global slot g."""
        L = self.seq_len
        sl = self.slot(g)
        if sl.kind == "pg19":
            return self.pg19.take(sl.index), np.zeros(1, dtype=np.int64), "pg19"
        toks, st = self.fwe.take(sl.index * L, L + 1)
        st = st[st < L]
        if len(st) == 0 or st[0] != 0:
            st = np.concatenate([[0], st])
        return toks, st.astype(np.int64), "fwe"

    def get_numpy(self, step: int, micro: int) -> dict:
        if not 0 <= micro < self.accum:
            raise IndexError(f"micro {micro} out of range [0, {self.accum})")
        L, mb = self.seq_len, self.micro_batch
        g0 = (step * self.accum + micro) * mb
        if self.stage == "LM2" and g0 + mb > self.n_sequences:
            raise IndexError(f"LM2 has {self.n_sequences} sequences")
        N = L * mb
        inp = np.empty(N, dtype=np.int64)
        lab = np.empty(N, dtype=np.int64)
        starts, kinds = [], []
        for j in range(mb):
            toks, st, kind = self.sequence(g0 + j)
            inp[j * L:(j + 1) * L] = toks[:L]
            lab[j * L:(j + 1) * L] = toks[1:]
            starts.append(st + j * L)
            kinds.append(kind)
        st = np.concatenate(starts)
        cu = np.concatenate([st, [N]]).astype(np.int32)
        seglen = np.diff(cu).astype(np.int64)
        pos = np.arange(N, dtype=np.int64) - np.repeat(st, seglen)
        h = hashlib.sha256()
        h.update(inp.astype("<u2").tobytes())
        h.update(lab.astype("<u2").tobytes())
        h.update(cu.astype("<i4").tobytes())
        return dict(input_ids=inp[None], labels=lab[None], cu_seqlens=cu, position_ids=pos[None],
                    max_seqlen=int(seglen.max()), sources=kinds, batch_hash=h.hexdigest())

    def get(self, step: int, micro: int) -> dict:
        """Torch view of :meth:`get_numpy` (CPU tensors; ids/labels/positions int64, cu_seqlens int32)."""
        import torch

        b = self.get_numpy(step, micro)
        for k in ("input_ids", "labels", "position_ids", "cu_seqlens"):
            b[k] = torch.from_numpy(b[k])
        return b

    def step_hash(self, step: int) -> str:
        """SHA-256 over the batch hashes of all micro-batches of a step (logged per step, §6.4)."""
        h = hashlib.sha256()
        for m in range(self.accum):
            h.update(self.get_numpy(step, m)["batch_hash"].encode())
        return h.hexdigest()


def effective_context_histogram(cu_seqlens: np.ndarray, edges: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Histogram over tokens of the same-document context available at each position
    (position_in_segment + 1), with log2 bin edges by default (§6.3 logging)."""
    cu = np.asarray(cu_seqlens, dtype=np.int64)
    seglen = np.diff(cu)
    if edges is None:
        edges = 2.0 ** np.arange(0, 17)  # 1 .. 65536
        edges = np.append(edges, np.inf)
    ctx = np.arange(int(cu[-1])) - np.repeat(cu[:-1], seglen) + 1
    return np.histogram(ctx, bins=edges)[0], edges

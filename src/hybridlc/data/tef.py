"""Task-exposure fine-tune stream (protocol §7.3; INTERFACES.md ``data/tef.py``).

``TEFStream(seed_index, steps, ...).get(step, micro)`` follows the ``StageStream.get``
contract (``input_ids[1,N]``, ``labels[1,N]``, ``cu_seqlens`` int32, ``position_ids[1,N]``,
``batch_hash``) plus ``loss_weights[1,N]`` (and ``answer_mask[1,N]`` for logging/tests). It is
a pure function of (seed_index, step, micro): nothing to save on resume, identical for every
k at a given seed index (RNG stream ``tef`` = 4, §8.1), and the 1,000-step stream is the
prefix of the 4,000-step stream.

Construction (per block of ``block_steps`` = 1,000 steps; S = block_steps x seqs/step
sequences of ``seq_len`` = 32K tokens):

* **Counts.** ``n_b = floor(share * S * seq_len / sum(buckets))`` synthetic items per length
  bucket (907 at the protocol defaults: floor(0.9 x 1,000 x 65,536 / 65,024) with 1 K = 1,024;
  the protocol's [INF] "~928" used decimal K, 0.9 x 65.5e6 / 63.5e3 -- see eval/constants.py;
  the 90/10 token share is exact either way), split over the 3 families as
  equally as possible (the +1 remainders rotate across buckets), so counts per bucket and per
  family are equal within +-1 per block (VT-17). Within a (bucket, family) cell the sub-types
  follow the §7.3 mix exactly by largest remainder (retrieval R1/R-load/R3/R-dist =
  .4/.4/.1/.1; M-hop forward/shuffled/reversed = .25/.5/.25; A-freq/A-count = .7/.3).
* **Repetitive filler.** In each retrieval and multi-hop cell, round(10%) of the items use
  the repetitive filler [IMPL: aggregation items are word lists without natural filler].
* **Packing.** Items of length seq_len fill a sequence each; the others are packed into the
  remaining sequences largest-first into the sequence with the most free space (LPT), then
  each sequence's items are shuffled and its gap is filled by one **replay** chunk (Stage-B
  mix: PG19-train 60% / FineWeb-Edu-train 40%) at a random slot between items. Sequences are
  then shuffled within the block.
* **Items** are generated on demand by the eval generators in namespace ``train`` (word list
  A, templates T1-T3, filler from the training corpora), each exactly its bucket length
  including its 4 sequential queries, with levels drawn from the §7.3 training ranges:
  D ~ U{0..20}; N log-uniform in [4, 2048] s.t. fact tokens <= 0.75 l; h ~ U{1..6};
  rho ~ U[1.1, 5]; U ~ U{1..5}; R-dist: 4 distances log-uniform in [16, F] + 12 distractors.
* **Segments.** Every item and every FineWeb-Edu replay chunk starts with ``<|endoftext|>``;
  a PG19 replay chunk is a mid-book slice (its own segment, like the loader's cut pieces).
  A FineWeb-Edu replay chunk holds several whole documents: each inner ``<|endoftext|>``
  starts a further segment (verification round 1), so ``<|endoftext|>`` occurs only at
  ``cu_seqlens`` starts. ``cu_seqlens`` has a boundary at every segment start, positions
  restart there.
* **Labels.** ``labels[t] = x[t+1]`` within a sequence, except before a segment that does
  not start with ``<|endoftext|>`` and at the last position of a sequence (IGNORE = pad id 1),
  i.e. separators are predicted as in pre-training.
* **Loss weights (D014).** ``answer_norm="block"`` (default since verification round 3,
  [IMPL] pending PI confirmation): every answer label of a 1,000-step block gets the same weight
  ``0.5 / (A_block / block_steps)`` and every other valid label ``0.5 / (O_block / block_steps)``,
  where ``A_block`` / ``O_block`` are the block's answer / other label counts. Over the block the
  objective is exactly 1/2 mean CE(answers) + 1/2 mean CE(rest) (weights sum to ``block_steps``,
  i.e. 1 per step on average), and since the length buckets have equal item (and answer-token)
  counts they get equal answer-loss weight. The batch carries ``loss_norm = 1.0``: the trainer
  divides the step's sum of w*CE by that constant, not by the step's own weight sum (which would
  undo the block normalisation). ``answer_norm="step"`` is the literal per-step reading: answer
  labels of each optimiser step share 1/2 (``w = 0.5/n_answer`` resp. ``0.5/n_other``; sum over
  the step = 1), which gave the 32K bucket ~0.29-0.30 of the answer-loss weight instead of 1/7
  (a step holding 32K items has few answer tokens; TEF-STEP-NORMALISED-ANSWER-WEIGHT-LENGTH-SKEW).
  Counts come from the block plan, so ``get`` stays a pure per-micro function.
"""
from __future__ import annotations

import hashlib
import heapq
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from hybridlc.eval.constants import (TEF_ANSWER_NORM, TEF_ANSWER_NORMS, TEF_ANSWER_WEIGHT, TEF_BUCKETS,
                                     TEF_FAMILY_MIX, TEF_RANGES, TEF_REPETITIVE_SHARE, TEF_SEQ_LEN,
                                     TEF_STEPS_PER_BLOCK, TEF_SYNTHETIC_SHARE, TEF_TEMPLATES, MAX_FACT_FRACTION)
from hybridlc.eval.rng import child_rng, tef_rng

IGNORE_INDEX = 1  # pad id = chunked_ce ignore_index
FAMILIES = tuple(TEF_FAMILY_MIX)
#: Stage-B mix used for replay chunks (§6.3): 60% PG19-train, 40% FineWeb-Edu-train.
REPLAY_PG19_SHARE = 0.6
#: Answer tokens per query of each generator family (4 digits; 1 token for aggregation).
ANSWER_LEN = {"retrieval": 4, "multihop": 4, "aggregation": 1}
N_QUERIES = 4


@dataclass(frozen=True)
class ItemSpec:
    uid: int
    length: int
    family: str
    subtype: str  # 'R1' | 'R-load' | 'R3' | 'R-dist' | 'M-hop:forward' | ... | 'A-freq' | 'A-count'
    repetitive: bool

    @property
    def n_answer_tokens(self) -> int:
        return N_QUERIES * ANSWER_LEN[self.family]


@dataclass(frozen=True)
class Segment:
    kind: str  # 'item' | 'replay'
    length: int
    spec: ItemSpec | None = None
    replay_kind: str | None = None  # 'pg19' | 'fwe'
    replay_uid: int = -1

    @property
    def starts_with_eot(self) -> bool:
        return self.kind == "item" or self.replay_kind == "fwe"


def _largest_remainder(total: int, shares: dict[str, float]) -> dict[str, int]:
    """Integer split of ``total`` proportional to ``shares`` (ties -> listed order)."""
    keys = list(shares)
    raw = [total * shares[k] / sum(shares.values()) for k in keys]
    base = [int(math.floor(r)) for r in raw]
    rem = total - sum(base)
    order = sorted(range(len(keys)), key=lambda i: (-(raw[i] - base[i]), i))
    for i in order[:rem]:
        base[i] += 1
    return dict(zip(keys, base))


# =========================================================================== providers
class StoreReplay:
    """Replay chunks from the Stage-B training corpora in ``data/tokens``."""

    def __init__(self, root: str | Path | None = None):
        from hybridlc.data.loader import get_store
        from hybridlc.data.sources import TOKENS_DIR

        root = str(root or TOKENS_DIR)
        self.fwe = get_store("fwe", "train", root)
        self.pg19 = get_store("pg19", "train", root)

    def get(self, kind: str, n: int, rng: np.random.Generator) -> np.ndarray:
        if kind == "fwe":  # whole documents (each starting with EOT), first n tokens
            parts, tot = [], 0
            while tot < n:
                d = int(rng.integers(self.fwe.n_docs))
                x = self.fwe.doc(d)
                parts.append(x)
                tot += len(x)
            return np.concatenate(parts)[:n].astype(np.int32)
        if kind == "pg19":  # mid-book slices (no EOT); concatenated slices if no book is long enough
            L = self.pg19.lengths
            parts, tot = [], 0
            while tot < n:
                need = n - tot
                elig = np.flatnonzero(L > need + 1)
                if len(elig):
                    b = int(elig[rng.integers(len(elig))])
                    s = int(rng.integers(1, L[b] - need + 1))
                    parts.append(self.pg19.doc(b)[s:s + need])
                    tot += need
                else:
                    b = int(np.argmax(L))
                    parts.append(self.pg19.doc(b)[1:])
                    tot += int(L[b]) - 1
            return np.concatenate(parts)[:n].astype(np.int32)
        raise ValueError(kind)


# =========================================================================== stream
class TEFStream:
    """Pure-function TEF batch source (see module docstring).

    Args:
        seed_index: seed index (stream key (s, 4)).
        steps: number of optimiser steps of this run (bounds check only; the stream does not
            depend on it, so the 1,000-step run is the prefix of the 4,000-step run).
        seq_len, micro_batch, accum: 32K x 1 x 2 by default (65,536 tokens/step).
        synthetic_share: 0.9 (FEAS forgetting-gate fallback: 0.5).
        answer_only: elicitation-check fallback (§7.3): only answer tokens carry loss weight.
        answer_norm: "block" (default) or "step": over which unit the D014 means are taken (see
            the module docstring; verification round 3).
        buckets: item lengths (each <= seq_len).
        block_steps: steps per exact-count block (1,000).
        tok / filler / replay / root: tokenizer, training-corpus filler provider, replay
            provider and token-store root (defaults: data/tokenizer, StoreFiller('train'),
            StoreReplay(root)).
    """

    def __init__(self, seed_index: int, steps: int, seq_len: int = TEF_SEQ_LEN, micro_batch: int = 1,
                 accum: int = 2, synthetic_share: float = TEF_SYNTHETIC_SHARE,
                 buckets: tuple[int, ...] = TEF_BUCKETS, block_steps: int = TEF_STEPS_PER_BLOCK,
                 tok: Any = None, filler: Any = None, replay: Any = None, root: str | Path | None = None,
                 repetitive_share: float = TEF_REPETITIVE_SHARE, answer_weight: float = TEF_ANSWER_WEIGHT,
                 answer_only: bool = False, mix: dict | None = None, answer_norm: str = TEF_ANSWER_NORM,
                 **_: Any):
        self.seed_index, self.steps = int(seed_index), int(steps)
        self.seq_len, self.micro_batch, self.accum = int(seq_len), int(micro_batch), int(accum)
        self.seqs_per_step = self.micro_batch * self.accum
        self.synthetic_share = float(synthetic_share)
        self.buckets = tuple(int(b) for b in buckets)
        if max(self.buckets) > self.seq_len:
            raise ValueError("a bucket is longer than the sequence")
        self.block_steps = int(block_steps)
        self.repetitive_share, self.answer_weight = float(repetitive_share), float(answer_weight)
        #: Elicitation-check fallback (§7.3): loss on answer tokens only (weight 1/n_answer, others 0).
        self.answer_only = bool(answer_only)
        if self.answer_only:
            self.answer_weight = 1.0
        self.family_mix = TEF_FAMILY_MIX if not mix else mix
        if answer_norm not in TEF_ANSWER_NORMS:
            raise ValueError(f"answer_norm {answer_norm!r} not in {TEF_ANSWER_NORMS}")
        self.answer_norm = str(answer_norm)
        self._root = root
        self._tok, self._filler, self._replay = tok, filler, replay

    # ------------------------------------------------------------------ lazy resources
    @property
    def tok(self):
        if self._tok is None:
            from hybridlc.data.tokenizer import load_tokenizer

            self._tok = load_tokenizer()
        return self._tok

    @property
    def filler(self):
        if self._filler is None:
            from hybridlc.eval.filler import StoreFiller
            from hybridlc.eval.tokens import token_space

            self._filler = StoreFiller(token_space(self.tok), "train", self._root)
        return self._filler

    @property
    def replay(self):
        if self._replay is None:
            self._replay = StoreReplay(self._root)
        return self._replay

    @property
    def eot_id(self) -> int:
        from hybridlc.eval.tokens import token_space

        return token_space(self.tok).eot_id

    # ------------------------------------------------------------------ plan
    def items_per_bucket(self) -> int:
        S = self.block_steps * self.seqs_per_step
        return int(math.floor(self.synthetic_share * S * self.seq_len / sum(self.buckets)))

    @lru_cache(maxsize=4)
    def block_plan(self, block: int) -> tuple[tuple[Segment, ...], ...]:
        """Sequences (in step order) of one block: tuples of segments, each summing to seq_len."""
        S = self.block_steps * self.seqs_per_step
        L = self.seq_len
        rng = tef_rng(self.seed_index, "plan", block)
        nb = self.items_per_bucket()
        specs: list[ItemSpec] = []
        for bi, ell in enumerate(self.buckets):
            base, r = divmod(nb, len(FAMILIES))
            fam_n = {f: base for f in FAMILIES}
            for j in range(r):
                fam_n[FAMILIES[(bi + j) % len(FAMILIES)]] += 1
            for fam in FAMILIES:
                sub_n = _largest_remainder(fam_n[fam], self.family_mix[fam])
                cell = [st for st, c in sub_n.items() for _ in range(c)]
                cell = [cell[i] for i in child_rng(rng, "cell", bi, fam).permutation(len(cell))]
                n_rep = int(round(self.repetitive_share * len(cell))) if fam != "aggregation" else 0
                for k, st in enumerate(cell):
                    specs.append(ItemSpec(len(specs), ell, fam, st, k < n_rep))
        full = [s for s in specs if s.length == L]
        rest = [s for s in specs if s.length < L]
        if len(full) > S:
            raise ValueError(f"{len(full)} full-length items > {S} sequences")
        bins: list[list[ItemSpec]] = [[s] for s in full]
        nfree = S - len(full)
        free: list[list[ItemSpec]] = [[] for _ in range(nfree)]
        heap = [(-L, i) for i in range(nfree)]  # (-remaining, bin index)
        heapq.heapify(heap)
        tie = child_rng(rng, "tie").permutation(len(rest))
        for j in sorted(range(len(rest)), key=lambda j: (-rest[j].length, tie[j])):
            s = rest[j]
            negrem, bi = heapq.heappop(heap)
            if -negrem < s.length:
                raise ValueError("TEF packing failed: an item does not fit (synthetic share too high)")
            free[bi].append(s)
            heapq.heappush(heap, (negrem + s.length, bi))
        bins += free
        seqs: list[tuple[Segment, ...]] = []
        ruid = 0
        for bi, b in enumerate(bins):
            brng = child_rng(rng, "bin", bi)
            b = [b[i] for i in brng.permutation(len(b))]
            segs = [Segment("item", s.length, spec=s) for s in b]
            gap = L - sum(s.length for s in b)
            if gap > 0:
                kind = "pg19" if brng.random() < REPLAY_PG19_SHARE else "fwe"
                segs.insert(int(brng.integers(len(segs) + 1)), Segment("replay", gap, replay_kind=kind, replay_uid=ruid))
                ruid += 1
            assert sum(s.length for s in segs) == L
            seqs.append(tuple(segs))
        order = rng.permutation(S)
        return tuple(seqs[i] for i in order)

    def plan_counts(self, block: int) -> dict:
        """Per-(bucket, family) and per-(bucket, subtype) item counts of a block (VT-17)."""
        fam: dict[tuple[int, str], int] = {}
        sub: dict[tuple[int, str], int] = {}
        rep = 0
        for seq in self.block_plan(block):
            for sg in seq:
                if sg.kind == "item":
                    s = sg.spec
                    fam[(s.length, s.family)] = fam.get((s.length, s.family), 0) + 1
                    sub[(s.length, s.subtype)] = sub.get((s.length, s.subtype), 0) + 1
                    rep += s.repetitive
        return dict(family=fam, subtype=sub, repetitive=rep)

    # ------------------------------------------------------------------ indexing
    def _seq_index(self, step: int, micro: int, j: int) -> tuple[int, int]:
        b, s = divmod(int(step), self.block_steps)
        return b, (s * self.accum + micro) * self.micro_batch + j

    def _seq_counts(self, segs: tuple[Segment, ...]) -> tuple[int, int]:
        """(answer labels, valid labels) of a planned sequence."""
        n_ans = sum(sg.spec.n_answer_tokens for sg in segs if sg.kind == "item")
        n_ign = 1  # last position
        pos = 0
        for sg in segs:
            if pos > 0 and not sg.starts_with_eot:
                n_ign += 1
            pos += sg.length
        return n_ans, self.seq_len - n_ign

    def step_counts(self, step: int) -> tuple[int, int]:
        """(answer tokens, other valid tokens) over all micro-batches of ``step``."""
        na = nv = 0
        for m in range(self.accum):
            for j in range(self.micro_batch):
                b, g = self._seq_index(step, m, j)
                a, v = self._seq_counts(self.block_plan(b)[g])
                na += a
                nv += v
        return na, nv - na

    @lru_cache(maxsize=4)
    def block_counts(self, block: int) -> tuple[int, int]:
        """(answer labels, other valid labels) over every sequence of ``block`` (block normalisation)."""
        na = nv = 0
        for seq in self.block_plan(block):
            a, v = self._seq_counts(seq)
            na += a
            nv += v
        return na, nv - na

    def step_weight_scale(self, step: int) -> tuple[float, float]:
        """Per-label loss weight (answer, other) used at ``step`` under the configured normalisation."""
        aw = self.answer_weight
        if self.answer_norm == "block":
            n_ans, n_oth = self.block_counts(int(step) // self.block_steps)
            n_ans, n_oth = n_ans / self.block_steps, n_oth / self.block_steps
        else:
            n_ans, n_oth = self.step_counts(step)
        if not n_ans:
            return 0.0, 1.0 / max(n_oth, 1)
        return aw / n_ans, (0.0 if self.answer_only else (1.0 - aw) / max(n_oth, 1))

    # ------------------------------------------------------------------ items
    def make_item(self, block: int, spec: ItemSpec) -> dict:
        """Generate one TEF item (sequential queries, exactly spec.length tokens)."""
        from hybridlc.eval.context import InfeasibleItem
        from hybridlc.eval.generators import generate

        base = tef_rng(self.seed_index, "item", block, spec.uid)
        last = None
        for attempt in range(16):
            rng = child_rng(base, "attempt", attempt)
            task, level = self._sample_level(spec, rng)
            try:
                it = generate(task, level, spec.length, 1, rng, "train", self.tok, self.filler)[0]
            except InfeasibleItem as e:  # resample the level (rare: e.g. long templates at 512)
                last = e
                continue
            it["meta"]["tef"] = dict(block=block, uid=spec.uid, family=spec.family, subtype=spec.subtype,
                                     attempt=attempt)
            return it
        raise RuntimeError(f"TEF item {spec} infeasible after 16 attempts: {last}")

    def _sample_level(self, spec: ItemSpec, rng: np.random.Generator) -> tuple[str, dict]:
        from hybridlc.eval import templates as TP
        from hybridlc.eval.tokens import token_space

        ell = spec.length
        tname = TEF_TEMPLATES[int(rng.integers(len(TEF_TEMPLATES)))]
        level: dict[str, Any] = dict(template=tname, filler="repetitive" if spec.repetitive else "natural")
        ts = token_space(self.tok)
        flen = TP.fact_len(ts, TP.CODE_TEMPLATES[tname])
        cap = int(MAX_FACT_FRACTION * ell // flen)  # max facts with fact tokens <= 0.75 l
        st = spec.subtype
        if st == "R1":
            D_lo, D_hi = TEF_RANGES["D"]
            level["D"] = int(rng.integers(D_lo, min(D_hi, cap - 1) + 1))
            return "R1", level
        if st == "R-load":
            n_lo, n_hi = TEF_RANGES["N"]
            hi = max(n_lo, min(n_hi, cap))
            level["N"] = int(round(math.exp(rng.uniform(math.log(n_lo), math.log(hi)))))
            return "R-load", level
        if st == "R3":
            u_lo, u_hi = TEF_RANGES["U"]
            level["U"] = int(rng.integers(u_lo, u_hi + 1))
            return "R3", level
        if st == "R-dist":
            nd = 12
            qlen = 4 * TP.code_query_len(ts, TP.CODE_TEMPLATES[tname]) + 3
            F = ell - 1 - (4 + nd) * flen - qlen
            hi = max(17, F)
            level["distances"] = sorted(int(round(math.exp(rng.uniform(math.log(16), math.log(hi))))) for _ in range(4))
            level["n_distractors"] = nd
            return "R-dist", level
        if st.startswith("M-hop"):
            h_lo, h_hi = TEF_RANGES["h"]
            level["h"] = int(rng.integers(h_lo, h_hi + 1))
            level["order"] = st.split(":")[1]
            return "M-hop", level
        if st == "A-freq":
            r_lo, r_hi = TEF_RANGES["rho"]
            level["rho"] = float(rng.uniform(r_lo, r_hi))
            return "A-freq", level
        if st == "A-count":
            return "A-count", level
        raise ValueError(st)

    def _replay_tokens(self, block: int, sg: Segment) -> np.ndarray:
        rng = tef_rng(self.seed_index, "replay", block, sg.replay_uid)
        x = np.asarray(self.replay.get(sg.replay_kind, sg.length, rng), dtype=np.int32)
        assert len(x) == sg.length
        if sg.replay_kind == "fwe":
            assert x[0] == self.eot_id
        return x

    def sequence(self, step: int, micro: int, j: int) -> dict:
        """Tokens, segment starts, answer mask and item metadata of one planned sequence."""
        b, g = self._seq_index(step, micro, j)
        segs = self.block_plan(b)[g]
        toks, amask, starts, eot_start, metas = [], [], [], [], []
        from hybridlc.eval.generators import flatten_item

        pos = 0
        for sg in segs:
            starts.append(pos)
            eot_start.append(sg.starts_with_eot)
            if sg.kind == "item":
                it = self.make_item(b, sg.spec)
                x, am = flatten_item(it)
                assert len(x) == sg.length, (len(x), sg.length)
                assert int(am.sum()) == sg.spec.n_answer_tokens
                metas.append(dict(start=pos, **it["meta"]["tef"], task=it["meta"]["task"], level=it["meta"]["level"],
                                  length=sg.length))
            else:
                x = self._replay_tokens(b, sg)
                am = np.zeros(sg.length, bool)
                # A FineWeb-Edu replay chunk is several whole documents, each led by
                # <|endoftext|>: every inner separator is a document boundary too (§6.4
                # intra-document masking, RoPE reset; §9.1 attention never crosses a
                # separator). Verification round 1, TEF-REPLAY-CROSS-DOCUMENT / DTS-01.
                inner = (np.flatnonzero(x[1:] == self.eot_id) + 1) if sg.replay_kind == "fwe" else np.zeros(0, int)
                for o in inner:
                    starts.append(pos + int(o))
                    eot_start.append(True)
                metas.append(dict(start=pos, replay=sg.replay_kind, length=sg.length, n_docs=1 + len(inner)))
            toks.append(x)
            amask.append(am)
            pos += sg.length
        return dict(tokens=np.concatenate(toks), answer=np.concatenate(amask), starts=np.asarray(starts, np.int64),
                    eot_start=np.asarray(eot_start, bool), segments=metas)

    # ------------------------------------------------------------------ batch
    def get_numpy(self, step: int, micro: int) -> dict:
        if not 0 <= micro < self.accum:
            raise IndexError(f"micro {micro} out of range [0, {self.accum})")
        if step < 0:
            raise IndexError("negative step")
        L, mb = self.seq_len, self.micro_batch
        N = L * mb
        inp = np.empty(N, np.int64)
        lab = np.full(N, IGNORE_INDEX, np.int64)
        ans_lab = np.zeros(N, bool)
        all_starts, segs = [], []
        for j in range(mb):
            sq = self.sequence(step, micro, j)
            x, o = sq["tokens"], j * L
            inp[o:o + L] = x
            nxt_ok = np.ones(L - 1, bool)  # label t valid iff t+1 is same segment or an EOT-led segment start
            for s, e in zip(sq["starts"], sq["eot_start"]):
                if s > 0 and not e:
                    nxt_ok[s - 1] = False
            lab[o:o + L - 1] = np.where(nxt_ok, x[1:], IGNORE_INDEX)
            ans_lab[o:o + L - 1] = sq["answer"][1:]
            all_starts.append(sq["starts"] + o)
            segs.append(sq["segments"])
        st = np.concatenate(all_starts)
        cu = np.r_[st, N].astype(np.int32)
        seglen = np.diff(cu).astype(np.int64)
        pos = np.arange(N, dtype=np.int64) - np.repeat(st, seglen)
        valid = lab != IGNORE_INDEX
        assert not (ans_lab & ~valid).any()
        w = np.zeros(N, np.float32)
        w_ans, w_oth = self.step_weight_scale(step)
        if w_ans:
            w[ans_lab] = w_ans
            w[valid & ~ans_lab] = w_oth
        else:  # no answer labels in the block/step: plain mean over the valid labels
            w[valid] = w_oth
        h = hashlib.sha256()
        h.update(inp.astype("<u2" if inp.max() < 65536 else "<i4").tobytes())
        h.update(lab.astype("<i4").tobytes())
        h.update(cu.astype("<i4").tobytes())
        h.update(w.astype("<f4").tobytes())
        out = dict(input_ids=inp[None], labels=lab[None], cu_seqlens=cu, position_ids=pos[None],
                   loss_weights=w[None], answer_mask=ans_lab[None], max_seqlen=int(seglen.max()),
                   segments=segs, batch_hash=h.hexdigest())
        if self.answer_norm == "block":
            # the trainer divides sum(w * CE) over the step by this constant (not by sum(w); see docstring)
            out["loss_norm"] = 1.0
        return out

    def get(self, step: int, micro: int) -> dict:
        """Torch view of :meth:`get_numpy` (CPU tensors)."""
        import torch

        b = self.get_numpy(step, micro)
        for k in ("input_ids", "labels", "position_ids", "cu_seqlens", "loss_weights", "answer_mask"):
            b[k] = torch.from_numpy(b[k])
        return b

    def step_hash(self, step: int) -> str:
        h = hashlib.sha256()
        for m in range(self.accum):
            h.update(self.get_numpy(step, m)["batch_hash"].encode())
        return h.hexdigest()

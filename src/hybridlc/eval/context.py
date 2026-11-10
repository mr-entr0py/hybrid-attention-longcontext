"""Token-space context assembly (protocol §9.1).

``assemble`` inserts pre-tokenised needles into a filler span of exactly
``F = T_ctx - 1 - sum(len(needles))`` tokens, so the context is exactly ``T_ctx`` tokens:
``[<|endoftext|>] + filler-with-needles``. Each needle has a target *filler offset* (insert
before filler token ``o``), which is snapped to the nearest sentence boundary within
+-``SNAP_WINDOW`` tokens; if no boundary is in range, the unsnapped offset is kept and the
needle is flagged ``snapped=False``. Needles whose relative order matters (M-hop forward /
reversed chains, R3 updates) keep their pre-snap order: snapped offsets are made
non-decreasing in pre-snap order (a running maximum of boundaries is still a boundary).

Recorded per needle: start/end (end exclusive) in context coordinates, the filler offset,
``depth = offset / F`` and ``distance = T_ctx - end`` (``query_start - needle_end``; the
first query token sits at index ``T_ctx``).

Needles specified by a *distance* (R-dist) get their offset solved so that the realised
distance (which includes needles inserted after them) matches the target before snapping.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hybridlc.eval.constants import SNAP_WINDOW
from hybridlc.eval.filler import boundaries, snap
from hybridlc.eval.tokens import TokenSpace


class InfeasibleItem(ValueError):
    """The requested needles and queries do not fit into T tokens."""


@dataclass
class Placed:
    """Assembly result."""

    context: np.ndarray  # int32[T_ctx]
    starts: np.ndarray  # int64[n_needles] (input order)
    ends: np.ndarray
    offsets: np.ndarray  # snapped filler offsets
    snapped: np.ndarray  # bool
    F: int

    @property
    def depth(self) -> np.ndarray:
        return self.offsets / max(self.F, 1)

    def distance(self) -> np.ndarray:
        return len(self.context) - self.ends


def solve_distance_offsets(F: int, lens: np.ndarray, target: np.ndarray, dist: np.ndarray, iters: int = 4) -> np.ndarray:
    """Offsets for needles with a requested distance (``dist`` not NaN); others keep ``target``.

    distance_i = (F - o_i) + sum of lengths of needles inserted after needle i.
    """
    o = target.astype(np.float64).copy()
    has = ~np.isnan(dist)
    o[has] = F - dist[has]
    for _ in range(iters):
        order = np.lexsort((np.arange(len(o)), o))
        after = np.zeros(len(o))
        cum = 0.0
        for i in order[::-1]:
            after[i] = cum
            cum += lens[i]
        o[has] = F - dist[has] + after[has]
    return np.clip(o, 0, F)


def assemble(ts: TokenSpace, filler: np.ndarray, needles: list[np.ndarray], target: np.ndarray,
             window: int = SNAP_WINDOW, do_snap: bool = True, tie_rank: np.ndarray | None = None) -> Placed:
    """Insert ``needles`` into ``filler`` at (snapped) offsets ``target`` (floats in [0, F]).

    ``tie_rank`` (optional, one sortable value per needle) breaks ties between equal targets
    (after clipping to [0, F]) and between equal snapped offsets; without it ties fall back to
    input order. Generators whose needle order matters when every target coincides (the
    no-filler C0 control has F = 0; the oracle C2 control puts every needle at F) must pass
    an explicit rank, otherwise the realised order is the generator's input order
    (verification round 1, MHOP-C0-C2-UNSHUFFLED).
    """
    F = len(filler)
    n = len(needles)
    target = np.clip(np.asarray(target, dtype=np.float64), 0, F)
    tie = np.arange(n) if tie_rank is None else np.asarray(tie_rank)
    if len(tie) != n:
        raise ValueError(f"tie_rank has {len(tie)} entries for {n} needles")
    pre = np.lexsort((np.arange(n), tie, target))  # pre-snap order (ties: tie_rank, then input order)
    raw = np.rint(target).astype(np.int64)
    offs = np.empty(n, dtype=np.int64)
    ok = np.zeros(n, dtype=bool)
    if F > 0 and do_snap:
        bpos = boundaries(ts, filler)
        for i in range(n):
            offs[i], ok[i] = snap(bpos, int(raw[i]), window, lo=0, hi=F)
    else:
        offs[:] = raw
        ok[:] = F == 0 or not do_snap
    # keep pre-snap order: running max along the pre-snap order
    offs[pre] = np.maximum.accumulate(offs[pre])
    order = np.lexsort((np.argsort(pre, kind="stable"), offs))  # by offset, then pre-snap rank
    pieces: list[np.ndarray] = [np.asarray([ts.eot_id], dtype=np.int32)]
    starts = np.empty(n, dtype=np.int64)
    ends = np.empty(n, dtype=np.int64)
    pos, cur = 1, 0
    for i in order:
        o = int(offs[i])
        if o > cur:
            pieces.append(filler[cur:o])
            pos += o - cur
            cur = o
        nd = needles[i]
        starts[i] = pos
        pieces.append(nd)
        pos += len(nd)
        ends[i] = pos
    pieces.append(filler[cur:])
    ctx = np.concatenate(pieces).astype(np.int32)
    assert len(ctx) == 1 + F + sum(len(x) for x in needles)
    return Placed(ctx, starts, ends, offs, ok, F)


def decile_depths(cell_rng: np.random.Generator, item_index: int, n_slots: int, item_rng: np.random.Generator) -> np.ndarray:
    """Depths of an item's queried needles, stratified uniformly within deciles across the cell.

    Slot ``s = n_slots * item_index + q``. Slots are grouped in blocks of 10; each block's
    deciles are a random permutation of 0..9 (from the cell's own RNG, keyed by block), so
    any prefix of whole blocks is exactly balanced and the assignment is prefix-stable.
    Within the decile the depth is uniform (item RNG).
    """
    from hybridlc.eval.rng import child_rng

    out = np.empty(n_slots)
    for q in range(n_slots):
        s = n_slots * item_index + q
        b, r = divmod(s, 10)
        perm = child_rng(cell_rng, "decile", b).permutation(10)
        out[q] = (perm[r] + item_rng.random()) / 10.0
    return out

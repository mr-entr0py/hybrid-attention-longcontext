"""Cloze item generators (protocol §9.1-9.4, §9.6 R0; TEF families §7.3).

Signature (INTERFACES.md)::

    gen(task_level: dict, T: int | None, n: int, rng: np.random.Generator,
        namespace: 'test'|'dev'|'train', tok, filler: FillerProvider | None = None) -> list[Item]

``Item = dict(context_ids=int32[T_ctx], queries=[dict(prompt_ids, answer_ids, candidates, meta)],
meta=dict, arrays=dict[str, np.ndarray])``.

Length rule (protocol §9.1 "every sample is exactly T tokens"):

* ``test`` / ``dev`` (query branching): every query of an item has the same
  ``len(prompt) + len(answer) = Lq`` and ``len(context) = T - Lq``, so each scored sample
  (context + one query) is exactly T tokens.
* ``train`` (TEF, §7.3): the 4 queries are appended sequentially; ``len(context) +
  sum_q (len(prompt_q) + len(answer_q)) = T``. Code-template prompts after the first get a
  leading ``"."`` token (". The code for ..."), so the flattened item reads naturally.
* ``T=None`` (or ``placement='nofill'``) is the C0 no-filler control: context =
  ``[EOT] + needles``.

Item ``i`` of a cell is generated from ``child_rng(rng, i)`` only (plus the cell-level
decile assignment), so the first m items of an n-item cell equal the m-item cell.

``task_level`` keys: the level parameter (``D``, ``N``, ``h``, ``rho``, ``U``, ``d``), and
optionally ``template`` (default 'T1'), ``placement`` ('spread' | 'oracle' | 'nofill';
C2 = 'oracle', C0 = 'nofill'), ``filler`` ('natural' | 'repetitive'), ``order`` (M-hop:
'shuffled' | 'forward' | 'reversed'), ``distances`` (R-dist), ``n_distractors`` (R-dist).
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np

from hybridlc.eval import templates as TP
from hybridlc.eval.constants import (ACOUNT_N_DISTRACTORS, ACOUNT_N_QUERIED, ACOUNT_RANGE, AFREQ_MAX_BACKGROUND_OCC,
                                     AFREQ_N_Y, AFREQ_PAIRS, MAX_CONTEXT_CANDIDATES, MAX_FACT_FRACTION,
                                     N_ABSENT_CANDIDATES, Q_PER_CONTEXT)
from hybridlc.eval.context import InfeasibleItem, Placed, assemble, decile_depths, solve_distance_offsets
from hybridlc.eval.filler import NATURAL_KINDS, FillerProvider
from hybridlc.eval.rng import child_rng
from hybridlc.eval.tokens import TokenSpace, token_space
from hybridlc.eval.words import WordList, namespace_list, word_list

Item = dict[str, Any]

#: Fact roles stored in ``arrays['fact_role']``.
ROLE_DISTRACTOR, ROLE_QUERIED, ROLE_LINK, ROLE_STALE = 0, 1, 2, 3

#: R-dist defaults (§9.4): 4 targets at these distances + 12 distractors.
RDIST_DISTANCES = (256, 2 * 1024, 8 * 1024, 30 * 1024)
RDIST_N_DISTRACTORS = 12


# =========================================================================== helpers
class _Ctx:
    """Per-call shared state."""

    def __init__(self, task: str, level: dict, T: int | None, rng: np.random.Generator, namespace: str, tok: Any,
                 filler: FillerProvider | None):
        if namespace not in ("test", "dev", "train"):
            raise ValueError(f"namespace must be test|dev|train, got {namespace!r}")
        self.task, self.level, self.T, self.cell_rng, self.namespace = task, dict(level), T, rng, namespace
        self.ts: TokenSpace = token_space(tok)
        self.wl: WordList = word_list(self.ts, namespace_list(namespace))
        self.placement = level.get("placement", "nofill" if T is None else "spread")
        if self.placement not in ("spread", "oracle", "nofill"):
            raise ValueError(f"bad placement {self.placement!r}")
        if self.placement != "nofill" and T is None:
            raise ValueError("T=None is only valid for the no-filler control")
        tname = level.get("template", "T1")
        allowed = TP.TRAIN_TEMPLATES if namespace == "train" else TP.EVAL_TEMPLATES
        if tname not in allowed and tname != "PK":
            raise ValueError(f"template {tname} not allowed in namespace {namespace} (allowed {allowed})")
        self.tname = tname
        self.filler_mode = level.get("filler", "natural")
        self._filler = filler

    @property
    def sequential(self) -> bool:
        return self.namespace == "train"

    def provider(self) -> FillerProvider:
        if self._filler is None:
            from hybridlc.eval.filler import StoreFiller

            self._filler = StoreFiller(self.ts, self.namespace)
        return self._filler

    def filler(self, F: int, irng: np.random.Generator) -> tuple[np.ndarray, dict]:
        """Draw the filler kind (natural: pg19/fwe 50/50) and get F tokens."""
        if self.placement == "nofill":
            return np.zeros(0, np.int32), dict(kind="none")
        if self.filler_mode == "repetitive":
            kind = "repetitive"
        elif self.filler_mode == "natural":
            kind = NATURAL_KINDS[int(irng.integers(2))]
        else:
            raise ValueError(f"bad filler mode {self.filler_mode!r}")
        return self.provider().get(kind, F, irng)


def _values(irng: np.random.Generator, k: int) -> np.ndarray:
    """k distinct uniform 4-digit values (0..9999)."""
    return irng.choice(10_000, size=k, replace=False).astype(np.int64)


def _absent(irng: np.random.Generator, used: set[int], k: int = N_ABSENT_CANDIDATES) -> list[int]:
    out: list[int] = []
    while len(out) < k:
        v = int(irng.integers(10_000))
        if v not in used and v not in out:
            out.append(v)
    return out


def _keys(ctx: _Ctx, irng: np.random.Generator, k: int) -> list[np.ndarray]:
    if k > ctx.wl.n_keys:
        raise InfeasibleItem(f"{k} keys requested, list has {ctx.wl.n_keys}")
    idx = irng.choice(ctx.wl.n_keys, size=k, replace=False)
    return [ctx.wl.key_ids(int(i)) for i in idx]


def _code_queries(ctx: _Ctx, tpl: TP.CodeTemplate, keys: list[np.ndarray], golds: list[int],
                  cands: list[tuple[list[int], list[str]]]) -> list[dict]:
    """Code-template queries (prompt + 4 digit answer) with value candidates."""
    ts = ctx.ts
    out = []
    lead = ts.enc(TP.SEQ_QUERY_LEAD)
    for q, (k, g, (cvals, ctypes)) in enumerate(zip(keys, golds, cands)):
        prompt = TP.query_prompt_ids(ts, tpl, k)
        if ctx.sequential and q > 0:
            prompt = np.concatenate([lead, prompt]).astype(np.int32)
        out.append(dict(prompt_ids=prompt, answer_ids=TP.value_ids(ts, g),
                        candidates=[TP.value_ids(ts, v).tolist() for v in cvals],
                        meta=dict(query_idx=q, key=ts.decode(k).strip(), value=f"{g:04d}", cand_types=ctypes,
                                  cand_values=[f"{v:04d}" for v in cvals])))
    return out


def _value_candidates(irng: np.random.Generator, gold: int, context_vals: list[int], used: set[int],
                      priority: list[tuple[int, str]] | None = None) -> tuple[list[int], list[str]]:
    """gold + <= 32 in-context values (priority ones first, e.g. R3 stale) + 4 absent values."""
    vals, types = [gold], ["gold"]
    pr = [(v, t) for v, t in (priority or []) if v != gold]
    for v, t in pr[:MAX_CONTEXT_CANDIDATES]:
        vals.append(v)
        types.append(t)
    rest = [v for v in context_vals if v != gold and v not in vals]
    room = MAX_CONTEXT_CANDIDATES - (len(vals) - 1)
    if len(rest) > room:
        rest = [rest[i] for i in sorted(irng.choice(len(rest), size=room, replace=False))]
    vals += rest
    types += ["context"] * len(rest)
    for v in _absent(irng, used):
        vals.append(v)
        types.append("absent")
    return vals, types


def _query_total(queries: list[dict], sequential: bool) -> int:
    lens = [len(q["prompt_ids"]) + len(q["answer_ids"]) for q in queries]
    if sequential:
        return sum(lens)
    if len(set(lens)) != 1:
        raise AssertionError(f"branched queries must have equal length, got {lens}")
    return lens[0]


def _finish(ctx: _Ctx, context: np.ndarray, queries: list[dict], meta: dict, arrays: dict) -> Item:
    meta.update(task=ctx.task, level={k: v for k, v in ctx.level.items()}, T=ctx.T, T_ctx=int(len(context)),
                namespace=ctx.namespace, template=ctx.tname, placement=ctx.placement,
                query_mode="sequential" if ctx.sequential else "branch")
    tot = len(context) + _query_total(queries, ctx.sequential)
    if ctx.T is not None and ctx.placement != "nofill":
        assert tot == ctx.T, (ctx.task, tot, ctx.T)
    meta["sample_len"] = int(tot)
    return dict(context_ids=context.astype(np.int32), queries=queries, meta=meta, arrays=arrays)


def _fact_arrays(keys: list[np.ndarray], values: list[int], placed: Placed, roles: list[int],
                 idx: list[int] | None = None) -> dict[str, np.ndarray]:
    idx = list(range(len(keys))) if idx is None else idx
    return dict(fact_key=np.asarray([keys[i] for i in idx], dtype=np.int32).reshape(-1, 2),
                fact_value=np.asarray([values[i] for i in idx], dtype=np.int32),
                fact_start=placed.starts[idx].astype(np.int32), fact_end=placed.ends[idx].astype(np.int32),
                fact_role=np.asarray([roles[i] for i in idx], dtype=np.int8))


def _code_template(ctx: _Ctx) -> TP.CodeTemplate:
    return TP.PASSKEY if ctx.tname == "PK" else TP.CODE_TEMPLATES[ctx.tname]


def _target_offsets(ctx: _Ctx, F: int, n: int, queried: list[int], qdepths: np.ndarray,
                    irng: np.random.Generator) -> np.ndarray:
    """Spread: queried needles at their (decile) depths, the others uniform in [0, F]."""
    if ctx.placement == "oracle":
        return np.full(n, float(F))  # all right before the query (input order)
    t = irng.random(n) * F
    for j, i in enumerate(queried):
        t[i] = qdepths[j] * F
    return t


def _ctx_len(ctx: _Ctx, needles_tot: int, qtot: int) -> int:
    """Context length: T - query tokens, or 1 + needles for the no-filler control."""
    if ctx.placement == "nofill":
        return 1 + needles_tot
    T_ctx = ctx.T - qtot
    if T_ctx - 1 - needles_tot < 0:
        raise InfeasibleItem(f"{ctx.task} {ctx.level}: needles {needles_tot} + queries {qtot} > T={ctx.T}")
    return T_ctx


def _place(ctx: _Ctx, irng: np.random.Generator, needles: list[np.ndarray], T_ctx: int,
           target_fn: Callable[[int], np.ndarray], dist: np.ndarray | None = None) -> tuple[Placed, dict]:
    tot = sum(len(x) for x in needles)
    F = T_ctx - 1 - tot
    filler, src = ctx.filler(F, irng)
    target = target_fn(F)
    if dist is not None:
        target = solve_distance_offsets(F, np.asarray([len(x) for x in needles], float), target, dist)
    placed = assemble(ctx.ts, filler, needles, target, do_snap=ctx.placement != "nofill")
    assert len(placed.context) == T_ctx
    return placed, src


def _query_meta_positions(q: dict, placed: Placed, i: int | list[int], ctx: _Ctx) -> None:
    ii = [i] if isinstance(i, (int, np.integer)) else list(i)
    m = q["meta"]
    m["needle_start"] = [int(placed.starts[j]) for j in ii]
    m["needle_end"] = [int(placed.ends[j]) for j in ii]
    m["snapped"] = [bool(placed.snapped[j]) for j in ii]
    j = ii[0]
    m["depth"] = None if ctx.placement == "nofill" else float(placed.depth[j])
    m["distance"] = int(len(placed.context) - placed.ends[j])


def _snap_rate(placed: Placed) -> float:
    return float(placed.snapped.mean()) if len(placed.snapped) else 1.0


# =========================================================================== retrieval
def _facts_task(ctx: _Ctx, i: int, irng: np.random.Generator, n_facts: int, queried: list[int],
                dist: np.ndarray | None = None) -> Item:
    """R1 / R-load / R-dist core: n_facts distinct keys, queries on ``queried`` fact indices."""
    ts, tpl = ctx.ts, _code_template(ctx)
    keys = _keys(ctx, irng, n_facts)
    vals = [int(v) for v in _values(irng, n_facts)]
    needles = [TP.fact_ids(ts, tpl, k, v) for k, v in zip(keys, vals)]
    tot = sum(len(x) for x in needles)
    if ctx.task == "R-load" and ctx.T is not None and ctx.placement != "nofill":
        if tot > MAX_FACT_FRACTION * ctx.T:
            raise InfeasibleItem(f"R-load fact tokens {tot} > {MAX_FACT_FRACTION} T = {MAX_FACT_FRACTION * ctx.T}")
    used = set(vals)
    golds = [vals[j] for j in queried]
    cands = [_value_candidates(irng, g, vals, used) for g in golds]
    queries = _code_queries(ctx, tpl, [keys[j] for j in queried], golds, cands)
    T_ctx = _ctx_len(ctx, tot, _query_total(queries, ctx.sequential))
    uq = sorted(set(queried))
    qd = decile_depths(ctx.cell_rng, i, len(uq), irng)
    placed, src = _place(ctx, irng, needles, T_ctx, lambda F: _target_offsets(ctx, F, n_facts, uq, qd, irng), dist)
    for q, j in zip(queries, queried):
        _query_meta_positions(q, placed, j, ctx)
    roles = [ROLE_QUERIED if j in uq else ROLE_DISTRACTOR for j in range(n_facts)]
    meta = dict(context_id=i, filler_source=src, n_facts=n_facts, snap_rate=_snap_rate(placed))
    return _finish(ctx, placed.context, queries, meta, _fact_arrays(keys, vals, placed, roles))


def _queried_subset(irng: np.random.Generator, n_facts: int, nq: int = Q_PER_CONTEXT) -> list[int]:
    if n_facts >= nq:
        return [int(x) for x in irng.choice(n_facts, size=nq, replace=False)]
    return [q % n_facts for q in range(nq)]  # TEF D < 3: facts are re-queried cyclically


def gen_r1(task_level: dict, T: int | None, n: int, rng: np.random.Generator, namespace: str, tok: Any,
           filler: FillerProvider | None = None) -> list[Item]:
    """R1: D + 1 same-format facts (distinct keys), 4 of them queried (§9.3)."""
    ctx = _Ctx("R1", task_level, T, rng, namespace, tok, filler)
    D = int(task_level["D"])
    out = []
    for i in range(n):
        irng = child_rng(rng, i)
        out.append(_facts_task(ctx, i, irng, D + 1, _queried_subset(irng, D + 1)))
    return out


def gen_rload(task_level: dict, T: int | None, n: int, rng: np.random.Generator, namespace: str, tok: Any,
              filler: FillerProvider | None = None) -> list[Item]:
    """R-load: N facts spread through filler, all keys distinct, fact tokens <= 0.75 T (asserted)."""
    ctx = _Ctx("R-load", task_level, T, rng, namespace, tok, filler)
    N = int(task_level["N"])
    out = []
    for i in range(n):
        irng = child_rng(rng, i)
        out.append(_facts_task(ctx, i, irng, N, _queried_subset(irng, N)))
    return out


def gen_rdist(task_level: dict, T: int | None, n: int, rng: np.random.Generator, namespace: str, tok: Any,
              filler: FillerProvider | None = None) -> list[Item]:
    """R-dist: 4 targets at query distances d (default 256/2K/8K/30K) + 12 distractors (§9.4).

    ``task_level['distances']`` overrides the 4 distances (the TEF samples them);
    ``task_level['filler']`` is 'natural' or 'repetitive'.
    """
    ctx = _Ctx("R-dist", task_level, T, rng, namespace, tok, filler)
    dists = [int(d) for d in task_level.get("distances", RDIST_DISTANCES)]
    nd = int(task_level.get("n_distractors", RDIST_N_DISTRACTORS))
    out = []
    for i in range(n):
        irng = child_rng(rng, i)
        nf = len(dists) + nd
        dist = np.full(nf, np.nan)
        dist[:len(dists)] = dists
        it = _facts_task(ctx, i, irng, nf, list(range(len(dists))), dist=dist)
        for q, d in zip(it["queries"], dists):
            q["meta"]["target_distance"] = d
        out.append(it)
    return out


def gen_r3(task_level: dict, T: int | None, n: int, rng: np.random.Generator, namespace: str, tok: Any,
           filler: FillerProvider | None = None) -> list[Item]:
    """R3: 4 keys, each written once and then updated U times; the latest value is queried (§9.4).

    Earlier writes of a key always precede its latest write. Candidates include every stale
    value of the queried key (error type 'stale').
    """
    ctx = _Ctx("R3", task_level, T, rng, namespace, tok, filler)
    U = int(task_level["U"])
    ts, tpl = ctx.ts, _code_template(ctx)
    out = []
    for i in range(n):
        irng = child_rng(rng, i)
        nk = Q_PER_CONTEXT
        keys = _keys(ctx, irng, nk)
        vals = [int(v) for v in _values(irng, nk * (U + 1))]
        # fact j = key (j // (U+1)), version (j % (U+1)); version U is the latest
        fkeys = [keys[j // (U + 1)] for j in range(nk * (U + 1))]
        needles = [TP.fact_ids(ts, tpl, k, v) for k, v in zip(fkeys, vals)]
        latest = [kk * (U + 1) + U for kk in range(nk)]
        used = set(vals)
        golds = [vals[j] for j in latest]
        cands = []
        for kk, g in enumerate(golds):
            stale = [(vals[kk * (U + 1) + u], "stale") for u in range(U)]
            cands.append(_value_candidates(irng, g, vals, used, priority=stale))
        queries = _code_queries(ctx, tpl, keys, golds, cands)
        tot = sum(len(x) for x in needles)
        T_ctx = _ctx_len(ctx, tot, _query_total(queries, ctx.sequential))
        qd = decile_depths(ctx.cell_rng, i, nk, irng)

        def target(F: int) -> np.ndarray:
            t = np.empty(len(needles))
            for kk in range(nk):
                if ctx.placement == "oracle":
                    t[kk * (U + 1):(kk + 1) * (U + 1)] = F
                    continue
                last = qd[kk] * F
                earlier = np.sort(irng.random(U) * last)
                t[kk * (U + 1):kk * (U + 1) + U] = earlier
                t[kk * (U + 1) + U] = last
            return t

        placed, src = _place(ctx, irng, needles, T_ctx, target)
        for q, kk in enumerate(range(nk)):
            _query_meta_positions(queries[q], placed, [latest[kk]] + [kk * (U + 1) + u for u in range(U)], ctx)
        roles = [ROLE_QUERIED if j in latest else ROLE_STALE for j in range(len(needles))]
        meta = dict(context_id=i, filler_source=src, n_facts=len(needles), U=U, snap_rate=_snap_rate(placed))
        out.append(_finish(ctx, placed.context, queries, meta, _fact_arrays(fkeys, vals, placed, roles)))
    return out


def gen_r0(task_level: dict, T: int | None, n: int, rng: np.random.Generator, namespace: str, tok: Any,
           filler: FillerProvider | None = None) -> list[Item]:
    """R0 cloze passkey (Tier 0 sanity check, §9.6): one 4-digit pass key, one query.

    [IMPL] Repetitive (RULER/passkey 'noise') filler by default, as in the classic passkey
    task; depth stratified in deciles.
    """
    lvl = dict(task_level)
    lvl.setdefault("template", "PK")
    lvl.setdefault("filler", "repetitive")
    ctx = _Ctx("R0", lvl, T, rng, namespace, tok, filler)
    ts, tpl = ctx.ts, TP.PASSKEY
    out = []
    for i in range(n):
        irng = child_rng(rng, i)
        v = int(irng.integers(10_000))
        needle = TP.fact_ids(ts, tpl, np.zeros(0, np.int32), v)
        cvals, ctypes = [v], ["gold"]
        for a in _absent(irng, {v}):
            cvals.append(a)
            ctypes.append("absent")
        q = dict(prompt_ids=TP.query_prompt_ids(ts, tpl), answer_ids=TP.value_ids(ts, v),
                 candidates=[TP.value_ids(ts, c).tolist() for c in cvals],
                 meta=dict(query_idx=0, key="pass key", value=f"{v:04d}", cand_types=ctypes,
                           cand_values=[f"{c:04d}" for c in cvals]))
        T_ctx = _ctx_len(ctx, len(needle), _query_total([q], False))
        qd = decile_depths(ctx.cell_rng, i, 1, irng)
        placed, src = _place(ctx, irng, [needle], T_ctx, lambda F: np.asarray([qd[0] * F]))
        _query_meta_positions(q, placed, 0, ctx)
        meta = dict(context_id=i, filler_source=src, n_facts=1, snap_rate=_snap_rate(placed))
        arrays = dict(fact_key=np.zeros((1, 2), np.int32) - 1, fact_value=np.asarray([v], np.int32),
                      fact_start=placed.starts.astype(np.int32), fact_end=placed.ends.astype(np.int32),
                      fact_role=np.asarray([ROLE_QUERIED], np.int8))
        out.append(_finish(ctx, placed.context, [q], meta, arrays))
    return out


# =========================================================================== multi-hop
def gen_mhop(task_level: dict, T: int | None, n: int, rng: np.random.Generator, namespace: str, tok: Any,
             filler: FillerProvider | None = None) -> list[Item]:
    """M-hop: 4 chains "The code for {k_i} is the code for {k_{i-1}}." ending in
    "The code for {k_0} is {v}."; each chain queried once at its head k_h (§9.3).

    h = number of links. ``order`` (within each chain, chains interleaved at random):
    'shuffled' (CS default; random order), 'forward' (the value fact first, then each link
    after the sentence it refers to: k_0, k_1, ..., k_h), 'reversed' (head link first: k_h,
    ..., k_1, then the value fact) [IMPL naming]. Candidates: the 4 chain values + 4 absent.
    """
    ctx = _Ctx("M-hop", task_level, T, rng, namespace, tok, filler)
    h = int(task_level["h"])
    order = task_level.get("order", "shuffled")
    if order not in ("shuffled", "forward", "reversed"):
        raise ValueError(f"bad order {order!r}")
    ts, tpl = ctx.ts, _code_template(ctx)
    nc = Q_PER_CONTEXT
    out = []
    for i in range(n):
        irng = child_rng(rng, i)
        keys = _keys(ctx, irng, nc * (h + 1))  # chain c uses keys[c*(h+1) + j] as k_j
        vals = [int(v) for v in _values(irng, nc)]
        needles, nkeys, nvals, roles, chain_of, step_of = [], [], [], [], [], []
        for c in range(nc):
            kc = keys[c * (h + 1):(c + 1) * (h + 1)]
            needles.append(TP.fact_ids(ts, tpl, kc[0], vals[c]))
            nkeys.append(kc[0])
            nvals.append(vals[c])
            roles.append(ROLE_QUERIED)
            chain_of.append(c)
            step_of.append(0)
            for j in range(1, h + 1):
                needles.append(TP.link_ids(ts, tpl, kc[j], kc[j - 1]))
                nkeys.append(kc[j])
                nvals.append(-1)
                roles.append(ROLE_LINK)
                chain_of.append(c)
                step_of.append(j)
        used = set(vals)
        cands = []
        for c in range(nc):
            cv = [vals[c]] + [vals[o] for o in range(nc) if o != c]
            ct = ["gold"] + ["other_chain"] * (nc - 1)
            for a in _absent(irng, used):
                cv.append(a)
                ct.append("absent")
            cands.append((cv, ct))
        heads = [keys[c * (h + 1) + h] for c in range(nc)]
        queries = _code_queries(ctx, tpl, heads, vals, cands)
        tot = sum(len(x) for x in needles)
        T_ctx = _ctx_len(ctx, tot, _query_total(queries, ctx.sequential))
        chain_of_a, step_of_a = np.asarray(chain_of), np.asarray(step_of)

        def target(F: int) -> np.ndarray:
            t = np.full(len(needles), float(F)) if ctx.placement == "oracle" else irng.random(len(needles)) * F
            if order == "shuffled":
                return t if ctx.placement != "oracle" else t + irng.random(len(needles)) * 1e-3
            if ctx.placement == "oracle":  # all at the end; encode the order by tiny increments
                t = t + irng.random(len(needles)) * 1e-3
            for c in range(nc):
                idx = np.flatnonzero(chain_of_a == c)
                ts_sorted = np.sort(t[idx])
                rank = step_of_a[idx] if order == "forward" else h - step_of_a[idx]
                t[idx] = ts_sorted[rank]
            return t

        placed, src = _place(ctx, irng, needles, T_ctx, target)
        for c in range(nc):
            idx = np.flatnonzero(chain_of_a == c)
            _query_meta_positions(queries[c], placed, [int(j) for j in idx], ctx)  # first = the value fact
            queries[c]["meta"]["chain_positions_order"] = "value fact, link 1..h"
        meta = dict(context_id=i, filler_source=src, n_facts=len(needles), h=h, order=order,
                    snap_rate=_snap_rate(placed))
        out.append(_finish(ctx, placed.context, queries, meta, _fact_arrays(nkeys, nvals, placed, roles)))
    return out


# =========================================================================== aggregation
def _word_list_context(ctx: _Ctx, irng: np.random.Generator, probe_items: list[np.ndarray], T_ctx: int | None,
                       banned: set[int]) -> tuple[np.ndarray, dict]:
    """Comma-separated word list ``" w1, w2, ..., wm"`` of exactly ``T_ctx - 1`` tokens after the EOT.

    Probe occurrences (single tokens) plus background words (each <= 3 times, tokens avoid
    ``banned``). Placement 'spread': fully shuffled; 'oracle': background first, then the
    shuffled probe occurrences right before the query; 'nofill': probes only.
    """
    ts = ctx.ts
    comma = ts.enc(",")
    assert len(comma) == 1
    comma_id = int(comma[0])
    probes = [np.asarray(p, np.int32) for p in probe_items]
    probes = [probes[j] for j in irng.permutation(len(probes))]
    cost = lambda items: sum(len(x) + 1 for x in items) - 1  # noqa: E731  (commas between items)
    bg: list[np.ndarray] = []
    if ctx.placement != "nofill":
        budget = (T_ctx - 1) - cost(probes) - 1  # remaining tokens, each background item costs len+1
        wl = ctx.wl
        pool = wl.background_excluding(banned)
        bag = np.concatenate([irng.permutation(pool) for _ in range(AFREQ_MAX_BACKGROUND_OCC)])
        lens = np.asarray([len(wl.background[j]) for j in bag])
        k, rem = 0, budget + 1  # rem = tokens still to fill with (len+1)-cost items
        chosen: list[int] = []
        while k < len(bag) and rem - (lens[k] + 1) >= 5:
            chosen.append(int(bag[k]))
            rem -= lens[k] + 1
            k += 1
        if k >= len(bag) and rem > 5:
            raise InfeasibleItem(f"background pool too small for T_ctx={T_ctx} ({len(pool)} words)")
        # exact fit of the last rem (0..4) tokens with the remaining bag words
        counts: dict[int, int] = {}
        for j in chosen:
            counts[j] = counts.get(j, 0) + 1
        rest_by_len: dict[int, list[int]] = {}
        for j in bag[k:]:
            j = int(j)
            rest_by_len.setdefault(len(wl.background[j]) + 1, []).append(j)
        def take(c: int) -> int | None:  # noqa: E306
            for j in rest_by_len.get(c, []):
                if counts.get(j, 0) < AFREQ_MAX_BACKGROUND_OCC:
                    rest_by_len[c].remove(j)
                    counts[j] = counts.get(j, 0) + 1
                    return j
            return None
        while rem > 0:
            got = None
            for c in (rem, 2, 3, 4, 5):
                if c <= rem and rem - c != 1:
                    got = take(c)
                    if got is not None:
                        rem -= c
                        chosen.append(got)
                        break
            if got is None:
                if rem == 1 and chosen:  # swap: drop one item of cost c, then fill c + 1
                    j = chosen.pop()
                    counts[j] -= 1
                    rem += len(wl.background[j]) + 1
                    continue
                raise InfeasibleItem(f"cannot fit the word list exactly (rem={rem})")
        bg = [wl.background[j] for j in chosen]
        bg = [bg[j] for j in irng.permutation(len(bg))]
    if ctx.placement == "oracle":
        items = bg + probes
        probe_mask = np.r_[np.zeros(len(bg), bool), np.ones(len(probes), bool)]
    else:
        items = bg + probes
        probe_mask = np.r_[np.zeros(len(bg), bool), np.ones(len(probes), bool)]
        if ctx.placement == "spread":
            p = irng.permutation(len(items))
            items = [items[j] for j in p]
            probe_mask = probe_mask[p]
    pieces = [np.asarray([ts.eot_id], np.int32)]
    for j, it in enumerate(items):
        if j:
            pieces.append(np.asarray([comma_id], np.int32))
        pieces.append(it)
    out = np.concatenate(pieces).astype(np.int32)
    if T_ctx is not None and ctx.placement != "nofill":
        assert len(out) == T_ctx, (len(out), T_ctx)
    return out, dict(kind="list" if ctx.placement != "nofill" else "none", n_items=len(items),
                     n_background=len(bg), n_probe_occurrences=len(probes))


def gen_afreq(task_level: dict, T: int | None, n: int, rng: np.random.Generator, namespace: str, tok: Any,
              filler: FillerProvider | None = None) -> list[Item]:
    """A-freq: comma-separated word list; 4 probe pairs with n_Y = 8, n_X = round(8 rho);
    background words <= 3 occurrences each; query "Which word appears more often, {X} or {Y}?
    Answer:" -> the winner's single token (§9.3). The order of X and Y in the question is
    random. Candidates: [X], [Y] (two-alternative rank accuracy, chance 0.5).

    [IMPL] The whole context is the word list (the review's "numbered word list" design, here
    comma-separated per §9.3): the background words are the "filler" of this task, so C0 =
    probes only and C2 = probe occurrences right before the query.
    """
    ctx = _Ctx("A-freq", task_level, T, rng, namespace, tok, filler)
    rho = float(task_level["rho"])
    nX, nY = int(round(AFREQ_N_Y * rho)), AFREQ_N_Y
    ts = ctx.ts
    qtpl = TP.AFREQ_TEMPLATES[ctx.tname]
    out = []
    for i in range(n):
        irng = child_rng(rng, i)
        pw = irng.choice(len(ctx.wl.probe_ids), size=2 * AFREQ_PAIRS, replace=False)
        pids = [int(ctx.wl.probe_ids[j]) for j in pw]
        X, Y = pids[0::2], pids[1::2]  # X wins
        probe_items = []
        for x, y in zip(X, Y):
            probe_items += [np.asarray([x], np.int32)] * nX + [np.asarray([y], np.int32)] * nY
        queries = []
        for q, (x, y) in enumerate(zip(X, Y)):
            first, second = (x, y) if irng.random() < 0.5 else (y, x)
            prompt = TP.render(ts, qtpl, word_x=first, word_y=second)
            queries.append(dict(prompt_ids=prompt, answer_ids=np.asarray([x], np.int32), candidates=[[x], [y]],
                                meta=dict(query_idx=q, key=f"{ts.decode([x]).strip()}|{ts.decode([y]).strip()}",
                                          value=ts.decode([x]).strip(), cand_types=["gold", "loser"],
                                          winner_first=bool(first == x), n_x=nX, n_y=nY, depth=None,
                                          distance=None, needle_start=[], needle_end=[])))
        qtot = _query_total(queries, ctx.sequential)
        T_ctx = None if ctx.placement == "nofill" else ctx.T - qtot
        context, src = _word_list_context(ctx, irng, probe_items, T_ctx, set(pids) | set(ctx.wl.animal_ids.tolist()))
        # positions of each probe's occurrences
        for q, (x, y) in enumerate(zip(X, Y)):
            px, py = np.flatnonzero(context == x), np.flatnonzero(context == y)
            assert len(px) == nX and len(py) == nY, (len(px), len(py))
            queries[q]["meta"]["needle_start"] = [int(p) for p in np.sort(np.r_[px, py])]
            queries[q]["meta"]["distance"] = int(len(context) - 1 - max(px.max(), py.max()))
        meta = dict(context_id=i, filler_source=src, rho=rho, n_pairs=AFREQ_PAIRS)
        arrays = dict(probe_x=np.asarray(X, np.int32), probe_y=np.asarray(Y, np.int32))
        out.append(_finish(ctx, context, queries, meta, arrays))
    return out


def gen_acount(task_level: dict, T: int | None, n: int, rng: np.random.Generator, namespace: str, tok: Any,
               filler: FillerProvider | None = None) -> list[Item]:
    """A-count: word list (as A-freq) containing 4 queried animals (one query each) and 4
    distractor animals, each animal occurring c ~ U{1..9} times; "How many times does {A}
    appear in the list? Answer:" + " " -> one digit token (§9.4). Candidates: the 9 digits.
    [IMPL] list format, animal vocabulary, and the per-animal uniform count.
    """
    ctx = _Ctx("A-count", task_level, T, rng, namespace, tok, filler)
    ts = ctx.ts
    qtpl = TP.ACOUNT_TEMPLATES[ctx.tname]
    lo, hi = ACOUNT_RANGE
    na = ACOUNT_N_QUERIED + ACOUNT_N_DISTRACTORS
    if len(ctx.wl.animal_ids) < na:
        raise InfeasibleItem(f"word list has {len(ctx.wl.animal_ids)} single-token animals, need {na}")
    out = []
    for i in range(n):
        irng = child_rng(rng, i)
        aw = irng.choice(len(ctx.wl.animal_ids), size=na, replace=False)
        aids = [int(ctx.wl.animal_ids[j]) for j in aw]
        counts = [int(c) for c in irng.integers(lo, hi + 1, size=na)]
        probe_items = []
        for a, c in zip(aids, counts):
            probe_items += [np.asarray([a], np.int32)] * c
        queries = []
        digits = [ts.digit_ids[d] for d in range(lo, hi + 1)]
        for q in range(ACOUNT_N_QUERIED):
            prompt = np.concatenate([TP.render(ts, qtpl, word=aids[q]), [ts.space_id]]).astype(np.int32)
            gold = ts.digit_ids[counts[q]]
            cands = [[gold]] + [[d] for d in digits if d != gold]
            ctypes = ["gold"] + ["count"] * (len(cands) - 1)
            queries.append(dict(prompt_ids=prompt, answer_ids=np.asarray([gold], np.int32), candidates=cands,
                                meta=dict(query_idx=q, key=ts.decode([aids[q]]).strip(), value=str(counts[q]),
                                          cand_types=ctypes, depth=None, distance=None, needle_start=[],
                                          needle_end=[])))
        qtot = _query_total(queries, ctx.sequential)
        T_ctx = None if ctx.placement == "nofill" else ctx.T - qtot
        context, src = _word_list_context(ctx, irng, probe_items, T_ctx, set(aids))
        for q in range(ACOUNT_N_QUERIED):
            pa = np.flatnonzero(context == aids[q])
            assert len(pa) == counts[q], (len(pa), counts[q])
            queries[q]["meta"]["needle_start"] = [int(p) for p in pa]
            queries[q]["meta"]["distance"] = int(len(context) - 1 - pa.max())
        meta = dict(context_id=i, filler_source=src, counts=counts)
        arrays = dict(animals=np.asarray(aids, np.int32), counts=np.asarray(counts, np.int32))
        out.append(_finish(ctx, context, queries, meta, arrays))
    return out


# =========================================================================== registry
GENERATORS: dict[str, Callable[..., list[Item]]] = {
    "R1": gen_r1, "R-load": gen_rload, "R-dist": gen_rdist, "R3": gen_r3, "R0": gen_r0,
    "M-hop": gen_mhop, "A-freq": gen_afreq, "A-count": gen_acount,
}


def generate(task: str, task_level: dict, T: int | None, n: int, rng: np.random.Generator, namespace: str,
             tok: Any, filler: FillerProvider | None = None) -> list[Item]:
    """Dispatch to a cloze generator (Tier-0 LM tasks live in ``eval.tier0``)."""
    if task not in GENERATORS:
        raise KeyError(f"unknown cloze task {task!r}; Tier-0 LM tasks are in hybridlc.eval.tier0")
    return GENERATORS[task](task_level, T, n, rng, namespace, tok, filler)


def flatten_item(item: Item) -> tuple[np.ndarray, np.ndarray]:
    """(tokens, answer_mask) of a sequential (TEF) item: context + prompt_1 + answer_1 + ... ."""
    parts, mask = [item["context_ids"]], [np.zeros(len(item["context_ids"]), bool)]
    for q in item["queries"]:
        parts += [q["prompt_ids"], q["answer_ids"]]
        mask += [np.zeros(len(q["prompt_ids"]), bool), np.ones(len(q["answer_ids"]), bool)]
    return np.concatenate(parts).astype(np.int32), np.concatenate(mask)


def code_fact_len(tok: Any, template: str = "T1") -> int:
    """Token length of one fact of a code template (R-load's 12-token facts for T1)."""
    ts = token_space(tok)
    return TP.fact_len(ts, TP.CODE_TEMPLATES[template])

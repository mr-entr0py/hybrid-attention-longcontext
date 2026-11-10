"""Scoring of frozen item files (protocol §9.2, §9.8).

``score_file(model, item_path, out_jsonl, batch_size)`` writes one JSONL row per query
(cloze tasks) or per item (Tier-0 LM tasks) plus a ``<stem>_meta.json`` header, and returns
the path written. Files are never overwritten: if ``out_jsonl`` exists, ``.r1``, ``.r2``, ...
is inserted before the suffix (the caller adds the registry note).

Cloze path (query branching, §9.1-9.2): one ``model.prefill(ctx[B, T_ctx])`` per batch of
contexts, then every query/candidate continuation is scored from the *same* cache with
``model.score_continuation(cache, cont[B, M])`` (which must not mutate the cache). The gold
continuation is ``prompt + answer``; EM_tf = every answer token is the argmax given the gold
prefix; ``logp_gold`` = summed answer-token log-probabilities. Candidates (gold first, then
in-context values, then absent values; equal token length) are scored as ``prompt +
candidate``; rank = 1 + #(other candidates with logp >= gold) (ties count against the
model), rank accuracy = (rank == 1). Error type of a miss = the type of the top candidate:
'binding' (wrong in-context value), 'absent', 'stale' (R3), 'wrong_chain' (M-hop),
'loser' (A-freq), 'wrong_count' (A-count), or 'non_candidate' (gold tops the candidates but
EM_tf fails).

Only continuation positions j >= len(prompt) >= 1 are read, i.e. the scorer never relies on
the j = 0 prediction (see the alignment convention in INTERFACES.md).

Tier-0 LM path (``score_lm_file``): one varlen ``model.forward`` per item, fp32 logits per
chunk, per-token NLL aggregated into the LM-1 position buckets, LM-3 distance buckets,
LM-2 totals or LM-4 copy losses.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from hybridlc.eval.constants import (AFREQ_WINDOW_FLOORS, LM1_BUCKET_EDGES, LM1_MAX_FWE_DOCS, LM1_N_PG19_BOOKS, LM3_DELTA_EDGES,
                                     LM4_SCORED_FROM, RESCORE_FRACTION, frozen_batch_size)
from hybridlc.eval.items import load_items, sha256_file

ERROR_OF_TYPE = {"context": "binding", "absent": "absent", "stale": "stale", "other_chain": "wrong_chain",
                 "loser": "loser", "count": "wrong_count"}
LM_TASKS = ("LM-1", "LM-2", "LM-4")
IGNORE_INDEX = 1


# =========================================================================== helpers
def _device(model: Any) -> torch.device:
    for attr in ("device",):
        d = getattr(model, attr, None)
        if isinstance(d, torch.device):
            return d
    w = getattr(model, "lm_head_weight", None)
    if isinstance(w, torch.Tensor):
        return w.device
    return next(model.parameters()).device


def _scoring_path(model: Any) -> str:
    return str(getattr(model, "scoring_path", "prefill+score_continuation(branched cache)"))


def unique_path(p: str | Path) -> Path:
    """``p`` if free, else ``stem.r1.suffix``, ``stem.r2.suffix``, ... (never overwrite)."""
    p = Path(p)
    if not p.exists():
        return p
    i = 1
    while True:
        q = p.with_name(f"{p.stem}.r{i}{p.suffix}")
        if not q.exists():
            return q
        i += 1


def _header(model: Any, item_path: Path, extra: dict) -> dict:
    from hybridlc.eval.items import eval_code_sha256

    h: dict[str, Any] = dict(item_file=str(item_path), item_sha256=sha256_file(item_path),
                             eval_code_sha256=eval_code_sha256())
    try:
        from hybridlc.utils.provenance import env_info

        h["env"] = env_info()
    except Exception as e:  # noqa: BLE001
        h["env_error"] = repr(e)
    cfg = getattr(model, "cfg", None) or getattr(getattr(model, "model", None), "cfg", None)
    if cfg is not None:
        try:
            import dataclasses

            from hybridlc.utils.provenance import config_hash

            h["model_config_hash"] = config_hash(dataclasses.asdict(cfg) if dataclasses.is_dataclass(cfg) else cfg)
        except Exception as e:  # noqa: BLE001
            h["model_config_hash_error"] = repr(e)
    h.update(extra)
    return h


def _write_meta(out: Path, header: dict) -> Path:
    mp = out.with_name(out.stem + "_meta.json")
    mp.write_text(json.dumps(header, indent=1, sort_keys=True, default=str) + "\n")
    return mp


# =========================================================================== cloze scoring
def _top_candidate(lp: list[float]) -> int:
    """Index of the top candidate; the gold (index 0) loses ties."""
    others = lp[1:]
    if not others:
        return 0
    j = int(np.argmax(others)) + 1
    return j if lp[j] >= lp[0] else 0


def _autocast(dev: torch.device, enabled: bool | None):
    """bf16 autocast on CUDA by default: the canonical inference precision (INTERFACES.md, models)."""
    on = (dev.type == "cuda") if enabled is None else bool(enabled)
    if on and dev.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    import contextlib

    return contextlib.nullcontext()


def score_items(model: Any, items: list[dict], batch_size: int, cell_name: str = "",
                score_candidates: bool = True, autocast: bool | None = None) -> list[dict]:
    """Score cloze items with the branched-cache path; returns one row dict per query."""
    with torch.no_grad(), _autocast(_device(model), autocast):
        return _score_items(model, items, batch_size, cell_name, score_candidates)


def _score_items(model: Any, items: list[dict], batch_size: int, cell_name: str, score_candidates: bool) -> list[dict]:
    dev = _device(model)
    path = _scoring_path(model)
    rows: list[dict] = []
    # group contexts by (T_ctx, n_queries) so prefill batches have no padding
    groups: dict[tuple[int, int], list[int]] = {}
    for i, it in enumerate(items):
        groups.setdefault((len(it["context_ids"]), len(it["queries"])), []).append(i)
    for (_, nq), idxs in sorted(groups.items()):
        for a in range(0, len(idxs), batch_size):
            bidx = idxs[a:a + batch_size]
            B = len(bidx)
            ctx = torch.from_numpy(np.stack([items[i]["context_ids"] for i in bidx]).astype(np.int64)).to(dev)
            _, cache = model.prefill(ctx)
            for q in range(nq):
                qs = [items[i]["queries"][q] for i in bidx]
                P = {len(x["prompt_ids"]) for x in qs}
                A = {len(x["answer_ids"]) for x in qs}
                if len(P) != 1 or len(A) != 1:
                    raise ValueError(f"{cell_name}: prompt/answer lengths differ within a batch: {P}, {A}")
                P, A = P.pop(), A.pop()
                prompt = np.stack([x["prompt_ids"] for x in qs]).astype(np.int64)
                ans = np.stack([x["answer_ids"] for x in qs]).astype(np.int64)
                cont = torch.from_numpy(np.concatenate([prompt, ans], 1)).to(dev)
                lp, am = model.score_continuation(cache, cont)
                lp_a = lp[:, P:P + A].float().cpu().numpy()
                am_a = am[:, P:P + A].cpu().numpy()
                em = (am_a == ans).all(1)
                nc = max(len(x["candidates"]) for x in qs)
                cand_lp = np.full((B, nc), -np.inf)
                cand_lp[:, 0] = lp_a.sum(1)
                if score_candidates and nc > 1:
                    # Candidates share the prompt: advance a copy of the cache through it once
                    # (model.extend, cache not mutated) and score each candidate from there.
                    # The gold/EM path above stays the canonical prompt+answer continuation.
                    p_cache = None
                    if hasattr(model, "extend"):
                        _, p_cache = model.extend(torch.from_numpy(prompt).to(dev), cache)
                    for c in range(1, nc):
                        cc = np.stack([np.asarray(x["candidates"][c] if c < len(x["candidates"]) else x["candidates"][0])
                                       for x in qs]).astype(np.int64)
                        if cc.shape[1] != A:
                            raise ValueError(f"{cell_name}: candidate length {cc.shape[1]} != answer length {A}")
                        if p_cache is not None:
                            lpc, _ = model.score_continuation(p_cache, torch.from_numpy(cc).to(dev))
                            cand_lp[:, c] = lpc.float().sum(1).cpu().numpy()
                        else:
                            cont_c = torch.from_numpy(np.concatenate([prompt, cc], 1)).to(dev)
                            lpc, _ = model.score_continuation(cache, cont_c)
                            cand_lp[:, c] = lpc[:, P:P + A].float().sum(1).cpu().numpy()
                    del p_cache
                for r, i in enumerate(bidx):
                    it, qq = items[i], qs[r]
                    m, im = qq["meta"], it["meta"]
                    ncr = len(qq["candidates"])
                    clp = cand_lp[r, :ncr].tolist()
                    if score_candidates and ncr > 1:
                        rank = 1 + int(sum(1 for v in clp[1:] if v >= clp[0]))
                        top = _top_candidate(clp)
                    else:
                        rank, top = None, None
                    ctypes = m.get("cand_types", ["gold"] * ncr)
                    if bool(em[r]) and (rank in (None, 1)):
                        err = "none"
                    elif top is not None and top > 0:
                        err = ERROR_OF_TYPE.get(ctypes[top], ctypes[top])
                    else:
                        err = "non_candidate"
                    rows.append(dict(
                        item_id=f"{cell_name}:{im['context_id']}:{m['query_idx']}", context_id=im["context_id"],
                        query_idx=m["query_idx"], task=im.get("task"), template=im.get("template"),
                        filler_source=(im.get("filler_source") or {}).get("kind"), T=im.get("T"), T_ctx=im.get("T_ctx"),
                        depth=m.get("depth"), distance=m.get("distance"), needle_start=m.get("needle_start"),
                        needle_end=m.get("needle_end"), gold_ids=ans[r].tolist(), argmax_ids=am_a[r].tolist(),
                        EM_tf=bool(em[r]), logp_gold=float(lp_a[r].sum()), logp_gold_tokens=lp_a[r].tolist(),
                        cand_logp=[float(v) for v in clp] if score_candidates else None, cand_types=ctypes,
                        rank=rank, rank_correct=None if rank is None else bool(rank == 1), n_cand=ncr,
                        chance=1.0 / ncr, error_type=err, scoring_path=path,
                        absent_in_context=_absent_in_context(it, qq, ctypes),
                        **_afreq_heuristics(it, qq, im.get("task"))))
            del cache
    return rows


def _contains(ctx: np.ndarray, sub: np.ndarray) -> bool:
    L = len(sub)
    if L == 0 or L > len(ctx):
        return False
    pos = np.flatnonzero(ctx[:len(ctx) - L + 1] == sub[0])
    return any(np.array_equal(ctx[p:p + L], sub) for p in pos)


def _absent_in_context(item: dict, q: dict, ctypes: list[str]) -> list[int] | None:
    """Indices of 'absent'-typed candidates whose token sequence nevertheless occurs in the context
    (e.g. a 4-digit year in the natural filler; 0.03-0.44% of absent candidates in the frozen cells).

    The frozen items draw absent values excluding only the fact values (verification round 2,
    ABSENT-CANDIDATES-PRESENT-IN-FILLER); this field lets the analysis treat such candidates as
    in-context when it reports error types. ``None`` when the query has no absent candidates.
    """
    idx = [c for c, t in enumerate(ctypes) if t == "absent" and c < len(q["candidates"])]
    if not idx:
        return None
    ctx = np.asarray(item["context_ids"])
    return [c for c in idx if _contains(ctx, np.asarray(q["candidates"][c]))]


def _afreq_heuristics(item: dict, q: dict, task: str | None) -> dict:
    """A-freq counting-free heuristic floors (verification round 1, AFREQ-RECENCY-FLOOR).

    In a uniformly shuffled list frequency and recency are confounded: "answer the probe that occurs
    last" (or first) scores ~rho/(1+rho), not ``chance`` = 1/2. Each A-freq row records whether these
    two heuristics would be correct on this query, so the per-cell heuristic floor can be reported
    next to the score (empty dict for other tasks; None when not computable).

    Verification round 3 (AFREQ-SHORT-WINDOW-FLOOR): rows also carry ``window{W}_count_heuristic``
    for W in ``AFREQ_WINDOW_FLOORS`` -- the score (1, 0, or 1/2 on a tie) of a counter that sees only
    the last W context tokens. At T <= 4K a last-1,024-token counter is near ceiling, so short-T
    A-freq measures local frequency comparison, not long-range integration; the floor that matters
    for a short-memory mechanism is this one, not the recency floor.
    """
    if task != "A-freq":
        return {}
    ctx = np.asarray(item["context_ids"])
    x = int(q["answer_ids"][0])
    others = [int(c[0]) for c in q["candidates"] if int(c[0]) != x]
    px = np.flatnonzero(ctx == x)
    py = np.flatnonzero(ctx == others[0]) if len(others) == 1 else np.zeros(0, int)
    win_keys = [f"window{W}_count_heuristic" for W in AFREQ_WINDOW_FLOORS]
    if not len(px) or not len(py):
        return dict(recency_heuristic_correct=None, primacy_heuristic_correct=None, **{k: None for k in win_keys})
    out: dict[str, Any] = dict(recency_heuristic_correct=bool(px.max() > py.max()),
                               primacy_heuristic_correct=bool(px.min() < py.min()))
    n = len(ctx)
    for k, W in zip(win_keys, AFREQ_WINDOW_FLOORS):
        cx, cy = int((px >= n - W).sum()), int((py >= n - W).sum())
        out[k] = 1.0 if cx > cy else (0.0 if cx < cy else 0.5)
    return out


# =========================================================================== LM scoring
@torch.no_grad()
def _nll(model: Any, ids: np.ndarray, seg_starts: np.ndarray | None, labels: np.ndarray | None,
         chunk: int = 4096) -> np.ndarray:
    """Per-position NLL of labels (default: next token; last position NaN) for one sequence."""
    dev = _device(model)
    N = len(ids)
    st = np.asarray([0] if seg_starts is None else seg_starts, dtype=np.int64)
    cu = np.r_[st, N].astype(np.int32)
    pos = np.arange(N) - np.repeat(st, np.diff(cu))
    x = torch.from_numpy(ids.astype(np.int64))[None].to(dev)
    h = model(x, torch.from_numpy(cu).to(dev), torch.from_numpy(pos)[None].to(dev))[0]
    w = model.lm_head_weight
    if labels is None:
        labels = np.r_[ids[1:], IGNORE_INDEX].astype(np.int64)
        valid = np.r_[np.ones(N - 1, bool), False]
    else:
        valid = labels != IGNORE_INDEX
    lab = torch.from_numpy(np.where(valid, labels, 0).astype(np.int64)).to(dev)
    out = torch.empty(N, dtype=torch.float32, device=dev)
    # fp32 logits: the GEMM runs with autocast disabled, otherwise it is executed in bf16 inside
    # ``score_lm_items``' autocast region and returns rounded bf16 logits (verification round 2,
    # TIER0-LM-LOGITS-BF16). The hidden states h come from the bf16-autocast forward as before.
    with torch.autocast(device_type=dev.type, enabled=False):
        wf = w.float()
        for a in range(0, N, chunk):
            lg = h[a:a + chunk].float() @ wf.T
            out[a:a + chunk] = torch.nn.functional.cross_entropy(lg, lab[a:a + chunk], reduction="none")
    o = out.cpu().numpy().astype(np.float64)
    o[~valid] = np.nan
    return o


def _bucket_sums(values: np.ndarray, keys: np.ndarray, edges: tuple[int, ...]) -> tuple[list[float], list[int]]:
    b = np.searchsorted(np.asarray(edges), keys, side="right") - 1
    ok = (b >= 0) & (b < len(edges) - 1) & ~np.isnan(values)
    s = np.bincount(b[ok], weights=values[ok], minlength=len(edges) - 1)
    c = np.bincount(b[ok], minlength=len(edges) - 1)
    return s.tolist(), c.astype(int).tolist()


def score_lm_items(model: Any, items: list[dict], cell_name: str = "", autocast: bool | None = None) -> list[dict]:
    with torch.no_grad(), _autocast(_device(model), autocast):
        return _score_lm_items(model, items, cell_name)


def _score_lm_items(model: Any, items: list[dict], cell_name: str) -> list[dict]:
    rows = []
    for it in items:
        m, arr = it["meta"], it.get("arrays", {})
        task = m["task"]
        ids = np.asarray(it["context_ids"])
        if task == "LM-2":
            nll = _nll(model, ids, arr["seg_starts"], np.asarray(arr["labels"]).astype(np.int64))
            ok = ~np.isnan(nll)
            row = dict(loss_sum=float(nll[ok].sum()), n_tokens=int(ok.sum()))
        else:
            nll = _nll(model, ids, None, None)
            # nll[t] is the loss of predicting token t+1 (t+1 preceding tokens incl. the EOT)
            if task == "LM-1":
                tgt_pos = np.arange(len(ids)) + 1
                s, c = _bucket_sums(nll, tgt_pos, LM1_BUCKET_EDGES)
                row = dict(pos_bucket_edges=list(LM1_BUCKET_EDGES), pos_loss_sum=s, pos_count=c)
                if "lm3_pos" in arr:
                    p = np.asarray(arr["lm3_pos"]).astype(np.int64)
                    v = nll[p - 1]  # loss of token at position p
                    s3, c3 = _bucket_sums(v, np.asarray(arr["lm3_delta"]), LM3_DELTA_EDGES)
                    row.update(lm3_delta_edges=list(LM3_DELTA_EDGES), lm3_loss_sum=s3, lm3_count=c3)
            elif task == "LM-4":
                P, a1, a2 = m["p_len"], m["p1_start"], m["p2_start"]
                sl1 = nll[a1 + LM4_SCORED_FROM - 1:a1 + P - 1]  # tokens 9..256 of P (1-based)
                sl2 = nll[a2 + LM4_SCORED_FROM - 1:a2 + P - 1]
                row = dict(variant=m["variant"], G=m["G"], loss_p1=float(np.mean(sl1)), loss_p2=float(np.mean(sl2)),
                           copy_gain=float(np.mean(sl1) - np.mean(sl2)), n_scored=int(len(sl1)))
            else:
                raise ValueError(f"unknown LM task {task}")
        row.update(item_id=f"{cell_name}:{m['context_id']}", context_id=m["context_id"], task=task, T=len(ids),
                   filler_source=m.get("filler_source"), scoring_path="forward(varlen)+chunked fp32 logits")
        rows.append(row)
    return rows


def lm_source_counts(rows: list[dict], task: str | None) -> dict:
    """Per-source item counts of a Tier-0 LM cell for the result header.

    For LM-1 the planned counts (§9.6: up to 300 FWE test-half documents >= 32,768 tokens plus 100
    PG19-test books) are reported next to the actual ones: only 78 of the 100 PG19-test books have
    >= 32,768 tokens, so the frozen LM-1 cell holds 201 FWE + 78 PG19 = 279 items (DEVIATION recorded
    in reports/s0_integration.md §12, verification round 3, LM1-PG19-BOOK-SHORTFALL).
    """
    counts: dict[str, int] = {}
    for r in rows:
        fs = r.get("filler_source")
        k = (fs.get("kind") if isinstance(fs, dict) else fs) or "unknown"
        counts[str(k)] = counts.get(str(k), 0) + 1
    out: dict[str, Any] = {"source_counts": dict(sorted(counts.items()))}
    if task == "LM-1":
        out["source_counts_planned"] = {"fwe": LM1_MAX_FWE_DOCS, "pg19": LM1_N_PG19_BOOKS}
        out["source_note"] = ("FWE: every eligible (>= T tokens) test-half document up to the cap; PG19: the "
                              "PG19-test books with >= T tokens (books shorter than T are excluded)")
    return out


# =========================================================================== entry points
def score_file(model: Any, item_path: str | Path, out_jsonl: str | Path, batch_size: int | None = None,
               score_candidates: bool = True, extra_meta: dict | None = None, autocast: bool | None = None) -> Path:
    """Score one item file; write JSONL rows + ``<stem>_meta.json``; return the JSONL path.

    ``batch_size=None`` uses the frozen per-length batch size (§9.2); a different value is
    recorded in the header as a deviation.
    """
    item_path = Path(item_path)
    t0 = time.time()
    items, cell = load_items(item_path)
    name = item_path.stem
    T = cell.get("T") or max(len(it["context_ids"]) for it in items)
    frozen = frozen_batch_size(int(T))
    bs = int(batch_size or frozen)
    if hasattr(model, "eval"):
        model.eval()
    dev = _device(model)
    use_ac = (dev.type == "cuda") if autocast is None else bool(autocast)
    if cell.get("task") in LM_TASKS:
        rows = score_lm_items(model, items, name, autocast=use_ac)
    else:
        rows = score_items(model, items, bs, name, score_candidates, autocast=use_ac)
    out = unique_path(out_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "x") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True, allow_nan=True) + "\n")
    if cell.get("task") in LM_TASKS:
        extra_meta = {**lm_source_counts(rows, cell.get("task")), **(extra_meta or {})}
    hdr = _header(model, item_path, {**dict(cell=cell, n_rows=len(rows), batch_size=bs, frozen_batch_size=frozen,
                                            batch_size_deviation=bs != frozen, scoring_path=_scoring_path(model),
                                            autocast_bf16=use_ac and dev.type == "cuda", device=str(dev),
                                            wall_time_s=time.time() - t0, out_jsonl=str(out)), **(extra_meta or {})})
    _write_meta(out, hdr)
    return out


def rescore_subset(n_contexts: int, cell_key: str, frac: float = RESCORE_FRACTION) -> list[int]:
    """Deterministic 5% context subset for the single-pass re-scoring check (§9.2, I-29)."""
    from hybridlc.eval.rng import eval_rng

    k = max(1, int(round(frac * n_contexts)))
    return sorted(int(i) for i in eval_rng("test", "rescore", cell_key).choice(n_contexts, size=k, replace=False))


#: FEAS-PREC v2 P3: fp32 single-pass references at T >= this length use the R-ATT fp32 reference
#: attention (MATH below; PI ruling D083).
RATT_MIN_T = 16384


def precision_flags() -> dict:
    """TF32 / matmul-precision flags in force (recorded in every FEAS-PREC v2 record)."""
    return dict(matmul_allow_tf32=bool(torch.backends.cuda.matmul.allow_tf32),
                cudnn_allow_tf32=bool(torch.backends.cudnn.allow_tf32),
                float32_matmul_precision=torch.get_float32_matmul_precision())


def rescore_check(model: torch.nn.Module, item_path: str | Path, rows_jsonl: str | Path,
                  frac: float = RESCORE_FRACTION, ref: str = "bf16", ref_model: torch.nn.Module | None = None) -> dict:
    """Re-score a 5% subset with a single-pass forward (``ForwardOnlyAdapter``) and compare EM_tf.

    ``ref="bf16"`` (default) is the frozen protocol §9.2 / I-29 check (FEAS-PREC v2 P2): the same
    model (bf16-rounded weights) under the same bf16 autocast as the scored rows; the protocol requires
    flip_rate < 1% and ``analysis.results.rescore_audit`` gates on this record. ``ref="fp32"`` is
    FEAS-PREC v2 P3: ``ref_model`` (the checkpoint's fp32 weights; default ``model``), autocast off,
    attention through R-ATT for T >= 16K (MATH below), TF32 flags as in force (recorded).

    Returns ``dict(n, flips, flip_rate, max_abs_dlogp, contexts, ref, pass_id, ...precision fields)``.
    """
    from hybridlc.eval.adapter import ForwardOnlyAdapter

    if ref not in ("bf16", "fp32"):
        raise ValueError(f"rescore_check ref must be 'bf16' or 'fp32', got {ref!r}")
    import contextlib

    items, cell = load_items(item_path)
    name = Path(item_path).stem
    sub = rescore_subset(len(items), name, frac)
    T = int(cell.get("T") or max(len(it["context_ids"]) for it in items))
    if ref == "bf16":
        m, use_ac, att = model, _device(model).type == "cuda", contextlib.nullcontext()
        backend, weights = "as P0 (bf16 autocast on CUDA)", "as scored (P0 weights)"
    else:
        m = ref_model if ref_model is not None else model
        use_ac = False
        weights = "ref_model (fp32 weights)" if ref_model is not None else "as scored (P0 weights; no ref_model)"
        if T >= RATT_MIN_T:
            from hybridlc.models.attention import fp32_reference_attention

            att, backend = fp32_reference_attention("tiled"), "R-ATT:tiled-fp32"
        else:
            att, backend = contextlib.nullcontext(), "MATH (fp32)"
    with att:
        ref_rows = score_items(ForwardOnlyAdapter(m), [items[i] for i in sub], 1, name, score_candidates=False,
                               autocast=use_ac)
    pass_meta = dict(ref=ref, pass_id="P2" if ref == "bf16" else "P3", autocast=bool(use_ac and _device(m).type == "cuda"),
                     attention_backend=backend, weights=weights, **precision_flags())
    stored = {}
    with open(rows_jsonl) as f:
        for line in f:
            r = json.loads(line)
            stored[r["item_id"]] = r
    flips, dmax = 0, 0.0
    for r in ref_rows:
        s = stored[r["item_id"]]
        flips += int(s["EM_tf"] != r["EM_tf"])
        dmax = max(dmax, abs(s["logp_gold"] - r["logp_gold"]))
    n = len(ref_rows)
    rows = [dict(item_id=r["item_id"], EM_tf=bool(r["EM_tf"]), logp_gold=float(r["logp_gold"])) for r in ref_rows]
    return dict(n=n, flips=flips, flip_rate=flips / max(n, 1), max_abs_dlogp=dmax, contexts=sub, rows=rows,
                **pass_meta)

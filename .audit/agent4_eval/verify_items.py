#!/usr/bin/env python
"""Agent-4 (evaluation audit) item verifier. READ-ONLY on experiments/items.

For every item file of the given namespace(s)/suite(s) (or a sample of items per file) checks:
  exact T, BOS/EOT layout, no separator, answer tokenisation, candidate sets (equal length, gold
  first, distinct, types/counts), needle span == rendered fact, key uniqueness in the context,
  gold value occurrences, depth deciles, distances, filler pools, snap rate; and for A-freq /
  A-count the counts and the heuristic shortcut rates; for M-hop the chain structure and the
  shortcut heuristics.

Writes one JSON line per file to --out and prints a compact summary.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path("/home/sedrica-lynx/Downloads/outeropt")
sys.path.insert(0, str(REPO / "src"))

from hybridlc.data.tokenizer import load_tokenizer  # noqa: E402
from hybridlc.eval import templates as TP  # noqa: E402
from hybridlc.eval.items import load_items  # noqa: E402
from hybridlc.eval.tokens import token_space  # noqa: E402
from hybridlc.eval.words import word_list  # noqa: E402

CODE_TASKS = ("R1", "R-load", "R-dist", "R3", "M-hop", "R0")
POOLS = {"test": {"pg19/test", "fwe/test"}, "dev": {"pg19/validation", "fwe/dev"}}


def count_sub(ctx: np.ndarray, sub: np.ndarray) -> list[int]:
    L = len(sub)
    if L == 0 or L > len(ctx):
        return []
    pos = np.flatnonzero(ctx[: len(ctx) - L + 1] == sub[0])
    return [int(p) for p in pos if np.array_equal(ctx[p:p + L], sub)]


def check_code_item(it, cell, ts, wlB, acc, fails):
    ctx = np.asarray(it["context_ids"])
    m = it["meta"]
    T = cell["T"]
    task = cell["task"]
    tname = m.get("template", "T1")
    tpl = TP.PASSKEY if tname == "PK" else TP.CODE_TEMPLATES[tname]
    a = it["arrays"]
    # layout
    if ctx[0] != ts.eot_id:
        fails["bos_missing"] += 1
    if (ctx[1:] == ts.eot_id).any():
        fails["eot_inside_ctx"] += 1
    # needle total vs C0
    if T is None:
        tot_needles = int((a["fact_end"] - a["fact_start"]).sum())
        if len(ctx) != 1 + tot_needles:
            fails["c0_len"] += 1
    # filler pool
    fs = m.get("filler_source") or {}
    acc["filler_kind"][fs.get("kind")] += 1
    if fs.get("pool") is not None:
        acc["filler_pool"][fs.get("pool")] += 1
        if fs["pool"] not in POOLS[cell["namespace"]]:
            fails["wrong_pool"] += 1
    acc["snap_rate"].append(m.get("snap_rate", 1.0))
    # facts overlap / order
    order = np.argsort(a["fact_start"])
    st, en = a["fact_start"][order], a["fact_end"][order]
    if (st[1:] < en[:-1]).any():
        fails["fact_overlap"] += 1
    # verify each fact span renders correctly
    for f in range(len(a["fact_start"])):
        span = ctx[int(a["fact_start"][f]):int(a["fact_end"][f])]
        role = int(a["fact_role"][f])
        if task == "R0":
            exp = TP.fact_ids(ts, tpl, np.zeros(0, np.int32), int(a["fact_value"][f]))
        elif role == 2:  # link: key, prev (prev not stored -> check prefix/suffix only)
            exp = None
            pre = TP.render(ts, tpl.link[:2], key=a["fact_key"][f])
            if not np.array_equal(span[:len(pre)], pre):
                fails["link_render"] += 1
        else:
            exp = TP.fact_ids(ts, tpl, a["fact_key"][f], int(a["fact_value"][f]))
        if exp is not None and not np.array_equal(span, exp):
            fails["fact_render"] += 1
    qlens = set()
    for q in it["queries"]:
        p, ans = np.asarray(q["prompt_ids"]), np.asarray(q["answer_ids"])
        qm = q["meta"]
        qlens.add(len(p) + len(ans))
        if T is not None and len(ctx) + len(p) + len(ans) != T:
            fails["T_mismatch"] += 1
        if (p == ts.eot_id).any() or (ans == ts.eot_id).any():
            fails["eot_in_query"] += 1
        if len(ans) != 4 or not all(int(t) in ts.digit_set for t in ans):
            fails["answer_not_4digits"] += 1
        if p[-1] != ts.space_id:
            fails["prompt_not_space_end"] += 1
        if ts.decode(ans) != qm["value"]:
            fails["answer_value_mismatch"] += 1
        # candidates
        cands = [np.asarray(c) for c in q["candidates"]]
        if any(len(c) != len(ans) for c in cands):
            fails["cand_len"] += 1
        if not np.array_equal(cands[0], ans):
            fails["cand0_not_gold"] += 1
        if len({tuple(c.tolist()) for c in cands}) != len(cands):
            fails["cand_dup"] += 1
        ct = qm["cand_types"]
        acc["n_cand"][len(cands)] += 1
        cnt = Counter(ct)
        if cnt.get("absent", 0) != 4:
            fails["absent_ne4"] += 1
        if cnt.get("context", 0) + cnt.get("stale", 0) > 32:
            fails["ctx_cands_gt32"] += 1
        fvals = set(int(v) for v in a["fact_value"].tolist() if v >= 0)
        for c, t in zip(qm["cand_values"], ct):
            if t == "absent" and int(c) in fvals:
                fails["absent_is_fact_value"] += 1
            if t in ("context", "stale", "other_chain") and int(c) not in fvals:
                fails["context_cand_not_fact"] += 1
        # absent candidate present in context
        for c, t in zip(cands, ct):
            if t == "absent":
                acc["absent_total"] += 1
                if count_sub(ctx, c):
                    acc["absent_in_ctx"] += 1
        # queried needle
        ns_, ne_ = int(qm["needle_start"][0]), int(qm["needle_end"][0])
        if task != "M-hop":
            span = ctx[ns_:ne_]
            if task == "R0":
                key = np.zeros(0, np.int32)
            elif tname == "T4":
                key = p[:2]
            else:
                off = len(TP.render(ts, tpl.query[:1]))
                key = p[off:off + 2]
            exp = TP.fact_ids(ts, tpl, key, int(qm["value"]))
            if not np.array_equal(span, exp):
                fails["queried_span_mismatch"] += 1
            if qm.get("distance") is not None and qm["distance"] != len(ctx) - ne_:
                fails["distance_mismatch"] += 1
            # prompt (without trailing space) occurrences in context: should be exactly once (the fact)
            occ = count_sub(ctx, p[:-1])
            acc["prompt_core_occ"][len(occ)] += 1
            # full prompt incl. space
            occ_full = count_sub(ctx, p)
            if task in ("R1", "R-load", "R-dist") and len(occ_full) != 1:
                fails["prompt_full_occ_ne1"] += 1
            if task != "R0":
                kocc = count_sub(ctx, key)
                acc["key_bigram_occ"][len(kocc)] += 1
            # gold digits occurrences in the context
            gocc = count_sub(ctx, ans)
            acc["gold_digit_occ"][min(len(gocc), 5)] += 1
            # value-token-window collisions: other facts sharing first 1/2/3 digits
            gv = qm["value"]
            others = [f"{int(v):04d}" for v in a["fact_value"].tolist() if f"{int(v):04d}" != gv and v >= 0]
            for kk in (1, 2, 3):
                acc[f"share_prefix{kk}"].append(sum(o[:kk] == gv[:kk] for o in others))
        else:
            # M-hop: head key appears once (link subject); full prompt (with space) never in ctx
            key = p[3:5] if tname == "T1" else p[:2]
            kocc = count_sub(ctx, key)
            acc["mhop_head_key_occ"][len(kocc)] += 1
            occ_full = count_sub(ctx, p)
            acc["mhop_full_prompt_occ"][len(occ_full)] += 1
        if qm.get("depth") is not None:
            acc["depth"].append(qm["depth"])
        if qm.get("distance") is not None:
            acc["distance"].append(qm["distance"])
    if len(qlens) != 1:
        fails["branch_len_unequal"] += 1


def mhop_shortcuts(it, acc):
    """Shortcut heuristics on M-hop items using the stored fact arrays."""
    a = it["arrays"]
    role, st, en, val = a["fact_role"], a["fact_start"], a["fact_end"], a["fact_value"]
    fkey = a["fact_key"]
    vf = np.flatnonzero(role == 1)  # value facts (one per chain)
    T_ctx = len(it["context_ids"])
    h = it["meta"]["h"]
    # chains: value fact index c*(h+1), links follow in input order
    nchains = len(vf)
    for qi, q in enumerate(it["queries"]):
        gold = int(q["meta"]["value"])
        c = qi  # chain c queried by query c
        base = c * (h + 1)
        head = base + h
        hpos = int(st[head])
        vpos = st[vf]
        vvals = val[vf]
        # H_last: most recent value fact
        acc["H_last"].append(int(vvals[np.argmax(vpos)]) == gold)
        acc["H_first"].append(int(vvals[np.argmin(vpos)]) == gold)
        acc["H_near_head"].append(int(vvals[np.argmin(np.abs(vpos - hpos))]) == gold)
        aft = vpos > hpos
        if aft.any():
            j = np.flatnonzero(aft)[np.argmin(vpos[aft] - hpos)]
            acc["H_next_after_head"].append(int(vvals[j]) == gold)
        else:
            acc["H_next_after_head"].append(int(vvals[np.argmax(vpos)]) == gold)
        bef = vpos < hpos
        if bef.any():
            j = np.flatnonzero(bef)[np.argmax(vpos[bef])]
            acc["H_prev_before_head"].append(int(vvals[j]) == gold)
        else:
            acc["H_prev_before_head"].append(int(vvals[np.argmin(vpos)]) == gold)
        # 1-hop resolve then nearest value fact to the resolved key's sentence (link h-1 or value fact)
        prev_idx = head - 1  # link h-1 or the value fact if h==1
        ppos = int(st[prev_idx])
        acc["H_1hop_near"].append(int(vvals[np.argmin(np.abs(vpos - ppos))]) == gold)
        # lexical overlap: value fact whose key shares a token with the head key
        hk = set(fkey[head].tolist())
        ov = [j for j, f in enumerate(vf) if hk & set(fkey[f].tolist())]
        if len(ov) == 1:
            acc["H_lex_overlap_unique"].append(int(vvals[ov[0]]) == gold)
        acc["n_lex_overlap"].append(len(ov))
        # chain contiguity: are all needles of chain c within the span between the min/max of chain?
        idx = np.arange(base, base + h + 1)
        span_lo, span_hi = st[idx].min(), en[idx].max()
        others_inside = sum(1 for j in range(len(st)) if (j < base or j > head) and span_lo < st[j] < span_hi)
        acc["chain_interleaved"].append(others_inside > 0)
        # realised order within chain (step positions)
        steps = st[idx]
        acc["order_forward"].append(bool(np.all(np.diff(steps) > 0)))
        acc["order_reversed"].append(bool(np.all(np.diff(steps) < 0)))


def afreq_heuristics(it, acc, ts):
    ctx = np.asarray(it["context_ids"])
    n = len(ctx)
    X, Y = it["arrays"]["probe_x"], it["arrays"]["probe_y"]
    comma = int(ts.enc(",")[0])
    if ctx[0] != ts.eot_id:
        acc["fail_bos"] += 1
    if (ctx[1:] == ts.eot_id).any():
        acc["fail_eot"] += 1
    # background word occurrence check: split items by comma
    body = ctx[1:]
    cut = np.flatnonzero(body == comma)
    pieces = np.split(body, cut)
    words = [tuple(p[1:].tolist()) if (k > 0) else tuple(p.tolist()) for k, p in enumerate(pieces)]
    wc = Counter(words)
    probe_set = {(int(x),) for x in X} | {(int(y),) for y in Y}
    bg_max = max([c for w, c in wc.items() if w not in probe_set] or [0])
    acc["bg_max_occ"].append(bg_max)
    if bg_max > 3:
        acc["fail_bg_gt3"] += 1
    for qi, q in enumerate(it["queries"]):
        x, y = int(X[qi]), int(Y[qi])
        px, py = np.flatnonzero(ctx == x), np.flatnonzero(ctx == y)
        acc["counts_ok"].append(len(px) == q["meta"]["n_x"] and len(py) == q["meta"]["n_y"])
        acc["rec"].append(px.max() > py.max())
        acc["prim"].append(px.min() < py.min())
        for W in (256, 1024, 2048, 4096, 8192):
            cx, cy = int((px >= n - W).sum()), int((py >= n - W).sum())
            acc[f"win{W}"].append(1.0 if cx > cy else (0.0 if cx < cy else 0.5))
        # second-half counts
        cx, cy = int((px >= n // 2).sum()), int((py >= n // 2).sum())
        acc["half2"].append(1.0 if cx > cy else (0.0 if cx < cy else 0.5))
        # "recency-weighted" vote: exponential decay with half-life 1024 tokens
        for hl in (512, 2048, 8192):
            wx = np.exp(-(n - px) / hl).sum()
            wy = np.exp(-(n - py) / hl).sum()
            acc[f"decay{hl}"].append(1.0 if wx > wy else (0.0 if wx < wy else 0.5))
        acc["winner_first"].append(bool(q["meta"]["winner_first"]))
        # query mention order heuristics: answer first-mentioned word
        acc["answer_first_mentioned"].append(bool(q["meta"]["winner_first"]))
        if len(q["answer_ids"]) != 1 or int(q["answer_ids"][0]) != x:
            acc["fail_answer"] += 1
        if [list(c) for c in q["candidates"]] != [[x], [y]]:
            acc["fail_cands"] += 1
        if len(ctx) + len(q["prompt_ids"]) + 1 != (it["meta"]["T"] or len(ctx) + len(q["prompt_ids"]) + 1):
            acc["fail_T"] += 1


def acount_checks(it, acc, ts):
    ctx = np.asarray(it["context_ids"])
    a = it["arrays"]
    for qi, q in enumerate(it["queries"]):
        an = int(a["animals"][qi])
        c = int(a["counts"][qi])
        acc["acount_ok"].append(int((ctx == an).sum()) == c)
        acc["acount_gold_digit"].append(int(q["answer_ids"][0]) == ts.digit_ids[c])
        # recency/heuristics: predicting the most common count (5) etc.
        acc["acount_counts"].append(c)
    if (ctx[1:] == ts.eot_id).any():
        acc["fail_eot"] += 1


def summarize(acc, task):
    out = {}
    for k, v in acc.items():
        if isinstance(v, Counter):
            out[k] = dict(sorted(((str(a), b) for a, b in v.items())))
        elif isinstance(v, list):
            if not v:
                continue
            arr = np.asarray(v, dtype=float)
            if k == "depth":
                h, _ = np.histogram(arr, bins=np.linspace(0, 1, 11))
                out["depth_decile_counts"] = h.tolist()
                out["depth_mean"] = float(arr.mean())
            elif k == "distance":
                out["distance_min_med_max"] = [float(arr.min()), float(np.median(arr)), float(arr.max())]
            elif k == "snap_rate":
                out["snap_rate_mean"] = float(arr.mean())
            elif k.startswith("share_prefix") or k in ("bg_max_occ", "n_lex_overlap", "acount_counts"):
                out[k + "_mean"] = float(arr.mean())
                if k == "acount_counts":
                    out["acount_count_hist"] = dict(Counter(int(x) for x in arr))
            else:
                out[k] = round(float(arr.mean()), 4)
                out[k + "_n"] = int(len(arr))
        else:
            out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", nargs="+", default=["test", "dev"])
    ap.add_argument("--suites", nargs="+", default=None)
    ap.add_argument("--only", default=None)
    ap.add_argument("--max-items", type=int, default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    tok = load_tokenizer(REPO / "data" / "tokenizer")
    ts = token_space(tok)
    wlB = word_list(ts, "B")
    fo = open(args.out, "w")
    t0 = time.time()
    for ns in args.ns:
        root = REPO / "experiments" / "items" / ns
        man = json.loads((root / "MANIFEST.json").read_text())
        for rel in sorted(man["files"]):
            if not rel.endswith(".npz"):
                continue
            suite = rel.split("/")[0]
            if args.suites and suite not in args.suites:
                continue
            if args.only and args.only not in rel:
                continue
            items, cell = load_items(root / rel)
            task = cell["task"]
            if task.startswith("LM-"):
                continue
            if args.max_items:
                items = items[: args.max_items]
            acc: dict = defaultdict(list)
            for k in ("filler_kind", "filler_pool", "n_cand", "prompt_core_occ", "key_bigram_occ", "gold_digit_occ",
                      "mhop_head_key_occ", "mhop_full_prompt_occ"):
                acc[k] = Counter()
            fails: Counter = Counter()
            acc["absent_total"] = 0
            acc["absent_in_ctx"] = 0
            for k in ("fail_bos", "fail_eot", "fail_bg_gt3", "fail_answer", "fail_cands", "fail_T"):
                acc[k] = 0
            nq = 0
            for it in items:
                nq += len(it["queries"])
                if task in CODE_TASKS:
                    check_code_item(it, cell, ts, wlB, acc, fails)
                    if task == "M-hop":
                        mhop_shortcuts(it, acc)
                elif task == "A-freq":
                    afreq_heuristics(it, acc, ts)
                    fs = it["meta"].get("filler_source") or {}
                    acc["filler_kind"][fs.get("kind")] += 1
                elif task == "A-count":
                    acount_checks(it, acc, ts)
            rec = dict(file=f"{ns}/{rel}", task=task, T=cell["T"], level=cell["level"], variant=cell.get("variant"),
                       n_items=len(items), n_queries=nq, fails=dict(fails), **summarize(acc, task))
            rec = {k: v for k, v in rec.items() if not (isinstance(v, (int,)) and k.startswith("fail_") and v == 0)}
            fo.write(json.dumps(rec, sort_keys=True) + "\n")
            fo.flush()
            print(f"[{time.time() - t0:7.1f}s] {ns}/{rel} n={len(items)} fails={dict(fails)}", flush=True)
    fo.close()


if __name__ == "__main__":
    main()

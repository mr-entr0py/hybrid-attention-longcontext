"""VT-17: TEF generator (protocol §7.3, §16).

Pass thresholds: equal counts per bucket and family (+-1 per 1,000 steps); answer-token mask
correct; no eval tuples. Plus the trainer's requirements on ``TEFStream`` (INTERFACES.md):
pure function of (seed, step, micro), prefix-stable in ``steps``, D014 weights (block-normalised
by default since verification round 3; the per-step rule via ``answer_norm="step"``),
``synthetic_share`` / ``answer_only`` fallbacks.

Counts are checked on the exact protocol configuration (32K x 1 x 2, 1,000-step blocks,
7 buckets); sequences are materialised for a sample of steps (a whole block is the slow test).
Runs with ``data/tokenizer`` + ``data/tokens`` when built, else the dev stand-ins.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from hybridlc.data.storeio import synthetic_corpus
from hybridlc.data.tef import FAMILIES, IGNORE_INDEX, StoreReplay, TEFStream
from hybridlc.eval.constants import TEF_BUCKETS, TEF_FAMILY_MIX
from hybridlc.eval.devtools import resolve_test_tokenizer
from hybridlc.eval.filler import StoreFiller, dev_text_filler
from hybridlc.eval.generators import generate
from hybridlc.eval.rng import eval_rng
from hybridlc.eval.tokens import token_space
from hybridlc.eval.words import word_list

REPO = Path(__file__).resolve().parents[1]
REAL = (REPO / "data" / "tokens" / "fwe" / "train.bin").exists() and (REPO / "data" / "tokenizer" / "tokenizer.json").exists()


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    tok, kind = resolve_test_tokenizer()
    ts = token_space(tok)
    if REAL:
        return dict(tok=tok, ts=ts, filler=StoreFiller(ts, "train"), replay=StoreReplay())
    root = tmp_path_factory.mktemp("tokens")
    synthetic_corpus(root, n_fwe=1500, n_books=12)
    return dict(tok=tok, ts=ts, filler=dev_text_filler(ts, "train"), replay=StoreReplay(root))


def stream(env, seed=0, steps=1000, **kw):
    return TEFStream(seed, steps, tok=env["tok"], filler=env["filler"], replay=env["replay"], **kw)


# ----------------------------------------------------------------------------- counts
def test_equal_counts_per_bucket_and_family(env):
    s = stream(env)
    assert s.seq_len == 32768 and s.seqs_per_step == 2 and s.buckets == TEF_BUCKETS
    nb = s.items_per_bucket()
    # 90% of 1,000 x 65,536 tokens over sum(buckets) = 63.5K tokens per bucket set
    assert nb == int(0.9 * 1000 * 65536 // sum(TEF_BUCKETS)) == 907
    for block in (0, 1, 3):
        c = s.plan_counts(block)
        fam = c["family"]
        for ell in TEF_BUCKETS:
            per = [fam[(ell, f)] for f in FAMILIES]
            assert sum(per) == nb
            assert max(per) - min(per) <= 1, (ell, per)
        tot = [sum(fam[(ell, f)] for ell in TEF_BUCKETS) for f in FAMILIES]
        assert max(tot) - min(tot) <= 1, tot  # per family over the block, +-1
        # sub-type mix exact up to rounding within each (bucket, family) cell
        sub = c["subtype"]
        for ell in TEF_BUCKETS:
            for f, mix in TEF_FAMILY_MIX.items():
                n = fam[(ell, f)]
                for st, share in mix.items():
                    assert abs(sub[(ell, st)] - share * n) < 1.0, (ell, st, sub[(ell, st)], share * n)
        # 10% repetitive filler within retrieval and multi-hop items
        n_rm = sum(fam[(ell, f)] for ell in TEF_BUCKETS for f in ("retrieval", "multihop"))
        assert abs(c["repetitive"] - 0.1 * n_rm) <= len(TEF_BUCKETS) * 2


def test_plan_packing_and_shares(env):
    s = stream(env)
    plan = s.block_plan(0)
    assert len(plan) == 2000
    syn = rep = 0
    for seq in plan:
        assert sum(sg.length for sg in seq) == 32768
        assert sum(sg.kind == "replay" for sg in seq) <= 1
        syn += sum(sg.length for sg in seq if sg.kind == "item")
        rep += sum(sg.length for sg in seq if sg.kind == "replay")
    assert syn + rep == 2000 * 32768
    assert 0.899 <= syn / (syn + rep) <= 0.901
    kinds = [sg.replay_kind for seq in plan for sg in seq if sg.kind == "replay"]
    assert 0.5 < np.mean([k == "pg19" for k in kinds]) < 0.7  # Stage-B mix 60/40


# ----------------------------------------------------------------------------- batches
@pytest.fixture(scope="module")
def batches(env):
    s = stream(env)
    steps = [0, 1, 7, 500, 999]
    return s, {(st, m): s.get_numpy(st, m) for st in steps for m in range(s.accum)}


def test_batch_contract(batches, env):
    s, bs = batches
    eot = env["ts"].eot_id
    for (st, m), b in bs.items():
        x, lab, cu, pos, w = b["input_ids"][0], b["labels"][0], b["cu_seqlens"], b["position_ids"][0], b["loss_weights"][0]
        assert x.shape == (32768,) and lab.shape == x.shape and w.dtype == np.float32
        assert cu[0] == 0 and cu[-1] == 32768 and cu.dtype == np.int32 and (np.diff(cu) > 0).all()
        seglen = np.diff(cu)
        assert (pos == np.arange(32768) - np.repeat(cu[:-1], seglen)).all()
        # labels: next token within the sequence, ignored before non-EOT segments and at the end
        for t0 in cu[1:-1]:
            if x[t0] != eot:
                assert lab[t0 - 1] == IGNORE_INDEX
            else:
                assert lab[t0 - 1] == eot  # separators are predicted
        ok = lab != IGNORE_INDEX
        assert (lab[:-1][ok[:-1]] == x[1:][ok[:-1]]).all() and lab[-1] == IGNORE_INDEX
        # every item segment is exactly its bucket length and starts with EOT
        for sg in b["segments"][0]:
            if "subtype" in sg:
                assert sg["length"] in TEF_BUCKETS and x[sg["start"]] == eot
                assert int(sg["start"]) in set(cu.tolist())
        assert (w[~ok] == 0).all()


def test_answer_mask_correct(batches, env):
    """The flagged label positions are exactly the answer tokens of the items' queries."""
    s, bs = batches
    from hybridlc.eval.generators import flatten_item

    for (st, m), b in list(bs.items())[:6]:
        x, am = b["input_ids"][0], b["answer_mask"][0]
        expect = np.zeros(32768, bool)
        for sg in b["segments"][0]:
            if "subtype" not in sg:
                continue
            spec = next(seg.spec for seg in s.block_plan(sg["block"])[s._seq_index(st, m, 0)[1]]
                        if seg.kind == "item" and seg.spec.uid == sg["uid"])
            it = s.make_item(sg["block"], spec)
            toks, amask = flatten_item(it)
            a = int(sg["start"])
            assert (x[a:a + len(toks)] == toks).all()
            expect[np.flatnonzero(amask) + a - 1] = True  # label t predicts token t+1 (never t+1 = 0)
            # answers are the gold answers of the item's queries
            ans = np.concatenate([q["answer_ids"] for q in it["queries"]])
            assert (toks[amask] == ans).all()
            assert len(it["queries"]) == 4
        assert (am == expect).all()


def test_loss_weights_d014(batches, env):
    """Block normalisation (default): every answer label of the block has weight 0.5/(A_block/1000),
    every other valid label 0.5/(O_block/1000); batches carry loss_norm = 1."""
    s, bs = batches
    assert s.answer_norm == "block"
    A, O = s.block_counts(0)
    for st in {k[0] for k in bs}:
        w = [bs[(st, m)]["loss_weights"][0] for m in range(s.accum)]
        am = [bs[(st, m)]["answer_mask"][0] for m in range(s.accum)]
        na, no = s.step_counts(st)
        assert na == sum(int(a.sum()) for a in am)
        valid = [bs[(st, m)]["labels"][0] != IGNORE_INDEX for m in range(s.accum)]
        assert no == sum(int(v.sum()) for v in valid) - na
        for x, a, v in zip(w, am, valid):
            assert np.allclose(x[a], 0.5 * 1000 / A, rtol=1e-6)
            assert np.allclose(x[v & ~a], 0.5 * 1000 / O, rtol=1e-6)
        assert all(bs[(st, m)]["loss_norm"] == 1.0 for m in range(s.accum))
    # the per-step rule (answer_norm="step"): answers of the step share 1/2, the rest 1/2
    ss = stream(env, answer_norm="step")
    st = 7
    b = [ss.get_numpy(st, m) for m in range(ss.accum)]
    tot = sum(float(x["loss_weights"][0].sum(dtype=np.float64)) for x in b)
    ans = sum(float(x["loss_weights"][0][x["answer_mask"][0]].sum(dtype=np.float64)) for x in b)
    assert abs(tot - 1.0) < 1e-5 and abs(ans - 0.5) < 1e-5 and all("loss_norm" not in x for x in b)
    with pytest.raises(ValueError):
        stream(env, answer_norm="item")


def test_block_weights_length_balanced(env):
    """TEF-STEP-NORMALISED-ANSWER-WEIGHT-LENGTH-SKEW (verification round 3): with block normalisation
    each length bucket gets 1/7 of the block's answer-loss weight (equal answer-token counts, within
    the +-12-token count rounding); the per-step rule gives the 32K bucket ~0.29-0.30. Weights are
    the ones get_numpy uses (step_weight_scale), applied to the block plan."""
    import collections

    for norm, check in (("block", lambda sh: max(abs(v - 1 / 7) for v in sh.values()) < 0.002),
                        ("step", lambda sh: sh[32768] > 0.25 and sh[512] < 0.12)):
        s = stream(env, answer_norm=norm)
        plan = s.block_plan(0)
        w_len: collections.Counter = collections.Counter()
        tot_w = 0.0
        for st in range(s.block_steps):
            wa, wo = s.step_weight_scale(st)
            na, no = s.step_counts(st)
            tot_w += na * wa + no * wo
            for m in range(s.accum):
                for j in range(s.micro_batch):
                    _, g = s._seq_index(st, m, j)
                    for sg in plan[g]:
                        if sg.kind == "item":
                            w_len[sg.spec.length] += sg.spec.n_answer_tokens * wa
        tot = sum(w_len.values())
        share = {L: v / tot for L, v in w_len.items()}
        assert abs(tot - 0.5 * s.block_steps) < 1e-6 * s.block_steps  # answers: 1/2 of the objective
        assert abs(tot_w - s.block_steps) < 1e-6 * s.block_steps       # 1 per step on average
        assert check(share), (norm, share)


def test_pure_and_prefix_stable(batches, env):
    s, bs = batches
    s2 = stream(env, steps=4000)  # TEF-4000 continuation: same items on the shared prefix
    for k in [(0, 0), (999, 1)]:
        assert s2.get_numpy(*k)["batch_hash"] == bs[k]["batch_hash"]
        assert s.get_numpy(*k)["batch_hash"] == bs[k]["batch_hash"]  # pure: call again
    # different seed index -> different stream; block 1 differs from block 0
    assert stream(env, seed=1).get_numpy(0, 0)["batch_hash"] != bs[(0, 0)]["batch_hash"]
    assert s2.block_plan(1) != s2.block_plan(0)


def test_fallback_kwargs(env):
    s = stream(env, synthetic_share=0.5)
    assert s.items_per_bucket() == int(0.5 * 65536 * 1000 // sum(TEF_BUCKETS))
    a = stream(env, answer_only=True)
    w = [a.get_numpy(3, m) for m in range(2)]
    A, _ = a.block_counts(0)
    for b in w:
        assert (b["loss_weights"][0][~b["answer_mask"][0]] == 0).all()
        assert np.allclose(b["loss_weights"][0][b["answer_mask"][0]], 1000 / A, rtol=1e-6)
    a_step = stream(env, answer_only=True, answer_norm="step")
    wa, wo = a_step.step_weight_scale(3)
    assert wo == 0.0 and abs(wa * a_step.step_counts(3)[0] - 1.0) < 1e-9


# ----------------------------------------------------------------------------- leakage
def _tuples(items, ts):
    out, keys = set(), set()
    for it in items:
        arr = it.get("arrays", {})
        if "fact_key" not in arr:
            continue
        for k, v in zip(arr["fact_key"], arr["fact_value"]):
            if v < 0 or k[0] < 0:
                continue
            key = ts.decode(k).strip()
            keys.add(key)
            out.add(hashlib.sha256(f"{key}|{int(v):04d}|{it['meta']['template']}".encode()).hexdigest())
    return out, keys


def test_no_eval_tuples(batches, env):
    s, bs = batches
    ts, tok = env["ts"], env["tok"]
    tef_items = []
    for (st, m), b in bs.items():
        for sg in b["segments"][0]:
            if "subtype" in sg:
                spec = next(seg.spec for seg in s.block_plan(sg["block"])[s._seq_index(st, m, 0)[1]]
                            if seg.kind == "item" and seg.spec.uid == sg["uid"])
                tef_items.append(s.make_item(sg["block"], spec))
    for it in tef_items:
        assert it["meta"]["namespace"] == "train" and it["meta"]["template"] in ("T1", "T2", "T3")
    tr, trk = _tuples(tef_items, ts)
    filler = dev_text_filler(ts, "test")
    ev_items = []
    for task, lvl in (("R1", {"D": 15}), ("R-load", {"N": 512}), ("R3", {"U": 4}), ("M-hop", {"h": 4})):
        ev_items += generate(task, lvl, 8192, 4, eval_rng("test", "vt17", task), "test", tok, filler)
    ev, evk = _tuples(ev_items, ts)
    assert len(tr) > 100 and len(ev) > 1000
    assert tr.isdisjoint(ev) and trk.isdisjoint(evk)
    A = word_list(ts, "A")
    a_words = set(A.adjectives) | set(A.nouns)
    assert all(set(k.split()) <= a_words for k in trk)


def test_every_subtype_feasible_at_512(env):
    """Smallest bucket: every family/sub-type x template from the widest training ranges fits."""
    s = stream(env)
    from hybridlc.data.tef import ItemSpec

    uid = 0
    for fam, mix in TEF_FAMILY_MIX.items():
        for st in mix:
            for rep in (False, True):
                for _ in range(6):
                    it = s.make_item(9999, ItemSpec(uid, 512, fam, st, rep))
                    uid += 1
                    from hybridlc.eval.generators import flatten_item

                    toks, am = flatten_item(it)
                    assert len(toks) == 512 and am.sum() == 4 * (1 if fam == "aggregation" else 4)


@pytest.mark.slow
def test_full_block_materialises(env):
    """Every sequence of block 0 builds (all 2,000 x 32K), with exact block-normalised weights (the
    block's answer labels carry 1/2 x 1,000, all weights 1,000; verification round 3), and
    <|endoftext|> occurs only at cu_seqlens starts (verification round 1, DTS-01)."""
    s = stream(env)
    eot = env["ts"].eot_id
    n_eot = 0
    blk_tot = blk_ans = 0.0
    for st in range(1000):
        tot = 0.0
        for m in range(2):
            b = s.get_numpy(st, m)
            tot += float(b["loss_weights"][0].sum(dtype=np.float64))
            blk_ans += float(b["loss_weights"][0][b["answer_mask"][0]].sum(dtype=np.float64))
            x, cu = b["input_ids"][0], b["cu_seqlens"]
            e = np.flatnonzero(x == eot)
            n_eot += len(e)
            assert np.isin(e, cu[:-1]).all(), (st, m, e[~np.isin(e, cu[:-1])][:5])
            assert (b["position_ids"][0][e] == 0).all()
        na, no = s.step_counts(st)
        wa, wo = s.step_weight_scale(st)
        assert abs(tot - (na * wa + no * wo)) < 1e-4
        blk_tot += tot
    assert abs(blk_tot - 1000.0) < 1e-2 and abs(blk_ans - 500.0) < 1e-2, (blk_tot, blk_ans)
    print(f"block 0: {n_eot} separators, all at segment starts; weight mass {blk_tot:.4f}, answers {blk_ans:.4f}")

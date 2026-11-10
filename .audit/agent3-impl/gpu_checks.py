"""Agent-3 implementation audit: small GPU checks (run ONLY under scripts/repro/gpu_lock.py).

  python .audit/agent3-impl/gpu_checks.py <test> [--frac F]
tests: devattr | grads | trainvsinf | long32k | all-small

Writes JSON to stdout (one line per result, prefixed 'RESULT ').
TRITON_CACHE_DIR must point at a private copy of the diag cache (never .cache/triton).
"""
from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
assert "TRITON_CACHE_DIR" in os.environ and ".cache/triton" not in os.environ["TRITON_CACHE_DIR"].replace("triton-diag", ""), \
    "set TRITON_CACHE_DIR to a private cache copy"
os.environ.setdefault("FLA_FLASH_QLA", "0")


def emit(name, **kw):
    print("RESULT " + json.dumps({"test": name, **kw}, default=str), flush=True)


# --------------------------------------------------------------------------------------------- devattr
def devattr():
    cu = ctypes.CDLL("libcuda.so.1")
    assert cu.cuInit(0) == 0
    dev = ctypes.c_int()
    assert cu.cuDeviceGet(ctypes.byref(dev), 0) == 0
    out = {}
    for name, attr in (("KERNEL_EXEC_TIMEOUT", 17), ("COMPUTE_PREEMPTION_SUPPORTED", 90),
                       ("COMPUTE_MODE", 20), ("INTEGRATED", 18)):
        v = ctypes.c_int()
        rc = cu.cuDeviceGetAttribute(ctypes.byref(v), attr, dev)
        out[name] = v.value if rc == 0 else f"rc={rc}"
    emit("devattr", **out)


# --------------------------------------------------------------------------------------------- helpers
def setup(frac):
    import torch

    from hybridlc.utils.gpu import set_production_env
    set_production_env(overwrite=False)
    torch.cuda.set_per_process_memory_fraction(frac, 0)
    torch.backends.cuda.enable_cudnn_sdp(False)
    return torch.device("cuda:0")


def packed_batch(N, lens, device, seed=7, vocab=16384):
    import torch
    assert sum(lens) == N
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(8, vocab, (1, N), generator=g)
    starts = torch.tensor([0, *torch.tensor(lens).cumsum(0).tolist()[:-1]])
    ids[0, starts] = 0
    cu = torch.tensor([0, *torch.tensor(lens).cumsum(0).tolist()], dtype=torch.int32)
    pos = torch.cat([torch.arange(n) for n in lens])[None]
    labels = torch.cat([ids[0, 1:], torch.tensor([1])])[None]
    return ids.to(device), labels.to(device), cu.to(device), pos.to(device)


# --------------------------------------------------------------------------------------------- grads
def grads(frac):
    import torch

    from hybridlc.models.build import build_model
    from hybridlc.models.config import ModelConfig
    from hybridlc.models.loss import chunked_ce
    dev = setup(frac)
    lens = [1, 63, 200, 1, 517, 731, 535]   # length-1 docs, 64-chunk straddles
    N = sum(lens)
    ids, labels, cu, pos = packed_batch(N, lens, dev)
    names = [("C0", {}), ("C3", {}), ("C12", {}), ("X-gc12", {}), ("X-E3", {}), ("O-nc3", {}), ("O-gla3", {}),
             ("C12", {"ssmax": True}), ("X-nope3", {})]
    for name, ov in names:
        cfg = ModelConfig.for_scale("D384", name, **ov)
        res = {}
        gsets = {}
        for ac in (False, True):
            model = build_model(cfg, 0).to(dev)
            model.train()
            model.set_activation_checkpointing(ac)
            with torch.autocast("cuda", torch.bfloat16):
                h = model(ids, cu, pos)
                loss, z, n = chunked_ce(h, model.lm_head_weight, labels)
            (loss + z).backward()
            missing = [k for k, p in model.named_parameters() if p.grad is None]
            nonfinite = [k for k, p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
            zero = [k for k, p in model.named_parameters() if p.grad is not None and float(p.grad.abs().max()) == 0.0]
            gsets[ac] = {k: p.grad.detach().float().clone() for k, p in model.named_parameters() if p.grad is not None}
            res[f"ac{int(ac)}"] = dict(loss=float(loss), z=float(z), n=int(n), n_params=sum(1 for _ in model.parameters()),
                                       missing=missing, nonfinite=nonfinite, zero=zero)
            del model, h, loss, z
            torch.cuda.empty_cache()
        # activation-checkpointing equivalence
        worst = (0.0, None)
        for k in gsets[False]:
            a, b = gsets[False][k], gsets[True][k]
            rel = float((a - b).norm() / a.norm().clamp_min(1e-30))
            if rel > worst[0]:
                worst = (rel, k)
        res["ac_equiv_worst_rel_grad_diff"] = worst
        res["ac_equiv_loss_equal"] = res["ac0"]["loss"] == res["ac1"]["loss"]
        emit("grads", config=name, overrides=ov, **res)
        del gsets
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------------------------- train vs inference
def _load_trained(dev):
    import torch

    from hybridlc.models.build import build_model
    from hybridlc.models.config import ModelConfig
    p = REPO / "experiments/s0/vt13_production/a/ckpt/S0-D384-C3-s901-8588adcc/final.pt"
    st = torch.load(p, map_location="cpu", weights_only=False)
    m = build_model(ModelConfig.from_dict(st["config"]["model"]), int(st["config"]["seed_index"]))
    m.load_state_dict(st["model"], strict=True)
    return m.to(dev).eval(), str(p)


def trainvsinf(frac):
    import numpy as np
    import torch
    import torch.nn.functional as F

    from hybridlc.data.loader import StageStream
    from hybridlc.models.build import build_model
    from hybridlc.models.config import ModelConfig
    dev = setup(frac)
    # real packed text: LM2 held-out stream, micro-batch 0 (2 x 4K with document boundaries)
    b = StageStream("LM2", 0).get(0, 0)
    ids, cu, pos, labels = (b[k].to(dev) for k in ("input_ids", "cu_seqlens", "position_ids", "labels"))
    bounds = b["cu_seqlens"].tolist()
    # choose up to 4 complete documents (start with EOT) of length >= 300
    docs = [(a, e) for a, e in zip(bounds[:-1], bounds[1:]) if e - a >= 300 and int(b["input_ids"][0, a]) == 0][:4]
    models = []
    tm, tp = _load_trained(dev)
    models.append(("C3-trained640", tm))
    for name in ("C0", "C12", "X-gc12"):
        models.append((name + "-init", build_model(ModelConfig.for_scale("D384", name), 0).to(dev).eval()))
    for mname, model in models:
        for prec in ("bf16", "fp32"):
            ac = torch.autocast("cuda", torch.bfloat16) if prec == "bf16" else torch.autocast("cuda", enabled=False)
            rows = []
            with torch.no_grad(), ac:
                h = model(ids, cu, pos)
                for a, e in docs:
                    n = e - a
                    P = n // 2
                    tl = model.logits(h[0, a:e]).float()              # training-path logits [n, V]
                    d = ids[:, a:e]
                    last, cache = model.prefill(d[:, :P])
                    ext, _ = model.extend(d[:, P:n - 1], cache)       # predictions for positions P..n-1
                    il = torch.cat([last[:, None].float(), ext.float()], 1)[0]   # [n-P, V] for targets P..n-1
                    # one decode chain for 8 tokens from a fresh prefill
                    last2, c2 = model.prefill(d[:, :P])
                    dl = [last2.float()]
                    for t in range(P, P + 7):
                        lg, c2 = model.decode_step(d[:, t:t + 1], c2)
                        dl.append(lg.float())
                    dl = torch.cat(dl, 0)                               # [8, V] targets P..P+7
                    ref = tl[P - 1:n - 1]                                # training logits predicting P..n-1
                    tgt = d[0, P:n]
                    lp_t = F.log_softmax(ref, -1).gather(-1, tgt[:, None])[:, 0]
                    lp_i = F.log_softmax(il, -1).gather(-1, tgt[:, None])[:, 0]
                    lp_d = F.log_softmax(dl, -1).gather(-1, tgt[:8, None])[:, 0]
                    rows.append(dict(
                        doc_len=n, P=P,
                        max_abs_dlogit_extend=float((il - ref).abs().max()),
                        mean_abs_dlogprob_extend=float((lp_i - lp_t).abs().mean()),
                        max_abs_dlogprob_extend=float((lp_i - lp_t).abs().max()),
                        argmax_agree_extend=float((il.argmax(-1) == ref.argmax(-1)).float().mean()),
                        max_abs_dlogit_decode=float((dl - ref[:8]).abs().max()),
                        max_abs_dlogprob_decode=float((lp_d - lp_t[:8]).abs().max()),
                        nll_train=float(-lp_t.mean()), nll_inf=float(-lp_i.mean()),
                        logit_scale=float(ref.abs().max())))
            emit("trainvsinf", model=mname, precision=prec, docs=rows)
        del model
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------------------------- 32K
def long32k(frac, names=("C0", "C3", "C12"), T=32768, scale="D384"):
    import torch

    from hybridlc.data.loader import StageStream
    from hybridlc.models.build import build_model
    from hybridlc.models.config import ModelConfig
    from hybridlc.models.loss import chunked_ce
    dev = setup(frac)
    s = StageStream("B", 0, seq_len=T)
    # micro-batches 0..3 of step 0; pick the one with the longest segment (PG19 window = one 32K segment)
    cands = [s.get(0, m) for m in range(4)]
    b = max(cands, key=lambda x: x["max_seqlen"])
    ids, cu, pos, labels = (b[k].to(dev) for k in ("input_ids", "cu_seqlens", "position_ids", "labels"))
    for name in names:
        if name == "C3-trained640":
            model, _ = _load_trained(dev)
        else:
            model = build_model(ModelConfig.for_scale(scale, name), 0).to(dev)
        model.train()
        model.set_activation_checkpointing(True)
        for m_ in model.modules():
            if hasattr(m_, "collect_attn_stats"):
                m_.collect_attn_stats = True
        torch.cuda.reset_peak_memory_stats(dev)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            with torch.autocast("cuda", torch.bfloat16):
                h = model(ids, cu, pos, max_seqlen=b["max_seqlen"])
                loss, z, n = chunked_ce(h, model.lm_head_weight, labels)
            (loss + z).backward()
            torch.cuda.synchronize()
            gn = torch.nn.utils.clip_grad_norm_([p for p in model.parameters()], 1e9)
            fin = all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            miss = [k for k, p in model.named_parameters() if p.grad is None]
            st_ = {n_: m_.attn_stats for n_, m_ in model.named_modules() if getattr(m_, "attn_stats", None)}
            emit("long32k", config=name, scale=scale, attn_stats=st_, T=T, max_seqlen=b["max_seqlen"], sources=b["sources"], loss=float(loss),
                 z=float(z), grad_norm=float(gn), grads_finite=bool(fin), hidden_finite=bool(torch.isfinite(h).all()),
                 missing=miss, peak_alloc_GiB=torch.cuda.max_memory_allocated(dev) / 2**30,
                 peak_reserved_GiB=torch.cuda.max_memory_reserved(dev) / 2**30, secs=time.perf_counter() - t0)
        except torch.cuda.OutOfMemoryError as e:
            st_ = {n_: m_.attn_stats for n_, m_ in model.named_modules() if getattr(m_, "attn_stats", None)}
            emit("long32k", config=name, scale=scale, T=T, oom=str(e)[:300], peak_alloc_GiB=torch.cuda.max_memory_allocated(dev) / 2**30)
        del model
        torch.cuda.empty_cache()




def grads_det(frac, names=("C3", "C12")):
    """Nondeterminism floor: the same AC-off backward twice (production mode), vs AC-on."""
    import torch

    from hybridlc.models.build import build_model
    from hybridlc.models.config import ModelConfig
    from hybridlc.models.loss import chunked_ce
    dev = setup(frac)
    lens = [1, 63, 200, 1, 517, 731, 535]
    ids, labels, cu, pos = packed_batch(sum(lens), lens, dev)
    for name in names:
        cfg = ModelConfig.for_scale("D384", name)
        gs = []
        for ac in (False, False, True, True):
            model = build_model(cfg, 0).to(dev)
            model.set_activation_checkpointing(ac)
            with torch.autocast("cuda", torch.bfloat16):
                h = model(ids, cu, pos)
                loss, z, n = chunked_ce(h, model.lm_head_weight, labels)
            (loss + z).backward()
            gs.append({k: p.grad.detach().float().clone() for k, p in model.named_parameters()})
            del model, h
            torch.cuda.empty_cache()

        def worst(a, b):
            w = (0.0, None)
            for k in a:
                r = float((a[k] - b[k]).norm() / a[k].norm().clamp_min(1e-30))
                if r > w[0]:
                    w = (r, k)
            return w
        emit("grads_det", config=name, off_vs_off=worst(gs[0], gs[1]), on_vs_on=worst(gs[2], gs[3]),
             off_vs_on=worst(gs[0], gs[2]))


def longtvi(frac, T=32768, tail=128):
    """Long-context train-vs-inference (training varlen path vs prefill+extend) and fp32-vs-bf16-rounded
    weights, on one PG19 32K window (StageStream B, seed 0), trained C3 + init C0/C12."""
    import torch
    import torch.nn.functional as F

    from hybridlc.data.loader import StageStream
    from hybridlc.models.build import build_model
    from hybridlc.models.config import ModelConfig
    from hybridlc.models.loss import chunked_ce
    dev = setup(frac)
    s = StageStream("B", 0, seq_len=T)
    b = max((s.get(0, m) for m in range(4)), key=lambda x: x["max_seqlen"])
    assert b["max_seqlen"] == T
    ids, cu, pos = (b[k].to(dev) for k in ("input_ids", "cu_seqlens", "position_ids"))
    tgt = ids[0, T - tail:]
    tm, _ = _load_trained(dev)
    for mname, model in (("C3-trained640", tm), ("C0-init", build_model(ModelConfig.for_scale("D384", "C0"), 0).to(dev).eval()),
                         ("C12-init", build_model(ModelConfig.for_scale("D384", "C12"), 0).to(dev).eval())):
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            h = model(ids, cu, pos, max_seqlen=T)
            tl = model.logits(h[0, T - tail - 1:T - 1]).float()          # predict last `tail` tokens
            nll_full = float(chunked_ce(h[0, :T - 1], model.lm_head_weight, ids[0, 1:])[0])
            del h
            last, cache = model.prefill(ids[:, :T - tail])
            ext, _ = model.extend(ids[:, T - tail:T - 1], cache)
            il = torch.cat([last[:, None].float(), ext.float()], 1)[0]
            del cache, ext
            torch.cuda.empty_cache()
            lp_t = F.log_softmax(tl, -1).gather(-1, tgt[:, None])[:, 0]
            lp_i = F.log_softmax(il, -1).gather(-1, tgt[:, None])[:, 0]
            # bf16-rounded weights (integration.evaluate canonical_bf16) vs fp32 master weights
            sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
            model.load_state_dict({k: v.to(torch.bfloat16).float() for k, v in sd.items()})
            h2 = model(ids, cu, pos, max_seqlen=T)
            tl2 = model.logits(h2[0, T - tail - 1:T - 1]).float()
            nll_full2 = float(chunked_ce(h2[0, :T - 1], model.lm_head_weight, ids[0, 1:])[0])
            del h2
            # attribution: round only GDN A_log/dt_bias
            model.load_state_dict(sd)
            model.load_state_dict({k: (v.to(torch.bfloat16).float() if (k.endswith("A_log") or k.endswith("dt_bias")) else v)
                                   for k, v in sd.items()})
            h3 = model(ids, cu, pos, max_seqlen=T)
            nll_full3 = float(chunked_ce(h3[0, :T - 1], model.lm_head_weight, ids[0, 1:])[0])
            tl3 = model.logits(h3[0, T - tail - 1:T - 1]).float()
            del h3
            model.load_state_dict(sd)
            lp_2 = F.log_softmax(tl2, -1).gather(-1, tgt[:, None])[:, 0]
            lp_3 = F.log_softmax(tl3, -1).gather(-1, tgt[:, None])[:, 0]
        emit("longtvi", model=mname, T=T, tail=tail,
             tvi_max_abs_dlogit=float((il - tl).abs().max()), tvi_mean_abs_dlogprob=float((lp_i - lp_t).abs().mean()),
             tvi_max_abs_dlogprob=float((lp_i - lp_t).abs().max()), tvi_argmax_agree=float((il.argmax(-1) == tl.argmax(-1)).float().mean()),
             nll_tail_train=float(-lp_t.mean()), nll_tail_inf=float(-lp_i.mean()),
             nll_full_fp32w=nll_full, nll_full_bf16w=nll_full2, nll_full_bf16_gates_only=nll_full3,
             bf16w_max_abs_dlogit_tail=float((tl2 - tl).abs().max()), bf16w_mean_abs_dlogprob_tail=float((lp_2 - lp_t).abs().mean()),
             bf16w_argmax_agree_tail=float((tl2.argmax(-1) == tl.argmax(-1)).float().mean()),
             gates_only_mean_abs_dlogprob_tail=float((lp_3 - lp_t).abs().mean()))
        del model
        torch.cuda.empty_cache()


def scorepath(frac, cells=("experiments/items/dev/DC/R-load_N32_4K.npz", "experiments/items/dev/DC/R1_D3_4K.npz",
                           "experiments/items/dev/DC/A-freq_rho2_4K.npz"), n_ctx=24):
    """eval/scorer.py consistency: gold logp from (cache, prompt+answer) [path 1, used for cand_lp[:,0]] vs gold
    scored like the candidates from the post-prompt cache p_cache [path 2]; rank flips if gold used path 2."""
    import numpy as np
    import torch

    from hybridlc.eval.items import load_items
    from hybridlc.eval.scorer import score_items
    dev = setup(frac)
    model, _ = _load_trained(dev)
    for cell in cells:
        items, meta = load_items(REPO / cell)
        items = items[:n_ctx]
        rows = score_items(model, items, batch_size=4, cell_name=Path(cell).stem)
        by_id = {(r["context_id"], r["query_idx"]): r for r in rows}
        diffs, flips, n = [], 0, 0
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            k = 0
            for it in items:
                ctx = torch.from_numpy(np.asarray(it["context_ids"], np.int64))[None].to(dev)
                _, cache = model.prefill(ctx)
                for q in it["queries"]:
                    r = by_id[(it["meta"]["context_id"], q["meta"]["query_idx"])]
                    k += 1
                    pr = torch.from_numpy(np.asarray(q["prompt_ids"], np.int64))[None].to(dev)
                    an = torch.from_numpy(np.asarray(q["answer_ids"], np.int64))[None].to(dev)
                    _, pc = model.extend(pr, cache)
                    lp2, _ = model.score_continuation(pc, an)
                    g2 = float(lp2.float().sum())
                    g1 = r["logp_gold"]
                    diffs.append(g2 - g1)
                    others = r["cand_logp"][1:] if r["cand_logp"] else []
                    rank1 = 1 + sum(1 for v in others if v >= g1)
                    rank2 = 1 + sum(1 for v in others if v >= g2)
                    flips += int((rank1 == 1) != (rank2 == 1))
                    n += 1
                del cache
        d = np.abs(np.asarray(diffs))
        emit("scorepath", cell=cell, n_queries=n, mean_abs_dgold=float(d.mean()), max_abs_dgold=float(d.max()),
             mean_signed_dgold=float(np.mean(diffs)), rank1_flips=flips,
             rank_acc=float(np.mean([r["rank_correct"] for r in rows if r["rank_correct"] is not None])),
             em=float(np.mean([r["EM_tf"] for r in rows])))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("test")
    ap.add_argument("--frac", type=float, default=0.17)
    ap.add_argument("--names", default="C0,C3,C12")
    ap.add_argument("--T", type=int, default=32768)
    ap.add_argument("--scale", default="D384")
    a = ap.parse_args()
    if a.test == "devattr":
        devattr()
    elif a.test == "grads":
        grads(a.frac)
    elif a.test == "trainvsinf":
        trainvsinf(a.frac)
    elif a.test == "grads_det":
        grads_det(a.frac)
    elif a.test == "longtvi":
        longtvi(a.frac, a.T)
    elif a.test == "scorepath":
        scorepath(a.frac)
    elif a.test == "long32k":
        long32k(a.frac, tuple(a.names.split(",")), a.T, a.scale)
    else:
        raise SystemExit(f"unknown test {a.test}")


"""Agent-3 implementation audit, round 2 (resumed session): small GPU checks on HEAD 7f8bed3.
Run ONLY under scripts/repro/gpu_lock.py with a private TRITON_CACHE_DIR and memory fraction <= 0.35.

  python .audit/agent3-impl/gpu_checks2.py <test> [--frac F]
tests: devattr | grads2 | isoshort | initscale | bf16attr | long32k
Prints one 'RESULT {json}' line per result.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
_tc = os.environ.get("TRITON_CACHE_DIR", "")
assert _tc and not _tc.rstrip("/").endswith((".cache/triton", ".cache/triton-diag")), "use a private Triton cache copy"
os.environ.setdefault("FLA_FLASH_QLA", "0")


def emit(name, **kw):
    print("RESULT " + json.dumps({"test": name, **kw}, default=str), flush=True)


def setup(frac):
    import torch

    from hybridlc.utils.gpu import set_production_env
    set_production_env(overwrite=False)
    torch.cuda.set_per_process_memory_fraction(frac, 0)
    torch.backends.cuda.enable_cudnn_sdp(False)
    return torch.device("cuda:0")


# ------------------------------------------------------------------------------------------- devattr
def devattr(_frac):
    cu = ctypes.CDLL("libcuda.so.1")
    assert cu.cuInit(0) == 0
    dev = ctypes.c_int()
    assert cu.cuDeviceGet(ctypes.byref(dev), 0) == 0
    out = {}
    for name, attr in (("KERNEL_EXEC_TIMEOUT", 17), ("COMPUTE_MODE", 20), ("COMPUTE_PREEMPTION_SUPPORTED", 90)):
        v = ctypes.c_int()
        rc = cu.cuDeviceGetAttribute(ctypes.byref(v), attr, dev)
        out[name] = v.value if rc == 0 else f"rc={rc}"
    ver = ctypes.c_int()
    cu.cuDriverGetVersion(ctypes.byref(ver))
    out["cuda_driver_api_version"] = ver.value
    emit("devattr", **out)


# ------------------------------------------------------------------------------------------- helpers
def packed(lens, dev, seed=5, vocab=16384):
    import torch
    g = torch.Generator().manual_seed(seed)
    N = sum(lens)
    starts = torch.tensor([0, *torch.tensor(lens).cumsum(0).tolist()[:-1]])
    ids = torch.randint(8, vocab, (1, N), generator=g)
    ids[0, starts] = 0
    cu = torch.tensor([0, *torch.tensor(lens).cumsum(0).tolist()], dtype=torch.int32)
    pos = torch.cat([torch.arange(n) for n in lens])[None]
    labels = torch.cat([ids[0, 1:], torch.tensor([1])])[None]
    return ids.to(dev), labels.to(dev), cu.to(dev), pos.to(dev), starts


# ------------------------------------------------------------------------------------------- grads2
def grads2(frac):
    """Every parameter receives a finite, non-zero gradient (one bf16 fwd/bwd on a packed row with
    length-1/2/3 documents and 64-chunk straddles) for every config family incl. ones not in round 1."""
    import torch

    from hybridlc.models.build import build_model
    from hybridlc.models.config import ModelConfig
    from hybridlc.models.loss import chunked_ce
    dev = setup(frac)
    lens = [1, 2, 3, 63, 200, 1, 517, 731, 530]
    ids, labels, cu, pos, _ = packed(lens, dev)
    names = [("C0", {}), ("C1", {}), ("C2", {}), ("C3", {}), ("C6", {}), ("C12", {}), ("P1-late", {}),
             ("P3-early", {}), ("X-gc12", {}), ("X-gc3", {}), ("X-E0", {}), ("O-gla0", {}), ("O-nc0", {}),
             ("X-nope12", {}), ("C3", {"ssmax": True}), ("C12", {"ssmax": True})]
    for name, ov in names:
        cfg = ModelConfig.for_scale("D384", name, **ov)
        model = build_model(cfg, 0).to(dev)
        model.train()
        with torch.autocast("cuda", torch.bfloat16):
            h = model(ids, cu, pos)
            loss, z, n = chunked_ce(h, model.lm_head_weight, labels)
        (loss + z).backward()
        missing = [k for k, p in model.named_parameters() if p.grad is None]
        nonfinite = [k for k, p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
        zero = [k for k, p in model.named_parameters() if p.grad is not None and float(p.grad.abs().max()) == 0.0]
        # gradient norms of the layer-0 / first full layer projections (init-scale diagnostics)
        gn = {}
        for l, blk in enumerate(model.layers):
            if blk.kind in "AC":
                for pn in ("q_proj", "k_proj", "v_proj", "o_proj"):
                    gn[f"L{l}.{pn}"] = float(getattr(blk.mixer, pn).weight.grad.norm())
                if blk.kind == "C":
                    gn[f"L{l}.conv"] = float(blk.mixer.conv.weight.grad.norm())
                break
        emit("grads2", config=name, overrides=ov, pattern=cfg.pattern, loss=float(loss), n_params=sum(1 for _ in model.parameters()),
             missing=missing, nonfinite=nonfinite, zero=zero, first_full_layer_grad_norms=gn)
        del model, h, loss, z
        torch.cuda.empty_cache()


# ------------------------------------------------------------------------------------------- isoshort
def isoshort(frac):
    """Document isolation with documents SHORTER than the conv width (1, 2, 3 tokens) and at chunk
    boundaries: (a) packed vs each document alone (rel RMS, max abs); (b) bit-identity of a document's
    hidden states when every other document is replaced (identical shapes)."""
    import torch

    from hybridlc.models.build import build_model
    from hybridlc.models.config import ModelConfig
    dev = setup(frac)
    lens = [1, 2, 3, 4, 5, 63, 64, 65, 1, 2, 128, 3]
    ids, labels, cu, pos, starts = packed(lens, dev, seed=9)
    b = cu.tolist()
    g = torch.Generator().manual_seed(17)
    for name in ("C0", "C3", "X-gc3", "X-E3", "O-gla3", "O-nc3"):
        model = build_model(ModelConfig.for_scale("D384", name), 0).to(dev).eval()
        for prec in ("bf16", "fp32"):
            ctx = torch.autocast("cuda", torch.bfloat16) if prec == "bf16" else torch.autocast("cuda", enabled=False)
            with torch.no_grad(), ctx:
                h = model(ids, cu, pos).float()
                parts = []
                for a, e in zip(b[:-1], b[1:]):
                    n = e - a
                    parts.append(model(ids[:, a:e], torch.tensor([0, n], dtype=torch.int32, device=dev),
                                       torch.arange(n, device=dev)[None]).float())
                h1 = torch.cat(parts, 1)
                rel = float((h - h1).pow(2).mean().sqrt() / h1.pow(2).mean().sqrt())
                mx = float((h - h1).abs().max())
                # per-document worst rel error for the short documents
                short = {}
                for j, (a, e) in enumerate(zip(b[:-1], b[1:])):
                    if e - a <= 5:
                        d = (h[:, a:e] - h1[:, a:e]).pow(2).mean().sqrt() / h1[:, a:e].pow(2).mean().sqrt()
                        short[f"doc{j}_len{e - a}"] = round(float(d), 7)
                bit = []
                for j in range(len(lens)):
                    ids2 = torch.randint(8, 16384, ids.shape, generator=g).to(dev)
                    ids2[0, starts.to(dev)] = 0
                    ids2[:, b[j]:b[j + 1]] = ids[:, b[j]:b[j + 1]]
                    h2 = model(ids2, cu, pos).float()
                    bit.append(bool(torch.equal(h2[:, b[j]:b[j + 1]], h[:, b[j]:b[j + 1]])))
            emit("isoshort", config=name, precision=prec, rel_rms_packed_vs_alone=rel, max_abs=mx,
                 short_doc_rel=short, bit_identical_when_others_replaced=all(bit), bit_per_doc=bit)
        del model
        torch.cuda.empty_cache()


# ------------------------------------------------------------------------------------------- initscale
def initscale(frac):
    """Per-layer mixer-output and MLP-output RMS at init (bf16 autocast, real LM2 text, 4K row)."""
    import torch

    from hybridlc.data.loader import StageStream
    from hybridlc.models.build import build_model
    from hybridlc.models.config import ModelConfig
    dev = setup(frac)
    bt = StageStream("LM2", 0).get(0, 0)
    ids, cu, pos = (bt[k].to(dev) for k in ("input_ids", "cu_seqlens", "position_ids"))
    for name in ("C0", "C3", "C12", "X-gc12", "X-gc3", "X-E3", "O-gla3", "O-nc3"):
        model = build_model(ModelConfig.for_scale("D384", name), 0).to(dev).eval()
        rec = {}
        hooks = []
        for l, blk in enumerate(model.layers):
            hooks.append(blk.mixer.register_forward_hook(
                lambda m, i, o, l=l, k=blk.kind: rec.setdefault(l, {}).update(kind=k, mixer=float(o.float().pow(2).mean().sqrt()))))
            hooks.append(blk.mlp.register_forward_hook(
                lambda m, i, o, l=l: rec.setdefault(l, {}).update(mlp=float(o.float().pow(2).mean().sqrt()))))
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            model(ids, cu, pos)
        for hk in hooks:
            hk.remove()
        by_kind = {}
        for l, r in rec.items():
            by_kind.setdefault(r["kind"], []).append(r["mixer"])
        emit("initscale", config=name, mean_mixer_rms_by_kind={k: sum(v) / len(v) for k, v in by_kind.items()},
             mean_mlp_rms=sum(r["mlp"] for r in rec.values()) / len(rec),
             per_layer={l: {"kind": r["kind"], "mixer": round(r["mixer"], 7), "mlp": round(r["mlp"], 6)} for l, r in sorted(rec.items())})
        del model
        torch.cuda.empty_cache()


# ------------------------------------------------------------------------------------------- bf16attr
def bf16attr(frac, T=32768, tail=512):
    """Attribution of the canonical bf16-rounded-weights perturbation (RR-D052i) on the trained C3 S0
    checkpoint at 32K: (a) all floating tensors rounded, (b) only GDN A_log/dt_bias rounded,
    (c) everything rounded EXCEPT A_log/dt_bias. Tail-token mean |d logprob| and full-window NLL."""
    import torch
    import torch.nn.functional as F

    from hybridlc.data.loader import StageStream
    from hybridlc.models.build import build_model
    from hybridlc.models.config import ModelConfig
    from hybridlc.models.loss import chunked_ce
    dev = setup(frac)
    p = REPO / "experiments/s0/vt13_production/a/ckpt/S0-D384-C3-s901-8588adcc/final.pt"
    st = torch.load(p, map_location="cpu", weights_only=False)
    model = build_model(ModelConfig.from_dict(st["config"]["model"]), int(st["config"]["seed_index"]))
    model.load_state_dict(st["model"], strict=True)
    model = model.to(dev).eval()
    sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
    s = StageStream("B", 0, seq_len=T)
    b = max((s.get(0, m) for m in range(4)), key=lambda x: x["max_seqlen"])
    ids, cu, pos = (b[k].to(dev) for k in ("input_ids", "cu_seqlens", "position_ids"))
    tgt = ids[0, T - tail:]
    gates = lambda k: k.endswith("A_log") or k.endswith("dt_bias")  # noqa: E731
    variants = {"fp32": lambda k, v: v,
                "all_bf16": lambda k, v: v.to(torch.bfloat16).float() if v.is_floating_point() else v,
                "gates_only_bf16": lambda k, v: v.to(torch.bfloat16).float() if gates(k) else v,
                "all_but_gates_bf16": lambda k, v: v if gates(k) else (v.to(torch.bfloat16).float() if v.is_floating_point() else v)}
    res = {}
    lp = {}
    for vn, fn in variants.items():
        model.load_state_dict({k: fn(k, v) for k, v in sd.items()})
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            h = model(ids, cu, pos, max_seqlen=T)
            nll = float(chunked_ce(h[0, :T - 1], model.lm_head_weight, ids[0, 1:])[0])
            tl = model.logits(h[0, T - tail - 1:T - 1]).float()
            del h
        lp[vn] = F.log_softmax(tl, -1).gather(-1, tgt[:, None])[:, 0]
        res[vn] = {"nll_full": nll, "argmax": tl.argmax(-1)}
    out = {}
    for vn in variants:
        if vn == "fp32":
            continue
        out[vn] = {"nll_full_delta": res[vn]["nll_full"] - res["fp32"]["nll_full"],
                   "tail_mean_abs_dlogprob": float((lp[vn] - lp["fp32"]).abs().mean()),
                   "tail_max_abs_dlogprob": float((lp[vn] - lp["fp32"]).abs().max()),
                   "tail_argmax_agree": float((res[vn]["argmax"] == res["fp32"]["argmax"]).float().mean())}
    emit("bf16attr", ckpt=str(p), T=T, tail=tail, nll_full_fp32=res["fp32"]["nll_full"], **out)


# ------------------------------------------------------------------------------------------- long32k
def long32k(frac, names=("C0", "C3", "C12"), T=32768):
    import torch

    from hybridlc.data.loader import StageStream
    from hybridlc.models.build import build_model
    from hybridlc.models.config import ModelConfig
    from hybridlc.models.loss import chunked_ce
    dev = setup(frac)
    s = StageStream("B", 0, seq_len=T)
    b = max((s.get(0, m) for m in range(4)), key=lambda x: x["max_seqlen"])
    ids, cu, pos, labels = (b[k].to(dev) for k in ("input_ids", "cu_seqlens", "position_ids", "labels"))
    for name in names:
        model = build_model(ModelConfig.for_scale("D384", name), 0).to(dev)
        model.train()
        model.set_activation_checkpointing(True)
        torch.cuda.reset_peak_memory_stats(dev)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.autocast("cuda", torch.bfloat16):
            h = model(ids, cu, pos, max_seqlen=b["max_seqlen"])
            loss, z, n = chunked_ce(h, model.lm_head_weight, labels)
        (loss + z).backward()
        torch.cuda.synchronize()
        gn = torch.nn.utils.clip_grad_norm_([p for p in model.parameters()], 1e9)
        fin = all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None)
        miss = [k for k, p in model.named_parameters() if p.grad is None]
        emit("long32k", config=name, T=T, max_seqlen=b["max_seqlen"], n_segments=int(cu.numel()) - 1, loss=float(loss),
             grad_norm=float(gn), grads_finite=fin, hidden_finite=bool(torch.isfinite(h).all()), missing=miss,
             peak_alloc_GiB=torch.cuda.max_memory_allocated(dev) / 2**30, secs=time.perf_counter() - t0)
        del model, h
        torch.cuda.empty_cache()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("test")
    ap.add_argument("--frac", type=float, default=0.30)
    a = ap.parse_args()
    {"devattr": devattr, "grads2": grads2, "isoshort": isoshort, "initscale": initscale, "bf16attr": bf16attr,
     "long32k": long32k}[a.test](a.frac)

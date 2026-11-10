"""R-ATT: the explicitly selected fp32 reference attention (FEAS-PREC v2 implementation requirement;
PI rulings D083, §1.3 of reports/pi_rulings_2026-11-10.md).

* relative RMS <= 1e-5 against the SDPA MATH reference at T in {512, 4096} for the full-sequence forward
  (with cu_seqlens document isolation), inference prefill, lower-right extend and decode, for F_g and F_gc
  (CPU at T = 512; CUDA, gpu-marked, at both lengths);
* peak memory <= 7 GiB for D384 C12 single pass (forward + full fp32 logits) and cache-path prefill +
  extend at T = 32,768 (gpu + slow);
* the default (production) paths are unchanged; the reference mode refuses half precision and its pinned
  EFFICIENT backend raises if it cannot run (never a silent fallback).
"""
from __future__ import annotations

import pytest
import torch

from hybridlc.models import attention as A
from hybridlc.models.attention import AttnMixer, attn_inference, attn_reference_fp32, fp32_reference_attention
from hybridlc.models.testing import cuda_device_or_skip, rel_rms

TOL = 1e-5


def _qkv(B, H, Tq, Tk, D, dev, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn(B, H, Tq, D, generator=g).to(dev)
    k = torch.randn(B, H, Tk, D, generator=g).to(dev)
    v = torch.randn(B, H, Tk, D, generator=g).to(dev)
    return q, k, v


def _math(q, k, v, scale):
    with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.MATH]):
        Tq, Tk = q.shape[2], k.shape[2]
        if Tq == Tk:
            return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale)
        if Tq == 1:
            return torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=scale)
        mask = torch.ones(Tq, Tk, dtype=torch.bool, device=q.device).tril(Tk - Tq)
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale)


@pytest.mark.parametrize("Tq,Tk", [(64, 64), (16, 80), (1, 77), (37, 37)])
def test_tiled_core_matches_math_cpu(monkeypatch, Tq, Tk):
    monkeypatch.setattr(A, "REFERENCE_TILE_BYTES", 4 * 2 * 3 * Tk * 7)  # force several query tiles
    q, k, v = _qkv(2, 3, Tq, Tk, 16, "cpu")
    assert rel_rms(attn_reference_fp32(q, k, v, 0.25, mode="tiled"), _math(q, k, v, 0.25)) <= TOL


def _mixer(conv: bool, dev, d=384, H=6, seed=3):
    torch.manual_seed(seed)
    m = AttnMixer(d, H, 64, conv=conv).to(dev).float().eval()
    with torch.no_grad():
        for p in m.parameters():
            p.normal_(0, 0.05) if p.dim() > 1 else p.uniform_(0.8, 1.2)
    return m


def _check_mixer(conv: bool, T: int, dev) -> dict[str, float]:
    """rel RMS of R-ATT vs the default fp32 (MATH) path for every attention call of a mixer."""
    m = _mixer(conv, dev)
    g = torch.Generator(device="cpu").manual_seed(T)
    x = torch.randn(1, T, m.d_model, generator=g).to(dev)
    out = {}
    with torch.no_grad():
        # full-sequence forward with two documents (cu_seqlens isolation)
        a = T // 3
        cu = torch.tensor([0, a, T], dtype=torch.int32, device=dev)
        pos = torch.cat([torch.arange(a), torch.arange(T - a)]).to(dev)[None]
        ref = m(x, cu, pos)
        with fp32_reference_attention("tiled"):
            got = m(x, cu, pos)
        out["forward_varlen"] = rel_rms(got, ref)
        # isolation: the second document's output does not depend on the first document
        x2 = x.clone()
        x2[:, :a] = torch.randn(1, a, m.d_model, generator=g).to(dev)
        with fp32_reference_attention("tiled"):
            got2 = m(x2, cu, pos)
        out["isolation_maxdiff"] = float((got2[:, a:] - got[:, a:]).abs().max())
        # prefill (T - 16), extend 15 (lower-right), decode 1
        P = T - 16
        y_ref, st_ref = m.prefill(x[:, :P])
        e_ref, st2_ref = m.extend(x[:, P:P + 15], st_ref, P)
        d_ref, _ = m.extend(x[:, P + 15:P + 16], st2_ref, P + 15)
        with fp32_reference_attention("tiled"):
            y, st = m.prefill(x[:, :P])
            e, st2 = m.extend(x[:, P:P + 15], st, P)
            d, _ = m.extend(x[:, P + 15:P + 16], st2, P + 15)
        out["prefill"] = rel_rms(y, y_ref)
        out["extend"] = rel_rms(e, e_ref)
        out["decode"] = rel_rms(d, d_ref)
    return out


@pytest.mark.parametrize("conv", [False, True], ids=["F_g", "F_gc"])
def test_mixer_paths_match_math_cpu_512(conv):
    r = _check_mixer(conv, 512, "cpu")
    assert r["isolation_maxdiff"] == 0.0, r
    for k in ("forward_varlen", "prefill", "extend", "decode"):
        assert r[k] <= TOL, r


def test_default_off_and_context_restores():
    assert A.REFERENCE_FP32_MODE is None
    with fp32_reference_attention("tiled"):
        assert A.REFERENCE_FP32_MODE == "tiled"
        with fp32_reference_attention("efficient"):
            assert A.REFERENCE_FP32_MODE == "efficient"
        assert A.REFERENCE_FP32_MODE == "tiled"
    assert A.REFERENCE_FP32_MODE is None
    with pytest.raises(ValueError):
        with fp32_reference_attention("math"):
            pass


def test_reference_mode_refuses_half_precision():
    q, k, v = (t.to(torch.bfloat16) for t in _qkv(1, 2, 8, 8, 16, "cpu"))
    with fp32_reference_attention("tiled"):
        with pytest.raises(RuntimeError, match="fp32 only"):
            attn_inference(q, k, v, 0.25)


def test_efficient_mode_raises_when_backend_cannot_run_cpu():
    q, k, v = _qkv(1, 2, 8, 8, 16, "cpu")
    with fp32_reference_attention("efficient"):
        with pytest.raises(RuntimeError):
            attn_inference(q, k, v, 0.25)  # no EFFICIENT kernel on CPU: raises, never falls back


def test_reference_mode_makes_no_math_sdpa_call(monkeypatch):
    calls = []
    real = A._sdpa
    monkeypatch.setattr(A, "_sdpa", lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    q, k, v = _qkv(1, 2, 32, 32, 16, "cpu")
    with fp32_reference_attention("tiled"):
        attn_inference(q, k, v, 0.25)
    assert calls == []
    attn_inference(q, k, v, 0.25)  # default fp32 path: SDPA (MATH)
    assert calls == [1]


# ------------------------------------------------------------------------------------------ CUDA
@pytest.mark.gpu
@pytest.mark.parametrize("T", [512, 4096])
@pytest.mark.parametrize("conv", [False, True], ids=["F_g", "F_gc"])
def test_mixer_paths_match_math_cuda(conv, T, record_property):
    dev = cuda_device_or_skip()
    r = _check_mixer(conv, T, dev)
    record_property("rel_rms", r)
    assert r["isolation_maxdiff"] == 0.0, r
    for k in ("forward_varlen", "prefill", "extend", "decode"):
        assert r[k] <= TOL, r


@pytest.mark.gpu
def test_efficient_mode_on_this_gpu(record_property):
    """Open item of the rulings (§13): does EFFICIENT_ATTENTION run in fp32 on this GPU? Either it runs and
    matches MATH within 1e-5, or it raises (and R-ATT uses the tiled mode). Never a silent fallback."""
    dev = cuda_device_or_skip()
    q, k, v = _qkv(1, 6, 512, 512, 64, dev)
    ref = _math(q, k, v, 0.125)
    try:
        with fp32_reference_attention("efficient"):
            got = attn_inference(q, k, v, 0.125)
            ext = attn_inference(q[:, :, -15:], k, v, 0.125)
    except RuntimeError as e:
        record_property("efficient_fp32", f"raises: {str(e)[:200]}")
        return
    record_property("efficient_fp32", {"prefill": rel_rms(got, ref), "extend": rel_rms(ext, ref[:, :, -15:])})
    assert rel_rms(got, ref) <= TOL and rel_rms(ext, ref[:, :, -15:]) <= TOL


@pytest.mark.gpu
@pytest.mark.slow
def test_peak_memory_d384_c12_32k(record_property):
    from hybridlc.models.build import build_model
    from hybridlc.models.config import ModelConfig

    dev = cuda_device_or_skip(0.44)
    T = 32768
    model = build_model(ModelConfig.for_scale("D384", "C12"), seed_index=902).to(dev).float().eval()
    ids = torch.randint(8, model.cfg.vocab_size, (1, T), device=dev)
    res = {}
    with torch.no_grad(), fp32_reference_attention("tiled"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(dev)
        logits = model.full_logits(ids)  # single pass: forward + full fp32 logits (P3's ForwardOnlyAdapter path)
        assert torch.isfinite(logits[0, -1]).all()
        del logits
        res["single_pass_peak_gib"] = torch.cuda.max_memory_allocated(dev) / 2**30
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(dev)
        _, cache = model.prefill(ids[:, :T - 16])
        lp, am = model.score_continuation(cache, ids[:, T - 16:])
        assert torch.isfinite(lp).all()
        del cache
        res["cache_prefill_extend_peak_gib"] = torch.cuda.max_memory_allocated(dev) / 2**30
    record_property("peak_gib", res)
    print(res)
    assert res["single_pass_peak_gib"] <= 7.0 and res["cache_prefill_extend_peak_gib"] <= 7.0, res

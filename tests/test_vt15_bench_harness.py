"""VT-15: benchmark-harness self-test (protocol §16, §10.2; I-18, U-10).

* backend assertion -- every inference SDPA call of the model runs with only the pinned backend
  enabled (FLASH for bf16; mem-efficient never reachable), and a backend that cannot run raises;
* no masks -- decode and prefill pass no tensor mask (decode: none; prefill: ``is_causal``; teacher-
  forced continuations: the ``causal_lower_right`` bias, not a materialised mask);
* cold-L2 flush effective -- cold/hot >= 1.5 on a 64 MiB KV decode (B = 1, L = 32K, 8 x 64, bf16);
* reproduces Reviewer B's Appendix A (A5 prefill, A7 decode) within 15 % (contended), min-of-N;
* FlexAttention device-side length update inside one captured graph (informational: a failure means
  the prefill-only Flex comparator, so it is reported as xfail, not as a VT-15 failure).

Written by the S0 integrator (no builder owned VT-15). Timings print with ``-s``.
"""
from __future__ import annotations

import pytest
import torch
from torch.nn.attention import SDPBackend

from hybridlc.models.testing import cuda_device_or_skip

pytestmark = pytest.mark.gpu

# Reviewer B, Appendix A (reports/reviews/review_B_round1.md), microseconds, min-of-N, contended.
APPENDIX_A_US = {
    ("prefill", "flash", 4096): 260.0, ("prefill", "flash", 32768): 12020.0,
    ("prefill", "cudnn", 32768): 11320.0,
    ("gdn_layer", None, 4096): 690.0, ("gdn_layer", None, 32768): 2550.0,
    ("decode_hot", "flash", 1, 32768): 62.0, ("decode_hot", "cudnn", 1, 32768): 43.0,
    ("decode_hot", "flash", 8, 32768): 600.0, ("decode_hot", "cudnn", 8, 32768): 591.0,
    ("decode_cold", "flash", 1, 32768): 90.0, ("decode_cold", "cudnn", 1, 32768): 85.0,
}
TOL = 0.15
# DEVIATION (integrator, S0): hot-L2 decode at B = 1, L = 32K has a 64 MiB KV working set = the L2 size,
# and its min-of-30 is bimodal under contention (measured in one minute: cuDNN 27-80 us, flash 60-88 us;
# Appendix A's 43 / 62 us lie between or on one mode). These two cells are printed but not asserted;
# the cold-L2 cells at the same shape (the §10.2 primary decode condition) are asserted.
# Likewise the eager GDN layer at T = 4K is launch-bound (Appendix A: 0.47 of its 0.69 ms is non-core
# kernels + launch overhead), so on this shared CPU it is bimodal too (measured 743 / 734 / 385 us in
# three runs). The 32K GDN layer (compute-bound) stays asserted.
INFORMATIONAL = {("decode_hot", "cudnn", 1, 32768), ("decode_hot", "flash", 1, 32768), ("gdn_layer", None, 4096)}
BE = {"flash": SDPBackend.FLASH_ATTENTION, "cudnn": SDPBackend.CUDNN_ATTENTION}


def _small_model(pattern="--A--A--A--A"):
    from hybridlc.models.build import build_model
    from hybridlc.models.config import ModelConfig

    return build_model(ModelConfig(d_model=384, d_ff=1024, n_layers=len(pattern), vocab_size=16384,
                                   pattern=pattern), 0)


def test_backend_pinned_and_no_masks() -> None:
    from hybridlc.bench.harness import sdpa_spy

    dev = cuda_device_or_skip()
    model = _small_model().to(dev).eval()
    ids = torch.randint(8, 16384, (2, 1024), device=dev)
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16), sdpa_spy() as spy:
        _, cache = model.prefill(ids)
        model.score_continuation(cache, torch.randint(8, 16384, (2, 9), device=dev))
        tok = torch.randint(8, 16384, (2, 1), device=dev)
        for _ in range(3):
            _, cache = model.decode_step(tok, cache)
    assert spy.calls, "no SDPA call recorded"
    kinds = {(c.q_shape[2] == c.k_shape[2], c.q_shape[2] == 1, c.mask_kind) for c in spy.calls}
    for c in spy.calls:
        assert c.dtype == torch.bfloat16
        assert c.flash and not c.efficient and not c.cudnn and not c.math, c   # only the pinned backend
        assert c.mask_kind != "tensor", c                                      # never a materialised mask
        if c.q_shape[2] == 1:
            assert c.mask_kind == "none" and not c.is_causal, c                # decode: no mask at all
        elif c.q_shape[2] == c.k_shape[2]:
            assert c.is_causal and c.mask_kind == "none", c                    # prefill: is_causal
    assert (True, False, "none") in kinds and (False, True, "none") in kinds and (False, False, "causal_bias") in kinds


def test_pinned_backend_that_cannot_run_raises() -> None:
    """A masked static cache cannot run under FLASH: pinning must raise, never fall back (A7)."""
    from hybridlc.bench.harness import sdpa_kernel

    cuda_device_or_skip()
    q = torch.randn(1, 8, 1, 64, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 8, 4096, 64, device="cuda", dtype=torch.bfloat16)
    mask = torch.ones(1, 1, 1, 4096, device="cuda", dtype=torch.bool)
    with pytest.raises(RuntimeError):
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
            torch.nn.functional.scaled_dot_product_attention(q, k, k, attn_mask=mask)


def test_cold_l2_flush_effective() -> None:
    from hybridlc.bench.harness import decode_attention_case, time_replays

    cuda_device_or_skip()
    ratios = {}
    for name in ("cudnn", "flash"):
        f = decode_attention_case(1, 32768, BE[name])          # K + V = 64 MiB = L2 size
        hot, cold = time_replays(f), time_replays(f, cold_l2=True)
        ratios[name] = cold["min_us"] / hot["min_us"]
        print(f"decode B=1 L=32K {name}: hot {hot['min_us']:.1f} us, cold {cold['min_us']:.1f} us, ratio {ratios[name]:.2f}")
    assert ratios["cudnn"] >= 1.5, ratios    # cuDNN is the primary decode backend (§10.2)


_TIMINGS: dict = {}


def _appendix_a_timings() -> dict:
    """min-of-N timing of every Appendix A cell (us); measured once, shared by the guard and the registered test."""
    if _TIMINGS:
        return _TIMINGS
    from hybridlc.bench.harness import decode_attention_case, gdn_layer_case, prefill_attention_case, time_replays

    cuda_device_or_skip()
    for key, ref in APPENDIX_A_US.items():
        kind = key[0]
        if kind == "prefill":
            f, cold = prefill_attention_case(key[2], BE[key[1]]), False
        elif kind == "gdn_layer":
            f, cold = gdn_layer_case(key[2]), False
        else:
            f, cold = decode_attention_case(key[2], key[3], BE[key[1]]), kind == "decode_cold"
        m = time_replays(f, cold_l2=cold)["min_us"]
        _TIMINGS[key] = m
        print(f"{key}: {m:9.1f} us vs Appendix A {ref:9.1f} us ({100 * (m / ref - 1):+.0f}%)")
        torch.cuda.empty_cache()
    return _TIMINGS


@pytest.mark.relaxed_guard(reason="VT-15 Appendix A within 15% on 8 of 11 cells (registered: all 11)")
def test_reproduces_appendix_a_within_15pct() -> None:
    """**Regression guard, not the registered criterion** (PI ruling D059 / RR-G0-4): the 8 asserted cells; the
    3 INFORMATIONAL cells are printed only. Registered (all 11 cells): ``test_vt15_registered_all_11_cells``."""
    got = _appendix_a_timings()
    bad = [(key, round(m, 1), APPENDIX_A_US[key], round(100 * (m / APPENDIX_A_US[key] - 1)))
           for key, m in got.items() if abs(m / APPENDIX_A_US[key] - 1) > TOL and key not in INFORMATIONAL]
    assert not bad, bad


@pytest.mark.xfail(strict=False, raises=AssertionError, reason="FAIL-as-registered; accepted residual RR-G0-4; PI ruling D059; criterion unchanged")
def test_vt15_registered_all_11_cells(record_property) -> None:
    """Registered criterion (protocol §16 VT-15 / §10.2): Appendix A reproduced within 15% on all 11 cells.
    Non-strict because the B = 1 hot-L2 and eager 4K GDN cells are bimodal; an XPASS is recorded, never a PASS."""
    got = _appendix_a_timings()
    rel = {"/".join(str(x) for x in k): round(m / APPENDIX_A_US[k] - 1, 4) for k, m in got.items()}
    record_property("vt15_rel_dev", rel)
    bad = {k: v for k, v in rel.items() if abs(v) > TOL}
    print(f"VT-15 registered (all 11 within 15%): out of bound {bad}")
    assert not bad, bad


def test_flex_device_side_length_update_in_one_graph() -> None:
    """Decode over a static KV buffer; the current length lives in a device tensor read by the
    mask_mod; one graph is captured once and replayed after in-place length updates."""
    from hybridlc.bench.harness import graphed

    cuda_device_or_skip()
    try:
        from torch.nn.attention.flex_attention import create_block_mask, flex_attention

        H, Dh, Lbuf = 8, 64, 8192 + 256
        kc = torch.randn(1, H, Lbuf, Dh, device="cuda", dtype=torch.bfloat16)
        vc = torch.randn_like(kc)
        q = torch.randn(1, H, 1, Dh, device="cuda", dtype=torch.bfloat16)
        lcur = torch.tensor(Lbuf, device="cuda")

        def mm(b, h, qi, ki):
            return ki < lcur
        bm = create_block_mask(mm, 1, 1, 1, Lbuf, device="cuda", _compile=True)
        fx = torch.compile(flex_attention)
        out = torch.empty_like(q)

        def f():
            out.copy_(fx(q, kc, vc, block_mask=bm))
        replay = graphed(f)
        for L in (100, 3000, 8192):
            lcur.fill_(L)
            replay()
            torch.cuda.synchronize()
            ref = torch.nn.functional.scaled_dot_product_attention(q.float(), kc[:, :, :L].float(), vc[:, :, :L].float())
            err = (out.float() - ref).abs().max().item()
            assert err < 2e-2, (L, err)
    except Exception as e:  # noqa: BLE001 -- protocol: a Flex failure selects the prefill-only comparator
        pytest.xfail(f"FlexAttention device-side length update failed -> prefill-only Flex comparator: {e!r}"[:400])

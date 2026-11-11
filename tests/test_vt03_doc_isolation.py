"""VT-03: document isolation (protocol §16).

Packed with ``cu_seqlens`` vs each document run alone, including the GDN short-conv state, the F_gc
conv and the per-document RoPE reset. Thresholds: GDN max abs diff 0.0 (measured exact, 05 §2.2);
attention relative RMS <= 1e-2 (bf16) / <= 1e-5 (fp32).

Written by the S0 integrator (no builder owned VT-03). Mixer level for every pattern char, plus the
full ``HybridLM`` (C0, C3, X-gc3) on its varlen training path. Linear mixers are checked in bf16 (the
training dtype) and fp32; documents deliberately start at offsets that are not multiples of the
64-token chunk size.
"""
from __future__ import annotations

import pytest
import torch

from hybridlc.models.config import ModelConfig
from hybridlc.models.mixers import build_mixer
from hybridlc.models.testing import cuda_device_or_skip, rel_rms

D = 256
LENS = (37, 100, 64, 300, 11, 200)      # odd offsets: 37, 137, 201, 501, 512 (chunk 64 straddled)


def _cfg(pattern: str) -> ModelConfig:
    return ModelConfig(d_model=D, d_ff=512, n_layers=len(pattern), vocab_size=512, pattern=pattern)


def _packed(dev, dtype, seed=3):
    g = torch.Generator(device="cpu").manual_seed(seed)
    N = sum(LENS)
    x = torch.randn(1, N, D, generator=g, dtype=torch.float64).to(dev, dtype)
    cu = torch.tensor([0, *torch.tensor(LENS).cumsum(0).tolist()], dtype=torch.int32, device=dev)
    pos = torch.cat([torch.arange(n) for n in LENS])[None].to(dev)
    return x, cu, pos


def _mixer(kind: str, dev, dtype):
    torch.manual_seed(0)
    m = build_mixer(kind, _cfg("A" if kind in "AC" else kind), layer_idx=0).to(device=dev, dtype=dtype)
    if kind in "AC":
        with torch.no_grad():
            m.q_norm.uniform_(0.5, 1.5)
            m.k_norm.uniform_(0.5, 1.5)
    return m.eval()


def _alone(m, x, cu, *, model=False):
    outs = []
    b = cu.tolist()
    for a, e in zip(b[:-1], b[1:]):
        xs = x[:, a:e]
        n = e - a
        cu1 = torch.tensor([0, n], dtype=torch.int32, device=x.device)
        pos1 = torch.arange(n, device=x.device)[None]
        outs.append(m(xs, cu1, pos1))
    return torch.cat(outs, 1)


def _perturb_others(x, cu, keep: int, seed=11):
    """Replace every document except ``keep`` with fresh noise (same shapes -> same GEMM tiling)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    x2 = torch.randn(x.shape, generator=g, dtype=torch.float64).to(x.device, x.dtype) * 2
    a, e = cu[keep].item(), cu[keep + 1].item()
    x2[:, a:e] = x[:, a:e]
    return x2


RELAXED = pytest.mark.relaxed_guard(reason="VT-03 GDN fp32 packed-vs-alone rel RMS <= 3e-3 (registered: max abs 0.0)")


@pytest.mark.gpu
@pytest.mark.parametrize("kind", ["-", "E", "N", "G"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, pytest.param(torch.float32, marks=RELAXED)])
def test_linear_mixer_isolation_exact(kind: str, dtype: torch.dtype) -> None:
    """bf16 (training dtype; where 05 §2.2 measured exactness): packed == alone, max abs diff 0.0.
    fp32: **regression guard, not the registered criterion** (PI ruling D059 / RR-G0-5; the registered
    criterion is ``test_vt03_registered_gdn_fp32_max_abs_zero``): packed vs alone differs only by
    GEMM-shape (M = N vs M = n_doc, TF32) rounding, so it is held to the GDN relative-RMS bound (3e-3);
    exact isolation is then asserted with identical shapes -- perturbing all *other* documents leaves a
    document's output bit-identical."""
    dev = cuda_device_or_skip()
    m = _mixer(kind, dev, dtype)
    x, cu, pos = _packed(dev, dtype)
    with torch.no_grad():
        y = m(x, cu, pos)
        y1 = _alone(m, x, cu)
        if dtype == torch.bfloat16:
            diff = (y.float() - y1.float()).abs().max().item()
            assert diff == 0.0, (kind, dtype, diff)
        else:
            e = rel_rms(y, y1)
            assert e <= 3e-3, (kind, e)
        for j in range(len(LENS)):
            y2 = m(_perturb_others(x, cu, j, seed=20 + j), cu, pos)
            a, b = cu[j].item(), cu[j + 1].item()
            assert torch.equal(y2[:, a:b], y[:, a:b]), (kind, dtype, j)


@pytest.mark.gpu
@pytest.mark.xfail(strict=False, raises=AssertionError, reason="FAIL-as-registered; accepted residual RR-G0-5; PI ruling D059; criterion unchanged")
def test_vt03_registered_gdn_fp32_max_abs_zero(record_property) -> None:
    """Registered criterion (protocol §16 VT-03, no dtype qualifier): GDN-family mixers packed with
    cu_seqlens vs each document alone, max abs diff 0.0 -- here in fp32 (the bf16 case is asserted above).
    Measured 9.4e-5 at S0 (GEMM-shape TF32). XFAIL expected; an XPASS is recorded, never a PASS of VT-03."""
    dev = cuda_device_or_skip()
    diffs = {}
    for kind in ("-", "E", "N", "G"):
        m = _mixer(kind, dev, torch.float32)
        x, cu, pos = _packed(dev, torch.float32)
        with torch.no_grad():
            diffs[kind] = (m(x, cu, pos) - _alone(m, x, cu)).abs().max().item()
    record_property("vt03_gdn_fp32_max_abs", diffs)
    print(f"VT-03 registered (fp32, max abs 0.0): {diffs}")
    assert all(v == 0.0 for v in diffs.values()), diffs


@pytest.mark.gpu
@pytest.mark.parametrize("kind", ["A", "C"])
@pytest.mark.parametrize("dtype,tol", [(torch.bfloat16, 1e-2), (torch.float32, 1e-5)])
def test_attention_mixer_isolation(kind: str, dtype: torch.dtype, tol: float) -> None:
    """F_g / F_gc training path: varlen_attn (bf16) or the per-document SDPA reference (fp32); the
    RoPE positions restart at 0 and the F_gc conv state is reset at every boundary."""
    dev = cuda_device_or_skip()
    m = _mixer(kind, dev, dtype)
    x, cu, pos = _packed(dev, dtype)
    with torch.no_grad():
        y = m(x, cu, pos)
        y1 = _alone(m, x, cu)
    e = rel_rms(y.float(), y1.float())
    assert e <= tol, (kind, dtype, e)


# NOTE: no "RoPE reset matters" negative control -- RoPE scores depend only on relative position,
# so a constant per-document offset is (correctly) a no-op up to rounding (measured 1.6e-6 rel RMS).
# Resetting position_ids per document is checked in VT-11 (loader) and used by every test here.


def _hlm(pattern: str, dev):
    from hybridlc.models.build import build_model

    return build_model(ModelConfig(d_model=384, d_ff=1024, n_layers=len(pattern), vocab_size=16384,
                                   pattern=pattern), 0).to(dev).eval()


@pytest.mark.gpu
@pytest.mark.parametrize("pattern,name", [("------------", "C0"), ("---A---A---A", "C3-like"),
                                          ("---C---C---C", "X-gc3-like")])
def test_full_model_isolation(pattern: str, name: str) -> None:
    """Full HybridLM varlen training path. Strict isolation with identical shapes: replacing every
    other document's tokens leaves a document's hidden states bit-identical (bf16 autocast and fp32).
    Packed vs alone changes the GEMM M dimension (embedding/MLP/projections) and, in fp32, the
    TF32 chunk kernels, so it is held to the bf16 attention bound (rel RMS <= 1e-2) at both precisions;
    the per-mixer thresholds of VT-03 are asserted by the mixer-level tests above."""
    dev = cuda_device_or_skip()
    model = _hlm(pattern, dev)
    g = torch.Generator(device="cpu").manual_seed(5)
    N = sum(LENS)
    starts = torch.tensor([0, *torch.tensor(LENS).cumsum(0).tolist()[:-1]])
    ids = torch.randint(8, 16384, (1, N), generator=g)
    ids[0, starts] = 0                                          # <|endoftext|> at document starts
    ids = ids.to(dev)
    cu = torch.tensor([0, *torch.tensor(LENS).cumsum(0).tolist()], dtype=torch.int32, device=dev)
    pos = torch.cat([torch.arange(n) for n in LENS])[None].to(dev)
    b = cu.tolist()
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
            for j in range(len(LENS)):
                ids2 = torch.randint(8, 16384, (1, N), generator=g).to(dev)
                ids2[0, starts] = 0
                ids2[:, b[j]:b[j + 1]] = ids[:, b[j]:b[j + 1]]
                h2 = model(ids2, cu, pos).float()
                assert torch.equal(h2[:, b[j]:b[j + 1]], h[:, b[j]:b[j + 1]]), (name, prec, j)
        e = rel_rms(h, h1)
        assert e <= 1e-2, (name, prec, e)

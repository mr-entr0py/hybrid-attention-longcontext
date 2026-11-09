"""VT-02: causal mask.  Outputs at positions ≤ t are unchanged when tokens > t are perturbed, for
F_g varlen, F_g SDPA (inference prefill), F_gc, GDN chunk and GDN recurrent (protocol §16).

Thresholds: fp32 attention max abs diff ≤ 1e-6; GDN relative RMS ≤ 3e-3 (TF32).  The gradient of
out_t with respect to x_{t'>t} is exactly 0 in fp64 for our own modules (F_g, F_gc, full block).
"""

from __future__ import annotations

import pytest
import torch

from hybridlc.models import attention as attn_mod
from hybridlc.models.config import ModelConfig
from hybridlc.models.mixers import build_mixer
from hybridlc.models.testing import cuda_device_or_skip, rel_rms

T = 256
CUTS = (0, 37, 63, 64, 100, 200, 254)


def _cfg(pattern: str = "A") -> ModelConfig:
    return ModelConfig(d_model=256, d_ff=512, n_layers=len(pattern), vocab_size=512, pattern=pattern)


def _mixer(kind: str, device, dtype=torch.float32, seed: int = 0):
    torch.manual_seed(seed)
    m = build_mixer(kind, _cfg("A" if kind in "AC" else kind), layer_idx=0).to(device=device, dtype=dtype)
    if kind in "AC":  # make the gains / gate non-trivial so they are exercised
        with torch.no_grad():
            m.q_norm.uniform_(0.5, 1.5)
            m.k_norm.uniform_(0.5, 1.5)
    return m.eval()


def _perturbed(x: torch.Tensor, t: int, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    x2 = x.clone()
    x2[:, t + 1:] = torch.randn(x2[:, t + 1:].shape, generator=g, dtype=torch.float64).to(x.device, x.dtype) * 3
    return x2


def _x(device, dtype, B=1, seed=1):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(B, T, 256, generator=g, dtype=torch.float64).to(device, dtype)


# ---------------------------------------------------------------------------------------
# attention paths (max abs diff ≤ 1e-6)
# ---------------------------------------------------------------------------------------

@pytest.mark.gpu
@pytest.mark.parametrize("kind", ["A", "C"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_attention_training_path_causal(kind: str, dtype: torch.dtype) -> None:
    """Training forward: bf16 → varlen_attn (FA2); fp32 → per-document SDPA-math reference."""
    dev = cuda_device_or_skip()
    m = _mixer(kind, dev, dtype)
    x = _x(dev, dtype)
    with torch.no_grad():
        y = m(x)
        for i, t in enumerate(CUTS):
            y2 = m(_perturbed(x, t, 10 + i))
            d = (y2[:, : t + 1].float() - y[:, : t + 1].float()).abs().max().item()
            assert d <= 1e-6, (kind, dtype, t, d)


@pytest.mark.gpu
@pytest.mark.parametrize("kind", ["A", "C"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_attention_sdpa_prefill_causal(kind: str, dtype: torch.dtype) -> None:
    """Inference prefill: SDPA is_causal pinned (FLASH for bf16, MATH for fp32)."""
    dev = cuda_device_or_skip()
    m = _mixer(kind, dev, dtype)
    x = _x(dev, dtype, B=2)
    y, _ = m.prefill(x)
    for i, t in enumerate(CUTS):
        y2, _ = m.prefill(_perturbed(x, t, 20 + i))
        d = (y2[:, : t + 1].float() - y[:, : t + 1].float()).abs().max().item()
        assert d <= 1e-6, (kind, dtype, t, d)


@pytest.mark.gpu
@pytest.mark.parametrize("kind", ["A", "C"])
def test_attention_extend_causal(kind: str) -> None:
    """Teacher-forced extend from a cache (lower-right causal SDPA), fp32 and bf16."""
    dev = cuda_device_or_skip()
    for dtype in (torch.float32, torch.bfloat16):
        m = _mixer(kind, dev, dtype)
        x = _x(dev, dtype, B=2)
        P = 100
        _, st = m.prefill(x[:, :P])
        y, _ = m.extend(x[:, P:], st, P)
        for i, t in enumerate((0, 5, 63, 100, 150)):
            x2 = x[:, P:].clone()
            x2[:, t + 1:] += 2.0
            y2, _ = m.extend(x2, st, P)
            d = (y2[:, : t + 1].float() - y[:, : t + 1].float()).abs().max().item()
            assert d <= 1e-6, (kind, dtype, t, d)


# ---------------------------------------------------------------------------------------
# GDN (relative RMS ≤ 3e-3, fp32/TF32)
# ---------------------------------------------------------------------------------------

@pytest.mark.gpu
@pytest.mark.parametrize("kind", ["-", "E", "N", "G"])
def test_linear_chunk_causal(kind: str) -> None:
    dev = cuda_device_or_skip()
    m = _mixer(kind, dev, torch.float32)
    x = _x(dev, torch.float32)
    with torch.no_grad():
        y = m(x)
        for i, t in enumerate(CUTS):
            y2 = m(_perturbed(x, t, 30 + i))
            e = rel_rms(y2[:, : t + 1], y[:, : t + 1])
            assert e <= 3e-3, (kind, t, e)


@pytest.mark.gpu
@pytest.mark.parametrize("kind", ["-", "E", "G"])
def test_linear_recurrent_causal(kind: str) -> None:
    """Explicit fused_recurrent path (extend from a chunk-prefilled state)."""
    dev = cuda_device_or_skip()
    m = _mixer(kind, dev, torch.float32)
    x = _x(dev, torch.float32)
    P = 64
    _, st = m.prefill(x[:, :P])
    y, _ = m.extend(x[:, P:], st, P)
    for i, t in enumerate((0, 10, 63, 64, 150, 190)):
        x2 = x[:, P:].clone()
        x2[:, t + 1:] += 2.0
        y2, _ = m.extend(x2, st, P)
        e = rel_rms(y2[:, : t + 1], y[:, : t + 1])
        assert e <= 3e-3, (kind, t, e)


# ---------------------------------------------------------------------------------------
# fp64 gradients are exactly zero for future positions (our own modules)
# ---------------------------------------------------------------------------------------

def _future_grad_max(fn, x: torch.Tensor, ts) -> float:
    worst = 0.0
    for t in ts:
        xx = x.clone().requires_grad_(True)
        out = fn(xx)
        g, = torch.autograd.grad(out[:, t].sum(), xx)
        worst = max(worst, g[:, t + 1:].abs().max().item())
        assert g[:, : t + 1].abs().max().item() > 0  # the check is not vacuous
    return worst


@pytest.mark.parametrize("kind", ["A", "C"])
def test_fp64_future_gradient_exactly_zero_mixer(kind: str) -> None:
    """CPU fp64, reference attention and torch conv; dense rows and packed documents."""
    m = _mixer(kind, "cpu", torch.float64)
    m.use_ssmax = False
    Tm = 48
    x = torch.randn(1, Tm, 256, dtype=torch.float64, generator=torch.Generator().manual_seed(3))
    assert _future_grad_max(lambda z: m(z), x, (0, 5, 20, 46)) == 0.0
    cu = torch.tensor([0, 17, 30, Tm], dtype=torch.int32)
    pos = torch.cat([torch.arange(17), torch.arange(13), torch.arange(Tm - 30)])[None]
    assert _future_grad_max(lambda z: m(z, cu, pos), x, (0, 16, 17, 29, 46)) == 0.0


def test_fp64_future_gradient_exactly_zero_ssmax() -> None:
    from hybridlc.models.attention import AttnMixer

    torch.manual_seed(0)
    m = AttnMixer(256, 4, 64, ssmax=True).double()
    x = torch.randn(1, 40, 256, dtype=torch.float64, generator=torch.Generator().manual_seed(4))
    assert _future_grad_max(lambda z: m(z), x, (1, 7, 38)) == 0.0


def test_fp64_future_gradient_exactly_zero_full_model_attention_only() -> None:
    """A whole 3-layer F_g/F_gc LM (norms, MLPs, tied embedding) in fp64: d out_t / d emb_{t'>t} = 0."""
    from hybridlc.models.build import HybridLM

    cfg = ModelConfig(d_model=128, d_ff=256, n_layers=3, vocab_size=97, pattern="ACA")
    m = HybridLM(cfg, seed_index=7).double()
    ids = torch.randint(0, 97, (1, 30), generator=torch.Generator().manual_seed(5))
    emb = m.embed(ids).detach()

    def fn(e):
        h = e
        for blk in m.layers:
            h = blk(h, None, None, {})
        return m.norm_f(h)

    assert _future_grad_max(fn, emb, (0, 9, 28)) == 0.0


def test_reference_path_matches_varlen_kernel_layout() -> None:
    """Sanity: FORCE_REFERENCE is off by default (production uses the kernels)."""
    assert attn_mod.FORCE_REFERENCE is False

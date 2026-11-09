"""VT-04: GDN chunk kernel vs a naive fp64 recurrence (protocol §16).

T ∈ {512, 32,768}; decays mixed (e^−0.5 … e^−1e−4) and near-unit (1e−4); geometries 4 × 64/128,
3 × 64/128, 1 × 256/512, 1 × 192/384.  Pass: relative RMS ≤ 3e-3 (fp32/TF32 inputs) and ≤ 1e-2
(bf16), flat across position bands; gradient vs fp64 autograd at T = 512: relative RMS ≤ 5e-3.

Every fla call uses keyword arguments only (VT-08).  The T = 32K cases are marked slow.
"""

from __future__ import annotations

import pytest
import torch

from hybridlc.models import linear as L
from hybridlc.models.testing import (band_rel_rms, cuda_device_or_skip, make_gdn_inputs, naive_gated_delta_autograd,
                                     naive_gated_delta_fp64, rel_rms)

GEOMETRIES = [(4, 64, 128), (3, 64, 128), (1, 256, 512), (1, 192, 384)]
TOL = {torch.float32: 3e-3, torch.bfloat16: 1e-2}


def _chunk(q, k, v, g, beta, dtype, h0=None, final=True):
    return L.chunk_gated_delta_rule(q=q.to(dtype), k=k.to(dtype), v=v.to(dtype), g=g.float(), beta=beta.to(dtype),
                                    initial_state=h0, output_final_state=final)


def _check(T: int, geo, decay: str) -> None:
    dev = cuda_device_or_skip()
    H, K, V = geo
    q, k, v, g, beta = make_gdn_inputs(1, T, H, K, V, decay, dev, seed=T + K + (decay == "near1"))
    ref, href = naive_gated_delta_fp64(q, k, v, g, beta)
    for dtype, tol in TOL.items():
        o, hT = _chunk(q, k, v, g, beta, dtype)
        assert torch.isfinite(o).all()
        e = rel_rms(o, ref)
        bands = band_rel_rms(o, ref, n_bands=4)
        assert e <= tol, (T, geo, decay, dtype, e)
        assert max(bands) <= tol, (T, geo, decay, dtype, bands)
        # state is in the default [K, V] layout here
        st = rel_rms(hT, href)
        assert st <= tol, (T, geo, decay, dtype, "state", st)


@pytest.mark.gpu
@pytest.mark.parametrize("decay", ["mixed", "near1"])
@pytest.mark.parametrize("geo", GEOMETRIES, ids=lambda g: f"{g[0]}x{g[1]}-{g[2]}")
def test_chunk_vs_naive_T512(geo, decay) -> None:
    _check(512, geo, decay)


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("decay", ["mixed", "near1"])
@pytest.mark.parametrize("geo", GEOMETRIES, ids=lambda g: f"{g[0]}x{g[1]}-{g[2]}")
def test_chunk_vs_naive_T32k(geo, decay) -> None:
    _check(32768, geo, decay)


@pytest.mark.gpu
@pytest.mark.parametrize("decay", ["mixed", "near1"])
@pytest.mark.parametrize("geo", GEOMETRIES, ids=lambda g: f"{g[0]}x{g[1]}-{g[2]}")
def test_chunk_gradient_vs_fp64_autograd(geo, decay) -> None:
    """Gradients of a random projection of the output w.r.t. q, k, v, g, beta at T = 512 (fp32 inputs)."""
    dev = cuda_device_or_skip()
    H, K, V = geo
    T = 512
    q, k, v, g, beta = make_gdn_inputs(1, T, H, K, V, decay, dev, seed=7 + K)
    w = torch.randn(1, T, H, V, dtype=torch.float64, generator=torch.Generator().manual_seed(11)).to(dev)
    leaves64 = [t.clone().requires_grad_(True) for t in (q, k, v, g, beta)]
    o64, _ = naive_gated_delta_autograd(*leaves64)
    grads64 = torch.autograd.grad((o64 * w).sum(), leaves64)
    leaves32 = [t.float().clone().requires_grad_(True) for t in (q, k, v, g, beta)]
    o32, _ = L.chunk_gated_delta_rule(q=leaves32[0], k=leaves32[1], v=leaves32[2], g=leaves32[3], beta=leaves32[4],
                                      output_final_state=False)
    grads32 = torch.autograd.grad((o32 * w.float()).sum(), leaves32)
    for name, a, b in zip("q k v g beta".split(), grads32, grads64):
        e = rel_rms(a, b)
        assert e <= 5e-3, (geo, decay, name, e)


@pytest.mark.gpu
def test_layer_gating_matches_naive_with_fla_gate_formula() -> None:
    """The in-kernel gate path used by GDNMixer (use_gate_in_kernel, sigmoid β, in-kernel L2 norm)
    equals the naive recurrence with α = exp(−exp(A_log)·softplus(a + dt_bias)), β = σ(b), q,k L2-normed."""
    dev = cuda_device_or_skip()
    torch.manual_seed(0)
    B, T, H, K, V = 1, 512, 4, 64, 128
    gen = torch.Generator().manual_seed(5)
    q = torch.randn(B, T, H, K, generator=gen).to(dev)
    k = torch.randn(B, T, H, K, generator=gen).to(dev)
    v = torch.randn(B, T, H, V, generator=gen).to(dev)
    a = torch.randn(B, T, H, generator=gen).to(dev)
    b = torch.randn(B, T, H, generator=gen).to(dev)
    A_log = torch.log(torch.empty(H).uniform_(0, 16, generator=gen)).to(dev)
    dt_bias = torch.randn(H, generator=gen).to(dev) * 0.5 - 4
    o, hT = L.chunk_gated_delta_rule(q=q, k=k, v=v, g=a, beta=b, output_final_state=True,
                                     use_qk_l2norm_in_kernel=True, use_beta_sigmoid_in_kernel=True,
                                     allow_neg_eigval=False, state_v_first=True, use_gate_in_kernel=True,
                                     A_log=A_log, dt_bias=dt_bias)
    g_ref = -A_log.double().exp() * torch.nn.functional.softplus(a.double() + dt_bias.double())
    ref, href = naive_gated_delta_fp64(torch.nn.functional.normalize(q.double(), dim=-1),
                                       torch.nn.functional.normalize(k.double(), dim=-1), v, g_ref,
                                       torch.sigmoid(b.double()))
    assert rel_rms(o, ref) <= 3e-3
    assert rel_rms(hT.transpose(-1, -2), href) <= 3e-3  # state_v_first=True stores [V, K]

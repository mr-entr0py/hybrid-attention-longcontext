"""VT-16: GDN-E at K = 256 / 192 and GLA — forward/backward finite; numerics as VT-04 (protocol §16).

The op-level GDN-E geometries (1 × 256/512, 1 × 192/384) vs the fp64 recurrence are covered in
``test_vt04_gdn_numerics.py``; here: the layers at both scales (packed varlen, bf16 autocast, as in
training) and GLA's chunk kernel vs an fp64 recurrence (relative RMS ≤ 3e-3 fp32 / ≤ 1e-2 bf16,
flat across position bands) plus its gradient vs fp64 autograd (≤ 5e-3).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from hybridlc.models import linear as L
from hybridlc.models.config import ModelConfig
from hybridlc.models.mixers import build_mixer
from hybridlc.models.testing import band_rel_rms, cuda_device_or_skip, naive_gla_fp64, rel_rms

TOL = {torch.float32: 3e-3, torch.bfloat16: 1e-2}


@pytest.mark.gpu
@pytest.mark.parametrize("scale,kind,K", [("D512", "E", 256), ("D384", "E", 192), ("D512", "G", 64), ("D384", "G", 64)])
def test_layer_fwd_bwd_finite_varlen_bf16(scale, kind, K) -> None:
    dev = cuda_device_or_skip()
    cfg = ModelConfig.for_scale(scale, "C0")
    torch.manual_seed(0)
    m = build_mixer(kind, cfg, 0).to(dev)
    assert m.head_k_dim == K
    N = 4096
    x = torch.randn(1, N, cfg.d_model, device=dev, requires_grad=True)
    cu = torch.tensor([0, 1000, 1001, 2500, 4096], device=dev, dtype=torch.int32)
    with torch.autocast("cuda", torch.bfloat16):
        y = m(x, cu)
    y.float().pow(2).mean().backward()
    assert torch.isfinite(y).all() and torch.isfinite(x.grad).all()
    for n, p in m.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), n


def _gla_inputs(dev, T, H, K, V, seed=0):
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn(1, T, H, K, generator=gen, dtype=torch.float64)
    k = torch.randn(1, T, H, K, generator=gen, dtype=torch.float64)
    v = torch.randn(1, T, H, V, generator=gen, dtype=torch.float64)
    # GLA's own parametrisation: logsigmoid(·)/16 (decays e^{-x}, x ∈ ~[0, 0.5])
    gk = F.logsigmoid(torch.randn(1, T, H, K, generator=gen, dtype=torch.float64) * 2) / 16
    return [t.to(dev) for t in (q, k, v, gk)]


@pytest.mark.gpu
@pytest.mark.parametrize("T", [512, 4096])
def test_gla_chunk_vs_naive(T) -> None:
    dev = cuda_device_or_skip()
    H, K, V = 4, 64, 128
    q, k, v, gk = _gla_inputs(dev, T, H, K, V, seed=T)
    ref, href = naive_gla_fp64(q, k, v, gk)
    for dtype, tol in TOL.items():
        o, hT = L.chunk_gla(q=q.to(dtype), k=k.to(dtype), v=v.to(dtype), g=gk.float(), initial_state=None,
                            output_final_state=True)
        assert torch.isfinite(o).all()
        assert rel_rms(o, ref) <= tol, (dtype, rel_rms(o, ref))
        assert max(band_rel_rms(o, ref)) <= tol
        assert rel_rms(hT, href) <= tol
        o_r, h_r = L.fused_recurrent_gla(q=q.to(dtype), k=k.to(dtype), v=v.to(dtype), gk=gk.float(),
                                         initial_state=None, output_final_state=True)
        assert rel_rms(o_r, ref) <= tol and rel_rms(h_r, href) <= tol


@pytest.mark.gpu
def test_gla_gradient_vs_fp64_autograd() -> None:
    dev = cuda_device_or_skip()
    H, K, V, T = 4, 64, 128, 512
    q, k, v, gk = _gla_inputs(dev, T, H, K, V, seed=9)
    w = torch.randn(1, T, H, V, dtype=torch.float64, generator=torch.Generator().manual_seed(1)).to(dev)
    leaves = [t.clone().requires_grad_(True) for t in (q, k, v, gk)]
    s = K ** -0.5
    h = torch.zeros(1, H, K, V, dtype=torch.float64, device=dev)
    outs = []
    for t in range(T):
        h = h * leaves[3][:, t].exp()[..., None] + leaves[1][:, t][..., None] * leaves[2][:, t][..., None, :]
        outs.append(torch.einsum("bhk,bhkv->bhv", leaves[0][:, t] * s, h))
    g64 = torch.autograd.grad((torch.stack(outs, 1) * w).sum(), leaves)
    l32 = [t.float().clone().requires_grad_(True) for t in (q, k, v, gk)]
    o32, _ = L.chunk_gla(q=l32[0], k=l32[1], v=l32[2], g=l32[3], initial_state=None, output_final_state=False)
    g32 = torch.autograd.grad((o32 * w.float()).sum(), l32)
    for name, a, b in zip("q k v gk".split(), g32, g64):
        assert rel_rms(a, b) <= 5e-3, (name, rel_rms(a, b))

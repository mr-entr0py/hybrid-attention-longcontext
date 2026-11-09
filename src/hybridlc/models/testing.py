"""Reference implementations and helpers shared by the model VT tests (VT-02, 04–10, 14, 16).

Nothing here is used on the training or evaluation path.
"""

from __future__ import annotations

import math

import torch


def rel_rms(a: torch.Tensor, ref: torch.Tensor) -> float:
    """‖a − ref‖_rms / ‖ref‖_rms (computed in fp64)."""
    a64, r64 = a.double(), ref.double()
    return float((a64 - r64).pow(2).mean().sqrt() / r64.pow(2).mean().sqrt().clamp_min(1e-300))


def rel_max(a: torch.Tensor, ref: torch.Tensor) -> float:
    """max|a − ref| / max|ref| (computed in fp64)."""
    a64, r64 = a.double(), ref.double()
    return float((a64 - r64).abs().max() / r64.abs().max().clamp_min(1e-300))


def band_rel_rms(a: torch.Tensor, ref: torch.Tensor, n_bands: int = 4, dim: int = 1) -> list[float]:
    """Relative RMS error per equal-width position band along ``dim``, normalised by the
    *global* reference RMS (so a band with a small signal is not over-weighted)."""
    T = ref.shape[dim]
    scale = ref.double().pow(2).mean().sqrt().clamp_min(1e-300)
    out = []
    for i in range(n_bands):
        a_ = a.narrow(dim, i * T // n_bands, T // n_bands).double()
        r_ = ref.narrow(dim, i * T // n_bands, T // n_bands).double()
        out.append(float((a_ - r_).pow(2).mean().sqrt() / scale))
    return out


@torch.no_grad()
def naive_gated_delta_fp64(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor,
                           scale: float | None = None, h0: torch.Tensor | None = None
                           ) -> tuple[torch.Tensor, torch.Tensor]:
    """Gated delta rule recurrence in fp64 (no grad), state layout [B, H, K, V]:

        S_t = α_t S_{t−1} + β_t k_t (v_t − α_t S_{t−1}ᵀ k_t)ᵀ,   o_t = S_tᵀ (scale·q_t),   α_t = exp(g_t)

    q, k: [B, T, H, K] (already L2-normalised if desired); v: [B, T, H, V]; g (log decay), beta: [B, T, H].
    """
    return _naive_gdn(q.double(), k.double(), v.double(), g.double(), beta.double(), scale,
                      None if h0 is None else h0.double())


def naive_gated_delta_autograd(q, k, v, g, beta, scale=None, h0=None):
    """Same recurrence, differentiable (use fp64 inputs for the gradient reference)."""
    return _naive_gdn(q, k, v, g, beta, scale, h0)


def _naive_gdn(q, k, v, g, beta, scale, h0):
    B, T, H, K = k.shape
    V = v.shape[-1]
    s = K ** -0.5 if scale is None else scale
    h = torch.zeros(B, H, K, V, dtype=q.dtype, device=q.device) if h0 is None else h0
    outs = []
    for t in range(T):
        h = h * g[:, t].exp()[..., None, None]
        vn = (v[:, t] - torch.einsum("bhk,bhkv->bhv", k[:, t], h)) * beta[:, t][..., None]
        h = h + k[:, t][..., None] * vn[..., None, :]
        outs.append(torch.einsum("bhk,bhkv->bhv", q[:, t] * s, h))
    return torch.stack(outs, 1), h


@torch.no_grad()
def naive_gla_fp64(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, gk: torch.Tensor, scale: float | None = None,
                   h0: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """GLA recurrence in fp64: S_t = diag(exp(gk_t)) S_{t−1} + k_t v_tᵀ; o_t = S_tᵀ (scale·q_t).

    q, k, gk: [B, T, H, K]; v: [B, T, H, V]; state [B, H, K, V]."""
    q, k, v, gk = (t.double() for t in (q, k, v, gk))
    B, T, H, K = k.shape
    V = v.shape[-1]
    s = K ** -0.5 if scale is None else scale
    h = torch.zeros(B, H, K, V, dtype=torch.float64, device=q.device) if h0 is None else h0.double()
    outs = []
    for t in range(T):
        h = h * gk[:, t].exp()[..., None] + k[:, t][..., None] * v[:, t][..., None, :]
        outs.append(torch.einsum("bhk,bhkv->bhv", q[:, t] * s, h))
    return torch.stack(outs, 1), h


def make_gdn_inputs(B: int, T: int, H: int, K: int, V: int, decay: str, device, seed: int = 0,
                    dtype: torch.dtype = torch.float64):
    """Random GDN op inputs (q, k unit-norm; beta ∈ (0, 1)).

    decay='mixed': per-token log-decay −exp(U[ln 1e-4, ln 0.5]) (α ∈ [e^−0.5, e^−1e−4]);
    decay='near1': constant log-decay −1e−4 (worst case for long-range accumulation).
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)

    def rn(*shape):
        return torch.randn(*shape, generator=gen, dtype=torch.float64).to(device)

    q = torch.nn.functional.normalize(rn(B, T, H, K), dim=-1)
    k = torch.nn.functional.normalize(rn(B, T, H, K), dim=-1)
    v = rn(B, T, H, V)
    beta = torch.sigmoid(rn(B, T, H))
    if decay == "mixed":
        u = torch.rand(B, T, H, generator=gen, dtype=torch.float64).to(device)
        g = -torch.exp(u * (math.log(0.5) - math.log(1e-4)) + math.log(1e-4))
    elif decay == "near1":
        g = torch.full((B, T, H), -1e-4, dtype=torch.float64, device=device)
    else:
        raise ValueError(decay)
    return tuple(t.to(dtype) for t in (q, k, v, g, beta))


def cuda_device_or_skip(fraction: float = 0.44):
    """Claim the shared GPU (memory cap) or skip the calling test."""
    import pytest

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    from hybridlc.utils.gpu import claim

    return claim(fraction)

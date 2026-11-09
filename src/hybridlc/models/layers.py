"""Shared building blocks: RMSNorm, SwiGLU MLP, RoPE tables, fp32-output logit GEMM.

Numerics conventions (protocol §3.1, §7.1):
- RMSNorm computes in fp32 and returns the input dtype (residual stream stays fp32 under
  bf16 autocast because the embedding output is fp32).
- RoPE uses the HF ``rotate_half`` convention (first half / second half), so the full
  mixer transplants 1:1 onto ``transformers`` Qwen3 (VT-09).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """RMSNorm with learnable gain (init 1), eps 1e-5, no bias; fp32 compute."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (xf * self.weight.float()).to(x.dtype)


class SwiGLU(nn.Module):
    """SwiGLU(z) = W_down(SiLU(W_gate z) ⊙ W_up z), no biases."""

    def __init__(self, d: int, d_ff: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d, d_ff, bias=False)
        self.up_proj = nn.Linear(d, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------------------

def rope_tables(position_ids: torch.Tensor, head_dim: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    """fp32 (cos, sin) of shape ``position_ids.shape + (head_dim,)`` in the HF layout
    (``emb = cat(freqs, freqs)``)."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=position_ids.device, dtype=torch.int64).float() / head_dim))
    freqs = position_ids.float()[..., None] * inv_freq
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: [..., T, H, D] (any float dtype); cos/sin: [..., T, D] fp32 broadcast over heads."""
    c = cos.unsqueeze(-2).to(x.dtype)
    s = sin.unsqueeze(-2).to(x.dtype)
    return x * c + rotate_half(x) * s


# ---------------------------------------------------------------------------------------
# Logits with fp32 output
# ---------------------------------------------------------------------------------------

def _compute_dtype(x: torch.Tensor) -> torch.dtype:
    """Dtype the GEMM runs in: the autocast dtype when autocast is active on x's device,
    else x's own dtype."""
    dev = x.device.type
    if torch.is_autocast_enabled(dev):
        return torch.get_autocast_dtype(dev)
    return x.dtype


def _mm_fp32_out(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a @ b with fp32 output.  For half inputs on CUDA this uses ``torch.mm(..., out_dtype=fp32)``:
    half GEMM with fp32 accumulate **and** fp32 output (no bf16 rounding of the logits)."""
    if a.dtype in (torch.bfloat16, torch.float16) and a.is_cuda:
        return torch.mm(a, b, out_dtype=torch.float32)
    return torch.mm(a.float(), b.float())


class _LinearFP32Out(torch.autograd.Function):
    """y = h @ Wᵀ in ``cdt`` with fp32 output; backward in ``cdt`` with fp32 accumulation."""

    @staticmethod
    def forward(ctx, h: torch.Tensor, w: torch.Tensor, cdt: torch.dtype) -> torch.Tensor:  # type: ignore[override]
        with torch.autocast(h.device.type, enabled=False):
            hc, wc = h.to(cdt), w.to(cdt)
            ctx.save_for_backward(hc, wc)
            ctx.h_dtype, ctx.w_dtype = h.dtype, w.dtype
            return _mm_fp32_out(hc, wc.t())

    @staticmethod
    def backward(ctx, gy: torch.Tensor):  # type: ignore[override]
        hc, wc = ctx.saved_tensors
        with torch.autocast(gy.device.type, enabled=False):
            g = gy.to(hc.dtype)
            dh = dw = None
            if ctx.needs_input_grad[0]:
                dh = _mm_fp32_out(g, wc).to(ctx.h_dtype)
            if ctx.needs_input_grad[1]:
                dw = _mm_fp32_out(g.t(), hc).to(ctx.w_dtype)
        return dh, dw, None


def logits_fp32(h: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """fp32 logits ``h @ weightᵀ`` for h of shape [..., d].  Runs the GEMM in the autocast
    dtype if autocast is active (bf16 in training/eval), with fp32 output."""
    shape = h.shape[:-1]
    out = _LinearFP32Out.apply(h.reshape(-1, h.shape[-1]), weight, _compute_dtype(h))
    return out.view(*shape, weight.shape[0])


def init_normal_(t: torch.Tensor, std: float, generator: torch.Generator) -> None:
    """In-place N(0, std²) draw on CPU with ``generator`` (device-independent), copied to t."""
    with torch.no_grad():
        if t.device.type == "cpu":
            t.normal_(0.0, std, generator=generator)
        else:
            t.copy_(torch.empty(t.shape, dtype=t.dtype).normal_(0.0, std, generator=generator))


def output_proj_std(init_std: float, n_layers: int) -> float:
    """Std of residual-output projections: init_std / √(2·n_layers) (= 0.02/√24 at L = 12)."""
    return init_std / math.sqrt(2 * n_layers)

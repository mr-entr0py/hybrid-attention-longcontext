"""Chunked cross-entropy with z-loss (protocol §7.1, §7.3).

``chunked_ce(hidden, weight, labels, chunk=4096, z_loss_coef=1e-4, ignore_index=1, loss_weights=None)``

* logits = hidden · weightᵀ in fp32, one ``chunk``-token slice at a time; each slice runs under
  non-reentrant ``torch.utils.checkpoint`` so its [chunk, V] fp32 logits are *recomputed* in backward
  instead of stored (at 32K × 16K V the full logits would be 2 GiB);
* ``loss``   = Σ_i w_i·CE_i / Σ_i w_i over non-ignored tokens (w ≡ 1 without ``loss_weights``; with
  the TEF weights of §7.3 this is ½·mean CE(answers) + ½·mean CE(rest));
* ``z_loss`` = z_loss_coef · mean_i logsumexp_i² over non-ignored tokens (already multiplied);
* ``n_tokens`` = number of non-ignored tokens (int64 tensor).

Ignored targets: ``labels == ignore_index`` (the pad id 1) and any negative label (−100).  This is
exactly the reference contract of ``hybridlc.training.fakes.fake_chunked_ce``.
"""

from __future__ import annotations

import torch
from torch.utils.checkpoint import checkpoint

from .layers import logits_fp32


def _chunk_terms(h: torch.Tensor, weight: torch.Tensor, y: torch.Tensor, valid: torch.Tensor,
                 w: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logits = logits_fp32(h, weight)                      # [c, V] fp32
    lse = torch.logsumexp(logits, dim=-1)                # [c]
    tgt = logits.gather(-1, y.masked_fill(~valid, 0)[:, None].long())[:, 0]
    ce = lse - tgt
    vf = valid.to(torch.float32)
    ws = vf if w is None else w.float() * vf
    return (ce * ws).sum(), ws.sum(), (lse.square() * vf).sum()


def chunked_ce(hidden: torch.Tensor, weight: torch.Tensor, labels: torch.Tensor, chunk: int = 4096,
               z_loss_coef: float = 1e-4, ignore_index: int = 1, loss_weights: torch.Tensor | None = None,
               ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """See the module docstring.  hidden [..., d] (already final-normed); weight [V, d]; labels [...]."""
    h = hidden.reshape(-1, hidden.shape[-1])
    y = labels.reshape(-1)
    if y.shape[0] != h.shape[0]:
        raise ValueError(f"labels ({y.shape[0]}) and hidden ({h.shape[0]}) token counts differ")
    valid = (y != ignore_index) & (y >= 0)
    w = loss_weights.reshape(-1) if loss_weights is not None else None
    ce_sum = h.new_zeros((), dtype=torch.float32)
    w_sum = h.new_zeros((), dtype=torch.float32)
    z_sum = h.new_zeros((), dtype=torch.float32)
    use_ckpt = torch.is_grad_enabled() and (h.requires_grad or weight.requires_grad)
    for a in range(0, h.shape[0], chunk):
        b = min(a + chunk, h.shape[0])
        args = (h[a:b], weight, y[a:b], valid[a:b], None if w is None else w[a:b])
        if use_ckpt:
            c, ws, z = checkpoint(_chunk_terms, *args, use_reentrant=False, preserve_rng_state=False)
        else:
            c, ws, z = _chunk_terms(*args)
        ce_sum = ce_sum + c
        w_sum = w_sum + ws
        z_sum = z_sum + z
    n = valid.sum()
    loss = ce_sum / w_sum.clamp_min(1e-30)
    z_loss = z_loss_coef * z_sum / n.clamp_min(1).to(torch.float32)
    return loss, z_loss, n

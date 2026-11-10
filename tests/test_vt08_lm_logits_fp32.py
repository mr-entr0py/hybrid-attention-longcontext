"""VT-08 regression, verification round 2 (TIER0-LM-LOGITS-BF16).

The Tier-0 LM path (``scorer._nll``, used by LM-1/2/3/4) must compute fp32 logits even though it is
called inside ``score_lm_items``' bf16 autocast region: the LM-head GEMM runs with autocast disabled.
Before the fix the GEMM ran in bf16 and returned rounded bf16 logits (per-token |dCE| up to ~0.08 on
the VT-13 checkpoint).
"""
from __future__ import annotations

import numpy as np
import torch

from hybridlc.eval import scorer
from hybridlc.models.testing import cuda_device_or_skip


class _HiddenOnly(torch.nn.Module):
    """forward(x, cu, pos) -> fixed fp32 hidden states; ``lm_head_weight`` like HybridLM."""

    def __init__(self, n: int, d: int, v: int, dev: torch.device):
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.h = torch.nn.Parameter(torch.randn(1, n, d, generator=g).to(dev), requires_grad=False)
        self.w = torch.nn.Parameter((torch.randn(v, d, generator=g) * 0.5).to(dev), requires_grad=False)

    @property
    def lm_head_weight(self) -> torch.Tensor:
        return self.w

    def forward(self, x, cu, pos):  # noqa: D401 - same signature as HybridLM.forward
        return self.h


def test_nll_logits_fp32_under_autocast():
    dev = cuda_device_or_skip()
    n, d, v = 512, 384, 4096
    m = _HiddenOnly(n, d, v, dev)
    ids = np.random.default_rng(0).integers(2, v, size=n)
    seen = []
    orig = torch.nn.functional.cross_entropy

    def spy(lg, *a, **k):
        seen.append(lg.dtype)
        return orig(lg, *a, **k)

    torch.nn.functional.cross_entropy = spy
    try:
        with torch.no_grad(), scorer._autocast(dev, True):  # noqa: SLF001 - the region score_lm_items uses
            got = scorer._nll(m, ids, None, None)  # noqa: SLF001
    finally:
        torch.nn.functional.cross_entropy = orig
    assert seen and all(t == torch.float32 for t in seen), seen
    # exact fp32 reference from the same hidden states
    with torch.no_grad():
        lg = (m.h[0].double() @ m.w.double().T)
        lab = torch.from_numpy(np.r_[ids[1:], 0]).to(dev)
        ref = torch.nn.functional.cross_entropy(lg, lab, reduction="none").cpu().numpy()
    ok = ~np.isnan(got)
    assert ok.sum() == n - 1
    err = np.abs(got[ok] - ref[ok]).max()
    assert err < 1e-3, err  # bf16 logits of this size give errors ~1e-1

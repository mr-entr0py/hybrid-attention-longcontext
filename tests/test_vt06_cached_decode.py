"""VT-06: cached vs uncached decode (protocol §16).

Prefill P + M cached decode steps vs one forward over P + M, for k ∈ {0, 3, 12} (plus F_gc and
GDN-E arms).  fp32 pass: argmax agreement 1.000 and max |Δlogit| ≤ 2e-2.  The bf16 criterion of
the protocol (EM flip rate < 1 % on a 5 % re-score) applies to trained checkpoints and is run by the
evaluation scorer; here bf16 is reported as a sanity bound only.
"""

from __future__ import annotations

import pytest
import torch

from hybridlc.models.build import build_model
from hybridlc.models.config import ModelConfig
from hybridlc.models.testing import cuda_device_or_skip

P, M = 192, 24


def _run(name: str, dtype: torch.dtype, scale: str = "D384"):
    dev = cuda_device_or_skip()
    cfg = ModelConfig.for_scale(scale, name)
    model = build_model(cfg, seed_index=900).to(dev).eval()
    gen = torch.Generator().manual_seed(3)
    ids = torch.randint(8, cfg.vocab_size, (2, P + M), generator=gen).to(dev)
    ac = torch.autocast("cuda", torch.bfloat16, enabled=(dtype == torch.bfloat16))
    with torch.no_grad(), ac:
        full = model.full_logits(ids)                         # [B, P+M, V]
        last, cache = model.prefill(ids[:, :P])
        step_logits = [last]
        for t in range(P, P + M - 1):
            lg, cache = model.decode_step(ids[:, t:t + 1], cache)
            step_logits.append(lg)
    cached = torch.stack(step_logits, 1)                      # predictions for positions P..P+M-1
    ref = full[:, P - 1:P + M - 1]
    assert cache.seq_len == P + M - 1
    return cached.float(), ref.float()


@pytest.mark.gpu
@pytest.mark.parametrize("name", ["C0", "C3", "C12", "X-gc12", "X-E3"])
def test_cached_decode_fp32(name: str) -> None:
    cached, ref = _run(name, torch.float32)
    dmax = (cached - ref).abs().max().item()
    agree = (cached.argmax(-1) == ref.argmax(-1)).float().mean().item()
    print(f"VT-06 fp32 {name}: max|Δlogit|={dmax:.3e} argmax agreement={agree:.4f}")
    assert dmax <= 2e-2, dmax
    assert agree == 1.0, agree


@pytest.mark.gpu
@pytest.mark.parametrize("name", ["C0", "C3", "C12"])
def test_cached_decode_bf16_sanity(name: str) -> None:
    cached, ref = _run(name, torch.bfloat16)
    dmax = (cached - ref).abs().max().item()
    agree = (cached.argmax(-1) == ref.argmax(-1)).float().mean().item()
    print(f"VT-06 bf16 {name}: max|Δlogit|={dmax:.3e} argmax agreement={agree:.4f} (random init; informational)")
    assert torch.isfinite(cached).all()
    assert dmax <= 0.25 and agree >= 0.9

"""VT-07: branched query scoring vs separate forward passes (protocol §16, §9.1 query branching).

One prefill per context, a snapshot, then Q = 4 queries scored from it with ``score_continuation``.
Pass (fp32): |Δ logp_gold| < 1e-3 and identical EM against separate passes.

Two references are used:
(a) *separate canonical passes*: a fresh prefill of the context for every query, then the same
    scoring call — isolates state leakage between branched queries (must agree to < 1e-3);
(b) *single-pass forward* over context + query (the §9.2 re-scoring path, chunk kernel throughout) —
    differs only by chunk-vs-recurrent numerics; asserted with the VT-06 logit bound (|Δ logp| ≤ 2e-2
    per token sum) and EM identical.
Also asserts that scoring never mutates the cache (bitwise).
"""

from __future__ import annotations

import pytest
import torch

from hybridlc.models.build import build_model
from hybridlc.models.config import ModelConfig
from hybridlc.models.testing import cuda_device_or_skip

P = 160
PROMPT, ANSWER = 6, 4


def _setup(name: str):
    dev = cuda_device_or_skip()
    cfg = ModelConfig.for_scale("D384", name)
    model = build_model(cfg, seed_index=901).to(dev).eval()
    gen = torch.Generator().manual_seed(4)
    ctx = torch.randint(8, cfg.vocab_size, (2, P), generator=gen).to(dev)
    queries = [torch.randint(8, cfg.vocab_size, (2, PROMPT + ANSWER), generator=gen).to(dev) for _ in range(4)]
    # make answers "learnable-looking": EM_tf is then not trivially all-False for every model
    for q in queries:
        q[:, PROMPT:] = q[:, :ANSWER]
    return model, ctx, queries


def _summ(lp: torch.Tensor, am: torch.Tensor, q: torch.Tensor):
    ans = slice(PROMPT, PROMPT + ANSWER)
    return lp[:, ans].sum(-1), (am[:, ans] == q[:, ans]).all(-1)


@pytest.mark.gpu
@pytest.mark.parametrize("name", ["C0", "C3", "C12", "X-gc12"])
def test_branched_vs_separate(name: str) -> None:
    model, ctx, queries = _setup(name)
    _, cache = model.prefill(ctx)
    snap = cache.clone()
    branched = [_summ(*model.score_continuation(cache, q), q) for q in queries]
    # (bitwise) the cache was not mutated by four scoring calls
    for a, b in zip(cache.layers, snap.layers):
        for key in a:
            assert torch.equal(a[key], b[key]), key
    assert torch.equal(cache.last_logits, snap.last_logits) and cache.seq_len == snap.seq_len
    worst_a = worst_b = 0.0
    for q, (lp_b, em_b) in zip(queries, branched):
        _, fresh = model.prefill(ctx)
        lp_s, em_s = _summ(*model.score_continuation(fresh, q), q)
        worst_a = max(worst_a, (lp_b - lp_s).abs().max().item())
        assert torch.equal(em_b, em_s)
        full = model.full_logits(torch.cat([ctx, q], 1)).float()        # [B, P+M, V]
        lp_full = torch.log_softmax(full[:, P - 1:-1], -1).gather(-1, q[..., None])[..., 0]
        lp_f, em_f = _summ(lp_full, full[:, P - 1:-1].argmax(-1), q)
        worst_b = max(worst_b, (lp_b - lp_f).abs().max().item())
        assert torch.equal(em_b, em_f)
    print(f"VT-07 {name}: |Δlogp_gold| branched vs separate={worst_a:.2e}; vs single-pass={worst_b:.2e}")
    assert worst_a < 1e-3
    assert worst_b <= 2e-2

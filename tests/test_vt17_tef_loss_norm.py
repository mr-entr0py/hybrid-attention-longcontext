"""VT-17: the trainer honours the TEF's block-normalised D014 weights (verification round 3,
TEF-STEP-NORMALISED-ANSWER-WEIGHT-LENGTH-SKEW).

With ``answer_norm="block"`` every answer label of a 1,000-step block carries the same absolute
weight, so a step's weight mass varies (few answer tokens in steps holding 32K items). The trainer
composes the step objective as ``sum_m (W_m / W_obj) * loss_m`` with ``loss_m`` the micro-batch's
weighted mean; ``W_obj`` must be the batches' constant ``loss_norm`` (1.0), not the step's own
weight sum W, or the block normalisation is undone (it would become the per-step rule again).
"""
from __future__ import annotations

import pytest
import torch

from hybridlc.training.fakes import fake_chunked_ce
from hybridlc.training.trainer import _objective_norm, _step_weights


def _micro(n: int, w_ans: float, w_oth: float, n_ans: int, gen: torch.Generator, norm: float | None):
    labels = torch.randint(2, 50, (1, n), generator=gen)
    labels[0, -1] = 1  # ignored last position
    w = torch.full((1, n), w_oth)
    w[0, :n_ans] = w_ans
    b = dict(labels=labels, loss_weights=w)
    if norm is not None:
        b["loss_norm"] = norm
    return b


@pytest.mark.parametrize("n_ans", [2, 40])
def test_objective_is_absolute_weighted_sum_with_loss_norm(n_ans):
    g = torch.Generator().manual_seed(0)
    d, V, n = 8, 50, 64
    wt = torch.randn(V, d, generator=g)
    batches = [_micro(n, 0.25, 0.01, n_ans, g, 1.0) for _ in range(2)]
    hidden = [torch.randn(1, n, d, generator=g) for _ in batches]
    W, N = _step_weights(batches, 1)
    W_obj = _objective_norm(batches, W)
    assert W_obj == 1.0 and abs(W - 1.0) > 1e-3  # the step's own mass is not 1
    obj = sum(fake_chunked_ce(h, wt, b["labels"], loss_weights=b["loss_weights"])[0] * (b["_w"] / W_obj)
              for h, b in zip(hidden, batches))
    # reference: sum over valid labels of w * CE (absolute weights, no per-step renormalisation)
    ref = 0.0
    for h, b in zip(hidden, batches):
        logits = h[0].float() @ wt.T
        ce = torch.nn.functional.cross_entropy(logits, b["labels"][0].clamp_min(0), reduction="none")
        valid = b["labels"][0] != 1
        ref = ref + (ce * b["loss_weights"][0] * valid).sum()
    assert torch.allclose(obj, ref, rtol=1e-5), (float(obj), float(ref))
    # without loss_norm the trainer keeps the exact per-step weighted mean (divide by W)
    for b in batches:
        b.pop("loss_norm")
    assert _objective_norm(batches, W) == W


def test_inconsistent_loss_norm_refused():
    g = torch.Generator().manual_seed(1)
    a, b = _micro(16, 0.5, 0.1, 2, g, 1.0), _micro(16, 0.5, 0.1, 2, g, None)
    with pytest.raises(ValueError, match="loss_norm"):
        _objective_norm([a, b], 1.0)
    c = _micro(16, 0.5, 0.1, 2, g, 2.0)
    with pytest.raises(ValueError, match="loss_norm"):
        _objective_norm([a, c], 1.0)
    with pytest.raises(ValueError, match="> 0"):
        _objective_norm([_micro(16, 0.5, 0.1, 2, g, 0.0)], 1.0)

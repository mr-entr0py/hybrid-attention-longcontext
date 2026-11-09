"""Exactness of the gradient-accumulation shim (zoology_compat._patch_microbatch). GPU, < 1 GiB.

Run: pytest scripts/s1_zoology/test_s1_microbatch.py -q

One optimiser step of Zoology's *original* ``Trainer.train_epoch`` on a 256-row batch is
compared with one step of the shimmed version (micro-batches of 64), from identical
weights, with dropout disabled (the only intended difference is the dropout RNG stream).
The batch deliberately has *unequal* numbers of labelled positions per row, so the
label-count weighting is exercised (MQAR itself has equal counts within a segment).
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


class _NullLogger:
    def __init__(self) -> None:
        self.records: list[dict] = []

    def log(self, m: dict) -> None:
        self.records.append(m)


def _model_and_batch(family: str):
    import zoology_compat as zc

    zc.claim_gpu()
    zc.setup_zoology(micro_batch=None)  # import zoology without installing the shim globally
    from s1_configs import build_runs
    from zoology.model import LanguageModel

    run = next(r for r in build_runs() if r.family == family)
    mcfg = run.config.model.model_copy(deep=True)
    mcfg.embed_dropout = 0.0
    seq = mcfg.sequence_mixer.kwargs["configs"]
    if "dropout" in seq[1]["kwargs"]:
        seq[1]["kwargs"]["dropout"] = 0.0
    torch.manual_seed(0)
    model = LanguageModel(mcfg).cuda()
    g = torch.Generator().manual_seed(1)
    B, T = 256, 64
    x = torch.randint(0, mcfg.vocab_size, (B, T), generator=g)
    y = torch.randint(0, mcfg.vocab_size, (B, T), generator=g)
    # Unequal label counts per row: row i keeps the last (4 + i % 13) positions.
    keep = torch.arange(T)[None, :] >= (T - (4 + torch.arange(B)[:, None] % 13))
    y = torch.where(keep, y, torch.full_like(y, -100))
    return model, [(x, y, [{}] * B)]


def _one_step(model, loader, train_epoch):
    import zoology.train as ztrain

    tr = ztrain.Trainer(model=model, train_dataloader=loader, test_dataloader=loader, max_epochs=1,
                        learning_rate=1e-2, weight_decay=0.1, logger=_NullLogger(), device="cuda")
    tr.loss_fn = torch.nn.CrossEntropyLoss()
    tr.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.1)
    # capture gradients just before the optimiser step
    grads: dict[str, torch.Tensor] = {}
    orig_step = tr.optimizer.step

    def step(*a, **k):
        for n, p in model.named_parameters():
            if p.grad is not None:
                grads[n] = p.grad.detach().clone()
        return orig_step(*a, **k)

    tr.optimizer.step = step
    train_epoch(tr, 0)
    return grads, tr.logger.records[-1]["train/loss"]


@pytest.mark.parametrize("family", ["attention", "gated_delta_net"])
def test_microbatch_gradient_equals_full_batch(family):
    import zoology.train as ztrain
    import zoology_compat as zc

    model, loader = _model_and_batch(family)
    original = ztrain.Trainer.train_epoch
    assert not getattr(original, "__zoology_compat__", False)

    m_full = copy.deepcopy(model)
    g_full, loss_full = _one_step(m_full, loader, original)

    zc._patch_microbatch(ztrain, 64)
    try:
        shim = ztrain.Trainer.train_epoch
        assert getattr(shim, "__zoology_compat__", False)
        m_mb = copy.deepcopy(model)
        g_mb, loss_mb = _one_step(m_mb, loader, shim)
    finally:
        ztrain.Trainer.train_epoch = original

    # Tolerances. Attention is fp32 end to end, so only summation order differs -> 1e-4.
    # Zoology's GDN casts q/k/v/beta/g to bf16 before the fla kernel; gradients of the tiny
    # per-head parameters (A_log, dt_bias, b/a_proj) are reductions over bf16 kernel outputs,
    # so a different reduction order moves them by up to ~1e-3 relative, still well below
    # bf16 eps (7.8e-3). We compare gradients (the shim's contract), not post-AdamW weights:
    # Adam's first step is ~lr*sign(g) and turns float noise in near-zero gradient entries
    # (unused embedding rows) into lr-sized parameter differences.
    tol = 1e-4 if family == "attention" else 2e-3
    assert abs(loss_full - loss_mb) < 1e-5 * max(1.0, abs(loss_full)), (loss_full, loss_mb)
    assert g_full.keys() == g_mb.keys() and len(g_full) > 0
    worst = max(((g_full[n] - g_mb[n]).norm() / g_full[n].norm().clamp_min(1e-12)).item() for n in g_full)
    print(f"[{family}] loss full={loss_full:.6f} micro={loss_mb:.6f}; max rel grad diff {worst:.2e} (tol {tol:g})")
    assert worst < tol

"""Logits → loss correctness (supporting VT-09 and §7.1): ``chunked_ce``.

* equals a dense fp32 reference (CE over non-ignored tokens, pad id 1 and −100 ignored);
* z-loss = 1e-4 · mean(logsumexp²) over loss tokens; n_tokens counts them;
* ``loss_weights`` gives Σ w·CE / Σ w (TEF ½/½ weighting);
* chunk size does not change the result; gradients equal the dense reference (recompute is exact);
* identical to the training-infra reference contract ``fakes.fake_chunked_ce``.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from hybridlc.models.loss import chunked_ce
from hybridlc.training.fakes import fake_chunked_ce


def _data(N=1000, d=64, V=300, seed=0, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    h = torch.randn(1, N, d, generator=g, dtype=dtype)
    w = torch.randn(V, d, generator=g, dtype=dtype) * 0.3
    y = torch.randint(0, V, (1, N), generator=g)
    y[0, ::7] = 1          # pad / ignore_index
    y[0, 3::11] = -100     # also ignored
    return h, w, y


def _dense_ref(h, w, y, coef=1e-4, lw=None):
    logits = (h.reshape(-1, h.shape[-1]) @ w.t()).double()
    yy = y.reshape(-1)
    valid = (yy != 1) & (yy >= 0)
    ce = F.cross_entropy(logits[valid], yy[valid], reduction="none")
    lse = torch.logsumexp(logits[valid], -1)
    if lw is None:
        loss = ce.mean()
    else:
        ww = lw.reshape(-1)[valid].double()
        loss = (ce * ww).sum() / ww.sum()
    return loss, coef * lse.square().mean(), valid.sum()


@pytest.mark.parametrize("chunk", [1000, 256, 77])
def test_chunked_ce_matches_dense(chunk) -> None:
    h, w, y = _data()
    loss, z, n = chunked_ce(h, w, y, chunk=chunk)
    rl, rz, rn = _dense_ref(h, w, y)
    assert torch.allclose(loss.double(), rl, rtol=1e-6) and torch.allclose(z.double(), rz, rtol=1e-6)
    assert int(n) == int(rn)


def test_loss_weights_and_fake_contract() -> None:
    h, w, y = _data(seed=1)
    lw = torch.rand(y.shape, generator=torch.Generator().manual_seed(2), dtype=torch.float64)
    loss, z, n = chunked_ce(h, w, y, chunk=128, loss_weights=lw)
    rl, rz, rn = _dense_ref(h, w, y, lw=lw)
    assert torch.allclose(loss.double(), rl, rtol=1e-6)
    y_fake = y.clone()
    y_fake[y_fake < 0] = 1  # the fake ignores only ignore_index; map −100 → pad for the comparison
    fl, fz, fn = fake_chunked_ce(h.float(), w.float(), y_fake, chunk=128, loss_weights=lw.float())
    assert torch.allclose(loss.float(), fl, rtol=1e-4) and torch.allclose(z.float(), fz, rtol=1e-4)
    assert int(n) == int(fn)


def test_gradients_with_recompute_equal_dense() -> None:
    h, w, y = _data(N=600, seed=3)
    h1, w1 = h.clone().requires_grad_(True), w.clone().requires_grad_(True)
    loss, z, _ = chunked_ce(h1, w1, y, chunk=100)
    (loss + z).backward()
    h2, w2 = h.clone().requires_grad_(True), w.clone().requires_grad_(True)
    rl, rz, _ = _dense_ref(h2, w2, y)
    (rl + rz).backward()
    # logits are computed in fp32 by design (even for fp64 inputs): fp32-level agreement
    for a, b in ((h1.grad, h2.grad), (w1.grad, w2.grad)):
        assert (a - b).abs().max() <= 1e-5 * b.abs().max()


def test_bf16_autocast_cpu_fp32_logits() -> None:
    """Under autocast the GEMM runs in bf16 but the CE/z-loss terms are fp32."""
    h, w, y = _data(seed=4, dtype=torch.float32)
    with torch.autocast("cpu", torch.bfloat16):
        loss, z, n = chunked_ce(h, w, y, chunk=200)
    assert loss.dtype == torch.float32 and z.dtype == torch.float32
    rl, rz, _ = _dense_ref(h.double(), w.double(), y)
    assert abs(loss.item() - rl.item()) < 2e-2

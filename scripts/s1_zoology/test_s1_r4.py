"""R4 tooling checks: production-F_g wrappers, varlen_attn usage in the bf16 arm, the R4 run list and
the collapse-abort rule (plan §2.0; prereg experiments/repro/R4/prereg.yaml reading C1).

    python -m pytest scripts/s1_zoology/test_s1_r4.py -q
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from zoology_compat import claim_gpu, collapse_abort_due, setup_zoology  # noqa: E402

setup_zoology(micro_batch=None)
import our_mixers as OM  # noqa: E402
from test_s1_our_mixers import _lm, _mixer, _swap, _zoology_cfg  # noqa: E402

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def test_collapse_abort_rule() -> None:
    lo = 0.01
    assert not collapse_abort_due([0.9, lo, lo])                 # epochs 0-2 never count
    assert not collapse_abort_due([0.9, lo, lo, lo, lo])         # epochs 3,4 only: two
    assert collapse_abort_due([0.9, lo, lo, lo, lo, lo])         # epochs 3,4,5 -> abort after epoch 5
    assert not collapse_abort_due([0.9, 0.9, 0.9, lo, lo, 0.06])
    assert collapse_abort_due([0.9] * 10 + [lo] * 3)
    assert not collapse_abort_due([0.9] * 10 + [lo, 0.05, lo])   # 0.05 is not < 0.05


def test_r4_run_list() -> None:
    ids = [c.run_id for c in OM.r4_runs()]
    assert ids == ["our_fg-d64-lr1.0e-02", "our_fg-d128-lr1.0e-03", "our_gdn-d128-lr1.0e-03", "our_gdn-d256-lr1.0e-03",
                   "our_fg_prod-d128-lr1.0e-03", "our_fg_prod_bf16-d128-lr1.0e-03",
                   "our_gdn_configA-d128-lr1.0e-03", "our_gdn_configA-d256-lr1.0e-03"]
    for c in OM.r4_runs():
        assert c.seed == 123 and c.data.seed == 123 and c.max_epochs == 32
        assert c.early_stopping_metric == "valid/accuracy" and c.early_stopping_threshold == 0.99


@cuda
def test_prod_fg_geometry_and_init() -> None:
    claim_gpu()
    zcfg = _zoology_cfg("attention", 128).model
    plain = _mixer(_lm(_swap(zcfg, "our_mixers.OurFgZoology")))
    prod = _mixer(_lm(_swap(zcfg, "our_mixers.OurFgProdZoology")))
    inner = prod.inner
    assert (inner.n_heads, inner.head_dim) == (2, 64)
    assert inner.use_qk_norm and inner.use_rope and inner.use_out_gate and inner.rope_theta == 5e5
    assert prod.out_proj.weight is inner.o_proj.weight
    assert abs(float(inner.o_proj.weight.std()) - 0.01) < 0.25 * 0.01        # Zoology rescale 0.02/sqrt(4)
    assert torch.all(inner.q_norm == 1) and torch.all(inner.k_norm == 1)
    extra = {k for k, _ in prod.named_parameters()} - {k for k, _ in plain.named_parameters()}
    assert extra == {"inner.q_norm", "inner.k_norm", "inner.g_proj.weight"}
    assert not prod.autocast_bf16 and _mixer(_lm(_swap(zcfg, "our_mixers.OurFgProdBf16Zoology"))).autocast_bf16


@cuda
@pytest.mark.parametrize("path,expect_varlen", [("our_mixers.OurFgProdZoology", False),
                                                ("our_mixers.OurFgProdBf16Zoology", True)])
def test_prod_fg_train_step_and_kernel_path(path: str, expect_varlen: bool) -> None:
    claim_gpu()
    import hybridlc.models.attention as att
    from zoology.data.multiquery_ar import MQARConfig

    calls = []
    orig = att._varlen_attn
    if orig is None:
        pytest.skip("varlen_attn unavailable")
    att._varlen_attn = lambda q, *a, **k: (calls.append(q.dtype), orig(q, *a, **k))[1]
    try:
        lm = _lm(_swap(_zoology_cfg("attention", 128).model, path)).cuda()
        seg = MQARConfig(vocab_size=8192, input_seq_len=256, num_examples=8, num_kv_pairs=16, seed=0).build(seed=0)
        x, y = seg.inputs.cuda(), seg.labels.cuda()
        opt = torch.optim.AdamW(lm.parameters(), lr=1e-3, weight_decay=0.1)
        losses = []
        for _ in range(3):
            logits = lm(x)
            assert logits.dtype == torch.float32
            loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), y.flatten(), ignore_index=-100)
            opt.zero_grad()
            loss.backward()
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in lm.parameters())
            assert all(p.dtype == torch.float32 for p in lm.parameters())
            opt.step()
            losses.append(float(loss))
        assert all(math.isfinite(v) for v in losses) and losses[-1] < losses[0]
    finally:
        att._varlen_attn = orig
    if expect_varlen:
        assert calls and all(d == torch.bfloat16 for d in calls)
    else:
        assert not calls

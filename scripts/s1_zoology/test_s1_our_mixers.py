"""S1(b) wrapper checks: our mixers construct, init and train-step inside Zoology's LanguageModel.

    python -m pytest scripts/s1_zoology/test_s1_our_mixers.py -q

GPU tests claim 0.44 of the card and stay well below 1 GiB.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from zoology_compat import claim_gpu, setup_zoology  # noqa: E402

setup_zoology(micro_batch=None)
import our_mixers as OM  # noqa: E402
from s1_configs import build_runs  # noqa: E402

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA (fla kernels)")


def _zoology_cfg(family: str, d: int):
    return next(r.config for r in build_runs() if r.family == family and r.d_model == d).model_copy(deep=True)


def _lm(model_cfg):
    from zoology.model import LanguageModel

    torch.manual_seed(0)
    return LanguageModel(model_cfg)


def _swap(model_cfg, path: str):
    model_cfg = model_cfg.model_copy(deep=True)
    model_cfg.sequence_mixer.kwargs["configs"][1] = {"name": path, "kwargs": {}}
    return model_cfg


def _mixer(lm):
    return lm.backbone.layers[1].sequence_mixer.mixer  # Hybrid at layer 1 -> configs[1]


def _n(m) -> int:
    return sum(p.numel() for p in m.parameters())


@cuda
@pytest.mark.parametrize("d", [64, 128])
def test_fg_matches_zoology_mha_geometry(d: int) -> None:
    claim_gpu()
    zcfg = _zoology_cfg("attention", 64 if d == 64 else 128).model
    ours = _lm(_swap(zcfg, "our_mixers.OurFgZoology"))
    ref = _lm(zcfg)
    m_ours, m_ref = _mixer(ours), _mixer(ref)
    inner = m_ours.inner
    assert inner.n_heads == 2 and inner.head_dim == d // 2
    assert not inner.use_qk_norm and not inner.use_rope and not inner.use_out_gate
    # Zoology MHA = 4d² weights + 4d biases; ours has no biases (documented residual difference).
    assert _n(m_ref) - _n(m_ours) == 4 * d
    # Zoology's name-based residual rescale reached our output projection via the alias.
    std_target = 0.02 / math.sqrt(2 * zcfg.n_layers)
    assert m_ours.out_proj.weight is inner.o_proj.weight
    assert abs(float(inner.o_proj.weight.std()) - std_target) < 0.25 * std_target
    assert abs(float(inner.q_proj.weight.std()) - 0.02) < 0.25 * 0.02
    assert m_ours.state_size(sequence_length=1024) == m_ref.state_size(sequence_length=1024)


@cuda
@pytest.mark.parametrize("d", [128, 256])
def test_gdn_matches_zoology_gdn_geometry(d: int) -> None:
    claim_gpu()
    zcfg = _zoology_cfg("gated_delta_net", d).model
    ours, ref = _lm(_swap(zcfg, "our_mixers.OurGdnZoology")), _lm(zcfg)
    m_ours, m_ref = _mixer(ours), _mixer(ref)
    inner = m_ours.inner
    assert (inner.num_heads, inner.head_k_dim, inner.head_v_dim) == (m_ref.num_heads, m_ref.head_k_dim, m_ref.head_v_dim)
    assert inner.use_gate is False and inner.use_short_conv and inner.conv_size == 4
    # Identical parameter shapes except Zoology's unused D (num_heads scalars).
    ours_shapes = {k.removeprefix("inner."): tuple(v.shape) for k, v in m_ours.named_parameters()}
    ref_shapes = {k: tuple(v.shape) for k, v in m_ref.named_parameters() if k != "D"}
    assert ours_shapes == ref_shapes
    assert m_ours.state_size(sequence_length=1024) == m_ref.state_size(sequence_length=1024)


@cuda
@pytest.mark.parametrize("path,family,d", [("our_mixers.OurFgZoology", "attention", 64),
                                           ("our_mixers.OurGdnZoology", "gated_delta_net", 128),
                                           ("our_mixers.OurGdnConfigA", "gated_delta_net", 256)])
def test_train_step_in_zoology_model(path: str, family: str, d: int) -> None:
    """fp32 forward/backward/AdamW step on an MQAR batch (T=256) through Zoology's LanguageModel."""
    claim_gpu()
    from zoology.data.multiquery_ar import MQARConfig

    lm = _lm(_swap(_zoology_cfg(family, d).model, path)).cuda()
    seg = MQARConfig(vocab_size=8192, input_seq_len=256, num_examples=8, num_kv_pairs=16, seed=0).build(seed=0)
    x, y = seg.inputs.cuda(), seg.labels.cuda()
    opt = torch.optim.AdamW(lm.parameters(), lr=1e-3, weight_decay=0.1)
    losses = []
    for _ in range(3):
        logits = lm(x)
        assert logits.dtype == torch.float32 and logits.shape == (*x.shape, 8192)
        loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), y.flatten(), ignore_index=-100)
        opt.zero_grad()
        loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in lm.parameters())
        opt.step()
        losses.append(float(loss))
    assert all(math.isfinite(v) for v in losses) and losses[-1] < losses[0]


def test_part_b_run_list() -> None:
    best = {("attention", 64): 1e-3, ("attention", 128): 1e-3,
            ("gated_delta_net", 128): 1e-3, ("gated_delta_net", 256): 10 ** -2.5}
    cfgs = OM.part_b_runs(best)
    assert [c.run_id for c in cfgs] == ["our_fg-d64-lr1.0e-03", "our_fg-d128-lr1.0e-03",
                                         "our_gdn-d128-lr1.0e-03", "our_gdn-d256-lr3.2e-03",
                                         "our_gdn_configA-d128-lr1.0e-03", "our_gdn_configA-d256-lr3.2e-03"]
    for c in cfgs:
        seq = c.model.sequence_mixer.kwargs["configs"]
        assert seq[0]["name"].endswith("BaseConv") and seq[1]["name"].startswith("our_mixers.")
        assert c.data.cache_dir.endswith("data/zoology") and c.max_epochs == 32

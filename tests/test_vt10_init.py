"""VT-10: initialisation (protocol §7.1, §8.1, §16).

* ``A_log`` and ``dt_bias`` (and o_norm) survive the global init: they equal what fla's own
  constructor draws under the mixer's stream-derived seed;
* short-conv kernels (GDN/GLA ShortConvolution, F_gc conv) follow fla's *model* init N(0, 0.02)
  (``GatedDeltaNetPreTrainedModel._init_weights``), not the constructor's U(-0.5, 0.5)
  (verification round 3, GDN-CONV-INIT-NOT-FLA-MODEL-INIT);
* output-projection std ≈ 0.02/√24, other linears ≈ 0.02, gains = 1;
* shared modules bit-identical across k for a given seed (SHA-256 of embedding/MLP/norm tensors in
  C0 vs C12 and the other table configs), different across seeds;
* same mixer type at the same layer is bit-identical across patterns;
* the caller's global RNG is untouched and construction is deterministic.
"""

from __future__ import annotations

import math

import pytest
import torch

from hybridlc.models.build import build_model
from hybridlc.models.config import ModelConfig
from hybridlc.models.mixers import build_mixer
from hybridlc.utils.provenance import sha256_tensor
from hybridlc.utils.seeds import MIXER_TYPE_ID, STREAMS, scoped_torch_seed, torch_generator

SCALE = "D384"
OUT_STD = 0.02 / math.sqrt(24)


def _shared_hashes(model) -> dict[str, str]:
    return {n: sha256_tensor(p) for n, p in model.named_parameters() if ".mixer." not in n}


@pytest.mark.parametrize("name", ["C0", "C12"])
def test_fla_tensors_keep_fla_init(name: str) -> None:
    s = 3
    cfg = ModelConfig.for_scale(SCALE, "C3" if name == "C12" else name)
    model = build_model(cfg, s)
    for l, ch in enumerate(cfg.pattern):
        if ch not in "-ENG":
            continue
        gen = torch_generator(s, STREAMS["init_mixer"], l, MIXER_TYPE_ID[ch])
        ctor_seed = int(torch.randint(0, 2**62, (1,), generator=gen))
        with scoped_torch_seed(ctor_seed):
            ref = build_mixer(ch, cfg, l)
        mx = model.layers[l].mixer
        assert torch.equal(mx.A_log, ref.A_log) and torch.equal(mx.dt_bias, ref.dt_bias)
        assert mx.A_log._no_weight_decay and mx.dt_bias._no_weight_decay
        for conv in ("q_conv1d", "k_conv1d", "v_conv1d"):
            w = getattr(mx, conv).weight
            # fla model init N(0, 0.02) (round 3), not the constructor's U(-1/2, 1/2) (std 0.289)
            assert not torch.equal(w, getattr(ref, conv).weight)
            assert abs(w.std().item() / 0.02 - 1) < 0.1 and w.abs().max().item() < 0.15, (conv, w.std())
        assert torch.equal(mx.o_norm.weight, torch.ones_like(mx.o_norm.weight))
        a = mx.A_log.exp()
        assert (a > 0).all() and (a < 16).all()
        dt = torch.nn.functional.softplus(mx.dt_bias)
        assert (dt >= 1e-4 - 1e-9).all() and (dt <= 0.1 + 1e-6).all()
        # projections were re-initialised (not fla's kaiming default)
        assert not torch.equal(mx.q_proj.weight, ref.q_proj.weight)


def test_projection_stds() -> None:
    cfg = ModelConfig.for_scale(SCALE, "C3")
    model = build_model(cfg, 0)
    for name, p in model.named_parameters():
        if p.ndim != 2:
            continue
        sd = p.std().item()
        if name.endswith("o_proj.weight") or name.endswith("down_proj.weight"):
            assert abs(sd / OUT_STD - 1) < 0.03, (name, sd)
        elif name == "embed.weight" or name.endswith("_proj.weight"):
            assert abs(sd / 0.02 - 1) < 0.03 or p.numel() < 5000, (name, sd)
        assert abs(p.mean().item()) < 5 * sd / math.sqrt(p.numel()) + 1e-6, name  # 5σ: ~100 tensors
    for name, p in model.named_parameters():
        if name.endswith("norm1.weight") or name.endswith("norm2.weight") or name in ("norm_f.weight",) \
                or name.endswith("q_norm") or name.endswith("k_norm"):
            assert torch.equal(p, torch.ones_like(p)), name
    # headwise gate W_g ~ N(0, 0.02) → σ(x W_g) ≈ 0.5 at init
    g = model.layers[2].mixer.g_proj.weight
    assert abs(g.std().item() / 0.02 - 1) < 0.05


def test_fgc_conv_and_ssmax_init() -> None:
    cfg = ModelConfig.for_scale(SCALE, "X-gc12", ssmax=True)
    model = build_model(cfg, 0)
    mx = model.layers[0].mixer
    w = mx.conv.weight
    assert abs(w.std().item() / 0.02 - 1) < 0.1 and w.abs().max().item() < 0.15  # N(0, 0.02), as GDN
    assert torch.allclose(mx.ssmax_s, torch.full_like(mx.ssmax_s, 1 / math.log(4096)))
    assert mx.ssmax_s.ndim == 1 and mx.q_norm.ndim == 1  # no weight decay by the ndim rule


def test_shared_modules_bit_identical_across_patterns() -> None:
    names = ["C0", "C1", "C3", "C12", "X-gc12", "X-E0", "O-gla3", "P3-late"]
    hashes = {n: _shared_hashes(build_model(ModelConfig.for_scale(SCALE, n), 5)) for n in names}
    ref = hashes["C0"]
    assert len(ref) == 1 + 12 * 5 + 1  # embed, 12×(2 norms + 3 MLP), final norm
    for n in names[1:]:
        assert hashes[n] == ref, n
    other = _shared_hashes(build_model(ModelConfig.for_scale(SCALE, "C0"), 6))
    assert other["embed.weight"] != ref["embed.weight"]
    assert other["layers.0.mlp.up_proj.weight"] != ref["layers.0.mlp.up_proj.weight"]


def test_mixers_bit_identical_same_type_same_layer() -> None:
    m1 = build_model(ModelConfig.for_scale(SCALE, "C1"), 2)
    m3 = build_model(ModelConfig.for_scale(SCALE, "C3"), 2)
    m12 = build_model(ModelConfig.for_scale(SCALE, "C12"), 2)
    m0 = build_model(ModelConfig.for_scale(SCALE, "C0"), 2)

    def h(m, l):
        return {n: sha256_tensor(p) for n, p in m.layers[l].mixer.named_parameters()}

    assert h(m1, 6) == h(m3, 6) == h(m12, 6)        # F_g at layer 6
    assert h(m0, 0) == h(m3, 0) == h(m1, 0)         # GDN-A at layer 0
    assert h(m0, 1) != h(m0, 0)                     # different layers differ


def test_global_rng_untouched_and_deterministic() -> None:
    torch.manual_seed(1234)
    before = torch.random.get_rng_state()
    a = build_model(ModelConfig.for_scale(SCALE, "C3"), 0)
    after = torch.random.get_rng_state()
    assert torch.equal(before, after)
    b = build_model(ModelConfig.for_scale(SCALE, "C3"), 0)
    sa, sb = a.state_dict(), b.state_dict()
    assert sa.keys() == sb.keys()
    assert all(torch.equal(sa[k], sb[k]) for k in sa)


def test_tied_embedding() -> None:
    m = build_model(ModelConfig.for_scale(SCALE, "C0"), 0)
    assert m.lm_head_weight is m.embed.weight
    assert not any(n.endswith("bias") and not n.endswith("dt_bias") for n, _ in m.named_parameters())  # no biases


def test_conv_init_matches_fla_model_init() -> None:
    """GDN-CONV-INIT-NOT-FLA-MODEL-INIT (verification round 3): the repo's GDN short-conv kernels have
    the distribution of fla's own model init (GatedDeltaNetForCausalLM, initializer_range 0.02), and
    the linear draws did not move when the convs were added (they are drawn after every linear)."""
    pytest.importorskip("fla")
    from fla.models import GatedDeltaNetConfig, GatedDeltaNetForCausalLM

    cfg = ModelConfig.for_scale("D512", "C3")
    model = build_model(cfg, 0)
    torch.manual_seed(0)
    ref = GatedDeltaNetForCausalLM(GatedDeltaNetConfig(
        hidden_size=512, num_heads=4, head_dim=64, expand_v=2, num_hidden_layers=1, vocab_size=1024,
        intermediate_size=1536, use_short_conv=True, conv_size=4, initializer_range=0.02))
    ra = ref.model.layers[0].attn
    mx = model.layers[0].mixer
    for conv in ("q_conv1d", "k_conv1d", "v_conv1d"):
        ours, theirs = getattr(mx, conv).weight, getattr(ra, conv).weight
        assert ours.shape == theirs.shape
        so, st = ours.std().item(), theirs.std().item()
        assert abs(so - st) / st < 0.15 and abs(so / 0.02 - 1) < 0.1, (conv, so, st)
    # every conv of every mixer kind is redrawn (F_gc too); GLA ShortConvolution too
    for name in ("X-gc12", "O-gla3"):
        m = build_model(ModelConfig.for_scale(SCALE, name), 1)
        convs = [p for n, p in m.named_parameters() if "conv" in n and n.endswith("weight")]
        assert convs and all(abs(p.std().item() / 0.02 - 1) < 0.15 for p in convs), name

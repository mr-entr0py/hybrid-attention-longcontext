"""VT-10: initialisation (protocol §7.1, §8.1, §16).

* ``A_log`` and ``dt_bias`` (and the GDN conv kernels / o_norm) survive the global init: they equal
  what fla's own constructor draws under the mixer's stream-derived seed;
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
            assert torch.equal(getattr(mx, conv).weight, getattr(ref, conv).weight)
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
    assert w.abs().max().item() <= 0.5 and w.std().item() > 0.2  # U(−1/√4, 1/√4): std 0.289
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

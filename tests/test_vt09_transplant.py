"""VT-09: weight-transplant logits (protocol §16, I-20).

(1) Our C12 with the headwise gate bypassed (test flag) and per-head QK gains tied ↔ transformers
    ``Qwen3ForCausalLM`` (hidden d, heads d/64, head_dim 64, kv heads = heads, intermediate d_ff,
    rope θ = 5e5, rms eps 1e-5, tied embeddings, no attention bias).
(2) Our C0 ↔ fla ``GatedDeltaNetForCausalLM`` (hidden d, 12 layers, heads d/128, head_dim 64,
    expand_v 2, gate, short conv 4, intermediate d_ff, eps 1e-5, tied, fuse_cross_entropy=False).
(3) Headwise gate unit test: the gated output equals the manual σ(x W_g)·o.
Pass: fp32 logits within 1e-3 relative (max|Δ| / max|ref|).
"""

from __future__ import annotations

import pytest
import torch

from hybridlc.models.build import build_model
from hybridlc.models.config import ModelConfig
from hybridlc.models.testing import cuda_device_or_skip, rel_max

SCALE = "D384"


def _qwen3(cfg: ModelConfig):
    from transformers import Qwen3Config, Qwen3ForCausalLM

    qc = Qwen3Config(vocab_size=cfg.vocab_size, hidden_size=cfg.d_model, intermediate_size=cfg.d_ff,
                     num_hidden_layers=cfg.n_layers, num_attention_heads=cfg.d_model // 64,
                     num_key_value_heads=cfg.d_model // 64, head_dim=64,
                     rope_parameters={"rope_type": "default", "rope_theta": 5e5}, rms_norm_eps=1e-5,
                     tie_word_embeddings=True, attention_bias=False, use_sliding_window=False,
                     max_position_embeddings=32768)
    qc._attn_implementation = "eager"
    torch.manual_seed(0)
    m = Qwen3ForCausalLM(qc).float().eval()
    # randomise the (otherwise all-ones) norm gains so the transplant actually tests them
    with torch.no_grad():
        for n, p in m.named_parameters():
            if p.ndim == 1:
                p.uniform_(0.6, 1.4)
    return m


def _transplant_qwen(ours, q) -> None:
    H = ours.cfg.d_model // 64
    sd = q.state_dict()
    with torch.no_grad():
        ours.embed.weight.copy_(sd["model.embed_tokens.weight"])
        ours.norm_f.weight.copy_(sd["model.norm.weight"])
        for l, blk in enumerate(ours.layers):
            p = f"model.layers.{l}."
            blk.norm1.weight.copy_(sd[p + "input_layernorm.weight"])
            blk.norm2.weight.copy_(sd[p + "post_attention_layernorm.weight"])
            for n in ("gate_proj", "up_proj", "down_proj"):
                getattr(blk.mlp, n).weight.copy_(sd[p + f"mlp.{n}.weight"])
            mx = blk.mixer
            for n in ("q_proj", "k_proj", "v_proj", "o_proj"):
                getattr(mx, n).weight.copy_(sd[p + f"self_attn.{n}.weight"])
            mx.q_norm.copy_(sd[p + "self_attn.q_norm.weight"].repeat(H))   # tie per-head gains
            mx.k_norm.copy_(sd[p + "self_attn.k_norm.weight"].repeat(H))
    ours.set_gate_bypass(True)


def test_c12_vs_qwen3_cpu_fp32() -> None:
    cfg = ModelConfig.for_scale(SCALE, "C12")
    ours = build_model(cfg, seed_index=0).eval()
    q = _qwen3(cfg)
    _transplant_qwen(ours, q)
    ids = torch.randint(8, cfg.vocab_size, (2, 96), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        ref = q(input_ids=ids).logits.float()
        mine = ours.full_logits(ids)
        e = rel_max(mine, ref)
        # cached path too: prefill + teacher-forced continuation
        last, cache = ours.prefill(ids[:, :64])
        rest, _ = ours.extend(ids[:, 64:-1], cache)
        e_cache = rel_max(torch.cat([last[:, None], rest], 1), ref[:, 63:-1])
        # packed varlen path: two documents in one row == each document alone (RoPE reset per doc)
        a, b = ids[0, :40], ids[1, :56]
        packed = torch.cat([a, b])[None]
        cu = torch.tensor([0, 40, 96], dtype=torch.int32)
        pos = torch.cat([torch.arange(40), torch.arange(56)])[None]
        hp = ours.logits(ours(packed, cu, pos))[0]
        ref_a = q(input_ids=a[None]).logits[0]
        ref_b = q(input_ids=b[None]).logits[0]
        e_packed = rel_max(hp, torch.cat([ref_a, ref_b]))
    print(f"VT-09 C12↔Qwen3: dense {e:.2e}, cached {e_cache:.2e}, packed {e_packed:.2e}")
    assert e <= 1e-3 and e_cache <= 1e-3 and e_packed <= 1e-3


@pytest.mark.gpu
def test_c12_vs_qwen3_gpu_bf16_varlen_close() -> None:
    """The production bf16 varlen_attn path against the fp32 Qwen3 reference (loose, informational)."""
    dev = cuda_device_or_skip()
    cfg = ModelConfig.for_scale(SCALE, "C12")
    ours = build_model(cfg, seed_index=0).eval()
    q = _qwen3(cfg)
    _transplant_qwen(ours, q)
    ours, q = ours.to(dev), q.to(dev)
    ids = torch.randint(8, cfg.vocab_size, (1, 256), generator=torch.Generator().manual_seed(2)).to(dev)
    cu = torch.tensor([0, 256], dtype=torch.int32, device=dev)
    with torch.no_grad():
        ref = q(input_ids=ids).logits.float()
        with torch.autocast("cuda", torch.bfloat16):
            mine = ours.logits(ours(ids, cu, torch.arange(256, device=dev)[None]))
    e = rel_max(mine, ref)
    print(f"VT-09 C12 bf16 varlen vs Qwen3 fp32: rel max {e:.2e}")
    assert e <= 5e-2


def _fla_gdn_lm(cfg: ModelConfig):
    from fla.models import GatedDeltaNetConfig, GatedDeltaNetForCausalLM

    fc = GatedDeltaNetConfig(hidden_size=cfg.d_model, num_hidden_layers=cfg.n_layers, num_heads=cfg.d_model // 128,
                             head_dim=64, expand_v=2, use_gate=True, use_short_conv=True, conv_size=4,
                             intermediate_size=cfg.d_ff, norm_eps=1e-5, tie_word_embeddings=True,
                             fuse_cross_entropy=False, vocab_size=cfg.vocab_size, hidden_ratio=None)
    torch.manual_seed(0)
    m = GatedDeltaNetForCausalLM(fc).float().eval()
    with torch.no_grad():
        for n, p in m.named_parameters():
            if n.endswith("norm.weight"):
                p.uniform_(0.6, 1.4)
    return m


def _transplant_fla(ours, f) -> None:
    sd = f.state_dict()
    with torch.no_grad():
        ours.embed.weight.copy_(sd["model.embeddings.weight"])
        ours.norm_f.weight.copy_(sd["model.norm.weight"])
        for l, blk in enumerate(ours.layers):
            p = f"model.layers.{l}."
            blk.norm1.weight.copy_(sd[p + "attn_norm.weight"])
            blk.norm2.weight.copy_(sd[p + "mlp_norm.weight"])
            for n in ("gate_proj", "up_proj", "down_proj"):
                getattr(blk.mlp, n).weight.copy_(sd[p + f"mlp.{n}.weight"])
            mixer_sd = {k[len(p + "attn."):]: v for k, v in sd.items() if k.startswith(p + "attn.")}
            missing, unexpected = blk.mixer.load_state_dict(mixer_sd, strict=True)
            assert not missing and not unexpected


@pytest.mark.gpu
def test_c0_vs_fla_gated_deltanet_lm_fp32() -> None:
    dev = cuda_device_or_skip()
    cfg = ModelConfig.for_scale(SCALE, "C0")
    ours = build_model(cfg, seed_index=0).eval()
    f = _fla_gdn_lm(cfg)
    _transplant_fla(ours, f)
    ours, f = ours.to(dev), f.to(dev)
    ids = torch.randint(8, cfg.vocab_size, (2, 200), generator=torch.Generator().manual_seed(3)).to(dev)
    with torch.no_grad():
        ref = f(input_ids=ids).logits.float()     # T = 200 > 64: fla uses its chunk mode
        mine = ours.full_logits(ids)
        e = rel_max(mine, ref)
        last, cache = ours.prefill(ids)
        e_last = rel_max(last, ref[:, -1])
        # packed: two documents in one row vs each alone through fla
        a, b = ids[0, :90], ids[1, :110]
        cu = torch.tensor([0, 90, 200], dtype=torch.int32, device=dev)
        pos = torch.cat([torch.arange(90), torch.arange(110)]).to(dev)[None]
        hp = ours.logits(ours(torch.cat([a, b])[None], cu, pos))[0]
        ref_ab = torch.cat([f(input_ids=a[None]).logits[0], f(input_ids=b[None]).logits[0]]).float()
        e_packed = rel_max(hp, ref_ab)
    print(f"VT-09 C0↔fla GDN LM: dense {e:.2e}, prefill-last {e_last:.2e}, packed {e_packed:.2e}")
    assert e <= 1e-3 and e_last <= 1e-3 and e_packed <= 1e-3


def test_headwise_gate_equals_manual() -> None:
    from hybridlc.models.attention import AttnMixer

    torch.manual_seed(0)
    m = AttnMixer(256, 4, 64).double()
    x = torch.randn(2, 33, 256, dtype=torch.float64)
    captured = {}
    h = m.o_proj.register_forward_hook(lambda mod, inp, out: captured.__setitem__("o", inp[0].clone()))
    m.gate_bypass = True
    y_bypass = m(x)
    o = captured["o"].view(2, 33, 4, 64)
    m.gate_bypass = False
    y = m(x)
    h.remove()
    manual = (o * torch.sigmoid(x @ m.g_proj.weight.t())[..., None]).reshape(2, 33, 256)
    assert torch.allclose(captured["o"], manual, rtol=0, atol=1e-12)
    assert torch.allclose(y, manual @ m.o_proj.weight.t(), rtol=0, atol=1e-12)
    assert not torch.allclose(y, y_bypass)
    # W_g ∈ ℝ^{H×d}, gate starts near 0.5 at the protocol init (std 0.02)
    assert tuple(m.g_proj.weight.shape) == (4, 256)

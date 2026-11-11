"""VT-22 (R3c, DEV-01): full-model transplant at the lengths we evaluate.

Pre-registration: ``experiments/repro/R3/prereg.yaml`` (copied from reproduction_plan.md §2.3 R3c).

(1) D384 C12 ↔ transformers ``Qwen3ForCausalLM`` (VT-09(1) mapping: gate bypassed, gains tied, θ 5e5)
(2) D384 C0 ↔ fla ``GatedDeltaNetForCausalLM`` (VT-09(2) mapping)
    at T = 4,096 (fp32) and T ∈ {16,384, 32,768} (bf16 autocast; ours on the production varlen path,
    Qwen3 through SDPA with the flash backend pinned, fla through its chunk path).
(3) D384 C3: packed (16K + 16K) vs each document alone, 32K, bf16.
Logits at every 16th position and the last 512 positions.

Pass (frozen): 4K fp32 rel max ≤ 1e-3; 16K/32K bf16 rel RMS ≤ 2e-2 and RMS error(last 1K) ≤
2 × RMS error(first 1K); C3 packed vs alone rel RMS ≤ 1e-2. No argmax criterion.

Raw output: ``experiments/repro/R3/vt22/vt22_transplant.jsonl``.
Run: ``pytest -q -s tests/test_vt22_long_transplant.py``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "repro"))
import r3_common as RC  # noqa: E402

SCALE = "D384"
TOL = {"fp32_rel_max": 1e-3, "bf16_rel_rms": 2e-2, "band_ratio": 2.0, "packed_rel_rms": 1e-2}


# ------------------------------------------------------------------ references (VT-09 mappings, copied)
def _qwen3(cfg, attn: str):
    from transformers import Qwen3Config, Qwen3ForCausalLM

    qc = Qwen3Config(vocab_size=cfg.vocab_size, hidden_size=cfg.d_model, intermediate_size=cfg.d_ff,
                     num_hidden_layers=cfg.n_layers, num_attention_heads=cfg.d_model // 64,
                     num_key_value_heads=cfg.d_model // 64, head_dim=64,
                     rope_parameters={"rope_type": "default", "rope_theta": 5e5}, rms_norm_eps=1e-5,
                     tie_word_embeddings=True, attention_bias=False, use_sliding_window=False,
                     max_position_embeddings=32768)
    qc._attn_implementation = attn
    torch.manual_seed(0)
    m = Qwen3ForCausalLM(qc).float().eval()
    with torch.no_grad():
        for _, p in m.named_parameters():
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
            mx.q_norm.copy_(sd[p + "self_attn.q_norm.weight"].repeat(H))
            mx.k_norm.copy_(sd[p + "self_attn.k_norm.weight"].repeat(H))
    ours.set_gate_bypass(True)


def _fla_gdn_lm(cfg):
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


# ------------------------------------------------------------------ helpers
def sample_positions(T: int) -> torch.Tensor:
    return torch.unique(torch.cat([torch.arange(0, T, 16), torch.arange(T - 512, T)]))


def _ours_logits(ours, ids, P, bf16: bool):
    dev = ids.device
    T = ids.shape[1]
    cu = torch.tensor([0, T], dtype=torch.int32, device=dev)
    pos = torch.arange(T, device=dev)[None]
    with torch.autocast("cuda", torch.bfloat16, enabled=bf16):
        h = ours(ids, cu, pos, max_seqlen=T)
    return ours.logits(h[0, P.to(dev)]).float()


def _ref_hidden_logits(ref, ids, P, bf16: bool, family: str):
    from torch.nn.attention import SDPBackend, sdpa_kernel

    ctx = sdpa_kernel([SDPBackend.FLASH_ATTENTION]) if (bf16 and family == "qwen3") else torch.autocast("cuda", enabled=False)
    with ctx, torch.autocast("cuda", torch.bfloat16, enabled=bf16):
        out = ref.model(input_ids=ids)
        h = out.last_hidden_state[0, P.to(ids.device)]
        lg = ref.lm_head(h)
    return lg.float()


def _metrics(mine: torch.Tensor, ref: torch.Tensor, P: torch.Tensor, T: int) -> dict:
    d = (mine.double() - ref.double())
    first = P < 1024
    last = P >= T - 1024
    e_first = float(d[first].pow(2).mean().sqrt())
    e_last = float(d[last].pow(2).mean().sqrt())
    return {"rel_rms": RC.rel_rms(mine, ref), "rel_max": RC.rel_max(mine, ref), "rms_err_first1k": e_first,
            "rms_err_last1k": e_last, "band_ratio": e_last / max(e_first, 1e-300),
            "ref_rms": RC.rms(ref), "n_positions": int(P.numel()),
            # finer bands (report-only): 8 equal position bands
            "band8_rel_rms": [float(d[(P >= T * i // 8) & (P < T * (i + 1) // 8)].pow(2).mean().sqrt() / ref.double().pow(2).mean().sqrt())
                              for i in range(8)]}


def _judge(row: dict, T: int, bf16: bool) -> bool:
    if not bf16:
        return row["rel_max"] <= TOL["fp32_rel_max"]
    return row["rel_rms"] <= TOL["bf16_rel_rms"] and row["band_ratio"] <= TOL["band_ratio"]


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.xfail(strict=False, raises=AssertionError, reason="FAIL-as-registered; accepted residual RR-VT22; PI ruling D072; criterion unchanged")
def test_vt22_long_transplant() -> None:
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    from hybridlc.models.build import build_model
    from hybridlc.models.config import ModelConfig
    from hybridlc.utils.gpu import claim

    dev = claim(0.44)
    meta = RC.run_meta("VT-22", {"criteria": TOL, "scale": SCALE})
    wr = RC.RawWriter("vt22", "vt22_transplant", meta)
    fails = []
    cases = [(4096, False), (16384, True), (32768, True)]
    for family, cfg_name in (("qwen3", "C12"), ("fla_gdn", "C0")):
        cfg = ModelConfig.for_scale(SCALE, cfg_name)
        ours = build_model(cfg, seed_index=0).eval()
        for T, bf16 in cases:
            if family == "qwen3":
                ref = _qwen3(cfg, "sdpa" if bf16 else "eager")
                _transplant_qwen(ours, ref)
            else:
                ref = _fla_gdn_lm(cfg)
                _transplant_fla(ours, ref)
            ours_d, ref_d = ours.to(dev), ref.to(dev)
            g = RC.torch_gen(22, T, 0 if family == "qwen3" else 1)
            ids = torch.randint(8, cfg.vocab_size, (1, T), generator=g).to(dev)
            P = sample_positions(T)
            with torch.no_grad():
                ref_lg = _ref_hidden_logits(ref_d, ids, P, bf16, family)
                torch.cuda.empty_cache()
                mine = _ours_logits(ours_d, ids, P, bf16)
            row = _metrics(mine.cpu(), ref_lg.cpu(), P, T)
            row.update(family=family, config=cfg_name, T=T, precision="bf16" if bf16 else "fp32",
                       peak_mem_gib=torch.cuda.max_memory_allocated() / 2**30)
            row["pass"] = bool(_judge(row, T, bf16))
            wr.write(row)
            print(f"VT-22 {family} {cfg_name} T={T} {row['precision']}: rel_rms {row['rel_rms']:.2e} rel_max "
                  f"{row['rel_max']:.2e} band {row['band_ratio']:.2f} pass={row['pass']}")
            if not row["pass"]:
                fails.append((family, T, row["precision"], row["rel_rms"], row["rel_max"], row["band_ratio"]))
            ref_d.cpu()
            ours.cpu()
            del ref, ref_d, mine, ref_lg
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    # (3) C3 packed 16K+16K vs each alone (32K, bf16)
    cfg = ModelConfig.for_scale(SCALE, "C3")
    ours = build_model(cfg, seed_index=0).eval().to(dev)
    L = 16384
    g = RC.torch_gen(22, 3)
    a = torch.randint(8, cfg.vocab_size, (1, L), generator=g).to(dev)
    b = torch.randint(8, cfg.vocab_size, (1, L), generator=g).to(dev)
    P = sample_positions(L).to(dev)
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        cu = torch.tensor([0, L, 2 * L], dtype=torch.int32, device=dev)
        pos = torch.cat([torch.arange(L), torch.arange(L)]).to(dev)[None]
        hp = ours(torch.cat([a, b], 1), cu, pos, max_seqlen=L)
        packed = [ours.logits(hp[0, P]).float(), ours.logits(hp[0, L + P]).float()]
        del hp
        alone = []
        for x in (a, b):
            h = ours(x, torch.tensor([0, L], dtype=torch.int32, device=dev), torch.arange(L, device=dev)[None],
                     max_seqlen=L)
            alone.append(ours.logits(h[0, P]).float())
            del h
    pk, al = torch.cat(packed).cpu(), torch.cat(alone).cpu()
    row = {"family": "packed_vs_alone", "config": "C3", "T": 2 * L, "precision": "bf16",
           "rel_rms": RC.rel_rms(pk, al), "rel_max": RC.rel_max(pk, al),
           "doc0_rel_rms": RC.rel_rms(packed[0].cpu(), alone[0].cpu()),
           "doc1_rel_rms": RC.rel_rms(packed[1].cpu(), alone[1].cpu()),
           "bitwise_equal": bool(torch.equal(pk, al))}
    row["pass"] = bool(row["rel_rms"] <= TOL["packed_rel_rms"])
    wr.write(row)
    print(f"VT-22 C3 packed vs alone: rel_rms {row['rel_rms']:.2e} (doc0 {row['doc0_rel_rms']:.2e}, doc1 "
          f"{row['doc1_rel_rms']:.2e}) bitwise={row['bitwise_equal']} pass={row['pass']}")
    if not row["pass"]:
        fails.append(("packed_vs_alone", 2 * L, "bf16", row["rel_rms"]))
    wr.close({"fails": fails, "verdict": "PASS" if not fails else "FAIL"})
    assert not fails, fails

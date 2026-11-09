"""Analytic parameter counts, FLOPs and inference-state bytes (protocol §3.3, §3.4, §10.1).

All formulas are closed form; the fla-mixer counts reproduce meta-device instantiation of fla 0.6.0
exactly (VT-01 checks both).  ``table_row`` regenerates one row of the protocol §3.4 table.

Formulas (per token, single document, nominal):

* params(F_g)  = 4·d·(H·hd) + 2·H·hd (per-head QK gains) + d·H (W_g) [+ H if SSMax]
* params(F_gc) = params(F_g) + 3·(H·hd)·W
* params(GDN)  = 2·d·H·d_k + d·HV·d_v (q,k,v) + 2·d·HV (a,b) + 2·HV (A_log, dt_bias)
                 + (2·H·d_k + HV·d_v)·W (convs, if any) + d·HV·d_v (g_proj, if gated) + d_v (o_norm)
                 + HV·d_v·d (o_proj)
* params(GLA)  = 2·d·K + 2·d·Vd (q,k,v,g) + d·16 + 16·K + K (low-rank gk) + (2K + Vd)·W + d_v + Vd·d
* FLOPs        = 2·N_non-emb + 2·d·V + k·2·T·d + Σ_linear H·(6·d_k·d_v + 2C(3·d_k + 2·d_v) + C²/3),  C = 64
* state bytes  = k·T·2·d·kv_bytes (KV) + Σ_linear [HV·d_k·d_v·4 + conv channels·W·4] + Σ_F_gc 3·d·W·4
"""

from __future__ import annotations

from typing import Any

from .config import SCALES, TEF_TOKENS, ModelConfig

#: GDN chunk size used in the FLOP formula (protocol §3.4).
CHUNK = 64
#: GLA low-rank gate dimension (fla default ``gate_low_rank_dim``).
GLA_GATE_RANK = 16


# ---------------------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------------------

def linear_dims(cfg: ModelConfig, ch: str) -> dict[str, Any]:
    """Head geometry of a linear mixer: H (qk heads), HV (value heads), d_k, d_v, conv flag."""
    g = cfg.linear_geometry(ch)
    d = cfg.d_model
    if g["kind"] == "gdn":
        H = g["num_heads"]
        dk = g["head_dim"]
        dv = int(dk * g["expand_v"])
        return dict(kind="gdn", H=H, HV=H, dk=dk, dv=dv, conv=g["use_short_conv"], gate=g["use_gate"])
    K = int(d * g["expand_k"])
    Vd = int(d * g["expand_v"])
    H = g["num_heads"]
    return dict(kind="gla", H=H, HV=H, dk=K // H, dv=Vd // H, conv=g["use_short_conv"], gate=g["use_output_gate"])


def mixer_params(cfg: ModelConfig, ch: str) -> int:
    """Parameter count of one mixer of pattern char ``ch``."""
    d, W = cfg.d_model, cfg.conv_size
    if ch in "AC":
        H, hd = cfg.n_attn_heads, cfg.attn_head_dim
        inner = H * hd
        n = 4 * d * inner + 2 * inner + d * H + (H if cfg.ssmax else 0)
        if ch == "C":
            n += 3 * inner * W
        return n
    g = linear_dims(cfg, ch)
    H, HV, dk, dv = g["H"], g["HV"], g["dk"], g["dv"]
    if g["kind"] == "gdn":
        n = 2 * d * H * dk + d * HV * dv + 2 * d * HV + 2 * HV
        if g["conv"]:
            n += (2 * H * dk + HV * dv) * W
        if g["gate"]:
            n += d * HV * dv
        n += dv + HV * dv * d
        return n
    K, Vd = H * dk, H * dv
    n = 2 * d * K + d * Vd + d * GLA_GATE_RANK + GLA_GATE_RANK * K + K + Vd * d
    if g["gate"]:
        n += d * Vd
    if g["conv"]:
        n += (2 * K + Vd) * W
    n += dv  # g_norm weight (head_v_dim)
    return n


def block_shared_params(cfg: ModelConfig) -> int:
    """MLP (3·d·d_ff) + two RMSNorm gains."""
    return 3 * cfg.d_model * cfg.d_ff + 2 * cfg.d_model


def param_counts(cfg: ModelConfig) -> dict[str, Any]:
    """dict(non_embedding, embedding, total, per_layer_mixer=[...], mixers={char: count})."""
    per = [mixer_params(cfg, ch) for ch in cfg.pattern]
    non_emb = sum(per) + cfg.n_layers * block_shared_params(cfg) + cfg.d_model
    emb = cfg.vocab_size * cfg.d_model
    return dict(non_embedding=non_emb, embedding=emb, total=non_emb + emb, per_layer_mixer=per,
                mixers={ch: mixer_params(cfg, ch) for ch in sorted(set(cfg.pattern))})


# ---------------------------------------------------------------------------------------
# FLOPs
# ---------------------------------------------------------------------------------------

def linear_core_flops(cfg: ModelConfig, ch: str, chunk: int = CHUNK) -> float:
    """Per-token chunked-core FLOPs of one linear mixer: H·(6·d_k·d_v + 2C(3d_k + 2d_v) + C²/3)."""
    g = linear_dims(cfg, ch)
    dk, dv = g["dk"], g["dv"]
    return g["HV"] * (6 * dk * dv + 2 * chunk * (3 * dk + 2 * dv) + chunk * chunk / 3)


def flops_per_token(cfg: ModelConfig, T: int) -> float:
    """Nominal forward FLOPs per token at training length T (protocol §3.4)."""
    pc = param_counts(cfg)
    f = 2 * pc["non_embedding"] + 2 * cfg.d_model * cfg.vocab_size
    for ch in cfg.pattern:
        if ch in "AC":
            f += 2 * T * cfg.d_model
        else:
            f += linear_core_flops(cfg, ch)
    return float(f)


def train_pflop(cfg: ModelConfig, stage_a_tokens: float | None = None, stage_b_tokens: float | None = None,
                tef_tokens: float = TEF_TOKENS) -> float:
    """3·fwd(4K)·A + 4·fwd(32K)·(B + TEF) in PFLOP (middle Stage-B option by default)."""
    s = SCALES[cfg.scale]
    a = s["stage_a_tokens"] if stage_a_tokens is None else stage_a_tokens
    b = s["stage_b_mid_tokens"] if stage_b_tokens is None else stage_b_tokens
    return (3 * flops_per_token(cfg, 4096) * a + 4 * flops_per_token(cfg, 32768) * (b + tef_tokens)) / 1e15


# ---------------------------------------------------------------------------------------
# State bytes
# ---------------------------------------------------------------------------------------

def linear_state_bytes(cfg: ModelConfig, ch: str) -> dict[str, int]:
    """fp32 recurrent state and fp32 conv caches of one linear mixer (per sequence)."""
    g = linear_dims(cfg, ch)
    st = g["HV"] * g["dk"] * g["dv"] * 4
    conv = (2 * g["H"] * g["dk"] + g["HV"] * g["dv"]) * cfg.conv_size * 4 if g["conv"] else 0
    return {"gdn_state": st, "conv": conv}


def state_bytes(cfg: ModelConfig, T: int, batch: int = 1, kv_bytes: int = 2) -> dict[str, int]:
    """Inference-state bytes after a prefill of T tokens, exactly as stored (bf16 KV by default)."""
    kv = gs = conv = 0
    for ch in cfg.pattern:
        if ch in "AC":
            kv += T * 2 * cfg.n_attn_heads * cfg.attn_head_dim * kv_bytes
            if ch == "C":
                conv += 3 * cfg.n_attn_heads * cfg.attn_head_dim * cfg.conv_size * 4
        else:
            lb = linear_state_bytes(cfg, ch)
            gs += lb["gdn_state"]
            conv += lb["conv"]
    out = {"kv": kv * batch, "gdn_state": gs * batch, "conv": conv * batch}
    out["total"] = out["kv"] + out["gdn_state"] + out["conv"]
    return out


def decode_bytes_moved(cfg: ModelConfig, T: int, kv_bytes: int = 2) -> float:
    """Bytes moved per decode step (§10.1): 2·N_total + k·T·4d (KV read) + 2·Σ S_lin."""
    pc = param_counts(cfg)
    sb = state_bytes(cfg, T, kv_bytes=kv_bytes)
    return 2 * pc["total"] + sb["kv"] + 2 * (sb["gdn_state"] + sb["conv"])


# ---------------------------------------------------------------------------------------
# §3.4 table
# ---------------------------------------------------------------------------------------

def table_row(scale: str, config_name: str) -> dict[str, Any]:
    """One row of protocol §3.4, with the same rounding as the published table."""
    cfg = ModelConfig.for_scale(scale, config_name)
    pc = param_counts(cfg)
    mib = {T: state_bytes(cfg, T)["total"] / 2**20 for T in (1024, 4096, 16384, 32768)}
    return dict(
        scale=scale, config=config_name, pattern=cfg.pattern,
        non_embedding=pc["non_embedding"], embedding=pc["embedding"], total=pc["total"],
        mflop_4k=round(flops_per_token(cfg, 4096) / 1e6, 1), mflop_32k=round(flops_per_token(cfg, 32768) / 1e6, 1),
        mib_1k=round(mib[1024], 2), mib_4k=round(mib[4096], 2), mib_16k=round(mib[16384], 2), mib_32k=round(mib[32768], 2),
        train_pflop=round(train_pflop(cfg)),
    )

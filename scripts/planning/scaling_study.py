#!/usr/bin/env python
"""Analytic scaling study (CPU, arithmetic only): parameters, inference state, FLOPs/token for hybrid models at larger sizes.

Generalises scripts/planning/flops_state.py (same formulas) to arbitrary d / L. Formulas (checked against the repo's
exact counts for D384/D512, see `check()`):
  full attention layer (F_g)  = 4 d^2 + Ha (d + 128)            Ha = d/64 heads, QK-RMSNorm 2x64 per head, head-wise gate
  Gated DeltaNet layer (GDN-A)= 2 d (64 H) + 3 d (128 H) + 2 d H + 4 (2*64 H + 128 H) + 2 H + 128     H = d/128 heads
  SwiGLU MLP                  = 3 d d_ff ; norms 2 d ; final norm d ; tied embedding V d
  KV cache per full layer     = 4 T d bytes (bf16, K and V, no sharing)
  linear state per GDN layer  = 4 H 64 128 + 16 (2*64 H + 128 H) bytes (fp32 state + conv caches)  (= 110,592 B at d=384)
Not a measurement: FLOPs ignore kernel efficiency; the state formulas were verified byte-exact on device at D384/D512 only.
"""
import json, sys
V = 16384; C = 64

def layer_params(d, dff):
    Ha, H = d // 64, max(1, d // 128)
    fg = 4 * d * d + Ha * (d + 128)
    gdn = 2 * d * 64 * H + 3 * d * 128 * H + 2 * d * H + 4 * (2 * 64 * H + 128 * H) + 2 * H + 128
    mlp = 3 * d * dff + 2 * d
    return fg + mlp, gdn + mlp, H

def state_lin(d):
    H = max(1, d // 128)
    return H * 64 * 128 * 4 + (H * 64 * 2 + H * 128) * 16

def gdn_core(H):
    return H * (6 * 64 * 128 + 2 * C * (3 * 64 + 2 * 128) + C * C / 3)

def model(d, L, dff, k, T):
    f, g, H = layer_params(d, dff)
    nonemb = k * f + (L - k) * g + d
    params = nonemb + V * d
    flops = 2 * nonemb + 2 * d * V + (L - k) * gdn_core(H) + k * 2 * T * d
    state = k * 4 * T * d + (L - k) * state_lin(d)
    return params, nonemb, flops, state

def check():
    # exact repo values: D384 gdn=595,334 fg=592,896 ; D512 gdn=1,056,904 fg=1,053,696 (mixer-only counts)
    for d, gdn, fg in ((384, 595_334, 592_896), (512, 1_056_904, 1_053_696)):
        Ha, H = d // 64, d // 128
        mine_fg = 4 * d * d + Ha * (d + 128)
        mine_g = 2 * d * 64 * H + 3 * d * 128 * H + 2 * d * H + 4 * (2 * 64 * H + 128 * H) + 2 * H + 128
        assert mine_fg == fg and mine_g == gdn, (d, mine_fg, fg, mine_g, gdn)
    assert state_lin(384) == 110_592
    return True

SIZES = [("D384 (study)", 384, 12, 1024), ("D512 (study)", 512, 12, 1536), ("D768 (proposed)", 768, 12, 2304),
         ("d1024 L16", 1024, 16, 2816), ("d2048 L24", 2048, 24, 5632), ("d4096 L32", 4096, 32, 11008)]

if __name__ == "__main__":
    check()
    out = []
    for name, d, L, dff in SIZES:
        for k in sorted({0, 1, 3, max(1, L // 4), L}):
            p, ne, fl, st = model(d, L, dff, k, 32768)
            _, _, _, st1k = model(d, L, dff, k, 1024)
            out.append({"model": name, "d": d, "L": L, "k": k, "params_M": round(p / 1e6, 1), "attn_share_pct": round(100 * k / L, 1),
                        "state_32K_MiB": round(st / 2**20, 2), "state_1K_MiB": round(st1k / 2**20, 2),
                        "max_32K_seqs_in_8GiB_state_only": int(8 * 2**30 // st), "fwd_GFLOP_per_tok_32K": round(fl / 1e9, 3)})
    if "--json" in sys.argv:
        print(json.dumps(out, indent=1)); sys.exit()
    print("| model | L | k | params (M) | attn % | state@32K (MiB) | 32K seqs in 8 GiB (state only) | fwd GFLOP/token @32K |")
    print("|---|---|---|---|---|---|---|---|")
    for r in out:
        print(f"| {r['model']} | {r['L']} | {r['k']} | {r['params_M']} | {r['attn_share_pct']} | {r['state_32K_MiB']} | "
              f"{r['max_32K_seqs_in_8GiB_state_only']:,} | {r['fwd_GFLOP_per_tok_32K']} |")
    print("\nCrossover token length T* where ONE full-attention layer's KV cache equals the WHOLE linear stack's state:")
    for name, d, L, dff in SIZES:
        print(f"- {name}: T* = (L-k)*S_lin/(4d) ~ {int((L - 1) * state_lin(d) / (4 * d)):,} tokens (k=1); single-layer break-even {int(state_lin(d) / (4 * d))} tokens")

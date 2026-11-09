# Meta-device parameter counts for the frozen mixer set (CPU only, no GPU).
import torch, json
from fla.layers import GatedDeltaNet, GatedLinearAttention

def n(m): return sum(p.numel() for p in m.parameters())

out = {}
with torch.device("meta"):
    for d in (384, 512):
        H = d // 128
        out[f"GDN-A d{d}"] = n(GatedDeltaNet(hidden_size=d, head_dim=64, num_heads=H, expand_v=2,
                                              use_gate=True, use_short_conv=True, conv_size=4))
        out[f"GDN-E d{d}"] = n(GatedDeltaNet(hidden_size=d, head_dim=d // 2, num_heads=1, expand_v=2,
                                              use_gate=True, use_short_conv=True, conv_size=4))
        out[f"GDN-nc d{d}"] = n(GatedDeltaNet(hidden_size=d, head_dim=64, num_heads=H, expand_v=2,
                                               use_gate=True, use_short_conv=False, conv_size=4))
        out[f"GDN-D d{d}"] = n(GatedDeltaNet(hidden_size=d, head_dim=128, num_heads=max(1, H // 2), expand_v=2,
                                              use_gate=True, use_short_conv=True, conv_size=4))
        out[f"GLA d{d}"] = n(GatedLinearAttention(hidden_size=d, num_heads=H, expand_k=0.5, expand_v=1.0,
                                                   use_short_conv=True, conv_size=4, use_output_gate=True))
        # analytic attention mixers
        Ha = d // 64
        Fg = 4 * d * d + 2 * Ha * 64 + d * Ha
        out[f"F_g d{d}"] = Fg
        out[f"F_gc d{d}"] = Fg + 3 * d * 4
        out[f"F_plain_perhead d{d}"] = 4 * d * d + 2 * Ha * 64
for k_, v in out.items(): print(f"{k_:22s} {v:>12,}")

# per-config non-embedding and total counts
V = 16384
dff = {384: 1024, 512: 1536}
print()
for d in (384, 512):
    mlp = 3 * d * dff[d]; norms = 2 * d; fin = d; emb = V * d
    gdn = out[f"GDN-A d{d}"]; fg = out[f"F_g d{d}"]
    for k in (0, 1, 2, 3, 6, 12):
        nonemb = (12 - k) * (gdn + mlp + norms) + k * (fg + mlp + norms) + fin
        print(f"d{d} k={k:2d}  non-emb {nonemb:>12,}  emb {emb:>11,}  total {nonemb+emb:>12,}")
    # GDN-E pure / F_gc all-full
    e = out[f"GDN-E d{d}"]
    print(f"d{d} GDN-E k=0 non-emb {12*(e+mlp+norms)+fin:,}; F_gc k=12 non-emb {12*(out[f'F_gc d{d}']+mlp+norms)+fin:,}")

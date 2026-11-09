import torch, warnings
warnings.filterwarnings("ignore")
from fla.layers import GatedDeltaNet
def n(m): return sum(p.numel() for p in m.parameters())
def gdn(d, **kw):
    with torch.device("meta"):
        m = GatedDeltaNet(hidden_size=d, **kw)
    parts = {k: sum(p.numel() for p in mod.parameters()) for k, mod in m.named_children()}
    own = sum(p.numel() for name,p in m.named_parameters(recurse=False))
    return n(m), parts, own, m
for d in (512, 384):
    print(f"==== d={d}  4d^2={4*d*d}")
    cfgs = {
      "A_gate_dk64_dv128_H{}".format(d//128): dict(head_dim=64, num_heads=d//128, expand_v=2, use_gate=True, use_short_conv=True, conv_size=4),
      "B_nogate_dk128_dv128_H{}".format(d//128): dict(head_dim=128, num_heads=d//128, expand_v=1, use_gate=False, use_short_conv=True, conv_size=4),
      "C_gate_dk128_dv128_H{}(5d2)".format(d//128): dict(head_dim=128, num_heads=d//128, expand_v=1, use_gate=True, use_short_conv=True, conv_size=4),
      "D_gate_dk128_dv256_H{}".format(d//256 if d>=256 else 1): dict(head_dim=128, num_heads=max(1,d//256), expand_v=2, use_gate=True, use_short_conv=True, conv_size=4),
    }
    for name, kw in cfgs.items():
        try:
            tot, parts, own, m = gdn(d, **kw)
            H, dk, dv = m.num_heads, m.head_k_dim, m.head_v_dim
            state = H*dk*dv
            conv = (kw["conv_size"]-1)*(2*H*dk + m.num_v_heads*dv)
            print(f"{name:32s} params={tot:>9,d} ({tot/(d*d):.4f} d^2, {100*(tot-4*d*d)/(4*d*d):+.2f}% vs MHA) state={state:,} el = {state*4/1024:.0f} KiB fp32; conv_state={conv:,} el")
            print("     parts:", parts, "own:", own)
        except Exception as e:
            print(name, "ERR", e)

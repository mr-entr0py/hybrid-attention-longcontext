"""CPU: per-layer output RMS at init of each mixer kind vs the residual stream (D384, seed 0), on real LM2 text.
Shows the effect of the D064 conv init N(0,0.02) on F_gc (no output norm) vs F_g and GDN-A (o_norm)."""
import sys, json
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
import torch
from hybridlc.models.build import build_model
from hybridlc.models.config import ModelConfig
from hybridlc.data.loader import StageStream
torch.set_num_threads(8)
b = StageStream("LM2", 0).get(0, 0)
T = 1024
ids = b["input_ids"][:, :T]
cu = b["cu_seqlens"]; cu = torch.cat([cu[cu < T], torch.tensor([T], dtype=cu.dtype)])
pos = b["position_ids"][:, :T]
out = {}
for name in ("C12", "X-gc12"):
    m = build_model(ModelConfig.for_scale("D384", name), 0).eval()
    rec = {}
    hooks = []
    for l, blk in enumerate(m.layers):
        def mk(l):
            def f(mod, inp, outp):
                rec.setdefault(l, {})["mixer_out_rms"] = float(outp.float().pow(2).mean().sqrt())
                rec[l]["resid_in_rms"] = float(inp[0].float().pow(2).mean().sqrt()) if False else None
            return f
        hooks.append(blk.mixer.register_forward_hook(mk(l)))
        def mk2(l):
            def f(mod, inp, outp):
                rec.setdefault(l, {})["mlp_out_rms"] = float(outp.float().pow(2).mean().sqrt())
            return f
        hooks.append(blk.mlp.register_forward_hook(mk2(l)))
    with torch.no_grad():
        # residual stream RMS entering each block
        h = m.embed(ids)
        resid = []
        ctx = {"max_seqlen": int((cu[1:] - cu[:-1]).max())}
        from hybridlc.models.layers import rope_tables
        ctx["rope"] = rope_tables(pos, 64, 5e5)
        for blk in m.layers:
            resid.append(float(h.pow(2).mean().sqrt()))
            h = blk(h, cu, pos, ctx)
    for hk in hooks: hk.remove()
    out[name] = {"layer0": {**rec[0], "resid_in_rms": resid[0]}, "layer6": {**rec[6], "resid_in_rms": resid[6]},
                 "mean_mixer_out_rms": sum(rec[l]["mixer_out_rms"] for l in rec) / len(rec),
                 "mean_mlp_out_rms": sum(rec[l]["mlp_out_rms"] for l in rec) / len(rec)}
    if name == "X-gc12":
        mx = m.layers[0].mixer
        x = m.layers[0].norm1(m.embed(ids))
        q, k, v, _ = mx._qkv(x, cu, None, False)
        vp = mx.v_proj(x)
        out[name]["v_rms_after_conv_silu"] = float(v.pow(2).mean().sqrt())
        out[name]["v_proj_rms"] = float(vp.pow(2).mean().sqrt())
        out[name]["conv_w_std"] = float(mx.conv.weight.std())
print(json.dumps(out, indent=1))

import time, sys, types, torch, torch.nn.functional as F
torch.cuda.set_per_process_memory_fraction(0.12)
dev, dt = 'cuda', torch.bfloat16
from torch.nn.attention.varlen import varlen_attn
# ---- 1) torch varlen_attn fwd/bwd + doc isolation
H, Dh = 8, 64
cu = torch.tensor([0, 700, 1500, 4096], device=dev, dtype=torch.int32)
q = torch.randn(4096, H, Dh, device=dev, dtype=dt, requires_grad=True)
o = varlen_attn(q, q, q, cu, cu, 2596, 2596, window_size=(-1, 0)); o.float().sum().backward()
o2 = F.scaled_dot_product_attention(*(q[700:1500].transpose(0,1)[None],)*3, is_causal=True)[0].transpose(0,1)
print("torch varlen_attn ok; doc-isolation diff vs SDPA", (o[700:1500]-o2).abs().max().item())
# ---- 2) FlexAttention document mask (compiled)
try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    doc = torch.zeros(4096, dtype=torch.int32, device=dev); doc[700:]=1; doc[1500:]=2
    def mm(b,h,qi,ki): return (qi>=ki) & (doc[qi]==doc[ki])
    bm = create_block_mask(mm, None, None, 4096, 4096, device=dev)
    fa = torch.compile(flex_attention)
    qq = q.detach().transpose(0,1)[None].clone().requires_grad_(True)
    of = fa(qq,qq,qq, block_mask=bm); of.float().sum().backward(); torch.cuda.synchronize()
    print("flex_attention doc-mask ok; diff vs varlen", (of[0].transpose(0,1)-o).abs().max().item())
except Exception as e:
    print("flex_attention FAILED", type(e).__name__, str(e)[:300])
# ---- 3) shim flash_attn API with torch kernels so fla.layers.attn.Attention works without flash-attn
def flash_attn_func(q, k, v, causal=True, window_size=(-1,-1), **kw):
    # q,k,v: [B, T, H, D]; GQA handled by repeat
    if k.shape[2] != q.shape[2]:
        r = q.shape[2]//k.shape[2]; k = k.repeat_interleave(r, 2); v = v.repeat_interleave(r, 2)
    Tq, Tk = q.shape[1], k.shape[1]
    assert window_size[0] < 0, "SWA not shimmed"
    if Tq == Tk:
        o = F.scaled_dot_product_attention(q.transpose(1,2), k.transpose(1,2), v.transpose(1,2), is_causal=causal)
    else:  # decode: queries are the last Tq positions -> bottom-right aligned causal
        if Tq == 1:
            o = F.scaled_dot_product_attention(q.transpose(1,2), k.transpose(1,2), v.transpose(1,2), is_causal=False)
        else:
            from torch.nn.attention.bias import causal_lower_right
            o = F.scaled_dot_product_attention(q.transpose(1,2), k.transpose(1,2), v.transpose(1,2), attn_mask=causal_lower_right(Tq, Tk))
    return o.transpose(1,2)
def flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, causal=True, window_size=(-1,-1), **kw):
    if k.shape[1] != q.shape[1]:
        r = q.shape[1]//k.shape[1]; k = k.repeat_interleave(r, 1); v = v.repeat_interleave(r, 1)
    return varlen_attn(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, window_size=(window_size[0], 0) if causal else window_size)
import fla, fla.layers.attn as _fattn
_fattn.flash_attn_func = flash_attn_func; _fattn.flash_attn_varlen_func = flash_attn_varlen_func
from fla.models import GatedDeltaNetConfig
from transformers import AutoModelForCausalLM
# ~ small hybrid: 12 layers, d=512, attention at layers [5, 11] (r=1/6)
cfg = GatedDeltaNetConfig(hidden_size=512, num_hidden_layers=12, num_heads=4, head_dim=128, expand_v=1.0, use_gate=False,
                          vocab_size=16384, hidden_ratio=4, max_position_embeddings=8192, fuse_cross_entropy=True,
                          tie_word_embeddings=True,
                          attn={'layers':[5,11], 'num_heads':8, 'num_kv_heads':8, 'qkv_bias':False, 'rope_theta':10000., 'window_size':None})
model = AutoModelForCausalLM.from_config(cfg).to(dev, dt)
n_all = sum(p.numel() for p in model.parameters()); n_emb = model.get_input_embeddings().weight.numel()
print("hybrid params total %.2fM, embedding %.2fM, non-emb %.2fM" % (n_all/1e6, n_emb/1e6, (n_all-n_emb)/1e6))
print([type(b.attn).__name__ for b in model.model.layers])
ids = torch.randint(0, 16384, (1, 4096), device=dev)
t0=time.time(); out = model(input_ids=ids, labels=ids, cu_seqlens=cu); out.loss.backward(); torch.cuda.synchronize()
print("hybrid varlen fwd/bwd ok loss %.3f (%.1fs)"%(out.loss.item(), time.time()-t0))
model.zero_grad(set_to_none=True)
# decode consistency: prefill+decode vs full forward
model.eval(); model.float()
with torch.no_grad():
    x = torch.randint(0, 16384, (1, 300), device=dev)
    full = model(input_ids=x).logits.float()
    o1 = model(input_ids=x[:, :256], use_cache=True); pkv = o1.past_key_values; outs=[o1.logits.float()]
    for t in range(256, 300):
        ot = model(input_ids=x[:, t:t+1], past_key_values=pkv, use_cache=True); pkv = ot.past_key_values; outs.append(ot.logits.float())
    dec = torch.cat(outs, 1)
    print("hybrid prefill+decode vs full logits max abs diff %.4f, argmax agreement %.4f" % ((dec-full).abs().max().item(), (dec.argmax(-1)==full.argmax(-1)).float().mean().item()))
    # state memory accounting
    tot=0
    for i in range(len(pkv)):
        st = pkv[i]
        for k_ in ('recurrent_state','attn_state','conv_state'):
            v_ = st[k_]
            if v_ is None: continue
            ts = v_ if isinstance(v_, (tuple,list)) else [v_]
            b = sum(t.numel()*t.element_size() for t in ts if torch.is_tensor(t)); tot+=b
            if i in (0,5): print("  layer", i, k_, [tuple(t.shape) for t in ts if torch.is_tensor(t)], [t.dtype for t in ts if torch.is_tensor(t)], b, "bytes")
    print("total cache bytes at T=300:", tot)
    g = model.generate(x[:, :32], max_new_tokens=16, do_sample=False)
    print("HF generate ok", g.shape)
print("peak mem MB", torch.cuda.max_memory_allocated()/2**20)

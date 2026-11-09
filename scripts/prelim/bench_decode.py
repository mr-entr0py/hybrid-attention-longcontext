import time, torch, torch.nn.functional as F
torch.cuda.set_per_process_memory_fraction(0.14)
dev, dt = 'cuda', torch.bfloat16
from torch.nn.attention.varlen import varlen_attn
from torch.nn.attention.bias import causal_lower_right
def flash_attn_func(q,k,v,causal=True,window_size=(-1,-1),**kw):
    q,k,v=(t.transpose(1,2) for t in (q,k,v)); Tq,Tk=q.shape[2],k.shape[2]
    if Tq==Tk: o=F.scaled_dot_product_attention(q,k,v,is_causal=True)
    elif Tq==1: o=F.scaled_dot_product_attention(q,k,v)
    else: o=F.scaled_dot_product_attention(q,k,v,attn_mask=causal_lower_right(Tq,Tk))
    return o.transpose(1,2)
import fla, fla.layers.attn as A; A.flash_attn_func=flash_attn_func
from fla.models import GatedDeltaNetConfig
from transformers import AutoModelForCausalLM
L=12
for name, layers in [("all-GDN",[]),("r=1/6",[5,11]),("all-attn",list(range(12)))]:
    cfg = GatedDeltaNetConfig(hidden_size=512, num_hidden_layers=L, num_heads=4, head_dim=128, expand_v=1.0, use_gate=False, vocab_size=16384,
        max_position_embeddings=32768, tie_word_embeddings=True,
        attn={'layers':layers,'num_heads':8,'num_kv_heads':8,'qkv_bias':False,'rope_theta':10000.,'window_size':None} if layers else None)
    m = AutoModelForCausalLM.from_config(cfg).to(dev, dt).eval()
    for ctx in [1024, 8192, 16384]:
        with torch.no_grad():
            x = torch.randint(0,16384,(1,ctx),device=dev)
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); base=torch.cuda.memory_allocated()
            t0=time.time(); o = m(input_ids=x, use_cache=True); torch.cuda.synchronize(); pre=time.time()-t0
            t0=time.time(); o = m(input_ids=x, use_cache=True); torch.cuda.synchronize(); pre=time.time()-t0
            pkv=o.past_key_values; tok=o.logits[:,-1:].argmax(-1)
            cache_b = torch.cuda.memory_allocated()-base
            for i in range(40):
                if i==8: torch.cuda.synchronize(); t0=time.time()
                o = m(input_ids=tok, past_key_values=pkv, use_cache=True); pkv=o.past_key_values; tok=o.logits[:,-1:].argmax(-1)
            torch.cuda.synchronize(); d=(time.time()-t0)/32
        print(f"{name:9s} ctx {ctx:6d}  prefill {pre*1e3:7.1f} ms  decode {d*1e3:6.2f} ms/tok (eager, B=1)  live-alloc-after-prefill {cache_b/2**20:7.1f} MB")
        del o, pkv; torch.cuda.empty_cache()
    del m; torch.cuda.empty_cache()

import time, torch, torch.nn.functional as F
torch.cuda.set_per_process_memory_fraction(0.14)
dev, dt = 'cuda', torch.bfloat16
def flash_attn_func(q,k,v,causal=True,window_size=(-1,-1),**kw):
    return F.scaled_dot_product_attention(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),is_causal=causal).transpose(1,2)
import fla, fla.layers.attn as A; A.flash_attn_func=flash_attn_func
from fla.models import GatedDeltaNetConfig
from transformers import AutoModelForCausalLM
T=4096; B=1
cfg = GatedDeltaNetConfig(hidden_size=512, num_hidden_layers=12, num_heads=4, head_dim=128, expand_v=1.0, use_gate=False, vocab_size=16384,
    max_position_embeddings=T, tie_word_embeddings=True, attn={'layers':[5,11],'num_heads':8,'num_kv_heads':8,'qkv_bias':False,'rope_theta':10000.,'window_size':None})
for mode in ["eager","compile"]:
    torch.manual_seed(0); m = AutoModelForCausalLM.from_config(cfg).to(dev, dt)
    fwd = torch.compile(m) if mode=="compile" else m
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, fused=True)
    ids = torch.randint(0,16384,(B,T),device=dev)
    t_first=time.time()
    for i in range(15):
        if i==5: torch.cuda.synchronize(); t0=time.time()
        loss = fwd(input_ids=ids, labels=ids).loss; loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
        if i==0: torch.cuda.synchronize(); print(mode, "first step %.1fs"%(time.time()-t_first))
    torch.cuda.synchronize(); d=(time.time()-t0)/10
    print(f"{mode}: {d*1e3:.0f} ms/step  {B*T/d/1e3:.1f}k tok/s  peak {torch.cuda.max_memory_allocated()/2**30:.2f} GB loss {loss.item():.3f}")
    del m, fwd, opt; torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()

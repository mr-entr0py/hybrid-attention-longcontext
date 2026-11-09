import time, torch, torch.nn.functional as F
torch.cuda.set_per_process_memory_fraction(0.14)
dev, dt = 'cuda', torch.bfloat16
from torch.nn.attention.varlen import varlen_attn
def flash_attn_func(q,k,v,causal=True,window_size=(-1,-1),**kw):
    return F.scaled_dot_product_attention(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),is_causal=causal).transpose(1,2)
def flash_attn_varlen_func(q,k,v,cu_seqlens_q,cu_seqlens_k,max_seqlen_q,max_seqlen_k,causal=True,window_size=(-1,-1),**kw):
    return varlen_attn(q,k,v,cu_seqlens_q,cu_seqlens_k,max_seqlen_q,max_seqlen_k,window_size=(window_size[0],0))
import fla, fla.layers.attn as A; A.flash_attn_func=flash_attn_func; A.flash_attn_varlen_func=flash_attn_varlen_func
from fla.models import GatedDeltaNetConfig
from transformers import AutoModelForCausalLM
L=12; T=4096
for name, layers in [("all-GDN r=0",[]),("r=1/6 [5,11]",[5,11]),("r=1/2",list(range(1,12,2))),("all-attn r=1",list(range(12)))]:
    cfg = GatedDeltaNetConfig(hidden_size=512, num_hidden_layers=L, num_heads=4, head_dim=128, expand_v=1.0, use_gate=False, vocab_size=16384,
        hidden_ratio=4, max_position_embeddings=T, tie_word_embeddings=True,
        attn={'layers':layers,'num_heads':8,'num_kv_heads':8,'qkv_bias':False,'rope_theta':10000.,'window_size':None} if layers else None)
    torch.manual_seed(0); m = AutoModelForCausalLM.from_config(cfg).to(dev, dt)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, fused=True)
    ids = torch.randint(0,16384,(1,T),device=dev); cu=torch.tensor([0,T],device=dev,dtype=torch.int32)
    for i in range(13):
        if i==3: torch.cuda.synchronize(); t0=time.time()
        loss = m(input_ids=ids, labels=ids, cu_seqlens=cu).loss; loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize(); dtm=(time.time()-t0)/10
    print(f"{name:14s} params {sum(p.numel() for p in m.parameters())/1e6:.2f}M  step {dtm*1e3:.0f} ms  {T/dtm/1e3:.1f}k tok/s  peak {torch.cuda.max_memory_allocated()/2**30:.2f} GB")
    del m, opt; torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()

# Which attention backends accept d_v != d_qk (twin attention: 4 heads x (64 qk / 128 v))? Is head duplication exact?
import torch, torch.nn.functional as F, time
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.nn.attention.varlen import varlen_attn
torch.cuda.set_per_process_memory_fraction(0.40)
dev='cuda'; B,H,T,E,Ev=1,4,4096,64,128
torch.manual_seed(0)
q=torch.randn(B,H,T,E,device=dev,dtype=torch.bfloat16,requires_grad=True)
k=torch.randn(B,H,T,E,device=dev,dtype=torch.bfloat16,requires_grad=True)
v=torch.randn(B,H,T,Ev,device=dev,dtype=torch.bfloat16,requires_grad=True)
ref=F.scaled_dot_product_attention(q.float(),k.float(),v.float(),is_causal=True)
for name,be in [('FLASH',SDPBackend.FLASH_ATTENTION),('EFFICIENT',SDPBackend.EFFICIENT_ATTENTION),('CUDNN',SDPBackend.CUDNN_ATTENTION),('MATH',SDPBackend.MATH)]:
    try:
        with sdpa_kernel(be):
            o=F.scaled_dot_product_attention(q,k,v,is_causal=True)
            o.float().sum().backward()
        print(f"SDPA {name:9s} Ev!=E fwd+bwd OK  max|o-ref|={ (o.float()-ref).abs().max().item():.2e}")
    except Exception as e: print(f"SDPA {name:9s} Ev!=E FAILED: {str(e).splitlines()[0][:110]}")
    q.grad=k.grad=v.grad=None
# FlexAttention
try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    bm=create_block_mask(lambda b,h,qi,ki: qi>=ki, B,H,T,T,device=dev)
    fa=torch.compile(flex_attention)
    o=fa(q,k,v,block_mask=bm); o.float().sum().backward()
    print(f"FlexAttention   Ev!=E fwd+bwd OK  max|o-ref|={(o.float()-ref).abs().max().item():.2e}")
except Exception as e: print("FlexAttention Ev!=E FAILED:", str(e).splitlines()[0][:150])
q.grad=k.grad=v.grad=None
# varlen_attn (FA2) with Ev != E
qv=q[0].transpose(0,1).contiguous(); kv=k[0].transpose(0,1).contiguous(); vv=v[0].transpose(0,1).contiguous()
cu=torch.tensor([0,T],device=dev,dtype=torch.int32)
try:
    o=varlen_attn(qv,kv,vv,cu,cu,T,T,window_size=(-1,0)); print("varlen_attn Ev!=E OK", o.shape)
except Exception as e: print("varlen_attn  Ev!=E FAILED:", str(e).splitlines()[0][:150])
# duplication trick: 8 heads x 64, heads (2h,2h+1) share q,k; v split in halves
qd=qv.repeat_interleave(2,dim=1); kd=kv.repeat_interleave(2,dim=1); vd=vv.view(T,H,2,64).reshape(T,2*H,64)
od=varlen_attn(qd,kd,vd,cu,cu,T,T,window_size=(-1,0)).view(T,H,128)
print(f"duplication via varlen_attn: max|o-ref|={(od.float()-ref[0].transpose(0,1)).abs().max().item():.2e}  (bf16 kernel vs fp32 ref)")

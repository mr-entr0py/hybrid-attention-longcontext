import torch, time, torch.nn.functional as F
dev='cuda'
B,T,H,D=1,4096,4,64
# SDPA flash backend
q=torch.randn(B,H,T,D,device=dev,dtype=torch.bfloat16)
from torch.nn.attention import sdpa_kernel, SDPBackend
for be in [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.CUDNN_ATTENTION]:
    try:
        with sdpa_kernel(be):
            o=F.scaled_dot_product_attention(q,q,q,is_causal=True); torch.cuda.synchronize()
        print('SDPA',be,'OK')
    except Exception as e: print('SDPA',be,'FAIL',str(e)[:100])
from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
qq=F.normalize(torch.randn(B,T,H,D,device=dev,dtype=torch.bfloat16),dim=-1).requires_grad_()
kk=F.normalize(torch.randn(B,T,H,D,device=dev,dtype=torch.bfloat16),dim=-1).requires_grad_()
vv=torch.randn(B,T,H,2*D,device=dev,dtype=torch.bfloat16).requires_grad_()
g=F.logsigmoid(torch.randn(B,T,H,device=dev,dtype=torch.float32))
beta=torch.rand(B,T,H,device=dev,dtype=torch.bfloat16)
try:
    o,s=chunk_gated_delta_rule(qq,kk,vv,g,beta,output_final_state=True); o.sum().backward(); torch.cuda.synchronize()
    print('GDN chunk fwd/bwd OK', o.shape, s.shape, s.dtype)
    o2,s2=fused_recurrent_gated_delta_rule(qq[:,:16],kk[:,:16],vv[:,:16],g[:,:16],beta[:,:16],output_final_state=True)
    o3,_=chunk_gated_delta_rule(qq[:,:16],kk[:,:16],vv[:,:16],g[:,:16],beta[:,:16])
    print('recurrent vs chunk maxdiff', (o2.float()-o3.float()).abs().max().item())
except Exception as e: print('GDN FAIL', repr(e)[:300])
try:
    from fla.ops.gla import chunk_gla
    gk=F.logsigmoid(torch.randn(B,T,H,D,device=dev))/16
    o,_=chunk_gla(qq.detach(),kk.detach(),vv.detach(),gk); torch.cuda.synchronize(); print('GLA chunk OK')
except Exception as e: print('GLA FAIL', repr(e)[:300])
try:
    from fla.layers import Mamba2
    m=Mamba2(hidden_size=256, head_dim=64, state_size=128, expand=2).to(dev).to(torch.bfloat16)
    y=m(torch.randn(1,1024,256,device=dev,dtype=torch.bfloat16)); print('Mamba2 layer OK', type(y))
except Exception as e: print('Mamba2 FAIL', repr(e)[:300])

import torch, torch.nn.functional as F
from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
torch.manual_seed(0); dev='cuda'
B,T,H,D=1,512,4,64
def naive(q,k,v,g,beta):
    q,k,v,g,beta=[x.float() for x in (q,k,v,g,beta)]
    S=torch.zeros(B,H,D,v.shape[-1],device=dev); out=[]
    scale=D**-0.5
    for t in range(T):
        kt=k[:,t]; vt=v[:,t]; a=g[:,t].exp()[...,None,None]; b=beta[:,t][...,None]
        S=S*a
        pred=torch.einsum('bhk,bhkv->bhv',kt,S)
        S=S+torch.einsum('bhk,bhv->bhkv',kt,b*(vt-pred))
        out.append(torch.einsum('bhk,bhkv->bhv',q[:,t]*scale,S))
    return torch.stack(out,1)
for dt in [torch.float32, torch.bfloat16]:
    q=F.normalize(torch.randn(B,T,H,D,device=dev),dim=-1).to(dt); k=F.normalize(torch.randn(B,T,H,D,device=dev),dim=-1).to(dt)
    v=torch.randn(B,T,H,2*D,device=dev).to(dt); g=F.logsigmoid(torch.randn(B,T,H,device=dev)+3); beta=torch.rand(B,T,H,device=dev).to(dt)
    ref=naive(q,k,v,g,beta)
    oc,_=chunk_gated_delta_rule(q,k,v,g,beta); orc,_=fused_recurrent_gated_delta_rule(q=q,k=k,v=v,g=g,beta=beta)
    print(dt,'chunk-ref',(oc.float()-ref).abs().max().item(),'rec-ref',(orc.float()-ref).abs().max().item(),'ref absmax',ref.abs().max().item())

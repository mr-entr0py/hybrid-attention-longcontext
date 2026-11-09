import torch, time, warnings; warnings.filterwarnings('ignore')
import torch.nn as nn, torch.nn.functional as F
from fla.layers import GatedDeltaNet, GatedLinearAttention
dev='cuda'; d=512; dt=torch.bfloat16
class Attn(nn.Module):
    def __init__(s,d,h=8):
        super().__init__(); s.h=h; s.qkv=nn.Linear(d,3*d,bias=False); s.o=nn.Linear(d,d,bias=False)
    def forward(s,x):
        B,T,_=x.shape; q,k,v=s.qkv(x).view(B,T,3,s.h,-1).permute(2,0,3,1,4)
        return s.o(F.scaled_dot_product_attention(q,k,v,is_causal=True).transpose(1,2).reshape(B,T,-1))
def mk(name):
    if name=='attn': return Attn(d)
    if name=='gdn': return GatedDeltaNet(hidden_size=d, head_dim=64, num_heads=4, expand_v=2)
    if name=='gla': return GatedLinearAttention(hidden_size=d, num_heads=4)
def run(m,x):
    y=m(x); y=y[0] if isinstance(y,tuple) else y; return y
for T in [2048,8192,32768]:
    B=max(1,32768//T)
    x=torch.randn(B,T,d,device=dev,dtype=dt,requires_grad=True)
    for name in ['attn','gdn','gla']:
        m=mk(name).to(dev,dt)
        for _ in range(3): run(m,x).sum().backward()
        torch.cuda.synchronize(); t=time.time(); n=10
        for _ in range(n): run(m,x).sum().backward()
        torch.cuda.synchronize(); ms=(time.time()-t)/n*1e3
        print(f'T={T:6d} B={B:2d} {name:5s} fwd+bwd {ms:7.2f} ms  ({B*T/ms*1e3/1e6:.2f} Mtok/s per layer)')

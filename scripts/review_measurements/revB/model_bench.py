import sys, time, math, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.nn.attention.varlen import varlen_attn
from fla.layers import GatedDeltaNet
torch.cuda.set_per_process_memory_fraction(0.46)   # ~7.1 GiB cap (CARLA shares the GPU)
dev='cuda'; d=512; L=12; V=16384; FF=1536; H=8; Dh=64
class Attn(nn.Module):
    def __init__(s):
        super().__init__(); s.qkv=nn.Linear(d,3*d,bias=False); s.o=nn.Linear(d,d,bias=False)
        s.qn=nn.Parameter(torch.ones(Dh)); s.kn=nn.Parameter(torch.ones(Dh))
    def forward(s,x,cu,maxlen,pos):
        T=x.shape[0]; q,k,v=s.qkv(x).view(T,3,H,Dh).unbind(1)
        q=F.rms_norm(q,(Dh,),s.qn); k=F.rms_norm(k,(Dh,),s.kn)
        inv=1.0/(1e4**(torch.arange(0,Dh,2,device=x.device,dtype=torch.float32)/Dh)); ang=pos[:,None].float()*inv[None]
        c,sn=ang.cos()[:,None,:].to(q.dtype),ang.sin()[:,None,:].to(q.dtype)
        def rope(t): t1,t2=t[...,::2],t[...,1::2]; return torch.stack((t1*c-t2*sn,t1*sn+t2*c),-1).flatten(-2)
        o=varlen_attn(rope(q),rope(k),v,cu,cu,maxlen,maxlen,window_size=(-1,0))
        return s.o(o.reshape(T,d))
class GDN(nn.Module):
    def __init__(s):
        super().__init__(); s.m=GatedDeltaNet(hidden_size=d,head_dim=64,num_heads=4,expand_v=2,use_gate=True,use_short_conv=True,conv_size=4,mode='chunk')
    def forward(s,x,cu,maxlen,pos): return s.m(x[None],cu_seqlens=cu.long())[0][0]
class Block(nn.Module):
    def __init__(s,attn):
        super().__init__(); s.n1=nn.RMSNorm(d); s.n2=nn.RMSNorm(d); s.mix=Attn() if attn else GDN()
        s.g=nn.Linear(d,FF,bias=False); s.u=nn.Linear(d,FF,bias=False); s.dn=nn.Linear(FF,d,bias=False)
    def forward(s,x,cu,maxlen,pos):
        x=x+s.mix(s.n1(x),cu,maxlen,pos); h=s.n2(x); return x+s.dn(F.silu(s.g(h))*s.u(h))
class LM(nn.Module):
    def __init__(s,k):
        super().__init__(); P=set([int((j+0.5)*L/k) for j in range(k)]) if k<L else set(range(L))
        s.emb=nn.Embedding(V,d); s.blocks=nn.ModuleList([Block(i in P) for i in range(L)]); s.nf=nn.RMSNorm(d)
    def forward(s,ids,tgt,cu,maxlen,pos,ckpt):
        x=s.emb(ids)
        for b in s.blocks: x=checkpoint(b,x,cu,maxlen,pos,use_reentrant=False) if ckpt else b(x,cu,maxlen,pos)
        x=s.nf(x); loss=0.0; C=4096
        for i in range(0,x.shape[0],C):   # chunked CE, logits recomputed in backward
            loss=loss+checkpoint(lambda h,t: F.cross_entropy((h@s.emb.weight.t()).float(),t,reduction='sum'),x[i:i+C],tgt[i:i+C],use_reentrant=False)
        return loss/x.shape[0]
def doc_lens(total,g):   # lognormal ~ FineWeb-Edu (median ~690, mean ~1150 of our tokens), docs split at seq boundaries
    out=[]; 
    while sum(out)<total: out.append(max(16,int(math.exp(torch.normal(6.54,1.0,(1,),generator=g).item()))))
    out[-1]-=sum(out)-total; return [x for x in out if x>0]
def bench(k,ntok,seqlen,ckpt,steps=6):
    torch.manual_seed(0); m=LM(k).to(dev); opt=torch.optim.AdamW(m.parameters(),lr=1e-3,fused=True)
    g=torch.Generator().manual_seed(0); lens=[]
    for _ in range(ntok//seqlen): lens+= (doc_lens(seqlen,g) if seqlen<=4096 else [seqlen])
    cu=torch.tensor([0]+list(torch.cumsum(torch.tensor(lens),0)),device=dev,dtype=torch.int32); maxlen=max(lens)
    pos=torch.cat([torch.arange(n,device=dev) for n in lens]); ids=torch.randint(0,V,(ntok,),device=dev); tgt=torch.randint(0,V,(ntok,),device=dev)
    torch.cuda.reset_peak_memory_stats(); ts=[]
    for i in range(steps+3):
        torch.cuda.synchronize(); t0=time.time()
        with torch.autocast('cuda',dtype=torch.bfloat16): loss=m(ids,tgt,cu,maxlen,pos,ckpt)
        loss.backward(); opt.step(); opt.zero_grad(set_to_none=True); torch.cuda.synchronize()
        if i>=3: ts.append(time.time()-t0)
    pk=torch.cuda.max_memory_allocated()/2**30; del m,opt; torch.cuda.empty_cache()
    return ntok/min(ts), ntok/sorted(ts)[len(ts)//2], pk
print('params(k=0,12):', sum(p.numel() for p in LM(0).parameters()), sum(p.numel() for p in LM(12).parameters()), flush=True)
for (k,ntok,seq,ck) in [(0,16384,4096,False),(3,16384,4096,False),(12,16384,4096,False),(12,32768,4096,False),(0,32768,32768,True),(3,32768,32768,True),(12,32768,32768,True),(12,32768,32768,False)]:
    try:
        best,med,pk=bench(k,ntok,seq,ck)
        print(f"k={k:2d} microbatch={ntok//1024}K seqlen={seq//1024}K ckpt={ck}: tok/s best {best/1e3:.1f}K median {med/1e3:.1f}K | peak alloc {pk:.2f} GiB",flush=True)
    except torch.OutOfMemoryError as e:
        print(f"k={k} microbatch={ntok//1024}K seqlen={seq//1024}K ckpt={ck}: OOM under 7.1GiB cap",flush=True); torch.cuda.empty_cache()

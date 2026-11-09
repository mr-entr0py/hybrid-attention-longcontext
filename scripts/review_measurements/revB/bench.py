import time, statistics, torch, torch.nn.functional as F
torch.cuda.set_per_process_memory_fraction(0.35)
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.nn.attention.varlen import varlen_attn
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from fla.layers import GatedDeltaNet
dev='cuda'; d=512; H=8; Dh=64
def timeit(fn, reps=25, warm=5):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); ts=[]
    for _ in range(reps):
        s=torch.cuda.Event(enable_timing=True); e=torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
    return min(ts), statistics.median(ts)
out=[]
# ---- 1. deterministic-mode cost for FA2 varlen backward (doc-packed 32K tokens = Stage-A-like micro-batch of 8x4K docs) ----
lens=[700,1500,300,4000,2500,9000,6000,8768]; T=sum(lens)
cu=torch.tensor([0]+list(torch.cumsum(torch.tensor(lens),0)),device=dev,dtype=torch.int32)
q=torch.randn(T,H,Dh,device=dev,dtype=torch.bfloat16,requires_grad=True); k=torch.randn_like(q,requires_grad=True); v=torch.randn_like(q,requires_grad=True); do=torch.randn_like(q)
def fb():
    o=varlen_attn(q,k,v,cu,cu,max(lens),max(lens),window_size=(-1,0)); o.backward(do)
m0=timeit(fb); torch.use_deterministic_algorithms(True); m1=timeit(fb); torch.use_deterministic_algorithms(False)
out.append(f"varlen_attn fwd+bwd 32K doc-packed: default min/med {m0[0]:.2f}/{m0[1]:.2f} ms ; deterministic {m1[0]:.2f}/{m1[1]:.2f} ms ; overhead(min) {100*(m1[0]/m0[0]-1):.0f}%")
lens2=[4096]*8; cu2=torch.tensor([0]+list(torch.cumsum(torch.tensor(lens2),0)),device=dev,dtype=torch.int32)
def fb2():
    o=varlen_attn(q,k,v,cu2,cu2,4096,4096,window_size=(-1,0)); o.backward(do)
m0=timeit(fb2); torch.use_deterministic_algorithms(True); m1=timeit(fb2); torch.use_deterministic_algorithms(False)
out.append(f"varlen_attn fwd+bwd 8x4K: default {m0[0]:.2f}/{m0[1]:.2f} ms ; deterministic {m1[0]:.2f}/{m1[1]:.2f} ms ; overhead(min) {100*(m1[0]/m0[0]-1):.0f}%")
qs=torch.randn(1,H,32768,Dh,device=dev,dtype=torch.bfloat16,requires_grad=True); ks=torch.randn_like(qs,requires_grad=True); vs=torch.randn_like(qs,requires_grad=True); dos=torch.randn_like(qs)
def fb3():
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION): o=F.scaled_dot_product_attention(qs,ks,vs,is_causal=True)
    o.backward(dos)
m0=timeit(fb3,reps=10); torch.use_deterministic_algorithms(True); m1=timeit(fb3,reps=10); torch.use_deterministic_algorithms(False)
out.append(f"SDPA-flash fwd+bwd single 32K seq: default {m0[0]:.2f}/{m0[1]:.2f} ms ; deterministic {m1[0]:.2f}/{m1[1]:.2f} ms ; overhead(min) {100*(m1[0]/m0[0]-1):.0f}%")
del q,k,v,do,qs,ks,vs,dos
# ---- 2. prefill (forward, no grad) per mixer core, batch 1 ----
def causal_mod(b,h,qi,ki): return qi>=ki
for line in out: print(line, flush=True)
out=[]
gdn=GatedDeltaNet(hidden_size=d,head_dim=64,num_heads=4,expand_v=2,use_gate=True,use_short_conv=True,conv_size=4,mode='chunk').to(dev).to(torch.bfloat16).eval()
for Tq in (4096, 32768):
    qq=torch.randn(1,H,Tq,Dh,device=dev,dtype=torch.bfloat16); kk=torch.randn_like(qq); vv=torch.randn_like(qq)
    flops=4*Tq*Tq*Dh*H/2  # causal QK^T + PV
    res={}
    with torch.no_grad():
        for name,be in [('flash',SDPBackend.FLASH_ATTENTION),('cudnn',SDPBackend.CUDNN_ATTENTION),('efficient',SDPBackend.EFFICIENT_ATTENTION)]:
            def f():
                with sdpa_kernel(be): F.scaled_dot_product_attention(qq,kk,vv,is_causal=True)
            try: res[name]=timeit(f)
            except Exception as e: res[name]=(float('nan'),float('nan'))
        bm=create_block_mask(causal_mod, 1,1,Tq,Tq,device=dev,_compile=True)
        fx=torch.compile(flex_attention)
        def f(): fx(qq,kk,vv,block_mask=bm)
        try: res['flex(triton)']=timeit(f)
        except Exception as e: res['flex(triton)']=(float('nan'),float('nan')); print('flex err',e)
        x=torch.randn(1,Tq,d,device=dev,dtype=torch.bfloat16)
        def f(): gdn(x)
        res['GDN full layer (proj+conv+chunk+gate)']=timeit(f)
    s=' ; '.join(f"{n} {m[0]:.2f}/{m[1]:.2f}ms" for n,m in res.items())
    out.append(f"prefill T={Tq}: {s}  | attn-core TFLOP/s(flash,min)={flops/res['flash'][0]/1e9:.1f}")
for line in out: print(line)

import statistics, torch, torch.nn.functional as F
torch.cuda.set_per_process_memory_fraction(0.35)
from torch.nn.attention import sdpa_kernel, SDPBackend
from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
dev='cuda'; d=512; H=8; Dh=64; HG=4; K=64; V=128
def timeit(fn, reps=30, warm=5):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); ts=[]
    for _ in range(reps):
        s=torch.cuda.Event(enable_timing=True); e=torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
    return min(ts), statistics.median(ts)
def graph(fn):
    s=torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): fn()
    torch.cuda.current_stream().wait_stream(s)
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): fn()
    return g.replay
with torch.no_grad():
    # GDN core chunk kernel alone (prefill)
    for T in (4096, 32768):
        q=F.normalize(torch.randn(1,T,HG,K,device=dev,dtype=torch.bfloat16),dim=-1); k=F.normalize(torch.randn_like(q),dim=-1)
        v=torch.randn(1,T,HG,V,device=dev,dtype=torch.bfloat16); g=-torch.rand(1,T,HG,device=dev)*0.1; b=torch.rand(1,T,HG,device=dev,dtype=torch.bfloat16)
        m=timeit(lambda: chunk_gated_delta_rule(q,k,v,g=g,beta=b))
        W=torch.randn(d,4*d+8,device=dev,dtype=torch.bfloat16); x=torch.randn(T,d,device=dev,dtype=torch.bfloat16)
        mp=timeit(lambda: x@W)
        print(f"prefill T={T}: GDN chunk core {m[0]:.2f}/{m[1]:.2f} ms ; 4d^2 projection GEMMs {mp[0]:.2f}/{mp[1]:.2f} ms ({2*T*d*(4*d+8)/mp[0]/1e9:.0f} TFLOP/s)", flush=True)
    # Decode step: one new token, cache length L, CUDA graph, batch B
    for B in (1, 8):
        for L in (4096, 32768):
            kc=torch.randn(B,H,L,Dh,device=dev,dtype=torch.bfloat16); vc=torch.randn_like(kc); qd=torch.randn(B,H,1,Dh,device=dev,dtype=torch.bfloat16)
            res={}
            for name,be in [('flash',SDPBackend.FLASH_ATTENTION),('cudnn',SDPBackend.CUDNN_ATTENTION),('efficient',SDPBackend.EFFICIENT_ATTENTION),('math',SDPBackend.MATH)]:
                def f():
                    with sdpa_kernel(be): F.scaled_dot_product_attention(qd,kc,vc)
                try: res[name]=timeit(graph(f))
                except Exception as e: res[name]=(float('nan'),float('nan'))
            # static-cache variant: boolean mask over a preallocated (L+256) buffer -> flash not eligible
            kcs=torch.randn(B,H,L+256,Dh,device=dev,dtype=torch.bfloat16); vcs=torch.randn_like(kcs)
            mask=torch.zeros(B,1,1,L+256,device=dev,dtype=torch.bool); mask[...,:L+1]=True
            def f():
                F.scaled_dot_product_attention(qd,kcs,vcs,attn_mask=mask)
            try: res['static+mask(auto)']=timeit(graph(f))
            except Exception as e: res['static+mask(auto)']=(float('nan'),float('nan')); print(e)
            bytes_kv=2*B*H*L*Dh*2
            s=' ; '.join(f"{n} {m[0]*1000:.0f}/{m[1]*1000:.0f}us" for n,m in res.items())
            print(f"decode attn-core B={B} L={L}: {s} | KV {bytes_kv/2**20:.0f}MiB -> roofline @960GB/s {bytes_kv/960e9*1e6:.0f}us", flush=True)
        # GDN decode step core
        qg=F.normalize(torch.randn(B,1,HG,K,device=dev,dtype=torch.bfloat16),dim=-1); kg=F.normalize(torch.randn_like(qg),dim=-1)
        vg=torch.randn(B,1,HG,V,device=dev,dtype=torch.bfloat16); gg=-torch.rand(B,1,HG,device=dev)*0.1; bg=torch.rand(B,1,HG,device=dev,dtype=torch.bfloat16)
        h0=torch.randn(B,HG,K,V,device=dev,dtype=torch.float32); hs=torch.empty_like(h0)
        def f():
            o,ht=fused_recurrent_gated_delta_rule(q=qg,k=kg,v=vg,g=gg,beta=bg,initial_state=h0,output_final_state=True); hs.copy_(ht)
        mg=timeit(graph(f))
        print(f"decode GDN-core B={B}: fused_recurrent+state copy {mg[0]*1000:.0f}/{mg[1]*1000:.0f}us | state {B*HG*K*V*4/2**10:.0f}KiB", flush=True)

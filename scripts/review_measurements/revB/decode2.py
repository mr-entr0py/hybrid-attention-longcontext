import statistics, torch, torch.nn.functional as F
torch.cuda.set_per_process_memory_fraction(0.35)
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
dev='cuda'; H=8; Dh=64; L=32768
flush=torch.empty(512*2**20,dtype=torch.uint8,device=dev)
def graph(fn):
    s=torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): fn()
    torch.cuda.current_stream().wait_stream(s); g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): fn()
    return g.replay
def t(fn,cold,reps=30):
    for _ in range(5): fn()
    ts=[]
    for _ in range(reps):
        if cold: flush.zero_()
        s=torch.cuda.Event(enable_timing=True); e=torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e)*1000)
    return min(ts)
with torch.no_grad():
    kc=torch.randn(1,H,L+256,Dh,device=dev,dtype=torch.bfloat16); vc=torch.randn_like(kc); q=torch.randn(1,H,1,Dh,device=dev,dtype=torch.bfloat16)
    mask=torch.zeros(1,1,1,L+256,device=dev,dtype=torch.bool); mask[...,:L+1]=True
    for name,be in [('FLASH',SDPBackend.FLASH_ATTENTION),('CUDNN',SDPBackend.CUDNN_ATTENTION),('EFFICIENT',SDPBackend.EFFICIENT_ATTENTION)]:
        def f():
            with sdpa_kernel(be): F.scaled_dot_product_attention(q,kc,vc,attn_mask=mask)
        try: f(); print(f"static-cache+bool-mask with {name}: runs, graphed cold {t(graph(f),True):.0f}us")
        except Exception as e: print(f"static-cache+bool-mask with {name}: {str(e).splitlines()[0][:90]}")
    # exact-shape (no mask) graphs, cold L2
    k2=kc[:,:,:L].contiguous(); v2=vc[:,:,:L].contiguous()
    for name,be in [('FLASH',SDPBackend.FLASH_ATTENTION),('CUDNN',SDPBackend.CUDNN_ATTENTION)]:
        def f():
            with sdpa_kernel(be): F.scaled_dot_product_attention(q,k2,v2)
        print(f"exact-shape {name}: graphed cold {t(graph(f),True):.0f}us")
    # FlexAttention decode (Triton-generated), static buffer + length mask via block mask
    Lcur=torch.tensor(L+1,device=dev)
    def mm(b,h,qi,ki): return ki<Lcur
    bm=create_block_mask(mm,1,1,1,L+256,device=dev,_compile=True)
    fx=torch.compile(flex_attention)
    def f(): fx(q,kc,vc,block_mask=bm)
    try: print(f"FlexAttention decode (static buffer, length via mask_mod): eager-call cold {t(f,True):.0f}us ; graphed cold {t(graph(f),True):.0f}us")
    except Exception as e: print('flex decode error:', str(e)[:200])

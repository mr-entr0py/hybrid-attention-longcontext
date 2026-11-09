import torch, torch.nn.functional as F
torch.cuda.set_per_process_memory_fraction(0.35)
from torch.nn.attention import sdpa_kernel, SDPBackend
dev='cuda'; flush=torch.empty(512*2**20,dtype=torch.uint8,device=dev)
def t(fn,reps=30):
    for _ in range(5): fn()
    ts=[]
    for _ in range(reps):
        flush.zero_(); s=torch.cuda.Event(enable_timing=True); e=torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e)*1000)
    return min(ts)
print('flash/mem_eff/cudnn/math enabled:', torch.backends.cuda.flash_sdp_enabled(), torch.backends.cuda.mem_efficient_sdp_enabled(), torch.backends.cuda.cudnn_sdp_enabled(), torch.backends.cuda.math_sdp_enabled())
with torch.no_grad():
    L=32768
    q=torch.randn(1,4,1,64,device=dev,dtype=torch.bfloat16); k=torch.randn(1,4,L,64,device=dev,dtype=torch.bfloat16); v=torch.randn(1,4,L,128,device=dev,dtype=torch.bfloat16)
    print(f"default dispatch, Ev!=E decode L=32K: {t(lambda: F.scaled_dot_product_attention(q,k,v)):.0f}us (eager, cold L2)")
    for name,be in [('CUDNN',SDPBackend.CUDNN_ATTENTION),('EFFICIENT',SDPBackend.EFFICIENT_ATTENTION)]:
        def f(be=be):
            with sdpa_kernel(be): return F.scaled_dot_product_attention(q,k,v)
        print(f"  pinned {name}: {t(f):.0f}us (eager, cold L2)")
    q8=torch.randn(1,8,1,64,device=dev,dtype=torch.bfloat16); k8=torch.randn(1,8,L,64,device=dev,dtype=torch.bfloat16); v8=torch.randn_like(k8)
    print(f"default dispatch, plain 8x64 decode L=32K: {t(lambda: F.scaled_dot_product_attention(q8,k8,v8)):.0f}us (eager, cold L2)")
    # prefill with Ev != E, causal, T=32K (what the evaluation prefill would hit)
    T=32768; qp=torch.randn(1,4,T,64,device=dev,dtype=torch.bfloat16); kp=torch.randn_like(qp); vp=torch.randn(1,4,T,128,device=dev,dtype=torch.bfloat16)
    print(f"default dispatch, Ev!=E causal prefill T=32K: {t(lambda: F.scaled_dot_product_attention(qp,kp,vp,is_causal=True),reps=10)/1000:.2f}ms")
    for name,be in [('CUDNN',SDPBackend.CUDNN_ATTENTION),('EFFICIENT',SDPBackend.EFFICIENT_ATTENTION)]:
        def f(be=be):
            with sdpa_kernel(be): return F.scaled_dot_product_attention(qp,kp,vp,is_causal=True)
        print(f"  pinned {name} prefill: {t(f,reps=10)/1000:.2f}ms")

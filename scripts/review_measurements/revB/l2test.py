import statistics, torch, torch.nn.functional as F
torch.cuda.set_per_process_memory_fraction(0.35)
from torch.nn.attention import sdpa_kernel, SDPBackend
p = torch.cuda.get_device_properties(0)
print('name', p.name, '| SMs', p.multi_processor_count, '| L2 bytes', getattr(p, 'L2_cache_size', 'n/a'), '| mem', round(p.total_memory/2**30,2), 'GiB')
dev='cuda'; H=8; Dh=64
flush = torch.empty(512*2**20, dtype=torch.uint8, device=dev)
def t(fn, l2flush, reps=30):
    for _ in range(5): fn()
    ts=[]
    for _ in range(reps):
        if l2flush: flush.zero_()
        s=torch.cuda.Event(enable_timing=True); e=torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e)*1000)
    return min(ts), statistics.median(ts)
with torch.no_grad():
    for L in (4096, 32768):
        kc=torch.randn(1,H,L,Dh,device=dev,dtype=torch.bfloat16); vc=torch.randn_like(kc); q=torch.randn(1,H,1,Dh,device=dev,dtype=torch.bfloat16)
        for name,be in [('flash',SDPBackend.FLASH_ATTENTION),('cudnn',SDPBackend.CUDNN_ATTENTION)]:
            def f():
                with sdpa_kernel(be): F.scaled_dot_product_attention(q,kc,vc)
            a=t(f,False); b=t(f,True)
            print(f"B=1 L={L} {name}: hot-L2 {a[0]:.0f}/{a[1]:.0f}us  flushed-L2 {b[0]:.0f}/{b[1]:.0f}us  (KV {2*L*H*Dh*2/2**20:.0f} MiB, DRAM roofline {2*L*H*Dh*2/960e9*1e6:.0f}us)")

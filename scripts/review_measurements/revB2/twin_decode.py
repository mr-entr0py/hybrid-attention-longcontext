# Twin attention decode at context L: native d_v != d_qk (4 heads, K 64 / V 128 -> 1.5 KiB/token) via cuDNN / EFFICIENT,
# vs duplicated 8x64 (K duplicated -> 2 KiB/token) via flash/cuDNN, vs plain MHA 8x64 (2 KiB/token). Exact-shape graphs, cold L2.
import torch, torch.nn.functional as F
torch.cuda.set_per_process_memory_fraction(0.35)
from torch.nn.attention import sdpa_kernel, SDPBackend
dev = 'cuda'; flush = torch.empty(512 * 2**20, dtype=torch.uint8, device=dev)
def graph(fn):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): fn()
    torch.cuda.current_stream().wait_stream(s); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): fn()
    return g.replay
def t(fn, cold, reps=40):
    for _ in range(5): fn()
    ts = []
    for _ in range(reps):
        if cold: flush.zero_()
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e) * 1000)
    return min(ts)
with torch.no_grad():
    for L in (4096, 32768):
        for B in (1, 8):
            qn = torch.randn(B, 4, 1, 64, device=dev, dtype=torch.bfloat16); kn = torch.randn(B, 4, L, 64, device=dev, dtype=torch.bfloat16); vn = torch.randn(B, 4, L, 128, device=dev, dtype=torch.bfloat16)
            qd = qn.repeat_interleave(2, 1); kd = kn.repeat_interleave(2, 1).contiguous(); vd = vn.view(B, 4, L, 2, 64).permute(0, 1, 3, 2, 4).reshape(B, 8, L, 64).contiguous()
            ref = F.scaled_dot_product_attention(qn.float(), kn.float(), vn.float())
            out = {}
            for name, be, args in [('native-cudnn', SDPBackend.CUDNN_ATTENTION, (qn, kn, vn)), ('native-efficient', SDPBackend.EFFICIENT_ATTENTION, (qn, kn, vn)),
                                   ('dup8x64-flash', SDPBackend.FLASH_ATTENTION, (qd, kd, vd)), ('dup8x64-cudnn', SDPBackend.CUDNN_ATTENTION, (qd, kd, vd))]:
                def f(be=be, args=args):
                    with sdpa_kernel(be): return F.scaled_dot_product_attention(*args)
                try:
                    o = f(); o = o if name.startswith('native') else o.view(B, 4, 2, 1, 64).permute(0, 1, 3, 2, 4).reshape(B, 4, 1, 128)
                    err = (o.float() - ref).abs().max().item()
                    out[name] = f"{t(graph(f), True):.0f}us (err {err:.1e})"
                except Exception as ex: out[name] = 'FAIL ' + str(ex).splitlines()[0][:60]
            kvn = B * L * 4 * (64 + 128) * 2; kvd = B * L * 8 * (64 + 64) * 2
            print(f"L={L} B={B}: KV native {kvn/2**20:.0f} MiB (roofline {kvn/960e9*1e6:.0f}us) vs duplicated {kvd/2**20:.0f} MiB (roofline {kvd/960e9*1e6:.0f}us) | " + ' | '.join(f"{k}: {v}" for k, v in out.items()), flush=True)

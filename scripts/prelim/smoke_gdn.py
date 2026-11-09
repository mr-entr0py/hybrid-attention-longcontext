import time, torch, torch.nn.functional as F
torch.cuda.set_per_process_memory_fraction(0.12)  # ~2GB cap on 16GB card
dev, dt = 'cuda', torch.bfloat16
import fla, triton
from fla.layers import GatedDeltaNet
print("fla", fla.__version__, "triton", triton.__version__, "torch", torch.__version__)
torch.manual_seed(0)
B, T, D = 2, 2048, 512
# GDN layer sized as in a ~50M model: H=4 heads, head_dim=128 (key_dim=512=D), expand_v=1, no gate -> 4D^2 params like attention
layer = GatedDeltaNet(hidden_size=D, num_heads=4, head_dim=128, expand_v=1.0, use_gate=False, mode='chunk', layer_idx=0).to(dev, dt)
print("GDN params", sum(p.numel() for p in layer.parameters()))
x = torch.randn(B, T, D, device=dev, dtype=dt, requires_grad=True)
t0=time.time(); y, _, _ = layer(x); y.float().pow(2).mean().backward(); torch.cuda.synchronize()
print("GDN chunk fwd/bwd ok (first call incl. autotune+JIT) %.1fs"%(time.time()-t0), y.shape, "grad finite", torch.isfinite(x.grad).all().item())
t0=time.time()
for _ in range(5):
    x.grad=None; y,_,_ = layer(x); y.float().pow(2).mean().backward()
torch.cuda.synchronize(); print("GDN fwd/bwd steady %.2f ms/iter"%((time.time()-t0)/5*1e3))
# Varlen / document masking via cu_seqlens: batch must be 1, packed
cu = torch.tensor([0, 700, 1500, 4096], device=dev, dtype=torch.int32)
xp = torch.randn(1, 4096, D, device=dev, dtype=dt, requires_grad=True)
yp,_,_ = layer(xp, cu_seqlens=cu); yp.float().sum().backward(); torch.cuda.synchronize()
# check isolation: output of doc 2 equals running doc 2 alone
with torch.no_grad():
    y2,_,_ = layer(xp[:, 700:1500].detach())
print("varlen ok; doc-isolation max abs diff", (yp[:,700:1500].detach().float()-y2.float()).abs().max().item())
# Chunk vs fused_recurrent consistency + cached decode
from fla.models.utils import Cache
layer.eval()
with torch.no_grad():
    xs = torch.randn(1, 256, D, device=dev, dtype=dt)
    yfull,_,_ = layer(xs)
    cache = Cache()
    ypre,_,cache = layer(xs[:, :200], past_key_values=cache, use_cache=True)
    outs=[ypre]
    for t in range(200,256):
        yt,_,cache = layer(xs[:, t:t+1], past_key_values=cache, use_cache=True); outs.append(yt)
    ydec = torch.cat(outs,1)
    print("prefill+recurrent decode vs full chunk: max abs diff", (ydec.float()-yfull.float()).abs().max().item(), "rel", ((ydec.float()-yfull.float()).norm()/yfull.float().norm()).item())
    st = cache[0]
    rs = st['recurrent_state']; print("recurrent_state", tuple(rs.shape), rs.dtype, "bytes", rs.numel()*rs.element_size(), "conv_state shapes", [tuple(c.shape) for c in st['conv_state']])
# SDPA causal (flash backend) fwd/bwd at bf16
from torch.nn.attention import sdpa_kernel, SDPBackend
q = torch.randn(B, 8, T, 64, device=dev, dtype=dt, requires_grad=True)
for be in [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.CUDNN_ATTENTION]:
    try:
        with sdpa_kernel(be):
            o = F.scaled_dot_product_attention(q,q,q,is_causal=True); o.sum().backward(); torch.cuda.synchronize()
            t0=time.time()
            for _ in range(10):
                o = F.scaled_dot_product_attention(q,q,q,is_causal=True); o.sum().backward()
            torch.cuda.synchronize()
        print("SDPA", be.name, "ok %.2f ms/iter fwd+bwd"%((time.time()-t0)/10*1e3))
    except Exception as e:
        print("SDPA", be.name, "FAILED", type(e).__name__, str(e)[:200])
print("peak mem MB", torch.cuda.max_memory_allocated()/2**20)

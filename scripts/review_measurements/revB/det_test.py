import os, time, torch, torch.nn.functional as F
torch.cuda.set_per_process_memory_fraction(0.30)
dev = 'cuda'
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.nn.attention.varlen import varlen_attn
res = {}
def bitwise(a, b): return all(torch.equal(x, y) for x, y in zip(a, b))
def maxdiff(a, b): return max((x.float()-y.float()).abs().max().item() for x, y in zip(a, b))

# ---------- 1. varlen_attn (FA2 inside torch) backward determinism ----------
torch.manual_seed(0)
H, D = 8, 64
lens = [700, 1500, 300, 4000, 2500, 9000, 6000, 8768]   # 32768 tokens, doc-packed
T = sum(lens)
cu = torch.tensor([0] + list(torch.cumsum(torch.tensor(lens), 0)), device=dev, dtype=torch.int32)
q = torch.randn(T, H, D, device=dev, dtype=torch.bfloat16, requires_grad=True)
k = torch.randn(T, H, D, device=dev, dtype=torch.bfloat16, requires_grad=True)
v = torch.randn(T, H, D, device=dev, dtype=torch.bfloat16, requires_grad=True)
do = torch.randn(T, H, D, device=dev, dtype=torch.bfloat16)
def run_varlen():
    for t in (q, k, v): t.grad = None
    o = varlen_attn(q, k, v, cu, cu, max(lens), max(lens), window_size=(-1, 0))
    o.backward(do)
    return [o.detach().clone(), q.grad.clone(), k.grad.clone(), v.grad.clone()]
a = run_varlen(); b = run_varlen()
res['varlen_attn fwd bitwise'] = torch.equal(a[0], b[0])
res['varlen_attn bwd bitwise (default)'] = bitwise(a[1:], b[1:]); res['varlen_attn bwd maxdiff'] = maxdiff(a[1:], b[1:])
try:
    torch.use_deterministic_algorithms(True)
    c = run_varlen(); d = run_varlen()
    res['varlen_attn bwd bitwise (deterministic mode)'] = bitwise(c[1:], d[1:])
except Exception as e:
    res['varlen_attn under use_deterministic_algorithms'] = 'RAISES: ' + str(e)[:160]
finally:
    torch.use_deterministic_algorithms(False)

# ---------- 2. SDPA backends backward determinism (causal, dense) ----------
qd = torch.randn(1, H, 16384, D, device=dev, dtype=torch.bfloat16, requires_grad=True)
kd = torch.randn_like(qd, requires_grad=True); vd = torch.randn_like(qd, requires_grad=True); dod = torch.randn_like(qd)
for name, be in [('FLASH', SDPBackend.FLASH_ATTENTION), ('CUDNN', SDPBackend.CUDNN_ATTENTION), ('EFFICIENT', SDPBackend.EFFICIENT_ATTENTION)]:
    def run():
        for t in (qd, kd, vd): t.grad = None
        with sdpa_kernel(be):
            o = F.scaled_dot_product_attention(qd, kd, vd, is_causal=True)
        o.backward(dod); return [qd.grad.clone(), kd.grad.clone(), vd.grad.clone()]
    try:
        a = run(); b = run()
        res[f'SDPA {name} bwd bitwise'] = bitwise(a, b); res[f'SDPA {name} bwd maxdiff'] = maxdiff(a, b)
    except Exception as e:
        res[f'SDPA {name}'] = 'ERR ' + str(e)[:120]
    try:
        torch.use_deterministic_algorithms(True)
        a = run(); b = run(); res[f'SDPA {name} bwd bitwise (det mode)'] = bitwise(a, b)
    except Exception as e:
        res[f'SDPA {name} under det mode'] = 'RAISES: ' + str(e)[:140]
    finally:
        torch.use_deterministic_algorithms(False)

# ---------- 3. fla GatedDeltaNet layer (conv + norm-gate + chunk) backward determinism, varlen ----------
from fla.layers import GatedDeltaNet
torch.manual_seed(0)
layer = GatedDeltaNet(hidden_size=512, head_dim=64, num_heads=4, expand_v=2, use_gate=True, use_short_conv=True, conv_size=4, mode='chunk').to(dev)
nparam = sum(p.numel() for p in layer.parameters())
x = torch.randn(1, T, 512, device=dev, dtype=torch.bfloat16, requires_grad=True)
dy = torch.randn(1, T, 512, device=dev, dtype=torch.bfloat16)
def run_gdn():
    layer.zero_grad(set_to_none=True); x.grad = None
    with torch.autocast('cuda', dtype=torch.bfloat16):
        y = layer(x, cu_seqlens=cu.long())[0]
    y.backward(dy)
    return [y.detach().clone(), x.grad.clone()] + [p.grad.clone() for p in layer.parameters()]
t0 = time.time(); a = run_gdn(); res['GDN first call s'] = round(time.time()-t0, 1)
b = run_gdn()
res['GDN params'] = nparam
res['GDN layer fwd bitwise'] = torch.equal(a[0], b[0])
res['GDN layer bwd (x + all param grads) bitwise'] = bitwise(a[1:], b[1:]); res['GDN bwd maxdiff'] = maxdiff(a[1:], b[1:])
for kk, vv in res.items(): print(f'{kk}: {vv}')

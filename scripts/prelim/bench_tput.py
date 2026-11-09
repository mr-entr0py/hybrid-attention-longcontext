"""Rough training-throughput microbenchmark for hybrid GDN/attention LMs.
Run under GPU contention -> numbers are LOWER BOUNDS. Not a project artifact."""
import sys, time, math, json
import torch, torch.nn as nn, torch.nn.functional as F
from fla.layers import GatedDeltaNet

torch.backends.cuda.matmul.allow_tf32 = True
dev = "cuda"

class RMSNorm(nn.Module):
    def __init__(s, d): super().__init__(); s.w = nn.Parameter(torch.ones(d))
    def forward(s, x): return F.rms_norm(x, (x.shape[-1],), s.w, 1e-6)

def rope(x, base=10000.0):
    B, H, T, D = x.shape
    inv = 1.0 / (base ** (torch.arange(0, D, 2, device=x.device).float() / D))
    t = torch.arange(T, device=x.device).float()
    f = torch.outer(t, inv)
    cos, sin = f.cos().to(x.dtype), f.sin().to(x.dtype)
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), -1).flatten(-2)

class Attn(nn.Module):
    def __init__(s, d, h):
        super().__init__(); s.h = h; s.qkv = nn.Linear(d, 3 * d, bias=False); s.o = nn.Linear(d, d, bias=False)
        s.qn = RMSNorm(d // h); s.kn = RMSNorm(d // h)
    def forward(s, x):
        B, T, D = x.shape
        q, k, v = s.qkv(x).view(B, T, 3, s.h, D // s.h).permute(2, 0, 3, 1, 4)
        q, k = rope(s.qn(q)), rope(s.kn(k))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return s.o(y.transpose(1, 2).reshape(B, T, D))

class MLP(nn.Module):
    def __init__(s, d, hid):
        super().__init__(); s.up = nn.Linear(d, 2 * hid, bias=False); s.down = nn.Linear(hid, d, bias=False)
    def forward(s, x):
        a, b = s.up(x).chunk(2, -1); return s.down(F.silu(a) * b)

class Block(nn.Module):
    def __init__(s, d, kind, i):
        super().__init__(); s.n1 = RMSNorm(d); s.n2 = RMSNorm(d); s.kind = kind
        if kind == "A":
            s.mix = Attn(d, d // 64); hid = 1536
        else:
            s.mix = GatedDeltaNet(hidden_size=d, expand_v=1, head_dim=128, num_heads=d // 128, layer_idx=i, mode="chunk"); hid = 1344
        s.mlp = MLP(d, hid)
    def forward(s, x):
        m = s.mix(s.n1(x))
        if isinstance(m, tuple): m = m[0]
        x = x + m
        return x + s.mlp(s.n2(x))

class LM(nn.Module):
    def __init__(s, pattern, d=512, V=32768):
        super().__init__(); s.emb = nn.Embedding(V, d)
        s.blocks = nn.ModuleList([Block(d, k, i) for i, k in enumerate(pattern)])
        s.nf = RMSNorm(d); s.head = nn.Linear(d, V, bias=False)
    def forward(s, idx, tgt):
        x = s.emb(idx)
        for b in s.blocks: x = b(x)
        return F.cross_entropy(s.head(s.nf(x)).float().view(-1, s.head.out_features), tgt.view(-1))

def run(pattern, T, B, steps=12, compile_=True):
    torch.manual_seed(0)
    m = LM(pattern).to(dev)
    n_all = sum(p.numel() for p in m.parameters()); n_emb = m.emb.weight.numel() + m.head.weight.numel()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, fused=True)
    if compile_:
        for b in m.blocks:
            if b.kind == "A": b.mix = torch.compile(b.mix)
            b.mlp = torch.compile(b.mlp)
    idx = torch.randint(0, 32768, (B, T), device=dev); tgt = torch.randint(0, 32768, (B, T), device=dev)
    torch.cuda.reset_peak_memory_stats()
    ts = []
    for i in range(steps):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = m(idx, tgt)
        loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
    dt = sorted(ts[4:])[len(ts[4:]) // 2]
    return dict(pattern="".join(pattern), T=T, B=B, params_total=n_all, params_nonemb=n_all - n_emb,
                tok_per_s=round(B * T / dt), peak_mem_GB=round(torch.cuda.max_memory_allocated() / 1e9, 2))

if __name__ == "__main__":
    out = []
    L = 12
    pats = {"r=1": ["A"] * L, "r=0": ["G"] * L, "r=1/4": ["G", "G", "G", "A"] * 3}
    for name, p in pats.items():
        for T, B in [(4096, 1), (8192, 1)]:
            try:
                r = run(p, T, B); r["name"] = name; out.append(r); print(json.dumps(r), flush=True)
            except Exception as e:
                print(json.dumps(dict(name=name, T=T, err=str(e)[:300])), flush=True)
            torch.cuda.empty_cache()

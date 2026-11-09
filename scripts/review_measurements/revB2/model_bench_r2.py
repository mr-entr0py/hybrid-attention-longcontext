# Round-2 whole-model throughput: D384 vs D512, plain MHA vs Reviewer A's "twin" attention,
# GDN-A vs GDN-E (1 head, d_k=d/2), GLA-twin. Contended GPU (CARLA). Min-of-N estimator.
import sys, time, math, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.nn.attention.varlen import varlen_attn
from fla.layers import GatedDeltaNet, GatedLinearAttention
from fla.modules import ShortConvolution, FusedRMSNormGated
torch.cuda.set_per_process_memory_fraction(0.40)
dev = 'cuda'; L = 12; V = 16384

def rope_tables(pos, Dh, theta=5e5):
    inv = 1.0 / (theta ** (torch.arange(0, Dh, 2, device=pos.device, dtype=torch.float32) / Dh))
    ang = pos[:, None].float() * inv[None]
    return ang.cos()[:, None, :], ang.sin()[:, None, :]
def rope(t, c, s):
    c = c.to(t.dtype); s = s.to(t.dtype); t1, t2 = t[..., ::2], t[..., 1::2]
    return torch.stack((t1 * c - t2 * s, t1 * s + t2 * c), -1).flatten(-2)

class Plain(nn.Module):          # MHA (d/64) x 64, QK-RMSNorm, RoPE, no conv, no gate
    def __init__(s, d):
        super().__init__(); s.d = d; s.H = d // 64
        s.qkv = nn.Linear(d, 3 * d, bias=False); s.o = nn.Linear(d, d, bias=False)
        s.qn = nn.Parameter(torch.ones(64)); s.kn = nn.Parameter(torch.ones(64))
    def forward(s, x, cu, maxlen, pos):
        T = x.shape[0]; q, k, v = s.qkv(x).view(T, 3, s.H, 64).unbind(1)
        c, sn = rope_tables(pos, 64)
        q = rope(F.rms_norm(q, (64,), s.qn), c, sn); k = rope(F.rms_norm(k, (64,), s.kn), c, sn)
        o = varlen_attn(q, k, v, cu, cu, maxlen, maxlen, window_size=(-1, 0))
        return s.o(o.reshape(T, s.d))

class Twin(nn.Module):           # Reviewer A's F_twin: H=d/128 heads, d_qk 64, d_v 128, conv4+SiLU on q/k/v, QK-RMSNorm, gated RMSNorm out
    def __init__(s, d):
        super().__init__(); s.d = d; s.H = d // 128
        s.q = nn.Linear(d, s.H * 64, bias=False); s.k = nn.Linear(d, s.H * 64, bias=False)
        s.v = nn.Linear(d, s.H * 128, bias=False); s.g = nn.Linear(d, s.H * 128, bias=False); s.o = nn.Linear(s.H * 128, d, bias=False)
        s.qc = ShortConvolution(s.H * 64, 4, activation='silu'); s.kc = ShortConvolution(s.H * 64, 4, activation='silu')
        s.vc = ShortConvolution(s.H * 128, 4, activation='silu')
        s.qn = nn.Parameter(torch.ones(64)); s.kn = nn.Parameter(torch.ones(64))
        s.on = FusedRMSNormGated(128, activation='swish')
    def forward(s, x, cu, maxlen, pos):
        T = x.shape[0]; H = s.H; cl = cu.long()
        q, _ = s.qc(s.q(x)[None], cu_seqlens=cl); k, _ = s.kc(s.k(x)[None], cu_seqlens=cl); v, _ = s.vc(s.v(x)[None], cu_seqlens=cl)
        c, sn = rope_tables(pos, 64)
        q = rope(F.rms_norm(q[0].view(T, H, 64), (64,), s.qn), c, sn); k = rope(F.rms_norm(k[0].view(T, H, 64), (64,), s.kn), c, sn)
        qd = q.repeat_interleave(2, 1); kd = k.repeat_interleave(2, 1); vd = v[0].reshape(T, 2 * H, 64)   # exact duplication trick
        o = varlen_attn(qd, kd, vd, cu, cu, maxlen, maxlen, window_size=(-1, 0)).view(T, H, 128)
        o = s.on(o, s.g(x).view(T, H, 128))
        return s.o(o.reshape(T, H * 128))

class Lin(nn.Module):
    def __init__(s, d, kind):
        super().__init__()
        if kind == 'gdnA': s.m = GatedDeltaNet(hidden_size=d, head_dim=64, num_heads=d // 128, expand_v=2, use_gate=True, use_short_conv=True, conv_size=4, mode='chunk')
        elif kind == 'gdnE': s.m = GatedDeltaNet(hidden_size=d, head_dim=d // 2, num_heads=1, expand_v=2, use_gate=True, use_short_conv=True, conv_size=4, mode='chunk')
        elif kind == 'gla': s.m = GatedLinearAttention(hidden_size=d, num_heads=d // 128, expand_k=0.5, expand_v=1.0, use_short_conv=True, conv_size=4, use_output_gate=True, mode='chunk')
    def forward(s, x, cu, maxlen, pos): return s.m(x[None], cu_seqlens=cu.long())[0][0]

class Block(nn.Module):
    def __init__(s, d, ff, mixer):
        super().__init__(); s.n1 = nn.RMSNorm(d); s.n2 = nn.RMSNorm(d); s.mix = mixer
        s.g = nn.Linear(d, ff, bias=False); s.u = nn.Linear(d, ff, bias=False); s.dn = nn.Linear(ff, d, bias=False)
    def forward(s, x, cu, maxlen, pos):
        x = x + s.mix(s.n1(x), cu, maxlen, pos); h = s.n2(x); return x + s.dn(F.silu(s.g(h)) * s.u(h))

class LM(nn.Module):
    def __init__(s, d, ff, k, full, lin):
        super().__init__(); P = set([int((j + 0.5) * L / k) for j in range(k)]) if k < L else set(range(L))
        s.emb = nn.Embedding(V, d)
        s.blocks = nn.ModuleList([Block(d, ff, (Plain(d) if full == 'plain' else Twin(d)) if i in P else Lin(d, lin)) for i in range(L)])
        s.nf = nn.RMSNorm(d)
    def forward(s, ids, tgt, cu, maxlen, pos, ckpt):
        x = s.emb(ids)
        for b in s.blocks: x = checkpoint(b, x, cu, maxlen, pos, use_reentrant=False) if ckpt else b(x, cu, maxlen, pos)
        x = s.nf(x); loss = 0.0; C = 4096
        for i in range(0, x.shape[0], C):
            loss = loss + checkpoint(lambda h, t: F.cross_entropy((h @ s.emb.weight.t()).float(), t, reduction='sum'), x[i:i + C], tgt[i:i + C], use_reentrant=False)
        return loss / x.shape[0]

def doc_lens(total, g):
    out = []
    while sum(out) < total: out.append(max(16, int(math.exp(torch.normal(6.54, 1.0, (1,), generator=g).item()))))
    out[-1] -= sum(out) - total; return [x for x in out if x > 0]

def bench(d, ff, k, full, lin, ntok, seqlen, ckpt, steps=6, warm=3):
    torch.manual_seed(0); m = LM(d, ff, k, full, lin).to(dev); opt = torch.optim.AdamW(m.parameters(), lr=1e-3, fused=True)
    nemb = sum(p.numel() for p in m.parameters()) - V * d
    g = torch.Generator().manual_seed(0); lens = []
    for _ in range(ntok // seqlen): lens += (doc_lens(seqlen, g) if seqlen <= 4096 else [seqlen])
    cu = torch.tensor([0] + list(torch.cumsum(torch.tensor(lens), 0)), device=dev, dtype=torch.int32); maxlen = max(lens)
    pos = torch.cat([torch.arange(n, device=dev) for n in lens]); ids = torch.randint(0, V, (ntok,), device=dev); tgt = torch.randint(0, V, (ntok,), device=dev)
    torch.cuda.reset_peak_memory_stats(); ts = []
    for i in range(steps + warm):
        torch.cuda.synchronize(); t0 = time.time()
        with torch.autocast('cuda', dtype=torch.bfloat16): loss = m(ids, tgt, cu, maxlen, pos, ckpt)
        loss.backward(); opt.step(); opt.zero_grad(set_to_none=True); torch.cuda.synchronize()
        if i >= warm: ts.append(time.time() - t0)
    pk = torch.cuda.max_memory_allocated() / 2 ** 30; fin = math.isfinite(loss.item()); del m, opt; torch.cuda.empty_cache()
    return ntok / min(ts), ntok / sorted(ts)[len(ts) // 2], pk, nemb, fin

cfgs = []
for stage, (ntok, seq, ck) in [('A', (8192, 4096, False)), ('B', (32768, 32768, True))]:
    for d, ff in [(512, 1536), (384, 1024)]:
        for (k, full, lin) in [(0, 'plain', 'gdnA'), (12, 'plain', 'gdnA'), (12, 'twin', 'gdnA'), (0, 'plain', 'gdnE'), (3, 'twin', 'gdnA'), (0, 'plain', 'gla')]:
            if lin == 'gla' and d == 512: continue
            cfgs.append((stage, d, ff, k, full, lin, ntok, seq, ck))
only = sys.argv[1] if len(sys.argv) > 1 else None
rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 1
sel = [c for c in cfgs if not only or c[0] == only]
order = []
for r in range(rounds): order += (sel if r % 2 == 0 else sel[::-1])
for (stage, d, ff, k, full, lin, ntok, seq, ck) in order:
    tag = f"stage {stage} d={d} k={k:2d} full={full if k>0 else '-':5s} lin={lin if k<12 else '-':4s}"
    try:
        t0 = time.time(); best, med, pk, nemb, fin = bench(d, ff, k, full, lin, ntok, seq, ck, *( (30, 8) if stage == 'A' else (6, 3) ))
        print(f"{tag}: tok/s best {best/1e3:6.1f}K median {med/1e3:6.1f}K | peak {pk:.2f} GiB | non-emb {nemb/1e6:.3f}M | finite={fin} | wall {time.time()-t0:.0f}s", flush=True)
    except torch.OutOfMemoryError:
        print(f"{tag}: OOM under cap", flush=True); torch.cuda.empty_cache()
    except Exception as e:
        print(f"{tag}: ERROR {type(e).__name__}: {str(e).splitlines()[0][:160]}", flush=True); torch.cuda.empty_cache()

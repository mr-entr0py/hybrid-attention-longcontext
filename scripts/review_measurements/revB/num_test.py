import time, math, torch, torch.nn.functional as F
torch.cuda.set_per_process_memory_fraction(0.30)
from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
dev='cuda'; torch.manual_seed(1)
B, T, H, K, V = 1, 32768, 4, 64, 128
def make(slow):
    q = F.normalize(torch.randn(B,T,H,K, device=dev, dtype=torch.float64), dim=-1)
    k = F.normalize(torch.randn(B,T,H,K, device=dev, dtype=torch.float64), dim=-1)
    v = torch.randn(B,T,H,V, device=dev, dtype=torch.float64)
    beta = torch.sigmoid(torch.randn(B,T,H, device=dev, dtype=torch.float64))
    if slow:  # near-1 decay: worst case for long-range accumulation
        g = -torch.full((B,T,H), 1e-4, device=dev, dtype=torch.float64)
    else:     # mixed per-token decays in [exp(-0.5), exp(-1e-4)]
        g = -torch.exp(torch.empty(B,T,H, device=dev, dtype=torch.float64).uniform_(math.log(1e-4), math.log(0.5)))
    return q,k,v,beta,g
@torch.no_grad()
def naive64(q,k,v,beta,g):
    h = torch.zeros(B,H,K,V, device=dev, dtype=torch.float64); o = torch.empty(B,T,H,V, device=dev, dtype=torch.float64)
    s = K ** -0.5
    for t in range(T):
        h = h * g[:,t].exp()[...,None,None]
        vn = (v[:,t] - torch.einsum('bhk,bhkv->bhv', k[:,t], h)) * beta[:,t][...,None]
        h = h + k[:,t][...,None] * vn[...,None,:]
        o[:,t] = torch.einsum('bhk,bhkv->bhv', q[:,t]*s, h)
    return o, h
for slow in (False, True):
    q,k,v,beta,g = make(slow)
    t0=time.time(); ref, href = naive64(q,k,v,beta,g); tn=time.time()-t0
    rms = ref.pow(2).mean().sqrt().item()
    for dt in (torch.float32, torch.bfloat16):
        args = [x.to(dt) for x in (q,k,v)]
        o, hT = chunk_gated_delta_rule(*args, g=g.float(), beta=beta.to(dt), output_final_state=True)
        err = (o.double()-ref).abs()
        bands = [(0,1024),(1024,4096),(4096,16384),(16384,32768)]
        rel = {f'{a//1024}K-{b//1024}K': round((err[:,a:b].pow(2).mean().sqrt()/rms).item(),5) for a,b in bands}
        mx = {f'{a//1024}K-{b//1024}K': round(err[:,a:b].max().item(),4) for a,b in bands}
        sterr = ((hT.double()-href).abs().max()/href.abs().max()).item()
        print(f"decay={'near1' if slow else 'mixed'} dtype={str(dt)[6:]} outRMS={rms:.3f} relRMSerr_by_pos={rel} maxabs_by_pos={mx} finalstate_relmax={sterr:.2e} (naive64 {tn:.1f}s)")
    # recurrent (decode) kernel vs chunk on the last 64 tokens, starting from the chunk state at T-64 (bf16)
    dt = torch.bfloat16; P = T-64
    a = [x.to(dt) for x in (q,k,v)]
    o_full, _ = chunk_gated_delta_rule(*a, g=g.float(), beta=beta.to(dt))
    _, h0 = chunk_gated_delta_rule(*[x[:, :P] for x in a], g=g[:, :P].float(), beta=beta[:, :P].to(dt), output_final_state=True)
    o_rec, _ = fused_recurrent_gated_delta_rule(q=a[0][:,P:], k=a[1][:,P:], v=a[2][:,P:], g=g[:,P:].float(), beta=beta[:,P:].to(dt), initial_state=h0, output_final_state=True)
    print(f"   branched(prefill chunk + recurrent tail) vs single chunk pass, bf16: maxabs={((o_rec.float()-o_full[:,P:].float()).abs().max().item()):.4f}  ref-maxabs(rec)={((o_rec.double()-ref[:,P:]).abs().max().item()):.4f}")

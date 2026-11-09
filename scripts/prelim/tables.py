import math
def gdn_params(d, H, dk, dv, gate=True, K=4):
    q=k=d*H*dk; v=d*H*dv; g=d*H*dv if gate else 0; o=H*dv*d
    ab=2*d*H; conv=K*(2*H*dk+H*dv); onorm=dv; own=2*H
    return q+k+v+g+o+ab+conv+onorm+own
def attn_params(d, dh=64, qknorm=True): return 4*d*d + (2*dh if qknorm else 0)
def mlp(d,dff): return 3*d*dff
def model(d,L,dff,V,k,gdn):
    nb = d*2
    a = attn_params(d)+mlp(d,dff)+nb; g = gdn+mlp(d,dff)+nb
    N = (L-k)*g + k*a + d
    return N, N+V*d, a, g
V=16384
print("F12 d512 L12 dff1536, GDN-A:")
gA = gdn_params(512,4,64,128); print(" GDN-A mixer", gA, " attn mixer", attn_params(512))
for k in (0,1,2,3,4,6,12):
    N,T,a,g = model(512,12,1536,V,k,gA); print(f"  k={k:2d} r={k/12:.3f} nonemb={N:,} total={T:,}  (blocks attn={a:,} gdn={g:,})")
gD = gdn_params(512,2,128,256); print(" GDN-D mixer", gD)
for k in (0,1,3): N,T,_,_=model(512,12,1536,V,k,gD); print(f"  F12-D k={k} nonemb={N:,} total={T:,}")
g3 = gdn_params(384,3,64,128); print(" d384 GDN-A(H3) mixer", g3)
for k in (0,3,12): N,T,_,_=model(384,12,1152,V,k,g3); print(f"  F12-S k={k} nonemb={N:,} total={T:,}")
for dff in (1024,960):
    for k in (0,2,6,24): N,T,_,_=model(384,24,dff,V,k,g3); print(f"  F24 dff={dff} k={k} nonemb={N:,} total={T:,}")
# FLOPs/token forward (dense = 2*nonemb + LM head), attention scores 2*T*d per attn layer (causal avg)
d=512
for k in (0,1,3,12):
    N,_,_,_=model(512,12,1536,V,k,gA); dense=2*N+2*d*V
    row=[]
    for T in (4096,32768):
        att=k*2*T*d; gdnx=(12-k)*0.25e6
        row.append(f"T={T}: fwd {(dense+att+gdnx)/1e6:.0f} MFLOP/tok (attn share {att/(dense+att+gdnx):.0%})")
    print(f"  k={k}: dense {dense/1e6:.1f} MFLOP/tok; "+"; ".join(row))
# Memory per sequence (bf16 KV, fp32 GDN state + fp32 conv width 4 as stored by fla)
kv_tok = 2*d*2          # bytes per token per attention layer
S = 4*64*128*4          # recurrent state bytes
C = (2*4*64+4*128)*4*4  # conv cache bytes (fla stores width conv_size=4, fp32)
print(f" KV/token/layer={kv_tok} B; GDN-A state={S} B + conv {C} B = {S+C} B ({(S+C)/1024:.0f} KiB); break-even T={ (S+C)/kv_tok:.0f} tokens")
for T in (1024,4096,32768):
    print("  T=%6d: "%T + "  ".join(f"k={k}:{(k*kv_tok*T+(12-k)*(S+C))/2**20:.2f}MiB" for k in (0,1,2,3,4,6,12)))
# stats
z=1.959964; zb=0.841621
for n in (400,500,1000,1250):
    print(f" n={n}: paired MDE (omega^2=.25) = {100*(z+zb)*math.sqrt(0.25/n):.2f} pp; Wilson half-width p=.5 ~ {100*z*math.sqrt(.25/n):.2f} pp")
for Q,icc in ((4,0.2),(4,0.5)):
    de=1+(Q-1)*icc; print(f" Q={Q} ICC={icc}: DE={de:.2f}; 400 contexts -> n_eff={400*Q/de:.0f}")
# Stage B time per token model (contended single-layer fwd+bwd: attn 32K 0.84 Mtok/s, GDN 32K 3.27 Mtok/s; dense+other 7.6 us/tok)
for k in (0,1,2,3,4,6,12):
    t=7.6+k/0.84+(12-k)/3.27; t2=t*1.33
    print(f"  StageB 32K k={k}: {1e6/t:,.0f} tok/s (no ckpt) {1e6/t2:,.0f} tok/s (+33% act-ckpt) -> 0.1B tokens {0.1e9/(1e6/t2)/3600:.2f} h")

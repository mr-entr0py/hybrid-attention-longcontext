# Delta-rule associative capacity: GDN-A geometry (H=4, dk=64, dv=128) vs GDN-E (H=1, dk=256, dv=512) at d=512,
# and D384 geometries (H=3,dk=64,dv=128) vs (H=1,dk=192,dv=384). Keys are random projections of random token
# embeddings (no learned partitioning across heads: every head writes every fact, beta=1, no decay).
# Readout = concatenation of head outputs; score = mean cosine(retrieved, true concatenated value) over all N facts,
# and the fraction of facts whose retrieved value is closer (cosine) to its own true value than to every other value.
import numpy as np
rng = np.random.default_rng(0)
def run(d, H, dk, dv, N, trials=3):
    cos_all, top1_all = [], []
    for _ in range(trials):
        X = rng.standard_normal((N, d)) / np.sqrt(d)                 # fact token representations
        Wk = rng.standard_normal((H, d, dk)); Wv = rng.standard_normal((H, d, dv))
        K = np.einsum('nd,hdk->hnk', X, Wk); K /= np.linalg.norm(K, axis=-1, keepdims=True)   # L2-normalised keys per head
        Vv = np.einsum('nd,hdv->hnv', X, Wv) / np.sqrt(d)
        S = np.zeros((H, dk, dv))
        for n in range(N):                                           # delta rule, beta = 1, alpha = 1
            k = K[:, n, :]; v = Vv[:, n, :]
            pred = np.einsum('hk,hkv->hv', k, S)
            S += np.einsum('hk,hv->hkv', k, v - pred)
        R = np.einsum('hnk,hkv->nhv', K, S).reshape(N, H * dv)       # query with the same key
        T = Vv.transpose(1, 0, 2).reshape(N, H * dv)
        Rn = R / np.linalg.norm(R, axis=1, keepdims=True); Tn = T / np.linalg.norm(T, axis=1, keepdims=True)
        C = Rn @ Tn.T
        cos_all.append(np.mean(np.diag(C))); top1_all.append(np.mean(np.argmax(C, axis=1) == np.arange(N)))
    return np.mean(cos_all), np.mean(top1_all)
for d, geoms in [(512, [(4, 64, 128), (1, 256, 512)]), (384, [(3, 64, 128), (1, 192, 384)])]:
    print(f"d={d}")
    for (H, dk, dv) in geoms:
        row = []
        for N in [16, 32, 64, 128, 192, 256, 512, 1024]:
            c, t = run(d, H, dk, dv, N)
            row.append(f"N={N}: cos {c:.2f} top1 {t:.2f}")
        print(f"  H={H} dk={dk} dv={dv} (H*dk={H*dk}, state {H*dk*dv*4//1024} KiB): " + " | ".join(row))

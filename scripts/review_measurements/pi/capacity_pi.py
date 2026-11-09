# Independent PI check of Reviewer B round-2 W5(a): does per-head d_k or total H*d_k govern delta-rule
# interference when every head writes every fact? beta=1, no decay, L2-normalised keys; fresh random
# projections per head; readout = concatenated head outputs (cosine to concatenated true value).
import numpy as np
rng = np.random.default_rng(7)
def run(d, H, dk, dv, N, trials=4):
    out = []
    for _ in range(trials):
        X = rng.standard_normal((N, d))
        K = np.einsum('nd,hdk->hnk', X, rng.standard_normal((H, d, dk)))
        K /= np.linalg.norm(K, axis=-1, keepdims=True)
        V = np.einsum('nd,hdv->hnv', X, rng.standard_normal((H, d, dv)))
        S = np.zeros((H, dv, dk))
        for n in range(N):
            k = K[:, n]; v = V[:, n]
            S += np.einsum('hv,hk->hvk', v - np.einsum('hvk,hk->hv', S, k), k)
        R = np.einsum('hvk,hnk->nhv', S, K).reshape(N, -1); T = V.transpose(1, 0, 2).reshape(N, -1)
        out.append(np.mean(np.sum(R * T, 1) / np.linalg.norm(R, axis=1) / np.linalg.norm(T, axis=1)))
    return np.mean(out)
for H, dk, dv in [(4, 64, 128), (1, 256, 512), (2, 128, 256)]:
    print(f"H={H} dk={dk} dv={dv} (H*dk={H*dk}, state {H*dk*dv} elems):",
          " ".join(f"N={N}:{run(512, H, dk, dv, N):.2f}" for N in (32, 64, 128, 256, 512)))

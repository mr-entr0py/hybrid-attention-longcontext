# Effect of SiLU-before-QK-RMSNorm (Reviewer A's twin) on softmax needle weight at init (unit gains, d_h = 64, scale 1/8).
# Needle key = query direction (cos = 1); distractor keys independent. Logit = 8 * g^2 * cos. Compare plain RMSNorm vs SiLU->RMSNorm.
import numpy as np
rng = np.random.default_rng(0); d = 64
def unit(x): return x / np.linalg.norm(x, axis=-1, keepdims=True)
silu = lambda x: x / (1 + np.exp(-x))
for name, f in [('plain', lambda x: x), ('silu', silu)]:
    q = unit(f(rng.standard_normal((200, d)))); k = unit(f(rng.standard_normal((40000, d))))
    c = q @ k.T
    print(f"{name:5s}: distractor cosine mean {c.mean():.3f} sd {c.std():.3f}")
    for g2 in (1.0, 1.56, 2.25):
        row = []
        for T in (1024, 4096, 16384, 32768):
            # needle weight = e^{8 g2} / (e^{8 g2} + sum_{T-1} e^{8 g2 cos_j}); estimate sum by sampled mean (per query)
            ex = np.exp(8 * g2 * c[:, :]).mean(axis=1) * (T - 1)
            w = np.exp(8 * g2) / (np.exp(8 * g2) + ex)
            row.append(f"{T//1024}K {w.mean():.2f}")
        print(f"   gain^2={g2}: needle weight " + "  ".join(row))

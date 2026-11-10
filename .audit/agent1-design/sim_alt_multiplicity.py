"""Agent-1 design audit: H1 power under alternative (pre-G5) multiplicity procedures and seed counts.

Same model as sim_h1_holm.py. H5a assumed strongly supported (p ~ 0) unless stated; H3 p = 1 (non-evaluable);
H1-min from the same seeds with R1 = 0.3. Adaptive 5 -> 7 (registered U-05 trigger) unless n fixed.
Procedures:
  REG   : registered Holm over {H1, H1-min, H3, H5a}
  FS1   : fixed sequence H1 -> H5a -> H1-min -> H3 (H1 at full alpha 0.05)
  WHOLM : weighted Holm, weights H1 .5, H5a .3, H1-min .1, H3 .1 (H1 first-step alpha 0.025)
"""
from __future__ import annotations

import numpy as np
from scipy import stats

rng = np.random.default_rng(5150)


def draw(R3, R1, G, n, NT, sigma, tau=2.0):
    mu = np.array([40.0, 40 + R1 * G, 40 + R3 * G, 40 + G])
    return mu[None, :, None] + tau * rng.standard_normal((NT, 1, n)) + sigma * rng.standard_normal((NT, 4, n))


def p_greater(L):
    n = L.shape[-1]
    t = L.mean(-1) / (L.std(-1, ddof=1) / np.sqrt(n))
    return stats.t.sf(t, n - 1)


def resid_sd(S):
    K, n = S.shape[1], S.shape[2]
    res = S - S.mean(2, keepdims=True) - S.mean(1, keepdims=True) + S.mean((1, 2), keepdims=True)
    return np.sqrt((res ** 2).sum((1, 2)) / ((K - 1) * (n - 1)))


def weighted_holm_first(p_h1, others, w_h1, w_others, alpha=0.05):
    """Weighted Holm (Holm 1979 weighted version): returns H1 rejection. others: list of (p, w)."""
    ps = [p_h1] + [o for o in others]
    ws = [w_h1] + w_others
    P = np.stack(ps, 1); W = np.array(ws)[None, :].repeat(P.shape[0], 0)
    rejected = np.zeros_like(P, bool)
    active = np.ones_like(P, bool)
    for _ in range(P.shape[1]):
        wsum = (W * active).sum(1, keepdims=True)
        thr = alpha * W / np.where(wsum > 0, wsum, 1)
        ratio = np.where(active, P / np.maximum(thr, 1e-300), np.inf)
        j = ratio.argmin(1)
        ok = ratio[np.arange(len(j)), j] <= 1
        newrej = np.zeros_like(rejected); newrej[np.arange(len(j)), j] = ok
        rejected |= newrej
        active &= ~newrej
        if not ok.any():
            break
    return rejected[:, 0]


def run(R3, G, sigma, n_fixed=None, NT=40000):
    if n_fixed is None:
        S = draw(R3, 0.3, G, 7, NT, sigma)
        S5 = S[:, :, :5]
        trig = (resid_sd(S5) > 2.5) | ((S5.max(2) - S5.min(2)).max(1) > 12)
        pH1 = np.empty(NT); pH1m = np.empty(NT)
        for big, X in ((False, S5), (True, S)):
            sel = trig == big
            L3 = X[sel][:, 2] - 0.2 * X[sel][:, 0] - 0.8 * X[sel][:, 3]
            L1 = X[sel][:, 1] - 0.2 * X[sel][:, 0] - 0.8 * X[sel][:, 3]
            pH1[sel] = p_greater(L3); pH1m[sel] = p_greater(L1)
    else:
        S = draw(R3, 0.3, G, n_fixed, NT, sigma)
        pH1 = p_greater(S[:, 2] - 0.2 * S[:, 0] - 0.8 * S[:, 3]); pH1m = p_greater(S[:, 1] - 0.2 * S[:, 0] - 0.8 * S[:, 3])
    p5 = np.zeros(NT); p3 = np.ones(NT)
    reg = weighted_holm_first(pH1, [pH1m, p3, p5], 0.25, [0.25, 0.25, 0.25])
    fs1 = pH1 < 0.05
    wh = weighted_holm_first(pH1, [p5, pH1m, p3], 0.5, [0.3, 0.1, 0.1])
    return reg.mean(), fs1.mean(), wh.mean()


def main():
    print("P(H1 supported). cols: REG (registered Holm) | FS1 (H1 first, alpha .05) | WHOLM (H1 weight .5)")
    for G in (20, 30):
        for sigma in (3.0, 5.0):
            for R3 in (0.8, 0.9, 1.0, 1.1):
                a = run(R3, G, sigma)
                print(f"adaptive 5->7  G={G} sigma={sigma} R3={R3}: REG {a[0]:.3f} | FS1 {a[1]:.3f} | WHOLM {a[2]:.3f}")
    print("\nFixed n (no trigger), sigma = 5, G = 30 / 20, R3 = 1.0:")
    for G in (30, 20):
        for n in (5, 7, 10, 12):
            a = run(1.0, G, 5.0, n_fixed=n)
            print(f"  G={G} n={n}: REG {a[0]:.3f} | FS1 {a[1]:.3f} | WHOLM {a[2]:.3f}")


if __name__ == "__main__":
    main()

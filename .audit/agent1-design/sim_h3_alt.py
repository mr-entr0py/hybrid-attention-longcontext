"""Agent-1 design audit: H3 evaluability/power, registered rule vs a pooled-denominator alternative.

REG : per-seed R ratios; evaluable iff EVERY seed has G >= 15 pp in both families (protocol §2/§11.7).
ALT : evaluable iff seed-POOLED G >= 15 pp in both families and every seed G > 5 pp;
      per-seed D3~_s = (S3r - S0r)_s / mean(Gr) - (S3h - S0h)_s / mean(Gh)  (plug-in pooled denominators).
Holm level for H3 taken as 0.0167 (H5a rejected first). Correlation of residuals across families 0.5.
"""
import numpy as np
from scipy import stats

rng = np.random.default_rng(99)


def sim(NT, n, sigma, Gr, Gh, Rr, Rh, rho=0.5, tau=2.0, alpha=0.0167):
    b = tau * rng.standard_normal((NT, 1, 1, n))
    u = rng.standard_normal((NT, 1, 3, n)); e = rng.standard_normal((NT, 2, 3, n))
    eps = sigma * (np.sqrt(rho) * u + np.sqrt(1 - rho) * e)
    mu = np.array([[40, 40 + Rr * Gr, 40 + Gr], [40, 40 + Rh * Gh, 40 + Gh]], float)
    S = mu[None, :, :, None] + b + eps
    G = S[:, :, 2] - S[:, :, 0]
    num = S[:, :, 1] - S[:, :, 0]

    def t_p(D):
        t = D.mean(-1) / (D.std(-1, ddof=1) / np.sqrt(n))
        return stats.t.sf(t, n - 1)

    D_reg = num[:, 0] / G[:, 0] - num[:, 1] / G[:, 1]
    ev_reg = (G >= 15).all(axis=(1, 2))
    sup_reg = ev_reg & (t_p(D_reg) < alpha)
    Gbar = G.mean(-1, keepdims=True)
    D_alt = num[:, 0] / Gbar[:, 0] - num[:, 1] / Gbar[:, 1]
    ev_alt = (Gbar[:, :, 0] >= 15).all(1) & (G > 5).all(axis=(1, 2))
    sup_alt = ev_alt & (t_p(D_alt) < alpha)
    return ev_reg.mean(), sup_reg.mean(), ev_alt.mean(), sup_alt.mean()


print("cols: REG evaluable | REG support | ALT evaluable | ALT support")
for n in (5, 7):
    for sigma in (3.0, 5.0):
        for (Gr, Gh, Rr, Rh, lab) in ((30, 20, 0.9, 0.6, "D3=.30 Ghop=20"), (30, 25, 0.9, 0.6, "D3=.30 Ghop=25"),
                                      (30, 20, 0.9, 0.75, "D3=.15 Ghop=20"), (30, 20, 0.8, 0.8, "D3=0 Ghop=20 (type I)")):
            r = sim(40000, n, sigma, Gr, Gh, Rr, Rh)
            print(f"n={n} sigma={sigma} {lab:22s}: REG ev {r[0]:.3f} sup {r[1]:.3f} | ALT ev {r[2]:.3f} sup {r[3]:.3f}")

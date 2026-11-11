"""Agent-1 design audit, session 2, addendum (CPU only).
F. U-05 trigger rate vs the SHARED seed effect tau (cancels in paired L_k; the raw-range clause still sees it).
G. FEAS-512 single-seed rule (seed 101): P(S12 - S0 >= 20 pp at one seed) vs true gap.
H. Per-seed SD of L3 implied by the PI-SIM model, and standardized effect at R3 = 1.0.
"""
import numpy as np
from scipy import stats
rng = np.random.default_rng(424242)

def u05(sigma, tau, n=5, NT=200_000):
    S = tau * rng.standard_normal((NT, 1, n)) + sigma * rng.standard_normal((NT, 4, n))
    res = S - S.mean(2, keepdims=True) - S.mean(1, keepdims=True) + S.mean((1, 2), keepdims=True)
    sd = np.sqrt((res ** 2).sum((1, 2)) / (3 * (n - 1)))
    rg = (S.max(2) - S.min(2)).max(1)
    return (sd > 2.5).mean(), (rg > 12).mean(), ((sd > 2.5) | (rg > 12)).mean()

print("F. U-05 (n0 = 5): P(resid-SD clause) | P(range clause) | P(fire)")
for sigma in (2.0, 3.0):
    for tau in (0.0, 2.0, 4.5):
        a, b, c = u05(sigma, tau)
        print(f"  sigma={sigma} tau={tau}: {a:.2f} | {b:.2f} | {c:.2f}")

print("\nG. FEAS-512 single seed, gap clause only, item SE on gap 1.9 pp: P(pass >= 20 pp)")
for sigma in (2.0, 3.0, 5.0):
    row = []
    for G in (15, 20, 22, 25, 30):
        x = G + np.sqrt(2) * sigma * rng.standard_normal(400_000) + 1.9 * rng.standard_normal(400_000)
        row.append(f"G={G}: {(x >= 20).mean():.2f}")
    print(f"  sigma={sigma}: " + " | ".join(row))

print("\nH. SD(L3 per seed) = sigma*sqrt(1 + 0.2^2 + 0.8^2) = sigma*1.296 (seed effect cancels: 1 - 0.2 - 0.8 = 0)")
for sigma in (2.0, 3.0, 5.0):
    for G in (15, 20, 30):
        sd = sigma * np.sqrt(1 + 0.04 + 0.64); eff = 0.2 * G
        print(f"  sigma={sigma} G={G}: SD(L3) {sd:.2f} pp, effect at R3=1.0 = {eff:.1f} pp, d = {eff/sd:.2f}")

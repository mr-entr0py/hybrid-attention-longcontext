"""Agent-1 design audit, session 2, addendum c (CPU only).
I. FEAS-384 U-01: registered 'both seeds >= 25 pp' vs alternative 'seed-mean >= 25 pp' (gap clause only).
J. 95% CI for a seed SD estimated from n = 3 runs (R5: Zoology 0.0455, ours 0.0557 on 1024/256).
"""
import numpy as np
from scipy import stats
rng = np.random.default_rng(31415)
item = 1.87  # FD 32K gap item SE at p = 0.5 (design_checks_r2.out, check A)
print("I. P(adopt D384): registered both-seeds | alternative seed-mean")
for sigma in (2.0, 3.0, 5.0):
    row = []
    for G in (20, 25, 28, 30, 33):
        x = G + np.sqrt(2) * sigma * rng.standard_normal((400_000, 2)) + item * rng.standard_normal((400_000, 2))
        row.append(f"G={G}: {(x >= 25).all(1).mean():.2f} | {(x.mean(1) >= 25).mean():.2f}")
    print(f"  sigma={sigma}: " + " || ".join(row))
print("\nJ. 95% CI of sigma from a 3-run sample SD (df = 2): [s*sqrt(2/chi2_.975), s*sqrt(2/chi2_.025)]")
lo, hi = np.sqrt(2 / stats.chi2.ppf(0.975, 2)), np.sqrt(2 / stats.chi2.ppf(0.025, 2))
for name, s in (("Zoology", 4.55), ("ours", 5.57)):
    print(f"  {name}: s = {s} pp -> [{s*lo:.1f}, {s*hi:.1f}] pp (factors {lo:.2f}x, {hi:.2f}x)")

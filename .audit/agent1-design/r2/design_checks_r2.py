"""Agent-1 design audit, session 2 (2026-11-10 ~21:00 IST): new CPU-only checks.

Model conventions follow PI-SIM (debate_record.md §6.2): per-seed config score
S[k,s] = mu_k + b_s + e_ks, b_s ~ N(0, tau^2) shared across configs at a seed, e ~ N(0, sigma^2).
Units: percentage points (pp). Nothing here touches the GPU or the repo outside this directory.

Checks:
 A. Item-level SE of a per-seed score (CS primary cell; FD 32K cell used by FEAS) from cell sizes.
 B. FEAS-384 'both seeds S12 - S0 >= 25 pp' rule: pass probability vs true gap, with FD item noise.
 C. U-05 trigger: firing probability and expected extra GPU-h (protocol §13.2: +15.3 h D384, +31.9 h D512).
 D. H1 support probability under the registered Holm-4 rule, fixed n = 5 vs n = 7, by Holm position
    (best: H5a and H3 rejected first -> H1 at alpha/2; typical: H5a first, H3 p = 1 -> alpha/3;
    worst: H1 smallest -> alpha/4), against the unadjusted alpha = 0.05 figure PI-SIM reports.
 E. 8K fallback without G-REF: P(H1 'supported') when the true gap G at 8K is small or negative.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

rng = np.random.default_rng(20261110 + 2)


def sec(t):
    print("\n" + "=" * 96 + "\n" + t + "\n" + "=" * 96)


# ------------------------------------------------------------------------------------------- A
def cell_se(p, contexts, q=4, de=1.6):
    return 100 * np.sqrt(p * (1 - p) / (contexts * q / de))


def family_se(p, contexts, n_r1=2, n_rl=5, de=1.6):
    s = cell_se(p, contexts, de=de)
    return 0.5 * np.sqrt((s / np.sqrt(n_r1)) ** 2 + (s / np.sqrt(n_rl)) ** 2)


# ------------------------------------------------------------------------------------------- B
def feas384_pass(G, sigma, item_se_gap, NT=400_000, thr=25.0):
    # gap per seed = G + residual (sigma*sqrt2, k0 and k12 residuals) + item noise on the gap
    x = G + np.sqrt(2) * sigma * rng.standard_normal((NT, 2)) + item_se_gap * rng.standard_normal((NT, 2))
    return (x >= thr).all(1).mean()


# ------------------------------------------------------------------------------------------- C
def u05_fire(sigma, n=5, tau=2.0, NT=200_000):
    S = tau * rng.standard_normal((NT, 1, n)) + sigma * rng.standard_normal((NT, 4, n))
    res = S - S.mean(2, keepdims=True) - S.mean(1, keepdims=True) + S.mean((1, 2), keepdims=True)
    sd = np.sqrt((res ** 2).sum((1, 2)) / (3 * (n - 1)))
    rngmax = (S.max(2) - S.min(2)).max(1)
    return ((sd > 2.5) | (rngmax > 12.0)).mean()


# ------------------------------------------------------------------------------------------- D
def h1_power(R3, G, sigma, n, alpha_eff, tau=2.0, NT=200_000):
    mu = np.array([40.0, 40 + R3 * G, 40 + G])  # k0, k3, k12
    S = mu[None, :, None] + tau * rng.standard_normal((NT, 1, n)) + sigma * rng.standard_normal((NT, 3, n))
    L = S[:, 1] - 0.2 * S[:, 0] - 0.8 * S[:, 2]
    t = L.mean(1) / (L.std(1, ddof=1) / np.sqrt(n))
    return (stats.t.sf(t, n - 1) < alpha_eff).mean()


# ------------------------------------------------------------------------------------------- E
def fallback_support(G, R3, sigma, n=5, alpha_eff=0.05 / 3, tau=2.0, NT=200_000):
    mu = np.array([50.0, 50 + R3 * G, 50 + G])
    S = mu[None, :, None] + tau * rng.standard_normal((NT, 1, n)) + sigma * rng.standard_normal((NT, 3, n))
    L = S[:, 1] - 0.2 * S[:, 0] - 0.8 * S[:, 2]
    t = L.mean(1) / (L.std(1, ddof=1) / np.sqrt(n))
    return (stats.t.sf(t, n - 1) < alpha_eff).mean()


def main():
    sec("A. Item-level SE (design effect 1.6 = 1 + 3*ICC at ICC 0.2; worst case p = 0.5 per cell)")
    for p in (0.5, 0.3):
        print(f"p={p}: CS cell (300x4) SE {cell_se(p, 300):.2f} pp | retrieval ladder mean per length "
              f"{family_se(p, 300):.2f} pp | mean of 16K+32K {family_se(p, 300) / np.sqrt(2):.2f} pp")
        f = family_se(p, 100)
        print(f"p={p}: FD cell (100x4) SE {cell_se(p, 100):.2f} pp | FD 32K retrieval ladder mean {f:.2f} pp | "
              f"gap S12-S0 item SE {np.sqrt(2) * f:.2f} pp")
    print("Protocol §9.3 claim: 0.5 pp (two lengths) to 0.7 pp (one length) per-seed primary-cell item SE.")

    sec("B. FEAS-384: P(both seeds S12 - S0 >= 25 pp at 32K), FD item noise on the gap = 1.9 pp (p=0.5)")
    item = np.sqrt(2) * family_se(0.5, 100)
    for sigma in (2.0, 3.0, 5.0):
        row = [f"G={G}: {feas384_pass(G, sigma, item):.2f}" for G in (20, 25, 28, 30, 33, 35, 40)]
        print(f"sigma={sigma}: " + " | ".join(row))
    print("(gap clause only; the adequacy clause S12 >= 0.85*S12(C0) in both seeds can only lower these)")

    sec("C. U-05 trigger: P(fire) after n0 seeds, and expected extra GPU-h (base, no contingency)")
    for sigma in (2.0, 2.5, 3.0, 4.0, 5.0):
        p5 = u05_fire(sigma, n=5); p4 = u05_fire(sigma, n=4)
        print(f"sigma={sigma}: D384 n0=5 P(fire) {p5:.2f} -> E[+h] {15.3 * p5:5.1f} | "
              f"D512 n0=4 P(fire) {p4:.2f} -> E[+h] {31.9 * p4:5.1f}")

    sec("D. H1: P(support) at R3 = 1.0 and 0.9, by Holm position, fixed n (no trigger), tau = 2")
    print("cols: unadj 0.05 | best a/2=0.025 | typical a/3=0.0167 | worst a/4=0.0125")
    for G in (20, 30):
        for sigma in (2.0, 3.0, 5.0):
            for n in (5, 7):
                for R3 in (1.0, 0.9):
                    v = [h1_power(R3, G, sigma, n, a) for a in (0.05, 0.025, 0.05 / 3, 0.0125)]
                    print(f"G={G} sigma={sigma} n={n} R3={R3}: " + " | ".join(f"{x:.3f}" for x in v))

    sec("E. 8K fallback with no G-REF at 8K: P(H1 'supported' at Holm a/3), n = 5, sigma = 3")
    for G, R3, note in ((5.0, 1.2, "tiny gap; hybrid ~ both"), (5.0, 1.0, "tiny gap"),
                        (0.0, 0.0, "no gap; L3 = S3 - S0 ... all equal"),
                        (-5.0, 0.6, "NEGATIVE gap: hybrid worse than k=0, R3 = 0.6"),
                        (-10.0, 0.5, "NEGATIVE gap: hybrid halfway between k=0 and worse k=12")):
        mu3_minus_mu0 = R3 * G
        L_true = mu3_minus_mu0 - 0.8 * G
        print(f"G={G:+5.1f} R3={R3:.1f} (true L3 {L_true:+.1f} pp; {note}): "
              f"P(support) {fallback_support(G, R3, 3.0):.3f}")
    print("Worked example (no noise): S0=55, S12=50, S3=52 -> L3 = 52 - 11 - 40 = +1 > 0 although "
          "R3 = (52-55)/(50-55) = 0.6 and the hybrid is below the all-linear model.")


if __name__ == "__main__":
    main()

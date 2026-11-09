# Monte Carlo of H1 decision rules (CPU, numpy/scipy-free t quantiles hard-coded).
# Per-run score S[k, seed] = mu_k + b_seed + e, e ~ N(0, s_e^2); paired-difference SD sigma_d = sqrt(2)*s_e.
# True: S0 = 40, S12 = 70 (G = 30 pp), S3 = S0 + R*G.
# Rule B/PROP: seed-level t (df=2) on per-seed R_s and on Delta_s = S12_s - S3_s (two-sided 95%):
#   supported  : lowerCI(R) >= 0.8 AND upperCI(Delta) <= 5 pp
#   falsified  : upperCI(R) < 0.8
# Rule A (round 1): L3 = S3 - 0.2 S0 - 0.8 S12, residual variance pooled over a stage with df_pool;
#   one-sided 95%: supported if lowerCI(L3) > 0, falsified if upperCI(L3) < 0.
import numpy as np
rng = np.random.default_rng(7)
t975_2, t95_11, t95_7 = 4.302653, 1.795885, 1.894579
def sim(R, sigma_d, n_sim=200000, df_pool=11):
    s_e = sigma_d / np.sqrt(2)
    G = 30.0; mu0, mu12 = 40.0, 70.0; mu3 = mu0 + R * G
    b = rng.normal(0, 3.0, (n_sim, 3))                      # seed block effect (cancels in paired stats)
    S0 = mu0 + b + rng.normal(0, s_e, (n_sim, 3))
    S3 = mu3 + b + rng.normal(0, s_e, (n_sim, 3))
    S12 = mu12 + b + rng.normal(0, s_e, (n_sim, 3))
    # Rule B / PROP
    Rs = (S3 - S0) / (S12 - S0); Ds = S12 - S3
    hwR = t975_2 * Rs.std(1, ddof=1) / np.sqrt(3); hwD = t975_2 * Ds.std(1, ddof=1) / np.sqrt(3)
    supB = (Rs.mean(1) - hwR >= 0.8) & (Ds.mean(1) + hwD <= 5.0)
    falB = (Rs.mean(1) + hwR < 0.8)
    # Rule A: pooled residual variance, df_pool (simulate s^2 ~ s_e^2 chi2_df/df)
    L = S3.mean(1) - 0.2 * S0.mean(1) - 0.8 * S12.mean(1)
    s2 = s_e**2 * rng.chisquare(df_pool, n_sim) / df_pool
    se = np.sqrt(s2 * (1 + 0.04 + 0.64) / 3)
    t = t95_11 if df_pool == 11 else t95_7
    supA = (L - t * se > 0); falA = (L + t * se < 0)
    return supB.mean(), falB.mean(), supA.mean(), falA.mean()
print("sigma_d  true_R | B/PROP: P(sup) P(fals) P(inconcl) | A(L3,pooled df=11): P(sup) P(fals) P(inconcl)")
for sd in (3.0, 5.0):
    for R in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1):
        sB, fB, sA, fA = sim(R, sd)
        print(f"{sd:5.1f}   {R:4.2f}   |  {sB:6.3f} {fB:6.3f} {1-sB-fB:6.3f}       |  {sA:6.3f} {fA:6.3f} {1-sA-fA:6.3f}")

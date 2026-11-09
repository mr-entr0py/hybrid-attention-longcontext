# PI adjudication sim: H1 contrast L3 = S3 - 0.2 S0 - 0.8 S12 (>=0 <=> R3 >= 0.8), one-sided alpha 0.05.
# Core table: configs k in {0,1,3,12} x n seeds, seed-blocked: S[k,s] = mu_k + b_s + e_ks.
# Tests:
#   POOL : additive OLS (config + seed) over the 4 x n core table, df = 3(n-1)  [Reviewer A, round 2]
#   PAIR : per-seed L3_s, one-sample t, df = n-1                                 [Reviewer B, round 2]
#   ADAPT: start with n0 seeds; effect-blind trigger -> add 2 seeds; then PAIR on final n.
#          trigger = (blinded pooled residual SD > 2.5 pp) OR (any config's raw seed range > 12 pp)
# Scenarios: homo (sigma=2 or 3 for all), hetero (sigma_3 = 8), bimodal (k=3 seed fails w.p. 1/3 by -25 pp,
#   mixture mean held at target), bimodal13 (k=1 and k=3 both bimodal).
import numpy as np
rng = np.random.default_rng(20261109)
T95 = {2: 2.919986, 3: 2.353363, 4: 2.131847, 5: 2.015048, 6: 1.943180, 7: 1.894579, 9: 1.833113,
       12: 1.782288, 15: 1.753050}
G, MU0 = 30.0, 40.0
KS = [0, 1, 3, 12]
def draw(scn, R3, n, NT, sigma, tau=2.0):
    mu = {0: MU0, 1: MU0 + 0.5 * G, 3: MU0 + R3 * G, 12: MU0 + G}
    sd = {k: sigma for k in KS}
    bim = set()
    if scn == 'hetero': sd[3] = 8.0
    if scn == 'bimodal': bim = {3}
    if scn == 'bimodal13': bim = {1, 3}
    b = tau * rng.standard_normal((NT, n))
    S = np.empty((NT, 4, n))
    for i, k in enumerate(KS):
        m = mu[k] + (25.0 / 3 if k in bim else 0.0)
        x = m + sd[k] * rng.standard_normal((NT, n)) + b
        if k in bim: x -= 25.0 * (rng.random((NT, n)) < 1 / 3)
        S[:, i, :] = x
    return S
def pooled_support(S):
    NT, K, n = S.shape
    rm = S.mean(2, keepdims=True); cm = S.mean(1, keepdims=True); gm = S.mean((1, 2), keepdims=True)
    res = S - rm - cm + gm
    df = (K - 1) * (n - 1)
    s2 = (res ** 2).sum((1, 2)) / df
    L = S[:, 2].mean(1) - 0.2 * S[:, 0].mean(1) - 0.8 * S[:, 3].mean(1)
    se = np.sqrt(s2 * 1.68 / n)
    return L - T95[df] * se > 0, np.sqrt(s2)
def paired_support(S):
    n = S.shape[2]
    Ls = S[:, 2] - 0.2 * S[:, 0] - 0.8 * S[:, 3]
    return Ls.mean(1) - T95[n - 1] * Ls.std(1, ddof=1) / np.sqrt(n) > 0
def adaptive(scn, R3, sigma, n0, NT=40000, sd_thr=2.5, rng_thr=12.0):
    S = draw(scn, R3, n0 + 2, NT, sigma)
    S0 = S[:, :, :n0]
    _, s_hat = pooled_support(S0)
    rngs = S0.max(2) - S0.min(2)
    trig = (s_hat > sd_thr) | (rngs.max(1) > rng_thr)
    sup_small = paired_support(S0); sup_big = paired_support(S)
    sup = np.where(trig, sup_big, sup_small)
    exp_n = n0 + 2 * trig.mean()
    return sup.mean(), trig.mean(), exp_n
NT = 60000
print("Columns: P(support).  At R3=0.80 this is the TYPE-I rate (nominal 0.05); at 0.90/1.00 it is POWER.")
for sigma in (2.0, 3.0):
    print(f"\n=== residual SD sigma = {sigma} pp (paired-difference SD = {sigma*np.sqrt(2):.2f}), tau_seed = 2 pp, G = 30 pp")
    for scn in ('homo', 'hetero', 'bimodal', 'bimodal13'):
        for R3 in (0.8, 0.9, 1.0):
            row = []
            for n in (3, 4, 5):
                S = draw(scn, R3, n, NT, sigma)
                ps, _ = pooled_support(S)
                row.append(f"n={n}: POOL {ps.mean():.3f} PAIR {paired_support(S).mean():.3f}")
            a3 = adaptive(scn, R3, sigma, 3); a4 = adaptive(scn, R3, sigma, 4)
            print(f"{scn:9s} R3={R3:.2f} | " + " | ".join(row) +
                  f" | ADAPT 3->5: {a3[0]:.3f} (trig {a3[1]:.2f}, E[n] {a3[2]:.2f}) | ADAPT 4->6: {a4[0]:.3f} (trig {a4[1]:.2f}, E[n] {a4[2]:.2f})")

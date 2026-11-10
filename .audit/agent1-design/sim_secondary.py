"""Agent-1 design audit: operating characteristics of H3, H4, H6', H5(a) TOST, triggers, selection biases.

Same seed model as PI-SIM: S = mu + b_s (shared across configs at a seed) + e (config x seed residual).
Units: pp. CPU only.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

rng = np.random.default_rng(777)


def t_one(x, alt="greater", null=0.0):
    n = x.shape[-1]
    m = x.mean(-1); se = x.std(-1, ddof=1) / np.sqrt(n); t = (m - null) / se
    if alt == "greater":
        return stats.t.sf(t, n - 1)
    if alt == "less":
        return stats.t.cdf(t, n - 1)
    return 2 * stats.t.sf(np.abs(t), n - 1)


def tost_one(x, margin, alpha=0.05):
    return (t_one(x, "greater", -margin) < alpha) & (t_one(x, "less", margin) < alpha)


def section(title):
    print("\n" + "=" * 100 + "\n" + title + "\n" + "=" * 100)


# ---------------------------------------------------------------------------------------------- H3
def h3(NT, n, sigma, Gr, Gh, Rr, Rh, rho=0.5, tau=2.0, alpha_holm=0.0167):
    """Per-seed D3 = R_ret(3) - R_hop(3); evaluable iff every seed has G >= 15 pp in both families."""
    mu0 = 40.0
    b = tau * rng.standard_normal((NT, 1, 1, n))  # (NT, fam, k, n)
    u = rng.standard_normal((NT, 1, 3, n))  # shared across families, per config x seed
    e = rng.standard_normal((NT, 2, 3, n))
    eps = sigma * (np.sqrt(rho) * u + np.sqrt(1 - rho) * e)
    mu = np.array([[mu0, mu0 + Rr * Gr, mu0 + Gr], [mu0, mu0 + Rh * Gh, mu0 + Gh]])  # fam x (k0, k3, k12)
    S = mu[None, :, :, None] + b + eps
    G = S[:, :, 2] - S[:, :, 0]
    R = (S[:, :, 1] - S[:, :, 0]) / G
    D = R[:, 0] - R[:, 1]
    evaluable = (G >= 15).all(axis=(1, 2))
    p = t_one(D, "greater")
    sup = evaluable & (p < alpha_holm)
    tost = evaluable & tost_one(D, 0.10)
    return evaluable.mean(), sup.mean(), tost.mean(), np.median(np.abs(D.std(-1, ddof=1) / np.sqrt(n)))


# ---------------------------------------------------------------------------------------------- H4
def h4(NT, n, sigma, c1_true=0.0, c2_true=0.0, tau=2.0, alpha=0.0125):
    """Arms early/mid/late; c1 = mid - (early+late)/2; c2 = late - early. Two-sided t; TOST +-5 pp."""
    b = tau * rng.standard_normal((NT, 1, n))
    mu = np.array([50 - c2_true / 2 - c1_true / 1.5 * 0.5, 50 + c1_true - c1_true / 1.5 * 0.5,
                   50 + c2_true / 2 - c1_true / 1.5 * 0.5])
    # simple: set early = 50 - c2/2, late = 50 + c2/2, mid = 50 + c1 (so c1 holds exactly)
    mu = np.array([50 - c2_true / 2, 50 + c1_true, 50 + c2_true / 2])
    S = mu[None, :, None] + b + sigma * rng.standard_normal((NT, 3, n))
    c1 = S[:, 1] - 0.5 * (S[:, 0] + S[:, 2]); c2 = S[:, 2] - S[:, 0]
    rej1 = t_one(c1, "two") < alpha
    no_eff = tost_one(c1, 5.0) & tost_one(c2, 5.0)
    return rej1.mean(), no_eff.mean()


# ---------------------------------------------------------------------------------------------- H6'
def h6p(NT, nE, nA, sigma, delta=0.0, alpha=0.0125):
    x = delta + sigma * rng.standard_normal((NT, nE))
    y = sigma * rng.standard_normal((NT, nA))
    mx, my = x.mean(1), y.mean(1); vx, vy = x.var(1, ddof=1) / nE, y.var(1, ddof=1) / nA
    se = np.sqrt(vx + vy); df = (vx + vy) ** 2 / (vx ** 2 / (nE - 1) + vy ** 2 / (nA - 1))
    t = (mx - my) / se
    rej = 2 * stats.t.sf(np.abs(t), df) < alpha
    tq = stats.t.ppf(0.95, df)
    tost = ((mx - my) - tq * se > -5) & ((mx - my) + tq * se < 5)
    return rej.mean(), tost.mean(), np.median(tq * se)


# ---------------------------------------------------------------------------------------------- H5a
def h5a(NT, n, sig_len, slope_true, sd_slope=0.5):
    """Per-seed OLS slope of Delta0 on log2T over 6 lengths; length-specific noise sig_len on each S."""
    x = np.arange(6.0); xc = x - x.mean()
    true_s = slope_true + sd_slope * rng.standard_normal((NT, n))
    noise = np.sqrt(2) * sig_len * rng.standard_normal((NT, n, 6))
    D = true_s[..., None] * x + noise
    b = (D * xc).sum(-1) / (xc ** 2).sum()
    p = t_one(b, "greater")
    return (p < 0.0125).mean(), tost_one(b, 2.0).mean(), np.median(b.std(-1, ddof=1) / np.sqrt(n))


def main():
    section("H3 (D3 = R_ret(3) - R_hop(3), per-seed ratio; Holm level ~0.0167; n = 5 (and 7 after U-05))")
    print("cols: P(evaluable) | P(support) | P(TOST +-0.10 'no task dependence') | median SE(D3)")
    for n in (5, 7):
        for sigma in (2.0, 3.0, 5.0):
            for (Gr, Gh, Rr, Rh, lab) in ((30, 25, 0.9, 0.6, "D3=0.30, Ghop=25"),
                                          (30, 20, 0.9, 0.6, "D3=0.30, Ghop=20"),
                                          (30, 20, 0.9, 0.75, "D3=0.15, Ghop=20"),
                                          (30, 25, 0.8, 0.8, "D3=0 (null), Ghop=25"),
                                          (40, 40, 0.8, 0.8, "D3=0 (null), both G=40")):
                ev, sup, tst, se = h3(20000, n, sigma, Gr, Gh, Rr, Rh)
                print(f"n={n} sigma={sigma} {lab:24s}: evaluable {ev:.3f} | support {sup:.3f} | TOST {tst:.3f} | SE {se:.3f}")

    section("H4 placement (paired, two-sided; Holm secondary family of 4 -> first step alpha 0.0125)")
    for n in (4, 5):
        for sigma in (2.0, 3.0, 5.0):
            for c1 in (0.0, 5.0, 7.3, 10.0):
                rej, _ = h4(20000, n, sigma, c1_true=c1)
                print(f"n={n} sigma={sigma} true c1={c1:4.1f}: P(reject c1 at 0.0125) = {rej:.3f}", end="")
                if c1 == 0.0:
                    _, ne = h4(20000, n, sigma)
                    print(f" | P(TOST 'no placement effect', both contrasts, true 0) = {ne:.3f}")
                else:
                    print()
    print("\nMDE check (80% power, two-sided, per-seed SD sigma*sqrt(1.5)):")
    for n in (4, 5):
        for a in (0.025, 0.0125):
            for sigma in (3.0, 5.0):
                tq = stats.t.ppf(1 - a / 2, n - 1); tb = stats.t.ppf(0.8, n - 1)
                print(f"  n={n} alpha2={a}: (t {tq:.3f} + {tb:.3f}) * 1.2247*sigma/sqrt(n) = "
                      f"{(tq + tb) * 1.2247 * sigma / np.sqrt(n):.1f} pp at sigma={sigma}")

    section("H6' (GDN-E k=0, 2 seeds, vs GDN-A k*, n seeds; Welch two-sided; TOST +-5 pp)")
    for nE in (2, 3, 5):
        for nA in (5, 7):
            for sigma in (2.0, 3.0, 5.0):
                rej0, tost0, hw = h6p(20000, nE, nA, sigma, 0.0)
                rej10, _, _ = h6p(20000, nE, nA, sigma, 10.0)
                print(f"nE={nE} nA={nA} sigma={sigma}: P(TOST equiv | true 0) = {tost0:.3f} (median 90% half-width "
                      f"{hw:.1f} pp) | P(reject | true 0) {rej0:.3f} | P(reject | true 10 pp) {rej10:.3f}")

    section("H5(a) slope test at Holm 0.0125 and 'no length effect' TOST +-2 pp/doubling")
    for n in (5, 7):
        for sig_len in (1.0, 2.0, 3.0):
            for slope in (0.0, 2.0, 4.0):
                pw, tst, se = h5a(20000, n, sig_len, slope)
                print(f"n={n} length-noise={sig_len} true slope={slope}: P(support) {pw:.3f} | P(TOST) {tst:.3f} | SE {se:.2f}")

    section("Single/two-seed triggers: false-fire rate under NO true effect, as a function of run-level noise")
    for s in (2.0, 3.0, 5.0):
        # U-02: mean over seeds 0-1 of S12(X-gc12) - S12(C12); paired by seed (b_s cancels)
        d = s * np.sqrt(2) * rng.standard_normal((200000, 2))
        dg = d.mean(1)
        u02 = (dg > 3).mean(); u02c = (dg > 6).mean()
        # U-06 at seed 0: R3 nope vs rope (shared C0), |dR3| > 0.1 or |dS12| > 3, G = 30, R = 0.8
        NT = 200000; G = 30.0
        S0 = 40 + s * rng.standard_normal(NT)
        S3r, S12r = 40 + 0.8 * G + s * rng.standard_normal(NT), 40 + G + s * rng.standard_normal(NT)
        S3n, S12n = 40 + 0.8 * G + s * rng.standard_normal(NT), 40 + G + s * rng.standard_normal(NT)
        dR = (S3n - S0) / (S12n - S0) - (S3r - S0) / (S12r - S0)
        u06 = ((np.abs(dR) > 0.1) | (np.abs(S12n - S12r) > 3)).mean()
        # saturation guard: 2x branch vs 1x (same Stage-A checkpoint): |dS12| > 5
        sat = (np.abs(s * np.sqrt(2) * rng.standard_normal(NT)) > 5).mean()
        print(f"run-level noise {s} pp: U-02 >3pp {u02:.3f}, >6pp {u02c:.3f} | U-06 fires {u06:.3f} | "
              f"saturation |dS12|>5 {sat:.3f}")
    print("\nU-02 miss rate when the true conv effect is +5 pp (operator wording wrongly allowed if dhat <= 3):")
    for s in (2.0, 3.0, 5.0):
        dg = 5 + s * np.sqrt(2) * rng.standard_normal((200000, 2))
        print(f"  noise {s}: P(dhat <= 3 | true 5) = {(dg.mean(1) <= 3).mean():.3f}")
    print("\nStill-closing flag (seed 0; |gap(4000) - gap(1000)| > 3 pp), checkpoint noise c per model score:")
    for c in (1.0, 1.5, 2.0):
        dl = c * 2 * rng.standard_normal(200000)  # four scores
        print(f"  c={c}: P(flag | no true change) = {(np.abs(dl) > 3).mean():.3f}")

    section("FEAS-384 gate: P(both seeds pass S12 - S0 >= 25 pp) vs true gap (paired difference noise)")
    for s in (2.0, 3.0, 5.0):
        row = []
        for g in (20, 25, 28, 30, 35):
            x = g + s * np.sqrt(2) * rng.standard_normal((200000, 2)) + 1.0 * rng.standard_normal((200000, 2))
            row.append(f"G={g}: {((x >= 25).all(1)).mean():.2f}")
        print(f"residual sigma {s}: " + " | ".join(row) + "   (adequacy clause not included)")

    section("G-REF seed-level selection on the same seeds: bias of L3 at the selected primary cell")
    NT, n, s = 100000, 5, 3.0
    for G in (14, 15, 17, 20):
        mu = np.array([40, 40 + 0.8 * G, 40 + G])
        S = mu[None, :, None] + 2 * rng.standard_normal((NT, 1, n)) + s * rng.standard_normal((NT, 3, n))
        dev_noise = 0.7 * rng.standard_normal((NT, 3, n))  # item-level noise of DG vs CS
        Gdev = (S[:, 2] + dev_noise[:, 2]).mean(1) - (S[:, 0] + dev_noise[:, 0]).mean(1)
        sel = Gdev >= 15
        L = (S[:, 1] - 0.2 * S[:, 0] - 0.8 * S[:, 2]).mean(1)
        print(f"true G={G}, R=0.8 (true L3 = 0): P(pass b) {sel.mean():.3f}; E[L3 | pass] = {L[sel].mean():+.3f} pp; "
              f"E[L3 | fail] = {L[~sel].mean():+.3f} pp; P(falsif. one-sided p<0.05 | pass) = "
              f"{(t_one((S[:, 1] - 0.2 * S[:, 0] - 0.8 * S[:, 2])[sel], 'less') < 0.05).mean():.3f}")

    section("k* selection on CS composition data, reused as H4 middle arm: bias of c1 (no true placement effect)")
    NT, n, s, G = 100000, 5, 3.0, 30.0
    for R1 in (0.2, 0.3, 0.4):
        S0 = 40 + s * rng.standard_normal((NT, n)); S12 = 70 + s * rng.standard_normal((NT, n))
        S1 = 40 + R1 * G + s * rng.standard_normal((NT, n))
        E1 = 40 + R1 * G + s * rng.standard_normal((NT, n)); L1 = 40 + R1 * G + s * rng.standard_normal((NT, n))
        Rhat = (S1.mean(1) - S0.mean(1)) / (S12.mean(1) - S0.mean(1))
        sel = Rhat >= 0.3  # k* = 1 selected
        c1 = (S1 - 0.5 * (E1 + L1)).mean(1)
        print(f"true R(1)={R1}: P(k*=1) {sel.mean():.3f}; E[c1 | k*=1] = {c1[sel].mean():+.2f} pp (true 0); "
              f"P(c1 two-sided p<0.0125 | k*=1) = {(t_one((S1 - 0.5 * (E1 + L1))[sel], 'two') < 0.0125).mean():.3f}")


if __name__ == "__main__":
    main()

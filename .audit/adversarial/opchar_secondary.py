"""Operating characteristics of secondary/trigger rules (adversarial review, 2026-11-10).
Pure numpy/scipy Monte Carlo; mirrors src/hybridlc/analysis/stats.py formulas (welch_t, tost_welch,
tost_one_sample, one_sample_t). Model: per-seed score = mu_config + seed effect b_s + e, b~N(0,tau^2), e~N(0,sigma^2).
"""
import numpy as np
from scipy import stats as sps

rng = np.random.default_rng(20261110)
R = 40000
ALPHA = 0.05

def tost_from(est, se, df, m, alpha=ALPHA):
    p_lo = sps.t.sf((est + m) / se, df)
    p_hi = sps.t.cdf((est - m) / se, df)
    return np.maximum(p_lo, p_hi) < alpha

def welch(x, y):
    nx, ny = x.shape[-1], y.shape[-1]
    vx, vy = x.var(-1, ddof=1) / nx, y.var(-1, ddof=1) / ny
    est = x.mean(-1) - y.mean(-1); se = np.sqrt(vx + vy)
    df = (vx + vy) ** 2 / (vx ** 2 / (nx - 1) + vy ** 2 / (ny - 1))
    return est, se, df

def pooled(x, y):
    nx, ny = x.shape[-1], y.shape[-1]
    sp2 = ((nx - 1) * x.var(-1, ddof=1) + (ny - 1) * y.var(-1, ddof=1)) / (nx + ny - 2)
    est = x.mean(-1) - y.mean(-1); se = np.sqrt(sp2 * (1 / nx + 1 / ny))
    return est, se, nx + ny - 2

print("== H6' : X-E0 (2 seeds) vs C{k*} (n seeds), unpaired; Welch primary, TOST +-5 pp ==")
for n in (5, 4):
    for sig in (2.0, 3.0):
        s_tot = np.sqrt(sig**2 + 2.0**2)   # unpaired: seed effect does not cancel (tau = 2 pp)
        for delta in (0.0, 5.0, 10.0, 15.0, 20.0):
            x = rng.normal(delta, s_tot, (R, 2)); y = rng.normal(0, s_tot, (R, n))
            est, se, df = welch(x, y)
            p = 2 * sps.t.sf(np.abs(est / se), df)
            eq = tost_from(est, se, df, 5.0)
            estp, sep, dfp = pooled(x, y)
            pp = 2 * sps.t.sf(np.abs(estp / sep), dfp)
            eqp = tost_from(estp, sep, dfp, 5.0)
            print(f" n={n} sigma={sig} (unpaired SD {s_tot:.2f})  true diff={delta:4.1f}pp : "
                  f"P(Welch p<.05)={np.mean(p<ALPHA):.3f}  P(TOST equiv, Welch)={np.mean(eq):.3f}  "
                  f"| pooled: P(p<.05)={np.mean(pp<ALPHA):.3f} P(equiv)={np.mean(eqp):.3f}  median Welch df={np.median(df):.2f}")

print("\n== H4 : 'no placement effect' needs TOST +-5 pp on BOTH c1 and c2 (paired over seeds) ==")
for n in (5, 4):
    for sig in (2.0, 3.0, 5.0):
        for eff in (0.0, 3.0):
            e = rng.normal(0, sig, (R, 3, n))  # residuals for early, mid, late (seed effect cancels in contrasts)
            early, mid, late = e[:, 0], e[:, 1] + eff, e[:, 2]
            c1 = mid - 0.5 * (early + late); c2 = late - early
            def tost1(x):
                est = x.mean(-1); se = x.std(-1, ddof=1) / np.sqrt(n)
                return tost_from(est, se, n - 1, 5.0)
            both = tost1(c1) & tost1(c2)
            # power of two-sided paired t for c1 at Holm first step (alpha/2)
            t1 = c1.mean(-1) / (c1.std(-1, ddof=1) / np.sqrt(n)); pc1 = 2 * sps.t.sf(np.abs(t1), n - 1)
            print(f" n={n} sigma={sig} true c1={eff}pp: P(no-placement-effect TOST both)={np.mean(both):.3f}  "
                  f"P(c1 rejects at 0.025)={np.mean(pc1<0.025):.3f}")

print("\n== U-02 trigger: delta_gc = seed-paired mean over 2 seeds of S12(X-gc12)-S12(C12) > 3 pp ==")
for sig in (2.0, 3.0):
    sd_diff = sig * np.sqrt(2)
    for d in (0.0, 1.5, 3.0, 4.5, 6.0, 9.0):
        est = rng.normal(d, sd_diff / np.sqrt(2), R)
        print(f" sigma={sig} true delta={d}: P(>3pp)={np.mean(est>3):.3f}  P(>6pp co-primary)={np.mean(est>6):.3f}")

print("\n== U-06 trigger: single-seed NoPE vs RoPE; fires if |dR3|>0.1 or |dS12|>3pp (true effect 0) ==")
for sig in (2.0, 3.0):
    for G in (20.0, 30.0, 50.0):
        # seed-0 paired: X-nope3 vs C3, X-nope12 vs C12; S0 shared (same C0 run)
        s0 = rng.normal(0, sig, R); s12 = rng.normal(G, sig, R); s3 = rng.normal(0.9 * G, sig, R)
        n12 = rng.normal(G, sig, R); n3 = rng.normal(0.9 * G, sig, R)
        R3 = (s3 - s0) / (s12 - s0); R3n = (n3 - s0) / (n12 - s0)
        fire = (np.abs(R3n - R3) > 0.1) | (np.abs(n12 - s12) > 3)
        print(f" sigma={sig} G={G}: P(trigger fires under no true effect)={np.mean(fire):.3f}")

print("\n== H3 : D3 = R_ret(3) - R_hop(3) per seed (paired, one-sided, n seeds); TOST +-0.10 ==")
for n in (5, 7):
    for sig in (2.0, 3.0):
        for Ghop in (15.0, 20.0, 30.0):
            for D in (0.0, 0.2, 0.3):
                Gret = 30.0
                b = rng.normal(0, 2.0, (R, 1, n))
                ret0 = rng.normal(40, sig, (R, n)); ret12 = rng.normal(40 + Gret, sig, (R, n))
                Rr = 0.9; Rh = Rr - D
                ret3 = rng.normal(40 + Rr * Gret, sig, (R, n))
                hop0 = rng.normal(20, sig, (R, n)); hop12 = rng.normal(20 + Ghop, sig, (R, n))
                hop3 = rng.normal(20 + Rh * Ghop, sig, (R, n))
                # per-seed G>=15 evaluability
                ok = ((ret12 - ret0) >= 15).all(-1) & ((hop12 - hop0) >= 15).all(-1)
                Rret = (ret3 - ret0) / (ret12 - ret0); Rhop = (hop3 - hop0) / (hop12 - hop0)
                d3 = Rret - Rhop
                est = d3.mean(-1); se = d3.std(-1, ddof=1) / np.sqrt(n)
                p = sps.t.sf(est / se, n - 1)
                eq = tost_from(est, se, n - 1, 0.10)
                print(f" n={n} sigma={sig} G_hop={Ghop} true D3={D}: P(evaluable)={np.mean(ok):.3f}  "
                      f"P(p<.05 & evaluable)={np.mean((p<ALPHA)&ok):.3f} (raw, before Holm)  "
                      f"P(TOST equiv & evaluable)={np.mean(eq&ok):.3f}  median per-seed SD(D3)={np.median(d3.std(-1,ddof=1)):.2f}")

"""Agent-1 design audit: operating characteristics of the H1 decision rule AS REGISTERED.

Model = PI-SIM (debate_record.md §6.2): S[k,s] = mu_k + b_s + e_ks, k in {0,1,3,12},
mu0 = 40, mu12 = 40 + G, mu3 = 40 + R3*G, mu1 = 40 + R1*G; b_s ~ N(0, tau^2), e ~ N(0, sigma^2).
Units: percentage points.

What PI-SIM reported: P(L3 lower 95% bound > 0) at unadjusted one-sided alpha = 0.05.
What the protocol actually requires (§2, §11.3; analysis/pipeline.py run_analysis, verdicts.py):
  support  : Holm over {H1, H1-min, H3, H5a} at FWER 0.05 (one-sided, pre-registered direction)
  falsify  : Holm over {H1, H1-min} (opposite direction) + guards (not simulated: dose/saturation/gates)
  adaptive : n0 -> n0+2 if blinded pooled residual SD > 2.5 pp or any anchor seed range > 12 pp.
Scenarios for the other members of the support family:
  'H5a_strong' : H5a p ~ 0 (rejected first) -> H1 is tested at 0.05/3 if H1-min/H3 not smaller
  'H5a_null'   : H5a p ~ U(0,1)
  H3 non-evaluable (p = 1) in both (its per-seed G >= 15 pp rule; see sim_h3.py)
  H1-min simulated from the same seeds with R1 (default 0.3, i.e. clearly below 0.8).
"""
from __future__ import annotations

import sys

import numpy as np
from scipy import stats

rng = np.random.default_rng(20261110)
MU0 = 40.0


def draw(R3, R1, G, n, NT, sigma, tau=2.0):
    mu = np.array([MU0, MU0 + R1 * G, MU0 + R3 * G, MU0 + G])  # k = 0, 1, 3, 12
    b = tau * rng.standard_normal((NT, 1, n))
    e = sigma * rng.standard_normal((NT, 4, n))
    return mu[None, :, None] + b + e


def one_sided_p(L, direction):
    n = L.shape[-1]
    m = L.mean(-1)
    se = L.std(-1, ddof=1) / np.sqrt(n)
    t = m / se
    if direction == "greater":
        return stats.t.sf(t, n - 1)
    return stats.t.cdf(t, n - 1)


def pooled_resid_sd(S):
    NT, K, n = S.shape
    rm = S.mean(2, keepdims=True); cm = S.mean(1, keepdims=True); gm = S.mean((1, 2), keepdims=True)
    res = S - rm - cm + gm
    return np.sqrt((res ** 2).sum((1, 2)) / ((K - 1) * (n - 1)))


def holm_reject(pmat, alpha=0.05):
    """pmat: (NT, m). Returns boolean (NT, m) of Holm rejections."""
    NT, m = pmat.shape
    order = np.argsort(pmat, axis=1)
    ps = np.take_along_axis(pmat, order, axis=1)
    thr = alpha / (m - np.arange(m))
    ok = ps <= thr
    # step-down: reject while all previous rejected
    cum = np.cumprod(ok, axis=1).astype(bool)
    rej = np.zeros_like(cum)
    np.put_along_axis(rej, order, cum, axis=1)
    return rej


def decision(R3, G, sigma, n0=5, adaptive=True, R1=0.3, NT=40000, h5a="strong", tau=2.0):
    S = draw(R3, R1, G, n0 + 2, NT, sigma, tau)
    S0 = S[:, :, :n0]
    if adaptive:
        trig = (pooled_resid_sd(S0) > 2.5) | ((S0.max(2) - S0.min(2)).max(1) > 12.0)
    else:
        trig = np.zeros(NT, bool)
    res = {}
    out_sup = np.zeros(NT, bool); out_fal = np.zeros(NT, bool); out_raw = np.zeros(NT, bool)
    for use_big in (False, True):
        sel = trig == use_big
        if not sel.any():
            continue
        X = S[sel] if use_big else S0[sel]
        L3 = X[:, 2] - 0.2 * X[:, 0] - 0.8 * X[:, 3]
        L1 = X[:, 1] - 0.2 * X[:, 0] - 0.8 * X[:, 3]
        p3s, p1s = one_sided_p(L3, "greater"), one_sided_p(L1, "greater")
        p3f, p1f = one_sided_p(L3, "less"), one_sided_p(L1, "less")
        m = X.shape[0]
        if h5a == "strong":
            p5 = np.zeros(m)
        elif h5a == "null":
            p5 = rng.random(m)
        else:
            p5 = np.ones(m)
        p_h3 = np.ones(m)
        sup = holm_reject(np.stack([p3s, p1s, p_h3, p5], 1))[:, 0]
        fal = holm_reject(np.stack([p3f, p1f], 1))[:, 0]
        out_sup[sel] = sup; out_fal[sel] = fal; out_raw[sel] = p3s < 0.05
    res["P_support_holm"] = out_sup.mean()
    res["P_falsify_test"] = out_fal.mean()
    res["P_inconclusive"] = 1 - out_sup.mean() - out_fal.mean()
    res["P_support_unadjusted"] = out_raw.mean()
    res["trigger_rate"] = trig.mean()
    return res


def main():
    print("== Sanity check: reproduce PI-SIM PAIR (unadjusted alpha 0.05), G = 30, tau = 2 ==")
    for sigma in (2.0, 3.0):
        for n in (4, 5):
            r = decision(1.0, 30, sigma, n0=n, adaptive=False, NT=60000)
            print(f"sigma={sigma} n={n} fixed: P(support, unadjusted) = {r['P_support_unadjusted']:.3f}")
        for n in (4, 5):
            r = decision(1.0, 30, sigma, n0=n, adaptive=True, NT=60000)
            print(f"sigma={sigma} ADAPT {n}->{n+2}: P(support, unadjusted) = {r['P_support_unadjusted']:.3f} "
                  f"(trigger {r['trigger_rate']:.2f})")
        r = decision(0.8, 30, sigma, n0=5, adaptive=True, NT=60000)
        print(f"sigma={sigma} ADAPT 5->7 R3=0.8: type I unadjusted = {r['P_support_unadjusted']:.3f}")
    print()
    print("== Registered rule: Holm-adjusted support / falsification-direction test, adaptive 5 -> 7 ==")
    print("columns: R3 | P(support) H5a-strong | P(support) H5a-null | P(falsif. test rejects) | P(neither) [H5a strong]")
    for G in (15, 20, 25, 30, 40):
        for sigma in (2.0, 3.0, 5.0):
            print(f"--- G = {G} pp, sigma = {sigma} pp ---")
            for R3 in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2):
                a = decision(R3, G, sigma, h5a="strong")
                b = decision(R3, G, sigma, h5a="null")
                print(f"R3={R3:.1f} | sup(H5a strong) {a['P_support_holm']:.3f} | sup(H5a null) {b['P_support_holm']:.3f}"
                      f" | fal-test {a['P_falsify_test']:.3f} | neither {a['P_inconclusive']:.3f}"
                      f" | unadj {a['P_support_unadjusted']:.3f} | trig {a['trigger_rate']:.2f}")
    print()
    print("== Expected P(definitive H1 verdict) under the PI's own prediction R3 ~ U[0.6, 1.1] (§2) ==")
    for G in (20, 30):
        for sigma in (2.0, 3.0, 5.0):
            grid = np.linspace(0.6, 1.1, 11)
            sup = np.mean([decision(r, G, sigma, NT=20000)["P_support_holm"] for r in grid])
            fal = np.mean([decision(r, G, sigma, NT=20000)["P_falsify_test"] for r in grid])
            print(f"G={G} sigma={sigma}: E[P(support)] = {sup:.3f}, E[P(falsification test rejects)] = {fal:.3f} "
                  f"(guards not applied; actual 'falsified' <= this), E[P(no verdict)] >= {1 - sup - fal:.3f}")


if __name__ == "__main__":
    sys.exit(main())

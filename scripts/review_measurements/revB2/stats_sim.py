# Type-I error and power of H1 tests (L3 = S3 - 0.2 S0 - 0.8 S12 >= 0) under Reviewer A's Tier-1 design:
# 22 runs = P{0,1,3,12} x 3 seeds + P{2,6} x 2 + Ref{12} x 2 + PEalt{1,12} x 2, seed-blocked.
# Tests: (a) A's pooled-variance additive model (config + seed), one-sided t, df = n - p;
#        (b) A's rule: Brown-Forsythe across configs, Welch-Satterthwaite fallback if p < .05;
#        (c) seed-level paired t on per-seed L3 (df = S-1), the PI/Reviewer-B rule, 3 seeds;
#        (d) same paired t with 5 seeds for k in {0,3,12}.
import numpy as np
from scipy import stats
rng = np.random.default_rng(1)
configs = [('P0', 3), ('P1', 3), ('P3', 3), ('P12', 3), ('P2', 2), ('P6', 2), ('Ref12', 2), ('PE1', 2), ('PE12', 2)]
G = 30.0; mu0 = 20.0; mu12 = mu0 + G
def scenario(name, R3):
    # returns per-config (mean, sampler) ; true L3 = (R3-0.8)*G
    mu = {'P0': mu0, 'P12': mu12, 'P1': mu0 + 0.5 * G, 'P2': mu0 + 0.7 * G, 'P6': mu12, 'Ref12': mu12, 'PE1': mu0 + 0.5 * G, 'PE12': mu12}
    target3 = mu0 + R3 * G
    sd = {c: 3.0 for c, _ in configs}
    bim = {}
    if name == 'hetero':   sd['P3'] = 8.0
    if name == 'bimodal':  bim['P3'] = (1 / 3, 25.0)          # each k=3 seed independently 'misses the transition' w.p. 1/3, -25 pp
    if name == 'bimodal2': bim['P3'] = (1 / 3, 25.0); bim['P1'] = (1 / 3, 25.0)
    def mean_of(c):
        if c in bim: p, drop = bim[c]; return (target3 if c == 'P3' else mu[c]) + p * drop   # so that E[S] equals target
        return target3 if c == 'P3' else mu[c]
    def draw(c, n):
        x = mean_of(c) + sd[c] * rng.standard_normal(n)
        if c in bim: p, drop = bim[c]; x -= drop * (rng.random(n) < p)
        return x
    return draw
def one_trial(draw, tau=2.0, extra_seeds=False):
    seeds5 = tau * rng.standard_normal(5)
    data = {c: draw(c, n) + seeds5[:n] for c, n in configs}
    # (a)/(b) additive OLS: S ~ config + seed
    rows, y = [], []
    cidx = {c: i for i, (c, _) in enumerate(configs)}
    for c, n in configs:
        for s in range(n):
            x = np.zeros(len(configs) + 2); x[cidx[c]] = 1
            if s >= 1: x[len(configs) + s - 1] = 1
            rows.append(x); y.append(data[c][s])
    X = np.array(rows); y = np.array(y)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None); res = y - X @ beta; df = len(y) - X.shape[1]; s2 = res @ res / df
    cvec = np.zeros(X.shape[1]); cvec[cidx['P3']] = 1; cvec[cidx['P0']] = -0.2; cvec[cidx['P12']] = -0.8
    XtXi = np.linalg.pinv(X.T @ X); Lhat = cvec @ beta; se = np.sqrt(s2 * cvec @ XtXi @ cvec)
    tcrit = stats.t.ppf(0.95, df); pooled_support = Lhat - tcrit * se > 0
    # Brown-Forsythe across the 9 configs
    groups = [data[c] for c, _ in configs]
    _, p_bf = stats.levene(*groups, center='median')
    if p_bf < 0.05:   # Welch-Satterthwaite on unpaired means
        parts = [(1.0, data['P3']), (-0.2, data['P0']), (-0.8, data['P12'])]
        v = [(c * c) * np.var(x, ddof=1) / len(x) for c, x in parts]; seW = np.sqrt(sum(v))
        dfW = sum(v) ** 2 / sum(vi ** 2 / (len(x) - 1) for vi, (_, x) in zip(v, parts))
        LW = sum(c * x.mean() for c, x in parts); rule_support = LW - stats.t.ppf(0.95, dfW) * seW > 0
    else: rule_support = pooled_support
    # (c) paired per-seed L3, 3 seeds
    Ls = data['P3'] - 0.2 * data['P0'] - 0.8 * data['P12']
    paired3 = Ls.mean() - stats.t.ppf(0.95, 2) * Ls.std(ddof=1) / np.sqrt(3) > 0
    # (d) 5 seeds for the three anchors
    d5 = {c: np.concatenate([data[c], draw(c, 2) + seeds5[3:5]]) for c in ('P0', 'P3', 'P12')}
    L5 = d5['P3'] - 0.2 * d5['P0'] - 0.8 * d5['P12']
    paired5 = L5.mean() - stats.t.ppf(0.95, 4) * L5.std(ddof=1) / np.sqrt(5) > 0
    return pooled_support, rule_support, paired3, paired5, p_bf < 0.05
NT = 20000
print("one-sided nominal alpha = 0.05; 'support' = lower 95% one-sided bound of L3 > 0")
for name in ['homo', 'hetero', 'bimodal', 'bimodal2']:
    for R3, lab in [(0.8, 'TYPE-I (true R3=0.80)'), (1.0, 'POWER  (true R3=1.00)')]:
        draw = scenario(name, R3)
        out = np.array([one_trial(draw) for _ in range(NT)])
        print(f"{name:9s} {lab}: pooled {out[:,0].mean():.3f} | A-rule(BF->Welch) {out[:,1].mean():.3f} | paired 3 seeds {out[:,2].mean():.3f} | paired 5 seeds {out[:,3].mean():.3f} | BF triggered {out[:,4].mean():.3f}")

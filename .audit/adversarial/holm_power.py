"""Power of the paired one-sided L_k t-test at Holm-adjusted levels (adversarial review).
PI-SIM (debate §6.2) quotes power at one-sided alpha = 0.05; the registered support rule is Holm over 4 tests."""
import numpy as np
from scipy import stats as sps

def power(delta, sd, n, alpha):
    ncp = delta / (sd / np.sqrt(n)); tc = sps.t.ppf(1 - alpha, n - 1)
    return sps.nct.sf(tc, n - 1, ncp)

G = 30.0
for sig in (2.0, 3.0):
    sd = sig * np.sqrt(1 + 0.2**2 + 0.8**2)   # per-seed SD of L_k (seed effect cancels; residual sigma per config)
    for R in (0.9, 1.0, 1.1):
        d = (R - 0.8) * G
        row = []
        for n in (4, 5, 6, 7):
            row.append(f"n={n}: a=.05 {power(d, sd, n, 0.05):.2f} | a=.0167 {power(d, sd, n, 0.05/3):.2f} | a=.0125 {power(d, sd, n, 0.0125):.2f}")
        print(f"sigma={sig} G={G} R={R} (L={d:.1f}pp, SD_L={sd:.2f}):\n   " + "\n   ".join(row))

# Monte Carlo of the full Holm family under a plausible scenario: H1 true (R3=1.0), H1-min false (R1=0.5),
# H3 not evaluable (p=1), H5a true with a strong effect (p tiny) -> H1 tested at alpha/2 after H5a, ... 
rng = np.random.default_rng(7)
Rr = 100000
for sig in (2.0, 3.0):
    for n in (5, 7):
        sd = sig*np.sqrt(1.68)
        for R3, R1, h5p in ((1.0, 0.5, "tiny"), (1.0, 0.5, "one"), (1.0, 1.0, "tiny"), (0.9, 0.5, "tiny")):
            L3 = rng.normal((R3-0.8)*G, sd, (Rr, n)); L1 = rng.normal((R1-0.8)*G, sd, (Rr, n))
            p3 = sps.t.sf(L3.mean(1)/(L3.std(1,ddof=1)/np.sqrt(n)), n-1)
            p1 = sps.t.sf(L1.mean(1)/(L1.std(1,ddof=1)/np.sqrt(n)), n-1)
            pH3 = np.ones(Rr)
            p5 = np.full(Rr, 1e-6) if h5p == "tiny" else np.ones(Rr)
            P = np.stack([p3, p1, pH3, p5], 1)
            order = np.argsort(P, 1); Ps = np.take_along_axis(P, order, 1)
            m = 4; adj = np.maximum.accumulate(np.minimum(1, (m - np.arange(m)) * Ps), axis=1)
            rej_sorted = adj < 0.05
            rej = np.zeros_like(rej_sorted); np.put_along_axis(rej, order, rej_sorted, 1)
            print(f"MC Holm: sigma={sig} n={n} R3={R3} R1={R1} H3 p=1 H5a p={h5p}: P(H1 supported)={rej[:,0].mean():.3f} "
                  f"(vs unadjusted {np.mean(p3<0.05):.3f})")

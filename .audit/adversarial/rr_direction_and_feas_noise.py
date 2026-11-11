"""CPU-only arithmetic for the continuation adversarial review (2026-11-10).
(A) Direction of an undetected bf16 GDN *training* handicap (RR-R5-power) on L3 and R3 under two allocation models.
    L3 = S3 - 0.2*S0 - 0.8*S12; R3 = (S3 - S0)/(S12 - S0). Handicap delta (pp) at k = 0.
    Model 'k0-only'   : only the all-GDN model loses delta (hybrids route retrieval through attention).
    Model 'per-layer' : each config loses delta*(12-k)/12 (handicap proportional to GDN layer count).
(B) Single-seed / two-seed FEAS decisions: P(misclassification) under a normal model with residual sigma per run.
    FEAS-5 SSMax: one paired difference (seed 101), adopt iff gain >= 3 pp at 32K (ignoring the 4K clause).
    FEAS-3 budget: seed-averaged (101, 102) difference, thresholds 3 / 10 pp.
"""
import numpy as np
from scipy.stats import norm
G = 30.0
print("(A) RR-R5-power direction (G = 30 pp, S0 = 40)")
for R3 in (0.75, 0.80, 0.85):
    for d in (3.0, 6.0, 11.5):
        S0, S12 = 40.0, 70.0; S3 = S0 + R3 * G
        for model in ("k0-only", "per-layer"):
            d0 = d; d3 = 0.0 if model == "k0-only" else d * 9 / 12; d12 = 0.0
            s0, s3, s12 = S0 - d0, S3 - d3, S12 - d12
            L_true = S3 - 0.2 * S0 - 0.8 * S12; L_obs = s3 - 0.2 * s0 - 0.8 * s12
            print(f"  R3={R3:.2f} delta={d:4.1f} {model:9s}: L3 {L_true:+.2f} -> {L_obs:+.2f} (shift {L_obs-L_true:+.2f} pp); "
                  f"R3 {R3:.3f} -> {(s3-s0)/(s12-s0):.3f}")
print("\n(B) FEAS decision noise (normal approx.)")
for sig in (2.0, 3.0):
    sd1 = sig * np.sqrt(2)          # one paired difference of two runs
    sd2 = sig                       # mean of two paired differences
    print(f"  sigma={sig}: FEAS-5 P(adopt | true gain 0) = {norm.sf(3/sd1):.2f}; P(not adopt | true gain 5) = {norm.cdf((3-5)/sd1):.2f}; "
          f"P(not adopt | true gain 8) = {norm.cdf((3-8)/sd1):.2f}")
    print(f"           FEAS-3 P(pick > smallest | true 0) = {norm.sf(3/sd2):.2f}; P(pick smallest | true 6) = {norm.cdf((3-6)/sd2):.2f}; "
          f"P(pick largest | true 6) = {norm.sf((10-6)/sd2):.2f}; P(not largest | true 12) = {norm.cdf((10-12)/sd2):.2f}")

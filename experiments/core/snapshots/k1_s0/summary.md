# Core runs: descriptive summary

UNREGISTERED, EXPLORATORY (D103-D105): FEAS entry BLOCKED; no LR sweep or freeze; development items; no hypothesis test, no verdict.

Retrieval ladder, exact match (mean over seeds; n = seeds)

| k | n | 4K | 8K | 16K | 32K |
|---|---|---|---|---|---|
| 0 | 3 | 0.122 | 0.105 | 0.097 | 0.096 |
| 1 | 1 | 0.978 | 0.976 | 0.958 | 0.863 |
| 3 | 3 | 0.960 | 0.951 | 0.914 | 0.803 |
| 12 | 3 | 0.974 | 0.966 | 0.933 | 0.846 |

Per run: ladder 32K / no-filler control 32K / M-hop 32K / A-freq 32K / P2 flip rate

- k0_s0: 0.087 / 0.094 / 0.248 / 0.598 / 0.0
- k1_s0: 0.863 / 0.961 / 0.247 / 0.554 / 0.0
- k3_s0: 0.732 / 0.939 / 0.259 / 0.621 / 0.0031914893617021275
- k12_s0: 0.883 / 0.873 / 0.254 / 0.606 / 0.0
- k0_s1: 0.096 / 0.100 / 0.245 / 0.589 / 0.0010638297872340426
- k3_s1: 0.823 / 0.951 / 0.256 / 0.505 / 0.002127659574468085
- k12_s1: 0.790 / 0.952 / 0.241 / 0.564 / 0.0010638297872340426
- k0_s2: 0.106 / 0.125 / 0.249 / 0.533 / 0.002127659574468085
- k3_s2: 0.854 / 0.961 / 0.243 / 0.520 / 0.0010638297872340426
- k12_s2: 0.866 / 0.960 / 0.246 / 0.508 / 0.0010638297872340426

Recovery R(k) at 16K and 32K per seed (blank if the 12-minus-0 gap is 5 points or less)

- k1 16K: s0: R=1.00 (gap 0.865)
- k1 32K: s0: R=0.98 (gap 0.795)
- k3 16K: s0: R=0.89 (gap 0.865), s1: R=1.04 (gap 0.803), s2: R=1.01 (gap 0.839)
- k3 32K: s0: R=0.81 (gap 0.795), s1: R=1.05 (gap 0.693), s2: R=0.98 (gap 0.760)

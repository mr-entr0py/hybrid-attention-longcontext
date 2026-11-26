# Core runs: descriptive summary

UNREGISTERED, EXPLORATORY (D103-D105): FEAS entry BLOCKED; no LR sweep or freeze; development items; no hypothesis test, no verdict.

Retrieval ladder, exact match (mean over seeds; n = seeds)

| k | n | 4K | 8K | 16K | 32K |
|---|---|---|---|---|---|
| 0 | 2 | 0.118 | 0.100 | 0.095 | 0.092 |
| 3 | 2 | 0.947 | 0.936 | 0.895 | 0.777 |
| 12 | 2 | 0.969 | 0.961 | 0.929 | 0.836 |

Per run: ladder 32K / no-filler control 32K / M-hop 32K / A-freq 32K / P2 flip rate

- k0_s0: 0.087 / 0.094 / 0.248 / 0.598 / 0.0
- k3_s0: 0.732 / 0.939 / 0.259 / 0.621 / 0.0031914893617021275
- k12_s0: 0.883 / 0.873 / 0.254 / 0.606 / 0.0
- k0_s1: 0.096 / 0.100 / 0.245 / 0.589 / 0.0010638297872340426
- k3_s1: 0.823 / 0.951 / 0.256 / 0.505 / 0.002127659574468085
- k12_s1: 0.790 / 0.952 / 0.241 / 0.564 / 0.0010638297872340426

Recovery R(k) at 16K and 32K per seed (blank if the 12-minus-0 gap is 5 points or less)

- k3 16K: s0: R=0.89 (gap 0.865), s1: R=1.04 (gap 0.803)
- k3 32K: s0: R=0.81 (gap 0.795), s1: R=1.05 (gap 0.693)

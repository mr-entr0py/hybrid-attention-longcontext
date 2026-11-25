# 007: Our attention and Gated DeltaNet layers inside Zoology's harness (R4 / gate G-RP(b))

**Status:** reproduced · **Date:** 2026-11-10 (07:25–08:31 IST) · **First commit:** `9d4bd98`

## Why
Experiment 001 validated the environment with Zoology's own layers. This experiment swaps in **our** layers and nothing else (data, training loop, seeds and evaluation stay Zoology's). Any difference can then be attributed to our implementation. It only runs after the weight-copy test VT-25 (experiment 005) passed.

## Spec
- Zoology @`1ad20d1` harness; seed 123; same widths and learning rates as 001.
- Cells:
  - our attention (configured like Zoology's MHA: no gate, no QK-norm, no RoPE) at d64 and d128;
  - our GDN wrapper at d128 and d256.
- **R4b smoke test:** the production attention layer (gate, QK-norm, RoPE) in fp32 and bf16.
- Pre-registration: `experiments/repro/R4/prereg.yaml`.

## Procedure
```bash
bash scripts/repro/r4_launch.sh
python scripts/s1_zoology/r4_analysis.py      # writes r4_results.json and r4_table.md
```

## Results
| Cell | Δ(ours − Zoology), overall accuracy | Tolerance |
|---|---|---|
| Attention d64 | −0.0001 | ±0.03 |
| Attention d128 | −0.0001 | ±0.03 |
| GDN d128 | −0.0004 | ±0.03 |
| GDN d256 | −0.0002 | ±0.03 |

- On the hardest cell (1024/256), our GDN scores 0.922 against Zoology's 0.925 (d128) and 0.955 against 0.957 (d256).
- R4b: the production attention layer reaches 1.000 in-distribution in fp32 and bf16.
- Exploratory: the gated GDN-A geometry scores 0.856 at d256 on the 1024/256 cell. This is a single seed and supports no claim.
- An Xid-8 fault occurred during the exploratory run; no gate run was affected.

Raw data: `experiments/repro/R4/`. Report: `reports/reproduction/results_R4.md`.

## Interpretation
Our layers behave like the reference layers to within 0.0004. Because most cells are at the ceiling, the test can only detect gaps of about 0.21 on the one live cell. Experiment 008 is the more sensitive comparison.

## Status
Reproduced.

# 006: LAMBADA on published checkpoints through our scorer (R7)

**Status:** partially reproduced · **Date:** 2026-11-10 (05:52–06:27 IST) · **First commit:** `9d4bd98`

## Why
Our scorer turns model log-probabilities into accuracies. Scoring public checkpoints with published LAMBADA numbers tests the scoring path end to end. The pure-GDN checkpoint additionally tests that fla 0.6.0 reproduces the original Gated DeltaNet semantics.

## Spec
- **Task:** `lambada_openai` (5,153 items); last-word exact match; fp32; batch 1.
- **Checkpoints:**
  - EleutherAI Pythia-160M and Pythia-70M (step 143000);
  - Wang et al.'s 340M GDN-pure model.
- **Anchors and pre-registered intervals:**
  - Pythia-160M 0.328 ± 0.015;
  - Pythia-70M 0.185 ± 0.015;
  - GDN-340M 0.348 ± 0.02.

## Procedure
```bash
setsid nohup bash scripts/repro/r7_launch.sh > experiments/repro/R7/logs/launch.log 2>&1 &
python scripts/repro/r7_compare.py
```

## Results
| Model | Ours (fp32) | Anchor interval | Verdict | fp16 weights (diagnostic) |
|---|---|---|---|---|
| Pythia-160M | 0.3538 | [0.313, 0.343] | FAIL | 0.3297 |
| Pythia-70M | 0.2271 | [0.170, 0.200] | FAIL | 0.1849 |
| GDN-340M | 0.3435 | [0.328, 0.368] | PASS | — |

- lm-evaluation-harness @`d6de8164` gives **exactly** our numbers: 0 disagreements over 10,306 items.
- An independent third scorer written by the auditor agrees as well.

Raw data: `experiments/repro/R7/`. Report: `reports/reproduction/results_R7.md`.

## Interpretation
- The scorer is correct.
- The published Pythia numbers were produced with fp16 weights; the 2023 harness loads `config.torch_dtype`. With fp16 weights we reproduce them to within 0.0014 / 0.0004.
- The plan had assumed fp16-vs-fp32 drift would fit inside ±0.015. It does not, so the criterion fails as registered.
- The GDN checkpoint passing shows that fla 0.6.0 matches the semantics the model was trained with.

## Status
Partially reproduced. The Pythia criteria are recorded as residual risk RR-R7; they don't block the pilots (ruling D076).

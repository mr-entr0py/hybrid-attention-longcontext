# 008: MQAR below ceiling at d = 64, Zoology's GDN vs ours (R5)

**Status:** reproduced · **Date:** 2026-11-10 · **First commit:** `9d4bd98`

## Why
Experiments 001 and 007 sit mostly at the accuracy ceiling. At d = 64 the GDN state is small enough that recall drops well below the ceiling. That gives a setting where (i) the published recall-vs-state trade-off can be seen, and (ii) a real difference between our implementation and the reference could show up.

## Spec
- Zoology harness @`1ad20d1`; GDN at d = 64 (state 17,152 values); LR 1e-3.
- Seeds 123, 124 and 125 for both Zoology's GDN and ours, with ours under bf16 autocast: 6 runs.
- Pre-registered rules, on the 1024/256 cell:
  - accuracy increases with width;
  - |Δ(ours − Zoology)| ≤ max(0.03, 2.78·SE);
  - bimodal seeds → inconclusive.
- Pre-registration: `experiments/repro/R5/prereg.yaml`.

## Procedure
```bash
bash scripts/repro/r5_launch.sh                # detached, resumable
python scripts/s1_zoology/r5_analysis.py       # writes r5_results.json and r5_table.md
```

## Results
| Run | Seed 123 | Seed 124 | Seed 125 | Mean (1024/256) |
|---|---|---|---|---|
| Zoology GDN | 0.282 | 0.347 | 0.260 | 0.296 |
| Our GDN (bf16) | 0.230 | 0.339 | 0.267 | 0.279 |

- Width ordering on 1024/256: 0.296 (d64) < 0.925 (d128) < 0.957 (d256). **PASS.**
- Δ = −0.0175, SE_W = 0.0415, threshold 0.115; 90% CI [−0.107, +0.072]. **PASS.**
- Bimodality rule did not fire.

Raw data: `experiments/repro/R5/`. Figure: `docs/figures/mqar_reproduction.png`.

## Interpretation
The recall/state trade-off that motivates the whole study reproduces on this hardware, and our implementation tracks the reference within seed noise. Power is low: the test excludes only a bf16 handicap larger than about 0.115 (at 50% power). The run started after a soak-test trigger; the PI accepted it in ruling D079, with residual risk RR-R5-power.

## Status
Reproduced.

# 005: Exactness against independent references (R3, VT-21 to VT-25)

**Status:** not passed as registered (VT-25 PASS) · **Date:** 2026-11-10 (05:45–07:05 IST) · **First commit:** `9d4bd98`

## Why
Unit tests written by the same people who wrote the code can share their blind spots. These five tests compare our components with *independent* references at the production lengths (4K–32K).

## Spec
Pre-registration: `experiments/repro/R3/prereg.yaml` (criteria copied from the reproduction plan §2.3).

| Test | Reference | Bound |
|---|---|---|
| VT-21 | Independent causal-conv implementation | 1e-4 absolute (fp32) |
| VT-22 | Our models vs Qwen3 / fla reference LMs at 4K–32K | 2e-2 relative RMS (bf16) |
| VT-23 | Our trainer vs an independent minimal training loop | 1e-5 for steps 2–10 |
| VT-24 | An oracle that must answer every 16K/32K item from its context | EM = 1 |
| VT-25 | Zoology mixers vs ours with copied weights | 1e-2 logits, 2e-2 gradients |

## Procedure
```bash
bash scripts/repro/r3_launch.sh vt21 3500 tests/test_vt21_conv_reference.py
bash scripts/repro/r3_launch.sh vt22 3500 tests/test_vt22_long_transplant.py   # likewise vt23, vt24, vt25
```

## Results
| Test | Verdict | Key numbers |
|---|---|---|
| VT-21 | FAIL | 372/384 pass. 12 fp32 weight gradients at 32K: 1.03–1.40e-4 absolute, where the gradient magnitude is ≈ 10³ (relative 1–2e-7, 1–2 fp32 ulp) |
| VT-22 | FAIL | All-GDN model at 16K/32K: 2.021e-2 / 2.004e-2 vs a 2e-2 bound. Each implementation's own bf16 drift is 2.09e-2. The error is flat across position |
| VT-23 | FAIL | Bitwise identical to the reference for steps 1–4; afterwards the two diverge chaotically from summation order. The reference diverges from itself by more under a reordering alone |
| VT-24 | FAIL | 39,793 of 39,800 queries answerable. The 7 misses are value-prefix collisions between two facts, a genuine generator flaw |
| VT-25 | PASS | GDN logits ≤ 4.9e-4, gradients ≤ 1.82e-2. Attention logits ≤ 3.3e-8 |

Raw data: `experiments/repro/R3/`. Report: `reports/reproduction/results_R3.md`. Independent audit: `reports/reproduction/audit_R2-R7.md`.

## Interpretation
Three of the four failures come from tolerances that were never calibrated against the real configuration. They are not code defects, and the audit confirmed this from the raw files. VT-24 exposed a real defect: two facts' values could share a prefix. The generator was fixed and the items regenerated; the new check VT-24r is pre-registered.

Per the integrity rule, failed criteria are not restated. They are recorded as named residual risks (RR-VT21, RR-VT22, RR-VT23) or followed by new pre-registered checks (VT-22b, VT-23b, VT-24r).

## Status
Not passed as registered. Routes are set in `reports/pi_rulings_2026-11-10.md` §4.

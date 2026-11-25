# 001: Zoology MQAR with Zoology's own code (R1 / gate G-RP(a))

**Status:** reproduced · **Date:** 2026-11-09 (17:35–21:00 IST) · **First commit with results:** `58e934e`

## Why
Before trusting our own models on recall tasks, we reproduce the canonical small-scale recall result: on multi-query associative recall (MQAR), attention solves the task while a fixed-state recurrent mixer degrades as the number of key-value pairs grows, and recovers as the state grows (Arora et al. 2023, *Zoology*). This tests the environment and hardware (sm_120, torch 2.11, fla 0.6.0), not our code.

## Spec
- **Code:** HazyResearch/zoology @ `1ad20d193b6113cae1e8f3c655c300d7b4b3f4bb`, clean. One API shim: fla 0.6.0 rejects the `head_first` argument (`scripts/s1_zoology/zoology_compat.py`).
- **Config:** `zoology/experiments/mqar_example_configs/original_mqar_configs.py`, unchanged.
  - Vocabulary 8,192, 2 layers, batch 256, 32 epochs, early stop at validation accuracy > 0.99.
  - Learning rates: logspace(−3, −1.5, 4).
- **Models:**
  - attention, 2 heads, d ∈ {64, 128};
  - Gated DeltaNet, 2 heads, d ∈ {128, 256}.
- **Runs:** 16 (2 families × 2 widths × 4 LRs). Seed 123 (Zoology's own).
- **Hardware:** RTX 5080 16 GB; 1.48 GPU-h.

## Procedure
```bash
uv pip install -e third_party/zoology --no-deps --no-build-isolation
bash scripts/s1_zoology/launch.sh                  # detached, resumable queue
python scripts/s1_zoology/summarize.py             # writes s1a_results.json and s1a_table.md
```

## Results
Best over LR, overall test accuracy across 7 test configurations. Raw data: `experiments/s1_zoology/runs/`, `s1a_results.json`.

| Model | d | Overall accuracy | 1024/256 cell |
|---|---|---|---|
| Attention | 64 | 0.99998 | 1.000 |
| Attention | 128 | 0.99995 | 1.000 |
| Gated DeltaNet | 128 | 0.98924 | 0.925 |
| Gated DeltaNet | 256 | 0.99377 | 0.957 |

GDN diverged at LR 3.2e-2 (accuracy 0.005). Pre-registered criterion (a): attention ≥ 0.95 at d ≥ 64, GDN ≥ 0.95 at d ≥ 128, GDN non-decreasing in d. **PASS.**

Figure: `docs/figures/mqar_reproduction.png`.

## Interpretation
The environment reproduces the published qualitative pattern: attention is at ceiling, and GDN loses accuracy only on the longest, most loaded test cell and improves with width.

What this does **not** establish: that *our* layer implementations match (see 007, 008). Most cells sit at the ceiling, so this is a smoke test with little power to detect small differences.

## Status
Reproduced. Gate report: `reports/gates/G-RP.md`.

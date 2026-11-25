# 003: Training throughput and memory on an idle GPU

**Status:** measured · **Date:** 2026-11-10 · **First commit:** `d4791cd`

## Why
The GPU-hour budget in the protocol was based on throughput measured while a simulator shared the GPU. Idle measurements set realistic run times per configuration and confirm the memory cap holds at 32K.

## Spec
- Scales D384 and D512; configs C0, C3, C12.
- Two shapes:
  - Stage A: 2 × 4K tokens, no activation checkpointing;
  - Stage B: 1 × 32K tokens, with activation checkpointing.
- 5 warm-up steps and 20 timed steps; memory fraction 0.9.
- Production code path: bf16 autocast, varlen attention, fla chunk kernels.

## Procedure
```bash
python scripts/bench_train_step.py --scales D384 D512 --configs C0 C3 C12 --shapes 4k 32k \
    --steps 20 --warmup 5 --mem-fraction 0.9 --out experiments/bench/train_step_idle_20261110.json
```
The D384 rows were re-run on a quiet GPU (`train_step_idle_20261110_D384.json`), because the first pass overlapped another project job.

## Results
Median tokens per second.

| Scale | Stage A (4K), k = 0 / 3 / 12 | Stage B (32K), k = 0 / 3 / 12 | Peak memory |
|---|---|---|---|
| D384 | 127K / 129K / 138K | 107K / 98K / 78K | ≤ 4.5 GiB |
| D512 | 97K / 98K / 99K | 79K / 72K / 56K | ≤ 5.8 GiB |

Figure: `docs/figures/training_throughput.png`.

## Interpretation
- At 4K, attention is cheap relative to the MLP and the projections, so k barely affects speed.
- At 32K, each attention layer costs throughput. Going from k = 0 to k = 12 is −27% at D384 and −29% at D512.
- Compute per token is therefore **not matched** across k. This is a stated limitation of the design.

## Status
Measured. Budget re-pricing is in `reports/pi_rulings_2026-11-10.md`.

# Scaling beyond 50M parameters: discussion and proposal (2026-11-24)

**Status: proposal, not pre-registered, nothing run.** Written from a team discussion; the numbers are extrapolations from
`scripts/planning/budget.py` (D384 / D512 per-run hours on an idle RTX 5080) and must be re-measured.

## Why consider more than 100M parameters
- Models under 100M are far from current long-context models. The report already scopes every conclusion to 28-49M
  parameters; a larger point narrows the extrapolation gap and tests whether the pattern (cost grows with k; recall
  saturates) still holds.
- The 5080 has the memory. Parameters + AdamW at ~105M need ~1.7 GB; the 32K activation-checkpointed step peaked at
  6.0 GiB at 49M and should stay well inside 16 GB on an exclusive GPU (the 7 GiB per-process cap exists for the shared display).
- The limit is time, not memory (see below), so the larger scale is a **secondary, clearly labelled scale check**, plus an
  **analytic (theoretical) extrapolation** that costs no GPU time.

## Suggested size
**D768, 12 layers, d_ff 2304: about 105M parameters** (non-embedding ~92M; ~2.2x the compute per token of D512).
Larger (e.g. ~300M) would take 25-30 h per run and is not feasible before the 27th.

| per run, same token budget as D512 (0.8B Stage A) | k = 0 | k = 3 | k = 12 |
|---|---|---|---|
| D512 (planner) | 3.6 h | 4.0 h | 4.8 h |
| D768 (extrapolated x2.2) | ~8 h | ~9 h | ~11 h |

A single seed over k in {0, 3, 12} is ~28 h. An LR sweep at this scale would cost ~9 h more, so the proposal reuses the D512
learning rate scaled by width and records that as a deviation.

## Proposed use, in priority order
1. **Analytic scaling study (no GPU, DONE: see docs/SCALING_STUDY.md).** Extend `scripts/planning/flops_state.py` to d = 768, 1024, 2048 and layer counts
   of 12/24/32: inference-state memory M(k, T) = k*4Td + (L-k)*S_lambda, FLOPs/token, and the largest batch at 32K in a fixed
   memory budget. Report where the full-attention KV cache overtakes the linear state at each size.
2. **FEAS-level scale probe (~7 h):** D768, C0 and C12, one seed, 0.3B tokens. It answers the main scientific risk in the
   report (is the all-attention reference competent at 16K/32K?) at a size closer to real models.
3. **Full single-seed k in {0, 3, 12} at D768 (~28 h)** only if the main D384/D512 core blocks (3 seeds) are already done.

## Protocol rules that apply
- A new scale is a protocol amendment: write a decision-log row and a pre-registration **before** the first run; label results
  secondary/exploratory; never tune toward a result.
- One seed cannot support the paired tests, so D768 results are descriptive (recovery R(k), cost frontier), not H1-H6 verdicts.
- Runs stay on the 5080 (the 4060 is ~3.7x slower and has 8 GB).

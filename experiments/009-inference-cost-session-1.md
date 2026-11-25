# 009: Inference cost benchmarks, session 1 of 2

**Status:** measured · **Date:** 2026-11-10 (13:41–14:52 IST) · **First commit:** `1e19f21`

## Why
RQ-D asks what each full-attention layer costs at inference. Speed does not depend on the trained weights, so cost can be measured early on randomly initialised models while the GPU is free. The protocol requires two sessions on different days, and this is the first.

## Spec
- **Configs:** C0, C1, C2, C3, C4, C6, C12, X-gc12, X-E0. **Scales:** D384 and D512.
- **Prefill:** B ∈ {1, 8}, T ∈ {1K, 4K, 16K, 32K}.
- **Decode:** L ∈ {1K, 4K, 16K, 32K}, B ∈ {1, 8, B_max}.
- **Method:**
  - exact-shape CUDA graphs with the attention backend pinned;
  - cold and hot L2;
  - 5 warm-up and ≥ 30 timed replays, min-of-N reported.
- **Idle criterion:** no foreign SM activity, polled every 5 s.

## Procedure
```bash
python scripts/benchmark_inference.py run --out experiments/bench/cost_session1 \
    --scales D384 D512 --configs C0 C1 C2 C3 C4 C6 C12 X-gc12 X-E0
```
Tables are regenerated from `rows.jsonl` by the script's summarise command; see `reports/cost_benchmarks_session1.md`.

## Results (D384, idle-compliant rows)
| | k = 0 | k = 3 | k = 12 |
|---|---|---|---|
| Prefill, B = 1, 32K | 52.0 ms | 77.4 ms | 154.3 ms |
| Decode per token, B = 1, 1K → 32K | 0.87 → 0.87 ms | 0.95 → 1.13 ms | 1.17 → 1.84 ms |
| Largest batch within 8 GiB at 32K | 5,185 | 56 | 14 |

- Measured inference-state bytes equal the analytic formula to the byte: 72/72 prefill caches and 768/768 decode buffers.
- 1,678 timed series; 0 replays discarded.

Figure: `docs/figures/inference_cost.png`. Raw data: `experiments/bench/cost_session1/`.

## Interpretation
- With k = 0, inference memory is constant and decode time is flat in context length.
- Each full-attention layer adds a KV cache that grows linearly with context, plus attention time. At 32K, k = 12 holds 370× fewer concurrent sequences than k = 0 in the same memory.
- These numbers are specific to this implementation (fla Triton kernels vs PyTorch SDPA/FA2 on sm_120, with unequal kernel maturity). The analytic costs are primary.
- Some D512 series overlapped desktop GPU activity and are appendix-only.

## Status
Measured. Session 2 (on a different day, with an idle GPU) is still owed.

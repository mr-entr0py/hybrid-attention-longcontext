# 002: Infrastructure verification suite VT-01 to VT-20 (gate G0)

**Status:** verified (G0 recorded as *not passed as registered* with logged deviation DEV-02) · **Date:** 2026-11-10 · **First commit:** `d6568c5`

## Why
A completed training run is not evidence of a correct experiment. Before any study run, every component is checked against fixed numeric thresholds (protocol §16):
- masking and document isolation;
- recurrent-vs-chunked equivalence;
- cached-vs-uncached decoding;
- weight transplants into reference implementations;
- determinism and resume;
- data hygiene and item validity;
- memory.

## Spec
- Code: `src/hybridlc/`. Tests: `tests/test_vt01_*.py` … `tests/test_vt20_*.py`.
- Environment: `requirements-lock.txt`.
- Hardware: RTX 5080 (sm_120).
- Thresholds: frozen in `reports/experimental_protocol.md` §16.

## Procedure
```bash
python scripts/repro/gpu_lock.py run --label G0-full-suite --timeout 3600 --wait-exclusive -- \
    python -m pytest tests/ -q -rfEs -p no:cacheprovider
```

## Results
- **447 of 447** VT-01…VT-20 tests pass. The suite had 456 tests at G0 time; it now has 601, after later regression tests were added.
- **Weight transplant:**
  - our all-attention model vs `transformers.Qwen3ForCausalLM`: max relative logit error **7.0e-7**;
  - our all-GDN model vs `fla` `GatedDeltaNetForCausalLM`: **4.6e-4** (bound 1e-3).
- **Inference-state bytes** equal the analytic formula to the byte at 1K and 32K.
- **Cached decode vs a full forward pass** (fp32): argmax agreement 1.000, max |Δlogit| ≤ 3.9e-3.
- **Peak memory:** 6.00 GiB allocated, against a 7 GiB cap.
- **Defects found by independent verifiers and fixed:**
  - the no-filler multi-hop control was unshuffled;
  - task-exposure replay crossed document boundaries;
  - a missing "still closing" rule;
  - about 12.7K held-out documents duplicated training text and were removed.

Reports: `reports/s0_integration.md`, `reports/gates/G0.md`.

## Interpretation
The model, data, evaluation and training code behave as specified. Four tests pass only under a stated reading of their threshold, and the PI ruled that G0 is recorded as "not passed as registered" rather than relabelled (`reports/pi_rulings_2026-11-10.md`, D058). The named residual risks are tracked in that file.

## Status
Verified, with named residual risks.

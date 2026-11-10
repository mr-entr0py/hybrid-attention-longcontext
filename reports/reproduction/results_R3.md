# R3 results: reference-reproduction battery VT-21 … VT-25 (gate G-RV)

**Executed:** 2026-11-10, 05:45–07:05 IST, by the R3 executor (workflow "Reproduction execution").
**Pre-registration:** `experiments/repro/R3/prereg.yaml`, written at 05:45 before any VT-21…25 test existed. It copies the criteria of `reports/reproduction/reproduction_plan.md` §2.3 (SHA-256 `6bc14af6…cebc`, which matches the DEV-01 row) and fixes how I read the ambiguous parts. No criterion was changed after a result was seen.
**Code state:** HEAD `110fd9e`, with a dirty tree. The S0 integrator is editing `src/hybridlc/**` at the same time, and every `_meta.json` records `git status`. I edited nothing under `src/`.
**Environment:** torch 2.11.0+cu128, triton 3.6.0, fla 0.6.0, transformers 5.22.2, RTX 5080 (sm_120), driver 595.99.02. The full `env_info` is in each `_meta.json`.
**GPU:** shared throughout. At various times it held the S0 VT-13 production runs (2 × 5.2 GB, later 1 × 5.2 GB), the R7 runs (≈ 2 GB) and an lm-eval diagnostic, alongside my jobs. All processes ran at the 0.44 memory cap. No Xid, NVRM, OOM-killer or watchdog lines appeared in `journalctl -k` during any run window; each `_meta.json` has a `kernel_events: []` field.

## Verdicts

| Item | Verdict | Key numbers (criterion) | Raw output |
|---|---|---|---|
| VT-21 (R3b) fla `causal_conv1d` vs independent reference | **FAIL** | 372 / 384 cases pass. All 12 failures are fp32 **dw** at T = 32,768: max abs 1.03–1.40e-4 (bound 1e-4), relative 1.1–2.3e-7. Forward ≤ 7.2e-7, dx ≤ 5.5e-7, bf16 ≤ 1.9e-3 (≤ 1e-2), leakage exactly 0, state exact | `experiments/repro/R3/vt21/vt21_conv.r1.jsonl` + `vt21_conv.r3.jsonl` (+ `.r3_meta.json`) |
| VT-22 (R3c) C12↔Qwen3, C0↔fla LM, 4K–32K; C3 packed | **FAIL** | C12: 4K rel max 4.3e-6; 16K/32K rel RMS 6.4e-3 / 6.9e-3, bands flat. C0: 4K rel max 4.5e-4; 16K/32K rel RMS **2.021e-2 / 2.004e-2** (bound 2e-2), bands flat (1.06 / 1.03). C3 packed vs alone: bitwise equal | `experiments/repro/R3/vt22/vt22_transplant.jsonl` |
| VT-23 (R3d) trainer vs independent loop | **FAIL** | Step-1 loss 4.9e-8 (≤ 1e-6) ✔; steps 2–10 max **3.0e-4** (≤ 1e-5) ✘; steps 11–50 max **1.77** (≤ 1e-3) ✘; final params **0.194** (≤ 1e-3) ✘; clipping exercised ✔; WD groups identical ✔. Isolation: bitwise-identical updates for steps 1–4 once the loss function and clip reduction are matched; a pure rounding change to the reference alone gives 3.0e-3 / 0.79 / 0.188 | `experiments/repro/R3/vt23/vt23_compare.r1.jsonl`, `attempt_20261110T062646/` |
| VT-24 (R3e) oracle answerability, 16K/32K | **FAIL** | 39,793 / 39,800 retrieval queries EM = 1 with logp = 4 ln 0.9. 7 misses in 3 files, all order-6 n-gram collisions between two in-context facts. M-hop EM = 0 in all 25 files ✔. Metadata oracle 39,800 / 39,800 ✔ | `experiments/repro/R3/vt24/run_20261110T055803/` |
| VT-25 (R3a) Zoology ↔ ours, weight copy | **PASS** | GDN matched-bf16: logits ≤ 4.9e-4 (≤ 1e-2), worst per-tensor grad 1.82e-2 (≤ 2e-2). Attention fp32: logits ≤ 3.3e-8 (≤ 1e-5), grads ≤ 5.9e-7 (≤ 1e-4) | `experiments/repro/R3/vt25/vt25_transplant.jsonl` |

**Effect on G-RV.** G-RV requires all of VT-21…VT-25 to pass, so with the verdicts above **G-RV does not pass on R3 as run**. Every failure is diagnosed below. Under plan §4.1 ("Fix only with a test that isolates the defect"), none of them isolates a code defect in `src/hybridlc`. They therefore go to the PI as decision items (§ "Requests to the PI"). I did not tune or re-run anything to change a verdict.

---

## VT-21 (R3b): fla `causal_conv1d` vs an independent reference: **FAIL** (12 of 384 cases: fp32 weight-gradient absolute bound at T = 32,768)

**Command:** `bash scripts/repro/r3_launch.sh vt21 3500 tests/test_vt21_conv_reference.py`, which runs `pytest -q -s tests/test_vt21_conv_reference.py`. Log: `experiments/repro/R3/logs/vt21_20261110T055607.log`, 05:56:07 to 06:46, cases 0–142, raw `experiments/repro/R3/vt21/vt21_conv.r1.jsonl`.

I stopped that run at 143 rows. Its float64 reference ran on the CPU at load average ≈ 43 and would have needed ≈ 4 h more. I then resumed with `R3_VT21_RESUME_FROM=experiments/repro/R3/vt21/vt21_conv.r1.jsonl bash scripts/repro/r3_launch.sh vt21resume 3000 tests/test_vt21_conv_reference.py`:
- log: `experiments/repro/R3/logs/vt21resume_20261110T064817.log`, 06:48:20–06:58:45;
- raw: `experiments/repro/R3/vt21/vt21_conv.r3.jsonl` + `vt21_conv.r3_meta.json` (cases 143–383, the 8 supplementary cases, and a summary over the union of all 384 main cases).

The resume changed only where the float64 reference runs: GPU, in independent 256-channel blocks, with exact streaming max and sum-of-squares accumulators. Before the switch, I re-ran three completed cases (3, 60, 120) with the new reference, and every metric agreed with the r1 rows to all printed digits. Same seeds and case ids; no criterion touched.

**Grid** (exactly as pre-registered): 384 cases, all forward and backward.
- channels {192, 384, 1152} (D384) and {256, 512, 1536} (D512);
- T ∈ {37, 300, 4,096, 32,768};
- dense B = 2, and packed (offsets 13/77/190/1,000/2,731/5,003/16,411; int32 `cu_seqlens` for the F_gc width, int64 for GDN widths, as the two production callers pass them);
- `initial_state` on and off;
- SiLU on and off;
- fp32 and bf16.

The entry point is fla's public `causal_conv1d(..., backend='triton')`. fla's backend dispatcher logged no override; only "Blackwell detected: registering default global_scratch allocator" appeared.

| Criterion | Measured (worst over the grid) | Result |
|---|---|---|
| fp32 forward max abs ≤ 1e-5 | y: **7.2e-7**; final state: **0** (exact) | PASS |
| fp32 dx max abs ≤ 1e-4 | **5.5e-7** | PASS |
| fp32 dw max abs ≤ 1e-4 | **1.40e-4**. Fails in **12 of the 48** fp32 cases at T = 32,768 (1.03e-4 to 1.40e-4); worst at T ≤ 4,096 is 5.1e-5 | **FAIL** |
| bf16 rel RMS ≤ 1e-2 | y 1.7e-3, dx 1.8e-3, dw 1.9e-3, final state 0 | PASS |
| Cross-document leakage exactly zero | all 192 packed cases: other documents' outputs and final states bitwise unchanged when document 0 is perturbed, and dx exactly 0 outside the document that carries the upstream gradient | PASS |
| Final conv state matches the reference | exact (0) in every case, fp32 and bf16 | PASS |

Report-only:
- the initial-state gradient dh0 has max abs 3.1e-7 in fp32 and rel RMS 1.9e-3 in bf16;
- `hybridlc.models.attention.causal_dwconv_reference` agrees with the independent reference to 4.4e-16 (y) and 0 (final state);
- the supplementary packed case with documents of length 2 and 3 (shorter than the kernel) passes all 8 variants.

**What the failure is.** The 12 failing cases are all fp32 weight gradients at T = 32,768:

| Case | Layout | Initial state | Activation | dw max abs | dw rel max |
|---|---|---|---|---|---|
| D=192 | packed | off | none | 1.27e-4 | 2.3e-7 |
| D=384 | dense | off | none | 1.07e-4 | 1.3e-7 |
| D=384 | dense | on | none | 1.03e-4 | 1.2e-7 |
| D=1152 | dense | off | none | 1.14e-4 | 1.3e-7 |
| D=1152 | dense | on | none | 1.32e-4 | 1.2e-7 |
| D=1152 | packed | on | none | 1.03e-4 | 1.5e-7 |
| D=256 | dense | off | none | 1.03e-4 | 1.3e-7 |
| D=256 | packed | off | none | 1.16e-4 | 2.3e-7 |
| D=512 | dense | off | none | 1.28e-4 | 1.1e-7 |
| D=1536 | dense | off | none | 1.40e-4 | 1.3e-7 |
| D=1536 | dense | on | none | 1.22e-4 | 1.2e-7 |
| D=1536 | packed | off | none | 1.19e-4 | 1.8e-7 |

- dw is a sum over B·T = 32K–65K products, so |dw| reaches 330–850 with N(0, 1) inputs and upstream gradients.
- The relative error is 1.1–2.3e-7, i.e. 1–2 fp32 ulps, and dw rel RMS is ≤ 1.6e-7 everywhere.
- The pre-registered reading applied the 1e-4 bound literally, as an absolute number. The plan's tolerance basis ("a 4-tap depthwise conv of O(1) values has fp32 error of about 1e-7") describes the forward pass and does not scale with the reduction length of dw.
- A wrong tap, a shift, a padding or `cu_seqlens` error, or a broken state layout would give O(1) relative errors. Nothing of that kind appears: forward, dx, state, leakage and bf16 all pass everywhere.

**Classification:** FAIL of the frozen criterion, caused by tolerance calibration (absolute dw bound vs reduction length). fla's conv kernel is correct to fp32 rounding in every configuration we use. → PI request 1.

**Concurrent external run.** At 06:23 the S0 verification workflow started `python -m pytest tests/ -q -rfEs -p no:cacheprovider` (PID 1683560). It collects the R3 test files too, and it is writing its own VT-21 rows to `experiments/repro/R3/vt21/vt21_conv.r2.jsonl`, with the CPU-reference version of the file that it loaded at collection. Those rows are not mine and are not used for any verdict here. Its VT-22…VT-25 runs, if they happen, will appear as further `.rN` or `attempt_*` / `run_*` outputs with their own PID in `_meta.json`.

---

## VT-22 (R3c): full-model transplant at 4K–32K: **FAIL** (C0 ↔ fla at 16K/32K bf16; marginal)

**Command:** `bash scripts/repro/r3_launch.sh vt22 4000 tests/test_vt22_long_transplant.py`, which runs `pytest -q -s tests/test_vt22_long_transplant.py`. Log: `experiments/repro/R3/logs/vt22_20261110T060147.log`. Raw: `experiments/repro/R3/vt22/vt22_transplant.jsonl` (+ `_meta.json`). Window 06:05:01–06:07:00, one attempt.

| Comparison | T | Precision | rel RMS | rel max | band ratio (last 1K / first 1K) | Criterion | Result |
|---|---|---|---|---|---|---|---|
| C12 ↔ Qwen3 | 4,096 | fp32 | 2.49e-6 | **4.25e-6** | 2.32 (report-only) | rel max ≤ 1e-3 | PASS |
| C12 ↔ Qwen3 (SDPA flash pinned) | 16,384 | bf16 | **6.35e-3** | 7.38e-3 | **0.99** | ≤ 2e-2 and ≤ 2 | PASS |
| C12 ↔ Qwen3 | 32,768 | bf16 | **6.88e-3** | 7.44e-3 | **0.98** | ≤ 2e-2 and ≤ 2 | PASS |
| C0 ↔ fla GDN LM | 4,096 | fp32 | 2.60e-4 | **4.54e-4** | 1.11 (report-only) | rel max ≤ 1e-3 | PASS |
| C0 ↔ fla GDN LM | 16,384 | bf16 | **2.021e-2** | 2.68e-2 | **1.06** | ≤ 2e-2 and ≤ 2 | **FAIL** (rel RMS) |
| C0 ↔ fla GDN LM | 32,768 | bf16 | **2.004e-2** | 2.31e-2 | **1.03** | ≤ 2e-2 and ≤ 2 | **FAIL** (rel RMS) |
| C3 packed (16K + 16K) vs each alone | 32,768 | bf16 | **0** (bitwise equal) | 0 | n/a | ≤ 1e-2 | PASS |

**What the failure is.** Both C0 bf16 cells exceed the 2e-2 rel-RMS bound by 0.2 % and 1 % of the bound. The position criterion passes with a flat profile: the 8-band rel RMS is 0.019–0.021 from the first band to the last at both lengths. That rules out the error that grows with position, which is what this test targets (state accumulation, `max_seqlen`).

**Post-verdict diagnosis** (report-only; `python scripts/repro/r3_vt22_diagnose.py --T 16384`, raw `experiments/repro/R3/vt22/vt22_diagnosis.jsonl`). It uses the same weights and ids as the verdict run, and the ours-bf16 ↔ fla-bf16 number reproduces exactly (2.021e-2):

| a vs reference (T = 16,384) | rel RMS | band ratio |
|---|---|---|
| ours fp32 vs fla fp32 | 2.61e-4 | 1.10 |
| ours bf16 vs fla fp32 | 2.15e-2 | 1.04 |
| fla bf16 vs fla fp32 | 2.09e-2 | 1.03 |
| ours bf16 vs fla bf16 | 2.02e-2 | 1.06 |

- In fp32 the two implementations agree to 2.6e-4.
- Under bf16 autocast, **each** implementation moves 2.1e-2 away from fp32, and our path is no worse than fla's own.
- The 2e-2 bound therefore sits at the intrinsic bf16-autocast error of a random-init 12-layer GDN LM. This is not a defect in our code.
- The tolerance was set by analogy to VT-04/VT-05 (single-kernel bf16) and never calibrated on a full LM.
- By contrast, C12 vs Qwen3 sits at 6–7e-3.

**Classification:** FAIL of the frozen criterion, caused by tolerance calibration. No code defect was isolated. → PI request 2.

---

## VT-23 (R3d): trainer vs independent reference loop: **FAIL** (trace and final-parameter tolerances; no semantic difference found)

**Command:** `pytest -q -s tests/test_vt23_reference_loop.py`, run detached through a memory-gated launcher. Log of the verdict attempt: `experiments/repro/R3/logs/vt23_20261110T062615.log`.
- Trainer run `R3-D384-C3-s900-d6613db5-vt23x20261110T062646`: own runs, ckpt and registry roots under `experiments/repro/R3/vt23/attempt_20261110T062646/`.
- Reference: `scripts/repro/r3_vt23_reference_loop.py` → `…/attempt_20261110T062646/ref/`.
- Raw comparison: `experiments/repro/R3/vt23/vt23_compare.r1.jsonl` (+ `_meta.json`). This is an offline re-comparison of the same attempt, needed because the first comparison misread the format of the trainer's WD-group summary; see the attempt log. `vt23_compare.jsonl` is the original comparison of the same run, and differs only in the WD-group check.
- Window 06:26:46–06:28:22. No kernel events.

**Configuration:**
- D384 C3, seed and data index 900, `StageStream('A')`, 2 × 4K micro-batches × 2 accumulation steps, 50 steps, 10 warm-up steps then constant at 2e-3;
- fp32, audit mode;
- **DEVIATION:** activation checkpointing on both sides. The pre-registered configuration needs more than the mandated 0.44 cap of 6.80 GiB (attempt 4: 6.58 GiB allocated at a 256 MiB request). This was recorded in the prereg addendum before any comparison existed. Measured peak with checkpointing: 2.66 GiB on both sides;
- `expandable_segments` allocator setting.

Initial parameters are identical: the reference process's SHA-256 per tensor equals that of `build_model(C3, 900)`. The LR schedule matched exactly, so my pre-declared warm-up reading lr(s) = η·min(1, (s+1)/10) is the trainer's convention.

| Criterion | Measured | Result |
|---|---|---|
| Step-1 loss rel diff ≤ 1e-6 | **4.9e-8** (step-1 grad norm 1.7e-7, report-only) | PASS |
| Steps 2–10: loss and pre-clip grad norm ≤ 1e-5 rel | loss ≤ 3.7e-6; grad norm **4.2e-5** at step 5, **3.0e-4** at step 10 | **FAIL** |
| Steps 11–50 ≤ 1e-3 rel | max **1.77** (grad norm, a step-20 spike: 1.88 vs 0.76); loss max 5.9e-3 | **FAIL** |
| ≥ 1 step with pre-clip norm > 1.0 | trainer 30 steps, reference 21 | PASS |
| Final parameters rel L2 ≤ 1e-3 | **0.194** | **FAIL** |
| Identical WD-group membership by name | identical: 142 decay / 58 no-decay tensors; no-decay = every norm gain, QK-norm, o_norm, `A_log`, `dt_bias`, `norm_f` | PASS |

The differences start at the rounding level and grow roughly ×10 every 2–3 steps: grad norm 3e-6 (step 2) → 4e-5 (5) → 3e-4 (10) → 1e-2 (14) → O(1) during the gradient spikes at steps 15–30.

**Post-verdict isolation** (report-only; `python scripts/repro/r3_vt23_diagnose.py experiments/repro/R3/vt23/attempt_20261110T062646` → `experiments/repro/R3/vt23/vt23_diagnosis.json`, `vt23_diagnosis_final_params.json`, `…/diag_*/`). Each line swaps one component of the reference, or applies a pure rounding change, and compares against the same trainer trace:

| Variant | vs | max rel, steps 2–10 | max rel, steps 11–50 | final params rel L2 |
|---|---|---|---|---|
| verdict reference (own loss, manual clip) | trainer | 3.0e-4 | 1.77 | 0.194 |
| D1: production `chunked_ce` + `clip_grad_norm_(foreach)` | trainer | 1.6e-5 | 1.20 | 0.194 |
| D2: production `chunked_ce` only | trainer | 3.0e-4 | 3.30 | 0.194 |
| **D3: own loop, CE summed in 2,048-token chunks instead of 4,096 (pure rounding change)** | **verdict reference** | **3.0e-3** | **0.79** | **0.188** |
| D1 repeated in a second process | D1 | 0 (bitwise, all 50 steps) | 0 | n/a |

What these show:
- **D1.** Gradients are **bitwise identical to the trainer for steps 1–4**: identical grad norms, with only the logged loss differing at 5e-8 through fp32 vs fp64 logging accumulation. So the trainer's optimizer, parameter groups, weight decay, schedule, accumulation weights and clipping reproduce an independent implementation exactly.
- The first difference is a 1-ulp change in the step-5 grad norm (6e-8 rel). The likely cause is the order of the norm reduction: the trainer passes parameters in optimizer-group order, the reference in module order.
- **D3.** A change that only re-orders a floating-point sum makes the reference diverge from **itself** by more than it diverges from the trainer (3e-3 vs 3e-4 at steps 2–10). It also reaches the same 19 % final-parameter distance.
- The loop is deterministic run to run (D1 repeat), so these numbers are not noise.

**Classification:** FAIL of the frozen criteria, caused by tolerance calibration. The plan assumed that rounding differences stay near 1e-7 for 10 steps. In this regime (fresh init, η = 2e-3 with warm-up, AdamW, clipping active on 21–30 of 50 steps) they amplify to about 3e-3 within 10 steps and to O(1) within 20. No trainer defect was isolated. The semantic checks the test was designed for all came out clean: step-1 loss assembly and z-loss (5e-8), WD groups, schedule indexing, clipping placement, and the bitwise step 1–4 updates in D1. → PI request 4.

---

## VT-24 (R3e): oracle answerability of the 16K/32K items: **FAIL** (7 of 39,800 retrieval queries)

**Command:** `bash scripts/repro/r3_launch.sh vt24 0 tests/test_vt24_oracle_long.py`, which runs `pytest -q -s tests/test_vt24_oracle_long.py` → `scripts/repro/r3_vt24_oracle.py --workers 12`. Log: `experiments/repro/R3/logs/vt24_20261110T055803.log`. Raw: `experiments/repro/R3/vt24/run_20261110T055803/` holds `vt24_files.jsonl` (one row per file), `vt24_summary.json` (metadata + summary) and `scores/<ns>/<suite>/*.jsonl` (all scorer rows + `_meta.json`). CPU only, 05:58:03–05:59:17, one attempt.

**Scope.** 77 files: 52 retrieval files (R1, R-load including N = 2,048 at 32K, R3, R-dist; variants -C2, -T4, -natural, -repetitive included) and 25 M-hop negative-control files, at 16K and 32K in test/CS, test/MS, dev/DG and dev/FD. Every file was scored by `eval.scorer.score_file(InductionLM(16384, order=6, p_hit=0.9), …, score_candidates=False)`.

| Criterion | Result |
|---|---|
| Every retrieval query EM_tf = 1 **and** \|logp_gold − 4 ln 0.9\| ≤ 1e-4 | **39,793 / 39,800**. 7 queries fail, in 3 files: `test/MS/R1-T4_D3_32K` (5 of 400), `dev/DG/R-load_N1024_16K` (1 of 800), `dev/FD/R-load_N1024_16K` (1 of 400; this is the same item as the DG miss, see below). 49 of 52 files pass |
| M-hop EM_tf < 0.5 per file | PASS: EM_tf = 0.000 in all 25 files |
| Supplementary metadata oracle (report-only) | 39,800 / 39,800 queries have len(ctx) + len(prompt) + len(answer) = T, and the gold digits and the key tokens lie inside the recorded (latest) needle span |

**Diagnosis of every miss** (`python scripts/repro/r3_vt24_diagnose.py experiments/repro/R3/vt24/run_20261110T055803` → `diagnosis_misses.json`).

All 7 misses have the same mechanism. The first 2–3 answer digits are copied correctly. By the 3rd or 4th digit, the order-6 n-gram window of the oracle no longer contains the distinguishing part of the key. The window then matches a **later, different fact** in the same context, one that shares the template suffix and the value prefix, and the oracle copies that fact's next digit. Examples:
- T4: needle " gentle property's code: 5176.", copied from "understood matrix's code: 5179." The T4 template puts no key token in the 6-gram once 3 digits are emitted (" code", ":", " ", d1, d2, d3).
- T1, R-load N = 1,024: needle "The code for helpful preparation is 4230.", copied from "The code for comfortable preparation is 4235." Only the noun is still in the window, and the noun is shared.

These are **not** any of the item-defect types in the plan's failure tree:
- no filler collision: the colliding occurrence is another needle fact;
- no snapping damage;
- no misplaced boundary: the metadata oracle confirms the layout and that the answer is present for every query.

Each item is uniquely answerable from its full key. The misses come from the oracle's limited context, so the plan's premise that "any miss is an item defect" does not hold for these 7 queries. Raising the oracle order does not rescue them (order 8 fixes the R-load case; orders 10/12 fail because the window then reaches past the prompt into the preceding context). This is report-only and does not change the verdict.

**Observation.** `dev/FD/R-load_N1024_16K.npz` (100 contexts) consists of the first 100 contexts of `dev/DG/R-load_N1024_16K.npz` (200 contexts; contexts are identical), so the FD miss duplicates the DG miss. The file hashes differ (8ef5846a… vs 5f2f0840…), presumably through cell metadata. I have not checked whether the FD/DG overlap is intended.

**Classification:** FAIL of the frozen criterion. No item is malformed. The 7 misses are value-prefix collisions between facts under an order-6 copy oracle → PI request 3. If the PI wants such collisions to be impossible, the fix belongs in the generator (for example, no two facts in a context sharing the template suffix and the first 3 value digits), followed by regeneration under a decision row. I did not regenerate anything.

---

## VT-25 (R3a): Zoology ↔ our mixers by weight copy: **PASS**

**Command:** `pytest -q -s -p no:cacheprovider tests/test_vt25_zoology_transplant.py` (foreground, 05:51:09–05:55:27, one attempt). Raw: `experiments/repro/R3/vt25/vt25_transplant.jsonl` (+ `_meta.json`, which records Zoology commit `1ad20d19…`).

Zoology's 2-layer MQAR `LanguageModel` (BaseConv + mixer) was built from `original_mqar_configs.py`. `OurGdnZoology` / `OurFgZoology` were swapped into the same config, and every parameter was copied by an explicit name map:
- GDN: identity under `inner.`, with Zoology's unused `D` left out;
- MHA: `Wqkv` split into rows [q | k | v] → `q_proj` / `k_proj` / `v_proj`, `out_proj` → the `o_proj` alias.

MHA biases were zeroed, and both models ran in `eval()` mode. The batches were one MQAR batch (B = 8, T = 256, 64 kv pairs, seed 31337) and one random batch.

| Family | d | Batch | Mode | logits rel RMS | worst per-tensor grad rel RMS (tensor) | global grad rel RMS | Criterion | Result |
|---|---|---|---|---|---|---|---|---|
| GDN | 64 | MQAR | matched bf16 casts | 1.59e-4 | 5.57e-3 (layer-1 norm1.weight) | 3.5e-4 | ≤ 1e-2 / ≤ 2e-2 | PASS |
| GDN | 64 | random | matched | 1.60e-4 | 5.11e-3 (k_proj) | 3.5e-4 | | PASS |
| GDN | 128 | MQAR | matched | 3.61e-4 | 7.79e-3 (dt_bias) | 8.6e-4 | | PASS |
| GDN | 128 | random | matched | 3.60e-4 | 1.49e-2 (dt_bias) | 8.1e-4 | | PASS |
| GDN | 256 | MQAR | matched | 4.91e-4 | **1.82e-2 (A_log)** | 1.1e-3 | | PASS (thin margin) |
| GDN | 256 | random | matched | 4.85e-4 | 5.76e-3 (a_proj) | 1.1e-3 | | PASS |
| Attention | 64 | MQAR | fp32 | 9.9e-9 | 3.5e-7 (q_proj) | 6.2e-9 | ≤ 1e-5 / ≤ 1e-4 | PASS |
| Attention | 64 | random | fp32 | 8.4e-9 | 4.4e-7 | 8.3e-9 | | PASS |
| Attention | 128 | MQAR | fp32 | 3.3e-8 | 4.2e-7 | 2.5e-8 | | PASS |
| Attention | 128 | random | fp32 | 3.3e-8 | 5.9e-7 | 2.9e-8 | | PASS |

**Report-only: GDN, ours as built (fp32) vs Zoology.** Logits rel RMS is 2.1e-4 (d64), 4.5e-4 (d128) and 6.1e-4 (d256). The worst per-tensor gradient error reaches 6.0e-2 (d256 MQAR, dt_bias). Gradients of the decay-gate scalars (A_log, dt_bias) are the most precision-sensitive tensors. This matters for R4/R5 at production precision.

**The "test-only flag".** No such flag exists in `src/hybridlc`. It is implemented in the test (`zoology_bf16_casts`):
- it wraps the documented module-level indirection `hybridlc.models.linear.chunk_gated_delta_rule` to cast q, k, v, the b-logit and the gate input to bf16;
- it adds a forward hook that casts the o_norm output to fp32 before `o_proj`.

The σ(b) and gate formula stay in-kernel on our side, which is the difference named in the plan's tolerance basis.

**Sensitivity check** (report-only; `python scripts/repro/r3_vt25_sensitivity.py`, raw `experiments/repro/R3/vt25/vt25_sensitivity.jsonl`). I planted these defects into the weight copy:

| Planted defect | Family, width | logits rel RMS | Threshold | Detected? |
|---|---|---|---|---|
| q/k swapped | GDN d128 | 0.105 | 1e-2 | yes |
| conv taps reversed | GDN d128 | 0.107 | 1e-2 | yes |
| v-head order swapped | GDN d128 | 0.107 | 1e-2 | yes |
| q/k swapped | attention d64 | 4.1e-5 | 1e-5 | yes, but only 4× the threshold |
| v-head order swapped | attention d64 | 1.4e-3 | 1e-5 | yes |

The q/k-swap margin on attention is small because random-init attention is nearly uniform.

**Implication for R4.** G-RP(b) may now be evaluated, since amendment §17.4 requires that VT-25 pass first.

---

## Attempt log (plan §1 rule 8: every attempt is logged)

| Item | Attempt | When | Outcome | Counted? |
|---|---|---|---|---|
| VT-21 | smoke (5 hand-picked cases, `python -` in the terminal, nothing written) | 05:48 | harness check | no |
| VT-21 | 1 (`logs/vt21_20261110T055100.log`, raw `vt21_conv.jsonl`, 13 rows) | 05:51–05:54 | **infra OOM**: GPU physically full (VT-13 2 × 5.2 GB + R7 + my VT-25); Triton autotune buffer | no (infra; rerun per §2.0) |
| VT-21 | 2 (`vt21_20261110T055607.log`, `vt21_conv.r1.jsonl`, cases 0–142) | 05:56–06:46 | stopped by me: CPU reference too slow under load ≈ 43 | yes (its 143 rows are part of the verdict) |
| VT-21 | 2-resume (`vt21resume_20261110T064817.log`, `vt21_conv.r3.jsonl`, cases 143–383 + supplementary) | 06:48–06:58 | completed; verdict FAIL | yes |
| VT-21 | external `pytest tests/` run by the S0 verifier (PID 1683560), `vt21_conv.r2.jsonl` | from ≈ 06:48 | not mine | no |
| VT-22 | 1 (`vt22_20261110T060147.log`) | 06:05–06:07 | completed; verdict FAIL | yes |
| VT-22 | diagnosis (`r3_vt22_diagnose.py`) | ≈ 06:10 | report-only | n/a |
| VT-23 | 1 (`attempt_20261110T060457`) | 06:04 | **infra OOM** (trainer exit 75; my VT-22 started 2 s later, plus VT-13 and R7) | no |
| VT-23 | 2 (`attempt_20261110T060937`) | 06:09 | **infra OOM** (VT-13 resume process restarted inside the window) | no |
| VT-23 | 3 (`attempt_20261110T062337`) | 06:23 | OOM at the **0.44 cap**: 6.11 GiB allocated + 0.52 GiB reserved-unallocated | no |
| VT-23 | 4 (`attempt_20261110T062537`, `expandable_segments`) | 06:25 | OOM at the 0.44 cap: 6.58 GiB allocated at a 256 MiB request → the configuration is infeasible at the cap | no |
| VT-23 | 5 (`attempt_20261110T062646`, activation checkpointing on both sides = DEVIATION, recorded in the prereg addendum before running) | 06:26–06:28 | completed; verdict FAIL | yes |
| VT-23 | re-comparison of attempt 5 (`vt23_compare.r1.jsonl`) | 06:30 | harness fix: the WD-group summary reader expected `decay`/`no_decay` lists, but the trainer writes `n_decay_tensors` / `no_decay_names`. Same run, same data; only the group check changed (FAIL → PASS); all other numbers identical | yes (verdict source) |
| VT-23 | diagnostics D1/D2/D3 + D1 repeat | 06:30–06:35 | report-only | n/a |
| VT-24 | 1 (`run_20261110T055803`) | 05:58–05:59 | completed; verdict FAIL | yes |
| VT-24 | miss diagnosis (`r3_vt24_diagnose.py`) | ≈ 06:00 | report-only | n/a |
| VT-25 | 1 (foreground pytest) | 05:51–05:55 | completed; verdict PASS | yes |
| VT-25 | sensitivity (`r3_vt25_sensitivity.py`) | ≈ 06:00 | report-only | n/a |

No verdict-bearing run was repeated after its verdict was read. VT-21 attempt 2 was stopped mid-way and resumed, not repeated. Its 143 finished rows count, and the resume ran only the missing case ids, with the same seeds.

## Requests to the PI

Each request asks for a decision row. None of them can be settled by tuning.

1. **VT-21 dw bound.** In fp32 the bound "≤ 1e-4 max abs for dw" is exceeded at T = 32,768 purely by fp32 summation over 32K–65K terms (relative error 1–2e-7). Options:
   - (a) restate the dw bound relative to max|dw_ref| (for example ≤ 1e-6), or scale the absolute bound with √(B·T), and re-run VT-21. The VT-21 code needs no change: rel_max and rel_rms are already recorded for every case;
   - (b) accept the named residual: "fla's conv dw is correct to fp32 rounding; the absolute bound was mis-specified".
2. **VT-22 C0 bf16 bound.** Each implementation's own bf16-autocast error vs its fp32 path is 2.1e-2 at 16K, which already exceeds the 2e-2 cross-implementation bound. Options: set the bound from a measured bf16-vs-fp32 self-error (for example ≤ 1.5 × fla's own bf16 drift), or accept the residual. The position-band criterion, which is the actual target, passed with flat profiles.
3. **VT-24 oracle collisions.** 7 / 39,800 queries miss through value-prefix collisions between facts under an order-6 copy oracle. These are not filler collisions, snapping damage or boundary errors (all 39,800 needles verified by metadata). Decide between:
   - (a) accept: items are well-posed, and the oracle's limited context is the cause;
   - (b) require the generator to forbid two facts in one context sharing the template suffix and the first 3 value digits (only T4 and same-noun T1 cases are affected), then regenerate before G5 with a new manifest and a VT-12 re-run;
   - (c) change the oracle to read the full key (for example, n-gram anchored on the key span).

   Also confirm whether dev/FD being the first 100 contexts of dev/DG is intended.
4. **VT-23 trace tolerances.** In this regime, rounding-order differences amplify to about 3e-3 within 10 steps and to O(1) by step 20. A reference loop compared with itself under a pure summation-order change misses the 1e-5 / 1e-3 / 1e-3 bounds by more than the trainer does. Options:
   - (a) re-state VT-23 as "bitwise identical when summation orders are matched", which D1 met for steps 1–4. Together with the passing step-1 loss, WD-group and clipping checks, this answers the identity question;
   - (b) shorten the trace window and lower η so that the dynamics stay contractive;
   - (c) accept the residual.

   Also note the logged DEVIATION: activation checkpointing was needed because the fp32 2 × 4K configuration does not fit the 0.44 cap.
5. **G-RV and R4.** VT-25 passed, so G-RP(b) / R4 may proceed. G-RV stays open until requests 1–4 are decided. If a criterion is restated, the affected VT re-runs in full under the new decision row; the raw data here already contains the relative metrics.

## Defects in `src/hybridlc` found by R3

**None isolated.** No VT-21…25 result points to a functional defect in `src/hybridlc/**`. What R3 does establish:
- fla's conv matches the reference to fp32 rounding, with zero leakage and exact state handling;
- C12 matches Qwen3 and C0 matches the fla LM in fp32 to ≤ 4.5e-4 at 4K, with flat error up to 32K in bf16;
- packed and unpacked runs are bitwise equal at 32K;
- the trainer's update path is bitwise identical to an independent loop when summation orders match;
- the item files are well-formed;
- our Zoology-geometry mixers match Zoology's by weight copy.

Minor observations for the S0 fixer (not defects):
- (i) the trainer's `start` record summarises WD groups as `{n_decay_tensors, no_decay_names}` rather than listing decay names. INTERFACES.md does not fix the format; I adapted my reader to it;
- (ii) the trainer computes the clip norm over parameters in optimizer-group order, so its norm reduction order differs from module order (a 1-ulp effect; only relevant to bitwise cross-implementation comparisons);
- (iii) a full `pytest tests/` now also runs VT-21…VT-25. With the current files that takes ≈ 10–15 min of GPU (VT-21 ≈ 10 min, VT-22 ≈ 2 min, VT-23 ≈ 2 min including a 50-step training run, VT-24 ≈ 1.5 min of CPU with 12 workers) and writes raw outputs under `experiments/repro/R3/`.

## Files

- Pre-registration: `experiments/repro/R3/prereg.yaml`
- Tests: `tests/test_vt21_conv_reference.py`, `tests/test_vt22_long_transplant.py`, `tests/test_vt23_reference_loop.py`, `tests/test_vt24_oracle_long.py`, `tests/test_vt25_zoology_transplant.py`
- Scripts: `scripts/repro/r3_common.py` (metrics, raw writer, run metadata incl. kernel-log events), `r3_launch.sh` / `r3_wait_gpu.sh` (detached, memory-gated launchers), `r3_vt23_reference_loop.py`, `r3_vt24_oracle.py`, `r3_vt24_diagnose.py`, `r3_vt22_diagnose.py`, `r3_vt25_sensitivity.py`
- VT-23 trainer configs: `experiments/repro/R3/vt23/{run_vt23.yaml, training_vt23.yaml}`
- Raw outputs and logs: `experiments/repro/R3/{vt21,vt22,vt23,vt24,vt25,logs}/`

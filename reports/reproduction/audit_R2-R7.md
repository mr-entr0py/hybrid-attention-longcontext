# Audit of reproduction results R2–R7

**Role:** independent adversarial auditor. I did not run R2–R7. **Date:** 2026-11-10, 12:05–13:00 IST.
**Scope:** `results_R2.md`, `results_R3.md`, `results_R4.md`, `results_R5.md`, `results_R7.md`; their raw outputs under `experiments/repro/`; the pre-registrations (`reproduction_plan.md` v1.0, `protocol_amendment.md`, `experiments/repro/{R3,R4,R5}/prereg.yaml`).
**Constraints kept:** nothing under `src/` was edited, and nothing was committed. Every GPU job of mine held `scripts/repro/gpu_lock.py` at memory fraction ≤ 0.44. One planned GPU job was moved to the CPU so as not to run beside an S0 gate run (§3.5).
**Pre-registration integrity:** `reproduction_plan.md` SHA-256 `6bc14af6…cebc` and `protocol_amendment.md` `bf19a536…38e6` both equal the DEV-01 row (re-hashed by me).

**Audit artefacts.** Scripts are in `experiments/repro/audit_R2-R7/scripts/` and raw outputs in `experiments/repro/audit_R2-R7/outputs/`. Every number I recomputed from the executors' raw files comes from one command:

```
python -I experiments/repro/audit_R2-R7/scripts/audit_recompute.py  →  outputs/recompute.jsonl
```

It reads the raw files directly and imports none of the executors' analysis code. The independent re-runs have their own scripts and outputs, which are cited where they are used:
- `audit_vt21_case.py` → `vt21_rerun.jsonl`
- `audit_vt24.py` → `vt24_rerun/`
- `audit_reeval_zoology.py` → `reeval.log`
- `audit_lambada.py` → `lambada_cpu.log`

---

## 0. Confirmed verdicts

| Item | Executor's verdict | **Audit verdict** | Independent re-check by the auditor | Condition or caveat |
|---|---|---|---|---|
| **R2** environment integrity | not passed | **FAIL (confirmed)**. (c) PASS; (b) partial; (a) not live; (d) recorded; (e) not done | (c): longest kernel recomputed from the 20 raw chrome traces = **25.243 ms**. (b): re-ran `infra_events.py scan` over the plan's window and got the same 8 events; self-tests 6/6. Kernel log re-read. | The pass rule "(a) and (b) live **before R4 starts**" was not met (R4 started 07:33:15). (a) was still violated during this audit (§3.5). |
| **R3** VT-21…VT-25 | FAIL (VT-25 PASS) | **FAIL (confirmed): VT-21 FAIL, VT-22 FAIL, VT-23 FAIL, VT-24 FAIL, VT-25 PASS** | VT-21: re-ran 3 fp32 T = 32K cases with my own seeds and float64 reference; dw fails the same way. VT-24: re-scored the 3 failing files plus 2 controls on the current code; same 7 misses. All five tests recomputed from raw. | No failure isolates a code defect; each is a mis-calibrated tolerance or oracle. A criterion may **not** be restated after its item has run (§3.1). VT-25's margin depends on the batch (§1.5). |
| **R4** G-RP(b) + R4b | PASS | **PASS (confirmed)**: G-RP(b) 4/4, R4b 2/2 | Re-evaluated the saved final weights of `our_gdn-d128-lr1.0e-03` with Zoology's own `Trainer.test`. Overall 0.9888560 and 1024/256 0.9224297 match the run JSON exactly. | Smoke grade at ceiling: MDD 0.21 on the only live cell, as the plan acknowledges. |
| **R5** d64 MQAR | PASS | **PASS (confirmed)**; counts for G-RV only once the PI rules on deviation D1 | Re-evaluated 3 of the 6 saved final weights; all match exactly. Rules 1–4 recomputed. | R5 is a gate run, executed after the O4 trigger without O4 (plan §2.2). Low power: excludes only a bf16 handicap > 0.115 (50% power) / 0.154 (80% power). |
| **R7** LAMBADA | FAIL (Pythia), Wang PASS | **FAIL (confirmed)** on both Pythia criteria; **Wang GDN-pure PASS (confirmed)** | My own scorer (plain `transformers`, neither hybridlc nor lm-eval) gives Pythia-70M fp32 **1170/5153 = 0.2271**, the same count as our scorer and lm-eval, and Pythia-160M fp32 **1823/5153 = 0.3538**, again the same count. In fp16 it gives 70M **0.1855** (ppl 142.15), against EleutherAI's record of 0.1846 (ppl 142.43). | The scorer is not implicated. The fp16-anchor explanation is supported empirically. Its code-path citation (old lm-eval `hf-causal` loading `config.torch_dtype`) is **not verified** (§1.6). |

---

## 1. Per-item findings

### 1.1 R2: environment integrity (FAIL confirmed)

**Criteria (plan §2.2):** (a) and (b) live before R4 starts; (c) longest kernel < 1 s; (d) and (e) recorded.

- **(c) PASS, confirmed.**
  - I parsed every `experiments/repro/r2/profile/traces/*.json.gz` (`cat == "kernel"`). Maxima match the summary JSONs to the microsecond: D512-C12 `train_warm_stageB` 25.243 ms (flash bwd), D384-C12 18.767 ms, D384-C0 1.752 ms, D512-C0 1.529 ms; prefill maximum 10.704 ms.
  - The workload is as pre-registered: activation checkpointing on (`r2_kernel_profile.py` L136), and a real Stage-B micro-batch (`n_docs` 1, PG19).
  - All 4 infra tags are clean: `flagged` false, `journal_ok` true, 0 events, only CARLA-env 124395 and gnome-remote-desktop 293356 beside the profiler.
- **(b) partial, confirmed.**
  - My re-run of `infra_events.py scan --since "2026-11-09 17:30" --until "2026-11-09 18:30"` (`outputs/audit_scan_1730_1830.json`) is identical to the executor's: 3 host-OOM lines, Xid 31 pid 799092, two RC watchdog + Xid 8 (pids 811688, 818052).
  - Self-tests: 6 passed (`outputs/r2_tooling_selftest_audit.log`).
  - **Not integrated:** at 12:12 IST, `grep -rl "infra_events|journalctl|krcWatchdog|NVRM"` over `src/`, `scripts/` and `tests/` (outside `scripts/repro` and `scripts/s1_zoology`) finds nothing.
- **(a) not live, confirmed.**
  - The lock-wait logs show 2–4 unlocked project GPU processes between 05:50 and 07:18: VT-13 trainers 1603959/1603960, VT-21/25 pytest, `r7_run.py`, `r7_lmeval_diag.py`, S0 `pytest tests/`, `.verify` checks.
  - My scan of the R4 and R5 infra JSONs finds other project GPU processes in **3 of 9** R4 windows (the report says 2; §1.7) and **6 of 10** R5 windows.
  - I observed another violation myself (§3.5). The only use of the lock outside the reproduction workstream that I found is the S0 "G0-full-suite" run, from 12:35:40.
- **(d) recorded.** The pid-to-time reasoning is sound; I re-checked the pause log and systemd lines. pid 799092 remains unidentified; the criterion only requires a record.
- **(e) not done by R2.**
  - Contention in the orchestrator's benchmark is confirmed: the D384 rows of `train_step_idle_20261110.json` list pids 1602000/1602001 and contain 2 OOM rows. The `_D384` file is clean.
  - The G0-T re-pricing of D041, R6 and R9 is not documented anywhere I could find.
- **Missing artefact.** There is no `experiments/repro/r2/prereg.yaml` (plan §1 rule 6). The readings are in `results_R2.md` §0, and their 05:46 timestamp cannot be verified. With a 40× margin on (c), no verdict depends on them.

### 1.2 R3 / VT-21: fla `causal_conv1d` vs reference (FAIL confirmed)

- **Recomputed from raw (r1 ∪ r3):**
  - 384 rows, 384 unique case ids, no duplicates. The external S0 file `vt21_conv.r2.jsonl` was excluded.
  - **372/384 pass.** The 12 failures (cases 56, 112, 116, 176, 180, 188, 240, 248, 304, 368, 372, 376) are all fp32 dw at T = 32,768 with forward and dx passing. Worst dw max-abs is 1.403e-4.
  - All 192 packed cases have `leak_fwd_zero` and `leak_bwd_zero`.
  - bf16 worst relative RMS is 1.88e-3.
- **Independent re-run** (`scripts/audit_vt21_case.py`, my own seeds and direct float64 shifted-sum reference, exclusive GPU 12:14:34–12:14:55, `outputs/vt21_rerun.jsonl`):

  | D | dw max-abs | dw max-abs (relative) | forward / dx max-abs |
  |---|---|---|---|
  | 1536 | 1.09e-4 | 1.08e-7 | ≤ 4.1e-7 |
  | 384 | 1.06e-4 | 1.17e-7 | ≤ 4.1e-7 |
  | 256 | 1.05e-4 | 1.15e-7 | ≤ 4.1e-7 |

  |dw| ≈ 900–1008 in these cases, so all three fail the 1e-4 absolute bound exactly as the executor's cases did. The failure is generic, not seed-specific.
- **Why the bound is unattainable.** At |dw| ∈ [512, 1024) one fp32 ulp is 6.1e-5, and at [1024, 2048) it is 1.2e-4. An absolute 1e-4 bound on outputs of magnitude ~10³ is therefore 1–2 ulp, which no fp32 implementation can meet reliably. This confirms the executor's diagnosis: the criterion is mis-specified, not fla's kernel.
- **Erratum.** The report says "|dw| reaches 330–850". The raw range at T = 32K fp32 is **332–1191**. The verdict is unaffected.

### 1.3 R3 / VT-22: full-model transplant at 4K–32K (FAIL confirmed)

- **Raw values.**
  - C0 ↔ fla in bf16: relative RMS **0.020208 (16K)** and **0.020043 (32K)**, above 0.02; band ratios 1.058 / 1.026, so the error profile is flat.
  - C12 passes everywhere (≤ 6.88e-3).
  - Diagnosis rows confirmed: fla bf16 vs fla fp32 is 0.020854, ours fp32 vs fla fp32 is 2.61e-4. The bound sits at the intrinsic bf16 drift of one implementation against itself.
- **The packed-vs-alone check is not vacuous.** The test runs two different forwards: `[a|b]` with `cu=[0,L,2L]` and reset positions, and then each document alone. Bitwise equality is a genuine pass.
- **Observation, not a finding.** C0 vs fla in fp32 at 4K gives relative max 4.5e-4: inside the 1e-3 bound, but 100× C12's 4.3e-6. This is plausibly a TF32 or kernel-path difference. Worth one line to the S0 models owner.

### 1.4 R3 / VT-23: trainer vs independent loop (FAIL confirmed)

- **Configuration.** As pre-registered: D384 C3, fp32 audit mode, `StageStream('A')` seed 900, micro-batch 2 × 4,096 with accum 2, 50 steps, 10 warm-up steps, η = 2e-3 (`training_vt23.yaml` inherits `stage_A.yaml`).
- **My recomputation from the raw traces and checkpoints:**

  | Check | Value | Result |
  |---|---|---|
  | Step-1 loss | 4.88e-8 | PASS |
  | Steps 2–10, grad norm | 3.03e-4 | **FAIL** |
  | Steps 11–50, grad norm | 1.77 (step 19) | **FAIL** |
  | Steps 11–50, loss | 1.57e-2 (step 24) | **FAIL** |
  | Final parameters, relative L2 | 0.1945 (200 = 200 tensors) | **FAIL** |
  | Steps with pre-clip norm > 1 | 30 (trainer) / 21 (reference) | PASS |
  | WD groups | identical no-decay name sets (58); decay counts 142 = 142 over identical parameter names | PASS |

- **Isolation claims verified.**
  - D1 is bitwise equal to the trainer at steps 1–4; steps 5 and 8 differ by 1 ulp, and steps 6–7 are bitwise again.
  - The D1 repeat is bitwise identical over all 50 steps.
  - D3 (a summation-order change only) against the verdict reference gives 3.04e-3 at steps 2–10 and 0.79 at steps 11–50. The reference loop misses the bounds against itself, so the trace tolerances cannot be met in this regime.
- **The FAIL is not explained away.** The verdict stays FAIL, and the identity evidence goes to the PI as evidence, not as a re-reading.
- **Errata.** "steps 11–50 … loss max 5.9e-3" should be **1.57e-2 (step 24)**. "a step-20 spike: 1.88 vs 0.76" should read **the maximum is at step 19: 2.479 vs 0.895**; step 20's relative difference is 1.5. The verdict is unaffected.
- **Deviation.** Activation checkpointing was enabled on both sides. It was declared in the prereg addendum (file mtime 06:26:15) before the verdict attempt (06:26:46). The OOM history in the attempt logs shows it was needed under the 0.44 cap, and it needs PI acknowledgement.
- **Post-comparison reader fix (WD groups FAIL → PASS).** This fixed a format reader, not a criterion, and my independent check above confirms PASS.

### 1.5 R3 / VT-24 (FAIL confirmed) and VT-25 (PASS confirmed)

**VT-24.**
- **File set is complete:** 52 retrieval files (R1, R-load, R3, R-dist and their variants at 16K/32K in the four directories) plus 25 M-hop controls, 77 in total.
- **Totals recomputed:** 39,800 queries, **7 misses in 3 files**; M-hop maximum EM is 0.0.
- **Re-run on the current code** (`scripts/audit_vt24.py`, CPU, `outputs/vt24_rerun/`): the 3 failing files give 5/400, 1/800 and 1/400 misses at the same `(context, query)` ids; clean control `test/CS/R1_D3_32K` 0/1200; negative control `M-hop_h5_32K` EM 0.
- **Mechanism verified.** All 7 colliding occurrences lie **inside another fact's span** (checked against `arr.fact_start`/`arr.fact_end`), so they are fact-vs-fact value-prefix collisions. None is a filler collision, snapping damage or a misplaced boundary.
- **Misleading field name.** The diagnosis field `collision_inside_needles: false` tests only the queried key's own needle spans, so it is not evidence either way.
- **The executor's open question on dev/FD is answered.** FD being the first 100 contexts of DG is **by design**: `eval/suites.py` says items are prefix-stable, so "dev cells that appear in several dev suites with different n (e.g. DG 200 vs DC/FD 100) contain identical leading items". I checked that every array of `FD/R-load_N1024_16K.npz` equals the first-100-context slice of DG.

**VT-25.**
- **Raw values confirmed.** The verdict is taken on the worst tensor, which is the stricter reading. GDN matched-bf16: worst gradient 1.816e-2 against 2e-2, logits ≤ 4.91e-4. Attention: logits ≤ 3.33e-8, gradients ≤ 5.85e-7.
- **Planted defects are detected:** GDN ≈ 10× above the bound; attention q/k swap only 4×.
- **New finding: the margin depends on the batch.**
  - S0's unlocked full-suite pytest runs re-ran VT-25 twice: `vt25_transplant.r1.jsonl` (pid 1919872, 08:17) and `.r2.jsonl` (pid 1949025, 09:24). Both PASS, but their MQAR-batch rows differ from the verdict run, while the random-batch rows are bitwise identical.
  - Root cause: Zoology `data/multiquery_ar.py` L137 fills noise positions with **global `torch.randint`**, so `seed=31337` does not pin the batch. It depends on how much of the RNG the process consumed earlier.
  - Effect: the worst-tensor gradient error at GDN d64 MQAR moves from 5.6e-3 to 1.77e-2 across batches. The maximum over the three runs is 1.82e-2, a 9% margin. The PASS stands, because the verdict run is the pre-registered one.
  - Recommendation for the test owner: seed the global torch RNG immediately before building the MQAR batch, and report the minimum margin over several batches.

### 1.6 R7: LAMBADA (FAIL on both Pythia criteria confirmed; Wang PASS confirmed)

- **Recomputed from per-item rows:**
  - 160M: 1823/5153 = **0.3537745**, outside [0.313, 0.343].
  - 70M: 1170/5153 = **0.2270522**, outside [0.170, 0.200].
  - Wang GDN-pure (lambada_openai): 0.3434892, inside [0.328, 0.368].
  - Against lm-eval: **0 flips** for each model; max |Δ log p| 0.0027 / 0.0068.
  - fp16-weights diagnostic: 0.3297 / 0.1849.
  - All runs used the pinned snapshots (50f5173d, a39f36b1, 14661a98). lm-eval source is at d6de8164.
- **Independent third scorer** (`scripts/audit_lambada.py`; plain `transformers` on CPU, imports neither hybridlc nor lm-eval; lm-eval's split and `_encode_pair` definition; `outputs/lambada_cpu.log`):

  | Run | Correct / 5153 | acc | ppl |
  |---|---|---|---|
  | Pythia-70M fp32 | 1170 | 0.22705 | 130.96 |
  | Pythia-70M fp16 | 956 | 0.18552 | 142.15 |
  | Pythia-160M fp32 | 1823 | 0.35377 | 37.25 |

  Three implementations agree exactly in fp32, so the FAIL is real and the scorer is not implicated.
- **Anchor precision.**
  - My fp16 result, 0.1855 / 142.15, sits next to EleutherAI's record of 0.18455 / 142.43; fp32 gives 0.2271 / 130.96. The empirical case for "the published anchors are fp16-inference numbers" is strong.
  - **Not verified:** the code-level citation (lm-eval @`2b9a6e94` `models/huggingface.py::_get_dtype`). That commit is absent from the shallow local clone, and network access is blocked in this sandbox.
  - Caveat: in some 2023 lm-eval versions the `hf-causal` key was registered to `gpt2.HFLM` rather than `huggingface.AutoCausalLM`, so the cited file may not be the code path that produced the records. The empirical evidence does not depend on which path ran.
- **Process.**
  - There is no `experiments/repro/R7/prereg.yaml`, and the readings were written after the runs (disclosed).
  - No verdict is sensitive to this: the primary configuration is the plan's text; both Pythia criteria FAIL; Wang passes under both BOS settings (0.3435 / 0.3511).
  - **Erratum:** "batch_size = 1 instead of the frozen 8" is not a deviation from the pre-registration. The plan says "One sequence per forward"; 8 is a scorer default.
  - The Wang `D` drop is verified: in fla `d30c0833`, `self.D` appears only in its declaration (L129–130).
- **Gate status.** R7 is complete. Its FAIL does not implicate `eval.scorer`, so under the G-RV wording it does not block FEAS verdicts. Plan §4.1 has no consequence row for "Pythia fails, scorer not implicated", so the PI must log one.

### 1.7 R4 and R5 details

**R4.**
- **Timing.** The prereg (mtime 07:25:59) precedes the code change (`zoology_compat.py` 07:26:31) and the queue start (07:33:15). Its header says "~07:35", which is earlier than claimed and harmless.
- **Configuration.** Zoology reference SHA-256 values match the prereg. "Matched LR" is S1(a)'s best-over-LR per width. Attention d64 is a 3-way tie at 0.999982, broken to 1e-2, and both sides use the same LR. Seed and data seed are 123; there is no autocast in the G-RP(b) cells.
- **Results.**
  - Δ: −0.00009, −0.00005, −0.00039, −0.00025.
  - Trajectory: the maximum gap is 0.0032, at epoch 15.
  - R4b in-distribution mean is 1.0 in both precisions. `varlen_attn` was called 3,039 times in the bf16 run and 0 times in the fp32 run.
- **Infra.** All 9 infra JSONs checked: the only event is the exploratory config-A d128 attempt a1. Every gate run finished before the 07:56:54 O4 trigger.
- **Erratum (no verdict impact).** `results_R4.md` lists "none" for other GPU processes in the `our_gdn_configA-d256` a1 window and says unlocked S0 processes overlapped "two of the 9 windows". The samples in `R4-our_gdn_configA-d256-lr1.0e-03-a1.infra.json` show pids 1930958 (≤ 1.75 GB) and 1931383 (1.66 GB) on the GPU at 08:28:47–08:30:38, so the count is **3 of 9**. All three are non-gate windows.

**R5.**
- **Timing.** The prereg (mtime **08:37:50**) precedes the code change (`our_mixers.py` 08:38:06), the smoke run (08:41) and the queue (08:42). The SHA-256 is `c51afb3f…6246`.
- **Recomputed rules.**
  - Rule 1: 0.2963 lies in [0.05, 0.95].
  - Rule 2: 0.29631 < 0.92501 < 0.95656.
  - Rule 3: Δ −0.01754, SE_W 0.04151, threshold 0.11539 → PASS.
  - Rule 4: spans 0.0875 / 0.1096, below 0.2 → does not fire.
  - Configuration: seeds (123–125) as pre-registered; lr 1e-3; d64; 32 epochs; nothing resumed, no smoke runs counted; autocast only in our arm.
- **Event handling verified.**
  - The 4 Xid-8 attempts appear both in the kernel log and in the infra JSONs.
  - Every reported run starts inside an event-free window.
  - Full repeats reproduce the same trajectory: epochs 0–2 are identical across attempts (s124 ours 0.7617, 0.8069, 0.8265; s123 Zoology 0.6704, 0.7306, 0.7654). So a repeat cannot select an outcome.
  - Deviation D2 (continue on Xid, at most 3 full repeats) was declared at 08:46, before the first result at 08:56.

---

## 2. Is any PASS one that could not have failed, or any FAIL explained away?

- **PASS results.**
  - VT-25 can fail: planted defects give 10× (GDN) and 4× (attention) the bound. Its margin is thin and depends on the batch (§1.5).
  - G-RP(b) can fail only for gross defects (MDD 0.21 on the live cell); this is pre-acknowledged smoke.
  - R4b can fail only for gross attention defects (smoke).
  - R5 can fail for a bf16 handicap above about 0.115; it is low-power but genuine.
  - Wang R7 can fail: ±0.02, with BOS-off at 0.3511 and bf16 at 0.3421 also inside.
  - R2(c) has a 40× margin.
  - None is vacuous.
- **FAIL results.** None was converted into a pass. Every executor kept the frozen verdict and routed the diagnosis to the PI. I checked each diagnosis against raw data: VT-21, VT-22, VT-23 and VT-24 hold (§1.2–1.5), and R7 holds (§1.6) except for the unverified code-path citation.
- **Post-hoc risk in the pre-registrations.**
  - R3's prereg file was last modified at 06:26:15 (the VT-23 addendum), after the VT-25, VT-24 and VT-22 verdict runs. Its body cannot be shown to be unchanged since 05:45. But every reading that bears on a verdict is the literal or stricter one: absolute dw bound "not rescaled", worst-tensor gradients for VT-25. An edit could not have created a PASS.
  - R2 and R7 have no prereg file (§1.1, §1.6).
  - Recommendation: log a hash of each prereg when it is written.

---

## 3. Cross-cutting findings

1. **Criteria cannot be restated after their item has run.** The plan's status line reads: "An item's criteria may be changed before it runs only by a new decision row, and never after it has run." Option (a) in R3's PI requests 1, 2 and 4 ("restate the bound and re-run") therefore conflicts with the pre-registration. The compliant routes are:
   - (i) a PI decision row accepting a **named residual risk** for each of VT-21 to VT-24 (G-RV's failure clause);
   - and/or (ii) **new** pre-registered items with new IDs and calibrated tolerances (for example VT-21b, a dw bound relative to max|dw_ref|), reported beside the original FAILs and never replacing them.

   The same applies to the Pythia criterion; the R7 executor already said so ("a new pre-registration, not a re-reading").
2. **The O4 soak is overdue, and the infra risk is concrete.**
   - The kernel log (`outputs/journalctl_k_events_20261109_1700_to_audit.txt`) shows **7 RC-watchdog + Xid-8 events** (2 in R1, 1 in R4, 4 in R5), all on channel 0x00000003 and **all during GDN training in the Zoology harness**: Zoology's GDN 5 times, ours twice (fp32 config-A and bf16). Nothing has fired since 09:47:14, and none in any other workload over the same period (VT-13 production, G0 throughput, R2 profiles, R3, R7).
   - Three of today's five events had no other project process on the GPU.
   - The CARLA lane was not running for 6 of the 7 events: it was paused 18:03:59–18:25:35 on 11-09 (pause log), and STATUS records it stopped since about 06:10 today. Only its CARLA-env python client (pid 124395) stayed on the GPU. A CARLA restart can explain at most the 18:25:42 event, which came 7 s after the lane resumed.
   - The R5 rate was 4 events in 1.17 h of windows.
   - Plan §2.2 says to run O4 "before any further gate run". That was breached by R5 (declared as D1) and, apparently, by the S0 "G0-full-suite" run started at 12:35:40.
   - Suggested O4 content: a d64 Zoology-harness GDN soak (the workload that reproduces the hang about every 18 min), the failing job rerun with `CUDA_LAUNCH_BLOCKING=1`, and a production-shape GDN-A soak.
3. **The code changed after the runs.**
   - `src/hybridlc/models/build.py` changed at 12:15: `HybridLM` now redraws short-conv kernels N(0, 0.02). `attention.py` changed only in a docstring.
   - `linear.py`, `mixers.py` and `config.py` still hash to the R4/R5 values (466d4fef / aacd67c5 / 1b6af31c). R4, R5 and VT-25 build mixers through `build_mixer` under Zoology's init, so they are unaffected.
   - VT-23 and R2(c) ran on the previous `HybridLM` init. Their verdicts do not depend on it, but any re-registered VT-23 must run on the frozen G0 code.
   - VT-21 and VT-22 do not depend on the model init.
4. **Precision in scoring (R7 side finding, confirmed in the raw summaries).** Under the scorer's default bf16 autocast, Pythia drops to 0.12 / 0.05, while Wang GDN is unaffected. `rescore_check` uses the same autocast for its reference. I recommend an fp32 reference in `rescore_check` and in R6(a). This is not a G-RV blocker.
5. **The single queue was still not observed during this audit.**
   - An S0 `pytest -q -m "not slow"` (pid 1992411) held the GPU **without the lock** at about 12:23–12:33. My exclusive-wait job timed out after 600 s.
   - From 12:35:40 an S0 "G0-full-suite" run held the lock correctly, so I moved my last GPU check to the CPU (`outputs/r2a_violation_observed_during_audit.txt`).
   - This is better than earlier today, but not yet "live".
6. **The ledger `reports/reproduction/results.md` does not exist** (plan rule 9, amendment §17.7). No executor owned it.

---

## 4. What remains before gate G-RV

| G-RV condition (amendment §13.3) | Status at 12:50 IST | What closes it |
|---|---|---|
| G0 passed | **Open.** The S0 gate report is pending. `s0_integration.md` lists decisions owed (D512 C3 pass-1 throughput miss, VT-15 deviation, VT-13 production criterion). The G0 full suite has been running under the lock since 12:35 | S0 gate report and PI decisions |
| G-RP (a) and (b) passed | **Met.** (a) 16/16 (R1); (b) 4/4 (R4), evaluated after VT-25 | — |
| VT-21…VT-25 pass | **Not met.** VT-21, VT-22, VT-23 and VT-24 FAIL; VT-25 PASS | Four PI decision rows accepting named residual risks, and optionally new calibrated items under new IDs (§3.1). VT-24 also needs a choice between accepting, regenerating items under a decision row, or changing the oracle. The FD ⊂ DG question is settled (by design) |
| R5 not FAIL | **Met (PASS)**, pending a PI ruling on D1 (O4 not run before this gate run) | Decision row |
| R6 not FAIL | **Open, not run.** Needs the integrated trainer at the G0 freeze candidate and parentless TEF-run support, and O4 first | S0 integration, then O4, then R6 |
| R2 (a) and (b) live; R2(c) recorded | (c) **met** (25.24 ms). (a) **not live**: the lock is not mandatory, and an unlocked S0 pytest ran at about 12:2x. (b) **not live**: no `infra_events` in the S0 runner, trainer or registry | Orchestrator makes the lock mandatory and shares one lock file with the S0 runner queue; S0 adds tagging to `summary.json` and the registry, using `scripts/repro/infra_events.py` as the reference implementation |
| R7 complete before any FEAS verdict | **Complete** (FAIL, scorer not implicated; Wang PASS) | A PI decision row on the Pythia FAIL, since plan §4.1 has no row for this case |
| O4 (plan §2.2 trigger, pending since 07:56:54) | **Overdue.** 5 events today | Run O4 before R6 and any further gate run |
| Not a G-RV condition, still owed | R2(e) G0-T and the re-pricing of R6, R9 and D041 (use the `_D384` idle file); the ledger `results.md`; errata in `results_R3.md` (VT-23 loss maximum and spike step, VT-21 dw magnitude range) and `results_R4.md` (3 of 9 windows contended, not 2); VT-25 RNG pinning; the G-RV review entry by the PI | Orchestrator, PI, test owner |

**No `src/hybridlc` defect** was found by any item or by this audit. The C0 fp32 4K figure (§1.3) and the bf16 note for `rescore_check` (§3.4) are observations for S0, not defects.

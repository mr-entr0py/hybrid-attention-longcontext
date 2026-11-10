# PI rulings, 2026-11-10: D045–D057, the G-RV residuals, and the pre-FEAS items

**Status: DRAFT.** Author: principal investigator (PI rulings step of workflow `wf_b92a39ce-f17`). Date: 2026-11-10, written 13:25–14:00 IST.
**Effect:** none of these rulings is in force until the orchestrator accepts this file and enters the decision rows of §11. The new pre-registrations in `experiments/prereg/` are written now, **before any of their items runs**, and are hashed in §7.
**What I ran:** no GPU job. Read-only checks only: a kernel-log scan (`scripts/repro/infra_events.py scan`, saved as `experiments/repro/O4/pi_klog_scan_20261110_0940_1331.json`), `nvidia-smi` queries, and reads of raw files cited below. I edited nothing in `src/`, `tests/`, `scripts/`, `configs/` or `reports/decisions.md`, and committed nothing.

**Inputs read:** `experimental_protocol.md` (v1.0, §7.3, §9, §13, §16–§19), `reproduction/reproduction_plan.md` (v1.0, SHA-256 `6bc14af6…8adcebc`, re-hashed: equals DEV-01), `reproduction/protocol_amendment.md` (`bf19a536…0e338e6`, equals DEV-01), `results_R2/R3/R4/R5/R7.md`, `audit_R2-R7.md`, `gates/G0.md`, `gates/G-RP.md`, `s0_integration.md` (§1, §6, §12–§16), `decisions.md` (D000–D057), `competing_explanations.md` (O10), `STATUS.md`; raw files: `experiments/repro/R3/vt23/vt23_diagnosis.json` and the VT-23 trainer trace, `experiments/repro/R3/vt24/run_20261110T055803/diagnosis_misses.json`, `experiments/items/test/MANIFEST.json`, `tests/test_vt03_doc_isolation.py`, `tests/test_vt18_opchar.py`, `src/hybridlc/eval/{scorer,devtools}.py`, `src/hybridlc/training/config.py`, `src/hybridlc/utils/gpu.py`, `scripts/repro/gpu_lock.py`.

## 0. The rule applied to every ruling

A pre-registered criterion is never restated after its item has run. For every item that failed as registered I chose exactly one of: **(i)** accept a **named residual risk** (ID `RR-…`, register in §6) with evidence; **(ii)** fix a genuine defect and run a **NEW check** pre-registered in `experiments/prereg/` before it runs; **(iii)** block. Where I add a new check beside an accepted residual, the new check answers a different, sharper question and is itself blocking; it never converts the original FAIL. A pre-run clarification of ambiguous frozen text (an item that has not run) is allowed by decision row and is labelled as such.

Vocabulary: **ACCEPT** (as proposed), **ACCEPT-AMENDED** (accepted with stated changes), **REJECT**. "FAIL-as-registered" means the literal pre-registered criterion was not met.

---

## 1. Rulings at a glance

| ID | Subject | Ruling (short) |
|---|---|---|
| PR-01 | D045 G0 verdict | ACCEPT-AMENDED: G0 is **closed as PASS-WITH-ACCEPTED-RESIDUALS**, not relabelled PASS. Six FAIL-as-registered sub-items accepted as RR-G0-1…6; two sub-items deferred by design |
| PR-02 | D046 VT readings | ACCEPT-AMENDED: VT-05, VT-07, VT-15 are FAIL-as-registered → residuals RR-G0-1/2/4. VT-03 and VT-18 pass as registered (VT-03 fp32-GDN reading covered by RR-G0-5 either way). H3 GLMM must exist before G5 (PR-28) |
| PR-03 | D047 VT-13 production | ACCEPT-AMENDED: attempt 1 (0.0075) stays a FAIL → RR-G0-3; attempt 2 is corroboration, not a verdict; new criterion only for future audits (`VT-13p2.yaml`) |
| PR-04 | D048 G0 throughput | ACCEPT-AMENDED: "contended" is **not** defined after the fact. D384 accepted (≥ 95K in both passes). D512 pass-1 miss → RR-G0-6; D512 decided by the new exclusive-GPU check G0-T512x (`R2e_G0T_exclusive.yaml`, D031's own reopen clause). G-throughput reference = exclusive measurements |
| PR-05 | D049 G-RP | ACCEPT: PASS |
| PR-06 | D050 TEF `answer_norm: block` | ACCEPT (pre-run clarification; D014's text has no "per step" and its rationale is length balance) |
| PR-07 | D051 conv init N(0, 0.02) | ACCEPT option (a) |
| PR-08 | D052 evaluation disclosures | ACCEPT; (i)'s measurement pre-registered in `FEAS_precision.yaml` (Q1) |
| PR-09 | D053 H5(a) label | ACCEPT |
| PR-10 | D054 G-div | ACCEPT literal reading + mandatory near-miss logging so the reopen criterion is checkable |
| PR-11 | D055 fla PyPI wheel | ACCEPT as DEVIATION; record the wheel's hashes in provenance and at G5 |
| PR-12 | D056 Triton cache | ACCEPT-AMENDED: separate diagnostics cache **now**; **restore** the production cache from the S0 archive **before FEAS** (not only at G5); re-archive at G5; durable copy |
| PR-13 | D057 FEAS entry list | ACCEPT-AMENDED: replaced by the checklist in §9 |
| PR-14 | VT-21 | (i) RR-VT21. No new check needed |
| PR-15 | VT-22 | (i) RR-VT22 **plus** blocking NEW check VT-22b (fp32 at 16K/32K, calibrated bf16) |
| PR-16 | VT-23 | (i) RR-VT23 for the trajectory criteria **plus** blocking NEW check VT-23b (one-step equivalence from identical states). Activation-checkpointing deviation acknowledged |
| PR-17 | VT-24 | (ii) genuine generator gap: fix (rule G-ORACLE, minimal repair), **regenerate before FEAS**, blocking NEW check VT-24r |
| PR-18 | VT-25 | PASS stands; pin the test's RNG (hygiene, not a re-verdict) |
| PR-19 | R7 | (i) RR-R7; scorer not implicated (3 independent scorers agree exactly); does **not** block FEAS verdicts. No fp16 "re-registration" (its result is already known) |
| PR-20 | R2(a)/(b) | FAIL-as-registered for the past (RR-R2); **must be implemented in trainer, runner, evaluate.py and pytest**; blocking NEW check R2-LIVE |
| PR-21 | O4 soak | Defined concretely in `O4_soak.yaml`: 2 h production-shape soak (4 × 30 min: C0 Stage A, C0 Stage B, C3 TEF, C12 Stage B) + bitwise determinism probes before and after; PASS = zero events and bitwise probes. **R5 and the 12:35 G0 suite results stand** |
| PR-22 | R5 deviation D1 | ACCEPT (RR-R5-D1); R5 counts as "not FAIL" for G-RV |
| PR-23 | R6 | Runs after O4-A PASS, R2-LIVE L1/L2 and VT-24r; criteria unchanged; open readings fixed in `R6.yaml` |
| PR-24 | R2(c), R2(d), R2(e), R4 | R2(c) PASS (25.24 ms) recorded; R2(d) recorded; R2(e) owed now (`R2e_G0T_exclusive.yaml`); R4 PASS |
| PR-25 | DTS-10 FEAS decision code, U-15 | Required; semantics pre-registered in `DTS10_acceptance.yaml`; blinding rule; FEAS training may start before the code is complete, FEAS results may not be read |
| PR-26 | `rescore_check` precision | Keep the frozen bf16 single-pass reference (I-29 path check) **and add** an fp32 reference as a pre-registered precision audit (`FEAS_precision.yaml`, Q3) |
| PR-27 | Infrastructure-failure status (new) | Xid/RC-watchdog-tagged attempt failures never count toward D031's "3 consecutive failures" or §8.1 permanent failure; resume, pause for the PI after 10 |
| PR-28 | H3 GLMM sensitivity (new) | Implement in-repo (numpy/scipy, no new package) with a VT-18 extension **before G5** |
| PR-29 | Process (new) | Prereg hashing, ledger `results.md`, errata, strict-xfail markers for accepted residual tests, standing-replicate clarification |
| PR-30 | Cost-benchmark session (observed) | Not a gate run; may finish; yields to the G-RV path; results tied to their code hash; U-14 resolved by the user's exclusive-GPU grant |

---

## 2. The PROPOSED rows D045–D057

### PR-01 · D045, G0 verdict: ACCEPT-AMENDED

**Ruling.** G0 is closed as **PASS-WITH-ACCEPTED-RESIDUALS**. It is not relabelled a clean PASS, because six sub-items did not meet their literal pre-registered criterion:

| Sub-item | Literal criterion | Measured | Disposition |
|---|---|---|---|
| VT-05 bf16 final state | ≤ 8e-3 relative max | 3.0e-3…8.47e-3; **2 of 32** seed-cases above 8e-3 (fla bf16 chunk kernel; recurrent kernel within 1.1e-5) | RR-G0-1 |
| VT-07 branched vs separate forward | \|Δ logp_gold\| < 1e-3 (fp32) | **2.4–3.8e-3** vs single-pass (TF32 chunk vs IEEE fused_recurrent); 0.0 vs identical-kernel prefill; EM identical | RR-G0-2 |
| VT-13 production resume | \|Δloss\| < 0.005 at resume + 500 | attempt 1 **0.0075**; attempt 2 0.0023 with clean-vs-clean floor 0.0042 | RR-G0-3 (PR-03) |
| VT-15 Appendix A within 15% | all 11 cells | 8 asserted; on the idle GPU cuDNN B = 1 hot-L2 decode **−30%** (faster: 30.0 vs 43 µs); FLASH B = 1 hot −0%; eager 4K GDN +6% | RR-G0-4 |
| VT-03 GDN packed vs alone | max abs 0.0 | bf16 exactly 0.0 (asserted); fp32 9.4e-5 from GEMM-shape TF32, with same-shape bit-identity asserted | pass if the 0.0 bound is read in the training dtype (its basis, 05 §2.2, is bf16); otherwise RR-G0-5. Accepted either way |
| Throughput, D512 Stage A | ≥ 70K "contended" | pass 1 C3 **67.2K** at 35% foreign SM; pass 2 82.0K at ≈ 20% | RR-G0-6 (PR-04) |

Clean passes: VT-01, 02, 04, 06 (fp32), 08–12, 14, 16–20 (447/447 G0 tests in `runs/_logs/g0_full_suite.log`), VRAM 6.00 GiB ≤ 7 GiB, transplant 7.0e-7 / 4.6e-4 ≤ 1e-3, Triton archive present (`f417e487…96e3de`).

**Why residual acceptance, not "fix and re-test":** no defect is isolated in any sub-item (G0.md §2; `s0_integration.md` §12). VT-05, VT-07 and VT-15 are deterministic numeric properties of fla's kernels or of L2 residency; re-testing gives the same numbers, and patching fla to IEEE fp32 would invalidate the other S0 numerics (D046's rejected alternative). The protocol's G0 failure clause ("fix and re-test") has nothing to fix; this decision row is a logged deviation from that clause.

**Deferred by design (not failures):**
- VT-06 bf16 trained-checkpoint EM flips → evaluated at FEAS as `FEAS_precision.yaml` Q2.
- VT-13 standing replicate → clarified in PR-29(e): two audit-mode runs of C12 seed 0's first 1,000 steps, one at S4 start and one after the last S4/S5 training run, bitwise against each other. Never compared with a production run.

**Condition.** G0 stays closed only if the pre-FEAS full-suite re-run (§10 step 16) reproduces: every VT-01…VT-20 test passes on the final pre-FEAS tree (with the regenerated items). Any VT-01…VT-20 failure reopens G0.

### PR-02 · D046, VT threshold readings: ACCEPT-AMENDED

- (1) VT-05 median-of-8 reading: not adopted as the verdict; the literal per-case failure is RR-G0-1. The test may keep asserting the median and per-seed ceiling 1e-2 as a regression guard, labelled as such in its docstring.
- (2) VT-07: RR-G0-2. The operative safeguard is the §9.2 re-score (FEAS-PREC Q2), now backed by the fp32 reference (PR-26).
- (3) VT-15: RR-G0-4. The D046 text must also say that the cuDNN B = 1 hot cell is faster than Appendix A on an idle GPU, so its deviation is not only contention (G0.md §5). Cold-L2 cells, the primary §10.2 condition, pass (+2% / −1%).
- (4) VT-03: PASS as registered (attention fp32 ≤ 1e-5 is asserted, `tests/test_vt03_doc_isolation.py` L97). The RoPE negative-control limitation is a test-design note; position resets are verified on the data side (VT-11). fp32-GDN exactness: RR-G0-5 under the fp32 reading.
- (5) VT-18: PASS as registered. The strict ±0.01 applies to every debate-§6.2 cell (`tests/test_vt18_opchar.py::test_pisim_debate_62_within_001`); the extra 329 cells are beyond the registration. The missing H3 GLMM is an analysis-plan gap, not a VT-18 failure: PR-28.

### PR-03 · D047, VT-13 production resume: ACCEPT-AMENDED

Attempt 1 FAILED as registered (0.0075 > 0.005; `runs/_logs/vt13_production.log`). Attempt 2 (`experiments/s0/vt13_production/result.json`: |a−b| 0.0023, |c−b| 0.0019, |a−c| 0.0042) is reported as corroboration: it shows the bound is about one unit of production noise, so the test as registered could not separate resume error from non-determinism. Accepting attempt 2 as the verdict would be selection on outcome (D047's own rejected alternative). **RR-G0-3** is accepted because resume correctness is established bitwise in audit mode for k ∈ {0, 3, 12}, including kill -9 mid-accumulation and inside a checkpoint write (`s0_integration.md` §6). The proposed max(0.005, 2·|a−c|) rule is pre-registered **only for future audits**, with three seeds: `experiments/prereg/VT-13p2.yaml`. None is scheduled; one becomes mandatory if checkpoint, resume, loader or RNG-state code changes.

### PR-04 · D048, G0 throughput: ACCEPT-AMENDED

- **Not adopted:** defining "contended" as the pass-2 condition. Choosing the definition that passes, after both passes were seen, is a restatement.
- **D384:** accepted. Every D384 cell passed in both passes (Stage A 97.7–111.5K ≥ 95K; C12 Stage B 45.8K ≥ 27K; `experiments/s0/g0*/g0_results.jsonl`). D384 is the FEAS-first path.
- **D512:** the pass-1 miss is RR-G0-6. The GPU is now exclusive to the project (user decision, STATUS 06:10), which fires D031's own reopen clause: "The GPU becomes exclusive → re-measure throughput at G0 and re-budget." The D512 question is therefore decided by the new criterion **G0-T512x** (exclusive GPU, production trainer, ≥ 70K for every k; FAIL → the pre-registered 0.6B Stage-A fallback for every D512 run), pre-registered in `experiments/prereg/R2e_G0T_exclusive.yaml`. It runs right after O4-A, not only if FEAS-384 fails, because it costs minutes.
- **G-throughput reference** = the R2e-X exclusive medians, not the contended pass-2 values (a reference 30% below a contended number would hide real degradation on an exclusive GPU). Changed before the guard has ever run.
- **D041 re-pricing** from R2e-X is owed (decision row), as DEV-01 requires.
- R2e-X also records D029's pre-declared reading (audit-mode overhead ≤ 3% → deterministic production). Expected not to fire (deterministic FA2 backward costs +43–66% [MEAS-B]), but the rule is the protocol's.

### PR-05 · D049, G-RP: ACCEPT

PASS on re-run numbers only, recomputed by the gate reporter from raw JSONs and confirmed by the audit's independent re-evaluation (`our_gdn-d128`: 0.9888560 / 0.9224297, exact). Known limits recorded, not grounds to withhold: (a) is at ceiling (±0.03 overall = ±0.21 on the live cell); R5 supplies the below-ceiling calibration.

### PR-06 · D050, TEF loss normalisation: ACCEPT

`answer_norm: block` is the implementation of D014. D014's decision text reads "½·mean CE over synthetic answer tokens + ½·mean CE over all other tokens", with no "per step", and its stated rationale is "Length balance removes the ≈ 20× short-item skew". The per-step reading gives the 32K bucket 0.29–0.30 of answer weight with a 27.5× per-token spread (`s0_integration.md` §12 item 14), which would break D014's purpose and the §14 H5(a) control "length-balanced TEF". This is a pre-run clarification: no TEF run of the study has happened. `answer_norm: step` remains available only through a new decision row.

### PR-07 · D051, conv-kernel init: ACCEPT (option (a), N(0, 0.02))

"fla's own initialisation" (§7.1, D008) is the model-level `_init_weights` that every fla model, `GatedDeltaNetForCausalLM` (the VT-09 target) and Wang's checkpoints use. The constructor's U(−0.5, 0.5) is PyTorch's `nn.Conv1d` default, not an fla choice. F_gc's conv follows, so δ_gc compares conv-vs-no-conv attention under one conv init. No study run is affected. R6, VT-23b and O4 run on this init, which tests it in the production trainer.

### PR-08 · D052, evaluation disclosures: ACCEPT

(i) bf16-rounded evaluation weights: DEVIATION accepted. The owed measurement is now pre-registered with D052's own threshold (≥ 1% EM flips for any k → fp32 weights before S4): `FEAS_precision.yaml` Q1. (ii) LM-1 = 279 items: accept; only 78 of the 100 PG19-test books reach 32,768 tokens. (iii) M-hop depths not decile-stratified: accept; items are identical across k, so depth composition is a fixed-item effect that cancels in paired k-contrasts. (iv) absent candidates in filler: accept; the analysis must use `absent_in_context` when labelling "absent" errors and computing rank accuracy. (v) 907 items per bucket: accept. (vi) repetitive filler in 10% of filler-bearing items: accept. (vii) short-T A-freq wording: accept.

### PR-09 · D053, H5(a) label: ACCEPT

Under a true H5(a) the k = 12 vs k = 0 gap is small at short T, so G-REF's discriminability part fails there by construction. Applying the full rule at every length would label every H5(a) result "conditional" and drop the short lengths from the passing-length slope regardless of reference competence. Pre-run analysis decision.

### PR-10 · D054, G-div reading: ACCEPT with a monitoring requirement

Keep the literal frozen text (EMA₀.₉₉ keeps updating). Freezing the EMA would add automatic restarts, interventions that could correlate with k. **Required now:** `training/gdiv.py` also computes, log-only, the frozen-EMA excursion ("loss > pre-excursion EMA + 1.0 for > 50 steps") and writes a `gdiv_near_miss` registry note whenever it holds without the literal trigger. D054's reopen criterion ("an S2 run shows a sustained 1.0–1.65-nat excursion that goes untriggered") is then checked from the registry at the FEAS verdict.

### PR-11 · D055, fla wheel: ACCEPT as DEVIATION

The differences from clone `98193113` on the GDN path are dispatch decorators, a Hopper-only branch and int64 offsets in `causal_conv1d`, which are irrelevant at offsets ≤ 38M. The wheel is what every VT, R3, R4, R5 and R7 result ran on. In particular VT-21 shows its conv correct to fp32 rounding with zero leakage, and R7 shows its GDN reproducing Wang's 340M LAMBADA (0.3435 vs 0.348). **Required:** record the wheel's `RECORD` file hash and `dist-info` SHA in `env_info()["fla_install"]` now, and in the G5 freeze row.

### PR-12 · D056, Triton cache: ACCEPT-AMENDED

1. **Now:** every reproduction, diagnostic, verification and test job sets `TRITON_CACHE_DIR=$REPO/.cache/triton-diag`, created by extracting the S0 archive `runs/_artifacts/triton_cache.tar` (`f417e487…96e3de`). pytest gets this from the new `tests/conftest.py` (R2-LIVE L1). The cost-benchmark driver already uses `.cache/triton_bench`. The trainer refuses an exp_stage S2…S8 run whose `TRITON_CACHE_DIR` is not the production dir.
2. **Before FEAS, not only at G5:** move the live `.cache/triton` aside **without deleting it**, to `.cache/triton-pre-restore-20261110/` (it holds the 3,205 files that reproduction jobs added at 06h and 08h). Restore the production dir from the S0 archive and verify the trainer-style cache hash (`be0c80b1…f74316`). On the idle GPU, run `scripts/warm_triton.py` for any shape added since S0, re-archive, and record the hash as the registry note `pre-FEAS-triton-cache`. FEAS runs must start with that hash. This is not "clearing mid-study" (§18): the study begins at S2, and the restore removes autotune entries made under contention (E10.2).
3. **At G5:** re-warm, re-archive and freeze that SHA (D056 as proposed).
4. **Durability:** copy every archived tarball outside the repository working tree (e.g. `~/outeropt_artifacts/`). Record the SHA-256 in the registry and in the G5 row.

### PR-13 · D057, FEAS entry list: ACCEPT-AMENDED

Replaced by the FEAS entry checklist in §9. It adds O4-A, R2-LIVE, VT-22b, VT-23b, VT-24r, R6, R2e-X, the cache restore, the infra-status rule, the blinding rule and the R8/R10 helpers.

---

## 3. G-RV items

### PR-14 · VT-21 (fla `causal_conv1d` vs reference): residual RR-VT21

- **FAIL-as-registered:** 12 of 384 cases, all fp32 dw at T = 32,768. Max abs 1.03–1.40e-4 against the absolute 1e-4 bound; relative 1.1–2.3e-7, i.e. 1–2 ulp. Raw: `experiments/repro/R3/vt21/vt21_conv.r1.jsonl` + `.r3.jsonl`.
- **Why the bound cannot be met:** |dw| reaches 332–1,191 (audit erratum). One fp32 ulp is 6.1e-5 on [512, 1024) and 1.2e-4 on [1024, 2048), so an absolute 1e-4 on such outputs is 1–2 ulp, which no fp32 implementation meets reliably (audit §1.2).
- **Independent confirmation:** the auditor re-ran with their own seeds and a float64 shifted-sum reference: dw 1.05–1.09e-4, relative ≈ 1.1e-7 (`experiments/repro/audit_R2-R7/outputs/vt21_rerun.jsonl`).
- **Every other VT-21 criterion passes in all 384 cases:** forward ≤ 7.2e-7, dx ≤ 5.5e-7, bf16 ≤ 1.9e-3, cross-document leakage exactly 0, final state exact.
- **No new check.** The relative error is already recorded for every case. A post-hoc relative criterion on the same data would be a restatement, and a re-run would only repeat known numbers. Standing lesson for every future tolerance: absolute bounds on reductions must scale with the reduction's magnitude.

### PR-15 · VT-22 (full-model transplant 4K–32K): residual RR-VT22 + blocking VT-22b

- **FAIL-as-registered:** C0 ↔ fla in bf16, relative RMS 2.021e-2 (16K) and 2.004e-2 (32K) against 2e-2. Band ratios 1.06 / 1.03 (flat). C12 passes at 6.4e-3 / 6.9e-3; 4K fp32 passes (C12 4.3e-6, C0 4.5e-4); C3 packed vs alone is bitwise equal (`experiments/repro/R3/vt22/vt22_transplant.jsonl`).
- **Diagnosis** (`vt22_diagnosis.jsonl`): fla bf16 vs fla fp32 is 2.09e-2, our bf16 vs fla bf16 2.02e-2, our fp32 vs fla fp32 2.6e-4. The registered bound sits at the reference's own bf16 drift. **RR-VT22 accepted.**
- **Gap that remains:** VT-22 asserted fp32 equivalence only at 4K. Its bf16 band rule cannot see a position-growing error smaller than the 2e-2 noise. **VT-22b** (`experiments/prereg/VT-22b.yaml`) is blocking for G-RV:
  - fp32 at 16K/32K: relative RMS ≤ 1e-3 and band ratio ≤ 2;
  - bf16 cross-implementation error ≤ 1.5 × the larger self-drift, with band ratio ≤ 2;
  - new weights (seed 1).

  A FAIL is a candidate E10.4 defect, and G-RV stays blocked.

### PR-16 · VT-23 (trainer vs independent loop): residual RR-VT23 + blocking VT-23b

- **FAIL-as-registered:**
  - step-1 loss 4.9e-8 (pass);
  - steps 2–10 grad-norm 3.0e-4 (> 1e-5);
  - steps 11–50 1.77 (> 1e-3; maximum at step 19: 2.479 vs 0.895, audit erratum);
  - final parameters 0.1945 (> 1e-3).

  Raw: `experiments/repro/R3/vt23/vt23_compare.r1.jsonl`.
- **Evidence that the update rule is identical:**
  - With the production loss and clip swapped in (D1), updates are **bitwise identical at steps 1–4**. The trainer trace shows pre-clip norms 2.79, 2.76, 2.29 and 1.80 at those steps, so clipping was active in every bitwise step (`vt23_diagnosis.json`; trainer `trace.jsonl` `gnorm_hex`, decoded by the PI).
  - A summation-order-only change makes the reference diverge from **itself** by 3.0e-3 / 0.79 / 0.188 (D3), more than it diverges from the trainer.
  - The D1 repeat is bitwise over 50 steps.
  - WD groups are identical (142 / 58); clipping was exercised (30 / 21 steps).
- **Ruling:** the trajectory criteria are unattainable in this regime (rounding grows ×10 every 2–3 steps). **RR-VT23 accepted.**
- **The question VT-23 was meant to answer is still owed.** D1 is bitwise only to step 4, and only with production loss and clip, so the warm-up boundary, the constant-LR phase and the independent loss and clip implementations at later steps remain open. **VT-23b** (`experiments/prereg/VT-23b.yaml`) is blocking:
  - it compares one step from identical trainer states at s ∈ {0, 1, 4, 8, 9, 10, 20, 35, 48};
  - loss and grad norm within ≤ 1e-5;
  - per-tensor clipped gradient (recovered from the AdamW moments) within ≤ 1e-4;
  - per-tensor update and v within ≤ 1e-4.
- **Deviation acknowledged:** activation checkpointing on both sides, recorded in the prereg addendum before the verdict attempt (06:26:15 < 06:26:46) and forced by the 0.44 cap (6.58 GiB at a 256 MiB request). VT-23b uses it too, for the same reason; recompute replays identical kernels.

### PR-17 · VT-24 (oracle answerability): genuine generator gap → fix, regenerate before FEAS, VT-24r

- **FAIL-as-registered:** 39,793 / 39,800 retrieval queries. The 7 misses are 5 in `test/MS/R1-T4_D3_32K`, 1 in `dev/DG/R-load_N1024_16K` and the same item again in `dev/FD` (FD is the leading-100 slice of DG by design; audit §1.5). There are **0 in test/CS**.
- **Mechanism, verified on raw data** (`diagnosis_misses.json`): a more recent fact in the same context matches the order-6 window before an answer digit. Under T4 the window holds no key token from digit 2 on, so a shared 2-digit prefix fails digit 3 (4 cases, e.g. "historical bridge's code: 3938" copied from "average mind's code: 3908") and a shared 3-digit prefix fails digit 4 (1 case). Under T1, digit 4's window keeps only the key noun ("helpful preparation … 4230" copied from "comfortable preparation … 4235").
- **Ruling: (ii).** The plan pre-registered "any miss is an item defect" with the consequence "regenerate the item files before G5 (new manifest, VT-12 rerun) and log a decision row" (plan §2.3 R3e; amendment part F). Arguing afterwards that these misses are not defects would be the restatement the integrity rule forbids.
- **The gap is genuine.** Values are drawn uniformly with no check that each queried value is determined within the copy window. Under T4 this adds value-side interference that T1 blocks by keeping the key inside the window, so the T4-vs-T1 template-transfer contrast confounds template with key-to-digit distance.
- **Cost:** about 2.5 minutes of CPU. No model has been evaluated on these items.
- **What "collision-free" can mean.** Full value-prefix uniqueness is impossible at R-load N = 1,024 / 2,048: there are only 1,000 three-digit prefixes (pigeonhole). The enforceable rule is therefore **G-ORACLE**: for every queried fact and every answer digit, the most recent earlier occurrence of the 6-token window before that digit lies in the queried fact.
- **Repair rule:** resample only the queried fact's 4 value digits, from a dedicated repair substream keyed by (rng_key, context, fact, attempt). Filler, keys, layout and every other fact stay unchanged; cells with no repair stay content-identical; prefix stability across DG / DC / FD is kept. TEF, M-hop, A-freq, A-count and Tier 0 are untouched.
- **Timing:** **before FEAS**, which is stricter than "before G5". FEAS reads FD and DC, and G-FULL and G-REF read DG later, so one item set must serve the whole study.
- **New check VT-24r** (`experiments/prereg/VT-24r.yaml`):
  - K1: VT-24's criteria, unchanged, on every retrieval file at **every** length and in every dev directory (wider than VT-24);
  - K2: the M-hop negative control;
  - K3: content stability;
  - K4: the full VT-12 suite;
  - K5: TEF batch hashes unchanged;
  - K6: unit test of G-ORACLE and the repair.

  Blocking for G-RV, R6 and FEAS. The old item set's FAIL stays on record (manifests at `97ef269`).

### PR-18 · VT-25: PASS stands

The verdict run is the pre-registered one (worst gradient 1.82e-2 against 2e-2). The audit found that the MQAR batch depends on global `torch.randint` (Zoology `multiquery_ar.py` L137). Test hygiene: seed the global RNG immediately before building the batch, and report, without gating, the minimum margin over 10 seeds. This is not a re-verdict. A future failure of this hygiene sweep goes to the PI as evidence, not as an automatic G-RV change.

### PR-19 · R7 (LAMBADA): residual RR-R7, not blocking

- **FAIL-as-registered:** Pythia-160M 0.3538 (1823/5153) outside [0.313, 0.343]; Pythia-70M 0.2271 (1170/5153) outside [0.170, 0.200].
- **The scorer is not implicated.** Our scorer, lm-eval @`d6de8164` (0 EM flips, max |Δ log p| 0.0068) and the auditor's plain-`transformers` scorer give **identical counts** in fp32 (`experiments/repro/R7/compare/`; `audit_R2-R7/outputs/lambada_cpu.log`).
- **Cause:** the anchors are fp16-inference numbers. fp16 weights give 0.3297 / 0.1849 against EleutherAI's records of 0.3284 / 0.1846. The code-path citation is unverified, but the numbers stand on their own.
- R7's three purposes are met: EM_tf = lm-eval acc; the context/answer split matches; and Wang's GDN-pure passes at 0.3435, so no fla semantic drift is visible.
- **Ruling:** RR-R7 accepted. Under the G-RV wording only a scorer-implicating failure blocks FEAS verdicts, so R7 does not. This row supplies the missing plan §4.1 consequence for "Pythia fails, scorer not implicated": no block; raise at the reproduction review.
- **No "R7b at fp16".** Its outcome is already known from the diagnostic runs; pre-registering it now would relabel a FAIL.
- **Consequence carried forward:** Pythia collapses under bf16 autocast (0.12 / 0.05) while Wang's GDN does not (0.3421). That is the reason for the fp32-reference precision audit of our own checkpoints (PR-26, FEAS-PREC Q3).

### PR-20 · R2(a) single GPU queue and R2(b) event tagging: implement in runner, trainer, evaluate.py and pytest; R2-LIVE

- **FAIL-as-registered:** "(a) and (b) live before R4 starts" was not met (R4 started 07:33:15). Unlocked project processes appeared in 3 of 9 R4 windows and 6 of 10 R5 windows, and an unlocked S0 pytest ran at 12:23–12:33 during the audit (audit §1.1, §3.5).
- **RR-R2 accepted for runs already made.** Those runs are accuracy runs with Zoology's fixed data order; contention changes timing, not the computation. Every R4/R5 gate result comes from an event-free window, and the R4/R5 saved weights re-evaluate exactly (audit §0).
- **Required implementation** (`experiments/prereg/R2_live.yaml` L1–L3):
  - one lock file;
  - the lock or a holder-descendant token is checked before the first CUDA call in `scripts/train.py`, `scripts/runner.py`, `scripts/evaluate.py`, the throughput, bench and warm scripts and every repro / s1_zoology driver; a refused start exits with code 78;
  - pytest takes the lock through a new `tests/conftest.py`;
  - an infra block goes into `summary.json`, every terminal registry row, a runner `attempt_end` row (crashed attempts included) and evaluation `_meta.json`;
  - gpumon records lock-holder ancestry every 5 s.
- **New check R2-LIVE (blocking):**
  - C1 tests: serialisation, refusal, pytest locking, tagging with an injected Xid line, tagging of killed attempts;
  - C2: zero unlocked project CUDA processes and an infra block on every run, over the windows from O4-A to R6 plus VT-22b, VT-23b and the pre-FEAS G0 suite.

### PR-21 · O4 soak: defined; R5 and the G0 suite stand

**Definition** (`experiments/prereg/O4_soak.yaml`):

- **O4-A (verdict).**
  - A1: a bitwise determinism probe (D384 C3, audit mode, 50 steps, two processes).
  - A2: a production soak through the production trainer under the lock, with tagging live. It runs **2.0 h** in four 30-min segments, each a fresh production-mode run: C0 Stage A (2 × 4K), C0 Stage B shapes (1 × 32K, checkpointing), C3 TEF shapes, and C12 Stage B shapes. All at D384, seeds 306–309, with the diagnostics cache.
  - A3: the probe repeated after the soak.
  - **PASS** = zero Xid, RC-watchdog or OOM lines with a readable journal; no segment ended by a CUDA error; A1 and A3 bitwise, both within themselves and with each other.
  - Basis: at the Zoology-harness rate (4 events in 1.17 h in R5), P(0 in 2 h) = 1.1e-3, and zero events bounds the production-shape rate at ≤ 1.5/h (95%).
- **FAIL** → no gate or production run until the PI chooses among a configuration-only change, headless operation (needs the user's consent) or acceptance under PR-27.
- **O4-B (diagnostic):** the R5 cell that hung twice (Zoology GDN d64 s125), alone, with `CUDA_LAUNCH_BLOCKING=1` and faulthandler stack dumps, for at most 60 min. It runs after the G-RV gate runs, or at once if O4-A fails.

**Do R5 and the 12:35 G0 suite stand? Yes, both.**

- R5's six reported runs all come from event-free windows. Every flagged attempt was discarded and repeated in full from epoch 0. Repeats are identical at the start (epochs 0–2), so a repeat could not select an outcome. The hang hit Zoology's own GDN 3 times and ours once. The audit's re-evaluation of the saved weights matches exactly (`results_R5.md`; audit §1.7).
- The G0 suite's window was scanned clean (`infra_events.py scan` 12:35–13:30: `events: []`). Its tests are deterministic, and its numbers reproduce earlier runs exactly: VT-21…24 match the R3 verdicts, and VT-20 matches the integrator to 0.01 GiB.
- The sequencing breach ("O4 before any further gate run") is recorded as RR-R5-D1 and RR-G0suite-seq.
- **Reopen trigger:** O4 finds evidence of silent corruption (a non-bitwise determinism probe, or any Xid other than 8 / RC watchdog in a run that did not crash). Then R5, the 12:35 suite and every run since 07:56:54 are re-examined.
- Supporting fact: there has been no event since 09:47:14 (PI scan to 13:31:52, `experiments/repro/O4/pi_klog_scan_20261110_0940_1331.json`). The cost-benchmark session running since 13:41 is extra inference-only exposure (report-only); it does not replace O4-A, which also exercises backward kernels.

### PR-22 · R5 deviation D1: ACCEPT

D1 (R5 ran after the O4 trigger without O4) was declared in the prereg (mtime 08:37:50) before any R5 run. O4 had not been assigned to the R5 executor, and the orchestrator dispatched R5. On the evidence in PR-21, R5 = **PASS** counts as "R5 not FAIL" for G-RV:
- Δ = −0.0175, threshold 0.1154;
- phenomenon 0.296 < 0.925 < 0.957;
- spans 0.0875 / 0.1096, so the bimodality rule does not fire.

Its power limit is part of the record: it excludes only a bf16 handicap larger than 0.115 (50% power) or 0.154 (80% power) on 1024/256, so FEAS-2 and k = 0 carry no R5 caveat, but no parity claim at the few-point level is made.

### PR-23 · R6: run after O4-A, R2-LIVE L1/L2 and VT-24r; readings fixed now

The criteria are plan §2.6, unchanged. `experiments/prereg/R6.yaml` fixes the readings the plan leaves open:

- **Metric:** the mean of R1 D = 3 and R-load N = 8 at 1K.
- **Looks:** at 50M, 100M and 150M tokens only. A score below 0.5 at 50M also continues, because FAIL is defined at 150M.
- **FAIL:** a score < 0.5 at 150M, or the format-mismatch signature (answer CE ≤ 0.5 × its early value while the score is < 0.05).
- **Plateau** with 0.5 ≤ score < 0.9 at 150M is **INCONCLUSIVE**; the plan's text omits this case.
- **Chain consistency:** measured on all 6,400 queries against a bf16 single-pass forward (the training-precision forward), with fp32 reported alongside.
- **Infra:** a flagged attempt is repeated in full, at most 3 times.
- **Code:** the production tree with D050 and D051, and evaluation items from the fixed generator.

### PR-24 · R2(c), R2(d), R2(e), R4

- R2(c) **PASS, recorded:** the longest kernel is 25.24 ms against < 1 s, recomputed by the audit from 20 traces.
- R2(d) **recorded:** pid 799092 is narrowed to two candidates, consistent with memory exhaustion by unlocked concurrent jobs.
- R2(e) **owed now** as R2e-X (PR-04).
- R4 / G-RP(b) **PASS** (PR-05).

---

## 4. Items listed in G0.md "Before S2 FEAS"

### PR-25 · DTS-10 FEAS decision code and U-15

- **Required before any FEAS verdict.** `experiments/prereg/DTS10_acceptance.yaml` fixes the exact semantics, ahead of any FEAS data:
  - FEAS-3 (which defines b*);
  - the FEAS / U-01 rule;
  - FEAS-1/2 with the 16K fallback and the stop rule;
  - FEAS-5 (SSMax);
  - the TEF forgetting gate (per-k seed means on b*);
  - the elicitation check;
  - U-15 (pooled within-stratum ANOVA ICC; DE = 1 + 3·ICC; fires iff DE > 2);
  - I-08 (every level at floor or ceiling, mixed allowed).

  It also fixes the ladder definitions, exact rational arithmetic, refusal on missing data, and the acceptance tests.
- **U-15 consequence.** The CS confirmatory 16K/32K cells hold 300 contexts (`experiments/items/test/MANIFEST.json`). If U-15 fires, contexts 301–600 are generated prefix-stably under the VT-24r repair rule, and VT-12 and VT-24r K1 are re-run on the extension before G5.
- **Timing and blinding.**
  - FEAS **training** may start once every other entry item holds and the DTS-10 input contract is committed. It may not wait on the verdict code.
  - **No FEAS evaluation output may be opened**, printed or summarised before the code passes DTS10-AT and a registry note `DTS10-commit` records its commit and SHA-256. The TEF gates are mid-FEAS decisions, so this deadline is the first TEF-1000 evaluation.
  - FEAS verdicts come only from `feas_decisions.json`, and the FEAS gate reviewer recomputes them independently from raw rows.
- **Also owed before the first FEAS Stage-A milestone is evaluated** (plan §2.8–2.10, audit triggers): the R8 ICL / induction helper, the R10 driver and the RP-2 LM-3 reading code.

### PR-26 · `rescore_check` reference precision: dual reference

- **Keep the frozen §9.2 / I-29 check exactly** (single-pass reference under the same bf16 autocast and bf16-rounded weights). It isolates **path** artefacts at production precision, and it is the operative safeguard named by RR-G0-2.
- **Add a pre-registered precision audit:** the same 5% subset re-scored by a single-pass forward with **fp32 weights and autocast off**. This answers the audit's recommendation (§3 item 4) without merging path and precision into one criterion.
- **Where it runs:** on every FEAS checkpoint (`FEAS_precision.yaml` Q3). If flips are ≥ 1% for any k, the PI chooses before S4 among fp32-compute scoring, fp32 weights, or a per-k caveat. After FEAS it runs on the seed-0 checkpoint of each Tier-A configuration, report-only; a rate ≥ 1% reopens the decision.
- **Implementation:** `rescore_check(..., ref="bf16"|"fp32")`, both records kept in `_eval_log.jsonl`. `rescore_audit` keeps gating on the bf16 path flips.
- **R6(a) unchanged** (PR-23).

### Triton re-archive and separate `TRITON_CACHE_DIR`

Ruled in PR-12: diagnostics cache now, production restore before FEAS, re-archive at G5, durable copies.

---

## 5. New decisions

### PR-27 · Infrastructure-failure status (runbook; recommended by plan §7)

An attempt that ends with a tagged infrastructure event is recorded as `infra_failed_attempt`. Tagged events are Xid, RC watchdog, host OOM-kill, or a CUDA launch timeout or illegal address with a matching NVRM line. Such attempts:
- never count toward D031's "3 consecutive failures = failed";
- never count toward §8.1's "fails permanently".

The runner resumes from the last checkpoint. After 10 infra attempts on one run, the queue **pauses** for a PI decision; the run is never auto-failed. G-OOM (3 consecutive device OOMs → failed) is unchanged, because an OOM on an exclusive, locked GPU is structural.

Two exceptions:
- Reproduction gate runs keep the plan's rule: a flagged window is repeated in full.
- The analysis reports infra resumes per run and per k (descriptive), so any correlation with k is visible.

This is a pre-study runbook change: no S2 run has happened.

### PR-28 · H3 GLMM sensitivity before G5

D021's logit-GLMM sensitivity is not implemented, and DEV-01 forbids new packages in `.venv` before G5. Because G5 freezes the analysis code with VT-18 passing, the GLMM must exist **before G5**. It is implemented in-repo with numpy / scipy only (scipy 1.18.1 is present): a binomial-logit model with random intercepts for seed and context, fitted by Laplace approximation. A VT-18 extension must recover planted D₃ on the logit scale. The logit stand-in stays as an additional sensitivity.

### PR-29 · Process items

- (a) Every new prereg is hashed when written. `experiments/prereg/SHA256SUMS` exists now, and executors verify the hash before the first run.
- (b) The orchestrator creates the ledger `reports/reproduction/results.md` (plan rule 9; audit §3 item 6). It has one row per item (R1–R7, VT-21…25, VT-22b, VT-23b, VT-24r, R2-LIVE, O4, R2e-X), each with the claim, criterion, verdict, residual ID and raw paths.
- (c) The audit's errata are appended to `results_R3.md` and `results_R4.md` as dated **Errata** sections, never silent edits: VT-21 |dw| 332–1,191; VT-23 loss maximum 1.57e-2 at step 24 and spike at step 19; R4 3 of 9 windows contended.
- (d) The VT-21, VT-22 and VT-23 tests are marked `@pytest.mark.xfail(strict=True, reason="accepted residual RR-…; PI ruling PR-14/15/16; criterion unchanged")`. The registered criteria stay executable and an unexpected pass becomes visible. The VT-24 test is not marked: on the regenerated items it is a subset of VT-24r K1.
- (e) The VT-13 standing replicate is clarified before it runs (PR-01). It consists of two audit-mode runs of C12 seed 0's first 1,000 steps, one at S4 start and one after the last S4/S5 training run, which must be bitwise identical in loss trace and parameter SHA-256.

### PR-30 · Cost-benchmark session (observed, ~13:41)

`scripts/benchmark_inference.py` (untracked) is running a §10.2 session under the lock (label `cost-session1`, max 2.5 h, `.cache/triton_bench`). Ruling:
- It is not a gate run, so it does not breach the O4 sequencing rule, and it may finish.
- From now on, G-RV-path jobs take precedence. Further cost sessions run only when no gate-path job is waiting.
- Measured costs are valid for the code hash they record. At S6 the costs are either re-measured or shown to come from the same `src/hybridlc/{models,bench}` hashes as the G5 freeze.
- D043(1) / U-14 (idle windows) is resolved by the user's exclusive-GPU grant. Measured costs are in scope as secondary, and analytic costs remain primary (D028).
- Any infra event in its windows is reported to the PI at once as O4-relevant evidence.

---

## 6. Named residual-risk register

| ID | Item | What is accepted | Evidence | Safeguard | Reopen if |
|---|---|---|---|---|---|
| RR-G0-1 | VT-05 | fla bf16 chunk final state up to 8.47e-3 (> 8e-3) in 2/32 seeds | G0.md §2; `s0_integration.md` §12 | VT-06/07; FEAS-PREC Q2/Q3 | Q2 or Q3 ≥ 1% for any k |
| RR-G0-2 | VT-07 | branched vs single-pass 2.4–3.8e-3 nats (fp32), EM identical | `s0_integration.md` §12 | §9.2 re-score (Q2), fp32 reference (Q3) | Q2 ≥ 1% |
| RR-G0-3 | VT-13 production | attempt 1 0.0075 > 0.005 | `runs/_logs/vt13_production*.log`; `experiments/s0/vt13_production/result.json` | bitwise audit-mode resume; VT-13p2 for future audits | resume code changes |
| RR-G0-4 | VT-15 | cuDNN B = 1 hot decode −30% (faster); 2 cells unasserted | G0.md §5 | cold-L2 cells (primary) pass | an idle cold-L2 cell fails |
| RR-G0-5 | VT-03 | GDN fp32 packed vs alone 9.4e-5 (only under the fp32 reading) | `s0_integration.md` §12 item 1 | same-shape bit-identity; bf16 exact | — |
| RR-G0-6 | G0 throughput | D512 Stage A 67.2K at 35% foreign SM | `experiments/s0/g0/g0_results.jsonl` | G0-T512x on the exclusive GPU | G0-T512x FAIL → 0.6B fallback |
| RR-VT21 | VT-21 | fp32 dw absolute bound (1–2 ulp at \|dw\| ~10³) | R3 raw; audit `vt21_rerun.jsonl` | every other VT-21 criterion passes | — |
| RR-VT22 | VT-22 | C0 bf16 cross-implementation 2.02e-2 > 2e-2 | `vt22_transplant.jsonl`, `vt22_diagnosis.jsonl` | VT-22b (blocking) | VT-22b FAIL |
| RR-VT23 | VT-23 | trajectory criteria unattainable (chaotic rounding growth) | `vt23_compare.r1.jsonl`, `vt23_diagnosis.json` | VT-23b (blocking) | VT-23b FAIL |
| RR-R7 | R7 | Pythia fp32 anchors outside ±0.015 | `experiments/repro/R7/`; audit `lambada_cpu.log` | 3 scorers agree; Wang PASS | a scorer discrepancy appears |
| RR-R2 | R2(a)/(b) | not live before R4; contended R4/R5 windows | audit §1.1, §1.7 | R2-LIVE (blocking) | — |
| RR-R5-D1 | R5 | ran after the O4 trigger without O4 | R5 prereg D1; audit §1.7 | event-free windows; full repeats | O4 finds silent corruption |
| RR-G0suite-seq | G0 suite 12:35 | ran after the O4 trigger without O4 | G0.md §1 scan | event-free; deterministic; re-run pre-FEAS | O4 finds silent corruption |
| RR-prereg | process | R3 prereg modified after some verdict runs; no R2/R7 prereg files | audit §2 | all readings literal or stricter; hashing rule PR-29(a) | — |

---

## 7. New pre-registrations (written before anything runs; `experiments/prereg/SHA256SUMS`)

| File | ID | SHA-256 |
|---|---|---|
| `O4_soak.yaml` | O4 | `08217d37e0857757e28d451994cb8a251c8f6c4121ea494b2c7d4c760d2b6d99` |
| `R2_live.yaml` | R2-LIVE | `125893126a64f19ea44a0281532e98d33bc45d3ee7285373c752468b39df0e16` |
| `R2e_G0T_exclusive.yaml` | R2e-X (G0-T512x, D029 reading) | `3fa66d49e2ba42cbb302e65760fcc5b8f3c979f8a25871cd8c68868e919b82fd` |
| `VT-22b.yaml` | VT-22b | `d2d095d61d97df2e370564117d69938e3adc4063d89551e49686c71df5a00117` |
| `VT-23b.yaml` | VT-23b | `91753a3cbd41ff0a9f7ed56d10026b4c26d15217f65a1421bb11707e5d2fab71` |
| `VT-24r.yaml` | VT-24r | `b596a88f301f425b66f7ceff9a5c517462c3d4aff71a94328637bccb64f9da33` |
| `R6.yaml` | R6 readings | `49ec85f6a235f29db0eab15d8878b3ae83a6a9b3418be02598c839e82a228816` |
| `FEAS_precision.yaml` | FEAS-PREC (Q1, Q2, Q3) | `06a80265e6580773e65bae80745fa3f6001847488ac34fb8950825e452985144` |
| `DTS10_acceptance.yaml` | DTS10-AT | `35339907c756bcdc98d4fabb6da24e2f3066708d613b10caabd8ab60f478d208` |
| `VT-13p2.yaml` | VT-13p2 (future audits only) | `ec6c93d5bee260319498d318677a5a048e54a98cae4c91944ad9bc356869fb25` |
| `README.md` | rules and index | `24d0a469ad6bae53b10692d21a611bd45f2a79505121cbf4826954e22fa65e87` |

---

## 8. G-RV closure checklist (for the independent G-RV gate)

G-RV (amendment §13.3) passes when **every** row holds, each with the evidence named:

| G-RV condition | Satisfied by |
|---|---|
| G0 passed | PR-01 row entered; pre-FEAS full suite: all VT-01…20 pass, VT-21/22/23 strict-xfail, VT-24 and VT-25 pass |
| G-RP (a), (b) | PR-05 row entered |
| VT-21 | RR-VT21 row entered |
| VT-22 | RR-VT22 row entered **and** VT-22b PASS |
| VT-23 | RR-VT23 row entered **and** VT-23b PASS |
| VT-24 | generator fix merged, items regenerated, **VT-24r PASS** (K1–K6), new manifests committed |
| VT-25 | PASS (verdict run); hygiene change of PR-18 merged |
| R5 not FAIL | PASS + PR-22 row entered |
| R6 not FAIL | R6 PASS or INCONCLUSIVE under `R6.yaml` |
| R2 (a), (b) live | **R2-LIVE PASS** (C1 + C2) |
| R2(c) recorded | done (25.24 ms) |
| R7 complete | complete; RR-R7 row entered |
| O4 (plan §2.2) | **O4-A PASS** |

If any item FAILs, G-RV stays closed and FEAS does not start. The PI may only accept a further named residual by a new row, under the same rule.

## 9. FEAS entry checklist (replaces D057)

FEAS-384 training may start only when all of these hold:

1. G-RV passed (§8).
2. Rows for PR-01…PR-30 entered in `decisions.md`.
3. Production Triton cache restored, and the `pre-FEAS-triton-cache` hash recorded (PR-12).
4. R2e-X done: G-throughput references set, G0-T512x outcome recorded, D041 re-priced by decision row (PR-04).
5. Infra-status rule (PR-27) and G-div near-miss logging (PR-10) live in the runner and trainer.
6. `rescore_check` dual reference implemented and tested (PR-26).
7. DTS-10 input contract committed (PR-25). Its full acceptance plus the `DTS10-commit` note must precede the first reading of any FEAS evaluation output.
8. R8, R10 and RP-2 LM-3 helpers ready before the first FEAS Stage-A milestone is evaluated (PR-25).
9. FEAS run YAMLs reviewed against the protocol: k ∈ {0, 12}, seeds 101/102, η 2.0e-3 with one restart at 1.4e-3 (D034), Stage-B branches 30M / 90M, SSMax arm (k = 12, seed 101), TEF 1,000 with D050 block normalisation, the milestones, and production precision.
10. Clean committed tree, including the regenerated item manifests, `tests/conftest.py`, the prereg files and the fla wheel hash in provenance (PR-11).
11. Ledger `reports/reproduction/results.md` current (PR-29b).

## 10. Execution order for the implementation step (one GPU job at a time, through the lock, fraction ≤ 0.44)

**Phase I: code and CPU work (no GPU needed except the R2-LIVE C1 GPU cases):**
1. R2-LIVE L1–L3 and its C1 tests.
2. Diagnostics `TRITON_CACHE_DIR`, the production-dir guard and the `conftest.py`.
3. G-ORACLE generator fix with K6, then regeneration, then VT-24r K1–K5 (CPU).
4. `rescore_check` dual reference.
5. PR-27 infra status in the runner.
6. PR-10 near-miss logging.
7. PR-29(d) strict-xfail markers and the PR-18 hygiene change.
8. Fla wheel hash in provenance.
9. DTS-10 code, in parallel; due by PR-25's deadline.
10. Ledger and errata.
11. Commit (orchestrator).

**Phase II: GPU, strictly sequential, after `cost-session1` releases the lock:**
12. Restore the production Triton cache and create `.cache/triton-diag` (PR-12).
13. O4-A (≈ 2.4 h). If FAIL, stop and wait for a PI row.
14. R2e-X (≈ 0.6 h).
15. VT-22b (≈ 0.1 h), then VT-23b (≈ 0.2 h).
16. R6 (0.3–0.6 h).
17. Pre-FEAS full G0 suite (≈ 0.3 h).
18. R2-LIVE C2 evaluation over windows 13–17.
19. O4-B (≤ 1 h).
20. Pre-FEAS warm and archive of the production cache.
21. Independent G-RV gate.

**Budget:** ≈ 5.5 GPU-h in total. O4 (≤ 3.4 h) is charged to the optional-items contingency (≤ 4.7 h) and the rest to S0/S1.

## 11. Decision-log rows to enter on acceptance

Enter one row per ruling, PR-01 → D058 … PR-30 → D087 in order. Use the ruling's first sentence as the Decision column, its "Why" as the Rationale column, the rejected alternative it names as Alternatives, and its reopen trigger (the §6 table, or as stated in the ruling) as Reopen. At the same time, mark the status of D045–D057 as "DECIDED by PI 2026-11-10 → D058–D070 (see `reports/pi_rulings_2026-11-10.md`)". The PROPOSED text stays in the log unchanged.

## 12. Not verified, and open questions for the user

- The root cause of the Xid-8 events. O4-B attempts a localisation, but CUDA_LAUNCH_BLOCKING may suppress the hang.
- Whether running this GPU headless would remove the RC-watchdog interplay. This needs the user, and is raised only if O4-A FAILs.
- The fp16 code path of the 2023 Pythia records (`lm-eval @2b9a6e94`). It is not needed for RR-R7.
- How many facts the G-ORACLE repair will change in cells outside VT-24's scope (< 16K, DC, dev/MS). This is reported by VT-24r and expected to be small, concentrated in T4 and high-N R-load.
- Whether R6's 2-layer configuration (`n_layers=2` through `ModelConfig.for_scale`) is supported without a driver. If not, the plan's fallback `scripts/repro/known_answer.py` applies.

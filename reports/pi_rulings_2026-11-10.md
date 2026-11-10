# PI rulings, 2026-11-10: D045–D057, the G-RV residuals, and the pre-FEAS items

**Status: FINAL.** Author: principal investigator (finalising step of workflow `wf_b92a39ce-f17`). Written 14:30–15:00 IST on 2026-11-10. This version supersedes the DRAFT of 13:25–14:00. The DRAFT, as reviewed, is archived unchanged at `reports/reviews/pi_rulings_2026-11-10.DRAFT.md` (SHA-256 `42e2907a37b791ed6ac236838ffe1f45c384f855a517bc7b4b9f609d21faf40a`).

**Basis:** the adversarial review `reports/reviews/review_pi_rulings_2026-11-10.md` (13:50–14:30). Its verdict was ACCEPT WITH REQUIRED CHANGES before Phase II. Every required change is accepted below. Where my implementation of a change differs from the reviewer's specific suggestion, §1 says how and why.

**Effect:** in force through decision rows **D058–D091**, appended to `reports/decisions.md` in this step. D045–D057 are marked ACCEPTED, REJECTED or AMENDED in place, with their PROPOSED text kept verbatim. The orchestrator commits (task T00).

**What I ran:** no GPU job. Read-only checks only:
- `sha256sum -c experiments/prereg/SHA256SUMS`: all 11 batch-1 files OK.
- No output exists for any pending item. Checked: `experiments/repro/` holds only `O4/pi_klog_scan_*.json`; `experiments/feas*` does not exist; `.cache/triton-diag` does not exist.
- Re-reads of the raw evidence the review cites:
  - `runs/_logs/warm_triton.log` lines 1 and last, and the `S0-triton-cache` row of `experiments/registry.jsonl`;
  - `experiments/bench/train_step_idle_20261110.json`;
  - `experiments/repro/R3/vt22/vt22_diagnosis.jsonl`;
  - `tests/test_vt0{3,5,7}_*.py`, `tests/test_vt15_bench_harness.py`, `tests/test_vt13_training.py` L405–415, L570–640;
  - `src/hybridlc/data/tef.py` L1–95, L360–380;
  - `src/hybridlc/models/attention.py` (backend pins);
  - `results_R5.md` L105–159 and `results_R7.md` L94–106.

**What I wrote:**
- this file;
- the decision rows;
- 14 batch-2 pre-registration files, `experiments/prereg/SHA256SUMS.v2` and `README.v2.md`, written 14:33–14:53 and hashed at 14:53:23, before any of their items has run.

I wrote nothing in `src/`, `tests/`, `scripts/` or `configs/`, and committed nothing.

---

## 0. Rules applied

1. **Integrity rule (unchanged).** A pre-registered criterion is never restated after its item has run. For every item that failed as registered I chose exactly one route:
   - **(i)** accept a named residual risk (`RR-…`, §7) with evidence;
   - **(ii)** fix a genuine defect and run a NEW check, pre-registered in `experiments/prereg/` before it runs;
   - **(iii)** block.

   A new check beside a residual answers a different, sharper question. It never converts the original FAIL. Pre-run changes to items that have not run are made only by a new file version plus a decision row.
2. **Verdict vocabulary (new, D088; batch-2 README rule 6).**
   - A gate or item reads **PASSED** only if every condition was met as registered. Otherwise it reads **NOT PASSED as registered**.
   - Each condition is reported as one of: MET; MET via new check (route ii, original FAIL stays); NOT MET → RR-… (route i, with its row); or NOT MET → blocking.
   - Permission to proceed is a separate line, for example "FEAS entry PERMITTED under G-RV's named-residual clause".
   - "PASS-WITH-…", "conditional pass" and counts like "N/N" that omit skips are not used.
3. **Unexpected passes (new, batch-2 README rule 7).** A test of a registered criterion that already failed is marked `xfail(strict=False)`. An XPASS is recorded and reported. It is not a PASS of the item and never blocks.

Ruling labels: ACCEPTED (as proposed), AMENDED (accepted with stated changes), REJECTED. "FAIL-as-registered" means the literal pre-registered criterion was not met.

---

## 1. Response to the adversarial review

### 1.1 Objections O1–O14

| # | Objection | Response | Change made (FINAL) | Where |
|---|---|---|---|---|
| O1 | Precision residuals that differ by k have safeguards that cannot see a k-difference and do not feed the FEAS verdict | **ACCEPTED** (two details modified, §1.2 items 1–2) | `FEAS_precision.v2.yaml`:<br>• Q1–Q4 are computed by the DTS-10 code before any FEAS rule;<br>• pooled **and restricted** per-k flip rates on informative cells, separately at 16K and 32K;<br>• new **Q4 between-k rule**: every ladder rule is computed from P0 and from a new fp32 cache-path pass **P4**, and the verdict uses P4 if any outcome differs or the 32K contrast S₁₂ − S₀ moves by > 0.02 for either seed;<br>• Q3's option (c) is allowed only if the contrast is unchanged within 0.02 under P4;<br>• the RR-G0-1/2 reopen triggers use restricted rates;<br>• RR-D052i is named | D065, D076, D082, D083, D090; `DTS10_acceptance.v2.yaml`; T14, T15, T19 |
| O2 | PR-04's G0-T512x is a check with a known outcome that replaces a pre-registered consequence | **ACCEPTED** | Logged deviation **DEV-03** waives the 0.6B fallback under D031's reopen clause. The G0 D512 throughput sub-item stays FAIL-as-registered (RR-G0-6). R2e-X v2 is a budgeting re-measurement, with a waiver-lapse guard that carries no evidential weight. The alternative "apply the fallback" is rejected (reason in PR-04) | D048 mark, D061; `R2e_G0T_exclusive.v2.yaml`; T36 |
| O3 | Gate labels relabel FAILs; VT-03 is classified two ways; "447/447" | **ACCEPTED** in full | • Vocabulary rule D088.<br>• G0 = **NOT PASSED as registered**, plus **DEV-02**: the fix-and-retest clause is not applied, and FEAS may start only under G-RV's named-residual clause.<br>• VT-03 GDN fp32 = FAIL-as-registered → RR-G0-5; the reading made after the result is withdrawn.<br>• Counts: "447 passed / 4 skipped; 13 test cases assert relaxed regression guards".<br>• Registered-criterion `xfail(strict=False)` tests for VT-03, VT-05, VT-07 and VT-15 | D058, D059, D088; §9; `G0suite_preFEAS.yaml`; T01, T18 |
| O4 | R6-R3(ii) will likely give a false FAIL | **ACCEPTED** (anchor modified, §1.2 item 3) | `R6.v2.yaml`:<br>• (ii) uses an absolute anchor: retrieval-family answer CE ≤ 0.50 nat/token **and** score < 0.05;<br>• report-only `--` (GDN) chain consistency, with a pre-registered RR-G0-1/2 reopen at ≥ 1% | D080; T16, T40 |
| O5 | O4's design and order | **ACCEPTED** | `O4_soak.v2.yaml`:<br>• **O4-Z**: 60-min soak of the failing Zoology GDN cell with normal launches; P(0 events \| R5 rate) = 0.033; outcome meanings pre-registered;<br>• then **O4-B**, then **O4-A**, all before any gate run;<br>• the DRAFT's deferral of O4-B is withdrawn;<br>• per-k resume monitor (flag F-infra) | D078, D082; T31–T33 |
| O6 | Two undefined outcomes would block G-RV (VT-22 XPASS, pinned VT-25) | **ACCEPTED** | `G0suite_preFEAS.yaml`:<br>• `strict=False` markers; an XPASS is recorded, is not a PASS and never blocks;<br>• VT-25 tiers: ≤ 2e-2 consistent; ≤ 4e-2 → RR-VT25, no block; > 4e-2 → block | D075, D086(d); T18, T41 |
| O7 | PR-12 rests on a false premise and a spliced hash; R2e-X's cache is ambiguous | **ACCEPTED** | `TRITON_prefeas.yaml`:<br>• fresh exclusive warm of an **empty** production cache before FEAS;<br>• full hashes, and an autotune-difference report;<br>• R2e-X runs on a fresh extraction;<br>• the G5 re-warm is add-only;<br>• errata for the spliced hash | D056 mark, D069; T01, T30, T34 |
| O8 | Parts of VT-22b and VT-24r cannot fail | **ACCEPTED** | VT-22b v2:<br>• predictions stated before the run;<br>• only C0 F1/F2 at 32K and the new C12 fp32 arm F3/F4 carry weight.<br>VT-24r v2:<br>• an independent checker, and K0 agreement on the pre-fix items;<br>• K5a leak unit test and K5b full-block hash | D072, D074; `VT-22b.v2.yaml`, `VT-24r.v2.yaml`; T13, T14, T38 |
| O9 | RR-G0-3's reopen trigger fires but no real-model resume check is scheduled | **ACCEPTED** | • Runner-side changes **count as resume code**.<br>• **VT-13r** is mandatory: trainer and runner paths, ≈ 0.55 GPU-h.<br>• `VT-13p2.v2.yaml` narrows the production audit's trigger (§1.2 item 5) | D060; `VT-13r.yaml`; T37 |
| O10 | PR-27 treats every Xid as infrastructure | **ACCEPTED** | PR-27 v2:<br>• infra status only for Xid 8 with an RC watchdog, Xid 79 and host OOM-kill;<br>• `suspect_sw_failure` for Xid 13/31/43, illegal-address faults and unknown Xids. It counts as a failure, and a recurrence at the same step stops the run;<br>• counts reported per k | D084; `R2_live.v2.yaml`; T10, T11 |
| O11 | A k-differential residual is unnamed (R5 power) | **ACCEPTED** | RR-R5-power is named and carried to the limitations | D079, D090; §7; T43 |
| O12 | PR-06 omits protocol §7.3 "in the step" | **ACCEPTED** | Recorded as pre-run protocol change **DEV-04**, with the change in 32K answer weight stated | D050 mark, D063 |
| O13 | Items missing before FEAS spends ~10 GPU-h | **ACCEPTED** | • `FEAS_smoke.yaml`;<br>• the D041 re-pricing covers FEAS-PREC P1–P4 and every new Phase II item;<br>• the O1, O4, O6 and O7 fixes are filed as batch-2 files | D061, D070; T35, T36 |
| O14 | Minor items | **ACCEPTED** (all five) | • G-div near-miss FEAS flag;<br>• R2-LIVE C2 invalidation rule;<br>• technical DTS-10 blinding;<br>• DTS-10 readings stated as readings;<br>• commit `scripts/benchmark_inference.py` | D067, D077, D082, D087; T03, T20 |

### 1.2 Where I implemented a required change differently from the reviewer's suggestion

All of these are pre-run.

1. **O1: informative cells are defined symmetrically.** A cell counts if max(EM_a, EM_b) > 0.05 and min(EM_a, EM_b) < 0.95. The review defined it "under P0". Conditioning on P0 alone would drop a cell that collapses under bf16, with P0 at floor and the reference high, and that is exactly RR-R7's failure shape.
2. **O1: the between-k rule compares P0 with P4, and P4 always runs.** P4 uses fp32 weights and fp32 compute on the same cache path. The review proposed "fp32 scoring when P0 and P1 differ".
   - P1 keeps compute in bf16, so it cannot see RR-R7's mechanism.
   - A trigger estimated from the 5% subset is too noisy at the 0.02 scale (about 140 queries per checkpoint at 32K).
   - P4 costs an estimated 1–2 GPU-h and is priced in the D041 re-pricing row.
3. **O4: the CE anchor is 0.50 nat per token on R1-type and R-load-type answer tokens.** The review suggested about 1.0 nat on all answer tokens.
   - Five-ninths of TEF answer tokens come from M-hop and aggregation items, which a 2-layer model may never learn.
   - 0.50 nat is the conservative side against a false FAIL.
   - The review's other option, "(ii) only at 150M", is rejected: at 150M, (ii) is redundant with (i).
4. **O8: K0 separates two checks.** The location rule (`loc_ok`) is what the repair enforces. The InductionLM prediction rule (`pred_ok`) is what VT-24's oracle tests.
   - A coincidental match, where another fact continues with the same digit, passes VT-24's oracle but not G-ORACLE. So "exactly the 7 misses" holds for `pred_ok` only.
   - The DRAFT's "equivalently, InductionLM predicts every gold digit" was not exact.
5. **O9: VT-13p2's trigger is narrowed.** VT-13p2 never ran.
   - With runner changes counted as resume code, v1 would mandate a 2.2 GPU-h statistical audit that has no power at 0.005, given the clean-vs-clean noise of 0.0042.
   - The bitwise VT-13r is strictly more sensitive on every code path that audit mode runs.
   - VT-13p2 now covers only production-only state code.
6. **O6: the upper VT-25 tier is 4e-2**, twice the bound and half the smallest planted-defect effect (8e-2), so a real regression still blocks.

### 1.3 New PI finding (not raised by the review)

**fp32 attention needs a memory-efficient path.** fp32 attention runs on the SDPA **MATH** backend (`src/hybridlc/models/attention.py` L18–23, L183–195, L206–221). MATH materialises a [heads × T × T] fp32 score matrix: about 25.8 GB at 32K and 6.4 GB at 16K for 6 heads, plus the softmax output.
- The DRAFT's fp32 references at 16K/32K for k > 0 (FEAS-PREC P3) therefore could not run under the 0.44 cap.
- A C12 fp32 transplant arm has the same problem.
- **Requirement R-ATT** (`FEAS_precision.v2.yaml`): an explicitly selected fp32 reference attention path, either exact query-tiled attention or SDPA pinned to EFFICIENT. It must be validated against MATH at short T (≤ 1e-5 relative RMS) and fit in 7 GiB at 32K. Production paths are unchanged. Task T14.

---

## 2. Rulings at a glance (FINAL)

| ID | Subject | Review | FINAL ruling | Row |
|---|---|---|---|---|
| PR-01 | D045 G0 verdict | OBJECTION | AMENDED:<br>• G0 = **NOT PASSED as registered**;<br>• six FAIL-as-registered sub-items → RR-G0-1…6;<br>• DEV-02 (no fix-and-retest without a defect);<br>• FEAS only under G-RV's named-residual clause;<br>• honest suite counts | D058 |
| PR-02 | D046 VT readings | OBJECTION (4) | AMENDED:<br>• VT-05, VT-07, VT-15 → RR-G0-1/2/4;<br>• **VT-03 = FAIL-as-registered → RR-G0-5**, no alternative wording;<br>• VT-18 PASS;<br>• registered-criterion xfail tests | D059 |
| PR-03 | D047 VT-13 production | OK (trigger fires) | AMENDED:<br>• attempt-1 FAIL stands (RR-G0-3);<br>• runner changes count as resume code → **VT-13r** mandatory;<br>• VT-13p2 v2 | D060 |
| PR-04 | D048 G0 throughput | OBJECTION | D048 REJECTED:<br>• D384 MET;<br>• D512 FAIL → RR-G0-6;<br>• **DEV-03 waives the 0.6B fallback** (D031);<br>• R2e-X v2 = budgeting only | D061 |
| PR-05 | D049 G-RP | OK | ACCEPTED: PASSED | D062 |
| PR-06 | D050 TEF `answer_norm: block` | OK with correction | ACCEPTED as pre-run protocol change **DEV-04** (§7.3 "in the step") | D063 |
| PR-07 | D051 conv init | OK | ACCEPTED, option (a) | D064 |
| PR-08 | D052 evaluation disclosures | OK, measurement weak | AMENDED:<br>• (i) → RR-D052i, measured by FEAS-PREC v2 Q1/Q4;<br>• (ii)–(vii) accepted | D065 |
| PR-09 | D053 H5(a) label | OK | ACCEPTED | D066 |
| PR-10 | D054 G-div | MINOR | AMENDED:<br>• literal reading;<br>• near-miss logging;<br>• **FEAS verdict flag** | D067 |
| PR-11 | D055 fla wheel | OK | ACCEPTED as DEVIATION; wheel hashes in provenance | D068 |
| PR-12 | D056 Triton cache | OBJECTION | AMENDED:<br>• diagnostics cache now;<br>• **fresh exclusive warm before FEAS (TC-PF)**;<br>• add-only at G5;<br>• hash corrected | D069 |
| PR-13 | D057 FEAS entry list | add O13 | AMENDED: replaced by §10 | D070 |
| PR-14 | VT-21 | OK, exemplary | Route (i) RR-VT21; no new check | D071 |
| PR-15 | VT-22 | MINOR | Route (i) RR-VT22 + blocking **VT-22b v2** (predictions; C12 fp32 arm) | D072 |
| PR-16 | VT-23 | OK | Route (i) RR-VT23 + blocking VT-23b (v1 unchanged) | D073 |
| PR-17 | VT-24 | OK with conditions | Route (ii) fix, regenerate before FEAS, blocking **VT-24r v2** | D074 |
| PR-18 | VT-25 | OBJECTION | The registered verdict PASS stands; RNG pinned; **pre-registered outcome tiers** | D075 |
| PR-19 | R7 | OK | Route (i) RR-R7, not blocking; k-relevance carried by FEAS-PREC v2 Q3/Q4 | D076 |
| PR-20 | R2(a)/(b) | OK | FAIL-as-registered (RR-R2); implement; blocking **R2-LIVE v2** with C2 invalidation | D077 |
| PR-21 | O4 soak | OBJECTION | **O4 v2**: O4-Z → O4-B → O4-A before any gate run; R5 and the 12:35 suite stand | D078 |
| PR-22 | R5 deviation D1 | OK | ACCEPTED; **RR-R5-power** named | D079 |
| PR-23 | R6 | OBJECTION | **R6 v2**: absolute CE anchor; report-only `--` chain check | D080 |
| PR-24 | R2(c), R2(d), R2(e), R4 | OK | R2(c) PASS recorded; R2(d) recorded; R2(e) = R2e-X v2; R4 PASS | D081 |
| PR-25 | DTS-10, U-15 | OK | **DTS10-AT v2**: precision rules first, verdict flags, readings, technical blinding | D082 |
| PR-26 | `rescore_check` precision | OBJECTION | **FEAS-PREC v2** (P0–P4, Q1–Q4) + R-ATT | D083 |
| PR-27 | Infrastructure status | OBJECTION | **PR-27 v2**: infra vs `suspect_sw_failure` | D084 |
| PR-28 | H3 GLMM | OK | Before G5, in-repo | D085 |
| PR-29 | Process | OBJECTION on (d) | Batch-2 hashing; ledger; errata; `strict=False`; standing replicate | D086 |
| PR-30 | Cost-benchmark session | OK | As DRAFT, and commit the script before citing numbers | D087 |
| PR-31 | Verdict vocabulary (new) | — | §0 rule 2 | D088 |
| PR-32 | Implementation order and Phase II entry (new) | — | § Implementation tasks | D089 |
| — | Residual register | — | §7 | D090 |
| — | Batch-2 pre-registrations | — | §8 | D091 |

---

## 3. The PROPOSED rows D045–D057

### PR-01 · D045 → D058, G0 verdict: AMENDED

**Ruling.** G0 is **NOT PASSED as registered**. Six sub-items did not meet their literal criterion:

| Sub-item | Literal criterion | Measured | Disposition |
|---|---|---|---|
| VT-03 GDN packed vs alone | max abs diff 0.0 (no dtype qualifier in §16) | bf16 exactly 0.0; **fp32 9.4e-5** (GEMM-shape TF32; same-shape bit-identity asserted) | FAIL-as-registered → **RR-G0-5** |
| VT-05 bf16 final state | ≤ 8e-3 relative max | 3.0e-3…8.47e-3; **2 of 32** seed-cases above 8e-3 (fla bf16 chunk kernel; recurrent kernel within 1.1e-5) | FAIL-as-registered → RR-G0-1 |
| VT-07 branched vs separate forward | \|Δ logp_gold\| < 1e-3 (fp32) | **2.4–3.8e-3** vs single-pass (TF32 chunk vs IEEE fused_recurrent); 0.0 vs identical-kernel prefill; EM identical | FAIL-as-registered → RR-G0-2 |
| VT-13 production resume | \|Δloss\| < 0.005 at resume + 500 | attempt 1 **0.0075** | FAIL-as-registered → RR-G0-3 (PR-03) |
| VT-15 Appendix A within 15% | all 11 cells | 8 asserted; cuDNN B = 1 hot-L2 decode **−30%** on the idle GPU (faster); FLASH B = 1 hot −0%; eager 4K GDN +6% | FAIL-as-registered → RR-G0-4 |
| Throughput, D512 Stage A | ≥ 70K tok/s, "contended" | pass 1 C3 **67.2K** at 35% foreign SM | FAIL-as-registered → RR-G0-6; fallback waived by DEV-03 (PR-04) |

**MET as registered:**
- VT-01, 02, 04, 06 (fp32), 08–12, 14 and 16–20;
- VRAM 6.00 GiB ≤ 7 GiB;
- transplant 7.0e-7 / 4.6e-4 ≤ 1e-3;
- Triton archive present (`f417e487a0341891873b8fd80f5f4475a01c8564c884da12bd0418966896e3de`). It was warmed under contention, which matters for PR-12, not for VT-19.

**Suite record** (`runs/_logs/g0_full_suite.log`, 12:35–12:53):
- VT-01…VT-20: **447 passed / 4 skipped / 0 failed**. The 4 skips are the env-gated VT-13 real-model audit ×3 and the production test.
- **13 of the passing test cases assert relaxed regression guards**, not the registered criterion: VT-03 fp32 linear mixers ×4, VT-05 bf16 ×4, VT-07 ×4 and VT-15 ×1.

**DEV-02 (logged deviation from G0's failure clause).** The frozen clause is "Fix and re-test", and it has no residual route.
- For these sub-items no defect is isolated (G0.md §2; `s0_integration.md` §12).
- Re-testing reproduces deterministic numbers (fla kernels; L2 residency) or measures a condition that has since changed (throughput).
- So the clause is not applied. The sub-items are carried as named residuals, and **FEAS may start only under G-RV's named-residual clause** (amendment §13.3: "No FEAS until G-RV passes, or until the PI logs a decision row that accepts a named residual risk").
- This is a deviation, not a pass. The DRAFT's label "PASS-WITH-ACCEPTED-RESIDUALS" is withdrawn.

**Deferred by design** (neither failure nor pass):
- The VT-06 bf16 trained-checkpoint sub-item moves to FEAS-PREC v2 Q2 (frozen §9.2).
- The VT-13 standing replicate is set by PR-29(e).

**Condition.** SUITE-PF outcome O-a must hold on the final pre-FEAS tree. Any VT-01…VT-20 failure there reopens G0 as a candidate regression.

### PR-02 · D046 → D059, VT threshold readings: AMENDED

- **(1) VT-05.** The median-of-8 reading is not adopted as the verdict. The per-case FAIL is RR-G0-1.
  - The median and the per-seed ceiling of 1e-2 stay as a regression guard, labelled as such.
  - A registered-criterion test (every seed-case ≤ 8e-3) is added with `xfail(strict=False)`.
- **(2) VT-07.** RR-G0-2.
  - The operative safeguards are FEAS-PREC v2 Q2 (frozen §9.2), Q3 and Q4.
  - A registered-criterion test (single-pass < 1e-3) is added with `xfail(strict=False)`.
- **(3) VT-15.** RR-G0-4.
  - The row records that the cuDNN B = 1 hot cell is *faster* than Appendix A on the idle GPU (G0.md §5).
  - The cold-L2 cells, which are the primary §10.2 condition, pass (+2% / −1%).
  - The registered 11-cell test is kept with `xfail(strict=False)` (non-strict, because the cells are bimodal).
- **(4) VT-03: FAIL-as-registered.**
  - The attention criteria pass (fp32 ≤ 1e-5 asserted, `tests/test_vt03_doc_isolation.py` L97).
  - The GDN criterion "max abs diff 0.0" carries no dtype qualifier in protocol §16, while the attention clause in the same cell does. fp32 measured 9.4e-5.
  - The DRAFT's "pass if the 0.0 bound is read in the training dtype" was a reading proposed after the result was visible, which prereg README rule 3 forbids. It is withdrawn.
  - → **RR-G0-5**, with no alternative wording.
  - The RoPE negative-control limitation is a test-design note; position resets are verified on the data side (VT-11).
- **(5) VT-18: PASS as registered.** The strict ±0.01 bound applies to every debate-§6.2 cell (`test_pisim_debate_62_within_001`); the other 329 cells go beyond the registration. The missing H3 GLMM is an analysis-plan gap, handled in PR-28.

### PR-03 · D047 → D060, VT-13 production resume: AMENDED

**Attempt 1 FAILED as registered** (0.0075 > 0.005; `runs/_logs/vt13_production.log`).
- Attempt 2 (`experiments/s0/vt13_production/result.json`: |a−b| 0.0023, |c−b| 0.0019, |a−c| 0.0042) is corroboration only. Taking it as the verdict would be selection on outcome.
- **RR-G0-3** is accepted because resume is bitwise exact in audit mode for k ∈ {0, 3, 12}, including kill -9 mid-accumulation and inside a checkpoint write (`s0_integration.md` §6).

**Re-verification is required (review O9).**
- **Runner-side changes count as resume code.** R2-LIVE v2 (lock and token before the first CUDA call in `train.py` and `runner.py`, infra blocks, runner `attempt_end` rows) and PR-27 v2 (which attempts the runner resumes) change the code that performs and records resumes. RR-G0-3's trigger has therefore fired.
- **VT-13r** (`experiments/prereg/VT-13r.yaml`) is mandatory on the final pre-FEAS tree:
  - D384 C3, audit mode, 200 steps, real data;
  - a clean runner run is compared with three resumed runs: (r1) a trainer-level kill -9 at step 137 mid-accumulation; (r2) a runner-level kill with an injected Xid-8 + RC-watchdog line, which must be classified `infra_failed_attempt` and auto-resumed; (r3) a kill inside the step-150 checkpoint write;
  - all three must be bitwise identical to the clean run, plus classification and counting checks;
  - ≈ 0.55 GPU-h;
  - mandatory again after any later change to the resume code listed in the file.
- **Why it matters for k:** resumes driven by hangs concentrate in GDN workloads (the O4 soak evidence, PR-21; review O5).

**VT-13p2.** The production statistical audit keeps its v1 design and criterion, max(0.005, 2·|a−c|). `VT-13p2.v2.yaml` narrows its trigger to code that runs only in production mode and touches checkpointed state (§1.2 item 5).

### PR-04 · D048 → D061, G0 throughput: D048 REJECTED, replaced

- **Not adopted:** defining "contended" as the pass-2 condition. Choosing the definition that passes, after both passes were seen, is a restatement.
- **D384: MET** in both passes:
  - Stage A 97.7–111.5K ≥ 95K;
  - C12 Stage B 45.8K ≥ 27K;
  - source `experiments/s0/g0*/g0_results.jsonl`.
- **D512 Stage A: FAIL-as-registered** (pass 1 C3 67.2K < 70K) → RR-G0-6.
- **DEV-03 (logged deviation): the pre-registered consequence "Stage A drops to 0.6B" is waived.**
  - Ground: D031's reopen clause ("The GPU becomes exclusive → re-measure throughput at G0 and re-budget"), fired by the user's exclusive-GPU grant (STATUS ~06:10).
  - The fallback existed to fit D512's Stage A into a contended budget. On the exclusive GPU it would cut D512 Stage-A tokens by 25% for no budget reason, and would weaken FEAS-512 if it is ever needed.
  - The fallback applies study-wide, so waiving it does not bias k.
  - The alternative of applying the fallback is rejected for these reasons.
- **Withdrawn:** the DRAFT's G0-T512x as a deciding test.
  - Its outcome is effectively known: the idle bench gives D512 Stage A 97.5 / 97.9 / 99.1K (`experiments/bench/train_step_idle_20261110.json`), and pass 2's best windows were 93.3–95.9K, against 70K.
  - Using such a check to replace a pre-registered consequence is the move PR-19 refuses for R7.
- **R2e-X v2** (`R2e_G0T_exclusive.v2.yaml`) re-measures throughput on the exclusive GPU, for budgeting and guard references only.
  - **Waiver-lapse guard:** if any exclusive D512 Stage-A median is < 70K, the waiver lapses and the fallback applies to every D512 run.
  - The guard is not expected to fire and carries no evidential weight.
  - Every report states the G0 FAIL beside the re-measurement.
- **G-throughput reference** = the R2e-X exclusive medians. This is set before the guard has ever run.
- **D041 re-pricing row** is owed after R2e-X. It must include FEAS-PREC v2 P1–P4, O4 v2, VT-13r, TC-PF, FEAS-SMOKE and VT-22b v2 (review O13).
- **D029 reading.** R2e-X records it: audit-mode overhead ≤ 3% → deterministic production. It is not expected to fire, since deterministic FA2 backward costs +43–66%.
- **Cache.** R2e-X runs on a fresh extraction of the TC-PF archive (PR-12).

### PR-05 · D049 → D062, G-RP: ACCEPTED

G-RP **PASSED** on re-run numbers only, recomputed by the gate reporter from raw JSONs and confirmed by the audit's independent re-evaluation (`our_gdn-d128`: 0.9888560 / 0.9224297, exact). The known limit is recorded and is not grounds to withhold: (a) is at ceiling (±0.03 overall = ±0.21 on the live cell). R5 supplies the below-ceiling calibration.

### PR-06 · D050 → D063, TEF loss normalisation: ACCEPTED as protocol change DEV-04

`answer_norm: block` is adopted.

**DEV-04 (pre-run protocol change).** Protocol §7.3 reads "½·mean CE over answer tokens of synthetic items + ½·mean CE over all other tokens **in the step**". The block reading departs from that wording. It resolves a conflict between §7.3's text and the purpose of D014 and §14:
- D014's decision text has no "per step", and its rationale is "length balance removes the ≈ 20× short-item skew".
- §14 names "length-balanced TEF" as an H5(a) control.
- The per-step reading gives the 32K bucket 0.29–0.30 of the answer weight, with a 27.5× per-token spread (`s0_integration.md` §12 item 14). The block reading gives 1/7 ≈ 0.143 per bucket.

The change is identical across k, but it changes exposure at the primary cell, so it is recorded as a protocol change, not a clarification (review O12). It is allowed because no TEF run of the study has happened. `answer_norm: step` remains available only through a new row.

### PR-07 · D051 → D064, conv-kernel init: ACCEPTED (option (a), N(0, 0.02))

- "fla's own initialisation" (§7.1, D008) is the model-level `_init_weights` that every fla model uses: `GatedDeltaNetForCausalLM` (the VT-09 target) and Wang's checkpoints.
- The constructor's U(−0.5, 0.5) is PyTorch's `nn.Conv1d` default, not an fla choice. F_gc's conv follows the same init.
- No study run is affected. R6, VT-23b, O4-A and FEAS-SMOKE all run on this init.

### PR-08 · D052 → D065, evaluation disclosures: AMENDED

**(i) bf16-rounded evaluation weights: DEVIATION accepted as named residual RR-D052i.**
- It differs by k: it perturbs GDN `A_log`/`dt_bias` (relative ≈ 3.5e-3), and low k has more GDN decay parameters. The effect plausibly grows with needle distance (`s0_integration.md` §12 item 10).
- It is measured by FEAS-PREC v2:
  - **Q1**: pooled and restricted per-length, per-k flip rates of P0 vs P1; any rate ≥ 1% → fp32 weights before S4;
  - **Q4**: the between-k rule on the FEAS verdict.

**(ii)–(vii) accepted as proposed:**
- (ii) LM-1 = 279 items, because only 78 PG19-test books reach 32,768 tokens.
- (iii) M-hop depths are not decile-stratified. The items are identical across k, so this cancels in paired contrasts.
- (iv) Absent candidates can occur in filler. The analysis must use `absent_in_context`.
- (v) 907 items per bucket.
- (vi) Repetitive filler in 10% of filler-bearing items.
- (vii) Short-T A-freq wording.

### PR-09 · D053 → D066, H5(a) label: ACCEPTED

Under a true H5(a), the k = 12 vs k = 0 gap is small at short T, so G-REF's discriminability part fails there by construction. Applying the full rule at every length would label every H5(a) result "conditional" and drop the short lengths from the slope, regardless of reference competence. This is a pre-run analysis decision.

### PR-10 · D054 → D067, G-div reading: AMENDED

- **Keep the literal frozen reading:** EMA₀.₉₉ keeps updating during an excursion. Freezing the EMA would add automatic restarts, interventions that could correlate with k.
- **Near-miss logging:** `training/gdiv.py` also computes a log-only frozen-EMA excursion (loss > pre-excursion EMA + 1.0 for > 50 steps). Whenever that holds without the literal trigger, it writes a `gdiv_near_miss` registry note.
- **FEAS consequence (pre-registered now; review O14):**
  - Any FEAS run with a near-miss note flags every FEAS verdict that reads that run (DTS10-AT v2 flag F-nearmiss).
  - The PI enters a decision row before such a verdict is acted on.
  - Reason: an untriggered 1.0–1.65-nat excursion in a k = 12 run could otherwise move U-01 to D512 unseen.
- **Reopen** if an S2 run shows an untriggered, sustained 1.0–1.65-nat excursion.

### PR-11 · D055 → D068, fla wheel: ACCEPTED as DEVIATION

- **Why the differences do not matter:** relative to clone `98193113`, the wheel's GDN-path differences are dispatch decorators, a Hopper-only branch and int64 offsets in `causal_conv1d`, all irrelevant at offsets ≤ 38M.
- **Every result already used the wheel:** VT, R3, R4, R5 and R7 all ran on it. VT-21 shows its conv correct to fp32 rounding with zero leakage, and R7 reproduces Wang's 340M LAMBADA with it (0.3435 vs 0.348).
- **Required:** record the wheel's `RECORD` file hash and `dist-info` SHA in `env_info()["fla_install"]` now, and again in the G5 freeze row.

### PR-12 · D056 → D069, Triton cache: AMENDED

**1. Now: separate caches for diagnostics and production.**
- Every reproduction, diagnostic, verification and test job uses `$REPO/.cache/triton-diag`, extracted from the S0 archive `runs/_artifacts/triton_cache.tar` (`f417e487…96e3de`).
- `tests/conftest.py` sets this cache. The cost benchmark keeps `.cache/triton_bench`.
- The trainer refuses an exp_stage S2…S8 run whose cache is neither the production dir nor a fresh extraction of the pre-FEAS archive, checked by hash at start.

**2. Before FEAS: TC-PF** (`TRITON_prefeas.yaml`).
- Move the live `.cache/triton` aside to `.cache/triton-pre-restore-20261110/`. Keep it, hash its tarball, and copy it out of the repo.
- Warm a **fresh, empty production cache on the exclusive GPU** (≈ 26 min; the S0 warm took 1,579 s).
- Archive it deterministically. Record the full 64-hex trainer-style hash and the archive SHA-256 in the registry note `pre-FEAS-triton-cache`.
- FEAS starts from this cache.
- **Coverage (W3):** FEAS-SMOKE and R2e-X's production cells must compile 0 new files in fresh extractions. Otherwise, re-warm from empty before FEAS.
- **Report-only:** the number of autotune choices that differ from the S0 archive.

**3. At G5: add-only re-warm.**
- Every pre-existing file must be byte-identical after the re-warm. Otherwise the PI logs a row, because a changed entry is a mid-study cache change.
- Freeze the new archive SHA.

**4. Durability.** Copy every archive outside the working tree (`~/outeropt_artifacts/`).

**Corrections.**
- **(a) The DRAFT's premise was false.** The S0 archive was itself warmed under contention: 2026-11-09 23:13–23:39 (`runs/_logs/warm_triton.log`), with CARLA running (STATUS 22:15, ~3.6 GB) and the VT-13 C12 audit running concurrently (the warm OOM'd it; `s0_integration.md` §6). Restoring it would not meet E10.2 ("pre-warmed on an idle GPU").
- **(b) The verification hash was spliced from two hashes.**
  - The S0 archive's trainer-style hash is `be0c80b1fdd326545ff3152904dec314fd81b76d318160c89f6cd285cd09fed7` (registry row `S0-triton-cache`, after the warm).
  - `1330ca4516aa0d8e15bbe01fd2c4a3e21f44abb31d6afd9c42efdf08f4f74316` is the pre-warm hash (warm log line 1).
  - "be0c80b1…f74316" splices the two. It originated in `s0_integration.md` §8 and was copied into G0.md §6 and the DRAFT. Errata are task T01.
  - Hashes are quoted in full from now on.

This is not "clearing mid-study" (§18): the study begins at S2, and the pre-restore cache is kept.

### PR-13 · D057 → D070, FEAS entry list: AMENDED

Replaced by the FEAS entry checklist in §10. It now also covers:
- O4 v2, R2-LIVE v2, VT-13r, VT-22b v2, VT-23b, VT-24r v2, R6 v2;
- SUITE-PF, TC-PF, FEAS-SMOKE;
- R2e-X v2 with the D041 re-pricing;
- PR-27 v2 and near-miss logging;
- FEAS-PREC v2 and R-ATT;
- DTS10-AT v2 with technical blinding;
- the R8, R10 and RP-2 helpers.

---

## 4. G-RV items

### PR-14 · VT-21 → D071: route (i), RR-VT21

- **FAIL-as-registered.** 12 of 384 cases fail, all fp32 dw at T = 32,768.
  - Max abs 1.03–1.40e-4 against the absolute bound 1e-4.
  - Relative error 1.1–2.3e-7, i.e. 1–2 ulp.
  - Raw: `experiments/repro/R3/vt21/vt21_conv.r1.jsonl` and `.r3.jsonl`.
- **Why the bound cannot be met.** |dw| reaches 332–1,191. One fp32 ulp is 6.1e-5 on [512, 1024) and 1.2e-4 on [1024, 2048), so an absolute 1e-4 is 1–2 ulp, which no fp32 implementation meets reliably.
- **Independent confirmation.** The auditor re-ran with their own seeds and a float64 reference: dw 1.05–1.09e-4 (`audit_R2-R7/outputs/vt21_rerun.jsonl`).
- **Every other VT-21 criterion passes in all 384 cases.** Forward ≤ 7.2e-7, dx ≤ 5.5e-7, bf16 ≤ 1.9e-3, leakage 0, final state exact.
- **No new check.** A relative criterion chosen afterwards on the same data would be a restatement.
- **Standing lesson.** Absolute bounds on reductions must scale with the reduction's magnitude.

### PR-15 · VT-22 → D072: route (i), RR-VT22 + blocking VT-22b v2

**FAIL-as-registered.** C0 ↔ fla in bf16 gives relative RMS 2.021e-2 (16K) and 2.004e-2 (32K) against 2e-2, with band ratios 1.06 / 1.03.
- C12 passes at 6.4e-3 / 6.9e-3.
- 4K fp32 passes (C12 4.3e-6, C0 4.5e-4).
- **Diagnosis:** fla bf16 vs fla fp32 is 2.09e-2, ours bf16 vs fla bf16 is 2.02e-2, and ours fp32 vs fla fp32 is 2.6e-4. The registered bound sits at the reference's own bf16 drift. **RR-VT22 is accepted.**

**VT-22b v2** (`VT-22b.v2.yaml`) is blocking for G-RV. It has three parts:
- **C0 fp32 at 16K/32K:** F1 rel RMS ≤ 1e-3 and F2 band ratio ≤ 2.
- **C0 calibrated bf16:** B1 ≤ 1.5 × the larger self-drift, and B2 band ratio ≤ 2.
- **New: a C12 fp32 arm vs `Qwen3ForCausalLM` at 16K/32K** (F3 ≤ 1e-3, F4 band ratio ≤ 2), run through R-ATT. RoPE and position handling matter only for k > 0, and C12's bf16 floor (6.4e-3) cannot see a position-growing error below about 1e-2.

**Predictions stated before the run** (review O8):
- at 16K: F1 ≈ 2.6e-4, F2 ≈ 1.10, B1 cross ≈ 2.0e-2 vs a bound of ≈ 3.2e-2, B2 ≈ 1.06;
- at 32K: B1 and B2 similar.

**Weight at G-RV.** Only C0 F1/F2 at 32K and F3/F4 carry evidential weight. The others must still pass, since a failure would signal a change since R3, but their pass is reported as "as predicted". New weights are used (seed 1). A FAIL is a candidate E10.4 defect.

### PR-16 · VT-23 → D073: route (i), RR-VT23 + blocking VT-23b (v1, unchanged)

**FAIL-as-registered.**
- Step-1 loss 4.9e-8 passes.
- Steps 2–10 grad norm 3.0e-4 (> 1e-5) fails.
- Steps 11–50: 1.77 (> 1e-3) fails; the maximum is at step 19.
- Final parameters 0.1945 (> 1e-3) fail.

**The update rule is identical.**
- With the production loss and clip swapped in (D1), updates are bitwise identical at steps 1–4. Clipping was active in every one of those steps: pre-clip norms 2.79 / 2.76 / 2.29 / 1.80, confirmed by the reviewer.
- A change of summation order alone makes the reference diverge from **itself** by more (D3).
- WD groups are identical; clipping was exercised.

**Ruling.** The trajectory criteria are unattainable in this regime. RR-VT23 is accepted.

**VT-23b (unchanged; the review judged it sound)** compares one step from identical states at s ∈ {0, 1, 4, 8, 9, 10, 20, 35, 48}:
- loss and grad norm within 1e-5;
- per-tensor clipped gradient, update and v within 1e-4.

**Deviation acknowledged:** activation checkpointing on both sides, forced by the 0.44 cap.

### PR-17 · VT-24 → D074: route (ii), genuine generator gap → fix, regenerate before FEAS, VT-24r v2

**FAIL-as-registered.** 39,793 of 39,800 queries pass; there are 7 misses in 3 files and none in test/CS.

**Mechanism** (`diagnosis_misses.json`): fact-vs-fact value-prefix collisions. The plan pre-registered "any miss is an item defect", with regeneration as the consequence (plan §2.3 R3e; amendment part F). Arguing now that these are not defects would be a restatement.

**The gap is genuine.**
- Values are drawn uniformly, with no check that each queried value is determined within the copy window.
- Under T4 this confounds template with key-to-digit distance.

**Rule G-ORACLE.** For every queried fact and answer digit, the most recent earlier occurrence of the 6-token window before that digit must lie in the queried fact.

**Repair.**
- Resample only that fact's 4 value digits, from a dedicated substream keyed by (rng_key, context, fact, attempt).
- Prefix stability across DG / DC / FD is kept.
- **The repair never runs in namespace `train`.** This is an explicit check (review O8), not a reliance on collisions being rare.

**Timing:** before FEAS, which is stricter than "before G5", so that one item set serves the whole study.

**VT-24r v2** (`VT-24r.v2.yaml`) checks:
- **Baselines first:** the K3 manifests and the K5b TEF block hash are committed before the fix.
- **K0:** the independent checker reproduces VT-24's 7 misses (`pred_ok`) on the pre-fix items and reports coincidental matches.
- **K1:** VT-24's criteria at every length and in every directory, scored by `InductionLM` + the scorer, an implementation independent of the repair's checker.
- **K2:** the M-hop negative control.
- **K3:** content stability.
- **K4:** the full VT-12 suite.
- **K5a:** leak unit test: a planted collision in a `train` item is not repaired, while the same collision in a `dev` item is.
- **K5b:** full-block TEF hash, seed index 101, unchanged.
- **K6:** planted-collision unit test.

VT-24r v2 is blocking for G-RV, R6 and FEAS. The old item set's FAIL stays on record (manifests at `97ef269`).

### PR-18 · VT-25 → D075: the registered verdict PASS stands; pre-registered outcome tiers

The verdict run is the pre-registered one: worst gradient 1.82e-2 against 2e-2.

**Hygiene.** The global torch RNG is seeded immediately before the MQAR batch is built (Zoology `multiquery_ar.py` L137). A 10-seed sweep reports the minimum margin.

**Outcome tiers**, pre-registered now (SUITE-PF O-d) on max(pinned, sweep):

| Value | Consequence |
|---|---|
| ≤ 2e-2 | Consistent with the verdict run |
| 2e-2 < value ≤ 4e-2 | **RR-VT25** ("margin not robust across MQAR batches") entered by PI row. G-RP(b)'s precondition is reported as "met on the registered verdict run". No block |
| > 4e-2 (2× the bound; planted defects gave ≥ 8e-2) | Candidate regression: G-RV blocked until diagnosed |

The VT-25 verdict stands in every tier. This resolves the DRAFT's conflict between "VT-25 pass" in the G-RV table and "evidence only".

### PR-19 · R7 → D076: route (i), RR-R7, not blocking

- **FAIL-as-registered.** Pythia-160M 0.3538 is outside [0.313, 0.343]; Pythia-70M 0.2271 is outside [0.170, 0.200].
- **The scorer is not implicated.** Our scorer, lm-eval @`d6de8164` and the auditor's plain `transformers` scorer give identical counts in fp32.
- **Cause.** The anchors are fp16-inference numbers: fp16 weights give 0.3297 / 0.1849, against EleutherAI's 0.3284 / 0.1846.
- **R7's purposes are met:** EM_tf = lm-eval acc; the context/answer split matches; Wang's GDN passes (0.3435).
- **Not blocking.** Under G-RV's wording, only a failure that implicates the scorer blocks FEAS verdicts.
- **No "R7b at fp16":** its outcome is already known.
- **Carried forward.** Pythia collapses under bf16 (0.12 / 0.05) while Wang's GDN does not (0.3421), and the cause is unexplained beyond excluding a RoPE mechanism. So this residual may differ between attention and GDN layers, and thus across k. FEAS-PREC v2 Q3/Q4 carry it: compute precision, with a between-k rule that feeds the FEAS verdict.

### PR-20 · R2(a)/(b) → D077: FAIL-as-registered for the past (RR-R2); implement; blocking R2-LIVE v2

**FAIL-as-registered.** "(a) and (b) live before R4 starts" was not met. Unlocked project processes ran in 3 of 9 R4 windows and 6 of 10 R5 windows, and an unlocked S0 pytest ran at 12:23–12:33.

**RR-R2** is accepted for runs already made. Those were accuracy runs with a fixed data order, and every R4/R5 gate result comes from an event-free window.

**Implementation** (`R2_live.v2.yaml` L1–L3):
- one lock;
- every CUDA entry point checks the lock, or a holder-descendant token, before its first CUDA call; a refused start exits with code 78;
- pytest takes the lock through `tests/conftest.py`;
- infra blocks carry the PR-27 v2 classification, in `summary.json`, every terminal row, runner `attempt_end` rows (crashed attempts included) and evaluation `_meta.json`;
- gpumon records lock ancestry.

**R2-LIVE v2:**
- **C1:** tests, including the classifier table.
- **C2:** compliance over every Phase II GPU window, from O4-Z to SUITE-PF.
- **New (review O14):** a C2 violation **invalidates every gate or timed run in whose window it falls**; that run is repeated in full (plan §2.2(a)).
- G-RV stays blocked until C2 is clean over a later window that contains a complete gate run.
- At most one re-judgement is allowed; after that, a PI row.

G-RV reports this condition as "MET via new check", never as an R2 PASS.

### PR-21 · O4 soak → D078: O4 v2; R5 and the 12:35 G0 suite stand

**Order:** all three parts run before any gate or timed run and before the TC-PF warm (`O4_soak.v2.yaml`).

**O4-Z (new, report-only, with pre-registered meaning).** The R5 cell that hung twice (Zoology GDN d64 s125) runs alone, with **normal launches**, repeated from epoch 0 for 60 min. At the R5 rate (3.4/h), P(0 events) = 0.033.
- **E_Z ≥ 1** means "the hang persists on the exclusive GPU". Consequences, all before FEAS:
  - RR-Xid may be entered, but only by a PI row;
  - FEAS runs under PR-27 v2 with the per-k resume monitor (F-infra);
  - VT-13r is required (it already is);
  - the PI raises headless operation with the user.
  - E_Z ≥ 1 is not by itself a block.
- **E_Z = 0** is evidence that the R5-rate trigger was contention-related. The runbook stays in force regardless.

**O4-B (diagnostic).** The same cell with `CUDA_LAUNCH_BLOCKING=1` and stack dumps, ≤ 60 min. It runs immediately after O4-Z. The DRAFT's deferral of O4-B past the gate runs was an unlogged deviation from plan §2.11/§2.2, and it is withdrawn.

**O4-A (verdict, unchanged from v1).**
- A1/A3 are bitwise determinism probes.
- A2 is a 2.0 h production-shape soak in four 30-min segments.
- PASS requires zero events, no CUDA-error endings, bitwise probes and no OOM.
- **Stated limit:** O4-A passes with probability ≥ 0.5 at any production-shape rate ≤ 0.35/h. A PASS is evidence against a high rate, not evidence of absence.
- **FAIL:** no gate or production run until a PI row chooses among a configuration-only change, headless operation (needs the user's consent), or acceptance under PR-27 v2 with RR-Xid.

**R5 and the 12:35 suite stand**, on the evidence in the DRAFT:
- event-free windows;
- full repeats from epoch 0;
- exact re-evaluation of saved weights;
- a clean scan for the suite window, and deterministic numbers.

The sequencing breaches are RR-R5-D1 and RR-G0suite-seq.

**Reopen trigger:** O4 finds evidence of silent corruption. Then R5, the 12:35 suite and every run since 07:56:54 are re-examined.

### PR-22 · R5 deviation D1 → D079: ACCEPTED; RR-R5-power named

D1 was declared in the prereg before any R5 run. R5 = **PASS** counts as "R5 not FAIL" for G-RV:
- Δ −0.0175 against a threshold of 0.1154;
- phenomenon 0.296 < 0.925 < 0.957;
- spans 0.0875 / 0.1096, so the bimodality rule does not fire.

**RR-R5-power (new; review O11).**
- R5 excludes only a bf16 GDN *training* handicap larger than 0.115 (50% power) or 0.154 (80% power) on the 1024/256 cell.
- A smaller handicap would lower every k = 0 baseline and inflate k-contrasts in attention's favour.
- FEAS-PREC covers scoring precision, not training precision, so nothing before S4 covers this residual.
- It is named in the limitations and in every k-contrast report. No parity claim at the few-point level is made.

### PR-23 · R6 → D080: R6 v2, before R6 runs

The criteria are plan §2.6, unchanged. `R6.v2.yaml` changes one reading and adds one report.

**R6-R3(ii): the format-mismatch signature (review O4).**
- **New reading:** at a look, the mean **retrieval-family** answer CE (R1-type and R-load-type answer tokens) over the 50 steps before the look is **≤ 0.50 nat/token**, **and** the score is < 0.05.
- **Basis:** ln 16,384 = 9.70 nats untrained; the digit prior is ln 10 = 2.30; 0.50 nat/token means a geometric-mean p(gold digit) ≥ 0.61.
- **Why v1 failed:** its relative anchor (≤ 0.5 × the steps-1–50 mean) fires as soon as the model learns "answers are digits", which turns (ii) into "score < 0.05 at any look ⇒ FAIL".
- **Logging:** non-semantic, metrics-only, adds the family CE. The loss and the VT-17 hashes are unchanged.

**R6-R9 (new, report-only).** On the `--` (GDN) model: chain consistency, cache path vs single pass, over 6,400 queries, with pooled and restricted rates.
- A rate ≥ 1% fires the RR-G0-1/RR-G0-2 reopen: a PI row before FEAS, after VT-06/07 are re-run on that checkpoint.
- It never changes R6's verdict.

**Unchanged:** metric, looks, INCONCLUSIVE, infra repeats and code. **Preconditions:** O4 v2 complete, R2-LIVE v2, VT-24r v2 PASS, and the logging merged.

### PR-24 · R2(c), R2(d), R2(e), R4 → D081

- **R2(c):** PASS, recorded (longest kernel 25.24 ms against < 1 s).
- **R2(d):** recorded (two candidate causes, consistent with memory exhaustion by unlocked jobs).
- **R2(e):** R2e-X v2 (PR-04).
- **R4 / G-RP(b):** PASS (PR-05).

---

## 5. Pre-FEAS items

### PR-25 · DTS-10 FEAS decision code and U-15 → D082: DTS10-AT v2

`experiments/prereg/DTS10_acceptance.v2.yaml` keeps every v1 definition and rule verbatim, and adds four things.

1. **Precision rules first.** feas.py computes FEAS-PREC v2 Q1–Q4 before any FEAS rule. Q4 selects the score source (P0, or P4) for every ladder rule. U-15 and the TEF forgetting gate read P0. I-08 reads P0, and also P1 if a Q4 trigger fires; a difference goes to a PI row before S3.
2. **Verdict flags.** A flagged verdict is computed but not acted on until a PI row:
   - **F-nearmiss:** any FEAS run with a G-div near-miss note.
   - **F-infra** (per-k resume monitor): |n(0) − n(12)| ≥ 3 infra or suspect attempts over the FEAS runs, or any single run with ≥ 3. The PI's row reviews VT-13r and the loss continuity at each resume.
   - **F-precision:** Q4 selected P4. This is reported only.
3. **Readings stated as readings:**
   - I-08 "mixed floor/ceiling levels trigger" (the reading that triggers most easily);
   - elicitation "both seeds", literal.
4. **Technical blinding** (review O14):
   - While no `DTS10-commit` note exists, `scripts/evaluate.py` prints no score, EM, logp, rank or flip value for S2 checkpoints. It shows only cell names, counts, paths and timings.
   - Analysis tools refuse S2 trees.
   - A test enforces both.
   - The procedural rule still applies, because the files remain readable.

**Timing.**
- The code and its acceptance tests must be complete before FEAS-SMOKE.
- The `DTS10-commit` note must precede the first reading of any FEAS evaluation output; the TEF-1000 gates are the latest deadline.
- Because FEAS-SMOKE exercises `feas_decide`, the code must be complete before FEAS-SMOKE, and so before FEAS training. The DRAFT's allowance to start FEAS training before the code is complete no longer applies.

**U-15 consequence (unchanged).** CS contexts 301–600 are generated prefix-stably under the VT-24r repair rule. VT-12 and VT-24r K1 are re-run on the extension before G5.

**Also owed before the first FEAS Stage-A milestone is evaluated:** the R8 ICL helper, the R10 driver and the RP-2 LM-3 reading code.

### PR-26 · `rescore_check` reference precision → D083: FEAS-PREC v2 and R-ATT

**Unchanged.** The frozen §9.2 / I-29 check (P2: single-pass, same bf16 autocast, bf16-rounded weights) isolates path artefacts. It is Q2, and its threshold and consequence are those of the frozen check.

**Added** (`FEAS_precision.v2.yaml`):
- **P3:** fp32 single-pass reference on the 5% subset. → Q3.
- **P4:** fp32 cache-path reference on the FEAS ladder verdict cells, all contexts. It always runs. → Q4.
- **P1:** fp32 weights, full FD. → Q1.

**Q3 consequences before S4:**
- **(a)** fp32-compute scoring;
- **(b)** fp32 weights only, allowed only if P1 ≈ P3;
- **(c)** production scoring with a per-k caveat, allowed **only if the 32K contrast is unchanged within 0.02 under P4 and no ladder outcome differs**. The no-op exit is removed.

**Restricted per-length rates** decide Q1 and Q3, together with the pooled rates.

**Precedence with Q2.** If Q2 fails for a checkpoint, its single-pass bf16 re-scores replace P0 for that checkpoint, and Q4 compares those with P4.

**R-ATT** (§1.3) is required for P3, P4 and VT-22b's C12 arm.

`rescore_audit` keeps gating on the bf16 path flips. R6(a) is unchanged.

---

## 6. New decisions

### PR-27 · Infrastructure-failure status → D084 (v2)

**Classes**, written to every infra block (review O10):

| Class | Events | Treatment |
|---|---|---|
| **`infra_failed_attempt`** | Xid 8 with an RC-watchdog line (or a `cudaErrorLaunchTimeout` matched to one); Xid 79; host OOM-kill (kernel or systemd-oomd) | Never counts toward D031's "3 consecutive failures" or §8.1's "fails permanently". The runner resumes from the last checkpoint. After 10 infra attempts on one run, the queue **pauses** for the PI; a run is never auto-failed |
| **`suspect_sw_failure`** | Xid 13, 31, 43; CUDA illegal address, illegal instruction or misaligned address without an RC-watchdog line; any other Xid | Counts as an ordinary failed attempt under D031. One resume. **A recurrence at the same optimizer step stops the run** for PI diagnosis, as a deterministic fault |

- **Unchanged:** G-OOM (3 consecutive device OOMs → failed), and reproduction gate runs still repeat a flagged window in full.
- **Reporting:** counts per run, per class and per k; FEAS flag F-infra (PR-25).
- **Status:** a pre-study runbook change; no S2 run has happened.

### PR-28 · H3 GLMM sensitivity before G5 → D085

Unchanged from the DRAFT:
- in-repo, numpy/scipy only: a binomial-logit model with random intercepts for seed and context, fitted by Laplace approximation;
- a VT-18 extension must recover planted D₃ on the logit scale;
- the logit stand-in remains as an additional sensitivity.

### PR-29 · Process → D086

- **(a) Hashing.**
  - Batch 1: `SHA256SUMS`, 11 files, 13:44.
  - Batch 2: `SHA256SUMS.v2`, 14 files, final at 14:53.
  - Superseded batch-1 files never ran and stay unchanged.
  - Executors verify the hash before the first run and copy the file to `experiments/repro/<ID>/prereg.yaml`.
- **(b) Ledger.** `reports/reproduction/results.md` gets one row per item, in the §0 vocabulary.
- **(c) Errata.** Dated Errata sections, never silent edits:
  - `results_R3.md`: VT-21 |dw| 332–1,191; VT-23 loss maximum 1.57e-2 at step 24, spike at step 19;
  - `results_R4.md`: 3 of 9 windows contended;
  - `s0_integration.md` §8 and G0.md §6: the spliced hash and the contended warm;
  - G0.md and STATUS.md: the G0 verdict wording and the "447/447" count.
- **(d) Registered-criterion tests.** Tests for VT-03 (GDN fp32), VT-05, VT-07, VT-15, VT-21, VT-22 and VT-23 are marked `xfail(strict=False, reason="FAIL-as-registered; accepted residual RR-…; ruling D0xx; criterion unchanged")`.
  - XPASS behaves as in §0 rule 3.
  - The relaxed guards are labelled in their docstrings.
  - The VT-24 test is unmarked; on the regenerated items it is a subset of VT-24r K1.
- **(e) VT-13 standing replicate,** clarified before it runs: two audit-mode runs of C12 seed 0's first 1,000 steps, one at S4 start and one after the last S4/S5 training run. They must be bitwise identical in loss trace and parameter SHA-256, and are never compared with a production run.

### PR-30 · Cost-benchmark session → D087

The ruling is as in the DRAFT:
- it is not a gate run and may finish;
- G-RV-path jobs take precedence;
- costs are valid for the code hash they record;
- U-14 is resolved by the user's exclusive-GPU grant;
- infra events are reported at once as O4 evidence.

**Added:** `scripts/benchmark_inference.py` is committed before any of its numbers is cited (T03). The session was still running at 14:12 (pid 2013291).

### PR-31 · Verdict vocabulary → D088

This is §0 rule 2. It applies to G0 (PR-01), to G-RV (§9) and to every later gate and item report, including the ledger.

### PR-32 · Implementation order and Phase II entry → D089

The next step executes **§ Implementation tasks** in order.
- **Phase II entry:** Phase II (GPU) starts only after T00–T23 are complete and committed and `cost-session1` has released the lock.
- **Exception:** T19 (DTS-10) may finish during Phase II, but must be done before T35 (FEAS-SMOKE).
- **GPU discipline:** one GPU job at a time through the lock, memory fraction ≤ 0.44.
- **Budget:** about 7.5 GPU-h for Phase II, against the DRAFT's 5.5.
  - O4 v2 (≤ 4.4 h) is charged to the optional-items contingency (≤ 4.7 h).
  - The rest is charged to S0/S1 verification.
  - The D041 re-pricing row (after R2e-X) reconciles the totals.

---

## 7. Named residual-risk register (FINAL; decision D090)

| ID | Item | What is accepted | Differs across k? | Evidence | Safeguard | Reopen if |
|---|---|---|---|---|---|---|
| RR-G0-1 | VT-05 | fla bf16 chunk final state up to 8.47e-3 (> 8e-3) in 2/32 seed-cases | **Yes**: GDN layers only (12 / 9 / 0 at k = 0 / 3 / 12) | G0.md §2; `s0_integration.md` §12 | VT-06/07; FEAS-PREC v2 Q2–Q4 (restricted, between-k); R6 v2 `--` chain report | restricted flip(P0,P2) or flip(P0,P3) ≥ 1% at 16K or 32K for any k; Q4 switch; R6 `--` chain flip ≥ 1% |
| RR-G0-2 | VT-07 | branched vs single-pass 2.4–3.8e-3 nats (fp32), EM identical | **Yes**: GDN path only; small | `s0_integration.md` §12 | as RR-G0-1 | as RR-G0-1 |
| RR-G0-3 | VT-13 production | attempt 1 0.0075 > 0.005 | Exposure: hang-driven resumes concentrate in GDN workloads | `runs/_logs/vt13_production*.log`; `experiments/s0/vt13_production/result.json` | S0 bitwise audit resume; **VT-13r** on the final tree; F-infra; VT-13p2 v2 for production-only code | VT-13r FAIL; resume code changes without a VT-13r re-run |
| RR-G0-4 | VT-15 | cuDNN B = 1 hot decode −30% (faster); 2 cells unasserted | Measured costs only (attention decode); accuracy no | G0.md §5 | cold-L2 primary cells pass; registered 11-cell test kept (non-strict xfail) | an idle cold-L2 cell fails |
| RR-G0-5 | VT-03 | GDN fp32 packed vs alone 9.4e-5 against max abs 0.0 | No: fp32 only; production bf16 exactly 0.0 | `s0_integration.md` §12 item 1 | same-shape bit-identity; bf16 exactness asserted | a bf16 packed-vs-alone difference ≠ 0 |
| RR-G0-6 | G0 throughput | D512 Stage A 67.2K < 70K (pass 1); 0.6B fallback waived (DEV-03) | No (study-wide) | `experiments/s0/g0/g0_results.jsonl` | R2e-X v2 waiver-lapse guard | an exclusive D512 Stage-A median < 70K → fallback applies |
| RR-VT21 | VT-21 | fp32 dw absolute bound (1–2 ulp at \|dw\| ~10³) | Exposure (conv in GDN and F_gc) at 1–2 ulp | R3 raw; audit `vt21_rerun.jsonl` | every other VT-21 criterion passes | — |
| RR-VT22 | VT-22 | C0 bf16 cross-implementation 2.02e-2 > 2e-2 | **Yes** (C0 path) | `vt22_transplant.jsonl`, `vt22_diagnosis.jsonl` | **VT-22b v2** (blocking; C0 32K fp32 and C12 fp32 informative) | VT-22b FAIL |
| RR-VT23 | VT-23 | trajectory criteria unattainable (chaotic rounding growth) | No (shared trainer) | `vt23_compare.r1.jsonl`, `vt23_diagnosis.json` | **VT-23b** (blocking) | VT-23b FAIL |
| RR-R7 | R7 | Pythia fp32 anchors outside ±0.015; unexplained bf16 collapse of attention LMs | **Potentially** (attention vs GDN) | `experiments/repro/R7/`; audit `lambada_cpu.log`; `results_R7.md` L96–105 | 3 scorers agree; Wang PASS; FEAS-PREC v2 Q3/Q4 | scorer discrepancy; Q3 rule fires; Q4 switch |
| RR-R2 | R2(a)/(b) | not live before R4; contended R4/R5 windows | No | audit §1.1, §1.7 | **R2-LIVE v2** (blocking) | — |
| RR-R5-D1 | R5 | ran after the O4 trigger without O4 | No | R5 prereg D1; audit §1.7 | event-free windows; full repeats | O4 finds silent corruption |
| **RR-R5-power** (new) | R5 | only a bf16 GDN training handicap > 0.115 (50% power) / 0.154 (80%) is excluded | **Yes**: a smaller handicap lowers k = 0 baselines and inflates k-contrasts in attention's favour | `results_R5.md`; PR-22 | none before S4 (FEAS-PREC covers scoring only); named in limitations and in every k-contrast report | evidence of a bf16 GDN training handicap (e.g. a Tier-C fp32-trained k = 0 control, if run) |
| **RR-D052i** (new) | D052(i) | evaluated weights bf16-rounded (`A_log`/`dt_bias` relative 3.5e-3) | **Yes** (GDN decay parameters; more at low k) | `s0_integration.md` §12 item 10 | FEAS-PREC v2 Q1 (restricted per length) and Q4 | Q1 rule fires |
| RR-G0suite-seq | G0 suite 12:35 | ran after the O4 trigger without O4 | No | G0.md §1 scan | event-free; deterministic; SUITE-PF re-run | O4 finds silent corruption |
| RR-prereg | process | R3 prereg modified after some verdict runs; no R2/R7 prereg files | No | audit §2 | every reading literal or stricter; hashing (D086) | — |
| *RR-VT25* (conditional) | VT-25 | pinned or sweep margin in (2e-2, 4e-2] | No (shared harness) | SUITE-PF O-d | tiers of PR-18 | > 4e-2 → blocking |
| *RR-Xid* (conditional) | environment | GDN-workload hangs persist on the exclusive GPU | Exposure (GDN workloads) | O4-Z, O4-A | PR-27 v2; F-infra; VT-13r | O4-A FAIL; F-infra flag |

The conditional residuals exist only if their trigger fires and a PI row enters them. The residuals that differ across k are:
- by construction or potentially: RR-G0-1, RR-G0-2, RR-VT22, RR-R7, RR-R5-power and RR-D052i;
- by exposure: RR-G0-3, RR-VT21 and RR-Xid.

Every FEAS and Tier-A report names them in its limitations section.

---

## 8. Pre-registrations

### 8.1 Batch 1 (written 13:38–13:44; `experiments/prereg/SHA256SUMS`; all OK on re-check)

| File | ID | Status |
|---|---|---|
| `O4_soak.yaml` | O4 v1 | superseded by v2 (never ran) |
| `R2_live.yaml` | R2-LIVE v1 | superseded by v2 (never ran) |
| `R2e_G0T_exclusive.yaml` | R2e-X v1 | superseded by v2 (never ran) |
| `VT-22b.yaml` | VT-22b v1 | superseded by v2 (never ran) |
| `VT-23b.yaml` | VT-23b v1 | **in force, unchanged** (`91753a3cbd41ff0a9f7ed56d10026b4c26d15217f65a1421bb11707e5d2fab71`) |
| `VT-24r.yaml` | VT-24r v1 | superseded by v2 (never ran) |
| `R6.yaml` | R6 v1 | superseded by v2 (never ran) |
| `FEAS_precision.yaml` | FEAS-PREC v1 | superseded by v2 (never ran) |
| `DTS10_acceptance.yaml` | DTS10-AT v1 | superseded by v2 (never ran; v1 rules kept verbatim inside v2) |
| `VT-13p2.yaml` | VT-13p2 v1 | trigger superseded by v2 (never ran) |
| `README.md` | rules 1–5, index | rules in force; index superseded by `README.v2.md` |

### 8.2 Batch 2 (written 14:33–14:53, hashed 14:53:23; `experiments/prereg/SHA256SUMS.v2`), before any item ran

`SHA256SUMS.v2` was first written at 14:41:38 and rewritten at 14:53:23. In between, three files were edited: Q2/Q4 precedence in `FEAS_precision.v2.yaml`, the same rule mirrored in `DTS10_acceptance.v2.yaml`, and the index wording in `VT-13r.yaml`. All three edits came before any item ran and before any commit. The hashes below are final.

| File | ID | SHA-256 | Can it fail? |
|---|---|---|---|
| `O4_soak.v2.yaml` | O4 v2 | `61b201063932feee68d796186bc80290b18cdb319391701c40262e89f1c9aa3b` | O4-A: yes, but weak against rates ≤ 0.35/h (stated). O4-Z: report-only, with pre-registered meanings. O4-B: descriptive |
| `R2_live.v2.yaml` | R2-LIVE v2 | `14ce0caa8f6cd9e9149d0302be250265b4f96b0d7b8437e998c0f2f73a803794` | Yes. C1 can fail. A C2 violation invalidates runs; only one re-judgement is allowed |
| `R2e_G0T_exclusive.v2.yaml` | R2e-X v2 | `24f8cda8235a6756f81ef8199c86b61c84830d6a2883ff5e93957cd7b4f61a96` | Budgeting measurement. The waiver-lapse guard is practically certain not to fire and carries no weight (stated). The D384 regression flag and the D029 reading can fire |
| `VT-22b.v2.yaml` | VT-22b v2 | `af6a092f39970b423e03502d0aae97124fe05b9cc186844a075136f66129a9a3` | Yes. C0 F1/F2 at 32K and C12 F3/F4 are informative; the rest is predicted, but must pass |
| `VT-24r.v2.yaml` | VT-24r v2 | `97a46f68b0cc5a397ff2bc426be367a2976f351a0f45e23869c1363ab0bb1869` | Yes. K0 (checker vs scorer), K1 (now on independent code), K3, K4, K5a, K5b, K6 |
| `R6.v2.yaml` | R6 v2 | `4b1804ebe8721b4114c96de15e70eed6c09bf64c2c9a4e55437e2cb360c44a85` | Yes. (i) at 150M; (ii) only on a genuine format-mismatch signature |
| `FEAS_precision.v2.yaml` | FEAS-PREC v2 | `28f17b379d089ad9740ed1a46b2a9d3a4b7f55a93aa6054992cc48e6970e0a04` | Yes. Q1/Q3 consequences before S4; Q2 per checkpoint; Q4 switches the verdict source |
| `DTS10_acceptance.v2.yaml` | DTS10-AT v2 | `1dc0e1ff53450e9005430b1ea113dabac94aa97321c4d1fe1e449b4812e1be8b` | Yes (acceptance tests) |
| `VT-13r.yaml` | VT-13r | `7145a7b5f3d8024241fbfcf2ba1bb94b27a865d1dad42ee7686d6caf924c643b` | Yes (bitwise) |
| `VT-13p2.v2.yaml` | VT-13p2 v2 | `47e2ba76fe75a13b40004fa3eba58add250711436bd2a5bf5fed9b066d8df678` | Yes, when triggered (future only) |
| `G0suite_preFEAS.yaml` | SUITE-PF | `8a6ada40888b4ac17992f84f7a15bdff522a8c1b6f100f0284836d6edced3030` | Yes. O-a regression; O-d > 4e-2. An XPASS is never a pass |
| `TRITON_prefeas.yaml` | TC-PF | `aa30ae48e1939b1313384d244d0deec28aa3a711ccf29f52c67f45139380a671` | Yes. W1/W2 environment; W3 coverage triggers a re-warm |
| `FEAS_smoke.yaml` | FEAS-SMOKE | `1799c5db88b092788b3564bc21376dd8e144aad599c0d1aaf6d6ef1c807b1ac1` | Yes. Integration only; it may be re-run after fixes, with every attempt recorded |
| `README.v2.md` | rules 6–7, index | `c696b7ebb4da6588c0e5a850942d2774247f5562e27d0c097b0062e6a03b8523` | — |

---

## 9. G-RV gate: conditions and reporting (replaces the DRAFT's §8)

The independent G-RV gate reports in the §0 vocabulary. Given what has already run, **G-RV cannot read PASSED**. Its verdict line will be "NOT PASSED as registered". The gate then states separately whether FEAS entry is PERMITTED under G-RV's named-residual clause or BLOCKED.

| G-RV condition (amendment §13.3) | Registered status now | Route to FEAS entry | Required evidence |
|---|---|---|---|
| G0 passed | NOT MET (PR-01) | → RR-G0-1…6, DEV-02 | Rows D058–D061 entered; SUITE-PF O-a holds (else blocking) |
| G-RP (a), (b) passed | MET | — | D062 |
| VT-21 pass | NOT MET | → RR-VT21 | D071 |
| VT-22 pass | NOT MET | → RR-VT22 **and** VT-22b v2 PASS | D072; `experiments/repro/VT-22b/` |
| VT-23 pass | NOT MET | → RR-VT23 **and** VT-23b PASS | D073; `experiments/repro/VT-23b/` |
| VT-24 pass | NOT MET | MET via new check: **VT-24r v2 PASS** (K0–K6), items regenerated, manifests committed | D074 |
| VT-25 pass | MET (registered verdict run) | SUITE-PF O-d ≤ 4e-2 (else blocking); RR-VT25 row if in (2e-2, 4e-2] | D075 |
| R5 not FAIL | MET | (RR-R5-D1, RR-R5-power named) | D079 |
| R6 not FAIL | pending | R6 v2 PASS or INCONCLUSIVE (FAIL → blocking) | D080 |
| R2 (a), (b) live | NOT MET as registered (RR-R2) | MET via new check: **R2-LIVE v2 PASS** (C1 + C2) | D077 |
| R2(c) recorded | MET (25.24 ms) | — | D081 |
| R7 complete (scorer not implicated) | MET as worded; R7 Pythia criteria FAIL → RR-R7 | — | D076 |
| O4 (plan §2.2) | pending | **O4-A PASS**; O4-Z and O4-B reported; their meanings applied | D078 |
| Resume re-verification (RR-G0-3 trigger fired) | pending | **VT-13r PASS** | D060 |

FEAS entry is **PERMITTED** only if no row ends in "blocking" and every RR named in the table has its decision row. Any FAIL of a new check keeps FEAS blocked. The PI may accept a further residual only by a new row, under the same rule.

## 10. FEAS entry checklist (replaces D057)

FEAS-384 training may start only when all of these hold:

1. The G-RV gate report says "FEAS entry PERMITTED" (§9).
2. Rows D058–D091 are entered, and every conditional residual that fired (RR-VT25, RR-Xid) has its row.
3. TC-PF is done:
   - fresh exclusive warm, archived;
   - full hashes recorded (`pre-FEAS-triton-cache`);
   - W3 coverage holds (0 new files in the FEAS-SMOKE and R2e-X extractions).
4. R2e-X v2 is done:
   - G-throughput references are set;
   - the waiver-lapse guard is evaluated;
   - the D384 regression flag and the D029 reading are recorded;
   - **D041 is re-priced by decision row** (including FEAS-PREC P1–P4 from the FEAS-SMOKE timings).
5. FEAS-SMOKE has PASSED on the commit that FEAS will run.
6. PR-27 v2 classes, the per-k counts, G-div near-miss logging and R2-LIVE v2 are live in the runner and trainer.
7. FEAS-PREC v2 passes P0–P4 and R-ATT are implemented and tested (T14, T15).
8. DTS10-AT v2 passes. Its technical blinding is live. The `DTS10-commit` note precedes the first reading of any FEAS evaluation output.
9. The R8, R10 and RP-2 LM-3 helpers are ready before the first FEAS Stage-A milestone is evaluated.
10. The FEAS run YAMLs are reviewed against the protocol and committed:
    - k ∈ {0, 12}, seeds 101 and 102;
    - η 2.0e-3 with one restart at 1.4e-3 (D034);
    - Stage-B branches 30M and 90M;
    - SSMax arm (k = 12, seed 101);
    - TEF 1,000 with `answer_norm: block` (D063);
    - the milestones, and production precision.
11. The tree is clean and committed. It includes the regenerated manifests, `tests/conftest.py`, both prereg batches, the fla wheel hash in provenance and `scripts/benchmark_inference.py`.
12. The ledger `reports/reproduction/results.md` is current and uses the §0 vocabulary.

---

## Implementation tasks

These are ordered. Phase 0 and Phase I run before any Phase II GPU job (D089). Owners are named by role; the files listed are the ones each task may create or change. Every acceptance test must pass, and its log must be saved, before the task counts as done. One GPU job at a time, through the lock, at memory fraction ≤ 0.44.

### Phase 0: records (orchestrator; now)

**T00. Commit the FINAL rulings and pre-registrations.**
- Owner: orchestrator.
- Files: `reports/pi_rulings_2026-11-10.md`, `reports/decisions.md`, `reports/reviews/review_pi_rulings_2026-11-10.md`, `reports/reviews/pi_rulings_2026-11-10.DRAFT.md`, `experiments/prereg/*` (both batches, both SHA256SUMS files, both READMEs), `experiments/repro/O4/pi_klog_scan_20261110_0940_1331.json`.
- Acceptance:
  - `cd experiments/prereg && sha256sum -c SHA256SUMS && sha256sum -c SHA256SUMS.v2` prints 25 × OK;
  - the commit time precedes every Phase II run;
  - `git status --short experiments/prereg reports` is empty.

**T01. Errata and record fixes.** Dated "Errata (2026-11-10, PI ruling D0xx)" sections; never silent edits.
- Owner: orchestrator (STATUS), gate reporter (G0.md), integrator (`s0_integration.md`), R3 and R4 executors (results files).
- Files: `reports/s0_integration.md` §8; `reports/gates/G0.md` §6 and verdict block; `reports/reproduction/results_R3.md`; `reports/reproduction/results_R4.md`; `reports/STATUS.md`.
- Content:
  - the full post-warm and pre-warm hashes, the splice error, and the fact that the S0 warm ran under contention;
  - the G0 verdict reworded to "NOT PASSED as registered; FEAS only under G-RV's named-residual clause";
  - "447 passed / 4 skipped; 13 relaxed-guard test cases";
  - the R3 and R4 errata of PR-29(c).
- Acceptance:
  - outside `reports/reviews/`, this file and `reports/decisions.md`, `grep -rn "f74316" reports/` matches only inside erratum text that names the splice;
  - with the same exclusions, `grep -rnE "447/447|447 of 447" reports/` matches only inside erratum text;
  - each changed file has its dated Errata section.

**T02. Ledger.**
- Owner: orchestrator.
- File: `reports/reproduction/results.md`.
- Content: one row per item (R1–R7, VT-21…25, VT-22b, VT-23b, VT-24r, R2-LIVE, O4-Z/B/A, R2e-X, VT-13r, SUITE-PF, TC-PF, FEAS-SMOKE). Each row has: claim | registered criterion | registered verdict | route (RR / new check) | raw paths. The residual register of §7 is appended, with RR-R5-power and RR-D052i in the limitations list.
- Acceptance:
  - every FAIL-as-registered item reads "FAIL (as registered)" and names its RR;
  - no row reads PASS for an item whose registered verdict was FAIL;
  - pending items read "not run".

**T03. Commit the cost-benchmark driver.**
- Owner: orchestrator.
- File: `scripts/benchmark_inference.py`.
- Acceptance:
  - `git ls-files scripts/benchmark_inference.py` is non-empty;
  - the blob hash equals the file the session ran, which `experiments/bench/cost_session1/session.jsonl` or the session report records; otherwise the session's numbers are flagged "code unverified" in its report.

### Phase I: code and CPU work

Owners may work in parallel. Changes merge in this order. No GPU, except the C1 GPU cases of T10 and the GPU-marked tests named here, which run under the lock.

**T10. Single GPU lock and infra module** (R2-LIVE v2 L1–L3).
- Owner: S0 training/runner, eval, tests.
- Files: `src/hybridlc/utils/gpu_lock.py` (new); `src/hybridlc/utils/infra.py` (new, with the PR-27 v2 classifier); `scripts/repro/gpu_lock.py` and `scripts/repro/infra_events.py` (thin CLIs); entry-point guards in `scripts/{train,runner,evaluate,g0_throughput,bench_train_step,benchmark_inference,warm_triton,vt13_production_resume}.py`, `scripts/repro/*.py` and `scripts/s1_zoology/*.py`; `tests/conftest.py` (new); `src/hybridlc/training/{trainer,gpumon,registry,queue}.py` (infra blocks, `attempt_end`, `lock_ok`); `src/hybridlc/integration/evaluate.py`.
- Acceptance: `tests/test_r2_live.py` passes every R2-LIVE v2 C1 case, including the classifier table, and `scripts/repro/test_r2_tooling.py` stays green. Log: `experiments/repro/R2-LIVE/c1_pytest.log`.

**T11. Infrastructure status v2** (PR-27 v2).
- Owner: S0 training/runner.
- Files: `src/hybridlc/training/queue.py`, `scripts/runner.py`, `src/hybridlc/training/registry.py` (statuses `infra_failed_attempt`, `suspect_sw_failure`; pause state), `src/hybridlc/utils/infra.py`.
- Acceptance: `tests/test_pr27_infra_status.py` (CPU, fakes) shows that:
  - each event class maps as in PR-27;
  - infra attempts are not counted by D031's rule;
  - the queue pauses after 10 infra attempts on one run;
  - a `suspect_sw_failure` counts as a failure;
  - a second suspect at the same optimizer step stops the run with a registry row;
  - per-k counts appear in the runner summary;
  - G-OOM behaviour is unchanged (existing tests green).

**T12. Triton cache separation** (PR-12 part 1).
- Owner: S0 training/runner, tests.
- Files: `tests/conftest.py`, `src/hybridlc/training/trainer.py` (S2–S8 guard: production dir, or an extraction whose hash equals the recorded pre-FEAS hash), `scripts/repro/make_diag_cache.sh` (new: extract the S0 archive after verifying `f417e487…96e3de`).
- Acceptance: `tests/test_triton_cache_guard.py` shows that:
  - an S2 run with the diagnostics cache exits 77;
  - an S2 run with the production dir is allowed;
  - an S2 run with a matching extraction is allowed;
  - a pytest session reports `TRITON_CACHE_DIR=.cache/triton-diag`.

**T13. G-ORACLE generator fix, regeneration and VT-24r v2** (CPU).
- Owner: eval, data, scripts.
- Files: `src/hybridlc/eval/oracle_check.py` (new; it must not import `eval.devtools`, `eval.scorer` or any model), `src/hybridlc/eval/{generators,context}.py` (repair substream; namespace check), `tests/test_vt12_g_oracle.py` (K6, K5a), `scripts/make_items.py`, `experiments/items/**/MANIFEST.json`, `experiments/items/vt24r_repairs.json`, `experiments/repro/VT-24r/*`.
- Order:
  1. **Baselines first:** `k5b_baseline.json` and K3 baselines are computed and committed on the pre-fix tree.
  2. The fix, then K6 and K5a.
  3. Regeneration.
  4. K0–K4 and K5b.
- Acceptance:
  - `experiments/repro/VT-24r/vt24r_summary.json` has verdict PASS for K0–K6;
  - an import-graph test shows `oracle_check` imports none of the forbidden modules;
  - the VT-12 suite is green;
  - the new manifests are committed.

**T14. fp32 reference attention path** (R-ATT).
- Owner: models.
- Files: `src/hybridlc/models/attention.py` (an explicitly selected fp32 reference mode for the full-sequence forward and for inference prefill / extend / decode; default paths unchanged); `tests/test_fp32_ref_attention.py` (new); `tests/test_vt08_scoring_path.py` (extended).
- Acceptance:
  - relative RMS ≤ 1e-5 vs MATH at T ∈ {512, 4096}, for prefill, lower-right extend and decode, for F_g and F_gc, with cu_seqlens isolation;
  - GPU, slow-marked, under the lock: peak ≤ 7 GiB for D384 C12 single-pass at T = 32,768 and for cache-path prefill + extend at 32K;
  - the bf16 production path still pins FLASH / cuDNN;
  - the reference mode raises if its backend cannot run.

**T15. FEAS-PREC v2 scoring passes.**
- Owner: eval, integration.
- Files: `scripts/evaluate.py`, `src/hybridlc/integration/evaluate.py`, `src/hybridlc/eval/scorer.py` (`rescore_check(ref="bf16"|"fp32")`; `--reference fp32-cache`; `--only` cell lists; P1, P3 and P4 without candidates; `_meta` precision fields); `tests/test_feas_precision_passes.py` (new).
- Acceptance:
  - on a small D384 checkpoint (`experiments/s0/smoke`), P0–P4 write result files whose `_meta` records weights dtype, autocast flag, TF32 flags, attention backend and pass id;
  - at T ≤ 4K, P4 and P3 agree on EM for every query, with |Δlogp| ≤ 2e-2 (the VT-06 fp32 logit scale);
  - `rescore_audit` still gates on bf16 path flips.

**T16. Per-family answer-CE logging** (R6 v2).
- Owner: data, S0 training.
- Files: `src/hybridlc/data/tef.py` (per-token answer-family codes), `src/hybridlc/training/trainer.py` (metrics-only `answer_ce_retrieval`); `tests/test_tef_family_ce.py` (new).
- Acceptance:
  - the logged value equals a hand computation on a fakes batch;
  - the loss is bitwise identical with logging on and off (CPU fakes);
  - VT-17 batch hashes for seed indices 0 and 1, steps 0–2, are unchanged.

**T17. G-div near-miss logging** (PR-10).
- Owner: S0 training.
- Files: `src/hybridlc/training/gdiv.py`, `src/hybridlc/training/trainer.py`; `tests/test_gdiv_near_miss.py` (new).
- Acceptance:
  - a planted +1.2-nat excursion for 60 steps writes one `gdiv_near_miss` note and no restart;
  - +1.0 nat for 40 steps writes no note;
  - existing G-div tests are green.

**T18. Registered-criterion markers and the VT-25 pin** (SUITE-PF).
- Owner: tests.
- Files:
  - `tests/test_vt03_doc_isolation.py`, `tests/test_vt05_recurrent_chunk.py`, `tests/test_vt07_branched.py`, `tests/test_vt15_bench_harness.py`: registered-criterion tests with `xfail(strict=False)`, and docstrings on the relaxed guards;
  - `tests/test_vt21_conv_reference.py`, `tests/test_vt22_long_transplant.py`, `tests/test_vt23_reference_loop.py`: converted to `xfail(strict=False)` with RR reasons;
  - `tests/test_vt25_zoology_transplant.py` (global RNG seeded before the batch);
  - `scripts/repro/vt25_seed_sweep.py` (new);
  - an outcome hook in `tests/conftest.py`.
- Acceptance:
  - `pytest --collect-only -q` lists the 7 registered-criterion tests with xfail markers whose reasons name the RR IDs;
  - `tests/test_suite_outcome_hook.py` shows that an XPASS is written to the outcome file and does not fail the session;
  - a script lists exactly the 13 relaxed-guard test cases.

**T19. DTS-10 decision code** (DTS10-AT v2).
- Owner: analysis.
- Files: `src/hybridlc/analysis/feas.py`, `scripts/feas_decide.py`, `tests/test_vt18_feas.py`.
- Due: before T35. This task may finish during Phase II.
- Acceptance: every DTS10-AT v2 acceptance test passes, including the precision boundaries, the flags and the CLI runs on the S0 smoke tree and on a synthetic full FEAS tree with P0–P4.

**T20. Technical blinding.**
- Owner: eval, analysis.
- Files: `scripts/evaluate.py`, `src/hybridlc/integration/evaluate.py`, `scripts/analyse_results.py`; `tests/test_dts10_blinding.py` (new).
- Acceptance: with stdout and stderr captured, while no `DTS10-commit` note exists:
  - an S2 checkpoint evaluation prints no score, EM, logp, rank or flip value;
  - `analyse_results.py` refuses S2 trees (non-zero exit);
  - a non-S2 control prints normally;
  - after a fake note, S2 prints normally.

**T21. FEAS run YAMLs.**
- Owner: configs.
- Files: `configs/experiments/feas/*.yaml`, generated by `python -m hybridlc.training.genconfigs`; `tests/test_feas_yamls.py` (new).
- Acceptance:
  - `genconfigs --check` passes;
  - the test asserts every §10 item-10 value: k, seeds, η and restart, branches, SSMax arm, TEF 1,000 with block normalisation, milestones, precision;
  - the PI or gate reviewer signs off in the ledger.

**T22. fla wheel provenance** (PR-11).
- Owner: utils.
- Files: `src/hybridlc/utils/provenance.py`; `tests/test_provenance_fla.py` (new).
- Acceptance: `env_info()["fla_install"]` holds the SHA-256 of the installed `RECORD` file and of the dist-info directory listing, and the test recomputes both.

**T23. Phase I closure.**
- Owner: orchestrator.
- Acceptance:
  - `pytest -m "not gpu" -q` passes;
  - the GPU-marked tests of T10, T14 and T18 pass under the lock;
  - the tree is clean and committed, and its commit is recorded in the ledger as the "pre-FEAS tree";
  - `cost-session1` has released the lock.

### Phase II: GPU, strictly sequential

**T30. Cache preparation** (CPU).
- Owner: orchestrator.
- Steps:
  - move `.cache/triton` to `.cache/triton-pre-restore-20261110/` (tar, SHA-256, copy to `~/outeropt_artifacts/`);
  - create `.cache/triton-diag` with `scripts/repro/make_diag_cache.sh`.
- Acceptance: both directories exist; the S0 archive SHA is verified; nothing is deleted.

**T31. O4-Z** (60 min).
- Owner: orchestrator or repro executor.
- Files: `experiments/repro/O4/O4-Z/`, `o4z_result.json`.
- Acceptance: exposure ≥ 60 min of locked GPU time; E_Z and stacks recorded; the pre-registered meaning applied and logged in the ledger.

**T32. O4-B** (≤ 60 min, `CUDA_LAUNCH_BLOCKING=1`).
- Files: `experiments/repro/O4/O4-B/`.
- Acceptance: descriptive record written (event times, stack, last epoch).

**T33. O4-A** (≈ 2.4 h).
- Files: `experiments/repro/O4/{A1,A2,A3}/`, `o4_result.json`.
- Acceptance: verdict PASS (P1–P4). **On FAIL, stop Phase II and wait for a PI row.**

**T34. TC-PF fresh exclusive warm** (≈ 26 min).
- Files: `runs/_artifacts/triton_cache_prefeas.tar{,.sha256}`, `experiments/repro/TC-PF/`, the registry note `pre-FEAS-triton-cache`, a durable copy.
- Acceptance: W1 and W2 hold; full hashes recorded; the autotune-difference report written. W3 is checked in T35 and T36.

**T35. FEAS-SMOKE** (≤ 0.2 h; needs T19–T21).
- Files: `experiments/feas_smoke/`.
- Acceptance:
  - every pass item of `FEAS_smoke.yaml` holds;
  - 0 new Triton files in the extraction;
  - timings recorded for the D041 re-pricing.
- On a defect: fix, commit, re-run. If the fix touches code exercised by T31–T34, re-run those items too.

**T36. R2e-X v2** (≈ 0.6 h), then the D041 re-pricing row.
- Files: `experiments/repro/R2e-X/`; a PI decision row.
- Acceptance:
  - `g0t_results.jsonl` and `summary.json` written;
  - 0 new production-mode files in the extraction;
  - waiver-lapse guard, D384 regression flag and D029 reading recorded;
  - the PI enters the D041 re-pricing row.

**T37. VT-13r** (≈ 0.55 h).
- Files: `experiments/repro/VT-13r/`.
- Acceptance: R1–R3 hold (`vt13r_result.json` verdict PASS).

**T38. VT-22b v2** (≈ 0.15 h).
- Files: `experiments/repro/VT-22b/`.
- Acceptance: F1–F4, B1 and B2 hold at both T; measured values are reported beside the predictions.

**T39. VT-23b** (≈ 0.2 h).
- Files: `experiments/repro/VT-23b/`.
- Acceptance: C1–C5 hold at every tested s; the regime check is satisfied.

**T40. R6 v2** (0.3–0.6 h).
- Files: `experiments/repro/R6/`.
- Acceptance: `r6_results.json` gives PASS or INCONCLUSIVE; the `--` chain-consistency report is written; a `--` rate ≥ 1% is escalated to a PI row.

**T41. SUITE-PF** (≈ 0.35 h).
- Files: `runs/_logs/suite_prefeas.*`, `experiments/repro/SUITE-PF/summary.json`.
- Acceptance:
  - O-a holds;
  - O-b and O-c are recorded;
  - O-d tier applied (an RR-VT25 row if in (2e-2, 4e-2]);
  - counts reported in the SUITE-PF format.

**T42. R2-LIVE C2 evaluation** (CPU).
- Files: `experiments/repro/R2-LIVE/c2_compliance.json`.
- Acceptance: zero violations over T31–T41. Otherwise, apply the v2 invalidation rule and repeat the affected runs.

**T43. Ledger and register update** (CPU).
- Owner: orchestrator, PI.
- Files: `reports/reproduction/results.md`, `reports/decisions.md` (rows for any conditional residual that fired), `reports/competing_explanations.md` (dated addendum naming RR-R5-power and RR-D052i).
- Acceptance: every Phase II item has a ledger row in the §0 vocabulary.

**T44. Independent G-RV gate** (CPU).
- Owner: independent gate reviewer.
- File: `reports/gates/G-RV.md`.
- Acceptance:
  - the report follows §9, verdict line first;
  - FEAS entry is PERMITTED only if no condition ends in "blocking";
  - every number in it is recomputed from raw files.

FEAS training starts only after T44 says PERMITTED and the §10 checklist holds.

---

## 12. Decision rows entered in this step

- **Status marks.** D045–D057 are marked in place. The PROPOSED text is kept verbatim after the mark.
  - **ACCEPTED:** D049, D051, D053, D055.
  - **AMENDED:** D045, D046, D047, D050 (as DEV-04), D052, D054, D056, D057.
  - **REJECTED:** D048.
- **New rows:**
  - D058–D070 are the FINAL versions of PR-01…PR-13, which replace D045–D057;
  - D071–D087 are PR-14…PR-30;
  - D088 is PR-31 (vocabulary) and D089 is PR-32 (implementation order and Phase II entry);
  - D090 is the residual register, and D091 is the batch-2 pre-registration index.
- **Logged deviations:**
  - DEV-02: G0 failure clause not applied (D058);
  - DEV-03: 0.6B fallback waived (D061);
  - DEV-04: §7.3 "in the step" (D063).

## 13. Not verified, and open questions for the user

- **Root cause of the Xid-8 events.** O4-Z and O4-B attempt a localisation. `CUDA_LAUNCH_BLOCKING` may suppress the hang.
- **Headless operation.** Whether running this GPU headless would remove the RC-watchdog interplay. This needs the user, and is raised if O4-Z shows persistent hangs or O4-A FAILs.
- **R-ATT backend on sm_120.** Whether EFFICIENT_ATTENTION runs in fp32 on sm_120 under torch 2.11 is not verified. If it does not, R-ATT uses exact query-tiled attention, and the VT-22b reference side would need the same. That case is BLOCKED → PI row, never a silent fallback.
- **P4's cost** (estimated at 1–2 GPU-h) is unmeasured until FEAS-SMOKE. If the estimate exceeds 3 GPU-h, a PI row is needed before FEAS.
- **Repair counts outside VT-24's scope.** How many facts the G-ORACLE repair changes there, and how many coincidental matches K0 finds, are reported by VT-24r.
- **The fp16 code path of the 2023 Pythia records** is not needed for RR-R7.
- **R6's 2-layer configuration.** Whether `ModelConfig.for_scale(..., n_layers=2)` works without a driver. If not, the plan's fallback `scripts/repro/known_answer.py` applies.

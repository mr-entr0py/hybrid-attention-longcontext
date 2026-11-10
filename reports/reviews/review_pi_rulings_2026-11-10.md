# Adversarial review: PI rulings of 2026-11-10 (D045–D057, G-RV residuals, pre-FEAS items)

**Role:** adversarial reviewer for scientific integrity (workflow `wf_b92a39ce-f17`). **Date:** 2026-11-10, 13:50–14:30 IST.
**Object:** `reports/pi_rulings_2026-11-10.md` (DRAFT) and the 11 files in `experiments/prereg/`.
**What I did:** read-only. I ran no GPU job, edited nothing except this file, and committed nothing. Checks I ran:
- `sha256sum -c experiments/prereg/SHA256SUMS`: all 11 files OK. The files are dated 13:38–13:44 and hashed at 13:44:24. No output exists yet for O4-A, R2-LIVE, R2e-X, VT-22b, VT-23b, VT-24r, R6 or FEAS-PREC. The only file in `experiments/repro/O4/` is the PI's log scan. So every new pre-registration was written before its item ran.
- A read-only kernel-log scan for 13:31–14:02 found no events (`journal_ok: true`). The cost session was running during that window. Output: scratchpad `klog_1331_1402.json`.
- Re-reads of the raw files cited below. Paths are given in each row.

---

## 0. Bottom line

**Verdict: ACCEPT WITH REQUIRED CHANGES before Phase II (GPU).** No ruling turns a FAIL into a clean PASS in its own text, and every FAIL-as-registered stays on the record. I checked this for VT-03, 05, 07, 13-prod and 15, D512 pass 1, VT-21–24, R2 and R7. The rulings that refuse to restate criteria are correct and should stand: PR-03, PR-14 and PR-19. VT-23b is a sound new test.

Five problems need fixing:
1. **Gate labels.** The labels relabel failures as passes at gate level (O3).
2. **PR-04.** It uses a check whose outcome is already known to replace a pre-registered consequence. PR-19 forbids exactly that move for R7 (O2).
3. **Safeguards for k-differential residuals.** Several accepted residuals affect GDN layers or attention layers but not both, so they differ across k by construction. Their safeguards cannot detect a difference between k, and they do not feed the FEAS verdict (O1).
4. **R6.** The new R6 file contains a reading that is likely to produce a false FAIL (O4).
5. **O4 soak.** O4-A is almost certain to pass, and the only informative part, the failing GDN workload, is deferred until after the gate runs (O5).

Every fix below is pre-run. Each takes a new file with a new ID or version suffix, or a decision row. Nothing that has already run is re-read.

---

## 1. Top objections (ranked by impact on the k-comparison and on record integrity)

### O1. The precision residuals differ across k, and their safeguards cannot see a k-difference and do not feed the FEAS verdict
**Rulings:** PR-02(2), PR-08(i), PR-19, PR-26; `FEAS_precision.yaml`.

**What differs across k:**
- RR-G0-1 (VT-05, bf16 chunk final state) and RR-G0-2 (VT-07, TF32 chunk vs IEEE recurrent) affect GDN layers only: 12, 9 and 0 GDN layers at k = 0, 3 and 12.
- D052(i) bf16-rounded evaluation weights perturb `A_log`/`dt_bias` (relative change 3.5e-3). `s0_integration.md` §12 item 10 says: "The perturbation may differ by k". The effect acts on GDN decay, so it grows with needle distance.
- RR-R7: an attention LM collapses under bf16 scoring (Pythia-160M 0.354 → 0.12; 70M 0.227 → 0.05), while the GDN model does not (0.3435 → 0.3421) (`results_R7.md` L96–105). The cause is unexplained; only a RoPE mechanism was excluded.

**Why the safeguards are inadequate:**
- (a) Q1–Q3 use absolute per-k flip rates pooled over all FD cells. At D384, many FD cells will be at floor or ceiling (k = 0 at 32K, M-hop), where flips cannot occur, so the pooled rate is diluted.
- (b) No criterion compares k = 0 with k = 12. A 0.9% vs 0% split passes.
- (c) There is no breakdown by length, although the D052(i) effect concentrates at 16K/32K.
- (d) Q3's option "(c) keep production scoring with a per-k caveat" is a no-op exit.
- (e) The Q1 and Q3 consequences apply "before S4". The FEAS verdicts (U-01 D384 vs D512, worth +58 GPU-h; FEAS-3 at 3/10 pp; FEAS-5 at 3/2 pp) are computed from P0 scores regardless (`DTS10_acceptance.yaml` inputs). Only Q2 feeds back into FEAS.

**Required (new file `FEAS_precision` v2, before FEAS):**
- Compute Q1 and Q3 before the DTS-10 verdict.
- Add per-k flip rates restricted to informative cells (0.05 < cell EM < 0.95 under P0), at 16K and at 32K separately.
- Add a between-k rule: if any FEAS rule's outcome, or S₁₂ − S₀ at 32K, differs between P0 and P1 scoring by more than 0.02, the verdict uses fp32 scoring of the U-01 verdict cells.
- Delete option (c), or allow it only when the k-contrast is unchanged within 0.02.
- Make the RR-G0-1 and RR-G0-2 reopen triggers use the restricted rates.

### O2. PR-04 / G0-T512x: a check with a known outcome replaces the pre-registered consequence of a FAIL
**What happened.** G0's frozen failure clause says: "On a D512 throughput failure, Stage A drops to 0.6B (pre-registered)". D512 pass 1 failed: C3 measured 67.2K against 70K (`experiments/s0/g0/g0_results.jsonl`). PR-04 instead lets the new G0-T512x decide D512.

**Its outcome is already known.** The idle bench gives D512 Stage A **97.5 / 97.9 / 99.1K** (`experiments/bench/train_step_idle_20261110.json`, 05:37). Pass 2's best windows were 93.3–95.9K. The threshold is 70K, a margin of about 35%. PR-19 refuses "R7b at fp16" because "its outcome is already known… pre-registering it now would relabel a FAIL". The same logic applies here.

**Does it differ across k?** No: the fallback applies study-wide. This is a record-integrity issue, not a bias.

**Required.** Record a decision row that says explicitly: "the pre-registered 0.6B fallback of the G0 D512 throughput FAIL is **waived** by logged deviation, on the ground of D031's exclusive-GPU reopen clause". Relabel G0-T512x as a budgeting re-measurement, not a test that decides or exonerates. The alternative is to apply the fallback.

### O3. Gate vocabulary relabels FAILs, VT-03 is classified two ways, and "447/447" overstates the suite
1. **G0.** PR-01 labels G0 "PASS-WITH-ACCEPTED-RESIDUALS", and the §8 row reads "G0 passed | PR-01 row entered". But six sub-items failed as registered, and G0's frozen failure clause has **no** residual route ("Fix and re-test"); the residual route belongs to G-RV.
2. **G-RV.** §8 says "G-RV passes when every row holds", and some rows are satisfied by "RR-… row entered". An independent gate that follows §8 would print "G-RV PASS".
   - **Required:** gate verdicts of the form "NOT PASSED as registered; FEAS proceeds under G-RV's named-residual clause (RR-…)". Never "PASS".
3. **VT-03.** PR-01 counts VT-03 among the six FAIL-as-registered items (RR-G0-5). PR-02(4) and §1 say "VT-03 … pass[es] as registered" and "pass if the 0.0 bound is read in the training dtype".
   - The frozen text ("GDN max abs diff 0.0", protocol §16) has no dtype qualifier, while the attention clause in the same cell does have one. Measured fp32: 9.4e-5.
   - A dtype reading made after the result is known violates the PI's own `experiments/prereg/README.md` rule 3 ("a reading made after any result is visible … may not create a PASS").
   - **Required:** VT-03 GDN fp32 = FAIL-as-registered → RR-G0-5, with no alternative wording.
4. **"447/447".** "Clean passes … (447/447 G0 tests)" omits 4 skipped tests (the VT-13 real-model audit ×3 and the production test; `reports/gates/G0.md` §1). It also counts tests that assert relaxed, unregistered bounds: the VT-05 median, VT-07 ≤ 2e-2 and VT-15 8 of 11.
   - **Required:** report "447 passed / 4 skipped; N assert relaxed regression guards". For consistency with PR-29(d), add registered-criterion xfail markers for VT-05 and VT-07. Use non-strict markers for bimodal VT-15 cells.

### O4. R6.yaml reading R6-R3(ii) makes a false FAIL likely
**The reading.** FAIL "at any look" if the answer CE over the last 50 steps is ≤ 0.5 × its mean over steps 1–50 **and** the score is < 0.05.

**Why it fails spuriously.**
- V = 16,384 (protocol §6.2), so the initial answer CE is about ln V = 9.70 nats.
- Learning only that answers are digits gives about ln 10 = **2.30**, which is already below half of any plausible steps-1–50 mean.
- (ii) therefore reduces to "score < 0.05 at any look ⇒ FAIL", including at 50M.
- That contradicts R6-R2 ("a score < 0.5 at 50M also continues") and the plan's 150M extension, which exists for slow learners (plan §2.6, "keep slow learning from being read as a bug").
- A false R6 FAIL blocks G-RV and starts a hunt for a bug that does not exist.

**Required (R6 v2 before R6 runs):**
- Anchor the CE condition at an absolute level well below the digit prior, for example ≤ 1.0 nat per answer token (the model solves training answers but not held-out ones), or evaluate (ii) only at 150M.
- Also add, report-only, the chain-consistency flip rate (cache path vs single pass) for the `--` (GDN) model. R6(a) covers only the attention `AA` cache path. The GDN path, which is where RR-G0-1/2 live, is otherwise first tested on a trained model at FEAS (Q2).

### O5. O4: the verdict-bearing soak tests a workload that has never failed, and the informative part is deferred past the gate runs
**The facts.**
- All 7 RC-watchdog/Xid-8 events happened during GDN training in the Zoology harness. They are "not tied to a fixed step", "also with no other project process", and the failure is `cudaErrorLaunchTimeout` (`results_R5.md` L105–159).
- The candidate causes include an fla/Triton kernel on sm_120 (`results_R5.md` L146).
- The hang is GDN-associated, so exposure is correlated with k.

**What O4 does.**
- O4-A soaks only production shapes, where no event has ever occurred.
- P(pass) = e^(−2λ) ≥ 0.5 for any rate λ ≤ 0.35/h. That rate would mean about 31 events over D384 Tier A, or about 51 on the D512 path.
- The A1/A3 probes run in audit mode, so they do not exercise the non-deterministic production kernels.
- The plan's O4 includes "the failing job rerun alone with `CUDA_LAUNCH_BLOCKING=1`" (plan §2.11), and O4 must run "before any further gate run" (§2.2). PR-21 makes that part a non-verdict O4-B and schedules it after R6, VT-22b and VT-23b. That is an unlogged sequencing deviation.

**Required:**
- Run O4-B before the gate runs (≤ 1 h).
- Add a report-only 60-min soak of Zoology GDN d64 s125 **without** launch blocking on the now-exclusive GPU. At the R5 rate (3.4/h), P(0 events) = 0.033.
- Pre-register what each outcome means. Persistent hangs → FEAS runs under PR-27 with a per-k resume monitor (PI review if resume counts for k = 0 and k = 12 differ by more than a stated number) and O9's resume check is mandatory. No hangs → evidence that the trigger was contention-related.

### O6. Strict-xfail and VT-25 RNG pinning create undefined, blocking outcomes
**VT-22 XPASS.**
- PR-29(d) marks VT-21, VT-22 and VT-23 `xfail(strict=True)`, and the §8 G0 row requires them to xfail in the pre-FEAS suite.
- VT-22 exceeds its bound by only 1% (16K) and 0.2% (32K).
- The pre-FEAS suite runs on `.cache/triton-diag`, which is the S0 archive (PR-12). That archive predates the fla-LM 16K/32K kernels that R3 autotuned into the live cache at 06h (`reports/gates/G0.md` §6), so those kernels will be re-autotuned.
- If VT-22's numbers move below 2e-2, the result is an XPASS. Strict mode turns that into a suite failure, which blocks G-RV because of an *unexpected pass*, and invites a relabel.

**VT-25 pinned batch.**
- PR-18 pins VT-25's batch, so the pre-FEAS VT-25 runs on a new batch. The worst-tensor error is batch-dependent: 5.6e-3 to 1.77e-2 at d64 and 1.82e-2 in the verdict run, against 2e-2 (audit §1.5).
- §8 requires "VT-25 pass" in the pre-FEAS suite, while PR-18 says such a failure is "evidence, not an automatic G-RV change". The two statements conflict.

**Required (pre-registered now):**
- An XPASS is recorded, is not a PASS of the item and does not block. Use `strict=False` or a custom marker.
- State the consequence of a pinned VT-25 above 2e-2 and of the 10-seed sweep now, for example RR-VT25 "margin not robust", with G-RP(b)'s precondition reported as "met on the registered run".

### O7. PR-12 (Triton cache) rests on a false premise and cites a garbled hash
1. **False premise.** PR-12 says the restore "removes autotune entries made under contention (E10.2)". But the S0 archive itself was warmed under contention.
   - The warm ran 23:13–23:39 on 11-09 (`runs/_logs/warm_triton.log`).
   - CARLA was running at that time (STATUS 22:15: 3.6 GB).
   - The VT-13 C12 audit ran concurrently, and the warm process (8.64 GiB) OOM'd it (`s0_integration.md` §6).
   - E10.2's safeguard is "pre-warmed **on an idle GPU**".
2. **Garbled hash.** The verification hash "`be0c80b1…f74316`" is spliced from two hashes:
   - its prefix comes from the post-warm hash `be0c80b1fdd3…cd09fed7` (`experiments/registry.jsonl` row `S0-triton-cache`);
   - its suffix comes from the pre-warm hash `1330ca45…f4f74316` (warm log, line 1);
   - the error originates in `s0_integration.md` §8 and was copied into G0.md and PR-12;
   - an executor cannot verify against it.
3. **R2e-X cache conflict.** R2e-X's precondition implies the production cache, but README rule 4 and PR-12(1) put every reproduction job on `triton-diag`. R2e-X's audit-mode cells would also compile kernels into whichever cache it uses.

**Required:**
- Before FEAS, warm a fresh, empty production cache on the exclusive GPU (about 26 min, from the 1,579 s log). Archive it and hash it.
- Report how many autotune choices differ from the S0 archive (report-only).
- State which cache R2e-X uses.
- Define the G5 "re-warm" as add-only, with byte-identical existing entries. Otherwise it is a mid-study cache change by PR-12's own definition ("the study begins at S2").

### O8. Parts of VT-22b and VT-24r cannot fail
**VT-22b.**
- B1 (≤ 1.5 × max self-drift) is the R3 executor's own proposal after the failure (`results_R3.md`, PI request 2). It was calibrated on VT-22's diagnosis: cross error 2.02e-2 against self-drift 2.09–2.15e-2, which puts the B1 bound near 3.2e-2.
- F1/F2 at 16K are also already known: 2.61e-4, band ratio 1.10 (`vt22_diagnosis.jsonl`).
- Only F1/F2 at **32K** carry new information.
- **Required:** state the predicted values before VT-22b runs, so that B1/B2 carry no weight at G-RV.
- The same "fp32 only at 4K" gap applies to C12: RoPE at 32K matters only for k > 0. Either justify the asymmetry (FA2-varlen has no fp32 path, and C12's bf16 floor of 6.4e-3 is 3× tighter) or add a C12 arm.

**VT-24r.**
- The repair loop enforces exactly K1's condition (G-ORACLE). K1 is informative only if the generator-side check is an implementation independent of `devtools.InductionLM` and the scorer. Require that.
- K5 samples only TEF steps 0–2 for seeds 0 and 1. TEF items come from the **same** eval generators in namespace `train` (`src/hybridlc/data/tef.py` L28, L363), and collisions are rare. A repair leaking into training would pass K5.
- **Required:** a unit test that a planted collision in a `train`-namespace item is *not* repaired, or a hash of a full 1,000-step TEF block.

### O9. RR-G0-3's reopen trigger fires, but no real-model resume check is scheduled
**The rules involved.**
- RR-G0-3 rests on the real-model audit-mode bitwise resume. That audit ran on a frozen snapshot of `a3dfff2` plus the builders' files (`s0_integration.md` §6).
- Its reopen trigger is "resume code changes".
- PR-03 and `VT-13p2.yaml` say an audit "becomes mandatory if checkpoint, resume, loader or RNG-state code changes".

**What changes the code.**
- R2-LIVE L1/L2: a per-job lock and token, and runner `attempt_end` rows.
- PR-27: the policy for resuming infra-failed attempts.
- Neither the audit nor VT-13p2 is in §10.

**What the planned suite covers.** The pre-FEAS suite (about 0.3 h) skips the env-gated real-model audit (`HYBRIDLC_VT13_FULL`, `tests/test_vt13_training.py` L624–630). Mitigation: the fakes-based bitwise kill-9/resume tests do run (L412, L577).

**Why it matters for k.** Resumes driven by hangs concentrate in GDN workloads, so resume exactness matters for the k-comparison.

**Required:** state whether runner-side changes count as resume code. Either way, run one real-model audit-mode kill/resume (C3, 200 steps, about 25 min) on the final pre-FEAS tree.

### O10. PR-27 classifies every Xid as infrastructure, which can hide a k-specific software defect
**The rule.** "Tagged events are Xid, RC watchdog, host OOM-kill, or a CUDA launch timeout or illegal address with a matching NVRM line." All of these count as infra, never count toward failure, and allow up to 10 resumes.

**The problem.** Xid 13, 31 and 43 and illegal-address faults are typically application faults. A shape-specific kernel bug, in GDN or in attention and hence specific to some k, would be filed as infrastructure, and its non-crashing occurrences would be invisible.

**Required:**
- Give infra status only to Xid 8/RC watchdog, Xid 79 and host OOM-kill.
- Treat Xid 13/31/43 and illegal address as "suspected software failure": a distinct status, where a recurrence at the same step stops the run for diagnosis.
- Report counts per k, as already planned.

### O11. An unnamed residual that differs across k: R5's power
R5 excludes only a bf16 GDN training handicap larger than 0.115 (50% power) or 0.154 (80% power). A smaller handicap would lower every k = 0 baseline and inflate k-contrasts in attention's favour.

PR-22's "FEAS-2 and k = 0 carry no R5 caveat" follows the pre-registration, which attaches a caveat only on FAIL. But the residual is real, and FEAS-PREC covers *scoring* precision, not *training* precision.

**Required:** name RR-R5-power in §6 and carry it to the limitations.

### O12. PR-06 omits the frozen §7.3 wording
Protocol §7.3 reads "… + ½·mean CE over all other tokens **in the step**". PR-06 says only that D014 "has no 'per step'".

The change is allowed because no TEF run has happened. But it is a pre-run protocol change that resolves a conflict between §7.3's text and D014/§14's purpose; it is not a pure clarification. It also halves the 32K bucket's answer weight relative to the per-step reading (0.29 → 0.143). That is identical across k, but it changes exposure at the primary cell. Record it as such.

### O13. Missing before FEAS spends about 10 GPU-h
1. **FEAS-shaped smoke on the final tree** (≤ 0.2 GPU-h), using the real FEAS YAMLs:
   - both Stage-B branches and the SSMax arm;
   - TEF with block normalisation;
   - runner, lock and infra tagging;
   - G-div near-miss logging;
   - FD scoring including the P0–P3 passes;
   - `feas_decide` on the smoke tree.

   The S0 e2e smoke ran on an older tree and caught a real bug (`s0_integration.md` §11).
2. **D041 re-pricing** must include the cost of FEAS-PREC P1, a second full-FD scoring of every checkpoint. O4 now takes 3.4 of the 4.7 h optional contingency.
3. **O1, O4, O6 and O7** fixed by new files before Phase II.

### O14. Minor items
- **PR-10.** A G-div near-miss has no consequence for the FEAS verdict. An untriggered excursion of 1.0–1.65 nats in a k = 12 FEAS run could push U-01 to D512. Pre-register: the verdict is flagged, and if the flagged run is decisive, the PI enters a row before acting.
- **R2-LIVE C2.** A violation should invalidate the gate or timed run whose window it falls in, as plan §2.2(a) requires ("No two project GPU processes run during a timed or gate run"); R2e-X is timed. As written, a violation only blocks G-RV, and C2 is re-judged "over the next window".
- **DTS-10 blinding** is procedural only. The orchestrator habitually quotes numbers in STATUS and tails logs. While FEAS is in progress, `evaluate.py` should print no scores before the `DTS10-commit` note.
- **DTS-10 readings.** I-08 "mixed allowed" is the reading that triggers most easily. The elicitation rule "for both seeds" matches §7.3. Both are fine before the run; state them as readings.
- **PR-30.** `scripts/benchmark_inference.py` is untracked. Commit it before any of its numbers is cited.

---

## 2. Ruling-by-ruling verdicts

| ID | Verdict | Note |
|---|---|---|
| PR-01 | OBJECTION | Label; VT-03 contradiction; 447/447 (O3) |
| PR-02 | OBJECTION | (4) VT-03 dtype reading made after the result (O3); (1)–(3) and (5) OK |
| PR-03 | OK | Correct refusal to use attempt 2 as the verdict; reopen trigger fires (O9) |
| PR-04 | OBJECTION | Check with a known outcome replaces a pre-registered consequence (O2); cache conflict (O7) |
| PR-05 | OK | — |
| PR-06 | OK with correction | §7.3 "in the step" (O12) |
| PR-07 | OK | Pre-run |
| PR-08 | OK, measurement weak | (i) needs the restricted, by-length, between-k form (O1) |
| PR-09 | OK | Pre-run |
| PR-10 | MINOR | Near-miss consequence at FEAS (O14) |
| PR-11 | OK | — |
| PR-12 | OBJECTION | False premise, garbled hash, R2e-X cache (O7) |
| PR-13 | — | Add the O13 items to the §9 checklist |
| PR-14 | OK, exemplary | No post-hoc relative criterion on the same data |
| PR-15 | MINOR | B1/B2 and 16K results already known; C12 asymmetry (O8) |
| PR-16 | OK | VT-23b can fail; its target defects give ≥ 1e-3 |
| PR-17 | OK with conditions | K1 independence; K5 leak test (O8). Regeneration is the conservative route, and the original FAIL is kept |
| PR-18 | OBJECTION | Undefined outcome; conflicts with the §8 row (O6) |
| PR-19 | OK | k-relevance of the bf16 collapse carried in O1 |
| PR-20 | OK | C2 invalidation rule (O14) |
| PR-21 | OBJECTION | O4 design and sequence (O5) |
| PR-22 | OK | Name the power residual (O11) |
| PR-23 | OBJECTION | R6-R3(ii) false-FAIL risk; add the `--` chain check (O4) |
| PR-24 | OK | — |
| PR-25 | OK | Enforce blinding technically (O14) |
| PR-26 | OBJECTION | Q3 no-op exit; consequences come after the FEAS verdict (O1) |
| PR-27 | OBJECTION | Xid classification (O10) |
| PR-28 | OK | — |
| PR-29 | OBJECTION on (d) | XPASS hazard (O6); (a)–(c) and (e) OK |
| PR-30 | OK | Commit the script |

## 3. Do the accepted residuals differ across k?

| RR | Differs across k? | Safeguard adequate? |
|---|---|---|
| RR-G0-1 (VT-05) | **Yes**: GDN only (12 / 9 / 0 layers) | **No**: pooled, diluted flip rate; no between-k test (O1) |
| RR-G0-2 (VT-07) | **Yes**: GDN path only; small (2.4–3.8e-3 nats) | Same as RR-G0-1 (O1) |
| RR-G0-3 (VT-13 prod) | **Exposure yes**: hang-driven resumes concentrate in GDN workloads | Yes if resume is exact; re-verify on the final tree (O9) |
| RR-G0-4 (VT-15) | Measured costs only (attention decode cell); accuracy no | Yes: secondary metric; the primary cold-L2 cells pass |
| RR-G0-5 (VT-03) | No (fp32 only; production bf16 is exact) | Yes |
| RR-G0-6 (throughput) | No (fallback is study-wide) | Yes; label issue (O2) |
| RR-VT21 | Exposure yes (conv in GDN and F_gc), at 1–2 ulp | Yes |
| RR-VT22 | **Yes** (C0 only) | VT-22b F1/F2 at 32K informative; C12 asymmetric (O8) |
| RR-VT23 | No (shared trainer) | VT-23b is sound |
| RR-R7 | **Potentially**: unexplained bf16 collapse of an attention LM | **No** (O1) |
| RR-R2, RR-R5-D1, RR-G0suite-seq, RR-prereg | No | Yes |
| *Unnamed:* R5 power | **Yes**: a bf16 GDN training handicap ≤ 0.115–0.154 is not excluded | Not covered (O11) |
| *Unnamed:* D052(i) bf16 evaluation weights | **Yes** (`A_log`/`dt_bias`) | Q1 pooled; needs a 16K/32K breakdown (O1) |

## 4. Can each new pre-registration fail?

| File | Can it fail? | Note |
|---|---|---|
| `R2e_G0T_exclusive.yaml` (G0-T512x) | **Practically no** | Idle measurement 97–99K vs 70K (O2) |
| `VT-22b.yaml` | F1/F2 at 32K yes; B1/B2 and 16K results already known | O8 |
| `VT-23b.yaml` | Yes | Sound |
| `VT-24r.yaml` | K1 tautological unless the code paths are independent; K3, K4, K6 yes; K5 weak | O8 |
| `O4_soak.yaml` | O4-A very unlikely to fail; O4-B is never verdict-bearing | O5 |
| `R2_live.yaml` | Yes (C1); C2 can be re-judged until it passes | O14 |
| `R6.yaml` | **Too easily**: likely false FAIL | O4 |
| `FEAS_precision.yaml` | Q2 yes; Q1 has a consequence only before S4; Q3 has no fail state | O1 |
| `DTS10_acceptance.yaml` | Yes | Sound |
| `VT-13p2.yaml` | Yes (future only) | Trigger arguably already fired (O9) |

## 5. Order of the required changes (all before Phase II step 12)

1. Corrections to the decision rows:
   - O3 vocabulary and VT-03;
   - the O2 waiver row;
   - O12 recorded as a protocol change;
   - O11 RR-R5-power;
   - the O7 hash and premise;
   - the O10 PR-27 classification.
2. New pre-registration files, hashed before their items run:
   - `R6` v2 (O4);
   - `FEAS_precision` v2 (O1);
   - `O4` v2 or an O4-B timing row plus the no-blocking soak (O5);
   - outcome rules for XPASS and the VT-25 pin (O6);
   - a VT-22b prediction note and C12 decision (O8);
   - VT-24r K1 independence and a K5 leak test (O8);
   - the R2-LIVE C2 invalidation rule (O14).
3. Additions to the §9 checklist: a fresh exclusive Triton warm (O7); one real-model resume audit (O9); a FEAS smoke (O13); FEAS-PREC costs added to D041 (O13); technical blinding (O14).

**Evidence base:** `reports/pi_rulings_2026-11-10.md`; `experiments/prereg/*`; `reports/gates/{G0,G-RP}.md`; `reports/reproduction/{audit_R2-R7,results_R3,results_R5,results_R7,reproduction_plan,protocol_amendment}.md`; `reports/s0_integration.md` §6, §8, §11, §12; `reports/experimental_protocol.md` §6.2, §7.3, §13.3, §16, §18; `reports/competing_explanations.md` O10; `experiments/bench/train_step_idle_20261110.json`; `experiments/s0/g0{,_rerun}/g0_results.jsonl`; `runs/_logs/warm_triton.log`; `experiments/registry.jsonl`; `experiments/repro/R3/vt23/attempt_20261110T062646/.../trace.jsonl` (pre-clip norms 2.79 / 2.76 / 2.29 / 1.80 confirmed); `tests/test_vt13_training.py`; `src/hybridlc/data/tef.py`.

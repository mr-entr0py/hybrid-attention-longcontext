# Continuation adversarial review (Agent 5): can the programme establish what it claims?

**Date:** 2026-11-10, verified 20:45–21:10 IST. **Role:** adversarial scientific reviewer (NeurIPS-level sceptic) of the whole programme as it stands: question, controls, hypotheses, tasks, the TEF fine-tune, statistics, cost comparisons and contribution.

**Status of this file.** This version supersedes the 15:43 draft, which the 17:00 usage limit interrupted before hand-off. The draft is preserved unchanged in git commit `0c54713`. I re-checked every number, script and citation in the draft against HEAD `7f8bed3` and the live working tree:
- all four draft scripts reproduce byte-identically (§3.0);
- stale or wrong line citations are corrected (§7);
- findings marked **NEW** were added in this pass.

**Mode:** report-only. I edited nothing in `src/`, `scripts/`, `tests/`, `configs/` or `experiments/`. I wrote only this file and scratch files under `.audit/adversarial/`.

**GPU:** none used. No trained model exists yet, so every question here is about design, statistics or frozen items. All computations are CPU-only numpy/scipy, plus read-only loads of frozen item files and of the TEF block planner. Avoiding the GPU also kept me clear of workflow `wf_b92a39ce-f17`'s O4/R6 jobs and of CARLA.

**Concurrency note.** `wf_b92a39ce-f17` was editing the tree during this review. `git status` at 20:56 showed `src/hybridlc/analysis/results.py`, `utils/provenance.py` and two tests modified. The verdict code I cite, `analysis/pipeline.py` and `analysis/verdicts.py`, is unchanged since `656425f` (`git log -- …` output, §3.8).

**Read, in full or in the relevant sections:**
- `reports/STATUS.md`, `continuation_audit.md`, `decision_log.md` (P001–P004), `status_log.md`;
- `experimental_protocol.md` v1.0 (all 20 sections, re-read in full in this pass);
- `decisions.md` (D000–D091, DEV-01…04);
- `pi_rulings_2026-11-10.md` (FINAL; §0–§10 in full);
- `competing_explanations.md` (in full);
- `reviews/debate_record.md` (I-01, I-11, I-12, I-23, I-26–I-28, §3, §6.2) and `review_pi_rulings_2026-11-10.md` (scanned for overlap);
- `literature/literature_review.md` (§0–§4, §8–§9, §12–§14);
- `reproduction/results_R5.md`, `results_R7.md`, `audit_R2-R7.md` (scanned), `reproduction_plan.md` (§2.6, §2.9);
- `gates/G0.md` (open items), `s0_integration.md` (§11–§13), `cost_benchmarks_session1.md` (§0–§2);
- `src/hybridlc/INTERFACES.md`, plus the analysis, eval and TEF code cited below.

**Conventions.**
- Every claim carries one of:
  - a file:line;
  - a command with its output (scripts and outputs are in `.audit/adversarial/`);
  - a literature reference with the access tag *as recorded in the project's own files*. I fetched no URL myself, so literature claims are "per" the cited project file.
- Severity:
  - **FATAL**: the named claim cannot follow from the design as registered, even if every run succeeds.
  - **MAJOR**: the claim follows only with a serious, currently unaddressed caveat, or is likely to be misread.
  - **MODERATE**: it weakens a control, a label or the stated power.
  - **MINOR**: hygiene or wording.
- Severity is judged **per claim**, not for the whole programme. A FATAL row means the named claim is unreachable, not that the study is.
- [EVID] marks evidence; [SPEC] marks my speculation or judgement; [INF] marks arithmetic or inference.

---

## 0. Verdict

### What the programme can establish
**The narrowly worded confirmatory package claim can be established.** That claim is: "k F_g layers of 12, the rest GDN-A as built, recover ≥ 80% of the k = 0 → 12 gap on literal key–value retrieval at 16–32K (pooled), after an identical ≈ 1.7K-item task-exposure fine-tune, at ≈ 28M parameters."

Also reachable:
- H1-min in the same narrow form;
- H5(a), the length growth of the k = 12 vs k = 0 gap at ≤ 32 facts;
- per-level N50 curves;
- exact analytic state-byte accounting (72/72 caches to the byte, `cost_benchmarks_session1.md` L23).

The pre-registration machinery is unusually strong:
- paired seeds with common random numbers;
- gating on independent dev items;
- a fixed level set for H5;
- guard-protected falsification;
- hashed pre-registrations and the D088 verdict vocabulary;
- a reproduction ladder.

### What it cannot establish as registered (FATAL per claim; all secondary or named-negative claims)
- H6′, "GDN-E recovers the gap" (F-1);
- H3, "no task dependence" (F-2);
- the null reading "aggregation needs little full attention" (F-3).

### Blind spots that bear on the headline H1 / H1-min reading
1. **Reference handicap (M-1).** An adequate-but-beaten all-full reference is never flagged, and R_best is vacuous by default.
2. **Literal copy (M-2).** One key token resolves ≥ 92% of queries on levels that carry ≈ 80% of the retrieval-family weight.
3. **Elicitation regime (M-3, NEW M-9).** The TEF gives each model ≈ 1,694 retrieval items per 1,000 steps, and the dose ladder spans only 4× on one seed. So "recovers most performance" is a statement about sample efficiency at that budget, not about capability.
4. **Power under Holm (m-0).** Advertised power is 0.88; the realistic figure is 0.55–0.70 at n = 5, σ = 3 pp.
5. **NEW: pooling hides 32K (m-14).** Pooling 16K and 32K in the primary cell, with no per-length reading, can report "supported at 16–32K" while k = 3 fails at 32K.
6. **NEW: the bias direction of RR-R5-power is indeterminate for L₃ (m-12, updated).** The PI register says the residual inflates k-contrasts in attention's favour. That holds for G and Δ₀, but for L₃ the shift is +0.2δ or −0.55δ depending on how a bf16 GDN handicap spreads across k.

**All of these are fixable before FEAS or G5** by decision rows, mostly at zero GPU cost. None touches the criterion of an item that has already run.

---

## 1. Severity-ranked findings

"Pre-run?" means the fix can be entered as a logged pre-run change. The study is pre-FEAS: S2 has not started and no study item has run (`reports/STATUS.md` L5–8; `continuation_audit.md` §4). Changes to frozen protocol text need a DEV row (precedent: DEV-04, `pi_rulings` L246–255).

| ID | Sev. | Claim affected | Finding | Evidence | Consequence if unaddressed | Recommendation (pre-run?) |
|---|---|---|---|---|---|---|
| **F-1** | **FATAL** (secondary claim) | H6′: "GDN-E recovers the gap" / "state worth as much as KV" | **Statistics.** 2 seeds of X-E0 against n₀ seeds of C{k*}, unpaired Welch: median df ≈ 2.6–2.8.<br>• The registered TOST ±5 pp declares "equivalent" in only **10–28% of worlds where the true difference is exactly 0**.<br>• "Differs" has 50–62% power at a true 10 pp.<br>**Confound.** GDN-E has 3.67× (D512) / 2.78× (D384) the recurrent-state bytes of GDN-A and 1 head instead of 4 / 3. "Per-head d_k at equal H·d_k" (D024) is therefore not the only change: H·d_k·d_v grows 4× / 3× | Sim: `.audit/adversarial/opchar_secondary.out`, H6′ block (n = 5: P(equiv \| Δ = 0) = 0.140 at σ = 3 and 0.279 at σ = 2; P(p < .05 \| Δ = 10) = 0.528 / 0.615).<br>Rule: protocol L75, L717–720; code `analysis/pipeline.py` L1063–1084, `stats.py` L232–235.<br>State bytes: protocol L159–160 (147,456 vs 540,672 B at D512; 110,592 vs 307,200 B at D384) | Almost every H6′ outcome will be "inconclusive". A Δ_E near 0 cannot be called recovery, and a positive Δ_E cannot be attributed to per-head d_k | Make H6′ estimation-only (Δ_E with CI; gap share with Fieller, already coded at `pipeline.py` L1076–1081), **or** run X-E{0,k*} at n₀ seeds on common seed indices with a paired test (≈ +5–8 GPU-h at D384 [INF from §13.2 rates]). Reword as "≈ 3–4× recurrent state, one head". Pre-run: yes |
| **F-2** | **FATAL** (named negative) | H3 "no task dependence" | TOST ±0.10 on the per-seed ratio difference D₃ is practically unattainable.<br>• Per-seed SD(D₃) is 0.11–0.30.<br>• P(equivalent & evaluable \| true D₃ = 0) is ≤ 0.25 even in the best case (n = 7, σ = 2, G_hop = 30 pp: 0.246; n = 5: 0.126), ≤ 0.04 whenever σ = 3, and ≤ 0.07 whenever G_hop ≤ 20 pp | Sim: `opchar_secondary.out`, H3 block. Rule: protocol L70; verdict code `verdicts.py` L99–102 | A null on task dependence can never be reported. Only "supported" or "inconclusive / not evaluable" are reachable | Declare the negative outcome unreachable in the protocol, or replace it with an estimation statement (D₃ with CI and the MDE). Pre-run: yes |
| **F-3** | **FATAL** (null reading) | RQ-C aggregation: "integration needs little full attention" | A-freq at 32K is largely solvable without long-range integration. On the frozen CS items, 32K ladder means (ρ ∈ {4, 2, 1.5, 1.25}; chance 0.5) are:<br>• recency heuristic **0.651**;<br>• counter over the last 4,096 tokens **0.708**;<br>• last 8,192 tokens **0.777**.<br>Only the last-8,192 floors at ρ ≤ 1.5 (0.694, 0.595) demand integration | `.audit/adversarial/afreq_windows.out` (§3.4), re-run identical. Consistent with `s0_integration.md` L359 (item 17), which covers only ≤ 4K and windows ≤ 4,096. Code records only W ∈ {256, 1,024, 4,096} (`INTERFACES.md` L177; `eval/scorer.py` L241–272) | "Aggregation insensitive to k" cannot be told apart from "every model uses a short-window or recency heuristic". A positive integration claim stays possible if scores clear the floors | Pre-register that integration wording requires exceeding the last-8,192 floor at ρ ≤ 1.5 by a stated margin. Add `window8192` to `AFREQ_WINDOW_FLOORS` and a `window4096/8192_floor` column to `cell_summary.csv`. Pre-run: yes |
| **M-1** | **MAJOR** | H1 / H1-min; "hybrid > all-full" | **Reference-handicap blind spot.**<br>(i) "Reference inversion" fires only when C12 **fails** adequacy (a).<br>(ii) R_best is chosen on DG, which is scored on C0 and C12 only unless inversion is checked, so by default **R_best ≡ R**.<br>(iii) A C12 that passes 0.8·S₁₂(no-filler) but is beaten by C3 by ≥ 5 pp yields "H1 supported" with no label | Protocol L509, L651–653.<br>`analysis/pipeline.py`: L231–232 returns `triggered=False` whenever `a_pooled`; L1001–1014 sets R_best = argmax over DG configs.<br>`analysis/verdicts.py` L53–126: no label for an adequate-but-beaten reference | The headline "a small proportion recovers most of the performance" can be driven by an all-full model handicapped at 16–32K. Candidate causes: thin Stage B of 30–90M tokens (protocol L112); QK-norm dilution at d_h = 64 (D013); small-scale all-full deficit (Qiao Table 2 [read-section, PI-VER in debate], protocol L871) | • Score DG on C1 and C3, EM only with candidate scoring off. Cost: ≈ 0.03–0.10 GPU-h per model by §9.9's prefill model; see m-6.<br>• Add a label-only rule, "reference exceeded": a hybrid's paired primary-cell score exceeds S₁₂ by ≥ 5 pp with paired p < 0.05.<br>• Print R_best beside every H1 / H1-min verdict.<br>Pre-run: yes |
| **M-2** | **MAJOR** | H1, H1-min, "k = 1 ≈ k = 12" (construct validity of "retrieval") | **Literal copy.**<br>• **100%** of R1 and R-load query prompts are literal copies of the fact prefix (" The code for {adj} {noun} is").<br>• Another fact shares the queried key's *noun* in 0.0% / 1.3% of R1 D = 3 / D = 15 queries at 32K, and in 1.8 / 7.8 / 30.1 / 50.9% at N = 32 / 128 / 512 / 1,024. This matches the analytic 1 − (1 − 1/1,390)^(N−1) [INF].<br>• So **≈ 80% of the retrieval-family weight** (R1 D3/D15 plus R-load N ∈ {8, 32, 128}; 0.5 + 0.5·3/5) sits on levels where matching (noun, " is") resolves ≥ 92% of queries.<br>• Keys are drawn at random, with no hard negatives.<br>• T1 is one of the three TEF templates: one third of TEF items use the evaluation template | `.audit/adversarial/key_overlap.out` (§3.3), re-run identical.<br>`eval/templates.py` L46–49; `eval/generators.py` L118–122; family weighting protocol L53–56.<br>TEF templates `eval/constants.py` L99; uniform draw `data/tef.py` L390 | This is induction-style copy, which one attention layer plus conv-supplied previous-token features is known to implement (Olsson 2209.11895 [read-section per A1, B2]; Wen et al. 2402.18510 [abstract-only per literature review §4]). H1-min support would then be largely a property of the task format. Literal matching overstates long-context ability (NoLiMa 2502.05167 [read-section per literature review §8]) | (a) Add an exploratory hard-negative cell at 16K and 32K: distractors share the adjective or the noun with each queried key, which forces 2-token binding. Evaluate it on all core models.<br>(b) Word H1 as "literal-copy retrieval".<br>Pre-run: yes (additive manifest entries) |
| **M-3** | **MAJOR** | H1 / H1-min support (elicitation; E1.7) | **The still-closing flag tracks only S₁₂ − S_k** (seed 0, TEF 1,000 vs 4,000).<br>• A k = 0 baseline still learning at 1,000 TEF steps inflates G, and so inflates R(k) whenever S_k < S₁₂, since dR/dS₀ = (S_k − S₁₂)/G² < 0.<br>• Example: S₀ 40 → 50, S₃ 64 → 66, S₁₂ 70 → 72 moves R₃ from 0.80 to 0.73 and raises no flag.<br>• The TEF LR (3e-4 constant) and length (1,000 steps) are single points, with no LR sensitivity arm | Protocol L383, L697–698; `analysis/pipeline.py` L861–889 (`g1`, `g4` use only `C12 − C{k}`, L885–886). C0 runs to 4,000 steps (protocol L381), so the data will exist | "H1 supported" may partly reflect GDN's slower elicitation (Jelassi 2402.01032 [read-section per B1]) rather than capability | Extend the flag: label "budget-conditional" if sign(L_k) changes or \|ΔR(k)\| > 0.1 between TEF 1,000 and 4,000 (seed 0; all four anchors are already run to 4,000). No new runs. Pre-run: yes |
| **M-9** (NEW) | **MAJOR** (wording / construct) | H1, H1-min, "k = 1 ≈ k = 12", "hybrid > all-full" | **"Capability elicited after identical exposure" is, quantitatively, sample efficiency at ≈ 1.7K retrieval items.**<br>• Per 1,000-step TEF block, each length bucket holds 121 R1 and 121 R-load items: **1,694 retrieval items, ≈ 6.8K supervised retrieval queries** in total.<br>• Only 242 of those items are R-load at 16K + 32K.<br>• The dose ladder (TEF 4,000) is a 4× span on seed 0 only.<br>• The literature reports ≈ 100× sample-efficiency gaps for copying between transformers and fixed-state models ("100x less samples", Jelassi per B1).<br>• A 4× ladder with a 3-pp flag cannot tell "GDN cannot" from "GDN has not yet" | `.audit/adversarial/tef_counts.out` (TEFStream.plan_counts, seed 0, block 0; `items_per_bucket() = 907`); `data/tef.py` L13–20.<br>Protocol L295 (TEF = 1,000 × 64K tokens), L381, L383.<br>`competing_explanations.md` L44 (E1.7) | Readers will take "recovers most long-context performance" as an architectural capacity statement. The design measures learnability within ≈ 1.7K–6.8K items. The protocol's own scope line (L739) is correct, but nothing quantifies it | • Wording rule: every H1 / H1-min sentence states the TEF budget (≈ 1.7K retrieval items) and the dose-ladder span (4×, seed 0).<br>• Optional (Tier C, pre-registered reading): continue seed-0 C0, C3 and C12 to TEF 16,000 (≈ +7 GPU-h idle at D384 [INF: 12,000 steps × 65,536 tokens at 107K / 98K / 78K tok/s, `status_log.md` L42–43]). Label H1 "budget-conditional" if sign(L₃) or \|ΔR₃\| > 0.1 changes.<br>Pre-run: yes |
| **M-4** | **MAJOR** | H3 (retrieval vs multi-hop) | (i) The two families differ in **operation type**. Retrieval is a one-step copy (M-2). M-hop needs h + 1 sequential lookups without chain of thought, and the link line continues " the code for…", so literal copy fails.<br>(ii) They also differ in **load**: R-load has up to 1,024 facts, M-hop ≤ 4(h + 1) ≤ 20.<br>(iii) Evaluability is fragile: every seed needs G ≥ 15 pp in both families. P(evaluable) is 0.53 at G_hop = 20 pp, σ = 3, and 0.03 at G_hop = 15.<br>(iv) The equated-level check equates S₁₂ only, not S₀ or load | `eval/templates.py` L46–49 (link vs query); protocol L70, L480, L710–713; `opchar_secondary.out` H3 block | H3 is likely "not evaluable". If supported, it is open to a "copy vs compose / low vs high load" reading that the design cannot exclude | • Add an exploratory load-matched pair (R-load at N = 4(h + 1) vs M-hop h, same T) to separate load from composition.<br>• Report non-evaluability prominently.<br>• Word H3 as "templated pointer-chasing vs literal copy".<br>Pre-run: yes |
| **M-5** | **MAJOR** (mechanistic wording) | Any "softmax memory vs delta-rule memory" reading | Compute is not matched: 104 → 501 MFLOP/token at 32K (D512). No arm separates "unbounded memory" from "more compute". Of A1-F2's four asymmetries (conv, gated output norm, L2-normalised q/k with learned decay, implicit position source), only the conv has a control. The gate asymmetry is removed by design | Protocol L258, §15 "Infeasible" (L905–907); debate I-01 (A1-F2 list, `debate_record.md` L134–139), I-28 | Mechanistic conclusions remain conjecture | Keep package wording. Name the two uncontrolled asymmetries (q/k normalisation with decay; implicit position) in the limitations. No change to runs |
| **M-6** | **MAJOR** (external validity) | Headline generality | One scale (D384 likely), one linear mechanism, and an elicited capability after a task-specific fine-tune on the evaluation template. "Small-scale parity hid multi-hop gaps at scale" (MiniMax M2 blog [read-full per literature review §4]) applies directly to H3. The literature review itself scopes the work as an "extension and partial replication" of Wang et al. (`literature_review.md` L57) | Protocol §11.9 (L735–740), §15; literature review §12.4 | Results can be misread as answering the README question in general | Already scoped (D040). The report must carry the scope sentence in every claim |
| **M-7** | **MAJOR** (secondary claim) | "Hybrid > all-full is genuine complementarity" | The rival controls are 1-seed (NoPE, SSMax in FEAS, the 2× branch, the dose ladder) or 2-seed (X-gc12). Their difference SD is ≈ 3–4 pp at σ = 2–3 [INF: σ√2], so only confounds of ≳ 8 pp can be excluded. The implicit-position asymmetry is untested. See also m-15: the register's own "genuine" rule is unreachable | Protocol §3.2 controls (L139–149), §11.6; `opchar_secondary.out` U-02 block | "Genuine complementarity" is not establishable. At most: "not explained by the tested controls at ≤ 2 seeds" | Wording rule (pre-run). Optionally run NoPE seed 1 unconditionally (see m-1) |
| **M-8** | **MAJOR** (wording rule) | Operator-level wording (§11.9) | The clause "both families give the same verdict" is decided by an F_gc-family H1 test on **2 seeds (df = 1)**. Its power is 0.27–0.39 raw (0.14–0.20 at Holm's first step) even at a true R = 1.0. `wording()` sets `same = True` whenever both families are "not supported". So the rule licenses operator-level wording exactly when the main family is *not* supported, through uninformative agreement | `analysis/pipeline.py` L1170–1181 (`fgc_family`, Holm over 2); `analysis/verdicts.py` L158–175; arithmetic in §3.2 | An unwarranted operator-level sentence, in exactly the case where the evidence is weakest | Count agreement only if the F_gc family is informative (≥ 3 seeds, or a 90% CI that excludes the support boundary). Otherwise forbid operator-level wording when δ_gc > 3 pp. Pre-run: yes |
| **m-0** | MODERATE | Stated power of H1 / H1-min | The operating characteristics in protocol §11.3 (power 0.88 at n = 5, σ = 3; 0.967 for the 5 → 7 rule) are at **unadjusted one-sided α = 0.05**, but the registered support rule is Holm over 4. Monte Carlo of the family at R₃ = 1.0, σ = 3: **P(H1 supported) = 0.55–0.70 at n = 5 and 0.83–0.92 at n = 7**. At R₃ = 0.9: 0.20 (n = 5), 0.33 (n = 7). VT-18 checks the unadjusted figures only | `debate_record.md` L999 ("All tests are one-sided at α = 0.05"); protocol L67, L675–678; `.audit/adversarial/holm_power.out` (re-run identical) | A higher chance of "inconclusive" than advertised. Seed budgeting rested on these figures | Restate power under Holm (VT-18 extension). Optionally adopt a fixed-sequence order (H1 → H1-min → H5a → H3): H1 keeps full α, and later tests lose power if an earlier one fails. Pre-run: yes (decision row) |
| **m-14** (NEW) | MODERATE | "Degradation at 32K"; H1 "at 16–32K" | The primary cell averages the retrieval ladder over every passing length in {16K, 32K} (protocol L59–62, L642). The per-level H1-robust analysis and the informative-level selection also pool the primary lengths. **No per-length H1 reading is registered.** A k = 3 that recovers 95% at 16K and 65% at 32K yields "H1 supported (16K+32K)" | `analysis/pipeline.py` L975–992 (`cs.level_weights(t, lv, Tp)` at L984) and L938 (`dg.level_weights(t, lv, Tp)`) | A 32K-specific failure, which is exactly the "degradation at 32K" outcome, is hidden inside a pooled support verdict | Register per-length L₃ and R₃ (16K, 32K) as secondary estimands, with a wording rule: if R₃(32K) < 0.8 while the pooled test supports, the sentence reads "supported for 16–32K pooled; not shown at 32K". Pre-run: yes; no new runs |
| **m-12** (UPDATED) | MODERATE | Every k-contrast (RR-R5-power) | A bf16 GDN *training* handicap below 0.115 (50% power) / 0.154 (80%) is not excluded (R5). The PI register says it would "inflate k-contrasts in attention's favour". **That holds for G and Δ₀ (H5a), but for L₃ the direction is indeterminate** [INF]:<br>• if only k = 0 loses δ, L₃ shifts by **+0.2δ** (toward support): R₃ 0.75 → 0.82 at δ = 11.5 pp, G = 30;<br>• if the loss scales with the number of GDN layers, L₃ shifts by **−0.55δ** (toward falsification): R₃ 0.85 → 0.68.<br>The R5 cell is a 2-layer MQAR model, so the mapping to LM retrieval pp is [SPEC] | `results_R5.md` L27–30; `pi_rulings` L515–519, L684; `.audit/adversarial/rr_direction_and_feas_noise.out` (A) | The residual can push H1 either way by up to ≈ 2–6 pp of L₃, against L₃ = 3 pp at R₃ = 0.9. "Attention's favour" is not a safe one-line limitation for H1 | State both directions in the RR-R5-power limitation. Keep the Tier-C fp32-trained k = 0 seed-0 control on the list (it resolves the k = 0-only case). Pre-run: yes (wording) |
| **m-15** (NEW) | MODERATE | Any "genuine" wording (O1, O2, O6) | The register's rule is: an outcome is reported as "genuine only after every rival row for it has been checked and its signature is absent". Several rows cannot be checked within Tiers A–B:<br>• **O1:** E1.8 (LR) needs Tier C #7; E1.2's diagnostic signature needs the unimplemented §9.4 diagnostics.<br>• **O2:** E2.8 (compute) is infeasible.<br>• **O6:** E6.5 needs Tier C #7; E6.8 needs Tier C #6 | `competing_explanations.md` L19, L39, L45, L61, L108, L111; protocol §15 Tier C table (L889–902); `G0.md` L249 (item 20, diagnostics not implemented) | Read literally, "genuine" is unreachable unless Tier C runs. Read loosely, it will be used anyway, and the rule becomes decorative | Amend the rule: "not explained by the checked rivals {list}; unchecked: {list, with reason}". Pre-run: yes |
| **m-16** (NEW) | MODERATE | "Elicited vs zero-shot" (I-27, E1.7, E11.1) | E1.7's second signature, "elicited and zero-shot rankings disagree → budget-conditional", has **no operational definition and no code**. Tier-0 enters the analysis only through the LM-4 slope (the H5a "exposure-robust" label), and the verdict emits a fixed scope sentence. No natural-text counterpart of H1 is registered, although LM-1 (per-position loss on 279 natural 32K documents) is measured for every model | `competing_explanations.md` L44, L152; protocol L521, L739; `analysis/pipeline.py` L85; `analysis/verdicts.py` L177–179; `grep` of `analysis/*.py` for Tier-0 / zero-shot (§3.8) | A disagreement between the synthetic verdict and natural-text long-range loss can be "reported" without consequence. The register promises a label the code cannot produce | Define now, before G5: (a) R_LM1(k) = (gain_k − gain₀)/(gain₁₂ − gain₀) on the LM-1 long-context gain, per seed, as a secondary estimand; (b) a disagreement rule (e.g. R_LM1(3) < 0.5 while H1 is supported → label "synthetic-only"). Pre-run: yes |
| m-1 | MODERATE | NoPE control (E2.6, E6.6) | The single-seed U-06 trigger (\|ΔR₃\| > 0.1 or \|ΔS₁₂\| > 3 pp) **fires 37–83% of the time with no true effect**, because single-seed R₃ noise is large. The arm is close to a coin flip and cannot exclude a PE × k interaction below ≈ 8 pp | `opchar_secondary.out` U-06 block; protocol L733 | NoPE adds a seed on noise, or misses real effects. "Conditional on RoPE" stays the only honest wording | Run NoPE seed 1 unconditionally (≈ 4 GPU-h at D384, §13.2), or state the detection limit. Pre-run: yes |
| m-2 | MODERATE | U-02 conv trigger (I-01) | δ_gc from 2 seeds: P(> 3 pp \| true 0) = 0.07 / 0.16 at σ = 2 / 3. P(miss \| true 4.5–6 pp) = 0.07–0.31 | `opchar_secondary.out` U-02 block; protocol L730–732 | Misrouting of operator wording and of the Tier-B F_gc arms at the boundary | Report δ_gc with its CI. Treat 2–4 pp as "indeterminate" in the wording |
| m-3 | MODERATE | H4 placement null / effect | "No placement effect" needs TOST ±5 pp on both contrasts. Under a true 0 at n = 5 it passes with P = 0.85 (σ = 2), 0.29 (σ = 3), 0.01 (σ = 5). A 3-pp c₁ is detected with P = 0.17–0.37. At k* = 1, near a capability transition, σ is plausibly large or bimodal (PI-SIM bimodal rows) | `opchar_secondary.out` H4 block; protocol L71, L716 | Placement will mostly be "inconclusive"; only effects ≳ 7 pp are found | State the σ-dependence. Consider an estimation-only H4 with MDE reporting. Optional: placement at n₀ + 2 seeds |
| m-4 | MODERATE | FEAS scale rule (U-01) | FEAS-2 (S₁₂ − S₀ ≥ 25 / 20 pp) treats a small gap as infeasibility and escalates to D512 (+58 GPU-h). A small gap caused by a *strong k = 0* (FEAS-1 adequacy met) is the opposite problem: more capacity makes GDN stronger. The outcome "GDN-A ≈ all-full at 32K" is turned into a re-scope instead of a reportable finding | Protocol L833; `configs/experiments/plan.yaml` L244 | A wasted escalation and a lost finding [SPEC: low-moderate probability, since R-load N ≥ 512 should floor k = 0] | Split the FEAS failure branch: FEAS-1 met but FEAS-2 failed with high S₀ → consult the user, and report it as a finding before any D512 spend. Pre-run: yes |
| m-5 | MODERATE | Rivals E1.2, E2.7, E3.6, E6.3 (QK-norm dilution) | The §9.4 attention diagnostics (needle mass, entropy, max logit) and the pathway diagnostics are **not implemented**. Yet FD asks for them on FEAS k = 12, and the rival register relies on them. They are not on the FEAS checklist | `gates/G0.md` L249 (item 20); protocol L502–503, L515; `plan.yaml` L191; FEAS checklist `pi_rulings` L763–791 (absent) | Dilution rivals cannot be discriminated for FEAS-5, degradation at 32K or inversion | Implement before S5. Keep FEAS checkpoints so the diagnostics can run post hoc on k = 12 (inference only) |
| m-6 (UPDATED) | MODERATE | Precision and budget of every confirmatory test | **Full-CS scoring cost is unmeasured.**<br>• S0 extrapolation: ≈ 3–4 GPU-h per checkpoint, against 0.12–0.29 h budgeted.<br>• A sibling audit's contended probe (frac 0.10, GPU at 95–100% util) gives 0.44–0.64 s/context on R1 D3 and **1.78–2.41 s/context on R-load N32**. Each candidate call costs ≈ 7.6–12.0 ms, so candidate scoring (36 candidates × 4 queries) dominates; prefill is only 15–99 ms.<br>FEAS-SMOKE times only `--n-contexts 2` FD cells | `s0_integration.md` L372; protocol §9.9 (L552); `.audit/agent6-schedule/eval_cost_probe.jsonl` (not my measurement; contended); `experiments/prereg/FEAS_smoke.yaml` | A mid-study cut to contexts or suites, decided after data exist, is a forking path | Measure one full CS pass at 32K on a FEAS checkpoint. **Pre-register now** that rank-accuracy candidates (a secondary metric) may be batched or subsampled, so that any cost cut never touches EM_tf, context counts or cells |
| m-7 | MODERATE | H5(a) null wording | The fixed level set is {R1 D 3, 15; R-load N 8, 32}, i.e. ≤ 32 facts, because higher N cannot fit at short T. "No length effect" therefore means "at ≤ 32 facts". If the TEF teaches selective writing, GDN becomes load-limited rather than length-limited [SPEC], and a null would be misread as "attention's benefit doesn't grow with context" | Protocol L72, L474–479 | Misreading of a null | Wording rule. Report the load × length factorial next to H5 |
| m-8 | MODERATE | All k-contrasts (train/eval shift) | TEF queries are **sequential**: queries 2–4 follow earlier Q/A pairs and are led by ".". Evaluation queries are **branched**, each directly after the context. TEF keys come from word list A, evaluation keys from list B. Attention may lean on in-context Q/A demonstrations and GDN on key-specific write policies, so either shift could be k-differential [SPEC]. R6 (T = 1K, 2-layer) would catch only a catastrophic mismatch | `INTERFACES.md` L91; `eval/generators.py` L125–139; protocol L376–377, L450; `reproduction_plan.md` §2.6 | Unknown-sign bias in the k-contrast | Add an exploratory list-A cell (held-out values), and report EM by TEF query position on a TEF-format dev cell. Pre-run: yes |
| m-9 | MODERATE | "k = 1 ≈ k = 12" | Only R(1) ≥ 0.8 is tested. Non-inferiority is registered for S₁₂ − S₃ only | Protocol L78; `analysis/pipeline.py` L955–960; `verdicts.py` L151–154 | An equivalence-style headline would be unregistered | Register S₁₂ − S₁ non-inferiority (5 pp) as a secondary estimand. Pre-run: yes |
| m-10 | MODERATE | Dose and saturation guards | Both are single-seed, with 3-pp and 5-pp thresholds near 1–2 SDs of a single-run difference [INF from σ = 2–3 pp]. The still-closing DC fallback uses a different level set (100 contexts) from the primary cell | Protocol L383, L698–699; `analysis/pipeline.py` L866–868, L894–910 | Spurious or missed "budget-conditional" labels | Report the guards with their noise. Prefer CS at TEF-4,000 (already allowed) |
| m-11 | MODERATE | H1-min (k = 1) | Abrupt single-layer circuit formation predicts bimodal seeds. PI-SIM: no few-seed test is valid (0.09–0.30) | Protocol L679; `debate_record.md` L322 | The H1-min verdict is unreliable under bimodality | Already handled by success counts. Make them co-reported with H1-min in the verdict sentence |
| m-13 (UPDATED) | MODERATE | Measured cost frontier | **Compliance.** Session 1 meets no strict G-cost row (resident CARLA client at 0% SM), and session 2 is owed.<br>**Representativeness.** At d = 384 the attention/dense FLOP ratio at 32K is 6.56, against 1.23 at d = 2,048. Decode at B = 1 is launch-bound (roofline fraction ≈ 0.07 for C0).<br>**Code hash.** `models/attention.py` changed in `0c54713`. The change is the opt-in R-ATT fp32 reference path; production FLASH/cuDNN paths are unchanged by its docstring. But D087 validates costs "for the code hash they record" | `cost_benchmarks_session1.md` L8, L29–32, L47; protocol L573–583 (projection table); D087 (`decisions.md` L96); `git diff 490798d HEAD -- src/hybridlc/models/attention.py` | Measured "speed-ups" are implementation- and scale-specific. Session-1 rows need re-measurement or a logged hash-equivalence reading | Keep analytic costs primary (done). Never present measured speed-ups without the projection table. Treat B_max-limited throughput as the only informative measured axis (§2, cost row) |
| n-1 | MINOR | FEAS → core transfer | FEAS Stage A is 0.30B tokens (`plan.yaml` L188: 300,023,808), half the core's 0.6B. Yet `reproduction_plan.md` L555 says "0.6B tokens" for FEAS end-of-Stage-A. FEAS-3, FEAS-5 and U-08 are decided on less-trained models | As cited | Pilot-to-main transfer; doc inconsistency | Correct the doc. Note the transfer caveat in the FEAS report |
| n-2 | MINOR | H5 "exposure-robust" label | LM-4 copies 256-token passages, plausibly beyond GDN-A's effective copy capacity at any gap [SPEC], so gain₀(g) may be ≈ 0 at all g ≥ 512 and Λ(g) flat. The label's sign would then be noise | Protocol L524 (LM-4) | The label is uninformative | Report Λ(g) with its CI; withhold the label if the CI spans 0 |
| n-3 | MINOR | H1 falsification rival E2.4 | Placement at k = 3 is tested only if k* = 3. When H1 is falsified, k* is likely 1 or 2 | Protocol L131; `competing_explanations.md` L57 | One falsification rival is untested | State it |
| n-4 | MINOR | Wording | "Multi-hop reasoning" is templated pointer-chasing with exact key matches (h ≤ 4 in evaluation, ≤ 6 in TEF) | `eval/templates.py` L46–49; protocol L375 | Overclaim | Use "pointer-chasing (h links)" |
| n-5 | MINOR | Primary-cell selection | G-REF selection on dev items shares seed noise with the test-item estimate. This is conservative for support and anti-conservative for falsification near the threshold (debate I-06 already notes it) | `debate_record.md` L392 | Small bias near the threshold | Report the per-seed adequacy margin |
| n-6 (NEW) | MINOR (process) | Measured costs (RQ-D) | **P003 adopted the permissive "activity reading" of G-cost after session 1 had run** and after the strict reading was known to exclude every row. This is the move PR-04 rejects ("Choosing the definition that passes, after both passes were seen, is a restatement"). The reading is outcome-blind with respect to the cost values and arguably sensible, and the strict reading will be reported as a sensitivity | `decision_log.md` L9; `cost_benchmarks_session1.md` L30, L32; `pi_rulings` L219, L194 | An inconsistently applied integrity standard, which a reviewer will notice | Re-file P003 as a logged deviation (DEV-xx), not an "interpretation". Make session 2 strictly compliant (stop the idle CARLA client) so the primary measured costs do not rest on the post-hoc reading |
| n-7 (NEW) | MINOR | Reference competence at 32K (feeds M-1) | FEAS-5 (SSMax, one paired difference on seed 101, adopt iff ≥ 3 pp at 32K) and FEAS-3 (Stage-B budget, 2-seed mean, thresholds 3 / 10 pp) are study-wide choices made near single-run noise.<br>• FEAS-5: P(adopt \| true 0) = 0.14 / 0.24 and P(reject \| true +5 pp) = 0.24 / 0.32 at σ = 2 / 3.<br>• FEAS-3: P(smallest \| true +6 pp) = 0.07 / 0.16 | Protocol L327, L834–835; `.audit/adversarial/rr_direction_and_feas_noise.out` (B) | A mis-set SSMax or budget lowers S₁₂ at 32K for every k. That is not a direct k-bias, but it feeds M-1 | Report both decisions with their noise. The 2× branch and G-FULL partly protect |

---

## 2. Outcome by outcome: the strongest alternative explanation, and whether the design can rule it out

**Reading the "rule out?" column:**
- **Yes:** a pre-registered check with adequate power exists.
- **Partly:** a check exists but is underpowered or covers only part of the rival.
- **No:** nothing in Tiers A–B discriminates it.

| Expected finding | Strongest alternative explanation | What the design does | Rule out? | Gap and fix |
|---|---|---|---|---|
| **H1 supported** (L₃ > 0) | **Handicapped reference:** C12 passes 0.8·S₁₂(no-filler) but is beaten by C3 at 16–32K. Possible causes: thin Stage B; d_h = 64 dilution; RoPE positions 4K → 32K seen only in Stage B and the TEF; small-scale all-full deficit. Then L₃ > 0 holds trivially.<br>*Second:* literal single-token copy (M-2).<br>*Third:* k = 0 under-elicited at ≈ 1.7K TEF items (M-3, M-9).<br>*Fourth:* an undetected bf16 GDN handicap at k = 0 only, which adds +0.2δ to L₃ (m-12) | G-REF adequacy; 2× branch (seed 0); SSMax (FEAS, seed 101); NoPE (seed 0); X-gc12 (2 seeds); still-closing flag on S₁₂ − S_k | **Partly** | Adequacy excludes collapse, not handicap. Inversion is defined only for C12 failing (a). R_best ≡ R by default (M-1). The k = 0 learning curve is unchecked (M-3), and the dose span is 4× (M-9). Fixes 1–3 in §4 |
| **H1 falsified** (L₃ < 0 with guards) | k = 3 still learning; long stage under-trained; centred placement poor at k = 3; few RoPE layers under-trained at long range; LR chosen on 4K loss disfavours hybrids; a per-GDN-layer bf16 handicap (−0.55δ on L₃, m-12) | Dose guard (seed 0: L₃ < 0 at 4,000 **and** a plateau); saturation guard (seed 0); NoPE (seed 0); Tier-C η*/√2 | **Mostly** for learnability and budget, since both guards are mandatory. **No** for placement unless k* = 3 (n-3). LR only in Tier C | Falsification is appropriately hard to reach. The residuals are single-seed guards (m-10) and m-12's falsification direction. The report should state the asymmetry: support needs only the test |
| **k = 1 ≈ k = 12** (H1-min supported) | (i) The task is literal copy: one attention layer plus conv "smeared keys" suffices by mechanism (Olsson [per A1, B2]; Wen et al. [abstract-only per lit. review]).<br>(ii) Ceiling compression for every k ≥ 1.<br>(iii) Handicapped reference (M-1).<br>(iv) The test establishes R(1) ≥ 0.8, not equivalence.<br>(v) A 16K-driven pooled result (m-14) | H1-robust per-level R (pooled over lengths); N50 and N/N50₀; informative levels (S₁₂ in 20–90%) | **No** for (i), (iv) and (v); **partly** for (ii) | Hard-negative cell (M-2); S₁₂ − S₁ non-inferiority (m-9); per-length R (m-14); success counts beside the verdict (m-11) |
| **Hybrid > all-full** | Missing conv; RoPE long positions; dilution; Stage-B budget; TEF learning speed; LR; GDN's implicit position/recency source (no arm); small-scale all-full deficit (Qiao [read-section, PI-VER in debate]) | X-gc12 (2 seeds); NoPE, SSMax, 2× and dose (1 seed each); inversion and R_best | **Partly.** Only confounds of ≳ 8 pp can be excluded (M-7). The inversion label does not fire if C12 is adequate (M-1). "Genuine" is unreachable by the register's rule (m-15) | Wording: "not explained by the tested controls (≤ 2 seeds)". DG on all core configurations (fix 1) |
| **Degradation at 32K** (all k lower at 32K than at 16K) | Under-trained long relative distances (STRING 2410.18745, `competing_explanations.md` L109 [read-section per PROP]: Stage B is 30–90M tokens, and relative distances ≥ 24K are a small fraction of trained pairs); QK-norm dilution (needle weight 0.64 → 0.05 from 1K to 32K at unit gains [MEAS-B], D013); needle–query **distance** grows with T; 32K is 1/7 of TEF answer weight; evaluation numerics at 32K | 2× branch (1 seed); SSMax (FEAS); NoPE (1 seed); R10 per-position loss (FEAS); VT-22b v2 (C12 fp32 vs Qwen3 at 16K/32K) and FEAS-PREC v2 P4 (numerics); R-dist (32K only); C2 oracle position (4K vs 32K, MS models) | **Partly.**<br>• Numerics: yes, if VT-22b v2 and FEAS-PREC v2 pass.<br>• Dilution: **no**, until the diagnostics exist (m-5).<br>• Distance vs length: only at 32K, and only for C0/C12 (2 seeds) and C1/C3 (seed 0). C2 separates "long prefix" from "distance" on MS models only.<br>• A k-specific 32K failure can be **hidden by 16K+32K pooling** (m-14) | Implement the diagnostics. Register per-length L₃/R₃ (m-14). Report distance-stratified scores at every T: depth and distance are recorded per query (protocol L436, §9.8) |
| **H3 supported** (R_ret(3) > R_hop(3)) | Copy (1 lookup) vs compose (h + 1 lookups, no chain of thought) plus a load mismatch (≤ 20 vs ≤ 1,024 facts). The ladder mix is arbitrary. Ratio noise: per-seed SD(D₃) of 0.11–0.30 | Equated-level sign check (S₁₂ only); GLMM and Fieller; M-order; Tier-C adjacent vs spread | **No** for copy vs compose; **partly** for difficulty (equates S₁₂, not S₀ or load) | Load-matched pair (M-4). Wording |
| **H3 null / not evaluable** | Multi-hop near floor for every k (G_hop < 15 pp), so not evaluable; TOST unattainable | Evaluability rule; TOST ±0.10 | A null is **unreachable** (F-2). "Not evaluable" is likely if G_hop ≲ 20 pp (P(evaluable) = 0.53 at σ = 3) | Make the estimation statement primary when not evaluable |
| **H5 supported** (gap grows with log₂T) | k = 0 near ceiling at short T (probability-scale compression); "length" = filler amount × distance; reference changing with T | Fixed level set (removes ladder composition, debate §6.4); equal-count TEF plus block normalisation (DEV-04, D063); G-REF label (D066); LM-4 corroboration; R-dist; load × length; C2 oracle | **Mostly** for exposure and ladder composition; **partly** for distance vs filler (32K only) | Add a logp_gold-scale slope as a "metric-robust" check, as H1 has. Report per-architecture R-dist (planned). Note that the H5(a) bias from RR-R5-power is in attention's favour (m-12) |
| **H5 null** | TEF-taught selective writing makes GDN load-limited, not length-limited [SPEC]; the low-load level set cannot show a load-dependent length effect | TOST ±2 pp per doubling | Partly | Scope the wording to "≤ 32 facts" (m-7) |
| **Placement effect** | Feature availability at layer 1 (only one GDN layer below); a final-layer copy shortcut at layer 11; seed noise | c₁, c₂ with Holm; spacing preserved; layer 0 avoided | **Partly:** only effects ≳ 7 pp are reliably detected (m-3) | MDE reporting |
| **Placement null** | Low power | TOST ±5 pp on both contrasts | Only if σ ≤ 2 pp (P = 0.85; 0.29 at σ = 3) | State the σ-dependence |
| **GDN-E recovers the gap** | ≈ 3–4× recurrent state (H·d_k·d_v ×4 / ×3), not only per-head d_k; 1 head vs 3–4; 2 unpaired seeds | Welch on 2 vs n seeds; TOST ±5 pp | **No** (F-1) | Estimation-only, or n₀ paired seeds |
| **Cost frontier** (hybrids dominate) | For any per-layer-additive cost, M(k) = G·(R(k) − k/12) is an identity, so "dominance" is just R(k) > k/12. At 32K the state axis is fixed by arithmetic: C1 already holds 49.2 MiB against C0's 1.27 MiB at D384 (protocol L193–194). Measured differences reflect kernel maturity (GDN chunk core roofline 0.056–0.110 at B = 1 vs FLASH 0.28–0.97), launch overhead at B = 1, and a d = 384 FLOP mix unrepresentative of production | Analytic primary (exact); measured secondary with graph capture, cold L2, min-of-N and a roofline-fraction table | The analytic part: **yes**, but it adds nothing beyond R(k). The measured part is implementation-scoped. Only B_max-limited throughput is non-additive and informative: at 32K, C0 92,590 vs C3 5,226 vs C12 1,314 tok/s (`cost_benchmarks_session1.md` L58–63) | Present the frontier as a re-plot of R(k) plus the B_max throughput (m-13). Do not let P003's post-hoc reading carry primary measured claims (n-6) |
| **Aggregation insensitive to k** | Every model sits at the recency or short-window floor: 0.651–0.777 ladder mean at 32K | Recency and window-1,024 floor columns | **No**, unless scores clear the last-8,192 floors at ρ ≤ 1.5 (F-3) | Fix 9 in §4 |

---

## 3. Quantitative checks (CPU-only; all reproducible)

All scripts and outputs are in `/home/sedrica-lynx/Downloads/outeropt/.audit/adversarial/`, run with `python -I` from the project venv. The models mirror `src/hybridlc/analysis/stats.py` (Welch, TOST, one-sample t) and PI-SIM's assumptions (debate §6.2: G = 30 pp, seed effect 2 pp, residual σ ∈ {2, 3} pp).

### 3.0 Re-verification of the 15:43 draft's scripts (NEW)

I re-ran all four scripts into `rerun/` and diffed them against the saved outputs. Command and output:

```
python -I holm_power.py > rerun/holm_power.out && diff holm_power.out rerun/holm_power.out && echo HOLM_IDENTICAL          -> HOLM_IDENTICAL
python -I afreq_windows.py > rerun/afreq_windows.out && diff …                                                           -> AFREQ_IDENTICAL
python -I key_overlap.py > rerun/key_overlap.out && diff …                                                               -> KEY_IDENTICAL
python -I opchar_secondary.py > rerun/opchar_secondary.out && diff …                                                     -> OPCHAR_IDENTICAL
```

I also checked the scripts' logic:
- They implement the registered tests: one-sample/paired t; Welch for H6′; TOST with α = 0.05; Holm step-down over 4.
- Item audits read the frozen `experiments/items/test/CS/*.npz` through `hybridlc.eval.items.load_items`.
- Pending VT-24r regeneration changes only value digits (`pi_rulings` L406–411), so the key-overlap and A-freq results carry over.

### 3.1 H1 power under the registered Holm rule (`holm_power.py`)

Analytic power of the per-seed L₃ t-test:

| σ | R₃ | n = 5: α .05 / .0167 / .0125 | n = 7: α .05 / .0167 / .0125 |
|---|---|---|---|
| 3 | 1.0 | 0.88 / 0.63 / 0.55 | 0.97 / 0.87 / 0.83 |
| 3 | 0.9 | 0.42 / 0.20 / 0.16 | 0.56 / 0.32 / 0.27 |
| 2 | 1.0 | 0.99 / 0.93 / 0.88 | 1.00 / 1.00 / 0.99 |

**Full Holm family, Monte Carlo** (10⁵ replicates). Scenario: H3 not evaluable (p = 1), H1-min false (R₁ = 0.5), H5a either strong (p ≈ 0) or absent. P(H1 supported):

| σ | n | R₃ = 1.0, H5a strong | R₃ = 1.0, H5a absent | R₃ = 0.9, H5a strong |
|---|---|---|---|---|
| 3 | 5 | 0.630 | 0.554 | 0.201 |
| 3 | 7 | 0.871 | 0.827 | 0.325 |
| 2 | 5 | 0.927 | 0.885 | 0.400 |

These compare with the protocol's 0.88 / 0.967 (§11.3), which are unadjusted (`debate_record.md` L999).

### 3.2 Secondary tests and triggers (`opchar_secondary.py`)

- **H6′.** Median Welch df ≈ 2.6–2.8.
  - P(TOST equivalent | Δ = 0): 0.14 / 0.28 (n = 5, σ = 3 / 2); 0.10 / 0.22 (n = 4).
  - P(p < .05 | Δ = 10 pp): 0.50–0.62.
- **H4** (n = 5). P(no-placement-effect | c = 0) = 0.85 / 0.29 / 0.01 at σ = 2 / 3 / 5. P(detect c₁ = 3 pp) = 0.37 / 0.17.
- **U-02.** P(δ̂ > 3 | δ = 0) = 0.07 / 0.16. P(δ̂ > 3 | δ = 6) = 0.94 / 0.84 (σ = 2 / 3).
- **U-06.** P(fires | no effect) = 0.37–0.83, depending on σ and G.
- **H3.**
  - P(evaluable) = 0.03 (G_hop = 15), 0.53–0.82 (G_hop = 20), ≈ 1.0 (G_hop = 30).
  - Power at D₃ = 0.2: 0.63–0.90 when G_hop = 30; 0.28–0.63 (unconditional) when G_hop = 20.
  - P(TOST equivalent & evaluable | D₃ = 0) ≤ 0.25 (best case n = 7, σ = 2, G_hop = 30: 0.246); ≤ 0.04 at σ = 3; ≤ 0.07 at G_hop ≤ 20.
- **F_gc family H1 at 2 seeds** (analytic, df = 1; per-seed SD_L = σ·√1.68). P(one-sided p < .05 | R = 1.0) = 0.27 / 0.39 at σ = 3 / 2. P(p < .025) = 0.14 / 0.20. At R = 0.8 the type-I rate is nominal.

### 3.3 Item audit: literal copy and key overlap (`key_overlap.py`; frozen `experiments/items/test/CS/`)

| Cell | Prompt is a literal copy of the fact prefix | Another fact shares the noun | Another fact shares the adjective |
|---|---|---|---|
| R1 D3 32K | 1.000 | 0.000 | 0.023 |
| R1 D15 32K | 1.000 | 0.013 | 0.096 |
| R-load N32 32K | 1.000 | 0.018 | 0.195 |
| R-load N128 32K | 1.000 | 0.078 | 0.554 |
| R-load N512 32K | 1.000 | 0.301 | 0.964 |
| R-load N1024 32K | 1.000 | 0.509 | 1.000 |
| R-load N1024 16K | 1.000 | 0.493 | 0.998 |

- Word list B has 154 adjectives × 1,390 nouns = 214,060 keys (list A: 158 × 1,369), from `hybridlc.eval.words.word_list`.
- **Analytic cross-check** [INF]: with keys drawn without replacement, P(noun shared) ≈ 1 − (1 − 1/1,390)^(N−1), which gives 2.2 / 8.7 / 30.8 / 52.1% at N = 32 / 128 / 512 / 1,024 (measured 1.8 / 7.8 / 30.1 / 50.9%).
- The rates depend on N, not on T, which is why 16K and 32K agree.

### 3.4 A-freq short-memory floors (`afreq_windows.py`; frozen CS items)

| T, ρ | Recency | Last-1,024 | Last-4,096 | Last-8,192 |
|---|---|---|---|---|
| 32K, ρ = 4 | 0.791 | 0.721 | 0.926 | 0.987 |
| 32K, ρ = 2 | 0.667 | 0.609 | 0.729 | 0.832 |
| 32K, ρ = 1.5 | 0.613 | 0.548 | 0.641 | 0.694 |
| 32K, ρ = 1.25 | 0.534 | 0.514 | 0.537 | 0.595 |
| **32K ladder mean** | **0.651** | 0.598 | **0.708** | **0.777** |
| 16K, ρ = 2 | 0.662 | 0.646 | 0.845 | 0.941 |

The 32K recency and 1,024 / 4,096 values agree with `s0_integration.md` L359 (0.791 / 0.721 / 0.926 for ρ = 4; 0.667 / 0.609 / 0.729 for ρ = 2), which validates the script. Ladder means are my arithmetic on the rows above.

### 3.5 TEF exposure counts (NEW; `tef_counts.py`, `tef_counts.out`)

`TEFStream(seed_index=0, steps=1000).plan_counts(0)` runs CPU-only and generates no items. It reports `items_per_bucket() = 907` and the following per bucket (every bucket from 512 to 32,768 is the same within ±1):

| Family | Items per bucket | Sub-types |
|---|---|---|
| Retrieval | 302–303 | R1 121, R-load 121, R3 30–31, R-dist 30 |
| Multi-hop | 302–303 | forward 76, shuffled 151, reversed 75–76 |
| Aggregation | 302–303 | A-freq 211–212, A-count 91 |

Per 1,000-step block this gives:
- **1,694 R1 + R-load items**, ×4 queries ≈ **6.8K supervised retrieval answers**;
- **242 R-load items at 16K + 32K** combined;
- with N log-uniform in [4, cap], P(N ≥ 512 at 32K) = ln 4 / ln 512 ≈ 0.22 [INF], i.e. ≈ 27 high-load R-load items at 32K per block.

### 3.6 Direction of RR-R5-power on L₃ (NEW; `rr_direction_and_feas_noise.py` part A)

G = 30 pp, S₀ = 40. The table gives L₃ (pp) and R₃ before → after an undetected k = 0 handicap δ:

| True R₃ | δ | "k = 0 only" (+0.2δ on L₃) | "per GDN layer" (−0.55δ on L₃) |
|---|---|---|---|
| 0.75 | 6 | L₃ −1.50 → −0.30; R₃ → 0.792 | L₃ → −4.80; R₃ → 0.667 |
| 0.75 | 11.5 | L₃ → +0.80; **R₃ → 0.819** | L₃ → −7.83; R₃ → 0.611 |
| 0.85 | 11.5 | L₃ +1.50 → +3.80; R₃ → 0.892 | L₃ → −4.83; **R₃ → 0.684** |

δ = 11.5 pp is R5's 50%-power MDD expressed in pp. Mapping it from a 2-layer MQAR model to LM retrieval is [SPEC].

### 3.7 FEAS decision noise (NEW; part B)

Normal approximation. A single paired difference has SD σ√2; a 2-seed mean has SD σ.

| σ | FEAS-5: P(adopt \| true 0) | P(not adopt \| true +5) | FEAS-3: P(not smallest \| true 0) | P(smallest \| true +6) | P(not largest \| true +12) |
|---|---|---|---|---|---|
| 2 | 0.14 | 0.24 | 0.07 | 0.07 | 0.16 |
| 3 | 0.24 | 0.32 | 0.16 | 0.16 | 0.25 |

### 3.8 Code audit (read-only; HEAD `7f8bed3`)

`git log -- src/hybridlc/analysis/pipeline.py src/hybridlc/analysis/verdicts.py` shows last changes at `656425f` (11-10 12:08), so every code citation is on the same code as the 15:43 draft.

| Topic | What the code does | Where |
|---|---|---|
| Reference inversion | Returns `triggered=False, reason="C12 adequate"` whenever `gref["a_pooled"]` | `pipeline.py` L231–232 |
| R_best | Takes the argmax over whatever configurations DG contains. Protocol L509 scores DG on C0 and C12 only by default | `pipeline.py` L1001–1014 |
| Operator wording | `same = (fv == cv)` compares "supported / not supported" labels, so two "not supported" families count as agreement. The F_gc family runs H1 on X-gc{1,3} vs X-gc12 with the seeds available (2 by design), Holm over 2 | `verdicts.py` L158–175; `pipeline.py` L1170–1181 |
| Still-closing | The gap is `C12 − C{k}` only | `pipeline.py` L885–886 |
| **Per-level H1 (NEW)** | `cs.level_weights(t, lv, Tp)` pools the primary lengths; informative levels likewise (`dg.level_weights(t, lv, Tp)`). There is no per-length output | `pipeline.py` L984, L938 |
| **Tier-0 (NEW)** | `grep -n -i 'zero.shot\|tier0\|Tier-0' src/hybridlc/analysis/*.py` finds only the P17 figure, the LM-4 slope helper and the fixed scope sentence | `figures.py` L13, L593, L608; `pipeline.py` L85; `verdicts.py` L177–178 |

### 3.9 Evaluation-cost probe (sibling audit's raw data; not my measurement)

`.audit/agent6-schedule/eval_cost_probe.jsonl` was recorded with `frac 0.10`. `nvidia-smi` showed 95–100% util and ~12.5 GB used during the probe, so it is contended. Per context:

| Config | T | prefill_ms | cand_call_ms | R1 D3 s/ctx | R-load N32 s/ctx |
|---|---|---|---|---|---|
| C0 | 4K / 16K / 32K | 16.0 / 26.8 / 72.7 | 11.9 / 12.0 / 11.8 | 0.46 / 0.49 / 0.52 | 1.87 / 1.89 / 1.92 |
| C3 | 4K / 16K / 32K | 14.9 / 28.5 / 97.2 | 11.9 / 11.6 / 11.9 | 0.47 / 0.44 / 0.64 | 1.81 / 1.86 / 2.41 |
| C12 | 4K / 16K | 16.6 / 99.0 | 7.9 / 7.8 | 0.44 / 0.51 | 1.78 / 1.77 |

The schedule audit owns the budget consequences. My point is narrower (m-6): pre-register now which secondary computation (candidate scoring) may be reduced, before any data exist.

---

## 4. Recommendations, ranked by value per cost

None of these is applied. Each needs a PI decision row; changes to frozen protocol text need a DEV row. Every one is **pre-run**: no affected item has run and FEAS has not started, so the integrity rule (`pi_rulings` §0 rule 1, L32–37) permits them as logged changes.

1. **Close the reference-handicap blind spot (M-1).**
   - Score DG on C1 and C3, and on the placement arms if run, with EM only and candidate scoring off.
   - Add a **label-only** rule, "reference exceeded by C{k}": a hybrid's paired primary-cell score exceeds S₁₂ by ≥ 5 pp with paired p < 0.05.
   - Print R_best, now non-vacuous, in every H1 and H1-min verdict sentence.
   - *Consequence:* the H1 test is unchanged; the wording becomes more honest; small evaluation cost.
2. **Make the elicitation regime explicit and testable (M-3, M-9).**
   - Extend still-closing to the support direction: label "budget-conditional" if sign(L_k) changes or |ΔR(k)| > 0.1 between TEF 1,000 and 4,000 (seed 0; no new runs).
   - State the TEF budget (≈ 1.7K retrieval items) and the 4× dose span in every H1 / H1-min sentence.
   - Optional Tier C: seed-0 C0/C3/C12 to TEF 16,000 (≈ 7 GPU-h idle), with the same reading pre-registered.
   - *Consequence:* possibly more conservative labels; no change to the test.
3. **Register per-length H1 readings (m-14).** L₃ and R₃ at 16K and 32K separately, as secondary estimands, with the wording rule of m-14. *Consequence:* no new runs; a 32K failure becomes visible.
4. **Add a hard-negative retrieval diagnostic (M-2).**
   - At 16K and 32K: ≥ 50% of queried keys get a distractor that shares the adjective or the noun.
   - Exploratory, BH-FDR. Evaluated on all core models.
   - *Consequence:* separates single-token copy from 2-token binding. New manifest entries before G5; CS is unchanged.
5. **Restate H1 power under Holm (m-0).** Then either accept it explicitly, or adopt a fixed-sequence order (H1 → H1-min → H5a → H3). *Consequence:* at n = 5, σ = 3, R = 1.0, H1 power returns from ≈ 0.55–0.70 to 0.88. Later hypotheses are tested only if earlier ones are supported.
6. **Fix the operator-wording clause (M-8).** Count family agreement only when the F_gc family has ≥ 3 seeds or a CI that excludes the support boundary. *Consequence:* fewer and better-founded operator-level sentences.
7. **Define the Tier-0 disagreement rule and a natural-text H1 counterpart (m-16), and repair the "genuine" rule (m-15).** *Consequence:* the register's promises become executable; no new runs.
8. **H6′ (F-1).** Make it estimation-only, or fund n₀ paired seeds. Correct "equal H·d_k" to say the arm has ≈ 3–4× the recurrent state and 1 head. *Consequence:* removes an unreachable verdict, or costs ≈ 5–8 GPU-h.
9. **H3 and aggregation (F-2, M-4, F-3).**
   - Declare H3's negative outcome unreachable.
   - Add an exploratory load-matched pair (R-load N = 4(h + 1) vs M-hop h). Lead with "not evaluable" when that applies.
   - Integration wording only if the 32K A-freq score at ρ ≤ 1.5 exceeds the last-8,192 floor by a pre-stated margin. Add 8,192 to `AFREQ_WINDOW_FLOORS`.
10. **State both directions of RR-R5-power (m-12)** in the limitations. Keep the Tier-C fp32-trained k = 0 control listed.
11. **FEAS branch split (m-4).** If FEAS-1 passes but FEAS-2 fails with high S₀, consult the user before any D512 spend, and record the result as a finding.
12. **Implement the §9.4 attention and pathway diagnostics (m-5)** before S5. Run them post hoc on FEAS k = 12 checkpoints, as FD specifies.
13. **Evaluation cost (m-6).** Measure one full CS pass on a FEAS checkpoint, and pre-register the candidate-scoring reduction rule now.
14. **Wording rules for the final report** (M-5, M-6, M-7, m-7, n-4):
    - "literal-copy retrieval";
    - "templated pointer-chasing";
    - "no length effect at ≤ 32 facts";
    - "hybrid advantage not explained by controls at ≤ 2 seeds";
    - the compute-vs-memory confound and the two uncontrolled mixer asymmetries named.
15. **Lower-priority additions:**
    - a non-inferiority estimand for S₁₂ − S₁ (m-9);
    - a list-A key cell and a TEF query-position check (m-8);
    - NoPE seed 1 unconditionally, or state the ≈ 8 pp detection limit (m-1);
    - re-file P003 as a DEV and make session 2 strictly compliant (n-6);
    - report FEAS-3/5 with their noise (n-7).

---

## 5. What is sound (no change recommended)

- **Paired-seed inference on L_k,** with common random numbers (shared data order and shared-module initialisation). The choice of paired over pooled inference is validated by PI-SIM. I independently reproduced the unadjusted figure: 0.88 at n = 5, σ = 3, α = .05 matches PI-SIM's 0.880.
- **Fixed level set for H5,** which removes the ladder-composition artefact (debate §6.4). Block-normalised TEF loss (DEV-04) restores length balance, and §3.5 confirms equal item counts per bucket (±1).
- **Independent dev items for gating;** no selection on test items.
- **Falsification guards** (dose and saturation) make a false "falsified" unlikely.
- **Analytic state bytes** are exact (VT-14-style check, 72/72). Measured costs are correctly demoted to secondary and appendix status.
- **The reproduction ladder (R1 → R4/R5 → R6)** tests the chain "their harness / our mixer / our harness". R7 cleared the scorer item by item against lm-eval.
- **k-differential residuals** are explicitly named (D090), with safeguards that can see a between-k difference (FEAS-PREC v2 Q1–Q4). The one correction is m-12's direction.
- **Verdict vocabulary** (D088) and **hashed pre-registrations** (D091) prevent relabelling. n-6 is the one place where the same standard was not applied.
- **The R-ATT change** (`0c54713`) is opt-in and leaves the production attention paths untouched, per its docstring and diff.

---

## 6. Not verified, and limits of this review

- **Literature claims** were not re-opened by me. Access tags are those recorded in the project files.
- **Direction of M-2's bias.** Single-token sufficiency also helps the GDN, whose conv(4) can supply 2-token key features. My claim is that the task format makes one attention layer sufficient by a known mechanism, not that it biases the k-contrast in a known direction [SPEC on direction].
- **Simulation assumptions.** Normal residuals with σ ∈ {2, 3} pp and G = 30 pp follow PI-SIM's assumptions. True σ, G and G_hop are unknown until FEAS and S4. Bimodality would make every few-seed result worse.
- **Untested speculation.** m-8 (train/eval shift being k-differential) and n-2 (LM-4 floor) are speculative.
- **m-12's magnitudes** map R5's MQAR accuracy MDD onto LM retrieval percentage points. Only the sign analysis is firm.
- **Evaluation cost (m-6)** rests on the S0 extrapolation plus a sibling's contended probe. I did not time anything.
- **Code.** I did not run the analysis pipeline end to end. The code-path findings (M-1, M-3, M-8, m-14, m-16) come from reading `pipeline.py` and `verdicts.py` at HEAD `7f8bed3`. These files are unchanged since `656425f`, and the concurrent workflow's working-tree edits do not touch them (`git status` at 20:56).
- **Sibling audits** (design, evaluation, implementation, schedule) were still running. I read only one sibling raw-data file (§3.9) and none of their conclusions, so overlaps such as Holm power or evaluation cost are independent.

## 7. Corrections to the 15:43 draft (made in this pass)

| Draft text | Correction | Basis |
|---|---|---|
| "`reports/STATUS.md` L100–104" (pre-run status) | `STATUS.md` L5–8. The file was restructured at `150d071` | `cat -n reports/STATUS.md` |
| Family weighting "protocol L44–48" | L53–56 (L44 is RQ-D) | `sed -n` of the protocol |
| LM-4 "protocol L578–579" | L524 (L578–579 is the projection-table header) | same |
| Projection table "L567–574" | L573–583 | same |
| H6′ "L718–721" | L717–720 (L721 is H2) | same |
| `reproduction_plan.md` L554 | L555 | `sed -n '552,556p'` |
| debate "L387–392" | L385–392 (statement at L392) | `grep -n` |
| A-freq short-memory ladder mean "≈ 0.65–0.71" | 0.651 (recency) / 0.708 (last-4,096) / 0.777 (last-8,192) | §3.4 arithmetic |
| m-12: RR-R5-power "inflates every k-contrast in attention's favour" (taken from the PI register) | True for G and Δ₀; indeterminate for L₃ (+0.2δ or −0.55δ) | §3.6 |
| m-13: "`models/attention.py` has since changed" | The change is the opt-in R-ATT path; production paths are unchanged, but D087's hash rule still applies | `git diff 490798d HEAD -- src/hybridlc/models/attention.py` |
| F-2 "≤ 0.05 for σ = 3 or G_hop ≤ 20" | ≤ 0.04 at σ = 3, but ≤ 0.07 at G_hop ≤ 20 (n = 7, σ = 2, G_hop = 20 gives 0.070). The draft's best-case bound "≤ 0.25" stands (0.246) | `grep 'true D3=0.0' opchar_secondary.out` |

## Appendix: commands

```bash
source /home/sedrica-lynx/Downloads/outeropt/.venv/bin/activate
cd /home/sedrica-lynx/Downloads/outeropt/.audit/adversarial
python -I opchar_secondary.py | tee opchar_secondary.out          # H6', H4, U-02, U-06, H3
python -I holm_power.py | tee holm_power.out                      # H1 power under Holm
python -I key_overlap.py | tee key_overlap.out                    # literal copy / key-token overlap (frozen CS items)
python -I afreq_windows.py | tee afreq_windows.out                # A-freq window and recency floors (frozen CS items)
python -I tef_counts.py | tee tef_counts.out                      # NEW: TEF block-plan counts (CPU, no item generation)
python -I rr_direction_and_feas_noise.py | tee rr_direction_and_feas_noise.out   # NEW: RR-R5-power direction; FEAS noise
mkdir -p rerun && for s in holm_power afreq_windows key_overlap opchar_secondary; do
  python -I $s.py > rerun/$s.out && diff $s.out rerun/$s.out && echo "$s IDENTICAL"; done
```

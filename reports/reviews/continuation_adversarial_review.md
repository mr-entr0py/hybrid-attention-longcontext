# Continuation adversarial review (Agent 5): can the programme establish what it claims?

**Date:** 2026-11-10. **Role:** adversarial scientific reviewer (NeurIPS-level sceptic) of the whole programme as it stands: question, controls, hypotheses, tasks, the TEF fine-tune, statistics, cost comparisons and contribution.
**Mode:** report-only. I edited nothing in `src/`, `scripts/`, `tests/`, `configs/` or `experiments/`. I wrote only this file and scratch files under `.audit/adversarial/`.
**GPU:** none used. Every computation below is CPU-only numpy/scipy, plus read-only loads of frozen item files.

**Read in full or in the relevant sections:**
- `reports/STATUS.md`, `continuation_audit.md`, `experimental_protocol.md` (v1.0, all 20 sections), `decisions.md` (D000–D091, DEV-01…04), `pi_rulings_2026-11-10.md` (§0–§2, §7–§10, §13), `competing_explanations.md`;
- `reviews/debate_record.md` (I-01…I-29, §3, §6.2–6.4), `literature/literature_review.md`;
- `reproduction/results_R5.md`, `results_R7.md`, `reproduction_plan.md` (§2.6, §2.9, §2.10, §4.2–4.3);
- `gates/G0.md` (§1, §7, §8, §9), `s0_integration.md` (§3, §11, §12, §13), `cost_benchmarks_session1.md` (summary);
- `src/hybridlc/INTERFACES.md`, plus the analysis, eval and data code cited below.

**Conventions.**
- Every claim carries file:line, a command with its output (scripts and outputs are in `.audit/adversarial/`), or a literature reference with the access tag *as recorded in the project's own files*. I fetched no URL myself, so literature claims are "per" the cited project file.
- Severity:
  - **FATAL**: the named claim cannot follow from the design as registered, even if every run succeeds.
  - **MAJOR**: the claim follows only with a serious, currently unaddressed caveat, or is likely to be misread.
  - **MODERATE**: it weakens a control, a label or the stated power.
  - **MINOR**: hygiene or wording.
- Severity is judged **per claim**, not for the whole programme.
- [EVID] marks evidence; [SPEC] marks my speculation or judgement.

---

## 0. Verdict

**The narrowly worded confirmatory package claim can be established.** That claim is: "k F_g layers of 12, the rest GDN-A as built, recover ≥ 80% of the k = 0 → 12 gap on literal key–value retrieval at 16–32K, after an identical task-exposure fine-tune, at ≈ 28M parameters." The pre-registration machinery is unusually strong:
- paired seeds with common random numbers;
- gating on independent dev items;
- a fixed level set for H5;
- guard-protected falsification;
- exact analytic costs (VT-14: 72/72 caches to the byte, `reports/cost_benchmarks_session1.md` L23);
- a reproduction ladder.

**Several claims the programme is set up to make cannot be established as registered (FATAL per claim):**
- H6′ "GDN-E recovers the gap" (equivalence);
- H3 "no task dependence";
- an "aggregation is insensitive to k" reading as long-range integration.

**Four MAJOR blind spots bear directly on the headline H1 / H1-min reading:**
1. **Reference handicap.** An adequate-but-beaten all-full reference is never flagged. The "R_best" safeguard is vacuous by default.
2. **Literal copy.** The retrieval family is literal-prefix copy in which one key token suffices for ≥ 92% of queries on 5 of 7 primary-cell levels.
3. **Learning speed.** The learning-speed control in the *support* direction ignores the k = 0 baseline.
4. **Power.** The advertised power ignores the Holm correction that the registered rule applies.

All are fixable before FEAS or G5, without touching any criterion of an item that has already run.

---

## 1. Severity-ranked findings

"Pre-G5?" means the fix can be entered as a logged pre-run change: no affected item has run, and S2 has not started (`reports/STATUS.md` L100–104; `continuation_audit.md` §4).

| ID | Sev. | Claim affected | Finding | Evidence | Consequence if unaddressed | Recommendation (pre-G5?) |
|---|---|---|---|---|---|---|
| **F-1** | **FATAL** (secondary claim) | H6′: "GDN-E recovers the gap" / "state worth as much as KV" | With 2 seeds of X-E0 against n₀ seeds of C{k*}, Welch df ≈ 2.6. The registered TOST ±5 pp declares "equivalent" in only **10–28% of worlds where the true difference is exactly 0**. "Differs" has 50–62% power at a true 10 pp. The arm is also confounded: GDN-E has 3.7× (D512) / 2.8× (D384) the recurrent-state bytes of GDN-A and 1 head instead of 3–4, so "per-head d_k at equal H·d_k" is not the only change | Sim: `.audit/adversarial/opchar_secondary.out`, H6′ block (n = 5, σ = 3: P(equiv \| Δ = 0) = 0.140; σ = 2: 0.279; P(p < .05 \| Δ = 10) = 0.53–0.62). Rule: `experimental_protocol.md` L75, L718–721; code `analysis/pipeline.py` L1063–1084, `stats.py` L232–235. State bytes: protocol §3.3 table (540,672 vs 147,456 B at D512; 307,200 vs 110,592 B at D384) | Almost every H6′ outcome will be "inconclusive". A reported Δ_E near 0 cannot be called recovery | Make H6′ estimation-only (Δ_E with CI, gap share with Fieller), **or** run X-E{0,k*} at n₀ seeds with a paired test on common seeds (≈ +5–8 GPU-h at D384 [INF from §13.2 rates]). Reword as "≈ 3–4× recurrent state, one head". Pre-G5: yes |
| **F-2** | **FATAL** (named negative) | H3 "no task dependence" | TOST ±0.10 on the per-seed ratio difference D₃ is essentially unattainable. Per-seed SD(D₃) is 0.11–0.30. P(equivalent \| true D₃ = 0) ≤ 0.25 in the best case (σ = 2, n = 7, G_hop = 30 pp) and ≤ 0.05 for σ = 3 or G_hop ≤ 20 | Sim: `opchar_secondary.out`, H3 block. Rule: protocol L70 | A null on task dependence can never be reported. Only "supported" or "inconclusive / not evaluable" are reachable | Declare the negative outcome unreachable in the protocol, or replace it with an estimation statement. Pre-G5: yes |
| **F-3** | **FATAL** (null reading) | RQ-C aggregation: "integration needs little full attention" | A-freq at 32K is largely solvable without long-range integration. On the frozen CS items, a counter over the last 4,096 tokens scores 0.926 / 0.729 / 0.641 / 0.537 (ρ = 4 / 2 / 1.5 / 1.25) and recency scores 0.791 / 0.667 / 0.613 / 0.534 (chance 0.5). A short-memory heuristic therefore reaches a 32K ladder mean of ≈ 0.65–0.71. Only last-8,192 floors of 0.694 (ρ = 1.5) and 0.595 (ρ = 1.25) demand integration | `.audit/adversarial/afreq_windows.py` output (§3.4). Consistent with `s0_integration.md` L359 (item 17), which documents only the ≤ 4K case and the 1,024 window | "Aggregation insensitive to k" cannot be told apart from "every model uses a short-window or recency heuristic" | Pre-register that integration wording requires exceeding the last-8,192 floor at ρ ≤ 1.5 by a stated margin. Add `window4096/8192_floor` columns (the code records 256/1,024/4,096 per row: `eval/scorer.py` L241–273). Pre-G5: yes |
| **M-1** | **MAJOR** | H1 / H1-min; "hybrid > all-full" | Reference-handicap blind spot: (i) "Reference inversion" fires only when C12 **fails** adequacy (a). (ii) R_best is chosen on DG, which is scored on C0 and C12 only unless inversion is checked, so by default **R_best ≡ R**. (iii) A C12 that passes 0.8·S₁₂(C0) but is beaten by C3 by ≥ 5 pp yields "H1 supported" with no label | Protocol L509, L651–653; `analysis/pipeline.py` L223–247 (inversion requires `not a_pooled`), L1001–1014 (R_best = argmax over DG configs); `analysis/verdicts.py` L53–126 (no label for an adequate-but-beaten reference) | The headline "a small proportion recovers most of the performance" can be driven by an all-full model handicapped at 16–32K (thin Stage B of 30–90M tokens; QK-norm dilution at d_h = 64; small-scale all-full deficit, Qiao Table 2 [PI-VER in debate]) | Score DG on C1 and C3 too (≈ 0.03–0.10 GPU-h per model, protocol §9.9), which makes R_best informative. Add a label-only rule "reference exceeded" (a hybrid's paired primary-cell score > S₁₂ by ≥ 5 pp, paired p < 0.05) to the H1 and H1-min verdict sentences, with R_best beside it. Pre-FEAS: yes, by decision row |
| **M-2** | **MAJOR** | H1, H1-min, "k = 1 ≈ k = 12" (construct validity of "retrieval") | **100%** of R1 and R-load query prompts are literal copies of the fact prefix (" The code for {adj} {noun} is" + " "). Another fact shares the queried key's *noun* in 0.0% / 1.3% of R1 D = 3 / D = 15 queries at 32K, and in 1.8 / 7.8 / 30.1 / 50.9% at N = 32 / 128 / 512 / 1,024. So **≈ 80% of the retrieval-family weight** sits on levels (both R1 levels and N ∈ {8, 32, 128}) where matching one key token resolves > 92% of queries. Keys are drawn at random (no hard negatives), and T1 is also a TEF template | `.audit/adversarial/key_overlap.py` output (§3.3); `eval/templates.py` L46–49, L85–87; `eval/generators.py` L118–122; family weighting protocol L44–48 | This is induction-style copy, which one attention layer plus the conv-supplied previous-token features is known to implement (Olsson 2209.11895 [read-section per A1, B2]; Wen et al. 2402.18510 [abstract-only per literature review §4]). H1-min support would then be largely a property of the task format. Literal matching overstates long-context ability (NoLiMa 2502.05167 [read-section per literature review §8]) | (a) Add an exploratory hard-negative cell at 16K and 32K: distractors share the adjective or the noun with each queried key, which forces 2-token binding. Evaluate it on all core models. (b) Word H1 as "literal-copy retrieval". Pre-G5: yes (additive manifest entries) |
| **M-3** | **MAJOR** | H1 / H1-min support (elicitation, E1.7) | The learning-speed control in the support direction tracks only S₁₂ − S_k (seed 0, TEF 1,000 vs 4,000). A k = 0 baseline still learning at 1,000 TEF steps inflates G, and inflates R(k) whenever S_k < S₁₂, since dR/dS₀ = (S_k − S₁₂)/G² < 0. Example: S₀ 40 → 50, S₃ 64 → 66, S₁₂ 70 → 72 moves R₃ from 0.80 to 0.73 with no flag. The TEF LR (3e-4 constant) and length (1,000 steps) are single points with no LR sensitivity arm | Protocol L383, L697–698; `analysis/pipeline.py` L861–889 (`still_closing` uses only `C12 − C{k}`). C0 is run to 4,000 steps (protocol L381), so the data exist | "H1 supported" may partly reflect GDN's slower elicitation (Jelassi 2402.01032 [read-section per B1]) rather than capability | Extend the flag: label "budget-conditional" if sign(L_k) changes or \|ΔR(k)\| > 0.1 between TEF 1,000 and 4,000 (seed 0; all four anchors are already run). No new runs. Pre-G5: yes |
| **M-4** | **MAJOR** | H3 (retrieval vs multi-hop) | (i) The two families differ in **operation type**: retrieval is a one-step copy (M-2), while M-hop needs h + 1 sequential lookups without chain of thought; the link line continues " the code for…", so literal copy fails. They also differ in **load**: R-load has up to 1,024 facts, M-hop ≤ 4(h + 1) ≤ 20. So D₃ > 0 is predicted by depth and composition limits that also bind all-full models. That is not the same claim as "needs a different amount of full attention". (ii) Evaluability is fragile: every seed needs G ≥ 15 pp in both families. P(evaluable) is 0.53 at G_hop = 20 pp and σ = 3, and 0.03 at G_hop = 15. (iii) The equated-level check equates S₁₂ only, not S₀ or load | `eval/templates.py` L46–49 (link vs query); protocol L70, L475–480 (M-hop spec), L710–713; `opchar_secondary.out` H3 block | H3 is likely "not evaluable". If supported, it is open to a "copy vs compose" reading that the design cannot exclude | Add an exploratory load-matched pair (R-load at N = 4(h + 1) vs M-hop h, same T) to separate load from composition. Report non-evaluability prominently. Word H3 as "templated pointer-chasing vs literal copy". Pre-G5: yes |
| **M-5** | **MAJOR** (mechanistic wording) | Any "softmax memory vs delta-rule memory" reading | Compute is not matched: 104 → 501 MFLOP/token at 32K (D512, protocol L258). No arm separates "unbounded memory" from "more compute". Only one of A1-F2's four asymmetries (the conv) has a control. The GDN's implicit recency/position source and the L2-normalised q/k have none | Protocol L258, §15 "Infeasible"; debate I-01 (A1-F2 list), I-28 | Mechanistic conclusions remain conjecture | Keep package wording; add the two uncontrolled asymmetries explicitly to the limitations. No change to runs |
| **M-6** | **MAJOR** (external validity) | Headline generality | One scale (D384 likely), one linear mechanism, an elicited capability after a task-specific fine-tune on the eval template. "Small-scale parity hid multi-hop gaps at scale" (MiniMax M2 blog [read-full per literature review §4]) applies directly to H3 | Protocol §11.9, §15; literature review §12.4 | Results can be misread as answering the README question in general | Already scoped (D040). The report must carry the scope sentence in every claim |
| **M-7** | **MAJOR** (secondary claim) | "Hybrid > all-full is genuine complementarity" | The rival controls are 1-seed (NoPE, SSMax in FEAS, the 2× branch, the dose ladder) or 2-seed (X-gc12). Their difference SD is ≈ 4 pp at σ = 3 [INF], so only confounds of ≳ 8 pp can be excluded. The implicit-position asymmetry is untested | Protocol §3.2 controls, §11.6; `opchar_secondary.out` U-02 block | "Genuine complementarity" is not establishable; at most "not explained by the tested controls at ≤ 2 seeds" | Wording rule (pre-G5). Optionally run NoPE seed 1 unconditionally (see m-1) |
| **M-8** | **MAJOR** (wording rule) | Operator-level wording (§11.9) | The clause "both families give the same verdict" is decided by an F_gc-family H1 test on **2 seeds (df = 1)**, with power 0.27–0.39 raw (0.14–0.20 at Holm's first step) even at true R = 1.0. The rule therefore licenses operator-level wording exactly when the main family is *not* supported, through uninformative agreement | `analysis/pipeline.py` L1170–1181; `analysis/verdicts.py` L160–175; my calculation in §3.2 (P(p < .05 \| R = 1.0, n = 2) = 0.27 / 0.39 at σ = 3 / 2) | An unwarranted operator-level sentence in exactly the case where the evidence is weakest | Count agreement only if the F_gc family is informative (≥ 3 seeds, or its 90% CI excludes the support boundary). Otherwise forbid operator-level wording when δ_gc > 3 pp. Pre-G5: yes |
| **m-0** | MODERATE | Stated power of H1 / H1-min | The operating characteristics in protocol §11.3 (power 0.88 at n = 5, σ = 3; 0.967 for the 5 → 7 rule) are at **unadjusted one-sided α = 0.05**, but the registered support rule is Holm over 4. Monte Carlo of the family at R₃ = 1.0, σ = 3: **P(H1 supported) = 0.55–0.70 at n = 5 and 0.83–0.92 at n = 7**. At R₃ = 0.9: 0.20 (n = 5) and 0.33 (n = 7). VT-18 checks the unadjusted figures only | `debate_record.md` L999 ("All tests are one-sided at α = 0.05"); protocol L67, L675–678; `.audit/adversarial/holm_power.out` | Higher chance of "inconclusive" than advertised. Seed budgeting rested on these figures | Restate power under Holm (VT-18 extension). Optionally adopt a fixed-sequence order (H1 → H1-min → H5a → H3): H1 keeps full α, and later tests lose power if an earlier one fails. Pre-G5: yes (decision row) |
| m-1 | MODERATE | NoPE control (E2.6, E6.6) | The single-seed U-06 trigger (\|ΔR₃\| > 0.1 or \|ΔS₁₂\| > 3 pp) **fires 37–83% of the time with no true effect**, because single-seed R₃ noise is large. The arm is close to a coin flip and cannot exclude a PE × k interaction below ≈ 8 pp | `opchar_secondary.out` U-06 block; protocol L733 | NoPE adds a seed on noise, or misses real effects. "Conditional on RoPE" stays the only honest wording | Run NoPE seed 1 unconditionally (≈ 4 GPU-h at D384, §13.2), or state the detection limit. Pre-G5: yes |
| m-2 | MODERATE | U-02 conv trigger (I-01) | δ_gc from 2 seeds: P(> 3 pp \| true 0) = 0.07 / 0.16 at σ = 2 / 3. P(miss \| true 4.5–6 pp) = 0.07–0.31 | `opchar_secondary.out` U-02 block; protocol L730–732 | Misrouting of operator wording and of Tier-B F_gc arms at the boundary | Report δ_gc with its CI. Treat 2–4 pp as "indeterminate" in the wording |
| m-3 | MODERATE | H4 placement null / effect | "No placement effect" needs TOST ±5 pp on both contrasts. Under a true 0 at n = 5 it passes with P = 0.85 (σ = 2), 0.29 (σ = 3) and 0.01 (σ = 5). A 3-pp c₁ is detected with P = 0.17–0.37. At k* = 1, near a capability transition, σ is plausibly large or bimodal (PI-SIM bimodal rows) | `opchar_secondary.out` H4 block; protocol L71, L716 | Placement will mostly be "inconclusive"; only effects ≳ 7 pp are found | State the σ-dependence. Consider an estimation-only H4 with MDE reporting. Optional: placement at n₀ + 2 seeds |
| m-4 | MODERATE | FEAS scale rule (U-01) | FEAS-2 (S₁₂ − S₀ ≥ 25 / 20 pp) treats a small gap as infeasibility and escalates to D512 (+58 GPU-h). A small gap caused by a *strong k = 0* (FEAS-1 adequacy met) is the opposite problem: more capacity makes GDN stronger. The outcome "GDN-A ≈ all-full at 32K" is turned into a re-scope instead of a reportable finding | Protocol L833; `configs/experiments/plan.yaml` L244 | A wasted escalation and a lost finding [SPEC: low-moderate probability, given that R-load N ≥ 512 should floor k = 0] | Split the FEAS failure branch: FEAS-1 met but FEAS-2 failed with high S₀ → consult the user, and report as a finding before any D512 spend. Pre-FEAS: yes |
| m-5 | MODERATE | Rivals E1.2, E2.7, E3.6, E6.3 (QK-norm dilution) | The §9.4 attention diagnostics (needle mass, entropy, max logit) and pathway diagnostics are **not implemented**, but the FD suite asks for them on FEAS k = 12 and the rival register relies on them. They are not on the FEAS checklist | `gates/G0.md` L249 (open item 20); protocol L502–503, L515; `grep` finds no implementation in `src/` (only the docstring at `eval/suites.py` L122–124) | Dilution rivals cannot be discriminated, for FEAS-5, degradation at 32K or inversion | Implement before S5. Keep FEAS checkpoints so the diagnostics can be run post hoc on k = 12 (inference-only) |
| m-6 | MODERATE | Precision and budget of every confirmatory test | Full-CS scoring cost is **unmeasured**: ≈ 3–4 GPU-h per checkpoint extrapolated from the S0 smoke, against 0.12–0.29 h budgeted (§9.9). FEAS-SMOKE times only `--n-contexts 2` FD cells | `s0_integration.md` L315, L372; protocol §9.9; `experiments/prereg/FEAS_smoke.yaml` (evaluation block) | A mid-study cut to contexts or suites, decided after data exist, is a forking path | Measure one full CS pass at 32K on a FEAS checkpoint. Pre-register any reduction rule before G5 |
| m-7 | MODERATE | H5(a) null wording | The fixed level set is {R1 D 3, 15; R-load N 8, 32}, i.e. ≤ 32 facts, because higher N cannot fit at short T. "No length effect" therefore means "at ≤ 32 facts". If the TEF teaches selective writing, GDN becomes load-limited rather than length-limited [SPEC], and the null would be misread as "attention's benefit doesn't grow with context" | Protocol L72, L475–480 | Misreading of a null | Wording rule. Report the load × length factorial next to H5 |
| m-8 | MODERATE | All k-contrasts (train/eval shift) | TEF queries are **sequential**: queries 2–4 follow earlier Q/A pairs, led by "."; eval queries are **branched**, each directly after the context. TEF keys come from word list A; eval keys from list B. Attention may lean on in-context Q/A demonstrations and GDN on key-specific write policies, so either shift could be k-differential [SPEC]. Neither is measured | `INTERFACES.md` L91; `eval/generators.py` L125–139; protocol L376–377, L450 | Unknown-sign bias in the k-contrast | Add an exploratory list-A cell (held-out values) and report EM by TEF query position on a TEF-format dev cell. Pre-G5: yes |
| m-9 | MODERATE | "k = 1 ≈ k = 12" | Only R(1) ≥ 0.8 is tested. Non-inferiority is registered for S₁₂ − S₃ only | Protocol L78; `analysis/verdicts.py` L148–150 | An equivalence-style headline would be unregistered | Register S₁₂ − S₁ non-inferiority (5 pp) as a secondary estimand. Pre-G5: yes |
| m-10 | MODERATE | Dose and saturation guards | Both are single-seed, with 3-pp and 5-pp thresholds near 1–2 SDs of a single-run difference [INF from σ = 2–3 pp]. The still-closing DC fallback uses a different level set (100 contexts) from the primary cell | Protocol L383, L698–699; `analysis/pipeline.py` L866–867, L891–910 | Spurious or missed "budget-conditional" labels | Report the guards with their noise. Prefer CS at TEF-4,000 (already allowed) |
| m-11 | MODERATE | H1-min (k = 1) | Abrupt single-layer circuit formation predicts bimodal seeds. PI-SIM: no few-seed test is valid (0.09–0.30) | Protocol L679; debate L322 | H1-min verdict unreliable under bimodality | Already handled by success counts. Make them co-reported with H1-min in the verdict sentence |
| m-12 | MODERATE | k = 0 baseline (RR-R5-power) | A bf16 GDN training handicap of < 0.115 (50% power) is not excluded, and it would inflate every k-contrast in attention's favour. There is no fp32-trained k = 0 control | `results_R5.md` L22–28; `pi_rulings` §7 RR-R5-power row | Unknown downward bias on S₀ | Keep it named (done). Consider a Tier-C fp32 k = 0 seed-0 run |
| m-13 | MODERATE | Measured cost frontier | Session 1 meets no strict G-cost row (resident CARLA client at 0% SM). Session 2 is owed. At d = 384, the attention/dense FLOP ratio at 32K is 6.56, against 1.23 at d = 2048. Decode at B = 1 is launch-bound (roofline 0.07 for C0). `models/attention.py` has since changed (git status), so D087 requires re-measurement or identical hashes | `cost_benchmarks_session1.md` L28–31, L53–55; protocol L567–574 (projection table); `git status` | Measured "speed-ups" are implementation- and scale-specific | Keep analytic costs primary (done). Never present measured speed-ups without the projection table |
| n-1 | MINOR | FEAS → core transfer | FEAS Stage A is 0.30B tokens (`plan.yaml` L188), half the core's 0.6B, yet `reproduction_plan.md` L554 says "0.6B tokens" for FEAS end-of-Stage-A. FEAS-3, FEAS-5 and U-08 are decided on less-trained models | As cited | Pilot-to-main transfer; doc inconsistency | Correct the doc. Note the transfer caveat in the FEAS report |
| n-2 | MINOR | H5 "exposure-robust" label | LM-4 copies 256-token passages, beyond GDN-A's state at any gap, so gain₀(g) may be ≈ 0 at all g ≥ 512 and Λ(g) is flat. The label's sign is then noise | Protocol L578–579 (LM-4) | Label uninformative | Report Λ(g) with its CI; withhold the label if the CI spans 0 |
| n-3 | MINOR | H1 falsification rival E2.4 | Placement at k = 3 is tested only if k* = 3. When H1 is falsified, k* is likely 1 or 2 | Protocol L131; `competing_explanations.md` L57 | One falsification rival untested | State it |
| n-4 | MINOR | Wording | "Multi-hop reasoning" is templated pointer-chasing with exact key matches (h ≤ 4 in eval, ≤ 6 in TEF) | `eval/templates.py` L46–49 | Overclaim | Use "pointer-chasing (h links)" |
| n-5 | MINOR | Primary-cell selection | G-REF selection on dev items shares seed noise with the test-item estimate: conservative for support, anti-conservative for falsification near the threshold (debate I-06 already notes this) | Debate L387–392 | Small bias near the threshold | Report the per-seed adequacy margin |

---

## 2. Outcome by outcome: the strongest alternative explanation, and whether the design can rule it out

**Reading the "rule out?" column:**
- **Yes:** a pre-registered check with adequate power exists.
- **Partly:** a check exists but is underpowered or covers only part of the rival.
- **No:** nothing in Tiers A–B discriminates it.

| Expected finding | Strongest alternative explanation | What the design does | Rule out? | Gap and fix |
|---|---|---|---|---|
| **H1 supported** (L₃ > 0) | **Handicapped reference.** C12 passes 0.8·S₁₂(C0) but is beaten by C3 at 16–32K: thin Stage B, d_h = 64 dilution, RoPE 4K → 32K positions seen only in Stage B and TEF, small-scale all-full deficit. Then L₃ > 0 holds trivially. *Second:* literal single-token copy (M-2). *Third:* k = 0 under-elicited at TEF 1,000 (M-3) | G-REF adequacy; 2× branch (seed 0); SSMax (FEAS, seed 101); NoPE (seed 0); X-gc12 (2 seeds); still-closing on S₁₂ − S_k | **Partly** | Adequacy excludes collapse, not handicap. Inversion is defined only for C12 failing (a). R_best ≡ R by default (M-1). The k = 0 learning curve is unchecked (M-3). Fixes R1–R3 (§4) |
| **H1 falsified** (L₃ < 0 with guards) | k = 3 still learning; long stage under-trained; centred placement poor at k = 3; few RoPE layers under-trained at long range; LR chosen on 4K loss disfavours hybrids | Dose guard (seed 0: L₃ < 0 at 4,000 **and** a plateau); saturation guard (seed 0); NoPE (seed 0); Tier-C η*/√2 | **Mostly** for learnability and budget (both guards are mandatory). **No** for placement unless k* = 3 (n-3); LR only in Tier C | Falsification is appropriately hard to reach. The residual is single-seed guards (m-10). The asymmetry (support needs only the test) should be stated in the report |
| **k = 1 ≈ k = 12** (H1-min supported) | (i) The task is literal copy: one attention layer plus conv "smeared keys" suffices by mechanism (Olsson [per A1, B2]). (ii) Ceiling compression for every k ≥ 1. (iii) Handicapped reference (M-1). (iv) The test establishes R(1) ≥ 0.8, not equivalence | H1-robust per-level R; N50 and N/N50₀; informative levels (S₁₂ in 20–90%) | **No** for (i) and (iv); **partly** for (ii) | Hard-negative cell (M-2); S₁₂ − S₁ non-inferiority (m-9); success counts beside the verdict (m-11) |
| **Hybrid > all-full** | Missing conv; RoPE long positions; dilution; Stage-B budget; TEF learning speed; LR; GDN's implicit position/recency source (no arm) | X-gc12 (2 seeds); NoPE, SSMax, 2×, dose (1 seed each); inversion and R_best | **Partly**: only confounds of ≳ 8 pp can be excluded (M-7). The inversion label does not fire if C12 is adequate (M-1) | Wording: "not explained by the tested controls (≤ 2 seeds)". DG on all core configs (R1) |
| **Degradation at 32K** (all k lower at 32K than 16K) | Exposure at 32K (Stage B 30–90M tokens; 32K items are 1/7 of TEF items); RoPE at 16–32K positions; QK-norm dilution (needle weight 0.64 → 0.05 from 1K to 32K at unit gains [MEAS-B], protocol D013); evaluation numerics at 32K; needle–query **distance** grows with T | 2× branch (1 seed); SSMax (FEAS); NoPE (1 seed); R10 per-position loss (FEAS); VT-22b v2 (C12 fp32 vs Qwen3 at 16K/32K) and FEAS-PREC P4 (numerics); R-dist (32K only) | **Partly.** Numerics: yes, if VT-22b and FEAS-PREC pass. Dilution: **no** until the diagnostics exist (m-5). Distance vs length: only at 32K and only for C0/C12 (2 seeds) and C1/C3 (seed 0) | Implement the diagnostics (R10 of §4). Report distance-stratified scores at every T (depth and distance are recorded per query, protocol §9.8) |
| **H3 supported** (R_ret(3) > R_hop(3)) | Copy (1 lookup) vs compose (h + 1 lookups, no chain of thought) and a load mismatch (≤ 20 vs ≤ 1,024 facts). The ladder mix is arbitrary. Ratio noise, with per-seed SD(D₃) of 0.11–0.30 | Equated-level sign check (S₁₂ only); GLMM and Fieller; M-order; Tier-C adjacent vs spread | **No** for copy vs compose; **partly** for difficulty (equates S₁₂, not S₀ or load) | Load-matched pair (M-4). Wording |
| **H3 null / not evaluable** | Multi-hop near floor for every k (G_hop < 15 pp), so not evaluable; TOST unattainable | Evaluability rule; TOST ±0.10 | A null is **unreachable** (F-2). "Not evaluable" is likely if G_hop ≲ 20 pp (P(evaluable) = 0.53 at σ = 3) | Make the estimation statement primary when not evaluable |
| **H5 supported** (gap grows with log₂T) | k = 0 at ceiling at short T (probability-scale compression); "length" = filler amount × distance; reference changing with T | Fixed level set (removes ladder composition, debate §6.4); equal-count TEF plus block normalisation (DEV-04, D063); G-REF label (D066); LM-4 corroboration; R-dist and load × length | **Mostly** for exposure and ladder composition; **partly** for distance vs filler (32K only) | Add a logp_gold-scale slope as a "metric-robust" check, as H1 has (pre-G5). Report per-architecture R-dist (planned) |
| **H5 null** | TEF-taught selective writing makes GDN load-limited, not length-limited [SPEC]; the low-load level set cannot show a load-dependent length effect | TOST ±2 pp per doubling | Partly | Scope the wording to "≤ 32 facts" (m-7) |
| **Placement effect** | Feature availability at layer 1 (only one GDN layer below); a final-layer copy shortcut at layer 11; seed noise | c₁, c₂ with Holm; spacing preserved; layer 0 avoided | **Partly**: only effects ≳ 7 pp are reliably detected (m-3) | MDE reporting |
| **Placement null** | Low power | TOST ±5 pp on both contrasts | Only if σ ≤ 2 pp (P = 0.85; 0.29 at σ = 3) | State the σ-dependence |
| **GDN-E recovers the gap** | ≈ 3–4× recurrent state, not only per-head d_k; 1 head vs 3–4; 2 seeds | Welch on 2 vs n seeds; TOST ±5 pp | **No** (F-1) | Estimation-only, or n₀ seeds |
| **Cost frontier** (hybrids dominate) | For any per-layer-additive cost M(k) = G·(R(k) − k/12) is an identity. Measured differences reflect kernel maturity (GDN chunk ≈ 10% of peak vs FA2 81% [MEAS-B]), launch overhead at B = 1, and a d = 384 FLOP mix unrepresentative of production | Analytic primary (exact); measured secondary with graph capture, cold L2, min-of-N and a roofline-fraction table | The analytic part: **yes**, but it adds nothing beyond R(k). The measured part: implementation-scoped. Only B_max-limited throughput is non-additive and informative | Present the frontier as a re-plot of R(k) plus the B_max throughput (m-13) |
| **Aggregation insensitive to k** | Every model sits at the recency or short-window floor (0.65–0.71 ladder mean at 32K) | Recency and window-1,024 floor columns | **No**, unless scores clear the last-8,192 floors at ρ ≤ 1.5 (F-3) | R8 of §4 |

---

## 3. New quantitative checks (CPU-only; all reproducible)

All scripts and outputs are in `/home/sedrica-lynx/Downloads/outeropt/.audit/adversarial/`, run with `python -I` from the project venv. The models mirror `src/hybridlc/analysis/stats.py` (Welch, TOST, one-sample t) and PI-SIM's assumptions (debate §6.2: G = 30 pp, seed effect 2 pp, residual σ ∈ {2, 3} pp).

### 3.1 H1 power under the registered Holm rule (`holm_power.py`, `holm_power.out`)

Analytic power of the per-seed L₃ t-test:

| σ | R₃ | n = 5: α .05 / .0167 / .0125 | n = 7: α .05 / .0167 / .0125 |
|---|---|---|---|
| 3 | 1.0 | 0.88 / 0.63 / 0.55 | 0.97 / 0.87 / 0.83 |
| 3 | 0.9 | 0.42 / 0.20 / 0.16 | 0.56 / 0.32 / 0.27 |
| 2 | 1.0 | 0.99 / 0.93 / 0.88 | 1.00 / 1.00 / 0.99 |

**Full Holm family, Monte Carlo.** Scenario: H3 not evaluable (p = 1), H1-min false (R₁ = 0.5), H5a either strong (p ≈ 0) or absent. P(H1 supported):

| σ | n | R₃ = 1.0, H5a strong | R₃ = 1.0, H5a absent | R₃ = 0.9, H5a strong |
|---|---|---|---|---|
| 3 | 5 | 0.630 | 0.554 | 0.201 |
| 3 | 7 | 0.871 | 0.827 | 0.325 |
| 2 | 5 | 0.927 | 0.885 | 0.400 |

These compare with the protocol's 0.88 / 0.967 (§11.3), which are unadjusted (`debate_record.md` L999).

### 3.2 Secondary tests and triggers (`opchar_secondary.py`, `opchar_secondary.out`)

- **H6′.** Median Welch df ≈ 2.6–2.8.
  - P(TOST equivalent \| Δ = 0): 0.14 / 0.28 (n = 5, σ = 3 / 2); 0.10 / 0.22 (n = 4).
  - P(p < .05 \| Δ = 10 pp): 0.50–0.62.
- **H4** (n = 5). P(no-placement-effect \| c = 0) = 0.85 / 0.29 / 0.01 at σ = 2 / 3 / 5. P(detect c₁ = 3 pp) = 0.37 / 0.17.
- **U-02.** P(δ̂ > 3 \| δ = 0) = 0.07 / 0.16. P(δ̂ > 3 \| δ = 6) = 0.94 / 0.84 (σ = 2 / 3).
- **U-06.** P(fires \| no effect) = 0.37–0.83, depending on σ and G.
- **H3.**
  - P(evaluable) = 0.03 (G_hop = 15), 0.53–0.82 (G_hop = 20), ≈ 1.0 (G_hop = 30).
  - Power at D₃ = 0.2: 0.63–0.90 when G_hop = 30, and 0.28–0.63 (unconditional) when G_hop = 20.
  - P(TOST \| D₃ = 0) ≤ 0.25.
- **F_gc family H1 at 2 seeds** (inline computation, documented in M-8). P(one-sided p < .05 \| R = 1.0) = 0.27 / 0.39 at σ = 3 / 2. P(p < .025) = 0.14 / 0.20. At R = 0.8 the type-I rate is nominal.

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

Word list B has 154 adjectives × 1,390 nouns = 214,060 keys (list A: 158 × 1,369), computed with `hybridlc.eval.words.word_list`.

### 3.4 A-freq short-memory floors (`afreq_windows.py`; frozen CS items)

| T, ρ | Recency | Last-1,024 | Last-4,096 | Last-8,192 |
|---|---|---|---|---|
| 32K, ρ = 4 | 0.791 | 0.721 | 0.926 | 0.987 |
| 32K, ρ = 2 | 0.667 | 0.609 | 0.729 | 0.832 |
| 32K, ρ = 1.5 | 0.613 | 0.548 | 0.641 | 0.694 |
| 32K, ρ = 1.25 | 0.534 | 0.514 | 0.537 | 0.595 |
| 16K, ρ = 2 | 0.662 | 0.646 | 0.845 | 0.941 |

The 32K recency and 1,024 / 4,096 values agree with `s0_integration.md` L359 (0.791 / 0.721 / 0.926 for ρ = 4; 0.667 / 0.609 / 0.729 for ρ = 2), which validates the script.

### 3.5 Code audit (read-only)

- **Reference inversion:** `analysis/pipeline.py` L231–232 returns `triggered=False` whenever C12 passes adequacy (a).
- **R_best:** L1001–1014 takes the argmax over whatever configurations DG contains. Protocol L509 scores DG on C0 and C12 only by default.
- **Operator wording:** `analysis/verdicts.py` L160–175 sets `op_ok=True` when both families give the same supported / not-supported label. The F_gc family test (`pipeline.py` L1170–1181) runs on X-gc{1,3} versus X-gc12 with the seeds available (2 by design).
- **Still-closing:** `pipeline.py` L882–886 computes the gap as `C12 − C{k}` only.

---

## 4. Recommendations, ranked by value per cost

None of these is applied. Each needs a PI decision row. Every one is **pre-run**: no affected item has run, and FEAS has not started, so the integrity rule (`pi_rulings` §0 rule 1) permits them as logged changes.

1. **Close the reference-handicap blind spot (M-1).**
   - Score DG on C1 and C3 (and on placement arms, if run). Cost: ≈ 0.03–0.10 GPU-h per model (§9.9).
   - Add a **label-only** rule: "reference exceeded by C{k}" when a hybrid's paired primary-cell score exceeds S₁₂ by ≥ 5 pp with paired p < 0.05.
   - Print R_best, now non-vacuous, in every H1 and H1-min verdict sentence.
   - *Consequence:* no change to the H1 test; more honest wording; small evaluation cost.
2. **Extend the still-closing check to the support direction (M-3).** Label "budget-conditional" if sign(L_k) changes or |ΔR(k)| > 0.1 between TEF 1,000 and 4,000 (seed 0; all anchors are already run to 4,000). *Consequence:* no new runs; possibly more conservative labels.
3. **Add a hard-negative retrieval diagnostic (M-2).**
   - At 16K and 32K: ≥ 50% of queried keys get a distractor that shares the adjective or the noun.
   - Exploratory, BH-FDR. Evaluated on all core models.
   - *Consequence:* separates single-token copy from 2-token binding. New manifest entries before G5; CS is unchanged.
4. **Restate H1 power under Holm (m-0).** Then either accept it explicitly, or adopt a fixed-sequence order (H1 → H1-min → H5a → H3). *Consequence:* H1 recovers ≈ 0.25 of power at n = 5, σ = 3, R = 1.0. Later hypotheses are tested only if earlier ones are supported.
5. **Fix the operator-wording clause (M-8).** Count family agreement only when the F_gc family has ≥ 3 seeds or a CI that excludes the support boundary. *Consequence:* fewer and better-founded operator-level sentences.
6. **Make H6′ estimation-only, or fund n₀ seeds (F-1).** Correct "equal H·d_k" to say the arm has ≈ 3–4× the recurrent state and 1 head. *Consequence:* removes an unreachable verdict, or costs ≈ 5–8 GPU-h.
7. **H3 (F-2, M-4).** Declare the negative outcome unreachable. Add an exploratory load-matched pair (R-load N = 4(h + 1) vs M-hop h). Lead with "not evaluable" when that applies. *Consequence:* honest reporting; small item cost.
8. **Aggregation integration threshold (F-3).** Integration wording only if the A-freq score at 32K and ρ ≤ 1.5 exceeds the last-8,192 floor by a pre-stated margin. Add 4,096 and 8,192 window floors to `cell_summary`.
9. **FEAS branch split (m-4).** If FEAS-1 passes but FEAS-2 fails with high S₀, consult the user before any D512 spend, and record the result as a finding.
10. **Implement the §9.4 attention and pathway diagnostics (m-5)** before S5. Run them post hoc on FEAS k = 12 checkpoints, as FD specifies.
11. **Measure full-CS scoring cost (m-6)** on one FEAS checkpoint. Pre-register any context-reduction rule before G5.
12. **Wording rules for the final report** (M-5, M-6, M-7, m-7, n-4):
    - "literal-copy retrieval";
    - "templated pointer-chasing";
    - "no length effect at ≤ 32 facts";
    - "hybrid advantage not explained by controls at ≤ 2 seeds";
    - the compute-vs-memory confound and the two uncontrolled mixer asymmetries named.
13. **Add a non-inferiority estimand for S₁₂ − S₁ (m-9).** Add a list-A key cell and a TEF query-position check (m-8). Run NoPE seed 1 unconditionally, or state the ≈ 8 pp detection limit (m-1).

---

## 5. What is sound (no change recommended)

- **Paired-seed inference on L_k,** with common random numbers (shared data order and shared-module initialisation). The choice of paired over pooled inference is validated by PI-SIM, and I independently reproduced the unadjusted figures: 0.88 at n = 5, σ = 3, α = .05 matches PI-SIM's 0.880.
- **Fixed level set for H5,** which removes the ladder-composition artefact (debate §6.4). Block-normalised TEF loss (DEV-04) restores length balance.
- **Independent dev items for gating;** no selection on test items.
- **Falsification guards** (dose and saturation) make a false "falsified" unlikely.
- **Analytic state bytes** are exact (VT-14, 72/72). Measured costs are correctly demoted to secondary and appendix status.
- **The reproduction ladder (R1 → R4/R5 → R6)** tests the chain "their harness / our mixer / our harness". R7 cleared the scorer item by item against lm-eval.
- **k-differential residuals** are explicitly named (D090), with safeguards that can see a between-k difference (FEAS-PREC v2 Q1–Q4).
- **Verdict vocabulary** (D088) and **hashed pre-registrations** (D091) prevent relabelling.

---

## 6. Not verified, and limits of this review

- **Literature claims** were not re-opened by me. Access tags are those recorded in the project files.
- **Severity of M-2 for GDN.** Single-token sufficiency also helps the GDN, since its key representation needs only one distinguishing token. My claim is that the task format makes one attention layer sufficient by a known mechanism, not that it biases the k-contrast in a known direction [SPEC on direction].
- **Simulation assumptions.** Normal residuals with σ ∈ {2, 3} pp and G = 30 pp follow PI-SIM's assumptions. True σ, G and G_hop are unknown until FEAS and S4. Bimodality would make every few-seed result worse.
- **m-8 is untested.** Whether the train/eval shifts are k-differential is speculative.
- **Evaluation cost (m-6)** rests on the S0 smoke extrapolation only.
- **Code.** I did not run the analysis pipeline. The code-path findings (M-1, M-3, M-8) come from reading `pipeline.py` and `verdicts.py` in the current working tree. Workflow `wf_b92a39ce-f17` is concurrently editing `src/`, but not `analysis/`, per `git status` at 15:4x.

## Appendix: commands

```bash
source /home/sedrica-lynx/Downloads/outeropt/.venv/bin/activate
cd /home/sedrica-lynx/Downloads/outeropt/.audit/adversarial
python -I opchar_secondary.py | tee opchar_secondary.out   # H6', H4, U-02, U-06, H3
python -I holm_power.py | tee holm_power.out               # H1 power under Holm
python -I key_overlap.py                                   # literal copy / key-token overlap (frozen CS items)
python -I afreq_windows.py                                 # A-freq window and recency floors (frozen CS items)
```

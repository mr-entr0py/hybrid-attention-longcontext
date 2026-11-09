# Review B, round 2: an adversarial critique of Review A, and my updated positions

**Reviewer:** B (experimental and engineering validity; adversarial). **Date:** 2026-11-09.

**What this review covers**
- Reviewer A's round-1 review (`reviews/review_A_round1.md`, "A"), especially its independent design (A §2–§4).
- My own round-1 review (`reviews/review_B_round1.md`, "B1").
- The initial proposal (`reports/initial_proposal.md`, "PROP").
- I read all three in full.

**Labels.** "A-F1" or "A-M9" means issue F1 or M9 in A's review. "B1-F1" is issue F1 in my round-1 review. W*n* (critiques of A), C*n* (changes of mind) and P*n* (positions held) are this document's own labels. H1–H6 always refer to the research hypotheses.

**Tags.**
- Access: [read-full], [read-section], [abstract-only], [code-inspected].
- Measurement: [MEAS-B2] is new in this round; [MEAS-B] is from my round 1.
- Epistemic: [EST] established in a cited source; [HYP] hypothesis; [INF] my inference, arithmetic or simulation.

**What I did for this round**
1. **Measurements on the project GPU [MEAS-B2].**
   - Conditions: CARLA was running (15–42 % utilisation). It restarted once during my session, its footprint grew from 3.9 to 6.25 GB, and other processes used ≈ 7.9 GB in total. My process was capped at 35–40 % of 15.9 GiB. Total GPU time ≈ 20 min.
   - Every timing is **contended**. I use min-of-N, as in B1.
   - What I measured:
     - A's twin attention, built and timed in the whole model;
     - GDN-E (A's state-size arm) and GLA-twin on sm_120;
     - D384 vs D512 training throughput;
     - attention-backend support for d_v ≠ d_qk, and decode cost.
2. **Three simulations [INF].**
   - Delta-rule key capacity: multi-head vs single-head.
   - Type-I error and power of A's pooled-variance test vs paired tests: 20,000 trials per cell, under homoscedastic, heteroscedastic and bimodal seed distributions.
   - Softmax needle weight when SiLU precedes QK-RMSNorm.
3. **Sources.**
   - I re-checked the sources my critique depends on (Appendix B), including A's reading of Wang et al. Table 7, which I confirm.
   - I inspected three industrial hybrid attention implementations in the project's `transformers` 5.22.2.
4. **Scope.** Scripts are in my session scratchpad (`…/scratchpad/revB2/`), not in the repository. Appendix A gives enough detail to reproduce every number. I wrote only this file.

---

## 0. Summary

**Verdict on Review A's design.**
- A materially improves PROP. I adopt or endorse six of its ideas (C1–C9 below):
  - the operator-isolating twin control, in modified form;
  - linear contrasts L_k instead of ratio estimators;
  - load ladders and balanced exposure;
  - the dose ladder;
  - the H6′ reformulation;
  - analytic projection of costs to larger scale.
- As written, however, **A's design still cannot establish its headline claims at long context.** Five defects are serious:
  1. **No long-length reference adequacy (W1).** The confirmatory H1 is read at **T = 8K**. A's "informative level" rule (k = 12 between 20 % and 90 %) cannot detect an all-full reference that has collapsed at 32K relative to its own short-context competence.
  2. **D384 discovery has no LM-regime feasibility gate (W2).** At D384 the model has 21.3M non-embedding parameters and sees 60M tokens at 32K. Qiao et al. excluded a 15M model trained on 1.5B tokens (half from long documents, at 16K) as "too small for stable long-context behavior".
  3. **A's pooled-variance inference is anti-conservative under the heteroscedasticity A itself predicts (W3).** In my simulation, type-I error is **14.5 %** at nominal 5 %, and A's Brown–Forsythe guard fires in only **1.8 %** of those trials. Under bimodal seeds, no 3-seed test is valid (23–29 %).
  4. **Internal inconsistencies (W5).**
     - A anchors capacity on H·d_k. Under that anchor, GDN-E has the *same* capacity as GDN-A, so H6′ would be a null manipulation.
     - A's own threshold prior (A-M9: k = 1 ≈ k = 0 for GDN) undermines three of its k = 1-targeted Tier-2 arms (12 runs).
     - A rejects ratio estimators for H1 but uses them for H3 and H6′.
  5. **Inherited and new measurement artefacts (W7).** A adopts PROP §10 unchanged. With native d_v ≠ d_qk, the twin's default SDPA dispatch decodes at 32K in **1,700 µs instead of 66 µs** (26×) [MEAS-B2].

**Where I change my mind (§3).**
- The conv/gate asymmetry must be controlled in the core, not conditionally (C1).
- My dual "bootstrap **and** df = 2 seed-t" rule makes H1 nearly unsupportable, so A-M1 applies to my design too. I adopt L_k with 5-seed anchors (C2).
- My k = 1 placement study must become conditional (C3).
- NoPE moves into Tier 1 (C4).
- B1-M4 was overstated: nondeterminism is an issue of reproducibility, not bias (C5).
- A's D384 throughput assumption is correct (C6).
- I also adopt balanced exposure, 15 % PG19 in Stage A, GLA in Tier 2 and A's two-contrast placement test (C7–C9).

**Where I hold firm (§4).**
- P1: long-length reference-adequacy gate (G-REF), with the longest adequate length primary.
- P2: an LM-regime feasibility pilot that decides the discovery scale and the Stage-B budget.
- P3: an artefact-controlled cost protocol.
- P4: seeds where the tests live (now: 5 for the confirmatory anchors).
- P5: RoPE θ = 5e5 by default, with no thin selection pilot.
- P6: exposure treated as a factor with a plateau criterion (now converged with A).
- P7: deterministic G0 and kill-and-resume.

§5 gives a reconciled design: Tier 1 ≈ 96 GPU-h contended (≈ 115 h with 20 % contingency) on the D384 path.

---

## 1. The strongest parts of A's design (verified where I could)

| A's element | My check | Verdict |
|---|---|---|
| **Twin principle** (A-F2): every non-operator component is identical in full and linear mixers | Olsson et al.'s smeared-key result is verbatim in the PDF text: "in-context learning does indeed form for one-layer models (when it didn't before), and it forms earlier for two-layer and larger models" ([2209.11895](https://arxiv.org/abs/2209.11895), local text L519–539 [read-section]). Conv after Q/K/V in softmax attention has a precedent in Primer ([2109.08668](https://arxiv.org/abs/2109.08668) [abstract-only]) | **Sound in principle.** I adopt it, modified (W6, C1) |
| Twin parameter counts | Whole-model instantiation: D512 twin k = 12 = 40,959,488 non-embedding; D384 = 21,283,200. This implies 1,052,928 / 593,152 per mixer, exactly A's figures [MEAS-B2] | Confirmed |
| Twin kernels via head duplication | Duplication through FA2 `varlen_attn` gives the same max error vs an fp32 reference (5.95e-3) as native kernels. `varlen_attn` and SDPA-FLASH reject d_v ≠ d_qk; cuDNN, EFFICIENT, MATH and FlexAttention accept it [MEAS-B2] | Confirmed |
| Twin training cost | Whole model at k = 12, 1 × 32K with checkpointing: twin vs plain **37.0K vs 35.9K tok/s** (D512) and **49.6K vs 49.0K** (D384). At 4K: −11 % (D512) and −17 % (D384), from the extra conv/norm kernels [MEAS-B2] | Feasible at no material cost |
| **GDN-E** (state-size arm) on sm_120, flagged by A as untested | Forward and backward run in the whole model at K = 256 (D512) and K = 192 (D384); losses finite. Chunk numerics vs an fp64 recurrence at T = 8K match config A: relative RMS 1.46–1.64e-3 (TF32), 4.9–6.0e-3 (bf16), flat across position bands. The `fused_recurrent` decode tail also works. Throughput: −11 to −15 % at 4K, −17 to −25 % at 32K vs GDN-A [MEAS-B2] | A's G0 risk is **resolved**: the arm is feasible |
| GLA-twin on sm_120 | Runs at D384: 128.5K tok/s (Stage A), 101.1K (Stage B) [MEAS-B2] | Feasible |
| **D384 throughput assumption** (A §2.11: 1.35× D512) | Measured D384/D512 ratios: 1.31–1.39× in Stage A and 1.30–1.36× at 32K for the endpoints [MEAS-B2] | **Confirmed** (C6) |
| Linear contrast L_k = S_k − 0.2·S₀ − 0.8·S₁₂ (A-F1, A-M1) | Algebraically equivalent to R ≥ 0.8 when G > 0, and the seed-block effects cancel (the coefficients sum to 0) [INF] | Adopt |
| Pooled-variance power | With homoscedastic σ = 3 pp, power at a true R₃ = 1.0 (G = 30 pp) is **0.81**, vs 0.54 for my 3-seed paired rule; type-I error is 5.1 % [INF, simulation] | A real gain, but fragile (W3) |
| Load ladder N ∈ {8, 32, 128, 512} | Identical to my B1-M6 ladder | Converged |
| Dose ladder (250 / 1,000 / 4,000 steps; "still closing" flag) | Same as my TEF-4× branch (B1-M2) | Converged |
| Balanced exposure: equal items per length bucket (A-M3) | Arithmetic checked: 65.5M / 63.5K ≈ 1,030 items per bucket | Adopt (C7) |
| Analytic cost projection | T/(13d) for MHA: 4.92 at d = 512 and 0.62 at d = 4096, T = 32K [INF, re-derived] | Adopt. It complements B1-F2 |
| A-M9: Wang et al.'s GDN prior is a threshold | Table 7 verified in the paper text: GDN 0.356 (pure), 0.341 (24:1), 0.352 (12:1), 0.425 (6:1), 0.436 (3:1); DeltaNet 0.417 at 24:1 vs 0.352 at 12:1 ([2507.06457](https://arxiv.org/abs/2507.06457), Table 7 [read-section]) | **Confirmed.** It forces changes in my design (C3) |
| H6′: is KV memory worth more than recurrent-state memory? | The right scientific question for a memory frontier | Adopt as secondary, with the fixes in W5(a)–(b) |

---

## 2. The weakest parts, what A's design cannot establish, and feasibility problems

### W1. No reference-adequacy guard at long lengths, and the confirmatory H1 is read at T = 8K (serious)

**What A does.**
- H1 and H1-min use R-load at **T = 8K** (A §2.8 and §3).
- H5 uses R-dist at T = 32K.
- A level counts as "informative" if k = 12 scores between 20 % and 90 % and Ĝ ≥ 15 pp.
- Nothing compares the reference's long-length accuracy with its own short-context or no-filler accuracy.

**Why this is a defect.**
1. **The title asks about long context.** The README grid runs from 1K to 32K, and A's design never tests H1 above 8K.
   - A's justification is that load matters more than elapsed context, citing "How Linear Attention Remembers" ([2609.33093](https://arxiv.org/abs/2609.33093) [abstract-only]).
   - That abstract states the result only "within the tested regime" for "pretrained GLA and GDN models", with no sizes given, and nothing about softmax layers at 32K.
   - Building the confirmatory estimand on an abstract-only claim about the *linear* layers ignores how the *attention* layers degrade with length.
2. **The attention side has length-specific failure modes that A's guard cannot see.**
   - **Dilution.** At d_h = 64 with unit QK-norm gains, needle weight falls from 0.64 at 1K to **0.05** at 32K (B1-M3; reproduced exactly this round).
   - **Head size.** Industrial models avoid this partly through head size. Qwen3-Next uses `head_dim = 256` (`configuration_qwen3_next.py` L107, transformers 5.22.2 [code-inspected]), so the logit cap at unit gains is 16, not 8, and the initial needle weight at 32K is ≈ 0.99 rather than 0.05 [INF]. Dilution is therefore a **small-head artefact**, and it falls exactly on our 32K cells.
   - **The twin makes it worse.** Its SiLU before QK-norm cuts needle weight at 32K from 0.71 to **0.39** at gain² = 1.56 (W6).
   - **Even large budgets leave small all-full models weak.** Qiao et al.'s S4 all-full model (104M non-embedding parameters), trained on ≈ 100B tokens at 16K, scored **35.95** on RULER-NIAH@16K, *below* both SWA hybrids (49.58 and 67.81) ([2606.15378v1](https://arxiv.org/html/2606.15378v1), Table 2 [read-section]). RULER is a prompted benchmark, so part of this deficit may be format rather than capability; our cloze format with TEF removes that part, but not the length effects listed above.
3. **Failure scenario for A's H5** (illustrative): R-dist at 32K.

   | d | 256 | 2K | 8K | 30K |
   |---|---|---|---|---|
   | S₁₂ (%), with C0 = 95 % | 85 | 70 | 50 | 30 |
   | S₀ (%) | 60 | 25 | 15 | 12 |
   | Δ₀ (pp) | 25 | 45 | 35 | 18 |

   - Every level is "informative" under A's rule (20 % ≤ S₁₂ ≤ 90 %, Ĝ ≥ 15 pp).
   - Yet the reference keeps less than a third of its no-filler competence at 30K.
   - The slope of Δ₀ on log₂ d is ≈ 0 to negative, so **H5 is spuriously "falsified"**, and H1-type recovery ratios at 30K are computed against a collapsed reference.

**Fix.** Use my G-REF gate (B1-F1), applied to A's ladders. It must pass per seed and per length.
- S₁₂(T) ≥ 0.8 · S₁₂(C0) on the ladder mean, where C0 is the same item without filler.
- S₁₂(T) − S₀(T) ≥ 15 pp.

Make **R-load a load × length factorial at T ∈ {8K, 32K}**: the same N levels and fact templates, differing only in filler, so N = 512 fills 75 % of an 8K context or 19 % of a 32K one.
- **Primary H1 cell:** the ladder mean at the **longest G-REF-passing length** (pre-registered order: 32K, then 16K, then 8K). 8K is always reported.
- This keeps A's load axis and my long-band priority. It separates load from length at fixed load: N = 8 at 8K vs 32K.
- Cost: evaluation only (≈ +0.05 GPU-h per model).

### W2. D384 as the discovery scale has no LM-regime feasibility gate (serious feasibility risk)

**Evidence** [EST, [read-section] [2606.15378v1](https://arxiv.org/html/2606.15378v1)]:
- Qiao's S1 has **15M** non-embedding parameters (S2: 31M). All models were pretrained "with a 16K context length on a 1:1 mixture of long and short datasets".
- The authors "exclude the S1 checkpoint at D=100N because its training budget is too small for stable long-context behavior". That run had 1.5B tokens, roughly 0.75B of them from long documents.
- A's D384 has 21.3M non-embedding parameters, 0.6B tokens at 4K plus **60M at 32K**. That is about **12× fewer long-document tokens at twice the length** [INF arithmetic]. My own D512 plan (0.2B at 32K) is also far below Qiao's budgets, which is why B1 made feasibility the first gate.

**What A's gates cover.**
- S2 (direct supervision) shows that the *task* is learnable when trained directly. It says nothing about whether an LM-pretrained 21M model with TEF reaches a competent 32K reference.
- G-P asks only for G ≥ 15 pp on "≥ 2 levels per family" at any length.
- G-I uses un-annealed 4K checkpoints (W9).
- None of them can stop a 22-run Tier 1 (45 GPU-h) whose 32K cells are uninterpretable.

**Fix: B1's FEAS gate decides the scale.**
- **Runs:** k ∈ {0, 12-twin} at D384, seeds 101–102. Stage A 0.3B, then Stage-B branches at 32K of 30M and 90M tokens, then TEF checkpoints at 250 and 1,000 steps. Evaluate dev R-load at 8K and 32K, plus C0. Cost ≈ 5 GPU-h at D384 [INF from MEAS-B2 throughput].
- **Pass rule:** FEAS-1 (S₁₂(32K) ≥ 0.8 · S₁₂(C0)) and FEAS-2 (S₁₂ − S₀ ≥ 20 pp at 32K). FEAS-3 sets the Stage-B budget from the 30M → 90M gain, as in B1.
- **Fallback ladder:**
  1. D384 fails → repeat at D512 with my B1 S2 branches (≈ 6–8 h).
  2. D512 passes → discover at D512, with the run set reduced as in §5.
  3. Both fail at 32K → L_ext = 16K.
  4. Fail at 16K too → re-scope with the user.

### W3. Pooled-variance inference is anti-conservative under the heteroscedasticity A predicts, and A's guard does not fire (serious)

**What A does (§2.10).**
- One residual variance, pooled over all 22 Tier-1 runs (df ≈ 11).
- A Brown–Forsythe (BF) test across k-groups; Welch–Satterthwaite if p < 0.05.
- In M2, A also predicts that configurations near a capability transition will have **bimodal** seed outcomes (e.g. 85 / 80 / 20 %).

**Simulation [INF].**
- Design: A's exact Tier-1 layout (P{0, 1, 3, 12} × 3 seeds, P{2, 6} / Ref / PE-alt × 2), with seed blocks (τ = 2 pp), G = 30 pp and 20,000 trials per cell.
- Test: one-sided α = 0.05 on L₃.
- Each cell shows type-I error at a true R₃ = 0.80 / power at a true R₃ = 1.00.

| Scenario | A pooled (df 11) | A rule (BF → Welch) | Paired, 3 seeds (df 2) | Paired, 5 seeds (df 4) | BF fires |
|---|---|---|---|---|---|
| Homoscedastic, σ = 3 pp | 0.051 / 0.81 | 0.053 / 0.81 | 0.052 / 0.54 | 0.050 / 0.88 | 6.0 % |
| k = 3 heteroscedastic, σ₃ = 8 pp | **0.145** / 0.56 | **0.145** / 0.56 | 0.051 / 0.22 | 0.051 / 0.38 | **1.8 %** |
| k = 3 bimodal (each seed fails w.p. ⅓, −25 pp) | **0.288** / 0.37 | **0.287** / 0.37 | **0.227** / 0.29 | **0.131** / 0.21 | 2.0 % |
| k = 1 and k = 3 both bimodal | 0.193 / 0.33 | 0.193 / 0.33 | 0.226 / 0.29 | 0.131 / 0.21 | 0.6 % |

**Readings.**
1. Under homoscedasticity, A's pooled test is valid and as powerful as 5 paired seeds.
2. Under the heteroscedasticity A predicts, it **triples the false-support rate**. BF with 2–3 runs per group is nearly powerless (it fires 1.8 % of the time), so A's protection is nominal.
3. **Under bimodality, no few-seed test is valid.** The false-support rate is dominated by never sampling the failure mode: (⅔)³ = 0.30 and (⅔)⁵ = 0.13.
   - A's bimodality trigger (seed span > 40 pp) never fires on a 25 pp mode.
   - My B1 trigger (σ_d > 5 pp) fires only when a failure has been observed.

**Fix (a change for me as well; C2).**
- **Primary inference for H1 and H1-min:** paired per-seed L_k contrasts with **5 seeds** for k ∈ {0, 1, 3, 12}.
- **Sensitivity analyses:** A's pooled-variance test and the hierarchical bootstrap.
- **Reserve seeds, effect-blind:** add them (up to 8 per configuration) when any confirmatory configuration's seed **range exceeds 12 pp**. At σ = 3 pp that is 4.0σ, just under the 99th percentile of the range for n = 3 (4.12σ), so it fires falsely in about 1 % of configurations, but it does fire on a 25 pp mode once one failure is seen.
- **New estimand:** per-configuration success counts with Clopper–Pearson CIs (success = above the midpoint between endpoint means).
- **Use A's S2 screen to look for bimodality cheaply:** 6–8 cheap direct-supervision seeds per k, to find configurations at risk before LM seeds are allocated.

### W4. A's falsification rules do not falsify

1. **"Level A: falsification in principle"** (A §3) rests on the assumption that "LM pretraining cannot add capacity that direct supervision cannot elicit". This fails in both directions [INF]:
   - A failure under direct supervision with ≤ 100M tokens and a two-LR budget can be an *optimisation* failure. Zoology sweeps 4 LRs per model for this reason.
   - Pretraining can supply features that direct training on random keys does not: token identity, previous-token and induction circuits.
   - A itself lists this as unresolved (A §5, Q10).
   - **Fix:** relabel Level A as a *task and ladder screen* that provides empirical capacity anchors (N50; see W5(a)), not a falsification.
2. **"Level B: falsified if, at any pre-registered informative R-load level, the upper bound of L₃ is < 0"** tests 4 levels with no multiplicity control.
   - The family-wise false-falsification rate is 1 − 0.95⁴ = **18.5 %** for independent levels, and between 5 % and 18.5 % for positively correlated levels [INF].
   - Informative levels are also selected on endpoint scores (S₀, S₁₂) that appear in L₃. Seed-level noise is shared, so the selection is correlated with the test noise.
   - **Fix:** falsify on the ladder mean, which is unselected. Report per-level results with Holm across levels. Judge informativeness on dev items, against thresholds frozen at G-F1.

### W5. Internal inconsistencies in A's design

**(a) The capacity anchor contradicts the state-size arm.**
- A anchors the ladder on **H·d_k** (192 at D384) and calls levels "0.04–2.7× of capacity".
- GDN-E is 1 head × d_k 256 (D512) or 192 (D384). Its H·d_k is **identical** to GDN-A's 4 × 64 (or 3 × 64). Under A's own anchor, H6′ manipulates nothing.
- Under the alternative view, where every head writes every fact, capacity is set by **per-head d_k**, and GDN-E is a 4× (3×) increase.

My simulation [INF] shows which view holds:
- Setup: random token embeddings, per-head random projections, L2-normalised keys, β = 1 delta-rule writes, no decay, readout = concatenation of head outputs. Values are the mean cosine to the true value.

| N facts | 32 | 64 | 128 | 256 | 512 |
|---|---|---|---|---|---|
| GDN-A geometry (4 × 64 / 128) | 0.84 | 0.69 | 0.50 | 0.33 | 0.23 |
| GDN-E geometry (1 × 256 / 512) | 0.95 | 0.91 | 0.84 | 0.74 | 0.64 |

- Without learned partitioning across heads, **per-head d_k governs interference**.
- Degradation is **gradual from N ≪ d_k**: 0.84 already at N = 32 for 64-dimensional heads.
- So the ladder levels are not phase boundaries, and the term "theory-anchored" overstates them.
- This agrees with 2609.33093, whose abstract reports "cross-fact causal coupling rather than independent KV-like storage" [abstract-only].
- Caveat: trained models can learn more orthogonal keys, β < 1 and head partitioning. That is an empirical question.

**Fix.**
- Anchor N *empirically*: N50 of the k = 0 model under direct supervision (A's S2) and in the final LM. Report N / N50₀.
- State H6′ as "per-head key dimension ×4 (×3) at equal parameters and equal H·d_k".

**(b) A's threshold prior undermines A's Tier-2 arms.**
- A-M9 (which I verified) predicts S(1) ≈ S(0) for GDN at 11:1.
- Yet three of A's Tier-2 estimands need k = 1 to differ from k = 0, or need k = 1 to have dynamic range:
  - H6′, the share of the k = 0 → 1 gap closed by GDN-E (arm S);
  - H-conv, [S₁ − S₀]_conv − [S₁ − S₀]_no-conv (arm C);
  - H4′, the depth contrast of the single layer, S(6) − ½[S(0) + S(11)] (arm D).
- Arms S, C and D are 12 runs (≈ 25 GPU-h). If the prior holds, the first two denominators vanish, all three contrasts sit near the floor, and the arms are uninformative.

**Fix: conditional targeting.**
- After Tier 1, define k* as the smallest k ∈ {1, 2, 3} whose R_ret has a lower CI ≥ 0.3 and an upper CI ≤ 0.9, i.e. the rising part of the curve.
- Run S, C, depth and GLA at k*.
- This uses composition results only, so placement and state contrasts stay unbiased. It also fixes my own k = 1 placement study (C3).

**(c) Ratio estimators are back.** A rejects R for H1 because ratio CIs are unstable as G → 0 (A-M1), yet its H3 is "R_ret(3) − R_hop(3) > 0" and its H6′ is a gap-share ratio.
- **Fix:** Fieller intervals for ratios; test H3 as the family × k interaction on the logit scale (GLMM), with R differences descriptive.

**(d) Difficulty-equated H3 may be undefined.** Levels where the k = 12 accuracies of different families lie within ±10 pp may not exist. Pre-register a fallback: the logit-scale interaction on ladder means.

### W6. The twin is not a neutral "full attention", and it weakens the reference where H1 and H5 live

1. **No industrial hybrid puts a short conv in its full layers** (`transformers` 5.22.2 [code-inspected]):
   - `Qwen3NextAttention` has per-head q/k-norm and an elementwise **sigmoid output gate** split from `q_proj` (`attn_output * torch.sigmoid(gate)`), but **no conv**. Only `Qwen3NextGatedDeltaNet` has `conv1d` (`modeling_qwen3_next.py` L229–300, L590).
   - `OlmoHybridAttention` has q/k-norm with no gate and no conv (`modeling_olmo_hybrid.py` L161–210).
   - `KimiLinearAttention` is "MLA … with NoPE" with no conv (`modeling_kimi_linear.py` L190–193).
   - So A's twin is a Primer-like attention ([2109.08668](https://arxiv.org/abs/2109.08668) [abstract-only]), not the attention used in the hybrids the README names. The output gate, by contrast, *is* industrial (Qwen3-Next).
2. **SiLU before QK-RMSNorm is not operator-neutral** [INF, simulation at initialisation].
   - It confines q and k to a cone: the mean cosine of random pairs rises from 0.000 to **0.115**, which shrinks the needle–distractor logit gap.
   - Needle weight at gain² = 1.56, at 1K / 4K / 16K / 32K: 0.95 / 0.83 / 0.56 / **0.39** with SiLU, vs 0.99 / 0.95 / 0.83 / 0.71 without.
   - It is learnable, but within a short Stage B it handicaps the k = 12 reference most at 32K.
   - In GDN, SiLU together with L2 normalisation is part of the q/k feature map of the *linear* kernel. A already treats the norm as operator-specific (QK-RMSNorm in place of L2). The same logic applies to the SiLU.
3. **Four heads vs eight.** A twin layer makes 4 lookups per position, against 8 in plain MHA. This may matter for multi-hop at small k [HYP]. A's bridge arm measures it, but only in Tier 2.

**Fix.**
- **Primer-style twin as primary:** short conv(4) on q/k/v, **no SiLU on q or k** (SiLU on v kept, as in GDN's v path), per-head QK-RMSNorm with **unshared** gains (B1-M3), and the GDN-identical gated per-head RMSNorm output.
- Conv and gate, which are operator-agnostic, are shared. The feature maps (exp vs SiLU + L2) are not. Parameters are unchanged.
- **Bridge in Tier 1:** plain MHA at k ∈ {3, 12} × 2 seeds (4 runs), so that the main H1 contrast exists for the literature design even if Tier 2 never runs.

### W7. Measurement: A inherits PROP §10, and the twin adds a 26× dispatch trap

- A §2.9 keeps PROP §10 for measured costs: CUDA-graph static-cache decode, batch sweep, interleaving. It therefore inherits every artefact in B1-F2:
  - the masked static-cache dispatch (1.80 ms vs 85–90 µs);
  - L2 residency (up to 2.5×);
  - launch-bound GDN;
  - unequal kernel maturity.
- **New [MEAS-B2].** For the twin's natural native implementation (q, k at 64; v at 128), default SDPA dispatch on torch 2.11 / sm_120 selects the **mem-efficient** kernel:
  - decode at 32K, B = 1: **1,700 µs**, vs **66 µs** with cuDNN pinned;
  - causal prefill at 32K: **20.3 ms** vs **10.7 ms**;
  - plain 8 × 64 takes 92 µs under default dispatch, consistent with flash.
- **Head duplication** avoids the trap, but then the measured KV is 2 KiB/token, and the twin's 25 % KV advantage disappears from measured costs:
  - 32K, B = 1: 84–90 µs duplicated vs 67 µs native-cuDNN;
  - 32K, B = 8: 616–625 µs vs **463 µs**, a ratio of 0.75 = 384 / 512 MiB.
- **Fix.** B1-F2's protocol (exact-shape graphs, no masks, backends pinned **and asserted**, cold and hot L2, FlexAttention comparator, roofline table), plus:
  - the twin runs native on cuDNN as primary, with the duplicated layout as a secondary;
  - CI asserts that the twin never reaches EFFICIENT at inference.

### W8. "Tier 3 replication" at D512 is a scale extension, not a replication

- D512 changes H (3 → 4), per-layer state, ladder anchors (0.04–2.7× → 0.03–2×) and LR.
- A non-replication would therefore be uninterpretable: chance cannot be told apart from scale.
- **Fix:** call it a *scale extension* with directional predictions frozen at G-F3. If replication is the goal, use fresh seeds at D384, which is cheaper.

### W9. Interim gate G-I uses predictors that are biased in unknown directions

- At 0.3B tokens a WSD run is mid-stable. The checkpoint is un-annealed, so the decay phase's gains are missing.
- It has only been trained at 4K, so 16K–32K levels are extrapolation even after a 250-step fine-tune.
- The decision to add a harder level can therefore err in either direction.
- **Fix:** freeze one extra top level per ladder from the start (R-load N = 1,024; hops = 5; ρ = 1.1). This costs evaluation only. Decide informativeness on final endpoints, using dev items.

### W10. The LM-stack equivalence gate is both too weak and too strict

- "|Δ val loss| ≤ 0.02 nats after 0.1B tokens" compares models that differ in initialisation and details. Legitimate differences can exceed 0.02 nats, while real bugs can stay below it (LM-loss seed noise is ≈ 0.002 nats per PROP).
- **Fix: training-free weight-transplant logit tests**, tolerance 1e-3 relative on fp32 logits:
  - plain k = 12 ↔ `transformers` `Qwen3ForCausalLM` (per-head `q_norm`/`k_norm` over `head_dim`, SwiGLU, no bias, `tie_word_embeddings=True`, `num_key_value_heads = num_attention_heads`; `modeling_qwen3.py` L218–230 [code-inspected]);
  - k = 0 ↔ fla `GatedDeltaNetForCausalLM` with `intermediate_size=1536`, `tie_word_embeddings=True` and a matched `norm_eps` (its config exposes these fields and an `attn` hybrid option; `fla/models/gated_deltanet/configuration_gated_deltanet.py` [code-inspected]).
- Keep A's 0.1B-token run only as a smoke test.

### W11. Shared-GPU operations and budget variance are not addressed

- **Memory.** During this session CARLA's footprint rose to 6.25 GB, and other processes used ≈ 7.9 GB [MEAS-B2], not the ≈ 5.9 GB in the brief.
- **Throughput variance.** The same configuration's contended Stage-B throughput varied by up to 28 % between consecutive rounds (D512 GDN-A k = 0: 71.7K vs 91.7K tok/s).
- **Budget.** A's 189 h plan has no contingency line.
- **Fix:** a memory cap ≤ 7 GiB (every measured peak is ≤ 4.7 GiB, so this is feasible), 20 % contingency, and checkpoints every 250 steps (B1 §2.10).

### W12. Smaller points

- **Filler-type manipulation in R-dist.** Repetitive filler lowers GDN interference *and* changes softmax dilution (many identical keys). Analyse filler effects per architecture (S₀ and S₁₂ separately), not only on the gap Δ₀.
- **Determinism.** A is silent on determinism. It matters for kill-and-resume validation on a shared GPU, where OOM kills are expected (P7).

---

## 3. Where I change my mind

| # | Was (B1) | Now | Why |
|---|---|---|---|
| **C1** | Conv only as a conditional Tier-C control (E19: Attn+conv if a hybrid beats k = 12) | **Operator isolation in the core:** the Primer-style twin of W6 is the primary full mixer, with plain MHA as a Tier-1 bridge at k ∈ {3, 12} | A-F2 is right that the asymmetry acts on the whole curve: one-layer induction and earlier circuit formation through token mixing (Olsson, verified verbatim). Conditioning the control on "a hybrid beats k = 12" misses its effect on the denominator S₁₂ |
| **C2** | "Supported" requires the bootstrap **and** a seed-level paired t (df = 2) | **L_k contrasts; paired 5-seed tests for k ∈ {0, 1, 3, 12} as primary; pooled-variance and bootstrap as sensitivity; range-triggered reserve seeds** | A-M1 applies to my design too: at σ = 3 pp, 3-seed paired power is 0.54 at R₃ = 1.0 (W3 simulation). Five seeds raise it to 0.88 and stay valid under heteroscedasticity, where pooled variance gives 14.5 % false support |
| **C3** | Placement as k = 1 at layers {1, 6, 11} × 3 seeds (Tier B) | **Placement at a conditional k*** (W5(b)), depth scan {early, middle, late} × 3 seeds, analysed with **A's two contrasts** (middle − mean(ends); linear trend), Holm over 2 | The GDN row in Wang Table 7 shows no gain at 12:1, so a k = 1 triad may compare three near-floor arms. A's two contrasts have MDE ≈ 7 pp at 3 seeds, vs ≈ 9 pp for 3 Holm pairwise contrasts [INF] |
| **C4** | NoPE as a Tier-C robustness arm | **NoPE at k ∈ {3, 12} × 2 seeds in Tier 1.** k = 0 is shared because GDN has no position encoding, so R₃ can be computed under NoPE | Qiao: NoPE on the full layers moves a 1:1 SWA-128 hybrid's RULER-NIAH@16K from 49.58 to 67.81 at S4 (verified; an SWA hybrid, not a linear-attention one). An effect that large on hybrids can change R₃. I prefer A's {1, 12} pair moved to {3, 12} so that it hits the H1 contrast |
| **C5** | B1-M4: k ≥ 1 nondeterminism is "an asymmetric noise source between arms" | **A reproducibility issue, not a bias.** Keep deterministic mode for G0, kill-and-resume and one standing replicate; production deterministic mode is optional (+7 %) | "All sources of nondeterminism have similar effects on measures of model diversity" and "even one-bit changes in initial parameters result in models converging to vastly different values" ([Summers & Dinneen, 2103.04514](https://arxiv.org/abs/2103.04514) [abstract-only]). Final-metric variance is seed-level either way. This also means seed-index blocking buys little, which is harmless in A's model |
| **C6** | (implicit) D384 would not save as much as A claims | **A's 1.35× assumption is correct** (1.30–1.39× measured). D384 is a sound *compute* choice | MEAS-B2. My objection to D384 is capability feasibility (W2), not cost |
| **C7** | TEF had no length balancing and a 50/50 synthetic/natural mix | **Adopt A's equal items per length bucket** (7 buckets, 512 to 32K), **15 % PG19 4K windows in Stage A**, and A's 90/10 mix with a forgetting gate (LM-2 rise ≤ 0.05 nats, and k-differential ≤ 0.02 nats) | A-M3's arithmetic is right: 7,200 short vs 370 long items under PROP option (i). Synthetic items mostly consist of natural filler anyway, so 90 % "synthetic" is milder than it sounds |
| **C8** | GLA in Tier C | **GLA-twin in Tier 2, at k ∈ {0, k*, 3}** | Wang Table 7 shows mechanism-specific curve shapes: GLA rises steeply at the first step, GDN is flat and then jumps. H3 and H2 claims would otherwise be "for GDN" only |
| **C9** | H6 descriptive | **H6′ as secondary** (W5(a) wording, conditional on k*), plus A's analytic projection to d ∈ {384, 512, 2048, 4096} | A-M6 is right. Measured costs at d = 512 overstate attention's share by ≈ 8× relative to d = 4096 |

**Also accepted from A:**
- **M10:** novelty is overstated, given MAD's ratio findings and Based's frontier framing. I take this per A and did not re-verify.
- **M11:** state claims in k at L = 12, not as "proportions".
- **m1:** Zoology targets from re-run numbers only.
- **m3:** inference-only pathway diagnostics (window the full layers vs reset GDN state), exploratory.
- **m8:** unit test that `A_log` and `dt_bias` survive global initialisation.
- **R_best** as a secondary estimand.

---

## 4. Where I hold firm

**P1. A long-length reference-adequacy gate, with the longest adequate length primary.** (B1-F1, W1)
- A's design has nothing that detects a collapsed 32K reference.
- New evidence makes the risk larger:
  - Qiao's 104M all-full model scored 35.95 on RULER-NIAH@16K after ≈ 100B tokens;
  - dilution at d_h = 64 is a small-head artefact (Qwen3-Next uses d_h = 256);
  - the twin's SiLU cone adds to it.
- G-REF costs nothing to run (one C0 condition) and is the only guard that makes a "supported" or "falsified" verdict about *long* context interpretable.

**P2. An LM-regime feasibility pilot decides the discovery scale and the Stage-B budget.** (B1-M1, W2)
- About 3 GPU-h at D384, with the D512 fallback pre-registered.
- A's direct-supervision screen is a complement (task and N50 calibration), not a substitute.

**P3. An artefact-controlled cost protocol, with analytic costs primary.** (B1-F2, W7)
- Two traps of 20× or more are now measured: the masked static cache and the twin's native-d_v dispatch.
- Unpinned measurements cannot carry a frontier claim.

**P4. Seeds where the tests live.**
- ≥ 3 seeds for any confirmatory arm. Two-seed arms (curve points, bridge, NoPE) are descriptive or sensitivity only.
- 5 seeds for the H1 and H1-min anchors (C2).
- Per-seed reporting, and success counts as an estimand.

**P5. RoPE θ = 5e5 as the default, with no selection pilot.** (B1-M7)
- A's pilot (k = 12 × 2 PE × 2 seeds; 0.3B + 30M at 32K, i.e. ≈ 229 steps at 32K, plus a fine-tune) chooses on 32K retrieval at a budget where both arms are probably weak.
- Because the alternative PE runs in Tier 1 anyway (C4), the pilot only decides which scheme is "primary", and it costs ≈ 4 GPU-h of A's S3.
- Pre-register RoPE θ = 5e5 as primary, with no override.
- The Tier-1 NoPE arm (k ∈ {3, 12} × 2 seeds) is reported alongside it. A's R_best estimand covers the case where NoPE gives the stronger all-full model.

**P6. Exposure as a factor with a plateau criterion.** Converged with A: a dose ladder with a "still closing" flag. I keep the requirement that H1 *falsification* needs the 4,000-step branch to agree (B1 §3, condition 4).

**P7. Deterministic G0 and bitwise kill-and-resume.**
- On a GPU shared with a simulator that restarts, OOM kills will happen.
- Bitwise resume (≈ +7 % if production also runs deterministically; ≈ 0 if only G0 does) is the cheapest way to verify that a resumed run equals an uninterrupted one.

**I also keep from B1:**
- per-head QK gains, with SSMax log-length scaling as a FEAS-decided option (B1-M3), now doubly motivated by W6;
- the canonical scoring path with a 5 % single-pass re-score (B1-m2);
- the warmed and archived Triton cache;
- the 250-step checkpoints and the memory-capped runner.

---

## 5. Reconciled design for the PI (my position entering round 3)

### 5.1 Decision table

| Item | A (round 1) | B (round 1) | **B (round 2)** |
|---|---|---|---|
| Full mixer | Twin: conv + SiLU on q/k/v, gate, 4 heads | Plain MHA + per-head gains / SSMax option | **Primer-style twin:** conv on q/k/v, no SiLU on q/k, gate, per-head gains. **Bridge:** plain MHA at k ∈ {3, 12} in Tier 1 |
| Discovery scale | D384 (D512 "replication") | D512 | **FEAS decides:** D384 if FEAS-1/2 pass there, else D512. D512 later is a "scale extension" |
| H1 primary cell | R-load ladder mean at 8K | Long band 16–32K, G-REF | **R-load at {8K, 32K}; primary = longest G-REF-passing length; unselected ladder mean** |
| H5 | Distance ladder at 32K, filler types | Slope in log₂T plus a distance-controlled design | **A's distance ladder plus load × length**, G-REF-passing cells only; filler analysed per architecture |
| Inference | Pooled variance, BF → Welch | Bootstrap **and** df = 2 seed-t | **Paired L_k with 5-seed anchors; pooled and bootstrap as sensitivity; range trigger at 12 pp; success counts** |
| Confirmatory family | H1, H1-min, H3, H5 | H1, H1-min, H3, H4, H5 | **H1, H1-min, H3 (logit-scale interaction), H5**, Holm. H4 at k* and H6′ are secondary |
| Placement | k = 1 depth {0, 6, 11}, Tier 2 | k = 1 at {1, 6, 11} × 3 | **k* depth scan × 3 seeds, A's two contrasts** |
| Position encoding | Pilot {NoPE, RoPE 5e5} + alt at k ∈ {1, 12} | RoPE 5e5; NoPE in Tier C | **RoPE 5e5 default; NoPE at k ∈ {3, 12} × 2 seeds in Tier 1** |
| Stage-B budget | Fixed (60M at D384) | FEAS-decided 0.1–0.3B | **FEAS-decided** |
| Exposure | Fine-tune, equal items per bucket, dose ladder | TEF, dose branch | **Converged:** A's fine-tune + forgetting gate + B's falsification condition |
| Costs | Analytic primary; PROP §10 measured | Analytic primary; artefact-controlled | **B1-F2 protocol plus twin-specific cuDNN pinning and assertion** |
| Equivalence | 0.02-nat LM-stack run | — | **Weight-transplant logit tests** (Qwen3ForCausalLM; fla GatedDeltaNetForCausalLM) |
| Mechanism generality | GLA-twin, Tier 2 | GLA, Tier C | **GLA-twin, Tier 2, at k ∈ {0, k*, 3}** |
| Interim gate | G-I at 0.3B | — | **Extra frozen top ladder level instead** |

### 5.2 Tier 1 on the D384 path (contended GPU-h, from MEAS-B2 throughput)

**Per-run cost at D384** [INF]: Stage A 0.6B at ≈ 135K tok/s ≈ 1.23 h; Stage B 60M at 0.15–0.37 h (k = 0 → k = 12); fine-tune 65.5M at 0.17–0.40 h. Total ≈ 1.6–2.0 h, mean ≈ 1.85 h.

| Block | Runs | GPU-h |
|---|---|---|
| S0 infrastructure: G0 numerics with relative thresholds; deterministic resume; weight-transplant equivalence; harness self-test; Triton cache | — | 3 |
| S1 Zoology MQAR (re-run numbers only) | small | 3 |
| A's direct-supervision screen: N50 anchors and a bimodality screen, 6 seeds × k ∈ {0, 1, 3, 12} at 8K, plus GDN-E / GLA / plain at k ∈ {0, 3} × 1 | ≈ 30 tiny | 5 |
| FEAS at D384: k ∈ {0, 12} × seeds 101–102; Stage-B branches 30M / 90M; TEF 250 / 1,000 | 4 | 5 (D512 fallback +8) |
| LR sweep: k ∈ {0, 3, 12} × 3 LRs × 0.15B | 9 | 3.5 |
| **Core:** P k ∈ {0, 1, 3, 12} × 5 seeds | 20 | 37 |
| Curve: P k ∈ {2, 6} × 2 seeds (descriptive) | 4 | 7.4 |
| Bridge: plain k ∈ {3, 12} × 2 | 4 | 7.6 |
| NoPE: twin k ∈ {3, 12} × 2 | 4 | 7.6 |
| Dose extension to 4,000 steps, 4 anchors (seed 0); 2× Stage-B branch at k ∈ {3, 12} (seed 0) | branches | 6 |
| Evaluation (≈ 0.2 h per model) and cost benchmarks | — | 7 + 4 |
| **Tier 1 total** | 32 main runs | **≈ 96 → ≈ 115 with 20 % contingency** |

**Tier 2 (conditional on k*), ≈ 40 h.**
- k* depth scan: 6 new runs.
- GDN-E at {0, k*} × 2.
- No-conv at {0, k*} × 2.
- GLA-twin at {0, k*, 3} × 2.
- Gated plain-attention control, if a hybrid beats k = 12.

**Tier 3.** D512 scale extension at k ∈ {0, 3, 12} × 3 seeds (≈ 40 h), optional.

**If FEAS forces D512.** Per-run cost is 3.2–4.5 h. Tier 1 becomes:
- P k ∈ {0, 1, 3, 12} × 4 seeds (16 runs, ≈ 62 h);
- bridge and NoPE at 1 seed each (4 runs, ≈ 16 h);
- k ∈ {2, 6} deferred to Tier 2.

That is ≈ 105 h, or ≈ 125 h with contingency. Under homoscedastic σ = 3 pp, the exact one-sided paired-t power at R₃ = 1.0 is 0.54 / **0.76** / 0.88 / 0.94 for 3 / 4 / 5 / 6 seeds [INF, noncentral t].

### 5.3 Decision rules (changes relative to B1 §3)

**H1 falsified** (for the stated scale, GDN-A and this recipe) only if **all** of the following hold:
1. FEAS, G0 and G-RP pass.
2. G-REF passes at the primary length for ≥ 4 of 5 seeds.
3. The one-sided 95 % upper bound of the paired per-seed L₃ is < 0 on the unselected ladder mean, and the pooled-variance and bootstrap analyses agree in sign.
4. The 4,000-step dose branch keeps L₃ < 0, and k = 3's learning curve has plateaued (< 3 pp change from 2,000 to 4,000 steps).
5. The 2× Stage-B branch changes S₁₂ at the primary length by ≤ 5 pp. Otherwise the verdict is "conditional on the long-stage budget".

**H1 supported:**
- the lower bound of L₃ > 0 under the same tests;
- G-REF passes;
- the bridge (plain) k = 3 vs k = 12 contrast has the same sign. If it does not, report "supported for twin mixers only".

---

## 6. Questions for round 3

1. **Is the Primer-style twin (no SiLU on q/k) acceptable to A?** My claim is that SiLU belongs to GDN's feature map, not to the operator-agnostic front end. A cheap check: one extra k = 12 run with A's exact twin in the FEAS pilot (+1.5 h).
2. **Does A accept G-REF?** It is the main open disagreement. If A thinks an 8K primary suffices, what observation would show that the 32K reference is competent?
3. **Five seeds vs pooled variance.** Under heteroscedasticity, does A accept that the BF guard is nominal (it fires 1.8 % of the time)? Is there a guard A prefers with demonstrated power at n = 2–3?
4. **Capacity anchor.** Does A agree that H6′ must be stated in per-head d_k terms, and that ladder levels should be anchored empirically (N50)?
5. **Bimodality.** Should "P(success) per configuration" become a named secondary estimand, so that a transition region is reported as a finding rather than as noise?
6. **Remaining uncertainties for both of us:**
   - the idle-GPU speed-up (CARLA showed only restart gaps of tens of seconds today);
   - whether TEF-elicited capability or zero-shot LM-3/LM-4 answers the README question when they disagree (A §5, Q1).

---

## Appendix A. Measurement and simulation log [MEAS-B2 / INF]

**Conditions.**
- Stack: torch 2.11.0+cu128, Triton 3.6.0, fla 0.6.0 (`98193113`), transformers 5.22.2.
- GPU: RTX 5080 shared with CARLA, 15–42 % utilisation. CARLA restarted once (CarlaUE4 3.9 → 6.25 GB; foreign total ≈ 7.9 GB). SM clock ≈ 2.69 GHz. My process was capped at 0.35–0.40 of 15.9 GiB.
- Timing: CUDA events, min-of-N.
- Scripts: `evne.py`, `model_bench_r2.py`, `num_test_r2.py`, `twin_decode.py`, `twin_dispatch.py`, `capacity_sim.py`, `stats_sim.py` and `silu_cone.py`, in the session scratchpad under `revB2/`.

| # | What | Setup | Result |
|---|---|---|---|
| B2-1 | Attention backends with d_qk = 64, d_v = 128 | B 1, H 4, T 4,096, bf16, causal; forward + backward vs fp32 math | FLASH: "No available kernel". EFFICIENT, cuDNN, MATH, FlexAttention: OK (max err 5.95e-3 / 5.95e-3 / 3.91e-3 / 5.95e-3). `varlen_attn`: rejects ("v must have shape (total_k, num_heads_k, head_size_og)"). Duplication into 8 × 64 via `varlen_attn`: max err 5.95e-3, i.e. exact |
| B2-2 | Whole-model training throughput, 2 interleaved rounds | As B1-A4 (doc-masked lognormal documents, fused AdamW, chunked CE), RoPE θ 5e5. **Stage A:** 2 × 4K, no checkpointing, 30 steps after 8 warm-ups. **Stage B:** 1 × 32K with checkpointing, 6 after 3. Values are best (other round) in K tok/s | **Stage A, D512:** GDN-A k0 114.5 (98.8); plain k12 117.4 (117.2); twin k12 104.3 (92.5); GDN-E k0 97.1 (93.3); twin k3 112.9 (101.2). **Stage A, D384:** GDN-A 150.3 (149.9); plain 163.6 (142.5); twin 136.5 (128.6); GDN-E 133.7 (129.7); twin k3 147.6 (125.9); GLA 128.5 (111.1). **Stage B, D512:** GDN-A 91.7 (71.7); plain 35.9 (35.8); twin 37.0 (30.3); GDN-E 69.0 (55.4); twin k3 66.6 (53.8). **Stage B, D384:** GDN-A 119.6 (97.2); plain 49.0 (40.3); twin 49.6 (40.4); GDN-E 99.8 (80.8); twin k3 71.6 (71.5); GLA 101.1 (99.9). Peak memory: 3.3–4.7 GiB (A), 1.8–2.6 GiB (B). All losses finite. Non-embedding parameters: D512 GDN-A 41.007M, plain 40.909M, twin 40.959M, GDN-E 40.975M; D384 21.309M / 21.245M / 21.283M / 21.294M; GLA 21.395M |
| B2-3 | GDN chunk numerics vs fp64 naive recurrence | T 8,192; (H, K, V) ∈ {(4, 64, 128), (1, 256, 512), (1, 192, 384)}; mixed and near-unit decay | Relative RMS: TF32 1.45–1.64e-3, bf16 4.9–6.2e-3, flat across position bands. Final-state relative max ≤ 1.4e-3 (fp32), ≤ 7.1e-3 (bf16). Chunk prefill + `fused_recurrent` tail vs single chunk pass: ≤ 2.0e-3 (bf16) |
| B2-4 | Twin decode, exact-shape graphs, cold L2 (512 MB flush), min of 40 | Native (4 × 64 q/k, 4 × 128 v) vs duplicated (8 × 64) | **32K, B = 1:** native cuDNN 67 µs (roofline 52), native EFFICIENT 1,698, duplicated flash 90, duplicated cuDNN 84 (roofline 70). **32K, B = 8:** 463 / 1,701 / 625 / 616 (rooflines 419 / 559). **4K, B = 1:** 16 / 219 / 23 / 16. **4K, B = 8:** 65 / 221 / 90 / 84 |
| B2-5 | Default SDPA dispatch, eager, cold L2 | All four backends enabled | Native d_v ≠ d_qk: decode at 32K 1,700 µs (= EFFICIENT; cuDNN pinned 66 µs); causal prefill at 32K 20.28 ms (= EFFICIENT 20.26; cuDNN 10.72). Plain 8 × 64 decode: 92 µs |
| B2-6 | Delta-rule capacity simulation | numpy; random embeddings; per-head random projections; L2-normalised keys; β = 1; no decay; 3 trials | Mean cosine to the true value at N = 16 / 32 / 64 / 128 / 192 / 256 / 512 / 1,024. 4 × 64: 0.92 / 0.84 / 0.69 / 0.50 / 0.40 / 0.33 / 0.23 / 0.18. 1 × 256: 0.98 / 0.95 / 0.91 / 0.84 / 0.78 / 0.74 / 0.64 / 0.58. D384: 3 × 64 ≈ 4 × 64; 1 × 192 ≈ 1 × 256. Top-1 among the N true values is lenient in high dimensions and not used |
| B2-7 | H1 test operating characteristics | scipy 1.14.1 in an isolated `uv run` environment (the project venv is untouched); A's Tier-1 layout; τ_seed = 2 pp; G = 30; 20,000 trials per cell | See the table in W3. Exact noncentral-t power of the paired test at R₃ = 1.0 (σ = 3 pp): 0.543 / 0.755 / 0.878 / 0.942 for n = 3 / 4 / 5 / 6. 99th percentile of the range for n = 3: 4.12σ |
| B2-8 | SiLU before QK-RMSNorm | numpy; d_h = 64; unit vectors; logit = 8·g²·cos; 40K sampled distractors | Mean distractor cosine 0.000 (plain) vs 0.115 (SiLU), SD 0.125 / 0.118. Needle weight at 1K / 4K / 16K / 32K. g² = 1: plain 0.64 / 0.31 / 0.10 / 0.05, SiLU 0.43 / 0.16 / 0.05 / 0.02. g² = 1.56: plain 0.99 / 0.95 / 0.83 / 0.71, SiLU 0.95 / 0.83 / 0.56 / 0.39. g² = 2.25: plain 1.00 / 1.00 / 1.00 / 0.99, SiLU 1.00 / 0.99 / 0.98 / 0.96 |

## Appendix B. Sources checked in round 2

| Source | Used for | Access |
|---|---|---|
| Olsson et al., [2209.11895](https://arxiv.org/abs/2209.11895) (local PDF text) | Smeared keys enable one-layer induction and earlier formation (L519–539); phase change at "1-3 billion tokens" in small models (L925) and "2.5 to 5 billion" (L349) | [read-section] |
| Primer, So et al., [2109.08668](https://arxiv.org/abs/2109.08668) | Precedent: "adding a depthwise convolution layer after each Q, K, and V projection in self-attention" | [abstract-only] |
| Summers & Dinneen, [2103.04514](https://arxiv.org/abs/2103.04514) | "All sources of nondeterminism have similar effects on measures of model diversity"; one-bit changes lead to "vastly different values" | [abstract-only] |
| Lee, Park, Kim, Ko, "How Linear Attention Remembers", [2609.33093](https://arxiv.org/abs/2609.33093) | Load effect larger than elapsed context "within the tested regime"; pretrained GLA/GDN, sizes not stated; cross-fact coupling; hybrids shift recall to the KV state | [abstract-only] |
| Qiao et al., [2606.15378v1](https://arxiv.org/html/2606.15378v1) | S1–S5 = 15 / 31 / 65 / 104 / 477M non-embedding; 16K context, 1:1 long/short; D ∈ {100N … 1000N}; S1@100N excluded as "too small for stable long-context behavior"; RULER-NIAH@16K at S4: Full 35.95, SWA-128 49.58, SWA-128-NoPE 67.81 (S5: 67.14 / 65.91 / 82.31); 1:1 main ratio; S4/S5 variants trained on ≈ 100B tokens | [read-section] |
| Wang et al., [2507.06457](https://arxiv.org/abs/2507.06457) (local PDF text, Table 7) | Per-mechanism recall by ratio; GDN flat up to 12:1 | [read-section] |
| `transformers` 5.22.2 (project venv) | `Qwen3NextAttention`: sigmoid output gate, per-head q/k-norm, no conv (L229–300); `Qwen3NextGatedDeltaNet` conv1d (L590); config `head_dim=256`, linear key/value head dim 128, `full_attention_interval` 4 (L93–131); `OlmoHybridAttention`: no gate or conv (L161–210); `KimiLinearAttention`: MLA with NoPE (L190–193); `Qwen3Attention`: per-head `q_norm`/`k_norm`, SwiGLU (L75–81, L218–230) | [code-inspected] |
| fla @ `98193113` (local clone) | `assert K <= 256` (`chunk_delta_h.py` L716, L772); GLA conv/gate options (`gla.py` L134–163); `ShortConvolution` (`causal_conv1d/ops.py` L1118); `FusedRMSNormGated` (`norm/fused_norm_gate.py` L967); `GatedDeltaNetConfig` fields | [code-inspected] |
| MAD [2403.17844](https://arxiv.org/abs/2403.17844), Based [2402.18668](https://arxiv.org/abs/2402.18668) | Novelty framing (A-M10) | per Review A; **not re-verified by me** |

**Scope note.** This review does not edit `README.md`, `reports/decisions.md`, PROP or any other agent's file. Nemotron-H ([2504.03624](https://arxiv.org/abs/2504.03624)) is not used here. It mixes attention, Mamba-2 and MLP layers and is not a controlled full-vs-linear ratio experiment.

# Review A, round 2: adversarial critique of Review B, and revised positions

**Reviewer:** A (scientific validity, adversarial). **Date:** 2026-11-09.
**Objects.**
- `reports/reviews/review_B_round1.md` ("B", "Design B"), read in full.
- My own `reports/reviews/review_A_round1.md` ("A1").
- `reports/initial_proposal.md` ("PROP").

**New verification this round** (Appendix A has the log):
- Code inspection of transformers 5.22.2 in the project venv: Qwen3-Next, Kimi-Linear and OLMo-hybrid attention layers.
- Re-checked the fla `98193113` lines that B cites.
- Primary sources: gated attention, Primer, Summers & Dinneen, QKNorm, Llama 3, and Wang et al. v1 *and* v2.
- Three CPU simulations, tagged **[MEAS-A]** (Appendix B):
  - softmax dilution vs retrieval readout;
  - power of the H1 decision rules;
  - an artefact in B's length ladder.

**What I did not run.**
- **No GPU code.** The GPU is shared with CARLA and other project processes.
- **B's GPU measurements** are cited as **[MEAS-B]** and were not re-run by me.

**Conventions.** As in A1.
- Access tags: [read-full], [read-section], [abstract-only], [code-inspected].
- **[EST]** established in the cited source; **[HYP]** hypothesis; **[INF]** my inference or arithmetic.
- Severity: FATAL (the claimed conclusion cannot follow) > MAJOR > MODERATE > MINOR.
- k = number of full-attention mixers among L = 12.
- Nemotron-H is not used anywhere below.

---

## 0. Verdict

Design B is the better engineering plan. In four respects it is also the better scientific plan than my own A1, and I adopt all four:
- feasibility first;
- the long band as the primary H1 cell, behind a reference-adequacy gate;
- artefact-controlled cost measurement;
- a budget built on measured throughput.

**Design B nevertheless cannot deliver its own headline verdicts.**

1. **B-1. Its H1 decision rule almost never says "supported" when H1 is true.**
   - In my Monte Carlo (G = 30 pp, σ_d = 3 pp, 3 seeds): P(supported | R = 0.9) = **0.11** and P(supported | R = 1.0) = **0.36**. When H1 is false, P(falsified | R = 0.7) = **0.21**.
   - The non-inferiority conjunct alone requires k = 3 to *beat* all-full by about 2.5 pp on average.
2. **B-2. The conv and gate asymmetry between mixers stays in place.**
   - The only control is in Tier C, and it fires on the wrong event ("a hybrid beats k = 12"). The asymmetry biases every recovery ratio R through its denominator.
3. **B-3. The H5 test has a built-in artefact.**
   - The R2 ladder adds harder levels as T grows (rule 12N ≤ 0.75T).
   - This alone produces a slope of **1.0–3.6 pp per doubling**, against B's ±2 pp equivalence margin.
4. **B-5, B-6. H6 and the upper half of H3 cannot be tested in Tiers A–B.**
   - For any cost that adds up layer by layer (state bytes, FLOPs, sequential latency), B's hull margin reduces *exactly* to R(k) > k/12.
   - There is no second pure-architecture point, and k ∈ {4, 6} is absent.
5. **B-4. G-REF as specified can wipe out the evidence.**
   - Per-seed exclusion can leave a cell with 2 seeds (df = 1).
   - It deletes exactly the outcome the closest small-scale study observed: an all-full model *below* hybrids at long length.

**Moderate issues:**
- M3, the dilution argument, is overstated: attention weight is not retrieval accuracy (my simulation).
- The SSMax adoption rule is decided by noise.
- Determinism is priced as a validity requirement when it is only replicability hygiene.
- N50 is right-censored for the models that matter.
- The ladder is bracketed on a weaker pilot.
- FlexAttention gives an equal toolchain, not equal kernel maturity.

**Net: merge the two designs (§7).**
- Tier A ≈ **108 GPU-h** base, ≈ 130 h with 20% contingency. B's Tier A is 96 / 115 h.
- The difference buys the conv control and the PE control, and drops production determinism.

---

## 1. Strongest parts of Design B (adopted)

| # | Element of Design B | Why it is strong | What I adopt |
|---|---|---|---|
| S1 | **S2 feasibility and reference-adequacy pilot in the LM regime at 32K**: k ∈ {0, 12}, Stage-B budget branches, TEF checkpoints | Neither PROP nor A1 shows that a 41M model can become competent at 32K at all. My direct-supervision screen tests capacity at this width, not LM-regime competence, and competence is what can fail. B's evidence: Qiao et al. report NIAH/RULER only for their ≥ 0.22B models; Wang et al. omit 340M recall as "insignificant" (both per B; the second also verified in A1) | It becomes the first scientific gate. My direct-supervision screen moves to Tier C (C-3) |
| S2 | **Long band {16K, 32K} as the primary H1 cell, plus a reference-adequacy gate** | B's failure table is a valid counterexample to pooling over lengths: pooled R = 1.05 while the reference keeps less than half its C0 competence at 32K. It also hits **A1**, whose H1 load ladder sat at T = 8K | Adopted, with three repairs (B-4) |
| S3 | **Measured cost artefacts** [MEAS-B]: (a) a masked static cache makes SDPA dispatch to the mem-efficient kernel, 1.80 ms vs 85–90 µs per layer at 32K; (b) L2 residency, 33 vs 84 µs; (c) launch-bound GDN at batch 1 | These would have corrupted RQ-D in PROP and in A1, which kept PROP's CUDA-graph protocol as its measured layer | B's §2.7 protocol adopted in full, plus two analytic additions (B-10) |
| S4 | **Measured throughput and memory under contention** [MEAS-B]: Stage A 88.9–90.5K tok/s at 4.5 GiB; Stage B 70.5K / 51K / 31.9K tok/s for k = 0 / 3 / 12 at 2.6 GiB; a 16K micro-batch OOMs under a 7.1 GiB cap | Replaces PROP's 80K tok/s and my own unsupported ÷1.6 "idle" factor | Every budget below uses B's numbers |
| S5 | **One Stage-A run per (k, seed), branched at the end-of-stable checkpoint S, with the LR decay entirely in Stage B** | Turns two confounds, long-stage budget and exposure, into cheap factors | Adopted. A1 decayed across the A/B boundary |
| S6 | **Exposure as a factor**: TEF checkpoints at 250 / 500 / 1,000 steps plus a 4× continuation | Independent agreement with A1 M4 | Agreed |
| S7 | **Engineering corrections**: per-head (unshared) QK-norm gains; relative G0(f) thresholds, since fla uses TF32; an explicit scoring path, because fla switches to `fused_recurrent` when `q_len <= 64` outside training (`fla/layers/gated_deltanet.py` L229–230 @`98193113`, re-checked [code-inspected]); an archived Triton cache; 250-step atomic checkpoints; a memory cap | Each fixes a concrete failure mode | Adopted |
| S8 | **k = 1 placement triad × 3 seeds** | Same direction as A1 M8 (a k = 1 depth scan), with more seeds | Adopted, at positions {1, 6, 11} (C-8) |
| S9 | **RoPE θ = 5×10⁵ from step 0 instead of a 1-seed pilot** | Llama 3 uses this base with 8K pretraining and cites it as "effective for context lengths up to 32,768" ([2407.21783 §3.2](https://arxiv.org/html/2407.21783) [read-section]) | Adopted (C-4) |
| S10 | **N50 capacity metric** | Calibration-free and easy to interpret | Adopted, with a ladder extension (B-9) |

---

## 2. Weakest parts of Design B (adversarial)

### B-1 (FATAL for the H1 and H1-min verdicts). The decision rule is nearly powerless even when H1 is true

**The rule** (B §2.8, inherited from PROP §2 and §11). "Supported" requires all three of:
- lower CI(R) ≥ 0.8;
- upper CI(Δ) ≤ 5 pp, where Δ = S₁₂ − S₃;
- an exposure plateau.

Both the hierarchical bootstrap **and** the seed-level paired t (df = 2) must agree.

**The non-inferiority conjunct.**
- B's own M5 table gives a half-width of 7.4 pp at 3 seeds (4.303 × 3/√3 at σ_d = 3 pp). That already exceeds B's 5 pp margin.
- Upper CI(Δ) ≤ 5 pp therefore needs Δ̂ ≤ −2.45 pp: **k = 3 must beat all-full on average**.
- At B's own G3 tolerance (σ_d = 5 pp), it needs Δ̂ ≤ −7.4 pp.
- So "recovers most of the performance" has silently become "outperforms".

**Monte Carlo** [MEAS-A; Appendix B.2]. Settings: G = 30 pp, 3 seed-blocked seeds, 200,000 replicates.

| True R | B / PROP rule: P(supported) | P(falsified) | A2 rule (L₃, blocked pooled variance, df = 6): P(supported) | P(falsified) |
|---|---|---|---|---|
| 0.6 | 0.000 | 0.552 | 0.000 | **0.952** |
| 0.7 | 0.000 | 0.212 | 0.000 | **0.511** |
| 0.8 (boundary) | 0.009 | 0.026 | 0.050 | 0.050 |
| 0.9 | **0.107** | 0.000 | **0.512** | 0.000 |
| 1.0 | **0.363** | 0.000 | **0.952** | 0.000 |
| 1.1 | 0.656 | 0.000 | 1.000 | 0.000 |

- Results are at σ_d = 3 pp.
- At σ_d = 5 pp, P(supported | R = 1.0) is 0.17 under B's rule and 0.64 under A2.

**Consequence.**
- PROP predicts R(3) ≈ 0.9–1.1. At R = 0.9, B's most likely H1 verdict is "inconclusive" (89%).
- Tier A of B (≈ 115 GPU-h) is aimed at the one question its rule can barely answer.

**Fix** (this is A1 M1, corrected).
- Test the contrast **L₃ = S₃ − 0.2·S₀ − 0.8·S₁₂** (L₃ ≥ 0 ⇔ R ≥ 0.8).
- Fit a seed-blocked linear model over the 4 × 3 core table, which gives residual **df = 6**. Use one-sided 95%, t = 1.943.
- Add a sensitivity analysis that pools only over k ∈ {1, 3, 12} (df = 4, t = 2.132), in case k = 0 is compressed at the floor. Add a Brown–Forsythe check.
- Report the bootstrap and the seed-level t, but do not require them.
- Make the 5 pp non-inferiority margin a secondary estimand.

**Correction to my own A1.** A1's df ≈ 11 assumed 22 runs in one stage. In the merged Tier A it is 6. The conclusion stands; the numbers above are the corrected ones.

**Against myself.**
- Pooling assumes run-level noise is homoscedastic across k. With 3 seeds per configuration, the heteroscedasticity check has little power.
- The df = 4 sensitivity analysis and plotting every seed are the honest guard.

### B-2 (FATAL for any mechanistic reading; MAJOR for the package reading). The mixer asymmetry is left in place, and its only control sits in Tier C behind the wrong trigger

**The asymmetry.**
- Design B pairs plain MHA (8 × 64, no conv, no gate) with GDN-A, which has a causal conv(4) + SiLU on q/k/v and a swish-gated per-head RMSNorm output.
- r is therefore collinear with "the share of layers that have token-shift and output gating" (A1 F2).

**B's only control** is S9c, "Attn+conv at k = 12 if any hybrid beats k = 12", in Tier C. That is PROP's E5b, which A1 F2 already showed to be insufficient.

**Why the trigger is wrong: the denominator.** Illustrative numbers [INF]:
- Package scores: S₀ = 40, S₁₂ = 70, S₃ = 64, so R = 0.80.
- Suppose attention with a conv reaches 76 at k = 12 but only 65 at k = 3, because hybrids already get local features from their GDN layers.
- Then R = 25/36 = **0.69**.
- A 6 pp shift in the reference moves R by about 0.1 without any hybrid beating all-full, so S9c never fires.

**Mechanism evidence.**
- With "smeared keys" (width-2 key mixing), in-context learning forms in one-layer models and earlier in deeper ones ([Olsson et al., 2209.11895](https://arxiv.org/abs/2209.11895) [read-section, verified in A1]).
- Primer's gains come "mostly" from squared ReLU and from "adding a depthwise convolution layer after each Q, K, and V projection in self-attention" ([2109.08668](https://arxiv.org/abs/2109.08668) [abstract-only]).
- A head-specific sigmoid gate after SDPA "consistently improves performance", "enhances training stability, tolerates larger learning rates" ([2505.06708](https://arxiv.org/abs/2505.06708) [abstract-only]).

**B's own text concedes the mechanism.**
- B lists the conv (E19) and the gate (E21) as rival explanations, but only for O6 ("a hybrid beats all-full").
- B explains placement through conv-derived features (E27, "no conv-derived features at layer 1"). In B's own design, layer 1 actually *receives* such features from GDN layer 0, so E27 looks like a slip; but it shows the mechanism is accepted.
- None of these is connected to O1 ("H1 supported").

**A refinement of my own A1 argument** [INF].
- In a hybrid, previous-token features reach the single attention layer from the linear layers below, through the conv *and* through fast-decay recurrence.
- So the asymmetry's main effect on H1 is not that the conv alone enables one-layer induction. It acts through the all-full reference, which has no cheap local mixing anywhere, and through learning speed at a fixed budget.
- This makes the **k = 12 control the decisive one**. A1's conv-free-GDN arm becomes a lower-priority mechanism test.

**What practice does** (transformers 5.22.2 as installed in the project venv; upstream [huggingface/transformers](https://github.com/huggingface/transformers) [code-inspected]):
- **Qwen3-Next** is the closest industrial GDN hybrid (`full_attention_interval` 4, i.e. 3:1). Its attention computes `attn_output = attn_output * torch.sigmoid(gate)`, with the gate taken from a doubled q-projection (`models/qwen3_next/modeling_qwen3_next.py` L269–300). It has per-head QK-norm, `head_dim` 256 and `partial_rotary_factor` 0.25 (`configuration_qwen3_next.py` L107, L128, L131), and **no conv in attention**.
- **Kimi-Linear's MLA** (`models/kimi_linear/modeling_kimi_linear.py` L190–313) and **OLMo-hybrid's attention** (`models/olmo_hybrid/modeling_olmo_hybrid.py` L161–242) have **neither a gate nor a conv**. Their linear layers do have convs.
- So "plain MHA" is not uniquely "the practice", and no practice puts a conv in attention.

**Fix (A2).**
1. **Remove the gate asymmetry by design.** Add a headwise sigmoid gate after SDPA: one scalar per head, W_g ∈ ℝ^{512×8}, +4,096 parameters.
   - In the gated-attention study, headwise gating added 1.6M parameters to a 15B MoE. It gave PPL 5.792, vs 5.761 for elementwise gating and 6.026 for the baseline ([2505.06708](https://arxiv.org/html/2505.06708) Table 1 [read-section]).
   - The cost here is +0.39% per mixer. Parameter matching *improves* (Appendix B.4).
2. **Measure the conv.** Add Attn+conv (FLA-style ShortConv(4) + SiLU on q/k/v, +6,144 parameters) at **k = 12 × 2 seeds in Tier A, unconditionally**.
   - If it beats the primary k = 12 by more than 3 pp, add Attn+conv at k ∈ {1, 3} × 2 seeds (Tier B) and report H1 in both attention families.
3. **Word every claim as a package claim**, i.e. "GDN layers as built, replacing attention layers", unless step 2 shows no dependence on the conv.

**A claim I withdraw from A1.**
- The gate's long-context benefit in that study is *extrapolative*. RULER at 32K without YaRN: 79.50 (baseline) vs 79.77 (gated). With YaRN at 64K: 37.51 vs 66.60 ([Table 5](https://arxiv.org/html/2505.06708) [read-section]).
- Our evaluation is in-distribution, so A1 F2's appeal to "long-context extrapolation" was beside the point.
- The case for the gate rests on symmetry, plus stability and LR tolerance, which matter for the shared-LR rule.

### B-3 (MAJOR). H5 is confounded by ladder composition, exposure per length, and selection of lengths by the gate

**(a) Ladder composition.**
- An R2 level N is used at length T only if 12N ≤ 0.75T:
  - {8, 32} at 1K;
  - {8, 32, 128} at 2K–4K;
  - {8, 32, 128, 512} at ≥ 8K.
- S_f(k, T) is the ladder mean, and H5 is the slope of the endpoint gap in log₂T.
- Suppose the gap is *constant in T at every level*, i.e. there is no length effect. The mean gap still rises, because harder levels enter at larger T.
- With illustrative per-level gaps of 10 / 20 / 40 / 60 pp (R2) and 30 pp (R1), the spurious slope is **3.55 pp per doubling** for R2 alone and **1.96 pp** pooled with R1. With every gap halved it is 0.98 pp [MEAS-A, B.3].
- B's H5 equivalence margin is ±2 pp per doubling. Composition alone can therefore block a TOST verdict of "no length effect", or manufacture a "supported" one.

**(b) Exposure per length.**
- The TEF specifies "synthetic items of 1K–32K" with no length distribution.
- If items are drawn uniformly over tokens, items at 1K outnumber items at 32K by 32:1. A1 M3 found the same kind of skew (≈ 20×) in PROP's option (i).

**(c) Selection.**
- H5 is computed "over G-REF-passing lengths".
- The lengths most likely to fail are the longest, so the slope is fitted on a range truncated by the outcome.

**Fix.**
- Fit H5 on a **fixed level set present at every T** (R1 D ∈ {3, 15}; R2 N ∈ {8, 32}), or in a mixed model with level fixed effects and level × log₂T terms.
- Draw TEF items with **equal counts per log₂-length bucket and per family**.
- Report H5 over all lengths with reference adequacy as a covariate, and the truncated version as a sensitivity.
- Use the zero-shot **LM-4 copy-after-gap slope** as an exposure-free co-primary (A1 M3).

### B-4 (MAJOR). G-REF as specified collapses df and deletes the most informative outcome

**Per-seed gating.**
- G-REF is applied "per seed and per length". B's falsification rule then needs "≥ 2 of the 3 seeds", followed by a seed-level t with df = 2.
- If the failing seed is excluded, df = 1 (t = 12.71) and nothing can be concluded at that length. If it is kept, the per-seed gate is cosmetic. B does not say which.

**Exclusion when k = 12 fails.**
- The closest small-scale study observed exactly this case. At 0.22B, the all-full baseline sat *below* SWA hybrids on in-distribution NIAH@16K (35.95 vs 49.58 and 67.81). It caught up only at 0.66B ([Qiao et al., 2606.15378v1](https://arxiv.org/html/2606.15378v1) Table 2 [read-section, verified in A1]).
- Under G-REF that outcome is "non-evaluable" for H1 and H5.
- Yet at this scale it is the most decision-relevant answer to "how much full attention does long context need?": less than all of it.

**Selection on a noisy gap.** Criterion 2 (S₁₂ − S₀ ≥ 15 pp), applied to the same items used for R, induces a winner's-curse bias when the true gap is near 15 pp.

**Fix.**
- Gate on the **seed-pooled** reference. Keep all seeds and report per-seed adequacy as a diagnostic.
- **Cross-fit**: split contexts into two pre-registered halves, gate and select on half A, and estimate on half B. This costs nothing, though it removes only the item-level part of the bias.
- When k = 12 fails adequacy but some hybrid passes, that length enters a **pre-registered secondary outcome, "reference inversion"**:
  - first check the 2× Stage-B saturation branch;
  - then report R_best, with the best configuration chosen on half A.

### B-5 (MAJOR). H6 cannot be tested in Tiers A–B: the measured hull margin is as tautological as the analytic one

**The identity** [INF, algebra; Appendix B.6].
- Let a cost add up layer by layer: c(k) = k·c_A(T) + (12 − k)·c_Λ(T) + c₀. This covers state bytes, FLOPs, and sequential prefill or decode latency.
- Then c̃(k) = (c(k) − c(0)) / (c(12) − c(0)) = **k/12 exactly, at every T**, and the hull margin is M(k) = G·(R(k) − k/12).
- B already labels the analytic margin tautological. The *measured* margin differs from it only through non-additive implementation effects (launch overhead, L2 residency), which are exactly the artefacts B worked to remove.

**What the README's H6 needs.**
- "Neither pure architecture reaches" requires a pure-architecture **frontier**, not a single point.
- B's only second pure-linear point, GDN config D (2× state), is in Tier C (S9d).

**Fix (A1 H6′, Tier B).**
- Add GDN-E (1 head, d_k 256, d_v 512; 4× state at equal parameters; 1,054,210 parameters, counted by meta-device instantiation in A1). Fall back to config D if the K = 256 kernel fails the G0 test.
- Run it at k ∈ {0, 1} × 2 seeds.
- The question becomes: is the +64 MiB of KV cache (k = 1 at 32K) worth more than +4.5 MiB of recurrent state (GDN-E, 12 layers × 384 KiB)?

### B-6 (MAJOR). H3 cannot be resolved above k = 3, and it compares families at difficulty levels that are not equated

**Missing grid points.**
- B drops k = 4 because it "sits in the region expected to be flat", and puts k = 6 in Tier C.
- That expectation comes from retrieval priors. For composition, the theory prior is a strict hierarchy in retrieval depth for hybrid blocks ([2610.25250](https://arxiv.org/abs/2610.25250) [abstract-only, A1]).
- Wang et al. report that the hybrid ratio affects some subtasks and not others (App. C, figure only [read-section, A1]).
- If the multi-hop family needs more than 3 layers (k80_hop > 3), B's grid can only report "somewhere between 3 and 12".

**Non-equated difficulty.**
- Each family has its own ladder (D and N, hops, ρ), so differences in R_f can reflect difficulty calibration (A1 F1).
- B's GLMM tests the family × k interaction on the logit scale, while R lives on the probability scale.
- For families with different baselines, any interaction short of a crossover can appear on one scale and vanish on the other [INF, standard measurement theory].

**Counting.**
- Counting (A1) is in the aggregation family score.
- Under the delta rule, repeated writes of one key converge to v rather than accumulating (A1 M5). That makes it a mechanism-specific task inside a family meant to be mechanism-general.

**Fix.**
- Add k = 6 × 2 seeds in Tier B, to give a descriptive k80 interval.
- Test H3 at difficulty-equated levels, i.e. levels where k = 12 accuracy is within ±10 pp across families.
- Pre-register the probability scale, with the logit-scale GLMM as a sensitivity.
- Make A2 (frequency) primary and A1 (counting) a diagnostic.

### B-7 (MODERATE). M3, QK-norm dilution, is real at initialisation but overstated, and the SSMax rule is decided by noise

**Reproduction.** My CPU simulation reproduces B's weights: 0.62 / 0.30 / 0.10 / 0.05 at 1K / 4K / 16K / 32K with unit gains [MEAS-A, B.1].

**Attention weight is not retrieval accuracy.**
- Same simulation, with an ideal linear readout among 4,096 candidate value codes and 3 same-format near-miss distractors (cos 0.5): accuracy is **1.000 at every length and every gain product tested** (1.0, 1.24, 1.56).
- The reason:
  - the weights sum to 1, so a common value mean cancels exactly;
  - the random part of the bulk shrinks as 1/√n_eff;
  - near-miss distractors get at least about 50× less weight than the needle at every T.
- What *does* shrink with T is the size of the needle's signal in the attention output (×12 from 1K to 32K at unit gains). That matters only relative to other content in the residual stream. It is an empirical question, not a ceiling [INF].

**Learnability.**
- B's own table needs the gain product to rise from 1.45 to 1.78 between 4K and 32K, i.e. +23%. The 4K value is what Stage A already rewards.
- Gains are excluded from weight decay, and Adam takes steps of roughly η. So 1,526 Stage-B steps plus the TEF are ample [INF].

**Precedent, which partly supports B.**
- Henry et al. diagnosed the same saturation problem for normalised QK. They scale logits by "a learnable scalar parameter initialized with g₀" = log₂(L² − L), where L is the 97.5th-percentile training length ([2010.04245](https://arxiv.org/abs/2010.04245), PDF §3 [read-section]).
- In PROP's parameterisation, unit gains give an effective scale of √d_h = 8. Henry's rule gives ≈ 24 at L = 4,096. So the initial temperature *is* low.
- The bound grows with √d_h, and d_h = 64 is small. Qwen3-Next uses `head_dim` 256 with QK-norm (`configuration_qwen3_next.py` L107 [code-inspected]).
- So this is a small-scale risk to monitor, not a structural cap.

**Rule inconsistency.**
- SSMax is adopted if it is "≥ plain QK-norm at 16K and 32K" on **one** seed. Under no true effect, that is close to a coin flip.
- B's own PE override, by contrast, requires a ≥ 5 pp margin.

**Fix.**
- Keep per-head gains (agree).
- Log maximum logit, attention entropy and needle mass (agree).
- Adopt SSMax, or a learnable per-head log-temperature as in QKNorm, **only on a ≥ 3 pp gain at 32K** on the S2 dev ladder. Otherwise run it as a robustness arm (B S9b).

### B-8 (MODERATE). Determinism is priced as a validity requirement; it is replicability hygiene

**B's measurements are correct and useful** [MEAS-B]:
- the FA2 backward pass is non-deterministic;
- k = 0 training is bitwise reproducible;
- cuDNN SDPA has no deterministic kernel.

**But the "asymmetric noise between arms" is not a bias** [INF]:
- Every (k, seed) is run once, and changing the seed already varies initialisation and data order.
- Training dynamics are chaotic: "all sources of nondeterminism have similar effects on measures of model diversity", and "even one-bit changes in initial parameters result in models converging to vastly different values" ([Summers & Dinneen, 2103.04514](https://arxiv.org/abs/2103.04514) [abstract-only]).
- So variation from atomics is already part of the seed variance that the inference uses. Determinism removes none of it from comparisons across seeds.

**Cost.**
- B estimates +7% across the core and +18% for k = 12, concentrated in the 32K stages, where the FA2 forward+backward kernel slows by 62–66% [MEAS-B].
- That is ≈ 4–5 GPU-h in Tier A, i.e. one reserve-seed run.

**Fix.**
- Use deterministic mode in the G0 audits and for one standing bitwise replicate: k = 12, seed 0, first 1,000 steps (B proposes this too).
- Run production in default mode.
- Reopen if G0 on an idle GPU measures ≤ 3% total overhead.

### B-9 (MODERATE). N50 is right-censored for the models that matter, and the ladder is bracketed on a weaker pilot

**Right-censoring.**
- R2 stops at N = 512. If models with k ≥ 1 stay above 50% at N = 512 at 16K–32K, their N50 is "> 512", and N50 cannot rank hybrids against all-full.
- B handles only left-censoring ("< 8").

**Drift from pilot to main runs.**
- S2 brackets the ladder on a pilot with a 0.3B-token Stage A; the main models get 0.8B.
- A1 F1's ceiling-drift argument against PROP's pilot applies equally to B's.

**Capacity wording.**
- N = 512 exceeds the key capacity of *one* GDN layer (H·d_k = 256).
- But k = 0 has 12 GDN layers. Its whole-model bound is 12 × 256 = 3,072 keys, if layers stored disjoint keys [INF].
- "k = 0 fails above capacity" is therefore a prediction about practical capacity, and has to be measured.

**Fix.**
- Add N = 1,024 at both 16K and 32K to the fixed primary level set (12·1,024 = 12,288 ≤ 0.75 × 16,384, exactly).
- Add N = 2,048 at 32K, used for N50 only.
- Pre-register one level above the range S2 brackets. This costs evaluation only.

### B-10 (MINOR). FlexAttention gives an equal toolchain, not equal maturity, and FLOP rooflines flatter chunked recurrences

**Maturity.**
- On Blackwell, fla pins the GDN forward-h kernel to `num_warps=[2]` "until Triton 3.8 is released" (`fla/ops/common/chunk_delta_h.py` L27–37). It also drops 8-warp configurations because of a race (`chunk_o.py` L28–35) [code-inspected, re-checked @`98193113`].
- B estimates the GDN chunk core at about 10% of peak, against 81% for FA2.
- FlexAttention runs within 5–12% of FA2 [MEAS-B]. That shows the *attention* side is mature in Triton too; it does not show that maturity is equal.

**Rooflines.**
- GDN's chunk mode has T/64 sequential inter-chunk steps, i.e. 512 at 32K.
- At batch 1 with 4 heads, each step exposes little parallelism. A FLOP roofline is therefore an optimistic bound on GDN prefill that may not be reachable by any implementation [INF].

**Fix.**
- Keep B's roofline-fraction table; that is the real fix.
- Add the sequential-depth term to the analytic prefill model.
- Label analytic latency as a lower bound for both mixers.
- Keep A1's projection to larger widths. Per layer, the ratio of attention-score FLOPs to dense FLOPs is ≈ T/(13d) at 32K: 4.9 at d = 512 but 0.6 at d = 4,096.

### B-11 (MINOR). Position-encoding robustness appears only in Tier C

- In Qiao et al.'s 0.22B hybrid, NoPE on the full layers moved NIAH@16K from 49.58 to 67.81 ([2606.15378v1](https://arxiv.org/html/2606.15378v1) Table 2 [read-section, A1]).
- B's H1 is therefore conditional on RoPE θ = 5e5, and the PE × k interaction is not measured before Tier C.
- **Fix:** NoPE at k ∈ {1, 12} × 1 seed in Tier A (8.5 h). Add a second seed in Tier B if |ΔR| > 0.1.

### B-12 (MINOR). How the main prior is read

**Mechanism averages.**
- B cites Wang et al.'s recall averaged over mechanisms (0.256 → 0.397).
- For GDN, the row is 0.356 / 0.341 / 0.352 / 0.425 / 0.436 for pure / 24:1 / 12:1 / 6:1 / 3:1. It is identical in v1 (Table 5) and v2 (Table 7), and it is **flat up to 12:1**.

**Version conflict on context length.**
- v1's Limitations say "a 2,048-token context window"; v2's say "a 4,096-token context window" ([v1](https://arxiv.org/html/2507.06457v1), [v2](https://arxiv.org/html/2507.06457v2) [read-section]).
- B and I each cited a different version.

**Consequence.** Wang et al. should set neither power nor ceilings, nor the expectation that the curve is "flat above k = 3".

### B-13 (MINOR). Multiplicity bookkeeping

- B's 5-test Holm family includes H4, but the placement runs are in Tier B, which is optional.
- B's Tier-A decision point settles H1, H1-min, H3 and H5. If Tier B never runs, the family changes after the data are in.
- **Fix:** Tier-A family {H1, H1-min, H3, H5}; a separate Tier-B family {H4 (2 contrasts), H6′}.

### B-14 (MINOR). Severity of B's F2

- B's F2 (cost-measurement artefacts) would be fatal only if measured costs were primary.
- Both designs make analytic costs primary, so it is a MAJOR measurement-hygiene issue. The fix is unchanged and adopted.

---

## 3. What Design B cannot establish

| README claim | Status under Design B | Why | Remedy in A2 |
|---|---|---|---|
| H1: a small share recovers most performance | A verdict is improbable even if H1 is true (P = 0.11 at R = 0.9) | B-1 | L₃ contrast with blocked pooled variance |
| H1 / H1-min read as "softmax memory vs recurrent memory" | Not identified; only package claims are licensed | B-2 | Gate by design; Attn+conv control at k = 12 |
| H3 when some family needs k > 3; H3 free of difficulty calibration | Not resolvable | B-6 | k = 6; difficulty-equated levels; pre-registered scale |
| H4 at its own 5 pp threshold | Detects only effects of ≳ 15 pp under B's rule (df = 2, Holm over 3 pairs; Appendix B.5) | Power | Pooled variance and 2 pre-specified contrasts: MDE ≈ 5–6 pp |
| H5 | Prone to artefacts | B-3 | Fixed level set; balanced TEF; LM-4 co-primary |
| H6 | Tautological in both analytic and measured form | B-5 | GDN-E as a second pure-linear point |
| Generality beyond GDN | No (GLA is Tier C) | Agreed scope | Word claims "for GDN" |
| Cost frontier beyond 50M parameters | No projection | B-10 | Projection formulas |
| Proportion vs count | Not separable (true of every design here) | L is fixed | Claims stated in k at L = 12 |
| Robustness to position encoding | Conditional on RoPE | B-11 | NoPE in Tier A |

---

## 4. Feasibility problems in Design B

1. **Pilot-to-main transfer.**
   - All five S2 rules (FEAS-1…5) are decided on a 0.3B-Stage-A base at seed 101.
   - A FEAS-1 failure at 32K would switch L_ext to 16K even if the full-budget reference would have passed.
   - **Fix (no cost, only reordering):** run core k = 12 seed 0 first and confirm FEAS-1 on it before switching.
2. **Single-seed thresholds.**
   - FEAS-3 (3 / 10 pp) and the SSMax rule (≥ 0) sit near seed noise (σ_d ≈ 3 pp in PROP's planning).
   - Misjudging FEAS-3 costs about +0.1B Stage-B tokens per run (k = 12: +0.87 h; ≈ +9 h over the core), or leaves the reference under-trained. The 2× branch would catch the second case.
   - Acceptable, but it should be stated.
3. **Determinism lands on the critical path.** Its overhead is concentrated in k = 12 Stage B and TEF at 32K: +62–66% on the attention forward+backward kernel [MEAS-B].
4. **The FlexAttention decode comparator may not exist.**
   - Decode with a device-side length inside one captured graph is unverified (B's own Q7).
   - Exact-shape graphs per context length L with cuDNN or flash remain, but the "equal-toolchain" comparator may be unavailable for decode.
5. **Stop rules can end the project without a long-context answer.**
   - This happens if G-REF fails at both 32K and 16K.
   - B's fallback (re-scope to ≤ 8K, or a larger model) needs the user's agreement *before* launch. Add it to PROP §17.
6. **Budget.**
   - Tier A ≈ 96 h (115 with contingency) and Tier B ≈ 128 h (154) are credible on B's measured numbers.
   - The ÷1.6 idle factor is gone. Any idle speed-up has to be measured in G0 if CARLA can pause.

---

## 5. Where I change my mind

| # | Change from A1 | Argument |
|---|---|---|
| C-1 | **A single primary scale: d = 512 (41.0M non-embedding / 49.4M total).** I drop A1's d = 384 discovery runs and the cross-scale Tier-3 "replication" | (a) B M1: 32K competence is not established even at 41M; a 28M model is less likely to pass G-REF. (b) All measured throughput and memory are at d = 512 [MEAS-B]. (c) My own new objection: a "replication" at another scale confounds scale with chance, so a disagreement could not be interpreted. (d) Power comes back through the blocked pooled-variance model (B-1), not through more runs. F12-S (d = 384) becomes a Tier-C scale check at k ∈ {0, 3, 12} |
| C-2 | **H1's primary cell is the long band {16K, 32K}, behind a repaired adequacy gate**, not A1's R-load ladder at T = 8K | B's F1 counterexample applies to A1 too. Load stays the difficulty axis *within* the band, as the fixed level set N ∈ {8, 32, 128, 512, 1,024} at both 16K and 32K |
| C-3 | **B's S2 is the first scientific gate.** A1's direct-supervision screen, and with it A1's "Level A" falsification route, becomes an optional Tier-C diagnostic | Direct supervision bounds capacity at this width. It says nothing about LM-regime competence at 32K, which is what can fail |
| C-4 | **No PE pilot; RoPE θ = 5e5 from step 0 in every arm** | B's argument that about 150 long steps cannot settle PE also holds for my 2-seed version. Llama 3 uses this base with 8K pretraining [read-section]. (I hold firm that the alternative PE is *measured* in Tier A; H-5) |
| C-5 | **B's cost-measurement protocol replaces A1 §2.9's measured layer.** A1's analytic layer and projections stay | S3 in §1; B-10 |
| C-6 | **Twin attention is no longer my primary full mixer.** The primary is MHA 8 × 64 plus a headwise gate, with the conv as a measured factor | No industrial hybrid puts a conv in attention (code-inspected, B-2), so a twin-only curve would not speak to practice. The twin's d_v = 128 ≠ d_qk = 64 needs non-standard kernels, which deepens the maturity problem B documents. The gate and conv asymmetries can be handled more cheaply. What I keep from A1 F2 is the requirement that they are handled **in Tier A** (H-1) |
| C-7 | **B's schedule and data:** Stage A 0.8B (stable LR); Stage B 0.2B at 32K (decay; budget set by FEAS-3); branching at S; TEF 50% synthetic / 50% Stage-B mix (A1 had 90/10) | B's branching principle. I keep A1's 15% PG19 4K windows in Stage A only as a cheap recommendation: document-masked FineWeb-Edu leaves most positions with ≤ 2K usable context (A1 M3) |
| C-8 | **Placement before GLA** in priority: B's k = 1 triad × 3 seeds at {1, 6, 11}, not A1's {0, 6, 11} | README RQ-B is an explicit question; generality across mechanisms is not. With a primary attention that has no conv, attention at layer 0 would see no preceding token mixing, which confounds depth with input features. B's layer 1 avoids that |
| C-9 | **H5's confirmatory test is the length slope** (the README's literal H5), on a fixed level set with balanced exposure. A1's distance-at-fixed-T × filler-type manipulation becomes the pre-registered mechanism follow-up | H5 is stated in terms of context length; A1 had replaced it with a mechanism test. The decomposition still decides *why* (A1 §4) |
| C-10 | **Engineering:** per-head QK-norm gains; relative G0(f) thresholds; an explicit fla scoring path with a 5% single-pass re-score; an archived Triton cache; B's runbook (memory fraction 0.5, 250-step atomic checkpoints, auto-resume, `contended_frac` logging) | S7 in §1 |
| C-11 | **Withdrawn from A1 F2:** the gate's "long-context extrapolation" benefit | It is irrelevant in-distribution: 79.50 vs 79.77 at 32K without YaRN (B-2) |
| C-12 | **Corrected A1 power numbers:** df = 6, not ≈ 11 | B-1. The conclusion is unchanged |

---

## 6. Where I hold firm

| # | Position | Argument |
|---|---|---|
| H-1 | **The mixer asymmetry must be handled in Tier A**: the gate by design (0 h) plus Attn+conv at k = 12 × 2 seeds (9.6 h) | B's S9c is PROP's E5b with the same flaw. The asymmetry acts on every R through the denominator, whether or not a hybrid wins (B-2 example: ΔR ≈ 0.1) |
| H-2 | **Inference by a linear contrast with blocked pooled variance.** The "bootstrap AND df = 2 t" conjunction and the 5 pp non-inferiority conjunct are not support conditions | The B-1 Monte Carlo. At R = 0.9, P(supported) rises from 0.11 to 0.51, and at R = 1.0 from 0.36 to 0.95, at **zero GPU cost**. It is the largest gain in scientific yield per GPU-hour available to the programme |
| H-3 | **H6 needs a second pure-architecture point** (GDN-E or config D) in Tier B | Otherwise H6 restates R(k) > k/12 for every additive cost (B-5) |
| H-4 | **k = 6 in Tier B** (2 seeds) | "Differently per task" requires being able to place k80 above 3 (B-6) |
| H-5 | **NoPE at k ∈ {1, 12} in Tier A** | B-11 |
| H-6 | **H5 needs a fixed level set, item-count-balanced TEF by length bucket, and an exposure-free co-primary** | B-3 |
| H-7 | **Frequency comparison is primary for aggregation; counting is a diagnostic** | Delta-rule overwrite (A1 M5) |
| H-8 | **H3 at difficulty-equated levels on a pre-registered scale; reserve seeds also triggered by bimodality** (seed span > 40 pp at a primary level), alongside B's σ_d > 5 pp trigger | B-6; A1 M2 (abrupt circuit formation; B's own E30 concedes it) |
| H-9 | **Framing.** The contribution is the controlled, per-family, length-resolved, seed-replicated curve at 49M, not the frontier idea | MAD reports a compute-optimal attention share of 25%, and 10% when state-optimality is considered ([2403.17844](https://arxiv.org/html/2403.17844) [read-section, A1]). Based already frames recall vs state size as a frontier. Wang et al.'s App. C is a direct H3 prior |

**Partial agreements (judgement calls, not hills to die on).**
- **Determinism (B-8):** agreed for audits and one standing replicate; not agreed for production.
- **M3 (B-7):** agreed on per-head gains and logging; not agreed on severity or on the adoption rule.
- **SSMax variant in S2:** keep it (≈ 2.4 h), but with a ≥ 3 pp rule.
- **TEF composition:** B's 50/50 accepted.

---

## 7. Merged design ("A2" = Design B + A1 repairs)

### 7.1 Architecture

- **Backbone** (as PROP): L = 12, d = 512, SwiGLU d_ff = 1536, V = 16,384 tied, pre-norm RMSNorm, no biases.
- **Full mixer:**
  - MHA 8 × 64 with **per-head** QK-RMSNorm gains;
  - RoPE θ = 5e5, with positions reset per document;
  - a **headwise sigmoid output gate** (W_g ∈ ℝ^{512×8});
  - no conv.
  - **1,053,696 parameters.**
- **Linear mixer:** GDN-A as PROP, 1,056,904 parameters.
- **Spread across k:** 38,496 parameters, i.e. 0.094% of non-embedding parameters (Appendix B.4).

### 7.2 Runs and budget

GPU-h are contended, from B's measured throughputs, with production in default (non-deterministic) mode. Per-run cost:
- k = 0: 3.55 h;
- k = 1: 3.65 h;
- k = 2: 3.75 h;
- k = 3: 3.95 h;
- k = 6: 4.18 h;
- k = 12: 4.81 h.

Appendix B.7 has the breakdown.

| Tier | Stage | Content | GPU-h |
|---|---|---|---|
| A | S0 | Infrastructure: G0 with deterministic audit and B's harness self-test; relative kernel thresholds; GDN-E kernel test at K = 256; exact parameter counts | 3 |
| A | S1 | Zoology MQAR (PROP RP-1, using re-run numbers only; A1 m1) | 3 |
| A | S2 | B's feasibility pilot (k ∈ {0, 12}, seed 101, Stage A 0.3B, Stage-B branches 0.1B / 0.3B, TEF checkpoints), plus the SSMax variant with a ≥ 3 pp adoption rule | 12 |
| A | S3 | LR sweep (B: 4 LRs × k ∈ {0, 3, 12} × 0.15B) | 5.6 |
| A | G5 | Freeze: protocol, items, analysis code, Triton cache | — |
| A | S4 | Core: k ∈ {0, 1, 3, 12} × seeds 0–2. **k = 12 seed 0 runs first** (FEAS-1 confirmed at full budget) | 47.9 |
| A | S4c | Controls: Attn+conv at k = 12 × 2 seeds (9.6); NoPE at k ∈ {1, 12} × 1 seed (8.5) | 18.1 |
| A | S5 | Branches at seed 0: TEF-4× for k ∈ {0, 1, 3, 12} (4.4); 2× Stage B for k = 12 (4.1) | 8.5 |
| A | — | Evaluation (≈ 21 models) and cost benchmarks in idle windows | 6.5 + 4 |
| | | **Tier A total** | **≈ 108.5 (≈ 130 with 20%)** |
| B | S6 | Curve: k ∈ {2, 6} × 2 seeds | 15.9 |
| B | S7 | Placement: k = 1 at {1, 11} × 3 seeds ({6} is shared with the core) | 21.9 |
| B | S9 | State size: GDN-E (or config D) at k ∈ {0, 1} × 2 seeds | 14.4 |
| B | cond. | Attn+conv at k ∈ {1, 3} × 2 seeds if the reference shift exceeds 3 pp (15.2); a second NoPE seed if \|ΔR\| > 0.1 (8.5) | ≤ 23.7 |
| B | — | Evaluation | 4 |
| | | **Tier B adds** | **≈ 56 (+ ≤ 24 conditional)**; cumulative ≈ 165 base |
| C | — | Optional, in priority order: (a) GLA at k ∈ {0, 1, 3} × 2 (≈ 22); (b) conv-free-GDN mechanism arm at k ∈ {0, 1} × 2 (≈ 14); (c) F12-S scale check at k ∈ {0, 3, 12} × 1 (≈ 8); (d) LR η*/√2 at k ∈ {3, 12} (≈ 9); (e) 3B tokens at k ∈ {3, 12} (≈ 22); (f) direct-supervision screen (≈ 4.5) | — |

**If Tiers A+B must approach B's ≈ 128 h base, cut in this order:**
1. NoPE to Tier B (−8.5 h);
2. state size to 1 seed (−7.2 h);
3. k = 6 to Tier C (−8.4 h);
4. placement to 2 seeds (−7.3 h).

After all four cuts the base is ≈ 133 h. The Attn+conv reference control and the decision rule are not cut.

### 7.3 Estimands and decision rules

**H1 and H1-min.**
- **Statistic:** L₃ (respectively L₁) on long-band, ladder-mean exact match (EM) over a **fixed** level set at 16K and 32K: R1 D ∈ {3, 15}; R2 N ∈ {8, 32, 128, 512, 1,024}.
- **Gate:** adequacy pooled over seeds, judged on context half A. Estimation uses half B.
- **Test:** blocked model, df = 6, one-sided 95%. Supported if the lower bound is > 0; falsified if the upper bound is < 0.
- **Guards** (all at seed 0 unless stated):
  1. *Budget:* if the k vs k = 12 gap shrinks by more than 3 pp between 1,000 and 4,000 TEF steps, the verdict is "budget-conditional".
  2. *Reference saturation:* the 2× Stage-B branch may raise S₁₂(32K) by at most 5 pp; otherwise report H1 against both references.
  3. *Conv:* the Attn+conv reference shift must be ≤ 3 pp; otherwise report H1 in both attention families.

**H3.**
- **Statistic:** R_ret(3) − R_hop(3) at difficulty-equated levels, on the probability scale.
- **Sensitivity:** the GLMM family × k interaction.

**H5.**
- **Statistic:** slope of Δ₀(T) over the fixed level set {R1 D ∈ {3, 15}; R2 N ∈ {8, 32}}, at all six T, with adequacy as a covariate.
- **Co-primary:** the zero-shot LM-4 slope.
- **Secondary:** distance at fixed T = 32K × filler type.

**Multiplicity and seeds.**
- Holm over {H1, H1-min, H3, H5}.
- A separate Tier-B family: H4 (middle − mean of ends; linear trend) and H6′.
- H2 is descriptive.
- Reserve seeds are triggered by a pooled σ_run > 4 pp, or by a seed span > 40 pp at a primary level.

### 7.4 Evaluation, exposure and cost

**Evaluation.** B §2.6, plus:
- the fixed level sets above;
- N = 2,048 at 32K, for N50 only;
- A2 (frequency) primary and A1 (counting) as a diagnostic;
- R-dist × filler at 32K (secondary);
- C0 and C2 controls;
- Tier 0 measured before the TEF;
- pre-registered context halves for cross-fitting.

**Exposure (TEF).**
- 1,000 steps × 64K tokens, 50% synthetic.
- **Equal item counts per log₂ bucket {1K, …, 32K} and per family.**
- A 50-step re-warm to 3e-4, then a constant LR.
- Checkpoints at 250 / 500 / 1,000 steps, plus 4,000 for the seed-0 anchors.
- LM-2 (4K validation loss) measured before and after.

**Cost.**
- B §2.7 measured protocol.
- Analytic primary costs, with the sequential-depth term and scale projections (B-10).
- H6′ framed on the GDN-E point.

---

## 8. Open items for round 3 and the PI

1. **Primary attention flavour:** headwise-gated MHA (A2) vs plain MHA (B, PROP). If plain is kept, the Tier-A control must become Attn+conv+gate at k = 12.
2. **Determinism in production** (B-8).
3. **State-size arm in Tier B (A2) or Tier C (B).**
4. **NoPE in Tier A with 1 seed (A2) or Tier C (B).**
5. **For the user:**
   - Can CARLA be paused for idle windows?
   - Is the user willing to agree up front to B's re-scope fallback (≤ 8K, or a larger model) if G-REF fails at both 32K and 16K?

---

## Appendix A. Verification log (Reviewer A, round 2, 2026-11-09)

| Claim used | Source | Access | Result |
|---|---|---|---|
| Qwen3-Next attention: sigmoid output gate from a doubled q-projection; per-head QK-norm; head_dim 256; partial RoPE 0.25; interval 4; no conv in attention | transformers 5.22.2 in `.venv`: `models/qwen3_next/modeling_qwen3_next.py` L240–303; `configuration_qwen3_next.py` L97–131 ([upstream](https://github.com/huggingface/transformers)) | code-inspected | Confirmed |
| Kimi-Linear MLA and OLMo-hybrid attention: no output gate, no conv (OLMo-hybrid has an optional NoPE mode) | `models/kimi_linear/modeling_kimi_linear.py` L190–313; `models/olmo_hybrid/modeling_olmo_hybrid.py` L161–242 | code-inspected | Confirmed |
| Gated attention: head-specific sigmoid gate after SDPA improves performance and stability and tolerates larger LRs | [2505.06708](https://arxiv.org/abs/2505.06708) | abstract-only | Confirmed |
| Gated attention, Table 1: headwise G1 adds 1.6M parameters; PPL 5.792 vs elementwise 5.761 vs baseline 6.026. Table 5: RULER 32K without YaRN 79.50 vs 79.77; with YaRN at 64K 37.51 vs 66.60 | [2505.06708 HTML](https://arxiv.org/html/2505.06708) | read-section | Confirmed |
| Primer: depthwise conv after each Q, K, V projection, plus squared ReLU | [2109.08668](https://arxiv.org/abs/2109.08668) | abstract-only | Confirmed |
| Summers & Dinneen: all nondeterminism sources have similar effects on model diversity; one-bit changes give vastly different models | [2103.04514](https://arxiv.org/abs/2103.04514) | abstract-only | Confirmed |
| QKNorm: learnable scalar g replaces 1/√d; g₀ = log₂(L² − L) with L the 97.5th-percentile length; motivated by softmax saturation on bounded cosines | [2010.04245](https://arxiv.org/abs/2010.04245) (PDF text via `pdftotext`) | read-section | Confirmed |
| Llama 3: RoPE base 500,000, "effective for context lengths up to 32,768"; 8K pretraining context | [2407.21783](https://arxiv.org/html/2407.21783) §2 and §3.2 | read-section | Confirmed |
| Wang et al.: v1 Limitations "2,048-token context window"; v2 "4,096-token"; GDN recall row 0.356 / 0.341 / 0.352 / 0.425 / 0.436 in both versions; 340M recall "insignificant" | [v1](https://arxiv.org/html/2507.06457v1), [v2](https://arxiv.org/html/2507.06457v2) | read-section | Confirmed. **The versions conflict** on training context |
| fla on Blackwell: `num_warps=[2]` until Triton 3.8; 8-warp race configurations dropped; `fused_recurrent` switch at `q_len <= 64` outside training | `fla/ops/common/chunk_delta_h.py` L25–37; `fla/ops/common/chunk_o.py` L27–35; `fla/layers/gated_deltanet.py` L226–232 @`98193113` ([repo](https://github.com/fla-org/flash-linear-attention)) | code-inspected | Confirmed (re-check of B) |
| Olsson smeared keys; Qiao Table 2 values; MAD findings; 2610.25250 hierarchy | As cited | As tagged in A1 | Verified in A1, not re-fetched |
| B's GPU numbers (artefacts, throughput, determinism, numerics) | B Appendix A | [MEAS-B] | **Not re-run** (shared GPU) |

## Appendix B. Arithmetic and simulations [MEAS-A unless marked INF]

The scripts are in my session scratchpad, outside the repository; the descriptions below are sufficient to re-create them:
- `/tmp/claude-1000/-home-sedrica-lynx-Downloads-outeropt/73b71b1c-85a4-4704-b75f-7c9b668d4052/scratchpad/sims/dilution_readout.py`
- `/tmp/claude-1000/-home-sedrica-lynx-Downloads-outeropt/73b71b1c-85a4-4704-b75f-7c9b668d4052/scratchpad/sims/h1_rule_power.py`
- `/tmp/claude-1000/-home-sedrica-lynx-Downloads-outeropt/73b71b1c-85a4-4704-b75f-7c9b668d4052/scratchpad/sims/h1_rule_power_df6.py`
- `/tmp/claude-1000/-home-sedrica-lynx-Downloads-outeropt/73b71b1c-85a4-4704-b75f-7c9b668d4052/scratchpad/sims/h5_ladder_artifact.py`

All were run with numpy from the project venv, on CPU.

### B.1 Dilution: attention weight vs ideal readout

**Setup.**
- d_h = 64. Keys are random unit vectors × 8 (RMS-normalised, unit gains). The needle key equals the query.
- 3 near-miss distractors have cos 0.5 with the query.
- Logits = G·q·k/8. Values are N(0, I₆₄); the mean cancels exactly because Σw = 1.
- Readout: argmax over 4,096 candidate codes (the needle code, the 3 distractor codes and 4,092 random codes).
- 400 trials per cell; numpy seed 20261109.

| G | n | Needle weight | Weight per near-miss | Ideal readout accuracy |
|---|---|---|---|---|
| 1.00 | 1K / 4K / 16K / 32K | 0.618 / 0.302 / 0.099 / 0.052 | 0.011 → 0.001 | 1.000 at every n |
| 1.24 | 1K / 4K / 16K / 32K | 0.886 / 0.689 / 0.364 / 0.224 | 0.006 → 0.002 | 1.000 at every n |
| 1.56 | 1K / 4K / 16K / 32K | 0.982 / 0.946 / 0.825 / 0.705 | ≈ 0.002 | 1.000 at every n |

**Caveat.** This readout ignores interference in the residual stream and downstream capacity, so it is an upper bound on robustness, just as B's weight table is a lower bound.

### B.2 Monte Carlo of the H1 decision rules

**Model.** S[k, seed] = μ_k + b_seed + e, with:
- b_seed ~ N(0, 3²);
- e ~ N(0, σ_d²/2);
- μ₀ = 40, μ₁₂ = 70 (G = 30 pp), μ₃ = 40 + 30R;
- 200,000 replicates.

**B / PROP rule.**
- Per-seed R_s and Δ_s; two-sided 95% t intervals with df = 2 (t = 4.303).
- Supported if lower R ≥ 0.8 **and** upper Δ ≤ 5 pp. Falsified if upper R < 0.8.
- The bootstrap leg is not simulated. With 3 seeds it is narrower than the t interval, so the t leg binds.

**A2 rule.**
- L₃ = S̄₃ − 0.2·S̄₀ − 0.8·S̄₁₂, with SE² = s²·1.68/3 and s² ~ σ_e²·χ²₆/6 (the blocked 4 × 3 residual).
- One-sided 95% interval, t₀.₉₅,₆ = 1.943.

**Full table.**

| σ_d | R | B: P(sup) | B: P(fals) | A2: P(sup) | A2: P(fals) |
|---|---|---|---|---|---|
| 3 | 0.5 / 0.6 / 0.7 | 0 / 0 / 0 | 0.828 / 0.552 / 0.212 | 0 / 0 / 0 | 0.999 / 0.952 / 0.511 |
| 3 | 0.8 | 0.009 | 0.026 | 0.050 | 0.050 |
| 3 | 0.9 / 1.0 / 1.1 | 0.107 / 0.363 / 0.656 | 0 / 0 / 0 | 0.512 / 0.952 / 1.000 | 0 / 0 / 0 |
| 5 | 0.5 / 0.6 / 0.7 | 0 / 0 / 0.001 | 0.482 / 0.275 / 0.110 | 0 / 0 / 0.004 | 0.910 / 0.641 / 0.262 |
| 5 | 0.8 | 0.013 | 0.027 | 0.050 | 0.051 |
| 5 | 0.9 / 1.0 / 1.1 | 0.061 / 0.170 / 0.319 | 0.003 / 0 / 0 | 0.261 / 0.640 / 0.909 | 0.004 / 0 / 0 |

**Non-inferiority arithmetic** [INF]:
- At σ_d = 3 pp: 4.303 × 3/√3 = 7.45 pp, so support needs Δ̂ ≤ −2.45 pp.
- At σ_d = 5 pp: 12.42 pp, so support needs Δ̂ ≤ −7.42 pp.

### B.3 Spurious H5 slope from ladder composition

**Rule.** Level N is used at T iff 12N ≤ 0.75T. The level sets are then {8, 32} at 1K, {8, 32, 128} at 2K and 4K, and all four from 8K.

**Assumption.** Endpoint gaps are constant in T: 10 / 20 / 40 / 60 pp for N = 8 / 32 / 128 / 512, and 30 pp for both R1 levels.

**Ladder-mean gaps by T** (1K → 32K):
- R2 alone: 15.0, 23.3, 23.3, 32.5, 32.5, 32.5.
- R1 + R2: 22.5, 26.0, 26.0, 31.7, 31.7, 31.7.

**OLS slope per doubling.**
- R2 alone: **3.55 pp**.
- R1 + R2: **1.96 pp**.
- R1 + R2 with all gaps halved: **0.98 pp**.

### B.4 Parameter counts for the A2 mixers [INF, arithmetic]

**Mixers.**
- **PROP's MHA:** 4·512² + 2·64 = 1,048,704.
- **With per-head gains:** 4·512² + 2·8·64 = 1,049,600.
- **Plus the headwise gate:** 512·8 = 4,096, giving **1,053,696**.
  - GDN-A has 1,056,904 (meta-device count, PROP and A1). The difference is −3,208, i.e. −0.30% per mixer.
- **Attn+conv:** +3·512·4 = 6,144, giving 1,059,840 (+0.28% vs GDN-A).

**Blocks and totals.**
- Attention block: 1,053,696 + 2,359,296 + 1,024 = 3,414,016. GDN block: 3,417,224.
- N(0) = 41,007,200 and N(12) = 40,968,704. The spread is 38,496 parameters, or 0.094%.

### B.5 Placement minimum detectable effect (MDE) [INF]

**Under B's rule.**
- Seed-level paired t with df = 2 and Holm over 3 pairs: two-sided α = 0.0167, so t = 7.65.
- SE = 3/√3 = 1.73 pp.
- MDE at 80% power ≈ (7.65 + 1.06) × 1.73 ≈ **15 pp**.

**Under A2.**
- Blocked model with the core plus 2 placement arms, 3 seeds: df = (6 − 1)(3 − 1) = 10.
- Two contrasts under Holm: t₀.₉₈₇₅,₁₀ ≈ 2.63.
- σ_e = 3/√2 = 2.12 pp.
- **Middle − mean of ends:** SE = 2.12·√(1/3 + 1/12 + 1/12) = 1.50 pp, so MDE ≈ (2.63 + 0.88) × 1.50 ≈ **5.3 pp**.
- **Linear trend:** SE = 2.12·√(2/3) = 1.73 pp, so MDE ≈ 6.1 pp.

### B.6 The additive-cost identity behind H6 [INF, algebra]

- Let c(k) = k·c_A(T) + (12 − k)·c_Λ(T) + c₀.
- Then c(k) − c(0) = k(c_A − c_Λ) and c(12) − c(0) = 12(c_A − c_Λ), so c̃(k) = k/12.
- The hull margin is M(k) = S(k) − [S(0) + c̃·G] = G·(R(k) − k/12).
- This holds for state bytes, FLOPs, and any latency that is a sum of per-layer times.

### B.7 Per-run cost (contended; from B's measured throughputs; no determinism) [INF]

| Component | k = 0 | k = 1 | k = 2 | k = 3 | k = 6 | k = 12 |
|---|---|---|---|---|---|---|
| Stage A, 0.8B at 88.9K tok/s | 2.50 h | 2.50 h | 2.50 h | 2.50 h | 2.50 h | 2.50 h |
| Stage B, 0.2B at 70.5 / 64 / 59 / 51 / 44 / 31.9K tok/s | 0.79 h | 0.87 h | 0.94 h | 1.09 h | 1.26 h | 1.74 h |
| TEF, 65.5M tokens at the same rates | 0.26 h | 0.28 h | 0.31 h | 0.36 h | 0.41 h | 0.57 h |
| **Total** | **3.55 h** | **3.65 h** | **3.75 h** | **3.95 h** | **4.18 h** | **4.81 h** |

**Branches.**
- TEF-4× adds 196.6M tokens: 0.77 / 0.85 / 1.07 / 1.71 h for k = 0 / 1 / 3 / 12.
- The 2× Stage-B branch for k = 12 is 0.4B at 31.9K tok/s plus the TEF: 3.48 + 0.57 = 4.05 h.

**Scope note.** This review writes only `reports/reviews/review_A_round2.md`. It does not edit `README.md`, `reports/decisions.md`, or any other agent's file.

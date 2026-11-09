# Debate record: Stage-3 review, rounds 1–2

*Moderated by the PI. Date: 2026-11-09. This is the only file written in this step.*

## How to read this record

**Inputs, read in full.**
- `README.md` and `reports/decisions.md`.
- `reports/initial_proposal.md` ("PROP", v0.1).
- `reports/reviews/review_A_round1.md` ("A1") and `review_A_round2.md` ("A2").
- `reports/reviews/review_B_round1.md` ("B1") and `review_B_round2.md` ("B2").

Issue labels from the reviews are kept, e.g. A1-F2, B1-M3, A2 B-4, B2 W3.

**Mandate.** For each contested issue this record gives:
- each side's argument;
- the resolution and why;
- for every disagreement that argument alone cannot settle, the pilot or ablation that settles it, with a pre-registered decision rule.

The README's research question, RQ-A–D and hypotheses H1–H6 are unchanged (§1).

**How issues were resolved, in order of precedence.**
1. **Hard constraints:** the README question and hypotheses; one RTX 5080 shared with CARLA; the user's budget.
2. **Reviewers' measurements on the project GPU** ([MEAS-B], [MEAS-B2]). All are contended timings.
3. **Algebra and simulation.** Where the reviewers' numbers conflicted, I ran an independent check ([PI-SIM]).
4. **Primary sources, with access tags.** I re-checked two load-bearing papers and four code claims myself ([PI-VER]).
5. **Judgement**, marked [PI-RULING]. Where a ruling goes against a reviewer's final (round-2) position, the dissent is recorded.

**Tags.**
- Access: [read-full], [read-section], [abstract-only], [code-inspected].
- Epistemic: [EST] established in the cited source; [HYP] hypothesis; [INF] inference or arithmetic.
- Provenance: [MEAS-B]/[MEAS-B2] are Reviewer B's GPU measurements; [MEAS-A] is Reviewer A's CPU simulations; [PI-SIM] is my CPU simulation (§6); [PI-VER] is a claim I re-checked this round.
- "Not re-verified by PI" marks claims taken from a review without re-checking.

**What I ran.** No GPU code.
- **CPU simulations** (numpy 2.5.4 in the project venv, run with `python -I` from outside the repo; scripts listed in §6):
  - operating characteristics of the competing H1 tests;
  - delta-rule capacity for three head geometries;
  - a hand re-derivation of the H5 ladder artefact;
  - parameter, state-byte, MDE and budget arithmetic.
- **Code inspection:**
  - transformers 5.22.2: Qwen3-Next, OLMo-hybrid and Kimi-Linear attention;
  - fla 0.6.0 at `98193113ecb7835143252a786e06933e371b67cc`.
- **Web:**
  - [2505.06708](https://arxiv.org/html/2505.06708), Tables 1 and 5;
  - [2606.15378v1](https://arxiv.org/html/2606.15378v1), §4.1, §4.3 and Table 2.

**Nemotron-H.** [2504.03624](https://arxiv.org/abs/2504.03624) mixes attention, Mamba-2 and MLP layers. It is not a controlled full-vs-linear ratio experiment and is not used as evidence anywhere in this record.

---

## 0. Outcome at a glance

"A2" and "B2" give each reviewer's final position. Outcome types:
- **Consensus:** both agreed by round 2.
- **Merge:** the resolution combines both final positions.
- **PI-RULING:** decided by judgement; the side it favours is named.
- **Experiment:** a pre-registered pilot or ablation decides (§3).

| # | Issue | A2 (final) | B2 (final) | Resolution | Type |
|---|---|---|---|---|---|
| I-01 | Full-attention mixer; conv/gate asymmetry | Gated MHA primary; Attn+conv control at k = 12 in Tier A | Primer-style twin primary; plain-MHA bridge | Headwise-gated MHA primary. Attn+conv (no SiLU on q/k) at k = 12 × 2 seeds, unconditional. Extends to k ∈ {1, 3} if it shifts the reference by > 3 pp | PI-RULING (for A); residual question → U-02 |
| I-02 | Primary scale | D512 only | D384 if feasible, else D512 | FEAS at D384 first, with a stricter margin; otherwise FEAS at D512 | Experiment → U-01 |
| I-03 | H1 estimator and rule | L₃ contrast; drop the bootstrap-AND-t conjunction | Same | L₃ and L₁, one-sided; 5 pp non-inferiority becomes secondary | Consensus |
| I-04 | Variance model and seeds | Seed-blocked pooled variance, df = 6 | Paired per-seed, 5 seeds | Paired primary; n₀ = 5 (D384) or 4 (D512), +2 seeds on an effect-blind trigger; pooled and bootstrap as sensitivity; success counts | PI-RULING (for B), from PI-SIM |
| I-05 | Primary H1 cell | Long band {16K, 32K} | Longest gate-passing length | Mean over the gate-passing lengths in {16K, 32K}; labelled 8K fallback | Merge |
| I-06 | Reference-adequacy gate | Seed-pooled; cross-fit; "reference inversion" outcome | Per seed; ≥ 4 of 5 seeds | Seed-pooled gate on independent dev items; verdict needs ≥ n−1 of n seeds adequate; reference inversion as a secondary outcome | Merge |
| I-07 | Difficulty ladders and anchors | Fixed levels plus N = 1,024 / 2,048 | Empirical anchors; extra top level; drop the interim gate | One frozen suite; N/N50₀ wording; interim gate dropped | Consensus (PI-VER) |
| I-08 | Feasibility pilot vs direct supervision | FEAS first; direct-supervision screen in Tier C | FEAS first; screen in Tier 1 | FEAS first; screen in Tier C unless FEAS shows floor or ceiling | PI-RULING (for A) |
| I-09 | Position encoding | RoPE 5e5; NoPE at k ∈ {1, 12} × 1 seed | RoPE 5e5; NoPE at k ∈ {3, 12} × 2 seeds | RoPE 5e5; NoPE at k ∈ {3, 12} × 1 seed, +1 seed on a trigger | Merge → U-06 |
| I-10 | QK-norm dilution; SSMax | Severity overstated; adopt only on ≥ 3 pp | Small-head artefact on the 32K cells | Per-head gains; SSMax only if ≥ 3 pp at 32K and ≤ 2 pp loss at 4K in FEAS | PI-RULING (A's rule) → U-03 |
| I-11 | Exposure fine-tune (TEF) mix | 50/50 | 90/10 plus forgetting gate | 90/10 default; FEAS forgetting gate, falling back to 50/50 | Experiment → U-08 |
| I-12 | Data, schedule, saturation branch | 2× branch at k = 12 | 2× branch at k ∈ {3, 12} | Schedule agreed; 2× Stage-B branch at k ∈ {0, 3, 12} | Merge (schedule consensus; PI-RULING on the branch set) |
| I-13 | H5 test | Length slope confirmatory | Distance ladder plus load × length | Length slope confirmatory (H5a); task dependence (H5b) secondary; distance × filler secondary | PI-RULING (for A; README wording) |
| I-14 | H3 test | Difficulty-equated levels, probability scale | Logit-scale GLMM primary | Per-seed R difference on the probability scale; equated-level check where defined; GLMM as sensitivity | PI-RULING |
| I-15 | H4 placement | k = 1 at {1, 6, 11} | Conditional k* | k* rule with explicit patterns | PI-RULING (for B); value → U-07 |
| I-16 | H6 and H6′ | Tier B, k ∈ {0, 1} | Per-head d_k wording; k* | Literal H6 descriptive; H6′ in Tier B at k ∈ {0, k*}, tested by a linear contrast | Merge (PI-VER) |
| I-17 | GLA and conv-free GDN arms | Tier C | Tier 2 | Tier C, with a promotion rule | PI-RULING (for A) → U-13 |
| I-18 | Cost measurement | B's protocol plus analytic layer | B's protocol | Analytic costs primary; B1 protocol; severity MAJOR | Consensus; Flex decode → U-10 |
| I-19 | Determinism | Audits plus one replicate | Same (conceded in B2 C5) | Production in default mode; deterministic G0 and standing replicate | Consensus |
| I-20 | Equivalence gates | 0.02-nat LM-stack run | Weight-transplant logit tests | Weight-transplant tests; 0.1B run kept as a smoke test | PI-RULING (for B) |
| I-21 | Multiplicity | {H1, H1-min, H3, H5}; separate Tier-B family | Same | Same; H5(b) joins the secondary family | Consensus |
| I-22 | H1 falsification | L₃ plus guards | Paired L₃ plus guards | Merged rule; minimum experiment ≈ 49 h (D384) or ≈ 90 h (D512) base | Merge |
| I-23 | Priors and novelty | Restate priors; credit MAD and Based | Accepts | Agreed | Consensus |
| I-24 | Shared-GPU operations and budget | Adopts B's runbook | Runbook plus 20% contingency | Agreed | Consensus |
| I-25 | Count vs proportion | Claims in k | Same | Agreed | Consensus |
| I-26 | LR protocol | B's sweep | Same | Agreed | Consensus |
| I-27 | Elicited vs zero-shot capability | Open | Open | Reporting rule only | Open (U-16) |
| I-28 | Compute vs memory; document masking | Open | — | Stated as limitations | Out of scope |

**Counts.** 28 issues: 9 consensus, 6 merges, 9 PI rulings, 2 decided by a pre-registered experiment, 2 left open or out of scope. A bundle of minor consensus points is recorded as I-29 (§2F).

**Experiment-decided parameters.** Several resolutions also fix a rule whose *value* comes from data: scale, k*, seed count, SSMax adoption, TEF mix and Stage-B budget. These are listed in §3.

---

## 1. Fixed constraints: the research question is not changed

**Kept verbatim from the README:** the research question, RQ-A–D and H1–H6. Every resolution below is an *operationalisation*:

- **H1 / H1-min.** "A small proportion recovers most of the performance" stays R ≥ 0.8, as in PROP. It is now tested as L_k = S_k − 0.2·S₀ − 0.8·S₁₂ ≥ 0, which is algebraically identical whenever G = S₁₂ − S₀ > 0.
- **H2** is reported descriptively. A test of g₁ > g_late is nearly implied by any concave curve (A1-M1); both reviewers agree.
- **H3.** "Required proportion" is measured by normalised recovery R_f.
- **H4.** Placement is tested at a fixed k. Which k (k*) is chosen from the composition data only.
- **H5** stays a test of *context length*, as the README words it. Needle–query distance is a mechanism follow-up.
- **H6** is reported as stated, plus a substantive version, H6′.
  - For any cost that adds up layer by layer, the hull margin is M(k) = G·(R(k) − k/12) [INF, algebra; A2 B-5, §6.4].
  - So literal H6 adds nothing beyond R(k) > k/12. H6′ adds a second pure-linear point. It adds a test; it replaces nothing.
- **"Proportion."** At fixed L = 12, the count k and the proportion k/12 are collinear. Results are reported as "k of 12" with r = k/12 alongside, never as scale-free proportions (A1-M11; accepted in B2).
- **"Full attention"** remains softmax attention over all previous tokens. The primary mixer (I-01) is standard MHA with a per-head output gate, which is still full softmax attention.
- **"Data fixed"** means identical data, data order and synthetic fine-tune for every configuration (PROP §17.5). This needs the user's confirmation (§5).

**Hardware.**
- One RTX 5080 16 GB (sm_120), shared with CARLA.
- Foreign processes used up to ≈ 7.9 GB [MEAS-B2].
- Clocks cannot be locked [MEAS-B].
- Contended throughput of the same configuration varied by up to 28% between rounds [MEAS-B2].

**Budget.** The user says 50+ GPU-h is acceptable. Every tier in §4 ends at a stop/continue decision point.

---

## 2. Record by issue

### A. What is manipulated

#### I-01. The full-attention mixer and the conv/gate asymmetry
**Type:** PI-RULING. Gate: consensus. Conv control in Tier A: consensus. Primary family: ruled for A. Residual empirical question: U-02.

**PROP §5.**
- Plain MHA 8 × 64 with QK-RMSNorm, gains shared across heads; no gate, no conv.
- E5b ("Attn+conv at k = 12") is optional, and runs only if some hybrid beats k = 12.

**A1-F2 (FATAL).**
- Each full→linear swap changes four things besides the memory operator:
  - a conv(4) + SiLU on q/k/v;
  - a swish-gated per-head RMSNorm output;
  - L2-normalised q/k with learned decay;
  - an implicit position source.
- Code: fla `gated_deltanet.py` L168–197 [code-inspected].
- Mechanism: Olsson et al.'s "smeared keys" let induction form in one-layer models, and form earlier in deeper ones ([2209.11895](https://arxiv.org/abs/2209.11895) [read-section]; verified verbatim by A1 and by B2).
- Fix: a "twin" full mixer (4 heads × d_k 64 / d_v 128, same conv and gate as GDN), with plain MHA as a bridge arm.

**B1.**
- Kept plain MHA.
- Conv control in Tier C, conditional on a hybrid beating k = 12 (S9c). Gate control also Tier C (E21).

**A2 (C-6, H-1).**
- Withdrew the twin as primary:
  - no industrial hybrid puts a conv in attention (Qwen3-Next, Kimi-Linear, OLMo-hybrid [code-inspected]);
  - the twin's d_v ≠ d_qk needs non-standard kernels.
- Proposed instead:
  - **Primary:** MHA 8 × 64 with per-head QK gains and a headwise sigmoid output gate, 1,053,696 parameters.
  - **Tier A control:** Attn+conv at k = 12 × 2 seeds, run regardless of results.
  - **Conditional:** Attn+conv at k ∈ {1, 3} if the reference shifts by > 3 pp.
  - **Wording:** all claims are package claims.
- Withdrew A1's "gate helps long-context extrapolation" argument. In-distribution RULER at 32K is 79.50 vs 79.77.

**B2 (C1, W6, W7).** Moved the other way.
- **Primary:** a Primer-style twin. Conv on q/k/v, no SiLU on q/k, GDN-identical gated per-head RMSNorm output, per-head gains, 4 heads × 64/128.
- **Tier 1 bridge:** plain MHA at k ∈ {3, 12} × 2 seeds.
- **Evidence:**
  - SiLU before QK-norm confines q/k to a cone. Needle weight at 32K falls from 0.71 to 0.39 at gain² = 1.56 [INF, simulation].
  - Native d_v ≠ d_qk decode at 32K under default dispatch takes 1,700 µs vs 66 µs with cuDNN pinned [MEAS-B2].
  - 4 lookups per position vs 8 might matter for multi-hop [HYP].

**Agreed by the end of round 2:**
1. The asymmetry acts on every R through the denominator S₁₂, so its control must be unconditional and in Tier A.
   - A2's example: S₀ = 40, S₁₂ = 70, S₃ = 64 gives R = 0.80. If attention with a conv reaches 76 at k = 12 and only 65 at k = 3, then R = 25/36 = 0.69, and no hybrid ever beats all-full.
2. An output gate is industrial practice and operator-agnostic.
3. No industrial attention layer has a conv.
4. Softmax q/k should not carry a SiLU.
5. QK gains should be per head and unshared.
6. Both attention families must be measured at least at k = 12.

**PI verification.**
- **Code** [PI-VER, code-inspected; transformers 5.22.2, [upstream](https://github.com/huggingface/transformers)]:
  - `Qwen3NextAttention` (`modeling_qwen3_next.py` L229) applies `attn_output * torch.sigmoid(gate)` at L300. Conv appears only in the GatedDeltaNet path (L318–349, L572). Default `head_dim` is 256 (`configuration_qwen3_next.py` L107).
  - `OlmoHybridAttention` (L161) has only `q_norm`/`k_norm`.
  - `KimiLinearAttention` (L190) is "MLA … with NoPE", with no gate and no conv.
- **Gated attention**, Table 1 (15A2B MoE) [PI-VER, read-section, [2505.06708](https://arxiv.org/html/2505.06708)]:

  | Variant | Added parameters | PPL |
  |---|---|---|
  | Baseline | 0 | 6.026 |
  | Headwise G1 | 1.6M | 5.792 |
  | Elementwise G1 | 201M | 5.761 |

**Resolution.**
- **Primary full mixer F_g.**
  - MHA: 8 × 64 at D512, 6 × 64 at D384.
  - Per-head, unshared QK-RMSNorm gains.
  - RoPE θ = 5e5, with positions reset per document.
  - Headwise sigmoid output gate: o_h ← σ(x·W_g)_h · o_h, with W_g ∈ ℝ^{d×H}; no bias, no conv.
  - Parameters: 1,053,696 at D512 (−0.30% vs GDN-A's 1,056,904) and 592,896 at D384 (−0.41% vs 595,334).
  - Non-embedding spread across k: 38,496 (0.094%) at D512 and 29,256 (0.137%) at D384 (§6.4).
- **Conv control F_gc** = F_g plus a causal depthwise ShortConv(4) on q, k and v.
  - The conv sits after the projections, before QK-norm and RoPE.
  - SiLU on v only, mirroring GDN's v path; no activation on q or k.
  - +3·d·4 parameters: 1,059,840 at D512, 597,504 at D384.
  - It keeps F_g's head count, so the control changes exactly one thing.
- **Runs.** F_gc at k = 12 × 2 seeds in Tier A, unconditionally.
- **Triggers.** These read only k = 12 models.
  - If the seed-paired mean of S₁₂(F_gc) − S₁₂(F_g) over seeds 0–1, at the H1 primary cell (I-05), exceeds **3 pp**: run F_gc at k ∈ {1, 3} × 2 seeds (Tier B, conditional) and report H1 and H1-min in both families.
  - If it exceeds **6 pp**: promote F_gc to a co-primary family (≥ 3 seeds at k ∈ {1, 3, 12}) before any Tier-B placement work.
- **Wording.**
  - Confirmatory claims are package claims: "k gated-MHA layers of 12, the rest GDN-A as built".
  - Operator-level wording ("softmax memory vs delta-rule memory") is allowed only if F_gc shifts the reference by ≤ 3 pp, or both families give the same verdict.
- **Twin geometry** (4 heads, d_v = 128) is not run in Tiers A–B. An optional Tier-C arm (k = 12 × 2 seeds) answers the head-geometry question (U-12).

**Why.**
- The README asks about full-attention layers as they are built in hybrids, so the package reading is primary [constraint].
- The headwise gate removes the gate asymmetry at +0.39% parameters per mixer, and gating is industrial practice.
- The conv is the asymmetry with the strongest mechanistic evidence, and F_gc isolates exactly it.
- The 8 × 64 geometry:
  - keeps FA2 `varlen_attn` for training and flash/cuDNN for inference;
  - avoids the measured d_v ≠ d_qk dispatch trap;
  - keeps KV bytes standard: 2 KiB per token per layer at D512.
- B2's SiLU objection is met, since q/k carry no SiLU.

**Dissent (B2).** B2 wants the operator-isolated curve to be the main curve. Accepted residual risk: if the conv trigger fires, the operator-isolated curve has only 2 seeds per point, unless the > 6 pp promotion applies.

#### I-02. Primary model scale
**Type:** UNRESOLVED between reviewers. Decided by the FEAS pilot (U-01), under a rule set by the PI.

**Positions.**
- **PROP and B1:** D512 (L = 12, d = 512; 41.0M non-embedding / 49.4M total parameters).
- **A1:** discovery at D384 (21.3M / 27.6M), plus a D512 "replication" in Tier 3.
- **A2 (C-1):** a single primary scale, D512. Reasons:
  - (a) 32K competence is not established even at 41M;
  - (b) all measurements were made at D512;
  - (c) a cross-scale "replication" confounds scale with chance;
  - (d) power should come from the inference method, not from more runs.
- **B2 (W2, W8, §5).**
  - Use D384 if FEAS passes there, otherwise D512; a later D512 run is a "scale extension", not a replication.
  - Measured D384/D512 throughput ratio: 1.30–1.39× [MEAS-B2].

**Evidence.** Qiao et al. [PI-VER, read-section, [2606.15378v1](https://arxiv.org/html/2606.15378v1)]:
- All models were pretrained "with a 16K context length on a 1:1 mixture of long and short datasets" (§4.1).
- They "exclude the S1 checkpoint at D=100N because its training budget is too small for stable long-context behavior" (§4.3). S1 has 15M non-embedding parameters (Table 1).

**Analysis** [INF].
- A2's remaining substantive argument, (a), is exactly what FEAS measures.
- (b) no longer holds: B2 measured D384 throughput and memory.
- (c) concerns replication, not the choice of a single primary scale. Both reviewers now agree there is no cross-scale replication.
- (d) is answered by I-04: valid paired inference needs seeds, and seeds are about 2× cheaper at D384.
  - Anchor set {0, 1, 3, 12}, one seed: 7.67 h at D384 vs 15.96 h at D512 (§6.4).
  - Tier A base: ≈ 80 h at D384 vs ≈ 138 h at D512.

**Resolution (pre-registered rule).**
1. **Run FEAS-384 first** (≈ 6 h):
   - D384, k ∈ {0, 12} × seeds 101–102, plus a k = 12 SSMax variant on one seed;
   - Stage A 0.3B tokens;
   - Stage-B branches at 32K of 30M and 90M tokens;
   - TEF checkpoints at 250 and 1,000 steps;
   - dev ladders at 4K, 8K, 16K and 32K, plus C0.
2. **Adopt D384 if and only if**, for both seeds at 32K on the dev ladder mean:
   - S₁₂ ≥ 0.85·S₁₂(C0), and
   - S₁₂ − S₀ ≥ 25 pp.

   These margins are stricter than FEAS-1/FEAS-2's 0.80 and 20 pp [INF: a buffer against per-seed G-REF failures in the smaller model].
3. **Otherwise run FEAS-512** (B1's S2, ≈ 12 h) with the standard FEAS-1/FEAS-2 thresholds, then B1's fallback ladder: L_ext = 16K; then stop and re-scope with the user.
4. **At the chosen scale,** the core k = 12 and k = 0 seed-0 runs go first, and FEAS-1/FEAS-2 are re-confirmed on those full-budget models before any other core run launches (A2 §4.1).
5. **No cross-scale replication** in Tiers A–B. The other scale is an optional Tier-C "scale extension", with directional predictions frozen before it runs.

**Reopen if:** never during the core. Switching scale after the G5 freeze voids the freeze.

#### I-25. Count vs proportion
**Type:** Consensus (A1-M11; accepted in B2 §3).
- Confirmatory claims are stated as "k of 12".
- The F24 family (L = 24, which separates count from proportion) is out of scope, because it also changes width and state size.

### B. Estimands and inference

#### I-03. H1 estimator and decision rule
**Type:** Consensus (A2 B-1; B2 C2).

**PROP's rule.**
- Ratio R, with the hierarchical bootstrap AND a seed-level t (df = 2) both required.
- Plus a 5 pp non-inferiority margin.

**A's objection (A1-M1; A2 B-1).**
- Monte Carlo [MEAS-A, not re-run by PI]: under PROP's rule, P(supported) is 0.11 at R = 0.9 and 0.36 at R = 1.0.
- The non-inferiority conjunct alone requires Δ̂ = Ŝ₁₂ − Ŝ₃ ≤ −2.45 pp at σ_d = 3 pp. In practice k = 3 would have to beat all-full.

**B's response.** B2 conceded: its own 3-seed paired power is only 0.54.

**Resolution.**
- **Statistics:** L₃ and L₁ on the primary cell (I-05), each tested one-sided at 95% within Holm (I-21).
- **Supported** if and only if the lower bound is > 0. G ≥ 15 pp is guaranteed by the reference gate.
- **Falsified** if and only if the upper bound is < 0 and the I-22 guards pass. Anything else is "inconclusive".
- **Secondary estimands:**
  - the 5 pp non-inferiority margin (S₁₂ − S₃);
  - R with a Fieller CI;
  - H1-robust: R ≥ 0.8 at every informative ladder level;
  - R_best: recovery relative to the best model in the family.
- **Gold-answer log-probability.** L_k on gold log-prob is a continuous secondary. A verdict is labelled "metric-robust" only if this agrees in sign.

#### I-04. Variance model, seed count and bimodality
**Type:** PI-RULING, informed by PI-SIM. Rules against A2's pooled primary; adopts B2's 5 seeds on the D384 path.

**Positions.**
- **A1-M1:** residual variance pooled over a whole stage (df ≈ 11), with a Brown–Forsythe check and Welch fallback.
- **A2 B-1:** a seed-blocked model pooled over the 4 × 3 core table (df = 6).
  - Monte Carlo: P(supported) = 0.51 at R = 0.9 and 0.95 at R = 1.0 [MEAS-A].
  - A2 concedes that pooling assumes equal variance across k ("against myself").
- **B2 W3:** pooling is anti-conservative under the unequal variance that A itself predicts.
  - With σ₃ = 8 pp, type-I error is 14.5% in A1's 22-run layout, and the Brown–Forsythe guard fires in only 1.8% of trials [INF; not re-run by PI].
  - With bimodal seeds, no 3-seed test is valid (23–29%).
  - Fix: paired per-seed L_k with 5 seeds; a 12 pp range trigger; per-configuration success counts.

**PI-SIM** (§6.2; 60,000 replicates; seed effect τ = 2 pp; G = 30 pp).
- **Equal variances:** both tests are valid. Pooled is more powerful. At residual σ = 3 pp and R = 1.0:

  | Seeds n | Pooled power | Paired power |
  |---|---|---|
  | 3 | 0.763 | 0.545 |
  | 4 | 0.885 | 0.755 |
  | 5 | 0.946 | 0.880 |

- **σ₃ = 8 pp at k = 3:** with the pooled model restricted to the core table, type-I error is 0.112–0.114 at σ = 3 and 0.133–0.136 at σ = 2. The paired test stays at 0.049–0.051. This reproduces B2's claim independently.
- **Bimodal seeds:** every test lies between 0.13 and 0.30.
- **Adaptive rule** (paired test; start with n₀ seeds; add 2 seeds to all anchors if the blinded pooled residual SD > 2.5 pp or any anchor's seed range > 12 pp):

  | Start → max | Type-I (equal / unequal variance) | Power at R = 1.0, σ = 3 | Power at R = 1.0, σ = 2 |
  |---|---|---|---|
  | 4 → 6 | 0.049–0.058 | 0.923 (E[n] = 5.5) | 0.979 (E[n] = 4.3) |
  | 5 → 7 | 0.050–0.054 | 0.967 (E[n] = 6.6) | 0.997 (E[n] = 5.3) |

- **The reviewers' power figures do not actually conflict.** A's "σ_d = 3 pp" is the SD of a *paired difference*, i.e. a residual SD of 2.12 pp. B's "σ = 3 pp" is the residual SD itself. PI-SIM reproduces both sets of numbers under their own conventions.

**Resolution.**
- **Primary test:** paired per-seed L_k t-test, df = n − 1.
- **Seeds:**
  - n₀ = 5 at D384 and 4 at D512, for k ∈ {0, 1, 3, 12};
  - +2 seeds for all four anchors (seeds 5–6 at D384, 4–5 at D512) if the effect-blind trigger fires;
  - the trigger is evaluated once, after the n₀ seeds of all four anchors finish.
- **Sensitivity analyses:** the seed-blocked pooled model (A2) and the hierarchical bootstrap. If either disagrees in sign with the primary verdict, the verdict is downgraded to "suggestive".
- **Named secondary estimand (B2):** per-configuration success counts with Clopper–Pearson CIs. A seed counts as a success if its primary-cell score is above the midpoint of the two endpoint means.
- Every seed is plotted. No seed is ever dropped.

**Why.**
- Confirmatory claims must stay valid under the unequal variance that both reviewers expect near capability transitions (A1-M2; B1-E30).
- Paired inference is valid there; pooled inference is not.
- The price of validity is seeds, and the adaptive rule buys them only when the observed variance requires it.

**Dissent (A2).** A2 holds that pooling is the largest free power gain available. It is kept as a pre-registered sensitivity analysis.

#### I-05. Primary cell for H1
**Type:** Merge.

**Positions.**
- PROP: pooled over all lengths 1K–32K.
- A1: the R-load ladder at 8K.
- B1: the long band {16K, 32K}. A2 adopted this.
- B2: the longest length that passes the reference gate, in the order 32K, 16K, 8K; R-load run as a load × length factorial.

**Evidence.**
- B1's counterexample: the pooled R = 1.05 ("supported") while the reference keeps less than half of its no-filler accuracy at 32K.
- B2-W1 shows the same failure for A1's 8K design.

**Resolution.**
- The primary cell is the ladder mean over the fixed R1/R2 level set (I-07), averaged over those lengths in {16K, 32K} that pass the reference gate (I-06).
- If only one length passes, use that length.
- If neither passes, use 8K, and label the verdict "H1 at 8K, not a long-context result".
- Never pool across length bands.
- R-load is evaluated at 8K, 16K and 32K (a load × length factorial). 8K is always reported.

**Why.** Averaging two passing long lengths roughly halves item noise compared with one length, keeps the long-context reading, and keeps B2's interpretable fallback order.

#### I-06. Reference-adequacy gate (G-REF)
**Type:** Merge of B1's criteria, A2's repairs and B2's dev-item rule.

**Positions.**
- **B1-F1:** applied per seed and per length. Pass requires S₁₂(T) ≥ 0.8·S₁₂(C0) and S₁₂(T) − S₀(T) ≥ 15 pp; failing lengths are excluded.
- **A2 B-4:**
  - Per-seed exclusion can collapse df to 1.
  - It also deletes "reference inversion", the very outcome the closest small-scale study reported. In Qiao et al. at S4, all-full scored 35.95 on NIAH@16K vs 49.58 and 67.81 for the SWA hybrids [PI-VER, Table 2].
  - Selecting on a noisy gap gives a winner's curse.
  - Fix: gate on seed-pooled means, cross-fit on context halves, add a reference-inversion outcome.
- **B2 W1/W4:** per seed and per length; ≥ 4 of 5 seeds for falsification; judge informativeness on dev items against frozen thresholds.

**Resolution.**
- **Criteria:** B1's two criteria.
- **Where evaluated:** on seed-pooled means of an independent dev item set (generator seed 77; ≥ 200 contexts per gate cell at 16K, 32K and C0). Item noise in the gate is therefore independent of the estimate, and all seeds are kept.
- **Verdict condition:** at the primary length, ≥ n−1 of n seeds must individually pass criterion (a). Otherwise that length is "reference-unstable", and the I-05 fallback applies.
- **Reference inversion** (A2), a pre-registered secondary outcome:
  - Triggered when k = 12 fails (a) while some hybrid passes its own (a) and exceeds S₁₂ by ≥ 5 pp.
  - First check the 2× Stage-B branch.
  - Then report R_best, with the best configuration chosen on dev items.

**Why dev items rather than A2's half-split.** Dev items remove the item-level selection bias without halving the precision of the estimate. Seed-level noise is still shared between gate and estimate, as A2 noted for its own fix.

#### I-21. Multiplicity
**Type:** Consensus (A2 B-13; B2 §5.1).
- **Confirmatory family:** {H1 (k = 3), H1-min (k = 1), H3, H5(a)}. Holm at α_FW = 0.05, one-sided in the pre-registered directions. Holm applies to whichever directional p-value decides each hypothesis.
- **Secondary family:** {H4 middle − mean(ends), H4 linear trend, H6′, H5(b)}, Holm.
- **Descriptive:** H2 (isotonic and saturating fits; k80 with CI) and the literal H6.
- **Exploratory:** everything else, BH-FDR at q = 0.10.
- **Falsification** is decided on unselected ladder means. Per-level results use Holm across levels (B2 W4: falsifying at "any of 4 levels" would give up to 18.5% false falsifications).

#### I-22. H1 falsification and the minimum experiment
**Type:** Merge of B1 §3, A2 §7.3 and B2 §5.3.

**Decision rule.** H1 is falsified, for the chosen scale, GDN-A, gated MHA and this recipe, only if **all** of the following hold:
1. **Gates pass:** G0, G-RP and FEAS-1/2. The reference gate passes at the primary length (seed-pooled dev items), and ≥ n−1 of n seeds are individually adequate.
2. **Test:** the paired per-seed one-sided 95% upper bound of L₃ is < 0 on the unselected primary-cell ladder mean. The pooled-variance and bootstrap analyses agree in sign.
3. **Dose guard:** at seed 0, the 4,000-step TEF branch keeps L₃ < 0, and k = 3 has plateaued (< 3 pp change from 2,000 to 4,000 steps).
4. **Saturation guard:** the 2× Stage-B branch changes S₁₂ at the primary length by ≤ 5 pp. Otherwise the verdict reads "conditional on the long-stage budget", and L₃ is recomputed on the 2× branches at k ∈ {0, 3, 12}.

**Not required for falsification:** the conv and NoPE controls. They decide whether the verdict extends from the package reading to the operator reading; a package-level falsification stands without them [INF].

**Minimum experiment.**
- **Contents:**
  - S0 (3 h) and a reduced S1 (1.5 h);
  - FEAS (6 h on the D384 path; 18 h on the D512 path, including the failed D384 pilot);
  - a reduced LR sweep: k ∈ {0, 12} × 3 LRs;
  - core k ∈ {0, 3, 12} × n₀ seeds;
  - seed-0 branches: TEF to 4,000 steps, and 2× Stage B, both at k ∈ {0, 3, 12};
  - retrieval-only evaluation.
- **Cost:**
  - D384 path: **≈ 49 GPU-h** base, ≈ 59 h with 20% contingency.
  - D512 path: **≈ 90 GPU-h** base, ≈ 107 h with contingency (§6.4).
- **For comparison:**
  - PROP: 66 h, at D512 with 3 seeds and no adequacy or learnability controls.
  - A1 Level B: 31 h, at D384 with 3 seeds and no feasibility pilot.
  - B1: 63 h (75 h with contingency), at D512 with 3 seeds.

**Withdrawn:** A1's "Level A" (falsification in principle under direct supervision). A2 and B2 agree it is a screen: it can fail for optimisation reasons, and pretraining can supply features that direct supervision cannot.

### C. Evaluation design and the secondary hypotheses

#### I-07. Difficulty ladders and capacity anchors
**Type:** Consensus, with a wording change verified by the PI.

**Positions.**
- **PROP:** one difficulty level per task, tuned so the pilot all-full model scores 40–85%.
- **A1-F1 and B1-M6:** replace it with ladders.
- **A1:** called the ladder levels "theory-anchored", using H·d_k as the capacity unit.
- **A2 B-9:**
  - add N = 1,024 at 16K and 32K, and N = 2,048 at 32K for N50 only;
  - the ladder must not be bracketed on a weaker pilot.
- **A2 B-3:** B1's rule "use level N at length T only if 12N ≤ 0.75T" by itself creates a spurious H5 slope.
- **B2 W5(a):**
  - per-head d_k, not H·d_k, governs interference;
  - degradation is gradual, so levels are not phase boundaries;
  - anchor the levels empirically, as N/N50₀.
- **B2 W9:** drop the interim gate G-I; freeze one extra top level from the start.

**PI verification.**
- **Capacity re-run** [PI-SIM, §6.3]. Mean cosine to the true value at N = 32 / 64 / 128 / 256 / 512 facts:

  | Head geometry | N = 32 | 64 | 128 | 256 | 512 |
  |---|---|---|---|---|---|
  | 4 × 64 (GDN-A) | 0.84 | 0.69 | 0.50 | 0.33 | 0.23 |
  | 1 × 256 (GDN-E) | 0.96 | 0.91 | 0.84 | 0.74 | 0.64 |
  | 2 × 128 (config D) | 0.92 | 0.84 | 0.71 | 0.55 | 0.42 |

  This confirms B2's numbers.
- **H5 artefact** re-derived by hand: 3.55 and 1.96 pp per doubling (§6.4).

**Resolution: the frozen suite.**
- **R1:** D ∈ {3, 15} same-format distractors, at all lengths.
- **R-load:**
  - N ∈ {8, 32, 128, 512} at 8K;
  - N ∈ {8, 32, 128, 512, 1,024} at 16K and 32K;
  - 12N ≤ 0.75T holds at every level, with equality for N = 512 at 8K, N = 1,024 at 16K and N = 2,048 at 32K;
  - N = 2,048 at 32K, used for N50 only.
- **M-hop:** h ∈ {1, 2, 3, 4}, plus a diagnostic top level h = 5.
- **A-freq:** ρ ∈ {4, 2, 1.5, 1.25}, plus a diagnostic top level ρ = 1.1.
- No interim gate, and nothing tuned on pilots.
- Levels are described as N/N50₀, where N50₀ is the k = 0 model's measured capacity. The "theory-anchored phase boundary" wording is withdrawn.
- **Scores:** the primary score is the equal-weight ladder mean of teacher-forced exact match (EM_tf). N50 (logistic in log₂N, with left- and right-censoring reported) is secondary.

#### I-13. H5: what is the confirmatory test?
**Type:** PI-RULING, for A2. The README's wording decides it. This goes against B2's final table, which promoted the distance ladder.

**Positions.**
- **PROP:** slope of the gap in log₂T, plus a family × T interaction.
- **A1:** distance at fixed T = 32K × filler type as the confirmatory test.
- **B1:** slope over G-REF-passing lengths, plus a distance decomposition.
- **A2:** the length slope (the README's literal H5), on a level set present at every T; reference adequacy as a covariate; zero-shot LM-4 as co-primary; distance × filler as secondary.
- **B2:** A1's distance ladder plus load × length, on G-REF-passing cells only, with filler effects analysed per architecture.

So each reviewer ended at the other's round-1 position.

**Resolution.**
- **H5(a), confirmatory.**
  - Statistic: slope of Δ₀(T) = S₁₂(T) − S₀(T) in log₂T over all six lengths.
  - Level set (present at every T): R1 D ∈ {3, 15}; R-load N ∈ {8, 32}.
  - Test: per-seed slopes, paired across seeds.
  - Reference adequacy is reported per length. If any length fails G-REF, the verdict is labelled "conditional on reference adequacy", and the slope over passing lengths only is reported as a pre-registered sensitivity analysis.
  - A "no length effect" claim needs TOST within ±2 pp per doubling.
- **H5(b), secondary family:** the per-seed difference between the retrieval and multi-hop slopes (multi-hop fixed set h ∈ {1, 2}).
- **Mechanism decomposition, secondary:**
  - distance d ∈ {256, 2K, 8K, 30K} at T = 32K × filler {natural, repetitive}, analysed per architecture (S₀ and S₁₂ separately, B2-W12);
  - the load × length factorial at fixed N.
- **LM-4 copy-after-gap slope:** an exposure-free corroboration. The label "exposure-robust" requires agreement in sign.

**Why.**
- The README states H5 in terms of context length, and distance is a mechanism, not the hypothesis [constraint].
- Truncating on a gate correlated with the outcome biases the slope (A2 B-3c), yet a collapsed reference must be flagged (B2-W1). The conditional label does both.

#### I-14. H3: statistic and scale
**Type:** PI-RULING. Neither final position is adopted unchanged.

**Positions.**
- **PROP:** k80 differences plus a GLMM.
- **A1:** R_ret(3) − R_hop(3) at difficulty-equated levels.
- **B1:** GLMM family × k interaction, plus |R_f − R_ret| ≥ 0.2.
- **A2:** equated levels on the probability scale; logit GLMM as sensitivity; k = 6 in Tier B; frequency comparison primary for aggregation.
- **B2 W5(c, d):**
  - R differences bring ratio estimators back, so use Fieller intervals;
  - the logit-scale GLMM should be primary;
  - equated levels may not exist.

**Resolution.**
- **Statistic:** the paired per-seed D₃ = R_ret(3) − R_hop(3) on primary-cell ladder means, on the probability scale.
- **Test:** one-sided, in the pre-registered direction R_ret > R_hop.
- **Evaluability:** each per-seed, per-family gap G must be ≥ 15 pp; otherwise that comparison is non-evaluable.
- **Sensitivity analyses:** Fieller and bootstrap intervals; the logit GLMM family × k interaction.
- **Difficulty-equated check:** where equated level pairs exist on dev items (|ΔS₁₂| ≤ 10 pp across families), "supported" also requires agreement in sign there.
- **Aggregation family:** A-freq is primary and A-count is a diagnostic, because under the delta rule repeated writes overwrite rather than accumulate (A1-M5; B did not contest). Aggregation contrasts sit in the secondary analyses.
- **Curve points:** k ∈ {2, 6} × 2 seeds in Tier B, giving descriptive k80 intervals. Both reviewers have k = 6.

**Why.**
- "Required proportion" is a normalised-curve quantity on the accuracy scale.
- A logit interaction tests a different null hypothesis, and it can appear or vanish under monotone rescaling unless the curves cross (A2 B-6).
- Equated levels may be undefined (B2-W5d), so they cannot carry the primary test.

#### I-15. H4: placement design
**Type:** PI-RULING, for B2's conditional k*, with patterns specified by the PI. Value decided by experiment (U-07).

**Positions.**
- **PROP:** k = 2 arms, 10 pairwise contrasts.
- **A1:** k = 1 depth scan at {0, 6, 11}, with two contrasts.
- **B1:** k = 1 at {1, 6, 11} × 3 seeds.
- **A2:** B1's triad analysed with A1's two contrasts.
- **B2 (C3, W5b):** choose k conditionally. Wang et al.'s GDN row is flat up to 12:1, so three k = 1 arms might all sit at the floor.

**Resolution.**
- **Choosing k\*:** the smallest k ∈ {1, 2, 3} with R̂_ret(k) ≥ 0.3 at the primary cell. Only composition runs are used; the k = 2 curve runs go first in Tier B. If none qualifies, k* = 3.
- **Patterns** (spacing preserved; "late" is the industrial end-of-group convention):

  | k* | Early | Middle (shared with composition) | Late |
  |---|---|---|---|
  | 1 | {1} | {6} | {11} |
  | 2 | {1, 7} | {3, 9} | {5, 11} |
  | 3 | {1, 5, 9} | {2, 6, 10} | {3, 7, 11} |

- **Contrasts:** middle − mean(ends) and a linear trend, Holm over 2 within the secondary family.
- **Seeds:** n₀ seeds (5 or 4) per new arm. If k* = 2, the shared middle arm is topped up to n₀ seeds.
- **Inference:** paired primary, pooled as sensitivity.
- **Minimum detectable effect** for the paired middle − ends contrast at 80% power [INF, §6.4]:

  | Seeds | σ = 3 pp | σ = 2 pp |
  |---|---|---|
  | 5 | ≈ 7.3 pp | ≈ 4.9 pp |
  | 4 | ≈ 9.5 pp | ≈ 6.3 pp |

**Why.** Placing the scan on the rising part of the curve guarantees dynamic range. Because k* is chosen from composition results only, the placement contrasts stay unbiased (B2-W5b).

**Dissent (A2).** A2 wanted k = 1 for its direct practical reading: where to put the single global layer. That reading is kept whenever k* = 1.

#### I-16. H6 and H6′
**Type:** Merge. The tautology is consensus; the tier and wording merge both positions.

**Positions.**
- **PROP:** a hull margin against the line joining the two pure models.
- **A1-M6:** this is nearly tautological; replace it with H6′ using a larger-state GDN (GDN-E).
- **B1:** make H6 descriptive.
- **A2 B-5:** the *measured* margin is just as tautological for additive costs; run H6′ in Tier B at k ∈ {0, 1}.
- **B2 W5(a, b):**
  - state the manipulation in per-head d_k;
  - run it at k* rather than k = 1;
  - use Fieller intervals for the gap-share ratio.

**PI verification.** The identity c̃(k) = k/12, hence M(k) = G·(R(k) − k/12), holds for any per-layer-additive cost [algebra, §6.4]. The capacity re-run in I-07 supports the per-head d_k wording.

**Resolution.**
- **Literal H6:** reported descriptively, with the identity stated. Frontier plots use analytic state bytes, FLOPs and roofline latency, plus measured costs under I-18.
- **H6′ (secondary family):**
  - **Arm:** GDN-E has 1 head with d_k = d/2 and d_v = d. Per-head key dimension is ×4 at D512 (×3 at D384), at equal parameters and equal H·d_k (1,054,210 parameters at D512, A1 meta-device count).
  - **Runs:** k ∈ {0, k*} × 2 seeds, Tier B.
  - **Primary H6′ contrast** (linear, no ratio): S(GDN-E, k = 0) − S(GDN-A, k*) at the primary cell, with TOST at ±5 pp.
  - **State bytes reported alongside:** at D512 and 32K, 6.19 MiB (GDN-E, k = 0) vs 65.55 MiB (GDN-A, k* = 1).
  - **Descriptive:** the share of the gap closed by GDN-E at k = 0, with a Fieller CI.
  - **Kernel feasibility** is no longer a risk: B2 ran K = 256 forward and backward on sm_120 with finite losses, and its chunk numerics match config A [MEAS-B2]. Fallback: config D (2 × 128).

### D. Training, exposure and position

#### I-08. LM-regime feasibility pilot vs direct-supervision screen
**Type:** FEAS-first is consensus. The tier of the direct-supervision (DS) screen is a PI-RULING against B2.

**Positions.**
- **A1:** a DS screen (S2), plus "Level A" falsification.
- **B1-M1:** an LM-regime feasibility pilot (FEAS) with a budget rule and a fallback ladder.
- **A2 C-3:** FEAS first; DS screen to Tier C.
- **B2:**
  - FEAS also decides the scale;
  - DS screen in Tier 1 (≈ 5 h) for N50 anchors and as a bimodality screen;
  - Level A is a screen, not a falsification.

**Resolution.**
- FEAS is the first scientific gate. Level A is withdrawn (I-22).
- The DS screen goes in Tier C. It is promoted into Tier A only if the FEAS dev ladders show floor or ceiling at every level for k = 0 or for k = 12. It would then diagnose whether the generator or the LM regime is at fault.

**Why.**
- The N50₀ anchor comes from FEAS and from the final k = 0 models.
- Bimodality under direct supervision need not carry over to LM-regime seeds. Success counts and the range trigger (I-04) act on the seeds that actually carry the verdict.

#### I-09. Position encoding
**Type:** RoPE default is consensus. The NoPE arm is a merge (U-06).

**Positions.**
- **PROP:** a three-way pilot.
- **A1:** a two-way, two-seed pilot, with the other encoding at k ∈ {1, 12} in Tier 1.
- **B1:** RoPE θ = 5e5 by default; NoPE in Tier C.
- **A2:** RoPE by default; NoPE at k ∈ {1, 12} × 1 seed in Tier A, and a second seed if |ΔR| > 0.1.
- **B2:** NoPE at k ∈ {3, 12} × 2 seeds in Tier 1.

**Evidence.**
- **Llama 3:** RoPE base 500,000, "effective for context lengths up to 32,768" ([2407.21783](https://arxiv.org/html/2407.21783) [read-section by A2]).
- **Qiao et al.:** NoPE on the full layers of a 1:1 SWA-128 hybrid raised NIAH@16K from 49.58 to 67.81 at S4 [PI-VER]. Note this is an SWA hybrid, not a linear-attention one.

**Resolution.**
- RoPE θ = 5e5 from step 0 in every arm, with no pilot.
- NoPE at k ∈ {3, 12}, seed 0, in Tier A.
- Seed 1 is added in Tier B if |ΔR₃| > 0.1 or |ΔS₁₂| > 3 pp at the primary cell.
- The NoPE arm is descriptive and for sensitivity only.

**Why.**
- k = 3 is where the confirmatory H1 lives (B2's argument).
- k = 1 is probably uninformative here, given the threshold-like GDN prior (A1-M9).
- One seed plus a trigger keeps the cost near A2's.

#### I-10. QK-norm softmax dilution and SSMax
**Type:** Resolved procedurally. Severity remains disputed but no longer affects any decision (U-03).

**Positions.**
- **B1-M3:**
  - QK-RMSNorm with shared unit gains caps logits at 8, so needle weight falls from 0.64 to 0.05 between 1K and 32K.
  - Adopt SSMax if it is ≥ plain on one seed at 16K and 32K, and ≤ 2 pp worse at 4K.
- **A2 B-7:**
  - Reproduces the weights, but an ideal readout stays at 1.000 accuracy at every length [MEAS-A].
  - Gains need to grow only +23%.
  - QKNorm precedent: a learnable scale initialised at log₂(L² − L) ([2010.04245](https://arxiv.org/abs/2010.04245) [read-section by A2]).
  - A single-seed "≥ 0" rule is a coin flip; require ≥ 3 pp.
- **B2:** this is a small-head artefact (Qwen3-Next uses head_dim 256, so its cap is 16), and it lands exactly on the 32K cells.

**Resolution.**
- Per-head, unshared gains (consensus).
- **SSMax:** q ← q·s_h·ln(t + 1), with s_h initialised to 1/ln 4096 ([2501.19399](https://arxiv.org/html/2501.19399) [read-section by B1]).
  - Tested at k = 12 inside FEAS.
  - Adopted only if it gains ≥ 3 pp at 32K and loses ≤ 2 pp at 4K, on the dev ladder mean.
  - If adopted, it applies identically to every arm.
- **Logging** on dev items at 32K: needle attention mass, attention entropy and effective maximum logit per layer.
- The non-chosen variant runs at k ∈ {3, 12} in Tier C.

#### I-11. Synthetic exposure fine-tune (TEF)
**Type:** Consensus except the synthetic share, which FEAS decides (U-08).

**Agreed.**
- An identical post-pretraining TEF (PROP's option ii).
- Equal item counts per log₂ length bucket {512, …, 32K} and per family.
- 1,000 steps × 64K tokens; a 50-step re-warm to 3e-4, then a constant LR.
- Checkpoints at 250 / 500 / 1,000 steps. Seed-0 anchors continue to 4,000 steps, with a checkpoint at 2,000.
- **"Still closing" flag:** if the k vs k = 12 gap changes by > 3 pp between 1,000 and 4,000 steps, the verdict is "budget-conditional".
- Falsification requires the 4,000-step branch to agree (B2 P6).
- Tier 0 is measured before the TEF; LM-2 (4K validation loss) before and after.

**Disputed: the synthetic share.** A1 proposed 90/10, then A2 accepted 50/50. B1 proposed 50/50, then B2 adopted 90/10 with a forgetting gate.

**Resolution.**
- **Default 90/10.** At 1,000 steps this gives ≈ 928 items per length bucket, vs ≈ 516 at 50/50 [INF: 0.9 or 0.5 × 65.5M tokens / 63.5K tokens per bucket set].
- **Forgetting gate (B2),** checked on the FEAS k ∈ {0, 12} TEF branches:
  - LM-2 may rise by ≤ 0.05 nats;
  - the rise may differ between k by ≤ 0.02 nats.
- If the gate fails, fall back to 50/50. If it still fails, lower the TEF LR to 1e-4. Both fallbacks are pre-registered.

#### I-12. Data, curriculum and schedule
**Type:** Merge. The schedule is consensus; which branches to run is a PI-RULING.

**Agreed.**
- **Stage A:** 0.8B tokens (D512) or 0.6B (D384) at 4K. Mix: 85% FineWeb-Edu + 15% PG19-train 4K windows. Constant LR after a 500-step warm-up. Checkpoint S at the end of the stable phase is the branch point.
- **Stage B:** at 32K, linear LR decay to 0.
  - Budget set by FEAS-3: {0.1, 0.2, 0.3}B at D512; {30, 60, 90}M at D384.
  - Mix: 60% PG19-train 32K windows + 40% FineWeb-Edu.
- Effective-context histograms are logged per stage.

**2× Stage-B saturation branch.** A2 proposed k = 12; B1 k ∈ {1, 12}; B2 k ∈ {3, 12}.
- **PI-RULING:** k ∈ {0, 3, 12} at seed 0, unconditionally. Cost: 8.4 h at D512, 2.5 h at D384.
- **Why:** R can only be recomputed at 2× if all three are run. The arm also tests B1-E8: whether hybrids gain more from a longer long-context stage.

#### I-26. LR protocol
**Type:** Consensus.
- **Sweep** at the chosen scale: k ∈ {0, 3, 12} × 4 LRs × 0.15B tokens, with a compressed WSD schedule. Extend the grid if the optimum lands on an edge.
- **Selection:**
  - a shared η* if minimax regret ≤ 0.02 nats; otherwise per-configuration LRs, interpolated in k;
  - never selected on task metrics.
- Control arms inherit their k's LR.
- An η*/√2 check runs in Tier C.
- Gating both mixers reduces the LR-tolerance asymmetry raised in A1-m2.

### E. Generality

#### I-17. GLA and conv-free GDN arms
**Type:** PI-RULING, against B2's Tier 2. Promotion rule in U-13.

**Positions.** A1: GLA-twin in Tier 2. B1 and A2: Tier C. B2: GLA-twin in Tier 2 at k ∈ {0, k*, 3}.

**Resolution.**
- **Tier C, priority 1: GLA** at k ∈ {0, k*, 3} × 2 seeds.
  - FLA `GatedLinearAttention` with conv and a gated output, 1,065,344 parameters at D512 (A1 count).
  - Runs on sm_120 [MEAS-B2].
- **Tier C, priority 2: conv-free GDN** at k ∈ {0, k*} × 2 seeds.
- **Promotion:** if none of Tier B's conditional arms fire, leaving ≥ 20 h unspent, GLA moves into Tier B.
- All claims are scoped "for GDN-A".

**Why.** RQ-A–D do not require generality across mechanisms. Placement is an explicit README question (RQ-B), and H6′ makes README H6 testable, so both rank higher.

### F. Measurement and engineering

#### I-18. Cost measurement protocol
**Type:** Consensus. One feasibility question is settled in S0 (U-10).

**B1-F2 (raised as FATAL).** The planned measurements carry artefacts of opposite sign [MEAS-B]:
- a masked static cache makes SDPA dispatch to the mem-efficient kernel: 1.80 ms vs 85–90 µs per layer at 32K;
- L2 residency speeds attention decode up to 2.5× (33 vs 84 µs);
- GDN at batch 1 is launch-bound;
- kernel maturity differs: FA2 reaches 81% of peak, fla's GDN chunk core ≈ 10%.

**A2 B-10 / B-14.**
- Adopted B's protocol.
- Severity is MAJOR, not fatal, because analytic costs are primary.
- FlexAttention gives an equal *toolchain*, not equal *maturity*.
- Add a sequential-depth term for chunked recurrence, and projections to larger widths.

**B2-W7.** The twin adds its own 26× dispatch trap. This is moot: no Tier-A/B arm has d_v ≠ d_qk.

**Resolution.**
- **Analytic, primary:**
  - state bytes exactly as stored;
  - FLOPs and bytes moved per token;
  - roofline latency at 112.6 TFLOP/s and 960 GB/s, labelled as lower bounds;
  - a sequential-depth term of T/64 chunk steps for GDN;
  - a projection table for d ∈ {384, 512, 2048, 4096}. Per full layer, attention-to-dense FLOPs ≈ T/(13d), i.e. 4.9 at d = 512 vs 0.62 at d = 4096 for T = 32K.
- **Measured, secondary,** under B1's protocol:
  - exact-shape CUDA graphs per context length; no masks;
  - attention backends pinned and asserted at runtime;
  - cold L2 (256 MB flush before each replay) and hot L2;
  - min-of-N with N ≥ 30; randomised interleaved blocks; 2 sessions;
  - reps with throttle flags discarded;
  - a kernel-efficiency (roofline-fraction) table;
  - FlexAttention reported as "equal toolchain, not equal maturity".
- CI asserts that no inference path dispatches to the mem-efficient kernel.
- GQA appears as bytes only, with no capability claim.
- **Severity:** MAJOR.

#### I-19. Determinism
**Type:** Consensus (B2 C5 concedes; [Summers & Dinneen, 2103.04514](https://arxiv.org/abs/2103.04514) [abstract-only, both reviewers]).

**B's measurements** [MEAS-B]:
- k = 0 training is bitwise reproducible; k = 12 is not (FA2 backward).
- Deterministic mode makes k = 12 bitwise, at about +7% across the core.
- fla's TF32 path gives errors of 7e-4 (fp32 inputs) and 3.1e-3 (bf16), which breaks PROP's absolute thresholds.

**Resolution.**
- **Deterministic mode** for:
  - the G0 audits: bitwise loss traces and parameter SHA-256 across processes for k ∈ {0, 3, 12}, plus bitwise kill-and-resume;
  - one standing replicate: k = 12, seed 0, first 1,000 steps.
- **Production** runs in default mode.
- **No cuDNN SDPA in training.**
- **Relative kernel thresholds:**
  - relative RMS ≤ 3e-3 on the TF32 path and ≤ 1e-2 for bf16;
  - at T ∈ {512, 32K}, in two decay regimes;
  - plus an fp64 gradient check at T = 512.
- **Production resumes** are logged and checked: |Δloss| < 0.005 at resume + 500 steps.

**Reopen if:** G0 on an idle GPU measures ≤ 3% overhead; production then runs deterministically.

#### I-20. Reproduction and equivalence gates
**Type:** PI-RULING, adopting B2-W10. A had no chance to respond, since B2 came after A2.

**Resolution.**
- **Zoology MQAR:** re-run numbers only. In the target figure, attention and sliding-window points share one marker (A1-m1; [zoology](https://github.com/HazyResearch/zoology) @`1ad20d1`).
- **Implementation equivalence:** training-free weight-transplant logit tests, relative tolerance 1e-3 on fp32 logits:
  1. our k = 12 model ↔ transformers `Qwen3ForCausalLM`, with the gate bypassed by a test-only flag, per-head gains tied to Qwen3's shared `q_norm`/`k_norm` weight, and the RoPE convention matched;
  2. our k = 0 model ↔ fla `GatedDeltaNetForCausalLM` with a matched configuration;
  3. a separate unit test of the headwise gate.
- A1's 0.1B-token comparison run is kept as a smoke test only.

#### I-24. Shared-GPU operations and budget accounting
**Type:** Consensus.
- **Memory:**
  - cap of ≤ 7 GiB per process; every measured peak is ≤ 4.7 GiB [MEAS-B, MEAS-B2];
  - Stage A uses 2 × 4K micro-batches without checkpointing;
  - Stage B and the TEF use 1 × 32K with checkpointing.
- **Checkpoints:** atomic, every 250 steps (tmp → fsync → `os.replace`); keep the last 2.
- **Runner:**
  - stateless data loader keyed on (seed, step);
  - auto-resume; an OOM is a logged event, and 3 consecutive failures mark a run failed;
  - `nvidia-smi` sampled every 30 s to compute `contended_frac`.
- **Budget accounting:** budgets use contended measured throughput, plus 20% contingency, with no idle-speed-up multiplier.

#### I-29. Smaller points settled by consensus
- **Scoring path.**
  - fla switches to `fused_recurrent` whenever q_len ≤ 64 outside training (`fla/layers/gated_deltanet.py` L229–230 @`98193113` [PI-VER, code-inspected]).
  - Fix: call the fla ops explicitly, and re-score 5% of items with a single-pass forward; the EM flip rate must stay < 1%.
- **Triton cache:** warmed for every training, evaluation and benchmark shape, then archived with its SHA-256 in the registry.
- **TF32 inside fla's GDN path** is disclosed.
- **Initialisation:** a unit test that `A_log` and `dt_bias` survive global initialisation.
- **Pathway diagnostics** (inference only, exploratory): window the full layers vs reset the GDN state; mean-ablate each attention layer; measure attention mass on the target.
- **"Information integration"** is scoped to literal-token aggregation.
- **Contexts per cell** are set from the intraclass correlation measured in FEAS. If the design effect exceeds 2, confirmatory retrieval cells get 600 contexts.

### G. Priors, framing and limits

#### I-23. Priors and novelty
**Type:** Consensus.

**Wang et al.** ([v1](https://arxiv.org/html/2507.06457v1), [v2](https://arxiv.org/html/2507.06457v2) [read-section by A1/A2; Table 7 confirmed by B2]):
- The GDN recall row is flat up to 12:1:

  | Pure | 24:1 | 12:1 | 6:1 | 3:1 |
  |---|---|---|---|---|
  | 0.356 | 0.341 | 0.352 | 0.425 | 0.436 |

- The results are single runs, and Table 5's "Avg" is not the column mean.
- The two versions conflict on training context: v1 says "2,048-token", v2 says "4,096-token".

**Consequences.**
- H1-min carries a pessimistic prior.
- Wang et al. is never used for power or ceiling planning.
- PROP's "strong H6 is plausible because 0.436 > 0.423" is withdrawn.

**Framing.**
- The contribution is a controlled, per-family, length-resolved, seed-replicated curve at ≈ 28–49M parameters, with an operator control.
- It must credit:
  - MAD: 25% attention compute-optimal, 10% when state-optimal ([2403.17844](https://arxiv.org/html/2403.17844) [read-section by A1]);
  - Based's recall–state frontier ([2402.18668](https://arxiv.org/abs/2402.18668) [abstract-only]);
  - Wang et al.'s Appendix C (subtask × ratio; figure only);
  - Qiao et al.;
  - [2610.06750](https://arxiv.org/abs/2610.06750), [2609.33093](https://arxiv.org/abs/2609.33093) and [2610.25250](https://arxiv.org/abs/2610.25250) [abstract-only].
- The "long-running agents" discussion is limited to mechanism-level statements (A1-m9).

#### I-27. Elicited vs zero-shot capability
**Type:** OPEN, conceptual (A1 Q1; B2 §6.6).
- Without exposure, small models sit at the floor on these task families (Wang et al.: 340M recall "insignificant").
- So the TEF-elicited scores answer the README's task-family questions. The zero-shot Tier-0 metrics (LM-3, LM-4) corroborate them.
- Disagreements between the two are reported, and claims are scoped "capability elicited after identical exposure".
- No experiment in budget decides which is *the* answer.

#### I-28. Out of scope
These are stated as limitations:
- **Compute vs memory:** full layers also have more compute per token at long T, and no arm separates the two (A1 Q2).
- **Document masking:** its interaction with k is untested (A1 Q6).
- **Count vs proportion:** F24 is out of scope.
- **Scale transfer** beyond the chosen scale (A1 Q4).

---

## 3. Unresolved items and the experiment that decides each

For D384/D512 costs, the first figure is the D384 path.

| ID | Open question (positions) | Deciding experiment (stage; cost) | Pre-registered decision rule | Default if not run |
|---|---|---|---|---|
| U-01 | Primary scale (A2: D512; B2: D384 if feasible) | FEAS-384: k ∈ {0, 12} × seeds 101–102 + SSMax variant; Stage A 0.3B; Stage B 30M/90M; TEF 250/1,000 (≈ 6 h). Then FEAS-512 if needed (≈ 12 h) | D384 iff both seeds at 32K reach S₁₂ ≥ 0.85·S₁₂(C0) and S₁₂ − S₀ ≥ 25 pp. Else D512 under FEAS-1 (0.8) and FEAS-2 (20 pp). Else L_ext = 16K. Else stop and re-scope (user) | D512 |
| U-02 | Does the missing conv handicap the all-full reference? (B2's case for a twin primary) | F_gc at k = 12 × 2 seeds, Tier A (4.7 / 10.2 h) | Shift > 3 pp → F_gc at k ∈ {1, 3} × 2 (7.8 / 16.1 h); report H1 in both families. Shift > 6 pp → F_gc becomes co-primary | Package wording only |
| U-03 | Is QK-norm dilution a real limit at 32K with d_h = 64? | SSMax k = 12 variant inside FEAS; needle-mass, entropy and max-logit logs; G-REF | Adopt SSMax iff ≥ 3 pp gain at 32K and ≤ 2 pp loss at 4K. Non-chosen variant at k ∈ {3, 12} in Tier C | Plain per-head gains |
| U-04 | Is the reference competent at long lengths, and what Stage-B budget is needed? | FEAS-1/2/3; full-budget re-check on seed 0; 2× Stage-B branch at k ∈ {0, 3, 12} (2.5 / 8.4 h) | FEAS-3: Stage-B gain ≤ 3 pp → smallest budget; 3–10 pp → middle; > 10 pp → largest. 2× branch: S₁₂ change ≤ 5 pp, else verdict "budget-conditional" with R recomputed | — (required) |
| U-05 | Seed variance, unequal variance, bimodality: how many seeds? (A2: pooled, 3; B2: paired, 5) | Trigger evaluated after n₀ seeds of all anchors | +2 seeds for all four anchors if the blinded pooled residual SD > 2.5 pp or any anchor's range > 12 pp (15.3 / 31.9 h). Success counts reported | n₀ seeds |
| U-06 | Position encoding × k (A2: {1, 12} × 1; B2: {3, 12} × 2) | NoPE at k ∈ {3, 12}, seed 0, Tier A (4.1 / 8.8 h) | Add seed 1 if \|ΔR₃\| > 0.1 or \|ΔS₁₂\| > 3 pp | Conditional on RoPE |
| U-07 | Placement k (A2: k = 1; B2: k*) | Tier-A composition results plus Tier-B k = 2 curve runs | k* = smallest k ∈ {1, 2, 3} with R̂_ret ≥ 0.3; if none, k* = 3; patterns as in I-15 | k* = 1 |
| U-08 | TEF synthetic share (90/10 vs 50/50) | FEAS k ∈ {0, 12} TEF branches (inside FEAS) | Forgetting gate: LM-2 rise ≤ 0.05 nats and between-k difference ≤ 0.02 nats → 90/10. Else 50/50, then LR 1e-4 | 90/10 |
| U-09 | Learning speed vs capability (both reviewers: open until measured) | TEF dose ladder 250/500/1,000/2,000/4,000 at seed-0 anchors (inside branches) | Gap change > 3 pp from 1,000 to 4,000 steps → "budget-conditional"; falsification requires plateau (< 3 pp, 2,000 → 4,000) | — |
| U-10 | Is a FlexAttention decode comparator feasible in one captured graph? (B1 Q7; A2 §4.4) | S0 harness self-test | If device-side length update fails: Flex comparator for prefill only; decode measured with per-length exact-shape graphs | Prefill only |
| U-11 | Do difficulty-equated level pairs exist for H3? (B2-W5d) | Dev items of the final k = 12 models | Pairs with \|ΔS₁₂\| ≤ 10 pp → sign check required; none → ladder means only, stated as such | Ladder means |
| U-12 | Does head geometry (4 vs 8 heads, d_v = 128) matter for the operator reading? (B2-W6.3, [HYP]) | Optional Tier-C twin at k = 12 × 2 (≈ 4.7 / 10.2 h) | Reported descriptively; no confirmatory role | Scope limitation |
| U-13 | GLA tier (A2: Tier C; B2: Tier 2) | Budget left after Tier B's conditional arms | No conditional arm fires (≥ 20 h unspent) → GLA moves into Tier B | Tier C |
| U-14 | Idle-GPU availability and speed-up (both reviewers) | User decision; G0 measurement in an idle window | No idle window → measured costs go to an appendix only, and H6/RQ-D are judged on analytic costs | Analytic only |
| U-15 | Contexts per cell (intraclass correlation) | FEAS dev evaluation | Design effect > 2 → 600 contexts for confirmatory retrieval cells; else 300–400 | 400 |
| U-16 | Elicited vs zero-shot as "the" answer | No experiment decides it | Report both; scope claims to "capability elicited after identical exposure" | — |

---

## 4. Consolidated plan implied by the resolutions

This is a summary for the protocol author, not the frozen protocol.

**Architecture** (both scales): L = 12, pre-norm RMSNorm, SwiGLU, tied vocabulary of 16,384, no biases, bf16 autocast with fp32 master weights and fp32 GDN state.

| | D384 path | D512 path |
|---|---|---|
| d / d_ff | 384 / 1,024 | 512 / 1,536 |
| F_g (primary full mixer) | 6 × 64, gated: 592,896 params | 8 × 64, gated: 1,053,696 params |
| GDN-A (primary linear mixer) | 3 × (64 / 128): 595,334 params | 4 × (64 / 128): 1,056,904 params |
| Non-embedding params, k = 0 / k = 12 | 21,309,384 / 21,280,128 | 41,007,200 / 40,968,704 |
| Total params, k = 0 | 27.6M | 49.4M |
| Seeds n₀ for anchors and placement | 5 | 4 |

**GPU-hours** (contended, base; §6.4 has the derivation):

| Block | Content | D384 | D512 |
|---|---|---|---|
| S0 | G0 audits, kernel tests, harness self-test, weight-transplant tests, Triton cache | 3.0 | 3.0 |
| S1 | Zoology MQAR, re-run numbers only | 3.0 | 3.0 |
| FEAS | FEAS-384; plus FEAS-512 on the D512 path | 6.0 | 18.0 |
| LR | k ∈ {0, 3, 12} × 4 LRs × 0.15B | 4.2 | 5.6 |
| Core | k ∈ {0, 1, 3, 12} × n₀; k = 12 and k = 0 seed 0 first | 38.4 | 63.8 |
| Controls | F_gc at k = 12 × 2; NoPE at k ∈ {3, 12} × 1 | 8.8 | 19.0 |
| Branches (seed 0) | TEF to 4,000 steps at {0, 1, 3, 12}; 2× Stage B at {0, 3, 12} | 5.8 | 12.8 |
| Evaluation and cost benchmarks | | 10.5 | 13.0 |
| **Tier A** (decision point: H1, H1-min, H3, H5a) | | **79.7 (95.6 with 20%)** | **138.3 (165.9)** |
| Tier A, conditional | +2 anchor seeds (U-05) | 15.3 | 31.9 |
| **Tier B** (k* = 1 shown) | Curve k ∈ {2, 6} × 2 (run first); placement at k* (2 arms × n₀); GDN-E at {0, k*} × 2; evaluation | **36.9** | **66.9** |
| Tier B, conditional | F_gc at k ∈ {1, 3} × 2; NoPE seed 1; top-up of the centred arm if k* = 2 | ≤ 17.4 | ≤ 32.4 |
| Tier C, in priority order | GLA; conv-free GDN; twin k = 12; non-chosen SSMax variant; direct-supervision screen; the other scale (k ∈ {0, 3, 12} × 1, "scale extension"); η*/√2; 4× tokens | — | — |

**Totals.** Tiers A+B base: ≈ 117 h on the D384 path (≈ 140 h with contingency), ≈ 205 h on the D512 path (≈ 246 h).

**Economy option for the D512 path, if the user caps the budget.** Use n₀ = 3 (still +2 on the trigger).
- Saves 16 h in Tier A.
- PI-SIM power at R = 1.0 drops to 0.82 (σ = 3) / 0.87 (σ = 2), from 0.92 / 0.98.

**Order of execution.**
1. S0, then S1.
2. FEAS-384; FEAS-512 only if needed (U-01).
3. LR sweep at the chosen scale.
4. G5 freeze: protocol, item files (including dev-gate items), analysis code tested on synthetic data, Triton cache with SHA-256.
5. Core k = 12 and k = 0 seed 0, then re-confirm FEAS-1/2 on these full-budget models.
6. The remaining core seeds, then the U-05 trigger.
7. Controls and seed-0 branches.
8. Evaluation; cost benchmarks in idle windows.
9. Tier-A decision point; evaluate the U-02 and U-06 triggers.
10. Tier B: curve runs → k* → placement and GDN-E → conditional arms.
11. Tier C by priority.

---

## 5. Decisions needed from the user

1. **CARLA idle windows.** Can CARLA be paused for 2 × 2 h? The windows would be used for G0 idle throughput, Triton warm-up of benchmark shapes, and the cost benchmarks. If not, measured costs are appendix-only (U-14).
2. **Re-scope fallback.** Do you agree in advance to the stop/re-scope branch? It applies if the reference fails at 32K and at 16K at both scales. Options: (a) a ≤ 8K question; (b) a larger model, possibly outside 20–60M.
3. **Budget ceiling.**
   - D384 path: Tier A ≈ 80 h base (96 h with contingency); Tier B adds ≈ 37 h.
   - D512 path: Tier A ≈ 138 h (166 h); Tier B adds ≈ 67 h.
   - Is the D512 economy option (n₀ = 3) acceptable if needed?
4. **"Data fixed".** Do you accept that this includes an identical synthetic fine-tune for every configuration (PROP §17.5)?
5. **Package framing.** Do you confirm that H1 is answered as a package claim (gated-MHA layers vs GDN-A layers, as built), with operator-level wording conditional on the conv control (I-01)?
6. **Job persistence.** Run `loginctl enable-linger`, so that `systemd-run --user` jobs survive logout, or accept the `nohup` fallback.

---

## 6. PI verification log and simulations

### 6.1 Sources and code re-checked this round

| Claim used in a resolution | Source | Access | Result |
|---|---|---|---|
| Headwise vs elementwise gate (Table 1, 15A2B MoE: +1.6M params, PPL 5.792; +201M, 5.761; baseline 6.026). RULER (Table 5): 79.50 vs 79.77 at 32K without YaRN; 37.51 vs 66.60 at 64K with YaRN | [2505.06708](https://arxiv.org/html/2505.06708) | read-section | Confirmed (I-01) |
| Qiao et al.: 16K pretraining on a 1:1 long/short mix (§4.1); S1 at D = 100N excluded as "too small for stable long-context behavior" (§4.3); S1–S5 = 15/31/65/104/477M non-embedding; NIAH@16K at S4: Full 35.95, SWA-128 49.58, SWA-128-NoPE 67.81 (S5: 67.14 / 65.91 / 82.31) | [2606.15378v1](https://arxiv.org/html/2606.15378v1) | read-section (first 100K characters) | Confirmed (I-02, I-06, I-09). The RoPE base (10⁵ per A1, Table 8) lies in the unread part, so it is attributed to A1 |
| Qwen3-Next attention: sigmoid output gate (L300); conv only in the GDN path; head_dim 256. OLMo-hybrid attention: only q/k norms. Kimi-Linear attention: MLA with NoPE, no gate or conv | transformers 5.22.2 in `.venv` (`models/qwen3_next/`, `models/olmo_hybrid/`, `models/kimi_linear/`) | code-inspected | Confirmed (I-01) |
| fla auto-switch to `fused_recurrent` at q_len ≤ 64 outside training | fla 0.6.0 @`98193113ecb7835143252a786e06933e371b67cc`, `fla/layers/gated_deltanet.py` L226–232 | code-inspected | Confirmed (I-29) |
| Versions | torch 2.11.0+cu128, triton 3.6.0, transformers 5.22.2, flash-linear-attention 0.6.0 (project venv) | — | Recorded |

**Not re-verified by PI** (accepted from the reviews, each with its reviewer's access tag):
- Olsson et al. (verified verbatim by A1 and B2);
- Wang et al. Table 7 (A1, A2, B2);
- MAD (A1);
- Llama 3 RoPE (A2);
- QKNorm (A2);
- Summers & Dinneen (both reviewers, abstract-only);
- SSMax and Chiang & Cholak (B1);
- Jelassi et al. (B1);
- the 2026 papers (A1, abstract-only);
- all [MEAS-B] and [MEAS-B2] GPU numbers. No GPU code was run by the PI.

### 6.2 PI-SIM: operating characteristics of the H1 tests

**Script:** `/tmp/claude-1000/-home-sedrica-lynx-Downloads-outeropt/73b71b1c-85a4-4704-b75f-7c9b668d4052/scratchpad/pi/h1_stats_pi.py`. The 5 → 7 rule is in `/tmp/claude-1000/-home-sedrica-lynx-Downloads-outeropt/73b71b1c-85a4-4704-b75f-7c9b668d4052/scratchpad/pi/h1_adapt57.py`.

**Model.** S[k, s] = μ_k + b_s + e_ks, for k ∈ {0, 1, 3, 12}.
- Means: μ₀ = 40, μ₁₂ = 70 (G = 30 pp), μ₁ = 55, μ₃ = 40 + 30R₃.
- Seed effect b_s ~ N(0, 2²).
- Residual e ~ N(0, σ²), with σ ∈ {2, 3} pp.
- 60,000 replicates per cell (40,000 for the adaptive rules).

**Scenarios.**
- **Hetero:** σ₃ = 8 pp at k = 3 only.
- **Bimodal:** each k = 3 seed independently drops by 25 pp with probability ⅓; the mixture mean is held at its target.

**Tests.**
- **POOL:** additive config + seed model on the 4 × n core table, df = 3(n − 1) (A2).
- **PAIR:** per-seed L₃, df = n − 1 (B2).
- **ADAPT n₀ → n₀+2:** PAIR, after adding 2 seeds to all anchors if the blinded pooled residual SD > 2.5 pp or any anchor's raw seed range > 12 pp.

All tests are one-sided at α = 0.05. "Support" means the lower bound of L₃ is > 0.

**P(support) at residual σ = 3 pp.** At R₃ = 0.8 this is the type-I error rate; at R₃ = 1.0 it is power.

| Scenario | R₃ | POOL n = 3 / 4 / 5 | PAIR n = 3 / 4 / 5 | ADAPT 3→5 | ADAPT 4→6 | ADAPT 5→7 |
|---|---|---|---|---|---|---|
| Homo | 0.8 | 0.049 / 0.050 / 0.050 | 0.049 / 0.050 / 0.049 | 0.052 | 0.051 | 0.053 |
| Homo | 1.0 | 0.763 / 0.885 / 0.946 | 0.545 / 0.755 / 0.880 | 0.819 (E[n] 4.34) | 0.923 (5.48) | 0.967 (6.59) |
| Hetero | 0.8 | 0.114 / 0.114 / 0.112 | 0.049 / 0.050 / 0.050 | 0.056 | 0.053 | 0.050 |
| Hetero | 1.0 | 0.466 / 0.554 / 0.621 | 0.216 / 0.299 / 0.372 | 0.379 | 0.448 | 0.514 |
| Bimodal | 0.8 | 0.280 / 0.197 / 0.151 | 0.224 / 0.186 / 0.133 | 0.178 | 0.117 | 0.086 |
| Bimodal | 1.0 | 0.312 / 0.391 / 0.449 | 0.292 / 0.202 / 0.208 | 0.253 | 0.296 | 0.297 |

**P(support) at residual σ = 2 pp.**

| Scenario | R₃ | POOL n = 3 / 4 / 5 | PAIR n = 3 / 4 / 5 | ADAPT 3→5 | ADAPT 4→6 | ADAPT 5→7 |
|---|---|---|---|---|---|---|
| Homo | 0.8 | 0.049 / 0.050 / 0.049 | 0.050 / 0.049 / 0.049 | 0.050 | 0.049 | 0.051 |
| Homo | 1.0 | 0.969 / 0.995 / 0.999 | 0.805 / 0.962 / 0.994 | 0.874 (E[n] 3.34) | 0.979 (4.31) | 0.997 (5.30) |
| Hetero | 0.8 | 0.136 / 0.135 / 0.133 | 0.050 / 0.051 / 0.050 | 0.062 | 0.058 | 0.054 |

**Readings.**
1. With equal variances, both POOL and PAIR are valid, and POOL is more powerful. This reproduces A2: its σ_d = 3 pp corresponds to residual σ ≈ 2.1 pp, where POOL with n = 3 gives ≈ 0.95–0.97.
2. With unequal variance, POOL's false-support rate more than doubles (0.11–0.14), while PAIR stays at nominal. This reproduces B2-W3.
3. With bimodal seeds, no few-seed test is valid. Error falls only with more seeds, which is why success counts are a named estimand.
4. The effect-blind trigger costs at most 0.012 in type-I error. It concentrates seeds where the variance is high.

### 6.3 PI-SIM: delta-rule capacity by head geometry

**Script:** `/tmp/claude-1000/-home-sedrica-lynx-Downloads-outeropt/73b71b1c-85a4-4704-b75f-7c9b668d4052/scratchpad/pi/capacity_pi.py`. Written independently of B2's script.

**Setup.**
- d = 512; fresh random projections per head; L2-normalised keys.
- Delta rule with β = 1 and no decay; every head writes every fact.
- Readout: concatenated head outputs; score = mean cosine to the true value; 4 trials.

**Results:**

| Head geometry | N = 32 | 64 | 128 | 256 | 512 |
|---|---|---|---|---|---|
| 4 × 64 / 128 (GDN-A; H·d_k = 256) | 0.84 | 0.69 | 0.50 | 0.33 | 0.23 |
| 2 × 128 / 256 (config D; H·d_k = 256) | 0.92 | 0.84 | 0.71 | 0.55 | 0.42 |
| 1 × 256 / 512 (GDN-E; H·d_k = 256) | 0.96 | 0.91 | 0.84 | 0.74 | 0.64 |

At equal H·d_k, interference is set by the per-head d_k, and degradation is gradual. This matches B2-W5(a). The caveat stands: trained models can learn head partitioning, β < 1 and more orthogonal keys.

### 6.4 Arithmetic [INF]

**Parameters.**
- **D512:**
  - F_g = 4·512² + 2·8·64 + 512·8 = 1,048,576 + 1,024 + 4,096 = **1,053,696**.
  - F_gc = F_g + 3·512·4 = **1,059,840**.
  - Blocks add SwiGLU (2,359,296) and norms (1,024).
  - N_non-emb(k) = (12 − k)·3,417,224 + k·3,414,016 + 512 = 41,007,200 − 3,208·k. So k = 12 gives 40,968,704, and the spread is 38,496 (0.094%).
- **D384** (d_ff = 1,024):
  - GDN-A = 595,334, which reproduces PROP's F12-S count from its components.
  - F_g = 4·384² + 2·6·64 + 384·6 = **592,896**.
  - F_gc = **597,504**.
  - N(k) = 21,309,384 − 2,438·k, consistent with B2's measured 21.309M at k = 0. So k = 12 gives 21,280,128, and the spread is 29,256 (0.137%).

**State bytes per sequence at T = 32K.**
- GDN-A layer: 4·64·128·4 B = 128 KiB, plus a ≈ 16 KiB conv cache = 144 KiB (D512).
- GDN-E layer: 1·256·512·4 B = 512 KiB, plus 16 KiB = 528 KiB. Twelve layers = 6.19 MiB.
- F_g KV cache: 2·d·2 B per token per layer, i.e. 2 KiB at D512 (64 MiB at 32K) and 1.5 KiB at D384 (48 MiB).
- k = 1 at D512: 11 × 144 KiB + 64 MiB = 65.55 MiB.

**The additive-cost identity behind H6.** If c(k) = k·c_A + (12 − k)·c_Λ + c₀, then c̃(k) = (c(k) − c(0))/(c(12) − c(0)) = k/12. Hence M(k) = S(k) − [S(0) + c̃(k)·G] = G·(R(k) − k/12).

**H5 ladder artefact** (A2 B.3, re-derived by hand).
- Under B1's rule (level N used at length T iff 12N ≤ 0.75T), the R2 levels in use are {8, 32} at 1K, {8, 32, 128} at 2K–4K, and all four from 8K.
- With length-constant gaps of 10 / 20 / 40 / 60 pp, the R2 ladder-mean gaps by length are 15, 23.3, 23.3, 32.5, 32.5, 32.5. The OLS slope is 62.1/17.5 = **3.55 pp per doubling**.
- Adding two R1 levels with constant 30 pp gaps gives **1.96 pp per doubling**.

**Placement MDE** (paired middle − mean(ends) contrast).
- Per-seed SD = σ·√1.5.
- Holm's first step uses two-sided α = 0.025.
- n = 5: (3.495 + 0.941)·1.2247σ/√5 = 7.3 pp at σ = 3, or 4.9 pp at σ = 2.
- n = 4: (4.177 + 0.978)·1.2247σ/2 = 9.5 pp at σ = 3, or 6.3 pp at σ = 2.

**Per-run costs** (contended).
- **D512**, from A2 App. B.7 (based on B1's measured 88.9K tok/s in Stage A and 70.5/64/59/51/44/31.9K tok/s in Stage B for k = 0/1/2/3/6/12):

  | k | 0 | 1 | 2 | 3 | 6 | 12 |
  |---|---|---|---|---|---|---|
  | Hours | 3.55 | 3.65 | 3.75 | 3.95 | 4.18 | 4.81 |

- **D384**: throughput scaled by 1.35 (Stage A) and 1.33 (Stage B and TEF), from B2's measured 1.30–1.39×. Stage A 0.6B, Stage B 60M, TEF 65.5M:

  | k | 0 | 1 | 2 | 3 | 6 | 12 |
  |---|---|---|---|---|---|---|
  | Hours | 1.76 | 1.80 | 1.83 | 1.90 | 1.98 | 2.21 |

- **Assumed overheads:** F_gc +6% [INF, from B2's −11% for the twin at 4K, which included extra norm kernels]; GDN-E +17% [INF, from B2's −11% to −25%].
- **Branches:** TEF to 4,000 steps adds 196.6M tokens. The 2× Stage-B branch is 0.4B tokens (D512) or 120M (D384) plus the TEF.

**Scope note.** This step writes only `reports/reviews/debate_record.md`. It does not edit `README.md`, `reports/decisions.md`, the proposal or any review. The decisions above should be entered in `reports/decisions.md` by its owner before the G5 freeze.

# Review A, round 1: scientific validity (adversarial)

**Reviewer:** A (scientific validity, adversarial). **Date:** 2026-11-09.
**Object of review:** `reports/initial_proposal.md` (v0.1), together with the evidence base it rests on.
**Read in full:** `reports/initial_proposal.md`, `reports/literature/literature_review.md` (*LR*), and `reports/literature/01_nemotron_h.md` … `05_implementation.md`.

**My own verification for this review.** Appendix A has the full log.
- I re-fetched the primary sources wherever a decision of mine depends on them:
  - Wang et al. 2507.06457v2: Tables 2, 5 and 7, and Appendices C and D;
  - Qiao et al. 2606.15378v1: Tables 1, 2 and 8, plus the restriction and conv experiments;
  - MAD 2403.17844: Findings 6–9;
  - Olsson et al. 2209.11895: PDF text extracted with `pdftotext`;
  - Stuffed Mamba, Canon layers, gated attention and Based;
  - four 2026 hybrid papers that the LR does not contain.
- I inspected FLA `fla/layers/gated_deltanet.py`, `fla/layers/gla.py` and `fla/ops/common/chunk_delta_h.py` at commit `98193113`.
- I counted parameters of every mixer I propose by meta-device instantiation in the project venv (torch 2.11.0+cu128, fla 0.6.0).
- I examined the Zoology target figure at `1ad20d1`.
- I ran **no GPU code**. At the time of review, `nvidia-smi` showed 100% utilisation and a second project process (pid 204085) next to CARLA, and I did not want to perturb anyone's measurement.

**Conventions.**
- Every literature claim carries a URL and an access tag: `[read-full]`, `[read-section]`, `[abstract-only]` or `[code-inspected]`.
- "(per 0X, not re-verified)" marks a claim I took from a specialist report without checking it myself.
- **[EST]** means reported with data in the cited source. **[HYP]** is a hypothesis. **[INF]** is my own inference, arithmetic or design judgement.
- k is the number of full-attention mixers among L = 12, and r = k/12.
- Nemotron-H ([2504.03624](https://arxiv.org/abs/2504.03624)) mixes attention, Mamba-2 and MLP layers and is not a controlled full-vs-linear ratio experiment. The proposal treats it correctly, and nothing below relies on it.

---

## 0. Verdict

**As written, the programme cannot establish its headline conclusions.** Two defects are each sufficient on their own, and both are cheap to fix.

1. **The H1/H3 statistic is not a property of the architecture.**
   - The recovery fraction R_f(k) is measured at a single difficulty. That difficulty is tuned so that the *0.2B-token pilot* all-full model scores 40–85%; the final models are then trained 5.5× longer.
   - Whether R(3) ≥ 0.8 is therefore jointly decided by architecture, difficulty knob and token budget. The same holds for every comparison of k80 across task families (H3).
   - Fix: theory-anchored difficulty ladders, with R read as a function of memory load (F1).
2. **The manipulation is not "the fraction of full attention with everything else fixed".**
   - Each F→Λ swap also adds a causal short conv with SiLU on q/k/v, an output gate with a per-head norm, and an implicit position source. The k = 12 endpoint is the only model in the family with none of these.
   - Two published mechanisms make this decisive for H1 specifically, not a nuisance:
     - Previous-token mixing lets **one** attention layer form induction circuits, and makes them form **earlier** ([Olsson et al.](https://arxiv.org/abs/2209.11895) [read-section]).
     - Neighbour-token mixing lifts NoPE attention to RoPE level ([Canon layers](https://arxiv.org/abs/2512.17351) [abstract-only]).
   - The proposal's only control (E5b) is optional and conditional. Fix: a "twin" full-attention mixer with identical projections, conv and gate (F2).

**Major problems on top of these:**
- The confirmatory test of H1 is most likely to end "inconclusive" (M1).
- Small-model floors and capability phase transitions are not designed for (M2).
- Training-length and exposure-per-length artefacts confound H5 (M3).
- Fixed exposure conflates learning speed with capability (M4).
- The H3 conclusions are specific to the delta rule (M5).
- H6 is nearly a tautology, and its measured costs do not transfer to larger scales (M6).
- Positional encoding is handled with a 1-seed pilot (M7).
- The placement arm cannot detect its own 5 pp threshold (M8).
- The proposal misreads the main prior, Wang et al. (M9).
- Novelty is overstated, with several close priors missing (M10).
- Count and proportion of attention layers cannot be separated (M11).

**What is good and should be kept:**
- intra-document masking in *both* mixers;
- an identical data order and blocked shared initialisation;
- parameter matching to within 0.24% at a fixed MLP width;
- cloze, exact-T, token-space items with query branching and leakage controls;
- in-distribution evaluation lengths;
- idle-GPU, CUDA-graph cost measurement;
- pre-registration, append-only registry and gates;
- careful scoping, and correct handling of Nemotron-H.

**My design in one paragraph (§2).**
- *Mixers.* The full and linear mixers are twins: identical q/k/v/g/o geometry, conv, gate and head count, so the swap changes only the memory operator (softmax over a KV cache vs a gated delta-rule state).
- *Discovery scale.* The discovery sweep runs at d = 384 (≈ 28M parameters, inside the 20–60M brief), where twice as many runs fit the budget.
- *Arms.* Pre-registered arms test the rival explanations directly:
  - plain MHA as a bridge to the literature design;
  - a second positional encoding;
  - GLA as a second linear mechanism;
  - a 3–4× larger GDN state at equal parameters;
  - GDN without its conv;
  - a k = 1 depth scan.
- *Exposure.* Natural-only pretraining is followed by an identical synthetic fine-tune that is balanced across lengths and has a dose ladder.
- *Evaluation.* Theory-anchored ladders measure memory load, distance, hops and frequency ratio.
- *Statistics.* Linear contrasts replace ratio estimators, run-level inference uses a pooled variance, and the confirmatory family has four tests.
- *Replication.* Tier 3 is a pre-registered replication at d = 512 with fresh seeds.
- *Budget.* Tier 1 ≈ 78 GPU-h conservative (≈ 50 idle); Tier 1+2 ≈ 127 (≈ 81 idle); Tiers 1–3 ≈ 189 (≈ 120 idle).

---

## 1. Critique of the initial proposal

Severity scale:
- **FATAL**: the claimed conclusion cannot follow from the design.
- **MAJOR**: the conclusion is at serious risk, or its scope is misstated.
- **MINOR**: should be fixed, but does not threaten the conclusions.

Each issue gives the problem, the evidence, the consequence and a concrete fix.

### F1 (FATAL, fixable). R_f(k) and k80_f are relative to task difficulty, and nothing guards against ceiling drift between pilot and final models

- **Problem.**
  - H1, H1-min and H3 use R_f(k) = (S_f(k) − S_f(0)) / (S_f(12) − S_f(0)) at one difficulty per task (§2, §9.2).
  - The difficulty knobs are tuned on dev items until the **pilot** all-full model, trained on 0.2B + 20M tokens, sits at 40–85% (§9.3). The final models train on 1.1B tokens.
  - R is bounded by each task's floor and ceiling. If every model with ≥ 1 attention layer saturates, R(1) = 1 whatever the architecture. If the task is hard, R shrinks.
  - "H1 supported / falsified" is therefore a joint property of (architecture, knob setting, budget).
  - H3 compares k80 across three *independently tuned* families, so it compares three difficulty calibrations rather than three intrinsic attention requirements.
- **Evidence that difficulty is not a nuisance here but the main axis.**
  - A delta-rule head stores at most d_k linearly independent keys exactly [INF, linear algebra; 02 §7 C6 states the same bound]. Config A therefore holds about 256 keys per layer.
  - The proposal's R2s, with 64 pairs, sits far below that capacity and "R2-dense" far above it, yet both are treated as variants of one family.
  - A 2026 mechanistic study of GLA/GDN reports that "recall and editability degrade as memory load grows, while elapsed context has a much smaller effect" ([2609.33093](https://arxiv.org/abs/2609.33093) [abstract-only]).
  - The value of R therefore depends mainly on memory load relative to state capacity, and that load is set by the experimenter.
- **Ceiling drift.**
  - G2 ("all-full within the target band") is checked only on the pilot. No gate re-checks headroom at 1.1B tokens.
  - A k = 12 model at 97% and hybrids at 92–97% would give R ≈ 0.9–1 purely by compression.
- **Consequence.** A "supported" H1 would only show that one tuned difficulty recovers 80%. It would not show the README's claim that "a small proportion of full-attention layers recovers most of the long-context performance".
- **Fix.**
  1. Replace single-difficulty headline cells with **pre-registered difficulty ladders whose levels are anchored in theory, not tuned** (§2.8):
     - memory load N ∈ {8, 32, 128, 512} facts at T = 8K, i.e. 0.04–2.7× GDN key capacity at d = 384;
     - target distance at fixed T = 32K;
     - hops 1–4;
     - frequency ratio from 4 down to 1.25.
  2. The primary family score is **accuracy averaged over the ladder**, i.e. the area under the psychometric curve.
     - Report R at each ladder level as the *difficulty-dependence result*, not as a nuisance.
     - Pre-register two readings: **H1-robust** (R ≥ 0.8 at every informative level, i.e. k = 12 between 20% and 90% and Ĝ ≥ 15 pp) and **H1-average** (R ≥ 0.8 on the ladder average).
  3. Test R ≥ 0.8 through the linear contrast **L_k = S_k − 0.2·S_0 − 0.8·S_12 ≥ 0**, which is equivalent when G > 0. It needs no ratio estimator (§2.10).
  4. Add an **interim gate that uses the endpoints only** (§2.11 G-I). If k = 12 scores ≥ 95% on every level of a ladder at mid-training, one pre-registered harder level is added. Hybrids are never consulted, so H1 inference stays unbiased.
  5. For H3, compare families at **difficulty-equated levels**, i.e. levels where k = 12 has the same accuracy in each family, in addition to k80.
- **Cost.** Evaluation only (≈ +0.05 GPU-h per model). No extra training.

### F2 (FATAL for RQ-A as phrased and for any mechanistic reading; fixable at almost no cost). The F→Λ swap changes four things besides the memory operator

- **Problem.**
  - The GDN mixer (§4–5) carries:
    - (a) a causal depthwise conv with kernel 4 and SiLU on q, k and v;
    - (b) a swish output gate after a per-head RMSNorm;
    - (c) L2-normalised q/k with learned decay;
    - (d) an implicit position source.
  - I confirmed (a) and (b) in `fla/layers/gated_deltanet.py` L168–197 at `98193113` ([repo](https://github.com/fla-org/flash-linear-attention) [code-inspected]).
  - The full mixer, plain MHA with 8 × 64 heads, has neither (a) nor (b), and it has twice as many heads.
  - So r is perfectly collinear with "fraction of layers that have token-shift and gating". The k = 12 endpoint is the only model with neither.
- **Why this acts on H1 itself.**
  1. **One-layer induction.**
     - Olsson et al.: induction heads "require a composition of attention heads, which is only possible with two or more layers".
     - With "smeared keys", where each key is a learned mix of the current and previous token's key (in effect a width-2 conv), "in-context learning does indeed form for one-layer models (when it didn't before), and it forms earlier for two-layer and larger models" ([2209.11895](https://arxiv.org/abs/2209.11895), PDF §"Argument 2" and the "Smeared key models" appendix [read-section]).
     - A GDN layer below an attention layer writes conv-mixed (previous-token) features into the residual stream. So in a hybrid a **single** attention layer can act as an induction or retrieval head, whereas in the all-full model two attention layers must compose.
     - "k = 1 recovers retrieval" may therefore mean "k = 1 plus conv recovers retrieval". At a fixed 1.1B-token budget, hybrids may also simply *learn the circuit sooner* [INF].
  2. **Position.**
     - Neighbour-token mixing ("Canon layers") is reported to "lift weak architectures like NoPE to match RoPE" ([2512.17351](https://arxiv.org/abs/2512.17351) [abstract-only]).
     - So the position-encoding (PE) pilot's outcome, and the PE confound itself, depend on *where the conv sits*.
  3. **Gating.**
     - A sigmoid output gate after scaled dot-product attention (SDPA) reportedly improves performance and stability, "tolerate[s] larger learning rates", mitigates attention sinks and improves long-context extrapolation ([2505.06708](https://arxiv.org/abs/2505.06708) [abstract-only]).
     - Here that gate is missing only from the full layers, and it interacts with the shared-LR rule (minor issue m2).
  - Supporting evidence from ablations:
    - Kimi: removing the conv moves validation perplexity from 5.65 to 5.70 (per 02, not re-verified).
    - Qiao et al. ablate GDN's conv at 15M non-embedding parameters (Table 7). I could not resolve its column labels, so I do not rely on it ([2606.15378](https://arxiv.org/html/2606.15378v1) [read-section]).
- **Why E5b does not fix it.**
  - E5b ("Attn+conv at k = 12") is optional and is run only "if any hybrid beats k = 12".
  - But the asymmetry biases the **whole curve**: through the denominator S(12), and through one-layer induction at small k. Its effect is not confined to the strong-H6 cell.
- **Consequence.**
  - RQ-A asks about "composition … when everything else is held fixed". That question is not answered.
  - Any statement of the form "softmax memory is needed for X" would be unsupported.
- **Fix.** Make the full mixer a **twin** of the linear mixer (§2.3):
  - the same projections: q, k: d → d/2; v, g: d → d; o: d → d;
  - the same conv(4) + SiLU on q/k/v;
  - the same head count, 4 × (d_k 64, d_v 128);
  - the same gated per-head RMSNorm output;
  - QK-RMSNorm in place of the kernel's L2 norm, which softmax needs to produce sharp logits.
  - Only the token-mixing operator then differs.
  - **Parameters at d = 512:** twin attention 1,052,928 (analytic) vs GDN-A 1,056,904 (meta-device count), i.e. −0.38% per mixer and about 0.12% across the whole k range.
  - **Kernels:** d_v = 128 can run on standard SDPA/varlen kernels by duplicating each QK head over two 64-dim value halves. This is mathematically identical and costs the same FLOPs as MHA 8 × 64.
  - **Bridge arm:** keep plain MHA (8 × 64, no conv, no gate) as a pre-registered bridge at k ∈ {1, 3, 12}. Results then still connect to the Jamba / Nemotron-style literature design.
  - **Conv-dependence arm:** plain MHA with conv-free GDN, k ∈ {0, 1}.
  - **Reference rule:** if the bridge shows that plain MHA at k = 12 beats the twin at k = 12 by more than 5 pp on the ladder average, the H1 reference becomes the stronger of the two (R_best, §2.2).
- **Cost.** Primary arm: 0 extra runs (re-parametrisation only). Bridge and conv arms: 2–8 runs at d = 384 (≈ 4–16 GPU-h).

### M1 (MAJOR). The confirmatory inference is underpowered, relies on a fragile ratio estimator, and spends α on near-tautologies

- **Numbers** [INF; Appendix B].
  - Take σ_d = 3 pp (the seed SD of a paired difference, as in §2 notes) and R ≈ 0.9.
  - The delta-method SD of a per-seed R̂ is 0.095 at G = 30 pp and 0.143 at G = 20 pp.
  - A claim counts as "supported" only if the **seed-level t-interval** also clears the threshold (§11). With df = 2, t = 4.303, so its half-width is **0.24 (G = 30) to 0.36 (G = 20)**.
  - Support therefore needs R̂ ≥ 1.04–1.16, and falsification needs R̂ ≤ 0.44–0.56.
  - The proposal's "0.15–0.2" is the undercovering bootstrap ([rliable](https://arxiv.org/abs/2108.13264), per LR/04, not re-verified).
  - At G3's tolerance of σ_d = 5 pp, the half-width grows to 0.40–0.59.
  - **The most likely outcome of the core test is "inconclusive".**
- **Ratio estimator.** CIs for R become unstable as the CI for G approaches 0, and G2 requires only G ≥ 15 pp, at the pilot.
- **Near-tautologies in the Holm family of 7.**
  - H2 (g₁ > g_late) follows from any concave S(k). It is also "supported" whenever S(12) < S(6), because g_late is then negative.
  - H6 (memory) is equivalent to R(k) > k/12, as the proposal itself notes.
  - Both spend α on almost nothing.
- **Fix.**
  - (i) Fit a **run-level mixed model with residual variance pooled over all runs of a stage**.
    - In my Tier 1 this gives df ≈ 11, t₀.₉₇₅ = 2.20, and an R half-width of 0.12 at G = 30.
    - Pre-register a heteroscedasticity check (Brown–Forsythe across k-groups) with a Welch fallback.
  - (ii) Test with the linear contrast L_k (F1).
  - (iii) Limit the confirmatory family to 4 tests (H1@3, H1@1, H3, H5). H2, H4 and H6′ become secondary.
  - (iv) Use gold-answer log-probability as a continuous co-primary. Continuous metrics have far higher signal-to-noise ([2406.10229](https://arxiv.org/abs/2406.10229), per LR/04, not re-verified).
  - (v) Trigger reserve seeds on pooled σ or on bimodality (M2).

### M2 (MAJOR). Small-model floors and capability phase transitions are not designed for

- **Zero-shot floors.**
  - Wang et al. omit recall for their 340M models: "340M models show insignificant recall capability due to their relatively small parameter count" (Table 2 note).
  - At 1.3B / 100B tokens, variable tracking scores 0.066 for the Transformer and 0.024 for GDN 3:1 (Table 5) ([2507.06457v2](https://arxiv.org/html/2507.06457v2) [read-section]).
  - At 41M parameters, *every* multi-hop and aggregation signal is therefore manufactured by synthetic exposure.
- **Phase transitions.**
  - In-context learning, carried by induction heads, appears abruptly: "roughly 2.5 to 5 billion tokens" in Olsson et al.'s large models and "approximately 1-3 billion tokens" in their small models, at roughly 1M tokens per step ([2209.11895](https://arxiv.org/abs/2209.11895) [read-section]).
  - Our 1.1B tokens, about 8.4K steps of 131K tokens, are the same order of magnitude [INF]. Some (config, seed) pairs may sit on the transition and give **bimodal** seed outcomes, for example 85% / 80% / 20%.
  - A 3-seed t-interval or percentile bootstrap summarises such outcomes badly.
  - Length generalisation is known to be seed-fragile ([2402.09371](https://arxiv.org/abs/2402.09371), per LR, abstract-only).
- **Consequence.** Endpoints or hybrids may sit at the floor or straddle a transition, which makes R uninformative or unstable.
- **Fix.**
  - (i) Run a **direct-supervision capability screen** (S2, §2.11) before any LM training. The exact architectures are trained directly on the task generators until they plateau. This shows each family is learnable and discriminating at this width, and anchors the ladders.
  - (ii) Use ladder averages, which are smoother than single cells.
  - (iii) Pre-register a bimodality rule: if seeds of one config span more than 40 pp at any primary ladder level, add seeds 3–4 for that config and report every seed.
  - (iv) Track dev learning curves at 6 milestones to locate transitions.
  - (v) Scope every claim to "capability elicited after identical exposure".

### M3 (MAJOR). Training-length and exposure-per-length artefacts confound H5, and "in-distribution at 32K" holds only nominally

- **(a) Effective context in Stage A is short.**
  - 59% of FineWeb-Edu tokens lie in documents of ≤ 2,048 GPT-2 tokens, and 76% in documents of ≤ 4,096 (05 §5.1 [MEAS on shard 000]).
  - With intra-document masking, **most Stage-A positions have less than about 2K tokens of usable context**.
- **(b) Stage B supplies long positions but few long dependencies.** It provides about 1,500 PG19 windows of 32K tokens (0.05B tokens) [INF arithmetic].
- **(c) Synthetic exposure is about 20× denser at short lengths.** Assuming log-uniform item lengths [INF arithmetic]:
  - 1% of Stage A at 256–4K ≈ **7,200 items**;
  - 5% of Stage B at 4K–32K ≈ **370 items** for all three families together.
  - At 16–32K each family therefore sees tens of items. "In-distribution" is true of positions, but hardly of the *tasks*.
- **(d) H5's length axis is confounded.** T is confounded with exposure per length, and with each architecture's length generalisation of a learned algorithm:
  - RoPE attention faces rarely seen large relative distances, given the left-skewed position frequencies ([STRING](https://arxiv.org/abs/2410.18745), per LR/03, not re-verified).
  - GDN learns its decay on short documents. For Mamba-2, the minimum training length needed to learn forgetting grows linearly with state size, "T_forget = 5.172·N_S − 4.469"; GDN is explicitly out of that paper's scope ([2410.07145](https://arxiv.org/html/2410.07145) [read-section]).
- **Consequence.** A positive H5 slope, or a gap at long T, may reflect exposure or curriculum rather than capability.
- **Fix.**
  1. Move synthetic exposure into an identical post-pretraining fine-tune with **equal item counts per length bucket** from 512 to 32K (§2.7).
  2. Add 15% PG19 4K windows to Stage A, so long coherent contexts appear throughout (ProLong's long + high-quality short rule, per LR).
  3. Log histograms of effective (same-document) context per stage.
  4. Make the **distance ladder at fixed T = 32K** the confirmatory H5 test, with a **filler-type manipulation**: natural vs low-information repetitive filler at the same distance. This separates interference from load from distance, and 2609.33093 predicts that filler type matters for GDN.
  5. Keep the length sweep at fixed distance as descriptive only.
  6. Use zero-shot copy-after-gap (LM-4) and repeat-token loss by distance (LM-3) as H5 co-primaries. They need no exposure.

### M4 (MAJOR). Fixed exposure measures learning speed as much as capability

- **Evidence.**
  - Qiao et al.: efficient attention "shapes [full attention's] optimization trajectory", and larger SWA windows delay the formation of retrieval heads ("Large-Window Laziness") ([2606.15378](https://arxiv.org/abs/2606.15378) [abstract-only + read-section]).
  - Olsson et al.: smeared keys make in-context learning form earlier (F2).
- **Consequence.** At a fixed exposure and token budget, R conflates "can" with "had learned by token 1.1B".
- **What the proposal has.** Learning curves, plus a fine-tune budget sweep on the endpoints only (O9). This is partial.
- **Fix.**
  - A **dose ladder** of fine-tune checkpoints at 250 / 1,000 / 4,000 steps for k ∈ {0, 1, 3, 12} (seed 0), with an asymptote fit.
  - Plateau capacity under direct supervision (S2).
  - A pre-registered "still closing" flag: if the k = 3 vs k = 12 gap shrinks by more than 3 pp between the last two dose points, report H1 as budget-conditional.
  - Optionally, extend the training budget 4× at k ∈ {3, 12} (Tier 3 add-on).

### M5 (MAJOR). The conclusions, especially H3, would probably not survive a different linear mechanism

- **The delta rule replaces stored values.** When the same key is written repeatedly, the stored value converges to v and does not accumulate. Counting (A1) is therefore mechanism-specific by construction [INF from the update rule in §4].
- **Wang et al.'s per-mechanism Table 7 shows curves of different shapes.** Values for pure / 24:1 / 12:1 / 6:1 / 3:1:

  | Mechanism | Pure | 24:1 | 12:1 | 6:1 | 3:1 | Shape |
  |---|---|---|---|---|---|---|
  | GDN | 0.356 | 0.341 | 0.352 | 0.425 | 0.436 | flat, then a jump |
  | HGRN2 | 0.203 | 0.303 | 0.368 | 0.434 | 0.395 | steep first step |
  | GLA | 0.144 | 0.293 | 0.288 | 0.330 | 0.332 | steep first step |
  | DeltaNet | 0.320 | 0.417 | 0.352 | 0.371 | 0.426 | non-monotone |

  Source: [2507.06457v2](https://arxiv.org/html/2507.06457v2) Table 7 [read-section].
- **Aggregation depends on the cheap layer too.** At 1.3B, CWE+FWE scores are 0.412 for HGRN2-6:1, 0.284 for the Transformer and 0.271 for GDN-3:1 (Table 5). Changing the cheap layer moves aggregation by more than 10 pp at a similar attention share.
- **What the proposal has.** A second mechanism only as option E5d, in a configuration that would not be geometry-matched.
- **Consequence.** Claims such as "aggregation needs less attention than retrieval", and the shape of the H1/H2 curve, are claims about GDN only.
- **Fix.**
  - A **GLA-twin** arm at k ∈ {0, 1, 3}, sharing k = 12 with the primary arm.
    - FLA's GLA with `use_short_conv=True` applies conv + SiLU to q/k/v and a swish-gated RMSNorm output (`fla/layers/gla.py` [code-inspected]). Its front and back end are therefore identical to GDN's.
    - It has 1,065,344 parameters at d = 512 (+0.8% vs GDN-A; meta-device count).
  - Make frequency comparison (relative) the primary aggregation task, with counting as a diagnostic.
  - Scope every H3 claim "for GDN" unless GLA replicates it.

### M6 (MAJOR). H6 is nearly a tautology as formulated, and measured efficiency does not transfer across scale

- **(a) The router line is a strawman.**
  - Memory is k·KV(T) + (12 − k)·S, where S is about three orders of magnitude smaller. So "H6 for memory" holds exactly when R(k) > k/12 (§2 notes), which is satisfied whenever the first attention layer helps at all.
  - No deployed system routes fractions of queries to two models in this way.
- **(b) Measured latency and throughput at d = 512 overstate attention's share of compute.**
  - Per full layer, attention-score FLOPs divided by dense FLOPs ≈ T/(13d) for MHA with d_ff = 3d.
  - At T = 32K that ratio is **4.9 at d = 512 but 0.6 at d = 4096** [INF arithmetic]. The position of the frontier shifts about 8×.
  - The same scale dependence applies to KV size relative to weights.
- **(c) The analytic GQA counterfactual (§10) assumes capability is unchanged when KV heads are cut.** That is untested.
- **(d) The framing is not new.**
  - Based traces a recall vs state-size "pareto frontier" ([2402.18668](https://arxiv.org/abs/2402.18668) [abstract-only]).
  - MAD reports compute-optimal and state-optimal hybridisation ratios (M10).
- **Fix.**
  - Replace H6 with **H6′: "Is KV memory worth more than recurrent-state memory?"**
    - Compare k = 1, which adds about 48–64 MiB of state at 32K, with a pure GDN of **3–4× state at equal parameters**, which adds only about 0.2–0.4 MiB per layer at any T (≈ 2–4.5 MiB for the whole model).
    - Concretely: GDN with 1 head, d_k = d/2, d_v = d, gate on, at k ∈ {0, 1}. At d = 512 it has 1,054,210 parameters (my count) vs 1,056,904 for config A.
  - Report analytic bytes and FLOPs as primary, with explicit projection formulas to larger scale (§2.9).
  - Report measured costs as secondary and scoped to this implementation.
  - Make no capability claims for GQA.

### M7 (MAJOR). The PE confound rests on a 1-seed pilot, and the all-full model may not be a valid ceiling at this scale

- **The pilot is thin.** It uses 1 seed per arm at 0.2B + 20M tokens. The scheme chosen for k = 12 is then imposed on the hybrids.
- **The PE × hybrid interaction is large in the closest-scale study.** NoPE on the full layers raised RULER-NIAH@16K from 49.58 to 67.81 in a 0.22B SWA-128 hybrid ([2606.15378v1](https://arxiv.org/html/2606.15378v1) Table 2 [read-section]).
- **The all-full model was not the ceiling there.**
  - In the same table, the all-full baseline, trained at 16K with RoPE base 10⁵ (Table 8), scored **35.95 on in-distribution NIAH@16K, below the hybrids' 49.58 and 67.81**.
  - It caught up only at 0.66B (67.14 vs 65.91 and 82.31).
  - At our scale, R > 1 could therefore mean "hybrids beat a weak reference" rather than "a small share recovers the full model".
- **The other PE is untested.** E5a, which runs the non-chosen PE, is optional.
- **Fix.**
  - Run 2 seeds per arm in the PE pilot.
  - Run the non-chosen PE at k ∈ {1, 12} in Tier 1 (≈ 8 GPU-h).
  - Pre-register R_best, defined relative to the best model in the family, as a secondary estimand.
  - Report the PE × k interaction as a result. The LR already claims it as novelty (G6).

### M8 (MAJOR). The placement arm cannot detect its own threshold and confounds three factors

- **Power.** It has 5 arms × 2 seeds and 10 Holm-adjusted pairwise contrasts against a 5 pp threshold.
  - With σ_run ≈ 3 pp and n = 2, SE(diff) ≈ 3 pp.
  - The first Holm step at α = 0.005 implies an MDE of **≈ 13 pp at 80% power** (Appendix B).
- **Confounds.** The arms mix:
  - depth with spacing (centred {3, 9} vs middle {5, 6});
  - depth with the number of layers between attention and the LM head;
  - and, if NoPE wins the pilot, depth with position availability, because attention at layer 0 receives no conv-derived positional or previous-token features in the current design.
- **Fix.**
  - A **k = 1 depth scan** at {0, 6, 11}, adding {3, 9} if budget allows, with 2–3 seeds each.
  - Two pre-specified contrasts: middle − mean(ends), and a linear depth trend.
  - Spacing, if wanted, as **one** contrast at k = 2 ({5, 6} vs {3, 9}).
  - Twin attention carries its own conv, which removes the layer-0 confound.

### M9 (MAJOR). The main prior is misread, and the predictions and choices built on it are unreliable

- **The LR's headline relies on mechanism averages.**
  - LR §0.3: "Retrieval is not flat in r … 0.256, 0.338, 0.340, 0.390, 0.397". These are averages over four mechanisms.
  - The **GDN row shows no gain at 24:1 or 12:1** (0.341 and 0.352 vs 0.356 for pure GDN). The gain appears only by 6:1 (0.425).
  - Our k = 1 of 12 is 11:1. For GDN, the closest published prior therefore predicts **R_ret(1) ≈ 0** [EST for Wang's setup: single runs, evaluation length unstated].
- **Wang's numbers are single runs and partly unexplained.**
  - DeltaNet scores 0.417 at 24:1 but 0.352 at 12:1, which suggests single-run noise of several points. Adjacent-ratio differences in Wang are within noise [INF].
  - Table 5's "Avg" is not the unweighted mean of its listed columns: the Transformer's columns average 0.370 against a reported 0.423 [read-section; INF arithmetic]. The weighting is undocumented.
  - The Transformer scores 0.559 on Singlevalue vs 0.875 for GDN-3:1. RULER lengths are unstated, and the paper mentions a 4,096-token window only in its Limitations, so extrapolation may drive the Transformer's deficit ([2507.06457v2](https://arxiv.org/html/2507.06457v2) [read-section]).
- **Several proposal elements rest on these numbers:**
  - the prediction R_ret(3) ≈ 0.9–1.1 (§2);
  - the claim that strong H6 is "plausible" because 0.436 > 0.423;
  - the choice of k = 2 for placement because 3:1 "may sit near ceiling" (LR D3).
- **Fix.**
  - Restate the priors. For GDN the prior is threshold-like, with no gain from one layer in about 12.
  - Pre-register H1-min with a pessimistic prior.
  - Never use Wang's numbers for power or ceiling planning.

### M10 (MAJOR, framing). Novelty is overstated, and close prior work is missing

- **MAD** ([2403.17844](https://arxiv.org/html/2403.17844) [read-section]):
  - "compute-optimal hybridization ratio is 25% attention across all IsoFLOP groups", shifting to 10% when state-optimality is considered (Finding 7);
  - striped hybrids beat both 0% and 100% attention;
  - MAD synthetic scores, including in-context recall, correlate with compute-optimal perplexity (Finding 9).
  - This is a controlled "how much attention" study with synthetic recall tasks and a state axis, at 70M–7B. The LR lists MAD only as abstract-only and omits these findings.
- **Based** frames recall vs state size as a Pareto frontier (M6). H6's framing is not new.
- **Wang et al., Appendix C**: "Single-Key, Multi-key, and QA sub tasks are noticeably affected by a changing ratio", while "CWE and FWE does not seem to correlate with the hybrid ratio" (figure only; 1.3B; single run) [read-section].
  - This is a *direct* prior for H3. The LR's G3 ("existing evidence is indirect") is overstated.
- **Qiao et al.** (15–477M non-embedding parameters): long-range information is "mainly introduced and processed by full attention", and GDN "store[s] little long-range information" in its state at inference in hybrids [read-section].
- **2026 work absent from the LR:**
  - hybrids "rely substantially more on attention than on the recurrent state" ([2610.06750](https://arxiv.org/abs/2610.06750) [abstract-only]);
  - load dominates elapsed context for GLA/GDN recall, and in hybrids recall memory "shifts mainly to the full-attention KV state" ([2609.33093](https://arxiv.org/abs/2609.33093) [abstract-only]);
  - conventional hybrid blocks need Ω(√n) capacity for two-function composition, with a strict hierarchy in retrieval depth ([2610.25250](https://arxiv.org/abs/2610.25250) [abstract-only]). This is a theoretical prior for H3.
- **Fix: reposition the contribution as follows.**
  - First seed-replicated, difficulty- and distance-resolved, in-distribution per-family curves over the full 0 → 1 range.
  - A mechanism-isolating twin design, a second linear mechanism and a state-size control.
  - Scale of 20–60M parameters.
  - Explicit tests of three mechanistic predictions:
    - one-layer induction via token mixing;
    - load vs distance;
    - composition depth vs number of attention layers.

### M11 (MAJOR, claim scope). Count and proportion are not separable

- **Problem.**
  - At fixed L = 12, k and r are collinear.
  - The most relevant theory concerns *counts*: one-layer induction with token mixing, composition depth ≥ hops, and Eitan et al.'s hierarchy in retrieval depth.
  - F24 (optional) changes width, state size and head count at the same time.
- **Consequence.** Statements about "proportion" (README H1; "11:1 / 5:1 / 3:1") are not identified.
- **Fix.**
  - State confirmatory claims in k at L = 12, e.g. "one (three) full layers of twelve".
  - Treat F24 as exploratory only, labelled as confounded by width.

### Minor issues

| # | Issue | Fix |
|---|---|---|
| m1 | **Reproduction gate.** In the Zoology target figure, "Attention" and "Sliding Window" share the **same black marker** (`zoology/analysis/paper/zoology_paper/zoology_results.png` @`1ad20d1` [code-inspected]), so attention targets "read by eye" cannot be verified. MQAR validates the mixers but not the LM training stack. | Use only re-run numbers for criteria (a) and (b). Add an **LM-stack equivalence gate**: our all-GDN model vs FLA's `GatedDeltaNetForCausalLM`, and our plain all-attention model vs a reference implementation, on identical 0.1B tokens, with \|Δ val loss\| ≤ 0.02 nats. |
| m2 | **Shared LR** is chosen on 4K val loss at 0.2B tokens. Gated layers tolerate larger LRs (2505.06708), so the minimax-regret LR may favour GDN-heavy configs. | The twin design gates both mixers, which removes most of this. Add an optional retrieval-level LR check at k ∈ {3, 12} at η*/√2 and √2·η* (1 seed). |
| m3 | **Mechanistic claims** (O1(a), O6: "a few attention layers implement retrieval") are behavioural evidence only. | Add inference-only interventions: windowing the full layers vs resetting GDN state every 1K tokens (as in Qiao), mean-ablation of each attention layer, and attention mass on the target. |
| m4 | **"Information integration"** is operationalised as literal-token counting or frequency. | Scope the claims accordingly. Make frequency comparison primary and counting diagnostic. |
| m5 | **GQA counterfactual** (§10) is analytic only. | Report bytes only, with no capability claim. |
| m6 | **QK-RMSNorm** at 32K (U4). | In the pilot, log attention entropy and target mass at 32K for k = 12. |
| m7 | Option (i) puts 1% synthetic data into Stage A, so **LM-tier metrics are no longer natural-only**. | Option (ii) keeps the base model clean. |
| m8 | **Initialisation.** A global N(0, 0.02) re-init must not overwrite GDN's `A_log` and `dt_bias`. GDN's gated output norm bounds its initial output; plain attention's does not, so early residual dynamics differ. | Add unit tests. The twin design removes the asymmetry. |
| m9 | **The "long-running agents" discussion** in README/§0 cannot be supported by synthetic retrieval in 50M models. | Restrict it to mechanism-level statements. |
| m10 | **Answers to the proposal's questions to reviewers (§17).** | Config A: yes, as a twin pair, with GDN-E/D as the state-size arm. QK-norm: yes. Placement: a k = 1 depth scan, not k = 2 / k = 3 arms. H1 thresholds: keep 0.8 but test it via L_k; make the 5 pp margin secondary. L = 12: fine for counts, but drop "proportion" claims. E5a/E5b: E5a into Tier 1; E5b superseded by the twin design plus the bridge arm. |

---

## 2. Independent experimental design

### 2.1 Principles

1. **Isolate the operator.** Full and linear mixers are twins, so the only manipulated difference is the memory operator, plus the position encoding that softmax requires. The usual literature design is kept as a bridge.
2. **Measure capability as a function of difficulty, not at a tuned point.** Ladders are anchored in theory: load relative to state capacity, distance, hops and ratio.
3. **Pit hypotheses against named competitors.** Each major rival explanation gets one arm or one manipulation (§4).
4. **Spend runs where inference is weakest.** Discovery runs at d = 384, where twice as many runs fit the budget; confirmation is a fresh-seed replication at d = 512.
5. **Infer at the level of runs.** Use pooled-variance contrasts with no ratio estimators, a small confirmatory family, and continuous co-primaries.

### 2.2 Estimands and hypotheses

All quantities are family scores S_f = accuracy averaged over the pre-registered ladder (§2.8), at in-distribution lengths, after identical exposure, unless stated otherwise.

| ID | Estimand | Test (one-sided unless stated) | Tier / role |
|---|---|---|---|
| **H1** | Retrieval recovery at k = 3: L₃ = S₃ − 0.2·S₀ − 0.8·S₁₂ (≥ 0 ⇔ R ≥ 0.8) | Supported if lower 95% CI(L₃) > 0 **and** S₁₂ − S₀ > 0. Falsified if upper 95% CI(L₃) < 0 | T1 confirmatory |
| **H1-min** | Same at k = 1 (L₁) | Same rule | T1 confirmatory |
| **H3** | Task dependence: R_ret(3) − R_hop(3) > 0, at difficulty-equated levels | Parametric bootstrap over seeds, cross-checked by the family × k interaction in the run-level mixed model | T1 confirmatory |
| **H5** | Distance dependence: slope of Δ₀(d) = S₁₂(d) − S₀(d) on log₂ d at fixed T = 32K (retrieval) | Positive slope (config × log₂d interaction) | T1 confirmatory |
| H1-robust | R ≥ 0.8 at every informative ladder level | Reported per level | T1 secondary |
| R_best | Recovery relative to the best model in the family | Same as H1 | T1 secondary |
| H2 | Curve shape | Isotonic and saturating fits; k80 with CI | T1 descriptive |
| H4′ | Depth at k = 1: S(6) − ½[S(0) + S(11)]; linear depth trend | Holm over 2 tests | T2 secondary confirmatory |
| H6′ | KV vs recurrent memory: share of the k = 0 → k = 1 gap closed by GDN-E (3–4× state, equal parameters) | Point estimate and CI; ≥ 0.5 counts as "state suffices" | T2 secondary confirmatory |
| H-mech | Generality: L₁ and L₃ for GLA-twin vs GDN-twin | Difference CI | T2 secondary confirmatory |
| H-conv | Conv dependence of one-layer sufficiency: [S₁ − S₀]_conv − [S₁ − S₀]_no-conv | Difference CI | T2 exploratory |
| Rep | H1, H1-min, H3, H5 at d = 512 with fresh seeds | Same rules, separate Holm family | T3 replication |

### 2.3 Architecture

- **Block (all models):** `h = x + Mix(RMSNorm(x)); y = h + SwiGLU(RMSNorm(h))`.
  - The same SwiGLU in every block, tied embeddings, V = 16,384 (the proposal's digit-isolating BPE tokenizer).
  - Pre-norm, no biases, z-loss, bf16 autocast with fp32 master weights, fp32 linear-attention state.
  - Placement for the composition arms is the proposal's centred-uniform rule, with layer 0 linear for k < 12.

| Scale | L | d | d_ff | Twin / GDN / GLA heads (d_k / d_v) | Plain-MHA heads | Non-emb / total params |
|---|---|---|---|---|---|---|
| **D384** (discovery; Tiers 1–2) | 12 | 384 | 1024 | 3 × (64 / 128) | 6 × 64 | ≈ 21.3M / ≈ 27.6M |
| **D512** (replication; Tier 3) | 12 | 512 | 1536 | 4 × (64 / 128) | 8 × 64 | ≈ 41.0M / ≈ 49.4M |

| Mixer | Definition | Params at d = 512 | State or cache per layer (d = 512) |
|---|---|---|---|
| **F_twin** (primary full) | q, k: d→d/2; v, g: d→d; ShortConv(4) + SiLU on q/k/v (as FLA); per-head QK-RMSNorm; PE from the pilot; causal softmax over the KV cache; per-head RMSNorm(o) ⊙ swish(g); W_o. Implemented with standard SDPA/varlen by duplicating each QK head over two 64-dim V halves | 1,052,928 (analytic) | K 256 + V 512 = **1.5 KiB/token** bf16 (48 MiB at 32K) + 16 KiB conv cache |
| **F_plain** (bridge) | MHA 8 × 64, QK-RMSNorm, no conv, no gate | 1,048,704 | 2 KiB/token (64 MiB at 32K) |
| **Λ_GDN** (primary linear) | `GatedDeltaNet(hidden_size=d, head_dim=64, num_heads=d//128, expand_v=2, use_gate=True, use_short_conv=True, conv_size=4)` | 1,056,904 (meta) | 128 KiB fp32 + 16 KiB conv cache |
| **Λ_GDN-E** (state ×3–4) | Same, but `num_heads=1, head_dim=d//2` | 1,054,210 (meta) | 512 KiB at d = 512 (4×); 288 KiB at d = 384 (3×, head_dim 192) |
| **Λ_GDN-nc** | Λ_GDN with `use_short_conv=False` | 1,052,808 (meta) | 128 KiB |
| **Λ_GLA** (second mechanism) | `GatedLinearAttention(hidden_size=d, num_heads=d//128, expand_k=0.5, expand_v=1.0, use_short_conv=True, conv_size=4, use_output_gate=True)` | 1,065,344 (meta) | 128 KiB + conv |

- **Spread.** All mixers are within 1.6% of each other. Twin vs GDN is −0.38% per mixer, i.e. at most 0.12% of total non-embedding parameters across k [INF from the counts].
- **Kernel feasibility.** FLA asserts K ≤ 256 (`fla/ops/common/chunk_delta_h.py` L716 [code-inspected]), so GDN-E is allowed by the code. It has not been run on sm_120, which has 99 KB of shared memory per SM (per 05 §4.3). G0 must test it; the fallback is config D (2 heads, 2× state).
- **Position encoding.**
  - Linear mixers have none.
  - Full mixers use the pilot winner of {NoPE, RoPE θ = 5×10⁵}, with 2 seeds per arm at k = 12. Rule: maximise the all-full model's dev ladder average for retrieval at 32K; break ties on val loss at 32K. θ is constant across stages.
  - The other scheme runs at k ∈ {1, 12} in Tier 1.

### 2.4 Runs

| Arm | Full / linear mixer | k (placement) | Seeds | Scale | Tier | Runs |
|---|---|---|---|---|---|---|
| P (composition) | F_twin / Λ_GDN | 0, 1, 3, 12 | 3 | D384 | 1 | 12 |
| P (curve) | F_twin / Λ_GDN | 2, 6 | 2 | D384 | 1 | 4 |
| Ref (reference check) | F_plain / — | 12 | 2 | D384 | 1 | 2 |
| PE-alt | F_twin / Λ_GDN, other PE | 1, 12 | 2 | D384 | 1 | 4 |
| G (mechanism) | F_twin / Λ_GLA | 0, 1, 3 (k = 12 shared) | 2 | D384 | 2 | 6 |
| S (state) | F_twin / Λ_GDN-E | 0, 1 (k = 12 shared) | 2 | D384 | 2 | 4 |
| B (bridge) | F_plain / Λ_GDN | 1, 3 (k = 0 shared with P; k = 12 = Ref) | 2 | D384 | 2 | 4 |
| C (conv dependence) | F_plain / Λ_GDN-nc | 0, 1 | 2 | D384 | 2 | 4 |
| D (depth scan) | F_twin / Λ_GDN | k = 1 at {0}, {11} ({6} from P) | 2 | D384 | 2 | 4 |
| Rep | F_twin / Λ_GDN | 0, 1, 3, 12 | 3 (fresh indices 10–12) | D512 | 3 | 12 |
| Optional add-ons | — | k = 2 at {5, 6}; LR ±√2 at k ∈ {3, 12}; 4× tokens at k ∈ {3, 12}; reserve seeds | 1–2 | D384 | 3 | ≤ 16 |

### 2.5 Data, tokenizer and curriculum

These are identical for every run of a given scale.

| Stage | Length | Tokens (D384 / D512) | Mix | Masking / LR |
|---|---|---|---|---|
| A | 4,096 | 0.6B / 1.0B | 85% FineWeb-Edu sample-10BT (shards ≥ 001, revision pinned); 15% PG19-train 4K windows | Intra-document masking in both mixers (`varlen_attn` + `cu_seqlens`); RoPE positions reset per document; warm-up 500 steps, then constant |
| B | 32,768 | 60M / 100M | 50% PG19-train 32K windows (book-bounded); 50% FineWeb-Edu packed | As A; WSD linear decay to 0 spanning the last 25% of A and all of B |
| C (exposure fine-tune) | Mixed 512–32K | 1,000 steps × 64K = 65.5M tokens (4,000 steps at anchors) | 90% synthetic, 10% natural replay (§2.7) | Constant LR from the pilot ({1e-4, 3e-4}, endpoint-only rule) |

- Held-out filler, PG19-test and the FineWeb shard 000 split follow the proposal.
- The data order is a pure function of (data seed, step) and is shared across configs. All data is pre-tokenised once.
- **Logged per stage:** the effective-context histogram (same-document tokens available at each position), the truncation fraction, and the count of synthetic items in each length bucket.

### 2.6 Optimisation and LR protocol

- **Optimiser.** AdamW (0.9, 0.95, ε 1e-8), weight decay 0.1, with none on norm gains, `A_log` or `dt_bias`. Gradient clip 1.0, z-loss 1e-4, batch 131,072 tokens, chunked cross-entropy at ≥ 16K.
- **LR sweep at D384.**
  - Grid {1.4, 2.0, 2.8}×10⁻³ at k ∈ {0, 3, 12}, each run 0.15B tokens. Extend the grid at an edge if the optimum lands there.
  - A shared LR is accepted by the proposal's minimax-regret rule (regret ≤ 0.02 nats on 4K val loss). Otherwise use per-config LRs with log-linear interpolation in k.
- **LR at D512 (Tier 3).** Re-sweep three LRs at k ∈ {0, 12}.
- **Selection.** LRs are never selected on long-context task metrics.

### 2.7 Synthetic exposure: an identical fine-tune with a dose ladder

- **Why a fine-tune and not mixing into pretraining (option i).** A fine-tune gives:
  - (a) a clean base model for the zero-shot tier;
  - (b) the ability to balance exposure across lengths **by item count** without bloating pretraining;
  - (c) a dose–response that separates learning speed from capacity (M4).
- **Fallback.** Use option (i) only if the fine-tune fails G-P discriminability on the endpoints.
- **Balancing.** Use 7 length buckets {512, 1K, 2K, 4K, 8K, 16K, 32K} with equal item counts, and equal counts across the three families. 1,000 steps then give about **1,030 items per bucket**, or about 340 per family per bucket [INF arithmetic: 65.5M / 63.5K].
- **Generators.**
  - Training uses the ladder distributions with a wider range: load up to 2× the top evaluation level, hops 1–5, ratios 5 → 1.1.
  - Train vs eval key lists A/B, templates T1–T3 vs T1 + T4, hashed (key, value, template) disjointness and held-out filler all follow the proposal (§9.1).
- **Dose.**
  - Every run is evaluated after 250 and 1,000 steps.
  - The anchors k ∈ {0, 1, 3, 12} (seed 0) continue to 4,000 steps.
  - Fit an asymptote (a 3-parameter exponential in log steps). The "still closing" flag of M4 applies.
- **Caution.** Fine-tuning can disturb the long-range routing of hybrids ([2606.11052](https://arxiv.org/abs/2606.11052) [abstract-only], CoT-SFT). The 10% natural replay, plus a check of zero-shot LM-4 before vs after the fine-tune on the anchors, guards against this.

### 2.8 Evaluation

The shared mechanics are the proposal's §9.1: cloze format, exactly T tokens in token space, teacher-forced exact match (EM) plus gold log-prob plus rank metrics, same-format distractors, query after context, Q-branching from a state snapshot, frozen hashed items, byte-identical items for every model.

| Family | Ladder (primary in **bold**) | Fixed | Contexts × queries per level | Purpose |
|---|---|---|---|---|
| Retrieval | **R-load:** N facts ∈ {8, 32, 128, 512} (0.04–2.7× of H·d_k = 192 at D384; 0.03–2× at D512) | T = 8K, natural filler, facts like "The code for {adj noun} is {d d d d}." | 400 × 4 | Load relative to linear-state capacity (H1 / H1-min) |
| Retrieval | **R-dist:** target distance d ∈ {256, 2K, 8K, 30K}, four targets per context (one query each) + 12 distractors; **filler ∈ {natural, repetitive}** | T = 32K | 400 per filler type (each context probes all 4 distances) | H5: distance vs load/interference |
| Retrieval | R-T: T ∈ {1K, 2K, 4K, 8K, 16K, 32K} | Target distance 512, 3 distractors | 400 × 4 | Descriptive (README length grid) |
| Retrieval | R-update: U ∈ {2, 4} overwrites; latest value; stale rate | T = 8K | 400 | Delta-rule overwrite diagnostic |
| Multi-hop | **M-hop:** h ∈ {1, 2, 3, 4} | T = 8K; 2 distractor chains; shuffled link order; links spread | 400 × 1 | H3 (composition depth) |
| Multi-hop | M-order: forward vs reversed at h = 2 | T = 8K | 200 | Link order and recurrence (JRT) |
| Aggregation | **A-freq:** frequency ratio ρ ∈ {4, 2, 1.5, 1.25} | T = 8K numbered word list | 400 × 4 probe pairs | Relative aggregation (mechanism-neutral) |
| Aggregation | A-count: c ∈ 1…9 | T = 8K | 400 | Mechanism-specific diagnostic |
| Controls | C0 (no filler), C2 (oracle position) | One level per primary ladder | 200 | Task logic vs long-input handling |
| Zero-shot tier (base model, before C) | LM-1 per-position loss to 32K; LM-2 val loss at 4K; LM-3 repeat-token loss by distance; **LM-4** copy-after-gap with random-token passages, gaps {0, 512, 2K, 8K, 30K} | — | 300 docs + 100 PG19-test books; 200 per gap | Exposure-free long-range use (co-primary for H5) |
| Diagnostics (inference only, exploratory) | Window the full layers to 1K vs reset GDN state every 1K; mean-ablate each attention layer; attention mass on the target; attention entropy at 32K | R-dist items | 200 | Which pathway carries long-range information (m3; 2610.06750) |

- **Informative level.** k = 12 scores between 20% and 90%, and Ĝ ≥ 15 pp.
- **Difficulty-equated comparison for H3.** Levels whose k = 12 accuracy lies within ±10 pp of each other across families.
- **Evaluation cost.** About 145M tokens per model, ≈ 0.1–0.25 GPU-h at D384 [INF].

### 2.9 Efficiency measurement

- **Primary: analytic.**
  - Memory: M(k, T) = k·T·H·(d_qk + d_v)·b_kv + (L − k)·(H·d_k·d_v·4 B + conv cache).
    - Twin at d = 512: 1.5 KiB per token per layer. Plain MHA: 2 KiB.
  - Forward FLOPs per token: dense ≈ 2·N_non-emb + 2·d·V, plus T·H·(d_qk + d_v) per full layer (causal average).
  - Projection to larger scale (twin geometry, d_ff = 3d): attention-to-dense FLOP ratio per full layer ≈ T/(17·d), and KV bytes per token relative to parameters. Tabulated for d ∈ {384, 512, 2048, 4096} and labelled [INF].
- **Secondary: measured, scoped to this implementation.**
  - As in the proposal §10: idle GPU (G-cost), CUDA-graph decode, batches {1, 8, 32, 128}, contexts 1K–32K, interleaved configs, two sessions.
  - Plus the largest batch that fits within 8 GB.
  - The twin's duplicated-head implementation costs MHA FLOPs; report both the analytic twin cost (768·T) and the measured cost.
- **No capability claims for GQA.**

### 2.10 Seeds and statistics

- **Seeds and blocking.**
  - Tier 1–2 seed indices are 0, 1, 2; reserves are 3 and 4; Tier 3 uses fresh indices 10–12.
  - Data order and the initialisation of shared modules are blocked by seed index (as in the proposal).
  - No seed is ever dropped. Diverged runs are reported.
- **Unit of inference.** The run.
  - Each run gives a ladder-averaged family score S_rf and a gold log-prob analogue, with a within-run item bootstrap SE (contexts resampled, stratified by level).
- **Model.** S_rf = μ_config,f + b_seed-index + e_r.
  - Residual variance is pooled within the stage, per family.
  - Brown–Forsythe test across k-groups; if p < 0.05, use Welch–Satterthwaite.
  - Contrasts: L_k for H1 / H1-min; config × log₂d for H5; family × k for H3.
  - Tier 1 df ≈ 22 runs − 9 configs − 2 blocks ≈ 11.
- **Multiplicity.**
  - Tier 1 confirmatory family {H1, H1-min, H3, H5}: Holm at α_FW = 0.05, one-sided in the pre-registered direction.
  - Tier 2 secondary family {H4′ ×2, H6′, H-mech}: Holm.
  - Tier 3 is a replication family.
  - Everything else is exploratory, with BH-FDR at q = 0.10.
- **Equivalence margins (TOST).** 5 pp; R = 0.8 via L_k; 0.01 nats; 10% on cost.
- **Sensitivity analyses.**
  - Hierarchical bootstrap (seed → context).
  - Every seed plotted.
  - Bouthillier's P(A > B) ([2103.03098](https://arxiv.org/abs/2103.03098), per LR/04, not re-verified).
  - The same tests on the log-prob co-primary.
- **Power** [INF; Appendix B].
  - With σ_run = 3 pp and n = 3, SE(L_k) = 3·√(1.68/3) = 2.24 pp.
  - The one-sided 95% margin at df = 11 is 4.0 pp, i.e. 0.13 in R at G = 30 pp.
  - Compare the proposal's binding interval of 0.24.

### 2.11 Stages, gates and budget

**Throughput assumptions** [INF, from the proposal's contended measurements; replaced at G0; "idle" = ÷1.6, which is an assumption]:
- D512 run: Stage A 3.47 h + Stage B ≈ 0.55 h + fine-tune ≈ 0.4 h ≈ **4.4 h** conservative.
- D384 run: about 1.35× the throughput of D512, giving ≈ **2.05 h** conservative (1.54 + 0.25 + 0.25 h).
- Direct-supervision run: ≈ 0.25 h.

| Stage | Content | Gate (pass criterion → if fail) | GPU-h conservative |
|---|---|---|---|
| S0 | Infrastructure (proposal G0a–f), plus: twin duplicated-head attention equals the reference d_v = 128 math (fp32, max abs error ≤ 1e-5); KV bytes = 1.5 KiB per token per layer; GDN-E (K = 192/256) and GLA forward/backward on sm_120; parameter counts as in §2.3; GDN `A_log`/`dt_bias` survive the global init | **G0** → fix. If GDN-E fails, fall back to config D | 3.0 |
| S1 | Zoology MQAR re-run (attention + GDN, 2 d_model each, best 2 LRs; re-run numbers only); LM-stack equivalence on 0.1B tokens at D384 | **G-RP:** MQAR (a)/(b) as in the proposal; \|Δ val loss\| ≤ 0.02 nats → debug | 3.2 |
| S2 | **Direct-supervision screen** at D384: P mixers k ∈ {0, 1, 2, 3, 6, 12} × 2 seeds; plus GLA {0, 3}, GDN-E {0}, plain {1, 12}, no-conv {1} (1 seed each). Trained on the generators until dev plateau (≤ 100M tokens) | **G-S:** for each family, k = 12 ≥ 80% at ≥ 1 level where k = 0 ≤ 50% → redesign the tasks. If k = 12 fails broadly, move discovery to D512 (pre-registered) | 4.5 |
| S3 | LR sweep (9 × 0.15B); PE pilot (k = 12 × 2 PE × 2 seeds, 0.3B + 30M at 32K); fine-tune LR pilot on the endpoints | **G-P:** LR regret rule; PE rule (§2.3); endpoints discriminable after the fine-tune (G ≥ 15 pp on ≥ 2 levels per family) | 7.9 |
| — | **G-F1 freeze:** protocol, items, analysis code and predictions hashed | No launch without it | — |
| S4 | Tier 1 runs (22), with dev milestones at 50, 100, 200, 400 and 600M tokens. **G-I** at 0.3B on P k ∈ {0, 12}, seeds 0–1 (250-step fine-tune on the interim checkpoints): may add one pre-registered harder level per family, or demote a family whose endpoints differ by < 5 pp at every level. Endpoints only | **G-V:** pooled σ_run ≤ 4 pp and no bimodality → otherwise add reserve seeds for the affected configs (≤ 8 runs) | 45.1 |
| S4e | Dose extension to 4,000 steps (4 anchors); evaluation; cost benchmarks | **G-cost** (idle GPU) | 14.5 |
| **Tier 1 total** | | | **≈ 78** (≈ 50 idle) |
| S5 | Tier 2 runs (22): G, S, B, C, D; evaluation | — | **≈ 49** (≈ 31 idle) |
| — | **G-F3 freeze** with directional predictions for D512 | — | — |
| S6 | Tier 3: D512 LR re-sweep (6 × 0.2B), Rep (12 runs), evaluation | Same gates | **≈ 62** (≈ 39 idle) |
| Optional | k = 2 adjacent (2 runs, 4 h); LR ±√2 (4 runs, 8 h); 4× tokens at k ∈ {3, 12} (2 runs, 13 h); reserve seeds (≤ 16 h) | — | ≤ 41 |

**Totals:** Tier 1 ≈ 78 h; Tiers 1+2 ≈ 127 h (≈ 81 idle); Tiers 1–3 ≈ 189 h (≈ 120 idle).

**Comparison with the proposal.** Its Tier C is 152 h conservative / 97 h idle. My Tiers 1+2 cost less and include the twin, bridge, PE, mechanism, state, conv and depth controls. If the user insists on 49M as the primary scale, Tier 1 can run at D512 without the k ∈ {2, 6} points, for ≈ 85 h.

### 2.12 How outcomes map to conclusions

| Observation | Conclusion licensed | Check that must also pass |
|---|---|---|
| k = 0 collapses at R-load ≥ 128 while k = 1 ≈ k = 12 at every level | "One full layer of twelve suffices for literal retrieval up to 2.7× linear-state load, for twin mixers at about 28M parameters" | C arm: if conv-free k = 1 fails, sufficiency depends on token mixing (Olsson mechanism) |
| k = 1 ≈ k = 0 and k = 3 ≈ k = 12 | A threshold, not diminishing returns; consistent with Wang's GDN row | D arm: depth of the single layer; dose ladder (still closing?) |
| R_hop(k) rises in steps with k ≥ h | Composition depth needs multiple attention layers; consistent with 2610.25250 | Difficulty-equated levels; GLA arm |
| Δ₀ grows with d at fixed T, and filler type does not matter | Distance or decay | PE-alt; LM-4 |
| Δ₀ depends on filler type at fixed d | Load / interference (2609.33093) | — |
| GDN-E closes ≥ 50% of the k = 0 → k = 1 gap | State capacity, not softmax, limits retrieval at that load; the memory frontier favours a bigger state | GLA-E is not run, so the claim is scoped to GDN |
| GLA reproduces the GDN curve | Curve shape is not delta-rule-specific | — |
| A hybrid beats k = 12 | Report R_best; examine twin vs plain and PE-alt before claiming complementarity | Ref, PE-alt, dose ladder |

---

## 3. Minimum experiment that would falsify H1

**H1 as falsified here.** For mixers that differ only in the memory operator (twin pair, §2.3), at in-distribution lengths after identical exposure, a model with k = 3 of 12 full layers recovers at least 80% of the all-full model's advantage over the all-linear model on literal retrieval. "H1-robust" requires this at every informative memory-load level.

A counterexample at any load level falsifies H1-robust. A counterexample on the ladder average falsifies H1-average.

**Level A: falsification in principle (cheapest, ≈ 4 GPU-h including a reduced G0).**
- **Runs.** D384 twin models trained *directly* on the R-load generator (T = 8K, N up to 1,024) until dev accuracy plateaus (≤ 100M tokens): k ∈ {0, 3, 12} × 3 seeds, ≈ 2.3 GPU-h.
- **Decision.** At any level where k = 12 ≥ 80% and k = 0 ≤ 50%, if the upper 95% CI of L₃ is < 0, then H1 fails *as a statement about capacity at this width*.
- **Reasoning.** Language-model pretraining cannot add capacity that direct supervision cannot elicit [INF; an assumption, stated as such].
- **If H1 survives Level A,** it remains untested in the LM regime, which is Level B.

**Level B: decisive falsification in the LM regime (≈ 31 GPU-h conservative, ≈ 19 idle, including the reference-guard run).**
- **Runs.** D384, F_twin / Λ_GDN, k ∈ {0, 3, 12} × 3 seeds. Each run: Stage A 0.6B + Stage B 60M + a 1,000-step exposure fine-tune. 9 runs ≈ 18.5 h.
- **Prerequisites:**
  - reduced G0 (3 h) and the LM-stack equivalence gate (1.2 h);
  - a mini LR sweep with 3 LRs at k ∈ {0, 12} (2.3 h);
  - a PE pilot with 2 arms at k = 12 and 1 seed (1.9 h).
- **Evaluation.** R-load ladder at T = 8K, plus the R-dist ladder at T = 32K (≈ 2 h).
- **Decision rule.**
  - H1 is **falsified** if, at any pre-registered informative R-load level, the upper one-sided 95% bound of L₃ is < 0 (run-level pooled-variance t, df = 4 with seed blocks), and the hierarchical bootstrap agrees.
  - H1-average is falsified if the same holds for the ladder average.
- **Validity guards (all pre-registered):**
  - (i) Endpoint informativeness: Ĝ ≥ 15 pp, and k = 12 between 20% and 90% at that level.
  - (ii) Training-budget guard: the k = 3 vs k = 12 gap must not shrink by more than 3 pp between the 250-step and 1,000-step fine-tune checkpoints. Otherwise the outcome is "inconclusive: learning speed", not "falsified".
  - (iii) Reference guard: a plain-MHA k = 12 run (1 seed, +2 h) must not exceed the twin k = 12 run by more than 5 pp. Otherwise test against the stronger reference.
- **Why k = 3.** Failure at k = 3 falsifies H1 for every k ≤ 3, *assuming S is monotone in k*. Level A's k ∈ {0, 1, 2, 3, 6, 12} screen (S2) gives cheap evidence on monotonicity.
- **Scope.** Falsification at D384 falsifies the universal "20–60M" claim but not a claim specific to 49M. The proposal's own minimum experiment at d = 512 costs ≈ 66 h.

---

## 4. Competing explanations and the discriminating experiment for each

| Claim or outcome | Competing explanation | Discriminating experiment (arm, cost) | Prediction that separates them |
|---|---|---|---|
| H1 true (k small ≈ k = 12) | Ceiling or tuned easiness | R-load ladder; R per level (evaluation only) | Ceiling: R → 1 only at low load; real: R ≈ 1 at loads above GDN capacity |
| | Task fits inside the linear state | Levels at 0.67× and 2.7× of H·d_k | k = 0 must fail above capacity for "attention needed" to be meaningful |
| | Conv-provided previous-token features let one attention layer act as an induction head (Olsson) | C arm (Λ_GDN-nc + F_plain at k ∈ {0, 1}) vs B arm at k = 1 | If one-layer sufficiency is conv-mediated, [S₁ − S₀] collapses without conv |
| | Weak all-full reference (no conv/gate, PE, small-scale RoPE deficit as in Qiao) | Twin vs plain at k = 12 (Ref); PE-alt at k = 12; R_best | If the reference is weak, R_best < R |
| | Lexical shortcut | Same-format distractors; binding-error rate; C0/C2 | Shortcut: high C0 and C2 accuracy but near-chance binding at depth |
| H1 false (k = 3 ≪ k = 12) | Undertraining or slower circuit formation | Dose ladder; dev milestones; S2 plateau; 4× tokens (optional) | Learning speed: the gap shrinks with dose and is absent in S2 |
| | LR suits the endpoints | LR ±√2 at k ∈ {3, 12} (optional, 8 h) | The gap changes sign or shrinks below 5 pp at another LR |
| | RoPE large distances undertrained in the few attention layers | PE-alt at k ∈ {1, 12} | Under NoPE the gap shrinks mainly at large d |
| | Centred placement suboptimal | D arm (k = 1 depth scan) | Best-depth k = 1 recovers more than the centred position |
| H5: gap grows with T or d | Distance or decay vs load/interference vs softmax dilution | R-dist at fixed T with natural vs repetitive filler; R-T at fixed d | Load: filler type matters at fixed d. Distance: d matters, filler type does not. Dilution (attention): T matters at fixed d |
| | Less exposure at long lengths | Item-count-balanced fine-tune; LM-4 zero-shot | Exposure artefact: slope present after the fine-tune, absent in LM-4 |
| H3: multi-hop needs more attention | Composition depth | Hop ladder at fixed k; optional k = 2 adjacent vs spread | Depth: the required k tracks h |
| | Multi-hop is simply harder (calibration) | Difficulty-equated levels | Calibration: the difference vanishes when k = 12 accuracy is equated |
| | Specific to the delta rule | G arm (GLA-twin) | Mechanism: the GLA curve differs |
| Aggregation insensitive to k | Delta rule cannot accumulate; softmax normalisation hides absolute counts | A-freq (relative) vs A-count; G arm | Mechanism: GLA counts better; GDN is weak only at absolute counting |
| A hybrid beats all-full | Conv/gate extras; LR stability of gated layers; faster induction formation; attention sinks | Twin vs plain; LR sweep including the k = 12 optimum; dose ladder; activation diagnostics | Extras: the advantage disappears with twin mixers |
| A placement effect | Depth role vs layer-0 position availability vs distance to the LM head | Twin depth scan (attention carries its own conv) | Persistence of a middle advantage with twin mixers implies a depth role |
| "A bigger state could replace attention" | State capacity, not softmax | S arm (GDN-E, 3–4× state at equal parameters) | H6′: GDN-E closes ≥ 50% of the k = 0 → 1 gap |
| Hybrids use attention for everything (pathway imbalance, 2610.06750) | GDN layers become local processors in hybrids | Inference-time windowing of full layers vs GDN-state reset | Imbalance: windowing attention drops hybrids below k = 0, while resetting GDN barely matters |
| Efficiency gains are implementation artefacts | Triton vs SDPA quality; launch overhead | Analytic bytes and FLOPs vs measured; CUDA graphs; batch sweep | Artefact: measured ratios depart from analytic ones at large batch × T |

---

## 5. Questions I consider unresolved

1. **Elicited vs native capability.** All multi-hop and aggregation signal at about 28–49M parameters is created by exposure. How much the zero-shot tier (LM-3/LM-4) agrees with the elicited tier is unknown. If they disagree, which one answers the README's question?
2. **Compute vs memory.** At fixed parameters, the full layers also have more compute per token at long T. No arm in my design separates "unbounded memory" from "more compute". SWA at matched compute is a different family; I list it as open, not as a fix.
3. **Is the twin attention a fair "full attention"?** Four heads with conv and a gate make a strong but non-standard Transformer. The bridge arm measures the difference, but a reader may still prefer plain MHA as the definition.
4. **Scale transfer.** MiniMax reports that small-scale parity hid multi-hop deficits at scale (per LR, read-full blog, not re-verified). Two scales (28M, 49M) cannot settle this.
5. **Count vs proportion** cannot be identified at fixed L and fixed parameters without changing width. F24 changes width and state size.
6. **Document masking** improves LM quality ([2402.13991](https://arxiv.org/abs/2402.13991), per LR, not re-verified), but it shortens effective contexts. Whether unmasked packing would teach "ignore irrelevant prefix" better, and whether that would interact with k, is untested.
7. **The 0.8 threshold** for "most" is a convention. The H1-robust per-level curve is more informative than any single threshold.
8. **QK-norm at 32K** (U4) and **NoPE dilution** at 32K in the all-full model: their interaction with k is not separable within budget.
9. **GDN-E on sm_120** (K = 192/256) is unverified. If it fails, the state-size arm drops to 2×, which may be too small to be decisive.
10. **Whether the direct-supervision bound (Level A) really bounds LM capability.** I assume it does for random-key retrieval [INF]. Natural-language pretraining could in principle supply useful features.

---

## Appendix A. Verification log (by Reviewer A, 2026-11-09)

| Claim used in a decision | Source | Access | Result |
|---|---|---|---|
| Wang et al.: Table 7 per-mechanism recall; Table 5 subtasks; App. C (subtask × ratio; CWE/FWE not correlated); App. D (linear then full layer in each block); 340M recall "insignificant"; training context and parameter matching not stated in the setup (4,096 appears only in Limitations); no seeds or error bars | [2507.06457v2](https://arxiv.org/html/2507.06457v2) | read-section | Confirmed. **New:** GDN row flat to 12:1; Table 5 "Avg" ≠ column mean; DeltaNet non-monotone |
| Qiao et al.: scales S1–S5 (15–477M non-emb); training at 16K (S5 extended to 32K); Muon; RoPE base 10⁵; 1:1 main, 1:3 only for SWA-128; restriction test (full attention carries long-range information; GDN stores little); all-full below hybrids on NIAH@16K at S4 (35.95 vs 49.58 / 67.81); NoPE effect | [2606.15378v1](https://arxiv.org/html/2606.15378v1) | read-section | Confirmed. Conv-ablation Table 7 column labels unresolved, so not relied on |
| MAD: compute-optimal ratio 25% attention, 10% when state-optimal; striped hybrids beat both endpoints; MAD scores correlate with compute-optimal perplexity | [2403.17844](https://arxiv.org/html/2403.17844) | read-section | Confirmed. Missing from the LR |
| Olsson et al.: induction needs ≥ 2 layers; smeared keys enable one-layer induction and earlier formation; phase change at ≈ 2.5–5B (large) and ≈ 1–3B (small models) tokens | [2209.11895](https://arxiv.org/abs/2209.11895) (PDF text) | read-section | Confirmed verbatim |
| Canon layers lift NoPE to RoPE level; neighbour-token mixing | [2512.17351](https://arxiv.org/abs/2512.17351) | abstract-only | Confirmed at abstract level |
| Gated attention: performance, stability, tolerance of larger LR, sink mitigation, long-context extrapolation | [2505.06708](https://arxiv.org/abs/2505.06708) | abstract-only | Confirmed at abstract level |
| Stuffed Mamba: T_forget linear in state size; GDN out of scope | [2410.07145](https://arxiv.org/html/2410.07145) | read-section | Confirmed |
| Based: recall vs state-size Pareto frontier | [2402.18668](https://arxiv.org/abs/2402.18668) | abstract-only | Confirmed |
| 2026: pathway imbalance; load > elapsed context; Ω(√n) composition bound; CoT-SFT degrades hybrid recall | [2610.06750](https://arxiv.org/abs/2610.06750), [2609.33093](https://arxiv.org/abs/2609.33093), [2610.25250](https://arxiv.org/abs/2610.25250), [2606.11052](https://arxiv.org/abs/2606.11052) | abstract-only | Confirmed at abstract level. Not in the LR |
| Searches for a pre-empting controlled full-vs-linear ratio sweep with per-family long-context evaluation at 20–60M (2 extended + 2 standard WebSearch queries, 2026-11-09) | WebSearch | — | None found. Closest are listed in M10. "Post-Boundary Bridge" ([2606.02680](https://arxiv.org/abs/2606.02680) [abstract-only]) uses SWA, not linear attention. HALO/HypeNet ([2601.22156](https://arxiv.org/abs/2601.22156) [abstract-only]) is distillation. Ring-linear ([2510.19338](https://arxiv.org/abs/2510.19338) [abstract-only]) gives no ratio in its abstract |
| FLA GDN: conv + SiLU on q/k/v; FusedRMSNormGated swish output; `A_log` ~ log U(0, 16); "ShortConvolution is crucial" warning; GVA option | `fla/layers/gated_deltanet.py` @`98193113` | code-inspected | Confirmed |
| FLA GLA with conv + SiLU and gated norm; FLA kernel K ≤ 256 assert | `fla/layers/gla.py`; `fla/ops/common/chunk_delta_h.py` L716 | code-inspected | Confirmed |
| Mixer parameter counts (GDN-A/D/E/nc, GLA-twin) | meta-device instantiation, venv torch 2.11.0+cu128, fla 0.6.0 | measured (no GPU) | 1,056,904 / 1,054,980 / 1,054,210 / 1,052,808 / 1,065,344 |
| Zoology target figure | `zoology_results.png` @`1ad20d1` | code-inspected | "Attention" and "Sliding Window" share one marker |

## Appendix B. Arithmetic [INF]

- **Per-seed R̂ (delta method).**
  - Assume Var(D) = Var(G) = σ_d² and Cov(D, G) = σ_d²/2, because S₀ is shared.
  - Then Var(R̂) ≈ (σ_d²/G²)·(1 + R² − R).
  - At σ_d = 3 pp and R = 0.9: SD = 0.095 (G = 30), 0.143 (G = 20), 0.191 (G = 15).
- **Half-widths.**
  - df = 2, t = 4.303: 0.237 / 0.355 / 0.474.
  - Pooled df = 11, t = 2.201: 0.121 / 0.182 / 0.243.
- **Linear contrast.**
  - L_k = S_k − 0.2·S₀ − 0.8·S₁₂, so Var(L_k) = (1 + 0.04 + 0.64)·σ_run²/n.
  - At σ_run = 3 pp and n = 3: SE = 2.24 pp.
  - One-sided t₀.₉₅,₁₁ = 1.796 gives a margin of 4.0 pp, i.e. ΔR = 4.0/G = 0.13 at G = 30.
- **Placement MDE.**
  - SE(diff) = 3·√(2/2) = 3 pp.
  - Holm's first step: two-sided α = 0.005, df ≈ 10, t ≈ 3.58.
  - MDE ≈ (3.58 + 0.88)·3 ≈ 13 pp.
- **Exposure counts in the proposal (option i).**
  - Log-uniform lengths on [256, 4K]: mean 1,385 tokens, so 10M tokens ≈ 7,220 items.
  - On [4K, 32K]: mean 13.5K, so 5M tokens ≈ 371 items.
  - Ratio ≈ 19.5×.
- **Balanced fine-tune (my design).** 65.5M tokens / (0.5 + 1 + 2 + 4 + 8 + 16 + 32)K ≈ 1,030 items per bucket.
- **Effective context.** FineWeb-Edu tokens in documents ≤ 2,048 GPT-2 tokens = 100 − 40.91 = 59.1% (05 §5.1).
- **Attention FLOP share per full layer** (causal average): T·H·(d_qk + d_v) / [2·(4d² + 3·d·d_ff)].
  - MHA (H·d_qk = H·d_v = d, d_ff = 3d): T/(13d). At T = 32K: 4.9 at d = 512, 0.62 at d = 4096.
  - Twin (d/2 + d): T/(17.3d). At T = 32K: 3.7 at d = 512.
- **Twin attention parameters.**
  - d = 512: 2·(512·256) + 3·(512·512) + (256 + 256 + 512)·4 + 128 + 128 = 1,052,928.
  - d = 384 (3 heads): 2·(384·192) + 3·(384·384) + 3,072 + 256 = 593,152, vs GDN-A 595,334 (per the proposal, F12-S).

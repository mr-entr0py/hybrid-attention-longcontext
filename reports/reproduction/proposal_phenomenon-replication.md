# Reproduction suite proposal: phenomenon replication

**Role:** reproduction-program designer, phenomenon-replication angle. **Date:** 2026-11-09.
**Scope:** published results to reproduce *before* S4 (our confirmatory runs). Each one must test a named component or methodological assumption of our pipeline. The focus is the **hybrid-ratio and long-context phenomena themselves**, reproduced at our scale or with published weights.
**Companion:** `reports/reproduction/proposal_pipeline-validation.md` (pipeline-validation angle, written in parallel). Some items overlap: LAMBADA scorer check (its M2), nanoGPT ladder (M4), ratio-loss pattern (M5), Olsson (M6), Jelassi (S1), RoPE extrapolation (S2). For those I give only the phenomenon-specific criteria and defer to its protocol. Items unique to this file: **P2** (Wang et al.'s public 1.3B ratio-sweep checkpoints through RULER *and* our own generators), the **production-mixer and timing additions to S1** (P1), the **in-run phase-change budget check** (P4a), the **length-behaviour panel on FEAS Stage-A checkpoints** (S2), and the **Zoology seed replicate** (S5).

**Constraints honoured.** Nothing in `src/`, `tests/`, `scripts/`, `configs/` or `data/` was edited. No GPU job was launched. No target number below comes from memory: each carries a URL, a table or figure, and an access tag.
- [read-section] / [read-full] / [code-inspected]: opened by me in this step.
- "per X": taken from a repo report and **not** re-opened by me.
- **(fig)**: read off a figure, so approximate.
- [INF]: my own inference or arithmetic.

---

## 0. Summary

| # | Tier | Reproduction (phenomenon) | Validates (ours) | Rival(s) addressed | Code | GPU-h |
|---|---|---|---|---|---|---|
| **P1** | must | Zoology MQAR: attention solves at small width; GDN accuracy grows with state (S1, protocol §17) **+ production-config mixers + timing fix** | Mixer correctness and recall ability, including the *production* F_g (RoPE θ = 5e5, QK-norm, gate) and GDN-A | E10.4, E12.3 | Original (Zoology @1ad20d1) **and** ours | 3.0 budgeted (realistic 3–5; §2.1) |
| **P2** | must | **Wang et al. 2507.06457 ratio trend on their public 1.3B checkpoints:** GDN recall pure ≈ 24:1 ≈ 12:1 < 6:1 ≈ 3:1. Measured with RULER (original) *and* our CS generators | Our evaluation suite's **sensitivity to k** on real ratio-varied GDN hybrids; generator and scorer on trained hybrids; the H1-min prior | E1.4, E1.5, E1.10, E3b.3, E10.3 | Original eval (RULER @db43f92) **and** ours | ≈ 2.0 |
| **P3** | must | **Ratio-loss shape** (Waleffe Fig. 4; Mamba-2 Table 2; Kimi; Bae): the first attention layer gives most of the loss gain, loss is flat for k ≥ 1, and the best hybrid is no worse than all-full. Run in our full LM pipeline | Full model build, param matching, trainer, LR rule; RP-2 made quantitative | E10.4, E1.8 / E2.5, E9.1 | Ours (no runnable original; §3.3) | 3.2 (D384) / 4.2 (D512) |
| **P4** | must | **Induction-head phase change** (Olsson 2209.11895): abrupt in-context-learning (ICL) jump to ≈ 0.4 nats. **Smeared keys** let 1-layer models form it | Training budget is past the phase change for every k; the premise of the conv-asymmetry rival | E1.6, E6.2, E13.1, E1.9, E2.3 | Ours | ≈ 0.6 |
| S1 | should | Copying sample-efficiency gap (Jelassi 2402.01032) with our production mixers | Size of the learning-speed confound before the TEF | E1.7, E2.2, E6.7, E1.3 | Ours (+ optional original Transformer side) | ≈ 1.5 |
| S2 | should | Length behaviour of Stage-A (4K-trained) checkpoints: RoPE extrapolation failure (Press 2108.12409); GDN retrieval decay beyond training length (GDN 2412.06464 Table 2) | Positions used and reset correctly; why Stage B at 32K is needed | E1.2, E6.6, E3b.2 | Ours (eval only) | ≈ 0.3 |
| S3 | should | LAMBADA on public checkpoints (defer to companion M2; add cached GPT-2) | Scorer EM/logp identity with lm-eval | E10.3, E1.10 | Original **and** ours | ≈ 0.1 extra |
| S4 | should | nanoGPT shakespeare-char best val 1.4697 (defer to companion M4) | Environment trains a reference model to a published number | env | Original | ≈ 0.2 |
| S5 | should | Second seed of Zoology's best cell per (family, width) | Gives G-RP(b)'s ±0.03 a measured basis | — | Original | ≈ 1.2 |
| N1–N5 | nice | modded-nanogpt track-3 truncated curve; full RULER on P2; Wang 340M floor check; depth curves; Jelassi phonebook | §5 | | | ≈ 4–6 |

**Totals [INF]:**
- **Must ≈ 8.8 GPU-h.** Of this, 3.0 h (P1) is already in the protocol's S1 budget and 1–2 h may overrun (§2.1). Worst case ≈ 11 h, under the ≈ 15 h ceiling.
- **Should ≈ 3.3 h.**
- **Nice ≈ 4–6 h.**

**Unique findings from this step that need PI attention now:**
1. **S1's cut rule was evaluated on the fastest family.** `experiments/s1_zoology/cut_decision.json` projects 0.073 GPU-h from the first attention run, which early-stopped after 1 epoch in 16.5 s. The first finished GDN run took 993.5 s (32 epochs, `gated_delta_net-d128-lr1.0e-03.json`). A GDN d256 smoke epoch took 121.8 s, which is ≈ 1.1 h per 32-epoch run if it does not early-stop [observed in run JSONs at 18:09; projection INF]. The 3.0 h budget can therefore be exceeded without the cut ever firing. §2.1.
2. **Wang et al.'s 1.3B GDN ratio sweep and Transformer baseline are public**, so the hybrid-ratio recall phenomenon can be re-measured on its original weights, with our own generators, before we train anything (P2). Collection: `m-a-p/hybrid-linear-attention-research` [code-inspected, HF collection API, this step].
3. **Olsson's phase change occurred at ≈ 1–3B tokens with ≈ 1M-token steps.** Our S3 runs have 1,145 steps of 131K tokens. They may end *before* the phase change for some k, and that would make loss comparisons across k misleading (P4a).

---

## 1. Research practices for reproductions (proposed; apply to every item)

1. **Pre-register before running.**
   - For each item, write the exact claim, its source (URL, table or figure, access tag), metric definition, config, tolerance and pass/fail/inconclusive rule.
   - Put these in `experiments/repro/<id>/prereg.yaml`. Commit its SHA-256 and a `decisions.md` row (D-REPRO-xx) *before* the first run.
   - A reproduction that is not pre-registered is reported as exploratory.
2. **Original code first, then ours, then swap one component at a time.**
   - Ladder: original code in our environment (tests the environment) → our module inside the original harness (tests the module) → our harness and data (tests the pipeline).
   - P1 implements this as (a) → (b) → (c). When a rung fails, the rung below localises the fault.
3. **Use replication grades, not one yes/no.**
   - *Numeric reproduction:* same setting, value within tolerance.
   - *Phenomenon replication:* same direction and shape at our scale, with directional criteria.
   - *Not reproducible here:* setting infeasible; reason stated.
   - Each item states which grade it targets.
4. **Base tolerances on seed variance, never on taste.**
   - Use the variance the paper reports. If it reports none, run ≥ 2 seeds of the original code (S5) and use ±max(2·SD_seed, 2·SE_item).
   - Item SE: binomial √(p(1−p)/n), with a design effect when queries share a context.
5. **Never read gate thresholds off a figure** (I-20).
   - Figure values (Zoology README, Waleffe Fig. 4) are context only.
   - Gates use re-run numbers or numbers printed in tables.
6. **Audit metric definitions before comparing numbers.**
   - Example: GPT-2 124M on LAMBADA. The GPT-2 model card's 45.99% is a different setup from lm-eval `lambada_openai`, which gives 0.3256 ([HF discussion #90](https://huggingface.co/openai-community/gpt2/discussions/90) [read-full]; that thread calls the 45.99 target unreproduced).
   - Wang's Table 7 "recall" does not state its RULER subset (§3.2). That item can therefore be a phenomenon replication only.
7. **Include a negative control in each reproduction**, i.e. a condition that should fail, so we know the measurement can fail. Examples: GDN at small state in MQAR; RoPE beyond the training length; a 1-layer model without smeared keys (P4b).
8. **Never tune our code to hit a number.**
   - A failure leads to diagnosis along the pre-written tree. A change to `src/` needs a mechanistic explanation, logged as a decision row with a regression test, before re-running.
   - Never re-run with "another seed until it passes". Every attempt is recorded and reported.
9. **Report failed and inconclusive reproductions** in `reports/reproduction/results.md`, with the same prominence as passes.
10. **Project timing from the slowest family** (lesson from S1's cut rule): re-project after the first run of every family.
11. **Firewall between reproduction and confirmatory work.**
    - Reproductions never use the frozen test-item generator (eval_test entropy 20261109). They use a dedicated `repro` stream (proposed entropy 31337).
    - Their results change nothing in the protocol except through pre-declared audit triggers.
    - P3 adds k ∈ {1, 2, 6} *LM-loss* runs before G5. That is pilot exposure to the IV, but on a different stage, seed and outcome than the confirmatory task scores. It is logged as such.
12. **Debate step before S4.** After the must tier, hold a written "reproduction review": a short record in `reports/reproduction/review.md`, with two adversarial readers if possible.
    - List which assumptions are validated, which are refuted, and which stay open.
    - Then make a go / fix / re-scope decision.
    - G-RP stays the only hard gate in the frozen protocol. The other items are audit triggers unless the PI promotes one to a gate in a decision row *before* it runs.

---

## 2. Must tier

### P1. Zoology MQAR (S1, G-RP), with production-mixer and timing additions

**Claim (phenomenon).**
- On MQAR, attention reaches near-perfect recall at small width.
- Recurrent mixers' recall rises with state size.
- Gated DeltaNet is near the top of the efficient mixers.

**Source.**
- [Zoology 2312.04927](https://arxiv.org/abs/2312.04927) [read-section per PROP].
- README "Recall-Memory Tradeoff" figure at `third_party/zoology/zoology/analysis/paper/zoology_paper/zoology_results.png` @1ad20d1 [code-inspected, viewed this step] **(fig)**. GDN points sit at ≈ 0.89 at ≈ 1.7e4 state, ≈ 0.99 at ≈ 6.6e4, and ≈ 1.0 at ≥ 2.6e5. Attention and sliding-window share one black marker, so attention cannot be read off (I-20).
- Mamba-2 §9.1 also reports "Increasing the state size from N = 16 to N = 64 and N = 256 consistently improves MQAR performance" ([2405.21060](https://arxiv.org/html/2405.21060) [read-section]).

**Validates.**
- (a) The environment runs Zoology faithfully on sm_120 (fla 0.6.0 shim, torchvision).
- (b) Our mixers in Zoology geometry match Zoology's.
- (c) **New here:** the mixers *as used in the LM* (F_g with RoPE θ = 5e5, QK-RMSNorm and headwise gate; GDN-A geometry) have recall ability. As frozen, S1 tests only a stripped-down F_g with no gate, no QK-norm and no RoPE.
- Rules out E10.4 (mixer bugs) and E12.3 (GDN numerics at other geometries).

**Config.** As frozen (§17; `reports/s1_reproduction.md` §2). Additions:
- **(c-1) Production F_g at d = 128:** 2 heads × 64, which matches Zoology's 2 heads. Run at Zoology-attention's best LR at d = 128 and at one neighbouring LR (2 runs).
- **(c-2) GDN-A:** already planned as the "exploratory config-A" arm at d ∈ {128, 256}, i.e. 1 and 2 heads of d_k 64 / d_v 128.
- **(d) Phenomenon read-out added to the summary.** Per-test-config accuracy vs (seq_len, num_kv_pairs) for each family. GDN accuracy vs analytic state size (H·d_k·d_v per layer), plotted beside the README figure, clearly labelled "figure context, not a gate".

**Scale-down.** None. This is the paper's own configuration.

**Pass criteria.**
- **G-RP (a) and (b): as frozen.** Best-over-LR overall accuracy: GDN ≥ 0.95 at d ≥ 128; attention ≥ 0.95 at d ≥ 64; GDN non-decreasing in d. Our mixers within ±0.03 of Zoology's at matched d and LR.
- **(c), pre-register now:**
  - Production F_g, mean accuracy over the **in-distribution test configs** (seq_len ≤ 256: 64/4, 64/8, 64/16, 128/32, 256/64): ≥ 0.95.
  - GDN-A at d = 256: ≥ 0.90 overall.
  - GDN-A at d = 128 is descriptive only (1 head; state 8,192 per layer, below the GDN points in the figure).
- **The 512/128 and 1024/256 configs are descriptive for production F_g.** They lie beyond the training length of 256 with RoPE. Rotary perplexity rises past training length ([Press et al. 2108.12409v2](https://arxiv.org/html/2108.12409v2) Table 2: rotary trained at L = 512 gives 20.07 at 512 and 31.58 at 2512 [read-section]). Their θ = 1e4 differs from our 5e5, so I pre-register no direction for our θ.
- **±0.03 has no measured basis yet.** Zoology reports no seed variance. S5 supplies one.

**GPU-h.**
- Protocol budget 3.0. Observed: attention runs take 16–17 s each (early stop at valid acc > 0.99 after epoch 1); GDN d128 at LR 1e-3 took 993.5 s (32 epochs) [observed, run JSONs].
- Realistic total: 3–5 h, depending on whether d256 GDN early-stops. (c-1) adds ≈ 0.1–0.3 h.

**Depends on S0:** `models/mixers.build_mixer` (for (b) and (c)). It exists in `src/hybridlc/models`; `our_mixers.py` will stop raising once it is wired.

**Failure → diagnosis.**
- (a) fails: environment. Check the fla head_first shim (`s1_reproduction.md` §3.2), TF32 / `FLA_FLASH_QLA`, and the Zoology data cache. Re-run one config with fla's `fused_recurrent` mode to rule out the chunk kernel.
- (b) fails while (a) passes: module mismatch. Work through the known residual list: biases, attention dropout 0.1, `out_proj` init name, 2 × d/2 heads.
- (c-1) fails while (b) passes:
  - First suspect RoPE pairing or positions (VT-09 at trained weights; companion M2(c)).
  - Then QK-norm temperature at small d: compute the max attention logit. With unit gains and head_dim 64, the logit is bounded by 64/√64 = 8 [INF].
  - Then the gate initialisation.
  - **Do not change F_g.** Log the failure. It raises the prior on E1.2 / E2.7 (temperature) and must be discussed at the reproduction review.

#### 2.1 Timing correction (recommendation; no change to the frozen runs)

- The frozen rule says "if the first timed run implies > 3.0 GPU-h, cut". It was applied to the attention d64 run (16.5 s × 16 = 0.073 h), so no cut was made.
- Applied to the first GDN run instead (993.5 s × 16 ≈ 4.4 h), it *would* have fired [INF].
- I recommend:
  - (i) Log this as an observation (DEV-candidate, PI decision).
  - (ii) Do not retro-cut runs already done.
  - (iii) If the GDN d256 rows look on course to exceed ≈ 4 h, apply the pre-declared fallback (2 best LRs per family, ties to the smaller LR) to the GDN d256 row only. The rule's own selection is unchanged.
- This is a budget rule, not a validity rule, so either choice leaves G-RP valid.

---

### P2. Wang et al.'s hybrid-ratio recall trend on their public 1.3B checkpoints (RULER + our generators)

**Claim.** In a controlled FineWeb-Edu sweep, averaged RULER recall for GDN hybrids is:
- pure **0.356**; 24:1 **0.341**; 12:1 **0.352**; 6:1 **0.425**; 3:1 **0.436**.
- Transformer: **0.423** (Table 5); in Fig. 2 a dashed line "at approximately 0.42" **(fig)**.

**Source.** [2507.06457v2](https://arxiv.org/html/2507.06457v2) Table 7 [read-section, this step; also PI-verified per `literature_review.md`]. Known gaps in the source:
- The RULER subtasks, context length and sample counts behind Table 7 are not stated.
- The caption says only "averaged". Table 6 averages over 340M and 1.3B, and the 340M recall is "insignificant" (per lit review).
- The full-attention layer is the *last* layer of each block.
- 1.3B models were trained on 100B tokens at 1M-token batches.
- The training context length is 2,048 (v1) or 4,096 (v2), unresolved (U1).

**Checkpoints (live).** `m-a-p/1.3B-100B-GatedDeltaNet-{pure, hybrid-24-1, hybrid-12-1, hybrid-6-1, hybrid-3-1}` and `m-a-p/transformer_1.3B_baseline` [code-inspected, HF collection API `m-a-p/hybrid-linear-attention-research`, this step]. Architecture details are per the companion file M2/S4 (config.json [code-inspected per companion]): attention in fla's `attn.py`, which imports `flash_attn`; RoPE θ = 1e4; no QK-norm.

**Why this is must-tier from the phenomenon angle.**
- It is the only *controlled GDN ratio sweep* with released weights.
- k/24 ∈ {0, 1, 2, 4, 8, 24} is close to our k/12 ∈ {0, 1, 3, 12} grid.
- The published shape already bears on our hypotheses: **1–2 of 24 attention layers did not raise averaged recall above pure GDN** (0.341 / 0.352 vs 0.356), while 4–8 did. That is a direct prior for H1-min (R(1) low), which the protocol already predicts [§2 "Predictions"].
- If **our** confirmatory generators (R1, R-load, M-hop, A-freq; cloze EM_tf) cannot discriminate 3:1 from pure on these models, our evaluation design is insensitive to the very manipulation we study. We should find that out before 86 GPU-h, not after.

**Design.**
- **(a) Original evaluation code: RULER @db43f92**, HF backend.
  - Subset: niah_single_1/2/3 and vt at 2K and 4K (because of U1), 100 samples per task per length, greedy. Base-model prompts as RULER ships them.
  - (Full 13-task RULER is N2.)
  - Run on all 6 checkpoints.
  - Hybrid and Transformer checkpoints need the companion M2's validated repro-only SDPA shim for `flash_attn_func`: max abs diff ≤ 1e-5 fp32 against a naive softmax, validated before use.
- **(b) Our evaluation code.**
  - Generate our CS-style items with *each model's own tokenizer* through the generators' `tok` argument, in a dedicated `repro` RNG stream (never eval_test):
    - R1 D ∈ {3, 15};
    - R-load N ∈ {8, 32, 128};
    - M-hop h ∈ {1, 2};
    - A-freq ρ ∈ {4, 2};
    - plus the C0 no-filler controls.
  - Lengths T ∈ {1K, 2K, 4K}; 100 contexts × 4 queries per cell.
  - Score with `eval.scorer`, through the companion M2 transplant wrapper or `ForwardOnlyAdapter`, as fp32 single-pass.
  - If a tokenizer breaks VT-12's "4 single-digit answer tokens", use rank accuracy only for that model and flag it.

**Pass criteria (pre-register; phenomenon grade).**
- **(a-i) Ordering on RULER subset:**
  - S(3:1) − S(pure) ≥ +0.04 (published +0.080);
  - |S(24:1) − S(pure)| ≤ 0.04 (published −0.015);
  - S(Transformer) − S(pure) ≥ +0.03 (published +0.067).
- **Tolerance:** with 4 tasks × 2 lengths × 100 items, the SE of a per-model mean is ≈ 0.018 and the SE of a difference is ≈ 0.025 [INF, binomial at p ≈ 0.4]. The thresholds are about half the published gaps, i.e. about 1.5–2 SE.
- **Numeric agreement with 0.356 … 0.436 is not expected** (unknown subset), so it is not a criterion.
- **(b-i) Sensitivity of our suite:** on the retrieval family (R1 + R-load ladder mean, per §2 definitions, at 2K and 4K), S_ret(3:1) − S_ret(pure) > 0 and S_ret(Transformer) − S_ret(pure) > 0, each with a paired-context bootstrap 95% CI excluding 0 (contexts are shared across models).
- **(b-ii) Floor guard:** Transformer C0 (no filler) R1 D = 3 EM_tf ≥ 0.5. If it is lower, (b-i) is labelled *uninformative*: our cloze format is not zero-shot-elicitable even at 1.3B. That is consistent with our TEF design (I-08), and is reported, not a failure.
- **(b-iii) Descriptive, but the most useful output:**
  - S_f(k/24) and recovery R_f(k) for all three families, with the H1/H1-min analogues L_k on these published weights;
  - binding-error rates;
  - R-load accuracy vs N relative to the pure-GDN N50.
  - These are a k-curve *prior*, never used for power or thresholds (I-23).

**Validates.**
- Generator and scorer on trained hybrids with real attention / GDN interplay.
- Sensitivity to k (E1.5 ceiling compression, E3b.3 floor).
- Shortcut resistance: binding errors at D = 15 (E1.4).
- The metric: EM_tf vs logp sign agreement (E1.10).

**Scale-down.** None (published weights). Two caveats:
- The models are 1.3B with full attention at block ends. Ours are 21–41M non-embedding with centred placement and our TEF.
- Agreement is directional only.

**GPU-h [INF].**
- (a) ≈ 6 models × 800 generations ≈ 1.0 h with batched HF generation and max_new_tokens ≤ 64.
- (b) ≈ 6 × 3,000 contexts at ≤ 4K ≈ 0.7 h.
- Shim validation ≈ 0.1 h.
- Total ≈ 2.0 h.
- Requires downloading 6 × ≈ 2.7 GB (bf16 1.3B, [INF]).

**Depends on S0:** `eval/generators.py` with a foreign `tok`, `eval/scorer.py`, `eval/adapter.py`; companion M2's shim / transplant wrapper; a RULER environment (wonderwords, nltk; install approval may be needed).

**Failure → diagnosis.**
- (a-i) fails while companion M2(b) (LAMBADA on the same checkpoints) passes: suspect the shim's RoPE / GQA path at 2–4K, or RULER prompt formatting for base models. Re-test the shim against the fla reference at 4K.
- (a-i) passes but (b-i) fails:
  - Our generator or scorer cannot detect k.
  - Check C0 vs filler. If C0 also shows no k difference, it is format elicitation, not long context: compare zero-shot vs TEF-like 2-shot prompts, diagnostic only.
  - If C0 differs but filler cells do not, inspect needle-snapping and depth.
  - **Raise at the reproduction review before G5.** This is the main early warning this suite can give.
- Both fail: shim or tokenizer path (bisect with companion M2(c)).

---

### P3. Ratio-loss shape in our pipeline (Waleffe Fig. 4, Mamba-2 Table 2, Kimi, Bae; RP-2 made quantitative)

**Claims (all controlled from-scratch sweeps).**

| Source | Setting | Numbers | Access |
|---|---|---|---|
| Mamba-2 §9.2.3, Table 2 | 350M, 48 layers, 7B Pile tokens, GPT-2 tokenizer | Perplexity by number of attention blocks: 0 → 8.60; 1 → 8.38; 2 → 8.32; 3 → 8.29; 4 → 8.29; 5 → 8.28; **6 → 8.26**; 7 → 8.27; 9 → 8.28; 11 → 8.30; 15 → 8.34; 24 → 8.50; Transformer++ 8.68. "Having around a 10% ratio of attention layers performs best"; "Adding in just a few attention layers already yields notable improvement" | [2405.21060](https://arxiv.org/html/2405.21060) [read-section, this step] |
| Waleffe §4.1, Fig. 4 | 130M, 24 layers; tokens and data not stated in the section | "validation loss is minimized when roughly 8% of the layers are self-attention layers". Digitised: 0/24 2.615, 1/24 2.562, 2/24 2.554, 3/24 2.555, 4/24 2.555, 5/24 2.560, 7/24 2.568, 12/24 2.593, 24/24 2.631 **(fig, ±0.002)** | Text: [2406.07887](https://arxiv.org/html/2406.07887) [read-section, this step]. Values: per nemotron agent [read-full, digitised] |
| Kimi Linear ablation | ~653M activated, 16 layers, 38.8B tokens | KDA:MLA 0:1 5.77, 1:1 5.66, 3:1 5.65, 7:1 5.70, 15:1 5.82 val ppl | [2510.26692](https://arxiv.org/abs/2510.26692) per lit review [read-section per agents] |
| Bae et al. | 1B, 60B tokens | DCLM loss: Transformer 2.750, 1:1 2.725, 1:3 2.733, 1:5 2.735, 1:12 2.741, Mamba 2.758 | [2510.04800](https://arxiv.org/abs/2510.04800) per lit review [read-section per agents] |
| GDN paper Table 3 | 1.3B, 100B FineWeb-Edu | Wiki ppl: Transformer++ 18.53, GDN 16.42, GDN-H1 16.07 (H1 uses SWA, not global attention) | [2412.06464](https://arxiv.org/html/2412.06464) [read-section, this step] |

**Derived shape statistics [INF].**
- First-layer share φ = (L₀ − L₁)/(L₀ − min_k L_k):
  - Waleffe (fig): (2.615 − 2.562)/(2.615 − 2.554) = 0.87;
  - Mamba-2 on log-perplexity: ln(8.60/8.38)/ln(8.60/8.26) = 0.0259/0.0403 = 0.64.
- The best hybrid beats all-full in all four sweeps.
- With GDN as the linear mixer, pure linear can beat a Transformer on loss at this data (GDN Table 3). So "all-full beats pure-linear" is **not** a published regularity and is not a criterion.

**Validates.**
- Our full `HybridLM` across patterns: param matching; shared-module initialisation; that F_g layers contribute in a 12-layer stack.
- The trainer, data and LR rule on the IV axis.
- Makes the protocol's RP-2 bug detector ("validation loss flat for k ≥ 1, largest step at 0 → 1") quantitative.
- Rules out E10.4 in the LM setting, probes E1.8 / E2.5 (LR favouring hybrids), and checks the expected E9.1 validity pattern.

**Code.** Ours only. No original code can run here:
- Waleffe's sweep needs Megatron's `ssm` branch with `mamba_ssm`, for which there is no torch-2.11 wheel.
- fla's hybrid attention needs `flash_attn`.
- The "original-code" side is covered by the weight-transplant equivalence (VT-09; companion M2(c)) and by P2's published-weights check.

**Config.**
- S3 recipe (§7.2): 1,145 steps × 131,072 tokens ≈ 0.15B at 4K, compressed WSD, chosen scale, **at η\*** after G-LR.
- New runs:
  - k ∈ {1, 2, 6} at seed index 100 (S3 already has k ∈ {0, 3, 12} at 100): 3 runs;
  - k ∈ {0, 1, 12} at two new seed indices, proposed 300 and 301 (a new "reproduction" range, logged): 6 runs.
- Metric: LM-2 subset loss (as S3), plus LM-3 repeat-token loss by Δ bucket on the final checkpoint.
- Total 9 runs, all logged as reproduction runs, outside the confirmatory family.

**Scale-down.** The claim is reported consistently from 130M to 1B, across Mamba-2, KDA and Mamba mixers, and on loss or perplexity. A pipeline that cannot show the k = 0 → 1 step and the flat k ≥ 1 region at 21–41M is suspect, either at scale or in the code. LM-3 (below) separates the two.

**Pass criteria (pre-register; phenomenon grade).** σ = pooled seed SD of LM-2 from the k ∈ {0, 1, 12} triplicates (df = 6). The prior is ≈ 0.002 nats; modded-nanogpt reports a 0.0019-nat SD over 80 reruns at 124M (per lit review [code-inspected per methodology agent]).

| ID | Criterion | Published anchor |
|---|---|---|
| P3-a | L₀ − L₁ > 0: paired over 3 seeds, one-sided t p < 0.05 **and** magnitude ≥ 0.005 nats | Mamba-2 0.026 nats (log-ppl); Waleffe 0.053 (fig) |
| P3-b | φ ≥ 0.5 (point estimate, seed-100 values; seed-averaged where available) | 0.64 / 0.87 [INF above] |
| P3-c | min_{k ∈ {1,2,3,6}} L_k ≤ L₁₂ + 2σ | Best hybrid < all-full in all four sweeps |
| P3-d (RP-2 part 2) | LM-3 loss at Δ ∈ [1K, 4K) is lower for k = 12 than for k = 0, paired over seeds | Associative-recall tokens explain 82% of the attention vs gated-conv perplexity gap (Zoology, [2312.04927](https://arxiv.org/abs/2312.04927) [read-section per lit review]) |

- **All four pass:** "phenomenon replicated".
- **a or d fails:** audit trigger (below).
- **b or c fails alone:** "partial; scale-dependent?", reported and discussed, no audit.

**GPU-h.** 9 × 0.35 h = 3.2 h (D384; S3 rate from §13.2) or 9 × 0.47 h = 4.2 h (D512) [INF].

**Depends on S0:** trainer, configs (`lr_sweep.yaml`), `analysis` for LM-2/LM-3, Tier-0 LM-3 generator. Sequencing: after S3, before G5.

**Failure → diagnosis.**
- **a fails (attention adds nothing):**
  - Check P4a's ICL score. If k ≥ 1 has not formed induction by step 1,145, it is a budget problem: see P4a and re-test on FEAS Stage-A checkpoints.
  - Otherwise: VT-02/03/09 at trained weights; `position_ids` reset per document; `varlen_attn` `cu_seqlens`; attention max-logit and entropy logs.
- **d fails while a passes:** long-range repeat copying is not carried by attention. Check document masking: does attention cross separators in LM-3 documents? Check LM-3 positions.
- **c fails strongly (all-full better than every hybrid by > 3σ):** not necessarily a bug.
  - Our F_g has QK-norm and a gate, and is a stronger attention than those papers used.
  - Check GDN-specific items: `A_log`/`dt_bias` excluded from weight decay (VT-10), the conv init, and the LR regret of k = 0 in S3.
  - Report as "non-replication of the U-shape at this scale".
- **Never adjust the LR rule or mixers to restore the shape.**

---

### P4. Induction-head phase change in our runs, plus the smeared-key test (Olsson et al.)

**Claims.** From [2209.11895](https://arxiv.org/abs/2209.11895), PDF text [read-section, this step]:
- ICL score is defined as "the loss of the 500th token in the context minus the average loss of the 50th token".
- ICL "develops abruptly in a narrow window early in training (roughly 2.5 to 5 billion tokens) … Before this window there is less than 0.15 nats of in-context learning, after it there is roughly 0.4 nats, an amount that remains constant … (except for the one layer model where not much in-context learning ever forms)".
- Small models: 1–6 layers, d = 768, context 8,192, "trained for 10,000 steps (~10 billion tokens)". The phase change "appears at approximately 1-3 billion tokens in the small models". LR warm-up "ramps up over the first 1.5e9 tokens".
- Smeared keys k_j ← σ(α)k_j + (1−σ(α))k_{j−1}: "in-context learning does indeed form for one-layer models (when it didn't before), and it forms earlier for two-layer and larger models."

**Why it matters to us [INF].**
- (i) **Budget.** Olsson's phase change came at ≈ 1,000–3,000 steps of ≈ 1M tokens, after warm-up. Our S3 / P3 runs have 1,145 steps of 131K tokens, and Stage A has 4,578 / 6,104 steps.
  - If any k has not crossed the phase change by the end of Stage A, the confirmatory comparison mixes pre- and post-transition models.
  - This would be an unrecognised form of E2.3 (under-trained base) and of E1.9 (abrupt circuit formation leading to bimodal seeds).
- (ii) **The conv-asymmetry rival's premise (E1.6 / E6.2 / E13.1).** GDN layers carry a short conv, and F_gc adds one to attention. Both are generalised smeared keys.
  - If our F_gc does not form ICL in one layer where F_g does not, the mechanism behind E1.6 is weaker in our code.
  - If it does, the X-gc12 control is well-motivated and δ_gc may be large.

**Design.**
- **(a) In-run measurement (≈ free).**
  - ICL score on 256 held-out FineWeb-Edu test-half documents with ≥ 512 tokens (median document is 688 tokens; `data/stats.json` [MEAS]). Per Olsson: mean loss at in-document position 500 minus position 50, in nats.
  - Measured every 50 steps on the P3 / S3 runs, and every 250 steps on FEAS Stage A (k ∈ {0, 12}).
  - Needs either an in-loop eval hook or weights-only snapshots every 50 steps (bf16, ≈ 55 MB at D384 [INF]). The current `base.yaml` keeps only the last 2 full checkpoints every 250 steps, so **this is an S0 request** (no edit made).
- **(b) Smeared-key arm** (6 tiny runs; ours).
  - `ModelConfig.for_scale(scale, pattern, n_layers=…)` with patterns `A`, `C`, `-` at n_layers = 1, and `AA`, `CC`, `--` at n_layers = 2.
  - Stage-A data, η\* (or the provisional 2e-3), seed 300, 100M tokens.
  - Our blocks include a SwiGLU MLP. Olsson's small models came "both with MLPs and without", so this is acceptable [read-section].

**Pass criteria (pre-register; phenomenon grade).** Write I = L@50 − L@500, so that "≈ 0.4 nats" means I ≈ +0.4.

| ID | Criterion | Anchor |
|---|---|---|
| P4-a1 | Every config C0, C1, C3, C12 reaches I ≥ 0.15 nats by the end of the FEAS Stage-A run, and has plateaued (\|ΔI\| < 0.02 over the last 20% of Stage-A steps) | "< 0.15 before, ≈ 0.4 after" |
| P4-a2 (descriptive) | Formation step t½ (first eval where I ≥ ½·I_final) per config and seed; abruptness = steps from 10% to 90% of I_final; spread of t½ across the 3 seeds at k ∈ {0, 1, 12} | Abrupt window |
| P4-b1 | 1-layer `A`: I < 0.15; 1-layer `C`: I ≥ 0.15 at 100M tokens | Smeared-key 1-layer result |
| P4-b2 | t½(`CC`) < t½(`AA`) | "forms earlier for two-layer" |
| P4-b3 (descriptive) | 1-layer `-` (GDN-A) | No published anchor |

- **Magnitudes are not expected to equal 0.4 nats.** Olsson used a 2^16 vocabulary and 8K context; we use 16K and 4K. Only the 0.15 boundary is used, and it is the paper's own pre/post boundary.

**GPU-h [INF].** (a) ≈ 0.1 h of eval overhead. (b) 6 runs of ≈ 4–6 min each (1–2 layers; dominated by embedding/head and overhead) ≈ 0.5 h. Total ≈ 0.6 h.

**Depends on S0:** trainer eval hook or snapshot cadence (request), `n_layers` override (exists in `ModelConfig`), LM-1-style per-position loss.

**Failure → diagnosis.**
- **P4-a1 fails for some k at the end of Stage A:** a **budget validity** problem.
  - Report at the reproduction review. Options: lengthen the warm-up / Stage A, or label the affected k.
  - FEAS is the natural place to see it, *before* G5.
  - This must be decided on P4 data from FEAS/S3 only, never on task scores.
- **P4-b1 fails because `A` forms ICL in 1 layer:** check for leakage of future tokens (VT-02) and that RoPE is not acting as a previous-token feature. An attention mask bug can create spurious "induction".
- **P4-b1 fails because `C` does not form:** check the F_gc conv's causal padding and its document reset. Report as "smeared-key premise not reproduced with ShortConv(4)".

---

## 3. Should tier

### S1. Copying sample-efficiency gap (Jelassi et al. 2402.01032)

**Claim** ([2402.01032](https://arxiv.org/html/2402.01032) §3, App. A/B [read-section, this step]):
- Transformers need "100x less samples than the best GSSMs" to learn to copy strings of length ≤ 300. No exact counts are given, only Fig. 1(a) **(fig)**.
- At lengths ≤ 30, Mamba needs about 10× more data (App. B.1).
- Under length generalisation (train ≤ 50, test to 1,000), GSSM accuracy "drops to zero almost immediately". RoPE generalises worse than NoPE / ALiBi.
- Setup: vocabulary of 30 (a–z + BOS/EOS/COPY), ≈ 160M models (Transformer 12L d1024; Mamba 24L d1024, S = 32), batch 64, context 420 (620 for Fig. 1a), AdamW 5e-5, 300 warm-up steps.
- Code: [sjelassi/transformers_ssm_copy](https://github.com/sjelassi/transformers_ssm_copy) (not cloned; the Mamba side needs `mamba_ssm`).

**Validates.** The *size* of the learning-speed confound (E1.7, E2.2, E6.7) with our production F_g vs GDN-A. This informs whether the TEF dose ladder is likely to bind. It also gives a positive control for LM-4 copy-after-gap.

**Config (scaled).**
- 4-layer, d = 256 models via `ModelConfig(n_layers=4, …)`: `AAAA`, `----`, `-A--`.
- Jelassi's task and vocabulary, packed contexts of 620, lengths U[1, 300]; seeds 300 and 301.
- Train until string-level copy accuracy ≥ 0.95 at L = 300, or until a cap of 0.4B tokens.
- Length-generalisation sub-arm (train ≤ 50, test {50, 100, 200, 400, 1,000}) for `AAAA` with RoPE θ = 5e5 vs NoPE, and for `----`.

**Pass criteria (phenomenon).**
- (i) `AAAA` reaches 0.95 at L = 300.
- (ii) samples-to-0.95 ratio `----`/`AAAA` ≥ 3, with direction from Jelassi. Mamba S = 32 is weaker than GDN-A, so a smaller magnitude than 100× is expected.
- (iii) Descriptive: `-A--` lies between the two.
- **A ratio < 1 (GDN faster) is a non-replication**, reported as evidence that E1.7 is weaker for the delta rule. It is not a bug signal by itself.

**GPU-h:** ≈ 1.5 [INF; small models, ≤ 0.4B tokens each, 6 runs + length arm].
**Depends on S0:** models, `n_layers` override, a tiny synthetic trainer loop (or the trainer with a synthetic stream).
**Failure → diagnosis:** (i) fails → F_g at long range in a tiny model: RoPE, causal mask, QK-norm logit cap (as P1(c-1)).

### S2. Length behaviour of Stage-A checkpoints (RoPE extrapolation; GDN beyond training length)

**Claims.**
- Rotary trained at L = 512 on WikiText-103: perplexity 20.07 @512, best 19.79 @712, 21.37 @1012, 31.58 @2512, 54.98 @8512. Trained at L = 1024: 19.33 @1024 → 31.17 @2024 → 64.91 @8024 ([Press et al. 2108.12409v2](https://arxiv.org/html/2108.12409v2) Tables 2–3 [read-section, this step]; θ = 1e4).
- GDN 1.3B trained at 4K, zero-shot ([2412.06464](https://arxiv.org/html/2412.06464) Table 2 [read-section, this step]):
  - S-NIAH-2 (number in haystack): 100.0 / 99.8 / 92.2 / 29.6 at 1K / 2K / 4K / 8K;
  - S-NIAH-1 (passkey, repetitive filler): 98.4 / 88.4 / 91.4 / 91.8.
- Stuffed Mamba: official Mamba-2 checkpoints (except 130M) have "near-perfect retrieval accuracy within 8K tokens, but poor or even zero accuracy on sequences longer than 16K" ([2410.07145](https://arxiv.org/html/2410.07145) [read-section, this step]; per-model values **(fig)** only).

**Validates.** Positions are used and reset per document. The design rationale that evaluation must stay inside the training length (Stage B at 32K) holds in *our* models. Context for E1.2 / E6.6 / E3b.2.

**Config.** Eval-only on the FEAS end-of-Stage-A checkpoints S (k ∈ {0, 12}, seeds 101/102; trained at 4K):
- LM-1 per-position loss to 32K;
- R0 cloze passkey at {1K, 2K, 4K, 8K, 16K, 32K}, 100 items each, repetitive filler as Stuffed Mamba;
- the same on the end-of-Stage-B checkpoint (already Tier 0).

**Pass criteria.**
- **Sanity (pass/fail):** LM-1 loss is non-increasing across buckets [64, 256) → [1K, 4K) for both k at Stage A (within-window context use; Olsson's per-token loss framing).
- **Descriptive, with direction pre-registered for k = 12 only:** the Stage-A loss in [8K, 16K) exceeds [2K, 4K) (rotary extrapolation), and the rise vanishes after Stage B within 32K.
- **No direction for k = 0:** GDN passkey on repetitive filler stayed high to 8K at 1.3B, while S-NIAH-2 collapsed.

**GPU-h:** ≈ 0.3. **Depends on S0:** Tier-0 generators and scorer; FEAS checkpoints.
**Failure → diagnosis:** sanity fails → data packing / position reset (VT-03/11).

### S3. LAMBADA scorer identity (defer to companion M2)

Companion M2 covers this with Pythia-70M/160M and Wang's 340M models. **Addition:** GPT-2 124M is already in `~/.cache/huggingface/hub/models--gpt2`, so it needs no model download.
- Published lm-eval `lambada_openai` 0-shot: acc **0.3256 ± 0.0065**, ppl **40.06 ± 1.49** ([HF discussion #90](https://huggingface.co/openai-community/gpt2/discussions/90) [read-full]; a community run on an old harness version).
- Criteria as companion M2: acc within ±0.015; per-item EM agreement with lm-eval ≥ 99.5%.
- Note the leading-space convention: lm-eval puts " word" in the continuation, while our protocol puts whitespace in the prompt (§6.2). The adapter must follow lm-eval for this test.
- ≈ 0.05 GPU-h.

### S4. nanoGPT shakespeare-char (defer to companion M4)

- Target: "the best validation loss is 1.4697" (6 layers, 6 heads, d = 384, context 256, ≈ 3 min on an A100), from `third_party/nanoGPT/README.md` L51 @3adf61e [code-inspected].
- Original code, 2 runs; tolerance ±0.05 [INF; no variance reported].
- It isolates environment faults (torch 2.11 / sm_120 / SDPA / compile) from our code.
- ≈ 0.2 h.

### S5. Zoology seed replicate (gives G-RP(b)'s ±0.03 a basis)

- Re-run Zoology's best LR per (family, d) with seed 124 (Zoology default 123): 4 runs.
- Use |Δ| between seeds as the empirical seed spread. If the spread exceeds 0.015, log that ±0.03 is < 2 SD and report G-RP(b) with that caveat. The frozen threshold is not changed.
- ≈ 1.2 h (dominated by GDN; P1 timings).

---

## 4. Nice tier

| ID | Item | Target / source | GPU-h |
|---|---|---|---|
| N1 | modded-nanogpt track-3 AdamW baseline, original `records/track_3_optimization/train_gpt_simple.py`, 1 GPU with grad accumulation, train_steps = 5,625 (so the schedule is unchanged), stopped at step 500 | val_loss 4.15174 @500 (6.18993 @125, 5.07445 @250) in log `results/a63a68d1-…txt`; batch 8 × 64 × 1024 tokens; PyTorch 2.10 on 8 GPUs (@4ea6b93 [code-inspected]). No seed SD at intermediate steps, so tolerance ±0.05 [INF]. Needs ≈ 0.3B tokens of `fineweb10B` GPT-2 shards | ≈ 1.5–2 |
| N2 | Full 13-task RULER on P2's 6 checkpoints at 2K / 4K | Wang Table 7 numbers (subset unknown) | ≈ 3 |
| N3 | P2(b) on Wang's **340M** sweep: does our cloze format lift the "insignificant" 340M recall floor? | Wang Table 2 caption (per lit review) | ≈ 0.5 |
| N4 | Depth curves (lost-in-the-middle U-shape for attention, recency for recurrent) from CS / MS depth metadata | [2307.03172](https://arxiv.org/abs/2307.03172), JRT [read-section per lit review] | 0 (analysis) |
| N5 | Pythia-410M vs Mamba-370M phonebook (HF slow-path Mamba) | Jelassi Fig. 1(c): 410M beats 2.8B GSSM for L ≥ 70 **(fig only)** | ≈ 0.5 |

## 5. Considered and not attempted (with reasons)

- **Wang Table 7 numeric values.** Unknown RULER subset and averaging → phenomenon grade only (P2).
- **Waleffe 8B RULER / phonebook** (Table 10; 100% phonebook to ≈ 5.5K): 8B weights, `mamba_ssm`, compute.
- **Stuffed Mamba from-scratch 36–85M passkey:** `mamba_ssm` unavailable. Per-model values are figure-only. Only the fit T_recall = 4.756·(1.365^N_S − 1) − 0.742 is printed [read-section].
- **Bae et al. NIAH vs ratio:** figure only; 350M–1B at 60B tokens.
- **Qiao et al. NoPE at 0.22B** (49.58 → 67.81): needs 16K pretraining at 0.22B, outside budget. Our X-nope arms test it in-design.
- **Nemotron-Flash SWA ratio CPT:** a 3B CPT, and SWA rather than linear layers.

---

## 6. Budget, order and dependencies

| Order | Item | Can start | GPU-h (must / should) |
|---|---|---|---|
| 1 | P1 (S1 running) + S5 | now (S5 after S1) | 3.0 (≤ 5) / 1.2 |
| 2 | S3 GPT-2 lm-eval half; S4 nanoGPT | now (no S0 dependency for the original-code halves) | — / 0.25 |
| 3 | P4b smeared-key arm; S1 copy | S0 models + trainer ready | 0.5 / 1.5 |
| 4 | P2 (a, b) | S0 eval + companion M2 shim validated | 2.0 / — |
| 5 | S2; P4a on FEAS | during S2 FEAS | 0.1 / 0.3 |
| 6 | P3 (+ P4a on its runs) | after G-LR, before G5 | 3.2 (D384) / — |
| 7 | **Reproduction review** (§1.12) → go / fix / re-scope | before G5 | — |

**Must ≈ 8.8 h (worst case ≈ 11 h). Should ≈ 3.3 h. Nice ≈ 5–6 h.**

**S0 requests** (no edits made by me):
1. Eval hook or weights-only snapshot cadence of 50 steps for P3 / P4a runs.
2. `tok`-parametrised generators usable with foreign tokenizers, plus a `repro` RNG namespace.
3. ICL-score helper (per-position loss at positions 50 and 500).
4. Decision-log rows for seed range 300–301 and for the P3 runs.

---

## 7. Sources opened for this proposal

| Source | Used for | Access |
|---|---|---|
| [2209.11895](https://arxiv.org/abs/2209.11895) Olsson et al. (PDF text) | ICL definition; 2.5–5B / 1–3B tokens; < 0.15 → ≈ 0.4 nats; smeared keys; small-model details | read-section (this step) |
| [2402.01032](https://arxiv.org/html/2402.01032) Jelassi et al. | Copy setup; "100x less samples"; App. B.1 ~10×; length generalisation; code URL | read-section (this step) |
| [2410.07145](https://arxiv.org/html/2410.07145) Stuffed Mamba | Scratch model table; passkey format; Mamba-2 "near-perfect within 8K … poor >16K"; T_recall fit | read-section (this step) |
| [2412.06464](https://arxiv.org/html/2412.06464) Gated DeltaNet | Table 2 S-NIAH; Table 3 LM; training setup | read-section (this step) |
| [2406.07887](https://arxiv.org/html/2406.07887) Waleffe et al. | §4.1 text (8% optimum, 130M / 24 layers, loss metric); phonebook | read-section (this step); Fig. 4 values per nemotron agent (fig) |
| [2405.21060](https://arxiv.org/html/2405.21060) Mamba-2 | §9.2.3 Table 2; §9.1 MQAR state-size trend | read-section (this step) |
| [2108.12409v2](https://arxiv.org/html/2108.12409v2) Press et al. (ALiBi) | Tables 2–3 rotary / sinusoidal extrapolation | read-section (this step) |
| [2507.06457v2](https://arxiv.org/html/2507.06457v2) Wang et al. | Table 7 GDN recall; Table 5 Transformer; block layout; tokens; release statement | read-section (this step) |
| [HF collection m-a-p/hybrid-linear-attention-research](https://huggingface.co/collections/m-a-p/hybrid-linear-attention-research) | Checkpoint availability (62 items; 1.3B GDN sweep + Transformer) | code-inspected (API, this step) |
| [HF discussion: gpt2 #90](https://huggingface.co/openai-community/gpt2/discussions/90) | lm-eval lambada_openai 0.3256 / 40.06; 45.99 not reproduced | read-full (this step) |
| `third_party/zoology` @1ad20d1: README figure, configs | P1 figure context | code-inspected (this step) |
| `third_party/nanoGPT` @3adf61e README | 1.4697; GPT-2 OWT 3.11 / 3.12 | code-inspected (this step) |
| `third_party/modded-nanogpt` @4ea6b93 track-3 README, log, script | AdamW baseline curve; batch; schedule | code-inspected (this step) |
| `experiments/s1_zoology/*` | S1 timings (16.5 s attention; 993.5 s GDN d128; 121.8 s / epoch GDN d256 smoke) | observed files (this step) |
| `reports/experimental_protocol.md`, `competing_explanations.md`, `literature_review.md`, `literature_db.yaml`, `s1_reproduction.md`, `INTERFACES.md`, companion proposal | Design context; Kimi / Bae / Zoology-paper numbers (per agents) | read-full / read-section |

**Not opened by me (cited per others):** the Zoology paper body (82% / 97.4%); Kimi Linear; Bae et al.; Waleffe Fig. 4 digitisation; the modded-nanogpt 80-rerun SD; Wang Table 2 (340M); RULER internals beyond the lit review.

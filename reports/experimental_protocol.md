# Experimental protocol v1.0 (frozen): How Much Full Attention Does Long Context Need?

*A controlled study of the number of full-attention layers (k of 12) in small hybrid language models*

**Status:** FROZEN 2026-11-09 by the PI. **Supersedes** `reports/initial_proposal.md` (v0.1) wherever they differ.
**Rationale source:** `reports/reviews/debate_record.md` (issues I-01…I-29, open items U-01…U-16). Decision rows D002–D044 in `reports/decisions.md`. Rival explanations: `reports/competing_explanations.md`. Machine-readable stage 0–2 plan: `configs/experiments/plan.yaml`.

**What "frozen" means.**
- Every design choice below is fixed now.
- Six values are fixed now *as rules* and filled in by those rules before the hash freeze (gate G5): scale, Stage-B budget, SSMax adoption, TEF synthetic share, contexts per confirmatory cell, learning rate η*. Each value is recorded in `reports/decisions.md` as it is decided.
- After G5 (end of S3), any change is a logged deviation ("DEV-xx" row in `decisions.md`, with its rationale).

**Conventions.**
- Access tags: [read-full], [read-section], [abstract-only], [code-inspected]. "per X" means the access was X's and was not re-checked by me. **[PI-VER]** means I re-checked it in this step (2026-11-09).
- Epistemic tags: **[EST]** established in the cited source; **[HYP]** hypothesis; **[INF]** inference, design choice or arithmetic.
- Measurement tags: [MEAS-B] and [MEAS-B2] are Reviewer B's measurements (contended GPU); [PI-SIM] is the PI's CPU simulation (debate record §6).
- k = number of full softmax-attention mixers among L = 12; r = k/12. Claims are stated as "k of 12" (I-25).
- **Nemotron-H** ([2504.03624](https://arxiv.org/abs/2504.03624)) mixes attention, Mamba-2 and MLP layers. It is not a controlled full-vs-linear ratio experiment and is used nowhere below as evidence.

**Corrections made while freezing (relative to the debate record).** Each is logged in `decisions.md`.
1. **FEAS-384 costs ≈ 9.8 GPU-h, not ≈ 6.** Its contents (2 seeds × 2 Stage-B branches, a TEF on each branch, an SSMax arm, evaluation), priced at the debate's own per-token rates, give 9.8 h (§13.2; D041).
2. **Frozen evaluation suites cost 9.8 h (D384 path) / 12.1 h (D512 path)**, plus 3 h of cost benchmarks (§9.9).
3. **Revised Tier A:** **≈ 85.7 GPU-h base (≈ 103 h with 20% contingency) on the D384 path** and **≈ 144 h (≈ 173 h) on the D512 path**. The debate record had 80 (96) and 138 (166). Tier B is unchanged (≈ 37 / 67 h).
4. **Minimum H1-falsification experiment:** ≈ 53 h (64 h) on the D384 path, ≈ 94 h (112 h) on the D512 path. The debate record had 49 (59) and 90 (107).
5. **Three operational gaps the debate left open are closed here:**
   - FEAS runs at a provisional LR of 2.0e-3, because FEAS precedes the LR sweep (D034).
   - TEF loss weighting: ½ answer tokens + ½ other tokens (D014).
   - Evaluation contexts contain no document separator (D039).

---

## 1. Research questions

Verbatim from the README; operationalisations follow.

- **RQ-A, Composition.** How does the proportion of full attention affect long-context performance when everything else is held fixed?
  - *Operationalisation:* family scores S_f(k, T) for k ∈ {0, 1, 2, 3, 6, 12}, T ∈ {1K, …, 32K}, all within the training length.
  - The confirmatory reading is at the long-context primary cell (§11.2).
- **RQ-B, Placement.** Does where the full-attention layers sit matter independently of how many there are?
  - *Operationalisation:* early / middle / late patterns at a fixed k*, which is chosen from composition data only (§3.2).
- **RQ-C, Task dependence.** Do retrieval, multi-hop reasoning and information integration need different amounts of full attention?
  - *Operationalisation:* normalised recovery R_f(k) per family, on the probability scale.
  - "Information integration" means literal-token aggregation (frequency comparison) (I-29).
- **RQ-D, Cost.** How do the capability differences relate to measured inference-state memory, prefill latency and decode throughput?
  - *Operationalisation:* analytic costs are primary (state bytes as stored, FLOPs, bytes moved, roofline latency).
  - Measured costs are secondary, under the artefact-controlled protocol of §10.

---

## 2. Hypotheses (pre-registered)

**Definitions** [INF].
- **Family scores.** S_f(k, T) is computed per seed. Each task's ladder mean (equal weight per level) is taken first, then the equal-weight mean of the family's primary tasks:
  - retrieval = {R1, R-load};
  - multi-hop = {M-hop};
  - aggregation = {A-freq}.
- **Gap and recovery.** G_f = S_f(12) − S_f(0) and R_f(k) = (S_f(k) − S_f(0)) / G_f.
- **H1 contrast.** L_k = S_k − 0.2·S₀ − 0.8·S₁₂. When G > 0, L_k ≥ 0 ⇔ R(k) ≥ 0.8 (I-03).
- **Primary cell** (I-05):
  - the per-seed mean over those of {16K, 32K} that pass G-REF (§11.2) of the retrieval ladder mean at that length;
  - if exactly one passes, that length;
  - if neither passes, 8K, with the verdict labelled "H1 at 8K, not a long-context result".
- **"Supported"** means the stated test rejects in the pre-registered direction after Holm. **"Falsified"** means the stated falsification rule holds. Anything else is **"inconclusive"**.

| ID | README hypothesis | IV (levels) | DV (estimand) | Pre-registered direction | Test and support rule | Falsifier / negative outcome | Family |
|---|---|---|---|---|---|---|---|
| **H1** | A small proportion of full-attention layers recovers most long-context performance | k ∈ {0, 3, 12}, centred placement | L₃ on the primary-cell retrieval ladder mean, using EM_tf | L₃ > 0 (R₃ ≥ 0.8) | Paired per-seed one-sided t (df = n − 1). Holm over {H1, H1-min, H3, H5a} at α_FW = 0.05. Supported iff the Holm-adjusted one-sided p (direction L₃ > 0) is < 0.05 | Falsified iff all four I-22 conditions hold (§11.6): gates pass; the falsification-direction p (L₃ < 0) is < 0.05 after Holm over {H1, H1-min}; the dose guard passes; the saturation guard passes | Confirmatory |
| **H1-min** | One full layer suffices for most of it | k ∈ {0, 1, 12} | L₁ | L₁ > 0 | As H1 | As H1, at k = 1 | Confirmatory |
| **H2** | Gains from adding full attention show diminishing returns | k ∈ {0, 1, 2, 3, 6, 12} | S_f(k) curve; isotonic and saturating fits; k80 with bootstrap CI; g₁ = S(1) − S(0) vs g_late = (S(12) − S(6))/6 | Concave | **Descriptive only** (I-21: any concave curve nearly implies the test) | "Linear or convex" reported if the isotonic R(k) stays within ±0.1 of k/12 at every k | Descriptive |
| **H3** | The required proportion depends on task type | Family × k, k ∈ {0, 3, 12} | D₃ = R_ret(3) − R_hop(3), per seed, at the primary cell, on the probability scale | D₃ > 0 | Paired one-sided t, Holm as H1. Supported iff the Holm-adjusted p < 0.05 **and**, where difficulty-equated level pairs exist (U-11), D₃ has the same sign at those levels. Evaluable only if every seed has G ≥ 15 pp in both families; otherwise p is set to 1 | "No task dependence" requires TOST within ±0.10 on D₃ | Confirmatory |
| **H4** | Placement matters independently of proportion | Placement ∈ {early, middle, late} at k* | c₁ = S_mid − ½(S_early + S_late); c₂ = S_late − S_early (primary cell, retrieval) | None pre-registered | Two-sided paired t, Holm within the secondary family | "No placement effect" requires both contrasts within TOST ±5 pp | Secondary |
| **H5(a)** | The benefit of full attention grows with context length… | T ∈ {1K, …, 32K} × k ∈ {0, 12} | Per-seed OLS slope b of Δ₀(T) = S₁₂(T) − S₀(T) on log₂T, over the **fixed level set** {R1 D ∈ {3, 15}; R-load N ∈ {8, 32}} at all six T | b > 0 | Paired one-sided t, Holm as H1. Labels: "conditional on reference adequacy" if any T fails G-REF; "exposure-robust" if the zero-shot LM-4 slope (§9.6) has the same sign | "No length effect" requires TOST within ±2 pp per doubling | Confirmatory |
| **H5(b)** | …differently per task | Family × T | Per-seed difference b_ret − b_hop (multi-hop fixed set h ∈ {1, 2}) | None pre-registered | Two-sided paired t, Holm within the secondary family | TOST ±2 pp per doubling | Secondary |
| **H6** (literal) | Hybrids occupy a capability–efficiency region neither pure architecture reaches | k ∈ {1, 2, 3, 6} vs endpoints | Hull margin M(k) = G·(R(k) − k/12), for any per-layer-additive cost | — | **Descriptive** (the identity is exact; I-16) | — | Descriptive |
| **H6′** | Is recurrent state worth more than KV cache? | GDN-E k = 0 vs GDN-A k = k* | Δ_E = S(GDN-E, 0) − S(GDN-A, k*), at the primary cell | None pre-registered | Two-sided Welch t (2 vs n seeds); pooled-variance version as sensitivity; state bytes reported alongside | "Equivalent" requires TOST within ±5 pp | Secondary |

**Secondary estimands** (reported, never decisive):
- non-inferiority Δ = S₁₂ − S₃, margin 5 pp;
- R with a Fieller CI;
- **H1-robust:** R ≥ 0.8 at every informative ladder level;
- **R_best:** recovery relative to the best configuration, chosen on dev items;
- L_k on gold log-probability. A verdict is labelled "metric-robust" only if this agrees in sign;
- per-configuration success counts with Clopper–Pearson CIs (§11.5);
- N50 capacity (§9.7).

**Predictions written down now** [HYP], to be checked later and not used in any test:
- R_ret(1) is low, given a threshold-like GDN prior (Wang et al. GDN recall 0.356 / 0.341 / 0.352 / 0.425 / 0.436 for pure / 24:1 / 12:1 / 6:1 / 3:1; [2507.06457](https://arxiv.org/abs/2507.06457) Table 7 [read-section per A1, A2, B2]). Wang's numbers are never used for power or ceiling planning (I-23).
- R_ret(3) is somewhere in 0.6–1.1.
- R_hop(3) < R_ret(3).
- Δ₀(T) grows with T for retrieval.
- GDN-E at k = 0 closes less than half of the k = 0 → k* gap at 32K.

---

## 3. Formal model family

### 3.1 Definition

- **Architecture vector.** a = (a₀, …, a₁₁) ∈ {F, Λ}¹². F is a full-attention mixer and Λ a linear mixer. Then k(a) = Σ 1[a_l = F], and the placement is P(a) = {l : a_l = F}.
- **Model.** The embedding E ∈ ℝ^{V×d} is tied with the output.
  - h⁰ = E[x].
  - For each layer l: u = h^l + Mix_{a_l}(RMSNorm₁(h^l)), then h^{l+1} = u + SwiGLU(RMSNorm₂(u)).
  - logits = RMSNorm_f(h¹²)·Eᵀ.
  - RMSNorm has eps 1e-5 and a learnable gain. There are **no biases** anywhere.
  - SwiGLU(z) = W_down(SiLU(W_gate z) ⊙ W_up z).
- **The only per-layer difference is Mix_{a_l}.** Norms, MLPs, embedding, initialisation scheme, optimiser, data, data order, schedule and TEF are identical across k.

**Scales.** Exactly one is chosen by rule U-01 (§13.3). The other is used only as an optional Tier-C "scale extension".

| Scale | d | d_ff | V (tied) | Full mixer F_g heads | GDN-A heads (d_k / d_v) | Seeds n₀ (anchors, placement) | Stage-A tokens | Stage-B options (FEAS-3) |
|---|---|---|---|---|---|---|---|---|
| **D384** | 384 | 1,024 | 16,384 | 6 × 64 | 3 × (64 / 128) | 5 (+2 on trigger) | 0.6B (4,578 steps) | 30M / 60M / 90M (229 / 458 / 687 steps) |
| **D512** | 512 | 1,536 | 16,384 | 8 × 64 | 4 × (64 / 128) | 4 (+2 on trigger) | 0.8B (6,104 steps) | 0.1B / 0.2B / 0.3B (763 / 1,526 / 2,289 steps) |

### 3.2 Layer patterns

Notation: `-` = GDN-A, `A` = F_g, `C` = F_gc, `E` = GDN-E. Layers are written 0 → 11, left to right.

**Composition, centred-uniform:** p_j = ⌊(j + ½)·12/k⌋. Layer 0 is linear whenever k < 12.

| Config | k | Positions | Pattern | Tier |
|---|---|---|---|---|
| C0 | 0 | ∅ | `------------` | A |
| C1 | 1 | {6} | `------A-----` | A |
| C2 | 2 | {3, 9} | `---A-----A--` | B (curve) |
| C3 | 3 | {2, 6, 10} | `--A---A---A-` | A |
| C4 | 4 | {1, 4, 7, 10} | `-A--A--A--A-` | C (optional) |
| C6 | 6 | {1, 3, 5, 7, 9, 11} | `-A-A-A-A-A-A` | B (curve) |
| C12 | 12 | all | `AAAAAAAAAAAA` | A |

**Placement at k*** (I-15). k* is the smallest k ∈ {1, 2, 3} with seed-pooled R̂_ret(k) ≥ 0.3 at the primary cell, using composition runs only. If none qualifies, k* = 3. Spacing is preserved across arms; "late" is the industrial end-of-group convention.

| k* | Early | Middle (= composition arm) | Late |
|---|---|---|---|
| 1 | {1} `-A----------` | {6} `------A-----` | {11} `-----------A` |
| 2 | {1, 7} `-A-----A----` | {3, 9} `---A-----A--` | {5, 11} `-----A-----A` |
| 3 | {1, 5, 9} `-A---A---A--` | {2, 6, 10} `--A---A---A-` | {3, 7, 11} `---A---A---A` |

**Controls.** Each uses the composition pattern of its k, with the mixer swapped as shown.

| Code | What changes | k | Tier |
|---|---|---|---|
| X-gc12 | F_g → F_gc (pattern `CCCCCCCCCCCC`) | 12 | A (2 seeds, unconditional) |
| X-gc1, X-gc3 | F_g → F_gc | 1, 3 | B, conditional on U-02 |
| X-nope3, X-nope12 | NoPE on all full layers | 3, 12 | A (seed 0) |
| X-E0, X-E{k*} | GDN-A → GDN-E | 0, k* | B (2 seeds) |
| O-gla0, O-gla{k*}, O-gla3 | GDN-A → GLA | 0, k*, 3 | C (promotion rule U-13) |
| O-nc0, O-nc{k*} | GDN-A → GDN without its short conv | 0, k* | C |
| O-twin12 | F_g → Primer-style twin (4 × 64 / 128, conv on q/k/v, GDN-identical gated output) | 12 | C |

### 3.3 Mixer definitions and parameter counts

GDN, GLA and conv-free counts are by meta-device instantiation of fla 0.6.0 [PI-VER]. F_g and F_gc counts are analytic. Script: `/tmp/claude-1000/-home-sedrica-lynx-Downloads-outeropt/73b71b1c-85a4-4704-b75f-7c9b668d4052/scratchpad/protocol/param_counts.py`.

| Mixer | Definition | Params D512 | Params D384 | Per-layer inference state |
|---|---|---|---|---|
| **F_g** (primary full) | MHA d/64 heads × 64. W_Q, W_K, W_V, W_O ∈ ℝ^{d×d}. Per-head **unshared** QK-RMSNorm gains (2·H·64). RoPE θ = 5e5 after QK-norm; positions reset per document. Scale 1/√64. Causal softmax over all previous tokens of the document. **Headwise sigmoid output gate:** o_h ← σ(x·W_g)_h·o_h, W_g ∈ ℝ^{d×H}, applied before W_O. No conv, no dropout. Optional SSMax (§6.6) | **1,053,696** | **592,896** | KV bf16: 4d B/token (2,048 / 1,536) |
| **F_gc** (conv control) | F_g plus causal depthwise ShortConv(4), no bias, on q, k, v after the projections and before QK-norm/RoPE. SiLU on v only, none on q/k. Conv respects document boundaries | 1,059,840 | 597,504 | KV + conv cache 3·d·4·4 B |
| **GDN-A** (primary linear) | `fla.layers.GatedDeltaNet(hidden_size=d, head_dim=64, num_heads=d//128, expand_v=2, use_gate=True, use_short_conv=True, conv_size=4, conv_bias=False, allow_neg_eigval=False, norm_eps=1e-5, mode='chunk')`. Update S_t = S_{t−1}·α_t(I − β_t k_t k_tᵀ) + β_t v_t k_tᵀ; output W_o[RMSNorm(o_t) ⊙ SiLU(x W_g)] | **1,056,904** | **595,334** | fp32 state H·64·128·4 B + fp32 conv cache (H·256)·4·4 B = 147,456 B (D512) / 110,592 B (D384) |
| **GDN-E** (H6′ arm) | As GDN-A but `num_heads=1, head_dim=d//2`. Per-head d_k ×4 (D512) / ×3 (D384), at equal H·d_k | 1,054,210 | 594,050 | 540,672 B / 307,200 B |
| GDN-nc (Tier C) | GDN-A with `use_short_conv=False` | 1,052,808 | 592,262 | 131,072 B / 98,304 B |
| GLA (Tier C) | `fla.layers.GatedLinearAttention(hidden_size=d, num_heads=d//128, expand_k=0.5, expand_v=1.0, use_short_conv=True, conv_size=4, use_output_gate=True)` | 1,065,344 | 602,432 | ≈ GDN-A |

**Fallback if GDN-E fails G0:** config D at D512 (2 × 128 / 256; 1,054,980 params). At D384 no parameter-matched 2-head geometry with power-of-two d_k exists; 2 × 96 would be tested in S0 only if needed. GDN-E was measured to run forward and backward on sm_120 at K = 256 and K = 192 [MEAS-B2].

**Sources.**
- GDN equations: [2412.06464](https://arxiv.org/abs/2412.06464) Eq. 10 [read-section per PROP].
- fla layer: `fla/layers/gated_deltanet.py` @`98193113ecb7835143252a786e06933e371b67cc` [code-inspected, PI-VER: constructor L91–108; `A_log`/`dt_bias` flagged `_no_weight_decay` at L155–166].
- Headwise gate: gated attention, [2505.06708](https://arxiv.org/html/2505.06708) Table 1. Headwise G1 adds 1.6M parameters to a 15B MoE: PPL 5.792 vs 6.026 baseline [read-section, PI-VER in debate]. The industrial precedent is the sigmoid gate in `Qwen3NextAttention` (transformers 5.22.2, `modeling_qwen3_next.py` L300) [code-inspected, PI-VER in debate].

### 3.4 Exact per-configuration table

**How to read the columns.**
- **FLOPs** are forward FLOPs per token at the training length: 4K for Stage A, 32K for Stage B.
  - FLOPs = 2·N_non-emb + 2·d·V + k·2·T·d (causal attention average) + (12 − k)·GDN core.
  - GDN core per head per token ≈ 6·d_k·d_v + 2C(3d_k + 2d_v) + C²/3, with chunk C = 64.
  - The values are nominal, i.e. for a single document. Document masking lowers the attention term in Stage A.
- **Training PFLOP per run** = 3 × fwd(4K) × Stage-A tokens + 4 × fwd(32K) × (Stage-B + TEF tokens). The 4 accounts for activation-checkpoint recompute. It uses the middle Stage-B option and is an upper bound.
- **State** = k·T·4d B (bf16 KV) + (12 − k)·S_lin, plus F_gc's conv cache. It is per sequence, in MiB.
- Script: `/tmp/claude-1000/-home-sedrica-lynx-Downloads-outeropt/73b71b1c-85a4-4704-b75f-7c9b668d4052/scratchpad/protocol/table.py` [INF].

| Scale | Config | Pattern (layer 0→11) | Non-emb params | Emb (tied) | Total | Fwd MFLOP/tok @4K | @32K | State MiB @1K | @4K | @16K | @32K | Train PFLOP/run |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| D512 | C0 | `------------` | 41,007,200 | 8,388,608 | 49,395,808 | 104.0 | 104.0 | 1.69 | 1.69 | 1.69 | 1.69 | 360 |
| D512 | C1 | `------A-----` | 41,003,992 | 8,388,608 | 49,392,600 | 107.7 | 137.1 | 3.55 | 9.55 | 33.55 | 65.55 | 404 |
| D512 | C2 | `---A-----A--` | 41,000,784 | 8,388,608 | 49,389,392 | 111.5 | 170.2 | 5.41 | 17.41 | 65.41 | 129.41 | 448 |
| D512 | C3 | `--A---A---A-` | 40,997,576 | 8,388,608 | 49,386,184 | 115.2 | 203.3 | 7.27 | 25.27 | 97.27 | 193.27 | 493 |
| D512 | C4 | `-A--A--A--A-` | 40,994,368 | 8,388,608 | 49,382,976 | 119.0 | 236.4 | 9.12 | 33.12 | 129.12 | 257.12 | 537 |
| D512 | C6 | `-A-A-A-A-A-A` | 40,987,952 | 8,388,608 | 49,376,560 | 126.5 | 302.7 | 12.84 | 48.84 | 192.84 | 384.84 | 625 |
| D512 | C12 | `AAAAAAAAAAAA` | 40,968,704 | 8,388,608 | 49,357,312 | 149.0 | 501.4 | 24.00 | 96.00 | 384.00 | 768.00 | 890 |
| D512 | X-gc12 | `CCCCCCCCCCCC` | 41,042,432 | 8,388,608 | 49,431,040 | 149.2 | 501.5 | 24.28 | 96.28 | 384.28 | 768.28 | 891 |
| D512 | X-E0 | `EEEEEEEEEEEE` | 40,974,872 | 8,388,608 | 49,363,480 | 110.9 | 110.9 | 6.19 | 6.19 | 6.19 | 6.19 | 384 |
| D384 | C0 | `------------` | 21,309,384 | 6,291,456 | 27,600,840 | 59.1 | 59.1 | 1.27 | 1.27 | 1.27 | 1.27 | 136 |
| D384 | C1 | `------A-----` | 21,306,946 | 6,291,456 | 27,598,402 | 61.9 | 83.9 | 2.66 | 7.16 | 25.16 | 49.16 | 154 |
| D384 | C2 | `---A-----A--` | 21,304,508 | 6,291,456 | 27,595,964 | 64.7 | 108.8 | 4.05 | 13.05 | 49.05 | 97.05 | 171 |
| D384 | C3 | `--A---A---A-` | 21,302,070 | 6,291,456 | 27,593,526 | 67.5 | 133.6 | 5.45 | 18.95 | 72.95 | 144.95 | 189 |
| D384 | C4 | `-A--A--A--A-` | 21,299,632 | 6,291,456 | 27,591,088 | 70.4 | 158.4 | 6.84 | 24.84 | 96.84 | 192.84 | 206 |
| D384 | C6 | `-A-A-A-A-A-A` | 21,294,756 | 6,291,456 | 27,586,212 | 76.0 | 208.1 | 9.63 | 36.63 | 144.63 | 288.63 | 241 |
| D384 | C12 | `AAAAAAAAAAAA` | 21,280,128 | 6,291,456 | 27,571,584 | 92.9 | 357.1 | 18.00 | 72.00 | 288.00 | 576.00 | 347 |
| D384 | X-gc12 | `CCCCCCCCCCCC` | 21,335,424 | 6,291,456 | 27,626,880 | 93.0 | 357.2 | 18.21 | 72.21 | 288.21 | 576.21 | 347 |
| D384 | X-E0 | `EEEEEEEEEEEE` | 21,293,976 | 6,291,456 | 27,585,432 | 62.6 | 62.6 | 3.52 | 3.52 | 3.52 | 3.52 | 144 |

**Parameter matching.**
- N(k) = 41,007,200 − 3,208·k at D512, a spread of 38,496 (0.094%).
- N(k) = 21,309,384 − 2,438·k at D384, a spread of 29,256 (0.137%).
- Placement arms have the same counts as C{k*}. NoPE arms equal their C{k}.
- Break-even context, where one F_g layer's KV equals one GDN-A layer's state: 72 tokens at both scales [INF].
- **Tokens per non-embedding parameter:** D384 (0.6B + 0.06B)/21.3M ≈ 31; D512 (0.8B + 0.2B)/41.0M ≈ 24. Both are before the TEF (+65.5M).

---

## 4. Linear mechanism

**Primary: Gated DeltaNet "config A" (GDN-A).** Reasons [EST evidence, INF choice]:
1. **Strongest standalone recall among candidates.** Jet-Nemotron block search retrieval: GDN 69.3 vs Mamba2 68.9 ([2508.15884](https://arxiv.org/abs/2508.15884) [read-section per PROP]). Pure-linear RULER: GDN 0.356 vs DeltaNet 0.320 ([2507.06457](https://arxiv.org/abs/2507.06457) [read-section per PROP]).
2. **Industrial use.** Qwen3-Next is a GDN 3:1 hybrid (transformers 5.22.2 config `full_attention_interval` = 4 [code-inspected per A2]).
3. **Kernels.** Pure-Triton fla kernels are verified on sm_120 for forward, backward, varlen and decode (05 §2 [MEAS]). Mamba-2's fast path needs `mamba_ssm`, which has no torch-2.11 wheel (05 §4.4 [code-inspected per 05]).
4. **Parameter matching** to F_g within 0.30% per mixer, with an identical MLP.
5. **State-to-width ratio d²/8**, the same as Qwen3-Next at d = 2048 (PROP §4 [config-inspected per PROP]).

**Rejected alternatives.**
- Gate off with d_k = 128: drops a published component.
- 4 heads × 128: forces the MLP width to differ by block type.
- A "twin" as primary (I-01): rejected, but available in Tier C.

**Secondary mechanisms** (Tier C unless promoted by U-13):
- GLA at k ∈ {0, k*, 3} × 2 seeds;
- conv-free GDN at k ∈ {0, k*} × 2 seeds.

**Scope:** every claim is scoped "for GDN-A" (I-17).

**Mechanism-relevant facts used later** [INF / PI-SIM]:
- In the PI's delta-rule simulation, per-head d_k sets interference. Mean cosine to the true value at N = 32 / 128 / 512 facts:
  - 0.84 / 0.50 / 0.23 for 4 × 64 heads;
  - 0.96 / 0.84 / 0.64 for 1 × 256.
- Degradation is gradual, so ladder levels are *not* phase boundaries (I-07, debate §6.3).
- The delta rule overwrites rather than accumulates, so counting is mechanism-specific and is a diagnostic only (I-14).

---

## 5. Fixed vs varied factors

| Factor | Status | Value / note |
|---|---|---|
| k (number of full layers of 12) | **Varied (primary IV)** | {0, 1, 3, 12} Tier A; {2, 6} Tier B; 4 optional |
| Placement at fixed k | **Varied (RQ-B only)** | early / middle / late at k* |
| Full-mixer variant | Varied in controls only | F_gc (U-02); NoPE (U-06); twin (Tier C) |
| Linear-mixer variant | Varied in controls only | GDN-E (H6′); GLA and GDN-nc (Tier C) |
| L, d, d_ff, V, tokenizer, norms, MLP | Fixed | §3.1 |
| Parameters | **Matched** | Non-embedding spread ≤ 0.094% (D512) / 0.137% (D384); controls ≤ 0.2% |
| Data, data order, tokens per stage | Fixed | Identical stream for every k at a given seed index (§6.4) |
| Synthetic exposure (TEF) | Fixed, identical | Same items, order and schedule for every k at a given seed (§7.3) |
| Optimiser and schedule | Fixed | §7. LR: shared η* or a per-config rule (§7.2) |
| Initialisation scheme | Fixed | Shared modules are bit-identical across k for a given seed (§8.1) |
| Position encoding | Fixed | RoPE θ = 5e5 on full layers in every arm, from step 0 (NoPE only in the control) |
| Precision and kernels | Fixed | bf16 autocast; fp32 master weights and GDN state; fla TF32 path disclosed |
| Evaluation items | Fixed | Byte-identical frozen item files (SHA-256) for every model |
| **Compute per token** | **Not matched** (limitation, I-28) | Attention FLOPs grow with k·T, e.g. 104 → 501 MFLOP/token at 32K (D512). Wall-clock differs (k = 12 Stage B runs ≈ 2.2× slower than k = 0 [MEAS-B]). "Unbounded memory" and "more compute" are not separated |
| Inference-state memory | Not matched (the RQ-D axis) | §3.4 |

---

## 6. Data, tokenizer, token budget, curriculum, packing, position encoding

### 6.1 Corpora, pinned

| Corpus | Use | Pin / licence |
|---|---|---|
| FineWeb-Edu `HuggingFaceFW/fineweb-edu`, config sample-10BT | Training: shards `sample/10BT/001_00000.parquet` onward (pre-tokenise 001–006). Held out: shard `000_00000` | Dataset commit `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9` (lastModified 2025-07-11), ODC-By ([HF API](https://huggingface.co/api/datasets/HuggingFaceFW/fineweb-edu) [abstract-only, PI-VER]) |
| PG19 via `emozilla/pg19` parquet mirror | Training: `data/train-000NN-of-00023-*.parquet`. Dev filler: `validation` (50 books). Test filler and LM-1: `test` (100 books) | Mirror commit `c021754c8e01c5b1cc83a1f549c1f97fbbb756b8` (lastModified 2023-10-09) ([HF API](https://huggingface.co/api/datasets/emozilla/pg19) [abstract-only, PI-VER]). The mirror's API exposes no licence field; Apache-2.0 per the `deepmind/pg19` card ([card](https://huggingface.co/datasets/deepmind/pg19) [abstract-only per 05]) |

**Held-out structure.**
- FineWeb-Edu shard 000 is split by SHA-256(doc_id) parity: even → test filler, LM-1 and LM-2; odd → dev filler.
- PG19 test → test filler and LM-1. PG19 validation → dev filler.
- Training PG19 books never appear in evaluation filler.
- A unit test asserts that the intersections of document hashes are empty (VT-11).
- Per-file SHA-256 values are recorded in `data/MANIFEST.json`.

### 6.2 Tokenizer

- **Type:** 16,384-entry byte-level BPE (HF `tokenizers` 0.23.3) with single-digit isolation (`Split(\d, isolated)` + `ByteLevel`).
- **Training text:** 1 GB, 80% FineWeb-Edu shard 001 and 20% PG19 train; sampling seed 0.
- **Special tokens:** `<|endoftext|>` (id 0; document separator/BOS), `<|pad|>` (id 1), `<|reserved_0…5|>` (ids 2–7).
- **Storage:** saved as `PreTrainedTokenizerFast`, with its SHA-256 in the registry.
- **Compression:** an equivalent 16K tokenizer gave +9.8% tokens vs GPT-2 on FineWeb-Edu and +10.1% on PG19 (05 §6 [MEAS]). All lengths in this protocol are in our tokens.
- **Answer encoding:** values are exactly 4 single-digit tokens. Any whitespace token before them belongs to the prompt. This is asserted by VT-12.

### 6.3 Curriculum and token budget (identical for every k)

| Phase | Length | Tokens (D384 / D512) | Mix | LR | Batch | Micro-batch |
|---|---|---|---|---|---|---|
| **A (stable)** | 4,096 | 0.6B / 0.8B (4,578 / 6,104 steps) | 85% FineWeb-Edu (document-masked packing) + 15% PG19-train 4K windows | 500-step linear warm-up to η*, then constant | 131,072 tokens (32 × 4K) | 2 × 4K flattened, no checkpointing; 16 accumulation steps |
| **S** | — | — | End-of-stable checkpoint, with optimiser state: the **branch point** | — | — | — |
| **B (decay)** | 32,768 | FEAS-3 choice; per-run costs assume 60M / 0.2B | 60% PG19-train 32K windows (book-bounded) + 40% FineWeb-Edu packed to 32K | Linear decay η* → 0 over B | 131,072 (4 × 32K) | 1 × 32K with per-block activation checkpointing (every k); 4 accumulation steps |
| **TEF** | Packed 32K | 65,536,000 (1,000 × 64K); seed-0 anchors continue to 4,000 steps | §7.3 | 50-step linear re-warm 0 → 3e-4, then constant 3e-4 | 65,536 (2 × 32K) | 1 × 32K with checkpointing |
| **2× Stage-B branch** | 32,768 | 2 × the FEAS-3 choice, from S | As B | Linear decay over the 2× steps | As B | As B |

**Mixing.** The proportions are exact per window: Stage A has 96 PG19 sequences per 640 sequences (20 steps), Stage B 12 PG19 per 20 sequences (5 steps). Within a window the order comes from a deterministic interleave keyed on (data_seed, step).

**Logging.** Effective-context histograms (same-document tokens available at each position) are logged per stage. FineWeb-Edu is long-poor: 59% of its tokens lie in documents of ≤ 2,048 GPT-2 tokens (05 §5.1 [MEAS]). The 15% PG19 share in Stage A mitigates this (I-12).

### 6.4 Packing and masking

- **EOS-separated sequential packing.** Each document is preceded by `<|endoftext|>`.
- **Long documents** are cut at sequence boundaries. Each cut piece is its own document segment.
- **Intra-document masking in both mixers:**
  - attention: `torch.nn.attention.varlen.varlen_attn(q, k, v, cu_seq_q, cu_seq_k, max_q, max_k, window_size=(-1, 0))`, the FA2 kernel inside torch 2.11 ([signature PI-VER]);
  - GDN: `cu_seqlens` resets both the state and the short-conv state; isolation was measured exact (05 §2.2 [MEAS]).
- **RoPE positions** are reset per document segment.
- **Data order** is a pure function of (data_seed = seed index, step), so a resumed run reproduces its batches.
- **Batch-index SHA-256** values are logged at every step.

### 6.5 Position encoding

- RoPE θ = 500,000 on every full-attention layer of every arm from step 0, with no pilot (I-09).
  - Llama 3 states: "We increase the RoPE base frequency hyperparameter to 500,000", which "showed this value to be effective for context lengths up to 32,768" ([2407.21783 §3.2](https://arxiv.org/html/2407.21783) [read-section, PI-VER]).
- Full rotary dimension 64. Applied after QK-RMSNorm.
- Linear layers have no explicit position encoding.
- **NoPE control:** X-nope3 and X-nope12 at seed 0 (Tier A). Seed 1 is added if |ΔR₃| > 0.1 or |ΔS₁₂| > 3 pp at the primary cell (U-06).
  - Motivation: NoPE on the full layers raised NIAH@16K from 49.58 to 67.81 in a 0.22B **sliding-window** hybrid ([2606.15378v1](https://arxiv.org/html/2606.15378v1) Table 2 [read-section, PI-VER in debate]).

### 6.6 QK-norm temperature (U-03)

- Per-head unshared gains are always used.
- **SSMax** ([2501.19399](https://arxiv.org/html/2501.19399) [read-section, PI-VER]) defines z_i ↦ n^{s z_i}/Σ_j n^{s z_j} (Eq. 2). It is implemented by scaling the query by s·log n, with s learnable per layer and head.
  - Here: q_t ← q_t·s_h·ln(t + 1), where t is the 0-based within-document position, s_h is learnable (no weight decay) and initialised to 1/ln 4096. The multiplier is then 1.00 at 4K and 1.25 at 32K.
- **Adoption rule** (FEAS, k = 12, seed 101, 30M/0.1B branch, TEF 1,000): adopt iff, on the R1/R-load dev ladder mean, it gains ≥ 3 pp at 32K **and** loses ≤ 2 pp at 4K.
- If adopted, it applies identically to every attention layer of every arm. The non-chosen variant runs at k ∈ {3, 12} in Tier C.

---

## 7. Optimisation

### 7.1 Fixed hyperparameters

| Item | Value |
|---|---|
| Optimiser | `torch.optim.AdamW(fused=True)`, β = (0.9, 0.95), ε = 1e-8 |
| Weight decay | 0.1 (decoupled: p ← p·(1 − η_t·0.1)) on every parameter with ndim ≥ 2, including the tied embedding, conv kernels and W_g. **None** on 1-D tensors (all norm gains, including QK-norm and GDN o_norm), on anything flagged `_no_weight_decay` (`A_log`, `dt_bias`), or on SSMax s_h |
| Gradient clipping | Global L2 norm 1.0 |
| z-loss | 1e-4·mean(logsumexp(logits)²) over loss tokens |
| Loss | Next-token cross-entropy. Logits in fp32, chunked over 4,096-token chunks with recompute. The pad token (id 1) is ignored. Document separators are predicted (standard) |
| Precision | bf16 autocast; fp32 master weights and Adam moments; fp32 GDN state; `FLA_FLASH_QLA=0` |
| Initialisation | Every `nn.Linear` N(0, 0.02). Output projections (F_g W_O, GDN o_proj, MLP W_down) N(0, 0.02/√24). Embedding N(0, 0.02). Norm and QK-norm gains 1. W_g N(0, 0.02), so the gate starts near 0.5. fla-specific tensors (`A_log` ~ log U(0, 16), `dt_bias`, conv kernels, o_norm) keep fla's own initialisation, asserted by VT-10 |
| Compilation | No `torch.compile` in training or evaluation. It made fla hybrids slower and hit recompile limits (05 §2.8 [MEAS]). Only the FlexAttention benchmark comparator compiles |
| Memory cap | `torch.cuda.set_per_process_memory_fraction(0.44)`, ≈ 7.0 GiB. Measured peaks are ≤ 4.7 GiB in every stage [MEAS-B, MEAS-B2] |

### 7.2 LR selection (S3; I-26)

- **Grid:** k ∈ {0, 3, 12} × η ∈ {1.0, 1.4, 2.0, 2.8}×10⁻³ at the chosen scale.
- **Each run:** 1,145 steps × 131,072 tokens ≈ 0.15B at 4K, seed index 100.
- **Compressed WSD:** 115 warm-up steps, constant to step 916, linear decay to 0 over the last 229 steps.
- **Metric:** final LM-2 subset loss (4K; 2,000 × 4K held-out sequences). **Never task metrics.**
- **Rule** (minimax regret):
  - regret_k(η) = loss_k(η) − min_η′ loss_k(η′);
  - η* = argmin_η max_k regret_k(η);
  - **shared η*** iff max_k regret_k(η*) ≤ 0.02 nats; otherwise per-config η*_k, with log η interpolated linearly in k for k ∉ {0, 3, 12}.
- **Divergence:** NaN, or final loss > min + 0.5, counts as infinite regret.
- **Edge extension:** if any optimum lies at a grid edge, add 0.7e-3 or 4.0e-3 for all three k and re-apply the rule.
- **Inheritance:** controls and Tier-B/C arms inherit η* of their k.
- **FEAS** runs before the sweep at the **provisional η = 2.0e-3** for k ∈ {0, 12}. If a FEAS run diverges, it restarts once at 1.4e-3 (D034).

### 7.3 Task-exposure fine-tune (TEF; I-11, U-08, U-09)

The TEF is identical for every configuration: the same item stream, keyed on the seed index, and the same schedule.

- **Start:** from the end of Stage B (LR 0), continuing the Adam moments.
- **Schedule:** 50-step re-warm to 3e-4, then constant.
- **Item lengths:** 7 buckets ℓ ∈ {512, 1K, 2K, 4K, 8K, 16K, 32K}, with **equal item counts per bucket and per family**. Each item is exactly ℓ tokens, including its queries.
- **Synthetic share:** **90% synthetic / 10% natural replay** by default, which gives ≈ 928 items per bucket per 1,000 steps [INF: 0.9 × 65.5M / 63.5K]. Replay documents use the Stage-B mix and fill packing gaps.
- **Item families and generators** (training namespaces):
  - **Retrieval:** R1-type 40%, R-load-type 40%, R3-type 10%, distance-type 10%.
  - **Multi-hop:** M-hop 100% (link order forward / shuffled / reversed = 0.25 / 0.5 / 0.25).
  - **Aggregation:** A-freq 70%, A-count 30%.
  - **Training ranges are wider than evaluation:** D ∈ U{0..20}; N log-uniform in [4, 2,048] subject to fact tokens ≤ 0.75ℓ; h ∈ U{1..6}; ρ ∈ U[1.1, 5]; U ∈ U{1..5}.
  - **Leakage controls:** word list A (eval uses B); templates T1–T3 (eval uses T1, plus a T4 slice); filler from training corpora only; 10% of items use repetitive filler.
  - **Queries:** each item has 4 queries appended sequentially after its context.
- **Loss (D014):** L_TEF = ½·mean CE over answer tokens of synthetic items + ½·mean CE over all other tokens in the step.
- **Checkpoints and evaluation:**
  - at 250 / 500 / 1,000 steps for every run (dev-checkpoint suite, §9.5);
  - seed-0 anchors (C0, C1, C3, C12) continue at constant LR on the same stream to **4,000 steps**, with checkpoints at 2,000 and 4,000;
  - Tier 0 (§9.6) is measured before the TEF; LM-2 before and after.
- **"Still closing" flag:** if S₁₂ − S_k at the primary cell changes by > 3 pp between TEF steps 1,000 and 4,000 (seed 0), that k's verdict is "budget-conditional".
- **FEAS rules for the TEF:**
  - **Forgetting gate:** LM-2 may rise by ≤ 0.05 nats for each of k ∈ {0, 12}, and the rises may differ by ≤ 0.02 nats → keep 90/10. Otherwise switch to 50/50; if that also fails, use TEF LR 1e-4 at 50/50 (final).
  - **Elicitation check:** if k = 12's dev score on R1 D = 3 at 4K is < 50% at TEF 1,000 for both FEAS seeds, switch to answer-only loss and re-run the FEAS TEF branches. If it still fails, promote the direct-supervision screen (I-08) and consult the user.

---

## 8. Seeds, RNG streams and runs per stage

### 8.1 Streams

Every stream derives from `numpy.random.SeedSequence(entropy=PROJECT_SALT = 271828182, spawn_key=(seed_index, stream_id, …))` and is converted to a `torch.Generator`. No global RNG state is used in the data or evaluation paths.

| Stream (id) | Spawn key | Shared across k at the same seed index? |
|---|---|---|
| init_shared (0) | (s, 0, crc32(module_path)) for embedding, MLPs, norms, final norm | **Yes**: bit-identical shared modules (common random numbers) |
| init_mixer (1) | (s, 1, layer_idx, mixer_type_id), with F_g = 1, F_gc = 2, GDN-A = 3, GDN-E = 4, GLA = 5, GDN-nc = 6 | Yes, for the same mixer type at the same layer |
| data_A (2), data_B (3) | (s, 2), (s, 3) | Yes: identical order |
| tef (4) | (s, 4): item generation and packing order | Yes |
| eval_test | entropy 20261109, keys (task, level, T) | Constant (frozen files) |
| eval_dev | entropy 77 | Constant |
| bench | entropy 0 (random tokens and random-init weights for timing) | Constant |
| bootstrap | entropy 2026 | Constant |

**Seed-index assignment.**
- Core: 0 … n₀−1. Trigger seeds: n₀, n₀+1 (5–6 at D384, 4–5 at D512).
- LR sweep: 100. FEAS: 101, 102. S0 audits: 900–902. S1: Zoology's own seeding (logged).
- **No seed is ever dropped or replaced.**
- A run that fails permanently (two divergences) is reported. Its seed block is incomplete; it counts as a failure in success counts. If fewer than 4 complete blocks remain, the next unused seed index is added for all four anchors. This rule is effect-blind.

### 8.2 Runs per stage

LM training runs; branches count as runs.

| Stage | D384 path | D512 path |
|---|---|---|
| S0 | Audit micro-runs only (3 configs × 200 steps × 2 processes, plus resume tests) | Same |
| S1 | ≤ 22 Zoology-scale runs (§17) | Same |
| S2 FEAS | 5 Stage-A + 9 Stage-B branches + 9 TEF branches (§13.3); conditional TEF reruns ≤ 4 | FEAS-384 as left, plus FEAS-512: 3 Stage-A + 5 Stage-B + 5 TEF |
| S3 LR | 12 (+3 per edge extension) | 12 (+3) |
| S4 core | 20 (k ∈ {0, 1, 3, 12} × 5) | 16 (× 4) |
| Trigger (U-05) | +8 | +8 |
| S5a controls | 4 (X-gc12 × 2, X-nope3, X-nope12) | 4 |
| S5b branches | 4 TEF-4000 continuations + 3 2×-Stage-B branches | Same |
| Tier B | 4 curve + 10 placement + 4 GDN-E (+ ≤ 9 conditional) | 4 + 8 + 4 (+ ≤ 8) |

---

## 9. Evaluation suite

### 9.1 Shared mechanics (I-07, I-29)

- **Cloze completions, not instructions.** Every sample is exactly T tokens, built in token space from pre-tokenised segments that are never round-tripped through text.
  - Needles snap to a sentence boundary within ±32 tokens. Actual positions, depth = start/F and distance d = query_start − needle_end are recorded.
  - Depths are stratified uniformly within deciles.
- **No document separator inside an evaluation context** (D039). Filler is either one held-out PG19-test span or held-out FineWeb-Edu documents joined by "\n\n", 50/50 per context, with the source recorded.
  - Training never lets attention cross a separator, so a context must be one document.
  - Repetitive filler (R-dist only) is "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again." repeated.
- **Templates.**
  - T1 fact: "The code for {key} is {v}." Query: "The code for {key} is" → v.
  - T4 (unseen, diagnostic slice): "{key}'s code: {v}." Query: "{key}'s code:".
  - T2–T3 are TEF paraphrases fixed in `src/eval/templates.py`.
- **Keys and words.**
  - Keys are "{adj} {noun}" built from the `wonderwords` nounlist/adjectivelist that RULER uses (`scripts/data/synthetic/niah.py` L97–98 @`db43f92ff634bbce1c33d1fb15858aec4bba509a` [code-inspected, PI-VER]; `wonderwords` is pinned in `uv.lock` at S0).
  - Words are filtered to those that are single tokens with a leading space, then split into lists A and B by SHA-256 parity.
  - RULER's `json/english_words.json` is a Git-LFS pointer in our clone (oid sha256:affcd6d…, 8,564,991 B) and is not used [PI-VER].
- **Values** are uniform 4-digit strings. Train/eval disjointness follows from the disjoint keys. The hashed (key, value, template) tuples are asserted disjoint (VT-12).
- **Query branching:** one prefill per context, a snapshot of KV cache and recurrent state, then Q = 4 queries scored from the snapshot.

### 9.2 Metrics and scoring path

- **Per query:**
  - **EM_tf** (primary): every answer token is the argmax given the gold prefix.
  - **logp_gold**: summed log-probability of the answer tokens.
  - **Rank accuracy**: the candidates are gold ∪ ≤ 32 in-context values ∪ 4 absent values, of equal token length; chance is 1/|C|.
  - **Error types**: binding (top candidate is a wrong in-context value), absent, and stale (R3).
- **A-freq:** EM_tf on the winner token; two-alternative rank accuracy (chance 0.5) as secondary.
- **M-hop:** rank among the 4 chain values ∪ 4 absent values; "wrong-chain" rate.
- **Canonical scoring path:**
  - Prefill uses the fla chunk op and SDPA-flash, pinned.
  - Queries use **explicit** `fused_recurrent_gated_delta_rule(...)` calls with keyword arguments only, and SDPA over the cached KV with cuDNN or flash pinned.
  - This avoids fla's automatic switch to `fused_recurrent` when q_len ≤ 64 without grad (`fla/layers/gated_deltanet.py` L227–230 [code-inspected, PI-VER]).
- **Re-scoring check:** 5% of items are re-scored with a single-pass forward; the EM flip rate must be < 1% (I-29).
- **Batching:** evaluation batch sizes are frozen per length (B = 8 for T ≤ 4K, 2 for 8K–16K, 1 for 32K) for every model.

### 9.3 Confirmatory suite (CS)

The CS runs on every Tier-A/B model at TEF 1,000 (test generator, seed 20261109). "Contexts × Q" is per cell.

| Family | Task | Levels | Lengths | Contexts × Q |
|---|---|---|---|---|
| Retrieval | **R1**: facts with same-format distractors | D ∈ {3, 15} (D + 1 facts; 4 queried) | 1K, 2K, 4K, 8K | 200 × 4 |
| | | D ∈ {3, 15} | **16K, 32K** | **300 × 4** (600 if U-15 fires) |
| | **R-load**: N facts spread through natural filler, all keys distinct, fact tokens ≤ 0.75T (asserted) | N ∈ {8, 32} | 1K, 2K, 4K | 200 × 4 |
| | | N ∈ {8, 32, 128, 512} | 8K | 200 × 4 |
| | | N ∈ {8, 32, 128, 512, 1,024} | **16K, 32K** | **300 × 4** (600 if U-15 fires) |
| | | N = 2,048 (N50 only) | 32K | 150 × 4 |
| Multi-hop | **M-hop**: chain "The code for {k_i} is the code for {k_{i−1}}." ending in "The code for {k_0} is {v}."; 4 chains per context, each queried once; links shuffled and spread | h ∈ {1, 2} | 1K, 2K, 4K | 200 × 4 |
| | | h ∈ {1, 2, 3, 4} | 8K, 16K, 32K | 200 × 4 |
| Aggregation | **A-freq**: comma-separated word list; 4 probe pairs, n_Y = 8, n_X = round(8ρ); background words ≤ 3 occurrences each; query "Which word appears more often, {X} or {Y}? Answer:" | ρ ∈ {4, 2} | 1K, 2K, 4K | 200 × 4 |
| | | ρ ∈ {4, 2, 1.5, 1.25} | 8K, 16K, 32K | 200 × 4 |
| Controls | **C0**: the same item generator with no filler, per level (not per length) | All CS levels except N = 2,048 | — | 200 × 4 |

**Why these sizes** [INF]. At 300 contexts × 4 queries and a design effect of 1.6, the item-level SE of the per-seed primary-cell score is ≈ 0.5 pp (two lengths) to 0.7 pp (one length). That is small next to the expected seed SD of 2–3 pp, which therefore dominates. U-15 still raises the confirmatory retrieval cells to 600 contexts if the FEAS design effect exceeds 2.

### 9.4 Mechanism and diagnostic suite (MS)

The MS runs on C0 and C12 (seeds 0–1) and on C1, C3, X-gc12 and X-nope (seed 0). It is never confirmatory, except H5(b), which uses CS cells only.

| Task | Spec | Lengths | Contexts × Q | Purpose |
|---|---|---|---|---|
| R-dist × filler | 4 targets at d ∈ {256, 2K, 8K, 30K} + 12 distractors; filler natural vs repetitive | T = 32K | 200 × 4 per filler | H5 mechanism, analysed per architecture (S₀ and S₁₂ separately) |
| Load × length | R-load at fixed N across 8K, 16K, 32K (from CS) | — | — | Load vs length |
| R3 updates | 4 keys, each updated U times; latest value queried; stale rate | 4K, 16K, 32K; U ∈ {2, 4} | 150 × 4 | Overwrite and recency |
| C2 oracle position | Relevant facts placed right before the query: R1 D = 3, R-load N = 32, M-hop h = 2, A-freq ρ = 2 | 4K, 32K | 150 × 4 | Long input vs distance |
| A-count | c ∈ 1…9, 4 distractor animals | 8K, 32K | 150 × 4 | Delta-rule overwrite diagnostic |
| M-order | h = 2, forward vs reversed | 16K | 150 × 4 | Link order × recurrence |
| Top levels | M-hop h = 5; A-freq ρ = 1.1 | 32K | 150 × 4 | Headroom |
| T4 slice | R1 D = 3, M-hop h = 2, A-freq ρ = 2 | 8K, 32K | 100 × 4 | Template transfer |
| Attention diagnostics (k ≥ 1) | Per layer and head: needle attention mass from the query, entropy, effective maximum logit | 32K dev, R1 D = 3 | 50 | QK-norm dilution (U-03) |
| Pathway diagnostics (exploratory) | Window full layers to 1K vs reset GDN state every 1K; mean-ablate each attention layer | R-dist natural | 100 | Which pathway carries long-range information |

### 9.5 Dev suites (generator seed 77)

- **DG (dev gate):**
  - Cells: R1 D ∈ {3, 15} and R-load N ∈ {8, 32, 128, 512, 1,024} at 16K and 32K, plus their C0 counterparts; 200 × 4 each.
  - Scored on C0 and C12, all seeds. Scored on all core configurations only when the reference-inversion check is needed (§11.2).
  - M-hop h ∈ {1, 2, 3, 4} at 16K and 32K (200 × 4) is scored on C12 only, for the difficulty-equating check (U-11).
- **DC (dev checkpoint):** R1 D = 3, R-load N ∈ {32, 128}, M-hop h = 2, A-freq ρ = 2 at 4K, 16K and 32K; 100 × 4. Used for TEF 250/500 (all runs) and 2,000 (seed-0 anchors), and FEAS 250-step checkpoints.
- **FD (FEAS dev):**
  - Retrieval: R1 D ∈ {3, 15} at 4K, 8K, 16K, 32K; R-load {8, 32} at 4K, {8 … 512} at 8K, {8 … 1,024} at 16K and 32K; C0 for all retrieval levels.
  - Other families: M-hop h ∈ {1 … 4} and A-freq ρ ∈ {4 … 1.25} at 8K and 32K.
  - 100 × 4 per cell. Plus LM-2 before and after the TEF, and attention diagnostics for k = 12.

### 9.6 Tier 0: zero-shot, on the Stage-B checkpoint before the TEF

| ID | Metric | Data and spec |
|---|---|---|
| LM-1 | Per-position loss | Up to 300 FineWeb-Edu shard-000 test-half documents with ≥ 32,768 tokens (count reported) plus 100 PG19-test books (first 32,768 tokens). Log-spaced buckets [0, 64) … [16K, 32K). Long-context gain L[256, 512) − L[16K, 32K); tail slope on log position over [1K, 32K) |
| LM-2 | 4K validation loss | 2,000 × 4K held-out sequences, packed like Stage A |
| LM-3 | Repeat-token loss by distance | 4-gram repeats on LM-1 documents, excluding 4-grams in the top 1% of a 100M-token training sample. Δ buckets [64, 256), [256, 1K), [1K, 4K), [4K, 16K), [16K, 32K) |
| LM-4 | Copy-after-gap | P + G + P. P is a 256-token held-out passage (a) or 256 random tokens from a fixed 4,096-token vocabulary subset (b). Gap G ∈ {0, 512, 2K, 8K, 30K}; 200 items per gap per variant. Copy gain = loss on tokens 9–256 of the first P minus the same on the second P. **H5 corroboration:** Λ(g) = gain₁₂(g) − gain₀(g), slope on log₂g over g ∈ {512, 2K, 8K, 30K}; "exposure-robust" if its sign matches b |
| R0 | Cloze passkey (sanity check) | 100 items at 4K and 32K |

### 9.7 Derived quantities

- **N50:** logistic fit of EM_tf on log₂N per model and length, using the R-load levels (including N = 2,048 at 32K). Left- and right-censoring are reported.
- **N50₀** is the k = 0 model's N50. Levels are described as N/N50₀; there is no "phase boundary" wording (I-07).
- **Informative level:** S₁₂ between 20% and 90% and Ĝ ≥ 15 pp (dev items).
- **Difficulty-equated pairs** (U-11): (retrieval level, multi-hop level) with |ΔS₁₂| ≤ 10 pp on DG items at the primary length.

### 9.8 Raw output storage

- **Item-level results:** `experiments/results/<run_id>/<ckpt>/<suite>/<task>_<level>_<T>.jsonl`, one line per query. Fields:
  - item_id, context_id, query_idx, template, filler_source;
  - T, depth, distance, needle positions;
  - gold and argmax token ids, EM_tf;
  - logp_gold (sum and per token), candidate log-probabilities and rank;
  - error type, scoring path.
- **Header file** (`_meta.json`): git commit, config hash, item-file SHA-256, evaluation code SHA-256, library versions and wall time.
- **Never overwritten.** A re-run writes a new file with a suffix, plus a registry note.
- **Item files** (token ids + metadata) live in `experiments/items/{test,dev}/` with a SHA-256 manifest and are frozen at G5.

### 9.9 Evaluation compute

Prefill-time model calibrated to [MEAS-B] (flash 12.02 ms at 32K; GDN layer 2.55 ms at 32K), ×1.3 for contention, ×1.2 overhead. Script: `/tmp/claude-1000/-home-sedrica-lynx-Downloads-outeropt/73b71b1c-85a4-4704-b75f-7c9b668d4052/scratchpad/protocol/eval_budget2.py` [INF].

| Suite | Tokens per model | k = 0 / 1 / 3 / 12, D384 | D512 |
|---|---|---|---|
| CS | 221M | 0.12 / 0.14 / 0.16 / 0.29 h | 0.17 / 0.19 / 0.23 / 0.41 h |
| MS | 84M | ≈ 0.04 – 0.12 h | ≈ 0.06 – 0.17 h |
| DG | 69M | 0.03 / – / – / 0.10 h | 0.05 / – / – / 0.14 h |
| DC | 27M | 0.01 – 0.04 h | 0.02 – 0.05 h |
| Tier 0 | 40M | 0.03 – 0.05 h | 0.03 – 0.07 h |

**Tier-A evaluation total:** 9.8 h (D384, n₀ = 5) / 12.1 h (D512, n₀ = 4). Cost benchmarks add 3.0 h.

---

## 10. Efficiency measurement protocol (RQ-D; I-18, U-10, U-14)

### 10.1 Analytic costs (primary)

- **State bytes, exactly as stored:** bf16 KV (4d B per token per full layer); fp32 GDN state; fp32 conv caches. Measured `nbytes` must equal the analytic value (VT-14).
- **FLOPs per token** for prefill and decode (§3.4 formula).
- **Bytes moved per decode step:** 2·N_total (bf16 weights) + k·T·4d (KV read) + 2·(12 − k)·S_lin (state read + write).
- **Roofline latency** at 112.6 dense-BF16 TFLOP/s and 960 GB/s (NVIDIA RTX Blackwell whitepaper Table 4, [PDF](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf) [read-section per 04]). It is labelled a **lower bound** for both mixers.
- **Sequential-depth term** for GDN chunked prefill: T/64 inter-chunk steps.
- **Literal H6 identity:** for any per-layer-additive cost, c̃(k) = k/12 and M(k) = G·(R(k) − k/12) (debate §6.4).

**Projection table** [INF].
- Per full layer at T = 32K, the attention-score/dense FLOP ratio is T/(4d + 3d_ff), i.e. T/(13d) for MHA with d_ff = 3d (debate I-18).
- Our D384 model has d_ff = 2.67d, so its actual ratio is 32,768/(4·384 + 3·1,024) ≈ 7.1.
- Break-even context (one layer's KV = one GDN-A-geometry state) is ≈ 72 tokens at every d.

| d | Attention/dense ratio @32K | KV B/token/layer | GDN-A-geometry state per layer |
|---|---|---|---|
| 384 | 6.56 | 1,536 | 108 KiB |
| 512 | 4.92 | 2,048 | 144 KiB |
| 2048 | 1.23 | 8,192 | 576 KiB |
| 4096 | 0.62 | 16,384 | 1,152 KiB |

### 10.2 Measured costs (secondary; scoped to "fla Triton + PyTorch SDPA on sm_120, this implementation")

**What is measured.**
- **Configs:** C0, C1, C2, C3, C4, C6, C12, X-gc12, X-E0 at the chosen scale.
- **Weights:** random-init (stream `bench`). Timing does not depend on trained weights, so benchmarks can run early.
- **Prefill:**
  - whole model; B ∈ {1, 8} × T ∈ {1K, 4K, 16K, 32K}, skipping shapes that exceed the memory budget;
  - one CUDA graph captured per exact (B, T), plus eager;
  - attention backend pinned to FLASH via `sdpa_kernel` (cuDNN also reported); FlexAttention comparator;
  - only last-token logits are computed (full logits at 16K × 16K V add 512 MB; 05 §2.4 [MEAS]).
- **Decode:**
  - per-token latency at context L ∈ {1K, 4K, 16K, 32K}; one-step **exact-shape** CUDA graph per L; **no masks**;
  - B ∈ {1, 8, B_max(L)}, where B_max is the largest batch within an 8 GiB state-plus-activation budget (analytic, then verified by allocation in an idle window);
  - cuDNN pinned as primary, flash also reported; FlexAttention if the U-10 self-test passes, otherwise Flex for prefill only;
  - GDN uses explicit `fused_recurrent` calls with the state `copy_`-ed into static buffers.
- **Kernel-efficiency table:** achieved TFLOP/s or GB/s and roofline fraction per kernel and shape (attention core, GDN chunk core, projections, decode steps).

**Hazards and their controls** [MEAS-B unless noted].

| Hazard | Measured size | Control |
|---|---|---|
| Masked static cache dispatches SDPA to the mem-efficient kernel | 1.80 ms vs 85–90 µs per layer at 32K | Exact-shape graphs, no masks. Runtime assertion that the pinned backend is used. A CI test asserts no inference path reaches mem-efficient |
| L2 residency (64 MiB L2) | 33 vs 84 µs | Report both cold L2 (zero a 256 MB buffer before every replay, untimed) and hot L2 |
| Launch-bound GDN at batch 1 | ≈ 0.47 of 0.69 ms per layer at 4K is non-core kernels and launch overhead | Graph capture per shape; eager reported separately |
| Unequal kernel maturity | FA2 at 81% of peak vs the GDN chunk core at ≈ 10% | Roofline-fraction table. FlexAttention is reported as "equal toolchain, not equal maturity" |

**Statistics.**
- 5 warm-up replays, then **N ≥ 30** timed replays per (config, shape, L2 mode).
- CUDA events (`torch.cuda.Event(enable_timing=True)`), with `torch.cuda.synchronize()` before reading.
- **Min-of-N is primary** ([Chen & Revels 1608.04295 §B](https://arxiv.org/abs/1608.04295) [read-section per B1]); median and IQR are secondary.
- Configs run in randomised, interleaved blocks. **2 sessions on different days.**
- Any replay with a non-idle `clocks_event_reasons` flag is discarded.
- Clocks, temperature and power are logged. Clocks cannot be locked: `-lgc` needs root [MEAS-B].
- The Triton cache is pre-warmed. `FLA_FLASH_QLA=0`.

**Idle-GPU requirement (G-cost).**
- No foreign compute process for ≥ 60 s before or during a session (polled every 5 s); utilisation < 5% at start.
- **If no idle window is granted:** contended min-of-N numbers go to an appendix only, and RQ-D/H6 are judged on analytic costs (U-14).

---

## 11. Pre-registered statistical analysis plan

### 11.1 Unit of inference

- **The training run (seed).** Per run, a family score at each length is computed from item-level EM_tf, as in §2.
- Seeds pair across configurations by seed index: shared data order and shared-module initialisation.
- Every seed is plotted.

### 11.2 Primary cell and reference-adequacy gate (G-REF; I-05, I-06)

**Criteria** (on DG items, seed-pooled means):
- **(a) Adequacy:** S₁₂(T) ≥ 0.80·S₁₂(C0), on the R1/R-load ladder mean.
- **(b) Discriminability:** S₁₂(T) − S₀(T) ≥ 15 pp.

**Pass rule.** T ∈ {16K, 32K} passes iff (a) and (b) hold seed-pooled **and** at least n − 1 of n seeds individually satisfy (a). Otherwise T is "reference-unstable".

**Primary cell.** The mean over passing lengths. If only one passes, that length. If neither passes, 8K, labelled "not long-context". Lengths are never pooled across bands.

**Order of operations** (pre-registered):
1. All four anchors reach n₀ seeds.
2. DG is scored.
3. The primary cell is fixed.
4. The U-05 trigger is evaluated (§11.4).
5. If seeds were added, G-REF is re-applied with all n seeds by the same rule. The trigger is not revisited.

**Reference inversion** (secondary outcome). Triggered when k = 12 fails (a) at a length while some hybrid passes its own (a) and exceeds S₁₂ by ≥ 5 pp on DG.
- First the 2× Stage-B branch is checked.
- Then R_best is reported, with the best configuration chosen on DG.

### 11.3 Confirmatory tests: H1, H1-min, H3, H5(a)

**Test statistics.**
- L_k: per seed s, L_{k,s} = S_{k,s} − 0.2·S_{0,s} − 0.8·S_{12,s}.
- One-sample t on {L_{k,s}}, df = n − 1.
- D₃ and b are tested in the same way.

**Holm.**
- **Support family** {H1, H1-min, H3, H5(a)}: one-sided p in the pre-registered direction, Holm at α_FW = 0.05. A non-evaluable H3 enters with p = 1.
- **Falsification family** {H1, H1-min}: one-sided p in the opposite direction, Holm at 0.05, plus the guards of §11.6.

**Sensitivity analyses.**
- Seed-blocked pooled-variance model over the 4 × n core table (df = 3(n − 1)) [A2].
- Hierarchical paired bootstrap, B = 10,000: resample seeds (same indices for every config), then contexts within (task, level, T) strata (same indices for every config); percentile CIs.
- If either disagrees in sign with the primary verdict, the verdict is downgraded to "suggestive".

**Additional labels.**
- "Metric-robust": L_k on gold log-probability agrees in sign.
- "Conditional on reference adequacy": H5(a) with any length failing G-REF. The slope over passing lengths only is then reported as a sensitivity analysis.

**Operating characteristics** [PI-SIM, debate §6.2; G = 30 pp, seed effect 2 pp].
- **Type I** (R = 0.8): paired test 0.049–0.051 under equal and unequal variance.
- **Adaptive 4 → 6:** type I 0.049–0.058; power at R = 1.0 is 0.923 (σ = 3 pp) / 0.979 (σ = 2 pp).
- **Adaptive 5 → 7:** power 0.967 / 0.997.
- **Bimodal seeds:** no few-seed test is valid (0.09–0.30), hence the success counts.

### 11.4 Seed trigger (U-05; effect-blind)

- **Evaluated once,** after n₀ seeds of all four anchors.
- **Fires if** the blinded pooled residual SD of the two-way additive fit (config + seed) to primary-cell scores exceeds 2.5 pp, **or** any anchor's seed range exceeds 12 pp.
- **Action:** add 2 seeds to all four anchors.

### 11.5 Success counts (named secondary estimand)

- Per configuration, the number of seeds with primary-cell score above the midpoint (S̄₀ + S̄₁₂)/2.
- Reported with Clopper–Pearson 95% CIs.
- A split of seed outcomes into two modes is reported as a finding, not as noise.

### 11.6 H1 falsification guards (I-22)

All four must hold for "falsified":
1. **Gates:** G0, G-RP, FEAS-1/2 and G-FULL pass. G-REF passes at the primary length, with ≥ n − 1 of n seeds adequate.
2. **Test:** on the unselected primary-cell ladder mean, the paired one-sided 95% upper bound of L₃ is < 0, i.e. the falsification-direction p is < 0.05 after Holm over {H1, H1-min}. Pooled and bootstrap analyses agree in sign.
3. **Dose guard:** at seed 0, the TEF-4,000 branch keeps L₃ < 0, and k = 3 has plateaued (|S₃(4,000) − S₃(2,000)| < 3 pp).
4. **Saturation guard:** the 2× Stage-B branch changes S₁₂ at the primary length by ≤ 5 pp. Otherwise the verdict is "conditional on the long-stage budget", and L₃ is recomputed on the 2× branches.

**Not required for falsification:** the conv and NoPE controls. They decide the operator wording (§11.9).

**Per-level results:** reported with Holm across levels (B2 W4).

### 11.7 H3, H5(b), H4, H6′, H2

- **H3 (I-14).**
  - D₃ per seed on primary-cell ladder means. Retrieval = {R1, R-load}; multi-hop = M-hop h ∈ {1, 2, 3, 4}.
  - Evaluable iff every seed has G ≥ 15 pp in both families.
  - Equated-level sign check (U-11).
  - Sensitivity analyses: Fieller and bootstrap intervals; binomial GLMM `correct ~ family × k + level + (1|context) + (1|seed) + (1|seed:k)` on the logit scale.
  - Aggregation: A-freq contrasts in the secondary analyses; A-count is a diagnostic.
- **H5(b):** per-seed b_ret − b_hop, two-sided.
- **H4 (I-15):**
  - c₁ and c₂ with paired t, two-sided; pooled-variance as sensitivity.
  - MDE at 80% power for c₁: ≈ 7.3 / 4.9 pp (n = 5, σ = 3 / 2 pp) and ≈ 9.5 / 6.3 pp (n = 4) (debate §6.4).
- **H6′ (I-16):**
  - Welch t on S(X-E0) vs S(C{k*}); TOST ±5 pp.
  - Share of the k = 0 → k* gap closed by GDN-E, with a Fieller CI (descriptive).
  - State bytes alongside: at D512 and 32K, 6.19 MiB (X-E0) vs 65.55 MiB (C1).
- **H2:**
  - Isotonic regression and the saturating fit S(k) = S₀ + G·(1 − e^{−k/κ})/(1 − e^{−12/κ}).
  - k80 (linear interpolation of isotonic R) with bootstrap CI; g₁ vs g_late.
- **Secondary family Holm:** {H4-c₁, H4-c₂, H6′, H5(b)} at 0.05, two-sided.
- **Exploratory:** BH-FDR at q = 0.10.
- **Equivalence margins:** 5 pp accuracy; 0.10 recovery difference; 2 pp per doubling; 0.01 nats LM loss; 10% latency/throughput.

### 11.8 Conv-control and NoPE triggers (read at the Tier-A decision point)

- **U-02.** δ_gc = seed-paired mean over seeds 0–1 of S₁₂(X-gc12) − S₁₂(C12) at the primary cell.
  - δ_gc > 3 pp → X-gc1 and X-gc3 × 2 seeds (Tier B). H1 and H1-min are reported in both families.
  - δ_gc > 6 pp → F_gc becomes co-primary (≥ 3 seeds at k ∈ {1, 3, 12}) before any placement work.
- **U-06.** |ΔR₃| > 0.1 or |ΔS₁₂| > 3 pp (NoPE vs RoPE, seed 0) → NoPE seed 1 (Tier B).

### 11.9 Wording rules

- Confirmatory claims are **package claims**: "k F_g layers of 12, the rest GDN-A as built" (I-01).
- **Operator-level wording** ("softmax memory vs delta-rule memory") is allowed only if δ_gc ≤ 3 pp, or both families give the same verdict.
- **Elicited vs zero-shot** (I-27): claims are scoped to "capability elicited after identical exposure"; Tier-0 disagreements are reported.
- **Scope:** the chosen scale, GDN-A, this recipe. "Consistent with" is used for prior work.

---

## 12. Plots

Each plot answers one question.
- **Generation:** each figure comes from `plots/make_<fig>.py`, which reads only `experiments/results/**` and the registry and writes PDF, PNG and a JSON of input hashes.
- **Style:** cividis for k; Okabe–Ito for families; every seed drawn; 95% CI type stated in the caption.
- **Axes:** log₂ length axes. The extrapolation region (if L_ext = 16K) is shaded.

| # | Plot | Answers |
|---|---|---|
| P1 | S_ret(k) at the primary cell, all seeds, with the R = 0.8 line and isotonic fit | RQ-A / H1, H2 |
| P2 | Forest plot of per-seed L₁ and L₃ with paired CIs, sensitivity CIs and success counts | H1, H1-min |
| P3 | R_f(k) for the three families on one probability-scale axis; per-seed D₃ | RQ-C / H3 |
| P4 | Δ₀(T) vs log₂T on the fixed level set, per-seed slopes, G-REF status per T; inset: LM-4 Λ(g) slope | H5(a) |
| P5 | Retrieval vs multi-hop length slopes | H5(b) |
| P6 | Early / middle / late at k*, per-seed lines, c₁ and c₂ | RQ-B / H4 |
| P7 | Primary-cell score vs analytic state bytes at 32K (log x), router line, Pareto front, GDN-E point | RQ-D / H6, H6′ |
| P8 | Measured vs analytic prefill and decode latency vs T per k (cold/hot L2), roofline fractions | RQ-D |
| P9 | B_max(T) within 8 GiB per k | RQ-D |
| P10 | Per-level R (H1-robust) and N50 vs k at 16K and 32K, levels as N/N50₀ | RQ-A difficulty dependence |
| P11 | G-REF panel: S₁₂(T)/S₁₂(C0) per T and seed; reference inversion | Validity (U-04) |
| P12 | TEF dose ladder (250 → 4,000) per anchor; gap vs steps | Validity (U-09) |
| P13 | 2× vs 1× Stage B at k ∈ {0, 3, 12} | Validity (U-04) |
| P14 | F_gc vs F_g at k = 12 (and k ∈ {1, 3} if triggered) | Validity (U-02) |
| P15 | NoPE vs RoPE at k ∈ {3, 12} | Validity (U-06) |
| P16 | R-dist × filler at 32K, S₀ and S₁₂ separately | H5 mechanism |
| P17 | LM-1 per-position loss and LM-3 repeat-token loss by distance, per k | RQ-A zero-shot |
| P18 | Strip plot of per-seed primary-cell scores (bimodality) | U-05 |

---

## 13. Staged plan, budget and go/no-go gates

### 13.1 Order of execution

These experimental stages S0–S8 are not the README's project stages. S0 corresponds to README Stage 4; S1–S3 to Stage 5; S4–S7 to Stage 6.

1. **S0 infrastructure and verification (G0).**
2. **S1 reproduction pilot first (G-RP):** Zoology MQAR, before any LM training.
3. **S2 FEAS-384**, then FEAS-512 only if required (U-01).
4. **S3 LR sweep** at the chosen scale (G-LR).
5. **G5 freeze:** item files and dev-gate items, analysis code tested on synthetic data (VT-18), configs, Triton cache with SHA-256, `uv.lock`, and a decision-log row with commit hashes.
6. **S4a:** C12 and C0, seed 0 (G-FULL).
7. **S4b:** remaining core seeds → G-REF → U-05 trigger.
8. **S5a controls and S5b seed-0 branches.**
9. **S6:** evaluation; cost benchmarks in idle windows.
10. **Tier-A decision point:** H1, H1-min, H3, H5(a) verdicts; the U-02, U-06 and U-13 triggers.
11. **S7 (Tier B):** curve first → k* → placement and GDN-E → conditional arms.
12. **S8 (Tier C)** by priority.

### 13.2 GPU-hour budget

Hours are contended, from the debate's measured per-token rates:
- Stage A: 88.9K tok/s (D512) and 120.0K (D384).
- Stage B and TEF: 70.5 / 64 / 59 / 51 / 44 / 31.9K tok/s for k = 0 / 1 / 2 / 3 / 6 / 12 at D512, ×1.33 at D384.
- Per run (Stage A + Stage B at the middle option + TEF 1,000):
  - D384: k0 1.76, k1 1.80, k2 1.83, k3 1.90, k6 1.98, k12 2.21 h;
  - D512: 3.55, 3.65, 3.75, 3.95, 4.18, 4.81 h.
- Overheads: F_gc +6%; GDN-E +17%.
- Script: `/tmp/claude-1000/-home-sedrica-lynx-Downloads-outeropt/73b71b1c-85a4-4704-b75f-7c9b668d4052/scratchpad/protocol/budget.py` [INF].

| Stage | Content | D384 path | D512 path |
|---|---|---|---|
| S0 | Infrastructure, verification tests VT-01…VT-20, Triton warm-up, throughput | 3.0 | 3.0 |
| S1 | Zoology MQAR reproduction | 3.0 | 3.0 |
| S2 | FEAS-384 (2 seeds × {30M, 90M} branches × TEF + SSMax arm + evaluation) | 9.8 | 9.8 (fails) |
| S2 | FEAS-512 (seed 101; {0.1B, 0.3B} branches; SSMax arm) | — | 11.8 |
| S3 | LR sweep, 12 × 0.15B | 4.2 | 5.6 |
| S4 | Core k ∈ {0, 1, 3, 12} × n₀ | 38.4 | 63.8 |
| S5a | X-gc12 × 2; X-nope3, X-nope12 × 1 | 8.8 | 19.0 |
| S5b | TEF → 4,000 at {0, 1, 3, 12}; 2× Stage B at {0, 3, 12} (seed 0) | 5.8 | 12.8 |
| S6 | Evaluation (frozen suites) | 9.8 | 12.1 |
| S6 | Cost benchmarks (idle windows) | 3.0 | 3.0 |
| **Tier A** | **Decision point** | **85.7 (102.9 with 20%)** | **143.9 (172.7)** |
| Tier A, conditional | +2 seeds on all four anchors (U-05) | 15.3 | 31.9 |
| **Tier B** | Curve k ∈ {2, 6} × 2 (7.6 / 15.9); placement at k* = 1, 2 arms × n₀ (18.0 / 29.2); GDN-E at {0, k*} × 2 (8.3 / 16.8); evaluation (3 / 5) | **36.9** | **66.9** |
| Tier B, conditional | X-gc1, X-gc3 × 2; NoPE seed 1; centred-arm top-up if k* = 2 | ≤ 17.5 | ≤ 32.4 |
| **Minimum H1-falsification experiment** | S0, reduced S1 (1.5 h), FEAS, LR k ∈ {0, 12} × 3, core k ∈ {0, 3, 12} × n₀, seed-0 TEF-4,000 and 2× branches at {0, 3, 12}, retrieval-only evaluation | **53.4 (64.1)** | **93.6 (112.3)** |

**Budget variance.**
- If FEAS-3 selects the largest Stage-B budget, the core grows by ≈ +2.7 h (D384) / +9 h (D512).
- If U-15 fires (600 contexts), evaluation grows by ≈ +2 h.
- **Economy option, D512 only:** n₀ = 3 saves 16 h in Tier A. Power at R = 1.0 drops to 0.82 / 0.87 (debate §4).

### 13.3 Gates (quantitative)

| Gate | When | Pass criteria | If it fails |
|---|---|---|---|
| **G0** | End of S0 | Every VT test (§16) passes at its threshold. Throughput, contended: Stage A ≥ 70K tok/s (D512) / ≥ 95K (D384) for every k; C12 Stage B ≥ 20K (D512) / ≥ 27K (D384). Peak VRAM ≤ 7 GiB in every stage. Weight-transplant logits within 1e-3 relative. Triton cache archived | Fix and re-test. On a D512 throughput failure, Stage A drops to 0.6B (pre-registered) |
| **G-RP** | End of S1 | (a) Zoology's own models, re-run, best over LR on overall test accuracy (all 7 test configs): GDN ≥ 0.95 at d_model ≥ 128; attention ≥ 0.95 at d_model ≥ 64; GDN accuracy non-decreasing in d_model. (b) **Our** mixers in the same harness within ±0.03 of Zoology's at matched d_model and LR. Criteria use re-run numbers only, never values read from the figure, whose attention and SWA markers coincide (I-20) | Debug the mixers or harness. No LM training until this passes |
| **FEAS** (U-01) | End of S2 | **D384 iff, for both seeds 101 and 102 at 32K** (R1/R-load FD ladder mean, TEF 1,000; branch = 30M if FEAS-3 picks 30M, else 90M): S₁₂ ≥ 0.85·S₁₂(C0) **and** S₁₂ − S₀ ≥ 25 pp | Run FEAS-512 under FEAS-1 (≥ 0.80·C0) and FEAS-2 (≥ 20 pp) at 32K. If it fails, L_ext = 16K at D512 when the same criteria hold at 16K: Stage B at 16,384 with 8 × 16K per step, and 32K cells become extrapolation. Otherwise **stop and re-scope with the user** |
| FEAS-3 (U-04) | End of S2 | Δ = S₁₂(large) − S₁₂(small branch) at 32K, seed-averaged. Δ ≤ 3 pp → smallest budget; 3–10 pp → middle; > 10 pp → largest | — (sets the value) |
| FEAS-5 (U-03) | End of S2 | SSMax: ≥ 3 pp gain at 32K and ≤ 2 pp loss at 4K | Plain per-head gains (default) |
| TEF gates (U-08) | End of S2 | Forgetting gate and elicitation check (§7.3) | Pre-registered fallbacks (§7.3) |
| ICC (U-15) | End of S2 | Design effect DE = 1 + 3·ICC on FD 16K/32K retrieval cells (one-way random effects, pooled over k ∈ {0, 12}) ≤ 2 | Confirmatory retrieval cells at 16K/32K use 600 contexts |
| Direct-supervision promotion (I-08) | End of S2 | Not (floor ≤ 5% EM, or ceiling ≥ 95%, at **every** FD level of a family for k = 0 or for k = 12) | Run the direct-supervision screen (Tier C item) before S3 |
| **G-LR** | End of S3 | No divergence at η*; regret rule (§7.2) | Per-config η; edge extension |
| **G5** | Before S4 | All frozen artefacts committed with hashes; decision row entered | No launch |
| **G-FULL** | After S4a | On DG items at 32K: S₁₂ ≥ 0.80·S₁₂(C0) and S₁₂ − S₀ ≥ 20 pp | Fail at 32K only → continue (16K primary), flagged. Fail at both 32K and 16K → **pause the queue and consult the user** (re-scope branch) |
| G-SEED (U-05) | After S4b | §11.4 | +8 runs |
| G-REF | Analysis | §11.2 | Fallback order 32K/16K → 8K, labelled |
| G-div | Continuous | NaN, or loss > EMA₀.₉₉ + 1.0 for > 50 steps | One automatic restart from the last checkpoint. A second failure is reported (§8.1) |
| G-OOM | Continuous | — | Auto-resume. 3 consecutive OOM failures mark the run failed |
| G-throughput | Continuous | tok/s within 30% of the G0 value | Below for > 1 h → re-plan the queue |
| G-cost | Before benchmarks | Idle criteria (§10.2) | Defer; appendix-only contended numbers |
| **Tier-A decision** | After S6 | Verdicts computed; U-02, U-06 and U-13 evaluated | If the primary cell fell back to 8K, consult the user before Tier B |

**Stop rules.**
- The FEAS fallback ladder is exhausted.
- G-FULL fails at both long lengths.
- Contingency is spent only on re-runs after gate failures or on pre-registered triggers.

---

## 14. Competing explanations (summary)

The full register, with discriminating experiments, predicted signatures and decision rules, is in `reports/competing_explanations.md`. The most consequential entries:

| Outcome | Strongest rival explanation | Discriminating experiment (tier) |
|---|---|---|
| H1 supported | Weak long-length reference (thin Stage B, RoPE, QK-norm dilution at d_h = 64) | G-REF; 2× Stage-B branch; attention diagnostics; SSMax rule; NoPE arm (A) |
| H1 supported | Conv asymmetry (smeared keys; [Olsson 2209.11895](https://arxiv.org/abs/2209.11895) [read-section per A1, B2]) | X-gc12 (A); X-gc1 and X-gc3 if δ_gc > 3 pp (B) |
| H1 supported | The task fits in the GDN state | R-load to N = 1,024 / 2,048; N/N50₀ (A) |
| H1 falsified | Learning speed / exposure | TEF-4,000 dose guard; still-closing flag (A) |
| H1 falsified | Under-trained long stage | 2× Stage-B branch at {0, 3, 12} (A) |
| H5 supported | Ladder composition or exposure per length | Fixed level set; length-balanced TEF (by design); LM-4 (A) |
| H5 supported | Decay over distance vs interference from load | R-dist × filler at 32K; load × length (A, MS) |
| H3 supported | Multi-hop is simply harder | Difficulty-equated levels (U-11); C0; logit sensitivity (A) |
| Hybrid > all-full | Small-scale all-full deficit ([Qiao 2606.15378v1](https://arxiv.org/html/2606.15378v1) Table 2: all-full 35.95 vs hybrids 49.58 / 67.81 at 0.22B [read-section, PI-VER in debate]) | Reference-inversion outcome; R_best; X-gc12; NoPE (A) |
| Measured ≠ analytic | Dispatch, L2, launch and kernel-maturity artefacts | §10 controls (A) |

---

## 15. Experiment tiers

**Essential: Tier A (≈ 86 / 144 GPU-h base).**
S0, S1, S2, S3, core k ∈ {0, 1, 3, 12} × n₀, X-gc12 × 2, NoPE {3, 12} × 1, seed-0 TEF-4,000 and 2× branches, frozen evaluation, cost benchmarks (analytic always; measured if idle windows exist). Tier A decides H1, H1-min, H3 and H5(a).

**Desirable: Tier B (≈ 37 / 67 GPU-h, + ≤ 17.5 / 32.4 conditional).**
- Curve C2 and C6 × 2 seeds (run first).
- Placement at k* (2 new arms × n₀).
- GDN-E at {0, k*} × 2 (H6′).
- Conditional: X-gc1 and X-gc3; NoPE seed 1; centred top-up if k* = 2.

**Optional: Tier C, in priority order.** D384-path costs; D512 ≈ 2×.

| # | Arm | Cost | Why |
|---|---|---|---|
| 1 | GLA at k ∈ {0, k*, 3} × 2 | ≈ 12.6 h | Mechanism generality. Promoted into Tier B if no Tier-B conditional arm fires and ≥ 20 h of the agreed budget remains (U-13) |
| 2 | Conv-free GDN at k ∈ {0, k*} × 2 | ≈ 6.9 h | Is one-layer sufficiency conv-mediated? |
| 3 | Twin attention at k = 12 × 2 | ≈ 4.9 h | Head geometry, 4 vs 8 heads (U-12) |
| 4 | Non-chosen SSMax variant at k ∈ {3, 12} | ≈ 4.1 h | Temperature parameterisation |
| 5 | Direct-supervision screen | ≈ 5 h | Capacity anchors; bimodality screen (I-08) |
| 6 | Other scale, k ∈ {0, 3, 12} × 1 + mini LR sweep | ≈ 8–15 h | "Scale extension", with directional predictions frozen before it runs (not a replication) |
| 7 | η*/√2 at k ∈ {3, 12}, seed 0 | ≈ 4.1 h | LR sensitivity |
| 8 | 4× Stage-A tokens at k ∈ {3, 12}, seed 0 | ≈ 12.5 h | Undertraining |
| 9 | C4 × 2 | ≈ 3.9 h | Curve resolution |
| 10 | k = 2 adjacent {5, 6} vs spread × 2 | ≈ 3.7 h | Composition depth for multi-hop |
| 11 | k = 1 at layer {0} × 2 | ≈ 3.6 h | Layer-0 placement |
| 12 | M2 variable tracking, A3 gather, BABILong qa1–3 after the TEF; lm-eval RULER cross-check on public small checkpoints | ≈ 2 h | Generator cross-checks |

**Infeasible or out of scope** (stated as limitations).
- **Count vs proportion (F24):** changes width and state size (I-25, I-28).
- **Compute vs memory:** sliding-window attention at matched compute is a different family (I-28).
- **Document-masking × k interaction** (I-28).
- **Scale transfer beyond the chosen scale** (I-28).
- **Prompted natural-language long-context benchmarks** (RULER prompts, HELMET, LongBench) as capability measures. 20–60M base models score near zero (05 P13; Wang et al.: 340M recall "insignificant" [read-section per A1, B1]).
- **Contexts above 32K.**
- **A Mamba-2 arm with official kernels:** `mamba_ssm` and `causal-conv1d` have no torch-2.11 wheel and there is no nvcc (05 §4.4 [code-inspected per 05]).
- **External FA2/FA3/FA4 packages:** no torch-2.11 wheel or no nvcc; FA3 is H100-only (05 §4.3).
- **Locked GPU clocks:** need root.
- **GQA capability claims:** bytes only.
- **Deterministic production training:** rejected unless G0 on an idle GPU measures ≤ 3% overhead (I-19). The measured cost is +43–66% on the FA2 forward+backward kernel [MEAS-B].

---

## 16. Implementation verification tests (S0; all must pass for G0)

| ID | Test | Pass threshold |
|---|---|---|
| VT-01 | Parameter counts per config and mixer, both scales | Exactly equal to §3.3–3.4 |
| VT-02 | **Causal mask.** Outputs at positions ≤ t are unchanged when tokens > t are perturbed: F_g varlen, F_g SDPA, F_gc, GDN chunk, GDN recurrent | fp32: attention max abs diff ≤ 1e-6, GDN relative RMS ≤ 3e-3 (TF32). The gradient of out_t with respect to x_{t′>t} is exactly 0 in fp64 for our own modules |
| VT-03 | **Document isolation.** Packed with `cu_seqlens` vs each document alone, including the conv state, the F_gc conv and RoPE reset | GDN max abs diff 0.0 (measured exact, 05 §2.2). Attention relative RMS ≤ 1e-2 (bf16) / ≤ 1e-5 (fp32) |
| VT-04 | **GDN chunk vs naive fp64 recurrence.** T ∈ {512, 32,768}; decays mixed (e^−0.5 … e^−1e-4) and near-unit (1e-4); geometries 4 × 64/128, 3 × 64/128, 1 × 256/512, 1 × 192/384 | Relative RMS ≤ 3e-3 (fp32/TF32 inputs) and ≤ 1e-2 (bf16), flat across position bands. Gradient check vs fp64 autograd at T = 512: relative RMS ≤ 5e-3. Measured 1.5–1.6e-3 and 5–6e-3 [MEAS-B, MEAS-B2] |
| VT-05 | **Recurrent vs chunked equivalence.** Chunk prefill vs token-by-token `fused_recurrent` with state carry; and chunk prefill + recurrent tail vs one chunk pass | Outputs relative RMS ≤ 3e-3 (fp32) / ≤ 1e-2 (bf16). Final state relative max ≤ 2e-3 (fp32) / ≤ 8e-3 (bf16). Measured ≤ 1.4e-3 / ≤ 7.1e-3 [MEAS-B2] |
| VT-06 | **Cached vs uncached decode.** Prefill P + M cached decode steps vs one forward over P + M, for k ∈ {0, 3, 12} | fp32: argmax agreement 1.000 and max \|Δlogit\| ≤ 2e-2 (measured 0.0099, 05 §2.3). bf16 trained checkpoint: EM flip rate < 1% on a 5% re-score |
| VT-07 | Branched query scoring vs separate forward passes | \|Δ logp_gold\| < 1e-3 (fp32); EM identical |
| VT-08 | Scoring path: fla ops called explicitly by keyword; no automatic mode switch; backend pinned and asserted | Assertions hold. A positional call is forbidden (it produced NaN, PROP §4 [MEAS]) |
| VT-09 | **Weight-transplant logits (I-20).** (1) Our C12 with the gate bypassed (test flag), per-head gains tied ↔ transformers `Qwen3ForCausalLM` (`hidden_size` d, heads d/64, `head_dim` 64, `num_key_value_heads` = heads, `intermediate_size` d_ff, `rope_parameters={"rope_type": "default", "rope_theta": 5e5}`, `rms_norm_eps` 1e-5, `tie_word_embeddings=True`, `attention_bias=False`; q_norm/k_norm over head_dim at `modeling_qwen3.py` L229–230 [code-inspected, PI-VER]). (2) Our C0 ↔ fla `GatedDeltaNetForCausalLM` (`hidden_size` d, 12 layers, `num_heads` d/128, `head_dim` 64, `expand_v` 2, `use_gate`, `use_short_conv`, `conv_size` 4, `intermediate_size` d_ff, `norm_eps` 1e-5, `tie_word_embeddings=True`, `fuse_cross_entropy=False`). (3) Headwise gate unit test | fp32 logits within 1e-3 relative. Gate output equals the manual σ(xW_g)·o |
| VT-10 | **Initialisation.** `A_log` and `dt_bias` survive the global init; output-projection std ≈ 0.02/√24; shared modules bit-identical across k for a given seed (SHA-256 of embedding/MLP/norm tensors in C0 vs C12) | All hold |
| VT-11 | **Data.** Batch-index SHA-256 identical across k per (seed, step); loader stateless; EOS separation; `cu_seqlens` and positions correct; stage token counts exact; PG19 windows book-bounded; held-out document-hash intersections empty; mixing proportions exact per window | All hold |
| VT-12 | **Evaluation items.** len == T exactly; each answer is 4 single-digit tokens (or 1 token for A-freq/A-count); train/eval (key, value, template) hash overlap = 0; no separator inside contexts; recorded positions match; item files byte-identical across models | All hold |
| VT-13 | **Determinism and resume (I-19).** Audit mode (`use_deterministic_algorithms(True)`, no cuDNN SDPA, `CUBLAS_WORKSPACE_CONFIG=:4096:8`): two 200-step runs per k ∈ {0, 3, 12} in separate processes; kill -9 at step 137 (mid-accumulation) and during a checkpoint write, then resume. Production mode: resume. Standing replicate: C12 seed 0, first 1,000 steps, audit mode | Audit: bitwise-identical loss traces and parameter SHA-256; resume bitwise identical. Production: \|Δloss\| < 0.005 at resume + 500. Replicate bitwise |
| VT-14 | Inference-state bytes: `nbytes` of cache tensors after prefill at T ∈ {1K, 32K} per config | Equal to the analytic §3.4 values to the byte |
| VT-15 | **Benchmark harness self-test (I-18, U-10).** Backend assertion; no masks; cold-L2 flush effective (cold/hot ≥ 1.5 on 64 MiB KV decode); reproduces [MEAS-B] Appendix A within 15% (contended); FlexAttention device-side length update inside one captured graph | All but the Flex test must pass. A Flex failure → prefill-only comparator |
| VT-16 | GDN-E at K = 256 / 192 and GLA: forward/backward finite; numerics as VT-04 | Pass, or the fallback of §3.3 |
| VT-17 | TEF generator: equal counts per bucket and family (±1 per 1,000 steps); answer-token mask correct; no eval tuples | All hold |
| VT-18 | **Analysis code on synthetic data.** The pipeline recovers planted L_k, D₃ and slopes, and reproduces PI-SIM operating characteristics (debate §6.2) within ±0.01 at 20,000 replicates | All hold |
| VT-19 | Triton cache warmed for every training, evaluation and benchmark shape (`IS_VARLEN` true and false; every length); tarball SHA-256 in the registry | Done |
| VT-20 | Memory: peak ≤ 7 GiB per process in Stage A, Stage B, TEF and evaluation at 32K, for C12 and C0 | Holds |

---

## 17. Reproduction pilot (S1, runs first; gate G-RP)

**Target.**
- Zoology MQAR recall–state trade-off, attention and Gated DeltaNet ([paper 2312.04927](https://arxiv.org/abs/2312.04927) [read-section per PROP]).
- Repo [HazyResearch/zoology](https://github.com/HazyResearch/zoology) @`1ad20d193b6113cae1e8f3c655c300d7b4b3f4bb`, config `zoology/experiments/mqar_example_configs/original_mqar_configs.py` [code-inspected, PI-VER].

**Config, exactly as in the file.**
- V = 8,192.
- Training: (64 tokens, 4 KV) × 100K; (128, 8) × 20K; (256, 16 / 32 / 64) × 20K each.
- Test: (64, 4 / 8 / 16), (128, 32), (256, 64), (512, 128), (1024, 256), 1,000 each.
- Batch 256 (test 32); 32 epochs; LR ∈ logspace(−3, −1.5, 4).
- 2-layer `TransformerBlock` with a BaseConv (kernel 3, implicit long conv) mixer before the sequence mixer.
- `attention`: MHA, 2 heads, dropout 0.1, d_model ∈ {32, 64, 128}.
- `gated_delta_net`: fla GDN, 2 heads, `use_gate=False`, short conv 4, d_model ∈ {64, 128, 256} (`models_repo.py` L5–29, L265–293 [code-inspected, PI-VER]).

**Runs (≤ 22).**
- Zoology attention at d_model ∈ {64, 128} and GDN at {128, 256}, × 4 LRs: 16 runs.
- Our attention module configured to match Zoology's MHA (no gate, no QK-norm, no RoPE) and our fla-GDN wrapper in Zoology's geometry, at d_model {64, 128} / {128, 256} and Zoology's best LR: 4 runs.
- Exploratory: config-A head geometry (d_k 64, d_v 128) at d_model ∈ {128, 256}: 2 runs.
- **Pre-declared cut:** if the first timed run implies > 3.0 GPU-h in total, cut to the 2 best LRs per family, chosen on the first d_model completed at 4 LRs.

**Pass criteria:** G-RP (§13.3).

**Logging:** wandb disabled. `cache_dir` set to `data/zoology/`.

**RP-2 (bug detector, not a gate).** On the S3 runs at η*:
- validation loss should be flat for k ≥ 1, with the largest step at 0 → 1;
- repeat-token loss at Δ ≥ 1K should be lower for k = 12 than for k = 0.

A deviation triggers an audit of data, masking and position encoding. It never triggers protocol tuning.

---

## 18. Engineering runbook, registry and freeze (I-19, I-24)

**Environment.**
- Pinned versions: torch 2.11.0+cu128, triton 3.6.0, flash-linear-attention 0.6.0 (clone `98193113`), transformers 5.22.2, tokenizers 0.23.3, numpy 2.5.4 [PI-VER: installed versions].
- `uv.lock` is committed. Versions are logged per run.
- **Production environment variables:** `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `PYTHONHASHSEED=0`, `TRITON_CACHE_DIR=$REPO/.cache/triton` (pre-warmed; never cleared mid-study), `FLA_CACHE_RESULTS=1`, `FLA_FLASH_QLA=0`. No cuDNN SDPA in training.
- **Production runs in default (non-deterministic) mode.** Audits and the standing replicate run in deterministic mode (I-19).

**Checkpoints.**
- Full checkpoints (fp32 weights, Adam state, RNG states, step, config hash, git commit, Triton-cache hash) every 250 steps, written atomically (tmp → fsync → `os.replace`). The last 2 are kept.
- Weights-only bf16 milestones at Stage-A tokens {50M, 100M, 200M, 400M}, S, end of Stage B, and TEF {250, 500, 1,000, (2,000, 4,000)}.
- Disk need ≈ 60 GB of 438 GB.

**Runner.**
- A single-GPU queue reads `experiments/registry.jsonl`. It is append-only; status changes are new lines.
- Each job runs as a `systemd-run --user` unit if linger is enabled, otherwise under `nohup`.
- Auto-resume. An OOM is a logged event.
- `nvidia-smi` every 30 s (utilisation, memory per process, clocks, `clocks_event_reasons`, power, temperature) → `contended_frac`.
- Guards: disk space > 50 GB free; throughput (§13.3).

**Run identity.**
- `run_id = {stage}-{scale}-{config}-s{seed}-{config_hash[:8]}`, e.g. `S4-D384-C3-s0-1a2b3c4d`.
- The config hash is SHA-256 of the semantic config.
- Runs refuse to start on a dirty git tree unless `--allow-dirty` (recorded).

**Logging.**
- `runs/<run_id>/metrics.jsonl` every 10 steps: loss, LR, gradient norm, z-loss, tok/s, memory, attention maximum logit and entropy, batch hash.
- Validation subset every 500 (Stage A) / 100 (Stage B) / 250 (TEF) steps.

**G5 freeze.** Commit, and record in a decision-log row with commit hashes:
- this protocol;
- `configs/experiments/*.yaml`;
- `src/`;
- the item-file manifest;
- the analysis code, with VT-18 passing;
- the Triton-cache SHA-256;
- `uv.lock`.

---

## 19. Status of open items and decisions needed from the user

| ID | Item | Decided by | Default if not run |
|---|---|---|---|
| U-01 | Scale | FEAS rule (§13.3) | D512 |
| U-02 | Conv handicap of the reference | δ_gc at the Tier-A decision point | Package wording only |
| U-03 | SSMax | FEAS-5 | Plain per-head gains |
| U-04 | Reference competence; Stage-B budget | FEAS-3, G-FULL, 2× branch | — (required) |
| U-05 | Seeds | §11.4 | n₀ |
| U-06 | Position encoding × k | NoPE trigger | Claims conditional on RoPE |
| U-07 | k* | §3.2 rule | k* = 1 |
| U-08 | TEF share | Forgetting gate | 90/10 |
| U-09 | Learning speed | Dose ladder; still-closing flag | — |
| U-10 | Flex decode comparator | VT-15 | Prefill only |
| U-11 | Equated levels | DG items on C12 | Ladder means only |
| U-12 | Head geometry | Tier C twin | Scope limitation |
| U-13 | GLA tier | Budget rule | Tier C |
| U-14 | Idle windows | **User** | Analytic costs only |
| U-15 | Contexts per cell | FEAS ICC | 300 at 16K/32K |
| U-16 | Elicited vs zero-shot | No experiment decides it | Report both |

**Decisions needed from the user** (defaults in brackets are used if no answer arrives):
1. **CARLA:** can it pause for **two 2-hour idle windows**, for G0 idle throughput, warm-up of benchmark shapes, and the cost benchmarks? [No: measured costs go to an appendix; RQ-D/H6 are analytic.] What is CARLA's peak VRAM? (It was measured at up to 6.25 GB, with ≈ 7.9 GB used by all foreign processes [MEAS-B2].)
2. **Re-scope consent:** if the all-full reference fails at 32K and 16K at both scales, choose between (a) a ≤ 8K question or (b) a larger model, possibly outside 20–60M. [Pause and ask.]
3. **Budget ceiling** (revised, §13.2):
   - D384 path: Tier A ≈ 86 h (103 h with contingency), Tier B + 37 h.
   - D512 path: ≈ 144 h (173 h), + 67 h.
   - Is the D512 economy option (n₀ = 3) acceptable? [Tier A only, then ask.]
4. **"Data fixed":** does it include an identical synthetic fine-tune (TEF) for every configuration? [Yes, as designed.]
5. **Framing:** is H1 answered at the level of layers *as built* (gated MHA vs GDN-A), with operator-level wording only if the conv control allows it? [Yes.]
6. **Job persistence:** run `loginctl enable-linger $USER` so `systemd-run --user` jobs survive logout, or accept `nohup`? [nohup.]

---

## 20. Sources

Access tags are as defined in the conventions; "per X" marks access not re-checked by me.

| Source | Used for | Access |
|---|---|---|
| [README.md](../README.md) | RQs and hypotheses (verbatim) | read-full |
| `reports/reviews/debate_record.md`, `review_A_round1/2.md`, `review_B_round1/2.md` | All resolutions, [MEAS-B], [MEAS-B2], [PI-SIM] | read-full |
| [2412.06464](https://arxiv.org/abs/2412.06464) Gated DeltaNet | Update rule | read-section per PROP |
| [fla @98193113](https://github.com/fla-org/flash-linear-attention) | GDN/GLA layers; mode switch L227–230; `_no_weight_decay`; kernel signatures; parameter counts | code-inspected, PI-VER |
| [2505.06708](https://arxiv.org/html/2505.06708) gated attention | Headwise gate (Table 1) | read-section, PI-VER in debate |
| [transformers](https://github.com/huggingface/transformers) 5.22.2 (venv) | Qwen3-Next gate; `Qwen3Config` fields; Qwen3 q/k-norm | code-inspected, PI-VER |
| [2407.21783](https://arxiv.org/html/2407.21783) Llama 3 §3.2 | RoPE base 500,000, effective to 32,768 | read-section, PI-VER |
| [2501.19399](https://arxiv.org/html/2501.19399) SSMax | Eq. 2; query scaling by s·log n; learnable s per layer and head | read-section, PI-VER |
| [2606.15378v1](https://arxiv.org/html/2606.15378v1) Qiao et al. | NoPE effect; small-scale all-full deficit; 16K pretraining | read-section, PI-VER in debate |
| [2507.06457](https://arxiv.org/abs/2507.06457) Wang et al. | GDN ratio row (prior only) | read-section per A1, A2, B2 |
| [2209.11895](https://arxiv.org/abs/2209.11895) Olsson et al. | Smeared keys | read-section per A1, B2 |
| [2312.04927](https://arxiv.org/abs/2312.04927), [zoology @1ad20d1](https://github.com/HazyResearch/zoology) | Reproduction target and configs | read-section per PROP; code-inspected, PI-VER |
| [RULER @db43f92](https://github.com/NVIDIA/RULER) | Word lists (niah.py L97–98); LFS pointer | code-inspected, PI-VER |
| [FineWeb-Edu API](https://huggingface.co/api/datasets/HuggingFaceFW/fineweb-edu), [emozilla/pg19 API](https://huggingface.co/api/datasets/emozilla/pg19), [deepmind/pg19 card](https://huggingface.co/datasets/deepmind/pg19) | Commit pins; licences | abstract-only (PI-VER for the APIs) |
| [2010.04245](https://arxiv.org/abs/2010.04245) QKNorm | Temperature precedent | read-section per A2 |
| [2103.04514](https://arxiv.org/abs/2103.04514) Summers & Dinneen | Nondeterminism ≈ seed variance | abstract-only per A2, B2 |
| [1608.04295](https://arxiv.org/abs/1608.04295) Chen & Revels | Min-of-N estimator | read-section per B1 |
| [2402.01032](https://arxiv.org/html/2402.01032) Jelassi et al. | Sample-efficiency gap (exposure confound) | read-section per B1 |
| [2403.17844](https://arxiv.org/html/2403.17844) MAD; [2402.18668](https://arxiv.org/abs/2402.18668) Based | Novelty framing and credit (I-23) | read-section per A1; abstract-only |
| [NVIDIA RTX Blackwell whitepaper](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf) | 112.6 TFLOP/s, 960 GB/s | read-section per 04 |
| [2410.02660](https://arxiv.org/abs/2410.02660) ProLong | Long + short data mix | read-section per PROP |
| [2609.33093](https://arxiv.org/abs/2609.33093), [2610.06750](https://arxiv.org/abs/2610.06750), [2610.25250](https://arxiv.org/abs/2610.25250) | Load vs context; pathway imbalance; composition bound (framing) | abstract-only per A1 |

**Not accessed or not re-verified in this step:** every [MEAS-B]/[MEAS-B2] GPU number (no GPU code was run by the PI); Wang et al. appendices; MAD findings; the Stuffed Mamba per-model figures; BABILong internals.

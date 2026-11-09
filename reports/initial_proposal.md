# Initial research proposal (v0.1): How Much Full Attention Does Long Context Need?

*A controlled study of hybrid full/linear attention ratios in ~50M-parameter language models*

**Status:** a draft for the Stage-3 independent review. It is **not frozen**: the protocol freezes at gate G5 (§12).
**Author:** PI consolidation of the five Stage-1 literature reports. **Date:** 2026-11-09.
**Evidence base:** `reports/literature/literature_review.md` (abbreviated *LR* below) and `reports/literature/literature_db.yaml`.
- Citations carry access tags `[read-full]`, `[read-section]`, `[abstract-only]` or `[code-inspected]`.
- **[EST]** marks established findings, **[HYP]** hypotheses, **[INF]** our own inference, design choices or arithmetic.
- **[MEAS]** marks agent measurements on the shared RTX 5080. They are indicative only.

---

## 0. Key decisions at a glance

| Decision | Choice | Main reason (details in the section given) |
|---|---|---|
| Manipulated variable | k = number of full softmax-attention mixers among L = 12 mixers; r = k/12 | Each layer is a block of one mixer and one MLP. Varying r swaps mixers only (§3). |
| Ratio grid | k ∈ {0, 1, 2, 3, 4, 6, 12}, i.e. r ∈ {0, 1/12, 1/6, 1/4, 1/3, 1/2, 1} | Includes both endpoints. Linear:full = pure, 11:1, 5:1, 3:1, 2:1, 1:1, all-full (§7). |
| Linear mechanism (single) | **Gated DeltaNet (GDN)**, FLA 0.6.0 kernels, "config A": 4 heads, d_k = 64, d_v = 128, output gate on, short conv 4 | Strongest standalone recall among the candidates; kernels verified on sm_120; matches MHA's 4d² to within 0.8% with the **same MLP** (§4). |
| Model | L = 12, d = 512, MHA 8 × 64 with QK-RMSNorm, SwiGLU 1536, vocab 16,384 (tied) | 41.0M non-embedding / 49.4M total parameters; spread across k is 0.24% (§6). |
| Default placement | Centred-uniform p_j = ⌊(j + ½)·12/k⌋; layer 0 is always linear for k < 12 | Even spacing (Waleffe); keeps the single k = 1 layer away from the last layer (§7). |
| Placement study | k = 2: centred {3, 9}, early {0, 1}, middle {5, 6}, late {10, 11}, end-of-group {5, 11} | Placement should matter most when attention is scarce (LR D3). |
| Data | Stage A: 1.0B tokens of FineWeb-Edu at 4K. Stage B: 0.1B tokens at **32K**, ≥ 50% PG19 books | Makes every evaluation length 1K–32K in-distribution (H5 validity) (§8). |
| Position encoding | One scheme for all k, chosen by a pilot that favours the r = 1 baseline: {RoPE θ = 1e4, RoPE θ = 5e5, NoPE}; non-chosen family ablated at k ∈ {1, 3, 12} | Position encoding is a first-order confound (LR §7). |
| Evaluation | Token-exact synthetic suite (retrieval / multi-hop / aggregation) at 1K–32K, cloze format, teacher-forced exact match (EM) + log-prob; 400 contexts per cell (×4 branched queries); zero-shot LM metrics | Small models score at the floor on instruction-style benchmarks (LR §8) (§9). |
| Synthetic exposure | Identical for every k: either (i) 1% / 5% synthetic tokens in Stages A / B, or (ii) an identical short fine-tune. Pilot rule decides | Without exposure, multi-hop and aggregation sit at the floor (§9.3). |
| Statistics | 3 seeds for k ∈ {0, 1, 3, 12}, 2 for the rest; hierarchical paired bootstrap **and** seed-level t; Holm over 7 confirmatory tests | Few-seed CIs undercover (LR §10) (§11). |
| Reproduction pilot | Zoology MQAR (pinned commit; attention + GDN) is a gate before any LM training | Validates mixers and trainer against a published small-scale result (§13). |
| Budget | Tier A ≈ 71 GPU-h conservative / ≈ 45 h on an idle GPU; Tier B ≈ 117 / 75; Tier C ≈ 152 / 97 | §12. |

---

## 1. Refined research questions

- **RQ-A, composition.** Hold fixed the parameters (spread ≤ 0.24%), the training tokens and their order, the recipe, the position-encoding scheme and the evaluation items. How does the family accuracy S_f(k, T) change with k ∈ {0, 1, 2, 3, 4, 6, 12}, for T ∈ {1K, 2K, 4K, 8K, 16K, 32K}, all within the training length?
  - Where does the knee sit? (k80_f and k90_f are the smallest k reaching 80% or 90% of the all-full gain.)
  - Does any hybrid match or beat all-full?
- **RQ-B, placement.** At fixed k = 2, does *which* layers are attention change S_f by ≥ 5 pp?
  - Specifically: spread vs clustered, early vs middle vs late, and the industry "end-of-group" convention.
- **RQ-C, task dependence.** Do retrieval, multi-hop composition and aggregation need different k (k80)?
  - Do their gaps to all-full grow differently with context length T and with needle–query distance d?
- **RQ-D, cost.** For each k, measure analytic and actual inference-state bytes, prefill latency and decode throughput (idle GPU, CUDA graphs).
  - Does any hybrid lie above the "router line" between the two pure endpoints in the capability–cost plane?
  - How far apart are analytic and measured costs for this implementation?
- **RQ-E, validity (secondary).** Are the conclusions robust to the position-encoding choice, to the linear layers' short conv, and to the GDN state size?

---

## 2. Hypotheses (falsifiable, pre-registered)

**Definitions.**
- **Families.** f ∈ {ret, hop, agg}. Each family score S_f is the equal-weight mean of its *headline* tasks' accuracy (§9). The score is pooled over in-distribution (ID) lengths unless stated otherwise.
- **Gap and recovery.** G_f = S_f(12) − S_f(0). R_f(k) = (S_f(k) − S_f(0)) / G_f.
- **Inference.** All CIs are 95% CIs from the hierarchical paired bootstrap (§11), and a claim is "supported" only if the seed-level t-interval agrees.
- **Interpretability gate (G2).** A family enters confirmatory tests only if G_f ≥ 15 pp at at least one ID length and the all-full score clears the floor by ≥ 20 pp.

| ID | Hypothesis | IV (levels) | DV | Expected direction | Supported if | Falsified if |
|---|---|---|---|---|---|---|
| **H1** | A small share of full attention recovers most long-context performance | k ∈ {0, 3, 12} (r = 0, 1/4, 1), centred placement | R_ret(3), with S_ret pooled over ID lengths | R ≥ 0.8 | lower CI(R_ret(3)) ≥ 0.80 **and** upper CI(S_ret(12) − S_ret(3)) ≤ 5 pp | upper CI(R_ret(3)) < 0.80 |
| **H1-min** | One full layer suffices for most of it | k ∈ {0, 1, 12} | R_ret(1) | R ≥ 0.8 | as H1, at k = 1 | upper CI(R_ret(1)) < 0.80 |
| **H2** | Returns diminish | k ∈ {0, 1, 2, 3, 4, 6, 12} | S_f(k) | concave: the first layer's gain g₁ = S(1) − S(0) exceeds the late per-layer gain g_late = (S(12) − S(6))/6 | lower CI(g₁ − g_late) > 0 (retrieval family; other families exploratory) | upper CI(g₁ − g_late) < +2 pp, i.e. the first layer is not meaningfully better than late layers (linear or increasing returns) |
| **H3** | The required share depends on the task | family × k | k80_f, linearly interpolated in k | k80_hop > k80_ret; aggregation differs [HYP] | some pairwise k80 difference has a CI excluding 0 **and** the family × k interaction in the GLMM passes Holm | all three pairwise k80 differences have CIs inside [−1, +1] layer (equivalence) |
| **H4** | Placement matters independently of the share | placement ∈ {centred, early, middle, late, end-of-group} at k = 2 | S_ret (pooled ID); other families exploratory | range ≥ 5 pp; early worst, middle or spread best [HYP from LR §5] | ≥ 1 of the 10 pairwise contrasts excludes 0 after Holm **and** that contrast is ≥ 5 pp | all 10 pairwise 90% CIs lie inside ±5 pp (TOST equivalence) |
| **H5** | The benefit of full attention grows with context length, differently per task | T ∈ {1K, …, 32K} (all ID) × k ∈ {0, 1, 3} | gap Δ_k,f(T) = S_f(12, T) − S_f(k, T) | positive slope in log₂T for retrieval; slopes differ across families | CI(slope b_ret, at k = 0) > 0 **and** the family × log₂T interaction passes Holm | all slopes have CIs inside ±2 pp per doubling (equivalence) |
| **H6** | Hybrids occupy a capability–cost region that neither pure model reaches | hybrids k ∈ {1, 2, 3, 4, 6}; cost c = inference-state bytes at T = 32K | hull margin M(k) = S_ret(k) − [S_ret(0) + c̃(k)·G_ret], where c̃(k) = (c(k) − c(0)) / (c(12) − c(0)) | M > 0 for some k | max over the 5 hybrids of lower CI(M) > 0 (Bonferroni ×5) | upper CI(M) ≤ 0 for every hybrid |

**Notes on the hypotheses** [INF].
- **For memory, c̃(k) = k/12 exactly.** M(k, T) = k·KV(T) + (12 − k)·S_GDN, so the difference M(k) − M(0) = k·(KV − S_GDN). M > 0 is therefore equivalent to R(k) > k/12: hybrids must beat a "router" that sends a fraction r of queries to the all-full model.
  - Consequence: if H1 holds at k = 3, H6 holds for memory automatically.
  - H6 adds information mainly through the *measured* costs (prefill latency at 32K, decode throughput at batch 1 and at the largest batch that fits). Those are secondary H6 tests.
- **H1 is demanding.** Suppose G = 30 pp and the seed SD of paired differences is 3 pp. Then the CI half-width of R is about 0.15–0.2. Support then needs R̂ ≳ 1.0, falsification needs R̂ ≲ 0.6, and anything in between is "inconclusive". Gate G3 (reserve seeds) addresses that case.
- **Strong H6 (exploratory).** A hybrid that Pareto-dominates all-full, i.e. S(k) ≥ S(12) at lower cost. Wang et al.'s GDN at 3:1 (0.436) above their Transformer (0.423) makes this plausible ([2507.06457](https://arxiv.org/abs/2507.06457) [read-section]).
- **Confirmatory family.** {H1, H1-min, H2, H3, H4, H5, H6}, Holm-adjusted at α_FW = 0.05. Every other cell is exploratory, with BH-FDR at q = 0.10.

**Predictions** [HYP], written down now so they can be checked later:
- R_ret(3) ≈ 0.9–1.1 within the training length.
- R_hop(3) is lower than R_ret(3).
- Aggregation is weakly sensitive to k.
- The gap Δ_0,ret(T) grows with T.
- A hybrid's state memory at 32K is ≤ 1/4 of all-full's while its accuracy is within 5 pp.

---

## 3. Formal model family

- **Architecture vector.** a = (a₀, …, a₁₁) ∈ {F, Λ}¹², where F is full softmax attention and Λ is linear (GDN). Then k(a) = Σ_l 1[a_l = F], r = k/12, and the placement is P(a) = {l : a_l = F}.
- **The model M(a).** Embedding E ∈ ℝ^{V×d} with tied output.
  - h⁽⁰⁾ = E[x].
  - For each layer l: u = h⁽ˡ⁾ + Mix_{a_l}(RMSNorm₁(h⁽ˡ⁾)), then h⁽ˡ⁺¹⁾ = u + SwiGLU(RMSNorm₂(u)).
  - logits = RMSNorm_f(h⁽¹²⁾) Eᵀ.
- **The only per-layer difference is Mix_{a_l}.** Norms, MLPs, embeddings, initialisation scheme, optimiser, data and schedule are identical.
- **Families studied.**
  - Composition family 𝓒 = {a : P(a) = P_c(k)}, k ∈ K = {0, 1, 2, 3, 4, 6, 12}, with P_c(k) = {⌊(j + ½)·12/k⌋ : j < k}.
  - Placement family 𝓟 = {a : k(a) = 2, P(a) ∈ {P_c(2), {0, 1}, {5, 6}, {10, 11}, {5, 11}}}.
- **Ratio conventions.** r counts mixers. The Nemotron-style "fraction of all sublayers" is k/24 = r/2. So Nemotron-H-8B's "7.7% of layers" equals 14.3% of mixers, which falls between our k = 1 (8.3%) and k = 2 (16.7%) ([01 §1.2](./literature/01_nemotron_h.md); configs [code-inspected]).

---

## 4. The primary linear mechanism: Gated DeltaNet ("config A")

**Mechanism.** Per head, from [2412.06464](https://arxiv.org/abs/2412.06464) Eq. 10 [read-section] and FLA `fla/layers/gated_deltanet.py` @`98193113` [code-inspected]:

- Inputs to the recurrence:
  - q_t = L2(SiLU(Conv₄(x W_q))), k_t = L2(SiLU(Conv₄(x W_k))), v_t = SiLU(Conv₄(x W_v));
  - decay α_t = exp(−exp(A_log) · softplus(x W_a + dt_bias));
  - write strength β_t = σ(x W_b).
- State update: S_t = S_{t−1} · α_t (I − β_t k_t k_tᵀ) + β_t v_t k_tᵀ.
- Output per head: o_t = S_t q_t / √d_k.
- Mixer output: W_o [RMSNorm_head(o_t) ⊙ SiLU(x W_g)].

**FLA call:**
`GatedDeltaNet(hidden_size=512, head_dim=64, num_heads=4, expand_v=2, use_gate=True, use_short_conv=True, conv_size=4, mode="chunk")`.
- Documents are kept separate by passing `cu_seqlens` (exact isolation [MEAS]).
- Decode uses `fused_recurrent_gated_delta_rule` **with keyword arguments only**. Positional calls silently produce NaN [MEAS].
- The recurrent state is kept in fp32.

**Why GDN** [EST evidence; INF choice] (LR §6):
1. Best standalone recall among the candidates:
   - Jet-Nemotron block search, retrieval score: GDN 69.3 vs Mamba2 68.9 ([2508.15884](https://arxiv.org/abs/2508.15884) [read-section]);
   - Wang et al., pure-linear RULER: GDN 0.356 vs DeltaNet 0.320 [read-section].
2. Used in industry: Qwen3-Next (GDN 3:1) [code-inspected], and Kimi's KDA is a GDN refinement [read-section].
3. Pure Triton kernels, verified forward/backward, varlen and decode on sm_120 [MEAS]. By contrast, Mamba-2's fast path needs `mamba_ssm`, which has no torch-2.11 wheel [code-inspected].
4. Can be matched to MHA's parameter count with the MLP unchanged (§6).

**Why config A rather than the alternatives the agents proposed** (LR §13, D1):
- **The output gate stays.** FLA's "crucial" warning is about the **short convolution**, not the gate (gated_deltanet.py L190–191 [code-inspected, PI]). Published GDNs keep the gate and use d_v = 2·d_k; FLA's defaults are `use_gate=True`, `expand_v=2`.
- **Config A's state-to-width ratio matches Qwen3-Next.** Its state is 4 × 64 × 128 = 32,768 = d²/8 elements; Qwen3-Next has 32 value heads × 128 × 128 = d²/8 at d = 2048 [config-inspected].
- **Rejected alternatives:**
  - Gate off, d_k = 128 (+0.99% params): drops a published component.
  - Gate on, 4 heads × 128 (5.04d²): forces the MLP width to differ by block type, so MLP capacity would co-vary with r.
- **The residual concern**, that head_dim 64 is small, is tested by an optional state-size arm (E5c): config D has 2 heads, d_k = 128, d_v = 256, gate on, 1,054,980 params, and a 2× larger state.

---

## 5. Exact layer definition

| Component | Specification |
|---|---|
| Block | Pre-norm: `h = x + Mix(RMSNorm(x)); y = h + SwiGLU(RMSNorm(h))`. RMSNorm eps 1e-5, learnable gain, no bias anywhere. |
| SwiGLU | `W_down( SiLU(W_gate·z) ⊙ (W_up·z) )`, d_ff = 1536 in every block. |
| FullAttn mixer | MHA, 8 heads × 64; W_Q, W_K, W_V, W_O ∈ ℝ^{512×512}. Per-head QK-RMSNorm (64-dim gains, shared across heads), then RoPE if the pilot selects it, scale 1/√64, causal. Training uses document-causal `torch.nn.attention.varlen.varlen_attn` with position ids reset per document. Evaluation and benchmarks use SDPA (flash/cuDNN). No output gate, no conv, no dropout. Our own module; FLA's `Attention` is not used (it needs `flash_attn` and cannot express NoPE [code-inspected]). |
| Linear mixer | GDN config A (§4). |
| Embedding / head | Tied, V = 16,384. Logits in fp32; chunked cross-entropy for T ≥ 16K. |
| Initialisation | Linear weights ~ N(0, 0.02). Output projections (W_O, GDN W_o, MLP W_down) scaled by 1/√(2·12). Norm gains 1. GDN A_log and dt_bias use FLA's defaults (A ~ U(0, 16) in log space, dt ∈ [1e-3, 0.1]). Shared modules are seeded per module path, so they are identical across k for a given seed (§11). |
| Precision | bf16 autocast, fp32 master weights and Adam state, fp32 GDN state. |

---

## 6. Model dimensions and parameter counts

**F12, the primary family.** L = 12, d = 512, MHA 8 × 64, GDN 4 × (64 / 128), d_ff = 1536, V = 16,384 tied. The GDN numbers below were obtained by instantiating `fla.layers.GatedDeltaNet` on the meta device.

**Parameter arithmetic:**
- **Embedding (tied):** 16,384 × 512 = 8,388,608.
- **Attention mixer:** 4 × 512² = 1,048,576, plus QK-norm gains 2 × 64 = 128, total **1,048,704**.
- **GDN-A mixer**, total **1,056,904** (+0.79% vs MHA):
  - W_q 512 × 256 = 131,072 and W_k 131,072;
  - W_v 512 × 512 = 262,144, W_g 262,144 and W_o 262,144;
  - W_a and W_b, 512 × 4 = 2,048 each;
  - convolutions: q 256 × 4 = 1,024, k 1,024, v 512 × 4 = 2,048;
  - per-head output norm 128;
  - A_log 4 and dt_bias 4.
- **SwiGLU:** 3 × 512 × 1536 = **2,359,296**.
- **Block norms:** 2 × 512 = 1,024. **Final norm:** 512.
- **Attention block:** 1,048,704 + 2,359,296 + 1,024 = **3,409,024**. **GDN block:** 1,056,904 + 2,359,296 + 1,024 = **3,417,224**.
- **Non-embedding total:** N(k) = (12 − k)·3,417,224 + k·3,409,024 + 512 = **41,007,200 − 8,200·k**.

| k | r | linear:full | N_non-emb | N_total (tied) |
|---|---|---|---|---|
| 0 | 0 | pure | 41,007,200 | 49,395,808 |
| 1 | 0.083 | 11:1 | 40,999,000 | 49,387,608 |
| 2 | 0.167 | 5:1 | 40,990,800 | 49,379,408 |
| 3 | 0.25 | 3:1 | 40,982,600 | 49,371,208 |
| 4 | 0.333 | 2:1 | 40,974,400 | 49,363,008 |
| 6 | 0.5 | 1:1 | 40,958,000 | 49,346,608 |
| 12 | 1 | all-full | 40,908,800 | 49,297,408 |

- **The spread across k is 98,400 parameters, 0.24%.** This is well inside the methodology agent's ±1% tolerance.
- **Cross-check:** the same formulas with the implementation agent's configuration (GDN gate off, d_k = d_v = 128, no QK-norm) give 49,420,384 / 49,295,872. Those match its measured 49.42M / 49.30M [MEAS].
- **Tokens per parameter:** 1.1B / 41.0M ≈ 27 per non-embedding parameter, ≈ 22 per total parameter. This is roughly Chinchilla's ≈ 20 ([2203.15556](https://arxiv.org/abs/2203.15556) [read-section]; ratio derived).

**Optional families** (E5 only; same formulas):

| Family | Purpose | Shape | N_non-emb at k = 0 / k = L | Total |
|---|---|---|---|---|
| F12-D | GDN state-size sensitivity | GDN config D (2 heads, d_k 128, d_v 256, gate on; 1,054,980 params) | 40,984,112 (−0.06% vs F12) | 49.37M |
| F12-S | Model-size check | d = 384, d_ff 1152, MHA 6 × 64, GDN-A with 3 heads (595,334 params) | 23,078,856 / 23,014,272 | 29.4M |
| F24 | Count vs fraction | L = 24, d = 384, d_ff 960. Its k ∈ {2, 6} matches F12's k ∈ {1, 3} in r with twice the count | 40,848,912 / 40,719,744 (−0.39% vs F12) | 47.1M |

---

## 7. Ratio set and placement patterns realisable with L = 12

Every k from 0 to 12 is realisable. The grid K = {0, 1, 2, 3, 4, 6, 12} covers linear:full ratios ∞, 11:1, 5:1, 3:1, 2:1, 1:1 and 0. It is close to Wang et al.'s 12:1, 6:1 and 3:1 [read-section].
- 7:1 (Jamba, MiniMax) is **not** realisable at L = 12. It needs L = 16 or the optional F24 family (k = 3 of 24).

In the patterns below, `-` is a GDN layer and `A` is full attention, written for layers 0…11.

| Set | k | Positions | Pattern |
|---|---|---|---|
| Composition (centred) | 0 | ∅ | `------------` |
| | 1 | {6} | `------A-----` |
| | 2 | {3, 9} | `---A-----A--` |
| | 3 | {2, 6, 10} | `--A---A---A-` |
| | 4 | {1, 4, 7, 10} | `-A--A--A--A-` |
| | 6 | {1, 3, 5, 7, 9, 11} | `-A-A-A-A-A-A` (strict alternation) |
| | 12 | all | `AAAAAAAAAAAA` |
| Placement (k = 2) | 2 | early {0, 1} | `AA----------` |
| | 2 | middle {5, 6} | `-----AA-----` |
| | 2 | late {10, 11} | `----------AA` |
| | 2 | end-of-group {5, 11} (Qwen3-Next / Olmo / Wang convention) | `-----A-----A` |
| Optional E5e (k = 1) | 1 | {0}, {11} | `A-----------`, `-----------A` |

**What the placement contrasts test:**
- **Spacing:** centred vs middle.
- **Depth:** early vs middle vs late.
- **Whether layer 0 should be linear** (early; Waleffe and Bae, [2406.07887](https://arxiv.org/abs/2406.07887) [read-full], [2510.04800](https://arxiv.org/abs/2510.04800) [read-section]).
- **The industry convention:** end-of-group.

---

## 8. Training data, tokenizer, token budget, curriculum, position-encoding policy

- **Tokenizer.**
  - 16,384-entry byte-level BPE with single-digit isolation (HF `tokenizers` 0.23.3).
  - Trained on 1 GB of text: 80% FineWeb-Edu sample-10BT shard 001 and 20% PG19 train.
  - Special tokens: `<|endoftext|>` (document separator / BOS), `<|pad|>`, and 6 reserved.
  - Saved as `PreTrainedTokenizerFast` with a SHA-256 hash.
  - [MEAS] It produces 9.8% more tokens than GPT-2 on FineWeb-Edu and 10.1% more on PG19. Lengths are reported in our tokens.
- **Stage A** (1.0B tokens, T = 4,096).
  - FineWeb-Edu sample-10BT ([card](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu), ODC-By [abstract-only]), shards 001 onward, with the HF revision pinned.
  - EOS-separated sequential packing, with **intra-document masking in both mixer types**: `varlen_attn` for attention, `cu_seqlens` state reset for GDN [MEAS], and RoPE positions reset per document.
- **Stage B** (0.1B tokens, T = 32,768; 16K only if gate G4 fails).
  - 50% PG19-train books ([card](https://huggingface.co/datasets/deepmind/pg19), Apache-2.0, through the `emozilla/pg19` parquet mirror at a pinned revision [abstract-only]). Books are chunked into 32K windows that respect book boundaries.
  - 45% FineWeb-Edu.
  - 5% synthetic tasks under exposure option (i); with option (ii) the split is 50% / 50% natural.
  - The mix follows the ProLong rule of long plus high-quality short data ([2410.02660](https://arxiv.org/abs/2410.02660) [read-section]).
- **Held out:**
  - FineWeb-Edu shard 000 provides evaluation filler and the LM-1/2/3 documents.
  - PG19 test (100 books) provides LM-1 and evaluation filler.
  - PG19 validation provides development filler.
  - Training PG19 books never appear in evaluation filler.
- **Identical stream for every k.** All data is pre-tokenised once into uint16 shards. The order is a pure function of (data_seed, step) and is identical across k for a given seed index. Batch-index hashes are logged.
- **Optimiser and schedule** (methodology agent's recipe; [Wortsman](https://arxiv.org/abs/2309.14322) [read-section]):
  - AdamW (0.9, 0.95, 1e-8), decoupled weight decay 0.1 on ≥ 2-D weights. A_log, dt_bias and gains are excluded.
  - Gradient clipping 1.0, z-loss 1e-4.
  - Batch 131,072 tokens: 32 × 4K in Stage A, 4 × 32K in Stage B.
  - 8,392 steps in total.
  - WSD schedule: linear warm-up over 500 steps, constant until 0.825B tokens (step 6,294), then linear decay to 0 at 1.1B tokens. The decay spans the end of Stage A and all of Stage B, with no re-warming.
  - Peak LR comes from the E1b sweep (§12).
- **Position encoding (pre-registered selection, baseline-favouring).**
  - Linear layers never get explicit position encoding.
  - For the full layers, pilot E1c trains k = 12 under {RoPE θ = 1e4, RoPE θ = 5e5, NoPE}, each for 0.2B tokens at 4K plus 20M at 32K.
  - **Rule:** pick the scheme that maximises the **all-full** model's dev retrieval accuracy at 32K, with val loss at 32K as the tie-break. Choosing the scheme best for the r = 1 baseline makes H1 harder to support.
  - θ stays constant across stages.
  - The non-chosen family (RoPE ↔ NoPE) is run at k ∈ {1, 3, 12} in E5a.
  - Motivation: NoPE-on-full effects in hybrids (Waleffe [read-full]; [2606.15378](https://arxiv.org/abs/2606.15378), [2501.18795](https://arxiv.org/abs/2501.18795) [read-section]).

---

## 9. Evaluation suite and sample sizes

### 9.1 Shared mechanics
This follows `03_evaluation.md` §5, with the RULER task logic and word lists borrowed under Apache-2.0 ([RULER](https://github.com/NVIDIA/RULER) `db43f92` [code-inspected]).
- **Cloze completions, not instructions.** Values are 4 single-digit tokens.
- **Every sample is exactly T tokens**, built from pre-tokenised segments that are never round-tripped through text. Needles snap to sentence boundaries within ±32 tokens, and actual positions are recorded.
- **Distractors and controls:**
  - same-format distractors in every retrieval task;
  - at least 2 distractor chains in every multi-hop task;
  - the query comes after the context (query-first is a secondary diagnostic only, because it favours recurrent models; [2307.03172](https://arxiv.org/abs/2307.03172), [2407.05483](https://arxiv.org/abs/2407.05483) [read-section]).
- **Metrics:**
  - **teacher-forced exact match (EM_tf)** is the headline; it equals greedy exact match under canonical tokenisation [INF];
  - gold log-probability;
  - rank accuracy among the in-context values plus 4 absent ones;
  - binding, absent and stale error rates.
- **Query branching:** one prefill, then Q = 4 queries scored from a snapshot of the KV cache and recurrent state. A unit test checks that branched scoring equals separate forward passes.
- **Leakage controls:**
  - keys come from word list A in training and list B in evaluation;
  - training uses templates T1–T3; evaluation uses T1 plus an unseen T4 slice;
  - a unit test asserts zero overlap of hashed (key, value, template) tuples.
- **Freezing:** evaluation items come from `eval_seed = 20261109` and dev items from seed 77. Generator code, configurations and item files are hashed and frozen at G5. Every model sees byte-identical items.
- **Lengths:** T ∈ {1K, 2K, 4K, 8K, 16K, 32K}. All are in-distribution when L_ext = 32K. If L_ext = 16K, the 32K cells are extrapolation and excluded from confirmatory tests.

### 9.2 Tasks and cells (per final model)

| Family | Task | Role | Cells | Contexts × queries |
|---|---|---|---|---|
| Retrieval | R1: single key in natural filler with D = 3 same-format distractors | **headline** | 6 lengths | 400 × 4 |
| | R2s: 64 key–value lines spread through natural filler | **headline** | 6 lengths | 400 × 4 |
| | R3: key updates (U ∈ {2, 4}); latest value, stale-value rate | secondary | 1K, 8K, 32K | 400 × 1 |
| | R1 with D ∈ {0, 15}; R2-dense (haystack made only of KV lines, far beyond GDN key capacity) | diagnostic | 4K, 32K | 200 × 4 |
| | R4: position sweep on R1 (11 depths) | diagnostic | 4K, 32K | 100 per depth |
| | Distance-controlled R1: needle–query distance d ∈ {256, 1K, 4K, 16K} | diagnostic (H5 mechanism) | T ∈ {16K, 32K} | 200 × 1 |
| Multi-hop | M1: 2-hop key chain, shuffled link order, 2 distractor chains | **headline** | 6 lengths | 400 × 1 |
| | M1 with hops ∈ {1, 4} × link order {forward, shuffled, reversed}; hops = 6 (out of distribution) | diagnostic | 4K, 32K | 200 × 1 |
| | M2: variable tracking, 4 chains, ordered output | secondary | 1K, 8K, 32K | 400 × 1 |
| Aggregation | A1: counting, c ∈ 1…9 with 4 distractor categories | **headline** | 6 lengths | 400 × 4 |
| | A2: frequency comparison at ratio 2:1 | **headline** | 6 lengths | 400 × 4 |
| | A3: gather 4 values in order of appearance | secondary | 1K, 8K, 32K | 400 × 1 |
| | A2 at ratios {3:1, 1.5:1}; A4 RULER common-word extraction (generation) | diagnostic | 4K (and 32K) | 200 / 100 |
| Controls | C0 (no filler), C2 (relevant facts placed right before the query) | every headline task | C0 short; C2 at 4K, 32K | 200 |

**Zero-shot tier 0** (all models, base checkpoint):
- **LM-1:** per-position loss on ≥ 300 held-out FineWeb-Edu documents and 100 PG19-test books (first 32,768 tokens), in log-spaced buckets.
- **LM-2:** validation loss at 4K.
- **LM-3:** loss on repeated tokens, resolved by distance (4-gram repeats, top-1% most frequent 4-grams excluded, Δ buckets from 64 to 32K). It is model-free, generalising Zoology's associative-recall hits ([2312.04927](https://arxiv.org/abs/2312.04927) [read-section]).
- **LM-4:** copy-after-gap, with 256-token natural or random passages and gaps {0, 512, 2K, 8K, 30K}, 200 items each.
- **R0:** cloze passkey as a sanity check.

**Tier 2 (optional E5):**
- BABILong qa1–qa3 after fine-tuning ([2406.10149](https://arxiv.org/abs/2406.10149) [read-section]);
- lm-eval `ruler` on 2–3 public small checkpoints, used only to cross-check our generator.

### 9.3 Synthetic exposure (needed for RQ-C; identical for every k)
- **Option (i):**
  - Stage A contains 1% synthetic items, 256–4K tokens long, covering all task families at in-distribution difficulty.
  - Stage B contains 5% synthetic items, 4K–32K tokens long.
- **Option (ii):** a natural-only base model, then an identical fine-tune of 500 steps × 64K tokens (≈ 33M tokens) on the synthetic mix at 1K–32K, constant LR 3e-4. Tier 0 is measured *before* fine-tuning.
- **Pre-registered choice (E1d):**
  - Select the option where the all-full pilot model reaches **40–85%** on each headline task at in-distribution lengths. The band is lower than the evaluation agent's 70–95% because pilot models see only 0.2B tokens.
  - The endpoint gap must be ≥ 3× the planned CI half-width.
  - Option (i) wins ties.
  - Difficulty knobs (D, number of pairs, hops, count range, frequency ratio) are tuned on **dev items only**, then frozen.
  - Birdie shows fixed mixtures can fail for SSMs ([2411.01030](https://arxiv.org/abs/2411.01030) [read-section]). A failure of k = 0 to learn the tasks is reported as a finding, with learning curves.

### 9.4 Sample sizes
- **Per headline cell:** 400 contexts × 4 queries. At an intraclass correlation (ICC) of 0.2 the design effect is 1.6, so n_eff ≈ 1000.
- **Paired minimum detectable effect** (MDE, α = .05, power .8, ω² = .25; [Miller](https://arxiv.org/abs/2411.00640) [read-section]): ≈ 4.4 pp per cell. Single-query cells (n = 400) give 7.0 pp. Pooling over 6 lengths × 2 tasks per family gives about 1.3 pp, at which point seed variance dominates (§11).
- **If the pilot ICC gives a design effect > 2**, contexts rise to 600 for the confirmatory retrieval family.
- **Dev suite for checkpoints:** 100 contexts at {1K, 4K, 16K}.
- **Evaluation compute estimate** [INF]:
  - the full suite is ≈ 390M tokens per model, ≈ 1.0 h at k = 12 and ≈ 0.35 h at k = 0 (assuming 40 TFLOP/s effective, unmeasured);
  - the core suite (headline + secondary + C0 + tier 0) is ≈ 0.45 h;
  - total ≈ 18–20 GPU-h (§12).

---

## 10. Efficiency measurement protocol (RQ-D)

- **Analytic inference-state memory per sequence:** M(k, T) = k · 2·d·T·2 B + (12 − k) · (H·d_k·d_v·4 B + conv cache).
  - GDN-A state is 131,072 B, plus a 16,384 B conv cache if FLA stores it in fp32 at width 4 [INF from the implementation agent's measured layout]. The total is 144 KiB per layer.
  - The break-even point against one attention layer's KV cache is 72 tokens. Full table in Appendix A.
- **Measured memory:** the sum of `nbytes` over cache tensors after a prefill of T tokens, plus Δ`max_memory_allocated`. Only last-token logits are computed: full logits at 16K × 16K vocab would add 512 MB ([MEAS], implementation agent).
- **Prefill latency:** batch 1, T ∈ {1K, …, 32K}; also batch 8 at T ≤ 8K.
- **Decode throughput:** a static-cache decode loop captured with `torch.cuda.CUDAGraph`, with the GDN state `copy_`-ed into static buffers. Batch ∈ {1, 8, 32, 128} where it fits, context ∈ {1K, 4K, 16K, 32K}, 256 generated tokens. Report ms/token and tokens/s.
  - Eager HF-style decode is launch-bound at about 5–7 ms/token, independent of architecture [MEAS], so it is **not** used.
- **Capacity metric:** the largest batch that fits in an 8 GB state-plus-activation budget at each T.
- **Conditions:**
  - an idle GPU (gate G-cost: no foreign compute process for ≥ 60 s, utilisation < 5%), with clocks, temperature and power logged;
  - CUDA-event timing: 3 warm-ups, ≥ 10 timed reps, median and IQR;
  - configs interleaved in random order;
  - **2 sessions on different days**;
  - pre-warmed Triton cache, `FLA_FLASH_QLA=0`, pinned software (uv.lock).
- **Reporting:** measured numbers next to analytic FLOPs and bytes. Claims are scoped to "this implementation" (fla Triton + PyTorch SDPA on sm_120).
- **Analytic GQA counterfactual** (KV × n_kv/8) as a sensitivity, without training.

---

## 11. Seeds and statistics

- **RNG streams** derive from `SeedSequence(PROJECT_SALT, spawn_key = (seed_index, stream, …))`:
  - initialisation is seeded per module path, so shared modules start identical across k;
  - data order is shared across k for a given seed index (blocking);
  - evaluation and dev generation use fixed seeds.
- **Seed indices:** main runs use 0, 1, 2. Seeds 3 and 4 are held in reserve and are added only through gate G3. The LR sweep uses 100+ and pilots use 101–103. No seed may be dropped or replaced; a run that diverges is reported.
- **Seeds per configuration:**
  - 3 for k ∈ {0, 1, 3, 12} (the H1, H1-min and H6 anchors);
  - 2 for k ∈ {2, 4, 6} and for each placement arm;
  - 1 for each E5 robustness arm.
- **Primary inference:** a hierarchical paired bootstrap with B = 10,000. Each replicate resamples seed indices, applying the same indices to every configuration, and then resamples contexts within (task, length) strata, again with the same indices for every configuration. CIs are percentile CIs.
- **Seed-level paired t** (df = S − 1) as a cross-check. A claim is "supported" only if **both** intervals exclude the null or threshold; if only one does, it is "suggestive". Bouthillier's P(A > B) is reported for every confirmatory contrast ([2103.03098](https://arxiv.org/abs/2103.03098) [read-section]).
- **Secondary model:** a binomial GLMM, `correct ~ k × log2T × family + (1|item) + (1|seed:config)`, used for the interaction tests in H3 and H5.
- **Multiplicity:** Holm over the 7 confirmatory tests; BH-FDR at q = 0.10 for exploratory cells. Any "no difference" claim requires TOST against the pre-registered margins:
  - 5 pp accuracy;
  - recovery 0.8;
  - 0.01 nats LM loss;
  - 10% throughput or latency.
- **Power** [INF]. The methodology agent's formula is SE ≈ √(σ_d²/S + SE_items²).
  - With a seed SD of paired differences σ_d = 3 pp and S = 3, the family-level MDE is ≈ 6 pp.
  - If the pilot measures σ_d > 5 pp, gate G3 adds seeds 3 and 4 to all four confirmatory configurations (+8 runs).
- **Reference noise level:** LM-loss seed noise at this scale is about 0.002 nats ([modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) [code-inspected]).

---

## 12. Staged plan, go/no-go gates and GPU-hour estimates (RTX 5080)

**Throughput assumptions** [INF from MEAS on a contended GPU]:
- **Stage A:** 80K tok/s for every k (measured at micro-batch 1: 64–104K), so 1.0B tokens take 3.47 h.
- **Stage B at 32K:** predicted by the single-layer measurements plus 33% for activation checkpointing:
  - k = 0: 0.42 h; k = 1: 0.45 h; k = 2: 0.48 h; k = 3: 0.51 h; k = 4: 0.55 h; k = 6: 0.61 h; k = 12: 0.81 h.
- **Full run:** 3.9–4.3 h conservative.
- **"Idle" column:** assumes a 1.6× speed-up on an idle GPU. This is an assumption; G0 replaces it with a measurement.

| Phase | Content | Runs | GPU-h, conservative | GPU-h, idle |
|---|---|---|---|---|
| **E0** Infrastructure | Unit tests (below); determinism audit; kill-and-resume test; throughput and VRAM at 4K and 32K for k ∈ {0, 3, 12}; warm the Triton cache on an idle GPU | short | 3 | 2 |
| **E1a** Reproduction pilot RP-1 | Zoology MQAR, attention + GDN (§13) | ≤ 24 tiny | 4 | 3 |
| **E1b** LR sweep | k ∈ {0, 3, 12} × LR {1.0, 1.4, 2.0, 2.8, 4.0}e-3 × 0.2B tokens at 4K, seed 100; selected on 4K val loss only | 15 | 10.4 | 6.5 |
| **E1c** Position-encoding pilot | k = 12 × {RoPE 1e4 (reuses the E1b run), RoPE 5e5, NoPE}, each 0.2B + 20M tokens at 32K | 2 + 3 extensions | 2.2 | 1.4 |
| **E1d** Exposure, discriminability and seed pilot | (i): k ∈ {0, 12} × seeds 101–103 × (0.2B + 20M); (ii): extend and fine-tune the E1b endpoint runs; dev evaluation | 6 + 2 | 7.3 | 4.6 |
| **G5 freeze** | Protocol, configs, analysis code and item files hashed | — | — | — |
| **E2** Confirmatory core | k ∈ {0, 1, 3, 12} × seeds {0, 1, 2}, 1.0B tokens at 4K + 0.1B at 32K | 12 | 48.2 | 30.1 |
| **E3** Curve | k ∈ {2, 4, 6} × seeds {0, 1} | 6 | 24.1 | 15.1 |
| **E4** Placement | k = 2: early, middle, late, end-of-group × seeds {0, 1}; centred is shared with E3 | 8 | 31.6 | 19.8 |
| Evaluation | Full suite for E2 (≈ 8.5 h), core suite for E3 and E4 (≈ 6.3 h), dev curves at 4 milestones (≈ 3.5 h) | — | 18.3 | ~12 |
| Cost benchmarks | Idle GPU only, 7 values of k, 2 sessions | — | 3 | 3 |
| **Total E0–E4** | | ≈ 53 | **≈ 152** | **≈ 97** |

**Tiers**, in order of execution, with a stop or continue decision at each:
- **Tier A** = E0 + E1 + E2 restricted to k ∈ {0, 3, 12} (9 runs) + their evaluation + a subset of benchmarks. **≈ 71 h conservative, ≈ 45 h idle.** This settles H1 and gives a first look at H6.
- **Tier B** = Tier A + k = 1 × 3 seeds + E3 + evaluation and benchmarks. **≈ 117 h, ≈ 75 h idle.** Adds H1-min, H2, H3, H5 and H6.
- **Tier C** = Tier B + E4, i.e. all of E0–E4. **≈ 152 h, ≈ 97 h idle.** Adds H4.
- **Conditional (gate G3 triggers):** +8 runs, ≈ +32 h.

**E5 robustness**, optional, in priority order:

| Priority | Arm | Runs | GPU-h | Why |
|---|---|---|---|---|
| a | Non-chosen position-encoding family at k ∈ {1, 3, 12} | 3 | 12.2 | Position encoding × r interaction (H5 and H1 validity) |
| b | Attn+conv: all-full with GDN's short conv on q, k, v (conditional: run if any hybrid beats k = 12) | 1 | 4.3 | Tells a conv effect apart from complementarity (§14) |
| c | GDN state size: F12-D at k ∈ {0, 1} | 2 | 7.8 | Does "how much attention" depend on linear state size? |
| d | Second mechanism: GLA (4 heads, 64/128, conv on) at k ∈ {0, 1, 3} | 3 | ~11.8 | Is the curve shape specific to the mechanism? |
| e | Single-layer placement k = 1 at {0} and {11} | 2 | 7.8 | Where to put a single layer |
| f | Model size: F12-S at k ∈ {0, 3, 12} | 3 | ~7 | Is the curve shape stable across size? |
| g | Budget: 3B tokens at k ∈ {3, 12} | 2 | ~22 | Rules out undertraining |

**Gates**, with thresholds adapted from `04_methodology.md` §8:

| Gate | When | Pass criteria | If it fails |
|---|---|---|---|
| **G0** infrastructure | before E1 | (a) Two 200-step runs are bitwise identical in audit mode. (b) A production rerun differs by < 0.01 in val loss at step 1000. (c) Kill-and-resume gives identical batch hashes and Δloss < 0.005 at resume + 500. (d) Idle throughput ≥ 50K tok/s at 4K. (e) Peak VRAM ≤ 10 GB in every stage. (f) **Unit tests:** GDN chunk vs naive recurrence max abs error ≤ 5e-4 (fp32) and ≤ 2e-3 (bf16); prefill + decode vs full forward argmax agreement 1.000 in fp32; document isolation exact for GDN and within bf16 tolerance for attention; parameter counts equal §6 exactly; branched scoring equals separate forward passes (\|Δ logp\| < 1e-3 in fp32); decode kernel called by keyword only | Fix the infrastructure. If (d) fails, fall back to 0.6B tokens in Stage A or to F12-S (pre-registered) |
| **G-RP** reproduction | after E1a | §13 criteria | Debug mixers or environment before E1b |
| **G1** LR | after E1b | No divergence; minimax-regret shared LR with regret ≤ 0.02 nats | Otherwise use per-config LRs (intermediate k interpolated log-linearly) |
| **G2** discriminability | after E1d | For each family at ≥ 1 in-distribution length: gap ≥ 15 pp; all-full within the target band; no cell with both endpoints < 10% or both > 95% | Retune difficulty on dev items. A family that still fails is demoted to exploratory |
| **G3** seed variance | after E1d | σ_d(retrieval, in-distribution) ≤ 5 pp; σ_seed(LM loss) ≤ 0.01 | Add seeds 3 and 4 to the confirmatory configurations |
| **G4** 32K feasibility | E0 / E1c | k = 12 Stage B at 32K fits in ≤ 10 GB with checkpointing + chunked cross-entropy, and runs at ≥ 10K tok/s | L_ext = 16K; 32K cells become extrapolation |
| **G5** freeze | before E2 | Everything committed with hashes; decision-log entry made by its owner | No launch |
| **G-div** | during runs | NaN, or loss > EMA + 1.0 for more than 50 steps | One automatic restart; a second failure is reported as a finding |
| **G-cost** | before benchmarks | idle-GPU criteria (§10) | Defer the benchmark; contended timings are never primary |

**Infrastructure** (as `04_methodology.md` §11–14):
- frozen-dataclass and YAML configs with SHA-256 config hashes;
- an append-only `experiments/registry.jsonl`;
- item-level evaluation JSONL; per-run `metrics.jsonl`;
- atomic checkpoints every 500 steps or 30 min;
- jobs run through `systemd-run --user`, which needs the user to enable linger; otherwise `nohup`.

---

## 13. Paper-reproduction pilot

**RP-0, numerical replication (part of G0).** Re-run the agents' kernel checks as unit tests:
- GDN chunk vs naive fp32 recurrence: 2.9e-4 (fp32) and 1.2e-3 (bf16) were measured by the architecture agent;
- cached decode vs full forward: argmax agreement 1.000, measured by the implementation agent.

**RP-1, a published small-scale result (gate G-RP).** Zoology's MQAR recall–state trade-off for attention and Gated DeltaNet ([paper](https://arxiv.org/abs/2312.04927) [read-section]; [repo](https://github.com/HazyResearch/zoology) @`1ad20d193b6113cae1e8f3c655c300d7b4b3f4bb` [code-inspected]).
- **Config:** `zoology/experiments/mqar_example_configs/original_mqar_configs.py`, restricted to the `attention` and `gated_delta_net` model families:
  - 2-layer models;
  - attention at d_model ∈ {32, 64, 128}; GDN at d_model ∈ {64, 128, 256} with 2 heads, `use_gate=False` and short conv 4, as Zoology specifies;
  - a BaseConv short-conv mixer in front of each, as in Zoology's recipe;
  - training sets (64, 4 KV) × 100K and (128, 8), (256, 16 / 32 / 64) × 20K each; test sets up to (1024, 256 KV); V = 8,192;
  - 4 LRs from logspace(−3, −1.5); 32 epochs; batch 256.
- **Published target:** the README "Recall-Memory Tradeoff" figure (`zoology/analysis/paper/zoology_paper/zoology_results.png` at the same commit).
  - GDN rises from ≈ 0.89 at its smallest state to ≈ 0.99–1.0 at larger states.
  - Attention reaches ≈ 1.0 at its larger settings.
  - Values were read by eye; the state-size units are Zoology's own.
- **Pass criteria:**
  - (a) Best-over-LR test accuracy of Zoology's own models: GDN ≥ 0.95 at d_model ≥ 128, attention ≥ 0.95 at d_model ≥ 64, and GDN accuracy increasing with state size, as in the figure.
  - (b) **Our** mixer modules (our SDPA attention; our fla-GDN wrapper in Zoology's geometry), placed in the same 2-layer harness with the same data, score within ±0.03 of Zoology's mixers at matched d_model and LR.
  - (c) Exploratory: config-A head geometry (d_k = 64, d_v = 128) at d_model ∈ {128, 256}, to calibrate the key-capacity interpretation of R2-dense.
- **Cost:** time the first run. Budget ≤ 4 GPU-h. If one run takes more than 15 min, cut to the 2 best LRs per model (pre-declared).
- **Why this target:**
  - it is cheap;
  - it has public code at a pinned commit;
  - it exercises both mixers and the training loop on the capability at the centre of H1 (associative recall);
  - it is the canonical "recall vs state size" framing behind our memory axis ([Based](https://arxiv.org/abs/2402.18668) [read-section]).

**RP-2, LM-level consistency with published patterns (not a gate; a bug detector).** From the E1b runs at η* (0.2B tokens):
- (i) The val-loss pattern should match the "flat for k ≥ 1, with the largest step at 0 → 1" shape seen in Mamba-2, Waleffe and Kimi ([2405.21060](https://arxiv.org/abs/2405.21060) [read-section], [2406.07887](https://arxiv.org/abs/2406.07887) [read-full], [2510.26692](https://arxiv.org/abs/2510.26692) [read-section]).
- (ii) Repeated-token loss at Δ ≥ 1K should be lower for k = 12 than for k = 0 (Zoology's associative-recall-hit pattern).
- A deviation triggers an audit of data, masking and position encoding. It does not trigger protocol tuning.

---

## 14. Competing explanations for likely outcomes, and the experiment that tells them apart

| Likely outcome | Competing explanations | Discriminating experiment or analysis |
|---|---|---|
| **O1.** Retrieval at k ∈ {1, 3} ≈ k = 12 (H1 supported) | (a) A few attention layers implement retrieval (Wen et al., [2402.18510](https://arxiv.org/abs/2402.18510) [abstract-only]). (b) The tasks fit within GDN's state capacity, so attention is not needed. (c) A shortcut (lexical anomaly) solves the task | G2 requires S(0) ≪ S(12). R2-dense (many more pairs than d_k) must separate the endpoints. The D = 0 vs D ≥ 3 contrast and the binding-error rate expose shortcuts. C0/C2 controls |
| **O2.** Some hybrid beats all-full (strong H6) | (a) Complementarity between local and global processing. (b) GDN's short conv (Kimi: no conv 5.70 vs 5.65 val ppl [read-section]). (c) Long relative positions are rarely trained in the all-full model even within 32K (STRING, [2410.18745](https://arxiv.org/abs/2410.18745) [read-section]) or the RoPE choice. (d) The shared LR favours hybrids. (e) The all-full model learns the synthetic formats more slowly under a fixed exposure. (f) Output gate or activation-spike effects ([2608.12149](https://arxiv.org/abs/2608.12149) [abstract-only]) | (b) Attn+conv control (E5b). (c) Position-encoding ablation (E5a), the distance-controlled design and per-position loss. (d) The G1 regret rule plus per-config-LR fallback. (e) Learning curves from Stage-B and fine-tuning checkpoints. (f) Activation-spike diagnostics, exploratory |
| **O3.** The gap grows with T (H5) | (a) Linear-state interference grows with the amount of filler. (b) Distance itself (decay of α_t). (c) Extrapolation, if any cell is beyond L_ext. (d) The all-full model under-uses long relative positions | Distance-controlled design: fix d and vary T, and vice versa. Only in-distribution cells are confirmatory. State-size arm (E5c): does a 2× state shift the T at which failure starts? Per-position loss |
| **O4.** Multi-hop needs more attention than retrieval (H3) | (a) Composition needs ≥ 2 attention operations (depth). (b) Multi-hop is simply harder, giving floor effects. (c) Link order interacts with recurrence (JRT, [2407.05483](https://arxiv.org/abs/2407.05483) [read-section]) | Compare hops 1, 2 and 4 at the same k and positions. Compare adjacent ({5, 6}) with spread ({3, 9}) placement at k = 2 on M1. Link-order factor. G2 applied per family |
| **O5.** Aggregation is insensitive to k | (a) Counting is not attention-limited at this scale. (b) Floor or ceiling | G2. A2 is binary with a known chance level, which gives tight CIs. Diagnostic ratios 3:1 / 1.5:1 |
| **O6.** A placement effect (H4) | (a) Layer depth plays a specific role (middle-layer retrieval, LR §5). (b) First-layer attention lacks conv-derived features. (c) Last-layer attention cannot feed later processing. (d) Seed noise | Early vs middle vs late; end-of-group vs centred; optional k = 1 at {0} / {6} / {11} (E5e); TOST; seeds |
| **O7.** LM loss flat while retrieval differs | Not an alternative explanation: this is a validity check consistent with Wang et al. and Qiao et al. | Report both. Never select on loss |
| **O8.** Measured speed gains smaller than analytic | (a) Kernel-launch overhead. (b) Triton GDN efficiency on sm_120. (c) Batch × T too small to be bandwidth-bound | CUDA-graph decode; batch sweep; analytic bytes and FLOPs; largest batch within the memory budget |
| **O9.** H1 fails (R(3) < 0.8) | (a) Genuine. (b) Undertraining: retrieval circuits have not formed by 1.1B tokens. (c) Exposure too small for the formats | Learning curves (still rising at the end?). 3B-token check (E5g). Fine-tune budget sweep (250 / 1000 / 4000 steps) on the endpoints |

---

## 15. Minimum experiment that would falsify H1

- **Prerequisites:** the G0 / G-RP / G1 / G2 / G4 pilots, about 27 GPU-h conservative or 17 h idle.
- **Configurations:** k ∈ {0, 3, 12} (r = 0, 1/4, 1) × seeds {0, 1, 2}, each trained for 1.0B tokens at 4K plus 0.1B at 32K. That is 9 runs, ≈ 36.4 GPU-h conservative or ≈ 23 h idle.
- **Evaluation:** the retrieval family (R1 with D = 3 and R2s) at ≥ 3 in-distribution lengths × 400 contexts × 4 queries. ≈ 3 GPU-h.
- **Decision rule:** H1 is **falsified** if the upper 95% CI of R_ret(3) is below 0.80 under *both* the hierarchical bootstrap and the seed-level t-interval.
- **Logic:** if 3 of 12 mixers do not recover 80% of the all-full gain, smaller shares are very unlikely to do so, **assuming S is monotone in k**. That assumption is checked by E3.
- **Total:** ≈ 66 GPU-h conservative, including pilots, or ≈ 43 h on an idle GPU. This is Tier A minus the k = 1 runs and the extended evaluation.

---

## 16. Main risks and mitigations

1. **Floor or ceiling effects at 50M.** Mitigations: gate G2, dev-only difficulty tuning, two difficulty levels as diagnostics, continuous metrics (log-prob, rank).
2. **A shared GPU (CARLA).** Contention invalidates timings and can change Triton autotune choices. Mitigations: idle windows for G0, cache warm-up and benchmarks; a `contended_frac` column in the registry; production runs need only *statistical* reproducibility.
3. **Position-encoding confound.** Mitigations: baseline-favouring pilot plus the E5a ablation; confirmatory cells are in-distribution only.
4. **Exposure confound (learnability vs capability).** Mitigations: identical exposure for every k; learning curves; the fine-tune budget sweep (O9).
5. **Implementation asymmetries.** Document masking in both mixers, identical initialisation of shared modules, and unit tests in G0 (f).
6. **Seed variance in long-context metrics** ([2402.09371](https://arxiv.org/abs/2402.09371) [abstract-only]). Mitigations: 3 seeds and G3 reserve seeds.
7. **Scale transfer.** Wang et al.'s 340M recall was at the floor ([2507.06457](https://arxiv.org/abs/2507.06457) v1 Table 2 [read-section]), and MiniMax reports that small-scale parity hid multi-hop deficits at scale ([blog](https://huggingface.co/blog/MiniMax-AI/why-did-m2-end-up-as-a-full-attention-model) [read-full]). Every claim is scoped to ≈ 50M parameters, 1.1B tokens, GDN and this recipe.
8. **Budget overrun.** Mitigations: tiers, with stop or continue after each; the pre-registered fallback to 0.6B tokens or F12-S if G0 throughput fails.
9. **Software drift in the shared venv** (installing fla changed transformers). Pin torch 2.11.0+cu128, triton 3.6.0, fla 0.6.0, transformers 5.22.2 and tokenizers 0.23.3 in `uv.lock`, and log versions per run.

---

## 17. Open issues and decisions needed

**For the user:**
1. **CARLA.** Can it be paused for idle windows? About 2 h is needed for G0 throughput and Triton cache warm-up, and 2 × ~1.5 h for the cost benchmarks. Training itself tolerates sharing if VRAM stays ≤ 10 GB.
2. **Job persistence.** Run `loginctl enable-linger` so `systemd-run --user` jobs survive logout, or accept the `nohup` fallback.
3. **Budget tier.** A (≈ 45 h idle), B (≈ 75 h) or C (≈ 97 h); conservative figures are 71 / 117 / 152 h.
4. **"Nemotron Edge".** Confirm that it means Nemotron-Flash ([2511.18890](https://arxiv.org/abs/2511.18890)). No NVIDIA artefact has that exact name ([01 §3](./literature/01_nemotron_h.md)).
5. **Synthetic exposure.** Accept that "data fixed" means "identical data *including* a small synthetic-task fraction or fine-tune". Without it, RQ-C (multi-hop and aggregation) is probably unmeasurable at this scale.

**For the Stage-3 reviewers** (each has an LR §13 entry):
- Is GDN config A (gate on, d_k = 64) acceptable? The alternatives are config B (gate off) and config D (2 heads, d_k = 128).
- Is QK-norm on acceptable? Unresolved U4 lists what would reopen it.
- Should placement be studied at k = 2 rather than k = 3?
- Are the H1 thresholds (R ≥ 0.8, a 5 pp non-inferiority margin) right?
- Is L = 12 fine-grained enough at small r? F24 is the remedy.
- Should E5a (position-encoding ablation) and E5b (Attn+conv) be promoted into the core, at about +17 h? The PI leans yes for E5a.

**Still unresolved** (LR §13.1): U1 (Wang et al.'s context length and parameter matching), U2 (Waleffe's sweep details), U3 (exposure option; empirical), U6 (how much the answer depends on GDN state size), U7 (idle-GPU multiplier).

**Ownership:** this proposal does not edit `README.md` or `reports/decisions.md`. Decisions D1–D20 should be entered in the decision log by its owner after review.

---

## Appendix A. Analytic inference-state memory per sequence (MiB)

KV cache: bf16, 2 KiB per token per attention layer. GDN-A: 144 KiB per layer (128 KiB fp32 state + 16 KiB conv cache).

| T | k = 0 | k = 1 | k = 2 | k = 3 | k = 4 | k = 6 | k = 12 |
|---|---|---|---|---|---|---|---|
| 1,024 | 1.69 | 3.55 | 5.41 | 7.27 | 9.12 | 12.84 | 24.00 |
| 4,096 | 1.69 | 9.55 | 17.41 | 25.27 | 33.12 | 48.84 | 96.00 |
| 32,768 | 1.69 | 65.55 | 129.41 | 193.27 | 257.12 | 384.84 | 768.00 |

**Forward FLOPs per token** at T = 32K, with attention scores at 2·T·d per attention layer (causal average):
- k = 0: ≈ 102M;
- k = 1: ≈ 135M (attention 25% of the total);
- k = 3: ≈ 202M (50%);
- k = 12: ≈ 501M (80%).

The dense part is 2·N_non-emb + 2·d·V ≈ 98.8M FLOPs per token.

# 01: Nemotron-H and its NVIDIA relatives: what they show, and do not show, about full-attention ratios

**Author role:** hybrid-architecture specialist (Stage 1 literature).
**Date:** 2026-11-09.
**Scope:** Nemotron-H (arXiv 2504.03624) read in full. Its predecessor, Waleffe et al. 2024 (arXiv 2406.07887), the 8B Mamba-2-Hybrid study that ran the attention-ratio ablation, also read in full. Then the NVIDIA follow-ups (Nemotron Nano 2, Nemotron-Flash, Nemotron 3 Nano, plus Jet-Nemotron and Hymba as relatives), the official Hugging Face `config.json` files, and the Megatron-LM hybrid code.

**Access-level tags** (used throughout):
- `[read-full]`: the whole paper, appendices included.
- `[read-section]`: only the named sections.
- `[abstract-only]`: the abstract only.
- `[code-inspected]`: files I opened and parsed myself.

**Evidence tags:**
- **[EST]** an established finding: the paper reports it with data.
- **[HYP]** a hypothesis or conjecture stated by the authors.
- **[INF]** my own inference, arithmetic or interpretation.

---

## 0. TL;DR for the project

1. **Nemotron-H never ablates the attention fraction.** It inherits "~8% of layers are self-attention, evenly dispersed" from Waleffe et al. 2024. In its own words: "As suggested by prior work (Waleffe et al., 2024), we set the number of attention layers to be roughly 8%" (§2.1, https://arxiv.org/html/2504.03624 `[read-full]`). Its comparisons also change several things at once. Nemotron-H-8B vs Nemotron-T-8B differs in depth (52 vs 32 layers), mixer type, and RoPE vs no position embedding. So Nemotron-H is **not** a controlled full-vs-linear ratio experiment and must never be cited as one.
2. **The only controlled ratio sweep in this family is Waleffe et al. Fig. 4.**
   - Setup: 130M-parameter, 24-layer Mamba-2/attention hybrids, parameter-matched per layer.
   - Metric: **validation loss only**. No retrieval, long-context or NIAH metric was measured in that sweep.
   - Values I read off the figure (digitized by eye, ±~0.002), as attention layers out of 24 → loss: 0/24 → 2.615; 1/24 → 2.562; 2/24 → 2.554; 3/24 → 2.555; 4/24 → 2.555; 5/24 → 2.560; 7/24 → 2.568; 12/24 → 2.593; 24/24 → 2.631.
   - **One attention layer captures ~87% of the loss gain** from 0 to the best ratio [INF from the digitized values]. The curve is U-shaped, and 100% attention is the worst point.
   - The MLP fraction used during this sweep is **not stated**, so the right half of the curve may be confounded (see §2.3) (https://arxiv.org/html/2406.07887 §4.1; figure checked in the PDF https://arxiv.org/pdf/2406.07887 p.9 `[read-full]`).
3. **Evidence on long context in this family is generally positive for few-attention hybrids, but it depends on the task.** All of it is confounded.
   - The 8B hybrid (4/56 attention layers) beats a same-data Transformer on RULER at 4K, 16K and 32K: average 81.99 vs 77.62, 74.19 vs 62.33, 69.93 vs 56.49.
   - It loses on LongBench multi-document QA (16K HotpotQA 42.17 vs 48.61) and on RULER keyword extraction (KWE) at every length [EST] (Waleffe Tables 9–10).
   - Nemotron-Flash-3B: going from 3 full-attention (FA) layers to 1 FA + 2 sliding-window (SWA) layers leaves short-context "recall" benchmarks unchanged (73.25 vs 73.77). But RULER `niah_single_3` at 32K drops from 0.69 to 0.32 [EST] (https://arxiv.org/pdf/2511.18890 Fig. 9 / Table 8 `[read-full]`).
   - This is direct prior support for H3 and H5. The authors recommend "at least two full attention layers even in SLMs".
4. **Placement evidence conflicts across settings:**
   - Waleffe: "no significantly better configuration than to evenly distribute", but measured only on validation loss [EST].
   - Jet-Nemotron, a post-training/distillation setting starting from Qwen2.5: learned placement "significantly outperforms" uniform placement, and different layers matter for MMLU vs retrieval [EST] (https://arxiv.org/html/2508.15884 §2.2 `[read-section]`).
   - Hymba: full attention in only the first, middle and last layers (SWA elsewhere) recovers recall [EST] (https://arxiv.org/html/2411.13676 §2.3 `[read-section]`).
   - Our H4 is therefore open, and it is worth testing **from scratch** with retrieval metrics.
5. **Directly reusable for us:**
   - The Megatron allocation algorithm for the "uniform" placement baseline. I re-implemented it and checked it reproduces the Waleffe 8B pattern exactly.
   - Per-layer parameter matching.
   - The Phonebook task and the RULER subset (NIAH ×8, VT, CWE, KWE).
   - The "mixer + MLP" block structure (Nemotron-H style `M-` / `*-`), so that r changes **only** the mixer.
6. **Inference-state memory at our scale [INF].** For a 12-layer, d=512 model with 8 KV heads × 64 dims in bf16, one full-attention layer at 4K context needs 8 MiB of KV cache. That already exceeds the entire Mamba-2 state of the other 11 layers (2.75 MiB). At small scale and long context, the memory axis of the frontier is almost entirely (number of full layers × context × KV width). This sets how fine-grained r must be at the low end.

---

## 1. Nemotron-H (arXiv 2504.03624, v4, 5 Sep 2025): full read

Source: https://arxiv.org/abs/2504.03624, HTML https://arxiv.org/html/2504.03624 `[read-full]`. This is v4, the version served on 2026-11-09. It has **no appendices**: the body ends at §6, followed by contributors and references.

### 1.1 Motivation
- Reasoning models scale with inference-time tokens. Self-attention costs per-token compute and KV memory that grow linearly with length, so attention "can fundamentally limit overall model intelligence" (§1) [HYP/motivation].
- Goal: "reduce inference cost for a given accuracy level" by replacing most attention layers with Mamba-2, which has constant per-token compute and memory (Abstract).

### 1.2 Architecture: paper text vs released configs

Paper (§2.1, Table 1) [EST]:
- Layer mix: about 8% of layers are self-attention, "evenly dispersed". The rest are an even split of FFN and Mamba-2 layers.
- 8B: 4 attention layers out of 52. 56B: 10 out of 118.
- Ordering rules:
  - (a) the first layer is Mamba-2;
  - (b) the last layer is an FFN;
  - (c) every attention layer is immediately followed by an FFN, as in a standard Transformer block.
- Mamba-2: head dim 64, expand 2, conv window 4. State dim 128 (8B) / 256 (56B). 8 groups.
- Attention: GQA with 8 KV heads. 32 query heads (8B) / 64 (56B).
- FFN: squared ReLU, hidden 21504 (8B) / 32768 (56B).
- **No position embeddings**, RMSNorm, untied embeddings, no dropout, no biases. A residual connection around every layer.

Released `hybrid_override_pattern` strings (`M` = Mamba-2, `*` = attention, `-` = MLP; positions are 0-indexed). I fetched and parsed them from `https://huggingface.co/<repo>/resolve/main/config.json` on 2026-11-09 `[code-inspected]`.

| Model (HF repo) | L | #* / #M / #- (or E = MoE) | attn % of all layers | attn % of mixers (*/(*+M)) | attention positions | d_model | KV heads × head_dim | SSM state / Mamba heads × hd |
|---|---|---|---|---|---|---|---|---|
| nvidia/Nemotron-H-8B-Base-8K | 52 | 4/24/24 | 7.7% | 14.3% | 7, 18, 29, 40 | 4096 | 8×128 | 128 / 128×64 |
| nvidia/Nemotron-H-56B-Base-8K | 118 | 10/54/54 | 8.5% | 15.6% | 7, 18, 29, 40, 51, 62, 73, 84, 95, 106 | 8192 | 8×128 | 256 / 256×64 |
| nvidia/Nemotron-H-47B-Base-8K (MiniPuzzle-pruned from 56B) | 98 | 5/45/48 | 5.1% | 10.0% | 17, 38, 49, 60, 86 | 8192 | 8×128 | 256 / 256×64 |
| nvidia/Nemotron-H-4B-Base-8K (pruned from 8B, per model card) | 52 | 4/24/24 | 7.7% | 14.3% | 7, 18, 29, 40 | 3072 | 8×96 | 128 / 112×64 |
| nvidia/Nemotron-H-8B-Reasoning-128K | 52 | same as 8B | | | | | | |
| nvidia/Nemotron-H-47B-Reasoning-128K | 98 | same as 47B | | | | | | |
| nvidia/NVIDIA-Nemotron-Nano-12B-v2-Base | 62 | 6/28/28 | 9.7% | 17.6% | 7, 16, 25, 34, 43, 52 | 5120 | 8×128 | 128 / 128×**80** |
| nvidia/NVIDIA-Nemotron-Nano-9B-v2 (and -Base) | 56 | 4/27/25 | 7.1% | 12.9% | 14, 21, 30, 39 | 4480 | 8×128 | 128 / 128×80 |
| nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B (Base & BF16) | 52 | 6 */23 M/23 E | 11.5% | 20.7% | 5, 12, 19, 26, 33, 42 | 2688 | 2×128 | 128 / 64×64 |
| nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-BF16 (`layers_block_type`) | 88 | 8 */40 M/40 E | 9.1% | 16.7% | 7, 16, 25, 36, 47, 58, 69, 78 | 4096 | 2×? | 128 / 128×? |

Exact 8B string: `M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M-`.

Exact 56B string: `M-M-M-M*-M-M-M-M-M*-` followed by the `M-M-M-M-M*-` motif ×8, then `M-M-M-M-M-`.

**Paper/config discrepancies I found [INF from code inspection]:**
1. **47B layer counts.** §3.5 of the paper says the 47B keeps "5 [attention], 44 Mamba-2 … 49 FFN". The config has **45 Mamba, 48 MLP**. The total of 98 matches, but the split is off by one.
2. **Nano 2 Mamba head dim.** §2.1 of the Nano 2 paper says "head dimension of 64". The released 12B and 9B configs have `mamba_head_dim: 80` with 128 heads, which is consistent with expand 2 (128×80 = 10240 = 2×5120).
3. **Nemotron-H 8B pattern is hand-placed.** It is **not** what Megatron's ratio allocator produces. `pattern_from_ratios(52, 4/52, 24/52)` gives `…M-*M-…`, with attention *after* an MLP. The released pattern always has `*-`, attention *before* an MLP, which satisfies rule (c) in §1.2. By contrast, the allocator **does** reproduce the Waleffe 8B pattern exactly (§5).
4. **"8%" is a fraction of all layers, MLPs included.** Measured over token mixers only (attention / (attention + Mamba)), the Nemotron-H family sits at **13–21%**. This matters for how our r is defined (§7).
5. **Pruning removed the outer attention layers.** Nano 2 9B (pruned from 12B) has no attention in its first 14 or last 16 layers. The 12B parent has attention at layers 7 and 52. So pruning appears to have removed the outermost attention layers and kept the 4 middle ones. This is an inference from the patterns; the paper does not list which layers were removed.

### 1.3 Role of attention and "why ~8%"
- The paper gives no first-principles rationale and no new ablation. It cites Waleffe et al. (§2.1). The rationale is effectively Waleffe's validation-loss sweep (§2.3 below) [EST that it is inherited; INF that no other justification is given].
- **Layer importance on the 56B (Fig. 9, §3.2)** [EST]. Importance is the MSE at the LM head when one layer is removed, averaged over 128 samples.
  - The most important layers sit at the beginning and end of the network.
  - "Some self-attention layers are ranked among the least important, particularly the 84th (7th self-attention layer). However, layers 40, 51, 62 and 73 seem to be more important compared to the other layers in their immediate vicinity."
  - This is weak evidence that attention layers are **not equally useful**, which bears on H4. It is measured on short-context output MSE, not on retrieval.
- **MiniPuzzle compression, 56B → 47B (§3)** [EST].
  - The search kept **5 of the 10** attention layers, 44 (config: 45) of 54 Mamba layers and 49 (config: 48) of 54 FFNs. FFN width went from 32768 to 30720.
  - After 63B tokens of distillation, the average score across the selection benchmarks was 70.6 (56B) vs 69.6 (47B, Table 6).
  - Those benchmarks are all short-context (MMLU, commonsense, GSM8K, code). **The 47B base was not evaluated on long context in this paper.** So "half the attention layers is near-lossless" holds only for short-context tasks [INF].
  - Mamba head and head-dim pruning "caused severe accuracy degradation", so attention and FFN layers were pruned more readily than Mamba internals [EST].

### 1.4 Training recipe (§2.2–2.4) [EST]
- **Data:** 15T tokens (8B), 20T (56B); seq len 8192; global batch 768 (6,291,456 tokens); no batch-size ramp.
- **Learning rate:** peak 8e-4 (8B) / 4e-4 (56B); warmup 8.3B / 25B tokens; cosine to 1% of peak.
- **Optimizer:** weight decay 0.1, Adam β = (0.9, 0.95).
- **Data mixture:** a 4-phase blend; phase switches at 60% and 80% of training, and the last 380B tokens form phase 4. A phased blend beat random ordering by 3.4% (8B, 1T tokens).
- **Precision:** the 56B trained in FP8, using per-tensor current scaling (E4M3 for forward tensors, E5M2 for gradients), with the first and last 4 layers kept in BF16. "Log likelihood … was not a reliable predictor" of downstream accuracy in their FP8 vs BF16 comparisons. A useful caution for us: do not use validation loss as a proxy for retrieval.
- **Framework:** Megatron-LM + Transformer Engine; TP=8, sequence parallel, DP=768.

### 1.5 Long-context recipe and evaluation [EST]
- **Base models are 8K only** (`max_position_embeddings` 8192 in the configs). The paper reports **no** long-context evaluation of the base models.
- **Long context arrives in post-training (§5.1–5.2.2):**
  - SFT on conversations up to 512K tokens, built by concatenation. Inserted turns include long-range references, multi-turn references, document-aggregation QA (documents up to 32K) and common-word retrieval with distractors (30–80K tokens, aimed at RULER CWE).
  - "Hybrid models can learn to operate on higher context length within just a few hundred steps." The model reaches "83% RULER score on 128k" in non-reasoning mode.
  - The text does not say clearly whether this is the 8B or the 47B [EST, with that ambiguity noted].
- **Pilot hybrid vs Transformer alignment (Table 9/10):** Nemotron-T-8B-Exp vs Nemotron-H-8B-Exp, same pretraining data. Post-training gives similar results (e.g. MT-Bench 7.9 vs 8.0, IFEval prompt 74.3 vs 74.7). No long-context numbers are reported for this pair.

### 1.6 Memory and throughput claims, and how they were measured [EST]
- Setup: 65,536 input tokens and 1,024 output tokens on H100, batch size as large as possible, output tokens/s/GPU.
- **Inference stacks differ:** "an initial Megatron-LM implementation for Nemotron-H inference and vLLM v0.7.3 for baselines". This confounds the comparison [INF].
- Results: 56B is 2.4× faster than Qwen-2.5-72B / Llama-3.1-70B (2.9× for the 47B), and 19.6× faster than Llama-3.1-405B. 8B is 1.8× faster than Qwen-2.5-7B and 3× faster than Llama-3.1-8B.
- Attributed to (a) constant compute per token in Mamba layers and (b) a smaller memory footprint that allows larger batches.
- **No per-component memory breakdown** (KV vs SSM state) is reported. My own arithmetic from the configs is in §6.

### 1.7 Base-model accuracy [EST]
- **8B (Table 5)**, against the same-data Transformer Nemotron-T-8B (32 layers, RoPE):
  - Nemotron-H is higher on 7/15 tasks and within 1 point on 4 more.
  - Examples: MMLU 72.8 vs 73.2; GSM8K 87.1 vs 89.0; HellaSwag 83.2 vs 82.5.
- **Confounds** in this "apples-to-apples" comparison [INF]: depth (52 vs 32 layers), position encoding (none vs RoPE), and the parameter split between FFN and mixers. It is not a single-variable comparison.

### 1.8 Limitations relevant to us [INF]
- No ratio or placement ablation, no long-context evaluation of the base models, no memory breakdown.
- Throughput was measured with different serving stacks.
- Scale is 8B–56B with 15–20T tokens, about 5 orders of magnitude more compute than our budget.

---

## 2. Waleffe et al. 2024, "An Empirical Study of Mamba-based Language Models" (arXiv 2406.07887, v1, 12 Jun 2024): full read

Source: https://arxiv.org/abs/2406.07887. I read the HTML in full, including Appendix A, and rendered PDF pages 9–11 and 14 to read Figs. 4, 5 and 7, whose values exist only as images (https://arxiv.org/pdf/2406.07887) `[read-full]`.

Code snapshot: https://github.com/NVIDIA/Megatron-LM/tree/ssm/examples/mamba (cited by the paper; I did not inspect that branch).

Checkpoints: `nvidia/mamba2-hybrid-8b-3t-4k`, `nvidia/mamba2-hybrid-8b-3t-128k`, `nvidia/mamba2-8b-3t-4k` on HF. These are Megatron `.pt` files with no HF `config.json`; I confirmed the file listings through the HF API `[code-inspected]`.

### 2.1 Research question and controls [EST]
- Pure Mamba, Mamba-2, the Mamba-2-Hybrid and a Transformer, all ~8B parameters, on the **same data** (1.1T and 3.5T tokens; 70% English / 15% non-English / 15% code; 256K SentencePiece vocabulary) and the **same hyperparameters**.
- 1.1T run: batch 256, LR 1e-4 → 1e-5. 3.5T run: batch 1024, LR 3e-4 → 3e-5.
- Common settings: warmup 122K samples, cosine, weight decay 0.1, β = (0.9, 0.95), BF16, seq len 4096.
- **Not held fixed:**
  - The Transformer uses RoPE, SwiGLU and LayerNorm, with 32 attention+MLP blocks.
  - Pure Mamba/Mamba-2 have 56 mixer layers and **no MLPs**.
  - The hybrid has 24 Mamba-2 + 4 attention + 28 MLP layers, GELU, RMSNorm and no PE.
  - **So pure-SSM vs hybrid changes the MLP content as well as attention**, a confound for anyone reading it as "adding 4 attention layers" [INF].

### 2.2 Pure SSM vs Transformer [EST]
- **Standard zero-shot tasks:** SSMs match or exceed. At 3.5T, average without MMLU: Mamba-2 70.63 vs Transformer 68.14.
- **MMLU 5-shot:** at 1.1T, Mamba-2 is 17 points lower. At 3.5T the gap narrows to 1.37 points (48.7 vs 50.07).
- **Cloze-format MMLU:** SSMs match the Transformer even at 1.1T. The authors' reading is that the knowledge is present but routing the answer to a single letter token needs attention [HYP].
- **Phonebook** (Jelassi et al. 2024; 2-shot; name→number lookup): pure SSMs fail beyond about 500 tokens (1.1T) and about 1000 tokens (3.5T). The Transformer is near 100% up to 4K.
  - SSMs show "fuzzy memory": the predicted numbers share digits with the true one.
  - The "Reversed" formulation, where the query is given first, helps but still degrades below 4K.

### 2.3 The ratio and placement ablations (§4.1, Appendix A): the key evidence
**Design:** all layer types are parameter-matched, by adjusting attention head count at constant head size and the MLP expansion factor, "such that self-attention and MLP layers have approximately the same number of parameters as Mamba layers". The layer ratio then equals the parameter ratio [EST].

**Attention fraction (Fig. 4):**
- Models: 130M parameters, 24 layers. Attention is placed with Algorithm 1 (even spacing).
- Metric: **validation loss only**.
- Minimum loss at "roughly 8%", i.e. 2/24 layers. The authors say 840M experiments "confirm" this; no 840M figure is shown [EST, 840M claim not verifiable].

Digitized values (my reading of PDF p.9 at 300 dpi; x positions match k/24):

| attention layers / 24 | 0 | 1 | 2 | 3 | 4 | 5 | 7 | 12 | 24 |
|---|---|---|---|---|---|---|---|---|---|
| % | 0 | 4.2 | 8.3 | 12.5 | 16.7 | 20.8 | 29.2 | 50 | 100 |
| val loss (≈) | 2.615 | 2.562 | 2.554 | 2.555 | 2.555 | 2.560 | 2.568 | 2.593 | 2.631 |

What the curve shows [INF]:
- (i) 1 attention layer captures (2.615−2.562)/(2.615−2.554) ≈ 87% of the achievable gain.
- (ii) The curve is flat from 8% to 17%.
- (iii) Loss rises steadily from about 20% to 100%, and the all-attention point is **worse than all-Mamba**.

**Unreported details limit interpretation:**
- The MLP ratio used in the attention sweep is never stated. If it was 0, the 100% point is a 24-layer attention-only network with **no MLPs**, which would inflate its loss for reasons unrelated to the attention/linear trade-off.
- Token budget, sequence length, and whether attention used RoPE in that sweep are not stated.
- No retrieval or long-context metric was measured.
- One seed; no error bars.

**MLP fraction:** at 8% attention, varying MLP from 5% to 50% shows "30%–50% of the layers can be MLPs without increasing model loss". 50% MLP trains 20% faster than 5% [EST].

**Position embeddings (Table 5):**
- Setup: 840M, 8% attention, 1.1T tokens at 4K, then 16B tokens of continued pretraining (CPT) at 16K.
- At 4K: RoPE (52.39 avg) ≈ none (52.19).
- After the 16K extension: none 53.43 > RoPE base 10K (52.61) > RoPE base 500K (51.52).
- Hence NoPE for hybrids [EST, on short-context commonsense tasks only; ~1-point differences, single runs].

**Ordering/placement:**
- "Our experiments found no significantly better configuration than to evenly distribute self-attention and MLP layers … We did not find it necessary to construct hybrid model architectures using a repeated block pattern." Measured on **validation loss**, with no numbers shown.
- A Mamba layer must come first so that it can supply positional information without PE (following Park et al. 2024) [EST qualitative].

**GQA vs MHA in hybrids:** validation perplexity +≈0.04% only [EST].

**Related-work claim (§5):** "the relative proportions of each type of hybrid component appears to be the key factor that determines the quality of the model" [HYP, generalised from the loss-only ablations].

### 2.4 Mamba-2-Hybrid 8B and its long-context extensions [EST]
**Architecture:**
- 56 layers: 24 Mamba-2 (42.9%), 4 attention (7.1%), 28 MLP (50%).
- d=4096, GQA 32 heads / 8 groups, 128 KV channels, MLP 4×, GELU, no PE.
- Pattern: `M+M+M++M+M*+M+M+M+M++M*+M+M+M+M+M*++M+M+M+M+M*+M++M+M+M+` (`+` = MLP).
- Training MFU 29.9% vs 30.7% for the Transformer (1024 H100s).

**Short-context tasks (3.5T):** the hybrid beats the Transformer on all 12 tasks, average 55.82 vs 53.17. MMLU 5-shot 53.60 vs 50.07. Early in training the hybrid trails the Transformer on MMLU; it overtakes at about 1.5T tokens (Fig. 6).

**Speed:** "nearly 8× faster" token generation at long context. This is **predicted** (Fig. 5, batch 32), not measured. I read about 7.3× at 128K, about 5.8× at 32K and about 4.6× at 16K off the curve [EST that it is predicted; values INF from the figure].

**Long-context CPT:**
- 4K → 16K/32K with 50B tokens of the same data, packed (no long-document selection).
- LR: linear warmup over 1.7B tokens, then cosine from 3e-5 to 3e-6.
- Full global attention in the 4 attention layers. The Transformer uses dynamic NTK RoPE scaling.

**Natural long-context tasks (Table 9, 9 tasks: NarrativeQA, Qasper, QuALITY, LongBench MultiFieldQA, HotpotQA, 2WikiMQA, Musique, TREC, TriviaQA):**

| model | 4K | 16K | 32K |
|---|---|---|---|
| Transformer | 40.66 | 45.24 | 45.57 |
| Hybrid | 40.83 | 44.60 | 44.16 |
| Pure Mamba-2 (4K) | 34.31 | | |

- The Transformer leads on **multi-document QA**: 16K HotpotQA 48.61 vs 42.17; 32K 2WikiMQA 35.52 vs 29.06.
- Authors' hypothesis: SSM states get "confused by documents irrelevant to the question". Packing unrelated documents during CPT may be to blame. Hybrids are also more prompt-sensitive: Musique ranged 10.63–16.16 for the hybrid vs 15.25–17.68 for the Transformer [HYP].

**RULER (Table 10; 13 tasks, 400 samples each, evaluated at the model's trained length):**

| | 4K | 16K | 32K |
|---|---|---|---|
| Mamba-2 (4K only) | 52.14 | – | – |
| Transformer | 77.62 | 62.33 | 56.49 |
| Hybrid | 81.99 | 74.19 | 69.93 |

Per-task highlights:
- **VT (multi-hop tracing):** Transformer 7.25 (16K) and 0.35 (32K) vs hybrid 83.2 and 79.9. The Transformer copies the one-shot example's answer.
- **KWE:** the Transformer is better at every length (32K: 53.5 vs 36.5).
- **CWE:** both collapse at 16K/32K (3–13).
- **Multi-key NIAH at 4K, pure Mamba-2:** MK-NIAH-2 13.75 and MK-NIAH-3 5.5. The hybrid gets 95.5 and 96.
- **RULER-HotpotQA:** the hybrid is *better*, unlike LongBench HotpotQA. The two benchmarks build distractors differently.

**Phonebook (Fig. 7):**
- Hybrid: 100% to 5.5K (trained at 4K), and perfect beyond the trained length for the 16K and 32K variants.
- Transformer: degrades approaching its trained length.
- The 128K hybrid is perfect beyond 150K tokens (Fig. 8).
- Reading the figure [INF]: in Fig. 7b the 16K hybrid stays near 1.0 to about 22K and falls to about 0.2 by about 30K. In Fig. 7c the 32K hybrid stays near 1.0 to about 45K.

**This all confounds PE with architecture [INF].** The Transformer uses RoPE plus NTK scaling; the hybrid uses NoPE. Length *extrapolation* differences may come from PE as much as from the ratio. Our sweep must hold PE fixed across r.

### 2.5 Appendix A: hybrid layer allocation algorithm [EST, code-verified in §5]
1. Place attention layers so that the runs of Mamba layers between them are as equal as possible, starting and ending with Mamba.
2. Distribute MLPs evenly without replacing attention, biased so the sequence starts with Mamba and ends with an MLP.

Table 11 examples:
- 24 layers / 8% attention: `MMMMMMM*MMMMMMMM*MMMMMMM`
- 24 / 17%: `MMMM*MMMM*MMMM*MMMM*MMMM`
- 24 / 8% / 50% MLP: `M+M+M++*M+M+M+M+*M++M+M+`

---

## 3. Follow-ups: identifying the "Nemotron Edge" paper

I searched for "Nemotron Edge" / "Nemotron-Edge" on 2026-11-09 (WebSearch, extended). **There is no NVIDIA paper or model with that exact name.** Candidates:

1. **Nemotron-Flash: Towards Latency-Optimal Hybrid Small Language Models** (Fu, Dong, et al., arXiv 2511.18890, Nov 2025). This is the best match for a "small/edge hybrid" paper: batch-size-1 latency is a central concern, it uses CUDA Graph, and Appendix Table 14 is captioned "our target is edge applications". 0.96B/2.7B models. **I treat this as the paper the user meant**, and covered it in full (§3.2).
2. **Nemotron 3.5 Bolt** (October 2026): NVIDIA's "edge" models. Bolt-E4B is a Gemma-4-E4B post-train (`Gemma4ForConditionalGeneration`, 42 layers, 7 full + 35 sliding attention, not a Mamba hybrid). Bolt-74B-A9B is a Nemotron-H-style Mamba/MoE/attention hybrid (88 layers, 8 attention), per the configs `[code-inspected]`. I found **no technical paper** for Bolt. The NVIDIA blog (https://blogs.nvidia.com/blog/nemotron-3-5-super-vl/) appeared only as a search result; I did not read it.
3. **Nemotron Nano 2** (arXiv 2508.14444) and **Nemotron 3 Nano** (arXiv 2512.20848) are "Nano" tier. The NVIDIA-NeMo/Nemotron repo describes the Nano tier as being for edge/PC (seen only as a search snippet), but these are 9B and 30B-A3B models, not edge-sized.

**Not accessed** (named so nobody assumes they were covered):
- Nemotron 3 white paper, arXiv 2512.20856.
- Nemotron 3 Ultra, arXiv 2606.15007.
- Group-aware SSM pruning, arXiv 2504.11409 (cited by the Nemotron-H-4B model card).

### 3.1 NVIDIA Nemotron Nano 2 (arXiv 2508.14444, v4)
`[read-section]`: Abstract, §1, §2.1, §2.5–2.7, §4 (pruning/NAS/distillation), §5. Data (§2.2–2.4) and alignment (§3) only skimmed. https://arxiv.org/html/2508.14444

**Architecture [EST]:** 12B base: 62 layers = 6 attention + 28 Mamba-2 + 28 FFN; d=5120; FFN 20480; GQA 40 query / 8 KV heads; "roughly 8%" attention "evenly dispersed"; NoPE.

**Recipe [EST]:**
- Pretraining: 20T tokens, seq 8192, batch 768. Warmup-stable-decay (WSD) schedule, stable LR 4.5e-4 → 4.5e-6, decayed over the final 3.6T tokens. Weight decay 0.1, β = (0.9, 0.95).
- Long-context phase: CPT at **512K** sequence length, constant LR 4.5e-6, 18.9B tokens. Global batch 12 (~6M tokens). TP 8 + context parallel 16.
- Blend: Phase-3 blend scaled to 80%, plus 20% synthetic long-document QA (Qwen-2.5-72B generating QA from 10% of 1K-token chunks of documents over 32K tokens).

**Long-context ablation on Nemotron-H 8B (Table 4), RULER-128K by CPT training length and synthetic data [EST]:**

| CPT length | 128K | 256K | 256K | 512K |
|---|---|---|---|---|
| synthetic data | yes | no | yes | yes |
| RULER-128K | 73.68 | 70.19 | 79.04 | 81.04 |

Authors' reasoning: longer sequences mean fewer coherent documents are cut by concat-and-chunk [HYP].

**Base-model RULER-128K [EST]:** 12B 84.74; 9B (pruned) 82.22; Gemma3-12B 80.70.

**Attention count fixed during pruning [EST]:**
- Depth-pruned candidates (52/54/56 layers) "keep the number of attention layers fixed at 4 … prior work has indicated that an attention-to-total-layers ratio between 7-8% is reasonable (NVIDIA, 2025)".
- Depth matters a lot: 6B-token-KD accuracy is 44.92 / 47.35 / 51.48 for 52 / 54 / 56 layers.
- So the 12B → 9B step went from **6 to 4 attention layers**, with a 2.5-point RULER-128K cost. That cost is confounded with width pruning and distillation [INF].
- Mamba-head pruning "caused severe accuracy degradation" at these compression ratios; FFN/embedding pruning sufficed.

**Throughput [EST]:** up to 6.3× Qwen3-8B at 8K input / 16K output, measured with vLLM on one A10G in bf16 at the maximum batch that fits. Memory budget 19.66 GiB at 128K context and batch 1.

### 3.2 Nemotron-Flash (arXiv 2511.18890, v1): read in full including Appendices A–G
https://arxiv.org/html/2511.18890; Fig. 9 read from the PDF (https://arxiv.org/pdf/2511.18890 p.10) `[read-full]`.

**Question:** what determines the *real-device latency* of small LMs? Depth–width ratio and operator choice.

**Depth–width (§3.1) [EST]:**
- Llama models at 6–30 blocks × several widths, 100B SmolLM-corpus tokens, AdamW, LR 5e-4.
- Deep-thin is parameter-efficient but not latency-optimal. The optimal depth grows with the latency budget (e.g. depth 12 is best at a 3 s, 1K-token, A100, batch-1 budget).
- A depth/width scaling law L(D,W,N) = L0 + a·D^−α + b·W^−β + c·N^−γ extrapolates to within 5.3% PPL.

**Operator study (§3.2, Table 1, Appendix A):**
- Setup: 500M models, 24 operators each (a mixer and an FFN each count as one), 100B SmolLM tokens, LR 5e-4. Candidates: Mamba, Mamba2, GLA, DeltaNet, Gated DeltaNet, RWKV7, SWA-512.
- Pure models [EST], as Wiki PPL / CR accuracy:

| operator | Wiki PPL | CR acc |
|---|---|---|
| Attention | 24.44 | 48.02 |
| Mamba2 | 24.36 | 47.72 |
| DeltaNet | 23.87 | 47.83 |
| Gated DeltaNet | 23.80 | 47.96 |
| GLA | 25.16 | 46.82 |
| RWKV | 25.47 | 46.46 |
| Mamba | 24.55 | 47.32 |

- Hybrid pairs [EST]: DeltaNet+Mamba2 23.37/48.03; DeltaNet+Attn 24.32/47.48; GatedDeltaNet+Attn 24.31/47.80; GLA+Attn 24.60/47.63. "Improvements from pairing with attention are less stable."
- **Caveat [INF]:** the attention in these hybrid pairs is *full* attention under the "Attn" label, but the search itself used SWA-512 for attention. All of these metrics are short-context.

**Evolutionary search (§3.2, Appendix C) [EST]:**
- Aging evolution with short-train (10B tokens) PPL as the proxy. Spearman correlation with full-train PPL: 88.8%. About 2 h on 32 A100s per candidate.
- Search space: 3 stages, ≤2 operator types per stage at ratios 0:1, 1:1, 1:2 or 1:3; 0–2 FFNs per mixer; ≤30 operators.
- Latency-optimal result: `[d,f,m2,f,a,f,m2,f, d,f,m2,f,a,f,m2,f, d,f,m2,f, d,f,m2,f]`, i.e. 2 attention among 12 mixers.
- The parameter-optimal search puts **more attention** in the model and makes it deeper. Its Table 2 string has 7 `a` among 15 mixers (4 m2, 4 d), 28 operators; in the search these are SWA-512 layers.
- The main limitation is acknowledged in Appendix G: "we primarily use language modeling as the search proxy, which may not fully capture … long-context understanding".

**Final models (Appendix Table 11; configs fetched) [EST + code-inspected]:**

| model | layer sequence | full attn among mixers | d | FFN | attn / KV heads |
|---|---|---|---|---|---|
| Nemotron-Flash-1B | `D F M2 F A F M2 F D F M2 F A F M2 F D F M2 F D F M2 F` | 2/12 (16.7%), positions 4, 12 | 2048 | 6144 | 16/4 |
| Nemotron-Flash-3B | 36 operators | 3/18 (16.7%), positions 4, 12, 20 | 3072 | 9216 | 24/6 |

- 1B mixers: 4 DeltaNet, 6 Mamba2, 2 attention.
- 3B mixers: 6 DeltaNet, 9 Mamba2, 3 attention.
- 256 learnable meta tokens (from Hymba). Weight normalization, a simplified nGPT that projects rows/columns to the unit norm after each step, worth +1.20 CR accuracy and −0.66 PPL on average.
- Training: 4.5T tokens, LR 1e-3, no weight decay, batch 2M tokens, context 4096, with the last 25B tokens at 29K. Mistral-NeMo-Minitron 131K-vocabulary tokenizer.

**Attention-configuration ablation (§4.3, Table 8, Fig. 9): the most directly relevant data in this report [EST].**
Starting from the pretrained 3B (3 FA layers), they ran 25B tokens of CPT at 29K context in three configurations. Fig. 9 gives RULER NIAH scores by context length:

| config | throughput (32K, max batch) | Recall (FDA/SWDE/SQuAD) | NIAH s1 / s2 / s3 @16K | @24K | @32K |
|---|---|---|---|---|---|
| 3 FA | 2939 tok/s | 73.25 | 1.00 / 1.00 / 0.89 | 1.00 / 0.99 / 0.93 | 0.96 / 0.91 / 0.69 |
| 2 FA + 1 SWA (8K window) | 3737 | 73.37 | 1.00 / 1.00 / 0.98 | 1.00 / 1.00 / 1.00 | 0.94 / 0.85 / 0.74 |
| 1 FA + 2 SWA | 4657 | 73.77 | 0.99 / 0.96 / 0.92 | 0.98 / 0.73 / 0.59 | **0.78 / 0.60 / 0.32** |

- At 4K and 8K, all three configurations score ≥0.94 on every NIAH task.
- **Interpretation [INF]:** short-context recall benchmarks are blind to the number of full-attention layers. The NIAH penalty from dropping to one FA layer appears only beyond 16K and grows with length. This is evidence for H3/H5 at ~3B scale, though only 3 configurations, one seed, CPT not from scratch, and SWA (not linear attention) as the replacement.
- Authors' recommendation: "maintaining at least two full attention layers even in SLMs".

**Latency/throughput measurement [EST]:**
- Latency: H100, batch 1, decoding 8K tokens with CUDA Graph.
- Throughput: 32K input at the maximum batch without OOM.
- Kernels: TensorRT-LLM AutoDeploy for attention, official Mamba2 kernels, FlashLinearAttention for DeltaNet.
- Appendix Table 14: for a 550M Llama at batch 1, PyTorch eager is 5–7× slower than optimised flows (e.g. 113.9 s vs 17.5 s for 8K tokens). **A warning for our own decode benchmarks** [INF].

### 3.3 Nemotron 3 Nano (arXiv 2512.20848, v1)
`[read-section]`: Abstract, §1, §2.1, §2.4–2.6. https://arxiv.org/html/2512.20848

**Architecture [EST]:**
- 52 layers: 23 Mamba-2 + 23 MoE + 6 GQA attention (2 KV heads, head dim 128). d=2688.
- MoE: 128 routed experts, top-6, 2 shared experts, squared ReLU, sigmoid router.
- 31.6B total / 3.2B active parameters; NoPE.
- Mixer fraction 20.7%, positions 5, 12, 19, 26, 33, 42 per the config. Note the attention fraction among mixers is **higher** than in Nemotron-H.

**Training [EST]:**
- 25T tokens, WSD with peak 1e-3 (stable for 20T), decayed to 1e-5 over 5T.
- Seq 8192, batch 3072 (~25M tokens).
- Long-context phase: 121B tokens, constant LR 1e-5. Blend: 20% doc-QA, 1% synthetic retrieval, 79% Phase-2 data. A mix of 512K and 4K sequences, because 512K-only CPT hurt short-context scores.

**Results [EST]:** RULER 64K / 128K / 256K = 87.50 / 82.92 / 75.44, vs Qwen3-30B-A3B-Base at 63.55 / 60.69 (–). Throughput 3.3× Qwen3-30B-A3B-Thinking at 8K input / 16K output on one H200.

**Relevance:** low for ratio questions. There is no ratio ablation, and MoE confounds everything. It does show the family's attention fraction drifting **up** over time (7.7% of layers in Nemotron-H-8B → 11.5% in Nemotron 3 Nano) [INF].

### 3.4 Jet-Nemotron (arXiv 2508.15884, v3)
`[read-section]`: Abstract, §1, §2.1–2.2, the model-family paragraph, the throughput paragraph, Appendix A.1 mentions. https://arxiv.org/html/2508.15884

**Method [EST]:** "PostNAS" starts from a pretrained full-attention model (Qwen2.5-1.5B/3B), freezes the MLPs, and trains a once-for-all super network with alternative linear-attention paths using feature distillation. Beam search then finds where full attention should go, using a task-dependent objective.

**Key findings (Fig. 5) [EST]:**
1. Not all attention layers matter. "For MMLU, only two layers exhibit critical importance, while for retrieval tasks, just two to three layers".
2. "Layers that are critical for MMLU accuracy are not necessarily important for retrieval tasks."
3. Math needs roughly the union of the MMLU and retrieval layers.
4. Learned placement "significantly outperforms" uniform placement.

**Final models [EST]:**
- 2B: full attention at layers 15 and 20, SWA at 21 and 22 (window 1152), JetBlock elsewhere (28 layers in total, inherited from Qwen2.5-1.5B).
- 4B: FA at 18, 21, 33; SWA at 7 layers (window 2048).
- Placement was guided by **retrieval for FA** and **MMLU for SWA**.
- 53.6× decode speedup vs Qwen3-1.7B at 256K. The theoretical maximum is 56×: (28 FA layers × 8 KV groups) / (2 × 2).

**Relevance:** high for H3/H4. But the setting is **retrofitting a pretrained Transformer**, where specific layers have already specialised. That is a different regime from from-scratch training, and Waleffe found no placement effect from scratch (on loss) [INF]. Another agent may own Jet-Nemotron in detail; a clone exists at `third_party/Jet-Nemotron`, which I did not inspect.

### 3.5 Hymba (arXiv 2411.13676, v1)
`[read-section]`: §2.3, Appendix B / Table 10, the training-settings lines. https://arxiv.org/html/2411.13676

**Design [EST]:** *parallel* hybrid heads, with attention and Mamba heads in the same layer, mean-fused. Ablations at 300M / 100B tokens.

**Results [EST]:**
- All-SWA (window 256) drops recall by >20%: Table 10 row 7 has recall avg 29.78 vs 49.90 for full attention (row 5).
- Restoring full attention in **3 layers (first, middle, last)** recovers it: 48.79 recall, throughput 2399.7 vs 876.7 tok/s, cache 41.19 MB vs 148.24 MB at 8K.
- Applying the same KV compression to a pure Transformer drops recall to 28.18 (row 10 vs 39.98 for row 6).
- Attention:Mamba parameter ratio: gains "saturate" at 1:2.12.

**Relevance:** medium. It is a parallel (head-level) hybrid with SWA rather than linear attention elsewhere, but its first/middle/last placement is a natural candidate for our placement arm.

---

## 4. Where each source sits relative to our hypotheses

| Hyp. | Supporting evidence (family) | Contradicting or complicating evidence | Status for us |
|---|---|---|---|
| H1 partial sufficiency | Waleffe Fig. 4: 1/24 attention gets ~87% of the loss gain [INF from EST]. 8B hybrid with 4/56 attention ≥ Transformer on RULER (4–32K) and Phonebook [EST]. Nemotron-Flash 2–3 FA layers are competitive on recall [EST]. | Loss ≠ retrieval. Transformer wins LongBench multi-doc QA and KWE [EST]. PE and MLP confounds. | Plausible; **untested at 20–60M with long-context metrics** |
| H2 diminishing returns | Waleffe Fig. 4: flat from 8–17%, then *rising* loss [EST] | Loss-based only; the MLP fraction at the 100% point is unknown | Plausible on loss; unknown on retrieval |
| H3 task dependence | Waleffe: hybrid ≫ Transformer on VT and NIAH, < on KWE and multi-doc QA [EST]. Jet-Nemotron: different layers serve MMLU vs retrieval [EST]. Nemotron-Flash: short recall flat, long NIAH drops [EST]. | Comparisons are across different models, not a ratio sweep | Strong prior that it holds; design the eval to show it |
| H4 placement | Jet-Nemotron (retrofit) learned > uniform. Hymba first/middle/last. Nemotron-H 56B: attention layer importance varies. Nano 2 pruning kept the middle attention layers [INF]. | Waleffe from scratch: even spacing is as good as anything tried (on loss) | **Genuinely open from scratch**: worth a placement arm |
| H5 benefit grows with length | Nemotron-Flash Fig. 9: the 1FA–3FA gap grows with context (1.00→0.32 vs 0.69) [EST]. Waleffe: Transformer RULER drops faster with length than the hybrid [EST] (PE-confounded). | Hybrid extrapolation may be a NoPE effect, not a ratio effect | Plausible; requires holding PE fixed |
| H6 frontier region | Waleffe U-curve: hybrid beats both endpoints on loss [EST]. 8B hybrid beats the Transformer on accuracy and speed [EST]. | Throughput measured on different stacks (Nemotron-H); speed predicted, not measured (Waleffe) | Plausible; we must measure both axes ourselves |

---

## 5. Megatron-LM hybrid code inspection `[code-inspected]`

**Repository:** https://github.com/NVIDIA/Megatron-LM, shallow sparse clone at `third_party/Megatron-LM`, commit `928ce7ce256b734f00f1c81ddd3a8958d6b61ebf` (2026-11-08).

**Files:**
- `megatron/core/ssm/mamba_hybrid_layer_allocation.py` is now a 5-line re-export of `megatron/core/models/hybrid/hybrid_layer_allocation.py`.
- `pattern_from_ratios(num_layers, attention_ratio, mlp_ratio)` (lines 220–271) implements Waleffe Algorithm 1 exactly:
  - `attention_count = round(N·a)`;
  - Mamba runs of length `(N−A)/(A+1)`, placed with a running-remainder `x < 0.5` test;
  - MLPs placed by the same rule among the remaining Mamba slots.
  - Its docstring calls the ratio arguments "deprecated"; the preferred interface is now an explicit `--hybrid-layer-pattern`.
- Symbols (`megatron/core/models/hybrid/layers/utils.py`): `M` Mamba, `G` GatedDeltaNet, `K` KDA, `*` attention, `-` MLP, `E` MoE, `W` sliding-window, `|` pipeline split, plus several MLA/DSv4 variants.
  - GDN/KDA layers (`megatron/core/ssm/gated_delta_net/{gdn,kda}.py`) exist in Megatron now, so a linear-attention hybrid sweep with GatedDeltaNet is expressible there too. I did not test it.
- `examples/mamba/train.sh` gives the reference small-scale config:
  - "800M": pattern `M-M-M--M-*M-M-M-M--*M-M-M-M-*M--M-M-M-*M-M--M-M-` (48 layers, 4 attention = 8.3%), d=1024, 16 heads.
  - Optimizer: LR 2.5e-4 → 2.5e-5 cosine, warmup 50K samples, 300B tokens at seq 4096.
  - Other settings: weight decay 0.1, clip 1.0, β = (0.9, 0.95), `--position-embedding-type none`, untied embeddings, RMSNorm, no bias, GQA with 8 groups.
  - The README notes `main` is "no longer compatible with the Mamba2-* checkpoints".

**My check:** I re-implemented the algorithm in plain Python, outside the repo, in the session scratchpad, and ran it [INF].
- It reproduces the Waleffe 8B string exactly for (56, 0.08, 0.5).
- It does **not** reproduce Nemotron-H-8B for (52, 4/52, 24/52). It yields `…M-*M-…` where the release has `…M*-…`.
- Uniform patterns it generates for small depths, usable as our uniform-placement baselines:
  - 12 layers: 1 → `MMMMMM*MMMMM`; 2 → `MMM*MMMM*MMM`; 3 → `MM*MMM*MM*MM`; 4 → `MM*M*MM*M*MM`; 6 → `M*M*M**M*M*M`.
  - 16 layers: 1 → `MMMMMMMM*MMMMMMM`; 2 → `MMMMM*MMMM*MMMMM`; 4 → `MM*MMM*MM*MMM*MM`.
  - 24 layers: 1 → `MMMMMMMMMMMM*MMMMMMMMMMM`; 2 → `MMMMMMM*MMMMMMMM*MMMMMMM`; 3 → `MMMMM*MMMMMM*MMMMM*MMMMM`; 6 → `MMM*MM*MMM*MM*MMM*MM*MMM`.
  - **Note:** at 50% the algorithm puts two attention layers adjacent (`**`). A hand-specified alternating pattern is cleaner for the r = 1/2 point [INF].

---

## 6. Inference-state memory: my arithmetic from the configs [INF]

**Formulas** (bf16 = 2 bytes; per sequence, batch 1):
- **KV cache** = 2 · n_kv · head_dim · 2 bytes · n_attn · context.
- **Mamba-2 state** = n_heads · head_dim · d_state · 2 bytes · n_mamba, **plus** a conv state of (d_inner + 2·n_groups·d_state) · (k−1) · 2 bytes · n_mamba.
- The SSM state is often kept in fp32 in practice, which doubles that term.

| model | n_attn | constant SSM+conv state | KV @8K | KV @32K | KV @128K |
|---|---|---|---|---|---|
| Nemotron-H-8B | 4 | 49.4 MiB | 128 MiB | 512 MiB | 2048 MiB |
| Nemotron-T-8B (32 attention layers, 8×128 KV) | 32 | – | 1024 MiB | 4096 MiB | 16384 MiB |
| Nemotron-H-56B | 10 | 438 MiB | 320 MiB | 1280 MiB | 5120 MiB |
| Nemotron-H-47B | 5 | 365 MiB | 160 MiB | 640 MiB | 2560 MiB |
| Nano-12B-v2 | 6 | 72 MiB | 192 MiB | 768 MiB | 3072 MiB |
| Nano-9B-v2 | 4 | 69 MiB | 128 MiB | 512 MiB | 2048 MiB |
| Nemotron-3-Nano-30B-A3B | 6 (2 KV heads) | 24 MiB | 48 MiB | 192 MiB | 768 MiB |

**Scaled to our range:** 12 mixer layers, d=512, 8 KV heads × 64, Mamba-2 with expand 2 (16 heads × 64) and d_state 128:
- With **1 full-attention layer**: 2 / 8 / 64 MiB of KV at 1K / 4K / 32K, vs a 2.75 MiB constant state for the other 11 layers.
- With **all 12 full**: 24 / 96 / 768 MiB of KV.

So:
- (a) At ≥2K context, even r = 1/12 is **KV-dominated**.
- (b) The memory axis is essentially linear in the number of full layers × context.
- (c) GQA (fewer KV heads) shifts the whole frontier. **n_kv and head_dim must be held fixed across r**, or reported as a separate axis.
- (d) At these sizes, absolute memory is tiny next to 16 GB. Memory will be a *derived* metric (bytes per sequence), and the real GPU bottleneck will be prefill latency at 32K for high-r models.

---

## 7. What transfers to a 20–60M controlled study, and what does not

### 7.1 Transfers (adopt or adapt)
1. **Define r over token mixers, not over all layers.** Use a fixed block = [mixer → MLP], Nemotron-H style `M-` / `*-`, so that changing r swaps *only* mixers.
   - Waleffe's pure-SSM baseline had no MLPs while the hybrid had 50% MLPs. That is exactly the confound to avoid.
   - Report both conventions: Nemotron-H-8B is "7.7% of layers" but "14.3% of mixers".
2. **Parameter matching per layer type.** Waleffe adjusted attention heads at constant head size, and MLP expansion, so that every layer type has about the same parameter count. For us: match each softmax-attention mixer to the linear mixer's parameter count, or report the total-parameter drift across r (target <2%) and fix it with the MLP width.
3. **Uniform placement baseline.** Use Waleffe/Megatron Algorithm 1 (§5 patterns), but **hand-fix r = 1/2 to strict alternation**. Pin the first layer to linear: Waleffe/Nemotron rule (a), with the first Mamba layer supplying position information under NoPE.
4. **Placement arm (H4).** At a fixed small k (e.g. 2 of 12 or 3 of 16), compare:
   - uniform (Algorithm 1);
   - first/middle/last (Hymba-style, k=3);
   - all-early;
   - all-middle;
   - all-late.
   Evaluate on **retrieval**, not just loss. Waleffe's "no effect" claim rests on loss alone, and Jet-Nemotron's "large effect" comes from the retrofit regime.
5. **Evaluation suite.** Reuse from these papers:
   - **Phonebook** (standard and reversed formulations; accuracy vs book length; also digit-level "fuzzy" accuracy, Fig. 3b);
   - **RULER**: NIAH single/multi-key/multi-value/multi-query, VT (multi-hop), CWE/KWE (aggregation). The RULER clone is at `third_party/ruler`.
   - These line up with our retrieval / multi-hop / aggregation taxonomy.
   - Evaluate at fixed lengths 1K / 2K / 4K / 8K / 16K / 32K, *including beyond the trained length*: Waleffe shows extrapolation differs by architecture.
   - Use ≥400 samples per cell, matching Waleffe's RULER protocol.
6. **Hold PE fixed across r.**
   - Waleffe's 840M result favours NoPE for hybrids after context extension. But a NoPE *all-full* endpoint is non-standard, and Waleffe compared RoPE-Transformer vs NoPE-hybrid.
   - Recommendation [INF]: run the main sweep with one PE scheme for every r, and run a PE ablation (RoPE vs NoPE) at 2 ratios, e.g. r = 1/12 and r = 1. Otherwise length-extrapolation results cannot be attributed to r.
7. **Two-stage context recipe.** Pretrain at short context, then run a CPT stage at the target length with a constant low LR:
   - Waleffe: 50B tokens at 16K/32K, LR 3e-5 → 3e-6.
   - Nano 2: 18.9B tokens at 512K, LR 4.5e-6.
   - NVIDIA found **longer-than-target CPT sequences plus synthetic long-document QA** improve RULER (73.7 → 81.0).
   - Waleffe found packing unrelated documents may hurt hybrids on distractor-heavy multi-doc QA [HYP].
   - For us [INF]: pretrain at 2K–4K, then CPT for ~5–10% of tokens at 16K–32K with the *same* data for every r. Prefer naturally long documents (e.g. books/PG-19-like) over packed short ones, or at least use document-boundary-aware masking.
8. **GQA is cheap in hybrids** (Waleffe: +0.04% PPL), but it changes the KV axis. Fix n_kv across r.
9. **Pilot cheaply.** Nemotron-Flash found 10B-token-PPL ranks correlate with 100B-token ranks (Spearman 0.888). A short-run screen of configurations is defensible **for PPL**. It has not been validated for retrieval metrics: pilots must check rank stability on Phonebook/NIAH as well.
10. **Do not use validation loss as the selection criterion.** Waleffe's ratio optimum is a loss optimum. Nemotron-H notes "models with better loss curves [can] perform worse on downstream evaluation". Nemotron-Flash Appendix G flags the LM-proxy limitation.

### 7.2 Does not transfer
1. **The "8%" number itself.**
   - At 12–16 layers, 8% is about 1 layer. The ratio grid is coarse: k ∈ {0, 1, 2, 3, 4, 6, 12} of 12.
   - The optimum was found on loss at 130M with an unknown MLP setting.
   - Nemotron 3 Nano is at 11.5% of layers / 20.7% of mixers, so even NVIDIA has drifted.
2. **Scale-dependent effects.** MMLU-format and in-context-learning gaps that close only after more than 1.5T tokens (Waleffe Fig. 6) are irrelevant at our ~1–5B-token budgets. Expect our pure-linear models to be *much* weaker on retrieval than any number reported here.
3. **State-size ratios.**
   - Mamba-2 state scales with d_model (heads = 2d/64), so at d = 384–640 the recurrent state is tiny.
   - The state/KV trade-off at 20–60M is not the trade-off at 8B.
   - Any "x% attention suffices" claim may shift because the linear layers' memory capacity is so much smaller [INF].
4. **Throughput claims.** H100/A10G, vLLM vs Megatron vs TRT-LLM, maximum-batch serving. Our RTX 5080 numbers with eager PyTorch or Triton kernels will be dominated by kernel launch and implementation effects. Nemotron-Flash Table 14 shows a 5–7× spread across stacks for the *same* model.
   - Use CUDA Graphs or `torch.compile` for decoding.
   - Report bytes and FLOPs alongside wall-clock time.
   - Benchmark on an idle GPU. The shared CARLA process must be stopped for timing runs.
5. **FP8 recipes, MiniPuzzle/Minitron compression, RL/SFT long-context tricks, MoE.** None of these belong in a from-scratch controlled sweep.
6. **The Mamba-2 CUDA path on sm_120 without nvcc.** NVIDIA's recipes assume `mamba_ssm` and `causal-conv1d` CUDA builds. On our machine the Triton paths (Mamba-2 SSD chunk scan, FlashLinearAttention for GLA/DeltaNet/GDN) are the realistic route. This is an engineering risk to verify in Stage 4 [INF].

### 7.3 Concrete configuration suggestions from this family (for the methodology stage to decide)
- **Depth:** 12 or 16 mixers (+12/16 MLPs), d_model 384–512.
  - Gives about 20–60M non-embedding parameters with a ~32K vocabulary [INF, to be verified in code].
  - 16 mixers gives finer r: k ∈ {0, 1, 2, 3, 4, 6, 8, 16}.
- **Linear mixer:** Mamba-2 (NVIDIA's choice) or Gated DeltaNet/DeltaNet. Nemotron-Flash shows DeltaNet-family operators on the 500M PPL-latency frontier, with FLA Triton kernels. **Pick one for the main sweep**: mixing linear operator types would add a factor.
- **Optimizer:** AdamW β = (0.9, 0.95), weight decay 0.1, cosine or WSD, warmup ~1–2% of tokens, grad clip 1.0, untied embeddings, RMSNorm, no biases.
  - This matches every NVIDIA recipe above.
  - Peak LR must be tuned at our scale: NVIDIA's 2.5e-4–1e-3 are for 0.8B–30B models.
- **Default placement:** uniform per Algorithm 1, first layer linear.
- **Placement arm:** at k = 2 or 3.
- **Evaluation:** Phonebook plus the RULER subset at 1K–32K; plus validation loss on in-distribution and long-document sets.

---

## 8. Open questions I could not resolve
1. The MLP ratio, token budget, sequence length and PE in Waleffe's 130M attention-ratio sweep (Fig. 4). The paper does not state them. The 840M "confirmation" has no figure.
2. Whether Nemotron-H's "83% RULER @128K" (§5.2.2) refers to the 8B or the 47B reasoning model.
3. Why the 47B config (45 M / 48 −) differs by one layer type from the text (44 / 49).
4. Which 12B layers Nano 2 pruning removed. My inference that the first and last attention layers were dropped comes from the pattern shape only.
5. Whether from-scratch placement effects exist on *retrieval* metrics. No NVIDIA paper tests this: Waleffe measured loss only, and Jet-Nemotron is a retrofit.
6. Bolt-74B-A9B and Bolt-E4B: no technical report found. Configs only.

---

## 9. Sources consulted (with access level)
- Nemotron-H, arXiv 2504.03624v4. https://arxiv.org/abs/2504.03624 / https://arxiv.org/html/2504.03624 `[read-full]` (no appendices exist).
- Waleffe et al., arXiv 2406.07887v1. https://arxiv.org/abs/2406.07887 / https://arxiv.org/pdf/2406.07887 `[read-full]` (incl. Appendix A; Figs. 4, 5, 7 read from the PDF).
- Nemotron Nano 2, arXiv 2508.14444v4. https://arxiv.org/html/2508.14444 `[read-section]`.
- Nemotron-Flash, arXiv 2511.18890v1. https://arxiv.org/html/2511.18890 / https://arxiv.org/pdf/2511.18890 `[read-full]`.
- Nemotron 3 Nano, arXiv 2512.20848v1. https://arxiv.org/html/2512.20848 `[read-section]`.
- Jet-Nemotron, arXiv 2508.15884v3. https://arxiv.org/html/2508.15884 `[read-section]`.
- Hymba, arXiv 2411.13676v1. https://arxiv.org/html/2411.13676 `[read-section]`.
- HF configs `[code-inspected]`: `https://huggingface.co/nvidia/{Nemotron-H-8B-Base-8K, Nemotron-H-56B-Base-8K, Nemotron-H-47B-Base-8K, Nemotron-H-4B-Base-8K, Nemotron-H-4B-Instruct-128K, Nemotron-H-8B-Reasoning-128K, Nemotron-H-47B-Reasoning-128K, NVIDIA-Nemotron-Nano-12B-v2-Base, NVIDIA-Nemotron-Nano-12B-v2, NVIDIA-Nemotron-Nano-9B-v2, NVIDIA-Nemotron-Nano-9B-v2-Base, NVIDIA-Nemotron-3-Nano-30B-A3B-BF16, NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16, Nemotron-Flash-1B, Nemotron-Flash-3B, NVIDIA-Nemotron-3.5-Bolt-74B-A9B-BF16, NVIDIA-Nemotron-3.5-Bolt-E4B-BF16, Hymba-1.5B-Base}/resolve/main/config.json`.
- Model cards read for one fact each: Nemotron-H-4B-Base-8K (pruned and distilled from the 8B with 380B tokens), Nemotron-H-8B-Base-8K.
- Megatron-LM @ `928ce7c`. https://github.com/NVIDIA/Megatron-LM `[code-inspected]`: `megatron/core/models/hybrid/hybrid_layer_allocation.py`, `megatron/core/models/hybrid/layers/utils.py`, `megatron/core/ssm/mamba_mixer.py` (state-cache API, skimmed), `examples/mamba/{README.md, train.sh}`.
- Web searches on 2026-11-09 for "Nemotron Edge". Results only, not read: unite.ai, oodaloop, the NVIDIA blog post on Nemotron 3.5 Super VL / Bolt, the github.com/NVIDIA-NeMo/Nemotron README snippet.

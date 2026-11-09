# 02 — Attention Architecture: Mechanisms, Hybrid Ratio/Placement Evidence, and Design Recommendation

Agent 1 (Attention Architecture Researcher) · 2026-11-09 · repo `outeropt`
Machine-readable companion: `reports/literature/db_architecture.yaml`

## How to read this report

- Each factual claim about a paper or repo carries a URL and an access tag:
  **[read-full]** (whole paper read), **[read-section]** (the relevant sections/tables of the HTML were read through WebFetch), **[abstract-only]** (arXiv abstract page only), **[code-inspected]** (source code read in `third_party/`, commit given), **[config-inspected]** (a released model config file was read).
  - WebFetch returns a model-summarised view of each page, so I did not check numbers against the original PDF typography. "[read-section]" means the relevant tables came through that pipeline. The important numbers are repeated verbatim in §3 so they can be spot-checked.
- Every statement is labelled with one of three epistemic tags:
  - **(E)** an established finding: reported in a primary source and replicated, or directly measured by me.
  - **(R)** a reported finding from one source, not replicated.
  - **(I)** my own inference or recommendation.
- **(M)** marks something I measured in this session on the project GPU. The scripts are in the session scratchpad. The GPU was **shared with CARLA (~5.7–8 GB, 28–36 % util)** during these runs, so **all timings are indicative only**.
- "Recollection" means a detail I know from prior reading but did not re-verify in this session. Each one is flagged and should be checked before anyone relies on it.

Code inspected (shallow clones already present in `third_party/`; I reused them and did not re-clone):

| Repo | Commit | Used for |
|---|---|---|
| fla-org/flash-linear-attention | `98193113ecb7835143252a786e06933e371b67cc` (2026-11-08) | layer definitions, param counts, kernels, hybrid config |
| NVlabs/GatedDeltaNet (`third_party/gated_deltanet`) | `b53d6d3a161267432a79c1c04af69fa52bddc921` | official GDN / GDN-H1 configs |
| NVlabs/Jet-Nemotron | `94cbd12404637950722a09f70c8649ff1b1f2793` | full-attention placement of released models |
| NVlabs/hymba | `df5b2a95b9905534eb162305a5414bc43e37e84b` | barebones global/SWA layout |
| HazyResearch/zoology | `1ad20d193b6113cae1e8f3c655c300d7b4b3f4bb` | MQAR generator |
| Installed venv | torch 2.11.0+cu128, triton 3.6.0, fla 0.6.0; GPU capability (12,0) | kernel smoke tests |

---

## 0. Executive summary

1. **No public study is a clean match for our design.** I found none that holds parameters, data and recipe fixed and sweeps the full-attention fraction *from 0 to 1* at the 20–60M scale while measuring retrieval, multi-hop and aggregation separately at 1K–32K.
   - The closest is *A Systematic Analysis of Hybrid Linear Attention* (Wang et al. 2025, arXiv 2507.06457) [read-section]. It sweeps 24:1, 12:1, 6:1, 3:1 and pure, at 340M and 1.3B, with a 4K context and RULER averaged into a single "recall" number. It does not report an all-full endpoint inside the same sweep or any placement manipulation, and it does not break results down by task.
   - *Hybrid Architectures for Language Models: Systematic Analysis and Design Insights* (Bae et al., Meta, arXiv 2510.04800) [read-section] sweeps Transformer:Mamba-2 at 1:1, 1:3, 1:5, 1:12 plus both endpoints at 350M and 1B. It includes a small placement ablation, but its quality measures are LM loss and short-task accuracy; NIAH appears only in figures.
   - *Rethinking the Role of Efficient Attention in Hybrid Architectures* (Qiao et al., Tsinghua, arXiv 2606.15378) [read-section] trains 15M–477M (non-embedding) models at 16K context. It mostly uses 1:1, and the only ratio contrast is 1:1 vs 1:3.
   - Our study therefore fills a real gap (I).
2. **Evidence across sources (E/R): LM loss is nearly flat in the ratio; recall is not.**
   - Language-modelling loss and short-task accuracy are almost insensitive to the fraction of full attention once it is ≥ ~4–8 % (2507.06457 Table 6; 2510.04800; Mamba-2 paper Table 2; Jamba Table 4; Kimi Linear Table 1).
   - Recall/retrieval rises steeply from 0 to the first few attention layers and then saturates. In 2507.06457 the RULER average is 0.256 for pure linear, 0.338 at 24:1, 0.390 at 6:1 and 0.397 at 3:1.
   - Several studies find hybrids *beat* all-full on LM loss: the Mamba-2 paper (6/48 attention layers, ppl 8.26 vs 8.68 for Transformer++), Bae et al. (1:1 hybrid 2.725 vs 2.750 DCLM loss), and Kimi Linear (3:1 val ppl 5.65 vs 5.77 for MLA-only).
   - This is direct prior support for H1, H2 and H6 on recall/LM, but at larger scale, at ≤ 8K context and with aggregated recall metrics. **H3 (task dependence) and H5 (growth with length) are essentially untested in controlled form.**
3. **Placement evidence is thin and mixed (R).**
   - Bae et al. report that "front" placement is bad and "middle" is recommended, but their table shows ≤ 0.004 nats differences, and End is sometimes best (Table 11).
   - Jet-Nemotron (arXiv 2508.15884, [read-section] + [code-inspected]) finds that *which* layers matter differs between MMLU and retrieval: only 2–3 layers are critical. Its released 2B model keeps full attention at layers 14 and 19 of 28 (0-indexed).
   - Waleffe et al. (arXiv 2406.07887) put attention evenly with the first layer recurrent.
   - Retrieval probing gains concentrate in middle full-attention layers (2606.15378). Samba's entropy analysis (arXiv 2406.07522) finds low-entropy "retrieval" attention in middle layers.
   - H4 is plausible but not established (I).
4. **Positional encoding interacts strongly with the ratio (R, multiple sources).**
   - Jamba, Waleffe et al.'s Mamba-2-Hybrid, and Kimi Linear (NoPE on all MLA layers) all drop RoPE from attention in hybrids.
   - In 2606.15378, NoPE on the full layers of a 0.22B SWA-128 hybrid raised RULER-NIAH@16K from 49.58 to 67.81.
   - In *Rope to Nope and Back Again* (arXiv 2501.18795), NoPE-global/RoPE-local at 1:3 beat a RoPE baseline at 128K–256K.
   - **The positional policy is therefore a first-order confound** and must be fixed by pre-registration (§6, §7).
5. **Recommendation (I): primary linear mechanism = Gated DeltaNet (GDN).**
   - Use it exactly as implemented in FLA (commit above), with key dim = d/2, value dim = d, output gate on and short conv k=4. That gives 4.03·d² params per mixer, within 1 % of MHA's 4·d².
   - Secondary mechanisms for generalisation:
     - (i) **GLA**: the same kernel family and projection geometry, but no delta rule and a channel-wise gate. Its standalone recall is much weaker (pure-GLA RULER average 0.144 vs GDN 0.356 in 2507.06457), so it tests whether the curve shape depends on standalone recall.
     - (ii) **Sliding-window attention with w = 64**, which has the same inference state as the GDN layer (≈ 128 KiB/layer). It is a "cheap softmax" control with a different inductive bias.
   - Not recommended as primary: Mamba-2 (param-matching awkward at 6.4–6.7·d² by default; fast path needs `mamba_ssm`/`causal_conv1d` CUDA builds and there is no system nvcc), RWKV-7 (Jet-Nemotron excluded it for low training throughput; many extra low-rank params), RetNet (8·d² by default and the weakest hybrid results).
6. **Verified on this machine (M).**
   - FLA's Triton `chunk_gated_delta_rule` forward and backward runs on the RTX 5080 (sm_120) with torch 2.11/triton 3.6.
   - It matches a naive fp32 recurrence. Max abs error is 2.9e-4 in fp32 and 1.2e-3 in bf16, against an output abs-max of ≈ 0.17.
   - `fused_recurrent_gated_delta_rule` (the decode kernel) matches to 6e-8 in fp32.
   - **Gotcha:** its positional signature is `(q, k, v, g, gk, gv, beta, …)`. Calling it positionally as `(q,k,v,g,beta)` silently produces NaN, so always pass keywords.
   - PyTorch SDPA FLASH, EFFICIENT and cuDNN backends all run on sm_120.
   - `flash_attn` is **not installed**, and FLA's own `Attention` layer raises ImportError without it. We must write our own SDPA-based attention mixer (I).
7. **Measured single-layer fwd+bwd speed at d=512, shared GPU (M, indicative only):**

| T | Attention (SDPA) | GDN | GLA |
|---|---|---|---|
| 2K | 5.76 Mtok/s | 4.10 Mtok/s | 5.92 Mtok/s |
| 8K | 2.61 Mtok/s | 4.38 Mtok/s | 5.91 Mtok/s |
| 32K | 0.84 Mtok/s | 3.27 Mtok/s | 5.16 Mtok/s |

   The attention/GDN crossover is ≈ 4K tokens, which is consistent with MiniMax's "a few thousand tokens" crossover claim.

---

## 1. Mathematical formulations

Notation:
- Model width d. A mixer has H heads with key dim d_k and value dim d_v per head.
- T is the context length, w the window, b the bytes per element (bf16 = 2, fp32 = 4).
- State/cache sizes are **per layer, per sequence**.
- Row-vector convention: S_t ∈ ℝ^{d_k×d_v}, o_t = S_tᵀ q_t (equivalently o_t = q_tᵀ S_t).

### 1.1 Softmax attention (baseline "full" layer)

o_t = Σ_{s≤t} softmax_s(q_tᵀk_s/√d_k) v_s

- **Inference state (KV cache):** 2 · H_kv · d_h · T · b. For MHA with H_kv·d_h = d this is **2·d·T·b**.
  - At d = 512 in bf16 that is 2 KiB/token/layer, i.e. **64 MiB per layer at T = 32K**. (E, arithmetic)
- **Prefill cost:** the score+value matmuls cost ≈ 2·T²·d FLOPs causal (4·T²·d without the causal half), plus 8·d²·T for the projections (forward).
- **Decode cost:** O(T·d) per token. Decode is memory-bandwidth-bound because the whole KV cache is read each step (Griffin §5 makes the same "time ≈ (params + batch × cache)/bandwidth" argument, https://arxiv.org/abs/2402.19427 [read-section]).
- **Numerics:** softmax is computed in fp32 inside flash/SDPA kernels and is stable.
- **Position:** none intrinsic. RoPE, NoPE (causal mask gives implicit position) or other schemes are added.
- **Params:** q, k, v, o = 4·d² (MHA, no bias).

### 1.2 Linear attention (Katharopoulos et al. 2020)

Source: https://arxiv.org/abs/2006.16236 [abstract-only]. The equations below are the standard form from that paper (recollection) and match FLA's `fla/layers/linear_attn.py` signature [code-inspected].

S_t = S_{t-1} + φ(k_t) v_tᵀ,  z_t = z_{t-1} + φ(k_t),  o_t = S_tᵀφ(q_t) / (z_tᵀφ(q_t)), with φ = elu+1 in the original paper.

- **State:** H·d_k·d_v (+ H·d_k for the normaliser).
- **No forgetting:** the state's norm grows without bound with T, and the denominator can become ill-conditioned. This is why every modern variant adds decay/gating and usually replaces the denominator with an output norm (E: GLA, RetNet, Mamba-2 and GDN all use decay plus output normalisation; see the respective entries).

### 1.3 RetNet (retention)

Source: https://arxiv.org/abs/2307.08621 [abstract-only].
- The abstract confirms parallel, recurrent and chunkwise forms and O(1) inference.
- From recollection (not re-verified): the update is S_t = γ_h S_{t-1} + k_t v_tᵀ with a **fixed, data-independent** per-head decay γ_h, plus xPos-style rotation on q/k.
- FLA `MultiScaleRetention` defaults are expand_k = 1 and expand_v = 2, which gives **8.0·d² params** (M, instantiated). Matching attention would need non-default dims.

### 1.4 GLA (Gated Linear Attention)

Source: https://arxiv.org/html/2312.06635 [read-section].

S_t = Diag(α_t) S_{t-1} + k_t v_tᵀ,  α_t = σ(x_t W_{α1} W_{α2} + b)^{1/τ}, with low-rank 16 and τ = 16.

- The output is gated by Swish(x W_r) after a per-head norm.
- Defaults: d_k = d/2, d_v = d, 4 heads. The paper states ≈ 4·d² params; I measured **4.05·d²** at d = 512 (M).
- **State:** H·(d/2H)·(d/H) = d²/(2H). At d = 512, H = 4 that is 32,768 elements.
- **Numerics:** cumulative gate products underflow. The paper uses log-space plus *secondary chunking* (fp32 inside sub-chunks, half precision across them) [read-section].
- FLA default `use_short_conv=False` [code-inspected]. The paper text I read does not mention a short conv.
- Reported results:
  - 1.3B/100B: Wiki ppl 17.22 vs Transformer++ 16.85.
  - Recall FDA/SWDE/SQuAD: 19.9/50.6/42.6 vs Transformer++ 27.4/66.6/31.5.

### 1.5 Mamba (S6) and Mamba-2 (SSD)

Sources: Mamba https://arxiv.org/abs/2312.00752 [abstract-only]; Mamba-2 https://arxiv.org/html/2405.21060 [read-section].

- **Mamba-1** (recollection): per channel, h_t = exp(Δ_t A) ⊙ h_{t-1} + Δ_t B_t x_t, y_t = C_tᵀ h_t, with diagonal A and N = 16.
  - State: d_inner·N.
- **Mamba-2/SSD** [read-section]: h_t = A_t h_{t-1} + B_t x_t with A_t = a_t·I (scalar per head) and y_t = C_tᵀ h_t. The paper says "P ∈ {64,128} is typically chosen."
  - In linear-attention terms, C ↔ q, B ↔ k, x ↔ v. The scalar data-dependent decay a_t = exp(Δ_t·A_h) makes Mamba-2 ≈ *scalar-gated* linear attention (I, standard reading of SSD).
  - **State:** d_inner·N plus conv state. With d = 512, expand = 2 and N = 128: 131,072 elements, i.e. 512 KiB in fp32. That is **4× the matched GDN state** below.
- **Params** (FLA `Mamba2`, M):
  - expand 2, N 128, P 64: **6.56·d²**.
  - expand 1, N 64: 3.28·d².
  - Matching 4·d² needs expand ≈ 1.25, which is non-standard.
- **Implementation** (M): in our venv FLA prints "fast path is not available … `selective_state_update` is None … Falling back to the naive implementation" and "Mamba2 does not recommend using Triton's conv1d backend". The real fast path requires `mamba_ssm`/`causal_conv1d` CUDA builds. With no system nvcc that is a risk for an sm_120 consumer GPU (I).

### 1.6 DeltaNet

Source: https://arxiv.org/abs/2406.06484 [abstract-only].

S_t = S_{t-1}(I − β_t k_t k_tᵀ) + β_t v_t k_tᵀ  (written with S ∈ ℝ^{d_v×d_k}; o_t = S_t q_t).

- This is one step of online least-squares (delta rule): the value currently stored under key k_t is *replaced* in proportion to β_t ∈ (0, 1).
- With ‖k_t‖₂ = 1, the transition (I − β k kᵀ) has eigenvalues {1 − β, 1, …, 1} ⊂ [0, 1]. The state cannot blow up (I, linear algebra; this is why FLA L2-normalises q and k).
- Training is parallelised with a WY/Householder representation.
- Abstract claims: a 1.3B/100B model beats Mamba and GLA on perplexity. Two hybrids beat the Transformer: SWA in every other layer, and 2 global attention layers.
- Gated DeltaNet's Table 2 (below) shows pure DeltaNet has the *best* S-NIAH-1 at 8K (98.8).

### 1.7 Gated DeltaNet (GDN) — recommended primary

Source: https://arxiv.org/html/2412.06464 [read-section]; FLA `fla/layers/gated_deltanet.py` [code-inspected].

S_t = S_{t-1} · α_t (I − β_t k_t k_tᵀ) + β_t v_t k_tᵀ,  o_t = S_t q_t   (paper Eq. 10)

FLA parameterisation [code-inspected]:
- **Projections:**
  - q, k: d → H·d_k
  - v, g: d → H_v·d_v
  - a, b: d → H_v (one scalar per head each)
  - o: H_v·d_v → d
- **Decay:** α_t = exp(g_t) with g_t = −exp(A_log) · softplus(a_proj(x) + dt_bias). This is Mamba-2-style; A_log is initialised to log U(0, 16), and dt_bias inverts softplus of dt ∈ [1e-3, 0.1]. Both have no weight decay.
- **Write strength:** β_t = σ(b_proj(x)). An optional `allow_neg_eigval` doubles β to allow eigenvalues in [−1, 1].
- **Short conv:** a causal depthwise conv of kernel 4 followed by SiLU, applied separately to q, k and v. The FLA code warns: *"ShortConvolution is crucial to the performance. Do not turn it off…"*
- **Normalisation:** q and k are L2-normalised in-kernel, and the scale is 1/√d_k.
- **Output:** per-head RMSNorm, gated by swish(g_proj(x)) (FusedRMSNormGated), then o_proj.
- **FLA default allocation:** H·d_k = 0.75·d and expand_v = 2, giving **≈ 6·d²** ("Similar to Mamba2, each layer contains around 6*hidden_size*hidden_size parameters"). The official NVlabs repo uses expand_k = 0.75 and expand_v = 1.5 [code-inspected]. Because of this, in the GDN paper the *pure GDN 0.4B model has 11 layers while GDN-H1 has 12* at the same width (`lit_gpt/config.py`): the papers match total params by changing layer count, which is a confound we must avoid (I).
- **Param-matched variant** (proposed here): H·d_k = d/2, d_v = 2·d_k (so H_v·d_v = d), gate on.
  - Cost: q 0.5 + k 0.5 + v 1 + g 1 + o 1 = 4·d², plus small extras. Measured **4.032·d² at d = 512** (1,056,904 vs 1,048,576 for MHA; +0.8 %) (M).
- **State:** H·d_k·d_v stored in **fp32** (the chunk kernel returns an fp32 final state; M). Plus a conv cache of (k−1)·(2·H·d_k + H_v·d_v) elements.
  - At d = 512, H = 4, d_k = 64, d_v = 128: 32,768 fp32 = **128 KiB**, plus 3·1024 bf16 = 6 KiB, total **≈ 134 KiB per layer, independent of T**.
  - Break-even against one MHA attention layer (2 KiB/token): **≈ 67 tokens**. At 32K the attention layer is ≈ 490× larger.
- **Complexity:**
  - Chunkwise training/prefill with chunk C = 64 costs per head ≈ O(T·C·(d_k + d_v) + T·d_k·d_v + T·C·d_k) FLOPs, linear in T.
  - Decode costs O(H·d_k·d_v) per token, constant in T.
- **Reported results** (paper, 1.3B/100B FineWeb-Edu, 4K):
  - S-NIAH-1 (1K/2K/4K/8K): GDN 98.4/88.4/91.4/91.8; Mamba2 99.2/98.8/65.4/30.4; DeltaNet 97.4/96.8/99.0/98.8.
  - S-NIAH-2: GDN 100/99.8/92.2/29.6.
  - Recall-intensive average: GDN 30.6, Mamba2 29.8, Transformer++ 37.0, **GDN-H1 39.0, GDN-H2 40.1**.
  - LongBench average: GDN 16.6, Transformer++ 11.0, H1 17.8, H2 18.4.
- **GDN-H1 layout** [code-inspected, `gated_delta_per_layer=2`, `local_window=2048`]: GDN and SWA-2048 alternate 1:1. The official hybrids use **sliding-window** attention, not global attention.

### 1.8 Kimi Delta Attention (KDA, Kimi Linear)

Source: https://arxiv.org/html/2510.26692 [read-section].

S_t = (I − β_t k_t k_tᵀ) Diag(α_t) S_{t-1} + β_t k_t v_tᵀ

- This is GDN with a **channel-wise** (per key dim) decay instead of a per-head scalar decay.
- d_k = d_v = 128. Short conv + Swish on q, k, v; L2-normalised q and k; low-rank sigmoid output gate after a head-wise RMSNorm.
- **All MLA (full) layers use NoPE.**
- FLA has `fla/layers/kda.py` (defaults: head_dim 128, 16 heads, expand_v 1, short conv on) [code-inspected].

### 1.9 RWKV-7 "Goose"

Source: https://arxiv.org/abs/2503.14456 [abstract-only].
- The abstract describes a "generalized delta rule with vector-valued gating and in-context learning rates, plus a relaxed value replacement rule". I did not read the exact update and do not reproduce it here.
- FLA `RWKV7Attention` has four extra low-rank projections (decay, gate, a, v) [code-inspected signature].
- Jet-Nemotron excluded RWKV-7 from its block search because "its training throughput was significantly lower" (https://arxiv.org/html/2508.15884 [read-section]).

### 1.10 HGRN2

Source: https://arxiv.org/abs/2404.07904 [abstract-only] (COLM 2024).
- HGRN2 adds outer-product state expansion to HGRN without adding parameters, and has a linear-attention interpretation.
- From recollection: forget gate f_t with an input gate tied to (1 − f_t), and a learned lower bound on the forget gate that increases with depth.
- FLA `HGRN2Attention(expand_ratio=128)` has **4.00·d² params** (M).
- **2507.06457 finds HGRN2 is the best *hybrid* partner** (6:1 RULER 0.434; zero-shot 0.456 at 340M), despite weaker standalone recall than GDN.

### 1.11 Based

Source: https://arxiv.org/html/2402.18668 [read-section].
- Global linear attention with a 2nd-order Taylor feature map 1 + qᵀk + (qᵀk)²/2 on projected d′ = 16 features, plus exact SWA with a tiny window (≤ 64–128) and short convs (filter 3).
- **State per head:** (1 + d′ + d′²)·d_v = 273·d_v.
- Layer mix: ~20 % linear attention, ~20 % SWA, ~60 % gated conv.
- Main result: an explicit **recall vs state-size Pareto frontier** on MQAR (Mamba best among prior work, Based extends it).
- Recall at 1.3B/50B, SWDE/FDA/SQuAD:

| Model | SWDE | FDA | SQuAD |
|---|---|---|---|
| Based | 64.45 | 30.40 | 41.62 |
| Mamba | 52.75 | 18.51 | 35.92 |
| Transformer++ | 76.50 | 80.47 | 43.47 |

### 1.12 Sliding-window attention (SWA) as the cheap layer

- KV cache = 2·H_kv·d_h·min(T, w)·b, bounded by the window.
- With MHA at d = 512 in bf16: w = 64 → **128 KiB**, which equals the GDN matched state (I, arithmetic); w = 512 → 1 MiB; w = 2048 → 4 MiB.
- Prefill is O(T·w·d); decode is O(w·d).
- Evidence:
  - MiniMax-01 Table 4: SWA hybrids score 46.8–53.9 NIAH at 1B vs 95.7 for the lightning hybrid (https://arxiv.org/html/2501.08313 [read-section]).
  - Large windows cause "Large-Window Laziness" in full-attention retrieval heads, and small-window SWA with NoPE-on-full is recommended (2606.15378 [read-section]).
  - Griffin's local attention (w = 1024) beats global MQA at 2K–8K at 400M (Appendix E).

### 1.13 RG-LRU (Griffin / RecurrentGemma)

Source: https://arxiv.org/html/2402.19427 [read-section].

r_t = σ(W_a x_t + b_a), i_t = σ(W_x x_t + b_x), a_t = a^{c·r_t} with a = σ(Λ) and c = 8; h_t = a_t ⊙ h_{t-1} + √(1 − a_t²) ⊙ (i_t ⊙ x_t)

- Computed in log-space (log a_t = −c·softplus(Λ)·r_t) for stability.
- **State:** a D_RNN vector, which is tiny. The block also has a Conv1D of width 4.
- Griffin pattern: 2 recurrent : 1 local-MQA (window 1024, RoPE).
- RecurrentGemma (https://arxiv.org/abs/2404.07839 [abstract-only]) is the open 2B/9B Griffin.

### 1.14 Summary table: mixer cost per layer (d = 512 worked example; M = measured by instantiation)

| Mixer | Update (core) | State/cache per layer | Params (FLA default) | Params (matched config) | Prefill | Decode/token | Uses RoPE? | Extras |
|---|---|---|---|---|---|---|---|---|
| Softmax MHA | softmax(qKᵀ)V | 2·d·T·b = 2 KiB/token → 64 MiB @32K | 4·d² | 4·d² | O(T²d) | O(Td) | yes / NoPE option | – |
| SWA (w) | windowed softmax | 2·d·w·b (w = 64 → 128 KiB) | 4·d² | 4·d² | O(Twd) | O(wd) | usually yes | – |
| Linear attn | S += φ(k)vᵀ | d²/H (+norm) | ~4·d² | 4·d² | O(T·d²/H) | O(d²/H) | no | feature map |
| RetNet | S = γS + kvᵀ | H·d_k·d_v | 8.0·d² (M) | needs custom dims | linear | const | xPos-like (recollection) | output gate |
| GLA | S = Diag(α)S + kvᵀ | d²/(2H) | 4.05·d² (M) | 4.06·d² with conv (M) | linear | const | no | low-rank gate, out gate, optional conv |
| Mamba-2 | h = a·h + B xᵀ | d_inner·N (512 KiB fp32 @ e2, N128) | 6.56·d² (M) | ~4·d² only at expand ≈ 1.25 | linear | const | no | conv4, gate z, D skip |
| DeltaNet | S = S(I − βkkᵀ) + βvkᵀ | d²/H | 4.04·d² (M) | same | linear | const | no | conv4, L2 qk |
| **GDN** | S = S·α(I − βkkᵀ) + βvkᵀ | **134 KiB** (H4, 64×128 fp32 + conv) | 6.05·d² (M) | **4.03·d² (M)** | linear | const | no | conv4×3, out gate, L2 qk |
| KDA | channel-wise α | H·128·128 | FLA kda.py | – | linear | const | no (MLA NoPE) | conv, low-rank gate |
| HGRN2 | Diag(f)S + (1 − f)⊗… | H·d_k·d_v (expand 128) | 4.00·d² (M) | same | linear | const | no | – |
| Based | Taylor-φ linear + SWA64 | 273·d_v per head + window | – | – | linear | const | – | conv3 |

---

## 2. Positional handling in published hybrids

| Model | Full-attn positional encoding | Linear/recurrent layers | Source |
|---|---|---|---|
| Jamba | **No RoPE** needed. At 1.3B/250B, NoPE vs RoPE: HellaSwag 39.6 vs 40.1, WinoGrande 71.5 vs 71.8 (Table 8) | Mamba (implicit position) | https://arxiv.org/html/2403.19887 [read-section] |
| Mamba-2-Hybrid (Waleffe) | **No position embeddings**. At 840M, RoPE gave similar base accuracy but lower accuracy after 16K extension | Mamba-2 | https://arxiv.org/html/2406.07887 [read-section] |
| Kimi Linear | **NoPE on all MLA layers**. The RoPE variant scored lower on long-context | KDA carries position | https://arxiv.org/html/2510.26692 [read-section] |
| MiniMax-01 | RoPE on **half** the head dim, base 10,000, softmax layers only | Lightning (not stated) | https://arxiv.org/html/2501.08313 [read-section] |
| Qwen3-Next | partial RoPE: `partial_rotary_factor` 0.25, `rope_theta` 1e7 | GDN, conv kernel 4 | https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct/raw/main/config.json [config-inspected] |
| Samba | SWA uses RoPE (base 10k). NoPE variant blew up beyond training length | Mamba conv4 | https://arxiv.org/html/2406.07522 [read-section] |
| Griffin | local MQA with RoPE | RG-LRU conv4 | https://arxiv.org/html/2402.19427 [read-section] |
| Bae et al. (Meta) | RoPE θ = 500k with Llama-3 scaling; SWA θ = 10k; suggests NoPE as a fix | Mamba-2 | https://arxiv.org/html/2510.04800 [read-section] |
| RNoPE-SWA (Cohere) | global full layers **NoPE**, local SWA **RoPE**, 3:1. Higher RoPE θ in local layers *hurts* NoPE retrieval (needle score 8.036 → 6.203 as θ goes 10k → 4M) | (all attention) | https://arxiv.org/html/2501.18795 [read-section] |
| 2606.15378 | NoPE on full layers of SWA-128 hybrid: S4 RULER-NIAH@16K 49.58 → 67.81; S5@32K 60.17 → 70.42 | SWA | https://arxiv.org/html/2606.15378v1 [read-section] |
| FLA hybrid config | `attn={'layers':[...], 'rope_theta':…}` **requires positive rope_theta**, so NoPE is not configurable without custom code | GDN etc. no RoPE | FLA `fla/models/configuration_utils.py` [code-inspected] |

**Take-away (I):**
- Linear layers in all these systems carry **no RoPE**. Position comes from decay plus short conv.
- Whether attention layers keep RoPE is a design axis with **documented large effects on long-context retrieval**.
- It interacts with r: at r = 1 there are no linear layers to provide position, so a NoPE all-full model is a weaker, non-standard baseline.
- See the policy in §5.4.

---

## 3. Evidence table: every ratio/placement result found (numbers verbatim from the fetched pages)

"Ratio" is written **linear:full** unless stated. *Controlled* means only the ratio changed inside that sweep (same width/tokens/recipe), as far as the paper describes.

| # | Study | Mechanism (cheap layer) | Scale / data / ctx | Ratios / placements tested | Key numbers | Controlled? | Access |
|---|---|---|---|---|---|---|---|
| 1 | Wang et al. 2025, *Systematic Analysis of Hybrid Linear Attention*, https://arxiv.org/html/2507.06457v2 | DeltaNet, GDN, GLA, HGRN, HGRN2, RetNet | 340M/20B and 1.3B/100B FineWeb-Edu, ctx 4K, FLA | 24:1, 12:1, 6:1, 3:1, pure (+ Transformer baseline). Placement: "interleaved in a fixed ratio r:1 … repeating the composite block" (positions not specified) | **RULER recall average across scales (Table 7):** pure 0.256 · 24:1 0.338 · 12:1 0.340 · 6:1 0.390 · 3:1 0.397. Per mechanism, pure→3:1: GDN 0.356→0.436; GLA 0.144→0.332; HGRN2 0.203→0.395 (6:1 0.434); DeltaNet 0.320→0.426. **LM average (Table 6)** flat: 0.472–0.480. Table 5: Transformer RULER 0.423 vs GDN-3:1 0.440, HGRN2-6:1 0.434 | Yes (ratio × mechanism) | read-section |
| 2 | Bae et al. (Meta), https://arxiv.org/html/2510.04800 | Mamba-2 (inter-layer); also intra-layer head split | 100M–3B; ablations 350M/1B, 60B tokens, ctx 8K | Transformer:Mamba = 1:0, 1:1, 1:3, 1:5, 1:12, 0:1; placement Front/Middle/End (1:1, 1:3, 1:5) | 1B DCLM / PG19 loss / few-shot: Transformer 2.750/2.875/52.0; Mamba 2.758/2.891/52.3; **1:1 2.725/2.847/54.0**; 1:3 2.733/2.859/54.0; 1:5 2.735/2.861/53.3; 1:12 2.741/2.866/53.1. **Placement (Table 11):** 1:3 Front 2.742/52.9, Middle 2.740/52.7, End 2.742/52.9; 1:5 Front 2.742/52.1, Middle 2.741/52.7, End 2.741/52.9. Text says front is worst, middle recommended; differences ≤ 0.004 nats. NIAH: Transformer ≈ 0 beyond 8K; hybrids OK to ≈ 1.5× train length (figure only) | Yes (16-layer examples; e.g., 1:5 uses attn at {5, 9, 13}) | read-section |
| 3 | Qiao et al. 2026, *Rethinking the Role of Efficient Attention in Hybrids*, https://arxiv.org/html/2606.15378v1 | SWA-128/512/2048, Lightning, Mamba-2, GDN | **15M, 31M, 65M, 104M, 477M non-emb**; up to 1000 tokens/param; ctx 16K; Muon | 1:1 alternation (main); 1:3 for SWA-128; head-wise vs layer-wise; NoPE-on-full | S4 (0.22B) 16K, Full vs SWA-128 vs SWA-128-NoPE: RULER 25.09/35.33/44.80; NIAH 35.95/49.58/67.81. S5 (0.66B) 32K: RULER 43.90/41.86/46.98. **1:3 vs 1:1:** similar loss; 1:3 worse log-LongPPL at small scale, gap closes with size. Restricting *full* attention to 2048 sharply hurts LongPPL; restricting efficient layers barely matters (even GDN) | Partly (ratio only 1:1 vs 1:3) | read-section |
| 4 | Kimi Linear, https://arxiv.org/html/2510.26692 | KDA vs MLA (NoPE) | ablation model 16 layers, 16 heads, ≈ 653M activated, 38.8B tokens, ctx 4K, same FLOPs | KDA:MLA = 0:1, 1:1, 3:1, 7:1, 15:1 | Train/val ppl: 0:1 9.45/5.77; 1:1 9.29/5.66; **3:1 9.23/5.65**; 7:1 9.23/5.70; 15:1 9.34/5.82. No conv: 9.29/5.70; no output gate 9.25/5.67 | Yes | read-section |
| 5 | Mamba-2 paper §9.2.3, https://arxiv.org/html/2405.21060 | Mamba-2 (SSD) | 350M, 48 layers | 0,1,2,3,4,5,6,7,9,11,15,24 attention layers out of 48 | Ppl: 0 → 8.60; 1 → 8.38; 2 → 8.32; 3 → 8.29; 4 → 8.29; 5 → 8.28; **6 → 8.26**; 7 → 8.27; 9 → 8.28; 11 → 8.30; 15 → 8.34; 24 → 8.50; Transformer++ 8.68. "Around 10 % … attention performs best" | Yes (LM ppl only) | read-section |
| 6 | Waleffe et al. 2024, https://arxiv.org/html/2406.07887 | Mamba-2 (+ MLP) | 130M/840M ablations; 8B/3.5T final | attention fraction sweep (values in figure) | Validation loss lowest at ≈ 8 % attention; MLP 30–50 % OK. 8B hybrid: 24 Mamba-2 + 4 attn + 28 MLP; first layer Mamba-2; attn evenly spread; NoPE; GQA. Hybrid beats Transformer on 12 short tasks (55.82 vs 53.17), MMLU 53.60 vs 50.07; RULER@16K 74.19 vs 62.33; Phonebook 100 % to ≈ 5.5K (train 4K). Pure Mamba/Mamba-2 phonebook fails from ≈ 500 tokens | Yes at small scale | read-section |
| 7 | Jamba, https://arxiv.org/html/2403.19887 | Mamba | 1.3B / 250B ablations | Attn:Mamba 1:3 vs 1:7; pure attn vs pure Mamba vs hybrid | 1:3 ≈ 1:7 ("virtually no performance difference"); OLLM 16.5 vs 16.0. ICL (attn / Mamba / hybrid): IMDB 84.1/48.8/90.9; QuAC 27.9/20.2/26.6; NarrativeQA 45.8/27.7/43.7. Final: 1 attn per 8 layers | Yes | read-section |
| 8 | MiniMax-01, https://arxiv.org/html/2501.08313 | Lightning attention (TransNormer) | 70M–7B scaling, ≤ 300B tokens, ctx 8K; hybrids at 1B/3B | 7:1 final (one softmax per 8 layers); hybrid vs pure vs SWA hybrids | Pure lightning falls short on NIAH. 1B NIAH: hybrid-lightning **95.7**, hybrid-HGRN2 91.8, hybrid-cosformer2 43.6; SWA hybrids 46.8–53.9. 3B: 98.0 vs 40.9–57.9. **Sweep behind 7:1 not reported** | Ratio sweep not shown | read-section |
| 9 | MiniMax M2 blog, https://huggingface.co/blog/MiniMax-AI/why-did-m2-end-up-as-a-full-attention-model | Lightning/SWA hybrids | production | – | Small-scale hybrids matched full attention on MMLU/BBH/MATH/LongBench but showed "clear deficits in complex, multi-hop reasoning" at scale. SWA-hybrid performance degraded with context in agentic tasks. Linear attention "far more sensitive to numerical precision" | No (qualitative) | read-full (blog) |
| 10 | Jet-Nemotron / PostNAS, https://arxiv.org/html/2508.15884 | linear blocks on frozen-MLP Qwen2.5 | 1.5B base | learned placement (super-net + beam search) vs uniform; block comparison | Only ≈ 2 layers critical for MMLU, 2–3 for retrieval, and they differ. Block table (MMLU/Math/Retrieval/Common): RetNet 53.6/29.9/63.7/58.1; Mamba2 51.5/26.0/68.9/57.5; GLA 55.8/31.2/66.6/58.5; DeltaNet 48.9/27.4/67.9/56.6; **GDN 55.6/32.3/69.3/58.7**. 2B config [code-inspected]: 28 layers, full attn at **{14, 19}**, SWA(1152) at {20, 21}, rest JetBlock. 4B: full at {17, 20, 32}, SWA at {5, 16, 19, 21, 22, 25, 27} of 36 | Post-hoc (distillation) | read-section + code-inspected |
| 11 | Mamba in the Llama, https://arxiv.org/html/2408.15237 | Mamba / Mamba-2 distilled from Llama-3-8B-Instruct | 8B distillation | 50 %, 25 %, 12.5 %, 0 % attention ("every n attention layers") | Mamba2-Llama3 zero-shot average: 50 % 63.84; 25 % 60.55; 12.5 % 58.21; 0 % 54.74. MT-Bench: 7.32/6.74/6.48/5.64 | Distillation, not pretraining | read-section |
| 12 | Samba, https://arxiv.org/html/2406.07522 | Mamba + SWA-2048 (+ MLP) | 421M–3.8B | layer-wise strategies at ≈ 1.7B | 15-task average: Samba 54.33; Mamba-SWA-MLP 53.77; Mamba 52.31; Mamba-MLP 51.38. One KV head improved ppl. Middle-layer attention low entropy (retrieval) | Partly | read-section |
| 13 | Griffin, https://arxiv.org/html/2402.19427 | RG-LRU + local MQA (1024) | 100M–14B | fixed 2:1; window ablation | No ratio/position ablation in text read. Window 1024 beats global MQA at 2K–8K (400M). Phonebook: Griffin perfect to ≈ window, then degrades | No ratio sweep | read-section |
| 14 | Gated DeltaNet, https://arxiv.org/html/2412.06464 | GDN (+ SWA-2048, + Mamba2) | 1.3B / 100B, ctx 4K | H1 = GDN:SWA 1:1 ([code-inspected] `gated_delta_per_layer=2`); H2 = Mamba2 + GDN + SWA | Recall average: GDN 30.6 → H1 39.0, H2 40.1 (Transformer++ 37.0). LongBench: 16.6 → 17.8 / 18.4 | Hybrid vs pure only | read-section + code-inspected |
| 15 | DeltaNet, https://arxiv.org/abs/2406.06484 | DeltaNet + SWA or + 2 global attention | 1.3B / 100B | SWA every other layer; 2 global layers | Both hybrids beat Transformer (abstract) | – | abstract-only |
| 16 | Based, https://arxiv.org/html/2402.18668 | Taylor linear + SWA64 + conv | 360M, 1.3B | ~20 % linear / 20 % SWA / 60 % conv | MQAR recall vs state-size frontier | Mix fixed | read-section |
| 17 | Zoology (Arora et al.), https://arxiv.org/abs/2312.04927 | gated conv vs attention | 17 models | hybrids with input-dependent sparse attention | 82 % of ppl gap due to associative recall; 70M attention > 1.4B gated conv on AR; hybrids close 97.4 % of gap | – | abstract-only + code-inspected (MQAR gen) |
| 18 | Repeat After Me (Jelassi et al.), https://arxiv.org/abs/2402.01032 | GSSMs vs Transformers | theory + pretrained | – | 2-layer Transformer can copy exponential-length strings; fixed-state GSSMs cannot copy beyond state | – | abstract-only |
| 19 | RNNs are not Transformers (yet) (Wen et al.), https://arxiv.org/abs/2402.18510 | RNN ± one attention layer | theory | +1 Transformer layer or RAG | RNNs cannot solve AR / IsTree even with CoT; **a single Transformer layer** closes the representational gap | – | abstract-only |
| 20 | Gather-and-Aggregate (Bick, Xing, Gu), https://arxiv.org/abs/2504.18574 | SSMs & hybrids | pretrained | – | Retrieval via few G&A heads; in hybrids attention takes the Aggregate role; disabling one G&A head in pruned Llama-3.1-8B drops MMLU 66 % → 25 % | – | abstract-only |
| 21 | Some Attention is All You Need for Retrieval, https://arxiv.org/abs/2510.19861 | RecurrentGemma-2B/9B, Jamba-Mini-1.6 | pretrained | ablate / sparsify attention | Attention ablation → 0 % retrieval; 15 % of heads keep near-perfect retrieval, 84 % MMLU | – | abstract-only |
| 22 | Rope to Nope and Back Again, https://arxiv.org/html/2501.18795 | SWA (RoPE) + full (NoPE) | Cohere-scale | 1:1, 1:3, 1:7 tested, 1:3 chosen | RULER retrieval @256K 57.1 → 74.8; QA @128K 36.0 → 46.0 | Partly | read-section |
| 23 | Qwen3-Next, https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct | GDN + gated attention | 80B-A3B | 3:1 (`full_attention_interval` 4; 48 layers = 12 × (3 GDN + 1 attn)) | GDN 16 QK heads / 32 V heads × 128; attn 16 Q / 2 KV × 256; partial RoPE 0.25 | No public ablation found | config-inspected (+ model card) |
| 24 | Hymba, https://arxiv.org/abs/2411.13676 | parallel SSM + attention heads per layer | 1.5B | – | 11.67× cache reduction vs Llama-3.2-3B. Barebones test config [code-inspected]: global layers [5, 11, 18, 25, 31], others SWA-2048 | No | abstract-only + code-inspected |
| 25 | Nemotron-H, https://arxiv.org/abs/2504.03624 | Mamba-2 + attention + MLP | 8B, 56B | not a ratio study | Up to 3× faster inference. **Mixes attention, Mamba-2 and MLP layers. NOT a controlled full-vs-linear ratio experiment** | No | abstract-only |
| 26 | Zamba, https://arxiv.org/abs/2405.16712 | Mamba + one shared attention block | 7B / 1T | – | shared attention module | No | abstract-only |
| 27 | Ring-linear (*Every Attention Matters*), https://arxiv.org/abs/2510.19338 | linear + softmax | 16B / 104B MoE | "systematically explored the ratio" | ratio and sweep not given in abstract | ? | abstract-only |
| 28 | Massive activations in hybrids, https://arxiv.org/abs/2608.12149 | GDN hybrids ≤ 1.3B | – | denser full attention | Pre-attention spikes just before full layers; full-attn output gates reduce them; deleting the 4 largest spike coordinates −21.9 % to −63.6 % accuracy | – | abstract-only |
| 29 | FlashMorph (*Morphing into Hybrid Attention Models*), https://arxiv.org/abs/2606.30562 | linear branch per layer, learned gates | conversion | budgeted subset selection | Learned selection beats heuristics (no numbers in abstract) | – | abstract-only |

### 3.1 What the evidence does and does not establish (for H1–H6)

- **H1 (partial sufficiency).** Supported for LM loss and aggregated recall at 0.3–1.3B, 4–8K context (#1, #2, #4, #5, #6, #7) (E).
  - In #1, the 3:1 hybrid *exceeds* the Transformer on the RULER average (0.440 GDN-3:1 vs 0.423).
  - For hard multi-hop, MiniMax (#9) reports that small-scale parity *did not hold at scale* (R). This is exactly the task-dependence risk (H3).
- **H2 (diminishing returns).** Supported (E):
  - Mamba-2 ppl falls from 8.60 → 8.38 with the first attention layer, versus 8.29 → 8.26 for layers 3–6.
  - RULER in #1 moves 0.256 → 0.338 at the first 4 % of attention, then 0.390 → 0.397 from 6:1 to 3:1.
  - Several studies find **non-monotonic** LM loss: too much attention is worse than ~10–25 % (#2, #4, #5).
- **H3 (task dependence).** Indirect evidence only:
  - Jet-Nemotron (#10): MMLU-critical and retrieval-critical layers differ.
  - Waleffe (#6): the hybrid beats the Transformer on NIAH but loses on keyword extraction / multi-document QA.
  - MiniMax (#9): multi-hop.
  - No controlled per-task ratio sweep exists (I).
- **H4 (placement).** Weak or mixed evidence (#2 small differences; #10 learned > uniform; #3 middle layers carry retrieval probes; #6 first layer recurrent) (R).
- **H5 (length).** Mostly qualitative:
  - All-full Transformers fail beyond the training length (#2, #6, #12) while hybrids extrapolate.
  - Within the training length there is no controlled ratio × length data (I).
  - **This confound (extrapolation vs capability) must be designed out** (§6).
- **H6 (frontier).** Partially supported (E/R): hybrids beat both endpoints on LM loss (#2, #4, #5) and on recall average (#1, #14). Measured-cost frontiers at small scale are not published.

---

## 4. Recommendation: primary and secondary linear mechanisms

### 4.1 Primary: Gated DeltaNet (FLA implementation)

Justification:

1. **Strongest standalone recall among the candidates in independent comparisons.**
   - Best block in Jet-Nemotron's controlled block search (Retrieval 69.3 vs Mamba2 68.9, GLA 66.6, RetNet 63.7) (#10).
   - Best pure-linear RULER in 2507.06457 (0.356 vs DeltaNet 0.320, HGRN2 0.203, GLA 0.144).
   - Better than Mamba2 on S-NIAH-1 at 4K/8K (91.4/91.8 vs 65.4/30.4) in its own paper (R).
   - (I) A strong linear partner makes the study conservative: if a few full layers still help GDN, the result generalises "upward". If GDN alone saturates some tasks, that is itself a task-dependence finding (H3).
2. **Industrially relevant.** Qwen3-Next uses GDN at 3:1 [config-inspected]. Kimi Linear uses KDA, a GDN refinement, at 3:1 [read-section]. It is the mechanism readers will map our results onto.
3. **Exact parameter matching to MHA** with no MLP or layer-count compensation: 4.03·d² vs 4·d² (M).
4. **Working, verified kernels on our hardware (M).** FLA chunk forward/backward and the fused recurrent decode kernel run on sm_120 with the installed torch/triton, and match a naive fp32 recurrence. No CUDA compilation is needed (pure Triton).
5. **Bounded, interpretable state.** The delta rule with L2-normalised keys is a contraction (§1.6). The state is H·d_k·d_v fp32, which gives a clean, analytically computable "inference-state memory" axis.
6. **Positional self-sufficiency.** Decay + short conv give GDN implicit position, so hybrids can run attention NoPE or RoPE (see §5.4).

Known weaknesses to monitor (I):
- GDN training is ~1.4× slower than GLA per layer at 2K (M).
- Pure GDN S-NIAH-2/3 collapse at 8K/4K (29.6, 27.6) (R), which leaves headroom for attention.
- The standalone ranking does not predict the hybrid ranking: HGRN2 beat GDN at 6:1 in #1.

### 4.2 Secondary mechanism 1: GLA (FLA)

- **Why:**
  - Same chunkwise kernel family and nearly identical projection geometry (d_k = d/2, d_v = d, Swish output gate, 4.05–4.06·d² params) (M).
  - It lacks the delta rule and uses a channel-wise sigmoid gate.
  - Its standalone recall is far weaker (pure 0.144 vs 0.356 RULER) and it gains most from hybridisation (0.144 → 0.332 at 3:1) (#1).
- **What it tests:** whether the curve shape (where the "knee" sits, H1/H2) depends on the cheap layer's own recall ability (I).
- **Configuration:** H = 4, d_k = 64, d_v = 128, `use_short_conv=True` (k = 4) so the short-conv policy matches GDN, `gate_logit_normalizer=16`, `gate_low_rank_dim=16` (FLA defaults [code-inspected]).
- It is also the fastest mixer measured (5.2–5.9 Mtok/s per layer, M).

### 4.3 Secondary mechanism 2: SWA with w = 64 (state-matched)

- **Why:** a "cheap softmax" control from a different family.
  - With MHA at d = 512 its bounded KV (2·512·64·2 B = 128 KiB) **equals** the GDN state, so it is a pure *inductive-bias* comparison at matched inference memory (I).
  - It is also the cheap layer in GDN-H1, Samba, Griffin, Jet-Nemotron and RNoPE-SWA. MiniMax-01 and 2606.15378 found SWA hybrids behave very differently from linear hybrids (NIAH 46.8–53.9 vs 95.7 at 1B; "Large-Window Laziness").
- **Implementation:** SDPA with a sliding-window mask, or FlexAttention, at the same width/heads as full attention. Params are identical, 4·d².
- **Optional:** w = 512 as a second point if budget allows.

### 4.4 Not recommended as primary (with reasons)

- **Mamba-2.**
  - Default 6.4–6.7·d² params. Matching needs expand ≈ 1.25, a non-standard config that makes results less comparable to the literature.
  - The fast path needs `mamba_ssm` + `causal_conv1d` CUDA extensions; FLA fell back to naive/Triton paths with warnings in our venv (M), and there is no system nvcc.
  - Its scalar decay is a special case of GDN without the delta rule (I).
  - Well covered by #2, #5 and #6 already.
- **RWKV-7.** Low training throughput (Jet-Nemotron) and 4 extra low-rank projections that complicate matching.
- **RetNet.** Fixed decay, weakest in #1 and #10, 8·d² default.
- **HGRN2.** A credible *tertiary* option because it was the best hybrid partner in #1 and has exactly 4.00·d². Choose it over GLA if Stage 2 prefers "best hybrid partner" over "same-family ablation" (I).
- **KDA.** GDN with channel-wise gates. Too close to GDN to add generalisation value at our budget.

---

## 5. Definition of a "layer" and the replacement scheme

### 5.1 Layer definition (I)

A layer ℓ is a pre-norm residual block with two sublayers:

```
h  = x + Mixer_ℓ(RMSNorm_1(x))
x' = h + SwiGLU(RMSNorm_2(h))
```

- `Mixer_ℓ ∈ {FullAttn, Linear}` is the **only** thing that varies.
- RMSNorms, SwiGLU width, residual path, embedding, LM head, init of shared parts, optimizer and LR schedule are identical for all layers and all models.
- A "full-attention fraction" r = k/L, where k is the number of layers with Mixer = FullAttn.

### 5.2 Mixer specs (param-matched; d = 512 shown, generalises with d)

**FullAttn (MHA)**
- q, k, v, o: d→d with no bias; H_a = d/64 heads with head_dim 64 (8 heads at d = 512).
- Causal SDPA (flash backend), positional policy per §5.4, no QK-norm, no output gate.
- Params 4·d².
- Rationale for MHA rather than GQA: it is standard at this scale, and the GQA KV saving is an orthogonal efficiency lever. We report GQA counterfactual memory analytically (KV scales by H_kv/H).

**Linear (GDN)**
- FLA `GatedDeltaNet(hidden_size=d, head_dim=64, num_heads=d//128, expand_v=2, use_gate=True, use_short_conv=True, conv_size=4, mode='chunk')`.
- That is H = 4 heads, d_k = 64, d_v = 128 at d = 512. Params 4.03·d².
- Decode uses `fused_recurrent_gated_delta_rule` with **keyword arguments only**.

**Linear (GLA)**
- FLA `GatedLinearAttention(hidden_size=d, num_heads=d//128, expand_k=0.5, expand_v=1.0, use_short_conv=True, conv_size=4)`.
- That is H = 4, d_k = 64, d_v = 128. Params 4.06·d².

**SWA**
- Same as FullAttn with window w = 64 (state-matched).

### 5.3 Short conv, gates, norms: policy (I)

| Element | Present in | Policy (primary) | Control |
|---|---|---|---|
| Short causal depthwise conv (k = 4) on q, k, v | GDN / GLA | Keep: it is part of the published mechanism (FLA warns it is "crucial"; Kimi ablation: no conv val ppl 5.70 vs 5.65). Params ≈ 4·(2·d/2 + d) = 8·d per layer (≈ 4K at d = 512, 0.4 % of mixer) | **One control run:** all-full model with the same short conv on q, k, v ("Attn+conv") at r = 1, and optionally at r = 3/12, to measure how much of any hybrid gain is the conv |
| Output gate | GDN / GLA (inside 4·d² budget) | Keep | Attention has no gate. Gated attention reduces massive activations (#28), so note this as a confound and do not add a gate to attention in the primary arm |
| Per-head output RMSNorm | GDN / GLA | Keep | – |
| L2 norm on q, k | GDN | Keep (required for stability) | – |
| Decay init (A_log ~ log U(0, 16), dt ∈ [1e-3, 0.1]) | GDN | FLA defaults; no weight decay on A_log/dt_bias | – |
| MLP | all layers | identical SwiGLU | – |

### 5.4 Positional policy (I; pre-register one primary, one ablation)

- Linear layers: **no RoPE** in all arms (as in all published systems).
- **Primary arm P-RoPE.** All full-attention layers use RoPE at every r (base θ chosen once for the training length; e.g., θ = 1e5 at 8K–16K training, following 2606.15378's 1e5 at 16K).
  - The all-full endpoint is then a standard RoPE Transformer, which reviewers expect.
  - Extrapolation beyond the training length is reported separately (§6.2).
- **Ablation arm P-NoPE.** NoPE on full layers for r < 1, at a subset of r (e.g., k ∈ {1, 3} of 12), plus an all-full NoPE run so the endpoint is also measured.
  - Motivation: Jamba, Waleffe, Kimi Linear, 2501.18795 and 2606.15378 all show NoPE-on-full helps long-context retrieval in hybrids.
- FLA's hybrid `attn` config cannot express NoPE (`rope_theta` must be > 0) [code-inspected]. Our own attention module handles both.

### 5.5 FLOPs: what is and isn't matched (I)

- Params per layer are matched within 1 %. **FLOPs cannot be matched while varying r**, because attention FLOPs grow with T and that difference *is* the manipulated variable.
- Forward FLOPs per token per layer at d = 512:

| Component | FLOPs/token/layer (fwd) |
|---|---|
| Projections + SwiGLU (≈ 2·params) | ≈ 6.4 M |
| Attention scores at T = 8K (2·d·T, causal average) | +8.4 M |
| Attention scores at T = 32K | +33.6 M |
| GDN chunk recurrence (H = 4, d_k = 64, d_v = 128, C = 64; rough estimate) | +0.2–0.3 M |

- An all-full L = 12 model therefore costs ≈ 2.2× the all-GDN model's training FLOPs per token at 8K.
- **Recommend matching training tokens (primary) and reporting training FLOPs and wall-clock per model.** Add an iso-FLOP check at one or two points only if Stage 2 judges it necessary.

---

## 6. Proposed configuration space

All sizes below assume tied embeddings and a 32K vocabulary, used only for illustration. The tokenizer is owned by another agent, so embedding params will change with its choice.

### 6.1 Model families

**F12 (PRIMARY) — L = 12, d = 512**
- Attention: 8 × 64. GDN: 4 heads, d_k 64, d_v 128.
- SwiGLU hidden 1408 (≈ 2.75·d, multiple of 64).
- Per layer ≈ 1.05 M (mixer) + 2.16 M (MLP) = **3.21 M**.
- Non-embedding ≈ **38.5 M**; total ≈ **54.9 M** with 32K tied vocab.
- Realisable r = k/12, k ∈ {0, 1, 2, 3, 4, 6, 12}, i.e. linear:full = ∞ (pure), 11:1, 5:1, 3:1, 2:1, 1:1, 0.

**F24 (count-vs-fraction check) — L = 24, d = 384**
- Attention: 6 × 64. GDN: 3 heads (64/128). SwiGLU 1024.
- Per layer ≈ 0.59 M + 1.18 M = 1.77 M. Non-embedding ≈ 42.5 M; total ≈ **54.8 M**, matched to F12.
- k ∈ {1, 2, 3, 6} gives r = 4.2 %, 8.3 %, 12.5 %, 25 %. These match ratios used in the literature: 23:1, 11:1, 7:1 (MiniMax/Jamba) and 3:1 (Qwen/Kimi).
- Run F24 at k ∈ {2, 3, 6} only.
  - k = 2 matches F12 k = 1 on fraction (8.3 %) but doubles the count of full layers.
  - k = 6 matches F12 k = 3 on fraction (25 %) at double the count.

**F12-S (size check) — L = 12, d = 384**
- Same geometry as F24 per layer; ≈ 21.2 M non-embedding, ≈ 33.5 M total.
- Run at k ∈ {0, 3, 12} to test whether the curve shape is stable across size.

- **d = 768 is out of scope** (≈ 85 M+ non-embedding at L = 12).
- **L = 8 is not recommended:** it only realises k ∈ {1, 2, 4} and the minimum r is 12.5 %, so it misses the low-r region where H1 lives.

### 6.2 Placement patterns (L = 12, 0-indexed)

**Primary rule ("centered-uniform"):** p_j = ⌊(j + ½)·L/k⌋, j = 0 … k−1.

| k | positions |
|---|---|
| 1 | {6} |
| 2 | {3, 9} |
| 3 | {2, 6, 10} |
| 4 | {1, 4, 7, 10} |
| 6 | {1, 3, 5, 7, 9, 11} |

- Layer 0 is always linear for k < 12. This follows Waleffe's "first layer Mamba-2" and Bae et al.'s "front is worst" (R).
- The rule is stated as a formula so it extends to F24.
- Alternative rule (Qwen3-Next style "end of group", `(i+1) % (L/k) == 0`): k = 3 → {3, 7, 11}. Include it as one cell of the placement study, not as primary.

**Placement study (H4)** at fixed k = 2 (r = 16.7 %, inside the low-r regime where placement should matter most):

| Pattern | Positions |
|---|---|
| Early | {0, 1} |
| Middle | {5, 6} |
| Late | {10, 11} |
| Spread / centered (primary) | {3, 9} |
| End-of-group | {5, 11} |
| Sandwich | {0, 11} |

- That is 6 cells; the centered cell is shared with the ratio sweep.
- If budget allows, add k = 1 at positions {0}, {6} and {11}. Middle vs end for a single layer is the most practically relevant question for "how little attention" (I).

### 6.3 Mechanism arms

| Arm | k values (F12) | Notes |
|---|---|---|
| GDN | all of {0, 1, 2, 3, 4, 6, 12} | primary |
| GLA | {0, 1, 3, 6} | reuses the shared k = 12 endpoint |
| SWA-64 | {0, 1, 3} | reuses k = 12. Note that k = 0 is an all-SWA model, i.e. a local-only Transformer |
| Attn+conv control | k = 12 (and optionally 3) | §5.3 |
| P-NoPE ablation | GDN k ∈ {1, 3, 12} | §5.4 |

- Seeds: at least 2 for the primary GDN sweep. Seed variance at 50M scale can be comparable to placement effects; Bae et al.'s placement differences were ≤ 0.004 nats (I).

### 6.4 Training context vs evaluation length (I; for Stage 2 to decide)

- All-full RoPE models fail beyond their training length (#2, #6, #12). Evaluating at 32K after training at 4K therefore measures extrapolation, not capability. That would confound H5 and bias the curve toward hybrids.
- **Recommendation:**
  - (a) Pretrain all models at T = 4K or 8K.
  - (b) Then run an **identical long-context continuation** for every model, e.g. ~5–10 % of tokens at 32K. This mirrors 2606.15378, which used ~5B tokens at 32K for its S5 model, with a RoPE θ increase applied identically.
  - (c) Report within-training-length results as primary and >T results as an "extrapolation" axis.
- Measured attention fwd+bwd at 32K is ≈ 39 ms per 32K tokens per layer vs ≈ 10 ms for GDN (M, shared GPU). An all-full 12-layer 32K continuation is therefore feasible (≈ 0.5 s mixing per 32K tokens).

### 6.5 Inference-state memory (analytic; F12, bf16 KV, fp32 GDN state)

- Per sequence: M(r, T) = k · 2·d·T·2 B + (12 − k) · 134 KiB.
- At T = 32K: each full layer costs 64 MiB, so k = 12 → 768 MiB, k = 3 → 192 MiB + 1.2 MiB, k = 0 → 1.6 MiB.
- Memory is linear in k at fixed T, so the memory axis of the frontier is essentially "k × T". The capability axis is what must be measured (I).
- The GDN state stays fp32 here because the FLA kernels keep the state in fp32 (M). MiniMax warns linear attention is precision-sensitive (#9).

---

## 7. Confounds introduced by the mechanism (and mitigations)

| # | Confound | Why it matters | Mitigation (I) |
|---|---|---|---|
| C1 | **Params matched, FLOPs not** | Full attention gets more compute per token at long T | Match tokens; report FLOPs and wall-clock; optional iso-FLOP spot check |
| C2 | **Short conv only in linear layers** | Conv supplies local n-gram / induction-friendly features (Kimi: no-conv ppl 5.70 vs 5.65); hybrid gains may partly be "conv gains" | "Attn+conv" control at r = 1 (and r = 3/12) |
| C3 | **Output gate only in linear layers** | Gates reduce massive activations / attention sinks (#28), which affects stability and retrieval | Document; optional gated-attention control if C2 shows large effects |
| C4 | **Positional asymmetry** | Linear layers have implicit position; attention needs RoPE or NoPE; NoPE-on-full has large documented effects (§2) | Pre-registered P-RoPE primary + P-NoPE ablation |
| C5 | **Extrapolation vs capability** | RoPE all-full fails beyond T_train | Long-context continuation for all models; separate extrapolation axis (§6.4) |
| C6 | **Head geometry / state capacity** | GDN H = 4 with 64×128 state per head; recall capacity of the delta rule is bounded by ≈ d_k orthogonal keys per head (≈ 256 total per layer at d = 512) (I, linear algebra). MQAR with more KV pairs than this is beyond exact capacity for a single linear layer | Report d_k, H; design synthetic task difficulty around this bound; consider one key-dim variant (d_k = 128, d_v = 256, H = 2: params 4.02·d² (M), state 2·128·256 = 65,536 fp32 = 256 KiB, i.e. 2× the elements, twice the key capacity per head) |
| C7 | **Optimiser / LR sensitivity differs by mixer** | GDN/Mamba-style A_log/dt params; no weight decay on them; a single LR may favour one endpoint | Short LR sweep (3 values) at r = 0 and r = 1 before the main sweep; use the LR that is best for both or the geometric mean; report |
| C8 | **Count vs fraction** | k and r are confounded at fixed L | F24 vs F12 at matched r (§6.1) |
| C9 | **Placement × ratio** | Changing k changes which depths are attention | Fixed formulaic placement rule for the ratio sweep; separate placement study at fixed k |
| C10 | **Standalone ≠ hybrid strength** | #1 shows HGRN2 > GDN in hybrids despite weaker standalone recall | Secondary mechanism (GLA) to test curve-shape generality |
| C11 | **Kernel numerics** | bf16 chunk kernel error ≈ 1e-3 vs fp32 naive (M); state in fp32 | Keep the fp32 state; measure the decode/prefill consistency test (chunk vs recurrent) in CI |
| C12 | **Kernel efficiency on sm_120** | Triton autotuning on consumer Blackwell may be sub-optimal, so timing reflects implementation, not mechanism | Report memory analytically and time separately; benchmark on an **idle** GPU (CARLA stopped); use CUDA graphs for decode at small batch |
| C13 | **SWA w = 64 state-matching assumes MHA bf16** | Changing KV precision or GQA changes the match | State the matching rule explicitly |
| C14 | **Benchmarks hide gaps** (MiniMax) | Short-task parity can mask multi-hop deficits | Per-task reporting (retrieval / multi-hop / aggregation), not RULER averages |

---

## 8. Implementation notes verified on this machine (M)

- **Environment:** torch 2.11.0+cu128, triton 3.6.0, fla 0.6.0, GPU capability (12, 0).
- **GDN kernels:**
  - `fla.ops.gated_delta_rule.chunk_gated_delta_rule(q, k, v, g, beta, output_final_state=True)` runs forward and backward on sm_120.
  - The final state is fp32 with shape [B, H, d_k, d_v].
  - Against a naive fp32 recurrence (T = 512, H = 4, d_k = 64, d_v = 128): chunk max abs error 2.9e-4 in fp32 and 1.2e-3 in bf16; recurrent kernel 6e-8 in fp32 and 4.6e-4 in bf16. The output abs-max is ≈ 0.17.
- **Decode-kernel gotcha:** `fused_recurrent_gated_delta_rule` positional order is `(q, k, v, g, gk, gv, beta, A_log, dt_bias, scale, initial_state, output_final_state, …)`. Passing beta positionally after g puts it in `gk` and gives NaN. **Always use keywords.**
- **GLA:** `chunk_gla` runs.
- **Mamba-2:** the FLA `Mamba2` layer runs, but on naive/Triton fallbacks with warnings.
- **Attention:** SDPA FLASH, EFFICIENT and CUDNN backends all run on sm_120. `flash_attn` is not installed, and FLA `fla.layers.attn.Attention` raises ImportError without it, so we must use our own SDPA attention (it also gives us NoPE and window control).
- **Indicative single-layer fwd+bwd throughput** (d = 512, bf16, 32K tokens per step, shared GPU):

| T | Attention | GDN | GLA |
|---|---|---|---|
| 2K | 5.76 Mtok/s | 4.10 Mtok/s | 5.92 Mtok/s |
| 8K | 2.61 Mtok/s | 4.38 Mtok/s | 5.91 Mtok/s |
| 32K | 0.84 Mtok/s | 3.27 Mtok/s | 5.16 Mtok/s |

- **Param counts** per mixer at d ∈ {384, 512, 640, 768} were produced by instantiation (§1.14). The script is in the session scratchpad (`params.py`).

---

## 9. Gaps, not-accessed items, and open questions

**Not accessed or not verified:**
- The official Qwen3-Next blog failed to render (qwen.ai returned only "Qwen"), so its claims about GDN vs SWA/Mamba2 are not verified. Only the HF model card and config were read.
- Read at abstract level only: Ring-linear ratio details, FlashMorph numbers, Massive-activations placement numbers, Zamba2, Falcon-H1, RecurrentGemma window/tokens.
- The exact RWKV-7 and HGRN2 update equations were not re-read.
- The Waleffe attention-fraction sweep values are only in a figure; I have the text "≈ 8 %".
- The Bae et al. NIAH numbers are only in figures.

**Inconsistencies noted in sources:**
- 2507.06457 text says 24:1 → 3:1 "nearly doubles RULER recall", but Table 7 shows 0.338 → 0.397.
- Bae et al. quote a 256 MiB Transformer cache, which does not match their stated 8-KV-head GQA config (≈ 64 MiB).
- Jet-Nemotron's paper uses 1-indexed layers (15, 20) while the config uses 0-indexed (14, 19).

**Open questions for Stage 2:**
- P-RoPE vs P-NoPE as primary.
- Training length (4K vs 8K) and the size of the long-context continuation.
- GLA vs HGRN2 as secondary 1.
- Whether to run the iso-FLOP check.
- Seed count under the budget.

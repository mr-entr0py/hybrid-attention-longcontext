# Methodology, step by step

This is a condensed, ordered guide to how the study is run. The authoritative text is `reports/experimental_protocol.md` (frozen v1.0) together with the logged deviations in `reports/decisions.md` (DEV-01…04) and `reports/pi_rulings_2026-11-10.md`. Where they differ, those documents win.

## Principles
1. **Research before implementation.** Every design choice traces back to a source, a measurement or an explicit constraint. These are recorded as decision rows with the alternatives that were rejected and the conditions under which they would be reopened.
2. **Controlled comparison first.** Only the token mixer of each layer changes between configurations. Everything else is identical: parameter counts within 0.14%, data, data order, initialisation of shared modules, schedule and synthetic exposure.
3. **Pre-registration.** Hypotheses, tests, thresholds, gates and the analysis code are fixed before results exist. Pre-registration files are hashed (`experiments/prereg/SHA256SUMS*`).
4. **Reproduce before novelty.** Published results are reproduced first with the original code, then with ours, swapping one component at a time.
5. **Independent verification.** Whoever builds a component never verifies it. Adversarial reviewers try to refute.
6. **Integrity.** A criterion is never restated after its test has run. A failure is reported as "NOT PASSED as registered", together with a named residual risk or a new pre-registered check. Negative results are reported.

## Step 1: Define the experimental object
- **Architecture vector.** a ∈ {F, Λ}¹², with k = number of F layers and r = k/12.
- **Block.** h ← h + Mix(RMSNorm(h)); then h ← h + SwiGLU(RMSNorm(h)). Embeddings are tied and there are no biases.
- **F (full attention).** MHA with 64-dim heads, per-head QK-RMSNorm, RoPE θ = 5·10⁵ with positions reset per document, a headwise sigmoid output gate, and causal attention within each document.
- **Λ (linear).** Gated DeltaNet:

  S_t = S_{t−1}·α_t(I − β_t k_t k_tᵀ) + β_t v_t k_tᵀ

  with an output gate and a short convolution (width 4). Its state is fp32.
- **Scales.** D384 (d = 384, d_ff = 1,024) or D512 (d = 512, d_ff = 1,536); vocabulary 16,384. A pilot rule picks one.
- **Patterns.** Full layers are spaced evenly: p_j = ⌊(j + ½)·12/k⌋. Layer 0 is linear whenever k < 12.

## Step 2: Data
- **Training data.** FineWeb-Edu sample-10BT shards 001–006, plus PG19 train, both pinned to exact revisions.
- **Held-out data.** FineWeb-Edu shard 000, split by the parity of SHA-256(doc_id); PG19 validation and test.
  - Held-out documents that exactly duplicate training text are removed.
- **Tokenizer.** Byte-level BPE with 16,384 entries and single-digit isolation, deterministic.
- **Packing.** Documents are separated by EOS. Masking is per document in both mixers (`cu_seqlens`), and positions reset at each document. Batches are a pure function of (seed, stage, step).

## Step 3: Training recipe (identical for every k)

| Phase | Length | Tokens | Mix | LR |
|---|---|---|---|---|
| A | 4K | 0.6B (D384) / 0.8B (D512) | 85% FineWeb-Edu packed + 15% PG19 windows | 500-step warm-up, then constant η* |
| B | 32K | 30–300M (set by a pilot rule) | 60% PG19 32K windows + 40% FineWeb-Edu | Linear decay to 0 |
| TEF | 32K packed | 65.5M (1,000 steps); seed-0 anchors run to 4,000 | 90% synthetic tasks (7 length buckets, equal per family) + 10% replay | Re-warm to 3e-4, then constant |

- **Optimiser.** AdamW with β = (0.9, 0.95) and weight decay 0.1 on matrices only. Gradient clip 1.0, z-loss 1e-4, bf16 autocast with fp32 master weights.
- **LR selection (S3).** Grid {1.0, 1.4, 2.0, 2.8}×10⁻³ × k ∈ {0, 3, 12}, scored by minimax regret on held-out LM loss, never on task metrics.
  - A single shared η* is used if the worst regret is ≤ 0.02 nats; otherwise η is set per configuration.

## Step 4: Evaluation
- **Item construction.** Every item is exactly T tokens, built in token space and frozen with SHA-256. Contexts contain no document separators, and filler comes only from held-out documents.
- **Templates.** "The code for {key} is {4-digit value}." Keys are adjective–noun pairs. Training and evaluation use disjoint word lists and templates.
- **Confirmatory suite (CS):**
  - R1: facts with distractors, D ∈ {3, 15}.
  - R-load: N ∈ {8 … 1,024} facts.
  - M-hop: shuffled chains, h ≤ 4.
  - A-freq: ρ ∈ {4 … 1.25}.
  - Lengths 1K–32K, 200–300 contexts × 4 queries per cell.
- **Diagnostics (MS):** distance vs filler (natural vs repetitive), updates and recency, oracle position, counting, link order, an unseen template, attention mass, and ablations of each attention layer.
- **Zero-shot (Tier 0):** per-position loss on long documents, repeat-token loss by distance, copy-after-gap, and a passkey test.
- **Metrics.** EM (teacher-forced exact match) is primary. Gold log-probability, rank accuracy and error types are secondary.
- **Scoring path.** One prefill per context; the inference state is snapshotted; queries are scored from the snapshot. Kernels are pinned explicitly, and 5% of items are re-scored with a single-pass forward.

## Step 5: Cost measurement
- **Primary (analytic).**
  - State bytes per sequence: k·T·4d (bf16 KV cache) + (12 − k)·S_lin (fp32 GDN state).
  - FLOPs, bytes moved per decode step, and a roofline lower bound.
- **Secondary (measured).**
  - Prefill at B ∈ {1, 8} and decode at B ∈ {1, 8, B_max}, for T and L from 1K to 32K.
  - Exact-shape CUDA graphs with backends pinned.
  - Cold and hot L2, at least 30 replays, min-of-N reported.
  - Two sessions on different days, on an idle GPU only.

## Step 6: Statistics (pre-registered)
- The unit of inference is the **training seed**. Seeds pair across k, because data order and shared initialisation are common.
- Family score S_f(k, T) per seed. Gap G = S₁₂ − S₀. Recovery R(k) = (S_k − S₀)/G.
- **H1:** L₃ = S₃ − 0.2·S₀ − 0.8·S₁₂ > 0, which is equivalent to R(3) ≥ 0.8. Paired one-sided t-test, Holm over {H1, H1-min, H3, H5a} at α = 0.05.
- **Primary cell.** The 16K/32K retrieval ladder mean, restricted to lengths that pass the reference-adequacy gate (G-REF). The fallback is 8K, labelled "not long-context".
- **Sensitivity analyses.** A seed-blocked pooled-variance model and a hierarchical paired bootstrap (10,000 resamples). If either disagrees in sign, the verdict becomes "suggestive".
- **Seed trigger.** If residual SD > 2.5 pp, or any configuration's seed range > 12 pp, add 2 seeds to every anchor. This rule never looks at the effect.
- **H1 falsification** requires all four guards: gates passed, a significant negative test, a dose guard (TEF-4000) and a saturation guard (doubling Stage B).

## Step 7: Staged execution and gates

| Stage | Purpose | Gate |
|---|---|---|
| S0 | Infrastructure plus verification tests VT-01…25 | G0 |
| S1/R1–R10 | Reproduce published results (Zoology MQAR, LAMBADA anchors, transplants, full-pipeline recall) | G-RP, G-RV |
| S2 FEAS | Pilots: choose scale, Stage-B budget, SSMax, TEF mix, contexts per cell | FEAS rules (DTS-10) |
| S3 | LR sweep | G-LR |
| G5 | Freeze items, configs, analysis code, Triton cache and lock file | G5 |
| S4 | Core runs: k ∈ {0, 1, 3, 12} × 4–5 seeds | G-FULL, G-REF, seed trigger |
| S5 | Controls: conv attention, NoPE, longer TEF, doubled Stage B | — |
| S6 | Frozen evaluation and cost benchmarks | Tier-A decision |
| S7/S8 | Tier B (k = 2 and 6, placement, GDN-E) and Tier C | — |

## Step 8: Reporting
- Each plot (P1–P18) answers one question and is regenerated by a script from raw results. Every seed is plotted, and CI types are stated.
- Claims are worded as packages ("k gated-attention layers of 12, the rest GDN-A, as built"). Operator-level wording is allowed only if the convolution control permits it.
- The report separates observations, findings, interpretations, hypotheses and limitations.

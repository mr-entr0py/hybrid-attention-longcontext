# Reproduction suite proposal: pipeline validation

**Role:** reproduction-programme designer, from the pipeline-validation angle. Every reproduction below is chosen because it checks one specific component or assumption of *our* pipeline.
**Date:** 2026-11-09. **Status:** proposal only. Nothing here has been run by me, and nothing in `src/`, `tests/`, `scripts/`, `configs/` or `data/` was touched.
**Relation to the frozen protocol (v1.0).**
- This suite *adds* checks around S1 (§17, gate G-RP). It changes no design choice.
- Each adopted item should get a `decisions.md` row (pre-G5, so this is not a deviation) that freezes its pass criterion **before** it runs.
- One recommendation, R-S1-2 in §2, concerns the S1 run that is in progress now. It should be logged *before* the remaining S1(a) results are read.

**Tags.**
- Access: [read-full], [read-section], [abstract-only], [code-inspected]. "per LR" means the number comes from `reports/literature/literature_review.md` and I did not re-check it.
- [MEAS-S1 interim] means read by me from `experiments/s1_zoology/runs/*.json` at about 18:00 on 2026-11-09, while the S1 queue was still running. These are interim, single-seed results.
- [INF] means my inference or arithmetic.
- **Figure-read numbers are marked "(fig, approx.)" and are never used as pass thresholds** (protocol I-20).

---

## 0. Summary

**Must tier: 7 items, ≈ 13.8 GPU-h** (≈ 5.8 h of this is S1, already in the protocol).

| # | Reproduction | Component it validates | New GPU-h |
|---|---|---|---|
| M1 | Zoology MQAR (protocol S1, in progress) + **2 fixes** | GDN/attention mixer recall, harness | (S1 ≈ 5.8 realistic vs 3.0 budgeted) |
| M2 | **Trained-weight transplant + LAMBADA**: Pythia-70M/160M, Wang et al. 340M GDN-pure, GDN-hybrid and Transformer checkpoints; lm-eval (original) vs our scorer | GDN and attention mixers *with trained weights* (RoPE θ, GQA→MHA, conv, gates), EM_tf/logp scorer, adapter | ≈ 1.0 |
| M3 | **Our Tier-0 harness on Wang et al.'s public 340M GDN ratio sweep** (pure, 1, 2, 4, 8 of 24 attention layers, plus Transformer) | LM-1/LM-3/LM-4/R0 code; whether Tier-0 metrics are k-sensitive at all; the E9.1 validity pattern | ≈ 0.5 |
| M4 | **nanoGPT shakespeare-char ladder** (original → our trainer with nanoGPT's model → our model) | Trainer: optimiser groups, schedule, accumulation, clipping, loss averaging, bf16, train/eval mode | ≈ 1.5 |
| M5 | **Hybrid-ratio LM-loss pattern** (Mamba-2 Table 2; Waleffe "≈ 8%") from the S3 LR sweep, plus k = 1 and 6 and one extra seed | Whole LM pipeline across k (RP-2 made quantitative); LR fairness | ≈ 1.4 (D384) / 1.9 (D512) |
| M6 | **Olsson induction / smeared-key test**: 1-layer vs 2-layer attention, 1-layer F_gc, and `-A` | Attention mixer + causal mask + RoPE (induction must form); F_gc conv; per-position loss code; the premise of rival E1.6 | ≈ 1.9 |
| M7 | **Discriminating MQAR** (S1-ext): GDN at d = 64 where accuracy is mid-range, Zoology vs ours × 3 seeds | "Our GDN = Zoology's GDN" in a regime where a bug would show | ≈ 1.7 |

**Should tier:** 4 items, ≈ 3.5–4.5 GPU-h (S1–S4 in §4). **Nice tier:** 4 items, ≈ 8–12 GPU-h (N1–N4).

**Three findings from inspection that need action now** (§2):
1. **The S1 cut rule under-priced S1(a) by about 75×.** It extrapolated from the first run, attention d64 at 16.5 s. GDN d128 runs take ≈ 993 s, and GDN d256 ≈ 122 s per epoch (≈ 65 min at 32 epochs). Realistic S1(a) is ≈ 5.5 GPU-h against 3.0 budgeted [MEAS-S1 interim; INF].
2. **G-RP is close to ceiling.** Attention d64 reaches overall accuracy 0.99998 at epoch 0 and early-stops at every LR. GDN d128 ends at 0.989 overall. Only the 1024/256 test cell discriminates (GDN d128 0.925; d256 0.911 after one smoke epoch). G-RP(b) (±0.03 on *overall* accuracy) therefore cannot detect a mixer bug that only shows at high load [MEAS-S1 interim].
3. **Wang et al.'s released "a-1" ratios are not a:1 linear:full.** In the HF configs, "3-1" has 8 of 24 layers as attention (r = 1/3, i.e. 2:1), "6-1" 4/24, "12-1" 2/24, "24-1" 1/24. Attention head count also changes across ratios (16 heads for 3-1, 32 for the others; GQA with 8 KV heads; RoPE θ = 1e4). The protocol's prior line ("Wang GDN recall … 24:1 / 12:1 / 6:1 / 3:1") should be read as k/24 ∈ {1, 2, 4, 8}, not r = ¼ [code-inspected, §M2 sources]. The paper text says "a fixed ratio r:1 (linear : full)" with r ∈ {24, 12, 6, 3} (2507.06457 App. D [read-section]). The released configs disagree with that text.

---

## 1. Principles that shaped the suite (proposed research practices)

These are proposed as standing rules for every reproduction in this project. P1–P6 are the core.

- **P1. Two runs per reproduction: original code, then our code, in the same setting.**
  - The original-code run answers "does our environment (GPU, torch 2.11, triton 3.6, fla 0.6.0, data) reproduce the published number?"
  - The our-code run answers "does our implementation match the original?"
  - Without both, a mismatch cannot be attributed. M1, M2, M4 and M7 are built this way.
- **P2. Swap one component at a time.** For example: original → original with our schedule → our trainer with the original model → our model. Each rung has its own pass criterion, so a failure points to one component.
- **P3. Pre-register targets, tolerances and failure actions.**
  - Write them in `decisions.md` before the run, and derive each tolerance from reported seed variance or reported SE.
  - Where the source gives no variance, measure it with ≥ 3 seeds of the *original* code first and set the tolerance from that (stated as such).
  - Figure-read numbers are never thresholds (I-20). They may only be qualitative/directional checks.
- **P4. Prefer reproductions whose failure is diagnostic, not only binary.** Prefer settings where a bug would move the result (mid-range accuracy, not ceiling). Ceiling or floor results validate "it runs", not "it matches". That is the lesson of finding 2.
- **P5. Never tune our code to hit a number without a mechanistic reason.**
  - A fix is allowed only if (i) a unit test or logit-level comparison isolates a defect, and (ii) the fix is justified independently of the target number.
  - Every change made after a failed reproduction is logged with the failing number, the diagnosis and the commit.
- **P6. Report failed and partial reproductions in the final report.** Report the number, the tolerance, the diagnosis and what was concluded. A reproduction that "passes" only after changes is reported with its whole history.
- **P7. Pin everything.** Record the repo commit, the HF model `sha`, the lm-eval commit (local clone `d6de816`), library versions, and every deviation from the original command line (a logger shim, an fla API shim, a missing CUDA extension). Treat every shim as a suspect until shown to be identity-preserving: for example, Zoology logs "Need to install causal_conv1d", so the fallback conv path must be confirmed equivalent ([MEAS-S1 interim] log line).
- **P8. Separate "pattern replication" from "number replication".** Our scale (≈ 28–49M, ≤ 1B tokens) cannot reproduce 130M–1.3B numbers. Those items are labelled *pattern* checks, their criteria are directional with pre-stated magnitudes, and a failure is evidence about our setting, not automatically a bug.
- **P9. Use published checkpoints wherever possible.**
  - Evaluation-only reproductions cost minutes and test evaluation code and mixer code against real trained weights.
  - Random-weight tests (VT-09) do not exercise trained regimes: large RoPE angles that actually matter, saturated gates, peaked attention.
- **P10. Positive and negative controls in every harness.**
  - Every metric we rely on must be shown to move when it should (positive control: a Transformer on random-token copying) and not move when it should not (negative control: shuffled context, or a 1-layer attention-only model without induction).

---

## 2. Immediate recommendations on S1 (protocol §17), which is running now

| ID | Observation | Recommendation |
|---|---|---|
| R-S1-1 | `cut_decision.json`: `first_run_wall_s` = 16.5, implied total = 0.073 GPU-h, so no cut. GDN d128 lr 1e-3 took 993.5 s (32 epochs, no early stop). The d256 smoke epoch took 121.8 s, i.e. ≈ 65 min for 32 epochs [MEAS-S1 interim] | Log a DEV/decision row now: re-evaluate the pre-declared cut on the **GDN** first-run time. That gives ≈ 4 × 16.5 min + 4 × 65 min + 8 × 0.3 min ≈ 5.5 h > 3.0 h, so the rule (as written) cuts d256 to its 2 best LRs, chosen at d128. This saves ≈ 2.2 h. The rule's intent ("if the first timed run implies > 3.0 GPU-h") was defeated by using the cheapest family as the timer |
| R-S1-2 | Overall-accuracy ceiling. Attention: 0.99998 at epoch 0 at all 4 LRs, so best-over-LR is uninformative. GDN d128: 0.9892 overall; per-cell 1.0 / 0.9999 / 1.0 / 1.0 / 1.0 / 0.9998 / **0.925** (1024/256) [MEAS-S1 interim] | Before d256 finishes, add to the G-RP report: (i) the per-cell table, with the **1024/256 cell as the discriminating cell**; (ii) for G-RP(b), compare our vs Zoology accuracy *on that cell* (tolerance in M7) as well as on overall. G-RP(a)/(b) as frozen stay the formal gate; the per-cell comparison is reported alongside |
| R-S1-3 | The early-stopping threshold of 0.99 interacts with "GDN non-decreasing in d_model". GDN d128 hovers at 0.989; a run that crosses 0.99 stops early, so final accuracies near 0.99 differ only by noise | Pre-declare before d256 completes: "non-decreasing" is judged with a tolerance of −0.005 on overall accuracy *and* on the 1024/256 cell. Without a tolerance, a ±0.001 noise flip fails G-RP(a) spuriously |
| R-S1-4 | Zoology prints "Need to install causal_conv1d" (no wheel for torch 2.11) | One-off check: compare the fallback BaseConv/ShortConv output with a naive torch conv on random input (max abs diff ≤ 1e-5, fp32), and record it as a setup deviation from the authors' environment |
| R-S1-5 | G-RP(b)'s ±0.03 tolerance is not derived from seed variance (Zoology runs use one seed, 123) | M7 measures seed SD and sets the tolerance for the discriminating cell. ±0.03 on overall accuracy stays the frozen gate |

---

## 3. Must tier

### M1. Zoology MQAR: S1 as frozen, plus R-S1-1…5

- **Claim to reproduce.** Attention solves MQAR at small width. An input-dependent recurrent mixer (GDN) reaches high MQAR accuracy, and its accuracy grows with state size, i.e. d_model. Accuracy falls with the number of KV pairs and sequence length.
  - Zoology's README figure (`zoology/analysis/paper/zoology_paper/zoology_results.png`) and Arora et al. 2023 ([2312.04927](https://arxiv.org/abs/2312.04927)) show this only as figures (fig, approx.).
  - No tabulated target exists, so the targets are the frozen G-RP thresholds, which use re-run numbers only ([zoology @1ad20d1](https://github.com/HazyResearch/zoology) `experiments/mqar_example_configs/original_mqar_configs.py` [code-inspected]).
- **Validates.**
  - Our attention and GDN mixer wrappers (`build_mixer`) in a recall task.
  - fla 0.6.0 GDN kernels on sm_120 under training.
  - Rivals ruled out: E10.4 (implementation bugs); partly E1.3, because it calibrates GDN capacity against load.
- **Code.** Original: Zoology @1ad20d1 plus the `head_first` shim (s1_reproduction.md §3). Ours: `scripts/s1_zoology/our_mixers.py` → `build_mixer`.
- **Config:** protocol §17, unchanged.
- **Pass criterion:** G-RP (frozen), plus the reporting additions R-S1-2/3.
- **GPU-h:** ≈ 5.5 h for (a) if not cut, ≈ 3.3 h with R-S1-1; (b) ≈ 2.5 h [INF from the timings above]. **Budget line ≈ 5.8 h** (with the cut).
- **Depends on S0:** `models/mixers.build_mixer` (contract exists; implementation in progress).
- **Failure → diagnosis.**
  - Zoology-original fails (a): environment problem. Check the fla shim (`head_first`), the conv fallback (R-S1-4), and fp32 vs bf16. Run Zoology's GDN on fla's own `GatedDeltaNet` layer test inputs.
  - Original passes, ours fails (b): diff module by module. Feed identical weights and inputs to Zoology's mixer and to ours. The known differences are bias, dropout, `out_proj` init name, and `use_gate` (s1_reproduction.md §5). Remove them one at a time.

### M2. Trained-weight transplant + published LAMBADA numbers (evaluation-method sanity and mixer correctness)

**Why.** VT-09 checks our mixers against transformers' Qwen3 and fla's GDN *at random initialisation*. Trained weights exercise regimes random weights do not: peaked attention at long range, where a RoPE or position-reset bug shows; saturated gates; large conv activations. A published downstream number on public checkpoints also tests the *scorer*. LAMBADA's lm-eval `acc` is greedy-match of every target token given the gold prefix, which is the same quantity as our EM_tf (§9.2) [INF from the protocol's definition of EM_tf].

**Targets (published).**

| Checkpoint | Task | Published | Source |
|---|---|---|---|
| EleutherAI/pythia-70m (final) | LAMBADA (OpenAI) acc, 0-shot | **0.185 ± 0.005** | Pythia [2304.01373](https://arxiv.org/abs/2304.01373) App. G.1 Table 7 [read-section] |
| EleutherAI/pythia-160m (final) | same | **0.328 ± 0.007** | same |
| m-a-p/340M-20B-GatedDeltaNet-pure-baseline (sha `14661a9`) | LMB acc | **0.348** | Wang et al. [2507.06457](https://arxiv.org/abs/2507.06457) Table 2 (340M) [read-section] |
| m-a-p/transformer_340M_baseline (sha `f07ebb6`) | LMB acc | **0.353** | same, Table 2 |

- Wang et al. state neither the LAMBADA variant nor a SE.
- At N = 5,153 and p ≈ 0.35, the binomial SE is ≈ 0.0066 [INF]. That is consistent with Pythia's reported ± (0.007 at p = 0.328).

**Checkpoint facts** [code-inspected, HF `config.json`, fetched 2026-11-09]:
- **GDN models:** `GatedDeltaNetForCausalLM`, hidden 1024, 24 layers, 4 heads × head_dim 256, `expand_v` 1, gate on, short conv 4, untied, V = 32,000, `norm_eps` 1e-6, `hidden_ratio` 4.
- **Hybrid "3-1" (sha `417aec6`):** attention at layers [2, 5, 8, 11, 14, 17, 20, 23], 16 heads, 8 KV heads, RoPE θ = 1e4, no QK-norm.
- **Transformer:** `TransformerForCausalLM`, 16 heads, MHA, RoPE θ = 1e4, tied.
- **Blocker for the original code:** fla 0.6.0's `fla/layers/attn.py` imports `flash_attn` (L26) and raises without it [code-inspected], so the hybrid and Transformer checkpoints cannot run in fla here as shipped.

**Design (P1, P2).**
1. **Original code.** lm-eval (local clone `d6de816`) `lambada_openai`, 0-shot:
   - Pythia via HF.
   - GDN-pure via fla `GatedDeltaNetForCausalLM` (pure GDN needs no flash_attn).
   - Transformer and 3-1 via fla, with a **repro-only SDPA shim** replacing `flash_attn_func`/`flash_attn_varlen_func` (causal; GQA by `repeat_interleave`). The shim is validated first: one fla attention layer, shim vs a naive softmax reference, max abs diff ≤ 1e-5 fp32.
   - Also run `lambada_standard`, because Wang's variant is unstated.
2. **Our code, logits.** Load the trained weights into modules built by `build_mixer`, inside a repro-only wrapper.
   - The wrapper is needed because `ModelConfig.__post_init__` refuses untied embeddings ("protocol §3.1 fixes tied embeddings") [code-inspected `src/hybridlc/models/config.py` L68–69]. It reuses our blocks and mixers with a separate untied head.
   - Overrides: GDN `num_heads=4, head_dim=256, expand_v=1`; attention `qk_norm=False, out_gate=False`, RoPE θ = 1e4, KV heads replicated to MHA (exact).
   - The Transformer checkpoint is tied, so it can go through `HybridLM` itself (pattern `A`×24, n_layers 24).
   - Compare fp32 logits vs the fla reference on held-out FineWeb-Edu text at T = 512 and 2,048, and at 4,096 and 8,192 (beyond max_position_embeddings 2,048; a numerics test only).
3. **Our scorer.** Write LAMBADA as item files in our format, per model tokenizer: context = passage minus final word, minus its last token; prompt = that last token; answer = the final-word tokens. Score with `eval.scorer.score_file`, with `ForwardOnlyAdapter` as a second path. Compare per item with lm-eval.

**Pass criteria (pre-register).**

| Check | Criterion | Justification |
|---|---|---|
| (a) environment | lm-eval Pythia-70M/160M within ±0.015 of 0.185/0.328 | ≈ 2× the reported SE, plus lm-eval version drift (Pythia's numbers predate `d6de816`) |
| (b) Wang checkpoints | lm-eval GDN-pure and Transformer within ±0.02 of 0.348 / 0.353 on `lambada_openai`, or on `lambada_standard` if only that matches (report both) | ≈ 3 SE; variant unknown |
| (c) transplant logits | fp32 max relative logit error ≤ 1e-3 (the VT-09 threshold) at T ≤ 2,048, and argmax agreement ≥ 0.999 at all T, for GDN-pure, 3-1 and Transformer | Same code path in exact arithmetic |
| (d) scorer | Per-item EM agreement between our scorer and lm-eval ≥ 99.5%, and aggregate |Δacc| ≤ 0.003 | Identical computation; residual from bf16 ties |

- **Validates:** our GDN and attention mixers with trained weights (RoPE convention: interleaved vs half-split; fla's L2 q/k norm and conv); EM_tf/logp scoring and continuation alignment; the adapter. **Rules out:** E10.4, E10.3 (scoring path) and part of E1.10 (metric artefact).
- **Scale-down:** none. These are published checkpoints, evaluation only.
- **GPU-h:** ≈ 1.0. LAMBADA ≈ 5k short items × 5 models × 2 variants, plus transplant tests [INF].
- **Depends on S0:** `build_mixer`, `AttnMixer`/`GDNMixer` with overrides, `eval.scorer`, `eval.adapter`. lm-eval is installed in `.venv`? (unverified; may need an install approval).
- **Failure → diagnosis.**
  - (a) fails: lm-eval / tokenizer / HF issue. Stop and fix the environment.
  - (b) fails while (a) passes: check the variant and the fla version drift (the checkpoints were saved with transformers 4.49 / 4.51 and an older fla: missing or renamed keys, the old `qk_activation`/`expand_k` fields). Load with `strict=True` and list mismatched keys.
  - (c) fails: bisect by layer (embedding → layer 0 GDN → first attention layer at index 2 → …). Common culprits: RoPE pairing convention, `norm_eps`, conv state orientation, gate activation, head ordering in GQA replication.
  - (d) fails: off-by-one in continuation alignment (INTERFACES "logprobs[:,0] from cache.last_logits"), or leading-space tokenisation.

### M3. Our Tier-0 harness on the Wang 340M public GDN ratio sweep (pattern; harness sensitivity)

- **Published pattern.**
  - LM metrics are flat across ratios. GDN LM-average per ratio: 24-1 0.468, 12-1 0.478, 6-1 0.481, 3-1 0.478, pure 0.487, averaged over 340M and 1.3B (2507.06457 Table 6 [read-section]).
  - Recall rises with more attention. GDN recall: 0.341 / 0.352 / 0.425 / 0.436 / pure 0.356 (Table 7 [read-section]; per LR also). Recall at 340M was "insignificant" and is not reported (Table 2 caption [read-section]).
  - With finding 3, the attention counts are k/24 ∈ {0, 1, 2, 4, 8}, plus the Transformer at 24/24.
- **What we run.** Six public checkpoints: GDN-pure, New-24-1, 12-1, 6-1, 3-1, Transformer. Each runs through **our** Tier-0 generators, re-tokenised with each model's own tokenizer and kept within 2,048 tokens:
  - LM-1 per-position loss, buckets up to [1K, 2K), on FineWeb-Edu shard-000 test-half and PG19-test;
  - LM-3 repeat loss by distance, Δ < 2K;
  - LM-4 copy-after-gap, variant (b) random tokens, gaps {0, 512, 1,500};
  - R0 cloze passkey at 2K.
- **Pre-registered checks.**
  - (i) **Positive control.** The Transformer's LM-4(b) second-copy loss on tokens 9–256 at gap 512 is ≤ 1.0 nat and below 25% of its first-copy loss. Random-token copying is the induction behaviour Olsson et al. describe; this magnitude is my threshold [INF], not a published number.
  - (ii) **Sensitivity.** The Spearman correlation between attention-layer count (0, 1, 2, 4, 8, 24) and the LM-4(b) copy gain at gap 1,500 is ≥ 0.8.
  - (iii) **Validity pattern (E9.1).** LM-2-style mean loss over [0, 2K) has a range across the 5 GDN-family models ≤ 0.05 nats, while the copy gain differs by ≥ 3× its bootstrap SE between pure and 3-1.
  - The ±0.05 nats is my tolerance [INF]. Wang publish no per-ratio loss at 340M; Table 6's accuracy spread of 0.019 is the only anchor.
- **Validates:** the LM-1/LM-3/LM-4/R0 code on real hybrids; that Tier-0 metrics can detect k at all, before we spend ≈ 86 GPU-h; the "exposure-robust" label (E3.4) and E9.1. It also gives a free k-direction prior from a GDN sweep trained on FineWeb-Edu with fla.
- **Scale-down:** none (evaluation only). Caveat: Wang's models are 340M and trained with a 2K or 4K context (unresolved U1). `max_position_embeddings` = 2,048 in the GDN configs is suggestive only [code-inspected].
- **GPU-h:** ≈ 0.5.
- **Depends on S0:** `eval/tier0.py` generators (need a `tok` re-tokenisation path or a repro wrapper), plus M2's loading wrapper.
- **Failure → diagnosis.**
  - (i) fails: harness bug, e.g. second-copy positions misaligned or loss averaged over the wrong tokens. Check on a hand-built 8-token example.
  - (ii) fails while (i) passes: either GDN copies as well as attention at ≤ 1.5K gaps for a 256-token span (genuine; the state of 4 × 256 × 256 per layer may hold it), or a metric artefact. Then try LM-4 at a larger span or gap within 2K and report. **This must not lead to changing LM-4's definition for our study without a logged decision.**

### M4. nanoGPT shakespeare-char ladder (training-loop correctness)

- **Claim.** "On one A100 GPU this training run takes about 3 minutes and the best validation loss is 1.4697" (nanoGPT README L51, @`3adf61e` [code-inspected]).
  - Config: 6 layers, 6 heads, d = 384, block 256, dropout 0.2, batch 64, LR 1e-3 cosine to 1e-4 over 5,000 iterations, warm-up 100, β₂ = 0.99 (`config/train_shakespeare_char.py` [code-inspected]).
  - No seed variance is reported.
- **Ladder (P2).** Each rung runs 3 seeds except r3/r4. `compile=False` throughout, since the protocol forbids compile.

| Rung | What runs | Compared with | Isolates |
|---|---|---|---|
| r0 | nanoGPT original config | the published 1.4697 | environment (GPU, torch, bf16) |
| r1 | nanoGPT, with schedule and β set to values our trainer supports (warm-up → constant, or our `lr_at` form; β₂ 0.95) | r0 (not tested: a context rung) | — |
| r2 | **Our trainer** (`training/trainer.py`) with `impl.build_model` pointing at nanoGPT's `GPT`, a repro char-level `StageStream` (each 256-char window its own segment: `cu_seqlens` per sample, as nanoGPT's independent windows), and the r1 hyperparameters | **r1, rung-for-rung** | Our optimiser groups, schedule, accumulation, clipping, loss averaging, z-loss off, train/eval mode (dropout!), checkpointing |
| r3 | Our trainer + our `HybridLM` C12 at d = 384, n_layers = 6 (pattern `AAAAAA`) | r2 (directional) | Model swap only |
| r4 | Our C0 (`------`) same setting | r3 (descriptive) | — |

- **Evaluation:** nanoGPT's `estimate_loss` with a fixed eval RNG on every rung's checkpoints at the same 250-iteration intervals; best-val as in nanoGPT.
- **Pass criteria.**
  - r0: mean best-val within ±0.03 of 1.4697. This is a provisional tolerance: no variance is published, and the seed SD from r0's 3 seeds replaces it if 3·SD > 0.03.
  - **r2 vs r1:** |Δmean| ≤ max(0.01, 2·pooled SD) nats, *and* the loss curves at matching iterations agree within the same tolerance.
  - r3: no threshold. Expect best-val ≤ r2 + 0.05 (a modern block should not be much worse) [INF]. A large gap triggers a component audit.
- **Validates:** the trainer contract (INTERFACES "Step objective"), independent of our model and data. **Rules out:** an infrastructure explanation of E2.3 (a broken base LM), and E10.x training bugs.
- **Scale-down:** none needed (it is tiny).
- **GPU-h:** ≈ 11 runs × ≈ 8 min contended ≈ 1.5 h [INF: 3 min on an A100, assuming ≈ 2× slower and contended].
- **Depends on S0:** `training/trainer.py` and `config.py` `impl` strings (exist per INTERFACES). Repro-only adapters live under a new `scripts/repro/` (to be created by whoever owns scripts/, not me).
- **Failure → diagnosis.**
  - r0 fails: environment (bf16 autocast, SDPA backend). Try fp32.
  - r2 ≠ r1: diff the per-step `trace.jsonl` against nanoGPT's loss at step 1 (identical init and batch ⇒ identical loss to ~1e-6 in fp32). Then check LR per step, the weight-decay group membership list, grad-norm before and after clipping, and that `model.eval()` is used in evaluation.

### M5. Hybrid-ratio LM-loss pattern (RP-2 made quantitative; pattern replication)

**Published patterns.**

- **Mamba-2, Table 2.** Perplexity of a 350M, 48-layer model trained to 7B tokens on the Pile with the GPT-2 tokenizer, by number of attention blocks ([2405.21060v1](https://arxiv.org/html/2405.21060v1) §9.2.3 Table 2 [read-section]):

  | Attention blocks | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 9 | 11 | 15 | 24 | Transformer++ |
  |---|---|---|---|---|---|---|---|---|---|---|---|---|---|
  | Perplexity | 8.60 | 8.38 | 8.32 | 8.29 | 8.29 | 8.28 | **8.26** | 8.27 | 8.28 | 8.30 | 8.34 | 8.50 | 8.68 |

  In log-ppl [INF]:
  - 0 → 1 is 0.026 nats, 64% of the 0 → best gain of 0.040;
  - the range over 1–11 blocks is 0.014 nats;
  - 24/48 is 0.029 nats above the best.
- **Waleffe et al., 130M / 24 layers.** "validation loss is minimized when roughly 8% of the layers are self-attention layers" ([2406.07887v1](https://arxiv.org/html/2406.07887v1) §4.1 [read-section]). Values exist only in Fig. 4. The digitised values (2.615 at 0, 2.562 at 1, 2.554 best, 2.631 at 24/24, per LR; fig, approx.) are **not** used as thresholds.
- **GDN-specific caveat.** On downstream accuracy, *pure* GDN averaged the highest (0.487 vs 0.468–0.481 for hybrids; 2507.06457 Table 6 [read-section]). So "the first attention layer lowers LM loss" is established for Mamba-2 hybrids, not for GDN.

**Our run.** S3 already trains k ∈ {0, 3, 12} × 4 LRs × 0.15B tokens (§7.2). Add:
- k = 1 and k = 6 at η* (or at the provisional 2.0e-3 if η* is not yet fixed);
- seed index 103 at k ∈ {0, 12} at η*, giving the paired seed-noise estimate σ_pair.

The metric is final LM-2-subset loss (as S3).

**Pre-registered checks.**

| ID | Check | Tolerance | Basis |
|---|---|---|---|
| P1 | loss(1) < loss(0) | drop > max(3·σ_pair, 0.005) | Mamba-2 0 → 1 = 0.026 nats; σ_pair measured |
| P2 | Flat among k ∈ {1, 3, 6}: range ≤ 0.02 nats | — | Mamba-2 range over 1–11 blocks = 0.014 nats |
| P3 | All-full is not the best: loss(12) ≥ min_k loss(k) | descriptive | Mamba-2: Transformer++ 0.050 nats above the best. GDN evidence mixed |
| P4 | The ranking of k at η* does not flip at the adjacent grid LRs | descriptive | LR fairness (E1.8) |

- **Interpretation rule.** P2 failing (a non-flat curve, e.g. a monotone 0.05-nat slope) triggers the RP-2 audit of data, masking and position encoding (§17) *before* S4. P1 or P3 failing alone is reported as "the GDN pattern differs from the Mamba-2 pattern" and audited only if LM-1 per-position curves also look anomalous (P8: pattern, not number).
- **Validates:** the whole LM stack across k (data identity across k, init sharing, varlen masking for both mixers) and LR-fairness assumptions. **Rules out:** E9.1 as an artefact; part of E1.8.
- **Scale-down:** 0.15B tokens at 4K vs 7B tokens at 350M. Why the scaled version still tests the claim: the pattern is reported at 130M, 350M and ≈ 1B (Bae, per LR) in different recipes, so it should not depend sharply on scale. If it does, that is itself useful to know before S4.
- **GPU-h:** 4 runs × 0.35 h (D384; S3 rate 4.2 h/12) ≈ 1.4 h, or 4 × 0.47 ≈ 1.9 h (D512).
- **Depends on S0/S3:** the full trainer and data path. Runs alongside S3.

### M6. Induction heads and smeared keys (Olsson et al.): attention correctness and the conv-asymmetry premise

**Published claims.** All from Olsson et al. 2022 ([transformer-circuits](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html); PDF text [read-section]).
- The in-context-learning score is defined as "the 50th token loss minus the 500th token loss". Before the phase change there is "less than 0.15 nats", after it "roughly 0.4 nats", "constant across many different model sizes (except for the one layer model where not much in-context learning ever forms)".
- Small models: d_model 768, 12 heads, context 8,192, ≈ 10B tokens. The phase change appears at ≈ 1–3B tokens, during an LR warm-up over the first 1.5e9 tokens.
- With the "smeared key" modification (k_j ← σ(α)k_j + (1 − σ(α))k_{j−1}), "in-context learning does indeed form for one-layer models (when it didn't before)".

**Why it matters for us.**
- **Rival E1.6** says GDN layers (with their short conv) give one attention layer previous-token features, so a single F_g can act as an induction head. Our F_gc (conv on q/k/v) is a superset of Olsson's smeared key.
- This test checks E1.6's mechanism premise and our F_gc implementation. It also checks the attention mixer itself: induction needs a correct causal mask, RoPE and composition.

**Configs.** Our `HybridLM`, D384 widths, our tokenizer, FineWeb-Edu Stage-A stream at seq 2,048, AdamW recipe §7.1 at a provisional LR of 2e-3, 2 seeds each:

| Config | Pattern (n_layers) | Prediction |
|---|---|---|
| I-1A | `A` (1) | ICL score low |
| I-2A | `AA` (2) | ICL ≈ 0.4 nats (Olsson) |
| I-1C | `C` (1), F_gc | ICL forms (smeared-key analogue) |
| I-GA | `-A` (2) | ICL forms (GDN supplies previous-token features) |
| I-2G | `--` (2) | descriptive |

These models *have* MLPs; Olsson's small models include variants with and without MLPs [read-section].

**Pre-registered checks.**
- (a) ICL(I-2A) ≥ 0.25 nats.
- (b) ICL(I-1A) ≤ 0.5·ICL(I-2A).
- (c) ICL(I-1C) ≥ 0.7·ICL(I-2A).
- (d) Repeated random tokens (a 256-token random sequence repeated once): I-2A's second-half loss ≤ 1.0 nat; I-1A's ≥ 3.0 nats.
- These ICL scores use docs ≥ 512 tokens from the held-out test half.
- The tolerances are deliberately wide [INF]: tokenizer, data and model size differ, and Olsson report no seed variance. Only (a) is anchored to a published magnitude (0.4).

**Scale-down and its risk.**
- 0.2B tokens vs ≈ 10B, and seq 2,048 vs 8,192. The ICL score needs only positions 50 and 500.
- The *timing* of the phase change in Olsson coincides with their warm-up, so it is not transferable. We test only the end state.
- If (a) fails at 0.2B, extend I-2A seed 0 to 0.6B tokens before declaring failure (pre-declared).

**Validates and rules out.**
- The attention mixer (causal mask, RoPE, per-head QK-norm), F_gc's conv, and the LM-1 per-position code.
- **E1.6 / E6.2 / E13.1:** if I-GA and I-1C form induction and I-1A does not, the conv-asymmetry mechanism is real at our scale, and X-gc12 is the right control.
- **E7.2** (layer-0 feature availability).

**Cost, dependencies, failure.**
- **GPU-h:** 10 runs × ≈ 11 min ≈ 1.9 h. Throughput is LM-head-dominated: 2·d·V ≈ 12.6 MFLOP/token vs ≈ 4.7 per layer, giving an estimated ≈ 300K tok/s [INF].
- **Depends on S0:** `ModelConfig(n_layers=1|2)`, the trainer, the Stage-A stream at 2K, and LM-1.
- **Failure → diagnosis.**
  - (a) fails with (d) also failing: attention bug. Check the RoPE application order (after QK-norm), the position reset with `cu_seqlens`, and the causal flag on `varlen_attn`. Run VT-02/03 at the trained checkpoint.
  - (a) passes but (c) fails: inspect F_gc's conv (VT-03 conv reset; the SiLU on v only).
  - (b) fails (1-layer forms ICL): check for leakage across positions, e.g. a non-causal conv or label shift.

### M7. Discriminating MQAR (S1-ext): our GDN vs Zoology's GDN where accuracy is mid-range

- **Claim (re-run numbers only).** Zoology's GDN (2 heads, `use_gate=False`, conv 4) loses accuracy as load per unit state rises [MEAS-S1 interim]:
  - d128: 1024/256 cell 0.925;
  - d256 after one epoch: 0.911.
  - At d = 64 the state is ≈ ¼ of d128's, so the 1024/256 and 512/128 cells should sit in a mid-range band (0.2–0.9). That is a prediction to verify on run 1, not a published number.
- **Design.** d_model = 64, Zoology's best GDN LR from S1(a) at d128 (no new LR sweep). 3 seeds (Zoology `seed` and `data_seed` 123, 124, 125) × {Zoology GDN, `OurGdnZoology`} = 6 runs.
  - Pre-declared fallback: if the 1024/256 cell is > 0.95 or < 0.05 for Zoology seed 123, add test configs (1024, 512) and (2048, 512) using Zoology's `MQARConfig`. The training set stays unchanged.
- **Pass criterion.** For the 1024/256 and 512/128 cells: |mean_ours − mean_Zoology| ≤ max(0.03, 2.5·SD_pooled/√3·√2) (Welch-style, 3 vs 3). SD comes from these very seeds, so the tolerance is variance-derived (R-S1-5).
- **Validates:** our GDN wrapper's numerics, initialisation and conv/gate plumbing in a regime where errors move accuracy. **Rules out:** E10.4, E12.3.
- **GPU-h:** 6 × ≈ 16.5 min ≈ 1.7 h (d128 timing as an upper bound) [INF].
- **Depends on:** S1(a) best LR and `build_mixer`.
- **Failure → diagnosis.** Same as M1(b). Additionally compare Zoology's and our per-layer outputs on identical weights; the differences should be ≤ fp32 noise.

---

## 4. Should tier

### S1. Copy-task sample efficiency (Jelassi et al. 2402.01032): the learning-speed confound

- **Claim.** On ≈ 160M-parameter models trained on strings of length ≤ 300 and evaluated at 300, "the transformers need 100x less samples than the best GSSMs to learn the copy task". For strings ≤ 30, Mamba needs about 10× more data (§3.2, Fig. 1(a); App. B.1, Fig. 9(b)) ([2402.01032](https://arxiv.org/html/2402.01032) [read-section]).
  - The sample counts themselves are figure-only (fig, approx.).
- **Ours.** D384-width, 6-layer `AAAAAA` vs `------` vs `--A---`, trained from scratch on Jelassi's copy format, using our tokenizer's digit or letter tokens; length ≤ 300 tokens; online batches of 64. Record examples until 95% string accuracy at length 300. 2 seeds.
- **Pass (directional):** samples_95(GDN) / samples_95(attention) ≥ 10. The 100× claim is for Mamba, not GDN; the threshold is set at 1/10 of it [INF].
- **Validates/rules out:** E1.7 / E2.2 / E6.7 (learning speed under fixed TEF exposure), whether the TEF dose ladder sizes are plausible, and the LM-4 positive control.
- **GPU-h:** ≈ 2–3 [INF].
- **Depends on:** the trainer with a custom stream.

### S2. RoPE extrapolation as a positive control for LM-1 and length tags

- **Claims.**
  - ALiBi trains a 1.3B model on length 1,024 that "extrapolates to input sequences of length 2048" and argues that current methods (sinusoidal, by contrast) do not allow efficient extrapolation ([2108.12409](https://arxiv.org/abs/2108.12409) [abstract-only]).
  - The Transformer's NIAH collapses beyond 8K at 8K training (Bae et al., per LR; figure-only).
  - Mamba-2 loss rises beyond the training length (Stuffed Mamba Fig. 1, [2410.07145](https://arxiv.org/html/2410.07145) [read-section]; no values given).
- **Ours.** Evaluate on LM-1 up to 16K and R0 at 8K/16K, at no training cost:
  - the FEAS **end-of-Stage-A** (4K) checkpoints at k ∈ {0, 12};
  - the end-of-Stage-B (32K) checkpoints.
- **Pass (directional, pre-registered):**
  - (i) Stage-A C12 LM-1 loss in [8K, 16K) exceeds its [2K, 4K) loss.
  - (ii) After Stage B, the [8K, 16K) loss is ≤ the [2K, 4K) loss + 0.02.
  - (iii) C0 is reported descriptively.
- **Validates/rules out:** the position handling and LM-1 extrapolation tagging; E1.2 / E2.6 / E6.6 (RoPE large distances), E3.5. If (ii) fails, Stage B did not teach long positions: a G-REF warning *before* S4.
- **GPU-h:** ≈ 0.3.
- **Depends on:** the FEAS checkpoints (S2).

### S3. Our generator vs RULER on a public checkpoint (protocol Tier C #12, promoted here)

- **What.** RULER `niah_single_2` / `niah_multikey_2` through lm-eval vs our R1/R-load reimplementation, on m-a-p/transformer_340M_baseline and Pythia-160m at 1K–2K.
- **Criterion (no published target):** per-cell accuracy difference ≤ 10 pp, *or* an explained difference, since RULER uses ≤-length fitting and instruction prompts (03 §RULER [code-inspected per LR]).
- **Validates:** the generator semantics.
- **GPU-h:** ≈ 0.5.

### S4. Wang 1.3B: LAMBADA for the GDN 24-1 row

- **Target.** GDN 24-1 at 1.3B: LMB 0.502 (Table 3 [read-section]). The checkpoint `m-a-p/1.3B-100B-GatedDeltaNet-hybrid-24-1` is in the collection [code-inspected, collection API].
- **Purpose.** A second, larger GDN transplant and LAMBADA point (M2 protocol). It tests the GDN mixer at d = 2048, 8 heads × 256.
- **GPU-h:** ≈ 0.3.

---

## 5. Nice tier

- **N1. Wang RULER Table 5.**
  - Targets: GatedDeltaNet-hybrid-3-1 average 0.440 vs Transformer 0.423 (2507.06457 Table 5 [read-section]).
  - Scale and length are **not stated** in the table, so the target is ambiguous; this is a weak reproduction. Assume 1.3B and 4K: lm-eval `ruler` on 2 models ≈ 2–3 GPU-h.
  - Validates the external benchmark path only.
- **N2. modded-nanogpt / llm.c-style GPT-2 124M to 3.28 FineWeb val loss on one GPU.**
  - Reference: the mean of 80 reruns is 3.2791 with SD 0.0019 (`records/track_1_short/2025-01-04_SoftCap/README.md` L27–28 @`4ea6b93` [code-inspected]); the rule is "Attain ≤3.28 mean val loss" (README Rules [code-inspected]).
  - Needs GPT-2-tokenised FineWeb (not Edu) and is multi-GPU-oriented. Estimated 3–5 GPU-h [INF]. It validates the environment only, not our code.
  - Low value for us relative to M4.
- **N3. Phonebook lookup (Jelassi §4.3)** on Pythia-410M: "even the smallest transformer (410M parameters) outperforms the largest GSSMs (2.8B parameters)" for L ≥ 70 [read-section]. The GSSM side needs `mamba_ssm`, which is unavailable, so only the Transformer half can run. ≈ 0.5 h.
- **N4. Stuffed-Mamba-style state behaviour for GDN.** Per-position loss and R0 passkey for C0 at Stage A (4K) beyond 4K. There is no published GDN number (Stuffed Mamba has no GDN, [read-section]), so this is descriptive only. Covered by S2's C0 arm.

**Considered and rejected.**
- *flame / torchtitan* re-training of a Wang model: torchtitan version conflicts (05 §… [code-inspected per LR]), and 20B tokens are far beyond budget.
- *GDN paper (2412.06464) S-NIAH table*: "No, we only provide code implementations" (third_party/gated_deltanet README Q3 [code-inspected]), so no checkpoints exist.
- *Waleffe 130M numbers*: figure-only.
- *Zoology Fig. 2 full sweep*: 448 configurations (README [code-inspected]); M7 extracts the discriminating part.

---

## 6. Budget, order and gate

| Tier | Items | GPU-h |
|---|---|---|
| Must | M1 (S1, with R-S1-1 cut) 5.8; M2 1.0; M3 0.5; M4 1.5; M5 1.4 (D384) / 1.9 (D512); M6 1.9; M7 1.7 | **≈ 13.8 (D384) / 14.3 (D512)**, of which ≈ 8.0 is new relative to the protocol's S1 line of 3.0 |
| Should | S1 2–3; S2 0.3; S3 0.5; S4 0.3 | ≈ 3.1–4.1 |
| Nice | N1 2–3; N2 3–5; N3 0.5 | ≈ 5.5–8.5 |

**Order** (each item lists the S0 parts it needs):
1. M1 (running).
2. M2 and M3: evaluation only; need `build_mixer`, scorer, tier0.
3. M4: needs the trainer.
4. M6 and M7: need trainer + model, and S1's LR.
5. M5: alongside S3.
6. S2: after FEAS.

M2 and M3 can start as soon as `build_mixer` and the scorer exist, *before* any LM training.

**Proposed gate G-RV** (an addendum to G0/G-RP; decision row needed):
- No S2 FEAS until M2(c)(d), M4 r2 and M6(a)(d) pass, or their failures are diagnosed and fixed under P5.
- M3, M5 and M7 are reported but non-blocking, except that a failure of M5-P2 or M3(i) triggers the RP-2-style audit.

---

## 7. Reporting template (per reproduction, stored in `experiments/repro/<id>/`)

1. The claim, quoted, with URL, table/figure and access tag.
2. Original-code command, with commit, environment and shims (each shim's own validation).
3. Our-code command, with commit and config hash.
4. The pre-registered tolerance and its derivation (seed SD, SE, or "provisional, replaced by measured SD").
5. Results per seed, with the raw JSONL path.
6. PASS / FAIL / PARTIAL.
7. If FAIL: diagnosis steps taken, defect found (test that isolates it), fix commit, re-run. **Never** a parameter change justified only by the target number.
8. Implication for the rival-explanation rows (`competing_explanations.md` status table).

---

## 8. Sources opened for this proposal

| Source | Used for | Access |
|---|---|---|
| [2405.21060v1](https://arxiv.org/html/2405.21060v1) Mamba-2 §9.2.3 Table 2 | Ratio perplexity pattern (M5) | read-section (WebFetch) |
| [2406.07887v1](https://arxiv.org/html/2406.07887v1) Waleffe §4.1 | "roughly 8%" statement; Fig. 4 is figure-only | read-section (WebFetch) |
| [2507.06457](https://arxiv.org/abs/2507.06457) Wang et al. §3.4, Tables 2, 3, 5, 6, 7, App. D | LAMBADA targets, ratio tables, r:1 statement | read-section (PDF text in the scratchpad, cross-checked with v2 HTML via WebFetch) |
| [HF collection m-a-p/hybrid-linear-attention-research](https://huggingface.co/collections/m-a-p/hybrid-linear-attention-research); `config.json` and API `sha` of the 340M GDN pure / 24-1 / 12-1 / 6-1 / 3-1, Transformer-340M, and 1.3B GDN pure / 3-1 | Checkpoint geometry, attention layer indices, finding 3 | code-inspected (curl, 2026-11-09) |
| [2304.01373](https://arxiv.org/abs/2304.01373) Pythia App. G.1 Table 7 | Pythia LAMBADA targets | read-section (PDF text) |
| [Olsson et al. 2022](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) | ICL score definition, 0.4 nats, 1-layer, smeared key, model details | read-section (PDF text) |
| [2402.01032](https://arxiv.org/html/2402.01032) Jelassi et al. §3.1–3.3, §4.3 | Copy sample efficiency, phonebook | read-section (WebFetch) |
| [2410.07145](https://arxiv.org/html/2410.07145) Stuffed Mamba | Setup; no GDN; Fig. 1 qualitative | read-section (WebFetch) |
| [2108.12409](https://arxiv.org/abs/2108.12409) ALiBi | Extrapolation claim | abstract-only |
| nanoGPT @`3adf61e` README L51, L149–152; `config/train_shakespeare_char.py` | M4 target and config | code-inspected |
| modded-nanogpt @`4ea6b93` README Rules; SoftCap record README L16–28 | N2 target and seed SD | code-inspected |
| zoology @`1ad20d1`: original_mqar_configs, iclr24 figure2 README/configs, composition configs, capacity scripts | M1/M7 configs; figure-only status | code-inspected |
| fla @`98193113` `fla/layers/attn.py` L26 (flash_attn import), `configuration_gated_deltanet.py` | Shim need; config fields | code-inspected |
| third_party/gated_deltanet @`b53d6d3` README Q3 | No released checkpoints | code-inspected |
| `src/hybridlc/models/config.py` L60–69; `eval/adapter.py` | Untied refusal; adapter contract | code-inspected (read only) |
| `experiments/s1_zoology/{cut_decision.json, runs/*.json, logs/*}` | Interim S1 timings and accuracies | MEAS-S1 interim (read only) |
| Project docs: README, protocol v1.0, competing_explanations, INTERFACES, literature_review, s1_reproduction | Context | read-full (protocol §1–20; literature_review in full); 0*.md read by grep for the relevant items only |

**Not verified by me:**
- Kimi and Bae ratio numbers (per LR only, not used as targets).
- Waleffe Fig. 4 digitised values.
- Whether lm-eval is installed in `.venv`.
- Whether the old-fla checkpoints load cleanly in fla 0.6.0 (M2 diagnoses this).
- The actual d = 64 MQAR accuracy band (M7 checks it on run 1).

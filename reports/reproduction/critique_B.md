# Critique B (revision 2): engineering feasibility and cost of the two reproduction proposals

**Reviewer:** adversarial reviewer B (engineering feasibility and cost).
**Written:** 2026-11-09, 22:15–22:45 IST. By then S1(a) had finished (final table regenerated at 22:26) and the S0 builders had reported done.
**Supersedes** the 18:50 draft of this file. That draft was written while S1 was still running, and its workflow step was cut off by the usage limit. Every fact below was re-checked in this session, and every target number was re-opened by me.

**Reviewed:**
- `proposal_pipeline-validation.md` ("PV": M1–M7, S1–S4, N1–N4);
- `proposal_phenomenon-replication.md` ("PR": P1–P4, S1–S5, N1–N5);
- read alongside `critique_A.md`. §8 lists where I disagree with it.

**Constraints kept.** I ran no GPU job and edited nothing except this file. My system use was limited to:
- `nvidia-smi`, `ps` and a read of `journalctl -k`;
- HF Hub metadata, plus the tokenizer files of one checkpoint;
- arXiv and transformer-circuits HTML and one arXiv PDF, downloaded to the scratchpad and parsed with `python -I`;
- one CPU-only instantiation of our `TokenSpace` with that downloaded tokenizer.

**Tags.**
- [read-section]: primary source opened and parsed by me in this session.
- [code-inspected]: a file or HF config read by me.
- [MEAS-S1]: read by me from `experiments/s1_zoology/` (final state, 22:26).
- [MEAS-S0]: from S0's own benchmark, `runs/bench_train_step/*.jsonl` (contended, 8 timed steps).
- [MEAS-R]: observed by me in this review, without the GPU.
- [INF]: my own inference or arithmetic.

No number in this file is read off a figure.

---

## 0. Bottom line

1. **Most must-tier items do not validate what they claim, and both proposals under-price them.** Every such item fails at least one of six tests:
   - **CF:** it cannot fail at the observed ceiling;
   - **FB:** it can fail with no bug in our code;
   - **NU:** it tests a component we do not use;
   - **RD:** an existing S0 test already covers it;
   - **IF:** it is infeasible here as specified;
   - **SD:** scaling down removes the phenomenon.

   Re-priced with S0's measured throughput (§1.3), the new must-tier work comes to:
   - PV: **≈ 10–12 GPU-h** (stated 8.0);
   - PR: **≈ 11–15 GPU-h** (stated 5.8).
2. **A cheaper set covers more of the real risk** (§5):
   - ≈ 0.3–0.5 new GPU-h of must-tier work, almost all of it deterministic tests in S0's own files;
   - plus S1(b), already budgeted;
   - plus a ≈ 1.3–1.5 GPU-h should tier.

   It adds checks that neither proposal has:
   - answerability of our 16K/32K items;
   - the fla conv kernel against a reference;
   - the full model at long T;
   - GPU-fault hygiene.
3. **The one literature reproduction that runs our mixers is done.** S1(a) passed G-RP(a) on all 16 cells. What remains of S1 is S1(b), and a weight-copy identity test (X2) should run before it.

**New facts since the proposals were written** (all verified in this session):

- **F1. S1(a) finished: PASS, 16 of 16 cells.**
  - Cost: 1.48 h of completed-run wall time, ≈ 1.75 GPU-h with the crashed attempts and smoke runs, against 3.0 budgeted.
  - Forecasts: PV ≈ 5.5 h, PR 3–5 h.
  - Waste: 41% of completed-run time went to two lr 3.2e-2 runs that collapsed at epoch 1 and still trained all 32 epochs [MEAS-S1].
- **F2. The d128 lr 1e-2 cell has completed.**
  - It crashed twice, with Xid 8 and `cudaErrorLaunchTimeout`, at 18:16 and 18:25.
  - A third attempt (22:13–22:26) completed next to CARLA, with no Xid event, scoring 0.9877. So the crash is not deterministic at this LR.
  - That attempt came from an uncommitted change to `run_queue.py` (6 attempts per run, per-epoch resume), made after the verdict had been computed.
  - The verdict is unchanged either way: the best LR stays 1e-3 [MEAS-S1, MEAS-R].
- **F3. S0's training benchmark is far below the planning rates, so LM-training items are under-priced by ≈ 1.7–2.3×.**
  - Contended Stage-A throughput, measured with the S1 queue and CARLA on the GPU [MEAS-S0]:
    - D384: 52.7–70.1K tok/s (plan 120K; G0 needs ≥ 95K);
    - D512: 63.2–66.6K (plan 88.9K; G0 needs ≥ 70K).
  - Both proposals price their LM-training items at the planning rate.
  - For example, PR-P3 becomes ≈ 6.0 GPU-h, not 3.2.
- **F4. Our item generators cannot use the Wang checkpoints' tokenizer.**
  - `TokenSpace(tok)` raises `ValueError: digit 0 is not a single token: [28705 28734]`. The Mistral-style SentencePiece tokenizer adds a dummy-prefix "▁".
  - PR-P2(b) and PV-M3 therefore need changes inside `src/hybridlc/eval` before they can run [MEAS-R].
- **F5. GPT-2 weights are not cached.** Only its tokenizer and config are (2.8 MB), so PR-S3's "needs no model download" is wrong [MEAS-R].
- **F6. The Wang 1.3B sweep is stored in fp32, at ≈ 33.4 GB for six repos** [code-inspected].
  - Sizes: 5.6–5.9 GB per GDN repo, 4.6 GB for the Transformer. PR-P2 assumed 6 × 2.7 GB in bf16.
  - Attention heads differ by ratio: the 3-1 model has 16 heads (head_dim 128); 24-1, 12-1 and 6-1 have 32 (head_dim 64).
  - Repo names: the 1.3B pure model is `m-a-p/1.3B-100B-GatedDeltaNet-pure`; the 340M sweep's 24-1 model is a "New-hybrid-24-1".
- **F7. VT-12 already has an answerability oracle, but only at 2K** [code-inspected].
  - `InductionLM` is an order-6 copy model, run end to end through our scorer.
  - R1, R-load and R3 must score EM_tf = 1 with exact logp; M-hop must fail.
  - Extending it to the 16K/32K cells costs no GPU time.
- **F8. Three S0 gaps matter more than any proposed reproduction** [code-inspected]:
  - no test compares fla's Triton `causal_conv1d`, which GDN-A and F_gc use, with our own `causal_dwconv_reference` on the GPU;
  - VT-03 has no test file. VT-09's packed cases cover document isolation for C12 and C0 at ≤ 200 tokens, but not F_gc, hybrids or bf16;
  - no full-model transplant runs beyond T = 256.
- **F9. The documentation lags the runs** [code-inspected]:
  - `s1_reproduction.md` still says "Status: BLOCKED. No training run has been executed".
  - It also calls the micro-batch fallback "not implemented", yet all 16 runs used `micro_batch = 64`.
  - `decisions.md` has no row for the S1 deviations, the crash or the retry policy.

---

## 1. Facts established in this review

### 1.1 S1 outcome, and what the S1 forecasts got wrong [MEAS-S1]

| | PV (R-S1-1) | PR (§2.1) | Actual |
|---|---|---|---|
| S1(a) GPU-h | ≈ 5.5 uncut / 3.3 cut | 3–5 | 1.48 completed + ≈ 0.25 crashed attempts + ≈ 0.05 smoke ≈ **1.75** |
| GDN d256, per run | ≈ 65 min (32 epochs) | ≈ 1.1 h without early stop | **0.6–1.2 min** (early stop after 1–2 epochs); lr 3.2e-2: 18.3 min (collapsed) |

**Why the forecasts failed.**
- Early stopping makes run time bimodal: one epoch, or 32. A single-run projection fails in both directions.
- Collapsed runs train to the end. The two lr 3.2e-2 runs took 1,090 s + 1,101 s, which is 41% of the 5,341 s of completed-run time.

**What the gate sees.** Overall accuracy is the unweighted mean of the 7 test cells (I checked this against the run JSON to 1e-7).

| Model | Best overall | Best LR | 1024/256 cell |
|---|---|---|---|
| GDN d128 | 0.98924 | 1e-3 | 0.925 |
| GDN d256 | 0.99377 | 1e-3 | 0.957 |
| Attention (all runs) | ≥ 0.9997 | — | — |

**Ceiling arithmetic.**
- ±0.03 on overall accuracy equals ±0.21 on the only cell that moves.
- Doubling the state (d128 → d256) moved that cell by just +0.032.
- So G-RP(b) cannot detect a mixer defect smaller than ≈ 6.6× the state-size effect that S1 exists to show.
- S1 is a smoke test. Both critiques reach this independently; it should be labelled as such.

### 1.2 The crash, the re-run and the queue change [MEAS-S1, MEAS-R]

**Kernel log:**
- 18:03:09: Xid 31 MMU fault (pid 799092).
- 18:16:50 and 18:25:42: RC watchdog "GPU is probably locked! Notify Timeout Seconds: 7", plus Xid 8. These match attempt 1 (died after epoch 15, t = 337 s) and attempt 2 (died after epoch 25, t = 525 s).
- No Xid events since 18:25.

**GPU state:** `nvidia-smi -q` shows "Display Attached: Yes" and "Display Active: Enabled".

**Attempt 3** started at 22:13 alongside CARLA (CarlaUE4 ≈ 5.9 GB, plus ≈ 1.4 GB of other CARLA processes). It completed 32 epochs in 775 s, with `resumed_at_epochs` = [].

**Interpretation [INF].**
- The two crashes coincided with several project GPU jobs running at once: the S1 queue plus VT-13, the benchmark and an ad-hoc `run_one.py`, as both earlier critiques observed.
- The clean re-run had only CARLA beside it.
- This is consistent with contention-triggered hangs. It is not established.

**Process issue.**
- The re-run was launched after the verdict had been computed, by an uncommitted queue change (`git status`: `M scripts/s1_zoology/run_queue.py`).
- It is harmless here: the new cell can only enter best-over-LR, and it scored 0.9877 < 0.9892.
- But re-run policy should be declared before verdicts are read (§7).

### 1.3 S0 throughput, and re-pricing the LM-training items [MEAS-S0, INF]

The benchmark comes from files `20261109T130432Z` and `…131220Z` (18:34–18:43 IST): 8 timed steps after 3 warm-up steps, reporting the median. In most rows the S1 queue was also on the GPU, alongside CARLA. The first sweep (`…130402Z`) ran out of memory in every cell.

| Scale | Config | Shape | Median tok/s | Best tok/s | Plan (§13.2) | G0 |
|---|---|---|---|---|---|---|
| D384 | C0 | 2 × 4K | 70,124 | 96,817 | 120.0K | ≥ 95K |
| D384 | C3 | 2 × 4K | 61,367 | 75,980 | 120.0K | ≥ 95K |
| D384 | C12 | 2 × 4K | 52,679 | 62,752 | 120.0K | ≥ 95K |
| D384 | C0 | 1 × 32K, ckpt | 41,021 | 46,017 | 93.8K | — |
| D384 | C12 | 1 × 32K, ckpt | 26,371 | 30,703 | 42.4K | ≥ 27K |
| D512 | C0 | 2 × 4K | 66,580 | 69,214 | 88.9K | ≥ 70K |
| D512 | C12 | 2 × 4K | 63,240 | 65,149 | 88.9K | ≥ 70K |
| D512 | C12 | 1 × 32K, ckpt | 34,029 | 35,645 | 31.9K | ≥ 20K |

This is not a valid G0 measurement, because another project job shared the GPU. But it is the only measurement we have.

**Re-pricing at D384** [INF]. k = 1, 2 and 6 are interpolated at 68K, 65K and 57K tok/s. Small models are scaled by FLOPs: the LM head (2·d·V ≈ 12.6 MFLOP per token, forward) and per-step overheads do not shrink with depth.

| Item | Stated GPU-h | Re-priced GPU-h | Basis |
|---|---|---|---|
| PV-M5 (k = 1, 6; seeds at k = 0, 12) | 1.4 | ≈ 2.7 | 0.61 + 0.73 + 0.59 + 0.79 |
| PR-P3 (9 runs) | 3.2 | ≈ 6.0 | 1.99 + 2 × 2.00 |
| PV-M6 (10 × 0.2B tokens, 1–2 layers) | 1.9 | ≈ 2.8–4.3 | 130–200K tok/s, not the 300K assumed |
| PR-P4b (6 × 0.1B tokens) | 0.5 | ≈ 0.8–1.3 | same |
| PR-S1 Jelassi (6 runs, cap 0.4B tokens each, plus length arm) | 1.5 | up to ≈ 2.7–4.4 | the cap binds if GDN never reaches 0.95 |
| PR-P2(b) (6 × 3,000 contexts × 4 queries) | 0.7 | ≈ 2.4–4 | `ForwardOnlyAdapter` re-runs a full forward per query: 72,000 forwards of a 1.3B model at 1–4K, ≈ 0.1–0.2 s each |
| PR-P2(a) (6 × 800 RULER generations) | 1.0 | ≈ 2–2.7 | ≈ 1.5–2 s per prefill plus 64-token decode |
| PV-M7 (6 runs × 32 epochs at d = 64) | 1.7 | ≈ 1.3–1.8 | 24–34 s per epoch, as measured at d128 |

**Consequence.** The re-priced must tiers are PV ≈ 10–12 GPU-h and PR ≈ 11–15 GPU-h, and both are on the path to FEAS. If the contended rates hold, the main budget (D041) is also under-priced in its Stage-A portions. A clean G0 throughput number is therefore needed before any LM-training reproduction is scheduled.

### 1.4 Tooling, checkpoints and tokenizers [code-inspected, MEAS-R]

**Packages.**
- Not installed in `.venv`: `lm_eval`, `datasets`, `accelerate`, `nltk`, `evaluate`, `sklearn`, `html2text`, `flash_attn`, `causal_conv1d`, `mamba_ssm`.
- Installed: `transformers`, `fla`, `zoology`, `torchvision`, `huggingface_hub`, `safetensors`, `wonderwords`.

**RULER @db43f92 will not run without changes:**
- `niah.py` imports `nltk` at the top;
- `json/PaulGrahamEssays.json` is absent (only a URL list and a downloader are present);
- `english_words.json` is a Git-LFS pointer;
- the HF wrapper uses `trust_remote_code` and `device_map="auto"`, which needs `accelerate`;
- the wrapper never imports `fla`, so fla models are not registered.

**fla 0.6.0.** `layers/attn.py` raises `ImportError` in `Attention.__init__` when `flash_attn` is missing (L71–72). Every Wang hybrid and Transformer checkpoint therefore needs a shim.

**Wang checkpoints** (HF `config.json` and file trees, fetched 22:17):
- All GDN models: head_dim 256, `expand_v` 1, gate on, conv 4, `norm_eps` 1e-6, `max_position_embeddings` 2,048. They also carry legacy keys (`qk_activation`, `expand_k`, `use_beta`, `qk_norm`), and were saved with transformers 4.49 or 4.51. Our GDN-A, by contrast, is head_dim 64 with `expand_v` 2.
- Attention layers at 1.3B (indices of 24 layers):

  | Model | Attention layer indices | Heads |
  |---|---|---|
  | 24-1 | [23] | 32 |
  | 12-1 | [11, 23] | 32 |
  | 6-1 | [6, 11, 17, 23] | 32 |
  | 3-1 | [2, 5, …, 23] | 16 |

  All use 8 KV heads and RoPE θ = 1e4.
- Transformer-1.3B: tied embeddings, 32 heads, `max_position_embeddings` 4,096.
- Sizes are listed in F6.

**Tokenizer of the Wang models.** It is `LlamaTokenizerFast` with `legacy` set, and encodes:
- "5" → `['▁', '5']`;
- " " → `['▁▁']`;
- " 1234" → `['▁▁', '1', '2', '3', '4']`.

Our `TokenSpace` requires every digit and " " to be single tokens, so it raises (F4). Our generators also assemble items from separately encoded segments. With a dummy-prefix tokenizer, that assembly produces token sequences the model would never see from a normal encode, so even a patched `TokenSpace` would yield off-distribution items.

**Downloads for my L2 item.**
- `EleutherAI/lambada_openai` `data/lambada_test.jsonl` is 1.82 MB and needs no `datasets` package.
- Pythia-160M is 375 MB of safetensors (HF sha `50f5173`).
- GPT-2: tokenizer only (F5).

**Original code on this machine.**
- nanoGPT runs on torch 2.11: `torch.cuda.amp.GradScaler` still exists, and `--compile=False` costs only speed.
- modded-nanogpt track 3 needs `@torch.compile`, NCCL `init_process_group` and FineWeb10B GPT-2 shards.

### 1.5 What S0 already covers [code-inspected]

| Proposed item | Already covered by | What is left |
|---|---|---|
| PV-M2(c): transplant with trained weights | VT-09: C12 ↔ `Qwen3ForCausalLM` (fp32 CPU at T = 96, including the cached path and a packed two-document row; bf16 GPU varlen at T = 256); C0 ↔ fla's GDN LM (T = 200, packed [0, 90, 200]). VT-04: GDN kernel against an fp64 recurrence at T = 32,768 | Only activation regimes from trained weights, and only in configurations we do not use (QK-norm off, gate off, θ = 1e4, GQA, K = 256) |
| PV-M4 r2: our trainer | VT-13 (28 tests): 4 schedules; weight-decay groups (fake and real model); accumulation equals full batch, including TEF weights; kill-9 resume; G-div; OOM; registry; protocol audit; 20-step real-model smoke. Loss tests in `test_vt09_loss.py` | An end-to-end update check against an independent loop, including clipping and z-loss → T1 |
| PV-M6 (attention correctness) and PR-P1(c-1) | VT-02 (9 tests, including an exactly-zero fp64 future gradient on an attention-only full model); VT-09 | None for correctness; the E1.6 premise only |
| PV-M2(d) / PR-S3: scorer | VT-06/07/08; VT-12 end-to-end scorer tests (the `InductionLM` oracle with exact EM_tf and logp; agreement with a single pass); the 5% production re-score | Only an external check of the metric definition |
| S1(b) / PV-M7 / PR-S5: "same function" | Nothing deterministic yet | **X2** |
| PV R-S1-4: conv fallback | Moot: the message comes from Zoology's Mamba import probe (`model.py` L270–277; `mixers/mamba.py` L11–14) | The real gap is **X1** |

---

## 2. Target numbers re-verified against the primary sources (in this session)

| # | Number(s) as the proposals use them | Used by | Source and location | Access | Verdict |
|---|---|---|---|---|---|
| 1 | GDN recall: 24:1 0.341, 12:1 0.352, 6:1 0.425, 3:1 0.436, pure 0.356 | Protocol §2 prior; PR-P2; PV-M3 | [2507.06457v2](https://arxiv.org/html/2507.06457v2) App. E, Table 7 | read-section (raw HTML parsed) | Exact |
| 2 | Per-subtask RULER scores (columns below) | PR-P2 (decisive) | Same, App. C, Table 5 | read-section | Exact (values below) |
| 3 | LAMBADA (LMB) at 340M: GDN pure 0.348, Transformer 0.353. At 1.3B: GDN 24-1 0.502, Transformer 0.474 | PV-M2, PV-S4 | Same, Tables 2 and 3. Variant and metric are not stated | read-section | Exact |
| 4 | Ratio definition and training context | PV finding 3; P2 | Same, App. D, §4.3, Limitations; HF configs (§1.4) | read-section; code-inspected | See below. **PV finding 3 is right** |
| 5 | GDN LM average: 24-1 0.468, 12-1 0.478, 6-1 0.481, 3-1 0.478, **pure 0.487** | My §3.2(b) | Same, Table 6 (downstream accuracy, averaged over 340M and 1.3B) | read-section | Exact |
| 6 | Perplexity by number of attention blocks: 0 → 8.60; 1 → 8.38; 2 → 8.32; 3 → 8.29; 4 → 8.29; 5 → 8.28; **6 → 8.26**; 7 → 8.27; 9 → 8.28; 11 → 8.30; 15 → 8.34; 24 → 8.50; Transformer++ 8.68 | PV-M5; PR-P3 | [2405.21060v1](https://arxiv.org/html/2405.21060v1) §9.2.3, Table 2. Setting: "350M model (48 layers) trained to 7B tokens on the Pile with the GPT-2 tokenizer" | read-section | Exact |
| 7 | Pythia LAMBADA (OpenAI), zero-shot: 70M **0.185 ± 0.005**, 160M **0.328 ± 0.007** | PV-M2; my L2 | [2304.01373v2](https://arxiv.org/abs/2304.01373) App. G.1, Table 7, standard suite. The ± is the binomial SE. The deduplicated suite (Table 8) gives 0.192 and 0.342 | read-section (PDF text) | Exact. Pin `EleutherAI/pythia-160m` (not `-deduped`) at a fixed `revision` |
| 8 | Olsson et al. (quotes below) | PV-M6; PR-P4; my I1 | [transformer-circuits 2022](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) | read-section | Quoted correctly by both proposals |
| 9 | Kimi Linear ablation: validation PPL 0:1 5.77, 1:1 5.66, **3:1 5.65**, 7:1 5.70, 15:1 5.82; training PPL 9.45, 9.29, 9.23, 9.23, 9.34 | PR-P3; my §3.2(b) | [2510.26692](https://arxiv.org/html/2510.26692) §5.2, Table 1 | read-section | Validation values match the literature review |
| 10 | nanoGPT "best validation loss is 1.4697" | PV-M4; PR-S4 | `third_party/nanoGPT/README.md` L51 @3adf61e | code-inspected | Exact |

**Row 2: Wang Table 5.**

| Model | Multikey | Multiquery | Multivalue | Singlevalue | CWE+FWE | QA | VT | Avg |
|---|---|---|---|---|---|---|---|---|
| GDN-hybrid-3-1 | 0.293 | 0.542 | 0.502 | **0.875** | 0.271 | 0.300 | **0.024** | 0.440 |
| Transformers | 0.332 | 0.464 | 0.472 | **0.559** | 0.284 | 0.412 | **0.066** | 0.423 |

**Row 4: ratio definition and training context.**
- App. D: "a fixed ratio r:1 (linear : full) … the memory footprint is reduced by roughly a factor of r … r ∈ {24, 12, 6, 3}".
- §4.3: "devote roughly one quarter of the layers to full attention".
- Limitations: "a 4,096-token context window".
- The released "r-1" configs place one attention layer per r layers, so 3-1 has 8 of 24 layers as attention (⅓, not ¼).

**Row 8: Olsson et al. quotes.**
- ICL score: "the loss of the 500th token in the context minus the average loss of the 50th token". Elsewhere the text uses the opposite sign.
- Magnitude: "less than 0.15 nats" before the phase change, "roughly 0.4 nats" after it, "(except for the one layer model where not much in-context learning ever forms)".
- Timing: "approximately 1–3 billion tokens in the small models".
- Warm-up "ramps up over the first 1.5e9 tokens", yet the window "doesn't appear to correspond to a scheduled change in learning rate, warmup, or weight decay".
- Small models: "10,000 steps (~10 billion tokens)".
- Smeared keys: "in-context learning does indeed form for one-layer models (when it didn't before)".

**Row 9: what the Kimi ablation is.**
- It uses a 16-layer model with 653M activated parameters.
- The paper calls its validation set distribution-shifted.
- It has **no pure-linear point**.

**No proposal misreads a number.** The problems are all in how the numbers are used (§3).

---

## 3. Item-by-item verdicts

### 3.1 Consolidated table

- Problem codes are defined in §0.
- Verdicts: KEEP, MOD (modify), REPL (replace), DEFER (to should or later), CUT.
- GPU-h are new hours, written as stated → re-priced (§1.3) → my verdict.

| Item | Claimed to validate | Problems | Verdict | GPU-h |
|---|---|---|---|---|
| PV-M1 / PR-P1(a, b): S1 as frozen | Environment; mixers | S1(a) done (PASS, F1). R-S1-1 cut is moot (actual ≈ 1.75 h). R-S1-3 is moot (d256 0.9938 ≥ d128 0.9892). R-S1-4 targets a harmless message (§1.5). R-S1-5 is superseded by X2 | KEEP R-S1-2 (per-cell report). Close the rest. **X2, then S1(b)** | 0 new (S1(b) ≈ 0.4, inside the S1 budget) |
| PR-P1(c-1): production F_g in MQAR | Production attention learns recall | CF [INF] (Zoology's attention scored 1.000 in every in-distribution cell within 1–3 epochs at every LR and width, so a ≥ 0.95 bar on those cells only catches a broken module); RD (VT-02, VT-09 bf16 varlen) | Free smoke run (≤ 2 min); not evidence | ≈ 0 |
| PR-P1(c-2): GDN-A at d256 ≥ 0.90 | GDN-A geometry | CF: equivalent to live cell ≥ 0.30, while GDN-A at d256 has the state of Zoology's GDN at d128, which scored 0.925 on that cell | MOD: descriptive, beside Zoology GDN d128 | 0 |
| PR-S5: Zoology seed replicate | Basis for ±0.03 | CF (attention seed spread is 0 at ceiling; 2 GDN seeds give 1 df) | CUT | 1.2 → 0 |
| PV-M7: MQAR at d = 64, 3 seeds × 2 arms | "Our GDN = Zoology's GDN" | FB (Zoology's GDN casts q, k, v, β, g to bf16, `mixers/gated_delta_net.py` L245–250, while ours runs fp32 in the same harness; the mid-range regime is chosen precisely so such differences show). The identity question is answered exactly by X2 | DEFER to should, run only if X2 passes and behavioural evidence is still wanted (§3.2c) | 1.7 → 1.3–1.8 → 0 must |
| PV-M2(a): Pythia LAMBADA via lm-eval | Environment | NU (it validates lm-eval); IF (`lm_eval`, `datasets`, `accelerate` absent; installing them changes the pinned env) | MOD → **L2** through our scorer (should) | 1.0 → ≈ 0.1 |
| PV-M2(b): Wang 340M GDN-pure 0.348 | fla GDN with trained weights | Variant unstated; loading a transformers-4.51 checkpoint into fla 0.6.0 is unverified | MOD → part of L2 | incl. |
| PV-M2(b): Transformer 0.353; 3-1 hybrid | Attention with trained weights | NU (fla `Attention`: GQA, no QK-norm, no gate, θ = 1e4); IF (`flash_attn` shim) | CUT | — |
| PV-M2(c): transplant logits | Our mixers with trained weights | RD (VT-09, VT-04); NU (override paths); IF (shim plus untied-head wrapper) | CUT; A2 covers long T on production paths | → 0 |
| PV-M2(d): scorer vs lm-eval per item | EM_tf definition | RD (VT-12 `InductionLM` end to end; VT-07; 5% re-score); IF | CUT | — |
| PV-M3: Tier 0 on the Wang 340M sweep | Tier-0 sensitivity to k | IF (F4); FB (Spearman ≥ 0.8 over 6 points, while published recall is non-monotone at k/24 = 0, 1, 2: 0.356, 0.341, 0.352); NU; Tier 0 only corroborates (D037); Wang call 340M recall "insignificant" (Table 2 caption); "New-hybrid-24-1" provenance | CUT | 0.5 → 0 |
| PR-P2(a): RULER on Wang 1.3B | Our suite detects k | FB (§3.2a: Table 5 contradicts the subset; ≈ 27% joint false alarm even under exact replication); IF (§1.4 RULER list; shim; 33.4 GB of fp32) | CUT | 1.0 → 2–2.7 → 0 |
| PR-P2(b): our generators on Wang 1.3B | Our suite detects k | IF (F4); RD (FEAS-2 already demands S₁₂ − S₀ ≥ 25 pp at 32K on our own models, which is the sensitivity that matters); NU (zero-shot 1.3B models with 2K configs are not TEF-trained 30M models); output may not be used for planning (I-23) | CUT; the answerability part becomes **O1** | 0.7 → 2.4–4 → 0 |
| PV-M4 / PR-S4: nanoGPT ladder | Trainer | RD (VT-13); NU (dropout and train/eval mode do not exist in our model); FB for r3/r4 (a dropout-free model on a 1 MB corpus overfits) [INF]; needs a GPT adapter and a character-level `StageStream` | REPL → **T1**; r0 optional (environment number) | 1.5 → ≈ 0.05 (+ 0.15 for r0) |
| PV-M5 / PR-P3: ratio-loss shape | LM pipeline across k | FB (§3.2b: the verdict depends on how k is mapped; no delta-rule anchor for the 0 → 1 step; PR makes P3-a an audit trigger); P3-c's min-of-4 is biased towards passing; under-priced ≈ 1.9× | MOD → **LM1-lite**: one run, k = 1 at η* (should). RP-2 read descriptively | 1.4 / 3.2 → 2.7 / 6.0 → 0.6 |
| PV-M6 / PR-P4b: Olsson smeared-key toy runs | Attention correctness; premise of rival E1.6 | RD (VT-02, VT-09); SD (0.1–0.2B tokens vs ≈ 10B, and Olsson's timing sits inside a 1.5e9-token warm-up); FB (no 1-layer ICL value in the text to anchor the 1-layer thresholds); decision-irrelevant (X-gc12 runs unconditionally, D005); under-priced | Measurement → **I1** (must). Toy runs DEFER | 1.9 / 0.5 → 2.8–4.3 / 0.8–1.3 → 0.1 |
| PR-P4a: ICL during training | Budget past the phase change | The 50-step snapshot cadence needs an S0 trainer change, while Stage-A milestones at {50M, 100M, 200M, 400M} tokens and S already exist. The 0.15-nat bar is a transformer number: a GDN lowers late-position loss without an induction phase change. C1 and C3 have no FEAS runs | MOD → **I1** | 0.1 |
| PV-S1 / PR-S1: Jelassi copy task | Learning-speed rival | SD (the gap is shown for Mamba with S = 32; GDN-A's state, 3 × 64 × 128 = 24,576 entries per layer at D384, dwarfs a ≈ 1.4 kbit string of 300 letters [INF]); FB (PR concedes a ratio < 1 "is not a bug signal"); RD (the TEF dose ladder and still-closing flag measure E1.7 in place) | CUT | 2–3 / 1.5 → up to 4.4 → 0 |
| PV-S2 / PR-S2: length behaviour on FEAS checkpoints | Position handling | Descriptive. Press et al.'s numbers (θ = 1e4, L = 512 / 1,024) do not transfer to θ = 5e5 with QK-norm | KEEP as a descriptive audit (should); PV-S2(ii) is a useful trigger | 0.3 |
| PV-S3: our generator vs RULER | Generator semantics | CF ("≤ 10 pp *or* an explained difference"); IF | CUT | 0.5 → 0 |
| PV-S4: Wang 1.3B 24-1 LAMBADA | A second transplant | IF (shim for its single attention layer); 5.8 GB of fp32 | CUT | 0.3 → 0 |
| PR-S3: GPT-2 LAMBADA | Scorer identity | IF (weights not cached, F5); the target is a community run on an old harness | MOD → L2 with Pythia (a published table) | ≈ 0.05 → incl. |
| PV-N1–N4, PR-N1–N5 | Environment; extras | NU; IF (compile, NCCL and GPT-2 shards; `mamba_ssm`); figure-only anchors | CUT | 4–8.5 → 0 |
| PV gate G-RV | — | Puts M2, M4 and M6 (≈ 4.4 h, plus lm-eval, a shim and two adapters) on FEAS's critical path, mostly re-testing VT-09 and VT-13 | REPL (§5) | — |

### 3.2 Notes on the decisive cases

**(a) PR-P2 cannot pass reliably even on a correct pipeline, and it duplicates FEAS.**

- **The subset contradicts the source.** P2 uses `niah_single_1/2/3` plus `vt`, which weights Table 5's Singlevalue and VT columns 3:1.
  - Transformer: (3·0.559 + 0.066)/4 = 0.436.
  - GDN-3-1: (3·0.875 + 0.024)/4 = 0.662.
  - The difference is −0.23 [INF].
  - For "S(Transformer) − S(pure) ≥ +0.03" to hold, pure GDN would have to sit ≥ 0.26 below the 3-1 hybrid on single-needle tasks.
- **Even exact replication fails about a quarter of the time.** Suppose Wang's averaged gaps held exactly on the subset (+0.080, −0.015, +0.067). With P2's own SE of 0.025 per difference, the three criteria pass jointly with probability ≈ Φ(1.6) × [Φ(2.2) − Φ(−1.0)] × Φ(1.48) ≈ 0.945 × 0.827 × 0.931 ≈ 0.73 [INF]. That is ≈ 27% false alarms.
- **The engineering is large.**
  - RULER needs 4 environment changes plus a patch.
  - Every hybrid needs a `flash_attn` shim.
  - The download is 33.4 GB of fp32.
  - Our generators reject the tokenizer (F4).
  - `ForwardOnlyAdapter` re-computes a forward per query, so P2(b) alone is ≈ 2.4–4 GPU-h, not 0.7.
- **The question is already answered more directly.** P2's stated purpose is "if our generators cannot discriminate 3:1 from pure on these models, our evaluation design is insensitive to the very manipulation we study". FEAS already answers that on our own models, before S3 and S4, with a hard gate (FEAS-2: S₁₂ − S₀ ≥ 25 pp at 32K). Intermediate-k sensitivity is the research question itself, not a validation.

**(b) PV-M5 / PR-P3: the anchors are Mamba-family, and the criteria can fire with no bug.**

- **PV-M5-P2's verdict depends on an unstated mapping.** The criterion is "range over k ∈ {1, 3, 6} ≤ 0.02 nats", anchored on Mamba-2's range over 1–11 blocks (0.014 nats). Mapped onto Mamba-2's 48 blocks:
  - **by count**, the range is ln(8.38/8.26) = 0.014 nats: pass;
  - **by attention fraction** (4, ≈ 12 and 24 of 48 blocks), it is ln(8.50/8.29) = 0.025 nats: fail.

  So a pipeline that reproduced Mamba-2 exactly could still trip "the RP-2 audit before S4".
- **No delta-rule sweep anchors the 0 → 1 step on loss:**
  - Kimi's KDA ablation has no pure-linear point.
  - On Kimi's in-distribution training PPL, 15:1 (1 of 16 layers attention) sits 0.012 nats above the best ratio.
  - On Kimi's distribution-shifted validation set, 15:1 is *worse* than pure MLA (5.82 vs 5.77).
  - For GDN itself, Wang's Table 6 puts pure GDN at the top of the LM averages (0.487 vs 0.468–0.481).
- **What follows.** "The first attention layer gives most of the loss gain" is established for Mamba-family stacks, not for delta-rule stacks. PR's P3-a audit trigger could send the team after a non-bug.
- **What I keep: LM1-lite, one run of k = 1 at η* on the S3 recipe.** The reason is engineering, not literature:
  - the core runs k = 1, but S3 never trains it, so k = 1's LR and stability are untested before G5;
  - the run also makes RP-2's "largest step at 0 → 1" clause evaluable (descriptive);
  - cost ≈ 0.6 GPU-h, contended.

**(c) PV-M7 vs X2 (I disagree with critique A on priority).**

- **What M7 tests, given X2.** X2 copies weights and compares forward outputs and gradients. If X2 passes, our mixer and Zoology's compute the same function and gradient. Zoology re-initialises every `nn.Linear` itself, and fla-specific tensors keep fla's own init in both. So M7 can then only measure precision and RNG effects. Critique A itself calls that "a recipe difference worth reporting, not a bug".
- **What M7 tests if X2 fails.** The wrapper must be fixed first, and M7 would be wasted.
- **Conclusion.** M7 never localises a defect that X2 cannot. Its one unique output is a seed spread for G-RP(b), a gate that is a smoke test anyway (§1.1). Make it a conditional should item.

**(d) PV-M2 / PR-S3 / PV-M3 → L2, a LAMBADA pair through our own scorer, with no lm-eval.**

- **Pythia half.** Pythia-160M at a pinned `revision`, target 0.328 ± 0.007, tolerance ±0.015; optionally Pythia-70M (0.185 ± 0.005). This anchors our scorer and adapter to a published table.
- **Wang half.** Wang's 340M GDN-pure through fla 0.6.0, target 0.348, tolerance ±0.02, reporting both the openai and standard variants because the paper does not say which.
  - It needs no shim.
  - It is the only check that fla 0.6.0's GDN layer (which `GDNMixer` subclasses) computes the same function as the fla version that trained published weights.
  - VT-04 checks the kernel against our own fp64 recurrence; VT-09(2) checks fla against fla.
- **Diagnosis.** If Pythia passes and Wang fails, the cause is fla drift or the variant.
- **Cost.** ≈ 0.1 GPU-h, ≈ 2 GB of downloads and a small HF wrapper. Items are written directly with each model's tokenizer, so `TokenSpace` is not involved.
- **Tier.** Should, not must: VT-12's oracle already covers alignment.

**(e) PV-M4 → T1.**
- What r2 claims to isolate is already in VT-13.
- A step-matched trace against a reference loop *on our model* also covers clipping and z-loss, and its tolerance is ≈ 1e-6 rather than 2·SD.
- r0, the original nanoGPT config compared with 1.4697, is an environment number only. S1 has already shown that this environment trains reference models.

**(f) PV-M6 / PR-P4 → I1.**
- Measure the ICL score and LM-4(b) gap-0 copy loss on checkpoints we already produce.
- No 50-step cadence is needed. Stage-A milestones give 5 points per FEAS run, and the last two retained S3 checkpoints give a plateau check.
- The toy runs are deferred: SD, unanchored thresholds, no decision depends on them, and they are under-priced.

---

## 4. Important validations that are missing (ranked by risk covered per GPU-h)

| ID | Check | Why it matters | Pass / fail (pre-register) | GPU-h | Natural owner |
|---|---|---|---|---|---|
| **O1** | `InductionLM` answerability at every confirmatory 16K/32K cell: R1 D ∈ {3, 15}; R-load up to N = 1,024 / 2,048; R3; R-dist | FEAS picks the scale (D512 adds ≈ 58 GPU-h) and L_ext from 32K item scores. Nothing yet shows those items are answerable at length: VT-12 checks structure at 32K and answerability only at 2K (F7) | EM_tf = 1 and logp_gold = 4·log 0.9 on every R1/R-load/R3 query; M-hop fails. Any miss is an item bug or an oracle key collision; inspect it before FEAS | 0 (CPU, minutes) | S0 eval |
| **X2** | Weight-copy equivalence, Zoology ↔ ours (critique A's X2). GDN with Zoology's bf16 casts reproduced; MHA with biases zeroed and dropout off | Makes S1(b) interpretable; replaces M7 and S5 as the identity check | Outputs and gradients equal within bf16 tolerance (GDN) or fp32 tolerance (MHA) | ≈ 0.01 | S1 |
| **X1** | fla Triton `causal_conv1d` vs `causal_dwconv_reference`: forward and backward, `cu_seqlens`, `initial_state`, fp32 and bf16. Plus F_gc packed-vs-separate (the rest of VT-03) | GDN-A and F_gc depend on this kernel; VT-09(2) compares fla with fla (F8). Also critique A's X1 | Max abs diff ≤ 1e-5 (fp32), ≤ 1e-2 relative (bf16); exactly zero leakage across documents | ≈ 0.01 | S0 models |
| **A2** | VT-09 at long T: C12 vs Qwen3 (SDPA) at 4K in fp32 and at 16K/32K in bf16; C0 vs fla's LM likewise; one hybrid (C3) packed-vs-separate at 32K in bf16 | The primary cell is 16K–32K, but the full model is never compared with a reference beyond 256 tokens. Replaces PV-M2(c) at the lengths we actually use | 1e-3 relative (fp32); the existing 5e-2 (bf16) | ≈ 0.05 | S0 models |
| **T1** | Our trainer vs a ≈ 60-line reference AdamW loop on our model: fp32, deterministic, same batches, clipping and z-loss on, 50–200 steps | Replaces M4. VT-13 checks components and config values, not the update end to end | Loss and gradient norm agree per step to ≈ 1e-6 relative; equal parameter checksum at the end | ≈ 0.05 | S0 training |
| **A3** | GPU-fault hygiene: (i) annotate each run with the `journalctl -k` Xid lines from its time window; (ii) one GPU queue for all agents; (iii) profile the longest kernel in one 32K Stage-B step and one 32K prefill, for C0 and C12; (iv) give infrastructure failures their own status in protocol §8.1 | Two Xid 8 hangs and one Xid 31 MMU fault in one evening, and the clean re-run points at contention (§1.2). Infrastructure failures correlated with k would become missing data correlated with k | No kernel > 1 s, far below the 7 s RC notify timeout (my threshold [INF]); every run that overlaps an Xid event is flagged | ≈ 0.1 | Orchestrator; S0 training |
| **G0-T** | A clean G0 throughput measurement (CARLA as the only foreign load), then re-price D041 and every LM-training item | F3: 44–58% of plan at D384 | As G0 | ≈ 0.1 (inside S0's 3.0 h) | S0 |
| **I1** | Olsson's ICL score (L@50 − L@500 on held-out documents ≥ 512 tokens) and LM-4(b) gap-0 copy loss, at existing Stage-A milestones and S3 final checkpoints | Shows whether induction formed and the budget passed the transition; a cheap Olsson anchor | k = 12: plateau (\|ΔI\| < 0.02 between the last two points) and second-copy loss ≪ first-copy loss. k = 0 descriptive | ≈ 0.1 | Eval |

**Gaps accepted, with no cheap anchor.**
- No small public checkpoint was trained at ≥ 16K.
- FEAS, G-REF, A2 and O1 must carry the long-context assurance.

---

## 5. The cheapest set that covers the most risk

| Tier | Order | ID | Check | Risk covered | Literature anchor | New GPU-h | Engineering | Replaces |
|---|---|---|---|---|---|---|---|---|
| Must | 0 | — | §6 decision rows | Audit trail; forking paths | — | 0 | Text | R-S1-1…5; PR §2.1 |
| Must | 1 | O1, X1, A2, T1 | Deterministic tests in S0's own files | Items at length; conv kernel; full model at long T; trainer | — | ≈ 0.1 | 4 tests | PV-M2(c), M4; P2(b)'s answerability role |
| Must | 2 | X2 → S1(b) | Identity test, then S1(b) as frozen | Mixer identity in Zoology's harness | Zoology (S1) | ≈ 0.01 (+ S1(b) ≈ 0.4, already budgeted) | 1 test | PV-M7, PR-S5 |
| Must | 3 | A3, G0-T | Fault hygiene; clean throughput | Infrastructure; budget | — | ≈ 0.2 | Registry annotation; profiler script | (new) |
| Must | 4 | I1 | ICL and copy loss at existing milestones | Induction formed; training budget | Olsson: < 0.15 before, ≈ 0.4 nats after | ≈ 0.1 | Eval script | PV-M6(a, d); PR-P4a |
| Should | 5 | L2 | LAMBADA pair through our scorer | Scorer on real LMs; fla GDN semantics | Pythia Table 7: 0.328 ± 0.007 (0.185 ± 0.005). Wang Table 2: 0.348 | ≈ 0.1 (+ ≈ 2 GB download) | HF wrapper; item writer | PV-M2(a, b, d); PR-S3 |
| Should | 6 | LM1-lite | k = 1 at η* (S3 recipe) | k = 1 LR and stability before G5; RP-2's 0 → 1 clause | Mamba-2 Table 2 (direction only) | ≈ 0.6 | None | PV-M5; PR-P3 |
| Should | 7 | X4 | Known-answer recall in our harness: 2-layer `AA` vs `--` on no-filler R1/R-load at 1K, ≤ 50M tokens (critique A's X4, cheaper) | Synthetic stream → loss mask → model → scorer, before FEAS's 9.8 h | — | ≈ 0.1–0.3 | Small run config | (new) |
| Should | 8 | R | Length behaviour on FEAS checkpoints | Descriptive audit | Press Tables 2–3 (context only) | ≈ 0.3 | Eval | PV-S2; PR-S2 |
| Should | 9 | r0 | nanoGPT original config | Environment number | 1.4697 | ≈ 0.15 | None | PR-S4 |
| Optional | — | M7 | Only if X2 passes and a behavioural check is still wanted | Training behaviour in a mid-range regime | Re-run numbers | ≈ 1.3–1.8 | None | — |
| Optional | — | A4 | Qwen3-0.6B-Base long-context positive control | LM-elicitability at 16K/32K | — | ≈ 0.3 + download | Positions-only logits (a 32K × 151,936 fp32 logit matrix is ≈ 19.9 GB); tokenizer compatibility unverified | — |

**Totals.**

| Plan | New must-tier GPU-h (stated → re-priced) | Should tier | New packages or environment changes |
|---|---|---|---|
| PV | 8.0 → ≈ 10–12 | 3.1–4.1 | lm-eval (plus `datasets`, `accelerate`, …), `flash_attn` shim, untied wrapper, nanoGPT adapter, character-level `StageStream`, foreign-tokenizer generators |
| PR | 5.8 → ≈ 11–15 | 3.3 | RULER environment (`nltk`, essays, LFS words, `accelerate`, fla patch), shim, foreign-tokenizer generators, 50-step snapshots, ≈ 33 GB of downloads |
| **This set** | **≈ 0.3–0.5** (+ S1(b) ≈ 0.4, already budgeted) | **≈ 1.3–1.5** (+ optional M7, A4) | None (`huggingface_hub` downloads only, for L2) |

**Gate replacing G-RV.** FEAS starts when all three hold:
1. G0 passes, with O1, X1, A2 and T1 added as VT tests, and with a clean throughput number.
2. G-RP(b) is reported, with X2's result and the per-cell table beside it.
3. A3's per-run Xid tagging is live, and the 32K kernel profile shows no kernel over 1 s.

**Non-blocking:**
- I1, LM1-lite and X4 are audit triggers or descriptive;
- L2 is informative only.

**Never gating:**
- shape criteria anchored on Mamba-family sweeps (M5/P3);
- the smeared-key toy runs (M6/P4b);
- anything measured on Wang's checkpoints.

---

## 6. Rows to log in `decisions.md` now (for the PI)

1. **S1(a) close-out.**
   - PASS: 16 of 16 cells, ≈ 1.75 GPU-h in total.
   - `gated_delta_net-d128-lr1.0e-02` completed on attempt 3 (22:13–22:26) after two Xid 8 crashes (18:16, 18:25), with an Xid 31 at 18:03.
   - Attempt 3 was launched after the verdict, by an uncommitted change to `run_queue.py` (6 attempts, per-epoch resume).
   - The verdict is the same with and without that cell (best LR 1e-3 at both widths).
   - Commit `run_queue.py`.
2. **S1 deviations.**
   - Exact micro-batch-64 gradient-accumulation shim (all 16 runs).
   - `head_first` shim.
   - Per-epoch resume. No final run used it: `resumed_at_epochs` is empty or absent in every run.
   - Update `s1_reproduction.md`, which still says "BLOCKED".
3. **Close resolved items.**
   - R-S1-1: the cut is moot. Record the lesson about forecasting from a single run.
   - R-S1-3: moot.
   - R-S1-4: the message comes from Zoology's Mamba import probe; retarget it as X1.
   - PR-S5: dropped.
   - Keep R-S1-2: the per-cell table, with "±0.03 overall = ±0.21 on 1024/256" stated.
4. **S1(b) pre-registration.** X2 first, then the 4 frozen runs plus the 2 exploratory runs. G-RP(b) stays as frozen, with X2's result and the per-cell results reported beside it.
5. **Annotation to the protocol §2 prior** (no design change).
   - Wang's released "r-1" models have one attention layer per r layers, so 3-1 is 8/24 = ⅓.
   - Attention head count varies with ratio.
   - The GDN configs have `max_position_embeddings` 2,048; the paper's Limitations say 4,096.
6. **Tooling policy.**
   - No new packages in the frozen `.venv` before G5.
   - Anything that needs them (lm-eval, RULER) goes in a separate venv and is labelled "original code + N shims".
7. **GPU policy.** All agents' GPU jobs go through one queue, and runs are tagged with any Xid events in their time window.

---

## 7. Research practices to add (engineering side)

These add to PV §1, PR §1 and critique A §9, which I otherwise endorse.

1. **Price from measured throughput.**
   - Use the contended throughput of the slowest config at the actual shape.
   - Model early stopping as bimodal.
   - Pre-declare a collapse-abort rule for harness runs, for example: stop when validation accuracy < 0.05 for 3 epochs after epoch 2. It must be declared before running, since it changes compute but not the best-over-LR outcome.
2. **Run a feasibility pre-flight before adopting any reproduction:**
   - packages present;
   - data files present (not LFS pointers);
   - checkpoint dtype and size against the 0.44 memory cap;
   - tokenizer accepted by `TokenSpace`;
   - `flash_attn` dependence;
   - downloads fit the disk.

   Each of P2, M3 and PR-S3 fails at least one of these.
3. **Classify every check before it runs:**
   - smoke: it can only catch a gross failure;
   - calibration: it has a stated minimum detectable defect and false-alarm rate;
   - phenomenon: directional, and a failure is not attributable to a bug.

   Write down both numbers for every pass criterion. For example, G-RP(b)'s minimum detectable defect is 0.21 on the live cell, and P2(a) has ≈ 27% false alarms.
4. **Ask identity questions with weight-copy tests.** Use training comparisons only for questions that weights cannot answer.
5. **Declare re-run and retry policy before any verdict is read.** Record it in the decision log, not only in an uncommitted script.
6. **Run one GPU job at a time, through one queue,** for anything that is timed, gated or longer than 15 minutes. Tag every run with the Xid events in its time window.
7. **Pin everything:**
   - HF `revision` SHAs (Pythia-160M `50f5173`; the Wang repo SHAs);
   - tokenizer files;
   - shim code hashes.
8. **Evaluate public checkpoints inside their training window** (2,048 for the Wang GDN configs) unless extrapolation is what is being studied.

---

## 8. Where I disagree with critique A

- **M7.**
  - A: keep, top priority.
  - B: a conditional should item (§3.2c). Once X2 passes, M7 can only reveal precision or RNG differences.
- **X0 soak (≈ 1 GPU-h).**
  - B: the lean A3 (≈ 0.1) is enough.
  - The lr 1e-2 cell completed cleanly when it shared the GPU only with CARLA. FEAS, with auto-resume plus Xid tagging, gives the same information while doing useful work.
  - Escalate to A's X0 if an Xid recurs.
- **M2-lite.**
  - A: lm-eval in a separate venv.
  - B: our scorer directly. VT-12 already validates alignment, and the external anchor we need is the published number, not lm-eval's code path.
- **LM-loss runs.**
  - A: drop all new runs.
  - B: keep one k = 1 run at η*, because S3 never trains a core config.
- **Answerability.**
  - A's X5 uses a public long-context model.
  - B: the existing oracle at 32K first (O1, zero GPU-h); a public model only as an optional extra (A4).
- **Agreement:**
  - drop P2, the new M5/P3 runs, Jelassi and the Nice tiers;
  - replace M4 with a reference-loop trace;
  - X1 and X2;
  - S1 is a smoke test.

---

## 9. Sources opened in this review

| Source | Used for | Access |
|---|---|---|
| [2507.06457v2](https://arxiv.org/html/2507.06457v2): Tables 2, 3, 5, 6, 7; App. D; §4.3; Limitations; training setup | §2 rows 1–5; §3.2(a, b) | read-section (raw HTML, parsed with `python -I`) |
| [2405.21060v1](https://arxiv.org/html/2405.21060v1) §9.2.3, Table 2 | §2 row 6; §3.2(b) | read-section |
| [2304.01373v2](https://arxiv.org/abs/2304.01373) App. G.1, Tables 7–8 | §2 row 7; L2 | read-section (PDF text) |
| [Olsson et al. 2022](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html): Key Concepts, Argument 1, smeared keys, Model Details | §2 row 8; I1 | read-section |
| [2510.26692](https://arxiv.org/html/2510.26692) §5.2, Table 1; scaling-law table | §2 row 9; §3.2(b) | read-section |
| HF API: `config.json`, file trees and SHAs for m-a-p 340M {pure-baseline, hybrid-3-1, New-hybrid-24-1}, 1.3B {pure, 24-1, 12-1, 6-1, 3-1} and the Transformers; collection listing; 1.3B-pure tokenizer files; `EleutherAI/pythia-160m`; `EleutherAI/lambada_openai` | §1.4; F4, F6 | code-inspected |
| `third_party/zoology` @1ad20d1: `mixers/gated_delta_net.py` L238–256, `train.py` L183–189, `model.py` L265–277, `mixers/mamba.py` L10–15 | §1.5; §3.2(c) | code-inspected |
| `third_party/ruler` @db43f92: `niah.py`, `json/`, `pred/model_wrappers.py`; `third_party/nanoGPT` @3adf61e: `train.py`, README; `third_party/modded-nanogpt` @4ea6b93 track 3 | §1.4 | code-inspected |
| fla 0.6.0 (installed): `layers/attn.py` L20–75 | §1.4 | code-inspected |
| `src/hybridlc`: `eval/tokens.py`, `eval/adapter.py`, `eval/devtools.py` (`InductionLM`), `models/attention.py` (conv paths), `models/layers.py` (RoPE), `models/linear.py`; `tests/` (VT-02, 04, 09, 12, 13 and the loss tests); `.pytest_cache/lastfailed` | §1.5; F4, F7, F8 | code-inspected (read only) |
| `experiments/s1_zoology/{runs, logs, s1a_results.json, s1a_table.md, cut_decision.json}`; `scripts/s1_zoology/{run_queue, run_one, our_mixers, test_s1_resume}.py`; `runs/bench_train_step/*.jsonl` | §1.1–1.3 | MEAS-S1, MEAS-S0 |
| `nvidia-smi -q`, `journalctl -k`, `ps`, `git status` | §1.2 | MEAS-R |
| Protocol v1.0 (§13, §16, §17, §18); `decisions.md`; `STATUS.md`; `INTERFACES.md`; `competing_explanations.md`; `literature_review.md` (Kimi entry); `s1_reproduction.md`; both proposals; `critique_A.md` | Context | read-full (proposals, §13–§18) / read-section (rest) |

## 10. Not verified by me

**GPU and test state:**
- The root cause of the Xid 8 and Xid 31 events, and which process caused the Xid 31.
- Throughput on a clean GPU.
- Whether the three tests in `.pytest_cache/lastfailed` (18:34) now pass. `STATUS.md` says the VT tests pass, but I ran no tests.

**Estimates that rest on inference:**
- The small-model and 1.3B-model time estimates in §1.3 are FLOP-scaled [INF].
- That Mamba-2's 48-layer stack has no MLPs, and that this is why its "24 blocks" is not all-attention [INF from §9.2.3 and the Table 3 variant list].

**Not checked:**
- Whether fla 0.6.0 loads the Wang 340M GDN-pure checkpoint cleanly. L2 would find out.
- Whether Qwen3-0.6B-Base's tokenizer passes `TokenSpace` (needed for A4 only).
- Whether HF `main` of Pythia-160M equals the weights behind Table 7. Pin `revision`.
- The GDN paper's S-NIAH table (cited by PR) and Waleffe's figure values. I did not re-open them, and nothing in this critique depends on them.

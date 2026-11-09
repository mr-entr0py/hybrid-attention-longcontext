# Critique B: engineering feasibility and cost of the two reproduction proposals

**Reviewer:** adversarial reviewer B (engineering feasibility and cost). **Date:** 2026-11-09, written 18:20–18:50 IST while the S1 queue was running.

**Proposals reviewed:**
- `proposal_pipeline-validation.md` ("PV"; items M1–M7, S1–S4, N1–N4);
- `proposal_phenomenon-replication.md` ("PR"; items P1–P4, S1–S5, N1–N5).

**What I did:**
- Read both proposals and protocol v1.0 (§2–§20, mainly §13, §16 and §17).
- Read `INTERFACES.md`, `s1_reproduction.md`, `decisions.md`, the S1 run JSONs and logs, the S0 test files and the relevant third-party code.
- Re-opened the primary source for 11 target numbers (§2).

**What I did not do:**
- I ran no GPU job.
- I edited nothing except this file.
- My only system queries were:
  - `nvidia-smi`;
  - one driver-API attribute read (`cuDeviceGetAttribute`, which creates no CUDA context);
  - HF Hub metadata (the tree API and `config.json`);
  - arXiv and transformer-circuits HTML, downloaded to the scratchpad and parsed with `python -I`.

**Tags.**
- [read-section]: primary source opened by me in this step.
- [code-inspected]: a file or HF config read by me.
- [MEAS-S1]: read by me from `experiments/s1_zoology/` between 18:18 and 18:40. These are interim, single-seed numbers.
- [MEAS-R]: observed by me in this review, without using the GPU.
- [INF]: my own inference or arithmetic.

No number below is read off a figure.

---

## 0. Bottom line

1. **The must tiers ask for ≈ 14 GPU-h of new work and a lot of new engineering before FEAS.**
   - New must-tier GPU time: PV ≈ 8.0 h, PR ≈ 5.8 h, partly overlapping. Should tiers add 6–8 h more.
   - New engineering on the path to FEAS:
     - lm-eval and RULER installs;
     - a `flash_attn` shim and an untied-embedding wrapper;
     - foreign-tokenizer generators;
     - a nanoGPT-in-our-trainer adapter;
     - a 50-step snapshot cadence.
   - Most must-tier items fall into at least one of these categories (§3):
     - they **cannot fail**, because they sit at ceiling;
     - they **can fail with no bug in our code**;
     - they **test code paths production does not use**;
     - they **re-test what S0 already tests**.
2. **A much cheaper set covers the same risks, plus two that both proposals miss** (§5):
   - ≈ 0.4–0.8 GPU-h of new GPU work;
   - one LM item of ≈ 1.05 h (D384) or 1.4 h (D512);
   - an optional should tier of ≈ 1.6–2.1 h.
3. **Urgent: decide before any GDN d256 result is read** (§6). The queue reaches d256 at about 18:45–18:50.
   - GDN d128 at lr 1e-2 has **failed twice** with `cudaErrorLaunchTimeout`.
   - The GPU's kernel watchdog is enabled.
   - The S1 cut that both proposals recommend **cannot run as coded**.

**Findings not in either proposal:**
- **F1.** Watchdog crashes in S1, with `KERNEL_EXEC_TIMEOUT` = 1 on this GPU (§1.1).
- **F2.** The "Need to install causal_conv1d" line is harmless. This closes PV R-S1-4 and the STATUS note without using the GPU (§1.2).
- **F3.** Zoology's GDN is not fla's layer, so the null hypothesis of PV-M7 is wrong (§1.3).
- **F4.** Wang et al.'s own per-subtask Table 5 contradicts PR-P2(a)'s RULER subset (§3.2a).
- **F5.** PV-M5's flatness criterion conflicts with its own Mamba-2 anchor (§3.2b).
- **F6.** Most reproduction tooling is not installed. The Wang 1.3B checkpoints are fp32 (≈ 33 GB for the sweep), and fp32 scoring does not fit under the memory cap (§1.5).
- **F7.** Nothing in either suite checks our models at the 16K–32K lengths of the primary cell (§4).

---

## 1. Facts established in this review

### 1.1 S1 status and the kernel watchdog [MEAS-S1, MEAS-R]

**The lr 1e-2 run fails reproducibly.**
- `gated_delta_net-d128-lr1.0e-02` crashed on both attempts:
  - attempt 1 at epoch 15 (t ≈ 337 s);
  - attempt 2 at epoch 25 (t ≈ 525 s).
- Both raised `torch.AcceleratorError: CUDA error: the launch timed out and was terminated`, reported at the first host sync (`zoology_compat.py` L195). The faulting kernel is therefore unknown.
- The queue logged `FAILED … after 2 attempts` at 18:25:42.

**The watchdog is on.**
- `nvidia-smi -q` shows "Display Attached: Yes" and "Display Active: Enabled". Xorg, gnome-shell and gnome-remote-desktop all run on this GPU.
- `CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT` = 1, so the driver's run-time limit on kernels is enabled.

**The lr 3.2e-2 run diverged.** It scored 0.9686 at epoch 0 and ≈ 0.0003 from epoch 1 on. Nothing stops a collapsed run early, so it trains all 32 epochs.

**Other GPU jobs ran beside the queue.**
- At 18:39 there were three project compute processes: 2.5 GB (S1), 3.9 GB and 0.4 GB.
- At about 18:24 another agent's ad-hoc run (`scratchpad/s1new/run_one.py --run-id gated_delta_net-d128-lr1.0e-02 --max-epochs 3`) ran next to the queue.
- GDN d128 epoch time varied with contention: 21 s, 31 s and 35.5 s per epoch in different runs. Every timing used by the cut rule and by both proposals' budgets is affected.

**Consequences.**
- **(a) The cut cannot run as coded.** `s1_analysis.best_lrs()` raises `ValueError(… only 3 of 4 LRs complete)` (L59–60), so the cut that PV R-S1-1 and PR §2.1 recommend fails. The protocol's own precondition ("the first d_model completed at 4 LRs") can no longer be met as written.
- **(b) The missing LR needs a rule.** `best_over_lr()` silently uses the 3 completed LRs (`n_lrs` = 3), so G-RP(a) will still be computed. The best is lr 1e-3, at 0.9892.
- **(c) Crashes in main runs are not defined.**
  - The main runner resumes crashed runs: in `training/queue.py`, `crashed` triggers auto-resume, with `max_crash_retries` = 2 before the run is marked `failed`.
  - Protocol §8.1 defines permanent failure only for divergence.
  - If watchdog crashes were correlated with k (for example, through attention-heavy 32K steps), infrastructure failures would become missing data correlated with k. Nobody has measured kernel durations at 32K.
- **(d) Neither proposal budgets for retries.** So far S1 has had 2 crashes in 9 queue attempts (11 if the two smoke runs are counted).

### 1.2 The "Need to install causal_conv1d" message is harmless [code-inspected]

- The message comes from Zoology's `_compute_state_size` (`zoology/model.py` L270–277).
  - That function imports `zoology.mixers.mamba` and `zoology.mixers.mamba2` inside bare `try/except` blocks, in order to run an `isinstance` test.
  - Both modules run `assert 0, print("Need to install causal_conv1d …")` at import (`mamba.py` L14, `mamba2.py` L26).
  - The message prints, the `AssertionError` is swallowed, and the block class is set to `None`. This explains why the message appears twice per run.
- `BaseConv` uses Zoology's own pure-torch `ShortConvolution` and `ImplicitLongConvolution`.
- **No fallback path is involved in S1's attention or GDN models.** PV R-S1-4 and the STATUS.md note can be closed with no test.

### 1.3 Zoology's GDN is not fla's `GatedDeltaNet` layer [code-inspected]

- `zoology/mixers/gated_delta_net.py` is Zoology's own layer wrapped around fla's `chunk_gated_delta_rule`.
- **Precision.** It casts q, k, v, β and g to **bf16** before calling the kernel (L246–250). Zoology trains with no autocast.
- **Weight decay.** `Trainer.fit` builds `AdamW(model.parameters(), weight_decay=0.1)` (`train.py` L185–189). This applies decay to *every* parameter, including `A_log`, `dt_bias` and the norm gains, and ignores `_no_weight_decay`.
- **Early stopping** triggers when `valid/accuracy > 0.99` (`config.py` L133–134).
- **Consequences.**
  - S1(b) and PV-M7 test our module inside Zoology's optimiser and precision regime. They do not test our trainer's weight-decay groups (§7.1) or our bf16-autocast path.
  - "Ours vs Zoology" differs by design, through fp32 vs bf16 kernel inputs, even when our code is correct.

### 1.4 How close to ceiling G-RP is [MEAS-S1, INF]

**What "overall accuracy" is.** Zoology's overall `valid/accuracy` is the unweighted mean of the 7 per-config accuracies. Check on GDN d128 at lr 1e-3:

(1 + 0.999875 + 1 + 1 + 1 + 0.9998125 + 0.92501)/7 = 0.98924, against a reported 0.9892427.

**Measured accuracies.**

| Run | Overall accuracy | Notes |
|---|---|---|
| Attention d64, all 4 LRs | 0.99998 after epoch 0 | Every cell ≥ 0.99988; early-stopped at epoch 0 |
| GDN d128, lr 1e-3 | 0.945 after one epoch | 1024/256 cell 0.925 at the end; the other six cells ≥ 0.9998 |
| GDN d128, lr 3.2e-3 | 0.965 after one epoch | 1024/256 cell 0.913 at the end |
| GDN d256 smoke | 0.987 after epoch 0 | 1024/256 cell 0.911 |

**What follows.**
- With six cells pinned at ceiling, G-RP(b)'s ±0.03 tolerance on overall accuracy **allows ±0.21 on the only cell that moves.**
- G-RP(a)'s 0.95 threshold is reached by GDN after about one epoch, and by attention after one epoch at every LR.
- S1 therefore catches only gross failures: no training, NaN, or wrong wiring.
- At ceiling, MQAR is also blind to bugs that make the model *too good*. A causal-mask leak raises accuracy rather than lowering it; VT-02 covers that case.
- **S1 is a smoke test, not a calibration.** That is worth having, but should be labelled as such.

### 1.5 Tooling and checkpoint facts [code-inspected]

**Packages missing from `.venv`:** `lm_eval`, `datasets`, `accelerate`, `nltk`, `html2text`, as well as `flash_attn`, `causal_conv1d` and `mamba_ssm`. Present: `wonderwords`, `transformers` 5.22.2, `requests`.

**RULER @db43f92 will not run without changes:**
- `niah.py` imports `nltk.sent_tokenize` (L42).
- It reads `json/PaulGrahamEssays.json` (L85). That file is absent and must be downloaded with `download_paulgraham_essay.py`.
- `english_words.json` is a Git-LFS pointer.
- The HF wrapper calls `AutoModelForCausalLM.from_pretrained(…, device_map="auto")`, which needs `accelerate`, and never runs `import fla`.
- Nothing in this repo has tested RULER's generation path on fla 0.6.0 models under transformers 5.22.2. VT-09 tests forward logits only.

**fla 0.6.0 will not build the hybrid or Transformer checkpoints.** `fla/layers/attn.py` raises `ImportError` in `Attention.__init__` when `flash_attn` is missing (L71–72). Those checkpoints therefore cannot even be instantiated without injecting a fake `flash_attn` module. Any result on them is "fla plus our shim", not original code.

**The Wang 1.3B checkpoints are fp32, not bf16.** File sizes from the HF tree API (2026-11-09):

| Checkpoint | Size (fp32 safetensors) |
|---|---|
| GDN pure | 5.87 GB |
| GDN 24-1 | 5.83 GB |
| GDN 3-1 | 5.60 GB |
| Transformer | 4.59 GB |

- The six-model sweep is ≈ 33 GB. PR-P2 assumed "6 × ≈ 2.7 GB (bf16)".
- The fp32 weights (≈ 5.9 GB) leave about 1 GB under the 0.44 cap (≈ 7 GiB). PR-P2(b)'s "fp32 single-pass" scoring at 4K therefore does not fit. Use bf16, or change the cap with a decision row.

**The training context of the Wang models is unclear.**
- The GDN configs have `max_position_embeddings` = 2,048 (340M and 1.3B); Transformer-1.3B has 4,096.
- The v2 Limitations section says "a 4,096-token context window".
- So U1 is still open, and 4K evaluation cells may be RoPE extrapolation for the hybrids' attention layers.

**Old config keys.** `qk_activation`, `qk_norm`, `expand_k` and `use_beta` are not parameters of fla 0.6.0's `GatedDeltaNetConfig`; they will be stored as stray attributes. Whether the state-dict keys still match is unverified.

**Our forward-only scorer has two limits** (`eval/adapter.py`, `ForwardOnlyAdapter.logits()` L63):
- It builds the full fp32 logit matrix [N, V].
- It recomputes one full forward per continuation.

This is fine for LAMBADA, and for V = 32,000 at ≤ 4K (≈ 0.5 GB). At 32K with V = 151,936 it is infeasible: 32,768 × 151,936 × 4 B ≈ 19.9 GB.

### 1.6 What S0 already tests [code-inspected]

**`tests/test_vt09_transplant.py` (weight transplant):**
- Our C12 against `Qwen3ForCausalLM`, with norm gains randomised in [0.6, 1.4]:
  - T = 96 on CPU in fp32, including the cached and packed paths;
  - T = 256 on GPU in bf16 with varlen.
- Our C0 against fla's `GatedDeltaNetForCausalLM` at T = 200 (chunk mode).
- Tolerance 1e-3 relative in fp32.

**Random weights are enough for this test.** F_g normalises q and k per head, so attention logits are O(1) even at random initialisation. RoPE pairing and θ errors therefore already show at T = 96 [INF]. For example, rotary pair 1 turns 0.664 rad/token at θ = 5e5 and 0.750 at θ = 1e4, so the two drift ≈ 8 rad apart over 96 tokens.

**`tests/test_vt13_training.py` covers:**
- all four LR schedules;
- weight-decay groups on the real model;
- gradient accumulation equal to the full batch, including TEF weights;
- G-div restarts;
- atomic checkpoints and kill -9 resume;
- a 20-step smoke run of the real model with decreasing loss.

**No VT test covers:**
- a full-model transplant at T > 256 (VT-04 tests only the GDN kernel at 32,768);
- kernel durations;
- whether the generated items can be answered at 16K–32K.

---

## 2. Target numbers re-verified against the primary sources

All 11 were opened by me in this step. Each matches what the proposals quote. Rows 2 and 10 add information the proposals did not use.

| # | Number(s) as used | Used by | Source and location | Verdict |
|---|---|---|---|---|
| 1 | GDN RULER recall: pure 0.356, 24:1 0.341, 12:1 0.352, 6:1 0.425, 3:1 0.436 | protocol §2 prior; PR-P2; PV-M3 | [2507.06457v2](https://arxiv.org/html/2507.06457v2) Table 7 [read-section] | Exact |
| 2 | Transformer RULER average 0.423 | PR-P2 | v2 Table 5 [read-section] | Exact. **Per-subtask values (new; decisive for PR-P2):** see the breakdown below. |
| 3 | 340M LAMBADA (LMB): GDN pure 0.348, Transformer 0.353 | PV-M2 | v2 Table 2 [read-section] | Exact |
| 4 | 1.3B LMB: GDN 24-1 0.502 | PV-S4 | v2 Table 3 [read-section] | Exact (Transformer 0.474) |
| 5 | "a fixed ratio r:1 (linear : full)", r ∈ {24, 12, 6, 3}, versus the released configs | PV finding 3 | v2 App. D text [read-section]; HF `config.json` [code-inspected] | **PV is right** (details below) |
| 6 | Mamba-2 perplexity by number of attention blocks: 8.60, 8.38, 8.32, 8.29, 8.29, 8.28, **8.26**, 8.27, 8.28, 8.30, 8.34, 8.50; Transformer++ 8.68 | PV-M5, PR-P3 | [2405.21060v1](https://arxiv.org/html/2405.21060v1) §9.2.3 Table 2 [read-section] | Exact |
| 7 | Pythia LAMBADA (OpenAI): 70M 0.185 ± 0.005, 160M 0.328 ± 0.007 | PV-M2 | [2304.01373](https://arxiv.org/html/2304.01373) App. G.1 Table 7 [read-section] | Exact. Not checked: whether HF `main` serves the paper-era weights, so pin `revision` |
| 8 | GDN 1.3B S-NIAH (see below) | PR-S2; my §3.2a | [2412.06464](https://arxiv.org/html/2412.06464) Table 2 [read-section] | Exact |
| 9 | Wiki perplexity: GDN 16.42, Transformer++ 18.53, GDN-H1 16.07 | PR-P3 | 2412.06464 Table 3 [read-section] | Exact |
| 10 | Olsson et al.: ICL score, 0.15 / 0.4 nats, timing (see below) | PV-M6, PR-P4 | [transformer-circuits.pub 2022](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) [read-section] | Quoted correctly by both. PV's "coincides with warm-up, so not transferable" is half right |
| 11 | nanoGPT best validation loss 1.4697 | PV-M4, PR-S4 | `third_party/nanoGPT/README.md` L51 @3adf61e [code-inspected] | Exact |

**Row 2: Wang Table 5 per-subtask breakdown.**

| Model | Multikey | Multiquery | Multivalue | Singlevalue | CWE+FWE | QA | VT | Average |
|---|---|---|---|---|---|---|---|---|
| Transformer | 0.332 | 0.464 | 0.472 | **0.559** | 0.284 | 0.412 | **0.066** | 0.423 |
| GDN-hybrid-3-1 | 0.293 | 0.542 | 0.502 | **0.875** | 0.271 | 0.300 | **0.024** | **0.440** |

Table 7 gives 0.436 for the same 3:1 ratio, so the two tables average differently.

**Row 5: released configs.**
- 3-1: attention at layers [2, 5, …, 23], i.e. 8 of 24 layers, with 16 heads.
- 24-1: attention at layer [23] only, with 32 heads.
- So each "r-1" model has 1 attention layer per r layers: linear:full is (r − 1):1, not r:1.

**Row 8: GDN 1.3B S-NIAH scores (1K / 2K / 4K / 8K).**
- S-NIAH-1: 98.4 / 88.4 / 91.4 / 91.8.
- S-NIAH-2: 100.0 / 99.8 / 92.2 / 29.6.
- S-NIAH-3: 86.6 / 84.2 / 27.6 (1K / 2K / 4K only).

**Row 10: Olsson et al. details.**
- **ICL score:** loss at the 500th token minus loss at the 50th token. The text uses both sign conventions.
- **Magnitude:** "less than 0.15 nats" before the phase change, "roughly 0.4 nats" after it. For one-layer models, "not much in-context learning ever forms".
- **Timing:**
  - main text: "roughly 2.5 to 5 billion tokens";
  - model details: "approximately 1–3 billion tokens in the small models".
- **Warm-up:**
  - it "ramps up over the first 1.5e9 tokens";
  - the authors say the window "doesn't appear to correspond to a scheduled change in learning rate, warmup, or weight decay".
- **Small models:** 2^16 vocabulary, 8,192-token context, d = 768, ≈ 10B tokens over 10,000 steps, positional embeddings "similar to Press et al.".
- **Smeared keys** make ICL form in one-layer models.

---

## 3. Item-by-item verdicts

### 3.1 Consolidated table

Verdict codes:
- **KEEP**: run as proposed.
- **MOD**: modify as stated.
- **REPL**: replace with the cheaper check named.
- **DEFER**: move to Nice / later.
- **CUT**: drop.

GPU-h are new hours. "→" gives my estimate.

| Item (PV / PR id) | Claimed to validate | Problem found | Verdict | GPU-h |
|---|---|---|---|---|
| S1 as frozen (M1 / P1 a, b) | Mixers, harness | Smoke test only (§1.4) | KEEP, plus the §6 decisions | 0 new |
| Re-apply the S1 cut (R-S1-1 / P1 §2.1) | Budget | Code raises on 3 of 4 LRs (§1.1a). Real reason to cut d256 lr 1e-2 and 3.2e-2: likely divergence or watchdog crash | MOD (§6 U2) | saves ≈ 1–2 |
| causal_conv1d check (R-S1-4) | Shim identity | Moot (§1.2) | CUT | 0 |
| −0.005 monotonicity tolerance (R-S1-3); per-cell report (R-S1-2) | Gate robustness | Correct | KEEP; log now | 0 |
| Production F_g in MQAR (P1 c-1) | Production attention | In-distribution cells are at 1.0 after one epoch, so it **cannot fail**. VT-09(1) and VT-02 already cover the production F_g | CUT | 0.1–0.3 → 0 |
| Zoology seed replicate (PR-S5) | Basis for ±0.03 | Attention seed spread is 0 at ceiling; GDN with 2 seeds gives 1 df | CUT (or 1 GDN run) | 1.2 → 0–0.3 |
| Discriminating MQAR, d = 64 (M7) | "Our GDN = Zoology's" | Tests fla's layer against Zoology's re-implementation, which **differ by design** (bf16 cast; fp32 harness). Can fail with our code correct; our-vs-fla is already VT-09(2) | REPL by A1 | 1.7 → 0.01 |
| LAMBADA via lm-eval, Pythia + Wang 340M (M2 a, b, d / PR-S3) | Scorer; environment | lm-eval and `datasets` not installed; installing them pre-G5 changes `uv.lock`. Legs (a) and (b) validate lm-eval and the shim, not our pipeline | MOD to L1 (our scorer only) | ≈ 1.0 → 0.05 |
| Transplant with Wang weights (M2 c) | Mixers with trained weights | Exercises override paths we never use (QK-norm off, gate off, θ = 1e4, GQA replication, untied head, eps 1e-6). Needs a wrapper and a shim. VT-09 already covers the production paths (§1.6) | REPL by A2 | → 0.05 |
| Our Tier-0 suite on Wang 340M (M3) | Tier-0 sensitivity to k | Tier 0 only corroborates (D037). Wang call 340M recall "insignificant" (Table 2 caption). Spearman ≥ 0.8 over 6 points is arbitrary, and Wang's own recall is non-monotone in k (k/24 = 0, 1, 2 → 0.356, 0.341, 0.352). Needs re-tokenised token stores | CUT | 0.5 → 0 |
| RULER on Wang 1.3B (P2 a) | Our suite detects k | **Subset contradicts the source (§3.2a)**; ≈ 27 % false alarms even under exact replication; tooling missing (§1.5) | CUT | 1.0 → 0 |
| Our generators on Wang 1.3B (P2 b) | Sensitivity to k on a real sweep | Zero-shot 1.3B ≠ TEF-trained 30M. Download ≈ 33 GB; fp32 does not fit; 4K cells may be extrapolation; re-computation makes the 0.7 h estimate ≈ 4× low | MOD to W1 (3 models, ≤ 2K, bf16); should tier | 0.7 → 1–1.5 |
| nanoGPT ladder (M4 / PR-S4) | Trainer | Components already in VT-13 (§1.6). Dropout and train/eval mode are irrelevant: production has no dropout. r3/r4 put our dropout-free model on a 1 MB corpus where nanoGPT needs dropout 0.2, so a false alarm is likely [INF]. The r2-vs-r1 tolerance max(0.01, 2·SD) is weaker than a step-matched trace | REPL by T1 (+ optional r0) | 1.5 → 0.05–0.2 |
| modded-nanogpt (N2 / PR-N1) | Environment | Needs compile, FlexAttention and GPT-2 shards; does not test our code | CUT | 1.5–5 → 0 |
| Ratio-loss shape (M5 / P3) | LM pipeline across k | k = 1 is needed: S3 has no k = 1, so RP-2's "largest step at 0 → 1" cannot be evaluated as frozen. But the M5 flatness criterion conflicts with its own anchor (§3.2b). P3's 9 runs buy precision on a shape that is descriptive for GDN (§3.2b) | MOD to LM1 (3 runs) | 1.4–4.2 → 1.05 / 1.4 |
| Olsson ICL during training (P4 a) | Training budget past the phase change | 1–3B-token timing does not transfer (it sits inside a 1.5e9-token warm-up with ≈ 1M-token steps). The 50-step snapshot cadence is an unnecessary S0 change. FEAS has only k ∈ {0, 12}, so C1/C3 cannot be checked before G5 except on S3/LM1 runs | MOD to I1 (evaluation only, at existing milestones) | ≈ 0.1 |
| Smeared-key toy runs (M6 / P4 b) | Premise of rival E1.6; attention correctness | Not decision-relevant: X-gc12 (Tier A) and X-gc1/3 (conditional) measure the conv effect in the real setting. The 1-layer thresholds rest on "not much ICL" plus a figure. 0.1–0.2B tokens vs ≈ 10B in Olsson, so a failure cannot be interpreted. The attention-correctness role is covered by VT-02, VT-09 and LM-4(b). The 300K tok/s throughput is unmeasured | DEFER (Nice) | 0.6–1.9 → 0 |
| Jelassi copy task (PV-S1 / PR-S1) | Learning-speed confound | Mechanism mismatch: GDN-A's state at D384 (3 × 64 × 128 = 24,576 fp32 entries per layer) dwarfs the ≈ 1.4 kbit in a 300-letter string [INF], so the gap may vanish for reasons unrelated to us. The TEF-4000 dose guard and the "still closing" flag measure this confound directly. Needs a custom stream | DEFER | 1.5–3 → 0 |
| RoPE length behaviour on FEAS checkpoints (PV-S2 / PR-S2) | Position handling | Cheap and descriptive, but not a position-bug detector (VT-03 and VT-09 are) | KEEP as descriptive | 0.3 |
| Our generator vs RULER (PV-S3); Wang 1.3B LAMBADA (PV-S4) | Generator; transplant | Need RULER, or only make sense with W1 | DEFER | 0.8 → 0 |
| Full RULER, phonebook, Wang 340M floor, Stuffed-Mamba (PV-N1, N3, N4 / PR-N2, N3, N5) | — | Not our pipeline. The GSSM half of the phonebook needs `mamba_ssm` or a slow path | CUT | 4–8 → 0 |
| Gate G-RV (PV §6) | — | Puts ≈ 4.4 GPU-h plus lm-eval, a shim and two adapters on FEAS's critical path, mostly re-testing VT-09 and VT-13 | REPL by the §5 gate | — |

### 3.2 Notes on the decisive cases

**(a) PR-P2(a): the RULER subset contradicts its own source.**
- **The Transformer's advantage is not in single-needle retrieval.** In Wang's Table 5 the Transformer scores 0.559 on Singlevalue against 0.875 for GDN-3-1. On VT both are at floor (0.066 and 0.024).
- **Mapping P2's subset onto those columns.** P2 uses niah_single_1/2/3 plus vt, so Singlevalue and VT weigh 3:1. On those columns Wang's numbers imply Transformer − GDN-3-1 ≈ (3·0.559 + 0.066)/4 − (3·0.875 + 0.024)/4 ≈ −0.23 [INF]. For "S(Transformer) − S(pure) ≥ +0.03" to pass, pure GDN would have to sit ≥ 0.25 below the 3-1 hybrid on single-needle tasks.
- **Pure GDN is near ceiling on those tasks.** The GDN paper's own pure 1.3B models (also 100B FineWeb-Edu tokens, but different checkpoints) score 88.4–99.8 on S-NIAH-1/2 at 2K.
- **Which criteria will fail.** The Transformer criterion is very likely to fail on a correct pipeline. The 3:1-vs-pure criterion may survive, since Wang's text says "Single-Key, Multi-key, and QA sub tasks are noticeably affected by a changing ratio" (figure-only). Even so, compression near the ceiling at 2K shrinks the gap.
- **Even exact replication fails often.** Suppose Wang's averaged gaps held exactly on the subset. With P2's own SE of 0.025 per difference, the three thresholds pass jointly only about 0.945 × 0.827 × 0.931 ≈ 0.73 of the time [INF].
- **Recommendation.** Combined with the tooling gaps (§1.5), drop P2(a). If a RULER cross-check is wanted later, use multikey/multiquery at ≤ 2K.

**(b) PV-M5's flatness criterion conflicts with its own anchor.**
- **The criterion and its anchor cover different ranges.** "Range over k ∈ {1, 3, 6} ≤ 0.02 nats" is justified by "Mamba-2 range over 1–11 blocks = 0.014 nats". But 1–11 of 48 blocks is 2–23 % attention, while k ∈ {1, 3, 6} of 12 is 8–50 %.
- **The verdict depends on an unstated mapping:**
  - **By attention fraction** (k/12 ↔ n/48), Table 2 gives 8.29 at 4/48, about 8.30–8.34 at 11–15/48, and 8.50 at 24/48. The range is ln(8.50/8.29) = 0.025 nats, above 0.02.
  - **By count**, the range is ln(8.38/8.26) = 0.014 nats, inside 0.02.
- **Consequence.** A faithful pipeline can trip "the RP-2 audit before S4".
- **Fix.**
  - Keep the k = 1 run.
  - Judge flatness on k ∈ {1, 3} only.
  - Make "all-full vs best hybrid" descriptive.
- **GDN is not Mamba-2.** Pure GDN beats Transformer++ on Wiki perplexity (16.42 vs 18.53, GDN Table 3), and Wang's Table 6 is flat. So "failures" of PV-M5-P1/P3 or of PR-P3-a/c cannot be read as bugs. That also applies to PR-P3-a's audit trigger. PR-P3's nine runs (k = 2 and 6 plus triplicates) buy precision on a shape that is descriptive for GDN.

**(c) PV-M7 compares the wrong pair; P1(c-1) cannot fail.**
- **What M7 really tests.** Our `GDNMixer` subclasses fla's `GatedDeltaNet`, and VT-09(2) already ties our C0 to fla's own LM within 1e-3. So M7 tests whether fla's layer equals Zoology's re-implementation, and those differ by design (§1.3).
- **Why it can fail with our code correct.** M7 deliberately seeks the mid-range regime, where the bf16-vs-fp32 difference in kernel inputs can exceed ±0.03.
- **A cheaper, exact alternative.** The two layers share parameter names (`q_proj`, `k_proj`, `v_proj`, `a_proj`, `b_proj`, `A_log`, `dt_bias`, `q/k/v_conv1d`, `o_norm`, `o_proj`). A weight-copy forward/backward test answers the identity question exactly in about a minute (A1).
- **P1(c-1).** The production F_g scores 1.0 on all in-distribution cells after one epoch, so this check cannot fail.

**(d) PV-M4 / PR-S4: the trainer is already unit-tested.**
- What M4's r2 rung would isolate (optimiser groups, schedule, accumulation, clipping) is already in VT-13.
- The one residual risk is an end-to-end mismatch with a reference loop. A **step-matched trace** catches that better (T1):
  - our trainer drives nanoGPT's `GPT`, on nanoGPT's batches;
  - fp32, deterministic mode, dropout 0, constant LR, z-loss 0, β set equal in both;
  - 50 steps;
  - loss, gradient norm and LR must agree to ≈ 1e-6 relative.
- This takes minutes, versus 11 runs of convergence plus a 2-SD tolerance.
- Keep r0, a single run of the original config, only if someone wants the environment number (≈ 10 min). S1 has already shown that this environment trains reference models.
- **Engineering:**
  - an adapter that returns pre-head hidden states plus `lm_head_weight`, reshaping varlen [1, N] batches into [B, 256];
  - a β override in `OptimConfig`.

**(e) PV-M2: keep only the leg that tests our code.**
- The useful leg is our scorer against an external number: L1 below. It catches common-mode alignment errors that VT-07 (branched vs separate passes, both ours) cannot.
- It needs no lm-eval:
  - download LAMBADA-OpenAI with `huggingface_hub`;
  - convert each passage to our item format, with the context minus the target and the last context token as the prompt;
  - wrap Pythia with `full_logits` for `ForwardOnlyAdapter`;
  - compare with Pythia Table 7 (0.185 and 0.328, ± 0.015).
- Install lm-eval, in a separate venv, only if L1 misses.

**(f) PV-M6 / PR-P4: measure ICL; skip the toy runs.**
- The useful part is a direct measurement on the runs we already make (I1):
  - Olsson's ICL score;
  - repeated-random-token copying, which is our LM-4 variant (b) at gap 0.
- Measure them at the planned bf16 milestones (Stage-A 50M, 100M, 200M and 400M tokens, then S; protocol §18 L989) and at the S3/LM1 final checkpoints.
- **Checks:**
  - C12 ICL ≥ 0.15 nats by S (the paper's "before" boundary);
  - C12's second-copy loss far below its first-copy loss.
- Report C0, C1 and C3 descriptively.
- No 50-step cadence is needed.

**(g) The S1 budget.**
- PV says ≈ 5.5 GPU-h; PR says 3–5 h. About 1.1 GPU-h had been spent by 18:40 [MEAS-S1].
- The d256 row decides the rest [INF]:

| d256 run | Expected behaviour | Expected cost |
|---|---|---|
| lr 1e-3, 3.2e-3 | The smoke epoch is already at 0.987 against an early-stop threshold of 0.99, so these may stop within a few epochs | ≈ 0.1–0.2 h each |
| lr 3.2e-2 | Likely diverges in epoch 1 as at d128, then trains all 32 epochs | ≈ 1 h at 122 s/epoch |
| lr 1e-2 | Risks the watchdog crash seen twice at d128 | — |

- Realistic S1(a) total: ≈ 2–5.5 h.
- **The real benefit of cutting at d256** is that it drops two runs that are likely to diverge or crash. Saving budget is secondary.

---

## 4. Important validations that are missing

**A1. Weight-copy equivalence between our mixers and Zoology's (GDN and MHA).**
- Copy the weights, then compare forward outputs and gradients.
  - GDN: match the precision first, i.e. bf16 kernel inputs as Zoology does.
  - Attention: zero the biases in Zoology's MHA and put it in eval mode, which removes dropout 0.1.
- Pass: equal to fp32/bf16 noise.
- This makes S1(b) interpretable and replaces M7. Cost ≈ 0.01 GPU-h.

**A2. VT-09 at long lengths: production C12 vs Qwen3 (SDPA), and C0 vs fla's GDN LM.**
- Lengths:
  - fp32 at T ∈ {4K, 8K};
  - bf16 with FLASH at {16K, 32K}, with the tolerance of the existing bf16 GPU test (5e-2).
- fp32 cannot go to 32K. Our fp32 path uses SDPA-MATH, which needs 6 × T² × 4 B per layer: ≈ 1.6 GB at 8K, ≈ 25.8 GB at 32K.
- **This is the only check at the lengths of the primary cell.** Every reproduction proposed runs at ≤ 4K. Cost ≈ 0.05 GPU-h.
- Code inspection already shows our RoPE tables are computed in fp32 (`models/layers.py` L50–56), so the classic bf16 RoPE-angle bug is absent. Nothing *tests* the full model beyond T = 256, though.

**A3. Kernel-duration audit and crash diagnosis.**
- **Profile** the longest single CUDA kernel in:
  - one Stage-B step and one TEF step (1 × 32K with checkpointing), for C12 and C0 at the chosen width;
  - a 32K evaluation prefill (B = 1).
- **Diagnose** the S1 lr 1e-2 crash:
  - run it alone, after the queue finishes;
  - with `CUDA_LAUNCH_BLOCKING=1`;
  - outside the S1 results directory.
- **Decide how infrastructure failures enter protocol §8.1.** That section currently defines permanent failure only for divergence.
- Cost ≈ 0.1–0.5 GPU-h. This is the only item that addresses missing data correlated with k.

**A4. A long-context positive control for our generators and scorer.**
- **Model:** Qwen3-0.6B-Base [code-inspected config]:
  - `max_position_embeddings` 32,768, `rope_theta` 1e6;
  - ungated, Apache-2.0, sha `da87bfb`;
  - the same HF class VT-09 uses as its reference.
- **Cells:**
  - R1 with D ∈ {3, 15} and R-load with N ∈ {32, 128}, at 4K, 16K and 32K, plus C0;
  - 50 contexts per cell, 1 query per context at 32K.
- **Pass [INF]:** at 32K, EM_tf on R1 D = 3 is ≥ the C0 score − 0.20.
- **Purpose.** It separates "items or scorer broken at long T" from "our models fail" before FEAS chooses the scale. There is no published target; this is a control, not a reproduction.
- **Needs:**
  - the foreign-tokenizer generator path (shared with W1);
  - a wrapper that computes logits only at the scored positions (see §1.5 on the 19.9 GB logit matrix).
- Cost ≈ 0.3 GPU-h [INF: one 32K forward ≈ 160 TFLOP ≈ 2–3 s].

**A5. One GPU queue for all agents.**
- Ad-hoc launches beside the S1 queue contaminate its timings and plausibly raise the risk of watchdog timeouts [INF].
- Route every GPU job longer than about 5 minutes through `scripts/runner.py`.

**A6. Decision-log hygiene for S1.**
- The micro-batch shim (`micro_batch` = 64 in every run) is marked "DEVIATION" in the run JSONs and unit-tested (`test_s1_microbatch.py`), but has no `decisions.md` row.
- `s1_reproduction.md` still says "BLOCKED" and "fallback not implemented".

**Gaps accepted, with no cheap anchor:**
- No public small checkpoint was trained at ≥ 16K, and the GDN paper released no weights. FEAS, G-REF, A2 and A4 must carry the long-context assurance.
- The seed SD of the confirmatory metric is estimated only from FEAS's two seeds.

---

## 5. Cheapest set that covers the most risk

| Order | ID | Check | Risk covered | Literature anchor | New GPU-h | Engineering | Replaces |
|---|---|---|---|---|---|---|---|
| 0 | S1-D | §6 decisions | Gate validity | Zoology (running) | 0 | Decision rows | R-S1-1…5; PR §2.1 |
| 1 | A1 | Weight-copy equivalence: ours vs Zoology | Mixer identity | — | ≈ 0.01 | Small test | M7, PR-S5, P1(c-1) |
| 2 | A2 | VT-09 at long lengths | Production paths at 4K–32K | — | ≈ 0.05 | Extend VT-09 | M2(c) |
| 3 | A3 | Kernel audit and crash diagnosis | Infrastructure; missing data | — | ≈ 0.1–0.5 | Profiler script | (new) |
| 4 | L1 | LAMBADA through our scorer on Pythia-70M/160M (+ cached GPT-2) | Scorer alignment against an external number | Pythia Table 7: 0.185 / 0.328, ± 0.015 | ≈ 0.05 | HF wrapper + item conversion; no lm-eval | M2(a, b, d); PR-S3 |
| 5 | T1 | Step-matched trace against nanoGPT | Trainer, end to end | nanoGPT (r0 1.4697 optional) | ≈ 0.05 (+0.15 for r0) | Adapter; β override | M4; PR-S4 |
| 6 | I1 | ICL score and LM-4(b) at gap 0, at existing milestones and S3/LM1 finals | Budget; induction formed | Olsson: < 0.15 before, ≈ 0.4 nats after | ≈ 0.1 | Evaluation script | P4a; M6(a, d) |
| 7 | LM1 | k = 1 at η*, plus 1 extra seed at k ∈ {0, 1}; RP-2 with corrected criteria (flatness on k ∈ {1, 3}; k = 0 → 1 drop > max(3σ, 0.005)) | LM pipeline across k | Mamba-2 Table 2 (directional) | 1.05 (D384) / 1.4 (D512) | None (S3 recipe) | M5; P3 |
| S | A4 | Long-context positive control | Answerability at 16K–32K | — | ≈ 0.3 | Foreign-tokenizer generators; positions-only logits | (new) |
| S | W1 | Our generators on Wang 1.3B pure, 3-1 and Transformer: R1, R-load and C0 at 1K and 2K; bf16; paired bootstrap; directional criteria as P2(b-i) | Sensitivity to k on a real sweep | Wang Table 7 (direction only) | ≈ 1–1.5 (+ ≈ 16 GB download) | flash_attn shim (validated against naive softmax); shared with A4 | P2(b), reduced |
| S | R | RoPE length behaviour on FEAS checkpoints | Descriptive | Press Tables 2–3 | 0.3 | Evaluation | PV-S2 / PR-S2 |

**Totals.**

| Plan | New must-tier GPU-h | Should tier |
|---|---|---|
| This set | ≈ 0.4–0.8, plus LM1 (1.05 / 1.4) | ≈ 1.6–2.1 |
| PV | ≈ 8.0 | 3.1–4.1 |
| PR | ≈ 5.8 | 3.3 |

**Proposed gate, replacing G-RV.** FEAS starts when all of the following hold:
- A1, A2, L1 and T1 pass.
- A3 shows that no kernel in a 32K training step or 32K prefill exceeds 100 ms. The 100 ms is my threshold [INF], not a measured watchdog limit.
- The S1 crash is either explained or reproduced only at lr ≥ 1e-2.

I1 and LM1 are audit triggers, as RP-2 is. W1 and A4 inform the reproduction review that PR proposes before G5.

---

## 6. Urgent S1 decisions

Log these now, before any GDN d256 result is seen.

- **U1. The failed lr 1e-2 run.**
  - Report it as an infrastructure failure.
  - Compute best-over-LR from the completed LRs, with `n_lrs` shown.
  - Do not re-run it inside S1. Diagnose it under A3.
- **U2. The d256 cut, as an explicit rule.** The code path raises (§1.1a), so state the rule directly:
  - run GDN d256 only at {1e-3, 3.2e-3}, the 2 best completed LRs at d128 (0.9892 and 0.9875);
  - rationale: lr 3.2e-2 diverged at epoch 1 at d128 (0.9686 → 0.0003) and then trains 32 epochs; lr 1e-2 crashed twice;
  - the choice depends only on d128 results, so it is impartial if logged before d256 output exists.
  - This agrees in effect with PV R-S1-1 and PR §2.1(iii).
- **U3. Monotonicity tolerance.** Adopt PV R-S1-3: "non-decreasing in d_model" is judged with a −0.005 tolerance. Without it, a d256 run that never crosses the 0.99 early-stop threshold and ends just below d128's 0.9892 fails the gate on noise.
- **U4. Per-cell reporting.** Adopt PV R-S1-2: report every cell, flag the 1024/256 cell as the only discriminating one, and state that ±0.03 on overall accuracy equals ±0.21 on that cell.
- **U5. Clean-up.**
  - Close R-S1-4 (§1.2).
  - Add the micro-batch DEVIATION row.
  - Update `s1_reproduction.md`.
  - Stop ad-hoc GPU launches beside the queue (A5).

---

## 7. Research practices to add (engineering side)

These add to PV §1 and PR §1, which I otherwise endorse.

1. **Classify every check before it runs.**
   - *Smoke:* can only catch gross failure.
   - *Calibration:* has a stated false-alarm probability and a minimum detectable defect.
   - *Phenomenon:* directional, and a failure cannot be attributed to a bug.

   Write down both numbers for each pass criterion. For example, S1 is a smoke test, and PR-P2(a) has ≈ 27 % false alarms even under exact replication.
2. **Ask identity questions with weight-copy tests, not training outcomes.** Keep training comparisons for questions that weights cannot answer, such as trainability under our optimiser and data.
3. **Count shims.**
   - Report "original code + N shims", and give each shim its own identity test.
   - If the original code cannot run without changing the pinned environment, use a separate venv and downgrade the item to phenomenon grade.
4. **Run one GPU job at a time, all through the runner**, for any run that is timed for a budget or cut rule, or that lasts more than 15 minutes.
5. **Make every reproduction run longer than 15 minutes resumable.** The S1 queue retries from scratch. Budget a retry margin (S1 so far: 2 crashes in 9 queue attempts). Give infrastructure failures their own status in §8.1.
6. **Pin every artefact:**
   - HF `revision` SHAs for checkpoints and datasets (whether Pythia `main` matches the paper-era weights is unverified);
   - tokenizer SHA;
   - lm-eval and RULER commits;
   - the code hash of each shim.
7. **Evaluate inside each model's training window** (2,048 for the Wang GDN configs), unless extrapolation is what is being studied.

---

## 8. Sources opened in this review

| Source | Used for | Access |
|---|---|---|
| [2507.06457v2](https://arxiv.org/html/2507.06457v2) Tables 2, 3, 5, 7; App. D; Limitations | Rows 1–5 of §2; §3.2a | read-section (HTML, grepped) |
| [2405.21060v1](https://arxiv.org/html/2405.21060v1) §9.2.3 Table 2 | Row 6; §3.2b | read-section |
| [2304.01373](https://arxiv.org/html/2304.01373) App. G.1 Table 7 | Row 7; L1 | read-section |
| [2412.06464](https://arxiv.org/html/2412.06464) Tables 2 and 3 | Rows 8–9; §3.2a | read-section |
| [Olsson et al. 2022](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) | Row 10; I1 | read-section |
| HF API: tree and `config.json` for m-a-p 340M/1.3B GDN pure, 24-1, 3-1, Transformer; Qwen/Qwen3-0.6B-Base; Qwen/Qwen2.5-0.5B | §1.5; A4 | code-inspected |
| `third_party/zoology` @1ad20d1: `model.py` L127–152, L225–285; `mixers/{mamba,mamba2,gated_delta_net,base_conv}.py`; `train.py` L180–205; `config.py` L130–140 | §1.2–1.3 | code-inspected |
| `third_party/ruler` @db43f92: `scripts/synthetic.yaml`, `data/synthetic/niah.py`, `pred/model_wrappers.py`, `json/` | §1.5 | code-inspected |
| `third_party/nanoGPT` @3adf61e: README L45–55, `train.py`, `model.py` | Row 11; T1 | code-inspected |
| fla 0.6.0 in `.venv`: `layers/attn.py` L20–75; `models/gated_deltanet/configuration_gated_deltanet.py` | §1.5 | code-inspected |
| `src/hybridlc`: `INTERFACES.md`, `eval/adapter.py`, `models/{attention,layers}.py`, `training/queue.py`; `tests/test_vt09_transplant.py`, `test_vt13_training.py` | §1.5–1.6; A2 | code-inspected (read only) |
| `scripts/s1_zoology/{s1_analysis,run_queue,zoology_compat,test_s1_microbatch}.py`; `experiments/s1_zoology/{cut_decision.json, runs/*.json, logs/*}` | §1.1, §1.4 | code-inspected; MEAS-S1 |
| `nvidia-smi -q`; `cuDeviceGetAttribute(KERNEL_EXEC_TIMEOUT)`; `ps` | §1.1 | MEAS-R |
| Protocol v1.0, `decisions.md`, `STATUS.md`, `s1_reproduction.md`, both proposals | Context | read-full (proposals, §13–§17); read-section (rest) |

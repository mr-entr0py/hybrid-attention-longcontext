# S1 reproduction pilot: Zoology MQAR (protocol §17, gate G-RP)

**Status (2026-11-09 22:30 IST):**

- **S1(a) is complete.** All 16 runs finished on this machine: 4 attention configs and 4 GDN configs, at 2 widths each.
- **G-RP criterion (a): PASS.** Every number below comes from our own re-runs (I-20); no value was read off the paper or README figure.
- **S1(b) (= R4 of the reproduction programme) is complete (2026-11-10, 07:33–08:31 IST). G-RP criterion (b): PASS, 4/4 cells; R4b production-F_g smoke: PASS in fp32 and bf16.** Results: §8; full report `reports/reproduction/results_R4.md`; raw `experiments/repro/R4/`. G-RP is therefore passed in full ((a) and (b)). Evaluated after VT-25 (R3a) passed, as amendment §13.3 requires.

Machine-readable results:

- `experiments/s1_zoology/s1a_results.json` holds the G-RP verdict, best-over-LR cells and per-run summaries;
- `experiments/s1_zoology/runs/<run_id>.json` holds full per-epoch histories and provenance;
- `experiments/s1_zoology/s1a_table.md` has the full table;
- `experiments/s1_zoology/cut_decision.json` records the cut decision;
- logs are in `experiments/s1_zoology/logs/`.

## 1. What was run (faithful to §17)

**Code**
- **Zoology version.** HazyResearch/zoology @`1ad20d193b6113cae1e8f3c655c300d7b4b3f4bb`, at `third_party/zoology`. Every run records the commit, and the working tree was clean (`zoology_dirty: false` in every run JSON). Zoology is installed editable in `.venv` with `--no-deps`, and torch was not changed (2.11.0+cu128; fla 0.6.0).
- **Configs are Zoology's own objects.** `scripts/s1_zoology/s1_configs.py` *executes* the pinned `zoology/experiments/mqar_example_configs/original_mqar_configs.py` and filters its `configs` list to the §17 families and widths.

**Data**
- V = 8192.
- Train segments: (64, 4) × 100K, (128, 8) × 20K, (256, 16/32/64) × 20K each.
- Test segments: (64, 4/8/16), (128, 32), (256, 64), (512, 128), (1024, 256), 1,000 each.
- Batch 256 (test 32).

**Training**
- 32 epochs, LR ∈ `np.logspace(-3,-1.5,4)` = {1.0e-3, 3.2e-3, 1.0e-2, 3.2e-2}.
- AdamW with weight decay 0.1, cosine schedule to 0, seed 123, data seed 123.
- The training loop is Zoology's own `zoology.train.train(config)`.

**Model**
- 2-layer TransformerBlock: layer 0 is BaseConv(k=3) and layer 1 is the mixer, with an Identity state mixer, no position embeddings and tied embeddings.
- **Attention:** MHA with 2 heads and dropout 0.1, at d ∈ {64, 128}.
- **GDN:** Zoology's fla `GatedDeltaNet` with 2 heads, `use_gate=False`, short conv 4 and expand_v 2, at d ∈ {128, 256}.

**Fields changed from the file**
- `data.cache_dir = data/zoology/`;
- `logger = LoggerConfig()`, which turns wandb off;
- `run_id`.

**Early stopping.** Zoology's `TrainConfig` default stops training once `valid/accuracy > 0.99`. The original file does not override it, so it stays active. This is "exactly as in the file", not a deviation. Runs marked `*` below stopped early.

**Metric for G-RP**
- We use the *final* validation record, which equals the wandb run-summary value that Zoology's analysis code reads. It is the overall accuracy over all 7 test configs.
- Per-config accuracies come from Zoology's `num_kv_pairs` slices. The best epoch is stored in each run JSON as secondary information only.

**Cut rule.** The first timed run (attention d64, LR 1e-3) took 16.5 s. That implies 0.07 GPU-h for 16 runs, which is under 3.0, so there was **no cut**.
- That estimate is low, because attention stops early while the GDN d128 runs take all 32 epochs.
- The actual total was **1.48 GPU-h**, so the cut would not have fired in hindsight either.

## 2. Results

### 2.1 Best over LR (G-RP cells)

All accuracies are final overall test accuracy.

| family | d_model | state size¹ | best LR | overall | 64/4 | 64/8 | 64/16 | 128/32 | 256/64 | 512/128 | 1024/256 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| attention | 64 | 525,056 | 1.0e-2² | **1.0000** | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| attention | 128 | 1,050,112 | 1.0e-3 | **0.9999** | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| gated_delta_net | 128 | 67,072 | 1.0e-3 | **0.9892** | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.925 |
| gated_delta_net | 256 | 265,216 | 1.0e-3 | **0.9938** | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.957 |

¹ Zoology's `LanguageModel.state_size(sequence_length=1024)`, summed over both layers (including the BaseConv layer), as logged by Zoology's own code.
² All 4 attention d64 LRs score ≥ 0.99998, so the argmax is decided by a difference in the 5th decimal place.

### 2.2 All 16 runs

`epochs*` marks an early stop (valid acc > 0.99).

| family | d | lr | epochs | overall | 64/4 | 64/8 | 64/16 | 128/32 | 256/64 | 512/128 | 1024/256 | wall min |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| attention | 64 | 1.0e-03 | 1* | 1.0000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.3 |
| attention | 64 | 3.2e-03 | 1* | 1.0000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.3 |
| attention | 64 | 1.0e-02 | 1* | 1.0000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.3 |
| attention | 64 | 3.2e-02 | 1* | 1.0000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.3 |
| attention | 128 | 1.0e-03 | 1* | 0.9999 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.4 |
| attention | 128 | 3.2e-03 | 1* | 0.9999 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.3 |
| attention | 128 | 1.0e-02 | 1* | 0.9999 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.4 |
| attention | 128 | 3.2e-02 | 3* | 0.9997 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.999 | 1.1 |
| gated_delta_net | 128 | 1.0e-03 | 32 | 0.9892 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.925 | 16.6 |
| gated_delta_net | 128 | 3.2e-03 | 32 | 0.9875 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.913 | 17.3 |
| gated_delta_net | 128 | 1.0e-02 | 32 | 0.9877 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.915 | 12.9 |
| gated_delta_net | 128 | 3.2e-02 | 32 | 0.0050 | 0.030 | 0.004 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 18.2 |
| gated_delta_net | 256 | 1.0e-03 | 2* | 0.9938 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.957 | 1.2 |
| gated_delta_net | 256 | 3.2e-03 | 1* | 0.9933 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.954 | 0.6 |
| gated_delta_net | 256 | 1.0e-02 | 1* | 0.9936 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.956 | 0.6 |
| gated_delta_net | 256 | 3.2e-02 | 32 | 0.0050 | 0.030 | 0.004 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 18.3 |

**Divergence at LR 3.2e-2 for GDN.** Both widths learned at first: the best epoch reached 0.969 at d128 (epoch 0) and 0.948 at d256 (epoch 1). Both then collapsed to chance, about 0.005, and never recovered within 32 epochs. Attention is stable at that LR. We report this as observed. It is the top edge of Zoology's own grid, and best-over-LR is unaffected.

## 3. G-RP criterion (a) (§13.3): **PASS**

Evaluated by `s1_analysis.grp_a` on best-over-LR final overall accuracy, with all 4 LRs present in every cell.

| check | value | pass |
|---|---|---|
| attention d=64 ≥ 0.95 | 1.0000 | ✔ |
| attention d=128 ≥ 0.95 | 0.9999 | ✔ |
| GDN d=128 ≥ 0.95 | 0.9892 | ✔ |
| GDN d=256 ≥ 0.95 | 0.9938 | ✔ |
| GDN non-decreasing in d (128 → 256) | 0.9892 → 0.9938 | ✔ |

**Caveat on the monotonicity check.** The two GDN cells sit on opposite sides of Zoology's early-stopping threshold of 0.99.
- d256 crosses it in epoch 0–1 and stops, so its final value is "just above 0.99".
- d128 never crosses it and trains all 32 epochs.

The overall-accuracy gap is therefore partly shaped by the stopping rule. The cleaner signal is the hardest test config, 1024/256, which lies beyond the training lengths. There GDN improves from 0.925 to 0.957 with d (best LR), and every non-divergent d256 run (0.954–0.957) beats every d128 run (0.913–0.925). That ordering holds even though d256 trained only 1–2 epochs against d128's 32.

## 4. Comparison with the paper's and README's qualitative findings

Zoology (Arora et al. 2023, [2312.04927](https://arxiv.org/abs/2312.04927)) and the repo README's MQAR recall–state plot make the qualitative claims below. GDN was added to the repo after the paper, so the GDN claims come from the README's plot. The figure is used only qualitatively here (I-20).

1. **Attention solves MQAR at small, constant width.** *Reproduced.* Attention reaches ≥ 0.9997 at d = 64 and d = 128, at every LR, within 1–3 epochs. It is perfect even on the 512/128 and 1024/256 test configs, which are longer than any training sequence; the model has no position embeddings, so length is not tied to position.
2. **Recurrent (fixed-state) mixers are limited by state size, and recall improves as d_model, and with it the state, grows.** *Reproduced for GDN.*
   - GDN solves every in-distribution test config (≤ 256 tokens) at both widths.
   - It loses accuracy only on the largest-KV, longest config (1024/256): 0.925 at state ≈ 67K and 0.957 at state ≈ 265K. That is the recall–state trade-off the README plot shows.
   - Attention's state, the KV cache, is 2–16× larger than GDN's at T = 1024 (525K–1.05M vs 67K–265K), and it stays at 1.000.
3. **Input-dependent recurrent mixers such as GDN reach high MQAR accuracy, unlike gated convolutions that need d to scale with T.** *Consistent.* GDN ≥ 0.99 overall at d ≥ 128. Gated-convolution baselines were not part of §17 and were not run.
4. **Accuracy falls with more KV pairs and longer sequences for the recurrent model.** *Reproduced.* For GDN, all the degradation is concentrated in 1024/256, and 512/128 is ≥ 0.9997.

## 5. Deviations and engineering notes (none changes the computation beyond float order)

1. **DEVIATION (API compatibility): fla `head_first` shim.**
   - At the pinned commit, `zoology/mixers/gated_delta_net.py` passes `head_first=False` to `chunk_gated_delta_rule`. fla 0.6.0, the version the protocol pins, removed that argument and raises if it is passed.
   - `zoology_compat._patch_fla_head_first` drops the keyword when it is False and refuses True. `head_first=False` is exactly the `[B,T,H,D]` layout that fla 0.6.0 always uses, so the math is identical.
2. **DEVIATION (memory): exact gradient accumulation.**
   - Zoology computes fp32 logits for the whole 256-row batch. At T = 256 that is [256, 256, 8192] = 2 GiB, plus log-softmax and gradient buffers. That is above our 0.44 GPU share next to CARLA, and the smoke run OOMed.
   - `zoology_compat._patch_microbatch` splits each optimizer batch into 64-row micro-batches and weights each micro-batch's mean CE by its share of labelled targets. The accumulated gradient therefore equals Zoology's full-batch gradient up to float summation order, with one optimizer step per 256-row batch.
   - Only the dropout RNG stream differs. It matters for attention only, which has dropout 0.1.
   - Unit-tested: `test_s1_microbatch.py` checks that the accumulated gradient equals the full-batch gradient.
   - Peak memory was 2.03–2.06 GiB in every run.
3. **Robustness: per-epoch checkpoint and resume (`zoology_compat._patch_fit_resumable`).**
   - This is a line-for-line copy of Zoology's `Trainer.fit` (AdamW, cosine stepped per epoch, early stopping checked before the scheduler step). It also saves the model, optimizer, scheduler and all RNG states after each epoch.
   - **Why it was added.** The GPU drives the display and is shared with CARLA. Two attempts of `gated_delta_net-d128-lr1.0e-02` were killed by the driver watchdog (`cudaErrorLaunchTimeout`) after epochs 15 and 25. At the time there was no checkpointing, so each attempt restarted from scratch.
   - **The reported run.** The reported result for that config comes from attempt 3, a fresh, uninterrupted run from epoch 0 (`resumed_at_epochs: []`). No reported run was resumed mid-training.
   - **The failed attempts.** Their logs are kept as `logs/gated_delta_net-d128-lr1.0e-02.attempt{1,2}.log`. At the epochs where they died they were tracking the same trajectory: valid acc 0.986 at epoch 25 in attempt 2.
4. **Logging only, no numerical effect:**
   - Zoology's wandb logger is replaced by `CaptureLogger`, which keeps exactly the metrics Zoology would log;
   - tqdm bars are disabled (`TQDM_DISABLE`);
   - the allocator uses `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
5. **Dependencies.** We added `pydantic` and `wandb`, plus `torchvision==0.26.0+cu128` with `--no-deps`, because Zoology's `model.py` imports `torchvision.ops.StochasticDepth`. torch, triton and fla are unchanged.
6. **"Need to install causal_conv1d" in the logs.** This message comes from `zoology/mixers/mamba.py` and `mamba2.py`, which print it when they are imported. Neither is used here. BaseConv, MHA and Zoology's GDN do not use `causal_conv1d`; Zoology's GDN uses fla's `ShortConvolution`. The message has no effect on S1.
7. **Precision.** Zoology's GDN casts q, k, v, beta and g to bf16 before the fla kernel. That is Zoology's own code, unchanged. Everything else runs in fp32.

## 6. Part (b) preparation (run on 2026-11-10; results in §8)

`scripts/s1_zoology/our_mixers.py` wraps the project's own mixer factory, `hybridlc.models.mixers.build_mixer` (INTERFACES.md `models/mixers.py`). It is used as a Zoology `ModuleConfig` path in layer 1, so it replaces the sequence mixer only.

**Wrappers**
- **`OurFgZoology`** is F_g (`A`) with 2 heads × d/2, **no output gate, no QK-norm and no RoPE**, and scale 1/√head_dim.
  - Zoology's init rescale (std 0.02/√(2·n_layers)) is applied by parameter *name* (`out_proj.weight`). The wrapper registers F_g's `o_proj` under the alias `out_proj` first, so our output projection gets the same init as Zoology's MHA. This is verified by test.
- **`OurGdnZoology`** is our fla GDN (`-`, the GDN-A code path) in Zoology's geometry: 2 heads, d_k = d/2, d_v = d, `use_gate=False`, conv 4, no conv bias. The parameter shapes are identical to Zoology's GDN, apart from Zoology's unused `D` (verified by test).
- **`OurGdnConfigA`** is exploratory: GDN-A defaults (d_k 64, d_v 128, H = d/128, gate on).

**Run list.** `part_b_runs(best_lr_from_s1a())` swaps only the mixer entry of Zoology's original configs. With the S1(a) best LRs it gives:

- `our_fg-d64-lr1.0e-02`
- `our_fg-d128-lr1.0e-03`
- `our_gdn-d128-lr1.0e-03`
- `our_gdn-d256-lr1.0e-03`
- `our_gdn_configA-d128-lr1.0e-03`
- `our_gdn_configA-d256-lr1.0e-03`

The attention d64 LR choice is effectively a tie (§2.1 note ²). Running these needs a small extension to `run_one.py` / `run_queue.py`, which will look runs up in `part_b_runs` as well as `build_runs`.

**Residual differences from Zoology's mixers.** These are inherent to our modules. They must be reported with the ±0.03 G-RP (b) comparison.
- **F_g:** no biases (Zoology's Wqkv and out_proj have bias=True, zero-initialised) and no attention dropout (Zoology uses 0.1).
- **GDN:** our mixer feeds the fla kernel in the input dtype (fp32 here), while Zoology casts to bf16. Our decay gate is computed in-kernel with the same formula.

## 7. Tests actually run (this session)

| command | result |
|---|---|
| `python -m pytest scripts/s1_zoology/test_s1_analysis.py -q` | `7 passed` |
| `python -m pytest scripts/s1_zoology/test_s1_microbatch.py -q` | `2 passed` |
| `python -m pytest scripts/s1_zoology/test_s1_our_mixers.py -q` | `8 passed` |
| `bash scripts/s1_zoology/launch.sh` (resumed queue) | Ran the missing run `gated_delta_net-d128-lr1.0e-02` to completion (rc=0, 775 s), then `summarize.py`. Output: 16 runs, G-RP (a) PASS |

`test_s1_our_mixers.py` covers four things:

- F_g and GDN geometry vs Zoology's MHA and GDN inside `zoology.model.LanguageModel`;
- Zoology's out_proj init rescale reaching F_g;
- a 3-step fp32 AdamW train step on an MQAR batch (T = 256) for each wrapper, with finite gradients and decreasing loss;
- the part-(b) run list.

## 8. S1(b) / R4 results (2026-11-10): G-RP (b) **PASS**, R4b **PASS**

Pre-registration `experiments/repro/R4/prereg.yaml` (criteria copied from `reports/reproduction/reproduction_plan.md` §2.4, written before any run). Harness, data, seeds (123/123) and shims as in §1/§5; the collapse-abort rule of plan §2.0 was active and never fired. Runner `scripts/s1_zoology/run_one_r4.py` via the queue `scripts/s1_zoology/run_queue_r4.py` (`bash scripts/repro/r4_launch.sh`); every attempt held the single GPU lock and was infra-tagged (`experiments/repro/R4/infra/`). `src/hybridlc/models/` equalled HEAD `110fd9e` for every run. Evaluation `python scripts/s1_zoology/r4_analysis.py` → `experiments/repro/R4/r4_results.json`, `r4_table.md`.

**G-RP (b): ours within ±0.03 of Zoology on final overall accuracy, matched d_model and LR** (MDD: ±0.03 overall = ±0.21 on the 1024/256 cell).

| ours | Zoology (§2.2) | ours | Zoology | Δ | pass | 1024/256 ours / Zoology | epochs ours / Zoology |
|---|---|---|---|---|---|---|---|
| our_fg d64 lr 1e-2 | attention d64 lr 1e-2 | 0.99989 | 0.99998 | −0.00009 | ✔ | 1.0000 / 1.0000 | 1* / 1* |
| our_fg d128 lr 1e-3 | attention d128 lr 1e-3 | 0.99989 | 0.99995 | −0.00005 | ✔ | 1.0000 / 1.0000 | 1* / 1* |
| our_gdn d128 lr 1e-3 | gated_delta_net d128 lr 1e-3 | 0.98886 | 0.98924 | −0.00039 | ✔ | 0.9224 / 0.9250 | 32 / 32 |
| our_gdn d256 lr 1e-3 | gated_delta_net d256 lr 1e-3 | 0.99352 | 0.99377 | −0.00025 | ✔ | 0.9549 / 0.9566 | 2* / 2* |

All in-distribution cells (≤ 256 tokens) are ≥ 0.9992 on both sides. Each of our runs stopped at the same epoch as Zoology's. Our GDN d128 tracks Zoology's validation curve within 0.0032 at every epoch. Parameter counts match except for Zoology's unused `D` (GDN) and the MHA biases (F_g), and state sizes are identical. The residual differences of §6 (fp32 vs bf16 casts in GDN; no biases and no dropout in F_g) have no effect at this resolution.

**R4b: production F_g** (QK-RMSNorm, RoPE θ = 5e5, headwise gate, 2 × 64, d128, lr 1e-3; wrappers `our_mixers.OurFgProdZoology` and `OurFgProdBf16Zoology`). In-distribution mean (64/4 … 256/64) is **1.0000 in fp32 and 1.0000 in bf16** (criterion ≥ 0.95). Overall accuracy is 0.9981 / 0.9980, and 1024/256 is 0.987 / 0.987 (descriptive: RoPE beyond the 256-token training length). The bf16 arm ran the FA2 `varlen_attn` kernel (3,039 calls, bf16 q); the fp32 arm ran the reference path (0 calls).

**Config-A (exploratory).** GDN-A geometry (d_k 64, d_v 128, output gate on): d128 (H = 1) overall 0.9324, 1024/256 0.576; d256 (H = 2) overall 0.9793, 1024/256 0.856, and it never early-stopped. Config-A d256 has the same per-layer state as GDN d128 (16,384), which scored 0.922 (ours) and 0.925 (Zoology) on 1024/256. This is a single seed and descriptive only.

**Infrastructure.** RC watchdog + Xid 8 hit our process during the exploratory config-A d128 run (07:56:54, epoch 18). It has the same signature as the two S1(a) crashes. That fires plan §2.2 trigger O4 (soak before any further gate run, e.g. R5). No gate run was affected. The config-A d128 run was resumed from its epoch-18 checkpoint under the retry policy and is labelled as resumed.

**Tests run:** `python -m pytest scripts/s1_zoology/test_s1_r4.py scripts/s1_zoology/test_s1_our_mixers.py scripts/s1_zoology/test_s1_resume.py scripts/s1_zoology/test_s1_microbatch.py -q` → 16 passed.

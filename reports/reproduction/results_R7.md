# R7: LAMBADA on published checkpoints, through our scorer (results)

**Item:** R7 (reproduction_plan.md §2.7; protocol_amendment.md row R7). **Run date:** 2026-11-10, 05:52–06:27 IST.
**Executor:** reproduction-execution agent. **Repo commit:** `110fd9e357edae461a84ddeb0ddc18228235e1da`. The tree was dirty because the S0 workflow was editing `src/hybridlc/**` at the same time; the dirty-file list is in every `summary/*.json → env`. Nothing in `src/` was edited by me.

## 0. Verdict

| Criterion (frozen, §2.7 "Pass") | Measured (primary configuration) | Interval | Verdict |
|---|---|---|---|
| Pythia-160M acc ∈ [0.313, 0.343] | **0.3538** ± 0.0067 (1823/5153) | [0.313, 0.343] | **FAIL** (+0.026 above the target 0.328) |
| Pythia-70M acc ∈ [0.170, 0.200] | **0.2271** ± 0.0058 (1170/5153) | [0.170, 0.200] | **FAIL** (+0.042 above the target 0.185) |
| Wang 340M GDN-pure within ±0.02 of 0.348 (lambada_openai, or lambada_standard if only that one matches) | lambada_openai **0.3435** ± 0.0066; lambada_standard 0.2216 ± 0.0058 | [0.328, 0.368] | **PASS** (on lambada_openai) |

**R7 overall: FAIL (both Pythia criteria). Wang GDN-pure: PASS.**

**Where the failure lies (failure tree executed in full, §3):**
- **The scorer is not implicated.** lm-eval @`d6de8164`, run on the same pinned checkpoints in the same configuration (fp32, batch 1, tokenizer-default special tokens), gives **exactly** our numbers. Accuracy matches to every digit: 0.3537745 / 0.2270522. Item by item there are **0 EM flips in 2 × 5,153 items**, and max |Δ log p| is 0.0027 / 0.0068 nats.
- **The cause is the anchor's inference precision.**
  - The published Pythia numbers are fp16-inference numbers. EleutherAI's own eval records for these exact weights (`EleutherAI/pythia` @`72307fb`, `evals/pythia-v1/pythia-{160m,70m}/zero-shot/*_step143000.json`) give acc 0.32835 / 0.18455 and ppl 38.06 / 142.43.
  - Those records were made with lm-eval's 2023 `hf-causal` backend. With no dtype given, it loads `config.torch_dtype`, which is `float16` for Pythia (lm-eval @`2b9a6e94` `models/huggingface.py::_get_dtype` [code-inspected]).
  - With fp16 weights, **our scorer gives 0.3297 / 0.1849 (ppl 38.25 / 141.29)**, inside both intervals and within 0.0014 / 0.0004 of the published values.
- **The plan's tolerance basis assumed fp16→fp32 drift fits inside ±0.015 (§2.7 "Tolerance basis"). Measured: it does not.** The shift is +0.024 (160M) and +0.042 (70M).
- These diagnostics do not change the verdict. The pre-registered configuration was fp32 and the result is FAIL.

**Gate consequence (for the PI; not decided here).**
- §2.7 "Gate role" and the protocol_amendment G-RV row block FEAS verdicts only for "a failure that implicates `eval.scorer`". The failure-tree step-3 evidence says the scorer is not implicated.
- Whether this R7 FAIL blocks FEAS verdicts therefore needs a PI decision row. Options: accept it as a localised anchor-precision failure, or re-register the Pythia criterion at the anchor's precision. The second is a new pre-registration, not a re-reading of this one.

**What R7 validated (§2.7 "Validates"):**
- (i) EM_tf = lm-eval `acc`: **confirmed**, 0/10,306 flips at fp32.
- (ii) The item writer's context/answer split equals lm-eval's: **confirmed**. Doc texts are identical (0 mismatches), every item round-trips (`decode(ctx) == doc_to_text`, `decode(cont) == doc_to_target` for all 5,153 items, for both tokenizers), and 0 items are prefix-inconsistent.
- (iii) fla 0.6.0's `GatedDeltaNet` reproduces Wang's published number from weights trained with a pre-March-2025 fla: **confirmed**. Δ = −0.0045 (0.7 SE). No GDN semantic drift is detected at this resolution.

## 1. Readings fixed before running

The plan was read first. These readings were stated in-session before any full run was launched; they are written here afterwards.
1. **Primary configuration** (the plan's "Configuration"):
   - all 5,153 items of `EleutherAI/lambada_openai` `data/lambada_test.jsonl`;
   - fp32 weights and **no autocast**. `score_file` defaults to bf16 autocast on CUDA, so `autocast=False` was passed explicitly. TF32 was off and `float32_matmul_precision="highest"`;
   - tokenizer-default special tokens. GPT-NeoX adds none. Wang's `LlamaTokenizer` has `add_bos_token: true`, so `<s>` is prepended. lm-eval @d6de8164 `tok_encode` does the same when `add_bos_token=None` [code-inspected];
   - acc = mean EM_tf.
2. **Split.** `doc_to_text = text.split(' ')[:-1]|join(' ')`, `doc_to_target = ' ' + last word`. The request is `(ctx, target)` with no extra delimiter (`api/task.py` L1370 @d6de8164). Then `_encode_pair`: trailing spaces move to the continuation, `whole = enc(ctx+cont)`, `cont_enc = whole[len(enc(ctx)):]`.
   - Our item: `context_ids = ctx_enc[:-1]`, `prompt_ids = [ctx_enc[-1]]`, `answer_ids = cont_enc`, candidates = [gold].
3. **Wang** passes if lambada_openai **or** lambada_standard is within ±0.02. Both are reported (plan text).
4. **Sensitivities, never criteria:** BOS flipped; bf16 autocast (the FEAS default scorer path); lambada_standard for Pythia. The fp16-weights runs and the lm-eval runs are failure-tree diagnostics and were added after the primary FAIL.

## 2. Setup

- **Pins** (all match the plan; `experiments/repro/R7/download_manifest.json` has the SHA-256 of every file):
  - `EleutherAI/pythia-160m` @`50f5173d…`, `EleutherAI/pythia-70m` @`a39f36b1…`, `m-a-p/340M-20B-GatedDeltaNet-pure-baseline` @`14661a98…`;
  - `EleutherAI/lambada_openai` @`900124bf` `data/lambada_test.jsonl`: 1,819,752 B, sha256 `4aa8d02cd17c…`, equal to the plan's pin;
  - `cimec/lambada` @`5953bd97` `plain_text/test-00000-of-00001.parquet`, read with pyarrow, so `datasets` was not needed.
- **Environment:** `.venv`: Python 3.12.3, torch 2.11.0+cu128, transformers 5.22.2, fla 0.6.0, triton 3.6.0, tokenizers 0.23.3. RTX 5080 (sm_120), driver 595.99.02.
- **GPU sharing:** the GPU was shared at 99–100% utilisation with the S0 jobs (`nvidia_smi_pre/post` in each summary). Memory fraction was 0.44.
- **GPU events:**
  - 0 kernel Xid/NVRM/OOM events during any run (`journalctl -k` checked per run, and over 05:45–06:27).
  - Two 64-item debug attempts of GDN hit a Triton launch OOM caused by co-tenant memory pressure (about 0.6 GB free).
  - I added a bounded OOM retry to the wrapper; a retried forward is deterministic, so values are unchanged. **0 retries occurred in all full runs** (`gpu_oom_retries` field).
- **Code** (all new, in `scripts/repro/`; nothing in `src/` edited):
  - `hf_adapter.py`: `HFLogitsWrapper` exposes `lm_head_weight` and `full_logits`, and is wrapped by `hybridlc.eval.adapter.ForwardOnlyAdapter` → `prefill` / `score_continuation`. It also has the loaders with strict key checks.
  - `lambada_items.py`: the item writer.
  - `r7_run.py`: the driver. It writes items, calls `hybridlc.eval.scorer.score_file`, and writes the summary with metadata.
  - `r7_download.py`: pinned downloads.
  - `r7_launch.sh` and `r7_launch2.sh`: resumable detached launchers.
  - `r7_lmeval_diag.py` and `r7_compare.py`: the lm-eval diagnostic (run only in `.venv-lmeval`) and the item-by-item comparison.
- **Load: documented key maps** (the plan allows "a documented key map"; zero missing or unexpected keys after mapping):
  - **Pythia** (transformers-4.24 checkpoint under 5.22):
    - `embed_out.weight → lm_head.weight`;
    - 36 / 18 legacy buffers dropped only after checks: each `attention.bias` equals the causal tril mask (checked), `masked_bias` = −inf, and `rotary_emb.inv_freq` matches the model's to 6.6e-5 (fp16 storage);
    - the head is untied (checked).
  - **Wang GDN:** 24 × `model.layers.*.attn.D` dropped.
    - fla commit `834989cc` (2025-03-12, "Remove redundant parameter") deletes only the declaration `self.D = nn.Parameter(torch.ones(H))`. In the parent file (`d30c0833`, saved in `provenance/`), `self.D` is referenced nowhere else.
    - All 24 D tensors are exactly **0.0**, so dropping them is a no-op even under the NVlabs reading `o + D·v`.
    - Honesty note: my first-written load check expected D = 1.0, its init value. The data showed 0.0 and I revised the check. This is a load sanity check, not an R7 criterion.
  - **Wang legacy config keys** (`qk_activation=silu`, `qk_norm=l2`, `use_beta=True`, `use_output_norm=True`, `expand_k=1`, `norm_first=False`) are ignored by fla 0.6.0's config. They match 0.6.0's hard-coded behaviour: silu short-conv activation, in-kernel q/k L2 norm, sigmoid beta, gated output RMSNorm (`fla/layers/gated_deltanet.py` L174–195, L310–332). The head is untied (checked).

## 3. Failure tree (§2.7), executed for Pythia

1. **Split and encoding vs `_encode_pair`.**
   - Checked on all 5,153 items, not just 20: exact text round-trip, 0 prefix-inconsistent encodings, 0 trailing-space moves.
   - The 20-item human-inspection dumps are in `items/*.inspect.json`.
   - **Clean.**
2. **EM_tf alignment** (the scorer reads `lp[:, P:P+A]` with P = 1, i.e. predictions from the end of the context). **Clean.** This is shown directly by step 3.
3. **lm-eval @`d6de8164` in a separate venv** (`.venv-lmeval`: the same torch, transformers and tokenizers versions; lm-eval built from a copy of `third_party/lm-evaluation-harness`):

   | Model | Ours (fp32) | lm-eval (fp32, bs 1) | EM flips / 5,153 | max \|Δ log p\| |
   |---|---|---|---|---|
   | Pythia-160M | 0.3537745 (ppl 37.2554) | 0.3537745 (ppl 37.2554) | **0** | 0.0027 |
   | Pythia-70M | 0.2270522 (ppl 130.962) | 0.2270522 (ppl 130.962) | **0** | 0.0068 |

   ⇒ **The fault is on the model, data or anchor side, not in our scorer.**
4. **Localisation beyond the plan's tree (diagnostic):**
   - **Weights are the evaluated weights.** `pytorch_model.bin` is byte-identical between `main` and `step143000`. Main's `model.safetensors`, step143000's `model.safetensors` and the `.bin` are tensor-identical (0 differing tensors; step143000's safetensors only lacks the legacy buffers). `EleutherAI/pythia-v1.1-{70m,160m}`, the repo named in EleutherAI's eval records, now redirects to `EleutherAI/pythia-{70m,160m}` at our pinned SHAs.
   - **Precision explains the gap.** Accuracy and ppl by inference precision:

     | Model | fp32 (ours = lm-eval) | fp16, lm-eval bs 1 | fp16, lm-eval bs 16 | **fp16 weights, our scorer** | EleutherAI 2023 record | bf16, lm-eval (weights) | bf16 autocast, our scorer |
     |---|---|---|---|---|---|---|---|
     | 160M acc (ppl) | 0.3538 (37.26) | 0.3320 (38.13) | 0.3266 (38.36) | **0.3297 (38.25)** | 0.3284 (38.06) | 0.1155 (104.7) | 0.1215 (109.0) |
     | 70M acc (ppl) | 0.2271 (130.96) | 0.1863 (142.46) | 0.1840 (142.58) | **0.1849 (141.29)** | 0.1846 (142.43) | 0.0439 (2673) | 0.0499 (2670) |

     - Old `hf-causal` with `dtype=None` loads `config.torch_dtype` = float16 [code-inspected, lm-eval @`2b9a6e94`, copy in `provenance/`].
     - In fp16 there are 138 / 73 item flips between our scorer and lm-eval. That is fp16 noise: lm-eval itself moves 0.332 → 0.327 between batch 1 and batch 16. In fp32 the agreement is exact.
   - **Pythia in bf16 collapses** (acc 0.05–0.12) both through lm-eval with bf16 weights and through our scorer's bf16 autocast. So this is a property of these models under bf16 in transformers 5.22, independent of the harness.
     - It is not explained by a RoPE-in-bf16 mechanism that could also hit our models: our `models/layers.py::rope_tables` computes angles as an elementwise fp32 product, which autocast leaves in fp32 [read-only check].
     - Wang GDN under bf16 autocast is robust: 0.3421 vs 0.3435.

## 4. All runs (our scorer; n = 5,153 each)

| Run (model_variant_bos_precision) | acc | SE | ppl | in interval | role |
|---|---|---|---|---|---|
| pythia-160m_openai_default_fp32 | **0.3538** | 0.0067 | 37.26 | no | **PRIMARY** |
| pythia-70m_openai_default_fp32 | **0.2271** | 0.0058 | 130.96 | no | **PRIMARY** |
| gdn340m_openai_default_fp32 | **0.3435** | 0.0066 | 30.90 | **yes** | **PRIMARY** |
| gdn340m_standard_default_fp32 | 0.2216 | 0.0058 | 108.98 | no | criterion alternative (Wang) |
| pythia-160m_openai_on_fp32 (BOS added) | 0.3573 | 0.0067 | 35.87 | no | sensitivity |
| pythia-70m_openai_on_fp32 (BOS added) | 0.2078 | 0.0057 | 150.77 | no | sensitivity |
| gdn340m_openai_off_fp32 (no BOS) | 0.3511 | 0.0066 | 30.64 | yes | sensitivity |
| pythia-160m_openai_default_bf16 (autocast) | 0.1215 | 0.0046 | 109.02 | no | sensitivity |
| pythia-70m_openai_default_bf16 (autocast) | 0.0499 | 0.0030 | 2669.8 | no | sensitivity |
| gdn340m_openai_default_bf16 (autocast) | 0.3421 | 0.0066 | 30.88 | yes | sensitivity |
| pythia-160m_standard_default_fp32 | 0.2238 | 0.0058 | 183.55 | no | sensitivity |
| pythia-70m_standard_default_fp32 | 0.1547 | 0.0050 | 907.09 | no | sensitivity |
| pythia-160m_openai_default_fp16w | 0.3297 | 0.0065 | 38.25 | yes | diagnostic (anchor precision) |
| pythia-70m_openai_default_fp16w | 0.1849 | 0.0054 | 141.29 | yes | diagnostic (anchor precision) |

Notes on the table:
- SE = sqrt(p(1−p)/(n−1)), lm-eval's `mean_stderr`.
- ppl = exp(−mean gold log-likelihood), lm-eval's lambada `perplexity`.
- Wang's 0.348 sits between our BOS-on (0.3435) and BOS-off (0.3511) values; both are within ±0.02.

## 5. Observations for the S0 owners (not defects; nothing in src/ found to be wrong)

1. `scorer._score_items` groups batches by (context length, number of queries) only, and raises `ValueError` when answer lengths differ inside a batch. LAMBADA answers are 1–9 tokens, so R7 used `batch_size=1`; this is recorded as `batch_size_deviation` in the headers. The change has no numeric effect, because `ForwardOnlyAdapter` recomputes per row. FEAS item files have equal answer lengths by design, so this does not affect FEAS.
2. `rescore_check` runs its single-pass reference under the **same** bf16 autocast as the main pass. It therefore cannot detect precision-induced errors. For Pythia, bf16 vs fp32 is 0.12 vs 0.35, while Wang GDN is robust. If the PI wants the 5% check to also cover precision, an fp32 reference would do it. This is a design remark only.

## 6. Deviations

- Key maps for both families (documented in §2; the plan permits them). The Wang D check was revised once after the data, as noted in §2.
- `batch_size=1` instead of the frozen 8 (§5.1); numerically irrelevant.
- The lm-eval venv is named `.venv-lmeval` (as the task specified) rather than the plan's `.venv-repro`.
- The diagnostics went beyond step 3 (fp16/bf16 dtypes, batch 16, the weight-identity check, EleutherAI's records, fla history). All are labelled diagnostic and none feeds a verdict.
- Launcher logging bug: `r7_launch.sh` prints `rc=0` even for failed runs, because `$?` is read after a command substitution. Two Pythia BOS sensitivities failed in pass 1: a second Pythia revision (step143000, downloaded for the weight-identity check) made the snapshot glob ambiguous. I fixed `snap()` to select the pinned revision and reran them in pass 2. Values come only from runs with summaries.
- GPU time ≈ 0.73 h (scorer 1,987 s including loads, lm-eval 641 s) against the plan's 0.1 h. The causes are recompute scoring (2 forwards per item, batch 1), 14 runs instead of 3, and a co-tenant GPU at 100%.
- Pythia ran with transformers 5.22.2 SDPA attention, not the 4.24 the checkpoints were saved with. fp32 agreement with lm-eval on the same stack is exact, so this affects only the comparison with the 2023 numbers, which §3 accounts for.

## 7. Commands and raw files (all under `experiments/repro/R7/`)

- Download: `python scripts/repro/r7_download.py` → `download_manifest.json`, `hf_cache/`.
- Runs: `setsid nohup bash scripts/repro/r7_launch.sh` (logs `logs/launch.log`), then `bash scripts/repro/r7_launch2.sh` (`logs/launch2.log`). Each run is `python scripts/repro/r7_run.py --model M --variant V --bos B --precision P`.
- Per run: `scores/<run>.jsonl` (per-item scorer rows) + `scores/<run>_meta.json` (scorer header: item sha, eval-code sha, env) + `summary/<run>.json` (acc, SE, ppl, interval, role, load report, OOM events, kernel events, nvidia-smi before and after, env including git commit and dirty files, command). Item files are in `items/*.npz`, with 20-item dumps in `*.inspect.json`.
- lm-eval: `.venv-lmeval/bin/python -I scripts/repro/r7_lmeval_diag.py --snapshot <pinned dir> [--dtype D --batch-size B] --out lmeval/<name>.jsonl` → `lmeval/*.jsonl`, `*.summary.json`. Logs are `logs/lmeval_*.log`; the install log is `logs/venv_lmeval_install.log`.
- Comparison: `python scripts/repro/r7_compare.py --scores … --lmeval … --data …` → `compare/*.json`.
- External provenance: `provenance/` holds EleutherAI's 160m/70m step143000 zero-shot records @`72307fb`, fla `d30c0833` `gated_deltanet.py`, the `834989cc` diff, and lm-eval `2b9a6e94` `models/huggingface.py`.
- Debug-only 64/50-item runs (never results) are in `debug/`.

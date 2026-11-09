# hybridlc module contracts (orchestrator-owned)

The source of truth for *what* to build is `reports/experimental_protocol.md` (v1.0, frozen).
This file fixes *how modules talk to each other*, so they can be built in parallel. If you have to change a contract, edit this file in the same commit and describe the change in your final report.

Conventions:
- Package root is `src/hybridlc`. Tests go in `tests/test_vtXX_<name>.py` (one file per VT id where practical). Mark tests that need a GPU with `@pytest.mark.gpu` and tests over 1 min with `@pytest.mark.slow`.
- **The GPU is shared** with a CARLA simulator, which uses ~6–8 GB and up to 100% util. Every process must call `hybridlc.utils.gpu.claim(fraction=0.44)` (≈7 GiB) before allocating, and tests should stay under 4 GiB.
- Never use the global RNG anywhere in the data, eval or init paths. All randomness goes through `hybridlc.utils.seeds`.
- Paths: large artefacts live under `data/`, `checkpoints/`, `runs/` (all gitignored). Results go to `experiments/`.

## utils
- `utils/seeds.py`
  - `PROJECT_SALT = 271828182`.
  - `seed_sequence(seed_index, stream_id, *extra) -> np.random.SeedSequence`.
  - `torch_generator(seed_index, stream_id, *extra, device='cpu') -> torch.Generator`.
  - `STREAMS = dict(init_shared=0, init_mixer=1, data_A=2, data_B=3, tef=4)`.
  - Constants: eval_test entropy 20261109, eval_dev 77, bench 0, bootstrap 2026.
  - `MIXER_TYPE_ID = {'A':1,'C':2,'-':3,'E':4,'G':5,'N':6}`.
- `utils/gpu.py`
  - `claim(fraction)` sets the per-process memory fraction.
  - `snapshot()` returns a dict of nvidia-smi fields (util, mem per process, clocks, clocks_event_reasons, power, temp).
- `utils/provenance.py`
  - `env_info()` returns versions, git commit, dirty flag, GPU name, driver.
  - `config_hash(obj)` is the SHA-256 of canonical JSON.
  - `sha256_file(path)`.

## models
- `models/config.py`: `@dataclass ModelConfig` with fields
  - `d_model`, `d_ff`, `n_layers=12`, `vocab_size=16384`;
  - `pattern: str` (length n_layers, chars from `- A C E G N`: GDN-A, F_g, F_gc, GDN-E, GLA, GDN-no-conv);
  - `attn_head_dim=64`, `rope_theta=5e5`, `nope=False`, `ssmax=False`, `init_std=0.02`, `tie_embeddings=True`, `norm_eps=1e-5`.
  - `ModelConfig.for_scale('D384'|'D512', pattern, **overrides)`.
  - `patterns.py`: `centred(k, L=12)`, `placement(kstar, 'early'|'middle'|'late')`, and named configs C0..C12, X-*, per protocol §3.2.
- `models/mixers.py` (added by S1 for protocol §17 part (b); consumer: `scripts/s1_zoology/our_mixers.py`):
  - `build_mixer(kind: str, cfg: ModelConfig, layer_idx: int, **overrides) -> nn.Module`. `kind` is a pattern char (`A`, `C`, `-`, `E`, `G`, `N`). `build_model` should use it, so the S1(b) module is the same code as the LM's.
  - Mixer `forward(x[B,T,d], cu_seqlens=None, position_ids=None) -> [B,T,d]`. With `cu_seqlens=None`, x is a dense batch of independent sequences (no padding) and positions are 0..T-1 per row.
  - F_g/F_gc overrides (defaults = protocol §3.3): `n_heads=d//attn_head_dim`, `head_dim=cfg.attn_head_dim`, `qk_norm=True`, `rope=not cfg.nope`, `out_gate=True`.
  - GDN overrides: keyword arguments passed through to `fla.layers.GatedDeltaNet` (`num_heads`, `head_dim`, `expand_v`, `use_gate`, `use_short_conv`, `conv_size`). Defaults = GDN-A.
- `models/build.py`: `build_model(cfg, seed_index) -> HybridLM`. Initialisation follows protocol §7.1 and §8.1: shared modules are bit-identical across patterns for the same seed_index, and fla-specific tensors keep fla's own init.
- `HybridLM(nn.Module)`
  - `forward(input_ids[1,N] long, cu_seqlens[int32, n_docs+1], position_ids[1,N] long) -> hidden[1,N,d]`. Varlen training path, final RMSNorm already applied.
  - `.lm_head_weight` is the tied embedding weight.
  - `prefill(input_ids[B,T]) -> (last_logits[B,V] fp32, InferenceCache)`. No grad, single document per row, no padding.
  - `decode_step(token_ids[B,1], cache) -> (logits[B,V] fp32, cache)`.
  - `score_continuation(cache, cont_ids[B,M]) -> per-token logprobs[B,M] and argmax[B,M]`. Teacher-forced from a cache snapshot without mutating it, using the explicit `fused_recurrent_gated_delta_rule` path for GDN (protocol §9.2).
- `InferenceCache`
  - `.clone()`.
  - `.state_nbytes() -> dict(kv=..., gdn_state=..., conv=..., total=...)`, which must equal the analytic values (VT-14).
- `models/loss.py`: `chunked_ce(hidden, weight, labels, chunk=4096, z_loss_coef=1e-4, ignore_index=1, loss_weights=None) -> (loss, z_loss, n_tokens)`. Logits are computed in fp32 per chunk with recompute. `loss_weights` supports the TEF ½/½ weighting.
- `models/analytic.py`: `param_counts(cfg)`, `flops_per_token(cfg, T)`, `state_bytes(cfg, T)`, matching protocol §3.4 (VT-01, VT-14).
- *Additions / clarifications by the model builder (S0; all backwards-compatible with the lines above):*
  - Mixers also accept keyword-only context `max_seqlen: int` and `rope=(cos, sin)` (precomputed once per model forward; linear mixers ignore them), and expose `prefill(x[B,T,d]) -> (y, state)` and `extend(x[B,M,d], state, pos0) -> (y, new_state)`; `extend` never writes into the input state. Modules: `models/attention.py` (`AttnMixer` = F_g/F_gc), `models/linear.py` (`GDNMixer`, `GLAMixer`, subclasses of the fla 0.6.0 layers with an explicit keyword-only forward; parameter names = fla's).
  - `HybridLM.forward(input_ids, cu_seqlens=None, position_ids=None, max_seqlen=None)`: `max_seqlen` optional (the loader's value avoids one GPU sync); with `cu_seqlens=None` input may be a dense [B,T] batch.
  - `HybridLM.set_activation_checkpointing(on)` (per-block, non-reentrant), `set_gate_bypass(on)` (VT-09 test flag), `logits(hidden) -> fp32`, `full_logits(ids[B,T])` (single-pass re-scoring path), `extend(token_ids[B,M], cache) -> (logits[B,M,V] fp32, new_cache)` (cache not mutated).
  - `decode_step` advances the passed cache in place and returns it. `score_continuation(cache, cont_ids, return_logits=False) -> (logprobs[B,M] fp32, argmax[B,M])` (+ logits[B,M,V] if requested); `logprobs[:,0]` comes from `cache.last_logits`.
  - Precision: the inference methods do not set autocast; callers run them under `torch.autocast('cuda', torch.bfloat16)` for the canonical path (bf16 KV, SDPA FLASH pinned via `attention.INFERENCE_HALF_BACKEND`; fp32 runs use the MATH backend). Training attention uses `varlen_attn` in bf16/fp16 and a per-document SDPA-math reference otherwise.
  - `InferenceCache(pattern, layers: list[dict[str, Tensor]], seq_len, last_logits, meta)`; per-layer keys: `k`,`v` [B,T,H,64] (+`conv` fp32 [B,3d,4] for F_gc) or `state` fp32 [B,HV,d_v,d_k] (+`conv_q/k/v` fp32 [B,D,4]). `last_logits` is excluded from `state_nbytes()`.
  - `chunked_ce` ignores `labels == ignore_index` **and** any negative label; z_loss is returned already multiplied by `z_loss_coef` (same contract as `training/fakes.fake_chunked_ce`).
  - `analytic.py` also has `mixer_params`, `linear_core_flops`, `train_pflop`, `decode_bytes_moved`, `table_row(scale, config)` (regenerates a §3.4 row with its rounding). `state_bytes(cfg, T, batch=1, kv_bytes=2)` returns `dict(kv, gdn_state, conv, total)`.
  - Attention-statistics hook: `AttnMixer.collect_attn_stats` / `.attn_stats = {"max_logit", "entropy"}` (first ≤ 512 tokens of the row's first document).
  - `models/testing.py`: fp64 reference recurrences and tolerance helpers for the VT tests (not used on any production path).

## data
- Layout:
  - `data/raw/...` holds the downloaded parquet.
  - `data/tokenizer/` holds the HF `PreTrainedTokenizerFast` and `tokenizer.sha256`.
  - `data/tokens/<corpus>/<split>.bin` is a flat uint16 token array. `<split>.idx.npy` holds int64 document start offsets (each document begins with `<|endoftext|>` id 0). `<split>.meta.json` carries counts and the source file SHA-256s.
  - `data/MANIFEST.json` lists every file's SHA-256, the HF commit pins and the split assignment (protocol §6.1).
- `data/tokenizer.py`: `load_tokenizer() -> PreTrainedTokenizerFast`. Special ids: eot 0, pad 1, reserved 2–7.
- `data/loader.py`: `class StageStream(stage: 'A'|'B'|'LM2'|'LR', data_seed: int, seq_len: int, micro_batch: int, accum: int, mix: dict)`
  - `get(step: int, micro: int) -> dict(input_ids[1,N], labels[1,N], cu_seqlens int32, position_ids[1,N], batch_hash: str)`.
  - It is a pure function of (data_seed, stage, step, micro), so there is no state to save on resume.
  - Labels are next-token targets (shifted, with -100/pad at document ends as the protocol specifies).
    *Resolved by the data builder:* §7.1 says "document separators are predicted", so `labels[t]` is the next token of the source stream (the last token of a document predicts the following `<|endoftext|>`; the last token of a sequence predicts its true continuation). Only positions with no next token get the pad id 1 (= `chunked_ce(ignore_index=1)`); -100 is never used.
  - Mixing proportions are exact per window (protocol §6.3–6.4).
  - Additions (data builder): constructor kwarg `root=` (token directory; tests use synthetic stores); `get()` also returns `max_seqlen: int` (for `varlen_attn`) and `sources: list['fwe'|'pg19']` (one per sequence); `get_numpy()` is the same without torch; `step_hash(step)` is the SHA-256 over a step's micro-batch hashes. Stage `'LR'` is the Stage-A recipe (callers pass data_seed 100, §7.2); `'LM2'` ignores data_seed and has `num_steps()` = 1,000 micro-batches of 2 × 4K.
- `data/tef.py`: `class TEFStream(seed_index, steps, ...)`. Same `get()` contract, plus `loss_weights[1,N]`. It uses eval generators in namespace `train`.
  - *Clarified by the eval/TEF builder:* `TEFStream(seed_index, steps, seq_len=32768, micro_batch=1, accum=2, synthetic_share=0.9, answer_only=False, buckets=(512,…,32768), block_steps=1000, tok=None, filler=None, replay=None, root=None, **ignored)`. Defaults load `data/tokenizer`, `eval.filler.StoreFiller(ts, 'train', root)` and `data.tef.StoreReplay(root)`. `get()` also returns `answer_mask[1,N]` (bool, label positions whose target is an answer token), `max_seqlen`, `segments` (per sequence: item/replay metadata). `loss_weights` are normalised over the whole optimiser step (answer labels share ½, all other valid labels ½; `answer_only=True` → answers share 1, others 0), as `FakeTEFStream`. Labels: next token within the sequence; IGNORE (pad id 1) before a segment that does not start with `<|endoftext|>` (PG19 replay slices) and at the last position. Prefix-stable in `steps`; exact counts per 1,000-step block (`block_plan(b)`, `plan_counts(b)`, `step_counts(step)`).

## eval
- `eval/templates.py` and `eval/words.py`: word lists A/B, keys, templates T1–T4 (protocol §9.1).
- `eval/generators.py`: one function per task (R1, R-load, R3, R-dist, M-hop, A-freq, A-count, C0/C2 controls, LM-1..LM-4, R0) with signature `gen(task_level: dict, T: int, n: int, rng: np.random.Generator, namespace: 'test'|'dev'|'train', tok) -> list[Item]`.
  - `Item` = `dict(context_ids: np.int32[T_ctx], queries: [dict(prompt_ids, answer_ids, candidates: list[list[int]], meta)], meta)`.
  - `len(context) + len(queries)` follows protocol §9.1 (sample is exactly T tokens).
- `eval/items.py`: write and read frozen item files at `experiments/items/{test,dev}/<suite>/<task>_<level>_<T>.pt` (or `.npz`), plus `MANIFEST.json` with SHA-256s.
- `eval/scorer.py`: `score_file(model, item_path, out_jsonl, batch_size)` writes per-query JSONL rows with the fields of protocol §9.8, plus a `_meta.json` header.
- `eval/suites.py`: suite definitions CS/MS/DG/DC/FD/Tier0 as data (task, levels, lengths, n).
- *Clarifications / additions by the eval builder (S0):*
  - Generators take an optional 7th argument `filler: eval.filler.FillerProvider | None` (`get(kind, n, rng) -> (int32[n], source dict)`, kinds `pg19|fwe|repetitive`); default `StoreFiller(ts, namespace)` over `data/tokens` with the §6.1 split (test → pg19/test + fwe/test; dev → pg19/validation + fwe/dev; train → pg19/train + fwe/train). Cloze generators are in `eval/generators.py` (`generate(task, …)`, tasks `R1 R-load R-dist R3 R0 M-hop A-freq A-count`; C0/C2/T4/filler/order are `task_level` switches `placement='nofill'|'oracle'`, `template='T4'`, `filler='repetitive'`, `order='forward'|'reversed'`); Tier-0 LM generators are in `eval/tier0.py` (`gen_lm1/gen_lm2/gen_lm4`, LM-3 = annotations on LM-1 items). `T=None` = C0 no-filler control.
  - Length rule: `test`/`dev` (branched queries): every query of an item has the same `len(prompt)+len(answer)`, and `len(context)+len(prompt)+len(answer) == T`. `train` (TEF): `len(context)+Σ_q(len(prompt_q)+len(answer_q)) == T` (queries appended sequentially; `generators.flatten_item(item) -> (tokens, answer_mask)`).
  - `Item` also carries `arrays: dict[str, np.ndarray]` (per-item variable-length arrays: `fact_key[n,2]`, `fact_value`, `fact_start`, `fact_end`, `fact_role`; `probe_x/probe_y`; `lm3_pos/lm3_delta`; …). Every context starts with `<|endoftext|>` (counted in T) and contains no other separator. Query meta: `query_idx, key, value, cand_types, depth, distance, needle_start, needle_end, snapped`. Candidates: gold first.
  - Item files are `.npz` (deterministic zip, see `eval/items.py`): `experiments/items/{test,dev}/<suite>/<task>[-<variant>]_<level>_<T>.npz` (`T` = 1K…32K or `nofill`), `save_items(path, items, cell_meta)` (refuses to overwrite) / `load_items(path) -> (items, cell_meta)`; `write_manifest(ns_root)`; regeneration is byte-identical. Generated by `scripts/make_items.py`.
  - Scorer: `score_file(model, item_path, out_jsonl, batch_size=None, score_candidates=True, autocast=None) -> Path` (None → frozen batch size of §9.2; bf16 autocast on CUDA by default). Relies on `score_continuation` alignment `logprobs[b,j] = log p(cont[b,j] | ctx, cont[b,:j])` (j = 0 from `cache.last_logits`) but only reads j ≥ len(prompt) ≥ 1; uses `model.extend` (if present) to score candidates from a post-prompt cache. Header file is `<jsonl stem>_meta.json`; an existing output gets a `.rN` suffix (never overwritten). Tier-0 items are scored by one `model.forward(ids, cu_seqlens, position_ids)` per item (`score_lm_items`). `rescore_check(model, item_path, rows_jsonl)` = the 5% single-pass check via `eval.adapter.ForwardOnlyAdapter` (uses `model.full_logits` when present).

## training
- Config:
  - `configs/model/<scale>/<name>.yaml` (generated by `python -m hybridlc.training.genconfigs`; `--check` verifies) hold the full `ModelConfig` field dict under `fields` for every named config of §3.2 (C*, P{k*}-early/late, X-gc*, X-nope*, X-E*, O-gla*, O-nc*, C12ssmax, C3ssmax).
  - `configs/training/*.yaml` are stage recipes (`base`, `stage_A`, `stage_B`, `stage_B_16k`, `tef`, `tef_4000`, `tef_5050`, `tef_5050_lr1e-4`, `tef_answer_only`, `lr_sweep`, `vt13_audit`, `vt13_replicate`, `smoke`) with `inherits:`.
  - A *run YAML* composes them: `{exp_stage, model, training, seed_index, lr, steps_option, steps_multiplier, parent, continue_parent_step, config_name, model_overrides, overrides, tag}` (examples in `configs/training/examples/`). `parent` is a parent run YAML (or `{run_id, step[, ckpt]}`); the child loads model **and AdamW state** from `checkpoints/<parent run_id>/final.pt`.
  - `training/config.py`: `load_run_config(path, cli_overrides=None, paths=None, tag=None) -> RunConfig` (dataclasses `OptimConfig`, `ScheduleConfig`, `DataConfig`, `ValidationConfig`, `CheckpointConfig`, `LoggingConfig`, `GDivConfig`, `OOMConfig`, `RuntimeConfig`, `ImplConfig`, `PathsConfig`). `RunConfig.config_hash()` is the SHA-256 of the *semantic* config (excludes paths, logging/checkpoint/validation cadence, memory cap, tag).
  - `run_id = f"{exp_stage}{stage_tag}-{scale}-{config_name}-s{seed}-{hash8}"` with `stage_tag` "" (Stage A, LR sweep), "B" (Stage B), "T" (TEF); e.g. `S4-D384-C3-s0-1a2b3c4d`, `S4B-D384-C3-s0-…`. An optional `tag` is appended as `-{tag}`.
  - `RunConfig.impl` holds `module:attr` import strings for `build_model`, `StageStream`, `TEFStream`, `chunked_ce`, `ModelConfig` (tests point them at `hybridlc.training.fakes`).
- `training/trainer.py`: `train(run_cfg, resume=True, allow_dirty=False) -> summary dict`.
  - AdamW fused, with the weight-decay groups of protocol §7.1, gradient clipping 1.0 and z-loss.
  - Schedules (`training/schedules.py: lr_at(step, ScheduleConfig)`): Stage A warm-up→constant, B linear decay, LR-sweep compressed WSD, TEF re-warm; pure functions of the step.
  - Grad accumulation, bf16 autocast, activation checkpointing for 32K.
  - G-div restart logic, metrics JSONL every 10 steps, validation intervals.
  - **Step objective (clarified by the training builder):** `chunked_ce` returns `loss` = weighted *mean* CE over the micro-batch's non-ignored tokens (ignored: `labels == ignore_index` or `< 0`), `z_loss` = coefficient × mean lse² (already multiplied, *not* included in `loss`), `n_tokens`. The trainer optimises Σ_m (W_m/W)·loss_m + Σ_m (n_m/N)·z_m, where W_m = Σ loss_weights (or n_m) over the micro-batch's valid tokens — i.e. the exact full-step mean (TEF: the step-level ½/½ weighting).
  - **Model hooks used (optional):** `HybridLM.set_activation_checkpointing(bool)` (else the trainer wraps each block of `model.blocks`/`model.layers`); any submodule with a boolean attribute `collect_attn_stats` sets `attn_stats = {"max_logit": float, "entropy": float}` after a forward while it is True (logged every 10 steps).
  - Outputs: `runs/<run_id>/{metrics.jsonl, trace.jsonl, config.resolved.yaml, env.json, summary.json, attempts.json, train.log, gpu.jsonl, gpu_summary.json}`. `metrics.jsonl` rows have `kind` ∈ {start, train, val, checkpoint, end}; after a resume, re-run steps appear again — readers keep the *last* row per (kind, step) (`trainer.read_jsonl_last_per_step`). `trace.jsonl` has one row per step with float64 bit patterns of loss and grad norm plus the step batch hash (VT-13).
  - Exit codes of `scripts/train.py`: 0 done, 75 OOM (retry), 76 diverged twice (failed), 77 refused (dirty tree without `--allow-dirty`, missing parent checkpoint, config-hash mismatch).
- `training/checkpoint.py`: atomic save/load of model, optimiser, scheduler step, RNG, config hash, git commit and Triton-cache hash. Keep the last 2 full checkpoints plus bf16 milestones. Layout `checkpoints/<run_id>/{step_%07d.pt, final.pt, milestones/step_%07d[_final]_bf16.pt}`.
- `training/registry.py`: append-only `experiments/registry.jsonl`. Statuses: queued, running, completed, oom, crashed, interrupted, failed, cancelled; annotations: diverged, restarted, monitor, throughput_alert.
- `scripts/runner.py`: a single-GPU queue under nohup with auto-resume and a GPU monitor (`runs/<run_id>/gpu.jsonl`). Subcommands `enqueue | run [--forever] | start | stop | status | cancel` (logic in `training/queue.py`; monitor definitions in `training/gpumon.py`: contended ⇔ foreign processes' summed SM% ≥ 5 in an `nvidia-smi pmon` sample).
- CLI: `scripts/train.py --config <yaml> [--resume|--no-resume] [--allow-dirty] [-o key=value] [--tag T] [--runs-root/--ckpt-root/--registry]`, or composition mode `--model <yaml> --training <yaml> --seed S --lr η --exp-stage S4`.
- **Requirements on `data/tef.py` from the trainer:** constructor `TEFStream(seed_index, steps, seq_len=, micro_batch=, accum=, **kwargs)` with the fallback kwargs `synthetic_share=` (default 0.9) and `answer_only=` (default False) used by `tef_5050*.yaml` / `tef_answer_only.yaml`; `get(step, micro)` must be *prefix-stable* in `steps` (the TEF-4000 continuation calls it for steps 1000…3999 with `steps=4000` and must see the same items for the shared prefix as a `steps=1000` stream); the trainer optimises Σ_step w·CE / Σ_step w, so for the exact step-level D014 ½/½ weighting `loss_weights` must be normalised over the whole optimiser step (answer tokens of the step share ½, all other loss tokens ½; a common scale factor is irrelevant), as `FakeTEFStream` does.

## analysis
(Contract filled in by the analysis builder; see each module's docstring.)
- `analysis/protocol.py`: every §2/§9/§11 constant (probability scale; 15 pp = 0.15). Changing one is a DEV-xx.
- `analysis/results.py`
  - `load_item_results(results_root, registry) -> DataFrame` with columns `run_id, config, k, seed, scale, branch, ckpt, suite, task, level, T, context_id, query_idx, em_tf, logp_gold` (+ `extra_fields`). `T == 0` marks C0 controls (file `<task>_<level>_C0.jsonl`). Levels are canonicalised to `D3, N32, h2, rho1.5`.
  - Per-row fields read: `context_id` (falls back to `item_id`), `query_idx`, `em_tf` (or `EM_tf`), `logp_gold` (a list is summed). **Context ids must be identical across models** for the same item file (paired analysis; a mismatch raises).
  - Suffixed re-runs (`R1_D3_16384.r1.jsonl`) are never read; they are listed in `df.attrs["skipped_reruns"]`. Directories under `experiments/results/` starting with `_` are not run directories (e.g. `_bench`).
  - Registry: rows are merged per `run_id` (later rows update earlier ones). Used keys: `config`, `seed_index`/`seed`, `scale`, `stage`, `k`, and **`branch`** ∈ {`main`, `tef4000`, `stageB2x`} (default `main`). Runs without a registry row are identified from `run_id = {stage}-{scale}-{config}-s{seed}-{hash8}`. A `{run_id: branch}` map can be passed to `scripts/analyse_results.py --branch-map` for runs whose rows lack `branch`.
  - Checkpoint directories: `<ckpt>` names ending in the TEF step count (`tef250`, `tef1000`, `tef2000`, `tef4000`). The confirmatory evaluation is the unique checkpoint of each `main` run in `CS`/`DG` (or `--ckpt`). Dose guard: `DC` at `tef2000` and `tef4000` (and `CS` at `tef4000` if scored) of the `tef4000` branch; saturation guard: `CS` of the `stageB2x` branch.
- `analysis/scores.py`: `CellTable.from_rows(df, suite, metric)`; `ladder_weights(spec, Ts, control_for=None)` (equal weight per level, then per task, then per length; N=2048 never enters a ladder); `scores(weights) -> config x seed DataFrame`.
- `analysis/stats.py`: estimands (`l_contrast`, `recovery`, `d3`, `slope_log2`), tests (`one_sample_t`, `welch_t`, `welch_contrast`, `pooled_two_sample_t`, `tost_*`, `holm`, `bh_fdr`), `pooled_contrast` (seed-blocked, df 3(n−1) on the core table), `additive_residual_sd`/`seed_trigger` (U-05), `clopper_pearson`/`success_counts`, `fieller*`, `isotonic_*`, `k80`, `saturating_fit`, `n50_logistic`, `hierarchical_bootstrap` (seeds, then contexts within strata; common indices across configs; entropy 2026), `paired_t_power`.
- `analysis/pipeline.py`: `run_analysis(df, AnalysisSpec) -> dict` (JSON-safe) implementing §11 in its order of operations. `AnalysisSpec.gates` carries G0/G-RP/FEAS-1/FEAS-2/G-FULL outcomes (missing = not passed ⇒ no "falsified").
- `analysis/verdicts.py`: `compute_verdicts(results) -> dict` (supported / falsified / suggestive (supported) / named negative outcomes / inconclusive; labels; §11.9 wording).
- `analysis/opchar.py` + `opchar_reference.py`: PI-SIM / reviewer simulations re-run through the library (VT-18).
- `analysis/synthetic.py`: planted-effect worlds written in the real result layout (VT-18, demos).
- `analysis/plotstyle.py`: cividis for k (`k_color`), Okabe–Ito for families, `plot_seeds` (every seed drawn), `savefig(fig, name, inputs)` → PDF + PNG + `<name>.inputs.json` (SHA-256 per input, aggregate hash, code hash, CI type).
- `analysis/figures.py` + `plots/make_pXX.py` (P1–P18): read only `experiments/results/**` and the registry. Extra input formats expected (not item-level): `experiments/results/_bench/*.jsonl` rows `{config, k, kind: prefill|decode|bmax, B, T, l2: cold|hot, min_ms, analytic_ms, b_max}` (P8, P9); Tier-0 suite `Tier0` files `LM-1_b<lo>_<T>.jsonl`, `LM-3_d<lo>_<T>.jsonl` with a per-row `loss`, `LM-4_g<gap>_<T>.jsonl` with `copy_gain` (P17, P4 inset); MS `R-dist_d<d>_32768.jsonl` with `filler_source` (P16).
- `scripts/analyse_results.py --gates gates.json --out experiments/analysis/<name>` writes `results.json`, `verdicts.json`, `inputs.json`.
- VT-18 tests: `tests/test_vt18_{stats,opchar,pipeline,plots}.py`.

## integration (S0 integrator; additions only, nothing above changed in meaning)
- **Registry identity.** The trainer's `running` rows now also carry `config_name, scale, seed_index, k, stage, exp_stage, branch, parent_run_id`. A run YAML may set `branch: main|tef4000|stageB2x` (default `main`; *non-semantic*: excluded from the config hash, so run_ids are unchanged). `analysis.results.load_registry` prefers `config_name` and ignores a `config` value that is a path (the runner's `queued` rows store the run-YAML path under `config`). `--branch-map` is therefore only needed for runs launched before this change.
- **Evaluation driver** `integration/evaluate.py` + `scripts/evaluate.py --ckpt <final.pt|step_*.pt|milestones/*_bf16.pt> --suite <CS|MS|DG|DC|FD|Tier0>... [--label tef1000] [--n-contexts N] [--rescore]`:
  - `load_checkpoint(path, device, runs_root=None) -> LoadedCheckpoint(model, run_id, stage, step, config, path, sha256, dtype, config_hash, git_commit)`; the model is rebuilt with `build_model(ModelConfig.from_dict(config['model']), seed_index)` and loaded `strict=True` (bf16 milestones cast up to fp32; their config comes from the run's `final.pt` or `runs/<run_id>/config.resolved.yaml`).
  - Result layout (§9.8, what `analysis.results` reads): `<results_root>/<run_id>/<label>/<suite>/<result_name>.jsonl` + `_meta.json`, with `result_name(item_stem)` the single mapping from item names: `R1_D3_16K → R1_D3_16384`, `R1-C0_D3_nofill → R1_D3_C0`, other variants keep their suffix (`M-hop-forward_h2_16384`), Tier-0 token lengths unchanged. Default labels: `tef<step>` (TEF), `B<step>`, `A<step>`.
  - Suite → namespace: CS/MS/Tier0 → `test`, DG/DC/FD → `dev` (the MS attention-diagnostics cell lives in `dev/MS/` and is read by the diagnostics code, not by this driver).
  - `--n-contexts N` scores the first N contexts with the same scorer functions; the header records `subset: true` (never confirmatory). Every scored cell appends one line to `<results_root>/_eval_log.jsonl` (the §9.8 "registry note"; kept out of the training registry so its state machine is untouched). Existing outputs are never overwritten (scorer `.rN` rule).
- **Benchmark harness** `bench/harness.py` (§10.2 primitives, validated by VT-15): `graphed(fn)`, `time_replays(fn, n=30, warmup=5, cold_l2=False)` (CUDA events, 256 MiB L2 flush before every timed call, min/median/IQR in µs), `sdpa_spy()` (records every `models.attention._sdpa` call with the backend flags in force and the mask kind), Appendix-A single-layer cases.
- **S0 scripts**: `scripts/g0_throughput.py` (G0 tok/s + peaks via the production trainer with a GPU monitor), `scripts/warm_triton.py` (VT-19 warm-up + deterministic tarball + registry `note` row `S0-triton-cache`), `scripts/s0_smoke_e2e.sh` (A → B → TEF → DC → analysis under `experiments/s0/smoke/`).

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
  - Mixing proportions are exact per window (protocol §6.3–6.4).
- `data/tef.py`: `class TEFStream(seed_index, steps, ...)`. Same `get()` contract, plus `loss_weights[1,N]`. It uses eval generators in namespace `train`.

## eval
- `eval/templates.py` and `eval/words.py`: word lists A/B, keys, templates T1–T4 (protocol §9.1).
- `eval/generators.py`: one function per task (R1, R-load, R3, R-dist, M-hop, A-freq, A-count, C0/C2 controls, LM-1..LM-4, R0) with signature `gen(task_level: dict, T: int, n: int, rng: np.random.Generator, namespace: 'test'|'dev'|'train', tok) -> list[Item]`.
  - `Item` = `dict(context_ids: np.int32[T_ctx], queries: [dict(prompt_ids, answer_ids, candidates: list[list[int]], meta)], meta)`.
  - `len(context) + len(queries)` follows protocol §9.1 (sample is exactly T tokens).
- `eval/items.py`: write and read frozen item files at `experiments/items/{test,dev}/<suite>/<task>_<level>_<T>.pt` (or `.npz`), plus `MANIFEST.json` with SHA-256s.
- `eval/scorer.py`: `score_file(model, item_path, out_jsonl, batch_size)` writes per-query JSONL rows with the fields of protocol §9.8, plus a `_meta.json` header.
- `eval/suites.py`: suite definitions CS/MS/DG/DC/FD/Tier0 as data (task, levels, lengths, n).

## training
- Config:
  - `configs/` holds YAML.
  - `training/config.py` defines dataclasses `RunConfig(model: ModelConfig, stage, optim, schedule, data, seed_index, run_id, parent_ckpt, ...)`, with `load_yaml` and `config_hash`.
  - `run_id = f"{stage}-{scale}-{config}-s{seed}-{hash8}"`.
- `training/trainer.py`: `train(run_cfg, resume=True)`.
  - AdamW fused, with the weight-decay groups of protocol §7.1, gradient clipping 1.0 and z-loss.
  - Schedules: WSD stage A constant, B linear decay, LR-sweep compressed WSD, TEF re-warm.
  - Grad accumulation, bf16 autocast, activation checkpointing for 32K.
  - G-div restart logic, metrics JSONL every 10 steps, validation intervals.
- `training/checkpoint.py`: atomic save/load of model, optimiser, scheduler step, RNG, config hash, git commit and Triton-cache hash. Keep the last 2 full checkpoints plus bf16 milestones.
- `training/registry.py`: append-only `experiments/registry.jsonl`.
- `scripts/runner.py`: a single-GPU queue under nohup with auto-resume and a GPU monitor (`runs/<run_id>/gpu.jsonl`).
- CLI: `scripts/train.py --config <yaml> [--resume] [--allow-dirty]`.

## analysis
- `analysis/stats.py`: family scores, L_k, D₃, slopes, paired t with Holm, hierarchical paired bootstrap, TOST, Fieller, isotonic/saturating fits, success counts with Clopper–Pearson (protocol §11).
- `analysis/verdicts.py`: applies the pre-registered rules and outputs verdict JSON.
- `analysis/plotstyle.py`: cividis for k, Okabe–Ito for families, and a `savefig(fig, name, inputs)` that writes PDF, PNG and an input-hash JSON.
- VT-18 tests the whole pipeline on synthetic planted data.

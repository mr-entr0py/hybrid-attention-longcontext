# 05 — Implementation report: stack, versions, pitfalls, data, tokenizer

**Author:** Agent 4 (Literature / Repository / Implementation Researcher)
**Date:** 2026-11-09
**Scope:** What to reuse and what to write for training a family of ~20–60M-param hybrid linear/full-attention LMs on one RTX 5080 (sm_120, 16 GB, shared). Covers recommended stack, verified package versions, pitfalls, dataset and tokenizer choices, and a repo layout sketch.
**Companion DB:** `reports/literature/db_implementation.yaml`

**Labels used in this report**
- Access level of each source: `[read-full]`, `[read-section]`, `[abstract-only]` (abstract page or dataset card read), `[code-inspected]` (shallow clone in `third_party/`, commit given).
- Kind of claim: **ESTABLISHED** (stated in a source), **VERIFIED-HERE** (I ran it on this machine), **INFERENCE** (my reasoning, not yet tested).

---

## 0. TL;DR recommendations

1. **Use `fla` (flash-linear-attention 0.6.0) as a kernel/layer library, not as a framework.**
   - Take `fla.layers.GatedDeltaNet` (Triton) as the linear mixer.
   - Use PyTorch-native attention (SDPA, plus `torch.nn.attention.varlen.varlen_attn` for document masking) as the full mixer.
   - Write our own ~600-line model + trainer + decode benchmark, nanoGPT-style.
   - Do **not** adopt `flame`/torchtitan. Do **not** depend on the `flash-attn` package.
   - VERIFIED-HERE: everything in this stack trains, masks documents and decodes with cache on the RTX 5080 (§2).
2. **Versions verified on this GPU (VERIFIED-HERE):**

   | Package | Version |
   |---|---|
   | torch | 2.11.0+cu128 (arch list includes `sm_120`) |
   | triton | 3.6.0 |
   | flash-linear-attention / fla-core | 0.6.0 |
   | transformers | 5.22.2 (pulled in by fla, which requires ≥5.6.2) |
   | cuDNN | 9.19.0.56 |
   | einops | 0.8.2 |
   | tokenizers | 0.23.3 |

   The driver is 595.99.02. These are installed now in `/home/sedrica-lynx/Downloads/outeropt/.venv`.
3. **Linear mixer = Gated DeltaNet (GDN).**
   - It is the gated linear-attention family used in Qwen3-Next / Qwen3.5 / Olmo Hybrid. It is one of the two best hybrid partners in the only controlled ratio study (Wang et al. 2025, §4.1).
   - It has fla's most-maintained kernels, including Blackwell-specific fixes.
4. **Parameter-match the two mixers exactly.** Use GDN with `use_gate=False, expand_v=1, num_heads=4, head_dim=128` at d=512 (q, k, v, o = 4d², plus 10K small params). It replaces a standard MHA layer (4d²), so the parameter count stays within 0.03% across all ratios. VERIFIED-HERE: params 49.42M for r=0 vs 49.30M for r=1 at L=12, d=512, V=16384.
5. **Tokenizer: train our own 16,384-entry byte-level BPE with single-digit splitting** on the training mix. Tie input and output embeddings.
   - Versus GPT-2's 50,257 vocab, this removes 17.3M embedding params at d=512, about a third of a 50M model.
   - It costs only +9.8% tokens on FineWeb-Edu and +10.1% on PG19 (VERIFIED-HERE, §6).
6. **Data:** FineWeb-Edu (ODC-By) for short/medium text, plus long documents for the 8K–32K regime.
   - FineWeb-Edu alone is long-poor: only 0.95% of docs exceed 8K GPT-2 tokens (13.1% of tokens), and 0.037% exceed 32K (VERIFIED-HERE, §5).
   - PG19 (Apache-2.0, 28,602 books; essentially all >32K tokens) supplies them, plus the openly-licensed Common Pile arXiv subset.
   - The mix should be the same for every ratio.
7. **The efficiency benchmarks need an idle GPU and CUDA-graph decode.**
   - Eager HF-style decode at this scale is launch-bound: about 5–7 ms/token regardless of context length or architecture (VERIFIED-HERE, §2.6).
   - So eager decode measures Python overhead, not architecture.
   - Training-throughput readings varied by ~1.6× between runs because the GPU is shared (66k vs 104k tok/s, same config).

---

## 1. Hardware/software facts (VERIFIED-HERE)

| Item | Value |
|---|---|
| GPU | RTX 5080, compute capability (12, 0), 16 GB. About 10 GB free at test time. Usage was 6.5–10.2 GB used by others: CARLA ≈ 6.3 GB, plus other agents' processes at times. |
| torch | 2.11.0+cu128. `torch.cuda.get_arch_list()` = `['sm_75','sm_80','sm_86','sm_90','sm_100','sm_120']` |
| triton | 3.6.0 (bundled with the torch wheel) |
| cuDNN | 91900 |
| SDPA backends on sm_120 | `FLASH_ATTENTION` ✓, `EFFICIENT_ATTENTION` ✓, `CUDNN_ATTENTION` ✓. All three ran fwd+bwd in bf16. At B=2, H=8, T=4096, Dh=64 the timings were flash 0.48 ms, cuDNN 0.54 ms, efficient 1.03 ms per fwd+bwd. |
| `torch.nn.attention.varlen.varlen_attn` | ✓ fwd/bwd with `cu_seqlens`, causal via `window_size=(-1,0)`. It wraps the FA2 kernel that ships inside PyTorch (`aten._flash_attention_forward`), so no external flash-attn is needed. |
| `flex_attention` + `create_block_mask` (document mask, compiled) | ✓. Matches `varlen_attn` to within bf16 noise (max abs diff 0.0156). |
| nvcc | not installed system-wide, so no source builds of CUDA extensions. |

Test scripts (they live in my scratchpad and are not part of the repo; reproduce from §2.7):
- `smoke_gdn.py`, `smoke_hybrid.py`: correctness.
- `bench_train.py`, `bench_decode.py`, `bench_compile.py`: speed.
- `scripts/lenstats.py`, `scripts/tok_exp.py`: data and tokenizer.

---

## 2. Verified smoke tests on the RTX 5080

All tests ran with `torch.cuda.set_per_process_memory_fraction(0.12–0.14)`, a cap of about 2 GB, to leave room for CARLA. Peak use was ≤1.3 GB.

### 2.1 GDN layer fwd/bwd (VERIFIED-HERE)
- Config: `GatedDeltaNet(hidden_size=512, num_heads=4, head_dim=128, expand_v=1.0, use_gate=False, mode='chunk')` → 1,058,952 params.
- B=2, T=2048 bf16 fwd+bwd OK with finite gradients.
- **The first call took 49 s** (Triton JIT plus autotune). Steady state was 2.67 ms/iter. A second process reuses the Triton cache in `~/.triton`: the hybrid model's first step fell from 55 s to 0.3 s.

### 2.2 Variable-length / document masking (VERIFIED-HERE)
- GDN with `cu_seqlens=[0,700,1500,4096]` (packed, batch=1) runs fwd/bwd.
- Document isolation is exact: the output for doc 2 equals running doc 2 alone, max abs diff 0.0. The short-conv state is also reset at boundaries.
- `varlen_attn` with the same `cu_seqlens` matches per-document SDPA to bf16 precision.

### 2.3 Full fla hybrid model (VERIFIED-HERE)
- Model: `GatedDeltaNetConfig(hidden_size=512, num_hidden_layers=12, num_heads=4, head_dim=128, expand_v=1, use_gate=False, vocab_size=16384, tie_word_embeddings=True, attn={'layers':[5,11], 'num_heads':8, 'num_kv_heads':8, 'qkv_bias':False, 'rope_theta':1e4, 'window_size':None})`.
- Parameters: 49.40M total (8.39M embedding, 41.01M non-embedding). Layer types: `[GDN×5, Attention, GDN×5, Attention]`.
- Packed varlen fwd/bwd works with `model(input_ids, labels, cu_seqlens=...)`.
- **Precondition:** fla's `Attention` class hard-requires `flash_attn` (it raises `ImportError` in `__init__` if `flash_attn_func is None`; see `fla/layers/attn.py` L26–33 and L71–72). I worked around this by assigning torch-backed functions to `fla.layers.attn.flash_attn_func` and `fla.layers.attn.flash_attn_varlen_func` after import (§2.7).
  - Injecting a fake `flash_attn` module into `sys.modules` instead **breaks transformers**: `ValueError: flash_attn.__spec__ is None`.
- **Prefill + token-by-token cached decode vs. one full forward:**
  - fp32: max |Δlogit| = 0.0099, argmax agreement 1.000.
  - bf16: argmax agreement 0.977 on a random-init model with near-uniform logits.
  - So cache handling is correct. bf16 introduces small chunk-vs-recurrent numeric differences (single-layer relative error 2.5e-3).
- `model.generate(...)` (HF API) works.

### 2.4 Inference-state memory accounting (VERIFIED-HERE; matches analytic formula)
Per-layer cache contents from `past_key_values[i]` (fla `Cache`, a `transformers` `Cache` subclass):

| Layer type | Tensors | Size |
|---|---|---|
| GDN | `recurrent_state` `(B, H=4, dk=128, dv=128)`, **float32 even when the model is bf16**: 262,144 B. Plus `conv_state` 3×`(B, 512, 4)` fp32: 24,576 B. | constant in T |
| Attention | `attn_state` = (K, V), each `(B, T, 512)`: 2·T·d·bytes | linear in T: 2 KB/token in bf16 |

Measured live memory after prefill, minus the retained logits tensor (T × 16384 × 2 B):

| ctx | all-GDN (12 GDN) | r=1/6 (2 attn) | all-attn (12 attn) |
|---|---|---|---|
| 16,384 | ≈3 MB | ≈66 MB | ≈384 MB (= 12·2·16384·512·2 B exactly) |

- **INFERENCE (from the formula):**
  - At 32K, one attention layer's KV cache is 64 MiB per sequence (bf16), against 0.28 MB for one GDN layer.
  - The crossover where one attention layer's KV equals one GDN state is about 128 tokens with fp32 state, or 64 tokens if the state were bf16.
  - So for every eval length here (≥1K), state memory is dominated by n_attn_layers × T.
- **Pitfall:** the HF forward returns `logits` for every position, which at 16K × 16K-vocab is 512 MB. That dwarfed the GDN cache in my first measurement. Memory benchmarks must compute logits only for the last position, or use `logits_to_keep=1` where supported.

### 2.5 Training throughput (VERIFIED-HERE, **shared GPU, indicative only**)
Setup: L=12, d=512, V=16384 tied, B=1×4096 tokens, bf16, AdamW (fused), eager, single packed document.

| Config | params | step | tok/s | peak mem |
|---|---|---|---|---|
| all-GDN (r=0) | 49.42M | 46 ms | 89.6k | 1.32 GB |
| r=1/6, attn @[5,11] | 49.40M | 62 ms | 66.0k | 1.28 GB |
| r=1/2 (odd layers) | 49.36M | 69 ms | 59.5k | 1.21 GB |
| all-attn (r=1) | 49.30M | 56 ms | 72.8k | 1.11 GB |

- A rerun of r=1/6 a few minutes later gave **39 ms/step (104k tok/s)**. That is about 1.6× run-to-run variance from contention with CARLA and other agents. These numbers say nothing about relative architecture speed. They only fix the order of magnitude for budgeting.
- **INFERENCE (budget):** at ≈70–100k tok/s, 1B tokens for a 50M model (≈20 tok/param) takes ≈3–4 GPU-h at T=4K. A sweep of ~12 configs × 1B tokens is ≈40–50 GPU-h before seeds.
  - Larger micro-batches (e.g., 16–32K tokens/step) should raise utilisation. I tested only 4K tokens/step because of the 2 GB cap.
  - At T=32K, causal attention FLOPs/token/layer (≈2·T·d ≈ 33.5M) exceed projection+MLP FLOPs (≈6.8M), so all-attention models train ~4–5× slower per token at 32K than at 4K (INFERENCE).
  - Plan long-context training as a short final phase, and give every ratio the same token budget.

### 2.6 Prefill / decode latency (VERIFIED-HERE, eager, B=1, shared GPU)

| Config | ctx | prefill | decode (ms/token) |
|---|---|---|---|
| all-GDN | 1K / 8K / 16K | 14 / 16 / 41 ms | 6.8 / 6.8 / 7.1 |
| r=1/6 | 1K / 8K / 16K | 14 / 20 / 47 ms | 7.0 / 7.0 / 6.8 |
| all-attn | 1K / 8K / 16K | 7 / 26 / 73 ms | 5.2 / 5.2 / 5.3 |

- Decode is **flat in context length and slightly *faster* for all-attention**. That is the signature of kernel-launch/Python overhead (fla's GDN step launches more kernels than SDPA), not memory bandwidth.
- **Prefill does show the expected quadratic trend:** all-attn goes 7 → 73 ms from 1K to 16K, while all-GDN goes 14 → 41 ms.
- Implications for the decode benchmark (INFERENCE):
  1. Write a static-buffer decode loop and capture one decode step with `torch.cuda.CUDAGraph`, or with `torch.compile(mode="reduce-overhead")` on a static cache. fla's `FLALayer.is_compileable = False` (`fla/models/cache_utils.py`), and `fused_recurrent_gated_delta_rule` returns a new final-state tensor, so the graph must `copy_` it into a static buffer.
  2. Sweep batch size (1, 8, 32, 128) and context (1K–32K), because the KV-cache bandwidth cost only shows up at large B·T.
  3. Report a "max batch that fits in X GB at context T" metric, which is architecture-sensitive and launch-overhead-free.

### 2.7 The working shim (load-bearing; this is what made fla hybrids run without `flash-attn`)
```python
import torch, torch.nn.functional as F
from torch.nn.attention.varlen import varlen_attn
from torch.nn.attention.bias import causal_lower_right
def flash_attn_func(q, k, v, causal=True, window_size=(-1, -1), **kw):      # q,k,v: [B,T,H,D]
    q, k, v = (t.transpose(1, 2) for t in (q, k, v)); Tq, Tk = q.shape[2], k.shape[2]
    if Tq == Tk:   o = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
    elif Tq == 1:  o = F.scaled_dot_product_attention(q, k, v)               # decode step
    else:          o = F.scaled_dot_product_attention(q, k, v, attn_mask=causal_lower_right(Tq, Tk))
    return o.transpose(1, 2)
def flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                           causal=True, window_size=(-1, -1), **kw):        # q,k,v: [total,H,D]
    return varlen_attn(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                       window_size=(window_size[0], 0) if causal else window_size)
import fla, fla.layers.attn as _A          # patch AFTER import; do NOT fake sys.modules['flash_attn']
_A.flash_attn_func, _A.flash_attn_varlen_func = flash_attn_func, flash_attn_varlen_func
```
- GQA (`num_kv_heads < num_heads`) needs `repeat_interleave` or `enable_gqa=True`. Sliding-window attention is not shimmed in the dense path.
- **Recommendation:** in our own code, write a small `SDPAAttention` module (RoPE + SDPA/varlen + our own KV cache) rather than monkeypatching. Keep the shim only for running fla's reference models as a cross-check.

### 2.8 `torch.compile` (VERIFIED-HERE)
- Compiling the whole fla hybrid made a training step **slower**: 53 ms vs 39 ms eager, with a first step of 5.9 s.
- Dynamo hit `recompile_limit (8)` because fla modules carry integer `layer_idx` attributes that Dynamo treats as static, which forces one recompile per layer.
- Options:
  - (a) Don't compile; fla already fuses norm, SwiGLU and CE.
  - (b) Compile only the MLP/norm sub-blocks.
  - (c) Set `torch._dynamo.config.allow_unspec_int_on_nn_module = True` (the hint printed by Dynamo; untested).

---

## 3. Recommended stack: reuse vs write

| Component | Recommendation | Why (evidence) |
|---|---|---|
| Linear mixer kernels | **Reuse** `fla.layers.GatedDeltaNet` / `fla.ops.gated_delta_rule.{chunk,fused_recurrent}_gated_delta_rule` from fla 0.6.0 (PyPI; matches clone `98193113ecb7`) | VERIFIED-HERE on sm_120, including varlen and cache. Blackwell-specific workarounds are in the code (§4.2). The NVlabs repo itself recommends fla kernels as "faster" with varlen support [code-inspected NVlabs/GatedDeltaNet README]. |
| Short conv | fla's Triton `ShortConvolution` (default `backend='triton'`) | Avoids `causal-conv1d`, which has no wheel for torch 2.11 (§4.4). |
| Full attention | **Write** a ~60-line module: RoPE (fla's `RotaryEmbedding` or own) + `F.scaled_dot_product_attention(is_causal=True)` for dense, `torch.nn.attention.varlen.varlen_attn` for packed docs, own KV cache | No external dependency. VERIFIED-HERE fwd/bwd/decode. Avoids fla's hard `flash_attn` import. |
| Block / model | **Write**: pre-norm block `x + mixer(RMSNorm(x))`, `x + SwiGLU(RMSNorm(x))`, with the mixer chosen per layer from a list `layer_types`. Mirror the fla/Qwen3-Next convention (`layer_types: list[str]`). | Full control of param matching, init, placement, logging. fla's `GatedDeltaNetConfig(attn={'layers': [...]})` is a convenient cross-check; it uses the same mechanism (`fla/models/configuration_utils.py:get_attention_spec`). |
| Fused CE | Reuse `fla.modules.FusedLinearCrossEntropyLoss` (or chunked CE) when T ≥ 16K | Logits at T=32K, V=16K are 1 GiB in bf16 (2 GiB fp32 grads) per sequence (INFERENCE). fla warns the fused-linear CE may reduce precision [code-inspected `configuration_utils.validate_loss_config`]. Compare against the plain CE path in a pilot. |
| Trainer | **Write** a nanoGPT-style single-GPU loop: uint16 memmap shards, BOS-delimited packing with `cu_seqlens`, AdamW (fused) or `torch.optim.Muon`, WSD/cosine schedule, bf16 autocast, grad-clip 1.0, deterministic data order shared across all ratios | nanoGPT is 336+330 lines and uses memmap uint16 + SDPA [code-inspected nanoGPT `3adf61e154c3`]. `flame` targets multi-GPU torchtitan, pins an old torchtitan commit (`0b44d4c`) and declares `transformers>=4.45`, whereas fla now needs `transformers>=5.6.2`. That is a version-conflict risk for no benefit on one GPU [code-inspected flame `24dea9133e21`]. |
| Optimizer | AdamW baseline. `torch.optim.Muon` (built into torch 2.11: lr 1e-3, wd 0.1, momentum 0.95, nesterov, 5 NS steps) as an optional pilot for 2-D hidden weights only | Muon is ~1.35× faster to target on the NanoGPT speedrun (author's claim) [read-full kellerjordan.github.io/posts/muon]. It applies only to 2-D hidden weights; embeddings, head and scalars use AdamW. Untested on GDN's `A_log`/`dt_bias`, which must go to AdamW without weight decay. fla marks them `_no_weight_decay = True` [code-inspected `fla/layers/gated_deltanet.py`]. |
| Speedrun tricks | **Do not** import modded-nanogpt's architectural tricks (value embeddings, U-net skips, attention-window warmup, FlexAttention sliding windows, logit softcap) | They change the attention/sequence-mixing pathway itself and would confound the full-vs-linear manipulation [code-inspected modded-nanogpt README record table `4ea6b937337a`]. Only optimizer-level tricks are neutral. |
| Synthetic recall training/eval | Reuse **zoology**'s MQAR generator `zoology/data/multiquery_ar.py` (pure numpy; params `num_kv_pairs`, `input_seq_len`, `power_a=0.01`, `random_non_queries`). Its GDN mixer imports fla. | [code-inspected zoology `1ad20d193b61`] |
| Long-context eval | **RULER** via lm-eval-harness (`lm_eval/tasks/ruler`: 13 tasks niah_*, vt, cwe, fwe, qa_squad, qa_hotpot; needs a tokenizer; default max length 4096; set `--metadata='{"max_seq_lengths":[1024,...,32768]}'`). **BABILong** via lm-eval (`babilong_qa1..qa20`) or the RMT-team repo (training split: 5k samples per task and length). | [code-inspected lm-eval `d6de81643928`, ruler `db43f92ff634`, babilong `7a6efee29f5c`]. **See pitfall P13**: 20–60M base models will score ~0 zero-shot on instruction-style prompts. |
| Perplexity-vs-position | fla `evals/ppl.py` (default data `fla-hub/pg19`, `--block_size 28672 --bucket_size 2048`), or our own | [code-inspected fla `evals/ppl.py`] |
| Decode benchmark | **Write** our own static-cache + CUDA-graph loop (§2.6) | HF `generate` is launch-bound at this scale (VERIFIED-HERE). |

**Do not use** (for this project):
- `flash-attn` FA2/FA3/FA4 packages (§4.3).
- `mamba_ssm` / `causal-conv1d` CUDA extensions (§4.4).
- `flame` / torchtitan (multi-GPU machinery, version conflicts).
- litgpt (not inspected; no need).
- The NVlabs GatedDeltaNet / Samba training codebases: lit-gpt-era, SlimPajama loaders, Slurm 4×8-GPU scripts [code-inspected `gated_deltanet/scripts/tsz512x4k_15B_gated_deltanet_h1_0.4B.sh`].

---

## 4. Package / kernel status on sm_120 (Blackwell consumer)

### 4.1 Triton
- **ESTABLISHED:** PyTorch 2.7 was the first stable line with Triton 3.3 and Blackwell support (pytorch.github.io PR #1989, via search snippet [abstract-only]: https://github.com/pytorch/pytorch.github.io/pull/1989).
- A later Triton bug mis-mapped sm_120 to `sm_120a` (tested on torch 2.9.1 / Triton 3.5.1), fixed in triton PR #9734 (search snippet [abstract-only]: https://github.com/triton-lang/triton/pull/9734).
- **VERIFIED-HERE:** Triton 3.6.0, bundled with torch 2.11.0+cu128, compiles and runs all GDN chunk/recurrent kernels plus flex_attention on sm_120.

### 4.2 fla on Blackwell: known issues and the workarounds in the code [code-inspected fla `98193113ecb7`]
- `fla/utils/hardware.py` defines `IS_NVIDIA_SM120 = capability == (12,0)` and `IS_NVIDIA_BLACKWELL = major in (10,12)`. On Blackwell it registers a Triton global-scratch allocator to avoid NullAllocator crashes (triton-lang/triton#10002).
- `fla/ops/utils/op.py`: on Blackwell, `tl.dot` is wrapped in inline asm to stop the `TritonGPUHoistTMEMAlloc` pass fusing add into dot.
  - Origin: fla issue #638, "GatedDeltaNet backward error on Blackwell: operand #2 does not dominate this use".
  - Opened 2025-11-11 (B200, Triton 3.5); fixed by PR #687 on 2025-12-19 [read-full https://github.com/fla-org/flash-linear-attention/issues/638].
- `fla/ops/common/chunk_o.py`: "Triton<3.8 races at 8 warps on Blackwell" (triton-lang/triton#10590, fla #1327). With Triton 3.6 the 8-warp configs are dropped automatically. **Implication:** our Triton (3.6) is in the "<3.8" branch; fla handles it.
- `fla/ops/gated_delta_rule/wy_fast.py` and `chunk_delta_h.py`: autotune restricted to `num_warps=[2]` on Blackwell (fla #913).
- `fla/ops/gated_delta_rule/backends/flash_qla.py`: Qwen's FlashQLA TileLang backend ships an **SM120 bf16 forward-only** kernel, used only for grad-free bf16 calls (inference). Not installed here, so the Triton path is used everywhere. Disable with `FLA_FLASH_QLA=0` if it gets installed, so train and eval numerics stay identical.
- `fla/layers/attn.py`: hard dependency on `flash_attn` (see §2.3).
- Env vars of interest (`ENVs.md`): `FLA_CONV_BACKEND` (`triton` default | `cuda`), `FLA_USE_TMA=0` (default), `FLA_TRIL_PRECISION=ieee` (default), `FLA_DISABLE_BACKEND_DISPATCH`.
- Install: `pip install flash-linear-attention[cuda]` (extras `cuda = torch>=2.7.0, triton>=3.3`). Base deps are `transformers>=5.6.2,<6` and `einops` (`pyproject.toml`).
  - I installed `uv pip install flash-linear-attention einops` into the shared venv. That **added transformers 5.22.2, tokenizers 0.23.3, huggingface-hub 2.4.0, safetensors 0.8.0** (other agents should know).

### 4.3 FlashAttention (Dao-AILab) on sm_120
- **FA2 (`flash-attn` 2.8.x):**
  - The source `setup.py` builds `FLASH_ATTN_CUDA_ARCHS` default `"80;90;100;110;120"`, with `compute_120f` on CUDA ≥ 12.9 [code-inspected flash-attention `478b5b012f9e`, `setup.py` L75–76, L103–167].
  - **Prebuilt wheels:** the latest release `v2.8.3.post2` (2026-10-13) has cp312 linux x86_64 wheels only for **torch 2.4–2.8** (queried via GitHub API, VERIFIED-HERE). There is **no wheel for torch 2.11**, and building needs nvcc, which we don't have.
  - Community reports say compiling FA2 for sm_120 crashes nvcc on backward kernels. This comes from a search-result summary only (not opened) [abstract-only, search result pointing to https://github.com/Dao-AILab/flash-attention/issues/2361]; treat as unverified.
  - Issue #1987 ("Support for NVIDIA Blackwell (sm_120)…", opened 2025-11-05, RTX PRO 6000) is still open, with no maintainer fix released. A contributor says FA2 is arch-agnostic and points to PRs #2222/#2268 [read-full https://github.com/Dao-AILab/flash-attention/issues/1987].
- **FA3 (`hopper/`):** H100/H800 only (README) [code-inspected].
- **FA4 (`flash-attn-4`, CuTeDSL, PyPI 4.0.0b38):** the repo now contains `flash_fwd_sm120.py` and `flash_bwd_sm120.py`. These **use SM80-era `mma.sync` with SM120's 99 KB SMEM**: "SM120 forward only supports num_splits=1" and "does not support paged KV or block sparsity" (`flash_attn/cute/config.py`) [code-inspected].
  - So on the 5080, FA4 brings no tcgen05/TMEM advantage over the FA2-class kernel already inside PyTorch (INFERENCE). Not tested.
- **Decision:** use PyTorch SDPA (flash or cuDNN backend) + `varlen_attn`. Both VERIFIED-HERE. One search summary also recommends SDPA → cuDNN on SM120 [abstract-only, search snippet; not relied on].

### 4.4 mamba_ssm / causal-conv1d
- Both `setup.py`s add `arch=compute_120,code=sm_120` [code-inspected mamba `e9594ce1c732` L194; causal-conv1d `cd81f0413cad` L192].
- PyPI has **sdist only**: mamba-ssm 2.3.2.post1, causal-conv1d 1.7.0.
- GitHub release wheels (VERIFIED-HERE via API) exist for cp312 x86_64 with `cu12torch{2.6…2.10}` and `cu13torch{2.9,2.10}` (plus NGC-style `25.xx/26.xx` tags), **but not torch 2.11**.
- Not needed: fla provides a Triton Mamba2 layer (`fla/layers/mamba2.py`) and Triton short-conv. If a Mamba2 control arm is ever wanted, one option is to pin torch 2.10 + cu128 and use the `cu12torch2.10cxx11abiTRUE` wheels (INFERENCE; untested on sm_120).

---

## 5. Datasets: numbers, licences, recommendation

### 5.1 Document-length distribution of FineWeb-Edu (VERIFIED-HERE)
- Source: shard `sample/10BT/000_00000.parquet` of FineWeb-Edu, 726,000 docs. I used the dataset's own `token_count` column (GPT-2 tokens), read remotely via `HfFileSystem` + pyarrow, column projection only.
- Summary: mean 1,040, median 628, p99 7,989, max 156,268.

| threshold | % of docs above | % of tokens in such docs |
|---|---|---|
| >2,048 | 9.275% | 40.91% |
| >4,096 | 3.021% | 24.24% |
| >8,192 | 0.951% | 13.14% |
| >16,384 | 0.203% | 5.21% |
| >32,768 | 0.037% | 1.79% |
| >65,536 | 0.006% | 0.46% |

- The HF datasets-server statistics over 1.455M rows of sample-10BT agree: mean 1,036.9, median 631 [read-full https://datasets-server.huggingface.co/statistics?dataset=HuggingFaceFW/fineweb-edu&config=sample-10BT&split=train].
- **INFERENCE:** a 2B-token FineWeb-Edu draw contains about 36M tokens in >32K-token documents and about 263M tokens in >8K docs. That is enough for 8K training, but thin for 32K. It is also mostly long web pages, not structured long documents.

### 5.2 Candidate corpora

| Dataset | Licence | Size | Long-doc content | Access | Verdict |
|---|---|---|---|---|---|
| **FineWeb-Edu** `HuggingFaceFW/fineweb-edu` | ODC-By | default 1.53B rows; sample-10BT 9.67M rows; sample-100BT 97.3M; sample-350BT 339M; Parquet [abstract-only, dataset card] | §5.1 | open | **Primary short/medium corpus.** Download `sample-10BT` (≈10B GPT-2 tokens; 14 shards of ≈2 GB; INFERENCE from shard 000_00000, 726K rows). |
| **PG19** `deepmind/pg19` | Apache-2.0 | train 28,602 / val 50 / test 100 books; 11.74 GB download [abstract-only, dataset card]. The original repo is script-based (viewer disabled). Parquet mirrors: `emozilla/pg19` (23 train shards of 0.26–0.30 GB + test 24.9 MB), `fla-hub/pg19`. | Train text median ≈301K chars, mean ≈379K chars (partial stats, 13.7K rows) [read-full datasets-server statistics] → median ≈80K tokens with our 16K BPE (3.44 B/token on PG19, §6). Essentially every book >32K. | open | **Long-document source** for the 8K–32K phase. Old (pre-1919) English is a domain shift. Use the PG19 *test* split for long-context perplexity only if it is held out from training. BABILong also uses PG19 as haystack, so exclude BABILong's haystack books or be aware of overlap. |
| **Common Pile arXiv (filtered)** `common-pile/arxiv_papers_filtered` | per-document CC BY / CC BY-SA / CC0; the card warns of possible license mis-assignment | 304,048 docs, ~19 GB UTF-8 text, 6.04 GB Parquet [abstract-only, card] | long technical papers (many >8K tokens; not measured) | open | Optional second long-doc domain (structured, cross-referencing). Paper: Common Pile v0.1, arXiv 2506.05209 [abstract-only via card]. |
| ProLong data `princeton-nlp/prolong-data-64K` | not stated on card | ≈31B tokens, 149 GB, **pre-tokenized with Llama-3 tokenizer**, MDS format, packed to 65,536 with doc `indices` [abstract-only, card] | code repos 20.9%, SlimPajama books 20.9%, FineWeb-Edu 20.9%, FineWeb 20.9%, plus small parts | open | **Not usable directly**: wrong tokenizer and 149 GB. Use as a *recipe* reference: long data (code repos + books) mixed with high-quality short data; train longer than eval length (ProLong, arXiv 2410.02660 [abstract-only]). |
| `togethercomputer/Long-Data-Collections` | — | — | — | **HTTP 401 / "not accessible without authentication"** (both card and datasets-server) | Not assessed. |
| SlimPajama-627B `cerebras/SlimPajama-627B` | — | — | — | **HTTP 401** at time of check | Not assessed. `DKYoon/SlimPajama-6B` mirror: 5.51M rows, 14.0 GB Parquet (datasets-server). Licence of mirror not checked. |
| TinyStories `roneneldan/TinyStories` | CDLA-Sharing-1.0 | 2.14M stories, 7.62 GB total files [abstract-only, card] | mean 896 chars, max 5,499 chars (≈ <1.5K tokens) [read-full datasets-server statistics] | open | **Reject for this study** (no long documents). Useful only for a 5-minute pipeline smoke test. |

### 5.3 Recommendation (INFERENCE; for methodology to decide)
- **Phase 1 (bulk, T=4K or 8K):** FineWeb-Edu sample-10BT, BOS/EOS-separated, packed with `cu_seqlens` document masking for *both* mixer types.
- **Phase 2 (long-context, T=16K–32K, last ~10–20% of tokens):** about 50% long docs (PG19 train + Common Pile arXiv, keeping only docs ≥ T/2 tokens) and 50% FineWeb-Edu, following ProLong's "long + high-quality short" mixing advice [abstract-only 2410.02660].
- **Identical shard order and token budget for every ratio:** pre-tokenize once into uint16 `.bin` shards with a fixed permutation seed.
- **Disk:** 10BT FineWeb-Edu ≈ 28 GB Parquet + PG19 ≈ 6.5 GB + arXiv ≈ 6 GB. Tokenized uint16 ≈ 2 bytes/token, so 11B tokens ≈ 22 GB. That fits easily in the 436 GB free.
- **Hold-out:** reserve PG19 test (100 books) and a FineWeb-Edu shard for loss-vs-position curves.

---

## 6. Tokenizer recommendation (VERIFIED-HERE numbers)

**Problem:** at 20–60M params, embedding params dominate.

| V | embed params @ d=384 | @ d=512 | @ d=640 | share of a 50M (d=512) model, tied |
|---|---|---|---|---|
| 50,257 (GPT-2) | 19.3M | 25.7M | 32.2M | 51% |
| 32,768 | 12.6M | 16.8M | 21.0M | 34% |
| 16,384 | 6.3M | 8.4M | 10.5M | 17% |
| 8,192 | 3.1M | 4.2M | 5.2M | 8% |

Untied doubles these. Pythia-70m is the cautionary example: vocab 50,304, d=512, untied (verified from its HF `config.json`), so ≈51.5M of its ~70M params are embeddings. SmolLM2-135M uses V=49,152, d=576, tied.

**Compression experiment (VERIFIED-HERE):**
- Training set: 117K FineWeb-Edu docs (550M chars) from `sample/10BT/001_00000.parquet`.
- Tokenizer: HF `tokenizers` 0.23.3 byte-level BPE with single-digit isolation (`Split(\d, isolated)` + `ByteLevel`). Training took 11–14 s.
- Evaluation: 3,000 held-out FineWeb-Edu docs, plus 20 PG19-test books truncated to 200K chars.

| tokenizer | FineWeb-Edu bytes/token | tokens vs GPT-2 | PG19 bytes/token | tokens vs GPT-2 |
|---|---|---|---|---|
| GPT-2 50,257 | 4.651 | 1.000 | 3.786 | 1.000 |
| BPE 32,768 + digits | 4.509 | 1.031 | 3.648 | 1.038 |
| **BPE 16,384 + digits** | **4.238** | **1.098** | **3.439** | **1.101** |
| BPE 8,192 + digits | 3.873 | 1.201 | 3.195 | 1.185 |

- GPT-2 tokenises `passkey 123456789` as `['pass','key','Ġ123','45','67','89']`. Our tokenizer gives one token per digit, so numeric needles have uniform, length-predictable token counts across RULER / passkey tasks. **INFERENCE:** that matters for retrieval accuracy at tiny scale and for exact control of context length.
- **Recommendation:** V = 16,384 (a multiple of 128, good for matmul tiling).
  - Train it on the *actual mix* (FineWeb-Edu + PG19 + arXiv), which should narrow the PG19 gap.
  - Reserve special tokens `<|endoftext|>` (BOS/EOS doc separator), `<|pad|>`, and a few task tokens for synthetic tasks (e.g., `<|kv|>`, `<|query|>`).
  - Tie embeddings.
  - Report context lengths in *our* tokens, and note the ≈10% offset relative to GPT-2 tokens when comparing to literature.
  - RULER/lm-eval require a HF tokenizer: save as `PreTrainedTokenizerFast` with `tokenizer.json`.

---

## 7. Model configuration notes for a controlled ratio study

### 7.1 Parameter matching (VERIFIED-HERE counts)
- A standard MHA attention layer (no bias) is 4d² params.
- fla's GDN default (`use_gate=True, expand_v=2, head_dim=256, num_heads=6` at d=2048) is about 6d² (docstring in `fla/layers/gated_deltanet.py`).
- **Swapping default GDN for attention changes params per layer by 50%.** The flame 340M configs compensate with depth: GDN 21 layers vs Transformer 24 layers [code-inspected flame `configs/*.json`]. That is impossible in a fixed-depth ratio sweep.
- **Option A (recommended primary):**
  - GDN `use_gate=False, expand_v=1, num_heads=H_l, head_dim=d/H_l`, so q, k, v, o = 4d², plus a_proj/b_proj (2·d·H_l), conv (3·d·4), and A_log/dt_bias/o_norm.
  - At d=512, H_l=4: 1,058,952 vs 1,048,576 for MHA with 8×64 heads (+0.99%/layer). Whole-model spread across r=0…1 is 0.24%.
  - fla warns that the output gate is "crucial"; it is GDN's swish-gated RMSNorm output.
- **Option B (gated pair):**
  - GDN `use_gate=True, expand_v=1` (q, k, v, g, o = 5d²) vs attention with a sigmoid output gate (`q_proj` → 2d, `attn_out * sigmoid(gate)`, as in transformers' `Qwen3NextAttention`; `modeling_qwen3_next.py` L269–300 [code-inspected]). That is also 5d².
  - Use as a robustness check if pilots show Option A's GDN is badly weakened.
- **Short conv:** GDN has it (`use_short_conv=True`); attention does not. That is part of what "linear layer" means here (fla: "ShortConvolution is crucial to the performance"). Document it explicitly as part of the mixer definition.

### 7.2 Positional encoding (confound for H5)
- fla's `Attention` always applies RoPE (`self.rotary = RotaryEmbedding(...)` unconditionally). GDN uses no positional encoding.
- If models are trained at ≤8K and evaluated at 32K, attention layers face RoPE extrapolation failure while GDN layers don't. That would conflate "full attention helps long context" with "RoPE fails beyond train length".
- **INFERENCE / options:**
  - (i) Train the final phase at the max eval length (32K).
  - (ii) Use a large `rope_theta` (e.g., 5e5) from the start.
  - (iii) Test NoPE attention in hybrids as an ablation.
  - Whatever is chosen must be identical across ratios.

### 7.3 Placement (for RQ-B / H4)
fla's `attn={'layers':[...]}` (or our `layer_types`) makes any placement a one-line config change. Reference patterns from code:

| Model | Placement | Source |
|---|---|---|
| Qwen3-Next | `full_attention_interval=4` → layers 3, 7, 11, … (last of each 4-block) | [code-inspected transformers 5.22.2 `configuration_qwen3_next.py` L130–134] |
| Qwen3.5 | same default (`full_attention_interval=4`) | [code-inspected `configuration_qwen3_5.py` L112–115] |
| Olmo Hybrid | `i % 4 == 3`; forces at least one attention layer (last) | [code-inspected `configuration_olmo_hybrid.py` L122–130] |
| Jamba | `attn_layer_period=8, attn_layer_offset=4` | [code-inspected `configuration_jamba.py` L89–111] |
| Jet-Nemotron-2B | 28 layers. Full attention at **0-indexed 14 and 19**; SWA (window 1152) at 20 and 21; JetBlock elsewhere. Found by PostNAS beam search on a once-for-all supernet. "Only two layers exhibit critical importance" for MMLU, and 2–3 for retrieval; the important layers differ by task. | [code-inspected Jet-Nemotron `94cbd1240463` `jetlm_2B.json`; read-section arXiv 2508.15884 HTML] |
| Jet-Nemotron-4B | 36 layers. Full attention at 0-indexed 17, 20, 32; SWA at 5, 16, 19, 21, 22, 25, 27 (window 2048). The repo config matches the paper's *text* (1-indexed 18, 21, 33); the paper's Table 9 differs. | [code-inspected `jetlm_4B.json`; read-section] |
| Hymba barebones test | 32-layer stack with global (full) attention at [5, 11, 18, 25, 31], SWA elsewhere | [code-inspected hymba `df5b2a95b990` `barebones_hymba/test_barebones_hymba.py` L74] |
| NVlabs GatedDeltaNet-H1 | `gated_delta_per_layer=2`, so GDN alternates with **SWA (window 2048)**, i.e. local, not full attention | [code-inspected `gated_deltanet/lit_gpt/config.py` L119–139] |
| Samba | Mamba + SWA (window 2048) | [code-inspected Samba `617c7a0f8c71` `lit_gpt/config.py`] |

Notes:
- **Nemotron-H** (arXiv 2504.03624) mixes attention, Mamba-2 and MLP layers. It is **not** a controlled full-vs-linear ratio experiment and is cited here only as an architecture.
- **SWA hybrids (Samba, GDN-H1, Hymba's SWA layers) are a different manipulation** (local vs global softmax) from ours.

### 7.4 Suggested base shape (INFERENCE)
- **~50M:** L=12, d=512, SwiGLU intermediate 1536 (fla `hidden_ratio=4` → 1536 after rounding), V=16384 tied.
  - Non-embedding 41M, embedding 8.4M. Attention 8 heads × 64; GDN 4 heads × 128.
  - Ratios on 12 layers: r ∈ {0, 1/12, 2/12, 3/12, 4/12, 6/12, 12/12}, i.e. 0, 1, 2, 3, 4, 6, 12 attention layers.
- **~25M:** L=12, d=384 (attention 6×64; GDN 3×128).
- **12 layers is the minimum for a useful ratio grid.** 24 layers at d=384 would give finer ratios (1/24) at about the same params, but is slower per token.

---

## 8. Pitfalls checklist

| # | Pitfall | Evidence | Mitigation |
|---|---|---|---|
| P1 | fla `Attention` raises ImportError without `flash_attn`, and no flash-attn wheel exists for torch 2.11 | code + GitHub API (VERIFIED-HERE) | Own SDPA attention module, or the shim in §2.7 |
| P2 | Faking `sys.modules['flash_attn']` breaks `transformers` import (`__spec__ is None`) | VERIFIED-HERE | Patch `fla.layers.attn` module attributes after import |
| P3 | First Triton call ≈50 s (JIT + autotune), and autotune can pick different configs run to run | VERIFIED-HERE | Warm up before timing. Persist `TRITON_CACHE_DIR`. fla autotune cache env vars are in `ENVs.md` (not tested). |
| P4 | Triton < 3.8 has an 8-warp race on Blackwell; fla restricts configs | code-inspected `chunk_o.py` | Keep fla ≥ 0.6.0; record Triton version in every run |
| P5 | **Shared GPU makes timings unreliable** (66k vs 104k tok/s for the same config minutes apart) | VERIFIED-HERE | Run all RQ-D benchmarks with CARLA stopped. Report median of ≥5 repeats. Log `nvidia-smi` util/clocks before and after. |
| P6 | Eager decode is launch-bound (≈5–7 ms/token flat in context) | VERIFIED-HERE | CUDA graphs + static cache + batch sweep |
| P7 | Retained full-sequence logits inflate memory measurements (512 MB at 16K × 16K vocab) | VERIFIED-HERE | Last-token logits only; measure cache tensors directly (`numel × element_size`) and also `max_memory_allocated` |
| P8 | GDN recurrent state is **fp32** in fla even for bf16 models | VERIFIED-HERE | Report state bytes with dtype; optionally also report a bf16-state variant analytically |
| P9 | Whole-model `torch.compile` is slower (recompile per layer on `layer_idx`) | VERIFIED-HERE | Don't compile, or compile sub-blocks |
| P10 | fla varlen requires packed batch=1 (`cu_seqlens` over the flattened batch); flame enforces `batch_size=1` with `--training.varlen` | code-inspected flame README | Pack B×T into 1×(B·T) with `cu_seqlens`; use the same masking for attention (`varlen_attn`) |
| P11 | Default GDN (6d²) ≠ attention (4d²) params | code-inspected | §7.1 Option A/B |
| P12 | RoPE extrapolation confound | code-inspected `fla/layers/attn.py` | §7.2 |
| P13 | **20–60M base LMs score near zero on zero-shot instruction-style RULER/BABILong** (INFERENCE). Zoology trains directly on MQAR; BABILong reports fine-tuned *recurrent memory transformers* handling 50M tokens [abstract-only 2406.10149]. | — | Use log-likelihood / exact-match on *trained* synthetic formats. Either mix synthetic tasks (MQAR, NIAH, variable tracking, BABILong-train) into pre-training at a fixed fraction for all ratios, or run a fixed per-task fine-tune. The eval protocol owner should decide. |
| P14 | Chunk (train) vs fused-recurrent (decode) GDN numerics differ (rel 2.5e-3 bf16, 1e-2 logits in fp32 model) | VERIFIED-HERE | Score evals with the chunk path (full forward), not via generation, where possible. fla auto-switches to `fused_recurrent` when `q_len ≤ 64` and not training (`fla/layers/gated_deltanet.py`). |
| P15 | fla `FlashQLA` backend (if installed) runs a different SM120 bf16 kernel at inference only | code-inspected | `FLA_FLASH_QLA=0`, or don't install it |
| P16 | Logits at long T: 32K × 16K vocab = 1 GiB bf16 per sequence | INFERENCE | Fused linear CE, or chunked CE for the 32K phase |
| P17 | Shared venv: fla pulled transformers 5.22.2 | VERIFIED-HERE | Pin in `pyproject.toml` / `uv.lock` |
| P18 | `deepmind/pg19` is script-based (viewer disabled) | dataset card | Use Parquet mirror `emozilla/pg19` (record revision hash) |
| P19 | GPT-2 tokenizer chunks digits irregularly | VERIFIED-HERE | Digit-split BPE (§6) |

---

## 9. Repo layout sketch (INFERENCE; for the infrastructure stage)

```
outeropt/
  pyproject.toml / uv.lock      # pin: torch==2.11.0+cu128, triton==3.6.0, flash-linear-attention==0.6.0,
                                #      transformers==5.22.2, tokenizers==0.23.3, einops==0.8.2, numpy, pyarrow, huggingface_hub
  configs/
    base_50m.yaml               # d, L, heads, vocab, rope_theta, optimizer, schedule, token budget, seeds
    ratios/r00.yaml ... r12.yaml      # layer_types lists only (inherit base)
    placement/{uniform,front,middle,back,end}_r2.yaml
  src/
    model/
      mixers.py                 # SDPAAttention (RoPE, dense/varlen, KV cache), GDNMixer (wraps fla.layers.GatedDeltaNet)
      block.py, lm.py           # pre-norm block, SwiGLU, tied embeddings, layer_types → mixers; param-count assert
      cache.py                  # static KV + recurrent state buffers; bytes accounting
    data/
      train_tokenizer.py        # 16K digit-split BPE on the mix → tokenizer.json (+ HF wrapper)
      prepare.py                # download (pinned revisions) → tokenize → uint16 shards + doc-boundary index
      loader.py                 # fixed-permutation packed batches with cu_seqlens; phase-1/phase-2 mixtures
      synthetic/                # MQAR (from zoology), NIAH/passkey, VT, BABILong-train adapters
    train.py                    # single-GPU loop; AdamW|Muon; WSD; bf16; logging (jsonl); checkpoints
    eval/
      ppl_by_position.py        # PG19-test / FineWeb hold-out loss vs position, 1K–32K
      synthetic_eval.py         # MQAR/NIAH/VT/BABILong scoring by log-likelihood / exact match
      ruler_lmeval.sh           # lm-eval RULER/BABILong with --metadata max_seq_lengths
    bench/
      state_memory.py           # analytic + measured cache bytes vs T
      prefill.py, decode_cudagraph.py   # idle-GPU, median of N, batch × context sweeps
  experiments/registry.jsonl    # one line per run: config hash, git sha, versions, GPU state, metrics paths
  third_party/                  # (gitignored) shallow clones listed in §10
```

---

## 10. Repositories inspected (shallow clones in `third_party/`)

| Repo | Commit (date) | What I looked at |
|---|---|---|
| fla-org/flash-linear-attention | `98193113ecb7` (2026-11-08) | layers/gated_deltanet.py, layers/attn.py, models/gated_deltanet, models/configuration_utils.py, models/modeling_utils.py, models/cache_utils.py, utils/hardware.py, ops/common/chunk_o.py, chunk_delta_h.py, ops/gated_delta_rule/{wy_fast,backends/flash_qla}.py, ops/utils/op.py, INSTALL.md, FAQs.md, ENVs.md, evals/, benchmarks/ |
| fla-org/flame | `24dea9133e21` (2026-04-23) | README, pyproject, configs/*.json |
| NVlabs/GatedDeltaNet (cloned by another agent as `gated_deltanet`) | `b53d6d3a1612` (2026-03-13) | README FAQ, lit_gpt/config.py, scripts |
| state-spaces/mamba | `e9594ce1c732` (2026-07-23) | setup.py arch flags; GitHub release wheel tags |
| Dao-AILab/causal-conv1d | `cd81f0413cad` (2026-08-20) | setup.py; release wheel tags |
| Dao-AILab/flash-attention | `478b5b012f9e` (2026-11-08) | README, setup.py, .github/workflows, flash_attn/cute/{flash_fwd_sm120,flash_bwd_sm120,config,interface}.py |
| karpathy/nanoGPT | `3adf61e154c3` (2025-11-12) | model.py, train.py |
| KellerJordan/modded-nanogpt | `4ea6b937337a` (2026-09-28) | README record table, train_gpt.py grep |
| HazyResearch/zoology | `1ad20d193b61` (2026-03-21) | data/multiquery_ar.py, mixers/hybrid.py, mixers/gated_delta_net.py, README |
| NVIDIA/RULER (cloned by another agent as `ruler`) | `db43f92ff634` (2026-11-03) | scripts/synthetic.yaml, data/synthetic |
| EleutherAI/lm-evaluation-harness | `d6de81643928` (2026-09-14) | tasks/ruler (README, utils), tasks/babilong |
| microsoft/Samba | `617c7a0f8c71` (2025-11-16) | lit_gpt/config.py |
| NVlabs/Jet-Nemotron | `94cbd1240463` (2025-11-30) | jetai/modeling/configs/jetlm_{2B,4B}.json, configuration_jet_nemotron.py |
| NVlabs/hymba | `df5b2a95b990` (2024-12-11) | barebones_hymba/test_barebones_hymba.py |
| pytorch/torchtitan | `3da9eac83509` (2026-11-08) | pyproject (not adopted) |
| booydar/babilong (cloned by another agent) | `7a6efee29f5c` | README task table, training-split link |
| princeton-nlp/HELMET (cloned by another agent) | `aeadacc6593e` | directory listing only |

Also inspected in the venv: transformers 5.22.2 `models/{qwen3_next,qwen3_5,olmo_hybrid,jamba,nemotron_h}/configuration_*.py`, `qwen3_next/modeling_qwen3_next.py`, and torch 2.11 `nn/attention/varlen.py`, `optim/_muon.py`.

**Housekeeping note:** while cloning I briefly created duplicate clones `third_party/RULER` and `third_party/GatedDeltaNet`. Other agents had cloned the same repos concurrently (as `ruler` and `gated_deltanet`), so I deleted my duplicates.

Not accessed or not inspected: litgpt, SmolLM training repo, torchtitan internals, Hymba full paper, Samba full paper, MiniMax-01, Jamba papers (abstract/code only as listed in the DB).

---

## 11. Sources (URLs and access levels)
- A Systematic Analysis of Hybrid Linear Attention (Wang et al., 2025), https://arxiv.org/abs/2507.06457. [read-section] via the HTML https://arxiv.org/html/2507.06457v1, summarised by fetch tool.
  - Ratios 24:1, 12:1, 6:1, 3:1 plus pure. Six linear variants. 340M/20B and 1.3B/100B tokens on FineWeb-Edu. RULER recall average 0.256 (pure linear) → 0.338 (24:1) → 0.390 (6:1) → 0.397 (3:1). LM stays flat. 2,048-token context. Param matching not stated. Models: https://huggingface.co/collections/m-a-p/hybrid-linear-attention-research-686c488a63d609d2f20e2b1e
- Gated Delta Networks (Yang, Kautz, Hatamizadeh, ICLR 2025), https://arxiv.org/abs/2412.06464 [abstract-only]; code https://github.com/NVlabs/GatedDeltaNet [code-inspected]
- flash-linear-attention, https://github.com/fla-org/flash-linear-attention [code-inspected]; issue #638 [read-full]
- flame, https://github.com/fla-org/flame [code-inspected]
- FlashAttention repo https://github.com/Dao-AILab/flash-attention [code-inspected]; issue #1987 [read-full]; issue #2361 (search snippet only) [abstract-only]
- Triton/PyTorch Blackwell: https://github.com/pytorch/pytorch.github.io/pull/1989, https://github.com/triton-lang/triton/pull/9734 (search snippets) [abstract-only]
- Mamba https://github.com/state-spaces/mamba, causal-conv1d https://github.com/Dao-AILab/causal-conv1d [code-inspected + release API]
- Muon blog https://kellerjordan.github.io/posts/muon/ [read-full]; modded-nanogpt https://github.com/KellerJordan/modded-nanogpt [code-inspected]
- nanoGPT https://github.com/karpathy/nanoGPT [code-inspected]
- Zoology https://arxiv.org/abs/2312.04927 [abstract-only]; https://github.com/HazyResearch/zoology [code-inspected]
- Based https://arxiv.org/abs/2402.18668 [abstract-only]
- RULER https://arxiv.org/abs/2404.06654 [abstract-only]; https://github.com/NVIDIA/RULER [code-inspected]
- BABILong https://arxiv.org/abs/2406.10149 [abstract-only]; https://github.com/booydar/babilong [code-inspected]
- lm-evaluation-harness https://github.com/EleutherAI/lm-evaluation-harness [code-inspected]
- Jet-Nemotron https://arxiv.org/abs/2508.15884 [abstract-only] + https://arxiv.org/html/2508.15884 [read-section]; https://github.com/NVlabs/Jet-Nemotron [code-inspected]
- Hymba https://arxiv.org/abs/2411.13676 [abstract-only]; https://github.com/NVlabs/hymba [code-inspected]
- Samba https://arxiv.org/abs/2406.07522 [abstract-only]; https://github.com/microsoft/Samba [code-inspected]
- ProLong https://arxiv.org/abs/2410.02660 [abstract-only]; data card https://huggingface.co/datasets/princeton-nlp/prolong-data-64K [abstract-only]
- MobileLLM https://arxiv.org/abs/2402.14905 [abstract-only] (embedding sharing at sub-billion scale)
- FineWeb-Edu card https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu [abstract-only]; shard statistics VERIFIED-HERE
- PG19 card https://huggingface.co/datasets/deepmind/pg19 [abstract-only]; mirror https://huggingface.co/datasets/emozilla/pg19
- Common Pile arXiv card https://huggingface.co/datasets/common-pile/arxiv_papers_filtered [abstract-only]; paper arXiv 2506.05209 (not opened)
- TinyStories card https://huggingface.co/datasets/roneneldan/TinyStories [abstract-only]
- SlimPajama-627B https://huggingface.co/datasets/cerebras/SlimPajama-627B and Long-Data-Collections https://huggingface.co/datasets/togethercomputer/Long-Data-Collections: **not accessible (HTTP 401)**
- transformers hybrid configs (Qwen3-Next, Qwen3.5, Olmo Hybrid, Jamba, Nemotron-H): installed transformers 5.22.2 [code-inspected]; Nemotron-H paper arXiv 2504.03624 is **not** a controlled ratio study (not opened by me).
- Small-model configs (Pythia-14m/70m, SmolLM2-135M, TinyStories-33M): HF `config.json` files (VERIFIED-HERE fetch).

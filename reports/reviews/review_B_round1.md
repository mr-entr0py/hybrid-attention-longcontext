# Review B, round 1: experimental and engineering validity (adversarial)

**Reviewer:** B (experimental and engineering validity; adversarial). **Date:** 2026-11-09.
**Object of review:** `reports/initial_proposal.md` v0.1 ("PROP"), with `reports/literature/literature_review.md` ("LR") and the specialist reports `01_nemotron_h.md` … `05_implementation.md`. I read all of them in full.
**Mandate:** try to show that the programme cannot establish its claimed conclusions, then fix it.

**How I checked things**
1. **Primary sources.** I re-fetched the sources that a decision in this review depends on. They are listed in Appendix B, each with an access tag.
2. **Code.** I inspected the fla clone at `98193113`, plus Triton 3.6.0 and torch 2.11.0+cu128 inside the project venv.
3. **Measurements on the project GPU, tagged [MEAS-B].**
   - CARLA was running throughout: 25–37 % utilisation, 5.3–7.3 GB used by other processes, SM clock 2.2–2.8 GHz.
   - My process was capped at 4.8–7.1 GiB.
   - Timings are CUDA-event min/median over 25–30 reps. They are therefore *contended*. The minimum is my primary estimator, because contention only ever adds time (justification: [Chen & Revels §B](https://arxiv.org/abs/1608.04295) [read-section]).
   - The scripts live in my session scratchpad, not in the repo. Appendix A gives enough detail to reproduce every number.

**Tags.**
- Access: [read-full], [read-section], [abstract-only], [code-inspected].
- Measurement: [MEAS-B] is mine. [MEAS] marks Stage-1 agents' measurements as they reported them.
- Epistemic: **[EST]** established in a cited source; **[HYP]** hypothesis; **[INF]** my inference or arithmetic.

---

## 0. Verdict in brief

**As written, PROP cannot establish its headline conclusions.** Two problems are fatal to the stated deliverables, and eight are major. All of them can be fixed within the budget, and Part 2 gives a design that fixes them.

**Top issues:**
1. **F1 (fatal). H1 and H5 are anchored on an all-full reference that nothing requires to be competent at long lengths, and they are pooled over lengths.**
   - The recipe makes an incompetent 16K–32K reference likely:
     - 763 Stage-B steps at 32K;
     - RoPE θ = 1e4 is a candidate base;
     - QK-norm logit dilution (see M3).
   - With such a reference, H1 can be "supported" by the 1K–8K cells while the reference itself collapses at 32K.
   - H5 can be "falsified" by the same collapse.
2. **F2 (fatal for RQ-D and the frontier). The planned efficiency measurements carry artefacts of up to 20–40×, with opposite signs.**
   - The natural static-cache CUDA-graph decode silently dispatches SDPA to the mem-efficient kernel: 1.8–2.0 ms vs 85–90 µs per layer at 32K [MEAS-B].
   - The 64 MiB L2 inflates attention decode speed by up to 2.5× [MEAS-B].
   - GDN layers are launch-bound at batch 1 and short T [MEAS-B].
   - The two sides run kernels of very different maturity.
3. **M1, M2 (major). Feasibility and exposure are unestablished.**
   - Nothing shows that a 41M-non-embedding model with 27 tokens per non-embedding parameter (0.1B of them at 32K) can learn 32K retrieval. The closest small-scale long-context hybrid study trained at 16K context for ≥ 100N tokens and reports no NIAH for its smallest models ([Qiao](https://arxiv.org/html/2606.15378v1) [read-section]).
   - The synthetic-exposure fix turns the main dependent variable into sample efficiency: transformers need "100x less samples" than GSSMs to learn copying ([Jelassi](https://arxiv.org/html/2402.01032) [read-section]).
4. **M5 (major). PROP's own decision rule makes H2, H3 and H4 unsupportable.**
   - "Supported" requires a seed-level t-interval. With the planned 2 seeds, df = 1 and t = 12.71, so the half-width is ≈ 27 pp at PROP's own σ_d = 3 pp.
   - The placement study (≈ 32 GPU-h) is therefore pre-determined to end "suggestive" at best.
5. **M4 (major). The reproducibility plan misplaces the source of nondeterminism.**
   - I measured the whole-model gradient: it is bitwise reproducible for k = 0, but not for k = 12 (FA2 backward). Deterministic mode fixes it for an estimated ≈ +7 % GPU-h across the core.
   - cuDNN SDPA has no deterministic kernel.
   - The G0(f) kernel-error thresholds would fail spuriously, because fla runs TF32 internally.

**What PROP gets right (I verified these).**
- Exact parameter matching: 49,395,808 (k = 0) and 49,297,408 (k = 12) total; 1,056,904 parameters per GDN mixer [MEAS-B, by instantiation].
- Document masking in both mixers.
- In-distribution evaluation lengths.
- Cloze-style, teacher-forced scoring; frozen item files.
- The LR regret rule.
- The Zoology reproduction gate.
- Tiered execution.
- **Per-run compute estimates are realistic.** I measured the proposed model end to end on the shared GPU:
  - Stage A: ≈ 88–90K tok/s at 4.5 GiB with 8K-token micro-batches.
  - Stage B at 32K with checkpointing: k = 0 → 70.5K tok/s, k = 3 → 51K, k = 12 → 27–32K, all at 2.6 GiB.
  - So G0(d), G0(e) and G4 pass with margin.
- **32K evaluation of the all-full model is computationally easy.** The attention core reaches 91.5 TFLOP/s at 32K, and a full suite takes about 0.5 h per k = 12 model. The binding constraint is learning, not compute.

**My design in one paragraph (Part 2).**
- **Feasibility first.** A ≈ 12 GPU-h pilot establishes, before anything else, that the all-full reference is competent at 32K and the all-linear model is not. A pre-registered rule then sets the long-stage budget.
- **Stage-A checkpoints feed several branches.** Each warmup-stable-decay (WSD) Stage-A checkpoint at the end of the stable phase is the starting point for several Stage-B and task-exposure variants. This makes the two main confounds (long-context budget and exposure) cheap factors.
- **Evaluation.** Difficulty ladders replace single tuned difficulty levels. The primary H1 cell is the 16K–32K band, behind an absolute reference-adequacy gate.
- **Allocation of runs.**
  - Core: k ∈ {0, 1, 3, 12} × 3 seeds.
  - Curve: k = 2 × 2 seeds.
  - Placement: where to put the *single* layer (k = 1 at layers 1, 6, 11) × 3 seeds.
- **Training determinism.** Training runs in PyTorch deterministic mode.
- **Efficiency.** Analytic costs are primary. Measured costs follow an artefact-controlled protocol: exact-shape CUDA graphs, pinned backends, cold and hot L2, FlexAttention as the equal-maturity comparator.
- **Budget (contended GPU-h, including 20 % contingency):** Tier A ≈ 115, Tier B ≈ 154.

---

## 1. Critique of the initial proposal

### 1.0 Summary table

| # | Severity | Issue (one line) | Fix (one line) | Extra GPU-h |
|---|---|---|---|---|
| F1 | **fatal** | H1/H5 are ratios on a reference whose long-length competence is never required; pooled over 1K–32K | Primary H1 on the 16K–32K band; absolute reference-adequacy gate G-REF; saturation branch; ladder-mean and N50 metrics | ≈ 5 |
| F2 | **fatal** (RQ-D/H6) | Decode/prefill benchmarks carry 20–40× dispatch, 2.5× L2 and launch artefacts, plus kernel-maturity asymmetry | Analytic-primary costs; exact-shape graphs; pinned and asserted backends; cold/hot L2; FlexAttention comparator; roofline table | ≈ 0 (re-engineering) |
| M1 | major | Long-context learnability at 27N tokens with 0.1B at 32K is unestablished, and the pilots cannot establish it | S2 feasibility and reference-adequacy pilot with a budget rule and a fallback ladder | ≈ +3 net |
| M2 | major | Synthetic exposure makes the DV a sample-efficiency measure; exposure is a hidden free parameter | Exposure becomes a factor: checkpoints 250/500/1000 plus a 4× continuation branch | ≈ 3.5 |
| M3 | major | QK-RMSNorm with shared unit gains caps logits at 8, giving length-dependent softmax dilution (needle weight 64 % → 5 % from 1K to 32K) | Per-head gains plus SSMax log-length query scaling, adopted by the S2 rule; diagnostics | ≈ 1 |
| M4 | major | Nondeterminism misattributed; cuDNN incompatible with the audit; k = 0 and k ≥ 1 carry asymmetric noise; G0(f) thresholds wrong | Deterministic mode in production; relative thresholds at T = 32K; archived Triton cache | ≈ +7 % |
| M5 | major | With 2 seeds, the seed-t half-width is ≈ 27 pp, so H2, H3 and H4 can never be "supported" under PROP's rule | Only arms with ≥ 3 seeds enter confirmatory tests; k = 1 placement triad × 3 seeds; H2 and H6 descriptive | ≈ −10 |
| M6 | major | Difficulty tuned on 0.2B-token pilots, then frozen for 1.1B-token models: ceiling or floor risk | Pre-registered difficulty ladders; ladder-mean EM; capacity N50 | 0 (evaluation only) |
| M7 | major | PE chosen from a single-seed pilot with only ≈ 150 steps at 32K; θ = 1e4 kept through 4K → 32K | Default RoPE θ = 5e5 from step 0; NoPE as a robustness arm | −2 |
| M8 | major | Shared-GPU operations: OOM interplay with CARLA, no clock locking, no fallback without idle windows | Runbook: memory cap, 6-minute checkpoints, auto-resume, analytic-primary contingency | 0 |
| m1–m9 | minor | Micro-batching; scoring path; autotune warm-up; evaluation cost; contingency; tautological H6; launch-bound prefill; LR horizon; logging | See §1.11 | — |

### 1.1 F1 (fatal): the H1/H5 statistics are uninterpretable without a competent long-length reference

**What PROP does.**
- H1 is R_ret(3) = (S(3) − S(0)) / (S(12) − S(0)), pooled over in-distribution lengths 1K–32K, with a non-inferiority margin (PROP §2).
- Gate G2 requires a ≥ 15 pp gap at **at least one** length and the all-full model ≥ 20 pp above floor (§2, §12).
- Nothing requires the all-full model to be competent at 16K–32K, which are exactly the lengths that make the question "long context".

**Why an incompetent reference is likely [INF, with evidence].**
1. **Thin long-length training.**
   - Stage B is 0.1B tokens = 763 optimiser steps at 32K, run during the tail of a linear decay that starts Stage B at ≈ 36 % of peak LR (PROP §8; my arithmetic from steps 6,294 → 8,392).
   - Stage A is document-masked FineWeb-Edu. Its median document is 628 GPT-2 tokens, and only 3.0 % of documents exceed 4K ([MEAS], 05 §5.1). So the attention layers meet long relative positions almost only in Stage B.
2. **Position encoding.** RoPE θ = 1e4, held constant through a 4K → 32K extension, is one of the three candidates. The choice is made from single-seed runs with a 20M-token, ≈ 150-step 32K extension (E1c; see M7).
3. **Softmax dilution.** QK-RMSNorm with shared unit-initialised gains bounds logits at √d_h = 8. That makes needle retrieval progressively harder with length unless the gains grow during Stage B (M3).

**Failure scenario (illustrative numbers).**

| | 1K | 2K | 4K | 8K | 16K | 32K |
|---|---|---|---|---|---|---|
| S(12, T) (%) | 90 | 88 | 85 | 80 | 62 | 45 |
| S(3, T) (%) | 90 | 88 | 86 | 82 | 66 | 50 |
| S(0, T) (%) | 70 | 50 | 30 | 20 | 12 | 10 |

- **H1 is "supported", strongly.** Pooled R(3) = Σ(S3 − S0) / Σ(S12 − S0) = 270 / 258 ≈ **1.05**.
  - Yet with S(12, C0) = 95 %, the reference keeps less than half of its own short-context competence at 32K.
  - The claim "a small amount of full attention is enough for long context" would then rest on the 1K–8K cells.
- **H5 is spuriously "not supported".** The gap S(12) − S(0) runs 20, 38, 55, 60, 50, 35 pp, which is non-monotone, so the slope in log₂T is ≈ 0.

**Fixes (all adopted in Part 2).**
1. **Primary H1 cell = long band {16K, 32K}.** The short band {1K, 2K, 4K} is secondary. Never pool across bands for H1 or H1-min.
2. **Reference-adequacy gate G-REF.** Pre-registered, per seed and per length. Pass requires both:
   - S(12, T) ≥ 0.8 · S(12, C0) on each headline retrieval task, where C0 is the same item with no filler;
   - S(12, T) − S(0, T) ≥ 15 pp.

   Lengths that fail are excluded from confirmatory H1/H5 and reported descriptively.
3. **Saturation branch.** Branch the seed-0 k = 12 run (and the k = 1 run) from the end-of-stable checkpoint into 2× the Stage-B budget.
   - If the 2× branch raises S(12, 32K) by more than 5 pp, the 1× reference counts as under-trained.
   - H1 is then reported against both references, and the S2 rule (§2.9) should already have picked a larger budget.
4. **Calibration-free metrics alongside R.**
   - EM averaged over a pre-registered difficulty ladder.
   - Retrieval capacity N50: the number of KV pairs retrieved at 50 % EM, from a logistic fit per model and length (M6).
5. **Remove the dilution ceiling** (M3), and use a RoPE base sized for 32K from step 0 (M7).
6. **Diagnostics on the reference at 32K:** needle attention mass, attention entropy and effective maximum logit per layer, all on dev items.

### 1.2 F2 (fatal for RQ-D and the frontier): the efficiency protocol measures implementation artefacts

**What PROP does (§10).**
- Decode: a static-cache decode loop captured with `torch.cuda.CUDAGraph`; batch {1, 8, 32, 128}; contexts 1K–32K; 256 generated tokens.
- Prefill: latency at batch 1, with median and IQR from ≥ 10 reps.
- Measurements on an idle GPU only.
- Claims scoped to "fla Triton + PyTorch SDPA on sm_120".

**Problems, each measured [MEAS-B]:**

(a) **The masked static-cache trap: a 20–40× inflation of attention's decode cost.**
- A preallocated KV buffer needs a length mask.
- With a boolean mask, default SDPA dispatch runs the attention decode step in **1.80 ms** (B = 1, L = 32K, hot cache). Forcing EFFICIENT gives 1.96 ms (cold).
- Exact-shape graphs run in **85–90 µs** cold with cuDNN or flash.
- Flash refuses masks ("No available kernel"). cuDNN accepts the mask at 90 µs cold, but only if it is pinned explicitly.
- Left as is, this would hand the linear layers an order-of-magnitude decode "advantage" that comes from the dispatcher, not the architecture.

(b) **L2 residency: up to 2.5× in the opposite direction.**
- The RTX 5080 has a 64 MiB L2 (`L2_cache_size` = 67,108,864 B; 84 SMs).
- Decode over a 64 MiB KV cache with cuDNN takes 33 µs hot vs **84 µs** after flushing L2.
  - The hot figure is *faster than the 70 µs DRAM roofline*.
  - Triton's own `do_bench` zeroes a 256 MB buffer before every rep for exactly this reason (`triton/backends/nvidia/driver.py` L754–761, `testing.py` L152) [code-inspected].
- The size of the effect depends on k·T: it applies when per-layer KV ≲ 64 MiB, i.e. k = 1 up to 32K, or k = 12 up to ≈ 2.7K. It therefore distorts the *shape* of the frontier, not just its level.

(c) **Launch-bound linear layers at batch 1.**
- At 4K, the fla GDN full layer prefills in 0.69 ms, whereas its chunk core plus projections take 0.22 ms. The other ≈ 0.47 ms is small elementwise and norm kernels plus launch overhead.
- The GDN decode step core takes 4–6 µs and is latency-bound.
- So "prefill latency at ≤ 4K, batch 1" measures kernel count.

(d) **Unequal kernel maturity.**
- **Attention side.** The FA2 core reaches **91.5 TFLOP/s** at 32K, i.e. 81 % of the 112.6 TFLOP/s dense-BF16 figure (spec as read by the methodology agent, 04 §10.1).
- **GDN side.** fla's GDN path on Blackwell is deliberately restricted [code-inspected, fla `98193113`]:
  - `chunk_delta_h.py` L27–37: `num_warps=[2]` "until Triton 3.8 is released";
  - `chunk_o.py` L27–33: 8-warp configs dropped for Triton < 3.8 (triton#10590, fla#1327);
  - `wy_fast.py` L21–22: reduced warp and stage lists;
  - `op.py` L63–85: an inline-asm `tl.dot` wrapper on SM100 and SM120.
- I estimate the GDN chunk core runs at roughly 10 % of peak at 32K [INF from a FLOP count over my 1.02 ms measurement].

**Consequence.** The signs of (a)–(d) differ, so the net bias of the measured frontier is unknown, and it can exceed the effects being measured. The project's stated output, a capability–efficiency trade-off, would not be interpretable as a property of the architecture.

(e) **Idle windows may never happen.** PROP's G-cost says "defer the benchmark", with no fallback.

**Fixes (adopted in §2.7).**
1. **Analytic costs are primary.**
   - State bytes, exactly as stored (fp32 GDN state; conv cache dtype included).
   - FLOPs and bytes moved per token, for prefill and decode.
   - Roofline latency at 112.6 TFLOP/s and 960 GB/s.

   Measured numbers validate these, but never replace them.
2. **Decode = per-token latency at context L, from an exact-shape CUDA graph captured per L.**
   - No masks.
   - The backend is pinned and asserted at runtime: cuDNN primary, flash also reported.
   - A 256-token loop is not needed for the metric.
3. **Equal-maturity comparator: FlexAttention**, which is Triton-generated like fla's kernels.
   - Decode at 32K: 91 µs cold, vs 90 µs (flash) and 85 µs (cuDNN).
   - Prefill: within 5–12 % of FA2/cuDNN at 4K and 32K.
   - So an equal-maturity attention baseline is available at almost no speed cost.
4. **Report cold L2** (256 MB flush before each replay) **and hot.** Whole-model measurements are primary; per-layer microbenchmarks are diagnostics only.
5. **Prefill:** report both a CUDA graph captured per shape and eager execution, at batch 1 and 8.
6. **Estimator and procedure.**
   - Min-of-N over ≥ 30 reps; median and IQR as secondary.
   - Randomised, interleaved blocks; 2 sessions.
   - Discard reps that show any `clocks_event_reasons` throttle flag.
   - Locking clocks is not available: nvidia-smi reports application clocks as "deprecated", and `-lgc` needs root [MEAS-B].
7. **Kernel-efficiency table:** achieved TFLOP/s or GB/s per kernel and shape, i.e. roofline fractions.
8. **Contingency:** if there is no idle window, contended min-of-N numbers go in an appendix only, and H6 is decided on analytic costs.

### 1.3 M1 (major): feasibility of the capability axis is unestablished, and the planned pilots cannot establish it

**Evidence.**
- **Wang et al.** state that "340M models show insignificant recall capability due to their relatively small parameter count" (v1, Table 2 caption), under "a 2,048-token context window" (v1, Limitations) ([2507.06457v1](https://arxiv.org/html/2507.06457v1) [read-section]).
- **Qiao et al.** [read-section] ([2606.15378v1](https://arxiv.org/html/2606.15378v1)):
  - Their S1–S3 models (15M, 31M, 65M non-embedding) were pretrained at 16K context on a 1:1 long/short mix, with D ∈ {100N, …, 1000N} tokens.
  - NIAH and RULER are reported only for S4 (0.22B) and S5 (0.66B).
  - S1 at D = 100N was excluded from the LongPPL fit as "too small for stable long-context behavior".
- **Stuffed Mamba** has from-scratch 36.4M / 47.0M / 84.6M configurations (Table 2). I could not confirm their per-model passkey numbers from the HTML text; they appear only in a figure ([2410.07145](https://arxiv.org/html/2410.07145) [read-section]). The positive small-scale evidence cited in LR §8 is therefore weaker than presented.
- **PROP** trains 41M non-embedding parameters on 1.1B tokens = 27N, with only 9 % at 32K. That is below every long-context budget in Qiao and far shorter in long-context exposure.

**Why PROP's pilots cannot settle it.**
- E1d trains k ∈ {0, 12} for 0.2B tokens plus 20M at 32K, i.e. ≈ 150 long steps.
- Its target band ("all-full 40–85 % at in-distribution lengths") is not tied to the long lengths.
- G2 needs a gap at only one length.
- A pass at 1K–4K says nothing about 32K competence.

**Fix: an S2 feasibility and reference-adequacy pilot before the LR sweep** (§2.9). It has a pre-registered rule choosing the Stage-B budget ∈ {0.1, 0.2, 0.3}B and L_ext ∈ {32K, 16K}, plus a stop/re-scope branch. Cost ≈ 12 GPU-h. It replaces E1c and E1d (≈ 9.5 h), for a net change of about +3 h. It protects about 100 h of downstream runs.

### 1.4 M2 (major): exposure-dependence; the dependent variable becomes sample efficiency

**Evidence.**
- Transformers "need 100x less samples than the best GSSMs to learn the copy task", at ≈ 160M parameters ([Jelassi et al.](https://arxiv.org/html/2402.01032) §3.2 [read-section]).
- Fixed synthetic mixtures failed to give SSMs phonebook retrieval in Birdie (LR §8, citing [2411.01030](https://arxiv.org/abs/2411.01030) [read-section, Stage-1]).

**Problem.**
- Under a small fixed exposure (option (i): ≈ 5M synthetic tokens at 4K–32K), the differences S(k) − S(0) mix capacity with learning speed.
- PROP's response is asymmetric: learning curves, plus the O9 fine-tune sweep *only if H1 fails*.
- But an H1 *success* can also be exposure-driven. Example: the reference learns formats quickly but generalises them poorly to long lengths, while hybrids catch up.

**Fix.**
- Exposure becomes a measured factor.
  - Every model's task-exposure fine-tune (TEF) is checkpointed at 250, 500 and 1,000 steps.
  - The core arms at seed 0 get a 4× continuation (to 4,000 steps).
- Capability claims are made at the exposure plateau, and the gap-vs-exposure curve is reported.
- Cost ≈ 3.5 GPU-h.

### 1.5 M3 (major): QK-RMSNorm produces a length-dependent softmax-dilution ceiling

**Mechanism.**
- PROP §5 specifies per-head QK-RMSNorm with 64-dimensional gains *shared across heads* and logit scale 1/√64.
- With unit gains, ‖q‖ = ‖k‖ = 8, so |logit| ≤ 8.
- For random 64-dimensional directions, distractor logits are ≈ N(0, 1).

**Simulation [MEAS-B, INF]: weight on a perfectly matched needle.**

| Gain product g_q·g_k | 1K | 4K | 16K | 32K |
|---|---|---|---|---|
| 1.00 (initialisation) | 0.64 | 0.31 | 0.10 | 0.05 |
| 1.56 | 0.99 | 0.95 | 0.83 | 0.71 |
| 2.25 | 1.00 | 1.00 | 1.00 | 0.99 |
| **Needed for 90 % weight** | **1.24** | **1.45** | **1.66** | **1.78** |

**Literature.**
- Softmax's maximum output approaches zero as input size grows. SSMax (Eq. 2) multiplies logits by s·log n ([Nakanishi, 2501.19399](https://arxiv.org/html/2501.19399) [read-section]).
- Scaling logits by log n has no effect when train and test lengths match, but helps when training is short and testing is long (+1 BLEU, p < 0.01; [Chiang & Cholak §5.3–5.4, Table 1](https://arxiv.org/abs/2202.12172) [read-section]).

**Why it matters.**
- The model can compensate: the gains are learnable and excluded from weight decay, and keys can become anti-aligned. But this has to happen within ≈ 763 Stage-B steps.
- Two consequences:
  1. The k = 12 reference may be capped at 32K (this feeds F1).
  2. Hybrids, whose few attention layers can specialise on retrieval while the GDN layers do local work, may beat all-full at long T **for a parameterisation reason**. This explanation is missing from PROP's O2 list.

**Fix (adopted in Part 2 through the S2 rule).**
- Use **per-head (unshared) QK-norm gains**.
- Add **SSMax-style query scaling** q_t ← q_t · s_h · ln(t + 1):
  - t is the within-document position, consistent with RoPE positions being reset per document;
  - s_h is learnable per head, initialised to 1/ln 4096, so the multiplier is 1.00 at 4K and 1.25 at 32K;
  - it is compatible with flash/cuDNN, because only q is pre-scaled.
- Apply it identically in every attention layer of every arm.
- Adopt it only if, at k = 12 in S2, it is ≥ plain QK-norm at 16K and 32K and no more than 2 pp worse at 4K.
- Log the per-layer effective maximum logit (s_h·ln(t+1)·Σ g_q g_k), the attention entropy and the needle mass on dev items.
- Run the non-chosen variant at k ∈ {1, 12} as a robustness arm.

### 1.6 M4 (major): the determinism and reproducibility plan misplaces the source of nondeterminism

**Measured [MEAS-B].**

| Path | Default mode | `torch.use_deterministic_algorithms(True)` |
|---|---|---|
| `varlen_attn` (FA2 inside PyTorch), backward | not bitwise; max \|Δgrad\| 4.9e-4 | bitwise |
| SDPA FLASH, backward | not bitwise; 9.8e-4 | bitwise |
| SDPA EFFICIENT, backward | not bitwise; 4.9e-4 | bitwise |
| SDPA cuDNN, backward | not bitwise; 1.2e-4 | **RuntimeError: "No available kernel"** |
| fla GatedDeltaNet layer (config A; conv, norm-gate, chunk; varlen), forward and all gradients | **bitwise** | bitwise |
| **Whole proposed model, all parameter gradients, k = 0** | **bitwise** | bitwise |
| **Whole proposed model, all parameter gradients, k = 12** | **not bitwise; 3.1e-4** | **bitwise; no op raises** |

**Code inspection (fla `98193113`) [code-inspected].**
- There is no `tl.atomic_add` in `ops/gated_delta_rule`, `ops/common`, `modules/causal_conv1d` or `modules/norm`. Atomics appear only in `path_attn`, `attn/backends/tilelang`, `atk`, `utils/csr`, `log_linear_attn` and `cyfa`.
- Conv and norm weight gradients are reduced through per-block partial sums followed by `.sum(0)` (`causal_conv1d/ops.py` L758–799; `fused_norm_gate.py` L567–624).
- **Triton autotune caching across processes.**
  - `FLA_CACHE_RESULTS` defaults to `'1'` and maps to Triton's `cache_results` (`fla/utils/env.py` L26, L112–113).
  - Triton 3.6's disk-cache key is: Triton hash, backend target, kernel source hash, cache-invalidating environment variables, tuning key and config list (`autotuner.py` L170–210).
  - Configs that carry per-config `pre_hook`s bypass the disk cache. I found none on the GDN path.

**Consequences.**
- PROP §11 (and 04 §11) attributes production nondeterminism to autotuning. Within a process, the actual source is the attention backward.
- In production mode, **k = 0 runs are exactly reproducible but k ≥ 1 runs are not.** That is an asymmetric noise source between arms (small, but avoidable).
- The audit mode must not route through cuDNN.

**Cost of deterministic mode.**
- On the FA2 forward+backward kernel [MEAS-B]:
  - +43 % for 8 × 4K sequences;
  - +62 % for a document-packed 32K sequence;
  - +66 % for a single 32K sequence.
- Per run in my design, scaled by attention's share of step time [INF]:
  - +0 % (k = 0), +2 % (k = 1), +6 % (k = 3), +18 % (k = 12);
  - **≈ +7 % across the core.**

**G0(f) thresholds are wrong.** I compared the chunk kernel with an fp64 naive recurrence at T = 32,768 [MEAS-B]:
- "fp32" inputs: relative RMS error 1.5–1.6e-3, because the GDN path uses TF32 internally. Max absolute error reaches **7e-4** in the near-unit-decay regime, which **violates PROP's "≤ 5e-4 (fp32)"**.
- bf16 inputs: maximum absolute error up to **3.1e-3**, which **violates "≤ 2e-3 (bf16)"**.
- The error is flat across position bands 0–1K … 16–32K, so there is **no long-sequence drift**. This de-risks 32K numerics.

**Fixes.**
1. Run all production runs in deterministic mode. Never use cuDNN SDPA in training.
2. G0(a): bitwise-identical loss traces and parameter SHA-256 across *processes*, for k ∈ {0, 3, 12}.
3. G0(c): bitwise kill-and-resume.
4. G0(f), relative thresholds at T ∈ {512, 32K}, in two decay regimes:
   - relative RMS ≤ 3e-3 for fp32/TF32 inputs;
   - relative RMS ≤ 1e-2 for bf16;
   - plus a gradient check against fp64 autograd through the naive recurrence at T = 512.
5. Warm and archive the Triton cache (SHA-256 in the registry) for **all** training, evaluation and benchmark shapes. The norm kernels are autotune-keyed on `NB`, i.e. per length (`fused_norm_gate.py` L32/L200; `l2norm.py` L65/L89) [code-inspected].

### 1.7 M5 (major): the seed and statistics design cannot support most confirmatory claims under PROP's own rule

**PROP's rule (§2, §11):** "supported" requires that **both** the hierarchical bootstrap **and** the seed-level paired t (df = S − 1) exclude the null. The seeds are 2 for k ∈ {2, 4, 6} and for every placement arm.

**Arithmetic [INF].** t₀.₉₇₅ = 12.71 (df = 1), 4.30 (df = 2), 2.78 (df = 4). At PROP's planning σ_d = 3 pp, the seed-level half-widths are:

| Seeds S | Half-width |
|---|---|
| 2 | **27.0 pp** |
| 3 | 7.4 pp |
| 5 | 3.7 pp |

**Consequences.**
- **H4** has 10 Holm-adjusted pairwise contrasts on 2-seed arms. It can be "supported" only if a contrast exceeds about 30 pp, yet from-scratch placement effects measured so far are tiny (≤ 0.004 nats on loss; LR §5, citing [2510.04800](https://arxiv.org/abs/2510.04800) [read-section, Stage-1]).
- **H2** uses the 2-seed k = 6 arm, and **H3** interpolates k80 through 2-seed points. Both are capped at "suggestive" in the same way.
- **About 32 GPU-h (E4) plus part of E3 buy outcomes that are inconclusive by construction.**

**Fix.**
- Only arms with ≥ 3 seeds enter confirmatory tests. The confirmatory family has 5 tests under Holm:
  - H1 at k = 3 (long band);
  - H1-min at k = 1 (long band);
  - H3 as a family × k interaction on {0, 1, 3, 12};
  - H5 on {0, 1, 3, 12};
  - H4 on a k = 1 placement triad {1, 6, 11} × 3 seeds.
- H2 (curve shape) and H6 (tautological in memory, since M > 0 ⟺ R > k/12, as PROP itself notes) are descriptive.
- Gold log-prob recovery is a continuous co-primary, which gives higher signal-to-noise.
- Keep PROP's variance-triggered reserve seeds. They are effect-blind, so they spend no α.

### 1.8 M6 (major): calibrating a single difficulty level on pilots does not transfer

**Problem.**
- PROP tunes difficulty on dev items so that the **0.2B-token** all-full pilot scores 40–85 %, then freezes that level for **1.1B-token** models (§9.3).
- The shift in competence from pilot to main runs is unknown and may depend on architecture.
- Ceiling risk: all k ≥ 1 near 100 % makes R ≈ 1 for every k, so H1 and H2 are trivially "supported".
- Floor risk: G2 demotes the family after the main runs have already been paid for.

**Fix.**
- The frozen suite contains **difficulty ladders**:
  - R2: N ∈ {8, 32, 128, 512} KV pairs; 512 exceeds GDN's ≈ 256 orthogonal-key capacity per layer (02 C6 [INF]);
  - R1: D ∈ {3, 15} distractors;
  - M1: hops ∈ {1, 2, 3};
  - A2: frequency ratio ρ ∈ {2, 1.5}.
- The primary score is **EM averaged over ladder levels**: equal weight per log-spaced level, which approximates the area under the difficulty curve.
- **N50** is the capacity metric.
- Nothing is tuned on pilots except the ladder *range*, which must bracket k = 12's 50 % point at 16K in S2.

### 1.9 M7 (major): choosing position encoding by a single-seed pilot, and keeping θ constant through 4K → 32K

**Problem.**
- E1c makes the choice from 1 seed × (0.2B + 20M at 32K) per scheme, scored on about 100 dev contexts per cell.
- PROP itself cites large seed variance for length generalisation (§16.6; [2402.09371](https://arxiv.org/abs/2402.09371) [abstract-only]).
- RoPE θ = 1e4 kept constant through an 8× context extension is a fragile recipe [INF]. A NoPE all-full model is a non-standard baseline (02 §2).
- The rule is "baseline-favouring" only in expectation.

**Fix.**
- **Default to RoPE θ = 5e5 in every arm from step 0.** It is one of PROP's own candidates, and it is sized so that no mid-run change of base is needed.
- Run NoPE at k ∈ {1, 12} as a robustness arm.
- Drop E1c (saves 2.2 h), or run it with 2 seeds and override the default only on a ≥ 5 pp margin at 32K.

### 1.10 M8 (major): operations on a shared GPU are under-planned

- **OOM interplay.** Without checkpointing, a 16K-token micro-batch OOMs under a 7.1 GiB cap for every k [MEAS-B]. CARLA's footprint moved between 5.3 and 7.3 GB during my session [MEAS-B]. Our allocations can starve CARLA (the user's workload), and the reverse can happen too.
- **Timing control is limited.**
  - Clocks cannot be locked [MEAS-B].
  - A non-zero "SW Power Capping" event counter was already present [MEAS-B].
  - L2 is shared, so contention distorts decode timing beyond plain time-slicing.
- **No fallback without idle windows** (F2e).
- **Checkpoints every 500 steps or 30 minutes** risk losing up to 30 minutes per crash.

**Fix: the runbook in §2.10.**

### 1.11 Minor issues

- **m1. Micro-batching and the speed-up assumption.**
  - Use 8K-token micro-batches (2 × 4K) without checkpointing in Stage A: 4.5–4.6 GiB peak, ≈ 88–90K tok/s for every k [MEAS-B]. Checkpointed 32K–64K micro-batches are slower (60–77K tok/s).
  - The "×1.6 idle" multiplier is unsupported: CARLA takes ≈ 25–37 % utilisation. Plan with the contended numbers.
- **m2. Scoring path.**
  - fla switches to `fused_recurrent` automatically whenever `q_len ≤ 64` without gradients (`gated_deltanet.py` L227–232) [code-inspected]. So C0 items and query continuations run a different kernel from long prefills.
  - Branched scoring (chunk prefill plus recurrent tail) differs from a single chunk pass by up to 1–2e-3 at bf16 [MEAS-B].
  - Fix: call the fla ops explicitly, re-score a random 5 % of items with a single-pass forward, and require an EM flip rate below 1 %.
- **m3. Triton warm-up coverage.**
  - The first call costs 46 s [MEAS-B].
  - The warm-up must cover the `IS_VARLEN` true and false branches and every evaluation length, because the norm kernels autotune per length.
- **m4. Evaluation cost is about 2× overestimated.**
  - PROP assumed 40 TFLOP/s; the attention core runs at 91.5 TFLOP/s at 32K.
  - My estimate is ≈ 25–30 min per k = 12 model and ≈ 10 min per k = 0 model for a ≈ 300M-token suite [INF from MEAS-B]. The spare compute can go into ladders.
- **m5. No contingency for gate failures.**
  - A G2 retune under option (i) re-runs E1d (≈ 7 h per iteration), and G-div restarts also cost time.
  - Add 20 % contingency plus explicit stop rules (§2.9).
- **m6. H6 for memory is tautological.** M > 0 ⟺ R(k) > k/12 (PROP's own note). Make it descriptive. H2 is nearly implied by H1 together with saturation, so make it exploratory.
- **m7. Prefill at batch 1 and short T is launch-bound.** Capture a graph per shape, as in F2c.
- **m8. LR horizon.** Selecting LR at 0.2B tokens favours a slightly higher LR than optimal for 1B tokens [INF]. The regret rule mitigates this. Tier C adds a rerun at η*/√2 for k ∈ {3, 12} at seed 0.
- **m9. Logging.** Add logging of attention maximum logit and entropy (ties to M3), gradient-norm spikes, and `clocks_event_reasons`, alongside G-div.

---

## 2. Independent experimental design ("Design B")

### 2.1 Principles

1. **Feasibility first.** Spend ≈ 12 GPU-h proving that the setting can express the effect: a competent all-full reference at 32K and a failing all-linear model. Only then spend ≈ 100 h measuring it.
2. **Branch from shared checkpoints.** One WSD Stage-A run per (k, seed). Its end-of-stable checkpoint S is the starting point for Stage-B and exposure variants. The two main confounds therefore become cheap factors.
3. **Calibration-free outcomes.** Difficulty ladders and capacity metrics instead of single tuned difficulty levels.
4. **The long band is primary,** behind an absolute reference-adequacy gate.
5. **Seeds only where tests live,** with ≥ 3 seeds for every confirmatory arm.
6. **Costs are analytic first,** and measured under an artefact-controlled protocol.
7. **Bitwise-reproducible training.** Infrastructure noise is removed rather than modelled.

### 2.2 Architecture

| Component | Choice | Notes |
|---|---|---|
| Backbone | L = 12, d = 512, pre-norm RMSNorm, SwiGLU d_ff = 1536 in every block, V = 16,384 tied, no biases | As PROP. Totals 49,395,808 (k = 0) / 49,297,408 (k = 12) [MEAS-B] |
| Full mixer | MHA 8 × 64; **per-head** QK-RMSNorm gains; RoPE θ = 5e5 with positions reset per document; SSMax query scale s_h·ln(t + 1), s_h learnable, init 1/ln 4096 (only if S2 adopts it) | Training: FA2 `varlen_attn` in deterministic mode. Evaluation: SDPA flash/cuDNN |
| Linear mixer | GDN config A: 4 heads, d_k = 64, d_v = 128, output gate, short conv 4; fla 0.6.0 Triton chunk kernel; fp32 state | 1,056,904 parameters [MEAS-B]. Decode kernel called with keywords only |
| Initialisation | As PROP: per-module seeding; shared modules identical across k for a given seed | Common random numbers |
| Precision | bf16 autocast, fp32 master weights and Adam state; `use_deterministic_algorithms(True)`; `CUBLAS_WORKSPACE_CONFIG=:4096:8`; `FLA_FLASH_QLA=0` | Deterministic training [MEAS-B] |

### 2.3 Ratios and placement

- **Composition grid** (centred-uniform placement as in PROP):
  - Core: k ∈ {0, 1, 3, 12}.
  - Curve: k = 2, giving positions {6}, {3, 9}, {2, 6, 10}.
  - Tier C: k = 6, strict alternation.
  - k = 4 is dropped: it sits in the region expected to be flat.
- **Placement study:** k = 1 at layers {1, 6, 11}, i.e. early but not first, middle, and last.
  - The middle arm is shared with the composition grid.
  - This answers the most practical "how little" question (where to put the *single* global layer), where a placement effect is most likely to be large.
  - k = 2 spread vs adjacent ({3, 9} vs {5, 6}), which tests multi-hop depth, is Tier C.
- **Not in Tiers A and B:** F24 (count vs fraction), GLA, SWA-64. They are listed in Tier C.

### 2.4 Data, tokenizer, curriculum, schedule

**Shared settings.**
- Tokenizer, corpora and held-out splits as in PROP: 16,384-entry BPE with digit splitting; FineWeb-Edu sample-10BT and PG19, both at pinned revisions.
- The token stream is identical across k for a given seed (stateless loader of (seed, step)).
- AdamW (0.9, 0.95, 1e-8), weight decay 0.1 on matrices only, gradient clipping 1.0, z-loss 1e-4.

| Phase | Tokens | Sequence | Data | LR | Batch | Notes |
|---|---|---|---|---|---|---|
| **A (stable)** | 0.8B (fallback 0.6B) | 4,096, document-masked | FineWeb-Edu | 500-step warm-up, then constant η* | 131,072 tokens = 16 × 8K micro-batches | ≈ 6,100 steps. Checkpoint **S** at 0.8B is the branch point |
| **B (decay)** | **0.2B** default; the S2 rule picks 0.1, 0.2 or 0.3B | 32,768 | 60 % PG19-train 32K windows (option: 40 % PG19 + 20 % Common Pile arXiv documents ≥ 16K), 40 % FineWeb-Edu document-masked | Linear decay η* → 0 | 4 × 32K; activation checkpointing for **every** k | Peak 2.6 GiB [MEAS-B]; ≈ 1,526 steps |
| **TEF** | 65.5M (1,000 × 64K) | Packed 32K; synthetic items of 1K–32K as separate documents | 50 % synthetic (all families; difficulty uniform over the ladder; training namespaces), 50 % Stage-B mix | 50-step re-warm to 3e-4, then constant | 64K | Checkpoints at 250 / 500 / 1,000. Core arms at seed 0 continue to 4,000 |

- Tokens per parameter: 1.0B / 41.0M ≈ 24N, plus 1.6N of TEF.
- Tier-0 zero-shot metrics are measured on the Stage-B checkpoint, before TEF. They are a clean architecture-only signal, unaffected by task exposure.

### 2.5 Task exposure

**Choice: an identical post-hoc TEF, i.e. PROP's option (ii) family.** Reasons:
- It keeps the base models clean, so Tier-0 metrics are exposure-free.
- It makes the exposure budget a branchable factor (M2).
- It has precedent: Samba and ProLong (LR §8).

**Leakage controls, as in PROP §9.1:**
- disjoint word lists, value ranges and templates (T1–T3 for training; T1 plus an unseen T4 for evaluation);
- filler from held-out documents;
- a unit test asserting zero overlap of hashed (key, value, template) tuples.

**Forgetting check:** report LM-2 (validation loss at 4K) before and after TEF.

### 2.6 Evaluation

| Family | Task | Frozen ladder | Lengths | Contexts × queries |
|---|---|---|---|---|
| **Retrieval** (confirmatory) | R1: single key in natural filler, same-format distractors | D ∈ {3, 15} | 1K, 2K, 4K, 8K, 16K, 32K | 300 × 4 |
| | R2: multi-key pairs spread through natural filler | N ∈ {8, 32, 128, 512}; a level is used at length T if 12N ≤ 0.75T | same | 300 × 4 |
| | R3: key updates; latest value; stale-value rate | U ∈ {2, 4} | 4K, 16K, 32K | 200 × 1 |
| **Multi-hop** | M1: key chains, shuffled order, 2 distractor chains | hops ∈ {1, 2, 3}; link order {forward, reversed} at 16K only | 1K, 4K, 16K, 32K | 300 × 1 |
| **Aggregation** | A2: frequency comparison (chance 0.5) | ρ ∈ {2, 1.5} | same | 300 × 4 |
| | A1: counting, c ≤ 9 | — | same | 300 × 4 |
| **Mechanism (H5)** | Distance-controlled R1 (D = 3) | needle–query distance d ∈ {256, 2K, 16K} | T ∈ {16K, 32K} | 200 × 1 |
| **Controls** | C0 (no filler); C2 (relevant facts placed just before the query) | per headline task | C0 short; C2 at 4K and 32K | 200 |
| **Tier 0** (pre-TEF, zero-shot) | LM-1 per-position loss; LM-3 repeat-token loss by distance; LM-4 copy-after-gap; cloze passkey | — | ≤ 32K | 400 documents / 200 items |

**Metrics.**
- EM_tf, averaged over ladder levels → S_f(k, T). This is primary.
- Gold log-probability is a continuous co-primary.
- Also reported: rank accuracy, binding / absent / stale error rates, and **N50** (from a logistic fit of EM on log₂N per model and T; reported as "< 8" when censored).

**Canonical scoring path.**
- Prefill: the chunk kernel for GDN and SDPA-flash for attention.
- Queries: scored from a state snapshot, with `fused_recurrent` called *explicitly*, and SDPA over the cached KV.
- Re-score 5 % of items with a single-pass forward; the EM flip rate must stay below 1 %.

**Item files.**
- Item files hold token IDs plus metadata; the generator is hashed and the files frozen.
- Every sample is exactly T tokens and is built in token space (PROP §9.1).

**Cost.** ≈ 300M tokens per model; ≈ 25–30 min for k = 12 and ≈ 10 min for k = 0 [INF from MEAS-B prefill rates].

### 2.7 Efficiency (RQ-D)

| Axis | Status | Protocol |
|---|---|---|
| Inference-state bytes vs T | **Primary (analytic)**, validated | Sum of cache-tensor `nbytes` after prefill (fp32 GDN state + conv cache + bf16 KV); last-token logits only |
| FLOPs/token and bytes/token (prefill, decode) | **Primary (analytic)** | Formulas plus `torch.utils.flop_counter` on the dense part; roofline latency at 112.6 TFLOP/s and 960 GB/s |
| Prefill latency | Secondary (measured) | Whole model; CUDA graph per exact (B, T) plus eager; B ∈ {1, 8}; T ∈ {1K, 4K, 16K, 32K}; attention backend pinned (flash) and asserted; **FlexAttention** comparator |
| Decode latency per token at context L | Secondary (measured) | One-step **exact-shape** graph per L ∈ {1K, 4K, 16K, 32K}, B ∈ {1, 8, B_max}; cuDNN primary, flash and FlexAttention reported; **no masked static cache** unless the backend is pinned to cuDNN/Flex and asserted |
| L2 handling | Required | Cold (256 MB buffer zeroed before each replay) and hot, both reported |
| Capacity | Descriptive | Largest batch within an 8 GiB state-plus-activation budget, analytic and verified by allocation |
| Kernel efficiency | Required table | Achieved TFLOP/s or GB/s per kernel (attention core, GDN chunk core, projections, decode steps); roofline fractions |
| Statistics and conditions | Required | Min-of-N over ≥ 30 reps (median and IQR secondary); randomised interleaved blocks; 2 sessions on different days; reps with throttle flags discarded; clocks, temperature and power logged; G-cost: no foreign compute process for 60 s and utilisation below 5 % |

**H6 frontier.**
- Analytic memory makes it tautological, so it is descriptive.
- For measured latency, report the hull margin with CIs over timing reps, and state alongside it how far the result depends on the roofline-fraction gap.

### 2.8 Seeds and statistics

**Seeds.**
- Core {0, 1, 3, 12}: seeds 0–2. Reserve seeds 3–4 are triggered only if, after the 3 seeds, σ_d for long-band retrieval exceeds 5 pp (variance-triggered and effect-blind).
- Curve k = 2: seeds 0–1.
- Placement k = 1 at layers {1, 11}: seeds 0–2.

**Inference.**
- Hierarchical paired bootstrap, resampling seed, then filler cluster, then context; B = 10,000; percentile CIs.
- Seed-level paired t.
- "Supported" requires both, as in PROP, now applied only to arms with ≥ 3 seeds.
- GLMM for interactions: `correct ~ k × band × family + ladder_level + (1|context) + (1|seed) + (1|seed:k)`.

**Multiplicity.**
- Confirmatory family: 5 tests, Holm at α = 0.05.
- Exploratory cells: BH-FDR at q = 0.10.
- "No difference" claims require TOST, with margins of 5 pp EM, recovery 0.8, 0.25 log₂ units of N50, and 10 % latency.

| H | Statistic | Supported if | Falsified if |
|---|---|---|---|
| **H1** | R_ret,long(3) on ladder-mean EM (long band, G-REF-passing lengths); Δ = S(12) − S(3) | lower CI(R) ≥ 0.8 **and** upper CI(Δ) ≤ 5 pp **and** exposure plateau (or the TEF-4× branch agrees) | upper CI(R) < 0.8 with G-REF passing **and** the seed-0 TEF-4× branch keeps R(3) < 0.8 (see §3) |
| **H1-min** | As H1, at k = 1 | as H1 | as H1 |
| **H3** | family × k interaction (GLMM, long band); per-family R_f(1) and R_f(3) | Holm-significant interaction **and** some \|R_f − R_ret\| ≥ 0.2 with CI excluding 0 | TOST: all \|R_f − R_ret\| within ±0.1 |
| **H4** | S_ret,long at k = 1, layers {1, 6, 11} | a Holm-adjusted pairwise CI excludes 0 **and** the contrast is ≥ 5 pp | TOST: every pair within ±5 pp |
| **H5** | Slope of Δ_k(T) = S(12, T) − S(k, T) in log₂T, k ∈ {0, 1}, over G-REF-passing lengths; plus the distance-controlled decomposition | slope CI > 0 under both methods | slope CIs within ±2 pp per doubling |
| H2 | Descriptive: isotonic and logistic fits in k | — | — |
| H6 | Descriptive: analytic hull margin; measured hull margin under the §2.7 protocol | — | — |

### 2.9 Stages, gates and budget

Budget is in contended GPU-h, from measured throughput:
- Stage A: 88.9K tok/s.
- Stage B at 32K: k = 0 → 70.5K, k = 3 → 51K, k = 12 → 31.9K tok/s (measured); k = 1 → ≈ 64K, k = 2 → ≈ 59K, k = 6 → ≈ 44K (interpolated as linear in k).
- Per run, including TEF and deterministic mode: k = 0 → 3.5 h, k = 1 → 3.7 h, k = 2 → 3.8 h, k = 3 → 4.1 h, k = 6 → 4.5 h, k = 12 → 5.5 h.

| Stage | Content | GPU-h | Gate: pass criteria → if it fails |
|---|---|---|---|
| **S0 Infrastructure** | Deterministic-mode G0(a)/(c); relative-threshold kernel and gradient tests at T ∈ {512, 32K}; parameter counts exact; benchmark-harness self-test; Triton cache warmed for all shapes and archived | 3 | Harness self-test: backend assertion, no masks, L2 flush, reproduces Appendix A within 15 %. Throughput floors: Stage A ≥ 70K tok/s and k = 12 Stage B ≥ 20K tok/s (contended) → otherwise fall back to Stage A 0.6B |
| **S1 Reproduction** | Zoology MQAR (PROP §13 RP-1, unchanged) | 3 | G-RP as PROP → debug |
| **S2 Feasibility and reference adequacy** | k ∈ {0, 12}, plus a k = 12 SSMax variant; seed 101; Stage A 0.3B; Stage-B branches at 32K of 0.1B and 0.3B (SSMax variant: 0.1B only); TEF checkpoints at 250/500/1000; dev ladders at 4K/16K/32K plus C0 | ≈ 12 | See the S2 rules below |
| **S3 LR** | k ∈ {0, 3, 12} × η ∈ {1.0, 1.4, 2.0, 2.8}e-3 × 0.15B at 4K (compressed WSD) | 5.6 | Regret ≤ 0.02 nats → shared η*. Otherwise per-config η (PROP rule). Add an edge point if the optimum sits at a grid edge |
| **G5 freeze** | Protocol, configs, analysis code with synthetic-data tests, item files, Triton cache: all hashed | — | No launch until complete |
| **S4 Core** | k ∈ {0, 1, 3, 12} × seeds 0–2 | 50.4 + 4 (evaluation) | G-REF per seed and length. G3 variance rule → reserve seeds (+≈ 34 h) |
| **S5 Branches** | Seed 0: TEF continued to 4,000 steps for k ∈ {0, 1, 3, 12}; 2× Stage B (0.4B) for k ∈ {1, 12} | 10.7 + 2 | Verdicts on learnability and on reference saturation |
| **S8 Costs** | §2.7 protocol, run in idle windows | 4 | G-cost → contingency (analytic-primary) |
| **Tier A** | S0–S5 + S8 + evaluation | **≈ 96 → ≈ 115 with 20 % contingency** | **Decision point:** H1, H1-min, H5 and H3 settled or declared inconclusive |
| **S6 Curve** | k = 2 × seeds 0–1 | 7.7 + 1 | — |
| **S7 Placement** | k = 1 at layers {1, 11} × seeds 0–2 | 22.2 + 1.5 | — |
| **Tier B** | Tier A + S6 + S7 | **≈ 128 → ≈ 154** | H4 |
| **S9 Robustness** (in priority order) | (a) NoPE at k ∈ {1, 12}, 9 h; (b) the non-chosen attention-temperature variant at k ∈ {1, 12}, 9 h; (c) Attn+conv at k = 12 if any hybrid beats k = 12, 5.5 h; (d) GDN config D (2× state) at k ∈ {0, 1}, 7 h; (e) k = 6 × 2 seeds, 9 h; (f) GLA at k ∈ {0, 1, 3}, 11 h; (g) 3B tokens at k ∈ {3, 12}, ≈ 22 h; (h) η*/√2 rerun at k ∈ {3, 12}, seed 0, 9 h | per item | Tier C, optional |

**S2 rules (all pre-registered).**
- **FEAS-1, reference adequacy.** At 32K: S(12) ≥ 0.8 · S(12, C0), using the R1/R2 ladder mean.
- **FEAS-2, discriminability.** At 32K: S(12) − S(0) ≥ 20 pp.
- **FEAS-3, Stage-B budget.** Let Δ = S12(0.3B) − S12(0.1B) at 32K. Then:
  - Δ ≤ 3 pp → 0.1B;
  - 3 < Δ ≤ 10 pp → 0.2B;
  - Δ > 10 pp → 0.3B.
- **FEAS-4, TEF length.** Use the first checkpoint whose 32K score is within 3 pp of the 1,000-step score.
- **FEAS-5, SSMax adoption.** Adopt the SSMax variant per the M3 rule.
- **Fallback ladder.**
  1. If FEAS-1 fails at 32K, set L_ext = 16K, judged on 16K scores from the same branches. The 32K cells become extrapolation.
  2. If FEAS-1 also fails at 16K, stop and re-scope with the user: a ≤ 8K question, or a larger model.

**Comparison with PROP.**
- PROP's conservative Tier A is 71 h, with no contingency and without k = 1.
- My Tier A without contingency and without k = 1 is ≈ 85 h.
- The difference buys:
  - S2 (net +3 h over E1c + E1d);
  - a 2× Stage B (+6 h);
  - TEF (+3.6 h);
  - S5 branches (+10.7 h);
  - deterministic mode (+3.5 h).

  All of these are needed to make an H1 verdict interpretable in either direction.
- If CARLA can be paused, every figure shrinks by the measured idle speed-up, which I estimate at 1.2–1.4× [INF] but which must be measured in G0.

**Stop rules.**
1. S2 fallback ladder exhausted → stop.
2. After S4, G-REF fails at 32K for ≥ 2 of 3 seeds → 32K is reported as non-evaluable for H1/H5, and S6/S7 are not started until the user decides.
3. Contended throughput falls more than 30 % below the G0 value for over 1 h → re-plan the queue.
4. Contingency is spent only on re-runs after gate failures.

### 2.10 Engineering runbook for the shared GPU

- **Memory.**
  - Cap with `torch.cuda.set_per_process_memory_fraction(0.5)` (≈ 7.7 GiB).
  - Stage A: 8K micro-batches (peak 4.6 GiB). Stage B and TEF: 1 × 32K with checkpointing (2.6 GiB) [MEAS-B].
  - Ask the user what memory CARLA needs at its peak.
- **Checkpoints.**
  - Every 250 steps (≈ 6 min), written atomically: `tmp` → `fsync` → `os.replace`. Keep the last 2.
  - Weights-only milestones at {50M, 100M, 200M, 400M} tokens, **S**, end of B, and TEF 250 / 500 / 1,000.
  - The data cursor is the step number (stateless loader).
- **Runner.**
  - A single-GPU queue (`systemd-run --user` once linger is enabled, otherwise `nohup`).
  - Auto-resume after OOM or crash. An OOM is a logged event, not a failure. Mark a run failed after 3 consecutive failures.
- **Determinism.**
  - Deterministic algorithms on, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `PYTHONHASHSEED=0`.
  - Pinned `TRITON_CACHE_DIR` with its SHA-256 in the registry; `FLA_CACHE_RESULTS=1`; `FLA_FLASH_QLA=0`.
  - No cuDNN SDPA in training.
  - **Replicate one core run bitwise** as a standing check.
- **Monitoring.**
  - `nvidia-smi` every 30 s: utilisation, memory per process, clocks, `clocks_event_reasons`, power, temperature. This feeds `contended_frac`.
  - Guards on tok/s and disk space.
  - Log attention maximum logit and entropy, gradient norm and z-loss.
- **Idle windows.** Ask the user for 2 × 2 h. Use them for G0 throughput, autotune warm-up for the benchmark shapes, and S8.

---

## 3. The minimum experiment that would falsify H1

**Target.** H1 says "a small proportion of full attention recovers most long-context performance". To refute it, show that **even the most generous small proportion**, k = 3 (r = ¼, the industrial 3:1), fails in the **long band**. The failure must also not be explained away by a weak reference, a learnability gap or an under-trained long stage.

**Design.**
1. **Prerequisites, ≈ 14 h.**
   - S0 (3 h) and S1 (3 h).
   - S2-lite (≈ 5.5 h): k ∈ {0, 12}, Stage A 0.3B, a single 0.2B Stage-B branch, TEF checkpoints. FEAS-1 and FEAS-2 must pass.
   - S3-lite (≈ 2.8 h): k ∈ {0, 12} × η ∈ {1.4, 2.0, 2.8}e-3 × 0.15B.
2. **Core trio, ≈ 39 h.** k ∈ {0, 3, 12} × seeds 0–2 on the full Design-B recipe: Stage A 0.8B, Stage B 0.2B at 32K, TEF, deterministic mode.
3. **Discriminating branches, seed 0, ≈ 8 h.**
   - TEF continued to 4,000 steps for k ∈ {0, 3, 12} (2.7 h).
   - 2× Stage B for k = 12 (5.1 h).
4. **Evaluation, ≈ 1.5 h.** Retrieval family only (R1 ladder D ∈ {3, 15}; R2 ladder N ∈ {8, 32, 128, 512}), at 4K, 16K and 32K, with 300 contexts × 4 queries, plus C0.

**Total.** ≈ 63 GPU-h contended, or ≈ 75 h with 20 % contingency. That drops to ≈ 54 h without deterministic mode, and without the 2× branch if S2 already showed Stage-B saturation. PROP's equivalent is ≈ 66 h, but without the adequacy and learnability controls.

**Decision rule.** H1 is **falsified** (for ≈ 50M parameters, GDN, this recipe) only if **all** of the following hold:
1. G0, G-RP, FEAS-1 and FEAS-2 pass.
2. G-REF passes at 16K and 32K for ≥ 2 of the 3 seeds.
3. The upper 95 % CI of R_ret,long(3) is below 0.80 under **both** the hierarchical bootstrap and the seed-level paired t (df = 2), on ladder-mean EM. Capacity recovery on log₂N50 must agree in direction.
4. **Learnability is excluded.** In the seed-0 TEF-4× branch, R_ret,long(3) stays below 0.80, and k = 3's learning curve has plateaued (less than 3 pp change between 2,000 and 4,000 steps).
5. **The reference is saturated.** The 2× Stage-B branch changes S(12, 32K) by ≤ 5 pp. If it rises by more, the verdict is stated as "conditional on the 1× long-context budget", and the k = 3 2× branch is added (+2.5 h) to recompute R against the 2× reference.

**Other outcomes.**
- **Supported:** lower CI(R) ≥ 0.8, non-inferiority within 5 pp, and conditions 1–2 hold.
- **Inconclusive:** anything else. G3 reserve seeds apply if σ_d exceeds 5 pp.

**Why this is enough.** Under monotonicity in k, failure at k = 3 implies failure at k ∈ {1, 2}. Monotonicity is assumed here and checked later by k ∈ {1, 2} in Tier B. Restricting to the long band ensures the verdict is about *long* context, and conditions 4–5 remove the two cheapest ways to explain a failure away.

---

## 4. Competing explanations and the discriminating experiment for each

| Likely outcome | Competing explanation | Discriminating experiment (Design B) | Pattern that discriminates |
|---|---|---|---|
| **O1.** H1 "supported" | E1. A few attention layers implement retrieval (genuine) | — | Survives every row below |
| | E2. Weak long-length reference (thin Stage B, PE, dilution) | G-REF (C0-normalised); 2× Stage-B branch; needle mass, entropy and maximum logit at 32K; temperature variant (S2, S9b) | Under E2: S(12, 32K)/S(12, C0) < 0.8, **or** the 2× branch gives > 5 pp, **or** needle mass at 32K is low |
| | E3. Task fits within GDN capacity | R2 ladder up to N = 512 (above the ≈ 256 key capacity); distance-controlled R1 | Under E3: k = 0 ≈ k = 12 for N ≤ 128 but diverges at N = 512 |
| | E4. Shortcuts (lexical anomaly, position heuristics) | D ∈ {3, 15}; binding-error rate; C2; depth sweep | Under E4: collapse at D = 15, high binding errors, C2 ≈ main condition |
| | E5. Ceiling | Ladder mean; N50 | Under E5: differences appear only at the top ladder levels |
| **O2.** H1 falsified | E6. Genuine | — | Survives every row below |
| | E7. Learnability under a fixed exposure | TEF 4× branch; learning curves at 250 / 500 / 1,000 / 4,000 | Under E7: the gap shrinks with exposure |
| | E8. Under-trained long stage or base LM | 2× Stage B at k = 1; LM-1 tail slope; Tier C (g): 3B tokens | Under E8: hybrids gain more from budget than k = 12 |
| | E9. Poor placement under the centred-uniform rule | k = 1 placement triad; Tier C: k = 3 end-of-group {3, 7, 11} | Under E9: the best single position recovers far more than the centred one |
| | E10. LR unfairness | Regret rule; per-config fallback; Tier C (h): η*/√2 | Under E10: the ranking flips at the alternative LR |
| **O3.** Gap grows with T (H5) | E11. The linear state is saturated by the amount of filler | Distance-controlled R1: fixed d, varying T; GDN config D (2× state) | Under E11: accuracy falls with T at fixed d, and 2× state delays the fall |
| | E12. Decay over distance (α_t) | Distance-controlled R1: fixed T, varying d | Under E12: accuracy depends on d, not on T |
| | E2. Reference collapse | G-REF | Gap is non-monotone, and S(12) falls relative to C0 |
| **O4.** Multi-hop needs more attention (H3) | E13. Composition needs ≥ 2 attention operations | Hops ∈ {1, 2, 3} at fixed k; Tier C: k = 2 adjacent {5, 6} vs spread {3, 9} | Under E13: k = 1 handles 1 hop but not 2; spread k = 2 handles 2 hops |
| | E14. Multi-hop is just harder (floor) | C0 controls; hop ladder; continuous log-prob | Under E14: C0 is also low, and the log-prob gap is uniform over k |
| | E15. Link order interacts with recurrence | Forward vs reversed order at 16K | Under E15: the order effect is large for k = 0 and small for k = 12 |
| **O5.** Aggregation insensitive to k | E16. Not attention-limited at this scale; E17. floor or ceiling | ρ ladder; C0; chance level 0.5 | Under E16: C0 is high and every k is equal at every ρ |
| **O6.** A hybrid beats all-full (strong H6) | E18. Local/global complementarity (genuine) | — | Survives every row below |
| | E19. Short conv | Attn+conv at k = 12 (S9c) | Under E19: the advantage vanishes once attention has the conv |
| | **E20. Temperature parameterisation (QK-norm dilution)** | Per-head gains + SSMax at k ∈ {1, 12} (S2 and S9b) | Under E20: the advantage vanishes with log-length scaling |
| | E21. Output gate / massive activations | Gated-attention control (Tier C) | Under E21: a gated all-full model closes the gap |
| | E10. LR | Per-config LR | — |
| **O7.** Measured speed-ups ≠ analytic | E22. Kernel maturity; E23. L2 residency; E24. launch overhead; E25. masked-cache dispatch | FlexAttention comparator; cold/hot L2; per-shape graphs; backend assertion; roofline fractions (§2.7) | Each artefact has a signature: cold vs hot gap ∝ L2-resident bytes; Flex vs cuDNN gap = maturity; graph vs eager = launch |
| **O8.** Placement effect at k = 1 | E26. Depth role; E27. no conv-derived features at layer 1; E28. the last layer cannot feed later layers; E29. seed noise | Positions {1, 6, 11} × 3 seeds; TOST; Tier C: position {0} | Under E28: layer 11 is the worst, with a large binding-error rate |
| **O9.** Seed-dominated noise | E30. Circuits form abruptly | Reserve seeds; per-seed TEF learning curves; continuous co-primary | Bimodal per-seed outcomes |
| **O10.** Arm differences caused by infrastructure | E31. FA2 nondeterminism (k ≥ 1 only), autotune picks, contention | Deterministic mode; pinned Triton cache; one bitwise replicate | Under E31: a replicate differs |

---

## 5. Questions I consider unresolved

1. **Can CARLA be paused for 2 × 2 h idle windows?** If not, measured costs are appendix-only and H6 is analytic. The idle speed-up is unknown. I estimate 1.2–1.4× from CARLA's ≈ 25–37 % utilisation [INF].
2. **Can a model with 41M non-embedding parameters become competent at 32K retrieval** with ≤ 0.3B long tokens plus 65M TEF tokens? No source settles this. S2 decides, and the stop/re-scope options need the user's prior agreement.
3. **Does SSMax with per-head temperature change the composition curve,** or only the reference's long-length level? S2 decides adoption; S9b checks the other choice.
4. **TF32 inside fla's GDN path.** It is on by default, and it explains the 1.5e-3 "fp32" error. Should it be disabled for numerical parity with attention's FP32-accumulate path? It is part of the mechanism's implementation, but it should be stated.
5. **Does the TEF re-warm (LR 3e-4) cause forgetting that differs by k?** Measure LM-2 before and after TEF. If it differs by more than 0.02 nats, lower the TEF LR.
6. **The ICC of branched queries,** and therefore the number of contexts needed. This is a pilot quantity.
7. **FlexAttention decode with a per-step length inside one captured graph.** I measured a static buffer with a fixed `mask_mod` length. Updating a device-side length without rebuilding the block mask still needs verification.
8. **Fraction vs count.** At L = 12, k = 1 is 8.3 %. H1 results are about counts as much as fractions; F24 is outside budget.
9. **Exposure option (i) vs (ii).** I chose TEF, i.e. (ii), for branchability and clean Tier-0 metrics. Option (i) is closer to production recipes. The other reviewers may disagree; the deciding factor should be whether exposure can be varied cheaply.
10. **Power and thermal behaviour in long benchmark sessions.** Software power-cap events were already logged, so a policy for discarding throttled reps needs testing.
11. **Is batch-1 decode representative at this scale?** The weights are 98 MB against a 64 MiB L2. The claims should say so.

---

## Appendix A. Measurement log [MEAS-B]

**Conditions.**
- RTX 5080: 84 SMs, 64 MiB L2, 15.46 GiB visible.
- torch 2.11.0+cu128, Triton 3.6.0, fla 0.6.0.
- CARLA running throughout: 25–37 % utilisation, 5.3–7.3 GB used by others, SM clock 2.2–2.8 GHz, ≈ 41–42 °C.
- Memory caps 4.8–7.1 GiB.
- Timings: CUDA events, min/median over 25–30 reps after 5 warm-ups.
- Scripts are in my session scratchpad (`…/scratchpad/revB/`): `det_test.py`, `det_model.py`, `num_test.py`, `bench.py`, `bench2.py`, `l2test.py`, `decode2.py`, `model_bench.py`, `model_bench_a.py`.

| # | What | Setup | Result |
|---|---|---|---|
| A1 | Backward determinism, kernels | bf16; `varlen_attn` with 8 documents (32,768 tokens); SDPA 1 × 8 × 16,384 × 64, causal; fla GDN config A with varlen | `varlen_attn`: forward bitwise; backward not (4.9e-4), bitwise in deterministic mode. SDPA FLASH 9.8e-4 → bitwise; EFFICIENT 4.9e-4 → bitwise; cuDNN 1.2e-4 → "No available kernel" in deterministic mode. GDN layer: forward and all gradients bitwise. First call 46.2 s |
| A2 | Whole-model determinism | Proposed architecture; 8,192 tokens (2 × 4K, document-packed); all parameter gradients | k = 0 bitwise in both modes; k = 12 default not (3.05e-4); k = 12 deterministic bitwise, no op raises |
| A3 | GDN chunk numerics vs fp64 naive recurrence | B = 1, T = 32,768, H = 4, d_k = 64, d_v = 128; decays mixed (e^−0.5 … e^−1e-4) or near-unit (1e-4) | fp32/TF32: relative RMS 1.46–1.62e-3, flat across position bands; max absolute error 3e-4 (mixed) / 7e-4 (near-unit). bf16: relative RMS 5.1–6.1e-3; max absolute 1.3e-3 / 3.1e-3. Final-state relative max 1.0–1.2e-3 (fp32) and 5.3–5.4e-3 (bf16). Chunk prefill + recurrent last-64 vs single pass, bf16: 1.0e-3 / 2.0e-3 |
| A4 | Training throughput and memory | Proposed model (49,395,808 / 49,297,408 params); bf16; fused AdamW stepping every micro-batch; FA2 varlen + RoPE + QK-RMSNorm; fla GDN-A; chunked CE (4K chunks, recomputed); lognormal documents (median ≈ 690 tokens) | Stage A, 8K micro-batch, no checkpointing: k = 0 / 3 / 12 → 88.9K / 90.5K / 90.5K tok/s (best), 4.48–4.64 GiB. 16K micro-batch without checkpointing: **OOM at 7.1 GiB** for every k. With checkpointing, 32K micro-batch: 77.2K / 72.3K / 63.8K tok/s, 2.54–2.59 GiB; 64K micro-batch: 76.6K (k = 0) and 60.3K (k = 12), ≈ 4.5 GiB. Stage B, 1 × 32K with checkpointing: k = 0 70.5K, k = 3 51.0K, k = 12 31.9K best / 27.2K median tok/s, ≈ 2.6 GiB. k = 12 at 32K without checkpointing: OOM |
| A5 | Prefill, single layer, no grad, B = 1 | d = 512 | T = 4K: flash 0.26 ms, cuDNN 0.23, efficient 0.47, FlexAttention 0.28; GDN full layer 0.69; GDN chunk core 0.11; 4d² projection 0.11 (76 TFLOP/s). T = 32K: flash 12.02 ms (91.5 TFLOP/s), cuDNN 11.32, efficient 25.61, FlexAttention 12.63; GDN layer 2.55; GDN core 1.02; projection 0.78 (89 TFLOP/s) |
| A6 | Deterministic-mode cost | FA2 forward + backward | `varlen_attn` 32K document-packed 9.46 → 15.33 ms (+62 %); 8 × 4K 5.40 → 7.70 ms (+43 %); SDPA-flash single 32K 44.0 → 73.0 ms (+66 %) |
| A7 | Decode step, single layer, CUDA-graph replay | KV bf16, 8 heads × 64 | B = 1, L = 32K, hot cache: flash 62 µs, cuDNN 43, efficient 1,510, math 558, static cache + bool mask under default dispatch **1,804**. B = 1, L = 4K: flash 17, cuDNN 7, efficient 183. B = 8, L = 32K: flash 600, cuDNN 591 (roofline 559). Cold cache, 32K: flash 90, cuDNN 85, FlexAttention 91; masked cuDNN 90; masked EFFICIENT 1,959; masked FLASH "No available kernel". Single call, hot → cold: cuDNN 33 → 84, flash 63 → 92. GDN `fused_recurrent` step plus state copy: 4–6 µs at B = 1 and B = 8 |
| A8 | QK-norm dilution | numpy; d_h = 64; distractor logits N(0, 1)·g²; needle logit 8·g² | See the table in §1.5 |

---

## Appendix B. Sources verified for this review

| Source | What I used it for | Access |
|---|---|---|
| Wang et al., [2507.06457v1](https://arxiv.org/html/2507.06457v1) | 340M recall "insignificant" (Table 2 caption); 2,048-token context (Limitations); recall averages 0.256 → 0.397 (v1 Table 5; table numbering differs between versions); Transformer 0.423; parameter matching and seeds not stated | [read-section] |
| Qiao et al., [2606.15378v1](https://arxiv.org/html/2606.15378v1) | S1–S3 trained at 16K with D ≥ 100N; NIAH/RULER only for S4/S5; S1 at 100N excluded as too small for stable long-context behaviour; restricting full attention to ≈ 2048 sharply raises log-LongPPL, restricting efficient layers barely matters (S4, 1000N) | [read-section] |
| Jelassi et al., [2402.01032](https://arxiv.org/html/2402.01032) | "100x less samples" to learn copying (≈ 160M parameters); 410M transformer beats 2.8B GSSMs on phonebook lookup with L ≥ 70 | [read-section] |
| Nakanishi, SSMax, [2501.19399](https://arxiv.org/html/2501.19399) | Softmax maximum → 0 as n grows; Eq. 2 multiplies logits by s·log n (equivalently, scales the query) | [read-section] |
| Chiang & Cholak, [2202.12172](https://arxiv.org/abs/2202.12172) | §5.3 log-length scaled attention; Table 1: no effect when train and test lengths match, +1 BLEU (p < 0.01) when training is short and testing long | [read-section] (PDF) |
| Chen & Revels, [1608.04295](https://arxiv.org/abs/1608.04295) | §B "Justifying the minimum estimator" | [read-section] (PDF) |
| Stuffed Mamba, [2410.07145](https://arxiv.org/html/2410.07145) | 36.4M / 47.0M / 84.6M configurations (Table 2); per-model passkey numbers not in the text | [read-section] |
| fla, [fla-org/flash-linear-attention](https://github.com/fla-org/flash-linear-attention) @ `98193113` (local clone) | Atomics; gradient reductions; Blackwell restrictions; autotune keys; mode auto-switch (files and lines cited in §1) | [code-inspected] |
| Triton 3.6.0 (venv) | Autotuner disk-cache key and `pre_hook` bypass (`runtime/autotuner.py` L170–260); `do_bench` L2 flush (`backends/nvidia/driver.py` L754–761) | [code-inspected] |
| PyTorch 2.11.0 (venv) | `varlen_attn` routes to `aten._flash_attention_forward/backward`; the cuDNN path is disabled there (`nn/attention/varlen.py` L29–31, L93, L326) | [code-inspected] |
| Stage-1 reports (01–05) and LR | FineWeb-Edu length statistics; RTX 5080 peak figures (04 §10.1); Bae placement ≤ 0.004 nats (LR §5); Birdie (LR §8); GDN key capacity (02 C6) | As tagged there |

**Not re-verified by me:** RTX 5080 whitepaper figures. My measured 91.5 TFLOP/s attention core and ≈ 0.9 TB/s effective DRAM bandwidth are consistent with them.

**Scope note:** this review does not edit `README.md`, `reports/decisions.md` or any other agent's file.

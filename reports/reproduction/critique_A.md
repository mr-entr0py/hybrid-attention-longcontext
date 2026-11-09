# Critique A: scientific validity of the two reproduction proposals

**Role:** adversarial reviewer A (scientific validity). **Date:** 2026-11-09, about 18:40 IST.

**Reviewed:**
- `proposal_pipeline-validation.md` ("PV": items M1–M7, S1–S4, N1–N4).
- `proposal_phenomenon-replication.md` ("PR": items P1–P4, S1–S5, N1–N5).

**Read for context:** README; protocol v1.0 (§2–§3, §7–§9, §13, §16, §17); `competing_explanations.md`; `INTERFACES.md`; `s1_reproduction.md`; `decisions.md`; and the relevant parts of `literature_review.md`, `03_evaluation.md` and `05_implementation.md`.

**What I did:**
- I ran no GPU job and edited nothing outside this file.
- I downloaded primary sources (arXiv HTML/PDF, HF `config.json`) into the session scratchpad and read the exact table rows.
- I inspected local code read-only (`third_party/zoology`, the installed fla 0.6.0, `src/hybridlc`, `tests/`, `scripts/s1_zoology`).
- I read the S1 run JSONs, the queue log and the kernel log between 18:20 and 18:35.

**Tags:**
- Access: [read-full], [read-section], [code-inspected].
- [MEAS-S1]: read by me from `experiments/s1_zoology/*`. These are interim, single-seed results.
- [KLOG]: read by me from `journalctl -k`.
- (fig, approx.): read off a figure, so approximate.
- [INF]: my inference or arithmetic.

---

## 0. Verdict on one screen

**Bottom line.** Most of the *literature* reproductions in both proposals would not validate what they claim to validate, for one of four reasons:
- the published anchor does not transfer to GDN at our scale;
- the tolerance cannot fail at the observed ceiling;
- the item tests a component or configuration we do not use;
- a deterministic unit test answers the same question in minutes.

Meanwhile the biggest live threat to "our setup is correct" is not in either proposal: **the S1 run is hitting GPU faults on this machine.** The validations that matter are cheap, about **3.0–3.6 new GPU-h**. The proposals' must tiers add about 8.0 new GPU-h (PV) and about 5.8 (PR, worst case about 8).

| # | Finding (verified) | Consequence | Action |
|---|---|---|---|
| F1 | **S1 lost a gate cell to GPU lockups.** `gated_delta_net-d128-lr1.0e-02` was "FAILED after 2 attempts" with `cudaErrorLaunchTimeout` [MEAS-S1]. The kernel log has `NVRM: krcWatchdog_IMPL: RC watchdog: GPU is probably locked! Notify Timeout Seconds: 7` plus **Xid 8** at 18:16:50 and again at 18:25:42 (the two failed attempts), and an **Xid 31 MMU fault** at 18:03:09 in another python process [KLOG]. The GPU reports `Display Attached: Yes`, `Display Active: Enabled`. A second GPU job (a scratchpad copy of `run_one.py`) was running next to the queue at 18:32. **Also:** GDN d128 at lr 3.2e-2 *diverged* (32 epochs, final overall 0.00504) [MEAS-S1]. Queue phase 2 (logged 18:43:57) schedules **all 4 LRs at d256, uncut**, including lr 1e-2. | A reproducible hang or fault under fla GDN training is a setup-validity problem that no literature reproduction would catch. The production GDN-A uses the same `chunk_gated_delta_rule`. The cause is not established. Both failures come from the high-LR end. | **X0** (§7): root-cause it and run a soak test before FEAS. Log a decision row now (§10). **Apply the pre-declared 2-best-LR cut to d256 now** ({1e-3, 3.2e-3}, chosen at d128), so the queue does not repeat the LR that hung the GPU twice (§3.6). |
| F2 | **G-RP cannot fail at the observed ceiling.** "Overall" accuracy is the *unweighted mean of the 7 test cells*: (1 + 0.999875 + 1 + 1 + 1 + 0.9998125 + 0.92501)/7 = 0.98924, which equals `final_overall_acc` [MEAS-S1, INF]. ±0.03 on overall therefore means ±0.21 on the one cell that discriminates (1024/256); "≥ 0.95 overall" means that cell ≥ 0.65. Attention scores 0.9999–1.0 in **every** cell at all four LRs after one epoch. | G-RP(b) cannot detect anything short of a catastrophic mixer defect. The monotonicity clause is noise at ceiling. | Keep G-RP as frozen, but report per cell. Pre-declare how a ceiling-noise failure is read **before any d256 result is read** (§3.6). Use M7 (GDN at d = 64) for real discrimination. |
| F3 | **R-S1-4 / the STATUS "causal_conv1d" worry is aimed at a non-issue.** The message comes from Zoology's `_compute_state_size`, which tries to import the Mamba modules inside `try/except` (`zoology/model.py` L270–277; `mixers/mamba.py` L11–14; `mamba2.py` L23–26). It has no effect on the GDN or attention runs. GDN uses fla's `ShortConvolution`, whose default backend is `'triton'` (`fla/modules/causal_conv1d/ops.py` L1118–1150) [code-inspected]. | The real gap: fla's Triton causal conv, used by Zoology's GDN, **our GDN-A and our F_gc**, is never checked against an independent reference. VT-09(2) compares fla with fla. No test in `tests/` compares it with `causal_dwconv_reference`. | Retarget it as **X1**, a unit test (§7). |
| F4 | **The S1(b) residual-difference list is incomplete.** Zoology's GDN casts q, k, v, β and g to **bf16** before the kernel (`mixers/gated_delta_net.py` L245–250), yet Zoology trains in fp32 (no autocast in `train.py`). In the same fp32 harness our mixers run fp32 kernels. Our attention then runs the fp32 SDPA-MATH reference path, **not** the bf16 `varlen_attn` (FA2) kernel used in LM training (`attention.py` L198–201) [code-inspected]. The micro-batch-64 accumulation shim is mathematically exact up to dropout RNG, but is recorded only as a `# DEVIATION` comment in `run_one.py` L106 and the run JSONs, not in `decisions.md`. | S1(b) and P1(c) compare different numerics, and do not exercise the production attention kernel. | Run "ours" arms under bf16 autocast as well. Add the bf16 casts to the S1(b) difference list. Log the shim (§10). |
| F5 | **P2 (Wang 1.3B RULER) contradicts the source's own per-subtask table.** In Wang Table 5, on *Singlevalue* (the niah_single tasks P2 picks) the Transformer scores **0.559** against **0.875** for GDN-hybrid-3-1. On VT both are near floor (0.066 / 0.024). Pure-GDN subtask values are not tabulated. The training context is 2,048 (v1) or 4,096 (v2). The released "3-1" has **8 of 24** attention layers, and attention head count changes with ratio (§1). | P2(a-i) ("Transformer − pure ≥ +0.03"; "3:1 − pure ≥ +0.04" on that subset) has no tabulated anchor, and its Transformer clause is contradicted by Table 5. A "failure" would say nothing about our pipeline. | **Drop P2 as designed.** Salvage its floor guard as X5 (§7). |
| F6 | **The M5/P3 ratio-loss anchor does not transfer to GDN.** Mamba-2 Table 2 is a 48-layer SSD + attention stack (no MLPs, per §9.2.3's "SSD and Attention" setup [INF]), 350M parameters and 7B Pile tokens. For GDN, Wang Table 6 has *pure* GDN with the best LM average (0.487 vs 0.468–0.481 for hybrids). Kimi has no pure-linear point, so it cannot anchor the 0 → 1 step. | P3-a/P1 ("the first attention layer lowers loss") can fail with no bug, and PR makes that failure an audit trigger. φ is a ratio of noisy small differences. P3-c's min-of-4 is biased towards passing. | **Drop the new k ∈ {1, 2, 6} runs.** Keep RP-2 (already in the protocol, free on S3). Optionally add one noise-floor seed (X7). |
| F7 | **Several items are redundant with S0 tests that already exist.** `tests/test_vt13_training.py` covers schedules, weight-decay groups (fake *and* real model), accumulation = full batch, kill-9 resume and a 20-step real-model smoke. `tests/test_vt09_transplant.py` covers C12↔Qwen3 (fp32 CPU and bf16 varlen GPU) and C0↔fla GDN. | M4's r2, M2(c) and the attention-correctness part of M6 add little. | Replace them with the deterministic tests X2–X3. |
| F8 | **Important validations are missing:** environment stability (X0); an independent conv check (X1); a Zoology↔ours weight-transplant equivalence test (X2); a trainer reference-loop test including clipping and z-loss (X3); **a known-answer recall test inside *our* harness** (X4); **solvability of our 16K/32K items** by a public long-context model (X5); and free audit triggers on S3/FEAS checkpoints (X6). | These are the checks whose failure would be diagnostic *and* would change what we do. | §7–§8. |

**Credit where due.**
- PV's finding 3 is correct (I re-verified it, §1). So are its cut-rule diagnosis, its "prefer diagnostic failures" principle (P4) and its "never tune to a number" rule (P5).
- PR's RoPE caution on P1(c-1), its "repro" RNG firewall and its idea of measuring ICL in-run (P4a) are sound.

---

## 1. Target numbers verified against the primary sources

| Claim (used by) | Source and location | What the source says | Access | Verdict |
|---|---|---|---|---|
| GDN recall by ratio (protocol §2 prior; PR P2; PV M3) | [2507.06457v2](https://arxiv.org/html/2507.06457v2) App. E **Table 7**, GatedDeltaNet column | 24:1 **0.341**, 12:1 **0.352**, 6:1 **0.425**, 3:1 **0.436**, Pure **0.356**. Caption: "Recall performance results … are averaged" (what is averaged is not stated) | read-section | ✓ exact |
| Transformer RULER 0.423; GDN-3-1 0.440 (PR P2; PV N1) | same, App. C **Table 5** (columns Multikey / Multiquery / Multivalue / Singlevalue / CWE+FWE / QA / VT / Avg) | GDN-hybrid-3-1: 0.293 / 0.542 / 0.502 / **0.875** / 0.271 / 0.300 / **0.024** / 0.440. Transformers: 0.332 / 0.464 / 0.472 / **0.559** / 0.284 / 0.412 / **0.066** / 0.423 | read-section | ✓. **The per-subtask values undercut P2** (F5) |
| 340M LAMBADA: GDN-pure 0.348, Transformer 0.353 (PV M2b) | same, **Table 2** (340M) | "GatedDeltaNet pure … LMB 0.348"; "Transformers … LMB 0.353". Caption: 340M recall "insignificant … We omit recall benchmarks" | read-section | ✓ exact. Note: the Table 2 GDN row *is* the pure model |
| 1.3B GDN 24-1 LAMBADA 0.502 (PV S4) | same, **Table 3** | "GatedDeltaNet 24-1 … 0.502" | read-section | ✓ |
| GDN LM average by ratio (PV M5 caveat) | same, **Table 6** | 24-1 0.468, 12-1 0.478, 6-1 0.481, 3-1 0.478, pure **0.487**; "averaged across 1.3B and 340M" | read-section | ✓. Pure GDN is best |
| Ratio semantics (PV finding 3) | same, App. D and §4.3 | "fixed ratio r:1 (linear : full) … memory footprint is reduced by roughly a factor of r … r ∈ {24, 12, 6, 3}". §4.3: "devote roughly one quarter of the layers to full attention … cutting KV-cache memory by a factor of four" | read-section | ✓. The text says ¼ |
| Released ratios (PV finding 3) | HF `config.json` of `m-a-p/*` (fetched 18:2x) | 340M and 1.3B "3-1": attention at layers [2, 5, 8, 11, 14, 17, 20, 23] = **8 of 24**, 16 heads / 8 KV. 340M "6-1": [6, 11, 17, 23], 32 heads. 1.3B "24-1": [23], 32 heads. GDN configs: `max_position_embeddings` 2048. `transformer_1.3B_baseline`: 32 heads, **8 KV heads**, `max_position_embeddings` 4096, tied | code-inspected | ✓. Released "3-1" is ⅓ attention, contradicting §4.3's ¼ |
| Attention geometry in those configs | fla 0.6.0 `fla/layers/attn.py`: `self.head_dim = self.hidden_size // self.num_heads` | So at 1.3B attention head_dim is **128** for 3-1 and **64** for 24-1 (at 340M: 64 vs 32) [INF from code + configs] | code-inspected | Wang's "ratio sweep" confounds attention head size with ratio |
| Training context (U1) | v1 and v2 Limitations | v1: "a 2,048-token context window". v2: "a 4,096-token context window" | read-section | ✓ still unresolved |
| Mamba-2 ratio perplexity (PV M5; PR P3) | [2405.21060v1](https://arxiv.org/html/2405.21060v1) §9.2.3 **Table 2** | 0: 8.60; 1: 8.38; 2: 8.32; 3: 8.29; 4: 8.29; 5: 8.28; **6: 8.26**; 7: 8.27; 9: 8.28; 11: 8.30; 15: 8.34; 24: 8.50; Transformer++: 8.68. Setting: "350M model (48 layers) trained to 7B tokens on the Pile with the GPT-2 tokenizer" | read-section | ✓ exact. PV's log arithmetic is also correct (0.0259 / 0.0403 = 0.64) |
| Olsson ICL score, magnitudes, 1-layer, smeared keys, timing (PV M6; PR P4) | [Olsson et al. 2022](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) | Key Concepts: "loss of the 500th token … minus the average loss of the 50th token". Argument 1 uses the opposite sign ("50th token loss minus the 500th"): "roughly 2.5 to 5 billion tokens … less than 0.15 nats … roughly 0.4 nats … (except for the one layer model where not much in-context learning ever forms)". Model details: 1–6 layers, "both with MLPs and without", context 8192, 2^16 vocabulary, d_model 768, "10,000 steps (~10 billion tokens)", phase change "approximately 1–3 billion tokens", warm-up "over the first 1.5e9 tokens". Yet: "doesn't appear to correspond to a scheduled change in learning rate, warmup, or weight decay". Smeared keys "only at one-layer and two-layer sizes": ICL "does indeed form for one-layer models … and it forms earlier for two-layer and larger models" | read-section | ✓. Both proposals quote real lines (the paper uses both sign conventions). No 1-layer ICL value is given in the text, so the 1-layer thresholds are unanchored |
| Pythia LAMBADA (PV M2a) | [2304.01373v2](https://arxiv.org/abs/2304.01373) App. G.1 "Raw Benchmark Scores – Zero Shot" | Lambada (OpenAI): 70M **0.185 ± 0.005**, 160M **0.328 ± 0.007** | read-section (PDF text) | ✓ exact (± is the binomial SE, not seed variance) |
| nanoGPT 1.4697 (PV M4; PR S4) | `third_party/nanoGPT/README.md` L51 | "the best validation loss is 1.4697". `train.py` defaults to `compile = True`, seed 1337 | code-inspected | ✓ |
| GPT-2 lm-eval LAMBADA (PR S3) | [HF discussion #90](https://huggingface.co/openai-community/gpt2/discussions/90) | acc 0.32564 ± 0.00653, ppl 40.06 ± 1.49, from an old harness (`hf-causal-experimental`; commits 441e6ac1 / b281b09) | read-section (WebFetch) | ✓, but a community number on an old harness |
| GDN MQAR vs state (PR P1 context; PV M7 premise) | Zoology README figure `zoology_results.png` @1ad20d1 | GDN points: about 0.89 at about 1.7e4 state and about 0.99 at about 6.6e4 (fig, approx.) | code-inspected (viewed) | ✓ consistent with our d128 run (0.989 at state 67,072 [MEAS-S1]). Supports M7's "mid-range at d = 64" |

**No proposal misread a number.** The problems lie in how the numbers are *used*:
- P2 applies a 7-group average to a 4-task subset whose own columns point the other way.
- M5/P3 transfer a Mamba-2 (SSD, no MLP) pattern to GDN.
- P4/M6 transfer token counts and a 0.15-nat boundary across tokenizer, data, scale and mixer.
- The protocol's prior line reads Wang's "3:1" as ¼ attention, while the released checkpoint is ⅓.

---

## 2. The four tests I applied to each item

A reproduction validates our setup only if all four hold:
- **V1. In-chain and as-used.** It exercises a component of our causal chain, *in the configuration and kernel path we use*.
- **V2. Can fail.** It runs in a regime where a realistic defect moves the outcome past the tolerance. Compute the minimum detectable defect, for example "±0.03 overall = ±0.21 on the one live cell".
- **V3. Anchor transfers.** The published number comes from the same mechanism family, scale regime and metric, or the item is labelled exploratory and cannot gate.
- **V4. Not dominated.** No deterministic test (weight transplant, reference loop, fp64 reference) answers the same question more cheaply and more sharply.

**Decision-relevance (the practical test):** before adopting an item, write down which outcome would change what we do, and what we would change. If no outcome changes anything, it is a demo, not a validation.

---

## 3. S1 / G-RP (PV M1 and R-S1-1…5; PR P1 and S5): what the running pilot can and cannot show

### 3.1 Environment events neither proposal covers (both were written before them)

- Queue log [MEAS-S1]:
  - `18:16:51 end gated_delta_net-d128-lr1.0e-02 rc=1`
  - `18:25:42 end … rc=1`
  - `FAILED … after 2 attempts`
  - Both attempts raised `torch.AcceleratorError: CUDA error: the launch timed out and was terminated`, inside `zoology_compat.train_epoch`.
- Kernel log [KLOG]:
  - Xid 8 and the RC-watchdog "GPU is probably locked!" at 18:16:50 (pid 811688) and at 18:25:42 (pid 818052), matching the two attempts.
  - Xid 31 (MMU fault, `FAULT_PDE ACCESS_TYPE_VIRT_WRITE`) at 18:03:09 in pid 799092, a python process I could not identify.
- At 18:43:57, `gated_delta_net-d128-lr3.2e-02` finished 32 epochs **diverged**: overall 0.00504, 1024/256 cell 0.0002 [MEAS-S1]. Attention d128 then reached 0.99995 / 0.99991 overall after one epoch (lr 1e-3 / 3.2e-3), 1.0 on 1024/256.
- Accuracy before the hang: attempt 1's validation accuracy oscillated between 0.962 and 0.981 up to epoch 15 [MEAS-S1]. Losing this cell does not change best-over-LR at d128: lr 1e-3 = 0.98924 and lr 3.2e-3 = 0.98746. But it is a permanent run failure that must be reported (protocol §8.1), and it is an environment signal.
- Possible causes, none established [INF]:
  - a Triton kernel stalling at a numerical state reached at lr 1e-2;
  - RC-watchdog sensitivity on a display-attached GPU;
  - interference from concurrent GPU jobs.
- Why this outranks most literature items: a stall in a 1–4 h Stage-A or Stage-B run is survivable through auto-resume (§18). An MMU fault in a project kernel would be a correctness problem.

### 3.2 Ceiling arithmetic (V2)

- Overall accuracy is the unweighted mean of 7 cells (verified to 1e-9, F2). With six cells at ≥ 0.9996:
  - ±0.03 overall ⇔ ±0.21 on 1024/256;
  - G-RP(a)'s 0.95 ⇔ 1024/256 ≥ 0.65;
  - PR's GDN-A "≥ 0.90" ⇔ ≥ 0.30.
- **Attention is at ceiling in every cell, including the 4× length-extrapolation cells.** Zoology's attention has no positional encoding; BaseConv supplies local position. So G-RP(b) for attention checks only that our module is not broken outright.
- **The only discriminating cell (1024/256) is an extrapolation cell**: 4× the training length, 4× the training load. Our confirmatory evaluations stay within the training length (protocol §1). M7 at d = 64 still discriminates mainly on the extrapolation cells. That is fine for a "same function?" check, but no MQAR item measures in-distribution capacity at these widths.

### 3.3 S1(b) and PV M7 answer "same function?" statistically; a weight transplant answers it exactly (V4)

- Zoology's GDN and `OurGdnZoology` call the same fla kernels with the same projections (2 heads × d/2, `expand_v` 2, conv 4, no gate). Zoology's unused `D` parameter does not enter the forward pass [code-inspected].
- Zoology re-initialises every `nn.Linear` inside its model, including ours, through `self.apply(_init_weights)`.
- So a **forward-and-gradient transplant test** (copy weights, compare outputs) settles equivalence in seconds. Tolerance: bf16 level for GDN, because of F4's casts.
- Attention: zero Zoology's biases and disable dropout 0.1 for the test. These are intended structural differences that the training comparison cannot separate from bugs anyway.
- The training comparison then only measures init and precision effects, and a mismatch there is not a bug.

### 3.4 R-S1-4 is mis-targeted (F3)

Replace "check Zoology's fallback conv" with X1: fla's Triton `causal_conv1d` against `F.conv1d` (or our own `causal_dwconv_reference`):
- forward and backward;
- with `cu_seqlens`;
- SiLU on / off;
- fp32 and bf16.

This covers GDN-A, F_gc, GLA and GDN-E.

### 3.5 P1(c-1) production F_g (V1)

- Keep it: an attention run costs about 17 s [MEAS-S1].
- But a fp32 run exercises `_attn_reference_varlen` (SDPA MATH), so **add a bf16-autocast arm** to exercise the production `varlen_attn` kernel.
- "≥ 0.95 on the in-distribution cells" cannot fail unless the module is broken. Report it as a smoke check, not a validation of temperature or gate behaviour.
- **P1(c-2)**: GDN-A at d = 256 has per-layer state 2 × 64 × 128 = 16,384, the same as Zoology's GDN at d = 128 (2 × 64 × 128), which scored 0.989. "≥ 0.90" is therefore lax by about 0.09 overall, about 0.6 on the live cell [INF]. Compare it with Zoology's GDN at d = 128 on 1024/256 using M7's tolerance, or make it descriptive.

### 3.6 Pre-declarations to log before any d256 result is read

GDN d256 starts at about 18:46 [MEAS-S1]. Its smoke epoch reached 0.987 overall, and early stopping fires above 0.99, so **a first d256 result could appear within minutes**; otherwise a run takes about 1 h [INF]. Log these first:
1. The per-cell table is part of the G-RP report (agree with PV R-S1-2).
2. **Monotonicity at ceiling.** Do not add a tolerance to the frozen gate after seeing d128 = 0.989 (PV R-S1-3 does this, and it is mildly data-informed). Pre-declare the reading instead: if G-RP(a) fails *only* because GDN(256) − GDN(128) ∈ [−0.005, 0) while both are ≥ 0.95, it is reported as **"FAIL at ceiling (uninformative)"**. The state-size claim is then decided by the d64 → 128 → 256 per-cell dose-response (M7). Otherwise "no LM training until G-RP passes" can stall the project on noise, or invite an ad hoc override.
3. The lost lr 1e-2 cell: best-over-LR is computed over 3 LRs at d128, and the failure is reported.
4. PR S5 (one extra seed of the best cell at ceiling) is dropped. A spread compressed by the ceiling would "validate" ±0.03 whatever its value. The seed basis comes from M7.
5. **Apply the cut now (my recommendation; PI decision).**
   - PV R-S1-1 and PR §2.1 both find that the cut was mis-triggered. Since then, queue phase 2 has scheduled all 4 d256 LRs [MEAS-S1].
   - At d128, lr 1e-2 hung the GPU twice and lr 3.2e-2 diverged (0.005). The pre-declared rule therefore picks {1e-3, 3.2e-3} for d256.
   - Applying it saves ≈ 2 h [INF] and avoids re-triggering the hang during X0.
   - Caveats: best-of-2 at d256 is barely biased, because the high LRs fail. lr 1e-3 is the grid's lower edge, so if d256 peaks there, report it as edge-limited.

---

## 4. Item-by-item verdicts

Verdicts: **KEEP**; **MODIFY**; **REPLACE** (by a deterministic test); **DEMOTE** (optional, non-blocking); **DROP**. "New GPU-h" is after my change [INF].

| Item | Claimed to validate | Verdict | Main reasons (V1–V4) | New GPU-h |
|---|---|---|---|---|
| PV M1 / PR P1(a,b): S1 as frozen | Environment; mixer equivalence | **KEEP**, reinterpreted | S1(a) does validate that the environment runs Zoology and that fla GDN learns MQAR to the level the figure shows. G-RP(b) has near-zero power (§3.2–3.3). One cell is lost (§3.1) | 0 (running) |
| PV R-S1-1 / PR §2.1: cut rule | Budget | KEEP (either version) | Budget-only | −1 to −2 |
| PV R-S1-2: per-cell report | Sensitivity | KEEP | Needed | 0 |
| PV R-S1-3: −0.005 tolerance | Avoid a spurious failure | **MODIFY** | Post hoc change to a frozen gate. Use the pre-declared reading in §3.6 instead | 0 |
| PV R-S1-4: conv fallback | Zoology fidelity | **REPLACE by X1** | Aimed at a non-issue (F3) | ≈0 |
| PV R-S1-5 / PR S5: seed basis for ±0.03 | Tolerance | **REPLACE by M7 seeds** | Variance at ceiling is compressed (V2) | −1.2 (PR) |
| PR P1(c-1): production F_g | Production mixer learns recall | KEEP, **plus a bf16 arm** | Free. Without bf16 it exercises the fp32 reference kernel (V1). Passes unless broken (V2) | ≈0.01 |
| PR P1(c-2): GDN-A ≥ 0.90 at d256 | GDN-A geometry | **MODIFY** | Same state as Zoology's GDN at d128 (0.989), so ≥ 0.90 is lax (V2) | 0 |
| **PV M7: MQAR at d = 64, 3 seeds × {Zoology, ours}** | Our GDN = Zoology's GDN where defects show | **KEEP, top priority** | d = 64 is Zoology's own sweep row; the figure puts GDN near 0.89 at about 1.7e4 state (fig, approx.), so the cells are mid-range (V2 ✓). Run X2 first. Add a bimodality rule (§5.7) and run "ours" in bf16 too | ≈1.0–1.7 |
| PV M2(a): Pythia LAMBADA via lm-eval | lm-eval environment | **KEEP-lite** | Not in our chain (V1 ✗), but it is the independent reference for M2(d). **lm-eval is not installed** in `.venv` (verified): use a separate venv | ≈0.05 |
| PV M2(b): Wang 340M **GDN-pure** LAMBADA 0.348 | fla 0.6.0 GDN semantics on trained weights | **KEEP** | Verified anchor. The pure model builds without flash_attn (`attn.py` only warns at import; `Attention.__init__` raises ImportError). Catches fla-version semantic drift in the layer we use (V1 partial ✓, V2 ✓, V3 ✓) | ≈0.05 |
| PV M2(b): Transformer 0.353 / 3-1 hybrid | Attention on trained weights | **DROP** | Needs an SDPA shim. fla's attention is not our F_g: no QK-norm, no gate, θ = 1e4, GQA (V1 ✗) | — |
| PV M2(c): transplant logits | Our mixers on trained weights | **DROP** (attention) / **DEMOTE** (GDN) | Attention runs a configuration we do not use. GDN is covered by VT-09(2) and by VT-04's near-unit decays at T = 32,768 against fp64 (V4 ✗). Also needs an untied-head wrapper | — |
| PV M2(d) / PR S3: scorer vs lm-eval per item | EM_tf definition | **KEEP-lite**, on GPT-2 (cached) and Pythia via `ForwardOnlyAdapter` | The cache path is already tested (VT-06/07, 5% re-score, I-29). The unique value is an independent definition check | ≈0.05 |
| PV M3: Tier-0 on Wang's 340M sweep | Harness k-sensitivity | **DEMOTE** (run only (i), and only if shims exist anyway) | (ii) and (iii) can fail with no bug (PV says so). Head size changes with ratio. The models are 2K-context | 0–0.5 |
| PV M4 / PR S4: nanoGPT ladder | Trainer | **REPLACE by X3** | r0 and r1 test nanoGPT and torch, not our chain (V1 ✗). r2's targets are already in `test_vt13_training.py`. Dropout and train/eval mode do not exist in our model (V4 ✗) | −1.5 |
| PV M5 / PR P3: ratio-loss shape | The whole LM pipeline across k | **DROP the new runs.** Keep RP-2 on S3. Optional X7 | Anchor does not transfer (F6, V3 ✗). Criteria are noise-dominated (§5.2, V2) | −1.4 (PV) / −3.2 to −4.2 (PR) |
| PV M6 / PR P4b: Olsson smeared keys | Attention correctness; E1.6 premise | **DEMOTE** to an optional pilot; **remove from G-RV** | The attention-correctness part is covered by VT-02/03/09 (V4). The 1-layer thresholds are unanchored, and 0.1–0.2B tokens is far below the 1–3B where the phase change occurred (V3). X-gc12 runs regardless, so no decision depends on it | 0–0.6 |
| PR P4a: ICL in-run | Budget validity | **MODIFY**, keep as a free measurement | **C1 and C3 have no FEAS Stage-A runs** (FEAS is k ∈ {0, 12}: protocol §7.2 and §8.2), so P4-a1 cannot be evaluated pre-G5 for them. The 0.15-nat boundary is a transformer number on Olsson's data, not a GDN or our-data number. Use a plateau criterion (§5.3) | ≈0.1 |
| **PR P2: Wang 1.3B RULER + our generators** | Suite sensitivity to k | **DROP as designed**; salvage the floor guard as X5 | F5 (V3 ✗). Our generators would run through `ForwardOnlyAdapter`, not the production cache path (V1 partial). Engineering: nltk is not installed; `PaulGrahamEssays.json` is absent (only a URL list and downloader); needs a flash_attn shim, 6 × about 2.7 GB of downloads, and an fla registration patch for RULER's HF path. Outputs cannot change any decision (I-23) | −2.0 |
| PV S1 / PR S1: Jelassi copy | Size of the learning-speed confound | **DROP** | The anchor is GSSM/Mamba, not GDN; copying is not the TEF task (V3 ✗). The protocol's TEF dose ladder (250 → 4,000) and still-closing flag measure this *in situ* (V4 ✗). GDN may never reach 0.95 at L = 300, leaving the ratio undefined | −1.5 to −3 |
| PV S2 / PR S2: length behaviour on FEAS checkpoints | Position handling | KEEP the **Stage-B check (PV (ii)) and PR's within-window sanity check** as free checks; (i) descriptive | Press et al.'s numbers (θ = 1e4, L = 512/1024) do not transfer to θ = 5e5 with QK-norm at 4K (V3), so (i) has no direction anchor | ≈0.3 |
| PV S3: RULER vs ours | Generator semantics | **DROP** | "≤ 10 pp *or an explained difference*" cannot fail | — |
| PV S4: Wang 1.3B 24-1 LAMBADA | Second transplant | **DROP** | Needs the shim (one attention layer); redundant | — |
| PV N1–N4, PR N1–N5 | Environment / extras | **DROP** | Not in our chain; or `mamba_ssm` is needed; or figure-only anchors | — |

---

## 5. Why the expensive or gating items fail the validity tests

### 5.1 PR P2 (Wang 1.3B ratio sweep through RULER and our generators)

1. **The subset contradicts the anchor.** On the chosen subset {niah_single_1/2/3, vt}, the only tabulated values (Table 5) put the Transformer *below* GDN-3-1: Singlevalue 0.559 vs 0.875, VT 0.066 vs 0.024. Weighting the subset as P2 does gives about 0.44 vs about 0.66 [INF]. Wang's text says single-key tasks vary with ratio, but only Fig. 5 (figure-only) shows by how much. Hence:
   - "S(Transformer) − S(pure) ≥ +0.03" is contradicted by the table;
   - "S(3:1) − S(pure) ≥ +0.04" is figure-anchored at best.
2. **Context length.** Training context is 2K in v1 and 4K in v2; GDN configs say `max_position_embeddings` 2048. At 4K the attention layers (θ = 1e4) may be extrapolating while pure GDN is not. That biases the contrasts against attention.
3. **The ratio is not a clean IV:**
   - "3-1" = 8 of 24 attention layers (⅓);
   - attention head_dim differs by 2× between ratios (verified: 1.3B 24-1 and 340M 6-1 use 32 heads, 3-1 uses 16, giving head_dim 128 vs 64 at 1.3B; 12-1 not fetched);
   - the Transformer baseline is GQA with 8 KV heads.
4. **No decision depends on it.** Protocol I-23 forbids using Wang's numbers for power or ceiling planning, so its "k-curve prior" output (b-iii) cannot change any decision. The useful kernel, "can a competent public model solve our cloze items?", becomes X5. It is cheaper with a pure GDN model (no shim) or a long-context public model.

### 5.2 PV M5 / PR P3 (ratio-loss shape at our scale)

1. **V3.**
   - Mamba-2 Table 2 is a 48-layer SSD stack in which attention blocks replace SSD blocks, with no MLPs [INF from §9.2.3: MLP variants appear only in the 2.7B comparison]: 350M parameters, 7B tokens.
   - Waleffe is figure-only and a Mamba-2 stack.
   - Bae (per the literature review) is Mamba.
   - Kimi's ablation, as quoted in PR, runs from KDA:MLA 0:1 (pure attention) to 15:1. It has **no pure-KDA point**, so it cannot support "the first attention layer gives most of the gain".
   - For GDN, Wang Table 6 has pure GDN with the best LM-accuracy average.
   - A GDN-A stack (short conv plus delta rule) at 21M non-embedding parameters, 0.15B tokens and 4K context, on long-poor FineWeb-Edu (median document 688 tokens per PR), may show a 0 → 1 step of roughly zero. That would be genuine, not a bug.
2. **V2.** The criteria are dominated by noise:
   - **φ** = (L₀ − L₁)/(L₀ − min L) is a ratio of differences that may be about 1–3σ each. Its point estimate from seed-100 values is nearly uninformative.
   - P3-c's min over 4 configurations against one is biased towards passing (winner's curse).
   - PV's σ_pair comes from one extra seed at two configurations (df about 2).
3. **What it costs if adopted as written.** PR makes P3-a an audit trigger, which would send the team chasing a non-bug. The diagnostic content (RP-2: the k = 0 → 12 step and LM-3 repeat-token loss at Δ ∈ [1K, 4K) for k = 12 vs 0) is **already in protocol §17 on the S3 runs, at no cost**.

### 5.3 PV M6 / PR P4 (Olsson)

- **Measurement side (P4a).** Measuring the ICL score L@50 − L@500 on held-out documents with ≥ 512 tokens is cheap and worthwhile. Change the criterion to:
  - (i) *k = 12*: the score has plateaued (|ΔI| < 0.02 over the last 20% of Stage A) on FEAS and S3 checkpoints;
  - (ii) report k = 0 descriptively.
- **Why the 0.15 boundary goes.** It is Olsson's transformer pre/post value on a 2^16-vocabulary dataset. A GDN model lowers late-position loss through its recurrent memory without any induction phase change.
- **C1 and C3 cannot be checked before G5** (no FEAS Stage A at k ∈ {1, 3}).
- **Smeared-key arm (P4b / M6).**
  - Olsson presents smeared keys "only at one-layer and two-layer sizes" trained to about 10B tokens. No 1-layer ICL value appears in the text.
  - Our runs differ: RoPE after QK-norm (Olsson's small models use a positional variant "similar to Press et al."); 16K vocabulary; SwiGLU MLPs; 0.1–0.2B tokens.
  - (a)/(P4-b2) can fail for budget reasons. The 1-layer thresholds (I < 0.15; ≤ ½ of 2-layer) are unanchored.
- **Decision impact.** None: X-gc12 runs unconditionally in Tier A, so the outcome changes nothing. Keep it optional, about 0.6 GPU-h, as a mechanism pilot, and **never as a FEAS gate**. Gating on it (PV G-RV) adds schedule risk with little diagnostic yield.

### 5.4 PV M2/M3: public checkpoints

M2(b) for GDN-pure is the one external behavioural anchor for *the library we use*: trained weights, fla 0.6.0, a verified 0.348.
- It is cheap and needs no shim. Keep it.
- Everything that needs the flash_attn shim runs fla's attention, which is not our F_g. It also stacks unvalidated wrappers (shim, untied head, GQA→MHA, overrides) in front of the test. Drop it.
- M3(i), the positive control that a Transformer copies random tokens, is a good harness check. It is worth running only if the shim exists for another reason.

### 5.5 PV M4 / PR S4: nanoGPT

- **What it isolates:** rung r0 checks torch and SDPA on sm_120, which S1 and VT-02/09 already exercise. Its 1.4697 comes from one A100 run with `compile = True`, and no variance is published.
- **Rung r2 is the only rung in our chain,** and it needs adapters:
  - a char-level `StageStream`;
  - nanoGPT's `GPT` returning pre-head hidden states;
  - schedule changes (rung r1).
  These adapters are new places for bugs.
- **The overlap with existing tests:** what r2 claims to isolate (optimiser groups, schedule, accumulation, clipping, loss averaging, resume) is mostly in `tests/test_vt13_training.py` already.
- **Replace it with X3:** our trainer against a 60-line plain-PyTorch AdamW loop, **on our model**, in fp32 deterministic mode with identical batches, matching loss traces to about 1e-6 over 50–200 steps. It also covers the two pieces not seen in the test list: grad-norm clipping and z-loss scaling.

### 5.6 Jelassi copy (both S1 items)

The learning-speed rival (E1.7, E2.2, E6.7) is already measured on the *actual* tasks by the TEF dose ladder and the still-closing flag. A copy task with a 30-symbol vocabulary at L ≤ 300, scaled to 4 or 6 layers, gives a number that does not map onto TEF dose sufficiency. Drop it.

### 5.7 PV M7: keep it, with these changes

1. Run **X2** (Zoology↔ours transplant) first.
   - If X2 passes, a failing M7 points to init, precision or training dynamics. That is a recipe difference worth reporting, not a bug.
   - If X2 fails, fix the wrapper before training anything.
2. Run "ours" in fp32 **and** under bf16 autocast. Zoology's GDN feeds bf16 inputs to the kernel (F4), and production runs bf16.
3. **Bimodality rule** (MQAR learning at capacity can split by seed). If either arm's 3 seeds span more than 0.2 on a discriminating cell, report seed-level outcomes, add 2 seeds per arm (pre-declared), and compare distributions descriptively. No pass/fail on means.
4. Keep PV's pre-declared fallback configurations (1024/512, 2048/512) in case d = 64 is not mid-range.

---

## 6. Redundancy map: proposed items vs S0 tests

| Proposed item | Already covered by | What remains unique |
|---|---|---|
| PV M2(c) transplant on trained weights | VT-09 (`test_c12_vs_qwen3_cpu_fp32`, `…gpu_bf16_varlen_close`, `test_c0_vs_fla_gated_deltanet_lm_fp32`); VT-04 (near-unit decays at T = 32,768 vs fp64) | Little: trained-regime activation scales, in a configuration we do not use |
| PV M2(d) / PR S3 scorer | VT-06, VT-07, VT-08, I-29 5% re-score | Independent EM_tf definition (lm-eval). Keep the lite version |
| PV M4 r2 trainer | `test_vt13_training.py`: schedules, weight-decay groups on the real model, `test_accumulation_equals_full_batch_tef_weights`, kill-9 resume, real-model smoke, protocol audit | Clipping and z-loss over many steps → X3 |
| PV M6 attention correctness | VT-02 (causality including fp64 zero future gradient), VT-03 (isolation; **test file not yet present**), VT-09 | The E1.6 mechanism premise (optional pilot) |
| S1(b) / M7 "same function" | Nothing yet | **X2** (deterministic); M7 for behaviour |
| R-S1-4 conv | Nothing: VT-09(2) is fla vs fla; VT-10 checks init equality only; VT-08 logs ops | **X1** |

---

## 7. Missing validations, ranked by risk covered per GPU-hour

| ID | Validation | Why it matters | Pre-registered pass / fail | GPU-h [INF] |
|---|---|---|---|---|
| **X0** | **Environment stability.** (a) Reproduce the lr 1e-2 hang alone on the GPU with `CUDA_LAUNCH_BLOCKING=1` to find the kernel; log loss and grad-norm finiteness per step. (b) Identify the Xid 31 process (an S0 test?). (c) A ≥ 1 h soak at production shapes under `journalctl -k` Xid monitoring: D384 C0 and C12 Stage-A 2 × 4K steps, a Stage-B 1 × 32K loop with checkpointing, GDN-E at K = 256 | F1. A hang or MMU fault in fla or attention kernels would invalidate long runs or silently corrupt them. Nothing else in either proposal can detect it | Zero Xid events in the soak and a root cause for F1. Otherwise a documented mitigation (headless display, LR-independent repro, kernel workaround) and a runner rule that tags any run overlapping an Xid event | ≈1.0 |
| **X1** | fla Triton `causal_conv1d` vs `F.conv1d` / `causal_dwconv_reference`: forward and backward, `cu_seqlens`, SiLU on/off, fp32 and bf16 | F3. Shared by GDN-A, F_gc, GLA and GDN-E; never checked independently | Max abs diff ≤ 1e-5 (fp32) and ≤ 1e-2 relative (bf16); exact zero across document boundaries | ≈0 |
| **X2** | Zoology↔ours weight-transplant equivalence. GDN in Zoology geometry: compare with Zoology's inputs cast to bf16 as Zoology does. Attention: Zoology MHA with biases zeroed and dropout off vs `OurFgZoology` | §3.3. Makes S1(b) and M7 interpretable | Outputs and gradients within bf16 tolerance (GDN) or fp32 tolerance (attention) | ≈0 |
| **X3** | Trainer vs a plain-PyTorch reference loop on our model: fp32, deterministic, identical batches, 50–200 steps, clipping and z-loss on | §5.5. Replaces M4 | Loss trace |Δ| ≤ 1e-6 per step; parameter checksum equal at the end | ≈0.05 |
| **X4** | **Known-answer recall in *our* harness**: our TEF stream restricted to R-load (no-filler or repetitive filler) at 1K, our trainer, a 2-layer `AA` vs `--` model, our scorer, from scratch, 25–50M tokens | The missing third rung of the ladder (Zoology harness + Zoology mixer → Zoology harness + our mixer → **our harness + our task**). It tests the synthetic-stream → loss-mask → model → scorer chain against a known answer before FEAS spends 9.8 h. Zoology's attention solved MQAR in one epoch, about 24M tokens [MEAS-S1, INF] | `AA`: EM_tf ≥ 0.9 at N ∈ {8, 32} within the token cap. `--` descriptive. If `AA` fails: pipeline bug | ≈0.3 |
| **X5** | **Item solvability at length**: our generator code (with a foreign `tok`) at 4K, 16K and 32K (R1 D ∈ {3, 15}, R-load N ∈ {8, 32, 128}, plus C0), scored for a ≤ 1.5B public base model whose card states ≥ 32K training context (verify the card before use; scorer via `ForwardOnlyAdapter` computing only the needed logits) | FEAS and G-REF decide scale (D512 adds about 58 GPU-h), L_ext and re-scoping, all on 16K/32K items that nothing has yet shown to be solvable. VT-12 checks structure, not semantics | C0 R1 D = 3 ≥ 0.9; 32K R1 D = 3 ≥ 0.7; R-load decreasing in N. A failure means inspecting the items before interpreting FEAS. **Never used for power or ceiling planning** (I-23) | ≈0.3 + download |
| **X6** | Free audit triggers on planned runs: RP-2 (k = 0 vs 12 LM-2; LM-3 at Δ ∈ [1K, 4K)) on S3; ICL score and plateau for k ∈ {0, 12} on FEAS and S3; LM-1 within-window monotonicity; LM-1 [8K, 16K) after Stage B (PV S2-ii) | Detects "attention does not use context" or "Stage B did not teach long positions" before G5 | As written in the protocol and PV S2; audit triggers, not gates. Needs the S0 eval-hook / snapshot request (PR) | ≈0.1 |
| X7 (optional) | S3 noise floor: seed 103 at k ∈ {0, 12} at η* | The LR rule's 0.02-nat regret threshold and any LM-level audit rest on single-seed runs | Report σ_pair; if σ_pair > 0.01 nats, flag LR selection as noise-limited | ≈0.7 (D384) |

---

## 8. The cheapest set that covers the most risk

**Order:**
1. **Now (0 GPU-h):** the §10 decision rows; the §3.6 pre-declarations before any d256 result is read.
2. **S0 add-ons (≈0.05 GPU-h):** X1, X2, X3, written as tests in S0's files. That is the builders' call, not mine.
3. **X0 (≈1.0):** before any further long GPU queue, and before FEAS.
4. **M7 with X2 first, plus the P1(c-1) bf16 arm (≈1.0–1.7):** these reuse the S1 harness.
5. **X4 (≈0.3)** as soon as the trainer and TEF stream are usable.
6. **M2-lite (≈0.15), in a separate venv:** lm-eval LAMBADA for GPT-2 and Pythia-70M/160M, compared with the published 0.185 / 0.328 ± 0.015; Wang 340M GDN-pure against 0.348 ± 0.02; our scorer's per-item agreement ≥ 99.5% on GPT-2.
7. **X5 (≈0.3)** before FEAS results are *interpreted*.
8. **X6** scheduled into FEAS and S3; X7 if the budget allows.

**Totals [INF]:**
- **≈3.0–3.6 new GPU-h**, plus 0.7 (X7) and 0.6 (smeared-key pilot) optional.
- Compare PV must ≈8.0 new and PR must ≈5.8 new (≈8 worst case).

**Revised gate G-RV′** (replaces PV's G-RV; needs a decision row):
- **No FEAS until all of these pass, or their failures are diagnosed under P5:** G0, G-RP (with the §3.6 reading), X0, X1–X3, M7 and X4.
- **Non-blocking:** M2-lite, X5 (but X5 must be done before FEAS outcomes are read), X6, X7 and the optional pilots.
- **Never gating:** M5/P3, M6/P4b and anything anchored on a figure or a different mixer family.

---

## 9. Comments on the proposed research practices

Most of PV §1 and PR §1 should be adopted: pre-register before running, swap one component at a time, base tolerances on variance, never tune to a number, report failures, keep a "repro" RNG firewall, project from the slowest family, hold a reproduction review. Amend as follows:

1. **"Original code first"**
   - Apply it only when the original artifact is in our causal chain, or is the only way to attribute a mismatch.
   - Re-running nanoGPT, modded-nanogpt or RULER validates tools we do not use.
   - When the original needs a shim (flash_attn → SDPA, `head_first`, micro-batching), call the result "original weights + modified code" and validate each shim on its own.
2. **Dominance rule (new).** If the question is "same function?", use a deterministic transplant or reference test, never a statistical training comparison.
3. **Minimum detectable defect (new).** Every tolerance states the smallest defect it can detect in the live cell. Example: "±0.03 overall = ±0.21 on 1024/256".
4. **Anchor-transfer statement (new).** Each target names its mechanism family, scale and metric. A mismatch makes the item *exploratory* and it may not gate. This would have caught M5/P3 and P2.
5. **"Tolerances from seed variance"** applies only where seeds can differ. Variance at ceiling is compressed and validates nothing (PR S5).
6. **Decision-relevance (new).** Write down which outcome changes which decision. Items with none (P2(b-iii) under I-23, M6 given an unconditional X-gc12) are demos.
7. **Environment integrity (new).**
   - Install reproduction tooling (lm-eval, nltk, RULER data) in a separate venv.
   - Do not change the frozen `.venv` or `uv.lock` before G5 without a decision row.
   - lm-eval at `d6de816` pulls in datasets, evaluate, scikit-learn and others [code-inspected `pyproject.toml`].
8. **Log deviations when they are made.** The S1 micro-batch shim lives only in a code comment and the run JSONs.
9. **GPU health is part of every result (new).** Record Xid and watchdog events per run, and flag any run that overlaps one.

---

## 10. Rows to log in `decisions.md` now (for the PI)

1. **S1 failure:** `gated_delta_net-d128-lr1.0e-02` failed permanently (2 × `cudaErrorLaunchTimeout`). Kernel log: Xid 8 with the RC watchdog at 18:16:50 and 18:25:42, and Xid 31 at 18:03:09. `gated_delta_net-d128-lr3.2e-02` diverged (0.005). G-RP(a) best-over-LR at d128 uses 3 LRs. Open X0. Apply the 2-best-LR cut to d256 ({1e-3, 3.2e-3}).
2. **S1 deviation:** exact micro-batch-64 gradient-accumulation shim (`zoology_compat._patch_microbatch`); only the dropout RNG stream differs. Also the `head_first` shim (already in `s1_reproduction.md` §3).
3. **S1 note:** the "Need to install causal_conv1d" message comes from Zoology's `_compute_state_size` probing the Mamba modules and has no effect on the runs. Replace R-S1-4 with X1.
4. **S1 interpretation, declared before d256 results are read:** the per-cell report; the "FAIL at ceiling (uninformative)" reading for monotonicity (§3.6); S5 dropped in favour of M7's seeds.
5. **S1(b) difference list addendum:** Zoology's GDN casts q, k, v, β and g to bf16; our fp32 arms run the fp32 reference attention kernel. Our arms should also run under bf16 autocast.
6. **Annotate the protocol §2 prior** (no design change): Wang's released "3-1" has attention in 8 of 24 layers (⅓), while the paper text says r:1 and "one quarter". Attention head count changes with ratio. The training context is 2,048 (v1) or 4,096 (v2).
7. **Reproduction tooling** goes in a separate venv; the frozen `.venv` is unchanged.

---

## 11. Sources opened for this critique

| Source | Used for | Access |
|---|---|---|
| [arXiv 2507.06457v2](https://arxiv.org/html/2507.06457v2) and [v1](https://arxiv.org/html/2507.06457v1): Tables 2, 3, 5, 6, 7; §3.4; §4.2–4.3; App. D; Limitations | §1, F5, F6 | read-section (raw HTML text) |
| HF `config.json`: `m-a-p/340M-20B-GatedDeltaNet-{hybrid-3-1, hybrid-6-1, pure-baseline}`, `1.3B-100B-GatedDeltaNet-{hybrid-3-1, hybrid-24-1}`, `transformer_{340M, 1.3B}_baseline` | §1 ratio and geometry facts | code-inspected (curl, 18:2x) |
| [arXiv 2405.21060v1](https://arxiv.org/html/2405.21060v1) §9.1 and §9.2.3, Table 2 | §1, §5.2 | read-section |
| [Olsson et al. 2022](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html): Key Concepts, Argument 1, Argument 2, Model Analysis details | §1, §5.3 | read-section |
| [arXiv 2304.01373v2](https://arxiv.org/abs/2304.01373) App. G.1 | §1 | read-section (PDF text) |
| [HF gpt2 discussion #90](https://huggingface.co/openai-community/gpt2/discussions/90) | §1 | read-section (WebFetch) |
| Zoology @1ad20d1: `model.py` L225–290, `mixers/{mamba, mamba2, gated_delta_net}.py`, `train.py`, README figure | F2–F4, §3 | code-inspected |
| fla 0.6.0 (installed): `layers/attn.py`, `modules/causal_conv1d/ops.py`, `models/modeling_utils.py`, `models/gated_deltanet/modeling_gated_deltanet.py` | F3, M2 feasibility, head_dim | code-inspected |
| `src/hybridlc/models/{attention, linear}.py`; `tests/` listing and test names; `scripts/s1_zoology/{zoology_compat, run_one, our_mixers}.py` | F4, F7, §6 | code-inspected (read only) |
| `third_party/ruler` @db43f92 (`niah.py`, `synthetic.yaml`, `json/`); `third_party/lm-evaluation-harness` @d6de816 `pyproject.toml`; nanoGPT README and `train.py` | P2 and M2 feasibility, M4 | code-inspected |
| `experiments/s1_zoology/{runs/*.json, logs/*, cut_decision.json}`; `journalctl -k`; `nvidia-smi -q` | F1, F2 | MEAS-S1 / KLOG (read only) |
| Project documents (protocol, competing explanations, INTERFACES, decisions, `s1_reproduction`, both proposals, `literature_review` (relevant parts), `03_evaluation` and `05_implementation` by grep) | Context | read-full (proposals, protocol sections used) / read-section |

## 12. Not verified by me

- The root cause of the Xid 8 and Xid 31 events, and which process caused the Xid 31.
- Whether Wang's checkpoints load cleanly in fla 0.6.0. I checked only that the pure model needs no flash_attn.
- Kimi, Bae and Waleffe numbers. I did not open them; they are cited from PR and the literature review. My Kimi point uses PR's own quoted row.
- The GDN paper's S-NIAH values and Press et al.'s Table 2. I did not open them; I do not rely on them.
- The context length of any candidate model for X5. It must be checked against the model card before use.
- The exact d = 64 accuracy band. The figure suggests mid-range (approximate).

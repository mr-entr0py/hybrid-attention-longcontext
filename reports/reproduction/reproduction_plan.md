# Reproduction programme: PI decision (v1.0)

**Role:** principal investigator. **Date:** 2026-11-09, about 22:50 IST.
**Status:** DECIDED. This file is the pre-registration of every item below. Its SHA-256 is recorded in decision row DEV-01 (`reports/decisions.md`). An item's criteria may be changed before it runs only by a new decision row, and never after it has run. The protocol text that this plan amends is in `reports/reproduction/protocol_amendment.md` (DEV-01).

**Inputs read in full:**
- both proposals (`proposal_pipeline-validation.md` = **PV**, `proposal_phenomenon-replication.md` = **PR**) and both critiques (`critique_A.md` = **A**, `critique_B.md` = **B**, revision 2);
- protocol v1.0 (§1–§20), `decisions.md` (D000–D044, D001a), `competing_explanations.md`, `src/hybridlc/INTERFACES.md` (including the integrator's 22:4x additions), `literature_review.md`;
- `s1_reproduction.md` (22:28 version) and `experiments/s1_zoology/*`; the S0 benchmark `runs/bench_train_step/*.jsonl`; the `tests/` listing (VT-03 now exists); the kernel log (`journalctl -k`); `nvidia-smi`.

**Constraints kept.** I ran no GPU job. I edited nothing in `src/`, `tests/`, `scripts/`, `configs/` or `data/`. Every target number below was re-opened by me in this step from the source cited (§8).

**Tags.**
- Access: [read-section], [read-full], [code-inspected]. **PI-VER** = opened by me in this step. "per X" = taken from document X and not re-opened by me.
- (fig, approx.) = read off a figure. Such values are never thresholds (protocol I-20).
- [MEAS-S1] = from `experiments/s1_zoology/`. [MEAS-S0] = from `runs/bench_train_step/` (contended). [KLOG] = kernel log.
- [INF] = my inference, design choice or arithmetic.

---

## 0. Decision on one screen

**What the user asked for.** Reproduce important published results, using original code where it helps, so that our setup and methodology are shown to agree with the literature before our own experiments start.

**What I decided.** The programme has two kinds of reproduction:
- **Literature reproductions.** Each must be in our causal chain, able to fail, and have an anchor that transfers to our setting:
  - Zoology MQAR, run as a ladder: "their code, their mixer" (R1) → an exact weight-copy check (R3a) → "their harness, our mixer" (R4, R5) → "our code, our task" (R6);
  - Pythia and Wang et al. LAMBADA numbers, scored by our scorer;
  - Olsson et al.'s induction-head signature, measured in our models;
  - the Mamba-2 / Waleffe ratio-vs-loss pattern, reported descriptively.
- **Reference reproductions.** Deterministic weight-copy and reference-implementation tests for every "is ours the same function?" question. A training comparison cannot answer that question sharply (A §3.3, B §3.2c).

| ID | Reproduction | Grade | Pass criterion (short) | New GPU-h | Gate role | When |
|---|---|---|---|---|---|---|
| R1 | Zoology MQAR with Zoology's own models (S1(a)) | smoke at ceiling | G-RP(a) as frozen: **PASS, 16/16** | 0 (≈ 1.75 spent) | G-RP(a) ✔ | done |
| R2 | Environment integrity: one GPU queue, per-run Xid/OOM tagging, 32K kernel profile, clean G0 throughput | environment | tagging live before R4; longest kernel < 1 s | ≈ 0.1 | G-RV | now |
| R3 | Reference-reproduction battery VT-21…VT-25: fla conv vs independent reference; C12↔Qwen3 and C0↔fla LM at 4K–32K; trainer vs independent loop; oracle answerability of 16K/32K items; Zoology↔ours mixer transplant | reference reproduction | per-test tolerances (from 1e-2 at bf16 down to 1e-6; EM = 1) | ≈ 0.12 | G-RV | before R4–R6 |
| R4 | S1(b): our mixers in Zoology's harness, plus a production-F_g smoke run (fp32 and bf16) | smoke | G-RP(b) ±0.03 as frozen, reported per cell | 0.6–0.9 | G-RP(b) | after R3a |
| R5 | MQAR at d_model = 64: Zoology's GDN vs ours under bf16 autocast, 3 seeds each | phenomenon + calibrated comparison | 1024/256 cell: d64 < d128 < d256; \|Δ(ours − Zoology)\| ≤ max(0.03, 2.78·SE) | 1.3–1.7 | G-RV | after R4 |
| R6 | Known-answer associative recall in **our** pipeline: 2-layer `AA` vs `--`, TEFStream → trainer → scorer | phenomenon (Zoology in our harness) + chain consistency | `AA` EM_tf ≥ 0.90 (R1 D = 3, R-load N = 8, 1K) within ≤ 150M tokens; scorer vs training forward flip rate < 1% | 0.3–0.6 | G-RV | after G0 |
| R7 | LAMBADA through our scorer: Pythia-160M and 70M (published table); Wang 340M GDN-pure through fla 0.6.0 | numeric reproduction | 0.328 ± 0.015; 0.185 ± 0.015; 0.348 ± 0.02 | ≈ 0.1 | must finish before any FEAS verdict | before FEAS |
| R8 | Induction heads / ICL score (Olsson) on FEAS Stage-A milestones and S3 finals | phenomenon | k = 12: repeated-random-token copy (second-copy loss ≤ 1.0 nat and ≤ 0.2 × first-copy loss); ICL ≥ 0.15 nats; plateau | ≈ 0.1 | audit trigger | during FEAS and S3 |
| R9 | Ratio-vs-loss shape (Mamba-2 Table 2; Waleffe §4.1) on S3 plus one k = 1 run at η*; RP-2 LM-3 clause | phenomenon, descriptive | shape class reported; LM-3: k = 12 below k = 0 at Δ ∈ [1K, 4K) | 0.65–0.7 | audit trigger (LM-3 clause only) | after S3, before G5 |
| R10 | Length behaviour of FEAS checkpoints (RoPE beyond 4K; long positions after Stage B) | descriptive (should tier) | within-window monotone; post-Stage-B long-position check | ≈ 0.3 | audit trigger | during FEAS |

**Totals [INF].**
- New must-tier work (R2–R9): **≈ 3.3–4.3 GPU-h**. Should tier (R10): ≈ 0.3. Already spent (R1): ≈ 1.75.
- **Programme total: ≈ 5.3–6.4 GPU-h**, against the 3.0 h S1 line of v1.0, so Tier A grows by about 3.4 h.
- Optional items run only on a pre-declared trigger, ≤ 4.7 h in all (§2.11).
- For comparison, the re-priced must tiers were PV ≈ 10–12 h and PR ≈ 11–15 h (B §1.3).

**Order relative to S2 FEAS.**
1. Before FEAS: R2 → R3 → R4 → R5 → R6 → R7. **Gate G-RV** then opens FEAS.
2. During FEAS: R8 and R10 run on FEAS checkpoints. R8's induction-timing reading must be in before S3 starts.
3. After S3: R9.
4. A written reproduction review comes before G5.

**Rulings on the disagreements** (§6 has the full table):
- **M7 → R5 gates FEAS** (A over B). Three reasons:
  - it completes the Zoology state-size reproduction with a point below ceiling;
  - it gives G-RP(b)'s ±0.03 a measured seed basis;
  - it tests GDN at production precision (bf16), where a handicap would bias every k = 0 baseline.
- **X4 → R6 gates FEAS** (A over B). A train/eval format bug would otherwise surface as "FEAS fails at 32K". Under the U-01 rule that is misread as a scale failure, and D512 costs about 58 GPU-h more.
- **Lean environment hygiene now; A's soak (X0) only if an Xid recurs** (B over A).
- **LAMBADA through our own scorer against the published tables.** lm-eval is used only as a diagnostic in a separate venv (B).
- **One k = 1 LM run** (B), read descriptively.
- **Answerability: the oracle test first** (B). A public long-context model is used only if FEAS-1 fails at 32K.

---

## 1. Principles and practices for every reproduction in this project

These are standing rules. In brackets: where each came from, and the PI's amendment where there is one.

1. **Decision relevance first** (A §2). Before adopting an item, write down which outcome would change which gate or decision, and what the change would be. An item where no outcome changes anything is a demonstration, not a validation; it is not run as part of the programme.
2. **Grade every check, and state what it can detect** (PR §1.3, A §9.3, B §7.3). Each item is one of the following:
   - *reference reproduction*: deterministic equivalence with an original implementation or an independent reference;
   - *numeric reproduction*: a published number within a stated tolerance;
   - *phenomenon replication*: direction or shape at our scale;
   - *smoke*: catches only gross failure;
   - *environment*.

   Each item also states its minimum detectable defect (MDD) and, where it can be computed, its false-alarm rate. Example: G-RP(b)'s ±0.03 on overall accuracy equals ±0.21 on the only cell that discriminates.
3. **Anchor transfer** (A §9.4). Each target names its mechanism family, scale and metric. If the anchor does not match our setting, the item is exploratory and may not gate. For instance, a Mamba-family loss pattern is not a delta-rule anchor.
4. **Original code first, but only where it is in our chain** (PV P1, PR §1.2, amended by A §9.1).
   - Run the original artefact when it is in our causal chain, or when it is the only way to attribute a mismatch (Zoology).
   - Then swap one component at a time (PV P2).
   - A result produced through a shim is labelled "original code + N shims", and each shim is validated on its own.
5. **Identity questions are answered by weight-copy or reference tests, not by statistical training comparisons** (A §9.2, B §7.4). A training comparison is run only for a question that copied weights cannot answer, such as precision regime, seed spread or learnability.
6. **Pre-register before running** (PV P3, PR §1.1). For each item, record:
   - the claim, its source (URL, table or figure, access tag) and the metric definition;
   - the exact configuration;
   - the tolerance and how it was derived;
   - the PASS / FAIL / INCONCLUSIVE rule and the failure tree.

   This file does that, and its hash is in DEV-01. Implementers copy the item block into `experiments/repro/<R-id>/prereg.yaml` before the first run.
7. **Tolerances come from variance, never from taste or figures** (PV P3, PR §1.4–1.5).
   - Use published SEs or measured seed spread. A provisional tolerance is labelled as such.
   - Figure-read values are context only (I-20).
   - Variance measured at ceiling validates nothing (A §9.5), so no seed replicate is run at ceiling.
8. **Never tune to hit a number** (PV P5, PR §1.8).
   - A code change after a failed reproduction needs a defect isolated by a test, and a mechanism that is independent of the target number.
   - Every attempt is logged. "Another seed until it passes" is forbidden.
9. **Report everything** (PV P6, PR §1.9). FAIL, PARTIAL and INCONCLUSIVE results get the same prominence as passes, with their full history, in the ledger `reports/reproduction/results.md`.
10. **Firewall between reproduction and confirmatory work** (PR §1.11).
    - Reproduction items and documents use the RNG stream `repro` (entropy 31337), never the test generator (20261109) or the dev-gate generator (77).
    - Reproduction training runs use seed indices 300–399. The exceptions are fixed in §2.0.
    - No reproduction output is used for power or ceiling planning, or for tuning (I-23).
    - Reproductions change the protocol only through the pre-declared audit triggers and escalations in §4.
11. **Environment integrity** (A §9.7, B §6.6). No new package goes into the frozen `.venv` before G5. Diagnostics that need lm-eval or RULER use a separate `.venv-repro`, and their results are labelled as in rule 4.
12. **Feasibility pre-flight before adopting an item** (B §7.2). Check that:
    - the packages are present;
    - data files are present and not Git-LFS pointers;
    - checkpoint dtype and size fit the 0.44 memory cap;
    - the tokenizer is accepted where `TokenSpace` is needed;
    - there is no `flash_attn` dependence;
    - downloads fit on the disk.
13. **Price from measured throughput** (PR §1.10, B §7.1).
    - Use the slowest configuration at the real shape.
    - Treat early-stopping run times as bimodal.
    - Harness runs have a pre-declared collapse-abort rule (§2.0).
14. **GPU health is part of every result** (A §9.9, B §7.6).
    - One GPU queue for all agents.
    - Every run is tagged with the Xid, RC-watchdog and host-OOM events in its time window.
    - A gate run that overlaps such an event is repeated.
15. **Declare re-run and retry policy before any verdict is read** (B §7.5). Record it here and in the decision log, not only in a script.
16. **Evaluate public checkpoints inside their training window**, unless extrapolation is what is being studied (B §7.8).
17. **Written reproduction review before G5** (PR §1.12): `reports/reproduction/review.md`, with adversarial readers where available, ending in go / fix / re-scope.

---

## 2. The programme, in execution order

### 2.0 Conventions common to all items

- **RNG and seeds.**
  - The `repro` stream (entropy 31337) is used for every reproduction item file and for sampling documents.
  - Reproduction training seed indices are 300–399:
    - R6 uses 300;
    - optional O2 uses 301–304.
  - Exceptions, by design:
    - R9 uses seed index 100, so that it is paired with the S3 runs (same data order, same shared-module initialisation);
    - optional O3 uses 103;
    - Zoology runs use Zoology's own seeds: 123 in R1/R4, and 123–125 in R5.
- **Retry policy, declared now.**
  - A run that fails for an infrastructure reason (Xid, watchdog, OOM, host OOM) is resumed from its last checkpoint. Harness runs get at most 6 attempts with per-epoch resume, which is the committed S1 queue behaviour. Training runs follow the runner's policy (D031).
  - A gate run whose window contains an infrastructure event is repeated in full.
  - No run is repeated after its verdict has been read, except where an item's rule says so.
  - **Collapse-abort rule for harness runs:** stop when validation accuracy has been below 0.05 for 3 consecutive epochs after epoch 2, and record the run as "collapsed". This changes compute, not best-over-LR results. Two S1(a) runs at lr 3.2e-2 trained 32 collapsed epochs, 41% of completed-run time (B §1.1).
- **Outcomes.**
  - PASS or FAIL as defined per item.
  - INCONCLUSIVE when an item's pre-declared regime condition fails: ceiling or floor, bimodal seeds, or an exhausted budget extension. An INCONCLUSIVE result never blocks a gate, but it is discussed at the review.
- **Results** go to `experiments/repro/<R-id>/`, with raw per-item JSONL and the commit and config hash. The ledger row goes in `reports/reproduction/results.md` (template: PV §7).

### 2.1 R1: Zoology MQAR with Zoology's own models (S1(a)): DONE, closed out

- **Claim** (Arora et al., Zoology, [2312.04927](https://arxiv.org/abs/2312.04927), [read-section per PROP]; repository README figure).
  - Attention solves multi-query associative recall (MQAR) at small width.
  - Fixed-state recurrent mixers recall more as their state grows.
  - Gated DeltaNet is near the top of the efficient mixers.
- **Numbers.** These exist only in a figure: README "Recall-Memory Tradeoff", `zoology/analysis/paper/zoology_paper/zoology_results.png` @`1ad20d1`, which I viewed (PI-VER) (fig, approx.). The GDN points are:
  - ≈ 0.89 at a state of ≈ 1.7e4;
  - ≈ 0.99 at ≈ 6.6e4;
  - ≈ 0.995 at ≈ 2.6e5;
  - ≈ 1.0 at ≈ 1e6.
- **Text corroboration** for the state-size trend: Mamba-2 §9.1, "increasing from N = 16 to N = 64 and N = 256 consistently improves performance on MQAR" ([2405.21060v1](https://arxiv.org/html/2405.21060v1) [read-section, PI-VER]).
- **Result** [MEAS-S1]. G-RP(a) PASS, 16/16 cells, using best-over-LR final overall accuracy:

  | Model | Overall | 1024/256 cell | State |
  |---|---|---|---|
  | attention d64 | 1.0000 | — | — |
  | attention d128 | 0.9999 | — | — |
  | GDN d128 | 0.9892 | 0.925 | 67,072 |
  | GDN d256 | 0.9938 | 0.957 | 265,216 |

  - The two GDN points agree with the figure's ≈ 0.99 at ≈ 6.6e4 and ≈ 0.995 at ≈ 2.6e5 (fig, approx.).
  - GDN at lr 3.2e-2 diverged at both widths (overall 0.0050).
  - `gated_delta_net-d128-lr1.0e-02` completed on attempt 3 (0.9877), after two crashes with Xid 8 and the RC watchdog. The verdict is unchanged.
- **Grade:** smoke-level reproduction at ceiling.
  - Overall accuracy is the unweighted mean of 7 cells. Six cells are at ≥ 0.9996.
  - So ±0.03 overall equals ±0.21 on the 1024/256 cell (A F2, B §1.1). Doubling d moved that cell by only +0.032.
- **What it validates.** This environment (torch 2.11.0+cu128, triton 3.6.0, fla 0.6.0, sm_120) trains Zoology's reference models to the published qualitative behaviour, and fla's GDN kernels learn associative recall under training.
- **What it cannot validate.** Our mixers (R3a, R4, R5) and our pipeline (R6).
- **Code.** HazyResearch/zoology @`1ad20d193b6113cae1e8f3c655c300d7b4b3f4bb`, plus the shims in `scripts/s1_zoology/zoology_compat.py`:
  - `head_first` dropped for fla 0.6.0;
  - exact gradient accumulation with micro-batches of 64;
  - per-epoch resume, which no reported run used.

  Provenance is recorded in each run JSON.
- **GPU-h spent:** 1.48 h of completed runs [MEAS-S1], plus ≈ 0.24 h of crashed attempts and ≈ 0.05 h of smoke runs, ≈ **1.75** in total (B §1.1).
- **Close-out actions** (all in DEV-01):
  - log the three shims as deviations;
  - log the attempt-3 episode. That attempt was launched after the verdict by an uncommitted change to `run_queue.py`; the S1 owner must commit that change;
  - log the lr 3.2e-2 divergence;
  - log that the "Need to install causal_conv1d" message comes from Zoology's Mamba import probe (`model.py` L270–277) and has no effect on these runs (A F3);
  - record the forecasting lesson behind rule 13.

### 2.2 R2: environment integrity (A's A3 and B's G0-T; A's X0 only on trigger)

- **Facts** [KLOG; nvidia-smi, read 22:40 IST].
  - Xid 31 (MMU fault, copy engine CE4, virtual write) at 18:03:09 in pid 799092, not identified.
  - RC watchdog "GPU is probably locked! Notify Timeout Seconds: 7" plus Xid 8 at 18:16:50 and 18:25:42. These match the two failed S1 attempts.
  - No Xid events since then.
  - A host-RAM OOM kill at 17:31:22, which hit an IDE process.
  - The display is attached and active.
  - At 22:40 two project GPU processes were running at once (a VT-13 training run and a VT-03 pytest), next to CARLA: 12.6 GB used, 95% utilisation.
- **What it protects.** Without these checks:
  - an infrastructure failure could silently corrupt a gate run;
  - such failures could become missing data that correlates with k;
  - all of this would be invisible in the results.
- **Configuration.**
  - (a) **One GPU queue.** Every project GPU job (gpu-marked tests, benchmarks, Zoology runs, training, evaluation) goes through one queue. No two project GPU processes run during a timed or gate run. CARLA is the only foreign load allowed.
  - (b) **Event tagging.** Each run stores the `journalctl -k` lines matching `NVRM: Xid|RC watchdog|Out of memory|oom-kill` between its start and end, as `infra_events` in the run JSON and the registry. A run with any such line is flagged.
  - (c) **Kernel profile.** Run `torch.profiler` (CUDA activities) over one Stage-B 1 × 32K training step (forward and backward with activation checkpointing) and one 32K prefill, for C0 and C12 at D384 (and at D512 if FEAS-512 is reached). Report the longest single kernel.
  - (d) Try to identify pid 799092 from the agents' logs, and record the outcome.
  - (e) **G0-T.** Measure clean throughput with CARLA as the only load (`scripts/g0_throughput.py`, owned by S0). Then re-price D041 and the LM-training items R6 and R9.
- **Pass.**
  - (a) and (b) are live before R4 starts.
  - (c): the longest kernel is under 1 s. That is ≥ 7× below the 7 s RC-watchdog notify timeout in the kernel log [INF].
  - (d) and (e) are recorded.
- **GPU-h:** ≈ 0.1 for (c); (e) is inside S0's 3.0 h.
- **Owners:** orchestrator (a); S0 training/runner (b), (c), (e); PI (d).
- **Failure tree.**
  - A kernel of ≥ 1 s: identify the op, then choose among a smaller tile or chunk (configuration only), a split prefill, or accepting it with a risk note. Any of these needs a PI decision row before FEAS.
  - **Trigger O4** (A's X0 soak): any new Xid (8, 13, 31, 43, 79) or RC-watchdog event that overlaps a project process. Run O4 before any further gate run.
  - Repeated hangs at one configuration: rerun it alone with `CUDA_LAUNCH_BLOCKING=1` and bisect to the kernel.
  - A host OOM: the runner enforces host-RAM headroom.

### 2.3 R3: reference-reproduction battery (VT-21…VT-25; gate G-RV)

These reproduce an original implementation's, or an independent reference's, outputs at the configuration we actually use. Their IDs extend protocol §16, but they gate **G-RV, not G0**, so the S0 gate does not change under the builders' feet. GPU total ≈ 0.12 h.

#### R3a / VT-25: Zoology ↔ our mixers, by weight copy (A's and B's X2)

- **Reference.** Zoology @`1ad20d1`:
  - `zoology/mixers/gated_delta_net.py` casts q, k, v, β and g to bf16 at L246–250 and the output back to fp32 at L295 [code-inspected, PI-VER];
  - `zoology/mixers/attention.py` MHA: `Wqkv` with `bias=True`, an `out_proj` with bias, attention dropout;
  - `model.py` L249 re-initialises every module with `self.apply(_init_weights)`.
- **Ours.** `scripts/s1_zoology/our_mixers.py` (`OurGdnZoology`, `OurFgZoology`) → `hybridlc.models.mixers.build_mixer`.
- **Configuration.**
  - Run inside Zoology's 2-layer `LanguageModel` (BaseConv + mixer), with GDN at d ∈ {64, 128, 256} and attention at d ∈ {64, 128}.
  - Copy weights from Zoology to ours with an explicit name map.
  - Inputs: one MQAR batch (B = 8, T = 256) and one random batch.
  - For the attention comparison, zero Zoology's MHA biases and switch dropout off. These are intended structural differences, which no training comparison could separate from bugs.
- **Pass.**
  - GDN at matched precision (ours with Zoology's bf16 casts, reproduced through a test-only flag): logits relative RMS ≤ 1e-2; parameter gradients relative RMS ≤ 2e-2. This is the VT-04/VT-05 bf16 level.
  - GDN, ours in fp32 vs Zoology: reported only; the expected error is at bf16 rounding level.
  - Attention in fp32: logits relative RMS ≤ 1e-5; gradients ≤ 1e-4.
- **Tolerance basis** [INF]. Both paths call the same fla kernel on bf16 inputs. But our layer computes the decay gate inside the kernel (`s1_reproduction.md` §6), while Zoology computes it in fp32 and then casts it. The two can therefore differ by bf16 rounding (2⁻⁸ ≈ 4e-3 relative per element), accumulated over the sequence. That bounds the matched-precision difference at the bf16 level of 1e-2. The defects this targets give O(1e-1) errors: gate formula, β, q/k L2-norm, conv orientation, head order, scale.
- **Owner:** S1. **Depends on:** `our_mixers.py` (exists).
- **Failure tree.**
  - GDN: name map and shapes → conv weight orientation and causal padding → g = −exp(A_log)·softplus(a + dt_bias) → β = σ(b) → q/k L2-norm and 1/√d_k → output norm on the `use_gate=False` path → head split order.
  - Attention: `Wqkv` packing order → softmax scale → causal mask → eval mode.

#### R3b / VT-21: fla's Triton `causal_conv1d` vs an independent reference (A's and B's X1)

- **Reference.** `hybridlc.models.attention.causal_dwconv_reference` (`attention.py` L94; pure PyTorch, independent of fla), plus `F.conv1d` with K − 1 left padding.
- **Under test.** fla 0.6.0 `fla/modules/causal_conv1d/ops.py::causal_conv1d`. It is used by GDN-A, GDN-E and GLA (`models/linear.py`), and by F_gc (`models/attention.py` L164–172).
- **Why it is needed.** The new VT-03 compares packed with unpacked inputs using **the same** fla kernel on both sides, so it cannot catch a conv that is wrong but consistent. VT-09(2) compares fla with fla.
- **Configuration.**
  - Channel counts of GDN-A's q/k (H·64) and v (H·128) paths and F_gc's 3d, at D384 and D512.
  - T ∈ {37, 300, 4,096, 32,768}.
  - Packed `cu_seqlens` with offsets that are not multiples of 64.
  - `initial_state` on and off; SiLU on and off; fp32 and bf16; forward and backward.
- **Pass.**
  - fp32: max absolute difference ≤ 1e-5 in the forward pass, and ≤ 1e-4 for the input and weight gradients.
  - bf16: relative RMS ≤ 1e-2.
  - Cross-document leakage is exactly zero.
  - The final conv state matches the reference to the same tolerances.
- **Tolerance basis** [INF]. A 4-tap depthwise conv of O(1) values has fp32 error of about 1e-7, while a reversed tap or an off-by-one shift gives O(1) errors. The bf16 threshold is the VT-04/VT-05 bf16 level.
- **Owner:** S0 models.
- **Failure tree.** Forward mismatch → tap orientation, padding, activation placement. Leakage → `seq_idx`/`cu_seqlens` handling. State mismatch → cache layout. Backward only → the kernel's backward pass. If fla itself is wrong, route the conv through the reference implementation at a speed cost; that needs a decision row.

#### R3c / VT-22: full-model transplant at the lengths we evaluate (B's A2)

- **Reference.**
  - transformers 5.22.2 `Qwen3ForCausalLM` with the VT-09(1) mapping: gate bypassed, gains tied, θ = 5e5.
  - fla 0.6.0 `GatedDeltaNetForCausalLM` with the VT-09(2) mapping.
- **Configuration.**
  - D384 C12 vs Qwen3, and C0 vs the fla LM, at:
    - T = 4,096 in fp32;
    - T ∈ {16,384, 32,768} in bf16. Ours runs on the production varlen path; the references use SDPA-flash bf16 and the fla chunk path in bf16.
  - Logits are compared at every 16th position and at the last 512 positions, to save memory.
  - C3 packed (16K + 16K) vs each document alone, at 32K in bf16.
- **Pass.**
  - 4K fp32: relative max ≤ 1e-3 (the VT-09 threshold).
  - 16K/32K bf16: relative RMS ≤ 2e-2, **and** the error is flat across position bands. Flat means the RMS error over the last 1K positions is ≤ 2× that over the first 1K.
  - There is no argmax criterion. The weights are random, as in VT-09, so the logits are near-uniform and argmax ties are ill-conditioned.
  - C3 packed vs alone: relative RMS ≤ 1e-2 (the VT-03 bf16 attention tolerance).
- **Basis** [INF]. The existing VT-09 runs only at T ≤ 256, while the primary cell is 16K–32K. A RoPE position or precision bug, or a state-accumulation bug, grows with position. The band criterion catches it even when the global RMS is still within tolerance. VT-04 uses the same "flat across position bands" idea for the kernel.
- **Owner:** S0 models.
- **Failure tree.** Error grows with position:
  - for C12 → RoPE position dtype, cos/sin cache length, rotate-half convention, θ;
  - for C0 → chunk-state accumulation, `max_seqlen`.

  A packed mismatch → `cu_seqlens` or the position reset at the 16K boundary.

#### R3d / VT-23: trainer vs an independent reference loop (B's T1, A's X3)

- **Reference.** A loop of about 60 lines written from the text of protocol §7.1–§7.2 only, without importing `training/optim.py` or `schedules.py`:
  - AdamW (0.9, 0.95, 1e-8, fused);
  - weight decay 0.1 on parameters with ndim ≥ 2, including the tied embedding; none on 1-D tensors, `A_log` or `dt_bias`;
  - global-L2 gradient clipping at 1.0;
  - z-loss 1e-4·mean(lse²);
  - mean CE over non-ignored tokens;
  - exact full-step accumulation.
- **Configuration.**
  - D384 C3, so both mixers are covered, in fp32 and deterministic audit mode (`use_deterministic_algorithms`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`).
  - Identical batches from `StageStream('A', data_seed=<an S0 audit index 900–902>)`.
  - 2 × 4K micro-batches with 2 accumulation steps, for 50 steps.
  - Schedule: 10 warm-up steps, then constant.
- **Pass.**
  - Step-1 loss: relative difference ≤ 1e-6.
  - Steps 2–10: per-step loss and pre-clip gradient norm within 1e-5 relative. Steps 11–50: within 1e-3 relative.
  - At least one step with a pre-clip norm above 1.0, so that clipping is exercised.
  - Final parameters: relative L2 difference ≤ 1e-3.
  - Identical weight-decay group membership by name.
- **Basis** [INF]. Same kernels in deterministic fp32, so differences come only from summation order (about 1e-7 per op). Training dynamics amplify such differences over many steps, so the tight bound applies only to the first 10 steps and a looser trace bound after that. Known defect classes deviate by ≥ 1e-4 within 10 steps:
  - a z-loss coefficient 10× too large changes the initial objective by about 1%;
  - a one-step LR offset during warm-up changes the step-1 LR by 10%;
  - clipping in the wrong place changes the first clipped update.

  VT-13 checks components and config values, but not the end-to-end update against an independent loop.
- **Owner:** S0 training.
- **Failure tree.** Mismatch at step 1 → loss assembly (W_m/W weights, z-loss) and label shift. Later drift → parameter groups, schedule indexing, where clipping happens (after accumulation, global over all parameters), accumulation normalisation.

#### R3e / VT-24: oracle answerability at the confirmatory lengths (B's O1)

- **Reference.** `hybridlc.eval.devtools.InductionLM(order=6, p_hit=0.9)`, an exact n-gram copy model. VT-12 already uses it at 2K and asserts logp_gold = 4·ln 0.9 [code-inspected, PI-VER, `test_vt12_items.py` L410].
- **Configuration.** Score every R1 (D ∈ {3, 15}), R-load (N = 8 … 1,024, and N = 2,048 at 32K), R3 and R-dist item file at 16K and 32K in `experiments/items/test/CS`, `test/MS`, `dev/DG` and `dev/FD`, through `eval.scorer.score_file` on CPU. M-hop files serve as a negative control.
- **Pass.**
  - Every retrieval query: EM_tf = 1 and logp_gold = 4·ln 0.9.
  - M-hop: EM_tf < 0.5, because the oracle cannot compose hops.
- **Basis.** The test is deterministic, so any miss is an item defect: a key n-gram colliding with filler, damage from needle snapping, or a misplaced prompt/answer boundary. Reading frozen item files with a model-free oracle leaks nothing about model performance.
- **GPU-h:** 0 (minutes of CPU). **Owner:** S0 evaluation.
- **Failure tree.** Inspect each miss. A collision → the generator re-samples filler spans that contain a key n-gram. Snapping damage → fix the snapping. Regenerate the item files before G5 (new manifest, VT-12 rerun) and log a decision row.

### 2.4 R4: S1(b), our mixers in Zoology's harness, plus a production-F_g smoke run

- **Claim.** As R1: our mixers, configured to Zoology's geometry, should behave like Zoology's.
- **Validates.** That our mixer *modules* train as Zoology's do inside the original harness. This is the second rung of the ladder.
- **Grade:** smoke, with an MDD of 0.21 on the live cell. The deterministic identity question is answered by R3a.
- **Code.** As R1, plus `our_mixers.py` (`part_b_runs(best_lr_from_s1a())`).
- **Configuration.** The frozen part (b) runs, in fp32 with no autocast, as v1.0 specifies:
  - `our_fg-d64-lr1.0e-02`
  - `our_fg-d128-lr1.0e-03`
  - `our_gdn-d128-lr1.0e-03`
  - `our_gdn-d256-lr1.0e-03`
  - exploratory: `our_gdn_configA-d128-lr1.0e-03` and `our_gdn_configA-d256-lr1.0e-03`

  Plus **R4b**: our *production* F_g (QK-RMSNorm, RoPE θ = 5e5, headwise gate; 2 heads × 64) at d = 128, lr 1e-3, once in fp32 and once under bf16 autocast. The bf16 run exercises the `varlen_attn` (FA2) kernel used in training (A §3.5).
- **Pass.**
  - **G-RP(b) as frozen:** ours within ±0.03 of Zoology on overall accuracy at matched d and LR. It is reported per cell, with its MDD, and only after R3a has passed.
  - R4b (smoke): mean of the in-distribution cells (64/4, 64/8, 64/16, 128/32, 256/64) ≥ 0.95 in both precisions. The 512/128 and 1024/256 cells lie beyond the 256-token training length with RoPE, so they are descriptive.
  - The config-A arms are descriptive. GDN-A at d = 256 has the same per-layer state (16,384) as Zoology's GDN at d = 128, which scored 0.925 on the 1024/256 cell (A §3.5, B §3.1).
- **GPU-h [INF].** About 0.6–0.9: three GDN runs that will probably not early-stop (≈ 13–17 min each, from the S1(a) d128 timings), plus short attention and smoke runs.
- **Depends on:** R3a; a `run_one.py` / `run_queue.py` extension to look up `part_b_runs` (S1 owner).
- **Failure tree.**
  - G-RP(b) fails while R3a passes → the remaining differences: init (the `out_proj` alias, fla-tensor inits), Zoology's optimiser groups, fp32 vs bf16 casts, the missing attention dropout. Swap one at a time; for example, an "ours with Zoology's casts" arm through the test flag.
  - R4b fails → RoPE, QK-norm temperature at small d, gate init; read the attention max-logit and entropy statistics. **Do not change F_g.** Log the failure; it raises the prior on E1.2/E2.7 and goes to the review.

### 2.5 R5: discriminating MQAR at d_model = 64 (PV's M7, with A §5.7's changes)

- **Claim.**
  - The recall–state trade-off has a GDN point below ceiling: ≈ 0.89 at ≈ 1.7e4 state (fig, approx.; viewed by the PI).
  - At d = 64, Zoology's GDN state should be about a quarter of d = 128's 67,072, i.e. ≈ 1.7e4 [INF]. The figure's points presumably come from this same configuration file (not verified).
  - Our S1(a) re-runs already match the figure at the two higher states (R1).
- **Validates.**
  - (i) It completes the Zoology reproduction in the regime where the trade-off is visible, which ceiling cells cannot show.
  - (ii) It gives G-RP(b)'s ±0.03 a measured seed basis.
  - (iii) It tests whether our GDN at **production precision** (bf16 autocast, i.e. bf16 projections and kernel inputs with fp32 state, §7.1) learns recall as Zoology's does in a capacity-limited regime. A precision handicap would bias every k = 0 baseline, and therefore FEAS-2 and every recovery estimate, in favour of attention.
- **Grade:** phenomenon (state dose-response) plus a calibrated comparison.
- **Code.** As R4.
- **Configuration.**
  - d = 64, lr 1e-3 (the best S1(a) LR at both other widths; not best-over-LR at d = 64; reported as such), 32 epochs, early stopping as in the file.
  - Seeds (`seed`, `data_seed`) ∈ {(123, 123), (124, 124), (125, 125)}.
  - Arms: Zoology's GDN (2 heads, `use_gate=False`, conv 4, `expand_v` 2), and `OurGdnZoology` (same geometry) with its forward wrapped in `torch.autocast('cuda', torch.bfloat16)` and the output cast to fp32.
  - 6 runs; micro-batch shim and collapse-abort rule as in §2.0.
- **Pass / fail / inconclusive** (primary cell = 1024/256; 512/128 and overall are secondary and descriptive):
  1. **Regime check.** Zoology's mean on the primary cell must lie in [0.05, 0.95]. If it is above 0.95, the primary cell becomes the lowest-accuracy cell among the seven, labelled "low power". If it is below 0.05, it becomes 512/128. No new test configurations are added.
  2. **Phenomenon (directional; re-run numbers only, I-20):** Zoology d64 mean < 0.925 (d128) < 0.957 (d256) on the primary cell. Zoology's d64 overall accuracy is reported next to the figure's ≈ 0.89 as context only.
  3. **Comparison.** Δ = mean_ours − mean_Zoology. **FAIL iff |Δ| > max(0.03, 2.78·SE_W)**, where SE_W = √(s²_ours/3 + s²_Zoology/3) and 2.78 = t₀.₉₇₅ at df ≈ 4 (Welch). Otherwise PASS. The 90% CI of Δ is also reported.
  4. **Bimodality.** If either arm's three seeds span more than 0.2 on the primary cell, add 2 seeds per arm (pre-declared, 4 runs). The comparison then becomes descriptive (INCONCLUSIVE), with no verdict on means.
- **Tolerance basis.** The 0.03 materiality floor is G-RP(b)'s own threshold. The t-term makes the test reject only differences that both matter and are distinguishable from seed noise. With a single primary cell, the false-alarm rate is ≤ 5%. The MDD is reported once the seed SD is measured.
- **GPU-h [INF].** 6 runs × 13–17 min, from the S1(a) d128 32-epoch timings (993.5 s with contention; 775 s on attempt 3) → **1.3–1.7**. The bimodality rule adds ≈ 0.9–1.1.
- **Depends on:** R3a passing.
- **Failure tree** (FAIL while R3a passes):
  1. Rerun ours with Zoology-style casts (test flag, 3 seeds). If the gap closes, bf16 projections change GDN's capacity. This is a property of "GDN-A as built" (D040 package wording), not a bug. Report it, add a caveat to how FEAS-2 and k = 0 are read, and raise it at the review.
  2. If the gap persists → compare initialisation statistics (`A_log`, `dt_bias`, conv) and the optimiser groups across arms.
  3. If still unexplained → FEAS is blocked, and the PI decides.

### 2.6 R6: known-answer associative recall in our own pipeline (A's X4)

- **Claim (phenomenon).** Zoology's result, re-run here (R1): two-layer models with attention learn multi-query associative recall within 1–3 epochs [MEAS-S1]. One MQAR epoch is about 24.3M tokens [INF from the training configs in R1]. Fixed-state mixers lose accuracy as the load grows. R6 replicates this inside *our* harness: our TEF item format, `TEFStream`, trainer, model and scorer. This is the third rung of the ladder:

  | Rung | Harness | Mixer | Task |
  |---|---|---|---|
  | R1 | Zoology's | Zoology's | Zoology's |
  | R4/R5 | Zoology's | ours | Zoology's |
  | **R6** | **ours** | **ours** | **ours** |
- **Validates.** The chain synthetic stream → answer mask and ½/½ loss → model → scorer, end to end, with a known answer, before FEAS spends about 10 GPU-h. It also checks that the scorer agrees with the training forward pass on a *trained* model.
- **Decision relevance.** A train/eval format mismatch, such as a whitespace token, template, query appending vs branching, or the answer-mask offset, would surface as "k = 12 fails at every length" in FEAS. Rule U-01 would read that as a scale failure: D512, about +58 GPU-h.
- **Code.** `hybridlc` at the G0 freeze-candidate commit (recorded per run): `data/tef.py` (TEFStream), `training/trainer.py`, `eval/scorer.py`, `scripts/evaluate.py`.
- **Configuration.**
  - Model: `ModelConfig.for_scale('D384', pattern='AA', n_layers=2)` and the same with `'--'`, built by `build_model(cfg, seed_index=300)`.
  - Data: `TEFStream(seed_index=300, seq_len=32768, micro_batch=1, accum=2, synthetic_share=1.0, buckets=(1024,))`. That is 65,536 tokens per step; 50M tokens = 763 steps. The TEF loss weighting (D014) is used.
  - AdamW as §7.1; schedule: 100-step warm-up to 1e-3, then constant. TEF's re-warm to 3e-4 assumes a pretrained start, which this model does not have.
  - Evaluation items: `repro` stream, namespace `dev` (word list B, template T1, dev filler), T = 1,024:
    - R1 D ∈ {3, 15} and R-load N ∈ {8, 32}, plus their C0 no-filler versions;
    - 200 contexts × 4 queries per cell, scored on the production cache path in bf16.
- **Pass.**
  - (a) **Chain consistency:** on the trained `AA` checkpoint, the EM_tf flip rate between the scorer (cache path) and the single-pass forward (`ForwardOnlyAdapter`) is < 1% (the I-29 threshold).
  - (b) **Known answer:** mean EM_tf of `AA` over {R1 D = 3, R-load N = 8} at 1K ≥ 0.90 after 50M tokens.
    - If 0.5 ≤ EM < 0.9, continue to 150M tokens (pre-declared). PASS if ≥ 0.90 by 150M.
    - INCONCLUSIVE if EM is still rising but < 0.90 at 150M. FEAS proceeds, flagged.
    - **FAIL** if EM < 0.5 at 150M, or if the answer-token training loss falls while held-out EM stays near 0.
  - (c) Descriptive: `--` per cell; the `AA` − `--` gap against N.
- **Tolerance basis** [INF]. 0.90 is a "has learned" level, not a calibrated one. Our items are harder than MQAR (natural-language template, multi-token keys, 4-token values, filler) but carry a much lower load (N = 8 in 1K). The extension and INCONCLUSIVE grades keep slow learning from being read as a bug.
- **GPU-h [INF].** 2 × 50M tokens at ≈ 130–200K tok/s (FLOP-scaled for a 2-layer, LM-head-dominated model; B §1.3) ≈ 0.2, plus evaluation → **≈ 0.3**; **≤ 0.6** with the extension.
- **Depends on:** G0; a run YAML for a from-scratch run with the TEF data source (`data.stream: tef`, `data.kwargs: {buckets: [1024], synthetic_share: 1.0}`, no parent; configs owner). If the trainer refuses a parentless TEF run, a `scripts/repro/known_answer.py` driver is used instead (scripts owner).
- **Failure tree.**
  - No learning, with high training answer loss → TEF stream (answer masks, loss weights) → trainer step objective → model. Also run R3d on the 2-layer configuration, and decode one TEF batch by hand.
  - Low training answer loss but low evaluation EM → train/eval format mismatch: templates, the whitespace token before values, sequential vs branched queries, filler namespace. Check (a) first.
  - (a) fails → cache-path scoring on trained weights: bf16 KV, RoPE in cached decode, `score_continuation` alignment. Rerun VT-06/07 on this checkpoint.

### 2.7 R7: LAMBADA on published checkpoints, through our scorer (B's L2, A's M2-lite)

- **Claims** [read-section, PI-VER]:

  | Checkpoint | Published LAMBADA | Source |
  |---|---|---|
  | Pythia-70M | 0.185 ± 0.005 | Pythia [2304.01373v2](https://arxiv.org/abs/2304.01373) App. G.1, Table 7 |
  | Pythia-160M | 0.328 ± 0.007 | same |
  | Wang et al. 340M GDN-pure | 0.348 | [2507.06457v2](https://arxiv.org/html/2507.06457v2) Table 2 |

  - Pythia: "Lambada (OpenAI)", zero-shot, fully trained suite, evaluated with the LM Evaluation Harness (§2.5). The deduplicated suite, Table 8, gives 0.192 / 0.342; it is not used.
  - Wang: the Table 2 row is labelled "GatedDeltaNet pure". The variant and SE are not stated. The library Wang trained with documents `lambada_openai` in its evaluation commands (`third_party/flash-linear-attention/evals/README.md` L29 [code-inspected, PI-VER]).
- **Validates.**
  - (i) Our EM_tf is lm-eval's `acc`. lm-eval @`d6de8164`: `acc = is_greedy` (`api/task.py` L1463–1466), and causal models encode context + continuation jointly and then split (`api/model.py::_encode_pair`) [code-inspected, PI-VER].
  - (ii) Our item writer's context/answer split matches lm-eval's.
  - (iii) fla 0.6.0's `GatedDeltaNet`, which `GDNMixer` subclasses, reproduces a published number from weights trained with an older fla. This is a check for semantic drift. VT-04 compares the kernel with *our* fp64 recurrence, and VT-09(2) compares fla with fla.
- **What it cannot validate.** HybridLM's cache path: Pythia runs through `ForwardOnlyAdapter`. That path is covered by VT-06/07/12 and R6(a).
- **Grade:** numeric reproduction.
- **Code.** `eval.scorer.score_file` via `eval.adapter.ForwardOnlyAdapter`, around a ≈ 30-line HF wrapper exposing `full_logits` and `lm_head_weight`. A LAMBADA item writer follows lm-eval's `lambada_openai.yaml` definition:
  - dataset `EleutherAI/lambada_openai`, test split;
  - context = the text minus its last space-delimited word;
  - target = " " + that last word;
  - tokens are split as `_encode_pair` does. The prompt is the context's last token, so the prompt is at least one token long.

  lm-eval itself is **not** imported. Both the wrapper and the writer live in a new `scripts/repro/` (scripts owner).
- **Pins** [code-inspected, PI-VER, HF API]:
  - `EleutherAI/pythia-160m` @`50f5173d932e8e61f858120bcb800b97af589f46`. The model card says branch `step143000` "corresponds exactly to the model checkpoint on the main branch" and that the retrained suite showed "no difference in benchmark performance".
  - `EleutherAI/pythia-70m` @`a39f36b100fe8a5377810d56c3f4789b9c53ac42`.
  - `m-a-p/340M-20B-GatedDeltaNet-pure-baseline` @`14661a98768e1f5d5493ff7b18f2d6809717a4c8`: fp32, 1.60 GB. Config: transformers 4.49.0, 24 layers, hidden 1024, 4 heads × `head_dim` 256, `expand_v` 1, untied, V = 32,000, `max_position_embeddings` 2,048, plus legacy keys `qk_activation`, `qk_norm`, `use_beta`, `expand_k`.
  - `EleutherAI/lambada_openai` `data/lambada_test.jsonl`: 1,819,752 B, LFS oid prefix `4aa8d02cd17c`.
  - Standard variant (Wang only): `cimec/lambada` test split (lm-eval's `lambada_standard.yaml`).
- **Configuration.**
  - fp32 forward; Pythia's fp16 weights are cast up.
  - One sequence per forward.
  - The tokenizer's default special tokens, as lm-eval uses when `add_bos_token` is unset (the Llama-style tokenizer adds a BOS token; GPT-NeoX does not). The BOS variant is a sensitivity analysis only.
  - Load with zero missing or unexpected keys, or with a documented key map.
  - Note: for items of ≤ 64 tokens, fla's GDN layer switches automatically to `fused_recurrent` when there is no grad. VT-05 shows that path is equivalent.
- **Pass.**
  - Pythia-160M acc ∈ [0.313, 0.343]; Pythia-70M ∈ [0.170, 0.200].
  - Wang GDN-pure within ±0.02 of 0.348 on `lambada_openai`, or on `lambada_standard` if only that one matches. Both are reported.
- **Tolerance basis** [INF].
  - ±0.015 ≈ 2× the published binomial SE (0.007; my recomputation at n = 5,153 gives 0.0065 / 0.0054). It also allows for harness-version drift (Pythia's numbers predate `d6de8164`) and fp16→fp32 numerics.
  - ±0.02 ≈ 3 binomial SE (0.0066), because the variant and BOS settings are unstated.
- **GPU-h:** ≈ 0.1. **Downloads:** ≈ 2.2 GB (Pythia 375 MB + 166 MB, Wang 1.6 GB, LAMBADA ≈ 1.8 MB per variant). If the standard-variant file needs the `datasets` package, use `.venv-repro`.
- **Gate role.** Not a G-RV item, but it must finish before any FEAS verdict is computed. A failure that implicates `eval.scorer`, the code shared with FEAS, blocks FEAS verdicts until it is fixed.
- **Failure tree.**
  - Pythia fails:
    1. compare the split and encoding with `_encode_pair` on 20 items inspected by hand;
    2. check EM_tf alignment (`logprobs[:,0]` comes from the end of the context);
    3. if still failing, install lm-eval @`d6de8164` in `.venv-repro` and compare `is_greedy` per item. That localises the fault to our scorer or to the model/tokenizer/data side.
  - Pythia passes but Wang fails → loading (renamed or missing keys; legacy fields that fla 0.6.0 ignores, such as `use_beta` or `qk_activation`), variant, BOS. If unresolved, report "fla 0.6.0 does not reproduce Wang's 340M LAMBADA; GDN semantic drift is not excluded" and raise it at the review. FEAS proceeds, because GDN-A correctness stays anchored by VT-04 and R3c.

### 2.8 R8: induction heads and the in-context-learning score (Olsson et al.; PR P4a, A/B's I1)

- **Claims** ([transformer-circuits 2022](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) [read-section, PI-VER]):
  - The ICL score is "the loss of the 500th token in the context minus the average loss of the 50th token in the context" (Key Concepts). Elsewhere the paper uses the opposite sign; we use I = L(50) − L(500), so that more ICL means a larger I.
  - "Before this window there is less than 0.15 nats of in-context learning, after it there is roughly 0.4 nats … (except for the one layer model where not much in-context learning ever forms)".
  - Induction heads are defined by their behaviour "on a repeated random sequence of tokens".
  - The small models had 1–6 layers, d_model 768, a 2^16 vocabulary and 8,192 context, and trained for "10,000 steps (~10 billion tokens)". The phase change came at "approximately 1–3 billion tokens", within a warm-up that "ramps up over the first 1.5e9 tokens".
- **Validates.**
  - That attention in our models forms the canonical in-context mechanism.
  - Whether the training budget passes the transition (E2.3, E1.9).
  - In which order things happen relative to S3's 0.15B tokens: did induction form before or after?
- **Grade:** phenomenon.
- **Anchor-transfer caveat.** Tokenizer, data, scale and position scheme differ (A §5.3). Two signals are therefore required (below), and the 0.15-nat boundary alone never triggers anything.
- **Configuration.**
  - **ICL score:** 2,000 documents with ≥ 512 tokens from the FineWeb-Edu shard-000 *dev* half (`repro` stream). Per-token loss at 1-based content positions 50 and 500 (counted after `<|endoftext|>`), from single-document forwards. A window version (positions 45–55 vs 495–505) is a sensitivity analysis.
  - **Induction signature:** 200 sequences, each `<|endoftext|>` + 256 random tokens from the fixed 4,096-token LM-4(b) subset + the same 256 tokens again (LM-4(b) at gap 0; `eval/tier0.py`). Metric: mean loss over tokens 9–256 of the first copy and of the second copy.
- **Checkpoints.**
  - FEAS Stage-A bf16 milestones at {50M, 100M, 200M, 400M} tokens and S (`stage_A.yaml` `milestone_tokens` [code-inspected, PI-VER]), for k ∈ {0, 12} and seeds 101, 102.
  - S3 final checkpoints at η* for k ∈ {0, 3, 12}, and R9's k = 1.
- **Pre-declared readings.**
  - (a) **k = 12 induction formed:** second-copy loss ≤ 1.0 nat **and** ≤ 0.2 × first-copy loss [INF threshold: a working induction circuit predicts repeated tokens almost deterministically].
  - (b) **k = 12 ICL:** I ≥ 0.15 nats, Olsson's own pre/post boundary.
  - Both (a) and (b) fail at S → **audit trigger**: masking, RoPE, position reset, QK-norm gains, attention statistics, and the budget trajectory. This happens *before* a FEAS failure may be read as a scale failure.
  - (a) passes and (b) fails → "data/tokenizer difference"; reported, no audit.
  - (a) fails and (b) passes → check the R8 generator and positions.
  - (c) **Plateau:** |I(S) − I(400M)| < 0.02 nats for k = 12. A failure is flagged in the FEAS report as a possible under-training signal (E2.3).
  - (d) k = 0: descriptive (there is no GDN anchor).
  - (e) Formation timing: the first milestone at which (a) holds.
  - **Escalation (decision relevance):** if (e) shows k = 12 forming induction only *after* 0.15B tokens, then S3 would choose η* on pre-induction models. The PI must log a decision row before S3 starts, choosing between running S3 as frozen with η* labelled "selected on pre-induction models" and extending the S3 runs. The choice must not look at task metrics.
- **GPU-h:** ≈ 0.1 (about 20 checkpoints × about 1.1M tokens of inference).
- **Depends on:** a small ICL-score helper (per-position loss at 50 and 500) and the LM-4(b) gap-0 items (S0 evaluation); `scripts/evaluate.py` milestone loading (exists).
- **Failure tree.** See the readings above. When k = 12 lacks induction, check in this order: VT-03 at trained weights → attention entropy and max-logit logs → whether the milestone trajectory is still rising (budget).

### 2.9 R9: ratio-vs-loss shape (Mamba-2, Waleffe), plus one k = 1 run at η* (PV M5 / PR P3 reduced, B's LM1-lite)

- **Claims** [read-section, PI-VER].
  - **Mamba-2** ([2405.21060v1](https://arxiv.org/html/2405.21060v1) §9.2.3, Table 2): a 350M model with 48 layers, trained on 7B Pile tokens with the GPT-2 tokenizer, an SSD + attention stack. Perplexity by number of attention blocks:

    | Attention blocks | 0 | 1 | 2 | 3 | 4 | 5 | **6** | 7 | 9 | 11 | 15 | 24 | Transformer++ |
    |---|---|---|---|---|---|---|---|---|---|---|---|---|---|
    | Perplexity | 8.60 | 8.38 | 8.32 | 8.29 | 8.29 | 8.28 | **8.26** | 8.27 | 8.28 | 8.30 | 8.34 | 8.50 | 8.68 |

    The paper: "Having around a 10% ratio of attention layers performs best". In nats [INF]:
    - 0 → 1 block: 0.026;
    - 0 → best: 0.040, so the first block gives 64% of the gain;
    - 24 blocks: 0.029 above the best;
    - Transformer++: 0.050 above the best.
  - **Waleffe et al.** ([2406.07887v1](https://arxiv.org/html/2406.07887v1) §4.1): "validation loss is minimized when roughly 8% of the layers are self-attention layers" (130M, 24 layers). Values exist only in Fig. 4 and are not used.
  - **Delta-rule counter-anchors.**
    - Gated DeltaNet ([2412.06464](https://arxiv.org/html/2412.06464) Table 3; 1.3B, 100B FineWeb-Edu tokens, 4K): Wiki perplexity GDN 16.42, Transformer++ 18.53, GDN-H1 (GDN + SWA) 16.07. So pure GDN beats the Transformer by 0.121 nats [INF].
    - Wang Table 6: the GDN LM-accuracy average is highest for *pure* GDN (0.487 vs 0.468–0.481 for the hybrids).
    - No delta-rule sweep with a pure-linear point reports loss per ratio (B §3.2b; my reading of the sources I opened).
- **Validates.**
  - (i) Whether our LM stack shows the published pattern. This is descriptive, because the anchors are Mamba-family.
  - (ii) **Engineering:** that k = 1 trains stably at η*. The core trains k = 1, but S3 never does (B §3.2b).
  - (iii) It makes RP-2's "largest step at 0 → 1" clause evaluable.
- **Grade:** phenomenon, descriptive. It never gates (I-23).
- **Code.** `scripts/train.py` in composition mode: `--model configs/model/<scale>/C1.yaml --training configs/training/lr_sweep.yaml --seed 100 --lr <η*> --exp-stage S3 --tag repro-R9`.
- **Configuration.** The S3 recipe: 1,145 steps × 131,072 tokens, compressed WSD 115/916/229, at the scale FEAS chooses. Metrics:
  - final LM-2 subset loss, exactly as S3;
  - LM-3 repeat-token loss by Δ bucket on dev-split documents.
- **This run never enters the η* minimax-regret rule.**
- **Pre-declared readings.** Differences are judged against the protocol's own 0.01-nat LM-loss equivalence margin (§11.7). For scale, modded-nanogpt's 80 reruns of a 124M model have SD 0.0019 nats (@`4ea6b93`, SoftCap record README, recomputed [code-inspected, PI-VER]); our σ is unmeasured (optional O3).
  - **Shape classes.** They are mutually exclusive and evaluated in this order:

    | Class | Condition |
    |---|---|
    | **U** (Mamba-family: a hybrid is best) | L₀ − min(L₁, L₃) > 0.01 **and** L₁₂ − min(L₁, L₃) > 0.01 |
    | **D** (all-full best) | L₁₂ < min(L₀, L₁, L₃) − 0.01 |
    | **G** (pure GDN best) | L₀ < min(L₁, L₃, L₁₂) − 0.01 |
    | **F** (flat) | max − min over {L₀, L₁, L₃, L₁₂} ≤ 0.01 |
    | other | described as it is |

  - **Hybrid-penalty flag** (can accompany any class): min(L₁, L₃) − L₀ > 0.02. A hybrid with 1–3 attention layers that is clearly worse than pure GDN lies outside the published delta-rule evidence. The nearest data point has the GDN + SWA hybrid 0.021 nats *better* than pure GDN (GDN Table 3), although that hybrid uses SWA rather than global attention. The flag triggers checks, not an audit (failure tree). By contrast, L₁₂ > L₀ alone is expected for a delta-rule stack (GDN Table 3) and is only reported.
  - **Mamba-2 prediction, mapped by attention fraction** [INF]. Our k = 1 (8.3%) and k = 3 (25%) bracket Mamba-2's optimum (6/48 = 12.5%). The pattern would predict L₁ ≈ L₃ ≈ the minimum, L₀ about 0.04 above it and L₁₂ about 0.05 above it. Mapping by block count would predict something different (B §3.2b), so the prediction is directional only.
  - **RP-2 LM-3 clause (audit trigger):**
    - *Primary,* on FEAS end-of-Stage-A checkpoints (k ∈ {0, 12}, seeds 101/102; 0.6B tokens): repeat-token loss at Δ ∈ [1K, 4K) must be lower for k = 12 than for k = 0 in both seeds.
    - If k = 12 ≥ k = 0 in both seeds → audit data, masking and positions before S3.
    - A split result → read the S3 finals (secondary).
    - If R8 shows k = 12 without induction at 0.15B tokens, the S3 reading is labelled "pre-induction" and cannot trigger anything.
  - **k = 1 stability:** if LM1-lite diverges (G-div) at η*, the PI logs a decision row before G5 on k = 1's LR, using §7.2's per-configuration log-linear rule.
- **GPU-h [INF].** 1,145 × 131,072 tokens at the contended rate. D384 4K medians are C0 70.1K and C3 61.4K tok/s [MEAS-S0], so C1 ≈ 68K → 0.61 h; ≈ 0.66–0.7 h at D512. **≈ 0.65–0.7** including evaluation.
- **Depends on:** S3 (η*), the trainer, the LM-3 generator (`eval/tier0.py`).
- **Failure tree.**
  - LM-3 clause fails → does attention cross separators in the LM-3 documents → position reset → LM-3 annotation positions → VT-03 at trained weights.
  - Hybrid-penalty flag → the G-LR regret table (did k ∈ {1, 3} have its own optimum at another LR?) → R8 status → LM-3 clause. If R8 and LM-3 look normal, it is a scale or budget phenomenon: report it, and give Tier C #8 (4× Stage-A tokens) and #7 (η*/√2) higher priority at the Tier-A decision. If they look abnormal, audit attention before G5.

### 2.10 R10 (should tier): length behaviour of FEAS checkpoints (PV S2 / PR S2)

- **Context, not targets.** Rotary Transformers' loss rises beyond their training length (Press et al., [2108.12409](https://arxiv.org/abs/2108.12409)). Recurrent models degrade differently beyond theirs (GDN paper; Stuffed Mamba). No numbers from these sources are used, because θ = 1e4 settings do not transfer to θ = 5e5 with QK-norm (A, B).
- **Configuration.**
  - LM-1-style per-position loss up to 32K on PG19 *validation* books, plus R0 passkey items (`repro` stream, namespace `dev`, 100 items each) at {4K, 8K, 16K, 32K}.
  - Checkpoints: end of Stage A (S) and the ends of the Stage-B branches (30M and 90M), for k ∈ {0, 12} and seeds 101/102.
- **Readings.**
  - (i) Within the training window, Stage-A loss is non-increasing from [64, 256) to [1K, 4K) for both k. A failure → audit packing and position reset (VT-03, VT-11).
  - (ii) After Stage B: for C12, the [8K, 16K) and [16K, 32K) losses are ≤ the [2K, 4K) loss + 0.02. A failure is a G-REF early warning (Stage B did not teach long positions), raised at the FEAS verdict.
  - (iii) Stage-A extrapolation beyond 4K: descriptive for both k; no direction is pre-registered at θ = 5e5.
- **GPU-h:** ≈ 0.3. **Depends on:** `scripts/evaluate.py` (Tier0) and R0 items.

### 2.11 Optional items: these run only on their pre-declared trigger

| ID | Item | Trigger | What it decides | GPU-h |
|---|---|---|---|---|
| O1 | Public long-context positive control (A's X5 / B's A4): our generators with the model's own tokenizer at 4K/16K/32K (R1 D = 3, R-load N = 32, C0), 50 contexts × 4 queries, logits computed only at the needed positions. Candidate: `Qwen/Qwen3-0.6B-Base` @`da87bfb` (config: `max_position_embeddings` 32,768, θ 1e6, bf16 [code-inspected, PI-VER]). Its trained context and its tokenizer's compatibility with `TokenSpace` must pass the pre-flight (rule 12) | FEAS-1 fails at 32K for both seeds, **before** FEAS-512 is launched | 32K R1 D = 3 EM ≥ 0.7 → the items can be solved by a long-context LM, so the failure is on the model side and the FEAS ladder continues. < 0.3 → inspect items and format before spending FEAS-512's 11.8 h. Never used for power or ceiling planning (I-23) | ≈ 0.3 + 1.2 GB download |
| O2 | Smeared-key mechanism pilot (Olsson Argument 2): 1-layer `A` vs `C`, 2-layer `AA` vs `CC`, D384 widths, 0.2B tokens | U-02 fires (δ_gc > 3 pp) at the Tier-A decision | Wording of the conv explanation (E1.6) only | ≈ 1.0 |
| O3 | S3 noise floor: seed 103 at k ∈ {0, 12} at η* | G-LR's maximum regret at η* lies within ±0.005 of the 0.02 threshold, or an R9 class depends on a difference within ±0.005 of the 0.01 margin | Robustness of G-LR and of the R9 reading | ≈ 1.3 |
| O4 | Environment soak (A's X0): ≥ 1 h at production shapes with Xid monitoring, plus the failing job rerun alone with `CUDA_LAUNCH_BLOCKING=1` | A new Xid or RC-watchdog event in a project process (R2) | Whether gate and production runs can continue | ≈ 1.0 |
| O5 | R5 bimodality seeds (+2 per arm) | R5 rule 4 | R5 becomes descriptive | ≈ 0.9–1.1 |

---

## 3. Tiers, budget and schedule

### 3.1 Tiers and hours (new GPU-h unless stated) [INF]

| Tier | Items | GPU-h | Role |
|---|---|---|---|
| Done | R1 | 1.75 (spent) | G-RP(a) PASS |
| **Must, gates FEAS (G-RV)** | R2 0.1; R3 0.12; R4 0.6–0.9 (G-RP(b)); R5 1.3–1.7; R6 0.3–0.6 | 2.4–3.4 | no FEAS until passed or diagnosed |
| **Must, before the FEAS verdict** | R7 0.1 | 0.1 | blocks FEAS verdicts only if the scorer is implicated |
| **Must, in-run audits** | R8 0.1; R9 0.65–0.7 | 0.75–0.8 | audit triggers and two PI escalations |
| Should | R10 0.3 | 0.3 | audit trigger |
| Optional | O1–O5 | ≤ 4.7 in all | trigger only |
| **Programme total** (R1–R10) | | **≈ 5.3–6.4** | v1.0 S1 line: 3.0 |

Effect on the protocol (§13.2; exact figures in the amendment), using the upper bound of 6.4 h:
- Tier A: 85.7 → 89.1 h base (106.9 h with 20% contingency) on the D384 path, and 143.9 → 147.3 h (176.8 h) on the D512 path.
- The minimum H1-falsification experiment replaces its "reduced S1 (1.5 h)" with R1–R7 (≈ 5.3 h): 53.4 → 57.2 h (68.6 h) and 93.6 → 97.4 h (116.9 h).

**Pricing basis.**
- Zoology runs: S1(a) timings [MEAS-S1].
- LM training: S0's contended medians [MEAS-S0]. These are 44–58% of the planning rates (B F3).
- G0-T will re-price R6 and R9. If they grow by more than 50%, the PI decides again whether to run R9 (rule in DEV-01).
- This programme does not resolve the separate, larger question of the main budget under D041. If the clean throughput confirms < 95K tok/s at D384, G0 fails its own throughput criterion, and that needs its own decision.

### 3.2 Queue order and dependencies

1. **Now, no GPU:** DEV-01 logged (this step). The S1 owner commits `run_queue.py`. R2(a) one-queue policy starts.
2. **R2(b), (c):** event tagging live; 32K kernel profile.
3. **R3a**, then **R4**: G-RP(b), with R3a's result and the per-cell table reported beside it.
4. **R5**: 6 runs, plus O5 if triggered.
5. **R3b–R3e** (run by S0 with its test suite; any time after the code is stable, before R6).
6. **R6**, after G0.
7. **R7**: any time after its wrapper exists, and before FEAS starts (it is short).
8. **G-RV review by the PI** (one ledger entry), then **S2 FEAS-384**.
9. **During FEAS:** R8 on the Stage-A milestones; its timing reading comes before S3. R10 on the Stage-A and Stage-B checkpoints. The RP-2 LM-3 primary clause is read on FEAS S. O1 only on its trigger.
10. **S3**, then **R9** (k = 1 at η*), then R8 on the S3 finals.
11. **Reproduction review** (`reports/reproduction/review.md`), then **G5**.

### 3.3 Gate G-RV (the exact protocol wording is in the amendment)

**FEAS starts only when all of these hold:**
- G0 has passed;
- G-RP (a) and (b) have passed, with (b) evaluated after R3a;
- VT-21…VT-25 pass;
- R5 is not FAIL;
- R6 is not FAIL;
- R2 (a) and (b) are live, and R2(c) is recorded (longest kernel < 1 s, or a logged mitigation).

**Also:** R7 must be complete before any FEAS verdict is computed.

**If it fails:** diagnose under §1 rules 4–8. FEAS starts only after the fix and a re-run of the affected items, or after a PI decision row that accepts a named residual risk.

---

## 4. How results feed decisions in the main protocol

### 4.1 Outcome → affected gate or decision → pre-declared consequence

| Item | Outcome | Gate / decision affected | Consequence (pre-declared) |
|---|---|---|---|
| R2 | New Xid / watchdog / host-OOM event in a project process | Validity of every gate run; G-OOM/G-div bookkeeping | O4 soak before the next gate run. Affected gate runs are repeated. A separate decision is recommended (§7) on giving infrastructure failures their own status, so that they never count as permanent failures correlated with k |
| R2 | Longest 32K kernel ≥ 1 s | FEAS start | Mitigation decision row before FEAS |
| R3 | Any VT-21…25 fails | G-RV; E10.4 status | Fix only with a test that isolates the defect. If the fix touches attention, GDN or conv code, rerun R4 and R5. An item defect (R3e) means items are regenerated before G5 |
| R4 | G-RP(b) fails | G-RP (frozen: no LM training) | Diagnosis ladder (§2.4) |
| R5 | FAIL | G-RV; how FEAS-2 and k = 0 are read; E10.4; D040 wording | Precision audit. If a bf16 handicap of GDN-A is confirmed and is not a defect, report it as a property of "GDN-A as built", and attach a caveat to FEAS-2 and to every R(k) |
| R5 | PASS | Tolerance basis of G-RP(b) | Measured seed SD reported next to ±0.03. The frozen threshold is not changed |
| R6 | FAIL | G-RV; protects the U-01 scale rule from a format bug being misattributed | Fix the pipeline before FEAS |
| R6 | INCONCLUSIVE | FEAS proceeds, flagged | If FEAS's elicitation check fires (k = 12 R1 D = 3 at 4K < 50%), check learnability and format first, before the answer-only fallback is taken as the explanation |
| R7 | Pythia fails and the scorer is implicated | FEAS verdicts | Blocked until fixed |
| R7 | Wang fails, Pythia passes | E10.4 for GDN semantics stays open | Raised at the review. FEAS proceeds; VT-04 and R3c remain the GDN anchors |
| R8 | k = 12 induction absent at S (both signals) | FEAS-1/2 interpretation; the U-01 fallback ladder | Audit attention before reading FEAS as a scale failure |
| R8 | Induction forms only after 0.15B tokens | S3 design | PI decision row before S3: run as frozen with the "pre-induction η*" label, or extend S3. Never selected on task metrics |
| R8 | k = 12 not plateaued at S | E2.3 under-training | Flagged in the FEAS report. Tier C #8 priority reconsidered at the Tier-A decision |
| R9 | LM-3 clause fails on FEAS S | Audit before S3 | Data, masking and position audit (§17 RP-2) |
| R9 | k = 1 diverges at η* | k = 1's LR before G5 | PI decision row under §7.2's per-configuration rule |
| R9 | Shape class (§4.2) | Wording; Tier C priorities; plausibility of E6.8; the expected E9.1 pattern | See §4.2. Never changes k levels, the LR rule, the TEF or any threshold |
| R10 | Within-window check fails | Audit before S3 | Packing and position reset |
| R10 | Post-Stage-B long-position check fails | G-REF early warning; how FEAS-3 is read | Raised at the FEAS verdict |
| O1 | (if triggered) The public model solves the 32K items / does not | FEAS fallback ladder | Proceed to FEAS-512 / inspect items and format first |

### 4.2 If the Waleffe / Mamba-2 ratio U-curve on loss does not appear at our scale

The anchors are in §2.9. In short:
- **Mamba-family stacks** show a 0 → 1 step of about 0.026 nats, a minimum at roughly 8–12% attention, and an all-attention model about 0.05 nats worse than the best hybrid.
- **Delta-rule stacks** have no comparable sweep. Their nearest evidence points the other way: pure GDN beats Transformer++ by 0.121 nats of Wiki perplexity at 1.3B, and pure GDN has the best LM-accuracy average in Wang et al.

So, for GDN-A at 21–41M parameters and 0.15B tokens, **the absence of the U-curve is an expected outcome, not a bug signal.** Each class implies:

- **U present.** Consistent with the Mamba-family pattern. No action. The validity pattern E9.1 (loss flat for k ≥ 1 while retrieval differs) is expected, so task scores remain the only dependent variable, as designed.
- **F (flat) or G (GDN-favoured): U absent.**
  1. No confirmatory consequence: every hypothesis (H1–H6) is defined on task scores, and loss is used only for η*, whose regret rule does not depend on the shape.
  2. LM loss then carries no information about whether attention is used at this scale. The **LM-3 clause and R8's induction checks become the only LM-level sanity signals.** Both are already in the programme.
  3. The literature review's expected "replication (a): flat LM loss across r and a large drop from k = 0 to k = 1" (§12.4) is reported as **not replicated in our setting**, citing GDN Table 3 and Wang Table 6 as consistent with that. It is not dropped quietly (rule 9).
  4. The external-validity paragraph of the report must say that our scale does not reproduce the loss regime of the 130M–350M Mamba-family studies. This scopes our "consistent with prior work" statements (§11.9).
- **D (all-full best on loss).** This would differ from every published sweep. It is plausible at tiny scale and short training [HYP].
  - The all-full reference then has no LM deficit, so E6.8 (a small-scale all-full deficit) becomes an unlikely explanation of any reference inversion. The denominator S₁₂ of R(k) is not handicapped on LM terms.
  - No design change.
- **Hybrid-penalty flag** (a 1- or 3-attention-layer hybrid more than 0.02 nats worse than pure GDN). This is outside the published delta-rule evidence. Check the G-LR regret table, then R8, then the LM-3 clause.
  - If all three are normal: report it as a scale or budget phenomenon. Tier C arms #8 (4× Stage-A tokens) and #7 (η*/√2) move up at the Tier-A decision, as discriminators of E2.3 and E1.8.
  - If any is abnormal: audit attention before G5.
- **Pre-induction.** If R8 shows k = 12 without induction at 0.15B tokens, the S3-scale shape is labelled "pre-induction", is uninformative, and is reported as such.

### 4.3 What the programme can never change

- The confirmatory family, the primary cell, the H1–H5 tests and the G-REF, FEAS and G-FULL thresholds.
- G-RP's thresholds.
- I-23: no reproduction output feeds power, ceiling or level choices.
- No code change justified only by a target number.

Reproductions act on the protocol only through the gate G-RV, the audit triggers above, and the two PI escalations: the S3 induction timing (R8) and k = 1's LR (R9). Each escalation is a logged decision row taken before the stage it affects.

---

## 5. Rejected candidates, with reasons

Problem codes: **CF** cannot fail at the observed ceiling; **FB** can fail with no bug; **NU** tests a component we do not use; **RD** redundant with an existing or planned test; **IF** infeasible here as specified; **SD** the phenomenon disappears when scaled down; **AT** the anchor does not transfer (rule 3); **NR** no decision depends on it (rule 1).

| Candidate (source) | Codes | Reason | What covers its intent |
|---|---|---|---|
| Wang 1.3B RULER subset + our generators on Wang's checkpoints (PR P2) | FB, AT, IF, RD, NR | On the chosen subtasks, Table 5 has the Transformer *below* GDN-3-1 (Singlevalue 0.559 vs 0.875; VT 0.066 vs 0.024) [read-section, PI-VER]. B §3.2a computes ≈ 27% joint false alarms even if Wang's gaps replicated exactly. The released ratios are not a clean IV: "3-1" places attention at layers {2, 5, …, 23} = 8 of 24, with 16 heads vs 32 for "24-1" [code-inspected, PI-VER]. Needs a `flash_attn` shim, ≈ 33 GB of fp32 downloads and RULER dependencies (B §1.4). Our generators reject the tokenizer (B F4). FEAS-2 tests sensitivity to k on our own models | FEAS-2; R3e; O1 |
| Tier-0 metrics on Wang's 340M ratio sweep (PV M3) | IF, FB, NU, NR | Tokenizer rejected by `TokenSpace` (B F4). Published recall is non-monotone at k/24 ∈ {0, 1, 2}. Tier 0 only corroborates (D037) | R8; R9 LM-3 clause |
| Wang Transformer / 3-1 hybrid LAMBADA and trained-weight transplants (PV M2(b) hybrid half, M2(c)) | NU, IF, RD | fla's `Attention` is not our F_g (GQA, no QK-norm, no gate, θ = 1e4). Needs a shim and an untied-head wrapper. VT-09/VT-04 and R3c cover the configurations we actually use | R3c; R7 (GDN-pure) |
| Per-item agreement with lm-eval as a must (PV M2(d), PR S3) | RD, IF | Scorer alignment is tested by VT-07/VT-12's oracle and R6(a). lm-eval is not installed, and installing it would change the frozen env | R7 against published numbers; lm-eval only in R7's failure tree, in `.venv-repro` |
| GPT-2 LAMBADA (PR S3) | IF, AT | Weights not cached (B F5). The target is a community run on an old harness | Pythia's published table (R7) |
| nanoGPT ladder (PV M4, PR S4) | NU, RD | r0 checks torch/SDPA, already exercised by R1 and VT-02/09; 1.4697 is one A100 run with no published variance (`nanoGPT` @`3adf61e` README L51 [code-inspected, PI-VER]). r2's targets are in VT-13. Our model has no dropout or train/eval mode | R3d (an independent loop on *our* model) |
| Multi-run ratio-loss sweep (PV M5: k = 1, 6 + seeds; PR P3: 9 runs) | AT, FB, NR | Mamba-family anchors. PV's flatness test flips with an unstated mapping of k to blocks (B §3.2b). P3-c's min-of-4 is biased towards passing. Re-priced at 2.7 / 6.0 GPU-h | R9: one k = 1 run, descriptive |
| Olsson smeared-key toy runs as must, and in PV's gate G-RV (PV M6, PR P4b) | SD, AT, RD, NR | No 1-layer ICL value is given in the text; 0.1–0.2B tokens vs ≈ 10B; X-gc12 runs unconditionally (D005); attention correctness is covered by VT-02/03/09 | R8 (measurement); O2 on U-02 |
| 50-step snapshot cadence for ICL (PR P4a) | IF (S0 change), RD | Stage-A milestones and S already give 5 points per FEAS run | R8 |
| Jelassi copy-task sample efficiency (PV S1, PR S1) | AT, SD, FB, RD | The anchor is Mamba with S = 32. GDN-A's state per layer (24,576 entries at D384) dwarfs a 300-letter string. The ratio is undefined if GDN never reaches 0.95. The TEF dose ladder and the still-closing flag measure E1.7 in place | TEF dose ladder; R8 |
| Our generator vs RULER on a public checkpoint (PV S3) | CF | "≤ 10 pp *or* an explained difference" cannot fail | Tier C #12 (unchanged) |
| Wang 1.3B 24-1 LAMBADA (PV S4) | IF, RD | Needs the shim for its one attention layer; 5.8 GB of fp32 | R7 |
| Zoology seed replicate at ceiling (PR S5); seed basis for ±0.03 (PV R-S1-5) | CF | Variance at ceiling is compressed | R5 seeds |
| Re-apply the S1 cut on GDN timing (PV R-S1-1, PR §2.1); −0.005 monotonicity tolerance (PV R-S1-3); "check Zoology's conv fallback" (PV R-S1-4) | moot / post hoc | S1(a) finished at ≈ 1.75 h. Monotonicity passed (0.9892 → 0.9938). Adding a tolerance after seeing the d128 number would be a post-hoc change to a frozen gate (A §3.6). The `causal_conv1d` message is Zoology's Mamba import probe | Rule 13 (lesson); R3b |
| All Nice tiers: modded-nanogpt (PV N2, PR N1); full RULER (PR N2); Wang 340M floor (PR N3); phonebook (PV N3, PR N5); depth curves (PR N4); Stuffed-Mamba-style GDN (PV N4); Wang RULER Table 5 (PV N1) | NU, IF, AT | Environment-only, `mamba_ssm`/compile/NCCL requirements, or figure-only anchors. The GDN paper released no checkpoints ("No, we only provide code implementations", `gated_deltanet` @`b53d6d3` README L90 [code-inspected, PI-VER]) | — |
| A 1 GPU-h soak by default (A's X0) | NR (for now) | No Xid event since 18:25. The lr 1e-2 cell completed cleanly with only CARLA beside it | R2 tagging; O4 on trigger |

---

## 6. Unresolved disagreements and how they will be settled

| # | Disagreement | Positions | Ruling now | How it is settled |
|---|---|---|---|---|
| 1 | Value of M7 once X2 exists | A: top priority and gating. B: conditional should (after X2, M7 can only show precision and RNG effects) | R5 is must and gating, for the three reasons in §2.5. Precision **is** decision-relevant (bias of the k = 0 baseline) | At the review, record whether R5 told us anything that R3a could not (a precision effect or the seed spread). If not, B's point becomes a standing rule for future reproductions: "behavioural comparisons only after an identity test, and only if a decision depends on the regime" |
| 2 | Known-answer test X4 | A: must and gating. B: should | R6 gates, because a format bug would be misread by U-01 as a scale failure (D512 adds about 58 GPU-h) | R6's outcome and its diagnosis history at the review |
| 3 | Environment | A: 1 h soak now. B: lean hygiene, escalate on recurrence | R2 now; O4 on trigger | Xid tagging across R4–R6 and FEAS. Any recurrence → O4 |
| 4 | lm-eval | A: separate venv as the reference. B: our scorer only | Our scorer against published tables; lm-eval only to localise a failure | R7's outcome |
| 5 | New LM-loss runs | A: none. B: one k = 1 at η* | One (R9), descriptive | R9's results; usefulness reviewed before G5 |
| 6 | Answerability | A: a public long-context model. B: oracle first | R3e is must; O1 runs on FEAS-1 failure | FEAS outcome |
| 7 | Olsson's 0.15-nat bar | PR: pass threshold. A/B: plateau only; the anchor does not transfer | Two-signal rule; audit only if both fail | R8 data: if random-token copying forms while ICL < 0.15, A/B's transfer point is confirmed |
| 8 | Ratio-loss anchors | PV/PR: Mamba-family shape as an audit trigger. A/B: the anchor does not transfer | Shape descriptive; only the LM-3 clause audits | No decision depends on the shape; R9 is reported against both anchor sets |
| 9 | Wang 1.3B ratio-sweep reproduction | PR: must. A/B: cut | Cut | Reopen only if an external reviewer asks for an external replication of Wang's trend: then RULER in `.venv-repro` as a post-study item (B estimates 4–7 GPU-h) |
| 10 | Pricing | B: LM items under-priced 1.7–2.3× | Priced at measured contended rates | G0-T re-pricing. If R6 or R9 grows by more than 50%, the PI decides R9 again before it runs |
| 11 | Root cause of the Xid 8/31 events | Unknown (A, B) | Not a ruling; an open fact | R2(b) tagging; O4 on recurrence; R2(d) |
| 12 | Whether S1/G-RP is "only a smoke test" | Both critiques: yes | Agreed. G-RP stays the formal gate; R5 supplies the calibration | — |

Points agreed by everyone are settled in DEV-01: PV finding 3 (Wang's released ratios) becomes an annotation to the protocol's §2 prior; the S1 deviations are logged; no new packages go into `.venv`; one GPU queue for all agents.

---

## 7. Actions and owners (requests; I edited none of these files)

| Owner | Action | For |
|---|---|---|
| PI (this step) | This plan; `protocol_amendment.md`; row DEV-01 | — |
| Orchestrator | One GPU queue for every agent, from now on. At 22:40, two project GPU jobs ran at once next to CARLA | R2(a) |
| S1 owner | Commit `run_queue.py`. Implement R3a (`scripts/s1_zoology/test_s1_transplant.py`), the R4 runs plus R4b, and the R5 runs. Report the per-cell tables with the MDD. Keep `s1_reproduction.md` up to date | R1, R3a, R4, R5 |
| S0 models | VT-21 (conv vs reference), VT-22 (long-T transplants), the 32K kernel profile | R3b, R3c, R2(c) |
| S0 training / runner | `infra_events` tagging (Xid, RC watchdog, host OOM); VT-23 (independent reference loop); G0-T; support for a parentless TEF run (R6) | R2(b), R3d, R2(e), R6 |
| S0 evaluation | VT-24 (oracle at 16K/32K); ICL-score helper and LM-4(b) gap-0 items; R10 evaluations | R3e, R8, R10 |
| Scripts owner | `scripts/repro/` containing: the HF wrapper and LAMBADA item writer (R7); the drivers for R6/R8/R10 if they cannot be done by configuration | R6–R8, R10 |
| Configs owner | Update `configs/experiments/plan.yaml` (S1 → programme, DEV-01); run YAMLs for R6 and R9 | R6, R9 |
| All implementers | Results in `experiments/repro/<R-id>/`; ledger rows in `reports/reproduction/results.md` | rule 9 |
| PI (later) | G-RV review; the two escalations (R8 timing before S3, R9 k = 1 LR); reproduction review before G5. **Separate decision recommended** (not part of DEV-01): infrastructure failures get their own status in §8.1 / D031, so that they are never "permanent failures" correlated with k | §4 |

---

## 8. Sources opened for this decision

| Source | Used for | Access |
|---|---|---|
| [arXiv 2507.06457v2](https://arxiv.org/html/2507.06457v2) Wang et al.: Tables 2, 3, 5, 6, 7; App. D; §4.3; Limitations | R7 target (0.348), the §2 prior annotation, the rejection of PR P2, R9 counter-anchor | read-section, PI-VER (raw HTML parsed with `python -I`) |
| HF `config.json` of `m-a-p/1.3B-100B-GatedDeltaNet-hybrid-{3-1, 24-1}` and `m-a-p/340M-20B-GatedDeltaNet-hybrid-3-1` | Attention layers {2, 5, …, 23} (16 heads) vs {23} (32 heads); 8 KV heads; θ 1e4; `max_position_embeddings` 2,048 | code-inspected, PI-VER |
| HF API and config of `m-a-p/340M-20B-GatedDeltaNet-pure-baseline` @`14661a9`, `EleutherAI/pythia-160m` @`50f5173`, `EleutherAI/pythia-70m` @`a39f36b`, `EleutherAI/lambada_openai` data tree, `Qwen/Qwen3-0.6B-Base` @`da87bfb` | R7 pins; O1 candidate | code-inspected, PI-VER |
| [Pythia model card](https://huggingface.co/EleutherAI/pythia-160m) | `step143000` = main; "no difference in benchmark performance" between versions | read-section, PI-VER (WebFetch) |
| [arXiv 2304.01373v2](https://arxiv.org/abs/2304.01373) Pythia: App. G.1, Tables 7–8; §2.5 | R7 targets | read-section, PI-VER (PDF text) |
| [arXiv 2405.21060v1](https://arxiv.org/html/2405.21060v1) Mamba-2: §9.1, §9.2.3, Table 2 | R1 corroboration; R9 anchor | read-section, PI-VER |
| [arXiv 2406.07887v1](https://arxiv.org/html/2406.07887v1) Waleffe et al. §4.1 | R9 anchor (text only) | read-section, PI-VER |
| [arXiv 2412.06464](https://arxiv.org/html/2412.06464) Gated DeltaNet: §4 setup, Table 3 | R9 counter-anchor | read-section, PI-VER |
| [Olsson et al. 2022](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html): Key Concepts, Argument 1, Argument 2, Model Analysis details | R8 | read-section, PI-VER |
| Zoology @`1ad20d1`: README figure; `mixers/gated_delta_net.py` L246–250, L295; `mixers/attention.py` MHA; `model.py` L249 | R1, R3a, R5 | code-inspected (figure viewed), PI-VER |
| lm-eval @`d6de8164`: `tasks/lambada/{lambada_openai, lambada_standard}.yaml`, `api/task.py`, `api/model.py::_encode_pair`, `models/huggingface.py` (BOS default) | R7 definitions | code-inspected, PI-VER |
| `third_party/flash-linear-attention/evals/README.md` L29 | Variant used in fla's evaluation commands | code-inspected, PI-VER |
| modded-nanogpt @`4ea6b93` SoftCap record README | 80 reruns, mean 3.2791, SD 0.0019 (recomputed) | code-inspected, PI-VER |
| nanoGPT @`3adf61e` README L51; `gated_deltanet` @`b53d6d3` README L90 | Rejected items | code-inspected, PI-VER |
| Project files: protocol v1.0; `decisions.md`; `s1_reproduction.md`; `experiments/s1_zoology/*`; `runs/bench_train_step/*`; `INTERFACES.md`; `configs/training/{stage_A, tef, lr_sweep, base}.yaml`; `src/hybridlc/eval/{adapter, devtools}.py`; `tests/test_vt0{3,9}*`, `test_vt12_items.py` | Context and facts | read-full / code-inspected |
| `journalctl -k`, `nvidia-smi`, `ps` (22:40 IST) | R2 facts | KLOG / observed |
| PV, PR, A, B | Everything they propose and argue | read-full |

## 9. Not verified

- The root cause of the Xid 8 and Xid 31 events, and the identity of pid 799092.
- Clean (uncontended) throughput, which G0-T will measure.
- Whether fla 0.6.0 loads the Wang 340M checkpoint cleanly (R7 will show), and whether `cimec/lambada` needs the `datasets` package.
- Zoology's GDN accuracy band at d = 64. The figure suggests ≈ 0.89 overall (approximate); R5's regime check handles any outcome.
- Whether the trainer accepts a parentless TEF run with `buckets=(1024,)` and `synthetic_share=1.0` (R6 dependency), and whether a 2-layer `AA` learns R-load within 150M tokens (R6 tests this).
- Qwen3-0.6B-Base's trained context length and whether its tokenizer passes `TokenSpace` (O1 pre-flight).
- The throughput of 1–2-layer models, which is a FLOP-scaled estimate.
- Kimi and Bae numbers: not used anywhere in this plan.

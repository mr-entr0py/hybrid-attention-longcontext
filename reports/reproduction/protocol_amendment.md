# Protocol amendment DEV-01: stage S1 expands into the reproduction programme

**Type:** logged deviation from protocol v1.0 (frozen 2026-11-09).
**Decided by:** PI, 2026-11-09, about 23:00 IST.
**Decision row:** DEV-01 in `reports/decisions.md`.
**Operational specification:** `reports/reproduction/reproduction_plan.md` v1.0, SHA-256 `6bc14af62d7ac5151b5906e7596fbc6b3d0e261f114f65286292a958e8adcebc`. That file is the pre-registration of every item named below.

**Why this is logged as a deviation although it comes before G5.** The protocol says that only post-G5 changes are DEV rows. But §17 and gate G-RP are frozen text, and G-RP is a hard gate ("no LM training until this passes"). Changing what S1 contains, and adding a gate before FEAS, is therefore recorded at the strictest level.

**How it applies.** `reports/experimental_protocol.md` is **not edited in place**. Wherever this amendment and v1.0 differ, this amendment governs, in the same way that v1.0 superseded `initial_proposal.md`. Sections not named here are unchanged.

**Scope.**
- §17 is replaced in full (part A).
- §13.1, §13.2 and §13.3 are amended (parts B–D).
- Consequential edits are made to §2 (annotation of a prior), §8.1, §8.2, §15, §16 and §20 (part E).
- The S1(a) close-out record is in part G. The decision-log row is in part H.

**Rationale, in short.**
1. The user asked that important published results be reproduced, so that our setup and methodology are shown to agree with the literature before our own experiments start.
2. S1 as frozen sits at ceiling: ±0.03 on overall accuracy equals ±0.21 on the only discriminating cell. Both critiques independently call it a smoke test.
3. The two proposals and two critiques show the following:
   - most candidate literature reproductions either fail to transfer to our setting, cannot fail, or are dominated by deterministic tests;
   - several checks with a high risk-per-GPU-hour are missing. These are: the conv kernel against a reference; transplants at the lengths we evaluate; the trainer against an independent loop; answerability of the 16K/32K items; a known-answer test in our own pipeline; GPU-fault hygiene.
4. A train/eval format bug would surface in FEAS as a 32K failure. The U-01 scale rule would read that as "model too small", and D512 costs about 58 GPU-h more.

---

## A. Replacement text for §17

> ## 17. Reproduction programme (S1; runs before any LM training; gates G-RP and G-RV)
>
> *Amended by DEV-01 (2026-11-09); replaces §17 of v1.0. The full specification of every item is in `reports/reproduction/reproduction_plan.md` v1.0 (SHA-256 recorded in DEV-01): claim and source numbers, configuration, pass criterion with how its tolerance was derived, GPU-h, dependencies, and failure tree. That file is the pre-registration of each item.*
>
> ### 17.1 Purpose, grades and rules
>
> - **Purpose.** Before any LM training, show four things:
>   - (i) the environment reproduces the published behaviour of the original code;
>   - (ii) our modules reproduce the outputs of the original implementations, or of independent references, at the configurations we use;
>   - (iii) our pipeline reproduces a known answer;
>   - (iv) published phenomena that the study relies on are visible in our setting.
> - **Grades.** Every item is a reference reproduction, numeric reproduction, phenomenon replication, smoke check or environment check, and states its minimum detectable defect.
> - **Rules** (plan §1):
>   - no value read off a figure is a threshold (I-20);
>   - no reproduction output feeds power, ceiling, level or tuning choices (I-23);
>   - no code change is justified only by a target number;
>   - every attempt and every outcome is reported.
>
> ### 17.2 Items, in execution order
>
> | ID | Item | Grade | Gate role |
> |---|---|---|---|
> | R1 | Zoology MQAR with Zoology's own models (v1.0 part (a)). Done: G-RP(a) PASS, 16/16 | smoke at ceiling | G-RP(a) |
> | R2 | Environment integrity: one GPU queue; each run tagged with the Xid, RC-watchdog and host-OOM events in its window; longest-kernel profile at 32K; clean G0 throughput | environment | G-RV |
> | R3 | VT-21…VT-25 (§17.4) | reference reproduction | G-RV |
> | R4 | Our mixers in Zoology's harness (v1.0 part (b), 6 runs), plus a production-F_g smoke run in fp32 and in bf16 (2 runs) | smoke | G-RP(b) |
> | R5 | Zoology's GDN vs ours under bf16 autocast at d_model = 64, lr 1e-3, seeds 123–125 (6 runs; +4 if the seeds are bimodal) | phenomenon + calibrated comparison | G-RV |
> | R6 | Known-answer associative recall in our pipeline: 2-layer `AA` and `--`, TEFStream (1K bucket) → trainer → scorer, seed index 300 | phenomenon + chain consistency | G-RV |
> | R7 | LAMBADA through our scorer. Pythia-160M (0.328 ± 0.007) and Pythia-70M (0.185 ± 0.005), Pythia App. G.1 Table 7; Wang et al. 340M GDN-pure (0.348), Table 2 | numeric reproduction | complete before any FEAS verdict |
> | R8 | Induction signature on repeated random tokens, and ICL score (Olsson et al.), on FEAS Stage-A milestones and S3 final checkpoints | phenomenon | audit trigger; escalation before S3 |
> | R9 | Ratio-vs-loss shape on S3 at η*, plus one k = 1 run (seed index 100, S3 recipe); RP-2 (§17.5) | phenomenon, descriptive | audit trigger (LM-3 clause); escalation if k = 1 diverges |
> | R10 | Length behaviour of FEAS checkpoints | descriptive (should tier) | audit trigger |
> | O1–O5 | Optional items, run only on their pre-declared trigger (plan §2.11) | — | — |
>
> ### 17.3 Zoology configuration (R1, R4, R5)
>
> - **Code and config.** HazyResearch/zoology @`1ad20d193b6113cae1e8f3c655c300d7b4b3f4bb`, config `zoology/experiments/mqar_example_configs/original_mqar_configs.py`, used **exactly as in the file**:
>   - V = 8,192;
>   - training: (64 tokens, 4 KV) × 100K; (128, 8) × 20K; (256, 16 / 32 / 64) × 20K each;
>   - test: (64, 4 / 8 / 16), (128, 32), (256, 64), (512, 128), (1024, 256), 1,000 each;
>   - batch 256 (test 32); 32 epochs; LR ∈ logspace(−3, −1.5, 4) for R1;
>   - a 2-layer `TransformerBlock` with a BaseConv mixer before the sequence mixer;
>   - attention: MHA, 2 heads, dropout 0.1;
>   - GDN: 2 heads, `use_gate=False`, short conv 4;
>   - early stopping at validation accuracy > 0.99, as the file has it;
>   - wandb disabled; `cache_dir` = `data/zoology/`.
> - **Logged shims (DEV-01):**
>   - fla's `head_first` keyword dropped;
>   - exact gradient accumulation in micro-batches of 64;
>   - per-epoch resume.
> - **Runs.**
>   - **R1:** 16 runs (done).
>   - **R4:**
>     - `our_fg` d64 at lr 1e-2;
>     - `our_fg` d128 at lr 1e-3;
>     - `our_gdn` d128 at lr 1e-3;
>     - `our_gdn` d256 at lr 1e-3;
>     - config-A d128 and d256 at lr 1e-3 (exploratory);
>     - production F_g (QK-RMSNorm, RoPE θ = 5e5, headwise gate; 2 heads × 64) at d128 and lr 1e-3, in fp32 and in bf16 autocast.
>   - **R5:** d64, lr 1e-3, seeds (`seed`, `data_seed`) ∈ {(123, 123), (124, 124), (125, 125)}, each run for both Zoology's GDN and `OurGdnZoology` with its forward under bf16 autocast.
> - **The v1.0 cut rule is retired**, because S1(a) finished at ≈ 1.75 GPU-h. Harness runs use the collapse-abort rule in §17.6.
>
> ### 17.4 Verification tests added by DEV-01 (gate G-RV; G0 is unchanged)
>
> | ID | Test | Pass threshold |
> |---|---|---|
> | VT-21 | fla's Triton `causal_conv1d` vs `causal_dwconv_reference` (pure PyTorch). Channel counts of GDN-A's q/k/v paths and of F_gc, at D384 and D512; T ∈ {37, 300, 4,096, 32,768}; packed `cu_seqlens` with offsets that are not multiples of 64; `initial_state` on and off; SiLU on and off; fp32 and bf16; forward and backward | fp32: max abs ≤ 1e-5 (forward), ≤ 1e-4 (gradients). bf16: relative RMS ≤ 1e-2. Zero cross-document leakage. Final conv state matches the reference |
> | VT-22 | Full-model transplant at the lengths we evaluate. C12 ↔ `Qwen3ForCausalLM` and C0 ↔ fla `GatedDeltaNetForCausalLM` (VT-09 mappings) at T = 4,096 in fp32 and T ∈ {16,384, 32,768} in bf16. C3 packed (16K + 16K) vs each document alone, at 32K in bf16 | 4K: relative max ≤ 1e-3. 16K/32K: relative RMS ≤ 2e-2, and the RMS error over the last 1K positions ≤ 2× that over the first 1K (no argmax criterion: random-init logits are near-uniform). Packed vs alone: relative RMS ≤ 1e-2 |
> | VT-23 | The trainer vs an independent ≈ 60-line reference loop written from the text of §7.1–7.2: D384 C3, fp32, audit mode, identical batches, 50 steps, gradient clipping exercised | Step-1 loss: relative difference ≤ 1e-6. Steps 2–10: per-step loss and pre-clip gradient norm within 1e-5 relative; steps 11–50 within 1e-3. Final parameters: relative L2 difference ≤ 1e-3. Identical weight-decay groups |
> | VT-24 | `InductionLM` (order 6, p_hit 0.9) on every R1, R-load, R3 and R-dist item file at 16K and 32K (test CS and MS; dev DG and FD), with M-hop files as a negative control | Retrieval: EM_tf = 1 and logp_gold = 4·ln 0.9 on every query. M-hop: EM_tf < 0.5 |
> | VT-25 | Zoology ↔ our mixers by weight copy, inside Zoology's `LanguageModel`. GDN at d ∈ {64, 128, 256}, with ours reproducing Zoology's bf16 casts through a test flag. Attention at d ∈ {64, 128}, with Zoology's biases zeroed and dropout off | GDN: logits relative RMS ≤ 1e-2, gradients ≤ 2e-2 (bf16 level). Attention: logits ≤ 1e-5, gradients ≤ 1e-4. Ours in fp32 vs Zoology: reported only |
>
> ### 17.5 RP-2 (bug detector, not a gate): amended
>
> - **LM-3 clause (audit trigger).**
>   - *Primary reading:* FEAS end-of-Stage-A checkpoints (k ∈ {0, 12}, seeds 101 and 102). Repeat-token loss at Δ ∈ [1K, 4K), on dev-split documents, must be lower for k = 12 than for k = 0 in both seeds.
>   - If k = 12 ≥ k = 0 in both seeds → audit data, masking and position encoding before S3.
>   - If the two seeds disagree → read the S3 final checkpoints at η* (secondary).
>   - If R8 shows k = 12 without induction at 0.15B tokens, the S3 reading is labelled "pre-induction" and triggers nothing.
> - **Loss-shape clause (descriptive only).**
>   - LM-2 loss at η* for k ∈ {0, 3, 12} (S3) and k = 1 (R9) is classified with the 0.01-nat equivalence margin of §11.7 as U, D, G or F, plus a hybrid-penalty flag (plan §2.9).
>   - It is reported beside Mamba-2 Table 2 and the delta-rule anchors, and triggers no audit.
>   - The k = 1 run never enters the η* rule.
> - A deviation never triggers protocol tuning.
>
> ### 17.6 Firewall, seeds and retry policy
>
> - **RNG.** The stream `repro` (entropy 31337) is used for every reproduction item and every document sample. Reproduction items never come from the eval_test (20261109) or eval_dev (77) generators. VT-24 reads the frozen item files with a model-free oracle only.
> - **Seed indices.**
>   - 300–399 for reproduction training runs: R6 = 300; O2 = 301–304.
>   - By design, R9 uses 100 (paired with S3) and O3 uses 103.
>   - Zoology runs use Zoology's own seeds: R1 and R4 use 123; R5 uses 123–125.
> - **Retry.**
>   - An infrastructure failure (Xid, watchdog, OOM, host OOM) → resume from the last checkpoint. Harness runs get at most 6 attempts; training runs follow D031.
>   - A gate run whose window contains an infrastructure event is repeated in full.
>   - No run is repeated after its verdict is read, unless this section or the plan says so.
>   - **Collapse-abort for harness runs:** stop when validation accuracy has been below 0.05 for 3 consecutive epochs after epoch 2, and record the run as "collapsed".
> - **Tooling.** No new packages go into `.venv` before G5. Diagnostics that need lm-eval or RULER use `.venv-repro`, and their results are labelled "original code + N shims".
>
> ### 17.7 Outcomes, reporting and review
>
> - Outcomes are PASS, FAIL or INCONCLUSIVE, as defined per item in the plan. An INCONCLUSIVE outcome never blocks a gate.
> - **Ledger:** `reports/reproduction/results.md`. For each item it records:
>   - the claim and its source;
>   - the command and commit;
>   - how the tolerance was derived;
>   - per-seed results, with paths to the raw data;
>   - the outcome and the diagnosis history;
>   - the implications for `competing_explanations.md`.
> - A written reproduction review (`reports/reproduction/review.md`), ending in go / fix / re-scope, takes place before G5.
>
> ### 17.8 Gates
>
> G-RP (thresholds unchanged) and G-RV are defined in §13.3.

---

## B. §13.1 Order of execution: replacement for items 2–5

> 2. **S1 reproduction programme, pre-FEAS part (§17):** R1 (done) → R2 → R3 (VT-21…VT-25) → R4 (G-RP(b)) → R5 → R6 → R7 → **G-RV**.
> 3. **S2 FEAS-384.**
>    - R8 and R10 run on its checkpoints. R8's induction-timing reading must be available before S3. RP-2's LM-3 clause is read on the end-of-Stage-A checkpoints.
>    - FEAS-512 follows only if required (U-01). If FEAS-1 failed at 32K, run O1 first.
> 4. **S3 LR sweep (G-LR)**, then R9 (k = 1 at η*), then R8 on the S3 final checkpoints.
> 5. **Reproduction review**, then the **G5 freeze** (contents as v1.0 item 5).
>
> v1.0 items 6–12 are renumbered; their content is unchanged.

---

## C. §13.2 GPU-hour budget: changed rows

| Stage | Content | D384 path | D512 path |
|---|---|---|---|
| S1 | Reproduction programme R1–R10 (§17). R1 ≈ 1.75 h spent; R2–R9 ≈ 3.3–4.3 h new; R10 ≈ 0.3 h. Budgeted at the upper bound | **6.4** (was 3.0) | **6.4** (was 3.0) |
| **Tier A** | Decision point | **89.1 (106.9 with 20%)** (was 85.7 (102.9)) | **147.3 (176.8)** (was 143.9 (172.7)) |
| **Minimum H1-falsification experiment** | S0; pre-FEAS reproduction items R1–R7 (≈ 5.3 h; replaces "reduced S1 (1.5 h)"); FEAS; LR k ∈ {0, 12} × 3; core k ∈ {0, 3, 12} × n₀; seed-0 TEF-4,000 and 2× branches at {0, 3, 12}; retrieval-only evaluation | **57.2 (68.6)** (was 53.4 (64.1)) | **97.4 (116.9)** (was 93.6 (112.3)) |

- **Optional items** O1–O5 (≤ 4.7 h in total) run only on their triggers. They are charged to contingency, which v1.0 allows for "pre-registered triggers".
- **Pricing basis.**
  - Zoology runs: the S1(a) timings.
  - LM-training items (R6, R9): S0's contended medians, `runs/bench_train_step/` (D384 4K: C0 70.1K, C3 61.4K, C12 52.7K tok/s).
- All other rows are unchanged until the clean G0 throughput measurement (R2(e)) is made, at which point D041's reopen criterion applies.

---

## D. §13.3 Gates: changed and new rows

| Gate | When | Pass criteria | If it fails |
|---|---|---|---|
| **G-RP** (amended wording; thresholds unchanged) | End of R4 | (a) As v1.0: passed 16/16 on 2026-11-09. (b) As v1.0 (our mixers within ±0.03 of Zoology's at matched d_model and LR), **evaluated only after VT-25 passes**. Both are reported with the per-cell table and its minimum detectable defect (±0.03 overall = ±0.21 on the 1024/256 cell). Criteria use re-run numbers only (I-20) | As v1.0: debug the mixers or the harness; no LM training until it passes |
| **G-RV** (new, DEV-01) | End of the pre-FEAS reproduction programme | All of: G0 passed; G-RP (a) and (b) passed; VT-21…VT-25 pass; R5 not FAIL; R6 not FAIL; R2 (a) and (b) live, and R2(c) recorded (longest kernel < 1 s, or a logged mitigation). **R7 must be complete before any FEAS verdict is computed.** An R7 failure that implicates `eval.scorer` blocks FEAS verdicts until it is fixed | Diagnose under §17 rules. Fix only defects that a test isolates, then rerun the affected items. **No FEAS until G-RV passes**, or until the PI logs a decision row that accepts a named residual risk |
| **Reproduction audit triggers** (new, DEV-01) | Continuous, S2–S3 | Each of the following triggers an audit: in R8, both induction signals fail for k = 12 at the end of Stage A; RP-2's LM-3 clause fails (§17.5); an R10 within-window or post-Stage-B check fails | Audit before the next stage; PI decision row; never protocol tuning |
| **PI escalations** (new, DEV-01) | Before S3 / before G5 | (1) R8 shows k = 12 forming induction only after 0.15B tokens → decision row before S3: run S3 as frozen with η* labelled "selected on pre-induction models", or extend S3. Task metrics are never used for this choice. (2) R9's k = 1 run diverges at η* (G-div) → decision row before G5 on k = 1's LR, using §7.2's per-configuration rule | — |

The stop rules are unchanged.

---

## E. Consequential edits

1. **§2, "Predictions written down now": annotation appended to the Wang prior line.**

   > *Annotation (DEV-01; PI-VER 2026-11-09):*
   > - *The five GDN recall values were re-verified against [2507.06457v2](https://arxiv.org/html/2507.06457v2) Table 7 [read-section].*
   > - *The released checkpoints place one attention layer per r layers. In both the 1.3B and the 340M "3-1" models, attention sits at layers {2, 5, 8, …, 23}, i.e. 8 of 24 layers, with 16 heads. The 1.3B "24-1" model has attention at {23}, with 32 heads. All of them use 8 KV heads and RoPE θ = 1e4, and the GDN configurations have `max_position_embeddings` 2,048 [code-inspected, HF `config.json`].*
   > - *Neither the paper's "r:1 (linear : full)" (App. D) nor its "roughly one quarter" (§4.3) describes the released 3-1 checkpoint. Read the row as attention in k/24 ∈ {0, 1, 2, 4, 8} layers.*
   > - *The training context is 4,096 tokens in v2 and 2,048 in v1 (Limitations).*
   > - *This remains a prior only (I-23).*

2. **§8.1, stream table: new row.**

   > `repro` | entropy 31337, keys (R-id, task, level, T) | Constant

   **Seed-index assignment: appended.**

   > Reproduction programme (§17.6): 300–399 (R6 = 300; O2 = 301–304). R9 = 100 (paired with S3). O3 = 103. Zoology's own seeds (R1 and R4: 123; R5: 123–125).

3. **§8.2, S1 row: replaced.**

   > S1 | 30 Zoology-harness runs (16 done, 8 in R4, 6 in R5; +4 if R5's bimodality rule fires); 2 known-answer runs (R6); 1 S3-recipe run (R9, after S3); evaluation-only items R7, R8, R10 | Same

4. **§15, Tier A.** "S1" becomes "S1 reproduction programme (R1–R9, §17)". R10 is should-tier. O1–O5 run on their triggers only. Tier C #12 is unchanged.

5. **§16: note appended after the table.**

   > VT-21…VT-25 (DEV-01) are specified in §17.4. They run with the S0 test suite but gate **G-RV, not G0**; G0's definition is unchanged.

6. **§20, Sources: rows added** (all [read-section] or [code-inspected], PI-VER 2026-11-09):
   - Pythia [2304.01373v2](https://arxiv.org/abs/2304.01373), App. G.1, Tables 7–8;
   - Mamba-2 [2405.21060v1](https://arxiv.org/html/2405.21060v1), §9.1, §9.2.3, Table 2;
   - Waleffe et al. [2406.07887v1](https://arxiv.org/html/2406.07887v1), §4.1;
   - Gated DeltaNet [2412.06464](https://arxiv.org/html/2412.06464), Table 3;
   - Olsson et al. 2022: Key Concepts, Arguments 1–2, Model Analysis details;
   - Wang et al. 2507.06457v2, Tables 2, 3, 5, 6, 7, App. D, §4.3 and Limitations (this upgrades the v1.0 tag "per A1, A2, B2");
   - HF configs and SHAs of `m-a-p/*`, `EleutherAI/pythia-{70m,160m}`, `EleutherAI/lambada_openai`, `Qwen/Qwen3-0.6B-Base`;
   - lm-eval @`d6de8164` LAMBADA task definitions and `acc = is_greedy`;
   - the Zoology README figure (viewed) and `mixers/gated_delta_net.py` L246–250;
   - modded-nanogpt @`4ea6b93` (80-rerun SD 0.0019).

---

## F. What does not change

- G-RP's thresholds.
- The research questions, H1–H6 and their tests, the confirmatory and secondary families, the primary cell, and the G-REF, FEAS, G-FULL and G-LR thresholds.
- I-20 and I-23.
- The LR rule: R9's k = 1 run is excluded from it.
- The TEF, the evaluation suites and the item files, unless VT-24 finds an item defect. In that case the items are regenerated before G5 under a decision row.
- Every budget row not listed in part C.
- No code is changed to hit a published number.

---

## G. S1(a) close-out record (logged under DEV-01)

1. **Outcome.** G-RP(a) PASS on 16 of 16 cells, by best-over-LR final overall accuracy [MEAS-S1]:

   | Model | Overall | 1024/256 cell | State |
   |---|---|---|---|
   | attention d64 | 1.0000 | — | — |
   | attention d128 | 0.9999 | — | — |
   | GDN d128 | 0.9892 | 0.925 | 67,072 |
   | GDN d256 | 0.9938 | 0.957 | 265,216 |

   GDN is non-decreasing in d. The best LR is 1e-3 at both GDN widths, which is the lower edge of the grid.
2. **Cost.** ≈ 1.75 GPU-h: 1.48 h of completed runs, ≈ 0.24 h of crashed attempts and ≈ 0.05 h of smoke runs.
   - The v1.0 cut rule did not fire: the first run took 16.5 s, which projected 0.07 h.
   - Lesson: price from the slowest family, and treat early-stopping run times as bimodal (plan §1, rule 13).
3. **Deviations.**
   - fla's `head_first` keyword is dropped (fla 0.6.0 removed it; `head_first=False` is the layout fla 0.6.0 always uses).
   - Exact gradient accumulation in micro-batches of 64 (unit-tested: `test_s1_microbatch.py`). Only the dropout RNG stream differs, which matters for attention only.
   - Per-epoch resume. No reported run was resumed.
4. **Infrastructure.**
   - `gated_delta_net-d128-lr1.0e-02` failed twice with `cudaErrorLaunchTimeout`. The kernel log shows the RC watchdog and Xid 8 at 18:16:50 and 18:25:42. An Xid 31 occurred at 18:03:09 in pid 799092, which was not identified.
   - Attempt 3 ran 22:13–22:26 with only CARLA beside it, as a fresh run from epoch 0 (775 s), and scored 0.9877.
   - That attempt was launched after the verdict had been computed, by an uncommitted change to `run_queue.py` (6 attempts, per-epoch resume). The verdict is unchanged.
   - **Actions:** the S1 owner commits `run_queue.py`; the retry policy is now declared (§17.6); R2 is in force.
5. **Divergence.** GDN at lr 3.2e-2 first learned (best epoch 0.969 at d128 and 0.948 at d256), then collapsed to 0.0050 at both widths. This is reported as observed.
6. **"Need to install causal_conv1d".** The message comes from Zoology's Mamba import probe (`model.py` L270–277) and has no effect on these runs. The real gap, fla's conv against a reference, is VT-21.
7. **Precision.** Zoology's GDN casts q, k, v, β and g to bf16 (`mixers/gated_delta_net.py` L246–250). This is recorded as a residual difference for R4. Production precision is tested in R5, and identity at matched precision in VT-25.
8. **Wang prior annotation:** part E.1.

---

## H. Decision-log row

The row appended to `reports/decisions.md` (table format) is reproduced here **without** the SHA-256 of this file, which can only be computed after the file is written. The row carries the SHA-256 of both files.

> DEV-01 | 2026-11-09 | **Logged deviation (pre-G5): S1 expands into the reproduction programme R1–R10** … (full text in `decisions.md`).

---

## I. Owners and files outside `reports/` (requests only; this amendment edits none of them)

| Owner | File(s) | Change |
|---|---|---|
| Configs owner | `configs/experiments/plan.yaml` | S1 entry → programme R1–R10, G-RV, DEV-01 reference. Run YAMLs for R6 and R9 |
| S1 owner | `scripts/s1_zoology/{run_queue.py (commit), run_one.py, test_s1_transplant.py}`; `reports/s1_reproduction.md` | R1 close-out, VT-25, R4, R5 |
| S0 models | `tests/test_vt21_conv_reference.py`, `tests/test_vt22_long_transplant.py`; profiling script | VT-21, VT-22, R2(c) |
| S0 training / runner | runner / `training/gpumon.py` (`infra_events`); `tests/test_vt23_reference_loop.py`; `scripts/g0_throughput.py` run | R2(b), VT-23, R2(e), R6 support |
| S0 evaluation | `tests/test_vt24_oracle_long.py`; ICL helper; LM-4(b) gap-0 items | VT-24, R8, R10 |
| Scripts owner | new `scripts/repro/` | R7 wrapper and LAMBADA item writer; R6/R8/R10 drivers if configuration alone is not enough |
| Orchestrator | `reports/STATUS.md`; GPU queue | R2(a) one-queue policy, effective immediately |

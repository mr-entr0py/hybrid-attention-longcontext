# Project timeline

All times are IST. Each entry gives what was done, who did it (which agent role), what came out of it, and the commit or document that holds it. The live state is in `reports/STATUS.md`; finer-grained history is in `reports/status_log.md`.

---

## 2026-11-07: Repository created
- `d48660e`: README with the project idea and goal: measure how the ratio of linear to full attention layers affects long-context capability, and find how little full attention is enough.

## 2026-11-09: Framing, research, design and review (Stages 1–3)

**12:00: Framing** (`a29fd44`)
- README extended with the abstract, RQ-A…D and H1–H6.
- Repository skeleton, `.gitignore`, and a decision log with D000 (the repo is the project's memory) and D001 (hardware).
- Hardware inspected: RTX 5080 16 GB, sm_120, driver 595.99. At the time it was shared with a CARLA simulator using about 6 GB.
- Environment: torch 2.11.0+cu128 installed and smoke-tested (bf16 forward/backward and SDPA at 4K).

**12:05–15:35: Research and review workflow.** Twelve agents, about 4.1M tokens, 3.5 h.

1. **Research.** Five specialist agents ran in parallel. Each wrote a report and a YAML literature DB.
   - `01_nemotron_h.md`. Full read of Nemotron-H, plus Waleffe et al. 2024 (the 8B Mamba-2 hybrid study, which has the only controlled attention-fraction sweep in that line of work, U-shaped on loss), plus the released configs and Megatron code.
     - Nemotron-H inherits its ~8% attention share; its Transformer baseline changes depth and position encoding at the same time.
     - The "Nemotron Edge" paper the user mentioned was identified as most likely Nemotron-Flash. Its result: one full-attention layer plus sliding windows drops NIAH at 32K from 0.69 to 0.32.
   - `02_architecture.md`. Linear attention, RetNet, GLA, Mamba-2, DeltaNet, Gated DeltaNet, RWKV-7, plus hybrids (Jamba, Griffin, Zamba, Samba, MiniMax-01, Qwen3-Next, Kimi Linear, Hymba, Jet-Nemotron).
     - Evidence table of every ratio and placement result found.
     - Recommendation: Gated DeltaNet as the primary linear mixer.
     - Main confound: position encoding (NoPE).
   - `03_evaluation.md`. RULER, NIAH, BABILong, LongBench, HELMET, MQAR, phonebook.
     - Natural-language benchmarks score at the floor for 20–60M models.
     - The recommended suite uses cloze and log-prob scoring after an identical synthetic exposure.
     - RULER's shortcuts and inexact lengths are documented.
   - `04_methodology.md`. Variables, matching (parameters and tokens), seed strategy, statistics (paired, hierarchical bootstrap, Holm), gates, budget, determinism, logging and plotting standards.
   - `05_implementation.md`. Verified on this GPU: torch 2.11, triton 3.6 and fla 0.6.0 run Gated DeltaNet forward and backward on sm_120, including varlen.
     - Not available: `mamba_ssm`, and the external FlashAttention wheels.
     - Dataset and tokenizer recommendations.
2. **PI consolidation.** `literature_review.md` (research gap, novelty stated honestly), the merged `literature_db.yaml`, and `initial_proposal.md`, which proposed a 7-point sweep at 49.4M with Gated DeltaNet.
3. **Two adversarial reviewers**, A (scientific validity) and B (engineering validity).
   - Round 1: each critiqued the proposal and wrote an independent design (`review_{A,B}_round1.md`). B ran real GPU measurements.
   - Round 2: each critiqued the other's design (`review_{A,B}_round2.md`).
4. **Debate moderator.** `debate_record.md` resolved I-01…I-29, kept U-01…U-16 open with the experiments that will settle them, and added CPU power simulations.
5. **Protocol frozen.** `experimental_protocol.md` v1.0, `configs/experiments/plan.yaml`, decision rows D002–D044, and `competing_explanations.md`.
   - Committed as `1cd75f4`. Planning and reviewer scripts archived in `scripts/`.

**15:37: Package skeleton.** `pyproject.toml`, `src/hybridlc/` and `INTERFACES.md`, the module contracts that let the builders work in parallel.

**15:40–22:10: S0 infrastructure build.** Six parallel builders, interrupted twice by usage limits and resumed from cache.
- **Models.** F_g and F_gc attention, GDN-A/E/nc and GLA wrappers, HybridLM with prefill, decode and continuation scoring, InferenceCache, chunked CE with z-loss, and analytic costs that reproduce every row of the protocol tables.
- **Data.**
  - Downloads: FineWeb-Edu shards 000–006 (15.1 GB) and PG19 (7.1 GB), pinned and checksum-verified.
  - 16,384-entry BPE tokenizer, deterministic (retraining gives a byte-identical file).
  - 17 GB token stores: 5.0B FineWeb-Edu tokens and 3.1B PG19 tokens.
  - StageStream loader, a pure function of (seed, stage, step) running at more than 5M tok/s.
  - Found and fixed: about 12.7K held-out documents that duplicated training text.
- **Evaluation.** All generators (R1, R-load, R3, R-dist, M-hop, A-freq, A-count, C0/C2 controls, Tier-0 LM-1…4), the scorer and the TEF stream.
- **Training.** Configs, schedules, AdamW parameter groups, atomic checkpoints with RNG state, G-div restarts, registry and runner queue.
- **Analysis.** Every pre-registered test, the bootstrap, verdict rules and plot style. VT-18 recovers effects planted in synthetic data.
- **R1 Zoology MQAR** (`scripts/s1_zoology/`). 16 runs with Zoology's own code, detached queue 17:35–21:00.
  - Attention ≥ 0.9997.
  - GDN 0.989 (d128) and 0.994 (d256).
  - Criterion (a) met.

**23:18: Reproduction programme designed.** Five agents: a pipeline-validation proposer, a phenomenon-replication proposer, two critics and a PI decision.
- Output: `reports/reproduction/reproduction_plan.md` (R1–R10, O1–O5), amendment DEV-01, and the new gate G-RV that opens the pilots.

## 2026-11-10: Integration, verification, reproductions, rulings, audits

**05:30–13:15: S0 integration and adversarial verification**
- The integrator wired all modules together and generated frozen dev and test item files (1.3 GB, SHA-256 manifest). It ran VT-13 determinism and resume tests (kill -9 mid-accumulation and during a checkpoint write), warmed and archived the Triton cache, and ran an end-to-end smoke test.
- **Three rounds of independent verifiers**, one on models and evaluation, one on data, training and statistics, each followed by a fixer. Real defects found and fixed:
  - the no-filler multi-hop control was unshuffled, so trivially solvable;
  - TEF replay let attention and recurrent state cross document boundaries;
  - the "still closing" rule was missing;
  - several minor issues.
- Gate reports `reports/gates/G0.md` (conditional) and `G-RP.md` (PASS). Proposed rows D045–D057.

**05:40: Idle-GPU training benchmark** (`experiments/bench/train_step_idle_20261110*.json`)
- D384: 127–138K tok/s at 4K; 107K / 98K / 78K at 32K for k = 0 / 3 / 12.
- D512: about 98K at 4K; 79K / 72K / 56K at 32K.
- Peak memory ≤ 5.8 GiB.

**06:10–12:50: Reproductions R2, R3, R4, R5, R7, then an independent audit**
- R2: longest 32K kernel 25.2 ms (PASS). The single-queue rule was not in force, so it is not passed as registered.
- R3: VT-25 PASS. VT-21…24 not passed as registered (uncalibrated tolerances; one genuine generator flaw).
- R4: PASS (Δ ≤ 0.0004).
- R5: PASS (0.296 → 0.925 → 0.957; Δ = −0.018).
- R7: GDN-340M PASS (0.3435). Pythia fp32 not passed as registered; the published fp16 anchors reproduce exactly.
- The audit (`reproduction/audit_R2-R7.md`) recomputed every result from raw files and confirmed every verdict.
- Git hygiene: GBs of artefacts had been swept into WIP commits. The unpushed history was rewritten (3.5 GB → 72 MB) and large artefacts are now git-ignored.

**13:20–15:20: PI rulings workflow**
- Draft rulings, then an adversarial review raising 14 objections (all accepted), then FINAL rulings: `reports/pi_rulings_2026-11-10.md`.
- Decision rows D058–D091 and DEV-02…04. 14 new hashed pre-registrations in `experiments/prereg/`: O4 soak v2, VT-24r v2, VT-22b, VT-23b, VT-13r, R6 v2, FEAS-PREC v2, DTS-10 acceptance v2, R2-LIVE v2, and others.
- Key rulings:
  - No failed criterion is relabelled. Each becomes a named residual risk or a new pre-registered check.
  - G0 is "NOT PASSED as registered", with deviation DEV-02.
  - New requirement R-ATT: an fp32 reference attention path that fits at 32K.

**13:41–14:52: Cost-benchmark session 1** (`reports/cost_benchmarks_session1.md`)
- 1,678 timed series, 9 configurations × 2 scales.
- State bytes match the analytic values to the byte.
- D384 prefill at 32K: 52 ms (k=0) vs 77 ms (k=3) vs 154 ms (k=12).
- Decode at k = 0 is flat in context length.
- Largest batch at 32K: 5,185 vs 56 vs 14.
- P003: the "activity reading" of idle compliance was adopted.

**15:18 onward: Continuation session**
- Continuation audit, `decision_log.md` (P-rows), the results ledger, the registry CSV and `knowledge/`.
- P004: a duplicate workflow instance was detected and stopped. One writer per tree.
- P001: hold on expensive GPU runs until the O4 soak localises the Xid-8 faults. Seven events so far; the display's watchdog is active on this GPU.
- **Six specialist audits** (report-only): experimental design, literature, implementation, evaluation and statistics, adversarial scientific review, and compute schedule.
  - No flaw is fatal to the narrow confirmatory claim.
  - Power is lower than planned (seed SD 4.6–5.6 pp, not 2–3 pp).
  - The H3 null is unreachable.
  - The reference-handicap blind spot and per-length reporting need pre-G5 fixes.
- Implementation of the rulings in progress:
  - GPU lock and kernel-event tagging wired in;
  - VT-24r generator fix and item regeneration;
  - FEAS decision code (`analysis/feas.py`, `scripts/feas_decide.py`);
  - FEAS-PREC precision passes, technical blinding, separate Triton caches, near-miss logging.

## 2026-11-11: Consolidation
- Session continued after a usage limit. Work in progress committed.
- Triton caches from the audits removed from tracking.
- `requirements-lock.txt` added.
- README rewritten as a full project reflection; `docs/TIMELINE.md` and `docs/METHODOLOGY.md` added.
- Commit authorship normalised to the repository owner's GitHub no-reply address before the first push.

## What comes next
1. Finish implementing the rulings.
2. Run the O4 soak, VT-24r/22b/23b/13r and R6.
3. An independent auditor decides G-RV.
4. **S2 FEAS** pilots (D384, k ∈ {0, 12}, seeds 101–102) choose the scale, Stage-B budget, SSMax and TEF mix.
5. S3 LR sweep, then G5 freeze (items, configs, analysis code, Triton cache, lock file).
6. S4–S6: core runs k ∈ {0, 1, 3, 12} × seeds, controls, frozen evaluation, verdicts on H1/H1-min/H3/H5a.
7. Tier B (k = 2 and 6, placement, GDN-E), analysis plots P1–P18, final report.

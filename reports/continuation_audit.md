# Continuation audit: 2026-11-10 ~15:40 IST

**Author:** orchestrator/PI (Opus 5.5).

**Method:** everything below was checked against files, logs, git, processes, `nvidia-smi` and `journalctl -k`. Nothing is taken from status labels alone.

## 1. Repository

- **Branch and commits.** On `main`, HEAD `7bee6af`. 29 local commits, none pushed. The `ninjayush` GitHub account has pull-only access.
- **Working tree.** 2 uncommitted paths. These are the PI-rulings workflow's in-progress implementation edits, which belong to the active workflow.
- **`.git` size.** 83 MB. Local unpushed history was rewritten on 2026-11-10 to purge item and checkpoint blobs. `.gitignore` now excludes `*.npz`, `*.npy`, `*.pt`, `*.tar`, `*.safetensors`, `hf_cache/` and `ckpt/`.
- **Disk.** 348 GB free of 901 GB. Usage: data/ 39 GB, experiments/ 14 GB, runs/ 2.7 GB.

## 2. Completed milestones (verified)

| Milestone | Evidence | Verdict |
|---|---|---|
| Literature: 5 agents, DB and review | `reports/literature/*` (190 KB DB) | done |
| Proposal, 2-reviewer debate, frozen protocol v1.0 | `reports/experimental_protocol.md`, `reports/reviews/` | done |
| Reproduction programme design (DEV-01) | `reports/reproduction/reproduction_plan.md`, `protocol_amendment.md` | done |
| S0 infrastructure | `src/hybridlc/**`, `reports/s0_integration.md`; full suite of 448 passed, 4 failed (VT-21 to VT-24, which belong to G-RV), 4 skipped | done |
| S0 bug fixes, verification rounds 1–3 | `reports/s0_integration.md` §§ "Verification round 1–3" | done; see §5 |
| G0 | `reports/gates/G0.md`: all §13.3 criteria met, subject to PI readings D045–D048 | resolved by PI rulings D045–D057, which are FINAL; the formal G0 status is to be re-stated by the G-RV auditor |
| G-RP (Zoology S1 a/b) | `reports/gates/G-RP.md`, recomputed from raw | **PASSED** |
| R1–R5, R7 + independent audit | `reports/reproduction/results_R*.md`, `audit_R2-R7.md` | R1/R4/R5 PASS; R2, R3 (VT-21 to VT-24) and R7 (Pythia fp32 anchors) **NOT PASSED as registered**, each diagnosed without a code defect |
| PI rulings on G0/G-RV residuals | `reports/pi_rulings_2026-11-10.md` FINAL; D058–D091 in `decisions.md`; 14 batch-2 pre-registrations, hashed before running, in `experiments/prereg/` | done |
| Idle training throughput | `experiments/bench/train_step_idle_20261110_D384.json` and `..._20261110.json` (D512 rows) | done |
| Cost bench session 1 | `experiments/bench/cost_session1/rows.jsonl` (1,896 rows); driver logged "session end" at 14:51 | measurements done; report being written by the resumed agent; idle compliance to be verified |

## 3. Active jobs

- **Workflow `wf_b92a39ce-f17`** (PI rulings → implement → O4 soak + new checks + R6 → G-RV auditor). The Decide phase is cached; the Implement agent is running.
- **Cost-bench agent:** writing `reports/cost_benchmarks_session1.md` only. No GPU use.
- **No detached GPU process** from the project is running.

## 4. Incomplete milestones

- G-RV, which blocks FEAS. Needed:
  - the VT-24 generator fix plus item regeneration, then VT-24r;
  - VT-22b, VT-23b, VT-13r and VT-13p2;
  - R2-LIVE (lock and event tags wired into the runner and trainer);
  - the O4 soak;
  - R6;
  - the DTS-10 FEAS decision code with acceptance tests;
  - FEAS-SMOKE;
  - Triton re-archive.
- Cost bench session 2. It needs a second idle window, on a different day.
- Then, in order: S2 FEAS → S3 LR sweep → G5 → S4–S6 Tier A → Tier B → analysis and report.

## 5. Outstanding bugs and risks

| Item | State |
|---|---|
| M-hop C0/C2 controls unshuffled | Fixed in round 1 (regression tests in `s0_integration.md`) |
| TEF replay cross-document leakage (DTS-01) | Fixed in round 1 |
| "Still-closing" rule (DTS-02) | Fixed in round 2; the pre-FEAS G0-suite re-run is pending |
| A-freq recency shortcut | Inherent to the frozen design. Documented as minor; affects secondary contrasts only |
| Held-out exact duplicates | Fixed (held-out filter), D052 |
| VT-24 value-prefix collisions (7 of 39,800) | Generator fix and regeneration pending (route ii, VT-24r) |
| **GPU Xid faults** | **DISCREPANCY.** STATUS.md and R2 said "no events since 2026-11-09 18:25". `journalctl -k` shows **5 more Xid-8 events on 2026-11-10** (07:56, 08:45, 09:29, 09:46, 09:47), each with an RC watchdog 7 s timeout and a python pid. They fall in the R4/R5 Zoology window, while CARLA was **stopped**. Total: 7 Xid-8 and 1 Xid-31. The CARLA-correlation hypothesis is weakened. Leading candidate: a single long kernel (> 7 s) in the Zoology harness path, possibly the eval or FFT long-conv; our own model's longest kernel is 25 ms. The display shares the GPU. **Rule applied: no expensive run until O4 localises the cause** (O4 is pre-registered in `experiments/prereg/O4_soak.v2.yaml`) |
| GPU sharing | CARLA restarted at ~15:00: two UE4 instances, 7.6–11 GB, 67–98% util. Free VRAM is about 5–7 GB against our 7 GB cap. Training at reduced margin risks OOM. Timing benchmarks are invalid while CARLA runs |

## 6. Unresolved PI or user decisions

1. Budget ceiling (asked 2026-11-09; no answer). Default: Tier A, then ask.
2. Policy if the all-full model fails at both long lengths. Default: pause and ask.
3. Headless GPU operation. Ask only if O4 shows persistent hangs.
4. CARLA schedule. It affects the OOM margin and the cost-bench idle windows.

## 7. Immediate next actions

1. Let `wf_b92a39ce-f17` finish: implementation → O4 → new checks → R6 → G-RV.
2. In parallel, mostly CPU-side, run the six specialist audits the user requested on 2026-11-10 (design, literature, implementation, evaluation, adversarial, scheduling). They are **report-only for `src/`** while the active workflow edits code. Fixes are applied after it ends.
3. Consolidate in a PI meeting, then open FEAS only if G-RV permits.
4. Scaffold the memory files the user requested:
   - `reports/decision_log.md`, a PI-level log that points to the protocol rows in `decisions.md`;
   - `reports/results.md`;
   - `experiments/registry.csv`, a CSV view of `registry.jsonl`;
   - `knowledge/`.

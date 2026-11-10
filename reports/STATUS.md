# Live status (orchestrator notes; update on every check)

## RESUME PROCEDURE (when the user says "continue")
1. `git add -A && git commit -m WIP` to save any partial agent output.
2. Resume every workflow listed under **Active workflows** whose latest notification showed failed agents: `Workflow({scriptPath: <path>, resumeFromRunId: <runId>})`. Finished agents replay from cache.
3. Check the detached jobs listed below (`pgrep`, tail their logs) and restart them with their launch scripts if they died.
4. Append a timestamped entry to this file.

## Active workflows (scripts live in ~/.claude/projects/-home-sedrica-lynx-Downloads-outeropt/73b71b1c-85a4-4704-b75f-7c9b668d4052/workflows/scripts/)
| Workflow | Script | Run ID | State |
|---|---|---|---|
| Cost benchmark session 1 (§10.2), background agent | (Agent, not workflow) | — | launched 2026-11-10 ~13:30; output experiments/bench/cost_session1/, reports/cost_benchmarks_session1.md |
| S0 build, integrate, verify, gate report | s0-infrastructure-and-repro-wf_a4a629b1-75d.js | wf_743ec7bf-78f | DONE: G0 conditional PASS (D045–D057 proposed), G-RP PASS |
| PI rulings → implement → soak + R6 → G-RV gate | pi-decisions-and-grv-wf_b92a39ce-f17.js | wf_b92a39ce-f17 | launched 2026-11-10 ~13:20 |
| Reproduction execution R2/R3/R7 → R4 → R5 → audit | reproduction-execution-r2-r7-wf_fd2b916e-fb5.js | wf_fd2b916e-fb5 | DONE: audit confirms R4 and R5 PASS; R2, R3 and R7 FAIL as pre-registered, none from a code defect |
| Reproduction-program design | reproduction-program-design-wf_b6052608-ded.js | wf_b6052608-ded | DONE: reports/reproduction/reproduction_plan.md |

## Detached jobs
| Job | Launch / resume | Log | State |
|---|---|---|---|
| S1(a) Zoology queue | `bash scripts/s1_zoology/launch.sh` | experiments/s1_zoology/logs/queue.log | all done |

## Log
**2026-11-09 ~18:10 IST**

- CARLA is paused by the user, so the GPU is in use by project jobs only.
- **Detached GPU job:** `scripts/s1_zoology/run_queue.py` (PID 730516). It is reparented to init, runs in its own session and ignores SIGHUP, so it survives agent and session exits. It is resumable: re-run `bash scripts/s1_zoology/launch.sh` and finished runs are skipped. Log: `experiments/s1_zoology/logs/queue.log`.
  - Attention d64: all 4 LRs reach valid accuracy 1.00 after epoch 0 (Zoology's early stop at >0.99).
  - GDN d128 lr1e-3 finished in about 17 min. The remaining GDN runs are in progress.
- **Workflows that a usage limit may interrupt.** Resume them with `Workflow({scriptPath, resumeFromRunId})`; finished agents replay from cache.
  - S0 build/verify: script `~/.claude/projects/-home-sedrica-lynx-Downloads-outeropt/73b71b1c-85a4-4704-b75f-7c9b668d4052/workflows/scripts/s0-infrastructure-and-repro-wf_a4a629b1-75d.js`, run `wf_743ec7bf-78f`
  - Reproduction-program design: script `.../workflows/scripts/reproduction-program-design-wf_b6052608-ded.js`, run `wf_b6052608-ded`
- **Note:** Zoology logs "Need to install causal_conv1d". Check whether its BaseConv falls back to a slower path with identical math, so S1 numbers are still faithful.

**2026-11-09 22:15 IST**

- **Usage limit (resets 22:00):** it interrupted the integrator and verifiers in the S0 workflow and the critic-B and PI steps in the reproduction design. Both workflows have been resumed; finished agents replay from cache.
- **CARLA is running again**, using about 3.6 GB.
- **S0 builders are done:**
  - models: all VT tests pass;
  - data: tokenizer, 17 GB of token stores, 22 VT-11 tests pass;
  - eval, training and analysis code is written.
- **S1(a) Zoology MQAR finished.** The table is in `experiments/s1_zoology/s1a_table.md`.
  - Attention reaches ≥ 0.9997 at both widths.
  - GDN reaches 0.989 at d128 and 0.994 at d256. It is non-decreasing in d, and it diverged at LR 3.2e-2.
  - Criterion (a) looks met; the gate report will confirm it formally.

**2026-11-09 ~23:30 IST**

- The reproduction plan is done: R1 (passed) through R10, plus a new gate G-RV that opens FEAS.
- **Next, after S0 finishes:** run R2 through R7 as a workflow, then compute G-RV.
- **Flag:** two Xid-8 GPU faults happened during S1. R2 should check for a display-watchdog cause (the desktop runs on this GPU). The longest kernel must stay under 1 s.

**2026-11-10 05:34 IST**

- The usage limit (reset 03:10) interrupted the integrator. Partial output, including the items in `experiments/items/` and `s0_integration.md`, is committed as 2fffe8f. The S0 workflow has been resumed. No detached jobs are running.

**2026-11-10 ~06:10 IST**

- The user gave the project the whole PC; CARLA is stopped.
- An idle-GPU throughput benchmark was saved to `experiments/bench/train_step_idle_20261110*.json`.
  - D384: Stage A 127–138K tok/s; Stage B at 32K, 107K / 98K / 78K for k = 0 / 3 / 12.
  - D512: 97–99K, then 79K / 72K / 56K.
  - Peak memory ≤ 5.8 GiB. This is about 1.3–2.4× faster than the contended budget rates.
  - The first D384 pass was contended by our own tests and has been discarded; see the `_D384` file for the clean run.
- Launched the reproduction-execution workflow. It runs in parallel with the S0 workflow.
- **Pending after S0:** R6 (needs the integrated trainer), then gate G-RV, then S2 FEAS.

**2026-11-10 12:20 IST**

- The usage limit (reset 10:30) interrupted the S0 Fixer r3 and gate reporter, and the reproduction auditor. Both workflows have been resumed.
- **Git hygiene.** The WIP commits had swept GBs of item `.npz` files, checkpoints and the HF cache into git. That local, unpushed history was rewritten: `.git` went from 3.5 GB to 72 MB, and no blob in history is above 5 MB. `.gitignore` now excludes `*.npz`, `*.npy`, `*.pt`, `*.tar`, `*.safetensors`, `hf_cache/` and `ckpt/`. Item files stay on disk; their SHA-256 manifests are in git.
- **Reproduction results (pre-audit):**
  - R2: kernel profile PASS (max 25 ms). The single-queue rule and event tagging are not yet enforced project-wide.
  - R3: VT-25 PASS. VT-21 to VT-24 FAIL against their frozen tolerances. The diagnosis says these are uncalibrated tolerances or oracles, not code defects; a PI decision is needed.
  - R4: PASS (Δ ≤ 0.0004).
  - R5: PASS. The d64 phenomenon reproduces (0.296 vs 0.925 vs 0.957), and ours vs Zoology's gives Δ = −0.018.
  - R7: the Pythia anchors FAIL in fp32 (0.354 / 0.227) but reproduce exactly in fp16 (0.330 / 0.185). lm-eval agrees with our scorer item for item. GDN 340M PASS (0.3435).
  - An Xid-8 plus RC watchdog fired during an R4 exploratory run, which triggers O4 (soak). The display runs on this GPU, with a 7 s watchdog.
- **Pending decisions (PI workflow after the audit):** R3 tolerance restatements, the R7 anchor precision, the O4 soak, and enforcing one GPU queue project-wide. Then R6 and gate G-RV.

**2026-11-10 ~13:00 IST**

- The audit confirmed every executor verdict (`reports/reproduction/audit_R2-R7.md`).
- **Waiting on the S0 workflow** (fixer r3, then the gate report). After that, the PI decision workflow decides:
  - R3: restate the tolerances, or fix the VT-24 generator collisions;
  - R7: re-register the Pythia anchors at fp16;
  - R5/D1: accept R5 even though the O4 soak did not run;
  - R2: enforce `gpu_lock` and event tags in the runner and trainer.
- Then R6, then G-RV.

**2026-11-10 ~13:20 IST**

- S0 is done.
  - G0 is a conditional PASS: 447 of 447 VT-01 to VT-20 checks pass. Clean throughput and memory criteria are met; peak VRAM is 6.0 GiB.
  - G-RP: PASS.
- Launched the PI rulings workflow. It covers the D045–D057 proposals and the G-RV residuals, then implementation, then the soak and R6, then an independent G-RV gate.
- **Next, if G-RV passes:** S2 FEAS.

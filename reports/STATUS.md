# Live status (orchestrator notes; update on every check)

## RESUME PROCEDURE (when the user says "continue")
1. `git add -A && git commit -m WIP` to save any partial agent output.
2. Resume every workflow listed under **Active workflows** whose latest notification showed failed agents: `Workflow({scriptPath: <path>, resumeFromRunId: <runId>})`. Finished agents replay from cache.
3. Check the detached jobs listed below (`pgrep`, tail their logs) and restart them with their launch scripts if they died.
4. Append a timestamped entry to this file.

## Active workflows (scripts live in ~/.claude/projects/-home-sedrica-lynx-Downloads-outeropt/73b71b1c-85a4-4704-b75f-7c9b668d4052/workflows/scripts/)
| Workflow | Script | Run ID | State |
|---|---|---|---|
| S0 build, integrate, verify, gate report | s0-infrastructure-and-repro-wf_a4a629b1-75d.js | wf_743ec7bf-78f | resumed 2026-11-10 05:34; integrator (partial output committed 2fffe8f) → verify → gate report |
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

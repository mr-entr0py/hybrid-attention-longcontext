# Live status (orchestrator notes; update on every check)

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

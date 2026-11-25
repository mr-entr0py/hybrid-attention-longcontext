# 004: Environment integrity (R2)

**Status:** not passed as registered (kernel-profile sub-item PASS) · **Date:** 2026-11-10 (05:40–07:25 IST) · **First commit:** `9d4bd98`

## Why
- The display shares this GPU, and the driver's 7-second watchdog kills any kernel that runs too long.
- Seven watchdog/Xid-8 faults occurred during recall runs.
- If long 32K kernels approach that limit, or several jobs share the GPU, runs can fail selectively and bias which configurations complete.

## Spec
Pre-registered in `reports/reproduction/reproduction_plan.md` §2.2.
- (a) Exactly one project GPU job at a time.
- (b) Every run tagged with kernel-log events.
- (c) The longest single kernel at 32K must be < 1 s.
- (d) Identify a process recorded in an Xid-31 event.

## Procedure
```bash
setsid nohup bash scripts/repro/r2c_launch.sh > experiments/repro/r2/launch.log 2>&1 &   # (c), one cell per process
python scripts/repro/infra_events.py --help                                           # (b) kernel-log tagger
python scripts/repro/gpu_lock.py --help                                               # (a) single-queue lock
```

## Results
| Sub-item | Result | Verdict |
|---|---|---|
| (c) 32K kernel profile | Longest warm kernel **25.24 ms** (D512-C12, flash-attention backward). That is 40× under 1 s and 277× under the watchdog | PASS |
| (b) event tagging | The tagger works and recovers every known event, but was not wired into the trainer at the time | not passed as registered |
| (a) single queue | The lock works, but 2–4 project jobs overlapped during the window | not passed as registered |
| (d) Xid-31 process | Narrowed to two candidates; not uniquely identified | recorded |

Raw data: `experiments/repro/r2/`. Report: `reports/reproduction/results_R2.md`.

## Interpretation
Kernel duration rules out the simplest explanation for the watchdog faults. The lock and the tagger have since been built into every entry point (PI ruling D077, "R2-LIVE v2"). A soak test (experiment 010) is pre-registered to localise the remaining faults.

## Status
Not passed as registered for the past window. The fixes are implemented, and re-verification is pre-registered.

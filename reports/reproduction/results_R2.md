# R2 results: environment integrity

**Item:** R2, reproduction plan v1.0 §2.2 (`reproduction_plan.md`, SHA-256 `6bc14af62d7a…8adcebc`, matching DEV-01).
**Executor:** reproduction-execution agent. **Window:** 2026-11-10, 05:40–07:25 IST.
**Code state:** git `110fd9e3` with a dirty tree. Other agents were editing `src/hybridlc/**` concurrently, and this executor did not edit it. torch 2.11.0+cu128, triton 3.6.0, fla 0.6.0, CUDA 12.8, driver 595.99.02 (open kernel module), RTX 5080 16 GB. Each profile JSON stores its own `env_info()`.

## Summary

| Sub-item | Criterion (frozen) | Result | Verdict |
|---|---|---|---|
| (c) 32K kernel profile | longest single kernel < 1 s | **25.24 ms** max over 4 cells × 3 warm phases (D512-C12, flash-attention backward, Stage-B PG19 32K micro-batch). This is 40× under 1 s and 277× under the 7 s RC-watchdog timeout | **PASS** |
| (b) per-run event tagging | live before R4 | The tooling works: it recovers exactly the plan's KLOG facts for 2026-11-09 17:30–18:30, and it tagged all 4 R2(c) runs (0 events, `journal_ok`). It is **not** integrated into the S0 trainer, runner or registry, which needs `src/` edits outside my ownership | **PARTIAL (not live project-wide)** |
| (a) one GPU queue | live before R4 | The tool `scripts/repro/gpu_lock.py` works (6/6 self-tests). The rule is **not observed**: from 05:47 to 07:07, 2–4 project GPU processes ran concurrently without any lock. The R2(c) job waited 77 min for an exclusive GPU | **NOT LIVE (rule violated in practice)** |
| (d) pid 799092 | recorded | Narrowed to a project python process started 18:00:41–18:03:09 IST. Two candidates; not uniquely identified (§5.2) | recorded |
| Xid-8 investigation | (task) | Both Xid-8 pids are S1 attempts 1 and 2. The second watchdog fires exactly 7 s after the CARLA lane runner restarted; the first has no identified external trigger. A display is attached and active, and the RM RC watchdog is active on this GPU (§5.3) | recorded |
| O4 trigger | new Xid/watchdog overlapping a project process | **No** NVRM Xid, RC-watchdog or OOM line since 2026-11-09 18:25:42. This covers the scans to 2026-11-10 07:20, which include today's VT-13 kill -9 run | not triggered |

**Overall R2: not yet passed.** R2(c) PASS and (d) recorded. (a) and (b) are not live across the project. That needs orchestrator enforcement plus S0 integration (§6), not more measurement.

---

## 0. Pre-run readings of ambiguous criteria (written 2026-11-10 05:46 IST, before any R2 GPU run)

The criteria are frozen. Where the text leaves room, my reading is fixed here, before results:

1. **R2(c) "longest single kernel".** It means the longest device-side duration of one CUDA kernel, as recorded by CUPTI through `torch.profiler` (chrome-trace events with `cat == "kernel"`). It does not mean a CPU-side op time or a whole-op time. Memcpy and memset records are reported, but they are not kernels.
2. **R2(c) workload set.** It is {C0, C12} × {D384, D512} × {one Stage-B 1 × 32K training step, one 32K prefill}. The task asks for D512 explicitly, so D512 is in the criterion set; the plan's "if FEAS-512 is reached" condition is overridden by the task. The training step is the production micro-step: bf16 autocast forward, `chunked_ce` with z-loss, backward with per-block activation checkpointing, grad clip, and the fused AdamW step. The prefill is `HybridLM.prefill(ids[1, 32768])` under bf16 autocast (frozen batch size 1 at 32K).
3. **R2(c) document layout.** The primary layout is a single 32,768-token document. This is the worst case for attention cost, and it matches the PG19 32K windows that make up 60% of Stage-B sequences. A real Stage-B micro-batch (`StageStream('B', 0, 32768, 1, 4).get(0, 0)`) is also profiled. Both count toward the criterion.
4. **R2(c) warm versus cold.** The criterion applies to steady-state calls: training after 2 untraced steps, and prefill after 1 untraced call. The first call in a process (`*_cold`, which includes Triton JIT and autotune launches) is reported as a risk note and does not count toward the criterion.
5. **R2(c) contention.** The criterion is evaluated on profiles taken while no other project process holds a CUDA context. These runs use `gpu_lock.py --wait-exclusive`. CARLA is the only foreign load the plan allows. Display and desktop clients are present on this GPU and are recorded. If a profile is taken under project contention, it is labelled as such and does not count.
6. **R2(c) verdict.** PASS iff the maximum over all warm phases of all 4 cells is < 1.000 s. FAIL otherwise; a FAIL triggers the plan's failure tree, and I would not tune anything.
7. **R2(a), (b) "live".** Status is reported in three parts: (i) the tooling exists and works in a demonstration; (ii) the R-items run by this executor use it; (iii) it is integrated into the S0 trainer, runner and registry. (iii) needs `src/hybridlc/**` edits, which this executor may not make. (b) works only if the tagger, run over the known window 2026-11-09 17:30–18:30, returns exactly the KLOG lines cited in the plan: 1 host OOM kill, Xid 31 at 18:03:09, RC watchdog + Xid 8 at 18:16:50, and RC watchdog + Xid 8 at 18:25:42.

---

## 1. R2(c): 32K kernel profile — PASS

**Command.** Run detached and resumable: `setsid nohup bash scripts/repro/r2c_launch.sh`. Each cell then runs:

```
python scripts/repro/infra_events.py run --tag R2c-<scale>-<cfg> --out-dir experiments/repro/r2/infra \
    --interval 2 --wait-exclusive --settle-s 90 --lock-timeout 14400 -- \
  python scripts/repro/r2_kernel_profile.py --scale <scale> --config <cfg> --out-dir experiments/repro/r2/profile
```

The memory fraction was 0.44, the seed was `build_model(seed_index=0)`, and token ids came from `np.random.default_rng(0)`.

**Raw outputs:**
- `experiments/repro/r2/profile/{D384,D512}-{C0,C12}.json`: per-phase max kernel, top-10 kernels, kernel counts, memcpy max, `env_info`;
- chrome traces in `experiments/repro/r2/profile/traces/*.json.gz`;
- logs in `experiments/repro/r2/profile_*.log`;
- per-run infra tags in `experiments/repro/r2/infra/R2c-*.infra.json`.

Longest single kernel (ms):

| cell | pattern | train_warm (single 32K doc) | train_warm_stageB | prefill_warm | train_cold | prefill_cold | longest kernel (warm) | peak alloc GiB |
|---|---|---|---|---|---|---|---|---|
| D384-C12 | `AAAAAAAAAAAA` | 18.76 | 18.77 | 8.17 | 18.76 | 8.23 | `pytorch_flash::flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel` | 2.43 |
| D384-C0 | `------------` | 1.75 | 1.75 | 0.73 | 1.74 | 0.73 | `chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64` | 2.41 |
| D512-C12 | `AAAAAAAAAAAA` | 25.00 | **25.24** | 10.70 | 25.00 | 10.62 | `pytorch_flash::flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel` | 2.94 |
| D512-C0 | `------------` | 1.53 | 1.53 | 0.73 | 1.54 | 0.73 | `chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64` | 2.92 |

- **Criterion:** the maximum over warm phases is 25.24 ms < 1000 ms → **PASS**. The prefill maximum is 10.7 ms, from the flash forward kernel. The longest memcpy or memset record was 0.18 ms.
- **Step totals.** Summed kernel time per warm training step was 741 ms (D384-C12), 319 ms (D384-C0), 1010 ms (D512-C12) and 425 ms (D512-C0). A step is roughly 3,900 kernels, each ≤ 25 ms, so no single launch comes near the watchdog scale.
- **Contention check.** During every profile, `compute_apps_seen` lists only:
  - the profile process itself;
  - pid 124395 (a CARLA-env python, `critscen.demo`, 454 MiB, foreign);
  - pid 293356 (`gnome-remote-desktop-daemon`, 305 MiB).
  No other project process was present. `clocks_event_reasons.active` was `0x0` in every sample. The clock-event counters did not change over any run (SW power cap, thermal and power-brake deltas were all 0 µs). Every run has `flagged=False` and 0 kernel-log events.

**Caveats** (none of them change the verdict):
- *Stage-B layout.* `StageStream('B',0,32768,1,4).get(0,0)` happened to be one PG19 32K window (`n_docs=1`). So the Stage-B phase repeated the single-document worst case on real tokens, and a FineWeb-packed micro-batch was not profiled. Packing splits the same 32K tokens into shorter documents. That lowers the work in each varlen attention launch, and the GDN chunk kernels run over the same total length, so the single-document figure is the upper bound [INF].
- *"Cold" is first call in a process with a warm on-disk Triton cache*, not a truly empty cache. The cold steps launch more kernels (≈ +440 to +650, autotune runs), but their longest kernel equals the warm one. An empty-cache first step was not measured.
- *First attempt lost.* Attempt 1 of D384-C12 crashed after its first profiled step, in my own trace bookkeeping (a relative `--out-dir` passed to `Path.relative_to`). No measurement was recorded. I fixed the bug and reran all cells: `profile_D384-C12.attempt1_pathbug.log`, `launch.log`.

## 2. R2(b): per-run infrastructure-event tagging — tool live, project integration missing

**Tool.** `scripts/repro/infra_events.py`, used two ways:
- as a wrapper (`run --tag … -- cmd`);
- in-process (`InfraTagger` context manager).

It records:
- `infra_events`: `journalctl -k` lines over [start, end + 3 s] matching the plan's regex `NVRM: Xid|RC watchdog|Out of memory|oom-kill`, parsed into kind, xid, pid and name;
- `systemd-oomd` kill lines;
- GPU samples: utilisation, memory, SM clock, temperature, power, every `clocks_event_reasons.*` flag, `display_active` and compute apps;
- the `nvidia-smi -q -d PERFORMANCE` clock-event counters at start and end, and their delta;
- the minimum host `MemAvailable`;
- git commit, dirty count, driver, and the lock wait.

A run with any event is `flagged`. `journal_ok` tells "no events" apart from "journal unreadable".

**Readability without sudo.**
- `journalctl -k` is readable, because the user is in group `adm`, and it shows the NVRM lines.
- `dmesg` is **not** readable ("read kernel buffer failed: Operation not permitted", i.e. `dmesg_restrict`).
- The tool therefore uses only the journal.

Source: `journal_status()` inside every `*.infra.json`.

**Validation against the plan's facts** (pre-registered reading 7):
- Command: `python scripts/repro/infra_events.py scan --since "2026-11-09 17:30" --until "2026-11-09 18:30"`.
- Output: `experiments/repro/r2/klog_scan_20261109_1730_1830.json`.
- It returns exactly:
  - host OOM at 17:31:22 (3 lines; the victim was `antigravity-ide`);
  - Xid 31 at 18:03:09, pid 799092;
  - RC watchdog + Xid 8 at 18:16:50, pid 811688;
  - RC watchdog + Xid 8 at 18:25:42, pid 818052.
- This is also asserted by `test_scan_known_window_matches_klog`.

**Self-tests.** `pytest -q scripts/repro/test_r2_tooling.py` gives **6 passed**: `experiments/repro/r2/tooling_selftest.log`.

**Live use.**
- All 4 R2(c) runs were tagged (`experiments/repro/r2/infra/R2c-*.infra.json`).
- Additional scans:
  - 2026-11-09 18:30 → 2026-11-10 05:45: 0 events (`klog_scan_since_20261109_1830.json`);
  - 2026-11-10 05:00 → 07:20: 0 events (`klog_scan_20261110_0500_now.json`).

**What is missing for "live".**
- The plan requires `infra_events` in **each run's JSON and the registry**, owned by S0 training/runner (`training/gpumon.py`, `scripts/runner.py`).
- `grep -rn "infra_events\|journalctl\|Xid" src scripts` found nothing in `src/` or the S0 scripts (05:4x IST).
- Trainer runs (`runs/<id>/summary.json`, `experiments/registry.jsonl`) are therefore still untagged. S0 runs launched outside this wrapper, such as today's VT-13 production test, carry no tag.
- Integration needs `src/hybridlc/**` edits, which this executor is not allowed to make, so this is handed to S0 (§6).

## 3. R2(a): single GPU queue — tool live, rule not observed

**Tool.** `scripts/repro/gpu_lock.py`.
- An exclusive `fcntl.flock` on `runs/gpu_queue.lock` (override: env `HYBRIDLC_GPU_LOCK`).
- The holder is described in `runs/gpu_queue.lock.holder.json`.
- The kernel drops the lock when its holder dies, so there are no stale locks.
- Interfaces: `run --label X [--timeout s] [--wait-exclusive [--settle-s s]] -- cmd`, `status`, and `with gpu_lock(...)`.
- `--wait-exclusive` also waits until **no other project process** holds a CUDA context. A process counts as project if its name, cmdline, cwd or `VIRTUAL_ENV` points into the repository. This catches agents that do not use the lock.
- Exit code 75 on timeout. The child's exit code is propagated, and SIGTERM, SIGINT and SIGHUP are forwarded to it.

**Self-tests** (4 of the 6):
- two jobs are serialised;
- timeout returns 75;
- SIGKILL of the holder frees the lock;
- the return code propagates.

**Known limit.** If the *wrapper* is SIGKILLed, its child survives and keeps the GPU without the lock (shown in the test). `--wait-exclusive` in later jobs covers this case.

**Detection fix during the run.**
- At 05:49 a sibling agent ran `python -` from the repository cwd with a CUDA context (VT-21 work). The first version missed it, because it only checked the name and cmdline.
- Detection was extended to cwd and `VIRTUAL_ENV` at 05:50, and the launcher was restarted (`launch.log`).
- The check now also catches the `.venv-lmeval` processes of R7.

**Observed compliance.** These are the lock-wait records of the R2(c) job (`profile_D384-C12.attempt1_pathbug.log`, `profile_D512-C0.log`):

| time (IST) | project GPU processes running concurrently, none using the lock |
|---|---|
| 05:47–06:03 | **two VT-13 production trainers in parallel** (`vt13_production_resume.py --parallel-clean`, pids 1603959/1603960, 5.2 GB each), plus a VT-21 heredoc, plus `r7_run.py` (gdn340m, pythia-70m), plus VT-21/VT-25 pytest |
| 06:05–06:23 | VT-21 pytest, `r7_run.py` (several variants), VT-13 run b, `.venv-lmeval` `r7_lmeval_diag.py` |
| 06:24–07:00 | VT-21 pytest, S0 full `pytest tests/`, further S0 processes |
| 07:14–07:18 | `.verify/check_mixers.py`, `.verify/check_model.py`, `ac_check.py` (S0 verifiers) |

The plan's rule is that "no two project GPU processes run during a timed or gate run", with CARLA as the only foreign load. It was broken several times in this window. The R2(c) job got an exclusive GPU only after waiting **4,638 s**, with a further 357 s before D512-C0.

**Contended benchmark file.** The orchestrator's `experiments/bench/train_step_idle_20261110.json` has its **D384 rows measured under contention**:
- pids 1602000/1602001, 5.18 GB each, appear in the rows' own `gpu.processes`;
- D384 C3/C12 4K ran out of memory;
- D384-C0 32K has a median of 0.817 s against a minimum of 0.309 s.

The rerun `train_step_idle_20261110_D384.json` is clean (only 124395, 293356 and itself), and so are the D512 rows of the first file. Any G0-T or R2(e) use should take D384 from the rerun only.

## 4. R2(e) G0-T

R2(e) is not part of this task, and no measurement was made here. The only fact recorded is the contention finding in §3.

## 5. Xid investigation

### 5.1 Timeline

Sources:
- `experiments/repro/r2/env/klog_20261109_1700_1830.txt`;
- `env/s1_queue.log` (copy of `experiments/s1_zoology/logs/queue.log`);
- the S1 attempt logs;
- `env/carla_lane_pause_log.txt` (copy of the CARLA lane's `pause_log.txt`, read-only);
- `env/carla_lane_systemd.txt`.

| time (IST, 2026-11-09) | event |
|---|---|
| 17:31:22 | Host OOM kill of `antigravity-ide`. CARLA UE4 servers were in the task list; they were SIGKILLed at 17:32:52 with their scope |
| 17:37:57 | CARLA `critscen-lane.service` started |
| 17:53:28 | S1 `gated_delta_net-d128-lr3.2e-03` started (completed rc=0 at 18:10:52) |
| 17:58:26 | S0 `vt13_reduced_C3` GPU pytest started. It passed after 715.8 s (`runs/_logs/vt13_reduced_C3.log`). It spawns GPU trainer subprocesses and SIGKILLs some of them by design (kill137, killckpt) |
| 18:02:03 | S0 models GPU test batch started (VT-06/07/08/09/14/16; log in the session scratchpad `logs/gpu_batch1.log`, mtime 18:05:55, 232.8 s). **33 of its tests failed with CUDA OOM**, with "2.19 MiB / 960 KiB free": the GPU was full, holding CARLA-env 0.45 GB, gnome-remote-desktop 0.31 GB, S1 2.46 GB, and processes 794810 (5.06 GB) and 796220/797635 (≈ 5.8 GB) |
| **18:03:09** | **Xid 31**, pid 799092 `python`: MMU fault, engine CE4 (copy engine), FAULT_PDE, ACCESS_TYPE_VIRT_WRITE |
| 18:03:50–59 | CARLA lane stopped and **PAUSED** "at user request (outeropt GPU load)" |
| 18:10:52 | S1 `gated_delta_net-d128-lr1.0e-02` attempt 1 started |
| **18:16:50** | **RC watchdog "GPU is probably locked! Notify Timeout Seconds: 7" + Xid 8**, pid 811688, channel 0x00000003. S1 attempt 1 died with `cudaErrorLaunchTimeout` after epoch 15 (queue: end rc=1 at 18:16:51) |
| 18:16:51 | S1 attempt 2 started |
| 18:25:35 | CARLA lane **resumed** (pause_log: "outeropt GPU load low: 12 GB free"; systemd: Started critscen-lane.service) |
| **18:25:42** | **RC watchdog + Xid 8**, pid 818052, channel 0x00000003. S1 attempt 2 died with `cudaErrorLaunchTimeout` after epoch 25 (queue: end rc=1 at 18:25:42) |
| 22:26:20 | S1 attempt 3 of the same cell completed (0.9877) while the CARLA lane was running (18:25:35–22:50:57) |
| 18:30 → 2026-11-10 07:20 | **No further NVRM or OOM events.** This period includes VT-13 audit kills (22:30–23:38), the VT-13 production test (to 00:54) and today's VT-13 run b, whose kill -9 at step 137 left `attempts.json n_starts=2` |

**Pid-to-time mapping.** Journal entries carry the pids of short-lived processes (CRON, tracker): 792884 at 18:00:41, 800982 at 18:03:38, 803998 at 18:05:22, 816318 at 18:14:34, 816636 at 18:15:01. Pids are allocated in increasing order, so:
- 811688 started between 18:05:22 and 18:14:34. That matches S1 attempt 1 (18:10:52).
- 818052 started after 18:15:01. That matches S1 attempt 2 (18:16:51).
- Both deaths coincide with the queue's rc=1 lines to the second. **The Xid-8 pids are the two S1 attempts.**

### 5.2 R2(d): pid 799092 (Xid 31)

- **Start time.** From the mapping, it is a `python` process that started between **18:00:41 and 18:03:09**. That excludes the S1 lr 3.2e-3 run (started 17:53), the VT-13 pytest parent (17:58) and CARLA-env pid 124395.
- **Agents' logs.** This project's agent transcripts were searched read-only. Two project candidates started in that interval:
  - (A) **the S0 models GPU test batch** (`pytest tests/test_vt06…test_vt16 -m gpu`), started 18:02:03 and ran through the fault; 33 of its tests hit CUDA OOM;
  - (B) **a trainer subprocess of the `vt13_reduced_C3` GPU test** (`test_vt13_protocol_audit`, 17:58–18:10). It launches trainers one after another (p1, p2, kill137, killckpt; pids 796220 and 797635 were alive at 18:02–18:05), and two of them SIGKILL themselves, one mid-accumulation and one during a checkpoint write.
- **Not identified uniquely.** Neither log names pid 799092.
- **Evidence against a deterministic "kill -9 → Xid 31" link.** The same kill paths ran at least 5 more times later with no Xid: 2 kills in each of the passing C0 and C3 cases of the VT-13 full audit (22:30–23:38), plus today's run b (kill -9 at step 137, `n_starts=2`). The failed C12 audit case and the 00:54 production test may add more.
- **Shared condition.** At 18:03 the device memory was **exhausted** (≤ 2 MiB free) by five concurrent processes, four of them project processes. That is a breach of the one-queue rule.
- Leading hypothesis [INF]: a copy-engine write fault during OOM or teardown races under full device memory. This is unconfirmed.

### 5.3 The two Xid-8 events

- **What they are.** The RM's RC watchdog declared "GPU is probably locked" after its 7 s notify timeout. RM then reset the channel, and the running CUDA context (S1) got `cudaErrorLaunchTimeout`.
- **Not a long project kernel.** The S1 workload is Zoology MQAR, 2 layers with T ≤ 1024, so its kernels take microseconds to milliseconds. The same cell completed on attempt 3. Neither event is consistent with one long kernel at R2(c) shapes either: those peak at 25 ms (§1).
- **Event 2 (18:25:42).** It fired **exactly 7.0 s** after the CARLA lane runner was resumed (18:25:35). The notify timeout is 7 s, so the GPU stopped answering at the moment CARLA restarted. The lane starts by launching a CARLA UE4 server, which is graphics plus compute work on this GPU (port-3090 lane per `runner.log`). Likely cause: CARLA (re)start starving or locking the GPU for ≥ 7 s while S1 was running [INF; strong temporal evidence, no direct trace].
- **Event 1 (18:16:50).** The CARLA lane was paused (18:03:59–18:25:35). No external trigger was found:
  - the system journal for 18:15:30–18:17 shows only a DHCP renewal (18:16:20) and nothing GPU-related;
  - the Xorg log has no entries after 17:32:53.
  The GPU clients at the time were Xorg, gnome-shell, gnome-remote-desktop (C+G), browsers and electron GPU processes, plus CARLA-env `critscen.demo` (pid 124395, a torch process). **Cause unknown.** Candidates are a GPU hang inside S1's own stream (a non-deterministic Triton/fla kernel stall) or a hang in a desktop or graphics client.
- **Is a display watchdog active on this GPU?** Yes:
  - `nvidia-smi -q` reports `Display Attached: Yes`, `Display Active: Enabled`, `Persistence Mode: Disabled`, `Compute Mode: Default` (`env/nvidia-smi-q_*.txt`, `env/display_query.csv`).
  - Xorg (vt2, since Nov 2), gnome-shell, xdg-desktop-portal-gnome, gnome-remote-desktop-daemon (C+G, 305 MiB), firefox, an electron app and gnome-clocks all hold graphics contexts on GPU 0 (`env/nvidia-smi_*.txt`, `env/display_clients.txt`).
  - The kernel log itself shows the RM RC watchdog (`krcWatchdog_IMPL`, 7 s notify timeout) active on this GPU.
  - No RM registry overrides are set (`/proc/driver/nvidia/params`: `RegistryDwords: ""`).
  - **Not verified:** whether the watchdog would be disabled without a display, or whether its timeout is tunable. The RM core ships as a binary blob in this DKMS tree; `/usr/src/nvidia-595.99.02` has no `krcWatchdog` source.
- **Implication for R2(c).** The longest kernels are ≤ 25 ms, so ordinary project kernels give a ≥ 277× margin to the watchdog. The residual watchdog risk comes from GPU-wide stalls such as CARLA restarts and desktop clients, not from our kernel lengths.

## 6. Hand-offs (no `src/` edits were made)

1. **S0 training/runner (R2(b) owner).** Add `infra_events` to `runs/<id>/summary.json` and to the registry's terminal rows. `scripts/repro/infra_events.py` (`kernel_events(start, end)`, `InfraTagger`) is a ready reference implementation: no dependencies beyond stdlib plus `journalctl` and `nvidia-smi`, and validated against the KLOG.
2. **Orchestrator (R2(a) owner).**
   - Require every project GPU job (S0 verifiers, R3/R7 executors, S0 runner) to start through `python scripts/repro/gpu_lock.py run --label … --wait-exclusive -- …`, or the in-process `gpu_lock()`.
   - The S0 runner's own queue does not use this lock today, so the two must share one lock file.
   - Without this, (a) cannot be "live", and the plan's no-concurrency condition for timed and gate runs is not met (§3 table).
3. **Benchmarks.** Do not use the D384 rows of `experiments/bench/train_step_idle_20261110.json`; they were contended. Use `train_step_idle_20261110_D384.json`.
4. **CARLA coordination.** The CARLA lane's resume rule ("outeropt GPU load low: 12 GB free") restarted CARLA while an S1 job was running, and a watchdog fired 7 s later. Gate runs should not overlap a CARLA (re)start.

## 7. Deviations and notes

- I ran D512 cells although the plan makes D512 conditional on FEAS-512. The task asked for them, and reading 2 says so.
- No CUDA_LAUNCH_BLOCKING bisection or O4 soak was run, because neither was triggered: there was no new event and no kernel ≥ 1 s.
- R2(c) attempt 1 was lost to a bookkeeping bug in my script (§1). The launcher's `rc=` log line was also wrong in its first version (`$(date)` reset `$?`); this was fixed before the measured runs.
- Defects found in `src/hybridlc/**`: none.

## 8. Files

- **Scripts:**
  - `scripts/repro/gpu_lock.py`, `scripts/repro/infra_events.py`, `scripts/repro/r2_kernel_profile.py`, `scripts/repro/r2c_launch.sh`, `scripts/repro/test_r2_tooling.py`;
  - SHA-256 at report time: gpu_lock `bf457994…`, infra_events `a6aa810b…`, r2_kernel_profile `a44f96a5…`, r2c_launch `6cdbb666…`, test `8c9b93de…`.
- **Outputs:**
  - `experiments/repro/r2/profile/*.json`, `profile/traces/*.json.gz`;
  - `infra/R2c-*.infra.json`;
  - `klog_scan_*.json`;
  - `tooling_selftest.log`, `launch.log`, `profile_*.log`;
  - `env/` (nvidia-smi, display query, driver params, KLOG extract, S1 queue log copy, CARLA lane pause and systemd lines, display clients).

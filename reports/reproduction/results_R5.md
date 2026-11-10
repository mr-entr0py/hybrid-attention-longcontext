# R5 results: discriminating MQAR at d_model = 64 (Zoology's GDN vs ours under bf16 autocast)

**Executed:** 2026-11-10, 08:37–10:08 IST, by the R5 executor (workflow "Reproduction execution").
**Pre-registration:** `experiments/repro/R5/prereg.yaml` (SHA-256 `c51afb3f…6246`). I wrote it at 08:37–08:40, before any R5 code change, smoke run or run. It copies plan §2.5, §2.0 and §2.11 (O5); the plan's SHA-256 `6bc14af6…cebc` matches the DEV-01 row, re-checked at 08:36. It fixes my readings M1–M2, RG1, PH1, CMP1–3, BM1, V1–V2, C1, P1 and I1–I2 before any result was seen. No criterion was changed after a result was seen.
**Precondition:** VT-25 (R3a) PASS (`results_R3.md`). R4 G-RP(b) PASS (`results_R4.md`).
**Code state:**
- HEAD `110fd9e`, with a dirty tree because S0 edits `src/` concurrently.
- `src/hybridlc/models/` equalled HEAD at the start of every run (`git_diff_quiet_src_models: true` in each run's provenance) and after the last run. SHA-256 prefixes: `attention.py` 7c9227ae, `linear.py` 466d4fef, `mixers.py` aacd67c5, `config.py` 1b6af31c. These are the same as in R4.
- I edited nothing under `src/`.

**Environment:** torch 2.11.0+cu128, triton 3.6.0, fla 0.6.0, RTX 5080 (sm_120), driver 595.99.02. Zoology @`1ad20d1`, clean. Memory fraction 0.44; peak memory 2.03 GiB per run. Each run JSON records all of this in `provenance`.

## Verdict

| Item | Verdict | Key numbers (criterion) | Raw output |
|---|---|---|---|
| **R5 overall** (reading V1) | **PASS** | rule 1 regime OK; rule 2 PASS; rule 3 PASS; rule 4 not fired | `experiments/repro/R5/r5_results.json`, `r5_table.md` |
| Rule 1: regime | primary cell = 1024/256 | Zoology's 3-seed mean on 1024/256 is 0.2963, which is inside [0.05, 0.95] | same |
| Rule 2: phenomenon (Zoology d64 < d128 < d256 on 1024/256) | **PASS** | **0.2963** < 0.9250 (R1 d128) < 0.9566 (R1 d256) | same; references `experiments/s1_zoology/runs/gated_delta_net-d{128,256}-lr1.0e-03.json` (SHA-256 verified) |
| Rule 3: comparison, FAIL iff \|Δ\| > max(0.03, 2.78·SE_W) | **PASS** | Δ(ours − Zoology) = **−0.0175**; SE_W = 0.0415; threshold = **0.1154** (the 2.78·SE_W term). 90% CI [−0.107, +0.072] (Welch df 3.85) | same |
| Rule 4: bimodality (an arm's span > 0.2 on the primary cell) | not fired | span: Zoology 0.0875; ours 0.1096 | same |
| O5 (+2 seeds per arm) | not triggered | rule 4 did not fire | — |
| Infrastructure | **4 RC-watchdog + Xid 8 events** in R5 windows; every hit run was repeated in full | details below | `experiments/repro/R5/infra/*.infra.json`, `experiments/repro/R5/r5_infra_summary.json` |

**What it shows, and its power.**
- **The phenomenon replicates strongly.** At d = 64, Zoology's GDN has a state of 17,152 (Zoology's own accounting; the figure gives ≈ 1.7e4). It reaches 0.296 on the 1024/256 cell, against 0.925 at 67,072 and 0.957 at 265,216. That is the recall–state trade-off in a regime below ceiling.
- **Our GDN at production precision (bf16 autocast) is indistinguishable from Zoology's within this test's power.**
  - The point estimate Δ = −0.0175 is also inside the 0.03 materiality floor.
  - The test is low-power on this cell. Seed SD is 0.045 (Zoology) and 0.056 (ours), so the MDD is 0.115 at 50% power and 0.154 at 80% power (df 4).
  - So R5 rules out a bf16 handicap larger than about 0.11–0.15 on the 1024/256 cell. It cannot rule out a handicap of a few points.

## Per-run table (final validation record = the metric of R1/R4; all 32 epochs, none early-stopped, none collapsed)

| Run | seed / data seed | overall | 64/4 | 64/8 | 64/16 | 128/32 | 256/64 | 512/128 | **1024/256** | wall (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| `gdn_zoology-d64-lr1.0e-03-s123` | 123/123 | 0.85814 | 1.0000 | 0.9999 | 1.0000 | 0.9999 | 0.9846 | 0.7407 | **0.2820** | 595 |
| `gdn_zoology-d64-lr1.0e-03-s124` | 124/124 | 0.87801 | 1.0000 | 1.0000 | 1.0000 | 0.9998 | 0.9924 | 0.8066 | **0.3472** | 620 |
| `gdn_zoology-d64-lr1.0e-03-s125` | 125/125 | 0.84702 | 1.0000 | 1.0000 | 1.0000 | 0.9992 | 0.9709 | 0.6993 | **0.2597** | 597 |
| `our_gdn_bf16-d64-lr1.0e-03-s123` | 123/123 | 0.83706 | 1.0000 | 0.9999 | 1.0000 | 0.9996 | 0.9710 | 0.6592 | **0.2297** | 605 |
| `our_gdn_bf16-d64-lr1.0e-03-s124` | 124/124 | 0.87604 | 1.0000 | 1.0000 | 1.0000 | 0.9998 | 0.9909 | 0.8023 | **0.3393** | 592 |
| `our_gdn_bf16-d64-lr1.0e-03-s125` | 125/125 | 0.85408 | 1.0000 | 0.9999 | 1.0000 | 0.9994 | 0.9807 | 0.7314 | **0.2673** | 600 |

Raw: `experiments/repro/R5/runs/<run_id>.json`. Weights: `experiments/repro/R5/weights/<run_id>.pt`. Per-epoch logs: `experiments/repro/R5/logs/<run_id>.attempt<k>.log`.

**Means, SDs and the comparison** (sample SD, ddof = 1, n = 3 per arm):

| Cell | Zoology mean ± SD | ours mean ± SD | Δ | SE_W | max(0.03, 2.78·SE_W) | Role |
|---|---|---|---|---|---|---|
| **1024/256** | 0.2963 ± 0.0455 | 0.2788 ± 0.0557 | **−0.0175** | 0.0415 | 0.1154 | **primary (criterion)** |
| 512/128 | 0.7489 ± 0.0541 | 0.7309 ± 0.0715 | −0.0179 | 0.0518 | 0.1439 | descriptive |
| 256/64 | 0.9826 | 0.9808 | −0.0018 | — | — | descriptive |
| overall | 0.8611 ± 0.0157 | 0.8557 ± 0.0195 | −0.0053 | 0.0145 | 0.0402 | descriptive |

**Descriptive only (not pre-registered):**
- **Paired by seed.** Both arms share the data seed and Zoology's fixed data order. Per-seed Δ on 1024/256: −0.0522, −0.0079 and +0.0075. Mean −0.0175, SD 0.031, paired SE 0.018. The 90% CI with t(2) = 2.92 is [−0.070, +0.035]. The sign is not consistent across seeds.
- **Trajectories.** The largest overall-accuracy gap at any epoch between the arms is 0.022 (s123), 0.006 (s124) and 0.018 (s125). The sign is mixed: ours is lower at s123 and higher at s125.
- **Zoology d64 overall vs the figure.** Overall accuracy is 0.861, next to the figure's ≈ 0.89. This is context only (I-20, rule 2). LR 1e-3 is not best-over-LR at d64 (plan).
- **Our fp32 arms from R4 on 1024/256, for reference.** d128 0.9224 and d256 0.9549. So our arm is also ordered d64 (0.279) < d128 < d256, although the precision differs from R5's.

**Tolerance basis for G-RP(b)** (plan §4.1, "R5 PASS" row). The measured seed SD of **overall** accuracy at d64 is 0.016 (Zoology) and 0.020 (ours). G-RP(b)'s frozen ±0.03 is therefore about 1.5–1.9 single-run SDs in this below-ceiling regime. At ceiling (R1/R4: overall ≥ 0.989) the variance is compressed, so ±0.03 is far looser there. The frozen threshold is not changed.

**Geometry and precision evidence (reading P1).**
- **Parameters.** 555,270 (Zoology) vs 555,268 (ours). The 2 extra are Zoology's unused `D` parameter, as in R4. State size is 17,152 in both arms.
- **GDN chunk-kernel entry points**, from call counters in `run_one_r5.py`, recorded in each run JSON as `gdn_kernel_calls`:
  - each arm made 97,248 calls, all on its own path (Zoology arm → `zoology.mixers.gated_delta_net`; ours → `hybridlc.models.linear`; 0 cross-calls);
  - q, k, v, g and β were **bf16** at the first call in both arms;
  - one residual difference: Zoology computes g in fp32 and then casts it to bf16, whereas our autocast arm feeds `a_proj`'s bf16 output to the in-kernel gate (`use_gate_in_kernel=True`).
- **Output.** The mixer output is promoted to fp32, and master weights are fp32 (`autocast_bf16_mixer: true` in our runs only).

## Commands

1. Pre-registration first: `experiments/repro/R5/prereg.yaml` (08:37–08:40).
2. Tooling tests: `python scripts/repro/gpu_lock.py run --label R5-tests --timeout 600 -- python -m pytest scripts/s1_zoology/test_s1_r5.py -q -p no:cacheprovider -x` → **7 passed** (21 s, 08:40). They cover:
   - the run list (seeds, LR, geometry);
   - in both arms on the GPU: d64 geometry, state size and parameter delta, a 3-step train step, and bf16 q/k/v/β at kernel entry;
   - the analysis rules on synthetic run files: pass, rule-3 fail, a widening t-term, bimodality, and the high-regime case with a rule-2 fail.

   Regression check: `pytest scripts/s1_zoology/test_s1_r4.py -k "run_list or collapse"` → 2 passed.
3. Smoke run, 1 epoch, not a result. It writes a `.smoke.json` file, which the analysis excludes: `python scripts/repro/gpu_lock.py run --label R5-smoke --timeout 600 -- python scripts/s1_zoology/run_one_r5.py --run-id our_gdn_bf16-d64-lr1.0e-03-s123 --max-epochs 1` (08:41; log `experiments/repro/R5/logs/smoke_our_gdn_bf16-s123.log`).
   - Disclosure: I saw its epoch-0 values (overall 0.6918, 1024/256 0.0637) after the pre-registration was frozen.
4. Queue: `bash scripts/repro/r5_launch.sh` (08:42).
   - It runs `scripts/s1_zoology/run_queue_r5.py`, interleaved by seed.
   - Each attempt is `scripts/repro/infra_events.py run --wait-exclusive --lock-timeout 600 -- python scripts/s1_zoology/run_one_r5.py --run-id <id>`, which holds the single GPU lock and writes one infra JSON per attempt.
   - After the first Xid I relaunched with `R5_CONTINUE_ON_XID=1 bash scripts/repro/r5_launch.sh` (08:46). The manual note is in `experiments/repro/R5/logs/queue.log`; deviation D2 below explains it.
   - The queue log is `experiments/repro/R5/logs/queue.log`. It ends with `verdict: PASS` at 10:07:44.
5. Evaluation: `python scripts/s1_zoology/r5_analysis.py` → `experiments/repro/R5/r5_results.json`, `experiments/repro/R5/r5_table.md`. The queue ran it automatically.
6. Infra summary: `python scripts/repro/r5_infra_summary.py` → `experiments/repro/R5/r5_infra_summary.json`.

## Infrastructure and run metadata

| Attempt | Window (IST) | rc | Infra events | Lock wait | Other GPU processes in the window (besides CARLA client 454 MiB and gnome-remote-desktop 305 MiB) |
|---|---|---|---|---|---|
| gdn_zoology s123 a1 | 08:42:01–08:45:19 | 1 | **RC watchdog + Xid 8, pid 1937763 (ours), 08:45:18**, epoch 10 | 0 s | none |
| gdn_zoology s123 a2 (full repeat) | 08:46:19–08:56:18 | 0 | 0 | 0 s | pid 1940836 `python`, ≈ 10 s, 532 MiB |
| our_gdn_bf16 s123 a1 | 08:56:19–09:06:28 | 0 | 0 | 0 s | none |
| gdn_zoology s124 a1 | 09:06:29–09:16:52 | 0 | 0 | 0 s | pid 1949025 S0 `pytest -q -m "not slow"` **without the lock**, ≈ 220 s, ≤ 2.8 GiB |
| our_gdn_bf16 s124 a1 | 09:24:26–09:29:11 | 1 | **RC watchdog + Xid 8, pid 1954398 (ours), 09:29:10**, epoch 14 | 453 s (waited for that pytest) | pid 1955523 `python`, ≈ 20 s, 400 MiB (about 3 min before the event) |
| our_gdn_bf16 s124 a2 (full repeat) | 09:29:12–09:39:08 | 0 | 0 | 0 s | 3 unlabelled `python` processes, ≤ 90 s, ≤ 2.5 GiB |
| gdn_zoology s125 a1 | 09:45:46–09:46:38 | 1 | **RC watchdog + Xid 8, pid 1963844 (ours), 09:46:38**, epoch 1 | 397 s | none |
| gdn_zoology s125 a2 (full repeat 1) | 09:46:40–09:47:15 | 1 | **RC watchdog + Xid 8, pid 1964377 (ours), 09:47:14**, epoch 1 | 0 s | pid 1964601 `python`, ≈ 20 s, 494 MiB |
| gdn_zoology s125 a3 (full repeat 2) | 09:47:36–09:57:37 | 0 | 0 | 20 s | none |
| our_gdn_bf16 s125 a1 | 09:57:39–10:07:42 | 0 | 0 | 0 s | 5 unlabelled `python` processes, ≤ 50 s each, ≤ 1.5 GiB |

- **All 6 reported results come from event-free windows.**
  - Every flagged attempt crashed with `cudaErrorLaunchTimeout`, so none of them produced a result JSON.
  - Each was repeated in full from epoch 0: its checkpoint was moved aside as `data/zoology/ckpt/<id>.flagged<k>.pt.discarded`, and no reported run has `resumed_at_epochs`.
  - The repeats are deterministic at the start. Epoch-0 values were identical across attempts (s124 ours 0.7617; s125 Zoology 0.6393).
  - s125 Zoology needed 3 attempts, inside the declared maximum of 3 full repeats.
- **The Xid pattern** (all four R5 events):
  - each is "RC watchdog: GPU is probably locked! Notify Timeout Seconds: 7" followed by Xid 8 on channel 0x00000003;
  - each surfaced at the host sync `int((y != -100).sum())` in the micro-batch shim, so the hung kernel itself is not localised;
  - three were in Zoology's own GDN (its bf16 casts) and one in ours (bf16 autocast), so the events are **not specific to our arm or to our code**;
  - two (s123 a1 and s125 a1) had no other project process on the GPU.
  - Counting R1 and R4, this machine has now had **7 Xid-8 events, all during GDN training in the Zoology harness**. The rate in R5 was 4 events in ≈ 1.17 GPU-h of windows.
  - s125 a1 and a2 both died in epoch 1, but a3 passed epoch 1, so this is not a deterministic hang at one step.
- **R2(a), one GPU queue: violated by other agents.** In 6 of the 10 windows, S0 processes, including a `pytest -m "not slow"` run, ran on the GPU without taking `scripts/repro/gpu_lock.py`. The plan's repeat rule (§2.0) covers infra events, not contention, so under prereg I1 these windows are recorded and not repeated, as in R4. Contention changes timing, not Zoology's fixed data order. There was no clock slowdown in any window (`any_slowdown: false`); the maximum temperature was 60 °C; `journal_ok` was true in all 10 windows.
- **GPU-h:**
  - 1.00 h of completed-run training compute (the 6 `wall_time_s`: 595 + 620 + 597 + 605 + 592 + 600 s);
  - 1.17 h summed over all 10 attempt windows (lock held, 4,202 s), of which ≈ 0.16 h was the 4 crashed attempts;
  - plus ≈ 0.02 h for the smoke run and tests.

  **Total ≈ 1.19 GPU-h**, below the plan's 1.3–1.7 because the d64 runs took ≈ 10 min each rather than 13–17. Lock waits caused by unlocked S0 processes (≈ 0.24 h) were spent idle, not computing.

## Deviations, readings and shims

1. **D1, O4 not run before R5 (sequencing; declared in the prereg before any run).**
   - Plan §2.2 says O4 must run before any further gate run once its trigger fires, and R4 fired it at 07:56:54.
   - O4 belongs to the orchestrator or PI and was not assigned to me. The orchestrator dispatched R5 with the R4 result in hand.
   - The trigger then fired four more times during R5. **The PI should decide whether O4 must still run before G-RV.** I recommend that it does; see the requests below.
2. **D2, queue behaviour on an Xid** (declared at 08:46, after the first event and before any R5 result existed; the first result was written at 08:56).
   - R4's queue stopped on any Xid. R5's queue, with `R5_CONTINUE_ON_XID=1`, instead **repeats the run in full** (plan §2.0: a gate run whose window contains an infra event is repeated in full) and continues, with at most 3 full repeats per run.
   - The rule that discards the partial run and its checkpoint now also covers non-Xid infra events. R4's code would have resumed those from the checkpoint.
   - This changes compute and robustness, not any criterion or metric.
3. **Readings** (all frozen in the prereg; none departs from the literal text): metric = final record (M1); regime inclusive (RG1); rule 2 against R1's unrounded d128/d256 values (PH1); the frozen 2.78 constant, with FAIL only on strict ">" (CMP1); bimodality strict, on the 3 initial seeds (BM1); overall verdict = worst of rules 2 and 3, INCONCLUSIVE under bimodality (V1). Under V1 the verdict would also be PASS under the less conservative reading, because rule 2 passes.
4. **Shims, as in R1/R4:** exact micro-batch-64 gradient accumulation; per-epoch resume (not used by any reported run); fla `head_first` shim (Zoology's GDN only); collapse-abort (never fired).
5. **bf16 scope (P1):** autocast covers only the mixer, as in R4b-2. Zoology's BaseConv, LM head and loss stay fp32 in both arms.
6. **Not done:** I wrote no ledger row in `reports/reproduction/results.md`, which does not exist and is outside my ownership.

## Files

- **New:**
  - `scripts/s1_zoology/run_one_r5.py`, `run_queue_r5.py`, `r5_analysis.py`, `test_s1_r5.py`;
  - `scripts/repro/r5_launch.sh`, `scripts/repro/r5_infra_summary.py`;
  - `experiments/repro/R5/` (`prereg.yaml`, `runs/`, `logs/`, `infra/`, `weights/`, `r5_results.json`, `r5_table.md`, `r5_infra_summary.json`).
- **Changed:** `scripts/s1_zoology/our_mixers.py`, appended only: `OurGdnBf16Zoology`, `r5_runs`, `r5_run_id`, the R5 constants. The R4 run-list tests still pass.
- **Defects found in `src/hybridlc`:** none. The Xid-8 hangs occur equally in Zoology's own GDN module, so they implicate the environment (driver, display watchdog, or an fla/Triton kernel on sm_120), not `hybridlc`.

## Consequences and requests

1. **G-RV:** R5 is **not FAIL** (PASS). Under plan §4.1, the only consequence of PASS is the tolerance basis: the seed SD of overall accuracy at d64 is 0.016 / 0.020, against G-RP(b)'s ±0.03, and the threshold is unchanged.
2. **Precision handicap (plan §2.5 (iii)).**
   - No bf16 handicap of our GDN is visible: Δ = −0.018 on the 1024/256 cell, with a sign that changes across seeds.
   - Only handicaps larger than about 0.11 (50% power) or 0.15 (80% power) on that cell are excluded.
   - The FEAS-2 / k = 0 reading therefore gets no caveat from R5, but R5 cannot certify parity at the few-point level.
3. **O4 is overdue, and the infra risk now matters for long runs.**
   - Seven Xid-8/RC-watchdog events have hit GDN training on this GPU. In R5 the rate was about one every 18 minutes of GDN windows.
   - R6, FEAS and S3 runs are hours long: without per-epoch-style resume, each event costs a run, and §2.0 makes gate runs repeat in full.
   - I recommend that the PI or orchestrator run O4 (soak plus `CUDA_LAUNCH_BLOCKING=1` bisect) before R6 and FEAS.
   - Facts for the O4 design: channel 0x00000003 every time; 7 s notify timeout; display attached (gnome-remote-desktop active); the hang hits both the Zoology-cast and the autocast GDN; it is not tied to a fixed step; it also happens with no other project process on the GPU.
4. **The single GPU queue (R2(a)) is not being honoured by the S0 workflow.** Unlocked `pytest` and `python` processes overlapped 6 of 10 R5 windows and delayed two attempts by 7.5 and 6.6 minutes. That matters more for timed gate runs, such as G0-T throughput, than for R5's accuracy numbers.

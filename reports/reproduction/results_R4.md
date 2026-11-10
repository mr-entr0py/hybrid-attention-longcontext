# R4 results: S1(b), our mixers in Zoology's harness, plus the production-F_g smoke run (R4b)

**Executed:** 2026-11-10, 07:25–08:31 IST, by the R4 executor (workflow "Reproduction execution").
**Pre-registration:** `experiments/repro/R4/prereg.yaml`. It was written before any R4 code change or run. It copies the criteria of `reports/reproduction/reproduction_plan.md` §2.4 (SHA-256 `6bc14af6…cebc`, which matches the DEV-01 row) and amendment §17.3/§17.6/§13.3. It fixes how I read the ambiguous parts (G1–G5, C1, R4b-1…3, I1–I2). No criterion was changed after a result was seen.
**Precondition:** VT-25 (R3a) **PASS** (`results_R3.md`; raw `experiments/repro/R3/vt25/vt25_transplant.jsonl`). R3 as a whole FAILs on VT-21/22/23/24, but none of those is a precondition of R4. Amendment §13.3 requires only that VT-25 pass before G-RP(b) is evaluated, so no diagnosis was needed before running.
**Code state:** HEAD `110fd9e`, with a dirty tree because the S0 integrator edits `src/` concurrently. Each run JSON records `git status --porcelain src` at its start. During R4 only `src/hybridlc/{eval,analysis,training,integration,utils,data}` and `INTERFACES.md` changed. **`src/hybridlc/models/` equalled HEAD throughout** (`git diff --quiet HEAD -- src/hybridlc/models`, checked after the last run). SHA-256 prefixes: `attention.py` 7c9227ae, `linear.py` 466d4fef, `mixers.py` aacd67c5, `config.py` 1b6af31c. I edited nothing under `src/`.
**Environment:** torch 2.11.0+cu128, triton 3.6.0, fla 0.6.0, RTX 5080 (sm_120), driver 595.99.02. Zoology @`1ad20d1`, clean. Seed 123 / data seed 123, which are Zoology's own (§17.6). All of this is recorded in each run's `provenance`.

## Verdicts

| Item | Verdict | Key numbers (criterion) | Raw output |
|---|---|---|---|
| **G-RP(b)**: ours within ±0.03 of Zoology on overall accuracy at matched d and LR, per cell | **PASS (4 / 4 cells)** | Δ(ours − Zoology): F_g d64 −0.0001; F_g d128 −0.0001; GDN d128 −0.0004; GDN d256 −0.0002 (each ≤ 0.03) | `experiments/repro/R4/runs/{our_fg-d64-lr1.0e-02,our_fg-d128-lr1.0e-03,our_gdn-d128-lr1.0e-03,our_gdn-d256-lr1.0e-03}.json`; Zoology: `experiments/s1_zoology/runs/*.json` (R1); evaluation `experiments/repro/R4/r4_results.json` |
| **R4b** (smoke): production F_g, in-distribution mean ≥ 0.95 in fp32 and in bf16 | **PASS (2 / 2)** | fp32 in-dist mean 1.0000; bf16 1.0000. bf16 run made 3,039 calls into `varlen_attn` with bf16 q; the fp32 run made 0 | `experiments/repro/R4/runs/our_fg_prod{,_bf16}-d128-lr1.0e-03.json` |
| Config-A (exploratory) | descriptive | GDN-A geometry: d128 overall 0.9324 (1024/256: 0.576); d256 overall 0.9793 (1024/256: 0.856) | `experiments/repro/R4/runs/our_gdn_configA-d{128,256}-lr1.0e-03.json` |
| Infrastructure | **O4 trigger fired** | RC watchdog + **Xid 8** at 07:56:54.80 in our process (pid 1915001) during the exploratory config-A d128 run, epoch 18. No gate run was affected | `experiments/repro/R4/infra/R4-our_gdn_configA-d128-lr1.0e-03-a1.infra.json` |

**Grade and what it shows.** R4 is a smoke test at ceiling. ±0.03 on overall accuracy equals ±0.21 on the 1024/256 cell, the only cell below ceiling (MDD, plan §1 rule 2). The observed differences are 50–300× smaller than the tolerance on overall accuracy. On the live 1024/256 cell they are −0.0026 (GDN d128) and −0.0017 (GDN d256), against the MDD of 0.21. Those per-cell values are descriptive, not a criterion. The trajectories also match: our GDN d128 tracks Zoology's epoch by epoch within 0.0032 (below). R4 shows that our mixer modules train like Zoology's inside the original harness. It cannot detect defects smaller than its MDD. The deterministic identity question was answered by VT-25.

## Commands

- Pre-registration written first: `experiments/repro/R4/prereg.yaml`.
- Tooling tests: `python -m pytest scripts/s1_zoology/test_s1_r4.py scripts/s1_zoology/test_s1_our_mixers.py scripts/s1_zoology/test_s1_resume.py scripts/s1_zoology/test_s1_microbatch.py -q -p no:cacheprovider` → **16 passed** (162.9 s).
- Smoke runs, 1 epoch each, not results (they write `.smoke.json`, which the analysis excludes): `python scripts/repro/gpu_lock.py run --label R4-smoke --timeout 600 -- python scripts/s1_zoology/run_one_r4.py --run-id <id> --max-epochs 1`, for `our_fg_prod_bf16-d128-lr1.0e-03` and `our_gdn_configA-d128-lr1.0e-03`. Logs: `experiments/repro/R4/logs/smoke_*.log`. They ran after the pre-registration was written. Disclosure: the R4b bf16 run early-stopped after 1 epoch, and Zoology's data order is fixed, so its smoke value (0.9980) equals the reported value. The criterion was frozen before the smoke ran.
- Queue: `bash scripts/repro/r4_launch.sh` (07:33:15). That runs `scripts/s1_zoology/run_queue_r4.py`, and each attempt is `scripts/repro/infra_events.py run --wait-exclusive --lock-timeout 600 -- python scripts/s1_zoology/run_one_r4.py --run-id <id>`. Each attempt therefore holds the single GPU lock (`scripts/repro/gpu_lock.py`) and gets an infra JSON. Log: `experiments/repro/R4/logs/queue.log`, with per-attempt logs `experiments/repro/R4/logs/<id>.attempt<k>.log`.
- Relaunch after the infra failure, for the two exploratory runs only: `bash scripts/repro/r4_launch.sh our_gdn_configA-d128-lr1.0e-03 our_gdn_configA-d256-lr1.0e-03` (08:05:49; manual note in `queue.log`).
- Evaluation: `python scripts/s1_zoology/r4_analysis.py` → `experiments/repro/R4/r4_results.json`, `experiments/repro/R4/r4_table.md`.

## G-RP(b): per-cell table (final overall accuracy; final validation record, the same metric as G-RP(a))

| Ours | Zoology (R1 re-run) | Ours | Zoology | Δ | ±0.03 | 64/4 | 64/8 | 64/16 | 128/32 | 256/64 | 512/128 | 1024/256 | Epochs (ours / Z) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `our_fg-d64-lr1.0e-02` | `attention-d64-lr1.0e-02` | 0.99989 | 0.99998 | −0.00009 | ✔ | 0.9992 / 1.0000 | 1.0000 / 0.9999 | 1.0000 / 1.0000 | 1.0000 / 1.0000 | 1.0000 / 1.0000 | 1.0000 / 1.0000 | 1.0000 / 1.0000 | 1* / 1* |
| `our_fg-d128-lr1.0e-03` | `attention-d128-lr1.0e-03` | 0.99989 | 0.99995 | −0.00005 | ✔ | 0.9995 / 0.9998 | 0.9998 / 0.9999 | 1.0000 / 1.0000 | 1.0000 / 1.0000 | 1.0000 / 1.0000 | 1.0000 / 1.0000 | 1.0000 / 1.0000 | 1* / 1* |
| `our_gdn-d128-lr1.0e-03` | `gated_delta_net-d128-lr1.0e-03` | 0.98886 | 0.98924 | −0.00039 | ✔ | 1.0000 / 1.0000 | 0.9999 / 0.9999 | 1.0000 / 1.0000 | 1.0000 / 1.0000 | 1.0000 / 1.0000 | 0.9997 / 0.9998 | **0.9224 / 0.9250** | 32 / 32 |
| `our_gdn-d256-lr1.0e-03` | `gated_delta_net-d256-lr1.0e-03` | 0.99352 | 0.99377 | −0.00025 | ✔ | 1.0000 / 1.0000 | 0.9999 / 0.9999 | 1.0000 / 1.0000 | 1.0000 / 1.0000 | 1.0000 / 1.0000 | 0.9999 / 1.0000 | **0.9549 / 0.9566** | 2* / 2* |

`*` = early stop (valid acc > 0.99, Zoology's default). Per-config cells are shown as ours / Zoology and are descriptive.
- **Same early-stopping regime.** Each cell stopped at the same epoch as Zoology's run (1/1, 1/1, 32/32, 2/2). So the caveat from G-RP(a), that the stopping rule shapes the overall number, applies equally to both sides.
- **Trajectory check (GDN d128, descriptive).** Validation accuracy by epoch, ours vs Zoology:
  - epoch 0: 0.9440 vs 0.9447;
  - epoch 5: 0.9782 vs 0.9804;
  - epoch 10: 0.9853 vs 0.9866;
  - epoch 20: 0.9874 vs 0.9880;
  - epoch 31: 0.9889 vs 0.9892.

  The largest gap at any epoch is 0.0032 (epoch 15: 0.9841 vs 0.9873).
- **Geometry.** Parameters 1,167,876 vs 1,167,878; the 2 extra are Zoology's unused `D` parameter. State size (Zoology's own accounting) 67,072 = 67,072 (d128) and 265,216 = 265,216 (d256). F_g has 256 fewer parameters at d64 (545,728 vs 545,984), because it has no biases (documented residual difference).
- **Residual differences still present in these runs.** None was isolated, because G-RP(b) passed:
  - our GDN runs fp32 into the fla kernel, whereas Zoology casts q/k/v/β/g to bf16;
  - F_g has no biases and no attention dropout.

  At this MDD none of them has a visible effect.
- **Run cost.** Wall time ours vs Zoology: GDN d128 685 s vs 994 s; F_g 72–74 s vs 17–23 s. That is compute time, not a criterion. Our F_g runs in fp32 through the per-row SDPA-math reference path (`varlen_attn` calls = 0).

## R4b: production F_g (QK-RMSNorm, RoPE θ = 5e5, headwise gate; 2 heads × 64; d = 128, lr 1e-3)

| Run | In-dist mean (64/4, 64/8, 64/16, 128/32, 256/64) | ≥ 0.95 | Overall | 512/128 (descr.) | 1024/256 (descr.) | Epochs | `varlen_attn` calls (first q dtype) |
|---|---|---|---|---|---|---|---|
| `our_fg_prod-d128-lr1.0e-03` (fp32) | **1.0000** (all five = 1.0) | ✔ | 0.99814 | 0.9998 | 0.9871 | 1* | 0 (reference path) |
| `our_fg_prod_bf16-d128-lr1.0e-03` (mixer under bf16 autocast) | **1.0000** (all five = 1.0) | ✔ | 0.99804 | 0.9998 | 0.9865 | 1* | **3,039** (torch.bfloat16) |

- **Configuration (readings R4b-1, R4b-2).**
  - `build_mixer('A', cfg, n_heads=2, head_dim=64)` with the protocol defaults. The run JSON records `use_qk_norm=True, use_rope=True, rope_theta=500000.0, use_out_gate=True`.
  - The o_proj is aliased as `out_proj`, as in `OurFgZoology`. So the only differences from `our_fg-d128` are QK-norm, RoPE and the gate.
  - The bf16 arm wraps the mixer's forward in `torch.autocast('cuda', bfloat16)` and promotes its output to fp32. The rest of Zoology's model and the master weights stay fp32.
  - `test_s1_r4.py::test_prod_fg_train_step_and_kernel_path` asserts that the bf16 arm calls `varlen_attn` with bf16 q and that the fp32 arm never does.
- **Descriptive cells beyond the training length.** On 1024/256 the production F_g scores 0.987 (fp32) and 0.987 (bf16), against 1.000 for the RoPE-free `our_fg-d128`. That is consistent with RoPE beyond the 256-token training length, as the plan anticipated. It is not a criterion.
- **Attention statistics after training** (first test batch per config, first row, ≤ 512 tokens). Max logit 11.75–12.40 and entropy 0.90–1.71 nats on the in-distribution configs. On 1024/256: max logit 9.3 (fp32) / 9.1 (bf16), entropy 3.03 / 3.04. The two precisions agree to within 0.041 nats in entropy and 0.15 in max logit on every config.
  - For reference, the RoPE-free, un-normalised `our_fg-d128` reaches max logit 21.0–23.1 (17.8 on 1024/256). The QK-norm caps the logit scale. This is descriptive, kept for the R4b failure tree, which did not fire.

## Config-A (exploratory, descriptive only)

| Run | Geometry | Overall | 512/128 | 1024/256 | Epochs | State (Zoology accounting, both layers) |
|---|---|---|---|---|---|---|
| `our_gdn_configA-d128-lr1.0e-03` | GDN-A defaults: H = 1, d_k 64, d_v 128, output gate on | 0.9324 | 0.9521 | 0.5757 | 32 (resumed at epoch 18, see infra) | 34,304 |
| `our_gdn_configA-d256-lr1.0e-03` | GDN-A defaults: H = 2, d_k 64, d_v 128, output gate on | 0.9793 | 0.9988 | 0.8562 | 32 | 68,608 |

- Both runs solve every in-distribution config (≥ 0.9995).
- **Comparison at matched state (descriptive).** Config-A d256 has the same per-layer recurrent state (2 × 64 × 128 = 16,384) and head geometry as `our_gdn-d128` and Zoology's GDN d128. The differences are d_model 256 vs 128 and the GDN-A output gate (`use_gate=True`).
  - It scores 0.856 on 1024/256, against 0.922 (ours, d128) and 0.925 (Zoology, d128).
  - It also never crosses the 0.99 early-stopping threshold in 32 epochs.
  - This is a single seed on a length-extrapolation cell (1024 tokens, trained ≤ 256). The plan makes it descriptive, so it supports no claim. I flag it for the reproduction review, because GDN-A with its gate is the production linear mixer.

## Infrastructure and run metadata (plan §2.2, R2(b) tagging; prereg I1/I2)

| Attempt | Window (IST) | rc | Infra events | Exclusive at start | Other project processes seen in window |
|---|---|---|---|---|---|
| our_fg-d64 a1 | 07:33:15–07:34:32 | 0 | 0 | yes | pid 1909514 (`python`, 698 MiB, unlabelled; appeared mid-window) |
| our_fg-d128 a1 | 07:34:33–07:35:51 | 0 | 0 | yes | none |
| our_gdn-d128 a1 | 07:35:52–07:47:20 | 0 | 0 | yes | none |
| our_gdn-d256 a1 | 07:47:22–07:48:57 | 0 | 0 | yes | none |
| our_fg_prod a1 | 07:48:58–07:50:04 | 0 | 0 | yes | none |
| our_fg_prod_bf16 a1 | 07:50:05–07:50:27 | 0 | 0 | yes | none |
| our_gdn_configA-d128 a1 | 07:50:29–07:56:55 | 1 | **RC watchdog + Xid 8, pid 1915001 (ours), channel 0x00000003, 07:56:54.80** | yes | none |
| our_gdn_configA-d128 a2 (resume, epoch 18) | 08:05:52–08:10:40 | 0 | 0 | yes | pid 1919872 (S0 `pytest -m "not slow"`, started without the lock) |
| our_gdn_configA-d256 a1 | 08:17:14–08:31:22 (lock wait 392 s for that S0 pytest) | 0 | 0 | yes | none |

- All windows have `journal_ok: true`. No clock slowdown flags appeared; max temperature was 67 °C.
- Besides our process, the only GPU processes present were the CARLA-env python client (pid 124395, 454 MiB) and gnome-remote-desktop (pid 293356, 305 MiB).
- Infra JSONs: `experiments/repro/R4/infra/R4-<id>-a<k>.infra.json`.
- **Gate runs.** All 4 G-RP(b) runs and both R4b runs had event-free windows, so none needs repeating (plan §2.0).
- **The Xid-8 episode.** Our config-A d128 process died with `cudaErrorLaunchTimeout` at `loss_total.item()` in epoch 18 (log `experiments/repro/R4/logs/our_gdn_configA-d128-lr1.0e-03.attempt1.log`). No other project process was on the GPU at the time.
  - This matches the two R1 crashes (2026-11-09 18:16:50 and 18:25:42): RC watchdog "Notify Timeout Seconds: 7", Xid 8, channel 0x00000003, GDN training in the Zoology harness, mid-run.
  - It is the third such event, and the first in a fp32 GDN (our module). The R1 events were in Zoology's bf16-cast GDN.
  - The queue stopped as designed (`queue.log`: "plan §2.2 trigger O4; stopping the queue").
- **Resume decision.**
  - Plan §2.2: trigger O4 means "Run O4 before any further gate run".
  - The two remaining runs were exploratory, not gate runs. So I resumed them under the declared retry policy (§2.0: infra failure → resume from the last per-epoch checkpoint).
  - The config-A d128 result is labelled "resumed at epoch 18" (`resumed_at_epochs: [18]`). It is descriptive only.
- **Wall time.** 07:33–08:31.
- **GPU-h:**
  - 0.69 h of training compute for the 8 runs (run JSON `wall_time_s`: 72 + 74 + 685 + 91 + 63 + 18 + 651 + 843 s);
  - **0.71 h** with process start-up, data loading and post-training diagnostics, summed over all 9 attempt windows (lock held, 2,548 s);
  - a further ≈ 0.07 h (estimated, not summed) for the smoke runs and the tooling tests.

  Total ≈ 0.78 h, inside the plan's 0.6–0.9.

## Deviations, readings and shims (all declared before running unless noted)

1. **Shims (as R1):** exact gradient accumulation in 64-row micro-batches, and per-epoch resume. Only config-A d128 used resume. The fla `head_first` shim applies to Zoology's GDN only; our mixers do not use it.
2. **Collapse-abort rule** (plan §2.0, reading C1), implemented as `zoology_compat.collapse_abort_due` and unit-tested. It never fired (`collapsed: false` in all 8 runs).
3. **Reading R4b-2** (bf16 autocast scoped to the mixer, not to Zoology's FFT BaseConv / LM head) is a reading of "under bf16 autocast". The plan's stated purpose, exercising `varlen_attn`, is shown to be met (3,039 calls).
4. **Concurrency (R2(a)).** Every R4 attempt held the single GPU lock and started with no other project process on the GPU (`--wait-exclusive`). Two S0 processes later started **without** the lock during two of the 9 windows (rows above). Neither window is a gate window that changes a verdict; the gate rows are event-free. R2(a) is still not observed project-wide (see `results_R2.md`).
5. **Not done:** I wrote no ledger row in `reports/reproduction/results.md`. That file does not exist and is outside my ownership. The orchestrator or PI should add it from this file.

## Files

- **New:**
  - `scripts/s1_zoology/run_one_r4.py`, `run_queue_r4.py`, `r4_analysis.py`, `test_s1_r4.py`;
  - `scripts/repro/r4_launch.sh`;
  - `experiments/repro/R4/` (`prereg.yaml`, `runs/`, `logs/`, `infra/`, `weights/` with the final state_dict of every run, `r4_results.json`, `r4_table.md`).
- **Changed:**
  - `scripts/s1_zoology/our_mixers.py`: `OurFgProdZoology`, `OurFgProdBf16Zoology`, `r4b_runs`, `r4_runs`;
  - `scripts/s1_zoology/zoology_compat.py`: opt-in `collapse_abort`, `CaptureLogger.model_ref`. R1 behaviour is unchanged because both default to off;
  - `reports/s1_reproduction.md`: part (b) results.
- **Defects found in `src/hybridlc`:** none.

## Consequences and requests

1. **G-RP is fully passed:** (a) 16/16 on 2026-11-09, (b) 4/4 now. G-RP(b) was evaluated after VT-25 passed, as amendment §13.3 requires.
2. **O4 trigger (plan §2.2).** A new Xid 8 + RC-watchdog event hit a project process at 07:56:54. Per the plan, **O4 (critique A's X0 soak) must run before any further gate run**, which starts with R5. That is a PI / orchestrator action, and I did not start it.
   - Facts for the O4 design: three events, all mid-training GDN in the Zoology harness, all channel 0x00000003, a 7 s notify timeout, and the display attached. gnome-remote-desktop was active this time.
   - In the latest event no other project process was on the GPU, so contention is not required for it to happen.
   - The failure-tree step "repeated hangs at one configuration → `CUDA_LAUNCH_BLOCKING=1` bisect" does not yet apply: these are three different configurations.
3. **For the reproduction review (descriptive):** GDN-A geometry with its output gate underperforms the ungated Zoology geometry at matched state on the 1024/256 extrapolation cell (0.856 vs 0.922/0.925; single seed).

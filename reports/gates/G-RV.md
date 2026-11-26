**NOT PASSED as registered**

**FEAS entry BLOCKED** under G-RV's named-residual clause (rulings §9). Four rows end in "blocking": G0 (SUITE-PF), VT-24 (VT-24r), VT-25 (SUITE-PF O-d) and R2(a)/(b) live (R2-LIVE). These are chiefly missing or non-conforming evidence. I found no demonstrated model, trainer or item defect. FEAS training has already run under DEV-10. Whatever is repaired later, FEAS entry stays NOT PASSED as registered, and every FEAS and later result must say so. FEAS evaluation under D100 (started 06:31 on 26 Nov) is not cleared by this gate.

# G-RV gate report (T44)

- **Reviewer:** a fresh AI agent (Claude), **not a human auditor**. I wrote none of the checks. Evidence gathered 26 Nov 2026 up to about 06:45 IST at HEAD `b8728ed`. The file was saved about 08:30 IST, after an SSH re-check interrupted the first write; HEAD was still `b8728ed` at 08:30:38, just before saving. Commit `494112a` (08:30:39) then changed `reports/decisions.md` and `reports/reproduction/results.md`. **This report does not cover that commit:** I saw only its time and file names, not its contents. Evidence was not re-gathered. Statements about running jobs, decision rows, the ledger and the tree state are as of 06:45. This file is untracked.
- **Method:** read-only work on repository files, `~/outeropt-queue/` (outside the repo) and the kernel journal. No GPU, no test suite. CPU only (CUDA hidden, nice 19): item content hashes, trace comparisons, a GPU-monitor and registry scan, and a re-run of the executor's `diag23.py`.
- **Blinding:** I read no FEAS score or result. I did not open `experiments/results/`, any `runs/S2*` `metrics.jsonl`, `train.log` or `summary.json`, or any checkpoint (existence checks only).
  - The study registry was parsed in memory. Only status, step, commit and infra fields (and cache-hash notes) were printed.
  - In FEAS run directories, `env.json`, `attempts.json` and `gpu_summary.json` were parsed. Only git fields were printed.
  - For FEAS-SMOKE (20-step integration artefacts, not FEAS) I printed statuses, precision metadata, key names and types of `feas_decisions.json`, and digit-free key counts from its S2 evaluation logs.
- **Not verified**, including one denied command, is listed in section 5.

## 1. §9 conditions

| Condition | Gate status | Evidence and my check |
|---|---|---|
| Prereg integrity (all rows) | hashes hold; two files unhashed | `SHA256SUMS` 11/11 and `SHA256SUMS.v2` 14/14 OK. Of the Phase II results I checked, only VT-24r and TC-PF cite a prereg SHA-256 (both match). VT-22b and VT-23b cite a path only; the VT-13r, O4 and R6 summaries I read show no hash (R6 keeps an identical prereg copy). `FEAS_R6_namespace_report.v1.yaml` and `FEAS_seed101.v1.yaml` are in neither SHA256SUMS file (D086). |
| G0 passed | **NOT MET: blocking** (RR-G0-1..6, DEV-02 and rows D058–D061 exist) | §9 also requires SUITE-PF O-a. SUITE-PF was not run as registered: no JUnit XML, no `suite_prefeas*` files in `runs/_logs/`, no `experiments/repro/SUITE-PF/summary.json`. I found logs only outside the repo (`~/outeropt-queue/gates/`). Run 1: **5 failed**, 585 passed, 4 skipped, 5 xfailed, 2 xpassed (VT-07, VT-15). Tests patched in `db892e7`; I read the diff (updated signatures and the new `attempt_end` row; no tolerance loosened). Run 2: 590 passed, 4 expected skips, 6 xfailed, 1 xpassed (VT-07), 0 failed; I found no record of its commit. `src/`, `tests/`, `configs/` are unchanged from `db892e7` to HEAD. Substantively favourable, not the registered run. |
| G-RP (a), (b) | MET | D062. `experiments/s1_zoology/s1a_results.json`: attention 0.99998 / 0.99995, GDN 0.98924 / 0.99377. `experiments/repro/R4/r4_results.json`: 4/4 cells within ±0.03. Read from aggregate files, not re-derived from per-run rows. |
| VT-21 | NOT MET → RR-VT21 (D071) | Row present. The 12/384 count was not recomputed. |
| VT-22 (+ VT-22b v2) | NOT MET → RR-VT22 (D072) and RR-VT22b (D092). VT-22b **FAIL as registered**, F4 only | Recomputed from `experiments/repro/VT-22b/vt22b.jsonl`: F1 6.6e-4 / 6.7e-4, F2 1.09 / 1.13, B1 2.16e-2 / 2.15e-2 against 3.63e-2, B2 0.99 / 1.11, F3 6.0e-6 / 8.1e-6, **F4 7.16 / 9.40 (limit 2)**. Commit `63658a1`, code paths clean (tree dirty elsewhere). **No infra block** (only `kernel_events: []`); the kernel journal shows no Xid in 17:02–17:03. D092 lifts the block only through §9's "further residual by a new row" (note A). |
| VT-23 (+ VT-23b) | NOT MET → RR-VT23 (D073) and RR-VT23b (D093). VT-23b **FAIL as registered**, C4 only | Recomputed from `onestep.jsonl`, 9 steps: C1 ≤ 5.9e-8, C2 ≤ 2.7e-7, C3 ≤ 1.2e-6, **C4 2.1e-4 to 1.06e-3 (limit 1e-4)**. Trainer run `63658a1`, code clean, `journal_ok = true`. I re-ran `diag23.py` (s = 0, 8, 35, 48 only): reference rounded to fp32 matches to ≤ 3.2e-5, the worst tensors (`dt_bias`) match exactly after rounding, `v` ≤ 3.3e-7, ulp floor up to 7.2e-3. This supports D093's reading. The script is tracked (`experiments/ops_5080_queue/diag23.py`); I found no saved output. |
| VT-24 (VT-24r v2, K0–K6) | **NOT MET: blocking** | `vt24r_summary.json` reads **FAIL** (K4 false). See section 2.1. Items: 100/100 dev and 126/126 npz test content hashes match the manifests (the one non-npz file matches by file SHA-256). Against `k3_baseline.json`, exactly the 5 repaired cells changed among 218; 8 cells are new (R6). Manifests committed. K1/K2 per-file rows: 116 retrieval and 48 M-hop files all pass; `k1k2.json` reports 78,800 retrieval queries, 0 failing. K5b baseline and after agree. |
| VT-25 (+ SUITE-PF O-d) | Registered verdict MET (D075); **O-d not evaluable: blocking** | `vt25_transplant.jsonl`: worst matched-bf16 GDN gradient 1.816e-2 ≤ 2e-2. I found no output of the 10-seed sweep (`vt25_seed_sweep.py`; the ledger says not run) and no recorded pinned value x, so no tier (including whether RR-VT25 fires) can be applied. |
| R5 not FAIL | MET (RR-R5-D1, RR-R5-power, D079) | `r5_table.md` / `r5_results.json`: PASS. Not re-derived. |
| R6 not FAIL | R6 v2 **FAIL as registered** → RR-R6 (D094); not blocking only by that row | Recomputed from per-query rows: R1 D=3 0.4575, R-load N=8 0.30625, score **0.381875 < 0.50** at 150M. AA chain flips 8/6400 = 0.125% (chain file). `63658a1`, code clean, `.cache/triton-diag`, `journal_ok = true`. R6 started 17:05 although its preconditions were not shown met: the VT-24r summary written at 16:43 read FAIL, and I found no C1 log. §9 says "FAIL → blocking"; see note A. RR-R6 plausibly differs across k. |
| R2 (a), (b) live (R2-LIVE v2, C1 + C2) | **NOT MET: blocking** | See section 2.2. |
| R2(c) | MET | Maximum over `experiments/repro/r2/profile/*.json`: 25,242.9 µs = 25.24 ms < 1 s. |
| R7 complete | MET as worded; Pythia FAIL → RR-R7 (D076) | Summary JSONs: Pythia-160M fp32 0.3538 (band 0.313–0.343) and 70M 0.2271 (0.17–0.20) fail; GDN-340M 0.3435 (0.328–0.368) passes. Not re-scored. |
| O4 | MET (O4-A PASS); O4-Z and O4-B reported (RR-Xid-Z, D095) | O4-A recomputed: all four probe traces (A1, A3, two processes each) are identical in loss, grad-norm and batch hash; parameter hashes agree in `o4_result.json`. S1–S4 each ≥ 1,802 s. Commit `ff652a47`, code clean. The A1, A3 and S1–S4 infra JSONs read `journal_ok = true`; my journal scan shows no Xid from A1 to A3. S1–S4 registries have no terminal row (SIGTERM time box). **O4-Z ran 0.663 h, below the required 60 min** (section 2.3). Its meaning (E_Z ≥ 1) is unaffected: 3 hangs (6 log lines). Precondition "C1 PASS" not shown. |
| Resume re-verification (VT-13r) | MET after an evaluator correction (D096); first output FAIL stays on record; subject to the C2 reading (2.2) | See section 2.4. |

**Note A (D092–D095).** The rows exist, are marked "APPROVED by the user", and D092–D095 were committed in `5fe896d` (25 Nov 20:50:55). §9 allows a further residual by a new row, so I count them. Reservations:
- Per D092 the approval was one blanket "approve all the recommendations", on recommendations drafted in the executor's plan. I found no separate PI-role ruling.
- The VT-22b and VT-23b preregs say "G-RV blocked until resolved". I found no documented walk of their failure trees, only the diagnoses.
- §9 reads "R6 FAIL → blocking". Whether the "further residual" sentence (written about new checks) reaches R6 is an interpretation. The PI should state it.
- The rows lack the §7 fields (differs across k, safeguard), and I found no update of the D090 register.
- D092–D094 cite evidence in `PLAN_TO_FRIDAY.md`, which I found only outside the repository.

## 2. The points requiring particular attention

### 2.1 VT-24r: saved verdict versus separate K4 evidence
My view: **the saved FAIL is the only registered verdict. VT-24r is not shown to PASS.** The FAIL comes from a missing K4 record, not from an observed test failure. The separate evidence does not replace it:
1. `k4_cpu_result.json` records `pass: null`. It has no commit, command or per-test list. The log I found (`~/outeropt-queue/vt12_cpu.log`, 57 passed, 1 deselected) is outside the repo.
2. The CPU run (14:41, 24 Nov) **predates the final item tree**: the R6 dev `.npz` files are newer, the dev manifest changed in `63658a1` (16:43:35), and `src/hybridlc/eval/suites.py` changed in `9abfe8d` (14:58). K4 requires the full VT-12 suite on the regenerated items with manifests verifying by content. I found no single run of it on the final tree. My content-hash check passes, but it is not the suite.
3. The one GPU test passed at 16:43 (`VT12-gpu-test.log`). The summary built seconds later still read K4 false; `k4_result.json` is absent.
4. `STATUS.md`, the audit checklist (K4 CPU part) and `PLAN_TO_FRIDAY.md` ("K0–K6 PASS") report more than the saved record shows.

The prereg's fail clause calls for a re-run under `VT-24r.v3.yaml` written before it runs. The current FAIL stays on record.

### 2.2 R2-LIVE
**C1.** I found no `c1_pytest.log`. The "30 passed" note is a CPU run that I found only in notes, and does not show the GPU case. SUITE-PF run 2 (full suite, no skips beyond the 4 expected) implies that the tests of `test_r2_live.py` did not fail later. C1 was required, with its log, before any Phase II job. Not met as registered.

**C2.** Not clean, and the window is not closed (it ends at a registered SUITE-PF). My scan (about 06:40):
- `lock_ok`: all 21,938 project-process samples true. No monitor-recorded violation.
- Unnamed `[No data]` processes: 8 samples from O4-Z through SUITE-PF-2; 1 more in the R6 replicates and 1 in FEAS-SMOKE. The five O4 ones (O4-Z, S1–S4) fall within 0.4 s of their window's end, so they are probably the project's own exiting process. I did not check the others' timing. The rule has no exemption for them.
- CARLA-env python: 5 samples in O4-Z. The monitor samples do not measure idleness.
- Rows without `journal_ok = true`: VT-13r r2 and r3, which belong to a gate run (not just a "test registry"). The FEAS-SMOKE injected `infra_failed_attempt` rows also lack a top-level `journal_ok`. VT-13r r3 has a `crashed` row with no infra fields.
- VT-22b has no infra block.
- Four study-registry FEAS `interrupted` rows have no infra fields (they fall after SUITE-PF run 2 and before any future registered run, so inside the open window).
- The FEAS-SMOKE injected-fault (fakes) registry has a `paused` row with no infra fields.
- Two or more project processes: seen in R6, SUITE-PF and SUITE-PF-2 samples (all `lock_ok`). The prereg's C2 list does not forbid this; plan §2.2(a) wording ("no two project GPU processes during a timed or gate run") needs a PI reading.

Literally applied, the invalidation clause would void every gate run whose window holds a violation (potentially including O4-A, VT-13r, VT-22b and SUITE-PF) and allows only one re-judgement. I do not apply that unilaterally. A PI row must rule on it.

**Script logic** (`scripts/repro/r2_live_c2.py`). Its `zero_violations: false` is correct, but a "true" from it would not be trustworthy:
- CARLA-env samples are excluded from the violation total without any idleness test.
- `TERMINAL` omits `interrupted`, so the four FEAS rows are missed.
- Requirement 3 covers only registry rows. Runs with no registry row or no infra block (VT-22b) cannot be flagged.
- The registry list omits the nested FEAS-SMOKE fakes registries (hence the `paused` row).
- `--final` changes only the label. The window has no end bound.
- `lock_ok` is flagged only when exactly `False` (no other non-true value occurred in my scan).
- Foreign names are allowed by substring match.
- The committed JSON dates from 02:25 and predates FEAS-SMOKE attempts 3–4.

### 2.3 O4-Z: short exposure, not in any decision row
The driver stopped O4-Z at 2,385.6 s on a `[No data]` "foreign process" (child 20, `SIGTERM(foreign)`). `state.json` carries the note "Part Z treated as VALID, not repeated"; the reason reads `timebox_equivalent(blocked_foreign_false_positive)`, and `o4z_result.json` says `valid_window: true`. The prereg requires 60 min and says a blocked window is repeated; T31 requires ≥ 60 min. **I found this in no decision row (D096 included) and not in the ledger**, only in `state.json`, the generated progress file and the outside-repo HANDOFF. Also, most O4-Z and O4-B children exited with rc −11. In the five rc −11 O4-Z child logs I sampled, a faulthandler "Timeout (0:00:20)!" dump appears near the end, and many such exits have no matching kernel event. I did not sample O4-B logs. The cause is not established, and the harness may be involved. Neither part is a pass criterion.

### 2.4 VT-13r after the evaluator fix
- R1 recomputed: p1, r1, r2 and r3 traces match over 200 steps (loss, grad-norm, batch hash). Final parameter hashes agree in the result file.
- `0f5d3d9` fixes two real evaluator bugs: the last registry row is `attempt_end`, not `completed`; and `failure_counts()` nests its result under `runs`. The first and second outputs differ only in R2/R3, the verdict and that nesting. I accept the correction. Strictly, the prereg's "new v2 file" clause is aimed at resume defects. D096 approves the re-evaluation.
- Commit `63658a1`, code paths clean. No change to `src/`, `configs/`, `train.py` or `runner.py` from there to `3c5119f` or HEAD, so the mandatory re-run was not triggered by those paths.
- **Gap:** the driver points `HYBRIDLC_JOURNAL_SOURCE` at an injected-journal file whenever a fault is set. `journal_ok` is false on r2's injected attempt and on all of r3 (r3 has no injection), so the real kernel journal was not consulted for those attempts. My scan finds no real Xid from 16:43 to 17:03.

### 2.5 R6, VT-22b, VT-23b and rows D092–D094
Numbers confirmed (section 1). Recording is adequate to identify each approval, with the reservations in note A. One further gap: VT-22b's C12 fp32 arm used R-ATT mode **`efficient`** (`vt22b.py` tries it first; the jsonl records it), while the code default for FEAS evaluation is **`tiled`** (`R-ATT:tiled-fp32` in smoke P4 and most k=12 32K P3 metadata). So the blocking long-context check did not exercise the reference path used there. T14's accuracy criterion compares with MATH only up to 4096. A search of `decisions.md`, `STATUS.md` and the ledger found no row or note on the choice of `tiled`. A tiled cross-check at 16K and 32K is needed before P4 or P3 decide anything. I do not claim `tiled` is wrong.

### 2.6 FEAS-SMOKE
**Registered verdict: not passed in any recorded attempt. Attempt 4 is also not a run of the registered test.**

| Attempt | Commit | Record |
|---|---|---|
| 1 | `9733c15` | FAIL (duplicate run ids; k=12 P3 OOM) |
| 2 | `7990d2d` | FAIL (`eval_exit`, W3, blinding control, fixture trace). Its "eval_meta ok" checked nothing |
| 3 aborted | not stated in its record | no result (wrong weights for P1/P3/P4) |
| 3 | `b3571c9` | FAIL, stopped at 0.3 GPU-h |
| 4 | `5ab7c79` | FAIL; the only failing driver item is W3 (1,424 new files). About 0.43 GPU-h |

| Change | Effect on registered criteria |
|---|---|
| `be68a66`, `7990d2d` | Continuing after a failed evaluation does not weaken (`eval_exit` still fails). The `gdiv.fallback_lr` offset goes beyond "shortened only by schedule/cadence overrides" (disclosed; no effect without a G-div restart). YAML copies replace CLI overrides (disclosed) |
| `2fd7924` | Item-name fix; requiring all P4 cells strengthens |
| `be48291` (D099) | **Relaxes a registered precondition** (memory fraction 0.44 → 0.85 for k=12 P0 + re-score). Attempts 1–2 failed `eval_exit` at 0.44. Attempt 4's `eval_exit` is therefore not met at the registered configuration |
| `b3571c9` | Strengthens (fp32 weights checked; stale results removed) |
| `5ab7c79` (D101) | **Relaxes the registered budget stop** (0.3 → 0.5 GPU-h; attempt 4 exceeds 0.3). The per-cell metadata check strengthens a vacuous one |

- In the diffs I read, no pass-item threshold was lowered. But D099 and D101 change registered conditions after attempts had run, and I found no new prereg version. "Only W3 fails" is true of the driver, not of the registered test.
- **D101 timestamp:** the row says approval "about 06:08 IST", but it was committed in `f13bbeb` at **06:05:07**, the same second as the driver change and attempt 4's lock acquisition. The recorded time is inconsistent with the commit. I cannot verify when approval was given.
- D099 and D100 are consistent with their commit time (05:42:49). D100 quotes attempt 2's counts (1,077 files; 129 of 133 fp32-only). Attempt 4's record shows 1,424 files and 172 fp32-only plus 4 bf16+fp32 kernels. D100 also sets aside TC-PF's registered remedy (re-warm and repeat).
- Driver checks narrower than the prereg: the decision trace is tested by case-insensitive substring (not "every ladder rule"); only one bf16 milestone per run is required; `infra_blocks` checks `attempt_end` rows and `summary.json`, not every terminal row (the fakes `paused` row lacks infra fields, so that registered item is arguably unmet); the blinding regex matches only a fixed list of keys followed by `=` or `:` and a number.
- What I confirmed for attempt 4: the statuses in its results and registry; 47 P0, 47 P1 and 18 P4 cells per checkpoint (from metadata files); precision fields by pass; k=12 32K P4 through R-ATT; per-cell wall time recorded; and in its S2 evaluation logs, the only `key=number` forms are cell counts, memory and wall times.

### 2.7 DEV-10
**The identity claim holds.** `src/`, `configs/`, `scripts/train.py` and `scripts/runner.py` have identical git object ids at `3c5119f`, `9733c15`, `7990d2d`, `2fd7924`, `f46fe85`, `5ab7c79` and HEAD, with no uncommitted change there.
- 13 FEAS runs, recorded commits: `3c5119f` (1), `9733c15` (1), `7990d2d` (2), `2fd7924` (3), `f46fe85` (6).
- All 13: `git_dirty_code_paths: []` (tree dirty elsewhere), `allow_dirty` false, completed at 2289 / 229 / 687 / 1000 steps as queued, `final.pt` exists.
- All completing attempts' infra fields: `infra_class` none, `journal_ok` true, 0 events, 0 disallowed foreign processes. No kernel Xid from 23:48 to 04:46.
- Production Triton cache: no file newer than FEAS start (12,821 files).
- FEAS training was **interrupted four times** (`runner stopped`), at times matching smoke work: C12 Stage A at step 94, a C0 Stage-B run at step 1, a C0 TEF run twice (steps 1 and 3). Each restarted with `start_step` 0. These `interrupted` rows lack infra fields (the matching `attempt_end` rows have them).
- Condition (2) is not triggered. Condition (1), nothing read before FEAS-SMOKE passes: FEAS-SMOKE has not passed. D100 permits **scoring**, not reading. No `DTS10-commit` note is in the study registry. I cannot verify that no one read validation output. Reading FEAS output needs an explicit PI row on condition (1) plus the DTS10-commit note.

## 3. §10 checklist

| # | Holds? | Basis |
|---|---|---|
| 1 gate says PERMITTED | **No** | this report |
| 2 rows D058–D091 and fired conditional residuals | Yes for rows to date | RR-Xid entered as RR-Xid-Z (D095). Whether RR-VT25 fires is unknown |
| 3 TC-PF | **No** | Warm, W1, W2 and archive SHA-256 (`a3830d40…`, recomputed) hold; the autotune report shows 45 of 121 common entries differ (report only). W3 fails (D100 residual); I found no R2e-X extraction check |
| 4 R2e-X and D041 re-pricing | **No** | not run per the ledger; D098 only approves doing it later. RR-G0-6's waiver-lapse guard is unevaluated |
| 5 FEAS-SMOKE PASSED on the FEAS commit | **No** | FAIL; attempt 4 ran on `5ab7c79` (training path identical) |
| 6 PR-27 v2, per-k counts, near-miss, R2-LIVE live | Code and tests present; suite run 2 had no failure | Run 2 is unregistered. R2-LIVE gate status remains NOT MET |
| 7 FEAS-PREC passes and R-ATT tested | Substantively yes | T14/T15 GPU logs (10 Nov), suite run 2, smoke. See the `tiled` gap (2.5) |
| 8 DTS10-AT, blinding, commit note before first reading | Tests and smoke blinding yes; note absent | "No reading yet" cannot be verified |
| 9 R8, R10, RP-2 LM-3 helpers | **Not shown** | text search of `scripts/` and `src/` Python found no file naming R10 or RP-2; LM-3 item code exists in `tier0.py`. Seed-101 FEAS Stage A has already finished |
| 10 FEAS YAMLs reviewed and committed | Values spot-checked; **no sign-off found in the ledger** | k 0/12, seeds 101/102, lr 2.0e-3, Stage-A `fallback_lr` 1.4e-3, 30M/90M, SSMax k=12 s101, TEF 1,000 with `answer_norm: block`. DEV-05: only seed 101 run. Milestones and production precision not fully reviewed; this is not a T21 sign-off |
| 11 tree clean and committed | **No** | `experiments/registry.jsonl` (holds the FEAS rows) and three logs modified; untracked smoke, FEAS and results paths. Manifests, `conftest.py` and `benchmark_inference.py` are tracked |
| 12 ledger current, §0 vocabulary | **No** | interim, stale, unaudited |

Other items: **SUITE-PF** not run as registered. **TC-PF W3** not met. Registered **R2e-X** not run.

## 4. Discrepancies between the executor's notes or ledger and the raw files

1. The ledger SUITE-PF row says "590 passed, 0 failed in run 2" and two runs without `--junitxml`. It omits **run 1's 5 failures**, the test patch, the xfail/XPASS/skip counts (O-b requires XPASS to be recorded) and the missing outputs. D096 does not mention the failures.
2. O4-Z's short exposure and manual validation are absent from the decision rows and ledger (2.3). The ledger says only "Xid-8 reproduced".
3. `STATUS.md` says "VT-24r and VT-13r PASS"; the checklist marks K4 (CPU) PASS; `PLAN_TO_FRIDAY.md` says K0–K6 PASS. The saved VT-24r verdict is FAIL. The ledger's correction does not say that the 57-test CPU run predates the final items and manifest.
4. The ledger R2-LIVE C2 row places the bad rows "in crash-injection test registries" (they include VT-13r gate-run rows). It omits VT-22b's missing infra block, the four FEAS `interrupted` rows and the smoke `paused` row. The committed scan is from 02:25.
5. The ledger's C1 note ("30 tests passing") was a CPU run for which I found no saved log.
6. The ledger FEAS-SMOKE row covers attempts 1–2 only and says a PI row is needed. D099–D101 and attempts 3a, 3 and 4 are missing, although the prereg requires every attempt in the ledger.
7. The ledger TC-PF row says "open: needs a PI row"; D100 now exists. Its evidence is superseded by attempt 4.
8. Attempt 2's record lists `eval_meta` as passed; that check was vacuous (admitted in attempt 3's record).
9. Attempt 4's "All other items passed" does not say in the same sentence that it ran under a changed memory cap and time budget and is not the registered test; the exceptions are listed below it.
10. D101's stated approval time is later than its own commit (2.6).
11. The ledger's "VT-13r PASS after an evaluator fix" omits `journal_ok = false` on r2/r3 rows.
12. R6 ran with its VT-24r and C1 preconditions not shown met; I found this in neither the ledger nor D096.
13. Two FEAS preregs are unhashed.
14. The ledger header says T44 is open; this report performs the review, it does not pass the gate.

## 5. What I could not verify
- **Denied command.** The session's permission classifier denied one command (stated reason: "Undermining Oversight"). It combined (a) reading selected fields of five verdict entries in the smoke `feas_decisions.json`, (b) reading the entry counts of the smoke `lm2_seed101.json` and `lm2_fixture.json`, (c) printing the source of `lm2_from_metrics`, and (d) text searches of the decision log, notes and source for the reference attention backend. None of it ran. I did not retry (a), (b) or (c), or obtain them another way. For 2.5 I then ran separate searches of the decision log, `STATUS.md`, the ledger, `attention.py` and `vt22b.py`, which read none of the smoke decision or LM-2 files; the outside-repo notes were left out.
  - Therefore **unverified**: the content of the FEAS-SMOKE decision trace (including whether every ladder and TEF rule was actually evaluated), and the LM-2 inputs to the fixture trace.
  - A key-names-only listing run **before** the denial showed an empty `inputs.lm2` object. I could not follow this up, so I draw no conclusion from it. Whether LM-2 values entered the fixture trace, as FEAS_smoke (b) requires, **should be checked by the PI**.
- That no person or agent read FEAS validation or evaluation output.
- The user's approvals (wording and times) beyond what the rows state.
- The tree and commit of the 57-test CPU run; the commit of suite run 2; the pinned VT-25 value in either suite run.
- VT-21 counts; R5 statistics; R4, G-RP and R7 from per-item or per-run rows; K0 and K3 internals (summaries and my hashes only).
- Anything needing a GPU, including the tests themselves and the `tiled` reference at long length.

## 6. Required before core training
**To clear the four blocking rows**
1. **VT-24r:** write and hash `VT-24r.v3.yaml`, then run the full VT-12 suite (all tests, GPU one under the lock) once on the committed final tree, with JUnit XML. Write a new summary beside the old FAIL.
2. **SUITE-PF (G0 and VT-25):** run exactly as registered (JUnit XML, outcome JSON, 10-seed VT-25 sweep, `summary.json`, `.cache/triton-diag`), record every XFAIL/XPASS/skip, apply O-d (add an RR-VT25 row if in tier; above 4e-2 stays blocking).
3. **R2-LIVE:** rerun C1 and save `c1_pytest.log` (the order failure stays on record). Get a PI row ruling on the existing C2 violations and the invalidation and one-re-judgement clause **before** the re-judging window. Fix the monitor/scan gaps (2.2) in committed code, without silently exempting `[No data]` or CARLA. Then produce a final, clean C2 over a window with at least one complete gate run. Any resume-code change also triggers VT-13r again.

**Also before core training**

4. A PI row (or rows) on: O4-Z's short exposure; suite run 1; the D101 time; R6's preconditions and whether §9 permits D094; the §7 register fields for RR-VT22b, RR-VT23b, RR-R6, RR-Xid-Z and RR-W3-fp32.
5. A new pre-registered 16K/32K check of the `tiled` R-ATT path before P3/P4 results are used.
6. FEAS-SMOKE: a PI row on the standing FAIL (D100 covers FEAS scoring only, not core training), or a conforming attempt after the cache is re-warmed. Resolve the decision-trace and LM-2 questions in section 5.
7. R2e-X v2, its W3 extraction check and the D041 re-pricing row. The Triton cache decision must precede the G5 freeze (add-only rule).
8. §10 items 9, 10 (T21 sign-off), 11 and 12: helpers, a clean committed tree with the study registry, a current ledger that fixes section 4, and hashes for the two unhashed preregs (dated as late).
9. Before any FEAS output is read: the DTS10-commit note and a PI row on DEV-10 condition (1). Label every FEAS and later result "FEAS entry NOT PASSED as registered (DEV-10)".
10. Then a short G-RV re-review of items 1–3. This report's BLOCKED stands until then, unless the PI records an explicit new deviation. Such a deviation would not make the gate PASSED.

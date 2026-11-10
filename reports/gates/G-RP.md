# Gate G-RP: reproduction pilot (S1)

**Verdict: PASS.** Criteria (a) and (b) both pass on our own re-run numbers.

- **Date:** 2026-11-10. **Gate reporter:** S0/S1 gate-report step.
- **Criteria:** protocol §13.3 row G-RP. The wording was amended by DEV-01 (`reports/reproduction/protocol_amendment.md` part D) and the thresholds are unchanged: (b) is evaluated only after VT-25 passes, and each criterion is reported with its per-cell table and minimum detectable defect (MDD).
- **Sources:** `reports/s1_reproduction.md`, `experiments/s1_zoology/{s1a_results.json, s1a_table.md, cut_decision.json, runs/*.json}`, `reports/reproduction/results_R4.md`, `experiments/repro/R4/{prereg.yaml, r4_results.json, runs/*.json}`, `reports/reproduction/results_R5.md`, `reports/reproduction/results_R3.md` (VT-25).
- **Independent recomputation (this step).** `python -I <scratchpad>/grp_recompute.py` reads only the raw run JSONs (`final_overall_acc`, `final_per_config_acc`) and does not call the S1/R4 analysis code. It reproduces every number in the two tables below to 5 decimals.
- **Rule I-20 respected:** every value is a re-run number. No value was read off the paper or the README figure.

---

## Criterion (a): Zoology's own models, best over LR, overall test accuracy over all 7 test configs: **PASS**

Setup:
- Zoology @`1ad20d193b6113cae1e8f3c655c300d7b4b3f4bb` with a clean tree (`zoology_dirty: false` in every run).
- `original_mqar_configs.py` is executed as is. The only changes are `cache_dir`, the wandb-off logger and `run_id`.
- LR grid {1.0e-3, 3.2e-3, 1.0e-2, 3.2e-2}, 16 runs, 1.48 GPU-h. The pre-declared cut did not fire: the first run implied 0.07 GPU-h against a 3.0 GPU-h limit.

| Check | Threshold | Measured (best LR) | Result |
|---|---|---|---|
| Attention, d_model = 64 | ≥ 0.95 | **0.99998** (lr 1e-2; all 4 LRs ≥ 0.99998) | PASS |
| Attention, d_model = 128 | ≥ 0.95 | **0.99995** (lr 1e-3) | PASS |
| GDN, d_model = 128 | ≥ 0.95 | **0.98924** (lr 1e-3; 1024/256 cell 0.925) | PASS |
| GDN, d_model = 256 | ≥ 0.95 | **0.99377** (lr 1e-3; 1024/256 cell 0.957) | PASS |
| GDN non-decreasing in d_model | 128 ≤ 256 | 0.98924 → 0.99377 | PASS |

Evidence:
- `experiments/s1_zoology/s1a_results.json` → `grp_a.verdict = "PASS"`, 4 LRs present in every cell.
- `python -m pytest scripts/s1_zoology/test_s1_analysis.py -q` → 7 passed (S1 session, `s1_reproduction.md` §7). It was re-run in this step; see "Tests run" below.

**Caveats (do not change the verdict):**
1. **Monotonicity and early stopping.** Zoology's default early stop fires at valid acc > 0.99. GDN d256 crosses it in epochs 0–1, while d128 trains all 32 epochs, so the overall-accuracy gap is partly shaped by the stopping rule. The cleaner ordering is on the 1024/256 cell, which lies beyond the training lengths. There every non-divergent d256 run (0.954–0.957) beats every d128 run (0.913–0.925).
2. **Extra point below the grid (R5, not part of (a)).** Zoology's GDN at d_model = 64, lr 1e-3, under bf16 autocast, seeds 123–125, gives overall 0.858 / 0.878 / 0.847 and 1024/256 0.282 / 0.347 / 0.260, with mean 0.296 on the 1024/256 cell. This extends the non-decreasing trend below the gate's grid: 0.296 < 0.925 < 0.957 on 1024/256. It is the only reproduced point below ceiling.
3. **Divergence at the grid edge.** GDN at lr 3.2e-2 diverged at both widths, falling to ≈ 0.005 after reaching 0.97 / 0.95. Best over LR is unaffected.
4. **Ceiling.** (a) is a ceiling check. ±0.03 on overall accuracy corresponds to ±0.21 on the only live cell, 1024/256 (critiques A F2 and B §1.1). R5 is the calibrated, below-ceiling complement (below).

---

## Criterion (b): our mixers in the same harness, within ±0.03 of Zoology's at matched d_model and LR: **PASS (4/4 cells)**

**Precondition.** VT-25, the Zoology ↔ ours weight-copy test, passes:
- R3, `results_R3.md`: GDN matched-bf16 logits ≤ 4.9e-4 (bound 1e-2), worst per-tensor gradient 1.82e-2 (bound 2e-2); attention fp32 logits ≤ 3.3e-8 (bound 1e-5).
- Re-run in this step inside the full suite (see "Tests run").
- R4 ran after VT-25 had passed, as amendment §13.3 requires.

**Run setup.** Pre-registration `experiments/repro/R4/prereg.yaml` was written before any run. Harness, data, seeds (123/123) and shims are as in S1(a). `src/hybridlc/models/` equalled HEAD `110fd9e` for every run.

| Ours | Zoology | Ours overall | Zoology overall | Δ | \|Δ\| ≤ 0.03 | 1024/256 ours / Zoology | Epochs ours / Zoology |
|---|---|---|---|---|---|---|---|
| `our_fg` d64, lr 1e-2 | attention d64, lr 1e-2 | 0.99989 | 0.99998 | −0.00009 | PASS | 1.000 / 1.000 | 1* / 1* |
| `our_fg` d128, lr 1e-3 | attention d128, lr 1e-3 | 0.99989 | 0.99995 | −0.00005 | PASS | 1.000 / 1.000 | 1* / 1* |
| `our_gdn` d128, lr 1e-3 | gated_delta_net d128, lr 1e-3 | 0.98886 | 0.98924 | −0.00039 | PASS | 0.922 / 0.925 | 32 / 32 |
| `our_gdn` d256, lr 1e-3 | gated_delta_net d256, lr 1e-3 | 0.99352 | 0.99377 | −0.00025 | PASS | 0.955 / 0.957 | 2* / 2* |

`*` marks an early stop at valid acc > 0.99.

**MDD.** ±0.03 overall equals ±0.21 on the 1024/256 cell. The largest observed per-cell gap on 1024/256 is 0.003.

Evidence:
- `experiments/repro/R4/r4_results.json` → `grp_b.verdict = "PASS"`. Recomputed in this step from `experiments/repro/R4/runs/*.json`.
- Our GDN d128 tracks Zoology's validation curve within 0.0032 at every epoch (`results_R4.md`).

**Residual differences inherent to our modules** (reported as required, no effect at this resolution):
- F_g has no biases (Zoology's MHA biases are zero-initialised) and no attention dropout (Zoology uses 0.1).
- Our GDN feeds the fla kernel in the input dtype (fp32 here), while Zoology casts to bf16.

**Supporting results, not part of the criterion:**
- **R4b, production F_g** (QK-norm, RoPE θ 5e5, headwise gate, d128). In-distribution mean is 1.0000 in fp32 and 1.0000 in bf16 (criterion ≥ 0.95). Overall accuracy is 0.9981 / 0.9980. The bf16 arm used the FA2 `varlen_attn` path (3,039 calls).
- **R5, the calibrated below-ceiling comparison.** Our GDN under bf16 autocast vs Zoology's GDN at d64, seeds 123–125.
  - Primary cell 1024/256: Δ = −0.0175, SE_W 0.0415, FAIL threshold 0.1154. **PASS.**
  - Power is low: the MDD is ≈ 0.115 at 50% power and 0.154 at 80%. R5 rules out a bf16 handicap larger than about 0.11–0.15 on that cell, but not one of a few points.
- **Config-A geometry** (exploratory, single seed): d128 overall 0.932, d256 0.979. Descriptive only.

---

## Deviations in S1 (all recorded in DEV-01 / `s1_reproduction.md` §5; none changes the computation beyond float order)

1. **fla `head_first` shim.** Zoology passes `head_first=False`, which fla 0.6.0 rejects. The shim drops the keyword when it is False. The layout is identical, so the math is unchanged.
2. **Exact micro-batch-64 gradient accumulation.** The full batch at T = 256 needs about 2 GiB of fp32 logits, which does not fit under the 0.44 cap. The accumulated gradient equals the full-batch gradient up to summation order (unit-tested). Only the attention dropout RNG stream differs.
3. **Per-epoch checkpoint/resume shim.** It was added after watchdog crashes. No reported (a) or (b) run was resumed. `gated_delta_net-d128-lr1.0e-02` is reported from attempt 3, a fresh uninterrupted run, after two RC-watchdog/Xid-8 crashes. Attempt 3 was launched by an uncommitted `run_queue.py` change; DEV-01 notes that this change must be committed.
4. **Dependencies.** `pydantic`, `wandb` and `torchvision==0.26.0+cu128` were installed with `--no-deps`. torch, triton and fla are unchanged.

## Infrastructure caveat (affects gate runs generally, not this verdict)

- RC-watchdog + Xid 8 events hit project processes: 2 in S1(a), 1 in R4's exploratory config-A d128 run, and 4 in R5 windows. Every R5 hit was repeated in full.
- The display runs on this GPU with a 7 s RC watchdog. R2(c) measured the longest kernel at 25 ms, so a single long kernel does not explain the events.
- Trigger O4 (soak) and the project-wide single-GPU-queue rule (R2(a)) remain PI items under G-RV (`results_R2.md`, `results_R5.md` §"Infrastructure").

## Relation to G-RV (DEV-01)

G-RP is one input to the new gate G-RV, which must pass before FEAS. The other G-RV inputs are outside this report and are **not yet passed**:
- VT-21 to VT-24 FAIL against their frozen tolerances, with no code defect isolated (`results_R3.md`);
- R6 has not been run;
- R2(a) and (b) are not live project-wide;
- R7's Pythia anchors FAIL in fp32 but reproduce at fp16, and the scorer is not implicated.

Each of these needs a PI decision row (`reports/gates/G0.md` §"Before S2 FEAS").

## Tests run in this step

| Command | Result |
|---|---|
| `python -I <scratchpad>/grp_recompute.py` (raw run JSONs only) | (a): attention 0.99998 / 0.99995, GDN 0.98924 / 0.99377, monotone. (b): Δ −0.00009, −0.00005, −0.00039, −0.00025, all within ±0.03 |
| `python -m pytest tests/test_vt25_zoology_transplant.py` (inside the full suite, `runs/_logs/g0_full_suite.log`) | see `reports/gates/G0.md` §1 |
| `python -m pytest scripts/s1_zoology/ -q` | see below |

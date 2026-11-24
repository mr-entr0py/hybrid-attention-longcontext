# G-RV audit checklist (draft, 2026-11-24; source: reports/pi_rulings_2026-11-10.md section 9)

Verdict line is fixed by the PI: "NOT PASSED as registered". The audit states separately whether FEAS entry is PERMITTED
(no row ends in "blocking" and every named RR has its decision row). Mark each line with the evidence path and commit.

| # | Condition | Needs | Evidence file | Status |
|---|---|---|---|---|
| 1 | G0 (RR-G0-1..6, DEV-02) | D058-D061 rows; SUITE-PF O-a holds | `experiments/repro/SUITE-PF/summary.json` | open (GPU) |
| 2 | G-RP (a),(b) | D062 | - | MET |
| 3 | VT-21 | RR-VT21 row (D071) | - | check row exists |
| 4 | VT-22 | RR-VT22 row + VT-22b v2 PASS | `experiments/repro/VT-22b/` | open (GPU, `vt22b.py`) |
| 5 | VT-23 | RR-VT23 row + VT-23b PASS | `experiments/repro/VT-23b/` | open (GPU, `vt23b.py`) |
| 6 | VT-24 | VT-24r v2 PASS (K0-K6), items regenerated, manifests committed | `experiments/repro/VT-24r/vt24r_summary.json` | K0-K3,K4(CPU),K5a/b,K6 PASS; 1 GPU test left |
| 7 | VT-25 | SUITE-PF O-d <= 4e-2 (RR-VT25 row if in (2e-2, 4e-2]) | SUITE-PF summary | open (GPU) |
| 8 | R5 not FAIL | RR-R5-D1, RR-R5-power rows | - | MET |
| 9 | R6 not FAIL | R6 v2 PASS or INCONCLUSIVE; `--` chain flip >= 1% -> PI row | `experiments/repro/R6/r6_results.json` | open (GPU, `r6.py run`) |
| 10 | R2 (a),(b) live | R2-LIVE v2 C1 PASS (`tests/test_r2_live.py`, `scripts/repro/test_r2_tooling.py`) + C2 zero violations over O4-Z..SUITE-PF | `experiments/repro/R2-LIVE/c2_compliance.json` | C1 tests pass on CPU (30 passed, 2026-11-24); C2 file not written |
| 11 | R2(c) recorded | - | - | MET (25.24 ms) |
| 12 | R7 | RR-R7 row | - | MET as worded |
| 13 | O4 (plan 2.2) | O4-A PASS; O4-Z, O4-B reported | `experiments/repro/O4/` | running |
| 14 | Resume re-verification | VT-13r PASS | `experiments/repro/VT-13r/vt13r_result.json` | open (GPU, `vt13r.py`) |

## Audit procedure
1. Run only after every GPU row above has its result file; do not read results of any run whose window holds an infra event.
2. For each result file: check the prereg hash it cites, the git commit recorded in the run (clean tree), and `infra` block (`journal_ok = true`, no flagged events).
3. C2 (T42, CPU): scan gpumon samples for `lock_ok = false` and foreign compute processes across the window; list every violation and invalidated run in the ledger.
4. Confirm each RR named in the table has a decision row; any FAIL of a new check keeps FEAS blocked.
5. Write `reports/gates/G-RV.md` (verdict line + FEAS-entry statement), independent of the author of the checks.

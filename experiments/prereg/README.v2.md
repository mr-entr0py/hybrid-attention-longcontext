# Pre-registrations, batch 2 (PI FINAL rulings, 2026-11-10; written before any of these items ran)

Source of authority: `reports/pi_rulings_2026-11-10.md` (status FINAL) and decision rows D058–D091 in
`reports/decisions.md`. Batch 1 (`README.md`, `SHA256SUMS`, written 13:38–13:44) stays on record unchanged; files of
batch 1 that batch 2 supersedes were never run. Batch 2 is hashed in `SHA256SUMS.v2`.

## Rules

Rules 1–5 of `README.md` apply unchanged to every file of both batches. Two rules are added:

6. **Verdict vocabulary** (decision D088). A gate or item verdict reads **PASSED** only if every condition was met as
   registered. Otherwise it reads **NOT PASSED as registered**, and each condition is reported as MET, MET via new
   check (route ii; the original FAIL stays), NOT MET → named residual RR-… (route i, with its decision row), or
   NOT MET → blocking. Permission to proceed (for example FEAS entry under G-RV's named-residual clause) is a
   separate line. "PASS-WITH-…", "conditional pass" and counts such as "N/N" that omit skips are not used.
7. **Unexpected passes.** A test that asserts a registered criterion already failed as registered is marked
   `xfail(strict=False)`. An XPASS is recorded with its measured values and reported to the PI. It is not a PASS of
   the item and never blocks; the item's verdict remains its registered verdict run.

## Files in force (batch 1 files not listed here are superseded)

| File | ID | Version | Gate role | Supersedes |
|---|---|---|---|---|
| `O4_soak.v2.yaml` | O4 (O4-Z meaning, O4-B diagnostic, O4-A verdict) | 2 | environment; before any gate run | `O4_soak.yaml` |
| `R2_live.v2.yaml` | R2-LIVE | 2 | G-RV ("R2 (a)/(b) live", route ii) | `R2_live.yaml` |
| `R2e_G0T_exclusive.v2.yaml` | R2e-X | 2 | budgeting re-measurement; guard references; D029 reading | `R2e_G0T_exclusive.yaml` |
| `VT-22b.v2.yaml` | VT-22b | 2 | G-RV (blocking) | `VT-22b.yaml` |
| `VT-23b.yaml` | VT-23b | 1 | G-RV (blocking) | — (unchanged) |
| `VT-24r.v2.yaml` | VT-24r | 2 | G-RV, R6, FEAS (blocking) | `VT-24r.yaml` |
| `R6.v2.yaml` | R6 readings | 2 | G-RV ("R6 not FAIL") | `R6.yaml` |
| `FEAS_precision.v2.yaml` | FEAS-PREC | 2 | FEAS verdict source (Q4); S4 scoring precision (Q1, Q3); VT-06 bf16 sub-item (Q2) | `FEAS_precision.yaml` |
| `DTS10_acceptance.v2.yaml` | DTS10-AT | 2 | any FEAS verdict | `DTS10_acceptance.yaml` |
| `VT-13r.yaml` | VT-13r | 1 | G-RV / FEAS entry (blocking) | — (new) |
| `VT-13p2.v2.yaml` | VT-13p2 | 2 | future production-only resume changes | `VT-13p2.yaml` (trigger only) |
| `G0suite_preFEAS.yaml` | SUITE-PF | 1 | G0 no-regression condition; G-RV | — (new) |
| `TRITON_prefeas.yaml` | TC-PF | 1 | FEAS entry | — (new) |
| `FEAS_smoke.yaml` | FEAS-SMOKE | 1 | FEAS entry | — (new) |

## Phase II order (one GPU job at a time through the lock, memory fraction ≤ 0.44)

O4-Z → O4-B → O4-A → TC-PF → FEAS-SMOKE → R2e-X → VT-13r → VT-22b → VT-23b → R6 → SUITE-PF; then R2-LIVE C2
(CPU) and the independent G-RV gate (CPU). VT-24r runs on CPU in Phase I.

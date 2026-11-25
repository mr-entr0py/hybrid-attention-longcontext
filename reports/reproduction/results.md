# Reproduction and gate ledger (T02/T43)

**INTERIM, 26 Nov 2026 about 02:30.** Compiled by the executor from the result files and decision rows below. It has not been audited (T44 is open and must be done by someone else). Vocabulary: D088. A FAIL stays a FAIL when a residual risk (RR) or a new check covers it.
**FEAS entry is NOT PASSED as registered** (DEV-10: FEAS training started before the entry checklist held).

| Item | Claim / criterion | Registered verdict | Route / follow-up | Source |
|---|---|---|---|---|
| R1, G-RP(a) | Zoology MQAR reproduction (attention at least 0.9997; GDN 0.989 / 0.994, non-decreasing) | PASS | — | D062; experiments/s1_zoology/ |
| R2(a) one GPU queue | live before R4 | **FAIL (as registered)** | RR-R2; new check R2-LIVE v2 | D077; results_R2.md |
| R2(b) event tagging | live before R4 | **FAIL (as registered)** | RR-R2; new check R2-LIVE v2 | D077; results_R2.md |
| R2(c) kernel profile | longest kernel under 1 s | PASS (25.24 ms) | — | D081 |
| R2(d) | process identified | recorded | — | D081 |
| VT-21 | fla causal conv vs reference | **FAIL (as registered)** | RR-VT21 | D071; results_R3.md |
| VT-22 | long-length transplant | **FAIL (as registered)** | RR-VT22; new check VT-22b | D072 |
| VT-22b v2 | F1–F4, B1, B2 | **FAIL (as registered)**: F4 only | RR-VT22b | D092; experiments/repro/VT-22b/ |
| VT-23 | trainer vs independent loop | **FAIL (as registered)** | RR-VT23; new check VT-23b | D073 |
| VT-23b | C1–C5 | **FAIL (as registered)**: C4 only | RR-VT23b | D093; experiments/repro/VT-23b/ |
| VT-24 | oracle answerability | **FAIL (as registered)** (7 of 39,800) | generator fix; new check VT-24r v2 | D074; results_R3.md |
| VT-24r v2 | K0–K6 | PASS | — | experiments/repro/VT-24r/ |
| VT-25 | Zoology weight transplant | PASS (registered verdict run) | 10-seed sweep not yet run | D075 |
| R4 / G-RP(b) | within ±0.03 of Zoology, 4 cells | PASS | — | results_R4.md |
| R5 | phenomenon reproduction | PASS (deviation D1) | RR-R5-D1, RR-R5-power | D079; results_R5.md |
| R6 v2 | 2-layer known answer at 150M | **FAIL (as registered)** (0.382 < 0.50) | RR-R6 | D094; experiments/repro/R6/ |
| R7 | published checkpoints | **FAIL (as registered)** on both Pythia criteria; GDN 340M PASS | RR-R7 | D076; results_R7.md |
| O4-A | production-shape soak | PASS | — | experiments/repro/O4/o4_result.json |
| O4-Z, O4-B | stress cell (descriptive) | Xid-8 reproduced | RR-Xid-Z | D095; experiments/repro/O4/ |
| VT-13r | resume bitwise exact (R1–R3) | PASS after an evaluator fix (same artefacts re-evaluated; first output kept) | auditor must check | D096; experiments/repro/VT-13r/ |
| TC-PF | W1, W2, W3 | W1, W2 hold. **W3 NOT MET** (see FEAS-SMOKE attempt 2: fp32 evaluation kernels missing) | **open: needs a PI row** | experiments/repro/TC-PF/ |
| FEAS-SMOKE | all listed pass items | **FAIL**, attempts 1 and 2. Open: P3 out of memory on k=12 R-load_N1024_C0; W3. Driver bugs fixed; P4 and blinding clean in a later diagnostic (not an attempt) | **open: needs a PI row, then attempt 3** | experiments/feas_smoke_attempt{1,2}/ATTEMPT.md |
| R2-LIVE C1 | tests | tests pass (30) | — | experiments/repro/R2-LIVE/ |
| R2-LIVE C2 | zero violations, O4-Z to SUITE-PF | **INTERIM scan is not clean** (window open). 10 samples with an unnamed foreign process; 8 rows without journal_ok = true and 1 row without infra fields (in crash-injection test registries). Also reported: 5 CARLA-env client samples (allowed only if idle; not checked) and 76 samples with two project processes (SUITE-PF, R6) | **open: needs PI reading after the final scan** | experiments/repro/R2-LIVE/c2_compliance.json |
| R2e-X v2 | exclusive throughput, TEF and audit cells | not run | open (then the D041 re-pricing row) | — |
| SUITE-PF | registered command plus VT-25 sweep | not run as registered (two runs without --junitxml; 590 passed, 0 failed in run 2) | open: SUITE-PF-3 | D096 |
| G-RV (T44) | independent audit | not run | open: must not be the executor | — |

## Recorded process deviations
- Phase II ran out of the registered order (D096).
- Overnight R6 replicates read the production Triton cache; no file was written (D096; D069).
- SUITE-PF-1 and -2 omitted --junitxml (D096).
- FEAS: seed 101 only (DEV-05), started before the entry checklist (DEV-10).

## Named residual risks
RR-G0-1…6, RR-R2, RR-VT21, RR-VT22, RR-VT22b, RR-VT23, RR-VT23b, RR-R5-D1, RR-R5-power, RR-R6, RR-R7, RR-Xid-Z, RR-D052i. See reports/decisions.md.

# Pre-registrations written by the PI on 2026-11-10 (before any of these items ran)

Source of authority: `reports/pi_rulings_2026-11-10.md` (status DRAFT until the orchestrator accepts it).
Pre-registration integrity documents that each file depends on (re-hashed 2026-11-10, equal to the DEV-01 row):
`reports/reproduction/reproduction_plan.md` 6bc14af6…8adcebc, `reports/reproduction/protocol_amendment.md` bf19a536…0e338e6.

## Rules for every file in this directory

1. A file here is written **before** its item runs. Its SHA-256 is listed in `SHA256SUMS` (written at the
   same time) and quoted in the rulings. An executor copies the file unchanged into
   `experiments/repro/<ID>/prereg.yaml` (or cites it) before the first run and verifies the hash.
2. A criterion may be changed **only before** its item runs, and only by a new file with a new ID or version
   suffix plus a decision row. After an item has run, its criteria are never restated. A FAIL stays a FAIL;
   the only routes are (i) a named residual risk accepted with evidence, (ii) a genuine defect fixed and a NEW
   check pre-registered here before it runs, or (iii) block.
3. Readings that resolve ambiguous text are fixed in the file. If an executor finds an ambiguity that the
   file does not resolve, the executor stops and asks the PI before the item runs; a reading made after any
   result is visible is reported as such and may not create a PASS.
4. Every GPU job runs through `scripts/repro/gpu_lock.py` (or the in-process lock that R2-LIVE integrates),
   one job at a time, memory fraction ≤ 0.44. Reproduction and diagnostic jobs use
   `TRITON_CACHE_DIR=$REPO/.cache/triton-diag`; only production study runs use `$REPO/.cache/triton`.
5. Outcomes: PASS / FAIL as defined per file; BLOCKED when a precondition fails or the environment makes a
   verdict impossible (never counted as PASS); INCONCLUSIVE only where a file pre-declares it.

## Index

| File | ID | Gate role | Blocking for |
|---|---|---|---|
| `O4_soak.yaml` | O4 (O4-A verdict, O4-B diagnostic) | environment | every further gate run and FEAS |
| `R2_live.yaml` | R2-LIVE | environment | G-RV (R2(a)/(b) "live") |
| `R2e_G0T_exclusive.yaml` | R2e-X (R2(e) G0-T + G0-T512x + D029 reading) | measurement + one criterion | G-throughput reference; D512 Stage-A budget |
| `VT-22b.yaml` | VT-22b | reference reproduction (new, complementary) | G-RV |
| `VT-23b.yaml` | VT-23b | reference reproduction (new, complementary) | G-RV |
| `VT-24r.yaml` | VT-24r | item regeneration + oracle check | G-RV, R6, FEAS |
| `R6.yaml` | R6 readings (criteria = plan §2.6, unchanged) | phenomenon + chain consistency | G-RV |
| `FEAS_precision.yaml` | FEAS-PREC (D052(i), VT-06 bf16 sub-item, fp32-reference audit) | FEAS measurement | S4 scoring precision |
| `DTS10_acceptance.yaml` | DTS10-AT | decision-code acceptance | any FEAS verdict |
| `VT-13p2.yaml` | VT-13p2 | future production-resume audits only | — (not scheduled) |

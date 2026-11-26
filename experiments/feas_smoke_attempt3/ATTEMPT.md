# FEAS-SMOKE attempt 3 (2026-11-26 05:44-06:03, commit b3571c9): FAIL - stopped at the registered 0.3 GPU-h limit
Completed: both CPU injections, all 13 training runs, P0/P1/P4 on both k=0 checkpoints, all passes on C12 B030, P0 on
C12 B090 (both k=12 P0 re-scores succeeded at mem fraction 0.85, D099). Not reached: the remaining evaluations, the
blinding control, W3, both decision-code checks. Found afterwards: the eval_meta check of attempts 1-2 looked for
`_meta.json` while metadata is written per cell (`<cell>_meta.json`), so their eval_meta "ok" checked nothing.

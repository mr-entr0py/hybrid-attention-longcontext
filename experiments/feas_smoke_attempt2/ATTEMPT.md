# FEAS-SMOKE attempt 2 (2026-11-26 01:17-01:33, commit 7990d2d): FAIL
Passed: suspect_sw, near_miss, infra_injection (classified and resumed), clean_runs, registry_chain, checkpoints,
infra_blocks, eval_meta, decide_refuses_incomplete.
Failed:
1. eval_exit - P3 of R-load_N1024_C0 on all three k=12 checkpoints: CUDA OOM (3.38 GiB allocation, MATH, cap 0.44).
   Real evaluation defect; same path as the full FEAS evaluation. OPEN: needs a user decision.
2. w3_new_triton_files - 1,077 new files in the fresh extraction, all written during the first evaluation pass;
   129 of 133 new kernels are fp32-only (the TC-PF warm holds bf16-mode kernels). Training wrote 0 new files to the
   production cache during FEAS. TC-PF W3 is NOT MET for fp32 evaluation passes. OPEN: needs a user decision.
3. blinding - driver bug (the control scored 0 cells). No score was printed for any S2 checkpoint.
4. decide_full_trace_on_fixture - driver bug (P4 scored 0 cells; P1 label doubled).
Driver fixes 2fd7924. Diagnostic (not an attempt, 01:48-01:51): P4 18/18 cells on all five TEF checkpoints, k=12 peak
3.42 GiB; S2 output 0 score leaks; the non-S2 control prints scores.

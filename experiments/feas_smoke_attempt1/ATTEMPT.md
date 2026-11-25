# FEAS-SMOKE attempt 1 (2026-11-26 00:29-00:39, commit 9733c15): FAIL
1. Driver bug: shortened 30M and 90M branches resolved to one run id (90M runs "completed in 0.1 s"). Fixed in the driver.
2. Evaluation defect: P3 (fp32 single-pass re-score) of R-load_N1024_C0 on the k=12 checkpoint ran out of memory at
   mem fraction 0.44 (MATH backend, used below 16K; one 3.38 GiB allocation). Same code path as the full FEAS evaluation
   (scorer.rescore_check). Not fixed: needs a decision (it changes the registered precision rule or the memory cap).

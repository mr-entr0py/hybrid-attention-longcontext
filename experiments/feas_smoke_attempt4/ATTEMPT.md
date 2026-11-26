# FEAS-SMOKE attempt 4 (2026-11-26 06:05-06:30, commit 5ab7c79): FAIL (as registered), only item w3_new_triton_files
All other items passed: both CPU injections, the infra injection (classified and resumed), clean runs, registry chains,
checkpoints, infra blocks, all 15 evaluations (P0 with P2/P3, P1, P4), per-cell precision metadata, blinding (no S2
score printed; the non-S2 control prints), both decision-code behaviours. GPU wall 1,515 s (0.42 GPU-h; limit D101 0.5;
the registered limit was 0.3). Exceptions in force: D099 (0.85 cap for k=12 P0 + re-score), D101 (time limit).
W3: 1,424 new files (list: w3_new_triton_files in w3_new_files.txt), all written after the last training checkpoint
(06:11:49); new kernel pointer dtypes: 172 fp32-only, 4 bf16+fp32. Production cache: 0 files written since FEAS start.
FEAS evaluation is permitted under D100. Mechanism deviation: runs shortened through generated YAML copies; the 90M
copies carry gdiv.fallback_lr 0.0014001 so that the two branches have distinct run ids.
Training path (src/, configs/, scripts/train.py, scripts/runner.py): identical to the FEAS launch commit 3c5119f.

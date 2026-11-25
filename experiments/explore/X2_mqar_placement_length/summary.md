# X2 (exploratory): placement of one attention layer, with longer test contexts

Model: 6-layer [conv, M]×3 at d=64. A = our attention (Zoology-matched), L = our GDN (bf16). Zoology's d64 recipe with early stopping at overall acc > 0.99. Test configs are Zoology's seven plus 2048/512 and 4096/1024.
Not pre-registered, not confirmatory. The queue was stopped at 23:48 by STOP files (GPU deadline 26 Nov 00:00), so seed 125 and the LLL baselines did not run.

| run | 1024/256 | 2048/512 | 4096/1024 | epochs |
|---|---|---|---|---|
| AAA s123 | 1.000 | 1.000 | 1.000 | 1 |
| ALL s123 | 1.000 | 1.000 | 1.000 | 1 |
| ALL s124 | 1.000 | 1.000 | 1.000 | 1 |
| LAL s123 | 0.999 | 0.990 | 0.939 | 6 |
| LAL s124 | 1.000 | 0.995 | 0.932 | 6 |
| LLA s123 | 1.000 | 0.984 | 0.671 | 32 |
| LLA s124 | 1.000 | 0.977 | 0.565 | 32 |

Observation, 2 seeds: at 4096/1024, a single attention layer placed first matches all-attention, placed in the middle reaches about 0.94, and placed last reaches about 0.57–0.67. The ordering holds in both seeds, even though LLA trained longest (no early stop).

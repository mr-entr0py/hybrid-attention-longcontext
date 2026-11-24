# Coverage of the four research questions and H1-H6 by 2026-11-27 (team requirement, 2026-11-24)

Requirement: answer RQ-A..RQ-D and test **all** hypotheses H1-H6. What is achievable, and at what evidence level
(D384 hours on an idle RTX 5080, from `scripts/planning/budget.py`; estimates):

| Hypothesis / RQ | Needs | GPU-h | Evidence level by the 27th |
|---|---|---|---|
| H1, H1-min (RQ-A) | core k in {0,1,3,12}, 3 seeds, frozen evals | 23 (5 seeds: 38) | confirmatory, reduced power (3 seeds, logged deviation) |
| H3 (RQ-C) | same runs, retrieval vs multi-hop vs aggregation | 0 extra | confirmatory, reduced power |
| H5 | same runs, evaluated at 1K-32K | 0 extra | confirmatory, reduced power |
| H2 (diminishing returns, RQ-A) | add k in {2, 6} | 7.6 | descriptive (single seed) |
| H4 (placement, RQ-B) | early/late placement at k* | 18.0 | descriptive (single seed) |
| H6 (RQ-D, cost frontier) | analytic costs (done: `docs/SCALING_STUDY.md`) + measured cost session 2 | 3.0 | descriptive |
| H6' (state vs KV at matched memory) | GDN-E control | 8.3 | descriptive (single seed) |
Mandatory before any of it: O4 soak, gate checks, FEAS pilots (9.8), LR sweep (4.2), eval (~10).

Total ~ 90-94 h against ~69 h of 5080 time (Tue 16:30 - Fri 14:00): short by ~21-25 h. Levers (each a logged deviation):
a second worker GPU for the single-seed Tier-B items, the smallest Stage-B budget, a reduced LR grid, one-seed FEAS.
Priority if it must be cut: core (H1/H1-min/H3/H5) > curve k=2,6 (H2) > GDN-E and cost (H6/H6') > placement (H4).
If a hypothesis cannot be tested in time, the talk says so explicitly rather than implying a result.

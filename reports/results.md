# Results (evolving)

**State: no main-study results yet.** Every training run of the main study is pending (S2+). Validation and reproduction results are listed below.

## Validation and reproduction (pipeline correctness, not study findings)
- **R1/G-RP(a), Zoology MQAR:** attention ≥ 0.9997 (d64/d128). GDN 0.989 (d128) and 0.994 (d256), non-decreasing in d. Source: `reports/gates/G-RP.md`.
- **R4/G-RP(b), our mixers in Zoology's harness:** within 0.0004 of Zoology in all 4 cells. Source: `reports/reproduction/results_R4.md`.
- **R5, MQAR d64:** the phenomenon reproduces below ceiling (0.296 → 0.925 → 0.957). Ours vs Zoology: Δ = −0.018, threshold 0.115 (low power). Source: `results_R5.md`.
- **R7, LAMBADA via our scorer:** GDN-340M 0.3435 (PASS). Pythia fp32 is NOT PASSED as registered; the fp16 anchors reproduce exactly. Scorer agreement was checked three ways. Source: `results_R7.md`.

## Observations (operational)
- Idle training throughput and memory: `experiments/bench/train_step_idle_20261110*.json`.

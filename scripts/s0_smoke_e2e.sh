#!/bin/bash
# S0 end-to-end smoke (integrator): Stage A 100 -> branch -> Stage B 20 -> TEF 20 -> DC scoring
# (10 contexts/cell) -> analysis. Everything goes under experiments/s0/smoke/ (own registry, runs,
# checkpoints and results), never into the study's registry.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
source .venv/bin/activate
export PYTHONHASHSEED=0
S=experiments/s0/smoke
ROOTS=(--runs-root $S/runs --ckpt-root $S/ckpt --registry $S/registry.jsonl)
for st in A B T; do
  echo "=== $(date +%T) train $st"
  python scripts/train.py --config $S/yaml/$st.yaml --allow-dirty "${ROOTS[@]}"
done
TRID=$(python scripts/train.py --config $S/yaml/T.yaml --print-run-id "${ROOTS[@]}" | tail -1)
echo "=== $(date +%T) eval DC on $TRID"
python scripts/evaluate.py --ckpt $S/ckpt/$TRID/milestones/step_0000020_final_bf16.pt --suite DC \
  --n-contexts 10 --label tef20 --results-root $S/results --runs-root $S/runs --rescore
echo "=== $(date +%T) analysis"
python scripts/analyse_results.py --results $S/results --registry $S/registry.jsonl --out $S/analysis --B 200
echo "=== $(date +%T) done"

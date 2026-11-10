#!/usr/bin/env bash
# R7 resumable launcher: each run is skipped if its summary exists (r7_run.py).
# Usage: setsid nohup bash scripts/repro/r7_launch.sh > experiments/repro/R7/logs/launch.log 2>&1 &
set -u
ROOT=/home/sedrica-lynx/Downloads/outeropt
cd "$ROOT"
source .venv/bin/activate
RUNS=(
  # PRIMARY (pre-registered pass/fail): lambada_openai, tokenizer-default BOS, fp32
  "pythia-160m openai default fp32"
  "pythia-70m openai default fp32"
  "gdn340m openai default fp32"
  # Wang criterion alternative (plan: "or on lambada_standard if only that one matches; both reported")
  "gdn340m standard default fp32"
  # sensitivities (not criteria)
  "pythia-160m openai on fp32"
  "pythia-70m openai on fp32"
  "gdn340m openai off fp32"
  "pythia-160m openai default bf16"
  "pythia-70m openai default bf16"
  "gdn340m openai default bf16"
  "pythia-160m standard default fp32"
  "pythia-70m standard default fp32"
)
for r in "${RUNS[@]}"; do
  set -- $r
  name="$1_$2_bos-$3_$4"
  echo "[$(date -Is)] start $name"
  python scripts/repro/r7_run.py --model "$1" --variant "$2" --bos "$3" --precision "$4" > "experiments/repro/R7/logs/$name.log" 2>&1
  echo "[$(date -Is)] end $name rc=$? $(tail -1 experiments/repro/R7/logs/$name.log | cut -c1-300)"
done
echo "[$(date -Is)] ALL DONE"

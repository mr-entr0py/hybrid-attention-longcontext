#!/usr/bin/env bash
# R7 second pass (resumable): waits for r7_launch.sh to finish, then reruns anything without a
# summary (incl. the Pythia BOS sensitivities that failed on the snapshot glob) and adds the
# fp16-weights diagnostic (the published-table precision) through OUR scorer.
set -u
ROOT=/home/sedrica-lynx/Downloads/outeropt
cd "$ROOT"
while pgrep -f "scripts/repro/r7_launch.sh" > /dev/null; do sleep 10; done
source .venv/bin/activate
RUNS=(
  "pythia-160m openai on fp32"
  "pythia-70m openai on fp32"
  "pythia-160m openai default bf16"
  "pythia-70m openai default bf16"
  "pythia-160m standard default fp32"
  "pythia-70m standard default fp32"
  "pythia-160m openai default fp16w"
  "pythia-70m openai default fp16w"
  "gdn340m openai off fp32"
  "gdn340m openai default bf16"
)
for r in "${RUNS[@]}"; do
  set -- $r
  name="$1_$2_bos-$3_$4"
  echo "[$(date -Is)] start $name"
  python scripts/repro/r7_run.py --model "$1" --variant "$2" --bos "$3" --precision "$4" > "experiments/repro/R7/logs/$name.log" 2>&1
  echo "[$(date -Is)] end $name rc=$? $(tail -1 experiments/repro/R7/logs/$name.log | cut -c1-300)"
done
echo "[$(date -Is)] ALL DONE (pass 2)"

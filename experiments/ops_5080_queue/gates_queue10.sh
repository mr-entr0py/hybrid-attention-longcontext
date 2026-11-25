#!/bin/bash
# Queue 10: descriptive R6 6000-step extensions (non-gate, diagnostics cache) as stoppable GPU work until 23:00.
# Stop before the next job: touch ~/outeropt-queue/STOP_FILLER
cd ~/Downloads/outeropt || exit 1
L=~/outeropt-queue/gates; PY=.venv/bin/python; ts(){ date "+%F %T"; }
export TRITON_CACHE_DIR=$HOME/Downloads/outeropt/.cache/triton-diag
for spec in "AA 303" "GDN 503" "AA 304" "GDN 504" "AA 305" "GDN 505" "AA 306" "GDN 506" "AA 307" "GDN 507"; do
  [ -e ~/outeropt-queue/STOP_FILLER ] && break
  left=$(( $(date -d "2026-11-25 23:00" +%s) - $(date +%s) )); [ $left -lt 900 ] && break
  set -- $spec; kind=$1; seed=$2; E=experiments/repro/R6/ext
  echo "[$(ts)] START R6-ext-$kind-$seed" >> $L/queue.log
  timeout --signal=INT --kill-after=60 $left $PY scripts/repro/gpu_lock.py run --label R6-ext-$kind-$seed --wait-exclusive -- $PY scripts/train.py --model configs/model/D384/R6-$kind.yaml --training configs/training/r6.yaml --exp-stage R6 --seed $seed --tag ext6000 --runs-root $E/runs --ckpt-root $E/ckpt --registry $E/registry.jsonl -o schedule.steps=6000 -o 'checkpoint.milestone_steps=[763,1526,2289,3000,4000,5000]' > $L/R6-ext-$kind-$seed.log 2>&1
  echo "[$(ts)] END R6-ext-$kind-$seed rc=$?" >> $L/queue.log
done
echo "[$(ts)] QUEUE-10 DONE" >> $L/queue.log

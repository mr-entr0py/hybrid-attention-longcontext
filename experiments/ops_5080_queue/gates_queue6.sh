#!/bin/bash
# Filler until noon: alternate AA seeds 361.. and GDN("--") seeds 501.. ; stop: touch ~/outeropt-queue/STOP
cd ~/Downloads/outeropt || exit 1
L=~/outeropt-queue/gates; PY=.venv/bin/python; ts(){ date "+%F %T"; }
for i in $(seq 0 40); do
 for kind in AA GDN; do
  [ -e ~/outeropt-queue/STOP ] && break 2
  h=$(date +%H%M); [ "$h" -ge 1142 ] && [ "$h" -lt 1800 ] && break 2
  if [ $kind = AA ]; then seed=$((361+i)); else seed=$((501+i)); fi
  E=experiments/repro/R6/seeds/s$seed
  echo "[$(ts)] START R6-$kind-seed-$seed" >> $L/queue.log
  $PY scripts/repro/gpu_lock.py run --label R6-$kind-seed-$seed --wait-exclusive -- $PY scripts/train.py --model configs/model/D384/R6-$kind.yaml --training configs/training/r6.yaml --exp-stage R6 --seed $seed --tag rep$seed --runs-root $E/runs --ckpt-root $E/ckpt --registry $E/registry.jsonl -o "checkpoint.milestone_steps=[763,1526]" > $L/R6-$kind-seed-$seed.log 2>&1
  echo "[$(ts)] END R6-$kind-seed-$seed rc=$?" >> $L/queue.log
 done
done
echo "[$(ts)] QUEUE-6 DONE" >> $L/queue.log

#!/bin/bash
# 5080 filler, open-ended: alternate AA seeds 378.. and GDN seeds 517.. ; stop: touch ~/outeropt-queue/STOP
cd ~/Downloads/outeropt || exit 1
L=~/outeropt-queue/gates; PY=.venv/bin/python; ts(){ date "+%F %T"; }
for i in $(seq 0 200); do
 for kind in AA GDN; do
  [ -e ~/outeropt-queue/STOP ] && break 2
  if [ $kind = AA ]; then seed=$((378+i)); else seed=$((517+i)); fi
  E=experiments/repro/R6/seeds/s$seed
  echo "[$(ts)] START R6-$kind-seed-$seed" >> $L/queue.log
  $PY scripts/repro/gpu_lock.py run --label R6-$kind-seed-$seed --wait-exclusive -- $PY scripts/train.py --model configs/model/D384/R6-$kind.yaml --training configs/training/r6.yaml --exp-stage R6 --seed $seed --tag rep$seed --runs-root $E/runs --ckpt-root $E/ckpt --registry $E/registry.jsonl -o "checkpoint.milestone_steps=[763,1526]" > $L/R6-$kind-seed-$seed.log 2>&1
  echo "[$(ts)] END R6-$kind-seed-$seed rc=$?" >> $L/queue.log
 done
done
echo "[$(ts)] QUEUE-8 DONE" >> $L/queue.log

#!/bin/bash
# Queue 4: GPU filler so the 5080 never idles after queue 3. Descriptive R6 seed replicates (AA, 2289 steps, seeds 301..), looping until ~/outeropt-queue/STOP exists.
cd ~/Downloads/outeropt || exit 1
L=~/outeropt-queue/gates; PY=$PWD/.venv/bin/python; ts() { date '+%F %T'; }
until grep -q "QUEUE-3 DONE" $L/queue.log; do sleep 15; done
for seed in 301 302 303 304 305 306 307 308; do
  [ -e ~/outeropt-queue/STOP ] && break
  E=experiments/repro/R6/seeds/s$seed
  echo "[$(ts)] START R6-seed-$seed" >> $L/queue.log
  $PY scripts/repro/gpu_lock.py run --label R6-seed-$seed --wait-exclusive -- $PY scripts/train.py --model configs/model/D384/R6-AA.yaml --training configs/training/r6.yaml --exp-stage R6 --seed $seed --tag rep$seed --runs-root $E/runs --ckpt-root $E/ckpt --registry $E/registry.jsonl -o 'checkpoint.milestone_steps=[763,1526]' > $L/R6-seed-$seed.log 2>&1
  echo "[$(ts)] END R6-seed-$seed rc=$?" >> $L/queue.log
done
echo "[$(ts)] QUEUE-4 DONE" >> $L/queue.log

#!/bin/bash
# Laptop filler until 12:00 noon: AA replicate seeds 407.. ; stop with touch ~/outeropt-4060/STOP
cd ~/outeropt-4060/wt-gates || exit 1
export PYTHONPATH=src TRITON_CACHE_DIR=$HOME/outeropt-4060/triton-cache
PY=$HOME/outeropt-4060/venv/bin/python; L=$HOME/outeropt-4060/logs
for seed in $(seq 407 430); do
  [ -e $HOME/outeropt-4060/STOP ] && break
  [ "$(date +%H%M)" -ge 1118 ] && [ "$(date +%H%M)" -lt 1800 ] && break   # no new 43-min seed after 11:18
  E=$HOME/outeropt-4060/r6seeds/s$seed
  echo "[$(date +%T)] start seed $seed" >> $L/queue.log
  $PY scripts/repro/gpu_lock.py run --label R6-laptop-seed-$seed -- $PY scripts/train.py --model configs/model/D384/R6-AA.yaml --training configs/training/r6.yaml --exp-stage R6 --seed $seed --tag rep$seed --runs-root $E/runs --ckpt-root $E/ckpt --registry $E/registry.jsonl -o 'checkpoint.milestone_steps=[763,1526]' > $L/r6_seed_$seed.log 2>&1
  echo "[$(date +%T)] end seed $seed rc=$?" >> $L/queue.log
done
echo "[$(date +%T)] LAPTOP QUEUE3 DONE" >> $L/queue.log

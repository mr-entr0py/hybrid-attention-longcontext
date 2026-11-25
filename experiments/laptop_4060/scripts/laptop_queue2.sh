#!/bin/bash
# Laptop filler: descriptive R6 AA replicates (seeds 401..), loops until ~/outeropt-4060/STOP exists.
cd ~/outeropt-4060/wt-gates || exit 1
export PYTHONPATH=src TRITON_CACHE_DIR=$HOME/outeropt-4060/triton-cache
PY=$HOME/outeropt-4060/venv/bin/python; L=$HOME/outeropt-4060/logs
for seed in 401 402 403 404 405 406; do
  [ -e $HOME/outeropt-4060/STOP ] && break
  E=$HOME/outeropt-4060/r6seeds/s$seed
  echo "[$(date +%T)] start seed $seed" >> $L/queue.log
  $PY scripts/repro/gpu_lock.py run --label R6-laptop-seed-$seed -- $PY scripts/train.py --model configs/model/D384/R6-AA.yaml --training configs/training/r6.yaml --exp-stage R6 --seed $seed --tag rep$seed --runs-root $E/runs --ckpt-root $E/ckpt --registry $E/registry.jsonl -o 'checkpoint.milestone_steps=[763,1526]' > $L/r6_seed_$seed.log 2>&1
  echo "[$(date +%T)] end seed $seed rc=$?" >> $L/queue.log
done

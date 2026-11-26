#!/bin/bash
# Laptop daytime 2026-11-26, restarted after the 12:21 suspend hang (hard stop 18:00): descriptive R6 AA 6000-step extensions (non-gate).
# train/dev namespace-factor scoring of its milestones. Stop: touch ~/outeropt-4060/STOP
cd ~/outeropt-4060/wt-gates || exit 1
export PYTHONPATH=src TRITON_CACHE_DIR=$HOME/outeropt-4060/triton-cache
B=$HOME/outeropt-4060; PY=$B/venv/bin/python; L=$B/logs; Q=$L/queue.log
echo "[$(date +%T)] QUEUE8 start" >> $Q
for seed in 410 411 412; do
  [ -e $B/STOP ] && break
  left=$(( $(date -d "2026-11-26 18:00" +%s) - $(date +%s) )); [ $left -lt 900 ] && break
  E=$B/r6ext/s$seed
  echo "[$(date +%T)] start ext6000 seed $seed (hard stop 18:00)" >> $Q
  timeout --signal=INT --kill-after=60 $left $PY scripts/repro/gpu_lock.py run --label R6-laptop-ext-$seed -- $PY scripts/train.py --model configs/model/D384/R6-AA.yaml --training configs/training/r6.yaml --exp-stage R6 --seed $seed --tag ext6000 --runs-root $E/runs --ckpt-root $E/ckpt --registry $E/registry.jsonl -o schedule.steps=6000 -o 'checkpoint.milestone_steps=[763,1526,2289,3000,4000,5000]' > $L/r6_ext_$seed.log 2>&1
  echo "[$(date +%T)] end ext6000 seed $seed rc=$?" >> $Q
  $PY scripts/repro/gpu_lock.py run --label R6-laptop-ext-score-$seed -- $PY $B/r6_nsfactor_ext.py $seed > $L/r6_ext_score_$seed.log 2>&1
  echo "[$(date +%T)] scored ext seed $seed rc=$?" >> $Q
done
echo "[$(date +%T)] QUEUE8 DONE" >> $Q

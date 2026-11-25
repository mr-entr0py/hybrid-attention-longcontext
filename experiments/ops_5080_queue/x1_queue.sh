#!/bin/bash
# X1 exploratory MQAR composition sweep (orchestrator, 2026-11-25). Hard stop 23:59.
cd ~/Downloads/outeropt || exit 1
L=experiments/explore/X1_mqar_composition/logs; PY=.venv/bin/python; ts(){ date "+%F %T"; }
export TRITON_CACHE_DIR=$HOME/Downloads/outeropt/.cache/triton-diag
for s in 123 124 125; do for k in 0 3 1 2; do
  id=mix-k$k-s$s
  [ -e experiments/explore/X1_mqar_composition/runs/$id.json ] && continue
  [ -e ~/outeropt-queue/STOP_X1 ] && break 2
  left=$(( $(date -d "2026-11-25 23:59" +%s) - $(date +%s) )); [ $left -lt 600 ] && break 2
  echo "[$(ts)] START $id" >> $L/queue.log
  timeout --signal=INT --kill-after=60 $left $PY scripts/repro/gpu_lock.py run --label X1-$id --wait-exclusive -- \
     $PY scripts/s1_zoology/run_one_mix.py --run-id $id > $L/$id.log 2>&1
  echo "[$(ts)] END $id rc=$?" >> $L/queue.log
done; done
echo "[$(ts)] X1 DONE" >> $L/queue.log

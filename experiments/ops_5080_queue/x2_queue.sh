#!/bin/bash
# X2 exploratory MQAR placement + length sweep; starts after X1. Placement-first order (after the Xid on LAL-s123). Hard stop 2026-11-26 12:00.
cd ~/Downloads/outeropt || exit 1
L=experiments/explore/X2_mqar_placement_length/logs; PY=.venv/bin/python; ts(){ date "+%F %T"; }
export TRITON_CACHE_DIR=$HOME/Downloads/outeropt/.cache/triton-diag
for id in mix2-LAL-s123 mix2-ALL-s124 mix2-LAL-s124 mix2-LLA-s124 mix2-ALL-s125 mix2-LAL-s125 mix2-LLA-s125 mix2-LLL-s123 mix2-AAA-s124 mix2-AAA-s125 mix2-LLL-s124; do
  [ -e experiments/explore/X2_mqar_placement_length/runs/$id.json ] && continue
  [ -e ~/outeropt-queue/STOP_X2 ] && break
  left=$(( $(date -d "2026-11-26 12:00" +%s) - $(date +%s) )); [ $left -lt 300 ] && break
  for attempt in 1 2; do  # one automatic retry (resumes from the per-epoch checkpoint) after an infra crash
    left=$(( $(date -d "2026-11-26 12:00" +%s) - $(date +%s) )); [ $left -lt 300 ] && break
    echo "[$(ts)] START $id attempt $attempt" >> $L/queue.log
    timeout --signal=INT --kill-after=60 $left $PY scripts/repro/gpu_lock.py run --label X2-$id --wait-exclusive -- \
       $PY scripts/s1_zoology/run_one_mix2.py --run-id $id > $L/$id.log 2>&1
    echo "[$(ts)] END $id attempt $attempt rc=$?" >> $L/queue.log
    [ -e experiments/explore/X2_mqar_placement_length/runs/$id.json ] && break
  done
done
echo "[$(ts)] X2 DONE" >> $L/queue.log

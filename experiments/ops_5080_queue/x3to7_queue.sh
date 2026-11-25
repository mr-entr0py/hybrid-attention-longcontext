#!/bin/bash
# X3-X7 exploratory MQAR experiments; start after X2 ends. Hard stop 2026-11-26 12:00. Stop early: touch ~/outeropt-queue/STOP_X3
cd ~/Downloads/outeropt || exit 1
PY=.venv/bin/python; ts(){ date "+%F %T"; }
export TRITON_CACHE_DIR=$HOME/Downloads/outeropt/.cache/triton-diag
mkdir -p experiments/explore/logs; L=experiments/explore/logs
while ! grep -q "X2 DONE" experiments/explore/X2_mqar_placement_length/logs/queue.log 2>/dev/null; do sleep 15; done
RUNS=(
 # X3 width: does more recurrent state (d128) shrink the gap, and does 1 attention layer still suffice?
 X3__d128__LLL__s123 X3__d128__LAL__s123 X3__d128__AAA__s123 X3__d128__LLL__s124 X3__d128__LAL__s124 X3__d128__AAA__s124
 # X4 placement at k=2 of 3
 X4__AAL__s123 X4__ALA__s123 X4__LAA__s123 X4__AAL__s124 X4__ALA__s124 X4__LAA__s124
 # X5 deeper: 6 mixers, k in {0,1,2,3,6}, centred placement
 X5__LLLALL__s123 X5__LALLAL__s123 X5__LALALA__s123 X5__AAAAAA__s123 X5__LLLLLL__s123
 X5__LLLALL__s124 X5__LALLAL__s124 X5__LALALA__s124 X5__AAAAAA__s124 X5__LLLLLL__s124
 # X6 recurrent state size at k=0 (h1 = 2x state, h4 = 0.5x state vs the default h2)
 X6__LLL__h1__s123 X6__LLL__h4__s123 X6__LLL__h1__s124 X6__LLL__h4__s124
 # X7 training length: full budget without early stopping (hybrid) and 2x epochs (all-GDN)
 X7__LAL__noes__s123 X7__LLL__e64__s123
)
for id in "${RUNS[@]}"; do
  exp=${id%%__*}; mkdir -p experiments/explore/$exp/runs
  [ -e experiments/explore/$exp/runs/$id.json ] && continue
  [ -e ~/outeropt-queue/STOP_X3 ] && break
  left=$(( $(date -d "2026-11-26 12:00" +%s) - $(date +%s) )); [ $left -lt 300 ] && break
  for attempt in 1 2; do  # one automatic retry (resumes from the per-epoch checkpoint) after an infra crash
    left=$(( $(date -d "2026-11-26 12:00" +%s) - $(date +%s) )); [ $left -lt 300 ] && break
    echo "[$(ts)] START $id attempt $attempt" >> $L/queue.log
    timeout --signal=INT --kill-after=60 $left $PY scripts/repro/gpu_lock.py run --label $id --wait-exclusive -- \
       $PY scripts/s1_zoology/run_one_mixg.py --run-id $id > $L/$id.log 2>&1
    echo "[$(ts)] END $id attempt $attempt rc=$?" >> $L/queue.log
    [ -e experiments/explore/$exp/runs/$id.json ] && break
  done
done
echo "[$(ts)] X3-X7 DONE" >> $L/queue.log

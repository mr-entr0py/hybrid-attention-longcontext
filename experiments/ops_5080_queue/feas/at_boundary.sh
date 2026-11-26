#!/bin/bash
# Usage: at_boundary.sh <label> <cache: diag> -- <command...>  : wait for the FEAS job now running to complete, pause the
# runner, run the command under the GPU lock, restart the runner (it queues behind the command on the lock).
cd ~/Downloads/outeropt || exit 1
Q=~/outeropt-queue/feas; PY=.venv/bin/python; ts(){ date "+%F %T"; }
label=$1; shift 3
RID=$(ps -eo args | grep -oE "scripts/train.py --config .*" | head -1 >/dev/null; grep -E "\[runner .*launch " $Q/runner.log | tail -1 | sed -E 's/.*launch ([^ ]+) .*/\1/')
echo "[$(ts)] waiting for $RID to complete"
until grep -E "\"run_id\": *\"$RID\"" experiments/registry.jsonl | grep -qE "\"status\": *\"completed\""; do sleep 10; done
echo "[$(ts)] $RID completed; pausing FEAS runner"; $PY scripts/runner.py stop; sleep 5
env TRITON_CACHE_DIR=$PWD/.cache/triton-diag $PY scripts/repro/gpu_lock.py run --label $label --wait-exclusive -- "$@" > $Q/$label.log 2>&1 &
J=$!
for i in $(seq 1 120); do grep -q "acquired" $Q/$label.log 2>/dev/null && break; sleep 5; done
echo "[$(ts)] restarting FEAS runner"
TRITON_CACHE_DIR=$PWD/.cache/triton setsid nohup $PY scripts/runner.py --wait-exclusive run >> $Q/runner.log 2>&1 < /dev/null &
wait $J; echo "[$(ts)] $label done rc=$?"

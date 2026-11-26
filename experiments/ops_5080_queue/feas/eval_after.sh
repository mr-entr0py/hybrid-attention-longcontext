#!/bin/bash
# Usage: eval_after.sh <seed> <configs> <config-yaml-name-of-last-run>: wait until that run completes, pause the runner,
# score the seed under the GPU lock, restart the runner (it queues behind the scoring), then write the summary.
cd ~/Downloads/outeropt || exit 1
Q=~/outeropt-queue/feas; PY=.venv/bin/python; ts(){ date "+%F %T"; }
seed=$1; cfgs=$2; last=$3
RID=$($PY - "$last" <<'PY'
import json,sys,pathlib
for r in map(json.loads, open("experiments/registry.jsonl")):
    if r.get("status")=="queued" and pathlib.Path(str(r.get("config",""))).name==sys.argv[1]: rid=r["run_id"]
print(rid)
PY
)
echo "[$(ts)] waiting for $RID ($last)"
until grep -F "\"$RID\"" experiments/registry.jsonl | grep -qE "\"status\": *\"completed\""; do sleep 15; done
echo "[$(ts)] $RID completed; pausing runner to score seed $seed"; $PY scripts/runner.py stop; sleep 5
env TRITON_CACHE_DIR=$PWD/.cache/triton-diag $PY scripts/repro/gpu_lock.py run --label CORE-EVAL-s$seed --wait-exclusive -- $PY scripts/repro/core_eval.py --seed $seed --configs $cfgs > $Q/core_eval_s$seed.log 2>&1 &
J=$!
for i in $(seq 1 120); do grep -q "acquired" $Q/core_eval_s$seed.log 2>/dev/null && break; sleep 5; done
echo "[$(ts)] restarting runner"
TRITON_CACHE_DIR=$PWD/.cache/triton setsid nohup $PY scripts/runner.py --wait-exclusive run >> $Q/runner.log 2>&1 < /dev/null &
wait $J; echo "[$(ts)] scoring seed $seed rc=$?"
nice -n 10 env CUDA_VISIBLE_DEVICES= $PY scripts/repro/core_summary.py > $Q/core_summary_s$seed.log 2>&1; echo "[$(ts)] summary rc=$?"

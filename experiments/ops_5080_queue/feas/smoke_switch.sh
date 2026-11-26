#!/bin/bash
# Run the FEAS-SMOKE GPU phase right after the first FEAS job completes: pause the FEAS runner, give the smoke the lock,
# restart the runner (it queues behind the smoke on the lock).
cd ~/Downloads/outeropt || exit 1
Q=~/outeropt-queue/feas; PY=.venv/bin/python; ts(){ date "+%F %T"; }
FIRST=S2-D384-C0-s101-5a437fa2
until grep -E "\"run_id\": *\"$FIRST\"" experiments/registry.jsonl | grep -qE "\"status\": *\"completed\""; do sleep 15; done
echo "[$(ts)] $FIRST completed; pausing FEAS runner"
$PY scripts/runner.py stop; sleep 5
env TRITON_CACHE_DIR=$PWD/.cache/triton-diag $PY scripts/repro/gpu_lock.py run --label FEAS-SMOKE --wait-exclusive -- \
   $PY scripts/repro/feas_smoke.py --phase gpu > $Q/smoke_gpu.log 2>&1 &
SMOKE=$!
for i in $(seq 1 120); do grep -q "acquired" $Q/smoke_gpu.log 2>/dev/null && break; sleep 5; done
echo "[$(ts)] smoke holds the lock (or 10 min passed); restarting FEAS runner"
TRITON_CACHE_DIR=$PWD/.cache/triton setsid nohup $PY scripts/runner.py --wait-exclusive run >> $Q/runner.log 2>&1 < /dev/null &
wait $SMOKE; echo "[$(ts)] smoke gpu phase rc=$?"
$PY scripts/repro/feas_smoke.py --phase verdict >> $Q/smoke_gpu.log 2>&1; echo "[$(ts)] smoke verdict rc=$?"

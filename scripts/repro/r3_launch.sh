#!/usr/bin/env bash
# Detached launcher for one R3 test file:
#   bash scripts/repro/r3_launch.sh <tag> <min_free_MiB> <pytest args...>
# Waits (up to 2 h) until the GPU has <min_free_MiB> free, then runs pytest.
# Log: experiments/repro/R3/logs/<tag>_<timestamp>.log . Re-running is safe (raw outputs never overwrite: .rN).
set -u
cd "$(dirname "$0")/../.."
source .venv/bin/activate
tag=$1; need=$2; shift 2
ts=$(date +%Y%m%dT%H%M%S)
log=experiments/repro/R3/logs/${tag}_${ts}.log
setsid nohup bash -c "
echo CMD: python -m pytest -q -s -p no:cacheprovider $*
for i in \$(seq 1 1440); do
  free=\$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  [ \"\$free\" -ge $need ] && break
  sleep 5
done
echo START \$(date -Iseconds) gpu_free_MiB=\$free
python -m pytest -q -s -p no:cacheprovider $*
echo EXIT=\$?
date -Iseconds" > "$log" 2>&1 < /dev/null &
echo "$log"

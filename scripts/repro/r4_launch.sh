#!/usr/bin/env bash
# Detached, resumable R4 queue (plan §2.4; prereg experiments/repro/R4/prereg.yaml).
#   bash scripts/repro/r4_launch.sh [run_id ...]      # no args = all 8 runs; re-run to resume (done runs are skipped)
# Log: experiments/repro/R4/logs/queue.log (appended). Per-attempt logs and infra JSONs next to it.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"
source .venv/bin/activate
mkdir -p experiments/repro/R4/logs
setsid nohup python scripts/s1_zoology/run_queue_r4.py "$@" >> experiments/repro/R4/logs/queue.log 2>&1 < /dev/null &
echo "r4 queue pid $!  log: experiments/repro/R4/logs/queue.log"

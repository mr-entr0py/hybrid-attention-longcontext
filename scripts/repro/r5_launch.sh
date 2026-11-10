#!/usr/bin/env bash
# Detached, resumable R5 queue (plan §2.5; prereg experiments/repro/R5/prereg.yaml).
#   bash scripts/repro/r5_launch.sh [run_id ...]      # no args = the 6 runs (+ O5 iff rule 4 fires); re-run to resume
# Log: experiments/repro/R5/logs/queue.log (appended). Per-attempt logs and infra JSONs next to it.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"
source .venv/bin/activate
mkdir -p experiments/repro/R5/logs
R5_CONTINUE_ON_XID="${R5_CONTINUE_ON_XID:-0}" setsid nohup python scripts/s1_zoology/run_queue_r5.py "$@" >> experiments/repro/R5/logs/queue.log 2>&1 < /dev/null &
echo "r5 queue pid $!  log: experiments/repro/R5/logs/queue.log"

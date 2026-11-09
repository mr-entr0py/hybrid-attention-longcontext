#!/usr/bin/env bash
# Launch the S1(a) Zoology MQAR queue detached (protocol §17). Resumable: re-run to continue.
# Prerequisite (needs user permission, see reports/s1_reproduction.md):
#   source .venv/bin/activate && uv pip install -e third_party/zoology --no-deps --no-build-isolation
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
source "$REPO/.venv/bin/activate"
mkdir -p "$REPO/experiments/s1_zoology/logs" "$REPO/data/zoology"
cd "$REPO"
nohup python scripts/s1_zoology/run_queue.py >> experiments/s1_zoology/logs/queue.log 2>&1 &
echo "queue pid $!  log: experiments/s1_zoology/logs/queue.log"

#!/usr/bin/env bash
# Launch the S1(a) Zoology MQAR queue detached (protocol §17). Resumable: re-run to continue.
# Prerequisite (done by the orchestrator): Zoology @1ad20d1 installed editable in .venv
#   (uv pip install -e third_party/zoology --no-deps), plus pydantic, wandb, torchvision 0.26.0+cu128.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
source "$REPO/.venv/bin/activate"
mkdir -p "$REPO/experiments/s1_zoology/logs" "$REPO/data/zoology"
cd "$REPO"
nohup python scripts/s1_zoology/run_queue.py >> experiments/s1_zoology/logs/queue.log 2>&1 &
echo "queue pid $!  log: experiments/s1_zoology/logs/queue.log"

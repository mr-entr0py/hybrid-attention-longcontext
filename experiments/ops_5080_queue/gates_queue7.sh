#!/bin/bash
cd ~/Downloads/outeropt || exit 1
while ! grep -q "QUEUE-6 DONE" ~/outeropt-queue/gates/queue.log; do sleep 60; done
echo "[$(date "+%F %T")] START R6-SCORE-SEEDS" >> ~/outeropt-queue/gates/queue.log
PYTHONPATH=src .venv/bin/python scripts/repro/gpu_lock.py run --label R6-score-seeds --wait-exclusive -- .venv/bin/python ~/outeropt-queue/r6_score_seeds.py > ~/outeropt-queue/gates/R6-score-seeds.log 2>&1
echo "[$(date "+%F %T")] END R6-SCORE-SEEDS rc=$?" >> ~/outeropt-queue/gates/queue.log

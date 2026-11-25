#!/bin/bash
cd ~/outeropt-4060/wt-gates || exit 1
L=~/outeropt-4060/logs
while ! grep -q "end seed 417" $L/queue.log; do sleep 15; done
PYTHONPATH=src TRITON_CACHE_DIR=$HOME/outeropt-4060/triton-cache ~/outeropt-4060/venv/bin/python scripts/repro/gpu_lock.py run --label R6-laptop-score -- ~/outeropt-4060/venv/bin/python ~/outeropt-4060/r6_score_laptop.py > $L/r6_score_laptop.log 2>&1
echo "[$(date +%T)] laptop scoring rc=$?" >> $L/queue.log

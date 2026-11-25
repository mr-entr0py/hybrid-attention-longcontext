#!/bin/bash
cd ~/outeropt-4060/wt-gates || exit 1
export PYTHONPATH=src TRITON_CACHE_DIR=$HOME/outeropt-4060/triton-cache
PY=$HOME/outeropt-4060/venv/bin/python; L=$HOME/outeropt-4060/logs
until ! pgrep -f "scripts/repro/r6.py run" >/dev/null; do sleep 20; done
echo "[$(date +%T)] start vt23b rehearsal" >> $L/queue.log
$PY scripts/repro/gpu_lock.py run --label VT-23b-laptop -- $PY scripts/repro/vt23b.py > $L/vt23b_laptop.log 2>&1
echo "[$(date +%T)] vt23b rc=$?" >> $L/queue.log

#!/bin/bash
# Post-O4 gate queue (sequential, each step under the GPU lock; a failing step is logged and the queue continues).
cd ~/Downloads/outeropt || exit 1
export PYTHONUNBUFFERED=1
PY=$PWD/.venv/bin/python
L=~/outeropt-queue/gates; mkdir -p $L
ts() { date '+%F %T'; }
step() { name=$1; shift; echo "[$(ts)] START $name" >> $L/queue.log
  $PY scripts/repro/gpu_lock.py run --label "$name" --wait-exclusive -- "$@" > $L/$name.log 2>&1
  echo "[$(ts)] END $name rc=$?" >> $L/queue.log; }
step VT12-gpu-test $PY -m pytest -q tests/test_vt12_items.py::test_tier0_items_and_lm_scoring -x
CUDA_VISIBLE_DEVICES="" $PY scripts/repro/vt24r_check.py summary > $L/vt24r_summary.log 2>&1; echo "[$(ts)] vt24r summary rc=$?" >> $L/queue.log
step VT-13r $PY scripts/repro/vt13r.py
step VT-22b $PY scripts/repro/vt22b.py
step VT-23b $PY scripts/repro/vt23b.py
step R6 $PY scripts/repro/r6.py run
echo "[$(ts)] QUEUE-1 DONE" >> $L/queue.log

#!/bin/bash
# Waits for queue 1 (R6 incl. the '--' run), applies the VT-13r evaluator fix, re-evaluates VT-13r (CPU, existing artifacts), then SUITE-PF.
cd ~/Downloads/outeropt || exit 1
L=~/outeropt-queue/gates; PY=$PWD/.venv/bin/python
until grep -q "QUEUE-1 DONE" $L/queue.log; do sleep 30; done
ts() { date '+%F %T'; }
git apply ~/outeropt-queue/vt13r_fix.patch 2>$L/fixapply.log && git add scripts/repro/vt13r.py && git commit -q -m "VT-13r evaluator fixes (see patch message)" && echo "[$(ts)] vt13r fix committed" >> $L/queue.log || echo "[$(ts)] vt13r fix FAILED to apply" >> $L/queue.log
cp experiments/repro/VT-13r/vt13r_result.json experiments/repro/VT-13r/vt13r_result.first_run_evaluator_bug.json
CUDA_VISIBLE_DEVICES="" PYTHONPATH=src $PY scripts/repro/vt13r.py --evaluate-only > $L/VT-13r-reeval.log 2>&1; echo "[$(ts)] VT-13r re-eval rc=$?" >> $L/queue.log
echo "[$(ts)] START SUITE-PF" >> $L/queue.log
$PY scripts/repro/gpu_lock.py run --label SUITE-PF --wait-exclusive -- $PY -m pytest tests/ -q -rfEsxX > $L/SUITE-PF.log 2>&1
echo "[$(ts)] END SUITE-PF rc=$?" >> $L/queue.log
echo "[$(ts)] QUEUE-2 DONE" >> $L/queue.log

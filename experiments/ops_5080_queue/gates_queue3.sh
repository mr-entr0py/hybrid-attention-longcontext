#!/bin/bash
# Queue 3 (waits for queue 2 / SUITE-PF): R6 diagnostics on the 5080 (non-gate, descriptive), then an extended AA run.
cd ~/Downloads/outeropt || exit 1
L=~/outeropt-queue/gates; PY=$PWD/.venv/bin/python
until grep -q "QUEUE-2 DONE" $L/queue.log; do sleep 30; done
ts() { date '+%F %T'; }
C=experiments/repro/R6/attempt1/AA/ckpt/*/milestones
export PYTHONPATH=src
echo "[$(ts)] START R6-diag" >> $L/queue.log
$PY scripts/repro/gpu_lock.py run --label R6-diag --wait-exclusive -- $PY ~/outeropt-queue/r6_diag.py s763 $(ls $C/step_0000763*) s1526 $(ls $C/step_0001526*) s2289 $(ls $C/step_0002289*) > $L/R6-diag.log 2>&1
echo "[$(ts)] END R6-diag rc=$?" >> $L/queue.log
E=experiments/repro/R6/ext
echo "[$(ts)] START R6-ext-AA-6000" >> $L/queue.log
$PY scripts/repro/gpu_lock.py run --label R6-ext --wait-exclusive -- $PY scripts/train.py --model configs/model/D384/R6-AA.yaml --training configs/training/r6.yaml --exp-stage R6 --seed 300 --tag ext6000 --runs-root $E/runs --ckpt-root $E/ckpt --registry $E/registry.jsonl -o schedule.steps=6000 -o 'checkpoint.milestone_steps=[2289,3000,4000,5000]' > $L/R6-ext.log 2>&1
echo "[$(ts)] END R6-ext rc=$?" >> $L/queue.log
echo "[$(ts)] QUEUE-3 DONE" >> $L/queue.log

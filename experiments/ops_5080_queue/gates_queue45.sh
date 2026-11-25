#!/bin/bash
# Queue 4/5 (replaces gates_queue4): after the 6000-step extension: R6 train/dev diagnostic, then AA replicate seeds until 21:50, then SUITE-PF rerun.
cd ~/Downloads/outeropt || exit 1
L=~/outeropt-queue/gates; PY=$PWD/.venv/bin/python; ts() { date '+%F %T'; }
until grep -q "QUEUE-3 DONE" $L/queue.log; do sleep 15; done
git apply ~/outeropt-queue/testfix.patch 2>$L/testfix.log && git add tests && git commit -q -m "Tests: update 5 stale tests for merged code (see patch)" && echo "[$(ts)] testfix committed" >> $L/queue.log || echo "[$(ts)] testfix FAILED" >> $L/queue.log
export PYTHONPATH=src
C=experiments/repro/R6/attempt1/AA/ckpt/*/milestones
echo "[$(ts)] START R6-diag2" >> $L/queue.log
$PY scripts/repro/gpu_lock.py run --label R6-diag2 --wait-exclusive -- $PY ~/outeropt-queue/r6_diag.py s763 $(ls $C/step_0000763*) s1526 $(ls $C/step_0001526*) s2289 $(ls $C/step_0002289*) > $L/R6-diag.log 2>&1
echo "[$(ts)] END R6-diag2 rc=$?" >> $L/queue.log
for seed in $(seq 301 360); do
  [ -e ~/outeropt-queue/STOP ] && break
  [ $(date +%H%M) -ge 2150 ] && [ $(date +%H%M) -lt 2359 ] && break
  E=experiments/repro/R6/seeds/s$seed
  echo "[$(ts)] START R6-seed-$seed" >> $L/queue.log
  $PY scripts/repro/gpu_lock.py run --label R6-seed-$seed --wait-exclusive -- $PY scripts/train.py --model configs/model/D384/R6-AA.yaml --training configs/training/r6.yaml --exp-stage R6 --seed $seed --tag rep$seed --runs-root $E/runs --ckpt-root $E/ckpt --registry $E/registry.jsonl -o 'checkpoint.milestone_steps=[763,1526]' > $L/R6-seed-$seed.log 2>&1
  echo "[$(ts)] END R6-seed-$seed rc=$?" >> $L/queue.log
done
echo "[$(ts)] START SUITE-PF-2" >> $L/queue.log
$PY scripts/repro/gpu_lock.py run --label SUITE-PF-2 --wait-exclusive -- $PY -m pytest tests/ -q -rfEsxX > $L/SUITE-PF-2.log 2>&1
echo "[$(ts)] END SUITE-PF-2 rc=$?" >> $L/queue.log
echo "[$(ts)] QUEUE-5 DONE" >> $L/queue.log

#!/bin/bash
# Queue 9 (after TC-PF): R6 namespace-factor diagnostic, then descriptive R6 extension runs (non-gate) as GPU filler.
# Every job here uses the DIAGNOSTICS Triton cache (D069), never the fresh production cache.
# Stop the filler part: touch ~/outeropt-queue/STOP_FILLER (checked before each job).
cd ~/Downloads/outeropt || exit 1
L=~/outeropt-queue/gates; PY=.venv/bin/python; ts(){ date "+%F %T"; }
export TRITON_CACHE_DIR=$HOME/Downloads/outeropt/.cache/triton-diag
while ! grep -q "END TC-PF" $L/queue.log; do sleep 20; done
echo "[$(ts)] START R6-NSFACTOR" >> $L/queue.log
$PY scripts/repro/gpu_lock.py run --label R6-nsfactor --wait-exclusive -- $PY ~/outeropt-queue/r6_nsfactor.py > $L/R6-nsfactor.log 2>&1
echo "[$(ts)] END R6-NSFACTOR rc=$?" >> $L/queue.log
# descriptive extension runs: training-length curve for a 2nd AA seed and for the '--' (GDN) model
for spec in "AA 301" "GDN 501" "AA 302" "GDN 502"; do
  [ -e ~/outeropt-queue/STOP_FILLER ] && break
  set -- $spec; kind=$1; seed=$2; E=experiments/repro/R6/ext
  echo "[$(ts)] START R6-ext-$kind-$seed" >> $L/queue.log
  $PY scripts/repro/gpu_lock.py run --label R6-ext-$kind-$seed --wait-exclusive -- $PY scripts/train.py --model configs/model/D384/R6-$kind.yaml --training configs/training/r6.yaml --exp-stage R6 --seed $seed --tag ext6000 --runs-root $E/runs --ckpt-root $E/ckpt --registry $E/registry.jsonl -o schedule.steps=6000 -o 'checkpoint.milestone_steps=[2289,3000,4000,5000]' > $L/R6-ext-$kind-$seed.log 2>&1
  echo "[$(ts)] END R6-ext-$kind-$seed rc=$?" >> $L/queue.log
done
echo "[$(ts)] QUEUE-9 DONE" >> $L/queue.log

#!/bin/bash
cd ~/outeropt-4060/repo
export PYTHONPATH=src TRITON_CACHE_DIR=$HOME/outeropt-4060/triton-cache CUBLAS_WORKSPACE_CONFIG=:4096:8
R=$HOME/outeropt-4060/feas_smoke; mkdir -p $R
run() { # name model training steps seed extra...
  n=$1; m=$2; t=$3; s=$4; seed=$5; shift 5
  echo "=== $n $(date +%T)" >> $R/summary.log
  ../venv/bin/python scripts/repro/gpu_lock.py run --label rehearsal-$n -- ../venv/bin/python scripts/train.py \
    --model configs/model/D384/$m.yaml --training configs/training/$t.yaml --exp-stage S0 --seed $seed --lr 2e-3 \
    -o schedule.steps=$s -o runtime.mem_fraction=0.85 -o checkpoint.every=10 -o validation.every=0 \
    --runs-root $R/runs --ckpt-root $R/ckpt --registry $R/registry.jsonl --tag $n "$@" > $R/$n.log 2>&1
  echo "$n rc=$? $(grep -E 'done|REFUSED|Error|error' $R/$n.log | tail -1 | cut -c1-200)" >> $R/summary.log
}
run A_C0   C0  stage_A 30 101
run A_C12  C12 stage_A 30 101
run B_C0   C0  stage_B 10 102
run B_C12  C12 stage_B 10 102
run TEF_C3 C3  tef     20 103
echo "ALL DONE $(date +%T)" >> $R/summary.log

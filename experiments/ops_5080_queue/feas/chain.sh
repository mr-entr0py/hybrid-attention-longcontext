#!/bin/bash
# Unattended chain (D103-D105 core runs). Runs on the 5080 without the laptop:
#   wait for the armed seed-1 scoring -> score seed 2 (k 0,12,3) -> score k=1 seed by seed -> finish blinded FEAS scoring.
# Also a watchdog: if the training runner process is gone while runs are still pending, restart it (paused runs stay
# paused). Never writes the DTS10-commit note and never runs a decision. Stop: touch ~/outeropt-queue/STOP_CHAIN
cd ~/Downloads/outeropt || exit 1
Q=~/outeropt-queue/feas; PY=.venv/bin/python; ts(){ date "+%F %T"; }
runner_up(){ pgrep -f "^\.venv/bin/python scripts/runner.py --wait-exclusive run" >/dev/null; }
pending(){ $PY scripts/runner.py status 2>/dev/null | grep -E "S4" | grep -qE "^(queued|running|interrupted|infra_failed|crashed)"; }

watchdog(){
  while [ ! -e ~/outeropt-queue/STOP_CHAIN ]; do
    sleep 300
    if ! runner_up && pending; then
      sleep 90
      if ! runner_up && pending; then
        echo "[$(ts)] watchdog: no runner while runs are pending; restarting it"
        TRITON_CACHE_DIR=$PWD/.cache/triton setsid nohup $PY scripts/runner.py --wait-exclusive run >> $Q/runner.log 2>&1 < /dev/null &
      fi
    fi
  done
}
watchdog & WD=$!
trap 'kill $WD 2>/dev/null' EXIT

done_line(){ grep -q "summary rc=" "$1" 2>/dev/null; }
echo "[$(ts)] chain: waiting for seed-1 scoring"
until done_line $Q/eval_after_s1.log || [ $(date +%s) -ge $(date -d "2026-11-27 01:00" +%s) ]; do [ -e ~/outeropt-queue/STOP_CHAIN ] && exit 0; sleep 30; done
step(){ # step <tag> <seed> <configs> <last-yaml>
  [ -e ~/outeropt-queue/STOP_CHAIN ] && exit 0
  echo "[$(ts)] chain: $1 (waits for $4)"
  left=$(( $(date -d "2026-11-27 10:00" +%s) - $(date +%s) ))
  if [ $left -lt 60 ]; then echo "[$(ts)] chain: $1 skipped (past the 10:00 cut-off)"; return 0; fi
  # a run that never completes (failed or paused) must not block the later steps
  timeout --kill-after=30 $left $Q/eval_after.sh $2 $3 $4 > $Q/eval_after_$1.log 2>&1 || echo "[$(ts)] chain: $1 did not finish normally (rc=$?)"
  cp $Q/core_eval_s$2.log $Q/core_eval_$1.log 2>/dev/null
  $PY - "$1" <<'PY'
import pathlib, shutil, sys
d = pathlib.Path("experiments/core/snapshots") / sys.argv[1]; d.mkdir(parents=True, exist_ok=True)
for f in ("summary.md", "summary.json"):
    p = pathlib.Path("experiments/core") / f
    if p.exists(): shutil.copy2(p, d / f)
PY
  echo "[$(ts)] chain: $1 finished"
}
step s2      2 C0,C12,C3 S4_D384_C3_s2_B060_T.yaml
step k1_s0   0 C1        S4_D384_C1_s0_B060_T.yaml
step k1_s1   1 C1        S4_D384_C1_s1_B060_T.yaml
step k1_s2   2 C1        S4_D384_C1_s2_B060_T.yaml
# anything a step missed (for example a run that finished late) is picked up here; finished ones are skipped
for s in 0 1 2; do
  $PY scripts/repro/gpu_lock.py run --label CORE-EVAL-sweep-s$s --wait-exclusive -- $PY scripts/repro/core_eval.py --seed $s --configs C0,C12,C3,C1 > $Q/core_eval_sweep_s$s.log 2>&1
done
nice -n 10 env CUDA_VISIBLE_DEVICES= $PY scripts/repro/core_summary.py > $Q/core_summary_final.log 2>&1
echo "[$(ts)] chain: core scoring complete; resuming blinded FEAS scoring"
[ -e ~/outeropt-queue/STOP_CHAIN ] && exit 0
$PY scripts/repro/gpu_lock.py run --label FEAS-EVAL --wait-exclusive -- $PY scripts/repro/feas_eval.py >> $Q/feas_eval.log 2>&1
echo "[$(ts)] chain: FEAS scoring rc=$? ; CHAIN DONE"

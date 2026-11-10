#!/usr/bin/env bash
# R2(c) launcher: one process per cell, each under the single-queue lock (waiting until no other
# project process holds a CUDA context) and wrapped by the infra-event tagger.
# Resumable: a cell whose profile JSON has a non-null max_kernel_us_warm is skipped.
# Run detached:  setsid nohup bash scripts/repro/r2c_launch.sh > experiments/repro/r2/launch.log 2>&1 &
set -u
REPO=/home/sedrica-lynx/Downloads/outeropt
cd "$REPO"
source .venv/bin/activate
OUT=experiments/repro/r2/profile
mkdir -p "$OUT" experiments/repro/r2/infra
CELLS=${CELLS:-"D384:C12 D384:C0 D512:C12 D512:C0"}
for cell in $CELLS; do
  scale=${cell%%:*}; cfg=${cell##*:}
  js="$OUT/$scale-$cfg.json"
  if [ -f "$js" ] && python -c "import json,sys; sys.exit(0 if json.load(open('$js')).get('max_kernel_us_warm') else 1)"; then
    echo "[launch] $(date '+%F %T') skip $scale-$cfg (done)"; continue
  fi
  echo "[launch] $(date '+%F %T') start $scale-$cfg"
  python scripts/repro/infra_events.py run --tag "R2c-$scale-$cfg" --out-dir experiments/repro/r2/infra \
      --interval 2 --wait-exclusive --settle-s 90 --lock-timeout 14400 -- \
      python scripts/repro/r2_kernel_profile.py --scale "$scale" --config "$cfg" --out-dir "$OUT" \
      > "experiments/repro/r2/profile_$scale-$cfg.log" 2>&1
  rc=$?
  echo "[launch] $(date '+%F %T') end $scale-$cfg rc=$rc"
done
echo "[launch] $(date '+%F %T') all done"

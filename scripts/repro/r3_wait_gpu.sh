#!/usr/bin/env bash
# Wait until no VT-13 production train.py is alive and the GPU has >= $1 MiB free for 30 s straight.
need=$1; ok=0
while [ $ok -lt 6 ]; do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  if pgrep -f "^/home/sedrica-lynx/Downloads/outeropt/.venv/bin/python /home/sedrica-lynx/Downloads/outeropt/scripts/train.py --config /home/sedrica-lynx/Downloads/outeropt/experiments/s0/vt13_production" >/dev/null || pgrep -f "^python scripts/vt13_production_resume.py" >/dev/null || [ "$free" -lt "$need" ]; then ok=0; else ok=$((ok+1)); fi
  sleep 5
done
echo "gpu ready free=$free $(date -Iseconds)"

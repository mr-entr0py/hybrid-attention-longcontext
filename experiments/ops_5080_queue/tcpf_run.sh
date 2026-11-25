#!/bin/bash
# TC-PF (task T34), steps 1-3 of experiments/prereg/TRITON_prefeas.yaml v1, executed literally.
# Step 4 (registry note 'pre-FEAS-triton-cache', tc_pf.json, durable copy of the new archive) is done by hand
# only after W1 (warm exit 0, full coverage in the log) and W2 (gpumon: no foreign compute) are checked.
set -u
REPO=$HOME/Downloads/outeropt; cd "$REPO" || exit 1
L=$HOME/outeropt-queue/gates; PY=$REPO/.venv/bin/python; OUT=$REPO/experiments/repro/TC-PF
ts(){ date "+%F %T"; }
mkdir -p "$OUT"; exec >> "$OUT/tcpf_driver.log" 2>&1
echo "[$(ts)] TC-PF driver start; waiting for QUEUE-8 DONE"
while ! grep -q "QUEUE-8 DONE" "$L/queue.log"; do sleep 20; done
echo "[$(ts)] START TC-PF" >> "$L/queue.log"

# preflight (abort, never improvise)
abort(){ echo "[$(ts)] ABORT: $*"; echo "[$(ts)] END TC-PF rc=ABORT ($*)" >> "$L/queue.log"; exit 1; }
DIRTY=$(git status --porcelain -- src scripts configs tests); [ -z "$DIRTY" ] || abort "code tree dirty: $DIRTY"
COMMIT=$(git rev-parse HEAD); echo "commit $COMMIT"
PRE_SHA=$(sha256sum experiments/prereg/TRITON_prefeas.yaml | cut -d' ' -f1)
grep -q "$PRE_SHA" experiments/prereg/SHA256SUMS experiments/prereg/SHA256SUMS.v2 || abort "prereg hash $PRE_SHA not in SHA256SUMS"
echo "prereg sha256 $PRE_SHA verified"
[ -e .cache/triton-pre-restore-20261110 ] && abort ".cache/triton-pre-restore-20261110 already exists"
[ -e runs/_artifacts/triton_cache_prefeas.tar ] && abort "runs/_artifacts/triton_cache_prefeas.tar already exists"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
pgrep -af "scripts/train.py|warm_triton|evaluate.py" && abort "project CUDA process alive"

# step 1: move the production dir aside (never deleted), deterministic tar, SHA-256, durable copy
mv .cache/triton .cache/triton-pre-restore-20261110 || abort "mv failed"
"$PY" - <<'EOF' || abort "pre-restore tar failed"
import sys, shutil
from pathlib import Path
REPO = Path.home() / "Downloads" / "outeropt"; sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(0, str(REPO / "src"))
from warm_triton import deterministic_tar
src = REPO / ".cache" / "triton-pre-restore-20261110"; dst = REPO / "runs" / "_artifacts" / "triton_cache_pre_restore_20261110.tar"
h = deterministic_tar(src, dst)
n = sum(1 for p in src.rglob("*") if p.is_file())
print(f"pre-restore dir: {n} files; tar {dst} sha256 {h}")
d = Path.home() / "outeropt_artifacts"; d.mkdir(exist_ok=True)
shutil.copy2(dst, d / dst.name); shutil.copy2(str(dst) + ".sha256", d / (dst.name + ".sha256"))
print("durable copy in", d)
EOF

# step 2: empty production dir and its trainer-style hash
mkdir .cache/triton || abort "mkdir failed"
"$PY" -c "import sys; sys.path.insert(0,'src'); from hybridlc.training.compat import triton_cache_hash; print('empty-dir trainer-style hash', triton_cache_hash('.cache/triton'))"

# step 3: the registered warm command, under the lock, into the empty production dir
echo "[$(ts)] warm start"
TRITON_CACHE_DIR="$REPO/.cache/triton" "$PY" scripts/repro/gpu_lock.py run --label TC-PF --wait-exclusive -- \
  "$PY" scripts/warm_triton.py --scales D384 D512 --archive runs/_artifacts/triton_cache_prefeas.tar > "$OUT/warm.log" 2>&1
rc=$?
echo "[$(ts)] warm rc=$rc"
echo "[$(ts)] END TC-PF rc=$rc (step 4 pending manual W1/W2 check)" >> "$L/queue.log"

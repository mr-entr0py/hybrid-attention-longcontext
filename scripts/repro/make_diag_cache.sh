#!/usr/bin/env bash
# Create the diagnostics Triton cache $REPO/.cache/triton-diag from the S0 archive (PI ruling D069 / PR-12).
#
#   bash scripts/repro/make_diag_cache.sh [--dest DIR] [--archive runs/_artifacts/triton_cache.tar]
#
# Every reproduction, diagnostic, verification and test job uses this cache (tests/conftest.py sets
# TRITON_CACHE_DIR to it); production study runs never do (the trainer refuses it for exp_stage S2..S8).
# The archive's SHA-256 must equal the recorded S0 archive hash before anything is extracted. An existing
# non-empty destination is never overwritten or deleted: the script refuses (move it aside yourself).
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ARCHIVE="$REPO/runs/_artifacts/triton_cache.tar"
DEST="$REPO/.cache/triton-diag"
EXPECTED="f417e487a0341891873b8fd80f5f4475a01c8564c884da12bd0418966896e3de"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dest) DEST="$2"; shift 2 ;;
    --archive) ARCHIVE="$2"; shift 2 ;;
    --expected) EXPECTED="$2"; shift 2 ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done
GOT="$(sha256sum "$ARCHIVE" | cut -d' ' -f1)"
if [[ "$GOT" != "$EXPECTED" ]]; then
  echo "REFUSED: $ARCHIVE sha256 $GOT != expected $EXPECTED" >&2
  exit 1
fi
if [[ -d "$DEST" ]] && [[ -n "$(ls -A "$DEST" 2>/dev/null)" ]]; then
  echo "REFUSED: $DEST exists and is not empty (never overwritten; move it aside first)" >&2
  exit 1
fi
mkdir -p "$DEST"
tar -xf "$ARCHIVE" -C "$DEST"
N="$(find "$DEST" -type f | wc -l)"
HASH="$(cd "$REPO" && "$REPO/.venv/bin/python" -c "import sys; sys.path.insert(0, 'src'); from hybridlc.training.compat import triton_cache_hash; print(triton_cache_hash(sys.argv[1]))" "$DEST")"
printf '{"dest": "%s", "archive": "%s", "archive_sha256": "%s", "n_files": %s, "trainer_style_hash": "%s", "ts": "%s"}\n' \
  "$DEST" "$ARCHIVE" "$GOT" "$N" "$HASH" "$(date -Iseconds)" | tee "$DEST.extract.json"

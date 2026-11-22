"""O4-Z / O4-B wrapper around scripts/s1_zoology/run_one_r5.py (prereg O4_soak.v2.yaml).

Runs the R5 cell with its outputs (result JSON and weights) redirected to $O4_DIR, so the R5 results are never
overwritten, and arms faulthandler.dump_traceback_later(20, repeat=True) into $O4_FAULT_LOG so a hang leaves a Python
stack. Takes the GPU lock token check first (R2-LIVE v2 L1), like every project entry point.
"""
from __future__ import annotations

import faulthandler
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from hybridlc.utils.gpu_lock import guard  # noqa: E402

guard(Path(__file__).name)
sys.path.insert(0, str(Path(__file__).resolve().parent))

if os.environ.get("O4_FAULT_LOG"):
    _fl = open(os.environ["O4_FAULT_LOG"], "a", buffering=1)
    faulthandler.dump_traceback_later(20, repeat=True, file=_fl)

import run_one_r5 as m  # noqa: E402

m.R5_DIR = Path(os.environ["O4_DIR"])  # results + weights go to O4, never to experiments/repro/R5
m.main()

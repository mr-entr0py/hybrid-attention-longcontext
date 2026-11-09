#!/usr/bin/env python
"""Sample nvidia-smi every 30 s into a JSONL file and keep a running contention summary.

    python scripts/gpu_monitor.py --out runs/<run_id>/gpu.jsonl --pid <trainer pid>
    python scripts/gpu_monitor.py --out /tmp/gpu.jsonl --duration 120 --interval 10
    python scripts/gpu_monitor.py --summarize runs/<run_id>/gpu.jsonl

Definitions of ``contended`` / ``throttled`` are in ``hybridlc.training.gpumon``. The runner starts one
monitor per job (``runs/<run_id>/gpu.jsonl`` and ``gpu_summary.json``).
"""
from __future__ import annotations

import argparse
import json
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hybridlc.training import gpumon  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out")
    ap.add_argument("--pid", type=int, default=None, help="stop when this process exits; own = it + children")
    ap.add_argument("--interval", type=float, default=30.0)
    ap.add_argument("--duration", type=float, default=None)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=gpumon.CONTENDED_SM_THRESHOLD,
                    help="foreign SM %% at or above which a sample counts as contended")
    ap.add_argument("--summarize", default=None, help="print the summary of an existing gpu.jsonl")
    a = ap.parse_args(argv)
    if a.summarize:
        print(json.dumps(gpumon.summarize_file(a.summarize), indent=1))
        return 0
    if not a.out:
        ap.error("--out is required")
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    s = gpumon.monitor(a.out, pid=a.pid, interval=a.interval, duration=a.duration, index=a.gpu,
                       threshold=a.threshold)
    print(json.dumps(s))
    return 0


if __name__ == "__main__":
    sys.exit(main())

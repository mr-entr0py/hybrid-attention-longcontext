#!/usr/bin/env python
"""P14 (protocol §12): F_gc vs F_g at k = 12 (and k in {1, 3} if triggered).

Reads only experiments/results/** and experiments/registry.jsonl; writes plots/out/P14.{pdf,png,inputs.json}.
Usage: python plots/make_p14.py [--results DIR] [--registry FILE] [--out DIR] [--B 10000] [--scale D384]
"""
from __future__ import annotations

from hybridlc.analysis.figures import cli_main

if __name__ == "__main__":
    cli_main("P14")

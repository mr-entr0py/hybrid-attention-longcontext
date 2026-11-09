#!/usr/bin/env python
"""P13 (protocol §12): 2x vs 1x Stage B at k in {0, 3, 12}.

Reads only experiments/results/** and experiments/registry.jsonl; writes plots/out/P13.{pdf,png,inputs.json}.
Usage: python plots/make_p13.py [--results DIR] [--registry FILE] [--out DIR] [--B 10000] [--scale D384]
"""
from __future__ import annotations

from hybridlc.analysis.figures import cli_main

if __name__ == "__main__":
    cli_main("P13")

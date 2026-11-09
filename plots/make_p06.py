#!/usr/bin/env python
"""P6 (protocol §12): Early / middle / late at k*, per-seed lines, c1 and c2 (H4).

Reads only experiments/results/** and experiments/registry.jsonl; writes plots/out/P6.{pdf,png,inputs.json}.
Usage: python plots/make_p06.py [--results DIR] [--registry FILE] [--out DIR] [--B 10000] [--scale D384]
"""
from __future__ import annotations

from hybridlc.analysis.figures import cli_main

if __name__ == "__main__":
    cli_main("P6")

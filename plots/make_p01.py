#!/usr/bin/env python
"""P1 (protocol §12): S_ret(k) at the primary cell, all seeds, R = 0.8 line and isotonic fit (H1, H2).

Reads only experiments/results/** and experiments/registry.jsonl; writes plots/out/P1.{pdf,png,inputs.json}.
Usage: python plots/make_p01.py [--results DIR] [--registry FILE] [--out DIR] [--B 10000] [--scale D384]
"""
from __future__ import annotations

from hybridlc.analysis.figures import cli_main

if __name__ == "__main__":
    cli_main("P1")

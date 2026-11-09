#!/usr/bin/env python
"""P3 (protocol §12): R_f(k) for the three families on one probability-scale axis; per-seed D3 (H3).

Reads only experiments/results/** and experiments/registry.jsonl; writes plots/out/P3.{pdf,png,inputs.json}.
Usage: python plots/make_p03.py [--results DIR] [--registry FILE] [--out DIR] [--B 10000] [--scale D384]
"""
from __future__ import annotations

from hybridlc.analysis.figures import cli_main

if __name__ == "__main__":
    cli_main("P3")

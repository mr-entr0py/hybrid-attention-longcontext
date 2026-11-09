#!/usr/bin/env python
"""P18 (protocol §12): Strip plot of per-seed primary-cell scores (bimodality, U-05).

Reads only experiments/results/** and experiments/registry.jsonl; writes plots/out/P18.{pdf,png,inputs.json}.
Usage: python plots/make_p18.py [--results DIR] [--registry FILE] [--out DIR] [--B 10000] [--scale D384]
"""
from __future__ import annotations

from hybridlc.analysis.figures import cli_main

if __name__ == "__main__":
    cli_main("P18")

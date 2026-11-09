#!/usr/bin/env python
"""P10 (protocol §12): Per-level R (H1-robust) and N50 vs k at 16K and 32K, levels as N/N50_0.

Reads only experiments/results/** and experiments/registry.jsonl; writes plots/out/P10.{pdf,png,inputs.json}.
Usage: python plots/make_p10.py [--results DIR] [--registry FILE] [--out DIR] [--B 10000] [--scale D384]
"""
from __future__ import annotations

from hybridlc.analysis.figures import cli_main

if __name__ == "__main__":
    cli_main("P10")

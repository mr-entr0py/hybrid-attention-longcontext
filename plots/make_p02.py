#!/usr/bin/env python
"""P2 (protocol §12): Forest plot of per-seed L1 and L3, paired CIs, sensitivity CIs and success counts (H1, H1-min).

Reads only experiments/results/** and experiments/registry.jsonl; writes plots/out/P2.{pdf,png,inputs.json}.
Usage: python plots/make_p02.py [--results DIR] [--registry FILE] [--out DIR] [--B 10000] [--scale D384]
"""
from __future__ import annotations

from hybridlc.analysis.figures import cli_main

if __name__ == "__main__":
    cli_main("P2")

#!/usr/bin/env python
"""P5 (protocol §12): Retrieval vs multi-hop length slopes (H5b).

Reads only experiments/results/** and experiments/registry.jsonl; writes plots/out/P5.{pdf,png,inputs.json}.
Usage: python plots/make_p05.py [--results DIR] [--registry FILE] [--out DIR] [--B 10000] [--scale D384]
"""
from __future__ import annotations

from hybridlc.analysis.figures import cli_main

if __name__ == "__main__":
    cli_main("P5")

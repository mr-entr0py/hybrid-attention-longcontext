#!/usr/bin/env python
"""P4 (protocol §12): Delta0(T) vs log2 T on the fixed level set, per-seed slopes, G-REF status per T (H5a).

Reads only experiments/results/** and experiments/registry.jsonl; writes plots/out/P4.{pdf,png,inputs.json}.
Usage: python plots/make_p04.py [--results DIR] [--registry FILE] [--out DIR] [--B 10000] [--scale D384]
"""
from __future__ import annotations

from hybridlc.analysis.figures import cli_main

if __name__ == "__main__":
    cli_main("P4")

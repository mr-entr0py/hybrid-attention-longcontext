#!/usr/bin/env python
"""P9 (protocol §12): B_max(T) within 8 GiB per k.

Reads only experiments/results/** and experiments/registry.jsonl; writes plots/out/P9.{pdf,png,inputs.json}.
Usage: python plots/make_p09.py [--results DIR] [--registry FILE] [--out DIR] [--B 10000] [--scale D384]
"""
from __future__ import annotations

from hybridlc.analysis.figures import cli_main

if __name__ == "__main__":
    cli_main("P9")

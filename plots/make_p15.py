#!/usr/bin/env python
"""P15 (protocol §12): NoPE vs RoPE at k in {3, 12}.

Reads only experiments/results/** and experiments/registry.jsonl; writes plots/out/P15.{pdf,png,inputs.json}.
Usage: python plots/make_p15.py [--results DIR] [--registry FILE] [--out DIR] [--B 10000] [--scale D384]
"""
from __future__ import annotations

from hybridlc.analysis.figures import cli_main

if __name__ == "__main__":
    cli_main("P15")

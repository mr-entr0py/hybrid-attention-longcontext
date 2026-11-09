#!/usr/bin/env python
"""P12 (protocol §12): TEF dose ladder (250 -> 4,000) per anchor; gap vs steps.

Reads only experiments/results/** and experiments/registry.jsonl; writes plots/out/P12.{pdf,png,inputs.json}.
Usage: python plots/make_p12.py [--results DIR] [--registry FILE] [--out DIR] [--B 10000] [--scale D384]
"""
from __future__ import annotations

from hybridlc.analysis.figures import cli_main

if __name__ == "__main__":
    cli_main("P12")

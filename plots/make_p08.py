#!/usr/bin/env python
"""P8 (protocol §12): Measured vs analytic prefill and decode latency vs T per k (cold/hot L2), roofline fractions.

Reads only experiments/results/** and experiments/registry.jsonl; writes plots/out/P8.{pdf,png,inputs.json}.
Usage: python plots/make_p08.py [--results DIR] [--registry FILE] [--out DIR] [--B 10000] [--scale D384]
"""
from __future__ import annotations

from hybridlc.analysis.figures import cli_main

if __name__ == "__main__":
    cli_main("P8")

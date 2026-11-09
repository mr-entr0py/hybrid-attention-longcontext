#!/usr/bin/env python
"""P7 (protocol §12): Primary-cell score vs analytic state bytes at 32K (log x), router line, Pareto front, GDN-E (H6, H6′).

Reads only experiments/results/** and experiments/registry.jsonl; writes plots/out/P7.{pdf,png,inputs.json}.
Usage: python plots/make_p07.py [--results DIR] [--registry FILE] [--out DIR] [--B 10000] [--scale D384]
"""
from __future__ import annotations

from hybridlc.analysis.figures import cli_main

if __name__ == "__main__":
    cli_main("P7")

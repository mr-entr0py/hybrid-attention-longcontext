#!/usr/bin/env python
"""P11 (protocol §12): G-REF panel: S12(T)/S12(C0) per T and seed; reference inversion.

Reads only experiments/results/** and experiments/registry.jsonl; writes plots/out/P11.{pdf,png,inputs.json}.
Usage: python plots/make_p11.py [--results DIR] [--registry FILE] [--out DIR] [--B 10000] [--scale D384]
"""
from __future__ import annotations

from hybridlc.analysis.figures import cli_main

if __name__ == "__main__":
    cli_main("P11")

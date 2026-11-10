"""Agent-1 design audit (read-only): local-window heuristic floors of the frozen A-freq CS items.

For each frozen test/CS A-freq cell, the share of queries that a counter restricted to the last
W context tokens answers correctly (ties = 1/2), for W in {256, 1024, 4096, 8192}, plus the
recency floor. Uses the project's own scorer helper for W <= 4096 and the same rule for 8192.
CPU only; writes nothing in the repo except this script's stdout.
"""
from __future__ import annotations

import glob
import os

import numpy as np

from hybridlc.eval.items import load_items
from hybridlc.eval.scorer import _afreq_heuristics

ROOT = "/home/sedrica-lynx/Downloads/outeropt/experiments/items/test/CS"


def win_floor(item, q, W):
    ctx = np.asarray(item["context_ids"])
    x = int(q["answer_ids"][0])
    others = [int(c[0]) for c in q["candidates"] if int(c[0]) != x]
    if len(others) != 1:
        return np.nan
    px = np.flatnonzero(ctx == x); py = np.flatnonzero(ctx == others[0])
    n = len(ctx)
    cx, cy = int((px >= n - W).sum()), int((py >= n - W).sum())
    return 1.0 if cx > cy else (0.0 if cx < cy else 0.5)


def main():
    print("cell | n_queries | recency | last256 | last1024 | last4096 | last8192 | full-count")
    for p in sorted(glob.glob(os.path.join(ROOT, "A-freq_rho*_*.npz"))):
        items, meta = load_items(p)
        rows = []
        for it in items:
            for q in it["queries"]:
                h = _afreq_heuristics(it, q, "A-freq")
                rows.append((h["recency_heuristic_correct"], h["window256_count_heuristic"],
                             h["window1024_count_heuristic"], h["window4096_count_heuristic"],
                             win_floor(it, q, 8192), win_floor(it, q, 10 ** 9)))
        a = np.array(rows, dtype=float)
        m = np.nanmean(a, axis=0)
        print(f"{os.path.basename(p)[:-4]} | {len(a)} | " + " | ".join(f"{v:.3f}" for v in m))


if __name__ == "__main__":
    main()

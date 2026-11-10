"""Read-only: window-counter floors of the frozen CS A-freq cells (what a short-memory counter scores)."""
import sys, numpy as np
sys.path.insert(0, "/home/sedrica-lynx/Downloads/outeropt/src")
from hybridlc.eval.items import load_items
root = "/home/sedrica-lynx/Downloads/outeropt/experiments/items/test/CS/"
Ws = (256, 1024, 2048, 4096, 8192)
for T in ("8K", "16K", "32K"):
    for rho in ("4", "2", "1.5", "1.25"):
        items, _ = load_items(root + f"A-freq_rho{rho}_{T}.npz")
        acc = {W: [] for W in Ws}; rec = []
        for it in items:
            ctx = np.asarray(it["context_ids"]); n = len(ctx)
            for q in it["queries"]:
                x = int(q["answer_ids"][0]); others = [int(c[0]) for c in q["candidates"] if int(c[0]) != x]
                if len(others) != 1: continue
                px = np.flatnonzero(ctx == x); py = np.flatnonzero(ctx == others[0])
                if not len(px) or not len(py): continue
                rec.append(px.max() > py.max())
                for W in Ws:
                    cx, cy = int((px >= n - W).sum()), int((py >= n - W).sum())
                    acc[W].append(1.0 if cx > cy else (0.0 if cx < cy else 0.5))
        print(f"A-freq rho={rho:>4} T={T:>3}: recency {np.mean(rec):.3f} | " +
              " | ".join(f"last-{W} {np.mean(acc[W]):.3f}" for W in Ws))

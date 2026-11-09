#!/usr/bin/env python
"""Score a training checkpoint on frozen evaluation suites (protocol §9.2, §9.8).

    python scripts/evaluate.py --ckpt checkpoints/<run_id>/milestones/step_0001000_final_bf16.pt \
        --suite CS [--label tef1000] [--n-contexts 10] [--results-root experiments/results]

Writes ``<results-root>/<run_id>/<label>/<suite>/<task>_<level>_<T>.jsonl`` + ``_meta.json`` per item
cell (never overwriting: re-runs get a ``.rN`` suffix) and appends one line per cell to
``<results-root>/_eval_log.jsonl``. ``--n-contexts`` scores only the first N contexts of each cell
(smoke tests / quick dev checks; recorded as ``subset`` in the header).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
os.environ.setdefault("TRITON_CACHE_DIR", str(REPO / ".cache" / "triton"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="final.pt / step_*.pt / milestones/*_bf16.pt")
    ap.add_argument("--suite", nargs="+", required=True, choices=["CS", "MS", "DG", "DC", "FD", "Tier0"])
    ap.add_argument("--label", default=None, help="checkpoint label (default tef<step> / B<step> / A<step>)")
    ap.add_argument("--items-root", default=str(REPO / "experiments" / "items"))
    ap.add_argument("--results-root", default=str(REPO / "experiments" / "results"))
    ap.add_argument("--runs-root", default=str(REPO / "runs"), help="to find config.resolved.yaml for milestones")
    ap.add_argument("--n-contexts", type=int, default=None)
    ap.add_argument("--only", default=None, help="only item files whose name contains this substring")
    ap.add_argument("--no-candidates", action="store_true", help="skip candidate (rank) scoring")
    ap.add_argument("--rescore", action="store_true", help="also run the 5%% single-pass re-score check")
    ap.add_argument("--mem-fraction", type=float, default=0.44)
    args = ap.parse_args()

    import torch

    from hybridlc.integration.evaluate import evaluate_checkpoint, load_checkpoint
    from hybridlc.utils.gpu import claim

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        claim(args.mem_fraction)
    ck = load_checkpoint(args.ckpt, dev, runs_root=args.runs_root)
    print(f"[eval] loaded {ck.run_id} step {ck.step} ({ck.dtype}) from {ck.path}", flush=True)
    recs = evaluate_checkpoint(ck, args.suite, label=args.label, items_root=args.items_root,
                               results_root=args.results_root, n_contexts=args.n_contexts, only=args.only,
                               score_candidates=not args.no_candidates, rescore=args.rescore)
    peak = torch.cuda.max_memory_allocated() if dev == "cuda" else 0
    print(json.dumps(dict(n_cells=len(recs), peak_mem_gib=peak / 2**30,
                          wall_s=sum(r["wall_s"] for r in recs))), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

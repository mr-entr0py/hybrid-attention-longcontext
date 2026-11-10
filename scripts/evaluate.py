#!/usr/bin/env python
"""Score a training checkpoint on frozen evaluation suites (protocol §9.2, §9.8).

    python scripts/evaluate.py --ckpt checkpoints/<run_id>/milestones/step_0001000_final_bf16.pt \
        --suite CS [--label tef1000] [--n-contexts 10] [--results-root experiments/results]

Writes ``<results-root>/<run_id>/<label>/<suite>/<task>_<level>_<T>.jsonl`` + ``_meta.json`` per item
cell (never overwriting: re-runs get a ``.rN`` suffix) and appends one line per cell to
``<results-root>/_eval_log.jsonl``. ``--n-contexts N`` scores only the first N contexts of each cell
(smoke tests / quick dev checks; recorded as ``subset`` in the header) and writes under the label
``<label>-nN``, which the analysis never reads.

FEAS-PREC v2 passes (PI ruling D083; experiments/prereg/FEAS_precision.v2.yaml):
  --pass P0          production default (bf16-rounded weights, bf16 autocast, branched cache, candidates)
  --pass P1          fp32 weights, bf16 autocast, no candidates            -> label <label>-P1
  --pass P4 | --reference fp32-cache
                     fp32 weights, autocast off, branched cache through R-ATT, no candidates -> <label>-P4
  --rescore-ref bf16|fp32|both
                     P2 (frozen §9.2 check, bf16; default) and/or P3 (fp32 single-pass, fp32 weights,
                     R-ATT at T >= 16K) on the 5% subset of a P0 run
  --only A,B --only C  cell-name substrings (any match)
Every result ``_meta.json`` records pass id, weights dtype, autocast flag, TF32 flags and attention backend.

R2-LIVE v2: takes the project GPU lock (or the holder's inherited token) before any CUDA call (exit 78
otherwise) and records the infra block in every ``_eval_log.jsonl`` line and result meta.
DTS-10 technical blinding: for an exp_stage S2 checkpoint, while no ``DTS10-commit`` note exists in
``--registry`` (or the study registry), no score / EM / logp / rank / flip value is printed or written
to the ``_eval_log.jsonl`` summary (cell names, counts, paths and timings only).
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
    ap.add_argument("--only", action="append", default=None,
                    help="only item files whose name contains one of these substrings (repeatable; comma lists)")
    ap.add_argument("--no-candidates", action="store_true", help="skip candidate (rank) scoring")
    rs = ap.add_mutually_exclusive_group()
    rs.add_argument("--rescore", dest="rescore", action="store_true", default=None,
                    help="run the 5%% single-pass re-score check on every suite (default: CS/DG/DC/FD/MS; never Tier0)")
    rs.add_argument("--no-rescore", dest="rescore", action="store_false",
                    help="skip the re-score check (the analysis then flags these cells)")
    ap.add_argument("--fp32-weights", action="store_true",
                    help="score a full checkpoint's fp32 weights (non-canonical; default rounds to bf16 like milestones)")
    ap.add_argument("--mem-fraction", type=float, default=0.44)
    ap.add_argument("--pass", dest="pass_id", choices=["P0", "P1", "P4"], default="P0")
    ap.add_argument("--reference", choices=["fp32-cache"], default=None, help="= --pass P4")
    ap.add_argument("--rescore-ref", choices=["bf16", "fp32", "both"], default="bf16")
    ap.add_argument("--registry", action="append", default=None,
                    help="registry checked for the DTS10-commit note (default: experiments/registry.jsonl)")
    args = ap.parse_args()
    if args.reference == "fp32-cache":
        args.pass_id = "P4"

    # R2-LIVE v2 L1: lock (or inherited token) before the first CUDA call; loading the model imports fla
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        from hybridlc.utils.gpu_lock import guard

        guard("evaluate")
    import time

    import torch

    from hybridlc.integration.blinding import blinded
    from hybridlc.integration.evaluate import PASSES, evaluate_checkpoint, load_checkpoint
    from hybridlc.utils.gpu import claim
    from hybridlc.utils.infra import InfraTagger

    t_start = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        claim(args.mem_fraction)
    tagger = InfraTagger(tag="evaluate", interval=5.0, sample=dev == "cuda").start()
    p = PASSES[args.pass_id]
    fp32_weights = args.fp32_weights or p["weights"] == "float32"
    ck = load_checkpoint(args.ckpt, dev, runs_root=args.runs_root, canonical_bf16=not fp32_weights)
    exp_stage = (ck.config or {}).get("exp_stage") or ck.run_id
    blind = blinded(exp_stage, args.registry)
    print(f"[eval] loaded {ck.run_id} step {ck.step} ({ck.dtype}; scored as {ck.eval_weights}) from {ck.path}; "
          f"pass {args.pass_id}{'; BLINDED (exp_stage S2, no DTS10-commit note)' if blind else ''}", flush=True)
    ref_model = None
    if args.pass_id == "P0" and args.rescore_ref in ("fp32", "both"):
        ref_model = load_checkpoint(args.ckpt, dev, runs_root=args.runs_root, canonical_bf16=False).model
    rescore = args.rescore if args.pass_id == "P0" else False
    recs = evaluate_checkpoint(ck, args.suite, label=args.label, items_root=args.items_root,
                               results_root=args.results_root, n_contexts=args.n_contexts, only=args.only,
                               score_candidates=(not args.no_candidates) and p["candidates"], rescore=rescore,
                               pass_id=args.pass_id, autocast=None if p["autocast"] else False,
                               reference_attention=p["reference_attention"], rescore_ref=args.rescore_ref,
                               ref_model=ref_model, blind=blind, infra=tagger)
    tagger.finish()
    peak = torch.cuda.max_memory_allocated() if dev == "cuda" else 0
    print(json.dumps(dict(n_cells=len(recs), pass_id=args.pass_id, blinded=blind, peak_mem_gib=peak / 2**30,
                          wall_s=round(time.time() - t_start, 2),
                          cell_wall_s=round(sum(r["wall_s"] for r in recs), 2))), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

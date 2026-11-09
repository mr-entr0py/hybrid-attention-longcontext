#!/usr/bin/env python
"""Run the pre-registered analysis (protocol §11) on item-level results and write results + verdicts.

Reads ``experiments/results/**`` (item-level JSONL, protocol §9.8) and ``experiments/registry.jsonl``;
writes ``<out>/results.json`` (every estimate, test and sensitivity analysis), ``<out>/verdicts.json``
(pre-registered verdicts, labels and wording) and ``<out>/inputs.json`` (input file SHA-256s).

Examples::

    python scripts/analyse_results.py --gates configs/analysis/gates.json --out experiments/analysis/tierA
    python scripts/analyse_results.py --synthetic --out /tmp/vt18_demo      # planted-effect demo world

``--gates`` is a JSON object {"G0": true, "G-RP": true, "FEAS-1": true, "FEAS-2": true, "G-FULL": true}
copied from reports/decisions.md; a gate that is missing counts as not passed, so "falsified" can never
be declared without it. ``--branch-map`` is a JSON {run_id: branch} for runs whose registry rows lack
a ``branch`` field (main / tef4000 / stageB2x).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

from hybridlc.analysis import verdicts as V
from hybridlc.analysis.pipeline import AnalysisSpec, dumps, run_analysis
from hybridlc.analysis.protocol import BOOTSTRAP_B
from hybridlc.analysis.results import load_item_results, load_registry
from hybridlc.utils.provenance import sha256_file


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="experiments/results")
    ap.add_argument("--registry", default="experiments/registry.jsonl")
    ap.add_argument("--out", default="experiments/analysis/latest")
    ap.add_argument("--gates", default=None, help="JSON file with gate outcomes")
    ap.add_argument("--branch-map", default=None, help="JSON {run_id: branch}")
    ap.add_argument("--failed-runs", default=None, help="JSON {config: n_permanently_failed}")
    ap.add_argument("--scale", default=None)
    ap.add_argument("--n0", type=int, default=None)
    ap.add_argument("--kstar", type=int, default=None)
    ap.add_argument("--ckpt", default=None, help="confirmatory checkpoint directory (default: the unique one)")
    ap.add_argument("--lm4-slope", type=float, default=None)
    ap.add_argument("--B", type=int, default=BOOTSTRAP_B)
    ap.add_argument("--no-bootstrap", action="store_true")
    ap.add_argument("--synthetic", action="store_true", help="generate a planted-effect world into --out first")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    if a.synthetic:
        from hybridlc.analysis.synthetic import World, WorldSpec, default_plan, write_tree
        w = World(WorldSpec())
        write_tree(w, default_plan(w), out / "synthetic")
        a.results, a.registry = str(out / "synthetic" / "results"), str(out / "synthetic" / "registry.jsonl")
        if a.gates is None:
            gates = {g: True for g in ("G0", "G-RP", "FEAS-1", "FEAS-2", "G-FULL")}
        else:
            gates = json.loads(Path(a.gates).read_text())
    else:
        gates = json.loads(Path(a.gates).read_text()) if a.gates else {}

    reg = load_registry(a.registry)
    if a.branch_map:
        bm = json.loads(Path(a.branch_map).read_text())
        reg = {rid: (dataclasses.replace(info, branch=bm[rid]) if rid in bm else info) for rid, info in reg.items()}
    t0 = time.time()
    df = load_item_results(a.results, reg)
    if df.empty:
        print(f"no item-level results under {a.results}", file=sys.stderr)
        return 2
    # Descriptive per-cell summary of every suite/checkpoint (always written; this is what dev-suite
    # checkpoint monitoring -- DC at TEF 250/500, FD, DG -- reads). Added by the S0 integrator.
    cells = (df.groupby(["run_id", "config", "seed", "branch", "ckpt", "suite", "task", "level", "T"], dropna=False)
               .agg(n_queries=("em_tf", "size"), n_contexts=("context_id", "nunique"), em_tf=("em_tf", "mean"),
                    logp_gold=("logp_gold", "mean")).reset_index())
    cells.to_csv(out / "cell_summary.csv", index=False)
    files = list(df.attrs.get("inputs", [])) + ([a.registry] if Path(a.registry).is_file() else [])
    (out / "inputs.json").write_text(json.dumps({f: sha256_file(f) for f in files}, indent=1))
    print(f"cell summary: {len(cells)} cells, {len(df):,} query rows -> {out / 'cell_summary.csv'}")
    if not (df["suite"] == "CS").any():
        print("no CS (confirmatory) rows: the §11 pipeline was not run (descriptive summary only)")
        (out / "results.json").write_text(json.dumps({"protocol": "v1.0 §11", "status": "not_run",
                                                      "reason": "no CS rows", "suites": sorted(df["suite"].unique())}))
        return 0
    spec = AnalysisSpec(main_ckpt=a.ckpt, scale=a.scale, n0=a.n0, gates=gates, B=a.B, bootstrap=not a.no_bootstrap,
                        kstar=a.kstar, lm4_slope=a.lm4_slope,
                        failed_runs=json.loads(Path(a.failed_runs).read_text()) if a.failed_runs else {})
    res = run_analysis(df, spec)
    ver = V.compute_verdicts(res)
    (out / "results.json").write_text(dumps(res))
    (out / "verdicts.json").write_text(V.dumps(ver))
    print(f"analysed {len(df):,} query rows from {len(files)} files in {time.time() - t0:.1f}s -> {out}")
    for h, v in ver["confirmatory"].items():
        lab = f" [{'; '.join(v['labels'])}]" if v["labels"] else ""
        print(f"  {v['sentence']}{lab}")
    for h, v in ver["secondary"].items():
        print(f"  {h}: {v['verdict']}")
    for w in ver["warnings"]:
        print(f"  warning: {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""S2 FEAS decisions (DTS-10; ``experiments/prereg/DTS10_acceptance.v2.yaml``) -> feas_decisions.json.

    python scripts/feas_decide.py --results experiments/results --registry experiments/registry.jsonl \\
        [--design design.json] [--lm2 lm2.json | --runs-root runs] [--ckpt tef1000] \\
        [--out experiments/feas/feas_decisions.json]
    python scripts/feas_decide.py --results <tree> --structure-only     # parse + count only, no rule, no score

Exit codes: 0 decisions written; 2 refused (missing/invalid inputs, each named on stderr; nothing imputed);
1 other error. CPU only (no CUDA import).

Technical blinding (DTS10-AT v2): while no ``DTS10-commit`` note exists in the registry used (or the study
registry) and the design has exp_stage S2 runs, stdout carries only the output path and counts, never an outcome
or a score; the decisions go to the JSON file only (the procedural rule still applies to opening it).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from hybridlc.analysis import feas as FE  # noqa: E402
from hybridlc.integration.blinding import STUDY_REGISTRY, dts10_committed, is_blinded_stage  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", required=True, help="results root (scripts/evaluate.py --results-root)")
    ap.add_argument("--registry", default=None, help="registry.jsonl (design, lineage, near-miss notes, infra rows)")
    ap.add_argument("--design", default=None, help="JSON list of DesignCell rows (default: from the registry)")
    ap.add_argument("--lm2", default=None, help="JSON {run_id: {before, after}} (default: --runs-root metrics)")
    ap.add_argument("--runs-root", default=None, help="runs root with <run_id>/metrics.jsonl (LM-2 val records)")
    ap.add_argument("--ckpt", default="tef1000", help="checkpoint label of the TEF-1000 evaluation")
    ap.add_argument("--out", default=str(REPO / "experiments" / "feas" / "feas_decisions.json"))
    ap.add_argument("--structure-only", action="store_true", help="parse and count every result file; no rules")
    a = ap.parse_args(argv)

    if a.structure_only:
        inv = FE.structure_inventory(a.results)
        n = sum(len(v) for v in inv["runs"].values())
        print(f"structure: {len(inv['runs'])} run(s), {n} ckpt/suite dir(s), {len(inv['problems'])} problem(s)")
        for p in inv["problems"][:20]:
            print(f"  problem: {p}", file=sys.stderr)
        if a.out and a.out != ap.get_default("out"):
            FE.write_json(inv, a.out)
        return 0 if not inv["problems"] else 2

    try:
        reg = FE.read_registry(a.registry) if a.registry else []
        design = FE.load_design_json(a.design) if a.design else FE.design_from_registry(reg)
        lm2 = json.loads(Path(a.lm2).read_text()) if a.lm2 else None
        inp = FE.load_inputs(a.results, design, ckpt=a.ckpt, lm2=lm2, runs_root=a.runs_root, registry=reg)
        out = FE.decide(inp)
    except FE.FeasInputError as e:
        print(f"REFUSED: {len(e.missing)} missing/invalid FEAS input(s) (nothing imputed):", file=sys.stderr)
        for m in e.missing:
            print(f"  missing: {m}", file=sys.stderr)
        return 2
    outp = Path(a.out)
    FE.write_json(out, outp)
    prec_dir = outp.parent / "precision"
    for q in ("Q1", "Q2", "Q3", "Q4"):
        FE.write_json(out["precision"][q], prec_dir / f"{q.lower()}.json")
    regs = [r for r in (a.registry, str(STUDY_REGISTRY)) if r]
    blind = any(is_blinded_stage(d.run_id) for d in design) and not dts10_committed(regs)
    print(f"wrote {outp} ({len(out['verdicts'])} verdicts, {out['inputs']['n_files_read']} files read; "
          f"precision blocks under {prec_dir})")
    if blind:
        print("BLINDED (DTS10-AT v2): no DTS10-commit note; outcomes are in the file only")
    else:
        h = out["header"]
        print(f"score source {h['score_source']}; flagged {h['flagged']}; held for PI row {h['held_for_pi_row']}")
        for v, r in out["verdicts"].items():
            print(f"  {v}: {r.get('outcome')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

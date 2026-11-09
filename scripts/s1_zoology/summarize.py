"""Aggregate S1(a) run JSONs into experiments/s1_zoology/s1a_results.json and a Markdown table.

    python scripts/s1_zoology/summarize.py

Reads only experiments/s1_zoology/runs/*.json (smoke files ``*.smoke.json`` are ignored), so
every number in the output comes from a re-run (protocol §13.3: "Criteria use re-run numbers only").
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import s1_analysis as A  # noqa: E402

REPO = HERE.parents[1]
RES = REPO / "experiments" / "s1_zoology"
TEST_COLS = ["T64_kv4", "T64_kv8", "T64_kv16", "T128_kv32", "T256_kv64", "T512_kv128", "T1024_kv256"]


def load_results(res: Path = RES) -> list[dict]:
    out = []
    for p in sorted((res / "runs").glob("*.json")):
        if p.name.endswith(".smoke.json"):
            continue
        out.append(json.loads(p.read_text()))
    return out


def markdown_table(results: list[dict]) -> str:
    hdr = ["family", "d", "lr", "epochs", "overall"] + TEST_COLS + ["wall min"]
    lines = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    order = {(f, d): i for i, (f, d) in enumerate(A.PHASE1[:1] + A.PHASE2[:1] + A.PHASE1[1:] + A.PHASE2[1:])}
    for r in sorted(results, key=lambda r: (order.get((r["family"], r["d_model"]), 99), r["lr"])):
        pc = r["final_per_config_acc"]
        cells = [r["family"], str(r["d_model"]), A.lr_tag(r["lr"]),
                 f"{r['epochs_run']}{'*' if r['early_stopped'] else ''}", f"{r['final_overall_acc']:.4f}"]
        cells += [f"{pc[c]:.3f}" if c in pc else "—" for c in TEST_COLS]
        cells.append(f"{r['wall_time_s'] / 60:.1f}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--res-dir", default=str(RES), help="results root (default experiments/s1_zoology)")
    res = Path(ap.parse_args().res_dir)
    results = load_results(res)
    grp = A.grp_a(results)
    cut = res / "cut_decision.json"
    summary = {
        "protocol": "§17 part (a); gate G-RP (a) §13.3",
        "n_runs": len(results),
        "total_wall_h": sum(r["wall_time_s"] for r in results) / 3600,
        "cut_decision": json.loads(cut.read_text()) if cut.exists() else None,
        "grp_a": grp,
        "runs": [{k: r[k] for k in ("run_id", "family", "d_model", "lr", "epochs_run", "early_stopped",
                                     "final_overall_acc", "final_per_config_acc", "best_overall_acc",
                                     "best_epoch", "wall_time_s", "peak_mem_gib")} for r in results],
    }
    (res / "s1a_results.json").write_text(json.dumps(summary, indent=1))
    (res / "s1a_table.md").write_text(markdown_table(results) + "\n")
    print(markdown_table(results))
    print(json.dumps(grp, indent=1))


if __name__ == "__main__":
    main()

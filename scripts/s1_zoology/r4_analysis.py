"""Evaluate R4 exactly as pre-registered (experiments/repro/R4/prereg.yaml; plan §2.4).

    python scripts/s1_zoology/r4_analysis.py

Reads experiments/repro/R4/runs/*.json (ours) and experiments/s1_zoology/runs/*.json (Zoology, R1 re-runs),
writes experiments/repro/R4/r4_results.json and experiments/repro/R4/r4_table.md. Pure CPU, no torch import.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
R4 = REPO / "experiments" / "repro" / "R4"
S1A_RUNS = REPO / "experiments" / "s1_zoology" / "runs"
TOL = 0.03
R4B_MIN = 0.95
CONFIGS = ["T64_kv4", "T64_kv8", "T64_kv16", "T128_kv32", "T256_kv64", "T512_kv128", "T1024_kv256"]
IN_DIST = CONFIGS[:5]
LABEL = {"T64_kv4": "64/4", "T64_kv8": "64/8", "T64_kv16": "64/16", "T128_kv32": "128/32", "T256_kv64": "256/64",
         "T512_kv128": "512/128", "T1024_kv256": "1024/256"}

GRP_B_CELLS = [  # (ours, zoology reference) — frozen in prereg.yaml
    ("our_fg-d64-lr1.0e-02", "attention-d64-lr1.0e-02"),
    ("our_fg-d128-lr1.0e-03", "attention-d128-lr1.0e-03"),
    ("our_gdn-d128-lr1.0e-03", "gated_delta_net-d128-lr1.0e-03"),
    ("our_gdn-d256-lr1.0e-03", "gated_delta_net-d256-lr1.0e-03"),
]
R4B = ["our_fg_prod-d128-lr1.0e-03", "our_fg_prod_bf16-d128-lr1.0e-03"]
CONFIG_A = [("our_gdn_configA-d128-lr1.0e-03", "gated_delta_net-d128-lr1.0e-03"),
            ("our_gdn_configA-d256-lr1.0e-03", "gated_delta_net-d256-lr1.0e-03")]


def load(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def grp_b_cell(ours: dict | None, ref: dict) -> dict:
    if ours is None:
        return {"status": "BLOCKED/NOT RUN", "pass": None}
    diff = ours["final_overall_acc"] - ref["final_overall_acc"]
    return {"ours": ours["final_overall_acc"], "zoology": ref["final_overall_acc"], "diff": diff,
            "pass": abs(diff) <= TOL,
            "per_config": {c: {"ours": ours["final_per_config_acc"].get(c), "zoology": ref["final_per_config_acc"].get(c),
                               "diff": (ours["final_per_config_acc"].get(c, float("nan"))
                                        - ref["final_per_config_acc"].get(c, float("nan")))} for c in CONFIGS},
            "epochs_ours": ours["epochs_run"], "epochs_zoology": ref["epochs_run"],
            "early_stopped_ours": ours["early_stopped"], "collapsed": ours.get("collapsed", False),
            "state_size_ours": ours["model_info"].get("state_size"), "state_size_zoology": ref["model_info"].get("state_size"),
            "params_ours": ours["model_info"].get("num_parameters"), "params_zoology": ref["model_info"].get("num_parameters"),
            "wall_min_ours": ours["wall_time_s"] / 60, "wall_min_zoology": ref["wall_time_s"] / 60}


def r4b_run(r: dict | None) -> dict:
    if r is None:
        return {"status": "BLOCKED/NOT RUN", "pass": None}
    pc = r["final_per_config_acc"]
    m = sum(pc[c] for c in IN_DIST) / len(IN_DIST)
    return {"in_dist_mean": m, "pass": m >= R4B_MIN, "overall": r["final_overall_acc"], "per_config": pc,
            "epochs": r["epochs_run"], "collapsed": r.get("collapsed", False),
            "varlen_attn_training": r.get("varlen_attn_training"), "attn_stats": r.get("attn_stats_post_training")}


def infra_for(rid: str) -> list[dict]:
    out = []
    for p in sorted((R4 / "infra").glob(f"R4-{rid}-a*.infra.json")):
        d = json.loads(p.read_text())
        out.append({"file": str(p.relative_to(REPO)), "flagged": d.get("flagged"), "journal_ok": d.get("journal_ok"),
                    "events": d.get("infra_events"), "returncode": d.get("returncode"),
                    "start": d.get("start_iso"), "end": d.get("end_iso"),
                    "lock_wait_exclusive": (d.get("lock") or {}).get("wait_exclusive"),
                    "compute_apps_seen": d.get("gpu", {}).get("compute_apps_seen"),
                    "any_slowdown": d.get("gpu", {}).get("any_slowdown"),
                    "git_commit": (d.get("meta") or {}).get("git_commit")})
    return out


def main() -> int:
    runs = {p.stem: json.loads(p.read_text()) for p in (R4 / "runs").glob("*.json")
            if not p.stem.endswith(".smoke") and ".flagged" not in p.stem}
    cells = {o: grp_b_cell(runs.get(o), load(S1A_RUNS / f"{z}.json")) for o, z in GRP_B_CELLS}
    statuses = [c["pass"] for c in cells.values()]
    grp_b = "PASS" if all(s is True for s in statuses) else ("FAIL" if any(s is False for s in statuses) else "BLOCKED")
    r4b = {rid: r4b_run(runs.get(rid)) for rid in R4B}
    rs = [v["pass"] for v in r4b.values()]
    r4b_v = "PASS" if all(s is True for s in rs) else ("FAIL" if any(s is False for s in rs) else "BLOCKED")
    cfa = {o: grp_b_cell(runs.get(o), load(S1A_RUNS / f"{z}.json")) for o, z in CONFIG_A}
    for v in cfa.values():
        v.pop("pass", None)  # descriptive only
    res = {"item": "R4", "prereg": "experiments/repro/R4/prereg.yaml", "tolerance": TOL,
           "mdd_note": "+-0.03 overall = +-0.21 on the 1024/256 cell (overall = unweighted mean of 7 cells)",
           "grp_b": {"verdict": grp_b, "cells": cells}, "r4b": {"verdict": r4b_v, "runs": r4b, "threshold": R4B_MIN},
           "config_a_descriptive": cfa,
           "infra": {rid: infra_for(rid) for rid in [o for o, _ in GRP_B_CELLS] + R4B + [o for o, _ in CONFIG_A]}}
    (R4 / "r4_results.json").write_text(json.dumps(res, indent=1))

    L = ["# R4 results table (generated by scripts/s1_zoology/r4_analysis.py)", "",
         f"G-RP(b): **{grp_b}** (|ours − Zoology| ≤ {TOL} on final overall accuracy, per cell). R4b: **{r4b_v}**.", "",
         "| run | vs Zoology | ours | Zoology | Δ | pass | " + " | ".join(LABEL[c] + " (ours/Z)" for c in CONFIGS) + " | epochs (ours/Z) |",
         "|---|---|---|---|---|---|" + "---|" * len(CONFIGS) + "---|"]
    for (o, z), c in zip(GRP_B_CELLS + CONFIG_A, list(cells.values()) + list(cfa.values())):
        if "ours" not in c:
            L.append(f"| {o} | {z} | — | — | — | {c['status']} |" + " |" * len(CONFIGS) + " |")
            continue
        pcs = " | ".join(f"{c['per_config'][k]['ours']:.4f}/{c['per_config'][k]['zoology']:.4f}" for k in CONFIGS)
        pas = {True: "✔", False: "✘"}.get(c.get("pass"), "descr.")
        L.append(f"| {o} | {z} | {c['ours']:.4f} | {c['zoology']:.4f} | {c['diff']:+.4f} | {pas} | {pcs} | "
                 f"{c['epochs_ours']}/{c['epochs_zoology']} |")
    L += ["", "| R4b run | in-dist mean | pass | overall | " + " | ".join(LABEL[c] for c in CONFIGS)
          + " | epochs | varlen_attn calls |", "|---|---|---|---|" + "---|" * len(CONFIGS) + "---|---|"]
    for rid, v in r4b.items():
        if "in_dist_mean" not in v:
            L.append(f"| {rid} | — | {v['status']} |" + " |" * (len(CONFIGS) + 3))
            continue
        L.append(f"| {rid} | {v['in_dist_mean']:.4f} | {'✔' if v['pass'] else '✘'} | {v['overall']:.4f} | "
                 + " | ".join(f"{v['per_config'][c]:.4f}" for c in CONFIGS)
                 + f" | {v['epochs']} | {(v['varlen_attn_training'] or {}).get('calls')} |")
    (R4 / "r4_table.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())

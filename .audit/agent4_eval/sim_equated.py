#!/usr/bin/env python
"""Agent-4: veto rate of the H3 U-11 equated-pair sign check (pipeline.equated_check) on a realistic
ladder where the true D3 = R_ret(3) - R_hop(3) = 0.9 - 0.7 = +0.2 at EVERY level (so any veto is false).
CPU only; uses the project's equated_check on synthetic CellTables (300 contexts x Q=4, 5 seeds)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path("/home/sedrica-lynx/Downloads/outeropt")
sys.path.insert(0, str(REPO / "src"))
from hybridlc.analysis.pipeline import equated_check  # noqa: E402
from hybridlc.analysis.scores import CellTable  # noqa: E402

T = 32768
RET = {("R1", "D3"): (0.85, 0.98), ("R1", "D15"): (0.55, 0.95), ("R-load", "N8"): (0.75, 0.97),
       ("R-load", "N32"): (0.40, 0.92), ("R-load", "N128"): (0.15, 0.80), ("R-load", "N512"): (0.05, 0.55),
       ("R-load", "N1024"): (0.03, 0.35)}
HOP = {("M-hop", "h1"): (0.45, 0.95), ("M-hop", "h2"): (0.30, 0.85), ("M-hop", "h3"): (0.20, 0.70),
       ("M-hop", "h4"): (0.15, 0.55)}
R_RET, R_HOP = 0.9, 0.7
SEEDS = [0, 1, 2, 3, 4]


def table(rng, levels, configs, R, n_ctx, seed_sd=0.02, run_sd=0.02):
    strata = [(t, lv, T) for (t, lv) in levels]
    seed_eff = rng.normal(0, seed_sd, len(SEEDS))
    run_eff = rng.normal(0, run_sd, (len(configs), len(SEEDS)))
    ctx = []
    for (t, lv) in levels:
        s0, s12 = levels[(t, lv)]
        mu = {"C0": s0, "C3": s0 + R * (s12 - s0), "C12": s12}
        u = rng.normal(0, 0.10, n_ctx)
        arr = np.empty((n_ctx, len(configs), len(SEEDS)))
        for ci, c in enumerate(configs):
            for si in range(len(SEEDS)):
                p = np.clip(mu[c] + seed_eff[si] + run_eff[ci, si] + u, 0, 1)
                arr[:, ci, si] = rng.binomial(4, p) / 4
        ctx.append(arr)
    return CellTable("X", "em_tf", strata, list(configs), list(SEEDS), ctx, [np.arange(n_ctx)] * len(strata),
                     np.ones((len(configs), len(SEEDS)), bool))


def merge(a: CellTable, b: CellTable) -> CellTable:
    return CellTable("CS", "em_tf", a.strata + b.strata, a.configs, a.seeds, a.ctx + b.ctx,
                     a.context_ids + b.context_ids, a.present)


def main(n_rep=400):
    rng = np.random.default_rng(99)
    vetoes, npairs, pair_counts = 0, [], {}
    for _ in range(n_rep):
        cs = merge(table(rng, RET, ["C0", "C3", "C12"], R_RET, 300), table(rng, HOP, ["C0", "C3", "C12"], R_HOP, 200))
        dg = table(rng, RET, ["C0", "C12"], R_RET, 200)
        dgh = table(rng, HOP, ["C12"], R_HOP, 200)
        res = equated_check(cs, dg, [T], 3, dgh)
        npairs.append(len(res["pairs"]))
        for p in res["pairs"]:
            key = f"{p['retrieval']}~{p['multihop']}"
            d = pair_counts.setdefault(key, [0, 0])
            d[0] += 1
            d[1] += p["D3"] <= 0
        vetoes += not res["consistent"]
    out = dict(true_D3_every_level=R_RET - R_HOP, n_rep=n_rep, false_veto_rate=vetoes / n_rep,
               mean_pairs=float(np.mean(npairs)),
               per_pair={k: dict(times_equated=v[0], negative_rate=v[1] / max(v[0], 1)) for k, v in sorted(pair_counts.items())})
    print(json.dumps(out, indent=1))
    (REPO / ".audit" / "agent4_eval" / "sim_equated.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

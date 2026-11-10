"""Evaluate R5 exactly as pre-registered (experiments/repro/R5/prereg.yaml; plan §2.5).

    python scripts/s1_zoology/r5_analysis.py

Reads experiments/repro/R5/runs/*.json (both arms; *.smoke.json and *.flagged*.json are excluded) and the R1 Zoology
reference JSONs; writes experiments/repro/R5/r5_results.json and experiments/repro/R5/r5_table.md. No torch import.
"""
from __future__ import annotations

import json
import math
import statistics as st
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
R5 = REPO / "experiments" / "repro" / "R5"
S1A_RUNS = REPO / "experiments" / "s1_zoology" / "runs"
R4_RUNS = REPO / "experiments" / "repro" / "R4" / "runs"
CONFIGS = ["T64_kv4", "T64_kv8", "T64_kv16", "T128_kv32", "T256_kv64", "T512_kv128", "T1024_kv256"]
LABEL = {"T64_kv4": "64/4", "T64_kv8": "64/8", "T64_kv16": "64/16", "T128_kv32": "128/32", "T256_kv64": "256/64",
         "T512_kv128": "512/128", "T1024_kv256": "1024/256"}
ARMS = ("gdn_zoology", "our_gdn_bf16")
SEEDS = (123, 124, 125)
O5_SEEDS = (126, 127)
T_FROZEN = 2.78          # plan §2.5 rule 3 (t0.975, df ~ 4), frozen constant (reading CMP1)
FLOOR = 0.03             # G-RP(b) materiality floor
REGIME = (0.05, 0.95)    # rule 1 (inclusive, reading RG1)
SPAN = 0.2               # rule 4 (strict, reading BM1)
REF_FILES = {128: S1A_RUNS / "gated_delta_net-d128-lr1.0e-03.json", 256: S1A_RUNS / "gated_delta_net-d256-lr1.0e-03.json"}
REF_SHA = {128: "df78ef73a2d19c4dc16f884fcd3597f78163f6998bcc5dd22ed6d9c78b754840",
           256: "d944412f9b308749a42f5993e0e55bda9cb8bb42c0b4958674abf7238f3f5f07"}


def rid(arm: str, seed: int) -> str:
    return f"{arm}-d64-lr1.0e-03-s{seed}"


def load(p: Path) -> dict | None:
    return json.loads(p.read_text()) if p.exists() else None


def sha256(p: Path) -> str:
    import hashlib

    return hashlib.sha256(p.read_bytes()).hexdigest()


def mean_sd(xs: list[float]) -> tuple[float, float]:
    return st.fmean(xs), (st.stdev(xs) if len(xs) > 1 else float("nan"))


def t_ppf(q: float, df: float) -> float:
    from scipy.stats import t

    return float(t.ppf(q, df))


def comparison(ours: list[float], zoo: list[float]) -> dict:
    mo, so = mean_sd(ours)
    mz, sz = mean_sd(zoo)
    no, nz = len(ours), len(zoo)
    se = math.sqrt(so ** 2 / no + sz ** 2 / nz)
    delta = mo - mz
    thr = max(FLOOR, T_FROZEN * se)
    vo, vz = so ** 2 / no, sz ** 2 / nz
    df_ws = (vo + vz) ** 2 / ((vo ** 2 / (no - 1) if no > 1 else 0) + (vz ** 2 / (nz - 1) if nz > 1 else 0)) \
        if (vo + vz) > 0 else 4.0
    t90 = t_ppf(0.95, df_ws)
    return {"mean_ours": mo, "sd_ours": so, "mean_zoology": mz, "sd_zoology": sz, "n_ours": no, "n_zoology": nz,
            "delta": delta, "se_w": se, "threshold": thr, "threshold_term": "floor 0.03" if thr == FLOOR else "2.78*SE_W",
            "fail": abs(delta) > thr, "ci90_welch": [delta - t90 * se, delta + t90 * se], "df_welch_satterthwaite": df_ws,
            "ci90_df4": [delta - 2.132 * se, delta + 2.132 * se],
            "mdd_t2.78_50pct_power": T_FROZEN * se, "mdd_80pct_power_df4": (2.776 + 0.941) * se}


def main() -> None:
    runs: dict[str, dict[int, dict]] = {a: {} for a in ARMS}
    for a in ARMS:
        for s in SEEDS + O5_SEEDS:
            r = load(R5 / "runs" / f"{rid(a, s)}.json")
            if r is not None:
                runs[a][s] = r
    refs = {d: load(p) for d, p in REF_FILES.items()}
    ref_ok = {d: sha256(p) == REF_SHA[d] for d, p in REF_FILES.items()}
    out: dict = {"prereg": "experiments/repro/R5/prereg.yaml", "prereg_sha256": sha256(R5 / "prereg.yaml"),
                 "reference_sha256_ok": ref_ok,
                 "runs": {a: {s: {"file": f"experiments/repro/R5/runs/{rid(a, s)}.json",
                                  "final_overall_acc": r["final_overall_acc"], "per_config": r["final_per_config_acc"],
                                  "epochs_run": r["epochs_run"], "early_stopped": r["early_stopped"],
                                  "collapsed": r.get("collapsed", False), "resumed_at_epochs": r.get("resumed_at_epochs"),
                                  "wall_time_s": r["wall_time_s"], "state_size": r["model_info"].get("state_size"),
                                  "num_parameters": r["model_info"].get("num_parameters"),
                                  "gdn_kernel_calls": r.get("gdn_kernel_calls")}
                              for s, r in sorted(runs[a].items())} for a in ARMS}}
    have3 = all(s in runs[a] for a in ARMS for s in SEEDS)
    if not have3:
        out["verdict"] = "INCOMPLETE (not all 6 runs present)"
        out["missing"] = [rid(a, s) for a in ARMS for s in SEEDS if s not in runs[a]]
        _write(out)
        return

    def cell_vals(arm: str, cell: str, seeds=SEEDS) -> list[float]:
        return [runs[arm][s]["final_per_config_acc"][cell] for s in seeds]

    # Rule 1: regime check on Zoology's 3-seed mean on 1024/256.
    zmean = {c: st.fmean(cell_vals("gdn_zoology", c)) for c in CONFIGS}
    m1024 = zmean["T1024_kv256"]
    if m1024 > REGIME[1]:
        lo = min(zmean.values())
        primary = [c for c in CONFIGS if zmean[c] == lo][-1]  # ties -> longer config
        regime = {"status": "above 0.95 -> lowest-accuracy cell, labelled 'low power'", "low_power": True}
    elif m1024 < REGIME[0]:
        primary, regime = "T512_kv128", {"status": "below 0.05 -> 512/128", "low_power": False}
    else:
        primary, regime = "T1024_kv256", {"status": "in [0.05, 0.95]", "low_power": False}
    out["rule1_regime"] = {"zoology_mean_1024_256": m1024, "primary_cell": primary, **regime,
                           "zoology_mean_by_cell": zmean}

    # Rule 4: bimodality on the 3 initial seeds.
    spans = {a: max(cell_vals(a, primary)) - min(cell_vals(a, primary)) for a in ARMS}
    fired = any(v > SPAN for v in spans.values())
    out["rule4_bimodality"] = {"span_by_arm": spans, "fired": fired,
                               "o5_runs_present": {a: sorted(s for s in O5_SEEDS if s in runs[a]) for a in ARMS}}
    seeds_used = SEEDS
    if fired and all(s in runs[a] for a in ARMS for s in O5_SEEDS):
        seeds_used = SEEDS + O5_SEEDS

    # Rule 2: phenomenon (Zoology arm vs R1's Zoology values on the primary cell).
    zd64 = st.fmean(cell_vals("gdn_zoology", primary, seeds_used))
    r128 = refs[128]["final_per_config_acc"][primary]
    r256 = refs[256]["final_per_config_acc"][primary]
    out["rule2_phenomenon"] = {"zoology_d64_mean": zd64, "ref_d128_R1": r128, "ref_d256_R1": r256,
                               "seeds": list(seeds_used), "pass": (zd64 < r128) and (r128 < r256),
                               "zoology_d64_overall_mean": st.fmean([runs["gdn_zoology"][s]["final_overall_acc"]
                                                                     for s in seeds_used]),
                               "figure_context_overall": "≈ 0.89 (fig, approx.; context only)"}
    # Rule 3: comparison.
    cmp_primary = comparison(cell_vals("our_gdn_bf16", primary, seeds_used), cell_vals("gdn_zoology", primary, seeds_used))
    out["rule3_comparison"] = {"cell": primary, "seeds": list(seeds_used), **cmp_primary,
                               "pass": not cmp_primary["fail"]}
    # Secondary / descriptive.
    desc = {}
    for c in CONFIGS:
        desc[c] = comparison(cell_vals("our_gdn_bf16", c, seeds_used), cell_vals("gdn_zoology", c, seeds_used))
    ov = comparison([runs["our_gdn_bf16"][s]["final_overall_acc"] for s in seeds_used],
                    [runs["gdn_zoology"][s]["final_overall_acc"] for s in seeds_used])
    out["descriptive"] = {"per_cell": desc, "overall": ov,
                          "seed_sd_overall_vs_grp_b_0.03": {"zoology": ov["sd_zoology"], "ours": ov["sd_ours"]}}
    r4 = {d: load(R4_RUNS / f"our_gdn-d{d}-lr1.0e-03.json") for d in (128, 256)}
    out["descriptive"]["ours_fp32_R4_primary_cell"] = {d: (r["final_per_config_acc"][primary] if r else None)
                                                      for d, r in r4.items()}

    # Verdict (reading V1).
    if fired:
        verdict = "INCONCLUSIVE (rule 4 bimodality; comparison descriptive)"
        if seeds_used == SEEDS:
            verdict += " — O5 runs pending"
    elif not out["rule3_comparison"]["pass"]:
        verdict = "FAIL (rule 3 comparison)"
    elif not out["rule2_phenomenon"]["pass"]:
        verdict = "FAIL (rule 2: phenomenon not replicated; comparison passes)"
    else:
        verdict = "PASS"
    out["verdict"] = verdict
    out["any_collapsed"] = any(r.get("collapsed", False) for a in ARMS for r in runs[a].values())
    _write(out)


def _write(out: dict) -> None:
    (R5 / "r5_results.json").write_text(json.dumps(out, indent=1, default=str))
    lines = [f"# R5 table (generated by scripts/s1_zoology/r5_analysis.py)", "", f"Verdict: **{out['verdict']}**", ""]
    if "rule1_regime" in out:
        p = out["rule1_regime"]["primary_cell"]
        lines += ["| Run | seed | overall | " + " | ".join(LABEL[c] for c in CONFIGS) + " | epochs | collapsed |",
                  "|---|---|---|" + "---|" * len(CONFIGS) + "---|---|"]
        for a in ARMS:
            for s, r in out["runs"][a].items():
                lines.append(f"| {a} | {s} | {r['final_overall_acc']:.5f} | "
                             + " | ".join(f"{r['per_config'][c]:.4f}" for c in CONFIGS)
                             + f" | {r['epochs_run']} | {r['collapsed']} |")
        c3 = out["rule3_comparison"]
        lines += ["", f"Primary cell: {LABEL[p]} ({out['rule1_regime']['status']})",
                  f"Rule 2: Zoology d64 mean {out['rule2_phenomenon']['zoology_d64_mean']:.5f} < d128 "
                  f"{out['rule2_phenomenon']['ref_d128_R1']:.5f} < d256 {out['rule2_phenomenon']['ref_d256_R1']:.5f}"
                  f" -> {'PASS' if out['rule2_phenomenon']['pass'] else 'FAIL'}",
                  f"Rule 3: Δ = {c3['delta']:+.5f}, SE_W = {c3['se_w']:.5f}, threshold = {c3['threshold']:.5f} "
                  f"({c3['threshold_term']}) -> {'PASS' if c3['pass'] else 'FAIL'}; 90% CI (Welch df "
                  f"{c3['df_welch_satterthwaite']:.2f}) [{c3['ci90_welch'][0]:+.5f}, {c3['ci90_welch'][1]:+.5f}]",
                  f"Rule 4: spans {out['rule4_bimodality']['span_by_arm']} -> fired={out['rule4_bimodality']['fired']}"]
    (R5 / "r5_table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())

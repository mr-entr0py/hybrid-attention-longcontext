"""Pure-Python S1(a) logic: run order, pre-declared cut rule, G-RP criterion (a).

No Zoology / torch imports, so it can be unit-tested anywhere
(``scripts/s1_zoology/test_s1_analysis.py``).

Protocol references:
* §17 "Runs": attention d ∈ {64, 128}, GDN d ∈ {128, 256}, × LR ∈ logspace(−3, −1.5, 4).
* §17 "Pre-declared cut": if the first timed run implies > 3.0 GPU-h in total, cut to the
  2 best LRs per family, chosen on the first d_model completed at 4 LRs.
* §13.3 G-RP (a): best over LR on overall test accuracy (all 7 test configs):
  GDN ≥ 0.95 at d_model ≥ 128; attention ≥ 0.95 at d_model ≥ 64; GDN accuracy
  non-decreasing in d_model.
"""
from __future__ import annotations

import math
from typing import Any, Iterable

# logspace(-3, -1.5, 4) as in original_mqar_configs.py (computed, not typed in).
LRS: tuple[float, ...] = tuple(10 ** (-3 + 0.5 * i) for i in range(4))
WIDTHS: dict[str, tuple[int, ...]] = {"attention": (64, 128), "gated_delta_net": (128, 256)}
FIRST_WIDTH: dict[str, int] = {f: w[0] for f, w in WIDTHS.items()}
PHASE1 = [("attention", 64), ("gated_delta_net", 128)]
PHASE2 = [("attention", 128), ("gated_delta_net", 256)]
N_RUNS_FULL = sum(len(w) for w in WIDTHS.values()) * len(LRS)  # 16
CUT_BUDGET_GPU_H = 3.0
CUT_KEEP_LRS = 2
GRP_THRESHOLD = 0.95


def lr_tag(lr: float) -> str:
    return f"{lr:.1e}"


def run_id(family: str, d_model: int, lr: float) -> str:
    return f"{family}-d{d_model}-lr{lr_tag(lr)}"


def phase_run_ids(phase: list[tuple[str, int]], lrs_by_family: dict[str, Iterable[float]] | None = None) -> list[str]:
    out = []
    for fam, d in phase:
        lrs = LRS if lrs_by_family is None else tuple(lrs_by_family[fam])
        out += [run_id(fam, d, lr) for lr in sorted(lrs)]
    return out


def cut_decision(first_run_wall_s: float, n_runs: int = N_RUNS_FULL, budget_h: float = CUT_BUDGET_GPU_H) -> dict[str, Any]:
    """Pre-declared rule: implied total = first run's wall time × number of planned runs."""
    implied_h = first_run_wall_s * n_runs / 3600.0
    return {"first_run_wall_s": first_run_wall_s, "n_runs_planned": n_runs,
            "implied_total_gpu_h": implied_h, "budget_gpu_h": budget_h, "cut": implied_h > budget_h}


def best_lrs(results: list[dict[str, Any]], family: str, d_model: int, k: int = CUT_KEEP_LRS) -> list[float]:
    """The k LRs with the highest final overall accuracy at (family, d_model).

    Ties are broken towards the smaller LR (deterministic)."""
    rows = [r for r in results if r["family"] == family and r["d_model"] == d_model and not r.get("smoke")]
    if len({round(math.log10(r["lr"]), 6) for r in rows}) < len(LRS):
        raise ValueError(f"{family} d={d_model}: only {len(rows)} of {len(LRS)} LRs complete")
    rows.sort(key=lambda r: (-r["final_overall_acc"], r["lr"]))
    return [r["lr"] for r in rows[:k]]


def best_over_lr(results: list[dict[str, Any]]) -> dict[str, dict[int, dict[str, Any]]]:
    """family -> d_model -> {acc, lr, run_id, n_lrs} using final overall accuracy."""
    table: dict[str, dict[int, dict[str, Any]]] = {}
    for r in results:
        if r.get("smoke"):
            continue
        cell = table.setdefault(r["family"], {}).get(r["d_model"])
        if cell is None or r["final_overall_acc"] > cell["acc"]:
            n = (cell or {}).get("n_lrs", 0)
            cell = {"acc": r["final_overall_acc"], "lr": r["lr"], "run_id": r["run_id"], "n_lrs": n}
            table[r["family"]][r["d_model"]] = cell
        cell["n_lrs"] = sum(1 for q in results if not q.get("smoke") and q["family"] == r["family"]
                            and q["d_model"] == r["d_model"])
    return table


def grp_a(results: list[dict[str, Any]], threshold: float = GRP_THRESHOLD) -> dict[str, Any]:
    """Evaluate G-RP criterion (a). Missing cells make the gate 'incomplete', never 'pass'."""
    t = best_over_lr(results)
    checks: list[dict[str, Any]] = []
    for fam, widths in WIDTHS.items():
        for d in widths:
            cell = t.get(fam, {}).get(d)
            checks.append({
                "check": f"{fam} d={d} best-over-LR >= {threshold}",
                "value": None if cell is None else cell["acc"],
                "best_lr": None if cell is None else cell["lr"],
                "n_lrs": 0 if cell is None else cell["n_lrs"],
                "pass": None if cell is None else bool(cell["acc"] >= threshold),
            })
    g = t.get("gated_delta_net", {})
    ws = WIDTHS["gated_delta_net"]
    if all(w in g for w in ws):
        accs = [g[w]["acc"] for w in ws]
        mono = all(b >= a for a, b in zip(accs, accs[1:]))
        checks.append({"check": "GDN best-over-LR accuracy non-decreasing in d_model",
                       "value": dict(zip(map(str, ws), accs)), "pass": bool(mono)})
    else:
        checks.append({"check": "GDN best-over-LR accuracy non-decreasing in d_model", "value": None, "pass": None})
    if any(c["pass"] is None for c in checks):
        verdict = "INCOMPLETE"
    else:
        verdict = "PASS" if all(c["pass"] for c in checks) else "FAIL"
    return {"criterion": "G-RP (a)", "metric": "final-epoch overall test accuracy (7 test configs), best over LR",
            "threshold": threshold, "checks": checks, "verdict": verdict, "best_over_lr": {
                f: {str(d): v for d, v in m.items()} for f, m in t.items()}}

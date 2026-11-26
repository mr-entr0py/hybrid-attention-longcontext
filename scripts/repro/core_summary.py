#!/usr/bin/env python
"""Descriptive summary of the scored core runs (UNREGISTERED, EXPLORATORY; D103-D105). CPU only.

Reads experiments/results/S4T-*/tef1000/FD. Per run: the retrieval ladder mean at each length (R1 and R-load
weighted equally, as in D018), the no-filler control ladder, M-hop and A-freq means, and the P2 re-score flip
rate. Per k: mean over the seeds present, with the individual seed values. Recovery R(k) = (S_k - S_0) /
(S_12 - S_0), per seed where all three exist. No hypothesis test and no verdict.
Output: experiments/core/summary.json and summary.md
"""
from __future__ import annotations

import json
import re
import statistics as st
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from hybridlc.analysis import feas as FE  # noqa: E402

LABEL = ("UNREGISTERED, EXPLORATORY (D103-D105): FEAS entry BLOCKED; no LR sweep or freeze; development items; "
         "no hypothesis test, no verdict.")


def em(path: Path) -> float:
    rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    rows = [r for r in rows if "EM_tf" in r]
    return sum(bool(r["EM_tf"]) for r in rows) / len(rows)


def mean(xs):
    return sum(xs) / len(xs)


def main() -> int:
    rows = [json.loads(x) for x in (REPO / "experiments" / "registry.jsonl").read_text().splitlines() if x.strip()]
    info = {}
    for r in rows:
        m = re.match(r"S4_D384_(C\d+)_s(\d+)_B060_T\.yaml$", Path(str(r.get("config", ""))).name)
        if r.get("status") == "queued" and m:
            info[r["run_id"]] = (int(m.group(1)[1:]), int(m.group(2)))
    lengths = {"4K": FE.T4K, "8K": FE.T8K, "16K": FE.T16K, "32K": FE.T32K}
    per = {}
    for rid, (k, seed) in sorted(info.items(), key=lambda x: (x[1][1], x[1][0])):
        d = REPO / "experiments" / "results" / rid / "tef1000" / "FD"
        if not (REPO / "experiments" / "core" / "logs" / f"{rid}.done").is_file():
            continue
        cells = {f.stem: em(f) for f in d.glob("*.jsonl")}
        out = {"k": k, "seed": seed, "run_id": rid, "ladder": {}, "control_ladder": {}}
        for name, T in lengths.items():
            a, b = FE.ladder_cells(T)
            c, e = FE.ladder_cells(T, c0=True)
            out["ladder"][name] = 0.5 * mean([cells[x] for x in a]) + 0.5 * mean([cells[x] for x in b])
            out["control_ladder"][name] = 0.5 * mean([cells[x] for x in c]) + 0.5 * mean([cells[x] for x in e])
        for fam, pre in (("M-hop", "M-hop_"), ("A-freq", "A-freq_")):
            out[fam] = {t: mean([v for n, v in cells.items() if n.startswith(pre) and n.endswith(f"_{T}")])
                        for t, T in (("8K", FE.T8K), ("32K", FE.T32K))}
        flips, n = 0.0, 0
        for f in d.glob("*_rescore.json"):
            p2 = json.loads(f.read_text()).get("P2")
            if p2:
                flips += p2["flips"]
                n += p2["n"]
        out["p2_flip_rate"] = flips / n if n else None
        per[f"k{k}_s{seed}"] = out
    by_k = {}
    for v in per.values():
        by_k.setdefault(v["k"], []).append(v)
    table = {}
    for k, vs in sorted(by_k.items()):
        table[k] = {t: {"mean": mean([v["ladder"][t] for v in vs]), "seeds": {v["seed"]: v["ladder"][t] for v in vs},
                        "sd": st.stdev([v["ladder"][t] for v in vs]) if len(vs) > 1 else None} for t in lengths}
    recovery = {}
    for seed in sorted({v["seed"] for v in per.values()}):
        s = {v["k"]: v for v in per.values() if v["seed"] == seed}
        if 0 in s and 12 in s:
            for k in (1, 3):
                if k in s:
                    for t in lengths:
                        gap = s[12]["ladder"][t] - s[0]["ladder"][t]
                        recovery.setdefault(f"k{k}", {}).setdefault(t, {})[seed] = {
                            "gap_12_minus_0": gap,
                            "R": (s[k]["ladder"][t] - s[0]["ladder"][t]) / gap if gap > 0.05 else None}
    res = {"label": LABEL, "runs": per, "retrieval_ladder_by_k": table, "recovery": recovery}
    out = REPO / "experiments" / "core"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(res, indent=1))
    md = [f"# Core runs: descriptive summary", "", LABEL, "", "Retrieval ladder, exact match (mean over seeds; n = seeds)", "",
          "| k | n | " + " | ".join(lengths) + " |", "|" + "---|" * (len(lengths) + 2)]
    for k, row in sorted(table.items()):
        n = len(row["32K"]["seeds"])
        md.append(f"| {k} | {n} | " + " | ".join(f"{row[t]['mean']:.3f}" for t in lengths) + " |")
    md += ["", "Per run: ladder 32K / no-filler control 32K / M-hop 32K / A-freq 32K / P2 flip rate", ""]
    for key, v in per.items():
        md.append(f"- {key}: {v['ladder']['32K']:.3f} / {v['control_ladder']['32K']:.3f} / {v['M-hop']['32K']:.3f} / "
                  f"{v['A-freq']['32K']:.3f} / {v['p2_flip_rate']}")
    md += ["", "Recovery R(k) at 16K and 32K per seed (blank if the 12-minus-0 gap is 5 points or less)", ""]
    for k, ts in recovery.items():
        for t in ("16K", "32K"):
            md.append(f"- {k} {t}: " + ", ".join(
                f"s{s}: R={'' if x['R'] is None else format(x['R'], '.2f')} (gap {x['gap_12_minus_0']:.3f})"
                for s, x in ts.get(t, {}).items()))
    (out / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

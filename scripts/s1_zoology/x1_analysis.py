"""X1 (exploratory) analysis: MQAR recall vs number of attention mixers k in a 6-layer [conv, M]x3 hybrid.

Reads experiments/explore/X1_mqar_composition/runs/*.json, writes summary.json, summary.md and
fig_x1.{pdf,png}. Descriptive only: no test, no verdict (X1 is not pre-registered as confirmatory).

    .venv/bin/python scripts/s1_zoology/x1_analysis.py
"""
from __future__ import annotations

import glob
import json
import os
import statistics as st
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D = os.path.join(ROOT, "experiments", "explore", "X1_mqar_composition")
CFG = ["T64_kv4", "T64_kv8", "T64_kv16", "T128_kv32", "T256_kv64", "T512_kv128", "T1024_kv256"]
LAB = ["64/4", "64/8", "64/16", "128/32", "256/64", "512/128", "1024/256"]
PAT = {0: "LLL", 1: "LAL", 2: "ALA", 3: "AAA"}


def main() -> None:
    runs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(D, "runs", "*.json"))) if ".smoke" not in f]
    by_k = defaultdict(list)
    for r in runs:
        by_k[int(r["k_attention"])].append(r)
    summ = {}
    for k in sorted(by_k):
        rs = by_k[k]
        cell = {c: [r["final_per_config_acc"][c] for r in rs] for c in CFG}
        summ[k] = {
            "pattern": PAT[k], "n_seeds": len(rs), "seeds": [r["seed"] for r in rs],
            "overall": [r["final_overall_acc"] for r in rs],
            "cell_mean": {c: st.mean(v) for c, v in cell.items()},
            "cell_sd": {c: (st.stdev(v) if len(v) > 1 else 0.0) for c, v in cell.items()},
            "cell_values": cell,
            "wall_min": [round(r["wall_time_s"] / 60, 1) for r in rs],
            "collapsed": [r["collapsed"] for r in rs],
        }
    json.dump(summ, open(os.path.join(D, "summary.json"), "w"), indent=1)
    lines = ["| k | pattern | seeds | overall (mean) | 256/64 | 512/128 | 1024/256 (mean ± sd) |", "|---|---|---|---|---|---|---|"]
    for k, s in summ.items():
        lines.append(f"| {k} | {s['pattern']} | {s['n_seeds']} | {st.mean(s['overall']):.3f} | "
                     f"{s['cell_mean']['T256_kv64']:.3f} | {s['cell_mean']['T512_kv128']:.3f} | "
                     f"{s['cell_mean']['T1024_kv256']:.3f} ± {s['cell_sd']['T1024_kv256']:.3f} |")
    open(os.path.join(D, "summary.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for d in glob.glob(os.path.expanduser("~/.cache/tectonic/bundles/data/*/")):
        for f in glob.glob(os.path.join(d, "STIXTwo*.otf")):
            font_manager.fontManager.addfont(f)
    plt.rcParams.update({"font.family": "STIX Two Text", "mathtext.fontset": "stix", "font.size": 8.5,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.color": "#e6e6e6", "legend.frameon": False, "pdf.fonttype": 42})
    seq = {0: "#9cc3f2", 1: "#3987e5", 2: "#1c5cab", 3: "#0d366b"}
    fig, axs = plt.subplots(1, 2, figsize=(5.5, 2.2))
    ax = axs[0]
    for k, s in summ.items():
        ax.plot(range(len(CFG)), [s["cell_mean"][c] for c in CFG], marker="o", ms=3.5, color=seq[k],
                label=f"$k={k}$ ({s['pattern']})", mec="white", mew=0.7)
    ax.set_xticks(range(len(CFG)))
    ax.set_xticklabels(LAB, rotation=40, fontsize=7)
    ax.set_ylabel("recall accuracy")
    ax.set_ylim(0, 1.04)
    ax.set_title("(a) recall vs. load, mean over seeds", loc="left", fontsize=8.5)
    ax.legend(fontsize=6.8, loc="lower left")
    ax = axs[1]
    ks = sorted(summ)
    for c, mk, col in (("T512_kv128", "s", "#55534e"), ("T1024_kv256", "o", "#000000")):
        for k in ks:
            ax.scatter([k] * len(summ[k]["cell_values"][c]), summ[k]["cell_values"][c], s=9, color=col, alpha=0.35,
                       marker=mk, linewidths=0)
        ax.plot(ks, [summ[k]["cell_mean"][c] for k in ks], marker=mk, ms=3.5, color=col,
                label=LAB[CFG.index(c)] + " cell")
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xlabel("attention mixers $k$ (of 3)")
    ax.set_ylim(0, 1.04)
    ax.set_title("(b) hardest cells vs. $k$ (dots: seeds)", loc="left", fontsize=8.5)
    ax.legend(fontsize=6.8, loc="lower right")
    fig.tight_layout(w_pad=1.5)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(D, f"fig_x1.{ext}"), bbox_inches="tight", dpi=200)
    print("wrote", os.path.join(D, "fig_x1.pdf"))


if __name__ == "__main__":
    main()

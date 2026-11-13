"""Data figures for the pre-project report (print versions, vector PDF).

Every plotted number is read from the repository's raw experiment files; the only
hand-entered values are the literature numbers in fig_prior(), transcribed from the
cited papers (Waleffe et al. values are read off their Fig. 4 and are approximate).

    source .venv/bin/activate && python report/figures/make_report_figures.py
"""
from __future__ import annotations

import glob
import json
import os
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "src"))

# ---- typography: the report's own typeface (Libertinus), so figures match the text ----
FONT_DIR = glob.glob(os.path.expanduser("~/.cache/tectonic/bundles/data/*/"))
for d in FONT_DIR:
    for f in glob.glob(os.path.join(d, "STIXTwo*.otf")):
        font_manager.fontManager.addfont(f)
FAMILY = "STIX Two Text" if any("STIX Two Text" in f.name for f in font_manager.fontManager.ttflist) else "DejaVu Serif"

INK, INK2, GRID = "#000000", "#444444", "#e6e6e6"
# categorical order (dataviz reference palette): blue, orange, aqua
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
# sequential single-hue ramp for k (amount of full attention), light -> dark
SEQ = {0: "#9cc3f2", 1: "#6da7ec", 2: "#5598e7", 3: "#3987e5", 4: "#2a78d6", 6: "#1c5cab", 12: "#0d366b"}

plt.rcParams.update({
    "font.family": FAMILY, "font.size": 8.5, "axes.titlesize": 8.5, "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.3,
    "mathtext.fontset": "stix",
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "lines.linewidth": 1.5, "lines.markersize": 4.2, "legend.frameon": False,
    "axes.titleweight": "normal", "axes.titlelocation": "left", "axes.linewidth": 0.7,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "pdf.fonttype": 42, "figure.dpi": 150,
})
TW = 5.5  # text width in inches (NeurIPS-like column)


def save(fig, name):
    fig.savefig(os.path.join(HERE, f"{name}.pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(os.path.join(HERE, f"{name}.png"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("wrote", name)


def jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def mk(ax_marker_color):
    return dict(mec="white", mew=0.8)


# ------------------------------------------------------------------------------------------
def fig_prior():
    fig, axs = plt.subplots(1, 2, figsize=(TW, 2.15))
    n_attn = [0, 1, 2, 3, 4, 5, 7, 12, 24]
    loss = [2.615, 2.562, 2.554, 2.555, 2.555, 2.560, 2.568, 2.593, 2.631]
    ax = axs[0]
    ax.plot(n_attn, loss, color=BLUE, marker="o", **mk(0))
    ax.set_xlabel("attention layers (of 24)")
    ax.set_ylabel("validation loss (nats)")
    ax.set_title("(a) Waleffe et al. (2024), 130M: loss", pad=4)
    ax.set_xticks([0, 4, 8, 12, 16, 20, 24])
    ax.annotate("all-attention", (24, 2.631), xytext=(17.2, 2.6255), fontsize=7, color=INK2)
    ax.annotate("all-Mamba-2", (0, 2.615), xytext=(1.2, 2.613), fontsize=7, color=INK2)

    ratio_lab = ["pure", "24:1", "12:1", "6:1", "3:1"]
    frac = [0, 1 / 25, 1 / 13, 1 / 7, 1 / 4]
    # Wang et al. 2507.06457, Table 7 (App. E): RULER recall, GDN column and the
    # average over DeltaNet, GDN, GLA and HGRN2 (verified in report/citation_check.md)
    gdn = [0.356, 0.341, 0.352, 0.425, 0.436]
    avg = [0.256, 0.338, 0.340, 0.390, 0.397]
    ax = axs[1]
    xs = [100 * f for f in frac]
    ax.plot(xs, gdn, color=ORANGE, marker="o", label="Gated DeltaNet hybrids", **mk(0))
    ax.plot(xs, avg, color=AQUA, marker="s", ls=(0, (3, 2)), label="mean over 4 linear types", **mk(0))
    for x_, r, l in zip(xs, avg, ratio_lab):
        ax.annotate(l, (x_, r), xytext=(0, -11), textcoords="offset points", ha="center", fontsize=6.8, color=INK2)
    ax.set_ylim(0.2, 0.46)
    ax.set_xlabel("full-attention layers (%)")
    ax.set_ylabel("RULER recall (average)")
    ax.set_title("(b) Wang et al. (2025): recall", pad=4)
    ax.legend(loc="lower right")
    fig.tight_layout(w_pad=2.2)
    save(fig, "fig_prior")


# ------------------------------------------------------------------------------------------
CFG_ORDER = ["T64_kv4", "T64_kv8", "T64_kv16", "T128_kv32", "T256_kv64", "T512_kv128", "T1024_kv256"]
CFG_LAB = ["64/4", "64/8", "64/16", "128/32", "256/64", "512/128", "1024/256"]


def _per_cfg(d):
    pc = d.get("final_per_config_acc") or d.get("per_config") or {}
    return [pc.get(c) for c in CFG_ORDER]


def fig_mqar():
    s1 = {os.path.basename(f)[:-5]: json.load(open(f))
          for f in glob.glob(os.path.join(ROOT, "experiments/s1_zoology/runs/*.json")) if ".smoke" not in f}
    r4 = {os.path.basename(f)[:-5]: json.load(open(f))
          for f in glob.glob(os.path.join(ROOT, "experiments/repro/R4/runs/*.json")) if ".smoke" not in f}
    r5 = json.load(open(os.path.join(ROOT, "experiments/repro/R5/r5_results.json")))["runs"]

    def mean(rows):
        rows = [r for r in rows if all(v is not None for v in r)]
        return [sum(c) / len(c) for c in zip(*rows)]

    zg = {64: mean([_per_cfg(v) for v in r5["gdn_zoology"].values()]),
          128: _per_cfg(s1["gated_delta_net-d128-lr1.0e-03"]),
          256: _per_cfg(s1["gated_delta_net-d256-lr1.0e-03"])}
    og = {64: mean([_per_cfg(v) for v in r5["our_gdn_bf16"].values()]),
          128: _per_cfg(r4["our_gdn-d128-lr1.0e-03"]),
          256: _per_cfg(r4["our_gdn-d256-lr1.0e-03"])}
    att = _per_cfg(s1["attention-d64-lr1.0e-02"])

    fig, ax = plt.subplots(figsize=(TW, 2.05))
    x = list(range(len(CFG_ORDER)))
    widths = {64: SEQ[1], 128: SEQ[4], 256: SEQ[12]}
    ax.plot(x, att, color=ORANGE, marker="D", ms=3.8, **mk(0))
    for d in (256, 128, 64):
        ax.plot(x, zg[d], color=widths[d], marker="o", **mk(0))
        ax.plot(x, og[d], color=widths[d], ls=(0, (3, 2)), marker="s", mfc="white", mew=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(CFG_LAB)
    ax.set_xlabel("MQAR test configuration (sequence length / number of key–value pairs)")
    ax.set_ylabel("recall accuracy")
    ax.set_ylim(0, 1.05)
    for d, dy in ((64, 0.0), (128, -0.05)):
        ax.text(6.12, zg[d][-1] + dy, f"{zg[d][-1]:.2f}", va="center", fontsize=7, color=INK2)
    ax.set_xlim(-0.2, 6.55)
    from matplotlib.lines import Line2D
    h1 = [Line2D([], [], color=ORANGE, marker="D", ms=3.8, mec="white", label="attention, $d=64$")] + [
        Line2D([], [], color=widths[d], marker="o", mec="white", label=f"Gated DeltaNet, $d={d}$") for d in (256, 128, 64)]
    h2 = [Line2D([], [], color=INK2, marker="o", mec="white", label="Zoology reference code"),
          Line2D([], [], color=INK2, ls=(0, (3, 2)), marker="s", mfc="white", label="our implementation, same harness")]
    leg1 = ax.legend(handles=h1, loc="lower left", title="mixer", title_fontsize=7.3, alignment="left")
    ax.add_artist(leg1)
    ax.legend(handles=h2, loc="lower left", bbox_to_anchor=(0.36, 0.0), title="implementation",
              title_fontsize=7.3, alignment="left", handlelength=2.6)
    fig.tight_layout()
    save(fig, "fig_mqar")


# ------------------------------------------------------------------------------------------
def fig_throughput():
    rows = {}
    for f, scale in (("train_step_idle_20261110_D384.json", "D384"), ("train_step_idle_20261110.json", "D512")):
        for r in jsonl(os.path.join(ROOT, "experiments/bench", f))[1:]:
            if r.get("scale") == scale and r.get("status") == "ok":
                rows[(scale, r["config"], r["shape"])] = r["tok_per_s_median"] / 1e3
    fig, axs = plt.subplots(1, 2, figsize=(TW, 2.05), sharey=True)
    cfgs = ["C0", "C3", "C12"]
    for ax, shape, title in zip(axs, ["4k", "32k"], ["(a) Stage A: $2\\times4$K tokens per micro-batch",
                                                   "(b) Stage B: $1\\times32$K tokens, checkpointed"]):
        w = 0.36
        for j, (scale, col) in enumerate((("D384", SEQ[1]), ("D512", SEQ[6]))):
            vals = [rows.get((scale, c, shape), 0) for c in cfgs]
            xs = [i + (j - 0.5) * w * 1.1 for i in range(len(cfgs))]
            bars = ax.bar(xs, vals, width=w, color=col, label=scale, edgecolor="white", linewidth=0.6)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 2.5, f"{v:.0f}", ha="center", fontsize=6.6, color=INK2)
        ax.set_xticks(range(len(cfgs)))
        ax.set_xticklabels(["$k=0$", "$k=3$", "$k=12$"])
        ax.set_title(title, pad=4)
        ax.grid(axis="x", visible=False)
    axs[0].set_ylabel("throughput (K tokens/s)")
    axs[0].set_ylim(0, 160)
    axs[1].legend(loc="upper right", ncol=2)
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig_throughput")


# ------------------------------------------------------------------------------------------
def fig_cost():
    from hybridlc.models.analytic import state_bytes
    from hybridlc.models.config import ModelConfig
    from hybridlc.models.patterns import centred
    rows = jsonl(os.path.join(ROOT, "experiments/bench/cost_session1/rows.jsonl"))
    ks = [0, 1, 3, 6, 12]
    fig, axs = plt.subplots(2, 2, figsize=(TW, 4.0))
    axs = axs.ravel()

    def logx(ax, ticks, labs):
        ax.set_xscale("log", base=2)
        ax.set_xticks(ticks)
        ax.set_xticklabels(labs)
        ax.minorticks_off()

    ax = axs[0]
    Ts = [2 ** i for i in range(9, 16)]
    for k in ks:
        cfg = ModelConfig.for_scale("D384", centred(k))
        ax.plot([t / 1024 for t in Ts], [state_bytes(cfg, t)["total"] / 2 ** 20 for t in Ts],
                color=SEQ[k], marker="o", ms=3.2, label=f"$k={k}$", **mk(0))
    logx(ax, [0.5, 1, 4, 16, 32], ["0.5K", "1K", "4K", "16K", "32K"])
    ax.set_yscale("log")
    ax.set_xlabel("context length")
    ax.set_ylabel("state per sequence (MiB)")
    ax.set_title("(a) Inference state (analytic, exact)", pad=4)

    def series(phase, key_x, k, filt):
        return sorted((r[key_x], r["min_us"]) for r in rows
                      if r.get("phase") == phase and r.get("status") == "ok" and r.get("scale") == "D384"
                      and r.get("k") == k and r.get("config") == f"C{k}" and filt(r))

    ax = axs[1]
    for k in ks:
        p = series("prefill", "T", k, lambda r: r["B"] == 1 and r["mode"] == "graph"
                   and r["backend"] in ("flash", "none") and r["l2"] == "cold")
        if p:
            ax.plot([t / 1024 for t, _ in p], [v / 1e3 for _, v in p], color=SEQ[k], marker="o", ms=3.2, **mk(0))
    logx(ax, [1, 4, 16, 32], ["1K", "4K", "16K", "32K"])
    ax.set_yscale("log")
    ax.set_xlabel("prompt length")
    ax.set_ylabel("prefill latency (ms)")
    ax.set_title("(b) Prefill, batch 1 (measured)", pad=4)

    ax = axs[2]
    for k in ks:
        p = series("decode", "L", k, lambda r: r["B"] == 1 and r["backend"] in ("cudnn", "none") and r["l2"] == "cold")
        if p:
            ax.plot([t / 1024 for t, _ in p], [v / 1e3 for _, v in p], color=SEQ[k], marker="o", ms=3.2, **mk(0))
    logx(ax, [1, 4, 16, 32], ["1K", "4K", "16K", "32K"])
    ax.set_xlabel("context length")
    ax.set_ylabel("decode latency (ms / token)")
    ax.set_title("(c) Decode, batch 1 (measured)", pad=4)

    ax = axs[3]
    bm = defaultdict(dict)
    for r in rows:
        if r.get("phase") == "bmax" and r.get("scale") == "D384" and r.get("B_max_verified"):
            bm[r["config"]][r["L"]] = r["B_max_verified"]
    for k in ks:
        d = bm.get(f"C{k}", {})
        if d:
            Ls = sorted(d)
            ax.plot([l / 1024 for l in Ls], [d[l] for l in Ls], color=SEQ[k], marker="o", ms=3.2, **mk(0))
    logx(ax, [1, 4, 16, 32], ["1K", "4K", "16K", "32K"])
    ax.set_yscale("log")
    ax.set_xlabel("context length")
    ax.set_ylabel("max. concurrent sequences")
    ax.set_title("(d) Largest batch within 8 GiB (verified)", pad=4)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.035), handlelength=2.0,
               title="full-attention layers of 12:", title_fontsize=7.3)
    fig.tight_layout(h_pad=1.2, w_pad=1.6, rect=(0, 0, 1, 0.94))
    save(fig, "fig_cost")
    # print the numbers used in the text, for cross-checking
    for k in (0, 3, 12):
        cfg = ModelConfig.for_scale("D384", centred(k))
        print(f"k={k}: state@32K = {state_bytes(cfg, 32768)['total'] / 2**20:.2f} MiB; "
              f"Bmax@32K = {bm.get(f'C{k}', {}).get(32768)}")


def fig_cost2():
    """Compact two-panel cost figure for the short report: state memory and largest batch."""
    from hybridlc.models.analytic import state_bytes
    from hybridlc.models.config import ModelConfig
    from hybridlc.models.patterns import centred
    rows = jsonl(os.path.join(ROOT, "experiments/bench/cost_session1/rows.jsonl"))
    ks = [0, 1, 3, 6, 12]
    fig, axs = plt.subplots(1, 3, figsize=(TW, 1.85))
    Ts = [2 ** i for i in range(10, 16)]
    tk = ([1, 4, 16, 32], ["1K", "4K", "16K", "32K"])
    ax = axs[0]
    for k in ks:
        cfg = ModelConfig.for_scale("D384", centred(k))
        ax.plot([t / 1024 for t in Ts], [state_bytes(cfg, t)["total"] / 2 ** 20 for t in Ts],
                color=SEQ[k], marker="o", ms=2.6, label=f"$k={k}$", **mk(0))
    ax.set_yscale("log"); ax.set_ylabel("state / sequence (MiB)")
    ax.set_title("(a) inference state (exact)", pad=3)
    ax = axs[1]
    for k in ks:
        p = sorted((r["L"], r["min_us"]) for r in rows if r.get("phase") == "decode" and r.get("status") == "ok"
                   and r.get("scale") == "D384" and r.get("k") == k and r.get("config") == f"C{k}"
                   and r["B"] == 1 and r["backend"] in ("cudnn", "none") and r["l2"] == "cold")
        ax.plot([t / 1024 for t, _ in p], [v / 1e3 for _, v in p], color=SEQ[k], marker="o", ms=2.6, **mk(0))
    ax.set_ylabel("decode (ms / token)")
    ax.set_title("(b) decode latency, batch 1", pad=3)
    ax = axs[2]
    bm = defaultdict(dict)
    for r in rows:
        if r.get("phase") == "bmax" and r.get("scale") == "D384" and r.get("B_max_verified"):
            bm[r["config"]][r["L"]] = r["B_max_verified"]
    for k in ks:
        d = bm.get(f"C{k}", {})
        Ls = sorted(d)
        ax.plot([l / 1024 for l in Ls], [d[l] for l in Ls], color=SEQ[k], marker="o", ms=2.6, **mk(0))
    ax.set_yscale("log"); ax.set_ylabel("sequences in 8 GiB")
    ax.set_title("(c) largest batch (verified)", pad=3)
    for ax in axs:
        ax.set_xscale("log", base=2); ax.set_xticks(tk[0]); ax.set_xticklabels(tk[1]); ax.minorticks_off()
        ax.set_xlabel("context length")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.08), handlelength=1.8)
    fig.tight_layout(w_pad=1.0, rect=(0, 0, 1, 0.93))
    save(fig, "fig_cost2")


if __name__ == "__main__":
    fig_cost2()
    fig_prior()
    fig_mqar()
    fig_throughput()
    fig_cost()

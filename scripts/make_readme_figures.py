"""Generate the README / docs figures from raw project data (no hand-typed numbers except the
clearly-labelled literature values in fig_prior_evidence, which are transcribed from the cited papers).

    python scripts/make_readme_figures.py            # writes docs/figures/*.png (+ .svg)

Inputs: experiments/s1_zoology/runs, experiments/repro/R4|R5, experiments/bench/*.json(l),
src/hybridlc/models/analytic.py (state bytes).
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
from matplotlib.patches import FancyBboxPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
OUT = os.path.join(ROOT, "docs", "figures")
os.makedirs(OUT, exist_ok=True)

# --- palette (dataviz reference palette, light surface) ---------------------------------------
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
# sequential blue ramp for k (magnitude of full attention), light -> dark
SEQ = {0: "#86b6ef", 1: "#6da7ec", 2: "#5598e7", 3: "#3987e5", 4: "#2a78d6", 6: "#1c5cab", 12: "#0d366b"}
FULL_C, LIN_C = "#1c5cab", "#cde2fb"

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True, "lines.linewidth": 2,
    "legend.frameon": False, "font.family": "DejaVu Sans",
})


def save(fig, name):
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def jsonl(path):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


# ------------------------------------------------------------------------------------------------
def fig_architecture():
    fig = plt.figure(figsize=(12, 5.2))
    # left: one block
    ax = fig.add_axes([0.0, 0.0, 0.34, 1.0]); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 14)
    ax.grid(False)

    def box(y, text, color, h=1.1, tc=INK):
        ax.add_patch(FancyBboxPatch((1.4, y), 7.2, h, boxstyle="round,pad=0.05,rounding_size=0.25",
                                    fc=color, ec=INK2, lw=1))
        ax.text(5, y + h / 2, text, ha="center", va="center", fontsize=9.5, color=tc)

    def arrow(y0, y1):
        ax.annotate("", xy=(5, y1), xytext=(5, y0), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.2))

    box(0.3, "token embedding  E  (V = 16,384, tied)", "#f0efec")
    arrow(1.45, 2.3)
    box(2.3, "RMSNorm", "#f0efec")
    arrow(3.45, 4.0)
    box(4.0, "Mixer  a_l ∈ {Full, Linear}", "#fde7dc", h=1.5)
    ax.text(5, 3.65, "", ha="center")
    arrow(5.55, 6.3)
    ax.text(5.35, 5.9, "⊕ residual", ha="left", fontsize=8.5, color=INK2)
    box(6.3, "RMSNorm", "#f0efec")
    arrow(7.45, 8.0)
    box(8.0, "SwiGLU MLP  (d_ff)", "#f0efec")
    arrow(9.15, 9.9)
    ax.text(5.35, 9.5, "⊕ residual", ha="left", fontsize=8.5, color=INK2)
    box(9.9, "final RMSNorm → logits (Eᵀ)", "#f0efec")
    ax.add_patch(FancyBboxPatch((0.9, 2.05), 8.2, 7.35, boxstyle="round,pad=0.05,rounding_size=0.3",
                                fc="none", ec=INK2, lw=1, ls="--"))
    ax.text(9.3, 5.7, "× 12", fontsize=12, fontweight="bold", color=INK2)
    ax.text(5, 12.9, "One hybrid block", ha="center", fontsize=11, fontweight="bold")
    ax.text(5, 12.2, "only the mixer differs between layers", ha="center", fontsize=9, color=INK2)

    # middle: the two mixers
    ax2 = fig.add_axes([0.35, 0.05, 0.27, 0.9]); ax2.axis("off"); ax2.set_xlim(0, 10); ax2.set_ylim(0, 14)
    ax2.grid(False)
    ax2.add_patch(FancyBboxPatch((0.3, 7.6), 9.4, 5.2, boxstyle="round,pad=0.05,rounding_size=0.3",
                                 fc="#e6effb", ec=FULL_C, lw=1.4))
    ax2.text(5, 12.2, "Full mixer  F  (softmax attention)", ha="center", fontweight="bold", color=FULL_C)
    for i, t in enumerate(["Q, K, V projections; 64-dim heads",
                           "per-head QK-RMSNorm, RoPE θ = 5·10⁵",
                           "causal softmax over the document",
                           "headwise sigmoid output gate",
                           "inference state: KV cache, grows with T"]):
        ax2.text(0.8, 11.2 - i * 0.75, "• " + t, fontsize=8.8)
    ax2.add_patch(FancyBboxPatch((0.3, 0.6), 9.4, 6.2, boxstyle="round,pad=0.05,rounding_size=0.3",
                                 fc="#f6f9fe", ec="#6da7ec", lw=1.4))
    ax2.text(5, 6.2, "Linear mixer  Λ  (Gated DeltaNet)", ha="center", fontweight="bold", color="#256abf")
    ax2.text(5, 5.2, r"$S_t = S_{t-1}\,\alpha_t\,(I-\beta_t k_t k_t^{\top}) + \beta_t v_t k_t^{\top}$",
             ha="center", fontsize=10)
    for i, t in enumerate(["short causal conv (width 4) on q, k, v",
                           "decay α_t, write strength β_t (learned)",
                           "output gate, d_k 64 / d_v 128",
                           "inference state: fixed-size matrix S"]):
        ax2.text(0.8, 4.1 - i * 0.75, "• " + t, fontsize=8.8)

    # right: layer patterns
    ax3 = fig.add_axes([0.685, 0.12, 0.31, 0.76])
    ax3.grid(False)
    from hybridlc.models.patterns import centred
    ks = [0, 1, 2, 3, 6, 12]
    for row, k in enumerate(ks):
        pat = centred(k)
        for l, ch in enumerate(pat):
            ax3.add_patch(plt.Rectangle((l + 0.06, row + 0.08), 0.88, 0.84,
                                        fc=FULL_C if ch == "A" else LIN_C, ec=SURF, lw=2))
    ax3.set_xlim(0, 12); ax3.set_ylim(len(ks), 0)
    ax3.set_yticks([i + 0.5 for i in range(len(ks))])
    ax3.set_yticklabels([f"k = {k}" for k in ks])
    ax3.set_xticks([l + 0.5 for l in range(12)]); ax3.set_xticklabels(range(12), fontsize=8)
    ax3.set_xlabel("layer index (input → output)")
    for s in ax3.spines.values():
        s.set_visible(False)
    ax3.tick_params(length=0)
    ax3.set_title("Configurations: k of 12 layers use full attention", loc="left")
    ax3.add_patch(plt.Rectangle((0.1, -0.85), 0.6, 0.45, fc=FULL_C, clip_on=False))
    ax3.text(0.85, -0.6, "Full (F)", va="center", fontsize=9)
    ax3.add_patch(plt.Rectangle((3.6, -0.85), 0.6, 0.45, fc=LIN_C, ec=INK2, lw=0.5, clip_on=False))
    ax3.text(4.35, -0.6, "Linear (Λ, Gated DeltaNet)", va="center", fontsize=9)
    save(fig, "architecture")


# ------------------------------------------------------------------------------------------------
def fig_prior_evidence():
    """Values transcribed from the papers (see reports/literature/*). Waleffe values were read off
    Fig. 4 of arXiv 2406.07887 and are approximate."""
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
    # Waleffe et al. 2024, 130M, 24 layers: validation loss vs # attention layers
    n_attn = [0, 1, 2, 3, 4, 5, 7, 12, 24]
    loss = [2.615, 2.562, 2.554, 2.555, 2.555, 2.560, 2.568, 2.593, 2.631]
    ax = axs[0]
    ax.plot(n_attn, loss, color=BLUE, marker="o", ms=6, mec=SURF, mew=1.5)
    ax.set_xlabel("attention layers (of 24)"); ax.set_ylabel("validation loss (nats)")
    ax.set_title("Loss is U-shaped in the ratio\nWaleffe et al. 2024, 130M (read off Fig. 4)", loc="left")
    ax.annotate("all-linear", (0, 2.615), xytext=(0.6, 2.6165), fontsize=8.5, color=INK2)
    ax.annotate("all-attention is worst", (24, 2.631), xytext=(16.5, 2.6315), fontsize=8.5, color=INK2)
    # Wang et al. 2025, 1.3B, RULER recall average vs full-attention fraction
    frac = [0, 1 / 25, 1 / 13, 1 / 7, 1 / 4]
    lab = ["pure\nlinear", "24:1", "12:1", "6:1", "3:1"]
    recall = [0.256, 0.338, 0.340, 0.390, 0.397]
    ax = axs[1]
    ax.plot([f * 100 for f in frac], recall, color=ORANGE, marker="o", ms=6, mec=SURF, mew=1.5)
    for f, r, l in zip(frac, recall, lab):
        ax.annotate(l, (f * 100, r), xytext=(0, -26), textcoords="offset points", ha="center",
                    fontsize=8, color=INK2)
    ax.set_ylim(0.18, 0.43)
    ax.set_xlabel("full-attention layers (%)"); ax.set_ylabel("RULER recall (avg)")
    ax.set_title("Recall keeps rising while LM metrics stay flat\nWang et al. 2025 (2507.06457), 1.3B, Table 7",
                 loc="left")
    fig.tight_layout()
    save(fig, "prior_evidence")


# ------------------------------------------------------------------------------------------------
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

    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    x = list(range(len(CFG_ORDER)))
    widths = {64: "#86b6ef", 128: "#2a78d6", 256: "#0d366b"}
    # Zoology GDN (reference code)
    zg = {64: mean([_per_cfg(v) for v in r5["gdn_zoology"].values()]),
          128: _per_cfg(s1["gated_delta_net-d128-lr1.0e-03"]),
          256: _per_cfg(s1["gated_delta_net-d256-lr1.0e-03"])}
    og = {64: mean([_per_cfg(v) for v in r5["our_gdn_bf16"].values()]),
          128: _per_cfg(r4["our_gdn-d128-lr1.0e-03"]),
          256: _per_cfg(r4["our_gdn-d256-lr1.0e-03"])}
    for d in (64, 128, 256):
        ax.plot(x, zg[d], color=widths[d], marker="o", ms=7, mec=SURF, mew=1.5, label=f"GDN d={d}, Zoology code")
        ax.plot(x, og[d], color=widths[d], ls="--", marker="s", ms=6, mfc=SURF, mew=1.5,
                label=f"GDN d={d}, our implementation")
    att = _per_cfg(s1["attention-d64-lr1.0e-02"])
    ax.plot(x, att, color=ORANGE, marker="D", ms=6, mec=SURF, mew=1.5, label="attention d=64, Zoology code")
    ax.set_xticks(x); ax.set_xticklabels(CFG_LAB)
    ax.set_xlabel("MQAR test configuration (sequence length / key-value pairs)")
    ax.set_ylabel("recall accuracy")
    ax.set_ylim(0, 1.04)
    ax.set_title("Reproduction: Zoology MQAR recall vs state size (R1, R4, R5)", loc="left")
    ax.text(6.05, zg[64][-1], f"{zg[64][-1]:.2f}", va="center", fontsize=8.5, color=INK2)
    ax.text(6.05, zg[128][-1] - 0.04, f"{zg[128][-1]:.2f}", va="center", fontsize=8.5, color=INK2)
    ax.legend(loc="lower left", fontsize=8.3, ncol=2)
    fig.tight_layout()
    save(fig, "mqar_reproduction")


# ------------------------------------------------------------------------------------------------
def fig_throughput():
    rows = {}
    for f, scale in (("train_step_idle_20261110_D384.json", "D384"), ("train_step_idle_20261110.json", "D512")):
        for r in jsonl(os.path.join(ROOT, "experiments/bench", f))[1:]:
            if r.get("scale") == scale and r.get("status") == "ok":
                rows[(scale, r["config"], r["shape"])] = r["tok_per_s_median"] / 1e3
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    cfgs = ["C0", "C3", "C12"]
    for ax, shape, title in zip(axs, ["4k", "32k"], ["Stage A: 2 × 4K tokens per micro-batch",
                                                   "Stage B: 1 × 32K tokens (activation checkpointing)"]):
        w = 0.36
        for j, (scale, col) in enumerate((("D384", "#86b6ef"), ("D512", "#1c5cab"))):
            vals = [rows.get((scale, c, shape), 0) for c in cfgs]
            xs = [i + (j - 0.5) * w * 1.08 for i in range(len(cfgs))]
            bars = ax.bar(xs, vals, width=w, color=col, label=scale)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{v:.0f}K", ha="center", fontsize=8, color=INK2)
        ax.set_xticks(range(len(cfgs))); ax.set_xticklabels(["k = 0", "k = 3", "k = 12"])
        ax.set_title(title, loc="left", fontsize=10)
        ax.grid(axis="x", visible=False)
    axs[0].set_ylabel("training throughput (K tokens/s)")
    axs[0].set_ylim(0, 160); axs[0].legend(loc="upper left", ncol=2)
    fig.suptitle("Measured training throughput on the RTX 5080 (idle GPU, median of 20 steps)",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    save(fig, "training_throughput")


# ------------------------------------------------------------------------------------------------
def fig_cost():
    from hybridlc.models.analytic import state_bytes
    from hybridlc.models.config import ModelConfig
    from hybridlc.models.patterns import centred
    rows = jsonl(os.path.join(ROOT, "experiments/bench/cost_session1/rows.jsonl"))
    ks = [0, 1, 3, 6, 12]
    fig, axs = plt.subplots(1, 4, figsize=(16, 3.9))

    ax = axs[0]
    Ts = [2 ** i for i in range(9, 16)]
    for k in ks:
        cfg = ModelConfig.for_scale("D384", centred(k))
        ax.plot([t / 1024 for t in Ts], [state_bytes(cfg, t)["total"] / 2 ** 20 for t in Ts],
                color=SEQ[k], marker="o", ms=4, label=f"k = {k}")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xticks([0.5, 1, 4, 16, 32]); ax.set_xticklabels(["0.5K", "1K", "4K", "16K", "32K"])
    ax.set_xlabel("context length"); ax.set_ylabel("inference state per sequence (MiB)")
    ax.set_title("Inference memory (analytic, exact)", loc="left")
    ax.legend(fontsize=8)

    def series(phase, key_x, k, filt):
        pts = sorted((r[key_x], r["min_us"]) for r in rows
                     if r.get("phase") == phase and r.get("status") == "ok" and r.get("scale") == "D384"
                     and r.get("k") == k and r.get("config") == f"C{k}" and filt(r))
        return pts

    ax = axs[1]
    for k in ks:
        p = series("prefill", "T", k, lambda r: r["B"] == 1 and r["mode"] == "graph" and r["backend"] in ("flash", "none")
                   and r["l2"] == "cold")
        if p:
            ax.plot([t / 1024 for t, _ in p], [v / 1e3 for _, v in p], color=SEQ[k], marker="o", ms=4, label=f"k = {k}")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xticks([1, 4, 16, 32]); ax.set_xticklabels(["1K", "4K", "16K", "32K"])
    ax.set_xlabel("prompt length"); ax.set_ylabel("prefill latency (ms)")
    ax.set_title("Prefill, batch 1 (measured)", loc="left")

    ax = axs[2]
    for k in ks:
        p = series("decode", "L", k, lambda r: r["B"] == 1 and r["backend"] in ("cudnn", "none") and r["l2"] == "cold")
        if p:
            ax.plot([t / 1024 for t, _ in p], [v / 1e3 for _, v in p], color=SEQ[k], marker="o", ms=4, label=f"k = {k}")
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 4, 16, 32]); ax.set_xticklabels(["1K", "4K", "16K", "32K"])
    ax.set_xlabel("context length"); ax.set_ylabel("decode latency per token (ms)")
    ax.set_title("Decode, batch 1 (measured)", loc="left")

    ax = axs[3]
    bm = defaultdict(dict)
    for r in rows:
        if r.get("phase") == "bmax" and r.get("scale") == "D384" and r.get("B_max_verified"):
            bm[r["config"]][r["L"]] = r["B_max_verified"]
    for k in ks:
        d = bm.get(f"C{k}", {})
        if d:
            Ls = sorted(d)
            ax.plot([l / 1024 for l in Ls], [d[l] for l in Ls], color=SEQ[k], marker="o", ms=4, label=f"k = {k}")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xticks([1, 4, 16, 32]); ax.set_xticklabels(["1K", "4K", "16K", "32K"])
    ax.set_xlabel("context length"); ax.set_ylabel("max concurrent sequences")
    ax.set_title("Largest batch in 8 GiB (verified)", loc="left")

    fig.suptitle("Inference cost vs number of full-attention layers k (D384, untrained weights; "
                 "session 1, min of ≥30 CUDA-graph replays, cold L2)", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    save(fig, "inference_cost")


if __name__ == "__main__":
    fig_architecture()
    fig_prior_evidence()
    fig_mqar()
    fig_throughput()
    fig_cost()

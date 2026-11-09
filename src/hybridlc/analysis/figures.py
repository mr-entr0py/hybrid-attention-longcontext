"""Protocol figures P1–P18 (protocol §12), built only from ``experiments/results/**`` and the registry.

Each ``pXX(ctx)`` draws one figure and writes it with :func:`.plotstyle.savefig` (PDF + PNG +
input-hash JSON). ``plots/make_pXX.py`` are thin command-line wrappers. When a figure's inputs do not
exist yet, a labelled empty-state figure is written instead (so the templates can be exercised before
the experiments run), and ``meta["status"] == "placeholder"``.

Input conventions beyond protocol §9.8 (documented in INTERFACES.md, analysis section):

* efficiency benchmarks (P8, P9): ``experiments/results/_bench/*.jsonl`` rows with
  ``config, k, kind ('prefill'|'decode'), B, T, l2 ('cold'|'hot'), min_ms, analytic_ms`` and, for
  P9, rows with ``kind = 'bmax', T, b_max``;
* Tier-0 LM-1 / LM-3 (P17): suite ``Tier0`` files ``LM-1_b<lo>_<T>.jsonl`` / ``LM-3_d<lo>_<T>.jsonl``
  with a per-row ``loss`` field (mean token loss of that bucket for one document);
* LM-4 (P4 inset): ``LM-4_g<gap>_<T>.jsonl`` rows with ``copy_gain``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import protocol as P
from .pipeline import AnalysisSpec, run_analysis
from .plotstyle import (FAMILY_COLOR, FAMILY_LABEL, FAMILY_MARKER, INK, INK_MUTED, k_color, log2_length_axis,
                        new_figure, placeholder, plot_seeds, savefig, shade_extrapolation)
from .results import load_item_results, load_registry
from .scores import CellTable, family_spec
from .stats import isotonic_recovery, recovery

CI_T = "95% CI: paired t across seeds (df = n - 1)"
CI_BOOT = "95% CI: hierarchical paired percentile bootstrap (B = 10,000)"


@dataclass
class FigureContext:
    results_root: Path = Path("experiments/results")
    registry_path: Path = Path("experiments/registry.jsonl")
    out_dir: Path = Path("plots/out")
    spec: AnalysisSpec = field(default_factory=AnalysisSpec)
    l_ext: int | None = None
    _df: pd.DataFrame | None = None
    _res: dict | None = None

    @property
    def df(self) -> pd.DataFrame:
        if self._df is None:
            reg = load_registry(self.registry_path)
            self._df = load_item_results(self.results_root, reg, extra_fields=("loss", "copy_gain", "filler_source",
                                                                               "distance"))
        return self._df

    @property
    def inputs(self) -> list[str]:
        files = list(self.df.attrs.get("inputs", []))
        if Path(self.registry_path).is_file():
            files.append(str(self.registry_path))
        return files

    @property
    def res(self) -> dict | None:
        if self._res is None and len(self.df):
            try:
                self._res = run_analysis(self.df, self.spec)
            except ValueError:
                self._res = None
        return self._res

    def table(self, suite: str = "CS", branch: str = "main", ckpt: str | None = None) -> CellTable | None:
        try:
            return CellTable.from_rows(self.df, suite, "em_tf", branch=branch, ckpt=ckpt or self.spec.main_ckpt)
        except (ValueError, KeyError):
            return None

    def save(self, fig, name: str, caption: str, ci: str = "", inputs: list[str] | None = None, **extra) -> dict:
        meta = savefig(fig, name, self.inputs if inputs is None else inputs, out_dir=self.out_dir, caption=caption,
                       ci_type=ci, extra=extra or None, code_file=__file__)
        return meta


def _empty(ctx: FigureContext, name: str, title: str, why: str) -> dict:
    fig, ax = new_figure()
    placeholder(ax[0, 0], f"{title}\n\n{why}")
    return ctx.save(fig, name, caption=f"{title} (no inputs yet: {why})", status="placeholder")


def _kx(configs: list[str]) -> list[str]:
    return sorted([c for c in configs if c in P.CONFIG_K], key=lambda c: P.CONFIG_K[c])


def _seed_series(tab: pd.DataFrame, cfg: str) -> pd.Series:
    return tab.loc[cfg].dropna() if cfg in tab.index else pd.Series(dtype=float)


# =============================================================================================
def p01(ctx: FigureContext) -> dict:
    """S_ret(k) at the primary cell, all seeds, R = 0.8 line and isotonic fit (H1, H2)."""
    r = ctx.res
    if r is None:
        return _empty(ctx, "P1", "P1 S_ret(k) at the primary cell", "no CS results")
    ret = pd.DataFrame(r["scores"]["retrieval"]).T
    ret.index = ret.index.astype(str)
    cfgs = _kx(list(ret.index))
    fig, ax = new_figure(width=3.6)
    a = ax[0, 0]
    ks = np.array([P.CONFIG_K[c] for c in cfgs])
    for c in cfgs:
        v = ret.loc[c].dropna().astype(float)
        a.scatter(np.full(len(v), P.CONFIG_K[c]), v, s=14, color=k_color(P.CONFIG_K[c]), alpha=0.7, zorder=3,
                  edgecolors="white", linewidths=0.5)
    means = np.array([ret.loc[c].astype(float).mean() for c in cfgs])
    if {"C0", "C12"} <= set(cfgs):
        n_s = np.array([ret.loc[c].notna().sum() for c in cfgs], float)
        r_iso = isotonic_recovery(ks, means, n_s)
        m0, m12 = means[list(ks).index(0)], means[list(ks).index(12)]
        a.plot(ks, m0 + r_iso * (m12 - m0), color=INK, lw=1.5, label="isotonic fit", zorder=2)
        a.axhline(m0 + P.R_TARGET * (m12 - m0), color=INK_MUTED, ls="--", lw=1, label="R = 0.8")
    a.scatter(ks, means, marker="_", s=200, color=INK, zorder=4, label="seed mean")
    a.set_xscale("symlog", linthresh=1)
    a.set_xticks(ks, [str(k) for k in ks])
    a.set_xlabel("k (F_g layers of 12)")
    a.set_ylabel("Retrieval score S_ret (EM_tf)")
    Tp = "+".join(f"{T // 1024}K" for T in r["primary"]["lengths"])
    a.set_title(f"Retrieval vs k, primary cell {Tp}")
    a.legend(loc="lower right")
    return ctx.save(fig, "P1", caption=f"Per-seed retrieval ladder mean at the primary cell ({Tp}); every seed shown.",
                    ci="none (all seeds shown)")


def p02(ctx: FigureContext) -> dict:
    """Forest plot of per-seed L1 and L3, paired CIs, sensitivity CIs and success counts (H1, H1-min)."""
    r = ctx.res
    if r is None:
        return _empty(ctx, "P2", "P2 L_k forest plot", "no CS results")
    fig, ax = new_figure(width=5.0, height=2.8)
    a = ax[0, 0]
    y = 0
    yt, yl = [], []
    for name, k in (("H1-min", 1), ("H1", 3)):
        t = r["tests"][name]
        if not t.get("evaluable"):
            continue
        ps = t["per_seed"]
        a.scatter(list(ps.values()), [y + 0.25] * len(ps), s=14, color=k_color(k), alpha=0.8, zorder=3)
        lo = t["estimate"] - (t["estimate"] - t["lower95"]) * 1.0
        a.errorbar(t["estimate"], y, xerr=[[t["estimate"] - lo], [t["upper95"] - t["estimate"]]], fmt="o",
                   color=k_color(k), capsize=3, lw=1.5, label=None)
        sens = r["sensitivity"].get(name, {})
        if "pooled" in sens:
            pr = sens["pooled"]
            a.errorbar(pr["estimate"], y - 0.25, xerr=[[pr["estimate"] - pr["lower95"]], [pr["upper95"] - pr["estimate"]]],
                       fmt="s", color=INK_MUTED, capsize=2, lw=1, ms=4)
        if "bootstrap" in sens:
            br = sens["bootstrap"]
            a.errorbar(br["mean"], y - 0.45, xerr=[[br["mean"] - br["lower95_one_sided"]],
                                                  [br["upper95_one_sided"] - br["mean"]]],
                       fmt="D", color=INK_MUTED, capsize=2, lw=1, ms=3)
        sc = r.get("success_counts", {}).get(f"C{k}")
        lab = f"L{k} (k={k})"
        if sc:
            lab += f"\n{sc['successes']}/{sc['n']} seeds > midpoint"
        yt.append(y)
        yl.append(lab)
        y += 1.4
    a.axvline(0, color=INK, lw=1)
    a.set_yticks(yt, yl)
    a.set_xlabel("L_k = S_k − 0.2·S_0 − 0.8·S_12  (> 0 ⇔ R(k) ≥ 0.8)")
    from matplotlib.lines import Line2D
    a.legend(handles=[Line2D([], [], marker="o", ls="none", color=INK_MUTED, ms=3, label="seed"),
                      Line2D([], [], marker="o", ls="-", color=INK_MUTED, label="paired t"),
                      Line2D([], [], marker="s", ls="-", color=INK_MUTED, ms=4, label="pooled model"),
                      Line2D([], [], marker="D", ls="-", color=INK_MUTED, ms=3, label="bootstrap")],
             loc="center left", bbox_to_anchor=(1.0, 0.5))
    a.set_title("H1 / H1-min (one-sided 95% bounds)")
    return ctx.save(fig, "P2", caption="Per-seed L_k with one-sided 90% (i.e. one-sided 95% bounds) intervals.",
                    ci=f"{CI_T}; pooled-variance model; {CI_BOOT} (one-sided 95% bounds shown)")


def p03(ctx: FigureContext) -> dict:
    """R_f(k) for the three families on one probability-scale axis; per-seed D3 (H3)."""
    r = ctx.res
    if r is None:
        return _empty(ctx, "P3", "P3 recovery by family", "no CS results")
    fig, ax = new_figure(ncols=2, width=3.2)
    a, b = ax[0, 0], ax[0, 1]
    for fam, tab in r["scores"].items():
        df = pd.DataFrame(tab).T.astype(float)
        cfgs = _kx(list(df.index))
        if not {"C0", "C12"} <= set(cfgs):
            continue
        per_seed = {}
        for s in df.columns:
            col = df[s]
            if col[["C0", "C12"]].notna().all():
                per_seed[s] = [recovery(col[c], col["C0"], col["C12"]) for c in cfgs]
        plot_seeds(a, [P.CONFIG_K[c] for c in cfgs], per_seed, color=FAMILY_COLOR[fam], marker=FAMILY_MARKER[fam],
                   label=FAMILY_LABEL[fam])
    a.axhline(P.R_TARGET, color=INK_MUTED, ls="--", lw=1)
    a.set_xscale("symlog", linthresh=1)
    kt = sorted({P.CONFIG_K[c] for fam_tab in r["scores"].values() for c in fam_tab if c in P.CONFIG_K})
    a.set_xticks(kt, [str(k) for k in kt])
    a.set_xlabel("k (F_g layers of 12)")
    a.set_ylabel("Recovery R_f(k)")
    a.legend(loc="upper left")
    h3 = r["tests"]["H3"]
    if h3.get("per_seed"):
        v = list(h3["per_seed"].values())
        b.scatter(np.zeros(len(v)), v, color=INK, s=16, alpha=0.7)
        b.axhline(0, color=INK_MUTED, lw=1)
        b.errorbar(0.3, h3["estimate"], yerr=[[h3["estimate"] - h3["lower95"]], [0]], fmt="o", color=INK, capsize=3)
        b.set_xlim(-0.5, 0.8)
        b.set_xticks([])
        b.set_ylabel("D3 = R_ret(3) - R_hop(3)")
        b.set_title("Per-seed D3" + ("" if h3.get("evaluable") else " (not evaluable)"))
    return ctx.save(fig, "P3", caption="Normalised recovery per family at the primary cell; every seed drawn.",
                    ci=CI_T + " (one-sided lower bound for D3)")


def p04(ctx: FigureContext) -> dict:
    """Delta0(T) vs log2 T on the fixed level set, per-seed slopes, G-REF status per T (H5a)."""
    cs = ctx.table("CS")
    r = ctx.res
    if cs is None or r is None or not {"C0", "C12"} <= set(cs.configs):
        return _empty(ctx, "P4", "P4 length dependence", "no CS results for C0 and C12")
    Ts = [T for T in P.ALL_LENGTHS if T in cs.lengths()]
    per_seed: dict[int, list[float]] = {}
    for T in Ts:
        sc = cs.scores(cs.ladder_weights(family_spec("retrieval", P.H5_FIXED_LEVELS["retrieval"]), [T]))
        for s in sc.columns:
            if pd.notna(sc.loc["C12", s]) and pd.notna(sc.loc["C0", s]):
                per_seed.setdefault(int(s), []).append(float(sc.loc["C12", s] - sc.loc["C0", s]))
    per_seed = {s: v for s, v in per_seed.items() if len(v) == len(Ts)}
    fig, ax = new_figure(width=3.8)
    a = ax[0, 0]
    plot_seeds(a, Ts, per_seed, color=FAMILY_COLOR["retrieval"], label="Δ0(T), seed mean")
    for T in Ts:
        st = r["gref"].get(str(T), {}).get("passes")
        a.text(T, 0.03, "G-REF ✓" if st else "G-REF ×", transform=a.get_xaxis_transform(), ha="center",
               va="bottom", fontsize=6, color=INK if st else INK_MUTED)
    log2_length_axis(a, Ts)
    shade_extrapolation(a, ctx.l_ext)
    a.set_ylabel("Δ0(T) = S12 - S0 (fixed level set)")
    h5 = r["tests"]["H5a"]
    a.set_title(f"H5(a): slope {h5.get('estimate', float('nan')):+.3f} per doubling")
    a.legend(loc="upper left")
    return ctx.save(fig, "P4", caption="Gap between k = 12 and k = 0 on R1 D∈{3,15}, R-load N∈{8,32}; G-REF per T marked.",
                    ci="none (all seeds shown); slope test: " + CI_T)


def p05(ctx: FigureContext) -> dict:
    """Retrieval vs multi-hop length slopes (H5b)."""
    r = ctx.res
    sec = (r or {}).get("secondary", {}).get("H5b", {})
    if not sec.get("evaluable"):
        return _empty(ctx, "P5", "P5 H5(b) slopes", "H5(b) not evaluable")
    h5 = r["tests"]["H5a"]["per_seed"]
    hop = sec["b_hop"]
    fig, ax = new_figure(width=3.0)
    a = ax[0, 0]
    for s in hop:
        a.plot([0, 1], [h5[s], hop[s]], color=INK_MUTED, lw=0.8, marker="o", ms=3, alpha=0.6)
    a.plot([0, 1], [np.mean(list(h5.values())), np.mean(list(hop.values()))], color=INK, lw=2, marker="o")
    a.set_xticks([0, 1], ["Retrieval", "Multi-hop"])
    a.axhline(0, color=INK_MUTED, lw=0.8)
    a.set_ylabel("Slope of Δ0 per doubling of T")
    a.set_title(f"H5(b): b_ret - b_hop = {sec['estimate']:+.3f}")
    return ctx.save(fig, "P5", caption="Per-seed length slopes, retrieval vs multi-hop (h ∈ {1, 2}).", ci=CI_T)


def p06(ctx: FigureContext) -> dict:
    """Early / middle / late at k*, per-seed lines, c1 and c2 (H4)."""
    r = ctx.res
    h4 = (r or {}).get("secondary", {}).get("H4", {})
    if not h4.get("evaluable"):
        return _empty(ctx, "P6", "P6 placement at k*", "placement runs missing")
    ks = h4["kstar"]
    ret = pd.DataFrame(r["scores"]["retrieval"]).T.astype(float)
    arms = [f"P{ks}-early", f"C{ks}", f"P{ks}-late"]
    fig, ax = new_figure(width=3.2)
    a = ax[0, 0]
    t = ret.loc[arms].dropna(axis=1)
    plot_seeds(a, [0, 1, 2], {s: t[s].tolist() for s in t.columns}, color=k_color(ks), label="seed mean")
    a.set_xticks([0, 1, 2], ["early", "middle", "late"])
    a.set_ylabel("Retrieval score (primary cell)")
    a.set_title(f"k* = {ks}: c1 {h4['c1']['estimate']:+.3f}, c2 {h4['c2']['estimate']:+.3f}")
    return ctx.save(fig, "P6", caption="Placement arms at k*; lines join a seed's three runs.", ci=CI_T + " (two-sided)")


def p07(ctx: FigureContext) -> dict:
    """Primary-cell score vs analytic state bytes at 32K (log x), router line, Pareto front, GDN-E (H6, H6′)."""
    r = ctx.res
    if r is None:
        return _empty(ctx, "P7", "P7 score vs state bytes", "no CS results")
    try:
        from hybridlc.models.analytic import state_bytes  # type: ignore
        from hybridlc.models.config import ModelConfig  # type: ignore
        from hybridlc.models.patterns import named_config  # type: ignore
    except ImportError:
        return _empty(ctx, "P7", "P7 score vs state bytes", "hybridlc.models.analytic not available yet")
    ret = pd.DataFrame(r["scores"]["retrieval"]).T.astype(float)
    scale = ctx.spec.scale or "D384"
    pts = []
    for c in ret.index:
        try:
            pat, ov = named_config(c)
            sb = state_bytes(ModelConfig.for_scale(scale, pat, **ov), 32768)
            sb = sb["total"] if isinstance(sb, dict) else float(sb)
        except Exception:  # noqa: BLE001 - analytic module contract may still be in flux
            continue
        pts.append((c, sb, ret.loc[c].dropna().tolist()))
    if not pts:
        return _empty(ctx, "P7", "P7 score vs state bytes", "no analytic state sizes")
    fig, ax = new_figure(width=3.8)
    a = ax[0, 0]
    for c, sb, v in pts:
        col = k_color(P.CONFIG_K.get(c, 0)) if c in P.CONFIG_K else INK_MUTED
        a.scatter([sb] * len(v), v, s=12, color=col, alpha=0.6)
        a.scatter([sb], [np.mean(v)], s=40, color=col, edgecolors="white", zorder=3,
                  marker="o" if c in P.CONFIG_K else "D")
        a.annotate(c, (sb, np.mean(v)), xytext=(4, 2), textcoords="offset points", fontsize=7, color=INK_MUTED)
    d = {c: (sb, np.mean(v)) for c, sb, v in pts}
    if {"C0", "C12"} <= set(d):
        a.plot([d["C0"][0], d["C12"][0]], [d["C0"][1], d["C12"][1]], ls=":", color=INK_MUTED, lw=1,
               label="router line (additive cost)")
    comp = sorted([d[c] for c in d if c in P.CONFIG_K])
    front, best = [], -np.inf
    for x, y in comp:
        if y > best:
            front.append((x, y))
            best = y
    if front:
        a.plot(*zip(*front), color=INK, lw=1.2, label="Pareto front")
    a.set_xscale("log")
    a.set_xlabel("Inference state at 32K (bytes, analytic, log)")
    a.set_ylabel("Retrieval score (primary cell)")
    a.legend(loc="lower right")
    return ctx.save(fig, "P7", caption="Capability vs analytic state memory; diamonds are controls (GDN-E).",
                    ci="none (all seeds shown)")


def _bench(ctx: FigureContext) -> pd.DataFrame | None:
    d = Path(ctx.results_root) / "_bench"
    files = sorted(d.glob("*.jsonl")) if d.exists() else []
    if not files:
        return None
    rows = [json.loads(line) for f in files for line in f.read_text().splitlines() if line.strip()]
    df = pd.DataFrame(rows)
    df.attrs["inputs"] = [str(f) for f in files]
    return df


def p08(ctx: FigureContext) -> dict:
    """Measured vs analytic prefill and decode latency vs T per k (cold/hot L2), roofline fractions."""
    b = _bench(ctx)
    if b is None or not {"kind", "T", "min_ms", "k"} <= set(b.columns):
        return _empty(ctx, "P8", "P8 measured vs analytic latency", "no experiments/results/_bench/*.jsonl")
    fig, ax = new_figure(ncols=2, width=3.3)
    for a, kind in zip(ax[0], ("prefill", "decode")):
        sub = b[b["kind"] == kind]
        for k, g in sub.groupby("k"):
            for l2, ls in (("cold", "-"), ("hot", "--")):
                gg = g[g.get("l2", "cold") == l2] if "l2" in g else g
                gg = gg.groupby("T")["min_ms"].min()
                if len(gg):
                    a.plot(gg.index, gg.values, ls=ls, color=k_color(k), marker="o", ms=3,
                           label=f"k={k} {l2}" if l2 == "cold" else None)
            if "analytic_ms" in g:
                an = g.groupby("T")["analytic_ms"].min()
                a.plot(an.index, an.values, ls=":", color=k_color(k), lw=1)
        a.set_yscale("log")
        log2_length_axis(a, sorted(sub["T"].unique()) or [1024])
        a.set_ylabel(f"{kind} latency, min of N (ms)")
        a.set_title(f"{kind}: solid cold L2, dashed hot, dotted roofline")
    ax[0, 0].legend(fontsize=6)
    return ctx.save(fig, "P8", caption="Min-of-N latency (fla Triton + SDPA on sm_120, this implementation).",
                    ci="none (min of N >= 30 replays)", inputs=b.attrs["inputs"])


def p09(ctx: FigureContext) -> dict:
    """B_max(T) within 8 GiB per k."""
    b = _bench(ctx)
    if b is None or "b_max" not in b.columns:
        return _empty(ctx, "P9", "P9 B_max(T) within 8 GiB", "no b_max rows in experiments/results/_bench")
    sub = b[b["kind"] == "bmax"]
    fig, ax = new_figure()
    a = ax[0, 0]
    for k, g in sub.groupby("k"):
        g = g.sort_values("T")
        a.plot(g["T"], g["b_max"], color=k_color(k), marker="o", label=f"k={k}")
    log2_length_axis(a, sorted(sub["T"].unique()))
    a.set_yscale("log", base=2)
    a.set_ylabel("Max batch within 8 GiB")
    a.legend()
    return ctx.save(fig, "P9", caption="Largest decode batch within an 8 GiB state + activation budget.",
                    inputs=b.attrs["inputs"])


def p10(ctx: FigureContext) -> dict:
    """Per-level R (H1-robust) and N50 vs k at 16K and 32K, levels as N/N50_0."""
    r = ctx.res
    if r is None:
        return _empty(ctx, "P10", "P10 per-level recovery and N50", "no CS results")
    fig, ax = new_figure(ncols=2, width=3.3)
    a, b = ax[0, 0], ax[0, 1]
    for name, k in (("H1-min", 1), ("H1", 3)):
        lv = r["secondary"]["per_level"].get(name, {}).get("levels", {})
        if lv:
            keys = list(lv)
            a.plot(range(len(keys)), [lv[x]["R"] for x in keys], marker="o", color=k_color(k), label=f"k={k}")
            a.set_xticks(range(len(keys)), keys, rotation=60, fontsize=6)
    a.axhline(P.R_TARGET, color=INK_MUTED, ls="--", lw=1)
    a.set_ylabel("R(k) at ladder level")
    a.legend()
    n50 = r["descriptive"].get("N50", {})
    for T, ls in (("16384", "-"), ("32768", "--")):
        tab = n50.get(T, {})
        cfgs = _kx(list(tab))
        if not cfgs:
            continue
        def med(c: str) -> tuple[float, bool]:
            # censored fits are never extrapolated: shown at the tested boundary with an open marker
            vals, cens = [], False
            for v in tab[c].values():
                if v["censoring"] == "none":
                    vals.append(v["n50"])
                elif v["censoring"] in ("left", "right"):
                    vals.append(min(v["levels"]) if v["censoring"] == "left" else max(v["levels"]))
                    cens = True
            return (float(np.median(vals)) if vals else np.nan), cens

        n0 = med("C0")[0] if "C0" in tab else np.nan
        pts = [(P.CONFIG_K[c], *med(c)) for c in cfgs]
        xs = [p_[0] for p_ in pts]
        ys = np.array([p_[1] for p_ in pts]) / (n0 if np.isfinite(n0) else 1)
        b.plot(xs, ys, ls=ls, color=INK, lw=1.2, label=f"T={int(T) // 1024}K")
        for (x, _, cens), y in zip(pts, ys):
            b.plot([x], [y], marker="o", ms=5, color=INK, mfc="white" if cens else INK)
    b.set_xscale("symlog", linthresh=1)
    b.set_xticks([0, 1, 2, 3, 6, 12], ["0", "1", "2", "3", "6", "12"])
    b.set_yscale("log", base=2)
    b.set_xlabel("k")
    b.set_ylabel("N50 / N50₀ (seed median)")
    b.set_title("open marker = censored fit", fontsize=7)
    b.legend()
    return ctx.save(fig, "P10", caption="Recovery per ladder level and logistic N50 capacity relative to k = 0.",
                    ci="point estimates (censoring reported in results JSON)")


def p11(ctx: FigureContext) -> dict:
    """G-REF panel: S12(T)/S12(C0) per T and seed; reference inversion."""
    r = ctx.res
    if r is None or not r.get("gref"):
        return _empty(ctx, "P11", "P11 G-REF panel", "no G-REF data")
    fig, ax = new_figure(width=3.6)
    a = ax[0, 0]
    Ts = sorted(int(T) for T in r["gref"])
    per_seed: dict[str, list[float]] = {}
    for T in Ts:
        for s, v in r["gref"][str(T)]["per_seed_ratio"].items():
            per_seed.setdefault(s, []).append(v)
    per_seed = {s: v for s, v in per_seed.items() if len(v) == len(Ts)}
    plot_seeds(a, Ts, per_seed, color=k_color(12), label="k = 12, seed mean")
    a.axhline(P.GREF_ADEQUACY, color=INK_MUTED, ls="--", lw=1, label="adequacy 0.80")
    log2_length_axis(a, Ts)
    a.set_ylabel("S12(T) / S12(C0)")
    inv = [T for T, v in r.get("reference_inversion", {}).items() if v.get("triggered")]
    a.set_title("G-REF adequacy" + (f"; reference inversion at {inv}" if inv else ""))
    a.legend(loc="lower left")
    return ctx.save(fig, "P11", caption="Reference adequacy of k = 12 per length and seed (DG at 16K/32K, CS elsewhere).",
                    ci="none (all seeds shown)")


def p12(ctx: FigureContext) -> dict:
    """TEF dose ladder (250 -> 4,000) per anchor; gap vs steps."""
    df = ctx.df
    if not len(df) or ctx.spec.dc_suite not in set(df["suite"]):
        return _empty(ctx, "P12", "P12 TEF dose ladder", "no DC results")
    d = df[(df["suite"] == ctx.spec.dc_suite) & df["config"].isin(P.ANCHOR_CONFIGS)
           & df["task"].isin(P.FAMILIES["retrieval"])]
    d = d.assign(step=d["ckpt"].str.extract(r"(\d+)$")[0].astype(float)).dropna(subset=["step"])
    if d.empty:
        return _empty(ctx, "P12", "P12 TEF dose ladder", "no tef<steps> checkpoints in DC")
    g = d.groupby(["config", "seed", "step"])["em_tf"].mean().reset_index()
    fig, ax = new_figure(width=3.6)
    a = ax[0, 0]
    for c in [c for c in P.ANCHOR_CONFIGS if c in set(g["config"])]:
        sub = g[g["config"] == c]
        per_seed = {}
        steps = sorted(sub["step"].unique())
        for s, gg in sub.groupby("seed"):
            gg = gg.set_index("step").reindex(steps)["em_tf"]
            per_seed[s] = gg.to_numpy()
        plot_seeds(a, steps, per_seed, color=k_color(P.CONFIG_K[c]), label=c)
    a.set_xscale("log", base=2)
    a.set_xlabel("TEF steps")
    a.set_ylabel("Retrieval EM (DC items, raw mean)")
    a.legend()
    return ctx.save(fig, "P12", caption="Dose ladder on DC items per anchor; seed lines where checkpoints exist.",
                    ci="none (all seeds shown)")


def _branch_pair(ctx: FigureContext, name: str, title: str, branch: str, cfgs: list[str]) -> dict:
    r = ctx.res
    if r is None:
        return _empty(ctx, name, title, "no CS results")
    Tp = r["primary"]["lengths"]
    t2 = ctx.table("CS", branch=branch)
    t1 = ctx.table("CS")
    if t2 is None or t1 is None:
        return _empty(ctx, name, title, f"no '{branch}' branch results")
    s2 = t2.scores(t2.ladder_weights(family_spec("retrieval"), [T for T in Tp if T in t2.lengths()]))
    s1 = t1.scores(t1.ladder_weights(family_spec("retrieval"), Tp))
    fig, ax = new_figure(width=3.2)
    a = ax[0, 0]
    for i, c in enumerate([c for c in cfgs if c in s2.index]):
        for s in s2.columns:
            if pd.notna(s2.loc[c, s]) and c in s1.index and pd.notna(s1.loc[c, s]):
                a.plot([i - 0.15, i + 0.15], [s1.loc[c, s], s2.loc[c, s]], color=k_color(P.CONFIG_K.get(c, 0)),
                       marker="o", lw=1.2)
    a.set_xticks(range(len(cfgs)), cfgs)
    a.set_ylabel("Retrieval score (primary cell)")
    a.set_title(f"{title}: left 1x, right {branch}")
    return ctx.save(fig, name, caption=f"{title}; each line is one seed.", ci="none")


def p13(ctx: FigureContext) -> dict:
    """2x vs 1x Stage B at k in {0, 3, 12}."""
    return _branch_pair(ctx, "P13", "P13 2x vs 1x Stage B", ctx.spec.saturation_branch, ["C0", "C3", "C12"])


def _control_pair(ctx: FigureContext, name: str, title: str, pairs: list[tuple[str, str]]) -> dict:
    r = ctx.res
    if r is None:
        return _empty(ctx, name, title, "no CS results")
    ret = pd.DataFrame(r["scores"]["retrieval"]).T.astype(float)
    pairs = [(a_, b_) for a_, b_ in pairs if a_ in ret.index and b_ in ret.index]
    if not pairs:
        return _empty(ctx, name, title, "control runs missing")
    fig, ax = new_figure(width=3.2)
    a = ax[0, 0]
    for i, (base, ctl) in enumerate(pairs):
        for s in ret.columns:
            if pd.notna(ret.loc[base, s]) and pd.notna(ret.loc[ctl, s]):
                a.plot([i - 0.15, i + 0.15], [ret.loc[base, s], ret.loc[ctl, s]], marker="o", color=INK_MUTED, lw=1)
    a.set_xticks(range(len(pairs)), [f"{b_} vs {c_}" for b_, c_ in pairs], fontsize=7)
    a.set_ylabel("Retrieval score (primary cell)")
    a.set_title(title)
    return ctx.save(fig, name, caption=f"{title}; each line is one seed.", ci="none")


def p14(ctx: FigureContext) -> dict:
    """F_gc vs F_g at k = 12 (and k in {1, 3} if triggered)."""
    return _control_pair(ctx, "P14", "P14 F_gc vs F_g", [("C12", "X-gc12"), ("C1", "X-gc1"), ("C3", "X-gc3")])


def p15(ctx: FigureContext) -> dict:
    """NoPE vs RoPE at k in {3, 12}."""
    return _control_pair(ctx, "P15", "P15 NoPE vs RoPE", [("C3", "X-nope3"), ("C12", "X-nope12")])


def p16(ctx: FigureContext) -> dict:
    """R-dist x filler at 32K, S0 and S12 separately (MS)."""
    df = ctx.df
    d = df[(df["task"] == "R-dist")] if len(df) else df
    if not len(d):
        return _empty(ctx, "P16", "P16 R-dist x filler", "no MS R-dist results")
    fig, ax = new_figure(ncols=2, width=3.0)
    for a, c in zip(ax[0], ("C0", "C12")):
        sub = d[d["config"] == c]
        for filler, g in sub.groupby(sub["filler_source"].fillna("natural")):
            m = g.groupby(["seed", "level"])["em_tf"].mean().unstack()
            xs = [float(str(x).lstrip("d")) for x in m.columns]
            plot_seeds(a, xs, {s: m.loc[s].to_numpy() for s in m.index}, color=INK if filler == "natural" else INK_MUTED,
                       label=str(filler))
        a.set_xscale("log", base=2)
        a.set_xlabel("Needle distance d (tokens)")
        a.set_title(f"{c}")
        a.legend()
    ax[0, 0].set_ylabel("EM_tf")
    return ctx.save(fig, "P16", caption="Distance dependence by filler type at 32K, per architecture.", ci="none")


def p17(ctx: FigureContext) -> dict:
    """LM-1 per-position loss and LM-3 repeat-token loss by distance, per k (Tier 0)."""
    df = ctx.df
    d = df[df["task"].isin(["LM-1", "LM-3"])] if len(df) else df
    if not len(d) or d["loss"].isna().all():
        return _empty(ctx, "P17", "P17 zero-shot LM-1 / LM-3", "no Tier0 LM-1/LM-3 rows with 'loss'")
    fig, ax = new_figure(ncols=2, width=3.2)
    for a, task in zip(ax[0], ("LM-1", "LM-3")):
        sub = d[d["task"] == task]
        for c, g in sub.groupby("config"):
            m = g.groupby(["seed", "level"])["loss"].mean().unstack()
            xs = [max(1.0, float(str(x).lstrip("bd"))) for x in m.columns]
            order = np.argsort(xs)
            plot_seeds(a, np.array(xs)[order], {s: m.loc[s].to_numpy()[order] for s in m.index},
                       color=k_color(P.CONFIG_K.get(c, 0)), label=c)
        a.set_xscale("log", base=2)
        a.set_xlabel("Position bucket" if task == "LM-1" else "Repeat distance bucket")
        a.set_ylabel("Loss (nats)")
        a.set_title(task)
    ax[0, 0].legend(fontsize=6)
    return ctx.save(fig, "P17", caption="Zero-shot Stage-B checkpoint, before the TEF.", ci="none")


def p18(ctx: FigureContext) -> dict:
    """Strip plot of per-seed primary-cell scores (bimodality, U-05)."""
    r = ctx.res
    if r is None:
        return _empty(ctx, "P18", "P18 per-seed strip plot", "no CS results")
    ret = pd.DataFrame(r["scores"]["retrieval"]).T.astype(float)
    cfgs = [c for c in ret.index]
    fig, ax = new_figure(width=3.8)
    a = ax[0, 0]
    for i, c in enumerate(cfgs):
        v = ret.loc[c].dropna()
        a.scatter(np.full(len(v), i), v, s=18, color=k_color(P.CONFIG_K[c]) if c in P.CONFIG_K else INK_MUTED,
                  alpha=0.8, edgecolors="white", linewidths=0.5)
        for s, y in v.items():
            a.annotate(str(s), (i, y), xytext=(4, 0), textcoords="offset points", fontsize=5, color=INK_MUTED)
    sc = r.get("success_counts", {})
    if sc:
        mid = next(iter(sc.values()))["midpoint"]
        a.axhline(mid, color=INK_MUTED, ls="--", lw=1, label="success midpoint")
        a.legend(loc="lower right")
    a.set_xticks(range(len(cfgs)), cfgs, rotation=45, fontsize=7)
    a.set_ylabel("Retrieval score (primary cell)")
    tr = r.get("seed_trigger", {})
    a.set_title(f"Per-seed scores; U-05 {'fired' if tr.get('fires') else 'did not fire'}")
    return ctx.save(fig, "P18", caption="Every seed's primary-cell score, labelled by seed index.", ci="none")


FIGURES = {f"P{i}": globals()[f"p{i:02d}"] for i in range(1, 19)}


def make(name: str, ctx: FigureContext | None = None) -> dict[str, Any]:
    """Build one figure by name ("P1" ... "P18")."""
    return FIGURES[name](ctx or FigureContext())


def cli_main(name: str, argv: list[str] | None = None) -> dict[str, Any]:
    """Command-line entry used by ``plots/make_pXX.py``."""
    import argparse

    ap = argparse.ArgumentParser(description=f"Protocol figure {name} ({FIGURES[name].__doc__.strip().splitlines()[0]})")
    ap.add_argument("--results", default="experiments/results")
    ap.add_argument("--registry", default="experiments/registry.jsonl")
    ap.add_argument("--out", default="plots/out")
    ap.add_argument("--B", type=int, default=P.BOOTSTRAP_B, help="bootstrap replicates (protocol: 10,000)")
    ap.add_argument("--scale", default=None)
    ap.add_argument("--l-ext", type=int, default=None, help="L_ext if the 16K fallback was taken (shades T > L_ext)")
    ap.add_argument("--kstar", type=int, default=None)
    a = ap.parse_args(argv)
    ctx = FigureContext(Path(a.results), Path(a.registry), Path(a.out),
                        AnalysisSpec(B=a.B, scale=a.scale, kstar=a.kstar), l_ext=a.l_ext)
    meta = make(name, ctx)
    print(f"{name}: {meta.get('status', 'ok')} -> {Path(a.out) / name}.pdf ({meta['n_inputs']} inputs, "
          f"sha256 {meta['inputs_sha256'][:12]})")
    return meta

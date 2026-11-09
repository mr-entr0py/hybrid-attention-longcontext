"""The pre-registered analysis plan (protocol §11), from item-level rows to a results dict.

:func:`run_analysis` executes, in the protocol's order of operations (§11.2):

1. G-REF per length on DG items (criteria (a) adequacy and (b) discriminability, seed-pooled, plus
   >= n - 1 of n seeds individually adequate), and the reference-inversion check;
2. primary-cell selection (passing lengths of {16K, 32K}; fallback 8K, labelled);
3. the U-05 seed trigger on the first n0 seeds of the four anchors (effect-blind);
4. per-seed family scores at the primary cell (retrieval / multi-hop / aggregation ladder means);
5. confirmatory tests H1, H1-min, H3, H5(a): paired one-sided t (df = n - 1), Holm over the support
   family and over the falsification family;
6. sensitivity analyses: seed-blocked pooled-variance model over the 4 x n core table and the
   hierarchical paired bootstrap (B = 10,000; seeds then contexts within strata, common indices);
   gold-log-probability L_k ("metric-robust");
7. I-22 falsification guards (gates, test, dose guard, saturation guard);
8. success counts with Clopper–Pearson CIs;
9. secondary estimands (non-inferiority, Fieller R, H1-robust and per-level results with Holm, R_best,
   H4, H6′, H5(b) with the secondary-family Holm, A-freq contrasts) and descriptive ones (H2 isotonic /
   saturating / k80 / g1 vs g_late, H6 hull margins, N50);
10. Tier-A decision-point triggers U-02 and U-06.

The output is plain JSON-serialisable data; :mod:`.verdicts` turns it into labelled verdicts.
Nothing here looks at an effect before the step the protocol allows (the trigger uses only the
blinded residual SD and raw seed ranges).
"""

from __future__ import annotations

import hashlib
import json
import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from . import protocol as P
from .scores import CellTable, family_spec
from .stats import (BootstrapDraws, fieller, fieller_paired, hierarchical_bootstrap, holm,
                    isotonic_recovery, k80, l_contrast, n50_logistic, one_sample_t, percentile_ci,
                    pooled_contrast, pooled_two_sample_t, recovery, saturating_fit, seed_trigger, slope_log2,
                    success_counts, tost_one_sample, tost_welch, welch_t)

RET, HOP, AGG = "retrieval", "multihop", "aggregation"
DEFAULT_N0 = {"D384": 5, "D512": 4}  # §8.2 / debate §6.2 resolution


@dataclass
class AnalysisSpec:
    """Run-time inputs of the analysis that are not item-level results.

    Everything numeric that the protocol fixes lives in :mod:`.protocol`; this only says *where* things
    are (suite / branch / checkpoint names) and carries facts decided elsewhere (gate outcomes).
    """

    cs_suite: str = "CS"
    dg_suite: str = "DG"
    dc_suite: str = "DC"
    main_branch: str = "main"
    main_ckpt: str | None = None
    """Checkpoint directory of the confirmatory evaluation (None: the unique one per run)."""
    dose_branch: str = "tef4000"
    dose_ckpts: tuple[str, str] = ("tef2000", "tef4000")
    saturation_branch: str = "stageB2x"
    scale: str | None = None
    n0: int | None = None
    """Seeds per anchor before the U-05 trigger (default: 5 at D384, 4 at D512)."""
    gates: dict[str, bool] = field(default_factory=dict)
    """Outcomes of G0, G-RP, FEAS-1, FEAS-2, G-FULL (from decisions.md). Missing = not passed."""
    failed_runs: dict[str, int] = field(default_factory=dict)
    """Per config, permanently failed runs (count as failures in success counts; §8.1)."""
    B: int = P.BOOTSTRAP_B
    bootstrap: bool = True
    kstar: int | None = None
    """k* if already fixed (decisions.md); otherwise computed from composition data (§3.2)."""
    lm4_slope: float | None = None
    """Zero-shot LM-4 slope of Lambda(g) on log2 g (§9.6), for the "exposure-robust" label."""
    lm4_slope_n_seeds: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


# =============================================================================================
# helpers
# =============================================================================================
def _f(x: Any) -> Any:
    """JSON-friendly conversion (numpy scalars/arrays, NaN -> None, DataFrames -> nested dicts)."""
    if isinstance(x, pd.DataFrame):
        return {str(i): {str(c): _f(v) for c, v in row.items()} for i, row in x.iterrows()}
    if isinstance(x, pd.Series):
        return {str(k): _f(v) for k, v in x.items()}
    if isinstance(x, dict):
        return {str(k): _f(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_f(v) for v in x]
    if isinstance(x, np.ndarray):
        return [_f(v) for v in x.tolist()]
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        v = float(x)
        return None if math.isnan(v) else (v if math.isfinite(v) else ("inf" if v > 0 else "-inf"))
    if hasattr(x, "to_dict"):
        return _f(x.to_dict())
    return x


def _row(tab: pd.DataFrame, cfg: str) -> pd.Series:
    return tab.loc[cfg].dropna() if cfg in tab.index else pd.Series(dtype=float)


def _paired(tab: pd.DataFrame, cfgs: Sequence[str]) -> pd.DataFrame:
    """Rows ``cfgs`` restricted to seeds where all of them are present (paired by seed index)."""
    sub = tab.loc[[c for c in cfgs]]
    return sub.loc[:, sub.notna().all(axis=0)]


def _agrees(lower: float, upper: float, direction: str) -> bool:
    """Sensitivity agreement [IMPL]: the one-sided 95% bound in the verdict's direction excludes 0."""
    return bool(lower > 0) if direction == "greater" else bool(upper < 0)


def table_hash(df: pd.DataFrame) -> str:
    cols = [c for c in ("run_id", "ckpt", "suite", "task", "level", "T", "context_id", "query_idx", "em_tf")
            if c in df.columns]
    h = hashlib.sha256(pd.util.hash_pandas_object(df[cols].sort_values(cols[:-1]), index=False).values.tobytes())
    return h.hexdigest()


# =============================================================================================
# G-REF and primary cell (§11.2)
# =============================================================================================
def gref_at(table: CellTable, T: int, *, low: str = "C0", high: str = "C12") -> dict[str, Any]:
    """G-REF criteria at length T on ``table`` (normally DG): (a), (b), per-seed (a), pass rule."""
    spec = family_spec(RET)
    w_T = table.ladder_weights(spec, [T])
    w_C = table.ladder_weights(spec, [T], control_for=T)
    sT, sC = table.scores(w_T), table.scores(w_C)
    s12, s12c = _row(sT, high), _row(sC, high)
    s0 = _row(sT, low)
    seeds = sorted(set(s12.index) & set(s12c.index))
    n = len(seeds)
    m12, m12c, m0 = float(s12.mean()), float(s12c.mean()), float(s0.mean()) if len(s0) else float("nan")
    a_pooled = m12 >= P.GREF_ADEQUACY * m12c
    b_pooled = (m12 - m0) >= P.GREF_DISCRIMINABILITY
    per_seed = {int(s): bool(s12[s] >= P.GREF_ADEQUACY * s12c[s]) for s in seeds}
    n_ad = sum(per_seed.values())
    passes = bool(a_pooled and b_pooled and n_ad >= n - 1 and n >= 1)
    return dict(T=T, suite=table.suite, S12=m12, S12_C0=m12c, S0=m0, ratio=m12 / m12c if m12c else float("nan"),
                gap=m12 - m0, a_pooled=bool(a_pooled), b_pooled=bool(b_pooled), per_seed_adequate=per_seed,
                n_adequate=n_ad, n=n, passes=passes, status="pass" if passes else "reference-unstable",
                per_seed_ratio={int(s): float(s12[s] / s12c[s]) for s in seeds})


def reference_inversion(table: CellTable, T: int, gref: Mapping[str, Any]) -> dict[str, Any]:
    """§11.2: C12 fails (a) while a hybrid passes its own (a) and beats S12 by >= 5 pp on DG."""
    if gref["a_pooled"]:
        return dict(triggered=False, reason="C12 adequate")
    spec = family_spec(RET)
    sT = table.scores(table.ladder_weights(spec, [T]))
    sC = table.scores(table.ladder_weights(spec, [T], control_for=T))
    cands = {}
    for c in table.configs:
        if c in ("C12", "C0"):
            continue
        mt, mc = float(_row(sT, c).mean()), float(_row(sC, c).mean())
        if np.isfinite(mt) and mt >= P.GREF_ADEQUACY * mc and mt - gref["S12"] >= P.REF_INVERSION_MARGIN:
            cands[c] = mt
    return dict(triggered=bool(cands), hybrids=cands, best=max(cands, key=cands.get) if cands else None)


def select_primary(gref_long: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    passing = [T for T in P.LONG_LENGTHS if gref_long.get(T, {}).get("passes", False)]
    if passing:
        return dict(lengths=passing, fallback=False, label=None)
    return dict(lengths=[P.FALLBACK_LENGTH], fallback=True,
                label="H1 at 8K, not a long-context result")


# =============================================================================================
# main entry point
# =============================================================================================
def run_analysis(df: pd.DataFrame, spec: AnalysisSpec | None = None) -> dict[str, Any]:
    spec = spec or AnalysisSpec()
    out: dict[str, Any] = {"protocol": "v1.0 §11", "spec": spec.to_dict(), "warnings": []}
    warn = out["warnings"].append

    cs = CellTable.from_rows(df, spec.cs_suite, "em_tf", branch=spec.main_branch, ckpt=spec.main_ckpt)
    cs_lp = CellTable.from_rows(df, spec.cs_suite, "logp_gold", branch=spec.main_branch, ckpt=spec.main_ckpt)
    try:
        dg: CellTable | None = CellTable.from_rows(df, spec.dg_suite, "em_tf", branch=spec.main_branch,
                                                   ckpt=spec.main_ckpt)
    except ValueError:
        dg = None
        warn("no DG suite: G-REF at 16K/32K evaluated on CS items instead")
    out["inputs"] = dict(n_rows=int(len(df)), table_sha256=table_hash(df), files=list(df.attrs.get("inputs", [])),
                         skipped_reruns=list(df.attrs.get("skipped_reruns", [])))
    scale = spec.scale or (str(df["scale"].dropna().iloc[0]) if "scale" in df and df["scale"].notna().any() else None)
    n0 = spec.n0 or DEFAULT_N0.get(scale or "", None)

    # ---------------------------------------------------------------- 1. G-REF
    gref: dict[int, dict] = {}
    inversion: dict[int, dict] = {}
    for T in cs.lengths():
        use = dg if (dg is not None and T in dg.lengths() and T in P.LONG_LENGTHS) else cs
        try:
            gref[T] = gref_at(use, T)
            if use is dg:
                inversion[T] = reference_inversion(dg, T, gref[T])
        except ValueError as e:
            warn(f"G-REF at T={T} not evaluable: {e}")
    out["gref"] = gref
    out["reference_inversion"] = inversion

    # ---------------------------------------------------------------- 2. primary cell
    primary = select_primary({T: gref[T] for T in P.LONG_LENGTHS if T in gref})
    Tp = primary["lengths"]
    out["primary"] = primary

    w = {RET: cs.ladder_weights(family_spec(RET), Tp)}
    for fam in (HOP, AGG):
        try:
            w[fam] = cs.ladder_weights(family_spec(fam), Tp)
        except ValueError as e:
            warn(f"family {fam} not available at the primary cell: {e}")
    S = {fam: cs.scores(wv) for fam, wv in w.items()}
    S_lp = {fam: cs_lp.scores(wv) for fam, wv in w.items()}
    out["scores"] = {fam: s for fam, s in S.items()}
    out["scores_logp"] = {fam: s for fam, s in S_lp.items()}
    ret = S[RET]

    # ---------------------------------------------------------------- 3. U-05 trigger
    anchors = [c for c in P.ANCHOR_CONFIGS if c in ret.index]
    trig: dict[str, Any] = {}
    if len(anchors) == 4 and n0:
        first = [s for s in ret.columns if s < n0]
        tab = ret.loc[anchors, first]
        if tab.notna().all().all() and len(first) == n0:
            tr = seed_trigger(tab)
            all_seeds = sorted(ret.loc[anchors].dropna(axis=1, how="any").columns)
            trig = dict(tr.to_dict(), n0=n0, seeds_used=[int(s) for s in first],
                        seeds_present=[int(s) for s in all_seeds])
            if tr.fires and len(all_seeds) < n0 + P.TRIGGER_EXTRA_SEEDS:
                warn("U-05 fired but the trigger seeds are not (yet) in the results")
            if not tr.fires and len(all_seeds) > n0:
                warn("U-05 did not fire but extra anchor seeds are present (protocol violation?)")
        else:
            trig = dict(status="not evaluable: incomplete first-n0 anchor table", n0=n0)
    else:
        trig = dict(status="not evaluable: anchors or n0 missing", n0=n0)
    out["seed_trigger"] = trig

    # ---------------------------------------------------------------- 4-5. confirmatory tests
    tests: dict[str, dict] = {}
    for name, k in (("H1", P.H1_K), ("H1-min", P.H1MIN_K)):
        tests[name] = h1_test(ret, f"C{k}", S_lp.get(RET))
    tests["H3"] = h3_test(S, cs, Tp, dg)
    tests["H5a"] = h5a_test(cs, gref, spec)
    out["tests"] = tests

    support_p = {h: (tests[h]["p_support"] if tests[h].get("evaluable", True) and tests[h].get("p_support") is not None
                     else 1.0) for h in P.SUPPORT_FAMILY}
    out["holm_support"] = holm(support_p)
    fals_p = {h: (tests[h]["p_falsify"] if tests[h].get("p_falsify") is not None else 1.0)
              for h in P.FALSIFICATION_FAMILY}
    out["holm_falsification"] = holm(fals_p)

    # ---------------------------------------------------------------- 6. sensitivity
    draws = None
    if spec.bootstrap:
        draws = _bootstrap(cs, Tp, spec)
    out["sensitivity"] = sensitivity(ret, tests, draws, cs, Tp)

    # ---------------------------------------------------------------- 7. I-22 guards
    out["guards"] = falsification_guards(df, spec, gref, primary, out, cs)

    # ---------------------------------------------------------------- 8. success counts
    out["success_counts"] = success_counts(ret.loc[anchors] if anchors else ret, failed=spec.failed_runs) \
        if {"C0", "C12"} <= set(ret.index) else {}

    # ---------------------------------------------------------------- 9. secondary & descriptive
    kstar = spec.kstar if spec.kstar is not None else select_kstar(ret)
    out["kstar"] = dict(value=kstar, fixed_by_spec=spec.kstar is not None)
    out["secondary"] = secondary(S, cs, dg, Tp, kstar, tests, draws)
    out["descriptive"] = descriptive(S, cs, Tp, draws)

    # ---------------------------------------------------------------- 10. Tier-A triggers
    out["triggers"] = tier_a_triggers(ret)
    out["fgc_family"] = fgc_family(ret)
    return _f(out)


# =============================================================================================
# confirmatory pieces
# =============================================================================================
def h1_test(ret: pd.DataFrame, cfg_k: str, ret_logp: pd.DataFrame | None = None, *, c0: str = "C0",
            c12: str = "C12") -> dict[str, Any]:
    """L_k per seed, one-sided t in both directions (§11.3). Missing configs -> not evaluable."""
    if not {cfg_k, c0, c12} <= set(ret.index):
        return dict(evaluable=False, reason=f"missing configs among {cfg_k}, {c0}, {c12}", p_support=None,
                    p_falsify=None)
    tab = _paired(ret, [cfg_k, c0, c12])
    if tab.shape[1] < 2:
        return dict(evaluable=False, reason="fewer than 2 paired seeds", p_support=None, p_falsify=None)
    L = l_contrast(tab.loc[cfg_k], tab.loc[c0], tab.loc[c12])
    tg, tl = one_sample_t(L.to_numpy(), "greater"), one_sample_t(L.to_numpy(), "less")
    R = recovery(tab.loc[cfg_k].mean(), tab.loc[c0].mean(), tab.loc[c12].mean())
    res = dict(evaluable=True, configs=[cfg_k, c0, c12], seeds=[int(s) for s in tab.columns],
               per_seed=dict(zip(map(int, tab.columns), L.to_numpy().tolist())), estimate=float(L.mean()),
               se=float(tg.se), df=float(tg.df), t=float(tg.t), p_support=float(tg.p), p_falsify=float(tl.p),
               lower95=float(tg.lower), upper95=float(tl.upper), R_pooled=float(R),
               G_pooled=float(tab.loc[c12].mean() - tab.loc[c0].mean()))
    if ret_logp is not None and {cfg_k, c0, c12} <= set(ret_logp.index):
        tl_ = _paired(ret_logp, [cfg_k, c0, c12])
        Llp = l_contrast(tl_.loc[cfg_k], tl_.loc[c0], tl_.loc[c12])
        res["logp"] = dict(estimate=float(Llp.mean()), per_seed=dict(zip(map(int, tl_.columns), Llp.tolist())))
    return res


def h3_test(S: Mapping[str, pd.DataFrame], cs: CellTable, Tp: Sequence[int], dg: CellTable | None,
            k: int = P.H1_K) -> dict[str, Any]:
    """D3 = R_ret(3) - R_hop(3) per seed (§2, §11.7); evaluable iff every seed has G >= 15 pp in both."""
    if HOP not in S:
        return dict(evaluable=False, reason="multi-hop family missing", p_support=None)
    cfgs = ["C0", f"C{k}", "C12"]
    ret, hop = S[RET], S[HOP]
    if not set(cfgs) <= set(ret.index) or not set(cfgs) <= set(hop.index):
        return dict(evaluable=False, reason="missing configs", p_support=None)
    tr, th = _paired(ret, cfgs), _paired(hop, cfgs)
    seeds = sorted(set(tr.columns) & set(th.columns))
    tr, th = tr[seeds], th[seeds]
    G_ret = tr.loc["C12"] - tr.loc["C0"]
    G_hop = th.loc["C12"] - th.loc["C0"]
    evaluable = bool((G_ret >= P.H3_MIN_GAP).all() and (G_hop >= P.H3_MIN_GAP).all())
    Rr = recovery(tr.loc[f"C{k}"], tr.loc["C0"], tr.loc["C12"])
    Rh = recovery(th.loc[f"C{k}"], th.loc["C0"], th.loc["C12"])
    D = np.asarray(Rr - Rh, float)
    tg = one_sample_t(D, "greater")
    tost = tost_one_sample(D, P.TOST_RECOVERY)
    # Fieller CIs of the seed-pooled recoveries (sensitivity, §11.7)
    f_ret = fieller_paired(np.asarray(tr.loc[f"C{k}"] - tr.loc["C0"]), np.asarray(G_ret))
    f_hop = fieller_paired(np.asarray(th.loc[f"C{k}"] - th.loc["C0"]), np.asarray(G_hop))
    equated = equated_check(cs, dg, Tp, k)
    sign_ok = equated["consistent"]
    # Logit-scale sensitivity (DEVIATION: stands in for the binomial GLMM; see report)
    logit_d3 = _logit_d3(cs, Tp, k)
    return dict(evaluable=evaluable, seeds=[int(s) for s in seeds], per_seed=dict(zip(map(int, seeds), D.tolist())),
                G_ret=dict(zip(map(int, seeds), G_ret.tolist())), G_hop=dict(zip(map(int, seeds), G_hop.tolist())),
                estimate=float(D.mean()), se=float(tg.se), df=float(tg.df), p_raw=float(tg.p),
                p_support=float(tg.p) if evaluable else 1.0, lower95=float(tg.lower), equated=equated,
                equated_sign_consistent=sign_ok, tost=tost.to_dict(), fieller_R_ret=f_ret.to_dict(),
                fieller_R_hop=f_hop.to_dict(), logit_scale=logit_d3)


def _logit_d3(cs: CellTable, Tp: Sequence[int], k: int) -> dict[str, Any]:
    """D3 on the empirical-logit scale of stratum means (closest available stand-in for the GLMM)."""
    cm = cs.cell_means()
    nq = np.array([a.shape[0] * P.Q_PER_CONTEXT for a in cs.ctx], float)[:, None, None]
    lg = np.log((cm * nq + 0.5) / ((1 - cm) * nq + 0.5))  # empirical logit per stratum
    res = {}
    try:
        vals = {}
        for fam in (RET, HOP):
            wv = cs.ladder_weights(family_spec(fam), Tp)
            used = wv != 0
            sc = np.tensordot(wv[used], lg[used], axes=(0, 0))
            sc = pd.DataFrame(np.where(cs.present, sc, np.nan), index=cs.configs, columns=cs.seeds)
            t = _paired(sc, ["C0", f"C{k}", "C12"])
            vals[fam] = recovery(t.loc[f"C{k}"], t.loc["C0"], t.loc["C12"])
        seeds = sorted(set(vals[RET].index) & set(vals[HOP].index))
        D = np.asarray(vals[RET][seeds] - vals[HOP][seeds], float)
        tg = one_sample_t(D, "greater")
        res = dict(estimate=float(D.mean()), lower95=float(tg.lower), p=float(tg.p))
    except (ValueError, KeyError) as e:
        res = dict(status=f"not evaluable: {e}")
    return res


def equated_check(cs: CellTable, dg: CellTable | None, Tp: Sequence[int], k: int) -> dict[str, Any]:
    """U-11: (retrieval level, multi-hop level) pairs with |dS12| <= 10 pp on DG at the primary length(s);
    D3 at those levels must have the same sign as the overall D3. No pairs -> condition vacuous."""
    if dg is None or "C12" not in dg.configs:
        return dict(pairs=[], consistent=True, note="no DG: vacuous")
    pairs, d3s = [], []
    try:
        ret_lv = [(t, lv) for t in ("R1", "R-load") for lv in sorted(set(dg.levels_at(t, Tp[0])))
                  if (t, lv) not in P.N50_ONLY_LEVELS]
        hop_lv = sorted(set(dg.levels_at("M-hop", Tp[0])))
    except (IndexError, ValueError):
        return dict(pairs=[], consistent=True, note="levels unavailable: vacuous")

    def s12(table: CellTable, task: str, lv: str) -> float:
        return float(_row(table.scores(table.level_weights(task, lv, Tp)), "C12").mean())

    def per_seed_R(task: str, lv: str) -> pd.Series:
        sc = cs.scores(cs.level_weights(task, lv, Tp))
        t = _paired(sc, ["C0", f"C{k}", "C12"])
        return recovery(t.loc[f"C{k}"], t.loc["C0"], t.loc["C12"])

    for (t, lv) in ret_lv:
        for h in hop_lv:
            try:
                if abs(s12(dg, t, lv) - s12(dg, "M-hop", h)) <= P.EQUATED_PAIR_MAX:
                    d = float((per_seed_R(t, lv) - per_seed_R("M-hop", h)).mean())
                    pairs.append(dict(retrieval=f"{t}:{lv}", multihop=h, D3=d))
                    d3s.append(d)
            except (ValueError, KeyError):
                continue
    return dict(pairs=pairs, consistent=bool(all(d > 0 for d in d3s)) if d3s else True,
                note=None if d3s else "no difficulty-equated pairs: vacuous")


def h5_slopes(cs: CellTable, family: str, Ts: Sequence[int], *, hi: str = "C12", lo: str = "C0",
              table: pd.DataFrame | None = None) -> pd.Series:
    """Per-seed OLS slope of Delta0(T) = S12(T) - S0(T) on log2 T over the fixed level set."""
    levels = P.H5_FIXED_LEVELS[family]
    deltas = []
    for T in Ts:
        sc = cs.scores(cs.ladder_weights(family_spec(family, levels), [T]))
        t = _paired(sc, [hi, lo])
        deltas.append(t.loc[hi] - t.loc[lo])
    D = pd.concat(deltas, axis=1).dropna()
    D.columns = list(Ts)
    return pd.Series(slope_log2(Ts, D.to_numpy()), index=D.index)


def h5a_test(cs: CellTable, gref: Mapping[int, Mapping[str, Any]], spec: AnalysisSpec) -> dict[str, Any]:
    Ts = [T for T in P.ALL_LENGTHS if T in cs.lengths()]
    if not {"C0", "C12"} <= set(cs.configs) or len(Ts) < 2:
        return dict(evaluable=False, reason="needs C0 and C12 at >= 2 lengths", p_support=None)
    b = h5_slopes(cs, RET, Ts)
    tg = one_sample_t(b.to_numpy(), "greater")
    tost = tost_one_sample(b.to_numpy(), P.TOST_SLOPE)
    failing = [T for T in Ts if not gref.get(T, {}).get("passes", False)]
    res: dict[str, Any] = dict(evaluable=True, lengths=Ts, per_seed=dict(zip(map(int, b.index), b.tolist())),
                               estimate=float(b.mean()), se=float(tg.se), df=float(tg.df), p_support=float(tg.p),
                               lower95=float(tg.lower), tost=tost.to_dict(), failing_gref=failing,
                               conditional_on_reference_adequacy=bool(failing))
    passing = [T for T in Ts if T not in failing]
    if failing and len(passing) >= 2:
        bp = h5_slopes(cs, RET, passing)
        res["passing_only"] = dict(lengths=passing, estimate=float(bp.mean()),
                                   p=float(one_sample_t(bp.to_numpy(), "greater").p) if len(bp) > 1 else None)
    elif failing:
        res["passing_only"] = dict(lengths=passing, estimate=None, note="fewer than 2 passing lengths")
    if spec.lm4_slope is not None:
        res["lm4_slope"] = spec.lm4_slope
        res["exposure_robust"] = bool(np.sign(spec.lm4_slope) == np.sign(res["estimate"]))
    else:
        res["exposure_robust"] = None
    return res


# =============================================================================================
# sensitivity (§11.3)
# =============================================================================================
def _bootstrap(cs: CellTable, Tp: Sequence[int], spec: AnalysisSpec) -> BootstrapDraws | None:
    weights: dict[str, np.ndarray] = {}
    for fam in (RET, HOP):
        try:
            weights[f"{fam}@primary"] = cs.ladder_weights(family_spec(fam), Tp)
        except ValueError:
            pass
    Ts = [T for T in P.ALL_LENGTHS if T in cs.lengths()]
    for fam in (RET, HOP):
        for T in Ts:
            try:
                weights[f"{fam}@fixed@{T}"] = cs.ladder_weights(family_spec(fam, P.H5_FIXED_LEVELS[fam]), [T])
            except ValueError:
                pass
    cfgs = [c for c in cs.configs if c in P.CONFIG_K]
    if not cfgs:
        return None
    return hierarchical_bootstrap(cs, weights, configs=cfgs, B=spec.B, key="confirmatory")


def _boot_ci(vals: np.ndarray) -> dict[str, float]:
    lo90, hi90 = percentile_ci(vals, 0.90)
    lo95, hi95 = percentile_ci(vals, 0.95)
    return dict(mean=float(np.nanmean(vals)), lower95_one_sided=lo90, upper95_one_sided=hi90,
                ci95=[lo95, hi95])


def boot_L(draws: BootstrapDraws, k_cfg: str, name: str = f"{RET}@primary") -> np.ndarray:
    g = draws.get
    return np.nanmean(g(name, k_cfg) - 0.2 * g(name, "C0") - 0.8 * g(name, "C12"), axis=1)


def boot_D3(draws: BootstrapDraws, k: int = 3) -> np.ndarray:
    g = draws.get
    with np.errstate(divide="ignore", invalid="ignore"):
        rr = recovery(g(f"{RET}@primary", f"C{k}"), g(f"{RET}@primary", "C0"), g(f"{RET}@primary", "C12"))
        rh = recovery(g(f"{HOP}@primary", f"C{k}"), g(f"{HOP}@primary", "C0"), g(f"{HOP}@primary", "C12"))
    return np.nanmean(rr - rh, axis=1)


def boot_slope(draws: BootstrapDraws, family: str, Ts: Sequence[int]) -> np.ndarray:
    g = draws.get
    D = np.stack([g(f"{family}@fixed@{T}", "C12") - g(f"{family}@fixed@{T}", "C0") for T in Ts], axis=-1)
    return np.nanmean(slope_log2(Ts, D), axis=1)


def sensitivity(ret: pd.DataFrame, tests: Mapping[str, dict], draws: BootstrapDraws | None, cs: CellTable,
                Tp: Sequence[int]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    anchors = [c for c in P.ANCHOR_CONFIGS if c in ret.index]
    core = _paired(ret, anchors) if len(anchors) == 4 else None
    for name, k in (("H1", P.H1_K), ("H1-min", P.H1MIN_K)):
        if not tests[name].get("evaluable"):
            continue
        rec: dict[str, Any] = {}
        if core is not None and core.shape[1] >= 2:
            con = {f"C{k}": 1.0, "C0": -P.L_WEIGHT_S0, "C12": -P.L_WEIGHT_S12}
            pg, pl = pooled_contrast(core, con, "greater"), pooled_contrast(core, con, "less")
            rec["pooled"] = dict(estimate=float(pg.estimate), se=float(pg.se), df=int(pg.df),
                                 lower95=float(pg.lower), upper95=float(pl.upper), p_greater=float(pg.p),
                                 p_less=float(pl.p), resid_sd=float(pg.resid_sd))
        if draws is not None:
            rec["bootstrap"] = _boot_ci(boot_L(draws, f"C{k}"))
        out[name] = rec
    if draws is not None and tests["H3"].get("per_seed") and f"{HOP}@primary" in draws.scores:
        out["H3"] = dict(bootstrap=_boot_ci(boot_D3(draws)))
    if draws is not None and tests["H5a"].get("evaluable"):
        out["H5a"] = dict(bootstrap=_boot_ci(boot_slope(draws, RET, tests["H5a"]["lengths"])))
    # agreement flags for both directions
    for name, rec in out.items():
        for meth, r in list(rec.items()):
            if not isinstance(r, dict):
                continue
            lo = r.get("lower95", r.get("lower95_one_sided"))
            hi = r.get("upper95", r.get("upper95_one_sided"))
            r["agrees_greater"] = _agrees(lo, hi, "greater")
            r["agrees_less"] = _agrees(lo, hi, "less")
    return out


# =============================================================================================
# I-22 guards (§11.6)
# =============================================================================================
def falsification_guards(df: pd.DataFrame, spec: AnalysisSpec, gref: Mapping[int, dict], primary: Mapping,
                         out: Mapping[str, Any], cs: CellTable) -> dict[str, Any]:
    res: dict[str, Any] = {}
    gates_ok = all(bool(spec.gates.get(g, False)) for g in P.REQUIRED_GATES)
    gref_ok = (not primary["fallback"]) and all(gref[T]["passes"] for T in primary["lengths"])
    res["gates"] = dict(required={g: bool(spec.gates.get(g, False)) for g in P.REQUIRED_GATES},
                        gref_at_primary=gref_ok, holds=bool(gates_ok and gref_ok))
    for name, k in (("H1", P.H1_K), ("H1-min", P.H1MIN_K)):
        t = out["tests"][name]
        if not t.get("evaluable"):
            res[name] = dict(holds=False, reason="not evaluable")
            continue
        holm_ok = out["holm_falsification"][name]["reject"]
        sens = out["sensitivity"].get(name, {})
        agree = all(r.get("agrees_less", False) for r in sens.values() if isinstance(r, dict))
        have = all(m in sens for m in ("pooled", "bootstrap"))
        res[name] = dict(test=dict(holm_reject=holm_ok, sensitivity_agree=bool(agree and have),
                                   holds=bool(holm_ok and agree and have)))
        res[name]["dose"] = dose_guard(df, spec, primary["lengths"], k)
        res[name]["saturation"] = saturation_guard(df, spec, primary["lengths"], k)
        res[name]["holds"] = bool(res["gates"]["holds"] and res[name]["test"]["holds"]
                                  and res[name]["dose"].get("holds", False)
                                  and res[name]["saturation"].get("holds", False))
    return res


def _branch_scores(df: pd.DataFrame, suite: str, branch: str, ckpt: str | None, Tp: Sequence[int]
                   ) -> pd.DataFrame | None:
    try:
        tab = CellTable.from_rows(df, suite, "em_tf", branch=branch, ckpt=ckpt)
    except ValueError:
        return None
    Ts = [T for T in Tp if T in tab.lengths()]
    if not Ts:
        return None
    return tab.scores(tab.ladder_weights(family_spec(RET), Ts))


def dose_guard(df: pd.DataFrame, spec: AnalysisSpec, Tp: Sequence[int], k: int) -> dict[str, Any]:
    """Seed 0: the TEF-4000 branch keeps L_k < 0 and |S_k(4000) - S_k(2000)| < 3 pp (§11.6.3).

    [IMPL] Both quantities use the DC suite (scored at TEF 2000 and 4000) on the primary lengths it
    contains; L_k at TEF 4000 uses the CS suite of the branch when it was scored."""
    c2k, c4k = spec.dose_ckpts
    s2 = _branch_scores(df, spec.dc_suite, spec.dose_branch, c2k, Tp)
    s4 = _branch_scores(df, spec.dc_suite, spec.dose_branch, c4k, Tp)
    s4cs = _branch_scores(df, spec.cs_suite, spec.dose_branch, c4k, Tp)
    if s2 is None or s4 is None:
        return dict(holds=False, reason="dose-branch results missing")
    cfg = f"C{k}"
    seed = min(s4.columns)
    src = s4cs if (s4cs is not None and {cfg, "C0", "C12"} <= set(s4cs.index)) else s4
    try:
        L4 = float(l_contrast(src.loc[cfg, seed], src.loc["C0", seed], src.loc["C12", seed]))
        plateau = float(abs(s4.loc[cfg, seed] - s2.loc[cfg, seed]))
    except KeyError as e:
        return dict(holds=False, reason=f"missing {e}")
    return dict(seed=int(seed), L_at_4000=L4, L_source=("CS" if src is s4cs else "DC"), plateau_diff=plateau,
                holds=bool(L4 < 0 and plateau < P.DOSE_PLATEAU))


def saturation_guard(df: pd.DataFrame, spec: AnalysisSpec, Tp: Sequence[int], k: int) -> dict[str, Any]:
    """2x Stage-B branch changes S12 at the primary length by <= 5 pp (§11.6.4); else "conditional on
    the long-stage budget" and L_k recomputed on the 2x branches."""
    s2x = _branch_scores(df, spec.cs_suite, spec.saturation_branch, spec.main_ckpt, Tp)
    s1x = _branch_scores(df, spec.cs_suite, spec.main_branch, spec.main_ckpt, Tp)
    if s2x is None or s1x is None or "C12" not in s2x.index:
        return dict(holds=False, reason="2x Stage-B results missing")
    seeds = [s for s in s2x.columns if pd.notna(s2x.loc["C12", s])]
    d = float(np.mean([s2x.loc["C12", s] - s1x.loc["C12", s] for s in seeds]))
    res: dict[str, Any] = dict(seeds=[int(s) for s in seeds], delta_S12=d, holds=bool(abs(d) <= P.SATURATION_MAX))
    cfg = f"C{k}"
    if {cfg, "C0", "C12"} <= set(s2x.index):
        res["L_on_2x"] = float(np.mean([l_contrast(s2x.loc[cfg, s], s2x.loc["C0", s], s2x.loc["C12", s])
                                        for s in seeds]))
    if not res["holds"]:
        res["label"] = "conditional on the long-stage budget"
    return res


# =============================================================================================
# secondary (§2 secondary estimands, §11.6 per-level, §11.7)
# =============================================================================================
def select_kstar(ret: pd.DataFrame) -> int:
    """§3.2: smallest k in {1, 2, 3} with seed-pooled R_ret(k) >= 0.3 (composition runs); else 3."""
    if not {"C0", "C12"} <= set(ret.index):
        return P.KSTAR_DEFAULT
    m0, m12 = ret.loc["C0"].mean(), ret.loc["C12"].mean()
    for k in P.KSTAR_CANDIDATES:
        c = f"C{k}"
        if c in ret.index and recovery(ret.loc[c].mean(), m0, m12) >= P.KSTAR_MIN_R:
            return k
    return P.KSTAR_DEFAULT


def informative_levels(dg: CellTable | None, Tp: Sequence[int]) -> list[tuple[str, str]]:
    """§9.7: S12 in [20%, 90%] and G >= 15 pp on dev items (seed-pooled), at the primary length(s)."""
    if dg is None or not {"C0", "C12"} <= set(dg.configs):
        return []
    out = []
    for t in FAMILY_TASKS_RET:
        for lv in sorted(set(dg.levels_at(t, Tp[0]))):
            if (t, lv) in P.N50_ONLY_LEVELS:
                continue
            try:
                sc = dg.scores(dg.level_weights(t, lv, Tp))
            except ValueError:
                continue
            s12, s0 = sc.loc["C12"].mean(), sc.loc["C0"].mean()
            if P.INFORMATIVE_S12[0] <= s12 <= P.INFORMATIVE_S12[1] and s12 - s0 >= P.INFORMATIVE_GAP:
                out.append((t, lv))
    return out


FAMILY_TASKS_RET = P.FAMILIES[RET]


def secondary(S: Mapping[str, pd.DataFrame], cs: CellTable, dg: CellTable | None, Tp: Sequence[int], kstar: int,
              tests: Mapping[str, dict], draws: BootstrapDraws | None) -> dict[str, Any]:
    ret = S[RET]
    out: dict[str, Any] = {}
    # non-inferiority Delta = S12 - S3 < 5 pp
    if {"C3", "C12"} <= set(ret.index):
        t = _paired(ret, ["C3", "C12"])
        dlt = (t.loc["C12"] - t.loc["C3"]).to_numpy()
        tt = one_sample_t(dlt, "less", null=P.NONINFERIORITY_MARGIN)
        out["noninferiority"] = dict(estimate=float(dlt.mean()), upper95=float(tt.upper), margin=P.NONINFERIORITY_MARGIN,
                                     p=float(tt.p), noninferior=bool(tt.p < P.ALPHA_FW))
    # R with Fieller CI, per family and k
    fr: dict[str, dict] = {}
    for fam, tab in S.items():
        for c in tab.index:
            k = P.CONFIG_K.get(c)
            if k in (None, 0, 12) or not {"C0", "C12"} <= set(tab.index):
                continue
            t = _paired(tab, [c, "C0", "C12"])
            if t.shape[1] < 2:
                continue
            fr.setdefault(fam, {})[c] = fieller_paired((t.loc[c] - t.loc["C0"]).to_numpy(),
                                                       (t.loc["C12"] - t.loc["C0"]).to_numpy()).to_dict()
    out["recovery_fieller"] = fr
    # per-level L_k with Holm across levels; H1-robust
    inf = informative_levels(dg, Tp)
    per_level: dict[str, Any] = {}
    for name, k in (("H1", P.H1_K), ("H1-min", P.H1MIN_K)):
        pv, rec = {}, {}
        for t in FAMILY_TASKS_RET:
            for lv in sorted(set(cs.levels_at(t, Tp[0]))):
                if (t, lv) in P.N50_ONLY_LEVELS:
                    continue
                try:
                    sc = cs.scores(cs.level_weights(t, lv, Tp))
                except ValueError:
                    continue
                r = h1_test(sc, f"C{k}")
                if r.get("evaluable"):
                    key = f"{t}:{lv}"
                    pv[key] = r["p_support"]
                    rec[key] = dict(estimate=r["estimate"], p=r["p_support"], R=r["R_pooled"],
                                    informative=(t, lv) in inf)
        if pv:
            hh = holm(pv)
            for key in rec:
                rec[key]["p_holm"] = hh[key]["p_holm"]
        robust = [rec[f"{t}:{lv}"]["R"] >= P.R_TARGET for (t, lv) in inf if f"{t}:{lv}" in rec]
        per_level[name] = dict(levels=rec, informative=[f"{t}:{lv}" for t, lv in inf],
                               h1_robust=(bool(all(robust)) if robust else None))
    out["per_level"] = per_level
    # R_best: best configuration chosen on DG
    sdg = None
    if dg is not None:
        try:
            sdg = dg.scores(dg.ladder_weights(family_spec(RET), Tp))
        except ValueError:
            out["R_best"] = dict(status="DG lacks the primary length(s)")
    if sdg is not None:
        means = sdg.mean(axis=1).dropna()
        best = str(means.idxmax())
        if best in ret.index and "C0" in ret.index:
            mb, m0 = ret.loc[best].mean(), ret.loc["C0"].mean()
            out["R_best"] = dict(best=best, chosen_on="DG", R={c: float(recovery(ret.loc[c].mean(), m0, mb))
                                                             for c in ret.index if c in P.CONFIG_K})
    # H4 placement at k*
    out["H4"] = h4_test(ret, kstar)
    # H6' GDN-E vs GDN-A at k*
    out["H6prime"] = h6prime_test(ret, kstar)
    # H5(b)
    Ts = [T for T in P.ALL_LENGTHS if T in cs.lengths()]
    try:
        b_ret = h5_slopes(cs, RET, Ts)
        b_hop = h5_slopes(cs, HOP, Ts)
        sd = sorted(set(b_ret.index) & set(b_hop.index))
        diff = (b_ret[sd] - b_hop[sd]).to_numpy()
        tt = one_sample_t(diff, "two-sided")
        out["H5b"] = dict(evaluable=True, per_seed=dict(zip(map(int, sd), diff.tolist())), estimate=float(diff.mean()),
                          p=float(tt.p), ci95=[float(tt.lower), float(tt.upper)],
                          tost=tost_one_sample(diff, P.TOST_SLOPE).to_dict(),
                          b_hop=dict(zip(map(int, sd), b_hop[sd].tolist())))
    except (ValueError, KeyError) as e:
        out["H5b"] = dict(evaluable=False, reason=str(e))
    sec_p = {"H4-c1": out["H4"].get("c1", {}).get("p", 1.0) if out["H4"].get("evaluable") else 1.0,
             "H4-c2": out["H4"].get("c2", {}).get("p", 1.0) if out["H4"].get("evaluable") else 1.0,
             "H6prime": out["H6prime"].get("p", 1.0) if out["H6prime"].get("evaluable") else 1.0,
             "H5b": out["H5b"].get("p", 1.0) if out["H5b"].get("evaluable") else 1.0}
    out["holm_secondary"] = holm(sec_p)
    # Aggregation (A-freq) contrasts, secondary
    if AGG in S:
        out["aggregation"] = {name: h1_test(S[AGG], f"C{k}") for name, k in (("L3", 3), ("L1", 1))}
    return out


def h4_test(ret: pd.DataFrame, kstar: int) -> dict[str, Any]:
    early, mid, late = f"P{kstar}-early", f"C{kstar}", f"P{kstar}-late"
    if not {early, mid, late} <= set(ret.index):
        return dict(evaluable=False, reason=f"placement runs {early}/{late} missing")
    t = _paired(ret, [early, mid, late])
    c1 = (t.loc[mid] - 0.5 * (t.loc[early] + t.loc[late])).to_numpy()
    c2 = (t.loc[late] - t.loc[early]).to_numpy()
    res: dict[str, Any] = dict(evaluable=True, kstar=kstar, seeds=[int(s) for s in t.columns])
    for name, x, con in (("c1", c1, {mid: 1.0, early: -0.5, late: -0.5}), ("c2", c2, {late: 1.0, early: -1.0})):
        tt = one_sample_t(x, "two-sided")
        pc = pooled_contrast(t, con, "two-sided")
        res[name] = dict(estimate=float(x.mean()), p=float(tt.p), ci95=[float(tt.lower), float(tt.upper)],
                         per_seed=dict(zip(map(int, t.columns), x.tolist())),
                         pooled=dict(estimate=float(pc.estimate), p=float(pc.p), df=int(pc.df)),
                         tost=tost_one_sample(x, P.TOST_ACCURACY).to_dict())
    res["no_placement_effect"] = bool(res["c1"]["tost"]["equivalent"] and res["c2"]["tost"]["equivalent"])
    return res


def h6prime_test(ret: pd.DataFrame, kstar: int) -> dict[str, Any]:
    e0, ck = "X-E0", f"C{kstar}"
    if not {e0, ck} <= set(ret.index):
        return dict(evaluable=False, reason=f"{e0} or {ck} missing")
    x, y = _row(ret, e0).to_numpy(), _row(ret, ck).to_numpy()
    if len(x) < 2 or len(y) < 2:
        return dict(evaluable=False, reason="fewer than 2 seeds")
    w = welch_t(x, y, "two-sided")
    pp = pooled_two_sample_t(x, y, "two-sided")
    res = dict(evaluable=True, estimate=float(w.estimate), p=float(w.p), df=float(w.df),
               ci95=[float(w.lower), float(w.upper)], pooled=dict(p=float(pp.p), df=float(pp.df)),
               tost=tost_welch(x, y, P.TOST_ACCURACY).to_dict(), n=[int(len(x)), int(len(y))])
    if "C0" in ret.index:
        z = _row(ret, "C0").to_numpy()
        nx, ny, nz = len(x), len(y), len(z)
        vx, vy, vz = x.var(ddof=1) / nx, y.var(ddof=1) / ny, z.var(ddof=1) / nz
        num, den = x.mean() - z.mean(), y.mean() - z.mean()
        dfw = (vx + vz) ** 2 / (vx ** 2 / (nx - 1) + vz ** 2 / (nz - 1))  # Welch df of the numerator
        res["share_of_gap_closed"] = fieller(num, den, vx + vz, vy + vz, vz, dfw).to_dict()
    return res


# =============================================================================================
# descriptive (H2, H6, N50)
# =============================================================================================
def descriptive(S: Mapping[str, pd.DataFrame], cs: CellTable, Tp: Sequence[int],
                draws: BootstrapDraws | None) -> dict[str, Any]:
    ret = S[RET]
    out: dict[str, Any] = {}
    curve = [c for c in P.CURVE_CONFIGS if c in ret.index]
    if {"C0", "C12"} <= set(curve) and len(curve) >= 3:
        ks = np.array([P.CONFIG_K[c] for c in curve])
        means = np.array([ret.loc[c].mean() for c in curve])
        n_s = np.array([ret.loc[c].notna().sum() for c in curve], float)
        r_iso = isotonic_recovery(ks, means, n_s)
        kk, ss = [], []
        for c in curve:
            v = ret.loc[c].dropna()
            kk += [P.CONFIG_K[c]] * len(v)
            ss += v.tolist()
        sat = saturating_fit(kk, ss)
        k80v = k80(ks, means, n_s)
        h2: dict[str, Any] = dict(configs=curve, k=ks.tolist(), mean=means.tolist(), R_isotonic=r_iso.tolist(),
                                  saturating=sat.to_dict(), k80=k80v,
                                  linear_or_convex=bool(np.all(np.abs(r_iso - ks / 12) <= P.H2_LINEAR_BAND)))
        m = dict(zip(curve, means))
        if "C1" in m:
            h2["g1"] = float(m["C1"] - m["C0"])
        if "C6" in m:
            h2["g_late"] = float((m["C12"] - m["C6"]) / 6)
        if draws is not None and f"{RET}@primary" in draws.scores:
            bm = np.stack([np.nanmean(draws.get(f"{RET}@primary", c), axis=1) for c in curve], axis=1)
            k80b = np.array([k80(ks, row, n_s) for row in bm])
            h2["k80_bootstrap_ci95"] = list(percentile_ci(k80b, 0.95))
        out["H2"] = h2
        G = float(m["C12"] - m["C0"])
        out["H6_hull_margin"] = {c: float(G * (recovery(m[c], m["C0"], m["C12"]) - P.CONFIG_K[c] / 12))
                                 for c in curve if c not in ("C0", "C12")}
    out["N50"] = n50_table(cs)
    return out


def n50_table(cs: CellTable) -> dict[str, Any]:
    """§9.7: logistic fit of EM on log2 N per model (config, seed) and length, R-load levels."""
    from .results import level_value
    res: dict[str, Any] = {}
    for T in cs.lengths():
        lv = [x for x in cs.levels_at("R-load", T)]
        if len(lv) < 3:
            continue
        idx = [cs.strata.index(("R-load", x, T)) for x in lv]
        N = [level_value(x) for x in lv]
        for ci, c in enumerate(cs.configs):
            for sj, s in enumerate(cs.seeds):
                if not cs.present[ci, sj]:
                    continue
                succ, tri = [], []
                for i in idx:
                    v = cs.ctx[i][:, ci, sj]
                    succ.append(float(np.nansum(v)) * P.Q_PER_CONTEXT)  # per-context means x Q
                    tri.append(float(np.isfinite(v).sum()) * P.Q_PER_CONTEXT)
                fit = n50_logistic(N, succ, tri)
                res.setdefault(str(T), {}).setdefault(c, {})[str(s)] = fit.to_dict()
    return res


# =============================================================================================
# Tier-A triggers and the F_gc family (§11.8, §11.9)
# =============================================================================================
def tier_a_triggers(ret: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if {"X-gc12", "C12"} <= set(ret.index):
        seeds = [s for s in P.U02_SEEDS if s in ret.columns and pd.notna(ret.loc["X-gc12", s])
                 and pd.notna(ret.loc["C12", s])]
        if seeds:
            d = float(np.mean([ret.loc["X-gc12", s] - ret.loc["C12", s] for s in seeds]))
            out["U02"] = dict(delta_gc=d, seeds=seeds, tier_b=bool(d > P.U02_TIER_B), co_primary=bool(d > P.U02_CO_PRIMARY))
    if {"X-nope3", "X-nope12", "C0", "C3", "C12"} <= set(ret.index) and 0 in ret.columns:
        s = 0
        r_nope = recovery(ret.loc["X-nope3", s], ret.loc["C0", s], ret.loc["X-nope12", s])
        r_rope = recovery(ret.loc["C3", s], ret.loc["C0", s], ret.loc["C12", s])
        dS12 = float(ret.loc["X-nope12", s] - ret.loc["C12", s])
        out["U06"] = dict(delta_R3=float(r_nope - r_rope), delta_S12=dS12,
                          fires=bool(abs(r_nope - r_rope) > P.U06_DELTA_R3 or abs(dS12) > P.U06_DELTA_S12))
    return out


def fgc_family(ret: pd.DataFrame) -> dict[str, Any]:
    """H1 / H1-min in the F_gc family (X-gc{1,3,12}), reported when U-02 fires (§11.8)."""
    if not {"X-gc12", "C0"} <= set(ret.index):
        return {}
    out = {}
    for name, k in (("H1", 3), ("H1-min", 1)):
        if f"X-gc{k}" in ret.index:
            out[name] = h1_test(ret, f"X-gc{k}", c12="X-gc12")
    if out:
        out["holm_support"] = holm({n: r.get("p_support", 1.0) or 1.0 for n, r in out.items()})
        out["holm_falsification"] = holm({n: r.get("p_falsify", 1.0) or 1.0 for n, r in out.items()
                                          if n in P.FALSIFICATION_FAMILY})
    return out


def dumps(results: Mapping[str, Any]) -> str:
    return json.dumps(results, indent=1, sort_keys=False, allow_nan=False)


__all__ = ["AnalysisSpec", "run_analysis", "gref_at", "select_primary", "h1_test", "h3_test", "h5a_test",
           "h5_slopes", "select_kstar", "dumps"]

_ = warnings  # kept for callers that filter analysis warnings

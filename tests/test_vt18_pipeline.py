"""VT-18 (part 3): the analysis pipeline recovers planted L_k, D3 and H5 slopes and applies the rules.

A synthetic world (``hybridlc.analysis.synthetic``) plants per-run stratum probabilities; the planted
value of every estimand is the analysis's own weight vector applied to the *true* probabilities, and
the tolerance is derived from the item-sampling SE (no empirical tuning).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from hybridlc.analysis import protocol as P
from hybridlc.analysis.pipeline import AnalysisSpec, h5_slopes, run_analysis
from hybridlc.analysis.results import load_item_results, load_registry
from hybridlc.analysis.scores import CellTable, family_spec
from hybridlc.analysis.stats import l_contrast, recovery, slope_log2
from hybridlc.analysis.synthetic import World, WorldSpec, default_plan, generate_dataframe, true_scores, write_tree
from hybridlc.analysis.verdicts import compute_verdicts

GATES = {g: True for g in P.REQUIRED_GATES}


def _world(**kw) -> tuple[World, pd.DataFrame]:
    spec = WorldSpec(**{"n_contexts": 150, **kw})
    w = World(spec)
    return w, generate_dataframe(w, default_plan(w))


@pytest.fixture(scope="module")
def base():
    w, df = _world()
    res = run_analysis(df, AnalysisSpec(B=1000, gates=GATES))
    return w, df, res, compute_verdicts(res)


def _sd_bound(w: World, cs: CellTable, wv: np.ndarray, coefs: dict[str, float], seed: int) -> float:
    """Upper bound on the item-sampling SD of sum_c coef_c * (wv @ cell means) for one seed.

    Var(cell mean) = p(1-p)(1 + (Q-1) rho) / (n_ctx Q) with rho = 1/(phi+1); configs share contexts, so
    the SD of the combination is bounded by sum_c |coef_c| SD_c (Cauchy–Schwarz)."""
    sp = w.spec
    rho = 1 / (sp.phi + 1)
    tot = 0.0
    for c, a in coefs.items():
        var = 0.0
        for wi, st in zip(wv, cs.strata):
            if wi:
                p = w.p_true(c, seed, *st)
                var += wi ** 2 * p * (1 - p) * (1 + (sp.n_queries - 1) * rho) / (sp.n_contexts * sp.n_queries)
        tot += abs(a) * np.sqrt(var)
    return tot


def test_primary_cell_and_gref_pass(base):
    _, _, res, ver = base
    assert res["primary"]["lengths"] == [16384, 32768] and not res["primary"]["fallback"]
    for T in ("16384", "32768"):
        assert res["gref"][T]["passes"] and res["gref"][T]["suite"] == "DG"
    assert ver["primary_cell"]["label"] is None


@pytest.mark.parametrize("name,k", [("H1", 3), ("H1-min", 1)])
def test_recovers_planted_L_k_per_seed(base, name, k):
    w, df, res, _ = base
    cs = CellTable.from_rows(df, "CS", "em_tf")
    Tp = res["primary"]["lengths"]
    wv = cs.ladder_weights(family_spec("retrieval"), Tp)
    seeds = list(w.spec.seeds)
    truth = true_scores(w, cs.strata, wv, ["C0", f"C{k}", "C12"], seeds)
    L_true = l_contrast(truth.loc[f"C{k}"], truth.loc["C0"], truth.loc["C12"])
    est = res["tests"][name]["per_seed"]
    for s in seeds:
        tol = 4 * _sd_bound(w, cs, wv, {f"C{k}": 1, "C0": 0.2, "C12": 0.8}, s)
        assert abs(est[str(s)] - L_true[s]) <= tol, (s, est[str(s)], L_true[s], tol)
    assert abs(res["tests"][name]["estimate"] - L_true.mean()) <= 0.01


def test_recovers_planted_D3(base):
    w, df, res, _ = base
    cs = CellTable.from_rows(df, "CS", "em_tf")
    Tp = res["primary"]["lengths"]
    seeds = list(w.spec.seeds)
    R = {}
    for fam in ("retrieval", "multihop"):
        t = true_scores(w, cs.strata, cs.ladder_weights(family_spec(fam), Tp), ["C0", "C3", "C12"], seeds)
        R[fam] = recovery(t.loc["C3"], t.loc["C0"], t.loc["C12"])
    D_true = (R["retrieval"] - R["multihop"]).mean()
    assert res["tests"]["H3"]["evaluable"]
    assert res["tests"]["H3"]["estimate"] == pytest.approx(D_true, abs=0.05)


def test_recovers_planted_h5_slope(base):
    w, df, res, _ = base
    cs = CellTable.from_rows(df, "CS", "em_tf")
    Ts = list(P.ALL_LENGTHS)
    seeds = list(w.spec.seeds)
    deltas = []
    for T in Ts:
        wv = cs.ladder_weights(family_spec("retrieval", P.H5_FIXED_LEVELS["retrieval"]), [T])
        t = true_scores(w, cs.strata, wv, ["C0", "C12"], seeds)
        deltas.append((t.loc["C12"] - t.loc["C0"]).to_numpy())
    b_true = slope_log2(Ts, np.stack(deltas, axis=1)).mean()
    # planted: gap grows by h5_slope per doubling (minus clipping, absent here)
    assert b_true == pytest.approx(w.spec.h5_slope, abs=0.003)
    assert res["tests"]["H5a"]["estimate"] == pytest.approx(b_true, abs=0.006)
    # the library slope helper and the pipeline agree
    assert h5_slopes(cs, "retrieval", Ts).mean() == pytest.approx(res["tests"]["H5a"]["estimate"], abs=1e-12)


def test_verdicts_on_planted_world(base):
    _, _, res, ver = base
    c = ver["confirmatory"]
    assert c["H1"]["verdict"] == "supported" and "metric-robust" in c["H1"]["labels"]
    assert c["H3"]["verdict"] == "supported"
    assert c["H5a"]["verdict"] == "supported"
    # H1-min is planted false (R1 = 0.45): the falsification test rejects after Holm, and the
    # sensitivity analyses agree in sign
    assert res["holm_falsification"]["H1-min"]["reject"]
    assert res["guards"]["H1-min"]["test"]["holds"]
    assert c["H1-min"]["verdict"] in ("falsified", "inconclusive")
    if c["H1-min"]["verdict"] == "inconclusive":  # only a guard may block it, and it must say which
        assert c["H1-min"]["notes"]
    # Holm over the support family uses exactly the four confirmatory hypotheses
    assert set(res["holm_support"]) == set(P.SUPPORT_FAMILY)
    assert set(res["holm_falsification"]) == set(P.FALSIFICATION_FAMILY)
    # sensitivity: pooled and bootstrap agree with H1's direction
    assert res["sensitivity"]["H1"]["pooled"]["agrees_greater"]
    assert res["sensitivity"]["H1"]["bootstrap"]["agrees_greater"]
    assert res["sensitivity"]["H1"]["pooled"]["df"] == 3 * (len(res["tests"]["H1"]["per_seed"]) - 1)
    json.dumps(res, allow_nan=False)
    json.dumps(ver, allow_nan=False)


def test_falsified_when_all_guards_hold():
    w, df = _world(n_contexts=400)
    res = run_analysis(df, AnalysisSpec(B=500, gates=GATES))
    ver = compute_verdicts(res)
    g = res["guards"]["H1-min"]
    assert res["guards"]["gates"]["holds"]
    assert g["test"]["holds"] and g["dose"]["holds"] and g["saturation"]["holds"]
    assert ver["confirmatory"]["H1-min"]["verdict"] == "falsified"
    # without the gate outcomes falsification is impossible
    res2 = run_analysis(df, AnalysisSpec(B=200, gates={}))
    assert compute_verdicts(res2)["confirmatory"]["H1-min"]["verdict"] == "inconclusive"


def test_saturation_guard_blocks_and_labels():
    w = World(WorldSpec(n_contexts=150))
    df = generate_dataframe(w, default_plan(w, saturation_shift=0.08))
    res = run_analysis(df, AnalysisSpec(B=200, gates=GATES))
    sat = res["guards"]["H1-min"]["saturation"]
    assert not sat["holds"] and sat["label"] == "conditional on the long-stage budget"
    ver = compute_verdicts(res)
    assert ver["confirmatory"]["H1-min"]["verdict"] != "falsified"
    assert "conditional on the long-stage budget" in ver["confirmatory"]["H1-min"]["labels"]


def test_gref_failure_moves_primary_cell():
    # S12 collapses at 32K only -> primary = 16K
    _, df = _world(n_contexts=80, s12_drop_32k=0.45)
    res = run_analysis(df, AnalysisSpec(bootstrap=False, gates=GATES))
    assert not res["gref"]["32768"]["passes"] and res["gref"]["16384"]["passes"]
    assert res["primary"]["lengths"] == [16384]
    # both long lengths unstable -> 8K with the pre-registered label
    _, df2 = _world(n_contexts=80, control_p=0.99, gap_base=0.15, h5_slope=0.0)
    res2 = run_analysis(df2, AnalysisSpec(bootstrap=False, gates=GATES))
    assert res2["primary"]["lengths"] == [P.FALLBACK_LENGTH]
    v = compute_verdicts(res2)
    assert "H1 at 8K, not a long-context result" in v["confirmatory"]["H1"]["labels"]
    assert not res2["guards"]["gates"]["holds"]  # G-REF at the primary length is part of guard 1


def test_null_length_effect_not_supported():
    _, df = _world(n_contexts=100, h5_slope=0.0)
    res = run_analysis(df, AnalysisSpec(bootstrap=False, gates=GATES))
    assert abs(res["tests"]["H5a"]["estimate"]) < 0.01
    assert not res["holm_support"]["H5a"]["reject"]


def test_h3_not_evaluable_enters_holm_with_p1():
    flat = {"retrieval": lambda k: {0: 0, 1: .45, 2: .7, 3: .95, 6: 1, 12: 1}[k],
            "multihop": lambda k: 0.0, "aggregation": lambda k: k / 12}
    spec = WorldSpec(n_contexts=60, recovery=flat, gap_base=0.45)
    w = World(spec)
    # multihop gap ~0 -> G_hop < 15 pp: make the C12 multihop gap vanish by zero recovery at k=12
    df = generate_dataframe(w, default_plan(w, dose=False, saturation=False))
    res = run_analysis(df, AnalysisSpec(bootstrap=False))
    assert not res["tests"]["H3"]["evaluable"]
    assert res["holm_support"]["H3"]["p"] == 1.0
    assert compute_verdicts(res)["confirmatory"]["H3"]["verdict"] in ("inconclusive", "no task dependence")


def test_seed_trigger_fires_on_noisy_and_bimodal_worlds():
    _, df = _world(n_contexts=60, resid_sd=0.002)
    assert run_analysis(df, AnalysisSpec(bootstrap=False))["seed_trigger"]["fires"] is False
    _, df = _world(n_contexts=60, resid_sd=0.05)
    tr = run_analysis(df, AnalysisSpec(bootstrap=False))["seed_trigger"]
    assert tr["fires"] is True and tr["resid_sd"] > P.TRIGGER_RESID_SD


def test_success_counts_see_bimodal_seeds():
    w, df = _world(n_contexts=60, resid_sd=0.002, bimodal={"C3": (0.5, 0.4)})
    res = run_analysis(df, AnalysisSpec(bootstrap=False))
    sc = res["success_counts"]["C3"]
    n_drop = sum(1 for s in w.spec.seeds if w.drop[("C3", s)] < 0)
    assert sc["n"] == len(w.spec.seeds)
    assert sc["successes"] == len(w.spec.seeds) - n_drop  # dropped seeds fall below the midpoint


def test_file_roundtrip_matches_in_memory(tmp_path):
    spec = WorldSpec(n_contexts=12, configs={"C0": 0, "C3": 3, "C12": 12}, seeds=(0, 1, 2))
    w = World(spec)
    plans = default_plan(w, dose=False, saturation=False)
    write_tree(w, plans, tmp_path)
    reg = load_registry(tmp_path / "registry.jsonl")
    df_f = load_item_results(tmp_path / "results", reg)
    df_m = generate_dataframe(w, plans)
    a = CellTable.from_rows(df_f, "CS", "em_tf")
    b = CellTable.from_rows(df_m, "CS", "em_tf")
    assert a.strata == b.strata and a.configs == b.configs and a.seeds == b.seeds
    np.testing.assert_allclose(a.cell_means(), b.cell_means())
    assert len(df_f.attrs["inputs"]) > 0
    # a re-scored file with a suffix is never read (§9.8)
    some = sorted((tmp_path / "results").rglob("R1_D3_16384.jsonl"))[0]
    (some.parent / "R1_D3_16384.r1.jsonl").write_text(some.read_text().replace('"em_tf":1', '"em_tf":0'))
    df_r = load_item_results(tmp_path / "results", reg)
    assert len(df_r.attrs["skipped_reruns"]) == 1 and len(df_r) == len(df_f)


def test_misaligned_contexts_are_rejected():
    _, df = _world(n_contexts=10, configs={"C0": 0, "C3": 3, "C12": 12}, seeds=(0, 1))
    bad = df.drop(df[(df["config"] == "C3") & (df["context_id"].str.endswith("-0")) & (df["suite"] == "CS")].index)
    with pytest.raises(ValueError, match="identical across models"):
        CellTable.from_rows(bad, "CS")

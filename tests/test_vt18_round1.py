"""VT-18 regressions from verification round 1 (DTS-02, DTS-03, DTS-04, DTS-05).

* DTS-02: the §7.3 "still closing" flag is computed (seed 0, S12 - S_k at the primary cell, TEF
  1,000 vs 4,000) and labels the verdict "budget-conditional"; CS source, DC fallback [IMPL].
* DTS-03: the N50 logistic fit never overflows (flat high-accuracy curves are right-censored) and
  a failure in a descriptive block cannot abort ``run_analysis``.
* DTS-04: a family at floor for every config (M-hop EM = 0) yields H3 "not evaluable", not a crash;
  ``percentile_ci`` of an all-NaN bootstrap is (nan, nan) and never "agrees".
* DTS-05: at T < 16K the H5(a) "conditional on reference adequacy" label uses adequacy (a) only.
"""
from __future__ import annotations

import numpy as np
import pytest

from hybridlc.analysis import protocol as P
from hybridlc.analysis.pipeline import AnalysisSpec, _agrees, run_analysis
from hybridlc.analysis.stats import n50_logistic, percentile_ci
from hybridlc.analysis.synthetic import World, WorldSpec, default_plan, generate_dataframe
from hybridlc.analysis.verdicts import compute_verdicts

GATES = {g: True for g in P.REQUIRED_GATES}


# ----------------------------------------------------------------------------- DTS-03
def test_n50_flat_curve_no_overflow():
    fit = n50_logistic([8, 32, 128, 512, 1024, 2048], [1020, 1021, 1020, 1019, 1020, 1019], [1200] * 6)
    assert fit.censoring == "right" and fit.slope < 0
    assert fit.log2_n50 > np.log2(2048) and (fit.n50 > 2048 or np.isinf(fit.n50))
    rng = np.random.default_rng(0)
    N = [8, 32, 128, 512, 1024, 2048]
    for p in (0.85, 0.97, 0.5, 0.03):
        for _ in range(300):
            f = n50_logistic(N, rng.binomial(1200, p, size=6), [1200] * 6)
            assert f.censoring in ("none", "left", "right", "undefined")
            if f.censoring == "right":
                assert f.n50 > 2048 or np.isinf(f.n50)
            if f.censoring == "left":
                assert f.n50 < 8


def test_flat_rload_world_does_not_abort():
    w = World(WorldSpec(n_contexts=60))
    df = generate_dataframe(w, default_plan(w, dose=False, saturation=False))
    rl = df["task"] == "R-load"
    rng = np.random.default_rng(1)
    df.loc[rl, "em_tf"] = (rng.random(int(rl.sum())) < 0.85).astype(float)  # accuracy independent of N
    res = run_analysis(df, AnalysisSpec(bootstrap=False, gates=GATES))
    assert "N50" in res["descriptive"]
    cens = [f["censoring"] for byc in res["descriptive"]["N50"].values() for bys in byc.values() for f in bys.values()]
    assert cens and set(cens) <= {"none", "left", "right", "undefined"}
    assert compute_verdicts(res)["confirmatory"]["H1"]["verdict"]


# ----------------------------------------------------------------------------- DTS-04
def test_percentile_ci_all_nan():
    lo, hi = percentile_ci(np.full(100, np.nan), 0.95)
    assert np.isnan(lo) and np.isnan(hi)
    lo, hi = percentile_ci(np.array([]), 0.90, "greater")
    assert np.isnan(lo)
    assert not _agrees(lo, hi, "greater") and not _agrees(lo, hi, "less")


def test_floor_multihop_world_h3_not_evaluable():
    w = World(WorldSpec(n_contexts=60))
    df = generate_dataframe(w, default_plan(w, dose=False, saturation=False))
    df.loc[df["task"] == "M-hop", "em_tf"] = 0.0  # every config at floor
    res = run_analysis(df, AnalysisSpec(B=200, gates=GATES))
    assert not res["tests"]["H3"]["evaluable"]
    assert res["holm_support"]["H3"]["p"] == 1.0
    assert "H3" not in res["sensitivity"]
    # verification round 2 (VERDICT-H3-NONEVALUABLE-NEGATIVE): never the named negative outcome
    assert compute_verdicts(res)["confirmatory"]["H3"]["verdict"] == "not evaluable"


# ----------------------------------------------------------------------------- DTS-02
@pytest.fixture(scope="module")
def closing_world():
    w = World(WorldSpec(n_contexts=300))
    return w


def test_still_closing_flags_planted_gap(closing_world):
    w = closing_world
    # planted: C3 loses 8 pp at TEF 4000 (S12 - S3 grows by ~8 pp); C1 unchanged
    df = generate_dataframe(w, default_plan(w, dose_shift={"C3": -0.08}))
    res = run_analysis(df, AnalysisSpec(B=200, gates=GATES))
    sc = res["still_closing"]
    assert sc["H1"]["evaluable"] and sc["H1"]["source"] == "CS"
    assert sc["H1"]["flag"] and sc["H1"]["delta"] > P.STILL_CLOSING
    assert not sc["H1-min"]["flag"]
    ver = compute_verdicts(res)["confirmatory"]
    assert P.STILL_CLOSING_LABEL in ver["H1"]["labels"]
    assert P.STILL_CLOSING_LABEL not in ver["H1-min"]["labels"]


def test_still_closing_no_shift_no_flag(closing_world):
    w = closing_world
    df = generate_dataframe(w, default_plan(w))
    res = run_analysis(df, AnalysisSpec(bootstrap=False, gates=GATES))
    for name in ("H1", "H1-min"):
        sc = res["still_closing"][name]
        assert sc["evaluable"] and not sc["flag"], sc
        assert abs(sc["delta"]) < P.STILL_CLOSING


def test_still_closing_dc_fallback_and_missing(closing_world):
    w = closing_world
    df = generate_dataframe(w, default_plan(w, dose_shift={"C1": +0.08}, dc_main=True))
    df = df[~((df["branch"] == "tef4000") & (df["suite"] == "CS"))].copy()  # branch CS not scored
    df.attrs.update(skipped_reruns=[], inputs=[])
    res = run_analysis(df, AnalysisSpec(bootstrap=False, gates=GATES))
    sc = res["still_closing"]["H1-min"]
    assert sc["evaluable"] and sc["source"] == "DC" and sc["impl_fallback"]
    assert sc["flag"] and sc["delta"] < -P.STILL_CLOSING  # C1 catches up: gap shrinks
    # no TEF-4000 branch at all -> not evaluable, noted, no label
    df2 = generate_dataframe(w, default_plan(w, dose=False, saturation=False))
    res2 = run_analysis(df2, AnalysisSpec(bootstrap=False, gates=GATES))
    assert not res2["still_closing"]["H1"]["evaluable"]
    v = compute_verdicts(res2)["confirmatory"]["H1"]
    assert P.STILL_CLOSING_LABEL not in v["labels"]
    assert any("still-closing" in n for n in v["notes"])


# ----------------------------------------------------------------------------- DTS-05
def test_h5a_label_uses_adequacy_only_below_16k():
    # gap grows from ~5 to ~30 pp; S12(T)/S12(C0) >= 0.8 at every T, so the reference is adequate
    # everywhere while discriminability (b) fails at 1K-2K.
    w = World(WorldSpec(n_contexts=150, gap_base=0.05, h5_slope=0.05, control_p=0.9, s0_base=0.7))
    df = generate_dataframe(w, default_plan(w, dose=False, saturation=False))
    res = run_analysis(df, AnalysisSpec(bootstrap=False, gates=GATES))
    g = res["gref"]
    short_b_fail = [T for T in ("1024", "2048") if not g[T]["b_pooled"]]
    assert short_b_fail, "world should make (b) fail at short T"
    assert all(g[T]["adequate"] for T in g)
    h5 = res["tests"]["H5a"]
    assert h5["failing_gref"] == [] and not h5["conditional_on_reference_adequacy"]
    assert "conditional on reference adequacy" not in compute_verdicts(res)["confirmatory"]["H5a"]["labels"]

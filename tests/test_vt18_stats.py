"""VT-18 (part 1): statistical primitives of protocol §11 against scipy / closed forms."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps

from hybridlc.analysis import stats as S
from hybridlc.analysis.opchar import brown_forsythe
from hybridlc.analysis.scores import CellTable

RNG = np.random.default_rng(20261109)


def test_one_sample_t_matches_scipy():
    x = RNG.normal(0.02, 0.03, size=(50, 5))
    for row in x:
        ours = S.one_sample_t(row, "greater")
        ref = sps.ttest_1samp(row, 0.0, alternative="greater")
        assert ours.p == pytest.approx(ref.pvalue, rel=1e-10)
        assert ours.df == 4
    lo = S.one_sample_t(x, "less")
    ref = sps.ttest_1samp(x, 0.0, axis=1, alternative="less")
    np.testing.assert_allclose(lo.p, ref.pvalue, rtol=1e-10)


def test_welch_and_pooled_match_scipy():
    a, b = RNG.normal(0, 1, 2), RNG.normal(0.5, 2, 5)
    w = S.welch_t(a, b)
    assert w.p == pytest.approx(sps.ttest_ind(a, b, equal_var=False).pvalue, rel=1e-10)
    pp = S.pooled_two_sample_t(a, b)
    assert pp.p == pytest.approx(sps.ttest_ind(a, b, equal_var=True).pvalue, rel=1e-10)


def test_welch_contrast_reduces_to_welch_t():
    a, b = RNG.normal(0, 1, 4), RNG.normal(0.3, 1.5, 6)
    wc = S.welch_contrast([a, b], [1.0, -1.0], "two-sided")
    w = S.welch_t(a, b, "two-sided")
    assert float(wc.p) == pytest.approx(float(w.p), rel=1e-12)
    assert float(wc.df) == pytest.approx(float(w.df), rel=1e-12)


def test_holm_and_bh_known_values():
    h = S.holm({"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.005})
    # sorted: d .005*4=.02, a .01*3=.03, c .03*2=.06, b .04*1 -> max(.06,.04)=.06
    assert h["d"]["p_holm"] == pytest.approx(0.02)
    assert h["a"]["p_holm"] == pytest.approx(0.03)
    assert h["c"]["p_holm"] == pytest.approx(0.06)
    assert h["b"]["p_holm"] == pytest.approx(0.06)
    assert [h[k]["reject"] for k in "abcd"] == [True, False, False, True]
    bh = S.bh_fdr({"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.005}, q=0.10)
    assert all(v["reject"] for v in bh.values())


def test_clopper_pearson_matches_scipy():
    for k, n in [(0, 5), (3, 5), (5, 5), (4, 7)]:
        lo, hi = S.clopper_pearson(k, n)
        ci = sps.binomtest(k, n).proportion_ci(0.95, method="exact")
        assert lo == pytest.approx(ci.low, abs=1e-12)
        assert hi == pytest.approx(ci.high, abs=1e-12)


def test_success_counts_midpoint_and_failures():
    tab = pd.DataFrame({0: [0.4, 0.5, 0.7], 1: [0.42, 0.60, 0.71], 2: [0.38, 0.52, 0.69]},
                       index=["C0", "C3", "C12"])
    sc = S.success_counts(tab, failed={"C3": 1})
    mid = 0.5 * (0.4 + 0.7)
    assert sc["C3"]["midpoint"] == pytest.approx(mid)
    assert sc["C3"]["successes"] == 1 and sc["C3"]["n"] == 4  # 0.60 > 0.55 only; failed run counts as failure
    assert sc["C12"]["successes"] == 3


def test_tost():
    x = np.array([0.01, -0.01, 0.005, 0.0, -0.004])
    assert S.tost_one_sample(x, 0.05).equivalent
    assert not S.tost_one_sample(x + 0.06, 0.05).equivalent


def test_fieller_reduces_to_t_interval_when_denominator_is_exact():
    a, va, b = 0.3, 0.01 ** 2, 2.0
    f = S.fieller(a, b, va, 0.0, 0.0, df=4)
    q = sps.t.ppf(0.975, 4)
    assert f.kind == "bounded"
    assert f.lower == pytest.approx((a - q * 0.01) / b)
    assert f.upper == pytest.approx((a + q * 0.01) / b)
    # unbounded when the denominator is not significantly non-zero
    assert S.fieller(0.3, 0.01, 1e-4, 1.0, 0.0, df=4).kind in ("unbounded", "exclusive")


def test_isotonic_k80_and_saturating():
    assert S.isotonic_increasing([1, 3, 2, 4]).tolist() == [1, 2.5, 2.5, 4]
    ks = [0, 1, 2, 3, 6, 12]
    means = [0.2, 0.5, 0.45, 0.6, 0.65, 0.7]  # violation at k=2 pooled with k=1
    r = S.isotonic_recovery(ks, means)
    assert r[0] == 0 and r[-1] == 1 and np.all(np.diff(r) >= 0)
    # k80: R at k=3 = 0.8 exactly
    assert S.k80(ks, means) == pytest.approx(3.0)
    kk = np.repeat(ks, 5).astype(float)
    true = S.saturating_curve(kk, 0.2, 0.5, 2.0)
    fit = S.saturating_fit(kk, true + RNG.normal(0, 1e-4, kk.size))
    assert fit.kappa == pytest.approx(2.0, rel=0.02) and fit.g == pytest.approx(0.5, abs=1e-3)


def test_n50_logistic_recovers_planted():
    N = np.array([8, 32, 128, 512, 1024, 2048], float)
    a, b = 6.0, -1.0  # N50 = 2^6 = 64
    m = np.full(N.size, 100000.0)
    y = m * (1 / (1 + np.exp(-(a + b * np.log2(N)))))
    fit = S.n50_logistic(N, y, m)
    assert fit.n50 == pytest.approx(64.0, rel=1e-3) and fit.censoring == "none"
    right = S.n50_logistic(N, m * 0.99, m)
    assert right.censoring in ("right", "undefined")


def test_pooled_contrast_complete_table():
    n = 5
    tab = pd.DataFrame(RNG.normal(0.5, 0.03, (4, n)), index=["C0", "C1", "C3", "C12"], columns=range(n))
    con = {"C3": 1.0, "C0": -0.2, "C12": -0.8}
    pc = S.pooled_contrast(tab, con, "greater")
    assert pc.df == 3 * (n - 1)
    L = tab.loc["C3"] - 0.2 * tab.loc["C0"] - 0.8 * tab.loc["C12"]
    assert float(pc.estimate) == pytest.approx(L.mean(), abs=1e-12)
    # residual SD of the additive fit equals the U-05 statistic
    assert float(pc.resid_sd) == pytest.approx(float(S.additive_residual_sd(tab.to_numpy())), rel=1e-10)
    # se = s * sqrt(1.68 / n)
    assert float(pc.se) == pytest.approx(float(pc.resid_sd) * np.sqrt(1.68 / n), rel=1e-10)


def test_seed_trigger_rules():
    base = pd.DataFrame(np.add.outer([0.2, 0.4, 0.6, 0.7], RNG.normal(0, 0.02, 5)),
                        index=["C0", "C1", "C3", "C12"], columns=range(5))
    assert not S.seed_trigger(base).fires  # pure additive table: residual SD 0, ranges ~ seed spread
    noisy = base.copy()
    noisy.loc["C3", 2] -= 0.13  # one bimodal seed: range > 12 pp
    tr = S.seed_trigger(noisy)
    assert tr.fires and any("range" in r for r in tr.reason)


def test_brown_forsythe_matches_scipy_levene():
    groups = [RNG.normal(0, s, (30, n)) for s, n in [(1, 3), (2, 3), (1, 2), (3, 2)]]
    ours = brown_forsythe(groups)
    for i in range(30):
        ref = sps.levene(*[g[i] for g in groups], center="median").pvalue
        assert ours[i] == pytest.approx(ref, rel=1e-8)


def test_paired_t_power_exact():
    assert S.paired_t_power(0.06, 0.03 * np.sqrt(1.68), 5) == pytest.approx(0.878, abs=5e-4)


def _toy_table(n_ctx=40, configs=("C0", "C3", "C12"), seeds=(0, 1, 2), noise=True):
    strata = [("R1", "D3", 16384), ("R-load", "N8", 16384)]
    ctx = []
    for _ in strata:
        a = np.empty((n_ctx, len(configs), len(seeds)))
        for i in range(len(configs)):
            for j in range(len(seeds)):
                a[:, i, j] = 0.2 + 0.25 * i + 0.01 * j + (RNG.normal(0, 0.2, n_ctx) if noise else 0)
        ctx.append(a)
    return CellTable("CS", "em_tf", strata, list(configs), list(seeds), ctx,
                     [np.arange(n_ctx)] * len(strata), np.ones((len(configs), len(seeds)), bool))


def test_hierarchical_bootstrap_common_indices():
    tab = _toy_table()
    # duplicate C3 as an extra config: identical draws iff the same seed and context indices are used
    tab2 = CellTable(tab.suite, tab.metric, tab.strata, tab.configs + ["C3b"], tab.seeds,
                     [np.concatenate([a, a[:, 1:2]], axis=1) for a in tab.ctx], tab.context_ids,
                     np.ones((4, 3), bool))
    w = tab2.ladder_weights({"R1": None, "R-load": None}, [16384])
    d = S.hierarchical_bootstrap(tab2, {"ret": w}, B=300, key="t")
    np.testing.assert_array_equal(d.get("ret", "C3"), d.get("ret", "C3b"))
    # deterministic given entropy/key
    d2 = S.hierarchical_bootstrap(tab2, {"ret": w}, B=300, key="t")
    np.testing.assert_array_equal(d.scores["ret"], d2.scores["ret"])


def test_hierarchical_bootstrap_without_context_noise_resamples_seeds_only():
    tab = _toy_table(noise=False)
    w = tab.ladder_weights({"R1": None, "R-load": None}, [16384])
    d = S.hierarchical_bootstrap(tab, {"ret": w}, B=200, key="x")
    vals = d.get("ret", "C0")
    # with constant contexts, every slot equals one of the three seed scores exactly
    seed_scores = {round(0.2 + 0.01 * j, 10) for j in range(3)}
    assert set(np.round(np.unique(vals), 10)) <= seed_scores

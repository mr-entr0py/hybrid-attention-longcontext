"""Operating characteristics of the H1 decision rules, computed with the analysis library (VT-18).

This module re-implements, *on top of* :mod:`hybridlc.analysis.stats`, the Monte Carlo set-ups of the
reviewer and PI simulations archived in ``scripts/review_measurements/``:

* ``pi/h1_stats_pi.py`` and ``pi/h1_adapt57.py`` (debate record §6.2, "PI-SIM"): POOL / PAIR /
  ADAPT n0 -> n0 + 2 on the 4 x n core table, scenarios homo / hetero / bimodal / bimodal13;
* ``revB2/stats_sim.py`` (Reviewer B round 2, W3): Reviewer A's 22-run Tier-1 layout, pooled df 11,
  A's Brown–Forsythe -> Welch rule, paired 3 and 5 seeds;
* ``revB2/power_n.py``: exact noncentral-t power of the paired test;
* ``sims/h1_rule_power.py`` and ``sims/h1_rule_power_df6.py`` (Reviewer A round 2, App. B): the
  B/PROP rule on per-seed R and Delta, and A's L3 rule with pooled df 11 / blocked df 6;
* ``sims/h5_ladder_artifact.py``: the deterministic ladder-composition artefact of the H5 slope.

The model is the one in each script: ``S[k, s] = mu_k + b_s + e_ks`` with scores on the probability
scale here (the scripts use pp; every pp constant is multiplied by :data:`~.protocol.PP`). The decision
functions are the library's own (``one_sample_t``, ``ols_contrast`` with ``additive_design``,
``seed_trigger_batch``, ``welch_contrast``, ``CellTable.ladder_weights`` / ``slope_log2``), so a match
with the scripts' numbers verifies the code that the real analysis runs.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from scipy import stats as sps

from .protocol import ALPHA_FW, PP
from .stats import (additive_design, ols_contrast, one_sample_t, paired_t_power, seed_trigger_batch,
                    slope_log2, welch_contrast)

OPCHAR_ENTROPY: int = 20261109
"""Entropy of the operating-characteristics RNG (the PI-SIM's own seed); fixed before any comparison."""

CORE_KS: tuple[int, ...] = (0, 1, 3, 12)
L3_COEF = np.array([-0.2, 0.0, 1.0, -0.8])  # over CORE_KS: L3 = S3 - 0.2 S0 - 0.8 S12


# =============================================================================================
# PI-SIM (debate §6.2): 4 x n core table
# =============================================================================================
def draw_core(rng: np.random.Generator, scenario: str, R3: float, n: int, NT: int, sigma: float, *,
              tau: float = 2 * PP, G: float = 30 * PP, mu0: float = 40 * PP, hetero_sd: float = 8 * PP,
              drop: float = 25 * PP, p_drop: float = 1 / 3) -> np.ndarray:
    """Draw ``(NT, 4, n)`` core tables exactly as ``pi/h1_stats_pi.py::draw`` (probability scale).

    * ``homo``: residual SD ``sigma`` for every k;
    * ``hetero``: SD 8 pp at k = 3 only;
    * ``bimodal``: each k = 3 seed drops by 25 pp w.p. 1/3 (mean held at target by adding 25/3 pp);
    * ``bimodal13``: the same at k = 1 and k = 3.
    """
    mu = {0: mu0, 1: mu0 + 0.5 * G, 3: mu0 + R3 * G, 12: mu0 + G}
    sd = {k: sigma for k in CORE_KS}
    bim: set[int] = set()
    if scenario == "hetero":
        sd[3] = hetero_sd
    elif scenario == "bimodal":
        bim = {3}
    elif scenario == "bimodal13":
        bim = {1, 3}
    elif scenario != "homo":
        raise ValueError(scenario)
    b = tau * rng.standard_normal((NT, n))
    S = np.empty((NT, 4, n))
    for i, k in enumerate(CORE_KS):
        m = mu[k] + (drop * p_drop if k in bim else 0.0)
        x = m + sd[k] * rng.standard_normal((NT, n)) + b
        if k in bim:
            x -= drop * (rng.random((NT, n)) < p_drop)
        S[:, i, :] = x
    return S


def _core_design(n: int) -> np.ndarray:
    cells = [(i, j) for i in range(4) for j in range(n)]  # row order = S[:, i, j] flattened
    return additive_design(cells, 4, n)


def pooled_core_test(S: np.ndarray, direction: str = "greater", alpha: float = ALPHA_FW):
    """POOL: seed-blocked additive OLS over the 4 x n core table, df = 3(n - 1) (A2; §11.3)."""
    NT, K, n = S.shape
    X = _core_design(n)
    c = np.zeros(X.shape[1])
    c[:4] = L3_COEF
    return ols_contrast(S.reshape(NT, K * n), X, c, direction, alpha=alpha)  # type: ignore[arg-type]


def paired_core_test(S: np.ndarray, direction: str = "greater", alpha: float = ALPHA_FW):
    """PAIR: one-sample t on per-seed L3 (df = n - 1; §11.3)."""
    L = S[:, 2] - 0.2 * S[:, 0] - 0.8 * S[:, 3]
    return one_sample_t(L, direction, alpha=alpha)  # type: ignore[arg-type]


def adaptive_support(S_full: np.ndarray, n0: int, alpha: float = ALPHA_FW) -> tuple[np.ndarray, np.ndarray]:
    """ADAPT n0 -> n0 + 2: U-05 trigger on the first n0 seeds, then PAIR on the final n (§11.4).

    Returns ``(support, trigger)`` boolean arrays of shape ``(NT,)``.
    """
    trig = seed_trigger_batch(S_full[:, :, :n0])
    small = paired_core_test(S_full[:, :, :n0], alpha=alpha).p < alpha
    big = paired_core_test(S_full, alpha=alpha).p < alpha
    return np.where(trig, big, small), trig


def pi_sim(NT: int = 20_000, *, rng: np.random.Generator | None = None,
           sigmas: Iterable[float] = (2.0, 3.0), scenarios: Iterable[str] = ("homo", "hetero", "bimodal", "bimodal13"),
           R3s: Iterable[float] = (0.8, 0.9, 1.0), ns: Iterable[int] = (3, 4, 5),
           adapt_n0: Iterable[int] = (3, 4, 5)) -> dict[tuple, float]:
    """All PI-SIM cells. Keys ``("pi", sigma_pp, scenario, R3, label)``; labels ``POOL_n3``, ``PAIR_n3``,
    ``ADAPT_3to5``, ``ADAPT_3to5_En`` (expected final n), ...; values are P(support)."""
    rng = rng or np.random.default_rng(OPCHAR_ENTROPY)
    out: dict[tuple, float] = {}
    for sig in sigmas:
        for scn in scenarios:
            for R3 in R3s:
                key = ("pi", float(sig), scn, float(R3))
                for n in ns:
                    S = draw_core(rng, scn, R3, n, NT, sig * PP)
                    out[key + (f"POOL_n{n}",)] = float((pooled_core_test(S).p < ALPHA_FW).mean())
                    out[key + (f"PAIR_n{n}",)] = float((paired_core_test(S).p < ALPHA_FW).mean())
                for n0 in adapt_n0:
                    S = draw_core(rng, scn, R3, n0 + 2, NT, sig * PP)
                    sup, trig = adaptive_support(S, n0)
                    out[key + (f"ADAPT_{n0}to{n0 + 2}",)] = float(sup.mean())
                    out[key + (f"ADAPT_{n0}to{n0 + 2}_En",)] = float(n0 + 2 * trig.mean())
    return out


# =============================================================================================
# Reviewer B round 2 (revB2/stats_sim.py): Reviewer A's 22-run Tier-1 layout
# =============================================================================================
TIER1_LAYOUT: tuple[tuple[str, int], ...] = (
    ("P0", 3), ("P1", 3), ("P3", 3), ("P12", 3), ("P2", 2), ("P6", 2), ("Ref12", 2), ("PE1", 2), ("PE12", 2))


def _tier1_means(R3: float, G: float, mu0: float) -> dict[str, float]:
    mu12 = mu0 + G
    return {"P0": mu0, "P12": mu12, "P1": mu0 + 0.5 * G, "P2": mu0 + 0.7 * G, "P6": mu12, "Ref12": mu12,
            "PE1": mu0 + 0.5 * G, "PE12": mu12, "P3": mu0 + R3 * G}


def draw_tier1(rng: np.random.Generator, scenario: str, R3: float, NT: int, *, sigma: float = 3 * PP,
               tau: float = 2 * PP, G: float = 30 * PP, mu0: float = 20 * PP, hetero_sd: float = 8 * PP,
               drop: float = 25 * PP, p_drop: float = 1 / 3, seed_sd: float | None = None
               ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Draw the 22-run layout plus 2 extra seeds of P0/P3/P12, as ``revB2/stats_sim.py``.

    Returns ``(data, extra)``: ``data[c]`` is ``(NT, n_c)`` on seeds 0..n_c-1; ``extra[c]`` is ``(NT, 2)``
    on seeds 3, 4 (for the 5-seed paired test). ``seed_sd`` overrides ``tau`` (used by the
    ``h1_rule_power`` replication, whose seed SD is 3 pp).
    """
    tau = tau if seed_sd is None else seed_sd
    mu = _tier1_means(R3, G, mu0)
    sd = {c: sigma for c, _ in TIER1_LAYOUT}
    bim: set[str] = set()
    if scenario == "hetero":
        sd["P3"] = hetero_sd
    elif scenario == "bimodal":
        bim = {"P3"}
    elif scenario == "bimodal2":
        bim = {"P3", "P1"}
    elif scenario != "homo":
        raise ValueError(scenario)
    seeds5 = tau * rng.standard_normal((NT, 5))

    def draw(c: str, n: int) -> np.ndarray:
        m = mu[c] + (p_drop * drop if c in bim else 0.0)
        x = m + sd[c] * rng.standard_normal((NT, n))
        if c in bim:
            x -= drop * (rng.random((NT, n)) < p_drop)
        return x

    data = {c: draw(c, n) + seeds5[:, :n] for c, n in TIER1_LAYOUT}
    extra = {c: draw(c, 2) + seeds5[:, 3:5] for c in ("P0", "P3", "P12")}
    return data, extra


def tier1_pooled(data: dict[str, np.ndarray], direction: str = "greater", alpha: float = ALPHA_FW):
    """A's pooled additive model over the 22 runs (config + seed), df = 22 - 11 = 11."""
    names = [c for c, _ in TIER1_LAYOUT]
    n_seeds = max(n for _, n in TIER1_LAYOUT)
    cells = [(i, s) for i, (c, n) in enumerate(TIER1_LAYOUT) for s in range(n)]
    X = additive_design(cells, len(names), n_seeds)
    Y = np.concatenate([data[c] for c in names], axis=1)
    cvec = np.zeros(X.shape[1])
    cvec[names.index("P3")], cvec[names.index("P0")], cvec[names.index("P12")] = 1.0, -0.2, -0.8
    return ols_contrast(Y, X, cvec, direction, alpha=alpha)  # type: ignore[arg-type]


def brown_forsythe(groups: list[np.ndarray]) -> np.ndarray:
    """Brown–Forsythe (Levene, median-centred) p-value across groups, batched over the leading dim.

    Equivalent to ``scipy.stats.levene(*groups, center='median')`` row by row.
    """
    z = [np.abs(g - np.median(g, axis=-1, keepdims=True)) for g in groups]
    ns = np.array([g.shape[-1] for g in z], float)
    N, K = ns.sum(), len(z)
    zbar_i = np.stack([g.mean(-1) for g in z], -1)
    zbar = sum(g.sum(-1) for g in z) / N
    between = (ns * (zbar_i - zbar[..., None]) ** 2).sum(-1)
    within = sum(((g - g.mean(-1, keepdims=True)) ** 2).sum(-1) for g in z)
    with np.errstate(divide="ignore", invalid="ignore"):
        F = (N - K) / (K - 1) * between / within
    return sps.f.sf(F, K - 1, N - K)


def stats_sim(NT: int = 20_000, *, rng: np.random.Generator | None = None,
              scenarios: Iterable[str] = ("homo", "hetero", "bimodal", "bimodal2")) -> dict[tuple, float]:
    """All cells of Reviewer B's W3 table. Keys ``("revB2", scenario, R3, label)`` with labels
    ``pooled``, ``A_rule``, ``paired3``, ``paired5``, ``BF_fires``."""
    rng = rng or np.random.default_rng(OPCHAR_ENTROPY + 1)
    out: dict[tuple, float] = {}
    for scn in scenarios:
        for R3 in (0.8, 1.0):
            data, extra = draw_tier1(rng, scn, R3, NT)
            pooled = tier1_pooled(data).p < ALPHA_FW
            bf_p = brown_forsythe([data[c] for c, _ in TIER1_LAYOUT])
            welch = welch_contrast([data["P3"], data["P0"], data["P12"]], [1.0, -0.2, -0.8]).p < ALPHA_FW
            rule = np.where(bf_p < 0.05, welch, pooled)
            L3 = data["P3"] - 0.2 * data["P0"] - 0.8 * data["P12"]
            paired3 = one_sample_t(L3, "greater").p < ALPHA_FW
            d5 = {c: np.concatenate([data[c], extra[c]], axis=1) for c in ("P0", "P3", "P12")}
            L5 = d5["P3"] - 0.2 * d5["P0"] - 0.8 * d5["P12"]
            paired5 = one_sample_t(L5, "greater").p < ALPHA_FW
            key = ("revB2", scn, float(R3))
            out[key + ("pooled",)] = float(pooled.mean())
            out[key + ("A_rule",)] = float(rule.mean())
            out[key + ("paired3",)] = float(paired3.mean())
            out[key + ("paired5",)] = float(paired5.mean())
            out[key + ("BF_fires",)] = float((bf_p < 0.05).mean())
    return out


def power_n_exact(ns: Iterable[int] = (3, 4, 5, 6)) -> dict[tuple, float]:
    """``revB2/power_n.py``: exact paired-t power at R3 = 1.0 (L = 6 pp), sigma 3 pp, seed effects cancel."""
    sL = 3 * PP * np.sqrt(1 + 0.04 + 0.64)
    return {("power_n", n): paired_t_power(6 * PP, sL, n) for n in ns}


# =============================================================================================
# Reviewer A round 2 (sims/h1_rule_power*.py): B/PROP rule vs A's L3 rule
# =============================================================================================
def prop_rule(S0: np.ndarray, S3: np.ndarray, S12: np.ndarray, *, r_target: float = 0.8,
              ni_margin: float = 5 * PP) -> tuple[np.ndarray, np.ndarray]:
    """B/PROP rule (superseded design, replicated only for VT-18): two-sided 95% seed-level t (df = n-1)
    on per-seed R_s and Delta_s = S12_s - S3_s. Supported iff lower(R) >= 0.8 and upper(Delta) <= 5 pp;
    falsified iff upper(R) < 0.8."""
    with np.errstate(divide="ignore", invalid="ignore"):
        Rs = (S3 - S0) / (S12 - S0)
    tR = one_sample_t(Rs, "two-sided", alpha=0.05)
    tD = one_sample_t(S12 - S3, "two-sided", alpha=0.05)
    return (tR.lower >= r_target) & (tD.upper <= ni_margin), tR.upper < r_target


def h1_rule_power(NT: int = 20_000, *, df_pool: int = 6, rng: np.random.Generator | None = None,
                  sigma_ds: Iterable[float] = (3.0, 5.0),
                  Rs: Iterable[float] = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1)) -> dict[tuple, float]:
    """Reviewer A's rule-power table. ``df_pool = 6``: the blocked 4 x 3 core table (C1 added at
    S0 + G/2, which only contributes residual df); ``df_pool = 11``: A1's 22-run layout, homoscedastic
    at s_e. Seed SD 3 pp; s_e = sigma_d / sqrt(2); G = 30 pp; mu0 = 40 pp. The scripts draw s^2 from a
    scaled chi-square with the stated df; fitting the real design has exactly that distribution.

    Keys ``("rule_df{df}", sigma_d, R, label)``, labels ``B_sup``, ``B_fal``, ``A_sup``, ``A_fal``.
    """
    if df_pool not in (6, 11):
        raise ValueError("df_pool must be 6 or 11")
    rng = rng or np.random.default_rng(OPCHAR_ENTROPY + df_pool)
    out: dict[tuple, float] = {}
    G, mu0 = 30 * PP, 40 * PP
    for sd_d in sigma_ds:
        s_e = sd_d * PP / np.sqrt(2)
        for R in Rs:
            if df_pool == 6:
                b = 3 * PP * rng.standard_normal((NT, 3))
                mus = (mu0, mu0 + 0.5 * G, mu0 + R * G, mu0 + G)
                S = np.stack([m + b + s_e * rng.standard_normal((NT, 3)) for m in mus], axis=1)
                S0, S3, S12 = S[:, 0], S[:, 2], S[:, 3]
                fit_g = pooled_core_test(S, "greater")
                fit_l = pooled_core_test(S, "less")
                assert fit_g.df == 6
            else:
                # A1's 22-run layout (mu0 shifted to 40 pp to match h1_rule_power.py; extra configs only add df).
                data, _ = draw_tier1(rng, "homo", R, NT, sigma=s_e, mu0=mu0, seed_sd=3 * PP)
                S0, S3, S12 = data["P0"], data["P3"], data["P12"]
                fit_g = tier1_pooled(data, "greater")
                fit_l = tier1_pooled(data, "less")
                assert fit_g.df == 11
            bs, bf = prop_rule(S0, S3, S12)
            key = (f"rule_df{df_pool}", float(sd_d), float(R))
            out[key + ("B_sup",)] = float(bs.mean())
            out[key + ("B_fal",)] = float(bf.mean())
            out[key + ("A_sup",)] = float((fit_g.p < ALPHA_FW).mean())
            out[key + ("A_fal",)] = float((fit_l.p < ALPHA_FW).mean())
    return out


# =============================================================================================
# H5 ladder artefact (sims/h5_ladder_artifact.py), through the real CellTable weight code
# =============================================================================================
@dataclass
class LadderArtefact:
    r2_only: float
    r1_r2: float
    r1_r2_halved: float
    gaps_r2_only: list[float]
    gaps_r1_r2: list[float]


def h5_ladder_artefact() -> LadderArtefact:
    """Spurious H5 slope when the R-load level set grows with T (B1's rule 12N <= 0.75T), constant gaps.

    Builds a :class:`~.scores.CellTable` whose cell means *are* the per-level gaps, takes B1's
    equal-weight-per-level mean at each T through ``CellTable.scores`` and the slope with ``slope_log2``.
    (The protocol avoids this artefact by fixing the H5 level set, ``protocol.H5_FIXED_LEVELS``.)
    """
    from .scores import CellTable

    Ts = [1024, 2048, 4096, 8192, 16384, 32768]

    def run(gap_r2: dict[int, float], gap_r1: dict[int, float] | None) -> tuple[float, list[float]]:
        strata, ctx = [], []
        for T in Ts:
            for N, g in gap_r2.items():
                if 12 * N <= 0.75 * T:
                    strata.append(("R-load", f"N{N}", T))
                    ctx.append(np.full((1, 1, 1), g))
            for D, g in (gap_r1 or {}).items():
                strata.append(("R1", f"D{D}", T))
                ctx.append(np.full((1, 1, 1), g))
        tab = CellTable("synthetic", "gap", strata, ["gap"], [0], ctx, [np.array(["c0"])] * len(ctx),
                        np.ones((1, 1), bool))
        tasks = ("R-load",) if gap_r1 is None else ("R1", "R-load")
        # B1's rule pools R1 and R-load levels with equal weight *per level* (not per task).
        ys = []
        for T in Ts:
            lv = [(t, l) for (t, l, TT) in strata if TT == T and t in tasks]
            w = np.zeros(len(strata))
            for i, s in enumerate(strata):
                if s[2] == T:
                    w[i] = 1.0 / len(lv)
            ys.append(float(tab.scores(w).iloc[0, 0]))
        return float(slope_log2(Ts, np.array(ys))), ys

    s1, g1 = run({8: 10.0, 32: 20.0, 128: 40.0, 512: 60.0}, None)
    s2, g2 = run({8: 10.0, 32: 20.0, 128: 40.0, 512: 60.0}, {3: 30.0, 15: 30.0})
    s3, _ = run({8: 5.0, 32: 10.0, 128: 20.0, 512: 30.0}, {3: 15.0, 15: 15.0})
    return LadderArtefact(s1, s2, s3, g1, g2)


def run_all(NT: int = 20_000) -> dict[tuple, float]:
    """Every replicated cell (PI-SIM, revB2, h1_rule_power df 6 and 11, exact power)."""
    out: dict[tuple, float] = {}
    out.update(pi_sim(NT))
    out.update(stats_sim(NT))
    out.update(h1_rule_power(NT, df_pool=6))
    out.update(h1_rule_power(NT, df_pool=11))
    out.update(power_n_exact())
    return out

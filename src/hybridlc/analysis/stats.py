"""Statistical primitives of the pre-registered analysis plan (protocol §11).

All functions are pure. Most accept numpy arrays with leading batch dimensions, so the
operating-characteristics simulator (:mod:`.opchar`) runs exactly the code that the real analysis
runs, on 20,000+ replicates at once.

Scale: scores are proportions in [0, 1]; contrasts and slopes are in the same units.

Contents
--------
* Estimands: :func:`l_contrast` (L_k), :func:`recovery` (R_f(k)), :func:`d3`, :func:`slope_log2`.
* Tests: :func:`one_sample_t` (paired per-seed, df = n - 1), :func:`welch_t`, :func:`tost_one_sample`,
  :func:`tost_welch`, :func:`holm`, :func:`bh_fdr`.
* Seed-blocked pooled-variance model: :func:`ols_contrast`, :func:`pooled_contrast`.
* Seed trigger U-05: :func:`additive_residual_sd`, :func:`seed_trigger`.
* Success counts: :func:`clopper_pearson`, :func:`success_counts`.
* Ratios: :func:`fieller`, :func:`fieller_paired`.
* Curves: :func:`isotonic_increasing`, :func:`saturating_fit`, :func:`k80`, :func:`n50_logistic`.
* Hierarchical paired bootstrap: :func:`hierarchical_bootstrap`, :func:`percentile_ci`.
"""

from __future__ import annotations

import zlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from scipy import optimize, special
from scipy import stats as sps

from .protocol import (ALPHA_FW, BOOTSTRAP_B, BOOTSTRAP_ENTROPY, CI_LEVEL, L_LAYERS, L_WEIGHT_S0,
                       L_WEIGHT_S12, R_TARGET, TRIGGER_RESID_SD, TRIGGER_SEED_RANGE)

Direction = Literal["greater", "less", "two-sided"]


# =============================================================================================
# Estimands
# =============================================================================================
def l_contrast(s_k: np.ndarray, s_0: np.ndarray, s_12: np.ndarray) -> np.ndarray:
    """L_k = S_k - 0.2 S_0 - 0.8 S_12 (§2). With G > 0, L_k >= 0 iff R(k) >= 0.8.

    pandas inputs keep their index (aligned by seed); numpy inputs broadcast.
    """
    if any(isinstance(v, (pd.Series, pd.DataFrame)) for v in (s_k, s_0, s_12)):
        return s_k - L_WEIGHT_S0 * s_0 - L_WEIGHT_S12 * s_12
    return np.asarray(s_k) - L_WEIGHT_S0 * np.asarray(s_0) - L_WEIGHT_S12 * np.asarray(s_12)


L_CONTRAST_COEFS: dict[str, float] = {"k": 1.0, "0": -L_WEIGHT_S0, "12": -L_WEIGHT_S12}


def recovery(s_k: np.ndarray, s_0: np.ndarray, s_12: np.ndarray) -> np.ndarray:
    """R(k) = (S_k - S_0) / (S_12 - S_0); NaN where the gap is exactly 0. pandas inputs keep their index."""
    if any(isinstance(v, pd.Series) for v in (s_k, s_0, s_12)):
        g = s_12 - s_0
        return ((s_k - s_0) / g.where(g != 0)).astype(float)
    s_k, s_0, s_12 = map(np.asarray, (s_k, s_0, s_12))
    g = s_12 - s_0
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(g != 0, (s_k - s_0) / g, np.nan)


def d3(ret: Mapping[int, np.ndarray], hop: Mapping[int, np.ndarray], k: int = 3) -> np.ndarray:
    """D_3 = R_ret(3) - R_hop(3) per seed (§2 H3). ``ret``/``hop`` map k -> per-seed scores."""
    return recovery(ret[k], ret[0], ret[12]) - recovery(hop[k], hop[0], hop[12])


def slope_log2(T: Sequence[float], y: np.ndarray) -> np.ndarray:
    """OLS slope of ``y`` on log2 T along the last axis (units of y per doubling of T).

    NaN entries in ``y`` are not allowed (pass only the lengths to be used).
    """
    x = np.log2(np.asarray(T, float))
    xc = x - x.mean()
    y = np.asarray(y, float)
    return (y * xc).sum(-1) / (xc * xc).sum()


# =============================================================================================
# t-based tests
# =============================================================================================
@dataclass
class TTest:
    """Result of a t test. Array-valued when the input had batch dimensions."""

    estimate: np.ndarray
    se: np.ndarray
    df: np.ndarray
    t: np.ndarray
    p: np.ndarray
    """p-value in the requested direction."""
    lower: np.ndarray
    """One-sided (1 - alpha) lower bound for ``greater``; two-sided lower CI limit otherwise."""
    upper: np.ndarray
    direction: str
    n: int | np.ndarray

    def to_dict(self) -> dict:
        f = lambda v: (float(v) if np.ndim(v) == 0 else np.asarray(v).tolist())  # noqa: E731
        return dict(estimate=f(self.estimate), se=f(self.se), df=f(self.df), t=f(self.t), p=f(self.p),
                    lower=f(self.lower), upper=f(self.upper), direction=self.direction,
                    n=f(self.n))


def _p_and_bounds(est, se, df, direction: Direction, alpha: float):
    with np.errstate(divide="ignore", invalid="ignore"):
        t = est / se
    if direction == "greater":
        p = sps.t.sf(t, df)
        q = sps.t.ppf(1 - alpha, df)
        lower, upper = est - q * se, np.full_like(np.asarray(est, float), np.inf)
    elif direction == "less":
        p = sps.t.cdf(t, df)
        q = sps.t.ppf(1 - alpha, df)
        lower, upper = np.full_like(np.asarray(est, float), -np.inf), est + q * se
    elif direction == "two-sided":
        p = 2 * sps.t.sf(np.abs(t), df)
        q = sps.t.ppf(1 - alpha / 2, df)
        lower, upper = est - q * se, est + q * se
    else:  # pragma: no cover
        raise ValueError(direction)
    return t, p, lower, upper


def one_sample_t(x: np.ndarray, direction: Direction = "greater", *, null: float = 0.0,
                 alpha: float = ALPHA_FW, axis: int = -1) -> TTest:
    """One-sample t on per-seed values (the paired per-seed test, df = n - 1; §11.3).

    ``x`` holds per-seed contrasts (e.g. L_{k,s}); NaNs are not allowed.
    """
    x = np.moveaxis(np.asarray(x, float), axis, -1)
    n = x.shape[-1]
    if n < 2:
        raise ValueError("need at least 2 seeds")
    est = x.mean(-1) - null
    se = x.std(-1, ddof=1) / np.sqrt(n)
    df = np.full_like(est, n - 1.0)
    t, p, lo, hi = _p_and_bounds(est, se, df, direction, alpha)
    return TTest(est + null, se, df, t, p, lo + null, hi + null, direction, n)


def welch_t(x: np.ndarray, y: np.ndarray, direction: Direction = "two-sided", *,
            alpha: float = ALPHA_FW) -> TTest:
    """Welch t on mean(x) - mean(y) with Welch–Satterthwaite df (H6′; §11.7)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    nx, ny = x.shape[-1], y.shape[-1]
    vx, vy = x.var(-1, ddof=1) / nx, y.var(-1, ddof=1) / ny
    est = x.mean(-1) - y.mean(-1)
    se = np.sqrt(vx + vy)
    with np.errstate(divide="ignore", invalid="ignore"):
        df = (vx + vy) ** 2 / (vx ** 2 / (nx - 1) + vy ** 2 / (ny - 1))
    t, p, lo, hi = _p_and_bounds(est, se, df, direction, alpha)
    return TTest(est, se, df, t, p, lo, hi, direction, nx + ny)


def welch_contrast(groups: Sequence[np.ndarray], coefs: Sequence[float], direction: Direction = "greater", *,
                   alpha: float = ALPHA_FW) -> TTest:
    """Unpaired linear contrast ``sum_i c_i mean(x_i)`` with Welch–Satterthwaite df.

    Each group has shape ``(..., n_i)`` (batch dims shared). Used for Reviewer A's Welch fallback in the
    operating-characteristics replication and available as a heteroscedastic sensitivity check.
    """
    parts = [(float(c), np.asarray(x, float)) for c, x in zip(coefs, groups)]
    v = [c * c * x.var(-1, ddof=1) / x.shape[-1] for c, x in parts]
    vs = sum(v)
    se = np.sqrt(vs)
    with np.errstate(divide="ignore", invalid="ignore"):
        df = vs ** 2 / sum(vi ** 2 / (x.shape[-1] - 1) for vi, (_, x) in zip(v, parts))
    est = sum(c * x.mean(-1) for c, x in parts)
    t, p, lo, hi = _p_and_bounds(est, se, df, direction, alpha)
    return TTest(est, se, df, t, p, lo, hi, direction, sum(x.shape[-1] for _, x in parts))


def paired_t_power(delta: float, sd: float, n: int, alpha: float = ALPHA_FW) -> float:
    """Exact power of the one-sided one-sample t test (noncentral t), true mean ``delta``, SD ``sd``."""
    ncp = delta / (sd / np.sqrt(n))
    tc = sps.t.ppf(1 - alpha, n - 1)
    return float(sps.nct.sf(tc, n - 1, ncp))


def pooled_two_sample_t(x: np.ndarray, y: np.ndarray, direction: Direction = "two-sided", *,
                        alpha: float = ALPHA_FW) -> TTest:
    """Classical pooled-variance two-sample t (H6′ sensitivity version)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    nx, ny = x.shape[-1], y.shape[-1]
    sp2 = ((nx - 1) * x.var(-1, ddof=1) + (ny - 1) * y.var(-1, ddof=1)) / (nx + ny - 2)
    est = x.mean(-1) - y.mean(-1)
    se = np.sqrt(sp2 * (1 / nx + 1 / ny))
    df = np.full_like(np.asarray(est, float), nx + ny - 2.0)
    t, p, lo, hi = _p_and_bounds(est, se, df, direction, alpha)
    return TTest(est, se, df, t, p, lo, hi, direction, nx + ny)


@dataclass
class TOST:
    """Two one-sided tests for equivalence within ``(-margin, +margin)``."""

    estimate: float
    margin: float
    p_lower: float
    p_upper: float
    p: float
    ci90: tuple[float, float]
    equivalent: bool

    def to_dict(self) -> dict:
        return dict(estimate=float(self.estimate), margin=self.margin, p_lower=float(self.p_lower),
                    p_upper=float(self.p_upper), p=float(self.p), ci90=[float(c) for c in self.ci90],
                    equivalent=bool(self.equivalent))


def _tost(est, se, df, margin, alpha):
    p_lo = sps.t.sf((est + margin) / se, df)  # H0: mu <= -margin
    p_hi = sps.t.cdf((est - margin) / se, df)  # H0: mu >= +margin
    q = sps.t.ppf(1 - alpha, df)
    p = max(float(p_lo), float(p_hi))
    return TOST(float(est), margin, float(p_lo), float(p_hi), p, (float(est - q * se), float(est + q * se)),
                p < alpha)


def tost_one_sample(x: np.ndarray, margin: float, *, alpha: float = ALPHA_FW) -> TOST:
    """Paired TOST on per-seed differences (H3 ±0.10, H4 ±5 pp, H5 ±2 pp/doubling)."""
    x = np.asarray(x, float)
    n = x.size
    return _tost(x.mean(), x.std(ddof=1) / np.sqrt(n), n - 1, margin, alpha)


def tost_welch(x: np.ndarray, y: np.ndarray, margin: float, *, alpha: float = ALPHA_FW) -> TOST:
    """Welch TOST on mean(x) - mean(y) (H6′ ±5 pp)."""
    w = welch_t(x, y, "two-sided", alpha=alpha)
    return _tost(float(w.estimate), float(w.se), float(w.df), margin, alpha)


# =============================================================================================
# Multiplicity
# =============================================================================================
def holm(pvals: Mapping[str, float], alpha: float = ALPHA_FW) -> dict[str, dict[str, float | bool]]:
    """Holm step-down adjustment. Returns ``{name: {"p": raw, "p_holm": adjusted, "reject": bool}}``.

    Adjusted p = running max over the ordered sequence of ``(m - i) * p_(i)``, capped at 1;
    ``reject`` iff adjusted p < alpha (equivalent to the sequential procedure).
    """
    names = list(pvals)
    p = np.array([float(pvals[n]) for n in names])
    m = len(p)
    order = np.argsort(p, kind="stable")
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return {n: {"p": float(p[i]), "p_holm": float(adj[i]), "reject": bool(adj[i] < alpha)}
            for i, n in enumerate(names)}


def bh_fdr(pvals: Mapping[str, float], q: float = 0.10) -> dict[str, dict[str, float | bool]]:
    """Benjamini–Hochberg step-up (exploratory family, q = 0.10)."""
    names = list(pvals)
    p = np.array([float(pvals[n]) for n in names])
    m = len(p)
    order = np.argsort(p, kind="stable")
    ranked = p[order] * m / np.arange(1, m + 1)
    adj_sorted = np.minimum.accumulate(ranked[::-1])[::-1].clip(max=1.0)
    adj = np.empty(m)
    adj[order] = adj_sorted
    return {n: {"p": float(p[i]), "p_bh": float(adj[i]), "reject": bool(adj[i] <= q)}
            for i, n in enumerate(names)}


# =============================================================================================
# Seed-blocked pooled-variance model (A2; §11.3 sensitivity) and U-05 trigger
# =============================================================================================
def additive_design(cells: Sequence[tuple[int, int]], n_configs: int, n_seeds: int) -> np.ndarray:
    """Design matrix for ``S ~ config + seed``: one indicator per config, seed dummies for seeds 1..

    ``cells`` lists the (config_index, seed_index) of every observation (row).
    """
    X = np.zeros((len(cells), n_configs + n_seeds - 1))
    for r, (c, s) in enumerate(cells):
        X[r, c] = 1.0
        if s > 0:
            X[r, n_configs + s - 1] = 1.0
    return X


@dataclass
class OLSContrast:
    estimate: np.ndarray
    se: np.ndarray
    df: int
    resid_sd: np.ndarray
    t: np.ndarray
    p: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    direction: str

    def to_dict(self) -> dict:
        f = lambda v: (float(v) if np.ndim(v) == 0 else np.asarray(v).tolist())  # noqa: E731
        return dict(estimate=f(self.estimate), se=f(self.se), df=int(self.df), resid_sd=f(self.resid_sd),
                    t=f(self.t), p=f(self.p), lower=f(self.lower), upper=f(self.upper),
                    direction=self.direction)


def ols_contrast(Y: np.ndarray, X: np.ndarray, c: np.ndarray, direction: Direction = "greater", *,
                 alpha: float = ALPHA_FW) -> OLSContrast:
    """Contrast ``c @ beta`` of an OLS fit with a pooled residual variance, batched over Y's leading dims.

    ``Y`` has shape ``(..., N)``; ``X`` is ``(N, p)`` and shared across the batch.
    """
    Y = np.asarray(Y, float)
    pinv = np.linalg.pinv(X)
    beta = Y @ pinv.T
    resid = Y - beta @ X.T
    rank = np.linalg.matrix_rank(X)
    df = X.shape[0] - rank
    if df < 1:
        raise ValueError("no residual degrees of freedom")
    s2 = (resid ** 2).sum(-1) / df
    XtXi = np.linalg.pinv(X.T @ X)
    est = beta @ c
    se = np.sqrt(s2 * float(c @ XtXi @ c))
    t, p, lo, hi = _p_and_bounds(est, se, np.full_like(est, float(df)), direction, alpha)
    return OLSContrast(est, se, df, np.sqrt(s2), t, p, lo, hi, direction)


def pooled_contrast(table: pd.DataFrame, contrast: Mapping[str, float], direction: Direction = "greater",
                    *, alpha: float = ALPHA_FW) -> OLSContrast:
    """Seed-blocked pooled-variance test of a zero-sum config contrast on a config x seed table.

    The fit uses every non-missing cell of ``table`` (rows = configs, columns = seeds). For the
    complete 4 x n core table this is A2's model with df = 3(n - 1) (§11.3).
    """
    if abs(sum(contrast.values())) > 1e-9:
        raise ValueError("pooled_contrast needs a contrast whose coefficients sum to 0")
    configs = list(table.index)
    seeds = list(table.columns)
    vals = table.to_numpy(float)
    cells = [(i, j) for i in range(len(configs)) for j in range(len(seeds)) if np.isfinite(vals[i, j])]
    X = additive_design(cells, len(configs), len(seeds))
    # drop seed dummies of seeds with no observation (cannot happen for complete tables)
    keep = np.ones(X.shape[1], bool)
    keep[len(configs):] = X[:, len(configs):].any(0)
    X = X[:, keep]
    y = np.array([vals[i, j] for i, j in cells])
    c = np.zeros(X.shape[1])
    for name, coef in contrast.items():
        c[configs.index(name)] = coef
    return ols_contrast(y, X, c, direction, alpha=alpha)


def additive_residual_sd(S: np.ndarray) -> np.ndarray:
    """Pooled residual SD of the two-way additive fit (config + seed) to a complete table.

    ``S`` has shape ``(..., K, n)`` (configs x seeds). df = (K - 1)(n - 1). This is the
    *effect-blind* quantity of the U-05 trigger: the config and seed means are removed.
    """
    S = np.asarray(S, float)
    K, n = S.shape[-2:]
    res = S - S.mean(-1, keepdims=True) - S.mean(-2, keepdims=True) + S.mean((-2, -1), keepdims=True)
    return np.sqrt((res ** 2).sum((-2, -1)) / ((K - 1) * (n - 1)))


@dataclass
class TriggerResult:
    resid_sd: float
    seed_range: dict[str, float]
    fires: bool
    reason: list[str]
    n_seeds: int

    def to_dict(self) -> dict:
        return dict(resid_sd=self.resid_sd, seed_range=self.seed_range, fires=self.fires,
                    reason=self.reason, n_seeds=self.n_seeds)


def seed_trigger(primary: pd.DataFrame, *, sd_threshold: float = TRIGGER_RESID_SD,
                 range_threshold: float = TRIGGER_SEED_RANGE) -> TriggerResult:
    """U-05 (§11.4): fires iff the blinded residual SD > 2.5 pp or any anchor's seed range > 12 pp.

    ``primary`` is the anchors x seeds table of primary-cell scores after n0 seeds (complete).
    """
    vals = primary.to_numpy(float)
    if not np.isfinite(vals).all():
        raise ValueError("seed trigger needs a complete anchors x seeds table")
    sd = float(additive_residual_sd(vals))
    rng = {str(c): float(np.ptp(primary.loc[c].to_numpy(float))) for c in primary.index}
    reason = []
    if sd > sd_threshold:
        reason.append(f"blinded residual SD {sd:.4f} > {sd_threshold}")
    over = [c for c, r in rng.items() if r > range_threshold]
    if over:
        reason.append(f"seed range > {range_threshold} for {over}")
    return TriggerResult(sd, rng, bool(reason), reason, vals.shape[1])


def seed_trigger_batch(S: np.ndarray, *, sd_threshold: float = TRIGGER_RESID_SD,
                       range_threshold: float = TRIGGER_SEED_RANGE) -> np.ndarray:
    """Vectorised U-05 rule on ``(..., K, n)`` tables (used by the operating-characteristics sims)."""
    S = np.asarray(S, float)
    rng = S.max(-1) - S.min(-1)
    return (additive_residual_sd(S) > sd_threshold) | (rng.max(-1) > range_threshold)


# =============================================================================================
# Success counts (§11.5)
# =============================================================================================
def clopper_pearson(k: int, n: int, level: float = CI_LEVEL) -> tuple[float, float]:
    """Exact (Clopper–Pearson) two-sided CI for a binomial proportion."""
    a = 1 - level
    lo = 0.0 if k == 0 else float(sps.beta.ppf(a / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(sps.beta.ppf(1 - a / 2, k + 1, n - k))
    return lo, hi


def success_counts(primary: pd.DataFrame, *, low: str = "C0", high: str = "C12",
                   level: float = CI_LEVEL, failed: Mapping[str, int] | None = None) -> dict[str, dict]:
    """Seeds above the midpoint (mean S_low + mean S_high)/2 per config, with Clopper–Pearson CIs.

    ``failed`` gives, per config, permanently failed runs (no score) that count as failures (§8.1).
    """
    mid = 0.5 * (np.nanmean(primary.loc[low].to_numpy(float)) + np.nanmean(primary.loc[high].to_numpy(float)))
    out: dict[str, dict] = {}
    for c in primary.index:
        v = primary.loc[c].to_numpy(float)
        v = v[np.isfinite(v)]
        nf = int((failed or {}).get(c, 0))
        k = int((v > mid).sum())
        n = int(v.size + nf)
        lo, hi = clopper_pearson(k, n, level) if n > 0 else (float("nan"), float("nan"))
        out[str(c)] = dict(successes=k, n=n, failed_runs=nf, proportion=k / n if n else float("nan"),
                           ci=[lo, hi], midpoint=float(mid))
    return out


# =============================================================================================
# Fieller CI for ratios (§2 secondary "R with a Fieller CI"; §11.7 H6′ share)
# =============================================================================================
@dataclass
class FiellerCI:
    ratio: float
    lower: float
    upper: float
    kind: Literal["bounded", "exclusive", "unbounded"]
    """``bounded``: [lower, upper]; ``exclusive``: (-inf, lower] U [upper, inf); ``unbounded``: R."""

    def to_dict(self) -> dict:
        return dict(ratio=float(self.ratio), lower=float(self.lower), upper=float(self.upper), kind=self.kind)


def fieller(a: float, b: float, v_aa: float, v_bb: float, v_ab: float, df: float,
            level: float = CI_LEVEL) -> FiellerCI:
    """Fieller CI for a/b given the (co)variances of the estimates a and b."""
    q = float(sps.t.ppf(1 - (1 - level) / 2, df))
    q2 = q * q
    A = b * b - q2 * v_bb
    Bh = a * b - q2 * v_ab
    C = a * a - q2 * v_aa
    disc = Bh * Bh - A * C
    ratio = a / b if b != 0 else float("nan")
    if A > 0:
        r = np.sqrt(max(disc, 0.0))
        return FiellerCI(ratio, (Bh - r) / A, (Bh + r) / A, "bounded")
    if disc < 0:
        return FiellerCI(ratio, -np.inf, np.inf, "unbounded")
    r = np.sqrt(disc)
    lo, hi = sorted(((Bh - r) / A, (Bh + r) / A))
    return FiellerCI(ratio, lo, hi, "exclusive")


def fieller_paired(num: np.ndarray, den: np.ndarray, level: float = CI_LEVEL) -> FiellerCI:
    """Fieller CI for mean(num)/mean(den) from per-seed paired values (df = n - 1).

    For R(k): ``num_s = S_k,s - S_0,s`` and ``den_s = S_12,s - S_0,s``.
    """
    num, den = np.asarray(num, float), np.asarray(den, float)
    n = num.size
    cov = np.cov(np.vstack([num, den]), ddof=1) / n
    return fieller(num.mean(), den.mean(), cov[0, 0], cov[1, 1], cov[0, 1], n - 1, level)


def fieller_independent(num_mean: float, num_var: float, den_mean: float, den_var: float, df: float,
                        level: float = CI_LEVEL) -> FiellerCI:
    """Fieller CI when numerator and denominator estimates are independent (cov = 0)."""
    return fieller(num_mean, den_mean, num_var, den_var, 0.0, df, level)


# =============================================================================================
# Curves: isotonic, saturating, k80, N50
# =============================================================================================
def isotonic_increasing(y: Sequence[float], w: Sequence[float] | None = None) -> np.ndarray:
    """Weighted pool-adjacent-violators fit of a non-decreasing sequence (x already sorted)."""
    y = np.asarray(y, float)
    w = np.ones_like(y) if w is None else np.asarray(w, float)
    vals: list[float] = []
    wts: list[float] = []
    cnt: list[int] = []
    for yi, wi in zip(y, w):
        vals.append(yi); wts.append(wi); cnt.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            v2, w2, c2 = vals.pop(), wts.pop(), cnt.pop()
            v1, w1, c1 = vals.pop(), wts.pop(), cnt.pop()
            wt = w1 + w2
            vals.append((v1 * w1 + v2 * w2) / wt); wts.append(wt); cnt.append(c1 + c2)
    return np.repeat(vals, cnt)


def isotonic_recovery(ks: Sequence[int], mean_scores: Sequence[float],
                      weights: Sequence[float] | None = None) -> np.ndarray:
    """Isotonic fit of S(k) rescaled to recovery R_iso(k) = (iso(k) - iso(0)) / (iso(12) - iso(0))."""
    ks = np.asarray(ks)
    order = np.argsort(ks)
    iso = np.empty(len(ks))
    iso[order] = isotonic_increasing(np.asarray(mean_scores, float)[order],
                                     None if weights is None else np.asarray(weights, float)[order])
    i0, i12 = int(np.argmin(ks)), int(np.argmax(ks))
    g = iso[i12] - iso[i0]
    return (iso - iso[i0]) / g if g > 0 else np.full(len(ks), np.nan)


def k80(ks: Sequence[int], mean_scores: Sequence[float], weights: Sequence[float] | None = None,
        target: float = R_TARGET) -> float:
    """k80: linear interpolation of isotonic R(k) at R = 0.8 (§11.7 H2). NaN if the gap is <= 0."""
    ks = np.asarray(ks, float)
    order = np.argsort(ks)
    r = isotonic_recovery(ks, mean_scores, weights)[order]
    kk = ks[order]
    if not np.isfinite(r).all():
        return float("nan")
    idx = int(np.argmax(r >= target))
    if idx == 0:
        return float(kk[0])
    r0, r1 = r[idx - 1], r[idx]
    return float(kk[idx - 1] + (target - r0) / (r1 - r0) * (kk[idx] - kk[idx - 1]))


def saturating_curve(k: np.ndarray, s0: float, g: float, kappa: float) -> np.ndarray:
    """S(k) = S0 + G (1 - exp(-k/kappa)) / (1 - exp(-12/kappa)) (§11.7 H2)."""
    k = np.asarray(k, float)
    return s0 + g * (-np.expm1(-k / kappa)) / (-np.expm1(-L_LAYERS / kappa))


@dataclass
class SaturatingFit:
    s0: float
    g: float
    kappa: float
    rss: float
    n: int
    kappa_at_bound: bool

    def to_dict(self) -> dict:
        return dict(s0=self.s0, g=self.g, kappa=self.kappa, rss=self.rss, n=self.n,
                    kappa_at_bound=self.kappa_at_bound)


def saturating_fit(k: Sequence[float], s: Sequence[float], *, kappa_bounds: tuple[float, float] = (0.05, 1e3)
                   ) -> SaturatingFit:
    """Least-squares fit of the saturating curve to every (k, per-seed score) point.

    kappa -> infinity approaches the linear curve; small kappa is a step at k = 1. kappa is fitted on
    the log scale within ``kappa_bounds`` and flagged when it lands on a bound.
    """
    k = np.asarray(k, float)
    s = np.asarray(s, float)
    lo, hi = np.log(kappa_bounds[0]), np.log(kappa_bounds[1])

    def resid(p):
        return saturating_curve(k, p[0], p[1], np.exp(p[2])) - s

    best = None
    for lk0 in np.linspace(lo + 0.1, hi - 0.1, 7):
        s0g = float(np.mean(s[k == k.min()])) if (k == k.min()).any() else float(s.min())
        g0 = float(np.mean(s[k == k.max()])) - s0g
        r = optimize.least_squares(resid, x0=[s0g, g0, lk0], bounds=([-np.inf, -np.inf, lo], [np.inf, np.inf, hi]))
        if best is None or r.cost < best.cost:
            best = r
    assert best is not None
    kap = float(np.exp(best.x[2]))
    at_bound = bool(abs(best.x[2] - lo) < 1e-3 or abs(best.x[2] - hi) < 1e-3)
    return SaturatingFit(float(best.x[0]), float(best.x[1]), kap, float(2 * best.cost), int(k.size), at_bound)


@dataclass
class N50Fit:
    n50: float
    intercept: float
    slope: float
    censoring: Literal["none", "left", "right", "undefined"]
    levels: list[float]
    log2_n50: float = float("nan")
    """-a/b, finite whenever the slope is negative (n50 may be inf/0 when 2^(-a/b) is not representable)."""

    def to_dict(self) -> dict:
        return dict(n50=self.n50, intercept=self.intercept, slope=self.slope, censoring=self.censoring,
                    levels=self.levels, log2_n50=self.log2_n50)


def n50_logistic(N: Sequence[float], successes: Sequence[float], trials: Sequence[float]) -> N50Fit:
    """Binomial-ML logistic fit EM ~ expit(a + b log2 N); N50 = 2^(-a/b) (§9.7).

    ``censoring`` is ``left`` if N50 < min level, ``right`` if N50 > max level (reported, never
    extrapolated silently) and ``undefined`` when the fitted slope is not negative.
    """
    N = np.asarray(N, float)
    y = np.asarray(successes, float)
    m = np.asarray(trials, float)
    x = np.log2(N)

    def nll(p):
        eta = p[0] + p[1] * x
        return -np.sum(y * eta - m * np.logaddexp(0.0, eta))

    def grad(p):
        mu = special.expit(p[0] + p[1] * x)
        r = y - m * mu
        return -np.array([r.sum(), (r * x).sum()])

    p0 = np.array([0.0, -0.5])
    res = optimize.minimize(nll, p0, jac=grad, method="BFGS")
    a, b = map(float, res.x)
    levels = sorted(N.tolist())
    if not b < 0:
        return N50Fit(float("nan"), a, b, "undefined", levels)
    # Work in log2 space: a flat high-accuracy curve (tiny negative slope, positive intercept)
    # puts -a/b far beyond the level range, and 2.0 ** (-a/b) overflowed (verification round 1,
    # DTS-03). Censoring is decided on log2 N50; n50 saturates to inf / 0 when unrepresentable.
    log2_n50 = float(-a / b)
    cens: Literal["none", "left", "right", "undefined"] = "none"
    if log2_n50 < x.min():
        cens = "left"
    elif log2_n50 > x.max():
        cens = "right"
    if log2_n50 > 1000:
        n50 = float("inf")
    elif log2_n50 < -1000:
        n50 = 0.0
    else:
        n50 = float(2.0 ** log2_n50)
    return N50Fit(n50, a, b, cens, levels, log2_n50)


# =============================================================================================
# Hierarchical paired bootstrap (§11.3)
# =============================================================================================
@dataclass
class BootstrapDraws:
    """Bootstrap replicates of per-seed-slot scores.

    ``scores[name]`` has shape ``(B, n_slots, n_configs)``. Every config receives the **same** seed
    draw in every slot (drawn from ``seeds``, the complete seed blocks of the configs bootstrapped)
    and the same context draws (common random numbers), as §11.3 requires ("resample seeds (same
    indices for every config), then contexts within strata (same indices for every config)").
    """

    configs: list[str]
    scores: dict[str, np.ndarray]
    n_seeds: dict[str, int]
    B: int
    entropy: int
    spawn_key: tuple[int, ...] = field(default_factory=tuple)
    seeds: list[int] = field(default_factory=list)
    """Seed indices resampled: those present for *every* config in ``configs`` (complete blocks)."""
    incomplete_seeds: list[int] = field(default_factory=list)
    """Seeds present for some but not all of ``configs`` (excluded; verification round 2)."""

    def get(self, name: str, config: str) -> np.ndarray:
        return self.scores[name][:, :, self.configs.index(config)]


def _seed_seq(entropy: int, key: str | int) -> np.random.SeedSequence:
    k = key if isinstance(key, int) else zlib.crc32(key.encode())
    return np.random.SeedSequence(entropy=entropy, spawn_key=(k,))


def hierarchical_bootstrap(table, weights: Mapping[str, np.ndarray], *, configs: Sequence[str] | None = None,
                           B: int = BOOTSTRAP_B, entropy: int = BOOTSTRAP_ENTROPY, key: str | int = 0,
                           chunk: int = 500) -> BootstrapDraws:
    """Two-stage paired bootstrap: seeds, then contexts within (task, level, T) strata.

    ``table`` is a :class:`~hybridlc.analysis.scores.CellTable`; ``weights`` maps names to stratum
    weight vectors (from ``table.ladder_weights``). Only the **complete seed blocks** of ``configs``
    (seeds present for every one of them; the same blocks the paired t-test uses) are resampled. For
    each replicate b and seed slot j (n_slots = number of complete blocks):

    1. one draw picks a seed from the complete blocks; it is the seed of slot j for *every* config;
    2. per stratum, contexts are drawn with replacement (same indices for every config), and the
       stratum mean of each config is recomputed on the drawn contexts of the drawn seed.

    Verification round 2 (ANA-BOOT-INCOMPLETE-SEED-BLOCKS): the earlier version drew each config's
    seed by position in its *own* seed list, so with one failed run (§8.1) the slots were unpaired.
    Missing data are never zero-filled (ANA-MISSING-CELL-SILENT-SEED-DROP): a NaN context mean in a
    used stratum of a complete block raises ``ValueError``. Callers bootstrap each estimand on the
    config set it uses, so a failure in an unrelated config never removes a seed.

    The RNG is ``SeedSequence(entropy=2026, spawn_key=(crc32(key),))`` (§8.1 bootstrap stream).
    """
    cfgs = list(configs) if configs is not None else list(table.configs)
    ci = [table.configs.index(c) for c in cfgs]
    pres = np.asarray(table.present, bool)[ci]  # (C, S)
    n_c = pres.sum(axis=1)
    if (n_c == 0).any():
        raise ValueError("a config has no seeds in this table")
    common = np.nonzero(pres.all(axis=0))[0]
    incomplete = np.nonzero(pres.any(axis=0) & ~pres.all(axis=0))[0]
    if common.size == 0:
        raise ValueError(f"configs {cfgs} share no complete seed block")
    n_slots = int(common.size)
    names = list(weights)
    W = np.stack([np.asarray(weights[nm], float) for nm in names])  # (n_w, n_strata)
    used = np.nonzero(np.abs(W).sum(0) > 0)[0]
    rng = np.random.default_rng(_seed_seq(entropy, key))
    out = {nm: np.full((B, n_slots, len(cfgs)), np.nan) for nm in names}
    # Data restricted to the configs and complete blocks; refuse missing context means.
    Xs = {}
    for s_idx in used:
        X = table.ctx[s_idx][:, ci, :][:, :, common]  # (n_ctx, C, S_common)
        if X.shape[0] == 0 or np.isnan(X).any():
            st = table.strata[s_idx]
            raise ValueError(f"stratum {st}: missing results for a complete seed block of {cfgs} "
                             f"(the bootstrap never zero-fills; re-score the cell)")
        Xs[s_idx] = X
    S = n_slots
    for b0 in range(0, B, chunk):
        bc = min(chunk, B - b0)
        U = rng.random((bc, n_slots))
        # pick[b, j] = column (position in ``common``) of the seed drawn for slot j, same for all configs
        pick1 = np.minimum((U * S).astype(int), S - 1)
        pick = np.repeat(pick1[:, :, None], len(cfgs), axis=2)
        acc = np.zeros((len(names), bc, n_slots, len(cfgs)))
        for s_idx in used:
            X = Xs[s_idx]
            n_ctx = X.shape[0]
            Xf = X.reshape(n_ctx, len(cfgs) * S)
            draws = rng.integers(0, n_ctx, size=(bc * n_slots, n_ctx))
            flat = draws + (np.arange(bc * n_slots)[:, None] * n_ctx)
            counts = np.bincount(flat.ravel(), minlength=bc * n_slots * n_ctx).reshape(bc * n_slots, n_ctx)
            M = (counts @ Xf / n_ctx).reshape(bc, n_slots, len(cfgs), S)
            vals = np.take_along_axis(M, pick[..., None], axis=-1)[..., 0]  # (bc, n_slots, C)
            acc += W[:, s_idx][:, None, None, None] * vals[None]
        for w_i, nm in enumerate(names):
            out[nm][b0:b0 + bc] = acc[w_i]
    seeds_all = list(getattr(table, "seeds", range(pres.shape[1])))
    return BootstrapDraws(cfgs, out, {c: int(n_slots) for c in cfgs}, B, entropy,
                          (key if isinstance(key, int) else zlib.crc32(key.encode()),),
                          seeds=[int(seeds_all[j]) for j in common],
                          incomplete_seeds=[int(seeds_all[j]) for j in incomplete])


def percentile_ci(values: np.ndarray, level: float = CI_LEVEL, direction: Direction = "two-sided"
                  ) -> tuple[float, float]:
    """Percentile CI of bootstrap replicates. One-sided: (q_alpha, inf) or (-inf, q_{1-alpha}).

    Returns (nan, nan) when no replicate is finite (e.g. a family at floor for both reference
    models, so every recovery is 0/0; verification round 1, DTS-04). NaN bounds never "agree"."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan")
    a = 1 - level
    if direction == "greater":
        return float(np.quantile(v, a)), float("inf")
    if direction == "less":
        return float("-inf"), float(np.quantile(v, 1 - a))
    return float(np.quantile(v, a / 2)), float(np.quantile(v, 1 - a / 2))


def paired_mean_over_slots(draws: BootstrapDraws, fn: Callable[[Callable[[str, str], np.ndarray]], np.ndarray]
                           ) -> np.ndarray:
    """Bootstrap distribution of the seed-mean of a per-seed statistic.

    ``fn(get)`` receives ``get(name, config) -> (B, n_slots)`` and returns per-slot values
    ``(B, n_slots)``; the result is their mean over valid slots, shape ``(B,)``.
    """
    per_slot = fn(draws.get)
    return np.nanmean(per_slot, axis=1)

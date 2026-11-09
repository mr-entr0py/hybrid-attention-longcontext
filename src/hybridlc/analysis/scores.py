"""Family scores from item-level rows (protocol §2 "Definitions", §11.1).

Everything a confirmatory statistic needs is a **linear** function of cell means, where a cell (a
"stratum") is one (task, level, T) of one suite:

* a task's ladder mean is the equal-weight mean over its levels at that T;
* a family score is the equal-weight mean of its primary tasks' ladder means;
* the primary-cell score is the equal-weight mean over the passing lengths (§11.2).

So each estimand is a weight vector over strata (:func:`ladder_weights`), and the per-seed score is
``weights @ cell_means``. The hierarchical bootstrap (:mod:`.stats`) re-uses the same weight vectors,
so point estimates and bootstrap replicates are computed by one code path.

A cell mean is the mean over contexts of the per-context mean over its Q queries. With equal Q per
context this equals the mean over queries.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .protocol import CONTROL_T, FAMILIES, N50_ONLY_LEVELS

Stratum = tuple[str, str, int]
"""(task, level, T); T == CONTROL_T for C0 controls."""

LevelSpec = Mapping[str, Sequence[str] | None]
"""task -> allowed levels (``None`` = every level present, except N50-only levels)."""


def family_spec(family: str, levels: Mapping[str, Sequence[str]] | None = None) -> dict[str, Sequence[str] | None]:
    """Level spec for a family's primary tasks, optionally restricted to fixed level sets."""
    tasks = FAMILIES[family]
    return {t: (None if levels is None else levels.get(t)) for t in tasks}


@dataclass
class CellTable:
    """Context-level means of one metric for one suite, aligned across configs and seeds.

    ``ctx[s]`` has shape ``(n_contexts_s, n_configs, n_seed_slots)``; entries for a (config, seed)
    that was not evaluated are NaN. Context ids are aligned across configs and seeds (the frozen item
    files are identical for every model, §9.8); a mismatch raises ``ValueError``.
    """

    suite: str
    metric: str
    strata: list[Stratum]
    configs: list[str]
    seeds: list[int]
    """Union of seed indices over configs, sorted. Column j of ``ctx`` is seed ``seeds[j]``."""
    ctx: list[np.ndarray]
    context_ids: list[np.ndarray]
    present: np.ndarray
    """Boolean (n_configs, n_seeds): True where that run has results in this suite."""
    inputs: list[str] = field(default_factory=list)

    # ---------------------------------------------------------------- construction
    @classmethod
    def from_rows(cls, df: pd.DataFrame, suite: str, metric: str = "em_tf", *,
                  configs: Sequence[str] | None = None, branch: str = "main",
                  ckpt: str | None = None) -> "CellTable":
        d = df[(df["suite"] == suite) & (df["branch"] == branch)]
        if ckpt is not None:
            d = d[d["ckpt"] == ckpt]
        else:
            n_ck = d.groupby("run_id")["ckpt"].nunique()
            if (n_ck > 1).any():
                raise ValueError(f"runs {list(n_ck[n_ck > 1].index)} have several checkpoints in suite "
                                 f"{suite!r}; pass ckpt=")
        if configs is not None:
            d = d[d["config"].isin(list(configs))]
        if d.empty:
            raise ValueError(f"no rows for suite={suite!r}, branch={branch!r}, ckpt={ckpt!r}")
        dup = d.groupby(["config", "seed"])["run_id"].nunique()
        if (dup > 1).any():
            raise ValueError(f"several runs map to the same (config, seed): {list(dup[dup > 1].index)}")
        cfgs = list(configs) if configs is not None else sorted(d["config"].unique(), key=_config_order)
        cfgs = [c for c in cfgs if c in set(d["config"])]
        seeds = sorted(int(s) for s in d["seed"].unique())
        ci = {c: i for i, c in enumerate(cfgs)}
        si = {s: j for j, s in enumerate(seeds)}
        present = np.zeros((len(cfgs), len(seeds)), bool)
        for (c, s), _ in d.groupby(["config", "seed"]):
            present[ci[c], si[int(s)]] = True
        g = (d.groupby(["task", "level", "T", "context_id", "config", "seed"], sort=True)[metric]
               .mean())
        strata: list[Stratum] = []
        ctx: list[np.ndarray] = []
        ids: list[np.ndarray] = []
        for (task, level, T), sub in g.groupby(level=["task", "level", "T"], sort=True):
            wide = sub.droplevel(["task", "level", "T"]).unstack(["config", "seed"])
            arr = np.full((wide.shape[0], len(cfgs), len(seeds)), np.nan)
            for (c, s) in wide.columns:
                arr[:, ci[c], si[int(s)]] = wide[(c, s)].to_numpy(float)
            # Alignment: every evaluated (config, seed) must have exactly the same contexts.
            ev = ~np.isnan(arr).all(axis=0)
            col_has = ~np.isnan(arr)
            for c_i, s_j in zip(*np.nonzero(ev)):
                if not col_has[:, c_i, s_j].all():
                    miss = int((~col_has[:, c_i, s_j]).sum())
                    raise ValueError(f"stratum {(task, level, T)}: config {cfgs[c_i]} seed {seeds[s_j]} "
                                     f"lacks {miss} of the contexts other models have (items must be "
                                     f"identical across models)")
            strata.append((str(task), str(level), int(T)))
            ctx.append(arr)
            ids.append(wide.index.to_numpy())
        return cls(suite=suite, metric=metric, strata=strata, configs=cfgs, seeds=seeds, ctx=ctx,
                   context_ids=ids, present=present, inputs=list(df.attrs.get("inputs", [])))

    # ---------------------------------------------------------------- accessors
    def cell_means(self) -> np.ndarray:
        """Stratum means, shape ``(n_strata, n_configs, n_seeds)``; NaN where not evaluated."""
        return np.stack([np.nanmean(a, axis=0) if a.shape[0] else np.full(a.shape[1:], np.nan)
                         for a in self.ctx]) if self.ctx else np.zeros((0, len(self.configs), len(self.seeds)))

    def levels_at(self, task: str, T: int) -> list[str]:
        return [lv for (t, lv, TT) in self.strata if t == task and TT == T]

    def lengths(self, include_control: bool = False) -> list[int]:
        Ts = sorted({T for (_, _, T) in self.strata})
        return [T for T in Ts if include_control or T != CONTROL_T]

    def has(self, config: str) -> bool:
        return config in self.configs

    def seeds_of(self, config: str) -> list[int]:
        i = self.configs.index(config)
        return [s for s, p in zip(self.seeds, self.present[i]) if p]

    def subset_configs(self, configs: Sequence[str]) -> "CellTable":
        idx = [self.configs.index(c) for c in configs]
        return CellTable(self.suite, self.metric, list(self.strata), list(configs), list(self.seeds),
                         [a[:, idx, :] for a in self.ctx], list(self.context_ids),
                         self.present[idx], list(self.inputs))

    # ---------------------------------------------------------------- weights and scores
    def ladder_weights(self, spec: LevelSpec, Ts: Sequence[int], *, control_for: int | None = None
                       ) -> np.ndarray:
        """Weight vector over strata for the equal-weight ladder/family/length mean.

        ``spec`` maps each task of the family to its allowed levels (``None`` = all present except
        N50-only levels). For every T in ``Ts`` and every task, the levels present at T (intersected
        with the allowed set) receive weight ``1 / (|levels| * |tasks| * |Ts|)``.

        ``control_for=T``: weights on the C0 control strata of exactly the level set used at T
        (the denominator of G-REF criterion (a), "on the R1/R-load ladder mean").
        """
        w = np.zeros(len(self.strata))
        pos = {s: i for i, s in enumerate(self.strata)}
        tasks = list(spec)
        Ts_eff = [control_for] if control_for is not None else list(Ts)
        for T in Ts_eff:
            for task in tasks:
                allowed = spec[task]
                lv = [x for x in self.levels_at(task, T) if (task, x) not in N50_ONLY_LEVELS]
                if allowed is not None:
                    lv = [x for x in lv if x in set(allowed)]
                    missing = set(allowed) - set(lv)
                    if missing:
                        raise ValueError(f"{self.suite}: task {task} at T={T} lacks fixed levels {sorted(missing)}")
                if not lv:
                    raise ValueError(f"{self.suite}: task {task} has no levels at T={T}")
                for x in lv:
                    key = (task, x, CONTROL_T if control_for is not None else T)
                    if key not in pos:
                        raise ValueError(f"{self.suite}: missing stratum {key}")
                    w[pos[key]] += 1.0 / (len(lv) * len(tasks) * len(Ts_eff))
        return w

    def level_weights(self, task: str, level: str, Ts: Sequence[int]) -> np.ndarray:
        """Weights for one ladder level averaged over ``Ts`` (per-level results, H1-robust)."""
        return self.ladder_weights({task: [level]}, Ts)

    def scores(self, weights: np.ndarray) -> pd.DataFrame:
        """Per-(config, seed) score ``weights @ cell_means`` as a config x seed DataFrame (NaN = absent)."""
        cm = self.cell_means()
        used = weights != 0
        val = np.tensordot(weights[used], cm[used], axes=(0, 0))
        val = np.where(self.present, val, np.nan)
        return pd.DataFrame(val, index=self.configs, columns=self.seeds)


def _config_order(c: str) -> tuple[int, int, str]:
    if c.startswith("C") and c[1:].isdigit():
        return (0, int(c[1:]), c)
    return (1, 0, c)


def family_score_table(table: CellTable, family: str, Ts: Iterable[int],
                       levels: Mapping[str, Sequence[str]] | None = None) -> pd.DataFrame:
    """Convenience: config x seed family scores averaged over ``Ts``."""
    return table.scores(table.ladder_weights(family_spec(family, levels), list(Ts)))


def per_length_family_scores(table: CellTable, family: str,
                             levels: Mapping[str, Sequence[str]] | None = None) -> pd.DataFrame:
    """Long DataFrame (config, seed, T, score) of the family score at every length present."""
    out = []
    for T in table.lengths():
        try:
            s = family_score_table(table, family, [T], levels)
        except ValueError:
            continue
        long = s.stack().dropna().rename("score").reset_index()
        long.columns = ["config", "seed", "score"]
        long["T"] = T
        out.append(long)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["config", "seed", "score", "T"])

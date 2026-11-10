#!/usr/bin/env python
"""Agent-4: planted-effect simulations of the §11 statistics code (CPU only, READ-ONLY on the repo).

Uses the project's own functions (hybridlc.analysis.stats / pipeline / scores) on synthetic context-level
tables shaped like the 16K+32K retrieval primary cell (R1 D3/D15, R-load N8..N1024; 300 contexts x Q=4).

P1  paired-bootstrap pairing: a planted constant config shift is recovered exactly (bootstrap SD of the
    contrast ~ 0 when there is no noise); sensitivity 'agreement' leniency vs the primary t-test.
P2  item x config interaction (crossed design): t-test (conditional on items) vs the project's nested
    bootstrap vs a crossed bootstrap (one context draw per replicate shared by all seed slots).
P3  Fieller CI coverage for R = mean(num)/mean(den) from per-seed pairs.
P4  U-05 trigger blindness: invariance to planted config effects.
P5  H3 equated-pair sign check: veto rate of a true D3 > 0 at a difficulty-equated pair with small G.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/home/sedrica-lynx/Downloads/outeropt")
sys.path.insert(0, str(REPO / "src"))
from hybridlc.analysis.pipeline import _agrees, _boot_ci, boot_L  # noqa: E402
from hybridlc.analysis.scores import CellTable, family_spec  # noqa: E402
from hybridlc.analysis.stats import (fieller_paired, hierarchical_bootstrap, l_contrast, one_sample_t,  # noqa: E402
                                     pooled_contrast, recovery, seed_trigger)

STRATA = [("R1", lv, T) for T in (16384, 32768) for lv in ("D3", "D15")] + \
         [("R-load", lv, T) for T in (16384, 32768) for lv in ("N8", "N32", "N128", "N512", "N1024")]
CONFIGS = ["C0", "C1", "C3", "C12"]
SEEDS = [0, 1, 2, 3, 4]
NCTX, Q = 300, 4


def make_table(rng, mu, seed_sd=0.02, run_sd=0.02, tau=0.0, item_sd=0.15, noise=True):
    """mu: dict config -> per-stratum true mean array. Returns CellTable with context-level means."""
    seed_eff = rng.normal(0, seed_sd, len(SEEDS))
    run_eff = rng.normal(0, run_sd, (len(CONFIGS), len(SEEDS)))
    ctx = []
    for t_i, st in enumerate(STRATA):
        u = rng.normal(0, item_sd, NCTX)  # item main effect (shared by every config & seed)
        b = rng.normal(0, tau, (NCTX, len(CONFIGS)))  # item x config interaction (shared by seeds)
        arr = np.empty((NCTX, len(CONFIGS), len(SEEDS)))
        for c_i, c in enumerate(CONFIGS):
            for s_i in range(len(SEEDS)):
                p = np.clip(mu[c][t_i] + seed_eff[s_i] + run_eff[c_i, s_i] + u + b[:, c_i], 0.0, 1.0)
                arr[:, c_i, s_i] = rng.binomial(Q, p) / Q if noise else p
        ctx.append(arr)
    ids = [np.arange(NCTX) for _ in STRATA]
    return CellTable("CS", "em_tf", list(STRATA), list(CONFIGS), list(SEEDS), ctx, ids,
                     np.ones((len(CONFIGS), len(SEEDS)), bool))


def crossed_bootstrap_L(table, w, B, rng, k="C3"):
    """Seeds resampled per slot (same for every config); contexts resampled ONCE per replicate per stratum
    and shared by all seed slots and configs (crossed items x seeds)."""
    ci = [table.configs.index(c) for c in ("C0", k, "C12")]
    S = len(table.seeds)
    used = np.nonzero(w)[0]
    out = np.empty(B)
    for b in range(B):
        pick = rng.integers(0, S, S)
        sc = np.zeros((3, S))
        for s_idx in used:
            X = table.ctx[s_idx][:, ci, :]
            draw = rng.integers(0, X.shape[0], X.shape[0])
            M = X[draw].mean(0)  # (3, S)
            sc += w[s_idx] * M[:, pick]
        L = sc[1] - 0.2 * sc[0] - 0.8 * sc[2]
        out[b] = L.mean()
    return out


def mu_for(L_true: float, s0=0.30, s12=0.80, spread=0.25):
    """Per-stratum means with a difficulty spread; S3 set so that the stratum-level L equals L_true."""
    d = np.linspace(-spread / 2, spread / 2, len(STRATA))
    m0, m12 = np.clip(s0 + d * 0.5, 0.02, 0.98), np.clip(s12 + d * 0.3, 0.02, 0.98)
    m3 = np.clip(0.2 * m0 + 0.8 * m12 + L_true, 0.0, 1.0)
    m1 = np.clip(0.5 * m0 + 0.5 * m12, 0, 1)
    return {"C0": m0, "C1": m1, "C3": m3, "C12": m12}


def p1_p2(n_rep=200, B=1000):
    rng = np.random.default_rng(20261110)
    res = {}
    for name, L_true, tau in (("boundary_L0_tau0", 0.0, 0.0), ("L+0.015_tau0", 0.015, 0.0), ("L+0.03_tau0", 0.03, 0.0),
                              ("boundary_L0_tau0.25", 0.0, 0.25)):
        mu = mu_for(L_true)
        rec = dict(t_reject=0, boot_agree=0, boot_agree_given_t=0, pooled_agree_given_t=0, crossed_agree=0,
                   both_agree_given_t=0, t_width=[], boot_width=[], crossed_width=[], est=[])
        for r in range(n_rep):
            tab = make_table(rng, mu, tau=tau)
            w = tab.ladder_weights(family_spec("retrieval"), [16384, 32768])
            sc = tab.scores(w)
            L = l_contrast(sc.loc["C3"], sc.loc["C0"], sc.loc["C12"]).to_numpy()
            tg = one_sample_t(L, "greater")
            rej = bool(tg.p < 0.05)
            rec["t_reject"] += rej
            rec["est"].append(float(L.mean()))
            rec["t_width"].append(float(L.mean() - tg.lower))
            dr = hierarchical_bootstrap(tab, {"retrieval@primary": w}, configs=["C0", "C3", "C12"], B=B,
                                        key=f"sim{r}")
            bl = boot_L(dr, "C3")
            ci = _boot_ci(bl)
            ag = _agrees(ci["lower95_one_sided"], ci["upper95_one_sided"], "greater")
            rec["boot_width"].append(float(L.mean() - ci["lower95_one_sided"]))
            pg = pooled_contrast(sc.loc[CONFIGS], {"C3": 1.0, "C0": -0.2, "C12": -0.8}, "greater")
            pag = bool(pg.lower > 0)
            rec["boot_agree"] += ag
            if rej:
                rec["boot_agree_given_t"] += ag
                rec["pooled_agree_given_t"] += pag
                rec["both_agree_given_t"] += (ag and pag)
            if tau > 0 or r < 100:
                cb = crossed_bootstrap_L(tab, w, 400, rng)
                lo = float(np.quantile(cb, 0.05))
                rec["crossed_agree"] += lo > 0
                rec["crossed_width"].append(float(L.mean() - lo))
        n_t = max(rec["t_reject"], 1)
        n_cross = n_rep if tau > 0 else min(100, n_rep)
        res[name] = dict(n_rep=n_rep, t_reject_rate=rec["t_reject"] / n_rep,
                         boot_lower_gt0_rate=rec["boot_agree"] / n_rep,
                         crossed_boot_lower_gt0_rate=rec["crossed_agree"] / n_cross,
                         P_boot_agrees_given_t_rejects=rec["boot_agree_given_t"] / n_t,
                         P_pooled_agrees_given_t_rejects=rec["pooled_agree_given_t"] / n_t,
                         P_both_agree_given_t_rejects=rec["both_agree_given_t"] / n_t,
                         median_halfwidth_t=float(np.median(rec["t_width"])),
                         median_halfwidth_nested_boot=float(np.median(rec["boot_width"])),
                         median_halfwidth_crossed_boot=float(np.median(rec["crossed_width"])) if rec["crossed_width"] else None,
                         sd_of_estimate_over_reps=float(np.std(rec["est"])),
                         mean_estimate=float(np.mean(rec["est"])))
    # P1 exactness: noiseless constant shift -> bootstrap reproduces the per-seed contrast exactly
    mu = mu_for(0.05)
    tab = make_table(np.random.default_rng(1), mu, seed_sd=0.0, run_sd=0.0, tau=0.0, item_sd=0.0, noise=False)
    w = tab.ladder_weights(family_spec("retrieval"), [16384, 32768])
    dr = hierarchical_bootstrap(tab, {"retrieval@primary": w}, configs=["C0", "C3", "C12"], B=200, key="exact")
    bl = boot_L(dr, "C3")
    sc = tab.scores(w)
    Lp = float(l_contrast(sc.loc["C3"], sc.loc["C0"], sc.loc["C12"]).mean())
    res["P1_noiseless_shift"] = dict(planted_L_point=Lp, boot_min=float(bl.min()), boot_max=float(bl.max()))
    return res


def p3_fieller(n_rep=4000):
    rng = np.random.default_rng(7)
    out = {}
    for n, G, sd_n, sd_d, rho in ((5, 0.40, 0.025, 0.025, 0.6), (5, 0.15, 0.025, 0.025, 0.6), (4, 0.40, 0.03, 0.03, 0.3)):
        R = 0.8
        cov = np.array([[sd_n ** 2, rho * sd_n * sd_d], [rho * sd_n * sd_d, sd_d ** 2]])
        hit, kinds = 0, {}
        for _ in range(n_rep):
            x = rng.multivariate_normal([R * G, G], cov, size=n)
            f = fieller_paired(x[:, 0], x[:, 1])
            kinds[f.kind] = kinds.get(f.kind, 0) + 1
            if f.kind == "bounded":
                hit += f.lower <= R <= f.upper
            elif f.kind == "unbounded":
                hit += 1
            else:
                hit += (R <= f.lower) or (R >= f.upper)
        out[f"n{n}_G{G}_sd{sd_n}_rho{rho}"] = dict(coverage=hit / n_rep, kinds=kinds)
    return out


def p4_trigger():
    rng = np.random.default_rng(3)
    base = pd.DataFrame(rng.normal(0.5, 0.02, (4, 5)), index=["C0", "C1", "C3", "C12"], columns=range(5))
    t0 = seed_trigger(base)
    shifted = base.add(pd.Series([-0.3, 0.1, 0.25, 0.4], index=base.index), axis=0)
    t1 = seed_trigger(shifted)
    return dict(resid_sd_base=t0.resid_sd, resid_sd_shifted=t1.resid_sd, ranges_equal=t0.seed_range == t1.seed_range,
                fires_equal=t0.fires == t1.fires)


def p5_equated(n_rep=4000):
    """Per-seed R at one level (as pipeline.equated_check does) averaged over seeds; true D3 = +0.2 overall
    and at the pair. Veto = mean per-seed D3 at the pair <= 0."""
    rng = np.random.default_rng(11)
    out = {}
    for G_lvl in (0.05, 0.10, 0.20, 0.40):
        veto, nanr = 0, 0
        for _ in range(n_rep):
            n = 5
            s0 = 0.5 + rng.normal(0, 0.02, n)
            # retrieval level: R=0.9 ; multi-hop level: R=0.7 ; same G at the level
            run = lambda: rng.normal(0, 0.02, n)  # noqa: E731
            r0, r3, r12 = s0 + run(), s0 + 0.9 * G_lvl + run(), s0 + G_lvl + run()
            h0, h3, h12 = s0 + run(), s0 + 0.7 * G_lvl + run(), s0 + G_lvl + run()
            d = recovery(r3, r0, r12) - recovery(h3, h0, h12)
            m = np.nanmean(d)
            if not np.isfinite(m):
                nanr += 1
                continue
            veto += m <= 0
        out[f"G_level={G_lvl}"] = dict(veto_rate=veto / n_rep, nan=nanr)
    return out


if __name__ == "__main__":
    which = sys.argv[1:] or ["p1p2", "p3", "p4", "p5"]
    res = {}
    if "p4" in which:
        res["P4_trigger_blindness"] = p4_trigger()
    if "p3" in which:
        res["P3_fieller"] = p3_fieller()
    if "p5" in which:
        res["P5_equated_veto"] = p5_equated()
    if "p1p2" in which:
        res["P1_P2_bootstrap"] = p1_p2()
    print(json.dumps(res, indent=1, default=str))
    (REPO / ".audit" / "agent4_eval" / f"sim_stats_{'_'.join(which)}.json").write_text(json.dumps(res, indent=1, default=str))

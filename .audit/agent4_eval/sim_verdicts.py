#!/usr/bin/env python
"""Agent-4 (resumed): operating characteristics of the §11 verdict machinery AS IMPLEMENTED (CPU, read-only).

Exact planted truth: context-level means are Gaussian (no clipping), so every estimand is linear and its
true value is known exactly. Uses the project's own functions (stats / scores / pipeline helpers).

S1  H1 verdict rule: P(Holm-support rejects H1), P(pooled 'agrees' | reject), P(bootstrap 'agrees' |
    reject), P(supported), P(suggestive) for R(3) in {0.8 (boundary), 0.9, 1.0}; two Holm contexts
    (other support-family p's favourable / unfavourable).
S2  Item x config interaction tau: type I of the paired t (conditional on the frozen items) at the
    boundary, and the SD of the estimator over item redraws vs the nested (project) and crossed
    bootstrap SDs.
S3  H3 equated-pair sign check: veto rate of a true D3 = +0.2 when the pair's level gap G is small,
    per-seed-R mean (as implemented) vs seed-pooled ratio (proposed), at single-level noise.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/home/sedrica-lynx/Downloads/outeropt")
sys.path.insert(0, str(REPO / "src"))
from hybridlc.analysis.pipeline import _agrees, _boot_ci, boot_L  # noqa: E402
from hybridlc.analysis.scores import CellTable, family_spec  # noqa: E402
from hybridlc.analysis.stats import (hierarchical_bootstrap, holm, l_contrast, one_sample_t,  # noqa: E402
                                     pooled_contrast, recovery)

OUT = Path(__file__).with_name("sim_verdicts.json")
STRATA = [("R1", lv, T) for T in (16384, 32768) for lv in ("D3", "D15")] + \
         [("R-load", lv, T) for T in (16384, 32768) for lv in ("N8", "N32", "N128", "N512", "N1024")]
CONFIGS = ["C0", "C1", "C3", "C12"]


def world(rng, R3, R1, n_seeds, n_ctx, G=0.30, s0=0.35, sigma_run=0.025, sigma_seed=0.02, item_sd=0.15,
          ctx_noise=0.22, tau=0.0, items=None):
    """CellTable with exact linear truth. ``items``: optional fixed (u, b) per stratum (frozen item set)."""
    d = np.linspace(-0.1, 0.1, len(STRATA))
    m0 = s0 + 0.5 * d
    m12 = m0 + G + 0.3 * d
    mu = {"C0": m0, "C12": m12, "C3": m0 + R3 * (m12 - m0), "C1": m0 + R1 * (m12 - m0)}
    seed_eff = rng.normal(0, sigma_seed, n_seeds)
    run_eff = rng.normal(0, sigma_run, (len(CONFIGS), n_seeds))
    ctx, ids, new_items = [], [], []
    for t_i, _ in enumerate(STRATA):
        if items is None:
            u = rng.normal(0, item_sd, n_ctx)
            b = rng.normal(0, tau, (n_ctx, len(CONFIGS))) if tau > 0 else np.zeros((n_ctx, len(CONFIGS)))
        else:
            u, b = items[t_i]
        new_items.append((u, b))
        arr = np.empty((n_ctx, len(CONFIGS), n_seeds))
        for c_i, c in enumerate(CONFIGS):
            arr[:, c_i, :] = (mu[c][t_i] + seed_eff[None, :] + run_eff[c_i][None, :] + u[:, None] + b[:, c_i][:, None]
                              + rng.normal(0, ctx_noise, (n_ctx, n_seeds)))
        ctx.append(arr)
        ids.append(np.arange(n_ctx))
    tab = CellTable("CS", "em_tf", list(STRATA), list(CONFIGS), list(range(n_seeds)), ctx, ids,
                    np.ones((len(CONFIGS), n_seeds), bool))
    return tab, new_items, mu


def true_L(mu, w, k="C3"):
    return float(w @ (mu[k] - 0.2 * mu["C0"] - 0.8 * mu["C12"]))


def s1(n_rep=300, B=1000, n_seeds=5, n_ctx=300):
    rng = np.random.default_rng(101)
    out = {}
    for R3 in (0.8, 0.85, 0.9, 1.0):
        for holm_ctx in ("favourable", "unfavourable"):
            rec = dict(reject=0, pooled_ag=0, boot_ag=0, supported=0, suggestive=0, raw_t_reject=0)
            Ls = []
            t0 = time.time()
            for r in range(n_rep):
                tab, _, mu = world(rng, R3, 0.4, n_seeds, n_ctx)
                w = tab.ladder_weights(family_spec("retrieval"), [16384, 32768])
                sc = tab.scores(w)
                L3 = l_contrast(sc.loc["C3"], sc.loc["C0"], sc.loc["C12"]).to_numpy()
                L1 = l_contrast(sc.loc["C1"], sc.loc["C0"], sc.loc["C12"]).to_numpy()
                p3 = float(one_sample_t(L3, "greater").p)
                p1 = float(one_sample_t(L1, "greater").p)
                Ls.append(L3.mean())
                if holm_ctx == "favourable":
                    ps = {"H1": p3, "H1-min": 1e-6, "H3": 1e-6, "H5a": 1e-6}
                else:
                    ps = {"H1": p3, "H1-min": p1, "H3": 1.0, "H5a": float(rng.uniform())}
                hh = holm(ps)
                rej = hh["H1"]["reject"]
                rec["raw_t_reject"] += p3 < 0.05
                if not rej:
                    continue
                rec["reject"] += 1
                pg = pooled_contrast(sc.loc[CONFIGS], {"C3": 1.0, "C0": -0.2, "C12": -0.8}, "greater")
                pag = _agrees(float(pg.lower), float("inf"), "greater")
                dr = hierarchical_bootstrap(tab, {"retrieval@primary": w}, configs=["C0", "C3", "C12"], B=B,
                                            key=f"s1-{R3}-{holm_ctx}-{r}")
                ci = _boot_ci(boot_L(dr, "C3"))
                bag = _agrees(ci["lower95_one_sided"], ci["upper95_one_sided"], "greater")
                rec["pooled_ag"] += pag
                rec["boot_ag"] += bag
                rec["supported"] += pag and bag
                rec["suggestive"] += not (pag and bag)
            nr = max(rec["reject"], 1)
            out[f"R3={R3}|{holm_ctx}"] = dict(
                n_rep=n_rep, true_L=true_L(mu, w), mean_Lhat=float(np.mean(Ls)),
                P_raw_t_reject=rec["raw_t_reject"] / n_rep, P_holm_reject=rec["reject"] / n_rep,
                P_pooled_agrees_given_reject=rec["pooled_ag"] / nr, P_boot_agrees_given_reject=rec["boot_ag"] / nr,
                P_supported=rec["supported"] / n_rep, P_suggestive=rec["suggestive"] / n_rep,
                seconds=round(time.time() - t0, 1))
            print(f"S1 R3={R3} {holm_ctx}: {out[f'R3={R3}|{holm_ctx}']}", flush=True)
    return out


def crossed_boot_sd(tab, w, B, rng):
    ci = [tab.configs.index(c) for c in ("C0", "C3", "C12")]
    S = len(tab.seeds)
    used = np.nonzero(w)[0]
    vals = np.empty(B)
    for b in range(B):
        pick = rng.integers(0, S, S)
        sc = np.zeros((3, S))
        for s_idx in used:
            X = tab.ctx[s_idx][:, ci, :]
            draw = rng.integers(0, X.shape[0], X.shape[0])
            sc += w[s_idx] * X[draw].mean(0)[:, pick]
        vals[b] = (sc[1] - 0.2 * sc[0] - 0.8 * sc[2]).mean()
    return float(vals.std()), float(np.quantile(vals, 0.05))


def s2(n_rep=300, n_seeds=5, n_ctx=300, B=400):
    rng = np.random.default_rng(202)
    out = {}
    for tau in (0.0, 0.1, 0.2, 0.3):
        rej_fixed, rej_redraw = 0, 0
        est_fixed, est_redraw = [], []
        nested_sd, crossed_sd, t_se = [], [], []
        nested_rej, crossed_rej = 0, 0
        _, frozen, _ = world(rng, 0.8, 0.4, n_seeds, n_ctx, tau=tau)  # one frozen item set
        for r in range(n_rep):
            # (a) frozen items (what the study does): new seeds/runs, same items
            tab, _, mu = world(rng, 0.8, 0.4, n_seeds, n_ctx, tau=tau, items=frozen)
            w = tab.ladder_weights(family_spec("retrieval"), [16384, 32768])
            sc = tab.scores(w)
            L = l_contrast(sc.loc["C3"], sc.loc["C0"], sc.loc["C12"]).to_numpy()
            tt = one_sample_t(L, "greater")
            rej_fixed += tt.p < 0.05
            est_fixed.append(L.mean())
            # (b) item population: items redrawn every replicate
            tab2, _, _ = world(rng, 0.8, 0.4, n_seeds, n_ctx, tau=tau)
            sc2 = tab2.scores(w)
            L2 = l_contrast(sc2.loc["C3"], sc2.loc["C0"], sc2.loc["C12"]).to_numpy()
            tt2 = one_sample_t(L2, "greater")
            rej_redraw += tt2.p < 0.05
            est_redraw.append(L2.mean())
            t_se.append(float(tt2.se))
            if r < 60:
                dr = hierarchical_bootstrap(tab2, {"retrieval@primary": w}, configs=["C0", "C3", "C12"], B=B,
                                            key=f"s2-{tau}-{r}")
                bl = boot_L(dr, "C3")
                nested_sd.append(float(np.std(bl)))
                nested_rej += float(np.quantile(bl, 0.05)) > 0
                csd, clo = crossed_boot_sd(tab2, w, B, rng)
                crossed_sd.append(csd)
                crossed_rej += clo > 0
        n_b = min(60, n_rep)
        out[f"tau={tau}"] = dict(
            n_rep=n_rep, true_L=0.0,
            typeI_t_frozen_items=rej_fixed / n_rep, frozen_item_offset=float(np.mean(est_fixed)),
            typeI_t_items_redrawn=rej_redraw / n_rep,
            sd_Lhat_items_redrawn=float(np.std(est_redraw)), mean_t_se=float(np.mean(t_se)),
            mean_nested_boot_sd=float(np.mean(nested_sd)), mean_crossed_boot_sd=float(np.mean(crossed_sd)),
            nested_boot_lower_gt0=nested_rej / n_b, crossed_boot_lower_gt0=crossed_rej / n_b)
        print(f"S2 tau={tau}: {out[f'tau={tau}']}", flush=True)
    return out


def s3(n_rep=4000, n=5):
    rng = np.random.default_rng(303)
    out = {}
    for G in (0.05, 0.10, 0.15, 0.25, 0.40):
        for sig in (0.03, 0.05):
            veto_ps, veto_pool = 0, 0
            for _ in range(n_rep):
                base = 0.5 + rng.normal(0, 0.02, n)
                e = lambda: rng.normal(0, sig, n)  # noqa: E731  single-level run noise
                r0, r3, r12 = base + e(), base + 0.9 * G + e(), base + G + e()
                h0, h3, h12 = base + e(), base + 0.7 * G + e(), base + G + e()
                d_ps = np.nanmean(recovery(r3, r0, r12) - recovery(h3, h0, h12))  # as implemented
                d_pool = recovery(r3.mean(), r0.mean(), r12.mean()) - recovery(h3.mean(), h0.mean(), h12.mean())
                veto_ps += not (d_ps > 0)
                veto_pool += not (d_pool > 0)
            out[f"G={G}|sigma_level={sig}"] = dict(veto_rate_per_seed_R_mean=veto_ps / n_rep,
                                                   veto_rate_pooled_ratio=veto_pool / n_rep)
    print("S3", json.dumps(out), flush=True)
    return out


if __name__ == "__main__":
    which = sys.argv[1:] or ["s3", "s2", "s1"]
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    for w in which:
        res[w.upper()] = {"s1": s1, "s2": s2, "s3": s3}[w]()
        OUT.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))

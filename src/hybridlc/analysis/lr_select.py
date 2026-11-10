"""LR selection by minimax regret (protocol §7.2, I-26; S3, decided at G-LR).

Input: the final LM-2 subset loss of each LR-sweep run, ``losses[k][eta]`` for k in {0, 3, 12}
(:data:`~.protocol.LR_GRID_K`) and eta on the grid (plus edge extensions). The rule, verbatim from §7.2:

* regret_k(eta) = loss_k(eta) - min_eta' loss_k(eta');
* **divergence**: NaN (or a missing final loss), or final loss > min + 0.5, counts as infinite regret;
* eta* = argmin_eta max_k regret_k(eta);
* **shared eta*** iff max_k regret_k(eta*) <= 0.02 nats; otherwise per-config eta*_k (each k's own
  argmin), with log eta interpolated linearly in k for k not in {0, 3, 12};
* **edge extension**: if any optimum lies at a grid edge, add 0.7e-3 or 4.0e-3 for all three k and
  re-apply the rule (:func:`select_lr` reports the extension needed in ``extend``);
* **inheritance**: controls and Tier-B/C arms inherit eta* of their k (:func:`eta_for_config`).

[IMPL] choices (the protocol does not fix them): ties in the minimax criterion are broken by the
smaller sum of finite regrets, then the smaller eta; "optimum" for the edge rule means the minimax
eta* and every per-k argmin; an eta not run for some k counts as infinite regret for that k; for
k > 12 or k < 0 the nearest end point's log eta is used (no extrapolation).

:func:`read_sweep_losses` reads the final ``kind == "val"`` rows of ``runs/<run_id>/metrics.jsonl``.
"""
from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from . import protocol as P


def _finite(x: Any) -> bool:
    return x is not None and isinstance(x, (int, float)) and math.isfinite(float(x))


def regrets(losses: Mapping[int, Mapping[float, float | None]],
            margin: float = P.LR_DIVERGENCE_MARGIN) -> dict[int, dict[float, float]]:
    """regret_k(eta) with divergence (NaN / missing / > min + margin) as +inf."""
    out: dict[int, dict[float, float]] = {}
    etas = sorted({float(e) for byk in losses.values() for e in byk})
    for k, byk in losses.items():
        fin = [float(v) for v in byk.values() if _finite(v)]
        if not fin:
            out[int(k)] = {e: math.inf for e in etas}
            continue
        m = min(fin)
        r = {}
        for e in etas:
            v = byk.get(e, byk.get(float(e)))
            r[e] = math.inf if (not _finite(v) or float(v) > m + margin) else float(v) - m
        out[int(k)] = r
    return out


def select_lr(losses: Mapping[int, Mapping[float, float | None]], *,
              ks: tuple[int, ...] = P.LR_GRID_K, max_shared_regret: float = P.LR_SHARED_MAX_REGRET,
              margin: float = P.LR_DIVERGENCE_MARGIN) -> dict[str, Any]:
    """Apply the §7.2 rule. Returns a JSON-friendly dict:

    ``eta_star`` (minimax eta), ``max_regret`` (its max_k regret), ``shared`` (bool), ``per_k``
    (eta*_k = each k's argmin, used when not shared), ``eta_by_k`` (what each grid k trains with),
    ``extend`` (edge extensions still to run, [] if none), ``edge_after_extension`` (an optimum is at
    an edge although both extensions were run: report to the PI), ``regrets`` and ``etas``.
    """
    missing = [k for k in ks if k not in losses]
    if missing:
        raise ValueError(f"LR sweep lacks k = {missing}")
    R = regrets({k: losses[k] for k in ks}, margin)
    etas = sorted({e for r in R.values() for e in r})
    if not etas:
        raise ValueError("no learning rates in the sweep")

    def crit(e: float) -> tuple[float, float, float]:
        rs = [R[k][e] for k in ks]
        return (max(rs), sum(x for x in rs if math.isfinite(x)), e)

    eta_star = min(etas, key=crit)
    max_reg = crit(eta_star)[0]
    if not math.isfinite(max_reg):
        raise ValueError("every learning rate diverges for some k (infinite minimax regret)")
    per_k = {k: min(etas, key=lambda e, k=k: (R[k][e], e)) for k in ks}
    shared = bool(max_reg <= max_shared_regret)
    optima = {eta_star, *per_k.values()}
    lo, hi = min(etas), max(etas)
    extend = []
    if lo in optima and not any(math.isclose(e, P.LR_EDGE_LOW) for e in etas):
        extend.append(P.LR_EDGE_LOW)
    if hi in optima and not any(math.isclose(e, P.LR_EDGE_HIGH) for e in etas):
        extend.append(P.LR_EDGE_HIGH)
    at_edge = bool((lo in optima or hi in optima) and not extend)
    return dict(eta_star=eta_star, max_regret=max_reg, shared=shared, per_k={int(k): v for k, v in per_k.items()},
                eta_by_k={int(k): (eta_star if shared else per_k[k]) for k in ks}, extend=extend,
                edge_after_extension=at_edge and lo != hi, etas=etas,
                regrets={int(k): {repr(e): (v if math.isfinite(v) else "inf") for e, v in r.items()}
                         for k, r in R.items()},
                rule="§7.2 minimax regret", final=not extend)


def eta_for_k(sel: Mapping[str, Any], k: int) -> float:
    """eta for any k: the shared eta*, or log eta interpolated linearly in k between grid k's."""
    if sel["shared"]:
        return float(sel["eta_star"])
    pts = sorted((int(kk), float(v)) for kk, v in sel["eta_by_k"].items())
    exact = dict(pts)
    if k in exact:
        return exact[k]
    if k <= pts[0][0]:
        return pts[0][1]
    if k >= pts[-1][0]:
        return pts[-1][1]
    for (k0, e0), (k1, e1) in zip(pts, pts[1:]):
        if k0 <= k <= k1:
            t = (k - k0) / (k1 - k0)
            return float(math.exp((1 - t) * math.log(e0) + t * math.log(e1)))
    raise AssertionError("unreachable")


def eta_for_config(sel: Mapping[str, Any], config: str) -> float:
    """Inheritance (§7.2): a control / Tier-B/C arm uses eta* of its k (``X-gc12`` -> k = 12)."""
    from .results import config_k

    k = config_k(config)
    if k is None:
        raise ValueError(f"cannot infer k for config {config!r}")
    return eta_for_k(sel, k)


def final_val_loss(metrics_jsonl: str | Path) -> float | None:
    """Final LM-2 loss of a run: the last ``kind == "val"`` row with ``final`` true (else None)."""
    p = Path(metrics_jsonl)
    if not p.is_file():
        return None
    last = None
    with p.open() as fh:
        for line in fh:
            if '"val"' not in line:
                continue
            r = json.loads(line)
            if r.get("kind") == "val" and r.get("final"):
                last = r.get("val_loss")
    return None if last is None else float(last)


def read_sweep_losses(runs: Mapping[str, tuple[int, float]], runs_root: str | Path = "runs"
                      ) -> dict[int, dict[float, float | None]]:
    """``runs`` maps run_id -> (k, eta); a run without a final val row counts as diverged (None)."""
    out: dict[int, dict[float, float | None]] = {}
    for rid, (k, eta) in runs.items():
        out.setdefault(int(k), {})[float(eta)] = final_val_loss(Path(runs_root) / rid / "metrics.jsonl")
    return out

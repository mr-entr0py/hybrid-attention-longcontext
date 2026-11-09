"""Pre-registered decision rules and wording labels (protocol §2, §11.3, §11.6, §11.7, §11.9).

:func:`compute_verdicts` maps the results dict of :func:`~.pipeline.run_analysis` to one verdict per
hypothesis. The vocabulary is fixed by §2:

* **supported**: the stated test rejects in the pre-registered direction after Holm (H3 additionally
  needs the difficulty-equated sign check, U-11);
* **falsified** (H1, H1-min only): all four I-22 guards hold (§11.6);
* **suggestive (supported)**: supported, but a sensitivity analysis (pooled-variance model or
  hierarchical bootstrap) does not agree in sign (§11.3). Agreement [IMPL]: the analysis' one-sided
  95% bound in the verdict's direction excludes zero;
* named negative outcomes when their TOST holds ("no task dependence", "no length effect", ...);
* **inconclusive** otherwise.

Labels attached to verdicts: "metric-robust", "H1 at 8K, not a long-context result",
"conditional on reference adequacy", "exposure-robust", "conditional on the long-stage budget". The
wording block applies §11.9 (package claims; operator-level wording only if δ_gc <= 3 pp or both mixer
families give the same verdict; scope).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from . import protocol as P

SUPPORTED, FALSIFIED, INCONCLUSIVE, SUGGESTIVE = "supported", "falsified", "inconclusive", "suggestive (supported)"

_STATEMENT = {
    "H1": "{k} F_g layers of 12 (the rest GDN-A as built) recover >= 80% of the k = 0 -> 12 retrieval gap",
    "H1-min": "{k} F_g layer of 12 (the rest GDN-A as built) recovers >= 80% of the k = 0 -> 12 retrieval gap",
    "H3": "retrieval recovers more of its gap than multi-hop at k = 3 (D3 > 0)",
    "H5a": "the k = 12 vs k = 0 retrieval gap grows with context length (slope per doubling > 0)",
}


def _fmt(x: Any, nd: int = 3) -> str:
    try:
        return f"{float(x):+.{nd}f}"
    except (TypeError, ValueError):
        return "n/a"


def _sens_agree(sens: Mapping[str, Any], name: str, direction: str) -> tuple[bool, list[str]]:
    rec = sens.get(name, {})
    bad = [m for m, r in rec.items() if isinstance(r, dict) and not r.get(f"agrees_{direction}", False)]
    return (not bad), bad


def confirmatory_verdict(name: str, res: Mapping[str, Any]) -> dict[str, Any]:
    t = res["tests"][name]
    hs = res["holm_support"][name]
    labels: list[str] = []
    notes: list[str] = []
    verdict = INCONCLUSIVE
    supported = bool(hs["reject"])
    if name == "H3":
        if not t.get("evaluable", False):
            notes.append("not evaluable (some seed has G < 15 pp in a family); entered Holm with p = 1")
            supported = False
        elif supported and not t.get("equated_sign_consistent", True):
            notes.append("Holm rejects but D3 changes sign at difficulty-equated levels (U-11)")
            supported = False
    if supported:
        ok, bad = _sens_agree(res.get("sensitivity", {}), name, "greater")
        verdict = SUPPORTED if ok else SUGGESTIVE
        if not ok:
            notes.append(f"downgraded: sensitivity analyses {bad} do not agree in sign (§11.3)")
    if name in P.FALSIFICATION_FAMILY:
        g = res.get("guards", {}).get(name, {})
        if g.get("holds") and not supported:
            verdict = FALSIFIED
        elif not supported:
            failed = [k for k in ("test", "dose", "saturation") if not g.get(k, {}).get("holds", False)]
            if not res.get("guards", {}).get("gates", {}).get("holds", False):
                failed.insert(0, "gates")
            if res["holm_falsification"][name]["reject"]:
                notes.append(f"falsification-direction test rejects but guards fail: {failed}")
        sat = g.get("saturation", {})
        if sat.get("label"):
            labels.append(sat["label"])
    if verdict == INCONCLUSIVE:
        tost = t.get("tost")
        if name == "H3" and tost and tost.get("equivalent"):
            verdict = "no task dependence"
        if name == "H5a" and tost and tost.get("equivalent"):
            verdict = "no length effect"
    # labels
    if name in ("H1", "H1-min") and "logp" in t:
        if (t["logp"]["estimate"] > 0) == (t["estimate"] > 0):
            labels.append("metric-robust")
    if name != "H5a" and res["primary"].get("label"):
        labels.append(res["primary"]["label"])
    if name == "H5a":
        if t.get("conditional_on_reference_adequacy"):
            labels.append("conditional on reference adequacy")
        if t.get("exposure_robust"):
            labels.append("exposure-robust")
    k = P.H1_K if name == "H1" else P.H1MIN_K
    stmt = _STATEMENT[name].format(k=k)
    lengths = "+".join(f"{T // 1024}K" for T in (t.get("lengths") or res["primary"]["lengths"]))
    sentence = (f"{name} {verdict}: {stmt} [estimate {_fmt(t.get('estimate'))}, Holm-adjusted one-sided p = "
                f"{hs['p_holm']:.4g}; cell {lengths}; n = {len(t.get('per_seed') or {})} seeds]")
    return dict(verdict=verdict, labels=labels, notes=notes, estimate=t.get("estimate"), p_raw=hs["p"],
                p_holm=hs["p_holm"], p_falsify_holm=(res["holm_falsification"].get(name, {}).get("p_holm")
                                                    if name in P.FALSIFICATION_FAMILY else None),
                sentence=sentence)


def secondary_verdicts(res: Mapping[str, Any]) -> dict[str, Any]:
    sec = res.get("secondary", {})
    hs = sec.get("holm_secondary", {})
    out: dict[str, Any] = {}
    h4 = sec.get("H4", {})
    if h4.get("evaluable"):
        rej = [c for c in ("c1", "c2") if hs.get(f"H4-{c}", {}).get("reject")]
        out["H4"] = dict(verdict=("placement effect (" + ", ".join(rej) + ")") if rej else
                         ("no placement effect" if h4.get("no_placement_effect") else INCONCLUSIVE),
                         p_holm={c: hs.get(f"H4-{c}", {}).get("p_holm") for c in ("c1", "c2")})
    else:
        out["H4"] = dict(verdict="not evaluated", reason=h4.get("reason"))
    for name, neg in (("H6prime", "equivalent"), ("H5b", "no difference in length slopes")):
        r = sec.get(name, {})
        if not r.get("evaluable"):
            out[name] = dict(verdict="not evaluated", reason=r.get("reason"))
            continue
        v = "differs" if hs.get(name, {}).get("reject") else (neg if r.get("tost", {}).get("equivalent") else INCONCLUSIVE)
        out[name] = dict(verdict=v, estimate=r.get("estimate"), p_holm=hs.get(name, {}).get("p_holm"))
    h2 = res.get("descriptive", {}).get("H2")
    if h2:
        out["H2"] = dict(verdict="linear or convex" if h2.get("linear_or_convex") else
                         "departs from linear (descriptive; see isotonic and saturating fits)",
                         k80=h2.get("k80"), k80_ci95=h2.get("k80_bootstrap_ci95"), descriptive=True)
    ni = sec.get("noninferiority")
    if ni:
        out["noninferiority_S12_minus_S3"] = dict(verdict="non-inferior" if ni["noninferior"] else "not shown",
                                                  estimate=ni["estimate"], upper95=ni["upper95"])
    return out


def wording(res: Mapping[str, Any], conf: Mapping[str, Any]) -> dict[str, Any]:
    trig = res.get("triggers", {})
    u02 = trig.get("U02")
    fam = res.get("fgc_family", {})
    same = None
    if fam and all(n in fam for n in ("H1", "H1-min")):
        fv = {n: ("supported" if fam["holm_support"][n]["reject"] else "not supported") for n in ("H1", "H1-min")}
        cv = {n: ("supported" if conf[n]["verdict"] in (SUPPORTED, SUGGESTIVE) else "not supported")
              for n in ("H1", "H1-min")}
        same = fv == cv
    if u02 is None:
        op_ok, why = False, "δ_gc not available (X-gc12 missing)"
    elif u02["delta_gc"] <= P.U02_TIER_B:
        op_ok, why = True, f"δ_gc = {u02['delta_gc']:+.3f} <= 3 pp"
    elif same:
        op_ok, why = True, "both mixer families give the same verdict"
    else:
        op_ok, why = False, f"δ_gc = {u02['delta_gc']:+.3f} > 3 pp and the F_gc family verdicts differ or are missing"
    return dict(package_claim=P.PACKAGE_CLAIM, operator_level_wording_allowed=op_ok, operator_level_reason=why,
                scope=P.SCOPE_STATEMENT, elicited_vs_zero_shot=(
                    "Claims are scoped to capability elicited after identical exposure (TEF); Tier-0 "
                    "disagreements are reported (I-27)."),
                fgc_co_primary=bool(u02 and u02.get("co_primary")), fgc_tier_b=bool(u02 and u02.get("tier_b")))


def compute_verdicts(res: Mapping[str, Any]) -> dict[str, Any]:
    conf = {h: confirmatory_verdict(h, res) for h in P.SUPPORT_FAMILY}
    out = dict(confirmatory=conf, secondary=secondary_verdicts(res), wording=wording(res, conf),
               primary_cell=res["primary"], seed_trigger=res.get("seed_trigger"),
               success_counts=res.get("success_counts"), warnings=res.get("warnings", []))
    return out


def dumps(v: Mapping[str, Any]) -> str:
    return json.dumps(v, indent=1, allow_nan=False)

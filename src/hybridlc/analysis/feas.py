"""DTS-10 FEAS decision code (``experiments/prereg/DTS10_acceptance.v2.yaml``; FEAS-PREC v2 Q1-Q4).

Every S2 FEAS verdict is computed here and only here (PI rulings D082, D083; no flip rate, contrast or ladder value
is judged by hand). The order is fixed by the pre-registration:

1. load every input; a missing cell, seed, pass (P0-P4), LM-2 value or a NaN anywhere -> :class:`FeasInputError`
   naming *all* missing items (never impute);
2. FEAS-PREC v2 Q1, Q2, Q3 and Q4 (``precision``); Q4 computes every ladder rule (FEAS-3, FEAS/U-01, FEAS-1/2,
   FEAS-5, elicitation) from P0 (or, for a checkpoint whose Q2 failed, its single-pass bf16 re-scores) and from P4,
   and selects the score source;
3. the ladder rules from the selected source; U-15 and the TEF forgetting gate from P0 / LM-2; I-08 from P0 (and
   P1 if any Q4 trigger fired);
4. the verdict flags F-nearmiss, F-infra, F-precision.

Arithmetic: scores are EM_tf means from integer counts, held as :class:`fractions.Fraction`; thresholds are exact
fractions (25 pp = 1/4); no epsilon. LM-2 values (nats) enter as the exact rational of their shortest decimal
representation (:func:`dec`).

Inputs (all paths are explicit; nothing is read implicitly):

* ``results_root/<run_id>/<ckpt>[-P1|-P4|-P2full]/FD/<task>_<level>_<T|C0>.jsonl`` (scripts/evaluate.py rows with
  ``item_id``, ``EM_tf``, ``logp_gold``) and ``<cell>_rescore.json`` beside each P0 file (P2 and P3 records);
* the design: one :class:`DesignCell` per FEAS TEF run (config, k, arm, seed, branch, scale, run_id), from the
  registry's ``queued`` rows matched to the generated FEAS run YAMLs (``genconfigs.feas_design``), or a JSON file;
* LM-2 before / after the TEF per TEF run (``runs/<run_id>/metrics.jsonl`` ``val`` records, or a JSON file);
* the registry (``gdiv_near_miss`` notes; ``infra_failed_attempt`` / ``suspect_sw_failure`` rows; lineage).

Readings fixed in code before any FEAS data exist (beyond the two stated in the pre-registration), each also
written into the rule trace:

* R-ssmax-branch: FEAS-5 reads branch b*; the SSMax arm exists only on the 30M branch (plan S2-DEC-SSMAX "small
  branch"). If b* is 90M, FEAS-5 is computed on the 30M branch (like-for-like plain vs SSMax) and the verdict is
  marked ``needs_pi_row``.
* R-i08-c0: I-08's "every FD level at every FD length" covers the length-bearing FD cells; the C0 (no-filler)
  controls have no FD length and are not part of a family's levels.
* R-d512-seeds: FEAS-3 on the FEAS-512 path averages over the seeds present (seed 101 by design).
* R-contrast-branch: C(pass, s) is computed on the b* that the same pass's FEAS-3 selects (a differing b* is a
  FEAS-3 outcome difference, which switches the source anyway).
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from fractions import Fraction as F
from pathlib import Path
from typing import Any, Callable, Iterable

# ------------------------------------------------------------------------------------------ constants
SEEDS = (101, 102)
K_LEVELS = (0, 12)
T4K, T8K, T16K, T32K = 4096, 8192, 16384, 32768
R1_LEVELS = ("D3", "D15")
RLOAD_LEVELS = {T32K: ("N8", "N32", "N128", "N512", "N1024"), T16K: ("N8", "N32", "N128", "N512", "N1024"),
                T8K: ("N8", "N32", "N128", "N512"), T4K: ("N8", "N32")}
RLOAD_C0 = ("N8", "N32", "N128", "N512", "N1024")
MHOP_LEVELS, MHOP_T = ("h1", "h2", "h3", "h4"), (T8K, T32K)
AFREQ_LEVELS, AFREQ_T = ("rho1.25", "rho1.5", "rho2", "rho4"), (T8K, T32K)
R1_T = (T4K, T8K, T16K, T32K)
M_QUERIES = 4

THR = {
    "feas3_small": F(3, 100), "feas3_mid": F(10, 100),
    "u01_ratio": F(85, 100), "u01_gap": F(25, 100),
    "d512_ratio": F(80, 100), "d512_gap": F(20, 100),
    "ssmax_gain": F(3, 100), "ssmax_loss": F(2, 100),
    "forget_rise": F(5, 100), "forget_diff": F(2, 100),
    "elicit": F(1, 2), "floor": F(5, 100), "ceiling": F(95, 100),
    "flip": F(1, 100), "info_lo": F(5, 100), "info_hi": F(95, 100), "dC": F(2, 100),
    "de": F(2),
}
PASSES = ("P0", "P1", "P2", "P3", "P4")
LADDER_RULES = ("FEAS-3", "U-01", "FEAS-1/2", "FEAS-5", "elicitation")
VERDICTS = ("FEAS-3", "U-01", "FEAS-1/2", "FEAS-5", "TEF-forgetting", "elicitation", "U-15", "I-08")
INFRA_STATUSES = ("infra_failed_attempt", "suspect_sw_failure")


class FeasInputError(Exception):
    """Missing or invalid inputs; ``missing`` lists every item (the CLI exits non-zero and names them)."""

    def __init__(self, missing: list[str]):
        self.missing = sorted(set(missing))
        super().__init__(f"{len(self.missing)} missing/invalid FEAS input(s): " + "; ".join(self.missing[:50]))


# ------------------------------------------------------------------------------------------ cells
def cell_name(task: str, level: str, T: int | str) -> str:
    return f"{task}_{level}_{T}"


def fd_cells() -> list[str]:
    """Every FD result-cell name (frozen FD design; checked against the items manifest in the tests)."""
    out = [cell_name("R1", d, t) for t in R1_T for d in R1_LEVELS]
    out += [cell_name("R-load", n, t) for t in (T4K, T8K, T16K, T32K) for n in RLOAD_LEVELS[t]]
    out += [cell_name("R1", d, "C0") for d in R1_LEVELS] + [cell_name("R-load", n, "C0") for n in RLOAD_C0]
    out += [cell_name("M-hop", h, t) for t in MHOP_T for h in MHOP_LEVELS]
    out += [cell_name("A-freq", r, t) for t in AFREQ_T for r in AFREQ_LEVELS]
    return out


def ladder_cells(T: int, c0: bool = False) -> tuple[list[str], list[str]]:
    """(R1 cells, R-load cells) of the retrieval ladder at length T (or its C0 counterparts of the same levels)."""
    tt: int | str = "C0" if c0 else T
    return [cell_name("R1", d, tt) for d in R1_LEVELS], [cell_name("R-load", n, tt) for n in RLOAD_LEVELS[T]]


def p4_cells(scale: str = "D384") -> list[str]:
    """FEAS ladder verdict cells scored by P4 (FEAS_precision.v2.yaml): 4K and 32K ladders and their C0 cells;
    on FEAS-512 checkpoints also the 16K ladder."""
    Ts = (T4K, T32K) + ((T16K,) if scale == "D512" else ())
    out: list[str] = []
    for T in Ts:
        a, b = ladder_cells(T)
        c, d = ladder_cells(T, c0=True)
        out += a + b + c + d
    return sorted(set(out))


def cell_T(cell: str) -> int | None:
    t = cell.rsplit("_", 1)[1]
    return None if t == "C0" else int(t)


def cell_family(cell: str) -> str:
    task = cell.split("_", 1)[0]
    return {"R1": "retrieval", "R-load": "retrieval", "M-hop": "multi-hop", "A-freq": "aggregation"}[task]


# ------------------------------------------------------------------------------------------ design
@dataclass(frozen=True)
class DesignCell:
    run_id: str          # the TEF run whose checkpoint is scored
    config: str
    k: int
    arm: str             # plain | ssmax
    seed: int
    branch: str          # 30M | 90M (D384); 100M | 300M (D512)
    scale: str = "D384"
    tef: str = "9010"    # 9010 | 5050 | 5050lr1e-4 | answer_only (conditional reruns)
    lineage: tuple[str, ...] = ()   # ancestor run_ids (Stage B, Stage A) for F-nearmiss / F-infra


def small_large(scale: str) -> tuple[str, str]:
    return ("30M", "90M") if scale == "D384" else ("100M", "300M")


def load_design_json(path: str | Path) -> list[DesignCell]:
    rows = json.loads(Path(path).read_text())
    return [DesignCell(**{**r, "lineage": tuple(r.get("lineage", ()))}) for r in rows]


def read_registry(path: str | Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    p = Path(path)
    if not p.is_file():
        raise FeasInputError([f"registry {p}"])
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def _parents(rows: list[dict[str, Any]]) -> dict[str, str]:
    par: dict[str, str] = {}
    for r in rows:
        if r.get("run_id") and r.get("parent_run_id"):
            par[r["run_id"]] = r["parent_run_id"]
    return par


def lineage_of(run_id: str, parents: dict[str, str]) -> tuple[str, ...]:
    out, cur = [], run_id
    while cur in parents and len(out) < 8:
        cur = parents[cur]
        out.append(cur)
    return tuple(out)


def design_from_registry(rows: list[dict[str, Any]]) -> list[DesignCell]:
    """FEAS TEF runs from the registry's ``queued`` rows whose run YAML is a generated FEAS TEF YAML."""
    from hybridlc.training.genconfigs import feas_design

    by_yaml = {Path(d["yaml"]).name: d for d in feas_design()}
    parents = _parents(rows)
    out: dict[str, DesignCell] = {}
    for r in rows:
        if r.get("status") != "queued" or not r.get("config"):
            continue
        d = by_yaml.get(Path(str(r["config"])).name)
        if d is None:
            continue
        rid = r["run_id"]
        out[rid] = DesignCell(run_id=rid, config=d["config"], k=int(d["k"]), arm=d["arm"], seed=int(d["seed"]),
                              branch=d["branch"], scale=d["scale"], lineage=lineage_of(rid, parents))
    return list(out.values())


# ------------------------------------------------------------------------------------------ loading
def _finite(x: Any) -> bool:
    try:
        return x is not None and math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _em(v: Any) -> bool | None:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)) and _finite(v) and float(v) in (0.0, 1.0):
        return bool(v)
    return None


@dataclass
class Inputs:
    design: list[DesignCell]
    scores: dict[str, dict[str, dict[str, dict[str, bool]]]] = field(default_factory=dict)  # pass->run->cell->item->EM
    lm2: dict[str, dict[str, float]] = field(default_factory=dict)                          # run -> before/after
    registry: list[dict[str, Any]] = field(default_factory=list)
    ckpt: str = "tef1000"
    files_read: list[str] = field(default_factory=list)

    def em(self, pass_id: str, run_id: str, cell: str) -> dict[str, bool]:
        return self.scores.get(pass_id, {}).get(run_id, {}).get(cell, {})


def _read_rows(path: Path, missing: list[str]) -> dict[str, bool] | None:
    out: dict[str, bool] = {}
    try:
        lines = path.read_text().splitlines()
    except OSError:
        missing.append(f"unreadable {path}")
        return None
    for i, ln in enumerate(lines):
        if not ln.strip():
            continue
        try:
            r = json.loads(ln)
        except json.JSONDecodeError:
            missing.append(f"{path}:{i + 1} not JSON")
            continue
        em = _em(r.get("EM_tf", r.get("em_tf")))
        if em is None or not r.get("item_id"):
            missing.append(f"{path}:{i + 1} EM_tf/item_id missing or not boolean")
            continue
        if "logp_gold" in r and not _finite(r["logp_gold"]):
            missing.append(f"{path}:{i + 1} logp_gold NaN/inf")
            continue
        out[str(r["item_id"])] = em
    if not out:
        missing.append(f"empty {path}")
    return out


def _check_meta(path: Path, pass_id: str, missing: list[str]) -> None:
    meta = path.with_name(path.stem + "_meta.json")
    if meta.is_file():
        try:
            m = json.loads(meta.read_text())
        except json.JSONDecodeError:
            missing.append(f"{meta} not JSON")
            return
        got = m.get("pass_id")
        if got is not None and got != pass_id:
            missing.append(f"{meta}: pass_id {got!r} where {pass_id} expected")


def required_cells(pass_id: str, scale: str) -> list[str]:
    return p4_cells(scale) if pass_id == "P4" else fd_cells()


def load_inputs(results_root: str | Path, design: list[DesignCell], *, ckpt: str = "tef1000",
                lm2: dict[str, dict[str, float]] | None = None, runs_root: str | Path | None = None,
                registry: list[dict[str, Any]] | None = None, p2full: Iterable[str] = ()) -> Inputs:
    """Read every pass for every design cell; raise :class:`FeasInputError` naming everything that is missing.
    ``p2full``: run_ids whose single-pass bf16 full re-scores (``<ckpt>-P2full``) are needed (Q2 consequence)."""
    root = Path(results_root)
    missing: list[str] = []
    inp = Inputs(design=list(design), registry=list(registry or []), ckpt=ckpt)
    have = {(d.k, d.arm, d.seed, d.branch, d.scale, d.tef) for d in design}
    # design completeness (D384 FEAS: k x seed x {30M, 90M} plain, and the SSMax arm)
    for k in K_LEVELS:
        for s in SEEDS:
            for b in small_large("D384"):
                if (k, "plain", s, b, "D384", "9010") not in have:
                    missing.append(f"design: k={k} seed={s} branch={b} (D384 plain TEF run)")
    if not any(d.arm == "ssmax" and d.scale == "D384" for d in design):
        missing.append("design: SSMax arm (D384, k=12, seed 101)")
    passes = {"P0": ckpt, "P1": f"{ckpt}-P1", "P4": f"{ckpt}-P4"}
    for d in design:
        for pid, lab in passes.items():
            run_scores = inp.scores.setdefault(pid, {}).setdefault(d.run_id, {})
            for cell in required_cells(pid, d.scale):
                f = root / d.run_id / lab / "FD" / f"{cell}.jsonl"
                if not f.is_file():
                    missing.append(f"{pid} {d.run_id}/{lab}/FD/{cell}")
                    continue
                rows = _read_rows(f, missing)
                _check_meta(f, pid, missing)
                inp.files_read.append(str(f))
                if rows is not None:
                    run_scores[cell] = rows
                if pid == "P0":
                    rf = f.with_name(f.stem + "_rescore.json")
                    if not rf.is_file():
                        missing.append(f"P2/P3 {d.run_id}/{lab}/FD/{cell}_rescore.json")
                        continue
                    try:
                        rec = json.loads(rf.read_text())
                    except json.JSONDecodeError:
                        missing.append(f"{rf} not JSON")
                        continue
                    inp.files_read.append(str(rf))
                    for rp in ("P2", "P3"):
                        rr = rec.get(rp)
                        if not rr or not rr.get("rows"):
                            missing.append(f"{rp} {d.run_id}/{lab}/FD/{cell} (re-score record missing)")
                            continue
                        sub: dict[str, bool] = {}
                        for r in rr["rows"]:
                            em = _em(r.get("EM_tf"))
                            if em is None or ("logp_gold" in r and not _finite(r["logp_gold"])):
                                missing.append(f"{rp} {d.run_id}/{lab}/FD/{cell} item {r.get('item_id')} invalid/NaN")
                                continue
                            sub[str(r["item_id"])] = em
                        inp.scores.setdefault(rp, {}).setdefault(d.run_id, {})[cell] = sub
        need_p2full = d.run_id in set(p2full)
        p0r, p2r = inp.scores.get("P0", {}).get(d.run_id, {}), inp.scores.get("P2", {}).get(d.run_id, {})
        fl = sum(int(p0r[c][i] != p2r[c][i]) for c in p2r if c in p0r for i in p2r[c] if i in p0r[c])
        nn = sum(1 for c in p2r if c in p0r for i in p2r[c] if i in p0r[c])
        if nn and F(fl, nn) >= THR["flip"]:
            need_p2full = True     # Q2 FAILS for this checkpoint: its single-pass bf16 re-scores replace P0
        if need_p2full:
            lab = f"{ckpt}-P2full"
            for cell in p4_cells(d.scale):
                f = root / d.run_id / lab / "FD" / f"{cell}.jsonl"
                if not f.is_file():
                    missing.append(f"P2full {d.run_id}/{lab}/FD/{cell} (Q2 FAIL consequence: single-pass bf16 re-score)")
                    continue
                rows = _read_rows(f, missing)
                if rows is not None:
                    inp.scores.setdefault("P2full", {}).setdefault(d.run_id, {})[cell] = rows
    # LM-2 before/after for every D384 plain 90/10 TEF run (TEF forgetting gate reads b*, both seeds, k in {0, 12})
    step = _ckpt_step(ckpt)
    for d in design:
        if d.arm != "plain":
            continue
        v = (lm2 or {}).get(d.run_id)
        if v is None and runs_root is not None:
            v = lm2_from_metrics(Path(runs_root) / d.run_id / "metrics.jsonl", step)
        if not v or not _finite(v.get("before")) or not _finite(v.get("after")):
            missing.append(f"LM-2 before/after TEF for {d.run_id} (k={d.k} seed={d.seed} branch={d.branch})")
            continue
        inp.lm2[d.run_id] = {"before": float(v["before"]), "after": float(v["after"])}
    if missing:
        raise FeasInputError(missing)
    return inp


def _ckpt_step(ckpt: str) -> int | None:
    digits = "".join(ch for ch in ckpt if ch.isdigit())
    return int(digits) if digits else None


def lm2_from_metrics(path: Path, step: int | None) -> dict[str, float] | None:
    """LM-2 at TEF start (val record with after_steps == 0, written by ``validation.at_start``) and after
    ``step`` TEF steps (val record with after_steps == step)."""
    if not path.is_file():
        return None
    before = after = None
    for ln in path.read_text().splitlines():
        try:
            r = json.loads(ln)
        except json.JSONDecodeError:
            continue
        if r.get("kind") != "val":
            continue
        if r.get("after_steps") == 0:
            before = r.get("val_loss")
        if step is not None and r.get("after_steps") == step:
            after = r.get("val_loss")
    return {"before": before, "after": after}


# ------------------------------------------------------------------------------------------ JSON helpers
def jf(x: Any) -> Any:
    """JSON form: Fractions as {"frac": "p/q", "value": float}; containers recursively."""
    if isinstance(x, F):
        return {"frac": f"{x.numerator}/{x.denominator}", "value": float(x)}
    if isinstance(x, dict):
        return {str(k): jf(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [jf(v) for v in (sorted(x) if isinstance(x, set) else x)]
    if isinstance(x, DesignCell):
        return jf(asdict(x))
    return x


# ------------------------------------------------------------------------------------------ ladder
class _Src:
    """Score source for the ladder rules: a pass id, with per-run overrides (Q2 single-pass replacements)."""

    def __init__(self, inp: Inputs, pass_id: str, override: dict[str, str] | None = None):
        self.inp, self.pass_id, self.override = inp, pass_id, dict(override or {})

    @property
    def name(self) -> str:
        return self.pass_id + (f"+P2full[{','.join(sorted(self.override))}]" if self.override else "")

    def cell_em(self, run_id: str, cell: str) -> F:
        pid = self.override.get(run_id, self.pass_id)
        rows = self.inp.em(pid, run_id, cell)
        if not rows:
            raise FeasInputError([f"{pid} {run_id} {cell} (needed by a ladder rule)"])
        return F(sum(rows.values()), len(rows))


def ladder(src: _Src, run_id: str, T: int, c0: bool = False) -> F:
    r1, rl = ladder_cells(T, c0)
    m1 = sum((src.cell_em(run_id, c) for c in r1), F(0)) / len(r1)
    m2 = sum((src.cell_em(run_id, c) for c in rl), F(0)) / len(rl)
    return F(1, 2) * m1 + F(1, 2) * m2


def _find(design: list[DesignCell], **kw: Any) -> DesignCell | None:
    hits = [d for d in design if all(getattr(d, k) == v for k, v in kw.items())]
    if len(hits) > 1:
        raise FeasInputError([f"design: duplicate cell {kw}"])
    return hits[0] if hits else None


def _need(design: list[DesignCell], **kw: Any) -> DesignCell:
    d = _find(design, **kw)
    if d is None:
        raise FeasInputError([f"design: {kw}"])
    return d


def feas3(src: _Src, design: list[DesignCell], scale: str = "D384", seeds: tuple[int, ...] = SEEDS) -> dict[str, Any]:
    small, large = small_large(scale)
    per = {}
    for s in seeds:
        dl = _need(design, k=12, arm="plain", seed=s, branch=large, scale=scale, tef="9010")
        ds = _need(design, k=12, arm="plain", seed=s, branch=small, scale=scale, tef="9010")
        a, b = ladder(src, dl.run_id, T32K), ladder(src, ds.run_id, T32K)
        per[s] = {"S12_large": a, "S12_small": b, "diff": a - b, "reads": [dl.run_id, ds.run_id]}
    delta = sum((v["diff"] for v in per.values()), F(0)) / len(per)
    if delta <= THR["feas3_small"]:
        choice = "smallest"
    elif delta <= THR["feas3_mid"]:
        choice = "middle"
    else:
        choice = "largest"
    bstar = small if choice == "smallest" else large
    budget = {"D384": {"smallest": "30M", "middle": "60M", "largest": "90M"},
              "D512": {"smallest": "100M", "middle": "200M", "largest": "300M"}}[scale][choice]
    return {"rule": "FEAS-3 (U-04)", "Delta": delta, "per_seed": per, "choice": choice, "budget": budget,
            "b_star": bstar, "outcome": budget, "reads": sorted({r for v in per.values() for r in v["reads"]}),
            "seeds": list(seeds)}


def _u01_seed(src: _Src, design: list[DesignCell], s: int, b: str, T: int, scale: str, ratio: F, gap: F) -> dict:
    d12 = _need(design, k=12, arm="plain", seed=s, branch=b, scale=scale, tef="9010")
    d0 = _need(design, k=0, arm="plain", seed=s, branch=b, scale=scale, tef="9010")
    S12, S12c0, S0 = ladder(src, d12.run_id, T), ladder(src, d12.run_id, T, c0=True), ladder(src, d0.run_id, T)
    ok1, ok2 = S12 >= ratio * S12c0, S12 - S0 >= gap
    return {"S12": S12, "S12_C0": S12c0, "S0": S0, "ratio_ok": ok1, "gap_ok": ok2, "pass": bool(ok1 and ok2),
            "reads": [d12.run_id, d0.run_id]}


def u01(src: _Src, design: list[DesignCell], bstar: str) -> dict[str, Any]:
    per = {s: _u01_seed(src, design, s, bstar, T32K, "D384", THR["u01_ratio"], THR["u01_gap"]) for s in SEEDS}
    adopted = all(v["pass"] for v in per.values())
    return {"rule": "FEAS (U-01) D384", "branch": bstar, "per_seed": per, "adopted": adopted,
            "outcome": "D384" if adopted else "FEAS-512",
            "reads": sorted({r for v in per.values() for r in v["reads"]})}


def feas12(src: _Src, design: list[DesignCell]) -> dict[str, Any]:
    d512 = [d for d in design if d.scale == "D512" and d.arm == "plain" and d.tef == "9010"]
    if not d512:
        return {"rule": "FEAS-1/2 D512", "outcome": "pending: FEAS-512 runs not present", "evaluated": False,
                "reads": []}
    seeds = tuple(sorted({d.seed for d in d512 if d.k == 12}))
    f3 = feas3(src, design, scale="D512", seeds=seeds)   # reading R-d512-seeds
    b = f3["b_star"]
    per32 = {s: _u01_seed(src, design, s, b, T32K, "D512", THR["d512_ratio"], THR["d512_gap"]) for s in seeds}
    if all(v["pass"] for v in per32.values()):
        out, per16 = "D512 at 32K", None
    else:
        per16 = {s: _u01_seed(src, design, s, b, T16K, "D512", THR["d512_ratio"], THR["d512_gap"]) for s in seeds}
        out = "D512 with L_ext = 16K" if all(v["pass"] for v in per16.values()) else "STOP: re-scope with the user"
    reads = set(f3["reads"]) | {r for v in per32.values() for r in v["reads"]}
    if per16:
        reads |= {r for v in per16.values() for r in v["reads"]}
    return {"rule": "FEAS-1/2 D512", "feas3_d512": f3, "branch": b, "per_seed_32K": per32, "per_seed_16K": per16,
            "outcome": out, "evaluated": True, "reading": "R-d512-seeds", "reads": sorted(reads)}


def feas5(src: _Src, design: list[DesignCell], bstar: str) -> dict[str, Any]:
    ss = [d for d in design if d.arm == "ssmax" and d.k == 12 and d.seed == 101 and d.scale == "D384" and d.tef == "9010"]
    if not ss:
        raise FeasInputError(["design: SSMax arm (k=12, seed 101)"])
    branches = {d.branch for d in ss}
    b = bstar if bstar in branches else sorted(branches)[0]
    needs_pi = b != bstar
    dss = _need(design, k=12, arm="ssmax", seed=101, branch=b, scale="D384", tef="9010")
    dpl = _need(design, k=12, arm="plain", seed=101, branch=b, scale="D384", tef="9010")
    l32s, l32p = ladder(src, dss.run_id, T32K), ladder(src, dpl.run_id, T32K)
    l4s, l4p = ladder(src, dss.run_id, T4K), ladder(src, dpl.run_id, T4K)
    gain, loss = l32s - l32p, l4p - l4s
    adopt = gain >= THR["ssmax_gain"] and loss <= THR["ssmax_loss"]
    return {"rule": "FEAS-5 (U-03) SSMax", "branch": b, "b_star": bstar, "needs_pi_row": needs_pi,
            "reading": "R-ssmax-branch" if needs_pi else None,
            "ladder_32K": {"ssmax": l32s, "plain": l32p}, "ladder_4K": {"ssmax": l4s, "plain": l4p},
            "gain_32K": gain, "loss_4K": loss, "adopted": bool(adopt),
            "outcome": "SSMax for all arms" if adopt else "plain per-head gains", "reads": [dss.run_id, dpl.run_id]}


def elicitation(src: _Src, design: list[DesignCell], bstar: str, tef: str = "9010") -> dict[str, Any]:
    per, reads = {}, []
    for s in SEEDS:
        d = _need(design, k=12, arm="plain", seed=s, branch=bstar, scale="D384", tef=tef)
        em = src.cell_em(d.run_id, cell_name("R1", "D3", T4K))
        per[s] = {"EM_R1_D3_4K": em, "below": em < THR["elicit"]}
        reads.append(d.run_id)
    fires = all(v["below"] for v in per.values())          # reading: "both seeds", literal
    return {"rule": "elicitation", "branch": bstar, "tef": tef, "per_seed": per, "fires": fires,
            "outcome": "answer-only loss, re-run the FEAS TEF branches" if fires else "no change",
            "reading": "both seeds, literal", "reads": reads}


def ladder_rules(src: _Src, design: list[DesignCell]) -> dict[str, Any]:
    f3 = feas3(src, design)
    b = f3["b_star"]
    u = u01(src, design, b)
    out = {"source": src.name, "FEAS-3": f3, "U-01": u, "FEAS-5": feas5(src, design, b),
           "elicitation": elicitation(src, design, b)}
    out["FEAS-1/2"] = feas12(src, design) if not u["adopted"] else {
        "rule": "FEAS-1/2 D512", "outcome": "not applicable (D384 adopted)", "evaluated": False, "reads": []}
    if out["elicitation"]["fires"] and any(d.tef == "answer_only" for d in design):
        out["elicitation"]["rerun"] = elicitation(src, design, b, tef="answer_only")
        out["elicitation"]["outcome"] = ("promote the direct-supervision screen (I-08) and consult the user"
                                         if out["elicitation"]["rerun"]["fires"] else
                                         "answer-only loss adopted (re-run condition cleared)")
    return out


def contrast(src: _Src, design: list[DesignCell], bstar: str) -> dict[int, F]:
    out = {}
    for s in SEEDS:
        d12 = _need(design, k=12, arm="plain", seed=s, branch=bstar, scale="D384", tef="9010")
        d0 = _need(design, k=0, arm="plain", seed=s, branch=bstar, scale="D384", tef="9010")
        out[s] = ladder(src, d12.run_id, T32K) - ladder(src, d0.run_id, T32K)
    return out


def outcomes(lr: dict[str, Any]) -> dict[str, Any]:
    return {r: lr[r]["outcome"] for r in LADDER_RULES}


# ------------------------------------------------------------------------------------------ precision
def _pair_cell(a: dict[str, bool], b: dict[str, bool]) -> dict[str, Any]:
    shared = sorted(set(a) & set(b))
    n = len(shared)
    flips = sum(int(a[i] != b[i]) for i in shared)
    ea, eb = sum(int(a[i]) for i in shared), sum(int(b[i]) for i in shared)
    if n == 0:
        return {"n": 0, "flips": 0, "em_a": None, "em_b": None, "informative": False}
    fa, fb = F(ea, n), F(eb, n)
    info = max(fa, fb) > THR["info_lo"] and min(fa, fb) < THR["info_hi"]      # strict, symmetric
    return {"n": n, "flips": flips, "em_a": fa, "em_b": fb, "informative": info}


def _rate(num: int, den: int) -> F | None:
    return F(num, den) if den else None


def flip_rates(inp: Inputs, a: str, b: str, runs_by_k: dict[int, list[str]], cells: list[str]) -> dict[str, Any]:
    """flip(a, b): pooled per k over all cells of all of k's checkpoints, and restricted per k and length over
    the informative cells (symmetric definition) at 16K and 32K; per-checkpoint pooled; every cell's counts."""
    per_cell: dict[str, dict[str, Any]] = {}
    pooled, restricted, per_ckpt = {}, {}, {}
    for k, runs in runs_by_k.items():
        pn = pd = 0
        rn = {T16K: 0, T32K: 0}
        rd = {T16K: 0, T32K: 0}
        info_cells = {T16K: [], T32K: []}
        for rid in runs:
            cn = cd = 0
            for c in cells:
                pc = _pair_cell(inp.em(a, rid, c), inp.em(b, rid, c))
                per_cell[f"{rid}/{c}"] = pc
                pn += pc["flips"]
                pd += pc["n"]
                cn += pc["flips"]
                cd += pc["n"]
                T = cell_T(c)
                if T in rn and pc["informative"]:
                    rn[T] += pc["flips"]
                    rd[T] += pc["n"]
                    info_cells[T].append(f"{rid}/{c}")
            per_ckpt[rid] = {"flips": cn, "n": cd, "rate": _rate(cn, cd)}
        pooled[k] = {"flips": pn, "n": pd, "rate": _rate(pn, pd)}
        restricted[k] = {T: {"flips": rn[T], "n": rd[T], "rate": _rate(rn[T], rd[T]),
                             "informative_cells": info_cells[T]} for T in (T16K, T32K)}
    return {"pair": f"{a}-{b}", "pooled": pooled, "restricted": restricted, "per_checkpoint": per_ckpt,
            "per_cell": per_cell}


def _any_ge(fr: dict[str, Any], thr: F) -> list[str]:
    hits = []
    for k, v in fr["pooled"].items():
        if v["rate"] is not None and v["rate"] >= thr:
            hits.append(f"pooled k={k}")
        for T, w in fr["restricted"][k].items():
            if w["rate"] is not None and w["rate"] >= thr:
                hits.append(f"restricted k={k} T={T}")
    return hits


def precision(inp: Inputs) -> dict[str, Any]:
    ckpts = [d for d in inp.design if d.tef == "9010"]
    runs_by_k = {k: [d.run_id for d in ckpts if d.k == k] for k in K_LEVELS}
    cells = fd_cells()
    q1f = flip_rates(inp, "P0", "P1", runs_by_k, cells)
    q1_hits = _any_ge(q1f, THR["flip"])
    q1 = {"measure": q1f, "fires": bool(q1_hits), "triggers": q1_hits,
          "consequence": ("from S4 on evaluation uses fp32 weights; PI row before S4; RR-D052i reopened" if q1_hits
                          else "canonical bf16-rounded weights stay (D052(i) confirmed at this scale)")}
    q2f = flip_rates(inp, "P0", "P2", runs_by_k, cells)
    failed = sorted(r for r, v in q2f["per_checkpoint"].items() if v["rate"] is None or v["rate"] >= THR["flip"])
    q2 = {"measure": q2f, "pass": not failed, "failed_checkpoints": failed,
          "verdict": "VT-06 bf16 sub-item PASS" if not failed else "FAIL -> E10.3 (single-pass bf16 re-scores for the verdict; diagnose before S4)",
          "report_only_restricted": q2f["restricted"]}
    q3f = flip_rates(inp, "P0", "P3", runs_by_k, cells)
    q3_hits = _any_ge(q3f, THR["flip"])
    p13 = flip_rates(inp, "P1", "P3", runs_by_k, cells)
    opt_b = all(w["rate"] is None or w["rate"] < THR["flip"] for k in K_LEVELS for w in p13["restricted"][k].values())
    # Q4 (between-k): every ladder rule from P0 (Q2 replacements first) and from P4
    over = {r: "P2full" for r in failed}
    s0, s4, s1 = _Src(inp, "P0", over), _Src(inp, "P4"), _Src(inp, "P1")
    lr0, lr4, lr1 = ladder_rules(s0, inp.design), ladder_rules(s4, inp.design), ladder_rules(s1, inp.design)
    c0 = contrast(s0, inp.design, lr0["FEAS-3"]["b_star"])
    c4 = contrast(s4, inp.design, lr4["FEAS-3"]["b_star"])
    o0, o4, o1 = outcomes(lr0), outcomes(lr4), outcomes(lr1)
    diffs = [r for r in LADDER_RULES if o0[r] != o4[r]]
    dC = {s: abs(c4[s] - c0[s]) for s in SEEDS}
    dC_hits = [s for s in SEEDS if dC[s] > THR["dC"]]
    switch = bool(diffs or dC_hits)
    triggers = [f"outcome differs: {r} (P0 {o0[r]!r} vs P4 {o4[r]!r})" for r in diffs] + \
               [f"|C(P4,{s}) - C(P0,{s})| = {float(dC[s]):.6f} > 0.02" for s in dC_hits]
    q4 = {"P0_source": s0.name, "outcomes_P0": o0, "outcomes_P4": o4, "outcomes_P1": o1,
          "outcome_differences": diffs, "contrast_P0": c0, "contrast_P4": c4, "abs_dC": dC,
          "switch": switch, "triggers": triggers, "selected": "P4" if switch else "P0",
          "P1_vs_P0_differences": [r for r in LADDER_RULES if o0[r] != o1[r]],
          "ladder_P0": lr0, "ladder_P4": lr4, "ladder_P1": lr1, "reading": "R-contrast-branch"}
    opt_c = (not diffs) and all(dC[s] <= THR["dC"] for s in SEEDS)
    q3 = {"measure": q3f, "fires": bool(q3_hits), "triggers": q3_hits, "P1_P3": p13,
          "options_allowed": {"a_fp32_compute": True, "b_fp32_weights_only": opt_b, "c_production_with_caveat": opt_c}
          if q3_hits else None,
          "consequence": ("PI decision row before S4 choosing among the allowed options" if q3_hits else
                          "production scoring stays; from S4 P3 on seed-0 checkpoint of each Tier-A config (report-only)")}
    return {"Q1": q1, "Q2": q2, "Q3": q3, "Q4": q4}


# ------------------------------------------------------------------------------------------ other rules
def dec(x: float) -> F:
    """A logged LM-2 value (nats) as the exact rational of its shortest decimal representation (``repr``), so a
    logged 3.05 is 61/20 and threshold equality is exact (DTS10 arithmetic: no epsilon)."""
    return F(repr(float(x)))


def tef_forgetting(inp: Inputs, bstar: str, tef: str = "9010") -> dict[str, Any]:
    per: dict[int, dict[int, Any]] = {}
    reads = []
    rise = {}
    for k in K_LEVELS:
        per[k] = {}
        for s in SEEDS:
            d = _find(inp.design, k=k, arm="plain", seed=s, branch=bstar, scale="D384", tef=tef)
            if d is None:
                continue
            v = inp.lm2[d.run_id]
            r = dec(v["after"]) - dec(v["before"])
            per[k][s] = {"before": dec(v["before"]), "after": dec(v["after"]), "rise": r}
            reads.append(d.run_id)
        if not per[k]:
            raise FeasInputError([f"LM-2 runs for k={k} branch={bstar} tef={tef}"])
        rise[k] = sum((x["rise"] for x in per[k].values()), F(0)) / len(per[k])
    keep = rise[0] <= THR["forget_rise"] and rise[12] <= THR["forget_rise"] and abs(rise[0] - rise[12]) <= THR["forget_diff"]
    return {"rule": "TEF forgetting (U-08)", "branch": bstar, "tef": tef, "per_k_seed": per, "rise_0": rise[0],
            "rise_12": rise[12], "abs_diff": abs(rise[0] - rise[12]), "keep": bool(keep), "reads": reads,
            "seeds": {k: sorted(per[k]) for k in K_LEVELS}}


def tef_gate(inp: Inputs, bstar: str) -> dict[str, Any]:
    g = tef_forgetting(inp, bstar)
    if g["keep"]:
        g["outcome"] = "keep 90/10"
        return g
    has5050 = any(d.tef == "5050" for d in inp.design)
    if not has5050:
        g["outcome"] = "50/50 (evaluate the 50/50 TEF reruns by the same rule; reruns pending)"
        return g
    rer = [d for d in inp.design if d.tef == "5050"]
    g5 = tef_forgetting(inp, rer[0].branch, tef="5050")
    g["rerun_5050"] = g5
    g["reads"] = g["reads"] + g5["reads"]
    g["outcome"] = "50/50" if g5["keep"] else "TEF LR 1e-4 at 50/50 (final, no further test)"
    return g


def icc_from_counts(strata: dict[Any, list[int]], m: int = M_QUERIES) -> dict[str, Any]:
    """One-way ANOVA ICC pooled within strata (DTS10 ICC_U-15), exact: per stratum, context counts c_i in 0..m.
    SSB = sum_s [sum_i c_i^2/m - (sum_i c_i)^2/(m n_s)], df_b = sum_s (n_s - 1);
    SSW = sum_s [sum_i c_i - sum_i c_i^2/m], df_w = sum_s n_s (m - 1)."""
    ssb = ssw = F(0)
    dfb = dfw = 0
    for counts in strata.values():
        n = len(counts)
        if n == 0:
            continue
        s1 = sum(counts)
        s2 = sum(c * c for c in counts)
        ssb += F(s2, m) - F(s1 * s1, m * n)
        ssw += F(s1) - F(s2, m)
        dfb += n - 1
        dfw += n * (m - 1)
    msb = ssb / dfb if dfb else F(0)
    msw = ssw / dfw if dfw else F(0)
    den = msb + (m - 1) * msw
    icc = (msb - msw) / den if den else F(0)
    icc_t = max(icc, F(0))
    de = 1 + (m - 1) * icc_t
    return {"MSB": msb, "MSW": msw, "df_b": dfb, "df_w": dfw, "ICC_raw": icc, "ICC": icc_t, "DE": de,
            "contexts": 600 if de > THR["de"] else 300}


def _context_counts(rows: dict[str, bool], where: str) -> list[int]:
    by: dict[str, list[bool]] = {}
    for iid, em in rows.items():
        parts = iid.split(":")
        if len(parts) < 3:
            raise FeasInputError([f"{where}: item_id {iid!r} has no context/query part"])
        by.setdefault(parts[-2], []).append(em)
    bad = [c for c, v in by.items() if len(v) != M_QUERIES]
    if bad:
        raise FeasInputError([f"{where}: contexts {bad[:5]} do not have m = {M_QUERIES} queries"])
    return [sum(v) for _, v in sorted(by.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else kv[0])]


def u15(inp: Inputs, bstar: str) -> dict[str, Any]:
    strata: dict[str, list[int]] = {}
    reads = []
    cells = [c for c in fd_cells() if cell_family(c) == "retrieval" and cell_T(c) in (T16K, T32K)]
    for k in K_LEVELS:
        for s in SEEDS:
            d = _need(inp.design, k=k, arm="plain", seed=s, branch=bstar, scale="D384", tef="9010")
            reads.append(d.run_id)
            for c in cells:
                strata[f"{c}|k{k}|s{s}|{bstar}"] = _context_counts(inp.em("P0", d.run_id, c), f"P0 {d.run_id}/{c}")
    r = icc_from_counts(strata)
    return {"rule": "U-15 ICC / design effect", "branch": bstar, "source": "P0", "n_strata": len(strata),
            "n_contexts": sum(len(v) for v in strata.values()), **r,
            "outcome": f"{r['contexts']} contexts for confirmatory retrieval cells at 16K/32K", "reads": reads}


def i08(inp: Inputs, bstar: str, pass_id: str = "P0") -> dict[str, Any]:
    out: dict[str, Any] = {"rule": "I-08 promotion (D033)", "branch": bstar, "source": pass_id,
                           "reading": "mixed floor/ceiling triggers (stated); R-i08-c0 (C0 controls excluded)"}
    cells = [c for c in fd_cells() if cell_T(c) is not None]
    reads, any_trig, per_k = [], False, {}
    for k in K_LEVELS:
        ds = [_need(inp.design, k=k, arm="plain", seed=s, branch=bstar, scale="D384", tef="9010") for s in SEEDS]
        reads += [d.run_id for d in ds]
        src = _Src(inp, pass_id)
        fams: dict[str, dict[str, Any]] = {}
        for c in cells:
            em = sum((src.cell_em(d.run_id, c) for d in ds), F(0)) / len(ds)
            f = fams.setdefault(cell_family(c), {"levels": {}, "trigger": True})
            state = "floor" if em <= THR["floor"] else "ceiling" if em >= THR["ceiling"] else "informative"
            f["levels"][c] = {"seed_mean_EM": em, "state": state}
            if state == "informative":
                f["trigger"] = False
        per_k[k] = fams
        any_trig = any_trig or any(f["trigger"] for f in fams.values())
    out.update(per_k=per_k, triggered=any_trig, reads=reads,
               outcome="run the direct-supervision screen before S3" if any_trig else "no promotion")
    return out


# ------------------------------------------------------------------------------------------ flags
def flags(inp: Inputs, verdict_reads: dict[str, list[str]], q4_selected: str) -> dict[str, Any]:
    lineage = {d.run_id: (d.run_id, *d.lineage) for d in inp.design}
    feas_runs = {r for v in lineage.values() for r in v}
    near = sorted({r["run_id"] for r in inp.registry if r.get("status") == "note" and r.get("note") == "gdiv_near_miss"
                   and r.get("run_id") in feas_runs})
    nm_flagged = {}
    for v, reads in verdict_reads.items():
        hit = sorted({nr for rid in reads for nr in near if nr in lineage.get(rid, (rid,))})
        if hit:
            nm_flagged[v] = hit
    k_of = {}
    for d in inp.design:
        for r in lineage[d.run_id]:
            k_of[r] = d.k
    per_run: dict[str, int] = {}
    for r in inp.registry:
        if r.get("status") in INFRA_STATUSES and r.get("run_id") in k_of:
            per_run[r["run_id"]] = per_run.get(r["run_id"], 0) + 1
    n_k = {k: sum(c for rid, c in per_run.items() if k_of[rid] == k) for k in K_LEVELS}
    heavy = sorted(rid for rid, c in per_run.items() if c >= 3)
    infra = abs(n_k[0] - n_k[12]) >= 3 or bool(heavy)
    return {
        "F-nearmiss": {"near_miss_runs": near, "flagged_verdicts": nm_flagged},
        "F-infra": {"n_per_k": n_k, "per_run": per_run, "runs_with_3_or_more": heavy, "fires": infra,
                    "flagged_verdicts": list(VERDICTS) if infra else []},
        "F-precision": {"fires": q4_selected == "P4", "note": "reported only (the Q4 rule is automatic)"},
    }


# ------------------------------------------------------------------------------------------ main entry
def decide(inp: Inputs) -> dict[str, Any]:
    """Every FEAS rule in the pre-registered order; returns the full decision record (see module docstring)."""
    trace: list[str] = ["inputs complete", "precision rules Q1-Q4 computed first (DTS10-AT v2 precision_rules)"]
    prec = precision(inp)
    q4 = prec["Q4"]
    lr = q4["ladder_P4"] if q4["selected"] == "P4" else q4["ladder_P0"]
    trace.append(f"Q4 selected {q4['selected']}" + (f" (triggers: {q4['triggers']})" if q4["switch"] else ""))
    bstar = lr["FEAS-3"]["b_star"]
    trace.append(f"FEAS-3 -> {lr['FEAS-3']['budget']} ; b* = {bstar}")
    tg = tef_gate(inp, bstar)
    icc = u15(inp, bstar)
    i8 = i08(inp, bstar, "P0")
    if q4["switch"]:
        i8_p1 = i08(inp, bstar, "P1")
        i8["P1_check"] = {"triggered_P1": i8_p1["triggered"], "differs": i8_p1["triggered"] != i8["triggered"],
                          "per_k_P1": i8_p1["per_k"]}
        if i8["P1_check"]["differs"]:
            i8["needs_pi_row"] = True
            trace.append("I-08 differs between P0 and P1 -> PI row before S3")
    verdicts = {"FEAS-3": lr["FEAS-3"], "U-01": lr["U-01"], "FEAS-1/2": lr["FEAS-1/2"], "FEAS-5": lr["FEAS-5"],
                "TEF-forgetting": tg, "elicitation": lr["elicitation"], "U-15": icc, "I-08": i8}
    reads = {v: list(r.get("reads", [])) for v, r in verdicts.items()}
    fl = flags(inp, reads, q4["selected"])
    for v, r in verdicts.items():
        f = []
        if v in fl["F-nearmiss"]["flagged_verdicts"]:
            f.append("F-nearmiss")
        if fl["F-infra"]["fires"]:
            f.append("F-infra")
        if fl["F-precision"]["fires"] and v in LADDER_RULES:
            f.append("F-precision")
        r["flags"] = f
        r["hold_for_pi_row"] = bool({"F-nearmiss", "F-infra"} & set(f)) or bool(r.get("needs_pi_row"))
    for v in VERDICTS:
        trace.append(f"{v}: {verdicts[v].get('outcome')}" + (f" [flags {verdicts[v]['flags']}]" if verdicts[v]["flags"] else ""))
    header = {"score_source": q4["selected"], "F-precision": fl["F-precision"]["fires"],
              "flagged": sorted(v for v, r in verdicts.items() if r["flags"]),
              "held_for_pi_row": sorted(v for v, r in verdicts.items() if r["hold_for_pi_row"])}
    return {"id": "DTS10 feas_decisions", "spec": "experiments/prereg/DTS10_acceptance.v2.yaml",
            "header": header, "precision": prec, "verdicts": verdicts, "flags": fl, "trace": trace,
            "inputs": {"design": [asdict(d) for d in inp.design], "lm2": inp.lm2, "ckpt": inp.ckpt,
                       "n_files_read": len(inp.files_read), "thresholds": THR}}


def structure_inventory(results_root: str | Path) -> dict[str, Any]:
    """Structure-only pass (no rule, no score): every run/ckpt/suite/cell file under the tree is parsed and
    counted; returns counts and any parse problems."""
    root = Path(results_root)
    inv: dict[str, Any] = {"runs": {}, "problems": []}
    for rd in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(("_", "."))):
        for cd in sorted(p for p in rd.iterdir() if p.is_dir()):
            for sd in sorted(p for p in cd.iterdir() if p.is_dir()):
                n_cells = n_rows = 0
                for f in sorted(sd.glob("*.jsonl")):
                    rows = _read_rows(f, inv["problems"])
                    n_cells += 1
                    n_rows += len(rows or {})
                inv["runs"].setdefault(rd.name, {})[f"{cd.name}/{sd.name}"] = {"cells": n_cells, "rows": n_rows}
    return inv


def write_json(obj: Any, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(jf(obj), indent=1, sort_keys=False, default=str) + "\n")


__all__ = ["DesignCell", "FeasInputError", "Inputs", "THR", "decide", "design_from_registry", "fd_cells",
           "icc_from_counts", "ladder", "load_design_json", "load_inputs", "p4_cells", "precision",
           "read_registry", "structure_inventory", "write_json"]

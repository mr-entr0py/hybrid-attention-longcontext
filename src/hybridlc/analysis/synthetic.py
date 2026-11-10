"""Synthetic item-level results with planted effects (VT-18; protocol §16).

The generator writes exactly the layout the real evaluation produces (protocol §9.8)::

    <root>/results/<run_id>/<ckpt>/<suite>/<task>_<level>_<T|C0>.jsonl   (one row per query)
    <root>/registry.jsonl

so the whole analysis (reading, scoring, tests, verdicts, plots) runs end to end on it.

Planted model (probability scale). For run (config c with k = k_c, seed s) and stratum
(task t, level l, length T)::

    p[c, s, t, l, T] = clip( S0[t, l, T] + G[t, l, T] * R_fam(t)(k_c) + b_s + e_{c,s} + d_{c,s}, 0.01, 0.99 )

* ``S0`` and ``G`` are smooth functions of difficulty and length; the retrieval gap grows by
  ``h5_slope`` per doubling of T on the H5 fixed level set, so the planted H5(a) slope is known;
* ``R_fam(k)`` is the planted recovery curve per family;
* ``b_s ~ N(0, seed_sd^2)`` is shared by every config at seed s (common random numbers);
* ``e_{c,s} ~ N(0, resid_sd^2)`` is the run-level residual; ``d_{c,s}`` an optional bimodal drop.

Per context, ``p_ctx = BetaInv(U_ctx; p*phi, (1-p)*phi)`` with ``U_ctx`` shared by every model (the
frozen item files are identical across models), so ``E[p_ctx] = p`` exactly and the intra-context
correlation of the Q = 4 queries is ``1/(phi + 1)``. Queries are Bernoulli(p_ctx).

The *planted truth* of any estimand is obtained by applying the analysis's own weight vectors to the
true ``p`` (:meth:`World.true_cell_means`), so recovery is checked without a second implementation.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import special
from scipy import stats as sps

from .protocol import ALL_LENGTHS, CONTROL_T, FAMILIES
from .results import RunInfo, append_registry, write_item_rows

TASK_FAMILY = {t: f for f, ts in FAMILIES.items() for t in ts}

# CS cells (protocol §9.3): task -> T -> levels
CS_LEVELS: dict[str, dict[int, tuple[str, ...]]] = {
    "R1": {T: ("D3", "D15") for T in ALL_LENGTHS},
    "R-load": {1024: ("N8", "N32"), 2048: ("N8", "N32"), 4096: ("N8", "N32"),
               8192: ("N8", "N32", "N128", "N512"),
               16384: ("N8", "N32", "N128", "N512", "N1024"),
               32768: ("N8", "N32", "N128", "N512", "N1024", "N2048")},
    "M-hop": {T: (("h1", "h2") if T <= 4096 else ("h1", "h2", "h3", "h4")) for T in ALL_LENGTHS},
    "A-freq": {T: (("rho4", "rho2") if T <= 4096 else ("rho4", "rho2", "rho1.5", "rho1.25")) for T in ALL_LENGTHS},
}
# DG cells (protocol §9.5): retrieval at 16K/32K (+C0 controls), scored on C0/C12 (all core configs
# for the reference-inversion check); M-hop at 16K/32K (no controls), scored on C12 only.
DG_RET_LEVELS: dict[str, dict[int, tuple[str, ...]]] = {
    "R1": {T: ("D3", "D15") for T in (16384, 32768)},
    "R-load": {T: ("N8", "N32", "N128", "N512", "N1024") for T in (16384, 32768)},
}
DG_MHOP_LEVELS: dict[str, dict[int, tuple[str, ...]]] = {
    "M-hop": {T: ("h1", "h2", "h3", "h4") for T in (16384, 32768)},
}
DG_LEVELS: dict[str, dict[int, tuple[str, ...]]] = {**DG_RET_LEVELS, **DG_MHOP_LEVELS}
# Tasks that have C0 (no-filler) control cells in each frozen suite (CS §9.3; DG §9.5 retrieval only;
# DC and the rest have none). Suites not listed get controls for every primary task (legacy default).
SUITE_CONTROL_TASKS: dict[str, tuple[str, ...]] = {
    "CS": ("R1", "R-load", "M-hop", "A-freq"),
    "DG": ("R1", "R-load"),
    "DC": (),
}
# DC cells (protocol §9.5)
DC_LEVELS: dict[str, dict[int, tuple[str, ...]]] = {
    "R1": {T: ("D3",) for T in (4096, 16384, 32768)},
    "R-load": {T: ("N32", "N128") for T in (4096, 16384, 32768)},
    "M-hop": {T: ("h2",) for T in (4096, 16384, 32768)},
    "A-freq": {T: ("rho2",) for T in (4096, 16384, 32768)},
}


def _difficulty(task: str, level: str) -> float:
    """Level difficulty in [0, 1] (0 easiest), used to shape S0/G smoothly."""
    v = float(level.lstrip("DNhro"))
    if task == "R1":
        return 0.2 if v <= 3 else 0.5
    if task == "R-load":
        return min(1.0, np.log2(v / 8) / np.log2(2048 / 8))
    if task == "M-hop":
        return (v - 1) / 3
    if task == "A-freq":
        return float(np.clip((4 - v) / 2.75, 0, 1))
    return 0.5


@dataclass
class WorldSpec:
    """Planted effects. All accuracy quantities on the probability scale."""

    configs: dict[str, int] = field(default_factory=lambda: {"C0": 0, "C1": 1, "C3": 3, "C12": 12})
    seeds: Sequence[int] = (0, 1, 2, 3, 4)
    recovery: dict[str, Callable[[int], float]] = field(default_factory=lambda: {
        "retrieval": lambda k: {0: 0.0, 1: 0.45, 2: 0.7, 3: 0.95, 6: 1.0, 12: 1.0}.get(k, min(1.0, k / 3)),
        "multihop": lambda k: {0: 0.0, 1: 0.25, 2: 0.4, 3: 0.55, 6: 0.85, 12: 1.0}.get(k, k / 12),
        "aggregation": lambda k: {0: 0.0, 1: 0.5, 2: 0.7, 3: 0.8, 6: 0.95, 12: 1.0}.get(k, k / 12),
    })
    s0_base: float = 0.35
    gap_base: float = 0.45
    h5_slope: float = 0.03
    """Planted growth of the retrieval gap per doubling of T (relative to 1K)."""
    hop_slope: float = 0.0
    seed_sd: float = 0.02
    resid_sd: float = 0.015
    control_p: float = 0.85
    """C0-control accuracy of the k = 12 model (S12(C0))."""
    s12_drop_32k: float = 0.0
    """Extra drop of S12 at 32K for every family (to make G-REF (a) fail at 32K)."""
    bimodal: dict[str, tuple[float, float]] = field(default_factory=dict)
    """config -> (probability, drop): per-run bimodal failure (§11.5)."""
    phi: float = 6.0
    n_contexts: int = 60
    n_queries: int = 4
    entropy: int = 314159


class World:
    """A concrete draw of the planted world: true probabilities for every (run, suite, stratum)."""

    def __init__(self, spec: WorldSpec):
        self.spec = spec
        rng = np.random.default_rng(spec.entropy)
        self.seed_eff = {s: spec.seed_sd * rng.standard_normal() for s in spec.seeds}
        self.run_eff = {(c, s): spec.resid_sd * rng.standard_normal() for c in spec.configs for s in spec.seeds}
        self.drop = {}
        for c in spec.configs:
            pr, d = spec.bimodal.get(c, (0.0, 0.0))
            for s in spec.seeds:
                self.drop[(c, s)] = -d if rng.random() < pr else 0.0
        self._rng = rng

    # ------------------------------------------------------------------ planted probabilities
    def p_true(self, config: str, seed: int, task: str, level: str, T: int, *, branch_shift: float = 0.0) -> float:
        sp = self.spec
        k = sp.configs[config]
        fam = TASK_FAMILY.get(task, "retrieval")
        diff = _difficulty(task, level)
        if T == CONTROL_T:
            # No-filler control: near-ceiling for all models, mildly increasing in k.
            base = sp.control_p - 0.08 * (1 - k / 12) - 0.05 * diff
            p = base + self.seed_eff[seed] * 0.5
            return float(np.clip(p, 0.01, 0.99))
        dbl = np.log2(T / 1024)
        s0 = sp.s0_base - 0.15 * diff - 0.02 * dbl
        gap = sp.gap_base * (1 - 0.3 * diff)
        if fam == "retrieval":
            gap += sp.h5_slope * dbl
        elif fam == "multihop":
            gap += sp.hop_slope * dbl
        r = sp.recovery[fam](k)
        p = s0 + gap * r + self.seed_eff[seed] + self.run_eff[(config, seed)] + self.drop[(config, seed)]
        if T == 32768:
            p -= sp.s12_drop_32k * (k / 12)
        p += branch_shift
        return float(np.clip(p, 0.01, 0.99))

    # ------------------------------------------------------------------ item rows
    def rows(self, config: str, seed: int, suite: str, levels: Mapping[str, Mapping[int, Sequence[str]]], *,
             with_controls: bool = True, branch_shift: float = 0.0, n_contexts: int | None = None,
             tag: str = "") -> dict[tuple[str, str, int], pd.DataFrame]:
        """Per-stratum query rows for one run and suite. Context difficulty is shared across models;
        ``tag`` (branch/checkpoint) makes the query draws of different evaluations independent."""
        sp = self.spec
        nc = n_contexts or sp.n_contexts
        strata: list[tuple[str, str, int]] = []
        for task, byT in levels.items():
            for T, lvs in byT.items():
                strata += [(task, lv, T) for lv in lvs]
        if with_controls:
            ctl_tasks = SUITE_CONTROL_TASKS.get(suite, ("R1", "R-load", "M-hop", "A-freq"))
            ctrl = {(t, lv, CONTROL_T) for (t, lv, _T) in strata if t in ctl_tasks
                    and (t, lv) != ("R-load", "N2048")}
            strata += sorted(ctrl)
        out = {}
        for (task, lv, T) in strata:
            p = self.p_true(config, seed, task, lv, T, branch_shift=branch_shift)
            key = f"{suite}|{task}|{lv}|{T}"
            u_rng = np.random.default_rng(np.random.SeedSequence(sp.entropy, spawn_key=(_crc(key),)))
            U = u_rng.random(nc)  # shared context difficulty (identical item files)
            a, b = p * sp.phi, (1 - p) * sp.phi
            p_ctx = sps.beta.ppf(U, a, b)
            q_rng = np.random.default_rng(np.random.SeedSequence(
                sp.entropy, spawn_key=(_crc(key), _crc(config), seed, _crc(f"{branch_shift:.6f}|{tag}"))))
            em = (q_rng.random((nc, sp.n_queries)) < p_ctx[:, None]).astype(np.int8)
            logp = np.log(np.clip(p_ctx, 1e-4, 1))[:, None] * 4 + 0.1 * q_rng.standard_normal((nc, sp.n_queries))
            ctx_ids = np.repeat(np.array([f"{suite}-{task}-{lv}-{T}-{i}" for i in range(nc)]), sp.n_queries)
            out[(task, lv, T)] = pd.DataFrame({
                "context_id": ctx_ids, "query_idx": np.tile(np.arange(sp.n_queries), nc),
                "em_tf": em.ravel().astype(float), "logp_gold": logp.ravel()})
        return out


def _crc(s: str) -> int:
    import zlib
    return zlib.crc32(s.encode())


def run_id_for(config: str, seed: int, stage: str = "S4T", scale: str = "D384") -> str:
    import hashlib
    h = hashlib.sha256(f"{stage}|{scale}|{config}|{seed}".encode()).hexdigest()[:8]
    return f"{stage}-{scale}-{config}-s{seed}-{h}"


@dataclass
class RunPlan:
    config: str
    seed: int
    suites: dict[str, tuple[str, Mapping[str, Mapping[int, Sequence[str]]]]]
    """suite name -> (ckpt name, level map)."""
    branch: str = "main"
    stage: str = "S4T"
    branch_shift: float = 0.0


def default_plan(world: World, *, dose: bool = True, saturation: bool = True, dg_all_configs: bool = True,
                 dose_shift: float | Mapping[str, float] = 0.0, saturation_shift: float = 0.0,
                 dc_main: bool = False, dg_mhop_configs: Sequence[str] | None = ("C12",)) -> list[RunPlan]:
    """CS + DG at TEF 1000; DC for the dose ladder; seed-0 branches for the I-22 guards.

    DG follows §9.5: retrieval cells (+ C0 controls) on C0/C12, or on every config with
    ``dg_all_configs`` (the reference-inversion check); M-hop cells on ``dg_mhop_configs`` only
    (default C12, as pre-registered; ``None`` = every config scored on DG, a non-protocol plan kept
    for tests). Verification round 3 (ANA-DG-PROTOCOL-DESIGN-REFUSED): M-hop used to be scored on
    every DG config, with M-hop C0 controls that the frozen DG items do not have.

    ``dose_shift``: shift of the TEF-4000 checkpoint (a float for every anchor, or per config, which
    plants a still-closing gap, §7.3). ``dc_main``: also score DC at TEF 1000 on the seed-0 main
    anchors (the still-closing DC fallback)."""
    plans = []
    for c in world.spec.configs:
        for s in world.spec.seeds:
            suites = {"CS": ("tef1000", CS_LEVELS)}
            dg: dict[str, dict[int, tuple[str, ...]]] = {}
            if dg_all_configs or c in ("C0", "C12"):
                dg.update(DG_RET_LEVELS)
            if dg_mhop_configs is None:
                if dg:
                    dg.update(DG_MHOP_LEVELS)
            elif c in dg_mhop_configs:
                dg.update(DG_MHOP_LEVELS)
            if dg:
                suites["DG"] = ("tef1000", dg)
            plans.append(RunPlan(c, s, suites))
    anchors = [c for c in ("C0", "C1", "C3", "C12") if c in world.spec.configs]
    s0 = world.spec.seeds[0]
    if dose:
        for c in anchors:
            shift = float(dose_shift.get(c, 0.0)) if isinstance(dose_shift, Mapping) else float(dose_shift)
            plans.append(RunPlan(c, s0, {"DC": ("tef2000", DC_LEVELS)}, branch="tef4000", stage="S5bT"))
            plans.append(RunPlan(c, s0, {"DC": ("tef4000", DC_LEVELS), "CS": ("tef4000", CS_LEVELS)},
                                 branch="tef4000", stage="S5bT", branch_shift=shift))
            if dc_main:
                plans.append(RunPlan(c, s0, {"DC": ("tef1000", DC_LEVELS)}, stage="S4T"))
    if saturation:
        for c in [c for c in ("C0", "C3", "C12") if c in world.spec.configs]:
            plans.append(RunPlan(c, s0, {"CS": ("tef1000", CS_LEVELS), "DG": ("tef1000", DG_RET_LEVELS)},
                                 branch="stageB2x", stage="S5bBT", branch_shift=saturation_shift))
    return plans


def _tag(pl: RunPlan, ckpt: str) -> str:
    return "" if pl.branch == "main" else f"{pl.branch}|{ckpt}"


def generate_dataframe(world: World, plans: Iterable[RunPlan], scale: str = "D384") -> pd.DataFrame:
    """In-memory long DataFrame with the columns of :func:`~.results.load_item_results`."""
    frames = []
    for pl in plans:
        rid = run_id_for(pl.config, pl.seed, pl.stage, scale)
        for suite, (ckpt, levels) in pl.suites.items():
            for (task, lv, T), d in world.rows(pl.config, pl.seed, suite, levels, branch_shift=pl.branch_shift,
                                               tag=_tag(pl, ckpt)).items():
                d = d.copy()
                d["run_id"], d["config"], d["k"], d["seed"] = rid, pl.config, world.spec.configs[pl.config], pl.seed
                d["scale"], d["branch"], d["ckpt"], d["suite"] = scale, pl.branch, ckpt, suite
                d["task"], d["level"], d["T"] = task, lv, T
                frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df["T"] = df["T"].astype(np.int64)
    df["seed"] = df["seed"].astype(np.int64)
    df.attrs.update(skipped_reruns=[], inputs=[])
    return df


def write_tree(world: World, plans: Iterable[RunPlan], root: str | Path, scale: str = "D384") -> Path:
    """Write the JSONL result tree and a registry under ``root`` (``results/`` and ``registry.jsonl``)."""
    root = Path(root)
    res = root / "results"
    reg = root / "registry.jsonl"
    seen = set()
    for pl in plans:
        rid = run_id_for(pl.config, pl.seed, pl.stage, scale)
        if rid not in seen:
            append_registry(reg, RunInfo(rid, pl.config, pl.seed, scale, pl.stage, pl.branch,
                                         world.spec.configs[pl.config]), status="completed")
            seen.add(rid)
        for suite, (ckpt, levels) in pl.suites.items():
            for (task, lv, T), d in world.rows(pl.config, pl.seed, suite, levels, branch_shift=pl.branch_shift,
                                               tag=_tag(pl, ckpt)).items():
                tname = "C0" if T == CONTROL_T else str(T)
                path = res / rid / ckpt / suite / f"{task}_{lv}_{tname}.jsonl"
                rows = [dict(item_id=r.context_id, context_id=r.context_id, query_idx=int(r.query_idx),
                             em_tf=int(r.em_tf), logp_gold=float(r.logp_gold), T=int(T))
                        for r in d.itertuples(index=False)]
                write_item_rows(path, rows)
            meta = res / rid / ckpt / suite / "_meta.json"
            meta.write_text(json.dumps({"synthetic": True, "spec_entropy": world.spec.entropy}))
    return root


def true_scores(world: World, strata: Sequence[tuple[str, str, int]], weights: np.ndarray,
                configs: Sequence[str], seeds: Sequence[int], branch_shift: float = 0.0) -> pd.DataFrame:
    """Planted value ``weights @ p_true`` per (config, seed) for a weight vector over ``strata``
    (pass ``CellTable.strata`` and a vector from ``CellTable.ladder_weights``)."""
    out = np.full((len(configs), len(seeds)), np.nan)
    for i, c in enumerate(configs):
        for j, s in enumerate(seeds):
            out[i, j] = sum(w * world.p_true(c, s, *st, branch_shift=branch_shift)
                            for w, st in zip(weights, strata) if w != 0)
    return pd.DataFrame(out, index=list(configs), columns=list(seeds))


def logit(p: np.ndarray) -> np.ndarray:
    return special.logit(np.clip(p, 1e-4, 1 - 1e-4))

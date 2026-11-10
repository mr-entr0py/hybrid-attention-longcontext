"""Reading item-level evaluation results and the run registry (protocol §9.8, §18).

Layout consumed (protocol §9.8)::

    experiments/results/<run_id>/<ckpt>/<suite>/<task>_<level>_<T>.jsonl   one line per query
    experiments/registry.jsonl                                             append-only run registry

* ``<T>`` is the context length in tokens, or the literal ``C0`` for the no-filler control cells.
* A re-scored file is never overwritten (§9.8); it carries a suffix after ``<T>`` (for example
  ``R1_D3_16384.r1.jsonl``). The analysis uses the un-suffixed original and lists the re-runs it
  skipped, so a re-run can only enter the analysis through an explicit, logged decision.
* Per-row fields used: ``context_id`` (falls back to ``item_id``), ``query_idx``, ``em_tf`` (0/1;
  ``EM_tf`` accepted) and ``logp_gold`` (summed log-probability; a list is summed). Every other field
  is optional and can be carried along through ``extra_fields``.

The registry gives each ``run_id`` its configuration, seed index, scale and branch. When a run is
not in the registry, the fields are parsed from ``run_id = f"{stage}-{scale}-{config}-s{seed}-{hash8}"``
(INTERFACES.md, training).
"""

from __future__ import annotations

import json
import re
import warnings
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .protocol import CONFIG_K, CONTROL_T, LEVEL_PARAM

RUN_ID_RE = re.compile(
    r"^(?P<stage>[A-Za-z0-9]+)-(?P<scale>D\d+)-(?P<config>.+)-s(?P<seed>\d+)-(?P<hash>[0-9a-fA-F]{8})$"
)
RESULT_FILE_RE = re.compile(r"^(?P<task>.+?)_(?P<level>[^_]+)_(?P<T>\d+|C0)(?P<suffix>.*)\.jsonl$")

EM_KEYS: tuple[str, ...] = ("em_tf", "EM_tf", "em")
LOGP_KEYS: tuple[str, ...] = ("logp_gold", "logp_gold_sum")
CONTEXT_KEYS: tuple[str, ...] = ("context_id", "item_id")


@dataclass(frozen=True)
class RunInfo:
    """Identity of a training run as needed by the analysis."""

    run_id: str
    config: str
    seed: int
    scale: str | None = None
    stage: str | None = None
    branch: str = "main"
    """``main`` for the confirmatory run; e.g. ``tef4000`` or ``stageB2x`` for seed-0 branches."""
    k: int | None = None


def parse_run_id(run_id: str) -> RunInfo | None:
    """Parse ``{stage}-{scale}-{config}-s{seed}-{hash8}``; return ``None`` if it does not match."""
    m = RUN_ID_RE.match(run_id)
    if m is None:
        return None
    cfg = m["config"]
    return RunInfo(run_id=run_id, config=cfg, seed=int(m["seed"]), scale=m["scale"], stage=m["stage"],
                   k=config_k(cfg))


def config_k(config: str) -> int | None:
    """k for composition configs (``C<k>``), controls with a trailing k (``X-gc12`` -> 12), else None."""
    if config in CONFIG_K:
        return CONFIG_K[config]
    m = re.search(r"(\d+)$", config)
    return int(m.group(1)) if m else None


def load_registry(path: str | Path) -> dict[str, RunInfo]:
    """Read the append-only registry. Later rows for the same ``run_id`` update earlier ones.

    Recognised keys: ``run_id``, ``config``, ``seed_index`` (or ``seed``), ``scale``, ``stage``,
    ``branch``, ``k``. Rows without ``run_id`` (notes) are ignored.
    """
    path = Path(path)
    merged: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return {}
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rid = row.get("run_id")
            if not rid:
                continue
            merged.setdefault(rid, {}).update({k: v for k, v in row.items() if v is not None})
    out: dict[str, RunInfo] = {}
    for rid, row in merged.items():
        parsed = parse_run_id(rid)
        # ``config_name`` (training rows) wins; the runner's ``queued`` rows store the run-YAML *path*
        # under ``config``, which is never a configuration name.
        cfg_field = row.get("config")
        if isinstance(cfg_field, str) and ("/" in cfg_field or cfg_field.endswith((".yaml", ".yml"))):
            cfg_field = None
        config = row.get("config_name") or cfg_field or (parsed.config if parsed else None)
        seed = row.get("seed_index", row.get("seed", parsed.seed if parsed else None))
        if config is None or seed is None:
            warnings.warn(f"registry row {rid!r} has no config/seed; skipped", stacklevel=2)
            continue
        k = row.get("k", config_k(str(config)))
        out[rid] = RunInfo(run_id=rid, config=str(config), seed=int(seed),
                           scale=row.get("scale") or (parsed.scale if parsed else None),
                           stage=row.get("stage") or (parsed.stage if parsed else None),
                           branch=str(row.get("branch", "main")), k=None if k is None else int(k))
    return out


def normalise_level(task: str, level: str | int | float) -> str:
    """Canonical level label ``f"{param}{value}"`` (``D3``, ``N32``, ``h2``, ``rho1.5``).

    Accepts ``3``, ``"3"``, ``"D3"``, ``"D=3"``, ``"d3"``, ``"rho=1.50"``. Unknown tasks are returned
    unchanged (as a string).
    """
    param = LEVEL_PARAM.get(task)
    s = str(level).strip()
    if param is None or param == "":
        return s
    m = re.fullmatch(r"(?i)(?:" + re.escape(param) + r")?\s*=?\s*([0-9]*\.?[0-9]+)", s)
    if m is None:
        return s
    val = float(m.group(1))
    txt = str(int(val)) if val.is_integer() else f"{val:g}"
    return f"{param}{txt}"


def level_value(level: str) -> float:
    """Numeric value of a canonical level label (``N32`` -> 32.0, ``rho1.5`` -> 1.5)."""
    m = re.search(r"([0-9]*\.?[0-9]+)$", level)
    if m is None:
        raise ValueError(f"no numeric value in level {level!r}")
    return float(m.group(1))


def parse_result_filename(name: str) -> tuple[str, str, int, str] | None:
    """``R-load_N32_16384.jsonl`` -> ``("R-load", "N32", 16384, "")``. ``C0`` maps to ``CONTROL_T``."""
    m = RESULT_FILE_RE.match(name)
    if m is None:
        return None
    task = m["task"]
    T = CONTROL_T if m["T"] == "C0" else int(m["T"])
    return task, normalise_level(task, m["level"]), T, m["suffix"]


def _first(row: dict[str, Any], keys: Sequence[str]) -> Any:
    for k in keys:
        if k in row:
            return row[k]
    return None


def _as_float(v: Any) -> float:
    if v is None:
        return float("nan")
    if isinstance(v, (list, tuple)):
        return float(np.sum(v))
    return float(v)


def load_item_results(
    results_root: str | Path,
    registry: dict[str, RunInfo] | None = None,
    *,
    suites: Iterable[str] | None = None,
    ckpts: Iterable[str] | None = None,
    run_ids: Iterable[str] | None = None,
    extra_fields: Sequence[str] = (),
) -> pd.DataFrame:
    """Load every per-query row under ``results_root`` into one long DataFrame.

    Columns: ``run_id, config, k, seed, scale, branch, ckpt, suite, task, level, T, context_id,
    query_idx, em_tf, logp_gold`` plus ``extra_fields``. ``T == CONTROL_T`` marks C0 controls.
    The attribute ``df.attrs["skipped_reruns"]`` lists suffixed files that were not read,
    ``df.attrs["skipped_subsets"]`` lists ``--n-contexts`` subset outputs that were not read (header
    ``subset: true`` or a ``<label>-n<N>`` checkpoint directory; a subset is never confirmatory --
    verification round 2, EVAL-SUBSET-SHADOWS-FULL) and ``df.attrs["inputs"]`` lists every file read
    (for the plot input-hash JSON).
    """
    root = Path(results_root)
    registry = registry or {}
    suites_f = set(suites) if suites is not None else None
    ckpts_f = set(ckpts) if ckpts is not None else None
    runs_f = set(run_ids) if run_ids is not None else None
    cols: dict[str, list[Any]] = {c: [] for c in (
        "run_id", "config", "k", "seed", "scale", "branch", "ckpt", "suite", "task", "level", "T",
        "context_id", "query_idx", "em_tf", "logp_gold", *extra_fields)}
    skipped: list[str] = []
    skipped_sub: list[str] = []
    inputs: list[str] = []
    if not root.exists():
        df = pd.DataFrame(cols)
        df.attrs.update(skipped_reruns=skipped, skipped_subsets=skipped_sub, inputs=inputs)
        return df
    for run_dir in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(("_", "."))):
        # directories starting with "_" (e.g. ``_bench``) hold non-item results and are skipped
        rid = run_dir.name
        if runs_f is not None and rid not in runs_f:
            continue
        info = registry.get(rid) or parse_run_id(rid)
        if info is None:
            warnings.warn(f"run {rid!r} is neither in the registry nor a parseable run_id; skipped",
                          stacklevel=2)
            continue
        for ckpt_dir in sorted(p for p in run_dir.iterdir() if p.is_dir()):
            if ckpts_f is not None and ckpt_dir.name not in ckpts_f:
                continue
            for suite_dir in sorted(p for p in ckpt_dir.iterdir() if p.is_dir()):
                if suites_f is not None and suite_dir.name not in suites_f:
                    continue
                for f in sorted(suite_dir.glob("*.jsonl")):
                    parsed = parse_result_filename(f.name)
                    if parsed is None:
                        warnings.warn(f"unrecognised result file name {f}", stacklevel=2)
                        continue
                    task, level, T, suffix = parsed
                    if suffix:
                        skipped.append(str(f))
                        continue
                    if SUBSET_LABEL_RE.search(ckpt_dir.name) or is_subset_result(f):
                        skipped_sub.append(str(f))
                        continue
                    inputs.append(str(f))
                    with f.open() as fh:
                        for line in fh:
                            if not line.strip():
                                continue
                            row = json.loads(line)
                            cols["run_id"].append(rid)
                            cols["config"].append(info.config)
                            cols["k"].append(info.k if info.k is not None else -1)
                            cols["seed"].append(info.seed)
                            cols["scale"].append(info.scale)
                            cols["branch"].append(info.branch)
                            cols["ckpt"].append(ckpt_dir.name)
                            cols["suite"].append(suite_dir.name)
                            cols["task"].append(task)
                            cols["level"].append(level)
                            cols["T"].append(T)
                            cols["context_id"].append(str(_first(row, CONTEXT_KEYS)))
                            cols["query_idx"].append(int(row.get("query_idx", 0)))
                            cols["em_tf"].append(_as_float(_first(row, EM_KEYS)))
                            cols["logp_gold"].append(_as_float(_first(row, LOGP_KEYS)))
                            for e in extra_fields:
                                cols[e].append(row.get(e))
    df = pd.DataFrame(cols)
    df["T"] = df["T"].astype(np.int64)
    df["seed"] = df["seed"].astype(np.int64)
    df.attrs.update(skipped_reruns=skipped, skipped_subsets=skipped_sub, inputs=inputs)
    if skipped_sub:
        warnings.warn(f"{len(skipped_sub)} subset (--n-contexts) result files were not read, e.g. {skipped_sub[0]}",
                      stacklevel=2)
    return df


SUBSET_LABEL_RE = re.compile(r"-n\d+$")
"""Checkpoint-directory suffix of ``--n-contexts`` subset evaluations (``tef1000-n10``)."""


def is_subset_result(path: str | Path) -> bool:
    """True if the result file's ``_meta.json`` header says it scored a subset of its cell."""
    fp = Path(path)
    mp = fp.with_name(fp.stem + "_meta.json")
    if not mp.is_file():
        return False
    try:
        return bool(json.loads(mp.read_text()).get("subset", False))
    except (OSError, ValueError):
        return False


def frozen_cell_counts(items_root: str | Path, suites: Iterable[str] = ("CS", "DG", "DC")
                       ) -> dict[str, dict[tuple[str, str, int], int]]:
    """suite -> {(task, level, T): n_contexts} from the frozen item manifests (result-file strata)."""
    from hybridlc.integration.evaluate import SUITE_NAMESPACE, result_name

    root = Path(items_root)
    expected: dict[str, dict[tuple[str, str, int], int]] = {}
    for suite in suites:
        man = root / SUITE_NAMESPACE.get(suite, "test") / "MANIFEST.json"
        if not man.is_file():
            continue
        files = json.loads(man.read_text()).get("files", {})
        for rel, info in files.items():
            if not rel.startswith(f"{suite}/"):
                continue
            parsed = parse_result_filename(result_name(Path(rel).stem) + ".jsonl")
            if parsed is None:
                raise ValueError(f"frozen cell {rel} does not map to a result-file name")
            task, level, T, _ = parsed
            key = (task, level, T)
            if key in expected.setdefault(suite, {}):
                raise ValueError(f"frozen cells collide on result stratum {key} in {suite}")
            expected[suite][key] = int(info.get("n_items", info.get("cell", {}).get("n", 0)))
    return expected


def required_cells(suite: str, cells: dict[tuple[str, str, int], int], config: str | None,
                   branch: str | None = "main") -> dict[tuple[str, str, int], int]:
    """The frozen cells of ``suite`` a run of ``config`` on ``branch`` must have (§9.5 DG plan).

    DG M-hop cells are required only of C12 runs on the main branch (U-11); every other suite and
    every other DG cell is required of every run that has rows in the suite."""
    from .protocol import DG_MHOP_CONFIGS, DG_MHOP_TASKS

    if suite != "DG":
        return dict(cells)
    need_hop = config in DG_MHOP_CONFIGS and (branch or "main") == "main"
    return {c: n for c, n in cells.items() if need_hop or c[0] not in DG_MHOP_TASKS}


def suite_completeness_audit(df: pd.DataFrame, items_root: str | Path,
                             suites: Iterable[str] = ("CS", "DG", "DC")) -> dict[str, Any]:
    """Check every (run, ckpt, suite) against the frozen item manifest (protocol §9.3-9.5).

    For each audited suite a run/checkpoint evaluated, every frozen cell of that suite must have a
    result with exactly the manifest's number of contexts. Returns ``dict(problems, n_checked,
    complete)``; ``problems`` lists ``{run_id, ckpt, suite, cell, issue}``. Verification round 2,
    ANA-MISSING-CELL-SILENT-SEED-DROP ("check cells and per-cell context counts against the frozen
    suite definition"). Cells are matched by result name (``integration.evaluate.result_name``).

    DG follows the §9.5 scoring plan (verification round 3, ANA-DG-PROTOCOL-DESIGN-REFUSED): every DG
    run needs every retrieval cell (R1/R-load and their C0 controls), but the M-hop cells are required
    only of the C12 main-branch runs (``required_cells``); C0 runs and hybrids scored for the
    reference-inversion check carry retrieval cells only.
    """
    expected = frozen_cell_counts(items_root, suites)
    problems: list[dict[str, Any]] = []
    n_checked = 0
    if df.empty:
        return dict(problems=problems, n_checked=0, complete=True)
    sub_df = df[df["suite"].isin(list(expected))]
    run_cfg = sub_df.groupby("run_id")["config"].first().to_dict() if "config" in sub_df.columns else {}
    run_branch = sub_df.groupby("run_id")["branch"].first().to_dict() if "branch" in sub_df.columns else {}
    got = sub_df.groupby(["run_id", "ckpt", "suite", "task", "level", "T"])["context_id"].nunique()
    for (rid, ck, suite), sub in got.groupby(level=["run_id", "ckpt", "suite"]):
        have = {(t, lv, int(T)): int(v) for (_, _, _, t, lv, T), v in sub.items()}
        need = required_cells(suite, expected[suite], run_cfg.get(rid), run_branch.get(rid, "main"))
        for cell, n_exp in need.items():
            n_checked += 1
            if cell not in have:
                problems.append(dict(run_id=rid, ckpt=ck, suite=suite, cell=list(cell), issue="missing"))
            elif have[cell] != n_exp:
                problems.append(dict(run_id=rid, ckpt=ck, suite=suite, cell=list(cell),
                                     issue=f"{have[cell]} contexts, manifest has {n_exp}"))
    return dict(problems=problems, n_checked=n_checked, complete=not problems)


def rescore_audit(results_root: str | Path, files: Iterable[str], suites: Iterable[str] = ("CS", "DG", "DC"),
                  max_flip_rate: float = 0.01) -> dict[str, Any]:
    """§9.2 / I-29 audit of the single-pass re-score (5% of items, EM flip rate < 1%).

    Every audited result file (suite in ``suites``) needs a re-score record in
    ``<results_root>/_eval_log.jsonl`` (written by ``integration.evaluate``; the latest record of a
    file is used). Flips and re-scored queries are **pooled per checkpoint** (``run_id/ckpt``, over
    all audited suites), and each checkpoint's pooled flip rate must be < ``max_flip_rate``.

    Verification round 2 (RESCORE-AUDIT-PER-CELL-ZERO-FLIP): the threshold used to be applied to
    each cell separately, but a cell re-scores only round(0.05 n) contexts x 4 queries (20-60
    queries), so a single flip (1.7-5%) failed the audit. Per-cell flips are now descriptive only
    (``cells_with_flips``), together with a pooled rate over everything audited and a one-sided 95%
    Clopper-Pearson upper bound per checkpoint (descriptive; the pass rule is the point estimate,
    as §9.2 states it).

    Returns ``dict(n_cells, n_ok, missing, unusable, per_checkpoint, failing, cells_with_flips,
    pooled, subset_only, passes)``; ``passes`` is False if any audited cell lacks a usable record
    or any checkpoint's pooled flip rate is >= ``max_flip_rate``.
    """
    from scipy import stats as sps

    root = Path(results_root)
    suites = set(suites)
    recs: dict[str, dict] = {}
    log = root / "_eval_log.jsonl"
    if log.is_file():
        with log.open() as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("rescore") is None and r.get("rescore_file") and r.get("out"):
                    # blinded evaluation (DTS10-AT v2): the bf16 P2 record lives next to the result file
                    try:
                        sealed = json.loads(Path(r["rescore_file"]).read_text())
                    except (OSError, json.JSONDecodeError):
                        sealed = {}
                    if sealed.get("P2") is not None:
                        r["rescore"] = sealed["P2"]
                if r.get("rescore") is not None and r.get("out"):
                    # gates on the bf16 path (P2) only; the fp32 P3 record ("rescore_fp32") is FEAS-PREC v2's
                    recs[str(Path(r["out"]).resolve())] = r["rescore"]  # latest record wins
    ok, missing, unusable, subset_only, with_flips = [], [], [], [], []
    groups: dict[str, dict[str, int]] = {}
    n = 0
    for f in files:
        fp = Path(f)
        if fp.parent.name not in suites:
            continue
        n += 1
        r = recs.get(str(fp.resolve()))
        if r is None:
            missing.append(str(f))
            continue
        if r.get("n") is None or r.get("flips") is None:
            unusable.append(str(f))
            continue
        key = f"{fp.parent.parent.parent.name}/{fp.parent.parent.name}"
        g = groups.setdefault(key, dict(n=0, flips=0, cells=0))
        g["n"] += int(r["n"])
        g["flips"] += int(r["flips"])
        g["cells"] += 1
        ok.append(str(f))
        if int(r["flips"]) > 0:
            with_flips.append(dict(file=str(f), flips=int(r["flips"]), n=int(r["n"])))
        if r.get("subset"):
            subset_only.append(str(f))

    def _cp_upper(k: int, m: int) -> float:
        return 1.0 if k >= m else float(sps.beta.ppf(0.95, k + 1, m - k))

    per_ck, failing = {}, []
    for key, g in sorted(groups.items()):
        rate = g["flips"] / g["n"] if g["n"] else float("nan")
        rec = dict(g, flip_rate=rate, cp_upper95=_cp_upper(g["flips"], g["n"]) if g["n"] else float("nan"),
                   passes=bool(g["n"] > 0 and rate < max_flip_rate))
        per_ck[key] = rec
        if not rec["passes"]:
            failing.append(dict(checkpoint=key, **rec))
    tot_n = sum(g["n"] for g in groups.values())
    tot_f = sum(g["flips"] for g in groups.values())
    pooled = dict(n=tot_n, flips=tot_f, flip_rate=(tot_f / tot_n) if tot_n else None,
                  cp_upper95=_cp_upper(tot_f, tot_n) if tot_n else None)
    return dict(n_cells=n, n_ok=len(ok), missing=missing, unusable=unusable, per_checkpoint=per_ck,
                failing=failing, cells_with_flips=with_flips, pooled=pooled, subset_only=subset_only,
                max_flip_rate=max_flip_rate, rule="pooled per checkpoint (run_id/ckpt) over audited suites",
                passes=bool(not missing and not unusable and not failing))


def eval_weights_audit(files: Iterable[str], suites: Iterable[str] = ("CS", "DG", "DC")) -> dict[str, Any]:
    """All audited result files must have been scored from the same weight precision (header field
    ``eval_weights``; verification round 1, BF16-MILESTONE-ARTIFACT-MIXING). Files without the field
    (scored before the fix) count as ``unknown``. ``consistent`` is True iff exactly one value occurs."""
    suites = set(suites)
    seen: dict[str, list[str]] = {}
    for f in files:
        fp = Path(f)
        if fp.parent.name not in suites:
            continue
        mp = fp.with_name(fp.stem + "_meta.json")
        v = "unknown"
        if mp.is_file():
            try:
                v = str(json.loads(mp.read_text()).get("eval_weights") or "unknown")
            except (OSError, ValueError):
                v = "unreadable"
        seen.setdefault(v, []).append(str(f))
    return dict(values={k: len(v) for k, v in seen.items()}, consistent=len(seen) <= 1,
                examples={k: v[:3] for k, v in seen.items()})


def write_item_rows(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write per-query rows as JSONL (used by the synthetic generator and tests)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r, separators=(",", ":")) + "\n")


def append_registry(path: str | Path, info: RunInfo, **extra: Any) -> None:
    """Append one registry row (used by the synthetic generator; the real registry is owned by training)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {**asdict(info), "seed_index": info.seed, **extra}
    with path.open("a") as fh:
        fh.write(json.dumps(row) + "\n")

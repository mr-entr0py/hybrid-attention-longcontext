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
        config = row.get("config") or (parsed.config if parsed else None)
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
    The attribute ``df.attrs["skipped_reruns"]`` lists suffixed files that were not read and
    ``df.attrs["inputs"]`` lists every file read (for the plot input-hash JSON).
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
    inputs: list[str] = []
    if not root.exists():
        df = pd.DataFrame(cols)
        df.attrs.update(skipped_reruns=skipped, inputs=inputs)
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
    df.attrs.update(skipped_reruns=skipped, inputs=inputs)
    return df


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

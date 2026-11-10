"""Run configuration: YAML loading, composition, semantic hashing and run identity.

A *run YAML* (what ``scripts/train.py --config`` takes) composes three things::

    exp_stage: S4                               # experimental stage label (protocol §13.1)
    model: configs/model/D384/C3.yaml           # generated model config (scale + pattern + fields)
    training: configs/training/stage_A.yaml     # stage recipe (inherits configs/training/base.yaml)
    seed_index: 0
    lr: 2.0e-3                                  # peak LR eta* (ignored by stages with a fixed LR)
    steps_option: null                          # e.g. "60M" for a Stage-B budget (protocol §3.1 table)
    steps_multiplier: 1                         # 2 for the 2x Stage-B branch
    parent: null                                # parent run YAML (branch point) or {run_id, ckpt}
    continue_parent_step: false                 # TEF-4000: continue step counter of the parent
    config_name: null                           # override of the model YAML's name (e.g. C12ssmax)
    branch: main                                # analysis branch: main | tef4000 | stageB2x (non-semantic)
    model_overrides: {}                         # e.g. {ssmax: true}
    overrides: {}                               # dotted keys into the training recipe

Relative paths are resolved against the repo root first, then against the YAML's own directory.

``run_id = f"{exp_stage}{stage_tag}-{scale}-{config_name}-s{seed}-{config_hash[:8]}"`` where
``stage_tag`` is "" for Stage A, "B" for Stage B, "T" for the TEF and "" for the LR sweep (whose
``exp_stage`` is already S3). Example: ``S4-D384-C3-s0-1a2b3c4d`` (protocol §18).
"""
from __future__ import annotations

import copy
import dataclasses
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .compat import REPO_ROOT, config_hash as _config_hash

STAGES = ("A", "B", "TEF", "LR")
STAGE_TAG = {"A": "", "B": "B", "TEF": "T", "LR": ""}

# Model fields of the ModelConfig contract (INTERFACES.md, models/config.py).
MODEL_FIELDS = ("d_model", "d_ff", "n_layers", "vocab_size", "pattern", "attn_head_dim", "rope_theta",
                "nope", "ssmax", "init_std", "tie_embeddings", "norm_eps")


# ----------------------------------------------------------------------------------------------
# dataclasses
# ----------------------------------------------------------------------------------------------
@dataclass
class OptimConfig:
    """Protocol §7.1."""
    name: str = "adamw"
    fused: bool = True
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    weight_decay: float = 0.1
    # Parameter names containing any of these substrings get no weight decay even if ndim >= 2.
    # 1-D tensors and tensors flagged ``_no_weight_decay`` are always excluded.
    no_wd_name_patterns: list[str] = field(default_factory=lambda: ["ssmax"])
    grad_clip: float = 1.0
    z_loss_coef: float = 1e-4
    ce_chunk: int = 4096
    ignore_index: int = 1  # <|pad|>


@dataclass
class ScheduleConfig:
    """LR schedule (protocol §6.3, §7.2, §7.3). ``kind`` in {warmup_constant, linear_decay, wsd}."""
    kind: str
    steps: int
    peak_lr: float
    warmup_steps: int = 0
    stable_until: int | None = None  # wsd only: last step of the constant phase (exclusive)
    start_step: int = 0              # > 0 only for continuation runs (TEF-4000)


@dataclass
class DataConfig:
    """Arguments of ``StageStream`` / ``TEFStream`` (INTERFACES.md, data)."""
    stream: str          # "stage" | "tef"
    stage: str           # StageStream stage: A | B | LR (TEF uses stream "tef")
    seq_len: int
    micro_batch: int
    accum: int
    mix: dict[str, float] = field(default_factory=dict)
    data_seed: int = 0   # = seed_index (protocol §6.4); filled by the loader
    kwargs: dict[str, Any] = field(default_factory=dict)  # extra stream kwargs (TEF share, ...)

    @property
    def tokens_per_step(self) -> int:
        return self.seq_len * self.micro_batch * self.accum


@dataclass
class ValidationConfig:
    """Held-out LM-2 subset evaluated every ``every`` steps (protocol §18)."""
    every: int = 500
    stage: str = "LM2"
    seq_len: int = 4096
    micro_batch: int = 2
    n_batches: int = 32           # micro-batches per interval evaluation
    data_seed: int = 0            # constant across runs so every run sees the same subset
    mix: dict[str, float] = field(default_factory=dict)
    final_n_batches: int = 0      # > 0: a larger evaluation at the end (LR sweep: 2,000 x 4K = 1,000)
    at_start: bool = False
    kwargs: dict[str, Any] = field(default_factory=dict)  # extra stream kwargs (tests: synthetic root)


@dataclass
class CheckpointConfig:
    every: int = 250
    keep_last: int = 2
    milestone_steps: list[int] = field(default_factory=list)    # bf16 weights-only snapshots
    milestone_tokens: list[int] = field(default_factory=list)   # converted to steps (ceil)
    save_final: bool = True                                     # full final.pt (branch point S etc.)


@dataclass
class LoggingConfig:
    every: int = 10
    trace_every_step: bool = True   # per-step trace.jsonl: loss/grad-norm (hex) + batch hash
    attn_stats: bool = True         # attention max-logit / entropy if the model exposes the hook


@dataclass
class GDivConfig:
    """Protocol §13.3 G-div: NaN, or loss > EMA_0.99 + 1.0 for > 50 steps -> one automatic restart."""
    ema_decay: float = 0.99
    margin: float = 1.0
    patience: int = 50
    max_restarts: int = 1
    # FEAS only (D034): restart once *from scratch* at this LR instead of from the last checkpoint.
    fallback_lr: float | None = None


@dataclass
class OOMConfig:
    max_consecutive: int = 3   # G-OOM: 3 consecutive OOM failures (no checkpoint progress) -> failed


@dataclass
class RuntimeConfig:
    device: str = "cuda"
    mem_fraction: float = 0.44
    precision: str = "bf16"                 # bf16 autocast, fp32 master weights
    audit: bool = False                     # deterministic algorithms (VT-13, standing replicate)
    activation_checkpointing: bool = False  # per block (Stage B / TEF at 32K)
    env: dict[str, str] = field(default_factory=dict)
    #: G-throughput reference (tok/s measured at G0 for this stage/scale/config); None disables the
    #: alert. A ``throughput_alert`` registry annotation is written once the windowed tok/s has been
    #: below ``(1 - throughput_tolerance) * g0_tok_s`` for more than ``throughput_alert_after_s``.
    g0_tok_s: float | None = None
    #: torch intra-op CPU threads (None = torch default). Non-semantic; tests use 1 on a busy host.
    cpu_threads: int | None = None
    throughput_tolerance: float = 0.30
    throughput_alert_after_s: float = 3600.0


@dataclass
class ImplConfig:
    """Import strings ``module:attr`` for the pieces owned by other builders (overridable in tests)."""
    model_factory: str = "hybridlc.models.build:build_model"
    stage_stream: str = "hybridlc.data.loader:StageStream"
    tef_stream: str = "hybridlc.data.tef:TEFStream"
    loss_fn: str = "hybridlc.models.loss:chunked_ce"
    model_config: str = "hybridlc.models.config:ModelConfig"


@dataclass
class PathsConfig:
    runs_root: str = "runs"
    ckpt_root: str = "checkpoints"
    registry: str = "experiments/registry.jsonl"

    def resolve(self, p: str) -> Path:
        q = Path(p)
        return q if q.is_absolute() else REPO_ROOT / q


@dataclass
class RunConfig:
    exp_stage: str
    stage: str
    scale: str
    config_name: str
    model: dict[str, Any]          # ModelConfig fields (see ``model_config()``)
    optim: OptimConfig
    schedule: ScheduleConfig
    data: DataConfig
    seed_index: int
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    gdiv: GDivConfig = field(default_factory=GDivConfig)
    oom: OOMConfig = field(default_factory=OOMConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    impl: ImplConfig = field(default_factory=ImplConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    parent_run_id: str | None = None
    parent_ckpt: str | None = None
    model_meta: dict[str, Any] = field(default_factory=dict)  # k, tier, ... (informational)
    source: str | None = None
    tag: str | None = None         # optional non-semantic suffix (e.g. VT-13 replicate "p2")
    branch: str = "main"           # analysis branch label (main | tef4000 | stageB2x); non-semantic
    run_id: str = ""

    # -- identity ---------------------------------------------------------------------------
    def semantic_dict(self) -> dict[str, Any]:
        """Everything that changes what is trained. Excludes paths, logging/checkpoint cadence,
        memory cap and other bookkeeping (protocol §18: "SHA-256 of the semantic config")."""
        d = asdict(self)
        for k in ("run_id", "paths", "logging", "checkpoint", "oom", "validation", "source",
                  "parent_ckpt", "model_meta", "tag", "branch"):
            d.pop(k, None)
        for k in ("mem_fraction", "g0_tok_s", "throughput_tolerance", "throughput_alert_after_s", "cpu_threads"):
            d["runtime"].pop(k, None)
        d["gdiv"] = {"fallback_lr": d["gdiv"]["fallback_lr"]}  # detection thresholds are fixed
        return d

    def config_hash(self) -> str:
        return _config_hash(self.semantic_dict())

    def compute_run_id(self) -> str:
        st = STAGE_TAG[self.stage]
        rid = f"{self.exp_stage}{st}-{self.scale}-{self.config_name}-s{self.seed_index}-{self.config_hash()[:8]}"
        return f"{rid}-{self.tag}" if self.tag else rid

    def model_config_hash(self) -> str:
        return _config_hash(self.model)

    # -- paths ------------------------------------------------------------------------------
    @property
    def run_dir(self) -> Path:
        return self.paths.resolve(self.paths.runs_root) / self.run_id

    @property
    def ckpt_dir(self) -> Path:
        return self.paths.resolve(self.paths.ckpt_root) / self.run_id

    @property
    def registry_path(self) -> Path:
        return self.paths.resolve(self.paths.registry)

    # -- model ------------------------------------------------------------------------------
    def model_config(self) -> Any:
        """Instantiate the contract ``ModelConfig`` (falls back to a namespace if absent)."""
        cls = _try_import(self.impl.model_config)
        if cls is None:
            cls = FallbackModelConfig
        names = {f.name for f in dataclasses.fields(cls)} if dataclasses.is_dataclass(cls) else None
        kw = {k: v for k, v in self.model.items() if names is None or k in names}
        dropped = set(self.model) - set(kw)
        if dropped & set(MODEL_FIELDS):
            raise ValueError(f"{cls.__name__} lacks contract fields {sorted(dropped & set(MODEL_FIELDS))}")
        return cls(**kw)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FallbackModelConfig:
    """Same fields as the ``ModelConfig`` contract; used only until models/config.py exists."""
    d_model: int
    d_ff: int
    pattern: str
    n_layers: int = 12
    vocab_size: int = 16384
    attn_head_dim: int = 64
    rope_theta: float = 5e5
    nope: bool = False
    ssmax: bool = False
    init_std: float = 0.02
    tie_embeddings: bool = True
    norm_eps: float = 1e-5


# ----------------------------------------------------------------------------------------------
# YAML helpers
# ----------------------------------------------------------------------------------------------
def _try_import(spec: str) -> Any:
    import importlib

    mod, _, attr = spec.partition(":")
    try:
        m = importlib.import_module(mod)
    except ImportError:
        return None
    return getattr(m, attr, None)


def import_object(spec: str) -> Any:
    obj = _try_import(spec)
    if obj is None:
        raise ImportError(f"cannot import {spec!r} (module or attribute missing)")
    return obj


def resolve_path(p: str | Path, base: Path | None = None) -> Path:
    q = Path(p)
    if q.is_absolute():
        return q
    if (REPO_ROOT / q).exists() or base is None:
        return REPO_ROOT / q
    return (base / q).resolve()


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_with_inherits(path: str | Path) -> dict[str, Any]:
    """Load a YAML with an optional ``inherits: <path or list>`` key (merged base-first)."""
    path = Path(path)
    d = load_yaml(path)
    parents = d.pop("inherits", None)
    if parents is None:
        return d
    if isinstance(parents, str):
        parents = [parents]
    merged: dict[str, Any] = {}
    for p in parents:
        merged = deep_merge(merged, load_with_inherits(resolve_path(p, path.parent)))
    return deep_merge(merged, d)


def set_dotted(d: dict, key: str, value: Any) -> None:
    parts = key.split(".")
    cur = d
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
    cur[parts[-1]] = value


RECIPE_SECTIONS: frozenset[str] = frozenset({"stage", "schedule", "data", "checkpoint", "optim", "validation",
                                             "logging", "gdiv", "oom", "runtime", "impl"})
"""Top-level keys of a training recipe that are read. Overrides go into the recipe, so any other
top-level key would be silently ignored (verification round 2, TRAIN-DTS06-REMEDY-NOOP: ``-o lr=...``
changed nothing)."""


def _check_override_key(key: str) -> None:
    head = key.split(".", 1)[0]
    if head not in RECIPE_SECTIONS:
        raise ValueError(f"override key {key!r}: {head!r} is not a training-recipe section "
                         f"{sorted(RECIPE_SECTIONS)} and would be ignored; the peak LR is "
                         "`-o schedule.peak_lr=<value>` (or the run YAML's `lr`)")


def parse_override(s: str) -> tuple[str, Any]:
    """``key.sub=value`` with a YAML-parsed value (so 1e-3, true, [1,2] work)."""
    key, _, raw = s.partition("=")
    if not key or not _:
        raise ValueError(f"override must be key=value, got {s!r}")
    return key.strip(), yaml.safe_load(raw)


def _build(cls: type, d: dict[str, Any] | None) -> Any:
    d = dict(d or {})
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(d) - names
    if unknown:
        raise ValueError(f"unknown keys for {cls.__name__}: {sorted(unknown)}")
    if cls is OptimConfig and "betas" in d:
        d["betas"] = tuple(d["betas"])
    return cls(**d)


# ----------------------------------------------------------------------------------------------
# run loading
# ----------------------------------------------------------------------------------------------
def _select_steps(steps_spec: Any, scale: str, option: str | None) -> int:
    """``steps`` may be an int, ``{scale: int}`` or ``{scale: {option: int}}``.

    In the option form, a ``default`` key names the option used when the run YAML gives none.
    """
    s = steps_spec
    if isinstance(s, dict):
        if scale not in s:
            raise ValueError(f"schedule.steps has no entry for scale {scale}: {list(s)}")
        s = s[scale]
    if isinstance(s, dict):
        s = dict(s)
        default = s.pop("default", None)
        option = option if option is not None else default
        if option is None:
            raise ValueError(f"this stage needs steps_option, one of {list(s)}")
        if str(option) not in {str(k) for k in s}:
            raise ValueError(f"steps_option {option!r} not in {list(s)}")
        s = {str(k): v for k, v in s.items()}[str(option)]
    return int(s)


def load_run_config(path: str | Path, cli_overrides: list[str] | None = None,
                    paths: PathsConfig | None = None, tag: str | None = None, _depth: int = 0) -> RunConfig:
    """Load and fully resolve a run YAML into a :class:`RunConfig` (with its ``run_id``)."""
    if _depth > 8:
        raise RecursionError("parent chain too deep")
    path = resolve_path(path)
    run = load_yaml(path)
    allowed = {"exp_stage", "model", "training", "seed_index", "lr", "steps_option", "steps_multiplier",
               "parent", "continue_parent_step", "config_name", "model_overrides", "overrides", "paths",
               "notes", "tag", "branch"}
    unknown = set(run) - allowed
    if unknown:
        raise ValueError(f"{path}: unknown run keys {sorted(unknown)}")

    mdoc = load_yaml(resolve_path(run["model"], path.parent))
    tdoc = load_with_inherits(resolve_path(run["training"], path.parent))
    for key, val in (run.get("overrides") or {}).items():
        _check_override_key(key)
        set_dotted(tdoc, key, val)
    for ov in cli_overrides or []:
        k, v = parse_override(ov)
        _check_override_key(k)
        set_dotted(tdoc, k, v)

    scale = mdoc["scale"]
    model_fields = dict(mdoc["fields"])
    model_fields.update(run.get("model_overrides") or {})
    missing = set(MODEL_FIELDS) - set(model_fields)
    if missing:
        raise ValueError(f"model config missing fields {sorted(missing)}")

    stage = tdoc["stage"]
    if stage not in STAGES:
        raise ValueError(f"stage {stage!r} not in {STAGES}")
    seed_index = int(run["seed_index"])

    # schedule
    sd = dict(tdoc["schedule"])
    steps = _select_steps(sd.pop("steps"), scale, run.get("steps_option"))
    steps *= int(run.get("steps_multiplier", 1))
    peak = sd.pop("peak_lr", "run")
    if peak == "run":
        if run.get("lr") is None:
            raise ValueError(f"{path}: this stage takes its peak LR from the run YAML's `lr`")
        peak = float(run["lr"])
    sched = _build(ScheduleConfig, {**sd, "steps": steps, "peak_lr": float(peak)})

    data = _build(DataConfig, {**tdoc["data"], "data_seed": seed_index})
    ckpt = _build(CheckpointConfig, tdoc.get("checkpoint"))
    tps = data.tokens_per_step
    ckpt.milestone_steps = sorted({*ckpt.milestone_steps,
                                   *(math.ceil(t / tps) for t in ckpt.milestone_tokens)})
    ckpt.milestone_steps = [s for s in ckpt.milestone_steps if 0 < s <= steps]

    cfg = RunConfig(
        exp_stage=str(run["exp_stage"]), stage=stage, scale=scale,
        config_name=str(run.get("config_name") or mdoc["name"]),
        model=model_fields, optim=_build(OptimConfig, tdoc.get("optim")), schedule=sched, data=data,
        seed_index=seed_index, validation=_build(ValidationConfig, tdoc.get("validation")),
        checkpoint=ckpt, logging=_build(LoggingConfig, tdoc.get("logging")),
        gdiv=_build(GDivConfig, tdoc.get("gdiv")), oom=_build(OOMConfig, tdoc.get("oom")),
        runtime=_build(RuntimeConfig, tdoc.get("runtime")), impl=_build(ImplConfig, tdoc.get("impl")),
        paths=paths or _build(PathsConfig, run.get("paths")),
        model_meta={k: mdoc[k] for k in mdoc if k not in ("fields",)}, source=str(path),
        tag=tag if tag is not None else run.get("tag"),
        branch=str(run.get("branch") or "main"),
    )

    # parent (branch point)
    parent = run.get("parent")
    parent_branch: str | None = None
    if parent:
        if isinstance(parent, str):
            pcfg = load_run_config(resolve_path(parent, path.parent), paths=cfg.paths, tag=cfg.tag,
                                   _depth=_depth + 1)
            parent_branch = pcfg.branch
            if pcfg.model != cfg.model:
                raise ValueError(f"parent {pcfg.run_id} has a different model config")
            cfg.parent_run_id = pcfg.run_id
            cfg.parent_ckpt = str(pcfg.ckpt_dir / "final.pt")
            if run.get("continue_parent_step"):
                cfg.schedule.start_step = pcfg.schedule.steps
        else:
            cfg.parent_run_id = parent["run_id"]
            cfg.parent_ckpt = parent.get("ckpt") or str(
                cfg.paths.resolve(cfg.paths.ckpt_root) / parent["run_id"] / "final.pt")
            if run.get("continue_parent_step"):
                cfg.schedule.start_step = int(parent["step"])
    if cfg.schedule.start_step >= cfg.schedule.steps:
        raise ValueError("start_step >= steps")
    cfg.branch = _resolve_branch(path, run, parent_branch)
    cfg.run_id = cfg.compute_run_id()
    return cfg


ANALYSIS_BRANCHES = ("main", "tef4000", "stageB2x")
"""Analysis branch labels the §11 pipeline reads (``analysis.pipeline.AnalysisSpec``)."""


def _resolve_branch(path: Path, run: dict[str, Any], parent_branch: str | None) -> str:
    """The run's analysis branch label (non-semantic: not in the config hash).

    Verification round 3 (S5B-BRANCH-LABEL-NOT-SET): the label used to default to ``main`` for every
    run, so a TEF-4000 continuation or a 2x Stage-B lineage launched without ``branch:`` was read by
    the analysis as a main-branch run (the dose guard and the §7.3 still-closing flag silently became
    not evaluable, or the analysis refused on duplicate (config, seed)). Now:

    * an explicit ``branch:`` must be one of :data:`ANALYSIS_BRANCHES`;
    * without one, a run inherits its YAML parent's non-main branch (a TEF after the 2x Stage-B
      branch is on ``stageB2x``), otherwise ``main``;
    * ``continue_parent_step: true`` (the TEF-4000 continuation) or ``steps_multiplier > 1`` (the
      2x Stage-B branch) with a resolved ``main`` branch is refused.
    """
    if run.get("branch") is not None:
        branch = str(run["branch"])
        if branch not in ANALYSIS_BRANCHES:
            raise ValueError(f"{path}: branch {branch!r} not in {ANALYSIS_BRANCHES}")
    else:
        branch = parent_branch if parent_branch and parent_branch != "main" else "main"
    if branch == "main":
        if run.get("continue_parent_step"):
            raise ValueError(f"{path}: continue_parent_step (the TEF-4000 continuation) needs an explicit "
                             "non-main analysis branch, e.g. `branch: tef4000`")
        if int(run.get("steps_multiplier", 1)) > 1:
            raise ValueError(f"{path}: steps_multiplier > 1 (the 2x Stage-B branch) needs an explicit "
                             "non-main analysis branch, e.g. `branch: stageB2x`")
    return branch


def run_config_from_dict(d: dict[str, Any]) -> RunConfig:
    """Inverse of ``RunConfig.to_dict`` (used to reload the resolved config stored in checkpoints)."""
    sub = {"optim": OptimConfig, "schedule": ScheduleConfig, "data": DataConfig, "validation": ValidationConfig,
           "checkpoint": CheckpointConfig, "logging": LoggingConfig, "gdiv": GDivConfig, "oom": OOMConfig,
           "runtime": RuntimeConfig, "impl": ImplConfig, "paths": PathsConfig}
    kw = dict(d)
    for k, cls in sub.items():
        if k in kw:
            kw[k] = _build(cls, kw[k])
    return RunConfig(**kw)


def dump_yaml(obj: Any, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(obj, f, sort_keys=False, default_flow_style=None, width=120)

"""S1(a) run list, taken verbatim from Zoology's original MQAR config file (protocol §17).

We *execute* ``zoology/experiments/mqar_example_configs/original_mqar_configs.py`` from the
pinned checkout, so the data configs (V = 8192, the 5 training and 7 test segments, batch
(256, 32)), the model configs (2-layer TransformerBlock, BaseConv(k=3) + sequence mixer,
Identity state mixer, no position embeddings) and the LR grid ``np.logspace(-3, -1.5, 4)``
are Zoology's own objects. We then only:

* keep the families/widths named in protocol §17 (attention d ∈ {64, 128}, GDN d ∈ {128, 256});
* set ``data.cache_dir = data/zoology/`` (protocol §17);
* replace the wandb ``LoggerConfig`` with an empty one (protocol §17: wandb disabled);
* drop ``predictions_path``/``collect_predictions`` (they are not fields of ``TrainConfig``
  at this commit, so pydantic already ignores them).

Every other TrainConfig field keeps Zoology's default, including ``max_epochs=32`` (set by
the file), ``weight_decay=0.1``, ``seed=123``, data ``seed=123`` and early stopping at
``valid/accuracy > 0.99`` (the TrainConfig default the file does not override).
"""
from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Any

from zoology_compat import DATA_CACHE_DIR, ZOOLOGY_DIR, setup_zoology

ORIGINAL_CONFIG_FILE = ZOOLOGY_DIR / "zoology" / "experiments" / "mqar_example_configs" / "original_mqar_configs.py"

from s1_analysis import PHASE1, PHASE2, WIDTHS as S1A_WIDTHS, lr_tag, run_id as make_run_id  # noqa: E402


@dataclass
class S1Run:
    run_id: str
    family: str
    d_model: int
    lr: float
    config: Any  # zoology.config.TrainConfig


def _load_original_module():
    setup_zoology()
    spec = importlib.util.spec_from_file_location("zoology_original_mqar_configs", ORIGINAL_CONFIG_FILE)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def build_runs() -> list[S1Run]:
    """Return the 16 S1(a) runs in execution order (PHASE1 then PHASE2, LRs ascending)."""
    from zoology.config import LoggerConfig

    mod = _load_original_module()
    runs: dict[tuple[str, int, str], S1Run] = {}
    for cfg in mod.configs:
        fam, d = cfg.model.name, cfg.model.d_model
        if fam not in S1A_WIDTHS or d not in S1A_WIDTHS[fam]:
            continue
        cfg = cfg.model_copy(deep=True)
        cfg.data.cache_dir = str(DATA_CACHE_DIR)
        cfg.logger = LoggerConfig()  # project/entity None -> no wandb
        rid = make_run_id(fam, d, cfg.learning_rate)
        cfg.run_id = rid
        runs[(fam, d, lr_tag(cfg.learning_rate))] = S1Run(rid, fam, d, float(cfg.learning_rate), cfg)
    ordered: list[S1Run] = []
    for fam, d in PHASE1 + PHASE2:
        ordered += sorted((r for (f, dd, _), r in runs.items() if f == fam and dd == d), key=lambda r: r.lr)
    assert len(ordered) == 16, f"expected 16 S1(a) runs, got {len(ordered)}"
    return ordered


def describe_config(cfg: Any) -> dict[str, Any]:
    """JSON-safe dump of the TrainConfig actually used (stored with each result)."""
    return cfg.model_dump(mode="json")


if __name__ == "__main__":
    for r in build_runs():
        c = r.config
        print(r.run_id, c.max_epochs, c.data.batch_size, c.early_stopping_metric, c.early_stopping_threshold,
              c.weight_decay, c.seed, c.data.seed, c.model.sequence_mixer.kwargs["configs"][1]["kwargs"])

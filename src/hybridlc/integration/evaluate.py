"""Checkpoint -> frozen item files -> item-level results (protocol §9.2, §9.8).

This module connects three builders' contracts and adds nothing to any of them:

* training writes ``checkpoints/<run_id>/{final.pt, step_*.pt, milestones/*_bf16.pt}`` whose payload
  carries ``run_id`` (and, for full checkpoints, the resolved run ``config``);
* eval writes item files ``experiments/items/{test,dev}/<suite>/<task>[-<variant>]_<level>_<T>.npz``
  (``T`` = ``1K``..``32K``, ``nofill`` for C0 controls) and scores them with
  :func:`hybridlc.eval.scorer.score_items` / :func:`~hybridlc.eval.scorer.score_lm_items`;
* analysis reads ``experiments/results/<run_id>/<ckpt>/<suite>/<task>_<level>_<T>.jsonl`` with
  ``T`` in tokens (``16384``) and the literal ``C0`` for the no-filler controls
  (:mod:`hybridlc.analysis.results`).

The one real mismatch between eval and analysis is the file *name*: :func:`result_name` is the
single place that maps an item-cell name to a result-file name.

``ckpt`` labels (directory under the run): ``tef<step>`` for TEF checkpoints (the analysis keys on
it, e.g. ``tef1000``), ``B<step>`` for Stage-B and ``A<step>`` for Stage-A / LR-sweep checkpoints.

Result files are never overwritten (the scorer's ``.rN`` suffix rule); every scored cell is also
logged as one line in ``<results_root>/_eval_log.jsonl`` (the "registry note" of §9.8, kept out
of the training registry so its state machine is unaffected).
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch

from hybridlc.utils.provenance import REPO_ROOT, sha256_file

#: Suite -> item namespace (protocol §9.3-9.6: CS/MS/Tier0 use the test generator, the dev suites
#: DG/DC/FD use generator seed 77).
SUITE_NAMESPACE: dict[str, str] = {"CS": "test", "MS": "test", "Tier0": "test",
                                   "DG": "dev", "DC": "dev", "FD": "dev"}
ITEMS_ROOT = REPO_ROOT / "experiments" / "items"
RESULTS_ROOT = REPO_ROOT / "experiments" / "results"

_T_RE = re.compile(r"^(?P<n>\d+)(?P<k>K?)$")


# ------------------------------------------------------------------------------------- naming
def result_name(item_stem: str) -> str:
    """Item-cell stem -> result-file stem (no extension).

    ``R1_D3_16K`` -> ``R1_D3_16384``; ``R1-C0_D3_nofill`` -> ``R1_D3_C0`` (analysis: C0 controls are
    ``<task>_<level>_C0``); other variants keep their suffix in the task name
    (``M-hop-forward_h2_16K`` -> ``M-hop-forward_h2_16384``); Tier-0 lengths already in tokens are
    kept (``LM-4_a-G512_1025``).
    """
    task, level, t = item_stem.rsplit("_", 2) if item_stem.count("_") >= 2 else (None, None, None)
    if task is None:
        raise ValueError(f"not an item-cell name: {item_stem!r}")
    if t == "nofill":
        if not task.endswith("-C0"):
            raise ValueError(f"nofill cell without -C0 variant: {item_stem!r}")
        return f"{task[:-3]}_{level}_C0"
    m = _T_RE.match(t)
    if m is None:
        raise ValueError(f"unrecognised length {t!r} in {item_stem!r}")
    T = int(m["n"]) * (1024 if m["k"] else 1)
    return f"{task}_{level}_{T}"


def ckpt_label(stage: str, step: int) -> str:
    """Checkpoint directory label: ``tef1000`` (TEF), ``B229`` (Stage B), ``A4578`` (Stage A / LR)."""
    return {"TEF": f"tef{step}", "B": f"B{step}"}.get(stage, f"A{step}")


# ------------------------------------------------------------------------------------- loading
@dataclass
class LoadedCheckpoint:
    model: torch.nn.Module
    run_id: str
    stage: str
    step: int
    config: dict[str, Any]
    path: Path
    sha256: str
    dtype: str
    config_hash: str | None = None
    git_commit: str | None = None


def _resolved_config(payload: dict[str, Any], path: Path, runs_root: Path | None) -> dict[str, Any]:
    if isinstance(payload.get("config"), dict):
        return payload["config"]
    import yaml

    rid = payload.get("run_id")
    cands = []
    if runs_root is not None and rid:
        cands.append(Path(runs_root) / rid / "config.resolved.yaml")
    # milestones live in checkpoints/<run_id>/milestones/; full checkpoints in checkpoints/<run_id>/
    for up in (path.parent, path.parent.parent):
        if (up / "final.pt").exists() and up / "final.pt" != path:
            return _resolved_config(torch.load(up / "final.pt", map_location="cpu", weights_only=False), up / "final.pt", runs_root)
    if rid:
        cands.append(REPO_ROOT / "runs" / rid / "config.resolved.yaml")
    for c in cands:
        if c.exists():
            return yaml.safe_load(c.read_text())
    raise FileNotFoundError(f"cannot find the resolved run config for {path} (run_id={rid}); pass runs_root")


def load_checkpoint(path: str | Path, device: str | torch.device = "cuda",
                    runs_root: str | Path | None = None) -> LoadedCheckpoint:
    """Rebuild the ``HybridLM`` of a training checkpoint (full fp32 or bf16 milestone) in fp32.

    The model is built with ``build_model(ModelConfig(**config['model']), seed_index)`` and the saved
    tensors are loaded with ``strict=True`` (a bf16 milestone is cast up to the fp32 master dtype).
    """
    from hybridlc.models.build import build_model
    from hybridlc.models.config import ModelConfig

    path = Path(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    cfg = _resolved_config(payload, path, Path(runs_root) if runs_root else None)
    model = build_model(ModelConfig.from_dict(dict(cfg["model"])), int(cfg["seed_index"]))
    sd = {k: (v.float() if v.is_floating_point() else v) for k, v in payload["model"].items()}
    model.load_state_dict(sd, strict=True)
    model.to(device).eval()
    return LoadedCheckpoint(model=model, run_id=str(payload.get("run_id") or cfg.get("run_id")),
                            stage=str(cfg["stage"]), step=int(payload["step"]), config=cfg, path=path,
                            sha256=sha256_file(path), dtype=str(payload.get("dtype", "float32")),
                            config_hash=payload.get("config_hash"), git_commit=payload.get("git_commit"))


# ------------------------------------------------------------------------------------- scoring
def suite_item_files(suite: str, items_root: str | Path = ITEMS_ROOT, only: str | None = None) -> list[Path]:
    root = Path(items_root) / SUITE_NAMESPACE[suite] / suite
    files = sorted(root.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"no item files under {root}")
    return [f for f in files if only is None or only in f.name]


def score_cell(model: torch.nn.Module, item_path: Path, out_jsonl: Path, *, n_contexts: int | None = None,
               batch_size: int | None = None, score_candidates: bool = True,
               extra_meta: dict[str, Any] | None = None) -> Path:
    """Score one item file, optionally only its first ``n_contexts`` contexts (smoke / dev subsets).

    Full cells go through :func:`hybridlc.eval.scorer.score_file` unchanged. A subset uses the same
    scorer functions on ``items[:n_contexts]`` and records ``n_contexts_scored`` / ``subset`` in the
    header (a subset is never a confirmatory result).
    """
    from hybridlc.eval import scorer
    from hybridlc.eval.constants import frozen_batch_size
    from hybridlc.eval.items import load_items

    if n_contexts is None:
        return scorer.score_file(model, item_path, out_jsonl, batch_size=batch_size,
                                 score_candidates=score_candidates, extra_meta=extra_meta)
    t0 = time.time()
    items, cell = load_items(item_path)
    items = items[:n_contexts]
    name = item_path.stem
    T = cell.get("T") or max(len(it["context_ids"]) for it in items)
    frozen = frozen_batch_size(int(T))
    bs = int(batch_size or frozen)
    model.eval()
    dev = next(model.parameters()).device
    use_ac = dev.type == "cuda"
    if cell.get("task") in scorer.LM_TASKS:
        rows = scorer.score_lm_items(model, items, name, autocast=use_ac)
    else:
        rows = scorer.score_items(model, items, bs, name, score_candidates, autocast=use_ac)
    out = scorer.unique_path(out_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "x") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True, allow_nan=True) + "\n")
    hdr = scorer._header(model, item_path, dict(  # noqa: SLF001 (same header as score_file)
        cell=cell, n_rows=len(rows), batch_size=bs, frozen_batch_size=frozen, batch_size_deviation=bs != frozen,
        scoring_path=scorer._scoring_path(model), autocast_bf16=use_ac, device=str(dev),  # noqa: SLF001
        wall_time_s=time.time() - t0, out_jsonl=str(out), subset=True, n_contexts_scored=len(items),
        **(extra_meta or {})))
    scorer._write_meta(out, hdr)  # noqa: SLF001
    return out


def rescore_subset_check(model: torch.nn.Module, item_path: Path, rows_jsonl: Path, n_contexts: int) -> dict:
    """§9.2 single-pass re-score restricted to the contexts a subset run actually scored: the frozen
    5 % subset of the cell intersected with ``range(n_contexts)`` (or context 0 if that is empty)."""
    from hybridlc.eval import scorer
    from hybridlc.eval.adapter import ForwardOnlyAdapter
    from hybridlc.eval.items import load_items

    items, _ = load_items(item_path)
    name = item_path.stem
    sub = [i for i in scorer.rescore_subset(len(items), name) if i < n_contexts] or [0]
    use_ac = next(model.parameters()).device.type == "cuda"
    ref = scorer.score_items(ForwardOnlyAdapter(model), [items[i] for i in sub], 1, name,
                             score_candidates=False, autocast=use_ac)
    stored = {r["item_id"]: r for r in map(json.loads, Path(rows_jsonl).read_text().splitlines())}
    flips = sum(int(stored[r["item_id"]]["EM_tf"] != r["EM_tf"]) for r in ref)
    dmax = max(abs(stored[r["item_id"]]["logp_gold"] - r["logp_gold"]) for r in ref)
    return dict(n=len(ref), flips=flips, flip_rate=flips / len(ref), max_abs_dlogp=dmax, contexts=sub, subset=True)


def evaluate_checkpoint(ck: LoadedCheckpoint, suites: Iterable[str], *, label: str | None = None,
                        items_root: str | Path = ITEMS_ROOT, results_root: str | Path = RESULTS_ROOT,
                        n_contexts: int | None = None, only: str | None = None,
                        score_candidates: bool = True, rescore: bool = False,
                        log: Any = print) -> list[dict[str, Any]]:
    """Score every cell of ``suites`` for one checkpoint; returns one record per cell written."""
    from hybridlc.eval import scorer

    label = label or ckpt_label(ck.stage, ck.step)
    results_root = Path(results_root)
    recs = []
    for suite in suites:
        for f in suite_item_files(suite, items_root, only):
            out = results_root / ck.run_id / label / suite / f"{result_name(f.stem)}.jsonl"
            t0 = time.time()
            extra = dict(run_id=ck.run_id, ckpt=label, ckpt_path=str(ck.path), ckpt_sha256=ck.sha256,
                         ckpt_step=ck.step, ckpt_dtype=ck.dtype, config_hash=ck.config_hash,
                         train_git_commit=ck.git_commit)
            written = score_cell(ck.model, f, out, n_contexts=n_contexts, score_candidates=score_candidates,
                                 extra_meta=extra)
            rec = dict(ts=time.time(), kind="eval", run_id=ck.run_id, ckpt=label, suite=suite, item_file=str(f),
                       out=str(written), rerun=written != out, n_contexts=n_contexts, wall_s=time.time() - t0)
            if rescore and suite != "Tier0":
                rec["rescore"] = (scorer.rescore_check(ck.model, f, written) if n_contexts is None
                                  else rescore_subset_check(ck.model, f, written, n_contexts))
            recs.append(rec)
            results_root.mkdir(parents=True, exist_ok=True)
            with open(results_root / "_eval_log.jsonl", "a") as fh:
                fh.write(json.dumps(rec, sort_keys=True, default=str) + "\n")
            if log:
                log(f"[eval] {ck.run_id}/{label}/{suite}/{written.name}  {rec['wall_s']:.1f}s")
    return recs

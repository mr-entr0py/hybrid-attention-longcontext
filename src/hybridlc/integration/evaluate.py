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
    eval_weights: str | None = None
    """Precision of the weights actually scored ("bf16-rounded" canonical, or "float32")."""


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


EVAL_WEIGHTS_CANONICAL = "bf16-rounded"
"""Every evaluated checkpoint is scored with bf16-rounded weights (verification round 1,
BF16-MILESTONE-ARTIFACT-MIXING): TEF 250/500/2,000 exist only as bf16 milestones (§18), so a full
fp32 checkpoint (final.pt / step_*.pt) is rounded to bf16 on load. Scores are then identical whether
a checkpoint is read from its milestone or its full checkpoint, and comparisons across TEF steps
(dose guard, still-closing flag) never mix artefact precisions. Recorded as ``eval_weights``."""


def load_checkpoint(path: str | Path, device: str | torch.device = "cuda",
                    runs_root: str | Path | None = None, canonical_bf16: bool = True) -> LoadedCheckpoint:
    """Rebuild the ``HybridLM`` of a training checkpoint (full fp32 or bf16 milestone) in fp32.

    The model is built with ``build_model(ModelConfig(**config['model']), seed_index)`` and the saved
    tensors are loaded with ``strict=True`` (a bf16 milestone is cast up to the fp32 master dtype).
    With ``canonical_bf16`` (default) floating tensors of a full fp32 checkpoint are first rounded
    to bf16 exactly as ``training.checkpoint.save_milestone`` does (see ``EVAL_WEIGHTS_CANONICAL``).
    """
    from hybridlc.models.build import build_model
    from hybridlc.models.config import ModelConfig

    path = Path(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    cfg = _resolved_config(payload, path, Path(runs_root) if runs_root else None)
    model = build_model(ModelConfig.from_dict(dict(cfg["model"])), int(cfg["seed_index"]))
    if canonical_bf16:
        sd = {k: (v.to(torch.bfloat16).float() if v.is_floating_point() else v) for k, v in payload["model"].items()}
    else:
        sd = {k: (v.float() if v.is_floating_point() else v) for k, v in payload["model"].items()}
    model.load_state_dict(sd, strict=True)
    model.to(device).eval()
    return LoadedCheckpoint(model=model, run_id=str(payload.get("run_id") or cfg.get("run_id")),
                            stage=str(cfg["stage"]), step=int(payload["step"]), config=cfg, path=path,
                            sha256=sha256_file(path), dtype=str(payload.get("dtype", "float32")),
                            config_hash=payload.get("config_hash"), git_commit=payload.get("git_commit"),
                            eval_weights=(EVAL_WEIGHTS_CANONICAL if canonical_bf16
                                          or str(payload.get("dtype", "float32")) == "bfloat16" else "float32"))


# ------------------------------------------------------------------------------------- scoring
def suite_item_files(suite: str, items_root: str | Path = ITEMS_ROOT, only: str | None = None) -> list[Path]:
    root = Path(items_root) / SUITE_NAMESPACE[suite] / suite
    files = sorted(root.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"no item files under {root}")
    return [f for f in files if only is None or only in f.name]


def score_cell(model: torch.nn.Module, item_path: Path, out_jsonl: Path, *, n_contexts: int | None = None,
               batch_size: int | None = None, score_candidates: bool = True,
               extra_meta: dict[str, Any] | None = None, autocast: bool | None = None) -> Path:
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
                                 score_candidates=score_candidates, extra_meta=extra_meta, autocast=autocast)
    t0 = time.time()
    items, cell = load_items(item_path)
    items = items[:n_contexts]
    name = item_path.stem
    T = cell.get("T") or max(len(it["context_ids"]) for it in items)
    frozen = frozen_batch_size(int(T))
    bs = int(batch_size or frozen)
    model.eval()
    dev = next(model.parameters()).device
    use_ac = (dev.type == "cuda") if autocast is None else bool(autocast and dev.type == "cuda")
    if cell.get("task") in scorer.LM_TASKS:
        rows = scorer.score_lm_items(model, items, name, autocast=use_ac)
    else:
        rows = scorer.score_items(model, items, bs, name, score_candidates, autocast=use_ac)
    out = scorer.unique_path(out_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "x") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True, allow_nan=True) + "\n")
    hdr = scorer._header(model, item_path, {**dict(  # noqa: SLF001 (same header as score_file)
        cell=cell, n_rows=len(rows), batch_size=bs, frozen_batch_size=frozen, batch_size_deviation=bs != frozen,
        scoring_path=scorer._scoring_path(model), autocast_bf16=use_ac, device=str(dev),  # noqa: SLF001
        wall_time_s=time.time() - t0, out_jsonl=str(out), subset=True, n_contexts_scored=len(items)),
        **(extra_meta or {})})
    scorer._write_meta(out, hdr)  # noqa: SLF001
    return out


def rescore_subset_check(model: torch.nn.Module, item_path: Path, rows_jsonl: Path, n_contexts: int,
                         ref: str = "bf16", ref_model: torch.nn.Module | None = None) -> dict:
    """§9.2 single-pass re-score restricted to the contexts a subset run actually scored: the frozen
    5 % subset of the cell intersected with ``range(n_contexts)`` (or context 0 if that is empty).
    ``ref`` / ``ref_model`` as in :func:`hybridlc.eval.scorer.rescore_check` (P2 bf16 / P3 fp32)."""
    import contextlib

    from hybridlc.eval import scorer
    from hybridlc.eval.adapter import ForwardOnlyAdapter
    from hybridlc.eval.items import load_items

    items, cell = load_items(item_path)
    name = item_path.stem
    sub = [i for i in scorer.rescore_subset(len(items), name) if i < n_contexts] or [0]
    T = int(cell.get("T") or max(len(it["context_ids"]) for it in items))
    m = model if ref == "bf16" or ref_model is None else ref_model
    use_ac = next(model.parameters()).device.type == "cuda" if ref == "bf16" else False
    att, backend = contextlib.nullcontext(), ("as P0 (bf16 autocast on CUDA)" if ref == "bf16" else "MATH (fp32)")
    if ref == "fp32" and T >= scorer.RATT_MIN_T:
        from hybridlc.models.attention import fp32_reference_attention

        att, backend = fp32_reference_attention("tiled"), "R-ATT:tiled-fp32"
    with att:
        rr = scorer.score_items(ForwardOnlyAdapter(m), [items[i] for i in sub], 1, name,
                                score_candidates=False, autocast=use_ac)
    stored = {r["item_id"]: r for r in map(json.loads, Path(rows_jsonl).read_text().splitlines())}
    flips = sum(int(stored[r["item_id"]]["EM_tf"] != r["EM_tf"]) for r in rr)
    dmax = max(abs(stored[r["item_id"]]["logp_gold"] - r["logp_gold"]) for r in rr)
    rows = [dict(item_id=r["item_id"], EM_tf=bool(r["EM_tf"]), logp_gold=float(r["logp_gold"])) for r in rr]
    return dict(n=len(rr), flips=flips, flip_rate=flips / len(rr), max_abs_dlogp=dmax, contexts=sub, subset=True, rows=rows,
                ref=ref, pass_id="P2" if ref == "bf16" else "P3", autocast=bool(use_ac), attention_backend=backend,
                **scorer.precision_flags())


RESCORE_SUITES: tuple[str, ...] = ("CS", "DG", "DC", "FD", "MS")
"""Suites re-scored by default (§9.2 / I-29: "5% of items"; every cloze suite, Tier0 excluded). FD
was added in verification round 2 (RESCORE-SCOPE-EXCLUDES-FD-MS) because the S2 gate decisions
(U-01, FEAS-3/5, U-15, I-08) read FD long before any CS scoring; MS for completeness. The analysis
audits CS/DG/DC cells for a passing record (pooled per checkpoint)."""


#: FEAS-PREC v2 scoring passes (experiments/prereg/FEAS_precision.v2.yaml). P2 / P3 are the re-score
#: records of a P0 run (``rescore_ref``); P1 and P4 are separate runs written under ``<label>-P1|-P4``.
PASSES = {
    "P0": dict(weights="bf16-rounded", autocast=True, reference_attention=None, candidates=True),
    "P1": dict(weights="float32", autocast=True, reference_attention=None, candidates=False),
    "P4": dict(weights="float32", autocast=False, reference_attention="tiled", candidates=False),
}


def _only_match(name: str, only: str | list[str] | None) -> bool:
    if only is None:
        return True
    pats = [only] if isinstance(only, str) else list(only)
    pats = [p for q in pats for p in str(q).split(",") if p]
    return any(p in name for p in pats)


def precision_meta(ck: "LoadedCheckpoint", pass_id: str, autocast: bool | None, reference_attention: str | None) -> dict:
    """Precision fields recorded in every result ``_meta.json`` (FEAS-PREC v2 passes_in_code)."""
    from hybridlc.eval.scorer import precision_flags
    from hybridlc.models import attention as A

    dev = next(ck.model.parameters()).device
    ac = (dev.type == "cuda") if autocast is None else bool(autocast and dev.type == "cuda")
    if reference_attention:
        be = {"tiled": "R-ATT:tiled-fp32", "efficient": "R-ATT:sdpa-efficient-fp32"}[reference_attention]
    else:
        be = f"sdpa:{A.INFERENCE_HALF_BACKEND.name}" if ac else "sdpa:MATH (fp32)"
    return dict(pass_id=pass_id, weights_dtype=getattr(ck, "eval_weights", None), autocast_bf16=ac,
                attention_backend=be, **precision_flags())


def evaluate_checkpoint(ck: LoadedCheckpoint, suites: Iterable[str], *, label: str | None = None,
                        items_root: str | Path = ITEMS_ROOT, results_root: str | Path = RESULTS_ROOT,
                        n_contexts: int | None = None, only: str | list[str] | None = None,
                        score_candidates: bool = True, rescore: bool | None = None,
                        log: Any = print, pass_id: str = "P0", autocast: bool | None = None,
                        reference_attention: str | None = None, rescore_ref: str = "bf16",
                        ref_model: torch.nn.Module | None = None, blind: bool = False,
                        infra: Any = None) -> list[dict[str, Any]]:
    """Score every cell of ``suites`` for one checkpoint; returns one record per cell written.

    ``rescore``: the §9.2 / I-29 single-pass re-score of 5% of items (EM flip rate < 1%). ``None``
    (default) = on for the suites in ``RESCORE_SUITES`` (CS, DG, DC, FD, MS), ``True`` = every suite
    but Tier0, ``False`` = off (verification round 1, RESCORE-CHECK-OPT-IN). The result goes into the
    cell's ``_eval_log.jsonl`` line, which ``analysis.results.rescore_audit`` checks.

    ``n_contexts`` (a subset of each cell) writes under the checkpoint label ``<label>-n<N>`` so a
    subset can never occupy the path of the full cell; the analysis skips such outputs
    (verification round 2, EVAL-SUBSET-SHADOWS-FULL).

    FEAS-PREC v2 (PI ruling D083): ``pass_id`` P0 (default) / P1 / P4 with ``autocast`` and
    ``reference_attention`` (R-ATT mode for every attention call of the pass); P1 and P4 write under
    ``<label>-P1`` / ``<label>-P4``. ``rescore_ref`` = "bf16" (P2, the frozen check; default), "fp32"
    (P3, single-pass with ``ref_model`` = fp32 weights) or "both"; every re-score record is also
    written next to the result file as ``<stem>_rescore.json``. ``only`` takes a substring or a list
    (comma-separated) of cell-name substrings.

    Technical blinding (DTS10-AT v2): with ``blind`` the score values never reach the log or the
    ``_eval_log.jsonl`` summary (re-score records are referenced by file only). ``infra`` (an
    ``InfraTagger``) adds the R2-LIVE v2 infra block to every ``_eval_log.jsonl`` line and result meta.
    """
    import contextlib

    from hybridlc.eval import scorer
    from hybridlc.utils.infra import registry_fields

    if rescore_ref not in ("bf16", "fp32", "both"):
        raise ValueError(f"rescore_ref must be bf16|fp32|both, got {rescore_ref!r}")
    label = label or ckpt_label(ck.stage, ck.step)
    if pass_id != "P0":
        label = f"{label}-{pass_id}"
    if n_contexts is not None:
        label = f"{label}-n{int(n_contexts)}"
    results_root = Path(results_root)
    pmeta = precision_meta(ck, pass_id, autocast, reference_attention)
    recs = []
    for suite in suites:
        for f in suite_item_files(suite, items_root, None):
            if not _only_match(f.name, only):
                continue
            out = results_root / ck.run_id / label / suite / f"{result_name(f.stem)}.jsonl"
            t0 = time.time()
            extra = dict(run_id=ck.run_id, ckpt=label, ckpt_path=str(ck.path), ckpt_sha256=ck.sha256,
                         ckpt_step=ck.step, ckpt_dtype=ck.dtype, eval_weights=getattr(ck, "eval_weights", None),
                         config_hash=ck.config_hash,
                         train_git_commit=ck.git_commit, **pmeta)
            blk = None
            if infra is not None:
                blk = infra.snapshot()
                extra["infra"] = blk
            if reference_attention:
                from hybridlc.models.attention import fp32_reference_attention

                att = fp32_reference_attention(reference_attention)
            else:
                att = contextlib.nullcontext()
            with att:
                written = score_cell(ck.model, f, out, n_contexts=n_contexts, score_candidates=score_candidates,
                                     extra_meta=extra, autocast=autocast)
            rec = dict(ts=time.time(), kind="eval", run_id=ck.run_id, ckpt=label, suite=suite, item_file=str(f),
                       out=str(written), rerun=written != out, n_contexts=n_contexts, wall_s=time.time() - t0,
                       pass_id=pass_id, blinded=bool(blind))
            if blk is not None:
                rec["infra"] = {**registry_fields(blk), "classification": blk.get("classification"),
                                "events": blk.get("events"), "journal_source": blk.get("journal_source")}
            do_rescore = (suite in RESCORE_SUITES) if rescore is None else bool(rescore)
            res_recs: dict[str, dict] = {}
            if do_rescore and suite != "Tier0":
                for ref in (("bf16", "fp32") if rescore_ref == "both" else (rescore_ref,)):
                    t1 = time.time()
                    rr = (scorer.rescore_check(ck.model, f, written, ref=ref, ref_model=ref_model) if n_contexts is None
                          else rescore_subset_check(ck.model, f, written, n_contexts, ref=ref, ref_model=ref_model))
                    rr["wall_s"] = time.time() - t1
                    res_recs["P2" if ref == "bf16" else "P3"] = rr
                rpath = written.with_name(written.stem + "_rescore.json")
                rpath.write_text(json.dumps(res_recs, indent=1, sort_keys=True, default=str) + "\n")
                rec["rescore_file"] = str(rpath)
                if not blind:
                    if "P2" in res_recs:
                        rec["rescore"] = res_recs["P2"]
                    if "P3" in res_recs:
                        rec["rescore_fp32"] = res_recs["P3"]
            recs.append(rec)
            results_root.mkdir(parents=True, exist_ok=True)
            with open(results_root / "_eval_log.jsonl", "a") as fh:
                fh.write(json.dumps(rec, sort_keys=True, default=str) + "\n")
            if log:
                msg = f"[eval] {ck.run_id}/{label}/{suite}/{written.name}  {rec['wall_s']:.1f}s"
                if not blind:
                    msg += "  " + _score_summary(written, res_recs)
                log(msg)
    return recs


def _score_summary(rows_path: Path, res_recs: dict[str, dict]) -> str:
    """One-line score summary of a written cell (never used under technical blinding)."""
    try:
        rows = [json.loads(x) for x in Path(rows_path).read_text().splitlines()]
    except OSError:
        return ""
    parts = []
    if rows and "EM_tf" in rows[0]:
        parts.append(f"EM={sum(bool(r['EM_tf']) for r in rows) / len(rows):.4f} (n={len(rows)})")
    for k, v in res_recs.items():
        parts.append(f"{k} flip={v.get('flip_rate', float('nan')):.4f}")
    return " ".join(parts)

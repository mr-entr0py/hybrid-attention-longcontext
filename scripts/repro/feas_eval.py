#!/usr/bin/env python
"""FEAS-384 evaluation (seed 101 + SSMax arm): FEAS-PREC v2 passes P0 (with P2/P3 re-score), P1 and P4 on the
TEF-1000 checkpoints, BLINDED (no DTS10-commit note: scripts/evaluate.py prints no score). This driver never
opens a result file and prints only run ids, pass ids, exit codes, cell counts and wall times.

    python scripts/repro/gpu_lock.py run --label FEAS-EVAL --wait-exclusive -- python scripts/repro/feas_eval.py

Refuses unless experiments/feas_smoke/smoke_result.json permits FEAS evaluation (decision D100). Decision D099:
mem fraction 0.85 only for the P0 + re-score step on k=12 checkpoints, only when the GPU is exclusive.
All passes load final.pt (P0/P2 round it to bf16 on load; P1/P3/P4 use its fp32 weights). Resumable: a pass with a
done marker in experiments/feas/logs/ is skipped. Uses a fresh extraction of the pre-FEAS Triton archive, so the
production cache is not written.
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "repro"))
import feas_smoke as S  # noqa: E402  (shared helpers: cache extraction, item mapping, exclusivity, leak regex)

OUT = REPO / "experiments" / "feas"
TEF_YAMLS = [f"S2_D384_{n}.yaml" for n in ("C0_s101_B030_T", "C0_s101_B090_T", "C12_s101_B030_T",
                                             "C12_s101_B090_T", "C12ssmax_s101_B030_T")]
PASS_ARGS = {"P4": ["--pass", "P4"], "P1": ["--pass", "P1"], "P0": ["--rescore", "--rescore-ref", "both"]}


def tef_runs() -> dict[str, tuple[str, Path]]:
    rows = S.reg_rows(REPO / "experiments" / "registry.jsonl")
    out = {}
    for r in rows:
        if r.get("status") == "queued" and Path(str(r.get("config", ""))).name in TEF_YAMLS:
            name = Path(r["config"]).name
            ck = Path(r["ckpt_dir"])
            ck = (ck if ck.is_absolute() else REPO / ck) / "final.pt"
            assert any(x.get("run_id") == r["run_id"] and x.get("status") == "completed" for x in rows), r["run_id"]
            assert ck.is_file(), ck
            out[name] = (r["run_id"], ck)
    assert sorted(out) == sorted(TEF_YAMLS), sorted(out)
    return out


def main() -> int:
    smoke = json.loads((REPO / "experiments" / "feas_smoke" / "smoke_result.json").read_text())
    if not smoke.get("feas_evaluation_permitted_under_D100"):
        print(f"REFUSED: FEAS-SMOKE does not permit evaluation (failing items: {smoke.get('failing_items')})")
        return 2
    runs = tef_runs()
    logs = OUT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    from hybridlc.analysis import feas as FE

    cache = S.fresh_cache() if not (S.CACHE_ROOT.parent / "triton-feas-eval").exists() else None
    keep = S.CACHE_ROOT.parent / "triton-feas-eval"
    if cache is not None:
        shutil.copytree(cache, keep)
    env = S.base_env(TRITON_CACHE_DIR=str(keep))
    results = REPO / "experiments" / "results"
    n_cells = {"P0": len(FE.required_cells("P0", "D384")), "P1": len(FE.required_cells("P1", "D384")),
               "P4": len(FE.required_cells("P4", "D384"))}
    failed = []
    # P4 first, then P1, then P0: the cheaper passes surface a problem early
    for pid in ("P4", "P1", "P0"):
        for yml in TEF_YAMLS:
            rid, ck = runs[yml]
            tag = f"{rid}_{pid}"
            done = logs / f"{tag}.done"
            if done.exists():
                continue
            extra = list(PASS_ARGS[pid])
            if pid == "P4":
                extra += ["--only", S.item_files(FE.required_cells("P4", "D384"))]
            if pid == "P0" and "C12" in yml:
                S.assert_gpu_exclusive()
                extra += ["--mem-fraction", S.EVAL_CAP_D099]
            cmd = [S.PY, "scripts/evaluate.py", "--ckpt", str(ck), "--results-root", str(results),
                   "--runs-root", str(REPO / "runs"), "--suite", "FD", "--label", "tef1000", *extra]
            t = time.time()
            log = logs / f"{tag}.log"
            with open(log, "w") as f:
                import subprocess

                rc = subprocess.run(cmd, cwd=REPO, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
            wall = time.time() - t
            label = "tef1000" if pid == "P0" else f"tef1000-{pid}"
            got = len([p for p in (results / rid / label / "FD").glob("*.jsonl")])
            leaks = len(S.SCORE_RE.findall(log.read_text()))
            ok = rc == 0 and got >= n_cells[pid] and leaks == 0
            S.log(f"FEAS-EVAL {pid} {rid}: rc={rc} cells={got}/{n_cells[pid]} wall={wall:.0f}s score_leaks={leaks}")
            if ok:
                done.write_text(json.dumps({"wall_s": wall, "cells": got, "cmd": cmd[1:]}))
            else:
                failed.append(tag)
            if leaks:
                raise SystemExit("a score-like value reached the log: stopping (blinding)")
    S.log(f"FEAS-EVAL finished; failed passes: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

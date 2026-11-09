"""VT-20: peak memory <= 7 GiB per process in Stage A, Stage B, TEF and evaluation at 32K, for C12 and
C0 (protocol §16), at both scales.

Each training case is the production recipe (real data streams, bf16, the production micro-batch and
checkpointing) run for 3 optimiser steps with accumulation 2 in a separate process under the 0.44
memory fraction (= 7.0 GiB of 15.9 GiB: exceeding it is an OOM, so exit 0 already proves the cap); the
trainer's logged ``mem_run_peak`` (max allocated) is also asserted. Evaluation: one 32K context of
every DC 32K cell scored with the canonical branched path (prefill + extend + candidate scoring) plus
the 32K Tier-0 LM-1 forward, max reserved asserted.

Written by the S0 integrator. ``gpu`` + ``slow`` (~6-10 min).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
CAP_GIB = 7.0
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
pytestmark = [pytest.mark.gpu, pytest.mark.slow, needs_cuda]
DATA_OK = (REPO / "data" / "tokens" / "pg19" / "train.bin").exists()
ITEMS_OK = (REPO / "experiments" / "items" / "dev" / "DC").is_dir()

RECIPES = {"A": "stage_A.yaml", "B": "stage_B.yaml", "TEF": "tef.yaml"}


def _train(tmp: Path, stage: str, scale: str, cfg: str) -> dict:
    run = tmp / f"{stage}.yaml"
    run.write_text(textwrap.dedent(f"""\
        exp_stage: S0VT20
        model: configs/model/{scale}/{cfg}.yaml
        training: configs/training/{RECIPES[stage]}
        seed_index: 0
        lr: 2.0e-3
        steps_option: 30M
        overrides:
          schedule.steps: 3
          data.accum: 2
          logging.every: 1
          checkpoint.every: 1000000
          checkpoint.save_final: false
          checkpoint.milestone_tokens: []
          checkpoint.milestone_steps: []
          validation.every: 0
        """))
    if scale == "D512":  # stage_B steps options are per scale; the override fixes steps anyway
        run.write_text(run.read_text().replace("steps_option: 30M", "steps_option: 100M"))
    cmd = [sys.executable, str(REPO / "scripts" / "train.py"), "--config", str(run), "--allow-dirty",
           "--runs-root", str(tmp / "runs"), "--ckpt-root", str(tmp / "ckpt"), "--registry", str(tmp / "reg.jsonl")]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, cwd=str(REPO),
                       env={**os.environ, "PYTHONHASHSEED": "0"})
    assert r.returncode == 0, (stage, scale, cfg, r.returncode, r.stdout[-2000:], r.stderr[-2000:])
    (rd,) = [d for d in (tmp / "runs").iterdir() if d.is_dir() and not d.name.startswith("_")]
    rows = [json.loads(x) for x in (rd / "metrics.jsonl").read_text().splitlines()]
    rows = [x for x in rows if x.get("kind") == "train"]
    return dict(peak_alloc_gib=max(x["mem_run_peak"] for x in rows) / 2**30,
                reserved_gib=max(x["mem_reserved"] for x in rows) / 2**30)


EVAL_SNIPPET = textwrap.dedent("""\
    import json, sys, torch
    from pathlib import Path
    from hybridlc.utils.gpu import claim
    claim(0.44)
    from hybridlc.models.build import build_model
    from hybridlc.training.config import load_yaml
    from hybridlc.integration.evaluate import score_cell
    scale, cfg, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    m = build_model(load_yaml(f"configs/model/{scale}/{cfg}.yaml")["fields"], 0).cuda().eval()
    cells = sorted(Path("experiments/items/dev/DC").glob("*_32K.npz")) + [Path("experiments/items/test/Tier0/LM-1_all_32K.npz")]
    for f in cells:
        score_cell(m, f, out / (f.stem + ".jsonl"), n_contexts=1)
    print(json.dumps(dict(max_alloc=torch.cuda.max_memory_allocated() / 2**30,
                          max_reserved=torch.cuda.max_memory_reserved() / 2**30, n_cells=len(cells))))
""")


@pytest.mark.skipif(not DATA_OK, reason="data/tokens required")
@pytest.mark.parametrize("scale", ["D384", "D512"])
@pytest.mark.parametrize("cfg", ["C0", "C12"])
@pytest.mark.parametrize("stage", ["A", "B", "TEF"])
def test_training_peak_memory(tmp_path, stage, scale, cfg):
    m = _train(tmp_path, stage, scale, cfg)
    print(f"VT-20 {stage} {scale} {cfg}: peak alloc {m['peak_alloc_gib']:.2f} GiB, reserved {m['reserved_gib']:.2f} GiB")
    assert m["peak_alloc_gib"] <= CAP_GIB and m["reserved_gib"] <= CAP_GIB, m


@pytest.mark.skipif(not ITEMS_OK, reason="frozen item files required")
@pytest.mark.parametrize("scale", ["D384", "D512"])
@pytest.mark.parametrize("cfg", ["C0", "C12"])
def test_eval_peak_memory_32k(tmp_path, scale, cfg):
    r = subprocess.run([sys.executable, "-c", EVAL_SNIPPET, scale, cfg, str(tmp_path)], capture_output=True,
                       text=True, timeout=1800, cwd=str(REPO), env={**os.environ, "PYTHONPATH": str(REPO / "src")})
    assert r.returncode == 0, r.stderr[-3000:]
    m = json.loads(r.stdout.strip().splitlines()[-1])
    print(f"VT-20 eval32K {scale} {cfg}: max alloc {m['max_alloc']:.2f} GiB, max reserved {m['max_reserved']:.2f} GiB")
    assert m["max_reserved"] <= CAP_GIB, m

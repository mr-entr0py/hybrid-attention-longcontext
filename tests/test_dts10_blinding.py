"""DTS-10 technical blinding (DTS10_acceptance.v2.yaml ``blinding.technical``; PI ruling D082), CPU.

With stdout and stderr captured, while no ``DTS10-commit`` note exists:
* an exp_stage S2 checkpoint evaluation prints no score, EM, logp, rank or flip value, and its
  ``_eval_log.jsonl`` summary carries no score values (re-score records referenced by file only);
* ``scripts/analyse_results.py`` refuses S2 result trees (non-zero exit, one-line message);
* a non-S2 control prints normally (per-cell EM and flip values);
* after a fake ``DTS10-commit`` note, the S2 evaluation prints normally and the refusal is gone.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable
SCORE_RE = re.compile(r"EM=|flip=|EM_tf|logp|flip_rate|rank|max_abs_dlogp")


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    from hybridlc.eval.devtools import resolve_test_tokenizer
    from hybridlc.eval.filler import StoreFiller, dev_text_filler
    from hybridlc.eval.generators import generate
    from hybridlc.eval.items import save_items
    from hybridlc.eval.rng import eval_rng
    from hybridlc.eval.tokens import token_space
    from hybridlc.models.build import build_model
    from hybridlc.models.config import ModelConfig

    tmp = tmp_path_factory.mktemp("blind")
    tok, _ = resolve_test_tokenizer()
    ts = token_space(tok)
    real = (REPO / "data" / "tokens" / "pg19" / "test.bin").exists()
    fill = StoreFiller(ts, "dev") if real else dev_text_filler(ts, "dev")
    items = generate("R1", {"D": 3}, 1024, 20, eval_rng("dev", "blinding-test", 1024), "dev", tok, fill)
    save_items(tmp / "items" / "dev" / "FD" / "R1_D3_1K.npz", items, dict(task="R1", T=1024, n=len(items)))
    mcfg = dict(d_model=64, d_ff=128, n_layers=2, vocab_size=ts.vocab_size if hasattr(ts, "vocab_size") else 16384,
                pattern="AA", attn_head_dim=64)
    mcfg["vocab_size"] = 16384
    model = build_model(ModelConfig.from_dict(mcfg), 0)
    ckpts = {}
    for stage in ("S2", "S4"):
        rid = f"{stage}T-tiny-C2-s0-0000000{1 if stage == 'S2' else 2}"
        p = tmp / "ckpt" / rid / "final.pt"
        p.parent.mkdir(parents=True)
        torch.save({"model": model.state_dict(), "run_id": rid, "step": 1000, "dtype": "float32",
                    "config": {"model": mcfg, "seed_index": 0, "stage": "TEF", "exp_stage": stage, "run_id": rid}}, p)
        ckpts[stage] = p
    return tmp, ckpts


def _eval(world, stage: str, tag: str) -> tuple[subprocess.CompletedProcess, Path]:
    tmp, ckpts = world
    res = tmp / f"res_{tag}"
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "HYBRIDLC_GPUMON": "0", "HYBRIDLC_JOURNAL_SOURCE": ":empty:"}
    r = subprocess.run([PY, str(REPO / "scripts" / "evaluate.py"), "--ckpt", str(ckpts[stage]), "--suite", "FD",
                        "--items-root", str(tmp / "items"), "--results-root", str(res), "--rescore",
                        "--registry", str(tmp / "registry.jsonl")],
                       env=env, capture_output=True, text=True, timeout=900, cwd=str(REPO))
    assert r.returncode == 0, r.stderr[-3000:]
    return r, res


def _analyse(world, res: Path) -> subprocess.CompletedProcess:
    tmp, _ = world
    return subprocess.run([PY, str(REPO / "scripts" / "analyse_results.py"), "--results", str(res), "--registry",
                           str(tmp / "registry.jsonl"), "--out", str(tmp / "an"), "--no-bootstrap"],
                          capture_output=True, text=True, timeout=600, cwd=str(REPO),
                          env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})


def test_blinding_end_to_end(world):
    tmp, _ = world
    assert not (tmp / "registry.jsonl").exists()
    # 1. S2 before the note: no score value anywhere in stdout / stderr / the _eval_log summary
    r, res = _eval(world, "S2", "s2_blind")
    assert "BLINDED" in r.stdout
    assert not SCORE_RE.search(r.stdout + r.stderr), (r.stdout + r.stderr)[-3000:]
    log = [json.loads(x) for x in (res / "_eval_log.jsonl").read_text().splitlines()]
    assert log and all("rescore" not in x and x.get("rescore_file") and x["blinded"] for x in log)
    assert not any(SCORE_RE.search(json.dumps({k: v for k, v in x.items() if k != "infra"})) for x in log)
    # scores were still written to the result files (sealed)
    rows = [json.loads(x) for x in Path(log[0]["out"]).read_text().splitlines()]
    assert rows and "EM_tf" in rows[0]
    # 2. analysis refuses the S2 tree
    a = _analyse(world, res)
    assert a.returncode == 4 and "DTS10-commit" in a.stderr and a.stderr.strip().count("\n") == 0
    # 3. a non-S2 control prints normally
    r, _ = _eval(world, "S4", "s4")
    assert "EM=" in r.stdout and "flip=" in r.stdout and "BLINDED" not in r.stdout
    # 4. after a (fake) DTS10-commit note: S2 prints normally, no refusal
    (tmp / "registry.jsonl").write_text(json.dumps({"ts": 0, "run_id": "DTS10-commit", "status": "note",
                                                    "commit": "f" * 40, "feas_py_sha256": "0" * 64}) + "\n")
    r, res2 = _eval(world, "S2", "s2_open")
    assert "EM=" in r.stdout and "BLINDED" not in r.stdout
    a = _analyse(world, res2)
    assert "DTS10-commit" not in a.stderr


def test_blinded_stage_reading():
    from hybridlc.integration.blinding import is_blinded_stage

    assert is_blinded_stage("S2") and is_blinded_stage("S2T-D384-C12-s101-abcd1234") and is_blinded_stage("S2B-x")
    assert not is_blinded_stage("S20") and not is_blinded_stage("S4") and not is_blinded_stage("S0SMT-D384")

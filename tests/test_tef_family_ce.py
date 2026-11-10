"""Per-family answer-CE logging (R6 v2 implementation requirement; PI ruling D080), CPU.

* ``family_answer_ce`` equals a hand computation; the trainer's logged step-0 ``answer_ce_retrieval``
  equals a hand computation on the fakes TEF batch with the initial model;
* the loss trajectory is bitwise identical with the logging on and off;
* VT-17 batch hashes (seed indices 0 and 1, steps 0-2) equal the values recorded on the pre-change tree
  (experiments/repro/VT-24r/k5b_baseline.json); TEFStream's ``answer_family`` codes mark exactly the
  answer labels (R1 -> 1, R-load -> 2, other answers -> 3).
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tests"))

from test_vt13_training import TRAIN, run_dir, write_tiny  # noqa: E402

from hybridlc.training.config import load_run_config  # noqa: E402
from hybridlc.training.trainer import build_stream, family_answer_ce, read_jsonl_last_per_step  # noqa: E402

PY = sys.executable


def test_family_answer_ce_unit():
    g = torch.Generator().manual_seed(0)
    h = torch.randn(1, 50, 8, generator=g)
    w = torch.randn(30, 8, generator=g)
    lab = torch.randint(0, 30, (1, 50), generator=g)
    fam = torch.zeros(1, 50, dtype=torch.int8)
    fam[0, [3, 7, 9, 20, 41]] = torch.tensor([1, 2, 3, 1, 2], dtype=torch.int8)
    s, n = family_answer_ce(h, w, lab, fam)
    idx = [3, 7, 20, 41]
    ref = sum(float(torch.logsumexp(h[0, i] @ w.T, -1) - (h[0, i] @ w.T)[lab[0, i]]) for i in idx)
    assert n == 4 and math.isclose(float(s), ref, rel_tol=1e-6)
    s0, n0 = family_answer_ce(h, w, lab, torch.zeros(1, 50, dtype=torch.int8))
    assert n0 == 0 and float(s0) == 0.0


def _run(tmp_path: Path, name: str, family: str) -> Path:
    run = write_tiny(tmp_path, steps=12, every=10, stage="TEF", accum=2, name=name)
    root = tmp_path / f"root_{name}"
    env = {**os.environ, "OMP_NUM_THREADS": "1", "HYBRIDLC_FAMILY_CE": family, "HYBRIDLC_GPUMON": "0",
           "HYBRIDLC_JOURNAL_SOURCE": ":empty:"}
    env.pop("HYBRIDLC_FAULT", None)
    r = subprocess.run([PY, str(TRAIN), "--config", str(run), "--allow-dirty", "--runs-root", str(root / "runs"),
                        "--ckpt-root", str(root / "ckpt"), "--registry", str(root / "registry.jsonl")],
                       env=env, capture_output=True, text=True, timeout=600, cwd=str(REPO))
    assert r.returncode == 0, r.stderr[-3000:]
    return root


def test_trainer_logs_hand_computed_value_and_loss_unchanged(tmp_path):
    on = _run(tmp_path, "on", "1")
    off = _run(tmp_path, "off", "0")
    tr_on = read_jsonl_last_per_step(run_dir(on) / "trace.jsonl")
    tr_off = read_jsonl_last_per_step(run_dir(off) / "trace.jsonl")
    assert [(r["loss_hex"], r["gnorm_hex"], r["batch_hash"]) for r in tr_on] == \
           [(r["loss_hex"], r["gnorm_hex"], r["batch_hash"]) for r in tr_off]  # bitwise identical
    fam = read_jsonl_last_per_step(run_dir(on) / "family_ce.jsonl")
    assert [r["step"] for r in fam] == list(range(12))
    assert not (run_dir(off) / "family_ce.jsonl").read_text().strip()
    # hand computation at step 0 with the initial model (no update applied yet)
    cfg = load_run_config(next(tmp_path.glob("on.yaml")))
    from hybridlc.training.fakes import build_fake_model

    model = build_fake_model(cfg.model_config(), cfg.seed_index).float().train()
    stream = build_stream(cfg)
    tot, n = 0.0, 0
    with torch.no_grad():
        for m in range(cfg.data.accum):
            b = stream.get(0, m)
            hid = model(b["input_ids"], b["cu_seqlens"], b["position_ids"])
            sel = (b["answer_family"][0] == 1) | (b["answer_family"][0] == 2)
            for i in torch.nonzero(sel).reshape(-1).tolist():
                lg = hid[0, i].double() @ model.lm_head_weight.double().T
                tot += float(torch.logsumexp(lg, -1) - lg[int(b["labels"][0, i])])
                n += 1
    assert n == fam[0]["n_tokens"] and math.isclose(fam[0]["answer_ce_retrieval"], tot / n, rel_tol=1e-5)
    mt = [r for r in read_jsonl_last_per_step(run_dir(on) / "metrics.jsonl", kind="train")]
    assert "answer_ce_retrieval" in mt[0]


def test_vt17_hashes_unchanged_and_family_codes():
    from hybridlc.data.tef import TEFStream

    base = REPO / "experiments" / "repro" / "VT-24r" / "k5b_baseline.json"
    if not ((REPO / "data" / "tokens" / "fwe" / "train.bin").exists() and base.exists()):
        pytest.skip("needs data/tokens and the pre-change VT-17 hashes")
    want = json.loads(base.read_text())
    assert want["provenance"]["pre_fix_tree"] is True
    for seed in (0, 1):
        s = TEFStream(seed, 1000)
        for step in range(3):
            for m in range(2):
                assert s.get_numpy(step, m)["batch_hash"] == want["vt17_batch_hashes"][str(seed)][f"{step}:{m}"]
    s = TEFStream(0, 1000)
    b = s.get_numpy(0, 0)
    fam, am = b["answer_family"][0], b["answer_mask"][0]
    assert ((fam != 0) == am).all()
    seq = s.sequence(0, 0, 0)
    for sg in seq["segments"]:
        if "task" not in sg:
            continue
        a, e = sg["start"], sg["start"] + sg["length"]
        codes = set(np.unique(seq["family"][a:e]).tolist()) - {0}
        want_code = {"R1": 1, "R-load": 2}.get(sg["subtype"], 3)
        assert codes == {want_code}, (sg["subtype"], codes)

"""Checkpoint/resume equivalence of run_one.py (zoology_compat._patch_fit_resumable). GPU, slow (~4 min).

Run: pytest scripts/s1_zoology/test_s1_resume.py -q -m slow

A 3-epoch smoke run of GDN d128 is executed (A) uninterrupted and (B) killed right after
validating epoch 1 (``--abort-after-epoch 1``; the checkpoint from the end of epoch 0 is on
disk) and then restarted, which resumes at epoch 1. The per-epoch validation accuracies and
per-epoch mean training losses must agree; the checkpoint must be deleted after success.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RUN_ID = "gated_delta_net-d128-lr1.0e-02"

pytestmark = [pytest.mark.slow, pytest.mark.gpu,
              pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")]


def _run(out: Path, *extra: str) -> int:
    cmd = [sys.executable, str(HERE / "run_one.py"), "--run-id", RUN_ID, "--max-epochs", "3",
           "--out-dir", str(out), *extra]
    return subprocess.call(cmd, cwd=str(REPO), stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
                           env={**os.environ, "S1_REPO_ROOT": str(REPO)})


def test_resume_matches_uninterrupted(tmp_path):
    ckpt = REPO / "data" / "zoology" / "ckpt" / f"{RUN_ID}.smoke3.pt"
    assert _run(tmp_path / "A", "--fresh") == 0
    assert _run(tmp_path / "B", "--fresh", "--abort-after-epoch", "1") == 3
    assert ckpt.exists()
    assert _run(tmp_path / "B") == 0
    assert not ckpt.exists()
    a = json.loads((tmp_path / "A" / f"{RUN_ID}.smoke.json").read_text())
    b = json.loads((tmp_path / "B" / f"{RUN_ID}.smoke.json").read_text())
    assert a["resumed_at_epochs"] == [] and b["resumed_at_epochs"] == [1]
    acc_a = [v["valid/accuracy"] for v in a["valid_history"]]
    acc_b = [v["valid/accuracy"] for v in b["valid_history"]]
    assert len(acc_a) == len(acc_b) == 3
    # fla kernels may be non-deterministic; observed agreement was exact, allow 1e-3 slack.
    assert max(abs(x - y) for x, y in zip(acc_a, acc_b)) < 1e-3
    la, lb = a["train_loss_by_epoch"], b["train_loss_by_epoch"]
    assert la.keys() == lb.keys()
    assert all(abs(la[k] - lb[k]) <= 1e-3 * max(1.0, abs(la[k])) for k in la)

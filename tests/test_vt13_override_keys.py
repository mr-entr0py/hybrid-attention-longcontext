"""VT-13 regression, verification round 2 (TRAIN-DTS06-REMEDY-NOOP).

* ``-o`` overrides write into the training recipe; a key whose top level is not a recipe section
  (e.g. ``lr``) was silently ignored, so the DTS-06 remedy "-o lr=<actual>" changed nothing. Such
  keys are now rejected; ``-o schedule.peak_lr=<x>`` changes the peak LR and the run_id.
* ``scripts/train.py --config X --lr Y`` silently ignored --lr; composition-only flags now error.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from hybridlc.training.config import load_run_config

REPO = Path(__file__).resolve().parents[1]
RUN = REPO / "configs" / "training" / "examples" / "S2_D384_C0_s101_A.yaml"


def test_unknown_override_section_rejected():
    base = load_run_config(RUN)
    with pytest.raises(ValueError, match="schedule.peak_lr"):
        load_run_config(RUN, ["lr=1.4e-3"])
    with pytest.raises(ValueError, match="not a training-recipe section"):
        load_run_config(RUN, ["foo.bar=1"])
    ok = load_run_config(RUN, ["schedule.peak_lr=1.4e-3"])
    assert ok.schedule.peak_lr == pytest.approx(1.4e-3) and base.schedule.peak_lr == pytest.approx(2e-3)
    assert ok.run_id != base.run_id


def test_train_cli_rejects_lr_with_config():
    r = subprocess.run([sys.executable, str(REPO / "scripts" / "train.py"), "--config", str(RUN), "--lr", "1e-3",
                        "--print-run-id"], capture_output=True, text=True, cwd=REPO)
    assert r.returncode != 0 and "--lr" in r.stderr
    r = subprocess.run([sys.executable, str(REPO / "scripts" / "train.py"), "--config", str(RUN),
                        "-o", "schedule.peak_lr=1.4e-3", "--print-run-id"], capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == load_run_config(RUN, ["schedule.peak_lr=1.4e-3"]).run_id

"""Regression for verification round 1, BF16-MILESTONE-ARTIFACT-MIXING.

The evaluation driver scores every checkpoint with bf16-rounded weights: loading a full fp32
checkpoint (final.pt) gives exactly the same model tensors as loading its bf16 milestone, so TEF
comparisons across checkpoints never mix artefact precisions. ``--fp32-weights`` keeps fp32.
Uses the VT-13 production checkpoint (D384 C3, 640 steps) when present.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from hybridlc.analysis.results import eval_weights_audit
from hybridlc.integration.evaluate import EVAL_WEIGHTS_CANONICAL, load_checkpoint

REPO = Path(__file__).resolve().parents[1]
RUN = REPO / "experiments" / "s0" / "vt13_production" / "c" / "ckpt" / "S0-D384-C3-s901-8588adcc"


@pytest.mark.skipif(not (RUN / "final.pt").is_file(), reason="VT-13 production checkpoint not present")
def test_final_and_milestone_load_identically():
    full = load_checkpoint(RUN / "final.pt", "cpu")
    ms = load_checkpoint(RUN / "milestones" / "step_0000640_final_bf16.pt", "cpu")
    fp = load_checkpoint(RUN / "final.pt", "cpu", canonical_bf16=False)
    assert full.eval_weights == ms.eval_weights == EVAL_WEIGHTS_CANONICAL and fp.eval_weights == "float32"
    a, b, c = full.model.state_dict(), ms.model.state_dict(), fp.model.state_dict()
    assert a.keys() == b.keys()
    n_diff_fp32 = 0
    for k in a:
        assert torch.equal(a[k], b[k]), k
        if a[k].is_floating_point() and not torch.equal(a[k], c[k]):
            n_diff_fp32 += 1
    assert n_diff_fp32 > 0  # the fp32 path really is different


def test_eval_weights_audit(tmp_path):
    files = []
    for i, (suite, v) in enumerate((("CS", "bf16-rounded"), ("DC", "bf16-rounded"), ("MS", "float32"))):
        f = tmp_path / suite / f"R1_D3_{i}.jsonl"
        f.parent.mkdir()
        f.write_text("")
        f.with_name(f.stem + "_meta.json").write_text(json.dumps({"eval_weights": v}))
        files.append(str(f))
    assert eval_weights_audit(files)["consistent"]
    Path(files[1]).with_name(Path(files[1]).stem + "_meta.json").write_text(json.dumps({"eval_weights": "float32"}))
    a = eval_weights_audit(files)
    assert not a["consistent"] and a["values"] == {"bf16-rounded": 1, "float32": 1}

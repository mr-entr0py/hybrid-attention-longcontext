"""VT-13 regressions from verification round 1 (DTS-06, DTS-08).

* DTS-06: after a D034 LR restart of a parent (Stage A at the fallback LR), a child that takes
  its peak LR from the stale run-YAML ``lr`` is refused (exit 77) instead of silently decaying
  from the old peak; relaunching it with the parent's actual LR succeeds; a TEF child (fixed peak)
  is unaffected.
* DTS-08: the trainer's ``running`` registry rows carry ``tokenizer_sha256`` and
  ``data_manifest_sha256`` (and env.json has them).

Uses the same tiny fake-model harness as tests/test_vt13_training.py (CPU, a few seconds).
"""
from __future__ import annotations

import json
from pathlib import Path

from hybridlc.training.registry import Registry
from hybridlc.training.trainer import RefusedError, _check_parent_lr

from test_vt13_training import run_dir, train_proc, write_tiny  # noqa: E402  (same tests/ dir)

import pytest

REPO = Path(__file__).resolve().parents[1]


def test_check_parent_lr_unit():
    st = {"trainer_state": {"peak_lr": 1.4e-3}, "config": {"schedule": {"peak_lr": 2.0e-3}}}
    with pytest.raises(RefusedError):
        _check_parent_lr(st, 2.0e-3, Path("p"))
    _check_parent_lr(st, 1.4e-3, Path("p"))  # relaunched with the actual LR
    _check_parent_lr(st, 3.0e-4, Path("p"))  # TEF: fixed peak
    ok = {"trainer_state": {"peak_lr": 2.0e-3}, "config": {"schedule": {"peak_lr": 2.0e-3}}}
    _check_parent_lr(ok, 2.0e-3, Path("p"))  # no restart in the lineage
    _check_parent_lr({}, 2.0e-3, Path("p"))  # old checkpoints without the fields


def test_child_of_lr_restarted_parent_refused(tmp_path):
    runA = write_tiny(tmp_path, name="A", steps=30, extra="gdiv: {fallback_lr: 1.0e-3}\n")
    root = tmp_path / "chain"
    r = train_proc(runA, root, fault={"nan_at": 25})
    assert r.returncode == 0, r.stderr[-2000:]
    recs = list(Registry(root / "registry.jsonl").records())
    assert any(x["status"] == "restarted" and x.get("lr") == 1e-3 for x in recs)
    # DTS-08: provenance on every running row
    running = [x for x in recs if x["status"] == "running"]
    assert running and all("tokenizer_sha256" in x and "data_manifest_sha256" in x for x in running)
    if (REPO / "data" / "tokenizer" / "tokenizer.sha256").is_file():
        assert running[0]["tokenizer_sha256"] and len(running[0]["tokenizer_sha256"]) == 64
    env = json.loads((run_dir(root) / "env.json").read_text())
    assert "tokenizer_sha256" in env and "fla_install" in env

    # child B with the stale lr (3e-3 = A's configured lr) -> refused
    runB = write_tiny(tmp_path, name="B", steps=10, stage="B", kind="linear_decay", parent=str(runA))
    r = train_proc(runB, root)
    assert r.returncode == 77, (r.returncode, r.stderr[-2000:])
    assert "relaunch with -o schedule.peak_lr=0.001" in r.stderr  # round 2: a remedy that works
    # relaunched with the parent's actual LR -> runs
    runB2 = write_tiny(tmp_path, name="B2", steps=10, stage="B", kind="linear_decay", parent=str(runA), lr=1e-3)
    r = train_proc(runB2, root)
    assert r.returncode == 0, r.stderr[-2000:]
    # TEF child (fixed peak 3e-4) of the restarted A is not affected
    runT = write_tiny(tmp_path, name="T", steps=4, stage="TEF", parent=str(runA), accum=2)
    r = train_proc(runT, root)
    assert r.returncode == 0, r.stderr[-2000:]

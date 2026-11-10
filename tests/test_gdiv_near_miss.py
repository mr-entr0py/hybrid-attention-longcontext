"""G-div near-miss logging (PI ruling D067 / PR-10): log-only frozen-EMA excursion, CPU.

* a planted +1.2-nat excursion for 60 steps writes exactly one ``gdiv_near_miss`` note and no restart;
* +1.0 nat for 40 steps writes no note; the literal trigger and its restarts are unchanged.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tests"))

from test_vt13_training import TRAIN, write_tiny  # noqa: E402

from hybridlc.training.config import GDivConfig  # noqa: E402
from hybridlc.training.gdiv import DivergenceError, GDivMonitor  # noqa: E402
from hybridlc.training.registry import Registry  # noqa: E402

PY = sys.executable


def _feed(delta: float, n: int, base: float = 5.0, start: int = 100, total: int = 400):
    g = GDivMonitor(GDivConfig())
    ev = []
    for t in range(total):
        g.update(t, base + (delta if start <= t < start + n else 0.0))
        e = g.pop_near_miss()
        if e:
            ev.append((t, e))
    return ev, g


def test_monitor_near_miss_planted():
    ev, g = _feed(1.2, 60)
    assert len(ev) == 1 and ev[0][0] == 150  # > 50 steps above the frozen pre-excursion EMA + 1.0
    assert ev[0][1]["excursion_start"] == 100 and abs(ev[0][1]["frozen_ema"] - 5.0) < 1e-12
    assert g.s.max_above < 51  # the literal EMA caught up: no G-div trigger
    assert _feed(1.0, 40)[0] == []   # +1.0 is not > EMA + 1.0, and 40 < 50 steps
    assert _feed(1.2, 45)[0] == []   # too short
    ev2, _ = _feed(1.2, 200)
    assert len(ev2) == 1             # one note per excursion


def test_monitor_state_roundtrip_and_literal_unchanged():
    g = GDivMonitor(GDivConfig())
    for t in range(120):
        g.update(t, 5.0 + (1.3 if t >= 100 else 0.0))
    g2 = GDivMonitor(GDivConfig(), g.state_dict())
    assert g2.state_dict() == g.state_dict()
    g = GDivMonitor(GDivConfig())
    g.update(0, 1.0)
    with pytest.raises(DivergenceError):
        for t in range(1, 200):
            g.update(t, 1.0 + 100.0 * t)  # runaway: literal trigger
    assert g.pop_near_miss() is None


def _run(tmp_path: Path, name: str, fault: dict) -> Path:
    run = write_tiny(tmp_path, steps=200, every=50, name=name, lr=1e-7)  # ~flat loss: EMA tracks it
    root = tmp_path / f"root_{name}"
    env = {**os.environ, "OMP_NUM_THREADS": "1", "HYBRIDLC_GPUMON": "0", "HYBRIDLC_JOURNAL_SOURCE": ":empty:",
           "HYBRIDLC_FAULT": json.dumps(fault)}
    r = subprocess.run([PY, str(TRAIN), "--config", str(run), "--allow-dirty", "--runs-root", str(root / "runs"),
                        "--ckpt-root", str(root / "ckpt"), "--registry", str(root / "registry.jsonl")],
                       env=env, capture_output=True, text=True, timeout=900, cwd=str(REPO))
    assert r.returncode == 0, r.stderr[-3000:]
    return root


def test_trainer_writes_one_note_and_no_restart(tmp_path):
    root = _run(tmp_path, "nm", {"loss_offset": [100, 60, 1.2]})
    recs = list(Registry(root / "registry.jsonl").records())
    notes = [r for r in recs if r["status"] == "note" and r.get("note") == "gdiv_near_miss"]
    assert len(notes) == 1, [r["status"] for r in recs]
    assert notes[0]["step"] == 150 and notes[0]["excursion_start"] == 100
    assert not any(r["status"] in ("diverged", "restarted") for r in recs)
    root2 = _run(tmp_path, "small", {"loss_offset": [100, 40, 1.0]})
    recs2 = list(Registry(root2 / "registry.jsonl").records())
    assert not any(r.get("note") == "gdiv_near_miss" for r in recs2)
    assert [r["status"] for r in recs2] == ["running", "completed"]

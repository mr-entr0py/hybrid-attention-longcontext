"""Regression for verification round 1, RESCORE-CHECK-OPT-IN (§9.2 / I-29).

* ``integration.evaluate.evaluate_checkpoint`` runs the 5% single-pass re-score by default on
  CS/DG/DC (``rescore=None``), not on MS/FD/Tier0, and can be forced on/off.
* ``analysis.results.rescore_audit`` flags CS/DG/DC result files with no re-score record or a
  flip rate >= 1% in ``_eval_log.jsonl``; ``scripts/analyse_results.py --require-rescore`` refuses.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from hybridlc.analysis.results import rescore_audit
from hybridlc.integration import evaluate as E


def _fake_items(root: Path) -> None:
    for ns, suite in (("test", "CS"), ("test", "MS"), ("dev", "DC"), ("dev", "FD")):
        d = root / ns / suite
        d.mkdir(parents=True)
        (d / "R1_D3_4K.npz").write_bytes(b"x")


# round 2 (RESCORE-SCOPE-EXCLUDES-FD-MS): the default re-score covers FD and MS too
@pytest.mark.parametrize("flag,expect", [(None, {"CS", "MS", "DC", "FD"}), (True, {"CS", "MS", "DC", "FD"}),
                                         (False, set())])
def test_evaluate_rescore_default(tmp_path, monkeypatch, flag, expect):
    items = tmp_path / "items"
    _fake_items(items)
    called = []

    def fake_score_cell(model, f, out, **kw):
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("")
        return out

    def fake_rescore(model, f, written):
        called.append(Path(written).parent.name)
        return dict(n=10, flips=0, flip_rate=0.0, max_abs_dlogp=0.0)

    from hybridlc.eval import scorer
    monkeypatch.setattr(E, "score_cell", fake_score_cell)
    monkeypatch.setattr(scorer, "rescore_check", fake_rescore)
    ck = SimpleNamespace(model=None, run_id="S4T-D384-C3-s0-x", stage="TEF", step=1000, path=Path("p"), sha256="0",
                         dtype="float32", config_hash="h", git_commit="g")
    E.evaluate_checkpoint(ck, ["CS", "MS", "DC", "FD"], items_root=items, results_root=tmp_path / "res",
                          rescore=flag, log=None)
    assert set(called) == expect
    audit = rescore_audit(tmp_path / "res", [str(p) for p in (tmp_path / "res").rglob("*.jsonl")
                                             if not p.name.startswith("_")])
    assert audit["n_cells"] == 2  # CS + DC audited
    assert audit["passes"] == (flag is not False)


def test_rescore_audit_missing_and_failing(tmp_path):
    res = tmp_path / "res"
    files = []
    for suite in ("CS", "DG", "DC", "MS"):
        f = res / "run" / "tef1000" / suite / "R1_D3_4096.jsonl"
        f.parent.mkdir(parents=True)
        f.write_text("")
        files.append(f)
    # round 2: flips are pooled per checkpoint (run/ckpt); here CS 0/40 + DG 2/40 = 2.5% -> fails
    log = [dict(out=str(files[0]), rescore=dict(n=40, flips=0, flip_rate=0.0)),
           dict(out=str(files[1]), rescore=dict(n=40, flips=2, flip_rate=0.05)),
           dict(out=str(files[3]), rescore=None)]
    (res / "_eval_log.jsonl").write_text("".join(json.dumps(r) + "\n" for r in log))
    a = rescore_audit(res, [str(f) for f in files])
    assert a["n_cells"] == 3 and a["n_ok"] == 2
    assert a["missing"] == [str(files[2])]
    assert [x["checkpoint"] for x in a["failing"]] == ["run/tef1000"]
    assert a["per_checkpoint"]["run/tef1000"]["flip_rate"] == pytest.approx(2 / 80)
    assert [x["file"] for x in a["cells_with_flips"]] == [str(files[1])]
    assert not a["passes"]

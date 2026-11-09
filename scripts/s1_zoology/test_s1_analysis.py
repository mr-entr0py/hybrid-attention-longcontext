"""Unit tests for the S1(a) decision logic (no GPU, no Zoology import).

Run: pytest scripts/s1_zoology/test_s1_analysis.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s1_analysis as A  # noqa: E402


def _res(fam, d, lr, acc, smoke=False):
    return {"family": fam, "d_model": d, "lr": lr, "final_overall_acc": acc,
            "run_id": A.run_id(fam, d, lr), "smoke": smoke}


def test_lr_grid_matches_original_config():
    # original_mqar_configs.py: np.logspace(-3, -1.5, 4)
    np.testing.assert_allclose(A.LRS, np.logspace(-3, -1.5, 4), rtol=1e-12)
    # run ids use the file's f"{lr:.1e}" formatting
    assert [A.lr_tag(x) for x in A.LRS] == ["1.0e-03", "3.2e-03", "1.0e-02", "3.2e-02"]


def test_run_order_and_count():
    ids = A.phase_run_ids(A.PHASE1) + A.phase_run_ids(A.PHASE2)
    assert len(ids) == A.N_RUNS_FULL == 16 and len(set(ids)) == 16
    assert ids[0] == "attention-d64-lr1.0e-03" and ids[4] == "gated_delta_net-d128-lr1.0e-03"


def test_cut_rule():
    assert not A.cut_decision(600)["cut"]          # 600 s × 16 = 2.67 h
    assert A.cut_decision(700)["cut"]              # 700 s × 16 = 3.11 h
    assert A.cut_decision(675)["implied_total_gpu_h"] == pytest.approx(3.0)
    assert not A.cut_decision(675)["cut"]          # exactly 3.0 h is not "> 3.0"


def test_best_lrs_requires_four_and_breaks_ties_low():
    rs = [_res("attention", 64, lr, a) for lr, a in zip(A.LRS, [0.9, 0.99, 0.99, 0.5])]
    assert A.best_lrs(rs, "attention", 64) == [A.LRS[1], A.LRS[2]]
    with pytest.raises(ValueError):
        A.best_lrs(rs[:3], "attention", 64)


def test_grp_a_pass_fail_incomplete():
    good = ([_res("attention", d, lr, 0.99) for d in (64, 128) for lr in A.LRS]
            + [_res("gated_delta_net", 128, lr, 0.96) for lr in A.LRS]
            + [_res("gated_delta_net", 256, lr, 0.97) for lr in A.LRS])
    assert A.grp_a(good)["verdict"] == "PASS"
    # GDN decreasing in d_model -> FAIL even though both are >= 0.95
    bad = [r if not (r["family"] == "gated_delta_net" and r["d_model"] == 256) else {**r, "final_overall_acc": 0.955}
           for r in good]
    out = A.grp_a(bad)
    assert out["verdict"] == "FAIL" and out["checks"][-1]["pass"] is False
    # attention below threshold at one width -> FAIL
    bad2 = [{**r, "final_overall_acc": 0.9} if r["family"] == "attention" and r["d_model"] == 64 else r for r in good]
    assert A.grp_a(bad2)["verdict"] == "FAIL"
    # missing cell -> INCOMPLETE; smoke runs never count
    assert A.grp_a(good[:-4])["verdict"] == "INCOMPLETE"
    assert A.grp_a([{**r, "smoke": True} for r in good])["verdict"] == "INCOMPLETE"


def test_best_over_lr_counts_lrs():
    rs = [_res("attention", 64, lr, a) for lr, a in zip(A.LRS, [0.1, 0.8, 0.3, 0.2])]
    t = A.best_over_lr(rs)["attention"][64]
    assert t["acc"] == 0.8 and t["lr"] == A.LRS[1] and t["n_lrs"] == 4


def test_summarize_end_to_end_on_synthetic_runs(tmp_path):
    """summarize.py reads run JSONs, ignores *.smoke.json, writes results JSON + table."""
    import json
    import summarize as S

    (tmp_path / "runs").mkdir()
    for fam, widths in A.WIDTHS.items():
        for d in widths:
            for lr in A.LRS:
                r = _res(fam, d, lr, 0.97 if d == widths[-1] else 0.96)
                r.update(epochs_run=32, early_stopped=False, wall_time_s=60.0, peak_mem_gib=1.0,
                         best_overall_acc=r["final_overall_acc"], best_epoch=31,
                         final_per_config_acc={c: 0.9 for c in S.TEST_COLS})
                (tmp_path / "runs" / f"{r['run_id']}.json").write_text(json.dumps(r))
    # a failing smoke file must be ignored
    (tmp_path / "runs" / "attention-d64-lr1.0e-03.smoke.json").write_text(json.dumps(
        {**_res("attention", 64, A.LRS[0], 0.0), "smoke": True}))
    sys.argv = ["summarize.py", "--res-dir", str(tmp_path)]
    S.main()
    out = json.loads((tmp_path / "s1a_results.json").read_text())
    assert out["n_runs"] == 16 and out["grp_a"]["verdict"] == "PASS"
    assert (tmp_path / "s1a_table.md").read_text().count("\n") == 18

"""VT-18 regressions from verification round 3 (analysis pipeline and run lineage labels).

* ANA-DG-PROTOCOL-DESIGN-REFUSED: the §9.5 DG plan (retrieval + C0 controls on C0/C12 or on every
  core config; M-hop on C12 only) is analysed, not refused. G-REF uses the retrieval table, U-11 the
  C12-only M-hop table, and the completeness audit requires DG M-hop of C12 main-branch runs only.
* ANA-REF-INVERSION-SILENT-NOT-PERFORMED: C12 failing G-REF (a) with no hybrid on DG gives
  "not evaluable (needed)" plus a warning, never triggered=False.
* U05-TRIGGER-RECOMPUTED-AT-FINAL-PRIMARY: U-05 is reported at the primary cell fixed with the first
  n0 seeds (§11.2 order), not at the final primary cell.
* S5B-BRANCH-LABEL-NOT-SET: lineage runs (TEF-4000 continuation, 2x Stage-B) need a non-main
  analysis branch; non-evaluable guards / still-closing flag are top-level warnings.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from hybridlc.analysis import protocol as P
from hybridlc.analysis.pipeline import AnalysisSpec, run_analysis
from hybridlc.analysis.results import frozen_cell_counts, required_cells, suite_completeness_audit
from hybridlc.analysis.scores import CellTable, family_spec
from hybridlc.analysis.stats import seed_trigger
from hybridlc.analysis.synthetic import World, WorldSpec, default_plan, generate_dataframe

GATES = {g: True for g in P.REQUIRED_GATES}
REPO = Path(__file__).resolve().parents[1]


# ----------------------------------------------------------------------------- DG scoring plan
@pytest.mark.parametrize("dg_all", [False, True])
def test_dg_protocol_plan_is_analysed(dg_all):
    """DG retrieval on C0/C12 (or all configs), M-hop on C12 only: the analysis runs, G-REF is on DG,
    and U-11 finds equated pairs from the C12-only M-hop table."""
    w = World(WorldSpec())
    df = generate_dataframe(w, default_plan(w, dg_all_configs=dg_all))
    dg = df[(df.suite == "DG") & (df.branch == "main")]
    assert set(dg.loc[dg.task == "M-hop", "config"]) == {"C12"}
    assert not ((dg.task == "M-hop") & (dg["T"] == P.CONTROL_T)).any()  # frozen DG has no M-hop controls
    assert set(dg.loc[dg.task.isin(P.DG_RETRIEVAL_TASKS), "config"]) == (
        {"C0", "C1", "C3", "C12"} if dg_all else {"C0", "C12"})
    res = run_analysis(df, AnalysisSpec(gates=GATES, bootstrap=False))
    assert res["primary"]["lengths"] == [16384, 32768]
    assert all(res["gref"][str(T)]["suite"] == "DG" for T in P.LONG_LENGTHS) or \
        all(res["gref"][T]["suite"] == "DG" for T in P.LONG_LENGTHS)
    assert res["tests"]["H3"]["equated"]["pairs"], res["tests"]["H3"]["equated"]
    inv = {int(k): v for k, v in res["reference_inversion"].items()}
    assert all(v["evaluable"] for v in inv.values())


def test_dg_retrieval_still_required_complete():
    """Completeness within the DG retrieval part is still enforced (round 2): a C0 run lacking one
    retrieval stratum makes the analysis refuse."""
    w = World(WorldSpec())
    df = generate_dataframe(w, default_plan(w, dg_all_configs=False))
    m = (df.suite == "DG") & (df.config == "C0") & (df.seed == 1) & (df.task == "R1") & (df.level == "D15") \
        & (df["T"] == 32768)
    bad = df[~m].copy()
    bad.attrs.update(df.attrs)
    with pytest.raises(ValueError, match="DG"):
        run_analysis(bad, AnalysisSpec(gates=GATES, bootstrap=False))


def test_dg_mhop_on_c0_is_ignored_with_warning():
    w = World(WorldSpec())
    df = generate_dataframe(w, default_plan(w, dg_all_configs=False, dg_mhop_configs=None))
    res = run_analysis(df, AnalysisSpec(gates=GATES, bootstrap=False))
    assert any("DG M-hop rows of configs ['C0']" in x for x in res["warnings"])


def test_dg_completeness_requires_mhop_of_c12_only():
    """Verifier's dg_completeness.py: a C0 DG run with every retrieval cell is complete; a C12 run
    without M-hop is not; a stageB2x C12 run needs no M-hop."""
    exp = {("R1", "D3", 16384): 2, ("R1", "D3", P.CONTROL_T): 2, ("M-hop", "h1", 16384): 2}
    assert set(required_cells("DG", exp, "C0")) == {("R1", "D3", 16384), ("R1", "D3", P.CONTROL_T)}
    assert set(required_cells("DG", exp, "C3")) == {("R1", "D3", 16384), ("R1", "D3", P.CONTROL_T)}
    assert set(required_cells("DG", exp, "C12")) == set(exp)
    assert set(required_cells("DG", exp, "C12", "stageB2x")) == {("R1", "D3", 16384), ("R1", "D3", P.CONTROL_T)}
    assert required_cells("CS", exp, "C0") == exp

    items = REPO / "experiments" / "items"
    if not (items / "dev" / "MANIFEST.json").is_file():
        pytest.skip("frozen items not present")
    full = frozen_cell_counts(items, suites=("DG",))["DG"]
    rows = []
    for (task, level, T), n in full.items():
        for cfg, rid in (("C0", "S4T-D384-C0-s0-aaaaaaaa"), ("C12", "S4T-D384-C12-s0-bbbbbbbb")):
            if task == "M-hop" and cfg == "C0":
                continue
            rows += [dict(run_id=rid, ckpt="tef1000", suite="DG", task=task, level=level, T=T, branch="main",
                          context_id=f"{task}{level}{T}-{c}", config=cfg) for c in range(n)]
    df = pd.DataFrame(rows)
    a = suite_completeness_audit(df, items, suites=("DG",))
    assert a["complete"], a["problems"][:3]
    no_hop = df[~((df.config == "C12") & (df.task == "M-hop") & (df.level == "h4") & (df["T"] == 32768))]
    a2 = suite_completeness_audit(no_hop, items, suites=("DG",))
    assert [p["cell"] for p in a2["problems"]] == [["M-hop", "h4", 32768]]


# ----------------------------------------------------------------------------- reference inversion
def test_reference_inversion_not_evaluable_without_hybrids():
    w = World(WorldSpec(s12_drop_32k=0.30))
    df = generate_dataframe(w, default_plan(w, dg_all_configs=False))
    res = run_analysis(df, AnalysisSpec(gates=GATES, bootstrap=False))
    inv = {int(k): v for k, v in res["reference_inversion"].items()}
    g32 = {int(k): v for k, v in res["gref"].items()}[32768]
    assert g32["a_pooled"] is False
    assert inv[32768]["evaluable"] is False and inv[32768]["triggered"] is None
    assert "needed" in inv[32768]["status"]
    assert any("reference inversion at T=32768" in x for x in res["warnings"])
    # With hybrids scored on DG the check runs (evaluable, explicit triggered bool).
    df2 = generate_dataframe(w, default_plan(w, dg_all_configs=True))
    inv2 = {int(k): v for k, v in run_analysis(df2, AnalysisSpec(gates=GATES, bootstrap=False))
            ["reference_inversion"].items()}
    assert inv2[32768]["evaluable"] is True and isinstance(inv2[32768]["triggered"], bool)
    assert set(inv2[32768]["examined"]) == {"C1", "C3"}


# ----------------------------------------------------------------------------- U-05 order
def test_u05_reported_at_decision_time_primary_cell():
    """Verifier's u05_order.py: trigger seeds 5, 6 make C12 inadequate at 32K on DG; the final primary
    cell is [16K] but the decision-time one (n0 = 5 seeds) is [16K, 32K], where U-05 was evaluated."""
    w = World(WorldSpec(seeds=tuple(range(7)), seed_sd=0.03, resid_sd=0.03))
    df = generate_dataframe(w, default_plan(w, dg_all_configs=False))
    m = (df.suite == "DG") & (df.config == "C12") & df.seed.isin([5, 6]) & (df["T"] == 32768)
    df.loc[m, "em_tf"] = 0.0
    spec = AnalysisSpec(gates=GATES, n0=5, bootstrap=False)
    dec = run_analysis(df[df.seed < 5].copy(), spec)
    fin = run_analysis(df, spec)
    assert dec["primary"]["lengths"] == [16384, 32768] and fin["primary"]["lengths"] == [16384]
    assert fin["primary"]["at_decision"]["lengths"] == [16384, 32768]
    assert fin["seed_trigger"]["primary_lengths"] == [16384, 32768]
    assert fin["seed_trigger"]["resid_sd"] == pytest.approx(dec["seed_trigger"]["resid_sd"], abs=1e-12)
    cs = CellTable.from_rows(df, "CS", "em_tf")
    wrong = seed_trigger(cs.scores(cs.ladder_weights(family_spec("retrieval"), [16384]))
                         .loc[list(P.ANCHOR_CONFIGS), [0, 1, 2, 3, 4]])
    assert abs(wrong.resid_sd - fin["seed_trigger"]["resid_sd"]) > 1e-4
    assert any("primary cell at the U-05 decision" in x for x in fin["warnings"])
    assert not any("protocol violation" in x for x in fin["warnings"])


def test_u05_unchanged_without_added_seeds():
    w = World(WorldSpec())
    df = generate_dataframe(w, default_plan(w, dg_all_configs=False))
    res = run_analysis(df, AnalysisSpec(gates=GATES, bootstrap=False))
    assert res["primary"]["at_decision"]["same_as_final"] is True
    assert res["seed_trigger"]["primary_lengths"] == res["primary"]["lengths"]


# ----------------------------------------------------------------------------- branch labels
def test_tef4000_example_resolves_to_branch_tef4000():
    from hybridlc.training.config import load_run_config

    cfg = load_run_config(REPO / "configs/training/examples/S4_D384_C3_s0_TEF4000.yaml")
    assert cfg.branch == "tef4000" and cfg.schedule.start_step == 1000
    # branch is non-semantic: relabelling the same config leaves the config hash / run_id unchanged.
    # (The literal run_id moved from ...-3fbbc785 to a new hash in round 3 because tef.yaml gained the
    # semantic `answer_norm: block` data kwarg, not because of the branch label.)
    rid = cfg.run_id
    cfg.branch = "stageB2x"
    assert cfg.compute_run_id() == rid and rid.startswith("S4T-D384-C3-s0-")


def _write_run(tmp: Path, name: str, **extra) -> Path:
    import yaml

    doc = dict(exp_stage="S4", model=str(REPO / "configs/model/D384/C3.yaml"),
               training=str(REPO / "configs/training/tef_4000.yaml"), seed_index=0,
               parent={"run_id": "S4T-D384-C3-s0-REPLACE_ME", "step": 1000})
    doc.update(extra)
    p = tmp / name
    p.write_text(yaml.safe_dump(doc))
    return p


def test_lineage_runs_need_non_main_branch(tmp_path):
    from hybridlc.training.config import load_run_config

    with pytest.raises(ValueError, match="branch: tef4000"):
        load_run_config(_write_run(tmp_path, "a.yaml", continue_parent_step=True))
    with pytest.raises(ValueError, match="branch: tef4000"):
        load_run_config(_write_run(tmp_path, "b.yaml", continue_parent_step=True, branch="main"))
    with pytest.raises(ValueError, match="not in"):
        load_run_config(_write_run(tmp_path, "c.yaml", continue_parent_step=True, branch="tef-4000"))
    assert load_run_config(_write_run(tmp_path, "d.yaml", continue_parent_step=True,
                                      branch="tef4000")).branch == "tef4000"
    # 2x Stage-B lineage: steps_multiplier > 1 needs a branch label
    sb = dict(training=str(REPO / "configs/training/stage_B.yaml"), lr=2e-3, parent=None,
              steps_option="60M")
    with pytest.raises(ValueError, match="branch: stageB2x"):
        load_run_config(_write_run(tmp_path, "e.yaml", steps_multiplier=2, **sb))
    pa = _write_run(tmp_path, "f.yaml", steps_multiplier=2, branch="stageB2x", **sb)
    assert load_run_config(pa).branch == "stageB2x"
    # a child of a YAML parent on stageB2x inherits the label (e.g. the TEF after the 2x Stage B)
    child = _write_run(tmp_path, "g.yaml", training=str(REPO / "configs/training/tef.yaml"), parent=str(pa))
    assert load_run_config(child).branch == "stageB2x"


def test_unevaluable_guards_are_top_level_warnings():
    w = World(WorldSpec())
    df = generate_dataframe(w, default_plan(w, dose=False, saturation=False))
    res = run_analysis(df, AnalysisSpec(gates=GATES, bootstrap=False))
    msgs = " | ".join(res["warnings"])
    assert "still-closing flag for H1 not evaluable" in msgs
    if res["tests"]["H1"].get("evaluable"):
        assert "dose guard for H1 not evaluable" in msgs and "saturation guard for H1" in msgs

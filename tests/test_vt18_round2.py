"""VT-18 regressions from verification round 2 (analysis pipeline).

* ANA-BOOT-INCOMPLETE-SEED-BLOCKS: the hierarchical bootstrap resamples only the complete seed blocks
  of the configs an estimand uses (same seed index for every config); U-05 is evaluated on complete
  first-n0 anchor blocks when a run failed permanently (§8.1), with a warning.
* ANA-MISSING-CELL-SILENT-SEED-DROP: a run missing one whole stratum is an error (never a silently
  dropped seed or a zero-filled bootstrap cell); cells/context counts are checked against the frozen
  manifest (``suite_completeness_audit``).
* VERDICT-H3-NONEVALUABLE-NEGATIVE: a non-evaluable H3 is "not evaluable", never "no task dependence".
* ANA-SCALE-AUTODETECT-N0: the scale (hence n0) comes from the CS main-branch runs, other scales and
  non-study seeds are excluded, and a non-evaluable U-05 is a verdict-visible warning.
* GREF-SILENT-CS-FALLBACK: G-REF at 16K/32K on CS items (DG lacks the length) is flagged and blocks
  "falsified".
* RESCORE-AUDIT-PER-CELL-ZERO-FLIP: the §9.2 re-score audit pools flips per checkpoint.
* EVAL-SUBSET-SHADOWS-FULL: ``--n-contexts`` outputs go to ``<label>-n<N>`` and are never read.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from hybridlc.analysis import protocol as P
from hybridlc.analysis import stats as S
from hybridlc.analysis.pipeline import AnalysisSpec, run_analysis
from hybridlc.analysis.results import (frozen_cell_counts, load_item_results, rescore_audit,
                                       suite_completeness_audit, write_item_rows)
from hybridlc.analysis.scores import CellTable
from hybridlc.analysis.synthetic import (CS_LEVELS, DC_LEVELS, RunPlan, World, WorldSpec, default_plan,
                                         generate_dataframe)
from hybridlc.analysis.verdicts import compute_verdicts

GATES = {g: True for g in P.REQUIRED_GATES}


def _drop(df: pd.DataFrame, mask) -> pd.DataFrame:
    out = df[~mask].copy()
    out.attrs.update(df.attrs)
    return out


# ----------------------------------------------------------------------------- bootstrap, unit level
def _const_table(missing: tuple[str, int] | None = None) -> CellTable:
    """Constant contexts; per-seed L3 = +0.02 exactly in every complete block (verifier's case)."""
    configs, seeds = ["C0", "C1", "C3", "C12"], [0, 1, 2, 3, 4]
    base = {"C0": 0.0, "C1": 0.15, "C3": 0.42, "C12": 0.50}
    strata = [("R1", "D3", 16384)]
    a = np.empty((30, 4, 5))
    for i, c in enumerate(configs):
        for j, s in enumerate(seeds):
            a[:, i, j] = base[c] + 0.1 * s
    present = np.ones((4, 5), bool)
    if missing:
        i, j = configs.index(missing[0]), seeds.index(missing[1])
        a[:, i, j] = np.nan
        present[i, j] = False
    return CellTable("CS", "em_tf", strata, configs, seeds, [a], [np.arange(30)], present)


def test_bootstrap_uses_complete_blocks_only():
    tab = _const_table(missing=("C3", 3))
    w = tab.ladder_weights({"R1": None}, [16384])
    d = S.hierarchical_bootstrap(tab, {"ret": w}, configs=["C0", "C3", "C12"], B=2000, key="t")
    assert d.seeds == [0, 1, 2, 4] and d.incomplete_seeds == [3]
    L = np.nanmean(d.get("ret", "C3") - 0.2 * d.get("ret", "C0") - 0.8 * d.get("ret", "C12"), axis=1)
    # every replicate is exactly the planted +0.02 (old code: mean -0.005, CI [-0.055, 0.045])
    np.testing.assert_allclose(L, 0.02, atol=1e-12)
    # configs not involved in the estimand do not remove seeds: C1 lacks nothing
    d1 = S.hierarchical_bootstrap(tab, {"ret": w}, configs=["C0", "C1", "C12"], B=50, key="t")
    assert d1.seeds == [0, 1, 2, 3, 4] and d1.incomplete_seeds == []


def test_bootstrap_refuses_nan_in_complete_block():
    tab = _const_table()
    tab.ctx[0][:, 2, 1] = np.nan  # C3 seed 1 marked present but its stratum is empty
    w = tab.ladder_weights({"R1": None}, [16384])
    with pytest.raises(ValueError, match="never zero-fills"):
        S.hierarchical_bootstrap(tab, {"ret": w}, configs=["C0", "C3", "C12"], B=10, key="t")


# ----------------------------------------------------------------------------- pipeline worlds
@pytest.fixture(scope="module")
def world_df():
    w = World(WorldSpec(n_contexts=60, seed_sd=0.06, resid_sd=0.03))
    return generate_dataframe(w, default_plan(w, dose=False, saturation=False))


def test_failed_run_u05_and_bootstrap_paired(world_df):
    df = _drop(world_df, (world_df["config"] == "C3") & (world_df["seed"] == 3))
    res = run_analysis(df, AnalysisSpec(B=300, gates=GATES))
    assert res["seed_blocks"]["incomplete"] == {"C3": [3]}
    assert any("incomplete seed blocks" in x for x in res["warnings"])
    tr = res["seed_trigger"]
    assert "fires" in tr and tr["seeds_used"] == [0, 1, 2, 4] and tr["incomplete_blocks"] == [3]
    assert res["tests"]["H1"]["seeds"] == [0, 1, 2, 4]
    assert res["bootstrap_seeds"]["H1"]["seeds"] == [0, 1, 2, 4]
    assert res["bootstrap_seeds"]["H1-min"]["seeds"] == [0, 1, 2, 3, 4]  # C1 complete: no seed lost
    b = res["sensitivity"]["H1"]["bootstrap"]
    # paired: the bootstrap centre matches the paired t estimate on the same blocks
    assert abs(b["mean"] - res["tests"]["H1"]["estimate"]) < 0.01


def test_missing_stratum_is_an_error(world_df):
    m = ((world_df["suite"] == "CS") & (world_df["config"] == "C3") & (world_df["seed"] == 2)
         & (world_df["task"] == "R-load") & (world_df["T"] == 32768))
    lv = world_df.loc[m, "level"].iloc[0]
    df = _drop(world_df, m & (world_df["level"] == lv))
    with pytest.raises(ValueError, match="missing"):
        CellTable.from_rows(df, "CS")
    with pytest.raises(ValueError, match="no seed is ever dropped"):
        run_analysis(df, AnalysisSpec(bootstrap=False, gates=GATES))
    # opt-out exists for descriptive use only
    t = CellTable.from_rows(df, "CS", require_complete=False)
    assert t.present[t.configs.index("C3"), t.seeds.index(2)]
    # an incomplete DG is an error too (never a silent CS fallback)
    mdg = ((world_df["suite"] == "DG") & (world_df["config"] == "C0") & (world_df["seed"] == 1)
           & (world_df["T"] == 16384) & (world_df["task"] == "R1"))
    with pytest.raises(ValueError):
        run_analysis(_drop(world_df, mdg), AnalysisSpec(bootstrap=False, gates=GATES))


def test_h3_not_evaluable_never_negative(world_df):
    res = run_analysis(world_df, AnalysisSpec(bootstrap=False, gates=GATES))
    res = json.loads(json.dumps(res))
    res["tests"]["H3"]["evaluable"] = False
    res["tests"]["H3"]["tost"] = dict(equivalent=True, p=0.01)
    res["holm_support"]["H3"]["reject"] = False
    v = compute_verdicts(res)["confirmatory"]["H3"]
    assert v["verdict"] == "not evaluable", v
    res["tests"]["H3"]["evaluable"] = True
    assert compute_verdicts(res)["confirmatory"]["H3"]["verdict"] == "no task dependence"


def test_scale_from_cs_runs_and_feas_excluded():
    w = World(WorldSpec(n_contexts=40, seeds=(0, 1, 2, 3), seed_sd=0.06, resid_sd=0.03))
    core = generate_dataframe(w, default_plan(w, dose=False, saturation=False), scale="D512")
    wf = World(WorldSpec(n_contexts=40, seeds=(101,), configs={"C0": 0, "C12": 12}))
    feas = generate_dataframe(wf, [RunPlan("C0", 101, {"FD": ("tef1000", DC_LEVELS)}, stage="S2T"),
                                   RunPlan("C12", 101, {"FD": ("tef1000", DC_LEVELS)}, stage="S2T")],
                              scale="D384")
    df = pd.concat([feas, core], ignore_index=True).sort_values("run_id", kind="stable").reset_index(drop=True)
    df.attrs.update(inputs=[], skipped_reruns=[])
    assert df["scale"].iloc[0] == "D384"  # the first row is a FEAS run (the verifier's trap)
    res = run_analysis(df, AnalysisSpec(bootstrap=False, gates=GATES))
    assert res["scale"] == dict(value="D512", n0=4, from_spec=False)
    assert "fires" in res["seed_trigger"] and res["seed_trigger"]["n0"] == 4
    # several scales among the CS main-branch runs -> refuse unless the scale is given
    other = generate_dataframe(w, [RunPlan("C0", 0, {"CS": ("tef1000", CS_LEVELS)}, stage="S9T")], scale="D384")
    df2 = pd.concat([core, other], ignore_index=True)
    df2.attrs.update(inputs=[], skipped_reruns=[])
    with pytest.raises(ValueError, match="several scales"):
        run_analysis(df2, AnalysisSpec(bootstrap=False))
    res2 = run_analysis(df2, AnalysisSpec(bootstrap=False, scale="D512"))
    assert res2["scale"]["value"] == "D512" and any("other scales" in x for x in res2["warnings"])


def test_u05_not_evaluable_is_warned(world_df):
    res = run_analysis(world_df, AnalysisSpec(bootstrap=False, gates=GATES, n0=9))
    assert "status" in res["seed_trigger"]
    assert any(x.startswith("U-05 seed trigger not evaluable") for x in res["warnings"])
    assert any("U-05" in x for x in compute_verdicts(res)["warnings"])


def test_gref_cs_fallback_blocks_falsified(world_df):
    df = _drop(world_df, (world_df["suite"] == "DG") & (world_df["T"] == 32768))
    res = run_analysis(df, AnalysisSpec(bootstrap=False, gates=GATES))
    assert res["gref_cs_fallback"] == [32768]
    assert any("G-REF at T=32768 evaluated on CS items" in x for x in res["warnings"])
    if 32768 in res["primary"]["lengths"]:
        assert res["guards"]["gates"]["gref_on_cs_items"] == [32768]
        assert not res["guards"]["gates"]["holds"]


# ----------------------------------------------------------------------------- re-score audit (pooled)
def _audit_tree(tmp_path: Path, flips_by_cell: dict[int, int], n_cells: int = 143, n_q: int = 60) -> tuple[Path, list[str]]:
    res = tmp_path / "res"
    files, log = [], []
    for i in range(n_cells):
        f = res / "S4T-D384-C3-s0-0123abcd" / "tef1000" / "CS" / f"R1_D{i}_4096.jsonl"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("")
        files.append(str(f))
        k = flips_by_cell.get(i, 0)
        log.append(dict(out=str(f), rescore=dict(n=n_q, flips=k, flip_rate=k / n_q)))
    (res / "_eval_log.jsonl").write_text("".join(json.dumps(r) + "\n" for r in log))
    return res, files


def test_rescore_audit_pools_per_checkpoint(tmp_path):
    # verifier's case: 143 CS cells, one flip in total -> pooled 1/8580, passes (old code failed on cell 7)
    res, files = _audit_tree(tmp_path / "a", {7: 1})
    a = rescore_audit(res, files)
    assert a["passes"] and not a["failing"]
    ck = a["per_checkpoint"]["S4T-D384-C3-s0-0123abcd/tef1000"]
    assert ck["n"] == 143 * 60 and ck["flips"] == 1 and ck["cp_upper95"] < 0.01
    assert [x["flips"] for x in a["cells_with_flips"]] == [1]
    # pooled rate >= 1% fails: 90 flips over 8580 queries = 1.05%
    res2, files2 = _audit_tree(tmp_path / "b", {i: 1 for i in range(90)})
    a2 = rescore_audit(res2, files2)
    assert not a2["passes"] and a2["failing"][0]["flip_rate"] == pytest.approx(90 / 8580)
    # a record without n/flips is unusable (not silently accepted)
    (res2 / "_eval_log.jsonl").write_text(json.dumps(dict(out=files2[0], rescore=dict(flip_rate=0.0))) + "\n")
    a3 = rescore_audit(res2, files2[:1])
    assert a3["unusable"] == [files2[0]] and not a3["passes"]


# ----------------------------------------------------------------------------- subsets
def test_subset_outputs_never_shadow_full_cells(tmp_path, monkeypatch):
    from hybridlc.integration import evaluate as E

    items = tmp_path / "items" / "test" / "CS"
    items.mkdir(parents=True)
    (items / "R1_D3_4K.npz").write_bytes(b"x")
    written = []

    def fake_score_cell(model, f, out, n_contexts=None, **kw):
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(dict(context_id="c0", query_idx=0, em_tf=1, logp_gold=-1.0)) + "\n")
        out.with_name(out.stem + "_meta.json").write_text(json.dumps(dict(subset=n_contexts is not None)))
        written.append(out)
        return out

    monkeypatch.setattr(E, "score_cell", fake_score_cell)
    ck = SimpleNamespace(model=None, run_id="S4T-D384-C3-s0-0123abcd", stage="TEF", step=1000, path=Path("p"),
                         sha256="0", dtype="float32", config_hash="h", git_commit="g")
    E.evaluate_checkpoint(ck, ["CS"], items_root=tmp_path / "items", results_root=tmp_path / "res", n_contexts=10,
                          rescore=False, log=None)
    assert written[-1].parent.parent.name == "tef1000-n10"
    E.evaluate_checkpoint(ck, ["CS"], items_root=tmp_path / "items", results_root=tmp_path / "res", rescore=False,
                          log=None)
    assert written[-1].parent.parent.name == "tef1000" and written[-1].name == "R1_D3_4096.jsonl"
    with pytest.warns(UserWarning, match="subset"):
        df = load_item_results(tmp_path / "res")
    assert set(df["ckpt"]) == {"tef1000"}
    assert len(df.attrs["skipped_subsets"]) == 1
    # a legacy subset written at the full path (header subset: true) is skipped as well
    legacy = tmp_path / "res2" / "S4T-D384-C3-s0-0123abcd" / "tef1000" / "CS" / "R1_D3_4096.jsonl"
    write_item_rows(legacy, [dict(context_id="c0", query_idx=0, em_tf=1, logp_gold=-1.0)])
    legacy.with_name(legacy.stem + "_meta.json").write_text(json.dumps(dict(subset=True)))
    with pytest.warns(UserWarning):
        df2 = load_item_results(tmp_path / "res2")
    assert df2.empty and df2.attrs["skipped_subsets"] == [str(legacy)]


# ----------------------------------------------------------------------------- completeness vs manifest
def test_suite_completeness_against_manifest(tmp_path):
    man = tmp_path / "items" / "test"
    man.mkdir(parents=True)
    (man / "MANIFEST.json").write_text(json.dumps({"files": {
        "CS/R1_D3_4K.npz": {"n_items": 3}, "CS/R1-C0_D3_nofill.npz": {"n_items": 2},
        "MS/R1_D3_4K.npz": {"n_items": 9}}}))
    rows = []
    for ctx in range(3):
        rows.append(dict(run_id="r", ckpt="tef1000", suite="CS", task="R1", level="D3", T=4096,
                         context_id=f"c{ctx}"))
    rows.append(dict(run_id="r", ckpt="tef1000", suite="CS", task="R1", level="D3", T=P.CONTROL_T, context_id="c0"))
    df = pd.DataFrame(rows)
    a = suite_completeness_audit(df, tmp_path / "items", suites=("CS",))
    assert not a["complete"] and a["n_checked"] == 2
    assert a["problems"] == [dict(run_id="r", ckpt="tef1000", suite="CS", cell=["R1", "D3", P.CONTROL_T],
                                  issue="1 contexts, manifest has 2")]
    df2 = pd.concat([df, pd.DataFrame([dict(rows[-1], context_id="c1")])], ignore_index=True)
    assert suite_completeness_audit(df2, tmp_path / "items", suites=("CS",))["complete"]


def test_frozen_manifests_parse_for_completeness():
    """Every frozen CS/DG/DC cell maps to its own result-file stratum (none silently unchecked)."""
    from hybridlc.integration.evaluate import ITEMS_ROOT

    if not (ITEMS_ROOT / "test" / "MANIFEST.json").is_file():
        pytest.skip("frozen items not present")
    exp = frozen_cell_counts(ITEMS_ROOT)
    for suite, ns in (("CS", "test"), ("DG", "dev"), ("DC", "dev")):
        files = json.loads((ITEMS_ROOT / ns / "MANIFEST.json").read_text())["files"]
        n_files = sum(1 for rel in files if rel.startswith(f"{suite}/"))
        assert n_files > 0 and len(exp[suite]) == n_files
        assert all(n > 0 for n in exp[suite].values())

"""FEAS-PREC v2 scoring passes P0-P4 (experiments/prereg/FEAS_precision.v2.yaml ``passes_in_code``;
PI ruling D083).

GPU (on the S0 smoke D384 C3 TEF checkpoint, experiments/s0/smoke, FD cells at T <= 4K, 3 contexts):
* P0 (with P2 + P3 re-score records), P1 and P4 write result files whose ``_meta`` records pass id,
  weights dtype, autocast flag, TF32 flags and attention backend;
* at T <= 4K P4 (fp32 cache path, R-ATT) and P3 (fp32 single pass) agree on EM for every query with
  |delta logp_gold| <= 2e-2 (the VT-06 fp32 logit scale).
CPU: ``rescore_audit`` still gates on the bf16 path (P2) flips, never on the fp32 P3 record.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SMOKE = REPO / "experiments" / "s0" / "smoke"
CKPT = SMOKE / "ckpt" / "S0SMT-D384-C3-s0-49de0d30" / "final.pt"
CELLS = ["R1_D3_4K", "R-load_N8_4K", "R1-C0_D3_nofill"]
META_FIELDS = ("pass_id", "weights_dtype", "autocast_bf16", "attention_backend", "matmul_allow_tf32",
               "cudnn_allow_tf32", "float32_matmul_precision")


@pytest.mark.gpu
@pytest.mark.slow
def test_passes_p0_to_p4_on_smoke_checkpoint(tmp_path, record_property):
    import torch

    from hybridlc.integration.evaluate import evaluate_checkpoint, load_checkpoint
    from hybridlc.models.testing import cuda_device_or_skip

    if not CKPT.exists():
        pytest.skip("S0 smoke checkpoint not present")
    dev = cuda_device_or_skip()
    res = tmp_path / "results"
    kw = dict(items_root=REPO / "experiments" / "items", results_root=res, n_contexts=3, only=CELLS, log=None)
    ck = load_checkpoint(CKPT, dev, runs_root=SMOKE / "runs")
    ck32 = load_checkpoint(CKPT, dev, runs_root=SMOKE / "runs", canonical_bf16=False)
    p0 = evaluate_checkpoint(ck, ["FD"], rescore=True, rescore_ref="both", ref_model=ck32.model, **kw)
    p1 = evaluate_checkpoint(ck32, ["FD"], rescore=False, pass_id="P1", score_candidates=False, **kw)
    p4 = evaluate_checkpoint(ck32, ["FD"], rescore=False, pass_id="P4", autocast=False,
                             reference_attention="tiled", score_candidates=False, **kw)
    assert len(p0) == len(p1) == len(p4) == len(CELLS)
    want = {"P0": ("bf16-rounded", True, "sdpa:FLASH_ATTENTION"), "P1": ("float32", True, "sdpa:FLASH_ATTENTION"),
            "P4": ("float32", False, "R-ATT:tiled-fp32")}
    for pid, recs in (("P0", p0), ("P1", p1), ("P4", p4)):
        for r in recs:
            meta = json.loads(Path(r["out"]).with_name(Path(r["out"]).stem + "_meta.json").read_text())
            for f in META_FIELDS:
                assert f in meta, (pid, f)
            assert (meta["pass_id"], meta["weights_dtype"], meta["autocast_bf16"], meta["attention_backend"]) == \
                   (pid, *want[pid]), (pid, meta["pass_id"], meta["weights_dtype"], meta["autocast_bf16"])
    agree = []
    for r0, r4 in zip(sorted(p0, key=lambda r: r["item_file"]), sorted(p4, key=lambda r: r["item_file"])):
        res_rec = json.loads(Path(r0["rescore_file"]).read_text())
        assert res_rec["P2"]["pass_id"] == "P2" and res_rec["P3"]["pass_id"] == "P3"
        assert res_rec["P3"]["autocast"] is False and res_rec["P3"]["attention_backend"] == "MATH (fp32)"
        rows4 = {x["item_id"]: x for x in map(json.loads, Path(r4["out"]).read_text().splitlines())}
        for q in res_rec["P3"]["rows"]:
            x = rows4[q["item_id"]]
            agree.append((q["item_id"], q["EM_tf"] == x["EM_tf"], abs(q["logp_gold"] - x["logp_gold"])))
    record_property("p3_vs_p4", {"n": len(agree), "em_disagree": sum(not a for _, a, _ in agree),
                                 "max_abs_dlogp": max(d for _, _, d in agree)})
    assert agree and all(a for _, a, _ in agree), agree
    assert max(d for _, _, d in agree) <= 2e-2
    torch.cuda.empty_cache()


def _audit_world(tmp_path: Path, p2_flips: int, p3_flips: int) -> dict:
    from hybridlc.analysis.results import rescore_audit

    root = tmp_path / f"w{p2_flips}_{p3_flips}"
    out = root / "S4-D384-C3-s0-aaaaaaaa" / "tef1000" / "DG" / "R1_D3_4096.jsonl"
    out.parent.mkdir(parents=True)
    out.write_text("")
    rec = dict(kind="eval", out=str(out), run_id="S4-D384-C3-s0-aaaaaaaa", ckpt="tef1000", suite="DG",
               rescore=dict(n=100, flips=p2_flips, flip_rate=p2_flips / 100, max_abs_dlogp=0.0, pass_id="P2"),
               rescore_fp32=dict(n=100, flips=p3_flips, flip_rate=p3_flips / 100, max_abs_dlogp=0.0, pass_id="P3"))
    (root / "_eval_log.jsonl").write_text(json.dumps(rec) + "\n")
    return rescore_audit(root, [str(out)], suites=("DG",))


def test_rescore_audit_gates_on_bf16_path_only(tmp_path):
    assert _audit_world(tmp_path, 0, 50)["passes"] is True     # fp32 P3 flips do not gate
    assert _audit_world(tmp_path, 2, 0)["passes"] is False     # bf16 P2 at 2% fails as before


def test_rescore_audit_reads_sealed_record_of_blinded_runs(tmp_path):
    from hybridlc.analysis.results import rescore_audit

    root = tmp_path / "sealed"
    out = root / "S2T-D384-C12-s101-bbbbbbbb" / "tef1000" / "DG" / "R1_D3_4096.jsonl"
    out.parent.mkdir(parents=True)
    out.write_text("")
    rf = out.with_name(out.stem + "_rescore.json")
    rf.write_text(json.dumps({"P2": dict(n=100, flips=3, flip_rate=0.03, max_abs_dlogp=0.0)}))
    (root / "_eval_log.jsonl").write_text(json.dumps(dict(kind="eval", out=str(out), rescore_file=str(rf))) + "\n")
    assert rescore_audit(root, [str(out)], suites=("DG",))["passes"] is False

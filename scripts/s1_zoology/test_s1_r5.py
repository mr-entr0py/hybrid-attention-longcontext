"""R5 tooling checks: d64 run list, the bf16 GDN wrapper (geometry, state size, kernel dtypes, train step) and the
pre-registered analysis rules on synthetic run files (prereg experiments/repro/R5/prereg.yaml).

    python -m pytest scripts/s1_zoology/test_s1_r5.py -q
"""
from __future__ import annotations

import json
import math
import shutil
import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from zoology_compat import claim_gpu, setup_zoology  # noqa: E402

setup_zoology(micro_batch=None)
import our_mixers as OM  # noqa: E402
import r5_analysis as RA  # noqa: E402
from test_s1_our_mixers import _lm, _mixer, _n  # noqa: E402

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def test_r5_run_list() -> None:
    cfgs = OM.r5_runs()
    assert [c.run_id for c in cfgs] == [f"{a}-d64-lr1.0e-03-s{s}" for s in (123, 124, 125)
                                         for a in ("gdn_zoology", "our_gdn_bf16")]
    for c in cfgs:
        s = int(c.run_id.rsplit("-s", 1)[1])
        assert c.seed == s and c.data.seed == s and c.max_epochs == 32 and c.model.d_model == 64
        assert abs(c.learning_rate - 1e-3) < 1e-15
        assert c.early_stopping_metric == "valid/accuracy" and c.early_stopping_threshold == 0.99
        seq = c.model.sequence_mixer.kwargs["configs"]
        assert seq[0]["name"].endswith("BaseConv")
        if c.run_id.startswith("gdn_zoology"):
            assert seq[1]["name"] == "zoology.mixers.gated_delta_net.GatedDeltaNet"
            assert seq[1]["kwargs"]["num_heads"] == 2 and seq[1]["kwargs"]["use_gate"] is False
            assert seq[1]["kwargs"]["conv_size"] == 4
        else:
            assert seq[1]["name"] == "our_mixers.OurGdnBf16Zoology"
    assert len(OM.r5_runs(include_o5=True)) == 10


@cuda
def test_bf16_gdn_geometry_state_and_kernel_dtypes() -> None:
    claim_gpu()
    import hybridlc.models.linear as hlin
    import zoology.mixers.gated_delta_net as zgdn

    zc, oc = OM.r5_runs()[:2]
    zlm, olm = _lm(zc.model).cuda(), _lm(oc.model).cuda()
    zm, om = _mixer(zlm), _mixer(olm)
    assert om.autocast_bf16 and (om.inner.num_heads, om.inner.head_k_dim, om.inner.head_v_dim) == (2, 32, 64)
    assert (zm.num_heads, zm.head_k_dim, zm.head_v_dim) == (2, 32, 64)
    assert om.state_size(1024) == zm.state_size(1024) == 2 * 32 * 64
    assert _n(om) + 2 == _n(zm)  # Zoology's unused D parameter (num_heads scalars), as at d128 in R4
    seen: dict[str, dict] = {}

    def wrap(mod, key):
        orig = mod.chunk_gated_delta_rule

        def w(*a, **k):
            seen.setdefault(key, {n: k[n].dtype for n in ("q", "k", "v", "g", "beta")})
            return orig(*a, **k)
        mod.chunk_gated_delta_rule = w
        return orig

    o1, o2 = wrap(zgdn, "z"), wrap(hlin, "o")
    try:
        from zoology.data.multiquery_ar import MQARConfig

        seg = MQARConfig(vocab_size=8192, input_seq_len=256, num_examples=8, num_kv_pairs=16).build(seed=0)
        x, y = seg.inputs.cuda(), seg.labels.cuda()
        for lm in (zlm, olm):
            opt = torch.optim.AdamW(lm.parameters(), lr=1e-3, weight_decay=0.1)
            losses = []
            for _ in range(3):
                logits = lm(x)
                assert logits.dtype == torch.float32
                loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), y.flatten(), ignore_index=-100)
                opt.zero_grad()
                loss.backward()
                assert all(p.dtype == torch.float32 for p in lm.parameters())
                assert all(p.grad is None or torch.isfinite(p.grad).all() for p in lm.parameters())
                opt.step()
                losses.append(float(loss))
            assert all(math.isfinite(v) for v in losses) and losses[-1] < losses[0]
    finally:
        zgdn.chunk_gated_delta_rule, hlin.chunk_gated_delta_rule = o1, o2
    assert all(v == torch.bfloat16 for v in seen["z"].values())             # Zoology casts q,k,v,beta,g
    assert seen["o"]["q"] == seen["o"]["k"] == seen["o"]["v"] == torch.bfloat16  # ours under autocast
    assert seen["o"]["beta"] == torch.bfloat16


def _fake_run(d: Path, arm: str, seed: int, c1024: float, overall: float) -> None:
    pc = {c: 1.0 for c in RA.CONFIGS}
    pc["T1024_kv256"] = c1024
    pc["T512_kv128"] = 0.99
    r = {"final_overall_acc": overall, "final_per_config_acc": pc, "epochs_run": 32, "early_stopped": False,
         "collapsed": False, "resumed_at_epochs": [], "wall_time_s": 1.0, "model_info": {"state_size": 1, "num_parameters": 1}}
    (d / "runs" / f"{RA.rid(arm, seed)}.json").write_text(json.dumps(r))


def _analyse(tmp_path: Path, monkeypatch, zoo: list[float], ours: list[float]) -> dict:
    (tmp_path / "runs").mkdir()
    shutil.copy(RA.R5 / "prereg.yaml", tmp_path / "prereg.yaml")
    for s, z, o in zip(RA.SEEDS, zoo, ours):
        _fake_run(tmp_path, "gdn_zoology", s, z, 0.9)
        _fake_run(tmp_path, "our_gdn_bf16", s, o, 0.9)
    monkeypatch.setattr(RA, "R5", tmp_path)
    monkeypatch.setattr(RA, "R4_RUNS", tmp_path / "none")
    RA.main()
    return json.loads((tmp_path / "r5_results.json").read_text())


def test_analysis_pass(tmp_path, monkeypatch) -> None:
    r = _analyse(tmp_path, monkeypatch, [0.60, 0.62, 0.61], [0.59, 0.60, 0.62])
    assert r["rule1_regime"]["primary_cell"] == "T1024_kv256"
    assert r["rule2_phenomenon"]["pass"] and r["rule3_comparison"]["pass"] and r["verdict"] == "PASS"
    assert r["rule3_comparison"]["threshold_term"] == "floor 0.03"


def test_analysis_fail_comparison(tmp_path, monkeypatch) -> None:
    r = _analyse(tmp_path, monkeypatch, [0.60, 0.61, 0.62], [0.50, 0.51, 0.52])
    assert r["verdict"].startswith("FAIL (rule 3")
    assert abs(r["rule3_comparison"]["delta"] + 0.10) < 1e-12


def test_analysis_t_term_widens(tmp_path, monkeypatch) -> None:
    # sd 0.05 per arm -> SE_W = 0.05*sqrt(2/3) = 0.0408 -> 2.78*SE = 0.1135 > 0.03; |Δ| = 0.10 passes
    r = _analyse(tmp_path, monkeypatch, [0.55, 0.60, 0.65], [0.45, 0.50, 0.55])
    c = r["rule3_comparison"]
    assert c["threshold_term"] == "2.78*SE_W" and abs(c["threshold"] - 2.78 * 0.05 * math.sqrt(2 / 3)) < 1e-9
    assert c["pass"]


def test_analysis_bimodal(tmp_path, monkeypatch) -> None:
    r = _analyse(tmp_path, monkeypatch, [0.30, 0.61, 0.62], [0.60, 0.61, 0.62])
    assert r["rule4_bimodality"]["fired"] and r["verdict"].startswith("INCONCLUSIVE")


def test_analysis_regime_high_and_phenomenon(tmp_path, monkeypatch) -> None:
    r = _analyse(tmp_path, monkeypatch, [0.96, 0.97, 0.98], [0.96, 0.97, 0.98])
    # mean 0.97 > 0.95 -> primary = lowest Zoology cell = 1024/256 itself (0.97 < 512/128's 0.99), labelled low power
    assert r["rule1_regime"]["low_power"] and r["rule1_regime"]["primary_cell"] == "T1024_kv256"
    assert not r["rule2_phenomenon"]["pass"] and r["verdict"].startswith("FAIL (rule 2")

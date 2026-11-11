"""T21: FEAS-384 run YAMLs (configs/experiments/feas/*.yaml) match the protocol (PI rulings §10 item 10).

Every value of item 10 is asserted on the *resolved* run config (``load_run_config``), not on the YAML text:
k in {0, 12}, seeds 101/102, eta 2.0e-3 with one restart at 1.4e-3 (D034), Stage-B branches 30M/90M, the SSMax
arm (k = 12, seed 101, 30M branch only), TEF 1,000 with answer_norm block (D063) at 90/10, the milestones, and
production precision. The files are generated (``genconfigs --check`` must pass)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from hybridlc.data.tef import TEF_SYNTHETIC_SHARE
from hybridlc.training.config import load_run_config
from hybridlc.training.genconfigs import FEAS_DIR, feas_design, feas_run_docs

REPO = Path(__file__).resolve().parents[1]
FEAS = REPO / FEAS_DIR


def _cfgs():
    return {p.name: load_run_config(p) for p in sorted(FEAS.glob("*.yaml"))}


def test_genconfigs_check_passes():
    r = subprocess.run([sys.executable, "-m", "hybridlc.training.genconfigs", "--check"], cwd=REPO,
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout + r.stderr
    assert sorted(p.name for p in FEAS.glob("*.yaml")) == sorted(feas_run_docs())


def test_design_cells():
    cfgs = _cfgs()
    assert len(cfgs) == 23
    tef = {n: c for n, c in cfgs.items() if c.stage == "TEF"}
    cells = {(c.model_meta["k"], c.config_name, c.seed_index) for c in tef.values()}
    assert {(k, s) for k, _, s in cells} == {(0, 101), (0, 102), (12, 101), (12, 102)}
    plain = [c for c in tef.values() if c.config_name in ("C0", "C12")]
    assert len(plain) == 8                                  # k x seed x {30M, 90M}
    ss = [c for c in tef.values() if c.config_name == "C12ssmax"]
    assert len(ss) == 1 and ss[0].seed_index == 101 and ss[0].model["ssmax"] is True and ss[0].model_meta["k"] == 12
    assert all(not c.model.get("ssmax") for c in plain)
    assert {r["yaml"].rsplit("/", 1)[1] for r in feas_design()} == set(tef)
    for c in cfgs.values():
        assert c.exp_stage == "S2" and c.scale == "D384" and c.branch == "main"


def test_lr_restart_and_stage_lengths():
    for name, c in _cfgs().items():
        if c.stage == "A":
            assert c.schedule.peak_lr == 2.0e-3 and c.schedule.kind == "warmup_constant"
            assert c.schedule.warmup_steps == 500 and c.schedule.steps == 2289       # 0.3B tokens (plan S2)
            assert c.gdiv.fallback_lr == 1.4e-3 and c.gdiv.max_restarts == 1         # D034
            assert c.parent_run_id is None
        elif c.stage == "B":
            assert c.schedule.peak_lr == 2.0e-3 and c.schedule.kind == "linear_decay"
            want = {"_B030.yaml": 229, "_B090.yaml": 687}[name[-10:]]
            assert c.schedule.steps == want, name                                    # 30M / 90M tokens
            assert c.data.tokens_per_step * c.schedule.steps in (30015488, 90046464)
            assert c.parent_run_id and c.parent_run_id.startswith("S2-D384-")
        else:
            assert c.stage == "TEF"
            assert c.schedule.steps == 1000 and c.schedule.peak_lr == 3.0e-4 and c.schedule.warmup_steps == 50
            assert c.parent_run_id and c.parent_run_id.startswith("S2B-D384-")


def test_tef_settings():
    for c in _cfgs().values():
        if c.stage != "TEF":
            continue
        assert c.data.stream == "tef" and c.data.kwargs.get("answer_norm") == "block"              # D063 / DEV-04
        assert c.data.kwargs.get("synthetic_share", TEF_SYNTHETIC_SHARE) == 0.9                    # 90/10 default
        assert not c.data.kwargs.get("answer_only", False)
        assert c.checkpoint.milestone_steps == [250, 500, 1000]                                    # TEF 250/500/1000
        assert c.validation.stage == "LM2" and c.validation.at_start is True and c.validation.every == 250


def test_milestones_and_precision():
    for c in _cfgs().values():
        assert c.runtime.precision == "bf16" and c.runtime.audit is False                          # production
        assert c.runtime.mem_fraction == 0.44
        if c.stage == "A":
            assert c.checkpoint.milestone_steps == [382, 763, 1526]   # 50M/100M/200M tokens (400M > 0.3B)
        assert c.checkpoint.save_final is True


@pytest.mark.parametrize("seed", [101, 102])
def test_parent_chain(seed):
    cfgs = _cfgs()
    for cfg in ("C0", "C12"):
        a = cfgs[f"S2_D384_{cfg}_s{seed}_A.yaml"]
        for br in ("B030", "B090"):
            b = cfgs[f"S2_D384_{cfg}_s{seed}_{br}.yaml"]
            t = cfgs[f"S2_D384_{cfg}_s{seed}_{br}_T.yaml"]
            assert b.parent_run_id == a.run_id and t.parent_run_id == b.run_id
            assert a.model == b.model == t.model

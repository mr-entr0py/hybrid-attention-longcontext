"""VT-13 (determinism and resume, protocol §16) plus the training-infrastructure unit tests
(§7 optimiser/schedules, §13.3 G-div / G-OOM, §18 checkpoints / registry / runner / run_id).

Tiers:
* default (CPU, seconds): tiny fake model + fake stream through the *real* trainer, in separate
  processes, with kill -9 mid-accumulation and during a checkpoint write (bitwise comparisons);
* ``gpu``: the same on CUDA in bf16 audit mode, and a 20-step smoke test of the real
  ``HybridLM`` + ``chunked_ce`` (real data if ``data/tokens`` is built, else the fake stream);
* ``gpu`` + ``slow`` + ``HYBRIDLC_VT13_FULL=1``: the protocol VT-13 audit — real model, real Stage-A
  data, k ∈ {0, 3, 12} × 200 steps × 2 processes, kill -9 at step 137 (mid-accumulation) and during
  the step-150 checkpoint write, resume; bitwise loss traces and parameter SHA-256. With
  ``HYBRIDLC_VT13_PRODUCTION=1`` also the production-mode check (|Δloss| < 0.005 at resume + 500).
"""
from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
TRAIN = REPO / "scripts" / "train.py"
RUNNER = REPO / "scripts" / "runner.py"
BASE = REPO / "configs" / "training" / "base.yaml"

from hybridlc.training import checkpoint as ckpt  # noqa: E402
from hybridlc.training.config import (GDivConfig, OptimConfig, ScheduleConfig, load_run_config,  # noqa: E402
                                      load_with_inherits)
from hybridlc.training.gdiv import DivergenceError, GDivMonitor  # noqa: E402
from hybridlc.training.registry import Registry  # noqa: E402
from hybridlc.training.schedules import lr_at  # noqa: E402
from hybridlc.training.trainer import read_jsonl_last_per_step  # noqa: E402

FAKE_IMPL = textwrap.dedent("""\
    impl:
      model_factory: "hybridlc.training.fakes:build_fake_model"
      stage_stream: "hybridlc.training.fakes:FakeStageStream"
      tef_stream: "hybridlc.training.fakes:FakeTEFStream"
      loss_fn: "hybridlc.training.fakes:fake_chunked_ce"
""")


# ----------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------
def _env(extra: dict | None = None) -> dict:
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "PYTHONHASHSEED": "0"}
    env.pop("HYBRIDLC_FAULT", None)
    env.update(extra or {})
    return env


def write_tiny(tmp: Path, *, device: str = "cpu", precision: str = "fp32", steps: int = 40, every: int = 10,
               accum: int = 3, stage: str = "A", kind: str = "warmup_constant", extra: str = "",
               parent: str | None = None, name: str = "run", lr: float = 3e-3, seed: int = 900,
               pattern: str = "-A-A", audit: bool = True) -> Path:
    """Tiny model + recipe + run YAML using the contract fakes (same trainer code path)."""
    model = tmp / "model.yaml"
    if not model.exists():
        model.write_text(textwrap.dedent(f"""\
            name: tiny
            scale: tiny
            k: {pattern.count('A')}
            fields: {{d_model: 32, d_ff: 64, n_layers: {len(pattern)}, vocab_size: 256, pattern: "{pattern}",
                      attn_head_dim: 16, rope_theta: 500000.0, nope: false, ssmax: false, init_std: 0.02,
                      tie_embeddings: true, norm_eps: 1.0e-5}}
            """))
    stream = "tef" if stage == "TEF" else "stage"
    warm = "warmup_steps: 5, " if kind != "linear_decay" else ""
    peak = "3.0e-4" if stage == "TEF" else "run"
    recipe = tmp / f"{name}.train.yaml"
    recipe.write_text(textwrap.dedent(f"""\
        inherits: {BASE}
        stage: {stage}
        schedule: {{kind: {kind}, {warm}peak_lr: {peak}, steps: {steps}}}
        data: {{stream: {stream}, stage: {stage}, seq_len: 64, micro_batch: 2, accum: {accum},
                kwargs: {{vocab_size: 256}}}}
        runtime: {{device: {device}, precision: {precision}, audit: {str(audit).lower()}, cpu_threads: 1}}
        checkpoint: {{every: {every}}}
        validation: {{every: 20, n_batches: 2, seq_len: 64, kwargs: {{vocab_size: 256}}}}
        """) + FAKE_IMPL + extra)
    run = tmp / f"{name}.yaml"
    body = f"exp_stage: T\nmodel: {model}\ntraining: {recipe}\nseed_index: {seed}\nlr: {lr}\n"
    if parent:
        body += f"parent: {parent}\n"
    run.write_text(body)
    return run


def train_proc(run_yaml: Path, root: Path, fault: dict | None = None, extra_args: list[str] | None = None,
               timeout: int = 600) -> subprocess.CompletedProcess:
    cmd = [sys.executable, str(TRAIN), "--config", str(run_yaml), "--allow-dirty",
           "--runs-root", str(root / "runs"), "--ckpt-root", str(root / "ckpt"),
           "--registry", str(root / "registry.jsonl"), *(extra_args or [])]
    extra = {"HYBRIDLC_FAULT": json.dumps(fault)} if fault else {}
    return subprocess.run(cmd, env=_env(extra), capture_output=True, text=True, timeout=timeout, cwd=str(REPO))


def run_dir(root: Path) -> Path:
    (d,) = [p for p in (root / "runs").iterdir() if p.is_dir() and not p.name.startswith("_")]
    return d


def trace_key(root: Path) -> list[tuple]:
    rows = read_jsonl_last_per_step(run_dir(root) / "trace.jsonl")
    return [(r["step"], r["loss_hex"], r["gnorm_hex"], r["batch_hash"]) for r in rows]


def summary(root: Path) -> dict:
    return json.loads((run_dir(root) / "summary.json").read_text())


def statuses(root: Path) -> list[str]:
    return [r["status"] for r in Registry(root / "registry.jsonl").records()]


# ----------------------------------------------------------------------------------------------
# schedules (protocol §6.3, §7.2, §7.3)
# ----------------------------------------------------------------------------------------------
def test_schedule_stage_a_warmup_constant():
    s = ScheduleConfig(kind="warmup_constant", steps=4578, peak_lr=2e-3, warmup_steps=500)
    assert lr_at(0, s) == pytest.approx(2e-3 / 500)
    assert lr_at(249, s) == pytest.approx(2e-3 * 250 / 500)
    assert lr_at(499, s) == pytest.approx(2e-3)
    assert all(lr_at(t, s) == 2e-3 for t in (500, 1000, 4577))
    assert all(lr_at(t + 1, s) >= lr_at(t, s) for t in range(600))


def test_schedule_stage_b_linear_decay():
    s = ScheduleConfig(kind="linear_decay", steps=229, peak_lr=2e-3)
    assert lr_at(0, s) == pytest.approx(2e-3)
    assert lr_at(228, s) == pytest.approx(2e-3 / 229)
    assert lr_at(229, s) == 0.0
    d = [lr_at(t, s) - lr_at(t + 1, s) for t in range(228)]
    assert max(d) - min(d) < 1e-15  # exactly linear


def test_schedule_lr_sweep_wsd_115_916_229():
    s = ScheduleConfig(kind="wsd", steps=1145, peak_lr=1.4e-3, warmup_steps=115, stable_until=916)
    assert lr_at(0, s) == pytest.approx(1.4e-3 / 115)
    assert lr_at(114, s) == pytest.approx(1.4e-3)
    assert all(lr_at(t, s) == pytest.approx(1.4e-3) for t in (115, 500, 915, 916))
    assert lr_at(1144, s) == pytest.approx(1.4e-3 / 229)
    assert lr_at(1145, s) == 0.0
    assert 1145 - 916 == 229
    assert sum(1 for t in range(1145) if lr_at(t, s) < 1.4e-3 * (1 - 1e-12) and t >= 115) == 228


def test_schedule_tef_rewarm():
    s = ScheduleConfig(kind="warmup_constant", steps=1000, peak_lr=3e-4, warmup_steps=50)
    assert lr_at(0, s) == pytest.approx(3e-4 / 50)
    assert lr_at(49, s) == pytest.approx(3e-4)
    assert lr_at(999, s) == pytest.approx(3e-4)
    s4 = ScheduleConfig(kind="warmup_constant", steps=4000, peak_lr=3e-4, warmup_steps=50, start_step=1000)
    assert lr_at(1000, s4) == pytest.approx(3e-4)  # continuation stays at the constant LR


# ----------------------------------------------------------------------------------------------
# optimiser groups (protocol §7.1)
# ----------------------------------------------------------------------------------------------
def _check_groups(model, expect_decay_substr, expect_nodecay_substr):
    from hybridlc.training.optim import build_optimizer

    opt, summ = build_optimizer(model, OptimConfig(), lr=1e-3)
    names = {id(p): n for n, p in model.named_parameters()}
    groups = {g["group_name"]: g for g in opt.param_groups}
    assert groups["decay"]["weight_decay"] == 0.1 and groups["no_decay"]["weight_decay"] == 0.0
    dn = [names[id(p)] for p in groups["decay"]["params"]]
    nn_ = [names[id(p)] for p in groups["no_decay"]["params"]]
    assert all(p.ndim >= 2 for p in groups["decay"]["params"])
    for s in expect_decay_substr:
        assert any(s in n for n in dn), (s, dn)
        assert not any(s in n for n in nn_), (s, nn_)
    for s in expect_nodecay_substr:
        assert any(s in n for n in nn_), (s, nn_)
        assert not any(s in n for n in dn), (s, dn)
    for p in groups["no_decay"]["params"]:
        assert p.ndim < 2 or getattr(p, "_no_weight_decay", False) or "ssmax" in names[id(p)]
    assert opt.defaults["betas"] == (0.9, 0.95) and opt.defaults["eps"] == 1e-8
    return dn, nn_


def test_weight_decay_groups_fake():
    from hybridlc.training.fakes import build_fake_model

    m = build_fake_model(dict(d_model=32, d_ff=64, vocab_size=256, pattern="-A-A", init_std=0.02), 0)
    _check_groups(m, ["embed.weight", "conv.weight", "gate.weight", "qkv.weight"], ["A_log", "q_norm", "n1.weight"])


def test_weight_decay_groups_real_model():
    build = pytest.importorskip("hybridlc.models.build")
    from hybridlc.models.config import ModelConfig

    cfg = ModelConfig.for_scale("D384", "C3", ssmax=True)
    m = build.build_model(cfg, 0)
    dn, nn_ = _check_groups(m, ["conv"], ["A_log", "dt_bias"])
    emb = m.lm_head_weight
    assert any(p is emb for g in [m.parameters()] for p in g)
    from hybridlc.training.optim import param_groups
    groups, _ = param_groups(m, OptimConfig())
    assert any(p is emb for p in groups[0]["params"]), "tied embedding must be decayed"
    # every 1-D tensor (norm gains incl. QK-norm, GDN o_norm, SSMax s) is excluded
    for n, p in m.named_parameters():
        if p.ndim < 2:
            assert n in nn_, n
        if "ssmax" in n.lower():
            assert n in nn_, n
    # the headwise gate W_g (2-D) is decayed
    gates = [n for n, p in m.named_parameters() if p.ndim == 2 and re.search(r"(^|\.)(g_proj|gate|w_g|W_g)", n)]
    assert gates and all(n in dn for n in gates), gates


# ----------------------------------------------------------------------------------------------
# G-div (protocol §13.3)
# ----------------------------------------------------------------------------------------------
def test_gdiv_nan_triggers():
    m = GDivMonitor(GDivConfig())
    m.update(0, 5.0)
    with pytest.raises(DivergenceError):
        m.update(1, float("nan"))


def test_gdiv_patience_50_steps():
    m = GDivMonitor(GDivConfig())
    for t in range(200):
        m.update(t, 5.0)
    # a spike lasting exactly 50 steps does not trigger (the EMA creeps up but stays below loss - 1)
    for t in range(200, 250):
        m.update(t, 9.0)
    assert m.s.above == 50
    m2 = GDivMonitor(GDivConfig(), m.state_dict())
    with pytest.raises(DivergenceError):
        m2.update(250, 9.0)  # the 51st step above EMA + 1.0
    # recovery resets the counter
    m3 = GDivMonitor(GDivConfig())
    for t in range(100):
        m3.update(t, 5.0)
    for t in range(30):
        m3.update(100 + t, 9.0)
    m3.update(130, 5.0)
    assert m3.s.above == 0


# ----------------------------------------------------------------------------------------------
# checkpoints and registry (protocol §18)
# ----------------------------------------------------------------------------------------------
def test_atomic_checkpoint_and_keep_last(tmp_path):
    from hybridlc.training.fakes import build_fake_model
    from hybridlc.training.optim import build_optimizer

    m = build_fake_model(dict(d_model=16, d_ff=32, vocab_size=64, pattern="-A", init_std=0.02), 0)
    opt, _ = build_optimizer(m, OptimConfig(), lr=1e-3)
    meta = {"config_hash": "x"}
    for s in (10, 20, 30):
        ckpt.save_full(tmp_path, s, model=m, optimizer=opt, meta=meta, trainer_state={}, keep_last=2)
    assert [s for s, _ in ckpt.list_step_checkpoints(tmp_path)] == [20, 30]

    class Boom(Exception):
        pass

    def hook(tmp):
        assert tmp.exists() and ".tmp." in tmp.name
        raise Boom

    with pytest.raises(Boom):
        ckpt.save_full(tmp_path, 40, model=m, optimizer=opt, meta=meta, trainer_state={}, fault_hook=hook)
    assert not (tmp_path / "step_0000040.pt").exists()
    assert [s for s, _ in ckpt.list_step_checkpoints(tmp_path)] == [20, 30]  # old set intact
    assert ckpt.cleanup_tmp(tmp_path)
    st = ckpt.load(ckpt.latest_checkpoint(tmp_path))
    assert st["step"] == 30 and all(v.dtype == torch.float32 for v in st["model"].values() if v.is_floating_point())
    p = ckpt.save_milestone(tmp_path, 7, m, meta)
    assert all(v.dtype == torch.bfloat16 for v in ckpt.load(p)["model"].values() if v.is_floating_point())


def test_registry_append_only_state(tmp_path):
    r = Registry(tmp_path / "reg.jsonl")
    r.append("a", "queued", config="x")
    r.append("b", "queued", config="y")
    r.append("a", "running", pid=1)
    r.append("a", "diverged", step=3)        # annotation: state unchanged
    r.append("a", "oom", ckpt_step=0)
    st = r.state()
    assert list(st) == ["a", "b"]
    assert st["a"]["status"] == "oom" and st["a"]["queue"]["config"] == "x"
    assert st["b"]["status"] == "queued"
    n_lines = len((tmp_path / "reg.jsonl").read_text().splitlines())
    r.append("a", "completed")
    assert len((tmp_path / "reg.jsonl").read_text().splitlines()) == n_lines + 1


# ----------------------------------------------------------------------------------------------
# configs and run identity (protocol §3.2, §6.3, §18)
# ----------------------------------------------------------------------------------------------
RUN_ID_RE = re.compile(r"^S\d[BT]?-D(384|512)-[A-Za-z0-9\-]+-s\d+-[0-9a-f]{8}$")


def test_model_configs_up_to_date():
    pytest.importorskip("hybridlc.models.config")
    from hybridlc.training.genconfigs import main

    assert main(["--check"]) == 0


def test_example_run_configs_resolve():
    ex = sorted((REPO / "configs" / "training" / "examples").glob("*.yaml"))
    assert ex
    ids = {}
    for p in ex:
        c = load_run_config(p)
        assert RUN_ID_RE.match(c.run_id), c.run_id
        ids[p.stem] = c
    a, b, t = ids["S2_D384_C0_s101_A"], ids["S2_D384_C0_s101_B030"], ids["S2_D384_C0_s101_B030_T"]
    assert (a.schedule.steps, a.schedule.warmup_steps, a.data.tokens_per_step) == (2289, 500, 131072)
    assert (b.schedule.kind, b.schedule.steps, b.data.seq_len, b.data.accum) == ("linear_decay", 229, 32768, 4)
    assert b.runtime.activation_checkpointing and b.parent_run_id == a.run_id
    assert b.parent_ckpt.endswith(f"{a.run_id}/final.pt")
    assert (t.schedule.peak_lr, t.schedule.warmup_steps, t.schedule.steps, t.data.tokens_per_step) == \
        (3e-4, 50, 1000, 65536)
    lr = ids["S3_D384_C3_lr2.0e-3"]
    assert (lr.schedule.kind, lr.schedule.warmup_steps, lr.schedule.stable_until, lr.schedule.steps) == \
        ("wsd", 115, 916, 1145)
    assert lr.validation.final_n_batches == 1000  # 2,000 x 4K LM-2 sequences
    assert ids["S4_D384_C3_s0_TEF4000"].schedule.start_step == 1000


@pytest.mark.parametrize("scale,stageA_steps,B", [("D384", 4578, {"30M": 229, "60M": 458, "90M": 687}),
                                                  ("D512", 6104, {"100M": 763, "200M": 1526, "300M": 2289})])
def test_every_model_config_composes(tmp_path, scale, stageA_steps, B):
    models = sorted((REPO / "configs" / "model" / scale).glob("*.yaml"))
    assert len(models) >= 30
    seen = set()
    for m in models:
        run = tmp_path / f"{m.stem}.yaml"
        run.write_text(f"exp_stage: S4\nmodel: {m}\ntraining: configs/training/stage_A.yaml\nseed_index: 0\nlr: 2.0e-3\n")
        c = load_run_config(run)
        assert c.schedule.steps == stageA_steps and c.data.tokens_per_step == 131072
        assert RUN_ID_RE.match(c.run_id) and c.run_id.startswith(f"S4-{scale}-{m.stem}-s0-")
        assert c.run_id not in seen
        seen.add(c.run_id)
        assert c.checkpoint.milestone_steps == [382, 763, 1526, 3052]  # 50/100/200/400M tokens
    for opt, steps in B.items():
        run = tmp_path / f"B{opt}.yaml"
        run.write_text(f"exp_stage: S4\nmodel: configs/model/{scale}/C3.yaml\ntraining: configs/training/stage_B.yaml\n"
                       f"seed_index: 0\nlr: 2.0e-3\nsteps_option: '{opt}'\n")
        assert load_run_config(run).schedule.steps == steps


def test_run_id_is_semantic_hash(tmp_path):
    p = tmp_path / "r.yaml"
    p.write_text("exp_stage: S4\nmodel: configs/model/D384/C3.yaml\ntraining: configs/training/stage_A.yaml\n"
                 "seed_index: 0\nlr: 2.0e-3\n")
    c0 = load_run_config(p)
    assert c0.run_id.startswith("S4-D384-C3-s0-") and c0.run_id[-8:] == c0.config_hash()[:8]
    # bookkeeping (checkpoint cadence, logging, memory cap) does not change the identity ...
    assert load_run_config(p, ["checkpoint.every=50", "logging.every=5", "runtime.mem_fraction=0.3"]).run_id == c0.run_id
    # ... while anything semantic does
    assert load_run_config(p, ["optim.weight_decay=0.0"]).run_id != c0.run_id
    assert load_run_config(p, ["runtime.audit=true"]).run_id != c0.run_id


def test_stage_recipes_match_protocol():
    A = load_with_inherits(REPO / "configs/training/stage_A.yaml")
    assert A["optim"]["betas"] == [0.9, 0.95] and A["optim"]["eps"] == 1e-8 and A["optim"]["weight_decay"] == 0.1
    assert A["optim"]["grad_clip"] == 1.0 and A["optim"]["z_loss_coef"] == 1e-4 and A["optim"]["fused"]
    assert A["checkpoint"]["every"] == 250 and A["checkpoint"]["keep_last"] == 2
    assert A["logging"]["every"] == 10 and A["validation"]["every"] == 500
    assert A["gdiv"] == {"ema_decay": 0.99, "margin": 1.0, "patience": 50, "max_restarts": 1, "fallback_lr": None}
    assert A["runtime"]["mem_fraction"] == 0.44 and A["runtime"]["precision"] == "bf16"
    assert not A["runtime"]["activation_checkpointing"]
    B = load_with_inherits(REPO / "configs/training/stage_B.yaml")
    assert B["validation"]["every"] == 100 and B["runtime"]["activation_checkpointing"]
    T = load_with_inherits(REPO / "configs/training/tef.yaml")
    assert T["validation"]["every"] == 250 and T["checkpoint"]["milestone_steps"] == [250, 500, 1000, 2000, 4000]


# ----------------------------------------------------------------------------------------------
# micro-batch weighting == full-batch objective
# ----------------------------------------------------------------------------------------------
def test_accumulation_equals_full_batch_tef_weights():
    from hybridlc.training.fakes import FakeTEFStream, build_fake_model, fake_chunked_ce
    from hybridlc.training.trainer import _step_weights

    torch.manual_seed(0)  # unused by the fakes; guards against accidental global RNG use
    s = FakeTEFStream(3, 10, seq_len=64, micro_batch=1, accum=3)
    m = build_fake_model(dict(d_model=16, d_ff=32, vocab_size=256, pattern="-A", init_std=0.02), 0)
    bs = [s.get(4, i) for i in range(3)]
    W, N = _step_weights(bs, 1)
    acc = sum(fake_chunked_ce(m(b["input_ids"], b["cu_seqlens"], b["position_ids"]), m.lm_head_weight, b["labels"],
                              loss_weights=b["loss_weights"])[0] * (b["_w"] / W) for b in bs)
    # direct: Σ w·CE / Σ w over the whole step
    num = den = 0.0
    for b in bs:
        h = m(b["input_ids"], b["cu_seqlens"], b["position_ids"])[0]
        ce = torch.nn.functional.cross_entropy(h @ m.lm_head_weight.T, b["labels"][0], reduction="none")
        w = b["loss_weights"][0] * (b["labels"][0] != 1)
        num = num + (w * ce).sum()
        den = den + w.sum()
    assert float(acc) == pytest.approx(float(num / den), rel=1e-5)


# ----------------------------------------------------------------------------------------------
# VT-13 with the fakes, CPU, separate processes (bitwise)
# ----------------------------------------------------------------------------------------------
def test_vt13_cpu_two_processes_and_kill9_resume(tmp_path):
    run = write_tiny(tmp_path)
    roots = {k: tmp_path / k for k in ("p1", "p2", "kill_mid_accum", "kill_in_ckpt")}
    assert train_proc(run, roots["p1"]).returncode == 0
    assert train_proc(run, roots["p2"]).returncode == 0
    r = train_proc(run, roots["kill_mid_accum"], fault={"kill_at": [27, 1]})
    assert r.returncode == -9, r.stderr[-2000:]
    assert train_proc(run, roots["kill_mid_accum"]).returncode == 0
    r = train_proc(run, roots["kill_in_ckpt"], fault={"kill_in_ckpt": 30})
    assert r.returncode == -9, r.stderr[-2000:]
    leftovers = list((roots["kill_in_ckpt"] / "ckpt").rglob("*.tmp.*"))
    assert leftovers and not list((roots["kill_in_ckpt"] / "ckpt").rglob("step_0000030.pt"))
    assert train_proc(run, roots["kill_in_ckpt"]).returncode == 0
    assert not list((roots["kill_in_ckpt"] / "ckpt").rglob("*.tmp.*"))

    ref = trace_key(roots["p1"])
    assert len(ref) == 40
    for k in ("p2", "kill_mid_accum", "kill_in_ckpt"):
        assert trace_key(roots[k]) == ref, k
        assert summary(roots[k])["param_sha256"] == summary(roots["p1"])["param_sha256"], k
    # the killed runs really resumed from a checkpoint (step 20), not from scratch
    for k in ("kill_mid_accum", "kill_in_ckpt"):
        runs = [x for x in Registry(roots[k] / "registry.jsonl").records() if x["status"] == "running"]
        assert len(runs) == 2 and runs[1]["start_step"] == 20 and "step_0000020" in runs[1]["resumed_from"]
    # metrics every 10 steps with the runbook fields
    rows = read_jsonl_last_per_step(run_dir(roots["p1"]) / "metrics.jsonl", kind="train")
    assert [x["step"] for x in rows] == [9, 19, 29, 39]
    for f in ("loss", "lr", "grad_norm", "z_loss", "tok_s", "batch_hash", "attn_max_logit", "attn_entropy"):
        assert rows[-1][f] is not None, f
    full = ckpt.load(next((roots["p1"] / "ckpt").rglob("final.pt")))
    for f in ("model", "optimizer", "rng", "step", "config_hash", "git_commit", "triton_cache_hash", "trainer_state"):
        assert f in full, f
    assert list((roots["p1"] / "ckpt").rglob("milestones/*final_bf16.pt"))
    assert statuses(roots["p1"]) == ["running", "completed"]


def test_gdiv_restart_and_second_divergence_fails(tmp_path):
    run = write_tiny(tmp_path)
    ref, a = tmp_path / "ref", tmp_path / "nan"
    assert train_proc(run, ref).returncode == 0
    r = train_proc(run, a, fault={"nan_at": 25})
    assert r.returncode == 0, r.stderr[-2000:]
    st = statuses(a)
    assert st == ["running", "diverged", "restarted", "running", "completed"], st
    recs = list(Registry(a / "registry.jsonl").records())
    assert recs[3]["start_step"] == 20  # restarted from the last checkpoint
    # the injected NaN aborted the step before the update, so the replayed trajectory is unchanged
    assert trace_key(a) == trace_key(ref)
    # max_restarts = 0 -> the first divergence is final (exit 76, failed)
    run2 = write_tiny(tmp_path, name="run2", extra="gdiv: {max_restarts: 0}\n")
    b = tmp_path / "nan2"
    r = train_proc(run2, b, fault={"nan_at": 12})
    assert r.returncode == 76, r.stderr[-2000:]
    assert statuses(b)[-1] == "failed"


def test_gdiv_fallback_lr_restarts_from_scratch(tmp_path):
    run = write_tiny(tmp_path, extra="gdiv: {fallback_lr: 1.0e-3}\n", steps=30)
    a = tmp_path / "fb"
    r = train_proc(run, a, fault={"nan_at": 25})
    assert r.returncode == 0, r.stderr[-2000:]
    recs = list(Registry(a / "registry.jsonl").records())
    rs = [x for x in recs if x["status"] == "restarted"][0]
    assert rs["mode"] == "from_scratch" and rs["lr"] == 1e-3
    runs = [x for x in recs if x["status"] == "running"]
    assert runs[-1]["start_step"] == 0 and runs[-1]["peak_lr"] == 1e-3
    assert list(a.glob("ckpt/*.diverged1"))


def test_oom_exit_code_and_resume(tmp_path):
    run = write_tiny(tmp_path)
    ref, a = tmp_path / "ref", tmp_path / "oom"
    assert train_proc(run, ref).returncode == 0
    r = train_proc(run, a, fault={"oom_at": 15})
    assert r.returncode == 75, r.stderr[-2000:]
    oom = [x for x in Registry(a / "registry.jsonl").records() if x["status"] == "oom"]
    assert oom and oom[0]["ckpt_step"] == 10
    assert train_proc(run, a).returncode == 0
    assert trace_key(a) == trace_key(ref)


def test_branch_carries_optimizer_and_tef_weights(tmp_path):
    runA = write_tiny(tmp_path, steps=20, name="A")
    runB = write_tiny(tmp_path, steps=10, name="B", stage="B", kind="linear_decay", parent=str(runA))
    runT = write_tiny(tmp_path, steps=10, name="T", stage="TEF", parent=str(runB), accum=2)
    root = tmp_path / "chain"
    r = train_proc(runB, root)
    assert r.returncode == 77 and "parent checkpoint" in r.stderr  # parent not trained yet -> refused
    for run in (runA, runB, runT):
        r = train_proc(run, root)
        assert r.returncode == 0, r.stderr[-3000:]
    cA, cB, cT = (load_run_config(x, paths=_paths(root)) for x in (runA, runB, runT))
    fa, fb, ft = (ckpt.load(c.ckpt_dir / "final.pt") for c in (cA, cB, cT))
    steps_of = lambda st: {int(v["step"]) for v in st["optimizer"]["state"].values()}  # noqa: E731
    assert steps_of(fa) == {20} and steps_of(fb) == {30} and steps_of(ft) == {40}  # AdamW state carried
    assert fb["trainer_state"]["parent"]["run_id"] == cA.run_id
    tr = read_jsonl_last_per_step(cB.run_dir / "trace.jsonl")
    assert tr[0]["lr"] == pytest.approx(3e-3) and tr[-1]["lr"] == pytest.approx(3e-4)  # decay from eta*
    trT = read_jsonl_last_per_step(cT.run_dir / "trace.jsonl")
    assert trT[0]["lr"] == pytest.approx(3e-4 / 5) and trT[-1]["lr"] == pytest.approx(3e-4)


def _paths(root: Path):
    from hybridlc.training.config import PathsConfig

    return PathsConfig(runs_root=str(root / "runs"), ckpt_root=str(root / "ckpt"), registry=str(root / "registry.jsonl"))


def test_dirty_tree_refused(tmp_path, monkeypatch):
    from hybridlc.training import trainer as T

    cfg = load_run_config(write_tiny(tmp_path), paths=_paths(tmp_path))
    monkeypatch.setattr(T, "git_dirty_paths", lambda: [" M src/x.py"])
    with pytest.raises(T.RefusedError):
        T.train(cfg, allow_dirty=False)
    assert statuses(tmp_path)[-1] == "failed"


def test_runner_queue_dependency_and_monitor(tmp_path):
    runA = write_tiny(tmp_path, steps=20, name="A")
    runB = write_tiny(tmp_path, steps=10, name="B", stage="B", kind="linear_decay", parent=str(runA))
    common = ["--registry", str(tmp_path / "registry.jsonl"), "--runs-root", str(tmp_path / "runs"),
              "--ckpt-root", str(tmp_path / "ckpt"), "--min-free-gb", "1", "--monitor-interval", "1"]
    for run in (runB, runA):  # child first: the runner must wait for the branch point
        r = subprocess.run([sys.executable, str(RUNNER), *common, "enqueue", str(run), "--allow-dirty"],
                           env=_env(), capture_output=True, text=True, timeout=120)
        assert r.returncode == 0, r.stderr
    r = subprocess.run([sys.executable, str(RUNNER), *common, "run"], env=_env(), capture_output=True, text=True,
                       timeout=600)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    reg = Registry(tmp_path / "registry.jsonl")
    st = reg.state()
    assert [v["status"] for v in st.values()] == ["completed", "completed"]
    order = [x["run_id"] for x in reg.records() if x["status"] == "running"]
    cA, cB = (load_run_config(x, paths=_paths(tmp_path)) for x in (runA, runB))
    assert order == [cA.run_id, cB.run_id]
    mon = [x for x in reg.records() if x["status"] == "monitor"]
    assert len(mon) == 2 and all("contended_frac" in x for x in mon)
    assert (cA.run_dir / "gpu.jsonl").exists() and (cA.run_dir / "train.log").exists()


def test_runner_marks_stale_running_interrupted_and_resumes(tmp_path):
    run = write_tiny(tmp_path, steps=20)
    common = ["--registry", str(tmp_path / "registry.jsonl"), "--runs-root", str(tmp_path / "runs"),
              "--ckpt-root", str(tmp_path / "ckpt"), "--min-free-gb", "1", "--no-monitor"]
    subprocess.run([sys.executable, str(RUNNER), *common, "enqueue", str(run), "--allow-dirty"], env=_env(),
                   check=True, capture_output=True, timeout=120)
    # a trainer killed with the machine: leaves a 'running' record with a dead pid and a checkpoint
    r = train_proc(run, tmp_path, fault={"kill_at": [15, 0]})
    assert r.returncode == -9
    r = subprocess.run([sys.executable, str(RUNNER), *common, "run"], env=_env(), capture_output=True, text=True,
                       timeout=600)
    assert r.returncode == 0, r.stdout[-3000:]
    st = [x["status"] for x in Registry(tmp_path / "registry.jsonl").records()]
    assert st == ["queued", "running", "interrupted", "running", "completed"], st


# ----------------------------------------------------------------------------------------------
# GPU tiers
# ----------------------------------------------------------------------------------------------
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")


@pytest.mark.gpu
@needs_cuda
def test_vt13_gpu_bf16_audit_fakes(tmp_path):
    run = write_tiny(tmp_path, device="cuda", precision="bf16", steps=30)
    a, b = tmp_path / "a", tmp_path / "b"
    assert train_proc(run, a).returncode == 0
    r = train_proc(run, b, fault={"kill_at": [17, 2]})
    assert r.returncode == -9
    assert train_proc(run, b).returncode == 0
    assert trace_key(a) == trace_key(b)
    assert summary(a)["param_sha256"] == summary(b)["param_sha256"]


def _real_available() -> tuple[bool, bool]:
    try:
        import hybridlc.models.build  # noqa: F401
        import hybridlc.models.loss  # noqa: F401
    except ImportError:
        return False, False
    data = (REPO / "data" / "tokens" / "fwe" / "train.bin").exists() and \
        (REPO / "data" / "tokens" / "pg19" / "train.bin").exists()
    return True, data


@pytest.mark.gpu
@needs_cuda
def test_real_model_smoke_20_steps(tmp_path):
    """20 steps of the real HybridLM (D384 C3) + chunked_ce in bf16 under the 0.44 memory cap; real
    Stage-A data when the token stores exist, else the contract fake stream at V = 16,384."""
    model_ok, data_ok = _real_available()
    if not model_ok:
        pytest.skip("hybridlc.models.build / loss not available")
    over = ["-o", "data.micro_batch=1", "-o", "data.accum=2", "-o", "checkpoint.every=10", "-o", "logging.every=5", "-o", "validation.every=10",
            "-o", "validation.n_batches=2"]
    if not data_ok:
        over += ["-o", "impl.stage_stream=hybridlc.training.fakes:FakeStageStream",
                 "-o", "data.kwargs={vocab_size: 16384, mean_doc: 600}"]
    run = REPO / "configs" / "training" / "examples" / "smoke_D384_C3.yaml"
    r = train_proc(run, tmp_path, extra_args=over, timeout=900)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    rows = read_jsonl_last_per_step(run_dir(tmp_path) / "metrics.jsonl", kind="train")
    assert len(rows) == 4 and all(math.isfinite(x["loss"]) for x in rows)
    assert rows[-1]["loss"] < rows[0]["loss"]
    assert rows[-1]["attn_max_logit"] is not None
    assert max(x["mem_run_peak"] for x in rows) < 4 * 2**30
    val = read_jsonl_last_per_step(run_dir(tmp_path) / "metrics.jsonl", kind="val")
    assert val and all(math.isfinite(v["val_loss"]) for v in val)


VT13_FULL = os.environ.get("HYBRIDLC_VT13_FULL") == "1"


@pytest.mark.gpu
@pytest.mark.slow
@needs_cuda
@pytest.mark.skipif(not VT13_FULL, reason="set HYBRIDLC_VT13_FULL=1 for the protocol VT-13 audit (~1 GPU-h)")
@pytest.mark.parametrize("cfg_name,seed", [("C0", 900), ("C3", 901), ("C12", 902)])
def test_vt13_protocol_audit(tmp_path, cfg_name, seed):
    model_ok, data_ok = _real_available()
    if not (model_ok and data_ok):
        pytest.skip("real model and data/tokens required")
    steps = int(os.environ.get("HYBRIDLC_VT13_STEPS", "200"))
    run = tmp_path / "run.yaml"
    run.write_text(f"exp_stage: S0\nmodel: configs/model/D384/{cfg_name}.yaml\n"
                   f"training: configs/training/vt13_audit.yaml\nseed_index: {seed}\nlr: 2.0e-3\n")
    ov = ["-o", f"schedule.steps={steps}"]
    # protocol points: kill at step 137 (mid-accumulation) and during the step-150 checkpoint write
    # (checkpoint every 50). A reduced run (HYBRIDLC_VT13_STEPS < 200) scales them down.
    kill_step, ckpt_kill = 137, 150
    if steps < 200:
        ov += ["-o", "checkpoint.every=20"]
        kill_step, ckpt_kill = int(steps * 0.685), 40
    roots = {k: tmp_path / k for k in ("p1", "p2", "kill137", "killckpt")}
    for k in ("p1", "p2"):
        r = train_proc(run, roots[k], extra_args=ov, timeout=7200)
        assert r.returncode == 0, r.stderr[-3000:]
    r = train_proc(run, roots["kill137"], fault={"kill_at": [kill_step, 7]}, extra_args=ov, timeout=7200)
    assert r.returncode == -9
    assert train_proc(run, roots["kill137"], extra_args=ov, timeout=7200).returncode == 0
    r = train_proc(run, roots["killckpt"], fault={"kill_in_ckpt": ckpt_kill}, extra_args=ov, timeout=7200)
    assert r.returncode == -9
    assert train_proc(run, roots["killckpt"], extra_args=ov, timeout=7200).returncode == 0
    ref = trace_key(roots["p1"])
    assert len(ref) == steps
    for k in ("p2", "kill137", "killckpt"):
        assert trace_key(roots[k]) == ref, k
        assert summary(roots[k])["param_sha256"] == summary(roots["p1"])["param_sha256"], k
    for k in ("kill137", "killckpt"):  # really resumed from a checkpoint, not from scratch
        runs = [x for x in Registry(roots[k] / "registry.jsonl").records() if x["status"] == "running"]
        assert len(runs) == 2 and runs[1]["start_step"] > 0, runs


@pytest.mark.gpu
@pytest.mark.slow
@needs_cuda
@pytest.mark.skipif(os.environ.get("HYBRIDLC_VT13_PRODUCTION") != "1", reason="set HYBRIDLC_VT13_PRODUCTION=1")
def test_vt13_production_resume(tmp_path):
    """Production mode (non-deterministic kernels): resume at step 137; |Δloss| < 0.005 at resume + 500
    (compared on the 10-step window mean of the training loss around step 637)."""
    model_ok, data_ok = _real_available()
    if not (model_ok and data_ok):
        pytest.skip("real model and data/tokens required")
    run = tmp_path / "run.yaml"
    run.write_text("exp_stage: S0\nmodel: configs/model/D384/C3.yaml\ntraining: configs/training/stage_A.yaml\n"
                   "seed_index: 901\nlr: 2.0e-3\n")
    ov = ["-o", "schedule.steps=640", "-o", "checkpoint.every=50", "-o", "validation.every=0",
          "-o", "checkpoint.milestone_tokens=[]"]
    a, b = tmp_path / "a", tmp_path / "b"
    assert train_proc(run, a, extra_args=ov, timeout=14400).returncode == 0
    assert train_proc(run, b, fault={"kill_at": [137, 3]}, extra_args=ov, timeout=14400).returncode == -9
    assert train_proc(run, b, extra_args=ov, timeout=14400).returncode == 0
    ta = {r["step"]: r["loss"] for r in read_jsonl_last_per_step(run_dir(a) / "trace.jsonl")}
    tb = {r["step"]: r["loss"] for r in read_jsonl_last_per_step(run_dir(b) / "trace.jsonl")}
    win = range(630, 640)
    d = abs(sum(ta[s] for s in win) / 10 - sum(tb[s] for s in win) / 10)
    assert d < 0.005, d

"""VT-23 (R3d, DEV-01): the production trainer vs an independent ≈ 60-line reference loop.

Pre-registration: ``experiments/repro/R3/prereg.yaml`` (copied from reproduction_plan.md §2.3 R3d).

Trainer side: ``scripts/train.py --config experiments/repro/R3/vt23/run_vt23.yaml`` (D384 C3, seed/data
index 900, Stage-A stream, 2 × 4K micro-batches × 2 accumulation, 50 steps, 10 warm-up steps then
constant at η = 2e-3, fp32, audit mode), with its own runs/ckpt roots and registry.
Reference side: ``scripts/repro/r3_vt23_reference_loop.py`` (written from protocol §7.1–7.2 text;
imports no hybridlc.training module). Both run in separate processes, one after the other.

Pass (frozen): step-1 loss rel diff ≤ 1e-6; steps 2–10 loss and pre-clip grad norm ≤ 1e-5 rel;
steps 11–50 ≤ 1e-3 rel; ≥ 1 step with pre-clip norm > 1.0; final parameters rel L2 ≤ 1e-3;
identical weight-decay group membership by name.

Raw output: ``experiments/repro/R3/vt23/attempt_<ts>/`` and ``vt23_compare.jsonl``.
Run: ``pytest -q -s tests/test_vt23_reference_loop.py``.
"""
from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "repro"))
import r3_common as RC  # noqa: E402

D = REPO / "experiments" / "repro" / "R3" / "vt23"
TOL = {"step1_loss": 1e-6, "steps2_10": 1e-5, "steps11_50": 1e-3, "final_param_rel_l2": 1e-3}


def _hexf(h: str) -> float:
    return struct.unpack(">d", bytes.fromhex(h))[0]


def _rel(a: float, b: float) -> float:
    return abs(a - b) / max(abs(b), 1e-300)


def _env() -> dict:
    # expandable_segments: allocator setting only (no numerical effect); attempts 1-3 hit the 0.44 cap
    # through fragmentation (6.11 GiB allocated + 0.52 GiB reserved-unallocated, 6.80 GiB allowed).
    env = {**os.environ, "PYTHONHASHSEED": "0", "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
           "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    env.pop("HYBRIDLC_FAULT", None)
    return env


def run_trainer(att: Path, tag: str) -> tuple[int, Path, Path, str]:
    py = sys.executable
    args = [py, str(REPO / "scripts" / "train.py"), "--config", str(D / "run_vt23.yaml"),
            "--runs-root", str(att / "runs"), "--ckpt-root", str(att / "ckpt"), "--registry", str(att / "registry.jsonl"),
            "--allow-dirty", "--no-resume", "--tag", tag]
    rid = subprocess.run(args + ["--print-run-id"], capture_output=True, text=True, cwd=REPO, env=_env()).stdout.strip()
    with open(att / "trainer.log", "w") as f:
        rc = subprocess.run(args, stdout=f, stderr=subprocess.STDOUT, cwd=REPO, env=_env()).returncode
    return rc, att / "runs" / rid, att / "ckpt" / rid, rid


def run_reference(att: Path) -> int:
    with open(att / "reference.log", "w") as f:
        return subprocess.run([sys.executable, str(REPO / "scripts" / "repro" / "r3_vt23_reference_loop.py"),
                               "--out", str(att / "ref")], stdout=f, stderr=subprocess.STDOUT, cwd=REPO,
                              env=_env()).returncode


def compare(run_dir: Path, ckpt_dir: Path, ref_dir: Path) -> dict:
    tr = {}
    for line in open(run_dir / "trace.jsonl"):
        r = json.loads(line)
        tr[r["step"]] = r                         # last record per step wins (resume rule)
    ref = {json.loads(l)["step"]: json.loads(l) for l in open(ref_dir / "ref_trace.jsonl")}
    start = next(json.loads(l) for l in open(run_dir / "metrics.jsonl") if json.loads(l).get("kind") == "start")
    steps = []
    for s in range(50):
        t, r = tr.get(s), ref.get(s)
        if t is None or r is None:
            steps.append({"step": s + 1, "missing": True})
            continue
        tl, tg = _hexf(t["loss_hex"]), _hexf(t["gnorm_hex"])
        steps.append({"step": s + 1, "loss_trainer": tl, "loss_ref": r["loss"], "loss_rel": _rel(tl, r["loss"]),
                      "gnorm_trainer": tg, "gnorm_ref": r["gnorm"], "gnorm_rel": _rel(tg, r["gnorm"]),
                      "z_trainer": t["z"], "z_ref": r["z"], "z_rel": _rel(t["z"], r["z"]),
                      "lr_trainer": t["lr"], "lr_ref": r["lr"], "batch_hash_trainer": t.get("batch_hash")})
    ck = torch.load(ckpt_dir / "final.pt", map_location="cpu", weights_only=False)
    pt = ck["model"]
    pr = torch.load(ref_dir / "ref_final.pt", map_location="cpu")
    keys = sorted(set(pt) & set(pr))
    num = sum(float((pt[k].double() - pr[k].double()).pow(2).sum()) for k in keys if pt[k].is_floating_point())
    den = sum(float(pr[k].double().pow(2).sum()) for k in keys if pr[k].is_floating_point())
    per_tensor = {k: RC.rel_rms(pt[k], pr[k]) for k in keys if pt[k].is_floating_point()}
    groups_t = start.get("optimizer", {})
    groups_r = json.loads((ref_dir / "ref_groups.json").read_text())
    res = {"steps": steps, "final_param_rel_l2": (num / den) ** 0.5, "final_param_per_tensor_rel_rms": per_tensor,
           "state_dict_key_mismatch": sorted(set(pt) ^ set(pr)),
           "groups_trainer": groups_t, "groups_ref": groups_r,
           # the trainer's start record summarises groups as {n_decay_tensors, no_decay_names, ...}; with the
           # same parameter set, equal no-decay name sets + equal decay counts <=> identical membership by name
           "groups_identical": (sorted(groups_t.get("no_decay_names", groups_t.get("no_decay", []))) == sorted(groups_r["no_decay"])
                                and groups_t.get("n_decay_tensors", len(groups_t.get("decay", []))) == len(groups_r["decay"])
                                and set(groups_r["decay"]) | set(groups_r["no_decay"]) == set(pt))}
    ok_steps = [x for x in steps if not x.get("missing")]
    res["n_steps_compared"] = len(ok_steps)
    res["step1_loss_rel"] = steps[0].get("loss_rel")
    res["step1_gnorm_rel_report_only"] = steps[0].get("gnorm_rel")
    res["max_rel_2_10"] = max(max(x["loss_rel"], x["gnorm_rel"]) for x in ok_steps if 2 <= x["step"] <= 10)
    res["max_rel_11_50"] = max(max(x["loss_rel"], x["gnorm_rel"]) for x in ok_steps if 11 <= x["step"] <= 50)
    res["n_clipped_trainer"] = sum(x["gnorm_trainer"] > 1.0 for x in ok_steps)
    res["n_clipped_ref"] = sum(x["gnorm_ref"] > 1.0 for x in ok_steps)
    res["lr_max_abs_diff"] = max(abs(x["lr_trainer"] - x["lr_ref"]) for x in ok_steps)
    checks = {
        "all_50_steps": len(ok_steps) == 50,
        "step1_loss": res["step1_loss_rel"] is not None and res["step1_loss_rel"] <= TOL["step1_loss"],
        "steps2_10": res["max_rel_2_10"] <= TOL["steps2_10"],
        "steps11_50": res["max_rel_11_50"] <= TOL["steps11_50"],
        "clipping_exercised": res["n_clipped_trainer"] >= 1 and res["n_clipped_ref"] >= 1,
        "final_params": res["final_param_rel_l2"] <= TOL["final_param_rel_l2"] and not res["state_dict_key_mismatch"],
        "wd_groups": res["groups_identical"],
    }
    res["checks"] = checks
    res["pass"] = all(checks.values())
    return res


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.xfail(strict=False, raises=AssertionError, reason="FAIL-as-registered; accepted residual RR-VT23; PI ruling D073; criterion unchanged")
def test_vt23_trainer_vs_reference_loop() -> None:
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    try:
        import hybridlc.training.trainer  # noqa: F401
    except Exception as e:  # pragma: no cover
        pytest.skip(f"BLOCKED: trainer not importable ({e!r})")
    ts = time.strftime("%Y%m%dT%H%M%S")
    att = D / f"attempt_{ts}"
    att.mkdir(parents=True, exist_ok=False)
    meta = RC.run_meta("VT-23", {"criteria": TOL, "attempt_dir": str(att.relative_to(REPO))})
    from hybridlc.models.build import build_model
    from hybridlc.models.config import ModelConfig
    from hybridlc.utils.provenance import sha256_tensor

    init_sha = {n: sha256_tensor(p.detach()) for n, p in build_model(ModelConfig.for_scale("D384", "C3"), 900).named_parameters()}
    rc_t, run_dir, ckpt_dir, rid = run_trainer(att, f"vt23x{ts}")
    meta.update(trainer_exit=rc_t, trainer_run_id=rid)
    assert rc_t == 0, f"trainer exit {rc_t}; see {att / 'trainer.log'}"
    rc_r = run_reference(att)
    meta["reference_exit"] = rc_r
    assert rc_r == 0, f"reference exit {rc_r}; see {att / 'reference.log'}"
    ref_meta = json.loads((att / "ref" / "ref_meta.json").read_text())
    meta["init_params_identical_ref_vs_build"] = ref_meta["init_param_sha256"] == init_sha
    res = compare(run_dir, ckpt_dir, att / "ref")
    res["init_params_identical"] = meta["init_params_identical_ref_vs_build"]
    wr = RC.RawWriter("vt23", "vt23_compare", meta)
    for s in res["steps"]:
        wr.write(s)
    summary = {k: v for k, v in res.items() if k not in ("steps", "final_param_per_tensor_rel_rms")}
    summary["worst_param_tensors"] = sorted(res["final_param_per_tensor_rel_rms"].items(), key=lambda kv: -kv[1])[:5]
    summary["verdict"] = "PASS" if res["pass"] else "FAIL"
    wr.close(summary)
    print("VT-23 summary:", json.dumps(summary, default=str)[:3000])
    assert res["init_params_identical"], "initial parameters differ between the reference process and build_model"
    assert res["pass"], res["checks"]

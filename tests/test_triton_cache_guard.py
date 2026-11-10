"""Triton cache separation (PI ruling D069 / PR-12 part 1; TRITON_prefeas.yaml ``use``), CPU with fakes.

* an exp_stage S2 run on the diagnostics cache exits 77 (refused, ``failed`` row with the reason);
* an S2 run on the production dir is allowed; so is an extraction whose trainer-style hash equals the
  recorded pre-FEAS hash (registry note ``pre-FEAS-triton-cache``); any other directory is refused;
* the GPU-lock escape hatch is refused for S2;
* a pytest session reports ``TRITON_CACHE_DIR=.cache/triton-diag``.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tests"))

from test_vt13_training import TRAIN, write_tiny  # noqa: E402

from hybridlc.training.compat import triton_cache_hash  # noqa: E402
from hybridlc.training.registry import Registry  # noqa: E402

PY = sys.executable
DIAG = REPO / ".cache" / "triton-diag"
PROD = REPO / ".cache" / "triton"


def _s2_run(tmp_path: Path, name: str) -> Path:
    run = write_tiny(tmp_path, steps=10, every=10, name=name)
    run.write_text(run.read_text().replace("exp_stage: T", "exp_stage: S2"))
    return run


def _go(run: Path, root: Path, cache: Path, **extra) -> subprocess.CompletedProcess:
    env = {**os.environ, "TRITON_CACHE_DIR": str(cache), "HYBRIDLC_GPUMON": "0", "OMP_NUM_THREADS": "1",
           "HYBRIDLC_JOURNAL_SOURCE": ":empty:", **extra}
    env.pop("HYBRIDLC_FAULT", None)
    return subprocess.run([PY, str(TRAIN), "--config", str(run), "--allow-dirty", "--runs-root", str(root / "runs"),
                           "--ckpt-root", str(root / "ckpt"), "--registry", str(root / "registry.jsonl")],
                          env=env, capture_output=True, text=True, timeout=600, cwd=str(REPO))


def test_s2_on_diag_cache_exits_77(tmp_path):
    r = _go(_s2_run(tmp_path, "diag"), tmp_path / "a", DIAG)
    assert r.returncode == 77, r.stderr[-2000:]
    last = list(Registry(tmp_path / "a" / "registry.jsonl").records())[-1]
    assert last["status"] == "failed" and "diagnostics cache" in last["reason"]


def test_s2_on_production_dir_allowed(tmp_path):
    r = _go(_s2_run(tmp_path, "prod"), tmp_path / "b", PROD)
    assert r.returncode == 0, r.stderr[-2000:]
    run = [x for x in Registry(tmp_path / "b" / "registry.jsonl").records() if x["status"] == "running"][0]
    assert run["triton_cache_check"]["kind"] == "production"


def test_s2_on_matching_extraction_allowed_other_refused(tmp_path):
    ext = tmp_path / "extract"
    ext.mkdir()
    for i in range(3):
        (ext / f"K{i}").mkdir()
        (ext / f"K{i}" / "kernel.cubin").write_bytes(os.urandom(64))
    h = triton_cache_hash(ext)
    root = tmp_path / "c"
    Registry(root / "registry.jsonl").append("pre-FEAS-triton-cache", "note", triton_cache_hash=h,
                                             archive_sha256="0" * 64)
    r = _go(_s2_run(tmp_path, "ext"), root, ext)
    assert r.returncode == 0, r.stderr[-2000:]
    run = [x for x in Registry(root / "registry.jsonl").records() if x["status"] == "running"][0]
    assert run["triton_cache_check"]["kind"] == "pre-FEAS extraction" and run["triton_cache_check"]["hash"] == h
    # a modified extraction (one added file) no longer matches -> refused
    other = tmp_path / "other"
    shutil.copytree(ext, other)
    (other / "K0" / "extra.json").write_text("{}")
    r = _go(_s2_run(tmp_path, "oth"), root, other)
    assert r.returncode == 77 and "hash differs" in r.stderr


def test_non_study_stage_not_checked(tmp_path):
    run = write_tiny(tmp_path, steps=10, every=10, name="t")
    r = _go(run, tmp_path / "d", DIAG)
    assert r.returncode == 0, r.stderr[-2000:]


def test_unlocked_hatch_refused_for_s2(tmp_path):
    r = _go(_s2_run(tmp_path, "unl"), tmp_path / "e", PROD, HYBRIDLC_GPU_UNLOCKED="1")
    assert r.returncode == 77 and "HYBRIDLC_GPU_UNLOCKED" in r.stderr
    # a labelled debug run (not S2..S8) may use it, and writes no row into the given (study) registry
    run = write_tiny(tmp_path, steps=10, every=10, name="dbg")
    r = _go(run, tmp_path / "f", DIAG, HYBRIDLC_GPU_UNLOCKED="1")
    assert r.returncode == 0, r.stderr[-2000:]
    assert not (tmp_path / "f" / "registry.jsonl").exists()
    assert list((tmp_path / "f" / "runs").rglob("debug_registry.jsonl"))


def test_pytest_session_uses_diag_cache():
    assert os.environ["TRITON_CACHE_DIR"] == str(DIAG)
    r = subprocess.run([PY, "-m", "pytest", "--co", "-m", "not gpu", "-p", "no:cacheprovider",
                        str(REPO / "tests" / "test_vt01_params.py")],
                       capture_output=True, text=True, timeout=300, cwd=str(REPO),
                       env={k: v for k, v in os.environ.items() if k != "TRITON_CACHE_DIR"})
    assert f"TRITON_CACHE_DIR={DIAG}" in r.stdout, r.stdout[-1500:]

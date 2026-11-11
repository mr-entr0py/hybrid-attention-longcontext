"""T18 / SUITE-PF outcome record (batch-2 README rule 7): an XPASS of a registered-criterion marker is written to
the outcome file with its measured values and does not fail the session; the 7 markers and 13 relaxed guards
are listed by scripts/repro/list_suite_markers.py."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

INNER = '''
import pytest

REASON = "FAIL-as-registered; accepted residual RR-TEST; PI ruling D000; criterion unchanged"

@pytest.mark.xfail(strict=False, raises=AssertionError, reason=REASON)
def test_registered_unexpectedly_met(record_property):
    record_property("measured", 1.5e-4)
    assert 1.5e-4 < 1e-3

@pytest.mark.xfail(strict=False, raises=AssertionError, reason=REASON)
def test_registered_still_failing(record_property):
    record_property("measured", 3.0e-3)
    assert 3.0e-3 < 1e-3

@pytest.mark.xfail(strict=False, raises=AssertionError, reason=REASON)
def test_registered_crashes():
    raise RuntimeError("environment broken")

def test_ordinary():
    assert True
'''


def _run(tmp_path: Path, body: str) -> tuple[subprocess.CompletedProcess, dict]:
    (tmp_path / "test_inner.py").write_text(body)
    xml = tmp_path / "suite.xml"
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "HYBRIDLC_GPUMON": "0", "PYTHONPATH": str(REPO / "src")}
    env.pop("HYBRIDLC_SUITE_OUTCOMES", None)
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-p", "hybridlc.utils.pytest_gpu",
                        "--rootdir", str(tmp_path), "-c", os.devnull, f"--junitxml={xml}", str(tmp_path / "test_inner.py")],
                       capture_output=True, text=True, env=env, cwd=tmp_path, timeout=300)
    out = json.loads((tmp_path / "suite_outcomes.json").read_text())
    return r, out


def test_xpass_recorded_and_does_not_fail_session(tmp_path):
    r, out = _run(tmp_path, INNER.replace("def test_registered_crashes", "def _unused").replace(
        'raise RuntimeError("environment broken")', "pass"))
    assert r.returncode == 0, r.stdout + r.stderr
    t = {k.split("::")[-1]: v for k, v in out["tests"].items()}
    xp = t["test_registered_unexpectedly_met"]
    assert xp["outcome"] == "xpassed"
    assert xp["xfail_marker"]["strict"] is False and "RR-TEST" in xp["xfail_marker"]["reason"]
    assert xp["recorded"]["measured"] == 1.5e-4
    assert t["test_registered_still_failing"]["outcome"] == "xfailed"
    assert t["test_registered_still_failing"]["recorded"]["measured"] == 3.0e-3
    assert t["test_ordinary"]["outcome"] == "passed"
    assert out["counts"] == {"xpassed": 1, "xfailed": 1, "passed": 1}
    assert out["exitstatus"] == 0
    assert out["lock"]["mode"] in ("gpu-hidden", "cpu-only")


def test_crash_in_registered_test_is_a_failure_not_an_xfail(tmp_path):
    r, out = _run(tmp_path, INNER)
    t = {k.split("::")[-1]: v for k, v in out["tests"].items()}
    assert t["test_registered_crashes"]["outcome"] == "failed"
    assert r.returncode == 1


def test_marker_listing_counts():
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "HYBRIDLC_GPUMON": "0"}
    r = subprocess.run([sys.executable, str(REPO / "scripts" / "repro" / "list_suite_markers.py")],
                       capture_output=True, text=True, env=env, cwd=REPO, timeout=600)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert "registered-criterion tests (xfail strict=False): 7" in r.stdout
    assert "relaxed regression guards: 13 test cases" in r.stdout

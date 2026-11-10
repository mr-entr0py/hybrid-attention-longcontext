"""Self-tests for the R2 tooling (CPU only; no CUDA context is created).

Run: ``pytest -q scripts/repro/test_r2_tooling.py``
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import gpu_lock as gl  # noqa: E402
import infra_events as ie  # noqa: E402

PY = sys.executable


@pytest.fixture()
def lockfile(tmp_path, monkeypatch):
    p = tmp_path / "q.lock"
    monkeypatch.setenv("HYBRIDLC_GPU_LOCK", str(p))
    return p


def _run_locked(label: str, sleep_s: float, extra=()):
    return subprocess.Popen([PY, str(HERE / "gpu_lock.py"), "run", "--label", label, *extra, "--",
                             PY, "-c", f"import time; print('start', time.time(), flush=True); time.sleep({sleep_s}); "
                             f"print('end', time.time(), flush=True)"],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=os.environ.copy())


def _times(out: str) -> tuple[float, float]:
    d = dict(ln.split() for ln in out.strip().splitlines())
    return float(d["start"]), float(d["end"])


def test_lock_serialises_two_jobs(lockfile):
    a = _run_locked("A", 3.0)
    time.sleep(0.8)
    b = _run_locked("B", 0.5)
    oa, _ = a.communicate(timeout=60)
    ob, eb = b.communicate(timeout=60)
    assert a.returncode == 0 and b.returncode == 0
    sa, ea = _times(oa)
    sb, _ = _times(ob)
    assert sb >= ea - 0.05, (sa, ea, sb)   # B started only after A ended
    assert "waiting" in eb or sb - sa > 2.5


def test_lock_timeout_exit_code(lockfile):
    a = _run_locked("A", 4.0)
    time.sleep(0.8)
    b = _run_locked("B", 0.1, extra=("--timeout", "1"))
    b.communicate(timeout=60)
    a.communicate(timeout=60)
    assert b.returncode == gl.EXIT_TIMEOUT


def test_lock_released_when_holder_killed(lockfile):
    import signal
    a = subprocess.Popen([PY, str(HERE / "gpu_lock.py"), "run", "--label", "A", "--", PY, "-c", "import time; time.sleep(60)"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=os.environ.copy(),
                         start_new_session=True)
    try:
        time.sleep(1.0)
        assert gl.is_locked(lockfile)
        a.kill()              # SIGKILL of the wrapper: the kernel drops the flock
        a.wait(timeout=30)
        time.sleep(0.2)
        # the orphaned child (sleep 60) survives SIGKILL of the wrapper, but the lock is free
        assert not gl.is_locked(lockfile)
        t0 = time.time()
        with gl.gpu_lock("C", timeout=5, verbose=False) as info:
            assert info["wait_s"] < 5
        assert time.time() - t0 < 5
    finally:
        try:
            os.killpg(a.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def test_returncode_propagates(lockfile):
    r = subprocess.run([PY, str(HERE / "gpu_lock.py"), "run", "--label", "rc", "--", PY, "-c", "raise SystemExit(3)"],
                       capture_output=True, text=True, env=os.environ.copy())
    assert r.returncode == 3


KNOWN = [
    "2026-11-09T18:03:09.512376+05:30 host kernel: NVRM: Xid (PCI:0000:02:00): 31, pid=799092, name=python, channel 0x0d000025, intr 00000000. MMU Fault: ENGINE CE4",
    "2026-11-09T18:16:50.579387+05:30 host kernel: NVRM: krcWatchdog_IMPL: RC watchdog: GPU is probably locked!  Notify Timeout Seconds: 7",
    "2026-11-09T18:16:50.581343+05:30 host kernel: NVRM: Xid (PCI:0000:02:00): 8, pid=811688, name=python, channel 0x00000003",
    "2026-11-09T17:31:22.127896+05:30 host kernel: Out of memory: Killed process 744173 (antigravity-ide) total-vm:1465211120kB",
]


def test_parse_lines():
    kinds = [ie.parse_line(x) for x in KNOWN]
    assert kinds[0]["kind"] == "xid" and kinds[0]["xid"] == 31 and kinds[0]["pid"] == "799092"
    assert kinds[1]["kind"] == "rc_watchdog"
    assert kinds[2]["kind"] == "xid" and kinds[2]["xid"] == 8 and kinds[2]["pid"] == "811688"
    assert kinds[3]["kind"] == "host_oom"
    assert all(ie.KLOG_PATTERN.search(x) for x in KNOWN)


@pytest.mark.skipif(not ie.journal_status()["journalctl_k_ok"], reason="kernel journal not readable")
def test_scan_known_window_matches_klog():
    """The plan's KLOG facts for 2026-11-09 17:30–18:30 must be recovered exactly."""
    t0 = time.mktime(time.strptime("2026-11-09 17:30", "%Y-%m-%d %H:%M"))
    t1 = time.mktime(time.strptime("2026-11-09 18:30", "%Y-%m-%d %H:%M"))
    res = ie.kernel_events(t0, t1)
    assert res["journal_ok"]
    xids = [(e["xid"], e["pid"], e["ts"][11:19]) for e in res["events"] if e["kind"] == "xid"]
    assert xids == [(31, "799092", "18:03:09"), (8, "811688", "18:16:50"), (8, "818052", "18:25:42")]
    assert [e["ts"][11:19] for e in res["events"] if e["kind"] == "rc_watchdog"] == ["18:16:50", "18:25:42"]
    assert any(e["kind"] == "host_oom" and e["ts"][11:19] == "17:31:22" for e in res["events"])

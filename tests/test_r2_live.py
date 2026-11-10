"""R2-LIVE v2 C1 (experiments/prereg/R2_live.v2.yaml; PI ruling D077): the single GPU lock, the entry-point
guards, the pytest-session lock, the infra block of trainer and runner, and the PR-27 v2 classifier table.

Every lock test uses a private lock file (``HYBRIDLC_GPU_LOCK``, honoured inside tests only) and strips the
session's inherited token, so it never interacts with the project lock ``runs/gpu_queue.lock``. Only the
first test needs CUDA (gpu-marked; the session holds the project lock while it runs).
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tests"))

from test_vt13_training import RUNNER, TRAIN, run_dir, write_tiny  # noqa: E402

from hybridlc.training.registry import Registry  # noqa: E402
from hybridlc.utils import gpu_lock as GL  # noqa: E402
from hybridlc.utils.infra import classify_events, parse_line  # noqa: E402

PY = sys.executable
XID8 = "NVRM: Xid (PCI:0000:02:00): 8, pid=4242, name=python, channel 0x00000003"
RCW = "NVRM: krcWatchdog_IMPL: RC watchdog: GPU is probably locked!  Notify Timeout Seconds: 7"


def _env(lock: Path | None = None, **extra: str) -> dict:
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "PYTHONHASHSEED": "0", "HYBRIDLC_GPUMON": "0"}
    for k in ("HYBRIDLC_FAULT", GL.TOKEN_ENV, GL.UNLOCKED_ENV, GL.WAIT_ENV, "HYBRIDLC_JOURNAL_SOURCE"):
        env.pop(k, None)
    if lock is not None:
        env[GL.LOCK_ENV] = str(lock)
    env.update(extra)
    return env


def _train(run_yaml: Path, root: Path, env: dict, fault: dict | None = None) -> subprocess.Popen:
    cmd = [PY, str(TRAIN), "--config", str(run_yaml), "--allow-dirty", "--runs-root", str(root / "runs"),
           "--ckpt-root", str(root / "ckpt"), "--registry", str(root / "registry.jsonl")]
    if fault:
        env = {**env, "HYBRIDLC_FAULT": json.dumps(fault)}
    return subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=str(REPO))


# ------------------------------------------------------------------------------------------ C1.1
@pytest.mark.gpu
def test_two_concurrent_train_runs_serialise(tmp_path):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    lock = tmp_path / "q.lock"
    env = _env(lock, **{GL.WAIT_ENV: "600"})
    for i in (0, 1):
        (tmp_path / f"r{i}").mkdir(exist_ok=True)
    runs = [write_tiny(tmp_path / f"r{i}", device="cuda", precision="bf16", steps=20, every=10, accum=2,
                       name=f"r{i}", audit=False) for i in (0, 1)]
    p0 = _train(runs[0], tmp_path / "root0", env)
    time.sleep(1.0)
    p1 = _train(runs[1], tmp_path / "root1", env)
    out = [p.communicate(timeout=900) for p in (p0, p1)]
    assert p0.returncode == 0, out[0][1][-3000:]
    assert p1.returncode == 0, out[1][1][-3000:]
    win = []
    for i in (0, 1):
        recs = list(Registry(tmp_path / f"root{i}" / "registry.jsonl").records())
        start = next(r["ts"] for r in recs if r["status"] == "running")
        end = next(r["ts"] for r in recs if r["status"] == "completed")
        summ = json.loads((run_dir(tmp_path / f"root{i}") / "summary.json").read_text())
        win.append((start, end, summ["infra"]["lock_wait_s"]))
    (s0, e0, w0), (s1, e1, w1) = win
    assert s1 >= e0 or s0 >= e1, win          # CUDA windows do not overlap
    assert max(w0, w1) > 0, win               # the second run waited for the lock


# ------------------------------------------------------------------------------------------ C1.2
PROBE = textwrap.dedent("""\
    import atexit, runpy, sys
    def _rep():
        t = sys.modules.get("torch")
        print("CUDA_INITIALIZED", bool(t is not None and t.cuda.is_initialized()), flush=True)
    atexit.register(_rep)
    sys.argv = [sys.argv[1]] + sys.argv[2:]
    runpy.run_path(sys.argv[0], run_name="__main__")
""")


@pytest.mark.parametrize("script,args", [
    ("scripts/bench_train_step.py", []),
    ("scripts/warm_triton.py", []),
    ("scripts/repro/r7_run.py", []),
    ("scripts/s1_zoology/run_one_r5.py", []),
    ("scripts/evaluate.py", ["--ckpt", "x.pt", "--suite", "FD"]),
    ("scripts/train.py", ["--config", "configs/training/examples/smoke_D384_C3.yaml"]),
])
def test_entry_point_without_lock_exits_78_before_cuda(tmp_path, script, args):
    lock = tmp_path / "q.lock"
    env = _env(lock)
    env.pop("CUDA_VISIBLE_DEVICES", None)  # a GPU entry point (the guard runs before any CUDA call)
    holder = subprocess.Popen([PY, str(REPO / "scripts/repro/gpu_lock.py"), "run", "--label", "other", "--",
                               PY, "-c", "import time; time.sleep(60)"], env=env, stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL, start_new_session=True)
    try:
        for _ in range(100):
            if GL.read_holder(lock).get("label") == "other":
                break
            time.sleep(0.1)
        r = subprocess.run([PY, "-c", PROBE, str(REPO / script), *args], env=env, capture_output=True, text=True,
                           timeout=300, cwd=str(REPO))
        assert r.returncode == GL.EXIT_NO_LOCK, (r.returncode, r.stdout[-1000:], r.stderr[-2000:])
        assert "CUDA_INITIALIZED False" in r.stdout, r.stdout
        assert "REFUSED" in r.stderr and r.stderr.strip().count("\n") <= 2
    finally:
        os.killpg(holder.pid, signal.SIGKILL)
        holder.wait(timeout=30)


def test_entry_point_with_inherited_token_and_escape_hatch(tmp_path):
    lock = tmp_path / "q.lock"
    env = _env(lock)
    code = ("import sys; sys.path.insert(0, 'src'); from hybridlc.utils.gpu_lock import guard; "
            "print('MODE', guard('probe')['mode'])")
    r = subprocess.run([PY, str(REPO / "scripts/repro/gpu_lock.py"), "run", "--label", "holder", "--", PY, "-c", code],
                       env=env, capture_output=True, text=True, timeout=120, cwd=str(REPO))
    assert r.returncode == 0 and "MODE inherited" in r.stdout, r.stdout + r.stderr
    r = subprocess.run([PY, "-c", code], env={**env, GL.UNLOCKED_ENV: "1"}, capture_output=True, text=True, timeout=60,
                       cwd=str(REPO))
    assert r.returncode == 0 and "MODE unlocked" in r.stdout
    # a stale token (lock free) does not count; the free lock is simply acquired
    r = subprocess.run([PY, "-c", code], env={**env, GL.TOKEN_ENV: "deadbeef"}, capture_output=True, text=True,
                       timeout=60, cwd=str(REPO))
    assert r.returncode == 0 and "MODE acquired" in r.stdout


def test_lock_override_is_test_only(tmp_path):
    env = _env(tmp_path / "q.lock")
    env.pop("PYTEST_CURRENT_TEST", None)
    code = ("import sys; sys.path.insert(0, 'src'); from hybridlc.utils.gpu_lock import guard; guard('probe')")
    r = subprocess.run([PY, "-c", code], env=env, capture_output=True, text=True, timeout=60, cwd=str(REPO))
    assert r.returncode == GL.EXIT_NO_LOCK and "only inside tests" in r.stderr


# ------------------------------------------------------------------------------------------ C1.3
INNER = textwrap.dedent("""\
    import os
    import pytest
    from hybridlc.utils import gpu_lock as GL

    @pytest.mark.gpu
    def test_gpu_marked_holds_lock():
        h = GL.read_holder()
        assert GL.is_locked() and h["label"] == "pytest" and h["pid"] == os.getpid()

    def test_cpu():
        pass
""")
INNER_CPU = textwrap.dedent("""\
    from hybridlc.utils import gpu_lock as GL

    def test_cpu_only_not_locked():
        assert not GL.is_locked()
""")


def test_pytest_session_lock(tmp_path):
    lock = tmp_path / "q.lock"
    (tmp_path / "pytest.ini").write_text("[pytest]\nmarkers =\n    gpu: needs CUDA\n")
    (tmp_path / "test_inner_gpu.py").write_text(INNER)
    (tmp_path / "test_inner_cpu.py").write_text(INNER_CPU)
    env = _env(lock, HYBRIDLC_TESTING="1")
    env.pop("PYTEST_CURRENT_TEST", None)
    base = [PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-p", "hybridlc.utils.pytest_gpu", "--rootdir",
            str(tmp_path)]
    r = subprocess.run([*base, str(tmp_path / "test_inner_gpu.py")], env=env, capture_output=True, text=True,
                       timeout=300, cwd=str(tmp_path))
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-2000:]
    assert not GL.is_locked(lock)  # released at session end
    r = subprocess.run([*base, str(tmp_path / "test_inner_cpu.py")], env=env, capture_output=True, text=True,
                       timeout=300, cwd=str(tmp_path))
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-2000:]
    # a session that needs the lock while another job holds it exits 78
    holder = subprocess.Popen([PY, str(REPO / "scripts/repro/gpu_lock.py"), "run", "--label", "other", "--",
                               PY, "-c", "import time; time.sleep(60)"], env={**env, "PYTEST_CURRENT_TEST": "x"},
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    try:
        for _ in range(100):
            if GL.read_holder(lock).get("label") == "other":
                break
            time.sleep(0.1)
        r = subprocess.run([*base, str(tmp_path / "test_inner_gpu.py")], env=env, capture_output=True, text=True,
                           timeout=300, cwd=str(tmp_path))
        assert r.returncode == GL.EXIT_NO_LOCK, r.stdout[-2000:]
    finally:
        os.killpg(holder.pid, signal.SIGKILL)
        holder.wait(timeout=30)


def test_session_triton_cache_is_diag():
    assert os.environ["TRITON_CACHE_DIR"] == str(REPO / ".cache" / "triton-diag")


# ------------------------------------------------------------------------------------------ C1.4
def _fake_run(tmp_path: Path, journal: str, name: str = "run") -> tuple[subprocess.CompletedProcess, Path]:
    run = write_tiny(tmp_path, steps=20, every=10, name=name)
    root = tmp_path / f"root_{name}"
    env = _env(None, HYBRIDLC_JOURNAL_SOURCE=journal)
    p = subprocess.run([PY, str(TRAIN), "--config", str(run), "--allow-dirty", "--runs-root", str(root / "runs"),
                        "--ckpt-root", str(root / "ckpt"), "--registry", str(root / "registry.jsonl")],
                       env=env, capture_output=True, text=True, timeout=600, cwd=str(REPO))
    return p, root


def test_trainer_infra_block_injected_events(tmp_path):
    jf = tmp_path / "journal.txt"
    jf.write_text(f"{XID8}\n{RCW}\n")  # undated lines are stamped at read time (inside the window)
    p, root = _fake_run(tmp_path, str(jf), "inj")
    assert p.returncode == 0, p.stderr[-3000:]
    blk = json.loads((run_dir(root) / "summary.json").read_text())["infra"]
    assert blk["flagged"] is True and blk["classification"] == "infra" and blk["n_events"] == 2, blk
    assert blk["journal_ok"] is True and blk["journal_source"] == str(jf)
    for k in ("window", "events", "oomd_ok", "foreign_gpu_processes", "unlocked_project_processes", "lock_wait_s"):
        assert k in blk, k
    done = [r for r in Registry(root / "registry.jsonl").records() if r["status"] == "completed"][0]
    assert done["infra_flagged"] is True and done["infra_class"] == "infra" and done["n_infra_events"] == 2
    assert done["journal_ok"] is True


def test_trainer_infra_block_empty_and_unreadable(tmp_path):
    p, root = _fake_run(tmp_path, ":empty:", "empty")
    assert p.returncode == 0, p.stderr[-3000:]
    blk = json.loads((run_dir(root) / "summary.json").read_text())["infra"]
    assert blk["flagged"] is False and blk["journal_ok"] is True and blk["classification"] == "none"
    p, root = _fake_run(tmp_path, ":unreadable:", "unread")
    assert p.returncode == 0, p.stderr[-3000:]
    blk = json.loads((run_dir(root) / "summary.json").read_text())["infra"]
    assert blk["journal_ok"] is False
    done = [r for r in Registry(root / "registry.jsonl").records() if r["status"] == "completed"][0]
    assert done["journal_ok"] is False and done["infra_flagged"] is False


# ------------------------------------------------------------------------------------------ C1.5
CLASSIFIER_TABLE = [
    ([XID8, RCW], "infra"),
    (["NVRM: Xid (PCI:0000:02:00): 79, pid=1, name=python, GPU has fallen off the bus."], "infra"),
    (["Out of memory: Killed process 744173 (python) total-vm:1465211120kB"], "infra"),
    (["systemd-oomd[812]: Killed /user.slice/... due to memory pressure"], "infra"),
    (["NVRM: Xid (PCI:0000:02:00): 13, pid=1, name=python, Graphics Exception"], "suspect_sw"),
    (["NVRM: Xid (PCI:0000:02:00): 31, pid=1, name=python, channel 0x0d000025, MMU Fault"], "suspect_sw"),
    (["NVRM: Xid (PCI:0000:02:00): 43, pid=1, name=python, channel 0x1"], "suspect_sw"),
    (["NVRM: Xid (PCI:0000:02:00): 8, pid=1, name=python, channel 0x00000003"], "suspect_sw"),  # no RC watchdog
    (["NVRM: Xid (PCI:0000:02:00): 109, pid=1, name=python, ctx switch timeout"], "suspect_sw"),  # unknown Xid
    ([], "none"),
]


@pytest.mark.parametrize("lines,cls", CLASSIFIER_TABLE)
def test_classifier_table_kernel_lines(lines, cls):
    evs = [parse_line(f"2026-11-10T10:00:00.000000+05:30 host kernel: {x}") for x in lines]
    _, got = classify_events(evs)
    assert got == cls


def test_classifier_cuda_errors_from_attempt_log():
    log = ["RuntimeError: CUDA error: an illegal memory access was encountered"]
    assert classify_events([], log)[1] == "suspect_sw"
    log = ["torch.AcceleratorError: CUDA error: misaligned address"]
    assert classify_events([], log)[1] == "suspect_sw"
    log = ["CUDA error: an illegal instruction was encountered"]
    assert classify_events([], log)[1] == "suspect_sw"
    rc = [parse_line(f"2026-11-10T10:00:00+05:30 host kernel: {RCW}")]
    log = ["RuntimeError: CUDA error: the launch timed out and was terminated (cudaErrorLaunchTimeout)"]
    assert classify_events(rc, log)[1] == "infra"
    assert classify_events([], log)[1] == "suspect_sw"


# ------------------------------------------------------------------------------------------ C1.6
def test_runner_attempt_end_for_sigkilled_child(tmp_path):
    run = write_tiny(tmp_path, steps=20, every=10)
    common = ["--registry", str(tmp_path / "registry.jsonl"), "--runs-root", str(tmp_path / "runs"),
              "--ckpt-root", str(tmp_path / "ckpt"), "--min-free-gb", "1", "--no-monitor"]
    env = _env(None, HYBRIDLC_JOURNAL_SOURCE=":empty:")
    subprocess.run([PY, str(RUNNER), *common, "enqueue", str(run), "--allow-dirty"], env=env, check=True,
                   capture_output=True, timeout=120)
    r = subprocess.run([PY, str(RUNNER), *common, "run"], env={**env, "HYBRIDLC_FAULT": json.dumps({"kill_at": [15, 1]})},
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    recs = list(Registry(tmp_path / "registry.jsonl").records())
    ends = [x for x in recs if x["status"] == "attempt_end"]
    assert len(ends) == 2, [x["status"] for x in recs]
    killed, ok = ends
    assert killed["exit_code"] == -signal.SIGKILL and killed["attempt"] == 1 and killed["last_step"] == 15
    assert ok["exit_code"] == 0 and ok["attempt"] == 2
    for e in ends:
        assert e["infra"]["journal_ok"] is True and "classification" in e["infra"]
        for k in ("infra_flagged", "infra_class", "journal_ok", "n_infra_events"):
            assert k in e, k
    st = [x["status"] for x in recs if x["status"] not in ("attempt_end", "monitor")]
    assert st == ["queued", "running", "crashed", "running", "completed"], st
    assert (tmp_path / "runs" / "runner_summary.json").exists()

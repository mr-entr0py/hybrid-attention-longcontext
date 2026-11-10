"""PR-27 v2 infrastructure status (PI ruling D084; R2_live.v2.yaml L2), CPU with training fakes.

* event classes map as ruled (classifier table: tests/test_r2_live.py; here the runner consequences);
* an infra attempt is resumed and never counted by D031's failure rule;
* the queue pauses a run after 10 infra attempts (never auto-failed);
* a ``suspect_sw_failure`` counts as a failure; a second one at the same optimizer step stops the run
  (``paused`` registry row);
* per-run / per-class / per-k counts appear in the runner summary;
* G-OOM is unchanged (tests/test_vt13_training.py::test_oom_exit_code_and_resume stays green).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tests"))

from test_vt13_training import RUNNER, write_tiny  # noqa: E402

from hybridlc.training.config import PathsConfig  # noqa: E402
from hybridlc.training.queue import Runner, RunnerConfig  # noqa: E402
from hybridlc.training.registry import Registry  # noqa: E402

PY = sys.executable
XID8 = "NVRM: Xid (PCI:0000:02:00): 8, pid=4242, name=python, channel 0x00000003"
RCW = "NVRM: krcWatchdog_IMPL: RC watchdog: GPU is probably locked!  Notify Timeout Seconds: 7"
XID31 = "NVRM: Xid (PCI:0000:02:00): 31, pid=4242, name=python, channel 0x0d000025, MMU Fault"


def _env(**extra):
    env = {**os.environ, "OMP_NUM_THREADS": "1", "PYTHONHASHSEED": "0", "HYBRIDLC_GPUMON": "0"}
    for k in ("HYBRIDLC_FAULT", "HYBRIDLC_GPU_LOCK_TOKEN", "HYBRIDLC_JOURNAL_SOURCE"):
        env.pop(k, None)
    env.update(extra)
    return env


def _runner_cli(tmp_path: Path, run_yaml: Path, env: dict) -> subprocess.CompletedProcess:
    common = ["--registry", str(tmp_path / "registry.jsonl"), "--runs-root", str(tmp_path / "runs"),
              "--ckpt-root", str(tmp_path / "ckpt"), "--min-free-gb", "1", "--no-monitor"]
    subprocess.run([PY, str(RUNNER), *common, "enqueue", str(run_yaml), "--allow-dirty"], env=env, check=True,
                   capture_output=True, timeout=120)
    return subprocess.run([PY, str(RUNNER), *common, "run"], env=env, capture_output=True, text=True, timeout=900)


def test_infra_attempt_resumed_and_not_counted(tmp_path):
    jf = tmp_path / "journal.txt"
    jf.write_text("")
    run = write_tiny(tmp_path, steps=20, every=10)
    env = _env(HYBRIDLC_JOURNAL_SOURCE=str(jf),
               HYBRIDLC_FAULT=json.dumps({"kill_at": [15, 1], "inject_journal": [XID8, RCW]}))
    r = _runner_cli(tmp_path, run, env)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    reg = Registry(tmp_path / "registry.jsonl")
    recs = list(reg.records())
    st = [x["status"] for x in recs if x["status"] not in ("attempt_end", "monitor")]
    assert st == ["queued", "running", "infra_failed_attempt", "running", "completed"], st
    ends = [x for x in recs if x["status"] == "attempt_end"]
    assert ends[0]["infra_class"] == "infra" and ends[0]["infra_flagged"] and ends[0]["n_infra_events"] == 2
    assert ends[1]["infra_class"] == "none"  # the resumed attempt's window holds no event
    runner = Runner(RunnerConfig(paths=PathsConfig(runs_root=str(tmp_path / "runs"), ckpt_root=str(tmp_path / "ckpt"),
                                                   registry=str(tmp_path / "registry.jsonl"))))
    (rid,) = reg.state()
    c = runner.failure_counts()["runs"][rid]
    assert c["infra_failed_attempt"] == 1 and c["counted_failures"] == 0 and c["crashed"] == 0
    summ = json.loads((tmp_path / "runs" / "runner_summary.json").read_text())
    assert summ["per_k"][str(c["k"])]["infra_failed_attempt"] == 1


def _mk_runner(tmp_path: Path, **kw) -> tuple[Runner, str, dict, Path]:
    rc = RunnerConfig(paths=PathsConfig(runs_root=str(tmp_path / "runs"), ckpt_root=str(tmp_path / "ckpt"),
                                        registry=str(tmp_path / "registry.jsonl")), monitor=False, **kw)
    runner = Runner(rc)
    rid = "S2T-D384-C12-s101-deadbeef"
    run_dir = tmp_path / "runs" / rid
    run_dir.mkdir(parents=True)
    runner.registry.append(rid, "queued", config="x.yaml", steps=100)
    runner.registry.append(rid, "running", k=12, start_step=0, pid=1)
    return runner, rid, {"run_id": rid, "run_dir": str(run_dir), "ckpt_dir": str(tmp_path / "ckpt" / rid)}, run_dir


def _trace_to(run_dir: Path, last_step: int) -> None:
    with open(run_dir / "trace.jsonl", "a") as f:
        for s in range(last_step):
            f.write(json.dumps({"step": s, "loss": 1.0}) + "\n")


def _blk(cls: str) -> dict:
    return {"classification": cls, "flagged": cls != "none", "journal_ok": True, "n_events": int(cls != "none")}


def test_pause_after_ten_infra_attempts(tmp_path):
    runner, rid, job, run_dir = _mk_runner(tmp_path)
    for i in range(10):
        _trace_to(run_dir, 5 + i)
        runner._post(job, -9, run_dir, _blk("infra"))
        if i < 9:
            assert runner.registry.status(rid) == "infra_failed_attempt"
            assert runner.next_job() is not None  # still scheduled (resumed)
    assert runner.registry.status(rid) == "paused"
    assert runner.next_job() is None            # never auto-failed, never scheduled until unpaused
    assert runner.failure_counts()["runs"][rid]["counted_failures"] == 0
    runner.unpause(rid)
    assert runner.next_job() is not None


def test_suspect_counts_and_recurrence_at_same_step_pauses(tmp_path):
    runner, rid, job, run_dir = _mk_runner(tmp_path)
    _trace_to(run_dir, 37)
    runner._post(job, -6, run_dir, _blk("suspect_sw"))
    assert runner.registry.status(rid) == "suspect_sw_failure"
    assert runner.failure_counts()["runs"][rid]["counted_failures"] == 1
    assert runner.next_job() is not None  # one resume
    runner._post(job, -6, run_dir, _blk("suspect_sw"))  # same last completed step (37)
    hist = runner.registry.history(rid)
    assert hist[-1]["status"] == "paused" and hist[-1]["step"] == 37 and "recurred" in hist[-1]["reason"]
    assert runner.next_job() is None


def test_suspect_at_different_steps_then_failed_by_d031(tmp_path):
    runner, rid, job, run_dir = _mk_runner(tmp_path, max_crash_retries=2)
    for n in range(3):
        _trace_to(run_dir, n + 1)  # progress between attempts: different last step each time
        runner._post(job, -6, run_dir, _blk("suspect_sw" if n != 1 else "none"))
    assert runner.registry.status(rid) == "failed"  # 3 counted failures (2 suspect + 1 crashed) > 2
    assert runner.failure_counts()["runs"][rid]["counted_failures"] == 3


def test_infra_attempts_do_not_reset_or_count_d031(tmp_path):
    runner, rid, job, run_dir = _mk_runner(tmp_path, max_crash_retries=2)
    _trace_to(run_dir, 3)
    runner._post(job, 1, run_dir, _blk("none"))
    for _ in range(4):
        runner._post(job, -9, run_dir, _blk("infra"))
    runner._post(job, 1, run_dir, _blk("none"))
    c = runner.failure_counts()["runs"][rid]
    assert c["counted_failures"] == 2 and c["infra_failed_attempt"] == 4
    assert runner.registry.status(rid) == "crashed"


def test_per_k_counts_in_summary(tmp_path):
    runner, rid, job, run_dir = _mk_runner(tmp_path)
    rid0 = "S2T-D384-C0-s101-cafef00d"
    runner.registry.append(rid0, "queued", config="y.yaml", steps=100)
    runner.registry.append(rid0, "running", k=0, start_step=0, pid=1)
    (tmp_path / "runs" / rid0).mkdir()
    _trace_to(run_dir, 4)
    runner._post(job, -9, run_dir, _blk("infra"))
    runner._post(job, -9, run_dir, _blk("infra"))
    runner._post({**job, "run_id": rid0, "run_dir": str(tmp_path / "runs" / rid0)}, -6, tmp_path / "runs" / rid0,
                 _blk("suspect_sw"))
    pk = json.loads(runner.write_summary().read_text())["per_k"]
    assert pk["12"]["infra_failed_attempt"] == 2 and pk["0"]["suspect_sw_failure"] == 1
    assert "per k:" in runner.status_table()


def test_oom_path_unchanged(tmp_path):
    """G-OOM: exit 75 is handled before any classification (3 consecutive OOMs without progress -> failed)."""
    runner, rid, job, run_dir = _mk_runner(tmp_path)
    for _ in range(3):
        runner.registry.append(rid, "oom", ckpt_step=None)  # written by the trainer on a CUDA OOM
        runner._post(job, 75, run_dir, _blk("infra"))
    assert runner.registry.status(rid) == "failed"
    assert [r["status"] for r in runner.registry.history(rid)].count("infra_failed_attempt") == 0

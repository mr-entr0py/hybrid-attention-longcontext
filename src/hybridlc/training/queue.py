"""Single-GPU job queue over the append-only registry (protocol §18 "Runner").

The queue *is* ``experiments/registry.jsonl``: ``enqueue`` appends a ``queued`` record, the trainer
appends ``running`` / ``completed`` / ``oom`` / ``diverged`` / ``failed``, and the runner appends
``crashed`` / ``interrupted`` / ``failed`` / ``monitor``. A run's state is its last queue-state record
(see :mod:`registry`). Jobs run strictly one at a time, FIFO by first enqueue, skipping jobs whose
parent (branch point) has not completed yet.

Retry policy:
* ``oom`` (G-OOM) → auto-resume; 3 consecutive OOMs without checkpoint progress → ``failed``;
* ``crashed`` (any other non-zero exit) / ``interrupted`` (runner or machine died) → auto-resume, up to
  ``max_crash_retries`` crashes since the last enqueue, then ``failed``;
* exit 76 (diverged twice) and 77 (refused) are terminal and recorded by the trainer.
Guards: free disk space under the checkpoint root must exceed ``min_free_gb`` (50 GB) before a launch.
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .compat import REPO_ROOT
from .config import PathsConfig, load_run_config
from .registry import RETRYABLE, TERMINAL, Registry

EXIT_OOM, EXIT_DIVERGED, EXIT_REFUSED = 75, 76, 77


def _log(msg: str) -> None:
    print(f"[runner {time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


@dataclass
class RunnerConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    python: str = sys.executable
    train_script: str = str(REPO_ROOT / "scripts" / "train.py")
    monitor_script: str = str(REPO_ROOT / "scripts" / "gpu_monitor.py")
    monitor_interval: float = 30.0
    monitor: bool = True
    poll: float = 30.0
    min_free_gb: float = 50.0
    max_crash_retries: int = 2
    max_oom_no_progress: int = 3
    extra_env: dict[str, str] = field(default_factory=dict)


class Runner:
    def __init__(self, rc: RunnerConfig):
        self.rc = rc
        self.registry = Registry(rc.paths.resolve(rc.paths.registry))
        self.child: subprocess.Popen | None = None
        self._stop = False

    # ------------------------------------------------------------------------------------------
    # enqueue / cancel
    # ------------------------------------------------------------------------------------------
    def enqueue(self, config: str, overrides: list[str] | None = None, allow_dirty: bool = False,
                tag: str | None = None, force: bool = False) -> str:
        cfg = load_run_config(config, cli_overrides=overrides, paths=self.rc.paths, tag=tag)
        st = self.registry.state().get(cfg.run_id)
        if st is not None and not force:
            if st["status"] not in TERMINAL:
                _log(f"{cfg.run_id} already in the queue ({st['status']}); not re-enqueued")
                return cfg.run_id
            if st["status"] == "completed":
                _log(f"{cfg.run_id} already completed; use --force to re-run")
                return cfg.run_id
        self.registry.append(cfg.run_id, "queued", config=str(Path(config).resolve()), overrides=overrides or [],
                             allow_dirty=allow_dirty, tag=tag, parent_run_id=cfg.parent_run_id,
                             parent_ckpt=cfg.parent_ckpt, config_hash=cfg.config_hash(), stage=cfg.stage,
                             exp_stage=cfg.exp_stage, scale=cfg.scale, config_name=cfg.config_name,
                             seed_index=cfg.seed_index, steps=cfg.schedule.steps,
                             run_dir=str(cfg.run_dir), ckpt_dir=str(cfg.ckpt_dir))
        _log(f"queued {cfg.run_id}")
        return cfg.run_id

    def cancel(self, run_id: str, reason: str = "user") -> None:
        self.registry.append(run_id, "cancelled", reason=reason)

    # ------------------------------------------------------------------------------------------
    # scheduling
    # ------------------------------------------------------------------------------------------
    def reconcile(self) -> None:
        """Mark ``running`` jobs whose process is gone as ``interrupted`` (they will be resumed)."""
        for rid, st in self.registry.state().items():
            if st["status"] != "running":
                continue
            pid = (st["last"] or {}).get("pid")
            if self.child is not None and pid == self.child.pid:
                continue
            if pid is None or not _pid_alive(int(pid)):
                self.registry.append(rid, "interrupted", stale_pid=pid)
                _log(f"{rid}: stale 'running' (pid {pid} gone) -> interrupted")

    def next_job(self) -> dict[str, Any] | None:
        state = self.registry.state()
        for rid, st in state.items():
            if st["status"] not in RETRYABLE or st["queue"] is None:
                continue
            q = st["queue"]
            parent = q.get("parent_run_id")
            if parent:
                pst = state.get(parent)
                pstatus = pst["status"] if pst else None
                if pstatus in ("failed", "cancelled"):
                    self.registry.append(rid, "failed", reason=f"parent {parent} {pstatus}")
                    continue
                if pstatus != "completed" and not (q.get("parent_ckpt") and Path(q["parent_ckpt"]).is_file()
                                                   and pst is None):
                    continue  # wait for the branch point
            return {"run_id": rid, **q}
        return None

    def pending(self) -> list[str]:
        return [rid for rid, st in self.registry.state().items() if st["status"] in RETRYABLE]

    def _disk_ok(self) -> bool:
        root = self.rc.paths.resolve(self.rc.paths.ckpt_root)
        probe = root if root.exists() else root.parent
        free_gb = shutil.disk_usage(probe).free / 1e9
        if free_gb < self.rc.min_free_gb:
            _log(f"disk guard: {free_gb:.1f} GB free < {self.rc.min_free_gb} GB; not launching")
            return False
        return True

    # ------------------------------------------------------------------------------------------
    # one job
    # ------------------------------------------------------------------------------------------
    def build_cmd(self, job: dict[str, Any]) -> list[str]:
        p = self.rc.paths
        cmd = [self.rc.python, self.rc.train_script, "--config", job["config"], "--resume",
               "--runs-root", str(p.resolve(p.runs_root)), "--ckpt-root", str(p.resolve(p.ckpt_root)),
               "--registry", str(p.resolve(p.registry))]
        for o in job.get("overrides") or []:
            cmd += ["-o", o]
        if job.get("tag"):
            cmd += ["--tag", job["tag"]]
        if job.get("allow_dirty"):
            cmd.append("--allow-dirty")
        return cmd

    def run_job(self, job: dict[str, Any]) -> int:
        rid = job["run_id"]
        run_dir = Path(job.get("run_dir") or self.rc.paths.resolve(self.rc.paths.runs_root) / rid)
        run_dir.mkdir(parents=True, exist_ok=True)
        cmd = self.build_cmd(job)
        env = {**os.environ, **self.rc.extra_env, "PYTHONUNBUFFERED": "1"}
        _log(f"launch {rid}: {' '.join(cmd)}")
        with open(run_dir / "train.log", "a") as logf:
            logf.write(f"\n===== launch {time.strftime('%Y-%m-%dT%H:%M:%S')} : {' '.join(cmd)}\n")
            logf.flush()
            self.child = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env, cwd=str(REPO_ROOT))
            mon = None
            if self.rc.monitor:
                mon = subprocess.Popen([self.rc.python, self.rc.monitor_script, "--out", str(run_dir / "gpu.jsonl"),
                                        "--pid", str(self.child.pid), "--interval", str(self.rc.monitor_interval)],
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=str(REPO_ROOT))
            code = self.child.wait()
            self.child = None
            if mon is not None:
                try:
                    mon.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    mon.terminate()
        self._post(job, code, run_dir)
        return code

    def _post(self, job: dict[str, Any], code: int, run_dir: Path) -> None:
        rid = job["run_id"]
        status = self.registry.status(rid)
        summ_path = run_dir / "gpu_summary.json"
        if summ_path.exists():
            try:
                self.registry.append(rid, "monitor", exit_code=code, **json.loads(summ_path.read_text()))
            except json.JSONDecodeError:
                pass
        if code == 0:
            if status != "completed":
                self.registry.append(rid, "crashed", exit_code=0, reason="exit 0 without a completed record")
            _log(f"{rid}: exit 0 ({self.registry.status(rid)})")
            return
        if code == EXIT_OOM:
            if status != "oom":
                from .checkpoint import latest_step

                ck = job.get("ckpt_dir") or self.rc.paths.resolve(self.rc.paths.ckpt_root) / rid
                self.registry.append(rid, "oom", exit_code=code, ckpt_step=latest_step(ck))
            n = self._oom_no_progress(rid)
            _log(f"{rid}: OOM (consecutive without progress: {n})")
            if n >= self.rc.max_oom_no_progress:
                self.registry.append(rid, "failed", reason=f"G-OOM: {n} consecutive OOM failures without progress")
            return
        if code in (EXIT_DIVERGED, EXIT_REFUSED):
            if status != "failed":
                self.registry.append(rid, "failed", exit_code=code,
                                     reason="diverged twice" if code == EXIT_DIVERGED else "refused")
            _log(f"{rid}: terminal exit {code}")
            return
        if code < 0 and self._stop:
            self.registry.append(rid, "interrupted", exit_code=code, reason="runner stopped")
            return
        tail = _tail(run_dir / "train.log", 30)
        self.registry.append(rid, "crashed", exit_code=code, log_tail=tail)
        n = self._crashes_since_enqueue(rid)
        _log(f"{rid}: crashed with exit {code} ({n} crash(es) since enqueue)")
        if n > self.rc.max_crash_retries:
            self.registry.append(rid, "failed", reason=f"{n} crashes")

    def _oom_no_progress(self, rid: str) -> int:
        n, last = 0, object()
        for r in self.registry.history(rid):
            if r["status"] == "queued":
                n, last = 0, object()
            elif r["status"] == "oom" and "ckpt_step" in r:
                n = n + 1 if r["ckpt_step"] == last else 1
                last = r["ckpt_step"]
        return n

    def _crashes_since_enqueue(self, rid: str) -> int:
        n = 0
        for r in self.registry.history(rid):
            if r["status"] == "queued":
                n = 0
            elif r["status"] == "crashed":
                n += 1
        return n

    # ------------------------------------------------------------------------------------------
    # main loop
    # ------------------------------------------------------------------------------------------
    def loop(self, forever: bool = False, max_jobs: int | None = None) -> int:
        """Run queued jobs one at a time. Without ``forever`` the loop exits when nothing is runnable."""
        def _term(signum, _frame):
            self._stop = True
            if self.child is not None:
                _log(f"signal {signum}: forwarding SIGTERM to pid {self.child.pid}")
                self.child.send_signal(signal.SIGTERM)

        signal.signal(signal.SIGTERM, _term)
        signal.signal(signal.SIGINT, _term)
        n = 0
        self.reconcile()
        while not self._stop:
            job = self.next_job()
            if job is None:
                if not forever:
                    break
                time.sleep(self.rc.poll)
                self.reconcile()
                continue
            if not self._disk_ok():
                if not forever:
                    break
                time.sleep(self.rc.poll)
                continue
            self.run_job(job)
            n += 1
            if max_jobs is not None and n >= max_jobs:
                break
        return n

    def status_table(self) -> str:
        rows = []
        for rid, st in self.registry.state().items():
            last = st["last"] or {}
            step = last.get("steps_done", last.get("step", last.get("ckpt_step", last.get("start_step", ""))))
            rows.append(f"{st['status'] or '?':12s} {rid:60s} step={step}")
        return "\n".join(rows) if rows else "(registry empty)"


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        return Path(f"/proc/{pid}/stat").read_text().split()[2] != "Z"
    except OSError:
        return False


def _tail(path: Path, n: int) -> str:
    try:
        return "".join(path.read_text(errors="replace").splitlines(keepends=True)[-n:])
    except OSError:
        return ""


def linger_enabled() -> bool:
    try:
        out = subprocess.run(["loginctl", "show-user", os.environ.get("USER", ""), "-p", "Linger"],
                             capture_output=True, text=True, timeout=10).stdout
        return out.strip().endswith("yes")
    except (OSError, subprocess.SubprocessError):
        return False

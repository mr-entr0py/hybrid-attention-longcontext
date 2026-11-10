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

R2-LIVE v2 (PI ruling D077): a CUDA job runs while the runner holds the project GPU lock (acquired per
job; the child inherits the token); every attempt is wrapped in an infra tagger and ends with an
``attempt_end`` registry row (infra block, exit code, last completed optimizer step, attempt number),
also for attempts killed by a signal or a CUDA error.

PR-27 v2 (PI ruling D084), applied to every non-zero exit that is not G-OOM / 76 / 77 / runner stop:
* ``infra`` class (Xid 8 + RC watchdog, Xid 79, host OOM-kill) -> ``infra_failed_attempt``: resumed,
  never counted toward D031's failure count; after ``max_infra_attempts`` (10) on one run -> ``paused``;
* ``suspect_sw`` class (Xid 13/31/43, illegal address/instruction, misaligned address, unknown Xid) ->
  ``suspect_sw_failure``: counts as a failure; one resume; a second one at the same last completed
  optimizer step -> ``paused`` (deterministic fault, PI diagnosis);
* no event -> ``crashed`` as before.
Counted failures (``crashed`` + ``suspect_sw_failure``) reset at ``queued`` / ``unpaused`` /
``completed``; more than ``max_crash_retries`` -> ``failed``. G-OOM is unchanged.
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
from .registry import HELD, RETRYABLE, TERMINAL, Registry

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
    max_infra_attempts: int = 10
    wait_exclusive: bool = False      # --wait-exclusive semantics for the per-job GPU lock
    infra_interval: float = 5.0
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

    def unpause(self, run_id: str, reason: str = "PI decision") -> None:
        if self.registry.status(run_id) not in HELD:
            raise ValueError(f"{run_id} is not paused ({self.registry.status(run_id)})")
        self.registry.append(run_id, "unpaused", reason=reason)

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

    def _needs_gpu(self, job: dict[str, Any]) -> bool:
        try:
            cfg = load_run_config(job["config"], cli_overrides=job.get("overrides") or [], paths=self.rc.paths,
                                  tag=job.get("tag"))
            return str(cfg.runtime.device).startswith("cuda")
        except Exception:  # noqa: BLE001 - when in doubt, treat the job as a GPU job (take the lock)
            return True

    def _lock_ctx(self, rid: str, needs_gpu: bool):
        import contextlib

        if not needs_gpu:
            return contextlib.nullcontext({"mode": "cpu"})
        from hybridlc.utils.gpu_lock import gpu_lock, verify_token

        inh = verify_token()
        if inh is not None:  # the runner itself runs under a lock holder: children inherit its token
            return contextlib.nullcontext({"mode": "inherited", **inh})
        return gpu_lock(f"runner:{rid}", timeout=None, wait_exclusive=self.rc.wait_exclusive)

    def run_job(self, job: dict[str, Any]) -> int:
        from hybridlc.utils.infra import InfraTagger

        rid = job["run_id"]
        run_dir = Path(job.get("run_dir") or self.rc.paths.resolve(self.rc.paths.runs_root) / rid)
        run_dir.mkdir(parents=True, exist_ok=True)
        cmd = self.build_cmd(job)
        needs_gpu = self._needs_gpu(job)
        attempt = 1 + sum(1 for r in self.registry.history(rid) if r["status"] == "attempt_end")
        with self._lock_ctx(rid, needs_gpu) as lock_info:
            env = {**os.environ, **self.rc.extra_env, "PYTHONUNBUFFERED": "1"}
            _log(f"launch {rid} (attempt {attempt}, lock {lock_info.get('mode', 'acquired')}): {' '.join(cmd)}")
            tagger = InfraTagger(tag=f"{rid}#attempt{attempt}", interval=self.rc.infra_interval, sample=needs_gpu).start()
            with open(run_dir / "train.log", "a") as logf:
                logf.write(f"\n===== launch {time.strftime('%Y-%m-%dT%H:%M:%S')} attempt {attempt} : {' '.join(cmd)}\n")
                logf.flush()
                log_pos = logf.tell()
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
            log_lines = _read_from(run_dir / "train.log", log_pos).splitlines()[-200:]
            res = tagger.finish(log_lines=log_lines)
        blk = res["infra"]
        from hybridlc.utils.infra import registry_fields

        self.registry.append(rid, "attempt_end", attempt=attempt, exit_code=code, last_step=last_completed_step(run_dir),
                             lock_mode=lock_info.get("mode", "acquired"),
                             infra={k: v for k, v in blk.items()}, **registry_fields(blk))
        self._post(job, code, run_dir, blk)
        return code

    def _post(self, job: dict[str, Any], code: int, run_dir: Path, infra: dict[str, Any] | None = None) -> None:
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
        cls = (infra or {}).get("classification", "none")
        last = last_completed_step(run_dir)
        if cls == "infra":
            self.registry.append(rid, "infra_failed_attempt", exit_code=code, last_step=last, log_tail=tail,
                                 infra_class=cls)
            n_inf = self._since_reset(rid, ("infra_failed_attempt",), resets=("queued", "unpaused"))
            _log(f"{rid}: infrastructure-failed attempt (exit {code}; {n_inf} since enqueue/unpause); resuming")
            if n_inf >= self.rc.max_infra_attempts:
                self.registry.append(rid, "paused", reason=f"{n_inf} infrastructure-failed attempts (PR-27 v2): PI decision",
                                     n_infra_attempts=n_inf)
            return
        if cls == "suspect_sw":
            prev = [r for r in self._since_last(rid, ("queued", "unpaused")) if r["status"] == "suspect_sw_failure"]
            self.registry.append(rid, "suspect_sw_failure", exit_code=code, last_step=last, log_tail=tail,
                                 infra_class=cls)
            if any(r.get("last_step") == last for r in prev):
                self.registry.append(rid, "paused", reason=f"suspect_sw_failure recurred at optimizer step {last} "
                                                           "(deterministic fault?): PI diagnosis (PR-27 v2)",
                                     step=last)
                _log(f"{rid}: suspected software failure recurred at step {last}; paused")
                return
        else:
            self.registry.append(rid, "crashed", exit_code=code, log_tail=tail, last_step=last)
        n = self._crashes_since_enqueue(rid)
        _log(f"{rid}: {'suspect_sw_failure' if cls == 'suspect_sw' else 'crashed'} with exit {code} "
             f"({n} counted failure(s))")
        if n > self.rc.max_crash_retries:
            self.registry.append(rid, "failed", reason=f"{n} counted failures (crashed / suspect_sw_failure)")

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
        """D031 counted failures: ``crashed`` and ``suspect_sw_failure`` since the last enqueue, unpause
        or completion (infrastructure-failed attempts never count, PR-27 v2)."""
        return self._since_reset(rid, ("crashed", "suspect_sw_failure"), resets=("queued", "unpaused", "completed"))

    def _since_last(self, rid: str, resets: tuple[str, ...]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for r in self.registry.history(rid):
            if r["status"] in resets:
                out = []
            else:
                out.append(r)
        return out

    def _since_reset(self, rid: str, statuses: tuple[str, ...], resets: tuple[str, ...]) -> int:
        return sum(1 for r in self._since_last(rid, resets) if r["status"] in statuses)

    def failure_counts(self) -> dict[str, Any]:
        """Per run, per class and per k counts of classified attempts (PR-27 v2 reporting; FEAS F-infra)."""
        runs: dict[str, dict[str, Any]] = {}
        for rid in self.registry.state():
            hist = self.registry.history(rid)
            k = next((r.get("k") for r in reversed(hist) if r["status"] == "running" and r.get("k") is not None), None)
            c = {"k": k, "infra_failed_attempt": 0, "suspect_sw_failure": 0, "crashed": 0, "oom": 0, "attempts": 0,
                 "counted_failures": self._crashes_since_enqueue(rid), "status": self.registry.status(rid)}
            for r in hist:
                if r["status"] in ("infra_failed_attempt", "suspect_sw_failure", "crashed", "oom"):
                    c[r["status"]] += 1
                elif r["status"] == "attempt_end":
                    c["attempts"] += 1
            runs[rid] = c
        per_k: dict[str, dict[str, int]] = {}
        for c in runs.values():
            pk = per_k.setdefault(str(c["k"]), {"infra_failed_attempt": 0, "suspect_sw_failure": 0, "crashed": 0,
                                                "oom": 0, "runs": 0})
            pk["runs"] += 1
            for key in ("infra_failed_attempt", "suspect_sw_failure", "crashed", "oom"):
                pk[key] += c[key]
        return {"runs": runs, "per_k": per_k}

    def write_summary(self) -> Path:
        out = self.rc.paths.resolve(self.rc.paths.runs_root) / "runner_summary.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"ts": time.time(), **self.failure_counts()}, indent=1, default=str))
        return out

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
        self.write_summary()
        return n

    def status_table(self) -> str:
        rows = []
        fc = self.failure_counts()
        for rid, st in self.registry.state().items():
            last = st["last"] or {}
            step = last.get("steps_done", last.get("step", last.get("ckpt_step", last.get("start_step", ""))))
            c = fc["runs"].get(rid, {})
            rows.append(f"{st['status'] or '?':20s} {rid:60s} step={step} infra={c.get('infra_failed_attempt', 0)} "
                        f"suspect={c.get('suspect_sw_failure', 0)} crashed={c.get('crashed', 0)}")
        if not rows:
            return "(registry empty)"
        rows.append("per k: " + json.dumps(fc["per_k"], sort_keys=True))
        return "\n".join(rows)


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


def _read_from(path: Path, pos: int) -> str:
    try:
        with open(path, errors="replace") as f:
            f.seek(pos)
            return f.read()
    except OSError:
        return ""


def last_completed_step(run_dir: Path) -> int | None:
    """Last completed optimizer step of a run directory (trace.jsonl, else metrics.jsonl), 1-based count
    of steps done; None if nothing was logged."""
    for name, key in (("trace.jsonl", "step"), ("metrics.jsonl", "step")):
        p = Path(run_dir) / name
        try:
            with open(p, "rb") as f:
                f.seek(0, 2)
                size = f.tell()
                f.seek(max(0, size - 65536))
                lines = f.read().decode(errors="replace").splitlines()
        except OSError:
            continue
        for ln in reversed(lines):
            try:
                r = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if name == "metrics.jsonl" and r.get("kind") not in ("train", "checkpoint"):
                continue
            if key in r:
                return int(r[key]) + 1
    return None


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

"""Single project GPU lock (reproduction plan §2.2 R2(a); R2-LIVE v2 L1, PI ruling D077).

One exclusive lock, an ``fcntl.flock`` on ``runs/gpu_queue.lock``, serialises every project GPU job.
The kernel drops a flock when its holder dies, so a crashed job never leaves a stale lock. The holder
is described in ``<lock>.holder.json`` (label, pid, token, wait); the flock is the source of truth.

Entry points (``scripts/train.py``, ``scripts/runner.py``, ``scripts/evaluate.py``, the benchmark,
warm and repro drivers) call :func:`guard` BEFORE their first CUDA call. ``guard`` returns when

* the process descends from the current holder: env ``HYBRIDLC_GPU_LOCK_TOKEN`` equals the holder
  file's token while the lock is held (the holder exports its token, so children inherit it), or
* it acquired the lock itself (waiting at most ``HYBRIDLC_GPU_LOCK_WAIT`` seconds, default 0), or
* the labelled debug escape hatch ``HYBRIDLC_GPU_UNLOCKED=1`` is set (no registry row may be written;
  the trainer refuses it for exp_stage S2..S8 and for reproduction gate runs),

and otherwise exits with code :data:`EXIT_NO_LOCK` (78) and a one-line message, before any CUDA
context exists. ``HYBRIDLC_GPU_LOCK`` (another lock file) is honoured only inside tests.

``wait_exclusive`` (CLI ``--wait-exclusive``, env ``HYBRIDLC_GPU_WAIT_EXCLUSIVE=1``) also waits until
no other process of this repository holds a CUDA context (lock-less project jobs; CARLA is foreign).
While a lock is held, a :class:`hybridlc.training.gpumon.LockMonitor` samples the compute processes
every 5 s and records ``lock_ok`` per project CUDA process (R2-LIVE v2 L3).
"""
from __future__ import annotations

import contextlib
import fcntl
import json
import os
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Iterator

REPO = Path(__file__).resolve().parents[3]
DEFAULT_LOCK = REPO / "runs" / "gpu_queue.lock"
VENV_MARKER = str(REPO / ".venv")
EXIT_TIMEOUT = 75
EXIT_NO_LOCK = 78
TOKEN_ENV = "HYBRIDLC_GPU_LOCK_TOKEN"
UNLOCKED_ENV = "HYBRIDLC_GPU_UNLOCKED"
LOCK_ENV = "HYBRIDLC_GPU_LOCK"
WAIT_ENV = "HYBRIDLC_GPU_LOCK_WAIT"
WAIT_EXCLUSIVE_ENV = "HYBRIDLC_GPU_WAIT_EXCLUSIVE"
WAIT_S_ENV = "HYBRIDLC_GPU_LOCK_WAIT_S"   # set by guard() for the trainer's infra block
MONITOR_ENV = "HYBRIDLC_GPUMON"           # "0" disables the lock-time 5-s sampler (unit tests)

_HELD: dict[str, Any] = {}   # process-lifetime lock taken by guard()


def _in_tests() -> bool:
    return bool(os.environ.get("PYTEST_CURRENT_TEST") or os.environ.get("HYBRIDLC_TESTING") == "1")


def lock_path() -> Path:
    ov = os.environ.get(LOCK_ENV)
    if ov:
        if not _in_tests():
            raise RuntimeError(f"{LOCK_ENV} overrides the project GPU lock and is honoured only inside tests "
                               "(PYTEST_CURRENT_TEST or HYBRIDLC_TESTING=1)")
        return Path(ov)
    return DEFAULT_LOCK


def holder_path(lp: Path | None = None) -> Path:
    lp = lp or lock_path()
    return lp.with_name(lp.name + ".holder.json")


def read_holder(lp: Path | None = None) -> dict[str, Any]:
    try:
        return json.loads(holder_path(lp).read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def is_locked(lp: Path | None = None) -> bool:
    lp = lp or lock_path()
    if not lp.exists():
        return False
    fd = os.open(lp, os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    except BlockingIOError:
        return True
    finally:
        os.close(fd)


# ------------------------------------------------------------------------------------------ processes
def ppid_of(pid: int) -> int | None:
    try:
        return int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[1])
    except (OSError, ValueError, IndexError):
        return None


def ancestors(pid: int) -> list[int]:
    out, p = [], ppid_of(pid)
    while p and p > 1 and p not in out:
        out.append(p)
        p = ppid_of(p)
    return out


def descends_from(pid: int, ancestor: int) -> bool:
    return pid == ancestor or ancestor in ancestors(pid)


def own_tree() -> set[int]:
    """This process, its ancestors and its descendants (they are 'us', not foreign jobs)."""
    own = {os.getpid(), *ancestors(os.getpid())}
    todo = [os.getpid()]
    while todo:
        q = todo.pop()
        for task in Path(f"/proc/{q}/task").glob("*"):
            try:
                kids = [int(c) for c in (task / "children").read_text().split()]
            except (OSError, ValueError):
                kids = []
            for k in kids:
                if k not in own:
                    own.add(k)
                    todo.append(k)
    return own


def is_project_process(pid: int, name: str = "") -> bool:
    try:
        cmd = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except OSError:
        cmd = name
    try:  # `python -` heredocs launched from the repo show neither path in name/cmdline
        cwd = os.readlink(f"/proc/{pid}/cwd")
    except OSError:
        cwd = ""
    try:
        venv_env = f"VIRTUAL_ENV={VENV_MARKER}".encode() in Path(f"/proc/{pid}/environ").read_bytes()
    except OSError:
        venv_env = False
    return bool(VENV_MARKER in name or VENV_MARKER in cmd or str(REPO) in cmd
                or cwd == str(REPO) or cwd.startswith(str(REPO) + "/") or venv_env)


def compute_apps() -> list[dict[str, Any]]:
    """Processes holding a CUDA context (``nvidia-smi --query-compute-apps``)."""
    try:
        r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20)
    except Exception as e:  # pragma: no cover
        return [{"error": f"{type(e).__name__}: {e}"}]
    out = []
    for line in r.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            out.append({"pid": int(parts[0]), "name": parts[1], "used_mib": parts[2]})
        except ValueError:
            continue
    return out


def project_gpu_processes(exclude: set[int] | None = None) -> list[dict[str, Any]]:
    """Compute apps whose executable / command line / cwd / venv belongs to this repository."""
    exclude = exclude or set()
    out = []
    for a in compute_apps():
        if "error" in a:
            out.append(a)
            continue
        if a["pid"] in exclude or not is_project_process(a["pid"], a["name"]):
            continue
        try:
            cmd = Path(f"/proc/{a['pid']}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except OSError:
            cmd = a["name"]
        out.append({**a, "cmd": cmd[:300]})
    return out


def lock_ok(pid: int, holder: dict[str, Any] | None = None) -> bool:
    """A project CUDA process is compliant iff it is the lock holder or descends from it (/proc ancestry)."""
    h = read_holder() if holder is None else holder
    hp = h.get("pid")
    return bool(hp) and is_locked() and descends_from(int(pid), int(hp))


# ------------------------------------------------------------------------------------------ the lock
def _write_holder(lp: Path, info: dict[str, Any]) -> None:
    hp = holder_path(lp)
    tmp = hp.with_suffix(".tmp")
    tmp.write_text(json.dumps(info, indent=1))
    os.replace(tmp, hp)


def _acquire(label: str, timeout: float | None, poll_s: float, wait_exclusive: bool, exclusive_settle_s: float,
             verbose: bool) -> tuple[int, dict[str, Any]]:
    lp = lock_path()
    lp.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lp, os.O_RDWR | os.O_CREAT, 0o664)
    t0 = time.time()
    waited_for: list[str] = []
    last_note = 0.0
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if timeout is not None and time.time() - t0 >= timeout:
                    h = read_holder(lp)
                    raise TimeoutError(f"gpu_lock: '{label}' did not get {lp} within {timeout:.0f} s "
                                       f"(held by {h.get('label')} pid {h.get('pid')})")
                if time.time() - last_note > 60:
                    h = read_holder(lp)
                    note = f"lock held by {h.get('label')} pid {h.get('pid')}" if h else "lock held"
                    waited_for.append(note)
                    if verbose:
                        print(f"[gpu_lock] {time.strftime('%H:%M:%S')} '{label}' waiting: {note}", file=sys.stderr, flush=True)
                    last_note = time.time()
                time.sleep(min(poll_s, max(0.05, (timeout or poll_s) / 4)))
        if wait_exclusive:
            last_note, clear_since = 0.0, None
            while True:
                others = [o for o in project_gpu_processes(exclude=own_tree()) if "error" not in o]
                if not others:
                    clear_since = clear_since or time.time()
                    if time.time() - clear_since >= exclusive_settle_s:
                        break
                else:
                    clear_since = None
                    if time.time() - last_note > 60:
                        note = "project GPU processes without the lock: " + ", ".join(
                            f"{o['pid']}:{o['cmd'][:80]}" for o in others)
                        waited_for.append(note)
                        if verbose:
                            print(f"[gpu_lock] {time.strftime('%H:%M:%S')} '{label}' waiting: {note}", file=sys.stderr, flush=True)
                        last_note = time.time()
                if timeout is not None and time.time() - t0 > timeout:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                    raise TimeoutError(f"gpu_lock: '{label}' GPU not project-exclusive within {timeout:.0f} s")
                time.sleep(max(poll_s, 5.0))
    except BaseException:
        os.close(fd)
        raise
    info = {"label": label, "pid": os.getpid(), "host": socket.gethostname(), "acquired_at": time.time(),
            "acquired_at_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "wait_s": time.time() - t0, "lock": str(lp),
            "wait_exclusive": wait_exclusive, "waited_for": waited_for[-10:], "token": uuid.uuid4().hex,
            "test_override": bool(os.environ.get(LOCK_ENV))}
    _write_holder(lp, info)
    if verbose:
        print(f"[gpu_lock] {time.strftime('%H:%M:%S')} '{label}' acquired after {info['wait_s']:.1f} s", file=sys.stderr, flush=True)
    return fd, info


def _release(fd: int, info: dict[str, Any], verbose: bool) -> None:
    lp = Path(info["lock"])
    with contextlib.suppress(OSError):
        if read_holder(lp).get("token") == info["token"]:
            holder_path(lp).unlink()
    with contextlib.suppress(OSError):
        fcntl.flock(fd, fcntl.LOCK_UN)
    with contextlib.suppress(OSError):
        os.close(fd)
    if verbose:
        print(f"[gpu_lock] {time.strftime('%H:%M:%S')} '{info['label']}' released", file=sys.stderr, flush=True)


def _start_monitor(info: dict[str, Any]):
    if os.environ.get(MONITOR_ENV, "1") == "0":
        return None
    try:
        from hybridlc.training.gpumon import LockMonitor

        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in info["label"])[:80]
        out = Path(info["lock"]).parent / "_logs" / "gpumon_lock" / f"{time.strftime('%Y%m%dT%H%M%S')}_{safe}_{os.getpid()}.jsonl"
        mon = LockMonitor(out, holder_pid=os.getpid(), lock_file=Path(info["lock"]))
        mon.start()
        info["gpumon"] = str(out)
        return mon
    except Exception as e:  # noqa: BLE001 - the monitor must never stop a job; recorded
        info["gpumon_error"] = f"{type(e).__name__}: {e}"
        return None


@contextlib.contextmanager
def gpu_lock(label: str, timeout: float | None = None, poll_s: float = 2.0, wait_exclusive: bool = False,
             exclusive_settle_s: float = 0.0, verbose: bool = True, monitor: bool = True,
             reentrant: bool = True) -> Iterator[dict[str, Any]]:
    """Hold the project GPU lock for the duration of the ``with`` block (exports the token to children).

    Yields the holder info (``label, pid, wait_s, token, acquired_at, waited_for``). Raises
    ``TimeoutError`` if the lock (and, with ``wait_exclusive``, a project-free GPU) is not obtained in
    ``timeout`` s. Re-entrant for descendants of the current holder: a process that inherited the
    holder's token (e.g. ``gpu_lock.py run`` launched by a queue that holds the lock) runs inside the
    holder's lock instead of waiting for it (``info['mode'] == 'inherited'``).
    """
    if reentrant:
        inh = verify_token()
        if inh is not None:
            if verbose:
                print(f"[gpu_lock] {time.strftime('%H:%M:%S')} '{label}' runs inside the holder's lock "
                      f"({inh.get('label')} pid {inh.get('pid')})", file=sys.stderr, flush=True)
            yield {"mode": "inherited", **inh}
            return
    fd, info = _acquire(label, timeout, poll_s, wait_exclusive, exclusive_settle_s, verbose)
    prev = os.environ.get(TOKEN_ENV)
    os.environ[TOKEN_ENV] = info["token"]
    mon = _start_monitor(info) if monitor else None
    try:
        yield info
    finally:
        if mon is not None:
            mon.stop()
        if prev is None:
            os.environ.pop(TOKEN_ENV, None)
        else:
            os.environ[TOKEN_ENV] = prev
        _release(fd, info, verbose)


def verify_token() -> dict[str, Any] | None:
    """The holder info if this process inherited the current holder's token (lock held), else None."""
    tok = os.environ.get(TOKEN_ENV)
    if not tok:
        return None
    try:
        lp = lock_path()
    except RuntimeError:
        return None
    h = read_holder(lp)
    if h.get("token") != tok or not is_locked(lp):
        return None
    return {**h, "descends": descends_from(os.getpid(), int(h.get("pid", -1)))}


def guard(label: str, *, wait_s: float | None = None, wait_exclusive: bool | None = None,
          verbose: bool = True) -> dict[str, Any]:
    """Entry-point guard: call BEFORE the first CUDA call. Returns ``{mode, ...}`` with mode in
    ``inherited | acquired | unlocked | held``; exits the process with code 78 otherwise."""
    if _HELD.get("info") is not None:
        return {"mode": "held", **_HELD["info"]}
    if os.environ.get(UNLOCKED_ENV) == "1":
        if verbose:
            print(f"[gpu_lock] '{label}': {UNLOCKED_ENV}=1 (labelled debug run without the lock; no registry rows)",
                  file=sys.stderr, flush=True)
        return {"mode": "unlocked", "label": label}
    try:
        inh = verify_token()
    except RuntimeError as e:
        print(f"[gpu_lock] REFUSED '{label}': {e}", file=sys.stderr, flush=True)
        sys.exit(EXIT_NO_LOCK)
    if inh is not None:
        os.environ[WAIT_S_ENV] = str(inh.get("wait_s", 0.0))
        return {"mode": "inherited", **inh}
    if wait_s is None:
        wait_s = float(os.environ.get(WAIT_ENV, "0") or 0)
    if wait_exclusive is None:
        wait_exclusive = os.environ.get(WAIT_EXCLUSIVE_ENV) == "1"
    try:
        fd, info = _acquire(label, timeout=wait_s, poll_s=1.0, wait_exclusive=wait_exclusive,
                            exclusive_settle_s=0.0, verbose=verbose and wait_s > 0)
    except (TimeoutError, RuntimeError) as e:
        print(f"[gpu_lock] REFUSED '{label}' (exit {EXIT_NO_LOCK}): no GPU lock and no inherited token: {e}. "
              f"Start it under `python scripts/repro/gpu_lock.py run --label ... -- <cmd>` or set {WAIT_ENV}=<s>.",
              file=sys.stderr, flush=True)
        sys.exit(EXIT_NO_LOCK)
    os.environ[TOKEN_ENV] = info["token"]
    os.environ[WAIT_S_ENV] = str(info["wait_s"])
    info["monitor"] = _start_monitor(info)
    _HELD.update(fd=fd, info={k: v for k, v in info.items() if k != "monitor"}, monitor=info["monitor"])
    return {"mode": "acquired", **_HELD["info"]}


def release_held() -> None:
    """Release a lock taken by :func:`guard` (normally the process exit does it)."""
    if _HELD.get("info") is None:
        return
    mon = _HELD.get("monitor")
    if mon is not None:
        mon.stop()
    _release(_HELD["fd"], _HELD["info"], verbose=False)
    if os.environ.get(TOKEN_ENV) == _HELD["info"]["token"]:
        os.environ.pop(TOKEN_ENV, None)
    _HELD.clear()


def status() -> dict[str, Any]:
    return {"lock": str(lock_path()), "locked": is_locked(), "holder": {k: v for k, v in read_holder().items() if k != "token"},
            "project_gpu_processes": project_gpu_processes(exclude=own_tree())}


__all__ = ["gpu_lock", "guard", "verify_token", "release_held", "lock_path", "holder_path", "read_holder", "is_locked",
           "project_gpu_processes", "compute_apps", "lock_ok", "descends_from", "own_tree", "status", "EXIT_TIMEOUT",
           "EXIT_NO_LOCK", "TOKEN_ENV", "UNLOCKED_ENV", "LOCK_ENV", "WAIT_ENV", "WAIT_EXCLUSIVE_ENV", "WAIT_S_ENV"]

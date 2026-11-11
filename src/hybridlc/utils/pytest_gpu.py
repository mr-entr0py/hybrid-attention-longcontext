"""pytest plugin of the project test suite (loaded by ``tests/conftest.py``; ``-p hybridlc.utils.pytest_gpu``).

1. **Triton cache (PI ruling D069 / PR-12).** Test sessions compile into the diagnostics cache
   ``$REPO/.cache/triton-diag`` (extracted from the S0 archive by ``scripts/repro/make_diag_cache.sh``),
   never into the production cache: ``TRITON_CACHE_DIR`` is set at import unless it already points to a
   non-production directory (e.g. the cost benchmark's ``.cache/triton_bench``).
2. **GPU lock (R2-LIVE v2 L1, D077).** A session takes the project GPU lock once, after collection, iff
   a selected test is ``gpu``-marked or collection initialised CUDA (importing ``fla`` does); a session
   started under the lock holder (``gpu_lock.py run -- pytest ...``) uses the inherited token. If the lock
   is busy the session exits with code 78 (``HYBRIDLC_GPU_LOCK_WAIT=<s>`` waits instead). A session whose
   marker expression excludes ``gpu`` (``-m "not gpu"``) hides the GPU (``CUDA_VISIBLE_DEVICES=``) before
   collection, so it never creates a CUDA context and never locks; so does ``--collect-only``.
3. **Outcome record (SUITE-PF; batch-2 README rule 7).** With ``--junitxml=<x>.xml`` (or
   ``HYBRIDLC_SUITE_OUTCOMES=<file>``) the session writes ``<x>_outcomes.json``: every test's outcome
   (passed, failed, skipped, xfailed, xpassed), the xfail reasons, and the values a test recorded with
   ``record_property`` (the registered-criterion tests record their measured values). An XPASS of a
   ``strict=False`` marker is recorded and reported; it never fails the session.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
PRODUCTION_CACHE = REPO / ".cache" / "triton"
DIAG_CACHE = REPO / ".cache" / "triton-diag"


def _set_triton_cache() -> None:
    cur = os.environ.get("TRITON_CACHE_DIR")
    if cur and Path(cur).resolve() != PRODUCTION_CACHE.resolve():
        return
    os.environ["TRITON_CACHE_DIR"] = str(DIAG_CACHE)


_set_triton_cache()

_STATE: dict = {"locked": None, "outcomes": {}, "t0": time.time()}


def _excludes_gpu(markexpr: str) -> bool:
    m = " ".join((markexpr or "").split())
    return "not gpu" in m


def pytest_configure(config: pytest.Config) -> None:
    # ``--collect-only`` (e.g. scripts/repro/list_suite_markers.py) never runs a test: hide the GPU so that
    # importing fla at collection creates no CUDA context, and take no lock.
    if _excludes_gpu(config.getoption("markexpr", "") or "") or config.getoption("collectonly", False):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        _STATE["gpu_hidden"] = True


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(session: pytest.Session, config: pytest.Config, items: list) -> None:
    _STATE["need_gpu"] = any(it.get_closest_marker("gpu") is not None for it in items)


def _cuda_initialized() -> bool:
    import sys

    torch = sys.modules.get("torch")
    try:
        return bool(torch is not None and torch.cuda.is_initialized())
    except Exception:  # noqa: BLE001
        return False


def pytest_collection_finish(session: pytest.Session) -> None:
    if _STATE.get("gpu_hidden"):
        _STATE["locked"] = {"mode": "gpu-hidden"}
        return
    if not (_STATE.get("need_gpu") or _cuda_initialized()):
        _STATE["locked"] = {"mode": "cpu-only"}
        return
    from hybridlc.utils.gpu_lock import EXIT_NO_LOCK, WAIT_ENV, _acquire, _HELD, _start_monitor, verify_token

    inh = verify_token()
    if inh is not None:
        _STATE["locked"] = {"mode": "inherited", "label": inh.get("label"), "pid": inh.get("pid")}
        return
    wait = float(os.environ.get(WAIT_ENV, "0") or 0)
    try:
        fd, info = _acquire("pytest", timeout=wait, poll_s=1.0, wait_exclusive=False, exclusive_settle_s=0.0,
                            verbose=wait > 0)
    except (TimeoutError, RuntimeError) as e:
        pytest.exit(f"[gpu_lock] pytest session needs the GPU lock (gpu-marked tests or CUDA initialised at "
                    f"collection) and could not get it: {e}. Run it under scripts/repro/gpu_lock.py run.",
                    returncode=EXIT_NO_LOCK)
    os.environ["HYBRIDLC_GPU_LOCK_TOKEN"] = info["token"]
    mon = _start_monitor(info)
    _HELD.update(fd=fd, info=info, monitor=mon)
    _STATE["locked"] = {"mode": "acquired", "label": "pytest", "pid": os.getpid()}


def pytest_report_header(config: pytest.Config) -> list[str]:
    return [f"TRITON_CACHE_DIR={os.environ.get('TRITON_CACHE_DIR')}"]


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call):
    outcome = yield
    rep = outcome.get_result()
    rec = _STATE["outcomes"].setdefault(rep.nodeid, {"outcome": None})
    xf = item.get_closest_marker("xfail")
    if rep.when == "setup" and rep.skipped:
        rec["outcome"] = "xfailed" if hasattr(rep, "wasxfail") else "skipped"
        rec["reason"] = str(rep.longrepr[-1]) if isinstance(rep.longrepr, tuple) else str(rep.longrepr)
    elif rep.when == "call":
        if hasattr(rep, "wasxfail"):
            rec["outcome"] = "xfailed" if rep.skipped else "xpassed"
            rec["xfail_reason"] = rep.wasxfail
        else:
            rec["outcome"] = rep.outcome
        rec["duration_s"] = round(rep.duration, 3)
    elif rep.failed and rec.get("outcome") in (None, "passed"):
        rec["outcome"] = "error"
    if xf is not None:
        rec["xfail_marker"] = {"reason": xf.kwargs.get("reason"), "strict": xf.kwargs.get("strict")}
    if rep.user_properties:
        rec["recorded"] = {k: v for k, v in rep.user_properties}


def _outcomes_path(config: pytest.Config) -> Path | None:
    env = os.environ.get("HYBRIDLC_SUITE_OUTCOMES")
    if env:
        return Path(env)
    xml = getattr(config.option, "xmlpath", None)
    if xml:
        p = Path(xml)
        return p.with_name(p.stem + "_outcomes.json")
    return None


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    out = _outcomes_path(session.config)
    if out is not None:
        counts: dict[str, int] = {}
        for r in _STATE["outcomes"].values():
            counts[str(r.get("outcome"))] = counts.get(str(r.get("outcome")), 0) + 1
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"ts": time.time(), "exitstatus": int(exitstatus), "counts": counts,
                                   "lock": _STATE.get("locked"), "triton_cache_dir": os.environ.get("TRITON_CACHE_DIR"),
                                   "tests": _STATE["outcomes"]}, indent=1, default=str))
    from hybridlc.utils.gpu_lock import release_held

    release_held()

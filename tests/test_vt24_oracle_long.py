"""VT-24 (R3e, DEV-01): oracle answerability of the 16K/32K item files (CPU only).

Pre-registration: ``experiments/repro/R3/prereg.yaml`` (copied from reproduction_plan.md §2.3 R3e).
Reference: ``hybridlc.eval.devtools.InductionLM(order=6, p_hit=0.9)`` scored through
``eval.scorer.score_file`` on every R1 / R-load / R3 / R-dist file at 16K and 32K in
test/CS, test/MS, dev/DG, dev/FD; M-hop files there are the negative control.

Pass (frozen): every retrieval query EM_tf = 1 and logp_gold = 4·ln 0.9; M-hop EM_tf < 0.5.
The runner (``scripts/repro/r3_vt24_oracle.py``) writes raw rows to
``experiments/repro/R3/vt24/run_<ts>/``.  Run: ``pytest -q -s tests/test_vt24_oracle_long.py``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "repro"))


@pytest.mark.slow
def test_vt24_inductionlm_answers_every_retrieval_item() -> None:
    import r3_vt24_oracle as V

    s = V.main(["--workers", "12"])
    assert s["n_retrieval_files"] > 0 and s["n_mhop_files"] > 0
    assert not s["errors"], s["errors"]
    assert not s["retrieval_files_failing"], (s["retrieval_files_failing"], s["retrieval_queries_failing"])
    assert not s["mhop_files_failing"], s["mhop_files_failing"]
    assert s["verdict"] == "PASS"

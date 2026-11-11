"""List the SUITE-PF registered-criterion tests and relaxed regression guards (PI rulings D059, D086(d); T18).

Collects the test suite (``--collect-only``, GPU hidden, no lock, nothing runs) and prints:
  * every test carrying an ``xfail`` marker whose reason starts "FAIL-as-registered" (expected: 7, one per
    VT-03 GDN fp32, VT-05, VT-07, VT-15, VT-21, VT-22, VT-23), with the RR ID and ruling named in the reason;
  * every test case carrying the ``relaxed_guard`` marker (expected: 13 in the 2026-11-10 tree: VT-03 fp32
    linear mixers x4, VT-05 bf16 x4, VT-07 x4, VT-15 x1).
Exit 0 iff the counts are exactly 7 and 13 and every xfail reason names an RR ID and a D-row.

  python scripts/repro/list_suite_markers.py [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EXPECTED_REGISTERED = 7
EXPECTED_RELAXED = 13
REASON_RE = re.compile(r"^FAIL-as-registered; accepted residual (RR-[A-Za-z0-9-]+); PI ruling (D\d{3}); criterion unchanged$")


class _Collector:
    def __init__(self) -> None:
        self.registered: list[dict] = []
        self.relaxed: list[dict] = []

    def pytest_collection_modifyitems(self, session, config, items) -> None:  # noqa: ANN001
        for it in items:
            xf = it.get_closest_marker("xfail")
            if xf is not None and str(xf.kwargs.get("reason", "")).startswith("FAIL-as-registered"):
                reason = xf.kwargs.get("reason")
                m = REASON_RE.match(reason or "")
                self.registered.append({"nodeid": it.nodeid, "reason": reason, "strict": xf.kwargs.get("strict"),
                                        "raises": getattr(xf.kwargs.get("raises"), "__name__", None),
                                        "rr": m.group(1) if m else None, "row": m.group(2) if m else None})
            rg = it.get_closest_marker("relaxed_guard")
            if rg is not None:
                self.relaxed.append({"nodeid": it.nodeid, "reason": rg.kwargs.get("reason")})


def collect() -> _Collector:
    import pytest

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("HYBRIDLC_GPUMON", "0")
    col = _Collector()
    rc = pytest.main(["--collect-only", "-q", "-p", "no:cacheprovider", str(REPO / "tests")], plugins=[col])
    if rc not in (0, 5):
        raise SystemExit(f"collection failed (pytest exit {rc})")
    return col


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    col = collect()
    print(f"\n== registered-criterion tests (xfail strict=False): {len(col.registered)}")
    for r in col.registered:
        print(f"  {r['nodeid']}  [{r['rr']}, {r['row']}, strict={r['strict']}, raises={r['raises']}]")
    print(f"== relaxed regression guards: {len(col.relaxed)} test cases")
    for r in col.relaxed:
        print(f"  {r['nodeid']}")
    ok = (len(col.registered) == EXPECTED_REGISTERED and len(col.relaxed) == EXPECTED_RELAXED
          and all(r["rr"] and r["row"] and r["strict"] is False for r in col.registered))
    out = {"registered": col.registered, "relaxed_guards": col.relaxed, "n_registered": len(col.registered),
           "n_relaxed": len(col.relaxed), "ok": ok}
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(out, indent=1))
    print("OK" if ok else f"MISMATCH: expected {EXPECTED_REGISTERED} registered / {EXPECTED_RELAXED} relaxed")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

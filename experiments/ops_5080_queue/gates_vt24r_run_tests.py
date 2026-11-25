#!/usr/bin/env python3
"""VT-24r K4 + K5a + K6 runner (CPU only; CUDA hidden so it can never touch the O4 window).

Runs tests/test_vt12_*.py once (K4 = full VT-12 suite on the regenerated items) and derives K5a/K6 from the
test_k5a_* / test_k6_* results in the same run. Writes experiments/repro/VT-24r/{k4_result.json,k5a_k6_result.json}
then `vt24r_check.py summary`. Lives outside scripts/ on purpose: touching code paths during O4 would make
scripts/train.py refuse the remaining O4-A segments (dirty tree).
"""
import json, os, subprocess, sys, time, xml.etree.ElementTree as ET
from pathlib import Path

REPO = Path.home() / "Downloads" / "outeropt"
OUT = REPO / "experiments" / "repro" / "VT-24r"
env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONHASHSEED="0", CUBLAS_WORKSPACE_CONFIG=":4096:8")
xml = OUT / "k4_pytest.xml"
files = sorted(str(p.relative_to(REPO)) for p in (REPO / "tests").glob("test_vt12_*.py"))
t0 = time.time()
r = subprocess.run(["nice", "-n", "19", str(REPO / ".venv/bin/python"), "-I", "-m", "pytest", "-q", "-p", "no:cacheprovider",
                    f"--junitxml={xml}", *files], cwd=REPO, env=env, capture_output=True, text=True)
(OUT / "k4_pytest.log").write_text(r.stdout[-20000:] + "\n" + r.stderr[-5000:])
cases = []
for tc in ET.parse(xml).getroot().iter("testcase"):
    st = "failed" if tc.find("failure") is not None or tc.find("error") is not None else "skipped" if tc.find("skipped") is not None else "passed"
    cases.append((tc.get("name"), st))
def grp(prefix):
    sel = [(n, s) for n, s in cases if n.startswith(prefix)]
    return {"n": len(sel), "passed": sum(s == "passed" for _, s in sel), "ok": bool(sel) and all(s == "passed" for _, s in sel)}
k4 = {"pass": r.returncode == 0, "n_tests": len(cases), "passed": sum(s == "passed" for _, s in cases),
      "failed": [n for n, s in cases if s == "failed"], "skipped": [n for n, s in cases if s == "skipped"],
      "files": files, "wall_s": round(time.time() - t0, 1), "cpu_only": True}
k56 = {"K5a": grp("test_k5a")["ok"], "K6": grp("test_k6")["ok"], "k5a": grp("test_k5a"), "k6": grp("test_k6")}
(OUT / "k4_result.json").write_text(json.dumps(k4, indent=1))
(OUT / "k5a_k6_result.json").write_text(json.dumps(k56, indent=1))
print(json.dumps({"k4": k4, "k5a_k6": k56})[:1500])
s = subprocess.run(["nice", "-n", "19", str(REPO / ".venv/bin/python"), "scripts/repro/vt24r_check.py", "summary"], cwd=REPO,
                   env=env, capture_output=True, text=True)
print(s.stdout[-1200:], s.stderr[-300:])

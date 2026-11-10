"""VT-23 post-verdict isolation diagnostics (report-only; the frozen verdict is not re-computed).

Re-runs the reference loop with single components swapped and compares each trace with the trainer
trace of the verdict attempt, plus a reference-vs-reference rounding control:
  D1 --loss chunked_ce --clip torch   (production loss fn + torch clipping, our optimizer/schedule/groups/assembly)
  D2 --loss chunked_ce                (production loss fn only)
  D3 --ce-chunk 2048                  (independent loop, CE summed in 2048-token chunks: a pure rounding change)
    python scripts/repro/r3_vt23_diagnose.py <attempt_dir>
Output: <attempt_dir>/diag_<name>/ and experiments/repro/R3/vt23/vt23_diagnosis.json
"""

# --- R2-LIVE v2 L1 entry-point guard (PI ruling D077): take the project GPU lock, or verify the
# holder's inherited token, BEFORE any import that can create a CUDA context; otherwise exit 78.
import sys as _sys_guard  # noqa: E402
from pathlib import Path as _Path_guard  # noqa: E402
if __name__ == "__main__":
    _sys_guard.path.insert(0, str(_Path_guard(__file__).resolve().parents[2] / "src"))
    from hybridlc.utils.gpu_lock import guard as _gpu_guard  # noqa: E402
    _gpu_guard(_Path_guard(__file__).name)
# ---
import json, os, struct, subprocess, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
att = Path(sys.argv[1]).resolve()
env = {**os.environ, "PYTHONHASHSEED": "0", "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
hexf = lambda h: struct.unpack(">d", bytes.fromhex(h))[0]
rid = next((att / "runs").iterdir()).name
tr = {}
for l in open(att / "runs" / rid / "trace.jsonl"):
    r = json.loads(l); tr[r["step"]] = (hexf(r["loss_hex"]), hexf(r["gnorm_hex"]))
def load(d): return {json.loads(l)["step"]: (json.loads(l)["loss"], json.loads(l)["gnorm"]) for l in open(d / "ref_trace.jsonl")}
def cmp(a, b):
    out = []
    for s in range(50):
        (la, ga), (lb, gb) = a[s], b[s]
        out.append({"step": s + 1, "loss_rel": abs(la - lb) / abs(lb), "gnorm_rel": abs(ga - gb) / abs(gb), "bitwise": la == lb and ga == gb})
    return out
res = {}
for name, args in [("D1_chunkedce_torchclip", ["--loss", "chunked_ce", "--clip", "torch"]),
                   ("D2_chunkedce", ["--loss", "chunked_ce"]), ("D3_cechunk2048", ["--ce-chunk", "2048"])]:
    d = att / f"diag_{name}"
    if not (d / "ref_trace.jsonl").exists():
        with open(att / f"diag_{name}.log", "w") as f:
            rc = subprocess.run([sys.executable, str(REPO / "scripts/repro/r3_vt23_reference_loop.py"), "--out", str(d), *args],
                                stdout=f, stderr=subprocess.STDOUT, cwd=REPO, env=env).returncode
        if rc: res[name] = {"exit": rc}; continue
    t = load(d)
    res[name] = {"vs_trainer": cmp(t, tr)}
    if name.startswith("D3"):
        res[name]["vs_verdict_reference"] = cmp(t, load(att / "ref"))
def summ(rows):
    pick = lambda lo, hi: max(max(r["loss_rel"], r["gnorm_rel"]) for r in rows if lo <= r["step"] <= hi)
    return {"all_bitwise": all(r["bitwise"] for r in rows), "n_bitwise": sum(r["bitwise"] for r in rows),
            "max_rel_1": pick(1, 1), "max_rel_2_10": pick(2, 10), "max_rel_11_50": pick(11, 50),
            "first_nonbitwise_step": next((r["step"] for r in rows if not r["bitwise"]), None)}
out = {k: {kk: summ(vv) for kk, vv in v.items() if isinstance(vv, list)} | ({"exit": v["exit"]} if "exit" in v else {}) for k, v in res.items()}
(REPO / "experiments/repro/R3/vt23/vt23_diagnosis.json").write_text(json.dumps({"attempt": str(att), "summary": out, "detail": res}, indent=1))
print(json.dumps(out, indent=1))

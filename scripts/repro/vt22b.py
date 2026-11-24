"""VT-22b: long-context fp32 equivalence of our C0 and C12 to their references (prereg experiments/prereg/VT-22b.v2.yaml).

    python scripts/repro/gpu_lock.py run --label VT-22b --wait-exclusive -- python scripts/repro/vt22b.py [--T 16384 32768]

Arms per T in {16384, 32768}: C0_fp32 (ours vs fla GatedDeltaNetForCausalLM), C0_bf16 (both bf16), C0_self_drift (each implementation's
bf16 vs its own fp32), C12_fp32 (ours through the fp32 reference attention path R-ATT vs Qwen3ForCausalLM, SDPA EFFICIENT_ATTENTION fp32).
Criteria: F1 C0 fp32 rel RMS <= 1e-3; F2 C0 fp32 band ratio <= 2; B1 C0 bf16 rel RMS <= 1.5 x max(self-drifts); B2 band ratio <= 2;
F3 C12 fp32 rel RMS <= 1e-3; F4 C12 fp32 band ratio <= 2. All at each T. BLOCKED (never PASS) if an arm cannot run under the cap.
Output: experiments/repro/VT-22b/{vt22b.jsonl, _meta.json}. Reuses the VT-22 helpers (tests/test_vt22_long_transplant.py) unchanged.
"""
# --- R2-LIVE v2 L1 entry-point guard: take the project GPU lock (or verify the holder's token) BEFORE any CUDA context.
import sys as _sys_guard  # noqa: E402
from pathlib import Path as _Path_guard  # noqa: E402
if __name__ == "__main__":
    _sys_guard.path.insert(0, str(_Path_guard(__file__).resolve().parents[2] / "src"))
    from hybridlc.utils.gpu_lock import guard as _gpu_guard  # noqa: E402
    _gpu_guard(_Path_guard(__file__).name)
# ---
import argparse
import contextlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tests"))
sys.path.insert(0, str(REPO / "scripts" / "repro"))
import r3_common as RC  # noqa: E402
import test_vt22_long_transplant as V  # noqa: E402

OUT = REPO / "experiments" / "repro" / "VT-22b"
PRED = {"F1_16K": 2.6e-4, "F2_16K": 1.10, "B1_16K_cross": 2.0e-2, "B2_16K": 1.06, "B2_32K": 1.03}  # stated before the run


def log(m):
    print(f"[vt22b {time.strftime('%H:%M:%S')}] {m}", flush=True)


@contextlib.contextmanager
def seed1():
    """VT-22 helpers seed their reference with torch.manual_seed(0); VT-22b needs seed 1 (prereg): force it during construction."""
    orig = torch.manual_seed
    torch.manual_seed = lambda s: orig(1)
    try:
        yield
    finally:
        torch.manual_seed = orig


def ids_for(T, vocab, dev):
    key = int.from_bytes(b"VT-22b", "big") % (2**31)  # stable int for the string key (python hash() is not stable across processes)
    rng = np.random.default_rng(np.random.SeedSequence(31337, spawn_key=(key, T)))
    return torch.from_numpy(rng.integers(8, vocab, size=(1, T))).long().to(dev)


def fp32_logits_ours(ours, ids, P):
    return V._ours_logits(ours, ids, P, False)


def run_c0(T, dev, cfg, out):
    from hybridlc.models.build import build_model
    with seed1():
        ref = V._fla_gdn_lm(cfg)
    ours = build_model(cfg, seed_index=0).eval()
    V._transplant_fla(ours, ref)
    ours, ref = ours.to(dev), ref.to(dev)
    ids = ids_for(T, cfg.vocab_size, dev)
    P = V.sample_positions(T)
    with torch.no_grad():
        fla32 = V._ref_hidden_logits(ref, ids, P, False, "fla_gdn").cpu()
        ours32 = V._ours_logits(ours, ids, P, False).cpu()
        fla16 = V._ref_hidden_logits(ref, ids, P, True, "fla_gdn").cpu()
        ours16 = V._ours_logits(ours, ids, P, True).cpu()
    rows = {"C0_fp32": V._metrics(ours32, fla32, P, T), "C0_bf16": V._metrics(ours16, fla16, P, T),
            "C0_self_drift_fla": V._metrics(fla16, fla32, P, T), "C0_self_drift_ours": V._metrics(ours16, ours32, P, T)}
    del ours, ref
    torch.cuda.empty_cache()
    return rows


def run_c12(T, dev, cfg, out):
    from hybridlc.models.attention import fp32_reference_attention
    from hybridlc.models.build import build_model
    from torch.nn.attention import SDPBackend, sdpa_kernel
    with seed1():
        q = V._qwen3(cfg, "sdpa")
    ours = build_model(cfg, seed_index=0).eval()
    V._transplant_qwen(ours, q)
    ours, q = ours.to(dev), q.to(dev)
    ids = ids_for(T, cfg.vocab_size, dev)
    P = V.sample_positions(T)
    mode_used = None
    with torch.no_grad():
        with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
            ref32 = V._ref_hidden_logits(q, ids, P, False, "fla_gdn").cpu()  # family flag only selects the bf16 backend; fp32 path here
        for mode in ("efficient", "tiled"):  # R-ATT: efficient if the backend can run fp32 on this GPU, else the tiled reference mode
            try:
                with fp32_reference_attention(mode):
                    ours32 = V._ours_logits(ours, ids, P, False).cpu()
                mode_used = mode
                break
            except RuntimeError as e:
                log(f"R-ATT mode {mode} raised: {str(e)[:120]}")
    del ours, q
    torch.cuda.empty_cache()
    if mode_used is None:
        return {"C12_fp32": {"blocked": "no R-ATT mode could run"}}
    r = V._metrics(ours32, ref32, P, T)
    r["ratt_mode"] = mode_used
    return {"C12_fp32": r}


def judge(rows_by_T):
    crit = {}
    for T, rows in rows_by_T.items():
        c0, c12 = rows.get("C0_fp32"), rows.get("C12_fp32")
        b = rows.get("C0_bf16")
        drift = max(rows["C0_self_drift_fla"]["rel_rms"], rows["C0_self_drift_ours"]["rel_rms"])
        crit[f"F1@{T}"] = c0["rel_rms"] <= 1e-3
        crit[f"F2@{T}"] = c0["band_ratio"] <= 2
        crit[f"B1@{T}"] = b["rel_rms"] <= 1.5 * drift
        crit[f"B2@{T}"] = b["band_ratio"] <= 2
        if c12 is None or "blocked" in c12:
            crit[f"F3@{T}"] = crit[f"F4@{T}"] = None
        else:
            crit[f"F3@{T}"] = c12["rel_rms"] <= 1e-3
            crit[f"F4@{T}"] = c12["band_ratio"] <= 2
    return crit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, nargs="+", default=[16384, 32768])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    from hybridlc.models.config import ModelConfig
    from hybridlc.utils.gpu import claim
    dev = claim(0.44)
    meta = RC.run_meta("VT-22b", {"T": a.T, "predictions": PRED, "prereg": "experiments/prereg/VT-22b.v2.yaml"})
    rows_by_T = {}
    with open(OUT / "vt22b.jsonl", "w") as f:
        for T in a.T:
            rows = {}
            for name, fn, cname in (("C0", run_c0, "C0"), ("C12", run_c12, "C12")):
                try:
                    r = fn(T, dev, ModelConfig.for_scale("D384", cname), OUT)
                except torch.OutOfMemoryError as e:
                    r = {f"{name}_fp32": {"blocked": f"OOM under cap: {str(e)[:120]}"}}
                rows.update(r)
                for k, v in r.items():
                    f.write(json.dumps({"T": T, "arm": k, **v}, default=str) + "\n")
                    f.flush()
                    log(f"T={T} {k}: " + ", ".join(f"{x}={v[x]:.3e}" for x in ("rel_rms", "band_ratio") if x in v) if "rel_rms" in v else f"T={T} {k}: {v}")
            rows_by_T[T] = rows
    blocked = any(isinstance(v, dict) and "blocked" in v for rows in rows_by_T.values() for v in rows.values())
    crit = judge(rows_by_T) if not any("C0_bf16" not in r for r in rows_by_T.values()) else {}
    verdict = "BLOCKED" if blocked or any(v is None for v in crit.values()) else ("PASS" if all(crit.values()) else "FAIL")
    (OUT / "_meta.json").write_text(json.dumps({**RC.finish_meta(meta), "criteria": crit, "verdict": verdict}, indent=1, default=str))
    log(f"verdict {verdict}: {crit}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())

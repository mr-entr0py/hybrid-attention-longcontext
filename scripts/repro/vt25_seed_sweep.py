"""VT-25 10-seed MQAR sweep (PI ruling D075 / PR-18; SUITE-PF vt25_pin and outcome O-d).

Runs the VT-25 GDN check (matched bf16 casts, Zoology <-> ours by weight copy) on the MQAR batch built with the
global torch RNG pinned to each of 10 seeds, and reports each seed's worst-tensor GDN gradient error (max over the
GDN widths). SUITE-PF O-d: x_sweep = max over the 10 seeds; with the pinned test's x,
max(x, x_sweep) <= 2e-2 consistent; (2e-2, 4e-2] -> RR-VT25 by PI row (no block); > 4e-2 -> candidate regression.
The VT-25 verdict (registered run 1.82e-2) stands in every tier; this script applies no tier, it only measures.

Run (GPU, under the lock):
  python scripts/repro/gpu_lock.py run --label VT25-sweep -- python scripts/repro/vt25_seed_sweep.py
Output: experiments/repro/SUITE-PF/vt25_seed_sweep.json (never overwritten: .rN suffix).
"""
from __future__ import annotations

# --- R2-LIVE v2 L1 entry-point guard (PI ruling D077): take the project GPU lock, or verify the
# holder's inherited token, BEFORE any import that can create a CUDA context; otherwise exit 78.
import sys as _sys_guard  # noqa: E402
from pathlib import Path as _Path_guard  # noqa: E402
if __name__ == "__main__":
    _sys_guard.path.insert(0, str(_Path_guard(__file__).resolve().parents[2] / "src"))
    from hybridlc.utils.gpu_lock import guard as _gpu_guard  # noqa: E402
    _gpu_guard(_Path_guard(__file__).name)
# ---

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
#: the 10 pinned sweep seeds (fixed in code before the sweep has ever run)
SWEEP_SEEDS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)


def _load_vt25():
    spec = importlib.util.spec_from_file_location("vt25", REPO / "tests" / "test_vt25_zoology_transplant.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _out_path(d: Path) -> Path:
    p, n = d / "vt25_seed_sweep.json", 0
    while p.exists():
        n += 1
        p = d / f"vt25_seed_sweep.r{n}.json"
    return p


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seeds", type=int, nargs="*", default=list(SWEEP_SEEDS))
    ap.add_argument("--out-dir", default=str(REPO / "experiments" / "repro" / "SUITE-PF"))
    a = ap.parse_args(argv)
    import torch

    from hybridlc.utils.gpu import claim
    from hybridlc.utils.provenance import env_info

    T = _load_vt25()
    zmod = T._setup()
    dev = claim(0.44)
    t0 = time.time()
    rows = []
    models = []
    for d in T.GDN_WIDTHS:
        zcfg = T._cfg(zmod, "gated_delta_net", d)
        zoo = T._lm(zcfg, 31337).to(dev).eval()
        ours = T._lm(T._swap(zcfg, "our_mixers.OurGdnZoology"), 7).to(dev).eval()
        m = T.transplant(zoo, ours, "gated_delta_net", d)
        models.append((d, zoo, ours, m))
    for s in a.seeds:
        x, y = T._batches(dev, pin_seed=s)["mqar"]
        per_width = {}
        for d, zoo, ours, m in models:
            zl, zg = T._fwd_bwd(zoo, x, y)
            with T.zoology_bf16_casts(ours):
                ol, og = T._fwd_bwd(ours, x, y)
            r = T._compare("gated_delta_net", d, m, zl, zg, ol, og, T.TOL["gdn_logits"], T.TOL["gdn_grad"])
            per_width[str(d)] = {"grad_worst": r["grad_worst"], "grad_worst_name": r["grad_worst_name"],
                                 "logits_rel_rms": r["logits_rel_rms"]}
        worst = max(v["grad_worst"] for v in per_width.values())
        rows.append({"seed": s, "gdn_grad_worst": worst, "per_width": per_width})
        print(f"VT-25 sweep seed {s}: worst-tensor GDN grad rel RMS {worst:.3e}", flush=True)
    out = {"item": "SUITE-PF VT-25 seed sweep", "ruling": "D075", "seeds": list(a.seeds),
           "x_sweep": max(r["gdn_grad_worst"] for r in rows), "min_margin_to_2e-2": 2e-2 - max(r["gdn_grad_worst"] for r in rows),
           "rows": rows, "pinned_test_seed": T.MQAR_PIN_SEED, "elapsed_s": round(time.time() - t0, 1),
           "env": env_info(), "torch": torch.__version__}
    od = Path(a.out_dir)
    od.mkdir(parents=True, exist_ok=True)
    p = _out_path(od)
    p.write_text(json.dumps(out, indent=1, default=str))
    print(f"x_sweep = {out['x_sweep']:.3e} -> {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

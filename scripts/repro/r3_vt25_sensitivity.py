"""VT-25 sensitivity check (report-only, not part of the pre-registered verdict).

Plants the defect classes named in reproduction_plan.md §2.3 R3a into the weight copy and records
the logits rel RMS, to show the 1e-2 / 1e-5 thresholds can detect them:
  GDN d128: q/k projections swapped; conv taps reversed (q, k, v); head order of v_proj swapped;
  attention d64: q/k slices of Wqkv swapped; v head halves swapped.
    python scripts/repro/r3_vt25_sensitivity.py
Output: experiments/repro/R3/vt25/vt25_sensitivity.jsonl
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

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tests"))
sys.path.insert(0, str(REPO / "scripts" / "repro"))
import r3_common as RC  # noqa: E402
import test_vt25_zoology_transplant as T25  # noqa: E402


def main() -> None:
    mod = T25._setup()
    from hybridlc.utils.gpu import claim

    dev = claim(0.44)
    wr = RC.RawWriter("vt25", "vt25_sensitivity", RC.run_meta("VT-25-sensitivity"))
    x, y = T25._batches(dev)["mqar"]
    P = T25.MIXER + "inner."
    for family, d, path, defects in (
            ("gated_delta_net", 128, "our_mixers.OurGdnZoology",
             ["none", "swap_qk", "conv_taps_reversed", "v_head_order"]),
            ("attention", 64, "our_mixers.OurFgZoology", ["none", "swap_qk", "v_head_order"])):
        zcfg = T25._cfg(mod, family, d)
        zoo = T25._lm(zcfg, 31337).to(dev).eval()
        zl, _ = T25._fwd_bwd(zoo, x, y)
        for defect in defects:
            ours = T25._lm(T25._swap(zcfg, path), 7).to(dev).eval()
            T25.transplant(zoo, ours, family, d)
            p = dict(ours.named_parameters())
            with torch.no_grad():
                if defect == "swap_qk":
                    q, k = p[P + "q_proj.weight"].clone(), p[P + "k_proj.weight"].clone()
                    p[P + "q_proj.weight"].copy_(k)
                    p[P + "k_proj.weight"].copy_(q)
                elif defect == "conv_taps_reversed":
                    for c in ("q", "k", "v"):
                        w = p[P + f"{c}_conv1d.weight"]
                        w.copy_(w.flip(-1))
                elif defect == "v_head_order":
                    w = p[P + "v_proj.weight"]
                    h = w.shape[0] // 2
                    w.copy_(torch.cat([w[h:], w[:h]], 0))
            if family == "gated_delta_net":
                with T25.zoology_bf16_casts(ours):
                    ol, _ = T25._fwd_bwd(ours, x, y)
                tol = T25.TOL["gdn_logits"]
            else:
                ol, _ = T25._fwd_bwd(ours, x, y)
                tol = T25.TOL["attn_logits"]
            e = RC.rel_rms(ol, zl)
            row = {"family": family, "d": d, "defect": defect, "logits_rel_rms": e, "threshold": tol,
                   "detected": bool(e > tol)}
            wr.write(row)
            print(row)
    wr.close({})


if __name__ == "__main__":
    main()

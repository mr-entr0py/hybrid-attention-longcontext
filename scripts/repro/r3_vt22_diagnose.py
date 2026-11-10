"""VT-22 post-verdict diagnosis (report-only; the frozen verdict is not re-computed).

C0 ↔ fla GatedDeltaNetForCausalLM at T=16384: separates the bf16 error of each side by comparing
ours-bf16, ours-fp32 and fla-bf16 against fla-fp32 (same weights, same ids as the VT-22 run).
    python scripts/repro/r3_vt22_diagnose.py [--T 16384]
Output: experiments/repro/R3/vt22/vt22_diagnosis.jsonl
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
import argparse, sys
from pathlib import Path
import torch
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tests")); sys.path.insert(0, str(REPO / "scripts" / "repro"))
import r3_common as RC
import test_vt22_long_transplant as V

ap = argparse.ArgumentParser(); ap.add_argument("--T", type=int, default=16384); a = ap.parse_args()
from hybridlc.models.build import build_model
from hybridlc.models.config import ModelConfig
from hybridlc.utils.gpu import claim
dev = claim(0.44)
wr = RC.RawWriter("vt22", "vt22_diagnosis", RC.run_meta("VT-22-diagnosis", {"T": a.T}))
cfg = ModelConfig.for_scale("D384", "C0")
ours = build_model(cfg, seed_index=0).eval(); ref = V._fla_gdn_lm(cfg); V._transplant_fla(ours, ref)
ours, ref = ours.to(dev), ref.to(dev)
T = a.T
ids = torch.randint(8, cfg.vocab_size, (1, T), generator=RC.torch_gen(22, T, 1)).to(dev)   # same ids as VT-22
P = V.sample_positions(T)
out = {}
with torch.no_grad():
    out["fla_fp32"] = V._ref_hidden_logits(ref, ids, P, False, "fla_gdn").cpu()
    out["fla_bf16"] = V._ref_hidden_logits(ref, ids, P, True, "fla_gdn").cpu()
    out["ours_fp32"] = V._ours_logits(ours, ids, P, False).cpu()
    out["ours_bf16"] = V._ours_logits(ours, ids, P, True).cpu()
for x, y in [("ours_bf16", "fla_bf16"), ("ours_bf16", "fla_fp32"), ("fla_bf16", "fla_fp32"), ("ours_fp32", "fla_fp32"),
             ("ours_bf16", "ours_fp32")]:
    m = V._metrics(out[x], out[y], P, T)
    row = {"a": x, "ref": y, "T": T, **{k: m[k] for k in ("rel_rms", "rel_max", "band_ratio", "band8_rel_rms")}}
    wr.write(row); print(x, "vs", y, f"rel_rms {m['rel_rms']:.3e} band {m['band_ratio']:.2f}")
wr.close({})

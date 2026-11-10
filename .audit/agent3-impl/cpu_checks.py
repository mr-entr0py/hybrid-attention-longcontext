"""Agent-3 implementation audit: CPU-only checks (no GPU, no CUDA context).

Run:  CUDA_VISIBLE_DEVICES= python .audit/agent3-impl/cpu_checks.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

import torch  # noqa: E402

from hybridlc.models.build import build_model  # noqa: E402
from hybridlc.models.config import ModelConfig  # noqa: E402
from hybridlc.training.config import OptimConfig  # noqa: E402
from hybridlc.training.optim import param_groups  # noqa: E402

out: dict = {}

# ---------------------------------------------------------------- 1. parameter counts vs protocol §3.4
TABLE = {  # non-embedding params, protocol §3.4
    ("D384", "C0"): 21_309_384, ("D384", "C1"): 21_306_946, ("D384", "C3"): 21_302_070,
    ("D384", "C12"): 21_280_128, ("D384", "X-gc12"): 21_335_424, ("D384", "X-E0"): 21_293_976,
    ("D512", "C0"): 41_007_200, ("D512", "C3"): 40_997_576, ("D512", "C12"): 40_968_704,
}
pc = {}
for (scale, name), want in TABLE.items():
    cfg = ModelConfig.for_scale(scale, name)
    with torch.device("meta"):
        m = build_model(cfg, None)  # meta: counts only
    tot = sum(p.numel() for p in m.parameters())
    emb = m.embed.weight.numel()
    pc[f"{scale}/{name}"] = dict(non_emb=tot - emb, want=want, ok=(tot - emb == want))
out["param_counts"] = pc

# ---------------------------------------------------------------- 2. shared bit-identity across k, mixer identity
def shared(m):
    return {n: p.detach().clone() for n, p in m.named_parameters() if ".mixer." not in n}

cfgs = {n: ModelConfig.for_scale("D384", n) for n in ("C0", "C3", "C12", "X-gc12", "C1")}
models = {n: build_model(c, 0) for n, c in cfgs.items()}
ref = shared(models["C0"])
bit = {}
for n, m in models.items():
    s = shared(m)
    bit[n] = all(torch.equal(s[k], ref[k]) for k in ref) and set(s) == set(ref)
out["shared_bit_identical_vs_C0"] = bit
# same mixer type at same layer: C3 layer 2 (A) vs C12 layer 2 (A); C0 layer 0 (-) vs C3 layer 0 (-)
def mix_eq(a, b, l):
    pa = dict(models[a].layers[l].mixer.named_parameters())
    pb = dict(models[b].layers[l].mixer.named_parameters())
    return set(pa) == set(pb) and all(torch.equal(pa[k], pb[k]) for k in pa)
out["mixer_identity"] = {"C3_vs_C12_layer2_A": mix_eq("C3", "C12", 2), "C0_vs_C3_layer0_GDN": mix_eq("C0", "C3", 0),
                         "C0_vs_C3_layer11_GDN": mix_eq("C0", "C3", 11), "C1_vs_C3_layer6_A": mix_eq("C1", "C3", 6)}
# seeds differ
m1 = build_model(cfgs["C3"], 1)
out["seed1_differs_embed"] = not torch.equal(m1.embed.weight, models["C3"].embed.weight)

# ---------------------------------------------------------------- 3. init statistics
stats = {}
m = models["C3"]
for n, p in m.named_parameters():
    if p.ndim >= 2:
        stats[n] = round(float(p.std()), 6)
out["init_std_examples"] = {k: v for k, v in stats.items() if k.startswith(("layers.0.", "layers.2.", "embed"))}
out["out_std_target"] = 0.02 / math.sqrt(24)
gdn = m.layers[0].mixer
out["gdn_A_exp_range"] = [float(gdn.A_log.exp().min()), float(gdn.A_log.exp().max())]
out["gdn_dt_softplus_range"] = [float(torch.nn.functional.softplus(gdn.dt_bias).min()),
                                float(torch.nn.functional.softplus(gdn.dt_bias).max())]

# ---------------------------------------------------------------- 4. optimiser groups
for name in ("C3", "X-gc12"):
    groups, names = param_groups(models[name], OptimConfig())
    nd = names["no_decay"]
    out[f"no_decay_{name}"] = sorted({n.split(".", 2)[-1] if n.startswith("layers.") else n for n in nd})
    out[f"decay_suffixes_{name}"] = sorted({n.split(".", 2)[-1] if n.startswith("layers.") else n for n in names["decay"]})
c12s = build_model(ModelConfig.for_scale("D384", "C12", ssmax=True), 0)
groups, names = param_groups(c12s, OptimConfig())
out["ssmax_in_no_decay"] = any("ssmax" in n for n in names["no_decay"])

# ---------------------------------------------------------------- 5. global RNG untouched (CPU)
st = torch.random.get_rng_state()
_ = build_model(cfgs["C3"], 5)
out["cpu_global_rng_untouched"] = torch.equal(st, torch.random.get_rng_state())

print(json.dumps(out, indent=1))

"""Audit R2-R7: recompute every verdict-bearing number from the executors' raw files (CPU only).

Run from the repo root:  python -I experiments/repro/audit_R2-R7/scripts/audit_recompute.py
Writes nothing; prints one JSON object per check. Independent of the executors' analysis scripts.
"""
import glob
import gzip
import json
import math
import os
import statistics as st
import struct

ROOT = os.getcwd()
P = lambda *a: os.path.join(ROOT, *a)  # noqa: E731
out = {}


def jl(f):
    return [json.loads(x) for x in open(f) if x.strip()]


# ---------------- R2(c): longest kernel straight from the chrome traces ----------------
r2 = {}
for f in sorted(glob.glob(P("experiments/repro/r2/profile/traces/*.json.gz"))):
    tr = json.load(gzip.open(f))
    ev = tr["traceEvents"] if isinstance(tr, dict) else tr
    k = [e.get("dur", 0) for e in ev if e.get("cat") == "kernel"]
    r2[os.path.basename(f)[:-8]] = round(max(k) / 1e3, 3)
warm = {k: v for k, v in r2.items() if "warm" in k}
out["R2c"] = {"max_kernel_ms_by_phase": r2, "max_warm_ms": max(warm.values()),
              "argmax_warm": max(warm, key=warm.get), "criterion_lt_1000ms": max(warm.values()) < 1000.0}

# ---------------- R3 / VT-21 ----------------
rows = jl(P("experiments/repro/R3/vt21/vt21_conv.r1.jsonl")) + jl(P("experiments/repro/R3/vt21/vt21_conv.r3.jsonl"))
main = [r for r in rows if r.get("grid") == "main"]
ids = [r["case"] for r in main]


def vt21_pass(r):
    if r["dtype"] == "float32":
        return (r["y_max_abs"] <= 1e-5 and r["fin_max_abs"] <= 1e-5 and r["dx_max_abs"] <= 1e-4
                and r["dw_max_abs"] <= 1e-4)
    return all(r[k + "_rel_rms"] <= 1e-2 for k in ("y", "fin", "dx", "dw"))


fails = [r for r in main if not vt21_pass(r)]
pk = [r for r in main if r["packed"]]
out["VT21"] = {"n_rows": len(main), "n_unique_cases": len(set(ids)), "n_pass": len(main) - len(fails),
               "fail_cases": sorted(r["case"] for r in fails),
               "fail_all_fp32_dw_T32768": all(r["dtype"] == "float32" and r["T"] == 32768 and r["dw_max_abs"] > 1e-4
                                              and r["y_max_abs"] <= 1e-5 and r["dx_max_abs"] <= 1e-4 for r in fails),
               "worst_fp32_dw_abs": max(r["dw_max_abs"] for r in main if r["dtype"] == "float32"),
               "dw_ref_maxabs_range_T32768_fp32": [min(r["dw_ref_maxabs"] for r in main if r["dtype"] == "float32" and r["T"] == 32768),
                                                   max(r["dw_ref_maxabs"] for r in main if r["dtype"] == "float32" and r["T"] == 32768)],
               "packed_cases": len(pk), "leak_zero_all": all(r["leak_fwd_zero"] and r["leak_bwd_zero"] for r in pk),
               "verdict": "PASS" if not fails else "FAIL"}

# ---------------- R3 / VT-22 ----------------
v22 = jl(P("experiments/repro/R3/vt22/vt22_transplant.jsonl"))
chk = []
for r in v22:
    if r["family"] == "packed_vs_alone":
        ok = r["rel_rms"] <= 1e-2
    elif r["precision"] == "fp32":
        ok = r["rel_max"] <= 1e-3
    else:
        ok = r["rel_rms"] <= 2e-2 and r["band_ratio"] <= 2.0
    chk.append({"family": r["family"], "T": r["T"], "prec": r["precision"], "rel_rms": r["rel_rms"],
                "rel_max": r["rel_max"], "band_ratio": r.get("band_ratio"), "pass": ok})
out["VT22"] = {"rows": chk, "verdict": "PASS" if all(c["pass"] for c in chk) else "FAIL"}

# ---------------- R3 / VT-23 ----------------
A = P("experiments/repro/R3/vt23/attempt_20261110T062646")
tr = jl(glob.glob(A + "/runs/*/trace.jsonl")[0])
rf = jl(A + "/ref/ref_trace.jsonl")
hx = lambda h: struct.unpack(">d", bytes.fromhex(h))[0]  # noqa: E731
L = [abs(hx(t["loss_hex"]) - r["loss"]) / abs(r["loss"]) for t, r in zip(tr, rf)]
G = [abs(hx(t["gnorm_hex"]) - r["gnorm"]) / abs(r["gnorm"]) for t, r in zip(tr, rf)]
m0 = json.loads(open(glob.glob(A + "/runs/*/metrics.jsonl")[0]).readline())
grp = json.load(open(A + "/ref/ref_groups.json"))
nd_ref = set(grp.get("no_decay", grp.get("no_decay_names", [])))
nd_tr = set(m0["optimizer"]["no_decay_names"])
d1 = jl(A + "/diag_D1_chunkedce_torchclip/ref_trace.jsonl")
d1b = jl(A + "/diag_D1b_repeat/ref_trace.jsonl")
d3 = jl(A + "/diag_D3_cechunk2048/ref_trace.jsonl")
G3 = [abs(a["gnorm"] - b["gnorm"]) / b["gnorm"] for a, b in zip(d3, rf)]
v23 = {"step1_loss_rel": L[0], "s2_10_loss_max": max(L[1:10]), "s2_10_gnorm_max": max(G[1:10]),
       "s11_50_loss_max": max(L[10:]), "s11_50_loss_argmax_step": 11 + L[10:].index(max(L[10:])),
       "s11_50_gnorm_max": max(G[10:]), "s11_50_gnorm_argmax_step": 11 + G[10:].index(max(G[10:])),
       "clipped_steps_trainer": sum(hx(t["gnorm_hex"]) > 1.0 for t in tr), "clipped_steps_ref": sum(r["gnorm"] > 1.0 for r in rf),
       "wd_no_decay_identical": nd_ref == nd_tr, "n_no_decay": len(nd_tr),
       "n_decay_trainer": m0["optimizer"]["n_decay_tensors"], "n_decay_ref": len(grp.get("decay", [])),
       "D1_bitwise_steps": [i + 1 for i in range(len(d1)) if hx(tr[i]["gnorm_hex"]) == d1[i]["gnorm"]][:8],
       "D1_repeat_identical": all(a["gnorm"] == b["gnorm"] and a["loss"] == b["loss"] for a, b in zip(d1, d1b)),
       "D3_vs_ref_s2_10": max(G3[1:10]), "D3_vs_ref_s11_50": max(G3[10:])}
try:
    import torch
    ck = torch.load(glob.glob(A + "/ckpt/**/final.pt", recursive=True)[0], map_location="cpu", weights_only=False)
    sd = ck.get("model", ck.get("model_state", ck))
    rr = torch.load(A + "/ref/ref_final.pt", map_location="cpu", weights_only=False)
    rsd = rr.get("model", rr.get("state_dict", rr))
    keys = sorted(set(sd) & set(rsd))
    num = sum(((sd[k].double() - rsd[k].double()) ** 2).sum() for k in keys)
    den = sum((rsd[k].double() ** 2).sum() for k in keys)
    v23.update({"n_params_trainer": len(sd), "n_params_ref": len(rsd), "n_common": len(keys),
                "final_params_rel_l2": float((num / den).sqrt())})
except Exception as e:  # pragma: no cover
    v23["final_params_error"] = repr(e)
v23["verdict"] = "PASS" if (v23["step1_loss_rel"] <= 1e-6 and max(v23["s2_10_loss_max"], v23["s2_10_gnorm_max"]) <= 1e-5
                            and max(v23["s11_50_loss_max"], v23["s11_50_gnorm_max"]) <= 1e-3
                            and v23["clipped_steps_trainer"] >= 1 and v23.get("final_params_rel_l2", 1) <= 1e-3
                            and v23["wd_no_decay_identical"]) else "FAIL"
out["VT23"] = v23

# ---------------- R3 / VT-24 ----------------
v24 = jl(P("experiments/repro/R3/vt24/run_20261110T055803/vt24_files.jsonl"))
ret = [r for r in v24 if r["role"] == "retrieval"]
neg = [r for r in v24 if r["role"] != "retrieval"]
out["VT24"] = {"retrieval_files": len(ret), "queries": sum(r["n_queries"] for r in ret),
               "misses": sum(r["n_misses"] for r in ret),
               "files_with_misses": {r["path"]: r["n_misses"] for r in ret if r["n_misses"]},
               "neg_files": len(neg), "neg_max_em": max(r["em_mean"] for r in neg),
               "verdict": "PASS" if sum(r["n_misses"] for r in ret) == 0 and max(r["em_mean"] for r in neg) < 0.5 else "FAIL"}

# ---------------- R3 / VT-25 (+ unplanned replications) ----------------
def v25(f):
    rr_ = jl(f)
    gated = [r for r in rr_ if "pass" in r]
    ok = all((r["logits_rel_rms"] <= 1e-2 and r["grad_worst"] <= 2e-2) if r["family"] == "gated_delta_net"
             else (r["logits_rel_rms"] <= 1e-5 and r["grad_worst"] <= 1e-4) for r in gated)
    return {"n_gated": len(gated), "worst_gdn_grad": max(r["grad_worst"] for r in gated if r["family"] == "gated_delta_net"),
            "worst_gdn_logits": max(r["logits_rel_rms"] for r in gated if r["family"] == "gated_delta_net"),
            "worst_attn_logits": max(r["logits_rel_rms"] for r in gated if r["family"] == "attention"),
            "worst_attn_grad": max(r["grad_worst"] for r in gated if r["family"] == "attention"),
            "verdict": "PASS" if ok else "FAIL"}


out["VT25"] = {os.path.basename(f): v25(f) for f in sorted(glob.glob(P("experiments/repro/R3/vt25/vt25_transplant*.jsonl")))}

# ---------------- R4 ----------------
pairs = [("our_fg-d64-lr1.0e-02", "attention-d64-lr1.0e-02"), ("our_fg-d128-lr1.0e-03", "attention-d128-lr1.0e-03"),
         ("our_gdn-d128-lr1.0e-03", "gated_delta_net-d128-lr1.0e-03"), ("our_gdn-d256-lr1.0e-03", "gated_delta_net-d256-lr1.0e-03")]
cells = []
for o, z in pairs:
    a = json.load(open(P("experiments/repro/R4/runs", o + ".json")))
    b = json.load(open(P("experiments/s1_zoology/runs", z + ".json")))
    cells.append({"ours": o, "zoology": z, "ours_acc": a["final_overall_acc"], "zoo_acc": b["final_overall_acc"],
                  "delta": a["final_overall_acc"] - b["final_overall_acc"], "seed": a["seed"], "data_seed": a["data_seed"],
                  "autocast": a["autocast_bf16_mixer"], "pass": abs(a["final_overall_acc"] - b["final_overall_acc"]) <= 0.03})
ind = ["T64_kv4", "T64_kv8", "T64_kv16", "T128_kv32", "T256_kv64"]
r4b = {}
for r in ("our_fg_prod-d128-lr1.0e-03", "our_fg_prod_bf16-d128-lr1.0e-03"):
    d = json.load(open(P("experiments/repro/R4/runs", r + ".json")))
    r4b[r] = {"indist_mean": sum(d["final_per_config_acc"][k] for k in ind) / 5, "varlen_calls": d["varlen_attn_training"]["calls"],
              "autocast": d["autocast_bf16_mixer"]}
out["R4"] = {"cells": cells, "GRPb": "PASS" if all(c["pass"] for c in cells) else "FAIL", "R4b": r4b,
             "R4b_verdict": "PASS" if all(v["indist_mean"] >= 0.95 for v in r4b.values()) else "FAIL"}

# ---------------- R5 ----------------
runs = {a: [json.load(open(P("experiments/repro/R5/runs", f"{a}-d64-lr1.0e-03-s{s}.json"))) for s in (123, 124, 125)]
        for a in ("gdn_zoology", "our_gdn_bf16")}
z = [d["final_per_config_acc"]["T1024_kv256"] for d in runs["gdn_zoology"]]
o = [d["final_per_config_acc"]["T1024_kv256"] for d in runs["our_gdn_bf16"]]
sz, so = st.stdev(z), st.stdev(o)
se = math.sqrt(so ** 2 / 3 + sz ** 2 / 3)
r1 = {w: json.load(open(P("experiments/s1_zoology/runs", f"gated_delta_net-d{w}-lr1.0e-03.json")))["final_per_config_acc"]["T1024_kv256"]
      for w in (128, 256)}
out["R5"] = {"zoology": z, "ours": o, "mean_z": st.mean(z), "mean_o": st.mean(o), "sd_z": sz, "sd_o": so,
             "rule1_in_regime": 0.05 <= st.mean(z) <= 0.95, "rule2": [st.mean(z), r1[128], r1[256]],
             "rule2_pass": st.mean(z) < r1[128] < r1[256], "delta": st.mean(o) - st.mean(z), "se_w": se,
             "threshold": max(0.03, 2.78 * se), "rule3_pass": abs(st.mean(o) - st.mean(z)) <= max(0.03, 2.78 * se),
             "span_z": max(z) - min(z), "span_o": max(o) - min(o),
             "rule4_fires": (max(z) - min(z) > 0.2) or (max(o) - min(o) > 0.2),
             "seeds_ok": all(d["seed"] == d["data_seed"] and d["lr"] == 1e-3 and d["d_model"] == 64 and not d["resumed_at_epochs"]
                             and not d["smoke"] for a in runs for d in runs[a]),
             "autocast_ours_only": [d["autocast_bf16_mixer"] for a in runs for d in runs[a]]}

# ---------------- R7 ----------------
r7 = {}
for m in ("pythia-160m", "pythia-70m"):
    ours = [r for r in jl(P(f"experiments/repro/R7/scores/{m}_openai_bos-default_fp32.jsonl")) if "EM_tf" in r]
    lm = jl(P(f"experiments/repro/R7/lmeval/{m}_lambada_openai.jsonl"))
    oi, li = {r["context_id"]: r for r in ours}, {r["doc_id"]: r for r in lm}
    common = set(oi) & set(li)
    f16 = [r for r in jl(P(f"experiments/repro/R7/scores/{m}_openai_bos-default_fp16w.jsonl")) if "EM_tf" in r]
    r7[m] = {"n": len(ours), "correct": sum(bool(r["EM_tf"]) for r in ours), "acc": sum(bool(r["EM_tf"]) for r in ours) / len(ours),
             "lmeval_acc": sum(bool(r["is_greedy"]) for r in lm) / len(lm), "flips": sum(bool(oi[k]["EM_tf"]) != bool(li[k]["is_greedy"]) for k in common),
             "max_abs_dlogp": max(abs(oi[k]["logp_gold"] - li[k]["ll"]) for k in common),
             "fp16w_acc": sum(bool(r["EM_tf"]) for r in f16) / len(f16)}
g = [r for r in jl(P("experiments/repro/R7/scores/gdn340m_openai_bos-default_fp32.jsonl")) if "EM_tf" in r]
r7["gdn340m_openai"] = {"n": len(g), "acc": sum(bool(r["EM_tf"]) for r in g) / len(g)}
r7["verdicts"] = {"pythia-160m": 0.313 <= r7["pythia-160m"]["acc"] <= 0.343, "pythia-70m": 0.170 <= r7["pythia-70m"]["acc"] <= 0.200,
                  "gdn340m": abs(r7["gdn340m_openai"]["acc"] - 0.348) <= 0.02}
out["R7"] = r7

for k, v in out.items():
    print(json.dumps({k: v}, default=float))

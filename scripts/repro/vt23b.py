"""VT-23b: one-step-from-state check of the production trainer update (prereg experiments/prereg/VT-23b.yaml).

    python scripts/repro/gpu_lock.py run --label VT-23b --wait-exclusive -- python scripts/repro/vt23b.py
    python scripts/repro/vt23b.py --selftest        # CPU only: checks the float64 AdamW against torch.optim.AdamW

1. trainer_run: scripts/train.py, D384 C3, fp32, audit mode, activation checkpointing, StageStream('A', data_seed=901), 2x4096 micro-batches,
   accum 2, 50 steps, 10 warm-up then constant lr 2e-3 (R3 VT-23 training yaml), full checkpoint after EVERY step (keep all).
2. For each s in S = {0,1,4,8,9,10,20,35,48} (0-based; s = 0 is the initial state) the independent reference (loss, global clip, AdamW written
   from protocol 7.1; imports neither hybridlc.training.optim/schedules nor the trainer's update code) is evaluated ONE step from the
   trainer's own state (theta_s, m_s, v_s) on the same batch and compared: C1 loss, C2 pre-clip grad norm (rel 1e-5); C3 clipped gradient vs
   g_tr = (m_{s+1} - 0.9 m_s)/0.1 per tensor (rel L2 1e-4); C4 reference AdamW on (theta_s, m_s, v_s, g_tr) vs the trainer's update and v_{s+1}
   (rel L2 1e-4); C5 weight-decay group membership by name. Regime check: at least one s with pre-clip norm > 1 and one with <= 1; else add
   s from (15, 25, 30, 40, 45). Output: experiments/repro/VT-23b/{trainer_run/, onestep.jsonl, _meta.json}.
"""
# --- R2-LIVE v2 L1 entry-point guard: take the project GPU lock (or verify the holder's token) BEFORE any CUDA context.
import sys as _sys_guard  # noqa: E402
from pathlib import Path as _Path_guard  # noqa: E402
if __name__ == "__main__" and "--selftest" not in _sys_guard.argv:
    _sys_guard.path.insert(0, str(_Path_guard(__file__).resolve().parents[2] / "src"))
    from hybridlc.utils.gpu_lock import guard as _gpu_guard  # noqa: E402
    _gpu_guard(_Path_guard(__file__).name)
# ---
import argparse
import json
import math
import os
import struct
import subprocess
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "repro"))
OUT = REPO / "experiments" / "repro" / "VT-23b"
S_BASE = [0, 1, 4, 8, 9, 10, 20, 35, 48]
S_EXTRA = [15, 25, 30, 40, 45]
SEED, LR, WARM, ACCUM, MICRO, SEQ = 901, 2.0e-3, 10, 2, 2, 4096
B1, B2, EPS, WD, CLIP, Z_COEF, PAD = 0.9, 0.95, 1e-8, 0.1, 1.0, 1e-4, 1


def log(m):
    print(f"[vt23b {time.strftime('%H:%M:%S')}] {m}", flush=True)


def hexf(h: str) -> float:
    return struct.unpack(">d", bytes.fromhex(h))[0]  # trace stores the big-endian IEEE-754 double as hex (as r3_vt23_diagnose.py)


def lr_at(s: int) -> float:
    return LR * min(1.0, (s + 1) / WARM)


def adamw64(theta, m, v, g, s, lr, wd):
    """Reference AdamW step in float64 (torch's documented bias-correction convention), step count t = s + 1."""
    t = s + 1
    theta = theta.double() * (1.0 - lr * wd)
    m_new = B1 * m.double() + (1.0 - B1) * g.double()
    v_new = B2 * v.double() + (1.0 - B2) * g.double() ** 2
    bc1, bc2 = 1.0 - B1 ** t, 1.0 - B2 ** t
    denom = v_new.sqrt() / math.sqrt(bc2) + EPS
    return theta - (lr / bc1) * m_new / denom, m_new, v_new


def rel(a, b) -> float:
    n = float(b.double().norm())
    return float((a.double() - b.double()).norm()) / n if n > 0 else (0.0 if float(a.double().norm()) == 0 else float("inf"))


def selftest() -> int:
    torch.manual_seed(0)
    p = torch.randn(64, 32, dtype=torch.float32)
    q = torch.nn.Parameter(p.clone())
    opt = torch.optim.AdamW([q], lr=2e-3, betas=(B1, B2), eps=EPS, weight_decay=WD)
    th, m, v = p.double(), torch.zeros(64, 32).double(), torch.zeros(64, 32).double()
    worst = 0.0
    for s in range(12):
        g = torch.randn(64, 32)
        q.grad = g.clone()
        for gr in opt.param_groups:
            gr["lr"] = lr_at(s)
        opt.step()
        th, m, v = adamw64(th, m, v, g, s, lr_at(s), WD)
        worst = max(worst, rel(th, q.detach()))
    log(f"selftest: float64 AdamW vs torch.optim.AdamW over 12 steps, worst rel diff {worst:.2e} (needs <= 1e-6)")
    return 0 if worst <= 1e-6 else 1


def run_trainer(steps: int) -> Path:
    root = OUT / "trainer_run"
    root.mkdir(parents=True, exist_ok=True)
    run = OUT / "run_vt23b.yaml"
    run.write_text(f"exp_stage: R3\nmodel: configs/model/D384/C3.yaml\ntraining: experiments/repro/R3/vt23/training_vt23.yaml\n"
                   f"seed_index: {SEED}\nlr: {LR}\n")
    env = dict(os.environ, PYTHONHASHSEED="0", CUBLAS_WORKSPACE_CONFIG=":4096:8", TRITON_CACHE_DIR=str(REPO / ".cache" / "triton-diag"))
    cmd = [sys.executable, "scripts/train.py", "--config", str(run), "-o", f"schedule.steps={steps}", "-o", "checkpoint.every=1",
           "-o", "checkpoint.keep_last=200", "--runs-root", str(root / "runs"), "--ckpt-root", str(root / "ckpt"),
           "--registry", str(root / "registry.jsonl")]
    log("trainer: " + " ".join(cmd[2:6]) + " ...")
    with open(root / "trainer.log", "ab") as f:
        rc = subprocess.run(cmd, cwd=REPO, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    assert rc == 0, f"trainer exited {rc}; see {root/'trainer.log'}"
    return root


def load_trace(root: Path) -> dict:
    tr = next((root / "runs").glob("*/trace.jsonl"))
    out = {}
    for line in tr.read_text().splitlines():
        j = json.loads(line)
        out[j["step"]] = {"loss": hexf(j["loss_hex"]), "gnorm": hexf(j["gnorm_hex"]), "batch_hash": j["batch_hash"]}
    return out


def ckpt_state(root: Path, s: int, names: list[str], model_init: dict):
    """(theta, m, v) dicts by parameter name after s updates; s = 0 is the initial state with zero moments."""
    if s == 0:
        z = {n: torch.zeros_like(model_init[n]) for n in names}
        return {n: model_init[n].clone() for n in names}, z, {n: z[n].clone() for n in names}
    from hybridlc.training import checkpoint as C
    ck = C.load(next((root / "ckpt").glob("*")) / f"step_{s:07d}.pt")
    st = ck["optimizer"]["state"]
    theta = {n: ck["model"][n].float() for n in names}
    m = {n: st[i]["exp_avg"].float() for i, n in enumerate(names)}
    v = {n: st[i]["exp_avg_sq"].float() for i, n in enumerate(names)}
    return theta, m, v


def ce_z_sums(h, w, y):
    logits = h.float() @ w.float().t()
    lse = torch.logsumexp(logits, dim=-1)
    valid = (y != PAD) & (y >= 0)
    tgt = logits.gather(1, y.clamp_min(0)[:, None])[:, 0]
    return torch.where(valid, lse - tgt, torch.zeros_like(lse)).sum(), torch.where(valid, lse * lse, torch.zeros_like(lse)).sum()


def reference_step(model, stream, s, dev):
    """Reference loss / pre-clip norm / clipped gradient for the step taking state s -> s+1, from the model's current weights."""
    from torch.utils.checkpoint import checkpoint
    model.zero_grad(set_to_none=True)
    batches = [stream.get(s, k) for k in range(ACCUM)]
    N = sum(int(((b["labels"] != PAD) & (b["labels"] >= 0)).sum()) for b in batches)
    ce_tot = torch.zeros((), dtype=torch.float64, device=dev)
    W = model.embed.weight
    for b in batches:
        ids, y = b["input_ids"].to(dev), b["labels"].to(dev)[0]
        h = model(ids, b["cu_seqlens"].to(dev), b["position_ids"].to(dev))[0]
        ce_s = torch.zeros((), device=dev)
        z_s = torch.zeros((), device=dev)
        for c0 in range(0, h.shape[0], 4096):
            ce_c, z_c = checkpoint(ce_z_sums, h[c0:c0 + 4096], W, y[c0:c0 + 4096], use_reentrant=False)
            ce_s, z_s = ce_s + ce_c, z_s + z_c
        ((ce_s + Z_COEF * z_s) / N).backward()
        ce_tot += ce_s.detach().double()
    params = list(model.parameters())
    gn = torch.sqrt(sum((p.grad.float() ** 2).sum() for p in params if p.grad is not None))
    coef = min(1.0, CLIP / (float(gn) + 1e-6))
    grads = {n: (p.grad.double() * coef) for n, p in model.named_parameters() if p.grad is not None}
    return float(ce_tot / N), float(gn), grads, batches


def check_s(s, root, model, stream, names, decay_names, trace, init, dev):
    theta_s, m_s, v_s = ckpt_state(root, s, names, init)
    theta_n, m_n, v_n = ckpt_state(root, s + 1, names, init)
    with torch.no_grad():
        for n, p in model.named_parameters():
            p.copy_(theta_s[n].to(dev))
    loss_r, gn_r, g_ref, batches = reference_step(model, stream, s, dev)
    from hybridlc.training.trainer import _step_hash
    tr = trace[s]
    row = {"s": s, "lr": lr_at(s), "batch_hash_equal": _step_hash(batches) == tr["batch_hash"],
           "loss_ref": loss_r, "loss_tr": tr["loss"], "C1_rel": abs(loss_r - tr["loss"]) / abs(tr["loss"]),
           "gn_ref": gn_r, "gn_tr": tr["gnorm"], "C2_rel": abs(gn_r - tr["gnorm"]) / abs(tr["gnorm"]), "clipped": tr["gnorm"] > CLIP}
    c3, c4, v4, worst3, worst4 = [], [], [], 0.0, 0.0
    for n in names:
        g_tr = (m_n[n].double() - B1 * m_s[n].double()) / (1.0 - B1)
        gr = g_ref.get(n)
        if float(g_tr.norm()) == 0.0:
            ok3 = gr is None or float(gr.norm()) == 0.0
            c3.append(ok3)
        else:
            r3 = rel(gr.cpu() if gr is not None else torch.zeros_like(g_tr), g_tr)
            worst3 = max(worst3, r3)
            c3.append(r3 <= 1e-4)
        wd = WD if n in decay_names else 0.0
        th_new, _, v_new = adamw64(theta_s[n], m_s[n], v_s[n], g_tr, s, lr_at(s), wd)
        d_ref, d_tr = th_new - theta_s[n].double(), theta_n[n].double() - theta_s[n].double()
        r4, rv = rel(d_ref, d_tr), rel(v_new, v_n[n])
        worst4 = max(worst4, r4, rv)
        c4.append(r4 <= 1e-4 and rv <= 1e-4)
    row.update({"C3_worst_rel": worst3, "C4_worst_rel": worst4, "C1": row["C1_rel"] <= 1e-5, "C2": row["C2_rel"] <= 1e-5,
                "C3": all(c3), "C4": all(c4)})
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--skip-trainer", action="store_true")
    ap.add_argument("--steps", type=int, default=50)
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    OUT.mkdir(parents=True, exist_ok=True)
    root = OUT / "trainer_run" if a.skip_trainer else run_trainer(a.steps)
    from hybridlc.data.loader import StageStream
    from hybridlc.models.build import build_model
    from hybridlc.models.config import ModelConfig
    from hybridlc.training.config import load_run_config
    from hybridlc.training.optim import param_groups
    from hybridlc.utils.gpu import claim, set_production_env
    set_production_env(overwrite=False)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.enable_cudnn_sdp(False)
    dev = claim(0.44)
    model = build_model(ModelConfig.for_scale("D384", "C3"), SEED).to(dev).train()
    model.set_activation_checkpointing(True)
    init = {n: p.detach().cpu().clone() for n, p in model.named_parameters()}
    cfg = load_run_config(OUT / "run_vt23b.yaml")
    _, grp = param_groups(model, cfg.optim)
    names = grp["decay"] + grp["no_decay"]  # optimizer state index order: decay group first
    ref_decay = {n for n, p in model.named_parameters() if p.ndim >= 2 and not getattr(p, "_no_weight_decay", False) and "ssmax" not in n}
    c5 = ref_decay == set(grp["decay"])
    trace = load_trace(root)
    stream = StageStream("A", SEED, SEQ, MICRO, ACCUM, None)
    rows, S = [], list(S_BASE)
    with open(OUT / "onestep.jsonl", "w") as f:
        def do(s):
            r = check_s(s, root, model, stream, names, set(grp["decay"]), trace, init, dev)
            rows.append(r)
            f.write(json.dumps(r) + "\n")
            f.flush()
            log(f"s={s}: C1 {r['C1_rel']:.1e} C2 {r['C2_rel']:.1e} C3 {r['C3_worst_rel']:.1e} C4 {r['C4_worst_rel']:.1e} clipped={r['clipped']} hash_eq={r['batch_hash_equal']}")
        for s in S:
            do(s)
        extras = list(S_EXTRA)
        while not (any(r["clipped"] for r in rows) and any(not r["clipped"] for r in rows)) and extras:
            do(extras.pop(0))
    regime = {"clipped_seen": any(r["clipped"] for r in rows), "unclipped_seen": any(not r["clipped"] for r in rows)}
    ok = c5 and all(r["batch_hash_equal"] and r["C1"] and r["C2"] and r["C3"] and r["C4"] for r in rows)
    meta = {"item": "VT-23b", "S_tested": [r["s"] for r in rows], "C5_group_membership": c5, "regime": regime,
            "verdict": "PASS" if ok else "FAIL", "prereg": "experiments/prereg/VT-23b.yaml"}
    (OUT / "_meta.json").write_text(json.dumps(meta, indent=1))
    log(f"verdict {meta['verdict']} C5={c5} regime={regime}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

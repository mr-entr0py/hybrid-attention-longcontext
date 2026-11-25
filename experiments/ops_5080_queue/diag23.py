import sys, json, torch
sys.path.insert(0, "scripts/repro"); sys.path.insert(0, "src")
import vt23b as V
from hybridlc.models.build import build_model
from hybridlc.models.config import ModelConfig
from hybridlc.training.config import load_run_config
from hybridlc.training.optim import param_groups
model = build_model(ModelConfig.for_scale("D384", "C3"), V.SEED)
init = {n: p.detach().cpu().clone() for n, p in model.named_parameters()}
cfg = load_run_config(V.OUT / "run_vt23b.yaml")
_, grp = param_groups(model, cfg.optim)
names = grp["decay"] + grp["no_decay"]; dec = set(grp["decay"])
root = V.OUT / "trainer_run"
for s in (0, 8, 35, 48):
    th_s, m_s, v_s = V.ckpt_state(root, s, names, init)
    th_n, m_n, v_n = V.ckpt_state(root, s + 1, names, init)
    worst = []
    for n in names:
        g = (m_n[n].double() - V.B1 * m_s[n].double()) / (1 - V.B1)
        wd = V.WD if n in dec else 0.0
        out = {}
        for tag, lr in (("lr_s", V.lr_at(s)), ("lr_s+1", V.lr_at(s + 1)), ("lr_s-1", V.lr_at(max(s - 1, 0)))):
            th_new, _, v_new = V.adamw64(th_s[n], m_s[n], v_s[n], g, s, lr, wd)
            D_tr = th_n[n].double() - th_s[n].double()
            out[tag] = V.rel(th_new - th_s[n].double(), D_tr)
            if tag == "lr_s":
                out["rounded32"] = V.rel(th_new.float().double() - th_s[n].double(), D_tr)
                ulp = torch.finfo(torch.float32).eps * th_n[n].double().abs()
                out["ulp_floor"] = float(ulp.norm() / D_tr.norm()) if float(D_tr.norm()) > 0 else 0
                out["v"] = V.rel(v_new, v_n[n])
        worst.append((out["lr_s"], n, out))
    worst.sort(key=lambda x: -x[0])
    r, n, o = worst[0]
    print(f"s={s} worst tensor {n}: " + " ".join(f"{k}={v:.2e}" for k, v in o.items()))
    print("   max over tensors: lr_s %.2e rounded32 %.2e ulp_floor %.2e v %.2e" % tuple(max(w[2][k] for w in worst) for k in ("lr_s", "rounded32", "ulp_floor", "v")))

"""VT-25 (R3a, DEV-01): Zoology ↔ our mixers by weight copy, inside Zoology's ``LanguageModel``.

Pre-registration: ``experiments/repro/R3/prereg.yaml`` (copied from reproduction_plan.md §2.3 R3a).

Reference: Zoology @1ad20d1, the 2-layer MQAR ``LanguageModel`` (BaseConv + mixer) built from
``original_mqar_configs.py``. Ours: ``scripts/s1_zoology/our_mixers.py`` (``OurGdnZoology``,
``OurFgZoology`` → ``hybridlc.models.mixers.build_mixer``) swapped into the same config. Every
Zoology parameter is copied into ours through the explicit name map below; Zoology's MHA biases
are zeroed and both models run in ``eval()`` (dropout off).

Pass (frozen):
  GDN, matched precision (ours reproducing Zoology's bf16 casts, test-only flag): logits rel RMS
  ≤ 1e-2, parameter gradients rel RMS ≤ 2e-2 (per tensor). GDN ours-fp32 vs Zoology: report only.
  Attention, fp32: logits rel RMS ≤ 1e-5, gradients ≤ 1e-4.

Raw output: ``experiments/repro/R3/vt25/vt25_transplant.jsonl``.
Run: ``pytest -q -s tests/test_vt25_zoology_transplant.py``.
"""
from __future__ import annotations

import contextlib
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "repro"))
sys.path.insert(0, str(REPO / "scripts" / "s1_zoology"))
import r3_common as RC  # noqa: E402

GDN_WIDTHS = (64, 128, 256)
ATTN_WIDTHS = (64, 128)
TOL = {"gdn_logits": 1e-2, "gdn_grad": 2e-2, "attn_logits": 1e-5, "attn_grad": 1e-4}
B, T, V = 8, 256, 8192
MIXER = "backbone.layers.1.sequence_mixer.mixer."


def _setup():
    from zoology_compat import setup_zoology

    setup_zoology(micro_batch=None)
    from s1_configs import _load_original_module

    return _load_original_module()


def _cfg(mod, family: str, d: int):
    return next(c for c in mod.configs if c.model.name == family and c.model.d_model == d).model.model_copy(deep=True)


def _swap(model_cfg, path: str):
    c = model_cfg.model_copy(deep=True)
    c.sequence_mixer.kwargs["configs"][1] = {"name": path, "kwargs": {}}
    return c


def _lm(model_cfg, seed: int):
    from zoology.model import LanguageModel

    torch.manual_seed(seed)
    return LanguageModel(model_cfg)


# ------------------------------------------------------------------------------------- name map
def name_map(family: str, ours_names: set[str], zoo_names: list[str]) -> dict[str, object]:
    """ours parameter name -> Zoology parameter name (or ('qkv', i) slices of Wqkv)."""
    m: dict[str, object] = {}
    for n in zoo_names:
        if not n.startswith(MIXER):
            m[n] = n                                    # embedding, BaseConv layer, norms (same Zoology code)
            continue
        leaf = n[len(MIXER):]
        if family == "gated_delta_net":
            if leaf == "D":
                continue                                # Zoology's unused parameter (no effect on output)
            m[MIXER + "inner." + leaf] = n
        else:
            if leaf == "Wqkv.weight":                   # "(three h d)" packing: rows [q | k | v]
                for i, proj in enumerate(("q_proj", "k_proj", "v_proj")):
                    m[MIXER + f"inner.{proj}.weight"] = ("qkv", i, n)
            elif leaf == "out_proj.weight":
                m[MIXER + "out_proj.weight"] = n        # alias of inner.o_proj.weight
            elif leaf in ("Wqkv.bias", "out_proj.bias"):
                continue                                # zeroed in Zoology; ours has no biases
            else:
                raise KeyError(n)
    missing = ours_names - set(m)
    assert not missing, f"unmapped parameters of ours: {sorted(missing)}"
    assert set(m) <= ours_names, sorted(set(m) - ours_names)
    return m


def _src(zoo_params: dict[str, torch.Tensor], spec, d: int) -> torch.Tensor:
    if isinstance(spec, tuple):
        _, i, n = spec
        return zoo_params[n][i * d:(i + 1) * d]
    return zoo_params[spec]


def transplant(zoo, ours, family: str, d: int) -> dict[str, object]:
    with torch.no_grad():
        if family == "attention":
            zm = zoo.get_submodule(MIXER[:-1])
            zm.Wqkv.bias.zero_()
            zm.out_proj.bias.zero_()
        zp = dict(zoo.named_parameters())
        op = dict(ours.named_parameters())
        m = name_map(family, set(op), list(zp))
        for on, spec in m.items():
            src = _src(zp, spec, d)
            assert op[on].shape == src.shape, (on, op[on].shape, src.shape)
            op[on].copy_(src)
    return m


# ---------------------------------------------------------------------- Zoology bf16 casts (flag)
@contextlib.contextmanager
def zoology_bf16_casts(ours):
    """Test-only flag: reproduce Zoology GDN's casts in our GDNMixer.

    Zoology casts q, k, v, sigmoid(b), g to bf16 before ``chunk_gated_delta_rule``, applies o_norm
    to the bf16 output and casts to fp32 before o_proj. Our layer keeps σ(·) and the gate formula in
    the kernel, so we cast its kernel inputs (q, k, v, b-logit, a) to bf16 and cast o_norm's output
    to fp32. Implemented through the module-level op indirection of ``hybridlc.models.linear``.
    """
    import hybridlc.models.linear as L

    orig = L.chunk_gated_delta_rule

    def wrapped(*, q, k, v, g, beta, **kw):
        bf = torch.bfloat16
        return orig(q=q.to(bf), k=k.to(bf), v=v.to(bf), g=g.to(bf), beta=beta.to(bf), **kw)

    hooks = [m.o_norm.register_forward_hook(lambda mod, inp, out: out.float())
             for m in ours.modules() if isinstance(m, L.GDNMixer)]
    assert hooks, "no GDNMixer found"
    L.chunk_gated_delta_rule = wrapped
    try:
        yield
    finally:
        L.chunk_gated_delta_rule = orig
        for h in hooks:
            h.remove()


def _batches(dev):
    from zoology.data.multiquery_ar import MQARConfig

    seg = MQARConfig(vocab_size=V, input_seq_len=T, num_examples=B, num_kv_pairs=64, seed=31337).build(seed=31337)
    g = RC.torch_gen(25, 1)
    rx = torch.randint(0, V, (B, T), generator=g)
    ry = torch.randint(0, V, (B, T), generator=g)
    return {"mqar": (seg.inputs.to(dev), seg.labels.to(dev)), "random": (rx.to(dev), ry.to(dev))}


def _fwd_bwd(model, x, y):
    model.zero_grad(set_to_none=True)
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1).float(), y.flatten(), ignore_index=-100)
    loss.backward()
    return logits.detach().float(), {n: (p.grad.detach().clone() if p.grad is not None else None)
                                     for n, p in model.named_parameters()}


def _compare(family, d, m, zl, zg, ol, og, tol_l, tol_g):
    row = {"logits_rel_rms": RC.rel_rms(ol, zl), "logits_rel_max": RC.rel_max(ol, zl)}
    per, worst, worst_name = {}, 0.0, None
    a_all, r_all = [], []
    for on, spec in m.items():
        zname = spec[2] if isinstance(spec, tuple) else spec
        zgrad = zg[zname]
        if isinstance(spec, tuple):
            zgrad = zgrad[spec[1] * d:(spec[1] + 1) * d]
        o = og[on]
        if zgrad is None or o is None:
            per[on] = None if (zgrad is None and o is None) else float("inf")
            continue
        e = RC.rel_rms(o, zgrad)
        per[on] = e
        a_all.append(o.flatten().double())
        r_all.append(zgrad.flatten().double())
        if e > worst:
            worst, worst_name = e, on
    row["grad_rel_rms_per_tensor"] = per
    row["grad_worst"], row["grad_worst_name"] = worst, worst_name
    row["grad_global_rel_rms"] = RC.rel_rms(torch.cat(a_all), torch.cat(r_all))
    row["any_missing_grad"] = any(v == float("inf") for v in per.values())
    if tol_l is not None:
        row["pass"] = bool(row["logits_rel_rms"] <= tol_l and worst <= tol_g and not row["any_missing_grad"])
    return row


@pytest.mark.gpu
def test_vt25_zoology_vs_ours_weight_copy() -> None:
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA (fla kernels)")
    mod = _setup()
    from hybridlc.utils.gpu import claim

    dev = claim(0.44)
    meta = RC.run_meta("VT-25", {"criteria": TOL})
    wr = RC.RawWriter("vt25", "vt25_transplant", meta)
    batches = _batches(dev)
    fails = []
    for family, widths, path in (("gated_delta_net", GDN_WIDTHS, "our_mixers.OurGdnZoology"),
                                 ("attention", ATTN_WIDTHS, "our_mixers.OurFgZoology")):
        for d in widths:
            zcfg = _cfg(mod, family, d)
            zoo = _lm(zcfg, 31337).to(dev).eval()
            ours = _lm(_swap(zcfg, path), 7).to(dev).eval()
            m = transplant(zoo, ours, family, d)
            for bname, (x, y) in batches.items():
                zl, zg = _fwd_bwd(zoo, x, y)
                if family == "gated_delta_net":
                    with zoology_bf16_casts(ours):
                        ol, og = _fwd_bwd(ours, x, y)
                    row = _compare(family, d, m, zl, zg, ol, og, TOL["gdn_logits"], TOL["gdn_grad"])
                    row.update(family=family, d=d, batch=bname, mode="matched_bf16_casts")
                    wr.write(row)
                    if not row["pass"]:
                        fails.append((family, d, bname, row["logits_rel_rms"], row["grad_worst"], row["grad_worst_name"]))
                    ol32, og32 = _fwd_bwd(ours, x, y)          # ours as built (fp32): report only
                    r32 = _compare(family, d, m, zl, zg, ol32, og32, None, None)
                    r32.update(family=family, d=d, batch=bname, mode="ours_fp32_report_only")
                    wr.write(r32)
                else:
                    ol, og = _fwd_bwd(ours, x, y)
                    row = _compare(family, d, m, zl, zg, ol, og, TOL["attn_logits"], TOL["attn_grad"])
                    row.update(family=family, d=d, batch=bname, mode="fp32")
                    wr.write(row)
                    if not row["pass"]:
                        fails.append((family, d, bname, row["logits_rel_rms"], row["grad_worst"], row["grad_worst_name"]))
                print(f"VT-25 {family} d{d} {bname}: logits {row['logits_rel_rms']:.2e} "
                      f"grad worst {row['grad_worst']:.2e} ({row['grad_worst_name']}) pass={row['pass']}")
            del zoo, ours
            torch.cuda.empty_cache()
    summary = {"n_gated": sum(1 for r in wr.rows if "pass" in r), "fails": fails,
               "verdict": "PASS" if not fails else "FAIL"}
    wr.close(summary)
    assert not fails, fails

"""VT-08: scoring path (protocol §9.2, §10.2, §16).

* fla ops are called explicitly and **by keyword only** (a positional call produced NaN, PROP §4);
* no automatic mode switch: prefill always uses the chunk kernel (even for ≤ 64 tokens without
  grad), continuation/decode always ``fused_recurrent_gated_delta_rule``;
* the SDPA backend is pinned and asserted: FLASH only for bf16 inference (mem-efficient and math
  disabled at call time), MATH only for fp32; a pinned backend that cannot run raises instead of
  silently falling back.
"""

from __future__ import annotations

import ast
import inspect

import pytest
import torch

from hybridlc.models import attention as A
from hybridlc.models import linear as L
from hybridlc.models.build import build_model
from hybridlc.models.config import ModelConfig
from hybridlc.models.testing import cuda_device_or_skip

FLA_OPS = ("chunk_gated_delta_rule", "fused_recurrent_gated_delta_rule", "chunk_gla", "fused_recurrent_gla",
           "causal_conv1d")


class _Spy:
    def __init__(self, name, fn, log):
        self.name, self.fn, self.log = name, fn, log

    def __call__(self, *args, **kwargs):
        if args:
            raise AssertionError(f"positional call to fla op {self.name}: {len(args)} positional args")
        self.log.append(self.name)
        return self.fn(**kwargs)


@pytest.fixture
def spies(monkeypatch):
    log: list[str] = []
    for name in FLA_OPS:
        monkeypatch.setattr(L, name, _Spy(name, getattr(L, name), log))
    sdpa_log: list[dict] = []
    real = A._sdpa

    def sdpa_spy(*args, **kwargs):
        sdpa_log.append(dict(flash=torch.backends.cuda.flash_sdp_enabled(),
                             mem_efficient=torch.backends.cuda.mem_efficient_sdp_enabled(),
                             math=torch.backends.cuda.math_sdp_enabled(),
                             cudnn=torch.backends.cuda.cudnn_sdp_enabled(),
                             dtype=args[0].dtype, Tq=args[0].shape[2], Tk=args[1].shape[2]))
        return real(*args, **kwargs)

    monkeypatch.setattr(A, "_sdpa", sdpa_spy)
    return log, sdpa_log


def test_source_calls_fla_ops_by_keyword_only() -> None:
    """Static check of models/linear.py and models/attention.py: no positional args to fla ops."""
    for mod in (L, A):
        tree = ast.parse(inspect.getsource(mod))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                fn = node.func
                name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
                if name in FLA_OPS or name == "_fla_causal_conv1d":
                    assert not node.args, f"{mod.__name__}: positional call to {name} at line {node.lineno}"


def test_fla_layer_auto_switch_exists_but_is_bypassed() -> None:
    """fla's layer forward switches to fused_recurrent for q_len ≤ 64 without grad; our mixer
    overrides forward and never calls the fla layer forward."""
    from fla.layers import GatedDeltaNet

    src = inspect.getsource(GatedDeltaNet.forward)
    assert "q_len <= 64" in src  # the hazard is real in fla 0.6.0
    assert L.GDNMixer.forward is not GatedDeltaNet.forward
    assert "super().forward" not in inspect.getsource(L.GDNMixer)


@pytest.mark.gpu
@pytest.mark.parametrize("name", ["C3", "O-gla3"])
def test_kernel_selection_and_keywords(spies, name) -> None:
    dev = cuda_device_or_skip()
    log, sdpa_log = spies
    cfg = ModelConfig.for_scale("D384", name)
    model = build_model(cfg, seed_index=902).to(dev).eval()
    ids = torch.randint(8, cfg.vocab_size, (2, 48), device=dev)  # ≤ 64 tokens, no grad: fla would switch
    chunk_op = "chunk_gla" if name == "O-gla3" else "chunk_gated_delta_rule"
    rec_op = "fused_recurrent_gla" if name == "O-gla3" else "fused_recurrent_gated_delta_rule"
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        log.clear(); sdpa_log.clear()
        _, cache = model.prefill(ids)
        assert set(log) == {chunk_op, "causal_conv1d"}, set(log)
        assert len(sdpa_log) == 3 and all(s["Tq"] == s["Tk"] for s in sdpa_log)
        log.clear(); sdpa_log.clear()
        model.score_continuation(cache, ids[:, :10])
        assert set(log) == {rec_op, "causal_conv1d"}, set(log)
        assert all(1 < s["Tq"] < s["Tk"] for s in sdpa_log)
        log.clear(); sdpa_log.clear()
        model.decode_step(ids[:, :1], cache)
        assert set(log) == {rec_op, "causal_conv1d"}, set(log)
        assert all(s["Tq"] == 1 for s in sdpa_log)
        log.clear()
        model(ids[:1].reshape(1, -1))  # training forward without grad, short: still chunk
        assert rec_op not in log
    # backend pinned for every bf16 inference SDPA call: FLASH only (no mem-efficient, no math)
    _, cache = None, None
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        sdpa_log.clear()
        _, cache = model.prefill(ids)
        model.score_continuation(cache, ids[:, :7])
        model.decode_step(ids[:, :1], cache)
    assert sdpa_log
    for s in sdpa_log:
        assert s["dtype"] == torch.bfloat16
        assert s["flash"] and not s["mem_efficient"] and not s["math"] and not s["cudnn"], s
    # fp32 inference: MATH only
    with torch.no_grad():
        sdpa_log.clear()
        _, cache = model.prefill(ids)
        model.score_continuation(cache, ids[:, :7])
    for s in sdpa_log:
        assert s["math"] and not s["flash"] and not s["mem_efficient"] and not s["cudnn"], s


@pytest.mark.gpu
def test_pinned_backend_raises_instead_of_falling_back() -> None:
    """FLASH cannot run head_dim 512; the pinned call must raise (no silent fallback)."""
    dev = cuda_device_or_skip()
    q = torch.randn(1, 2, 8, 512, device=dev, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError):
        A.attn_inference(q, q, q, scale=512 ** -0.5)
    # sanity: the same call works at head_dim 64
    q = torch.randn(1, 2, 8, 64, device=dev, dtype=torch.bfloat16)
    assert torch.isfinite(A.attn_inference(q, q, q, scale=0.125)).all()


@pytest.mark.gpu
def test_positional_call_detected_by_spy(spies) -> None:
    """The spy itself rejects a positional call (guards against a silent regression of the test)."""
    cuda_device_or_skip()
    with pytest.raises(AssertionError, match="positional"):
        L.fused_recurrent_gated_delta_rule(torch.zeros(1))


# ---------------------------------------------------------------------------------------------
# R-ATT (FEAS-PREC v2, PI ruling D083): the explicitly selected fp32 reference mode leaves the
# production pins untouched and never runs in half precision.
# ---------------------------------------------------------------------------------------------
@pytest.mark.gpu
def test_production_pins_unchanged_around_reference_mode(spies) -> None:
    dev = cuda_device_or_skip()
    _, sdpa_log = spies
    cfg = ModelConfig.for_scale("D384", "C3")
    model = build_model(cfg, seed_index=902).to(dev).eval()
    ids = torch.randint(8, cfg.vocab_size, (1, 96), device=dev)
    with torch.no_grad():
        with A.fp32_reference_attention("tiled"):
            sdpa_log.clear()
            _, cache = model.prefill(ids)
            model.score_continuation(cache, ids[:, :7])
            assert sdpa_log == []  # tiled fp32: no SDPA (MATH) call at all
            with pytest.raises(RuntimeError, match="fp32 only"), torch.autocast("cuda", torch.bfloat16):
                model.prefill(ids)
        sdpa_log.clear()
        with torch.autocast("cuda", torch.bfloat16):
            _, cache = model.prefill(ids)
            model.score_continuation(cache, ids[:, :7])
    assert sdpa_log and all(s["flash"] and not s["mem_efficient"] and not s["math"] and not s["cudnn"]
                            for s in sdpa_log)


def test_reference_mode_off_by_default() -> None:
    assert A.REFERENCE_FP32_MODE is None and A.INFERENCE_HALF_BACKEND == A.SDPBackend.FLASH_ATTENTION

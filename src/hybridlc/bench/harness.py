"""Benchmark-harness primitives (protocol §10.2; self-tested by VT-15).

* :func:`graphed` -- capture a callable into one CUDA graph (exact shape) and return its replay;
* :func:`time_replays` -- 5 warm-ups, then N >= 30 timed calls with CUDA events; optional cold L2
  (zero a 256 MiB buffer before every call, outside the timed region); min-of-N primary, median and
  IQR secondary;
* :func:`sdpa_spy` -- record every inference SDPA call of ``hybridlc.models.attention`` together with
  the backend flags active at call time (backend assertion, "no masks" assertion);
* :func:`decode_attention_case` / :func:`prefill_attention_case` / :func:`gdn_layer_case` -- the
  single-layer setups of Reviewer B's Appendix A (A5, A7), used for the reproduction check.

The model-level §10.2 benchmark driver (configs x shapes, interleaved blocks, idle-window gate) is
built on these primitives at S-benchmark time; this module is what VT-15 validates.
"""
from __future__ import annotations

import contextlib
import statistics
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

L2_FLUSH_BYTES = 256 * 2**20          # §10.2: zero a 256 MB buffer before every replay (L2 is 64 MiB)
N_TIMED, N_WARMUP = 30, 5


# ------------------------------------------------------------------------------------- timing
def graphed(fn: Callable[[], Any], warmup: int = 3) -> Callable[[], None]:
    """Capture ``fn`` (static inputs) into a CUDA graph; returns ``graph.replay``."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(warmup):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    return g.replay


_FLUSH: dict[int, torch.Tensor] = {}


def _flush_buf(nbytes: int) -> torch.Tensor:
    b = _FLUSH.get(nbytes)
    if b is None:
        b = _FLUSH[nbytes] = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
    return b


def time_replays(fn: Callable[[], Any], *, n: int = N_TIMED, warmup: int = N_WARMUP, cold_l2: bool = False,
                 flush_bytes: int = L2_FLUSH_BYTES) -> dict[str, float]:
    """Time ``fn`` with CUDA events. Returns microseconds: min (primary), median, IQR."""
    flush = _flush_buf(flush_bytes) if cold_l2 else None
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        if flush is not None:
            flush.zero_()                     # untimed: before the start event
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1e3)
    q = statistics.quantiles(ts, n=4)
    return dict(min_us=min(ts), median_us=statistics.median(ts), iqr_us=q[2] - q[0], n=n)


# ------------------------------------------------------------------------------------- SDPA spy
@dataclass
class SdpaCall:
    q_shape: tuple[int, ...]
    k_shape: tuple[int, ...]
    dtype: torch.dtype
    is_causal: bool
    mask_kind: str                     # "none" | "causal_bias" | "tensor"
    flash: bool
    cudnn: bool
    efficient: bool
    math: bool


@dataclass
class SdpaSpy:
    calls: list[SdpaCall] = field(default_factory=list)


@contextlib.contextmanager
def sdpa_spy() -> Iterator[SdpaSpy]:
    """Wrap ``hybridlc.models.attention._sdpa`` (the pointer every inference call uses) and record the
    backend flags in force at call time (``sdpa_kernel`` sets them)."""
    from hybridlc.models import attention as A

    spy = SdpaSpy()
    orig = A._sdpa  # noqa: SLF001

    def wrapped(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kw):
        if attn_mask is None:
            kind = "none"
        elif isinstance(attn_mask, torch.Tensor) and type(attn_mask) is torch.Tensor:
            kind = "tensor"
        else:
            kind = "causal_bias"
        b = torch.backends.cuda
        spy.calls.append(SdpaCall(tuple(q.shape), tuple(k.shape), q.dtype, bool(is_causal), kind,
                                  b.flash_sdp_enabled(), b.cudnn_sdp_enabled(), b.mem_efficient_sdp_enabled(),
                                  b.math_sdp_enabled()))
        return orig(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale, **kw)

    A._sdpa = wrapped  # noqa: SLF001
    try:
        yield spy
    finally:
        A._sdpa = orig  # noqa: SLF001


# ------------------------------------------------------------------------------------- Appendix-A cases
def decode_attention_case(B: int, L: int, backend: SDPBackend, H: int = 8, Dh: int = 64,
                          graph: bool = True) -> Callable[[], None]:
    """One-step decode attention over an exact-shape bf16 KV cache (no mask), §10.2 / A7."""
    kc = torch.randn(B, H, L, Dh, device="cuda", dtype=torch.bfloat16)
    vc = torch.randn_like(kc)
    q = torch.randn(B, H, 1, Dh, device="cuda", dtype=torch.bfloat16)

    def f():
        with sdpa_kernel([backend]):
            F.scaled_dot_product_attention(q, kc, vc)
    f()  # raises if the pinned backend cannot run
    return graphed(f) if graph else f


def prefill_attention_case(T: int, backend: SDPBackend, H: int = 8, Dh: int = 64) -> Callable[[], None]:
    """Causal attention core, B = 1, no grad (A5)."""
    qq = torch.randn(1, H, T, Dh, device="cuda", dtype=torch.bfloat16)
    kk, vv = torch.randn_like(qq), torch.randn_like(qq)

    def f():
        with torch.no_grad(), sdpa_kernel([backend]):
            F.scaled_dot_product_attention(qq, kk, vv, is_causal=True)
    f()
    return f


def gdn_layer_case(T: int, d: int = 512) -> Callable[[], None]:
    """Full fla GDN-A layer (projections + conv + chunk + gate), bf16, B = 1, no grad (A5)."""
    from fla.layers import GatedDeltaNet

    gdn = GatedDeltaNet(hidden_size=d, head_dim=64, num_heads=d // 128, expand_v=2, use_gate=True,
                        use_short_conv=True, conv_size=4, mode="chunk").to("cuda", torch.bfloat16).eval()
    x = torch.randn(1, T, d, device="cuda", dtype=torch.bfloat16)

    def f():
        with torch.no_grad():
            gdn(x)
    f()
    return f

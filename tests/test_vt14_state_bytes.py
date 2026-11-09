"""VT-14: inference-state bytes (protocol §10.1, §16).

``nbytes`` of the cache tensors after a (bf16-autocast) prefill at T ∈ {1K, 32K} equals the analytic
§3.4 value to the byte, per config and scale; batch scales linearly.
"""

from __future__ import annotations

import pytest
import torch

from hybridlc.models.analytic import state_bytes
from hybridlc.models.build import build_model
from hybridlc.models.config import ModelConfig
from hybridlc.models.patterns import TABLE_CONFIGS
from hybridlc.models.testing import cuda_device_or_skip

#: Protocol §3.4 "State MiB" column values at 1K and 32K (exact MiB, before rounding to 2 dp).
PROTOCOL_MIB = {("D512", "C0", 1024): 1.69, ("D512", "C12", 32768): 768.00, ("D384", "C3", 32768): 144.95,
                ("D512", "X-gc12", 1024): 24.28, ("D384", "X-E0", 1024): 3.52}


def _measure(scale: str, name: str, T: int, B: int = 1) -> dict[str, int]:
    dev = cuda_device_or_skip()
    cfg = ModelConfig.for_scale(scale, name)
    model = build_model(cfg, seed_index=0).to(dev).eval()
    ids = torch.randint(8, cfg.vocab_size, (B, T), device=dev)
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        _, cache = model.prefill(ids)
    nb = cache.state_nbytes()
    # dtypes as stored: bf16 KV, fp32 recurrent state and conv caches
    for st in cache.layers:
        for key, t in st.items():
            assert t.dtype == (torch.bfloat16 if key in ("k", "v") else torch.float32), (key, t.dtype)
    del model, cache
    torch.cuda.empty_cache()
    return nb


@pytest.mark.gpu
@pytest.mark.parametrize("scale", ["D384", "D512"])
@pytest.mark.parametrize("name", TABLE_CONFIGS + ["O-gla3", "O-nc3"])
def test_state_bytes_1k(scale: str, name: str) -> None:
    nb = _measure(scale, name, 1024)
    assert nb == state_bytes(ModelConfig.for_scale(scale, name), 1024), (nb,)


@pytest.mark.gpu
def test_state_bytes_batch_linear() -> None:
    nb = _measure("D384", "C3", 512, B=3)
    assert nb == state_bytes(ModelConfig.for_scale("D384", "C3"), 512, batch=3)


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("scale", ["D384", "D512"])
@pytest.mark.parametrize("name", TABLE_CONFIGS)
def test_state_bytes_32k(scale: str, name: str) -> None:
    nb = _measure(scale, name, 32768)
    assert nb == state_bytes(ModelConfig.for_scale(scale, name), 32768), (nb,)


def test_analytic_matches_protocol_mib() -> None:
    for (scale, name, T), mib in PROTOCOL_MIB.items():
        assert round(state_bytes(ModelConfig.for_scale(scale, name), T)["total"] / 2**20, 2) == mib
    # per-layer constants of §3.3
    assert state_bytes(ModelConfig.for_scale("D512", "C0"), 1)["total"] == 12 * 147_456
    assert state_bytes(ModelConfig.for_scale("D384", "C0"), 1)["total"] == 12 * 110_592
    assert state_bytes(ModelConfig.for_scale("D512", "X-E0"), 1)["total"] == 12 * 540_672
    assert state_bytes(ModelConfig.for_scale("D384", "X-E0"), 1)["total"] == 12 * 307_200
    assert state_bytes(ModelConfig.for_scale("D512", "O-nc0"), 1)["total"] == 12 * 131_072
    assert state_bytes(ModelConfig.for_scale("D384", "O-nc0"), 1)["total"] == 12 * 98_304

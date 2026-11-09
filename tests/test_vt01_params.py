"""VT-01: parameter counts per config and per mixer, both scales — exactly protocol §3.3–3.4.

Also checks that the analytic FLOP / state-byte columns of §3.4 are reproduced with the table's own
rounding, and cross-checks the closed-form fla counts against meta-device instantiation (the method
of ``scripts/planning/param_counts.py``).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import torch

from hybridlc.models.analytic import mixer_params, param_counts, table_row
from hybridlc.models.config import ModelConfig
from hybridlc.models.patterns import TABLE_CONFIGS

PROTOCOL = Path(__file__).resolve().parents[1] / "reports" / "experimental_protocol.md"

#: Protocol §3.3 mixer table (params D512, D384).
MIXER_TABLE = {
    "A": (1_053_696, 592_896),   # F_g
    "C": (1_059_840, 597_504),   # F_gc
    "-": (1_056_904, 595_334),   # GDN-A
    "E": (1_054_210, 594_050),   # GDN-E
    "N": (1_052_808, 592_262),   # GDN-nc
    "G": (1_065_344, 602_432),   # GLA
}


def _protocol_rows() -> list[list[str]]:
    rows = []
    for line in PROTOCOL.read_text().splitlines():
        if re.match(r"\| D(384|512) \| (C\d+|X-gc12|X-E0) \|", line):
            rows.append([c.strip() for c in line.strip().strip("|").split("|")])
    return rows


@pytest.mark.parametrize("ch", sorted(MIXER_TABLE))
@pytest.mark.parametrize("scale", ["D512", "D384"])
def test_mixer_params_analytic(ch: str, scale: str) -> None:
    cfg = ModelConfig.for_scale(scale, "C0")
    want = MIXER_TABLE[ch][0 if scale == "D512" else 1]
    assert mixer_params(cfg, ch) == want


@pytest.mark.parametrize("ch", sorted(MIXER_TABLE))
@pytest.mark.parametrize("scale", ["D512", "D384"])
def test_mixer_params_instantiated(ch: str, scale: str) -> None:
    """Our actual modules (built through build_mixer, on the meta device) have the table's counts."""
    from hybridlc.models.mixers import build_mixer

    cfg = ModelConfig.for_scale(scale, "C0")
    with torch.device("meta"):
        m = build_mixer(ch, cfg, layer_idx=0)
    assert sum(p.numel() for p in m.parameters()) == MIXER_TABLE[ch][0 if scale == "D512" else 1]


def test_protocol_table_rows_exact() -> None:
    rows = _protocol_rows()
    assert len(rows) == 2 * len(TABLE_CONFIGS) == 18
    for c in rows:
        r = table_row(c[0], c[1])
        mine = [r["scale"], r["config"], f"`{r['pattern']}`", f"{r['non_embedding']:,}", f"{r['embedding']:,}",
                f"{r['total']:,}", f"{r['mflop_4k']:.1f}", f"{r['mflop_32k']:.1f}", f"{r['mib_1k']:.2f}",
                f"{r['mib_4k']:.2f}", f"{r['mib_16k']:.2f}", f"{r['mib_32k']:.2f}", f"{r['train_pflop']}"]
        assert mine == c, (c, mine)


@pytest.mark.parametrize("scale", ["D512", "D384"])
@pytest.mark.parametrize("name", TABLE_CONFIGS + ["O-gla3", "O-nc3", "X-nope3", "P3-late", "X-E3"])
def test_built_model_counts_match_analytic(scale: str, name: str) -> None:
    from hybridlc.models.build import HybridLM

    cfg = ModelConfig.for_scale(scale, name)
    with torch.device("meta"):
        m = HybridLM(cfg, seed_index=None)
    n = sum(p.numel() for p in m.parameters())
    assert n == param_counts(cfg)["total"]


def test_parameter_matching_formula() -> None:
    """N(k) = 41,007,200 − 3,208·k (D512) and 21,309,384 − 2,438·k (D384); placement/NoPE equal C{k}."""
    for k in (0, 1, 2, 3, 4, 6, 12):
        assert param_counts(ModelConfig.for_scale("D512", f"C{k}"))["non_embedding"] == 41_007_200 - 3_208 * k
        assert param_counts(ModelConfig.for_scale("D384", f"C{k}"))["non_embedding"] == 21_309_384 - 2_438 * k
    for scale in ("D384", "D512"):
        for ks in (1, 2, 3):
            base = param_counts(ModelConfig.for_scale(scale, f"C{ks}"))["total"]
            for where in ("early", "middle", "late"):
                assert param_counts(ModelConfig.for_scale(scale, f"P{ks}-{where}"))["total"] == base
        for k in (3, 12):
            assert (param_counts(ModelConfig.for_scale(scale, f"X-nope{k}"))["total"]
                    == param_counts(ModelConfig.for_scale(scale, f"C{k}"))["total"])


def test_closed_form_equals_fla_meta_instantiation() -> None:
    """The method of scripts/planning/param_counts.py (fla layers on the meta device)."""
    from fla.layers import GatedDeltaNet, GatedLinearAttention

    def n(m):
        return sum(p.numel() for p in m.parameters())

    for d in (384, 512):
        cfg = ModelConfig.for_scale("D512" if d == 512 else "D384", "C0")
        H = d // 128
        with torch.device("meta"):
            assert n(GatedDeltaNet(hidden_size=d, head_dim=64, num_heads=H, expand_v=2, use_gate=True,
                                   use_short_conv=True, conv_size=4)) == mixer_params(cfg, "-")
            assert n(GatedDeltaNet(hidden_size=d, head_dim=d // 2, num_heads=1, expand_v=2, use_gate=True,
                                   use_short_conv=True, conv_size=4)) == mixer_params(cfg, "E")
            assert n(GatedLinearAttention(hidden_size=d, num_heads=H, expand_k=0.5, expand_v=1.0,
                                          use_short_conv=True, conv_size=4, use_output_gate=True)) == mixer_params(cfg, "G")


def test_ssmax_adds_h_params() -> None:
    cfg = ModelConfig.for_scale("D512", "C12", ssmax=True)
    assert mixer_params(cfg, "A") == MIXER_TABLE["A"][0] + 8

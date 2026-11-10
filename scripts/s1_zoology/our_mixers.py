"""S1 part (b): OUR mixers inside Zoology's MQAR harness (protocol §17, gate G-RP (b)).

Status: READY TO RUN (wrappers construct and train-step inside Zoology's ``LanguageModel``;
see ``test_s1_our_mixers.py``). The part-(b) runs themselves are scheduled later; this module
only provides the mixers and the run list.

The wrappers call the project's own mixer factory, ``hybridlc.models.mixers.build_mixer``
(INTERFACES.md ``models/mixers.py``), so S1(b) exercises exactly the code the LMs use.

How Zoology uses a sequence mixer
---------------------------------
``zoology.mixers.hybrid.Hybrid`` instantiates ``configs[layer_idx % 2]`` as
``import_from_str(name)(d_model=d, layer_idx=i, **kwargs)``. In the original MQAR config,
layer 0 is ``BaseConv(kernel 3)`` and layer 1 is the sequence mixer under test. The mixer is
called as ``mixer(u)`` with ``u: float32[B, T, d]`` (dense batch, one sequence per row, no
padding, no position ids) and must return ``float32[B, T, d]``. ``state_size(sequence_length)``
is optional (only logged). ``LanguageModel.__init__`` then applies ``_init_weights`` to every
sub-module: every ``nn.Linear`` / ``nn.Embedding`` gets N(0, 0.02), biases are zeroed, and every
parameter whose *name* contains ``out_proj.weight`` is re-drawn with std 0.02/sqrt(2·n_layers).

Wrappers (dotted paths for Zoology's ModuleConfig; put scripts/s1_zoology on sys.path)
-------------------------------------------------------------------------------------
* ``our_mixers.OurFgZoology``: F_g (pattern char ``A``) configured to match Zoology's MHA
  (protocol §17): 2 heads, head_dim = d/2, **no output gate, no QK-norm, no RoPE**, causal
  softmax with scale 1/sqrt(head_dim). Zoology's MHA output projection is called ``out_proj`` and
  receives the depth-rescaled init; our F_g calls it ``o_proj``. The wrapper registers the same
  tensor under the alias ``out_proj`` *before* ``inner`` so Zoology's name-based rescale applies
  to it too (``named_parameters`` reports a shared tensor under its first name).
* ``our_mixers.OurGdnZoology``: our fla-GDN mixer (pattern char ``-``, i.e. the GDN-A code path)
  in Zoology's geometry: 2 heads, d_k = d/2, expand_v = 2 (d_v = d), use_gate = False, short
  conv 4, conv_bias False (same as ``add_gated_delta_net`` in ``models_repo.py``). Zoology's
  GDN output projection is ``o_proj`` too, so neither gets the rescale (matched).
* ``our_mixers.OurGdnConfigA``: exploratory (protocol §17, 2 runs): GDN-A head geometry
  d_k = 64, d_v = 128, H = d/128, use_gate = True (all GDN-A defaults), at d ∈ {128, 256}.

Known residual differences (not configurable in our modules; report next to G-RP (b))
-------------------------------------------------------------------------------------
* F_g vs Zoology MHA: F_g has no biases (Zoology Wqkv/out_proj have bias=True, zero-initialised)
  and no attention dropout (Zoology 0.1). F_g uses separate q/k/v projections (same init
  distribution as the fused Wqkv). In fp32 on CUDA F_g runs per-row SDPA (MATH backend); Zoology
  materialises the scores with einsum. Same maths.
* GDN vs Zoology GDN: Zoology casts q, k, v, beta, g to **bf16** before ``chunk_gated_delta_rule``
  and computes the decay gate outside the kernel; our GDNMixer keeps the input dtype (fp32 in
  Zoology's fp32 harness) and computes the gate in-kernel (``use_gate_in_kernel=True``, identical
  formula ``-exp(A_log)·softplus(a + dt_bias)``). Zoology's GDN also holds an unused ``D``
  parameter (4 scalars; no effect on the output).

Run list (4 + 2 runs, protocol §17) is produced by :func:`part_b_runs` from Zoology's
original config objects, with only the sequence-mixer entry swapped.
"""
from __future__ import annotations

from typing import Any, Callable

import torch
from torch import nn

# Zoology geometry (models_repo.py add_attention / add_gated_delta_net at the pinned commit).
ZOOLOGY_NUM_HEADS = 2
ZOOLOGY_GDN_EXPAND_V = 2
ZOOLOGY_GDN_CONV = 4
ZOOLOGY_VOCAB = 8192
ZOOLOGY_N_LAYERS = 2
# GDN-A geometry (protocol §3.3).
GDN_A_HEAD_DIM = 64
GDN_A_EXPAND_V = 2

# Part (b) widths (protocol §17) and the exploratory config-A widths.
PART_B_WIDTHS = {"our_fg": (64, 128), "our_gdn": (128, 256)}
CONFIG_A_WIDTHS = (128, 256)


def _build_mixer(kind: str, d_model: int, layer_idx: int | None, **overrides: Any) -> nn.Module:
    """Call the hybridlc mixer factory (INTERFACES.md ``models/mixers.py``)."""
    from hybridlc.models.config import ModelConfig
    from hybridlc.models.mixers import build_mixer

    # A 2-layer config only carries shared constants (norm eps, conv size, GDN-A defaults);
    # its pattern is irrelevant because we build a single mixer. d_ff is unused (Zoology has
    # no MLP in these configs). attn_head_dim must divide d_model; the F_g head geometry is set
    # explicitly via overrides, so pick the largest valid default ≤ 64.
    attn_head_dim = 64 if d_model % 64 == 0 else d_model // ZOOLOGY_NUM_HEADS
    cfg = ModelConfig(d_model=d_model, d_ff=4 * d_model, n_layers=ZOOLOGY_N_LAYERS, vocab_size=ZOOLOGY_VOCAB,
                      pattern=kind * ZOOLOGY_N_LAYERS, attn_head_dim=attn_head_dim)
    return build_mixer(kind, cfg, layer_idx=0 if layer_idx is None else int(layer_idx), **overrides)


class _ZoologyMixerAdapter(nn.Module):
    """Adapts a hybridlc mixer (``forward(x, cu_seqlens=None, position_ids=None)``) to Zoology's call."""

    def __init__(self, inner: nn.Module, d_model: int, state_fn: Callable[[int], int]) -> None:
        super().__init__()
        self.inner = inner
        self.d_model = d_model
        self._state_fn = state_fn

    def forward(self, u: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        # Dense batch of independent sequences: no cu_seqlens; positions are implicit 0..T-1
        # (irrelevant here: RoPE is off for F_g and GDN has no positional encoding).
        return self.inner(u, cu_seqlens=None, position_ids=None)

    def state_size(self, sequence_length: int = 2048, **_: Any) -> int:
        return int(self._state_fn(sequence_length))


class OurFgZoology(_ZoologyMixerAdapter):
    """Our F_g as Zoology MHA: 2 × (d/2) heads, no gate, no QK-norm, no RoPE."""

    def __init__(self, d_model: int, layer_idx: int | None = None, num_heads: int = ZOOLOGY_NUM_HEADS,
                 **_: Any) -> None:
        head_dim = d_model // num_heads
        inner = _build_mixer("A", d_model, layer_idx, n_heads=num_heads, head_dim=head_dim,
                             qk_norm=False, rope=False, out_gate=False)
        nn.Module.__init__(self)
        # Alias registered first => named_parameters() reports it as "out_proj.weight", so
        # Zoology's name-based residual rescale (model.py::_init_weights) hits it, as for MHA.
        self.out_proj = inner.o_proj
        self.inner = inner
        self.d_model = d_model
        # Same accounting as zoology.mixers.attention.MHA.state_size: K and V per token.
        self._state_fn = lambda T: 2 * d_model * T


class OurFgProdZoology(_ZoologyMixerAdapter):
    """R4b: our *production* F_g (protocol §3.3 defaults: QK-RMSNorm, RoPE θ = cfg.rope_theta = 5e5,
    headwise output gate) with 2 heads × 64 (d = 128). Pre-registration: experiments/repro/R4/prereg.yaml
    reading R4b-1. The o_proj alias is registered exactly as in :class:`OurFgZoology`, so the only
    differences from ``our_fg-d128`` are QK-norm, RoPE and the gate."""

    autocast_bf16 = False

    def __init__(self, d_model: int, layer_idx: int | None = None, num_heads: int = ZOOLOGY_NUM_HEADS,
                 head_dim: int = 64, **_: Any) -> None:
        inner = _build_mixer("A", d_model, layer_idx, n_heads=num_heads, head_dim=head_dim)
        assert inner.use_qk_norm and inner.use_rope and inner.use_out_gate, "production F_g defaults changed"
        nn.Module.__init__(self)
        self.out_proj = inner.o_proj  # alias first (see OurFgZoology)
        self.inner = inner
        self.d_model = d_model
        self._state_fn = lambda T: 2 * num_heads * head_dim * T

    def forward(self, u: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        if not self.autocast_bf16:
            return self.inner(u, cu_seqlens=None, position_ids=None)
        # Reading R4b-2: autocast scoped to the mixer; output promoted back to Zoology's fp32 stream.
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = self.inner(u, cu_seqlens=None, position_ids=None)
        return y.float()


class OurFgProdBf16Zoology(OurFgProdZoology):
    """R4b bf16 arm: :class:`OurFgProdZoology` with its forward under bf16 autocast (varlen_attn / FA2)."""

    autocast_bf16 = True


class OurGdnZoology(_ZoologyMixerAdapter):
    """Our fla-GDN mixer in Zoology's geometry (2 heads, d_k = d/2, d_v = d, no gate, conv 4)."""

    def __init__(self, d_model: int, layer_idx: int | None = None, num_heads: int = ZOOLOGY_NUM_HEADS,
                 **_: Any) -> None:
        head_dim = d_model // num_heads
        inner = _build_mixer("-", d_model, layer_idx, num_heads=num_heads, head_dim=head_dim,
                             expand_v=ZOOLOGY_GDN_EXPAND_V, use_gate=False, use_short_conv=True,
                             conv_size=ZOOLOGY_GDN_CONV, conv_bias=False)
        dk, dv = head_dim, head_dim * ZOOLOGY_GDN_EXPAND_V
        # Same accounting as zoology.mixers.gated_delta_net.GatedDeltaNet.state_size.
        super().__init__(inner, d_model, lambda T: num_heads * dk * dv)


class OurGdnConfigA(_ZoologyMixerAdapter):
    """Exploratory: GDN-A head geometry (d_k 64, d_v 128, H = d/128, output gate on)."""

    def __init__(self, d_model: int, layer_idx: int | None = None, **_: Any) -> None:
        h = d_model // (GDN_A_HEAD_DIM * GDN_A_EXPAND_V)
        assert h >= 1, f"config-A geometry needs d_model >= 128, got {d_model}"
        inner = _build_mixer("-", d_model, layer_idx)  # defaults = GDN-A (protocol §3.3)
        super().__init__(inner, d_model, lambda T: h * GDN_A_HEAD_DIM * GDN_A_HEAD_DIM * GDN_A_EXPAND_V)


# ---------------------------------------------------------------------------------------------
# Run list for part (b). Imports Zoology lazily (via s1_configs/zoology_compat) so this module
# stays importable as a mixer path without side effects.
# ---------------------------------------------------------------------------------------------
_SWAP = {"our_fg": ("attention", "our_mixers.OurFgZoology"),
         "our_gdn": ("gated_delta_net", "our_mixers.OurGdnZoology"),
         "our_gdn_configA": ("gated_delta_net", "our_mixers.OurGdnConfigA")}


def best_lr_from_s1a(results_json: str | None = None) -> dict[tuple[str, int], float]:
    """Zoology's best LR per (family, d_model) from ``experiments/s1_zoology/s1a_results.json``."""
    import json
    from pathlib import Path

    from zoology_compat import RESULTS_DIR

    path = Path(results_json) if results_json else RESULTS_DIR / "s1a_results.json"
    best = json.loads(path.read_text())["grp_a"]["best_over_lr"]
    return {(fam, int(d)): float(cell["lr"]) for fam, cells in best.items() for d, cell in cells.items()}


def part_b_runs(best_lr: dict[tuple[str, int], float]) -> list[Any]:
    """Build the 4 + 2 part-(b) TrainConfigs.

    ``best_lr[(zoology_family, d_model)]`` is Zoology's best LR from S1(a) at that width
    (protocol §17 "at Zoology's best LR"; G-RP (b) compares "at matched d_model and LR", so the
    best LR is taken per width). The exploratory config-A runs reuse GDN's best LR at that width.
    """
    from s1_configs import build_runs  # Zoology's original configs (data, model, LR grid)

    base = {(r.family, r.d_model): r for r in build_runs()
            if (r.family, r.d_model) in best_lr and abs(r.lr - best_lr[(r.family, r.d_model)]) < 1e-12}
    out = []
    for tag, (fam, mixer_path) in _SWAP.items():
        widths = CONFIG_A_WIDTHS if tag == "our_gdn_configA" else PART_B_WIDTHS[tag]
        for d in widths:
            cfg = base[(fam, d)].config.model_copy(deep=True)
            seq = cfg.model.sequence_mixer.kwargs["configs"]
            assert seq[0]["name"].endswith("BaseConv"), seq
            seq[1] = {"name": mixer_path, "kwargs": {}}
            cfg.model.name = tag
            cfg.run_id = f"{tag}-d{d}-lr{cfg.learning_rate:.1e}"
            out.append(cfg)
    return out


# R4b (plan §2.4): production F_g at d = 128, lr 1e-3 (= Zoology attention's best LR at d = 128).
R4B_RUNS = {"our_fg_prod": "our_mixers.OurFgProdZoology", "our_fg_prod_bf16": "our_mixers.OurFgProdBf16Zoology"}
R4B_D_MODEL = 128


def r4b_runs(best_lr: dict[tuple[str, int], float]) -> list[Any]:
    """The 2 R4b TrainConfigs: Zoology's attention d128 config at its best LR, mixer swapped."""
    from s1_configs import build_runs

    lr = best_lr[("attention", R4B_D_MODEL)]
    base = next(r for r in build_runs()
                if r.family == "attention" and r.d_model == R4B_D_MODEL and abs(r.lr - lr) < 1e-12)
    out = []
    for tag, mixer_path in R4B_RUNS.items():
        cfg = base.config.model_copy(deep=True)
        seq = cfg.model.sequence_mixer.kwargs["configs"]
        assert seq[0]["name"].endswith("BaseConv"), seq
        seq[1] = {"name": mixer_path, "kwargs": {}}
        cfg.model.name = tag
        cfg.run_id = f"{tag}-d{R4B_D_MODEL}-lr{cfg.learning_rate:.1e}"
        out.append(cfg)
    return out


def r4_runs() -> list[Any]:
    """All 8 R4 runs in queue order: the 4 G-RP(b) cells, the 2 R4b runs, then the 2 config-A runs."""
    best = best_lr_from_s1a()
    pb = part_b_runs(best)
    grp = [c for c in pb if not c.run_id.startswith("our_gdn_configA")]
    cfa = [c for c in pb if c.run_id.startswith("our_gdn_configA")]
    return grp + r4b_runs(best) + cfa


# ---------------------------------------------------------------------------------------------
# R5 (plan §2.5; prereg experiments/repro/R5/prereg.yaml): MQAR at d_model = 64, Zoology's GDN vs ours
# under bf16 autocast, seeds (123,123), (124,124), (125,125) [+ (126,126), (127,127) only if O5 fires].
# ---------------------------------------------------------------------------------------------
class OurGdnBf16Zoology(OurGdnZoology):
    """R5 arm: :class:`OurGdnZoology` with its forward under ``torch.autocast('cuda', bfloat16)`` (production
    precision: bf16 projections / conv / kernel inputs, fp32 state inside the fla kernel) and the output promoted
    back to Zoology's fp32 residual stream (prereg reading P1, same scoping as R4b-2)."""

    autocast_bf16 = True

    def forward(self, u: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = self.inner(u, cu_seqlens=None, position_ids=None)
        return y.float()


R5_D_MODEL = 64
R5_LR = 1e-3
R5_SEEDS = (123, 124, 125)
O5_SEEDS = (126, 127)  # prereg reading BM1: only if the bimodality rule fires
R5_ARMS = {"gdn_zoology": None, "our_gdn_bf16": "our_mixers.OurGdnBf16Zoology"}  # None = Zoology's own mixer


def _zoology_gdn_d64_base() -> Any:
    """Zoology's own d64 GDN TrainConfig at lr 1e-3 from original_mqar_configs.py (as R1: cache_dir, logger only)."""
    from s1_configs import _load_original_module
    from zoology.config import LoggerConfig
    from zoology_compat import DATA_CACHE_DIR

    mod = _load_original_module()
    hits = [c for c in mod.configs if c.model.name == "gated_delta_net" and c.model.d_model == R5_D_MODEL
            and abs(float(c.learning_rate) - R5_LR) < 1e-12]
    assert len(hits) == 1, f"expected one Zoology GDN d64 lr1e-3 config, got {len(hits)}"
    cfg = hits[0].model_copy(deep=True)
    cfg.data.cache_dir = str(DATA_CACHE_DIR)
    cfg.logger = LoggerConfig()
    return cfg


def r5_run_id(arm: str, seed: int) -> str:
    return f"{arm}-d{R5_D_MODEL}-lr{R5_LR:.1e}-s{seed}"


def r5_runs(include_o5: bool = False) -> list[Any]:
    """R5 TrainConfigs in queue order, interleaved by seed (Zoology then ours per seed)."""
    base = _zoology_gdn_d64_base()
    seeds = R5_SEEDS + (O5_SEEDS if include_o5 else ())
    out = []
    for s in seeds:
        for arm, mixer_path in R5_ARMS.items():
            cfg = base.model_copy(deep=True)
            seq = cfg.model.sequence_mixer.kwargs["configs"]
            assert seq[0]["name"].endswith("BaseConv") and seq[1]["name"].endswith("GatedDeltaNet"), seq
            if mixer_path is not None:
                seq[1] = {"name": mixer_path, "kwargs": {}}
                cfg.model.name = arm
            cfg.seed = int(s)
            cfg.data.seed = int(s)
            cfg.run_id = r5_run_id(arm, s)
            out.append(cfg)
    return out

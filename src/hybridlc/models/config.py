"""Model configuration (protocol §3.1–3.3).

Pattern characters (layer 0 → 11, left to right):

=====  =========  ==========================================================
char   mixer      definition
=====  =========  ==========================================================
``-``  GDN-A      fla GatedDeltaNet, d//128 heads × (64 / 128), conv 4, gate
``A``  F_g        full MHA d//64 × 64, headwise sigmoid gate, QK-RMSNorm, RoPE
``C``  F_gc       F_g + causal depthwise ShortConv(4) on q, k, v
``E``  GDN-E      GatedDeltaNet with 1 head × (d/2 / d)
``G``  GLA        fla GatedLinearAttention (Tier C)
``N``  GDN-nc     GDN-A without its short conv (Tier C)
=====  =========  ==========================================================
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

#: Attention-type ("full") mixer characters and linear-type mixer characters.
FULL_MIXERS = frozenset("AC")
LINEAR_MIXERS = frozenset("-EGN")
ALL_MIXERS = FULL_MIXERS | LINEAR_MIXERS

#: Per-scale constants (protocol §3.1 table and §6.3 token budgets).
SCALES: dict[str, dict[str, Any]] = {
    "D384": dict(d_model=384, d_ff=1024, stage_a_tokens=600_000_000, stage_b_mid_tokens=60_000_000),
    "D512": dict(d_model=512, d_ff=1536, stage_a_tokens=800_000_000, stage_b_mid_tokens=200_000_000),
}

#: TEF tokens per run (1,000 steps × 65,536; protocol §6.3).
TEF_TOKENS: int = 65_536_000


@dataclass
class ModelConfig:
    d_model: int
    d_ff: int
    n_layers: int = 12
    vocab_size: int = 16384
    pattern: str = "------------"
    attn_head_dim: int = 64
    rope_theta: float = 5e5
    nope: bool = False
    ssmax: bool = False
    init_std: float = 0.02
    tie_embeddings: bool = True
    norm_eps: float = 1e-5
    # Fixed by the protocol; exposed so that analytic code and modules agree on one value.
    conv_size: int = 4
    #: GDN-A geometry: heads = d // gdn_head_ratio, head_dim (d_k) = gdn_head_dim, expand_v = 2.
    gdn_head_dim: int = 64
    gdn_expand_v: float = 2.0
    #: Free-form label of the named config (e.g. "C3", "X-gc12"); not semantic for hashing purposes
    #: beyond being recorded.
    name: str = ""

    def __post_init__(self) -> None:
        if len(self.pattern) != self.n_layers:
            raise ValueError(f"pattern length {len(self.pattern)} != n_layers {self.n_layers}")
        bad = set(self.pattern) - ALL_MIXERS
        if bad:
            raise ValueError(f"unknown mixer chars {sorted(bad)} in pattern {self.pattern!r}")
        if self.d_model % self.attn_head_dim:
            raise ValueError("d_model must be divisible by attn_head_dim")
        if not self.tie_embeddings:
            raise ValueError("protocol §3.1 fixes tied embeddings")

    # ---- derived geometry -------------------------------------------------------------
    @property
    def n_attn_heads(self) -> int:
        return self.d_model // self.attn_head_dim

    @property
    def k(self) -> int:
        """Number of full-attention mixers (F_g or F_gc)."""
        return sum(c in FULL_MIXERS for c in self.pattern)

    def linear_geometry(self, char: str) -> dict[str, Any]:
        """Constructor kwargs of the fla linear mixer for pattern char ``char`` (protocol §3.3)."""
        d = self.d_model
        if char in "-N":
            return dict(kind="gdn", hidden_size=d, head_dim=self.gdn_head_dim, num_heads=d // 128,
                        expand_v=self.gdn_expand_v, use_gate=True, use_short_conv=(char == "-"),
                        conv_size=self.conv_size, conv_bias=False, allow_neg_eigval=False,
                        norm_eps=self.norm_eps, mode="chunk")
        if char == "E":
            return dict(kind="gdn", hidden_size=d, head_dim=d // 2, num_heads=1,
                        expand_v=self.gdn_expand_v, use_gate=True, use_short_conv=True,
                        conv_size=self.conv_size, conv_bias=False, allow_neg_eigval=False,
                        norm_eps=self.norm_eps, mode="chunk")
        if char == "G":
            # Protocol §3.3 lists exactly these kwargs; everything else is fla's default.
            return dict(kind="gla", hidden_size=d, num_heads=d // 128, expand_k=0.5, expand_v=1.0,
                        use_short_conv=True, conv_size=self.conv_size, use_output_gate=True, mode="chunk")
        raise ValueError(f"{char!r} is not a linear mixer")

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ModelConfig":
        names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})

    @classmethod
    def for_scale(cls, scale: str, pattern: str, **overrides: Any) -> "ModelConfig":
        """``ModelConfig.for_scale('D384'|'D512', pattern_or_named_config, **overrides)``.

        ``pattern`` may be a raw 12-char pattern or a named config understood by
        :func:`hybridlc.models.patterns.named_config` (e.g. ``"C3"``, ``"X-nope12"``,
        ``"P2-late"``, ``"X-E3"``); named configs may carry overrides (e.g. ``nope=True``).
        """
        if scale not in SCALES:
            raise ValueError(f"unknown scale {scale!r}; expected one of {sorted(SCALES)}")
        s = SCALES[scale]
        kw: dict[str, Any] = dict(d_model=s["d_model"], d_ff=s["d_ff"])
        if len(pattern) == overrides.get("n_layers", 12) and set(pattern) <= ALL_MIXERS:
            kw["pattern"] = pattern
        else:
            from .patterns import named_config

            pat, extra = named_config(pattern)
            kw["pattern"] = pat
            kw["name"] = pattern
            kw.update(extra)
        kw.update(overrides)
        return cls(**kw)

    @property
    def scale(self) -> str:
        for name, s in SCALES.items():
            if s["d_model"] == self.d_model and s["d_ff"] == self.d_ff:
                return name
        return f"d{self.d_model}"

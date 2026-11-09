"""Inference cache (protocol §9.1 query branching, §10.1 state bytes, VT-14).

Per layer one dict of tensors, keyed by what they are:

  full layers (F_g / F_gc) : ``k``, ``v`` [B, T, H, 64] in the compute dtype (bf16 under autocast)
                             and, for F_gc, ``conv`` fp32 [B, 3d, 4];
  linear layers (GDN / GLA): ``state`` fp32 [B, HV, d_v, d_k] and ``conv_q``/``conv_k``/``conv_v``
                             fp32 [B, D, 4] (absent for GDN-nc).

``last_logits`` (fp32 [B, V]) is the prediction for the next token after the cached prefix; it is
kept so that teacher-forced scoring of a continuation can start from a snapshot.  It is *not*
inference state in the protocol sense and is excluded from :meth:`state_nbytes`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

_KV_KEYS = ("k", "v")
_STATE_KEYS = ("state",)
_CONV_KEYS = ("conv", "conv_q", "conv_k", "conv_v")


@dataclass
class InferenceCache:
    pattern: str
    layers: list[dict[str, torch.Tensor]]
    seq_len: int
    last_logits: torch.Tensor | None = None
    meta: dict = field(default_factory=dict)

    @property
    def batch_size(self) -> int:
        for st in self.layers:
            for t in st.values():
                return int(t.shape[0])
        return 0

    def clone(self) -> "InferenceCache":
        """Deep copy of every tensor (the snapshot used for query branching)."""
        return InferenceCache(
            pattern=self.pattern,
            layers=[{k: t.clone() for k, t in st.items()} for st in self.layers],
            seq_len=self.seq_len,
            last_logits=None if self.last_logits is None else self.last_logits.clone(),
            meta=dict(self.meta),
        )

    def state_nbytes(self) -> dict[str, int]:
        """Bytes of the stored state tensors, by category (KV, recurrent state, conv caches)."""
        out = {"kv": 0, "gdn_state": 0, "conv": 0}
        for st in self.layers:
            for key, t in st.items():
                nb = t.numel() * t.element_size()
                if key in _KV_KEYS:
                    out["kv"] += nb
                elif key in _STATE_KEYS:
                    out["gdn_state"] += nb
                elif key in _CONV_KEYS:
                    out["conv"] += nb
                else:  # pragma: no cover - guard against silently uncounted tensors
                    raise KeyError(f"uncategorised cache tensor {key!r}")
        out["total"] = out["kv"] + out["gdn_state"] + out["conv"]
        return out

    def assign(self, other: "InferenceCache") -> "InferenceCache":
        """Replace this cache's contents with ``other``'s (used by ``decode_step``)."""
        self.layers, self.seq_len, self.last_logits = other.layers, other.seq_len, other.last_logits
        return self

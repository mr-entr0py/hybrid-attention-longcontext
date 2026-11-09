"""Single-pass reference implementation of the scoring contract (INTERFACES.md, models).

``ForwardOnlyAdapter(model)`` wraps any module with the training contract
``forward(input_ids[1,N], cu_seqlens, position_ids) -> hidden[1,N,d]`` and
``lm_head_weight`` and exposes ``prefill`` / ``score_continuation`` / ``decode_step`` by
*recomputing* full forwards. It is

* the "single-pass forward" of the 5% re-scoring check (protocol §9.2, I-29), and
* a test double for the scorer (any toy model becomes scoreable).

It is exact but slow (O(T) per continuation); never use it for the production path.

Alignment convention (also written into INTERFACES.md): ``score_continuation(cache,
cont[B,M])`` returns ``logprobs[b, j] = log p(cont[b, j] | context, cont[b, :j])`` and
``argmax[b, j] = argmax_v p(v | context, cont[b, :j])``. Position j = 0 is predicted from
the logits at the end of the cached context.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class RecomputeCache:
    """Cache = the context ids themselves (B, T)."""

    ids: torch.Tensor

    def clone(self) -> "RecomputeCache":
        return RecomputeCache(self.ids.clone())

    def state_nbytes(self) -> dict:
        n = self.ids.numel() * self.ids.element_size()
        return dict(kv=0, gdn_state=0, conv=0, total=n)


class ForwardOnlyAdapter:
    """Scoring-contract view of a forward-only LM (see module docstring)."""

    scoring_path = "single_pass_forward"

    def __init__(self, model: torch.nn.Module, chunk: int = 4096):
        self.model = model
        self.chunk = chunk

    @property
    def device(self) -> torch.device:
        return self.model.lm_head_weight.device

    @torch.no_grad()
    def logits(self, ids_row: torch.Tensor) -> torch.Tensor:
        """fp32 logits [N, V] of one sequence (single document)."""
        ids_row = ids_row.to(self.device)
        n = ids_row.numel()
        if hasattr(self.model, "full_logits"):  # HybridLM's own single-pass path
            return self.model.full_logits(ids_row[None].long())[0].float()
        cu = torch.tensor([0, n], dtype=torch.int32, device=self.device)
        pos = torch.arange(n, device=self.device)[None]
        h = self.model(ids_row[None].long(), cu, pos)[0]
        w = self.model.lm_head_weight
        return torch.cat([(h[a:a + self.chunk].float() @ w.float().T) for a in range(0, n, self.chunk)])

    @torch.no_grad()
    def prefill(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, RecomputeCache]:
        last = torch.stack([self.logits(r)[-1] for r in input_ids])
        return last, RecomputeCache(input_ids.clone())

    @torch.no_grad()
    def decode_step(self, token_ids: torch.Tensor, cache: RecomputeCache) -> tuple[torch.Tensor, RecomputeCache]:
        ids = torch.cat([cache.ids.to(token_ids.device), token_ids], dim=1)
        last = torch.stack([self.logits(r)[-1] for r in ids])
        return last, RecomputeCache(ids)

    @torch.no_grad()
    def score_continuation(self, cache: RecomputeCache, cont_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        T, M = cache.ids.shape[1], cont_ids.shape[1]
        lps, ams = [], []
        for b in range(cont_ids.shape[0]):
            full = torch.cat([cache.ids[b].to(cont_ids.device), cont_ids[b]])
            lg = self.logits(full)[T - 1:T - 1 + M]  # predicts cont[0..M-1]
            lp = torch.log_softmax(lg, -1)
            tgt = cont_ids[b].to(lp.device).long()
            lps.append(lp.gather(1, tgt[:, None])[:, 0])
            ams.append(lg.argmax(-1))
        return torch.stack(lps), torch.stack(ams)

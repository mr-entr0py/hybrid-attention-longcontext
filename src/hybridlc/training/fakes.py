"""Tiny stand-ins for the model / data / loss contracts, used by the training-infra tests.

They follow INTERFACES.md exactly (signatures and return shapes), so the trainer exercised with
them is the same code that runs the real ``HybridLM`` / ``StageStream`` / ``chunked_ce``. They are
selected through ``RunConfig.impl`` import strings, e.g.
``impl.model_factory = "hybridlc.training.fakes:build_fake_model"``.

Nothing here uses the global RNG: all randomness is drawn from SeedSequence-derived generators.
"""
from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .compat import STREAMS, seed_sequence, torch_generator


# ----------------------------------------------------------------------------------------------
# model
# ----------------------------------------------------------------------------------------------
class _FakeAttn(nn.Module):
    """Causal single-head attention over packed documents (dense mask; tiny N only).

    Exposes the optional attention-statistics hook of INTERFACES.md: when ``collect_attn_stats``
    is True the forward stores ``attn_stats = {"max_logit": float, "entropy": float}``.
    """

    def __init__(self, d: int):
        super().__init__()
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.gate = nn.Linear(d, 1, bias=False)  # stands in for W_g (ndim 2 -> decayed)
        self.q_norm = nn.Parameter(torch.ones(d))  # 1-D gain -> no decay
        self.collect_attn_stats = False
        self.attn_stats: dict[str, float] | None = None

    def forward(self, x: torch.Tensor, doc_ids: torch.Tensor) -> torch.Tensor:
        n, d = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = F.rms_norm(q, (d,), self.q_norm)
        logits = (q @ k.T) / d ** 0.5
        causal = torch.ones(n, n, dtype=torch.bool, device=x.device).tril()
        mask = causal & (doc_ids[:, None] == doc_ids[None, :])
        logits = logits.masked_fill(~mask, float("-inf"))
        p = logits.float().softmax(-1)
        if self.collect_attn_stats:
            with torch.no_grad():
                ent = -(p * p.clamp_min(1e-30).log()).sum(-1).mean()
                self.attn_stats = {"max_logit": logits[mask].float().max().item(), "entropy": ent.item()}
        o = (p.to(v.dtype) @ v) * torch.sigmoid(self.gate(x))
        return self.o(o)


class _FakeBlock(nn.Module):
    def __init__(self, d: int, d_ff: int, mixer: str):
        super().__init__()
        self.n1 = nn.RMSNorm(d)
        self.n2 = nn.RMSNorm(d)
        self.mixer_kind = mixer
        if mixer == "A":
            self.mix = _FakeAttn(d)
        else:  # stands in for a linear mixer: causal depthwise conv + projection
            self.conv = nn.Conv1d(d, d, 4, groups=d, bias=False)  # 3-D kernel -> decayed
            self.proj = nn.Linear(d, d, bias=False)
            self.A_log = nn.Parameter(torch.zeros(d))
            self.A_log._no_weight_decay = True  # type: ignore[attr-defined]
        self.up = nn.Linear(d, 2 * d_ff, bias=False)
        self.down = nn.Linear(d_ff, d, bias=False)

    def forward(self, h: torch.Tensor, doc_ids: torch.Tensor) -> torch.Tensor:
        x = self.n1(h)
        if self.mixer_kind == "A":
            m = self.mix(x, doc_ids)
        else:
            # causal conv that does not cross documents: zero inputs from other docs per shift
            xs = []
            for s in range(4):
                if s == 0:
                    xs.append(x)
                    continue
                sh = torch.zeros_like(x)
                sh[s:] = x[:-s]
                same = torch.zeros_like(doc_ids, dtype=torch.bool)
                same[s:] = doc_ids[s:] == doc_ids[:-s]
                xs.append(sh * same[:, None])
            w = self.conv.weight[:, 0, :]  # [d, 4] ; tap 3 multiplies the current token
            c = sum(xs[s] * w[:, 3 - s] for s in range(4))
            m = self.proj(F.silu(c)) * torch.exp(-torch.exp(self.A_log))
        u = h + m
        a, b = self.up(self.n2(u)).chunk(2, dim=-1)
        return u + self.down(F.silu(a) * b)


class FakeHybridLM(nn.Module):
    """``HybridLM`` contract: ``forward(input_ids[1,N], cu_seqlens, position_ids) -> hidden[1,N,d]``."""

    def __init__(self, cfg: Any, seed_index: int):
        super().__init__()
        get = (lambda k: cfg[k]) if isinstance(cfg, dict) else (lambda k: getattr(cfg, k))
        d, d_ff, V, pattern = get("d_model"), get("d_ff"), get("vocab_size"), get("pattern")
        self.embed = nn.Embedding(V, d)
        self.blocks = nn.ModuleList(_FakeBlock(d, d_ff, "A" if c in "AC" else "-") for c in pattern)
        self.norm_f = nn.RMSNorm(d)
        self._init(seed_index, get("init_std"))

    def _init(self, seed_index: int, std: float) -> None:
        gen = torch_generator(seed_index, STREAMS["init_shared"], 12345)
        with torch.no_grad():
            for name, p in self.named_parameters():
                if p.ndim >= 2:
                    p.copy_(torch.randn(p.shape, generator=gen) * std)

    @property
    def lm_head_weight(self) -> torch.Tensor:
        return self.embed.weight

    def forward(self, input_ids: torch.Tensor, cu_seqlens: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        ids = input_ids[0]
        n = ids.shape[0]
        lens = (cu_seqlens[1:] - cu_seqlens[:-1]).long()
        doc_ids = torch.repeat_interleave(torch.arange(len(lens), device=ids.device), lens)
        assert doc_ids.shape[0] == n, "cu_seqlens must cover the packed sequence"
        h = self.embed(ids)
        for blk in self.blocks:
            h = blk(h, doc_ids)
        return self.norm_f(h)[None]


def build_fake_model(cfg: Any, seed_index: int) -> FakeHybridLM:
    return FakeHybridLM(cfg, seed_index)


# ----------------------------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------------------------
_STAGE_CODE = {"A": 2, "B": 3, "LR": 2, "LM2": 7, "TEF": 4}
IGNORE_INDEX = 1  # <|pad|>, the ignore_index of chunked_ce (as in hybridlc.data.loader)


def _pack(rng: np.random.Generator, n_tokens: int, vocab: int, mean_doc: int) -> tuple[np.ndarray, list[int]]:
    """EOS(0)-prefixed documents packed into ``n_tokens``; returns ids and document starts."""
    ids = np.empty(n_tokens, dtype=np.int64)
    starts, pos = [], 0
    while pos < n_tokens:
        L = int(min(n_tokens - pos, max(2, rng.geometric(1.0 / mean_doc))))
        starts.append(pos)
        ids[pos] = 0
        # a learnable toy distribution: tokens follow (prev*7+3) % V with noise
        x = int(rng.integers(8, vocab))
        for i in range(1, L):
            x = (x * 7 + 3) % vocab if rng.random() < 0.8 else int(rng.integers(8, vocab))
            ids[pos + i] = max(x, 8)
        pos += L
    return ids, starts


class FakeStageStream:
    """``StageStream`` contract: pure function of (data_seed, stage, step, micro)."""

    vocab_size = 256

    def __init__(self, stage: str, data_seed: int, seq_len: int, micro_batch: int, accum: int,
                 mix: dict | None = None, vocab_size: int | None = None, mean_doc: int = 48):
        self.stage, self.data_seed, self.seq_len = stage, data_seed, seq_len
        self.micro_batch, self.accum, self.mix = micro_batch, accum, mix or {}
        self.vocab = vocab_size or self.vocab_size
        self.mean_doc = mean_doc

    def _rng(self, step: int, micro: int) -> np.random.Generator:
        return np.random.default_rng(seed_sequence(self.data_seed, _STAGE_CODE[self.stage], step, micro))

    def get(self, step: int, micro: int) -> dict[str, Any]:
        """Same semantics as the real loader: ``labels[t]`` is the next token of the source stream
        (so a document's last token predicts the next ``<|endoftext|>``); the final position of each
        sequence has no continuation here and gets ``IGNORE_INDEX`` (= pad id 1)."""
        rng = self._rng(step, micro)
        all_ids, cu, pos_ids, labels = [], [0], [], []
        for j in range(self.micro_batch):
            ids, starts = _pack(rng, self.seq_len + 1, self.vocab, self.mean_doc)
            lab = ids[1:].copy()
            lab[-1] = IGNORE_INDEX
            ids = ids[:-1]
            starts = [s for s in starts if s < self.seq_len]
            bounds = starts + [self.seq_len]
            pos = np.empty_like(ids)
            for a, b in zip(bounds[:-1], bounds[1:]):
                pos[a:b] = np.arange(b - a)
                cu.append(j * self.seq_len + b)
            all_ids.append(ids)
            labels.append(lab)
            pos_ids.append(pos)
        ids = np.concatenate(all_ids)
        lab = np.concatenate(labels)
        h = hashlib.sha256()
        h.update(ids.astype("<u2").tobytes())
        h.update(lab.astype("<u2").tobytes())
        h.update(np.asarray(cu, dtype="<i4").tobytes())
        return {
            "input_ids": torch.from_numpy(ids)[None],
            "labels": torch.from_numpy(lab)[None],
            "cu_seqlens": torch.tensor(cu, dtype=torch.int32),
            "position_ids": torch.from_numpy(np.concatenate(pos_ids))[None],
            "batch_hash": h.hexdigest(),
        }


class FakeTEFStream(FakeStageStream):
    """``TEFStream`` contract: as StageStream plus ``loss_weights[1,N]``.

    Weights implement the D014 ½/½ loss *per optimiser step*: answer tokens share ½ and all other
    loss tokens share ½, normalised over all micro-batches of the step (so Σw over the step is 1).
    """

    def __init__(self, seed_index: int, steps: int, seq_len: int = 64, micro_batch: int = 1,
                 accum: int = 2, **kw: Any):
        super().__init__("TEF", seed_index, seq_len, micro_batch, accum, **kw)
        self.steps = steps

    def _answer_mask(self, step: int, micro: int, labels: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng(seed_sequence(self.data_seed, STREAMS["tef"], step, micro, 99))
        return (rng.random(labels.shape) < 0.1) & (labels != IGNORE_INDEX)

    def get(self, step: int, micro: int) -> dict[str, Any]:
        b = super().get(step, micro)
        n_ans = n_oth = 0
        masks = []
        for m in range(self.accum):
            lab = super().get(step, m)["labels"][0].numpy() if m != micro else b["labels"][0].numpy()
            am = self._answer_mask(step, m, lab)
            n_ans += int(am.sum())
            n_oth += int(((lab != IGNORE_INDEX) & ~am).sum())
            if m == micro:
                masks.append(am)
        am = masks[0]
        lab = b["labels"][0].numpy()
        w = np.zeros(lab.shape, dtype=np.float32)
        if n_ans:
            w[am] = 0.5 / n_ans
        if n_oth:
            w[(lab != IGNORE_INDEX) & ~am] = (0.5 if n_ans else 1.0) / n_oth
        b["loss_weights"] = torch.from_numpy(w)[None]
        return b


# ----------------------------------------------------------------------------------------------
# loss
# ----------------------------------------------------------------------------------------------
def fake_chunked_ce(hidden: torch.Tensor, weight: torch.Tensor, labels: torch.Tensor, chunk: int = 4096,
                    z_loss_coef: float = 1e-4, ignore_index: int = 1,
                    loss_weights: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference implementation of the ``chunked_ce`` contract as clarified in INTERFACES.md.

    loss   = Σ_i w_i·CE_i / Σ_i w_i over non-ignored tokens (w ≡ 1 without ``loss_weights``)
    z_loss = z_loss_coef · mean_i logsumexp_i² over non-ignored tokens (already multiplied)
    n_tokens = number of non-ignored tokens (int64 tensor)
    Logits are fp32 per chunk (no recompute here; the real one recomputes).
    """
    h = hidden.reshape(-1, hidden.shape[-1])
    y = labels.reshape(-1)
    w = loss_weights.reshape(-1).float() if loss_weights is not None else None
    ce_sum = h.new_zeros((), dtype=torch.float32)
    w_sum = h.new_zeros((), dtype=torch.float32)
    z_sum = h.new_zeros((), dtype=torch.float32)
    n = torch.zeros((), dtype=torch.int64, device=h.device)
    for a in range(0, h.shape[0], chunk):
        hs, ys = h[a:a + chunk], y[a:a + chunk]
        logits = (hs.float() @ weight.float().T)
        valid = ys != ignore_index
        lse = torch.logsumexp(logits, -1)
        ce = F.cross_entropy(logits, ys.masked_fill(~valid, 0), reduction="none")
        ws = w[a:a + chunk] if w is not None else torch.ones_like(ce)
        ws = ws * valid
        ce_sum = ce_sum + (ce * ws).sum()
        w_sum = w_sum + ws.sum()
        z_sum = z_sum + (lse.square() * valid).sum()
        n = n + valid.sum()
    loss = ce_sum / w_sum.clamp_min(1e-30)
    z = z_loss_coef * z_sum / n.clamp_min(1).float()
    return loss, z, n

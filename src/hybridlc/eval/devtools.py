"""Development stand-ins used by tests while the real artefacts are being built.

* ``dev_tokenizer(out_dir)`` trains a small byte-level BPE with the protocol's
  pre-tokenizer (``Split(\\d, isolated)`` + ``ByteLevel``) and special tokens (ids 0-7)
  on whatever English text is at hand (the torchtitan C4 test asset if present, else
  the wonderwords lists and a few synthetic sentences). It is **not** the study
  tokenizer; it only lets the eval code be tested tokenizer-agnostically.
* ``project_or_dev_tokenizer`` returns ``data/tokenizer/`` when it exists.
* ``english_texts()`` returns plain-English documents for synthetic filler.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
C4_ASSET = REPO_ROOT / "third_party" / "torchtitan" / "tests" / "assets" / "c4_test" / "data.json"
SPECIALS = ["<|endoftext|>", "<|pad|>"] + [f"<|reserved_{i}|>" for i in range(6)]


def english_texts(max_docs: int | None = None) -> list[str]:
    """English documents for dev tokenizers and synthetic filler (C4 sample if available)."""
    docs: list[str] = []
    if C4_ASSET.is_file():
        with open(C4_ASSET) as f:
            for line in f:
                docs.append(json.loads(line)["text"])
                if max_docs and len(docs) >= max_docs:
                    break
    if not docs:
        from hybridlc.eval.words import wonderwords_lists

        raw = wonderwords_lists()
        nouns, adjs = raw["noun"], raw["adjective"]
        for i in range(2000):
            sents = [f"The {adjs[(i * 7 + j) % len(adjs)]} {nouns[(i * 13 + j * 3) % len(nouns)]} is near the "
                     f"{nouns[(i + j * 11) % len(nouns)]}." for j in range(40)]
            docs.append(" ".join(sents))
    return docs


def dev_tokenizer(out_dir: str | Path, vocab_size: int = 16384) -> Any:
    """Train (or load) a dev BPE tokenizer with the protocol's pre-tokenizer and specials."""
    from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    out_dir = Path(out_dir)
    if (out_dir / "tokenizer.json").is_file():
        return PreTrainedTokenizerFast.from_pretrained(str(out_dir))
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(r"\d"), behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True),
    ])
    tok.decoder = decoders.ByteLevel()
    from hybridlc.eval.words import wonderwords_lists

    raw = wonderwords_lists()
    texts = english_texts() + [" ".join(v) for v in raw.values()]
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=SPECIALS,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), min_frequency=2,
                                  show_progress=False)
    tok.train_from_iterator(texts, trainer=trainer)
    hf = PreTrainedTokenizerFast(tokenizer_object=tok, bos_token=SPECIALS[0], eos_token=SPECIALS[0],
                                 pad_token=SPECIALS[1], additional_special_tokens=SPECIALS[2:],
                                 model_max_length=1 << 20, clean_up_tokenization_spaces=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    hf.save_pretrained(str(out_dir))
    return hf


def project_or_dev_tokenizer(dev_dir: str | Path) -> tuple[Any, str]:
    """The study tokenizer from ``data/tokenizer/`` if present, else a dev tokenizer."""
    proj = REPO_ROOT / "data" / "tokenizer"
    if (proj / "tokenizer.json").is_file() and (proj / "tokenizer.sha256").is_file():
        from hybridlc.data.tokenizer import load_tokenizer

        return load_tokenizer(proj), "project"
    return dev_tokenizer(dev_dir), "dev"


DEV_TOK_DIR = REPO_ROOT / "data" / "dev_tokenizer"


def resolve_test_tokenizer() -> tuple[Any, str]:
    """Tokenizer for tests: ``data/tokenizer/`` when trained, else a cached dev BPE in ``data/dev_tokenizer``."""
    return project_or_dev_tokenizer(DEV_TOK_DIR)


class InductionLM:
    """Scoring-contract test double: an n-gram *copy* model ("induction head").

    The prediction after position t is the token that followed the most recent earlier
    occurrence of the ``order``-gram ending at t (else ``fallback``); log-probability
    log(p_hit) on the prediction and the rest spread uniformly. It solves key-value retrieval
    with distinct keys (and R3's latest value), and cannot do multi-hop, so it exercises
    both EM paths of the scorer. Implements ``prefill`` / ``score_continuation`` /
    ``decode_step`` with ``logprobs[b, j] = log p(cont[b, j] | ctx, cont[b, :j])``.
    """

    scoring_path = "test-double:induction"

    def __init__(self, vocab_size: int, order: int = 3, p_hit: float = 0.9, fallback: int = 0):
        import numpy as np

        self.V, self.order, self.fallback = int(vocab_size), int(order), int(fallback)
        self.lp_hit = float(np.log(p_hit))
        self.lp_miss = float(np.log((1 - p_hit) / (vocab_size - 1)))
        import torch

        self.device = torch.device("cpu")

    def _predict(self, seq) -> "Any":
        """pred[t] = predicted token at position t+1."""
        import numpy as np

        x = np.asarray(seq, dtype=np.int64)
        n, k = len(x), self.order
        pred = np.full(n, self.fallback, dtype=np.int64)
        last: dict[tuple, int] = {}
        for t in range(k - 1, n):
            g = tuple(x[t - k + 1:t + 1].tolist())
            if g in last and last[g] + 1 < n:
                pred[t] = x[last[g] + 1]
            last[g] = t
        return pred

    def prefill(self, input_ids):
        import torch

        from hybridlc.eval.adapter import RecomputeCache

        ids = input_ids.cpu()
        logits = torch.full((ids.shape[0], self.V), self.lp_miss)
        for b in range(ids.shape[0]):
            logits[b, int(self._predict(ids[b].numpy())[-1])] = self.lp_hit
        return logits, RecomputeCache(ids.clone())

    def score_continuation(self, cache, cont_ids):
        import numpy as np
        import torch

        T = cache.ids.shape[1]
        cont = cont_ids.cpu().numpy()
        lps, ams = [], []
        for b in range(cont.shape[0]):
            full = np.concatenate([cache.ids[b].numpy(), cont[b]])
            pred = self._predict(full)[T - 1:T - 1 + cont.shape[1]]
            lps.append(np.where(pred == cont[b], self.lp_hit, self.lp_miss))
            ams.append(pred)
        return torch.tensor(np.stack(lps), dtype=torch.float32), torch.tensor(np.stack(ams))

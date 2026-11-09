"""Token-space helpers shared by every generator (protocol §6.2, §9.1).

Items are built by concatenating pre-tokenised segments; text is never round-tripped
through the tokenizer after assembly. ``TokenSpace`` caches the segment encodings and a
few vocabulary tables for one tokenizer. Everything is tokenizer-agnostic: the special
ids, digit ids and the whitespace token are looked up, never hard-coded.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

#: Characters that end a sentence for boundary snapping (a token whose text, stripped of
#: trailing spaces/quotes, ends with one of these, or that contains a newline).
_SENT_END = (".", "!", "?")
_TRAIL = " \t\"')]”’"


class TokenSpace:
    """Cached encodings and vocabulary tables for one HF tokenizer."""

    def __init__(self, tok: Any):
        self.tok = tok
        self.vocab_size = int(len(tok))
        eot = tok.convert_tokens_to_ids("<|endoftext|>")
        if eot is None or (getattr(tok, "unk_token_id", None) is not None and eot == tok.unk_token_id):
            eot = tok.eos_token_id
        if eot is None:
            raise ValueError("tokenizer has no <|endoftext|> / eos token")
        self.eot_id = int(eot)
        self._cache: dict[str, np.ndarray] = {}
        # Digits must be single tokens (our tokenizer isolates digits; GPT-2 also has one
        # token per bare digit). Values are rendered as [space_id] + 4 digit ids.
        self.digit_ids = []
        for d in range(10):
            ids = self.enc(str(d))
            if len(ids) != 1:
                raise ValueError(f"digit {d} is not a single token: {ids}")
            self.digit_ids.append(int(ids[0]))
        self.digit_set = frozenset(self.digit_ids)
        sp = self.enc(" ")
        if len(sp) != 1:
            raise ValueError(f"' ' is not a single token: {sp}")
        self.space_id = int(sp[0])
        self.fingerprint = self._fingerprint()

    # ------------------------------------------------------------------ encoding
    def enc(self, text: str) -> np.ndarray:
        """Encode ``text`` without special tokens (cached). Returns a read-only int32 array."""
        a = self._cache.get(text)
        if a is None:
            ids = self.tok(text, add_special_tokens=False)["input_ids"]
            a = np.asarray(ids, dtype=np.int32)
            a.setflags(write=False)
            self._cache[text] = a
        return a

    def enc_many(self, texts: list[str]) -> list[np.ndarray]:
        """Batch-encode (fast path for word lists); results are cached."""
        todo = [t for t in texts if t not in self._cache]
        if todo:
            out = self.tok(todo, add_special_tokens=False)["input_ids"]
            for t, ids in zip(todo, out):
                a = np.asarray(ids, dtype=np.int32)
                a.setflags(write=False)
                self._cache[t] = a
        return [self._cache[t] for t in texts]

    def value_ids(self, value: str) -> np.ndarray:
        """Token ids of a digit string (one token per digit)."""
        return np.asarray([self.digit_ids[int(c)] for c in value], dtype=np.int32)

    def decode(self, ids: Any) -> str:
        return self.tok.decode([int(i) for i in ids])

    def natural_value_ok(self) -> bool:
        """True iff the tokenizer itself encodes ' 1234' as [space] + 4 digit tokens.

        Holds for the protocol tokenizer (digit isolation); False for GPT-2 (dev only).
        """
        return self.enc(" 1234").tolist() == [self.space_id, *self.value_ids("1234").tolist()]

    # ------------------------------------------------------------------ tables
    @property
    def ends_sentence(self) -> np.ndarray:
        """bool[V]: token i ends a sentence (a needle may be inserted right after it)."""
        return _ends_sentence_table(self)

    def _fingerprint(self) -> str:
        """SHA-256 identifying the tokenizer (sha of tokenizer.json when saved, else of its JSON dump)."""
        p = Path(str(getattr(self.tok, "name_or_path", "")))
        f = p / "tokenizer.sha256"
        if f.is_file():
            return f.read_text().strip()
        try:
            s = self.tok.backend_tokenizer.to_str()
        except Exception:  # noqa: BLE001
            s = repr(sorted(self.tok.get_vocab().items()))
        return hashlib.sha256(s.encode("utf-8")).hexdigest()


_TABLES: dict[int, np.ndarray] = {}


def _ends_sentence_table(ts: TokenSpace) -> np.ndarray:
    key = id(ts)
    t = _TABLES.get(key)
    if t is None:
        texts = ts.tok.batch_decode([[i] for i in range(ts.vocab_size)])
        t = np.zeros(ts.vocab_size, dtype=bool)
        for i, s in enumerate(texts):
            if "\n" in s:
                t[i] = True
                continue
            r = s.rstrip(_TRAIL)
            if r.endswith(_SENT_END) and len(s.strip()) > 0:
                t[i] = True
        t[ts.eot_id] = False
        t.setflags(write=False)
        _TABLES[key] = t
    return t


_SPACES: dict[int, TokenSpace] = {}


def token_space(tok: Any) -> TokenSpace:
    """Cached ``TokenSpace`` per tokenizer object."""
    ts = _SPACES.get(id(tok))
    if ts is None or ts.tok is not tok:
        ts = TokenSpace(tok)
        _SPACES[id(tok)] = ts
    return ts

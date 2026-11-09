"""Word lists A/B, keys and A-freq/A-count vocabularies (protocol §9.1).

* Source: the ``wonderwords`` noun and adjective lists that RULER's ``niah.py`` uses
  (L97-98 @db43f92); verbs are used only as A-freq/A-count background words.
* Filter: words that are a **single token with a leading space** (``" " + word``).
* Split: SHA-256(word, UTF-8) parity of the full digest (its last hex nibble), the same
  parity convention as the held-out document split in ``hybridlc.data.sources``.
  **even -> list A (train / TEF), odd -> list B (evaluation)**.
* Keys are ``"{adj} {noun}"``, i.e. two single tokens ``[" adj", " noun"]``.

[IMPL] A-count needs animal names, which wonderwords does not tag; the fixed list
``ANIMALS`` below is used (also filtered and split by parity). A-freq probe words are the
single-token nouns of the namespace list minus animals.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from hybridlc.eval.tokens import TokenSpace

#: [IMPL] Common animal nouns for A-count (filtered to single tokens and split A/B by parity).
ANIMALS = (
    "cat", "dog", "horse", "cow", "pig", "sheep", "goat", "chicken", "duck", "goose", "rabbit", "mouse",
    "rat", "fox", "wolf", "bear", "lion", "tiger", "deer", "elephant", "monkey", "zebra", "giraffe",
    "camel", "donkey", "mule", "bat", "owl", "eagle", "hawk", "crow", "dove", "pigeon", "parrot",
    "swan", "frog", "toad", "snake", "lizard", "turtle", "fish", "shark", "whale", "dolphin", "seal",
    "crab", "lobster", "spider", "ant", "bee", "wasp", "fly", "moth", "beetle", "worm", "snail",
    "squirrel", "beaver", "otter", "badger", "moose", "bull", "ox", "lamb", "calf", "puppy", "kitten",
    "hen", "rooster", "turkey", "penguin", "ostrich", "panda", "koala", "kangaroo", "leopard",
    "cheetah", "hippo", "rhino", "gorilla", "buffalo", "bison", "salmon", "trout", "oyster", "squid",
    "octopus", "butterfly", "mosquito", "raven", "sparrow", "robin", "falcon", "vulture", "hamster",
    "pony", "lamb", "yak", "llama", "jaguar", "panther", "hyena", "coyote", "skunk", "raccoon",
)


def sha_parity(word: str) -> int:
    """0 (even) or 1 (odd): parity of the last hex nibble of SHA-256(word)."""
    return int(hashlib.sha256(word.encode("utf-8")).hexdigest()[-1], 16) % 2


def namespace_list(namespace: str) -> str:
    """Word list of a namespace: TEF ('train') uses A, evaluation ('test'/'dev') uses B."""
    if namespace == "train":
        return "A"
    if namespace in ("test", "dev"):
        return "B"
    raise ValueError(f"unknown namespace {namespace!r}")


def wonderwords_lists() -> dict[str, list[str]]:
    """Raw wonderwords lists, de-duplicated and sorted (as RULER does with its key set)."""
    from wonderwords import random_word

    get = random_word._get_words_from_text_file  # noqa: SLF001 - the exact call RULER makes
    return {name: sorted(set(get(f"{name}list.txt"))) for name in ("noun", "adjective", "verb")}


@dataclass
class WordList:
    """All words of one list (A or B) for one tokenizer."""

    name: str
    adjectives: list[str]
    nouns: list[str]
    adj_ids: np.ndarray  # int32, single-token ids of " adj"
    noun_ids: np.ndarray
    probe_words: list[str]  # A-freq probes (single-token nouns, not animals)
    probe_ids: np.ndarray
    animal_words: list[str]
    animal_ids: np.ndarray
    background: list[np.ndarray]  # token arrays of " word" (any length 1..4), not animals
    background_words: list[str]
    _inv: dict[int, np.ndarray] = field(default_factory=dict, repr=False)

    @property
    def n_keys(self) -> int:
        return len(self.adj_ids) * len(self.noun_ids)

    def key_ids(self, key_index: int) -> np.ndarray:
        """Two token ids ``[" adj", " noun"]`` of key number ``key_index``."""
        a, n = divmod(int(key_index), len(self.noun_ids))
        return np.asarray([self.adj_ids[a], self.noun_ids[n]], dtype=np.int32)

    def key_text(self, key_index: int) -> str:
        a, n = divmod(int(key_index), len(self.noun_ids))
        return f"{self.adjectives[a]} {self.nouns[n]}"

    def background_excluding(self, banned_ids: set[int]) -> np.ndarray:
        """Indices of background words whose tokens avoid every id in ``banned_ids``."""
        if not self._inv:
            inv: dict[int, list[int]] = {}
            for i, arr in enumerate(self.background):
                for t in set(arr.tolist()):
                    inv.setdefault(t, []).append(i)
            self._inv = {t: np.asarray(v, dtype=np.int64) for t, v in inv.items()}
        bad = np.zeros(len(self.background), dtype=bool)
        for t in banned_ids:
            if t in self._inv:
                bad[self._inv[t]] = True
        return np.flatnonzero(~bad)


_LISTS: dict[tuple[int, str], WordList] = {}


def single_token_words(ts: TokenSpace, words: list[str]) -> tuple[list[str], np.ndarray]:
    """Words whose ``" " + word`` is exactly one token, with those token ids (first occurrence wins)."""
    encs = ts.enc_many([" " + w for w in words])
    keep, ids, seen = [], [], set()
    for w, e in zip(words, encs):
        if len(e) == 1 and int(e[0]) not in seen and int(e[0]) != ts.eot_id:
            keep.append(w)
            ids.append(int(e[0]))
            seen.add(int(e[0]))
    return keep, np.asarray(ids, dtype=np.int32)


def word_list(ts: TokenSpace, name: str) -> WordList:
    """Build (cached) list ``name`` in {'A','B'} for this tokenizer."""
    key = (id(ts), name)
    if key in _LISTS:
        return _LISTS[key]
    want = 0 if name == "A" else 1
    raw = wonderwords_lists()
    pick = lambda ws: [w for w in ws if sha_parity(w) == want]  # noqa: E731
    adjs, adj_ids = single_token_words(ts, pick(raw["adjective"]))
    nouns, noun_ids = single_token_words(ts, pick(raw["noun"]))
    animals_all = sorted(set(ANIMALS))
    animal_words, animal_ids = single_token_words(ts, pick(animals_all))
    animal_set = set(animals_all)
    probe_words = [w for w in nouns if w not in animal_set]
    probe_ids = np.asarray([noun_ids[nouns.index(w)] for w in probe_words], dtype=np.int32)
    # Background words: every list word (noun/adj/verb) except animals, as token arrays of
    # length 1..4 (" word"). Animal *tokens* are banned too so A-count is unambiguous.
    animal_tok = set(animal_ids.tolist())
    bg_words = sorted({w for k in ("noun", "adjective", "verb") for w in pick(raw[k])} - animal_set)
    bg_encs = ts.enc_many([" " + w for w in bg_words])
    background, background_words = [], []
    for w, e in zip(bg_words, bg_encs):
        if 1 <= len(e) <= 4 and ts.eot_id not in e.tolist() and not (set(e.tolist()) & animal_tok):
            background.append(e)
            background_words.append(w)
    wl = WordList(name, adjs, nouns, adj_ids, noun_ids, probe_words, probe_ids, animal_words, animal_ids,
                  background, background_words)
    _LISTS[key] = wl
    return wl


def describe(tok: Any) -> dict[str, Any]:
    """Sizes of every list for a tokenizer (reported in the item MANIFEST)."""
    from hybridlc.eval.tokens import token_space

    ts = token_space(tok)
    out = {}
    for name in ("A", "B"):
        wl = word_list(ts, name)
        out[name] = dict(adjectives=len(wl.adj_ids), nouns=len(wl.noun_ids), keys=wl.n_keys,
                         probes=len(wl.probe_ids), animals=len(wl.animal_ids), background=len(wl.background),
                         background_tokens_x3=int(3 * sum(len(b) + 1 for b in wl.background)))
    return out

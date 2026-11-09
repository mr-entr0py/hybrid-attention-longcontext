"""Cloze templates T1-T4 (protocol §9.1) rendered in token space.

Every template is a tuple of *parts*; a part is either a literal string (encoded on its
own, once, and cached) or a slot:

* ``KEY`` / ``PREV``: the two key tokens ``[" adj", " noun"]`` (keys always carry their
  leading space, so every fact starts with a space-prefixed token, which is what follows a
  sentence end in natural text);
* ``VALUE``: ``[" "] + 4 single-digit tokens``. The whitespace token belongs to the prompt
  (protocol §6.2), so a query prompt is ``render(query) + [" "]`` and its answer is the 4
  digit tokens;
* ``WORD_X`` / ``WORD_Y`` / ``WORD``: single word tokens of the aggregation tasks (A-freq
  answers are the winner's single token ``" X"``; A-count answers are one digit token after
  a whitespace token that belongs to the prompt).

T1 is the protocol's evaluation template ("The code for {key} is {v}."); T4 is the unseen
diagnostic template ("{key}'s code: {v}."). T2-T3 are TEF paraphrases [IMPL]: protocol §9.1
says they are "fixed in src/eval/templates.py", i.e. here. Literal strings in the M-hop
link of T1 are the protocol's ("The code for {k_i} is the code for {k_{i-1}}."); the links
of T2-T4 and all aggregation phrasings other than A-freq T1 are [IMPL].
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hybridlc.eval.tokens import TokenSpace

KEY, PREV, VALUE = "<KEY>", "<PREV>", "<VALUE>"
WORD_X, WORD_Y, WORD = "<WORD_X>", "<WORD_Y>", "<WORD>"
SLOTS = (KEY, PREV, VALUE, WORD_X, WORD_Y, WORD)


@dataclass(frozen=True)
class CodeTemplate:
    """Fact / link / query forms of one key-value template."""

    name: str
    fact: tuple[str, ...]
    link: tuple[str, ...]
    query: tuple[str, ...]  # the prompt up to (excluding) the value's whitespace token


CODE_TEMPLATES: dict[str, CodeTemplate] = {
    "T1": CodeTemplate("T1",
                       fact=(" The code for", KEY, " is", VALUE, "."),
                       link=(" The code for", KEY, " is the code for", PREV, "."),
                       query=(" The code for", KEY, " is")),
    "T2": CodeTemplate("T2",
                       fact=(" Code for the", KEY, ":", VALUE, "."),
                       link=(" Code for the", KEY, ": same as for the", PREV, "."),
                       query=(" Code for the", KEY, ":")),
    "T3": CodeTemplate("T3",
                       fact=(" The", KEY, " goes with the number", VALUE, "."),
                       link=(" The", KEY, " goes with the same number as the", PREV, "."),
                       query=(" The", KEY, " goes with the number")),
    "T4": CodeTemplate("T4",
                       fact=(KEY, "'s code:", VALUE, "."),
                       link=(KEY, "'s code:", PREV, "'s code."),
                       query=(KEY, "'s code:")),
}

#: R0 cloze passkey [IMPL wording, after the classic passkey task].
PASSKEY = CodeTemplate("PK", fact=(" The pass key is", VALUE, ". Remember it."), link=(),
                       query=(" The pass key is",))

#: A-freq question (T1 = protocol wording). Starts with ".\n", which terminates the word list
#: (and, in the TEF's sequential queries, the previous answer).
AFREQ_TEMPLATES: dict[str, tuple[str, ...]] = {
    "T1": (".\nWhich word appears more often,", WORD_X, " or", WORD_Y, "? Answer:"),
    "T2": (".\nWhich of", WORD_X, " and", WORD_Y, " occurs more times in the list? Answer:"),
    "T3": (".\nMore frequent word,", WORD_X, " or", WORD_Y, "? Answer:"),
    "T4": (".\nOf", WORD_X, " and", WORD_Y, ", the word that appears more often is"),
}

#: A-count question [IMPL]; followed by the whitespace token, answer = one digit token.
ACOUNT_TEMPLATES: dict[str, tuple[str, ...]] = {
    "T1": (".\nHow many times does", WORD, " appear in the list? Answer:"),
    "T2": (".\nNumber of times", WORD, " occurs in the list:"),
    "T3": (".\nCount of", WORD, " in the list:"),
    "T4": (".\nThe word", WORD, " appears this many times:"),
}

#: Templates per namespace (§7.3 leakage controls): TEF uses T1-T3, evaluation T1 (+ T4 slice).
TRAIN_TEMPLATES = ("T1", "T2", "T3")
EVAL_TEMPLATES = ("T1", "T4")

#: Separator token text between sequential TEF queries of code templates (". The code for ...").
SEQ_QUERY_LEAD = "."


def render(ts: TokenSpace, parts: tuple[str, ...], **slots: np.ndarray | list[int] | int) -> np.ndarray:
    """Concatenate pre-tokenised parts; slot values are token-id arrays (or single ids)."""
    out: list[np.ndarray] = []
    for p in parts:
        if p in SLOTS:
            name = p.strip("<>").lower()
            v = slots[name]
            out.append(np.atleast_1d(np.asarray(v, dtype=np.int32)))
        else:
            out.append(ts.enc(p))
    return np.concatenate(out).astype(np.int32) if out else np.zeros(0, np.int32)


def value_slot(ts: TokenSpace, value: int) -> np.ndarray:
    """``[" "] + 4 digit tokens`` of a value in 0..9999 (rendered with leading zeros)."""
    return np.concatenate([[ts.space_id], ts.value_ids(f"{int(value):04d}")]).astype(np.int32)


def value_ids(ts: TokenSpace, value: int) -> np.ndarray:
    """The 4 answer tokens of a value."""
    return ts.value_ids(f"{int(value):04d}")


def fact_ids(ts: TokenSpace, tpl: CodeTemplate, key: np.ndarray, value: int) -> np.ndarray:
    return render(ts, tpl.fact, key=key, value=value_slot(ts, value))


def link_ids(ts: TokenSpace, tpl: CodeTemplate, key: np.ndarray, prev: np.ndarray) -> np.ndarray:
    return render(ts, tpl.link, key=key, prev=prev)


def query_prompt_ids(ts: TokenSpace, tpl: CodeTemplate, key: np.ndarray | None = None) -> np.ndarray:
    """Prompt of a code query: ``render(query) + [" "]`` (the whitespace belongs to the prompt)."""
    slots = {} if key is None else {"key": key}
    return np.concatenate([render(ts, tpl.query, **slots), [ts.space_id]]).astype(np.int32)


def fact_len(ts: TokenSpace, tpl: CodeTemplate) -> int:
    """Token length of one fact (keys are always 2 tokens, values 1 + 4)."""
    return len(render(ts, tpl.fact, key=np.zeros(2, np.int32), value=np.zeros(5, np.int32)))


def link_len(ts: TokenSpace, tpl: CodeTemplate) -> int:
    return len(render(ts, tpl.link, key=np.zeros(2, np.int32), prev=np.zeros(2, np.int32)))


def code_query_len(ts: TokenSpace, tpl: CodeTemplate) -> int:
    """Prompt + answer length of a code query."""
    return len(query_prompt_ids(ts, tpl, np.zeros(2, np.int32))) + 4

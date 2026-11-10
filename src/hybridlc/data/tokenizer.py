"""16,384-entry byte-level BPE with single-digit isolation (protocol §6.2).

``build_tokenizer_model`` constructs the untrained ``tokenizers.Tokenizer``;
``train_tokenizer`` trains it from an iterator of texts and wraps it as a
``PreTrainedTokenizerFast``; ``load_tokenizer`` loads the frozen artefact from
``data/tokenizer/``.
"""
from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path

from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, trainers

from hybridlc.data.sources import TOKENIZER_DIR

VOCAB_SIZE = 16384
EOT = "<|endoftext|>"
PAD = "<|pad|>"
RESERVED = [f"<|reserved_{i}|>" for i in range(6)]
SPECIAL_TOKENS = [EOT, PAD, *RESERVED]  # ids 0..7 in this order
EOT_ID = 0
PAD_ID = 1
N_SPECIAL = len(SPECIAL_TOKENS)


def build_tokenizer_model() -> Tokenizer:
    """Untrained BPE: ``Split(\\d, isolated)`` then ``ByteLevel`` (GPT-2 regex)."""
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(r"\d"), behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True),
    ])
    tok.decoder = decoders.ByteLevel()
    return tok


def train_tokenizer(texts: Iterable[str], out_dir: Path = TOKENIZER_DIR, length: int | None = None):
    """Train on ``texts`` and save a ``PreTrainedTokenizerFast`` to ``out_dir``.

    Returns the HF tokenizer. Writes ``tokenizer.sha256`` (SHA-256 of
    ``tokenizer.json``).
    """
    from transformers import PreTrainedTokenizerFast

    tok = build_tokenizer_model()
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        min_frequency=2,
        show_progress=False,
    )
    tok.train_from_iterator(texts, trainer=trainer, length=length)
    for i, s in enumerate(SPECIAL_TOKENS):
        assert tok.token_to_id(s) == i, (s, tok.token_to_id(s))
    assert tok.get_vocab_size() == VOCAB_SIZE, tok.get_vocab_size()
    hf = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        bos_token=EOT, eos_token=EOT, pad_token=PAD,
        additional_special_tokens=RESERVED,
        model_max_length=1 << 20,
        clean_up_tokenization_spaces=False,
    )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hf.save_pretrained(str(out_dir))
    (out_dir / "tokenizer.sha256").write_text(tokenizer_sha256(out_dir) + "\n")
    return hf


def tokenizer_sha256(tok_dir: Path = TOKENIZER_DIR) -> str:
    """SHA-256 of the saved ``tokenizer.json``."""
    return hashlib.sha256((Path(tok_dir) / "tokenizer.json").read_bytes()).hexdigest()


def load_tokenizer(tok_dir: Path = TOKENIZER_DIR, verify: bool = True):
    """Load the frozen ``PreTrainedTokenizerFast``. Special ids: eot 0, pad 1, reserved 2-7."""
    from transformers import PreTrainedTokenizerFast

    tok_dir = Path(tok_dir)
    if verify:
        want = (tok_dir / "tokenizer.sha256").read_text().strip()
        got = tokenizer_sha256(tok_dir)
        if want != got:
            raise RuntimeError(f"tokenizer.json SHA-256 mismatch: {got} != recorded {want}")
    tok = PreTrainedTokenizerFast.from_pretrained(str(tok_dir))
    assert tok.convert_tokens_to_ids(EOT) == EOT_ID and tok.convert_tokens_to_ids(PAD) == PAD_ID
    return tok


def load_raw_tokenizer(tok_dir: Path = TOKENIZER_DIR, plain_specials: bool = True) -> Tokenizer:
    """The underlying ``tokenizers.Tokenizer`` (fast batch encoding).

    With ``plain_specials`` the literal string ``<|endoftext|>`` in a document is
    encoded as ordinary text, so id 0 only ever comes from our own separators.
    """
    tok = Tokenizer.from_file(str(Path(tok_dir) / "tokenizer.json"))
    tok.encode_special_tokens = plain_specials
    return tok

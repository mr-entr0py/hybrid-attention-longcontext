"""Pinned corpus sources and split assignment (protocol §6.1).

Everything that identifies *which bytes* the study trains and evaluates on lives
here, so that download, tokenisation, manifest and loader code share one source
of truth.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

#: Repository root (src/hybridlc/data/sources.py -> parents[3]).
REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
TOKENIZER_DIR = DATA_DIR / "tokenizer"
TOKENS_DIR = DATA_DIR / "tokens"
MANIFEST_PATH = DATA_DIR / "MANIFEST.json"

FWE_REPO = "HuggingFaceFW/fineweb-edu"
FWE_REVISION = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
#: sample-10BT shards 000..006 (000 = held out; 001..006 = training).
FWE_FILES = [f"sample/10BT/{i:03d}_00000.parquet" for i in range(7)]
FWE_HELDOUT_FILE = FWE_FILES[0]
FWE_TRAIN_FILES = FWE_FILES[1:]
FWE_TOKENIZER_FILE = FWE_FILES[1]  # tokenizer training text (§6.2)

PG19_REPO = "emozilla/pg19"
PG19_REVISION = "c021754c8e01c5b1cc83a1f549c1f97fbbb756b8"


@dataclass(frozen=True)
class Split:
    """One tokenised (corpus, split) pair and its role in the protocol."""

    corpus: str  # 'fwe' | 'pg19'
    split: str  # file stem under data/tokens/<corpus>/
    role: str  # human-readable role from §6.1


#: All tokenised splits. Roles per protocol §6.1.
SPLITS: list[Split] = [
    Split("fwe", "train", "training (shards 001-006)"),
    Split("fwe", "test", "held-out shard 000, SHA-256(doc_id) even: test filler, LM-1, LM-2"),
    Split("fwe", "dev", "held-out shard 000, SHA-256(doc_id) odd: dev filler"),
    Split("pg19", "train", "training (train-000NN-of-00023)"),
    Split("pg19", "validation", "dev filler (50 books)"),
    Split("pg19", "test", "test filler and LM-1 (100 books)"),
]


def raw_path(repo: str, filename: str) -> Path:
    """Local path of a downloaded raw file."""
    return RAW_DIR / repo.replace("/", "__") / filename


def doc_hash(doc_id: str) -> str:
    """SHA-256 hex digest of a document identifier (UTF-8)."""
    return hashlib.sha256(doc_id.encode("utf-8")).hexdigest()


def fwe_heldout_split(doc_id: str) -> str:
    """Split of a FineWeb-Edu shard-000 document: parity of SHA-256(doc_id).

    The parity is that of the full 256-bit digest, i.e. of its last hex nibble.
    even -> 'test', odd -> 'dev' (§6.1).
    """
    return "test" if int(doc_hash(doc_id)[-1], 16) % 2 == 0 else "dev"


def pg19_split_of_file(filename: str) -> str:
    """Map a PG19 mirror parquet filename to its split name."""
    stem = Path(filename).name
    for s in ("train", "validation", "test"):
        if stem.startswith(s + "-"):
            return s
    raise ValueError(f"unrecognised PG19 file {filename}")

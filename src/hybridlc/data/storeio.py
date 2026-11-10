"""Write token stores in the ``data/tokens`` layout (used by tests and other modules' fixtures)."""
from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np


def write_token_store(root: Path | str, corpus: str, split: str, docs: Sequence[np.ndarray],
                      add_eot: bool = True) -> Path:
    """Write ``docs`` (token arrays without the leading EOT if ``add_eot``) as a store."""
    d = Path(root) / corpus
    d.mkdir(parents=True, exist_ok=True)
    arrs = [np.concatenate([[0], np.asarray(x)]).astype(np.uint16) if add_eot else np.asarray(x, np.uint16)
            for x in docs]
    lens = np.array([len(a) for a in arrs], dtype=np.int64)
    np.concatenate(arrs).tofile(d / f"{split}.bin")
    np.save(d / f"{split}.idx.npy", np.concatenate([[0], np.cumsum(lens)]).astype(np.int64))
    (d / f"{split}.meta.json").write_text(json.dumps(dict(corpus=corpus, split=split, synthetic=True,
                                                          n_docs=len(arrs), n_tokens=int(lens.sum()))))
    return d / f"{split}.bin"


def synthetic_corpus(root: Path | str, seed: int = 0, n_fwe: int = 6000, n_books: int = 60,
                     fwe_mean: int = 900, book_range: tuple[int, int] = (20_000, 140_000)) -> None:
    """A small random corpus (fwe/train, fwe/test, fwe/dev, pg19/train) with realistic length shapes.

    Token values are drawn in [8, 16384) so id 0 appears only as separators.
    """
    g = np.random.default_rng(seed)

    def docs(n, lens):
        return [g.integers(8, 16384, size=int(m), dtype=np.int64) for m in lens[:n]]

    fl = np.maximum(1, g.lognormal(np.log(fwe_mean) - 0.5, 1.0, size=n_fwe)).astype(int)
    fl[:3] = [70_000, 9000, 1]  # include a very long and a one-token document
    write_token_store(root, "fwe", "train", docs(n_fwe, fl))
    tl = np.maximum(1, g.lognormal(np.log(fwe_mean) - 0.5, 1.0, size=n_fwe // 2)).astype(int)
    write_token_store(root, "fwe", "test", docs(len(tl), tl))
    write_token_store(root, "fwe", "dev", docs(len(tl), tl))
    bl = g.integers(*book_range, size=n_books)
    bl[0] = 32768 * 2 - 1  # a book whose length (with EOT) is exactly 2 x 32K
    bl[1] = 1000  # a book too short for any window
    write_token_store(root, "pg19", "train", docs(n_books, bl))

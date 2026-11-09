"""Deterministic RNG streams (protocol §8.1).

Every random number used for initialisation, data order, TEF item generation,
evaluation item generation, benchmarking and bootstrapping is derived from a
``numpy.random.SeedSequence`` keyed on ``PROJECT_SALT`` and a *spawn key*
``(seed_index, stream_id, *extra)``.  Nothing in the data / eval / init paths may
touch the global ``torch`` / ``numpy`` / ``random`` RNG state.

Usage::

    g = torch_generator(seed_index=0, stream_id=STREAMS["init_shared"], extra=crc32("embed.weight"))
    w.normal_(0.0, 0.02, generator=g)
"""

from __future__ import annotations

import zlib
from contextlib import contextmanager
from typing import Iterator

import numpy as np
import torch

#: Global entropy for all project streams (protocol §8.1).
PROJECT_SALT: int = 271828182

#: Stream ids used as the second element of the spawn key.
STREAMS: dict[str, int] = dict(init_shared=0, init_mixer=1, data_A=2, data_B=3, tef=4)

#: Constant-entropy streams (they are *not* keyed on a seed index).
EVAL_TEST_ENTROPY: int = 20261109
EVAL_DEV_ENTROPY: int = 77
BENCH_ENTROPY: int = 0
BOOTSTRAP_ENTROPY: int = 2026

#: Mixer type ids for the init_mixer stream spawn key (s, 1, layer_idx, mixer_type_id).
#: Keys are the pattern characters of ``ModelConfig.pattern``.
MIXER_TYPE_ID: dict[str, int] = {"A": 1, "C": 2, "-": 3, "E": 4, "G": 5, "N": 6}

_U63 = (1 << 63) - 1


def crc32(s: str) -> int:
    """Unsigned CRC-32 of a UTF-8 string (used to key per-module init streams)."""
    return zlib.crc32(s.encode("utf-8")) & 0xFFFFFFFF


def seed_sequence(seed_index: int, stream_id: int, *extra: int) -> np.random.SeedSequence:
    """SeedSequence(entropy=PROJECT_SALT, spawn_key=(seed_index, stream_id, *extra))."""
    key = tuple(int(v) for v in (seed_index, stream_id, *extra))
    if any(v < 0 for v in key):
        raise ValueError(f"spawn key entries must be non-negative, got {key}")
    return np.random.SeedSequence(entropy=PROJECT_SALT, spawn_key=key)


def entropy_sequence(entropy: int, *key: int) -> np.random.SeedSequence:
    """SeedSequence for the constant-entropy streams (eval_test, eval_dev, bench, bootstrap)."""
    return np.random.SeedSequence(entropy=int(entropy), spawn_key=tuple(int(v) for v in key))


def _seed_from_ss(ss: np.random.SeedSequence) -> int:
    # 63-bit seed so that torch.Generator.manual_seed never sees a value it reinterprets.
    return int(ss.generate_state(1, dtype=np.uint64)[0]) & _U63


def torch_seed(seed_index: int, stream_id: int, *extra: int) -> int:
    """A 63-bit integer seed derived from the stream (for APIs that only accept an int)."""
    return _seed_from_ss(seed_sequence(seed_index, stream_id, *extra))


def torch_generator(seed_index: int, stream_id: int, *extra: int, device: str | torch.device = "cpu") -> torch.Generator:
    """A ``torch.Generator`` seeded from the stream (default on CPU, so draws are device-independent)."""
    g = torch.Generator(device=device)
    g.manual_seed(torch_seed(seed_index, stream_id, *extra))
    return g


def numpy_rng(seed_index: int, stream_id: int, *extra: int) -> np.random.Generator:
    """A numpy ``Generator`` (PCG64) on the stream."""
    return np.random.default_rng(seed_sequence(seed_index, stream_id, *extra))


@contextmanager
def scoped_torch_seed(seed: int) -> Iterator[None]:
    """Temporarily seed the *global* CPU torch RNG, restoring its previous state on exit.

    Needed only for third-party constructors (fla layers) that draw from the global RNG
    internally; it lets us keep their own init distribution while (a) deriving the
    randomness from a project stream and (b) leaving the caller's global RNG untouched.
    """
    state = torch.random.get_rng_state()
    try:
        torch.random.manual_seed(seed)
        yield
    finally:
        torch.random.set_rng_state(state)

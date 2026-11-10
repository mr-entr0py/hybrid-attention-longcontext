"""RNG helpers for the data path (protocol §8.1).

All randomness is derived from ``numpy.random.SeedSequence(entropy=PROJECT_SALT,
spawn_key=(seed_index, stream_id, *extra))``. The canonical implementation lives
in ``hybridlc.utils.seeds`` (owned by another module); if it is not importable
yet we fall back to an implementation of the *same contract* (INTERFACES.md), so
the derived streams are identical either way. ``tests/test_vt11_data.py``
asserts the equality whenever ``hybridlc.utils.seeds`` exists.
"""
from __future__ import annotations

import zlib

import numpy as np

try:  # pragma: no cover - exercised once utils lands
    from hybridlc.utils.seeds import PROJECT_SALT, STREAMS, seed_sequence  # type: ignore
    HAVE_UTILS_SEEDS = True
except Exception:  # noqa: BLE001
    HAVE_UTILS_SEEDS = False
    PROJECT_SALT = 271828182
    STREAMS = dict(init_shared=0, init_mixer=1, data_A=2, data_B=3, tef=4)

    def seed_sequence(seed_index: int, stream_id: int, *extra: int) -> np.random.SeedSequence:
        """Contract copy of ``hybridlc.utils.seeds.seed_sequence``."""
        return np.random.SeedSequence(entropy=PROJECT_SALT,
                                      spawn_key=(int(seed_index), int(stream_id), *map(int, extra)))

#: Fixed entropy of the frozen test-namespace evaluation streams (§8.1).
EVAL_TEST_ENTROPY = 20261109


def fallback_seed_sequence(seed_index: int, stream_id: int, *extra: int) -> np.random.SeedSequence:
    """The contract implementation, always available (used by VT-11 to cross-check)."""
    return np.random.SeedSequence(entropy=PROJECT_SALT,
                                  spawn_key=(int(seed_index), int(stream_id), *map(int, extra)))


def generator(seed_index: int, stream_id: int, *extra: int) -> np.random.Generator:
    """A private numpy Generator (PCG64) for one data sub-stream."""
    return np.random.Generator(np.random.PCG64(seed_sequence(seed_index, stream_id, *extra)))


def eval_test_generator(*key: str | int) -> np.random.Generator:
    """Generator in the frozen eval_test namespace, keyed by strings/ints (crc32 of strings)."""
    k = [zlib.crc32(x.encode()) if isinstance(x, str) else int(x) for x in key]
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy=EVAL_TEST_ENTROPY, spawn_key=tuple(k))))

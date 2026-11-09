"""Seeded RNG streams for evaluation items and the TEF (protocol §8.1).

* ``test`` namespace: ``SeedSequence(entropy=20261109, spawn_key=(task, level, T))``.
* ``dev`` namespace: ``SeedSequence(entropy=77, spawn_key=(task, level, T))``.
* ``train`` namespace (TEF): ``seed_sequence(seed_index, STREAMS['tef'], ...)`` with the
  project salt, so the stream is shared across k at a given seed index.

Strings in keys are mapped with ``zlib.crc32`` (stable across processes, unlike ``hash``).
The global numpy / python RNGs are never touched.
"""
from __future__ import annotations

import zlib

import numpy as np

try:  # canonical implementation (utils is owned by another module)
    from hybridlc.utils.seeds import PROJECT_SALT, STREAMS  # type: ignore
except Exception:  # noqa: BLE001 - contract copy of INTERFACES.md until utils lands
    PROJECT_SALT = 271828182
    STREAMS = dict(init_shared=0, init_mixer=1, data_A=2, data_B=3, tef=4)

try:
    from hybridlc.utils.seeds import EVAL_TEST_ENTROPY, EVAL_DEV_ENTROPY  # type: ignore
except Exception:  # noqa: BLE001
    EVAL_TEST_ENTROPY = 20261109
    EVAL_DEV_ENTROPY = 77

NAMESPACE_ENTROPY = {"test": EVAL_TEST_ENTROPY, "dev": EVAL_DEV_ENTROPY}


def key_int(x: str | int) -> int:
    """Stable integer for a spawn-key component (crc32 of strings)."""
    if isinstance(x, (int, np.integer)):
        return int(x)
    return zlib.crc32(str(x).encode("utf-8"))


def eval_seed_sequence(namespace: str, *key: str | int) -> np.random.SeedSequence:
    """SeedSequence of an evaluation cell, keyed by (task, level, T[, variant])."""
    if namespace not in NAMESPACE_ENTROPY:
        raise ValueError(f"eval namespace must be test|dev, got {namespace!r}")
    return np.random.SeedSequence(entropy=NAMESPACE_ENTROPY[namespace], spawn_key=tuple(key_int(k) for k in key))


def eval_rng(namespace: str, *key: str | int) -> np.random.Generator:
    """PCG64 generator of an evaluation cell."""
    return np.random.Generator(np.random.PCG64(eval_seed_sequence(namespace, *key)))


def tef_seed_sequence(seed_index: int, *extra: str | int) -> np.random.SeedSequence:
    """SeedSequence of the TEF stream (stream id 4) for one seed index."""
    return np.random.SeedSequence(entropy=PROJECT_SALT,
                                  spawn_key=(int(seed_index), int(STREAMS["tef"]), *(key_int(e) for e in extra)))


def tef_rng(seed_index: int, *extra: str | int) -> np.random.Generator:
    """PCG64 generator in the TEF stream."""
    return np.random.Generator(np.random.PCG64(tef_seed_sequence(seed_index, *extra)))


def child_rng(rng: np.random.Generator, *index: str | int) -> np.random.Generator:
    """Independent child generator ``index`` (one or more str/int keys) of ``rng``'s seed sequence.

    Unlike ``rng.spawn`` this does not depend on how many children were spawned
    before, so item ``i`` of a cell is identical whatever ``n`` is (prefix stability:
    the first 100 items of a 200-item cell equal the 100-item cell).
    """
    ss = rng.bit_generator.seed_seq
    if not isinstance(ss, np.random.SeedSequence):
        raise TypeError("child_rng needs a Generator built from a SeedSequence")
    child = np.random.SeedSequence(entropy=ss.entropy,
                                   spawn_key=tuple(ss.spawn_key) + tuple(key_int(i) for i in index))
    return np.random.Generator(np.random.PCG64(child))

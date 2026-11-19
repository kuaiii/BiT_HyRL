"""Deterministic seed derivation for every stochastic decision.

A stream is addressed by ``(dataset, attack, seed, tag)``, so shards can run in
any order and still reproduce bit-for-bit.
"""

from __future__ import annotations

import hashlib

import numpy as np

__all__ = [
    "RngError",
    "SEED_TAGS",
    "derive_seed_sequence",
    "derive",
    "derive_int",
    "derive_permutation",
]

MAX_SEED = 2**32 - 1

SEED_TAGS = (
    "topology/repair",
    "attack",
    "policy",
    "ppo",
    "reward",
    "bootstrap",
    "torch",
)


class RngError(ValueError):
    """Raised when a seed is outside the domain accepted by ``SeedSequence``."""


def _check_seed(seed: int) -> int:
    """Validate ``seed`` and return it as a plain ``int``."""
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise RngError(f"seed must be an integer, got {type(seed).__name__}")
    value = int(seed)
    if value < 0 or value > MAX_SEED:
        raise RngError(f"seed must lie in [0, {MAX_SEED}], got {value}")
    return value


def _tag_digest(*path: object) -> int:
    """Hash an ordered tuple of tags into a 64-bit integer."""
    key = "|".join(str(item) for item in path)
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


def derive_seed_sequence(seed: int, *path: object) -> np.random.SeedSequence:
    """Build the :class:`numpy.random.SeedSequence` for ``(seed, tag path)``."""
    root = _check_seed(seed)
    return np.random.SeedSequence([root, _tag_digest(*path)])


def derive(seed: int, *path: object) -> np.random.Generator:
    """Return the named random stream ``(seed, *path)``."""
    return np.random.default_rng(derive_seed_sequence(seed, *path))


def derive_int(seed: int, *path: object, low: int = 0, high: int = 2**31 - 1) -> int:
    """Derive a single integer, e.g. for a framework-level seed."""
    if low >= high:
        raise RngError(f"derive_int requires low < high, got low={low} high={high}")
    generator = derive(seed, *path, "int")
    return int(generator.integers(int(low), int(high)))


def derive_permutation(seed: int, n: int, *path: object) -> np.ndarray:
    """Deterministic permutation of ``range(n)`` for the given tag path."""
    generator = derive(seed, *path, "perm")
    return generator.permutation(int(n))

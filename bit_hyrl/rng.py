"""The single random-source entry point of :mod:`bit_hyrl`.

Every stochastic decision in the package must obtain its stream from
:func:`derive` (or :func:`derive_int` for a scalar seed).  This is the code-side
answer to the reproducibility requirement.

Design contract
---------------
Given ``(seed, *tags)`` the derivation is a pure function:

``key = "|".join(str(t) for t in tags)`` → ``h = int(sha256(key)[:8], LE)`` →
``SeedSequence([seed, h])`` → ``default_rng(ss)``.

Consequences (each one is covered by ``tests/test_rng_determinism.py``):

1. Two calls with the same ``(seed, tags)`` yield bit-identical streams.
2. Different ``tags`` yield statistically independent streams, and reordering
   the tags changes the stream.
3. Distinct ``tags`` never share a stream, so adding a new consumer to a
   pipeline cannot shift an existing consumer's draws.

Hard constraint
---------------
The package must never touch process-global random state (the classic
``seed(value)`` functions of :mod:`random` / :mod:`numpy.random`), and must
never draw from the module-level implicit generator.  Doing so would make
results depend on call order somewhere else in the process.  This is enforced
statically by ``tests/test_no_global_seed.py``, which greps the whole
package for those call sites.

Only the *constructors* ``numpy.random.SeedSequence`` and
``numpy.random.default_rng`` are allowed — they create fresh, independent state
instead of mutating shared state.
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

#: Maximum seed accepted by :class:`numpy.random.SeedSequence` (2**32 - 1 is the
#: documented upper bound for the integer form).
MAX_SEED = 2**32 - 1

#: Canonical tag paths used across the package.  Keeping them in one tuple makes
#: accidental tag drift (and therefore silent stream drift) a reviewable diff.
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
    """Validate ``seed`` and return it as a plain ``int``.

    Args:
        seed: Root seed supplied by the CLI or a config file.

    Returns:
        The seed coerced to ``int``.

    Raises:
        RngError: If ``seed`` is not an integer in ``[0, 2**32 - 1]``.
    """
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise RngError(f"seed 必须是整数，收到 {type(seed).__name__}")
    value = int(seed)
    if value < 0 or value > MAX_SEED:
        raise RngError(f"seed 必须落在 [0, {MAX_SEED}]，收到 {value}")
    return value


def _tag_digest(*path: object) -> int:
    """Hash an ordered tuple of tags into a 64-bit integer.

    Args:
        *path: Tag components (``str`` or ``int``); order matters.

    Returns:
        Unsigned 64-bit digest of the joined key.
    """
    key = "|".join(str(item) for item in path)
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


def derive_seed_sequence(seed: int, *path: object) -> np.random.SeedSequence:
    """Build the :class:`numpy.random.SeedSequence` for ``(seed, tag path)``.

    Args:
        seed: Root seed.
        *path: Tag components identifying the consumer.

    Returns:
        A fresh ``SeedSequence`` whose entropy is ``[seed, sha256(tags)]``.
    """
    root = _check_seed(seed)
    return np.random.SeedSequence([root, _tag_digest(*path)])


def derive(seed: int, *path: object) -> np.random.Generator:
    """Return the named random stream ``(seed, *path)``.

    Args:
        seed: Root seed.
        *path: Tag components identifying the consumer, e.g.
            ``derive(seed, "topology", "BA-100", "repair")``.

    Returns:
        A ``numpy.random.Generator`` unique to this tag path.
    """
    return np.random.default_rng(derive_seed_sequence(seed, *path))


def derive_int(seed: int, *path: object, low: int = 0, high: int = 2**31 - 1) -> int:
    """Derive a single integer, e.g. for a framework-level seed.

    The ``"int"`` tag is appended to the path so that drawing a scalar never
    consumes from — and never perturbs — the stream that :func:`derive` returns
    for the same path.

    Args:
        seed: Root seed.
        *path: Tag components identifying the consumer.
        low: Inclusive lower bound.
        high: Exclusive upper bound.

    Returns:
        An integer in ``[low, high)``.

    Raises:
        RngError: If ``low >= high``.
    """
    if low >= high:
        raise RngError(f"derive_int 需要 low < high，收到 low={low} high={high}")
    generator = derive(seed, *path, "int")
    return int(generator.integers(int(low), int(high)))


def derive_permutation(seed: int, n: int, *path: object) -> np.ndarray:
    """Deterministic permutation of ``range(n)`` for the given tag path.

    Args:
        seed: Root seed.
        n: Length of the permutation.
        *path: Tag components identifying the consumer.

    Returns:
        An ``int64`` array of shape ``(n,)`` holding ``0..n-1`` exactly once.
    """
    generator = derive(seed, *path, "perm")
    return generator.permutation(int(n))

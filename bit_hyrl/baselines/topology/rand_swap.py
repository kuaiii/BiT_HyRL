"""RAND-SWAP baseline: the degree-preserving null model.

Accepts every legal 2-switch, so the output keeps ``N``, ``M`` and the degree
sequence but randomises the correlations.
"""

from __future__ import annotations

from .._common import finalize_construct, seeded_legacy
from ._rewire import propose_swap

__all__ = ["NAME", "construct"]

NAME = "RAND-SWAP"


def construct(G, *, seed: int = 0, swaps: int = 400, **params):
    """Random degree-preserving rewiring of ``G`` (same ``N``/``M``/degree sequence)."""
    import random

    rng = random.Random(int(seed) ^ 0x2A11)
    with seeded_legacy(seed):
        current = G.copy()
        edges = list(current.edges())
        accepted = 0
        for _ in range(swaps):
            candidate, candidate_edges = propose_swap(current, edges, rng=rng)
            if candidate is None:
                continue
            current, edges = candidate, candidate_edges
            accepted += 1

    out = finalize_construct(current, NAME)
    out.graph["search_evals"] = 0
    out.graph["search_accepted"] = accepted
    return out

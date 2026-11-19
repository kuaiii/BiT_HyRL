"""ASSORT-SWAP baseline: assortativity-driven rewiring.

Reference: Xulvi-Brunet & Sokolov, Phys. Rev. E 70:066102 (2004).
"""

from __future__ import annotations

from .._common import finalize_construct, seeded_legacy
from ._rewire import degree_product_sum, propose_swap

__all__ = ["NAME", "construct"]

NAME = "ASSORT-SWAP"


def construct(G, *, seed: int = 0, swaps: int = 400, p: float = 1.0, **params):
    """Assortative degree-preserving rewiring of ``G`` (same ``N``/``M``)."""
    import random

    rng = random.Random(int(seed) ^ 0xA550)
    with seeded_legacy(seed):
        current = G.copy()
        edges = list(current.edges())
        accepted = 0
        current_score = degree_product_sum(current)
        for _ in range(swaps):
            candidate, candidate_edges = propose_swap(current, edges, rng=rng)
            if candidate is None:
                continue
            score = degree_product_sum(candidate)
            if score > current_score or rng.random() > p:
                current, edges, current_score = candidate, candidate_edges, score
                accepted += 1

    out = finalize_construct(current, NAME)
    out.graph["search_evals"] = 0
    out.graph["search_accepted"] = accepted
    return out

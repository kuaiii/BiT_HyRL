"""ASSORT-SWAP topology baseline -- assortativity-driven rewiring (Xulvi-Brunet).

Reference: Xulvi-Brunet & Sokolov, *Reshuffling scale-free networks: From random to
assortative networks*, Phys. Rev. E 70:066102 (2004).

The move set is the same 2-switch as every other rewiring baseline; the acceptance
rule is purely **local and objective-free**: accept a swap when it increases the
degree-product sum ``sum_{(u,v)} k_u k_v``, i.e. when it links high-degree nodes to
high-degree nodes (and low to low).  With probability ``1 - p`` a random swap is
accepted instead, which reproduces the paper's mixing parameter.

Why it belongs in the baseline table: BiT's design claim is that a **bimodal**
degree sequence plus hub-hub attachment buys resilience.  Assortative rewiring is the
cheapest way to obtain the "hub-hub" half of that claim *without* changing the degree
sequence, so it isolates how much of BiT's gain is assortativity alone versus the
degree-sequence shaping.  It costs ``O(1)`` per move (no robustness evaluation), which
is why it needs no budget parameter -- and why it is the fair "free lunch" control.
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

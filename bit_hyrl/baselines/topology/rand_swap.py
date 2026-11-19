"""RAND-SWAP topology baseline -- the degree-preserving null model.

Accepts every legal 2-switch, never looks at robustness.  The result has exactly the
input's ``N``, ``M`` and **degree sequence**, but randomised correlations, so it is
the reference against which any claim of the form "our rewiring helps" must be read:

* ``R(RAND-SWAP) ~ R(input)``  -> the metric is blind to correlations and the whole
  comparison is carried by the degree sequence;
* ``R(RAND-SWAP) > R(input)``  -> part of the gain reported by *any* method is just
  the destruction of the original (often disassortative / geometric) correlations;
* ``R(method) > R(RAND-SWAP)`` -> the method beats a same-budget randomiser, which is
  the weakest defensible statement for a topology paper.

It is also the only baseline that is exactly budget-neutral in ``M``: the 2-switch
conserves the edge count, so no reviewer can attribute a gain to extra links.
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

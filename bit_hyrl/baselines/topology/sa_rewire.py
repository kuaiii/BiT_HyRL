"""SA-R baseline: simulated annealing on robustness with degree-preserving 2-swaps.

Shares the move set and the scoring function with :mod:`r_greedy`; only the
acceptance rule differs.
"""

from __future__ import annotations

import math

from .._common import finalize_construct, seeded_legacy
from ._rewire import FAST_R_STEPS, propose_swap, r_paper

__all__ = ["NAME", "construct"]

NAME = "SA-R"


def construct(
    G,
    *,
    seed: int = 0,
    proposals: int = 800,
    steps: int = FAST_R_STEPS,
    t0_frac: float = 0.02,
    t_end_frac: float = 0.0002,
    **params,
):
    """Annealed degree-preserving rewiring of ``G`` (same ``N``/``M``)."""
    import random

    rng = random.Random(int(seed) ^ 0x5A17)
    with seeded_legacy(seed):
        current = G.copy()
        current_r = r_paper(current, steps)
        edges = list(current.edges())
        best, best_r = current, current_r
        t0 = max(t0_frac * current_r, 1e-6)
        t_end = max(t_end_frac * current_r, 1e-9)
        ratio = (t_end / t0) ** (1.0 / max(proposals - 1, 1))
        temperature = t0
        evals = 1
        accepted = 0
        for _ in range(proposals):
            candidate, candidate_edges = propose_swap(current, edges, rng=rng)
            if candidate is None:
                temperature *= ratio
                continue
            candidate_r = r_paper(candidate, steps)
            evals += 1
            delta = candidate_r - current_r
            if delta >= 0 or rng.random() < math.exp(delta / temperature):
                current, current_r, edges = candidate, candidate_r, candidate_edges
                accepted += 1
                if current_r > best_r + 1e-12:
                    best, best_r = current, current_r
            temperature *= ratio

    out = finalize_construct(best, NAME)
    out.graph["search_evals"] = evals
    out.graph["search_accepted"] = accepted
    return out

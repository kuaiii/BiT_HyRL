"""ROMEN baseline: reinforcement-learning search over link additions.

Positions come from node ``pos`` attributes or a locally seeded generator.  The
optimiser draws from the ambient torch RNG, so runs are not bit-reproducible
across processes.
"""

from __future__ import annotations

from .._common import finalize_construct, positions_or_seeded, seeded_legacy
from ._romen_impl import optimize_network_robustness

__all__ = ["NAME", "construct"]

NAME = "ROMEN"


def construct(G, *, seed: int = 0, population_size: int = 3, generations: int = 50,
              **params):
    """Rebuild ``G`` with ROMEN (same ``N``, connected)."""
    with seeded_legacy(seed):
        positions = positions_or_seeded(G, seed, scale=1000.0)
        result = optimize_network_robustness(
            initial_graph=G,
            positions=positions,
            population_size=population_size,
            generations=generations,
            verbose=False,
        )
    graph = result.get("optimized_graph", G) if isinstance(result, dict) else result
    return finalize_construct(graph if graph is not None else G, NAME)

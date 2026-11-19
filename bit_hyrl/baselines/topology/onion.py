"""ONION baseline: degree-preserving edge swaps driven by the onion decomposition."""

from __future__ import annotations

from .._common import finalize_construct, seeded_legacy
from ._onion_impl import OnionNetworkOptimizer

__all__ = ["NAME", "construct"]

NAME = "ONION"


def construct(G, *, seed: int = 0, max_iterations: int = 2000, **params):
    """Rebuild ``G`` with ONION (same ``N``, connected).  ``seed`` is accepted for a
    uniform baseline interface; ONION itself is deterministic given the graph."""
    with seeded_legacy(seed):
        optimizer = OnionNetworkOptimizer(G)
        try:
            graph = optimizer.optimize_network(max_iterations=max_iterations, verbose=False)
        except TypeError:
            graph = optimizer.optimize_network(max_iterations=max_iterations)
    return finalize_construct(graph if graph is not None else G, NAME)

"""ONION topology baseline — ported from ``network_construction/ONION/onion.py``.

Logic copied in (``_onion_impl``); no import of the old tree.  The original
module-level ``matplotlib.use('TkAgg')`` / rcParams are stripped in the impl.
"""

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
        except TypeError:  # some builds do not accept ``verbose``
            graph = optimizer.optimize_network(max_iterations=max_iterations)
    return finalize_construct(graph if graph is not None else G, NAME)

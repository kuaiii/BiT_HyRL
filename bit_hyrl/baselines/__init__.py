"""Baselines: five topology reconstruction methods and two placement proxies.

Topology baselines share ``construct(G, *, seed=0, **params) -> nx.Graph`` (same N,
connected); placement proxies share ``select(graph, budget, seed=0) -> list[int]``.
The two placement entries carry the ``-sim`` suffix because they approximate the
original methods rather than reproducing them.
"""

from __future__ import annotations

from .placement.registry import PLACEMENT_REGISTRY
from .topology.registry import CONSTRUCT_REGISTRY

__all__ = ["CONSTRUCT_REGISTRY", "PLACEMENT_REGISTRY"]

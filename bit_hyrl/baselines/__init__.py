"""Ported baselines: 5 topology reconstruction methods + 2 placement proxies.

Topology baselines share ``construct(G, *, seed=0, **params) -> nx.Graph`` (same N,
connected); placement proxies share ``select(graph, budget, seed=0) -> list[int]``.
The two placement proxies are honestly marked ``-sim`` (they are proxies, not the
original methods).  The logic is copied from the old ``network_construction`` /
``scripts`` tree — the package never imports it.
"""

from __future__ import annotations

from .placement.registry import PLACEMENT_REGISTRY
from .topology.registry import CONSTRUCT_REGISTRY

__all__ = ["CONSTRUCT_REGISTRY", "PLACEMENT_REGISTRY"]

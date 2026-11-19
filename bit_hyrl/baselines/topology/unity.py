"""UNITY topology baseline — ported from ``network_construction/UNITY/UNITY.py``.

Logic copied in (``_unity_impl``); no import of the old tree.  Uses
``UNITY.generate_matched_topology`` which aligns the reconstructed graph to the
input's ``N``/``M`` (with a wider comm-range fallback when edges are short).
"""

from __future__ import annotations

from .._common import finalize_construct, seeded_legacy
from ._unity_impl import UNITY

__all__ = ["NAME", "construct"]

NAME = "UNITY"


def construct(G, *, seed: int = 0, area_diameter: float = 500,
              comm_range: float = 200, **params):
    """Rebuild ``G`` with UNITY (N/M aligned, connected)."""
    with seeded_legacy(seed):
        unity = UNITY(area_diameter=area_diameter, comm_range=comm_range)
        graph = unity.generate_matched_topology(G)
        if graph.number_of_edges() < G.number_of_edges():
            graph = UNITY(area_diameter=area_diameter, comm_range=1000).generate_matched_topology(G)
    return finalize_construct(graph, NAME)

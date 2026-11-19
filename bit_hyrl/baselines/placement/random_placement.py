"""Random placement — the absolute lower bound for any structured placement.

Places the ``budget`` controllers uniformly at random among the nodes.  It exists to
answer one question the paper otherwise cannot: *is any structure at all better than
placing blindly?*  A method that fails to beat this has no placement contribution.

Seeded, so the same ``(graph, budget, seed)`` reproduces the same set.
"""

from __future__ import annotations

import networkx as nx

__all__ = ["NAME", "select"]

NAME = "Random-placement"


def select(graph: nx.Graph, budget: int, seed: int = 0) -> list[int]:
    """Return ``budget`` distinct nodes drawn uniformly at random."""
    nodes = sorted(graph.nodes())
    if budget >= len(nodes):
        return nodes
    generator = nx.utils.create_random_state(seed)
    chosen = generator.choice(len(nodes), size=budget, replace=False)
    return [nodes[int(index)] for index in chosen]

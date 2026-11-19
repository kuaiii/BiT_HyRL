"""QDLM baseline: quantum-inspired learning model over a geometric candidate edge set.

Node positions are required; they are normalised to ``0..n-1`` with ``pos`` in
``[0, 1]`` before the search.
"""

from __future__ import annotations

import networkx as nx

from .._common import finalize_construct, normalized_env_graph, seeded_legacy
from ._qdlm_impl import QuantumEnvironment, QuantumLearningModel

__all__ = ["NAME", "construct"]

NAME = "QDLM"


def construct(G, *, seed: int = 0, pop_size: int = 20, max_iterations: int = 30,
              comm_range: float = 0.4, **params):
    """Rebuild ``G`` with QDLM (same ``N``, connected)."""
    with seeded_legacy(seed):
        g_norm, _norm_pos = normalized_env_graph(G, seed)
        env = QuantumEnvironment(G=g_norm, comm_range=comm_range)
        if getattr(env, "m", 0) == 0:
            return finalize_construct(g_norm, NAME)
        best_adj = QuantumLearningModel(
            env, pop_size=pop_size, max_iterations=max_iterations
        ).run(verbose=False)
    return finalize_construct(nx.from_numpy_array(best_adj), NAME)

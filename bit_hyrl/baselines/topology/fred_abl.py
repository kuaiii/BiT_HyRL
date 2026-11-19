"""FRED-ABL topology baseline — ported from ``network_construction/FRED_ABL/fred_abl_optimization.py``.

Logic copied in (``_fred_impl``); no import of the old tree (the unused
``src.utils.logger`` import is dropped in the impl).  Requires node positions;
nodes are normalised to ``0..n-1`` with ``pos`` in ``[0, 1]`` before the Bayesian
search, matching the original adapter.  Depends on scikit-learn + scipy.
"""

from __future__ import annotations

import networkx as nx

from .._common import finalize_construct, normalized_env_graph, seeded_legacy
from ._fred_impl import FRED_ABL, IIoTEnv

__all__ = ["NAME", "construct"]

NAME = "FRED-ABL"


def construct(G, *, seed: int = 0, iterations: int = 15, initial_samples: int = 3,
              comm_range: float = 0.4, **params):
    """Rebuild ``G`` with FRED-ABL (same ``N``, connected)."""
    with seeded_legacy(seed):
        g_norm, _norm_pos = normalized_env_graph(G, seed)
        env = IIoTEnv(G=g_norm, comm_range=comm_range)
        best_adj = FRED_ABL(env).run(
            iterations=iterations, initial_samples=initial_samples, verbose=False
        )
    return finalize_construct(nx.from_numpy_array(best_adj), NAME)

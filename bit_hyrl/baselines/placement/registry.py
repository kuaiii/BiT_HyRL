"""Registry of the ported placement proxies: ``name -> select`` (``-sim`` suffixed)."""

from __future__ import annotations

from . import kmedian, madqn_sim, mwcl_sim, random_placement

#: ``str -> callable(graph, budget, seed=0) -> list[int]``.
PLACEMENT_REGISTRY = {
    "MWCL-sim": mwcl_sim.select,
    "MADQN-sim": madqn_sim.select,
    # lower bound (blind placement) and textbook facility-location baseline
    "Random-placement": random_placement.select,
    "KMedian": kmedian.select,
}

__all__ = ["PLACEMENT_REGISTRY"]

"""R-GREEDY topology baseline -- Schneider-style greedy robustness optimisation.

Reference: Schneider, Moreira, Andrade, Havlin & Herrmann, *Mitigation of malicious
attacks on networks*, PNAS 108(10):3838-3841 (2011).

The published method repeatedly picks, from a pool of random **degree-preserving**
double-edge swaps, the one that improves the robustness ``R`` most, and stops when no
candidate improves.  That is exactly what this module does, so it is the canonical
"search on a fixed budget" opponent for a one-shot construction such as BiT:

* same ``N``, same ``M``, same degree sequence as the input (``M`` is preserved by
  the 2-switch, so no baseline gets to buy robustness with extra links);
* connectivity is a hard constraint (see :func:`_rewire.propose_swap`);
* the fitness is :func:`effrnet._fast_robustness` -- the same cheap ``R`` (GCC/N area
  on a degree-targeted removal grid) the Eff-R-Net baseline already optimises.

``budget`` counts **robustness evaluations**, not swaps: the number is written to
``graph.graph["search_evals"]`` so the paper can state the evaluation budget next to
BiT's *one* construction instead of comparing incomparable "iterations".
"""

from __future__ import annotations

from .._common import finalize_construct, seeded_legacy
from ._rewire import FAST_R_STEPS, propose_swap, r_paper

__all__ = ["NAME", "construct"]

NAME = "R-GREEDY"


def construct(
    G,
    *,
    seed: int = 0,
    budget: int = 800,
    candidates: int = 32,
    steps: int = FAST_R_STEPS,
    **params,
):
    """Optimise ``R`` of ``G`` by greedy degree-preserving swaps (same ``N``/``M``)."""
    evals = 0
    rounds = 0
    with seeded_legacy(seed):
        current = G.copy()
        current_r = r_paper(current, steps)
        evals += 1
        edges = list(current.edges())
        while evals < budget:
            best, best_r, best_edges = None, current_r, None
            for _ in range(candidates):
                if evals >= budget:
                    break
                candidate, candidate_edges = propose_swap(current, edges)
                evals += 1
                if candidate is None:
                    continue
                candidate_r = r_paper(candidate, steps)
                if candidate_r > best_r + 1e-12:
                    best, best_r, best_edges = candidate, candidate_r, candidate_edges
            rounds += 1
            if best is None:  # local optimum under this neighbourhood
                break
            current, current_r, edges = best, best_r, best_edges

    out = finalize_construct(current, NAME)
    out.graph["search_evals"] = evals
    out.graph["search_rounds"] = rounds
    return out

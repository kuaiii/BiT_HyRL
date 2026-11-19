"""Registry of the ported topology baselines: ``name -> construct``.

Two families share one interface ``construct(G, *, seed=0, **params) -> nx.Graph``
(same ``N``, connected):

* **construction** -- ONION / ROMEN / UNITY / FRED-ABL / QDLM / Eff-R-Net rebuild the
  edge set from scratch (possibly changing ``M``);
* **rewiring** -- R-GREEDY / SA-R / ASSORT-SWAP / RAND-SWAP keep ``N``, ``M`` and the
  degree sequence and only change *who is connected to whom* (2-switch moves), so they
  are the family that isolates "search" from "degree-sequence design".
"""

from __future__ import annotations

from . import (
    assort_swap,
    effrnet,
    fred_abl,
    grg,
    grg_proxy,
    onion,
    qdlm,
    rand_swap,
    r_greedy,
    romen,
    sa_rewire,
    unity,
)

#: ``str -> callable(G, *, seed=0, **params) -> nx.Graph`` (same N, connected).
CONSTRUCT_REGISTRY = {
    # --- construction (edge set rebuilt, M may change) ---
    "ONION": onion.construct,
    "ROMEN": romen.construct,
    "UNITY": unity.construct,
    "FRED-ABL": fred_abl.construct,
    "QDLM": qdlm.construct,
    "Eff-R-Net": effrnet.construct,
    "GRG": grg.construct,
    "GRG-proxy": grg_proxy.construct,
    # --- rewiring (N, M, degree sequence preserved) ---
    "R-GREEDY": r_greedy.construct,
    "SA-R": sa_rewire.construct,
    "ASSORT-SWAP": assort_swap.construct,
    "RAND-SWAP": rand_swap.construct,
}

__all__ = ["CONSTRUCT_REGISTRY"]

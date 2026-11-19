"""Registry of the ported topology baselines (``name -> construct``).

All constructors share the signature ``construct(G, *, seed=0, **params)`` and
return a connected graph with the same ``N`` as the input.
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
    "ONION": onion.construct,
    "ROMEN": romen.construct,
    "UNITY": unity.construct,
    "FRED-ABL": fred_abl.construct,
    "QDLM": qdlm.construct,
    "Eff-R-Net": effrnet.construct,
    "GRG": grg.construct,
    "GRG-proxy": grg_proxy.construct,
    "R-GREEDY": r_greedy.construct,
    "SA-R": sa_rewire.construct,
    "ASSORT-SWAP": assort_swap.construct,
    "RAND-SWAP": rand_swap.construct,
}

__all__ = ["CONSTRUCT_REGISTRY"]

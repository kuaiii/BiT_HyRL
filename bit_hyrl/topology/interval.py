"""The admissible interval of the hub factor ``c``, in one place.

Every experiment that deploys the bimodal construction needs the same three numbers:
the smallest ``c`` that keeps the family identifiable, the largest ``c`` allowed by the
simple-graph limit, the skeleton-percolation condition and the hub-share bound, and the
geometric midpoint of the two.  They used to be recomputed inside each script, which is
how the same graph ended up with two different operating points.  This module is the
single definition; :mod:`scripts.family_bench` and every evaluation script import it.

Derivation (Section 3): with ``k_max = round(c kbar)`` and ``kbar = 2M/N``, degree
conservation fixes ``k_min = (2M - k_max)/(N-1)``.  The four conditions are

    (i)   k_max - k_min >= 2 and k_min >= 2      -> c_min, discrete
    (ii)  k_max <= N - 1                         -> c <= (N-1)/kbar
    (iii) k_min >= 2                             -> c <= N - 2(N-1)/kbar
    (iv)  K(K+3)/[2N(N-1)] <= E_0/N              -> c <= [sqrt(9+8(N-1)E_0) - 3]/(2kbar)

where ``E_0`` is the global efficiency of the un-reconstructed graph.
"""

from __future__ import annotations

import numpy as np

__all__ = ["bimodal_interval", "deployed_c", "VARIANT_OF"]


def bimodal_interval(n: int, m: int, e0: float) -> dict[str, float]:
    """Return ``c_min``, the three upper bounds, their minimum and the midpoint.

    Args:
        n: Node count.
        m: Edge count.
        e0: Global efficiency of the un-reconstructed graph, used by the hub-share
            bound only.

    Returns:
        ``{"c_min", "c_struct", "c_perc", "c_dir", "c_up", "c_mid"}``.  ``c_min`` is
        ``None`` when no integer ``c`` satisfies (i), which happens only on graphs too
        sparse to admit two distinct degrees.
    """
    kbar = 2.0 * m / n
    c_min: float | None = None
    c = 1.0
    while c < 400.0:
        k_max = int(np.floor(c * kbar + 0.5))
        k_min = int((2 * m - k_max) // (n - 1))
        if k_max - k_min >= 2 and k_min >= 2:
            c_min = round(c, 3)
            break
        c += 0.01
    c_struct = (n - 1) / kbar
    c_perc = n - 2.0 * (n - 1) / kbar
    c_dir = (np.sqrt(9.0 + 8.0 * (n - 1) * e0) - 3.0) / (2.0 * kbar)
    c_up = float(min(c_struct, c_perc, c_dir))
    return {
        "c_min": c_min,
        "c_struct": float(c_struct),
        "c_perc": float(c_perc),
        "c_dir": float(c_dir),
        "c_up": c_up,
        "c_mid": float(np.sqrt(c_min * c_up)) if c_min else float("nan"),
    }


#: variant name -> which point of the interval it deploys.  ``BiT-HyRL`` is the
#: deployed method (geometric midpoint); the legacy tag ``BiT-A`` is kept as an alias.
VARIANT_OF = {"BiT-R": "c_min", "BiT-HyRL": "c_mid", "BiT-A": "c_mid", "BiT-E": "c_up"}


def deployed_c(n: int, m: int, e0: float, variant: str = "BiT-HyRL") -> float:
    """The ``c`` a named variant uses on a graph with this ``(N, M, E_0)``.

    The default is ``BiT-HyRL`` (the geometric midpoint) since the paper deploys that
    variant as the method; ``BiT-R`` and ``BiT-E`` stay available for the ablation rows.
    """
    interval = bimodal_interval(n, m, e0)
    return round(float(interval[VARIANT_OF[variant]]), 3)

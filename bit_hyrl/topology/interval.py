"""Admissible interval of the hub factor ``c``."""

from __future__ import annotations

import numpy as np

__all__ = ["bimodal_interval", "deployed_c", "VARIANT_OF"]


def bimodal_interval(n: int, m: int, e0: float) -> dict[str, float]:
    """Return ``c_min``, the three upper bounds, their minimum and the midpoint."""
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


VARIANT_OF = {"BiT-R": "c_min", "BiT-HyRL": "c_mid", "BiT-A": "c_mid", "BiT-E": "c_up"}


def deployed_c(n: int, m: int, e0: float, variant: str = "BiT-HyRL") -> float:
    """The ``c`` a named variant uses on a graph with this ``(N, M, E_0)``."""
    interval = bimodal_interval(n, m, e0)
    return round(float(interval[VARIANT_OF[variant]]), 3)

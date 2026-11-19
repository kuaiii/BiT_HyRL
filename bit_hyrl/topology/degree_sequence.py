"""Degree-sequence planning for the bimodal construction.

The construction is driven by ``(n, m, c, h)``; this module turns them into a
validated integer degree sequence and reports why a request is infeasible.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

__all__ = [
    "DegreeSequenceError",
    "DegreePlan",
    "round_half_up",
    "plan_degree_sequence",
    "build_degree_list",
    "validate_plan",
    "negative_degree_count",
]

HUB_DEGREE_FLOOR = 2


class DegreeSequenceError(ValueError):
    """Raised when a degree plan violates its invariants."""


def round_half_up(value: float) -> int:
    """Round half away from zero, i.e. ``floor(x + 0.5)`` for ``x >= 0``."""
    number = float(value)
    if not math.isfinite(number):
        raise DegreeSequenceError(f"round_half_up needs a finite value, got {number!r}")
    return int(math.floor(number + 0.5))


@dataclass(frozen=True)
class DegreePlan:
    """The exact degree budget of one BiT construction."""

    n: int
    m: int
    c: float
    h: int
    k_bar: float
    k_max: int
    k_low: int
    r: int
    degree_sequence: tuple[int, ...]
    negative_degree_count: int
    degenerate_fallback: bool
    sum_degree: int
    parity_touched: bool = False
    capped_at_n_minus_1: bool = False
    degenerate_reason: str | None = None

    def degree_histogram(self) -> dict[int, int]:
        """Return ``{degree: count}`` for the *planned* sequence, sorted by degree."""
        histogram: dict[int, int] = {}
        for degree in self.degree_sequence:
            histogram[degree] = histogram.get(degree, 0) + 1
        return dict(sorted(histogram.items()))

    def hub_nodes(self) -> tuple[int, ...]:
        """Return the node ids carrying ``k_max`` (``0 .. h-1``)."""
        return tuple(range(self.h))

    def mid_nodes(self) -> tuple[int, ...]:
        """Return the ``r`` node ids carrying ``k_low + 1``."""
        return tuple(range(self.h, self.h + self.r))

    def low_nodes(self) -> tuple[int, ...]:
        """Return the ``n - h - r`` node ids carrying ``k_low``."""
        return tuple(range(self.h + self.r, self.n))

    def n_distinct_degrees(self) -> int:
        """Number of distinct degree values in the sequence."""
        return len(set(self.degree_sequence))

    def c_star(self) -> float:
        """Return ``(N - 1) / k_bar``, the ``c`` at which ``k_max`` saturates."""
        if self.k_bar <= 0.0:
            return 0.0
        return (self.n - 1) / self.k_bar

    def c_norm(self) -> float:
        """Return ``c / c_star``; ``1.0`` means "exactly at the ceiling"."""
        star = self.c_star()
        return (self.c / star) if star > 0.0 else 0.0

    def check_conservation(self) -> None:
        """Assert every structural invariant of the plan."""
        validate_plan(self)

    def to_dict(self) -> dict[str, Any]:
        """Plain-``dict`` view, JSON-serialisable (sequence becomes a list)."""
        return {
            "n": self.n,
            "m": self.m,
            "c": self.c,
            "h": self.h,
            "k_bar": self.k_bar,
            "k_max": self.k_max,
            "k_low": self.k_low,
            "r": self.r,
            "degree_sequence": list(self.degree_sequence),
            "negative_degree_count": self.negative_degree_count,
            "degenerate_fallback": self.degenerate_fallback,
            "degenerate_reason": self.degenerate_reason,
            "sum_degree": self.sum_degree,
            "parity_touched": self.parity_touched,
            "capped_at_n_minus_1": self.capped_at_n_minus_1,
            "c_star": self.c_star(),
            "c_norm": self.c_norm(),
            "degree_histogram": {str(k): v for k, v in self.degree_histogram().items()},
            "n_distinct_degrees": self.n_distinct_degrees(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DegreePlan":
        """Rebuild a plan from its :meth:`to_dict` output."""
        sequence = data["degree_sequence"]
        return cls(
            n=int(data["n"]),
            m=int(data["m"]),
            c=float(data["c"]),
            h=int(data["h"]),
            k_bar=float(data["k_bar"]),
            k_max=int(data["k_max"]),
            k_low=int(data["k_low"]),
            r=int(data["r"]),
            degree_sequence=tuple(int(v) for v in sequence),
            negative_degree_count=int(data["negative_degree_count"]),
            degenerate_fallback=bool(data["degenerate_fallback"]),
            sum_degree=int(data["sum_degree"]),
            parity_touched=bool(data.get("parity_touched", False)),
            capped_at_n_minus_1=bool(data.get("capped_at_n_minus_1", False)),
            degenerate_reason=data.get("degenerate_reason"),
        )


def validate_plan(plan: DegreePlan) -> None:
    """Check every invariant of ``plan``; raise :class:`DegreeSequenceError` otherwise."""
    if plan.n < 2:
        raise DegreeSequenceError(f"n must be >= 2, got {plan.n}")
    if plan.m < 1:
        raise DegreeSequenceError(f"m must be >= 1, got {plan.m}")
    if not 1 <= plan.h < plan.n:
        raise DegreeSequenceError(f"h must lie in [1, n), got h={plan.h} n={plan.n}")
    if len(plan.degree_sequence) != plan.n:
        raise DegreeSequenceError(
            f"degree sequence has length {len(plan.degree_sequence)} != n={plan.n}"
        )

    total = sum(plan.degree_sequence)
    if total != plan.sum_degree:
        raise DegreeSequenceError(
            f"sum_degree field {plan.sum_degree} disagrees with the actual sum {total}"
        )
    if total != 2 * plan.m:
        raise DegreeSequenceError(
            f"conservation violated: sum k={total}, 2m={2 * plan.m} (diff {total - 2 * plan.m})"
        )
    if total % 2 != 0:
        raise DegreeSequenceError("the degree sum is always even; this branch means the formula was changed")
    if plan.parity_touched:
        raise DegreeSequenceError("parity_touched must stay False: the parity repair branch is unreachable")
    if plan.k_low < 1:
        raise DegreeSequenceError(
            f"k_low={plan.k_low} < 1: the conservation argument fails, so 2M >= h*k_max + (N-h) does not hold"
        )
    if plan.k_max > plan.n - 1:
        raise DegreeSequenceError(f"k_max={plan.k_max} > n-1={plan.n - 1}, not a simple graph")
    if plan.k_max < HUB_DEGREE_FLOOR:
        raise DegreeSequenceError(f"k_max={plan.k_max} < {HUB_DEGREE_FLOOR}")
    if not 0 <= plan.r <= plan.n - plan.h:
        raise DegreeSequenceError(f"r={plan.r} out of range, expected 0 <= r <= n-h={plan.n - plan.h}")
    if plan.negative_degree_count != plan.n - plan.h - plan.r:
        raise DegreeSequenceError(
            f"negative_degree_count={plan.negative_degree_count} != n-h-r="
            f"{plan.n - plan.h - plan.r}"
        )
    for left, right in zip(plan.degree_sequence, plan.degree_sequence[1:]):
        if left < right:
            raise DegreeSequenceError(
                "degree_sequence must be non-increasing (nodes 0..h-1 are hubs, then r mid nodes)"
            )


def _assemble(n: int, m: int, c: float, h: int, k_bar: float, k_max: int,
              k_low: int, r: int, *, degenerate_fallback: bool,
              degenerate_reason: str | None, capped: bool) -> DegreePlan:
    """Build and validate a :class:`DegreePlan` from solved integers."""
    sequence = [k_max] * h + [k_low + 1] * r + [k_low] * (n - h - r)
    plan = DegreePlan(
        n=n,
        m=m,
        c=c,
        h=h,
        k_bar=k_bar,
        k_max=k_max,
        k_low=k_low,
        r=r,
        degree_sequence=tuple(sequence),
        negative_degree_count=n - h - r,
        degenerate_fallback=degenerate_fallback,
        sum_degree=sum(sequence),
        parity_touched=False,
        capped_at_n_minus_1=capped,
        degenerate_reason=degenerate_reason,
    )
    validate_plan(plan)
    return plan


def negative_degree_count(n: int, h: int, r: int) -> int:
    """Return ``n - h - r``, the number of nodes at the base low degree."""
    return n - h - r


def plan_degree_sequence(n: int, m: int, c: float = 1.6, h: int = 1) -> DegreePlan:
    """Solve the BiT degree budget for ``(n, m, c, h)``."""
    for name, value in (("n", n), ("m", m), ("h", h)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise DegreeSequenceError(f"{name} must be an integer, got {value!r}")
    if n < 2:
        raise DegreeSequenceError(f"n must be >= 2, got {n}")
    if m < 1:
        raise DegreeSequenceError(f"m must be >= 1, got {m}")
    if not 1 <= h < n:
        raise DegreeSequenceError(f"h must lie in [1, n), got h={h} n={n}")
    if not math.isfinite(float(c)) or float(c) <= 0.0:
        raise DegreeSequenceError(f"c must be a finite positive number, got {c!r}")

    k_bar = 2.0 * m / n
    raw_k_max = round_half_up(c * k_bar)
    k_max = max(HUB_DEGREE_FLOOR, min(n - 1, raw_k_max))
    capped = k_max == n - 1

    remainder_budget = 2 * m - h * k_max
    if remainder_budget < 0:
        raise DegreeSequenceError(
            f"2m={2 * m} < h*k_max={h * k_max}: the {h} hubs need more degree than the graph has, "
            "lower c or h"
        )
    k_low = remainder_budget // (n - h)
    r = remainder_budget - k_low * (n - h)

    if k_max <= k_low:
        reason = (
            f"k_max={k_max} <= k_low={k_low}: c={c} is too small and the bimodal shape collapses, "
            "falling back to k_max=k_low+1 and re-solving k_low to keep sum k = 2m"
        )
        k_low = (2 * m - h) // n
        r = 2 * m - h - k_low * n
        k_max = k_low + 1
        return _assemble(
            n, m, float(c), h, k_bar, k_max, k_low, r,
            degenerate_fallback=True,
            degenerate_reason=reason,
            capped=capped and k_max == n - 1,
        )

    return _assemble(
        n, m, float(c), h, k_bar, k_max, k_low, r,
        degenerate_fallback=False,
        degenerate_reason=None,
        capped=capped,
    )


def build_degree_list(plan: DegreePlan) -> list[int]:
    """Return the degree sequence as a mutable list indexed by node id."""
    return list(plan.degree_sequence)

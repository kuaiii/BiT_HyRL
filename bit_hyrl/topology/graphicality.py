"""Graphicality testing and deterministic Havel-Hakimi realisation."""

from __future__ import annotations

import bisect
import heapq
from dataclasses import dataclass, replace
from typing import Any, Iterable, Sequence

from .degree_sequence import (
    DegreePlan,
    DegreeSequenceError,
    plan_degree_sequence,
    validate_plan,
)

__all__ = [
    "InfeasibleDegreeSequence",
    "GraphicalityLog",
    "erdos_gallai",
    "is_graphical",
    "havel_hakimi",
    "rebalance_plan",
    "realize_with_downgrade",
]


class InfeasibleDegreeSequence(DegreeSequenceError):
    """Raised when no admissible downgrade of a plan becomes realisable."""

    def __init__(self, message: str, log: "GraphicalityLog") -> None:
        super().__init__(message)
        self.log = log


@dataclass(frozen=True)
class GraphicalityLog:
    """Audit trail of the downgrade retry loop."""

    attempts: tuple[dict[str, Any], ...] = ()
    final_realized: bool = False
    final_k_max: int = 0
    final_k_low: int = 0
    downgrade_steps: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Plain-``dict`` view for JSON output."""
        return {
            "attempts": [dict(entry) for entry in self.attempts],
            "final_realized": self.final_realized,
            "final_k_max": self.final_k_max,
            "final_k_low": self.final_k_low,
            "downgrade_steps": self.downgrade_steps,
            "n_attempts": len(self.attempts),
        }


def erdos_gallai(sequence: Sequence[int]) -> bool:
    """Decide graphicality with the Erdős–Gallai theorem in ``O(n log n)``."""
    degrees = sorted((int(value) for value in sequence), reverse=True)
    n = len(degrees)
    if n == 0:
        return True
    if degrees[-1] < 0 or degrees[0] > n - 1:
        return False
    if sum(degrees) % 2 != 0:
        return False

    prefix = [0] * (n + 1)
    for index, value in enumerate(degrees):
        prefix[index + 1] = prefix[index] + value

    negated = [-value for value in degrees]
    for k in range(1, n + 1):
        boundary = bisect.bisect_right(negated, -k)
        if boundary < k:
            boundary = k
        ge_count = boundary - k
        lt_sum = prefix[n] - prefix[boundary]
        rhs = k * (k - 1) + k * ge_count + lt_sum
        if prefix[k] > rhs:
            return False
    return True


def is_graphical(sequence: Sequence[int]) -> bool:
    """Return whether ``sequence`` is realisable by a simple graph."""
    return erdos_gallai(sequence)


def havel_hakimi(sequence: Sequence[int]) -> list[tuple[int, int]] | None:
    """Realise ``sequence`` deterministically, or return ``None`` if impossible."""
    degrees = [int(value) for value in sequence]
    n = len(degrees)
    if n == 0:
        return []
    if any(value < 0 for value in degrees):
        return None
    if any(value > n - 1 for value in degrees):
        return None
    if sum(degrees) % 2 != 0:
        return None

    residual = list(degrees)
    heap: list[tuple[int, int]] = [(-value, node) for node, value in enumerate(residual) if value > 0]
    heapq.heapify(heap)
    edges: list[tuple[int, int]] = []

    while heap:
        neg_degree, hub = heapq.heappop(heap)
        degree = -neg_degree
        if degree != residual[hub]:
            continue
        if degree == 0:
            continue
        residual[hub] = 0

        targets: list[int] = []
        while len(targets) < degree:
            if not heap:
                return None
            neg_target, target = heapq.heappop(heap)
            target_degree = -neg_target
            if target_degree != residual[target] or target_degree == 0:
                continue
            residual[target] = target_degree - 1
            targets.append(target)

        for target in targets:
            if residual[target] > 0:
                heapq.heappush(heap, (-residual[target], target))
            edges.append((hub, target) if hub <= target else (target, hub))

    if sum(residual) != 0:
        return None
    if len(set(edges)) != len(edges):  # pragma: no cover - defensive
        return None
    return edges


def rebalance_plan(plan: DegreePlan, k_max: int) -> DegreePlan:
    """Return a conserving plan with a new ``k_max``."""
    n = plan.n
    h = plan.h
    if not 1 <= k_max <= n - 1:
        raise DegreeSequenceError(f"k_max={k_max} is outside [1, {n - 1}] after rebalancing")
    budget = 2 * plan.m - h * k_max
    if budget < 0:
        raise DegreeSequenceError(f"2m-h*k_max={budget} is negative after rebalancing")
    k_low = budget // (n - h)
    r = budget - k_low * (n - h)
    if k_max <= k_low:
        raise DegreeSequenceError(
            f"k_max={k_max} <= k_low={k_low} after rebalancing, the bimodal shape is gone"
        )
    sequence = [k_max] * h + [k_low + 1] * r + [k_low] * (n - h - r)
    rebalanced = replace(
        plan,
        k_max=k_max,
        k_low=k_low,
        r=r,
        degree_sequence=tuple(sequence),
        negative_degree_count=n - h - r,
        sum_degree=sum(sequence),
        capped_at_n_minus_1=(k_max == n - 1),
        degenerate_fallback=False,
        degenerate_reason=None,
    )
    validate_plan(rebalanced)
    return rebalanced


def realize_with_downgrade(plan: DegreePlan,
                           max_attempts: int = 64) -> tuple[DegreePlan, GraphicalityLog]:
    """Find a realisable neighbour of ``plan`` by lowering ``k_max`` step by step."""
    if max_attempts < 1:
        raise DegreeSequenceError(f"max_attempts must be >= 1, got {max_attempts}")

    attempts: list[dict[str, Any]] = []
    current = plan
    for attempt in range(max_attempts):
        edges = havel_hakimi(current.degree_sequence)
        ok = edges is not None
        attempts.append(
            {
                "attempt": attempt,
                "k_max": current.k_max,
                "k_low": current.k_low,
                "r": current.r,
                "hh_ok": ok,
                "reason": "ok" if ok else "havel_hakimi_failed",
            }
        )
        if ok:
            log = GraphicalityLog(
                attempts=tuple(attempts),
                final_realized=True,
                final_k_max=current.k_max,
                final_k_low=current.k_low,
                downgrade_steps=attempt,
            )
            return current, log

        next_k_max = current.k_max - 1
        if next_k_max < 2 or next_k_max <= current.k_low:
            attempts.append(
                {
                    "attempt": attempt + 1,
                    "k_max": next_k_max,
                    "k_low": current.k_low,
                    "r": current.r,
                    "hh_ok": False,
                    "reason": "k_max fell below k_low, cannot downgrade further",
                }
            )
            break
        try:
            current = rebalance_plan(current, next_k_max)
        except DegreeSequenceError as exc:
            attempts.append(
                {
                    "attempt": attempt + 1,
                    "k_max": next_k_max,
                    "k_low": current.k_low,
                    "r": current.r,
                    "hh_ok": False,
                    "reason": f"rebalance failed: {exc}",
                }
            )
            break

    log = GraphicalityLog(
        attempts=tuple(attempts),
        final_realized=False,
        final_k_max=current.k_max,
        final_k_low=current.k_low,
        downgrade_steps=len(attempts) - 1,
    )
    raise InfeasibleDegreeSequence(
        f"still not graphical after {max_attempts} downgrades (last k_max={current.k_max}, "
        f"k_low={current.k_low}）",
        log,
    )


def graphicality_report(sequence: Iterable[int]) -> dict[str, Any]:
    """Small diagnostic record about one candidate sequence."""
    values = [int(value) for value in sequence]
    n = len(values)
    return {
        "n": n,
        "sum": sum(values),
        "sum_is_even": sum(values) % 2 == 0,
        "max": max(values) if values else 0,
        "min": min(values) if values else 0,
        "erdos_gallai": erdos_gallai(values),
        "havel_hakimi_ok": havel_hakimi(values) is not None,
    }


def realize_sequence(sequence: Sequence[int]) -> list[tuple[int, int]]:
    """Realise a sequence or raise :class:`InfeasibleDegreeSequence`."""
    edges = havel_hakimi(sequence)
    if edges is None:
        raise InfeasibleDegreeSequence(
            f"degree sequence is not graphical (n={len(sequence)}, sum k={sum(sequence)})",
            GraphicalityLog(),
        )
    return edges


def default_plan(n: int, m: int, c: float = 1.6, h: int = 1) -> DegreePlan:
    """Convenience wrapper around :func:`plan_degree_sequence`."""
    return plan_degree_sequence(n, m, c=c, h=h)

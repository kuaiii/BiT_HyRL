"""Graphicality testing and deterministic Havel–Hakimi realisation.

Two design decisions carry all the weight here:

**Determinism without randomness.**  The Havel–Hakimi loop always picks the
remaining node with the largest residual degree, ties broken by ascending node
id, and connects it to the ``d`` nodes with the largest residual degree under
the same tie-break.  No random draws are involved, so realising the same
:class:`~bit_hyrl.topology.degree_sequence.DegreePlan` twice produces a
bit-identical edge set.  That gives an extremely sharp test: hash the edge set
twice and compare.

**Downgrading instead of patching.**  When a plan is not realisable, the retry
lowers ``k_max`` by one and re-solves ``k_low`` / ``r`` *from conservation*, so
every intermediate plan still satisfies ``Σk = 2m``.  Patch-style repairs
("add or drop an edge until it works") are exactly what broke reviewer
confidence in the old implementation.

A note on the design document's wording: it describes the downgrade as
"decrease ``k_max`` and ``k_low`` by one simultaneously".  That cannot be done
while keeping ``r`` in ``[0, n-h)`` — writing ``k_low' = k_low - 1`` forces
``r' = r + n``, which is out of range by construction.  This implementation
therefore decrements ``k_max`` and lets ``k_low`` and ``r`` fall out of the
conservation identity, which is the only downgrade that stays conserving at
every step.
"""

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
    """Audit trail of the downgrade retry loop.

    Attributes:
        attempts: One entry per attempt:
            ``{"attempt", "k_max", "k_low", "r", "hh_ok", "reason"}``.
        final_realized: Whether the last attempt succeeded.
        final_k_max: ``k_max`` of the plan that was returned (or last tried).
        final_k_low: ``k_low`` of the plan that was returned (or last tried).
        downgrade_steps: Number of downgrades applied (``len(attempts) - 1`` on
            success).
    """

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
    """Decide graphicality with the Erdős–Gallai theorem in ``O(n log n)``.

    A non-increasing sequence ``d`` is the degree sequence of a simple graph iff

    ``Σ_{i<=k} d_i <= k(k-1) + Σ_{i>k} min(d_i, k)``  for all ``1 <= k <= n``,

    with ``Σ d_i`` even and ``d_1 <= n - 1``.

    Args:
        sequence: Candidate degree sequence (any order).

    Returns:
        ``True`` if the sequence is realisable by a simple graph.
    """
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

    # ``degrees`` is non-increasing, so ``-degrees`` is non-decreasing.  For the
    # k-th inequality the summation over ``i > k`` splits at the first index with
    # ``d_i < k``; a bisect on the negated list finds it in O(log n), and the
    # pointer can only move left as k grows, so per-k cost stays O(log n).
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
    """Realise ``sequence`` deterministically, or return ``None`` if impossible.

    The algorithm is textbook Havel–Hakimi with a max-heap keyed by
    ``(-residual_degree, node_id)``.  Node ids are used directly as vertex
    labels, so ``sequence[i]`` is the degree of node ``i`` — the explicit
    assignment demanded by the plan.

    Args:
        sequence: Degree sequence indexed by node id.

    Returns:
        A list of canonical ``(min, max)`` edge pairs, or ``None`` when the
        sequence is not graphical.
    """
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
            continue  # stale entry, superseded by a later decrement
        if degree == 0:
            continue
        residual[hub] = 0

        targets: list[int] = []
        while len(targets) < degree:
            if not heap:
                return None  # not enough positive-degree partners left
            neg_target, target = heapq.heappop(heap)
            target_degree = -neg_target
            if target_degree != residual[target] or target_degree == 0:
                continue  # stale
            # Decrement immediately but do *not* push back yet: pushing during the
            # selection loop would let the same neighbour be chosen twice and
            # would create a parallel edge, silently inflating the degree.
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
    """Return a conserving plan with a new ``k_max``.

    ``k_low`` and ``r`` are re-derived from ``R = 2m - h·k_max`` so the edge count
    is preserved exactly.

    Args:
        plan: Current plan.
        k_max: Desired hub degree.

    Returns:
        A validated plan.

    Raises:
        DegreeSequenceError: If the resulting plan violates an invariant (e.g.
            ``k_low`` would drop below 1).
    """
    n = plan.n
    h = plan.h
    if not 1 <= k_max <= n - 1:
        raise DegreeSequenceError(f"rebalance 后 k_max={k_max} 越界 [1, {n - 1}]")
    budget = 2 * plan.m - h * k_max
    if budget < 0:
        raise DegreeSequenceError(f"rebalance 后 2m-h·k_max={budget} < 0")
    k_low = budget // (n - h)
    r = budget - k_low * (n - h)
    if k_max <= k_low:
        raise DegreeSequenceError(
            f"rebalance 后 k_max={k_max} <= k_low={k_low}，双峰结构已消失"
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
    """Find a realisable neighbour of ``plan`` by lowering ``k_max`` step by step.

    Each failed attempt lowers ``k_max`` by one and re-solves ``k_low`` / ``r``
    from conservation, so every attempt is itself a valid, conserving plan.
    The loop stops as soon as Havel–Hakimi succeeds, or as soon as further
    downgrading would break ``k_max > k_low`` / ``k_low >= 1``.

    Args:
        plan: Starting plan (usually from
            :func:`~bit_hyrl.topology.degree_sequence.plan_degree_sequence`).
        max_attempts: Hard cap on the number of attempts.

    Returns:
        ``(realised_plan, log)`` where ``log.final_realized`` is ``True``.

    Raises:
        InfeasibleDegreeSequence: If no attempt within the budget succeeds. The
            exception carries the log so the caller can report every attempt.
    """
    if max_attempts < 1:
        raise DegreeSequenceError(f"max_attempts 必须 >= 1，收到 {max_attempts}")

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
        # Stop *before* producing an invalid plan: k_max must stay above k_low,
        # and lowering k_max raises k_low, so this window closes quickly.
        if next_k_max < 2 or next_k_max <= current.k_low:
            attempts.append(
                {
                    "attempt": attempt + 1,
                    "k_max": next_k_max,
                    "k_low": current.k_low,
                    "r": current.r,
                    "hh_ok": False,
                    "reason": "k_max 降到 k_low 以下，无法继续降阶",
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
                    "reason": f"rebalance 失败：{exc}",
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
        f"{max_attempts} 次降阶尝试后仍不可图化（最后一次 k_max={current.k_max}, "
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
    """Realise a sequence or raise :class:`InfeasibleDegreeSequence`.

    A thin convenience wrapper used by tests and by the CLI's ``verify`` pass.
    """
    edges = havel_hakimi(sequence)
    if edges is None:
        raise InfeasibleDegreeSequence(
            f"度序列不可图化（n={len(sequence)}, Σk={sum(sequence)}）",
            GraphicalityLog(),
        )
    return edges


def default_plan(n: int, m: int, c: float = 1.6, h: int = 1) -> DegreePlan:
    """Convenience wrapper around :func:`plan_degree_sequence`."""
    return plan_degree_sequence(n, m, c=c, h=h)

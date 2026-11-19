"""Degree-sequence planning for the BiT (bimodality-tunable) construction.

The whole construction is driven by four integers, ``(n, m, c, h)``:

.. code-block:: text

    k_bar = 2M / N                              # kept as a float, never pre-rounded
    k_max = max(2, min(N-1, round_half_up(c * k_bar)))
    R     = 2M - h * k_max
    k_low = R // (N - h)
    r     = R - k_low * (N - h)                 # 0 <= r <= N - h
    seq   = [k_max] * h + [k_low + 1] * r + [k_low] * (N - h - r)

``c`` is the *only* shape knob: ``c -> 1`` collapses the two modes, ``c`` large
pushes everything into one super-hub connected to every other node.

Four properties are structural consequences of this definition, not extra
constraints bolted on afterwards.  They are stated explicitly because the
earlier implementation derived ``k_max`` and ``k_min`` from two independent
closed forms and therefore did not conserve the edge count at all.

1. **Σk = 2M holds identically.**
   ``h·k_max + (k_low+1)·r + k_low·(N−h−r) = h·k_max + k_low·(N−h) + r
   = h·k_max + R = 2M``.
   There is consequently **no** "fix up the edge sum" step anywhere in this
   package.  The old 495-vs-588 discrepancy was caused precisely by such a
   repair step papering over two incompatible roundings.

2. **``k_low >= 1`` is a corollary, not a constraint.**  Because
   ``k_low = ⌊R / (N−h)⌋`` and ``R = 2M − h·k_max``, any graph with
   ``2M >= h·k_max + (N−h)`` — the ordinary prerequisite of a connected simple
   graph — yields ``k_low >= 1``.  The code therefore expresses it as an
   invariant check and never as "if k_low < 1 then set k_low = 1", which would
   reintroduce the conservation error.

3. **The parity branch is unreachable.**  Kept as the constant field
   ``parity_touched`` (always ``False``) purely so the code can be compared
   line-by-line against the old paper description; enforcing it as an assertion
   rather than as a live branch is what makes the dead code impossible to
   resurrect by accident.  The design document fixes the wording, quoted here
   verbatim so the two cannot drift apart:

   奇偶修正分支不可达：因为 `Σk = h·k_max + k_low·(N−h) + r = 2M` 恒等成立，
   而 `2M` 恒为偶数。旧实现之所以必须存在"奇偶修补"，恰恰因为它的 `k_max` 与
   `k_min` 来自两个互不守恒的独立公式，求和后可能为奇数；本式不存在该问题。

4. **``round_half_up`` is implemented explicitly** as ``floor(x + 0.5)``.
   Python's built-in ``round`` is banker's rounding (``round(2.5) == 2``), which
   makes ``k_max`` jitter in an inexplicable way across ``c`` values.

Two normalised ``c`` views are exported on every plan
------------------------------------------------------
``c`` is *not* comparable across graphs.  ``k_max = round(c·k̄)`` saturates at
``N − 1``, so the critical value at which saturation starts is

.. code-block:: text

    c_star = (N - 1) / k_bar
    c_norm = c / c_star          # 1.0 == exactly at the ceiling

and it spans almost an order of magnitude over the nine evaluation graphs
(BA-100 ≈ 17.0, BA-200 ≈ 33.7, BA-300 ≈ 50.3, BA-400 ≈ 50.4, BA-500 ≈ 83.7,
Cogentco ≈ 79.4, Colt ≈ 65.7, GtsCe ≈ 57.1, UsCarrier ≈ 65.6).  A sweep that
holds ``c`` fixed across graphs is therefore not a controlled comparison; the
sensitivity table is keyed on ``c_norm`` and carries raw ``c`` as a side column.
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

#: Node ids ``0 .. h-1`` are the hubs; the plan assigns the highest degree to
#: them explicitly (see :meth:`DegreePlan.hub_nodes`).
HUB_DEGREE_FLOOR = 2


class DegreeSequenceError(ValueError):
    """Raised when a degree plan violates its invariants.

    This is a *contract* violation, not a numerical accident: reaching it means
    the formula was changed, not that an input graph is unusual.
    """


def round_half_up(value: float) -> int:
    """Round half away from zero, i.e. ``floor(x + 0.5)`` for ``x >= 0``.

    Python's built-in :func:`round` uses banker's rounding, so ``round(2.5) == 2``
    and ``round(0.5) == 0``.  Using it here would make ``k_max`` depend on the
    parity of the fractional part, producing the "jitter between ``c`` grid
    points" that the old implementation could not explain.

    Args:
        value: A finite real number.

    Returns:
        The nearest integer, ties resolved upwards.

    Raises:
        DegreeSequenceError: If ``value`` is not finite.
    """
    number = float(value)
    if not math.isfinite(number):
        raise DegreeSequenceError(f"round_half_up 需要有限值，收到 {number!r}")
    return int(math.floor(number + 0.5))


@dataclass(frozen=True)
class DegreePlan:
    """The exact degree budget of one BiT construction.

    The tuple ``degree_sequence`` is simultaneously

    * sorted in non-increasing order, and
    * **indexed by node id**: node ``i`` must be given degree
      ``degree_sequence[i]``.

    Those two readings coincide, and that is deliberate: it makes the degree
    assignment explicit (which node is the hub, which ``r`` nodes get
    ``k_low + 1``) instead of leaving it to whatever order the realiser happens
    to use.  Because the assignment is explicit, the realised histogram can be
    compared with :meth:`degree_histogram` by strict equality — there is no room
    for an "approximately matching" degree distribution.

    Attributes:
        n: Number of nodes.
        m: Number of edges *after* deduplication and self-loop removal.
        c: Hub factor.
        h: Number of hubs (nodes ``0 .. h-1``).
        k_bar: ``2m / n`` as an unrounded float.
        k_max: Hub degree.
        k_low: Base low degree.
        r: Number of low-degree nodes that receive ``k_low + 1``.
        degree_sequence: Degree per node id, non-increasing.
        negative_degree_count: ``n - h - r``, the number of nodes at ``k_low``.
            (Name kept from the design document for grep-ability.)
        degenerate_fallback: Whether the ``k_max <= k_low`` rollback ran (§6.1.1).
        sum_degree: ``sum(degree_sequence)``; must equal ``2m``.
        parity_touched: Always ``False``; retained for comparison with the old
            paper description (see module docstring, property 3).
        capped_at_n_minus_1: Whether ``k_max`` hit the ``n - 1`` ceiling, i.e.
            whether ``c`` is above its effective upper bound.
        degenerate_reason: Human-readable cause when ``degenerate_fallback``.
    """

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

    # -- derived views ----------------------------------------------------- #
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
        """Return ``(N - 1) / k_bar``, the ``c`` at which ``k_max`` saturates.

        ``k_max = round_half_up(c·k̄)`` reaches the simplicity ceiling ``N - 1``
        once ``c·k̄ >= N - 1.5``; ``(N-1)/k̄`` is the exact-ish crossover and is
        the natural unit for a *cross-graph* comparison of ``c``.

        Returns:
            The crossover factor, or ``0.0`` when ``k_bar`` is zero.
        """
        if self.k_bar <= 0.0:
            return 0.0
        return (self.n - 1) / self.k_bar

    def c_norm(self) -> float:
        """Return ``c / c_star``; ``1.0`` means "exactly at the ceiling"."""
        star = self.c_star()
        return (self.c / star) if star > 0.0 else 0.0

    # -- invariants -------------------------------------------------------- #
    def check_conservation(self) -> None:
        """Assert every structural invariant of the plan.

        Raises:
            DegreeSequenceError: If any invariant is violated. The message names
                the invariant, because "the numbers do not add up" is useless
                during a paper revision.
        """
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
    """Check every invariant of ``plan``; raise :class:`DegreeSequenceError` otherwise.

    The checks are, in the order they are stated in the module docstring:

    * ``sum(degree_sequence) == 2m`` (conservation);
    * ``sum(degree_sequence) % 2 == 0`` (parity — unreachable branch, asserted);
    * ``k_low >= 1`` (corollary, asserted rather than patched);
    * ``k_max <= n - 1`` (simplicity);
    * ``0 <= r <= n - h`` (the remainder of the division; ``r == n - h`` is the
  degenerate-rollback boundary where every non-hub node carries ``k_low + 1``);
    * ``len(degree_sequence) == n`` and the sequence is non-increasing;
    * ``negative_degree_count == n - h - r``.
    """
    if plan.n < 2:
        raise DegreeSequenceError(f"n 必须 >= 2，收到 {plan.n}")
    if plan.m < 1:
        raise DegreeSequenceError(f"m 必须 >= 1，收到 {plan.m}")
    if not 1 <= plan.h < plan.n:
        raise DegreeSequenceError(f"h 必须落在 [1, n)，收到 h={plan.h} n={plan.n}")
    if len(plan.degree_sequence) != plan.n:
        raise DegreeSequenceError(
            f"度序列长度 {len(plan.degree_sequence)} != n={plan.n}"
        )

    total = sum(plan.degree_sequence)
    if total != plan.sum_degree:
        raise DegreeSequenceError(
            f"sum_degree 字段 {plan.sum_degree} 与实际和 {total} 不一致"
        )
    if total != 2 * plan.m:
        raise DegreeSequenceError(
            f"守恒被破坏：Σk={total}，2m={2 * plan.m}（差 {total - 2 * plan.m}）"
        )
    if total % 2 != 0:
        # Unreachable while Σk == 2m holds; kept as an explicit guard so that the
        # dead "parity repair" branch of the old implementation can never return.
        raise DegreeSequenceError("Σk 恒为偶数；若触发说明公式已被改动")
    if plan.parity_touched:
        raise DegreeSequenceError("parity_touched 必须恒为 False（奇偶修补分支不可达）")
    if plan.k_low < 1:
        raise DegreeSequenceError(
            f"k_low={plan.k_low} < 1：守恒式推论失效，说明前提 2M >= h·k_max + (N−h) 不成立"
        )
    if plan.k_max > plan.n - 1:
        raise DegreeSequenceError(f"k_max={plan.k_max} > n-1={plan.n - 1}，不是简单图")
    if plan.k_max < HUB_DEGREE_FLOOR:
        raise DegreeSequenceError(f"k_max={plan.k_max} < {HUB_DEGREE_FLOOR}")
    if not 0 <= plan.r <= plan.n - plan.h:
        raise DegreeSequenceError(f"r={plan.r} 越界，应满足 0 <= r <= n-h={plan.n - plan.h}")
    if plan.negative_degree_count != plan.n - plan.h - plan.r:
        raise DegreeSequenceError(
            f"negative_degree_count={plan.negative_degree_count} != n-h-r="
            f"{plan.n - plan.h - plan.r}"
        )
    for left, right in zip(plan.degree_sequence, plan.degree_sequence[1:]):
        if left < right:
            raise DegreeSequenceError(
                "degree_sequence 必须非递增（节点 0..h-1 是 hub，r 个 mid 节点紧随其后）"
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
    """Solve the BiT degree budget for ``(n, m, c, h)``.

    Args:
        n: Number of nodes.
        m: Number of edges (after dedupe and self-loop removal).
        c: Hub factor. ``k_max = round_half_up(c * k_bar)``, clipped to
            ``[2, n-1]``.
        h: Number of hubs.

    Returns:
        A validated :class:`DegreePlan` whose ``sum_degree == 2 * m`` exactly.

    Raises:
        DegreeSequenceError: If the arguments are inadmissible, if the ``2m < h·k_max``
            prerequisite fails, or if the solved ``k_low`` falls below 1 — all of
            which mean the *input* is wrong, never that the formula needs a patch.
    """
    for name, value in (("n", n), ("m", m), ("h", h)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise DegreeSequenceError(f"{name} 必须是整数，收到 {value!r}")
    if n < 2:
        raise DegreeSequenceError(f"n 必须 >= 2，收到 {n}")
    if m < 1:
        raise DegreeSequenceError(f"m 必须 >= 1，收到 {m}")
    if not 1 <= h < n:
        raise DegreeSequenceError(f"h 必须落在 [1, n)，收到 h={h} n={n}")
    if not math.isfinite(float(c)) or float(c) <= 0.0:
        raise DegreeSequenceError(f"c 必须是有限正数，收到 {c!r}")

    k_bar = 2.0 * m / n
    raw_k_max = round_half_up(c * k_bar)
    k_max = max(HUB_DEGREE_FLOOR, min(n - 1, raw_k_max))
    capped = k_max == n - 1

    remainder_budget = 2 * m - h * k_max
    if remainder_budget < 0:
        raise DegreeSequenceError(
            f"2m={2 * m} < h·k_max={h * k_max}：h={h} 个 hub 的度预算已超过总度数，"
            "降低 c 或 h"
        )
    k_low = remainder_budget // (n - h)
    r = remainder_budget - k_low * (n - h)

    if k_max <= k_low:
        # ---- §6.1.1 degenerate rollback, still conserving ------------------ #
        # "Bimodality" has collapsed into a single mode (only reachable for
        # low mean degree with small c, e.g. Cogentco at c = 1.0).  Rather than
        # forcing a spurious hub, set k_max = k_low + 1 and re-solve k_low from
        # conservation:
        #   h(k_low+1) + (k_low+1)r + k_low(n-h-r) = k_low·n + h + r = 2m
        #
        # Why not the "obvious" alternative of dropping k_max *and* k_low by one
        # simultaneously?  Because conservation forbids it.  Starting from
        #   h·k_max + k_low·(n−h) + r = 2m
        # and replacing k_max → k_max−1, k_low → k_low−1, r → r' gives
        #   h(k_max−1) + (k_low−1)(n−h) + r' = 2m
        # Subtracting the original identity leaves
        #   −h − (n−h) + (r' − r) = 0   ⇒   r' = r + n,
        # which violates the division-remainder range 0 <= r' < n − h.  The
        # simultaneous decrement is therefore not a solution at all; only
        # ``k_max ← k_max − 1`` with ``k_low``/``r`` re-solved from ``2m`` is.
        reason = (
            f"k_max={k_max} <= k_low={k_low}（c={c} 过小，双峰退化为单峰），"
            "按 k_max=k_low+1 回退并重解 k_low 以保持 Σk=2m"
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

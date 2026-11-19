"""Attack-order producers for the evaluation layer.

The published protocol needs three tiers of node-removal order:

1. **Tier A — static centralities** (``degree`` / ``betweenness`` /
   ``closeness`` / ``coreness`` / ``eigenvector``).
2. **Tier B — adaptive / structural heuristics** (:mod:`.advanced`):
   ``adaptive``, ``correlated``, ``multistage``.
3. **Tier C — a learned dismantler** (``GDM``), reached only through a lazy,
   read-only bridge into the vendored ``network_dismantling/GDM`` tree.

This package currently ships Tier B and Tier C, because those are the four
attackers the paper's ``tab:advanced_attacks`` (Table 7) reports and that no
other module of the package produced.  Tier A lives in
:mod:`bit_hyrl.control.feasibility` as
:func:`~bit_hyrl.control.feasibility.provisional_attack_sequence`, which is
the historical producer and is deliberately **not** duplicated here.

Every producer has the same shape: ``f(graph, *, seed=0) -> list[int]``,
returning a **permutation of the node ids** in removal order.  That contract is
what lets the sweep script treat an attack as a drop-in replacement for
``provisional_attack_sequence`` without branching on which family it belongs to.
"""

from __future__ import annotations

from .advanced import (
    ACTION_TABLE,
    AdvancedAttackError,
    GdmUnavailableError,
    adaptive_sequence,
    assert_permutation,
    correlated_sequence,
    gdm_sequence,
    multistage_sequence,
)

__all__ = [
    "ACTION_TABLE",
    "AdvancedAttackError",
    "GdmUnavailableError",
    "adaptive_sequence",
    "assert_permutation",
    "correlated_sequence",
    "gdm_sequence",
    "multistage_sequence",
]

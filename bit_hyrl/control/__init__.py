"""Control plane: capacity model, single-valued assignment, cascade engine.

Deliberately ignorant of *how* the topology was produced and of the RL layer: it
takes an :class:`networkx.Graph` plus a controller list, and returns a
:class:`~bit_hyrl.control.cascade.CascadeTrace`.  That is what makes the
``structural`` mode reproducible against the already-published results and the
``capacity`` mode an honest opt-in extension rather than a redefinition.

Four modules, one direction of dependency:

* :mod:`.capacity` — the ``(n_c, C)`` budget, the per-component feasibility audit
  and the utilisation probe (``u_nearest`` / ``u_after_spill`` / ``feasible``);
* :mod:`.assignment` — the single-valued, minimal-deviation switch → controller
  map (nearest first, spill only when capacity forces it, **never** balancing);
* :mod:`.cascade` — the two cascade semantics on one loop;
* :mod:`.feasibility` — the capacity verdict as a function of the removal
  fraction, which is the evidence the analysis actually needs.

Layer rule: ``control`` may import :mod:`bit_hyrl.topology`'s
pure helpers (e.g. ``round_half_up``) but never ``rl``, and ``topology`` never
imports ``control``.
"""

from __future__ import annotations

from .assignment import (
    Assignment,
    assign_single_controller,
    bfs_rows,
    reassign_excess,
)
from .capacity import (
    DEFAULT_CONTROLLER_RATIO,
    DEFAULT_DEMAND,
    DEFAULT_THETA,
    GOLDEN_CAPACITY,
    CapacityError,
    CapacityModel,
    ComponentCapacity,
    UtilisationProbe,
    capacity_model,
    capacity_per_controller,
    component_capacity_audit,
    n_controllers_for,
    probe_max_utilisation,
)
from .cascade import (
    CASCADE_MODES,
    CascadeError,
    CascadeStep,
    CascadeTrace,
    recompute_loads,
    run_cascade,
    surviving_components,
)
from .feasibility import (
    PROVISIONAL_ATTACK_METHODS,
    FeasibilityCurve,
    FeasibilityPoint,
    ModeGap,
    OnsetRecord,
    controllers_degree_ranked,
    density_floor,
    feasibility_curve,
    format_feasibility_table,
    format_onset_table,
    mode_gap,
    onset_records,
    provisional_attack_sequence,
    write_feasibility_outputs,
    write_onset_outputs,
)

__all__ = [
    "CASCADE_MODES",
    "DEFAULT_CONTROLLER_RATIO",
    "DEFAULT_DEMAND",
    "DEFAULT_THETA",
    "GOLDEN_CAPACITY",
    "PROVISIONAL_ATTACK_METHODS",
    "Assignment",
    "CapacityError",
    "CapacityModel",
    "CascadeError",
    "CascadeStep",
    "CascadeTrace",
    "ComponentCapacity",
    "FeasibilityCurve",
    "FeasibilityPoint",
    "ModeGap",
    "OnsetRecord",
    "UtilisationProbe",
    "assign_single_controller",
    "bfs_rows",
    "capacity_model",
    "capacity_per_controller",
    "component_capacity_audit",
    "controllers_degree_ranked",
    "density_floor",
    "feasibility_curve",
    "format_feasibility_table",
    "format_onset_table",
    "mode_gap",
    "n_controllers_for",
    "onset_records",
    "probe_max_utilisation",
    "provisional_attack_sequence",
    "reassign_excess",
    "recompute_loads",
    "run_cascade",
    "surviving_components",
    "write_feasibility_outputs",
    "write_onset_outputs",
]

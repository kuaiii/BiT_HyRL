"""The analysis layer: 2x2 ablation (T11), matrix (T12), stats (T12), figures (T12).

Everything here **reads** the frozen :class:`~bit_hyrl.evaluation.record.EvaluationRecord`
metric set (via :func:`bit_hyrl.evaluation.runner.run_evaluation`) and never
touches the topology / control / cascade / capacity logic.  All randomness is
derived from :mod:`bit_hyrl.rng`; no process-global seeding, and no ``print``
in these library modules (the CLI is the only place that prints).

The ``figures`` submodule is intentionally **not** imported here so that importing
the package does not drag in ``matplotlib`` for the tabular commands.
"""

from __future__ import annotations

from .ablation import run_ablation, write_ablation_outputs
from .matrix import run_matrix, write_matrix_outputs
from .stats import paired_stats, write_stats_outputs

__all__ = [
    "run_ablation",
    "write_ablation_outputs",
    "run_matrix",
    "write_matrix_outputs",
    "paired_stats",
    "write_stats_outputs",
]

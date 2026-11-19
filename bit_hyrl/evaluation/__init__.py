"""Evaluation layer: end-to-end runner, record schema and metrics.

The runner is graph-agnostic — it serves both the *original* substrate and the
*BiT-reconstructed* proposal through one ``topology`` selector, so the two
robustness curves needed for interpretation A come from a single code path.
"""

from __future__ import annotations

from .metrics import calculate_csa, calculate_h_c, calculate_h_c_norm
from .record import EvaluationRecord
from .runner import TOPOLOGIES, build_eval_graph, run_evaluation

__all__ = [
    "TOPOLOGIES",
    "EvaluationRecord",
    "build_eval_graph",
    "calculate_csa",
    "calculate_h_c",
    "calculate_h_c_norm",
    "run_evaluation",
]

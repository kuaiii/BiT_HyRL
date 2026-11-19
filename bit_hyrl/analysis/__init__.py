"""Analysis helpers: ablation, method matrix, paired statistics and figures."""

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

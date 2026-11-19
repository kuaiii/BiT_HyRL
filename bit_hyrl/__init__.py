"""BiT-HyRL: bimodal topology construction, controller placement and resilience
evaluation.

Subpackages: ``topology`` (BiT), ``control`` (capacity, assignment, cascade),
``rl`` (PPO placement), ``evaluation`` (metrics, runner) and ``analysis``
(tables, statistics, figures).
"""

from __future__ import annotations

from .rng import derive, derive_int, derive_permutation, derive_seed_sequence
from .version import EVAL_SCHEMA_VERSION, SCHEMA_VERSION, VERSION
from .io.graph_io import (
    GraphLoadError,
    GraphSpecMismatch,
    edge_list_sha256,
    graph_fingerprint,
    load_graph,
    load_graph_spec,
    load_graph_with_meta,
)
from .io.jsonlog import dump_json, dumps, load_json, read_jsonl, sha256_json, write_jsonl
from .topology import (
    DEFAULT_C_GRID,
    ConstructionError,
    ConstructionResult,
    DegreePlan,
    DegreeSequenceError,
    InfeasibleDegreeSequence,
    TopologyDiagnostics,
    bc_internal,
    build_bimodal_graph,
    diagnose,
    havel_hakimi,
    kappa,
    plan_degree_sequence,
    realize_with_downgrade,
    sweep_c,
    sweep_dataset,
)

__all__ = [
    "DEFAULT_C_GRID",
    "EVAL_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "VERSION",
    "ConstructionError",
    "ConstructionResult",
    "DegreePlan",
    "DegreeSequenceError",
    "GraphLoadError",
    "GraphSpecMismatch",
    "InfeasibleDegreeSequence",
    "TopologyDiagnostics",
    "bc_internal",
    "build_bimodal_graph",
    "derive",
    "derive_int",
    "derive_permutation",
    "derive_seed_sequence",
    "diagnose",
    "dump_json",
    "dumps",
    "edge_list_sha256",
    "graph_fingerprint",
    "havel_hakimi",
    "kappa",
    "load_graph",
    "load_graph_spec",
    "load_graph_with_meta",
    "load_json",
    "plan_degree_sequence",
    "read_jsonl",
    "realize_with_downgrade",
    "sha256_json",
    "sweep_c",
    "sweep_dataset",
    "write_jsonl",
]

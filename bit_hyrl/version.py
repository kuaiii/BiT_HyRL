"""Version pins for the BiT-HyRL rebuild.

Single place where every version string lives, so that a record produced months
ago can be tied back to the exact schema that wrote it.
"""

from __future__ import annotations

__all__ = [
    "VERSION",
    "SCHEMA_VERSION",
    "EVAL_SCHEMA_VERSION",
    "TOPOLOGY_SCHEMA_VERSION",
    "SWEEP_SCHEMA_VERSION",
    "FINGERPRINT_SCHEMA_VERSION",
]

#: Package version.
VERSION = "1.0.0"

#: Root schema marker written into every JSON artefact.
SCHEMA_VERSION = "bit_hyrl/1.0"

#: ``EvaluationRecord`` schema.
EVAL_SCHEMA_VERSION = "bit_hyrl/eval/1.0"

#: ``DegreePlan`` / ``ConstructionResult`` schema.
TOPOLOGY_SCHEMA_VERSION = "bit_hyrl/topology/1.0"

#: ``sweep_c`` row schema.
SWEEP_SCHEMA_VERSION = "bit_hyrl/sweep/1.0"

#: ``graph_fingerprints.json`` schema.
FINGERPRINT_SCHEMA_VERSION = "bit_hyrl/fingerprint/1.0"

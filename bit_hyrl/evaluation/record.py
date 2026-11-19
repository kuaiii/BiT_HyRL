"""Frozen evaluation record.

One :class:`EvaluationRecord` captures a single ``(dataset, topology, attack,
mode)`` evaluation: the robustness curve plus the final-state control metrics.  The
frozen dataclass + ``to_dict`` + :meth:`save` give a stable, canonical JSON
artefact (one file per run under ``results/records/``) so that later stages
(T11 ablation, T12 matrix/stats) can read them back without re-running the cascade.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..io.paths import make_run_id, records_dir
from ..version import EVAL_SCHEMA_VERSION


@dataclass(frozen=True)
class EvaluationRecord:
    """One evaluated ``(dataset, topology, attack, mode)`` point.

    Attributes:
        schema_version: Eval-record schema marker.
        dataset: Dataset name, e.g. ``"BA-400"``.
        topology: ``"original"`` (load the input graph) or ``"bit"``
            (BiT-reconstruct from ``(N, M)``).
        c: Hub factor used for the BiT reconstruction (only meaningful when
            ``topology == "bit"``; always recorded for traceability).
        h: Number of hubs in the BiT reconstruction.
        seed: Root seed.
        n_controllers: Controller count placed by the degree-ranked policy.
        attack: Attack method (``degree`` / ``random`` / ``betweenness`` /
            ``controllers``).
        mode: Cascade mode (``structural`` / ``capacity``).
        policy: Controller-placement policy label (``"degree_ranked"``).
        f_grid: ``f_direct`` of every cascade step — the ``r_curve`` x-axis.
        r_curve: ``gcc_size / N`` of every cascade step — the y-axis.
        csa_final: Final-state Control Supply Availability (denominator fixed at N).
        h_c_final: Final-state load-share control entropy (nats).
        h_c_norm_final: Final-state normalized control entropy — our CCE proxy.
        auc_r: Trapezoidal area under ``r_curve`` vs ``f_grid`` (pure-GCC area).
        collapse_f: Final failed fraction (``trace.final_fraction``).
        audit_ok: Whether the cascade audit identity
            ``direct + cascade + alive == N`` held at every step.

    Metric-layer upgrade (additive; defaults keep old records loadable):
        auc_gcc / auc_csa / auc_hc: Trapezoidal area of the per-step GCC / CSA /
            H_C curve (each a [0, 1] curve).  ``r_res = (auc_gcc + auc_csa +
            auc_hc) / 3`` is a **mean curve area**, NOT a survival fraction —
            ``r_res > 0.5`` does not mean "50 % survives".
        r_res: Mean of the three AUCs (the headline composite, audit-corrected).
        collapse_f_gcc02: First ``f`` where GCC drops below ``0.2`` (``1.0`` if it
            never does) — the H_C-independent collapse sub-metric.
        r_res_half: ``r_res`` restricted to ``f <= 0.5`` (AUC x 2, mean of three),
            where topology/placement differences live.
        eval_seconds: Wall-clock of this evaluation run.
        placement_seconds: Wall-clock the caller spent selecting the controllers
            (PPO ``.select``); ``0.0`` for degree-ranked.
        csa_curve / hc_curve: Per-step CSA(f) and H_C(f) (``[0, 1]``) curves.
    """

    schema_version: str = EVAL_SCHEMA_VERSION
    dataset: str = ""
    topology: str = "bit"
    c: float = 1.6
    h: int = 1
    seed: int = 0
    n_controllers: int = 0
    attack: str = "degree"
    mode: str = "structural"
    policy: str = "degree_ranked"
    f_grid: tuple[float, ...] = field(default_factory=tuple)
    r_curve: tuple[float, ...] = field(default_factory=tuple)
    csa_final: float = 0.0
    h_c_final: float = 0.0
    h_c_norm_final: float = 0.0
    auc_r: float = 0.0
    collapse_f: float = 0.0
    audit_ok: bool = True
    # ---- metrics upgrade (additive; defaults keep old records loadable) ---- #
    auc_gcc: float = 0.0
    auc_csa: float = 0.0
    auc_hc: float = 0.0
    r_res: float = 0.0
    collapse_f_gcc02: float = 1.0
    r_res_half: float = 0.0
    eval_seconds: float = 0.0
    placement_seconds: float = 0.0
    csa_curve: tuple[float, ...] = field(default_factory=tuple)
    hc_curve: tuple[float, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        """Canonical JSON-ready mapping (sorted keys when written)."""
        return {
            "schema_version": self.schema_version,
            "dataset": self.dataset,
            "topology": self.topology,
            "c": self.c,
            "h": self.h,
            "seed": self.seed,
            "n_controllers": self.n_controllers,
            "attack": self.attack,
            "mode": self.mode,
            "policy": self.policy,
            "f_grid": list(self.f_grid),
            "r_curve": list(self.r_curve),
            "csa_final": self.csa_final,
            "h_c_final": self.h_c_final,
            "h_c_norm_final": self.h_c_norm_final,
            "auc_r": self.auc_r,
            "collapse_f": self.collapse_f,
            "audit_ok": self.audit_ok,
            "auc_gcc": self.auc_gcc,
            "auc_csa": self.auc_csa,
            "auc_hc": self.auc_hc,
            "r_res": self.r_res,
            "collapse_f_gcc02": self.collapse_f_gcc02,
            "r_res_half": self.r_res_half,
            "eval_seconds": self.eval_seconds,
            "placement_seconds": self.placement_seconds,
            "csa_curve": list(self.csa_curve),
            "hc_curve": list(self.hc_curve),
        }

    @property
    def run_id(self) -> str:
        """Canonical run identifier built from the record's own fields."""
        return make_run_id(
            self.dataset, self.topology, self.c, self.policy, self.attack, self.seed
        )

    def filename(self) -> str:
        """The JSON filename this record writes to (``<run_id>.json``)."""
        return f"{self.run_id}.json"

    def save(self, path: str | Path | None = None) -> Path:
        """Write the record as canonical JSON.

        Args:
            path: Destination file; defaults to ``records_dir() / <run_id>.json``.

        Returns:
            The destination path.
        """
        target = (
            Path(path)
            if path is not None
            else records_dir() / self.filename()
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            self.to_dict(), indent=2, sort_keys=True, ensure_ascii=False
        )
        target.write_text(payload + "\n", encoding="utf-8")
        return target

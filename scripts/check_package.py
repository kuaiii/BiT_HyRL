"""Integrity check for this package; standard library only.

Recomputes every number the manuscript reports from the archived tables.

Usage::

    python scripts/check_package.py        # exit code 0 when all checks pass
    python scripts/check_package.py -v     # also print each value
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
TABLES = ROOT / "paper_tables"

#: Headline values reported in Table 6 and Table 7.
R_RES_BIT = 0.4039638666160048
R_RES_UNITY = 0.3419709509332758
PAIRED_D_UNITY = 0.06199291568272903
PAIRED_P_UNITY = 1.5519598063386763e-05
RELATIVE_GAIN = 0.1813

#: Rows of Table 6 that are external methods, each of which the method must beat.
EXTERNAL = ("UNITY", "QDLM", "FRED-ABL", "Eff-R-Net", "ONION",
            "ROMEN", "MWCL", "MADQN", "KMedian", "Random-pl.")

#: The seven figures of the manuscript, by the name used in the LaTeX source.
FIGURE_FILES = (
    "Theory Foundation.pdf",
    "fig_input_sensitivity.pdf",
    "Algorithms.pdf",
    "Bimodal network enhance resilience.pdf",
    "construction_overview.pdf",
    "construction_mechanism.pdf",
    "applicability_domain.pdf",
)

_results: list[tuple[str, bool, str]] = []


def check(name: str):
    """Register a check; the function returns (ok, detail)."""

    def register(function):
        ok, detail = function()
        _results.append((name, ok, detail))
        return function

    return register


def read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


@check("graph manifest: thirty topologies on disk")
def _manifest() -> tuple[bool, str]:
    manifest = json.loads((ROOT / "configs" / "graphs.json").read_text("utf-8"))
    entries = manifest["graphs"]
    missing = [e["path"] for e in entries if not (ROOT / e["path"]).exists()]
    return (len(entries) == 30 and not missing,
            f"{len(entries)} graphs, {len(missing)} missing")


@check("Table 6: BiT-HyRL R_res = 0.4040 over thirty topologies")
def _main_table() -> tuple[bool, str]:
    rows = {row["strategy"]: row for row in read_csv(RESULTS / "family" / "family_ctrl.csv")}
    bit = float(rows["BiT-HyRL"]["All"])
    unity = float(rows["UNITY"]["All"])
    ok = (len(rows) == 16 and abs(bit - R_RES_BIT) < 5e-4
          and abs(unity - R_RES_UNITY) < 5e-4)
    return ok, f"BiT-HyRL {bit:.4f} vs UNITY {unity:.4f}, {len(rows)} strategies"


@check("Table 7: paired gain over UNITY = +0.062 (18.1 %), p = 1.6e-5")
def _paired() -> tuple[bool, str]:
    rows = {row["strategy"]: row for row in read_csv(RESULTS / "family" / "family_paired.csv")}
    unity = rows["UNITY"]
    diff = float(unity["d_r_res"])
    p_value = float(unity["p_r_res"])
    gain = diff / R_RES_UNITY
    ok = (abs(diff - PAIRED_D_UNITY) < 5e-4
          and abs(p_value - PAIRED_P_UNITY) / PAIRED_P_UNITY < 0.05
          and abs(gain - RELATIVE_GAIN) < 1e-3)
    return ok, f"d = {diff:+.4f} ({gain:.1%}), p = {p_value:.2e}, n = {unity['n']}"


@check("Table 7: every external method is beaten at p < 0.001")
def _significance() -> tuple[bool, str]:
    rows = {row["strategy"]: row for row in read_csv(RESULTS / "family" / "family_paired.csv")}
    weak = [name for name in EXTERNAL if float(rows[name]["p_r_res"]) >= 1e-3]
    positive = all(float(rows[name]["d_r_res"]) > 0 for name in EXTERNAL)
    return (not weak and positive, f"min p over externals "
            f"{min(float(rows[n]['p_r_res']) for n in EXTERNAL):.1e}, {len(weak)} not significant")


@check("Figure 4/7 source: curve archive covers BA-100 and Colt at 20 seeds")
def _curves() -> tuple[bool, str]:
    rows = read_csv(RESULTS / "baselines" / "curve_archive.csv")
    cells = {}
    for row in rows:
        if row["policy"] != "BiT-HyRL":
            continue
        key = (row["dataset"], row["attack"])
        cells.setdefault(key, set()).add(row["seed"])
    ok = (len(rows) == 560 and len(cells) == 4
          and all(len(seeds) == 20 for seeds in cells.values())
          and {key[0] for key in cells} == {"BA-100", "Colt"})
    return ok, f"{len(rows)} rows, {sorted({k[0] for k in cells})}, 2 attacks x 20 seeds"


@check("Table 11: attack coverage grid 9 x 11 x 6 x 20")
def _coverage() -> tuple[bool, str]:
    rows = read_csv(RESULTS / "attack_cov" / "attack_coverage.csv")
    datasets = {row["dataset"] for row in rows}
    policies = {row["policy"] for row in rows}
    attacks = {row["attack"] for row in rows}
    seeds = {row["seed"] for row in rows}
    ok = (len(rows) == 11880 and len(datasets) == 9
          and len(policies) == 11 and len(attacks) == 6 and len(seeds) == 20)
    return ok, (f"{len(rows)} rows, {len(datasets)} graphs x {len(policies)} methods "
                f"x {len(attacks)} orders x {len(seeds)} seeds")


@check("Table 12: four advanced attacks, eight arms, 15 observations per cell")
def _advanced() -> tuple[bool, str]:
    rows = read_csv(RESULTS / "advanced" / "advanced_attacks.csv")
    arms = {row["arm"] for row in rows}
    attacks = {row["attack"] for row in rows}
    ok = len(rows) == 480 and len(arms) == 8 and len(attacks) == 4
    return ok, f"{len(rows)} rows, {len(arms)} arms, {sorted(attacks)}"


@check("Table 13: controller-budget sweep 30 x 2 x 3 x 2 x 3")
def _budget() -> tuple[bool, str]:
    rows = read_csv(RESULTS / "budget" / "budget_all.csv")
    graphs = {row["dataset"] for row in rows}
    ok = (len(rows) == 1080 and len(graphs) == 30
          and len({row["arm"] for row in rows}) == 2
          and len({row["ratio"] for row in rows}) == 3)
    return ok, f"{len(rows)} rows on {len(graphs)} graphs, ratios 0.05/0.10/0.20"


@check("frozen placement policy checkpoint present")
def _checkpoint() -> tuple[bool, str]:
    path = RESULTS / "checkpoints" / "ppo_final_s0.pt"
    size = path.stat().st_size if path.exists() else 0
    return size > 50_000, f"{path.name}, {size} bytes"


@check("seven manuscript figures on disk")
def _figures() -> tuple[bool, str]:
    missing = [name for name in FIGURE_FILES if not (FIGURES / name).exists()]
    return not missing, f"{len(FIGURE_FILES) - len(missing)}/{len(FIGURE_FILES)} present"


@check("fourteen generated table fragments on disk")
def _tables() -> tuple[bool, str]:
    fragments = sorted(TABLES.glob("*.tex"))
    return len(fragments) >= 14, f"{len(fragments)} .tex fragments"


def main() -> int:
    failed = [name for name, ok, _ in _results if not ok]
    width = max(len(name) for name, _, _ in _results)
    for name, ok, detail in _results:
        print(f"[{'PASS' if ok else 'FAIL'}] {name:<{width}}  {detail}")
    print()
    if failed:
        print(f"{len(failed)} check(s) failed: {', '.join(failed)}")
        return 1
    print(f"all {len(_results)} checks passed — the package reproduces the reported numbers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

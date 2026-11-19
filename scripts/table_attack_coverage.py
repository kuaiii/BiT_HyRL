"""Regenerate the attack-coverage table body (Table 11) from the shipped merged table.

``attack_coverage.py`` rebuilds the merged table itself, but that path needs the
per-cell shard directory (``results/attack_cov*``) and the per-cell JSON records,
which are not part of this package.  The merged table *is* shipped, so this wrapper
re-uses the same formatting function and only skips the merging step.

Usage
-----
::

    python scripts/table_attack_coverage.py                 # -> paper_tables/tab_attack_coverage.tex
    python scripts/table_attack_coverage.py --out paper_tables
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from attack_coverage import coverage_table

ROOT = Path(__file__).resolve().parents[1]
MERGED = ROOT / "results" / "attack_cov" / "attack_coverage.csv"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="attack-coverage table from the merged CSV")
    parser.add_argument("--out", default=str(ROOT / "paper_tables"))
    args = parser.parse_args(argv)

    frame = pd.read_csv(MERGED)
    text = coverage_table(frame)

    target = Path(args.out) / "tab_attack_coverage.tex"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    print(f"[out] {target}  ({len(frame)} rows, {frame['policy'].nunique()} strategies, "
          f"{frame['attack'].nunique()} orders, {frame['seed'].nunique()} seeds)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

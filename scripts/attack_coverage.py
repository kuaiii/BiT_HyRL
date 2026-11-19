"""Merge the attack-coverage shards and emit the six-attack comparison table."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

COV = ROOT / "results" / "attack_cov"

#: the nine topologies of Table 5 — the suite this table is defined on
NINE = ("BA-100", "BA-200", "BA-300", "BA-400", "BA-500",
        "GtsCe", "UsCarrier", "Colt", "Cogentco")

ATTACKS = ("degree", "betweenness", "closeness", "coreness", "eigenvector", "random")
TARGETED = ATTACKS[:-1]

ORDER = ("BiT-HyRL", "UNITY", "QDLM", "FRED-ABL", "Eff-R-Net", "ONION", "ROMEN",
         "MWCL-sim", "MADQN-sim", "KMedian", "Random-placement")

EXCLUDED: dict[str, tuple[str, ...]] = {}


def _exclusion_note(present: Sequence[str]) -> str:
    """Caption sentence for every baseline deliberately left out of the table."""
    notes: list[str] = []
    for name, failed in EXCLUDED.items():
        if name in present or not failed:
            continue
        covered = len(NINE) - len(failed)
        listed = (failed[0] if len(failed) == 1
                  else ", ".join(failed[:-1]) + f" and {failed[-1]}")
        notes.append(
            f"{name} is not listed: it did not run to completion on {listed}, so it covers "
            f"only {covered} of the {len(NINE)} topologies and its row would average a "
            "different graph set than the rows below."
        )
    return " ".join(notes)


def _read(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    keep = ["dataset", "policy", "attack", "seed", "auc_gcc", "auc_csa", "auc_hc", "r_res"]
    missing = [c for c in keep if c not in frame.columns]
    if missing:
        raise SystemExit(f"{path} is missing columns {missing}")
    return frame[keep]


def _from_records(seeds: int, skip: Sequence[str] = ()) -> pd.DataFrame:
    """Read the per-evaluation JSON records for the four new attacks."""
    import json

    rows = []
    for path in (ROOT / "results" / "records").glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if payload.get("attack") not in ATTACKS[1:5]:
            continue
        if payload.get("seed", 0) >= seeds:
            continue
        if payload.get("policy") in skip:
            continue
        rows.append({
            "dataset": payload.get("dataset"),
            "policy": payload.get("policy"),
            "attack": payload.get("attack"),
            "seed": payload.get("seed"),
            "auc_gcc": payload.get("auc_gcc"),
            "auc_csa": payload.get("auc_csa"),
            "auc_hc": payload.get("auc_hc"),
            "r_res": payload.get("r_res"),
        })
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise SystemExit("no results for the new attacks in the record directory")
    return frame.dropna(subset=["r_res"])


def load(seeds: int = 10) -> pd.DataFrame:
    """Merge the four new attacks with the archived degree/random runs."""
    romen_path = ROOT / "results" / "baselines" / "romen" / "romen_9x6.csv"
    romen = pd.DataFrame()
    if romen_path.exists():
        romen = _read(romen_path)
        romen = romen[romen["seed"] < seeds]

    bit = pd.DataFrame()
    for fresh_bit, pattern in ((ROOT / "results" / "attack_cov_HyRL", "BiT-HyRL__*.csv"),
                               (ROOT / "results" / "attack_cov", "BiT-R__*.csv")):
        if not fresh_bit.exists():
            continue
        shards = sorted(fresh_bit.glob(pattern))
        if not shards:
            continue
        bit = pd.concat([_read(path) for path in shards], ignore_index=True)
        bit = bit[bit["seed"] < seeds].copy()
        bit["policy"] = "BiT-HyRL"
        break

    pieces = []
    archived = (ROOT / "results" / "baselines" / "method_matrix" / "matrix_scalar.csv",
                ROOT / "results" / "baselines" / "method_matrix" / "placement_new" / "matrix_scalar.csv")
    for path in archived:
        if path.exists():
            frame = _read(path)
            frame = frame[frame["attack"].isin(("degree", "random")) & (frame["seed"] < seeds)]
            if not romen.empty:
                frame = frame[frame["policy"] != "ROMEN"]
            if not bit.empty:
                frame = frame[frame["policy"] != "BiT-HyRL"]
            pieces.append(frame)
    if not romen.empty:
        pieces.append(romen)
    if not bit.empty:
        pieces.append(bit)
    pieces.append(_from_records(seeds, skip=("ROMEN",) if not romen.empty else ()))

    frame = pd.concat(pieces, ignore_index=True)
    frame = frame[frame["dataset"].isin(NINE) & frame["attack"].isin(ATTACKS)]
    frame = frame[frame["policy"].isin(ORDER)]
    duplicated = frame.duplicated(["dataset", "policy", "attack", "seed"]).sum()
    if duplicated:
        print(f"[warn] dropped {duplicated} duplicate records")
        frame = frame.drop_duplicates(["dataset", "policy", "attack", "seed"])
    return frame


def coverage_table(frame: pd.DataFrame) -> str:
    """Rows = strategy, columns = attack, entries = mean R_res over the nine graphs."""
    per_graph = frame.groupby(["policy", "attack", "dataset"])["r_res"].mean()
    by_attack = per_graph.groupby(level=["policy", "attack"]).mean().unstack()
    by_attack = by_attack.reindex(columns=list(ATTACKS))
    by_attack["targeted"] = by_attack[list(TARGETED)].mean(axis=1)

    present = [name for name in ORDER if name in by_attack.index]
    columns = list(ATTACKS) + ["targeted"]
    best = by_attack.loc[present, columns].max()

    lines: list[str] = []
    for name in present:
        row = by_attack.loc[name]
        cells = []
        for attack in ATTACKS:
            value = row[attack]
            text = f"{value:.3f}"
            if abs(value - best[attack]) < 5e-4:
                text = "\\textbf{" + text + "}"
            cells.append(text)
        targeted = f"{row['targeted']:.3f}"
        if abs(row["targeted"] - best["targeted"]) < 5e-4:
            targeted = "\\textbf{" + targeted + "}"
        lines.append(f"{name} & " + " & ".join(cells) + f" & {targeted} \\\\")
    body = "\n".join(lines)

    seeds = int(frame["seed"].nunique())
    absent = [n for n in ORDER if n not in by_attack.index]
    notes = [f"{n} is absent from the merged shards and could not be tabulated."
             for n in absent]
    exclusion = _exclusion_note(present)
    if exclusion:
        notes.append(exclusion)
    footnote = (" " + " ".join(notes)) if notes else ""

    head = ("\\begin{table*}[!ht]\n\\centering\n\\footnotesize\n"
            "\\setlength{\\tabcolsep}{6pt}\n"
            "\\caption{Attack coverage. Mean $R_{\\mathrm{res}}$ over the nine topologies of "
            f"Table~\\ref{{tab:comprehensive_comparison}}, {seeds} seeds per cell, under each of the five "
            "targeted orderings and the uniform random attack. The last column averages the five targeted "
            "columns and is the quantity the balanced protocol of Section~\\ref{s:Evaluation} would use if "
            "the targeted mode were defined on all five rather than on degree ordering alone. Best entry of "
            "each column in \\textbf{bold}." + footnote + "}\n"
            "\\label{tab:attack_coverage}\n"
            "\\begin{tabular*}{\\textwidth}{@{}l @{\\extracolsep{\\fill}} *{7}{c}@{}}\n\\toprule\n"
            "\\textbf{Strategy} & \\textbf{Degree} & \\textbf{Betweenness} & \\textbf{Closeness} & "
            "\\textbf{Coreness} & \\textbf{Eigenvector} & \\textbf{Random} & \\textbf{Targeted mean} \\\\\n"
            "\\midrule\n")
    body = body.replace("\nUNITY &", "\n\\midrule\nUNITY &", 1)
    tail = "\n\\bottomrule\n\\end{tabular*}\n\\end{table*}\n"
    return head + body + tail


def check_complete(frame: pd.DataFrame, seeds: int) -> pd.Series:
    """Compare each ``(policy, attack)`` row count with the expected ``9 x seeds``."""
    wanted = len(NINE) * seeds
    counts = frame.groupby(["policy", "attack"]).size()
    short = counts[counts != wanted]
    if len(short):
        print(f"[warn] these cells do not have {wanted} rows (9 graphs x {seeds} seeds):")
        print(short.to_string())
    else:
        print(f"[ok] every cell has {wanted} rows (9 graphs x {seeds} seeds)")
    return counts


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="attack-coverage merge and table")
    parser.add_argument("--out", default=None, help="paper_tables directory")
    parser.add_argument("--seeds", type=int, default=20, help="use the first k seeds")
    args = parser.parse_args(argv)

    frame = load(args.seeds)
    out_csv = COV / "attack_coverage.csv"
    frame.to_csv(out_csv, index=False)
    print(f"[out] {out_csv}  ({len(frame)} rows, {frame['policy'].nunique()} strategies, "
          f"{frame['attack'].nunique()} attacks, {frame['seed'].nunique()} seeds)")

    complete = frame.groupby(["policy", "attack"]).size().unstack()
    print("\nrows per cell (expected: 9 graphs x seeds):")
    print(complete.to_string())
    check_complete(frame, args.seeds)

    if args.out:
        target = Path(args.out) / "tab_attack_coverage.tex"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(coverage_table(frame), encoding="utf-8")
        print(f"\n[out] {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

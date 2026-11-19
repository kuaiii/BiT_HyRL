"""Merge the attack-coverage shards and emit the six-attack comparison table.

The main tables in the manuscript use the balanced two-mode protocol (targeted vs
random).  This script answers the reviewer question that protocol leaves open: does
the ranking survive if the targeted attack is something other than degree ordering?

It reads the four shard directories written by ``method-matrix`` plus the two ablation
shards, checks that every (strategy, attack, graph, seed) cell is present exactly once,
and writes

* ``results/attack_cov/attack_coverage.csv`` — long table, one row per cell;
* ``paper_tables/tab_attack_coverage.tex``      — the manuscript table.

Usage
-----
    python scripts/attack_coverage.py --out "../../02-第二次投稿 RESS/rebuttal/paper_tables"
"""

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

#: the proposed method plus every external baseline; the three ablation arms are
#: controls rather than competitors and stay in Table 5 on the balanced protocol
ORDER = ("BiT-HyRL", "UNITY", "QDLM", "FRED-ABL", "Eff-R-Net", "ONION", "ROMEN",
         "MWCL-sim", "MADQN-sim", "KMedian", "Random-placement")

#: baselines of Table 5 that cannot appear in this table, mapped to the topologies
#: they did not finish on.  Their cell means would average a smaller graph set than
#: every other row, so they are dropped from ``ORDER`` and the caption must say so
#: explicitly — an unexplained omission reads like a selection effect.
#:
#: ROMEN used to be listed here (it was never completed on BA-400/BA-500); the
#: seed-sharded re-run in ``results/baselines/romen/romen_9x6.csv`` covers all nine
#: topologies, so it now sits in ``ORDER`` and nothing has to be explained away.
EXCLUDED: dict[str, tuple[str, ...]] = {}


def _exclusion_note(present: Sequence[str]) -> str:
    """Caption sentence for every baseline deliberately left out of the table.

    Args:
        present: Strategy names that do appear in the table.

    Returns:
        LaTeX-safe prose (empty when nothing is excluded), listing each dropped
        baseline together with the topologies it failed to complete on.
    """
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
        raise SystemExit(f"{path} 缺少列 {missing}")
    return frame[keep]


def _from_records(seeds: int, skip: Sequence[str] = ()) -> pd.DataFrame:
    """Read the per-evaluation JSON records for the four new attacks.

    ``run_evaluation`` persists one file per cell, so a shard that is killed still
    leaves its finished cells behind; rebuilding from the records therefore recovers
    work that the shard CSVs (written once, at the end) would have lost.

    Args:
        seeds: Number of seeds to keep (rows with a larger seed are dropped).
        skip: Strategy names to ignore.  Used for ROMEN, whose six columns are read
            from one dedicated file instead, so that its row comes from a single run.
    """
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
        raise SystemExit("记录目录里没有找到新攻击的结果")
    return frame.dropna(subset=["r_res"])


def load(seeds: int = 10) -> pd.DataFrame:
    """Merge the four new attacks with the archived degree/random runs.

    The archived matrix already evaluated degree and random on exactly these nine
    graphs, twenty seeds and the same strategies, so reusing it keeps the six columns
    on one protocol and doubles as a consistency check.

    ROMEN is the exception: its archived run stopped at fifteen topologies, and the
    seed-sharded re-run in ``results/baselines/romen/romen_9x6.csv`` covers all nine graphs
    for all six attacks.  When that file is present it replaces the archived ROMEN
    rows entirely.  ROMEN is a torch genetic algorithm that does not guarantee
    bit-for-bit agreement across processes, so a row mixing archived and fresh cells
    would average two different constructions; ``merge_romen_sweep.py --compare``
    quantifies how far apart the two runs actually are.

    Only the eleven strategies of :data:`ORDER` are kept: the 2x2 ablation arms
    (``degree_ranked``/``ppo``) are evaluated on *both* the original and the
    reconstructed topology, so they hold two rows per ``(dataset, attack, seed)`` and
    would break the one-row-per-cell invariant this long table promises.  They belong
    to the ablation table, not here.
    """
    romen_path = ROOT / "results" / "baselines" / "romen" / "romen_9x6.csv"
    romen = pd.DataFrame()
    if romen_path.exists():
        romen = _read(romen_path)
        romen = romen[romen["seed"] < seeds]

    # second generation: the deployed bimodal row is rebuilt at the per-graph interval
    # rule, so its six columns must all come from the same run.  Mixing a fresh degree
    # column with archived centrality columns would put two constructions in one row.
    # the deployed row is BiT-HyRL (the geometric midpoint of the admissible interval),
    # so its shards live in ``attack_cov_HyRL``; the older ``attack_cov`` directory holds
    # the BiT-R run and is only a fallback for graphs the new sweep has not reached
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
        print(f"[warn] 丢弃 {duplicated} 条重复记录")
        frame = frame.drop_duplicates(["dataset", "policy", "attack", "seed"])
    return frame


def coverage_table(frame: pd.DataFrame) -> str:
    """Rows = strategy, columns = attack, entries = mean R_res over the nine graphs."""
    per_graph = frame.groupby(["policy", "attack", "dataset"])["r_res"].mean()
    by_attack = per_graph.groupby(level=["policy", "attack"]).mean().unstack()
    by_attack = by_attack.reindex(columns=list(ATTACKS))
    by_attack["targeted"] = by_attack[list(TARGETED)].mean(axis=1)

    present = [name for name in ORDER if name in by_attack.index]
    # the winner of each column, for bold marking; "targeted" is a column too, and the
    # caption promises the best entry of *each* column, so it must not be left plain
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
    # a rule between the proposed family and the baselines
    body = body.replace("\nUNITY &", "\n\\midrule\nUNITY &", 1)
    tail = "\n\\bottomrule\n\\end{tabular*}\n\\end{table*}\n"
    return head + body + tail


def check_complete(frame: pd.DataFrame, seeds: int) -> pd.Series:
    """Compare each ``(policy, attack)`` row count with the expected ``9 x seeds``.

    A short cell means a shard died in the middle of a graph (the shard CSV is only
    written at the very end, the per-cell records are not), so this check is the gate
    between a finished batch and a table that silently averages fewer graphs.

    Args:
        frame: The merged long table.
        seeds: Number of seeds the table is meant to cover.

    Returns:
        The per-cell row counts, indexed by ``(policy, attack)``.
    """
    wanted = len(NINE) * seeds
    counts = frame.groupby(["policy", "attack"]).size()
    short = counts[counts != wanted]
    if len(short):
        print(f"[warn] 以下格子不是 {wanted} 行（9 图 × {seeds} 种子）：")
        print(short.to_string())
    else:
        print(f"[ok] 每格均为 {wanted} 行（9 图 × {seeds} 种子）")
    return counts


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="attack-coverage merge and table")
    parser.add_argument("--out", default=None, help="paper_tables 目录")
    parser.add_argument("--seeds", type=int, default=20, help="使用前 k 个种子")
    args = parser.parse_args(argv)

    frame = load(args.seeds)
    out_csv = COV / "attack_coverage.csv"
    frame.to_csv(out_csv, index=False)
    print(f"[out] {out_csv}  ({len(frame)} rows, {frame['policy'].nunique()} strategies, "
          f"{frame['attack'].nunique()} attacks, {frame['seed'].nunique()} seeds)")

    complete = frame.groupby(["policy", "attack"]).size().unstack()
    print("\n每格行数（应为 9 图 × 种子数）：")
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

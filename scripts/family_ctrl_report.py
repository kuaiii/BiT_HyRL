"""Family table and paired statistics for the deployment experiment."""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from scipy.stats import wilcoxon

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from make_paper_tables import _paired_frame  # noqa: E402

CTRL = REPO_ROOT / "results" / "family" / "ctrl"
OUT = REPO_ROOT / "results" / "family"
TABLES = REPO_ROOT / "paper_tables"

FAMILIES = ("BA", "ER", "WS", "ISP", "AS")
RENAME = {"BiT-HyRL": r"BiT-HyRL", "BiT-A": r"BiT-HyRL",
          "BiT-R": "BiT-R", "BiT-E": "BiT-E",
          "MWCL-sim": "MWCL", "MADQN-sim": "MADQN",
          "Random-placement": "Random-pl."}
ORDER = (r"BiT-HyRL", "BiT-R", "BiT-E", "BiT-c1.0", "BiT-degOnly", "BiT-norecon",
         "UNITY", "QDLM", "FRED-ABL", "Eff-R-Net", "ONION", "ROMEN",
         "MWCL", "MADQN", "KMedian", "Random-pl.")
ABLATION = {"BiT-R", "BiT-E", "BiT-c1.0", "BiT-degOnly", "BiT-norecon"}
METRICS = (("$GCC$", "auc_gcc"), ("$CSA$", "auc_csa"), ("$H_C$", "auc_hc"),
           ("$R_{\\mathrm{res}}$", "r_res"))


def family_of(dataset: str) -> str:
    prefix = dataset.split("-")[0]
    return prefix if prefix in ("BA", "ER", "WS") else ("AS" if prefix == "AS" else "ISP")


def load() -> pd.DataFrame:
    archive = _paired_frame()
    archive["label"] = archive["policy"].map(lambda name: RENAME.get(name, name))
    archive["source"] = "archive"
    pieces = [archive]
    paths = sorted(glob.glob(str(CTRL / "*.csv"))) + sorted(
        glob.glob(str(CTRL.parent / "ctrl_romen" / "*.csv")))
    if paths:
        fresh = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
        fresh["label"] = fresh["policy"].map(lambda name: RENAME.get(name, name))
        fresh["source"] = "rerun"
        fresh_labels = set(fresh["label"]) - {"ROMEN"}
        pieces = [archive[~archive["label"].isin(fresh_labels)], fresh]
    frame = pd.concat(pieces, ignore_index=True)
    frame["family"] = frame["dataset"].map(family_of)
    return _merge_romen(frame)


def _merge_romen(frame: pd.DataFrame) -> pd.DataFrame:
    """Complete the ROMEN row graph by graph."""
    romen = frame[frame["label"] == "ROMEN"]
    if romen.empty:
        return frame
    fresh_set = _fresh_datasets()
    if not fresh_set:
        return frame
    stale = romen[romen["dataset"].isin(fresh_set) & romen["source"].eq("archive")]
    if not stale.empty:
        frame = frame.drop(index=stale.index)
    covered = frame.loc[frame["label"] == "ROMEN", "dataset"].nunique()
    print(f"[romen] {covered} topologies ({len(fresh_set)} from the completion sweep)",
          flush=True)
    return frame


def _fresh_datasets() -> set[str]:
    """Topologies present in the completion shards (``ctrl_romen``)."""
    fresh: set[str] = set()
    for path in glob.glob(str(CTRL.parent / "ctrl_romen" / "*.csv")):
        frame = pd.read_csv(path, usecols=["dataset"])
        fresh.update(frame["dataset"].astype(str).unique())
    return fresh


def table(frame: pd.DataFrame) -> pd.DataFrame:
    per_graph = (frame.groupby(["label", "dataset", "family"])
                 [["r_res", "auc_gcc", "auc_csa", "auc_hc"]].mean().reset_index())
    rows = []
    for label_name in ORDER:
        subset = per_graph[per_graph["label"] == label_name]
        if subset.empty:
            continue
        entry = {"strategy": label_name,
                 "graphs": subset["dataset"].nunique(),
                 "All": subset["r_res"].mean()}
        for family in FAMILIES:
            block = subset[subset["family"] == family]
            entry[family] = block["r_res"].mean() if not block.empty else np.nan
        for _, column in METRICS[:3]:
            entry[column] = subset[column].mean()
        rows.append(entry)
    return pd.DataFrame(rows)


def tex_body(frame: pd.DataFrame) -> str:
    ranked = [name for name in ORDER if name not in ABLATION]
    best = {column: frame[column].max() for column in ("All", "auc_gcc",
                                                       "auc_csa", "auc_hc")}
    second = {column: frame[column].nlargest(2).iloc[-1] for column in best}
    lines: list[str] = []
    for _, row in frame.iterrows():
        name = row["strategy"]
        cells = [name]
        for family in FAMILIES:
            value = row[family]
            cells.append("--" if pd.isna(value) else f"{value:.3f}")
        cells.append(f"{row['All']:.4f}")
        for column in ("auc_gcc", "auc_csa", "auc_hc"):
            text = f"{row[column]:.3f}"
            if name in ranked:
                if abs(row[column] - best[column]) < 5e-4:
                    text = r"\textbf{" + text + "}"
                elif abs(row[column] - second[column]) < 5e-4:
                    text = r"\underline{" + text + "}"
            cells.append(text)
        if name in ranked:
            if abs(row["All"] - best["All"]) < 5e-5:
                cells[6] = r"\textbf{" + cells[6] + "}"
            elif abs(row["All"] - second["All"]) < 5e-5:
                cells[6] = r"\underline{" + cells[6] + "}"
        lines.append(" & ".join(cells) + r" \\")
        if name == "BiT-E":
            lines.append(r"\midrule")
        if name == "BiT-norecon":
            lines.append(r"\midrule")
    return "\n".join(lines)


def paired(frame: pd.DataFrame) -> pd.DataFrame:
    """Exact paired signed-rank tests against the deployed row, Holm over the four metrics."""
    per_graph = (frame.groupby(["label", "dataset"])[["r_res", "auc_gcc", "auc_csa", "auc_hc"]]
                 .mean().reset_index())
    reference = per_graph[per_graph["label"] == r"BiT-HyRL"].set_index("dataset")
    rows = []
    for label_name in ORDER:
        if label_name == r"BiT-HyRL":
            continue
        other = per_graph[per_graph["label"] == label_name].set_index("dataset")
        shared = reference.index.intersection(other.index)
        if len(shared) < 6:
            continue
        entry = {"strategy": label_name, "n": len(shared)}
        pvalues = {}
        for column in ("r_res", "auc_gcc", "auc_csa", "auc_hc"):
            difference = reference.loc[shared, column] - other.loc[shared, column]
            entry[f"d_{column}"] = difference.mean()
            half = (student_t.ppf(0.975, len(difference) - 1)
                    * difference.std(ddof=1) / np.sqrt(len(difference)))
            entry[f"lo_{column}"] = difference.mean() - half
            entry[f"hi_{column}"] = difference.mean() + half
            try:
                pvalues[column] = wilcoxon(difference).pvalue
            except ValueError:
                pvalues[column] = 1.0
        order = sorted(pvalues, key=pvalues.get)
        adjusted, running = {}, 0.0
        for rank, column in enumerate(order):
            running = max(running, min(1.0, pvalues[column] * (len(order) - rank)))
            adjusted[column] = running
        entry.update({f"p_{column}": adjusted[column] for column in adjusted})
        rows.append(entry)
    return pd.DataFrame(rows)


def stars(value: float) -> str:
    return "^{***}" if value < 1e-3 else "^{**}" if value < 1e-2 else "^{*}" if value < 0.05 else ""


def main() -> int:
    frame = load()
    body = table(frame)
    OUT.mkdir(parents=True, exist_ok=True)
    body.to_csv(OUT / "family_ctrl.csv", index=False)

    TABLES.mkdir(exist_ok=True)
    (TABLES / "tab_family_comparison.tex").write_text(tex_body(body) + "\n", encoding="utf-8")

    stats = paired(frame)
    stats.to_csv(OUT / "family_paired.csv", index=False)
    lines = []
    for _, row in stats.iterrows():
        cells = [row["strategy"], str(int(row["n"]))]
        for column in ("r_res", "auc_gcc", "auc_csa", "auc_hc"):
            cells.append(f"${row['d_' + column]:+.3f}$")
            cells.append(f"${row['p_' + column]:.1e}{stars(row['p_' + column])}$")
        lines.append(" & ".join(cells) + r" \\")
    (TABLES / "tab_paired_vs_bit.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(body.round(4).to_string(index=False))
        print()
        print(stats.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

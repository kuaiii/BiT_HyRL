"""Emit the manuscript tables and figure data from the merged benchmark."""

from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "results" / "family"
TABLES = REPO_ROOT / "paper_tables"

DISPLAY = {
    "Original": "Original",
    "Random": "Random",
    "BA": "BA",
    "WS": "WS",
    "BiT-R": r"\textbf{BiT-R}",
    "BiT-HyRL": r"\textbf{BiT-HyRL}",
    "BiT-E": r"\textbf{BiT-E}",
    "UNITY": "UNITY",
    "FRED-ABL": "FRED-ABL",
    "QDLM": "QDLM",
    "ONION": "ONION",
    "Eff-R-Net": "Eff-R-Net",
    "ROMEN": "ROMEN",
}
ORDER = list(DISPLAY)
FAMILIES = ("BA", "ER", "WS", "AS", "ISP")
AREAS = ("R_gcc_tar", "R_gcc_ran", "R_e_tar", "R_e_ran")

ALIAS = {"BiT-A": "BiT-HyRL"}


def load(shards: Iterable[str]) -> pd.DataFrame:
    paths: list[str] = []
    for pattern in shards:
        paths.extend(sorted(glob.glob(pattern)))
    if not paths:
        raise SystemExit("no input CSV found (the shards are not shipped; point --input at family_bench.csv)")
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    if "status" in frame.columns:
        frame = frame[frame["status"].astype(str).str.startswith("ok")].copy()
    frame = frame.drop_duplicates(["dataset", "method", "seed"], keep="last")
    frame["method"] = frame["method"].replace(ALIAS)
    for column in ("A",) + AREAS + ("kappa", "construct_s", "kbar", "n"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["method"] = pd.Categorical(frame["method"], categories=ORDER, ordered=True)
    return frame.sort_values(["method", "dataset", "seed"]).reset_index(drop=True)


def aggregate(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Method means, and the same resolved by family."""
    per_cell = (frame.groupby(["dataset", "family", "method"], observed=True)
                .agg({**{column: "mean" for column in ("A",) + AREAS},
                      "kappa": "mean", "construct_s": "mean", "kbar": "mean", "n": "mean",
                      "seed": "count"})
                .reset_index())
    per_cell = per_cell.rename(columns={"seed": "n_seeds"})
    seed_sd = (frame.groupby(["dataset", "method"], observed=True)["A"].std()
               .rename("sd_seed").reset_index())
    per_cell = per_cell.merge(seed_sd, on=["dataset", "method"], how="left")

    methods = (per_cell.groupby("method", observed=True)
               .agg({"A": "mean", **{column: "mean" for column in AREAS},
                     "kappa": "mean", "construct_s": "mean", "kbar": "mean"})
               .reset_index())
    methods["sd_graph"] = (per_cell.groupby("method", observed=True)["A"].std().values)
    methods["sd_seed"] = (per_cell.groupby("method", observed=True)["sd_seed"].mean().values)
    methods["n_graphs"] = per_cell.groupby("method", observed=True)["dataset"].nunique().values
    methods = methods.sort_values("A", ascending=False)

    family = (per_cell.groupby(["method", "family"], observed=True)
              .agg({"A": "mean", "n": "nunique"}).reset_index()
              .rename(columns={"n": "n_graphs"}))
    wide = family.pivot(index="method", columns="family", values="A")
    wide = wide.reindex(columns=[name for name in FAMILIES if name in wide.columns])
    wide["All"] = methods.set_index("method")["A"]
    wide = wide.reindex([name for name in ORDER if name in wide.index])
    return methods, wide.reset_index()


def tex_construction(methods: pd.DataFrame, frame: pd.DataFrame) -> str:
    """Body of ``tab:construction_budget``: thirteen constructions on thirty graphs."""
    present = [name for name in ORDER if name in set(methods["method"])]
    best = {column: methods[column].max() for column in ("A",) + AREAS}
    second = {column: methods[column].nlargest(2).iloc[-1] for column in ("A",) + AREAS}
    lines: list[str] = []
    groups = [("Original", "Random", "BA", "WS"),
              ("UNITY", "FRED-ABL", "QDLM", "Eff-R-Net", "ONION", "ROMEN"),
              ("BiT-R", "BiT-HyRL", "BiT-E")]
    for group in groups:
        for name in group:
            if name not in present:
                continue
            row = methods[methods["method"] == name].iloc[0]
            cells = [DISPLAY[name]]
            cells.append(f"{row['kappa']:.2f}")
            for column in AREAS:
                value = row[column]
                text = f"{value:.3f}"
                if abs(value - best[column]) < 1e-9:
                    text = r"\textbf{" + text + "}"
                elif abs(value - second[column]) < 1e-9:
                    text = r"\underline{" + text + "}"
                cells.append(text)
            value = row["A"]
            text = f"{value:.3f}"
            if abs(value - best["A"]) < 1e-9:
                text = r"\textbf{" + text + "}"
            elif abs(value - second["A"]) < 1e-9:
                text = r"\underline{" + text + "}"
            cells.append(text)
            cells.append(f"{row['construct_s']:.2f}" if row["construct_s"] > 0.005
                         else r"$<$0.01")
            lines.append(" & ".join(cells) + r" \\")
        lines.append(r"\midrule")
    return "\n".join(lines[:-1])


def tex_family(family: pd.DataFrame, methods: pd.DataFrame, counts: dict[str, int]) -> str:
    """Body of ``tab:family_A``: BiT versus the best non-BiT constructor, per family."""
    deleted = [name for name in ORDER if name.startswith("BiT")]
    complete = set(methods.loc[methods["n_graphs"] == methods["n_graphs"].max(), "method"])
    baselines = [name for name in ORDER if name not in deleted and name in complete]
    lines: list[str] = []
    for name in [column for column in FAMILIES if column in family.columns]:
        column = family.set_index("method")[name]
        best_baseline = max((column.get(other, np.nan) for other in baselines
                             if other in column.index), default=np.nan)
        winner = max((other for other in baselines if other in column.index),
                     key=lambda other: column.get(other, -np.inf), default="--")
        bit = max(column.get(other, np.nan) for other in deleted if other in column.index)
        delta = (bit - best_baseline) / best_baseline * 100.0
        lines.append(
            f"{name:<4} ({counts.get(name, 0)}) & {bit:.3f} & {winner} {best_baseline:.3f} & "
            f"${'+' if delta >= 0 else '-'}{abs(delta):.1f}\\%$ \\\\")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="emit tables + figure data from the family benchmark")
    parser.add_argument("--input", nargs="+", default=[str(OUT / "family_bench.csv")],
                        help="merged export (default) or any CSV with the shard columns")
    parser.add_argument("--shards", nargs="+", default=None,
                        help="per-shard CSVs, used only if the merged export is rebuilt")
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    source = args.shards if args.shards else args.input
    frame = load(source)
    merged = out / "family_bench.csv"
    if {Path(p).resolve() for p in source} == {merged.resolve()}:
        print(f"[keep] {merged} left untouched (it is the input; label aliases apply in memory only)")
    else:
        frame.to_csv(merged, index=False)

    methods, family = aggregate(frame)
    methods.to_csv(out / "agg_method.csv", index=False)
    family.to_csv(out / "agg_family.csv", index=False)

    per_graph = (frame.groupby(["dataset", "family", "method"], observed=True)
                 .agg({"A": "mean", "kbar": "mean", "kappa": "mean"}).reset_index())
    per_graph.to_csv(out / "fig_points.csv", index=False)
    bars = (family.melt(id_vars="method", var_name="family", value_name="A")
            .dropna(subset=["A"]))
    bars.to_csv(out / "fig_family_A.csv", index=False)

    TABLES.mkdir(exist_ok=True)
    (TABLES / "tab_construction_budget.tex").write_text(
        tex_construction(methods, frame) + "\n", encoding="utf-8")
    (TABLES / "tab_family_A.tex").write_text(
        tex_family(family, methods,
                   frame.groupby("family")["dataset"].nunique().to_dict()) + "\n",
        encoding="utf-8")

    with pd.option_context("display.width", 160, "display.max_columns", 30):
        print(methods.round(4).to_string(index=False))
        print()
        print(family.round(4).to_string(index=False))
    print(f"\n[out] {out}/family_bench.csv  ({len(frame)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

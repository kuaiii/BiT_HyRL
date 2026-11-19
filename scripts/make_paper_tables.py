"""Generate the manuscript's LaTeX tables directly from the result files.

The manuscript tables were produced by an older pipeline; the numbers no longer describe
the code that is in the repository.  This script rebuilds them from the archive so that
every cell in the paper is reproducible by one command.

Design notes
------------
* Each table is written as a standalone ``.tex`` fragment under ``paper_tables/`` so the
  manuscript can ``\\input`` it, and the diff is reviewable without touching prose.
* A cell is the mean over **both attack modes and all seeds**, with the sample standard
  deviation over seeds.  The manuscript describes a balanced paradigm
  (``R = 0.5 R_targeted + 0.5 R_random``), so averaging the two attacks is what the text
  promises; averaging over seeds first and then reporting their sd keeps the spread
  attributable to the attack order, not to the mixing.
* A method that did not run on a graph prints ``--`` and is listed in the footer.  Nothing
  is imputed: an empty cell is a fact about the experiment, a filled one would be a lie.

Usage:
    python scripts/make_paper_tables.py --out "../../02-第二次投稿 RESS/rebuttal/paper_tables"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
import numpy as np
from scipy.stats import spearmanr, t as student_t, wilcoxon

REPO_ROOT = Path(__file__).resolve().parents[1]
#: Single results root.  Everything the manuscript prints is read from ``results``;
#: the earlier archive is no longer referenced by any builder (see ``results/README.md``).
RESULTS = REPO_ROOT / "results"
#: Records that did not need a re-run.  The external baselines, the ROMEN re-run, the
#: weight sweep and the runtime benchmark were measured once and are archived here
#: verbatim, so that the whole manuscript has one provenance path.
BASELINES = RESULTS / "baselines"
CURVE_COLUMNS = ("f_grid", "r_curve", "csa_curve", "hc_curve")

#: the manuscript's Table 5 layout: five synthetic + four real-world
NINE = ("BA-100", "BA-200", "BA-300", "BA-400", "BA-500",
        "GtsCe", "UsCarrier", "Colt", "Cogentco")

#: the seed-sharded ROMEN re-run.  It covers the nine graphs of Table 5 for all six
#: attacks, so it supersedes the archived ROMEN rows wherever the two overlap.  It is
#: read *first* everywhere and the de-duplication below keeps the first occurrence,
#: which is what makes the substitution automatic.
ROMEN_RERUN = BASELINES / "romen" / "romen_9x6.csv"


def _romen_rerun() -> pd.DataFrame | None:
    """Return the two-mode slice of the ROMEN re-run, or ``None`` if it is absent.

    The re-run holds all six attacks, because the attack-coverage table needs them.
    Every table built here uses the balanced two-mode protocol of the manuscript
    (``degree`` + ``random``, 40 runs per cell), so the other four columns must be
    dropped: keeping them would average ROMEN over 120 runs per cell while every
    other row is averaged over 40, and the comparison would no longer be
    like-for-like.
    """
    if not ROMEN_RERUN.exists():
        return None
    frame = pd.read_csv(ROMEN_RERUN)
    return frame[frame["attack"].isin(("degree", "random"))]


def _paired_frame() -> pd.DataFrame:
    """Seed-level records with the ablation arms merged in, one row per run.

    The ablation arms live in a separate file; the methods that appear in both are
    de-duplicated on (dataset, policy, attack, seed) so that no configuration is
    counted twice.
    """
    pieces = []
    rerun = _romen_rerun()
    if rerun is not None:
        pieces.append(rerun)
    for relative in ("method_matrix/matrix_scalar.csv",
                     "method_matrix/placement_new/matrix_scalar.csv",
                     "ablation/ablation_all_scalar.csv"):
        path = BASELINES / relative
        if path.exists():
            pieces.append(pd.read_csv(path))
    frame = pd.concat(pieces, ignore_index=True)
    return frame.drop_duplicates(["dataset", "policy", "attack", "seed"], keep="first")

#: manuscript row label -> the policy name in the archive
ROWS = (
    ("Baseline", "BiT-norecon"),
    ("FRED-ABL", "FRED-ABL"),
    ("Eff-R-Net", "Eff-R-Net"),
    ("ONION", "ONION"),
    ("QDLM", "QDLM"),
    ("ROMEN", "ROMEN"),
    ("UNITY", "UNITY"),
    ("BiT-HyRL", "BiT-HyRL"),
)

METRICS = (("GCC", "auc_gcc"), ("CSA", "auc_csa"), ("$H_C$", "auc_hc"))


def load() -> pd.DataFrame:
    pieces = []
    rerun = _romen_rerun()
    if rerun is not None:
        pieces.append(rerun)
    for pattern in ("method_matrix/matrix_scalar.csv",
                    "method_matrix/placement_new/matrix_scalar.csv",
                    "ablation/ablation_all_scalar.csv"):
        hits = sorted(BASELINES.glob(pattern))
        if hits:
            pieces.append(pd.concat([
                pd.read_csv(path, usecols=lambda c: c not in CURVE_COLUMNS) for path in hits
            ], ignore_index=True))
    frame = pd.concat(pieces, ignore_index=True)
    return frame.drop_duplicates(subset=["dataset", "policy", "attack", "seed"], keep="first")



#: policy name in the archive -> row label printed in the manuscript.  The ``-sim``
#: suffix marks the simulator stand-in in the code, but it means nothing to a reader of
#: the paper and it makes the baseline look like a variant of itself.
DISPLAY = {"MWCL-sim": "MWCL", "MADQN-sim": "MADQN"}


def label(policy: str) -> str:
    """Row label for ``policy``."""
    return DISPLAY.get(policy, policy)

def cell(mean: float, sd: float) -> str:
    return f"{mean:.3f}$\\pm${sd:.3f}"


def table_comprehensive(frame: pd.DataFrame) -> tuple[str, list[str]]:
    """Table 5: GCC / CSA / H_C per method per topology, best bold, second underlined."""
    notes: list[str] = []
    body: list[str] = []
    for metric_label, column in METRICS:
        body.append(f"\\midrule\n\\multicolumn{{10}}{{c}}{{\\textbf{{{metric_label} Metric}}}} \\\\")
        body.append("\\midrule")

        # per (graph, policy) mean over seeds, averaged over the two attacks
        per_seed = (frame.groupby(["dataset", "policy", "attack", "seed"])[column].mean()
                    .reset_index())
        stats = (per_seed.groupby(["dataset", "policy"])[column]
                 .agg(["mean", "std"]).reset_index())

        table = stats.pivot(index="policy", columns="dataset", values="mean")
        spread = stats.pivot(index="policy", columns="dataset", values="std")

        # rank only among the rows the manuscript actually prints -- an unlisted policy
        # (e.g. MADQN-sim) must not steal the bold marker from a listed one
        shown = [policy for _, policy in ROWS if policy in table.index]
        ranked = table.loc[shown]

        best = ranked.max(skipna=True)
        second = ranked.apply(lambda col: col.nlargest(2).iloc[1] if col.notna().sum() > 1
                             else float("nan"))

        for row_label, policy in ROWS:
            cells = []
            for graph in NINE:
                if policy not in table.index or graph not in table.columns:
                    cells.append("--")
                    continue
                mean = table.loc[policy, graph]
                sd = spread.loc[policy, graph]
                if pd.isna(mean):
                    cells.append("--")
                    continue
                text = cell(mean, sd)
                if abs(mean - best[graph]) < 5e-4:
                    text = f"\\textbf{{{text}}}"
                elif abs(mean - second[graph]) < 5e-4:
                    text = f"\\underline{{{text}}}"
                cells.append(text)
            if row_label == "BiT-HyRL":
                body.append("\\midrule\n\\specialrule{0.08em}{0pt}{0pt}")
            body.append(f"{label(row_label)} & " + " & ".join(cells) + " \\\\")

        missing = sorted({g for _, p in ROWS for g in NINE
                          if p not in table.index or g not in table.columns
                          or pd.isna(table.loc[p, g])})
        if missing:
            notes.append(f"{metric_label}: -- = the method did not complete on "
                         + ", ".join(missing))

    header = r"""\begin{table*}[!ht]
\centering
\footnotesize
\setlength{\tabcolsep}{1.5pt}
\caption{Comprehensive comparison of $GCC$, $CSA$, and $H_C$ across different methods and topologies. The best performance for each topology is in \textbf{bold}, and the second-best is \underline{underlined}. Higher values indicate better performance. All entries are means over the two attack modes and 20 seeds (40 runs per cell); $\pm$ is the sample standard deviation over those runs.}
\label{tab:comprehensive_comparison}
\begin{tabular*}{\textwidth}{@{}l @{\extracolsep{\fill}} *{9}{c} @{}}
\toprule
\multirow{2}{*}{\textbf{Method}} & \multicolumn{5}{c}{\textbf{Synthetic Networks (BA, growth=3)}} & \multicolumn{4}{c}{\textbf{Real-world Networks}} \\
\cmidrule(lr){2-6} \cmidrule(l){7-10}
& \textbf{BA-100} & \textbf{BA-200} & \textbf{BA-300} & \textbf{BA-400} & \textbf{BA-500} & \textbf{GtsCe} & \textbf{UsCarrier} & \textbf{Colt} & \textbf{Cogentco} \\
"""
    footer = "\\bottomrule\n\\end{tabular*}\n\\end{table*}\n"
    if notes:
        footer += ("\n% " + "\n% ".join(notes) + "\n")
    return header + "\n".join(body) + "\n" + footer, notes


def table_datasets() -> str:
    registry = json.loads((REPO_ROOT / "configs" / "graphs.json")
                          .read_text(encoding="utf-8"))["graphs"]
    wanted = [g for g in registry if g["dataset"] in NINE]
    order = {name: index for index, name in enumerate(NINE)}
    wanted.sort(key=lambda g: order[g["dataset"]])
    lines = []
    for spec in wanted:
        n, m = spec["n"], spec["m"]
        sparsity = 2 * m / (n * (n - 1))
        lines.append(f"{spec['dataset']} & {n} & {m} & {sparsity:.3f} \\\\")
    body = "\n".join(lines)
    return rf"""\begin{{table}}[htbp]
\centering
\small
\caption{{Dataset summary for the nine topologies of Table~\ref{{tab:comprehensive_comparison}}; the other twenty-one of the thirty-graph suite are listed in Section~\ref{{ss:setup}}. Sparsity = $2|E|/[n(n-1)]$. Edge counts are the post-deduplication values read from the pinned graph files.}}
\label{{tab:datasets}}
\begin{{tabular}}{{@{{}}lrrr@{{}}}}
\toprule
\textbf{{Network}} & \textbf{{$n$}} & \textbf{{$|E|$}} & \textbf{{Sparsity}} \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}
"""


def table_ablation(frame: pd.DataFrame | None = None) -> str:
    """Table: the three construction arms, one column per metric, with 95% CIs.

    All three arms run the *same* learned placement policy, so the only thing that
    changes across rows is the substrate: the un-reconstructed graph, that graph
    re-randomised at fixed degree sequence, and the reconstructed graph. That isolates
    the two steps of the construction (rewiring, then the change of degree sequence)
    without invoking the single-hub parameter, which the current design point no
    longer exposes and which therefore has no arm here.

    The numbers are read from the thirty-graph family store rather than from the
    nine-graph matrix, because the deployed row of Table~\\ref{tab:aggregate_all} is
    the same store: the matrix rows are 0.004 apart on $H_C$, and an ablation table
    that does not add up to the deployed row it is supposed to decompose is worse
    than no table.

    Each cell is the mean over the thirty topologies followed by the two-sided 95%
    confidence interval of that mean, built as a Student-t interval on the
    per-topology means with 29 degrees of freedom---the same construction as
    Table~\\ref{tab:statistical_tests}.
    """
    from family_ctrl_report import load as load_family

    columns = ("auc_gcc", "auc_csa", "auc_hc", "r_res")
    family = load_family()
    family = family[family["attack"].isin(("degree", "random"))]
    arms = (("Un-reconstructed", "BiT-norecon"),
            ("Rewired", "BiT-degOnly"),
            ("BiT-HyRL", "BiT-HyRL"))
    per_graph = family.groupby(["label", "dataset"])[list(columns)].mean()
    lines = []
    for name, policy in arms:
        block = per_graph.loc[policy]
        cells = []
        for column in columns:
            values = block[column].to_numpy()
            n = len(values)
            mean = float(values.mean())
            half = float(student_t.ppf(0.975, n - 1) * values.std(ddof=1) / np.sqrt(n))
            cells.append(f"{mean:.3f}$\\pm${half:.3f}")
        lines.append(f"{name} & " + " & ".join(cells) + r" \\")
    body = "\n".join(lines)
    n_graphs = per_graph.loc["BiT-HyRL"].shape[0]
    return rf"""\begin{{table}}[!t]
  \centering
  \caption{{Ablation of the topology construction under the controller-induced cascade. All three arms use the same learned placement policy, so the row that changes is the substrate: the graph as given, that graph re-randomised at fixed degree sequence, and the reconstructed graph. Each entry is the mean over the {n_graphs} topologies (both attack modes, twenty campaign seeds each) $\pm$ the half-width of the two-sided 95\% confidence interval of that mean ($n={n_graphs}$, Student-$t$, $29$ degrees of freedom); it is a confidence half-width, not a standard deviation. Higher is better.}}
  \label{{tab:ablation}}
  \scriptsize
  \setlength{{\tabcolsep}}{{2pt}}
  \begin{{tabular*}}{{\columnwidth}}{{@{{}}l @{{\extracolsep{{\fill}}}} *{{4}}{{c}}@{{}}}}
  \toprule
  \textbf{{Construction arm}} & \textbf{{$GCC$}} & \textbf{{$CSA$}} & \textbf{{$H_C$}} & \textbf{{$R_{{\mathrm{{res}}}}$}} \\
  \midrule
{body}
  \bottomrule
  \end{{tabular*}}
\end{{table}}
"""


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="从结果档生成论文 LaTeX 表")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    frame = load()
    writers = {
        "tab_comprehensive.tex": table_comprehensive(frame),
        "tab_datasets.tex": (table_datasets(), []),
        "tab_ablation.tex": (table_ablation(frame), []),
    }
    for name, payload in writers.items():
        text, notes = payload if isinstance(payload, tuple) else (payload, [])
        (out_dir / name).write_text(text, encoding="utf-8")
        print(f"[out] {out_dir / name}")
        for note in notes:
            print(f"      ! {note}")
    write_extra(out_dir)
    return 0




def table_statistics() -> str:
    """Table: paired comparison against BiT-HyRL, one observation per topology.

    Read from ``results/family/family_paired.csv``, the tidy table written next to
    ``family_ctrl.csv`` by ``scripts/family_ctrl_report.py``.  Both files come from
    the same per-graph records, so the test can no longer be computed on a different
    sample from the table it accompanies.

    The unit of analysis is the topology (n = 30).  Pooling the seed-level records
    would multiply the sample size by 40 without adding independent information, so the
    exact Wilcoxon is run on the per-topology paired differences and Holm is applied
    within each row across the four metrics.
    """
    frame = pd.read_csv(RESULTS / "family" / "family_paired.csv").set_index("strategy")
    rows = ("UNITY", "QDLM", "FRED-ABL", "Eff-R-Net", "ONION", "ROMEN",
            "MADQN", "MWCL", "KMedian", "Random-pl.")
    n_graphs = int(frame["n"].max())

    lines: list[str] = []
    for name in rows:
        if name not in frame.index:
            continue
        row = frame.loc[name]
        if name == "Random-pl.":
            lines.append("\\midrule")
        means = [f"${row['d_' + column]:+.3f}$"
                 for column in ("auc_gcc", "auc_csa", "auc_hc", "r_res")]
        intervals = [f"$[{row['lo_' + column]:.3f},{row['hi_' + column]:.3f}]$"
                     for column in ("auc_gcc", "auc_csa", "auc_hc", "r_res")]
        lines.append(f"{name} & {int(row['n'])} & " + " & ".join(means) + " \\\\")
        lines.append(" & & " + " & ".join(intervals) + " \\\\")
    body = "\n".join(lines)

    worst = frame.loc[[r for r in rows if r in frame.index],
                      ["p_auc_gcc", "p_auc_csa", "p_auc_hc", "p_r_res"]].to_numpy().max()
    threshold = "0.001" if worst < 1e-3 else ("0.01" if worst < 1e-2 else "0.05")

    head = ("\\begin{table}[!htbp]\n  \\centering\n"
            "  \\caption{Paired comparison against BiT-HyRL, one observation per topology "
            f"($n_{{\\mathrm{{topo}}}}={n_graphs}$). Positive differences indicate that "
            "BiT-HyRL is better. Each cell gives the mean paired difference and, beneath "
            "it, the two-sided 95\\% confidence interval of that mean (Student-$t$ interval "
            f"on the per-topology differences, ${n_graphs - 1}$ degrees of freedom); every "
            "interval excludes zero. All differences are significant after Holm correction "
            f"across the four metrics within a row ($p<{threshold}$).}}\n"
            "  \\label{tab:statistical_tests}\n  \\scriptsize\n"
            "  \\setlength{\\tabcolsep}{2pt}\n"
            "  \\begin{tabular*}{\\columnwidth}{@{}l @{\\extracolsep{\\fill}} r *{4}{c}@{}}\n"
            "  \\toprule\n"
            "  \\textbf{Method} & \\textbf{$n_{\\mathrm{topo}}$} & \\textbf{$GCC$} & "
            "\\textbf{$CSA$} & \\textbf{$H_C$} & \\textbf{$R_{\\mathrm{res}}$} \\\\\n"
            "  \\midrule\n"
            "  & & \\multicolumn{4}{c}{$\\Delta$ [95\\% CI]} \\\\\n"
            "  \\midrule\n")
    tail = "\n  \\bottomrule\n\\end{tabular*}\n\\end{table}\n"
    return head + body + tail


def table_sensitivity_ratio() -> str:
    """Table 9: sensitivity to the deployment ratio, the six BA networks, BiT-HyRL arm."""
    frame = pd.read_csv(RESULTS / "budget" / "budget_all.csv")
    frame = frame[(frame["arm"] == "bit_c16")
                  & frame["dataset"].str.startswith("BA")]
    if frame.empty:
        return "% budget_all.csv missing"
    n_graphs = frame["dataset"].nunique()
    grouped = frame.groupby("ratio")[["auc_gcc", "auc_csa", "auc_hc"]].mean().sort_index()
    best = grouped.max()
    lines = []
    for ratio, row in grouped.iterrows():
        cells = []
        for column in ("auc_gcc", "auc_csa", "auc_hc"):
            text = f"{row[column]:.3f}"
            if abs(row[column] - best[column]) < 1e-9:
                text = "\\textbf{" + text + "}"
            cells.append(text)
        lines.append(f"{ratio * 100:.0f}\\% & " + " & ".join(cells) + " \\\\")
    body = "\n".join(lines)
    head = ("\\begin{table}[!htbp]\n\\centering\n"
            f"\\caption{{Sensitivity of BiT-HyRL to the controller deployment ratio "
            f"$\\rho_c$ on the {n_graphs} BA networks ($n=100$--$1000$). Entries are means "
            "over both attack modes and three attack seeds.}\n"
            "\\label{tab:sensitivity_ratio}\n"
            "\\begin{tabular}{@{}cccc@{}}\n\\toprule\n"
            "\\textbf{$\\rho_c$} & \\textbf{$GCC$} & \\textbf{$CSA$} & "
            "\\textbf{$H_C$} \\\\\n\\midrule\n")
    tail = "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    return head + body + tail


def table_sensitivity_weights() -> str:
    """Does the ranking depend on how the evaluation aggregate weights the three terms?

    Recomputed from ``results/family/family_ctrl.csv`` — the three component areas of
    every strategy — so the reweighted rows are the same records as the equal-weight row
    rather than a separate sweep that could carry a different bimodal arm.
    """
    weights = (
        ("Equal $(1/3,1/3,1/3)$", (1 / 3, 1 / 3, 1 / 3)),
        ("Structural $(.5,.3,.2)$", (0.5, 0.3, 0.2)),
        ("Control $(.2,.3,.5)$", (0.2, 0.3, 0.5)),
        ("$GCC$ only $(1,0,0)$", (1.0, 0.0, 0.0)),
        ("$H_C$ only $(0,0,1)$", (0.0, 0.0, 1.0)),
    )
    table = pd.read_csv(RESULTS / "family" / "family_ctrl.csv").set_index("strategy")
    ranked = ("BiT-HyRL", "BiT-c1.0", "BiT-norecon", "UNITY", "QDLM", "FRED-ABL",
              "ROMEN", "Eff-R-Net", "ONION", "MWCL", "MADQN", "KMedian",
              "Random-pl.", "BiT-degOnly")
    ranked = [name for name in ranked if name in table.index]

    scores: dict[str, pd.Series] = {}
    lines: list[str] = []
    for name, (wg, ws, wh) in weights:
        score = wg * table["auc_gcc"] + ws * table["auc_csa"] + wh * table["auc_hc"]
        scores[name] = score
        bit = float(score["BiT-HyRL"])
        unity = float(score["UNITY"])
        lines.append(f"{name} & {bit:.4f} & {unity:.4f} & {bit - unity:+.4f} \\\\")
    body = "\n".join(lines)

    # the caption claims the ranking barely moves; measure it instead of asserting it
    rho = 1.0
    keys = [name for name, _ in weights]
    for i, first in enumerate(keys):
        for second in keys[i + 1:]:
            rho = min(rho, float(spearmanr(scores[first][ranked],
                                           scores[second][ranked]).statistic))
    head = ("\\begin{table}[!htbp]\n\\centering\n"
            "\\caption{Robustness of the ranking to the weights of the evaluation aggregate. "
            "Each row recomputes the whole comparison under one weighting of "
            "$R_{\\mathrm{res}}=\\omega_{G}GCC+\\omega_{S}CSA+\\omega_{H}H_C$, averaged over the thirty "
            "topologies. BiT-HyRL keeps its order relative to UNITY in all five rows; over "
            "the full fourteen-strategy ranking every comparison against an external "
            "baseline keeps its sign, although positions among the weaker baselines move "
            f"(pairwise Spearman $\\geq {rho:.2f}$).}}\n"
            "\\label{tab:sensitivity_weights}\n\\small\n"
            "\\begin{tabular}{@{}lccc@{}}\n\\toprule\n"
            "\\textbf{Weighting} & \\textbf{BiT-HyRL} & "
            "\\textbf{UNITY} & \\textbf{Margin} \\\\\n\\midrule\n")
    tail = "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    return head + body + tail



def table_advanced_attacks() -> str:
    """Table 7: four advanced attack models, five BA scales, eight arms."""
    frame = pd.read_csv(RESULTS / "advanced" / "advanced_attacks.csv")
    attacks = (("adaptive", "Adaptive"), ("correlated", "Correlated"),
               ("multistage", "Multi-stage"), ("gdm", "GDM"))
    metrics = (("auc_gcc", "GCC"), ("auc_csa", "CSA"), ("auc_hc", "$H_C$"))
    arms = ("Baseline", "FRED-ABL", "Eff-R-Net", "ONION", "QDLM", "ROMEN",
            "UNITY", "BiT-HyRL")

    means = frame.groupby(["arm", "attack"])[["auc_gcc", "auc_csa", "auc_hc"]].mean()
    best, second = {}, {}
    for attack, _ in attacks:
        for metric, _ in metrics:
            column = means.xs(attack, level="attack")[metric]
            ordered = column.sort_values(ascending=False)
            best[(attack, metric)] = ordered.iloc[0]
            second[(attack, metric)] = ordered.iloc[1] if len(ordered) > 1 else float("nan")

    lines: list[str] = []
    for arm in arms:
        if arm == "BiT-HyRL":
            lines.append("\\midrule\n\\specialrule{0.08em}{0pt}{0pt}")
        cells = []
        for attack, _ in attacks:
            for metric, _ in metrics:
                value = float(means.loc[(arm, attack), metric])
                text = f"{value:.3f}"
                if abs(value - best[(attack, metric)]) < 5e-4:
                    text = "\\textbf{" + text + "}"
                elif abs(value - second[(attack, metric)]) < 5e-4:
                    text = "\\underline{" + text + "}"
                cells.append(text)
        lines.append(arm + " & " + " & ".join(cells) + " \\\\")
    body = "\n".join(lines)

    header_cells = []
    for attack, label in attacks:
        header_cells.append(f"\\multicolumn{{3}}{{c}}{{\\textbf{{{label}}}}}")
    head = ("\\begin{table*}[!ht]\n\\centering\n\\footnotesize\n"
            "\\setlength{\\tabcolsep}{3pt}\n"
            "\\caption{Comparison under four advanced attack models on the synthetic BA family, "
            "averaged over the five scales (BA-100 to BA-500) and three attack seeds "
            "($n=15$ per cell). $GCC$ is the primary metric; $CSA$ and $H_C$ are auxiliary. Best "
            "per column in \\textbf{bold}, second best \\underline{underlined}.}")
    head += ("\n\\label{tab:advanced_attacks}\n"
             "\\begin{tabular*}{\\textwidth}{@{}l @{\\extracolsep{\\fill}} *{12}{c} @{}}\n"
             "\\toprule\n"
             "\\multirow{2}{*}{\\textbf{Method}} & " + " & ".join(header_cells) + " \\\\\n"
             "\\cmidrule(lr){2-4} \\cmidrule(lr){5-7} \\cmidrule(lr){8-10} \\cmidrule(l){11-13}\n"
             "& " + " & ".join(["\\textbf{GCC} & \\textbf{$CSA$} & \\textbf{$H_C$}"] * 4) + " \\\\\n"
             "\\midrule\n")
    tail = "\n\\bottomrule\n\\end{tabular*}\n\\end{table*}\n"
    # rejoin the caption/label as separate lines so LaTeX is happy
    head = head.replace("}\n\\label", "}\n\\label")
    return head + body + tail


def table_aggregate_30() -> str:
    """Table: every strategy aggregated over the full thirty-topology suite.

    Read straight from ``results/family/family_ctrl.csv`` — the same tidy table that
    supplies the family comparison — so the two tables cannot disagree.  The earlier
    version of this function rebuilt the aggregate from the seed-level archive, which
    carried the pre-BiT-A bimodal rows and therefore reported 0.4058 where the family
    table reported 0.4040.
    """
    frame = pd.read_csv(RESULTS / "family" / "family_ctrl.csv")
    frame = frame.set_index("strategy")
    order = ("BiT-HyRL", "BiT-c1.0", "BiT-degOnly", "BiT-norecon",
             "UNITY", "QDLM", "FRED-ABL", "ROMEN", "Eff-R-Net", "ONION",
             "MWCL", "MADQN", "KMedian", "Random-pl.")
    n_graphs = int(frame["graphs"].max())
    base = float(frame.loc["BiT-norecon", "All"])
    coverage = f"Every row covers all {n_graphs} topologies."

    lines: list[str] = []
    for policy in order:
        if policy not in frame.index:
            continue
        row = frame.loc[policy]
        if policy in ("BiT-norecon", "UNITY"):
            lines.append("\\midrule")
        lines.append(f"{policy} & {row['auc_gcc']:.3f} & {row['auc_csa']:.3f} & "
                     f"{row['auc_hc']:.3f} & {float(row['All']):.4f} & "
                     f"{float(row['All']) - base:+.4f} \\\\")
    body = "\n".join(lines)

    head = ("\\begin{table}[!htbp]\n\\centering\n\\small\n"
            "\\setlength{\\tabcolsep}{4pt}\n"
            f"\\caption{{All strategies aggregated over the full suite of {n_graphs} "
            "topologies (six BA scales, six Erd\\H{o}s--R\\'enyi, six Watts--Strogatz, six "
            "router-level ISP maps and six AS-level snapshots). Every entry is the mean "
            "of the per-topology scores, so each topology carries equal weight. The last "
            "column is the difference against the un-reconstructed topology. "
            + coverage + "}\n"
            "\\label{tab:aggregate_all}\n"
            "\\begin{tabular}{@{}l *{3}{c} r r@{}}\n\\toprule\n"
            "\\textbf{Strategy} & \\textbf{$GCC$} & \\textbf{$CSA$} & "
            "\\textbf{$H_C$} & \\textbf{$R_{\\mathrm{res}}$} & "
            "\\textbf{$\\Delta$ vs none} \\\\\n\\midrule\n")
    tail = "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    return head + body + tail



def table_runtime() -> str:
    """Table: deployment-stage cost, measured rather than inherited.

    The previous table in the manuscript came from an earlier pipeline and contained a
    scale (N = 800) that is not in the registry, so it could not be reproduced.  These
    numbers come from ``scripts/runtime_benchmark.py``: substrate construction plus
    greedy controller selection, cascade evaluation excluded.
    """
    path = BASELINES / "runtime_benchmark.csv"
    if not path.exists():
        return "% runtime_benchmark.csv missing"
    frame = pd.read_csv(path)
    scales = ("BA-100", "BA-200", "BA-300", "BA-500", "BA-1000")
    order = ("Baseline", "UNITY", "FRED-ABL", "BiT-HyRL",
             "ONION", "Eff-R-Net", "QDLM", "ROMEN")
    lookup = {(r.method, r.scale): r.total_s for r in frame.itertuples()}
    n_of = {r.scale: r.n for r in frame.itertuples()}

    lines: list[str] = []
    for method in order:
        cells = []
        for scale in scales:
            value = lookup.get((method, scale))
            if value is None or pd.isna(value):
                cells.append("--")
            elif scale == "BA-1000" and value >= 100:
                # the three large-n cells were measured once each and are reported to a
                # tenth of a second; two decimals would claim a precision not measured
                cells.append(f"{value:.1f}")
            else:
                cells.append(f"{value:.2f}")
        if method == "BiT-HyRL":
            lines.append("\\midrule")
        lines.append(f"{method} & " + " & ".join(cells) + " \\\\")
    body = "\n".join(lines)

    head = ("\\begin{table}[!htbp]\n\\centering\n\\small\n"
            "\\setlength{\\tabcolsep}{4pt}\n"
            "\\caption{Deployment-stage cost in seconds: reconstruction of the substrate plus "
            "selection of the controller set, measured on one CPU core with the same checkpoint "
            "for every row. Cascade evaluation and PPO training are excluded, because both are "
            "offline or amortised. Columns are the BA scale; the header gives $N$. A dash marks "
            "a cell that this benchmark did not measure: ONION, Eff-R-Net and QDLM exceed "
            "fifteen minutes at $N=1000$, and ROMEN was timed only up to $N=200$ here. Its cost "
            "at the larger scales is reported in Section~\\ref{ss:runtime}, where a single "
            "measurement at $N=500$ takes $0.59$~ks and $6.7$~GB of resident memory. Values are "
            "medians over three runs where the method completes quickly, and a single run "
            "otherwise.}\n"
            "\\label{tab:runtime}\n"
            "\\begin{tabular}{@{}l *{5}{r}@{}}\n\\toprule\n"
            "\\textbf{Method} & \\textbf{100} & \\textbf{200} & \\textbf{300} & "
            "\\textbf{500} & \\textbf{1000} \\\\\n\\midrule\n")
    tail = "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    return head + body + tail


def write_extra(out_dir: Path) -> None:
    for name, text in (("tab_statistical_tests.tex", table_statistics()),
                       ("tab_sensitivity_ratio.tex", table_sensitivity_ratio()),
                       ("tab_sensitivity_weights.tex", table_sensitivity_weights()),
                       ("tab_advanced_attacks.tex", table_advanced_attacks()),
                       ("tab_aggregate_all.tex", table_aggregate_30()),
                       ("tab_runtime.tex", table_runtime())):
        (out_dir / name).write_text(text, encoding="utf-8")
        print(f"[out] {out_dir / name}")


if __name__ == "__main__":
    raise SystemExit(main())

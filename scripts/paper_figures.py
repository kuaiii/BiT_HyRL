"""Build the data figures from the archived results."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "results"
ARCHIVE = RESULTS / "baselines" / "curve_archive.csv"
CURVE_DIR = RESULTS / "baselines" / "curves"
CTRL_DIR = RESULTS / "family" / "ctrl"

#: Columns every panel needs; the archive carries them as JSON-encoded point lists.
CURVE_COLUMNS = ["dataset", "policy", "attack", "seed",
                 "f_grid", "r_curve", "csa_curve", "hc_curve",
                 "auc_gcc", "auc_csa", "auc_hc", "r_res"]

#: manuscript row label -> archive policy name (same mapping as the tables)
POLICIES = (
    ("Baseline", "BiT-norecon"),
    ("FRED-ABL", "FRED-ABL"),
    ("Eff-R-Net", "Eff-R-Net"),
    ("ONION", "ONION"),
    ("QDLM", "QDLM"),
    ("UNITY", "UNITY"),
    ("BiT-HyRL", "BiT-HyRL"),
)
COLOURS = {
    "Baseline": "#527A9B",
    "FRED-ABL": "#EE929D",
    "Eff-R-Net": "#E6B65C",
    "ONION": "#82AA98",
    "QDLM": "#A99AC8",
    "UNITY": "#D47F65",
    "BiT-HyRL": "#16BCCD",
}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8.5,
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


def bold_ticks(axis: "plt.Axes") -> None:
    """Bold tick labels; the journal sets them semibold to match the bold axis titles."""
    for tick in axis.get_xticklabels() + axis.get_yticklabels():
        tick.set_fontweight("bold")


def load_curves() -> pd.DataFrame:
    """Curves for every policy the two figures plot."""
    if ARCHIVE.exists():
        frame = pd.read_csv(ARCHIVE)
    else:
        parts = [pd.read_csv(path) for path in sorted(CURVE_DIR.glob("curves_*.csv"))] \
            if CURVE_DIR.exists() else []
        for path in sorted(CTRL_DIR.glob("BiT-HyRL__*.csv")):
            parts.append(pd.read_csv(path))
        if not parts:
            raise SystemExit(
                f"curve data missing: {ARCHIVE}\n"
                "run python scripts/export_curves.py first, then copy "
                "results/baselines/curves/ back."
            )
        frame = pd.concat(parts, ignore_index=True)

    for column in CURVE_COLUMNS:
        if column not in frame.columns:
            frame[column] = None
    frame = frame[CURVE_COLUMNS].drop_duplicates(
        subset=["dataset", "policy", "attack", "seed"], keep="last")

    missing = [label for label, policy in POLICIES
               if policy not in set(frame["policy"].astype(str))]
    if missing:
        print(f"[warn] these curves are not in the archive yet, the figure will miss them: {', '.join(missing)}")
    return frame


def load_ablation(frame: pd.DataFrame) -> pd.DataFrame:
    """``Baseline`` lives in the archive under its ablation label; keep the old call shape."""
    return frame[frame["policy"] == "BiT-norecon"]


def parse_curve(text: str) -> np.ndarray:
    return np.fromstring(str(text).strip("[]"), sep=",")


def mean_curve(frame: pd.DataFrame, dataset: str, policy: str, attack: str,
               column: str) -> tuple[np.ndarray, np.ndarray]:
    subset = frame[(frame["dataset"] == dataset) & (frame["policy"] == policy)
                   & (frame["attack"] == attack)]
    if subset.empty:
        return np.array([]), np.array([])
    grid = parse_curve(subset["f_grid"].iloc[0])
    curves = np.vstack([parse_curve(value) for value in subset[column]])
    return grid, curves.mean(axis=0)


MATH_STYLE = {"mathtext.default": "bfit", "mathtext.fontset": "stix"}

#: Subplot box of the comparison block, shared by the canvas-height sum and subplots_adjust.
MARGINS = {"left": 0.075, "right": 0.992, "bottom": 0.125, "top": 0.962}
BASE_HEIGHT_IN = 4.85
PANEL_SHRINK_MM = 6.0
HSPACE, WSPACE = 0.33, 0.16
TREND_HEIGHT_IN = 3.78
TREND_TITLE_FS, TREND_AXIS_FS, TREND_TICK_FS = 8.4, 7.8, 6.8


def figure_comparison(frame: pd.DataFrame, out: Path, bar_style: bool = False,
                      stem: str = "Comparison_topology") -> None:
    """Four panels on BA-100 (legacy comparison figure); see :func:`_comparison_panels`."""
    with plt.rc_context(MATH_STYLE):
        _comparison_panels(frame, out, bar_style, stem)


def _comparison_panels(frame: pd.DataFrame, out: Path, bar_style: bool,
                       stem: str) -> None:
    """Replaces Comparison_topology.pdf (four panels on BA-100)."""
    rows = 2
    usable = MARGINS["top"] - MARGINS["bottom"]
    height = BASE_HEIGHT_IN - (rows + HSPACE) * (PANEL_SHRINK_MM / 25.4) / usable
    figure, axes = plt.subplots(2, 2, figsize=(7.0, height))

    subset = frame[frame["dataset"] == "BA-100"]
    per_seed = subset.groupby(["policy", "seed"])["r_res"].mean().reset_index()
    order = [policy for _, policy in POLICIES if policy in set(per_seed["policy"])]
    labels = [label for label, policy in POLICIES if policy in order]
    data = [per_seed.loc[per_seed["policy"] == policy, "r_res"].to_numpy() for policy in order]
    means = {label: float(values.mean()) for label, values in zip(labels, data)}
    ranked = sorted(labels, key=lambda name: -means[name])
    position = np.arange(len(order))

    aggregates = {}
    for attack, name in (("degree", "Targeted"), ("random", "Random"), ("both", "Weighted")):
        values = []
        for policy in order:
            rows_ = subset[subset["policy"] == policy]
            if attack == "both":
                values.append(float(rows_["r_res"].mean()))
            else:
                values.append(float(rows_.loc[rows_["attack"] == attack, "r_res"].mean()))
        aggregates[name] = values

    peak = max(max(aggregates[name]) for name in aggregates)
    y_top = 0.05 * np.ceil((peak + 0.04) / 0.05)
    y_ticks = np.arange(0.0, y_top + 1e-9, 0.1)

    for axis, attack, title in ((axes[0, 0], "degree", "(a) Targeted attack"),
                                (axes[0, 1], "random", "(b) Random attack")):
        for label in ranked:
            policy = dict(POLICIES)[label]
            grid, curve = mean_curve(frame, "BA-100", policy, attack, "r_curve")
            if grid.size == 0:
                continue
            axis.plot(grid, curve,
                      lw=2.4 if label == "BiT-HyRL" else 1.15,
                      color=COLOURS[label],
                      label=f"{label} ($R_{{\\mathrm{{res}}}}$={means[label]:.3f})",
                      solid_capstyle="round",
                      zorder=5 if label == "BiT-HyRL" else 3)
        axis.set_xlabel("Removed fraction $f$")
        axis.set_ylabel("GCC", fontstyle="italic")
        axis.set_title(title, loc="left")
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1.01)
        axis.legend(loc="upper right", frameon=False, handlelength=1.8,
                    handletextpad=0.45, borderpad=0.10, labelspacing=0.12,
                    borderaxespad=0.28, prop=dict(weight="bold", size=7.0))
        bold_ticks(axis)

    box = axes[1, 0].boxplot(data, positions=position, widths=0.58, patch_artist=True,
                             medianprops=dict(color="black", lw=1.0),
                             flierprops=dict(markersize=2, markeredgewidth=0.4))
    for patch, label in zip(box["boxes"], labels):
        patch.set_facecolor(COLOURS[label])
        patch.set_alpha(0.78)
        patch.set_edgecolor("black")
        patch.set_linewidth(0.6)
    axes[1, 0].set_xticks(position)
    axes[1, 0].set_xticklabels(labels, rotation=30, ha="right")
    axes[1, 0].set_ylabel("$R_{\\mathrm{res}}$ (mean of both attacks)")
    axes[1, 0].set_title("(c) Distribution over 20 seeds", loc="left")
    caps = [float(np.max(box["caps"][2 * i + 1].get_ydata())) for i in range(len(labels))]
    tops = [float(np.max(values)) for values in data]
    axes[1, 0].set_ylim(0.0, y_top)
    axes[1, 0].set_yticks(y_ticks)
    axes[1, 0].set_xlim(-0.55, 6.55)
    for label, x, cap, top in zip(labels, position, caps, tops):
        axes[1, 0].annotate(f"{means[label]:.3f}", (x, cap),
                            textcoords="offset points",
                            xytext=(0, 3.2 if top <= cap else 9.0),
                            ha="center", va="bottom", fontsize=6.2,
                            fontweight="bold", color="0.10")
    bold_ticks(axes[1, 0])

    if bar_style:
        width = 0.27
        for offset, name, alpha, hatch in ((-width, "Targeted", 0.42, "///"),
                                           (0.0, "Random", 0.70, ""),
                                           (width, "Weighted", 1.0, "...")):
            axes[1, 1].bar(position + offset, aggregates[name], width=width * 0.95,
                           color=[COLOURS[label] for label in labels], alpha=alpha,
                           edgecolor="black", linewidth=0.4, hatch=hatch)
        from matplotlib.patches import Patch
        axes[1, 1].legend(handles=[Patch(facecolor="0.55", edgecolor="black", hatch="///",
                                         label="Targeted"),
                                   Patch(facecolor="0.55", edgecolor="black",
                                         label="Random"),
                                   Patch(facecolor="0.55", edgecolor="black", hatch="...",
                                         label="Weighted")],
                          frameon=False, ncol=3, loc="upper left", handlelength=1.5,
                          columnspacing=0.9, fontsize=6.2, borderpad=0.2,
                          labelspacing=0.25)
        axes[1, 1].set_ylim(0.0, y_top)
        axes[1, 1].set_yticks(y_ticks)
        axes[1, 1].set_xlim(-0.55, 6.55)
        for x, value in zip(position, aggregates["Weighted"]):
            label = axes[1, 1].annotate(f"{value:.3f}", (x + width, value),
                                        textcoords="offset points", xytext=(0, 3.0),
                                        ha="center", va="bottom", fontsize=5.5,
                                        fontweight="bold", color="0.10")
            label.set_path_effects([patheffects.withStroke(linewidth=1.6,
                                                           foreground="white")])
    else:
        series = (("Targeted", "0.58", "--", "s", 1.1, 3.0),
                  ("Random", "0.30", "-.", "o", 1.1, 3.2),
                  ("Weighted", "0.05", "-", "D", 1.7, 3.6))
        for name, colour, style, marker, width, size in series:
            axes[1, 1].plot(position, aggregates[name], ls=style, marker=marker,
                            color=colour, lw=width, ms=size, mfc="white", mew=0.9,
                            label=name, zorder=5 if name == "Weighted" else 3)
        axes[1, 1].legend(frameon=False, ncol=3, loc="upper left", handlelength=1.9,
                          columnspacing=1.0, fontsize=6.2, borderpad=0.2,
                          labelspacing=0.25)
        axes[1, 1].set_ylim(0.0, y_top)
        axes[1, 1].set_yticks(y_ticks)
        axes[1, 1].set_xlim(-0.5, 7.2)

    axes[1, 1].set_xticks(position)
    axes[1, 1].set_xticklabels(labels, rotation=30, ha="right")
    axes[1, 1].set_ylabel("$R_{\\mathrm{res}}$")
    axes[1, 1].set_title("(d) Targeted vs random vs weighted", loc="left")
    bold_ticks(axes[1, 1])

    figure.subplots_adjust(wspace=WSPACE, hspace=HSPACE, **MARGINS)

    if not bar_style:
        figure.canvas.draw()
        box = axes[1, 1].get_window_extent()
        low, high = axes[1, 1].get_ylim()
        per_point = (high - low) / box.height if box.height else 0.001
        clearance = 8.0 * (figure.dpi / 72.0) * per_point
        for x in position:
            stacked = sorted(((aggregates[name][x], name) for name in aggregates),
                             key=lambda item: -item[0])
            crowded = any(upper[0] - lower[0] < clearance
                          for upper, lower in zip(stacked, stacked[1:]))
            for rank, (value, name) in enumerate(stacked):
                if crowded and rank == 0:
                    xytext, align, valign = (-9.0, 5.5), "right", "bottom"
                elif crowded and rank == 1:
                    xytext, align, valign = (5.5, 1.0), "left", "center"
                elif crowded and rank == 2:
                    xytext, align, valign = (0, -4.8), "center", "top"
                else:
                    xytext, align, valign = (0, 3.4), "center", "bottom"
                colour = {"Targeted": "0.58", "Random": "0.30", "Weighted": "0.05"}[name]
                label = axes[1, 1].annotate(f"{value:.3f}", (x, value),
                                            textcoords="offset points", xytext=xytext,
                                            ha=align, va=valign, fontsize=5.8,
                                            color=colour)
                label.set_path_effects([patheffects.withStroke(linewidth=1.5,
                                                               foreground="white")])

    for suffix in ("pdf", "png"):
        figure.savefig(out / f"{stem}.{suffix}")
    plt.close(figure)
    print(f"[fig] {out / f'{stem}.pdf'}")


def figure_trends(frame: pd.DataFrame, ablation: pd.DataFrame, out: Path) -> None:
    """Fig. 4 --- 'Bimodal network enhance resilience.pdf'; see :func:`_trend_panels`."""
    with plt.rc_context(MATH_STYLE):
        _trend_panels(frame, ablation, out)


def _trend_panels(frame: pd.DataFrame, ablation: pd.DataFrame, out: Path) -> None:
    """GCC / CSA / H_C under degree-ordered removal on BA-100 and Colt."""
    merged = pd.concat([frame, ablation], ignore_index=True)
    merged = merged.drop_duplicates(subset=["dataset", "policy", "attack", "seed"],
                                    keep="first")
    available = [label for label, policy in POLICIES
                 if policy in set(merged["policy"])]
    panels = (("r_curve", "GCC"), ("csa_curve", "CSA"), ("hc_curve", "H_C"))
    figure, axes = plt.subplots(2, 3, figsize=(7.2, TREND_HEIGHT_IN), sharex=True)

    policies = dict(POLICIES)
    for row, dataset in enumerate(("BA-100", "Colt")):
        for column, (key, name) in enumerate(panels):
            axis = axes[row, column]
            for label in available:
                grid, curve = mean_curve(merged, dataset, policies[label], "degree", key)
                if grid.size == 0:
                    continue
                line, = axis.plot(grid, curve,
                                  lw=2.7 if label == "BiT-HyRL" else 1.35,
                                  color=COLOURS[label], label=label,
                                  solid_capstyle="round",
                                  zorder=5 if label == "BiT-HyRL" else 3)
                if label == "BiT-HyRL":
                    line.set_path_effects([patheffects.withStroke(linewidth=4.0,
                                                                  foreground="white")])
            axis.set_title(f"({'abcdef'[row * 3 + column]}) {dataset}: ${name}$",
                           loc="left", fontsize=TREND_TITLE_FS)
            axis.grid(False)
            axis.set_xlim(0, 1)
            if row == 1:
                axis.set_xlabel("Removed fraction $f$", fontsize=TREND_AXIS_FS)
            else:
                axis.tick_params(labelbottom=False)
            if column == 0:
                axis.set_ylabel("Normalised value", fontsize=TREND_AXIS_FS)
            bold_ticks(axis)
            for tick in axis.get_xticklabels() + axis.get_yticklabels():
                tick.set_fontsize(TREND_TICK_FS)

    for column in range(3):
        top = max(float(axes[row, column].get_ylim()[1]) for row in range(2))
        for row in range(2):
            axes[row, column].set_ylim(0.0, top)

    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    figure.subplots_adjust(left=0.074, right=0.995, bottom=0.124, top=0.858,
                           wspace=0.24, hspace=0.285)
    left = axes[0, 0].get_position().x0
    right = axes[0, 2].get_position().x1
    figure.legend(handles, legend_labels, loc="lower left",
                  bbox_to_anchor=(left, 0.928, right - left, 0.0),
                  mode="expand", ncol=len(legend_labels), frameon=False,
                  handlelength=1.5, handletextpad=0.45, borderaxespad=0.0, borderpad=0.0,
                  columnspacing=1.0, prop=dict(weight="normal", size=7.0))
    for suffix in ("pdf", "png"):
        figure.savefig(out / f"Bimodal network enhance resilience.{suffix}")
    plt.close(figure)
    print(f"[fig] {out / 'Bimodal network enhance resilience.pdf'}")


def load_scalars() -> pd.DataFrame:
    """Per-run scalars for all thirty topologies (Fig. 7 reads these, not the curves)."""
    parts = []
    for relative in ("baselines/method_matrix/matrix_scalar.csv",
                     "baselines/ablation/ablation_all_scalar.csv"):
        path = RESULTS / relative
        if path.exists():
            parts.append(pd.read_csv(path, usecols=["dataset", "policy", "attack",
                                                    "seed", "r_res"]))
    if not parts:
        raise SystemExit("scalar tables missing: results/baselines/*/*_scalar.csv")
    return pd.concat(parts, ignore_index=True)


FAMILY_SHAPE = (("ISP", "o"), ("WS", "s"), ("AS", "^"), ("ER", "D"), ("BA", "v"))
TIE_SIGMA = 0.5
BOUNDARY_K = 2.45
#: Fill and rim of the plotted margin; the rim is the fill darkened, as in the other figures.
HERO_FILL, HERO_EDGE = "#16BCCD", "#0E8A96"


def family_of(dataset: str) -> str:
    """Topology family from the dataset name; the six real backbones are the ISP family."""
    for prefix in ("WS-", "ER-", "BA-", "AS-"):
        if dataset.startswith(prefix):
            return prefix[:-1]
    return "ISP"


def figure_domain(frame: pd.DataFrame, out: Path) -> None:
    """Where the method leads and where it does not, against mean degree."""
    import json
    from matplotlib.colors import to_rgb
    registry = {g["dataset"]: g for g in
                json.loads((REPO_ROOT / "configs" / "graphs.json")
                           .read_text(encoding="utf-8"))["graphs"]}
    per_graph = frame.groupby(["dataset", "policy"])["r_res"].mean().unstack()
    external = [name for name in per_graph.columns
                if name not in ("BiT-HyRL", "BiT-c1.0", "BiT-degOnly", "BiT-norecon")]
    if "BiT-HyRL" not in per_graph.columns or not external:
        return
    margin = (per_graph["BiT-HyRL"] - per_graph[external].max(axis=1)).dropna()
    seed_sd = frame[frame["policy"] == "BiT-HyRL"].groupby("dataset")["r_res"].std()
    kbar = pd.Series({name: 2.0 * spec["m"] / spec["n"]
                      for name, spec in registry.items()})

    points = pd.DataFrame({"margin": margin,
                           "sd": seed_sd.reindex(margin.index),
                           "k": kbar.reindex(margin.index)}).dropna()
    points["family"] = [family_of(name) for name in points.index]
    points["state"] = "leads"
    points.loc[points["margin"].abs() < TIE_SIGMA * points["sd"], "state"] = "tie"
    points.loc[points["margin"] < 0, "state"] = "overtaken"

    points["x"] = points["k"]
    for _, column in points[points.duplicated("k", keep=False)].groupby("k"):
        points.loc[column.index, "x"] = column["k"] + np.linspace(-0.055, 0.055, len(column))

    figure, axis = plt.subplots(figsize=(3.36, 2.62))
    axis.axvline(BOUNDARY_K, color="0.45", lw=0.8, ls=(0, (4.0, 2.2)), zorder=1)
    axis.axhline(0, color="0.35", lw=0.75, zorder=1)

    style = {"leads": dict(face=HERO_FILL, fill="full", rim=0.75, z=5),
             "tie": dict(face=HERO_FILL, fill="left", rim=0.75, z=5),
             "overtaken": dict(face="white", fill="full", rim=0.95, z=6)}
    for family, marker in FAMILY_SHAPE:
        for state in ("leads", "tie", "overtaken"):
            subset = points[(points["family"] == family) & (points["state"] == state)]
            if subset.empty:
                continue
            spec = style[state]
            axis.plot(subset["x"], subset["margin"], ls="", marker=marker, ms=5.6,
                      mfc=spec["face"], mec=HERO_EDGE, mew=spec["rim"],
                      fillstyle=spec["fill"], zorder=spec["z"])

    axis.text(BOUNDARY_K + 0.05, points["margin"].max(), "$\\bar{k}=2.5$",
              ha="left", va="top", fontsize=5.8, color="0.35")
    axis.set_xlabel("Mean degree $\\bar{k}$")
    axis.set_ylabel("$\\Delta R_{\\mathrm{res}}$ vs best external baseline")
    axis.set_xlim(points["k"].min() - 0.22, points["k"].max() + 0.22)
    axis.set_ylim(-0.125, 0.122)
    axis.set_xticks([2, 3, 4, 5, 6])
    axis.set_yticks(np.arange(-0.10, 0.101, 0.05))
    axis.yaxis.set_major_formatter(plt.FormatStrFormatter("%.2f"))
    bold_ticks(axis)

    family_handles = [plt.Line2D([], [], marker=marker, ls="", ms=5.0,
                                 mfc=HERO_FILL, mec=HERO_EDGE, mew=0.75, label=family)
                      for family, marker in FAMILY_SHAPE]
    figure.legend(handles=family_handles, ncol=5, loc="lower center", frameon=False,
                  handletextpad=0.35, columnspacing=1.0, fontsize=6.8,
                  borderaxespad=0.0, bbox_to_anchor=(0.5, -0.012))
    state_handles = [plt.Line2D([], [], marker="o", ls="", ms=5.9,
                                mfc=style[key]["face"], mec=HERO_EDGE,
                                mew=style[key]["rim"] + 0.1,
                                fillstyle=style[key]["fill"], label=text)
                     for key, text in (("leads", "leads every baseline"),
                                       ("tie", "tie ($<0.5\\,\\sigma$)"),
                                       ("overtaken", "overtaken"))]
    axis.legend(handles=state_handles, loc="lower right", frameon=False, fontsize=7.0,
                handletextpad=0.55, labelspacing=0.45, borderaxespad=0.30)
    figure.subplots_adjust(left=0.145, right=0.988, bottom=0.185, top=0.972)
    for suffix in ("pdf", "png"):
        figure.savefig(out / f"applicability_domain.{suffix}")
    plt.close(figure)
    print(f"[fig] {out / 'applicability_domain.pdf'}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="draw the data figures")
    parser.add_argument("--out", default=None, help="manuscript directory (figures are written here)")
    parser.add_argument("--check", action="store_true",
                        help="report missing curves only, do not draw")
    parser.add_argument("--only", choices=("all", "comparison", "trends"), default="all",
                        help="redraw one figure only (default: both)")
    parser.add_argument("--d-style", choices=("bars", "lines"), default="bars",
                        help="panel (d) of Fig. 4: grouped bars (default) or lines")
    parser.add_argument("--with-domain", action="store_true",
                        help="also redraw Fig. 7 from the scalar tables")
    args = parser.parse_args(argv)

    if args.check:
        frame = load_curves()
        for dataset in ("BA-100", "Colt"):
            for label, policy in POLICIES:
                subset = frame[(frame["dataset"] == dataset) & (frame["policy"] == policy)]
                seeds = sorted(set(int(seed) for seed in subset["seed"]))
                attacks = sorted(set(subset["attack"].astype(str)))
                status = "ok  " if seeds and len(attacks) == 2 else "MISS"
                print(f"  {status} {dataset:<8} {label:<10} seeds={len(seeds):<2} "
                      f"attacks={','.join(attacks) or '-'}")
        return 0

    if args.out is None:
        raise SystemExit("--out is required (or use --check to report coverage)")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    frame = load_curves()
    ablation = load_ablation(frame)
    if args.only in ("all", "comparison"):
        bars = args.d_style == "bars"
        figure_comparison(frame, out, bar_style=bars,
                          stem="Comparison_topology" if bars
                          else "Comparison_topology_lines")
    if args.only in ("all", "trends"):
        figure_trends(frame, ablation, out)
    if args.with_domain:
        figure_domain(load_scalars(), out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""The substrate-construction figure, redrawn in the layout of the old Fig. 4.

Two files come out of here:

``construction_overview.pdf``  (Fig. 5)
    Four panels.  (a) and (b) are the GCC curves of the eight drawn constructions on
    ``BA-100`` under degree-ordered and uniform removal, averaged over the five construction
    seeds;
    (c) is the distribution of the four-area mean ``A`` over the thirty substrates; (d)
    splits that mean into its two attack modes, so the two bars average to the number
    printed above them.  Colour is the method throughout, so a curve in (a) and a box in
    (c) are the same construction.

``construction_mechanism.pdf``  (Fig. 6)
    Two panels, both on the thirty substrates: ``A`` against the second-moment ratio
    ``kappa`` and against the construction wall clock.  These are the two panels that used
    to sit under the family bars; they carry the mechanism claim and the cost claim, which
    is why they are separated from the ranking figure.

Every number is read from files the benchmark already wrote: ``curves_BA-100.csv``
(``export_construction_curves.py``), ``family_bench.csv`` and ``agg_method.csv``.  Nothing
is recomputed here, so the figures cannot drift away from the tables.

Usage
-----
::

    python scripts/fig_construction.py --out ../03-第三次投稿\\ RESS/manuscript
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "results" / "family"

#: ``np.trapz`` was renamed to ``np.trapezoid`` in NumPy 2.  Resolving it once here keeps
#: ``getattr(np, "trapezoid", np.trapz)`` from evaluating the removed name as its default.
_TRAPEZOID = getattr(np, "trapezoid", None) or getattr(np, "trapz")

#: manuscript label -> (colour, drawn label).  The teal of ``BiT-HyRL`` is the one used in
#: the other figures; the four reference rows stay grey/brown so they read as controls.
STYLE = {
    "BiT-HyRL": ("#16BCCD", "BiT-HyRL"),
    "Random": ("#9e9e9e", "Random"),
    "UNITY": ("#EE929D", "UNITY"),
    "WS": ("#8c6d46", "WS"),
    "QDLM": ("#A9E7E2", "QDLM"),
    "FRED-ABL": ("#E8B88A", "FRED-ABL"),
    "Eff-R-Net": ("#7FB3D5", "Eff-R-Net"),
    "ROMEN": ("#b15928", "ROMEN"),
    "ONION": ("#8CCB9B", "ONION"),
    "BA": ("#c49a6c", "BA"),
    "Original": ("#4d4d4d", "Original"),
}
#: The reference rows of Table 7 are constructions, not competitors; they are drawn dashed
#: so that the legend of (a)--(b) separates them from the five methods.
REFERENCE = {"WS", "BA", "Original"}
#: What the figure draws.  ``Random (ER)`` stays in Table 7 as the stochastic control but is
#: not drawn: it is one more reference construction and it cluttered the ranking panels
#: without changing their order.
METHODS = ("BiT-HyRL", "UNITY", "WS", "QDLM", "FRED-ABL",
           "ROMEN", "BA", "Original")

#: Sequential ramp of panel (e), in the accent hue of the figure rather than the default
#: ``Blues``: the same cyan family as ``#16BCCD``, passed through ``#5BCFDB`` in the middle.
#: Positions are chosen so that no integer rank lands near the luminance at which the cell
#: label has to flip from white to ink, and so that lightness increases monotonically--the
#: strip therefore still decodes in greyscale.  ``rank 1`` is the dark end.
RANK_RAMP = ((0.00, "#0A3B42"), (0.14, "#0C4E57"), (0.28, "#116672"),
             (0.40, "#3AA9B7"), (0.62, "#5BCFDB"), (0.82, "#93DEE7"),
             (1.00, "#D9EFF3"))


def _rank_colormap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("rank_accent", RANK_RAMP, N=256)


def _cell_ink(colour) -> str:
    """White or ink for a rank label, decided by the luminance of the cell underneath."""
    def channel(value: float) -> float:
        return value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4
    luminance = (0.2126 * channel(colour[0]) + 0.7152 * channel(colour[1])
                 + 0.0722 * channel(colour[2]))
    return "white" if luminance < 0.20 else "0.10"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.labelweight": "bold",
    "axes.titlesize": 8.5,
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

#: Bold math in Times-like glyphs, as in the other figures.
MATH_STYLE = {"mathtext.default": "bfit", "mathtext.fontset": "stix"}
#: Subplot box and row/column gaps of the four-panel block, shared with the old Fig. 4 so
#: the two figures stand at the same height in the text.
MARGINS = {"left": 0.075, "right": 0.992, "bottom": 0.085, "top": 0.972}
BASE_HEIGHT_IN, PANEL_SHRINK_MM = 4.85, 6.0
HSPACE, WSPACE = 0.33, 0.16
#: The two-panel body keeps the height of the old Fig. 4; (e) is a strip underneath.
OVERVIEW_HEIGHT_IN = 5.85
STRIP_RATIO = 0.47

#: Row order and short labels of the ordering strip in (e).
ORDER_ROWS = ("degree", "betweenness", "closeness", "coreness", "eigenvector", "random")
ORDER_LABELS = ("Degree", "Betweenness", "Closeness", "Coreness", "Eigenvector", "Random")


def bold_ticks(axis) -> None:
    for tick in axis.get_xticklabels() + axis.get_yticklabels():
        tick.set_fontweight("bold")


def load(dataset: str):
    """Curves, per-(graph, construction) scores and the method means."""
    curves = pd.read_csv(DATA / f"curves_{dataset}.csv")
    bench = pd.read_csv(DATA / "family_bench.csv")
    means = pd.read_csv(DATA / "agg_method.csv")
    curves["f"] = curves["f"].astype(float)
    curves["gcc"] = curves["gcc"].astype(float)
    return curves, bench, means


def _aligned_labels(names, values, size=6.4):
    """``name 0.469`` for every entry, with the numbers in one straight column.

    The legend has no second column of its own, so the gap is padded with ordinary spaces
    measured against the widest name; padding on the inside of the string is what survives
    the legend's own width calculation, trailing spaces are not.
    """
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextPath
    prop = FontProperties(family="serif", weight="bold", size=size)

    def advance(text: str) -> float:
        return TextPath((0, 0), text, prop=prop).get_extents().width

    space = advance("n n") - advance("nn")
    if space <= 1e-3:
        return [f"{name} {value:.3f}" for name, value in zip(names, values)]
    widest = max(advance(name) for name in names) + space
    return [f"{name}{' ' * max(int(round((widest - advance(name)) / space)), 1)}{value:.3f}"
            for name, value in zip(names, values)]


def panel_curves(axis, curves: pd.DataFrame, dataset: str, attack: str, title: str,
                 ranking: list[str]) -> None:
    series = []
    for method in ranking:
        subset = curves[(curves["method"] == method) & (curves["attack"] == attack)]
        if subset.empty:
            continue
        grid = np.sort(subset["f"].unique())
        values = (subset.pivot_table(index="seed", columns="f", values="gcc")
                  .to_numpy().mean(axis=0))
        # The number printed beside each entry is the area under the curve that is drawn,
        # i.e. GCC integrated over f for this substrate under this mode -- the same object
        # as R_GCC in Table 7, and one the reader can check against the line itself.  It is
        # deliberately *not* the thirty-substrate A of (c): a score of a different scope
        # printed next to a single substrate's curve is what made the old figure invite
        # 0.428-against-0.271 comparisons.
        series.append((method, grid, values, float(_TRAPEZOID(values, grid))))

    labels = _aligned_labels([STYLE[method][1] for method, _, _, _ in series],
                             [area for _, _, _, area in series])
    for (method, grid, values, _), label in zip(series, labels):
        axis.plot(grid, values, lw=2.4 if method == "BiT-HyRL" else 1.15,
                  color=STYLE[method][0], solid_capstyle="round",
                  ls=(0, (4, 1.6)) if method in REFERENCE else "-",
                  label=label,
                  zorder=5 if method == "BiT-HyRL" else 3)
    axis.set_xlabel("Removed fraction $f$")
    axis.set_ylabel("GCC", fontstyle="italic")
    axis.set_title(title, loc="left")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1.01)
    axis.legend(loc="upper right", frameon=False, handlelength=1.8, handletextpad=0.45,
                borderpad=0.10, labelspacing=0.10, borderaxespad=0.26,
                prop=dict(weight="bold", size=6.4))
    bold_ticks(axis)


def panel_orders(axis, orders: pd.DataFrame, ranking: list[str]) -> None:
    """Ranking strip: one row per attack ordering, one column per construction.

    The six orderings are not four variants of one idea: ``degree``, ``betweenness``,
    ``closeness``, ``coreness`` and ``eigenvector`` all attack by centrality, whereas
    ``random`` is a uniform failure.  Drawing them together is what makes the two
    non-first cells of ``BiT-HyRL`` legible instead of hidden, and it is also what stops a
    reader from accusing the panel of cherry-picking the favourable orderings.
    """
    rank = (orders.pivot(index="order", columns="method", values="rank_mean")
            .reindex(list(ORDER_ROWS))[ranking])
    values = rank.to_numpy(dtype=float)

    from matplotlib.colors import Normalize
    cmap = _rank_colormap()
    lowest = 1.0
    highest = float(np.nanmax(values))
    axis.imshow(values, cmap=cmap, vmin=lowest, vmax=highest,
                aspect="auto", interpolation="nearest")
    scale = Normalize(vmin=lowest, vmax=highest)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            if not np.isfinite(value):
                continue
            axis.text(column, row, f"{int(value)}", ha="center", va="center",
                      fontsize=6.2, fontweight="bold", color=_cell_ink(cmap(scale(value))))

    axis.set_xticks(np.arange(len(ranking)))
    axis.set_xticklabels([STYLE[name][1] for name in ranking], rotation=26, ha="right")
    axis.set_yticks(np.arange(len(ORDER_ROWS)))
    axis.set_yticklabels(list(ORDER_LABELS))
    axis.set_xticks(np.arange(-0.5, len(ranking), 1), minor=True)
    axis.set_yticks(np.arange(-0.5, len(ORDER_ROWS), 1), minor=True)
    # Grey rather than white: the pale end of the ramp is nearly paper white, and white
    # separators would dissolve into it.
    axis.grid(which="minor", color="0.72", lw=0.9)
    axis.tick_params(which="minor", length=0)
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.6)
        spine.set_color("0.35")

    # The column of the proposed construction is boxed so that the eye lands on it first.
    # A single stroke cannot work on a teal ramp--the accent colour of the method would
    # disappear into the cells--so the box is drawn dark on a white halo.
    from matplotlib.patches import Rectangle
    try:
        column = ranking.index("BiT-HyRL")
    except ValueError:
        column = None
    if column is not None:
        axis.add_patch(Rectangle((column - 0.5, -0.5), 1.0, len(ORDER_ROWS), fill=False,
                                 ec="white", lw=3.0, zorder=6))
        axis.add_patch(Rectangle((column - 0.5, -0.5), 1.0, len(ORDER_ROWS), fill=False,
                                 ec="0.10", lw=1.1, zorder=7))

    axis.set_title("(e) Rank among the eight constructions under each ordering "
                   "(1 = most resilient)", loc="left", pad=1.5)
    bold_ticks(axis)


def build_overview(curves, bench, means, orders, out: Path) -> None:
    with plt.rc_context(MATH_STYLE):
        ranking = [name for name in METHODS if name in set(curves["method"])]
        table = means.set_index("method")
        mean_a = {name: float(table.loc[name, "A"]) for name in ranking}
        ranking.sort(key=lambda name: -mean_a[name])

        figure = plt.figure(figsize=(7.0, OVERVIEW_HEIGHT_IN))
        # The strip is a separate block, not a third row of the same grid: the rotated
        # x tick labels of (c) and (d) are tall, and a flat 3x2 grid puts them on top of
        # the title of (e).
        outer = figure.add_gridspec(2, 1, height_ratios=(2.0, STRIP_RATIO),
                                    left=MARGINS["left"], right=MARGINS["right"],
                                    bottom=MARGINS["bottom"], top=MARGINS["top"],
                                    hspace=0.34)
        body = outer[0].subgridspec(2, 2, hspace=HSPACE, wspace=WSPACE)
        axes = np.array([[figure.add_subplot(body[0, 0]), figure.add_subplot(body[0, 1])],
                         [figure.add_subplot(body[1, 0]), figure.add_subplot(body[1, 1])]])
        strip = figure.add_subplot(outer[1])

        # per-graph score: the seed mean of A for one substrate, then one point per graph
        per_graph = (bench.groupby(["dataset", "method"])["A"].mean().reset_index())
        data = [per_graph.loc[per_graph["method"] == name, "A"].to_numpy() for name in ranking]
        position = np.arange(len(ranking))

        # (d) must be additively consistent with (c): the two bars are the mean of the two
        # metric areas inside one attack mode, so the printed A is *literally* the mean of
        # the two bar heights.  Plotting the bare GCC areas beside A would put a 0.40 bar
        # next to a 0.27 figure and invite exactly the reading the old version produced.
        aggregates = {
            "Targeted": [0.5 * (float(table.loc[name, "R_gcc_tar"])
                                + float(table.loc[name, "R_e_tar"])) for name in ranking],
            "Random": [0.5 * (float(table.loc[name, "R_gcc_ran"])
                              + float(table.loc[name, "R_e_ran"])) for name in ranking],
            "A": [mean_a[name] for name in ranking],
        }
        # The boxes of (c) reach above the two bar means (BiT-HyRL's worst substrate sits at
        # 0.369 against a mean of 0.271), so the axis has to clear the per-graph maximum as
        # well, not just the bar heights; otherwise the top whisker and its label are clipped.
        peak = max(max(max(values) for values in aggregates.values()),
                   max(float(np.max(values)) for values in data))
        y_top = 0.05 * np.ceil((peak + 0.02) / 0.05)
        y_ticks = np.arange(0.0, y_top + 1e-9, 0.1)

        panel_curves(axes[0, 0], curves, "BA-100", "tar", "(a) Targeted attack (BA-100)",
                     ranking)
        panel_curves(axes[0, 1], curves, "BA-100", "ran", "(b) Random attack (BA-100)",
                     ranking)

        box = axes[1, 0].boxplot(data, positions=position, widths=0.58, patch_artist=True,
                                 medianprops=dict(color="black", lw=1.0),
                                 flierprops=dict(markersize=2, markeredgewidth=0.4))
        for patch, name in zip(box["boxes"], ranking):
            patch.set_facecolor(STYLE[name][0])
            patch.set_alpha(0.78)
            patch.set_edgecolor("black")
            patch.set_linewidth(0.6)
        axes[1, 0].set_xticks(position)
        axes[1, 0].set_xticklabels([STYLE[name][1] for name in ranking], rotation=30,
                                   ha="right")
        axes[1, 0].set_ylabel("$A$ (four-area mean)")
        axes[1, 0].set_title("(c) Distribution over the thirty substrates", loc="left")
        axes[1, 0].set_ylim(0.0, y_top)
        axes[1, 0].set_yticks(y_ticks)
        axes[1, 0].set_xlim(-0.55, len(ranking) - 0.45)
        caps = [float(np.max(box["caps"][2 * i + 1].get_ydata())) for i in range(len(ranking))]
        tops = [float(np.max(values)) for values in data]
        for name, x, cap, top in zip(ranking, position, caps, tops):
            axes[1, 0].annotate(f"{mean_a[name]:.3f}", (x, cap), textcoords="offset points",
                                xytext=(0, 3.2 if top <= cap else 9.0), ha="center",
                                va="bottom", fontsize=6.0, fontweight="bold", color="0.10")
        bold_ticks(axes[1, 0])

        width = 0.30
        for offset, name, alpha, hatch in ((-0.175, "Targeted", 0.40, "///"),
                                           (0.175, "Random", 0.70, "")):
            axes[1, 1].bar(position + offset, aggregates[name], width=width,
                           color=[STYLE[method][0] for method in ranking], alpha=alpha,
                           edgecolor="black", linewidth=0.4, hatch=hatch)
        # A is the mean of the two bars, so it is marked at its own level rather than drawn
        # as a third bar: a third bar of a different quantity is what made the old version
        # print 0.271 beside two bars of 0.40.
        axes[1, 1].plot(position, aggregates["A"], ls="", marker="D", ms=3.4,
                        mfc="0.10", mec="white", mew=0.6, zorder=8,
                        label="$A$ (mean of the two)")
        from matplotlib.patches import Patch
        # The bars are the two attack modes; with the stochastic construction gone from the
        # panels, "Random" alone would read as a method name again, so the legend spells out
        # what the two bars are.  The entries are listed [targeted, A, random] because the
        # legend fills column-wise: two columns of two rows then read "targeted | random"
        # on the first line and A underneath, which is the only way all three fit the panel.
        axes[1, 1].legend(handles=[Patch(facecolor="0.55", edgecolor="black", hatch="///",
                                         label="Targeted attack"),
                                   plt.Line2D([], [], ls="", marker="D", ms=3.4,
                                              mfc="0.10", mec="white", mew=0.6,
                                              label="$A$ (mean of the two)"),
                                   Patch(facecolor="0.55", edgecolor="black",
                                         label="Random attack")],
                          frameon=False, ncol=2, loc="upper left", handlelength=1.4,
                          columnspacing=0.8, fontsize=6.0, borderpad=0.2, labelspacing=0.18)
        axes[1, 1].set_ylim(0.0, y_top)
        axes[1, 1].set_yticks(y_ticks)
        axes[1, 1].set_xlim(-0.55, len(ranking) - 0.45)
        axes[1, 1].set_xticks(position)
        axes[1, 1].set_xticklabels([STYLE[name][1] for name in ranking], rotation=30,
                                   ha="right")
        axes[1, 1].set_ylabel("Area (mean over metrics)")
        axes[1, 1].set_title("(d) Split by attack mode", loc="left")
        bold_ticks(axes[1, 1])

        panel_orders(strip, orders, ranking)

        figure.savefig(out / "construction_overview.pdf")
        figure.savefig(out / "construction_overview.png", dpi=330)
        plt.close(figure)
        print(f"[out] {out / 'construction_overview.pdf'}")


#: Marker areas of the two panels.  One value per role, used in both panels, so a dot in
#: (a) and the dot of the same construction in (b) read as the same object.
DOT, HERO_DOT = 15.0, 54.0
HERO = "BiT-HyRL"
#: Left edge of (b)'s log axis: a whole decade, so that the first major tick is labelled.
#: Nothing sits left of BA at 26.5 ms once Original is removed, but a decade marker at the
#: corner still tells the reader where the scale starts.
PLOT_X_MIN = 1.0e-2
#: ``Original`` is the graph as given: it has no construction step, so it has no abscissa in
#: (b), whose axis is a wall clock.  Placing it at a made-up 5 ms made it read as the fastest
#: construction on the panel.  It stays in (a), where the abscissa is a property of the graph
#: and not of the process, and the caption says so.
NO_COST = "Original"


def _edge(colour: str, factor: float = 0.70):
    """A darker version of a fill colour, for a hairline rim on the dots.

    Two of the seven fills are pale by design (``QDLM`` above all); at 4 pt across and 100 %
    print they lose their outline on white.  A rim in the fill's own hue, darkened, restores
    the boundary without introducing a second colour or changing the palette.
    """
    from matplotlib.colors import to_rgb
    return tuple(channel * factor for channel in to_rgb(colour))


def build_mechanism(points: pd.DataFrame, means: pd.DataFrame, ranking: list[str],
                    out: Path) -> None:
    with plt.rc_context(MATH_STYLE):
        figure, axes = plt.subplots(1, 2, figsize=(7.0, 2.02))
        table = means.set_index("method")
        # ``fig_points.csv`` also carries BiT-E, BiT-R, Random, ONION and Eff-R-Net, which
        # the ranking figure does not draw.  The connectors are built from the drawn rows
        # only: joining points that are not on the panel produced segments with one end in
        # mid-air (150 of 360) and a comb of vertical segments wherever a hidden
        # construction shared the kappa of a drawn one.
        drawn = points[points["method"].isin(ranking)].copy()

        # Connectors first, so the dots sit on top of them.  One line per substrate; the
        # claim is conditional on the substrate, because the graph-to-graph offsets are
        # larger than the effect.  Ties in kappa are walked upwards in A, so a shared kappa
        # gives one vertical step instead of a zigzag.
        for _, group in drawn.groupby("dataset"):
            group = group.sort_values(["kappa", "A"])
            axes[0].plot(group["kappa"], group["A"], color="0.40", lw=0.34, alpha=0.30,
                         zorder=1, solid_capstyle="round")

        for name in ranking:
            subset = drawn[drawn["method"] == name]
            hero = name == HERO
            dots = axes[0].scatter(subset["kappa"], subset["A"],
                                   s=HERO_DOT if hero else DOT,
                                   alpha=1.0 if hero else 0.85,
                                   color=STYLE[name][0], linewidths=0.0 if hero else 0.35,
                                   edgecolors=_edge(STYLE[name][0]),
                                   label=STYLE[name][1],
                                   zorder=6 if hero else 3)
            if hero:
                dots.set_path_effects([patheffects.withStroke(linewidth=0.85,
                                                              foreground="white")])
        axes[0].set_xscale("log")
        axes[0].set_xticks([2, 3, 5, 8, 13, 20, 30])
        axes[0].set_xticklabels(["2", "3", "5", "8", "13", "20", "30"])
        axes[0].minorticks_off()
        axes[0].set_yticks(np.arange(0.10, 0.351, 0.05))
        axes[0].set_ylim(0.075, 0.385)
        axes[0].yaxis.set_major_formatter(plt.FormatStrFormatter("%.2f"))
        axes[0].set_xlabel(r"Second-moment ratio $\kappa$")
        axes[0].set_ylabel("$A$ (four-area mean)")
        axes[0].set_title("(a) Mechanism: $A$ against $\\kappa$", loc="left")
        bold_ticks(axes[0])

        # (b) is the same quantity on a zoomed vertical scale, and it carries the reference
        # line of BiT-HyRL's A: no slower construction reaches it.  The repeated y label is
        # dropped for that reason and replaced by an in-panel reminder of the scale.
        reference = float(table.loc[HERO, "A"])
        for name in ranking:
            if name == NO_COST:
                continue
            hero = name == HERO
            seconds = float(table.loc[name, "construct_s"])
            dots = axes[1].scatter(seconds, float(table.loc[name, "A"]),
                                   s=HERO_DOT if hero else DOT,
                                   alpha=1.0 if hero else 0.85,
                                   color=STYLE[name][0], linewidths=0.0 if hero else 0.35,
                                   edgecolors=_edge(STYLE[name][0]))
            if hero:
                dots.set_path_effects([patheffects.withStroke(linewidth=0.85,
                                                              foreground="white")])
            if hero:
                dots.set_path_effects([patheffects.withStroke(linewidth=0.85,
                                                              foreground="white")])
        axes[1].axhline(reference, color="0.45", lw=0.8, ls=(0, (4.0, 2.0)), zorder=1)
        axes[1].text(0.982, reference, "$A$ of BiT-HyRL",
                     transform=axes[1].get_yaxis_transform(), ha="right", va="bottom",
                     fontsize=5.9, color="0.35")
        axes[1].set_xscale("log")
        axes[1].set_xlim(PLOT_X_MIN, 1.5e3)
        axes[1].set_yticks(np.arange(0.20, 0.281, 0.02))
        axes[1].set_ylim(0.186, 0.292)
        axes[1].yaxis.set_major_formatter(plt.FormatStrFormatter("%.2f"))
        axes[1].set_xlabel("Construction wall clock (s, one core)")
        axes[1].set_title("(b) Cost: $A$ against construction time", loc="left")
        axes[1].text(0.018, 0.965, "$A$, vertical scale zoomed", transform=axes[1].transAxes,
                     ha="left", va="top", fontsize=5.6, color="0.40")
        bold_ticks(axes[1])

        handles = [plt.Line2D([], [], marker="o", ls="",
                              ms=5.4 if name == HERO else 3.7,
                              color=STYLE[name][0], label=STYLE[name][1],
                              mec="white" if name == HERO else _edge(STYLE[name][0]),
                              mew=0.0 if name == HERO else 0.35)
                   for name in ranking]
        figure.legend(handles=handles, ncol=8, loc="lower center", frameon=False,
                      handletextpad=0.28, columnspacing=1.05, fontsize=6.6,
                      borderaxespad=0.0, bbox_to_anchor=(0.5, -0.062))
        figure.subplots_adjust(wspace=0.145, left=0.072, right=0.992, bottom=0.196,
                               top=0.905)
        figure.savefig(out / "construction_mechanism.pdf")
        figure.savefig(out / "construction_mechanism.png", dpi=330)
        plt.close(figure)
        print(f"[out] {out / 'construction_mechanism.pdf'}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="construction figure in the Fig. 4 layout")
    parser.add_argument("--dataset", default="BA-100")
    parser.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "figures"))
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    curves, bench, means = load(args.dataset)
    # the benchmark writes the deployed arm as ``BiT-A``; the tables and the text call it
    # ``BiT-HyRL``, so the rename happens once, here, rather than in every panel.
    bench["method"] = bench["method"].replace({"BiT-A": "BiT-HyRL"})
    points = pd.read_csv(DATA / "fig_points.csv")
    orders = pd.read_csv(DATA / "attack_orders" / "attack_orders_summary.csv")
    ranking = [name for name in METHODS if name in set(curves["method"])]

    build_overview(curves, bench, means, orders, out)
    build_mechanism(points, means, ranking, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from pathlib import Path

import numpy as np
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib.pyplot import savefig, subplots

from vrtool.decision_making.strategies.strategy_base import StrategyBase
from vrtool.flood_defence_system.dike_traject import (
    DikeTraject,
    get_section_length_in_traject,
)


def plot_lcc(
    strategies_list: list[StrategyBase],
    traject: DikeTraject,
    output_dir: Path,
    flip: bool,
    fig_size: tuple[int, int],
    title_in: str,
    greedymode: str,
    color: list[ListedColormap],
):
    # TODO This should not be necessary:
    strategies_list[0].OptimalSolution["LCC"] = (
        strategies_list[0].OptimalSolution["LCC"].astype(np.float32)
    )
    strategies_list[0].SatisfiedStandardSolution["LCC"] = (
        strategies_list[0].SatisfiedStandardSolution["LCC"].astype(np.float32)
    )
    strategies_list[1].FinalSolution["LCC"] = (
        strategies_list[1].FinalSolution["LCC"].astype(np.float32)
    )

    # now for 2 strategies: plots an LCC bar chart
    cumlength, xticks1, middles = get_section_length_in_traject(
        traject.probabilities["Length"]
        .loc[traject.probabilities.index.get_level_values(1) == "Overflow"]
        .values
    )
    if not color:
        color = sns.cubehelix_palette(
            n_colors=4, start=1.9, rot=1, gamma=1.5, hue=1.0, light=0.8, dark=0.3
        )
    fig, (ax, ax1) = subplots(
        nrows=1,
        ncols=2,
        figsize=fig_size,
        sharey="row",
        gridspec_kw={
            "width_ratios": [20, 1],
            "wspace": 0.08,
            "left": 0.03,
            "right": 0.98,
        },
    )
    for i in cumlength:
        ax.axvline(x=i, color="gray", linestyle="-", linewidth=0.5, alpha=0.5)
    widths = (
        traject.probabilities["Length"]
        .loc[traject.probabilities.index.get_level_values(1) == "Overflow"]
        .values
        / 2
    )
    if greedymode == "Optimal":
        _greedy_solution = strategies_list[0].OptimalSolution["LCC"].values / 1e6
    elif greedymode == "SatisfiedStandard":
        _greedy_solution = (
            strategies_list[0].SatisfiedStandardSolution["LCC"].values / 1e6
        )

    ax.bar(
        np.subtract(middles, 0.45 * widths),
        _greedy_solution,
        widths * 0.9,
        color=color[0],
        label="Optimized",
    )
    ax.bar(
        np.add(middles, 0.45 * widths),
        strategies_list[1].FinalSolution["LCC"].values / 1e6,
        widths * 0.9,
        color=color[1],
        label="Target rel.",
    )

    # make x-axis nice
    ax.set_xlim(left=0, right=np.max(cumlength))
    labels_xticks = []
    for i in traject.sections:
        labels_xticks.append("S" + i.name[-2:])
    ax.set_xticks(middles)
    ax.set_xticklabels(labels_xticks)
    ax.tick_params(axis="x", rotation=90)
    # make y-axis nice
    lcc_max = (
        np.max(
            [
                strategies_list[0].OptimalSolution["LCC"].values,
                strategies_list[1].FinalSolution["LCC"].values,
            ]
        )
        / 1e6
    )
    if lcc_max < 10:
        ax.set_ylim(bottom=0, top=np.ceil(lcc_max / 2) * 2)
    if lcc_max >= 10:
        ax.set_ylim(bottom=0, top=np.ceil(lcc_max / 5) * 5)
    ax.set_ylabel("Cost in M€")
    ax.get_xticklabels()
    ax.tick_params(axis="both", bottom=False)

    # add a legend
    ax1.axis("off")
    ax.text(
        0,
        0.8,
        "Total LCC Optimized = {:.0f}".format(
            np.sum(_greedy_solution.astype(np.float32))
        )
        + " M€ \n"
        + "Total LCC Target rel. = {:.0f}".format(
            np.sum(strategies_list[1].FinalSolution["LCC"].values / 1e6)
        )
        + " M€",
        horizontalalignment="left",
        transform=ax.transAxes,
    )
    if flip:
        ax.invert_xaxis()
    ax.legend(bbox_to_anchor=(1.0001, 0.85))  # reposition!
    ax.grid(axis="y", linewidth=0.5, color="gray", alpha=0.5)
    if title_in:
        ax.set_title(title_in)

    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    savefig(output_dir / "LCC.png", dpi=300, bbox_inches="tight", format="png")

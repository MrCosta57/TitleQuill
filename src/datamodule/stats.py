"""
This script computes and plots statistics about the OAGKX dataset.
"""

from __future__ import annotations
import os
import pickle
import re
from os import path
import argparse
from typing import Any, Counter, Dict
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
from dataset import load_oagkx_dataset, OAGKXItem


def line_plot(
    data: Dict,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    save_path: str = "plot.png",
):
    """
    Function to plot a line plot of the data.

    :param data: Dictionary containing the data to plot.
    :param title: Title of the plot.
    :param xlabel: Label for the x-axis.
    :param ylabel: Label for the y-axis.
    :param save_path: Path to save the
    """

    # ----------- HYERPARAMETERS ------------

    def format(x, pos):
        return f"{int(x):,}"

    FIGSIZE = (12, 7)
    STYLE = "ggplot"
    AXES_FONT = 14
    TITLE_FONT = 16
    TICK_SIZE = 12
    # YPADDING   = 10 ** 6

    PLOT_ARGS = {"marker": "o", "linestyle": "-", "color": "b"}

    GRID_ARGS = {"which": "both", "linestyle": "--", "linewidth": 0.5}

    # ----------------------------------------

    # Set the style
    plt.style.use(STYLE)

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=FIGSIZE)

    # Sort data by key
    data = dict(sorted(data.items()))

    # Plot the data
    x = list(data.keys())
    y = list(data.values())

    ax.plot(x, y, **PLOT_ARGS)

    # Customize labels and title
    ax.set_xlabel(xlabel, fontsize=AXES_FONT)
    ax.set_ylabel(ylabel, fontsize=AXES_FONT)
    ax.set_title (title, fontsize=TITLE_FONT)

    # Set y-axis limits to the original scale of the data
    # ax.set_ylim(min(y) - YPADDING, max(y) + YPADDING)
    ax.yaxis.set_major_formatter(FuncFormatter(format))
    ax.tick_params(axis="x", labelsize=TICK_SIZE)
    ax.tick_params(axis="y", labelsize=TICK_SIZE)

    # Add grid lines
    ax.grid(True, **GRID_ARGS)

    # Save the plot if a save_path is provided
    print(f"Saving plot to {save_path}")
    fig.savefig(save_path, bbox_inches="tight")


def plot_stats(stats: Dict[str, Dict[int, int]], out_dir: str):
    """
    Save plots of the statistics in the stats dictionary.

    :param stats: Dictionary containing the statistics to plot.
    :param out_dir: Directory to save the plots.
    """

    def dict_to_cumulative(dict_: Dict[int, int]) -> Dict[int, int]:
        """Transpose a count dictionary into a cumulative-count dictionary"""

        sorted_keys = sorted(dict_.keys())
        cumulative_sum = 0
        cumulative_dict = {k: 0 for k in sorted_keys}

        # Iterate over the sorted keys and calculate the cumulative sum
        for key in sorted_keys:
            cumulative_sum += dict_[key]
            cumulative_dict[key] = cumulative_sum

        return cumulative_dict

    for cumulative in [False, True]:

        for name, data in stats.items():

            if cumulative:
                data = dict_to_cumulative(data)
                name += "_cumulative"

            line_plot(
                data=data,
                title=f"{name}",
                xlabel="Values",
                ylabel="Count",
                save_path=path.join(out_dir, f"{name}.png"),
            )


def mapping_function(elements: Dict) -> Dict[str, Any]:
    """
    :param elements:  A dictionary containing the elements of the dataset.
    """
    
    stats = {
        "abstract_length": [
            OAGKXItem.from_data(
                title="", abstract=a, keywords_str=""
            ).abstract_word_count
            for a in elements["abstract"]
        ],
        "title_length": [
            OAGKXItem.from_data(title=t, abstract="", keywords_str="").title_word_count
            for t in elements["title"]
        ],
        "keywords_count": [
            len(OAGKXItem.from_data(title="", abstract="", keywords_str=k).keywords)
            for k in elements["keywords"]
        ],
        # 'keywords_in_abstract': len(item.keywords_in_abstract),
    }
    return stats


def main(args: argparse.Namespace):

    out_dir = args.out_dir
    data_dir = args.data_dir

    # Create output directory
    os.makedirs(out_dir, exist_ok=True)

    # Load dataset
    dataset = load_oagkx_dataset(
        data_dir=data_dir,
        split_size=(1.0, 0.0, 0.0),
        # just_one_file=True
    )["train"]

    # dataset = dataset.select(range(100))

    # Compute stats
    stats_dataset = dataset.map(
        mapping_function, remove_columns=dataset.column_names, batched=True
    )

    print(stats_dataset)
    print(stats_dataset[0])

    # Extract stats count per column
    stats_count = {
        k: dict(sorted(Counter(stats_dataset[k]).items()))
        for k in stats_dataset.column_names
    }

    # Save stats for future use
    with open(path.join(out_dir, "stats.pkl"), "wb") as f:
        pickle.dump(stats_count, f)

    # Plot stats
    plot_stats(stats=stats_count, out_dir=out_dir)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/OAGKX")
    parser.add_argument("--out_dir", type=str, default="figures")
    args = parser.parse_args()
    main(args)

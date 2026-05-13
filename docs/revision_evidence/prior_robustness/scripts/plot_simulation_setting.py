#!/usr/bin/env python
"""Plot the controlled prior-robustness simulation design and results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
EVIDENCE_DIR = ROOT / "rebuttal_pipeline" / "evidence" / "prior_robustness"
SUMMARY_PATH = EVIDENCE_DIR / "tables" / "simulation_prior_robustness_summary_full.csv"
RESULTS_PATH = EVIDENCE_DIR / "tables" / "simulation_prior_robustness_results_full.csv"
FIGURE_DIR = EVIDENCE_DIR / "figures"
SUPPLY_FIG_DIR = ROOT / "GenomeBio" / "supply_figs"
FIGURE_NAME = "prior_robustness_simulation_setting.pdf"


def draw_route_panel(ax: plt.Axes) -> None:
    ax.set_axis_off()
    ax.set_xlim(-0.35, 3.45)
    ax.set_ylim(-1.85, 1.45)

    nodes = {
        "Centric target": (0.0, 0.0, "#4477AA"),
        "Inner node": (1.0, 0.0, "#66CCEE"),
        "Middle node": (2.0, 0.0, "#66CCEE"),
        "Informative source": (3.0, 0.0, "#228833"),
        "Decoy target": (0.0, -1.0, "#CC6677"),
        "Decoy source": (3.0, -1.0, "#BBBBBB"),
    }
    for label, (x, y, color) in nodes.items():
        ax.scatter(x, y, s=680, color=color, edgecolor="black", linewidth=0.9, zorder=3)
        ax.text(x, y - 0.27, label, ha="center", va="top", fontsize=8)

    def edge(start: tuple[float, float], end: tuple[float, float], color: str, style: str = "-") -> None:
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops=dict(arrowstyle="-", color=color, lw=2.0, linestyle=style),
            zorder=2,
        )

    edge((3.0, 0.0), (2.0, 0.0), "#228833")
    edge((2.0, 0.0), (1.0, 0.0), "#228833")
    edge((1.0, 0.0), (0.0, 0.0), "#228833")
    edge((3.0, -1.0), (0.0, -1.0), "#999999", "--")

    ax.text(1.5, 0.45, "Correct prior routes signal", ha="center", fontsize=9, color="#228833")
    ax.text(
        1.5,
        -1.62,
        "Wrong decoy prior:\nno short route to sources",
        ha="center",
        va="center",
        fontsize=9,
        color="#882255",
    )
    ax.text(
        1.5,
        1.17,
        "Synthetic modules place class signal in source nodes\nthree graph hops from true centric targets",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
    )


def draw_result_panel(ax: plt.Axes) -> None:
    summary = pd.read_csv(SUMMARY_PATH).set_index("condition")
    results = pd.read_csv(RESULTS_PATH)
    rows = [
        ("Correct\ngraph", "correct_graph", "#4477AA"),
        ("Depth 1", "correct_graph_depth1", "#88CCEE"),
        ("Wrong\ndecoy", "wrong_graph_decoy", "#CC6677"),
        ("Bad\nfeatures", "bad_feature", "#AA3377"),
        ("Random\ngraph", "random_graph", "#DDCC77"),
        ("Partial\nrandom", "partial_inner_random", "#228833"),
    ]
    means = [summary.loc[key, "test_roc_auc_mean"] for _, key, _ in rows]
    mins = [summary.loc[key, "test_roc_auc_min"] for _, key, _ in rows]
    maxs = [summary.loc[key, "test_roc_auc_max"] for _, key, _ in rows]
    colors = [color for _, _, color in rows]

    x = np.arange(len(rows))
    rng = np.random.default_rng(20260512)
    for idx, (_, key, color) in enumerate(rows):
        values = results.loc[results["condition"] == key, "test_roc_auc"].dropna().to_numpy()
        jitter = rng.uniform(-0.14, 0.14, size=len(values))
        ax.scatter(
            np.full(len(values), idx) + jitter,
            values,
            s=22,
            color=color,
            alpha=0.65,
            edgecolor="white",
            linewidth=0.35,
            zorder=3,
        )
        ax.vlines(idx, mins[idx], maxs[idx], color="black", linewidth=1.2, zorder=4)
        ax.hlines([mins[idx], maxs[idx]], idx - 0.09, idx + 0.09, color="black", linewidth=1.2, zorder=4)
        ax.scatter(idx, means[idx], s=62, color=color, edgecolor="black", linewidth=0.8, zorder=5)
    ax.axhline(0.5, color="#444444", lw=1.0, linestyle="--")
    y_upper = min(1.05, max(maxs) + 0.05)
    ax.set_ylim(0.35, y_upper)
    ax.set_ylabel("Test ROC-AUC")
    ax.set_xticks(x)
    ax.set_xticklabels([label for label, _, _ in rows], fontsize=8)
    ax.set_title("Prior-quality stress test", fontsize=11, fontweight="bold")
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.7)
    ax.set_axisbelow(True)

    label_ceiling = ax.get_ylim()[1] - 0.035
    for idx, mean in enumerate(means):
        label_y = min(maxs[idx] + 0.025, label_ceiling)
        ax.text(idx, label_y, f"{mean:.2f}", ha="center", va="bottom", fontsize=8)


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    SUPPLY_FIG_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6), gridspec_kw={"width_ratios": [1.2, 1.0]})
    draw_route_panel(axes[0])
    draw_result_panel(axes[1])
    fig.tight_layout(w_pad=2.2)

    for out_dir in [FIGURE_DIR, SUPPLY_FIG_DIR]:
        fig.savefig(out_dir / FIGURE_NAME, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

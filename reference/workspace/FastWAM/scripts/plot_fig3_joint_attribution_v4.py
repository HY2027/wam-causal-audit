#!/usr/bin/env python3
"""Figure 3 v4: Figure-2-v5-matched styling with frozen Figure 3 data.

This script changes only typography, layout, and graphical presentation.  It
reads the same audited tables as Figure 3 v3 and does not recompute estimates,
confidence intervals, source aggregates, or validation membership.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from main_figure_common import ASSETS, DATA_ROOT, require, save_figure, sha256
from main_figure_layout_v2 import add_panel_titles, configure_layout_v2, style_legend


F1 = ASSETS / "fig3_f1_attribution_data.csv"
F3 = ASSETS / "fig3_f3g_attribution_data.csv"
RESTORE = ASSETS / "fig3_restoration_source_data.csv"
F3_RAW = (
    DATA_ROOT
    / "experiments/followup_s1_s2_s3/s1_f3g_attribution/v3/02_per_case_attribution.csv"
)

# Immutable-input identities used by Figure 3 v3.
EXPECTED_SHA256 = {
    F1: "6589758cb93eed300f92ed850f5212cd13d3bda96e4cb178fd110bd6ba0070f3",
    F3: "756065d87ac367415666712802ff8567317b700236432ae5d4464b7eb7a99754",
    RESTORE: "f1490743b28a935dca866a5d2eee80814ba8cf856f69c20172505da2ff1962ff",
    F3_RAW: "3c3a5bad447b8f97576799248c269b1c47d9b7c2471f8d9d596507546ab5e228",
}

SERIES_ORDER = [
    "Node (propagation allowed)",
    "Strict current",
    "Strict future",
]
COLORS = {
    "Node (propagation allowed)": "#4D4D4D",
    "Strict current": "#0072B2",
    "Strict future": "#D55E00",
}
TITLES = [
    "(a) F1 · PreGrasp",
    "(b) F3-G · validation",
    "(c) Late vs. early restoration",
]


def verify_frozen_inputs(f1: pd.DataFrame, f3: pd.DataFrame, restore: pd.DataFrame) -> None:
    """Fail closed if any v3 plotting input or registered row set changed."""
    for path, expected in EXPECTED_SHA256.items():
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"Frozen input changed: {path}\nexpected {expected}\nactual   {actual}")
    if len(f1) != 14 or len(f3) != 12 or len(restore) != 20:
        raise ValueError("Frozen Figure 3 plotting-table row count changed")
    if sorted(f1.loc[f1.series == SERIES_ORDER[0], "signed_dose_cm"].tolist()) != [-4.0, 4.0]:
        raise ValueError("F1 node measurements must remain restricted to ±4 cm")
    if f1["series"].drop_duplicates().tolist() != [
        "Strict current", "Strict future", "Node (propagation allowed)"
    ]:
        raise ValueError("F1 frozen condition order changed")
    if f3["series"].drop_duplicates().tolist() != SERIES_ORDER:
        raise ValueError("F3-G frozen condition order changed")


def plot_attribution_panel(ax, data: pd.DataFrame, *, disconnect_node: bool) -> None:
    """Draw one frozen attribution panel with Figure 2 v5 visual weights."""
    for series in SERIES_ORDER:
        q = data[data.series == series].sort_values("signed_dose_cm")
        if q.empty:
            continue
        x = q.signed_dose_cm.to_numpy(float)
        y = q.estimate.to_numpy(float)
        lo = q.ci_low.to_numpy(float)
        hi = q.ci_high.to_numpy(float)
        node = series == SERIES_ORDER[0]
        ax.errorbar(
            x,
            y,
            yerr=[y - lo, hi - y],
            color=COLORS[series],
            marker="D" if node else "o",
            linestyle="none" if node and disconnect_node else (":" if node else "-"),
            linewidth=1.6,
            markersize=4.2,
            elinewidth=0.8,
            capthick=0.8,
            capsize=1.7,
            label=series,
            zorder=4 if node else 3,
        )
    ax.axhline(0, color="0.60", linewidth=0.65, zorder=0)
    ax.set_xlabel("Signed dose (cm)")
    ax.set_ylabel("Normalized radial effect", fontsize=8.2, labelpad=2.5)
    ax.grid(axis="y", color="0.92", linewidth=0.45, zorder=0)


def main() -> None:
    require([F1, F3, RESTORE, F3_RAW])
    f1 = pd.read_csv(F1)
    f3 = pd.read_csv(F3)
    restore = pd.read_csv(RESTORE)
    verify_frozen_inputs(f1, f3, restore)

    # These 98 validation rows are the identical raw points shown in v3.
    f3raw = pd.read_csv(F3_RAW)
    f3val = f3raw[f3raw.split == "validation"]
    if len(f3val) != 98:
        raise ValueError("Frozen F3-G validation-row count changed")

    # Exactly the same typography helper and canvas geometry as Figure 2 v5.
    configure_layout_v2()
    fig = plt.figure(figsize=(7.1, 2.72))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1])
    axes = [fig.add_subplot(gs[0, index]) for index in range(3)]

    plot_attribution_panel(axes[0], f1, disconnect_node=True)
    plot_attribution_panel(axes[1], f3, disconnect_node=False)

    # Preserve the v3 source-level validation points and deterministic jitter.
    rng = np.random.default_rng(20260922)
    for raw_series, color in (("R_node", "#4D4D4D"), ("R_C", "#0072B2"), ("R_F", "#D55E00")):
        for dose, q in f3val.groupby("signed_dose_cm"):
            axes[1].scatter(
                np.full(len(q), dose) + rng.uniform(-0.025, 0.025, len(q)),
                q[raw_series],
                s=3,
                color=color,
                alpha=0.055,
                edgecolors="none",
                zorder=1,
            )

    # Preserve all 20 source-level restoration pairs and both frozen summaries.
    ax = axes[2]
    for _, row in restore.iterrows():
        ax.plot(
            [0, 1],
            [row.early_error, row.late_error],
            color="#8A8A8A",
            alpha=0.42,
            linewidth=0.65,
            zorder=1,
        )
        ax.scatter(
            [0, 1],
            [row.early_error, row.late_error],
            s=9,
            color=["#56B4E9", "#009E73"],
            alpha=0.70,
            edgecolors="none",
            zorder=2,
        )
    ax.scatter(
        [0, 1],
        [restore.early_error.mean(), restore.late_error.mean()],
        s=38,
        marker="D",
        color=["#0072B2", "#009E73"],
        edgecolor="white",
        linewidth=0.75,
        zorder=6,
    )
    ax.set_xticks([0, 1], ["Early\nL0–4", "Late\nL25–29"])
    ax.set_ylabel("Residual error to full restoration", fontsize=8.2, labelpad=2.5)
    ax.grid(axis="y", color="0.92", linewidth=0.45, zorder=0)
    ax.set_ylim(top=max(ax.get_ylim()[1], restore.early_error.max() * 1.22))
    ax.text(
        0.96,
        0.88,
        f"n={len(restore)} source trajectories\nDonor-aggregated",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
        zorder=20,
        bbox={
            "boxstyle": "round,pad=0.22",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.90,
        },
    )

    # Match Figure 2 v5 exactly: equal columns, common title y, compact legend.
    fig.subplots_adjust(left=0.072, right=0.995, bottom=0.215, top=0.780, wspace=0.43)
    add_panel_titles(fig, axes, TITLES, y_pad=0.018)
    handles, labels = axes[0].get_legend_handles_labels()
    legend = fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=3,
        frameon=True,
        handlelength=1.40,
        handletextpad=0.45,
        columnspacing=0.85,
        borderaxespad=0.25,
        borderpad=0.25,
        labelspacing=0.20,
    )
    style_legend(legend)
    legend.get_frame().set_edgecolor("0.88")
    legend.get_frame().set_alpha(0.78)
    legend.get_frame().set_linewidth(0.30)

    save_figure(fig, "fig3_joint_attribution_v4", write_prefix_preview=False)
    plt.close(fig)


if __name__ == "__main__":
    main()

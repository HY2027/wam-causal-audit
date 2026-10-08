#!/usr/bin/env python3
"""Figure 2 v5: final typography and legend-spacing refinement.

This is a layout/label-only revision of v4.  It reads the same immutable
72-row plotting table and does not recompute responses or intervals.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from main_figure_common import ASSETS, MODEL_COLORS, MODEL_ORDER, identity, require, save_figure
from main_figure_layout_v2 import add_panel_titles, configure_layout_v2, style_legend

SOURCE = ASSETS / "fig2_physical_response_data.csv"
LAYOUT_AUDIT = ASSETS / "fig2_v5_layout_audit.md"
PANELS = [
    ("(a) F1 · PreGrasp", "Gripper-closing command response"),
    ("(b) F3 · PrePlace", "Object–goal radial response"),
    ("(c) F3 · Transport", "Object–goal radial response"),
]
EXPECTED_DOSES = [-4.0, -2.0, -1.0, 1.0, 2.0, 4.0]


def main() -> None:
    require([SOURCE])
    data = pd.read_csv(SOURCE)
    if data["model"].drop_duplicates().tolist() != MODEL_ORDER:
        raise ValueError("Model order in the immutable plotting table changed")
    for title, _ in PANELS:
        panel = data[data.panel == title]
        if sorted(panel.signed_dose_cm.unique().tolist()) != EXPECTED_DOSES:
            raise ValueError(f"Registered doses changed for {title}")

    radial = data[data.panel.isin([PANELS[1][0], PANELS[2][0]])]
    radial_low = float(radial.ci_low.min())
    radial_high = float(radial.ci_high.max())
    radial_margin = 0.08 * (radial_high - radial_low)
    radial_ylim = (radial_low - radial_margin, radial_high + radial_margin)

    configure_layout_v2()
    fig = plt.figure(figsize=(7.1, 2.72))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1])
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    legend_handles = None
    legend_labels = None

    for ax, (title, ylabel) in zip(axes, PANELS):
        panel = data[data.panel == title]
        for model in MODEL_ORDER:
            z = panel[panel.model == model].sort_values("signed_dose_cm")
            x = z.signed_dose_cm.to_numpy(float)
            y = z.estimate.to_numpy(float)
            lo = z.ci_low.to_numpy(float)
            hi = z.ci_high.to_numpy(float)
            ax.plot(
                x, y, marker="o", ms=4.2, lw=1.6,
                color=MODEL_COLORS[model], label=model, zorder=3,
            )
            ax.fill_between(
                x, lo, hi, color=MODEL_COLORS[model], alpha=.12,
                linewidth=0, zorder=1,
            )
        ax.axhline(0, color="0.60", lw=.65, zorder=0)
        ax.axvline(0, color="0.82", lw=.60, zorder=0)
        ax.set_xlabel("Signed physical displacement (cm)")
        ax.set_ylabel(ylabel, fontsize=8.2, labelpad=2.5)
        ax.set_xticks(EXPECTED_DOSES)
        ax.grid(axis="y", color="0.92", lw=.45, zorder=0)
        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

    axes[1].set_ylim(*radial_ylim)
    axes[2].set_ylim(*radial_ylim)

    # Move the axes/title band upward and the legend downward so the two bands
    # are compact without overlapping.  Titles still share one figure y.
    fig.subplots_adjust(left=.072, right=.995, bottom=.215, top=.780, wspace=.43)
    title_y, _ = add_panel_titles(fig, axes, [item[0] for item in PANELS], y_pad=.018)
    legend = fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(.5, .955),
        ncol=4,
        frameon=True,
        handlelength=1.40,
        handletextpad=.45,
        columnspacing=.85,
        borderaxespad=.25,
        borderpad=.25,
        labelspacing=.20,
    )
    style_legend(legend)
    legend.get_frame().set_edgecolor("0.88")
    legend.get_frame().set_alpha(.78)
    legend.get_frame().set_linewidth(.30)

    save_figure(fig, "fig2_physical_response_v5", write_prefix_preview=False)
    plt.close(fig)

    meta = identity(SOURCE)
    LAYOUT_AUDIT.write_text(
        "# Figure 2 v5 layout audit\n\n"
        "Status: **PASS_LAYOUT_AND_LABEL_ONLY**.\n\n"
        f"- Immutable input: `{meta['path']}` ({meta['bytes']} bytes; SHA-256 `{meta['sha256']}`).\n"
        "- No response, curve, confidence interval, model color/order, registered dose, or statistic changed.\n"
        "- Original v4 legend: `loc=upper center`, `bbox_to_anchor=(0.5, 0.995)`, "
        "`columnspacing=1.05`, default `handletextpad=0.8`, `handlelength=1.55`; axes top 0.745 and title y 0.763.\n"
        "- New v5 legend: `loc=upper center`, `bbox_to_anchor=(0.5, 0.955)`, "
        "`columnspacing=0.85`, `handletextpad=0.45`, `handlelength=1.40`, "
        "`borderaxespad=0.25`; axes top 0.780. The frame edge is 0.88 gray at alpha 0.78.\n"
        f"- New common panel-title y coordinate: {title_y:.6f}.\n"
        "- Original y-label source size: 8.8 pt. New y-label source size: 8.2 pt "
        "(6.8% reduction), with labelpad reduced to 2.5 pt. Tick and panel-title sizes are unchanged.\n"
        "- Panel (a) readout: after action de-normalization, "
        "`ClosingDrive = sum_t max(-g_t, 0)`, where `g_t` is the gripper action/command channel. "
        "It is neither a robot--object radial translation nor force or executed displacement.\n"
        "- Panel (a) label changed from `Closing-command response` to "
        "`Gripper-closing command response` to make the commanded channel explicit.\n"
        "- Panels (b,c) use `sum_t <translation_t, u_object-to-goal>` and were shortened from "
        "`Object--goal radial command response` to `Object--goal radial response`; their common "
        "command-space definition remains explicit in the caption.\n"
        f"- Shared F3 y limits remain [{radial_ylim[0]:.6g}, {radial_ylim[1]:.6g}].\n"
    )


if __name__ == "__main__":
    main()

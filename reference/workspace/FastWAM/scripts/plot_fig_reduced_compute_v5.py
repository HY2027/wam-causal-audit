#!/usr/bin/env python3
"""Reduced-compute Figure v5: style/layout revision of frozen Figure 5 v3.

No experiment, fit, bootstrap, trajectory aggregation, smoothing, filtering, or
statistical estimator is run here.  The script reads the same audited plotting
tables and frozen summary rows used by Figure 5 v3, verifies their identities,
and changes only graphical presentation and wording.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from main_figure_common import ASSETS, DATA_ROOT, require, save_figure, sha256
from main_figure_layout_v2 import add_panel_titles, configure_layout_v2, style_legend


SCALING = DATA_ROOT / "phase1_evidence_mechanism_posthoc_v1_20260909"
TIMING = ASSETS / "budget_confirmation_timing_audit.csv"
EXAMPLE = ASSETS / "budget_response_example.csv"
RESID = ASSETS / "fig5_trajectory_residuals.csv"
STATS = SCALING / "joint_scaling_residual_statistics_units_corrected.csv"

EXPECTED_SHA256 = {
    TIMING: "a212cc2ce92f9639a444d118d0247a049b66c9c5b14fa4077bcfd8eaef383b95",
    EXAMPLE: "e24f066a7020dd4595ade8b75c3f2c435a9b8191cbf3b60926d7973cc54d5f90",
    RESID: "31f88e1fc5d68cd23499c1238f12ed5aadb8b893566715e27452750fb78d50e8",
    STATS: "174a5795e3acf1419458d359e9de72807ccb21d0e147dff0ac2da850d0f31252",
}

EXPECTED_CASE = "JNT__task5__demo000418__TRANSPORT__F3G_RADIAL__p1cm"
EXPECTED_UNIT = "normalized policy translation-command unit"
EXPECTED_LAMBDA = 0.9081050896952964
EXPECTED_Q = 3.288657662241220
EXPECTED_Q_LOW = -9.051067393136970
EXPECTED_Q_HIGH = 13.629866820843880
EXPECTED_RMSE_UNSCALED = 0.0079440745462477
EXPECTED_RMSE_CROSSFIT = 0.0078123558407253

TITLES = [
    "(a) Paired policy computation",
    "(b) F3-G · +1 cm response",
    "(c) Held-fold trajectory residuals",
]


def frozen_summary(stats: pd.DataFrame) -> tuple[float, float, float]:
    """Read, but do not re-estimate, the registered Q and interval rows."""
    q = stats[
        (stats.endpoint == "FIT_BASIS")
        & (stats.scope == "ALL")
        & (stats.stratum.astype(str) == "ALL")
    ].set_index("metric")
    required = {"rmse_unscaled", "rmse_scaled_crossfit", "fraction_unscaled_SSE_removed"}
    if not required.issubset(q.index):
        raise ValueError("Frozen shared-gain summary is incomplete")
    Q = 100.0 * float(q.loc["fraction_unscaled_SSE_removed", "estimate"])
    q_low = 100.0 * float(q.loc["fraction_unscaled_SSE_removed", "ci_low"])
    q_high = 100.0 * float(q.loc["fraction_unscaled_SSE_removed", "ci_high"])
    checks = (
        np.isclose(Q, EXPECTED_Q),
        np.isclose(q_low, EXPECTED_Q_LOW),
        np.isclose(q_high, EXPECTED_Q_HIGH),
        np.isclose(float(q.loc["rmse_unscaled", "estimate"]), EXPECTED_RMSE_UNSCALED),
        np.isclose(float(q.loc["rmse_scaled_crossfit", "estimate"]), EXPECTED_RMSE_CROSSFIT),
    )
    if not all(checks):
        raise ValueError("Frozen Q/CI/RMSE values changed")
    return Q, q_low, q_high


def verify_inputs(
    timing: pd.DataFrame,
    example: pd.DataFrame,
    resid: pd.DataFrame,
    stats: pd.DataFrame,
) -> tuple[float, float, float]:
    """Fail closed on any data, pairing, semantics, or registered-value drift."""
    for path, expected in EXPECTED_SHA256.items():
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"Frozen input changed: {path}\nexpected {expected}\nactual   {actual}")

    if len(timing) != 50 or timing.trajectory_id.nunique() != 50:
        raise ValueError("Panel (a) must retain 50 unique paired trajectories")
    if not timing[["native_completed", "k5_completed"]].to_numpy(bool).all():
        raise ValueError("The frozen recipient-completion audit is no longer 50/50 in both arms")
    saving = 1.0 - timing.k5_cumulative_policy_time / timing.native_cumulative_policy_time
    if not np.allclose(saving, timing.saving_fraction, rtol=0, atol=2e-15):
        raise ValueError("Paired cumulative policy-time saving definition changed")
    if round(100.0 * float(timing.saving_fraction.median()), 1) != 23.7:
        raise ValueError("Frozen paired median saving no longer rounds to 23.7%")

    if len(example) != 32 or example.chunk_position.tolist() != list(range(32)):
        raise ValueError("Panel (b) must retain all 32 predicted action-chunk positions")
    if example.source_id.nunique() != 1 or example.source_id.iloc[0] != EXPECTED_CASE:
        raise ValueError("Panel (b) stable-ID F3-G example changed")
    if set(example.factor) != {"F3-G radial target displacement"} or set(example.signed_dose_cm) != {1.0}:
        raise ValueError("Panel (b) factor or +1 cm dose changed")
    if set(example.contrast) != {"natural donor-minus-recipient budget-condition increment"}:
        raise ValueError("Panel (b) response contrast changed")
    if set(example.unit) != {EXPECTED_UNIT}:
        raise ValueError("Panel (b) command unit changed")
    if not example.heldout_lambda.eq(EXPECTED_LAMBDA).all():
        raise ValueError("Panel (b) held-fold gain changed")
    if not example.gain_training_trajectory_count.eq(40).all() or not example.fold_id.eq(0).all():
        raise ValueError("Panel (b) held-fold mapping changed")
    if not np.allclose(
        example.gain_adjusted_native_response,
        EXPECTED_LAMBDA * example.native_response,
        rtol=0,
        atol=2e-16,
    ):
        raise ValueError("Panel (b) stored gain-adjusted vector changed")

    if len(resid) != 50 or resid.trajectory_id.nunique() != 50 or resid.candidate_id.nunique() != 50:
        raise ValueError("Panel (c) must retain one point per 50 held-out trajectories")
    if sorted(resid.fold.unique().tolist()) != [0, 1, 2, 3, 4]:
        raise ValueError("Panel (c) held-fold coverage changed")
    if not resid.donor_endpoint_rows.eq(12).all():
        raise ValueError("Panel (c) registered residual-vector coverage changed")

    return frozen_summary(stats)


def main() -> None:
    require([TIMING, EXAMPLE, RESID, STATS])
    timing = pd.read_csv(TIMING)
    example = pd.read_csv(EXAMPLE)
    resid = pd.read_csv(RESID)
    stats = pd.read_csv(STATS)
    Q, q_low, q_high = verify_inputs(timing, example, resid, stats)

    # Exact typography and title/legend contract used by Figures 2 v5 and 3 v4.
    configure_layout_v2()
    fig = plt.figure(figsize=(7.1, 2.72))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.30, 0.40, 0.30])
    axes = [fig.add_subplot(gs[0, index]) for index in range(3)]

    # (a) Preserve every paired trajectory and the two median summaries.
    ax = axes[0]
    for _, row in timing.iterrows():
        ax.plot(
            [0, 1],
            [row.native_cumulative_policy_time, row.k5_cumulative_policy_time],
            color="0.55",
            alpha=0.32,
            linewidth=0.65,
            zorder=1,
        )
        ax.scatter(
            [0, 1],
            [row.native_cumulative_policy_time, row.k5_cumulative_policy_time],
            s=9,
            color=["#222222", "#D55E00"],
            alpha=0.45,
            edgecolors="none",
            zorder=2,
        )
    ax.scatter(
        [0, 1],
        [timing.native_cumulative_policy_time.median(), timing.k5_cumulative_policy_time.median()],
        s=38,
        marker="D",
        color=["#222222", "#D55E00"],
        edgecolor="white",
        linewidth=0.75,
        zorder=6,
    )
    ax.set_xticks([0, 1], ["Native\n$K{=}10$", "$K{=}5$"])
    ax.set_ylabel("Cumulative policy computation (s)", fontsize=8.2, labelpad=2.5)
    ax.grid(axis="y", color="0.92", linewidth=0.45, zorder=0)
    y0, y1 = ax.get_ylim()
    max_time = timing[["native_cumulative_policy_time", "k5_cumulative_policy_time"]].max().max()
    ax.set_ylim(y0, max(y1, max_time * 1.38))
    ax.text(
        0.05,
        0.96,
        "Median saving = 23.7%\n50/50 continuations completed",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        color="0.18",
        zorder=20,
    )

    # (b) Preserve the 32 stored positions and all three unsmoothed vectors.
    ax = axes[1]
    x = example.chunk_position
    native_line, = ax.plot(
        x,
        example.native_response,
        color="#222222",
        linewidth=1.6,
        label=r"Native $K{=}10$",
        zorder=3,
    )
    gain_line, = ax.plot(
        x,
        example.gain_adjusted_native_response,
        color="#0072B2",
        linewidth=1.4,
        linestyle="--",
        label="Gain-adjusted native",
        zorder=3,
    )
    k5_line, = ax.plot(
        x,
        example.k5_response,
        color="#D55E00",
        linewidth=1.6,
        label=r"$K{=}5$",
        zorder=3,
    )
    ax.axhline(0, color="0.60", linewidth=0.65, zorder=0)
    ax.set_xlim(0, 31)
    ax.set_xticks([0, 10, 20, 30])
    lower, _ = ax.get_ylim()
    peak = example[["native_response", "gain_adjusted_native_response", "k5_response"]].max().max()
    ax.set_ylim(lower, max(0.075, peak * 1.16))
    ax.set_xlabel("Predicted action-chunk position")
    ax.set_ylabel(
        "Radial command response\n(normalized command units)",
        fontsize=8.2,
        labelpad=2.5,
    )
    ax.grid(axis="y", color="0.92", linewidth=0.45, zorder=0)

    # (c) Identical x/y ranges make the no-improvement reference exactly 45°.
    ax = axes[2]
    limit = max(resid.unscaled_rmse.max(), resid.crossfit_rmse.max()) * 1.07
    ax.plot([0, limit], [0, limit], color="0.45", linestyle="--", linewidth=0.8, zorder=1)
    ax.scatter(
        resid.unscaled_rmse,
        resid.crossfit_rmse,
        s=16,
        alpha=0.62,
        color="#7A5AA6",
        edgecolors="none",
        zorder=3,
    )
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Unscaled response RMSE")
    ax.set_ylabel("Cross-fitted gain RMSE", fontsize=8.2, labelpad=2.5)
    ax.grid(color="0.92", linewidth=0.45, zorder=0)
    ax.text(
        0.96,
        0.06,
        rf"$Q = {Q:.2f}\%$" + "\n" + rf"95% CI $[{q_low:.2f}\%,\,{q_high:.2f}\%]$",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.5,
        color="0.18",
        zorder=20,
    )

    # Match Figure 3 v4's axes band, common title baseline, and top legend.
    fig.subplots_adjust(left=0.072, right=0.995, bottom=0.215, top=0.780, wspace=0.46)
    add_panel_titles(fig, axes, TITLES, y_pad=0.018)
    legend = fig.legend(
        [native_line, gain_line, k5_line],
        [native_line.get_label(), gain_line.get_label(), k5_line.get_label()],
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

    save_figure(fig, "fig_reduced_compute_v5", write_prefix_preview=False)
    plt.close(fig)


if __name__ == "__main__":
    main()

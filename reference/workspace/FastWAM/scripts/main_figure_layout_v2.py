#!/usr/bin/env python3
"""Shared layout-only helpers for the v2 main-paper figures."""
from __future__ import annotations

from collections.abc import Iterable, Sequence

import matplotlib as mpl


SERIF_FALLBACKS = [
    "TeX Gyre Termes",
    "Times New Roman",
    "Nimbus Roman",
]

# These source sizes are chosen for the widest current main-figure PDF.  At the
# ICLR figure* insertion width (\textwidth in the repository preview), ticks
# and legends remain at least 7.5 pt after scaling.
PANEL_TITLE_SIZE = 9.5
AXIS_LABEL_SIZE = 8.8
TICK_LABEL_SIZE = 8.5
LEGEND_SIZE = 8.5
ANNOTATION_SIZE = 8.5


def configure_layout_v2() -> None:
    """Apply the typography contract without changing plotting data."""
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": SERIF_FALLBACKS,
        "font.size": TICK_LABEL_SIZE,
        "mathtext.fontset": "stix",
        "axes.labelsize": AXIS_LABEL_SIZE,
        "xtick.labelsize": TICK_LABEL_SIZE,
        "ytick.labelsize": TICK_LABEL_SIZE,
        "legend.fontsize": LEGEND_SIZE,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })


def add_panel_titles(
    fig,
    axes: Sequence,
    titles: Sequence[str],
    y_pad: float = 0.012,
    fontsize: float = PANEL_TITLE_SIZE,
    fontweight: str = "bold",
):
    """Place collinear, axis-centred panel titles in figure coordinates."""
    if len(axes) != len(titles):
        raise ValueError("axes and titles must have the same length")
    fig.canvas.draw()
    boxes = [ax.get_position() for ax in axes]
    title_y = max(box.y1 for box in boxes) + y_pad
    artists = []
    for box, title in zip(boxes, titles):
        artists.append(fig.text(
            (box.x0 + box.x1) / 2.0,
            title_y,
            title,
            ha="center",
            va="bottom",
            fontsize=fontsize,
            fontweight=fontweight,
        ))
    return title_y, artists


def add_panel_group_titles(
    fig,
    axes_groups: Sequence[Sequence],
    titles: Sequence[str],
    y_pad: float = 0.012,
    fontsize: float = PANEL_TITLE_SIZE,
    fontweight: str = "bold",
):
    """Place titles centred over multi-axes column groups at one shared y."""
    if len(axes_groups) != len(titles):
        raise ValueError("axes_groups and titles must have the same length")
    fig.canvas.draw()
    group_boxes = [[ax.get_position() for ax in group] for group in axes_groups]
    title_y = max(box.y1 for boxes in group_boxes for box in boxes) + y_pad
    artists = []
    for boxes, title in zip(group_boxes, titles):
        left = min(box.x0 for box in boxes)
        right = max(box.x1 for box in boxes)
        artists.append(fig.text(
            (left + right) / 2.0,
            title_y,
            title,
            ha="center",
            va="bottom",
            fontsize=fontsize,
            fontweight=fontweight,
        ))
    return title_y, artists


def place_safe_annotation(
    ax,
    text: str,
    loc: str = "upper left",
    *,
    fontsize: float = ANNOTATION_SIZE,
    x_pad: float = 0.04,
    y_pad: float = 0.04,
    **kwargs,
):
    """Place a consistently styled annotation in a low-risk axes corner."""
    locations = {
        "upper left": (x_pad, 1.0 - y_pad, "left", "top"),
        "upper right": (1.0 - x_pad, 1.0 - y_pad, "right", "top"),
        "lower left": (x_pad, y_pad, "left", "bottom"),
        "lower right": (1.0 - x_pad, y_pad, "right", "bottom"),
        "center left": (x_pad, 0.52, "left", "center"),
        "center right": (1.0 - x_pad, 0.52, "right", "center"),
    }
    if loc not in locations:
        raise ValueError(f"Unsupported annotation location: {loc}")
    x, y, ha, va = locations[loc]
    style = {
        "transform": ax.transAxes,
        "ha": ha,
        "va": va,
        "fontsize": fontsize,
        "zorder": 20,
        "bbox": {
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.85,
        },
    }
    style.update(kwargs)
    return ax.text(x, y, text, **style)


def style_legend(legend) -> None:
    """Apply the compact, minimally framed legend style."""
    if legend is None:
        return
    legend.set_zorder(30)
    frame = legend.get_frame()
    frame.set_facecolor("white")
    frame.set_alpha(0.88)
    frame.set_edgecolor("0.82")
    frame.set_linewidth(0.35)


def title_geometry(axes: Iterable, title_y: float) -> dict:
    """Serializable geometry used by the layout audit."""
    result = {"title_y": float(title_y), "panels": []}
    for ax in axes:
        box = ax.get_position()
        result["panels"].append({
            "axes_left": float(box.x0),
            "axes_right": float(box.x1),
            "axes_bottom": float(box.y0),
            "axes_top": float(box.y1),
            "title_x": float((box.x0 + box.x1) / 2.0),
        })
    return result

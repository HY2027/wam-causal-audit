#!/usr/bin/env python3
"""Shared, read-only helpers for the audited WAM main-paper figures."""
from __future__ import annotations
from wam_causal_audit.paths import resolve as _release_path

import hashlib
import json
from pathlib import Path

import matplotlib as mpl
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(_release_path('@DATA@/wam_factor_routing_v5'))
ASSETS = ROOT / "figure_assets"
FIGURES = ROOT / "figures"

MODEL_ORDER = ["Direct", "Joint", "IDM", "ImageWAM"]
MODEL_KEY = {"Direct": "direct", "Joint": "joint", "IDM": "idm", "ImageWAM": "imagewam"}
MODEL_COLORS = {"Direct": "#0072B2", "Joint": "#D55E00", "IDM": "#009E73", "ImageWAM": "#CC79A7"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def identity(path: Path) -> dict:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def require(paths) -> None:
    missing = [str(Path(p)) for p in paths if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError("Missing immutable inputs:\n" + "\n".join(missing))


def configure_style(font_size: float = 7.5) -> None:
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["TeX Gyre Termes", "Times New Roman", "Nimbus Roman"],
        "mathtext.fontset": "stix",
        "font.size": max(font_size, 8.5),
        "axes.titlesize": max(font_size + 0.7, 9.5), "axes.titleweight": "bold",
        "axes.labelsize": max(font_size, 8.8), "legend.fontsize": max(font_size - 0.5, 8.5),
        "xtick.labelsize": max(font_size - 0.4, 8.5), "ytick.labelsize": max(font_size - 0.4, 8.5),
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    })


def save_figure(fig, stem: str, dpi: int = 300, write_prefix_preview: bool = True) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    ASSETS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / f"{stem}.png", dpi=dpi, bbox_inches="tight")
    # The source PDF is already rendered at the 7-inch ICLR full-column-pair width.
    (ASSETS / f"{stem}_iclr_preview.pdf").write_bytes((FIGURES / f"{stem}.pdf").read_bytes())
    prefix = stem.split("_", 1)[0]
    if write_prefix_preview and prefix in {"fig2", "fig3", "fig4", "fig5"}:
        (ASSETS / f"{prefix}_iclr_preview.pdf").write_bytes((FIGURES / f"{stem}.pdf").read_bytes())


def dump_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def task_cluster_bootstrap(frame, value: str, seed: int, reps: int = 10_000):
    """Fixed-task equal-weight bootstrap, resampling registered base states within task."""
    tasks = sorted(frame["task_id"].unique())
    task_values = [frame.loc[frame.task_id == task, value].dropna().to_numpy(float) for task in tasks]
    if not task_values or any(len(v) == 0 for v in task_values):
        raise ValueError("Bootstrap cell has an empty task")
    estimate = float(np.mean([v.mean() for v in task_values]))
    rng = np.random.default_rng(seed)
    # Vectorized but exactly the same fixed-task, within-task resampling design.
    task_boots = []
    for values in task_values:
        indices = rng.integers(0, len(values), size=(reps, len(values)))
        task_boots.append(values[indices].mean(axis=1))
    boots = np.mean(np.stack(task_boots, axis=1), axis=1)
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return estimate, float(lo), float(hi), len(tasks)

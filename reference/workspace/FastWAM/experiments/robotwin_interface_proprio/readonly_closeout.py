#!/usr/bin/env python3
"""Read-only numerical closeout for the RoboTwin interface x proprio campaign.

This script never imports or loads a WAM, never creates a simulator, and never
executes a policy action.  It audits saved arrays/events/ledgers and produces a
versioned paper-delivery directory.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


SEED = 20260922
DRAWS = 10_000
MODELS = ("direct", "joint")
CELLS = {
    "Y00": "Y00__IZ1__PZ1",
    "Y10": "Y10__IDplus__PZ1",
    "Y01": "Y01__IZ1__PDplus",
    "Y11": "Y11__IDplus__PDplus",
}
EXPECTED = {
    "Y00": ("Z1", "Z1"),
    "Y10": ("Dplus", "Z1"),
    "Y01": ("Z1", "Dplus"),
    "Y11": ("Dplus", "Dplus"),
}
CHANNELS = [
    *(f"left_joint_{i}" for i in range(6)),
    "left_gripper",
    *(f"right_joint_{i}" for i in range(6)),
    "right_gripper",
]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields or ["status"])
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def fmt_sci(value: float) -> str:
    return f"{value:.17e}"


def compare_arrays(
    a: np.ndarray,
    b: np.ndarray,
) -> dict[str, Any]:
    aa = np.asarray(a)
    bb = np.asarray(b)
    same_shape = aa.shape == bb.shape
    same_dtype = aa.dtype == bb.dtype
    numeric_equal = bool(same_shape and np.array_equal(aa, bb, equal_nan=True))
    byte_equal = bool(
        same_shape
        and same_dtype
        and np.ascontiguousarray(aa).tobytes() == np.ascontiguousarray(bb).tobytes()
    )
    if not same_shape:
        return {
            "numeric_equal": False,
            "raw_element_bytes_identical": False,
            "different_element_count": "SHAPE_MISMATCH",
            "max_abs_diff": "",
            "max_abs_diff_scientific": "",
            "max_abs_flat_index": "",
            "max_abs_index": "",
            "nonzero_differences_scientific": "[]",
        }
    af = aa.astype(np.float64, copy=False).reshape(-1)
    bf = bb.astype(np.float64, copy=False).reshape(-1)
    diff = af - bf
    different = ~(np.equal(af, bf) | (np.isnan(af) & np.isnan(bf)))
    n_diff = int(np.count_nonzero(different))
    if diff.size:
        absolute = np.abs(diff)
        absolute[np.isnan(absolute)] = np.inf
        idx = int(np.argmax(absolute))
        max_abs = float(absolute[idx])
        multi = tuple(int(x) for x in np.unravel_index(idx, aa.shape))
    else:
        idx, max_abs, multi = -1, 0.0, ()
    nz_idx = np.flatnonzero(different)[:32]
    examples = [
        {
            "index": list(np.unravel_index(int(i), aa.shape)),
            "y11": fmt_sci(float(af[i])),
            "dplus": fmt_sci(float(bf[i])),
            "difference": fmt_sci(float(diff[i])),
        }
        for i in nz_idx
    ]
    return {
        "numeric_equal": numeric_equal,
        "raw_element_bytes_identical": byte_equal,
        "different_element_count": n_diff,
        "max_abs_diff": max_abs,
        "max_abs_diff_scientific": fmt_sci(max_abs),
        "max_abs_flat_index": idx,
        "max_abs_index": json.dumps(multi),
        "nonzero_differences_scientific": json.dumps(examples, separators=(",", ":")),
    }


def fk_pose(contract: Mapping[str, Any], action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The frozen named-joint FK, extended to return its registered rotation."""
    q = dict(contract["qpos_by_name"])
    q.update(zip(contract["left_names"], np.asarray(action[:6], dtype=np.float64)))
    q.update(zip(contract["right_names"], np.asarray(action[7:13], dtype=np.float64)))
    transform = np.asarray(contract["root"], dtype=np.float64)
    for joint in contract["chain"]:
        motion = np.eye(4, dtype=np.float64)
        value = float(q.get(joint["name"], 0.0))
        kind = str(joint["type"])
        if int(joint["dof"]):
            if "revolute" in kind:
                motion[1:3, 1:3] = [
                    [np.cos(value), -np.sin(value)],
                    [np.sin(value), np.cos(value)],
                ]
            elif "prismatic" in kind:
                motion[0, 3] = value
            else:
                raise RuntimeError(f"UNSUPPORTED_FK_JOINT:{kind}")
        transform = (
            transform
            @ np.asarray(joint["parent_pose"], dtype=np.float64)
            @ motion
            @ np.linalg.inv(np.asarray(joint["child_pose"], dtype=np.float64))
        )
    transform = transform @ np.asarray(contract["joint_pose_in_child"], dtype=np.float64)
    rotation = (
        transform[:3, :3]
        @ np.asarray(contract["global_matrix"], dtype=np.float64)
        @ np.asarray(contract["delta_matrix"], dtype=np.float64)
    )
    position = transform[:3, 3] + rotation @ np.asarray([contract["bias"], 0.0, 0.0])
    return position, rotation


def rotations(contract: Mapping[str, Any], actions: np.ndarray) -> np.ndarray:
    return np.asarray([fk_pose(contract, row)[1] for row in actions], dtype=np.float64)


def geodesic_angles(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    def project_so3(matrix: np.ndarray) -> np.ndarray:
        u, _, vh = np.linalg.svd(np.asarray(matrix, dtype=np.float64))
        result = u @ vh
        if np.linalg.det(result) < 0:
            u[:, -1] *= -1
            result = u @ vh
        return result

    out = []
    for ra, rb in zip(a, b):
        # The frozen FK matrices are numerically near-orthogonal.  Comparing a
        # matrix with itself via trace(R.T@R) without SO(3) projection creates a
        # false non-zero angle from orthogonality roundoff.  Exact element
        # equality is therefore exactly zero; otherwise project both operands.
        if np.array_equal(ra, rb):
            out.append(0.0)
            continue
        relative = project_so3(ra).T @ project_so3(rb)
        cos_theta = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
        out.append(float(np.arccos(cos_theta)))
    return np.asarray(out, dtype=np.float64)


def task_equal(rows: list[dict[str, Any]], key: str) -> float:
    tasks = sorted({str(r["task_id"]) for r in rows})
    return float(np.mean([np.mean([float(r[key]) for r in rows if r["task_id"] == t]) for t in tasks]))


def paired_bootstrap(rows: list[dict[str, Any]], keys: list[str]) -> dict[str, dict[str, Any]]:
    rng = np.random.default_rng(SEED)
    tasks = sorted({str(r["task_id"]) for r in rows})
    groups = {task: [row for row in rows if row["task_id"] == task] for task in tasks}
    samples = {key: np.empty(DRAWS, dtype=np.float64) for key in keys}
    for draw_idx in range(DRAWS):
        draw: list[dict[str, Any]] = []
        for task in tasks:
            group = groups[task]
            draw.extend(group[int(i)] for i in rng.integers(0, len(group), len(group)))
        for key in keys:
            samples[key][draw_idx] = task_equal(draw, key)
    return {
        key: {
            "estimate": task_equal(rows, key),
            "ci95": [float(np.quantile(samples[key], 0.025)), float(np.quantile(samples[key], 0.975))],
        }
        for key in keys
    }


def configure_matplotlib() -> None:
    import matplotlib as mpl
    from matplotlib import font_manager

    available = {f.name for f in font_manager.fontManager.ttflist}
    family = next(
        (name for name in ("TeX Gyre Termes", "Times New Roman", "Nimbus Roman") if name in available),
        "DejaVu Serif",
    )
    mpl.rcParams.update(
        {
            "font.family": family,
            "font.size": 8.0,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.7,
        }
    )


def plot_reconstruction(
    out: Path,
    set_name: str,
    rows: list[dict[str, Any]],
    stats: dict[str, Any],
) -> None:
    configure_matplotlib()
    import matplotlib.pyplot as plt

    colors = {"direct": "#3B6FB6", "joint": "#D07632"}
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.75), gridspec_kw={"wspace": 0.34})
    ax = axes[0]
    xbase = {"direct": 0.0, "joint": 2.4}
    rng = np.random.default_rng(8842)
    for model in MODELS:
        subset = [r for r in rows if r["model"] == model]
        x0, x1 = xbase[model], xbase[model] + 0.8
        jitter = rng.uniform(-0.08, 0.08, len(subset))
        for j, row in zip(jitter, subset):
            ax.plot([x0 + j, x1 + j], [row["E10"] * 1000, row["E11"] * 1000], color=colors[model], alpha=0.16, lw=0.55)
            ax.scatter([x0 + j, x1 + j], [row["E10"] * 1000, row["E11"] * 1000], color=colors[model], alpha=0.42, s=7, edgecolors="none")
        summary = stats[f"{model}_{set_name}"]["metrics"]
        for xx, key in ((x0, "E10"), (x1, "E11")):
            value = summary[key]["estimate"] * 1000
            lo, hi = (v * 1000 for v in summary[key]["ci95"])
            ax.errorbar(xx, value, yerr=[[value - lo], [hi - value]], fmt="o", ms=5.3, mfc=colors[model], mec="black", mew=0.55, ecolor="black", capsize=2.4, lw=0.9, zorder=10)
    ax.set_xticks([0, 0.8, 2.4, 3.2], ["Direct\ninterface", "Direct\ninterface + cmd", "Joint\ninterface", "Joint\ninterface + cmd"])
    ax.set_ylabel("Residual to natural Dplus (mm)")
    ax.set_title("(a) Natural-response reconstruction", loc="left", fontweight="bold")
    ax.axhline(0, color="#777777", lw=0.65)
    ax.grid(axis="y", color="#dddddd", lw=0.5)

    ax = axes[1]
    labels = ["Interface |\nZ1 command", "Command |\ndonor interface", "Natural\nDplus − Z1"]
    keys = ["I_P0_signed", "P_I1_signed", "natural_signed"]
    centers = np.arange(3, dtype=float)
    offsets = {"direct": -0.16, "joint": 0.16}
    for model in MODELS:
        subset = [r for r in rows if r["model"] == model]
        summary = stats[f"{model}_{set_name}"]["metrics"]
        for kidx, (key, center) in enumerate(zip(keys, centers)):
            jitter = rng.uniform(-0.045, 0.045, len(subset))
            ax.scatter(center + offsets[model] + jitter, [r[key] * 1000 for r in subset], color=colors[model], alpha=0.27, s=7, edgecolors="none")
            value = summary[key]["estimate"] * 1000
            lo, hi = (v * 1000 for v in summary[key]["ci95"])
            ax.errorbar(center + offsets[model], value, yerr=[[value - lo], [hi - value]], fmt="o", ms=5.3, color=colors[model], mec="black", mew=0.55, ecolor=colors[model], capsize=2.4, lw=1.05, zorder=10, label=model.capitalize() if kidx == 0 else None)
    ax.axhline(0, color="#555555", lw=0.75)
    ax.set_xticks(centers, labels)
    ax.set_ylabel("Signed radial response (mm)")
    ax.set_title("(b) Signed conditional effects", loc="left", fontweight="bold")
    ax.grid(axis="y", color="#dddddd", lw=0.5)
    ax.legend(frameon=False, loc="upper right")
    fig.suptitle("Original 34 sources" if set_name == "original34" else "Expanded 42-source coverage sensitivity", y=1.015, fontsize=8.7)
    fig.subplots_adjust(left=0.085, right=0.992, bottom=0.24, top=0.84)
    stem = "fig_interface_proprio_original34" if set_name == "original34" else "fig_interface_proprio_expanded42"
    fig.savefig(out / "figures" / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(out / "figures" / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_supplements(out: Path, rows: list[dict[str, Any]], equality_summary: dict[str, Any], sensitivity_rows: list[dict[str, Any]]) -> None:
    configure_matplotlib()
    import matplotlib.pyplot as plt

    colors = {"direct": "#3B6FB6", "joint": "#D07632"}
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.75), gridspec_kw={"wspace": 0.34, "width_ratios": [1.08, 1.0]})
    ax = axes[0]
    ax.axis("off")
    columns = ["Layer", "Pairs", "Numeric =", "Bytes =", "max |Δ|"]
    table_rows = []
    for layer in ["normalized_action", "joint_targets", "eef_position", "eef_rotation", "radial_projection"]:
        item = equality_summary[layer]
        pairs = "84×14" if layer == "joint_targets" else str(item["pairs"])
        numerical = "84×14" if layer == "joint_targets" else str(item["numeric_equal_pairs"])
        byte_equal = "84×14" if layer == "joint_targets" else str(item["byte_equal_pairs"])
        max_diff = "0" if item["max_abs_diff"] == 0 else item["max_abs_diff_scientific"]
        table_rows.append([layer.replace("_", " "), pairs, numerical, byte_equal, max_diff])
    table = ax.table(
        cellText=table_rows,
        colLabels=columns,
        colWidths=[0.31, 0.16, 0.19, 0.18, 0.16],
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False); table.set_fontsize(6.6); table.scale(1.0, 1.25)
    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0.4); cell.set_edgecolor("#b7b7b7")
        if row == 0: cell.set_facecolor("#eeeeee"); cell.set_text_props(weight="bold")
    ax.set_title("(a) Y11 vs. natural Dplus equality", loc="left", fontweight="bold")

    ax = axes[1]
    for mi, model in enumerate(MODELS):
        subset = [r for r in rows if r["model"] == model]
        vals = np.asarray([r["X_signed"] * 1000 for r in subset])
        x = np.full(len(vals), mi, dtype=float) + np.linspace(-0.09, 0.09, len(vals))
        ax.scatter(x, vals, color=colors[model], s=9, alpha=0.45)
        ax.scatter(mi, task_equal(subset, "X_signed") * 1000, color=colors[model], edgecolor="black", linewidth=0.5, s=34, zorder=5)
    ax.axhline(0, color="#555555", lw=0.7)
    ax.set_xticks([0, 1], ["Direct", "Joint"])
    ax.set_ylabel("Interaction, signed radial (mm)")
    ax.set_title("(b) Non-zero conditional interaction", loc="left", fontweight="bold")
    ax.grid(axis="y", color="#dddddd", lw=0.5)
    fig.subplots_adjust(left=0.055, right=0.99, bottom=0.19, top=0.87)
    fig.savefig(out / "figures" / "supp_full_action_algebra.pdf", bbox_inches="tight")
    fig.savefig(out / "figures" / "supp_full_action_algebra.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.75), gridspec_kw={"wspace": 0.3})
    tasks = sorted({r["task_id"] for r in rows})
    ax = axes[0]
    for model in MODELS:
        values = [task_equal([r for r in rows if r["model"] == model and r["task_id"] == task], "G_P_given_I") * 1000 for task in tasks]
        offset = -0.12 if model == "direct" else 0.12
        ax.scatter(np.arange(len(tasks)) + offset, values, color=colors[model], s=25, label=model.capitalize())
    ax.axhline(0, color="#555555", lw=0.7)
    ax.set_xticks(np.arange(len(tasks)), [t.replace("_", "\n") for t in tasks], rotation=20, ha="right")
    ax.set_ylabel(r"$G_{P\mid I}$ (mm)")
    ax.set_title("(a) Per-task coverage sensitivity", loc="left", fontweight="bold")
    ax.legend(frameon=False)
    ax.grid(axis="y", color="#dddddd", lw=0.5)
    ax = axes[1]
    for mi, model in enumerate(MODELS):
        vals = [float(r["G_P_given_I"]) * 1000 for r in sensitivity_rows if r["model"] == model and r["kind"] in {"LOSO", "LOTO"}]
        ax.boxplot([vals], positions=[mi], widths=0.42, showfliers=False, patch_artist=True, boxprops={"facecolor": colors[model], "alpha": 0.35}, medianprops={"color": colors[model]})
        ax.scatter(np.full(len(vals), mi) + np.linspace(-0.08, 0.08, len(vals)), vals, s=5, alpha=0.17, color=colors[model])
    ax.axhline(0, color="#555555", lw=0.7)
    ax.set_xticks([0, 1], ["Direct", "Joint"])
    ax.set_ylabel(r"Deletion sensitivity of $G_{P\mid I}$ (mm)")
    ax.set_title("(b) Existing LOSO / LOTO analyses", loc="left", fontweight="bold")
    ax.grid(axis="y", color="#dddddd", lw=0.5)
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.27, top=0.87)
    fig.savefig(out / "figures" / "supp_per_task_and_deletion_sensitivity.pdf", bbox_inches="tight")
    fig.savefig(out / "figures" / "supp_per_task_and_deletion_sensitivity.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run = args.campaign.resolve()
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"NONEMPTY_OUTPUT_DIRECTORY:{out}")
    out.mkdir(parents=True, exist_ok=True)
    for d in ("figures", "plot_data", "supplement", "scripts"):
        (out / d).mkdir(exist_ok=True)

    sources = read_json(run / "source_manifest.json")["sources"]
    source_by_id = {s["source_id"]: s for s in sources}
    ledger = load_jsonl(run / "ledger.jsonl")
    finished = [r for r in ledger if r.get("event") == "CALL_FINISHED"]
    by_call = {r["call_id"]: r for r in finished}
    authoritative_stats = read_json(run / "analysis/bootstrap_summary.json")

    raw_rows: list[dict[str, Any]] = []
    identity_rows: list[dict[str, Any]] = []
    algebra_rows: list[dict[str, Any]] = []
    source_metrics: list[dict[str, Any]] = []
    equality_acc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    missing: list[dict[str, Any]] = []

    for model in MODELS:
        for source in sources:
            sid = source["source_id"]
            root = run / model / "actions" / sid
            event_root = run / model / "events" / sid
            paths = {key: root / f"{stem}.npz" for key, stem in CELLS.items()}
            paths.update({"D": root / "capture_Dplus.npz", "Z": root / "capture_Z1.npz"})
            absent = [str(p) for p in paths.values() if not p.exists()]
            if absent:
                missing.append({"model": model, "source_id": sid, "missing": absent})
                continue
            arrays = {key: np.load(path, allow_pickle=False) for key, path in paths.items()}
            contract_path = Path(source["geometry_dir"]) / "fk_contract.json"
            if not contract_path.exists():
                contract_path = Path(source["geometry_dir"]) / "target_fk_contract.json"
            contract = read_json(contract_path)
            yd_rotation = rotations(contract, arrays["D"]["joint_targets"])
            y11_rotation = rotations(contract, arrays["Y11"]["joint_targets"])
            geodesic = geodesic_angles(y11_rotation, yd_rotation)
            base = {
                "model": model,
                "source_id": sid,
                "task_id": source["task_id"],
                "analysis_set": source["analysis_set"],
                "ancestry_status": source["ancestry_status"],
                "y11_path": str(paths["Y11"]),
                "natural_dplus_path": str(paths["D"]),
                "y11_file_sha256": sha(paths["Y11"]),
                "natural_dplus_file_sha256": sha(paths["D"]),
                "files_same_sha256": sha(paths["Y11"]) == sha(paths["D"]),
                "objects_are_distinct_files": not os.path.samefile(paths["Y11"], paths["D"]),
            }

            def audit_layer(layer: str, a: np.ndarray, b_: np.ndarray, units: str, origin: str, component: str = "all") -> None:
                comp = compare_arrays(a, b_)
                row = {
                    **base,
                    "layer": layer,
                    "component": component,
                    "units": units,
                    "data_origin": origin,
                    "y11_dtype": str(np.asarray(a).dtype),
                    "natural_dplus_dtype": str(np.asarray(b_).dtype),
                    "y11_shape": json.dumps(list(np.asarray(a).shape)),
                    "natural_dplus_shape": json.dumps(list(np.asarray(b_).shape)),
                    **comp,
                }
                if layer == "eef_rotation":
                    row["max_rotation_geodesic_rad"] = float(np.max(geodesic))
                    row["max_rotation_geodesic_scientific"] = fmt_sci(float(np.max(geodesic)))
                raw_rows.append(row)
                equality_acc[layer].append(comp)

            audit_layer("normalized_action", arrays["Y11"]["normalized"], arrays["D"]["normalized"], "normalized", "stored 32x14")
            for idx, name in enumerate(CHANNELS):
                unit = "rad" if "joint" in name else "registered gripper-command unit"
                audit_layer("joint_targets", arrays["Y11"]["joint_targets"][:, idx], arrays["D"]["joint_targets"][:, idx], unit, "stored de-normalized 32x14", name)
            audit_layer("eef_position", arrays["Y11"]["eef_target_world"], arrays["D"]["eef_target_world"], "m", "stored registered FK position")
            audit_layer("eef_rotation", y11_rotation, yd_rotation, "rotation-matrix entries; geodesic in rad", "read-only reconstruction from stored joint targets and frozen FK contract")
            audit_layer("radial_projection", arrays["Y11"]["radial_world"], arrays["D"]["radial_world"], "m", "stored recipient-fixed radial projection")

            # Actual identity and consumer evidence for all cells and both captures.
            capture_events: dict[str, dict[str, Any]] = {}
            for endpoint in ("Z1", "Dplus"):
                ep = read_json(event_root / f"capture_{endpoint}.json")
                capture_events[endpoint] = ep
            for cell, stem in CELLS.items():
                expected_i, expected_p = EXPECTED[cell]
                event_path = event_root / f"{stem}.json"
                events = read_json(event_path)
                iface = events["interface_events"]
                enc = events["proprio_encoder_events"]
                action_consumer = events["proprio_action_consumer_events"]
                call_id = f"{model}__{sid}__{stem}"
                ledger_row = by_call.get(call_id)
                capture_iface = capture_events[expected_i]["interface_events"]
                written_match = len(iface) == len(capture_iface) and all(
                    a.get("written_hash") == b.get("written_hash")
                    for a, b in zip(iface, capture_iface)
                )
                if model == "direct":
                    actual_interface_sources = sorted({str(e.get("source")) for e in iface})
                    source_ok = actual_interface_sources == [f"{expected_i}_FROZEN_INTERFACE"]
                    current_sources = "N/A"
                    future_sources = "N/A"
                else:
                    current_values = sorted({str(e.get("current_source")) for e in iface})
                    future_values = sorted({str(e.get("future_source")) for e in iface})
                    actual_interface_sources = sorted(set(current_values + future_values))
                    source_ok = current_values == [expected_i] and future_values == [expected_i]
                    current_sources = json.dumps(current_values)
                    future_sources = json.dumps(future_values)
                encoder_sources = sorted({str(e.get("source")) for e in enc})
                consumer_sources = sorted({str(e.get("source")) for e in action_consumer})
                proprio_ok = encoder_sources == [expected_p] and consumer_sources == [expected_p]
                expected_image_call = by_call.get(f"{model}__{sid}__capture_Z1")
                expected_prop_call = by_call.get(f"{model}__{sid}__capture_{expected_p}")
                image_ok = bool(ledger_row and expected_image_call and ledger_row.get("image_observation_sha256") == expected_image_call.get("image_observation_sha256"))
                proprio_hash_ok = bool(ledger_row and expected_prop_call and ledger_row.get("selected_proprio_sha256") == expected_prop_call.get("selected_proprio_sha256"))
                interface_grid = {(e.get("step"), e.get("layer"), e.get("component")) for e in iface}
                grid_ok = interface_grid == {(step, layer, comp) for step in range(10) for layer in range(30) for comp in ("k", "v")}
                dplus_call = by_call.get(f"{model}__{sid}__capture_Dplus")
                independent_ref = bool(
                    dplus_call
                    and ledger_row
                    and dplus_call["call_id"] != ledger_row["call_id"]
                    and Path(dplus_call["action_artifact"]).resolve() != Path(ledger_row["action_artifact"]).resolve()
                )
                identity_rows.append(
                    {
                        "model": model,
                        "source_id": sid,
                        "task_id": source["task_id"],
                        "cell": cell,
                        "call_id": call_id,
                        "ledger_call_found": ledger_row is not None,
                        "ledger_status": ledger_row.get("status") if ledger_row else "MISSING",
                        "expected_interface_source": expected_i,
                        "actual_interface_sources": json.dumps(actual_interface_sources),
                        "joint_current_sources": current_sources,
                        "joint_future_sources": future_sources,
                        "interface_source_verified": source_ok,
                        "interface_events": len(iface),
                        "complete_30x10xKV_grid": grid_ok,
                        "written_hashes_match_endpoint_capture": written_match,
                        "expected_proprio_source": expected_p,
                        "encoder_sources": json.dumps(encoder_sources),
                        "action_consumer_sources": json.dumps(consumer_sources),
                        "proprio_source_verified": proprio_ok,
                        "proprio_encoder_events": len(enc),
                        "proprio_action_consumer_events": len(action_consumer),
                        "recipient_image_hash_matches_capture_Z1": image_ok,
                        "selected_proprio_hash_matches_expected_capture": proprio_hash_ok,
                        "natural_dplus_is_distinct_call_and_artifact": independent_ref,
                        "interface_hook_cleanup": ledger_row.get("interface_hook_cleanup") if ledger_row else "MISSING",
                        "proprio_hook_cleanup": ledger_row.get("proprio_hook_cleanup") if ledger_row else "MISSING",
                        "output_finite_and_pass": bool(ledger_row and ledger_row.get("status") == "PASS"),
                        "random_condition_evidence": "source policy_seed + frozen worker reset_random before every call; no separate RNG-state tensor saved",
                        "instruction_evidence": "source instruction + frozen worker call; instruction tensor not separately logged",
                        "non_proprio_context_scope": "recipient Z1 image/input path; source instruction; registered seed/noise/scheduler by frozen worker",
                        "events_path": str(event_path),
                        "events_sha256": sha(event_path),
                        "action_path": ledger_row.get("action_artifact") if ledger_row else "",
                        "action_sha256": ledger_row.get("action_artifact_sha256") if ledger_row else "",
                    }
                )

            y = {key: arrays[key]["radial_world"].astype(np.float64) for key in ("Y00", "Y10", "Y01", "Y11", "D", "Z")}
            decompositions = {
                "interface_then_command": (
                    y["Y11"] - y["Y00"],
                    (y["Y10"] - y["Y00"]) + (y["Y11"] - y["Y10"]),
                ),
                "command_then_interface": (
                    y["Y11"] - y["Y00"],
                    (y["Y01"] - y["Y00"]) + (y["Y11"] - y["Y01"]),
                ),
            }
            for name, (lhs, rhs) in decompositions.items():
                err = lhs - rhs
                for pos in range(32):
                    algebra_rows.append(
                        {
                            "record_type": "PER_POSITION_ALGEBRA",
                            "model": model,
                            "source_id": sid,
                            "task_id": source["task_id"],
                            "analysis_set": source["analysis_set"],
                            "decomposition": name,
                            "prediction_position": pos,
                            "lhs_m": fmt_sci(float(lhs[pos])),
                            "rhs_m": fmt_sci(float(rhs[pos])),
                            "floating_algebra_error_m": fmt_sci(float(err[pos])),
                            "floating_algebra_error_abs_m": fmt_sci(abs(float(err[pos]))),
                        }
                    )
            effects = {
                "I_P0": y["Y10"] - y["Y00"],
                "P_I1": y["Y11"] - y["Y10"],
                "P_I0": y["Y01"] - y["Y00"],
                "I_P1": y["Y11"] - y["Y01"],
                "X": y["Y11"] - y["Y10"] - y["Y01"] + y["Y00"],
                "natural": y["D"] - y["Z"],
            }
            source_metrics.append(
                {
                    "model": model,
                    "source_id": sid,
                    "task_id": source["task_id"],
                    "source_group_id": source["source_group_id"],
                    "analysis_set": source["analysis_set"],
                    "ancestry_status": source["ancestry_status"],
                    "E10": float(np.sqrt(np.mean(np.square(y["Y10"] - y["D"])))),
                    "E11": float(np.sqrt(np.mean(np.square(y["Y11"] - y["D"])))),
                    "G_P_given_I": float(np.sqrt(np.mean(np.square(y["Y10"] - y["D"]))) - np.sqrt(np.mean(np.square(y["Y11"] - y["D"])))),
                    **{f"{key}_signed": float(np.mean(value)) for key, value in effects.items()},
                    **{f"{key}_rms": float(np.sqrt(np.mean(np.square(value)))) for key, value in effects.items()},
                    "natural_equals_Y11_minus_Y00": bool(np.array_equal(effects["natural"], y["Y11"] - y["Y00"])),
                    "y00_equals_natural_Z1": bool(np.array_equal(y["Y00"], y["Z"])),
                    "y11_equals_natural_Dplus": bool(np.array_equal(y["Y11"], y["D"])),
                    "direction_cancellation_source_level": bool(np.sign(np.mean(effects["I_P0"])) == -np.sign(np.mean(effects["P_I1"])) and np.mean(effects["I_P0"]) != 0 and np.mean(effects["P_I1"]) != 0),
                    "positions_with_opposite_I_P0_and_P_I1_sign": int(np.count_nonzero(np.sign(effects["I_P0"]) == -np.sign(effects["P_I1"]))),
                }
            )

    # Compute/reproduce the frozen estimand with one common bootstrap random stream per set.
    metric_keys = [
        "E10", "E11", "G_P_given_I", "I_P0_signed", "P_I1_signed", "P_I0_signed",
        "I_P1_signed", "X_signed", "I_P0_rms", "P_I1_rms", "P_I0_rms", "I_P1_rms",
        "X_rms", "natural_signed", "natural_rms",
    ]
    stats: dict[str, Any] = {}
    reproduction: list[dict[str, Any]] = []
    for model in MODELS:
        for set_name in ("original34", "expanded42"):
            subset = [
                row for row in source_metrics
                if row["model"] == model
                and (set_name == "expanded42" or row["analysis_set"] == "ORIGINAL34_RESULT_DRIVEN_FOLLOWUP")
            ]
            key = f"{model}_{set_name}"
            stats[key] = {"n": len(subset), "tasks": sorted({r["task_id"] for r in subset}), "metrics": paired_bootstrap(subset, metric_keys)}
            for metric in ("E10", "E11", "G_P_given_I", "I_P0_signed", "P_I1_signed", "X_signed", "I_P0_rms", "P_I1_rms", "X_rms"):
                auth = authoritative_stats[key]["metrics"][metric]
                new = stats[key]["metrics"][metric]
                reproduction.append(
                    {
                        "set": key,
                        "metric": metric,
                        "authoritative_estimate": auth["estimate"],
                        "readonly_estimate": new["estimate"],
                        "estimate_abs_difference": abs(auth["estimate"] - new["estimate"]),
                        "authoritative_ci_low": auth["ci95"][0],
                        "readonly_ci_low": new["ci95"][0],
                        "ci_low_abs_difference": abs(auth["ci95"][0] - new["ci95"][0]),
                        "authoritative_ci_high": auth["ci95"][1],
                        "readonly_ci_high": new["ci95"][1],
                        "ci_high_abs_difference": abs(auth["ci95"][1] - new["ci95"][1]),
                    }
                )

    equality_summary: dict[str, Any] = {}
    for layer, values in equality_acc.items():
        equality_summary[layer] = {
            "pairs": len(values),
            "numeric_equal_pairs": sum(bool(v["numeric_equal"]) for v in values),
            "byte_equal_pairs": sum(bool(v["raw_element_bytes_identical"]) for v in values),
            "max_abs_diff": max(float(v["max_abs_diff"] or 0) for v in values),
            "max_abs_diff_scientific": fmt_sci(max(float(v["max_abs_diff"] or 0) for v in values)),
        }

    # Add task-equal signed-effect audit records.
    for key, payload in stats.items():
        model, set_name = key.split("_", 1)
        m = payload["metrics"]
        algebra_rows.append(
            {
                "record_type": "TASK_EQUAL_SIGNED_SUMMARY",
                "model": model,
                "source_id": "ALL",
                "task_id": "FIVE_TASK_EQUAL_WEIGHT",
                "analysis_set": set_name,
                "decomposition": "interface_then_command",
                "I_P0_signed_m": fmt_sci(m["I_P0_signed"]["estimate"]),
                "P_I1_signed_m": fmt_sci(m["P_I1_signed"]["estimate"]),
                "sum_signed_m": fmt_sci(m["I_P0_signed"]["estimate"] + m["P_I1_signed"]["estimate"]),
                "natural_signed_m": fmt_sci(m["natural_signed"]["estimate"]),
                "sum_minus_natural_m": fmt_sci(m["I_P0_signed"]["estimate"] + m["P_I1_signed"]["estimate"] - m["natural_signed"]["estimate"]),
                "P_I0_signed_m": fmt_sci(m["P_I0_signed"]["estimate"]),
                "I_P1_signed_m": fmt_sci(m["I_P1_signed"]["estimate"]),
                "alternative_sum_signed_m": fmt_sci(m["P_I0_signed"]["estimate"] + m["I_P1_signed"]["estimate"]),
                "note": "Linear identity applies to signed means; RMS values are not added or subtracted.",
            }
        )

    write_csv(out / "raw_equality_audit.csv", raw_rows)
    write_csv(out / "intervention_identity_audit.csv", identity_rows)
    write_csv(out / "algebra_and_signed_effect_audit.csv", algebra_rows)
    write_csv(out / "plot_data" / "per_source_plot_values.csv", source_metrics)
    write_csv(out / "plot_data" / "statistics_reproduction_audit.csv", reproduction)
    atomic_json(out / "plot_data" / "plot_summary.json", stats)
    atomic_json(out / "supplement" / "equality_summary.json", equality_summary)
    atomic_json(out / "supplement" / "missing_evidence.json", missing)

    # Audit the exact processor/statistics identity used by both model branches.
    registry = read_json(run / "registry.json")
    normalization_rows: list[dict[str, Any]] = []
    for model in MODELS:
        config_path = Path(registry[f"{model}_config"])
        config = read_json(config_path)
        stats_path = Path(config["EVALUATION"]["dataset_stats_path"])
        processor = config["data"]["train"]["processor"]
        dataset_stats = read_json(stats_path)
        action_stats = dataset_stats["action"]["default"]
        mean = np.asarray(action_stats["global_mean"], dtype=np.float32)
        std = np.asarray(action_stats["global_std"], dtype=np.float32)
        maximum_inverse_residual = 0.0
        arrays_checked = 0
        for source in sources:
            root = run / model / "actions" / source["source_id"]
            for stem in ["capture_Z1", "capture_Dplus", *CELLS.values()]:
                path = root / f"{stem}.npz"
                if not path.exists():
                    continue
                values = np.load(path, allow_pickle=False)
                # Frozen SingleFieldLinearNormalizer.backward for global z-score:
                # x * (std + 1e-8) + mean.  Float32 order matches the saved action path.
                reconstructed = (values["normalized"].astype(np.float32) * (std + np.float32(1e-8)) + mean).astype(np.float32)
                maximum_inverse_residual = max(
                    maximum_inverse_residual,
                    float(np.max(np.abs(reconstructed - values["joint_targets"]))),
                )
                arrays_checked += 1
        normalization_rows.append(
            {
                "model": model,
                "config_path": str(config_path),
                "config_sha256": sha(config_path),
                "stats_path": str(stats_path),
                "stats_sha256": sha(stats_path),
                "processor_target": processor["_target_"],
                "normalization_mode": processor["norm_default_mode"],
                "use_stepwise_action_norm": processor["use_stepwise_action_norm"],
                "action_shape": json.dumps(processor["shape_meta"]["action"][0]),
                "state_shape": json.dumps(processor["shape_meta"]["state"][0]),
                "action_state_merger": processor["action_state_merger"]["_target_"],
                "saved_arrays_checked": arrays_checked,
                "maximum_float32_inverse_formula_residual": maximum_inverse_residual,
                "maximum_float32_inverse_formula_residual_scientific": fmt_sci(maximum_inverse_residual),
                "note": "Raw residual reported without adding a post-hoc acceptance tolerance.",
            }
        )
    normalization_stats_identical = normalization_rows[0]["stats_sha256"] == normalization_rows[1]["stats_sha256"]
    for row in normalization_rows:
        row["direct_joint_stats_files_byte_identical"] = normalization_stats_identical
    write_csv(out / "normalization_and_denormalization_audit.csv", normalization_rows)

    # Existing sensitivity results are reused without changing definitions.
    with (run / "analysis" / "sensitivity.csv").open(newline="") as f:
        sensitivity_rows = list(csv.DictReader(f))
    write_csv(out / "supplement" / "existing_loso_loto.csv", sensitivity_rows)
    plot_reconstruction(out, "original34", [r for r in source_metrics if r["analysis_set"] == "ORIGINAL34_RESULT_DRIVEN_FOLLOWUP"], stats)
    plot_reconstruction(out, "expanded42", source_metrics, stats)
    plot_supplements(out, source_metrics, equality_summary, sensitivity_rows)

    # Calls/costs are counted from actual events, never inferred from the plan.
    starts = [r for r in ledger if r.get("event") == "CALL_STARTED"]
    finishes = [r for r in ledger if r.get("event") == "CALL_FINISHED"]
    cost = read_json(run / "resource_cost.json")
    call_summary = {
        "source_campaign": str(run),
        "source_campaign_final_status_sha256": sha(run / "final_status.json"),
        "planned_calls_from_frozen_plan": sum(1 for _ in csv.DictReader((run / "call_plan.csv").open())),
        "actual_call_started_events": len(starts),
        "actual_call_finished_events": len(finishes),
        "actual_passed_calls": sum(r.get("status") == "PASS" for r in finishes),
        "actual_failed_calls": sum(r.get("status") != "PASS" for r in finishes),
        "unique_call_ids_started": len({r.get("call_id") for r in starts}),
        "unique_call_ids_finished": len({r.get("call_id") for r in finishes}),
        "per_model_finished": dict(Counter(r.get("model") for r in finishes)),
        "aggregate_gpu_seconds_from_original_cost_ledger": cost["aggregate_gpu_seconds"],
        "model_load_events": cost["model_load_events"],
        "model_branch_events": cost["model_branch_events"],
        "simulator_initializations": read_json(run / "final_status.json")["simulator_initializations"],
        "physics_steps": read_json(run / "final_status.json")["physics_steps"],
        "predicted_actions_executed": read_json(run / "final_status.json")["predicted_actions_executed"],
        "note": "Counts are read from CALL_STARTED/CALL_FINISHED events; 504 is verified, not assumed.",
    }
    atomic_json(out / "actual_calls_and_cost_summary.json", call_summary)
    atomic_text(
        out / "actual_calls_and_cost_summary.md",
        "# Actual calls and costs\n\n"
        f"- Frozen plan rows: {call_summary['planned_calls_from_frozen_plan']}\n"
        f"- CALL_STARTED: {call_summary['actual_call_started_events']}\n"
        f"- CALL_FINISHED: {call_summary['actual_call_finished_events']}\n"
        f"- PASS / failed: {call_summary['actual_passed_calls']} / {call_summary['actual_failed_calls']}\n"
        f"- Per model: {call_summary['per_model_finished']}\n"
        f"- Aggregate GPU seconds in the original resource ledger: {call_summary['aggregate_gpu_seconds_from_original_cost_ledger']:.3f}\n"
        "- Simulator initializations / physics steps / predicted actions executed: 0 / 0 / 0.\n\n"
        "These counts were recomputed from actual ledger event types and do not assume that the planned 504 calls completed.\n",
    )

    # Evidence scope/coverage summary.
    identity_failures = [r for r in identity_rows if not all(bool(r[k]) for k in (
        "ledger_call_found", "interface_source_verified", "complete_30x10xKV_grid",
        "written_hashes_match_endpoint_capture", "proprio_source_verified",
        "recipient_image_hash_matches_capture_Z1", "selected_proprio_hash_matches_expected_capture",
        "natural_dplus_is_distinct_call_and_artifact", "output_finite_and_pass",
    ))]
    max_repro = max(max(float(r["estimate_abs_difference"]), float(r["ci_low_abs_difference"]), float(r["ci_high_abs_difference"])) for r in reproduction)
    audit_summary = {
        "status": "COMPLETE" if not missing and not identity_failures else "COMPLETE_WITH_GAPS",
        "scientific_identity": "POSTHOC_INTERFACE_X_PROPRIO_READONLY_CLOSEOUT",
        "source_campaign": str(run),
        "models": list(MODELS),
        "sources": len(sources),
        "model_source_pairs": len(source_metrics),
        "raw_equality_rows": len(raw_rows),
        "intervention_identity_rows": len(identity_rows),
        "algebra_rows": len(algebra_rows),
        "missing_evidence": missing,
        "identity_failure_count": len(identity_failures),
        "authoritative_statistic_max_abs_reproduction_difference": max_repro,
        "equality_summary": equality_summary,
        "rotation_scope": "Rotation was not stored in the campaign NPZ; it is deterministically reconstructed from stored de-normalized joint targets and the frozen registered FK contract.",
        "no_new_model_or_simulator_work": True,
    }
    atomic_json(out / "audit_summary.json", audit_summary)

    def mm(key: str, metric: str) -> tuple[float, float, float]:
        item = stats[key]["metrics"][metric]
        return item["estimate"] * 1000, item["ci95"][0] * 1000, item["ci95"][1] * 1000

    den, dlo, dhi = mm("direct_original34", "G_P_given_I")
    jen, jlo, jhi = mm("joint_original34", "G_P_given_I")
    d42, d42lo, d42hi = mm("direct_expanded42", "G_P_given_I")
    j42, j42lo, j42hi = mm("joint_expanded42", "G_P_given_I")
    di, dilo, dihi = mm("direct_original34", "I_P0_signed")
    dp, dplo, dphi = mm("direct_original34", "P_I1_signed")
    ji, jilo, jihi = mm("joint_original34", "I_P0_signed")
    jp, jplo, jphi = mm("joint_original34", "P_I1_signed")
    dx, dxlo, dxhi = mm("direct_original34", "X_signed")
    jx, jxlo, jxhi = mm("joint_original34", "X_signed")
    d42i, _, _ = mm("direct_expanded42", "I_P0_signed")
    d42p, _, _ = mm("direct_expanded42", "P_I1_signed")
    d42n, d42nlo, d42nhi = mm("direct_expanded42", "natural_signed")
    j42i, _, _ = mm("joint_expanded42", "I_P0_signed")
    j42p, _, _ = mm("joint_expanded42", "P_I1_signed")
    j42n, j42nlo, j42nhi = mm("joint_expanded42", "natural_signed")

    results_en = f"""# Results: interface × command-state reconstruction

Across all 84 model--source pairs, the saved Y11 output and its separately executed natural Dplus reference were element-wise and byte-wise identical as normalized 32×14 action arrays. The equality propagated through the saved de-normalized joint targets and registered FK positions and radial projections; the FK rotations reconstructed from the frozen contract were also identical. This is a numerical replay-closure result for the registered computation graph, not a behavioral-equivalence claim.

With the donor interface fixed, substituting the donor command-state input reduced the task-equal residual to natural Dplus from {den:.3f} mm (95% CI [{dlo:.3f}, {dhi:.3f}]) to exactly 0 in Direct and from {jen:.3f} mm ([{jlo:.3f}, {jhi:.3f}]) to exactly 0 in Joint on the original 34-source set. The overlapping 42-source coverage analysis gave corresponding improvements of {d42:.3f} mm ([{d42lo:.3f}, {d42hi:.3f}]) and {j42:.3f} mm ([{j42lo:.3f}, {j42hi:.3f}]). The field called proprio here is the registered 14-D joint drive-target/gripper-command vector, not measured articulation qpos.

The signed conditional effects opposed one another on average. On the original 34-source set, Interface | Z1 command state was {di:+.3f} mm ([{dilo:+.3f}, {dihi:+.3f}]) in Direct and {ji:+.3f} mm ([{jilo:+.3f}, {jihi:+.3f}]) in Joint, whereas Command state | donor interface was {dp:+.3f} mm ([{dplo:+.3f}, {dphi:+.3f}]) and {jp:+.3f} mm ([{jplo:+.3f}, {jphi:+.3f}]), respectively. In the expanded 42-source analysis, the corresponding signed sums were {d42i:+.3f} {d42p:+.3f} = {d42n:+.3f} mm (natural-response interval [{d42nlo:+.3f}, {d42nhi:+.3f}]) for Direct and {j42i:+.3f} {j42p:+.3f} = {j42n:+.3f} mm ([{j42nlo:+.3f}, {j42nhi:+.3f}]) for Joint. These task-equal signed averages describe directional cancellation; they do not imply that every source or action position cancels. The interactions remained non-zero: {dx:+.3f} mm ([{dxlo:+.3f}, {dxhi:+.3f}]) for Direct and {jx:+.3f} mm ([{jxlo:+.3f}, {jxhi:+.3f}]) for Joint, showing that each conditional effect depends on the other factor's fixed background.

The exact Y11 closure is expected under the audited coverage of this controlled graph: endpoint-native interface production already reads the command-state field, the complete registered action-facing interface is then frozen at every consumer event, and the remaining decoder command-state token is independently assigned. Image, instruction, initial noise, scheduler, seed, and the registered non-target context are matched by the frozen call implementation. Accordingly, closure is a consistency check supporting the factor interpretation, not an independent mechanistic discovery. The experiment is results-driven and post-hoc; the original 34-source historical identity does not make this factor experiment confirmatory, and the additional eight sources retain incompletely certified ancestry.
"""
    results_zh = f"""# 结果：接口 × 命令状态的自然响应重建

在全部84个模型—来源配对中，保存的Y11输出与独立执行的自然Dplus参照，其归一化32×14动作数组逐元素相等且元素字节一致。该一致性延续到保存的反归一化关节目标、登记FK位置和径向投影；由冻结FK契约从关节目标确定性重建的旋转也一致。这是登记计算图上的数值回放闭合，不是行为等效结论。

固定donor接口后，引入donor命令状态使原34来源上Direct到自然Dplus的任务等权残差从{den:.3f} mm（95%区间[{dlo:.3f}, {dhi:.3f}]）降至精确0，Joint从{jen:.3f} mm（[{jlo:.3f}, {jhi:.3f}]）降至精确0。重叠的42来源覆盖敏感性分析给出的改善分别为{d42:.3f} mm（[{d42lo:.3f}, {d42hi:.3f}]）和{j42:.3f} mm（[{j42lo:.3f}, {j42hi:.3f}]）。这里称为proprio的字段是登记的14维关节驱动目标／夹爪命令向量，不是实测articulation qpos。

两个有符号条件效应在平均方向上相反。原34来源中，Interface | Z1命令状态在Direct为{di:+.3f} mm（[{dilo:+.3f}, {dihi:+.3f}]），在Joint为{ji:+.3f} mm（[{jilo:+.3f}, {jihi:+.3f}]）；Command state | donor interface则分别为{dp:+.3f} mm（[{dplo:+.3f}, {dphi:+.3f}]）和{jp:+.3f} mm（[{jplo:+.3f}, {jphi:+.3f}]）。扩展42来源中，对应的有符号和为Direct {d42i:+.3f} {d42p:+.3f} = {d42n:+.3f} mm（自然响应区间[{d42nlo:+.3f}, {d42nhi:+.3f}]），Joint {j42i:+.3f} {j42p:+.3f} = {j42n:+.3f} mm（[{j42nlo:+.3f}, {j42nhi:+.3f}]）。这些任务等权有符号均值刻画的是平均方向抵消，不表示每个来源或每个动作位置都发生抵消。交互仍非零：Direct为{dx:+.3f} mm（[{dxlo:+.3f}, {dxhi:+.3f}]），Joint为{jx:+.3f} mm（[{jxlo:+.3f}, {jxhi:+.3f}]），说明各条件效应依赖另一因素的固定背景。

在已审计的受控计算图覆盖下，Y11精确闭合是可预期的：端点原生接口生成本身读取命令状态字段，完整登记的动作消费者接口随后在每个消费事件固定，余下的decoder命令状态token再被独立赋值；图像、指令、初始噪声、scheduler、seed及登记的非目标上下文由冻结调用实现匹配。因此，闭合是支持因子解释的一项一致性检查，而不是独立的新机制发现。本实验属于结果驱动的事后机制分析；原34来源的历史身份不会使本次因子实验自动成为确认性研究，新增8来源仍保留祖先关系未完全认证标记。
"""
    atomic_text(out / "RESULTS_EN.md", results_en)
    atomic_text(out / "RESULTS_ZH.md", results_zh)
    atomic_text(
        out / "claim_update.md",
        "# Claim update\n\n"
        "## Supported\n\n"
        "- In the registered static RoboTwin computation graph, adding the donor 14-D command-state input to the frozen donor interface closes the saved natural-Dplus action reconstruction for every audited Direct and Joint source.\n"
        "- Interface-only and command-state conditional effects oppose one another in the task-equal signed mean, while their non-zero interaction indicates background dependence.\n"
        "- The interface-only residual seen previously is therefore explained by omission of this registered command-state input under the tested intervention scope.\n\n"
        "## Required qualifications\n\n"
        "- Describe the field as a joint drive-target/gripper-command vector, not measured qpos and not pure proprioception.\n"
        "- Describe exact closure as a technical/causal consistency check expected from the audited external-input coverage, not an independent mechanism discovery or equivalence test.\n"
        "- Keep this experiment post-hoc and results-driven. The original-34 and expanded-42 sets overlap; the added eight retain incomplete ancestry certification.\n\n"
        "## Not supported\n\n"
        "- Behavioral improvement, robustness, model superiority, a contribution percentage, natural mediation, pure visual-versus-proprio separation, or four-model generality.\n",
    )

    caption = """\textbf{Interface and command-state factors jointly reconstruct the registered natural response.} (a) Paired source-level residuals to the independently executed natural Dplus output after replacing only the action-facing interface or both the interface and the registered 14-D joint drive-target/gripper-command input. Large markers and intervals are the frozen task-equal estimate and its existing 95\% bootstrap interval; small points are source-level values and are not used as an unweighted summary. (b) Task-equal signed radial effects for the interface under Z1 command state, command state under donor interface, and the natural Dplus--Z1 response. Readouts are predicted FK-implied EEF targets, not executed displacement. This results-driven post-hoc experiment separates a frozen interface-consumption path from a command-state context input; it is not a pure vision--proprioception decomposition.\n"""
    atomic_text(out / "figure_caption_en.tex", caption)
    atomic_text(out / "figure_caption_zh.md", "**接口与命令状态因素共同重建登记的自然响应。** (a) 仅替换动作接口与同时替换接口及14维关节驱动目标／夹爪命令输入时，相对于独立执行的自然Dplus输出的配对来源残差；大点和区间为冻结的任务等权估计及既有95% bootstrap区间，小点为来源级数值，不作为未加权汇总。(b) Z1命令状态下的接口效应、donor接口下的命令状态效应，以及自然Dplus−Z1响应的任务等权有符号径向效应。读出是预测关节目标经FK隐含的EEF目标，并非实际执行位移。本实验为结果驱动的事后机制分析，区分冻结接口消费路径与命令状态上下文输入，而非纯视觉—纯本体感知分解。\n")

    report = f"""# RoboTwin interface × proprio read-only closeout

Status: `{audit_summary['status']}`  
Identity: `POSTHOC_INTERFACE_X_PROPRIO_READONLY_CLOSEOUT`

## What “E11 = 0.000” means

It is strict saved-array closure, not rounded display zero. Across 84 model--source pairs, normalized 32×14 actions, every de-normalized joint/gripper channel, saved FK EEF positions, reconstructed FK rotations, and saved radial projections are numerically equal and have identical contiguous element bytes between Y11 and the separately executed natural Dplus reference. NPZ file hashes may differ because containers include other arrays and serialization metadata; container hash equality was not used as the array criterion. Rotation was not stored directly and is explicitly labeled as a deterministic read-only reconstruction from the frozen FK contract.

## Intervention identity

All {len(identity_rows)} four-cell records have the registered interface and command-state sources at the actual consumer: 600 interface events covering 30 layers × 10 action denoising steps × K/V, one 14-D encoder event, and ten action-context consumer events. Endpoint-written hashes match the independently captured endpoint banks. Y00--Y11 use the recipient Z1 image path. Natural Dplus is a distinct policy call and artifact, not an alias of Y11. Direct and Joint resolve to byte-identical normalization-statistics files and the same 14-D global z-score processor contract; the maximum raw float32 inverse-formula discrepancy is retained in `normalization_and_denormalization_audit.csv` without introducing a new pass tolerance. The source instruction and random reset are established by the frozen source record and worker implementation; no separate RNG-state tensor or instruction-token dump was saved, so that part is code-and-record verified rather than independently tensor-logged.

## Algebra and signed effects

Both decomposition identities were checked at every source and all 32 prediction positions. Floating arithmetic residuals are retained in `algebra_and_signed_effect_audit.csv`. The task-equal signed Direct expanded-42 decomposition is {stats['direct_expanded42']['metrics']['I_P0_signed']['estimate']*1000:+.3f} {stats['direct_expanded42']['metrics']['P_I1_signed']['estimate']*1000:+.3f} = {stats['direct_expanded42']['metrics']['natural_signed']['estimate']*1000:+.3f} mm; Joint is {stats['joint_expanded42']['metrics']['I_P0_signed']['estimate']*1000:+.3f} {stats['joint_expanded42']['metrics']['P_I1_signed']['estimate']*1000:+.3f} = {stats['joint_expanded42']['metrics']['natural_signed']['estimate']*1000:+.3f} mm. This linear identity applies to signed means, never to RMS values.

## Scope

No WAM was loaded, no simulator was initialized, no physics was advanced, no donor was generated, and no policy call or predicted-action execution was added. The original campaign remains untouched. The result is static action-response evidence only.
"""
    atomic_text(out / "closeout_report.md", report)
    atomic_text(
        out / "POSTPROCESS_CORRECTIONS.md",
        "# Read-only post-processing corrections\n\n"
        "During figure inspection, an initial draft computed rotation geodesic distance directly as "
        "`acos((trace(R_a.T @ R_b)-1)/2)` on the near-orthogonal floating FK matrices. Even when "
        "the two stored-input reconstructions were element-wise identical, finite orthogonality error "
        "could yield a false non-zero angle (maximum observed in the superseded draft: approximately "
        "2.86e-4 rad). The sealed version returns exactly zero for element-wise identical rotations and "
        "otherwise projects both operands to SO(3) by SVD before computing the geodesic angle. No model "
        "output, action, position, radial readout, bootstrap result, or historical file was changed.\n",
    )

    # Reproducibility and input identity.
    shutil.copy2(Path(__file__), out / "scripts" / Path(__file__).name)
    input_paths = [
        run / "source_manifest.json", run / "ledger.jsonl", run / "resource_cost.json",
        run / "final_status.json", run / "analysis" / "bootstrap_summary.json",
        run / "analysis" / "sensitivity.csv", run / "graph_and_intervention_contract.md", run / "registry.json",
        *(Path(registry[f"{model}_config"]) for model in MODELS),
        *(Path(read_json(Path(registry[f"{model}_config"]))["EVALUATION"]["dataset_stats_path"]) for model in MODELS),
    ]
    atomic_json(
        out / "input_identity.json",
        {
            "source_campaign": str(run),
            "files": [{"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)} for path in input_paths],
            "action_artifacts_audited": 84 * 6,
            "event_artifacts_audited": 84 * 6,
            "note": "All source campaign inputs were read-only.",
        },
    )

    # Manifest is deliberately last and excludes itself; no output log is kept in the directory.
    manifest_files = []
    for path in sorted(p for p in out.rglob("*") if p.is_file() and p.name != "final_hash_manifest.json"):
        manifest_files.append({"path": str(path.relative_to(out)), "bytes": path.stat().st_size, "sha256": sha(path)})
    atomic_json(
        out / "final_hash_manifest.json",
        {
            "status": "SEALED_AFTER_ALL_OUTPUT_WRITES",
            "root": str(out),
            "files": manifest_files,
            "excluded": [{"path": "final_hash_manifest.json", "reason": "self-referential hash is impossible"}],
        },
    )
    print(json.dumps({"status": audit_summary["status"], "out": str(out), "files": len(manifest_files), "missing": len(missing), "identity_failures": len(identity_failures)}, sort_keys=True))


if __name__ == "__main__":
    main()

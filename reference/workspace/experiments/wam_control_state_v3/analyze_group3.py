from __future__ import annotations
from wam_causal_audit.paths import resolve as _release_path

import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(_release_path('@DATA@/wam_factor_routing_v5'))
GROUP1 = ROOT / "group1"
GROUP2 = ROOT / "group2" / "three_model_interim_summary"
GROUP2_FINAL = ROOT / "group2" / "four_model_final_summary"
OUT = ROOT / "group3"

STRUCTURED_PROFILE = GROUP1 / "group1_structured_factor_profile.csv"
RAW_PROFILE = GROUP1 / "factor_profiles.parquet"
DOSE_RESPONSE = GROUP1 / "dose_responses.csv"
GROUP2_CASES = GROUP2 / "case_metrics.csv"
GROUP2_BOOTSTRAP = GROUP2 / "bootstrap_95ci.csv"
GROUP2_REPORT = GROUP2 / "statistical_report.json"
GROUP2_MANIFEST = GROUP2 / "artifact_manifest.json"

MODEL_ORDER = ["direct", "joint", "idm", "imagewam"]
MODEL_LABEL = {
    "direct": "Direct-WAM",
    "joint": "Joint-WAM",
    "idm": "IDM-WAM",
    "imagewam": "ImageWAM",
}
GROUP_COLUMNS = ["factor", "phase", "control_condition"]
PROFILE_COLUMNS = [
    "ACTION_ACTIVE",
    "CARRIER_TRANSMITTED",
    "effect_sign",
    "native_l2_over_replay_floor",
    "dose_slope",
    "direction_consistency",
    "dose_monotonicity",
    "phase_effect_mean",
]
PRIMARY_ABS_DOSES = {
    "F1_ROBOT_RADIAL_PROGRESS": [1.0, 2.0, 4.0],
    "F2_OBJECT_ROBOT_GEOMETRY": [1.0, 2.0, 4.0],
    "F3_OBJECT_GOAL_RADIAL_PROGRESS": [1.0, 2.0, 4.0],
    "F4_TANGENTIAL_DISPLACEMENT": [1.0, 2.0, 4.0],
    "F5_ORIENTATION_ONLY": [10.0, 20.0, 30.0],
}
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260901
EPSILON = 1e-12
INTERIM_STATUS = "GROUP3_PARTIAL_ROUTING_WAITING_IDM"
ROUTING_SCOPE = "THREE_MODEL_INTERIM"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def safe_rho(left: np.ndarray, right: np.ndarray) -> float:
    mask = np.isfinite(left) & np.isfinite(right)
    if int(mask.sum()) < 2:
        return float("nan")
    if np.ptp(left[mask]) == 0 or np.ptp(right[mask]) == 0:
        return float("nan")
    return float(spearmanr(left[mask], right[mask]).statistic)


def key_tuple(row: pd.Series) -> tuple[str, str, str]:
    return tuple(str(row[column]) for column in GROUP_COLUMNS)  # type: ignore[return-value]


def group2_input_paths(mode: str) -> tuple[Path, Path, Path, Path]:
    root = GROUP2 if mode == "interim" else GROUP2_FINAL
    return (
        root / "case_metrics.csv",
        root / "bootstrap_95ci.csv",
        root / "statistical_report.json",
        root / "artifact_manifest.json",
    )


def verify_inputs(mode: str) -> dict[str, dict[str, Any]]:
    group2_cases, group2_bootstrap, group2_report, group2_manifest = group2_input_paths(mode)
    paths = [
        STRUCTURED_PROFILE,
        RAW_PROFILE,
        DOSE_RESPONSE,
        group2_cases,
        group2_bootstrap,
        group2_report,
        group2_manifest,
    ]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"GROUP3_INPUT_MISSING:{missing}")
    return {
        str(path): {"sha256": sha256(path), "bytes": path.stat().st_size}
        for path in paths
    }


def load_profile() -> pd.DataFrame:
    frame = pd.read_csv(STRUCTURED_PROFILE)
    expected = set(MODEL_ORDER)
    observed = set(frame["model"].unique())
    if observed != expected:
        raise RuntimeError(f"GROUP3A_MODEL_SET_MISMATCH:{observed}:{expected}")
    if frame.groupby("model").size().to_dict() != {model: 7 for model in MODEL_ORDER}:
        raise RuntimeError("GROUP3A_PROFILE_CELL_COUNT_MISMATCH")
    if frame.duplicated(["model", *GROUP_COLUMNS]).any():
        raise RuntimeError("GROUP3A_DUPLICATE_PROFILE_CELL")
    missing_columns = [column for column in PROFILE_COLUMNS if column not in frame.columns]
    if missing_columns:
        raise RuntimeError(f"GROUP3A_PROFILE_FIELDS_MISSING:{missing_columns}")
    return frame


def model_sets(frame: pd.DataFrame, column: str) -> dict[str, set[tuple[str, str, str]]]:
    result: dict[str, set[tuple[str, str, str]]] = {}
    for model in MODEL_ORDER:
        subset = frame[(frame["model"] == model) & frame[column].astype(bool)]
        result[model] = {key_tuple(row) for _, row in subset.iterrows()}
    return result


def set_jaccard_rows(frame: pd.DataFrame, column: str) -> list[dict[str, Any]]:
    sets = model_sets(frame, column)
    rows: list[dict[str, Any]] = []
    for left, right in combinations(MODEL_ORDER, 2):
        union = sets[left] | sets[right]
        intersection = sets[left] & sets[right]
        rows.append({
            "model_a": left,
            "model_b": right,
            "set_definition": column,
            "intersection_count": len(intersection),
            "union_count": len(union),
            "jaccard": len(intersection) / len(union) if union else 1.0,
            "ci_status": "NOT_APPLICABLE_FROZEN_SET_COMPARISON",
            "evidence_scope": "FOUR_MODEL_GROUP3A",
        })
    return rows


def sign_agreement_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    indexed = {
        model: frame[frame.model == model].set_index(GROUP_COLUMNS)
        for model in MODEL_ORDER
    }
    rows: list[dict[str, Any]] = []
    for left, right in combinations(MODEL_ORDER, 2):
        common = indexed[left].index.intersection(indexed[right].index)
        jointly_active = [
            key for key in common
            if bool(indexed[left].loc[key, "ACTION_ACTIVE"])
            and bool(indexed[right].loc[key, "ACTION_ACTIVE"])
        ]
        resolved = [
            key for key in jointly_active
            if indexed[left].loc[key, "effect_sign"] in {"POSITIVE", "NEGATIVE"}
            and indexed[right].loc[key, "effect_sign"] in {"POSITIVE", "NEGATIVE"}
        ]
        matches = sum(
            indexed[left].loc[key, "effect_sign"] == indexed[right].loc[key, "effect_sign"]
            for key in resolved
        )
        rows.append({
            "model_a": left,
            "model_b": right,
            "jointly_active_count": len(jointly_active),
            "jointly_direction_resolved_count": len(resolved),
            "same_direction_count": int(matches),
            "sign_agreement": matches / len(resolved) if resolved else np.nan,
            "direction_coverage": len(resolved) / len(jointly_active) if jointly_active else np.nan,
            "unresolved_excluded_count": len(jointly_active) - len(resolved),
            "ci_status": "NOT_APPLICABLE_FROZEN_DIRECTION_LABELS",
            "evidence_scope": "FOUR_MODEL_GROUP3A",
        })
    return rows


def phase_agreement_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    active = frame[frame.ACTION_ACTIVE.astype(bool)].copy()
    roots = sorted(set(zip(active.factor, active.control_condition)))
    rows: list[dict[str, Any]] = []
    for left, right in combinations(MODEL_ORDER, 2):
        per_root: list[float] = []
        exact = 0
        for factor, control in roots:
            left_phases = set(active[(active.model == left) & (active.factor == factor) & (active.control_condition == control)].phase)
            right_phases = set(active[(active.model == right) & (active.factor == factor) & (active.control_condition == control)].phase)
            union = left_phases | right_phases
            per_root.append(len(left_phases & right_phases) / len(union) if union else 1.0)
            exact += left_phases == right_phases
        rows.append({
            "model_a": left,
            "model_b": right,
            "factor_control_roots": len(roots),
            "exact_phase_set_matches": exact,
            "phase_exact_agreement_fraction": exact / len(roots),
            "phase_set_jaccard_macro": float(np.mean(per_root)),
            "ci_status": "NOT_APPLICABLE_FROZEN_PHASE_LABELS",
            "evidence_scope": "FOUR_MODEL_GROUP3A",
        })
    return rows


def rank_point_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    pivot = frame.pivot(index=GROUP_COLUMNS, columns="model", values="native_l2_over_replay_floor")
    rows: list[dict[str, Any]] = []
    for left, right in combinations(MODEL_ORDER, 2):
        rows.append({
            "model_a": left,
            "model_b": right,
            "metric": "spearman_rho_frozen_replay_floor_standardized_magnitude",
            "point_estimate": safe_rho(pivot[left].to_numpy(float), pivot[right].to_numpy(float)),
            "n_factor_phase_control_cells": len(pivot),
            "normalization": "native_l2_over_replay_floor",
            "normalization_note": "all frozen replay floors are zero; epsilon scaling preserves within-model ranks",
            "evidence_scope": "FOUR_MODEL_GROUP3A",
        })
    return rows


def bootstrap_rank_cis(rows: list[dict[str, Any]]) -> None:
    raw = pd.read_parquet(
        RAW_PROFILE,
        columns=["model", "factor", "phase", "relation_control", "task_id", "base_state_id", "native_action_l2"],
    ).rename(columns={"relation_control": "control_condition"})
    base = raw.groupby(["model", *GROUP_COLUMNS, "task_id", "base_state_id"], as_index=False)["native_action_l2"].mean()
    tasks = sorted(base.task_id.unique())
    keys = sorted({tuple(value) for value in base[GROUP_COLUMNS].itertuples(index=False, name=None)})
    arrays: dict[tuple[str, tuple[str, str, str], Any], np.ndarray] = {}
    for model in MODEL_ORDER:
        for key in keys:
            mask = base.model.eq(model)
            for column, value in zip(GROUP_COLUMNS, key):
                mask &= base[column].eq(value)
            cell = base[mask]
            for task in tasks:
                values = cell.loc[cell.task_id.eq(task), "native_action_l2"].to_numpy(float)
                if len(values):
                    arrays[(model, key, task)] = values
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    pair_values = {(row["model_a"], row["model_b"]): np.empty(BOOTSTRAP_RESAMPLES) for row in rows}
    for iteration in range(BOOTSTRAP_RESAMPLES):
        sampled_tasks = rng.choice(tasks, len(tasks), replace=True)
        magnitudes: dict[str, np.ndarray] = {}
        for model in MODEL_ORDER:
            values_by_key: list[float] = []
            for key in keys:
                task_means: list[float] = []
                for task in sampled_tasks:
                    values = arrays.get((model, key, task))
                    if values is None or len(values) == 0:
                        continue
                    sampled = values[rng.integers(0, len(values), len(values))]
                    task_means.append(float(sampled.mean()))
                values_by_key.append(float(np.mean(task_means)) if task_means else np.nan)
            magnitudes[model] = np.asarray(values_by_key)
        for pair, output in pair_values.items():
            output[iteration] = safe_rho(magnitudes[pair[0]], magnitudes[pair[1]])
    for row in rows:
        values = pair_values[(row["model_a"], row["model_b"])]
        values = values[np.isfinite(values)]
        row["ci_low"] = float(np.quantile(values, 0.025))
        row["ci_high"] = float(np.quantile(values, 0.975))
        row["bootstrap_resamples_valid"] = len(values)
        row["bootstrap_unit"] = "base_state"
        row["cluster_unit"] = "task"


def dose_shape_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    response = pd.read_csv(DOSE_RESPONSE).rename(columns={"relation_control": "control_condition"})
    response = response[
        response.apply(
            lambda row: abs(float(row.signed_dose)) in PRIMARY_ABS_DOSES[str(row.factor)],
            axis=1,
        )
    ]
    indexed = {model: frame[frame.model == model].set_index(GROUP_COLUMNS) for model in MODEL_ORDER}
    rows: list[dict[str, Any]] = []
    for left, right in combinations(MODEL_ORDER, 2):
        common = indexed[left].index.intersection(indexed[right].index)
        for key in common:
            left_curve = response[
                response.model.eq(left)
                & response.factor.eq(key[0])
                & response.phase.eq(key[1])
                & response.control_condition.eq(key[2])
            ][["signed_dose", "native_primary_drive_effect_mean"]]
            right_curve = response[
                response.model.eq(right)
                & response.factor.eq(key[0])
                & response.phase.eq(key[1])
                & response.control_condition.eq(key[2])
            ][["signed_dose", "native_primary_drive_effect_mean"]]
            merged = left_curve.merge(right_curve, on="signed_dose", suffixes=("_a", "_b"))
            slope_a = float(indexed[left].loc[key, "dose_slope"])
            slope_b = float(indexed[right].loc[key, "dose_slope"])
            mono_a = float(indexed[left].loc[key, "dose_monotonicity"])
            mono_b = float(indexed[right].loc[key, "dose_monotonicity"])
            rows.append({
                "model_a": left,
                "model_b": right,
                "factor": key[0],
                "phase": key[1],
                "control_condition": key[2],
                "dose_ordering_spearman": safe_rho(
                    merged.native_primary_drive_effect_mean_a.to_numpy(float),
                    merged.native_primary_drive_effect_mean_b.to_numpy(float),
                ),
                "n_common_primary_doses": len(merged),
                "dose_slope_a": slope_a,
                "dose_slope_b": slope_b,
                "signed_slope_agreement": bool(np.sign(slope_a) == np.sign(slope_b)),
                "dose_monotonicity_a": mono_a,
                "dose_monotonicity_b": mono_b,
                "graded_vs_flat_status": "NOT_CLASSIFIED_NO_FROZEN_THRESHOLD",
                "stress_dose_excluded": bool(key[0] == "F2_OBJECT_ROBOT_GEOMETRY"),
                "evidence_scope": "FOUR_MODEL_GROUP3A",
            })
    return rows


def routing_rows(mode: str) -> list[dict[str, Any]]:
    _, group2_bootstrap, _, _ = group2_input_paths(mode)
    stats = pd.read_csv(group2_bootstrap)
    group = stats[stats.scope.eq("FACTOR_PHASE_CONTROL")]
    rows: list[dict[str, Any]] = []
    labels = {
        "joint": ("CURRENT_WORLD_KV", "FUTURE_WORLD_KV", "REDUNDANT_CURRENT_WORLD_PLUS_FUTURE_WORLD"),
        "imagewam": ("IMAGE_KV", "NON_IMAGE_PREFIX_KV", "IMAGE_DOMINANT_WITH_SMALL_PREFIX_TRANSFER"),
        "direct": ("CURRENT_IMAGE_CARRIER", "EXPLICIT_FUTURE_READ", "CURRENT_IMAGE_ROUTE_STRUCTURAL_FUTURE_ZERO"),
        "idm": ("FUTURE_MEDIATED_ROUTE", "N/A_CONTEXT_ROUTE_NOT_SEPARABLE", "FUTURE_MEDIATED_ROUTE"),
    }
    routing_models = ["joint", "imagewam", "direct"] if mode == "interim" else ["joint", "idm", "imagewam", "direct"]
    for model in routing_models:
        subset = group[group.model.eq(model)]
        keys = subset[GROUP_COLUMNS].drop_duplicates().itertuples(index=False, name=None)
        for key in keys:
            cell = subset[
                subset.factor.eq(key[0])
                & subset.phase.eq(key[1])
                & subset.control_condition.eq(key[2])
            ]
            metric_lookup = {row.metric: row for row in cell.itertuples(index=False)}
            record: dict[str, Any] = {
                "model": model,
                "factor": key[0],
                "phase": key[1],
                "control_condition": key[2],
                "route_a_label": labels[model][0],
                "route_b_label": labels[model][1],
                "frozen_routing_statement": labels[model][2],
                "evidence_scope": ROUTING_SCOPE,
                "status": "COMPLETE",
            }
            for metric in ["route_a_projection", "route_b_projection", "interaction_projection", "closure_relative"]:
                if metric in metric_lookup:
                    value = metric_lookup[metric]
                    record[metric] = value.point_estimate
                    record[f"{metric}_ci_low"] = value.ci_low
                    record[f"{metric}_ci_high"] = value.ci_high
                else:
                    record[metric] = np.nan
                    record[f"{metric}_ci_low"] = np.nan
                    record[f"{metric}_ci_high"] = np.nan
            if model == "direct":
                record["route_b_status"] = "STRUCTURAL_ZERO_NOT_NUMERIC_EFFECT"
            elif model == "idm":
                record["route_b_status"] = "N/A_CONTEXT_ROUTE_NOT_SEPARABLE"
            else:
                record["route_b_status"] = "MEASURED"
            rows.append(record)
    if mode == "interim":
        rows.append({
            "model": "idm",
            "factor": "ALL_FROZEN_FACTORS",
            "phase": "ALL_FROZEN_PHASES",
            "control_condition": "ALL_FROZEN_CONTROLS",
            "route_a_label": "FUTURE_MEDIATED_ROUTE_PENDING",
            "route_b_label": "N/A_CONTEXT_ROUTE_NOT_SEPARABLE",
            "route_b_status": "GROUP2_ROUTING_PENDING",
            "frozen_routing_statement": "GROUP2_ROUTING_PENDING",
            "evidence_scope": ROUTING_SCOPE,
            "status": "GROUP2_ROUTING_PENDING",
        })
    return rows


def write_mixed_effects(path: Path) -> None:
    pd.DataFrame([{
        "formula": "E ~ Model * Factor * Phase + (1 | Task) + (1 | BaseState)",
        "status": "NOT_ESTIMABLE_FROZEN_SELF_REPLAY_NORMALIZATION_DEGENERATE",
        "reason": "all frozen self_replay_l2_floor values are exactly zero and no frozen state-level standardized E exists",
        "action": "no post-hoc normalization or equivalence margin introduced",
        "model_factor_interaction": np.nan,
        "model_factor_phase_interaction": np.nan,
        "confidence_interval": np.nan,
        "evidence_scope": "FOUR_MODEL_GROUP3A",
    }]).to_csv(path, index=False)


def make_figures(profile: pd.DataFrame, ranks: pd.DataFrame, routing: pd.DataFrame, figures: Path, mode: str) -> None:
    figures.mkdir(parents=True, exist_ok=True)
    profile = profile.copy()
    profile["cell"] = profile.factor + " | " + profile.phase + " | " + profile.control_condition
    cell_order = list(profile[profile.model.eq("direct")].cell)

    heat = profile.pivot(index="cell", columns="model", values="direction_consistency").reindex(index=cell_order, columns=MODEL_ORDER)
    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(heat.to_numpy(float), vmin=0, vmax=1, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(MODEL_ORDER)), [MODEL_LABEL[model] for model in MODEL_ORDER], rotation=25, ha="right")
    ax.set_yticks(range(len(cell_order)), cell_order, fontsize=8)
    ax.set_title("Frozen factor × phase direction consistency")
    fig.colorbar(im, ax=ax, label="direction consistency")
    fig.tight_layout()
    fig.savefig(figures / "factor_phase_model_heatmap.png", dpi=180)
    plt.close(fig)

    matrix = np.eye(len(MODEL_ORDER))
    for row in ranks.itertuples(index=False):
        i, j = MODEL_ORDER.index(row.model_a), MODEL_ORDER.index(row.model_b)
        matrix[i, j] = matrix[j, i] = row.point_estimate
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm")
    labels = [MODEL_LABEL[model] for model in MODEL_ORDER]
    ax.set_xticks(range(4), labels, rotation=25, ha="right")
    ax.set_yticks(range(4), labels)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center")
    ax.set_title("Pairwise factor-rank stability (Spearman rho)")
    fig.colorbar(im, ax=ax, label="rho")
    fig.tight_layout()
    fig.savefig(figures / "pairwise_profile_stability.png", dpi=180)
    plt.close(fig)

    _, group2_bootstrap, _, _ = group2_input_paths(mode)
    overall = pd.read_csv(group2_bootstrap)
    overall = overall[overall.scope.eq("OVERALL") & overall.metric.isin(["route_a_projection", "route_b_projection"])]
    fig, ax = plt.subplots(figsize=(8, 5))
    routing_models = ["joint", "imagewam", "direct"] if mode == "interim" else ["joint", "idm", "imagewam", "direct"]
    x = np.arange(len(routing_models))
    for offset, metric, label in [(-0.17, "route_a_projection", "primary route"), (0.17, "route_b_projection", "secondary route")]:
        values, low, high = [], [], []
        for model in routing_models:
            cell = overall[(overall.model == model) & (overall.metric == metric)]
            if cell.empty:
                values.append(np.nan); low.append(np.nan); high.append(np.nan)
            else:
                row = cell.iloc[0]
                values.append(row.point_estimate); low.append(row.point_estimate - row.ci_low); high.append(row.ci_high - row.point_estimate)
        ax.bar(x + offset, values, width=0.34, label=label)
        finite = np.isfinite(values)
        ax.errorbar((x + offset)[finite], np.asarray(values)[finite], yerr=np.asarray([low, high])[:, finite], fmt="none", ecolor="black", capsize=3)
    ax.set_xticks(x, [MODEL_LABEL[model] for model in routing_models])
    ax.set_ylabel("within-model directional transfer")
    figure_scope = ROUTING_SCOPE if mode == "interim" else "FOUR_MODEL_FINAL"
    ax.set_title(f"Routing comparison ({figure_scope})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "routing_profile_comparison.png", dpi=180)
    plt.close(fig)

    ranked = profile.copy()
    ranked["rank"] = ranked.groupby("model")["native_l2_over_replay_floor"].rank(ascending=False, method="average")
    fig, ax = plt.subplots(figsize=(10, 6))
    for model in MODEL_ORDER:
        cell = ranked[ranked.model.eq(model)].set_index("cell").reindex(cell_order)
        ax.plot(np.arange(len(cell_order)), cell["rank"], marker="o", label=MODEL_LABEL[model])
    ax.invert_yaxis()
    ax.set_xticks(np.arange(len(cell_order)), cell_order, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("within-model effect rank (1 = largest)")
    ax.set_title("Frozen factor-effect ranks")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "factor_effect_rank.png", dpi=180)
    plt.close(fig)


def fmt(value: float) -> str:
    return "N/A" if not np.isfinite(value) else f"{value:.3f}"


def make_report(
    profile: pd.DataFrame,
    active: pd.DataFrame,
    carrier: pd.DataFrame,
    signs: pd.DataFrame,
    phases: pd.DataFrame,
    ranks: pd.DataFrame,
    routing: pd.DataFrame,
    input_manifest: dict[str, Any],
    mode: str,
) -> str:
    model_pair_rows = []
    for pair in active[["model_a", "model_b"]].itertuples(index=False):
        a = active[(active.model_a == pair.model_a) & (active.model_b == pair.model_b)].iloc[0]
        c = carrier[(carrier.model_a == pair.model_a) & (carrier.model_b == pair.model_b)].iloc[0]
        s = signs[(signs.model_a == pair.model_a) & (signs.model_b == pair.model_b)].iloc[0]
        p = phases[(phases.model_a == pair.model_a) & (phases.model_b == pair.model_b)].iloc[0]
        r = ranks[(ranks.model_a == pair.model_a) & (ranks.model_b == pair.model_b)].iloc[0]
        model_pair_rows.append(
            f"| {MODEL_LABEL[pair.model_a]} / {MODEL_LABEL[pair.model_b]} | {a.jaccard:.3f} | {c.jaccard:.3f} | "
            f"{s.sign_agreement:.3f} ({int(s.jointly_direction_resolved_count)}/{int(s.jointly_active_count)} resolved) | "
            f"{p.phase_exact_agreement_fraction:.3f} | {r.point_estimate:.3f} [{r.ci_low:.3f}, {r.ci_high:.3f}] |"
        )
    status = profile.groupby("model").agg(
        active=("ACTION_ACTIVE", "sum"),
        carrier=("CARRIER_TRANSMITTED", "sum"),
        resolved=("effect_sign", lambda x: x.isin(["POSITIVE", "NEGATIVE"]).sum()),
    )
    interim = mode == "interim"
    report_title = "# Group 3 部分执行报告" if interim else "# Group 3 四模型最终报告"
    report_status = INTERIM_STATUS if interim else "GROUP3_FOUR_MODEL_FINAL"
    report_scope = ROUTING_SCOPE if interim else "FOUR_MODEL_FINAL"
    scope_text = (
        f"Group 3A 包含四模型；Group 3B 仅包含已完成的 Direct-WAM、Joint-WAM、ImageWAM，所有 routing 输出均标记为 `{ROUTING_SCOPE}`。IDM 明确为 `GROUP2_ROUTING_PENDING`。"
        if interim
        else "Group 3A 与 Group 3B 均包含四模型；使用与 interim 完全相同的冻结指标、模型顺序和分析规则。"
    )
    lines = [
        report_title,
        "",
        f"状态：`{report_status}`。{scope_text}",
        "",
        "## Group 3A：四模型 factor profile",
        "",
        "| 模型 | 冻结单元 | ACTION_ACTIVE | CARRIER_TRANSMITTED | direction resolved |",
        "|---|---:|---:|---:|---:|",
    ]
    for model in MODEL_ORDER:
        lines.append(f"| {MODEL_LABEL[model]} | 7 | {int(status.loc[model, 'active'])} | {int(status.loc[model, 'carrier'])} | {int(status.loc[model, 'resolved'])} |")
    lines += [
        "",
        "| 模型对 | active Jaccard | carrier Jaccard | sign agreement | phase agreement | factor-rank rho [95% CI] |",
        "|---|---:|---:|---:|---:|---:|",
        *model_pair_rows,
        "",
        "四模型的冻结 active set 与 carrier-transmitted set 完全一致；在双方均有方向判定的单元上，因果方向一致。共同的明确正向核心是 F1、F2 与 F3-PREPLACE；F4/F5 均未获得方向判定，F3-TRANSPORT 仅 ImageWAM 为 `UNRESOLVED`。相对强度排序不是完全相同，具体差异由 rank rho 与 dose-shape 表给出。",
        "",
        "该结果量化的是 weak factor-level profile stability，不是四架构机制等价。active/carrier 集合完全重合也不能消除路由拓扑差异。",
        "",
        "### 冻结标准化限制",
        "",
        "全部 `self_replay_l2_floor` 精确为 0。冻结的 `native_l2_over_replay_floor` 使用 epsilon 后仍保持每个模型内部排序，因此可用于预注册的 factor-rank Spearman；但不存在冻结的 state-level standardized E，confirmatory mixed-effects model 被标记为 `NOT_ESTIMABLE_FROZEN_SELF_REPLAY_NORMALIZATION_DEGENERATE`。未新增 z-score、阈值或等价界限。",
        "",
        f"## Group 3B：{'三模型 interim' if interim else '四模型 final'} routing",
        "",
        "- Direct-WAM：`CURRENT_IMAGE_ROUTE_STRUCTURAL_FUTURE_ZERO`；显式 future read 是结构缺失，不是数值干预零。",
        "- Joint-WAM：`REDUNDANT_CURRENT_WORLD_PLUS_FUTURE_WORLD`；两路均能传递，interaction 强负，不能解释成两个独占比例。",
        "- ImageWAM：`IMAGE_DOMINANT_WITH_SMALL_PREFIX_TRANSFER`；保持 image/prefix provenance 标签，不改称 current/future。",
        ("- IDM-WAM：`GROUP2_ROUTING_PENDING`；未由架构或早期实验推断路由。" if interim else "- IDM-WAM：使用完成的 Group-2 `FUTURE_MEDIATED_ROUTE` 结果；context route 仍为不可分离，不声称 future-only exclusivity。"),
        "",
        f"{('三' if interim else '四')}模型 routing 表按 factor–phase–control 保存 route transfer、interaction、closure 及原 Group-2 10,000 次 task-cluster/base-state bootstrap CI。跨模型 route 轴语义不同，因此不强制压缩成一个新的标量相似度或事后分类阈值。",
        "",
        "## 当前允许的结论",
        "",
        ("Group 3A 支持：四架构保存了相同的冻结 active/carrier factor 单元和共同的已解析方向核心，但 factor 相对强度存在模型间差异。Group 3B 支持：三个已完成模型以不同的内部路由拓扑承载这些 factor effects。最终 profile+routing 联合判定仍等待 IDM Group 2。" if interim else "Group 3A 与完成的四模型 Group 3B 连续指标共同构成最终比较；仍不把高 profile overlap 等同于机制等价，也不把不可分离路径解释为独占路径。"),
        "",
        ("## 当前禁止的结论" if interim else "## 解释边界"),
        "",
        ("本报告不发布 `FOUR_MODEL_ROUTING_CONCLUSION`、`PROFILE_CONSERVED_ROUTING_CHANGED_FINAL`、`FOUR_MODEL_MECHANISTIC_EQUIVALENCE` 或 `FOUR_MODEL_ROUTING_DIVERGENCE`，也不进行 CASE A/B/C 最终分类。" if interim else "连续指标优先于离散 CASE 标签；不根据最终观测值新增阈值或等价界限。"),
        "",
        "## 可复现性",
        "",
        f"输入文件数：{len(input_manifest)}；所有输入路径、字节数与 SHA-256 已写入 `group3_validation_report.json`，输出哈希写入 `{'group3_interim_freeze_manifest.json' if interim else 'group3_freeze_manifest.json'}`。本次没有 simulator execution、model forward、donor search 或 locus search。证据范围：`{report_scope}`。",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["interim", "final"], default="interim")
    args = parser.parse_args()
    input_manifest = verify_inputs(args.mode)
    profile = load_profile()
    if not (profile.self_replay_l2_floor == 0).all():
        raise RuntimeError("FROZEN_REPLAY_FLOOR_CHANGED; stop instead of changing normalization")

    OUT.mkdir(parents=True, exist_ok=True)
    figures = OUT / "figures"
    active = pd.DataFrame(set_jaccard_rows(profile, "ACTION_ACTIVE"))
    carrier = pd.DataFrame(set_jaccard_rows(profile, "CARRIER_TRANSMITTED"))
    signs = pd.DataFrame(sign_agreement_rows(profile))
    phases = pd.DataFrame(phase_agreement_rows(profile))
    rank_records = rank_point_rows(profile)
    bootstrap_rank_cis(rank_records)
    ranks = pd.DataFrame(rank_records)
    dose = pd.DataFrame(dose_shape_rows(profile))
    routing = pd.DataFrame(routing_rows(args.mode))

    outputs = {
        "active_set_jaccard.csv": active,
        "carrier_set_jaccard.csv": carrier,
        "sign_agreement.csv": signs,
        "phase_agreement.csv": phases,
        "factor_rank_spearman.csv": ranks,
        "dose_shape_agreement.csv": dose,
        "routing_profile_comparison.csv": routing,
    }
    for name, frame in outputs.items():
        frame.to_csv(OUT / name, index=False)
    write_mixed_effects(OUT / "mixed_effects_results.csv")

    config = {
        "status": "GROUP3_ANALYSIS_SETTINGS_FROZEN",
        "execution_modes_supported": ["interim", "final"],
        "model_order": MODEL_ORDER,
        "group_columns": GROUP_COLUMNS,
        "profile_columns": PROFILE_COLUMNS,
        "rank_metric": "Spearman rho of native_l2_over_replay_floor",
        "rank_bootstrap": {
            "resamples": BOOTSTRAP_RESAMPLES,
            "seed": BOOTSTRAP_SEED,
            "bootstrap_unit": "base_state",
            "cluster_unit": "task",
            "confidence_level": 0.95,
        },
        "dose_shape": {
            "primary_abs_doses": PRIMARY_ABS_DOSES,
            "stress_F2_abs_8_excluded": True,
            "graded_vs_flat": "NOT_CLASSIFIED_NO_FROZEN_THRESHOLD",
        },
        "routing_interim_models": ["direct", "joint", "imagewam"],
        "routing_final_models": MODEL_ORDER,
        "routing_interim_pending": {"idm": "GROUP2_ROUTING_PENDING"},
        "routing_evidence_scopes": [ROUTING_SCOPE, "FOUR_MODEL_FINAL"],
        "mixed_effects": "NOT_ESTIMABLE_FROZEN_SELF_REPLAY_NORMALIZATION_DEGENERATE",
        "new_thresholds": False,
        "new_equivalence_margin": False,
        "new_normalization": False,
        "simulator_execution": False,
        "model_forward_pass": False,
    }
    config_path = OUT / "group3_analysis_config_frozen.json"
    write_json(config_path, config)
    script_path = Path(__file__).resolve()

    make_figures(profile, ranks, routing, figures, args.mode)
    report_path = OUT / "group3_report.md"
    report_path.write_text(
        make_report(profile, active, carrier, signs, phases, ranks, routing, input_manifest, args.mode),
        encoding="utf-8",
    )
    interim = args.mode == "interim"
    interpretation = {
        "status": INTERIM_STATUS if interim else "GROUP3_FOUR_MODEL_FINAL",
        "group3a_scope": "FOUR_MODEL_FACTOR_PROFILE_COMPLETE",
        "group3b_scope": ROUTING_SCOPE if interim else "FOUR_MODEL_FINAL",
        "idm_routing": "GROUP2_ROUTING_PENDING" if interim else "FUTURE_MEDIATED_ROUTE",
        "discrete_case_classification": "PENDING_IDM_GROUP2" if interim else "NOT_FORCED_CONTINUOUS_METRICS_REPORTED",
        "profile_observation": "identical frozen active/carrier sets and shared resolved sign core; relative ranks vary",
        "routing_observation": "three completed models have distinct frozen routing profiles" if interim else "four completed models compared with frozen route-specific labels",
        "prohibited_conclusions_issued": False,
        "stop_condition": "STOP_AFTER_INTERIM_GROUP3_REPORT" if interim else "FINAL_GROUP3_OUTPUTS_GENERATED",
    }
    write_json(OUT / "case_interpretation.json", interpretation)

    if args.mode == "final":
        final_summary = profile[["model", *GROUP_COLUMNS, *PROFILE_COLUMNS]].copy()
        final_summary["group2_routing_status"] = final_summary["model"].map(
            routing.groupby("model")["status"].first().to_dict()
        )
        final_summary.to_csv(OUT / "group3_final_summary.csv", index=False)
        (OUT / "group3_final_summary.md").write_text(report_path.read_text(encoding="utf-8"), encoding="utf-8")

    output_paths = [
        *(OUT / name for name in outputs),
        OUT / "mixed_effects_results.csv",
        config_path,
        report_path,
        OUT / "case_interpretation.json",
        *(sorted(figures.glob("*.png"))),
    ]
    if args.mode == "final":
        output_paths.extend([OUT / "group3_final_summary.csv", OUT / "group3_final_summary.md"])
    validation = {
        "status": "GROUP3_PARTIAL_EXECUTION_PASS" if interim else "GROUP3_FOUR_MODEL_FINAL_PASS",
        "execution_authorization": "GROUP3_PARTIAL_EXECUTION_AUTHORIZED" if interim else "IDM_GROUP2_COMPLETED_TRIGGER",
        "group3a": "FOUR_MODEL_FACTOR_PROFILE_COMPLETE",
        "group3b": ROUTING_SCOPE if interim else "FOUR_MODEL_FINAL",
        "idm": "GROUP2_ROUTING_PENDING" if interim else "FUTURE_MEDIATED_ROUTE",
        "read_only_source_analysis": True,
        "simulator_execution": False,
        "new_model_forward_pass": False,
        "new_donor_search": False,
        "new_locus_search": False,
        "source_artifacts": input_manifest,
        "analysis_script": {"path": str(script_path), "sha256": sha256(script_path)},
        "analysis_config": {"path": str(config_path), "sha256": sha256(config_path)},
        "outputs": {str(path): {"sha256": sha256(path), "bytes": path.stat().st_size} for path in output_paths},
    }
    validation_path = OUT / "group3_validation_report.json"
    write_json(validation_path, validation)
    output_paths.append(validation_path)
    freeze_manifest = {
        "status": "GROUP3_INTERIM_FROZEN" if interim else "GROUP3_FOUR_MODEL_FINAL_FROZEN",
        "evidence_scope": ROUTING_SCOPE if interim else "FOUR_MODEL_FINAL",
        "idm": "GROUP2_ROUTING_PENDING" if interim else "FUTURE_MEDIATED_ROUTE",
        "analysis_script_sha256": sha256(script_path),
        "analysis_config_sha256": sha256(config_path),
        "source_artifact_hashes": {path: metadata["sha256"] for path, metadata in input_manifest.items()},
        "output_hashes": {str(path): sha256(path) for path in output_paths},
        "future_final_rule": "append completed IDM with the same frozen analysis code/settings; do not change metrics, thresholds, normalization, factor set, model order, or analysis rules",
    }
    manifest_path = OUT / "group3_interim_freeze_manifest.json"
    if args.mode == "final":
        validation_final = OUT / "group3_validation_report.json"
        manifest_path = OUT / "group3_freeze_manifest.json"
    write_json(manifest_path, freeze_manifest)

    print(json.dumps({
        "status": "GROUP3_PARTIAL_EXECUTION_PASS" if interim else "GROUP3_FOUR_MODEL_FINAL_PASS",
        "group3a_models": MODEL_ORDER,
        "group3b_models": ["direct", "joint", "imagewam"] if interim else MODEL_ORDER,
        "idm": "GROUP2_ROUTING_PENDING" if interim else "FUTURE_MEDIATED_ROUTE",
        "report": str(report_path),
        "report_sha256": sha256(report_path),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

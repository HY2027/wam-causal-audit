#!/usr/bin/env python3
"""Analyze Joint new-trajectory response prediction and closed-loop confirmation."""

from __future__ import annotations
from wam_causal_audit.paths import resolve as _release_path

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(_release_path('@DATA@/wam_factor_routing_v5/joint_idm_compute_confirmatory_v1_20260909'))
OUT = ROOT / "joint_new_trajectory_confirmatory"
PRED = OUT / "joint_new_prelabel_predictions.csv"
LABEL = OUT / "joint_new_trajectory_response_damage.csv"
EPISODES = OUT / "joint_new_trajectory_episode_outcomes.csv"
NBOOT = 10000
SEED = 20260909


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dump(path: Path, value) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def hierarchy_samples(data: pd.DataFrame, rng: np.random.Generator):
    tasks = sorted(data.task_id.unique())
    groups = {(task, traj): group.index.to_numpy() for (task, traj), group in data.groupby(["task_id", "trajectory_id"])}
    indices = []
    for task in rng.choice(tasks, len(tasks), replace=True):
        trajectories = sorted(traj for t, traj in groups if t == task)
        for trajectory in rng.choice(trajectories, len(trajectories), replace=True):
            indices.extend(groups[(task, trajectory)])
    return indices


def rmse_difference(data: pd.DataFrame, model_a: str, model_b: str, k: int) -> dict:
    part = data[(data.K == k) & data.model.isin([model_a, model_b])]
    wide = part.pivot(index=["case_id", "task_id", "trajectory_id", "damage_radial_l2"],
                      columns="model", values="predicted_damage_radial_l2").reset_index().dropna()
    def value(frame):
        target = frame.damage_radial_l2.to_numpy(float)
        a = np.sqrt(np.mean((frame[model_a].to_numpy(float) - target) ** 2))
        b = np.sqrt(np.mean((frame[model_b].to_numpy(float) - target) ** 2))
        return float(a - b)
    point = value(wide); rng = np.random.default_rng(SEED + k + sum(map(ord, model_a + model_b)))
    draws = np.empty(NBOOT)
    for index in range(NBOOT):
        draws[index] = value(wide.loc[hierarchy_samples(wide, rng)])
    return {"K": k, "model_a": model_a, "model_b": model_b,
            "rmse_difference_a_minus_b": point, "ci_low": float(np.quantile(draws, 0.025)),
            "ci_high": float(np.quantile(draws, 0.975)), "bootstrap_resamples": NBOOT,
            "bootstrap_hierarchy": "task then source trajectory", "negative_favors_model_a": True,
            "cases": int(wide.case_id.nunique()), "trajectories": int(wide.groupby(["task_id", "trajectory_id"]).ngroups)}


def singlecall() -> None:
    predictions = pd.read_csv(PRED); labels = pd.read_csv(LABEL)
    data = predictions.merge(labels[["case_id", "K", "damage_radial_l2", "task_id", "trajectory_id"]],
                             on=["case_id", "K", "task_id", "trajectory_id"], validate="many_to_one")
    data["squared_error"] = (data.predicted_damage_radial_l2 - data.damage_radial_l2) ** 2
    rows = []
    for (k, model), group in data.groupby(["K", "model"]):
        rows.append({"scope": "OVERALL", "task_id": "ALL", "K": k, "model": model,
                     "rmse": float(np.sqrt(group.squared_error.mean())), "mae": float(np.abs(group.predicted_damage_radial_l2-group.damage_radial_l2).mean()),
                     "cases": int(group.case_id.nunique()), "trajectories": int(group.groupby(["task_id", "trajectory_id"]).ngroups)})
        for task, task_data in group.groupby("task_id"):
            rows.append({"scope": "TASK", "task_id": int(task), "K": k, "model": model,
                         "rmse": float(np.sqrt(task_data.squared_error.mean())),
                         "mae": float(np.abs(task_data.predicted_damage_radial_l2-task_data.damage_radial_l2).mean()),
                         "cases": int(task_data.case_id.nunique()), "trajectories": int(task_data.trajectory_id.nunique())})
    evaluation = pd.DataFrame(rows); evaluation.to_csv(OUT / "joint_new_prediction_evaluation.csv", index=False)
    comparisons = []
    for k in (5, 8):
        comparisons.append(rmse_difference(data, "simple_features_plus_causal", "simple_features", k))
        comparisons.append(rmse_difference(data, "simple_plus_causal", "full_response", k))
    comparisons_df = pd.DataFrame(comparisons); comparisons_df.to_csv(OUT / "joint_new_prediction_paired_comparisons.csv", index=False)
    primary = comparisons[0]
    report = {
        "status": "JOINT_NEW_TRAJECTORY_RESPONSE_PREDICTION_EVALUATED",
        "created_unix": time.time(), "states": int(labels.candidate_id.nunique()),
        "cases": int(labels.case_id.nunique()), "tasks": sorted(int(x) for x in labels.task_id.unique()),
        "primary_comparison": primary,
        "primary_supports_incremental_causal_value": bool(primary["ci_high"] < 0),
        "prediction_freeze_sha256": sha(OUT / "joint_new_prediction_freeze_manifest.json"),
        "labels_sha256": sha(LABEL), "evaluation_sha256": sha(OUT / "joint_new_prediction_evaluation.csv"),
        "comparisons_sha256": sha(OUT / "joint_new_prediction_paired_comparisons.csv"),
        "response_prediction_not_behavior_prediction": True,
    }
    dump(OUT / "joint_new_prediction_evaluation_manifest.json", report)
    print(json.dumps(report))


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--mode", choices=["singlecall"], required=True)
    args = parser.parse_args(); singlecall()


if __name__ == "__main__": main()

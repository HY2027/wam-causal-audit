from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import RESULTS, TASKS, atomic_json, sha, write_csv


N_BOOT = 10_000
SEED = 20260923 + 2


def interval(values: np.ndarray) -> list[float]:
    finite = values[np.isfinite(values)]
    return [float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975))]


def main() -> None:
    completion = json.loads((RESULTS / "stage5_completion.json").read_text())
    data = pd.read_parquet(RESULTS / "closed_loop_raw.parquet")
    valid = data[(data["status"] == "PASS") & data["budget"].notna()].copy()
    pivot = valid.pivot(index=["pair_id", "task_id", "seed"], columns="budget")
    complete_index = pivot.dropna(subset=[("success", 5.0), ("success", 10.0)]).index
    rows = []
    for pair_id, task, seed in complete_index:
        k10 = valid[(valid.pair_id == pair_id) & (valid.budget == 10)].iloc[0]
        k5 = valid[(valid.pair_id == pair_id) & (valid.budget == 5)].iloc[0]
        rows.append({
            "pair_id": pair_id,
            "task_id": task,
            "seed": int(seed),
            "success_K10": bool(k10.success),
            "success_K5": bool(k5.success),
            "success_difference": int(bool(k5.success)) - int(bool(k10.success)),
            "policy_seconds_K10": float(k10.policy_compute_seconds),
            "policy_seconds_K5": float(k5.policy_compute_seconds),
            "paired_cumulative_saving": 1.0 - float(k5.policy_compute_seconds) / float(k10.policy_compute_seconds),
            "seconds_per_call_K10": float(k10.median_policy_seconds_per_call),
            "seconds_per_call_K5": float(k5.median_policy_seconds_per_call),
            "calls_K10": int(k10.policy_calls),
            "calls_K5": int(k5.policy_calls),
            "environment_steps_K10": int(k10.environment_steps),
            "environment_steps_K5": int(k5.environment_steps),
        })
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("NO_COMPLETE_CLOSED_LOOP_PAIRS")
    write_csv(RESULTS / "closed_loop_paired_metrics.csv", frame.to_dict("records"))
    available_tasks = [task for task in TASKS if bool((frame.task_id == task).any())]
    task_equal_success_difference = float(np.mean([
        frame.loc[frame.task_id == task, "success_difference"].mean() for task in available_tasks
    ]))
    point = {
        "frozen_pairs": 50,
        "complete_pairs": len(frame),
        "K10_success": int(frame.success_K10.sum()),
        "K5_success": int(frame.success_K5.sum()),
        "paired_success_difference": task_equal_success_difference,
        "K10_success_K5_failure": int((frame.success_K10 & ~frame.success_K5).sum()),
        "K10_failure_K5_success": int((~frame.success_K10 & frame.success_K5).sum()),
        "median_paired_cumulative_saving": float(frame.paired_cumulative_saving.median()),
        "median_seconds_per_call_K10": float(frame.seconds_per_call_K10.median()),
        "median_seconds_per_call_K5": float(frame.seconds_per_call_K5.median()),
        "median_per_call_saving": float(np.median(1.0 - frame.seconds_per_call_K5 / frame.seconds_per_call_K10)),
        "policy_call_count_K10": {"min": int(frame.calls_K10.min()), "median": float(frame.calls_K10.median()), "max": int(frame.calls_K10.max())},
        "policy_call_count_K5": {"min": int(frame.calls_K5.min()), "median": float(frame.calls_K5.median()), "max": int(frame.calls_K5.max())},
        "task_clusters_with_complete_pairs": len(available_tasks),
        "tasks_with_complete_pairs": available_tasks,
    }
    rng = np.random.default_rng(SEED)
    draws = {key: np.empty(N_BOOT) for key in ("success_difference", "cumulative_saving", "seconds_per_call_K10", "seconds_per_call_K5", "per_call_saving")}
    by_task = {task: frame.index[frame.task_id == task].to_numpy() for task in available_tasks}
    for b in range(N_BOOT):
        selected = []
        task_level_success = []
        for task in rng.choice(np.asarray(available_tasks, dtype=object), size=len(available_tasks), replace=True):
            available = by_task[task]
            draw = rng.choice(available, size=len(available), replace=True).tolist()
            selected.extend(draw)
            task_level_success.append(float(frame.loc[draw, "success_difference"].mean()))
        sample = frame.loc[selected]
        draws["success_difference"][b] = np.mean(task_level_success)
        draws["cumulative_saving"][b] = sample.paired_cumulative_saving.median()
        draws["seconds_per_call_K10"][b] = sample.seconds_per_call_K10.median()
        draws["seconds_per_call_K5"][b] = sample.seconds_per_call_K5.median()
        draws["per_call_saving"][b] = np.median(1.0 - sample.seconds_per_call_K5 / sample.seconds_per_call_K10)
    ci = {key: interval(value) for key, value in draws.items()}
    result = {
        "status": "COMPLETE" if len(frame) == 50 else "PARTIAL_COMPLETE_PAIRS_FROZEN_50_ESTIMAND_NOT_FULLY_OBSERVED",
        "point": point,
        "ci95": ci,
        "bootstrap": {"replicates": N_BOOT, "seed": SEED, "paired_seed_unit": True, "task_cluster_resampling": True, "conditional_on_tasks_with_complete_pairs": len(available_tasks) < 5},
        "policy_compute_excludes_simulator_and_rendering": True,
        "no_noninferiority_claim": True,
        "stage5_completion": completion,
    }
    atomic_json(RESULTS / "closed_loop_results.json", result)
    atomic_json(RESULTS / "closed_loop_analysis_identity.json", {
        "analysis_code": str(Path(__file__)), "analysis_code_sha256": sha(Path(__file__)),
        "closed_loop_raw_sha256": sha(RESULTS / "closed_loop_raw.parquet"),
        "result_sha256": sha(RESULTS / "closed_loop_results.json"),
    })
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()

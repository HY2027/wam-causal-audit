from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from common import RESULTS, TASKS, atomic_json, sha, write_csv


BOOTSTRAP_SEED = 20260923
N_BOOT = 10_000


def read_folds() -> dict[str, int]:
    with (RESULTS / "fold_assignment.csv").open() as stream:
        return {row["source_id"]: int(row["fold"]) for row in csv.DictReader(stream)}


def groups(tasks: np.ndarray) -> list[np.ndarray]:
    # During the hierarchical bootstrap, repeated task draws are assigned
    # synthetic labels so each of the five cluster draws retains equal weight.
    return [np.flatnonzero(tasks == task) for task in np.unique(tasks)]


def task_equal_mean(values: np.ndarray, tasks: np.ndarray) -> float:
    return float(np.mean([np.mean(values[idx]) for idx in groups(tasks) if len(idx)]))


def task_equal_mse(vectors: np.ndarray, tasks: np.ndarray) -> float:
    per_source = np.mean(np.square(vectors), axis=1)
    return task_equal_mean(per_source, tasks)


def fit_lambda(x: np.ndarray, y: np.ndarray, tasks: np.ndarray, indices: np.ndarray | None = None) -> float:
    if indices is None:
        indices = np.arange(len(tasks))
    numerator, denominator = [], []
    for task in np.unique(tasks[indices]):
        idx = indices[tasks[indices] == task]
        if len(idx):
            numerator.append(np.mean(np.sum(x[idx] * y[idx], axis=1)))
            denominator.append(np.mean(np.sum(x[idx] * x[idx], axis=1)))
    den = float(np.mean(denominator))
    return float(np.mean(numerator) / den) if den > 0 else float("nan")


def crossfit(x: np.ndarray, y: np.ndarray, tasks: np.ndarray, folds: np.ndarray) -> tuple[np.ndarray, dict[int, float]]:
    prediction = np.empty_like(y)
    lambdas = {}
    for fold in range(5):
        train = np.flatnonzero(folds != fold)
        lam = fit_lambda(x, y, tasks, train)
        lambdas[fold] = lam
        prediction[folds == fold] = lam * x[folds == fold]
    return prediction, lambdas


def metrics(x: np.ndarray, y: np.ndarray, tasks: np.ndarray, folds: np.ndarray) -> dict[str, Any]:
    prediction, lambdas = crossfit(x, y, tasks, folds)
    sse0 = task_equal_mse(y - x, tasks)
    sseg = task_equal_mse(y - prediction, tasks)
    return {
        "RMSE_unscaled_mm": float(np.sqrt(sse0)),
        "RMSE_gain_mm": float(np.sqrt(sseg)),
        "Q": float(1 - sseg / sse0) if sse0 > 0 else float("nan"),
        "fold_lambdas": {str(k): v for k, v in lambdas.items()},
        "full_data_lambda": fit_lambda(x, y, tasks),
        "prediction": prediction,
    }


def resample_indices(tasks: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    output: list[int] = []
    bootstrap_task_labels: list[str] = []
    sampled_tasks = rng.choice(np.asarray(TASKS, dtype=object), size=len(TASKS), replace=True)
    for draw_index, task in enumerate(sampled_tasks):
        available = np.flatnonzero(tasks == task)
        selected = rng.choice(available, size=len(available), replace=True).tolist()
        output.extend(selected)
        bootstrap_task_labels.extend([f"draw_{draw_index}:{task}"] * len(selected))
    return np.asarray(output, dtype=int), np.asarray(bootstrap_task_labels, dtype=str)


def interval(values: np.ndarray) -> list[float]:
    finite = values[np.isfinite(values)]
    return [float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975))]


def bootstrap_natural(r10: np.ndarray, r5: np.ndarray, tasks: np.ndarray, folds: np.ndarray) -> dict[str, Any]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    norm10 = np.linalg.norm(r10, axis=1)
    norm5 = np.linalg.norm(r5, axis=1)
    draws = {k: np.empty(N_BOOT) for k in (
        "K10_mean_norm_mm", "K5_mean_norm_mm", "norm_change_pct",
        "vector_rmse_mm", "RMSE_unscaled_mm", "RMSE_gain_mm", "Q",
    )}
    for b in range(N_BOOT):
        idx, bt = resample_indices(tasks, rng)
        bx, by, bf = r10[idx], r5[idx], folds[idx]
        n10 = task_equal_mean(norm10[idx], bt)
        n5 = task_equal_mean(norm5[idx], bt)
        draws["K10_mean_norm_mm"][b] = n10
        draws["K5_mean_norm_mm"][b] = n5
        draws["norm_change_pct"][b] = 100 * (n5 - n10) / n10
        draws["vector_rmse_mm"][b] = np.sqrt(task_equal_mse(by - bx, bt))
        gm = metrics(bx, by, bt, bf)
        for key in ("RMSE_unscaled_mm", "RMSE_gain_mm", "Q"):
            draws[key][b] = gm[key]
    return {key: {"ci95": interval(value)} for key, value in draws.items()}


def bootstrap_mechanism(x: np.ndarray, y: np.ndarray, tasks: np.ndarray, folds: np.ndarray) -> dict[str, Any]:
    rng = np.random.default_rng(BOOTSTRAP_SEED + 1)
    nx, ny = np.linalg.norm(x, axis=1), np.linalg.norm(y, axis=1)
    draws = {k: np.empty(N_BOOT) for k in (
        "K10_mean_norm_mm", "K5_mean_norm_mm", "norm_change_pct",
        "vector_rmse_mm", "RMSE_unscaled_mm", "RMSE_gain_mm", "Q",
    )}
    for b in range(N_BOOT):
        idx, bt = resample_indices(tasks, rng)
        bx, by, bf = x[idx], y[idx], folds[idx]
        mx, my = task_equal_mean(nx[idx], bt), task_equal_mean(ny[idx], bt)
        draws["K10_mean_norm_mm"][b] = mx
        draws["K5_mean_norm_mm"][b] = my
        draws["norm_change_pct"][b] = 100 * (my - mx) / mx if mx != 0 else np.nan
        draws["vector_rmse_mm"][b] = np.sqrt(task_equal_mse(by - bx, bt))
        gain = metrics(bx, by, bt, bf)
        draws["RMSE_unscaled_mm"][b] = gain["RMSE_unscaled_mm"]
        draws["RMSE_gain_mm"][b] = gain["RMSE_gain_mm"]
        draws["Q"][b] = gain["Q"]
    return {key: {"ci95": interval(value)} for key, value in draws.items()}


def main() -> None:
    if json.loads((RESULTS / "stage3_completion.json").read_text())["status"] != "COMPLETE":
        raise RuntimeError("STAGE3_INCOMPLETE")
    if json.loads((RESULTS / "stage4_completion.json").read_text())["status"] != "COMPLETE":
        raise RuntimeError("STAGE4_INCOMPLETE")
    natural = np.load(RESULTS / "natural_response_raw.npz")
    source_ids = natural["source_id"].astype(str)
    tasks = natural["task_id"].astype(str)
    fold_map = read_folds()
    folds = np.asarray([fold_map[source] for source in source_ids], dtype=int)
    r10 = (natural["dplus_k10_radial"] - natural["z1_k10_radial"]) * 1000.0
    r5 = (natural["dplus_k5_radial"] - natural["z1_k5_radial"]) * 1000.0
    norm10, norm5 = np.linalg.norm(r10, axis=1), np.linalg.norm(r5, axis=1)
    mean10, mean5 = task_equal_mean(norm10, tasks), task_equal_mean(norm5, tasks)
    natural_point = {
        "K10_mean_norm_mm": mean10,
        "K5_mean_norm_mm": mean5,
        "norm_change_pct": 100 * (mean5 - mean10) / mean10,
        "vector_rmse_mm": float(np.sqrt(task_equal_mse(r5-r10, tasks))),
        **{k:v for k,v in metrics(r10, r5, tasks, folds).items() if k != "prediction"},
    }
    natural_ci = bootstrap_natural(r10, r5, tasks, folds)

    strict = np.load(RESULTS / "strict_source_raw.npz")
    if not np.array_equal(strict["source_id"].astype(str), source_ids):
        raise RuntimeError("STRICT_NATURAL_SOURCE_ORDER_MISMATCH")
    effects = {}
    formulas = {
        "future_given_Cdonor": ("A11", "A10"),
        "future_given_Crecipient": ("A01", "A00"),
        "current_given_Frecipient": ("A10", "A00"),
        "current_given_Fdonor": ("A11", "A01"),
        "joint": ("A11", "A00"),
    }
    for name, (hi, lo) in formulas.items():
        x = (strict[f"K10_{hi}_radial"] - strict[f"K10_{lo}_radial"]) * 1000.0
        y = (strict[f"K5_{hi}_radial"] - strict[f"K5_{lo}_radial"]) * 1000.0
        mx, my = task_equal_mean(np.linalg.norm(x, axis=1), tasks), task_equal_mean(np.linalg.norm(y, axis=1), tasks)
        point = {
            "K10_mean_norm_mm": mx, "K5_mean_norm_mm": my,
            "norm_change_pct": 100*(my-mx)/mx if mx != 0 else None,
            "vector_rmse_mm": float(np.sqrt(task_equal_mse(y-x, tasks))),
            "signed_mean_K10_mm": task_equal_mean(np.mean(x, axis=1), tasks),
            "signed_mean_K5_mm": task_equal_mean(np.mean(y, axis=1), tasks),
            "gain": {k:v for k,v in metrics(x,y,tasks,folds).items() if k != "prediction"},
        }
        point["bootstrap"] = bootstrap_mechanism(x,y,tasks,folds)
        effects[name] = point
    interaction10 = (strict["K10_A11_radial"]-strict["K10_A10_radial"]-strict["K10_A01_radial"]+strict["K10_A00_radial"])*1000
    interaction5 = (strict["K5_A11_radial"]-strict["K5_A10_radial"]-strict["K5_A01_radial"]+strict["K5_A00_radial"])*1000
    effects["interaction"] = {
        "K10_signed_mean_mm": task_equal_mean(np.mean(interaction10,axis=1), tasks),
        "K5_signed_mean_mm": task_equal_mean(np.mean(interaction5,axis=1), tasks),
        "vector_rmse_mm": float(np.sqrt(task_equal_mse(interaction5-interaction10,tasks))),
    }
    gain = metrics(r10,r5,tasks,folds)
    prediction_rows=[]
    for i,source in enumerate(source_ids):
        for position in range(32):
            prediction_rows.append({
                "source_id":source,"task_id":tasks[i],"fold":int(folds[i]),"position":position,
                "K10_response_mm":r10[i,position],"K5_response_mm":r5[i,position],
                "lambda_crossfit":gain["fold_lambdas"][str(folds[i])],
                "prediction_mm":gain["prediction"][i,position],
                "unscaled_residual_mm":r5[i,position]-r10[i,position],
                "gain_residual_mm":r5[i,position]-gain["prediction"][i,position],
            })
    try:
        import pandas as pd
        pd.DataFrame(prediction_rows).to_parquet(RESULTS/"gain_predictions.parquet",index=False)
    except Exception as exc:
        write_csv(RESULTS/"gain_predictions.csv",prediction_rows)
        atomic_json(RESULTS/"gain_predictions_parquet_failure.json",{"error":repr(exc),"csv_fallback":True})
    bootstrap_rows=[]
    for metric,value in natural_ci.items():
        bootstrap_rows.append({"analysis":"natural_primary34","metric":metric,"estimate":natural_point.get(metric),"ci_low":value["ci95"][0],"ci_high":value["ci95"][1],"replicates":N_BOOT})
    future=effects["future_given_Cdonor"]
    for metric,value in future["bootstrap"].items():
        estimate=future[metric] if metric in future else future["gain"].get(metric)
        bootstrap_rows.append({"analysis":"future_given_Cdonor_primary34","metric":metric,"estimate":estimate,"ci_low":value["ci95"][0],"ci_high":value["ci95"][1],"replicates":N_BOOT})
    write_csv(RESULTS/"bootstrap_summary.csv",bootstrap_rows)
    result={
        "status":"PRIMARY_OPENLOOP_AND_MECHANISM_COMPLETE",
        "identity":"ORIGINAL_FROZEN_34",
        "source_count":len(source_ids),"task_count":len(set(tasks)),
        "natural":natural_point,"natural_bootstrap":natural_ci,
        "strict_effects":effects,
        "bootstrap":{"replicates":N_BOOT,"seed":BOOTSTRAP_SEED,"task_source_paired":True},
        "A11_minus_A00_not_natural_response":True,
        "units":"mm FK-derived predicted EEF target radial coordinate",
    }
    atomic_json(RESULTS/"primary_results.json",result)
    atomic_json(RESULTS/"analysis_identity.json",{
        "analysis_code":str(Path(__file__)),"analysis_code_sha256":sha(Path(__file__)),
        "natural_raw_sha256":sha(RESULTS/"natural_response_raw.npz"),
        "strict_raw_sha256":sha(RESULTS/"strict_source_raw.npz"),
        "primary_results_sha256":sha(RESULTS/"primary_results.json"),
    })
    print(json.dumps({"status":result["status"],"natural":natural_point,"future_Cdonor":future},sort_keys=True))


if __name__=="__main__":
    main()

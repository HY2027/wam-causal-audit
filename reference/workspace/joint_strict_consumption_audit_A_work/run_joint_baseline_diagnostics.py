#!/usr/bin/env python3
"""Frozen Joint ordinary-sensitivity and CKA diagnostics on old B states."""
from __future__ import annotations
from wam_causal_audit.paths import resolve as _release_path

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch

WORK = Path(__file__).resolve().parent
sys.path.insert(0, str(WORK))
import run_experiment_a as A  # noqa: E402
import run_c_early_stop as C  # noqa: E402
from capture import make_capture  # noqa: E402

ROOT = Path(_release_path('@DATA@/wam_factor_routing_v5/joint_idm_compute_confirmatory_v1_20260909'))
OUT = ROOT / "joint_new_trajectory_confirmatory" / "baseline_diagnostics_old_development"
OLD_BASE = Path(_release_path('@DATA@/wam_factor_routing_v5/joint_experiment_B_distance_v1'))
SYNC = OLD_BASE / "B_SYNC_V1"
BROOT = OLD_BASE / "B_forward_development_validation_v1"
STATS = Path(_release_path('@DATA@/BadWAM/models/LIQIIIII/badwam-libero-joint-wam/dataset_stats.json'))
AMPS = (0.01, 0.03, 0.10)
DIRECTION_SEEDS = (920101, 920102, 920103, 920104)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(16 * 1024 * 1024), b""): h.update(block)
    return h.hexdigest()


def dump(path: Path, value: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(A.jsonable(value), indent=2, sort_keys=True) + "\n"); tmp.replace(path)


def write(path: Path, rows: list[dict[str, Any]]):
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys: keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys); writer.writeheader(); writer.writerows(rows)
    tmp.replace(path)


def states() -> list[dict[str, Any]]:
    rows = pd.read_csv(OLD_BASE / "B_split_registry.csv")
    rows = rows[rows.selection_status.str.startswith("SELECTED")].sort_values(["task_id", "trajectory_id", "policy_call"])
    return rows.to_dict("records")


def calibration_states() -> list[dict[str, Any]]:
    out = []
    for task, group in pd.DataFrame(states()).query("split == 'fit'").groupby("task_id"):
        trajs = sorted(group.trajectory_id.unique())
        for trajectory in (trajs[0], trajs[-1]):
            out.append(group[group.trajectory_id == trajectory].sort_values("policy_call").iloc[0].to_dict())
    return out


def obs_path(row: Mapping[str, Any]) -> Path:
    return SYNC / "synced_recipients" / str(row["candidate_id"]) / "recipient_policy_observation.npz"


def seed(row: Mapping[str, Any]) -> int:
    return 820000 + int(row["task_id"]) * 1000 + int(row["trajectory_id"]) * 20 + int(row["policy_call"])


def axis_map() -> dict[str, np.ndarray]:
    data = pd.read_csv(BROOT / "B_factorial_metrics.csv").drop_duplicates("candidate_id")
    return {str(r.candidate_id): np.asarray([r.projection_axis_x, r.projection_axis_y, r.projection_axis_z], float) for r in data.itertuples()}


def denormalizer():
    stats = json.loads(STATS.read_text())["action"]["default"]
    low = np.asarray(stats["stepwise_min"], float); high = np.asarray(stats["stepwise_max"], float)
    if low.shape[0] == 1: low = np.repeat(low, 32, axis=0); high = np.repeat(high, 32, axis=0)
    def fn(value):
        result = (np.asarray(value, float) + 1.0) * 0.5 * (high - low) + low
        result[..., 6] = -(result[..., 6] * 2.0 - 1.0)
        return result
    return fn


def schedule_mapping(schedule: Mapping[int, list[dict[str, torch.Tensor]]]):
    return {(step, layer): values for step, layers in schedule.items() for layer, values in enumerate(layers)}


def perturb_future(cache: Mapping[tuple[int, int], Mapping[str, torch.Tensor]], amplitude: float,
                   direction_seed: int, sign: int):
    result: dict[tuple[int, int], dict[str, torch.Tensor]] = {}
    for (step, layer), values in sorted(cache.items()):
        result[(step, layer)] = {}
        for ci, component in enumerate(("k", "v")):
            value = values[component].detach().clone(); tpg = value.shape[1] // 3
            future = value[:, tpg:]
            rms = torch.sqrt(torch.mean(future.float() ** 2)).to(future.dtype)
            generator = torch.Generator(device=value.device).manual_seed(direction_seed + 100003 * step + 1009 * layer + 37 * ci)
            noise = torch.randn(future.shape, generator=generator, device=value.device, dtype=value.dtype)
            noise_rms = torch.sqrt(torch.mean(noise.float() ** 2)).to(noise.dtype)
            value[:, tpg:] = future + sign * amplitude * rms * noise / torch.clamp(noise_rms, min=torch.finfo(noise.dtype).tiny)
            result[(step, layer)][component] = value
    return result


def linear_cka(x: torch.Tensor, y: torch.Tensor) -> tuple[float, bool]:
    x = x.detach().float()[0]; y = y.detach().float()[0]
    x = x - x.mean(dim=0, keepdim=True); y = y - y.mean(dim=0, keepdim=True)
    gx = x @ x.T; gy = y @ y.T
    numerator = torch.sum(gx * gy); denominator = torch.sqrt(torch.sum(gx * gx) * torch.sum(gy * gy))
    if not torch.isfinite(denominator) or float(denominator) <= torch.finfo(torch.float32).eps:
        return math.nan, False
    value = float((numerator / denominator).item())
    return value, math.isfinite(value)


def cka_rows(row: Mapping[str, Any], schedule: Mapping[int, list[dict[str, torch.Tensor]]]):
    rows = []
    for k in (8, 5):
        values = []
        for layer, (early, final) in enumerate(zip(schedule[k - 1], schedule[9])):
            tpg = early["k"].shape[1] // 3
            for component in ("k", "v"):
                value, defined = linear_cka(early[component][:, tpg:], final[component][:, tpg:])
                rows.append({"candidate_id": row["candidate_id"], "task_id": row["task_id"],
                             "trajectory_id": row["trajectory_id"], "policy_call": row["policy_call"],
                             "split": row["split"], "K": k, "layer": layer, "component": component,
                             "linear_CKA": value, "defined": defined})
                if defined: values.append(value)
        rows.append({"candidate_id": row["candidate_id"], "task_id": row["task_id"],
                     "trajectory_id": row["trajectory_id"], "policy_call": row["policy_call"],
                     "split": row["split"], "K": k, "layer": "AGGREGATE", "component": "MEAN_LAYER_COMPONENT",
                     "linear_CKA": float(np.mean(values)) if values else math.nan, "defined": bool(values)})
    return rows


def component_metrics(delta: np.ndarray, axis: np.ndarray):
    tr = delta[:, :3]; radial = tr @ axis
    return {"radial_l2": float(np.linalg.norm(radial)), "translation_l2": float(np.linalg.norm(tr)),
            "rotation_l2": float(np.linalg.norm(delta[:, 3:6])), "gripper_l2": float(np.linalg.norm(delta[:, 6]))}


def one(runner, row: Mapping[str, Any], amplitudes: tuple[float, ...], axis: np.ndarray):
    item = runner._prepared(A.load_npz(obs_path(row)), str(row["instruction"])); s = seed(row)
    start = time.perf_counter()
    native, diag, schedule = C.early_stop_infer(runner, item, s, 10, capture_all=True)
    cache = schedule_mapping(schedule); dn = denormalizer(); native_denorm = dn(native.numpy())
    outputs = []
    for amplitude in amplitudes:
        for direction_seed in DIRECTION_SEEDS:
            for sign in (-1, 1):
                changed = perturb_future(cache, amplitude, direction_seed, sign)
                action, run = A.strict_infer(runner, item, s, cache, changed)
                delta_norm = action.numpy() - native.numpy(); delta_denorm = dn(action.numpy()) - native_denorm
                outputs.append({"candidate_id": row["candidate_id"], "task_id": row["task_id"],
                                "trajectory_id": row["trajectory_id"], "policy_call": row["policy_call"],
                                "split": row["split"], "amplitude_rms": amplitude,
                                "direction_seed": direction_seed, "sign": sign,
                                "normalized_full_action_relative_change": float(np.linalg.norm(delta_norm) / max(np.linalg.norm(native.numpy()), np.finfo(float).eps)),
                                **component_metrics(delta_denorm, axis),
                                "video_calls": run["hook"]["video_calls"], "action_calls": run["hook"]["action_calls"]})
    elapsed = time.perf_counter() - start
    return outputs, cka_rows(row, schedule), {"candidate_id": row["candidate_id"], "seconds": elapsed,
        "native_world_steps": diag["world_branch_steps"], "native_action_steps": diag["action_branch_steps"],
        "perturbed_inferences": len(outputs), "peak_memory_bytes": diag["peak_memory_bytes"]}


def freeze_spec():
    spec = {
        "status": "JOINT_BASELINE_DIAGNOSTIC_EXECUTION_SPEC_FROZEN",
        "parent_baseline_protocol_sha256": sha(ROOT / "joint_new_trajectory_confirmatory" / "joint_baseline_prediction_protocol.json"),
        "ordinary_sensitivity": {
            "location": "actual future K/V consumer, all 30 layers x 10 denoising events",
            "seeds": list(DIRECTION_SEEDS), "signs": [-1, 1], "amplitude_candidates_rms": list(AMPS),
            "noise": "independent standard normal for K/V and each registered consumer event; normalized to event-tensor RMS; repeated hook uses fixed tensor",
            "calibration_states": "old fit split, each task first call of lowest and highest trajectory ID",
            "selection": "smallest amplitude whose median normalized full-action relative change lies in [0.01,0.05], otherwise closest to 0.03",
            "state_feature": "mean denormalized radial_l2 over 4 directions x 2 signs; other units retained separately",
        },
        "representation_similarity": {
            "metric": "linear CKA", "samples": "future tokens", "features": "channels",
            "comparisons": {"K8": "world forward 8 versus 10", "K5": "world forward 5 versus 10"},
            "aggregation": "unweighted mean over defined layer x K/V values",
        },
        "diagnostic_cost_included": True, "full_inference_required": True,
        "no_new_trajectory_label_read": True, "no_model_weight_change": True,
        "created_unix": time.time(), "code_sha256": sha(Path(__file__)),
    }
    dump(OUT / "joint_baseline_diagnostic_execution_spec.json", spec)
    print(json.dumps({"status": spec["status"], "sha256": sha(OUT / "joint_baseline_diagnostic_execution_spec.json")}))


def run(mode: str, gpu: int, shard: int, shards: int):
    all_states = calibration_states() if mode == "calibration" else states()
    own = [row for index, row in enumerate(all_states) if index % shards == shard]
    if mode == "formal":
        amp = float(json.loads((OUT / "ordinary_sensitivity_amplitude_freeze.json").read_text())["selected_amplitude_rms"])
        amplitudes = (amp,)
    else: amplitudes = AMPS
    runner = make_capture("joint", gpu); axes = axis_map(); sensitivity = []; cka = []; costs = []
    for index, row in enumerate(own, 1):
        a, b, c = one(runner, row, amplitudes, axes[str(row["candidate_id"])])
        sensitivity += a; cka += b; costs.append(c)
        print(json.dumps({"mode": mode, "shard": shard, "state": index, "total": len(own), "candidate": row["candidate_id"]}), flush=True)
    prefix = "calibration" if mode == "calibration" else "formal"
    write(OUT / f"{prefix}_shards/shard_{shard:02d}_sensitivity.csv", sensitivity)
    write(OUT / f"{prefix}_shards/shard_{shard:02d}_cka.csv", cka)
    write(OUT / f"{prefix}_shards/shard_{shard:02d}_cost.csv", costs)
    dump(OUT / f"{prefix}_shards/shard_{shard:02d}.json", {"status": "COMPLETE", "gpu": gpu,
         "states": len(own), "sensitivity_rows": len(sensitivity), "cka_rows": len(cka),
         "code_sha256": sha(Path(__file__))})


def freeze_amplitude():
    parts = [pd.read_csv(path) for path in sorted((OUT / "calibration_shards").glob("shard_*_sensitivity.csv"))]
    data = pd.concat(parts, ignore_index=True)
    medians = data.groupby("amplitude_rms").normalized_full_action_relative_change.median().to_dict()
    candidates = [a for a in AMPS if 0.01 <= medians[a] <= 0.05]
    selected = min(candidates) if candidates else min(AMPS, key=lambda a: abs(medians[a] - 0.03))
    value = {"status": "ORDINARY_SENSITIVITY_AMPLITUDE_FROZEN_BEFORE_NEW_TRAJECTORIES",
             "selected_amplitude_rms": selected, "median_relative_change_by_amplitude": medians,
             "calibration_rows": len(data), "new_trajectory_effect_read": False,
             "calibration_input_hashes": {p.name: sha(p) for p in sorted((OUT / "calibration_shards").glob("*.csv"))},
             "created_unix": time.time(), "code_sha256": sha(Path(__file__))}
    dump(OUT / "ordinary_sensitivity_amplitude_freeze.json", value); print(json.dumps(value))


def finalize():
    for metric in ("sensitivity", "cka", "cost"):
        parts = [pd.read_csv(path) for path in sorted((OUT / "formal_shards").glob(f"shard_*_{metric}.csv"))]
        data = pd.concat(parts, ignore_index=True)
        path = OUT / f"old_B_{metric}_diagnostics.csv"; data.to_csv(path, index=False)
    manifest = {"status": "JOINT_OLD_DEVELOPMENT_BASELINE_DIAGNOSTICS_COMPLETE",
                "states": len(states()), "amplitude_freeze_sha256": sha(OUT / "ordinary_sensitivity_amplitude_freeze.json"),
                "sensitivity_sha256": sha(OUT / "old_B_sensitivity_diagnostics.csv"),
                "cka_sha256": sha(OUT / "old_B_cka_diagnostics.csv"),
                "cost_sha256": sha(OUT / "old_B_cost_diagnostics.csv"),
                "new_trajectory_effect_read": False, "created_unix": time.time()}
    dump(OUT / "joint_old_B_diagnostic_manifest.json", manifest); print(json.dumps(manifest))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--mode", choices=["freeze-spec", "calibration", "freeze-amplitude", "formal", "finalize"], required=True)
    parser.add_argument("--gpu", type=int, default=0); parser.add_argument("--shard", type=int, default=0); parser.add_argument("--shards", type=int, default=3)
    args = parser.parse_args()
    if args.mode == "freeze-spec": freeze_spec()
    elif args.mode in ("calibration", "formal"): run(args.mode, args.gpu, args.shard, args.shards)
    elif args.mode == "freeze-amplitude": freeze_amplitude()
    else: finalize()


if __name__ == "__main__": main()

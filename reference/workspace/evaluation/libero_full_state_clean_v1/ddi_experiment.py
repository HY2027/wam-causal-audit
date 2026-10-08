#!/usr/bin/env python3
"""Direct Drift Injection (DDI) experiment for the full-state LIBERO pool.

The branch point is an approach-phase clean environment step whose EEF is
8--12 cm from the first manipulation target.  The clean policy prefix is
replayed solely to reconstruct model-side state (KV cache / RNG); the MuJoCo
state at the branch point is then restored exactly from the audited clean
trace before the arm-only teleport is applied.
"""
from __future__ import annotations
from wam_causal_audit.paths import resolve as _release_path

import argparse
import hashlib
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
LIBERO_ROOT = Path(_release_path('@WORKSPACE@/LIBERO'))
for path in (ROOT, LIBERO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from full_state_writer import FullStateRolloutWriter  # noqa: E402
from policy_adapters import (  # noqa: E402
    LingBotAdapter,
    Pi05Adapter,
    PolicyAdapter,
    PolicyOutput,
    VLAJEPAAdapter,
    make_adapter,
)
from run_clean import ENVIRONMENT_SEED, _env_step, _lingbot_keyframe  # noqa: E402
from task_specs import TASK_SPECS  # noqa: E402


CLEAN_ROOT = Path(_release_path('@WORKSPACE@/results/libero_full_state_clean_v1'))
DEFAULT_OUTPUT_ROOT = Path(_release_path('@WORKSPACE@/results/libero_ddi_v1'))
TASKS = (0, 2, 3, 5, 9)
D_CM_VALUES = (0.0, 1.0, 2.0, 3.0, 5.0, 8.0)
N_SAMPLES = 8


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _load_calls(clean_dir: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (clean_dir / "policy_calls.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]


def _branch_dir(root: Path, model: str, task_id: int, init_index: int, d_cm: float) -> Path:
    return root / "ddi" / model / f"task{task_id:02d}" / f"init_{init_index:03d}" / f"d_{d_cm:04.1f}cm"


def _safe_entity(entity: str) -> str:
    return entity.replace("-", "_")


def _source_dir(model: str, task_id: int, init_index: int) -> Path:
    return CLEAN_ROOT / "clean" / model / f"task{task_id:02d}" / f"init_{init_index:03d}"


def _vector(value: Any, width: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    return array if width is None else array[:width]


def _target_position(row: pd.Series, target: str) -> np.ndarray:
    return _vector(row[f"entity__{_safe_entity(target)}__pos"], 3)


def _choose_trigger(
    steps: pd.DataFrame,
    task_id: int,
    *,
    require_policy_boundary: bool = False,
    distance_band_m: tuple[float, float] = (0.08, 0.12),
    target_distance_m: float = 0.10,
) -> tuple[pd.Series, dict[str, Any]]:
    spec = TASK_SPECS[task_id]
    target = spec.manipulated_objects[0]
    phase = steps["phase"].astype(str)
    # The automatic task3 phase segmenter assigns the physical approach to
    # the bowl to the tail of `open_drawer` (the drawer handle and bowl are
    # co-located in this layout).  Include that first compound stage so the
    # 8--12 cm geometric criterion, rather than a brittle label boundary,
    # determines the injection point.
    is_approach = phase.str.startswith("approach_")
    if task_id == 3:
        is_approach |= phase.eq("open_drawer")
    candidates = steps[is_approach].copy()
    if require_policy_boundary:
        candidates = candidates[candidates["policy_call_idx"].notna()].copy()
    if candidates.empty:
        raise RuntimeError("no approach-phase policy boundary in clean trace")
    distances = []
    for _, row in candidates.iterrows():
        distances.append(float(np.linalg.norm(_vector(row["eef_pos"], 3) - _target_position(row, target))))
    candidates["_target_distance"] = distances
    lower, upper = (float(distance_band_m[0]), float(distance_band_m[1]))
    in_band = candidates[(candidates["_target_distance"] >= lower) & (candidates["_target_distance"] <= upper)]
    distance_band_exception = False
    if in_band.empty:
        # Preserve the model-specific clean seed pool.  A single Pi task5
        # trace stays at a 13.19-cm EEF-to-object-origin offset throughout
        # its successful grasp, so no discrete 8--12 cm row exists.  Keep
        # the nearest clean approach state rather than silently substituting
        # a different seed; flag it prominently for stratified reporting.
        selected = candidates.iloc[int(np.argmin(np.abs(candidates["_target_distance"].to_numpy() - target_distance_m)))]
        distance_band_exception = True
    else:
        selected = in_band.iloc[int(np.argmin(np.abs(in_band["_target_distance"].to_numpy() - target_distance_m)))]
    prior_calls = steps[(steps["policy_call_idx"].notna()) & (steps["env_step"] <= int(selected["env_step"]))]
    if prior_calls.empty:
        raise RuntimeError("injection point occurs before the first policy call")
    active_call_idx = int(prior_calls.iloc[-1]["policy_call_idx"])
    at_policy_boundary = bool(pd.notna(selected["policy_call_idx"]))
    return selected, {
        "target_entity": target,
        "target_distance_clean_m": float(selected["_target_distance"]),
        "requested_distance_band_m": [lower, upper],
        "requested_target_distance_m": float(target_distance_m),
        "distance_band_exception": distance_band_exception,
        "phase": str(selected["phase"]),
        "env_step": int(selected["env_step"]),
        "active_clean_policy_call_idx": active_call_idx,
        "at_policy_boundary": at_policy_boundary,
        "selection": (
            "approach-phase policy boundary closest to 0.10m within [0.08,0.12]m; clear action queue and force a fresh inference"
            if require_policy_boundary and not distance_band_exception else
            "nearest available approach policy boundary; no discrete 8--12cm boundary exists (flagged exception); clear action queue and force a fresh inference"
            if require_policy_boundary else
            "approach-phase env step closest to 0.10m within [0.08,0.12]m; clear action queue and force a fresh inference"
            if not distance_band_exception else
            "nearest available approach env step; no discrete 8--12cm point exists (flagged exception); clear action queue and force a fresh inference"
        ),
    }


def _clean_policy_obs(clean_dir: Path, call: Mapping[str, Any], row: pd.Series) -> dict[str, np.ndarray]:
    with np.load(clean_dir / call["obs_frame_ref"]["path"], allow_pickle=False) as arrays:
        agent = np.asarray(arrays["agentview_image"]).copy()
        wrist = np.asarray(arrays["robot0_eye_in_hand_image"]).copy()
    return {
        "agentview_image": agent,
        "robot0_eye_in_hand_image": wrist,
        "robot0_eef_pos": _vector(row["eef_obs_pos"], 3).astype(np.float32),
        "robot0_eef_quat": _vector(row["eef_obs_quat_xyzw"], 4).astype(np.float32),
        "robot0_gripper_qpos": _vector(row["gripper_qpos"], 2).astype(np.float32),
        "robot0_gripper_qvel": _vector(row["gripper_qvel"], 2).astype(np.float32),
    }


def _refresh_obs(env: Any) -> dict[str, Any]:
    env.env.sim.forward()
    env._update_observables(force=True)
    return dict(env.env._get_observations())


def _restore_row(env: Any, row: pd.Series, qpos_override: np.ndarray | None = None) -> dict[str, Any]:
    from robosuite.utils.binding_utils import MjSimState

    sim = env.env.sim
    qpos = _vector(row["sim_qpos"]) if qpos_override is None else np.asarray(qpos_override, dtype=np.float64).copy()
    qvel = _vector(row["sim_qvel"])
    sim.set_state(MjSimState(float(row["sim_time"]), qpos.copy(), qvel.copy()))
    for attr, column in (("ctrl", "sim_ctrl"), ("mocap_pos", "sim_mocap_pos"), ("mocap_quat", "sim_mocap_quat"), ("userdata", "sim_userdata")):
        saved = _vector(row[column])
        target = np.asarray(getattr(sim.data, attr))
        if saved.size == target.size:
            target[:] = saved.reshape(target.shape)
    sim.forward()
    _synchronize_low_level_controller(env)
    return _refresh_obs(env)


def _synchronize_low_level_controller(env: Any) -> None:
    """Clear OSC's stale goal after a teleport without changing physical state."""
    robot = env.env.robots[0]
    controller = robot.controller
    indices = getattr(robot, "_ref_joint_pos_indexes", None)
    if indices is None or not hasattr(controller, "update_initial_joints"):
        return
    controller.update_initial_joints(np.asarray(env.env.sim.data.qpos[indices], dtype=np.float64).copy())


def _direction_for(model: str, task_id: int, init_index: int) -> tuple[np.ndarray, int]:
    token = f"ddi-v1/{model}/task{task_id}/init{init_index}".encode()
    seed = int.from_bytes(hashlib.sha256(token).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    direction = rng.standard_normal(7)
    direction /= np.linalg.norm(direction)
    return direction.astype(np.float64), seed


def _arm_qpos_indices(meta: Mapping[str, Any]) -> np.ndarray:
    result = []
    for name in meta["dof_map"]["categories"]["robot_arm"]:
        start, stop = meta["dof_map"]["joints"][name]["qpos_slice"]
        if stop - start != 1:
            raise RuntimeError(f"non-scalar arm joint {name}")
        result.append(int(start))
    if len(result) != 7:
        raise RuntimeError(f"expected 7 arm qpos indices, got {result}")
    return np.asarray(result, dtype=np.int64)


def _arm_joint_limits(sim: Any, meta: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    lower, upper = [], []
    for name in meta["dof_map"]["categories"]["robot_arm"]:
        joint_id = int(meta["dof_map"]["joints"][name]["joint_id"])
        bounds = np.asarray(sim.model.jnt_range[joint_id], dtype=np.float64)
        lower.append(bounds[0])
        upper.append(bounds[1])
    return np.asarray(lower), np.asarray(upper)


def _eef_pos(sim: Any) -> np.ndarray:
    return np.asarray(sim.data.get_body_xpos("gripper0_eef"), dtype=np.float64).copy()


def _new_robot_contacts(sim: Any, baseline: set[tuple[str, str]]) -> list[dict[str, str]]:
    result = []
    for index in range(int(sim.data.ncon)):
        contact = sim.data.contact[index]
        geom1 = sim.model.geom_id2name(int(contact.geom1)) or f"geom_{int(contact.geom1)}"
        geom2 = sim.model.geom_id2name(int(contact.geom2)) or f"geom_{int(contact.geom2)}"
        pair = tuple(sorted((str(geom1), str(geom2))))
        if pair in baseline:
            continue
        body1 = sim.model.body_id2name(int(sim.model.geom_bodyid[int(contact.geom1)])) or ""
        body2 = sim.model.body_id2name(int(sim.model.geom_bodyid[int(contact.geom2)])) or ""
        robot1 = body1.startswith("robot0_") or body1.startswith("gripper0_")
        robot2 = body2.startswith("robot0_") or body2.startswith("gripper0_")
        if robot1 ^ robot2:
            distance = float(contact.dist)
            result.append({
                "geom_a": pair[0], "geom_b": pair[1], "body_a": body1, "body_b": body2,
                "distance_m": distance,
                "penetration_depth_m": max(0.0, -distance),
            })
    return result


def _contact_set(sim: Any) -> set[tuple[str, str]]:
    pairs = set()
    for index in range(int(sim.data.ncon)):
        contact = sim.data.contact[index]
        g1 = sim.model.geom_id2name(int(contact.geom1)) or f"geom_{int(contact.geom1)}"
        g2 = sim.model.geom_id2name(int(contact.geom2)) or f"geom_{int(contact.geom2)}"
        pairs.add(tuple(sorted((str(g1), str(g2)))))
    return pairs


def _drift_qpos(
    env: Any,
    row: pd.Series,
    meta: Mapping[str, Any],
    direction: np.ndarray,
    d_m: float,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    sim = env.env.sim
    clean_qpos = _vector(row["sim_qpos"])
    clean_eef = _eef_pos(sim)
    if d_m == 0.0:
        return clean_qpos.copy(), {
            "requested_d_m": 0.0,
            "realized_d_m": 0.0,
            "q_delta": np.zeros(7),
            "eef_delta_xyz": np.zeros(3),
            "status": "ok",
        }
    indices = _arm_qpos_indices(meta)
    lower, upper = _arm_joint_limits(sim, meta)
    q_arm = clean_qpos[indices]
    max_alpha = math.inf
    for value, unit, lo, hi in zip(q_arm, direction, lower, upper):
        if unit > 1e-12:
            max_alpha = min(max_alpha, (hi - value) / unit)
        elif unit < -1e-12:
            max_alpha = min(max_alpha, (lo - value) / unit)
    max_alpha = max(0.0, 0.98 * float(max_alpha))
    if not math.isfinite(max_alpha) or max_alpha <= 0.0:
        return None, {"requested_d_m": d_m, "status": "invalid_joint_limit", "q_delta": direction}

    def evaluate(alpha: float) -> tuple[np.ndarray, np.ndarray, float]:
        candidate = clean_qpos.copy()
        candidate[indices] = q_arm + alpha * direction
        _restore_row(env, row, candidate)
        eef = _eef_pos(sim)
        return candidate, eef, float(np.linalg.norm(eef - clean_eef))

    # Find the first crossing while retaining a fixed joint-space direction.
    grid = np.linspace(0.0, max_alpha, 97)
    values = [evaluate(float(alpha))[2] for alpha in grid]
    crossings = np.flatnonzero(np.asarray(values) >= d_m)
    if crossings.size == 0:
        _, eef, maximum = evaluate(float(grid[-1]))
        return None, {
            "requested_d_m": d_m,
            "status": "invalid_unreachable_before_joint_limit",
            "max_reachable_d_m": maximum,
            "q_delta": direction,
            "eef_delta_xyz_at_limit": eef - clean_eef,
        }
    right_index = int(crossings[0])
    left, right = float(grid[max(0, right_index - 1)]), float(grid[right_index])
    candidate = clean_qpos.copy()
    eef = clean_eef.copy()
    for _ in range(18):
        mid = 0.5 * (left + right)
        candidate, eef, current = evaluate(mid)
        if current < d_m:
            left = mid
        else:
            right = mid
    candidate, eef, current = evaluate(0.5 * (left + right))
    return candidate, {
        "requested_d_m": d_m,
        "realized_d_m": current,
        "q_delta": candidate[indices] - q_arm,
        "eef_delta_xyz": eef - clean_eef,
        "status": "ok",
    }


def _snapshot_rng(adapter: PolicyAdapter) -> dict[str, Any] | None:
    if isinstance(adapter, Pi05Adapter):
        response = dict(adapter.client.infer({"__temporal_control__": {"operation": "snapshot"}}))
        return {"kind": "pi05", "key_data": np.asarray(response["temporal_rng"]["key_data"], dtype=np.uint32)}
    if isinstance(adapter, VLAJEPAAdapter):
        return {"kind": "vla_jepa", "state": dict(adapter.policy.client.rng_snapshot())}
    return None


def _restore_rng(adapter: PolicyAdapter, snapshot: Mapping[str, Any] | None) -> None:
    if snapshot is None:
        return
    if snapshot["kind"] == "pi05":
        adapter.client.infer({"__temporal_control__": {"operation": "restore", "key_data": snapshot["key_data"]}})
    elif snapshot["kind"] == "vla_jepa":
        adapter.policy.client.rng_restore(dict(snapshot["state"]))
    else:
        raise ValueError(snapshot["kind"])


def _sample_chunks(
    adapter: PolicyAdapter,
    raw_obs: Mapping[str, Any],
    call_idx: int,
    first_call: bool,
    n: int = N_SAMPLES,
) -> np.ndarray:
    chunks = []
    for _ in range(n):
        output = adapter.infer(raw_obs, call_idx=call_idx, env_step=0, first_call=first_call)
        chunks.append(np.asarray(output.action_chunk, dtype=np.float32))
    if any(chunk.shape != chunks[0].shape for chunk in chunks):
        raise RuntimeError("N=8 sample action chunks have inconsistent shapes")
    return np.stack(chunks, axis=0)


def _sample_lingbot_chunks_from_snapshot(
    adapter: LingBotAdapter,
    raw_obs: Mapping[str, Any],
    call_idx: int,
    first_call: bool,
    snapshot_label: str,
    n: int = N_SAMPLES,
) -> np.ndarray:
    """Draw N independent LingBot chunks without corrupting its temporal cache.

    A LingBot action forward at frame zero updates the streaming VAE cache even
    though it does not advance the policy chunk index.  Repeating that forward
    for a variance estimate therefore makes the second/third sample invalid.
    Restore the full server snapshot after every sample, but put back the
    *post-sample* RNG state so the next draw remains an independent stochastic
    sample rather than an identical replay.
    """
    chunks = []
    for _ in range(n):
        output = adapter.infer(raw_obs, call_idx=call_idx, env_step=0, first_call=first_call)
        chunks.append(np.asarray(output.action_chunk, dtype=np.float32))
        advanced_rng = adapter.rng_snapshot()
        adapter.temporal_restore(snapshot_label)
        adapter.rng_restore(advanced_rng)
    if any(chunk.shape != chunks[0].shape for chunk in chunks):
        raise RuntimeError("LingBot N=8 sample action chunks have inconsistent shapes")
    return np.stack(chunks, axis=0)


def _gripper_values(samples: np.ndarray, model: str) -> np.ndarray:
    if model == "lingbot_va":
        return samples[:, 6, ...]
    return samples[..., 6]


def _variance(samples: np.ndarray, model: str) -> dict[str, float]:
    return {
        "overall": float(np.mean(np.var(samples.astype(np.float64), axis=0))),
        "gripper": float(np.mean(np.var(_gripper_values(samples, model).astype(np.float64), axis=0))),
    }


def _ratio(drift: float, clean: float, eps: float = 1e-12) -> float | None:
    if clean <= eps and drift <= eps:
        return None
    if clean <= eps:
        return float("inf")
    return float(drift / clean)


def _max_abs_action(lhs: np.ndarray, rhs: np.ndarray) -> float | None:
    if lhs.shape != rhs.shape:
        return None
    return float(np.max(np.abs(lhs.astype(np.float64) - rhs.astype(np.float64))))


def _replay_prefix(
    *,
    env: Any,
    adapter: PolicyAdapter,
    clean_dir: Path,
    steps: pd.DataFrame,
    calls: list[dict[str, Any]],
    injection_step: int,
    task_description: str,
) -> tuple[dict[str, Any], pd.Series]:
    by_step = {int(row.env_step): row for _, row in steps.iterrows()}
    by_call = {int(call["call_idx"]): call for call in calls}
    trigger_step = int(injection_step)
    first_step = int(calls[0]["env_step"])
    raw_obs: dict[str, Any] | None = None
    # Reproduce the settling prefix exactly before policy reset.
    for step in range(1, first_step + 1):
        row = by_step[step]
        raw_obs, _, _, _ = env.step(_vector(row["action_from_previous_step"], 7).tolist())
        raw_obs = dict(raw_obs)
    if _env_step(env) != first_step:
        raise RuntimeError("settling replay did not reach first policy boundary")
    adapter.reset(task_description, {"experiment": "libero_ddi_v1", "condition": "ddi_prefix_replay"})
    mismatch = []
    # Calls starting before the branch are replayed.  For an injection in the
    # middle of a chunk, only its clean executed prefix is stepped, but the
    # client cache is updated with precisely those observed keyframes.
    prior = [call for call in calls if int(call["env_step"]) < trigger_step]
    for ordinal, call in enumerate(prior):
        call_idx = int(call["call_idx"])
        start_step = int(call["env_step"])
        if _env_step(env) != start_step:
            raise RuntimeError(f"prefix replay at env step {_env_step(env)}, expected call boundary {start_step}")
        clean_row = by_step[start_step]
        output = adapter.infer(
            _clean_policy_obs(clean_dir, call, clean_row),
            call_idx=call_idx,
            env_step=start_step,
            first_call=(call_idx == 0),
        )
        mismatch_value = _max_abs_action(np.asarray(output.action_chunk), np.asarray(call["action_chunk"], dtype=np.float32))
        mismatch.append(mismatch_value)
        next_step = min(trigger_step, int(prior[ordinal + 1]["env_step"]) if ordinal + 1 < len(prior) else trigger_step)
        expected_count = next_step - start_step
        if len(output.executable_actions) < expected_count:
            raise RuntimeError(
                f"prefix chunk too short at call {call_idx}: policy={len(output.executable_actions)}, clean interval={expected_count}"
            )
        keyframes = []
        for offset in range(expected_count):
            step = start_step + offset + 1
            row = by_step[step]
            raw_obs, _, _, _ = env.step(_vector(row["action_from_previous_step"], 7).tolist())
            raw_obs = dict(raw_obs)
            if isinstance(adapter, LingBotAdapter):
                frame = _lingbot_keyframe(output, offset, raw_obs)
                if frame is not None:
                    keyframes.append(frame)
        if isinstance(adapter, LingBotAdapter):
            adapter.after_chunk(output, keyframes, done=False)
    if _env_step(env) != trigger_step:
        raise RuntimeError("prefix replay did not reach injection boundary")
    finite_mismatch = [value for value in mismatch if value is not None]
    return {
        "num_replayed_policy_calls": len(prior),
        "max_abs_action_chunk_mismatch": max(finite_mismatch) if finite_mismatch else None,
        "all_action_shapes_matched": all(x is not None for x in mismatch),
    }, by_step[trigger_step]


def _failure_metrics(rows: Sequence[Mapping[str, Any]], target: str, injection_step: int, success: bool) -> dict[str, Any]:
    if not rows:
        return {"failure_type": "no_post_injection_steps"}
    target_column = f"entity__{_safe_entity(target)}__pos"
    eef = np.asarray([_vector(row["eef_pos"], 3) for row in rows])
    obj = np.asarray([_vector(row[target_column], 3) for row in rows])
    xy = np.linalg.norm(eef[:, :2] - obj[:, :2], axis=1)
    xyz = np.linalg.norm(eef - obj, axis=1)
    actions = [row.get("action_from_previous_step") for row in rows]
    close_indices = [idx for idx, action in enumerate(actions) if action is not None and float(_vector(action, 7)[6]) > 0.0]
    signs = [int(float(_vector(action, 7)[6]) > 0.0) for action in actions if action is not None]
    changes = sum(a != b for a, b in zip(signs[:-1], signs[1:]))
    first_close = None
    if close_indices:
        index = close_indices[0]
        first_close = {
            "env_step": int(rows[index]["env_step"]),
            "distance_xyz_m": float(xyz[index]),
            "distance_xy_m": float(xy[index]),
            "delta_t_env_steps": int(rows[index]["env_step"]) - int(injection_step),
        }
    lift = float(np.max(obj[:, 2]) - obj[0, 2])
    if success:
        failure = "success"
    elif not close_indices or float(np.min(xy)) > 0.055:
        failure = "could_not_reach"
    elif first_close is not None and first_close["distance_xyz_m"] > 0.060:
        failure = "reached_but_missed_grasp"
    elif lift > 0.025:
        failure = "dropped_after_lift"
    else:
        failure = "other_timeout"
    return {
        "target_entity": target,
        "min_eef_target_xy_distance_m": float(np.min(xy)),
        "min_eef_target_xyz_distance_m": float(np.min(xyz)),
        "first_close": first_close,
        "gripper_open_close_transition_count": int(changes),
        "object_lift_peak_m": lift,
        "failure_type": failure,
    }


def run_branch(
    *,
    adapter: PolicyAdapter,
    suite: Any,
    task_id: int,
    init_index: int,
    d_cm: float,
    output_root: Path,
    physical_gpu: int,
    overwrite: bool,
    trigger_override: Mapping[str, Any] | None = None,
    direction_override: np.ndarray | None = None,
    direction_metadata: Mapping[str, Any] | None = None,
    condition_name: str = "direct_drift_injection",
) -> dict[str, Any]:
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    spec = TASK_SPECS[task_id]
    clean_dir = _source_dir(adapter.model_name, task_id, init_index)
    clean_meta = json.loads((clean_dir / "meta.json").read_text(encoding="utf-8"))
    steps = pd.read_parquet(clean_dir / "steps.parquet")
    calls = _load_calls(clean_dir)
    if trigger_override is None:
        trigger_row, trigger = _choose_trigger(
            steps,
            task_id,
            require_policy_boundary=isinstance(adapter, LingBotAdapter),
        )
    else:
        trigger = dict(trigger_override)
        matches = steps[steps["env_step"] == int(trigger["env_step"])]
        if len(matches) != 1:
            raise RuntimeError(f"trigger override env_step={trigger['env_step']} is absent or ambiguous")
        trigger_row = matches.iloc[0]
    d_m = float(d_cm) / 100.0
    branch_dir = _branch_dir(output_root, adapter.model_name, task_id, init_index, d_cm)
    result_path = branch_dir / "result.json"
    if result_path.exists() and not overwrite:
        return json.loads(result_path.read_text(encoding="utf-8"))
    task = suite.get_task(task_id)
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=adapter.render_resolution, camera_widths=adapter.render_resolution)
    env.seed(ENVIRONMENT_SEED)
    writer: FullStateRolloutWriter | None = None
    started = time.time()
    result: dict[str, Any] = {
        "schema_version": "libero_ddi_cc_v1_result" if condition_name == "collision_conditioned_direct_drift" else "libero_ddi_v1_result",
        "model": adapter.model_name,
        "task_id": task_id,
        "init_state_index": init_index,
        "d_cm": float(d_cm),
        "source_clean_rollout": str(clean_dir),
        "branch_dir": str(branch_dir),
        "status": "unknown",
        "success": False,
        "trigger": trigger,
    }
    try:
        env.reset()
        env.set_init_state(suite.get_task_init_states(task_id)[init_index])
        prefix, source_trigger_row = _replay_prefix(
            env=env,
            adapter=adapter,
            clean_dir=clean_dir,
            steps=steps,
            calls=calls,
            injection_step=int(trigger["env_step"]),
            task_description=task.language,
        )
        result["prefix_replay"] = prefix
        # Restore the audited clean state exactly before perturbing arm qpos.
        clean_obs = _restore_row(env, source_trigger_row)
        sim = env.env.sim
        clean_eef = _eef_pos(sim)
        baseline_contacts = _contact_set(sim)
        if direction_override is None:
            direction, direction_seed = _direction_for(adapter.model_name, task_id, init_index)
        else:
            direction = np.asarray(direction_override, dtype=np.float64).reshape(7)
            direction /= np.linalg.norm(direction)
            direction_seed = None if direction_metadata is None else direction_metadata.get("rng_seed")
        drift_qpos, drift = _drift_qpos(env, source_trigger_row, clean_meta, direction, d_m)
        result["direction_joint_unit"] = direction
        result["direction_seed"] = direction_seed
        if direction_metadata is not None:
            result["direction_screen"] = dict(direction_metadata)
        result["drift"] = drift
        if drift_qpos is None:
            result.update({"status": "invalid", "invalid_reason": drift["status"], "wall_time_s": time.time() - started})
            _write_json(result_path, result)
            return result
        drift_obs = _restore_row(env, source_trigger_row, drift_qpos)
        contacts = _new_robot_contacts(sim, baseline_contacts)
        result["teleport_collision_contacts"] = contacts
        if contacts:
            result.update({"status": "invalid", "invalid_reason": "teleport_collision", "wall_time_s": time.time() - started})
            _write_json(result_path, result)
            return result
        achieved_eef = _eef_pos(sim)
        result["drift"]["realized_d_after_restore_m"] = float(np.linalg.norm(achieved_eef - clean_eef))
        # The client action queue is deliberately discarded at the branch.  A
        # mid-chunk injection intentionally abandons its remaining actions;
        # deletion makes that forced-replan invariant explicit for audit.
        if isinstance(adapter, VLAJEPAAdapter) and hasattr(adapter.policy, "raw_actions"):
            delattr(adapter.policy, "raw_actions")
        forced_call_idx = int(trigger["active_clean_policy_call_idx"]) + (0 if bool(trigger["at_policy_boundary"]) else 1)
        result["action_queue_reset"] = {
            "at_clean_policy_boundary": bool(trigger["at_policy_boundary"]),
            "forced_policy_call_idx": forced_call_idx,
            "vla_raw_actions_deleted": isinstance(adapter, VLAJEPAAdapter),
        }
        if isinstance(adapter, LingBotAdapter):
            # LingBot's streaming VAE and transformer cache are persistent.
            # The N=8 probe must be non-destructive; otherwise repeated
            # frame-zero forwards append incompatible VAE history and kill the
            # websocket with a 1011 error.  Pair clean/drift draws from the
            # same initial cache/RNG state and restore that state once more
            # before the actual post-injection policy call.
            sample_label = f"ddi_n8_task{task_id}_init{init_index}_d{d_cm:g}"
            adapter.temporal_snapshot(sample_label)
            try:
                clean_samples = _sample_lingbot_chunks_from_snapshot(
                    adapter, clean_obs, forced_call_idx, forced_call_idx == 0, sample_label
                )
                # Pair the drift probe to exactly the same cache/RNG starting
                # point as the clean probe.  The helper itself preserves the
                # advanced RNG between its eight samples.
                adapter.temporal_restore(sample_label)
                drift_samples = _sample_lingbot_chunks_from_snapshot(
                    adapter, drift_obs, forced_call_idx, forced_call_idx == 0, sample_label
                )
                adapter.temporal_restore(sample_label)
            finally:
                adapter.temporal_drop_snapshot(sample_label)
            n8_rng_restored = True
        else:
            snapshot = _snapshot_rng(adapter)
            clean_samples = _sample_chunks(adapter, clean_obs, forced_call_idx, forced_call_idx == 0)
            _restore_rng(adapter, snapshot)
            drift_samples = _sample_chunks(adapter, drift_obs, forced_call_idx, forced_call_idx == 0)
            _restore_rng(adapter, snapshot)
            n8_rng_restored = snapshot is not None
        samples_path = branch_dir / "first_post_injection_call_n8.npz"
        branch_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(samples_path, clean_action_chunks=clean_samples, drift_action_chunks=drift_samples)
        clean_var, drift_var = _variance(clean_samples, adapter.model_name), _variance(drift_samples, adapter.model_name)
        result["n8_first_policy_call"] = {
            "n": N_SAMPLES,
            "path": str(samples_path.relative_to(branch_dir)),
            "clean_variance": clean_var,
            "drift_variance": drift_var,
            "drift_to_clean_ratio": {key: _ratio(drift_var[key], clean_var[key]) for key in clean_var},
            "array_shape": list(clean_samples.shape),
            "rng_snapshot_restored_before_main_call": n8_rng_restored,
        }
        writer = FullStateRolloutWriter(
            branch_dir,
            model=adapter.model_name,
            task_id=task_id,
            init_state_index=init_index,
            task_description=task.language,
            task_spec=spec,
            bddl_file=bddl,
            environment_seed=ENVIRONMENT_SEED,
            physical_gpu=physical_gpu,
            image_width=adapter.render_resolution,
            image_height=adapter.render_resolution,
        )
        # A prior interrupted attempt has no result.json.  Its state/call files
        # must not be appended to when the branch is resumed.
        writer.prepare(overwrite=overwrite or (branch_dir / "steps.parquet").exists())
        injection_step = _env_step(env)
        writer.record_state(env_step=injection_step, env=env, raw_obs=drift_obs, action_from_previous_step=None, reward=None, done=False, phase="ddi_injection")
        main_infer_started = time.perf_counter()
        main_output = adapter.infer(drift_obs, call_idx=forced_call_idx, env_step=injection_step, first_call=forced_call_idx == 0)
        main_infer_latency_ms = (time.perf_counter() - main_infer_started) * 1000.0
        result["main_post_injection_vs_clean_chunk_max_abs"] = (
            _max_abs_action(np.asarray(main_output.action_chunk), np.asarray(calls[forced_call_idx]["action_chunk"], dtype=np.float32))
            if bool(trigger["at_policy_boundary"]) and forced_call_idx < len(calls) else None
        )
        policy_calls = 0
        policy_steps = 0
        success = False
        termination = "timeout"
        output = main_output
        call_idx = forced_call_idx
        while not success and policy_steps < adapter.max_policy_steps:
            infer_start = time.perf_counter()
            if policy_calls > 0:
                output = adapter.infer(drift_obs, call_idx=call_idx, env_step=_env_step(env), first_call=False)
            infer_latency_ms = (time.perf_counter() - infer_start) * 1000.0 if policy_calls > 0 else main_infer_latency_ms
            writer.record_policy_call(
                # The writer is a self-contained post-injection trace and
                # therefore numbers calls from zero.  Keep the original clean
                # call number as explicit provenance rather than creating a
                # non-contiguous JSONL sequence.
                call_idx=policy_calls,
                env_step=_env_step(env),
                env=env,
                raw_obs=drift_obs,
                action_chunk=output.action_chunk,
                infer_latency_ms=infer_latency_ms,
                model_input_metadata=output.model_input_metadata,
                executed_action_indices=output.executed_action_indices,
                latents_video=output.latents_video,
                latent_alignment=output.latent_alignment,
                extra={
                    "ddi_post_injection": policy_calls == 0,
                    "ddi_d_cm": d_cm,
                    "source_clean_policy_call_idx": call_idx,
                },
            )
            keyframes = []
            chunk_complete = True
            for offset, action in enumerate(output.executable_actions):
                if policy_steps >= adapter.max_policy_steps:
                    chunk_complete = False
                    break
                # Some LIBERO wrappers expose a completed episode only on the
                # next `step` call.  Guard before stepping so a successful
                # completion is never turned into a ValueError exception.
                if bool(getattr(env.env, "_episode_terminated", False)):
                    success = bool(env.env._check_success())
                    termination = "success" if success else "terminated"
                    chunk_complete = False
                    break
                drift_obs, reward, done, _ = env.step(_vector(action, 7).astype(np.float32).tolist())
                drift_obs = dict(drift_obs)
                policy_steps += 1
                writer.record_state(
                    env_step=_env_step(env), env=env, raw_obs=drift_obs, action_from_previous_step=action,
                    reward=reward, done=bool(done), phase="ddi_post_injection",
                )
                if isinstance(adapter, LingBotAdapter):
                    frame = _lingbot_keyframe(output, offset, drift_obs)
                    if frame is not None:
                        keyframes.append(frame)
                if done:
                    success, termination, chunk_complete = True, "success", False
                    break
            if chunk_complete:
                adapter.after_chunk(output, keyframes, done=False)
            policy_calls += 1
            call_idx += 1
        metrics = _failure_metrics(writer._rows, spec.manipulated_objects[0], injection_step, success)
        result.update({
            "status": "valid",
            "success": success,
            "termination": termination,
            "post_injection_policy_steps": policy_steps,
            "post_injection_policy_calls": policy_calls,
            "metrics": metrics,
            "wall_time_s": time.time() - started,
        })
        writer.finish(
            success=success,
            termination=termination,
            model_metadata=adapter.model_metadata(),
            policy_protocol=adapter.policy_protocol(),
            extra_meta={"condition": condition_name, "ddi": result},
        )
    except Exception as exc:
        result.update({
            "status": "exception", "termination": f"exception:{type(exc).__name__}", "exception_traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)), "wall_time_s": time.time() - started})
        if writer is not None and writer._rows:
            writer.finish(success=False, termination=result["termination"], model_metadata=adapter.model_metadata(), policy_protocol=adapter.policy_protocol(), extra_meta={"condition": condition_name, "ddi": result})
    finally:
        try:
            env.close()
        except Exception:
            pass
    _write_json(result_path, result)
    print(json.dumps(_jsonable(result), sort_keys=True), flush=True)
    return result


def _seed_pool(model: str, task_id: int) -> list[int]:
    summary = json.loads((CLEAN_ROOT / "clean_summary.json").read_text(encoding="utf-8"))
    combo = summary["combinations"][f"{model}/task{task_id:02d}"]
    pool = [int(value) for value in combo["success_seed_pool"]]
    if len(pool) != 10:
        raise RuntimeError(f"clean pool does not contain 10 seeds: {model}/task{task_id:02d}")
    return pool


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["pi05", "fastwam", "lingbot_va", "vla_jepa"])
    parser.add_argument("--tasks", default="0,2,3,5,9")
    parser.add_argument("--d-cm", default="0,1,2,3,5,8")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--gpu", type=int, required=True, choices=[1, 2, 3])
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--only-init-index", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    if args.gpu == 1 and args.model != "lingbot_va":
        raise ValueError("GPU1 is reserved for LingBot-VA only")
    from libero.libero import benchmark

    suite = benchmark.get_benchmark_dict()["libero_10"]()
    adapter = make_adapter(args.model, args.host, args.port)
    task_ids = [int(value) for value in args.tasks.split(",")]
    d_values = [float(value) for value in args.d_cm.split(",")]
    for task_id in task_ids:
        if task_id not in TASKS:
            raise ValueError(f"unsupported task {task_id}")
        seeds = _seed_pool(args.model, task_id)
        if args.only_init_index is not None:
            if args.only_init_index not in seeds:
                raise ValueError(f"init_state_index {args.only_init_index} is not in the approved clean pool")
            seeds = [args.only_init_index]
        for init_index in seeds:
            for d_cm in d_values:
                run_branch(
                    adapter=adapter, suite=suite, task_id=task_id, init_index=init_index, d_cm=d_cm,
                    output_root=args.output_root, physical_gpu=args.gpu, overwrite=args.overwrite,
                )


if __name__ == "__main__":
    main()

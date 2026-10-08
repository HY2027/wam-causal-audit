from __future__ import annotations
from wam_causal_audit.paths import resolve as _release_path

import csv
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


WORK = Path(__file__).resolve().parent
ROOT = Path(_release_path('@WORKSPACE@/runs/embodiment_phase_gating_decomposition'))
OLD_ROOT = Path(_release_path('@WORKSPACE@/runs/counterfactual_empty_location_closed_loop/rollouts/full'))
for candidate in (
    Path(_release_path('@WORKSPACE@/counterfactual_empty_location_work')),
    Path(_release_path('@WORKSPACE@/synchronization_lag_work')),
    WORK,
):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import protocol as old_protocol  # noqa: E402
import sync_protocol  # noqa: E402


MODELS = ("direct", "joint", "idm", "imagewam")
TASKS = (0, 1)
SMOKE_STATES = (0, 1, 2)
FULL_STATES = tuple(range(10))
CONDITIONS = (
    "fresh",
    "radial_backward_2cm",
    "equal_radius_tangential_2cm",
    "orientation_10deg",
    "orientation_30deg",
    "combined",
)
MANIPULATED_CONDITIONS = CONDITIONS[1:]
SEED_BASE = {"direct": 710000, "joint": 720000, "idm": 730000, "imagewam": 740000}

# Frozen protocol constants. They are written to experiment_config.json before model outcomes.
RADIAL_DISPLACEMENT_M = 0.02
TANGENTIAL_CHORD_M = 0.02
ORIENTATION_10_RAD = math.radians(10.0)
ORIENTATION_30_RAD = math.radians(30.0)
IK_MAX_ITERATIONS = 200
IK_DAMPING = 2e-4
IK_STEP_LIMIT_RAD = 0.03
IK_POSITION_CONVERGENCE_M = 5e-5
IK_ORIENTATION_CONVERGENCE_RAD = 1e-3
POSE_POSITION_TOLERANCE_M = 2e-4
POSE_ORIENTATION_TOLERANCE_RAD = math.radians(0.2)
DISTANCE_CONSTRAINT_TOLERANCE_M = 2e-4
CHORD_CONSTRAINT_TOLERANCE_M = 2e-4
JOINT_LIMIT_MARGIN_RAD = 1e-7
CONTACT_PENETRATION_TOLERANCE_M = 1e-5
CLOSE_COMMAND_THRESHOLD = old_protocol.CLOSE_COMMAND_THRESHOLD
OPEN_APERTURE_M = old_protocol.OPEN_APERTURE_M
CLOSING_SIGN = 1.0


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(jsonable(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(jsonable(value), separators=(",", ":"))
                if isinstance(value, (dict, list, tuple, np.ndarray)) else jsonable(value)
                for key, value in row.items()
            })
    temporary.replace(path)


def sha_array(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    return hashlib.sha256(memoryview(array.view(np.uint8))).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def observation_hash(obs: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for key in sorted(obs):
        value = np.asarray(obs[key])
        if value.dtype.hasobject:
            continue
        contiguous = np.ascontiguousarray(value)
        digest.update(key.encode("utf-8"))
        digest.update(str(contiguous.dtype).encode("utf-8"))
        digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        digest.update(memoryview(contiguous.view(np.uint8)))
    return digest.hexdigest()


def save_observation(path: Path, obs: dict[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    manifest: dict[str, Any] = {}
    for key, item in obs.items():
        value = np.asarray(item)
        if value.dtype.hasobject:
            continue
        arrays[key] = value.copy()
        manifest[key] = {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": sha_array(value)}
    temporary = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)
    return {"path": path, "sha256": observation_hash(arrays), "fields": manifest}


def load_observation(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key].copy() for key in payload.files}


def old_clean_dir(model: str, task: int, state: int) -> Path:
    return OLD_ROOT / model / f"task_{task}" / f"state_{state:02d}" / "clean"


def case_dir(model: str, task: int, state: int) -> Path:
    return ROOT / "cases" / model / f"task_{task}" / f"state_{state:02d}"


def geometry_dir(model: str, task: int, state: int) -> Path:
    return ROOT / "geometry" / model / f"task_{task}" / f"state_{state:02d}"


def source_observation_path(model: str, task: int, state: int) -> Path:
    return geometry_dir(model, task, state) / "source_observation.npz"


def donor_observation_path(model: str, task: int, state: int, condition: str) -> Path:
    if condition == "fresh":
        return source_observation_path(model, task, state)
    return geometry_dir(model, task, state) / f"donor_observation__{condition}.npz"


def geometry_path(model: str, task: int, state: int, condition: str) -> Path:
    return geometry_dir(model, task, state) / f"geometry__{condition}.json"


def selected_clean_call(model: str, task: int, state: int) -> dict[str, Any]:
    clean = old_clean_dir(model, task, state)
    result = json.loads((clean / "result.json").read_text(encoding="utf-8"))
    with (clean / "actions.csv").open(newline="", encoding="utf-8") as stream:
        actions = list(csv.DictReader(stream))
    calls = json.loads((clean / "policy_calls.json").read_text(encoding="utf-8"))
    attempt_step = result.get("first_grasp_attempt_step")
    if attempt_step is None or result.get("real_target_acquired_step") is None:
        raise AssertionError(f"Not a validated clean-success trajectory: {model} task={task} state={state}")
    attempt_step = int(attempt_step)
    attempt_rows = [row for row in actions if int(row["step"]) == attempt_step]
    if len(attempt_rows) != 1 or attempt_rows[0]["closure_attempt"] != "True":
        raise AssertionError("Clean first-grasp-attempt lineage is inconsistent")
    call_id = int(attempt_rows[0]["policy_call"])
    call_rows = [row for row in actions if int(row["policy_call"]) == call_id]
    start_step = min(int(row["step"]) for row in call_rows)
    call = calls[call_id]
    if int(call["policy_call"]) != call_id:
        raise AssertionError("Policy-call table index mismatch")
    return {
        "model": model,
        "task_id": task,
        "source_state_id": state,
        "instruction": result["instruction"],
        "target_object": result["target_object"],
        "selected_policy_call": call_id,
        "selected_call_start_step": start_step,
        "validated_first_grasp_attempt_step": attempt_step,
        "first_grasp_offset_within_chunk": attempt_step - start_step,
        "clean_target_acquired_step": int(result["real_target_acquired_step"]),
        "seed": int(call["seed"]),
        "old_clean_action_sha256": call["action_sha256"],
        "old_source_rep_sha256": call["source_rep_sha256"],
        "old_source_simulator_hash": call["real_sim_hash_before"],
        "old_clean_dir": clean,
        "old_result_sha256": sha_file(clean / "result.json"),
        "old_actions_sha256": sha_file(clean / "actions.csv"),
        "old_policy_calls_sha256": sha_file(clean / "policy_calls.json"),
    }


@dataclass
class ReplayedSource:
    env: Any
    task_object: Any
    obs: dict[str, Any]
    selection: dict[str, Any]
    source_flat: np.ndarray
    ctrl: np.ndarray
    qfrc_applied: np.ndarray
    xfrc_applied: np.ndarray
    replay_audit: dict[str, Any]


def replay_selected_source(model: str, task: int, state: int) -> ReplayedSource:
    selection = selected_clean_call(model, task, state)
    clean = old_clean_dir(model, task, state)
    with (clean / "actions.csv").open(newline="", encoding="utf-8") as stream:
        actions = list(csv.DictReader(stream))
    env, task_object, obs, _ = old_protocol.exact_source_env(task, state)
    for row in actions:
        if int(row["step"]) >= int(selection["selected_call_start_step"]):
            break
        obs, _, _, _ = env.step(json.loads(row["action"]))
        obs = dict(obs)
    sim = old_protocol.fgl.get_sim(env)
    trajectory = np.load(clean / "trajectory.npz", allow_pickle=False)
    index = int(selection["selected_call_start_step"])
    replay_eef = np.asarray(old_protocol.fgl.get_eef_pos(env, obs), dtype=np.float64)
    target_body = old_protocol.body_name(env, old_protocol.TARGETS[task])
    replay_target = np.asarray(old_protocol.fgl.body_pos(sim, target_body), dtype=np.float64)
    replay_aperture = float(old_protocol.aperture(obs))
    eef_error = float(np.max(np.abs(replay_eef - trajectory["eef"][index])))
    target_error = float(np.max(np.abs(replay_target - trajectory["target_xyz"][index])))
    aperture_error = abs(replay_aperture - float(trajectory["gripper_aperture"][index]))
    if max(eef_error, target_error, aperture_error) > 1e-10:
        env.close()
        raise AssertionError(
            f"Clean state replay mismatch {model}/task{task}/state{state}: "
            f"eef={eef_error}, target={target_error}, aperture={aperture_error}"
        )
    source_flat = np.asarray(sim.get_state().flatten(), dtype=np.float64).copy()
    replay_hash = old_protocol.simulator_hash(env, obs)
    replay_audit = {
        **selection,
        "source_state_sha256": sha_array(source_flat),
        "source_observation_sha256": observation_hash(obs),
        "source_simulator_hash_current_renderer": replay_hash,
        "old_source_simulator_hash_match": replay_hash == selection["old_source_simulator_hash"],
        "eef_replay_max_abs_error_m": eef_error,
        "target_replay_max_abs_error_m": target_error,
        "gripper_aperture_replay_abs_error_m": aperture_error,
        "kinematic_replay_exact_to_1e_10": True,
        "source_eef_position_m": replay_eef,
        "source_target_position_m": replay_target,
        "source_gripper_aperture_m": replay_aperture,
    }
    return ReplayedSource(
        env=env,
        task_object=task_object,
        obs=obs,
        selection=selection,
        source_flat=source_flat,
        ctrl=np.asarray(sim.data.ctrl, dtype=np.float64).copy(),
        qfrc_applied=np.asarray(sim.data.qfrc_applied, dtype=np.float64).copy(),
        xfrc_applied=np.asarray(sim.data.xfrc_applied, dtype=np.float64).copy(),
        replay_audit=replay_audit,
    )


def source_sign(task: int, state: int) -> int:
    # Outcome-independent and balanced 10/10 across the 20 source IDs.
    return 1 if (task * 10 + state) % 2 == 0 else -1


def _rotation_about_axis(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("Invalid rotation axis")
    return Rotation.from_rotvec(axis / norm * float(angle)).as_matrix()


def desired_pose(
    condition: str,
    source_position: np.ndarray,
    source_rotation: np.ndarray,
    target_position: np.ndarray,
    sign: int,
) -> tuple[np.ndarray | None, np.ndarray | None, dict[str, Any]]:
    p = np.asarray(source_position, dtype=np.float64)
    rotation = np.asarray(source_rotation, dtype=np.float64)
    target = np.asarray(target_position, dtype=np.float64)
    vector = p - target
    radius = float(np.linalg.norm(vector))
    xy_radius = float(np.linalg.norm(vector[:2]))
    meta: dict[str, Any] = {
        "source_target_distance_m": radius,
        "source_xy_radius_m": xy_radius,
        "rotation_sign": int(sign),
        "rotation_sign_rule": "sign=+1 iff (task_id*10+source_state_id) is even",
    }
    if radius <= 1e-12:
        return None, None, {**meta, "construction_error": "zero_source_target_distance"}
    if condition == "fresh":
        return p.copy(), rotation.copy(), meta
    if condition == "radial_backward_2cm":
        return p + RADIAL_DISPLACEMENT_M * vector / radius, rotation.copy(), meta
    if condition == "equal_radius_tangential_2cm":
        if TANGENTIAL_CHORD_M > 2.0 * xy_radius:
            return None, None, {**meta, "construction_error": "tangential_chord_exceeds_xy_diameter"}
        alpha = 2.0 * math.asin(TANGENTIAL_CHORD_M / (2.0 * xy_radius))
        c, s = math.cos(sign * alpha), math.sin(sign * alpha)
        rotated = vector.copy()
        rotated[:2] = np.array([c * vector[0] - s * vector[1], s * vector[0] + c * vector[1]])
        meta.update({"requested_tangential_angle_rad": sign * alpha, "requested_tangential_angle_deg": math.degrees(sign * alpha)})
        return target + rotated, rotation.copy(), meta
    approach_axis = rotation[:, 2].copy()
    if condition in {"orientation_10deg", "orientation_30deg"}:
        angle = ORIENTATION_10_RAD if condition == "orientation_10deg" else ORIENTATION_30_RAD
        desired_rotation = _rotation_about_axis(approach_axis, sign * angle) @ rotation
        meta.update({"requested_orientation_angle_rad": sign * angle, "requested_orientation_angle_deg": math.degrees(sign * angle)})
        return p.copy(), desired_rotation, meta
    if condition == "combined":
        enlarged = vector * ((radius + RADIAL_DISPLACEMENT_M) / radius)
        enlarged_xy_radius = float(np.linalg.norm(enlarged[:2]))
        if TANGENTIAL_CHORD_M > 2.0 * enlarged_xy_radius:
            return None, None, {**meta, "construction_error": "combined_tangential_chord_exceeds_xy_diameter"}
        alpha = 2.0 * math.asin(TANGENTIAL_CHORD_M / (2.0 * enlarged_xy_radius))
        c, s = math.cos(sign * alpha), math.sin(sign * alpha)
        rotated = enlarged.copy()
        rotated[:2] = np.array([c * enlarged[0] - s * enlarged[1], s * enlarged[0] + c * enlarged[1]])
        desired_rotation = _rotation_about_axis(approach_axis, sign * ORIENTATION_30_RAD) @ rotation
        meta.update({
            "requested_tangential_angle_rad": sign * alpha,
            "requested_tangential_angle_deg": math.degrees(sign * alpha),
            "requested_orientation_angle_rad": sign * ORIENTATION_30_RAD,
            "requested_orientation_angle_deg": math.degrees(sign * ORIENTATION_30_RAD),
            "combined_construction_order": "radial radius +2cm, then same-new-radius XY rotation with 2cm chord, plus approach-axis orientation30",
        })
        return target + rotated, desired_rotation, meta
    raise ValueError(condition)


def site_pose(sim: Any, site_id: int) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray(sim.data.site_xpos[site_id], dtype=np.float64).copy(),
        np.asarray(sim.data.site_xmat[site_id], dtype=np.float64).reshape(3, 3).copy(),
    )


def rotation_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(Rotation.from_matrix(np.asarray(a) @ np.asarray(b).T).as_rotvec()))


def _descendant(model: Any, body: int, ancestor: int) -> bool:
    current = int(body)
    while current > 0:
        if current == ancestor:
            return True
        current = int(model.body_parentid[current])
    return current == ancestor


def collision_audit(sim: Any, partition: sync_protocol.StatePartition) -> dict[str, Any]:
    model = sim.model
    robot = set(partition.robot_body_ids)
    object_roots = set(partition.movable_root_body_ids)
    contacts: list[dict[str, Any]] = []
    for index in range(int(sim.data.ncon)):
        contact = sim.data.contact[index]
        distance = float(contact.dist)
        if distance >= -CONTACT_PENETRATION_TOLERANCE_M:
            continue
        geom1, geom2 = int(contact.geom1), int(contact.geom2)
        body1, body2 = int(model.geom_bodyid[geom1]), int(model.geom_bodyid[geom2])
        robot1, robot2 = body1 in robot, body2 in robot
        if not (robot1 or robot2):
            continue
        name1 = str(model.body_id2name(body1) or "world")
        name2 = str(model.body_id2name(body2) or "world")
        category = None
        if robot1 and robot2 and body1 != body2:
            category = "self_collision"
        elif robot1 != robot2:
            other = body2 if robot1 else body1
            if any(_descendant(model, other, root) for root in object_roots):
                category = "robot_object_collision"
            elif any(token in str(model.body_id2name(other) or "").lower() for token in ("floor", "table", "counter")):
                category = "robot_table_collision"
        if category is not None:
            contacts.append({
                "contact_index": index,
                "category": category,
                "body1": name1,
                "body2": name2,
                "geom1": str(model.geom_id2name(geom1) or geom1),
                "geom2": str(model.geom_id2name(geom2) or geom2),
                "penetration_m": -distance,
            })
    return {
        "contacts": contacts,
        "robot_object_collision_count": sum(row["category"] == "robot_object_collision" for row in contacts),
        "robot_table_collision_count": sum(row["category"] == "robot_table_collision" for row in contacts),
        "self_collision_count": sum(row["category"] == "self_collision" for row in contacts),
        "collision_free": not contacts,
    }


def solve_site_ik(
    sim: Any,
    partition: sync_protocol.StatePartition,
    site_name: str,
    desired_position: np.ndarray,
    desired_rotation: np.ndarray,
) -> dict[str, Any]:
    model = sim.model
    site_id = int(model.site_name2id(site_name))
    arm_qpos = np.asarray(partition.arm_qpos, dtype=int)
    arm_qvel = np.asarray(partition.arm_qvel, dtype=int)
    q_start = np.asarray(sim.data.qpos[arm_qpos], dtype=np.float64).copy()
    joint_ids = [int(model.joint_name2id(name)) for name in partition.provenance["arm_joint_names"]]
    ranges = np.asarray([model.jnt_range[joint] for joint in joint_ids], dtype=np.float64)
    limited = np.asarray([bool(model.jnt_limited[joint]) for joint in joint_ids], dtype=bool)
    iterations = 0
    for iterations in range(1, IK_MAX_ITERATIONS + 1):
        position, rotation = site_pose(sim, site_id)
        position_error = np.asarray(desired_position, dtype=np.float64) - position
        rotation_error = Rotation.from_matrix(np.asarray(desired_rotation) @ rotation.T).as_rotvec()
        if np.linalg.norm(position_error) <= IK_POSITION_CONVERGENCE_M and np.linalg.norm(rotation_error) <= IK_ORIENTATION_CONVERGENCE_RAD:
            break
        jac_position = np.asarray(sim.data.get_site_jacp(site_name), dtype=np.float64).reshape(3, -1)[:, arm_qvel]
        jac_rotation = np.asarray(sim.data.get_site_jacr(site_name), dtype=np.float64).reshape(3, -1)[:, arm_qvel]
        jacobian = np.vstack([jac_position, jac_rotation])
        error = np.concatenate([position_error, rotation_error])
        dq = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + IK_DAMPING * np.eye(6), error)
        dq = np.clip(dq, -IK_STEP_LIMIT_RAD, IK_STEP_LIMIT_RAD)
        candidate = np.asarray(sim.data.qpos[arm_qpos], dtype=np.float64) + dq
        candidate[limited] = np.clip(
            candidate[limited],
            ranges[limited, 0] + JOINT_LIMIT_MARGIN_RAD,
            ranges[limited, 1] - JOINT_LIMIT_MARGIN_RAD,
        )
        sim.data.qpos[arm_qpos] = candidate
        sim.forward()
    final_position, final_rotation = site_pose(sim, site_id)
    position_error_m = float(np.linalg.norm(final_position - np.asarray(desired_position)))
    orientation_error_rad = rotation_distance(np.asarray(desired_rotation), final_rotation)
    q_final = np.asarray(sim.data.qpos[arm_qpos], dtype=np.float64).copy()
    lower_margin = q_final - ranges[:, 0]
    upper_margin = ranges[:, 1] - q_final
    limits_ok = bool(np.all(~limited | ((lower_margin >= -1e-12) & (upper_margin >= -1e-12))))
    converged = bool(position_error_m <= POSE_POSITION_TOLERANCE_M and orientation_error_rad <= POSE_ORIENTATION_TOLERANCE_RAD)
    return {
        "ik_converged": converged,
        "ik_iterations": iterations,
        "ik_position_error_m": position_error_m,
        "ik_orientation_error_rad": orientation_error_rad,
        "arm_joint_start_rad": q_start,
        "arm_joint_final_rad": q_final,
        "arm_joint_delta_rad": q_final - q_start,
        "arm_joint_delta_l2_rad": float(np.linalg.norm(q_final - q_start)),
        "arm_joint_limits_rad": ranges,
        "arm_joint_limited": limited,
        "arm_joint_lower_margin_rad": lower_margin,
        "arm_joint_upper_margin_rad": upper_margin,
        "joint_limits_valid": limits_ok,
        "desired_eef_position_m": np.asarray(desired_position),
        "achieved_eef_position_m": final_position,
        "desired_eef_rotation_matrix": np.asarray(desired_rotation),
        "achieved_eef_rotation_matrix": final_rotation,
    }


def construct_donor(
    donor_env: Any,
    source: ReplayedSource,
    task: int,
    state: int,
    condition: str,
) -> tuple[dict[str, Any] | None, dict[str, Any], np.ndarray | None]:
    sim = old_protocol.fgl.get_sim(donor_env)
    sim.set_state_from_flattened(source.source_flat.copy())
    sim.data.ctrl[:] = source.ctrl
    sim.data.qfrc_applied[:] = source.qfrc_applied
    sim.data.xfrc_applied[:] = source.xfrc_applied
    sim.forward()
    partition = sync_protocol.state_partition(donor_env, task)
    site_id = int(donor_env.env.robots[0].eef_site_id)
    site_name = str(sim.model.site_id2name(site_id))
    source_position, source_rotation = site_pose(sim, site_id)
    target_position = np.asarray(old_protocol.fgl.body_pos(sim, partition.target_body_name), dtype=np.float64).copy()
    target_quaternion = np.asarray(sim.data.get_body_xquat(partition.target_body_name), dtype=np.float64).copy()
    source_qpos = np.asarray(sim.data.qpos, dtype=np.float64).copy()
    source_qvel = np.asarray(sim.data.qvel, dtype=np.float64).copy()
    source_scene_positions = np.stack([
        np.asarray(sim.data.get_body_xpos(name), dtype=np.float64) for name in partition.movable_root_names
    ])
    source_scene_quaternions = np.stack([
        np.asarray(sim.data.get_body_xquat(name), dtype=np.float64) for name in partition.movable_root_names
    ])
    source_aperture = float(old_protocol.aperture(source.obs))
    sign = source_sign(task, state)
    desired_position, desired_rotation, construction = desired_pose(
        condition, source_position, source_rotation, target_position, sign
    )
    base: dict[str, Any] = {
        "model": source.selection["model"],
        "task_id": task,
        "source_state_id": state,
        "condition": condition,
        "source_policy_call": source.selection["selected_policy_call"],
        "source_call_start_step": source.selection["selected_call_start_step"],
        "source_first_grasp_attempt_step": source.selection["validated_first_grasp_attempt_step"],
        "source_first_grasp_offset_within_chunk": source.selection["first_grasp_offset_within_chunk"],
        "source_state_sha256": sha_array(source.source_flat),
        "source_observation_sha256": observation_hash(source.obs),
        "source_eef_site": site_name,
        "source_eef_position_m": source_position,
        "source_eef_rotation_matrix": source_rotation,
        "source_approach_axis_world": source_rotation[:, 2],
        "target_body_name": partition.target_body_name,
        "source_target_position_m": target_position,
        "source_target_quaternion_wxyz": target_quaternion,
        "source_target_distance_m": float(np.linalg.norm(source_position - target_position)),
        "source_gripper_aperture_m": source_aperture,
        "external_scene_must_remain_unchanged": True,
        "real_source_simulator_modified": False,
        **construction,
    }
    if desired_position is None or desired_rotation is None:
        base.update({
            "donor_valid": False,
            "validity_label": "DONOR_INVALID",
            "invalid_reasons": [construction.get("construction_error", "desired_pose_construction_failed")],
            "fallback_intervention_used": False,
            "representation_extracted": False,
        })
        return None, base, None
    if condition == "fresh":
        ik = {
            "ik_converged": True,
            "ik_iterations": 0,
            "ik_position_error_m": 0.0,
            "ik_orientation_error_rad": 0.0,
            "joint_limits_valid": True,
            "desired_eef_position_m": desired_position,
            "achieved_eef_position_m": source_position,
            "desired_eef_rotation_matrix": desired_rotation,
            "achieved_eef_rotation_matrix": source_rotation,
            "arm_joint_delta_rad": np.zeros(len(partition.arm_qpos)),
            "arm_joint_delta_l2_rad": 0.0,
        }
    else:
        ik = solve_site_ik(sim, partition, site_name, desired_position, desired_rotation)
    obs = dict(donor_env.env._get_observations(force_update=True))
    final_position, final_rotation = site_pose(sim, site_id)
    donor_target_position = np.asarray(old_protocol.fgl.body_pos(sim, partition.target_body_name), dtype=np.float64)
    donor_target_quaternion = np.asarray(sim.data.get_body_xquat(partition.target_body_name), dtype=np.float64)
    donor_scene_positions = np.stack([
        np.asarray(sim.data.get_body_xpos(name), dtype=np.float64) for name in partition.movable_root_names
    ])
    donor_scene_quaternions = np.stack([
        np.asarray(sim.data.get_body_xquat(name), dtype=np.float64) for name in partition.movable_root_names
    ])
    final_qpos = np.asarray(sim.data.qpos, dtype=np.float64)
    final_qvel = np.asarray(sim.data.qvel, dtype=np.float64)
    target_distance = float(np.linalg.norm(final_position - donor_target_position))
    displacement = final_position - source_position
    source_vector = source_position - target_position
    final_vector = final_position - donor_target_position
    bearing_source = math.atan2(source_vector[1], source_vector[0])
    bearing_final = math.atan2(final_vector[1], final_vector[0])
    bearing_change = math.atan2(math.sin(bearing_final - bearing_source), math.cos(bearing_final - bearing_source))
    orientation_difference = rotation_distance(final_rotation, source_rotation)
    approach_axis_error = math.acos(float(np.clip(np.dot(final_rotation[:, 2], source_rotation[:, 2]), -1.0, 1.0)))
    scene_qpos_exact = bool(np.array_equal(final_qpos[list(partition.scene_qpos)], source_qpos[list(partition.scene_qpos)]))
    scene_qvel_exact = bool(np.array_equal(final_qvel[list(partition.scene_qvel)], source_qvel[list(partition.scene_qvel)]))
    gripper_qpos_exact = bool(np.array_equal(final_qpos[list(partition.gripper_qpos)], source_qpos[list(partition.gripper_qpos)]))
    gripper_qvel_exact = bool(np.array_equal(final_qvel[list(partition.gripper_qvel)], source_qvel[list(partition.gripper_qvel)]))
    target_pose_exact = bool(
        np.array_equal(donor_target_position, target_position)
        and np.array_equal(donor_target_quaternion, target_quaternion)
    )
    scene_pose_error = max(
        float(np.max(np.abs(donor_scene_positions - source_scene_positions))),
        float(np.max(np.abs(donor_scene_quaternions - source_scene_quaternions))),
    )
    finite_state = bool(np.isfinite(final_qpos).all() and np.isfinite(final_qvel).all())
    finite_obs = True
    for value in obs.values():
        array = np.asarray(value)
        if array.dtype.kind in "biufc" and not np.isfinite(array).all():
            finite_obs = False
            break
    collisions = collision_audit(sim, partition)
    source_radius = float(np.linalg.norm(source_vector))
    distance_change = target_distance - source_radius
    chord = float(np.linalg.norm(displacement))
    radial_direction_cosine = float(np.dot(displacement, source_vector) / ((np.linalg.norm(displacement) + 1e-12) * source_radius))
    geometry_ok = bool(ik["ik_converged"])
    constraint_checks: dict[str, bool] = {}
    if condition == "fresh":
        constraint_checks["fresh_pose_exact"] = bool(np.array_equal(final_position, source_position) and np.array_equal(final_rotation, source_rotation))
    elif condition == "radial_backward_2cm":
        constraint_checks.update({
            "radial_displacement_2cm": abs(chord - RADIAL_DISPLACEMENT_M) <= DISTANCE_CONSTRAINT_TOLERANCE_M,
            "target_distance_plus_2cm": abs(distance_change - RADIAL_DISPLACEMENT_M) <= DISTANCE_CONSTRAINT_TOLERANCE_M,
            "radial_direction_preserved": radial_direction_cosine >= 0.999,
            "orientation_preserved": orientation_difference <= POSE_ORIENTATION_TOLERANCE_RAD,
        })
    elif condition == "equal_radius_tangential_2cm":
        constraint_checks.update({
            "equal_target_radius": abs(target_distance - source_radius) <= DISTANCE_CONSTRAINT_TOLERANCE_M,
            "tangential_chord_2cm": abs(chord - TANGENTIAL_CHORD_M) <= CHORD_CONSTRAINT_TOLERANCE_M,
            "orientation_preserved": orientation_difference <= POSE_ORIENTATION_TOLERANCE_RAD,
        })
    elif condition in {"orientation_10deg", "orientation_30deg"}:
        requested = ORIENTATION_10_RAD if condition == "orientation_10deg" else ORIENTATION_30_RAD
        constraint_checks.update({
            "eef_position_preserved": chord <= POSE_POSITION_TOLERANCE_M,
            "target_distance_preserved": abs(target_distance - source_radius) <= DISTANCE_CONSTRAINT_TOLERANCE_M,
            "orientation_angle_achieved": abs(orientation_difference - requested) <= POSE_ORIENTATION_TOLERANCE_RAD,
            "approach_axis_preserved": approach_axis_error <= POSE_ORIENTATION_TOLERANCE_RAD,
        })
    elif condition == "combined":
        radial_only_position = target_position + source_vector * (
            (source_radius + RADIAL_DISPLACEMENT_M) / source_radius
        )
        combined_tangential_chord = float(np.linalg.norm(final_position - radial_only_position))
        constraint_checks.update({
            "target_distance_plus_2cm": abs(distance_change - RADIAL_DISPLACEMENT_M) <= DISTANCE_CONSTRAINT_TOLERANCE_M,
            "tangential_chord_2cm_after_radial": abs(combined_tangential_chord - TANGENTIAL_CHORD_M) <= CHORD_CONSTRAINT_TOLERANCE_M,
            "orientation_30_achieved": abs(orientation_difference - ORIENTATION_30_RAD) <= POSE_ORIENTATION_TOLERANCE_RAD,
            "approach_axis_preserved": approach_axis_error <= POSE_ORIENTATION_TOLERANCE_RAD,
            "bearing_changed": abs(bearing_change) > 1e-3,
        })
        base["combined_tangential_chord_m"] = combined_tangential_chord
    geometry_ok = geometry_ok and all(constraint_checks.values())
    invariants = {
        "joint_limits_valid": bool(ik["joint_limits_valid"]),
        "finite_simulator_state": finite_state,
        "finite_observation": finite_obs,
        "external_scene_qpos_bit_exact": scene_qpos_exact,
        "external_scene_qvel_bit_exact": scene_qvel_exact,
        "external_scene_pose_max_abs_error": scene_pose_error,
        "external_scene_pose_unchanged": scene_pose_error <= 1e-12,
        "target_pose_bit_exact": target_pose_exact,
        "gripper_qpos_bit_exact": gripper_qpos_exact,
        "gripper_qvel_bit_exact": gripper_qvel_exact,
        # The source policy observation returned by env.step and a later
        # force_update can differ by a few micrometres in the derived width
        # sensor. Physical aperture is fixed by preserving both finger qpos
        # and qvel bit-exactly; the sensor refresh delta remains reported.
        "gripper_aperture_abs_change_m": abs(float(old_protocol.aperture(obs)) - source_aperture),
        "gripper_aperture_unchanged": bool(gripper_qpos_exact and gripper_qvel_exact),
        "gripper_aperture_validity_basis": "finger qpos and qvel bit-exact; force_update sensor delta diagnostic only",
        # FRESH is the validated real clean source itself, so a legitimate
        # pre-existing source contact is diagnostic rather than an invalid
        # synthetic donor. Every manipulated donor still requires no
        # penetrating robot-object/table/self contact.
        "collision_validity_passed": bool(condition == "fresh" or collisions["collision_free"]),
        "collision_validity_rule": "FRESH source contact allowed; manipulated donors must be collision-free",
    }
    required_invariants = [value for value in invariants.values() if isinstance(value, (bool, np.bool_))]
    valid = bool(geometry_ok and all(required_invariants))
    invalid_reasons: list[str] = []
    if not ik["ik_converged"]:
        invalid_reasons.append("ik_nonconvergence")
    invalid_reasons.extend(key for key, value in constraint_checks.items() if not value)
    invalid_reasons.extend(key for key, value in invariants.items() if isinstance(value, bool) and not value)
    donor_flat = np.asarray(sim.get_state().flatten(), dtype=np.float64).copy()
    base.update({
        **ik,
        **collisions,
        **invariants,
        "constraint_checks": constraint_checks,
        "all_required_geometry_constraints_pass": all(constraint_checks.values()),
        "donor_eef_position_m": final_position,
        "donor_eef_rotation_matrix": final_rotation,
        "donor_approach_axis_world": final_rotation[:, 2],
        "donor_target_position_m": donor_target_position,
        "donor_target_quaternion_wxyz": donor_target_quaternion,
        "donor_target_distance_m": target_distance,
        "target_distance_change_m": distance_change,
        "eef_displacement_m": displacement,
        "eef_displacement_norm_m": chord,
        "radial_direction_cosine": radial_direction_cosine,
        "bearing_change_rad": bearing_change,
        "bearing_change_deg": math.degrees(bearing_change),
        "orientation_difference_rad": orientation_difference,
        "orientation_difference_deg": math.degrees(orientation_difference),
        "approach_axis_error_rad": approach_axis_error,
        "donor_state_sha256": sha_array(donor_flat),
        "donor_observation_sha256": observation_hash(obs),
        "donor_valid": valid,
        "validity_label": "VALID" if valid else "DONOR_INVALID",
        "invalid_reasons": invalid_reasons,
        "fallback_intervention_used": False,
        "representation_extracted": False,
    })
    return (obs if valid else None), base, donor_flat


def flatten_geometry(row: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "model", "task_id", "source_state_id", "condition", "source_policy_call",
        "source_call_start_step", "source_first_grasp_attempt_step", "source_first_grasp_offset_within_chunk",
        "source_state_sha256", "source_observation_sha256", "donor_state_sha256", "donor_observation_sha256",
        "validity_label", "donor_valid", "invalid_reasons", "ik_converged", "ik_iterations",
        "ik_position_error_m", "ik_orientation_error_rad", "joint_limits_valid", "finite_simulator_state",
        "finite_observation", "collision_free", "robot_object_collision_count", "robot_table_collision_count",
        "self_collision_count", "external_scene_qpos_bit_exact", "external_scene_qvel_bit_exact",
        "external_scene_pose_max_abs_error", "target_pose_bit_exact", "gripper_qpos_bit_exact",
        "gripper_qvel_bit_exact", "gripper_aperture_abs_change_m", "source_target_distance_m",
        "donor_target_distance_m", "target_distance_change_m", "eef_displacement_norm_m", "radial_direction_cosine",
        "bearing_change_rad", "bearing_change_deg", "orientation_difference_rad", "orientation_difference_deg",
        "approach_axis_error_rad", "source_eef_position_m", "donor_eef_position_m", "source_target_position_m",
        "source_approach_axis_world", "donor_approach_axis_world", "arm_joint_delta_l2_rad",
        "constraint_checks", "all_required_geometry_constraints_pass", "real_source_simulator_modified",
        "fallback_intervention_used", "representation_extracted",
    ]
    return {key: row.get(key) for key in keys}

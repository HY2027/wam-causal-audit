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

WORK = Path(__file__).resolve().parent
PREVIOUS_WORK = Path(_release_path('@WORKSPACE@/counterfactual_empty_location_work'))
PREVIOUS_ROOT = Path(_release_path('@WORKSPACE@/runs/counterfactual_empty_location_closed_loop'))
ROOT = Path(_release_path('@WORKSPACE@/runs/representation_state_synchronization_lag'))
for path in (PREVIOUS_WORK, WORK):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import protocol as previous  # noqa: E402

TASKS = (0, 1)
SMOKE_STATES = tuple(range(5))
FULL_STATES = tuple(range(10))
DELTAS = (1, 2, 4)
LAG_FACTORS = ("robot_lag_b", "scene_lag_b", "both_lag_b", "gripper_lag_b")
ENDPOINTS = ("self", "fresh_b", "frozen_b")
MAX_ASSIGNED_CALLS = 4
H_EVAL = 120
CLOSE_COMMAND_THRESHOLD = previous.CLOSE_COMMAND_THRESHOLD
OPEN_APERTURE_M = previous.OPEN_APERTURE_M
TARGET_LIFT_M = previous.TARGET_LIFT_M
NEAR_OBJECT_M = previous.NEAR_OBJECT_M


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(jsonable(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"no rows: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows([
            {key: json.dumps(jsonable(value), separators=(",", ":")) if isinstance(value, (dict, list, tuple)) else jsonable(value) for key, value in row.items()}
            for row in rows
        ])
    temp.replace(path)


def sha_array(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    return hashlib.sha256(memoryview(array.view(np.uint8))).hexdigest()


def selected_locations() -> dict[tuple[int, int], list[dict[str, Any]]]:
    return previous.selected_locations()


def exact_source_env(task_id: int, state_id: int):
    return previous.exact_source_env(task_id, state_id)


def simulator_hash(env: Any, obs: dict[str, Any]) -> str:
    return previous.simulator_hash(env, obs)


def aperture(obs: dict[str, Any]) -> float:
    return previous.aperture(obs)


def closure_attempt(obs: dict[str, Any], action: np.ndarray) -> bool:
    return previous.closure_attempt(obs, action)


def _joint_widths(joint_type: int) -> tuple[int, int]:
    # MuJoCo: free, ball, slide, hinge.
    return {0: (7, 6), 1: (4, 3), 2: (1, 1), 3: (1, 1)}[int(joint_type)]


def _descendant(model: Any, body: int, ancestor: int) -> bool:
    while body > 0:
        if body == ancestor:
            return True
        body = int(model.body_parentid[body])
    return body == ancestor


@dataclass(frozen=True)
class StatePartition:
    nq: int
    nv: int
    robot_qpos: tuple[int, ...]
    robot_qvel: tuple[int, ...]
    arm_qpos: tuple[int, ...]
    arm_qvel: tuple[int, ...]
    gripper_qpos: tuple[int, ...]
    gripper_qvel: tuple[int, ...]
    scene_qpos: tuple[int, ...]
    scene_qvel: tuple[int, ...]
    robot_body_ids: tuple[int, ...]
    movable_root_body_ids: tuple[int, ...]
    movable_root_names: tuple[str, ...]
    target_body_name: str
    target_qpos_address: int
    provenance: dict[str, Any]


def state_partition(env: Any, task_id: int) -> StatePartition:
    sim = previous.fgl.get_sim(env)
    model = sim.model
    robot = env.env.robots[0]
    arm_names = tuple(robot.robot_joints)
    gripper_names = tuple(robot.gripper.joints)

    def indices(names: tuple[str, ...]) -> tuple[tuple[int, ...], tuple[int, ...]]:
        qpos: list[int] = []
        qvel: list[int] = []
        for name in names:
            joint = int(model.joint_name2id(name))
            qw, vw = _joint_widths(int(model.jnt_type[joint]))
            qa = int(model.jnt_qposadr[joint])
            va = int(model.jnt_dofadr[joint])
            qpos.extend(range(qa, qa + qw))
            qvel.extend(range(va, va + vw))
        return tuple(qpos), tuple(qvel)

    arm_qpos, arm_qvel = indices(arm_names)
    gripper_qpos, gripper_qvel = indices(gripper_names)
    robot_qpos = tuple(sorted(set(arm_qpos + gripper_qpos)))
    robot_qvel = tuple(sorted(set(arm_qvel + gripper_qvel)))
    scene_qpos = tuple(index for index in range(int(model.nq)) if index not in set(robot_qpos))
    scene_qvel = tuple(index for index in range(int(model.nv)) if index not in set(robot_qvel))

    movable_roots = []
    for joint in range(int(model.njnt)):
        if int(model.jnt_type[joint]) == 0:
            body = int(model.jnt_bodyid[joint])
            if body not in movable_roots:
                movable_roots.append(body)
    robot_body_ids = tuple(
        body for body in range(int(model.nbody))
        if str(model.body_id2name(body) or "").startswith(("robot0_", "gripper0_"))
    )
    target_body_name = previous.body_name(env, previous.TARGETS[task_id])
    _, target_addr = previous.target_free_joint(sim, target_body_name)
    movable_names = tuple(str(model.body_id2name(body) or body) for body in movable_roots)
    provenance = {
        "arm_joint_names": arm_names,
        "gripper_joint_names": gripper_names,
        "robot_qpos_indices": robot_qpos,
        "robot_qvel_indices": robot_qvel,
        "arm_qpos_indices": arm_qpos,
        "arm_qvel_indices": arm_qvel,
        "gripper_qpos_indices": gripper_qpos,
        "gripper_qvel_indices": gripper_qvel,
        "scene_qpos_indices": scene_qpos,
        "scene_qvel_indices": scene_qvel,
        "robot_body_names": [str(model.body_id2name(body) or body) for body in robot_body_ids],
        "movable_root_body_names": movable_names,
        "target_body_name": target_body_name,
        "target_qpos_address": target_addr,
    }
    return StatePartition(
        nq=int(model.nq), nv=int(model.nv), robot_qpos=robot_qpos, robot_qvel=robot_qvel,
        arm_qpos=arm_qpos, arm_qvel=arm_qvel, gripper_qpos=gripper_qpos,
        gripper_qvel=gripper_qvel, scene_qpos=scene_qpos, scene_qvel=scene_qvel,
        robot_body_ids=robot_body_ids, movable_root_body_ids=tuple(movable_roots),
        movable_root_names=movable_names, target_body_name=target_body_name,
        target_qpos_address=target_addr, provenance=provenance,
    )


def _quat_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    a = a / (np.linalg.norm(a) + 1e-12); b = b / (np.linalg.norm(b) + 1e-12)
    return float(2.0 * math.acos(float(np.clip(abs(np.dot(a, b)), -1.0, 1.0))))


def capture_snapshot(env: Any, obs: dict[str, Any], partition: StatePartition) -> dict[str, Any]:
    sim = previous.fgl.get_sim(env)
    flat = np.asarray(sim.get_state().flatten(), dtype=np.float64).copy()
    qpos = np.asarray(sim.data.qpos, dtype=np.float64).copy()
    qvel = np.asarray(sim.data.qvel, dtype=np.float64).copy()
    scene_positions = np.stack([np.asarray(sim.data.get_body_xpos(name), dtype=float) for name in partition.movable_root_names])
    scene_quaternions = np.stack([np.asarray(sim.data.get_body_xquat(name), dtype=float) for name in partition.movable_root_names])
    return {
        "flat": flat,
        "qpos": qpos,
        "qvel": qvel,
        "ctrl": np.asarray(sim.data.ctrl, dtype=np.float64).copy(),
        "qfrc_applied": np.asarray(sim.data.qfrc_applied, dtype=np.float64).copy(),
        "xfrc_applied": np.asarray(sim.data.xfrc_applied, dtype=np.float64).copy(),
        "eef": np.asarray(previous.fgl.get_eef_pos(env, obs), dtype=float),
        "eef_quat": np.asarray(obs["robot0_eef_quat"], dtype=float).copy(),
        "arm_qpos": qpos[list(partition.arm_qpos)].copy(),
        "robot_qpos": qpos[list(partition.robot_qpos)].copy(),
        "gripper_qpos": qpos[list(partition.gripper_qpos)].copy(),
        "gripper_aperture_m": aperture(obs),
        "scene_positions": scene_positions,
        "scene_quaternions": scene_quaternions,
        "state_sha256": sha_array(flat),
    }


def support_metrics(current: dict[str, Any], past: dict[str, Any]) -> dict[str, float]:
    scene_delta = np.asarray(current["scene_positions"]) - np.asarray(past["scene_positions"])
    return {
        "D_robot_eef_m": float(np.linalg.norm(np.asarray(current["eef"]) - np.asarray(past["eef"]))),
        "joint_state_distance_l2": float(np.linalg.norm(np.asarray(current["arm_qpos"]) - np.asarray(past["arm_qpos"]))),
        "robot_pose_distance_l2": float(np.linalg.norm(np.asarray(current["robot_qpos"]) - np.asarray(past["robot_qpos"]))),
        "eef_orientation_difference_rad": _quat_distance(current["eef_quat"], past["eef_quat"]),
        "D_gripper_aperture_m": float(abs(float(current["gripper_aperture_m"]) - float(past["gripper_aperture_m"]))),
        "gripper_joint_state_distance_l2": float(np.linalg.norm(np.asarray(current["gripper_qpos"]) - np.asarray(past["gripper_qpos"]))),
        "D_scene_rms_m": float(np.sqrt(np.mean(scene_delta ** 2))),
        "D_scene_max_object_displacement_m": float(np.max(np.linalg.norm(scene_delta, axis=1))),
    }


def _robot_object_collisions(sim: Any, partition: StatePartition) -> list[dict[str, Any]]:
    model = sim.model
    robot_bodies = set(partition.robot_body_ids)
    object_roots = set(partition.movable_root_body_ids)

    def object_root(body: int) -> int | None:
        for root in object_roots:
            if _descendant(model, body, root):
                return root
        return None

    collisions = []
    for index in range(int(sim.data.ncon)):
        contact = sim.data.contact[index]
        b1 = int(model.geom_bodyid[int(contact.geom1)])
        b2 = int(model.geom_bodyid[int(contact.geom2)])
        r1 = b1 in robot_bodies
        r2 = b2 in robot_bodies
        if r1 == r2:
            continue
        other = b2 if r1 else b1
        root = object_root(other)
        if root is None or float(contact.dist) > 0.0:
            continue
        collisions.append({
            "contact_index": index,
            "robot_body": str(model.body_id2name(b1 if r1 else b2) or ""),
            "object_root_body": str(model.body_id2name(root) or ""),
            "other_body": str(model.body_id2name(other) or ""),
            "penetration_m": float(-contact.dist),
        })
    return collisions


def hybrid_donor_observation(
    donor_env: Any,
    partition: StatePartition,
    current: dict[str, Any],
    past: dict[str, Any],
    task_id: int,
    b_xyz: np.ndarray,
    factor: str,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    if factor not in LAG_FACTORS and factor != "fresh_b":
        raise ValueError(factor)
    sim = previous.fgl.get_sim(donor_env)
    flat = np.asarray(current["flat"], dtype=np.float64).copy()
    qpos = flat[1:1 + partition.nq]
    qvel = flat[1 + partition.nq:1 + partition.nq + partition.nv]
    if factor in {"robot_lag_b", "both_lag_b"}:
        qpos[list(partition.robot_qpos)] = np.asarray(past["qpos"])[list(partition.robot_qpos)]
        qvel[list(partition.robot_qvel)] = np.asarray(past["qvel"])[list(partition.robot_qvel)]
    if factor in {"scene_lag_b", "both_lag_b"}:
        qpos[list(partition.scene_qpos)] = np.asarray(past["qpos"])[list(partition.scene_qpos)]
        qvel[list(partition.scene_qvel)] = np.asarray(past["qvel"])[list(partition.scene_qvel)]
    if factor == "gripper_lag_b":
        qpos[list(partition.gripper_qpos)] = np.asarray(past["qpos"])[list(partition.gripper_qpos)]
        qvel[list(partition.gripper_qvel)] = np.asarray(past["qvel"])[list(partition.gripper_qvel)]

    target_addr = partition.target_qpos_address
    target_quat_before = qpos[target_addr + 3:target_addr + 7].copy()
    qpos[target_addr:target_addr + 3] = np.asarray(b_xyz, dtype=np.float64)[:3]
    meta: dict[str, Any] = {
        "factor": factor,
        "current_state_sha256": current["state_sha256"],
        "past_state_sha256": past["state_sha256"],
        "target_B": np.asarray(b_xyz, dtype=float),
        "real_environment_modified": False,
    }
    try:
        sim.set_state_from_flattened(flat)
        sim.data.ctrl[:] = np.asarray(current["ctrl"])
        sim.data.qfrc_applied[:] = np.asarray(current["qfrc_applied"])
        sim.data.xfrc_applied[:] = np.asarray(current["xfrc_applied"])
        sim.forward()
        finite_state = bool(np.isfinite(sim.data.qpos).all() and np.isfinite(sim.data.qvel).all())
        target_exact = bool(np.array_equal(np.asarray(sim.data.qpos[target_addr:target_addr + 3]), np.asarray(b_xyz, dtype=np.float64)[:3]))
        quat_exact = bool(np.array_equal(np.asarray(sim.data.qpos[target_addr + 3:target_addr + 7]), target_quat_before))
        collisions = _robot_object_collisions(sim, partition)
        obs = dict(donor_env.env._get_observations(force_update=True))
        finite_obs = all(not isinstance(value, np.ndarray) or np.isfinite(value).all() for value in obs.values())
        valid = finite_state and finite_obs and target_exact and quat_exact and not collisions
        meta.update({
            "hybrid_state_sha256": sha_array(np.asarray(sim.get_state().flatten(), dtype=np.float64)),
            "finite_simulator_state": finite_state,
            "finite_observation": finite_obs,
            "target_at_fixed_B_bit_exact": target_exact,
            "target_orientation_preserved_bit_exact": quat_exact,
            "robot_object_collisions": collisions,
            "robot_object_collision_count": len(collisions),
            "hybrid_state_valid": valid,
            "validity_label": "VALID" if valid else "HYBRID_STATE_INVALID",
        })
        return (obs if valid else None), meta
    except Exception as exc:
        meta.update({
            "hybrid_state_valid": False,
            "validity_label": "HYBRID_STATE_INVALID",
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "robot_object_collisions": [],
            "robot_object_collision_count": 0,
        })
        return None, meta


def previous_endpoint_dir(model: str, task: int, state: int, condition: str, donor: str | None = None) -> Path:
    base = PREVIOUS_ROOT / "rollouts" / "full" / model / f"task_{task}" / f"state_{state:02d}"
    if condition == "self":
        return base / "self_donor"
    return base / condition / str(donor).lower() / "until_first_grasp_attempt"

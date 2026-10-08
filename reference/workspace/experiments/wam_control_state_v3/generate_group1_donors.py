from __future__ import annotations
from wam_causal_audit.paths import resolve as _release_path

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from group1_config import (
    COLLISION_PENETRATION_TOLERANCE_M,
    DOSE_TOLERANCE_M,
    EEF_OBJECT_RELATIVE_POSITION_TOLERANCE_M,
    FACTOR_SPECS,
    GROUP1_ROOT,
    MODELS,
    PHASES,
    POSE_ORIENTATION_TOLERANCE_RAD,
    POSE_POSITION_TOLERANCE_M,
    observation_sha256,
    jsonable,
    sha256_array,
    sha256_file,
    write_json,
)


for candidate in (
    Path(_release_path('@WORKSPACE@/counterfactual_empty_location_work')),
    Path(_release_path('@WORKSPACE@/synchronization_lag_work')),
    Path(_release_path('@WORKSPACE@/embodiment_phase_gate_work')),
):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import phase_protocol as phase_protocol  # noqa: E402
import protocol as old_protocol  # noqa: E402
import sync_protocol  # noqa: E402


SMOKE_CASES = (
    ("F1_ROBOT_RADIAL_PROGRESS", "PREGRASP", 1.0, "STANDARD"),
    ("F2_OBJECT_ROBOT_GEOMETRY", "PREGRASP", 1.0, "OBJECT_ONLY"),
    ("F2_OBJECT_ROBOT_GEOMETRY", "PREGRASP", 1.0, "OBJECT_AND_GOAL_RIGID_SHIFT"),
    ("F3_OBJECT_GOAL_RADIAL_PROGRESS", "TRANSPORT", 1.0, "STANDARD"),
    ("F3_OBJECT_GOAL_RADIAL_PROGRESS", "PREPLACE", 1.0, "STANDARD"),
    ("F4_TANGENTIAL_DISPLACEMENT", "PREGRASP", 1.0, "STANDARD"),
    ("F5_ORIENTATION_ONLY", "PREGRASP", 10.0, "STANDARD"),
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(jsonable(row), sort_keys=True) + "\n")
    temporary.replace(path)


def save_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def load_phase_state(env: Any, row: dict[str, Any]) -> tuple[Any, dict[str, np.ndarray]]:
    sim = old_protocol.fgl.get_sim(env)
    with np.load(row["recipient_state_path"], allow_pickle=False) as saved:
        arrays = {key: np.asarray(saved[key]).copy() for key in saved.files}
    sim.set_state_from_flattened(arrays["flat"])
    sim.data.ctrl[:] = arrays["ctrl"]
    sim.data.qfrc_applied[:] = arrays["qfrc_applied"]
    sim.data.xfrc_applied[:] = arrays["xfrc_applied"]
    sim.forward()
    if sha256_array(np.asarray(sim.get_state().flatten(), dtype=np.float64)) != row["recipient_state_hash"]:
        raise AssertionError(f"Recipient restore mismatch: {row['base_state_id']}")
    obs = dict(env.env._get_observations(force_update=True))
    return sim, obs


def body_name(env: Any, identity: str) -> str:
    return old_protocol.body_name(env, identity)


def free_joint_address(sim: Any, name: str) -> int:
    body = int(sim.model.body_name2id(name))
    joints = [joint for joint in range(int(sim.model.njnt)) if int(sim.model.jnt_bodyid[joint]) == body]
    if len(joints) != 1 or int(sim.model.jnt_type[joints[0]]) != 0:
        raise AssertionError(f"Expected one free joint for {name}; got {joints}")
    return int(sim.model.jnt_qposadr[joints[0]])


def attachment_flag(env: Any, target_identity: str) -> bool:
    inner = env.env
    return bool(
        inner._check_grasp(
            gripper=inner.robots[0].gripper,
            object_geoms=inner.objects_dict[target_identity].contact_geoms,
        )
    )


def penetrating_contacts(sim: Any) -> set[tuple[str, str]]:
    contacts: set[tuple[str, str]] = set()
    for index in range(int(sim.data.ncon)):
        contact = sim.data.contact[index]
        if float(contact.dist) >= -COLLISION_PENETRATION_TOLERANCE_M:
            continue
        bodies = sorted(
            (
                str(sim.model.body_id2name(int(sim.model.geom_bodyid[int(contact.geom1)])) or "world"),
                str(sim.model.body_id2name(int(sim.model.geom_bodyid[int(contact.geom2)])) or "world"),
            )
        )
        contacts.add((bodies[0], bodies[1]))
    return contacts


def robot_observation_hash(obs: dict[str, Any]) -> str:
    # Camera streams may carry the robot prefix (for example
    # robot0_eye_in_hand_image) but are RGB, not proprioception.
    return observation_sha256(
        {
            key: value
            for key, value in obs.items()
            if key.startswith("robot0_") and "image" not in key.lower()
        }
    )


def finite_observation(obs: dict[str, Any]) -> bool:
    for value in obs.values():
        array = np.asarray(value)
        if array.dtype.kind in "biufc" and not np.isfinite(array).all():
            return False
    return True


def render_valid(obs: dict[str, Any]) -> bool:
    rgb = [np.asarray(value) for key, value in obs.items() if "image" in key.lower()]
    return bool(rgb and all(value.size and np.isfinite(value).all() for value in rgb))


def angle_between_quaternions(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64) / np.linalg.norm(a)
    b = np.asarray(b, dtype=np.float64) / np.linalg.norm(b)
    return float(2.0 * math.acos(float(np.clip(abs(np.dot(a, b)), -1.0, 1.0))))


def make_env(task: int, state: int) -> Any:
    env, _, _, _ = old_protocol.exact_source_env(task, state)
    return env


def construct(
    env: Any,
    row: dict[str, Any],
    factor: str,
    signed_dose: float,
    relation_control: str,
) -> tuple[dict[str, Any] | None, dict[str, Any], dict[str, np.ndarray] | None]:
    sim, source_obs = load_phase_state(env, row)
    partition = sync_protocol.state_partition(env, int(row["task_id"]))
    target_name = body_name(env, row["target_identity"])
    goal_name = body_name(env, row["goal_identity"])
    target_addr = free_joint_address(sim, target_name)
    goal_addr = free_joint_address(sim, goal_name)
    site_id = int(env.env.robots[0].eef_site_id)
    site_name = str(sim.model.site_id2name(site_id))
    eef0, rotation0 = phase_protocol.site_pose(sim, site_id)
    target0 = np.asarray(sim.data.get_body_xpos(target_name), dtype=np.float64).copy()
    target_quat0 = np.asarray(sim.data.get_body_xquat(target_name), dtype=np.float64).copy()
    goal0 = np.asarray(sim.data.get_body_xpos(goal_name), dtype=np.float64).copy()
    goal_quat0 = np.asarray(sim.data.get_body_xquat(goal_name), dtype=np.float64).copy()
    qpos0 = np.asarray(sim.data.qpos, dtype=np.float64).copy()
    qvel0 = np.asarray(sim.data.qvel, dtype=np.float64).copy()
    flat0 = np.asarray(sim.get_state().flatten(), dtype=np.float64).copy()
    contacts0 = penetrating_contacts(sim)
    attached0 = attachment_flag(env, row["target_identity"])
    aperture0 = float(old_protocol.aperture(source_obs))
    source_robot_obs_hash = robot_observation_hash(source_obs)
    dose_native = signed_dose / 100.0 if factor != "F5_ORIENTATION_ONLY" else math.radians(signed_dose)
    requested_vector = np.zeros(3, dtype=np.float64)
    ik: dict[str, Any] = {
        "ik_converged": True,
        "ik_position_error_m": 0.0,
        "ik_orientation_error_rad": 0.0,
        "joint_limits_valid": True,
    }

    if factor in {"F1_ROBOT_RADIAL_PROGRESS", "F2_OBJECT_ROBOT_GEOMETRY"}:
        unit = target0 - eef0
        unit /= np.linalg.norm(unit)
    else:
        unit = goal0 - target0
        unit /= np.linalg.norm(unit)

    if factor == "F1_ROBOT_RADIAL_PROGRESS":
        requested_vector = -dose_native * unit
        ik = phase_protocol.solve_site_ik(sim, partition, site_name, eef0 + requested_vector, rotation0)
    elif factor == "F2_OBJECT_ROBOT_GEOMETRY":
        requested_vector = dose_native * unit
        sim.data.qpos[target_addr : target_addr + 3] += requested_vector
        if relation_control == "OBJECT_AND_GOAL_RIGID_SHIFT":
            sim.data.qpos[goal_addr : goal_addr + 3] += requested_vector
        sim.forward()
    elif factor == "F3_OBJECT_GOAL_RADIAL_PROGRESS":
        requested_vector = -dose_native * unit
        ik = phase_protocol.solve_site_ik(sim, partition, site_name, eef0 + requested_vector, rotation0)
        sim.data.qpos[target_addr : target_addr + 3] += requested_vector
        sim.forward()
    elif factor == "F4_TANGENTIAL_DISPLACEMENT":
        vector = eef0 - target0
        xy_radius = float(np.linalg.norm(vector[:2]))
        chord = abs(dose_native)
        if chord > 2.0 * xy_radius:
            return None, {
                "case_id": f"{row['base_state_id']}__{factor}__{relation_control}__dose{signed_dose:+g}",
                "model": row["model"],
                "task_id": row["task_id"],
                "source_state_id": row["source_state_id"],
                "base_state_id": row["base_state_id"],
                "phase": row["phase"],
                "factor": factor,
                "relation_control": relation_control,
                "signed_dose": signed_dose,
                "dose_unit": FACTOR_SPECS[factor]["dose_unit"],
                "primary_dose": True,
                "status": "DONOR_INVALID",
                "donor_valid": False,
                "invalid_reasons": ["TANGENTIAL_CHORD_EXCEEDS_XY_DIAMETER"],
                "checks": {"tangential_chord_within_xy_diameter": False},
                "source_state_hash": sha256_array(flat0),
                "source_observation_hash": observation_sha256(source_obs),
                "selection_used_motioncos": False,
                "selection_used_action_effect": False,
                "selection_used_success": False,
                "representation_extracted": False,
            }, None
        alpha = math.copysign(2.0 * math.asin(chord / (2.0 * xy_radius)), signed_dose)
        c, s = math.cos(alpha), math.sin(alpha)
        rotated = vector.copy()
        rotated[:2] = (c * vector[0] - s * vector[1], s * vector[0] + c * vector[1])
        desired = target0 + rotated
        requested_vector = desired - eef0
        ik = phase_protocol.solve_site_ik(sim, partition, site_name, desired, rotation0)
    elif factor == "F5_ORIENTATION_ONLY":
        desired_rotation = Rotation.from_rotvec(rotation0[:, 2] * dose_native).as_matrix() @ rotation0
        ik = phase_protocol.solve_site_ik(sim, partition, site_name, eef0, desired_rotation)
    else:
        raise ValueError(factor)

    donor_obs = dict(env.env._get_observations(force_update=True))
    eef1, rotation1 = phase_protocol.site_pose(sim, site_id)
    target1 = np.asarray(sim.data.get_body_xpos(target_name), dtype=np.float64).copy()
    target_quat1 = np.asarray(sim.data.get_body_xquat(target_name), dtype=np.float64).copy()
    goal1 = np.asarray(sim.data.get_body_xpos(goal_name), dtype=np.float64).copy()
    goal_quat1 = np.asarray(sim.data.get_body_xquat(goal_name), dtype=np.float64).copy()
    qpos1 = np.asarray(sim.data.qpos, dtype=np.float64).copy()
    qvel1 = np.asarray(sim.data.qvel, dtype=np.float64).copy()
    contacts1 = penetrating_contacts(sim)
    attached1 = attachment_flag(env, row["target_identity"])
    aperture1 = float(old_protocol.aperture(donor_obs))
    displacement = (target1 - target0) if factor == "F2_OBJECT_ROBOT_GEOMETRY" else (eef1 - eef0)
    orientation_change = phase_protocol.rotation_distance(rotation1, rotation0)
    target_orientation_change = angle_between_quaternions(target_quat1, target_quat0)
    goal_orientation_change = angle_between_quaternions(goal_quat1, goal_quat0)
    new_contacts = sorted(contacts1 - contacts0)
    checks: dict[str, bool] = {
        "ik_converged": bool(ik["ik_converged"]),
        "joint_limits_valid": bool(ik["joint_limits_valid"]),
        "finite_state": bool(np.isfinite(qpos1).all() and np.isfinite(qvel1).all()),
        "finite_observation": finite_observation(donor_obs),
        "donor_rendering_success": render_valid(donor_obs),
        "no_new_collision": not new_contacts,
    }
    dose_error = 0.0
    if factor == "F1_ROBOT_RADIAL_PROGRESS":
        dose_error = float(np.linalg.norm(displacement - requested_vector))
        checks.update({
            "signed_dose_achieved": dose_error <= DOSE_TOLERANCE_M,
            "eef_orientation_unchanged": orientation_change <= POSE_ORIENTATION_TOLERANCE_RAD,
            "target_pose_unchanged": np.array_equal(qpos1[target_addr : target_addr + 7], qpos0[target_addr : target_addr + 7]),
            "goal_pose_unchanged": np.array_equal(qpos1[goal_addr : goal_addr + 7], qpos0[goal_addr : goal_addr + 7]),
            "scene_qvel_unchanged": np.array_equal(qvel1[list(partition.scene_qvel)], qvel0[list(partition.scene_qvel)]),
            "gripper_state_unchanged": np.array_equal(qpos1[list(partition.gripper_qpos)], qpos0[list(partition.gripper_qpos)]),
        })
    elif factor == "F2_OBJECT_ROBOT_GEOMETRY":
        dose_error = float(np.linalg.norm((target1 - target0) - requested_vector))
        expected_goal = goal0 + requested_vector if relation_control == "OBJECT_AND_GOAL_RIGID_SHIFT" else goal0
        checks.update({
            "signed_dose_achieved": dose_error <= DOSE_TOLERANCE_M,
            "robot_qpos_unchanged": np.array_equal(qpos1[list(partition.robot_qpos)], qpos0[list(partition.robot_qpos)]),
            "robot_qvel_unchanged": np.array_equal(qvel1[list(partition.robot_qvel)], qvel0[list(partition.robot_qvel)]),
            "robot_proprio_observation_unchanged": robot_observation_hash(donor_obs) == source_robot_obs_hash,
            "target_orientation_unchanged": target_orientation_change <= POSE_ORIENTATION_TOLERANCE_RAD,
            "goal_position_as_registered": float(np.linalg.norm(goal1 - expected_goal)) <= POSE_POSITION_TOLERANCE_M,
            "goal_orientation_unchanged": goal_orientation_change <= POSE_ORIENTATION_TOLERANCE_RAD,
        })
    elif factor == "F3_OBJECT_GOAL_RADIAL_PROGRESS":
        eef_object_error = float(np.linalg.norm((target1 - eef1) - (target0 - eef0)))
        dose_error = max(
            float(np.linalg.norm((eef1 - eef0) - requested_vector)),
            float(np.linalg.norm((target1 - target0) - requested_vector)),
        )
        checks.update({
            "signed_dose_achieved": dose_error <= DOSE_TOLERANCE_M,
            "attachment_flag_unchanged": attached1 == attached0 and attached1,
            "gripper_aperture_unchanged": abs(aperture1 - aperture0) <= POSE_POSITION_TOLERANCE_M,
            "eef_object_relative_position_unchanged": eef_object_error <= EEF_OBJECT_RELATIVE_POSITION_TOLERANCE_M,
            "object_orientation_unchanged": target_orientation_change <= POSE_ORIENTATION_TOLERANCE_RAD,
            "goal_pose_unchanged": (
                float(np.linalg.norm(goal1 - goal0)) <= POSE_POSITION_TOLERANCE_M
                and goal_orientation_change <= POSE_ORIENTATION_TOLERANCE_RAD
            ),
        })
    elif factor == "F4_TANGENTIAL_DISPLACEMENT":
        source_radius = float(np.linalg.norm(eef0 - target0))
        donor_radius = float(np.linalg.norm(eef1 - target1))
        dose_error = abs(float(np.linalg.norm(displacement)) - abs(dose_native))
        checks.update({
            "euclidean_chord_achieved": dose_error <= DOSE_TOLERANCE_M,
            "radial_distance_unchanged": abs(donor_radius - source_radius) <= DOSE_TOLERANCE_M,
            "eef_orientation_unchanged": orientation_change <= POSE_ORIENTATION_TOLERANCE_RAD,
            "target_pose_unchanged": np.array_equal(qpos1[target_addr : target_addr + 7], qpos0[target_addr : target_addr + 7]),
            "goal_pose_unchanged": np.array_equal(qpos1[goal_addr : goal_addr + 7], qpos0[goal_addr : goal_addr + 7]),
        })
    else:
        dose_error = abs(orientation_change - abs(dose_native))
        checks.update({
            "orientation_dose_achieved": dose_error <= POSE_ORIENTATION_TOLERANCE_RAD,
            "eef_position_unchanged": float(np.linalg.norm(eef1 - eef0)) <= POSE_POSITION_TOLERANCE_M,
            "object_position_unchanged": float(np.linalg.norm(target1 - target0)) <= POSE_POSITION_TOLERANCE_M,
            "goal_pose_unchanged": np.array_equal(qpos1[goal_addr : goal_addr + 7], qpos0[goal_addr : goal_addr + 7]),
        })

    valid = all(checks.values())
    status = "VALID" if valid else "DONOR_INVALID"
    if factor == "F2_OBJECT_ROBOT_GEOMETRY" and relation_control == "OBJECT_AND_GOAL_RIGID_SHIFT" and not valid:
        status = "STRICT_RELATION_CONTROL_UNAVAILABLE"
    donor_flat = np.asarray(sim.get_state().flatten(), dtype=np.float64).copy()
    metadata = {
        "case_id": f"{row['base_state_id']}__{factor}__{relation_control}__dose{signed_dose:+g}",
        "model": row["model"],
        "task_id": row["task_id"],
        "source_state_id": row["source_state_id"],
        "base_state_id": row["base_state_id"],
        "phase": row["phase"],
        "factor": factor,
        "relation_control": relation_control,
        "signed_dose": signed_dose,
        "dose_unit": FACTOR_SPECS[factor]["dose_unit"],
        "primary_dose": signed_dose in FACTOR_SPECS[factor]["primary_signed_doses"],
        "status": status,
        "donor_valid": valid,
        "invalid_reasons": [key for key, value in checks.items() if not value],
        "checks": checks,
        "requested_vector_m": requested_vector,
        "achieved_displacement_m": displacement,
        "dose_absolute_error": dose_error,
        "source_attachment_flag": attached0,
        "donor_attachment_flag": attached1,
        "source_gripper_aperture_m": aperture0,
        "donor_gripper_aperture_m": aperture1,
        "source_eef_position_m": eef0,
        "donor_eef_position_m": eef1,
        "source_object_position_m": target0,
        "donor_object_position_m": target1,
        "source_goal_position_m": goal0,
        "donor_goal_position_m": goal1,
        "eef_orientation_change_rad": orientation_change,
        "object_orientation_change_rad": target_orientation_change,
        "new_penetrating_contacts": new_contacts,
        "source_state_hash": sha256_array(flat0),
        "donor_state_hash": sha256_array(donor_flat),
        "source_observation_hash": observation_sha256(source_obs),
        "donor_observation_hash": observation_sha256(donor_obs),
        "source_rgb_hash": observation_sha256({key: val for key, val in source_obs.items() if "image" in key.lower()}),
        "donor_rgb_hash": observation_sha256({key: val for key, val in donor_obs.items() if "image" in key.lower()}),
        "source_proprio_hash": source_robot_obs_hash,
        "donor_proprio_hash": robot_observation_hash(donor_obs),
        "ik": ik,
        "selection_used_motioncos": False,
        "selection_used_action_effect": False,
        "selection_used_success": False,
        "representation_extracted": False,
    }
    arrays = {
        "flat": donor_flat,
        "ctrl": np.asarray(sim.data.ctrl, dtype=np.float64).copy(),
        "qfrc_applied": np.asarray(sim.data.qfrc_applied, dtype=np.float64).copy(),
        "xfrc_applied": np.asarray(sim.data.xfrc_applied, dtype=np.float64).copy(),
    }
    return (donor_obs if valid else None), metadata, (arrays if valid else None)


def cases_for_row(row: dict[str, Any]) -> list[tuple[str, float, str]]:
    cases: list[tuple[str, float, str]] = []
    for factor, spec in FACTOR_SPECS.items():
        if row["phase"] not in spec["phases"]:
            continue
        doses = list(spec["primary_signed_doses"]) + list(spec.get("stress_signed_doses", []))
        controls = spec.get("relation_controls", ["STANDARD"])
        cases.extend((factor, float(dose), control) for dose in doses for control in controls)
    return cases


def run_case(
    env: Any,
    row: dict[str, Any],
    factor: str,
    dose: float,
    relation_control: str,
    smoke: bool,
) -> dict[str, Any]:
    base = GROUP1_ROOT / ("technical_smoke" if smoke else "donor_bank") / row["model"] / row["base_state_id"]
    case_name = f"{factor}__{relation_control}__dose_{dose:+g}".replace("+", "p").replace("-", "m")
    metadata_path = base / case_name / "counterfactual_qc.json"
    if not smoke and metadata_path.is_file():
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        required_resume_fields = {
            "case_id", "factor", "model", "base_state_id", "signed_dose",
            "donor_valid", "checks", "selection_used_motioncos",
            "selection_used_action_effect", "selection_used_success",
        }
        if "donor_valid" not in existing:
            existing["donor_valid"] = existing.get("status") == "VALID"
            existing.setdefault("checks", {"legacy_explicit_invalid": False})
            write_json(metadata_path, existing)
        if required_resume_fields.issubset(existing):
            existing["counterfactual_qc_path"] = str(metadata_path)
            existing["resume_status"] = "EXISTING"
            return existing
    obs, metadata, arrays = construct(env, row, factor, dose, relation_control)
    state_path = base / case_name / "donor_state.npz"
    observation_path = base / case_name / "donor_observation.npz"
    if arrays is not None and obs is not None:
        save_npz(state_path, **arrays)
        save_npz(observation_path, **obs)
        metadata.update({
            "donor_state_path": str(state_path),
            "donor_state_file_sha256": sha256_file(state_path),
            "donor_observation_path": str(observation_path),
            "donor_observation_file_sha256": sha256_file(observation_path),
        })
    metadata["counterfactual_qc_path"] = str(metadata_path)
    write_json(metadata_path, metadata)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, default="direct")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--tasks", default="0,1,2,3,4")
    args = parser.parse_args()
    registry = read_jsonl(GROUP1_ROOT / "phase_registry.jsonl")
    tasks = {int(value) for value in args.tasks.split(",") if value}
    rows = [
        row for row in registry
        if row["model"] == args.model and row["task_id"] in tasks and row.get("available") is True
    ]
    manifest: list[dict[str, Any]] = []
    if args.smoke:
        row_map = {(row["task_id"], row["source_state_id"], row["phase"]): row for row in rows}
        for factor, phase, dose, control in SMOKE_CASES:
            row = row_map[(0, 0, phase)]
            env = make_env(0, 0)
            try:
                result = run_case(env, row, factor, dose, control, True)
                manifest.append(result)
                print(json.dumps({key: result[key] for key in ("case_id", "status", "invalid_reasons")}), flush=True)
            finally:
                env.close()
        # A technical smoke validates construction and explicit safety
        # rejection.  Some near-object tangential/orientation poses are known
        # a priori to intersect the object; they must be labelled invalid, not
        # admitted or used to change the frozen dose/state rules.
        smoke_passes = []
        for result in manifest:
            failed = set(result["invalid_reasons"])
            smoke_passes.append(
                result["donor_valid"]
                or (
                    failed == {"no_new_collision"}
                    and result["status"] == "DONOR_INVALID"
                    and all(
                        value
                        for key, value in result["checks"].items()
                        if key != "no_new_collision"
                    )
                )
            )
        passed = all(smoke_passes)
        write_json(
            GROUP1_ROOT / "technical_smoke" / "counterfactual_smoke_summary.json",
            {
                "status": "PASS" if passed else "FAIL",
                "smoke_case_passes": smoke_passes,
                "safety_rejections_are_not_admitted": True,
                "thresholds_or_doses_changed_from_outcomes": False,
                "cases": manifest,
            },
        )
        if not passed:
            raise SystemExit(2)
    else:
        for task in sorted(tasks):
            task_rows = [row for row in rows if row["task_id"] == task]
            if not task_rows:
                continue
            env = make_env(task, 0)
            try:
                for row in task_rows:
                    for factor, dose, control in cases_for_row(row):
                        result = run_case(env, row, factor, dose, control, False)
                        manifest.append(result)
                        if len(manifest) % 25 == 0:
                            print(
                                json.dumps(
                                    {
                                        "model": args.model,
                                        "completed": len(manifest),
                                        "valid": sum(bool(item.get("donor_valid", False)) for item in manifest),
                                        "latest_case_id": result["case_id"],
                                    }
                                ),
                                flush=True,
                            )
            finally:
                env.close()
        write_json(
            GROUP1_ROOT / "donor_bank" / args.model / "manifest.json",
            {"model": args.model, "cases": len(manifest), "valid": sum(bool(row.get("donor_valid", False)) for row in manifest)},
        )
        write_jsonl(GROUP1_ROOT / "donor_bank" / args.model / "counterfactual_qc.jsonl", manifest)


if __name__ == "__main__":
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    main()

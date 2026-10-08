from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from task_specs import TaskSpec


SCHEMA_VERSION = "libero_full_state_clean_v1"
CAMERA_NAMES = ("agentview", "robot0_eye_in_hand")


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


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(_jsonable(payload), ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _safe(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", str(value)).strip("_")


def _sim(env: Any) -> Any:
    inner = getattr(env, "env", env)
    return getattr(inner, "sim", getattr(env, "sim", None))


def _id2name(model: Any, kind: str, idx: int) -> str | None:
    candidates = {
        "joint": ("joint_id2name", "jnt_id2name"),
        "body": ("body_id2name",),
        "camera": ("camera_id2name", "cam_id2name"),
        "actuator": ("actuator_id2name",),
    }[kind]
    for attr in candidates:
        fn = getattr(model, attr, None)
        if callable(fn):
            try:
                return fn(idx)
            except Exception:
                pass
    return None


def _name2id(model: Any, kind: str, name: str) -> int:
    candidates = {
        "body": ("body_name2id",),
        "camera": ("camera_name2id", "cam_name2id"),
        "joint": ("joint_name2id", "jnt_name2id"),
    }[kind]
    for attr in candidates:
        fn = getattr(model, attr, None)
        if callable(fn):
            try:
                return int(fn(name))
            except Exception:
                pass
    count = int(getattr(model, {"body": "nbody", "camera": "ncam", "joint": "njnt"}[kind]))
    for idx in range(count):
        if _id2name(model, kind, idx) == name:
            return idx
    raise KeyError(f"unknown MuJoCo {kind}: {name}")


def _joint_slices(model: Any, joint_id: int) -> tuple[slice, slice]:
    q0 = int(model.jnt_qposadr[joint_id])
    q1 = int(model.nq) if joint_id + 1 == int(model.njnt) else int(model.jnt_qposadr[joint_id + 1])
    v0 = int(model.jnt_dofadr[joint_id])
    v1 = int(model.nv) if joint_id + 1 == int(model.njnt) else int(model.jnt_dofadr[joint_id + 1])
    return slice(q0, q1), slice(v0, v1)


def _entity_for_joint(joint_name: str, body_name: str, spec: TaskSpec) -> str | None:
    haystack = f"{joint_name} {body_name}".lower()
    for entity in spec.relevant_entities:
        if entity.lower() in haystack:
            return entity
    return None


def build_dof_map(sim: Any, spec: TaskSpec) -> dict[str, Any]:
    model = sim.model
    joints: dict[str, Any] = {}
    for joint_id in range(int(model.njnt)):
        joint_name = _id2name(model, "joint", joint_id) or f"joint_{joint_id}"
        body_id = int(model.jnt_bodyid[joint_id])
        body_name = _id2name(model, "body", body_id) or f"body_{body_id}"
        qpos_slice, qvel_slice = _joint_slices(model, joint_id)
        entity = _entity_for_joint(joint_name, body_name, spec)
        lowered = joint_name.lower()
        if joint_name in spec.articulation_joints:
            category = "task_articulation"
        elif lowered.startswith("robot0_joint"):
            category = "robot_arm"
        elif "gripper" in lowered or "finger_joint" in lowered:
            category = "robot_gripper"
        elif entity in spec.manipulated_objects:
            category = "task_manipulated_object"
        elif entity is not None:
            category = "task_destination_or_fixture"
        else:
            category = "distractor_or_other"
        joints[joint_name] = {
            "joint_id": joint_id,
            "body_id": body_id,
            "body_name": body_name,
            "qpos_slice": [qpos_slice.start, qpos_slice.stop],
            "qvel_slice": [qvel_slice.start, qvel_slice.stop],
            "qpos_width": qpos_slice.stop - qpos_slice.start,
            "qvel_width": qvel_slice.stop - qvel_slice.start,
            "joint_type": int(model.jnt_type[joint_id]),
            "category": category,
            "task_entity": entity,
        }
    actuators = {}
    for actuator_id in range(int(getattr(model, "nu", 0))):
        name = _id2name(model, "actuator", actuator_id) or f"actuator_{actuator_id}"
        actuators[name] = {
            "actuator_id": actuator_id,
            "transmission_joint_id": int(model.actuator_trnid[actuator_id][0]),
            "ctrlrange": np.asarray(model.actuator_ctrlrange[actuator_id], dtype=np.float64).tolist(),
        }
    return {
        "nq": int(model.nq),
        "nv": int(model.nv),
        "na": int(getattr(model, "na", 0)),
        "nu": int(getattr(model, "nu", 0)),
        "joints": joints,
        "actuators": actuators,
        "categories": {
            category: [name for name, row in joints.items() if row["category"] == category]
            for category in sorted({row["category"] for row in joints.values()})
        },
    }


def _intrinsics(width: int, height: int, fovy_deg: float) -> dict[str, Any]:
    fy = 0.5 * float(height) / math.tan(0.5 * math.radians(float(fovy_deg)))
    fx = fy
    cx = (float(width) - 1.0) / 2.0
    cy = (float(height) - 1.0) / 2.0
    return {
        "width": width,
        "height": height,
        "fovy_deg": float(fovy_deg),
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "K": [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        "convention": "pinhole_from_mujoco_vertical_fovy_centered_pixels",
    }


def camera_snapshot(sim: Any, width: int, height: int) -> dict[str, Any]:
    result = {}
    for name in CAMERA_NAMES:
        camera_id = _name2id(sim.model, "camera", name)
        world_from_camera = np.eye(4, dtype=np.float64)
        world_from_camera[:3, :3] = np.asarray(sim.data.cam_xmat[camera_id], dtype=np.float64).reshape(3, 3)
        world_from_camera[:3, 3] = np.asarray(sim.data.cam_xpos[camera_id], dtype=np.float64).reshape(3)
        result[name] = {
            "camera_id": camera_id,
            "body_id": int(sim.model.cam_bodyid[camera_id]),
            "intrinsics": _intrinsics(width, height, float(sim.model.cam_fovy[camera_id])),
            "model_pos": np.asarray(sim.model.cam_pos[camera_id], dtype=np.float64),
            "model_quat_wxyz": np.asarray(sim.model.cam_quat[camera_id], dtype=np.float64),
            "world_from_camera": world_from_camera,
            "camera_from_world": np.linalg.inv(world_from_camera),
        }
    return result


def _hash_arrays(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(str(value.dtype).encode())
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _body_pose(sim: Any, entity: str) -> tuple[list[float], list[float], str | None]:
    candidates = [f"{entity}_main", entity]
    matches = [
        _id2name(sim.model, "body", body_id)
        for body_id in range(int(sim.model.nbody))
        if str(_id2name(sim.model, "body", body_id) or "").startswith(entity)
    ]
    for body_name in candidates + [x for x in matches if x]:
        try:
            body_id = _name2id(sim.model, "body", body_name)
            return (
                np.asarray(sim.data.body_xpos[body_id], dtype=np.float64).reshape(3).tolist(),
                np.asarray(sim.data.body_xquat[body_id], dtype=np.float64).reshape(4).tolist(),
                body_name,
            )
        except Exception:
            continue
    return [], [], None


class GripperHysteresis:
    def __init__(self) -> None:
        self.state = "unknown"

    def update(self, gripper_qpos: Sequence[float]) -> tuple[str, float]:
        qpos = np.asarray(gripper_qpos, dtype=np.float64).reshape(-1)
        aperture = float(abs(qpos[0] - qpos[1])) if qpos.size >= 2 else float("nan")
        if aperture >= 0.06:
            self.state = "open"
        elif aperture <= 0.04:
            self.state = "closed"
        return self.state, aperture


class FullStateRolloutWriter:
    def __init__(
        self,
        rollout_dir: Path,
        *,
        model: str,
        task_id: int,
        init_state_index: int,
        task_description: str,
        task_spec: TaskSpec,
        bddl_file: Path,
        environment_seed: int,
        physical_gpu: int,
        image_width: int = 256,
        image_height: int = 256,
        condition: str = "clean",
    ) -> None:
        self.rollout_dir = Path(rollout_dir)
        self.model = str(model)
        self.task_id = int(task_id)
        self.init_state_index = int(init_state_index)
        self.task_description = str(task_description)
        self.task_spec = task_spec
        self.bddl_file = Path(bddl_file)
        self.environment_seed = int(environment_seed)
        self.physical_gpu = int(physical_gpu)
        self.image_width = int(image_width)
        self.image_height = int(image_height)
        self.condition = str(condition)
        self.steps_path = self.rollout_dir / "steps.parquet"
        self.calls_path = self.rollout_dir / "policy_calls.jsonl"
        self.meta_path = self.rollout_dir / "meta.json"
        self.frames_dir = self.rollout_dir / "observation_frames"
        self.latents_dir = self.rollout_dir / "latents"
        self._rows: list[dict[str, Any]] = []
        self._row_for_env_step: dict[int, int] = {}
        self._calls = 0
        self._dof_map: dict[str, Any] | None = None
        self._camera_initial: dict[str, Any] | None = None
        self._entity_joint_names: dict[str, list[str]] = {}
        self._gripper = GripperHysteresis()
        self.created_at_unix = time.time()

    def prepare(self, overwrite: bool = False) -> None:
        if self.rollout_dir.exists() and not overwrite and self.meta_path.exists():
            raise FileExistsError(f"completed rollout already exists: {self.rollout_dir}")
        self.rollout_dir.mkdir(parents=True, exist_ok=True)
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.latents_dir.mkdir(parents=True, exist_ok=True)
        if overwrite:
            for path in (self.steps_path, self.calls_path, self.meta_path):
                if path.exists():
                    path.unlink()

    def _initialize_maps(self, sim: Any) -> None:
        if self._dof_map is not None:
            return
        self._dof_map = build_dof_map(sim, self.task_spec)
        for entity in self.task_spec.relevant_entities:
            self._entity_joint_names[entity] = [
                name
                for name, row in self._dof_map["joints"].items()
                if row["task_entity"] == entity
            ]
        self._camera_initial = camera_snapshot(sim, self.image_width, self.image_height)

    def _joint_values(self, sim: Any, names: Sequence[str], kind: str) -> list[float]:
        assert self._dof_map is not None
        source = sim.data.qpos if kind == "qpos" else sim.data.qvel
        values: list[float] = []
        key = f"{kind}_slice"
        for name in names:
            start, stop = self._dof_map["joints"][name][key]
            values.extend(np.asarray(source[start:stop], dtype=np.float64).tolist())
        return values

    def record_state(
        self,
        *,
        env_step: int,
        env: Any,
        raw_obs: Mapping[str, Any],
        action_from_previous_step: Sequence[float] | None,
        reward: float | None,
        done: bool,
        phase: str = "unsegmented",
    ) -> None:
        env_step = int(env_step)
        if env_step in self._row_for_env_step:
            raise ValueError(f"duplicate env_step {env_step} in {self.rollout_dir}")
        sim = _sim(env)
        # Synchronize all derived kinematics with the qpos/qvel snapshot.  The
        # observation returned by robosuite can otherwise be one controller
        # update behind the final MuJoCo state of env.step().
        sim.forward()
        self._initialize_maps(sim)
        assert self._dof_map is not None
        qpos = np.asarray(sim.data.qpos, dtype=np.float64).copy()
        qvel = np.asarray(sim.data.qvel, dtype=np.float64).copy()
        act = np.asarray(getattr(sim.data, "act", []), dtype=np.float64).copy()
        ctrl = np.asarray(getattr(sim.data, "ctrl", []), dtype=np.float64).copy()
        mocap_pos = np.asarray(getattr(sim.data, "mocap_pos", []), dtype=np.float64).copy()
        mocap_quat = np.asarray(getattr(sim.data, "mocap_quat", []), dtype=np.float64).copy()
        userdata = np.asarray(getattr(sim.data, "userdata", []), dtype=np.float64).copy()

        obs_eef_pos = np.asarray(raw_obs["robot0_eef_pos"], dtype=np.float64).reshape(3)
        obs_eef_quat_xyzw = np.asarray(raw_obs["robot0_eef_quat"], dtype=np.float64).reshape(4)
        eef_body = "gripper0_eef"
        try:
            eef_pos = np.asarray(sim.data.get_body_xpos(eef_body), dtype=np.float64).reshape(3)
            eef_quat_wxyz = np.asarray(sim.data.get_body_xquat(eef_body), dtype=np.float64).reshape(4)
            eef_vel = np.concatenate([
                np.asarray(sim.data.get_body_xvelp(eef_body), dtype=np.float64).reshape(3),
                np.asarray(sim.data.get_body_xvelr(eef_body), dtype=np.float64).reshape(3),
            ])
        except Exception:
            eef_body = "robot0_eef"
            eef_pos = np.asarray(sim.data.get_body_xpos(eef_body), dtype=np.float64).reshape(3)
            eef_quat_wxyz = np.asarray(sim.data.get_body_xquat(eef_body), dtype=np.float64).reshape(4)
            eef_vel = np.concatenate([
                np.asarray(sim.data.get_body_xvelp(eef_body), dtype=np.float64).reshape(3),
                np.asarray(sim.data.get_body_xvelr(eef_body), dtype=np.float64).reshape(3),
            ])
        gripper_qpos = np.asarray(raw_obs["robot0_gripper_qpos"], dtype=np.float64).reshape(-1)[:2]
        gripper_qvel = np.asarray(raw_obs.get("robot0_gripper_qvel", []), dtype=np.float64).reshape(-1)[:2]
        gripper_state, gripper_aperture = self._gripper.update(gripper_qpos)

        object_qpos: dict[str, list[float]] = {}
        object_qvel: dict[str, list[float]] = {}
        object_pose: dict[str, dict[str, Any]] = {}
        dynamic_entity_columns: dict[str, Any] = {}
        for entity, joint_names in self._entity_joint_names.items():
            entity_qpos = self._joint_values(sim, joint_names, "qpos")
            entity_qvel = self._joint_values(sim, joint_names, "qvel")
            pos, quat_wxyz, body_name = _body_pose(sim, entity)
            object_qpos[entity] = entity_qpos
            object_qvel[entity] = entity_qvel
            object_pose[entity] = {"body_name": body_name, "pos": pos, "quat_wxyz": quat_wxyz}
            prefix = f"entity__{_safe(entity)}"
            dynamic_entity_columns[f"{prefix}__qpos"] = entity_qpos
            dynamic_entity_columns[f"{prefix}__qvel"] = entity_qvel
            dynamic_entity_columns[f"{prefix}__pos"] = pos
            dynamic_entity_columns[f"{prefix}__quat_wxyz"] = quat_wxyz

        articulation_qpos = {
            name: self._joint_values(sim, [name], "qpos")
            for name in self.task_spec.articulation_joints
        }
        articulation_qvel = {
            name: self._joint_values(sim, [name], "qvel")
            for name in self.task_spec.articulation_joints
        }
        cameras = camera_snapshot(sim, self.image_width, self.image_height)
        row: dict[str, Any] = {
            "env_step": env_step,
            "sim_time": float(sim.data.time),
            "sim_qpos": qpos.tolist(),
            "sim_qvel": qvel.tolist(),
            "sim_act": act.tolist(),
            "sim_ctrl": ctrl.tolist(),
            "sim_mocap_pos": mocap_pos.reshape(-1).tolist(),
            "sim_mocap_quat": mocap_quat.reshape(-1).tolist(),
            "sim_userdata": userdata.tolist(),
            "sim_state_sha256": _hash_arrays(qpos, qvel, act, ctrl, mocap_pos, mocap_quat, userdata),
            "eef_pose": np.concatenate([eef_pos, eef_quat_wxyz]).tolist(),
            "eef_pos": eef_pos.tolist(),
            "eef_quat_wxyz": eef_quat_wxyz.tolist(),
            "eef_obs_pos": obs_eef_pos.tolist(),
            "eef_obs_quat_xyzw": obs_eef_quat_xyzw.tolist(),
            "eef_vel": eef_vel.tolist(),
            "eef_linear_vel": eef_vel[:3].tolist(),
            "eef_angular_vel": eef_vel[3:].tolist(),
            "eef_body_name": eef_body,
            "gripper_qpos": gripper_qpos.tolist(),
            "gripper_qvel": gripper_qvel.tolist(),
            "gripper_aperture": gripper_aperture,
            "gripper_state": gripper_state,
            "object_qpos_json": json.dumps(object_qpos, sort_keys=True),
            "object_qvel_json": json.dumps(object_qvel, sort_keys=True),
            "object_pose_json": json.dumps(object_pose, sort_keys=True),
            "articulation_qpos_json": json.dumps(articulation_qpos, sort_keys=True),
            "articulation_qvel_json": json.dumps(articulation_qvel, sort_keys=True),
            "agentview_world_from_camera": np.asarray(cameras["agentview"]["world_from_camera"]).reshape(-1).tolist(),
            "eye_in_hand_world_from_camera": np.asarray(cameras["robot0_eye_in_hand"]["world_from_camera"]).reshape(-1).tolist(),
            "action_from_previous_step": None if action_from_previous_step is None else np.asarray(action_from_previous_step, dtype=np.float64).reshape(-1)[:7].tolist(),
            "reward": None if reward is None else float(reward),
            "done": bool(done),
            "policy_call_idx": None,
            "infer_latency_ms": None,
            "phase": str(phase),
            **dynamic_entity_columns,
        }
        for name, value in articulation_qpos.items():
            row[f"articulation__{_safe(name)}__qpos"] = value
            row[f"articulation__{_safe(name)}__qvel"] = articulation_qvel[name]
        self._row_for_env_step[env_step] = len(self._rows)
        self._rows.append(row)

    def record_policy_call(
        self,
        *,
        call_idx: int,
        env_step: int,
        env: Any,
        raw_obs: Mapping[str, Any],
        action_chunk: np.ndarray,
        infer_latency_ms: float,
        model_input_metadata: Mapping[str, Any],
        executed_action_indices: Sequence[int],
        latents_video: np.ndarray | None = None,
        latent_alignment: Mapping[str, Any] | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        call_idx = int(call_idx)
        env_step = int(env_step)
        if call_idx != self._calls:
            raise ValueError(f"policy call index is not contiguous: expected {self._calls}, got {call_idx}")
        if env_step not in self._row_for_env_step:
            raise ValueError(f"policy call at env_step {env_step} has no state row")
        state_row = self._rows[self._row_for_env_step[env_step]]
        state_row["policy_call_idx"] = call_idx
        state_row["infer_latency_ms"] = float(infer_latency_ms)

        frames_path = self.frames_dir / f"call_{call_idx:04d}.npz"
        agentview = np.ascontiguousarray(np.asarray(raw_obs["agentview_image"], dtype=np.uint8))
        wrist = np.ascontiguousarray(np.asarray(raw_obs["robot0_eye_in_hand_image"], dtype=np.uint8))
        np.savez_compressed(frames_path, agentview_image=agentview, robot0_eye_in_hand_image=wrist)
        frame_hashes = {
            "agentview_image_sha256": _hash_arrays(agentview),
            "robot0_eye_in_hand_image_sha256": _hash_arrays(wrist),
        }

        latent_ref = None
        if latents_video is not None:
            latent_path = self.latents_dir / f"call_{call_idx:04d}_latents_video.npz"
            latent_array = np.asarray(latents_video)
            np.savez_compressed(latent_path, latents_video=latent_array)
            latent_ref = {
                "path": str(latent_path.relative_to(self.rollout_dir)),
                "key": "latents_video",
                "shape": list(latent_array.shape),
                "dtype": str(latent_array.dtype),
                "sha256": _hash_arrays(latent_array),
                "alignment": dict(latent_alignment or {}),
            }

        cameras = camera_snapshot(_sim(env), self.image_width, self.image_height)
        row = {
            "call_idx": call_idx,
            "env_step": env_step,
            "action_chunk": np.asarray(action_chunk, dtype=np.float32).tolist(),
            "action_chunk_shape": list(np.asarray(action_chunk).shape),
            "executed_action_indices": [int(x) for x in executed_action_indices],
            "obs_frame_ref": {
                "path": str(frames_path.relative_to(self.rollout_dir)),
                "keys": ["agentview_image", "robot0_eye_in_hand_image"],
                "raw_shape": {"agentview": list(agentview.shape), "eye_in_hand": list(wrist.shape)},
                **frame_hashes,
                "model_input_metadata": dict(model_input_metadata),
            },
            "latents_video": latent_ref,
            "infer_latency_ms": float(infer_latency_ms),
            "sim_state_sha256": state_row["sim_state_sha256"],
            "camera_world_from_camera": {
                name: np.asarray(payload["world_from_camera"]).tolist()
                for name, payload in cameras.items()
            },
        }
        if extra:
            row.update(dict(extra))
        _append_jsonl(self.calls_path, row)
        self._calls += 1

    def finish(
        self,
        *,
        success: bool,
        termination: str,
        model_metadata: Mapping[str, Any],
        policy_protocol: Mapping[str, Any],
        extra_meta: Mapping[str, Any] | None = None,
    ) -> None:
        if not self._rows:
            raise RuntimeError("cannot finish an empty rollout")
        assert self._dof_map is not None and self._camera_initial is not None
        frame = pd.DataFrame(self._rows)
        temp_steps = self.steps_path.with_suffix(".parquet.tmp")
        frame.to_parquet(temp_steps, index=False, engine="pyarrow")
        os.replace(temp_steps, self.steps_path)
        payload: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "model": self.model,
            "task_suite": "libero_10",
            "task_id": self.task_id,
            "init_state_index": self.init_state_index,
            "environment_seed": self.environment_seed,
            "physical_gpu": self.physical_gpu,
            "condition": self.condition,
            "task_description": self.task_description,
            "bddl_file": str(self.bddl_file),
            "target_objects": list(self.task_spec.manipulated_objects),
            "task_relevant_entities": list(self.task_spec.relevant_entities),
            "articulation_joints": list(self.task_spec.articulation_joints),
            "requested_phases": list(self.task_spec.phases),
            "success": bool(success),
            "termination": str(termination),
            "num_env_state_rows": len(self._rows),
            "num_policy_calls": self._calls,
            "env_step_range": [int(self._rows[0]["env_step"]), int(self._rows[-1]["env_step"])],
            "created_at_unix": self.created_at_unix,
            "finished_at_unix": time.time(),
            "paths": {
                "steps_parquet": "steps.parquet",
                "policy_calls_jsonl": "policy_calls.jsonl",
                "observation_frames": "observation_frames",
                "latents": "latents",
            },
            "full_sim_state_contract": {
                "primary_replay_fields": ["sim_time", "sim_qpos", "sim_qvel", "sim_act"],
                "additional_fields": ["sim_ctrl", "sim_mocap_pos", "sim_mocap_quat", "sim_userdata"],
                "state_hash": "sim_state_sha256",
                "record_timing": "state after sim.forward at each env timestep; action_from_previous_step led into this row",
            },
            "dof_map": self._dof_map,
            "entity_joint_map": self._entity_joint_names,
            "camera_parameters": {
                "initial_extrinsics": self._camera_initial,
                "dynamic_extrinsics": {
                    "agentview": "steps.agentview_world_from_camera",
                    "robot0_eye_in_hand": "steps.eye_in_hand_world_from_camera",
                    "note": "eye-in-hand extrinsics vary every env step and therefore are stored per row",
                },
            },
            "gripper_state_definition": {
                "measure": "abs(robot0_gripper_qpos[0] - robot0_gripper_qpos[1])",
                "open_if_greater_or_equal": 0.06,
                "closed_if_less_or_equal": 0.04,
                "between_thresholds": "retain previous state (unknown until first threshold crossing)",
            },
            "eef_velocity_definition": "MuJoCo body world-frame linear velocity followed by angular velocity for gripper0_eef",
            "eef_pose_definition": "synchronized MuJoCo gripper0_eef body xpos + xquat (wxyz) after sim.forward; raw observation pose is retained in eef_obs_* columns",
            "model_metadata": dict(model_metadata),
            "policy_protocol": dict(policy_protocol),
        }
        if extra_meta:
            payload.update(dict(extra_meta))
        _write_json(self.meta_path, payload)

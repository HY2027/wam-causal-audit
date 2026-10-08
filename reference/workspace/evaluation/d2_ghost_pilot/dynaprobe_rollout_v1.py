from __future__ import annotations

import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from d2_ghost_scheduler import body_pos, body_quat, get_joint_qpos, get_sim, joint_qpos_slice, joint_qvel_slice, jsonable, qpos_error


SCHEMA_VERSION = "dynaprobe_rollout_v1"
STEP_COLUMNS = [
    "env_step",
    "action",
    "eef_pose",
    "sim_eef_pos",
    "qpos",
    "sim_qpos",
    "sim_qvel",
    "gripper",
    "target_obj_pose",
    "eef_target_xy_dist_m",
    "eef_target_3d_dist_m",
    "sim_eef_target_xy_dist_m",
    "sim_eef_target_3d_dist_m",
    "distractor_poses",
    "infer_latency_ms",
]
CAMERA_NAMES = ("agentview", "robot0_eye_in_hand")


def _sim_joint_map(sim: Any) -> dict[str, Any]:
    """Full MuJoCo joint/dof map; RSR selects robot0 / gripper entries from it."""
    model = sim.model
    out: dict[str, Any] = {}
    for jid in range(int(getattr(model, "njnt", 0) or 0)):
        name = None
        try:
            name = model.joint_id2name(jid)
        except Exception:
            pass
        if not name:
            name = f"joint_{jid}"
        try:
            qs = joint_qpos_slice(sim, str(name))
            vs = joint_qvel_slice(sim, str(name))
            out[str(name)] = {
                "joint_id": jid, "qpos_slice": [int(qs.start), int(qs.stop)],
                "qvel_slice": None if vs is None else [int(vs.start), int(vs.stop)],
                "is_robot_or_gripper": bool("robot0" in str(name).lower() or "gripper" in str(name).lower()),
            }
        except Exception:
            continue
    return out


def _safe_part(value: Any, max_len: int = 128) -> str:
    out = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    return out[:max_len] or "none"


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(jsonable(row), ensure_ascii=False, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(jsonable(payload), f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def _camera_id(model: Any, name: str) -> int:
    for attr in ("camera_name2id", "cam_name2id"):
        fn = getattr(model, attr, None)
        if callable(fn):
            return int(fn(name))
    fn = getattr(model, "name2id", None)
    if callable(fn):
        for kind in ("camera", "cam"):
            try:
                return int(fn(name, kind))
            except Exception:
                pass
    for attr in ("camera_id2name", "cam_id2name"):
        fn = getattr(model, attr, None)
        if callable(fn):
            count = int(getattr(model, "ncam", 0) or 0)
            for idx in range(count):
                if fn(idx) == name:
                    return idx
    raise KeyError(f"Cannot map camera name to id: {name}")


def _intrinsic_from_fovy(width: int, height: int, fovy_deg: float) -> dict[str, Any]:
    fovy_rad = math.radians(float(fovy_deg))
    fy = 0.5 * float(height) / math.tan(0.5 * fovy_rad)
    fx = fy
    cx = (float(width) - 1.0) * 0.5
    cy = (float(height) - 1.0) * 0.5
    return {
        "width": int(width),
        "height": int(height),
        "fovy_deg": float(fovy_deg),
        "fx": float(fx),
        "fy": float(fy),
        "cx": float(cx),
        "cy": float(cy),
        "matrix": [[float(fx), 0.0, float(cx)], [0.0, float(fy), float(cy)], [0.0, 0.0, 1.0]],
        "convention": "pinhole_from_mujoco_vertical_fovy_centered_pixels",
    }


def camera_snapshot(env: Any, *, width: int, height: int, camera_names: Sequence[str] = CAMERA_NAMES) -> dict[str, Any]:
    sim = get_sim(env)
    out: dict[str, Any] = {}
    for camera_name in camera_names:
        camera_id = _camera_id(sim.model, camera_name)
        fovy = float(np.asarray(sim.model.cam_fovy).reshape(-1)[camera_id])
        world_pos = np.asarray(sim.data.cam_xpos[camera_id], dtype=np.float64).reshape(3)
        world_xmat = np.asarray(sim.data.cam_xmat[camera_id], dtype=np.float64).reshape(3, 3)
        world_from_camera = np.eye(4, dtype=np.float64)
        world_from_camera[:3, :3] = world_xmat
        world_from_camera[:3, 3] = world_pos
        camera_from_world = np.linalg.inv(world_from_camera)
        model_pos = np.asarray(sim.model.cam_pos[camera_id], dtype=np.float64).reshape(-1)[:3]
        model_quat = np.asarray(sim.model.cam_quat[camera_id], dtype=np.float64).reshape(-1)[:4]
        out[str(camera_name)] = {
            "camera_id": int(camera_id),
            "intrinsic": _intrinsic_from_fovy(width, height, fovy),
            "model_pos": model_pos,
            "model_quat": model_quat,
            "model_body_id": int(np.asarray(sim.model.cam_bodyid).reshape(-1)[camera_id])
            if hasattr(sim.model, "cam_bodyid")
            else None,
            "world_pos": world_pos,
            "world_xmat": world_xmat,
            "world_from_camera": world_from_camera,
            "camera_from_world": camera_from_world,
            "extrinsic_convention": "MuJoCo sim.data.cam_xpos/cam_xmat at the logged env_step",
        }
    return out


def _robot_qpos(raw_obs: Mapping[str, Any]) -> np.ndarray:
    if "robot0_joint_pos" not in raw_obs:
        raise KeyError("DynaProbe v1 requires LIBERO raw_obs['robot0_joint_pos'] for steps.qpos")
    return np.asarray(raw_obs["robot0_joint_pos"], dtype=np.float32).reshape(-1).copy()


def _eef_pose(raw_obs: Mapping[str, Any]) -> np.ndarray:
    pos = np.asarray(raw_obs.get("robot0_eef_pos", []), dtype=np.float32).reshape(-1)[:3]
    quat = np.asarray(raw_obs.get("robot0_eef_quat", []), dtype=np.float32).reshape(-1)[:4]
    if pos.size != 3 or quat.size != 4:
        raise KeyError("DynaProbe v1 requires robot0_eef_pos and robot0_eef_quat for steps.eef_pose")
    return np.concatenate([pos, quat]).astype(np.float32)


def _gripper_scalar(raw_obs: Mapping[str, Any]) -> float | None:
    if "robot0_gripper_qpos" not in raw_obs:
        return None
    arr = np.asarray(raw_obs["robot0_gripper_qpos"], dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return None
    return float(np.mean(arr))


def _target_pose(env: Any, task: Any) -> np.ndarray:
    sim = get_sim(env)
    pos = body_pos(sim, task.target_body_name).astype(np.float32)
    quat = body_quat(sim, task.target_body_name).astype(np.float32)
    return np.concatenate([pos.reshape(3), quat.reshape(4)]).astype(np.float32)


def _sim_eef_pos(sim: Any) -> np.ndarray:
    """Kinematic EEF position directly from MuJoCo, never from an obs cache."""
    for body_name in ("gripper0_eef", "robot0_eef"):
        try:
            return body_pos(sim, body_name).astype(np.float32)
        except Exception:
            continue
    raise KeyError("Could not locate a MuJoCo robot EEF body (tried gripper0_eef, robot0_eef)")


def _first_event_step(events: Sequence[Any], phase: str) -> int | None:
    for event in events:
        if getattr(event, "phase", None) == phase and getattr(event, "low_level_step", None) is not None:
            return int(event.low_level_step)
    return None


def _severity_from_condition(condition: str) -> str:
    if str(condition) == "clean":
        return "clean"
    if "long" in str(condition):
        return "L2"
    return "L1"


class DynaProbeRolloutV1Writer:
    def __init__(
        self,
        *,
        root: Path,
        axis: str,
        model: str,
        task: str,
        seed: int,
        condition: str,
        image_width: int,
        image_height: int,
        camera_names: Sequence[str] = CAMERA_NAMES,
    ) -> None:
        self.root = Path(root)
        self.axis = str(axis)
        self.model = str(model)
        self.task = str(task)
        self.seed = int(seed)
        self.condition = str(condition)
        self.image_width = int(image_width)
        self.image_height = int(image_height)
        self.camera_names = tuple(str(x) for x in camera_names)
        self.rollout_dir = (
            self.root
            / _safe_part(self.axis)
            / _safe_part(self.model)
            / _safe_part(self.task)
            / f"seed_{self.seed:03d}"
            / _safe_part(self.condition)
        )
        self.steps_path = self.rollout_dir / "steps.parquet"
        self.policy_calls_path = self.rollout_dir / "policy_calls.jsonl"
        self.event_path = self.rollout_dir / "event.json"
        self.meta_path = self.rollout_dir / "meta.json"
        self.latents_dir = self.rollout_dir / "latents"
        self._step_rows: list[dict[str, Any]] = []
        self._camera_static: dict[str, Any] | None = None
        self._sim_joint_map: dict[str, Any] | None = None
        self._camera_poses_by_env_step: list[dict[str, Any]] = []
        self._policy_call_camera_poses: list[dict[str, Any]] = []
        self._target_set_errors: list[float] = []
        self.created_at_unix = time.time()

    def prepare(self, *, overwrite: bool = False) -> None:
        self.rollout_dir.mkdir(parents=True, exist_ok=True)
        self.latents_dir.mkdir(parents=True, exist_ok=True)
        if overwrite:
            for path in (self.steps_path, self.policy_calls_path, self.event_path, self.meta_path):
                if path.exists():
                    path.unlink()
        elif self.policy_calls_path.exists():
            self.policy_calls_path.unlink()

    def _record_camera_pose(self, *, env: Any, env_step: int, destination: list[dict[str, Any]], **extra: Any) -> None:
        cameras = camera_snapshot(
            env,
            width=self.image_width,
            height=self.image_height,
            camera_names=self.camera_names,
        )
        if self._camera_static is None:
            self._camera_static = {
                name: {
                    "camera_id": payload["camera_id"],
                    "intrinsic": payload["intrinsic"],
                    "model_pos": payload["model_pos"],
                    "model_quat": payload["model_quat"],
                    "model_body_id": payload["model_body_id"],
                    "extrinsic_convention": payload["extrinsic_convention"],
                }
                for name, payload in cameras.items()
            }
        destination.append({"env_step": int(env_step), "cameras": cameras, **extra})

    def record_step(
        self,
        *,
        env_step: int,
        env: Any,
        task: Any,
        raw_obs: Mapping[str, Any],
        action: Sequence[float] | None,
        infer_latency_ms: float | None,
    ) -> None:
        action_arr = None if action is None else np.asarray(action, dtype=np.float32).reshape(-1)[:7].copy()
        sim = get_sim(env)
        if self._sim_joint_map is None:
            self._sim_joint_map = _sim_joint_map(sim)
        eef_pose = _eef_pose(raw_obs)
        sim_eef_pos = _sim_eef_pos(sim)
        target_pose = _target_pose(env, task)
        eef_target_delta = eef_pose[:3].astype(np.float64) - target_pose[:3].astype(np.float64)
        sim_eef_target_delta = sim_eef_pos.astype(np.float64) - target_pose[:3].astype(np.float64)
        self._step_rows.append(
            {
                "env_step": int(env_step),
                "action": None if action_arr is None else action_arr.tolist(),
                "eef_pose": eef_pose.tolist(),
                "sim_eef_pos": sim_eef_pos.tolist(),
                "qpos": _robot_qpos(raw_obs).tolist(),
                "sim_qpos": np.asarray(sim.data.qpos, dtype=np.float64).reshape(-1).tolist(),
                "sim_qvel": np.asarray(sim.data.qvel, dtype=np.float64).reshape(-1).tolist(),
                "gripper": _gripper_scalar(raw_obs),
                "target_obj_pose": target_pose.tolist(),
                "eef_target_xy_dist_m": float(np.linalg.norm(eef_target_delta[:2])),
                "eef_target_3d_dist_m": float(np.linalg.norm(eef_target_delta)),
                "sim_eef_target_xy_dist_m": float(np.linalg.norm(sim_eef_target_delta[:2])),
                "sim_eef_target_3d_dist_m": float(np.linalg.norm(sim_eef_target_delta)),
                "distractor_poses": [],
                "infer_latency_ms": None if infer_latency_ms is None else float(infer_latency_ms),
            }
        )
        self._record_camera_pose(env=env, env_step=int(env_step), destination=self._camera_poses_by_env_step)

    def record_policy_call(
        self,
        *,
        call_idx: int,
        env_step: int,
        env: Any,
        action_chunk: Sequence[Sequence[float]] | np.ndarray,
        obs_frame_ref: str,
        perturbation_in_obs: bool,
        event_phase: str,
        infer_latency_ms: float | None,
        target_set_error: float | None,
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        if target_set_error is not None and np.isfinite(float(target_set_error)):
            self._target_set_errors.append(float(target_set_error))
        self._record_camera_pose(
            env=env,
            env_step=int(env_step),
            destination=self._policy_call_camera_poses,
            call_idx=int(call_idx),
        )
        row = {
            "call_idx": int(call_idx),
            "env_step": int(env_step),
            "action_chunk": np.asarray(action_chunk, dtype=np.float32).tolist(),
            "obs_frame_ref": str(obs_frame_ref),
            "perturbation_in_obs": bool(perturbation_in_obs),
            "event_phase": str(event_phase),
            "infer_latency_ms": None if infer_latency_ms is None else float(infer_latency_ms),
            "target_set_error": None if target_set_error is None else float(target_set_error),
            "latent_path": None,
            "stitch_layout": None,
            "frame_to_envstep": None,
            "frame_alignability": None,
        }
        if extra:
            row.update(dict(extra))
        _append_jsonl(self.policy_calls_path, row)

    def write_steps(self) -> None:
        self.rollout_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(self._step_rows, columns=STEP_COLUMNS)
        df.to_parquet(self.steps_path, index=False, engine="pyarrow")

    def write_event(
        self,
        *,
        scheduler: Any,
        task: Any,
        invalid: bool,
        invalid_reason: str,
        delta_record: Mapping[str, Any] | None,
    ) -> None:
        custom_event = getattr(scheduler, "to_dynaprobe_event", None)
        if callable(custom_event):
            payload = custom_event(invalid=invalid, invalid_reason=invalid_reason, delta_record=delta_record)
            payload.setdefault("severity_level", _severity_from_condition(self.condition))
            payload.setdefault("occlusion_span", None)
            payload.setdefault("restore_env_step", None)
            _write_json(self.event_path, payload)
            return

        delta_xy = None if getattr(scheduler, "delta_xy", None) is None else np.asarray(scheduler.delta_xy).tolist()
        magnitude = None
        if delta_xy is not None:
            magnitude = float(np.linalg.norm(np.asarray(delta_xy, dtype=np.float64).reshape(-1)[:2]))
        t0_env_step = _first_event_step(getattr(scheduler, "events", []), "B_visible")
        if t0_env_step is None and getattr(scheduler, "trigger_record", None) is not None:
            t0_env_step = int(scheduler.trigger_record.low_level_step)
        payload = {
            "tuple": {
                "target": getattr(task, "target_object_name", None),
                "trigger": {
                    "type": "eef_target_distance",
                    "radius_m": getattr(scheduler, "trigger_distance_m", None),
                    "hit_step": None
                    if getattr(scheduler, "trigger_record", None) is None
                    else int(scheduler.trigger_record.low_level_step),
                },
                "transform": {
                    "type": "target_free_joint_xy_translation",
                    "delta_xy": delta_xy,
                    "selected_delta": delta_record,
                },
                "magnitude": magnitude,
                "transition": "policy_observation_window",
                "visibility": "policy_observation_visible" if self.condition != "clean" else "clean",
                "reversal": "restore_target_A",
                "dose": int(getattr(scheduler, "b_observation_calls", 0)),
            },
            "t0_env_step": t0_env_step,
            "restore_env_step": getattr(scheduler, "return_step", None),
            "occlusion_span": None,
            "dose_obs_count": int(getattr(scheduler, "b_observation_calls", 0)),
            "validation": {
                "trigger_hit": bool(getattr(scheduler, "trigger_record", None) is not None),
                "set_error_max": float(max(self._target_set_errors)) if self._target_set_errors else 0.0,
                "invalid": bool(invalid),
                "invalid_reason": str(invalid_reason or ""),
            },
            "severity_level": _severity_from_condition(self.condition),
        }
        _write_json(self.event_path, payload)

    def write_meta(
        self,
        *,
        summary: Mapping[str, Any],
        suite: str,
        instruction: str,
        source_step_log_path: Path,
        source_video_path: Path,
        official_hash: str,
        server_metadata: Mapping[str, Any] | None,
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "model": self.model,
            "axis": self.axis,
            "task": self.task,
            "seed": self.seed,
            "condition": self.condition,
            "suite": str(suite),
            "instruction": str(instruction),
            "created_at_unix": self.created_at_unix,
            "paths": {
                "rollout_dir": str(self.rollout_dir),
                "steps_parquet": str(self.steps_path),
                "policy_calls_jsonl": str(self.policy_calls_path),
                "event_json": str(self.event_path),
                "meta_json": str(self.meta_path),
                "source_step_log_path": str(source_step_log_path),
                "source_video_path": str(source_video_path),
            },
            "summary": dict(summary),
            "official_config_hash": str(official_hash),
            "field_sources": {
                "steps.env_step": "LIBERO env timestep",
                "steps.action": "executed low-level action passed to env.step",
                "steps.eef_pose": "raw_obs robot0_eef_pos + robot0_eef_quat",
                "steps.sim_eef_pos": "MuJoCo gripper0_eef body xpos (kinematic, post-forward)",
                "steps.qpos": "raw_obs robot0_joint_pos",
                "steps.sim_qpos": "MuJoCo sim.data.qpos full vector",
                "steps.sim_qvel": "MuJoCo sim.data.qvel full vector",
                "steps.eef_target_xy_dist_m": "norm(robot0_eef_pos[:2] - target_obj_pose[:2])",
                "steps.eef_target_3d_dist_m": "norm(robot0_eef_pos - target_obj_pose)",
                "steps.sim_eef_target_xy_dist_m": "norm(sim_eef_pos[:2] - target_obj_pose[:2])",
                "steps.sim_eef_target_3d_dist_m": "norm(sim_eef_pos - target_obj_pose)",
                "mujoco_joint_map": self._sim_joint_map,
                "steps.gripper": "mean(raw_obs robot0_gripper_qpos)",
                "steps.target_obj_pose": "MuJoCo target body xpos + xquat",
                "policy_calls.action_chunk": "raw pi0.5 result['actions'] when present",
            },
            "camera_parameters": self._camera_static or {},
            "camera_poses_by_env_step": self._camera_poses_by_env_step,
            "policy_call_camera_poses": self._policy_call_camera_poses,
            "obs_frame_ref_convention": {
                "type": "rollout_video_frame_plus_deterministic_pi05_preprocess",
                "policy_input_keys": ["agentview_image", "robot0_eye_in_hand_image"],
                "preprocess": "raw_obs image [::-1, ::-1], resize_with_pad to 224x224, convert_to_uint8",
                "note": "No extra RGB image files are written by DynaProbe v1 for pi0.5.",
            },
            "server_metadata": dict(server_metadata or {}),
            "latents": None,
        }
        if extra:
            payload.update(dict(extra))
        _write_json(self.meta_path, payload)

    def finish(
        self,
        *,
        summary: Mapping[str, Any],
        suite: str,
        instruction: str,
        scheduler: Any,
        task: Any,
        invalid: bool,
        invalid_reason: str,
        delta_record: Mapping[str, Any] | None,
        source_step_log_path: Path,
        source_video_path: Path,
        official_hash: str,
        server_metadata: Mapping[str, Any] | None,
        extra_meta: Mapping[str, Any] | None = None,
    ) -> None:
        self.write_steps()
        self.write_event(
            scheduler=scheduler,
            task=task,
            invalid=invalid,
            invalid_reason=invalid_reason,
            delta_record=delta_record,
        )
        self.write_meta(
            summary=summary,
            suite=suite,
            instruction=instruction,
            source_step_log_path=source_step_log_path,
            source_video_path=source_video_path,
            official_hash=official_hash,
            server_metadata=server_metadata,
            extra=extra_meta,
        )


def target_set_error(env: Any, task: Any, expected_qpos: Sequence[float] | None) -> float | None:
    if expected_qpos is None:
        return None
    actual = get_joint_qpos(get_sim(env), task.target_joint_name)
    return qpos_error(actual, expected_qpos)

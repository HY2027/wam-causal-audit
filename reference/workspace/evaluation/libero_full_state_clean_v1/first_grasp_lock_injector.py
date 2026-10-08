from __future__ import annotations

"""Finite first-grasp lock used by the LIBERO retry experiment.

The object free joint is re-committed only while ``state == "locked"``.  The
release transition performs one final exact commit with zero qvel, after which
``after_step`` is read-only.  This explicit state boundary is intentional: it
prevents the permanent-object-lock bug found in the legacy online injector.
"""

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from d2_ghost_scheduler import (
    body_pos,
    get_eef_pos,
    get_joint_qpos,
    get_sim,
    joint_qvel_slice,
    jsonable,
    set_target_qpos_and_forward,
    target_gripper_contact,
)


@dataclass
class FirstGraspLockInjector:
    target_object_name: str
    target_body_name: str
    target_joint_name: str
    near_distance_m: float = 0.15
    close_command_threshold: float = 0.5
    open_aperture_m: float = 0.055
    closed_aperture_m: float = 0.045
    depart_z_m: float = 0.015
    depart_distance_m: float = 0.015
    min_lock_steps: int = 4
    release_clear_steps: int = 3
    immediate_check_steps: int = 3
    immediate_displacement_tolerance_m: float = 0.005
    immediate_up_tolerance_m: float = 0.003

    state: str = "armed"
    lock_qpos: np.ndarray | None = None
    lock_body_pos: np.ndarray | None = None
    trigger_eef_pos: np.ndarray | None = None
    trigger_distance_m: float | None = None
    trigger_env_step: int | None = None
    release_env_step: int | None = None
    lock_steps: int = 0
    writes_after_release: int = 0
    max_locked_target_displacement_m: float = 0.0
    max_locked_target_up_m: float = 0.0
    release_qvel_norm: float | None = None
    release_had_gripper_contact: bool | None = None
    immediate_post_release_max_displacement_m: float = 0.0
    immediate_post_release_max_up_m: float = 0.0
    post_release_steps_seen: int = 0
    gripper_opened_after_release: bool = False
    retry_env_step: int | None = None
    retry_distance_m: float | None = None
    regrasp_success: bool = False
    blind_continuation: bool = False
    _release_eef_pos: np.ndarray | None = None
    _previous_gripper_command: float | None = None
    _release_clear_streak: int = 0
    _write_trace: list[dict[str, Any]] = field(default_factory=list)
    _step_trace: list[dict[str, Any]] = field(default_factory=list)

    @staticmethod
    def gripper_aperture(raw_obs: Mapping[str, Any]) -> float:
        qpos = np.asarray(raw_obs.get("robot0_gripper_qpos", []), dtype=np.float64).reshape(-1)
        return float(abs(qpos[0] - qpos[1])) if qpos.size >= 2 else float("nan")

    def _distance(self, env: Any, raw_obs: Mapping[str, Any] | None) -> tuple[np.ndarray, np.ndarray, float]:
        sim = get_sim(env)
        eef = get_eef_pos(env, raw_obs)
        target = body_pos(sim, self.target_body_name)
        return eef, target, float(np.linalg.norm(eef - target))

    def before_action(
        self,
        env: Any,
        raw_obs: Mapping[str, Any],
        action: Sequence[float],
        *,
        env_step: int,
    ) -> bool:
        """Arm the lock immediately before the first nearby close token.

        LIBERO's deployed convention is positive=close, negative=open.  The
        physical-aperture gate rejects stale positive commands after closure.
        """
        action_value = np.asarray(action, dtype=np.float64).reshape(-1)
        command = float(action_value[6]) if action_value.size >= 7 else float("nan")
        triggered = False
        if self.state == "armed" and np.isfinite(command):
            eef, target, distance = self._distance(env, raw_obs)
            aperture = self.gripper_aperture(raw_obs)
            # Use the first *eligible* nearby close token.  If a model starts
            # issuing close slightly before entering the 15-cm band, requiring
            # a command sign edge would miss the physical close onset.
            nearby_close = command > self.close_command_threshold
            if nearby_close and distance <= self.near_distance_m and aperture >= self.open_aperture_m:
                sim = get_sim(env)
                self.lock_qpos = get_joint_qpos(sim, self.target_joint_name)
                self.lock_body_pos = target.copy()
                self.trigger_eef_pos = eef.copy()
                self.trigger_distance_m = distance
                self.trigger_env_step = int(env_step)
                self.state = "locked"
                self._commit(env, env_step=int(env_step), phase="lock_before_first_close")
                triggered = True
        self._previous_gripper_command = command
        return triggered

    def _commit(self, env: Any, *, env_step: int, phase: str) -> None:
        if self.state == "released":
            self.writes_after_release += 1
            raise RuntimeError("first_grasp_lock attempted a target write after release")
        if self.lock_qpos is None:
            raise RuntimeError("lock qpos is unavailable")
        zeroed = set_target_qpos_and_forward(env, self.target_joint_name, self.lock_qpos)
        sim = get_sim(env)
        actual = get_joint_qpos(sim, self.target_joint_name)
        self._write_trace.append({
            "env_step": int(env_step),
            "phase": str(phase),
            "qvel_zeroed": bool(zeroed),
            "max_qpos_error": float(np.max(np.abs(actual - self.lock_qpos))),
        })

    def after_step(
        self,
        env: Any,
        raw_obs: Mapping[str, Any],
        action: Sequence[float],
        *,
        env_step: int,
    ) -> str:
        """Apply the finite lock or observe the unmodified post-release state."""
        action_value = np.asarray(action, dtype=np.float64).reshape(-1)
        command = float(action_value[6]) if action_value.size >= 7 else float("nan")
        if self.state == "locked":
            self._commit(env, env_step=int(env_step), phase="hold_first_grasp_failed")
            self.lock_steps += 1
            eef, target, distance = self._distance(env, raw_obs)
            assert self.lock_body_pos is not None and self.trigger_eef_pos is not None and self.trigger_distance_m is not None
            displacement = float(np.linalg.norm(target - self.lock_body_pos))
            up = float(target[2] - self.lock_body_pos[2])
            self.max_locked_target_displacement_m = max(self.max_locked_target_displacement_m, displacement)
            self.max_locked_target_up_m = max(self.max_locked_target_up_m, up)
            aperture = self.gripper_aperture(raw_obs)
            contact = bool(target_gripper_contact(env, self.target_object_name))
            departed = bool(
                (eef[2] - self.trigger_eef_pos[2] >= self.depart_z_m)
                or (distance - self.trigger_distance_m >= self.depart_distance_m)
            )
            if departed and not contact:
                self._release_clear_streak += 1
            else:
                self._release_clear_streak = 0
            self._step_trace.append({
                "env_step": int(env_step), "state": "locked", "gripper_command": command,
                "gripper_aperture_m": aperture, "eef_target_distance_m": distance,
                "eef_z_from_trigger_m": float(eef[2] - self.trigger_eef_pos[2]),
                "target_displacement_m": displacement, "target_up_m": up,
                "target_gripper_contact": contact, "departed": departed,
                "release_clear_streak": self._release_clear_streak,
            })
            if (
                self.lock_steps >= self.min_lock_steps
                and aperture <= self.closed_aperture_m
                and self._release_clear_streak >= self.release_clear_steps
            ):
                # The final hold commit already put the object exactly at A and
                # zeroed both linear and angular free-joint velocities.
                sim = get_sim(env)
                qvel_slice = joint_qvel_slice(sim, self.target_joint_name)
                qvel = (
                    np.asarray(sim.data.qvel[qvel_slice], dtype=np.float64).reshape(-1)
                    if qvel_slice is not None else np.asarray([np.nan], dtype=np.float64)
                )
                self.release_qvel_norm = float(np.linalg.norm(qvel))
                self.release_had_gripper_contact = contact
                self.release_env_step = int(env_step)
                self._release_eef_pos = eef.copy()
                self.state = "released"
                return "released"
            return "locked"

        if self.state == "released":
            eef, target, distance = self._distance(env, raw_obs)
            assert self.lock_body_pos is not None and self._release_eef_pos is not None
            displacement = float(np.linalg.norm(target - self.lock_body_pos))
            up = float(target[2] - self.lock_body_pos[2])
            aperture = self.gripper_aperture(raw_obs)
            contact = bool(target_gripper_contact(env, self.target_object_name))
            self.post_release_steps_seen += 1
            if self.post_release_steps_seen <= self.immediate_check_steps:
                self.immediate_post_release_max_displacement_m = max(
                    self.immediate_post_release_max_displacement_m, displacement
                )
                self.immediate_post_release_max_up_m = max(self.immediate_post_release_max_up_m, up)
            if command < -self.close_command_threshold or aperture >= self.open_aperture_m:
                self.gripper_opened_after_release = True
            if (
                self.retry_env_step is None
                and self.gripper_opened_after_release
                and command > self.close_command_threshold
                and distance <= self.near_distance_m
            ):
                self.retry_env_step = int(env_step)
                self.retry_distance_m = distance
            if self.retry_env_step is not None and (up >= 0.01 or displacement >= 0.015) and contact:
                self.regrasp_success = True
            if self.retry_env_step is None and displacement <= self.immediate_displacement_tolerance_m:
                extra_up = float(eef[2] - self._release_eef_pos[2])
                extra_away = float(distance - np.linalg.norm(self._release_eef_pos - self.lock_body_pos))
                if extra_up >= 0.02 or extra_away >= 0.02:
                    self.blind_continuation = True
            self._step_trace.append({
                "env_step": int(env_step), "state": "released", "gripper_command": command,
                "gripper_aperture_m": aperture, "eef_target_distance_m": distance,
                "target_displacement_m": displacement, "target_up_m": up,
                "target_gripper_contact": contact,
            })
            return "released"
        return self.state

    def final_event(self, *, final_success: bool, simulation_video: str | None, predicted_video: str | None) -> dict[str, Any]:
        first_failed = bool(
            self.release_env_step is not None
            and self.max_locked_target_displacement_m <= 1e-6
            and self.max_locked_target_up_m <= 1e-6
            and self.release_had_gripper_contact is False
        )
        immediate_safe = bool(
            self.post_release_steps_seen >= self.immediate_check_steps
            and self.immediate_post_release_max_displacement_m <= self.immediate_displacement_tolerance_m
            and self.immediate_post_release_max_up_m <= self.immediate_up_tolerance_m
        )
        valid = bool(
            self.trigger_env_step is not None
            and first_failed
            and immediate_safe
            and (self.release_qvel_norm is not None and self.release_qvel_norm <= 1e-10)
            and self.writes_after_release == 0
            and simulation_video is not None
        )
        return jsonable({
            "schema_version": "libero_first_grasp_lock_v1_event",
            "trigger_env_step": self.trigger_env_step,
            "release_env_step": self.release_env_step,
            "release_clear_steps_required": self.release_clear_steps,
            "release_clear_streak_at_release": self._release_clear_streak,
            "lock_steps": self.lock_steps,
            "trigger_distance_m": self.trigger_distance_m,
            "lock_qpos": self.lock_qpos,
            "lock_body_pos": self.lock_body_pos,
            "first_grasp_failed": first_failed,
            "blind_continuation": self.blind_continuation,
            "retry": self.retry_env_step is not None,
            "retry_env_step": self.retry_env_step,
            "retry_distance_m": self.retry_distance_m,
            "retry_latency": None if self.retry_env_step is None or self.release_env_step is None else int(self.retry_env_step - self.release_env_step),
            "regrasp_success": self.regrasp_success,
            "final_success": bool(final_success),
            "release_qvel_norm": self.release_qvel_norm,
            "release_had_gripper_contact": self.release_had_gripper_contact,
            "max_locked_target_displacement_m": self.max_locked_target_displacement_m,
            "max_locked_target_up_m": self.max_locked_target_up_m,
            "immediate_post_release_max_displacement_m": self.immediate_post_release_max_displacement_m,
            "immediate_post_release_max_up_m": self.immediate_post_release_max_up_m,
            "writes_after_release": self.writes_after_release,
            "simulation_video": simulation_video,
            "predicted_video": predicted_video,
            "validation": {
                "triggered": self.trigger_env_step is not None,
                "released": self.release_env_step is not None,
                "first_grasp_failed": first_failed,
                "release_zero_velocity": self.release_qvel_norm is not None and self.release_qvel_norm <= 1e-10,
                "release_without_gripper_contact": self.release_had_gripper_contact is False,
                "post_release_immediate_safe": immediate_safe,
                "no_writes_after_release": self.writes_after_release == 0,
                "simulation_video_present": simulation_video is not None,
                "valid": valid,
            },
            "write_trace": self._write_trace,
            "step_trace": self._step_trace,
        })

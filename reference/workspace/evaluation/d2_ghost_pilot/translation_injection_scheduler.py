from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from d2_ghost_scheduler import (
    body_pos,
    body_quat,
    get_eef_pos,
    get_joint_qpos,
    get_sim,
    jsonable,
    qpos_error,
    set_target_qpos_and_forward,
    target_gripper_contact,
)
from d2_pilot_utils import target_problem_contacts


@dataclass
class TranslationTriggerRecord:
    low_level_step: int
    distance_m: float
    eef_pos: list[float]
    target_center: list[float]
    target_displacement_m: float
    target_gripper_contact: bool


def _contact_mentions_gripper(contact: Mapping[str, Any]) -> bool:
    text = " ".join(str(contact.get(key, "")) for key in ("geom1", "geom2", "body1", "body2")).lower()
    return any(token in text for token in ("gripper", "finger", "eef", "hand"))


class TranslationInjectionScheduler:
    """Kinematic target-object XY translation, using the D2 MuJoCo qpos path."""

    def __init__(
        self,
        *,
        target_object_name: str,
        target_body_name: str,
        target_joint_name: str,
        trigger_radius_m: float,
        delta_xy_m: Sequence[float],
        n_slide_steps: int,
        direction_rule: str,
        axis: str = "DYN",
        set_error_tolerance: float = 1e-5,
        trigger_only: bool = False,
    ) -> None:
        delta = np.asarray(delta_xy_m, dtype=np.float64).reshape(-1)
        if delta.size != 2:
            raise ValueError(f"delta_xy_m must have shape (2,), got {delta.shape}")
        steps = int(n_slide_steps)
        if steps < 1:
            raise ValueError(f"n_slide_steps must be >= 1, got {n_slide_steps}")
        self.target_object_name = str(target_object_name)
        self.target_body_name = str(target_body_name)
        self.target_joint_name = str(target_joint_name)
        self.trigger_distance_m = float(trigger_radius_m)
        self.delta_xy = delta.copy()
        self.n_slide_steps = steps
        self.direction_rule = str(direction_rule)
        self.axis = str(axis)
        self.set_error_tolerance = float(set_error_tolerance)
        self.trigger_only = bool(trigger_only)

        self.requested_delta_xy = delta.copy()
        self.trigger_type = "eef_target_distance"
        self.a_qpos: np.ndarray | None = None
        self.b_qpos: np.ndarray | None = None
        self.a_body_pos: np.ndarray | None = None
        self.b_body_pos: np.ndarray | None = None
        self.a_body_quat: np.ndarray | None = None
        self.b_body_quat: np.ndarray | None = None
        self.trigger_record: TranslationTriggerRecord | None = None
        self.return_step: int | None = None
        self.b_observation_calls = 0
        self.initial_endpoint_static = False

        self.state = "armed"
        self.invalid_reason = ""
        self._set_errors: list[float] = []
        self._qpos_trace: list[dict[str, Any]] = []

    @property
    def invalid(self) -> bool:
        return self.state == "invalid"

    @property
    def t0_env_step(self) -> int | None:
        if self.trigger_record is None:
            return None
        return int(self.trigger_record.low_level_step)

    def capture_a(self, env: Any, raw_obs: Mapping[str, Any] | None = None) -> None:
        sim = get_sim(env)
        qpos = get_joint_qpos(sim, self.target_joint_name)
        if qpos.size < 7:
            raise ValueError(
                f"Translation injection expects a free-joint qpos with >=7 values for "
                f"{self.target_joint_name}, got shape {qpos.shape}"
            )
        self.a_qpos = qpos.copy()
        self.delta_xy = self._resolve_delta_xy(env, raw_obs)
        self.b_qpos = qpos.copy()
        self.b_qpos[:2] = self.a_qpos[:2] + self.delta_xy
        self.a_body_pos = body_pos(sim, self.target_body_name)
        self.a_body_quat = body_quat(sim, self.target_body_name)
        self.b_body_pos = self.a_body_pos.copy()
        self.b_body_pos[:2] = self.b_body_pos[:2] + self.delta_xy
        self.b_body_quat = self.a_body_quat.copy()

    def no_trigger(self) -> bool:
        return self.trigger_record is None

    def trigger_at_current_state(
        self,
        env: Any,
        raw_obs: Mapping[str, Any] | None,
        *,
        low_level_step: int,
        trigger_type: str = "after_chunk_1",
    ) -> dict[str, Any]:
        if self.trigger_record is not None:
            return self.maybe_apply_after_step(env, raw_obs, low_level_step=low_level_step)
        if self.a_qpos is None:
            self.capture_a(env, raw_obs)

        sim = get_sim(env)
        eef_pos = get_eef_pos(env, raw_obs)
        target_center = body_pos(sim, self.target_body_name)
        distance = float(np.linalg.norm(eef_pos - target_center))
        moved = float(np.linalg.norm(get_joint_qpos(sim, self.target_joint_name)[:3] - self.a_qpos[:3]))
        gripper_contact = target_gripper_contact(env, self.target_object_name)
        self.trigger_type = str(trigger_type or "forced")
        self.trigger_record = TranslationTriggerRecord(
            low_level_step=int(low_level_step),
            distance_m=distance,
            eef_pos=eef_pos.tolist(),
            target_center=target_center.tolist(),
            target_displacement_m=moved,
            target_gripper_contact=bool(gripper_contact),
        )
        if gripper_contact:
            self._mark_invalid("trigger_hit_after_target_gripper_contact")
            return self._record_trace(
                env,
                low_level_step=low_level_step,
                phase="invalid",
                fraction=0.0,
                eef_pos=eef_pos,
                target_center=target_center,
                distance=distance,
            )
        self.state = "sliding"
        return self.maybe_apply_after_step(env, raw_obs, low_level_step=low_level_step)

    def restore_a(self, env: Any) -> None:
        if self.a_qpos is not None:
            set_target_qpos_and_forward(env, self.target_joint_name, self.a_qpos)

    def apply_static_endpoint_at_start(
        self,
        env: Any,
        raw_obs: Mapping[str, Any] | None,
        *,
        low_level_step: int = 0,
    ) -> dict[str, Any]:
        if self.a_qpos is None:
            self.capture_a(env)

        self.initial_endpoint_static = True
        sim = get_sim(env)
        eef_pos = get_eef_pos(env, raw_obs)
        target_center = body_pos(sim, self.target_body_name)
        distance = float(np.linalg.norm(eef_pos - target_center))
        gripper_contact = target_gripper_contact(env, self.target_object_name)
        if self.trigger_record is None:
            self.trigger_record = TranslationTriggerRecord(
                low_level_step=int(low_level_step),
                distance_m=distance,
                eef_pos=eef_pos.tolist(),
                target_center=target_center.tolist(),
                target_displacement_m=0.0,
                target_gripper_contact=bool(gripper_contact),
            )
        if gripper_contact:
            self._mark_invalid("initial_endpoint_target_gripper_collision")
            return self._record_trace(
                env,
                low_level_step=low_level_step,
                phase="invalid",
                fraction=1.0,
                eef_pos=eef_pos,
                target_center=target_center,
                distance=distance,
                gripper_contact=True,
            )

        target_qpos = self.b_qpos.copy()
        commit = self._commit_qpos_with_collision_check(env, target_qpos, check_gripper_contact=True)
        if not commit["committed"]:
            return self._record_trace(
                env,
                low_level_step=low_level_step,
                phase="invalid",
                fraction=1.0,
                requested_qpos=target_qpos,
                set_error=commit["set_error"],
                problem_contacts=commit["problem_contacts"],
                gripper_contact=commit["gripper_contact"],
                eef_pos=eef_pos,
                target_center=target_center,
                distance=distance,
            )

        self.state = "static_endpoint"
        self.return_step = None
        self.b_body_pos = body_pos(get_sim(env), self.target_body_name)
        self.b_body_quat = body_quat(get_sim(env), self.target_body_name)
        return self._record_trace(
            env,
            low_level_step=low_level_step,
            phase="static_endpoint",
            fraction=1.0,
            requested_qpos=target_qpos,
            set_error=commit["set_error"],
            problem_contacts=commit["problem_contacts"],
            gripper_contact=commit["gripper_contact"],
            eef_pos=eef_pos,
            target_center=target_center,
            distance=distance,
        )

    def maybe_apply_after_step(self, env: Any, raw_obs: Mapping[str, Any] | None, *, low_level_step: int) -> dict[str, Any]:
        if self.a_qpos is None:
            self.capture_a(env, raw_obs)

        sim = get_sim(env)
        if self.invalid:
            return self._record_trace(env, low_level_step=low_level_step, phase="invalid", fraction=None)

        eef_pos = get_eef_pos(env, raw_obs)
        target_center = body_pos(sim, self.target_body_name)
        distance = float(np.linalg.norm(eef_pos - target_center))

        if self.trigger_record is None:
            if distance > self.trigger_distance_m:
                return self._record_trace(
                    env,
                    low_level_step=low_level_step,
                    phase="armed",
                    fraction=0.0,
                    eef_pos=eef_pos,
                    target_center=target_center,
                    distance=distance,
                )
            moved = float(np.linalg.norm(get_joint_qpos(sim, self.target_joint_name)[:3] - self.a_qpos[:3]))
            gripper_contact = target_gripper_contact(env, self.target_object_name)
            self.trigger_record = TranslationTriggerRecord(
                low_level_step=int(low_level_step),
                distance_m=distance,
                eef_pos=eef_pos.tolist(),
                target_center=target_center.tolist(),
                target_displacement_m=moved,
                target_gripper_contact=bool(gripper_contact),
            )
            if gripper_contact:
                self._mark_invalid("trigger_hit_after_target_gripper_contact")
                return self._record_trace(
                    env,
                    low_level_step=low_level_step,
                    phase="invalid",
                    fraction=0.0,
                    eef_pos=eef_pos,
                    target_center=target_center,
                    distance=distance,
                )
            self.state = "sliding"

        if self.trigger_only:
            # Stationary-occlusion condition: the target must not drift when
            # the arm contacts it during the visual blackout.  Re-commit the
            # captured free-joint qpos every physics step and zero its qvel.
            requested_qpos = self.a_qpos.copy()
            set_target_qpos_and_forward(env, self.target_joint_name, requested_qpos)
            actual_qpos = get_joint_qpos(get_sim(env), self.target_joint_name)
            set_error = qpos_error(actual_qpos, requested_qpos)
            self._set_errors.append(float(set_error))
            target_center = body_pos(get_sim(env), self.target_body_name)
            self.state = "holding"
            return self._record_trace(
                env,
                low_level_step=low_level_step,
                phase="trigger_only",
                fraction=0.0,
                requested_qpos=requested_qpos,
                set_error=set_error,
                eef_pos=eef_pos,
                target_center=target_center,
                distance=distance,
            )

        fraction = self._fraction_for_step(int(low_level_step))
        phase = "instant" if self.n_slide_steps == 1 and fraction >= 1.0 else "sliding"
        if fraction >= 1.0 and self.n_slide_steps > 1:
            phase = "holding"
        already_holding = self.state == "holding"
        target_qpos = self.a_qpos.copy()
        target_qpos[:2] = self.a_qpos[:2] + self.delta_xy * fraction
        commit = self._commit_qpos_with_collision_check(
            env,
            target_qpos,
            check_gripper_contact=not already_holding,
        )
        if not commit["committed"]:
            return self._record_trace(
                env,
                low_level_step=low_level_step,
                phase="invalid",
                fraction=fraction,
                requested_qpos=target_qpos,
                set_error=commit["set_error"],
                problem_contacts=commit["problem_contacts"],
                gripper_contact=commit["gripper_contact"],
                eef_pos=eef_pos,
                target_center=target_center,
                distance=distance,
            )
        if fraction >= 1.0:
            self.state = "holding"
            self.return_step = None
            self.b_body_pos = body_pos(get_sim(env), self.target_body_name)
            self.b_body_quat = body_quat(get_sim(env), self.target_body_name)
        return self._record_trace(
            env,
            low_level_step=low_level_step,
            phase=phase,
            fraction=fraction,
            requested_qpos=target_qpos,
            set_error=commit["set_error"],
            problem_contacts=commit["problem_contacts"],
            gripper_contact=commit["gripper_contact"],
            eef_pos=eef_pos,
            target_center=target_center,
            distance=distance,
        )

    def _fraction_for_step(self, low_level_step: int) -> float:
        t0 = self.t0_env_step
        if t0 is None:
            return 0.0
        if self.n_slide_steps <= 1:
            return 1.0
        elapsed = max(0, int(low_level_step) - int(t0) + 1)
        return float(min(1.0, elapsed / float(self.n_slide_steps)))

    def _resolve_delta_xy(self, env: Any, raw_obs: Mapping[str, Any] | None) -> np.ndarray:
        rule = self.direction_rule
        requested = np.asarray(self.requested_delta_xy, dtype=np.float64).reshape(2)
        magnitude = float(np.linalg.norm(requested))
        if rule == "explicit_delta_xy" or magnitude < 1e-12:
            return requested.copy()

        sim = get_sim(env)
        eef_pos = get_eef_pos(env, raw_obs)
        target_pos = body_pos(sim, self.target_body_name)
        eef_to_target = np.asarray(
            [target_pos[0] - eef_pos[0], target_pos[1] - eef_pos[1]],
            dtype=np.float64,
        )
        norm = float(np.linalg.norm(eef_to_target))
        if not np.isfinite(norm) or norm < 1e-9:
            eef_to_target = np.asarray([1.0, 0.0], dtype=np.float64)
        else:
            eef_to_target = eef_to_target / norm
        perpendicular = np.asarray([-eef_to_target[1], eef_to_target[0]], dtype=np.float64)
        perp_norm = float(np.linalg.norm(perpendicular))
        if not np.isfinite(perp_norm) or perp_norm < 1e-9:
            perpendicular = np.asarray([0.0, 1.0], dtype=np.float64)
        else:
            perpendicular = perpendicular / perp_norm

        directions = {
            "+perpendicular_to_EEF_target": perpendicular,
            "-perpendicular_to_EEF_target": -perpendicular,
            "+EEF_to_target_direction": eef_to_target,
            "-EEF_to_target_direction": -eef_to_target,
        }
        if rule not in directions:
            raise ValueError(
                f"Unknown translation direction_rule {rule!r}; supported rules are "
                f"explicit_delta_xy and {sorted(directions)}"
            )
        return directions[rule] * magnitude

    def _commit_qpos_with_collision_check(
        self,
        env: Any,
        target_qpos: np.ndarray,
        *,
        check_gripper_contact: bool = True,
    ) -> dict[str, Any]:
        sim = get_sim(env)
        previous_qpos = get_joint_qpos(sim, self.target_joint_name)
        qvel_zeroed = set_target_qpos_and_forward(env, self.target_joint_name, target_qpos)
        actual_qpos = get_joint_qpos(sim, self.target_joint_name)
        set_error = float(qpos_error(actual_qpos, target_qpos))
        problem_contacts = target_problem_contacts(env, self.target_object_name)
        gripper_contact = target_gripper_contact(env, self.target_object_name)
        if not check_gripper_contact and gripper_contact:
            problem_contacts = [
                contact
                for contact in problem_contacts
                if not _contact_mentions_gripper(contact)
            ]
        gripper_collision = bool(gripper_contact) if check_gripper_contact else False
        if problem_contacts or gripper_collision or set_error > self.set_error_tolerance:
            set_target_qpos_and_forward(env, self.target_joint_name, previous_qpos)
            reasons = []
            if problem_contacts:
                reasons.append("target_problem_collision")
            if gripper_collision:
                reasons.append("target_gripper_collision")
            if set_error > self.set_error_tolerance:
                reasons.append(f"set_error>{self.set_error_tolerance:g}")
            self._mark_invalid(";".join(reasons) or "translation_candidate_invalid")
            return {
                "committed": False,
                "qvel_zeroed": bool(qvel_zeroed),
                "set_error": set_error,
                "problem_contacts": problem_contacts,
                "gripper_contact": bool(gripper_contact),
            }
        self._set_errors.append(set_error)
        return {
            "committed": True,
            "qvel_zeroed": bool(qvel_zeroed),
            "set_error": set_error,
            "problem_contacts": problem_contacts,
            "gripper_contact": bool(gripper_contact),
        }

    def _mark_invalid(self, reason: str) -> None:
        self.state = "invalid"
        if not self.invalid_reason:
            self.invalid_reason = str(reason or "translation_injection_invalid")

    def _record_trace(
        self,
        env: Any,
        *,
        low_level_step: int,
        phase: str,
        fraction: float | None,
        requested_qpos: Sequence[float] | None = None,
        set_error: float | None = None,
        problem_contacts: Sequence[Mapping[str, Any]] | None = None,
        gripper_contact: bool | None = None,
        eef_pos: Sequence[float] | None = None,
        target_center: Sequence[float] | None = None,
        distance: float | None = None,
    ) -> dict[str, Any]:
        sim = get_sim(env)
        actual_qpos = get_joint_qpos(sim, self.target_joint_name)
        actual_body_pos = body_pos(sim, self.target_body_name)
        actual_body_quat = body_quat(sim, self.target_body_name)
        row = {
            "env_step": int(low_level_step),
            "phase": str(phase),
            "fraction": None if fraction is None else float(fraction),
            "requested_qpos": None if requested_qpos is None else np.asarray(requested_qpos, dtype=np.float64).tolist(),
            "actual_qpos": actual_qpos.tolist(),
            "actual_body_pose": np.concatenate([actual_body_pos, actual_body_quat]).tolist(),
            "set_error": None if set_error is None else float(set_error),
            "problem_contacts": [] if problem_contacts is None else list(problem_contacts),
            "gripper_contact": None if gripper_contact is None else bool(gripper_contact),
            "eef_pos": None if eef_pos is None else np.asarray(eef_pos, dtype=np.float64).reshape(-1)[:3].tolist(),
            "target_center": None
            if target_center is None
            else np.asarray(target_center, dtype=np.float64).reshape(-1)[:3].tolist(),
            "eef_target_distance_m": None if distance is None else float(distance),
            "invalid": bool(self.invalid),
            "invalid_reason": self.invalid_reason,
        }
        self._qpos_trace.append(row)
        return row

    def to_dynaprobe_event(
        self,
        *,
        invalid: bool | None = None,
        invalid_reason: str | None = None,
        delta_record: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        event_invalid = bool(self.invalid if invalid is None else invalid)
        reason = self.invalid_reason or str(invalid_reason or "")
        magnitude = float(np.linalg.norm(self.delta_xy))
        t0 = self.t0_env_step
        trigger_type = "episode_start_static_endpoint" if self.initial_endpoint_static else self.trigger_type
        transition = (
            "initial_endpoint_static"
            if self.initial_endpoint_static
            else ("instant" if self.n_slide_steps == 1 else "uniform_slide")
        )
        interpolation = (
            "endpoint applied once before first policy observation"
            if self.initial_endpoint_static
            else "fraction=min(1,(env_step-t0+1)/n_slide_steps)"
        )
        return {
            "tuple": {
                "target": self.target_object_name,
                "trigger": {
                    "type": trigger_type,
                    "radius_m": self.trigger_distance_m,
                    "hit_step": t0,
                },
                "transform": {
                    "type": "target_free_joint_xy_translation",
                    "requested_delta_xy_m": self.requested_delta_xy.tolist(),
                    "requested_delta_xy_cm": (self.requested_delta_xy * 100.0).tolist(),
                    "delta_xy_m": self.delta_xy.tolist(),
                    "delta_xy_cm": (self.delta_xy * 100.0).tolist(),
                    "selected_delta": delta_record,
                    "direction_rule": self.direction_rule,
                    "n_slide_steps": int(self.n_slide_steps),
                    "discrete_interpolation": interpolation,
                    "set_mode": "kinematic_qpos_set",
                    "initial_endpoint_static": bool(self.initial_endpoint_static),
                },
                "magnitude": magnitude,
                "transition": transition,
                "visibility": "policy_observation_visible",
                "reversal": None,
                "dose": int(self.b_observation_calls),
            },
            "t0_env_step": t0,
            "restore_env_step": None,
            "occlusion_span": None,
            "dose_obs_count": int(self.b_observation_calls),
            "target_qpos_by_env_step": list(self._qpos_trace),
            "translation_injection": {
                "target_body_name": self.target_body_name,
                "target_joint_name": self.target_joint_name,
                "a_qpos": None if self.a_qpos is None else self.a_qpos.tolist(),
                "b_qpos": None if self.b_qpos is None else self.b_qpos.tolist(),
                "a_body_pos": None if self.a_body_pos is None else self.a_body_pos.tolist(),
                "b_body_pos": None if self.b_body_pos is None else self.b_body_pos.tolist(),
                "collision_check": "D2 target_problem_contacts + target_gripper_contact before commit",
                "trigger_record": None if self.trigger_record is None else jsonable(self.trigger_record.__dict__),
                "initial_endpoint_static": bool(self.initial_endpoint_static),
                "trigger_only": bool(self.trigger_only),
            },
            "validation": {
                "trigger_hit": bool(self.trigger_record is not None),
                "set_error_max": float(max(self._set_errors)) if self._set_errors else 0.0,
                "invalid": event_invalid,
                "invalid_reason": reason,
                "collision_checked": True,
            },
            "severity_level": "L1" if self.n_slide_steps == 1 else "L2",
        }

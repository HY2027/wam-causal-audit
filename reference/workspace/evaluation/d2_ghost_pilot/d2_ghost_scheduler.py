from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np


OFFICIAL_CAMERA_NAMES = ("agentview", "robot0_eye_in_hand")
OFFICIAL_OBS_KEYS = ("agentview_image", "robot0_eye_in_hand_image")


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def safe_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def name_aliases(object_name: str) -> list[str]:
    base = str(object_name).lower()
    no_instance = re.sub(r"_\d+$", "", base)
    aliases = [base, no_instance, base.replace("_", ""), no_instance.replace("_", "")]
    out: list[str] = []
    for alias in aliases:
        clean = safe_name(alias)
        if clean and clean not in out:
            out.append(clean)
    return out


def name_is_related(name: str | None, object_name: str) -> bool:
    if not name:
        return False
    clean_name = safe_name(str(name))
    return any(alias in clean_name or clean_name in alias for alias in name_aliases(object_name))


def get_sim(env):
    if hasattr(env, "sim"):
        return env.sim
    if hasattr(env, "env") and hasattr(env.env, "sim"):
        return env.env.sim
    if hasattr(env, "_env") and hasattr(env._env, "sim"):
        return env._env.sim
    raise AttributeError("Cannot find MuJoCo sim on env.")


def model_names(model, kind: str) -> list[str]:
    attr = f"{kind}_names"
    names = getattr(model, attr, None)
    if names is not None:
        return [str(x) for x in names if x]
    count_attr = {"joint": "njnt", "body": "nbody", "geom": "ngeom"}.get(kind)
    getter = getattr(model, f"{kind}_id2name", None)
    if count_attr is None or getter is None:
        return []
    out: list[str] = []
    for idx in range(int(getattr(model, count_attr, 0) or 0)):
        name = getter(idx)
        if name:
            out.append(str(name))
    return out


def joint_id(model, joint_name: str) -> int:
    if hasattr(model, "joint_name2id"):
        return int(model.joint_name2id(joint_name))
    if hasattr(model, "name2id"):
        return int(model.name2id(joint_name, "joint"))
    raise AttributeError("Cannot map joint name to id.")


def body_id(model, body_name: str) -> int:
    if hasattr(model, "body_name2id"):
        return int(model.body_name2id(body_name))
    if hasattr(model, "name2id"):
        return int(model.name2id(body_name, "body"))
    raise AttributeError("Cannot map body name to id.")


def geom_id(model, geom_name: str) -> int:
    if hasattr(model, "geom_name2id"):
        return int(model.geom_name2id(geom_name))
    if hasattr(model, "name2id"):
        return int(model.name2id(geom_name, "geom"))
    raise AttributeError("Cannot map geom name to id.")


def _addr_to_slice(addr: Any, fallback_size: int = 1) -> slice:
    if isinstance(addr, slice):
        return addr
    if isinstance(addr, tuple):
        return slice(int(addr[0]), int(addr[1]))
    start = int(addr)
    return slice(start, start + int(fallback_size))


def joint_qpos_slice(sim, joint_name: str) -> slice:
    model = sim.model
    if hasattr(model, "get_joint_qpos_addr"):
        addr = model.get_joint_qpos_addr(joint_name)
        if isinstance(addr, (tuple, slice)):
            return _addr_to_slice(addr)
        jid = joint_id(model, joint_name)
        jnt_type = np.asarray(getattr(model, "jnt_type", []))
        size = 7 if jnt_type.size and int(jnt_type[jid]) == 0 else 1
        return _addr_to_slice(addr, fallback_size=size)
    jid = joint_id(model, joint_name)
    qposadr = np.asarray(model.jnt_qposadr, dtype=np.int64)
    start = int(qposadr[jid])
    end = int(qposadr[jid + 1]) if jid + 1 < qposadr.size else int(sim.data.qpos.shape[0])
    return slice(start, end)


def joint_qvel_slice(sim, joint_name: str) -> slice | None:
    model = sim.model
    if hasattr(model, "get_joint_qvel_addr"):
        try:
            addr = model.get_joint_qvel_addr(joint_name)
            if isinstance(addr, (tuple, slice)):
                return _addr_to_slice(addr)
            jid = joint_id(model, joint_name)
            jnt_type = np.asarray(getattr(model, "jnt_type", []))
            size = 6 if jnt_type.size and int(jnt_type[jid]) == 0 else 1
            return _addr_to_slice(addr, fallback_size=size)
        except Exception:
            pass
    if hasattr(model, "jnt_dofadr"):
        jid = joint_id(model, joint_name)
        dofadr = np.asarray(model.jnt_dofadr, dtype=np.int64)
        start = int(dofadr[jid])
        end = int(dofadr[jid + 1]) if jid + 1 < dofadr.size else int(sim.data.qvel.shape[0])
        return slice(start, end)
    return None


def get_joint_qpos(sim, joint_name: str) -> np.ndarray:
    if hasattr(sim.data, "get_joint_qpos"):
        try:
            return np.asarray(sim.data.get_joint_qpos(joint_name), dtype=np.float64).reshape(-1).copy()
        except Exception:
            pass
    sl = joint_qpos_slice(sim, joint_name)
    return np.asarray(sim.data.qpos[sl], dtype=np.float64).reshape(-1).copy()


def set_joint_qpos(sim, joint_name: str, qpos: Sequence[float]) -> None:
    arr = np.asarray(qpos, dtype=np.float64).reshape(-1)
    if hasattr(sim.data, "set_joint_qpos"):
        try:
            sim.data.set_joint_qpos(joint_name, arr)
            return
        except Exception:
            pass
    sl = joint_qpos_slice(sim, joint_name)
    if sl.stop - sl.start != arr.size:
        raise ValueError(f"qpos size mismatch for {joint_name}: slice={sl}, qpos_shape={arr.shape}")
    sim.data.qpos[sl] = arr


def zero_joint_qvel(sim, joint_name: str) -> bool:
    sl = joint_qvel_slice(sim, joint_name)
    if sl is None:
        return False
    sim.data.qvel[sl] = 0.0
    return True


def set_target_qpos_and_forward(env, joint_name: str, qpos: Sequence[float]) -> bool:
    sim = get_sim(env)
    set_joint_qpos(sim, joint_name, qpos)
    qvel_zeroed = zero_joint_qvel(sim, joint_name)
    if hasattr(sim, "forward"):
        sim.forward()
    return bool(qvel_zeroed)


def body_pos(sim, body_name: str) -> np.ndarray:
    if hasattr(sim.data, "get_body_xpos"):
        try:
            return np.asarray(sim.data.get_body_xpos(body_name), dtype=np.float64).reshape(-1)[:3].copy()
        except Exception:
            pass
    bid = body_id(sim.model, body_name)
    return np.asarray(sim.data.body_xpos[bid], dtype=np.float64).reshape(-1)[:3].copy()


def body_quat(sim, body_name: str) -> np.ndarray:
    if hasattr(sim.data, "get_body_xquat"):
        try:
            return np.asarray(sim.data.get_body_xquat(body_name), dtype=np.float64).reshape(-1)[:4].copy()
        except Exception:
            pass
    bid = body_id(sim.model, body_name)
    return np.asarray(sim.data.body_xquat[bid], dtype=np.float64).reshape(-1)[:4].copy()


def get_eef_pos(env, raw_obs: Mapping[str, Any] | None = None) -> np.ndarray:
    if raw_obs is not None and "robot0_eef_pos" in raw_obs:
        return np.asarray(raw_obs["robot0_eef_pos"], dtype=np.float64).reshape(-1)[:3].copy()
    inner = getattr(env, "env", env)
    robots = getattr(inner, "robots", getattr(env, "robots", []))
    robot = robots[0] if robots else None
    if robot is not None and hasattr(robot, "_eef_xpos"):
        return np.asarray(robot._eef_xpos, dtype=np.float64).reshape(-1)[:3].copy()
    raise RuntimeError("Could not read robot0 EEF position.")


def get_contacts(env, max_contacts: int = 256) -> list[dict[str, Any]]:
    sim = get_sim(env)
    contacts: list[dict[str, Any]] = []
    ncon = int(getattr(sim.data, "ncon", 0))
    for idx in range(min(ncon, max_contacts)):
        con = sim.data.contact[idx]
        geom1 = sim.model.geom_id2name(con.geom1) if hasattr(sim.model, "geom_id2name") else str(con.geom1)
        geom2 = sim.model.geom_id2name(con.geom2) if hasattr(sim.model, "geom_id2name") else str(con.geom2)
        contacts.append({"geom1": geom1, "geom2": geom2, "dist": float(getattr(con, "dist", math.nan))})
    return contacts


def contact_has_target_gripper(contact: Mapping[str, Any], target_object_name: str) -> bool:
    names = [str(contact.get("geom1", "")), str(contact.get("geom2", ""))]
    has_target = any(name_is_related(name, target_object_name) for name in names)
    has_gripper = any(("gripper" in name.lower() or "finger" in name.lower()) for name in names)
    return bool(has_target and has_gripper)


def target_gripper_contact(env, target_object_name: str) -> bool:
    return any(contact_has_target_gripper(c, target_object_name) for c in get_contacts(env))


def render_camera(sim, camera_name: str, image_size: int) -> np.ndarray:
    try:
        img = sim.render(camera_name=camera_name, width=image_size, height=image_size, depth=False)
    except TypeError:
        img = sim.render(width=image_size, height=image_size, camera_name=camera_name, depth=False)
    if isinstance(img, tuple):
        img = img[0]
    arr = np.asarray(img)
    if arr.ndim == 3 and arr.shape[-1] > 3:
        arr = arr[..., :3]
    arr = np.flipud(arr)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def render_official_cameras(env, image_size: int = 256) -> dict[str, np.ndarray]:
    sim = get_sim(env)
    if hasattr(sim, "forward"):
        sim.forward()
    return {name: render_camera(sim, name, image_size) for name in OFFICIAL_CAMERA_NAMES}


def target_pose(env, body_name: str, joint_name: str) -> dict[str, Any]:
    sim = get_sim(env)
    return {
        "body_name": body_name,
        "body_pos": body_pos(sim, body_name),
        "body_quat": body_quat(sim, body_name),
        "joint_name": joint_name,
        "joint_qpos": get_joint_qpos(sim, joint_name),
    }


def qpos_error(a: Sequence[float], b: Sequence[float]) -> float:
    arr_a = np.asarray(a, dtype=np.float64).reshape(-1)
    arr_b = np.asarray(b, dtype=np.float64).reshape(-1)
    if arr_a.shape != arr_b.shape:
        return float("inf")
    return float(np.max(np.abs(arr_a - arr_b))) if arr_a.size else 0.0


def normalize_xy(vec: Sequence[float]) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float64).reshape(-1)
    out = np.zeros(2, dtype=np.float64)
    out[: min(2, arr.size)] = arr[: min(2, arr.size)]
    norm = float(np.linalg.norm(out))
    if norm < 1e-9 or not np.isfinite(norm):
        return np.array([1.0, 0.0], dtype=np.float64)
    return out / norm


@dataclass
class GhostEvent:
    call_index: int
    low_level_step: int | None
    phase: str
    state: str
    b_observation_calls: int
    target_qpos: np.ndarray
    qvel_zeroed: bool

    def to_json(self) -> dict[str, Any]:
        return jsonable(self.__dict__)


@dataclass
class TriggerRecord:
    low_level_step: int
    distance_m: float
    eef_pos: np.ndarray
    target_center: np.ndarray
    target_displacement_m: float
    target_gripper_contact: bool
    invalid_reason: str = ""

    def to_json(self) -> dict[str, Any]:
        return jsonable(self.__dict__)


@dataclass
class D2GhostScheduler:
    condition: str
    target_object_name: str
    target_body_name: str
    target_joint_name: str
    delta_xy: np.ndarray
    hold_observation_calls: int
    trigger_distance_m: float = 0.25
    target_motion_tol_m: float = 0.01
    state: str = "armed"
    a_qpos: np.ndarray | None = None
    b_qpos: np.ndarray | None = None
    a_body_pos: np.ndarray | None = None
    b_body_pos: np.ndarray | None = None
    trigger_record: TriggerRecord | None = None
    invalid_reason: str = ""
    b_observation_calls: int = 0
    return_step: int | None = None
    min_trigger_distance_m: float = 0.0
    require_target_directed: bool = False
    trigger_cos_threshold: float = 0.5
    _last_eef_pos: np.ndarray | None = None
    _last_target_distance_m: float | None = None
    events: list[GhostEvent] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.condition not in {"clean", "ghost_short", "ghost_long"}:
            raise ValueError(f"Unknown D2 condition: {self.condition}")
        if self.condition == "clean":
            self.hold_observation_calls = 0
        if self.hold_observation_calls < 0:
            raise ValueError("hold_observation_calls must be non-negative")
        self.delta_xy = np.asarray(self.delta_xy, dtype=np.float64).reshape(2)

    def capture_a(self, env) -> None:
        sim = get_sim(env)
        self.a_qpos = get_joint_qpos(sim, self.target_joint_name)
        if self.a_qpos.size < 7:
            raise RuntimeError(f"{self.target_joint_name} is not a free joint qpos of size 7: {self.a_qpos.shape}")
        self.b_qpos = self.a_qpos.copy()
        self.b_qpos[:2] = self.a_qpos[:2] + self.delta_xy
        self.a_body_pos = body_pos(sim, self.target_body_name)
        set_target_qpos_and_forward(env, self.target_joint_name, self.b_qpos)
        self.b_body_pos = body_pos(sim, self.target_body_name)
        set_target_qpos_and_forward(env, self.target_joint_name, self.a_qpos)

    def update_delta(self, env, delta_xy: Sequence[float]) -> None:
        a_qpos, _, _ = self._require_a()
        self.delta_xy = np.asarray(delta_xy, dtype=np.float64).reshape(2)
        self.b_qpos = a_qpos.copy()
        self.b_qpos[:2] = a_qpos[:2] + self.delta_xy
        sim = get_sim(env)
        set_target_qpos_and_forward(env, self.target_joint_name, self.b_qpos)
        self.b_body_pos = body_pos(sim, self.target_body_name)
        set_target_qpos_and_forward(env, self.target_joint_name, a_qpos)

    def _require_a(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.a_qpos is None or self.b_qpos is None or self.a_body_pos is None:
            raise RuntimeError("capture_a(env) must be called before using the scheduler.")
        return self.a_qpos, self.b_qpos, self.a_body_pos

    def observe_after_step(self, env, raw_obs: Mapping[str, Any] | None, low_level_step: int) -> None:
        if self.condition == "clean" or self.state != "armed":
            return
        a_qpos, _, a_body_pos = self._require_a()
        sim = get_sim(env)
        eef = get_eef_pos(env, raw_obs)
        target_center = body_pos(sim, self.target_body_name)
        distance = float(np.linalg.norm(eef[:3] - target_center[:3]))
        if distance >= self.trigger_distance_m or distance <= self.min_trigger_distance_m:
            self._last_eef_pos = eef.copy()
            self._last_target_distance_m = distance
            return
        if self.require_target_directed:
            if self._last_eef_pos is None or self._last_target_distance_m is None:
                self._last_eef_pos = eef.copy()
                self._last_target_distance_m = distance
                return
            movement = eef[:3] - self._last_eef_pos[:3]
            to_target = target_center[:3] - self._last_eef_pos[:3]
            mv_norm = float(np.linalg.norm(movement))
            target_norm = float(np.linalg.norm(to_target))
            if mv_norm < 1e-9 or target_norm < 1e-9:
                self._last_eef_pos = eef.copy()
                self._last_target_distance_m = distance
                return
            cos = float(np.dot(movement, to_target) / (mv_norm * target_norm))
            if cos <= self.trigger_cos_threshold or distance >= float(self._last_target_distance_m):
                self._last_eef_pos = eef.copy()
                self._last_target_distance_m = distance
                return
        self._last_eef_pos = eef.copy()
        self._last_target_distance_m = distance
        if distance >= self.trigger_distance_m:
            return
        contact = target_gripper_contact(env, self.target_object_name)
        target_displacement = float(np.linalg.norm(target_center[:3] - a_body_pos[:3]))
        qpos_now = get_joint_qpos(sim, self.target_joint_name)
        qpos_moved = qpos_error(qpos_now, a_qpos) > self.target_motion_tol_m
        invalid = ""
        if contact:
            invalid = "trigger_after_target_gripper_contact"
        elif target_displacement > self.target_motion_tol_m or qpos_moved:
            invalid = "trigger_after_target_moved"
        self.trigger_record = TriggerRecord(
            low_level_step=int(low_level_step),
            distance_m=distance,
            eef_pos=eef.copy(),
            target_center=target_center.copy(),
            target_displacement_m=target_displacement,
            target_gripper_contact=bool(contact),
            invalid_reason=invalid,
        )
        if invalid:
            self.state = "invalid"
            self.invalid_reason = invalid
        else:
            self.state = "pending_apply_b"

    def before_policy_observation_call(self, env, call_index: int, low_level_step: int | None = None) -> GhostEvent:
        a_qpos, b_qpos, _ = self._require_a()
        qvel_zeroed = False
        if self.condition == "clean":
            phase = "A_clean"
            target = a_qpos
        elif self.state == "pending_apply_b":
            qvel_zeroed = set_target_qpos_and_forward(env, self.target_joint_name, b_qpos)
            self.state = "holding_b"
            self.b_observation_calls = 1
            phase = "B_visible"
            target = b_qpos
        elif self.state == "holding_b":
            if self.b_observation_calls < self.hold_observation_calls:
                qvel_zeroed = set_target_qpos_and_forward(env, self.target_joint_name, b_qpos)
                self.b_observation_calls += 1
                phase = "B_visible"
                target = b_qpos
            else:
                qvel_zeroed = set_target_qpos_and_forward(env, self.target_joint_name, a_qpos)
                self.state = "restored_a"
                self.return_step = None if low_level_step is None else int(low_level_step)
                phase = "A_restored"
                target = a_qpos
        elif self.state == "restored_a":
            qvel_zeroed = set_target_qpos_and_forward(env, self.target_joint_name, a_qpos)
            phase = "A_after_restore"
            target = a_qpos
        elif self.state == "armed":
            phase = "A_waiting_for_trigger"
            target = get_joint_qpos(get_sim(env), self.target_joint_name)
        elif self.state == "invalid":
            phase = "invalid"
            target = get_joint_qpos(get_sim(env), self.target_joint_name)
        else:
            phase = self.state
            target = get_joint_qpos(get_sim(env), self.target_joint_name)
        event = GhostEvent(
            call_index=int(call_index),
            low_level_step=None if low_level_step is None else int(low_level_step),
            phase=phase,
            state=self.state,
            b_observation_calls=int(self.b_observation_calls),
            target_qpos=np.asarray(target, dtype=np.float64).copy(),
            qvel_zeroed=bool(qvel_zeroed),
        )
        self.events.append(event)
        return event

    def restore_a(self, env) -> float:
        a_qpos, _, _ = self._require_a()
        set_target_qpos_and_forward(env, self.target_joint_name, a_qpos)
        return qpos_error(get_joint_qpos(get_sim(env), self.target_joint_name), a_qpos)

    def no_trigger(self) -> bool:
        return self.condition != "clean" and self.trigger_record is None and self.state == "armed"

    def to_json(self) -> dict[str, Any]:
        return {
            "condition": self.condition,
            "target_object_name": self.target_object_name,
            "target_body_name": self.target_body_name,
            "target_joint_name": self.target_joint_name,
            "delta_xy": jsonable(self.delta_xy),
            "hold_observation_calls": self.hold_observation_calls,
            "trigger_distance_m": self.trigger_distance_m,
            "min_trigger_distance_m": self.min_trigger_distance_m,
            "require_target_directed": self.require_target_directed,
            "trigger_cos_threshold": self.trigger_cos_threshold,
            "target_motion_tol_m": self.target_motion_tol_m,
            "state": self.state,
            "a_qpos": jsonable(self.a_qpos),
            "b_qpos": jsonable(self.b_qpos),
            "a_body_pos": jsonable(self.a_body_pos),
            "b_body_pos": jsonable(self.b_body_pos),
            "trigger_record": None if self.trigger_record is None else self.trigger_record.to_json(),
            "invalid_reason": self.invalid_reason,
            "b_observation_calls": self.b_observation_calls,
            "return_step": self.return_step,
            "events": [event.to_json() for event in self.events],
        }

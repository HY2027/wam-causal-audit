"""Shared restore_v2 utilities for the isolated Joint/IDM confirmation round."""
from __future__ import annotations
from wam_causal_audit.paths import resolve as _release_path

import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

WORK = Path(__file__).resolve().parent
ROOT = Path(_release_path('@DATA@/wam_factor_routing_v5/joint_idm_compute_confirmatory_v1_20260909'))
JOINT = ROOT / "joint_new_trajectory_confirmatory"
IDM = ROOT / "idm_world_compute_development_validation"


def module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None: raise RuntimeError(f"MODULE_UNAVAILABLE:{path}")
    value = importlib.util.module_from_spec(spec); sys.modules[name] = value; spec.loader.exec_module(value); return value


OLD = module("confirmatory_old_protocol", Path(_release_path('@WORKSPACE@/counterfactual_empty_location_work/protocol.py')))
D = module("confirmatory_closedloop_core", WORK / "run_c_closedloop.py")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(16 * 1024 * 1024), b""): h.update(block)
    return h.hexdigest()


def raw_hash(value: Any) -> str:
    a = np.ascontiguousarray(np.asarray(value)); return hashlib.sha256(memoryview(a.view(np.uint8))).hexdigest()


def observation_hash(values: Mapping[str, Any]) -> str:
    h = hashlib.sha256()
    for key in sorted(values):
        a = np.ascontiguousarray(np.asarray(values[key])); h.update(key.encode()); h.update(str(a.dtype).encode())
        h.update(str(tuple(a.shape)).encode()); h.update(memoryview(a.view(np.uint8)))
    return h.hexdigest()


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as z: return {k: z[k].copy() for k in z.files}


def registry(kind: str) -> list[dict[str, str]]:
    if kind == "joint":
        v3 = JOINT / "joint_new_trajectory_registry_restore_v2_execution_v3.csv"
        amended = JOINT / "joint_new_trajectory_registry_restore_v2_execution_v2.csv"
        path = v3 if v3.exists() else (amended if amended.exists() else JOINT / "joint_new_trajectory_registry.csv")
    else:
        amended = IDM / "idm_development_validation_registry_restore_v2_execution_v2.csv"
        path = amended if amended.exists() else IDM / "idm_development_validation_registry.csv"
    return list(csv.DictReader(path.open()))


def set_protected(sim: Any, values: Mapping[str, np.ndarray]):
    sim.set_state_from_flattened(np.asarray(values["flat"]).copy())
    for name in ("ctrl", "mocap_pos", "mocap_quat", "qfrc_applied", "xfrc_applied", "qacc_warmstart", "plugin_state"):
        if name in values and hasattr(sim.data, name): getattr(sim.data, name)[:] = values[name]


def protected(sim: Any) -> dict[str, np.ndarray]:
    out = {"flat": np.asarray(sim.get_state().flatten()).copy(), "time": np.asarray([sim.data.time])}
    for name in ("qpos", "qvel", "act", "ctrl", "mocap_pos", "mocap_quat", "qfrc_applied", "xfrc_applied", "qacc_warmstart", "plugin_state"):
        if hasattr(sim.data, name): out[name] = np.asarray(getattr(sim.data, name)).copy()
    return out


def restore(row: Mapping[str, Any]):
    env, _, _ = OLD.make_env(int(row["task_id"]), int(row["init_state_id"])); sim = OLD.fgl.get_sim(env)
    state = load_npz(row["state_path"]); set_protected(sim, state)
    if "gripper_current_action" not in state:
        env.close(); raise RuntimeError(f"RESTORE_V2_GRIPPER_FIELD_MISSING:{row['candidate_id']}")
    gripper = env.env.robots[0].gripper; value = np.asarray(state["gripper_current_action"], dtype=np.float64)
    if value.shape != np.asarray(gripper.current_action).shape:
        env.close(); raise RuntimeError(f"RESTORE_V2_GRIPPER_SHAPE:{row['candidate_id']}:{value.shape}")
    gripper.current_action = value.copy()
    controller = env.env.robots[0].controller
    execution_fields = {
        "controller_goal_ori": "goal_ori", "controller_goal_pos": "goal_pos",
        "controller_ori_ref": "ori_ref", "controller_relative_ori": "relative_ori",
    }
    for source, target in execution_fields.items():
        if source in state:
            setattr(controller, target, np.asarray(state[source]).copy())
    if "controller_new_update" in state:
        controller.new_update = bool(np.asarray(state["controller_new_update"]).reshape(-1)[0])
    if "env_timestep" in state: env.env.timestep = int(np.asarray(state["env_timestep"]).reshape(-1)[0])
    if "env_cur_time" in state: env.env.cur_time = float(np.asarray(state["env_cur_time"]).reshape(-1)[0])
    if "env_done" in state: env.env.done = bool(np.asarray(state["env_done"]).reshape(-1)[0])
    sim.forward(); obs = dict(env.env._get_observations(force_update=True))
    # qacc_warmstart is a solver input at the execution boundary. mj_forward
    # recomputes it, so restore the frozen continuous-boundary value last.
    if "qacc_warmstart" in state: sim.data.qacc_warmstart[:] = state["qacc_warmstart"]
    frozen = load_npz(row["observation_path"])
    if raw_hash(state["flat"]) != row["flat_state_sha256"]:
        env.close(); raise RuntimeError(f"RESTORE_V2_STATE_HASH:{row['candidate_id']}")
    if observation_hash(obs) != row["observation_content_sha256"] or observation_hash(frozen) != row["observation_content_sha256"]:
        env.close(); raise RuntimeError(f"RESTORE_V2_OBSERVATION_HASH:{row['candidate_id']}")
    if not np.array_equal(np.asarray(gripper.current_action), value):
        env.close(); raise RuntimeError(f"RESTORE_V2_GRIPPER_VALUE:{row['candidate_id']}")
    return env, obs, state


def seed(row: Mapping[str, Any]) -> int:
    return int(row["policy_seed"])

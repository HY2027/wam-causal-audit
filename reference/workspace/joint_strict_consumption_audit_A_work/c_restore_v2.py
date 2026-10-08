#!/usr/bin/env python3
"""Explicit minimal restore_v2 for Joint-WAM Experiment C posthoc reassessment."""
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
HIST = Path(_release_path('@DATA@/wam_factor_routing_v5/joint_experiment_B_distance_v1/C_final_test_v1'))
OUT = HIST / "restore_v2_posthoc_reassessment_20260909"
SIDECARS = OUT / "state_sidecars"


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None: raise RuntimeError(f"MODULE_UNAVAILABLE:{path}")
    value = importlib.util.module_from_spec(spec); sys.modules[name] = value
    spec.loader.exec_module(value); return value


D = _module("restore_v2_closedloop", WORK / "run_c_closedloop.py")
OLD = _module("restore_v2_old_protocol", Path(_release_path('@WORKSPACE@/counterfactual_empty_location_work/protocol.py')))


SCHEMA_VERSION = "JOINT_C_RESTORE_V2_GRIPPER_CURRENT_ACTION_SIDECAR_V1"


def array_hash(value: Any) -> str:
    a = np.ascontiguousarray(np.asarray(value))
    h = hashlib.sha256(); h.update(str(a.dtype).encode()); h.update(str(tuple(a.shape)).encode())
    h.update(memoryview(a.view(np.uint8))); return h.hexdigest()


def raw_array_hash(value: Any) -> str:
    a = np.ascontiguousarray(np.asarray(value)); return hashlib.sha256(memoryview(a.view(np.uint8))).hexdigest()


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(16 * 1024 * 1024), b""): h.update(block)
    return h.hexdigest()


def observation_hash(values: Mapping[str, Any]) -> str:
    # Authoritative historical implementation imported by run_experiment_a.
    return D.A.observation_sha256(values)


def sidecar_path(candidate_id: str) -> Path:
    return SIDECARS / candidate_id / "restore_v2_execution_state.npz"


def metadata_path(candidate_id: str) -> Path:
    return SIDECARS / candidate_id / "restore_v2_execution_state.json"


def load_sidecar(candidate_id: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    path = sidecar_path(candidate_id); meta_path = metadata_path(candidate_id)
    if not path.is_file() or not meta_path.is_file():
        raise RuntimeError(f"RESTORE_V2_REQUIRED_SIDECAR_MISSING:{candidate_id}")
    with np.load(path, allow_pickle=False) as z: state = {k: z[k].copy() for k in z.files}
    meta = json.loads(meta_path.read_text())
    if meta.get("schema_version") != SCHEMA_VERSION: raise RuntimeError(f"RESTORE_V2_SCHEMA_MISMATCH:{candidate_id}")
    if file_hash(path) != meta.get("sidecar_file_sha256"): raise RuntimeError(f"RESTORE_V2_SIDECAR_HASH_MISMATCH:{candidate_id}")
    if array_hash(state["gripper_current_action"]) != meta.get("gripper_current_action_sha256"):
        raise RuntimeError(f"RESTORE_V2_GRIPPER_HASH_MISMATCH:{candidate_id}")
    return state, meta


def restore(row: Mapping[str, Any]):
    """Restore only frozen MuJoCo fields plus explicit gripper.current_action."""
    candidate_id = str(row["candidate_id"])
    env, _, _ = OLD.make_env(int(row["task_id"]), int(row["init_state_id"]))
    sim = OLD.fgl.get_sim(env)
    snap = D.load_npz(HIST / "synced_recipients" / candidate_id / "recipient_state.npz")
    execution, meta = load_sidecar(candidate_id)
    if raw_array_hash(snap["flat"]) != meta["historical_sync_flat_raw_sha256"]:
        env.close(); raise RuntimeError(f"RESTORE_V2_RECIPIENT_STATE_HASH_MISMATCH:{candidate_id}")
    D.set_protected(sim, snap)
    # Minimal explicit addition. Never infer a missing value from qpos and never
    # copy an object/__dict__.
    gripper = env.env.robots[0].gripper
    expected_shape = np.asarray(gripper.current_action).shape
    value = np.asarray(execution["gripper_current_action"], dtype=np.float64)
    if value.shape != expected_shape:
        env.close(); raise RuntimeError(f"RESTORE_V2_GRIPPER_SHAPE_MISMATCH:{candidate_id}:{value.shape}:{expected_shape}")
    gripper.current_action = value.copy()
    sim.forward()
    obs = dict(env.env._get_observations(force_update=True))
    if observation_hash(obs) != meta["recipient_observation_hash"]:
        env.close(); raise RuntimeError(f"RESTORE_V2_OBSERVATION_HASH_MISMATCH:{candidate_id}")
    return env, obs, snap, execution, meta


def registry() -> list[dict[str, str]]:
    return [r for r in csv.DictReader((HIST / "final_test_registry.csv").open())
            if r.get("selection_status", "").startswith("SELECTED")]

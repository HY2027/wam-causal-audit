from __future__ import annotations
from wam_causal_audit.paths import resolve as _release_path

import csv
import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import imageio.v2 as imageio
import numpy as np

from d2_ghost_scheduler import (
    D2GhostScheduler,
    body_pos,
    body_quat,
    get_contacts,
    get_eef_pos,
    get_joint_qpos,
    get_sim,
    jsonable,
    model_names,
    name_is_related,
    qpos_error,
    set_target_qpos_and_forward,
    target_gripper_contact,
)


ROOT = Path(_release_path('@WORKSPACE@'))
RESULT_ROOT = ROOT / "results" / "d2_ghost_pilot"
LIBERO_ROOT = ROOT / "LIBERO"

STEP_LOG_DIR = RESULT_ROOT / "d2_step_logs"
VIDEO_DIR = RESULT_ROOT / "videos"
SUMMARY_CSV = RESULT_ROOT / "d2_rollout_summary.csv"

CONDITIONS = ("clean", "ghost_short", "ghost_long")
HOLD_CALLS = {"clean": 0, "ghost_short": 1, "ghost_long": 4}

SUMMARY_FIELDS = [
    "model",
    "task_id",
    "seed",
    "condition",
    "success",
    "triggered",
    "invalid",
    "invalid_reason",
    "no_trigger",
    "trigger_step",
    "return_step",
    "A",
    "B",
    "delta_xy",
    "num_policy_calls_B_visible",
    "target_pose_restore_error",
    "official_config_hash",
    "video_path",
    "step_log_path",
]


def rollout_key(model: str, task_id: int, seed: int, condition: str) -> tuple[str, int, int, str]:
    return (str(model), int(task_id), int(seed), str(condition))


def completed_rollout_keys(require_files: bool = True) -> set[tuple[str, int, int, str]]:
    done: set[tuple[str, int, int, str]] = set()
    if not SUMMARY_CSV.exists():
        return done
    with SUMMARY_CSV.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                key = rollout_key(row["model"], int(row["task_id"]), int(row["seed"]), row["condition"])
            except Exception:
                continue
            if require_files:
                step_log = Path(row.get("step_log_path") or "")
                video = Path(row.get("video_path") or "")
                if not step_log.exists() or not video.exists():
                    continue
            done.add(key)
    return done


def prepare_rollout_files(step_log_path: Path, video_path: Path, overwrite: bool = False) -> None:
    step_log_path.parent.mkdir(parents=True, exist_ok=True)
    video_path.parent.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for path in (step_log_path, video_path):
            if path.exists():
                path.unlink()


@dataclass(frozen=True)
class D2TaskConfig:
    task_id: int
    target_object_name: str
    target_body_name: str
    target_joint_name: str
    target_alias: str
    preferred_delta_xy: tuple[float, float]
    requested_delta_m: float
    avoid_body_names: tuple[str, ...] = ()


TASK_CONFIGS = {
    5: D2TaskConfig(
        task_id=5,
        target_object_name="black_book_1",
        target_body_name="black_book_1_main",
        target_joint_name="black_book_1_joint0",
        target_alias="book",
        preferred_delta_xy=(-0.02796627713949291, -0.06417076704354623),
        requested_delta_m=0.07,
    ),
    9: D2TaskConfig(
        task_id=9,
        target_object_name="white_yellow_mug_1",
        target_body_name="white_yellow_mug_1_main",
        target_joint_name="white_yellow_mug_1_joint0",
        target_alias="yellow and white mug",
        preferred_delta_xy=(0.05331091472283877, 0.013525766943660257),
        requested_delta_m=0.055,
        avoid_body_names=("porcelain_mug_1_main",),
    ),
}


def add_libero_path() -> None:
    value = str(LIBERO_ROOT.resolve())
    if value not in sys_path():
        sys_path().insert(0, value)


def sys_path() -> list[str]:
    import sys

    return sys.path


def safe_name(value: str, max_len: int = 128) -> str:
    out = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    return out[:max_len] or "none"


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(jsonable(row), ensure_ascii=False, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def append_summary(row: Mapping[str, Any]) -> None:
    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    exists = SUMMARY_CSV.exists()
    with SUMMARY_CSV.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({field: json.dumps(jsonable(row.get(field)), ensure_ascii=False) if field in {"A", "B", "delta_xy"} else row.get(field, "") for field in SUMMARY_FIELDS})
        f.flush()
        os.fsync(f.fileno())


def write_video(path: Path, frames: Sequence[np.ndarray], fps: int = 24) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        return
    imageio.mimsave(path, [np.asarray(frame, dtype=np.uint8) for frame in frames], fps=fps)


def stack_obs_frames(raw_obs: Mapping[str, Any], flip_lr: bool = False) -> np.ndarray:
    agent = np.ascontiguousarray(np.asarray(raw_obs["agentview_image"])[::-1])
    wrist = np.ascontiguousarray(np.asarray(raw_obs["robot0_eye_in_hand_image"])[::-1])
    if flip_lr:
        agent = np.ascontiguousarray(agent[:, ::-1])
        wrist = np.ascontiguousarray(wrist[:, ::-1])
    return np.hstack([agent[..., :3], wrist[..., :3]]).astype(np.uint8)


def current_raw_obs(env) -> dict[str, Any]:
    sim = get_sim(env)
    if hasattr(sim, "forward"):
        sim.forward()
    try:
        if hasattr(env, "_post_process"):
            env._post_process()
        if hasattr(env, "_update_observables"):
            env._update_observables(force=True)
        inner = getattr(env, "env", env)
        if hasattr(inner, "_get_observations"):
            return dict(inner._get_observations())
        if hasattr(env, "_get_observations"):
            return dict(env._get_observations())
    except Exception as exc:
        raise RuntimeError(f"Could not regenerate current raw observation after ghost pose update: {exc!r}") from exc
    try:
        flat_state = np.asarray(sim.get_state().flatten(), dtype=np.float64)
        if hasattr(env, "regenerate_obs_from_state"):
            return dict(env.regenerate_obs_from_state(flat_state))
    except Exception:
        pass
    raise RuntimeError("Could not regenerate current raw observation after ghost pose update.")


def lingbot_obs(raw_obs: Mapping[str, Any]) -> dict[str, np.ndarray]:
    return {
        "observation.images.agentview_rgb": np.ascontiguousarray(np.asarray(raw_obs["agentview_image"])[::-1]),
        "observation.images.eye_in_hand_rgb": np.ascontiguousarray(np.asarray(raw_obs["robot0_eye_in_hand_image"])[::-1]),
    }


def load_stage1_success_seeds(model: str, task_id: int, limit: int = 10) -> list[int]:
    path = RESULT_ROOT / "baseline_official_clean.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing Stage 1 baseline CSV: {path}")
    seeds: list[int] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("model") != model or int(row.get("task_id", -1)) != int(task_id):
                continue
            if str(row.get("success", "")).lower() == "true":
                seeds.append(int(row["seed"]))
            if len(seeds) >= limit:
                break
    if len(seeds) < limit:
        raise RuntimeError(f"Only found {len(seeds)} Stage 1 success seeds for {model} task {task_id}.")
    return seeds


def official_config_hash(payload: Mapping[str, Any], files: Sequence[Path] = ()) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(jsonable(payload), sort_keys=True).encode("utf-8"))
    for path in files:
        digest.update(str(path).encode("utf-8"))
        if path.exists():
            digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def make_scheduler(task: D2TaskConfig, condition: str) -> D2GhostScheduler:
    return D2GhostScheduler(
        condition=condition,
        target_object_name=task.target_object_name,
        target_body_name=task.target_body_name,
        target_joint_name=task.target_joint_name,
        delta_xy=np.asarray(task.preferred_delta_xy, dtype=np.float64),
        hold_observation_calls=HOLD_CALLS[condition],
    )


def target_problem_contacts(env, target_object_name: str) -> list[dict[str, Any]]:
    allowed = ("table", "floor", "ground", "arena", "workspace")
    problems: list[dict[str, Any]] = []
    for contact in get_contacts(env):
        names = [str(contact.get("geom1", "")), str(contact.get("geom2", ""))]
        if not any(name_is_related(name, target_object_name) for name in names):
            continue
        if any(any(token in name.lower() for token in allowed) for name in names):
            continue
        if float(contact.get("dist", 0.0)) < -0.002:
            problems.append(dict(contact))
    return problems


def avoid_body_distances(env, task: D2TaskConfig) -> dict[str, float]:
    sim = get_sim(env)
    target = body_pos(sim, task.target_body_name)
    out: dict[str, float] = {}
    for name in task.avoid_body_names:
        try:
            out[name] = float(np.linalg.norm(target - body_pos(sim, name)))
        except Exception:
            out[name] = float("nan")
    if task.task_id == 9:
        for body_name in model_names(sim.model, "body"):
            if "microwave_1" not in body_name:
                continue
            try:
                out[f"microwave_body::{body_name}"] = float(np.linalg.norm(target - body_pos(sim, body_name)))
            except Exception:
                pass
    return out


def min_finite(values: Mapping[str, float], default: float = 999.0) -> float:
    finite = [float(v) for v in values.values() if np.isfinite(float(v))]
    return min(finite) if finite else default


def candidate_deltas(env, task: D2TaskConfig, raw_obs: Mapping[str, Any] | None) -> list[tuple[str, np.ndarray]]:
    preferred = np.asarray(task.preferred_delta_xy, dtype=np.float64).reshape(2)
    sim = get_sim(env)
    target_xy = body_pos(sim, task.target_body_name)[:2]
    eef_xy = get_eef_pos(env, raw_obs)[:2]
    away_eef = target_xy - eef_xy
    if np.linalg.norm(away_eef) < 1e-9:
        away_eef = np.array([1.0, 0.0], dtype=np.float64)
    away_eef = away_eef / np.linalg.norm(away_eef)
    perp = np.array([-away_eef[1], away_eef[0]], dtype=np.float64)
    directions: list[tuple[str, np.ndarray]] = [
        ("sanity_preferred", preferred / max(np.linalg.norm(preferred), 1e-9)),
        ("away_from_eef", away_eef),
        ("perp_left", perp),
        ("perp_right", -perp),
        ("toward_eef", -away_eef),
        ("+x", np.array([1.0, 0.0])),
        ("-x", np.array([-1.0, 0.0])),
        ("+y", np.array([0.0, 1.0])),
        ("-y", np.array([0.0, -1.0])),
    ]
    if task.task_id == 9:
        for avoid_name in task.avoid_body_names:
            try:
                avoid_xy = body_pos(sim, avoid_name)[:2]
            except Exception:
                continue
            away = target_xy - avoid_xy
            if np.linalg.norm(away) > 1e-9:
                directions.insert(1, (f"away_from_{avoid_name}", away / np.linalg.norm(away)))
    out: list[tuple[str, np.ndarray]] = []
    seen: set[tuple[int, int]] = set()
    for name, direction in directions:
        norm = float(np.linalg.norm(direction))
        if norm < 1e-9:
            continue
        delta = np.asarray(direction[:2], dtype=np.float64) / norm * float(task.requested_delta_m)
        key = (int(round(delta[0] * 1000)), int(round(delta[1] * 1000)))
        if key in seen:
            continue
        seen.add(key)
        out.append((name, delta))
    return out


def choose_delta_for_episode(env, task: D2TaskConfig, scheduler: D2GhostScheduler, raw_obs: Mapping[str, Any] | None) -> dict[str, Any]:
    if scheduler.a_qpos is None:
        raise RuntimeError("scheduler.capture_a(env) must run before choose_delta_for_episode")
    a_qpos = scheduler.a_qpos.copy()
    attempts: list[dict[str, Any]] = []
    for direction_name, delta_xy in candidate_deltas(env, task, raw_obs):
        set_target_qpos_and_forward(env, task.target_joint_name, a_qpos)
        b_qpos = a_qpos.copy()
        b_qpos[:2] = a_qpos[:2] + delta_xy
        set_target_qpos_and_forward(env, task.target_joint_name, b_qpos)
        actual_delta = get_joint_qpos(get_sim(env), task.target_joint_name)[:2] - a_qpos[:2]
        actual_delta_norm = float(np.linalg.norm(actual_delta))
        problem_contacts = target_problem_contacts(env, task.target_object_name)
        gripper_contact = target_gripper_contact(env, task.target_object_name)
        avoid_dist = avoid_body_distances(env, task)
        task09_clear = True
        if task.task_id == 9:
            other_mug_dist = avoid_dist.get("porcelain_mug_1_main", 999.0)
            microwave_dists = {k: v for k, v in avoid_dist.items() if k.startswith("microwave_body::")}
            task09_clear = bool(other_mug_dist > 0.10 and min_finite(microwave_dists, default=999.0) > 0.06)
        valid = bool(
            0.8 * task.requested_delta_m <= actual_delta_norm <= 1.2 * task.requested_delta_m
            and not problem_contacts
            and not gripper_contact
            and task09_clear
        )
        attempt = {
            "direction_name": direction_name,
            "delta_xy": actual_delta.copy(),
            "requested_delta_xy": delta_xy.copy(),
            "actual_delta_norm": actual_delta_norm,
            "problem_contacts": problem_contacts,
            "target_gripper_contact": bool(gripper_contact),
            "avoid_body_distances": avoid_dist,
            "task09_clearance_ok": task09_clear,
            "valid": valid,
        }
        attempts.append(attempt)
        if valid:
            set_target_qpos_and_forward(env, task.target_joint_name, a_qpos)
            scheduler.update_delta(env, actual_delta)
            return {**attempt, "attempts": attempts}
    set_target_qpos_and_forward(env, task.target_joint_name, a_qpos)
    best = attempts[-1] if attempts else {}
    raise RuntimeError(f"No valid D2 delta for task {task.task_id}: {json.dumps(jsonable(best), indent=2)}")


def eef_quat(raw_obs: Mapping[str, Any] | None) -> list[float] | None:
    if raw_obs is not None and "robot0_eef_quat" in raw_obs:
        return np.asarray(raw_obs["robot0_eef_quat"], dtype=np.float64).reshape(-1)[:4].tolist()
    return None


def gripper_state(raw_obs: Mapping[str, Any] | None, action: Sequence[float] | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if raw_obs is not None and "robot0_gripper_qpos" in raw_obs:
        out["qpos"] = np.asarray(raw_obs["robot0_gripper_qpos"], dtype=np.float64).reshape(-1).tolist()
    if action is not None:
        arr = np.asarray(action, dtype=np.float64).reshape(-1)
        if arr.size:
            out["action"] = float(arr[-1])
    return out


def current_target_pos(env, task: D2TaskConfig) -> np.ndarray:
    return body_pos(get_sim(env), task.target_body_name)


def pose_summary(env, task: D2TaskConfig, scheduler: D2GhostScheduler) -> dict[str, Any]:
    a_qpos = scheduler.a_qpos
    b_qpos = scheduler.b_qpos
    current = get_joint_qpos(get_sim(env), task.target_joint_name)
    return {
        "A": None
        if a_qpos is None
        else {
            "qpos": a_qpos.copy(),
            "body_pos": scheduler.a_body_pos.copy() if scheduler.a_body_pos is not None else None,
        },
        "B": None
        if b_qpos is None
        else {
            "qpos": b_qpos.copy(),
            "body_pos": scheduler.b_body_pos.copy() if scheduler.b_body_pos is not None else None,
        },
        "target_pose_restore_error": None if a_qpos is None else qpos_error(current, a_qpos),
        "target_body_pos": current_target_pos(env, task),
        "target_body_quat": body_quat(get_sim(env), task.target_body_name),
    }

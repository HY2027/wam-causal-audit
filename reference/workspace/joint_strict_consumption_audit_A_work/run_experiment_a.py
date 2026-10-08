#!/usr/bin/env python3
from __future__ import annotations
from wam_causal_audit.paths import resolve as _release_path

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from PIL import Image


WORK = Path(__file__).resolve().parent
BADWAM = Path(_release_path('@DATA@/BadWAM'))
G1 = Path(_release_path('@DATA@/wam_control_state_v3/group1_factor_phase'))
G2 = Path(_release_path('@DATA@/wam_factor_routing_v5/group2/joint_current_future'))
REGISTRY = Path(_release_path('@WORKSPACE@/step0_tensor_inventory_work/step0_1_v2/locus_registry_v2.json'))
OUT_DEFAULT = Path(_release_path('@DATA@/wam_factor_routing_v5/joint_experiment_A_strict_consumer_v1'))

for candidate in (
    WORK,
    Path(_release_path('@WORKSPACE@/step1_step2_51locus_work')),
    Path(_release_path('@WORKSPACE@/week1_audit_work')),
    Path(_release_path('@WORKSPACE@/experiments/wam_control_state_v3')),
    Path(_release_path('@WORKSPACE@/badwam_joint_causal_work')),
    Path(_release_path('@WORKSPACE@/badwam_idm_causal_audit_work')),
    Path(_release_path('@WORKSPACE@/badwam_idm_same_frame_goal_work')),
    BADWAM,
    BADWAM / "src",
    BADWAM / "experiments/libero",
    BADWAM / "experiments/first_grasp_lock",
    Path(_release_path('@WORKSPACE@/LIBERO')),
    Path(_release_path('@WORKSPACE@/evaluation/d2_ghost_pilot')),
):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from capture import make_capture  # noqa: E402
from protocol import load_npz, observation_sha256, tensor_sha256  # noqa: E402
from run_group0 import model_weight_hash  # noqa: E402
from run_joint_smoke_pair import QKVController, controlled_infer  # noqa: E402


PROTOCOL_VERSION = "joint-experiment-A-strict-consumer-v1"
VISUALIZATION_AMENDMENT_VERSION = "joint-experiment-A-visualization-amendment-v1"
FACTOR = "F1_ROBOT_RADIAL_PROGRESS"
PHASE = "PREGRASP"
CONTROL = "STANDARD"
D_STAR_CM = 4.0
DOSES_CM = (1.0, 2.0, 4.0)
TASKS = tuple(range(5))
STATES = tuple(range(10))
CURRENT_GROUP = 0
FUTURE_GROUPS = (1, 2)
EPS = 1e-12


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(jsonable(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows for {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def tensor_collection_hash(values: list[torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(tensor_sha256(value).encode())
    digest.update(str(len(values)).encode())
    return digest.hexdigest()


def array_hash(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(tuple(array.shape)).encode())
    digest.update(memoryview(array.view(np.uint8)))
    return digest.hexdigest()


def prepared_hashes(prepared: Mapping[str, Any]) -> dict[str, str]:
    return {
        "image": tensor_sha256(prepared["image"]),
        "proprio": tensor_sha256(prepared["proprio"]),
        "context": tensor_sha256(prepared["context"]),
        "context_mask": tensor_sha256(prepared["context_mask"]),
        "prompt": hashlib.sha256(str(prepared["prompt"]).encode()).hexdigest(),
    }


class VideoKVController:
    """Capture or jointly clamp both temporal sources at the real K/V consumer edge.

    `_build_expert_attention_io` returns K after RoPE and V after projection.  Its
    result is concatenated directly with action K/V before mixed attention, so
    replacement here occurs after positional encoding and before consumption.
    """

    def __init__(self, model: Any) -> None:
        self.model = model
        self.original = model.mot._build_expert_attention_io
        self.block_map = {id(block): ("video", i) for i, block in enumerate(model.video_expert.blocks)}
        self.block_map.update({id(block): ("action", i) for i, block in enumerate(model.action_expert.blocks)})
        self.step = -1
        self.capture = False
        self.read_only = False
        self.cache: dict[tuple[int, int], dict[str, torch.Tensor]] = {}
        self.current_cache: Mapping[tuple[int, int], Mapping[str, torch.Tensor]] | None = None
        self.future_cache: Mapping[tuple[int, int], Mapping[str, torch.Tensor]] | None = None
        self.tokens_per_group = 0
        self.video_calls = 0
        self.action_calls = 0
        self.injected_calls = 0
        self.current_exact = 0
        self.future_exact = 0
        self.max_current_error = 0.0
        self.max_future_error = 0.0
        self.first_event: dict[str, Any] | None = None
        self.last_event: dict[str, Any] | None = None

    def install(self) -> None:
        def wrapped(expert: Any, block: Any, *args: Any, **kwargs: Any):
            result = list(self.original(expert, block, *args, **kwargs))
            modality, layer = self.block_map[id(block)]
            if modality == "action":
                self.action_calls += 1
                return tuple(result)
            self.video_calls += 1
            key = (self.step, layer)
            if self.capture:
                self.cache[key] = {"k": result[1].detach().clone(), "v": result[2].detach().clone()}
            if self.current_cache is None and self.future_cache is None:
                return tuple(result)
            if self.current_cache is None or self.future_cache is None:
                raise AssertionError("Strict clamp requires both current and future caches")
            if result[1].ndim != 3 or result[1].shape[1] % 3:
                raise AssertionError(f"Unexpected Joint video K/V shape {tuple(result[1].shape)}")
            tpg = result[1].shape[1] // 3
            self.tokens_per_group = int(tpg)
            for component, slot in (("k", 1), ("v", 2)):
                current = self.current_cache[key][component].to(result[slot])[:, :tpg]
                future = self.future_cache[key][component].to(result[slot])[:, tpg:]
                result[slot] = torch.cat((current, future), dim=1)
                current_error = float(torch.max(torch.abs(result[slot][:, :tpg] - current)).item())
                future_error = float(torch.max(torch.abs(result[slot][:, tpg:] - future)).item())
                self.max_current_error = max(self.max_current_error, current_error)
                self.max_future_error = max(self.max_future_error, future_error)
                self.current_exact += int(torch.equal(result[slot][:, :tpg], current))
                self.future_exact += int(torch.equal(result[slot][:, tpg:], future))
            self.injected_calls += 1
            event = {
                "step": self.step,
                "layer": layer,
                "location": "MoT._build_expert_attention_io:return -> torch.cat(video,action) -> mixed_attention",
                "k_is_post_rope": True,
                "v_is_post_projection": True,
                "tokens_per_temporal_group": tpg,
            }
            if self.first_event is None:
                self.first_event = event
            self.last_event = event
            return tuple(result)

        self.model.mot._build_expert_attention_io = wrapped

    def uninstall(self) -> None:
        self.model.mot._build_expert_attention_io = self.original

    def summary(self, expected_steps: int, expected_layers: int) -> dict[str, Any]:
        expected_video_calls = expected_steps * expected_layers
        return {
            "video_calls": self.video_calls,
            "action_calls": self.action_calls,
            "injected_video_calls": self.injected_calls,
            "expected_video_calls": expected_video_calls,
            "hook_reached_all_video_sites": self.video_calls == expected_video_calls,
            "hook_reached_all_action_sites": self.action_calls == expected_video_calls,
            "strict_injection_reached_all_video_sites": self.injected_calls in (0, expected_video_calls),
            "current_component_exact_count": self.current_exact,
            "future_component_exact_count": self.future_exact,
            "expected_component_exact_count_when_clamped": 2 * expected_video_calls,
            "all_current_consumed_values_exact": self.injected_calls == 0 or self.current_exact == 2 * expected_video_calls,
            "all_future_consumed_values_exact": self.injected_calls == 0 or self.future_exact == 2 * expected_video_calls,
            "max_current_abs_error": self.max_current_error,
            "max_future_abs_error": self.max_future_error,
            "tokens_per_temporal_group": self.tokens_per_group or None,
            "first_event": self.first_event,
            "last_event": self.last_event,
        }


def infer(
    runner: Any,
    video_prepared: Mapping[str, Any],
    action_prepared: Mapping[str, Any],
    seed: int,
    controller: VideoKVController | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    # Synchronize so this duration contains model inference only.  Rendering,
    # encoding, hashing, and artifact writes are timed separately by callers.
    torch.cuda.synchronize(runner.model.device)
    inference_started = time.perf_counter()
    if controller is not None:
        controller.install()
    try:
        result = controlled_infer(
            runner.policy,
            video_prepared["image"],
            video_prepared["context"],
            video_prepared["context_mask"],
            action_prepared["context"],
            action_prepared["context_mask"],
            seed=seed,
            controller=controller,
        )
    finally:
        if controller is not None:
            controller.uninstall()
    torch.cuda.synchronize(runner.model.device)
    inference_seconds = time.perf_counter() - inference_started
    action = result["action"].detach().cpu().float()
    small = {
        "video_timesteps": result["video_timesteps"].detach().cpu().float(),
        "video_deltas": result["video_deltas"].detach().cpu().float(),
        "action_timesteps": result["action_timesteps"].detach().cpu().float(),
        "action_deltas": result["action_deltas"].detach().cpu().float(),
        "initial_video_noise_hash": tensor_sha256(result["initial_video_noise"]),
        "initial_action_noise_hash": tensor_sha256(result["initial_action_noise"]),
        "first_frame_latent_hash": tensor_sha256(result["first_frame_latent"]),
        "tokens_per_group": int(result["tokens_per_group"]),
        "model_inference_seconds": inference_seconds,
    }
    del result
    return action, small


def capture_source(runner: Any, prepared: Mapping[str, Any], seed: int) -> tuple[torch.Tensor, VideoKVController, dict[str, Any]]:
    controller = VideoKVController(runner.model)
    controller.capture = True
    action, run = infer(runner, prepared, prepared, seed, controller)
    run["hook"] = controller.summary(runner.policy.num_inference_steps, len(runner.model.video_expert.blocks))
    return action, controller, run


def strict_infer(
    runner: Any,
    recipient: Mapping[str, Any],
    seed: int,
    current: Mapping[tuple[int, int], Mapping[str, torch.Tensor]],
    future: Mapping[tuple[int, int], Mapping[str, torch.Tensor]],
) -> tuple[torch.Tensor, dict[str, Any]]:
    controller = VideoKVController(runner.model)
    controller.current_cache = current
    controller.future_cache = future
    action, run = infer(runner, recipient, recipient, seed, controller)
    run["hook"] = controller.summary(runner.policy.num_inference_steps, len(runner.model.video_expert.blocks))
    hook = run["hook"]
    if not all((hook["hook_reached_all_video_sites"], hook["hook_reached_all_action_sites"],
                hook["strict_injection_reached_all_video_sites"], hook["all_current_consumed_values_exact"],
                hook["all_future_consumed_values_exact"])):
        raise AssertionError(f"STRICT_CONSUMER_ASSERTION_FAILED:{hook}")
    return action, run


def continuous_env_action(action: torch.Tensor, processor: Any) -> np.ndarray:
    import run_first_grasp_lock as fgl

    processed = np.asarray(fgl._denormalize_action(action, processor)[0], dtype=np.float32)
    processed[..., -1] = processed[..., -1] * 2 - 1
    processed = fgl.invert_gripper_action(processed)
    return np.asarray(processed, dtype=np.float32)


def binary_env_action(action: torch.Tensor, processor: Any) -> np.ndarray:
    value = continuous_env_action(action, processor)
    value[..., -1] = np.sign(value[..., -1])
    return value


def max_abs(a: torch.Tensor | np.ndarray, b: torch.Tensor | np.ndarray) -> float:
    aa = a.detach().cpu().numpy() if torch.is_tensor(a) else np.asarray(a)
    bb = b.detach().cpu().numpy() if torch.is_tensor(b) else np.asarray(b)
    return float(np.max(np.abs(aa.astype(np.float64) - bb.astype(np.float64))))


def load_registry() -> tuple[list[dict[str, Any]], dict[tuple[int, int, float], dict[str, Any]]]:
    phases = [
        row for row in read_jsonl(G1 / "phase_registry.jsonl")
        if row.get("model") == "joint" and row.get("phase") == PHASE
    ]
    donors = [
        row for row in read_jsonl(G1 / "donor_bank/joint/counterfactual_qc.jsonl")
        if row.get("factor") == FACTOR and row.get("phase") == PHASE and row.get("relation_control") == CONTROL
    ]
    index = {(int(row["task_id"]), int(row["source_state_id"]), float(row["signed_dose"])): row for row in donors}
    phases.sort(key=lambda row: (int(row["task_id"]), int(row["source_state_id"])))
    return phases, index


def registry_validity_rows(phases: list[dict[str, Any]], donors: dict[tuple[int, int, float], dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for phase in phases:
        task, state = int(phase["task_id"]), int(phase["source_state_id"])
        for dose in (-4.0, -2.0, -1.0, 1.0, 2.0, 4.0):
            donor = donors.get((task, state, dose))
            rows.append({
                "task_id": task,
                "source_state_id": state,
                "base_state_id": phase["base_state_id"],
                "phase": PHASE,
                "factor": FACTOR,
                "signed_dose_cm": dose,
                "registry_entry_present": donor is not None,
                "donor_valid": donor is not None and donor.get("donor_valid") is True,
                "invalid_reasons": "" if donor is None else "|".join(donor.get("invalid_reasons", [])),
                "donor_observation_path": "" if donor is None else donor.get("donor_observation_path", ""),
                "case_id": "" if donor is None else donor.get("case_id", ""),
            })
    return rows


def _camera_id(model: Any, name: str) -> int:
    for attr in ("camera_name2id", "cam_name2id"):
        method = getattr(model, attr, None)
        if callable(method):
            return int(method(name))
    raise KeyError(f"Camera unavailable: {name}")


def _camera_snapshot(sim: Any, width: int, height: int, names: list[str]) -> dict[str, Any]:
    cameras: dict[str, Any] = {}
    for name in names:
        camera_id = _camera_id(sim.model, name)
        fovy = float(np.asarray(sim.model.cam_fovy).reshape(-1)[camera_id])
        fy = 0.5 * float(height) / math.tan(0.5 * math.radians(fovy))
        fx, cx, cy = fy, (float(width) - 1.0) / 2.0, (float(height) - 1.0) / 2.0
        world_from_camera = np.eye(4, dtype=np.float64)
        world_from_camera[:3, :3] = np.asarray(sim.data.cam_xmat[camera_id], dtype=np.float64).reshape(3, 3)
        world_from_camera[:3, 3] = np.asarray(sim.data.cam_xpos[camera_id], dtype=np.float64).reshape(3)
        cameras[name] = {
            "camera_id": camera_id,
            "intrinsics": {
                "width": width, "height": height, "fovy_deg": fovy,
                "fx": fx, "fy": fy, "cx": cx, "cy": cy,
                "K": [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
                "convention": "pinhole_from_mujoco_vertical_fovy_centered_pixels",
            },
            "model_pos": np.asarray(sim.model.cam_pos[camera_id], dtype=np.float64),
            "model_quat_wxyz": np.asarray(sim.model.cam_quat[camera_id], dtype=np.float64),
            "world_from_camera": world_from_camera,
            "camera_from_world": np.linalg.inv(world_from_camera),
            "extrinsic_convention": "MuJoCo sim.data.cam_xpos/cam_xmat after exact saved-state restore",
        }
    return cameras


def _observation_pose_metadata(obs: Mapping[str, np.ndarray], phase: Mapping[str, Any]) -> dict[str, Any]:
    target, goal = str(phase["target_identity"]), str(phase["goal_identity"])
    return {
        "robot": {
            "eef_position_m": np.asarray(obs["robot0_eef_pos"]),
            "eef_quaternion_xyzw": np.asarray(obs["robot0_eef_quat"]),
            "joint_position_rad": np.asarray(obs["robot0_joint_pos"]),
            "gripper_joint_position": np.asarray(obs["robot0_gripper_qpos"]),
        },
        "target": {
            "identity": target,
            "position_m": np.asarray(obs[f"{target}_pos"]),
            "quaternion_wxyz": np.asarray(obs[f"{target}_quat"]),
        },
        "placement_goal": {
            "identity": goal,
            "position_m": np.asarray(obs[f"{goal}_pos"]),
            "quaternion_wxyz": np.asarray(obs[f"{goal}_quat"]),
        },
    }


def _save_rgb(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.asarray(value)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected uint8 RGB image, got {image.dtype} {image.shape}")
    Image.fromarray(image, mode="RGB").save(path, format="PNG", compress_level=6)


def _visual_candidate_registry(phases: list[dict[str, Any]], donors: dict[tuple[int, int, float], dict[str, Any]]) -> dict[str, Any]:
    selected = []
    signed_grid = (-4.0, -2.0, -1.0, 1.0, 2.0, 4.0)
    for task in TASKS:
        task_phases = [row for row in phases if int(row["task_id"]) == task]
        complete = [
            row for row in task_phases
            if all(donors.get((task, int(row["source_state_id"]), dose), {}).get("donor_valid") is True for dose in signed_grid)
        ]
        pool = complete if complete else sorted(
            task_phases,
            key=lambda row: (-sum(donors.get((task, int(row["source_state_id"]), dose), {}).get("donor_valid") is True for dose in signed_grid), int(row["source_state_id"])),
        )
        if not pool:
            raise RuntimeError(f"NO_VISUAL_CANDIDATE_TASK_{task}")
        row = sorted(pool, key=lambda value: int(value["source_state_id"]))[0] if complete else pool[0]
        selected.append({
            "candidate_case_id": f"A_FIGURE_CANDIDATE__task{task}__state{int(row['source_state_id']):02d}",
            "task_id": task,
            "source_state_id": int(row["source_state_id"]),
            "base_state_id": row["base_state_id"],
            "all_six_signed_donors_technically_valid": row in complete,
            "valid_signed_doses_cm": [dose for dose in signed_grid if donors.get((task, int(row["source_state_id"]), dose), {}).get("donor_valid") is True],
        })
    return {
        "protocol": VISUALIZATION_AMENDMENT_VERSION,
        "frozen_before_formal_model_forward": True,
        "selection_rule": "For each task, lowest source_state_id with all six signed-dose donors passing frozen technical QC; if none, highest technical coverage then lowest source_state_id. No action/model outcome is inspected.",
        "scientific_outcome_used": False,
        "candidates": selected,
        "all_result_cases_retained": True,
    }


def run_visuals(args: argparse.Namespace) -> None:
    """Freeze visualization/provenance assets without running any WAM."""
    import run_first_grasp_lock as fgl
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    root = args.output / "visualization"
    root.mkdir(parents=True, exist_ok=True)
    phases, donors = load_registry()
    candidate_registry = _visual_candidate_registry(phases, donors)
    atomic_json(root / "visualization_candidate_registry.json", candidate_registry)
    suite = benchmark.get_benchmark_dict()["libero_object"]()
    index_rows: list[dict[str, Any]] = []
    render_seconds = 0.0
    write_seconds = 0.0
    technical_failures = 0
    image_comparisons = 0
    image_bit_exact = 0
    for task_id in TASKS:
        task = suite.get_task(task_id)
        bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env = OffScreenRenderEnv(
            bddl_file_name=bddl,
            camera_heights=fgl.LIBERO_ENV_RESOLUTION,
            camera_widths=fgl.LIBERO_ENV_RESOLUTION,
        )
        env.seed(42)
        env.reset()
        sim = fgl.get_sim(env)
        for phase in [row for row in phases if int(row["task_id"]) == task_id]:
            state_id = int(phase["source_state_id"])
            sources: list[tuple[str, float, Path, Path, str]] = [(
                "recipient", 0.0, Path(phase["recipient_state_path"]), Path(phase["recipient_observation_path"]), phase["base_state_id"]
            )]
            for signed in (-4.0, -2.0, -1.0, 1.0, 2.0, 4.0):
                row = donors.get((task_id, state_id, signed))
                if row is not None and row.get("donor_valid") is True:
                    sources.append((f"donor_{signed:+g}cm", signed, Path(row["donor_state_path"]), Path(row["donor_observation_path"]), row["case_id"]))
            for label, signed, state_path, obs_path, case_id in sources:
                asset_dir = root / "assets" / f"task_{task_id}" / f"state_{state_id:02d}" / label.replace("+", "p").replace("-", "m")
                metadata_path = asset_dir / "metadata.json"
                if metadata_path.is_file():
                    meta = json.loads(metadata_path.read_text(encoding="utf-8"))
                    index_rows.append(meta["index_row"])
                    image_comparisons += len(meta.get("images", []))
                    image_bit_exact += sum(bool(row.get("bit_exact")) for row in meta.get("images", []))
                    continue
                raw_obs = load_npz(obs_path)
                render_started = time.perf_counter()
                with np.load(state_path, allow_pickle=False) as archive:
                    saved = {key: archive[key].copy() for key in archive.files}
                sim.set_state_from_flattened(saved["flat"])
                for key in ("ctrl", "qfrc_applied", "xfrc_applied"):
                    if key in saved:
                        getattr(sim.data, key)[:] = saved[key]
                sim.forward()
                restored_obs = dict(env.env._get_observations(force_update=True))
                camera_names = sorted(key[:-6] for key in raw_obs if key.endswith("_image"))
                first_image = np.asarray(raw_obs[f"{camera_names[0]}_image"])
                cameras = _camera_snapshot(sim, int(first_image.shape[1]), int(first_image.shape[0]), camera_names)
                render_elapsed = time.perf_counter() - render_started
                render_seconds += render_elapsed
                write_started = time.perf_counter()
                image_records = []
                for camera in camera_names:
                    policy_rgb = np.asarray(raw_obs[f"{camera}_image"])
                    sim_rgb = np.asarray(restored_obs[f"{camera}_image"])
                    policy_path = asset_dir / f"policy_input_rgb__{camera}.png"
                    sim_path = asset_dir / f"sim_render_same_view__{camera}.png"
                    _save_rgb(policy_path, policy_rgb)
                    _save_rgb(sim_path, sim_rgb)
                    image_records.append({
                        "camera": camera,
                        "policy_input_rgb_path": str(policy_path),
                        "sim_render_path": str(sim_path),
                        "policy_rgb_array_hash": array_hash(policy_rgb),
                        "sim_render_array_hash": array_hash(sim_rgb),
                        "same_shape": policy_rgb.shape == sim_rgb.shape,
                        "bit_exact": np.array_equal(policy_rgb, sim_rgb),
                        "max_abs_error": int(np.max(np.abs(policy_rgb.astype(np.int16) - sim_rgb.astype(np.int16)))),
                    })
                pose = _observation_pose_metadata(raw_obs, phase)
                requested_vector = None
                achieved_vector = None
                if signed != 0.0:
                    donor_row = donors[(task_id, state_id, signed)]
                    requested_vector = donor_row.get("requested_vector_m")
                    achieved_vector = donor_row.get("achieved_displacement_m")
                index_row = {
                    "task_id": task_id, "source_state_id": state_id, "base_state_id": phase["base_state_id"],
                    "source_label": label, "case_id": case_id, "signed_dose_cm": signed,
                    "metadata_path": str(metadata_path), "technical_status": "CAPTURED",
                }
                metadata = {
                    "protocol": VISUALIZATION_AMENDMENT_VERSION,
                    "single_call_label_zh": "预测动作，未执行",
                    "single_call_label_en": "PREDICTED_ACTION_NOT_EXECUTED",
                    "closed_loop_video_status": "NOT_APPLICABLE_PROTOCOL_PROHIBITS_CLOSED_LOOP",
                    "no_execution_trajectory_generated": True,
                    "task_id": task_id, "source_state_id": state_id, "base_state_id": phase["base_state_id"],
                    "case_id": case_id, "source_label": label,
                    "requested_signed_dose_cm": signed,
                    "requested_displacement_vector_m": requested_vector,
                    "achieved_displacement_vector_m": achieved_vector,
                    "actual_displacement_norm_cm": None if achieved_vector is None else 100.0 * float(np.linalg.norm(achieved_vector)),
                    "state_path": str(state_path), "state_file_sha256": sha256_file(state_path),
                    "observation_path": str(obs_path), "observation_file_sha256": sha256_file(obs_path),
                    "poses": pose, "cameras": cameras, "images": image_records,
                    "render_restore_seconds": render_elapsed,
                    "index_row": index_row,
                }
                atomic_json(metadata_path, metadata)
                image_comparisons += len(image_records)
                image_bit_exact += sum(bool(row["bit_exact"]) for row in image_records)
                write_elapsed = time.perf_counter() - write_started
                write_seconds += write_elapsed
                index_rows.append(index_row)
        env.close()
    write_csv(root / "visualization_index.csv", index_rows)
    manifest = {
        "status": "VISUALIZATION_ASSETS_FROZEN",
        "protocol": VISUALIZATION_AMENDMENT_VERSION,
        "candidate_registry_sha256": sha256_file(root / "visualization_candidate_registry.json"),
        "visualization_index_sha256": sha256_file(root / "visualization_index.csv"),
        "asset_source_count": len(index_rows),
        "technical_failures": technical_failures,
        "policy_rgb_vs_same_state_rerender_comparisons": image_comparisons,
        "policy_rgb_vs_same_state_rerender_bit_exact": image_bit_exact,
        "policy_rgb_vs_same_state_rerender_not_bit_exact": image_comparisons - image_bit_exact,
        "rerender_interpretation": "Both lossless artifacts are retained. Pixel mismatch is descriptive and is not used for state/case selection; the frozen raw policy RGB remains authoritative for inference.",
        "model_forward_passes": 0,
        "render_restore_seconds": render_seconds,
        "image_metadata_write_seconds": write_seconds,
        "inference_seconds": 0.0,
        "timing_domains_separated": True,
        "closed_loop_video_status": "NOT_APPLICABLE_PROTOCOL_PROHIBITS_CLOSED_LOOP",
    }
    atomic_json(root / "visualization_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False))


def source_cache_hash(cache: Mapping[tuple[int, int], Mapping[str, torch.Tensor]], temporal: str) -> str:
    values = []
    for key in sorted(cache):
        for component in ("k", "v"):
            tensor = cache[key][component]
            tpg = tensor.shape[1] // 3
            values.append(tensor[:, :tpg] if temporal == "current" else tensor[:, tpg:])
    return tensor_collection_hash(values)


def legacy_infer(runner: Any, recipient: Mapping[str, Any], seed: int,
                 donor_cache: Mapping[tuple[int, int], Mapping[str, torch.Tensor]], temporal: str) -> torch.Tensor:
    controller = QKVController(runner.model)
    controller.patch_cache = dict(donor_cache)
    controller.patch_modality = "video"
    controller.patch_layers = set(range(len(runner.model.video_expert.blocks)))
    controller.patch_temporal_groups = (0,) if temporal == "current" else (1, 2)
    action, _ = infer(runner, recipient, recipient, seed, controller)  # compatible controller API
    return action


def run_metadata(runner: Any, gpu_id: int) -> dict[str, Any]:
    model = runner.model
    cfg = runner.policy.cfg
    action_ts, action_ds = model.infer_action_scheduler.build_inference_schedule(
        runner.policy.num_inference_steps, model.device, model.torch_dtype, shift_override=None
    )
    video_ts, video_ds = model.infer_video_scheduler.build_inference_schedule(
        runner.policy.num_inference_steps, model.device, model.torch_dtype, shift_override=None
    )
    try:
        git_head = subprocess.check_output(["git", "-C", str(BADWAM), "rev-parse", "HEAD"], text=True).strip()
        git_diff = subprocess.check_output(["git", "-C", str(BADWAM), "diff", "--stat"], text=True).strip()
    except Exception:
        git_head, git_diff = "UNAVAILABLE", "UNAVAILABLE"
    props = torch.cuda.get_device_properties(model.device)
    return {
        "protocol_version": PROTOCOL_VERSION,
        "model": "Joint-WAM",
        "factor": FACTOR,
        "phase": PHASE,
        "gpu_requested": gpu_id,
        "gpu_name": props.name,
        "gpu_total_memory_bytes": props.total_memory,
        "checkpoint_path": str(runner.policy.checkpoint.resolve()),
        "checkpoint_link_path": str(runner.policy.checkpoint),
        "dataset_stats_path": str(runner.policy.stats_path.resolve()),
        "loaded_weight_hash": model_weight_hash(model),
        "git_head": git_head,
        "git_tracked_diff_stat": git_diff,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "dtype": str(model.torch_dtype),
        "mixed_precision": str(cfg.mixed_precision),
        "attention_implementation": "torch.nn.functional.scaled_dot_product_attention compatibility path",
        "video_attention_mask_mode": str(model.video_expert.video_attention_mask_mode),
        "num_layers": len(model.video_expert.blocks),
        "num_heads": int(model.video_expert.num_heads),
        "attention_head_dim": int(model.video_expert.attn_head_dim),
        "actual_action_denoising_steps_N": len(action_ts),
        "actual_video_denoising_steps_N": len(video_ts),
        "configured_num_inference_steps": int(runner.policy.num_inference_steps),
        "action_timesteps": action_ts.detach().cpu().float().tolist(),
        "action_deltas": action_ds.detach().cpu().float().tolist(),
        "video_timesteps": video_ts.detach().cpu().float().tolist(),
        "video_deltas": video_ds.detach().cpu().float().tolist(),
        "video_scheduler": type(model.infer_video_scheduler).__name__,
        "action_scheduler": type(model.infer_action_scheduler).__name__,
        "policy_seed_rule": "frozen per-state phase_registry.policy_seed",
        "random_device": str(cfg.EVALUATION.rand_device),
        "action_horizon": int(runner.policy.action_horizon),
        "num_video_frames": int(runner.policy.num_video_frames),
        "other_action_conditions": [
            "language-plus-proprio context via action block cross-attention at every layer",
            "action timestep modulation",
            "seeded initial action noise and recurrent action denoising latent",
        ],
        "video_reads_action": False,
        "video_action_feedback_evidence": "video_pre receives action=None and video query mask has no action columns",
        "strict_cache_position": {
            "K": "after q/k projection, RMSNorm, and RoPE; before video/action K concatenation",
            "V": "after v projection; before video/action V concatenation",
            "repeat_position_encoding_avoided": True,
        },
        "consumer_slice_source": "frozen registry plus runtime token-count validation; not inferred from locus numbering",
        "current_token_interval": "[0,tokens_per_group)",
        "future_token_interval": "[tokens_per_group,3*tokens_per_group)",
        "d_star_cm": D_STAR_CM,
        "dose_grid_cm": list(DOSES_CM),
        "d_star_resolution": "Group-1/2 frozen F1 primary doses are ±1/±2/±4 cm; d*=4 cm is the unique value whose quarter/half/full grid exactly equals existing frozen donors",
        "cps_20_step_optimization_used_as_action_denoising_count": False,
    }


def choose_a0(phases: list[dict[str, Any]], donors: dict[tuple[int, int, float], dict[str, Any]]) -> list[dict[str, Any]]:
    # Frozen before model execution: the two lowest source IDs in tasks 0 and 1
    # having the +d* preregistered F1 donor technically valid.
    selected = []
    for task in (0, 1):
        eligible = []
        for phase in phases:
            if int(phase["task_id"]) != task:
                continue
            state = int(phase["source_state_id"])
            if donors.get((task, state, D_STAR_CM), {}).get("donor_valid") is True:
                eligible.append(phase)
        selected.extend(eligible[:2])
    if len(selected) != 4:
        raise RuntimeError("A0_TWO_TASK_TWO_STATE_SELECTION_INFEASIBLE")
    return selected


def a0_one(runner: Any, phase: dict[str, Any], donor: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    task, state = int(phase["task_id"]), int(phase["source_state_id"])
    seed = int(phase["policy_seed"])
    recipient_obs = load_npz(Path(phase["recipient_observation_path"]))
    donor_obs = load_npz(Path(donor["donor_observation_path"]))
    recipient = runner._prepared(recipient_obs, phase["instruction"])
    donor_prepared = runner._prepared(donor_obs, phase["instruction"])
    rows = []

    base, _ = infer(runner, recipient, recipient, seed, None)
    repeat, _ = infer(runner, recipient, recipient, seed, None)
    rows.append({"task_id": task, "source_state_id": state, "check": "native_repeat", "pass": torch.equal(base, repeat), "max_abs_error": max_abs(base, repeat)})

    read_ctl = VideoKVController(runner.model)
    read_ctl.read_only = True
    readonly, readonly_run = infer(runner, recipient, recipient, seed, read_ctl)
    read_summary = read_ctl.summary(runner.policy.num_inference_steps, len(runner.model.video_expert.blocks))
    rows.append({"task_id": task, "source_state_id": state, "check": "read_only_hook", "pass": torch.equal(base, readonly), "max_abs_error": max_abs(base, readonly)})
    rows.append({"task_id": task, "source_state_id": state, "check": "hook_real_path_coverage", "pass": read_summary["hook_reached_all_video_sites"] and read_summary["hook_reached_all_action_sites"], "max_abs_error": 0.0})

    recipient_action, recipient_trace, recipient_run = capture_source(runner, recipient, seed)
    rows.append({"task_id": task, "source_state_id": state, "check": "capture_is_read_only", "pass": torch.equal(base, recipient_action), "max_abs_error": max_abs(base, recipient_action)})
    identity, identity_run = strict_infer(runner, recipient, seed, recipient_trace.cache, recipient_trace.cache)
    rows.append({"task_id": task, "source_state_id": state, "check": "same_value_both_sources", "pass": torch.equal(base, identity), "max_abs_error": max_abs(base, identity)})

    donor_action, donor_trace, donor_run = capture_source(runner, donor_prepared, seed)
    donor_replay, donor_replay_run = infer(runner, donor_prepared, donor_prepared, seed, None)
    rows.append({"task_id": task, "source_state_id": state, "check": "full_donor_exogenous_replay", "pass": torch.equal(donor_action, donor_replay), "max_abs_error": max_abs(donor_action, donor_replay)})

    current_only, current_run = strict_infer(runner, recipient, seed, donor_trace.cache, recipient_trace.cache)
    future_only, future_run = strict_infer(runner, recipient, seed, recipient_trace.cache, donor_trace.cache)
    rows.append({"task_id": task, "source_state_id": state, "check": "C_donor_F_recipient_consumed_exact", "pass": current_run["hook"]["all_future_consumed_values_exact"] and current_run["hook"]["all_current_consumed_values_exact"], "max_abs_error": max(current_run["hook"]["max_current_abs_error"], current_run["hook"]["max_future_abs_error"])})
    rows.append({"task_id": task, "source_state_id": state, "check": "C_recipient_F_donor_consumed_exact", "pass": future_run["hook"]["all_future_consumed_values_exact"] and future_run["hook"]["all_current_consumed_values_exact"], "max_abs_error": max(future_run["hook"]["max_current_abs_error"], future_run["hook"]["max_future_abs_error"])})

    frozen_path = G1 / "model_runs/joint/cases" / donor["case_id"] / "result.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    rows.append({"task_id": task, "source_state_id": state, "check": "recipient_matches_frozen_Group1", "pass": tensor_sha256(base) == frozen["recipient_action_hash"], "max_abs_error": 0.0 if tensor_sha256(base) == frozen["recipient_action_hash"] else float("nan")})
    rows.append({"task_id": task, "source_state_id": state, "check": "donor_matches_frozen_Group1", "pass": tensor_sha256(donor_action) == frozen["donor_action_hash"], "max_abs_error": 0.0 if tensor_sha256(donor_action) == frozen["donor_action_hash"] else float("nan")})

    detail = {
        "task_id": task,
        "source_state_id": state,
        "base_state_id": phase["base_state_id"],
        "signed_dose_cm": float(donor["signed_dose"]),
        "seed": seed,
        "recipient_observation_hash": observation_sha256(recipient_obs),
        "donor_observation_hash": observation_sha256(donor_obs),
        "recipient_prepared_hashes": prepared_hashes(recipient),
        "donor_prepared_hashes": prepared_hashes(donor_prepared),
        "recipient_action_hash": tensor_sha256(base),
        "donor_action_hash": tensor_sha256(donor_action),
        "strict_current_only_hash": tensor_sha256(current_only),
        "strict_future_only_hash": tensor_sha256(future_only),
        "recipient_current_cache_hash": source_cache_hash(recipient_trace.cache, "current"),
        "recipient_future_cache_hash": source_cache_hash(recipient_trace.cache, "future"),
        "donor_current_cache_hash": source_cache_hash(donor_trace.cache, "current"),
        "donor_future_cache_hash": source_cache_hash(donor_trace.cache, "future"),
        "recipient_capture": recipient_run,
        "donor_capture": donor_run,
        "identity_clamp": identity_run,
        "strict_current_only": current_run,
        "strict_future_only": future_run,
        "read_only_hook": {"run": readonly_run, "hook": read_summary},
        "donor_replay": donor_replay_run,
        "full_donor_replay_external_inputs": ["input_image", "language_plus_proprio_context", "context_mask", "video_noise", "action_noise", "video/action scheduler state"],
        "forbidden_intermediate_overrides_used": False,
    }
    del recipient_trace, donor_trace
    gc.collect()
    torch.cuda.empty_cache()
    return rows, detail


def run_a0(args: argparse.Namespace) -> None:
    out = args.output / "a0"
    out.mkdir(parents=True, exist_ok=True)
    phases, donors = load_registry()
    selected = choose_a0(phases, donors)
    selection = {
        "selection_frozen_before_model_execution": True,
        "selection_rule": "For tasks 0 and 1, choose the two lowest source_state_id values with the frozen +d* F1 donor technically valid",
        "states": [{"task_id": int(p["task_id"]), "source_state_id": int(p["source_state_id"]), "base_state_id": p["base_state_id"]} for p in selected],
        "dose_for_A0_cm": D_STAR_CM,
    }
    atomic_json(out / "a0_selection.json", selection)
    torch.set_num_threads(args.threads)
    started = time.time()
    runner = make_capture("joint", args.gpu)
    torch.cuda.reset_peak_memory_stats(runner.model.device)
    before_hash = model_weight_hash(runner.model)
    metadata = run_metadata(runner, args.gpu)
    all_rows, details = [], []
    for phase in selected:
        donor = donors[(int(phase["task_id"]), int(phase["source_state_id"]), D_STAR_CM)]
        rows, detail = a0_one(runner, phase, donor)
        all_rows.extend(rows)
        details.append(detail)
        atomic_json(out / "progress.json", {"completed_states": len(details), "total_states": len(selected), "last": detail["base_state_id"]})
    after_hash = model_weight_hash(runner.model)
    passed = all(bool(row["pass"]) for row in all_rows) and before_hash == after_hash
    write_csv(out / "replay_checks.csv", all_rows)
    atomic_json(out / "a0_details.json", details)
    metadata.update({
        "a0_pass": passed,
        "weight_hash_before": before_hash,
        "weight_hash_after": after_hash,
        "model_weight_hash_unchanged": before_hash == after_hash,
        "runtime_seconds": time.time() - started,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(runner.model.device),
        "formal_execution_authorized_by_A0": passed,
    })
    atomic_json(out / "a0_report.json", metadata)
    print(json.dumps({"status": "A0_PASS" if passed else "A0_FAILED", "output": str(out), "runtime_seconds": metadata["runtime_seconds"], "peak_cuda_memory_bytes": metadata["peak_cuda_memory_bytes"]}, ensure_ascii=False))
    if not passed:
        raise SystemExit(2)


def old_group2_actions(case_id: str) -> dict[str, np.ndarray] | None:
    path = G2 / "cases" / case_id / "actions.npz"
    if not path.is_file():
        return None
    with np.load(path) as archive:
        return {key: np.asarray(archive[key], dtype=np.float32) for key in ("A00", "A10", "A01", "A11", "DONOR_NATIVE")}


def state_output_dir(root: Path, phase: Mapping[str, Any]) -> Path:
    return root / "cases" / f"task_{int(phase['task_id'])}" / f"state_{int(phase['source_state_id']):02d}"


def run_formal_state(runner: Any, phase: dict[str, Any], donors: dict[tuple[int, int, float], dict[str, Any]], root: Path) -> dict[str, Any]:
    task, state = int(phase["task_id"]), int(phase["source_state_id"])
    out = state_output_dir(root, phase)
    complete = out / "complete.json"
    if complete.is_file():
        existing = json.loads(complete.read_text(encoding="utf-8"))
        if existing.get("status") == "COMPLETED" and existing.get("protocol_version") == PROTOCOL_VERSION:
            return existing
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    inference_seconds = 0.0
    artifact_write_seconds = 0.0
    seed = int(phase["policy_seed"])
    recipient_obs = load_npz(Path(phase["recipient_observation_path"]))
    recipient = runner._prepared(recipient_obs, phase["instruction"])
    recipient_action, recipient_trace, recipient_run = capture_source(runner, recipient, seed)
    inference_seconds += float(recipient_run["model_inference_seconds"])
    recipient_cont = continuous_env_action(recipient_action, runner.processor)
    recipient_bin = binary_env_action(recipient_action, runner.processor)

    action_arrays: dict[str, np.ndarray] = {
        "NATURAL__0__normalized": recipient_action.numpy(),
        "NATURAL__0__env_continuous": recipient_cont,
        "NATURAL__0__env_binary": recipient_bin,
    }
    # Preserve the exact bfloat16 model-input RGB tensor as raw uint16 bits,
    # while lossless uint8 camera RGB and simulator re-renders live under
    # visualization/assets.
    policy_rgb_tensors: dict[str, np.ndarray] = {
        "recipient__bfloat16_bits": recipient["image"].detach().cpu().contiguous().view(torch.uint16).numpy(),
    }
    config_rows: list[dict[str, Any]] = []
    replay_rows: list[dict[str, Any]] = []
    cache_rows: list[dict[str, Any]] = [{
        "task_id": task, "source_state_id": state, "source_label": "0",
        "current_cache_hash": source_cache_hash(recipient_trace.cache, "current"),
        "future_cache_hash": source_cache_hash(recipient_trace.cache, "future"),
    }]
    config_rows.append({
        "task_id": task, "source_state_id": state, "dose_magnitude_cm": 0.0,
        "C_source_cm": 0.0, "F_source_cm": 0.0, "condition": "A00",
        "action_key": "STRICT__C0__F0", "executed": True,
        "model_inference_computed": True, "environment_action_executed": False,
        "visualization_label": "PREDICTED_ACTION_NOT_EXECUTED",
        "action_hash": tensor_sha256(recipient_action), "hook_all_exact": True,
    })
    for suffix, value in (("normalized", recipient_action.numpy()), ("env_continuous", recipient_cont), ("env_binary", recipient_bin)):
        action_arrays[f"STRICT__C0__F0__{suffix}"] = value

    valid_donors: dict[float, tuple[torch.Tensor, VideoKVController, dict[str, Any], Mapping[str, Any]]] = {}
    for magnitude in DOSES_CM:
        signed_available = []
        for signed in (-magnitude, magnitude):
            donor_row = donors.get((task, state, signed))
            if donor_row is None or donor_row.get("donor_valid") is not True:
                replay_rows.append({"task_id": task, "source_state_id": state, "signed_dose_cm": signed, "check": "donor_availability", "pass": False, "max_abs_error": "", "reason": "MISSING_OR_FROZEN_QC_INVALID"})
                continue
            donor_obs = load_npz(Path(donor_row["donor_observation_path"]))
            donor_prepared = runner._prepared(donor_obs, phase["instruction"])
            donor_action, donor_trace, donor_run = capture_source(runner, donor_prepared, seed)
            inference_seconds += float(donor_run["model_inference_seconds"])
            replay, _ = infer(runner, donor_prepared, donor_prepared, seed, None)
            # The replay is a separate full model call and is part of inference timing.
            # Re-run metadata is intentionally not mixed with artifact I/O time.
            replay_action, replay_meta = replay, _
            inference_seconds += float(replay_meta["model_inference_seconds"])
            replay = replay_action
            replay_rows.append({"task_id": task, "source_state_id": state, "signed_dose_cm": signed, "check": "full_donor_exogenous_replay", "pass": torch.equal(donor_action, replay), "max_abs_error": max_abs(donor_action, replay), "reason": ""})
            label = f"p{magnitude:g}" if signed > 0 else f"m{magnitude:g}"
            action_arrays[f"NATURAL__{label}__normalized"] = donor_action.numpy()
            action_arrays[f"NATURAL__{label}__env_continuous"] = continuous_env_action(donor_action, runner.processor)
            action_arrays[f"NATURAL__{label}__env_binary"] = binary_env_action(donor_action, runner.processor)
            policy_rgb_tensors[f"donor_{label}__bfloat16_bits"] = donor_prepared["image"].detach().cpu().contiguous().view(torch.uint16).numpy()
            cache_rows.append({
                "task_id": task, "source_state_id": state, "source_label": f"{signed:+g}",
                "current_cache_hash": source_cache_hash(donor_trace.cache, "current"),
                "future_cache_hash": source_cache_hash(donor_trace.cache, "future"),
            })
            valid_donors[signed] = (donor_action, donor_trace, donor_run, donor_row)
            signed_available.append(signed)
        if len(signed_available) != 2:
            for c in (0.0, magnitude, -magnitude):
                for f in (0.0, magnitude, -magnitude):
                    if c == 0.0 and f == 0.0:
                        continue
                    config_rows.append({
                        "task_id": task, "source_state_id": state, "dose_magnitude_cm": magnitude,
                        "C_source_cm": c, "F_source_cm": f, "condition": "STRICT_GRID",
                        "action_key": "", "executed": False,
                        "model_inference_computed": False, "environment_action_executed": False,
                        "visualization_label": "PREDICTION_UNAVAILABLE_FROZEN_DONOR_QC",
                        "action_hash": "", "hook_all_exact": "",
                    })
            for signed in list(valid_donors):
                if abs(signed) == magnitude:
                    del valid_donors[signed]
            gc.collect(); torch.cuda.empty_cache()
            continue

        cache_for = {0.0: recipient_trace.cache, magnitude: valid_donors[magnitude][1].cache, -magnitude: valid_donors[-magnitude][1].cache}
        for c in (0.0, magnitude, -magnitude):
            for f in (0.0, magnitude, -magnitude):
                if c == 0.0 and f == 0.0:
                    continue
                action, run = strict_infer(runner, recipient, seed, cache_for[c], cache_for[f])
                inference_seconds += float(run["model_inference_seconds"])
                c_label = "0" if c == 0 else (f"p{magnitude:g}" if c > 0 else f"m{magnitude:g}")
                f_label = "0" if f == 0 else (f"p{magnitude:g}" if f > 0 else f"m{magnitude:g}")
                key = f"STRICT__C{c_label}__F{f_label}"
                action_arrays[f"{key}__normalized"] = action.numpy()
                action_arrays[f"{key}__env_continuous"] = continuous_env_action(action, runner.processor)
                action_arrays[f"{key}__env_binary"] = binary_env_action(action, runner.processor)
                hook = run["hook"]
                config_rows.append({
                    "task_id": task, "source_state_id": state, "dose_magnitude_cm": magnitude,
                    "C_source_cm": c, "F_source_cm": f, "condition": "STRICT_GRID",
                    "action_key": key, "executed": True,
                    "model_inference_computed": True, "environment_action_executed": False,
                    "visualization_label": "PREDICTED_ACTION_NOT_EXECUTED",
                    "action_hash": tensor_sha256(action),
                    "hook_all_exact": hook["all_current_consumed_values_exact"] and hook["all_future_consumed_values_exact"],
                })

        # Original single-source implementation is paired only at d* and both signs.
        if magnitude == D_STAR_CM:
            for signed in (-magnitude, magnitude):
                donor_action, donor_trace, _, donor_row = valid_donors[signed]
                for temporal in ("current", "future"):
                    legacy_started = time.perf_counter()
                    legacy = legacy_infer(runner, recipient, seed, donor_trace.cache, temporal)
                    # legacy_infer synchronizes inside infer; wall here only adds tiny Python overhead.
                    inference_seconds += time.perf_counter() - legacy_started
                    label = f"p{magnitude:g}" if signed > 0 else f"m{magnitude:g}"
                    key = f"LEGACY__{temporal}__{label}"
                    action_arrays[f"{key}__normalized"] = legacy.numpy()
                    action_arrays[f"{key}__env_continuous"] = continuous_env_action(legacy, runner.processor)
                    action_arrays[f"{key}__env_binary"] = binary_env_action(legacy, runner.processor)
                    old = old_group2_actions(donor_row["case_id"])
                    expected = None if old is None else old["A10" if temporal == "current" else "A01"]
                    replay_rows.append({
                        "task_id": task, "source_state_id": state, "signed_dose_cm": signed,
                        "check": f"legacy_{temporal}_matches_frozen_Group2", "pass": expected is not None and np.array_equal(legacy.numpy(), expected),
                        "max_abs_error": "" if expected is None else max_abs(legacy, expected), "reason": "FROZEN_GROUP2_MISSING" if expected is None else "",
                    })

        for signed in (magnitude, -magnitude):
            del valid_donors[signed]
        gc.collect(); torch.cuda.empty_cache()

    all_replays_pass = all(bool(row["pass"]) for row in replay_rows if row["check"] != "donor_availability")
    radial = np.asarray(phase["object_position_m"], dtype=np.float64) - np.asarray(phase["eef_position_m"], dtype=np.float64)
    radial /= max(float(np.linalg.norm(radial)), EPS)
    arrow_rows: list[dict[str, Any]] = []
    for key, value in action_arrays.items():
        if not key.endswith("__env_continuous"):
            continue
        action_key = key.removesuffix("__env_continuous")
        continuous = np.asarray(value, dtype=np.float64).reshape(-1, np.asarray(value).shape[-1])
        binary = np.asarray(action_arrays[f"{action_key}__env_binary"], dtype=np.float64).reshape(continuous.shape)
        for step, action in enumerate(continuous):
            translation = action[:3]
            radial_component = float(np.dot(translation, radial))
            orthogonal = translation - radial_component * radial
            arrow_rows.append({
                "task_id": task, "source_state_id": state, "action_key": action_key,
                "action_step": step, "predicted_action_not_executed": True,
                "translation_x_raw": action[0], "translation_y_raw": action[1], "translation_z_raw": action[2],
                "radial_component_raw": radial_component,
                "orthogonal_translation_l2_raw": float(np.linalg.norm(orthogonal)),
                "rotation_x_raw": action[3] if action.size > 3 else "",
                "rotation_y_raw": action[4] if action.size > 4 else "",
                "rotation_z_raw": action[5] if action.size > 5 else "",
                "gripper_continuous_raw": action[-1], "gripper_binary_command": binary[step, -1],
            })
    write_started = time.perf_counter()
    atomic_npz(out / "actions.npz", **action_arrays)
    atomic_npz(out / "policy_input_rgb_tensor_bits.npz", **policy_rgb_tensors)
    write_csv(out / "configurations.csv", config_rows)
    write_csv(out / "replay_checks.csv", replay_rows)
    write_csv(out / "source_cache_hashes.csv", cache_rows)
    write_csv(out / "action_arrow_data.csv", arrow_rows)
    artifact_write_seconds += time.perf_counter() - write_started
    complete_payload = {
        "status": "COMPLETED" if all_replays_pass else "TECHNICAL_CHECK_FAILED",
        "protocol_version": PROTOCOL_VERSION,
        "task_id": task,
        "source_state_id": state,
        "base_state_id": phase["base_state_id"],
        "recipient_observation_path": phase["recipient_observation_path"],
        "recipient_observation_hash": observation_sha256(recipient_obs),
        "recipient_action_hash": tensor_sha256(recipient_action),
        "seed": seed,
        "strict_configurations_executed": sum(row["executed"] is True for row in config_rows),
        "strict_configurations_intended": 25,
        "natural_sources_executed": len([key for key in action_arrays if key.startswith("NATURAL__") and key.endswith("__normalized")]),
        "runtime_seconds": time.time() - started,
        "model_inference_seconds": inference_seconds,
        "artifact_write_seconds": artifact_write_seconds,
        "non_inference_non_write_seconds": max(0.0, time.time() - started - inference_seconds - artifact_write_seconds),
        "visualization_render_seconds": 0.0,
        "timing_domains_separated": True,
        "action_semantics": "PREDICTED_ACTION_NOT_EXECUTED",
        "closed_loop_video_status": "NOT_APPLICABLE_PROTOCOL_PROHIBITS_CLOSED_LOOP",
        "actions_sha256": sha256_file(out / "actions.npz"),
        "recipient_capture": recipient_run,
    }
    atomic_json(complete, complete_payload)
    del recipient_trace
    gc.collect(); torch.cuda.empty_cache()
    return complete_payload


def run_formal(args: argparse.Namespace) -> None:
    a0_path = args.output / "a0/a0_report.json"
    if not a0_path.is_file() or json.loads(a0_path.read_text(encoding="utf-8")).get("a0_pass") is not True:
        raise RuntimeError("STOP_A0_NOT_PASSED")
    phases, donors = load_registry()
    validity = registry_validity_rows(phases, donors)
    selected_phases = [phase for index, phase in enumerate(phases) if index % args.num_shards == args.shard_index]
    if not selected_phases:
        raise RuntimeError(f"EMPTY_SHARD:{args.shard_index}/{args.num_shards}")
    shard_root = args.output / "shards" / f"shard_{args.shard_index:02d}_of_{args.num_shards:02d}"
    shard_root.mkdir(parents=True, exist_ok=True)
    shard_state_keys = {(int(row["task_id"]), int(row["source_state_id"])) for row in selected_phases}
    write_csv(shard_root / "validity.csv", [row for row in validity if (int(row["task_id"]), int(row["source_state_id"])) in shard_state_keys])
    visual_manifest = args.output / "visualization/visualization_manifest.json"
    if not visual_manifest.is_file():
        raise RuntimeError("STOP_VISUALIZATION_AMENDMENT_NOT_FROZEN")
    torch.set_num_threads(args.threads)
    runner = make_capture("joint", args.gpu)
    before_hash = model_weight_hash(runner.model)
    manifest = run_metadata(runner, args.gpu)
    manifest.update({
        "stage": "FORMAL_RUNNING",
        "physical_gpu_id": args.physical_gpu,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "shard_assignment_rule": "sorted frozen phase registry index modulo num_shards",
        "cross_shard_case_overlap_permitted": False,
        "a0_report_sha256": sha256_file(a0_path),
        "visualization_manifest_sha256": sha256_file(visual_manifest),
        "phase_registry_sha256": sha256_file(G1 / "phase_registry.jsonl"),
        "donor_registry_sha256": sha256_file(G1 / "donor_bank/joint/counterfactual_qc.jsonl"),
        "locus_registry_sha256": sha256_file(REGISTRY),
        "registered_states_global": len(phases),
        "registered_states_this_shard": len(selected_phases),
        "frozen_valid_donors_this_shard": sum(row["donor_valid"] is True for row in validity if (int(row["task_id"]), int(row["source_state_id"])) in shard_state_keys),
        "formal_started_unix": time.time(),
    })
    atomic_json(shard_root / "run_manifest.json", manifest)
    completed = []
    for phase in selected_phases:
        result = run_formal_state(runner, phase, donors, args.output)
        completed.append(result)
        atomic_json(shard_root / "formal_progress.json", {
            "completed_states": len(completed), "total_registered_states_this_shard": len(selected_phases),
            "last_base_state_id": phase["base_state_id"],
            "completed_strict_configurations": sum(int(row.get("strict_configurations_executed", 0)) for row in completed),
        })
    after_hash = model_weight_hash(runner.model)
    manifest.update({
        "stage": "FORMAL_FORWARD_COMPLETE",
        "formal_finished_unix": time.time(),
        "weight_hash_before": before_hash,
        "weight_hash_after": after_hash,
        "model_weight_hash_unchanged": before_hash == after_hash,
        "completed_states": len(completed),
        "technical_failure_states": sum(row["status"] != "COMPLETED" for row in completed),
        "completed_strict_configurations": sum(int(row.get("strict_configurations_executed", 0)) for row in completed),
    })
    atomic_json(shard_root / "run_manifest.json", manifest)
    print(json.dumps({"status": manifest["stage"], "shard": args.shard_index, "completed_states": len(completed), "output": str(shard_root)}, ensure_ascii=False))


def run_merge(args: argparse.Namespace) -> None:
    phases, donors = load_registry()
    manifests = []
    for shard in range(args.num_shards):
        path = args.output / "shards" / f"shard_{shard:02d}_of_{args.num_shards:02d}" / "run_manifest.json"
        if not path.is_file():
            raise RuntimeError(f"STOP_MISSING_SHARD_MANIFEST:{path}")
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("stage") != "FORMAL_FORWARD_COMPLETE":
            raise RuntimeError(f"STOP_SHARD_NOT_COMPLETE:{path}:{row.get('stage')}")
        manifests.append((path, row))
    result_rows = []
    seen: set[tuple[int, int]] = set()
    for phase in phases:
        task, state = int(phase["task_id"]), int(phase["source_state_id"])
        path = state_output_dir(args.output, phase) / "complete.json"
        if not path.is_file():
            raise RuntimeError(f"STOP_MISSING_CASE:{task}:{state}")
        result = json.loads(path.read_text(encoding="utf-8"))
        key = (task, state)
        if key in seen:
            raise RuntimeError(f"STOP_DUPLICATE_CASE:{key}")
        seen.add(key)
        result_rows.append({
            "task_id": task, "source_state_id": state, "base_state_id": phase["base_state_id"],
            "status": result["status"], "strict_configurations_executed": result["strict_configurations_executed"],
            "natural_sources_executed": result["natural_sources_executed"],
            "model_inference_seconds": result.get("model_inference_seconds", ""),
            "artifact_write_seconds": result.get("artifact_write_seconds", ""),
            "case_complete_path": str(path), "case_complete_sha256": sha256_file(path),
            "actions_path": str(path.parent / "actions.npz"), "actions_sha256": result["actions_sha256"],
        })
    validity = registry_validity_rows(phases, donors)
    write_csv(args.output / "validity.csv", validity)
    write_csv(args.output / "all_result_index.csv", result_rows)
    manifest = {
        "status": "FORMAL_FORWARD_COMPLETE_MERGED",
        "protocol_version": PROTOCOL_VERSION,
        "shard_count": args.num_shards,
        "shard_manifests": [{"path": str(path), "sha256": sha256_file(path), "physical_gpu_id": row.get("physical_gpu_id")} for path, row in manifests],
        "registered_states": len(phases), "completed_unique_states": len(seen),
        "no_duplicate_cases": len(seen) == len(phases),
        "technical_failure_states": sum(row["status"] != "COMPLETED" for row in result_rows),
        "completed_strict_configurations": sum(int(row["strict_configurations_executed"]) for row in result_rows),
        "model_inference_seconds_sum_across_gpus": sum(float(row["model_inference_seconds"]) for row in result_rows),
        "artifact_write_seconds_sum_across_gpus": sum(float(row["artifact_write_seconds"]) for row in result_rows),
        "all_result_index_sha256": sha256_file(args.output / "all_result_index.csv"),
        "validity_sha256": sha256_file(args.output / "validity.csv"),
        "visualization_manifest_sha256": sha256_file(args.output / "visualization/visualization_manifest.json"),
        "environment_actions_executed": 0,
        "closed_loop_video_status": "NOT_APPLICABLE_PROTOCOL_PROHIBITS_CLOSED_LOOP",
    }
    atomic_json(args.output / "run_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("a0", "visuals", "formal", "merge"))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--physical-gpu", type=int, default=None)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--output", type=Path, default=OUT_DEFAULT)
    args = parser.parse_args()
    if args.mode == "a0":
        run_a0(args)
    elif args.mode == "visuals":
        run_visuals(args)
    elif args.mode == "merge":
        run_merge(args)
    else:
        if not 0 <= args.shard_index < args.num_shards:
            parser.error("--shard-index must be in [0, --num-shards)")
        run_formal(args)


if __name__ == "__main__":
    main()

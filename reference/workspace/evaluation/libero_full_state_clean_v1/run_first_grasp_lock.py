#!/usr/bin/env python3
from __future__ import annotations
from wam_causal_audit.paths import resolve as _release_path

"""LIBERO first-grasp-lock intervention.

This runner deliberately reuses the clean policy adapters and full-state
writer.  It adds only a finite target-free-joint lock, per-env-step two-camera
video, and LingBot predicted-video collection.
"""

import argparse
import csv
import fcntl
import json
import os
import re
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import imageio.v2 as imageio
import numpy as np

HERE = Path(__file__).resolve().parent
for path in (HERE, Path(_release_path('@WORKSPACE@/LIBERO')), Path(_release_path('@WORKSPACE@/evaluation/d2_ghost_pilot'))):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from d2_pilot_utils import current_raw_obs  # noqa: E402
from first_grasp_lock_injector import FirstGraspLockInjector  # noqa: E402
from full_state_writer import FullStateRolloutWriter  # noqa: E402
from policy_adapters import LingBotAdapter, PolicyAdapter, make_adapter  # noqa: E402
from run_clean import ENVIRONMENT_SEED, _env_step, _lingbot_keyframe  # noqa: E402
from run_occ_still import resolve_target_body_name, resolve_target_free_joint_name  # noqa: E402
from task_specs import TASK_SPECS  # noqa: E402


CLEAN_ROOT = Path(_release_path('@WORKSPACE@/results/libero_full_state_clean_v1'))
DEFAULT_OUT = Path(_release_path('@WORKSPACE@/results/libero_first_grasp_lock_v1'))
SUMMARY_FIELDS = (
    "model", "task_id", "seed", "first_grasp_failed", "blind_continuation", "retry",
    "regrasp_success", "final_success", "retry_latency",
)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def clean_seeds(model: str, task_id: int) -> list[int]:
    payload = json.loads((CLEAN_ROOT / "clean_summary.json").read_text(encoding="utf-8"))
    seeds = [int(x) for x in payload["combinations"][f"{model}/task{task_id:02d}"]["success_seed_pool"]]
    if len(seeds) != 10:
        raise RuntimeError(f"expected 10 successful clean task{task_id} seeds for {model}, got {seeds}")
    return seeds


def paired_clean_dir(model: str, task_id: int, seed: int) -> Path:
    return CLEAN_ROOT / "clean" / model / f"task{task_id:02d}" / f"init_{seed:03d}"


def first_grasp_target(model: str, task_id: int, seed: int) -> tuple[str, dict[str, Any]]:
    """Resolve the first manipulated object from the paired clean trajectory.

    Task0 manipulates two objects.  Using its clean phase segmentation avoids
    silently locking the second object if task ordering ever changes.
    """
    spec = TASK_SPECS[task_id]
    phase_path = paired_clean_dir(model, task_id, seed) / "phases.json"
    if not phase_path.exists():
        raise RuntimeError(f"paired clean phase file is missing: {phase_path}")
    phases = json.loads(phase_path.read_text(encoding="utf-8"))
    diagnostics = phases.get("diagnostics", {})
    object_order = diagnostics.get("object_order") or []
    grasp_1 = diagnostics.get("grasp_1") or diagnostics.get("grasp") or {}
    target = grasp_1.get("entity") or (object_order[0] if object_order else None)
    if target not in spec.manipulated_objects:
        raise RuntimeError(
            f"paired clean first-grasp target {target!r} is not in task{task_id} "
            f"manipulated objects {spec.manipulated_objects}"
        )
    if object_order and target != object_order[0]:
        raise RuntimeError(
            f"paired clean phase disagreement: grasp_1={target!r}, object_order={object_order}"
        )
    return str(target), {
        "phase_path": str(phase_path),
        "grasp_1_entity": target,
        "object_order": object_order,
    }


def stack_cameras(raw_obs: Mapping[str, Any]) -> np.ndarray:
    agent = np.ascontiguousarray(np.asarray(raw_obs["agentview_image"], dtype=np.uint8)[::-1])
    wrist = np.ascontiguousarray(np.asarray(raw_obs["robot0_eye_in_hand_image"], dtype=np.uint8)[::-1])
    return np.hstack([agent[..., :3], wrist[..., :3]])


class SimulationVideo:
    def __init__(self, path: Path, fps: int = 20) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=8)
        self.frames = 0

    def append(self, raw_obs: Mapping[str, Any]) -> None:
        self.writer.append_data(stack_cameras(raw_obs))
        self.frames += 1

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None


def numeric_video_key(path: Path) -> int:
    match = re.search(r"(\d+)$", path.stem)
    return int(match.group(1)) if match else -1


def collect_lingbot_predictions(
    *, server_save_root: Path, trace_condition: str, seed: int,
    task_id: int, rollout_dir: Path, expected_calls: int, wait_seconds: float = 30.0,
) -> tuple[str | None, dict[str, Any]]:
    source = server_save_root / "real" / f"task{task_id:02d}" / trace_condition / f"ep{seed:03d}_seed{seed}"
    deadline = time.time() + float(wait_seconds)
    files: list[Path] = []
    while time.time() < deadline:
        files = sorted(source.glob("pred_video_*.mp4"), key=numeric_video_key)
        if len(files) >= expected_calls:
            break
        time.sleep(0.5)
    dest_dir = rollout_dir / "predicted_calls"
    dest_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for index, source_file in enumerate(files):
        dest = dest_dir / f"call_{index:04d}_{source_file.name}"
        shutil.copy2(source_file, dest)
        copied.append(dest)
    output = rollout_dir / "predicted.mp4"
    frame_count = 0
    if copied:
        writer = imageio.get_writer(output, fps=10, codec="libx264", quality=8)
        try:
            for path in copied:
                reader = imageio.get_reader(path)
                try:
                    for frame in reader:
                        writer.append_data(np.asarray(frame, dtype=np.uint8))
                        frame_count += 1
                finally:
                    reader.close()
        finally:
            writer.close()
    audit = {
        "server_source": str(source),
        "expected_policy_calls": int(expected_calls),
        "chunk_videos_found": len(files),
        "chunk_videos_copied": len(copied),
        "concatenated_frames": frame_count,
        "complete": bool(len(copied) == expected_calls and frame_count > 0 and output.exists()),
    }
    return (str(output) if output.exists() else None), audit


def upsert_summary(root: Path, result: Mapping[str, Any]) -> None:
    path = root / "summary.csv"
    lock_path = root / ".summary.csv.lock"
    root.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        rows: list[dict[str, str]] = []
        if path.exists():
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
        # Migrate the original task5-only summary in place.
        for row in rows:
            if not row.get("task_id"):
                row["task_id"] = "5"
        key = (str(result["model"]), str(result["task_id"]), str(result["seed"]))
        rows = [row for row in rows if (row.get("model"), row.get("task_id"), row.get("seed")) != key]
        event = result["event"]
        rows.append({
            "model": str(result["model"]), "task_id": str(result["task_id"]), "seed": str(result["seed"]),
            "first_grasp_failed": str(bool(event["first_grasp_failed"])),
            "blind_continuation": str(bool(event["blind_continuation"])),
            "retry": str(bool(event["retry"])),
            "regrasp_success": str(bool(event["regrasp_success"])),
            "final_success": str(bool(event["final_success"])),
            "retry_latency": "" if event["retry_latency"] is None else str(event["retry_latency"]),
        })
        rows.sort(key=lambda row: (row["model"], int(row["task_id"]), int(row["seed"])))
        temporary = root / f".summary.csv.{os.getpid()}.tmp"
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
            writer.writeheader(); writer.writerows(rows)
        os.replace(temporary, path)


def run_attempt(
    *, adapter: PolicyAdapter, suite: Any, task_id: int, seed: int, attempt: int,
    output_root: Path, physical_gpu: int, overwrite: bool,
    lingbot_server_save_root: Path | None,
) -> dict[str, Any]:
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    task = suite.get_task(task_id)
    spec = TASK_SPECS[task_id]
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    seed_root = output_root / "first_grasp_lock" / adapter.model_name / f"task{task_id:02d}" / f"init_{seed:03d}"
    rollout_dir = seed_root / f"attempt_{attempt:02d}"
    result_path = rollout_dir / "result.json"
    if result_path.exists() and not overwrite:
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if bool(existing.get("validation_valid")):
            return existing

    env = None
    writer = None
    video = None
    injector = None
    success = False
    termination = "not_started"
    error_text = None
    calls = 0
    policy_steps = 0
    started = time.time()
    trace_condition = f"first_grasp_lock_rngpaired_v2_attempt{attempt:02d}"
    prediction_audit: dict[str, Any] | None = None
    target_audit: dict[str, Any] | None = None
    predicted_video: str | None = None
    simulation_video: str | None = None
    try:
        print(json.dumps({"progress": "create_env", "model": adapter.model_name, "task_id": task_id, "seed": seed, "attempt": attempt}), flush=True)
        env = OffScreenRenderEnv(
            bddl_file_name=bddl,
            camera_heights=adapter.render_resolution,
            camera_widths=adapter.render_resolution,
        )
        env.seed(ENVIRONMENT_SEED)
        writer = FullStateRolloutWriter(
            rollout_dir, model=adapter.model_name, task_id=task_id, init_state_index=seed,
            task_description=task.language, task_spec=spec, bddl_file=bddl,
            environment_seed=ENVIRONMENT_SEED, physical_gpu=physical_gpu,
            image_width=adapter.render_resolution, image_height=adapter.render_resolution,
            condition="first_grasp_lock",
        )
        writer.prepare(overwrite=overwrite or (rollout_dir / "steps.parquet").exists())
        video = SimulationVideo(rollout_dir / "simulation.mp4")
        print(json.dumps({"progress": "reset_env", "model": adapter.model_name, "seed": seed, "attempt": attempt}), flush=True)
        env.reset()
        raw_obs = dict(env.set_init_state(suite.get_task_init_states(task_id)[seed]))
        target_object, target_audit = first_grasp_target(adapter.model_name, task_id, seed)
        target_body = resolve_target_body_name(env, target_object)
        target_joint = resolve_target_free_joint_name(env, target_object, target_body)
        injector = FirstGraspLockInjector(
            target_object_name=target_object, target_body_name=target_body,
            target_joint_name=target_joint,
        )
        writer.record_state(env_step=_env_step(env), env=env, raw_obs=raw_obs,
                            action_from_previous_step=None, reward=None, done=False, phase="pre_grasp")
        video.append(raw_obs)
        for _ in range(adapter.num_steps_wait):
            raw_obs, reward, done, _ = env.step(list(adapter.dummy_action)); raw_obs = dict(raw_obs)
            writer.record_state(env_step=_env_step(env), env=env, raw_obs=raw_obs,
                                action_from_previous_step=adapter.dummy_action, reward=reward,
                                done=bool(done), phase="pre_grasp")
            video.append(raw_obs)
            if done:
                success, termination = True, "success_during_settle"
                break
        if not success:
            print(json.dumps({"progress": "reset_policy", "model": adapter.model_name, "seed": seed, "attempt": attempt}), flush=True)
            adapter.reset(task.language, {
                "experiment": "libero_first_grasp_lock_v1", "model": adapter.model_name,
                "task": f"task{task_id:02d}", "condition": trace_condition,
                "episode": seed, "seed": seed, "attempt": attempt, "physical_gpu": physical_gpu,
            })

        while not success and policy_steps < adapter.max_policy_steps:
            call_step = _env_step(env)
            print(json.dumps({"progress": "policy_call", "call": calls, "env_step": call_step,
                              "lock_state": injector.state}), flush=True)
            infer_started = time.perf_counter()
            output = adapter.infer(raw_obs, call_idx=calls, env_step=call_step, first_call=(calls == 0))
            infer_latency_ms = (time.perf_counter() - infer_started) * 1000.0
            writer.record_policy_call(
                call_idx=calls, env_step=call_step, env=env, raw_obs=raw_obs,
                action_chunk=output.action_chunk, infer_latency_ms=infer_latency_ms,
                model_input_metadata=output.model_input_metadata,
                executed_action_indices=output.executed_action_indices,
                latents_video=output.latents_video, latent_alignment=output.latent_alignment,
                extra={**output.extra, "first_grasp_lock_state": injector.state,
                       "trigger_env_step": injector.trigger_env_step,
                       "release_env_step": injector.release_env_step,
                       "policy_rng_seed": getattr(adapter, "policy_rng_seed", None)},
            )
            keyframes: list[Mapping[str, np.ndarray]] = []
            chunk_complete = True
            for offset, action in enumerate(output.executable_actions):
                if policy_steps >= adapter.max_policy_steps:
                    chunk_complete = False
                    break
                injector.before_action(env, raw_obs, action, env_step=_env_step(env))
                raw_obs, reward, done, _ = env.step(np.asarray(action, dtype=np.float32).reshape(-1)[:7].tolist())
                raw_obs = dict(raw_obs)
                policy_steps += 1
                state_before = injector.state
                injector.after_step(env, raw_obs, action, env_step=_env_step(env))
                if state_before == "locked":
                    # env.step observations precede the exact qpos re-commit.
                    raw_obs = current_raw_obs(env)
                phase = "first_grasp_locked" if injector.state == "locked" else (
                    "post_release" if injector.state == "released" else "pre_grasp"
                )
                writer.record_state(env_step=_env_step(env), env=env, raw_obs=raw_obs,
                                    action_from_previous_step=action, reward=reward,
                                    done=bool(done), phase=phase)
                video.append(raw_obs)
                if isinstance(adapter, LingBotAdapter):
                    keyframe = _lingbot_keyframe(output, offset, raw_obs)
                    if keyframe is not None:
                        keyframes.append(keyframe)
                if done:
                    success, termination, chunk_complete = True, "success", False
                    break
            if chunk_complete:
                adapter.after_chunk(output, keyframes, done=False)
            calls += 1
        if not success:
            termination = "no_first_grasp_trigger" if injector.trigger_env_step is None else (
                "no_release" if injector.release_env_step is None else "timeout"
            )
    except Exception as exc:
        termination = f"exception:{type(exc).__name__}"
        error_text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        print(error_text, file=sys.stderr, flush=True)
    finally:
        if video is not None:
            video.close()
            if video.frames > 0 and video.path.exists():
                simulation_video = str(video.path)
        if isinstance(adapter, LingBotAdapter) and lingbot_server_save_root is not None:
            try:
                predicted_video, prediction_audit = collect_lingbot_predictions(
                    server_save_root=lingbot_server_save_root, trace_condition=trace_condition,
                    seed=seed, task_id=task_id, rollout_dir=rollout_dir, expected_calls=calls,
                )
            except Exception as exc:
                prediction_audit = {"complete": False, "error": repr(exc)}
        if injector is None:
            event: dict[str, Any] = {
                "schema_version": "libero_first_grasp_lock_v1_event",
                "validation": {"valid": False}, "setup_error": True,
            }
        else:
            event = injector.final_event(
                final_success=success, simulation_video=simulation_video,
                predicted_video=predicted_video,
            )
        if isinstance(adapter, LingBotAdapter):
            prediction_ok = bool(prediction_audit and prediction_audit.get("complete"))
            event["validation"]["lingbot_predicted_video_complete"] = prediction_ok
            event["validation"]["valid"] = bool(event["validation"].get("valid") and prediction_ok)
            event["prediction_video_audit"] = prediction_audit
        clean_dir = paired_clean_dir(adapter.model_name, task_id, seed)
        paired_clean_meta = clean_dir / "meta.json"
        paired_clean_success = False
        if paired_clean_meta.exists():
            paired_clean_success = bool(json.loads(paired_clean_meta.read_text(encoding="utf-8")).get("success"))
        policy_rng_seed = getattr(adapter, "policy_rng_seed", None)
        rng_seed_matches = policy_rng_seed is not None and int(policy_rng_seed) == int(seed)
        event["policy_rng_seed"] = policy_rng_seed
        event["paired_clean_seed"] = int(seed)
        event["paired_clean_rollout"] = str(clean_dir)
        event["first_grasp_target"] = None if injector is None else injector.target_object_name
        event["first_grasp_target_audit"] = target_audit
        event["validation"]["policy_rng_seed_matches_clean_seed"] = bool(rng_seed_matches)
        event["validation"]["paired_clean_rollout_exists"] = bool(paired_clean_meta.exists())
        event["validation"]["paired_clean_rollout_success"] = bool(paired_clean_success)
        event["validation"]["valid"] = bool(
            event["validation"].get("valid")
            and rng_seed_matches
            and paired_clean_meta.exists()
            and paired_clean_success
        )
        event["termination"] = termination
        event["exception_traceback"] = error_text
        if writer is not None and writer._rows:
            writer.finish(
                success=success, termination=termination,
                model_metadata=adapter.model_metadata(), policy_protocol=adapter.policy_protocol(),
                extra_meta={
                    "first_grasp_lock": event, "policy_steps": policy_steps,
                    "pairing": {
                        "paired_clean_seed": int(seed),
                        "paired_clean_rollout": str(clean_dir),
                        "policy_rng_seed": policy_rng_seed,
                        "only_intended_difference": "finite first_grasp_lock intervention",
                    },
                    "wall_time_s": time.time() - started, "exception_traceback": error_text,
                    "first_grasp_lock_video_paths": {
                        "simulation_video": simulation_video, "predicted_video": predicted_video,
                    },
                },
            )
        rollout_dir.mkdir(parents=True, exist_ok=True)
        write_json(rollout_dir / "event.json", event)
        result = {
            "schema_version": "libero_first_grasp_lock_v1_result",
            "model": adapter.model_name, "task_id": task_id, "seed": seed, "attempt": attempt,
            "success": success, "termination": termination, "policy_steps": policy_steps,
            "policy_calls": calls, "wall_time_s": time.time() - started,
            "rollout_dir": str(rollout_dir), "event_path": str(rollout_dir / "event.json"),
            "validation_valid": bool(event.get("validation", {}).get("valid")), "event": event,
        }
        write_json(result_path, result)
        write_json(seed_root / "latest_result.json", result)
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=("lingbot_va", "fastwam", "pi05", "vla_jepa"))
    parser.add_argument("--tasks", default="5", help="comma-separated LIBERO-10 task ids; default: 5")
    parser.add_argument("--seeds", help="comma-separated clean-success init indices; default is each task's full pool")
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--lingbot-server-save-root", type=Path)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    if args.model == "lingbot_va" and args.lingbot_server_save_root is None:
        raise ValueError("LingBot requires --lingbot-server-save-root so predicted videos can be collected and verified")
    from libero.libero import benchmark
    suite = benchmark.get_benchmark_dict()["libero_10"]()
    print(json.dumps({"progress": "create_adapter", "model": args.model, "port": args.port}), flush=True)
    adapter = make_adapter(args.model, args.host, args.port)
    print(json.dumps({"progress": "adapter_ready", "model": adapter.model_name}), flush=True)
    task_ids = [int(x) for x in args.tasks.split(",")]
    if not task_ids or any(task_id not in TASK_SPECS for task_id in task_ids):
        raise ValueError(f"tasks must be selected from {sorted(TASK_SPECS)}")
    for task_id in task_ids:
        pool = clean_seeds(adapter.model_name, task_id)
        seeds = [int(x) for x in args.seeds.split(",")] if args.seeds else pool
        if any(seed not in pool for seed in seeds):
            raise ValueError(f"requested task{task_id} seed is outside the clean-success pool {pool}")
        for seed in seeds:
            accepted = None
            for attempt in range(args.max_attempts):
                result = run_attempt(
                    adapter=adapter, suite=suite, task_id=task_id, seed=seed, attempt=attempt,
                    output_root=args.output_root, physical_gpu=args.gpu,
                    overwrite=args.overwrite,
                    lingbot_server_save_root=args.lingbot_server_save_root,
                )
                print(json.dumps({k: result[k] for k in ("model", "task_id", "seed", "attempt", "success", "termination", "validation_valid")}, sort_keys=True), flush=True)
                if result["validation_valid"]:
                    accepted = result
                    break
            if accepted is None:
                print(json.dumps({"model": adapter.model_name, "task_id": task_id, "seed": seed, "status": "invalid_after_max_attempts"}), flush=True)
            else:
                upsert_summary(args.output_root, accepted)


if __name__ == "__main__":
    main()

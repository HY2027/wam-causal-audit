#!/usr/bin/env python3
from __future__ import annotations
from wam_causal_audit.paths import resolve as _release_path

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping

import imageio.v2 as imageio
import numpy as np

ROOT = Path(__file__).resolve().parent
LIBERO_ROOT = Path(_release_path('@WORKSPACE@/LIBERO'))
for path in (ROOT, LIBERO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from full_state_writer import FullStateRolloutWriter  # noqa: E402
from policy_adapters import LingBotAdapter, PolicyAdapter, PolicyOutput, make_adapter  # noqa: E402
from task_specs import TASK_SPECS  # noqa: E402


DEFAULT_OUTPUT_ROOT = Path(_release_path('@WORKSPACE@/results/libero_full_state_clean_v1'))
ENVIRONMENT_SEED = 7


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, path)


def _env_step(env: Any) -> int:
    return int(getattr(getattr(env, "env", env), "timestep"))


def _lingbot_keyframe(output: PolicyOutput, executable_offset: int, raw_obs: Mapping[str, Any]) -> dict[str, np.ndarray] | None:
    raw_shape = tuple(int(x) for x in output.extra["raw_action_shape"])
    token_count = raw_shape[2]
    action_per_frame = int(output.extra["action_per_frame"])
    flat_index = int(output.executed_action_indices[executable_offset])
    token_index = flat_index % token_count
    if (token_index + 1) % action_per_frame != 0:
        return None
    return {
        "observation.images.agentview_rgb": np.ascontiguousarray(np.asarray(raw_obs["agentview_image"])[::-1]),
        "observation.images.eye_in_hand_rgb": np.ascontiguousarray(np.asarray(raw_obs["robot0_eye_in_hand_image"])[::-1]),
    }


def _stack_simulation_cameras(raw_obs: Mapping[str, Any]) -> np.ndarray:
    agent = np.ascontiguousarray(np.asarray(raw_obs["agentview_image"], dtype=np.uint8)[::-1])
    wrist = np.ascontiguousarray(np.asarray(raw_obs["robot0_eye_in_hand_image"], dtype=np.uint8)[::-1])
    return np.hstack([agent[..., :3], wrist[..., :3]])


def run_episode(
    *,
    adapter: PolicyAdapter,
    suite: Any,
    task_id: int,
    init_state_index: int,
    output_root: Path,
    physical_gpu: int,
    overwrite: bool,
    save_simulation_video: bool = False,
    trace_experiment: str = "libero_full_state_clean_v1",
    trace_condition: str = "clean",
) -> dict[str, Any]:
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    task = suite.get_task(task_id)
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    rollout_dir = output_root / "clean" / adapter.model_name / f"task{task_id:02d}" / f"init_{init_state_index:03d}"
    env = OffScreenRenderEnv(
        bddl_file_name=bddl,
        camera_heights=adapter.render_resolution,
        camera_widths=adapter.render_resolution,
    )
    env.seed(ENVIRONMENT_SEED)
    writer = FullStateRolloutWriter(
        rollout_dir,
        model=adapter.model_name,
        task_id=task_id,
        init_state_index=init_state_index,
        task_description=task.language,
        task_spec=TASK_SPECS[task_id],
        bddl_file=bddl,
        environment_seed=ENVIRONMENT_SEED,
        physical_gpu=physical_gpu,
        image_width=adapter.render_resolution,
        image_height=adapter.render_resolution,
    )
    writer.prepare(overwrite=overwrite)
    simulation_path = rollout_dir / "simulation.mp4"
    simulation_writer = (
        imageio.get_writer(simulation_path, fps=20, codec="libx264", quality=8)
        if save_simulation_video else None
    )
    success = False
    termination = "not_started"
    error_text = None
    call_idx = 0
    policy_steps = 0
    started = time.time()
    try:
        env.reset()
        raw_obs = dict(env.set_init_state(suite.get_task_init_states(task_id)[init_state_index]))
        if simulation_writer is not None:
            simulation_writer.append_data(_stack_simulation_cameras(raw_obs))
        writer.record_state(
            env_step=_env_step(env),
            env=env,
            raw_obs=raw_obs,
            action_from_previous_step=None,
            reward=None,
            done=False,
        )
        for _ in range(adapter.num_steps_wait):
            raw_obs, reward, done, _ = env.step(list(adapter.dummy_action))
            raw_obs = dict(raw_obs)
            if simulation_writer is not None:
                simulation_writer.append_data(_stack_simulation_cameras(raw_obs))
            writer.record_state(
                env_step=_env_step(env),
                env=env,
                raw_obs=raw_obs,
                action_from_previous_step=adapter.dummy_action,
                reward=reward,
                done=bool(done),
            )
            if done:
                success = True
                termination = "success_during_settle"
                break
        if not success:
            trace_meta = {
                "experiment": str(trace_experiment),
                "model": adapter.model_name,
                "task": f"task{task_id:02d}",
                "condition": str(trace_condition),
                "episode": init_state_index,
                "seed": init_state_index,
                "physical_gpu": physical_gpu,
            }
            adapter.reset(task.language, trace_meta)

        while not success and policy_steps < adapter.max_policy_steps:
            call_env_step = _env_step(env)
            infer_started = time.perf_counter()
            output = adapter.infer(raw_obs, call_idx=call_idx, env_step=call_env_step, first_call=(call_idx == 0))
            infer_latency_ms = (time.perf_counter() - infer_started) * 1000.0
            writer.record_policy_call(
                call_idx=call_idx,
                env_step=call_env_step,
                env=env,
                raw_obs=raw_obs,
                action_chunk=output.action_chunk,
                infer_latency_ms=infer_latency_ms,
                model_input_metadata=output.model_input_metadata,
                executed_action_indices=output.executed_action_indices,
                latents_video=output.latents_video,
                latent_alignment=output.latent_alignment,
                extra=output.extra,
            )
            keyframes: list[Mapping[str, np.ndarray]] = []
            chunk_complete = True
            for executable_offset, action in enumerate(output.executable_actions):
                if policy_steps >= adapter.max_policy_steps:
                    chunk_complete = False
                    break
                raw_obs, reward, done, _ = env.step(np.asarray(action, dtype=np.float32).reshape(-1)[:7].tolist())
                raw_obs = dict(raw_obs)
                if simulation_writer is not None:
                    simulation_writer.append_data(_stack_simulation_cameras(raw_obs))
                policy_steps += 1
                writer.record_state(
                    env_step=_env_step(env),
                    env=env,
                    raw_obs=raw_obs,
                    action_from_previous_step=action,
                    reward=reward,
                    done=bool(done),
                )
                if isinstance(adapter, LingBotAdapter):
                    keyframe = _lingbot_keyframe(output, executable_offset, raw_obs)
                    if keyframe is not None:
                        keyframes.append(keyframe)
                if done:
                    success = True
                    termination = "success"
                    chunk_complete = False
                    break
            if chunk_complete:
                adapter.after_chunk(output, keyframes, done=False)
            call_idx += 1
        if not success:
            termination = "timeout"
    except Exception as exc:
        termination = f"exception:{type(exc).__name__}"
        error_text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        print(error_text, file=sys.stderr, flush=True)
    finally:
        if simulation_writer is not None:
            simulation_writer.close()
        if writer._rows:
            writer.finish(
                success=success,
                termination=termination,
                model_metadata=adapter.model_metadata(),
                policy_protocol=adapter.policy_protocol(),
                extra_meta={
                    "policy_steps": policy_steps,
                    "wall_time_s": time.time() - started,
                    "exception_traceback": error_text,
                    "simulation_video": str(simulation_path) if simulation_path.exists() else None,
                },
            )
        try:
            env.close()
        except Exception:
            pass
    result = {
        "model": adapter.model_name,
        "task_id": task_id,
        "init_state_index": init_state_index,
        "success": success,
        "termination": termination,
        "policy_steps": policy_steps,
        "policy_calls": call_idx,
        "wall_time_s": time.time() - started,
        "simulation_video": str(simulation_path) if simulation_path.exists() else None,
        "rollout_dir": str(rollout_dir),
    }
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def _load_existing(output_root: Path, model: str, task_id: int, index: int) -> dict[str, Any] | None:
    rollout_dir = output_root / "clean" / model / f"task{task_id:02d}" / f"init_{index:03d}"
    meta_path = rollout_dir / "meta.json"
    steps_path = rollout_dir / "steps.parquet"
    calls_path = rollout_dir / "policy_calls.jsonl"
    if not (meta_path.exists() and steps_path.exists() and calls_path.exists()):
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return {
        "model": model,
        "task_id": task_id,
        "init_state_index": index,
        "success": bool(meta["success"]),
        "termination": meta["termination"],
        "policy_steps": int(meta.get("policy_steps", 0)),
        "policy_calls": int(meta.get("num_policy_calls", 0)),
        "wall_time_s": float(meta.get("wall_time_s", 0.0)),
        "rollout_dir": str(rollout_dir),
        "resumed": True,
    }


def update_summary(output_root: Path, records: list[dict[str, Any]], target_successes: int, max_init_index: int) -> dict[str, Any]:
    by_combo: dict[str, Any] = {}
    for record in sorted(records, key=lambda row: (row["model"], row["task_id"], row["init_state_index"])):
        key = f"{record['model']}/task{record['task_id']:02d}"
        combo = by_combo.setdefault(key, {"model": record["model"], "task_id": record["task_id"], "attempts": []})
        combo["attempts"].append(record)
    for combo in by_combo.values():
        attempts = combo["attempts"]
        pool = [int(row["init_state_index"]) for row in attempts if row["success"]][:target_successes]
        combo["success_seed_pool"] = pool
        combo["num_attempted"] = len(attempts)
        combo["num_successes"] = sum(bool(row["success"]) for row in attempts)
        combo["clean_success_rate"] = combo["num_successes"] / len(attempts) if attempts else 0.0
        combo["eligible"] = len(pool) >= target_successes
        combo["exclusion_reason"] = None if combo["eligible"] else f"fewer than {target_successes} successes through init_state_index {max_init_index}"
    payload = {
        "schema_version": "libero_full_state_clean_v1_summary",
        "protocol": {
            "target_successes_per_model_task": target_successes,
            "init_state_scan": [0, max_init_index],
            "environment_seed": ENVIRONMENT_SEED,
            "pairing": "within each model/task-specific successful seed pool",
            "perturbation_experiments_run": False,
            "allowed_physical_gpus": [2, 3],
            "model_specific_gpu_exception": {"lingbot_va": [1, 2, 3]},
        },
        "combinations": by_combo,
    }
    _write_json(output_root / "clean_summary.json", payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["pi05", "fastwam", "lingbot_va", "vla_jepa"])
    parser.add_argument("--tasks", default="0,2,3,5,9")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--gpu", type=int, required=True, choices=[1, 2, 3])
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--target-successes", type=int, default=10)
    parser.add_argument("--max-init-index", type=int, default=13)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--only-init-index", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    if args.gpu == 1 and args.model != "lingbot_va":
        raise ValueError("physical GPU 1 is allowed only for LingBot-VA; other models remain restricted to GPU 2/3")
    from libero.libero import benchmark

    suite = benchmark.get_benchmark_dict()["libero_10"]()
    adapter = make_adapter(args.model, args.host, args.port)
    all_records: list[dict[str, Any]] = []
    existing_summary = args.output_root / "clean_summary.json"
    if existing_summary.exists() and not args.overwrite:
        old = json.loads(existing_summary.read_text(encoding="utf-8"))
        for combo in old.get("combinations", {}).values():
            all_records.extend(combo.get("attempts", []))
    seen = {(row["model"], int(row["task_id"]), int(row["init_state_index"])) for row in all_records}
    task_ids = [int(x) for x in args.tasks.split(",")]
    for task_id in task_ids:
        if task_id not in TASK_SPECS:
            raise ValueError(f"unsupported task {task_id}")
        successes = 0
        indices = [args.only_init_index] if args.only_init_index is not None else list(range(args.max_init_index + 1))
        for init_state_index in indices:
            existing = None if args.overwrite else _load_existing(args.output_root, adapter.model_name, task_id, init_state_index)
            if existing is not None:
                record = existing
            else:
                record = run_episode(
                    adapter=adapter,
                    suite=suite,
                    task_id=task_id,
                    init_state_index=init_state_index,
                    output_root=args.output_root,
                    physical_gpu=args.gpu,
                    overwrite=args.overwrite,
                )
            key = (adapter.model_name, task_id, init_state_index)
            if key not in seen:
                all_records.append(record)
                seen.add(key)
            else:
                all_records = [row for row in all_records if (row["model"], int(row["task_id"]), int(row["init_state_index"])) != key]
                all_records.append(record)
            update_summary(args.output_root, all_records, args.target_successes, args.max_init_index)
            successes += int(bool(record["success"]))
            if args.only_init_index is None and successes >= args.target_successes:
                break
    update_summary(args.output_root, all_records, args.target_successes, args.max_init_index)


if __name__ == "__main__":
    main()

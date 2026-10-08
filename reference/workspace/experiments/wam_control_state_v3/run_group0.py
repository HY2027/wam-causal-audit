from __future__ import annotations
from wam_causal_audit.paths import resolve as _release_path

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from config import (
    CAUSAL_CUTS,
    DECODE_TARGETS,
    HARDWARE_AUTHORITY,
    MODELS,
    RADIAL_ROOT,
    RESULT_ROOT,
    sha256_file,
    write_json,
)


STEP_WORK = Path(_release_path('@WORKSPACE@/step1_step2_51locus_work'))
WEEK1 = Path(_release_path('@WORKSPACE@/week1_audit_work'))
OLD_EXPERIMENT = Path(_release_path('@WORKSPACE@/experiments/mechanism_guided_stress'))
BADWAM = Path(_release_path('@DATA@/BadWAM'))
FROZEN_LAUNCH_ROOT = Path(_release_path('@WORKSPACE@'))
FROZEN_WAN_VAE = (
    FROZEN_LAUNCH_ROOT
    / "checkpoints/DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors"
)
for path in (
    STEP_WORK,
    WEEK1,
    OLD_EXPERIMENT,
    BADWAM,
    BADWAM / "src",
    BADWAM / "experiments/libero",
    BADWAM / "experiments/first_grasp_lock",
    Path(_release_path('@WORKSPACE@/LIBERO')),
    Path(_release_path('@WORKSPACE@/evaluation/d2_ghost_pilot')),
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import attackwam.attacks  # noqa: F401,E402
from capture import make_capture  # noqa: E402
from hardware_authority import classify_hardware_name  # noqa: E402
from protocol import load_npz, observation_sha256, selected_clean_call, tensor_sha256  # noqa: E402
from registry_interventions import selected_loci  # noqa: E402
from run_step2 import ENGINES  # noqa: E402


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def clean_dir(task: int, state: int, model: str) -> Path:
    candidate = (
        RADIAL_ROOT
        / "clean_candidates"
        / model
        / f"task_{task}"
        / f"state_{state:02d}"
        / "clean"
    )
    if candidate.is_dir():
        return candidate
    return (
        Path(_release_path('@WORKSPACE@/runs/counterfactual_empty_location_closed_loop/rollouts/full'))
        / model
        / f"task_{task}"
        / f"state_{state:02d}"
        / "clean"
    )


def selected_call_start(task: int, state: int, model: str) -> dict[str, Any]:
    root = clean_dir(task, state, model)
    result = json.loads((root / "result.json").read_text(encoding="utf-8"))
    actions = read_csv(root / "actions.csv")
    calls = json.loads((root / "policy_calls.json").read_text(encoding="utf-8"))
    attempt = int(result["first_grasp_attempt_step"])
    attempt_rows = [row for row in actions if int(row["step"]) == attempt]
    if len(attempt_rows) != 1 or attempt_rows[0]["closure_attempt"] != "True":
        raise AssertionError("frozen first-grasp lineage mismatch")
    call_id = int(attempt_rows[0]["policy_call"])
    start_step = min(
        int(row["step"]) for row in actions if int(row["policy_call"]) == call_id
    )
    return {
        "instruction": result["instruction"],
        "target_object": result["target_object"],
        "selected_call_start_step": start_step,
        "seed": int(calls[call_id]["seed"]),
    }


def load_observation(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def restore_source(
    task_id: int, state_id: int, model: str
) -> tuple[Any, Any, dict[str, Any], dict[str, Any]]:
    import run_first_grasp_lock as fgl
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()["libero_object"]()
    task = suite.get_task(task_id)
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=bddl,
        camera_heights=fgl.LIBERO_ENV_RESOLUTION,
        camera_widths=fgl.LIBERO_ENV_RESOLUTION,
    )
    env.seed(42)
    env.reset()
    obs = dict(env.set_init_state(suite.get_task_init_states(task_id)[state_id]))
    for _ in range(30):
        obs, _, _, _ = env.step(fgl.get_libero_dummy_action())
        obs = dict(obs)
    sim = fgl.get_sim(env)
    if state_id < 10:
        pair_path = (
            Path(_release_path('@WORKSPACE@/runs/state_conditioned_rep_replacement/state_pairs'))
            / f"task_{task_id}"
            / f"pair_{state_id:02d}"
            / "paired_state.npz"
        )
        with np.load(pair_path, allow_pickle=False) as archive:
            initial = archive["state_a"].copy()
        sim.set_state_from_flattened(initial)
        sim.forward()
        obs = dict(env.env._get_observations(force_update=True))
    selection = selected_call_start(task_id, state_id, model)
    actions = read_csv(clean_dir(task_id, state_id, model) / "actions.csv")
    for row in actions:
        if int(row["step"]) >= int(selection["selected_call_start_step"]):
            break
        obs, _, _, _ = env.step(json.loads(row["action"]))
        obs = dict(obs)
    geometry = RADIAL_ROOT / "geometry" / model / f"task_{task_id}" / f"state_{state_id:02d}"
    with np.load(geometry / "source_state.npz", allow_pickle=False) as archive:
        saved_state = {key: archive[key].copy() for key in archive.files}
    reference = load_observation(geometry / "source_observation.npz")
    sim = fgl.get_sim(env)
    shape_match = set(reference) <= set(obs) and all(
        np.asarray(obs[key]).shape == np.asarray(reference[key]).shape for key in reference
    )
    errors = {
        key: float(
            np.max(
                np.abs(
                    np.asarray(obs[key], dtype=np.float64)
                    - np.asarray(reference[key], dtype=np.float64)
                )
            )
        )
        if np.asarray(reference[key]).size
        else 0.0
        for key in reference
        if key in obs and np.asarray(obs[key]).shape == np.asarray(reference[key]).shape
    }
    non_image = [value for key, value in errors.items() if not key.endswith("_image")]
    non_image_max = max(non_image, default=math.inf)
    state_checks = {
        "flat": np.array_equal(np.asarray(sim.get_state().flatten()), saved_state["flat"]),
        "ctrl": np.array_equal(np.asarray(sim.data.ctrl), saved_state["ctrl"]),
        "qfrc_applied": np.array_equal(np.asarray(sim.data.qfrc_applied), saved_state["qfrc_applied"]),
        "xfrc_applied": np.array_equal(np.asarray(sim.data.xfrc_applied), saved_state["xfrc_applied"]),
    }
    audit = {
        "source_state_path": str(geometry / "source_state.npz"),
        "source_observation_path": str(geometry / "source_observation.npz"),
        "source_state_sha256": sha256_file(geometry / "source_state.npz"),
        "source_observation_sha256": sha256_file(geometry / "source_observation.npz"),
        "observation_shape_match": shape_match,
        "observation_bit_exact": shape_match
        and all(np.array_equal(np.asarray(obs[key]), reference[key]) for key in reference),
        "non_image_observation_max_abs_error": non_image_max,
        "simulator_state_bit_exact": state_checks,
        "replay_pass": bool(shape_match and non_image_max <= 1e-10 and all(state_checks.values())),
    }
    if not audit["replay_pass"]:
        env.close()
        raise AssertionError(f"source branch replay mismatch: {audit}")
    return env, task, reference, audit


def collection_hash(observation: Mapping[str, np.ndarray], keys: list[str]) -> str:
    digest = hashlib.sha256()
    for key in sorted(keys):
        value = np.ascontiguousarray(observation[key])
        digest.update(key.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        digest.update(memoryview(value.view(np.uint8)))
    return digest.hexdigest()


def observation_component_hashes(observation: Mapping[str, np.ndarray]) -> dict[str, str]:
    rgb = [key for key in observation if key.endswith("_image")]
    proprio = [
        key
        for key in observation
        if key.startswith("robot0_") and not key.endswith("_image")
    ]
    return {
        "observation_sha256": observation_sha256(observation),
        "rgb_sha256": collection_hash(observation, rgb),
        "proprio_sha256": collection_hash(observation, proprio),
    }


def model_weight_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            digest.update(name.encode())
            digest.update(str(tuple(parameter.shape)).encode())
            digest.update(str(parameter.dtype).encode())
            digest.update(tensor_sha256(parameter).encode())
    return digest.hexdigest()


@torch.no_grad()
def native_action(runner: Any, model: str, observation: Mapping[str, Any], instruction: str, seed: int) -> torch.Tensor:
    if model == "direct":
        from direct_core import action_from_cache, build_current_cache

        prepared = runner._prepared(observation, instruction)
        current = build_current_cache(runner.policy, prepared)
        return action_from_cache(
            runner.policy, prepared, current, current["cache"], seed=seed
        )["action"].detach().cpu().float()
    if model == "joint":
        from run_joint_smoke_pair import controlled_infer

        prepared = runner._prepared(observation, instruction)
        return controlled_infer(
            runner.policy,
            prepared["image"],
            prepared["context"],
            prepared["context_mask"],
            prepared["context"],
            prepared["context_mask"],
            seed=seed,
            controller=None,
        )["action"].detach().cpu().float()
    if model == "idm":
        from common import action_from_latent, action_noise, generate_video_latent

        prepared = runner._prepared(observation, instruction)
        latent = generate_video_latent(runner.policy, prepared, seed=seed)["latent"]
        return action_from_latent(
            runner.policy,
            prepared,
            latent,
            action_noise(runner.policy, seed),
            capture_cache=True,
        )["action"].detach().cpu().float()

    from imagewam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
    from run_libero_object_causal import model_input

    image, proprio = model_input(runner.model, runner.processor, runner.cfg, observation)
    return runner.model.infer_action(
        prompt=DEFAULT_PROMPT.format(task=instruction),
        input_image=image.to("cuda"),
        action_horizon=16,
        proprio=proprio,
        num_inference_steps=10,
        sigma_shift=None,
        seed=int(seed),
        rand_device="cpu",
        tiled=False,
        world_rep_intervention=None,
    )["action"].detach().cpu().float()


def capture_summary(runner: Any, observation: Mapping[str, Any], instruction: str, seed: int) -> dict[str, str]:
    captured = runner.capture(observation, instruction, seed)
    result = {
        "action_sha256": tensor_sha256(captured["action"]),
        "representation_sha256": str(captured["representation_sha256"]),
        "feature_bundle_sha256": str(captured.get("feature_bundle_sha256")),
    }
    del captured
    gc.collect()
    torch.cuda.empty_cache()
    return result


def locus_hashes(engine: Any, condition: str, loci: list[str]) -> dict[str, str]:
    return {locus: engine.locus_hash(condition, locus) for locus in loci}


def scheduler_state(runner: Any, model: str) -> dict[str, Any]:
    policy = getattr(runner, "policy", None)
    adapter = getattr(runner, "adapter", None)
    return {
        "model": model,
        "num_inference_steps": int(
            getattr(policy, "num_inference_steps", 10)
            if policy is not None
            else 10
        ),
        "action_horizon": int(
            getattr(policy, "action_horizon", 16)
            if policy is not None
            else 16
        ),
        "replan_steps": int(
            getattr(policy, "replan_steps", getattr(adapter, "replan_steps", 0))
        ),
        "rand_device": "cpu",
        "seed_controls_diffusion_and_action_denoising": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--gpu-id", type=int, required=True)
    args = parser.parse_args()
    model = args.model
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu_id))
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    if not FROZEN_WAN_VAE.is_file():
        raise FileNotFoundError(
            f"Frozen local checkpoint is missing; refusing implicit download: {FROZEN_WAN_VAE}"
        )
    os.chdir(FROZEN_LAUNCH_ROOT)

    frozen_path = RESULT_ROOT / "configs" / "frozen_group0_v3.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    case = frozen["group0_cases"][model]
    source = load_npz(Path(case["recipient_observation"]["path"]))
    donor = load_npz(Path(case["donor_observation"]["path"]))
    selection = selected_clean_call(model, 0, 0)
    instruction = selection["instruction"]
    seed = int(selection["seed"])

    env, _task, _restored_observation, restore_audit = restore_source(0, 0, model)
    env.close()
    del env

    runner = make_capture(model, 0)
    runner.model.eval()
    hardware_name = torch.cuda.get_device_name(0)
    hardware_class = classify_hardware_name(hardware_name)
    weight_before = model_weight_hash(runner.model)
    native_first = native_action(runner, model, source, instruction, seed)
    native_second = native_action(runner, model, source, instruction, seed)
    no_hook_exact = bool(torch.equal(native_first, native_second))
    source_dump = capture_summary(runner, source, instruction, seed)
    read_only_exact = source_dump["action_sha256"] == tensor_sha256(native_first)
    donor_first = capture_summary(runner, donor, instruction, seed)
    donor_second = capture_summary(runner, donor, instruction, seed)
    donor_repeat_exact = donor_first == donor_second

    engine = ENGINES[model](runner, source, donor, instruction, seed)
    causal = list(CAUSAL_CUTS[model])
    decode = [DECODE_TARGETS[model]]
    causal_identity = engine.infer("clean", "clean", causal)
    decode_identity = engine.infer("clean", "clean", decode)
    identity_checks = {
        "engine_clean_matches_no_hook": bool(torch.equal(engine.clean, native_first)),
        "causal_identity_bit_exact": bool(torch.equal(causal_identity, engine.clean)),
        "decode_identity_bit_exact": bool(torch.equal(decode_identity, engine.clean)),
        "causal_identity_max_abs": float((causal_identity - engine.clean).abs().max()),
        "decode_identity_max_abs": float((decode_identity - engine.clean).abs().max()),
    }
    identity_exact = all(
        identity_checks[key]
        for key in (
            "engine_clean_matches_no_hook",
            "causal_identity_bit_exact",
            "decode_identity_bit_exact",
        )
    )
    all_loci = list(dict.fromkeys(causal + decode))
    representation_hashes = {
        "recipient": locus_hashes(engine, "clean", all_loci),
        "donor": locus_hashes(engine, "fault", all_loci),
    }
    weight_after = model_weight_hash(runner.model)
    historical = json.loads(
        (
            RADIAL_ROOT
            / "cases"
            / model
            / "task_0"
            / "state_00"
            / "dose_0p0cm"
            / "result.json"
        ).read_text(encoding="utf-8")
    )
    checks = {
        "no_hook_replay_bit_exact": no_hook_exact,
        "read_only_activation_dump_bit_exact": read_only_exact,
        "same_value_identity_copy_bit_exact": identity_exact,
        "clean_simulator_state_restore_bit_exact": bool(
            restore_audit["replay_pass"]
            and all(restore_audit["simulator_state_bit_exact"].values())
        ),
        "model_weight_hash_unchanged": weight_before == weight_after,
        "donor_repeat_generation_bit_exact": donor_repeat_exact,
        "hardware_authority_match": hardware_class == HARDWARE_AUTHORITY[model],
    }
    result = {
        "stage": "GROUP_0_TECHNICAL_VALIDATION",
        "model": model,
        "status": "PASS" if all(checks.values()) else "TECHNICAL_STOP",
        "checks": checks,
        "case_selection": case,
        "instruction": instruction,
        "target_identity": selection["target_object"],
        "seed": seed,
        "scheduler_state": scheduler_state(runner, model),
        "hardware_id": hardware_name,
        "hardware_class": hardware_class,
        "checkpoint_hash": weight_before,
        "model_weight_hash_before": weight_before,
        "model_weight_hash_after": weight_after,
        "recipient_state_hash": case["recipient_state"]["sha256"],
        "donor_state_hash": case["donor_state"]["sha256"],
        "recipient_hashes": observation_component_hashes(source),
        "donor_hashes": observation_component_hashes(donor),
        "recipient_action_hash": tensor_sha256(native_first),
        "recipient_action_repeat_hash": tensor_sha256(native_second),
        "donor_action_hash": donor_first["action_sha256"],
        "donor_action_repeat_hash": donor_second["action_sha256"],
        "read_only_dump": source_dump,
        "locus_tensor_hashes": representation_hashes,
        "identity_checks": identity_checks,
        "restore_audit": restore_audit,
        "historical_native_action_sha256": historical.get("native_action_sha256"),
        "matches_historical_same_hardware_baseline": (
            tensor_sha256(native_first) == historical.get("native_action_sha256")
        ),
        "frozen_group0_config_sha256": sha256_file(frozen_path),
        "motion_cos_used_for_admission": False,
        "generated_actions_executed_in_simulator": False,
        "completed_at_unix": time.time(),
    }
    output = RESULT_ROOT / "technical_validation" / model / "result.json"
    write_json(output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

from __future__ import annotations
from wam_causal_audit.paths import resolve as _release_path

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


WORK = Path(__file__).resolve().parent
for candidate in (
    WORK,
    Path(_release_path('@WORKSPACE@/step1_step2_51locus_work')),
    Path(_release_path('@WORKSPACE@/week1_audit_work')),
):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from capture import make_capture  # noqa: E402
from config import HARDWARE_AUTHORITY, RESULT_ROOT  # noqa: E402
from group1_config import FACTOR_SPECS, sha256_file  # noqa: E402
from hardware_authority import classify_hardware_name  # noqa: E402
from protocol import load_npz, tensor_sha256  # noqa: E402
from run_group0 import (  # noqa: E402
    FROZEN_LAUNCH_ROOT,
    FROZEN_WAN_VAE,
    model_weight_hash,
    native_action,
    observation_component_hashes,
    scheduler_state,
)


ROOT = Path(_release_path('@DATA@/wam_factor_routing_v5'))
OUT = ROOT / "group4a_heldout_factor_execution"
PHASE_REGISTRY = OUT / "phase_registry.jsonl"
DONOR_REGISTRY = OUT / "donor_bank/counterfactual_qc.jsonl"
REGISTRY = ROOT / "group4/group4_heldout_state_registry.csv"
REGISTRY_SHA256 = "42f0e5c52f270727db4febe3c61f4e01fd392609cec71846f718cbb6c01b9fbf"
SEED_BASES = {"direct": 710000, "joint": 720000, "idm": 730000, "imagewam": 740000}
EPSILON = 1e-12


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def save_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def policy_seed(model: str, phase: dict[str, Any]) -> int:
    # This is the frozen Group-1 policy seed rule applied to the model-
    # independent held-out trajectory action boundary.
    return int(
        SEED_BASES[model]
        + int(phase["task_id"]) * 10_000
        + int(phase["source_state_id"]) * 100
        + int(phase["source_step_before_action"])
    )


def cosine(left: np.ndarray, right: np.ndarray) -> float | None:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left.reshape(-1), right.reshape(-1)) / denominator) if denominator > EPSILON else None


def phase_drives(runner: Any, action: torch.Tensor, object_to_goal: np.ndarray) -> dict[str, float]:
    env_action = np.asarray(runner.normalized_to_env(action), dtype=np.float64)
    direction = np.asarray(object_to_goal, dtype=np.float64)
    direction /= np.linalg.norm(direction) + EPSILON
    translation = env_action[:, :3]
    gripper = env_action[:, 6]
    transport = float(np.sum(translation @ direction))
    return {
        "ClosingDrive": float(np.maximum(-gripper, 0.0).sum()),
        "gripper_drive": float((-gripper).sum()),
        "release_open_drive": float(np.maximum(gripper, 0.0).sum()),
        "TransportDrive": transport,
        "goal_approach_drive": transport,
    }


def validate_dose(row: dict[str, Any]) -> bool:
    spec = FACTOR_SPECS[row["factor"]]
    allowed = list(spec["primary_signed_doses"]) + list(spec.get("stress_signed_doses", []))
    controls = list(spec.get("relation_controls", ["STANDARD"]))
    return bool(
        row["phase"] in spec["phases"]
        and float(row["signed_dose"]) in [float(value) for value in allowed]
        and row["relation_control"] in controls
        and bool(row["primary_dose"]) == (float(row["signed_dose"]) in [float(value) for value in spec["primary_signed_doses"]])
    )


def execute_case(
    runner: Any,
    model: str,
    donor: dict[str, Any],
    phase: dict[str, Any],
    checkpoint_hash: str,
    recipient_cache: dict[str, torch.Tensor],
    smoke: bool,
) -> dict[str, Any]:
    root = OUT / ("technical_validation/model_smoke" if smoke else "model_runs") / model
    case_root = root / "cases" / donor["case_id"]
    result_path = case_root / "result.json"
    if not smoke and result_path.is_file():
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if existing.get("status") == "COMPLETED":
            return existing

    if sha256_file(REGISTRY) != REGISTRY_SHA256:
        raise RuntimeError("STOP_GROUP4_REGISTRY_HASH_MISMATCH")
    if not validate_dose(donor):
        raise RuntimeError(f"STOP_FACTOR_DOSE_IDENTITY:{donor['case_id']}")
    if donor["base_state_id"] != phase["base_state_id"]:
        raise RuntimeError(f"STOP_PHASE_IDENTITY:{donor['case_id']}")
    if donor["source_state_hash"] != phase["recipient_state_hash"]:
        raise RuntimeError(f"STOP_RECIPIENT_STATE_HASH:{donor['case_id']}")
    if sha256_file(Path(phase["recipient_state_path"])) != phase["recipient_state_file_sha256"]:
        raise RuntimeError(f"STOP_PHASE_STATE_FILE_HASH:{donor['case_id']}")
    if sha256_file(Path(donor["donor_state_path"])) != donor["donor_state_file_sha256"]:
        raise RuntimeError(f"STOP_DONOR_STATE_FILE_HASH:{donor['case_id']}")
    if sha256_file(Path(donor["donor_observation_path"])) != donor["donor_observation_file_sha256"]:
        raise RuntimeError(f"STOP_DONOR_OBSERVATION_FILE_HASH:{donor['case_id']}")

    recipient_obs = load_npz(Path(phase["recipient_observation_path"]))
    donor_obs = load_npz(Path(donor["donor_observation_path"]))
    seed = policy_seed(model, phase)
    started = time.time()
    cache_key = phase["base_state_id"]
    if cache_key not in recipient_cache:
        recipient_cache[cache_key] = native_action(
            runner, model, recipient_obs, phase["instruction"], seed
        ).detach().cpu().float()
    recipient_action = recipient_cache[cache_key]
    donor_action = native_action(
        runner, model, donor_obs, phase["instruction"], seed
    ).detach().cpu().float()
    if recipient_action.shape != donor_action.shape or not torch.isfinite(donor_action).all().item():
        raise RuntimeError(f"STOP_ACTION_ARTIFACT_INVALID:{donor['case_id']}")

    object_to_goal = np.asarray(phase["goal_position_m"], dtype=np.float64) - np.asarray(
        phase["object_position_m"], dtype=np.float64
    )
    recipient_drives = phase_drives(runner, recipient_action, object_to_goal)
    donor_drives = phase_drives(runner, donor_action, object_to_goal)
    effect = donor_action.numpy().astype(np.float64) - recipient_action.numpy().astype(np.float64)
    case_root.mkdir(parents=True, exist_ok=True)
    action_path = case_root / "actions.npz"
    save_npz(
        action_path,
        RECIPIENT_NATIVE=recipient_action.numpy(),
        DONOR_NATIVE=donor_action.numpy(),
    )
    repeat_checks: dict[str, bool] = {}
    if smoke:
        recipient_repeat = native_action(runner, model, recipient_obs, phase["instruction"], seed).detach().cpu().float()
        donor_repeat = native_action(runner, model, donor_obs, phase["instruction"], seed).detach().cpu().float()
        repeat_checks = {
            "no_hook_replay_bit_exact": bool(torch.equal(recipient_action, recipient_repeat)),
            "donor_repeat_generation_bit_exact": bool(torch.equal(donor_action, donor_repeat)),
            "action_shapes_match": recipient_action.shape == donor_action.shape,
            "all_actions_finite": bool(torch.isfinite(recipient_action).all() and torch.isfinite(donor_action).all()),
        }
    result = {
        "status": "COMPLETED" if not smoke or all(repeat_checks.values()) else "TECHNICAL_STOP",
        "stage": "GROUP4A_NATIVE_SMOKE" if smoke else "GROUP4A_HELDOUT_FACTOR_PROFILE_NATIVE",
        "model": model,
        "case_id": donor["case_id"],
        "heldout_base_state_id": phase["heldout_base_state_id"],
        "base_state_id": phase["base_state_id"],
        "task_id": int(phase["task_id"]),
        "source_state_id": int(phase["source_state_id"]),
        "source_trajectory_id": phase["source_trajectory_id"],
        "source_episode_index": int(phase["source_episode_index"]),
        "source_step_before_action": int(phase["source_step_before_action"]),
        "phase": donor["phase"],
        "factor": donor["factor"],
        "relation_control": donor["relation_control"],
        "signed_dose": float(donor["signed_dose"]),
        "dose_unit": donor["dose_unit"],
        "primary_dose": bool(donor["primary_dose"]),
        "instruction": phase["instruction"],
        "target_identity": phase["target_identity"],
        "goal_identity": phase["goal_identity"],
        "seed": seed,
        "policy_seed_rule": f"{SEED_BASES[model]} + task*10000 + state*100 + source_step_before_action",
        "scheduler_state": scheduler_state(runner, model),
        "hardware_id": torch.cuda.get_device_name(0),
        "hardware_class": classify_hardware_name(torch.cuda.get_device_name(0)),
        "hardware_authority_expected": HARDWARE_AUTHORITY[model],
        "hardware_authority_match": classify_hardware_name(torch.cuda.get_device_name(0)) == HARDWARE_AUTHORITY[model],
        "checkpoint_hash": checkpoint_hash,
        "heldout_registry_sha256": REGISTRY_SHA256,
        "recipient_state_hash": phase["recipient_state_hash"],
        "phase_state_file_sha256": phase["recipient_state_file_sha256"],
        "donor_state_hash": donor["donor_state_hash"],
        "donor_state_file_sha256": donor["donor_state_file_sha256"],
        "recipient_hashes": observation_component_hashes(recipient_obs),
        "donor_hashes": observation_component_hashes(donor_obs),
        "recipient_action_hash": tensor_sha256(recipient_action),
        "donor_action_hash": tensor_sha256(donor_action),
        "actions_path": str(action_path),
        "actions_file_sha256": sha256_file(action_path),
        "native_action_effect": {
            "action_l2": float(np.linalg.norm(effect)),
            "action_cosine_to_recipient": cosine(donor_action.numpy().astype(np.float64), recipient_action.numpy().astype(np.float64)),
            "max_elementwise_delta": float(np.max(np.abs(effect))),
        },
        "recipient_phase_drives": recipient_drives,
        "donor_phase_drives": donor_drives,
        "counterfactual_qc_path": donor["counterfactual_qc_path"],
        "counterfactual_qc_sha256": sha256_file(Path(donor["counterfactual_qc_path"])),
        "factor_and_dose_identity": True,
        "phase_state_hash_verified": True,
        "simulator_restore_integrity_source": str(OUT / "technical_validation/phase_restore_audit.jsonl"),
        "action_artifact_provenance_verified": True,
        "repeat_checks": repeat_checks,
        "routing_intervention_executed": False,
        "model_output_used_for_state_selection": False,
        "generated_action_executed_in_simulator": False,
        "runtime_seconds": time.time() - started,
    }
    atomic_json(result_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("direct", "joint", "idm", "imagewam"), required=True)
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--cpu-threads", type=int, default=16)
    args = parser.parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu_id))
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(1)
    if not FROZEN_WAN_VAE.is_file():
        raise FileNotFoundError(f"Frozen checkpoint dependency missing: {FROZEN_WAN_VAE}")
    if sha256_file(REGISTRY) != REGISTRY_SHA256:
        raise RuntimeError("STOP_GROUP4_REGISTRY_HASH_MISMATCH")
    os.chdir(FROZEN_LAUNCH_ROOT)

    phases = {row["base_state_id"]: row for row in read_jsonl(PHASE_REGISTRY)}
    donors = sorted(
        [row for row in read_jsonl(DONOR_REGISTRY) if row.get("donor_valid") is True],
        key=lambda row: row["case_id"],
    )
    if len(donors) != 596:
        raise RuntimeError(f"STOP_VALID_DONOR_CARDINALITY:{len(donors)}")
    if args.smoke:
        donors = donors[:1]
    elif args.max_cases is not None:
        donors = donors[: args.max_cases]

    group0_path = RESULT_ROOT / "technical_validation" / args.model / "result.json"
    group0 = json.loads(group0_path.read_text(encoding="utf-8"))
    if group0.get("status") != "PASS" or not all(group0.get("checks", {}).values()):
        raise RuntimeError(f"STOP_GROUP0_NOT_PASS:{args.model}")
    runner = make_capture(args.model, 0)
    runner.model.eval()
    checkpoint_hash = model_weight_hash(runner.model)
    if checkpoint_hash != group0["checkpoint_hash"]:
        raise RuntimeError(f"STOP_CHECKPOINT_HASH_CHANGED:{args.model}")

    recipient_cache: dict[str, torch.Tensor] = {}
    results: list[dict[str, Any]] = []
    for index, donor in enumerate(donors, 1):
        result = execute_case(
            runner,
            args.model,
            donor,
            phases[donor["base_state_id"]],
            checkpoint_hash,
            recipient_cache,
            args.smoke,
        )
        results.append(result)
        print(json.dumps({
            "model": args.model,
            "completed": index,
            "total": len(donors),
            "case_id": donor["case_id"],
            "status": result["status"],
            "runtime_seconds": result["runtime_seconds"],
        }), flush=True)
        if result["status"] != "COMPLETED":
            raise SystemExit(2)
    final_weight_hash = model_weight_hash(runner.model)
    summary = {
        "status": "PASS" if final_weight_hash == checkpoint_hash else "TECHNICAL_STOP",
        "stage": "GROUP4A_NATIVE_SMOKE" if args.smoke else "GROUP4A_HELDOUT_FACTOR_PROFILE_NATIVE",
        "model": args.model,
        "smoke": args.smoke,
        "completed_cases": len(results),
        "eligible_valid_cases": len(donors),
        "all_frozen_donor_cases": 920,
        "valid_qc_donor_cases": 596,
        "checkpoint_hash": checkpoint_hash,
        "final_weight_hash": final_weight_hash,
        "model_weight_hash_unchanged": final_weight_hash == checkpoint_hash,
        "hardware_id": torch.cuda.get_device_name(0),
        "hardware_class": classify_hardware_name(torch.cuda.get_device_name(0)),
        "hardware_authority_match": classify_hardware_name(torch.cuda.get_device_name(0)) == HARDWARE_AUTHORITY[args.model],
        "group0_result_path": str(group0_path),
        "group0_result_sha256": sha256_file(group0_path),
        "group0_gates_inherited": group0["checks"],
        "heldout_registry_sha256": REGISTRY_SHA256,
        "routing_intervention_executed": False,
        "group4b_started": False,
        "group5_started": False,
    }
    summary_path = OUT / (
        f"technical_validation/model_smoke/{args.model}/run_summary.json"
        if args.smoke else f"model_runs/{args.model}/run_summary.json"
    )
    atomic_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    del runner, recipient_cache, results
    gc.collect()
    torch.cuda.empty_cache()
    if summary["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

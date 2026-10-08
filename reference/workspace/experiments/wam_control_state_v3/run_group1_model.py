from __future__ import annotations
from wam_causal_audit.paths import resolve as _release_path

import argparse
import gc
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from config import CAUSAL_CUTS, DECODE_TARGETS, RESULT_ROOT, sha256_file, write_json


WORK = Path(__file__).resolve().parent
STEP_WORK = Path(_release_path('@WORKSPACE@/step1_step2_51locus_work'))
WEEK1 = Path(_release_path('@WORKSPACE@/week1_audit_work'))
for candidate in (WORK, STEP_WORK, WEEK1):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

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
from capture import make_capture  # noqa: E402
from run_step2 import ENGINES  # noqa: E402


GROUP1_ROOT = RESULT_ROOT / "group1_factor_phase"
EPSILON = 1e-12


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def save_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def cosine(left: np.ndarray, right: np.ndarray) -> float | None:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator > EPSILON else None


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


def action_metrics(
    action: torch.Tensor,
    recipient: torch.Tensor,
    donor: torch.Tensor,
) -> dict[str, Any]:
    action_array = action.detach().cpu().float().numpy().astype(np.float64)
    recipient_array = recipient.detach().cpu().float().numpy().astype(np.float64)
    donor_array = donor.detach().cpu().float().numpy().astype(np.float64)
    effect = action_array - recipient_array
    donor_effect = donor_array - recipient_array
    denominator = float(np.dot(donor_effect.reshape(-1), donor_effect.reshape(-1)))
    transfer = float(np.dot(effect.reshape(-1), donor_effect.reshape(-1)) / (denominator + EPSILON))
    residual = effect - transfer * donor_effect
    closure_delta = action_array - donor_array
    return {
        "action_l2": float(np.linalg.norm(effect)),
        "action_cosine": cosine(effect, donor_effect),
        "directional_transfer": transfer,
        "orthogonal_residual": float(np.linalg.norm(residual)),
        "max_elementwise_residual": float(np.max(np.abs(closure_delta))),
        "bit_exact_closure": bool(np.array_equal(action_array, donor_array)),
        "allclose_closure": bool(np.allclose(action_array, donor_array, rtol=1e-5, atol=1e-6)),
        "donor_effect_l2": float(np.sqrt(denominator)),
    }


def case_output(model: str, case_id: str, smoke: bool) -> Path:
    root = GROUP1_ROOT / ("technical_smoke" if smoke else "model_runs") / model / "cases"
    return root / case_id


def execute_case(
    runner: Any,
    model: str,
    donor_row: dict[str, Any],
    phase_row: dict[str, Any],
    noncausal: str,
    checkpoint_hash: str,
    smoke: bool,
) -> dict[str, Any]:
    out = case_output(model, donor_row["case_id"], smoke)
    result_path = out / "result.json"
    if not smoke and result_path.is_file():
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if existing.get("status") == "COMPLETED":
            return existing
    recipient = load_npz(Path(phase_row["recipient_observation_path"]))
    donor = load_npz(Path(donor_row["donor_observation_path"]))
    instruction = phase_row["instruction"]
    seed = int(phase_row["policy_seed"])
    started = time.time()
    engine = ENGINES[model](runner, recipient, donor, instruction, seed)
    recipient_native = engine.clean.detach().cpu().float()
    donor_native = native_action(runner, model, donor, instruction, seed).detach().cpu().float()
    full = engine.infer("clean", "fault", list(CAUSAL_CUTS[model])).detach().cpu().float()
    decode = engine.infer("clean", "fault", [DECODE_TARGETS[model]]).detach().cpu().float()
    noncausal_action = engine.infer("clean", "fault", [noncausal]).detach().cpu().float()
    actions = {
        "RECIPIENT_NATIVE": recipient_native,
        "DONOR_NATIVE": donor_native,
        "FULL_CAUSAL_CUT_SWAP": full,
        "DECODE_LOCUS_SWAP": decode,
        "NONCAUSAL_MATCHED_SWAP": noncausal_action,
    }
    all_loci = list(dict.fromkeys(list(CAUSAL_CUTS[model]) + [DECODE_TARGETS[model], noncausal]))
    locus_hashes = {
        "recipient": {locus: engine.locus_hash("clean", locus) for locus in all_loci},
        "donor": {locus: engine.locus_hash("fault", locus) for locus in all_loci},
    }
    object_to_goal = (
        np.asarray(phase_row["goal_position_m"], dtype=np.float64)
        - np.asarray(phase_row["object_position_m"], dtype=np.float64)
    )
    conditions = {
        name: {
            "action_hash": tensor_sha256(action),
            "metrics": action_metrics(action, recipient_native, donor_native),
            "phase_drives": phase_drives(runner, action, object_to_goal),
        }
        for name, action in actions.items()
    }
    out.mkdir(parents=True, exist_ok=True)
    action_path = out / "actions.npz"
    save_npz(action_path, **{name: value.numpy() for name, value in actions.items()})
    smoke_checks: dict[str, bool] = {}
    if smoke:
        recipient_repeat = native_action(runner, model, recipient, instruction, seed).detach().cpu().float()
        donor_repeat = native_action(runner, model, donor, instruction, seed).detach().cpu().float()
        smoke_checks = {
            "engine_recipient_matches_native_repeat": bool(torch.equal(recipient_native, recipient_repeat)),
            "donor_repeat_generation_bit_exact": bool(torch.equal(donor_native, donor_repeat)),
            "all_five_actions_finite": all(torch.isfinite(action).all().item() for action in actions.values()),
            "all_action_shapes_match": len({tuple(action.shape) for action in actions.values()}) == 1,
            "locus_hashes_present": all(set(side) == set(all_loci) for side in locus_hashes.values()),
        }
    result = {
        "status": "COMPLETED" if not smoke or all(smoke_checks.values()) else "TECHNICAL_STOP",
        "stage": "GROUP1_MODEL_SMOKE" if smoke else "GROUP1_FACTOR_PHASE_MODEL_RUN",
        "model": model,
        "case_id": donor_row["case_id"],
        "base_state_id": donor_row["base_state_id"],
        "task_id": donor_row["task_id"],
        "source_state_id": donor_row["source_state_id"],
        "phase": donor_row["phase"],
        "factor": donor_row["factor"],
        "relation_control": donor_row["relation_control"],
        "signed_dose": donor_row["signed_dose"],
        "primary_dose": donor_row["primary_dose"],
        "instruction": instruction,
        "target_identity": phase_row["target_identity"],
        "seed": seed,
        "scheduler_state": scheduler_state(runner, model),
        "hardware_id": torch.cuda.get_device_name(0),
        "hardware_class": classify_hardware_name(torch.cuda.get_device_name(0)),
        "checkpoint_hash": checkpoint_hash,
        "recipient_state_hash": phase_row["recipient_state_hash"],
        "donor_state_hash": donor_row["donor_state_hash"],
        "recipient_hashes": observation_component_hashes(recipient),
        "donor_hashes": observation_component_hashes(donor),
        "recipient_action_hash": conditions["RECIPIENT_NATIVE"]["action_hash"],
        "donor_action_hash": conditions["DONOR_NATIVE"]["action_hash"],
        "all_intervened_action_hashes": {
            name: conditions[name]["action_hash"]
            for name in ("FULL_CAUSAL_CUT_SWAP", "DECODE_LOCUS_SWAP", "NONCAUSAL_MATCHED_SWAP")
        },
        "causal_cut": list(CAUSAL_CUTS[model]),
        "decode_locus": DECODE_TARGETS[model],
        "noncausal_matched_locus": noncausal,
        "locus_tensor_hashes": locus_hashes,
        "conditions": conditions,
        "actions_path": str(action_path),
        "actions_file_sha256": sha256_file(action_path),
        "counterfactual_qc_path": donor_row["counterfactual_qc_path"],
        "smoke_checks": smoke_checks,
        "generated_actions_executed_in_simulator": False,
        "motioncos_used_for_admission": False,
        "runtime_seconds": time.time() - started,
    }
    write_json(result_path, result)
    del engine, actions
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("direct", "joint", "idm", "imagewam"), required=True)
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--cpu-threads", type=int)
    args = parser.parse_args()
    if args.cpu_threads is not None:
        if args.cpu_threads <= 0:
            raise ValueError("--cpu-threads must be positive")
        torch.set_num_threads(args.cpu_threads)
        torch.set_num_interop_threads(1)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu_id))
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    if not FROZEN_WAN_VAE.is_file():
        raise FileNotFoundError(f"Frozen checkpoint dependency missing: {FROZEN_WAN_VAE}")
    os.chdir(FROZEN_LAUNCH_ROOT)

    phases = {row["base_state_id"]: row for row in read_jsonl(GROUP1_ROOT / "phase_registry.jsonl")}
    donors = [
        row for row in read_jsonl(GROUP1_ROOT / "donor_bank" / args.model / "counterfactual_qc.jsonl")
        if row["donor_valid"]
    ]
    donors.sort(key=lambda row: row["case_id"])
    if args.smoke:
        preferred = [
            row for row in donors
            if row["factor"] == "F2_OBJECT_ROBOT_GEOMETRY"
            and row["relation_control"] == "OBJECT_ONLY"
            and row["primary_dose"]
        ]
        donors = preferred[:1]
    elif args.max_cases is not None:
        donors = donors[: args.max_cases]
    noncausal_registry = json.loads(
        (RESULT_ROOT / "registries" / "noncausal_matched_v3.json").read_text(encoding="utf-8")
    )
    noncausal = noncausal_registry["selections"][args.model]["locus_id"]
    runner = make_capture(args.model, 0)
    runner.model.eval()
    checkpoint_hash = model_weight_hash(runner.model)
    group0 = json.loads(
        (RESULT_ROOT / "technical_validation" / args.model / "result.json").read_text(encoding="utf-8")
    )
    if checkpoint_hash != group0["checkpoint_hash"]:
        raise AssertionError("Model checkpoint hash changed since Group 0")
    results: list[dict[str, Any]] = []
    for index, donor_row in enumerate(donors, 1):
        result = execute_case(
            runner,
            args.model,
            donor_row,
            phases[donor_row["base_state_id"]],
            noncausal,
            checkpoint_hash,
            args.smoke,
        )
        results.append(result)
        print(
            json.dumps(
                {
                    "model": args.model,
                    "completed": index,
                    "total": len(donors),
                    "case_id": donor_row["case_id"],
                    "status": result["status"],
                    "runtime_seconds": result["runtime_seconds"],
                }
            ),
            flush=True,
        )
        if result["status"] != "COMPLETED":
            raise SystemExit(2)
    final_weight_hash = model_weight_hash(runner.model)
    summary = {
        "status": "PASS" if final_weight_hash == checkpoint_hash else "TECHNICAL_STOP",
        "model": args.model,
        "smoke": args.smoke,
        "completed_cases": len(results),
        "eligible_cases": len(donors),
        "checkpoint_hash": checkpoint_hash,
        "final_weight_hash": final_weight_hash,
        "model_weight_hash_unchanged": final_weight_hash == checkpoint_hash,
        "hardware_id": torch.cuda.get_device_name(0),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
    }
    summary_path = (
        GROUP1_ROOT / "technical_smoke" / f"model_smoke_{args.model}.json"
        if args.smoke
        else GROUP1_ROOT / "model_runs" / args.model / "run_summary.json"
    )
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2))
    if summary["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

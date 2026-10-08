#!/usr/bin/env python3
"""Execute frozen Group-5C controlled prospective cases for one model/shard."""

from __future__ import annotations
from wam_causal_audit.paths import resolve as _release_path

import argparse
import csv
import gc
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch


WORK = Path(__file__).resolve().parent
STEP = Path(_release_path('@WORKSPACE@/step1_step2_51locus_work'))
WEEK1 = Path(_release_path('@WORKSPACE@/week1_audit_work'))
for candidate in (WORK, STEP, WEEK1):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from capture import make_capture  # noqa: E402
from hardware_authority import classify_hardware_name  # noqa: E402
from protocol import load_npz, tensor_sha256  # noqa: E402
from run_group0 import (  # noqa: E402
    FROZEN_LAUNCH_ROOT,
    FROZEN_WAN_VAE,
    model_weight_hash,
    native_action,
)
from run_group1_model import phase_drives  # noqa: E402
from run_group2_joint import save_npz  # noqa: E402
from run_step2 import ENGINES  # noqa: E402


ROOT = Path(_release_path('@DATA@/wam_factor_routing_v5'))
GROUP5 = ROOT / "group5"
STAGE0 = GROUP5 / "group5c_controlled_stage0"
FEASIBILITY = GROUP5 / "group5c_feasibility_amendment"
G4A = ROOT / "group4a_heldout_factor_execution"
OUT = GROUP5 / "group5c_controlled_execution_v3"
PROTOCOL = Path(
    _release_path('@WORKSPACE@/external_protocols/prospective_original_request.txt')
)

COUNTERFACTUALS = STAGE0 / "group5c_counterfactual_registry.csv"
PREDICTIONS = STAGE0 / "group5c_prospective_predictions.csv"
STAGE0_MANIFEST = STAGE0 / "group5c_stage0_freeze_manifest.json"
TRANSFORM_VALIDATION = STAGE0 / "group5c_transform_validation.json"
VALIDITY_MATRIX = FEASIBILITY / "group5c_validity_by_cell.csv"
FEASIBILITY_REPORT = FEASIBILITY / "group5c_feasibility_report.json"
FEASIBILITY_MANIFEST = FEASIBILITY / "group5c_feasibility_manifest.json"
PHASE_REGISTRY = G4A / "phase_registry.jsonl"

MODELS = ("direct", "joint", "idm", "imagewam")
EXPECTED_CASES = 374
EXPECTED_PREDICTIONS = 1496
EXPECTED_HASHES = {
    COUNTERFACTUALS: "991609b722c462091e8f86fe2d4af3c79df43683cad0eb65741d2333fe12a25c",
    PREDICTIONS: "be26d7a37afc23503d155d41b87644abd412fecfd695db1a735c52cc6586823a",
    STAGE0_MANIFEST: "47c020bd66081fdca780e29bab0611ea9fb8e05b6ec9684aaaaeed19e276db63",
    VALIDITY_MATRIX: "984d15f0e87f2c09aa7df302f4fe3a4b1fb628ea2cd0b6d546a7b4b3d1e40097",
    FEASIBILITY_REPORT: "d6f29a8b2f0bb9a38541d62cbf60dd13feb54ddeeb0a35b2de2969b01b816b22",
    FEASIBILITY_MANIFEST: "398e0a6ac1bd96bd74206a444169733218bea86cf5637776050c60444714c2c3",
}
EXPECTED_CHECKPOINTS = {
    "direct": "5cf64ce908ffb13766ccee08ed98d17f81a60e084fdebc5c6911cf8e69572472",
    "joint": "c72f63caa4799790ae7f246f1b4914103929a9d95c85ae4ec927f2e53877ccc4",
    "idm": "38fe5352ddce311150a5d55ae6fab0d58e2fa87a155a9997398e0072e7528ab4",
    "imagewam": "6af92e2f9b29d36408dda693b172e60c20868da05e6442d0994a88faed3909ac",
}
HARDWARE = {"direct": "RTX4090", "joint": "RTX4090", "idm": "BLACKWELL", "imagewam": "RTX4090"}
PRIMARY_DRIVE = {"PREGRASP": "ClosingDrive", "PREPLACE": "goal_approach_drive"}
EPSILON = 1e-12


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def vector_metrics(action: np.ndarray, clean: np.ndarray, physical: np.ndarray) -> dict[str, Any]:
    delta = np.asarray(action, dtype=np.float64) - np.asarray(clean, dtype=np.float64)
    target = np.asarray(physical, dtype=np.float64) - np.asarray(clean, dtype=np.float64)
    denominator = float(np.dot(target.reshape(-1), target.reshape(-1)))
    target_norm = float(np.sqrt(denominator))
    delta_norm = float(np.linalg.norm(delta))
    transfer = float(np.dot(delta.reshape(-1), target.reshape(-1)) / (denominator + EPSILON))
    cosine_denominator = delta_norm * target_norm
    cosine = (
        float(np.dot(delta.reshape(-1), target.reshape(-1)) / cosine_denominator)
        if cosine_denominator > EPSILON
        else None
    )
    orthogonal = delta - transfer * target
    closure = np.asarray(action, dtype=np.float64) - np.asarray(physical, dtype=np.float64)
    return {
        "directional_transfer": transfer,
        "action_l2": delta_norm,
        "action_cosine": cosine,
        "orthogonal_residual": float(np.linalg.norm(orthogonal)),
        "closure_l2": float(np.linalg.norm(closure)),
        "relative_closure_l2": float(np.linalg.norm(closure) / (target_norm + EPSILON)),
        "max_elementwise_residual": float(np.max(np.abs(closure))),
        "bit_exact_closure": bool(np.array_equal(action, physical)),
        "allclose_closure": bool(np.allclose(action, physical, rtol=1e-6, atol=1e-7)),
    }


def verify_inputs() -> tuple[list[dict[str, str]], dict[str, dict[str, str]], dict[str, dict[str, Any]], dict[str, str]]:
    for path, expected in EXPECTED_HASHES.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"STOP_GROUP5C_INPUT_HASH:{path}")
    if not PROTOCOL.is_file():
        raise RuntimeError("STOP_GROUP5C_EXECUTION_PROTOCOL_MISSING")
    stage0 = json.loads(STAGE0_MANIFEST.read_text(encoding="utf-8"))
    feasibility = json.loads(FEASIBILITY_MANIFEST.read_text(encoding="utf-8"))
    if stage0.get("status") != "GROUP5_CONTROLLED_STAGE0_BLOCKED":
        raise RuntimeError("STOP_GROUP5C_STAGE0_PROVENANCE_STATUS")
    if feasibility.get("status") != "GROUP5C_TECHNICAL_FEASIBILITY_PASS":
        raise RuntimeError("STOP_GROUP5C_FEASIBILITY_STATUS")
    cases = read_csv(COUNTERFACTUALS)
    predictions = read_csv(PREDICTIONS)
    if len(cases) != EXPECTED_CASES or len(predictions) != EXPECTED_PREDICTIONS:
        raise RuntimeError("STOP_GROUP5C_EXECUTION_CARDINALITY")
    if any(row["technical_qc_status"] != "PASS" for row in cases):
        raise RuntimeError("STOP_GROUP5C_EXECUTION_NONVALID_CASE")
    valid_ids = set(feasibility["valid_case_ids"])
    if valid_ids != {row["source_group4_case_id"] for row in cases}:
        raise RuntimeError("STOP_GROUP5C_VALID_CASE_IDENTITY")
    if valid_ids & set(feasibility["invalid_case_ids"]):
        raise RuntimeError("STOP_GROUP5C_VALID_INVALID_OVERLAP")
    prediction_map: dict[str, dict[str, str]] = {}
    for row in predictions:
        key = f"{row['model']}::{row['counterfactual_id']}"
        if key in prediction_map:
            raise RuntimeError("STOP_GROUP5C_DUPLICATE_PREDICTION")
        prediction_map[key] = row
    expected_keys = {
        f"{model}::{row['counterfactual_id']}" for row in cases for model in MODELS
    }
    if set(prediction_map) != expected_keys:
        raise RuntimeError("STOP_GROUP5C_PREDICTION_CASE_COVERAGE")
    phases = {row["base_state_id"]: row for row in read_jsonl(PHASE_REGISTRY)}
    if any(row["recipient_state_id"] not in phases for row in cases):
        raise RuntimeError("STOP_GROUP5C_PHASE_IDENTITY")
    source_hashes = {str(path): EXPECTED_HASHES[path] for path in EXPECTED_HASHES}
    source_hashes[str(TRANSFORM_VALIDATION)] = sha256(TRANSFORM_VALIDATION)
    source_hashes[str(PROTOCOL)] = sha256(PROTOCOL)
    return cases, prediction_map, phases, source_hashes


def group4a_source(model: str, case: dict[str, str]) -> tuple[Path, dict[str, Any], np.ndarray, np.ndarray]:
    path = G4A / "model_runs" / model / "cases" / case["source_group4_case_id"] / "result.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        value.get("status") != "COMPLETED"
        or value.get("routing_intervention_executed") is not False
        or value.get("case_id") != case["source_group4_case_id"]
        or value.get("checkpoint_hash") != EXPECTED_CHECKPOINTS[model]
    ):
        raise RuntimeError(f"STOP_GROUP5C_GROUP4A_SOURCE:{model}:{case['counterfactual_id']}")
    actions_path = Path(value["actions_path"])
    if sha256(actions_path) != value["actions_file_sha256"]:
        raise RuntimeError(f"STOP_GROUP5C_GROUP4A_ACTION_HASH:{model}:{case['counterfactual_id']}")
    with np.load(actions_path, allow_pickle=False) as archive:
        clean = np.asarray(archive["RECIPIENT_NATIVE"], dtype=np.float32)
        physical = np.asarray(archive["DONOR_NATIVE"], dtype=np.float32)
    return path, value, clean, physical


def result_root(model: str, counterfactual_id: str, smoke: bool, smoke_label: str | None) -> Path:
    if smoke:
        return OUT / "technical_validation" / model / str(smoke_label) / counterfactual_id
    return OUT / "model_runs" / model / "cases" / counterfactual_id


def execute_case(
    runner: Any,
    model: str,
    case: dict[str, str],
    prediction: dict[str, str],
    phase: dict[str, Any],
    checkpoint_hash: str,
    runner_hash: str,
    protocol_hash: str,
    smoke: bool,
    smoke_label: str | None,
) -> dict[str, Any]:
    target = result_root(model, case["counterfactual_id"], smoke, smoke_label)
    result_path = target / "result.json"
    if not smoke and result_path.is_file():
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if (
            existing.get("status") == "COMPLETED"
            and existing.get("runner_sha256") == runner_hash
            and existing.get("protocol_sha256") == protocol_hash
        ):
            return existing

    started = time.time()
    source_path, source, frozen_clean, frozen_physical = group4a_source(model, case)
    if sha256(Path(case["recipient_observation_path"])) != case["recipient_observation_file_sha256"]:
        raise RuntimeError("STOP_GROUP5C_RECIPIENT_OBSERVATION_HASH")
    if sha256(Path(case["counterfactual_observation_path"])) != case["counterfactual_observation_file_sha256"]:
        raise RuntimeError("STOP_GROUP5C_COUNTERFACTUAL_OBSERVATION_HASH")
    if source["recipient_state_hash"] != case["recipient_state_hash"]:
        raise RuntimeError("STOP_GROUP5C_RECIPIENT_STATE_HASH")
    if source["donor_state_hash"] != case["counterfactual_state_hash"]:
        raise RuntimeError("STOP_GROUP5C_COUNTERFACTUAL_STATE_HASH")
    if int(source["seed"]) < 0:
        raise RuntimeError("STOP_GROUP5C_SEED")

    recipient_obs = load_npz(Path(case["recipient_observation_path"]))
    counterfactual_obs = load_npz(Path(case["counterfactual_observation_path"]))
    engine = ENGINES[model](runner, recipient_obs, counterfactual_obs, case["instruction"], int(source["seed"]))
    clean = engine.clean.detach().cpu().float()
    # Condition B is always the native policy call on the physical donor.
    # engine.fault is retained only as the donor-carrier capture endpoint: in
    # Joint/IDM (and potentially other architectures) it can intentionally
    # combine donor carrier content with recipient-side inputs and therefore
    # must not be relabelled as PHYSICAL_COUNTERFACTUAL.
    physical = native_action(
        runner,
        model,
        counterfactual_obs,
        case["instruction"],
        int(source["seed"]),
    ).detach().cpu().float()
    if not np.array_equal(clean.numpy(), frozen_clean):
        raise RuntimeError("STOP_GROUP5C_CLEAN_NOT_BIT_EXACT_GROUP4A")
    if not np.array_equal(physical.numpy(), frozen_physical):
        raise RuntimeError("STOP_GROUP5C_PHYSICAL_NOT_BIT_EXACT_GROUP4A")

    route = json.loads(prediction["route_intervention_target"])
    secondary = json.loads(prediction["secondary_route_targets"])
    noncausal = prediction["frozen_noncausal_target"]
    if not route or not noncausal:
        raise RuntimeError("STOP_GROUP5C_EMPTY_FROZEN_ROUTE")
    injection = engine.infer("clean", "fault", route).detach().cpu().float()
    same_value = engine.infer("clean", "clean", route).detach().cpu().float()
    negative = engine.infer("clean", "fault", [noncausal]).detach().cpu().float()
    if not torch.equal(same_value, clean):
        raise RuntimeError("STOP_GROUP5C_SAME_VALUE_NOT_BIT_EXACT")

    actions: dict[str, torch.Tensor] = {
        "CLEAN_RECIPIENT": clean,
        "PHYSICAL_COUNTERFACTUAL": physical,
        "CAUSAL_ROUTE_INJECTION": injection,
        "SAME_VALUE_CONTROL": same_value,
        "FROZEN_NONCAUSAL_CONTROL": negative,
    }
    if model == "joint":
        expected_secondary = {"current_only", "future_only"}
        if set(secondary) != expected_secondary:
            raise RuntimeError("STOP_GROUP5C_JOINT_SECONDARY_ROUTE")
        actions["JOINT_CURRENT_ONLY"] = engine.infer(
            "clean", "fault", secondary["current_only"]
        ).detach().cpu().float()
        actions["JOINT_FUTURE_ONLY"] = engine.infer(
            "clean", "fault", secondary["future_only"]
        ).detach().cpu().float()
    if not all(torch.isfinite(action).all().item() for action in actions.values()):
        raise RuntimeError("STOP_GROUP5C_NONFINITE_ACTION")

    smoke_checks: dict[str, bool] = {}
    if smoke:
        repeat = engine.infer("clean", "fault", route).detach().cpu().float()
        negative_identity = engine.infer("clean", "clean", [noncausal]).detach().cpu().float()
        smoke_checks = {
            "clean_reproduces_group4a_bit_exact": True,
            "physical_reproduces_group4a_bit_exact": True,
            "same_value_full_route_bit_exact": bool(torch.equal(same_value, clean)),
            "same_value_noncausal_bit_exact": bool(torch.equal(negative_identity, clean)),
            "donor_route_repeat_bit_exact": bool(torch.equal(repeat, injection)),
            "all_actions_finite": True,
            "no_simulator_execution": True,
            "frozen_route_only": True,
            "frozen_prediction_only": True,
        }
        if not all(smoke_checks.values()):
            raise RuntimeError(f"STOP_GROUP5C_SMOKE:{smoke_checks}")

    target.mkdir(parents=True, exist_ok=True)
    actions_path = target / "actions.npz"
    save_npz(actions_path, **{name: action.numpy() for name, action in actions.items()})
    object_to_goal = np.asarray(phase["goal_position_m"], dtype=np.float64) - np.asarray(
        phase["object_position_m"], dtype=np.float64
    )
    drives = {name: phase_drives(runner, action, object_to_goal) for name, action in actions.items()}
    primary_drive = PRIMARY_DRIVE[case["phase"]]
    clean_drive = drives["CLEAN_RECIPIENT"][primary_drive]
    phase_effects = {name: value[primary_drive] - clean_drive for name, value in drives.items()}
    predicted_sign = prediction["predicted_phase_response_sign"]
    injection_effect = phase_effects["CAUSAL_ROUTE_INJECTION"]
    observed_sign = "POSITIVE" if injection_effect > 0 else "NEGATIVE" if injection_effect < 0 else "ZERO"
    sign_correct = observed_sign == predicted_sign

    metrics = {
        name: vector_metrics(action.numpy(), clean.numpy(), physical.numpy())
        for name, action in actions.items()
        if name != "CLEAN_RECIPIENT"
    }
    result = {
        "status": "COMPLETED",
        "stage": "GROUP5C_CONTROLLED_EXECUTION_SMOKE" if smoke else "GROUP5C_CONTROLLED_EXECUTION",
        "model": model,
        "counterfactual_id": case["counterfactual_id"],
        "source_group4_case_id": case["source_group4_case_id"],
        "task_id": int(case["task_id"]),
        "recipient_state_id": case["recipient_state_id"],
        "recipient_heldout_base_state_id": case["recipient_heldout_base_state_id"],
        "factor": case["factor"],
        "phase": case["phase"],
        "relation_control": case["relation_control"],
        "signed_dose": float(case["signed_dose"]),
        "dose_unit": case["dose_unit"],
        "dose_role": case["dose_role"],
        "technical_counterfactual_status": "TECHNICALLY_VALID",
        "prediction_id": prediction["prediction_id"],
        "predicted_phase_response_sign": predicted_sign,
        "observed_injection_phase_response_sign": observed_sign,
        "sign_prediction_evaluable": True,
        "sign_prediction_correct": sign_correct,
        "primary_drive": primary_drive,
        "phase_drives": drives,
        "phase_effects_from_clean": phase_effects,
        "routing_metrics": metrics,
        "frozen_causal_route": prediction["frozen_causal_route"],
        "route_intervention_target": route,
        "secondary_route_targets": secondary,
        "frozen_noncausal_target": noncausal,
        "idm_decode_intervention_status": prediction["idm_decode_intervention_status"],
        "future_only_exclusivity_claimed": False,
        "action_hashes": {name: tensor_sha256(action) for name, action in actions.items()},
        "recipient_state_hash": case["recipient_state_hash"],
        "counterfactual_state_hash": case["counterfactual_state_hash"],
        "recipient_observation_hash": case["recipient_observation_hash"],
        "counterfactual_observation_hash": case["counterfactual_observation_hash"],
        "seed": int(source["seed"]),
        "checkpoint_hash": checkpoint_hash,
        "hardware_id": torch.cuda.get_device_name(0),
        "hardware_class": classify_hardware_name(torch.cuda.get_device_name(0)),
        "runner_sha256": runner_hash,
        "protocol_sha256": protocol_hash,
        "source_input_hashes": {str(path): expected for path, expected in EXPECTED_HASHES.items()},
        "actions_path": str(actions_path),
        "actions_sha256": sha256(actions_path),
        "group4a_result_path": str(source_path),
        "group4a_result_sha256": sha256(source_path),
        "same_value_identity_bit_exact": True,
        "smoke_label": smoke_label,
        "smoke_checks": smoke_checks,
        "simulator_calls": 0,
        "closed_loop_rollout_executed": False,
        "posthoc_case_selection": False,
        "prediction_redefined": False,
        "route_reselected": False,
        "dose_changed": False,
        "runtime_seconds": time.time() - started,
    }
    atomic_json(result_path, result)
    del engine
    gc.collect()
    torch.cuda.empty_cache()
    return result


def failure_result(
    model: str,
    case: dict[str, str],
    runner_hash: str,
    protocol_hash: str,
    error: BaseException,
) -> dict[str, Any]:
    target = result_root(model, case["counterfactual_id"], False, None)
    value = {
        "status": "TECHNICAL_MODEL_EXECUTION_FAILURE",
        "stage": "GROUP5C_CONTROLLED_EXECUTION",
        "model": model,
        "counterfactual_id": case["counterfactual_id"],
        "source_group4_case_id": case["source_group4_case_id"],
        "task_id": int(case["task_id"]),
        "factor": case["factor"],
        "phase": case["phase"],
        "signed_dose": float(case["signed_dose"]),
        "error_type": type(error).__name__,
        "error": str(error),
        "traceback": traceback.format_exc(),
        "runner_sha256": runner_hash,
        "protocol_sha256": protocol_hash,
        "case_replaced": False,
        "posthoc_case_selection": False,
    }
    atomic_json(target / "result.json", value)
    gc.collect()
    torch.cuda.empty_cache()
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument("--smoke-label")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=16)
    parser.add_argument("--max-cases", type=int)
    args = parser.parse_args()
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard specification")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu_id))
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(1)
    if not FROZEN_WAN_VAE.is_file():
        raise FileNotFoundError(FROZEN_WAN_VAE)

    cases, prediction_map, phases, source_hashes = verify_inputs()
    runner_path = Path(__file__).resolve()
    runner_hash = sha256(runner_path)
    protocol_hash = sha256(PROTOCOL)
    cases = sorted(cases, key=lambda row: row["counterfactual_id"])
    smoke = args.smoke_label is not None
    if smoke:
        # Exercise a proprio/attached-object-changing transform so that the
        # smoke cannot pass merely because an F2 donor leaves proprio fixed.
        cases = [
            row
            for row in cases
            if row["factor"] == "F3_OBJECT_GOAL_RADIAL_PROGRESS"
            and row["phase"] == "PREPLACE"
        ][:1]
        if len(cases) != 1:
            raise RuntimeError("STOP_GROUP5C_F3_SMOKE_CASE_MISSING")
    else:
        smoke_root = OUT / "technical_validation" / args.model
        smoke_summaries = sorted(smoke_root.glob("*/run_summary.json"))
        if not smoke_summaries:
            raise RuntimeError(f"STOP_GROUP5C_SMOKE_NOT_PRESENT:{args.model}")
        smoke_values = [json.loads(path.read_text(encoding="utf-8")) for path in smoke_summaries]
        if not any(
            value.get("status") == "PASS"
            and value.get("runner_sha256") == runner_hash
            and value.get("protocol_sha256") == protocol_hash
            for value in smoke_values
        ):
            raise RuntimeError(f"STOP_GROUP5C_SMOKE_NOT_FROZEN:{args.model}")
        cases = cases[args.shard_index :: args.num_shards]
        if args.max_cases is not None:
            cases = cases[: args.max_cases]

    os.chdir(FROZEN_LAUNCH_ROOT)
    runner = make_capture(args.model, 0)
    runner.model.eval()
    hardware_id = torch.cuda.get_device_name(0)
    hardware_class = classify_hardware_name(hardware_id)
    if hardware_class != HARDWARE[args.model]:
        raise RuntimeError(f"STOP_GROUP5C_HARDWARE_AUTHORITY:{args.model}:{hardware_id}")
    checkpoint_hash = model_weight_hash(runner.model)
    if checkpoint_hash != EXPECTED_CHECKPOINTS[args.model]:
        raise RuntimeError(f"STOP_GROUP5C_CHECKPOINT_HASH:{args.model}")

    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for index, case in enumerate(cases, 1):
        prediction = prediction_map[f"{args.model}::{case['counterfactual_id']}"]
        try:
            result = execute_case(
                runner,
                args.model,
                case,
                prediction,
                phases[case["recipient_state_id"]],
                checkpoint_hash,
                runner_hash,
                protocol_hash,
                smoke,
                args.smoke_label,
            )
            results.append(result)
        except Exception as error:
            if smoke:
                raise
            failure = failure_result(args.model, case, runner_hash, protocol_hash, error)
            failures.append(failure)
        print(
            json.dumps(
                {
                    "model": args.model,
                    "smoke": smoke,
                    "shard_index": args.shard_index,
                    "completed": index,
                    "total": len(cases),
                    "successful": len(results),
                    "technical_failures": len(failures),
                    "case_id": case["counterfactual_id"],
                }
            ),
            flush=True,
        )

    final_weight_hash = model_weight_hash(runner.model)
    summary = {
        "status": "PASS" if final_weight_hash == checkpoint_hash and not failures else "COMPLETE_WITH_TECHNICAL_FAILURES",
        "stage": "GROUP5C_CONTROLLED_EXECUTION_SMOKE" if smoke else "GROUP5C_CONTROLLED_EXECUTION_SHARD",
        "model": args.model,
        "smoke_label": args.smoke_label,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "assigned_cases": len(cases),
        "completed_cases": len(results),
        "technical_execution_failures": len(failures),
        "technical_failure_case_ids": [row["counterfactual_id"] for row in failures],
        "runner_sha256": runner_hash,
        "protocol_sha256": protocol_hash,
        "source_hashes": source_hashes,
        "checkpoint_hash": checkpoint_hash,
        "final_weight_hash": final_weight_hash,
        "model_weight_hash_unchanged": final_weight_hash == checkpoint_hash,
        "hardware_id": hardware_id,
        "hardware_class": hardware_class,
        "no_posthoc_case_selection": True,
        "no_prediction_redefinition": True,
        "no_route_reselection": True,
        "no_dose_change": True,
        "simulator_calls": 0,
    }
    if smoke:
        summary_path = OUT / "technical_validation" / args.model / str(args.smoke_label) / "run_summary.json"
    else:
        summary_path = OUT / "model_runs" / args.model / "shards" / f"shard_{args.shard_index}" / "run_summary.json"
    atomic_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    del runner, results
    gc.collect()
    torch.cuda.empty_cache()
    if not smoke and failures:
        raise SystemExit(3)
    if final_weight_hash != checkpoint_hash:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

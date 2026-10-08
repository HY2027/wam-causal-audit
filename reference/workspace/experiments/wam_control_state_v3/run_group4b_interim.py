from __future__ import annotations
from wam_causal_audit.paths import resolve as _release_path

import argparse
import gc
import hashlib
import json
import os
import sys
import time
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
from config import HARDWARE_AUTHORITY, RESULT_ROOT  # noqa: E402
from hardware_authority import classify_hardware_name  # noqa: E402
from protocol import load_npz, tensor_sha256  # noqa: E402
from run_group0 import FROZEN_LAUNCH_ROOT, FROZEN_WAN_VAE, model_weight_hash  # noqa: E402
from run_group1_model import phase_drives  # noqa: E402
from run_group2_joint import save_npz, scalar_decomposition, vector_metrics  # noqa: E402
from run_step2 import DirectEngine, ImageEngine, JointEngine  # noqa: E402


ROOT = Path(_release_path('@DATA@/wam_factor_routing_v5'))
G4A = ROOT / "group4a_heldout_factor_execution"
OUT = ROOT / "group4b_heldout_routing_interim"
AUTHORIZATION = OUT / "GROUP4B_THREE_MODEL_INTERIM_AUTHORIZED.json"
AUTHORIZATION_SHA = OUT / "GROUP4B_THREE_MODEL_INTERIM_AUTHORIZED.sha256"
PHASE_REGISTRY = G4A / "phase_registry.jsonl"
DONOR_REGISTRY = G4A / "donor_bank/counterfactual_qc.jsonl"
HELDOUT_REGISTRY = ROOT / "group4/group4_heldout_state_registry.csv"
HELDOUT_REGISTRY_SHA256 = "42f0e5c52f270727db4febe3c61f4e01fd392609cec71846f718cbb6c01b9fbf"
NONCAUSAL_REGISTRY = Path(_release_path('@DATA@/wam_control_state_v3/registries/noncausal_matched_v3.json'))
MODELS = ("direct", "joint", "imagewam")
EXPECTED_CASES = 544
EPSILON = 1e-12
SEED_BASES = {"direct": 710000, "joint": 720000, "imagewam": 740000}
ENGINE = {"direct": DirectEngine, "joint": JointEngine, "imagewam": ImageEngine}
ROUTES = {
    "direct": {
        "full": [f"DIRECT_L{i:02d}" for i in range(1, 7)],
        "components": {},
        "interpretation": "CURRENT_IMAGE_CARRIER_WITH_STRUCTURAL_ZERO_FUTURE_READ",
    },
    "joint": {
        "full": [f"JOINT_L{i:02d}" for i in range(1, 13)],
        "components": {
            "current": [f"JOINT_L{i:02d}" for i in (1, 3, 5, 7, 9, 11)],
            "future": [f"JOINT_L{i:02d}" for i in (2, 4, 6, 8, 10, 12)],
        },
        "interpretation": "CURRENT_WORLD_X_FUTURE_WORLD_2X2",
    },
    "imagewam": {
        "full": [f"IMAGEWAM_L{i:02d}" for i in range(1, 7)],
        "components": {
            "image": [f"IMAGEWAM_L{i:02d}" for i in range(1, 6)],
            "prefix": ["IMAGEWAM_L06"],
        },
        "interpretation": "IMAGE_KV_X_NON_IMAGE_PREFIX_KV_2X2",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def verify_authorization(runner_hash: str) -> str:
    expected = AUTHORIZATION_SHA.read_text(encoding="utf-8").split()[0]
    actual = sha256(AUTHORIZATION)
    payload = json.loads(AUTHORIZATION.read_text(encoding="utf-8"))
    if expected != actual or payload.get("status") != "GROUP4B_THREE_MODEL_INTERIM_AUTHORIZED":
        raise RuntimeError("STOP_GROUP4B_INTERIM_AUTHORIZATION_MISMATCH")
    if payload.get("models") != list(MODELS) or payload.get("idm_status") != "PENDING":
        raise RuntimeError("STOP_GROUP4B_INTERIM_MODEL_SCOPE_MISMATCH")
    if payload.get("heldout_registry_sha256") != HELDOUT_REGISTRY_SHA256:
        raise RuntimeError("STOP_GROUP4B_HELDOUT_REGISTRY_AUTHORIZATION_MISMATCH")
    if payload.get("runner_sha256") != runner_hash:
        raise RuntimeError("STOP_GROUP4B_RUNNER_NOT_FROZEN")
    required = (
        "primary_dose_only",
        "isolated_output_namespaces",
        "no_four_model_final_conclusion",
        "no_new_locus_search",
        "no_closed_loop_rollout",
    )
    if not all(payload.get(key) is True for key in required):
        raise RuntimeError("STOP_GROUP4B_AUTHORIZATION_BOUNDARY_MISMATCH")
    return actual


def group4a_source(model: str, case_id: str) -> tuple[Path, dict[str, Any], np.ndarray, np.ndarray]:
    result_path = G4A / "model_runs" / model / "cases" / case_id / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if (
        result.get("status") != "COMPLETED"
        or result.get("routing_intervention_executed") is not False
        or result.get("heldout_registry_sha256") != HELDOUT_REGISTRY_SHA256
        or result.get("hardware_authority_match") is not True
    ):
        raise RuntimeError(f"STOP_GROUP4A_SOURCE_INVALID:{model}:{case_id}")
    action_path = Path(result["actions_path"])
    if sha256(action_path) != result["actions_file_sha256"]:
        raise RuntimeError(f"STOP_GROUP4A_ACTION_HASH_MISMATCH:{model}:{case_id}")
    with np.load(action_path, allow_pickle=False) as archive:
        recipient = np.asarray(archive["RECIPIENT_NATIVE"], dtype=np.float32)
        donor = np.asarray(archive["DONOR_NATIVE"], dtype=np.float32)
    if tensor_sha256(torch.from_numpy(recipient)) != result["recipient_action_hash"]:
        raise RuntimeError(f"STOP_GROUP4A_RECIPIENT_ACTION_HASH:{model}:{case_id}")
    if tensor_sha256(torch.from_numpy(donor)) != result["donor_action_hash"]:
        raise RuntimeError(f"STOP_GROUP4A_DONOR_ACTION_HASH:{model}:{case_id}")
    return result_path, result, recipient, donor


def action_metrics(action: np.ndarray, recipient: np.ndarray, native_donor: np.ndarray) -> dict[str, Any]:
    delta = action.astype(np.float64) - recipient.astype(np.float64)
    donor_effect = native_donor.astype(np.float64) - recipient.astype(np.float64)
    metrics = vector_metrics(delta, donor_effect)
    closure = action.astype(np.float64) - native_donor.astype(np.float64)
    return {
        "directional_transfer": metrics["directional_projection"],
        "orthogonal_residual": metrics["orthogonal_residual"],
        "action_l2": float(np.linalg.norm(delta)),
        "max_elementwise_residual": float(np.max(np.abs(closure))),
        "closure_l2": float(np.linalg.norm(closure)),
        "relative_closure_l2": float(np.linalg.norm(closure) / (np.linalg.norm(donor_effect) + EPSILON)),
        "bit_exact_closure": bool(np.array_equal(action, native_donor)),
        "allclose_closure": bool(np.allclose(action, native_donor, rtol=1e-6, atol=1e-7)),
    }


def result_root(model: str, case_id: str, smoke: bool) -> Path:
    base = OUT / ("technical_validation" if smoke else "model_runs") / model
    return base / "cases" / case_id


def execute_case(
    runner: Any,
    model: str,
    donor: dict[str, Any],
    phase: dict[str, Any],
    checkpoint_hash: str,
    authorization_hash: str,
    runner_hash: str,
    noncausal: str,
    smoke: bool,
) -> dict[str, Any]:
    target = result_root(model, donor["case_id"], smoke)
    result_path = target / "result.json"
    if not smoke and result_path.is_file():
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if (
            existing.get("status") == "COMPLETED"
            and existing.get("authorization_sha256") == authorization_hash
            and existing.get("runner_sha256") == runner_hash
        ):
            return existing

    if donor.get("donor_valid") is not True or donor.get("primary_dose") is not True:
        raise RuntimeError(f"STOP_NONPRIMARY_OR_INVALID_DONOR:{donor['case_id']}")
    if donor["base_state_id"] != phase["base_state_id"]:
        raise RuntimeError(f"STOP_PHASE_IDENTITY:{donor['case_id']}")
    if donor["source_state_hash"] != phase["recipient_state_hash"]:
        raise RuntimeError(f"STOP_RECIPIENT_STATE_IDENTITY:{donor['case_id']}")
    if sha256(Path(phase["recipient_observation_path"])) != phase["recipient_observation_file_sha256"]:
        raise RuntimeError(f"STOP_RECIPIENT_OBSERVATION_HASH:{donor['case_id']}")
    if sha256(Path(donor["donor_observation_path"])) != donor["donor_observation_file_sha256"]:
        raise RuntimeError(f"STOP_DONOR_OBSERVATION_HASH:{donor['case_id']}")

    source_path, source, frozen_recipient, frozen_donor = group4a_source(model, donor["case_id"])
    expected_seed = int(
        SEED_BASES[model]
        + int(phase["task_id"]) * 10_000
        + int(phase["source_state_id"]) * 100
        + int(phase["source_step_before_action"])
    )
    if int(source["seed"]) != expected_seed:
        raise RuntimeError(f"STOP_SEED_IDENTITY:{donor['case_id']}")
    recipient_observation = load_npz(Path(phase["recipient_observation_path"]))
    donor_observation = load_npz(Path(donor["donor_observation_path"]))
    started = time.time()
    engine = ENGINE[model](
        runner,
        recipient_observation,
        donor_observation,
        phase["instruction"],
        int(source["seed"]),
    )
    a00_tensor = engine.clean.detach().cpu().float()
    if not np.array_equal(a00_tensor.numpy(), frozen_recipient):
        raise RuntimeError(f"STOP_A00_GROUP4A_BIT_EXACT_MISMATCH:{model}:{donor['case_id']}")

    route = ROUTES[model]
    tensors: dict[str, torch.Tensor] = {"A00": a00_tensor}
    if model == "direct":
        tensors["CARRIER"] = engine.infer("clean", "fault", route["full"]).detach().cpu().float()
    else:
        names = list(route["components"])
        tensors["A10"] = engine.infer("clean", "fault", route["components"][names[0]]).detach().cpu().float()
        tensors["A01"] = engine.infer("clean", "fault", route["components"][names[1]]).detach().cpu().float()
        tensors["A11"] = engine.infer("clean", "fault", route["full"]).detach().cpu().float()
    tensors["NEGATIVE"] = engine.infer("clean", "fault", [noncausal]).detach().cpu().float()

    smoke_checks: dict[str, bool] = {}
    if smoke:
        identities = {
            "full_same_value_bit_exact": torch.equal(
                engine.infer("clean", "clean", route["full"]).detach().cpu().float(), a00_tensor
            ),
            "negative_same_value_bit_exact": torch.equal(
                engine.infer("clean", "clean", [noncausal]).detach().cpu().float(), a00_tensor
            ),
        }
        for name, loci in route["components"].items():
            identities[f"{name}_same_value_bit_exact"] = torch.equal(
                engine.infer("clean", "clean", loci).detach().cpu().float(), a00_tensor
            )
        if route["components"]:
            sets = [set(value) for value in route["components"].values()]
            component_disjoint = not bool(sets[0] & sets[1])
            union_matches_full = set().union(*sets) == set(route["full"])
        else:
            component_disjoint = True
            union_matches_full = True
        full_key = "CARRIER" if model == "direct" else "A11"
        full_repeat = engine.infer("clean", "fault", route["full"]).detach().cpu().float()
        smoke_checks = {
            "A00_reproduces_group4a_recipient_bit_exact": True,
            **identities,
            "routing_components_disjoint": component_disjoint,
            "routing_component_union_matches_full_carrier": union_matches_full,
            "donor_full_carrier_repeat_bit_exact": torch.equal(full_repeat, tensors[full_key]),
            "all_actions_finite": all(torch.isfinite(action).all().item() for action in tensors.values()),
            "heldout_registry_hash_fixed": sha256(HELDOUT_REGISTRY) == HELDOUT_REGISTRY_SHA256,
            "primary_dose_only": True,
            "no_simulator_execution": True,
        }
        if not all(smoke_checks.values()):
            raise RuntimeError(f"STOP_GROUP4B_SMOKE_FAILED:{model}:{smoke_checks}")

    arrays = {name: action.numpy() for name, action in tensors.items()}
    arrays["DONOR_NATIVE"] = frozen_donor
    if model != "direct":
        a00 = arrays["A00"].astype(np.float64)
        a10 = arrays["A10"].astype(np.float64)
        a01 = arrays["A01"].astype(np.float64)
        a11 = arrays["A11"].astype(np.float64)
        first, second = list(route["components"])
        arrays[f"phi_{first}"] = 0.5 * ((a10 - a00) + (a11 - a01))
        arrays[f"phi_{second}"] = 0.5 * ((a01 - a00) + (a11 - a10))
        arrays["interaction"] = a11 - a10 - a01 + a00
        arrays["total"] = a11 - a00

    target.mkdir(parents=True, exist_ok=True)
    actions_path = target / "actions.npz"
    save_npz(actions_path, **arrays)
    object_to_goal = np.asarray(phase["goal_position_m"], dtype=np.float64) - np.asarray(
        phase["object_position_m"], dtype=np.float64
    )
    drive_values = {
        name: phase_drives(runner, action, object_to_goal)
        for name, action in tensors.items()
        if name != "NEGATIVE"
    }
    metrics = {
        name: action_metrics(action.numpy(), arrays["A00"], frozen_donor)
        for name, action in tensors.items()
        if name != "A00"
    }
    decomposition = scalar_decomposition(drive_values) if model != "direct" else None
    result = {
        "status": "COMPLETED",
        "stage": "GROUP4B_THREE_MODEL_INTERIM_HELDOUT_ROUTING_SMOKE" if smoke else "GROUP4B_THREE_MODEL_INTERIM_HELDOUT_ROUTING",
        "model": model,
        "case_id": donor["case_id"],
        "task_id": int(donor["task_id"]),
        "heldout_base_state_id": phase["heldout_base_state_id"],
        "base_state_id": donor["base_state_id"],
        "factor": donor["factor"],
        "phase": donor["phase"],
        "control_condition": donor["relation_control"],
        "signed_dose": float(donor["signed_dose"]),
        "dose_unit": donor["dose_unit"],
        "primary_dose": True,
        "route_interpretation": route["interpretation"],
        "current_future_interpretation_claimed": model == "joint",
        "explicit_future_read": "STRUCTURAL_ZERO" if model == "direct" else "PRESENT_OR_NOT_APPLICABLE",
        "full_carrier_loci": route["full"],
        "routing_components": route["components"],
        "negative_control_locus": noncausal,
        "negative_control_source": str(NONCAUSAL_REGISTRY),
        "authorization_sha256": authorization_hash,
        "runner_sha256": runner_hash,
        "heldout_registry_sha256": HELDOUT_REGISTRY_SHA256,
        "checkpoint_hash": checkpoint_hash,
        "hardware_id": torch.cuda.get_device_name(0),
        "hardware_class": classify_hardware_name(torch.cuda.get_device_name(0)),
        "seed": int(source["seed"]),
        "recipient_state_hash": source["recipient_state_hash"],
        "donor_state_hash": source["donor_state_hash"],
        "action_hashes": {name: tensor_sha256(action) for name, action in tensors.items()},
        "native_donor_action_hash": source["donor_action_hash"],
        "routing_metrics": metrics,
        "scalar_routing": decomposition,
        "phase_drives": drive_values,
        "smoke_checks": smoke_checks,
        "actions_path": str(actions_path),
        "actions_sha256": sha256(actions_path),
        "group4a_result_path": str(source_path),
        "group4a_result_sha256": sha256(source_path),
        "group4a_actions_path": source["actions_path"],
        "group4a_actions_sha256": source["actions_file_sha256"],
        "counterfactual_qc_path": donor["counterfactual_qc_path"],
        "counterfactual_qc_sha256": sha256(Path(donor["counterfactual_qc_path"])),
        "model_forward_calls_new": (
            (7 if model == "direct" else 11) if smoke else (4 if model == "direct" else 6)
        ),
        "simulator_calls": 0,
        "closed_loop_rollout_executed": False,
        "idm_status": "PENDING",
        "four_model_conclusion_issued": False,
        "runtime_seconds": time.time() - started,
    }
    atomic_json(result_path, result)
    del engine
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
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
        raise FileNotFoundError(FROZEN_WAN_VAE)
    if sha256(HELDOUT_REGISTRY) != HELDOUT_REGISTRY_SHA256:
        raise RuntimeError("STOP_GROUP4B_HELDOUT_REGISTRY_HASH")
    runner_path = Path(__file__).resolve()
    runner_hash = sha256(runner_path)
    authorization_hash = verify_authorization(runner_hash)
    noncausal_payload = json.loads(NONCAUSAL_REGISTRY.read_text(encoding="utf-8"))
    noncausal = noncausal_payload["selections"][args.model]["locus_id"]
    phases = {row["base_state_id"]: row for row in read_jsonl(PHASE_REGISTRY)}
    donors = sorted(
        [
            row
            for row in read_jsonl(DONOR_REGISTRY)
            if row.get("donor_valid") is True and row.get("primary_dose") is True
        ],
        key=lambda row: row["case_id"],
    )
    if len(donors) != EXPECTED_CASES:
        raise RuntimeError(f"STOP_GROUP4B_PRIMARY_CASE_CARDINALITY:{len(donors)}")
    if args.smoke:
        donors = donors[:1]
    else:
        smoke_summary = OUT / "technical_validation" / args.model / "run_summary.json"
        smoke = json.loads(smoke_summary.read_text(encoding="utf-8"))
        if (
            smoke.get("status") != "PASS"
            or smoke.get("authorization_sha256") != authorization_hash
            or smoke.get("runner_sha256") != runner_hash
        ):
            raise RuntimeError(f"STOP_GROUP4B_SMOKE_NOT_FROZEN:{args.model}")
        if args.max_cases is not None:
            donors = donors[: args.max_cases]

    os.chdir(FROZEN_LAUNCH_ROOT)
    group0_path = RESULT_ROOT / "technical_validation" / args.model / "result.json"
    group0 = json.loads(group0_path.read_text(encoding="utf-8"))
    if group0.get("status") != "PASS" or not all(group0.get("checks", {}).values()):
        raise RuntimeError(f"STOP_GROUP0_NOT_PASS:{args.model}")
    runner = make_capture(args.model, 0)
    runner.model.eval()
    hardware_id = torch.cuda.get_device_name(0)
    hardware_class = classify_hardware_name(hardware_id)
    if hardware_class != HARDWARE_AUTHORITY[args.model] or hardware_class != "RTX4090":
        raise RuntimeError(f"STOP_GROUP4B_HARDWARE_AUTHORITY:{args.model}:{hardware_id}")
    checkpoint_hash = model_weight_hash(runner.model)
    if checkpoint_hash != group0["checkpoint_hash"]:
        raise RuntimeError(f"STOP_GROUP4B_CHECKPOINT_HASH:{args.model}")

    results: list[dict[str, Any]] = []
    for index, donor in enumerate(donors, 1):
        result = execute_case(
            runner,
            args.model,
            donor,
            phases[donor["base_state_id"]],
            checkpoint_hash,
            authorization_hash,
            runner_hash,
            noncausal,
            args.smoke,
        )
        results.append(result)
        print(
            json.dumps(
                {
                    "model": args.model,
                    "stage": "GROUP4B_INTERIM_SMOKE" if args.smoke else "GROUP4B_INTERIM",
                    "completed": index,
                    "total": len(donors),
                    "case_id": donor["case_id"],
                    "runtime_seconds": result["runtime_seconds"],
                }
            ),
            flush=True,
        )
    final_hash = model_weight_hash(runner.model)
    summary = {
        "status": "PASS" if final_hash == checkpoint_hash else "TECHNICAL_STOP",
        "stage": "GROUP4B_THREE_MODEL_INTERIM_HELDOUT_ROUTING_SMOKE" if args.smoke else "GROUP4B_THREE_MODEL_INTERIM_HELDOUT_ROUTING",
        "model": args.model,
        "completed_cases": len(results),
        "eligible_primary_cases": len(donors),
        "authorization_sha256": authorization_hash,
        "runner_sha256": runner_hash,
        "checkpoint_hash": checkpoint_hash,
        "final_weight_hash": final_hash,
        "model_weight_hash_unchanged": final_hash == checkpoint_hash,
        "hardware_id": hardware_id,
        "hardware_class": hardware_class,
        "heldout_registry_sha256": HELDOUT_REGISTRY_SHA256,
        "full_carrier_loci": ROUTES[args.model]["full"],
        "routing_components": ROUTES[args.model]["components"],
        "negative_control_locus": noncausal,
        "primary_dose_only": True,
        "simulator_calls": 0,
        "closed_loop_rollouts": 0,
        "idm_status": "PENDING",
        "four_model_conclusion_issued": False,
        "output_namespace": str(OUT / ("technical_validation" if args.smoke else "model_runs") / args.model),
    }
    summary_path = OUT / ("technical_validation" if args.smoke else "model_runs") / args.model / "run_summary.json"
    atomic_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    del runner, results
    gc.collect()
    torch.cuda.empty_cache()
    if summary["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

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
from hardware_authority import classify_hardware_name  # noqa: E402
from protocol import load_npz, tensor_sha256  # noqa: E402
from run_group0 import FROZEN_LAUNCH_ROOT, FROZEN_WAN_VAE, model_weight_hash  # noqa: E402
from run_group1_model import phase_drives  # noqa: E402
from run_group2_joint import read_jsonl, save_npz, sha256, vector_metrics, write_json  # noqa: E402
from run_step2 import IDMEngine  # noqa: E402


G1 = Path(_release_path('@DATA@/wam_control_state_v3/group1_factor_phase'))
V5 = Path(_release_path('@DATA@/wam_factor_routing_v5'))
BASE_AUTH = V5 / "group2/GROUP2_ROUTING_AUTHORIZED.json"
BASE_AUTH_SHA = V5 / "group2/GROUP2_ROUTING_AUTHORIZED.sha256"
CONCURRENCY_AUTH = V5 / "group2/GROUP2_CROSS_MODEL_CONCURRENCY_AUTHORIZED.json"
CONCURRENCY_AUTH_SHA = V5 / "group2/GROUP2_CROSS_MODEL_CONCURRENCY_AUTHORIZED.sha256"
OUT = V5 / "group2/idm_future_mediated"
DUAL_GPU_GATE = OUT / "technical_gates/idm_blackwell_dual_gpu_smoke_gate.json"
FUTURE_LOCUS = ["IDM_L14"]
EXPECTED_CASES = 1727
EPSILON = 1e-12


def verify_authorizations() -> tuple[str, str]:
    base_expected = BASE_AUTH_SHA.read_text(encoding="utf-8").split()[0]
    concurrency_expected = CONCURRENCY_AUTH_SHA.read_text(encoding="utf-8").split()[0]
    base_actual = sha256(BASE_AUTH)
    concurrency_actual = sha256(CONCURRENCY_AUTH)
    base = json.loads(BASE_AUTH.read_text(encoding="utf-8"))
    concurrency = json.loads(CONCURRENCY_AUTH.read_text(encoding="utf-8"))
    if base_expected != base_actual or base.get("status") != "GROUP2_ROUTING_AUTHORIZED":
        raise RuntimeError("GROUP2_BASE_AUTHORIZATION_MISMATCH")
    if concurrency_expected != concurrency_actual or concurrency.get("status") != "GROUP2_CROSS_MODEL_CONCURRENCY_AUTHORIZED":
        raise RuntimeError("GROUP2_CONCURRENCY_AUTHORIZATION_MISMATCH")
    if base.get("idm_context_route_status") != "IDM_CONTEXT_ROUTE_NOT_SEPARABLE":
        raise RuntimeError("IDM_CONTEXT_ROUTE_STATUS_MISMATCH")
    if base.get("idm_routing_statement") != "FUTURE_MEDIATED_ROUTE" or base.get("idm_exclusivity_claimed"):
        raise RuntimeError("IDM_ROUTING_INTERPRETATION_MISMATCH")
    if base.get("idm_decode_mechanistic_inference") != "N/A":
        raise RuntimeError("IDM_DECODE_EXCLUSION_MISMATCH")
    return base_actual, concurrency_actual


def frozen_group1(case_id: str) -> tuple[Path, dict[str, Any], dict[str, np.ndarray]]:
    result_path = G1 / "model_runs/idm/cases" / case_id / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if (
        result.get("status") != "COMPLETED"
        or result.get("causal_cut") != FUTURE_LOCUS
        or result.get("hardware_class") != "BLACKWELL"
    ):
        raise RuntimeError(f"IDM_GROUP1_SOURCE_INVALID:{case_id}")
    action_path = Path(result["actions_path"])
    if sha256(action_path) != result["actions_file_sha256"]:
        raise RuntimeError(f"IDM_GROUP1_ACTION_ARTIFACT_HASH:{case_id}")
    with np.load(action_path, allow_pickle=False) as archive:
        arrays = {
            "RECIPIENT_NATIVE": np.asarray(archive["RECIPIENT_NATIVE"], dtype=np.float32),
            "DONOR_NATIVE": np.asarray(archive["DONOR_NATIVE"], dtype=np.float32),
            "FULL_CAUSAL_CUT_SWAP": np.asarray(archive["FULL_CAUSAL_CUT_SWAP"], dtype=np.float32),
        }
    expected = {
        "RECIPIENT_NATIVE": result["recipient_action_hash"],
        "DONOR_NATIVE": result["donor_action_hash"],
        "FULL_CAUSAL_CUT_SWAP": result["all_intervened_action_hashes"]["FULL_CAUSAL_CUT_SWAP"],
    }
    for name, value in arrays.items():
        if tensor_sha256(torch.from_numpy(value)) != expected[name]:
            raise RuntimeError(f"IDM_GROUP1_ACTION_TENSOR_HASH:{case_id}:{name}")
    return result_path, result, arrays


def action_metrics(action: np.ndarray, recipient: np.ndarray, donor: np.ndarray) -> dict[str, Any]:
    delta = action.astype(np.float64) - recipient.astype(np.float64)
    donor_effect = donor.astype(np.float64) - recipient.astype(np.float64)
    projection = vector_metrics(delta, donor_effect)
    residual = action.astype(np.float64) - donor.astype(np.float64)
    return {
        "directional_transfer": projection["directional_projection"],
        "orthogonal_residual": projection["orthogonal_residual"],
        "action_l2": float(np.linalg.norm(delta)),
        "donor_action_residual_l2": float(np.linalg.norm(residual)),
        "donor_action_relative_residual": float(np.linalg.norm(residual) / (np.linalg.norm(donor_effect) + EPSILON)),
        "max_elementwise_residual": float(np.max(np.abs(residual))),
        "bit_exact_closure": bool(np.array_equal(action, donor)),
        "allclose_closure": bool(np.allclose(action, donor, rtol=1e-6, atol=1e-7)),
    }


def execute_case(
    runner: Any,
    donor: dict[str, Any],
    phase: dict[str, Any],
    checkpoint_hash: str,
    base_hash: str,
    concurrency_hash: str,
    runner_hash: str,
    smoke_label: str | None,
) -> dict[str, Any]:
    smoke = smoke_label is not None
    target = OUT / (f"technical_smoke/{smoke_label}" if smoke else "cases") / donor["case_id"]
    result_path = target / "result.json"
    if not smoke and result_path.is_file():
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if (
            existing.get("status") == "COMPLETED"
            and existing.get("runner_sha256") == runner_hash
            and existing.get("base_authorization_sha256") == base_hash
        ):
            return existing

    started = time.time()
    recipient_observation = load_npz(Path(phase["recipient_observation_path"]))
    donor_observation = load_npz(Path(donor["donor_observation_path"]))
    source_path, source, frozen = frozen_group1(donor["case_id"])
    engine = IDMEngine(
        runner,
        recipient_observation,
        donor_observation,
        phase["instruction"],
        int(phase["policy_seed"]),
    )
    recipient = engine.clean.detach().cpu().float()
    future_intervention = engine.fault.detach().cpu().float()
    full_carrier = engine.infer("clean", "fault", FUTURE_LOCUS).detach().cpu().float()
    if not np.array_equal(recipient.numpy(), frozen["RECIPIENT_NATIVE"]):
        raise RuntimeError(f"IDM_GROUP2_A00_GROUP1_BIT_EXACT:{donor['case_id']}")
    if not np.array_equal(full_carrier.numpy(), frozen["FULL_CAUSAL_CUT_SWAP"]):
        raise RuntimeError(f"IDM_GROUP2_FULL_CARRIER_GROUP1_BIT_EXACT:{donor['case_id']}")

    recipient_future_hash = engine.locus_hash("clean", FUTURE_LOCUS[0])
    donor_future_hash = engine.locus_hash("fault", FUTURE_LOCUS[0])
    expected_loci = source["locus_tensor_hashes"]
    if recipient_future_hash != expected_loci["recipient"][FUTURE_LOCUS[0]]:
        raise RuntimeError(f"IDM_GROUP2_RECIPIENT_FUTURE_HASH:{donor['case_id']}")
    if donor_future_hash != expected_loci["donor"][FUTURE_LOCUS[0]]:
        raise RuntimeError(f"IDM_GROUP2_DONOR_FUTURE_HASH:{donor['case_id']}")

    smoke_checks: dict[str, bool] = {}
    if smoke:
        identity = engine.infer("clean", "clean", FUTURE_LOCUS).detach().cpu().float()
        full_repeat = engine.infer("clean", "fault", FUTURE_LOCUS).detach().cpu().float()
        smoke_checks = {
            "recipient_reproduces_group1_bit_exact": True,
            "full_carrier_reproduces_group1_bit_exact": True,
            "same_value_identity_bit_exact": bool(torch.equal(identity, recipient)),
            "full_carrier_repeat_bit_exact": bool(torch.equal(full_repeat, full_carrier)),
            "future_intervention_matches_full_carrier_bit_exact": bool(torch.equal(future_intervention, full_carrier)),
            "recipient_future_hash_matches_group1": True,
            "donor_future_hash_matches_group1": True,
            "all_actions_finite": bool(
                torch.isfinite(recipient).all()
                and torch.isfinite(future_intervention).all()
                and torch.isfinite(full_carrier).all()
            ),
            "context_route_not_artificially_separated": True,
            "decode_locus_excluded": True,
            "simulator_calls_zero": True,
        }
        if not all(smoke_checks.values()):
            raise RuntimeError(f"IDM_GROUP2_SMOKE_FAILED:{smoke_checks}")

    tensors = {
        "RECIPIENT_FUTURE_MEDIATED": recipient,
        "DONOR_FUTURE_INTERVENTION": future_intervention,
        "FULL_FUTURE_MEDIATED_CARRIER": full_carrier,
    }
    target.mkdir(parents=True, exist_ok=True)
    actions_path = target / "actions.npz"
    save_npz(
        actions_path,
        RECIPIENT_NATIVE=recipient.numpy(),
        DONOR_NATIVE=frozen["DONOR_NATIVE"],
        DONOR_FUTURE_INTERVENTION=future_intervention.numpy(),
        FULL_FUTURE_MEDIATED_CARRIER=full_carrier.numpy(),
    )
    object_to_goal = np.asarray(phase["goal_position_m"], dtype=np.float64) - np.asarray(
        phase["object_position_m"], dtype=np.float64
    )
    phase_values = {
        name: phase_drives(runner, action, object_to_goal)
        for name, action in tensors.items()
    }
    result = {
        "status": "COMPLETED",
        "stage": "GROUP2B_IDM_FUTURE_MEDIATED_SMOKE" if smoke else "GROUP2B_IDM_FUTURE_MEDIATED_ROUTE",
        "model": "idm",
        "case_id": donor["case_id"],
        "task_id": int(donor["task_id"]),
        "source_state_id": int(donor["source_state_id"]),
        "base_state_id": donor["base_state_id"],
        "factor": donor["factor"],
        "phase": donor["phase"],
        "control_condition": donor["relation_control"],
        "signed_dose": float(donor["signed_dose"]),
        "primary_dose": True,
        "route_statement": "FUTURE_MEDIATED_ROUTE",
        "future_only_route_share_claimed": False,
        "context_route_status": "IDM_CONTEXT_ROUTE_NOT_SEPARABLE",
        "artificial_context_future_2x2_constructed": False,
        "decode_transfer_status": "UNINFORMATIVE_SAME_VALUE_NOOP",
        "decode_transfer_used_for_inference": False,
        "future_carrier_loci": FUTURE_LOCUS,
        "recipient_future_latent_hash": recipient_future_hash,
        "donor_future_latent_hash": donor_future_hash,
        "future_latent_values_differ": recipient_future_hash != donor_future_hash,
        "routing_metrics": {
            "donor_future_intervention": action_metrics(
                future_intervention.numpy(), recipient.numpy(), frozen["DONOR_NATIVE"]
            ),
            "full_future_mediated_carrier": action_metrics(
                full_carrier.numpy(), recipient.numpy(), frozen["DONOR_NATIVE"]
            ),
        },
        "phase_drives": phase_values,
        "base_authorization_sha256": base_hash,
        "concurrency_authorization_sha256": concurrency_hash,
        "runner_sha256": runner_hash,
        "checkpoint_hash": checkpoint_hash,
        "hardware_id": torch.cuda.get_device_name(0),
        "hardware_class": classify_hardware_name(torch.cuda.get_device_name(0)),
        "seed": int(phase["policy_seed"]),
        "recipient_state_hash": phase["recipient_state_hash"],
        "donor_state_hash": donor["donor_state_hash"],
        "action_hashes": {name: tensor_sha256(value) for name, value in tensors.items()},
        "native_donor_action_hash": source["donor_action_hash"],
        "actions_path": str(actions_path),
        "actions_sha256": sha256(actions_path),
        "group1_result_path": str(source_path),
        "group1_result_sha256": sha256(source_path),
        "smoke_label": smoke_label,
        "smoke_checks": smoke_checks,
        "model_forward_calls_new": 6 if smoke else 4,
        "simulator_calls": 0,
        "closed_loop_rollout_executed": False,
        "runtime_seconds": time.time() - started,
    }
    write_json(result_path, result)
    del engine
    gc.collect()
    torch.cuda.empty_cache()
    return result


def verify_dual_gpu_gate(runner_hash: str) -> str:
    gate = json.loads(DUAL_GPU_GATE.read_text(encoding="utf-8"))
    if (
        gate.get("status") != "IDM_GROUP2_DUAL_BLACKWELL_SMOKE_PASS"
        or gate.get("runner_sha256") != runner_hash
        or gate.get("formal_num_shards") != 2
        or gate.get("cross_gpu_action_hashes_bit_exact") is not True
        or gate.get("shards_disjoint_and_complete") is not True
    ):
        raise RuntimeError("IDM_GROUP2_DUAL_GPU_GATE_MISMATCH")
    return sha256(DUAL_GPU_GATE)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument("--smoke-label", choices=("gpu4", "gpu5"))
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
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
    base_hash, concurrency_hash = verify_authorizations()
    runner_path = Path(__file__).resolve()
    runner_hash = sha256(runner_path)
    phases = {row["base_state_id"]: row for row in read_jsonl(G1 / "phase_registry.jsonl")}
    donors = sorted(
        [
            row
            for row in read_jsonl(G1 / "donor_bank/idm/counterfactual_qc.jsonl")
            if row.get("donor_valid") is True and row.get("primary_dose") is True
        ],
        key=lambda row: row["case_id"],
    )
    if len(donors) != EXPECTED_CASES or len({row["case_id"] for row in donors}) != EXPECTED_CASES:
        raise RuntimeError(f"IDM_GROUP2_CASE_CARDINALITY:{len(donors)}")
    smoke = args.smoke_label is not None
    if smoke:
        donors = donors[:1]
    else:
        if args.num_shards != 2 or args.shard_index not in (0, 1):
            raise RuntimeError("IDM_GROUP2_FORMAL_REQUIRES_TWO_FROZEN_SHARDS")
        gate_hash = verify_dual_gpu_gate(runner_hash)
        donors = [row for index, row in enumerate(donors) if index % 2 == args.shard_index]
        if len(donors) != (864 if args.shard_index == 0 else 863):
            raise RuntimeError("IDM_GROUP2_SHARD_CARDINALITY")
        if args.max_cases is not None:
            donors = donors[: args.max_cases]
    os.chdir(FROZEN_LAUNCH_ROOT)
    runner = make_capture("idm", 0)
    runner.model.eval()
    hardware_id = torch.cuda.get_device_name(0)
    if classify_hardware_name(hardware_id) != "BLACKWELL":
        raise RuntimeError(f"IDM_GROUP2_HARDWARE_AUTHORITY:{hardware_id}")
    checkpoint_hash = model_weight_hash(runner.model)
    group0_path = Path(_release_path('@DATA@/wam_control_state_v3/technical_validation/idm/result.json'))
    group0 = json.loads(group0_path.read_text(encoding="utf-8"))
    if group0.get("status") != "PASS" or checkpoint_hash != group0["checkpoint_hash"]:
        raise RuntimeError("IDM_GROUP2_CHECKPOINT_HASH_MISMATCH")

    results: list[dict[str, Any]] = []
    for index, donor in enumerate(donors, 1):
        result = execute_case(
            runner,
            donor,
            phases[donor["base_state_id"]],
            checkpoint_hash,
            base_hash,
            concurrency_hash,
            runner_hash,
            args.smoke_label,
        )
        results.append(result)
        print(json.dumps({
            "model": "idm",
            "stage": "GROUP2B_SMOKE" if smoke else "GROUP2B_FUTURE_MEDIATED",
            "shard_index": None if smoke else args.shard_index,
            "completed": index,
            "total": len(donors),
            "case_id": donor["case_id"],
            "runtime_seconds": result["runtime_seconds"],
        }), flush=True)
    final_hash = model_weight_hash(runner.model)
    summary = {
        "status": "PASS" if final_hash == checkpoint_hash else "TECHNICAL_STOP",
        "stage": "GROUP2B_IDM_FUTURE_MEDIATED_SMOKE" if smoke else "GROUP2B_IDM_FUTURE_MEDIATED_ROUTE_SHARD",
        "model": "idm",
        "smoke_label": args.smoke_label,
        "num_shards": None if smoke else args.num_shards,
        "shard_index": None if smoke else args.shard_index,
        "completed_cases": len(results),
        "eligible_cases": len(donors),
        "base_authorization_sha256": base_hash,
        "concurrency_authorization_sha256": concurrency_hash,
        "runner_sha256": runner_hash,
        "dual_gpu_gate_sha256": None if smoke else gate_hash,
        "checkpoint_hash": checkpoint_hash,
        "final_weight_hash": final_hash,
        "model_weight_hash_unchanged": final_hash == checkpoint_hash,
        "hardware_id": hardware_id,
        "hardware_class": classify_hardware_name(hardware_id),
        "route_statement": "FUTURE_MEDIATED_ROUTE",
        "future_only_route_share_claimed": False,
        "context_route_status": "IDM_CONTEXT_ROUTE_NOT_SEPARABLE",
        "decode_transfer_used_for_inference": False,
        "output_namespace": str(
            OUT / (f"technical_smoke/{args.smoke_label}" if smoke else f"shards/shard_{args.shard_index}")
        ),
    }
    summary_path = OUT / (
        f"technical_smoke/{args.smoke_label}/run_summary.json"
        if smoke
        else f"shards/shard_{args.shard_index}/run_summary.json"
    )
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    del runner, results
    gc.collect()
    torch.cuda.empty_cache()
    if summary["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

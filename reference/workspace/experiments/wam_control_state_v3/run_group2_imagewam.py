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
from run_group2_joint import (  # noqa: E402
    read_jsonl,
    save_npz,
    scalar_decomposition,
    sha256,
    vector_metrics,
    write_json,
)
from run_step2 import ImageEngine  # noqa: E402


G1_RAW = Path(_release_path('@DATA@/wam_control_state_v3/group1_factor_phase'))
V5 = Path(_release_path('@DATA@/wam_factor_routing_v5'))
BASE_AUTH = V5 / "group2/GROUP2_ROUTING_AUTHORIZED.json"
BASE_AUTH_SHA = V5 / "group2/GROUP2_ROUTING_AUTHORIZED.sha256"
CONCURRENCY_AUTH = V5 / "group2/GROUP2_CROSS_MODEL_CONCURRENCY_AUTHORIZED.json"
CONCURRENCY_AUTH_SHA = V5 / "group2/GROUP2_CROSS_MODEL_CONCURRENCY_AUTHORIZED.sha256"
OUT = V5 / "group2/imagewam_image_prefix"
IMAGE = [f"IMAGEWAM_L{i:02d}" for i in range(1, 6)]
PREFIX = ["IMAGEWAM_L06"]
FULL = [f"IMAGEWAM_L{i:02d}" for i in range(1, 7)]
EPS = 1e-12


def verify_authorizations() -> tuple[str, str]:
    base_expected = BASE_AUTH_SHA.read_text(encoding="utf-8").split()[0]
    concurrency_expected = CONCURRENCY_AUTH_SHA.read_text(encoding="utf-8").split()[0]
    base_actual = sha256(BASE_AUTH)
    concurrency_actual = sha256(CONCURRENCY_AUTH)
    base = json.loads(BASE_AUTH.read_text(encoding="utf-8"))
    concurrency = json.loads(CONCURRENCY_AUTH.read_text(encoding="utf-8"))
    if base_expected != base_actual or base.get("status") != "GROUP2_ROUTING_AUTHORIZED":
        raise RuntimeError("GROUP2_BASE_AUTHORIZATION_MISMATCH")
    if (concurrency_expected != concurrency_actual
            or concurrency.get("status") != "GROUP2_CROSS_MODEL_CONCURRENCY_AUTHORIZED"):
        raise RuntimeError("GROUP2_CONCURRENCY_AUTHORIZATION_MISMATCH")
    if concurrency.get("final_scientific_review_order") != ["joint", "idm", "imagewam", "direct"]:
        raise RuntimeError("GROUP2_REVIEW_ORDER_MISMATCH")
    return base_actual, concurrency_actual


def result_dir(case_id: str, smoke: bool) -> Path:
    return OUT / ("technical_smoke" if smoke else "cases") / case_id


def execute_case(
    runner: Any,
    donor: dict[str, Any],
    phase: dict[str, Any],
    checkpoint_hash: str,
    base_authorization_hash: str,
    concurrency_authorization_hash: str,
    runner_hash: str,
    smoke: bool,
) -> dict[str, Any]:
    target = result_dir(donor["case_id"], smoke)
    result_path = target / "result.json"
    if not smoke and result_path.is_file():
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if (existing.get("status") == "COMPLETED"
                and existing.get("base_authorization_sha256") == base_authorization_hash
                and existing.get("concurrency_authorization_sha256") == concurrency_authorization_hash
                and existing.get("runner_sha256") == runner_hash):
            return existing

    started = time.time()
    recipient_observation = load_npz(Path(phase["recipient_observation_path"]))
    donor_observation = load_npz(Path(donor["donor_observation_path"]))
    engine = ImageEngine(
        runner,
        recipient_observation,
        donor_observation,
        phase["instruction"],
        int(phase["policy_seed"]),
    )

    group1_path = G1_RAW / "model_runs/imagewam/cases" / donor["case_id"] / "result.json"
    group1 = json.loads(group1_path.read_text(encoding="utf-8"))
    with np.load(group1["actions_path"]) as frozen:
        frozen_a00 = np.asarray(frozen["RECIPIENT_NATIVE"], dtype=np.float32)
        frozen_a11 = np.asarray(frozen["FULL_CAUSAL_CUT_SWAP"], dtype=np.float32)
        frozen_donor = np.asarray(frozen["DONOR_NATIVE"], dtype=np.float32)

    a00_tensor = engine.clean.detach().cpu().float()
    if tensor_sha256(a00_tensor) != group1["recipient_action_hash"] or not np.array_equal(a00_tensor.numpy(), frozen_a00):
        raise RuntimeError(f"A00_GROUP1_BIT_EXACT_MISMATCH:{donor['case_id']}")
    a10_tensor = engine.infer("clean", "fault", IMAGE).detach().cpu().float()
    a01_tensor = engine.infer("clean", "fault", PREFIX).detach().cpu().float()
    a00 = a00_tensor.numpy()
    a10 = a10_tensor.numpy()
    a01 = a01_tensor.numpy()
    a11 = frozen_a11

    smoke_checks: dict[str, bool] = {}
    if smoke:
        identity_image = engine.infer("clean", "clean", IMAGE).detach().cpu().float()
        identity_prefix = engine.infer("clean", "clean", PREFIX).detach().cpu().float()
        identity_full = engine.infer("clean", "clean", FULL).detach().cpu().float()
        reconstructed_a11 = engine.infer("clean", "fault", FULL).detach().cpu().float()
        smoke_checks = {
            "image_same_value_bit_exact": bool(torch.equal(identity_image, a00_tensor)),
            "prefix_same_value_bit_exact": bool(torch.equal(identity_prefix, a00_tensor)),
            "full_same_value_bit_exact": bool(torch.equal(identity_full, a00_tensor)),
            "A11_reproduces_group1_full_swap_bit_exact": bool(np.array_equal(reconstructed_a11.numpy(), a11)),
            "A00_reproduces_group1_recipient_bit_exact": True,
            "image_prefix_locus_sets_disjoint": not bool(set(IMAGE) & set(PREFIX)),
        }
        if not all(smoke_checks.values()):
            raise RuntimeError(f"IMAGEWAM_GROUP2_SMOKE_FAILED:{smoke_checks}")

    total = a11.astype(np.float64) - a00.astype(np.float64)
    image_delta = a10.astype(np.float64) - a00.astype(np.float64)
    prefix_delta = a01.astype(np.float64) - a00.astype(np.float64)
    phi_image = 0.5 * (image_delta + (a11.astype(np.float64) - a01.astype(np.float64)))
    phi_prefix = 0.5 * (prefix_delta + (a11.astype(np.float64) - a10.astype(np.float64)))
    interaction = a11.astype(np.float64) - a10.astype(np.float64) - a01.astype(np.float64) + a00.astype(np.float64)
    object_to_goal = np.asarray(phase["goal_position_m"], dtype=np.float64) - np.asarray(phase["object_position_m"], dtype=np.float64)
    tensors = {
        "A00": a00_tensor,
        "A10": a10_tensor,
        "A01": a01_tensor,
        "A11": torch.from_numpy(a11),
    }
    drive_values = {name: phase_drives(runner, action, object_to_goal) for name, action in tensors.items()}
    donor_residual = a11.astype(np.float64) - frozen_donor.astype(np.float64)
    donor_effect = frozen_donor.astype(np.float64) - a00.astype(np.float64)
    actions_path = target / "actions.npz"
    save_npz(
        actions_path,
        A00=a00,
        A10=a10,
        A01=a01,
        A11=a11,
        DONOR_NATIVE=frozen_donor,
        phi_image=phi_image,
        phi_prefix=phi_prefix,
        interaction=interaction,
        total=total,
    )
    result = {
        "status": "COMPLETED",
        "stage": "GROUP2C_IMAGEWAM_IMAGE_PREFIX",
        "case_id": donor["case_id"],
        "model": "imagewam",
        "task_id": donor["task_id"],
        "source_state_id": donor["source_state_id"],
        "base_state_id": donor["base_state_id"],
        "factor": donor["factor"],
        "phase": donor["phase"],
        "control_condition": donor["relation_control"],
        "signed_dose": donor["signed_dose"],
        "primary_dose": donor["primary_dose"],
        "route_interpretation": "IMAGE_KV_X_NON_IMAGE_PREFIX_KV_COMPONENT_DECOMPOSITION",
        "current_future_interpretation_claimed": False,
        "base_authorization_sha256": base_authorization_hash,
        "concurrency_authorization_sha256": concurrency_authorization_hash,
        "runner_sha256": runner_hash,
        "checkpoint_hash": checkpoint_hash,
        "hardware_id": torch.cuda.get_device_name(0),
        "hardware_class": classify_hardware_name(torch.cuda.get_device_name(0)),
        "seed": phase["policy_seed"],
        "recipient_state_hash": phase["recipient_state_hash"],
        "donor_state_hash": donor["donor_state_hash"],
        "image_loci": IMAGE,
        "prefix_loci": PREFIX,
        "A00_source": "GROUP2_RECOMPUTED_AND_GROUP1_BIT_EXACT_VALIDATED",
        "A11_source": "GROUP1_FROZEN_FULL_CAUSAL_CUT_SWAP_REUSED",
        "action_hashes": {name: tensor_sha256(action) for name, action in tensors.items()},
        "group1_expected_hashes": {
            "A00": group1["recipient_action_hash"],
            "A11": group1["all_intervened_action_hashes"]["FULL_CAUSAL_CUT_SWAP"],
            "DONOR_NATIVE": group1["donor_action_hash"],
        },
        "vector_routing": {
            "A10_image_minus_A00": vector_metrics(image_delta, total),
            "A01_prefix_minus_A00": vector_metrics(prefix_delta, total),
            "phi_image": vector_metrics(phi_image, total),
            "phi_prefix": vector_metrics(phi_prefix, total),
            "interaction": vector_metrics(interaction, total),
            "total_l2": float(np.linalg.norm(total)),
        },
        "scalar_routing": scalar_decomposition(drive_values),
        "donor_action_closure": {
            "l2": float(np.linalg.norm(donor_residual)),
            "relative_l2": float(np.linalg.norm(donor_residual) / (np.linalg.norm(donor_effect) + EPS)),
            "bit_exact": bool(np.array_equal(a11, frozen_donor)),
        },
        "actions_path": str(actions_path),
        "actions_sha256": sha256(actions_path),
        "group1_result_path": str(group1_path),
        "group1_result_sha256": sha256(group1_path),
        "smoke_checks": smoke_checks,
        "model_forward_calls_new": 7 if smoke else 4,
        "simulator_calls": 0,
        "runtime_seconds": time.time() - started,
    }
    write_json(result_path, result)
    del engine
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--cpu-threads", type=int, default=16)
    args = parser.parse_args()
    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(1)
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    if not FROZEN_WAN_VAE.is_file():
        raise FileNotFoundError(FROZEN_WAN_VAE)

    base_authorization_hash, concurrency_authorization_hash = verify_authorizations()
    runner_path = Path(__file__).resolve()
    runner_hash = sha256(runner_path)
    os.chdir(FROZEN_LAUNCH_ROOT)
    phases = {row["base_state_id"]: row for row in read_jsonl(G1_RAW / "phase_registry.jsonl")}
    donors = [
        row for row in read_jsonl(G1_RAW / "donor_bank/imagewam/counterfactual_qc.jsonl")
        if row["donor_valid"] and row["primary_dose"]
    ]
    donors.sort(key=lambda row: row["case_id"])
    if args.smoke:
        donors = donors[:1]
    elif args.max_cases is not None:
        donors = donors[:args.max_cases]

    runner = make_capture("imagewam", 0)
    runner.model.eval()
    hardware_id = torch.cuda.get_device_name(0)
    if classify_hardware_name(hardware_id) != "RTX4090":
        raise RuntimeError(f"IMAGEWAM_HARDWARE_AUTHORITY_MISMATCH:{hardware_id}")
    checkpoint_hash = model_weight_hash(runner.model)
    group0 = json.loads((Path(_release_path('@DATA@/wam_control_state_v3/technical_validation/imagewam/result.json'))).read_text(encoding="utf-8"))
    if checkpoint_hash != group0["checkpoint_hash"]:
        raise RuntimeError("IMAGEWAM_CHECKPOINT_HASH_MISMATCH")

    results = []
    for index, donor in enumerate(donors, 1):
        result = execute_case(
            runner,
            donor,
            phases[donor["base_state_id"]],
            checkpoint_hash,
            base_authorization_hash,
            concurrency_authorization_hash,
            runner_hash,
            args.smoke,
        )
        results.append(result)
        print(json.dumps({
            "model": "imagewam",
            "stage": "GROUP2C",
            "completed": index,
            "total": len(donors),
            "case_id": result["case_id"],
            "runtime_seconds": result["runtime_seconds"],
        }), flush=True)

    final_hash = model_weight_hash(runner.model)
    summary = {
        "status": "PASS" if final_hash == checkpoint_hash else "TECHNICAL_STOP",
        "stage": "GROUP2C_IMAGEWAM_IMAGE_PREFIX_SMOKE" if args.smoke else "GROUP2C_IMAGEWAM_IMAGE_PREFIX",
        "completed_cases": len(results),
        "eligible_cases": len(donors),
        "base_authorization_sha256": base_authorization_hash,
        "concurrency_authorization_sha256": concurrency_authorization_hash,
        "runner_sha256": runner_hash,
        "checkpoint_hash": checkpoint_hash,
        "final_weight_hash": final_hash,
        "model_weight_hash_unchanged": final_hash == checkpoint_hash,
        "hardware_id": hardware_id,
        "hardware_class": classify_hardware_name(hardware_id),
        "torch_num_threads": torch.get_num_threads(),
        "image_loci": IMAGE,
        "prefix_loci": PREFIX,
        "current_future_interpretation_claimed": False,
        "output_namespace": str(OUT),
    }
    write_json(OUT / ("technical_smoke/summary.json" if args.smoke else "run_summary.json"), summary)
    print(json.dumps(summary, indent=2), flush=True)
    if summary["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

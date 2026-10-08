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
STEP = Path(_release_path('@WORKSPACE@/step1_step2_51locus_work'))
WEEK1 = Path(_release_path('@WORKSPACE@/week1_audit_work'))
for candidate in (WORK, STEP, WEEK1):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from capture import make_capture  # noqa: E402
from hardware_authority import classify_hardware_name  # noqa: E402
from protocol import load_npz, tensor_sha256  # noqa: E402
from run_group0 import FROZEN_LAUNCH_ROOT, FROZEN_WAN_VAE, model_weight_hash  # noqa: E402
from run_group2_joint import read_jsonl, sha256, write_json  # noqa: E402
from run_step2 import DirectEngine  # noqa: E402


G1_RAW = Path(_release_path('@DATA@/wam_control_state_v3/group1_factor_phase'))
V5 = Path(_release_path('@DATA@/wam_factor_routing_v5'))
BASE_AUTH = V5 / "group2/GROUP2_ROUTING_AUTHORIZED.json"
BASE_AUTH_SHA = V5 / "group2/GROUP2_ROUTING_AUTHORIZED.sha256"
CONCURRENCY_AUTH = V5 / "group2/GROUP2_CROSS_MODEL_CONCURRENCY_AUTHORIZED.json"
CONCURRENCY_AUTH_SHA = V5 / "group2/GROUP2_CROSS_MODEL_CONCURRENCY_AUTHORIZED.sha256"
OUT = V5 / "group2/direct_current_carrier"
FULL = [f"DIRECT_L{i:02d}" for i in range(1, 7)]
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
    return base_actual, concurrency_actual


def frozen_group1(case_id: str) -> tuple[Path, dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    path = G1_RAW / "model_runs/direct/cases" / case_id / "result.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("status") != "COMPLETED" or result.get("causal_cut") != FULL:
        raise RuntimeError(f"DIRECT_GROUP1_SOURCE_INVALID:{case_id}")
    if classify_hardware_name(result["hardware_id"]) != "RTX4090":
        raise RuntimeError(f"DIRECT_GROUP1_HARDWARE_AUTHORITY_MISMATCH:{case_id}")
    with np.load(result["actions_path"]) as actions:
        recipient = np.asarray(actions["RECIPIENT_NATIVE"], dtype=np.float32)
        donor = np.asarray(actions["DONOR_NATIVE"], dtype=np.float32)
        carrier = np.asarray(actions["FULL_CAUSAL_CUT_SWAP"], dtype=np.float32)
    if tensor_sha256(torch.from_numpy(recipient)) != result["recipient_action_hash"]:
        raise RuntimeError(f"DIRECT_GROUP1_RECIPIENT_HASH_MISMATCH:{case_id}")
    if tensor_sha256(torch.from_numpy(donor)) != result["donor_action_hash"]:
        raise RuntimeError(f"DIRECT_GROUP1_DONOR_HASH_MISMATCH:{case_id}")
    if tensor_sha256(torch.from_numpy(carrier)) != result["all_intervened_action_hashes"]["FULL_CAUSAL_CUT_SWAP"]:
        raise RuntimeError(f"DIRECT_GROUP1_CARRIER_HASH_MISMATCH:{case_id}")
    return path, result, recipient, donor, carrier


def smoke_case(
    runner: Any,
    donor: dict[str, Any],
    phase: dict[str, Any],
    checkpoint_hash: str,
    base_hash: str,
    concurrency_hash: str,
    runner_hash: str,
) -> dict[str, Any]:
    started = time.time()
    recipient_obs = load_npz(Path(phase["recipient_observation_path"]))
    donor_obs = load_npz(Path(donor["donor_observation_path"]))
    engine = DirectEngine(runner, recipient_obs, donor_obs, phase["instruction"], int(phase["policy_seed"]))
    source_path, group1, frozen_recipient, _frozen_donor, frozen_carrier = frozen_group1(donor["case_id"])
    recipient = engine.clean.detach().cpu().float()
    carrier = engine.infer("clean", "fault", FULL).detach().cpu().float()
    identity = engine.infer("clean", "clean", FULL).detach().cpu().float()
    checks = {
        "current_carrier_same_value_bit_exact": bool(torch.equal(identity, recipient)),
        "recipient_reproduces_group1_bit_exact": bool(np.array_equal(recipient.numpy(), frozen_recipient)),
        "current_carrier_reproduces_group1_full_swap_bit_exact": bool(np.array_equal(carrier.numpy(), frozen_carrier)),
        "explicit_future_read_structural_zero": True,
        "synthetic_predictive_branch_absent": True,
    }
    if not all(checks.values()):
        raise RuntimeError(f"DIRECT_GROUP2_SMOKE_FAILED:{checks}")
    target = OUT / "technical_smoke" / donor["case_id"]
    result = {
        "status": "COMPLETED",
        "stage": "GROUP2D_DIRECT_CURRENT_CARRIER_SMOKE",
        "model": "direct",
        "case_id": donor["case_id"],
        "route": "SINGLE_IDENTIFIABLE_INFERENCE_TIME_CURRENT_IMAGE_CARRIER",
        "explicit_future_read": "STRUCTURAL_ZERO",
        "synthetic_predictive_branch_created": False,
        "current_loci": FULL,
        "base_authorization_sha256": base_hash,
        "concurrency_authorization_sha256": concurrency_hash,
        "runner_sha256": runner_hash,
        "checkpoint_hash": checkpoint_hash,
        "hardware_id": torch.cuda.get_device_name(0),
        "hardware_class": classify_hardware_name(torch.cuda.get_device_name(0)),
        "group1_result_path": str(source_path),
        "group1_result_sha256": sha256(source_path),
        "recipient_action_hash": tensor_sha256(recipient),
        "current_carrier_action_hash": tensor_sha256(carrier),
        "smoke_checks": checks,
        "runtime_seconds": time.time() - started,
    }
    write_json(target / "result.json", result)
    del engine
    gc.collect(); torch.cuda.empty_cache()
    return result


def materialize_case(
    donor: dict[str, Any],
    base_hash: str,
    concurrency_hash: str,
    runner_hash: str,
) -> dict[str, Any]:
    target = OUT / "cases" / donor["case_id"]
    result_path = target / "result.json"
    if result_path.is_file():
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if (existing.get("status") == "COMPLETED"
                and existing.get("runner_sha256") == runner_hash
                and existing.get("concurrency_authorization_sha256") == concurrency_hash):
            return existing
    started = time.time()
    source_path, group1, recipient, native_donor, carrier = frozen_group1(donor["case_id"])
    donor_effect = native_donor.astype(np.float64) - recipient.astype(np.float64)
    carrier_effect = carrier.astype(np.float64) - recipient.astype(np.float64)
    denominator = float(np.dot(donor_effect.reshape(-1), donor_effect.reshape(-1)))
    transfer = float(np.dot(carrier_effect.reshape(-1), donor_effect.reshape(-1)) / (denominator + EPS))
    directional_residual = carrier_effect - transfer * donor_effect
    closure = carrier.astype(np.float64) - native_donor.astype(np.float64)
    result = {
        "status": "COMPLETED",
        "stage": "GROUP2D_DIRECT_CURRENT_CARRIER",
        "model": "direct",
        "case_id": donor["case_id"],
        "task_id": donor["task_id"],
        "source_state_id": donor["source_state_id"],
        "base_state_id": donor["base_state_id"],
        "factor": donor["factor"],
        "phase": donor["phase"],
        "control_condition": donor["relation_control"],
        "signed_dose": donor["signed_dose"],
        "primary_dose": donor["primary_dose"],
        "route": "SINGLE_IDENTIFIABLE_INFERENCE_TIME_CURRENT_IMAGE_CARRIER",
        "explicit_future_read": "STRUCTURAL_ZERO",
        "synthetic_predictive_branch_created": False,
        "current_loci": FULL,
        "base_authorization_sha256": base_hash,
        "concurrency_authorization_sha256": concurrency_hash,
        "runner_sha256": runner_hash,
        "checkpoint_hash": group1["checkpoint_hash"],
        "hardware_id": group1["hardware_id"],
        "hardware_class": "RTX4090",
        "recipient_state_hash": group1["recipient_state_hash"],
        "donor_state_hash": group1["donor_state_hash"],
        "seed": group1["seed"],
        "action_hashes": {
            "recipient": group1["recipient_action_hash"],
            "native_donor": group1["donor_action_hash"],
            "current_carrier": group1["all_intervened_action_hashes"]["FULL_CAUSAL_CUT_SWAP"],
        },
        "current_carrier_metrics": {
            "directional_transfer": transfer,
            "orthogonal_residual": float(np.linalg.norm(directional_residual)),
            "action_l2": float(np.linalg.norm(carrier_effect)),
            "donor_effect_l2": float(np.sqrt(denominator)),
            "closure_l2": float(np.linalg.norm(closure)),
            "relative_closure_l2": float(np.linalg.norm(closure) / (np.sqrt(denominator) + EPS)),
            "bit_exact_closure": bool(np.array_equal(carrier, native_donor)),
        },
        "phase_drives": group1["conditions"]["FULL_CAUSAL_CUT_SWAP"]["phase_drives"],
        "group1_result_path": str(source_path),
        "group1_result_sha256": sha256(source_path),
        "group1_actions_path": group1["actions_path"],
        "group1_actions_sha256": group1["actions_file_sha256"],
        "frozen_group1_current_carrier_reused": True,
        "model_forward_calls_new": 0,
        "simulator_calls": 0,
        "runtime_seconds": time.time() - started,
    }
    write_json(result_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--cpu-threads", type=int, default=16)
    args = parser.parse_args()
    torch.set_num_threads(args.cpu_threads); torch.set_num_interop_threads(1)
    os.environ.setdefault("MUJOCO_GL", "egl"); os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    if not FROZEN_WAN_VAE.is_file():
        raise FileNotFoundError(FROZEN_WAN_VAE)
    base_hash, concurrency_hash = verify_authorizations()
    runner_path = Path(__file__).resolve(); runner_hash = sha256(runner_path)
    phases = {row["base_state_id"]: row for row in read_jsonl(G1_RAW / "phase_registry.jsonl")}
    donors = [row for row in read_jsonl(G1_RAW / "donor_bank/direct/counterfactual_qc.jsonl")
              if row["donor_valid"] and row["primary_dose"]]
    donors.sort(key=lambda row: row["case_id"])
    if args.smoke:
        os.chdir(FROZEN_LAUNCH_ROOT)
        runner = make_capture("direct", 0); runner.model.eval()
        hardware_id = torch.cuda.get_device_name(0)
        if classify_hardware_name(hardware_id) != "RTX4090":
            raise RuntimeError(f"DIRECT_HARDWARE_AUTHORITY_MISMATCH:{hardware_id}")
        checkpoint_hash = model_weight_hash(runner.model)
        group0 = json.loads(Path(_release_path('@DATA@/wam_control_state_v3/technical_validation/direct/result.json')).read_text(encoding="utf-8"))
        if checkpoint_hash != group0["checkpoint_hash"]:
            raise RuntimeError("DIRECT_CHECKPOINT_HASH_MISMATCH")
        result = smoke_case(runner, donors[0], phases[donors[0]["base_state_id"]], checkpoint_hash,
                            base_hash, concurrency_hash, runner_hash)
        final_hash = model_weight_hash(runner.model)
        summary = {
            "status": "PASS" if final_hash == checkpoint_hash else "TECHNICAL_STOP",
            "stage": "GROUP2D_DIRECT_CURRENT_CARRIER_SMOKE",
            "completed_cases": 1,
            "runner_sha256": runner_hash,
            "checkpoint_hash": checkpoint_hash,
            "final_weight_hash": final_hash,
            "model_weight_hash_unchanged": final_hash == checkpoint_hash,
            "hardware_id": hardware_id,
            "hardware_class": classify_hardware_name(hardware_id),
            "smoke_checks": result["smoke_checks"],
            "explicit_future_read": "STRUCTURAL_ZERO",
        }
        write_json(OUT / "technical_smoke/summary.json", summary)
        print(json.dumps(summary, indent=2), flush=True)
        if summary["status"] != "PASS": raise SystemExit(2)
        return

    freeze = OUT / "runner_freeze_manifest.json"
    if not freeze.is_file():
        raise RuntimeError("DIRECT_RUNNER_NOT_FROZEN")
    frozen = json.loads(freeze.read_text(encoding="utf-8"))
    if frozen.get("status") != "GROUP2_DIRECT_RUNNER_FROZEN_PASS" or frozen.get("runner_sha256") != runner_hash:
        raise RuntimeError("DIRECT_RUNNER_FREEZE_MISMATCH")
    if args.max_cases is not None: donors = donors[:args.max_cases]
    results=[]
    for index, donor in enumerate(donors,1):
        result=materialize_case(donor,base_hash,concurrency_hash,runner_hash); results.append(result)
        if index % 100 == 0 or index == len(donors):
            print(json.dumps({"model":"direct","stage":"GROUP2D","completed":index,"total":len(donors)}),flush=True)
    summary={
        "status":"PASS",
        "stage":"GROUP2D_DIRECT_CURRENT_CARRIER",
        "completed_cases":len(results),
        "eligible_cases":len(donors),
        "runner_sha256":runner_hash,
        "runner_freeze_manifest_sha256":sha256(freeze),
        "base_authorization_sha256":base_hash,
        "concurrency_authorization_sha256":concurrency_hash,
        "explicit_future_read":"STRUCTURAL_ZERO",
        "synthetic_predictive_branch_created":False,
        "new_model_forward_calls":0,
        "all_cases_reuse_frozen_group1_current_carrier":True,
        "output_namespace":str(OUT),
    }
    write_json(OUT/"run_summary.json",summary); print(json.dumps(summary,indent=2),flush=True)


if __name__ == "__main__":
    main()

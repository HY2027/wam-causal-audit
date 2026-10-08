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
from run_group0 import FROZEN_LAUNCH_ROOT, model_weight_hash  # noqa: E402
from run_group1_model import phase_drives  # noqa: E402
from run_group2_joint import read_jsonl, save_npz, sha256, write_json  # noqa: E402
from run_group2_idm import action_metrics  # noqa: E402
from run_step2 import IDMEngine  # noqa: E402


ROOT = Path(_release_path('@DATA@/wam_factor_routing_v5'))
G4A = ROOT / "group4a_heldout_factor_execution"
OUT = ROOT / "group4b_heldout_routing_interim"
AUTH = OUT / "GROUP4B_IDM_APPEND_AUTHORIZED.json"
AUTH_SHA = OUT / "GROUP4B_IDM_APPEND_AUTHORIZED.sha256"
GATE = OUT / "technical_validation/idm/idm_dual_blackwell_smoke_gate.json"
PHASES = G4A / "phase_registry.jsonl"
DONORS = G4A / "donor_bank/counterfactual_qc.jsonl"
REGISTRY = ROOT / "group4/group4_heldout_state_registry.csv"
REGISTRY_SHA = "42f0e5c52f270727db4febe3c61f4e01fd392609cec71846f718cbb6c01b9fbf"
GROUP0 = Path(_release_path('@DATA@/wam_control_state_v3/technical_validation/idm/result.json'))
FUTURE = ["IDM_L14"]
NEGATIVE = ["IDM_L01"]
EXPECTED = 544


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_authorization(runner_hash: str) -> str:
    expected = AUTH_SHA.read_text(encoding="utf-8").split()[0]
    actual = file_hash(AUTH)
    value = json.loads(AUTH.read_text(encoding="utf-8"))
    if expected != actual or value.get("status") != "GROUP4B_IDM_APPEND_AUTHORIZED":
        raise RuntimeError("GROUP4B_IDM_AUTHORIZATION_MISMATCH")
    if value.get("runner_sha256") != runner_hash or value.get("heldout_registry_sha256") != REGISTRY_SHA:
        raise RuntimeError("GROUP4B_IDM_FROZEN_INPUT_MISMATCH")
    if value.get("route_statement") != "FUTURE_MEDIATED_ROUTE" or value.get("future_only_exclusivity_claimed"):
        raise RuntimeError("GROUP4B_IDM_ROUTE_INTERPRETATION_MISMATCH")
    return actual


def source_case(case_id: str) -> tuple[Path, dict[str, Any], np.ndarray, np.ndarray]:
    path = G4A / "model_runs/idm/cases" / case_id / "result.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        value.get("status") != "COMPLETED"
        or value.get("hardware_class") != "BLACKWELL"
        or value.get("heldout_registry_sha256") != REGISTRY_SHA
        or value.get("routing_intervention_executed") is not False
    ):
        raise RuntimeError(f"GROUP4B_IDM_GROUP4A_SOURCE_INVALID:{case_id}")
    action_path = Path(value["actions_path"])
    if file_hash(action_path) != value["actions_file_sha256"]:
        raise RuntimeError(f"GROUP4B_IDM_GROUP4A_ACTION_HASH:{case_id}")
    with np.load(action_path, allow_pickle=False) as archive:
        recipient = np.asarray(archive["RECIPIENT_NATIVE"], dtype=np.float32)
        donor = np.asarray(archive["DONOR_NATIVE"], dtype=np.float32)
    return path, value, recipient, donor


def execute(
    runner: Any,
    donor: dict[str, Any],
    phase: dict[str, Any],
    checkpoint_hash: str,
    authorization_hash: str,
    runner_hash: str,
    smoke_label: str | None,
) -> dict[str, Any]:
    smoke = smoke_label is not None
    case_root = OUT / (
        f"technical_validation/idm/smoke/{smoke_label}/{donor['case_id']}"
        if smoke
        else f"model_runs/idm/cases/{donor['case_id']}"
    )
    result_path = case_root / "result.json"
    if not smoke and result_path.is_file():
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if existing.get("status") == "COMPLETED" and existing.get("runner_sha256") == runner_hash:
            return existing
    if donor.get("donor_valid") is not True or donor.get("primary_dose") is not True:
        raise RuntimeError("GROUP4B_IDM_INVALID_DONOR")
    source_path, source, frozen_recipient, frozen_donor = source_case(donor["case_id"])
    recipient_obs = load_npz(Path(phase["recipient_observation_path"]))
    donor_obs = load_npz(Path(donor["donor_observation_path"]))
    started = time.time()
    engine = IDMEngine(runner, recipient_obs, donor_obs, phase["instruction"], int(source["seed"]))
    recipient = engine.clean.detach().cpu().float()
    future_intervention = engine.fault.detach().cpu().float()
    full = engine.infer("clean", "fault", FUTURE).detach().cpu().float()
    negative = engine.infer("clean", "fault", NEGATIVE).detach().cpu().float()
    if not np.array_equal(recipient.numpy(), frozen_recipient):
        raise RuntimeError(f"GROUP4B_IDM_A00_GROUP4A_BIT_EXACT:{donor['case_id']}")
    smoke_checks: dict[str, bool] = {}
    if smoke:
        future_identity = engine.infer("clean", "clean", FUTURE).detach().cpu().float()
        negative_identity = engine.infer("clean", "clean", NEGATIVE).detach().cpu().float()
        full_repeat = engine.infer("clean", "fault", FUTURE).detach().cpu().float()
        smoke_checks = {
            "recipient_reproduces_group4a_bit_exact": True,
            "future_same_value_identity_bit_exact": bool(torch.equal(future_identity, recipient)),
            "negative_same_value_identity_bit_exact": bool(torch.equal(negative_identity, recipient)),
            "future_intervention_matches_full_carrier_bit_exact": bool(torch.equal(future_intervention, full)),
            "full_carrier_repeat_bit_exact": bool(torch.equal(full_repeat, full)),
            "all_actions_finite": bool(all(torch.isfinite(x).all().item() for x in (recipient, future_intervention, full, negative))),
            "decode_excluded": True,
            "context_route_not_artificially_separated": True,
            "simulator_calls_zero": True,
        }
        if not all(smoke_checks.values()):
            raise RuntimeError(f"GROUP4B_IDM_SMOKE_FAILED:{smoke_checks}")
    recipient_future_hash = engine.locus_hash("clean", FUTURE[0])
    donor_future_hash = engine.locus_hash("fault", FUTURE[0])
    case_root.mkdir(parents=True, exist_ok=True)
    actions_path = case_root / "actions.npz"
    save_npz(
        actions_path,
        RECIPIENT_NATIVE=recipient.numpy(),
        DONOR_NATIVE=frozen_donor,
        DONOR_FUTURE_INTERVENTION=future_intervention.numpy(),
        FULL_FUTURE_MEDIATED_CARRIER=full.numpy(),
        NEGATIVE_CONTROL=negative.numpy(),
    )
    object_to_goal = np.asarray(phase["goal_position_m"], dtype=np.float64) - np.asarray(phase["object_position_m"], dtype=np.float64)
    result = {
        "status": "COMPLETED",
        "stage": "GROUP4B_IDM_APPEND_SMOKE" if smoke else "GROUP4B_IDM_APPEND_HELDOUT_ROUTING",
        "model": "idm",
        "case_id": donor["case_id"],
        "task_id": int(donor["task_id"]),
        "heldout_base_state_id": phase["heldout_base_state_id"],
        "base_state_id": donor["base_state_id"],
        "factor": donor["factor"],
        "phase": donor["phase"],
        "control_condition": donor["relation_control"],
        "signed_dose": float(donor["signed_dose"]),
        "primary_dose": True,
        "route_statement": "FUTURE_MEDIATED_ROUTE",
        "future_only_exclusivity_claimed": False,
        "context_route_status": "IDM_CONTEXT_ROUTE_NOT_SEPARABLE",
        "decode_transfer_used_for_inference": False,
        "future_carrier_loci": FUTURE,
        "negative_control_locus": NEGATIVE[0],
        "recipient_future_latent_hash": recipient_future_hash,
        "donor_future_latent_hash": donor_future_hash,
        "routing_metrics": {
            "donor_future_intervention": action_metrics(future_intervention.numpy(), recipient.numpy(), frozen_donor),
            "full_future_mediated_carrier": action_metrics(full.numpy(), recipient.numpy(), frozen_donor),
            "negative_control": action_metrics(negative.numpy(), recipient.numpy(), frozen_donor),
        },
        "phase_drives": {
            "RECIPIENT": phase_drives(runner, recipient, object_to_goal),
            "FULL_FUTURE_MEDIATED_CARRIER": phase_drives(runner, full, object_to_goal),
            "NEGATIVE_CONTROL": phase_drives(runner, negative, object_to_goal),
        },
        "action_hashes": {
            "RECIPIENT": tensor_sha256(recipient),
            "DONOR_FUTURE_INTERVENTION": tensor_sha256(future_intervention),
            "FULL_FUTURE_MEDIATED_CARRIER": tensor_sha256(full),
            "NEGATIVE_CONTROL": tensor_sha256(negative),
        },
        "native_donor_action_hash": source["donor_action_hash"],
        "authorization_sha256": authorization_hash,
        "runner_sha256": runner_hash,
        "checkpoint_hash": checkpoint_hash,
        "hardware_id": torch.cuda.get_device_name(0),
        "hardware_class": classify_hardware_name(torch.cuda.get_device_name(0)),
        "heldout_registry_sha256": REGISTRY_SHA,
        "actions_path": str(actions_path),
        "actions_sha256": file_hash(actions_path),
        "group4a_result_path": str(source_path),
        "group4a_result_sha256": file_hash(source_path),
        "smoke_label": smoke_label,
        "smoke_checks": smoke_checks,
        "simulator_calls": 0,
        "closed_loop_rollout_executed": False,
        "four_model_conclusion_issued": False,
        "runtime_seconds": time.time() - started,
    }
    write_json(result_path, result)
    del engine
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument("--smoke-label", choices=("gpu4", "gpu5"))
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=16)
    args = parser.parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu_id))
    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(1)
    runner_hash = file_hash(Path(__file__).resolve())
    authorization_hash = verify_authorization(runner_hash)
    phase_rows = {row["base_state_id"]: row for row in read_jsonl(PHASES)}
    donors = sorted(
        [row for row in read_jsonl(DONORS) if row.get("donor_valid") is True and row.get("primary_dose") is True],
        key=lambda row: row["case_id"],
    )
    if len(donors) != EXPECTED:
        raise RuntimeError(f"GROUP4B_IDM_CASE_CARDINALITY:{len(donors)}")
    smoke = args.smoke_label is not None
    if smoke:
        donors = donors[:1]
    else:
        gate = json.loads(GATE.read_text(encoding="utf-8"))
        if gate.get("status") != "GROUP4B_IDM_DUAL_BLACKWELL_SMOKE_PASS" or gate.get("runner_sha256") != runner_hash:
            raise RuntimeError("GROUP4B_IDM_DUAL_GPU_GATE_MISMATCH")
        if args.num_shards != 2 or args.shard_index not in (0, 1):
            raise RuntimeError("GROUP4B_IDM_REQUIRES_TWO_SHARDS")
        donors = donors[args.shard_index::2]
        if len(donors) != 272:
            raise RuntimeError("GROUP4B_IDM_SHARD_CARDINALITY")
    os.chdir(FROZEN_LAUNCH_ROOT)
    runner = make_capture("idm", 0)
    runner.model.eval()
    hardware_id = torch.cuda.get_device_name(0)
    if classify_hardware_name(hardware_id) != "BLACKWELL":
        raise RuntimeError("GROUP4B_IDM_HARDWARE_AUTHORITY")
    group0 = json.loads(GROUP0.read_text(encoding="utf-8"))
    checkpoint_hash = model_weight_hash(runner.model)
    if group0.get("status") != "PASS" or checkpoint_hash != group0["checkpoint_hash"]:
        raise RuntimeError("GROUP4B_IDM_CHECKPOINT_HASH")
    results = []
    for index, donor in enumerate(donors, 1):
        result = execute(runner, donor, phase_rows[donor["base_state_id"]], checkpoint_hash, authorization_hash, runner_hash, args.smoke_label)
        results.append(result)
        print(json.dumps({"model":"idm","stage":"GROUP4B_SMOKE" if smoke else "GROUP4B","shard_index":None if smoke else args.shard_index,"completed":index,"total":len(donors),"case_id":donor["case_id"],"runtime_seconds":result["runtime_seconds"]}), flush=True)
    final_hash = model_weight_hash(runner.model)
    summary = {
        "status": "PASS" if final_hash == checkpoint_hash else "TECHNICAL_STOP",
        "stage": "GROUP4B_IDM_APPEND_SMOKE" if smoke else "GROUP4B_IDM_APPEND_HELDOUT_ROUTING_SHARD",
        "smoke_label": args.smoke_label,
        "shard_index": None if smoke else args.shard_index,
        "completed_cases": len(results),
        "runner_sha256": runner_hash,
        "authorization_sha256": authorization_hash,
        "checkpoint_hash": checkpoint_hash,
        "final_weight_hash": final_hash,
        "model_weight_hash_unchanged": final_hash == checkpoint_hash,
        "hardware_id": hardware_id,
        "hardware_class": classify_hardware_name(hardware_id),
        "route_statement": "FUTURE_MEDIATED_ROUTE",
        "future_only_exclusivity_claimed": False,
        "four_model_conclusion_issued": False,
    }
    summary_path = OUT / (
        f"technical_validation/idm/smoke/{args.smoke_label}/run_summary.json"
        if smoke else f"model_runs/idm/shards/shard_{args.shard_index}/run_summary.json"
    )
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2), flush=True)
    if summary["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

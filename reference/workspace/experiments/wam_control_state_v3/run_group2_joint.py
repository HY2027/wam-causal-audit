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
from protocol import load_npz, tensor_sha256  # noqa: E402
from run_group0 import FROZEN_LAUNCH_ROOT, FROZEN_WAN_VAE, model_weight_hash  # noqa: E402
from run_group1_model import phase_drives  # noqa: E402
from run_step2 import JointEngine  # noqa: E402


G1_RAW = Path(_release_path('@DATA@/wam_control_state_v3/group1_factor_phase'))
V5 = Path(_release_path('@DATA@/wam_factor_routing_v5'))
AUTHORIZATION = V5 / "group2/GROUP2_ROUTING_AUTHORIZED.json"
AUTHORIZATION_SHA = V5 / "group2/GROUP2_ROUTING_AUTHORIZED.sha256"
OUT = V5 / "group2/joint_current_future"
CURRENT = [f"JOINT_L{i:02d}" for i in (1, 3, 5, 7, 9, 11)]
FUTURE = [f"JOINT_L{i:02d}" for i in (2, 4, 6, 8, 10, 12)]
FULL = [f"JOINT_L{i:02d}" for i in range(1, 13)]
EPS = 1e-12


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def verify_authorization() -> str:
    expected = AUTHORIZATION_SHA.read_text(encoding="utf-8").split()[0]
    actual = sha256(AUTHORIZATION)
    payload = json.loads(AUTHORIZATION.read_text(encoding="utf-8"))
    if expected != actual or payload.get("status") != "GROUP2_ROUTING_AUTHORIZED":
        raise RuntimeError("GROUP2_AUTHORIZATION_MISMATCH")
    if payload.get("execution_order", [None])[0] != "joint":
        raise RuntimeError("GROUP2_EXECUTION_ORDER_MISMATCH")
    return actual


def vector_metrics(delta: np.ndarray, total: np.ndarray) -> dict[str, float | None]:
    delta = np.asarray(delta, dtype=np.float64)
    total = np.asarray(total, dtype=np.float64)
    denominator = float(np.dot(total.reshape(-1), total.reshape(-1)))
    if denominator <= EPS:
        return {"directional_projection": None, "orthogonal_residual": float(np.linalg.norm(delta))}
    projection = float(np.dot(delta.reshape(-1), total.reshape(-1)) / denominator)
    residual = delta - projection * total
    return {"directional_projection": projection, "orthogonal_residual": float(np.linalg.norm(residual))}


def scalar_decomposition(values: dict[str, dict[str, float]]) -> dict[str, dict[str, float | None]]:
    output: dict[str, dict[str, float | None]] = {}
    metrics = sorted(values["A00"])
    for metric in metrics:
        y00, y10 = values["A00"][metric], values["A10"][metric]
        y01, y11 = values["A01"][metric], values["A11"][metric]
        phi_current = 0.5 * ((y10 - y00) + (y11 - y01))
        phi_future = 0.5 * ((y01 - y00) + (y11 - y10))
        interaction = y11 - y10 - y01 + y00
        total = y11 - y00
        output[metric] = {
            "A00": y00, "A10": y10, "A01": y01, "A11": y11,
            "phi_current": phi_current,
            "phi_future": phi_future,
            "interaction": interaction,
            "total": total,
            "signed_current_share": None if abs(total) <= EPS else phi_current / total,
            "signed_future_share": None if abs(total) <= EPS else phi_future / total,
        }
    return output


def result_dir(case_id: str, smoke: bool) -> Path:
    return OUT / ("technical_smoke" if smoke else "cases") / case_id


def execute_case(
    runner: Any,
    donor: dict[str, Any],
    phase: dict[str, Any],
    checkpoint_hash: str,
    authorization_hash: str,
    smoke: bool,
) -> dict[str, Any]:
    target = result_dir(donor["case_id"], smoke)
    result_path = target / "result.json"
    if not smoke and result_path.is_file():
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if existing.get("status") == "COMPLETED" and existing.get("authorization_sha256") == authorization_hash:
            return existing
    started = time.time()
    recipient_observation = load_npz(Path(phase["recipient_observation_path"]))
    donor_observation = load_npz(Path(donor["donor_observation_path"]))
    engine = JointEngine(runner, recipient_observation, donor_observation, phase["instruction"], int(phase["policy_seed"]))

    group1_path = G1_RAW / "model_runs/joint/cases" / donor["case_id"] / "result.json"
    group1 = json.loads(group1_path.read_text(encoding="utf-8"))
    with np.load(group1["actions_path"]) as frozen:
        frozen_a00 = np.asarray(frozen["RECIPIENT_NATIVE"], dtype=np.float32)
        frozen_a11 = np.asarray(frozen["FULL_CAUSAL_CUT_SWAP"], dtype=np.float32)
        frozen_donor = np.asarray(frozen["DONOR_NATIVE"], dtype=np.float32)
    a00_tensor = engine.clean.detach().cpu().float()
    if tensor_sha256(a00_tensor) != group1["recipient_action_hash"] or not np.array_equal(a00_tensor.numpy(), frozen_a00):
        raise RuntimeError(f"A00_GROUP1_BIT_EXACT_MISMATCH:{donor['case_id']}")
    a10_tensor = engine.infer("clean", "fault", CURRENT).detach().cpu().float()
    a01_tensor = engine.infer("clean", "fault", FUTURE).detach().cpu().float()
    a00 = a00_tensor.numpy(); a10 = a10_tensor.numpy(); a01 = a01_tensor.numpy(); a11 = frozen_a11

    smoke_checks: dict[str, bool] = {}
    if smoke:
        identity_current = engine.infer("clean", "clean", CURRENT).detach().cpu().float()
        identity_future = engine.infer("clean", "clean", FUTURE).detach().cpu().float()
        identity_full = engine.infer("clean", "clean", FULL).detach().cpu().float()
        reconstructed_a11 = engine.infer("clean", "fault", FULL).detach().cpu().float()
        smoke_checks = {
            "current_same_value_bit_exact": bool(torch.equal(identity_current, a00_tensor)),
            "future_same_value_bit_exact": bool(torch.equal(identity_future, a00_tensor)),
            "full_same_value_bit_exact": bool(torch.equal(identity_full, a00_tensor)),
            "A11_reproduces_group1_full_swap_bit_exact": bool(np.array_equal(reconstructed_a11.numpy(), a11)),
            "A00_reproduces_group1_recipient_bit_exact": True,
            "current_future_locus_sets_disjoint": not bool(set(CURRENT) & set(FUTURE)),
        }
        if not all(smoke_checks.values()):
            raise RuntimeError(f"JOINT_GROUP2_SMOKE_FAILED:{smoke_checks}")

    total = a11.astype(np.float64) - a00.astype(np.float64)
    current_delta = a10.astype(np.float64) - a00.astype(np.float64)
    future_delta = a01.astype(np.float64) - a00.astype(np.float64)
    phi_current = 0.5 * (current_delta + (a11.astype(np.float64) - a01.astype(np.float64)))
    phi_future = 0.5 * (future_delta + (a11.astype(np.float64) - a10.astype(np.float64)))
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
        A00=a00, A10=a10, A01=a01, A11=a11, DONOR_NATIVE=frozen_donor,
        phi_current=phi_current, phi_future=phi_future, interaction=interaction, total=total,
    )
    result = {
        "status": "COMPLETED",
        "stage": "GROUP2A_JOINT_CURRENT_FUTURE",
        "case_id": donor["case_id"],
        "model": "joint",
        "task_id": donor["task_id"],
        "source_state_id": donor["source_state_id"],
        "base_state_id": donor["base_state_id"],
        "factor": donor["factor"],
        "phase": donor["phase"],
        "control_condition": donor["relation_control"],
        "signed_dose": donor["signed_dose"],
        "primary_dose": donor["primary_dose"],
        "authorization_sha256": authorization_hash,
        "checkpoint_hash": checkpoint_hash,
        "hardware_id": torch.cuda.get_device_name(0),
        "seed": phase["policy_seed"],
        "recipient_state_hash": phase["recipient_state_hash"],
        "donor_state_hash": donor["donor_state_hash"],
        "current_loci": CURRENT,
        "future_loci": FUTURE,
        "A00_source": "GROUP2_RECOMPUTED_AND_GROUP1_BIT_EXACT_VALIDATED",
        "A11_source": "GROUP1_FROZEN_FULL_CAUSAL_CUT_SWAP_REUSED",
        "action_hashes": {name: tensor_sha256(action) for name, action in tensors.items()},
        "group1_expected_hashes": {
            "A00": group1["recipient_action_hash"],
            "A11": group1["all_intervened_action_hashes"]["FULL_CAUSAL_CUT_SWAP"],
            "DONOR_NATIVE": group1["donor_action_hash"],
        },
        "vector_routing": {
            "A10_minus_A00": vector_metrics(current_delta, total),
            "A01_minus_A00": vector_metrics(future_delta, total),
            "phi_current": vector_metrics(phi_current, total),
            "phi_future": vector_metrics(phi_future, total),
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
    gc.collect(); torch.cuda.empty_cache()
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
    authorization_hash = verify_authorization()
    os.chdir(FROZEN_LAUNCH_ROOT)
    phases = {row["base_state_id"]: row for row in read_jsonl(G1_RAW / "phase_registry.jsonl")}
    donors = [
        row for row in read_jsonl(G1_RAW / "donor_bank/joint/counterfactual_qc.jsonl")
        if row["donor_valid"] and row["primary_dose"]
    ]
    donors.sort(key=lambda row: row["case_id"])
    if args.smoke:
        donors = donors[:1]
    elif args.max_cases is not None:
        donors = donors[:args.max_cases]
    runner = make_capture("joint", 0); runner.model.eval()
    checkpoint_hash = model_weight_hash(runner.model)
    group0 = json.loads((Path(_release_path('@DATA@/wam_control_state_v3/technical_validation/joint/result.json'))).read_text(encoding="utf-8"))
    if checkpoint_hash != group0["checkpoint_hash"]:
        raise RuntimeError("JOINT_CHECKPOINT_HASH_MISMATCH")
    results = []
    for index, donor in enumerate(donors, 1):
        result = execute_case(runner, donor, phases[donor["base_state_id"]], checkpoint_hash, authorization_hash, args.smoke)
        results.append(result)
        print(json.dumps({
            "model": "joint", "stage": "GROUP2A", "completed": index, "total": len(donors),
            "case_id": result["case_id"], "runtime_seconds": result["runtime_seconds"],
        }), flush=True)
    final_hash = model_weight_hash(runner.model)
    summary = {
        "status": "PASS" if final_hash == checkpoint_hash else "TECHNICAL_STOP",
        "stage": "GROUP2A_JOINT_CURRENT_FUTURE_SMOKE" if args.smoke else "GROUP2A_JOINT_CURRENT_FUTURE",
        "completed_cases": len(results),
        "eligible_cases": len(donors),
        "authorization_sha256": authorization_hash,
        "checkpoint_hash": checkpoint_hash,
        "final_weight_hash": final_hash,
        "model_weight_hash_unchanged": final_hash == checkpoint_hash,
        "hardware_id": torch.cuda.get_device_name(0),
        "torch_num_threads": torch.get_num_threads(),
        "current_loci": CURRENT,
        "future_loci": FUTURE,
        "old_rgb_proprio_group2_started": False,
    }
    write_json(OUT / ("technical_smoke/summary.json" if args.smoke else "run_summary.json"), summary)
    print(json.dumps(summary, indent=2), flush=True)
    if summary["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

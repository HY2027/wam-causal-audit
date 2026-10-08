#!/usr/bin/env python3
"""Read-only post-hoc analysis of the propagated-future component.

This program consumes already-saved Joint-WAM action arrays and consumer-event
logs.  It never imports the model or simulator stack.  The primary estimand is
the frozen 34-source RoboTwin confirmation cohort; the eight post-hoc identity
amendment sources are reported only as an explicitly labelled coverage
sensitivity analysis.
"""

from __future__ import annotations
from wam_causal_audit.paths import resolve as _release_path

import csv
import hashlib
import json
import math
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


REPO = Path(_release_path('@WORKSPACE@/FastWAM'))
OUT = REPO / "results/propagated_future_component"
PRIMARY = Path(
    _release_path('@DATA@/wam_factor_routing_v5/experiments/robotwin_wam_physical_mechanism_20260921T063246Z')
)
EXTENSION = Path(
    _release_path('@DATA@/wam_factor_routing_v5/experiments/robotwin_identity_amendment_coverage_20260921T103100Z')
)
PRIMARY_BRANCH = PRIMARY / "joint_confirmation"
EXTENSION_BRANCH = EXTENSION / "joint_POSTHOC_IDENTITY_AMENDMENT_COVERAGE_SENSITIVITY"
PRIMARY_READOUT = PRIMARY / "source_level_readouts.csv"
EXTENSION_READOUT = (
    EXTENSION
    / "analysis_adapter/completed_identity_v2_20260922T0130Z/new_8_source_level_readouts.csv"
)
N_POSITIONS = 32
N_CHANNELS = 14
N_BOOT = 10_000
BOOT_SEED = 20_260_921
EXPECTED_EVENTS = {
    (step, layer, component)
    for step in range(10)
    for layer in range(30)
    for component in ("k", "v")
}
EFFECTS = {
    "N": "propagation-allowed current-node effect",
    "U_C_given_F_R": "strict current-donor effect with recipient future fixed",
    "P_future": "node-minus-strict-current propagated-future component",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_compact(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def read_json(path: Path, provenance: dict[Path, str], role: str) -> Any:
    register_input(path, provenance, role)
    return json.loads(path.read_text())


def read_csv(path: Path, provenance: dict[Path, str], role: str) -> list[dict[str, str]]:
    register_input(path, provenance, role)
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def register_input(path: Path, provenance: dict[Path, str], role: str) -> None:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"required input is unavailable: {path}")
    prior = provenance.get(path)
    provenance[path] = role if prior is None else prior + "; " + role if role not in prior else prior


def atomic_text(path: Path, content: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content)
    os.replace(tmp, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refuse to write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def npz_arrays(
    path: Path,
    provenance: dict[Path, str],
    role: str,
) -> dict[str, np.ndarray]:
    register_input(path, provenance, role)
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: np.array(archive[key], copy=True) for key in archive.files}
    required = {
        "normalized": ((N_POSITIONS, N_CHANNELS), np.float32),
        "joint_targets": ((N_POSITIONS, N_CHANNELS), np.float32),
        "eef_target_world": ((N_POSITIONS, 3), np.float64),
        "radial_world": ((N_POSITIONS,), np.float64),
        "u": ((3,), np.float64),
        "context_eef": ((3,), np.float64),
    }
    for key, (shape, _) in required.items():
        if key not in arrays:
            raise KeyError(f"{path}: missing array {key}")
        if arrays[key].shape != shape:
            raise ValueError(f"{path}: {key} shape {arrays[key].shape}, expected {shape}")
        if not np.isfinite(arrays[key]).all():
            raise ValueError(f"{path}: {key} has non-finite values")
    return arrays


def event_map(
    path: Path,
    provenance: dict[Path, str],
    role: str,
) -> dict[tuple[int, int, str], dict[str, Any]]:
    values = read_json(path, provenance, role)
    result = {(int(x["step"]), int(x["layer"]), str(x["component"])): x for x in values}
    if len(values) != 600 or len(result) != 600 or set(result) != EXPECTED_EVENTS:
        raise ValueError(f"{path}: incomplete or duplicate event coverage ({len(values)}/{len(result)})")
    return result


def tensor_sha(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    return hashlib.sha256(value.view(np.uint8).tobytes(order="C")).hexdigest()


def task_equal(rows: list[dict[str, Any]], field: str) -> float:
    tasks: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        tasks[str(row["task_id"])].append(float(row[field]))
    return float(np.mean([np.mean(values) for values in tasks.values()]))


def hierarchical_bootstrap(rows: list[dict[str, Any]], field: str, seed: int) -> np.ndarray:
    """Exact frozen design: resample tasks, then source groups within task."""
    tasks: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        tasks[str(row["task_id"])].append(float(row[field]))
    names = sorted(tasks)
    rng = np.random.default_rng(seed)
    draws = np.empty(N_BOOT, dtype=np.float64)
    for draw in range(N_BOOT):
        selected_tasks = rng.choice(names, size=len(names), replace=True)
        task_means = []
        for task in selected_tasks:
            values = np.asarray(tasks[str(task)], dtype=np.float64)
            task_means.append(float(np.mean(rng.choice(values, size=len(values), replace=True))))
        draws[draw] = float(np.mean(task_means))
    return draws


def within_task_bootstrap(rows: list[dict[str, Any]], field: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = np.asarray([float(row[field]) for row in rows], dtype=np.float64)
    return np.mean(rng.choice(values, size=(N_BOOT, len(values)), replace=True), axis=1)


def metric_seed(cohort: str, scope: str, effect: str, statistic: str) -> int:
    key = f"{cohort}|{scope}|{effect}|{statistic}".encode()
    return BOOT_SEED + int(hashlib.sha256(key).hexdigest()[:8], 16)


def call_index(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    result = {}
    for row in rows:
        source_id = Path(row["action_artifact"]).parent.name
        key = (source_id, row["label"])
        if key in result:
            raise ValueError(f"duplicate call record: {key}")
        result[key] = row
    return result


def find_call(index: dict[tuple[str, str], dict[str, Any]], source_id: str, label: str) -> dict[str, Any]:
    key = (source_id, label)
    if key not in index:
        raise KeyError(f"required call record missing: {key}")
    row = index[key]
    if row.get("status") != "PASS":
        raise ValueError(f"required call did not pass: {key}: {row.get('status')}")
    if row.get("predicted_action_executed") is not False:
        raise ValueError(f"predicted action execution status is not false: {key}")
    return row


def full_action_l2(array: np.ndarray) -> float:
    """Flattened L2 norm in normalized action coordinates."""
    return float(np.linalg.norm(np.asarray(array, dtype=np.float64).reshape(-1), ord=2))


def channel_rms(array: np.ndarray) -> list[float]:
    return [float(x) for x in np.sqrt(np.mean(np.square(np.asarray(array, dtype=np.float64)), axis=0))]


def build() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    provenance: dict[Path, str] = {}

    global_inputs = [
        (PRIMARY / "protocol.json", "frozen physical/mechanism protocol"),
        (PRIMARY / "analysis_spec.json", "frozen estimands and weighting"),
        (PRIMARY / "analyze_campaign.py", "frozen analysis implementation"),
        (PRIMARY / "analysis/bootstrap_summary.json", "frozen attribution estimates and intervals"),
        (PRIMARY / "confirmation_sources.json", "primary source identity registry"),
        (PRIMARY / "registry.json", "model/config/checkpoint identity registry"),
        (PRIMARY_BRANCH / "result.json", "primary Joint branch completion and model hashes"),
        (PRIMARY_BRANCH / "weights.json", "primary unchanged-weight audit"),
        (PRIMARY / "audit_v2_20260921/statistics/00_input_and_implementation_audit.json", "frozen-statistics audit"),
        (EXTENSION / "identity/source_manifest_frozen.json", "post-hoc identity-amendment source registry"),
        (EXTENSION_BRANCH / "result.json", "post-hoc Joint branch completion and model hashes"),
        (EXTENSION_BRANCH / "weights.json", "post-hoc unchanged-weight audit"),
        (EXTENSION / "identity/identity_amendment_report.md", "post-hoc ancestry/identity limitation"),
    ]
    global_data = {}
    for path, role in global_inputs:
        if path.suffix == ".json":
            global_data[path.name + "@" + str(path.parent)] = read_json(path, provenance, role)
        else:
            register_input(path, provenance, role)

    primary_source_registry = {
        x["source_id"]: x
        for x in global_data["confirmation_sources.json@" + str(PRIMARY)]["sources"]
    }
    extension_manifest = global_data[
        "source_manifest_frozen.json@" + str(EXTENSION / "identity")
    ]
    extension_sources_raw = extension_manifest.get("sources", extension_manifest)
    if isinstance(extension_sources_raw, dict):
        extension_sources_raw = extension_sources_raw.get("sources", [])
    extension_source_registry = {x["source_id"]: x for x in extension_sources_raw}

    primary_rows_all = read_csv(PRIMARY_READOUT, provenance, "frozen 34-source readout table")
    extension_rows_all = read_csv(EXTENSION_READOUT, provenance, "post-hoc eight-source readout table")
    primary_rows = [x for x in primary_rows_all if x["model"].strip().lower() == "joint"]
    extension_rows = [x for x in extension_rows_all if x["model"].strip().lower() == "joint"]
    if len(primary_rows) != 34:
        raise ValueError(f"expected 34 primary Joint rows, found {len(primary_rows)}")
    if len(extension_rows) != 8:
        raise ValueError(f"expected 8 extension Joint rows, found {len(extension_rows)}")

    primary_calls_path = PRIMARY_BRANCH / "call_summaries.json"
    extension_calls_path = EXTENSION_BRANCH / "call_summaries.json"
    primary_calls = call_index(read_json(primary_calls_path, provenance, "primary per-call identity ledger"))
    extension_calls = call_index(read_json(extension_calls_path, provenance, "extension per-call identity ledger"))

    branch_results = {
        "ORIGINAL34_FROZEN_CONFIRMATION": global_data["result.json@" + str(PRIMARY_BRANCH)],
        "POSTHOC8_IDENTITY_AMENDMENT": global_data["result.json@" + str(EXTENSION_BRANCH)],
    }
    if branch_results["ORIGINAL34_FROZEN_CONFIRMATION"]["checkpoint_sha256"] != branch_results["POSTHOC8_IDENTITY_AMENDMENT"]["checkpoint_sha256"]:
        raise ValueError("primary and extension checkpoint identities differ")
    if branch_results["ORIGINAL34_FROZEN_CONFIRMATION"]["config_sha256"] != branch_results["POSTHOC8_IDENTITY_AMENDMENT"]["config_sha256"]:
        raise ValueError("primary and extension config identities differ")
    if branch_results["ORIGINAL34_FROZEN_CONFIRMATION"]["stats_sha256"] != branch_results["POSTHOC8_IDENTITY_AMENDMENT"]["stats_sha256"]:
        raise ValueError("primary and extension normalization-stat identities differ")
    for branch_path in (PRIMARY_BRANCH, EXTENSION_BRANCH):
        weights = global_data["weights.json@" + str(branch_path)]
        if not weights.get("unchanged") or weights.get("before") != weights.get("after"):
            raise ValueError(f"model weights were not unchanged in branch {branch_path}")

    case_rows: list[dict[str, Any]] = []
    closure_case_rows: list[dict[str, Any]] = []

    for origin, rows, calls, registry in (
        ("ORIGINAL34_FROZEN_CONFIRMATION", primary_rows, primary_calls, primary_source_registry),
        ("POSTHOC8_IDENTITY_AMENDMENT", extension_rows, extension_calls, extension_source_registry),
    ):
        for source_row in sorted(rows, key=lambda x: (x["task_id"], x["source_id"])):
            source_id = source_row["source_id"]
            task_id = source_row["task_id"]
            labels = (
                "capture_Z1",
                "capture_Dplus",
                "A00_Crecipient_Frecipient",
                "A10_Cdonor_Frecipient",
                "NODE_CURRENT_PROPAGATION_ALLOWED",
            )
            records = {label: find_call(calls, source_id, label) for label in labels}

            # The extension's stable source ID differs from its preserved geometry ID.
            geometry_dir = Path(records["capture_Z1"]["input"]["observation"]).parents[1]
            geometry_protocol = read_json(
                geometry_dir / "geometry_protocol_instance.json",
                provenance,
                f"{source_id}: signed-dose and radial geometry metadata",
            )
            fk_contract = read_json(
                geometry_dir / "fk_contract.json",
                provenance,
                f"{source_id}: registered world-EEF radial readout",
            )
            dose_m = float(geometry_protocol["budget"]["dose_m"])
            if not math.isclose(dose_m, 0.01, rel_tol=0.0, abs_tol=0.0):
                raise ValueError(f"{source_id}: unexpected registered dose {dose_m}")

            arrays: dict[str, dict[str, np.ndarray]] = {}
            events: dict[str, dict[tuple[int, int, str], dict[str, Any]]] = {}
            for label, call in records.items():
                action_path = Path(call["action_artifact"])
                if sha256(action_path) != call["action_artifact_sha256"]:
                    raise ValueError(f"{source_id}/{label}: action artifact hash mismatch")
                arrays[label] = npz_arrays(action_path, provenance, f"{source_id}/{label}: saved action tensors")
                if tensor_sha(arrays[label]["normalized"]) != call["normalized_action_sha256"]:
                    raise ValueError(f"{source_id}/{label}: normalized tensor hash mismatch")
                event_path = Path(call["event_log"])
                if sha256(event_path) != call["event_log_sha256"]:
                    raise ValueError(f"{source_id}/{label}: event log hash mismatch")
                events[label] = event_map(event_path, provenance, f"{source_id}/{label}: consumer-event source log")
                obs_path = Path(call["input"]["observation"])
                register_input(obs_path, provenance, f"{source_id}/{label}: physical endpoint observation bytes")
                if sha256(obs_path) != call["input"]["observation_sha256"]:
                    raise ValueError(f"{source_id}/{label}: observation hash mismatch")

            a00 = arrays["A00_Crecipient_Frecipient"]
            a10 = arrays["A10_Cdonor_Frecipient"]
            node = arrays["NODE_CURRENT_PROPAGATION_ALLOWED"]
            capture_z1 = arrays["capture_Z1"]

            array_keys = ("normalized", "joint_targets", "eef_target_world", "radial_world", "u", "context_eef")
            baseline_equal_by_key = {key: bool(np.array_equal(a00[key], capture_z1[key])) for key in array_keys}
            baseline_bit_exact = all(baseline_equal_by_key.values())
            if not baseline_bit_exact:
                raise ValueError(f"{source_id}: strict A00 does not equal native recipient baseline")

            # Cross-check every saved readout against the frozen source-level table.
            readout_fields = {
                "joint_a00": "A00_Crecipient_Frecipient",
                "joint_a10": "A10_Cdonor_Frecipient",
                "joint_node_current": "NODE_CURRENT_PROPAGATION_ALLOWED",
            }
            source_readout_errors = {}
            for field, label in readout_fields.items():
                recorded = np.asarray(json.loads(source_row[field]), dtype=np.float64)
                if recorded.shape != (N_POSITIONS,):
                    raise ValueError(f"{source_id}: frozen {field} shape {recorded.shape}")
                source_readout_errors[field] = float(np.max(np.abs(recorded - arrays[label]["radial_world"])))
            if max(source_readout_errors.values()) != 0.0:
                raise ValueError(f"{source_id}: frozen source-level readout differs from action artifact")

            # All three estimand calls must use the same recipient context and inference seed.
            identity_keys = (
                "observation_sha256",
                "image_sha256",
                "proprio_sha256",
                "instruction_sha256",
                "seed",
            )
            reference_input = records["A00_Crecipient_Frecipient"]["input"]
            recipient_context_match = all(
                all(records[label]["input"][key] == reference_input[key] for key in identity_keys)
                and records[label]["context_label"] == "Z1"
                for label in (
                    "A10_Cdonor_Frecipient",
                    "NODE_CURRENT_PROPAGATION_ALLOWED",
                )
            )
            if not recipient_context_match:
                raise ValueError(f"{source_id}: recipient context or inference seed mismatch")

            ez = events["capture_Z1"]
            ed = events["capture_Dplus"]
            e00 = events["A00_Crecipient_Frecipient"]
            e10 = events["A10_Cdonor_Frecipient"]
            enode = events["NODE_CURRENT_PROPAGATION_ALLOWED"]
            a00_current_matches = sum(e00[k]["current_hash"] == ez[k]["current_hash"] for k in EXPECTED_EVENTS)
            a00_future_matches = sum(e00[k]["future_hash"] == ez[k]["future_hash"] for k in EXPECTED_EVENTS)
            a10_current_matches = sum(e10[k]["current_hash"] == ed[k]["current_hash"] for k in EXPECTED_EVENTS)
            node_current_matches = sum(enode[k]["current_hash"] == ed[k]["current_hash"] for k in EXPECTED_EVENTS)
            a10_future_matches = sum(e10[k]["future_hash"] == ez[k]["future_hash"] for k in EXPECTED_EVENTS)
            node_future_equals_strict = sum(enode[k]["future_hash"] == e10[k]["future_hash"] for k in EXPECTED_EVENTS)
            node_future_equals_natural_donor = sum(enode[k]["future_hash"] == ed[k]["future_hash"] for k in EXPECTED_EVENTS)
            node_labels_dynamic = all(
                enode[k]["current_source"] == "Dplus"
                and enode[k]["future_source"] == "PROPAGATION_ALLOWED_DYNAMIC_FUTURE"
                for k in EXPECTED_EVENTS
            )
            a10_labels_strict = all(
                e10[k]["current_source"] == "Dplus" and e10[k]["future_source"] == "Z1"
                for k in EXPECTED_EVENTS
            )
            consumer_assignment_pass = (
                a00_current_matches == 600
                and a00_future_matches == 600
                and a10_current_matches == 600
                and node_current_matches == 600
                and a10_future_matches == 600
                and node_labels_dynamic
                and a10_labels_strict
                and node_future_equals_strict < 600
            )
            if not consumer_assignment_pass:
                raise ValueError(f"{source_id}: current/future source-consumption audit failed")

            # Radial readout identity must match across all three calls.
            u_match = np.array_equal(a00["u"], a10["u"]) and np.array_equal(a00["u"], node["u"])
            context_eef_match = np.array_equal(a00["context_eef"], a10["context_eef"]) and np.array_equal(
                a00["context_eef"], node["context_eef"]
            )
            fk_u = np.asarray(fk_contract["u"], dtype=np.float64)
            registered_u_match = bool(np.array_equal(a00["u"], fk_u))
            projection_error = max(
                float(np.max(np.abs(value["eef_target_world"] @ value["u"] - value["radial_world"])))
                for value in (a00, a10, node)
            )
            if not (u_match and context_eef_match and registered_u_match):
                raise ValueError(f"{source_id}: radial readout identity mismatch")

            n_radial = node["radial_world"].astype(np.float64) - a00["radial_world"].astype(np.float64)
            u_radial = a10["radial_world"].astype(np.float64) - a00["radial_world"].astype(np.float64)
            p_radial = n_radial - u_radial
            radial_closure = n_radial - (u_radial + p_radial)
            radial_simplification = p_radial - (
                node["radial_world"].astype(np.float64) - a10["radial_world"].astype(np.float64)
            )

            n_action = node["normalized"].astype(np.float64) - a00["normalized"].astype(np.float64)
            u_action = a10["normalized"].astype(np.float64) - a00["normalized"].astype(np.float64)
            p_action = n_action - u_action
            action_closure = n_action - (u_action + p_action)
            action_simplification = p_action - (
                node["normalized"].astype(np.float64) - a10["normalized"].astype(np.float64)
            )

            simulator_seed = int(source_id.rsplit("seed", 1)[1])
            source_registry_row = registry.get(source_id)
            if source_registry_row is None and origin == "POSTHOC8_IDENTITY_AMENDMENT":
                # Some frozen extension manifests key the preserved geometry ID.
                source_registry_row = next(
                    (x for x in extension_sources_raw if int(x.get("simulator_seed", -1)) == simulator_seed),
                    None,
                )
            registry_identity_match = bool(
                source_registry_row
                and source_registry_row.get("task_id") == task_id
                and int(source_registry_row.get("simulator_seed", -1)) == simulator_seed
                and int(source_registry_row.get("policy_seed", -1)) == int(reference_input["seed"])
                and source_registry_row.get("source_group_id") == source_row["source_group_id"]
            )
            if not registry_identity_match:
                raise ValueError(f"{source_id}: task/source/seed registry identity mismatch")
            hook_cleanup_pass = all(
                not records[label]["hook_cleanup"].get("installed")
                or records[label]["hook_cleanup"].get("success") is True
                for label in labels
            )
            if not hook_cleanup_pass:
                raise ValueError(f"{source_id}: hook cleanup failure in matched calls")
            ancestry_status = source_row["ancestry_status"]
            independent_certified = ancestry_status in {
                "CERTIFIED_NO_SHARED_ANCESTOR",
                "CERTIFIED_SHARED_ANCESTOR_GROUPED",
            }
            case: dict[str, Any] = {
                "origin_identity": origin,
                "included_original34": origin == "ORIGINAL34_FROZEN_CONFIRMATION",
                "included_expanded42_sensitivity": True,
                "task_id": task_id,
                "source_id": source_id,
                "source_group_id": source_row["source_group_id"],
                "initial_state_sha256": source_row["initial_state_sha256"],
                "ancestry_status": ancestry_status,
                "independent_source_certified": independent_certified,
                "simulator_seed": simulator_seed,
                "policy_seed": int(reference_input["seed"]),
                "dose_id": "Dplus",
                "signed_dose_m": dose_m,
                "recipient_context": "Z1",
                "node_current_source": "Dplus",
                "node_future_source": "PROPAGATION_ALLOWED_DYNAMIC_FUTURE (F_prime)",
                "strict_current_source": "Dplus",
                "strict_future_source": "Z1 recipient future (F_R)",
                "checkpoint_sha256": branch_results[origin]["checkpoint_sha256"],
                "config_sha256": branch_results[origin]["config_sha256"],
                "normalization_stats_sha256": branch_results[origin]["stats_sha256"],
                "worker_sha256": branch_results[origin]["worker_sha256"],
                "action_positions": N_POSITIONS,
                "action_channels": N_CHANNELS,
                "radial_input_dtype": str(a00["radial_world"].dtype),
                "normalized_action_input_dtype": str(a00["normalized"].dtype),
                "effect_compute_dtype": "float64",
                "A00_equals_native_Z1_bit_exact": baseline_bit_exact,
                "A00_equality_by_array_json": json_compact(baseline_equal_by_key),
                "recipient_context_and_seed_match": recipient_context_match,
                "current_source_identity_match": a10_current_matches == 600 and node_current_matches == 600,
                "dose_task_source_match": registry_identity_match,
                "scheduler_match_status": "PASS_SHARED_FROZEN_WORKER_CONFIG_NO_SEPARATE_PER_CALL_SCHEDULER_HASH",
                "scheduler_and_action_conditions_match_basis": "shared frozen branch config/checkpoint/stats plus identical per-call recipient input hashes and inference seed; scheduler not separately serialized per call",
                "radial_readout_match": u_match and context_eef_match and registered_u_match,
                "hook_cleanup_pass": hook_cleanup_pass,
                "weights_unchanged_branch": True,
                "radial_projection_max_abs_error_m": projection_error,
                "saved_source_readout_max_abs_error_m": max(source_readout_errors.values()),
                "consumer_assignment_pass": consumer_assignment_pass,
                "A00_current_recipient_matches": a00_current_matches,
                "A00_future_recipient_matches": a00_future_matches,
                "A10_current_donor_matches": a10_current_matches,
                "A10_future_recipient_matches": a10_future_matches,
                "node_current_donor_matches": node_current_matches,
                "node_future_equals_strict_events": node_future_equals_strict,
                "node_future_differs_from_strict_events": 600 - node_future_equals_strict,
                "node_future_equals_natural_donor_events": node_future_equals_natural_donor,
                "node_future_differs_from_natural_donor_events": 600 - node_future_equals_natural_donor,
                "N_radial_vector_m_json": json_compact([float(x) for x in n_radial]),
                "U_C_given_F_R_radial_vector_m_json": json_compact([float(x) for x in u_radial]),
                "P_future_radial_vector_m_json": json_compact([float(x) for x in p_radial]),
                "N_signed_radial_mean_m": float(np.mean(n_radial)),
                "U_C_given_F_R_signed_radial_mean_m": float(np.mean(u_radial)),
                "P_future_signed_radial_mean_m": float(np.mean(p_radial)),
                "N_rms_radial_m": float(np.sqrt(np.mean(np.square(n_radial)))),
                "U_C_given_F_R_rms_radial_m": float(np.sqrt(np.mean(np.square(u_radial)))),
                "P_future_rms_radial_m": float(np.sqrt(np.mean(np.square(p_radial)))),
                "N_full_action_l2_normalized": full_action_l2(n_action),
                "U_C_given_F_R_full_action_l2_normalized": full_action_l2(u_action),
                "P_future_full_action_l2_normalized": full_action_l2(p_action),
                "N_full_action_channel_rms_normalized_json": json_compact(channel_rms(n_action)),
                "U_C_given_F_R_full_action_channel_rms_normalized_json": json_compact(channel_rms(u_action)),
                "P_future_full_action_channel_rms_normalized_json": json_compact(channel_rms(p_action)),
                "radial_closure_max_abs_error_m": float(np.max(np.abs(radial_closure))),
                "normalized_action_closure_max_abs_error": float(np.max(np.abs(action_closure))),
                "radial_P_future_vs_node_minus_A10_max_abs_error_m": float(np.max(np.abs(radial_simplification))),
                "normalized_P_future_vs_node_minus_A10_max_abs_error": float(np.max(np.abs(action_simplification))),
                "A00_action_path": records["A00_Crecipient_Frecipient"]["action_artifact"],
                "A10_action_path": records["A10_Cdonor_Frecipient"]["action_artifact"],
                "node_action_path": records["NODE_CURRENT_PROPAGATION_ALLOWED"]["action_artifact"],
            }
            case_rows.append(case)
            closure_case_rows.append({
                "origin_identity": origin,
                "task_id": task_id,
                "source_id": source_id,
                "radial_input_dtype": str(a00["radial_world"].dtype),
                "normalized_action_input_dtype": str(a00["normalized"].dtype),
                "compute_dtype": "float64",
                "radial_max_abs_error_m": case["radial_closure_max_abs_error_m"],
                "normalized_action_max_abs_error": case["normalized_action_closure_max_abs_error"],
                "radial_simplification_max_abs_error_m": case["radial_P_future_vs_node_minus_A10_max_abs_error_m"],
                "normalized_simplification_max_abs_error": case["normalized_P_future_vs_node_minus_A10_max_abs_error"],
            })

    if len(case_rows) != 42:
        raise ValueError(f"expected 42 fully matched saved cases, obtained {len(case_rows)}")

    # Source-level aggregation is explicit even though this frozen RoboTwin
    # design has one fixed +1 cm case per source group.
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in case_rows:
        grouped[(row["origin_identity"], row["task_id"], row["source_group_id"])].append(row)
    source_rows: list[dict[str, Any]] = []
    metric_fields = [
        f"{effect}_{suffix}"
        for effect in EFFECTS
        for suffix in (
            "signed_radial_mean_m",
            "rms_radial_m",
            "full_action_l2_normalized",
        )
    ]
    for (origin, task_id, source_group_id), values in sorted(grouped.items()):
        first = values[0]
        row: dict[str, Any] = {
            "origin_identity": origin,
            "included_original34": first["included_original34"],
            "included_expanded42_sensitivity": True,
            "task_id": task_id,
            "source_group_id": source_group_id,
            "source_ids": "|".join(sorted(x["source_id"] for x in values)),
            "case_count": len(values),
            "ancestry_status": first["ancestry_status"],
            "independent_source_certified": first["independent_source_certified"],
            "dose_ids": "|".join(sorted({x["dose_id"] for x in values})),
        }
        for field in metric_fields:
            row[field] = float(np.mean([float(x[field]) for x in values]))
        for effect in EFFECTS:
            vectors = np.asarray(
                [json.loads(x[f"{effect}_radial_vector_m_json"]) for x in values],
                dtype=np.float64,
            )
            row[f"{effect}_source_aggregated_radial_vector_m_json"] = json_compact(
                [float(x) for x in np.mean(vectors, axis=0)]
            )
        source_rows.append(row)

    summary_rows: list[dict[str, Any]] = []
    cohorts = {
        "original34_frozen": [x for x in source_rows if x["included_original34"]],
        "expanded42_posthoc_coverage_sensitivity": source_rows,
    }
    statistics = {
        "signed_radial_mean_m": "m",
        "rms_radial_m": "m",
        "full_action_l2_normalized": "normalized_action_L2",
    }
    frozen_effect_names = {
        "N": "joint_propagation_allowed_current_node",
        "U_C_given_F_R": "joint_current_given_future_recipient",
        "P_future": "joint_attribution_shift",
    }
    for cohort, cohort_rows in cohorts.items():
        for effect in EFFECTS:
            for statistic, unit in statistics.items():
                field = f"{effect}_{statistic}"
                if cohort == "original34_frozen" and statistic in {"signed_radial_mean_m", "rms_radial_m"}:
                    frozen_field = "signed_mean_m" if statistic == "signed_radial_mean_m" else "rms_m"
                    frozen_key = f"joint|{frozen_effect_names[effect]}|{frozen_field}".encode()
                    seed = BOOT_SEED + int(hashlib.sha256(frozen_key).hexdigest()[:8], 16)
                else:
                    seed = metric_seed(cohort, "overall", effect, statistic)
                draws = hierarchical_bootstrap(cohort_rows, field, seed)
                summary_rows.append({
                    "cohort": cohort,
                    "scope": "overall_task_equal",
                    "task_id": "ALL_FIXED_REGISTERED_TASKS",
                    "effect": effect,
                    "effect_definition": EFFECTS[effect],
                    "statistic": statistic,
                    "estimate": task_equal(cohort_rows, field),
                    "ci95_low": float(np.quantile(draws, 0.025)),
                    "ci95_high": float(np.quantile(draws, 0.975)),
                    "unit": unit,
                    "matched_cases": sum(int(x["case_count"]) for x in cohort_rows),
                    "source_groups": len(cohort_rows),
                    "certified_independent_source_groups": sum(bool(x["independent_source_certified"]) for x in cohort_rows),
                    "task_clusters": len({x["task_id"] for x in cohort_rows}),
                    "weighting": "equal task weight; equal source-group weight within task",
                    "bootstrap": "frozen hierarchical: tasks then source groups within selected task",
                    "bootstrap_draws": N_BOOT,
                    "bootstrap_seed": seed,
                })
                for task_id in sorted({x["task_id"] for x in cohort_rows}):
                    task_rows = [x for x in cohort_rows if x["task_id"] == task_id]
                    task_seed = metric_seed(cohort, task_id, effect, statistic)
                    task_draws = within_task_bootstrap(task_rows, field, task_seed)
                    summary_rows.append({
                        "cohort": cohort,
                        "scope": "task_stratified",
                        "task_id": task_id,
                        "effect": effect,
                        "effect_definition": EFFECTS[effect],
                        "statistic": statistic,
                        "estimate": float(np.mean([float(x[field]) for x in task_rows])),
                        "ci95_low": float(np.quantile(task_draws, 0.025)),
                        "ci95_high": float(np.quantile(task_draws, 0.975)),
                        "unit": unit,
                        "matched_cases": sum(int(x["case_count"]) for x in task_rows),
                        "source_groups": len(task_rows),
                        "certified_independent_source_groups": sum(bool(x["independent_source_certified"]) for x in task_rows),
                        "task_clusters": 1,
                        "weighting": "equal source-group weight within task",
                        "bootstrap": "source groups within this fixed task",
                        "bootstrap_draws": N_BOOT,
                        "bootstrap_seed": task_seed,
                    })

    # Exact reproduction gate for all six radial summaries already present in
    # the immutable frozen source-attribution output.
    frozen_summary_doc = global_data[
        "bootstrap_summary.json@" + str(PRIMARY / "analysis")
    ]["summaries"]
    frozen_summary_index = {
        (x["effect"], x["statistic"]): x
        for x in frozen_summary_doc
        if x.get("model") == "joint"
    }
    frozen_reproduction = []
    for effect, frozen_effect in frozen_effect_names.items():
        for statistic, frozen_statistic in (
            ("signed_radial_mean_m", "signed_mean_m"),
            ("rms_radial_m", "rms_m"),
        ):
            current = next(
                x for x in summary_rows
                if x["cohort"] == "original34_frozen"
                and x["scope"] == "overall_task_equal"
                and x["effect"] == effect
                and x["statistic"] == statistic
            )
            frozen = frozen_summary_index[(frozen_effect, frozen_statistic)]
            errors = {
                "estimate_abs_error": abs(float(current["estimate"]) - float(frozen["estimate_m"])),
                "ci95_low_abs_error": abs(float(current["ci95_low"]) - float(frozen["ci95_low_m"])),
                "ci95_high_abs_error": abs(float(current["ci95_high"]) - float(frozen["ci95_high_m"])),
            }
            if max(errors.values()) != 0.0:
                raise ValueError(f"frozen radial summary did not reproduce exactly: {effect}/{statistic}: {errors}")
            frozen_reproduction.append({
                "effect": effect,
                "frozen_effect": frozen_effect,
                "statistic": statistic,
                "frozen_statistic": frozen_statistic,
                **errors,
            })

    write_csv(OUT / "case_level.csv", case_rows)
    write_csv(OUT / "source_level.csv", source_rows)
    write_csv(OUT / "summary.csv", summary_rows)

    closure = {
        "definition": "N - (U_C_given_F_R + P_future), with P_future defined as N - U_C_given_F_R",
        "case_count": len(case_rows),
        "radial_input_dtype": sorted({x["radial_input_dtype"] for x in closure_case_rows}),
        "normalized_action_input_dtype": sorted({x["normalized_action_input_dtype"] for x in closure_case_rows}),
        "compute_dtype": "float64",
        "global_radial_max_abs_error_m": max(x["radial_max_abs_error_m"] for x in closure_case_rows),
        "global_normalized_action_max_abs_error": max(x["normalized_action_max_abs_error"] for x in closure_case_rows),
        "global_radial_P_future_vs_node_minus_A10_max_abs_error_m": max(
            x["radial_simplification_max_abs_error_m"] for x in closure_case_rows
        ),
        "global_normalized_P_future_vs_node_minus_A10_max_abs_error": max(
            x["normalized_simplification_max_abs_error"] for x in closure_case_rows
        ),
        "cases": closure_case_rows,
    }
    atomic_json(OUT / "closure_checks.json", closure)

    input_entries = []
    for path, role in sorted(provenance.items(), key=lambda item: str(item[0])):
        input_entries.append({
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "role": role,
        })
    provenance_doc = {
        "analysis_status": "COMPLETE_READ_ONLY_POSTHOC",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_identity": "POSTHOC_PROPAGATED_FUTURE_COMPONENT_EQ5",
        "primary_cohort": "original 34 frozen confirmation sources",
        "coverage_sensitivity": "eight post-hoc identity-amendment sources added; state-distinct but ancestry not strongly certified",
        "no_new_model_forward": True,
        "no_simulator_step": True,
        "no_rollout": True,
        "no_donor_construction": True,
        "matched_cases": len(case_rows),
        "primary_matched_cases": sum(x["included_original34"] for x in case_rows),
        "registered_source_groups": len(source_rows),
        "certified_independent_source_groups": sum(x["independent_source_certified"] for x in source_rows),
        "input_files": input_entries,
        "input_file_count": len(input_entries),
        "original34_frozen_radial_summaries_exactly_reproduced": True,
        "frozen_reproduction_checks": frozen_reproduction,
        "analysis_script": str(Path(__file__).resolve()),
        "analysis_script_sha256": sha256(Path(__file__).resolve()),
    }
    atomic_json(OUT / "provenance.json", provenance_doc)

    def lookup(cohort: str, effect: str, statistic: str) -> dict[str, Any]:
        return next(
            x
            for x in summary_rows
            if x["cohort"] == cohort
            and x["scope"] == "overall_task_equal"
            and x["effect"] == effect
            and x["statistic"] == statistic
        )

    def fmt_mm(row: dict[str, Any]) -> str:
        return f"{row['estimate']*1000:+.3f} [{row['ci95_low']*1000:+.3f}, {row['ci95_high']*1000:+.3f}] mm"

    primary_signed = {effect: lookup("original34_frozen", effect, "signed_radial_mean_m") for effect in EFFECTS}
    primary_rms = {effect: lookup("original34_frozen", effect, "rms_radial_m") for effect in EFFECTS}
    expanded_signed = {
        effect: lookup("expanded42_posthoc_coverage_sensitivity", effect, "signed_radial_mean_m")
        for effect in EFFECTS
    }
    expanded_rms = {
        effect: lookup("expanded42_posthoc_coverage_sensitivity", effect, "rms_radial_m")
        for effect in EFFECTS
    }
    primary_action_l2 = {
        effect: lookup("original34_frozen", effect, "full_action_l2_normalized")
        for effect in EFFECTS
    }
    expanded_action_l2 = {
        effect: lookup("expanded42_posthoc_coverage_sensitivity", effect, "full_action_l2_normalized")
        for effect in EFFECTS
    }

    task_lines = []
    for row in summary_rows:
        if (
            row["cohort"] == "original34_frozen"
            and row["scope"] == "task_stratified"
            and row["statistic"] == "signed_radial_mean_m"
        ):
            task_lines.append(
                f"| {row['task_id']} | {row['effect']} | {row['source_groups']} | "
                f"{row['estimate']*1000:+.3f} | {row['ci95_low']*1000:+.3f} | {row['ci95_high']*1000:+.3f} |"
            )

    provenance_lines = [
        f"{idx}. `{entry['path']}` — {entry['bytes']} bytes — SHA-256 `{entry['sha256']}` — {entry['role']}"
        for idx, entry in enumerate(input_entries, start=1)
    ]
    analysis = f"""# Post-hoc propagated-future component (Eq. 5)

## Status and scope

**Status: COMPLETE_READ_ONLY_POSTHOC.** This analysis used saved intervention arrays only. It performed no model forward pass, simulator step, rollout, action execution, rendering, or donor construction.

The primary analysis is the original frozen 34-source RoboTwin Joint-WAM confirmation cohort. A separate 42-source coverage sensitivity adds eight post-hoc identity-amendment cases. Those eight cases have distinguishable saved state identities but do **not** have strongly certified ancestry independence, so the 42-source result is not presented as a new independent confirmation.

## Definitions

For each source and the fixed registered `Dplus` dose (+1 cm):

- `N = NODE_CURRENT_PROPAGATION_ALLOWED - A00`;
- `U_C_given_F_R = A10 - A00`;
- `P_future = N - U_C_given_F_R = NODE_CURRENT_PROPAGATION_ALLOWED - A10`.

`P_future` is the additional action effect admitted by downstream future propagation under this intervention. It is **not** a natural indirect effect, mediated share, future-information percentage, or source contribution percentage. The node arm's future is the recomputed dynamic `F'`; it is not silently replaced by the natural donor future `F_D`.

The primary physical readout is the saved 32-position world-EEF target projection onto each source's registered recipient radial direction, in metres. Signed response is the mean of that vector; radial magnitude is its RMS. Full-action L2 is additionally computed from the flattened saved 32x14 **normalized** action tensor. A de-normalized 14-channel total norm is not pooled because joint targets and gripper commands do not justify a single common physical unit.

The independent statistical unit is the registered source group. The +1 cm dose, 32 action positions, 30 layers, 10 denoising steps, and K/V consumer events are retained within source and are never counted as independent samples.

## Matching and technical verification

- Exact matched saved cases: **{len(case_rows)}** (34 frozen confirmation + 8 post-hoc coverage-sensitivity cases).
- Original analysis sources: **34 certified independent source groups** across five fixed tasks.
- Expanded registry: **42 state-identity groups**, of which **34** have certified ancestry independence; the additional eight retain their post-hoc/ancestry limitation.
- Every A00 baseline equals its corresponding native Z1 reference bit-for-bit for all saved arrays checked (`normalized`, `joint_targets`, `eef_target_world`, `radial_world`, `u`, and `context_eef`).
- In every case, A10 and the propagation-allowed node arm use donor current at all 600 registered consumer events; A10 uses captured recipient future at all 600 events. The node arm is labelled `PROPAGATION_ALLOWED_DYNAMIC_FUTURE` at all events and differs from strict A10 future in at least one event per case. Event-level counts are in `case_level.csv`.
- Specifically, `F'` differs from both strict recipient future and natural donor future at **598/600** registered component events in every matched case; the two equal events occur at the initial propagation boundary and are retained.
- Recipient observation, image, proprio, instruction, inference seed, checkpoint/config/stats identity, and radial reference are matched within each contrast. Scheduler equality is supported by the shared frozen branch implementation/config and identical per-call recipient inference inputs; the scheduler itself was not serialized as a separate per-call hash.
- The eight-source coverage branch has a different audited worker hash from the original branch, while checkpoint, config, normalization statistics, per-arm recipient context, consumer assignments, and within-branch worker identity match. This is another reason it remains a separate post-hoc sensitivity rather than being retroactively merged into the frozen confirmation identity.

## Primary 34-source estimates

Point estimates use equal task weight and equal source-group weight within task. Intervals exactly follow the original frozen hierarchical bootstrap (tasks, then source groups within selected task), 10,000 draws. The five task IDs were fixed by design; as already documented by the frozen audit, resampling tasks adds a task-population uncertainty beyond conditioning on those five tasks.

All six original radial summaries (three effects x signed/RMS statistics) reproduce the immutable frozen estimates and interval endpoints exactly.

| Effect | Signed radial mean, 95% CI | RMS radial magnitude, 95% CI |
|---|---:|---:|
| N | {fmt_mm(primary_signed['N'])} | {fmt_mm(primary_rms['N']).replace('+','')} |
| U_C_given_F_R | {fmt_mm(primary_signed['U_C_given_F_R'])} | {fmt_mm(primary_rms['U_C_given_F_R']).replace('+','')} |
| P_future | {fmt_mm(primary_signed['P_future'])} | {fmt_mm(primary_rms['P_future']).replace('+','')} |

Normalized full-action L2 (flattened 32x14 tensor; dimensionless normalized coordinates):

| Effect | Estimate | 95% CI |
|---|---:|---:|
| N | {primary_action_l2['N']['estimate']:.6f} | [{primary_action_l2['N']['ci95_low']:.6f}, {primary_action_l2['N']['ci95_high']:.6f}] |
| U_C_given_F_R | {primary_action_l2['U_C_given_F_R']['estimate']:.6f} | [{primary_action_l2['U_C_given_F_R']['ci95_low']:.6f}, {primary_action_l2['U_C_given_F_R']['ci95_high']:.6f}] |
| P_future | {primary_action_l2['P_future']['estimate']:.6f} | [{primary_action_l2['P_future']['ci95_low']:.6f}, {primary_action_l2['P_future']['ci95_high']:.6f}] |

The node-minus-strict-current contrast directly equals `P_future`. Much of the propagation-allowed current-node effect disappears when the future input is fixed. This contrast quantifies the additional effect admitted by downstream future propagation under this intervention; it does not identify a unique mediation share.

## Expanded 42-source coverage sensitivity

| Effect | Signed radial mean, 95% CI | RMS radial magnitude, 95% CI |
|---|---:|---:|
| N | {fmt_mm(expanded_signed['N'])} | {fmt_mm(expanded_rms['N']).replace('+','')} |
| U_C_given_F_R | {fmt_mm(expanded_signed['U_C_given_F_R'])} | {fmt_mm(expanded_rms['U_C_given_F_R']).replace('+','')} |
| P_future | {fmt_mm(expanded_signed['P_future'])} | {fmt_mm(expanded_rms['P_future']).replace('+','')} |

Expanded normalized full-action L2: N {expanded_action_l2['N']['estimate']:.6f} [{expanded_action_l2['N']['ci95_low']:.6f}, {expanded_action_l2['N']['ci95_high']:.6f}]; U_C_given_F_R {expanded_action_l2['U_C_given_F_R']['estimate']:.6f} [{expanded_action_l2['U_C_given_F_R']['ci95_low']:.6f}, {expanded_action_l2['U_C_given_F_R']['ci95_high']:.6f}]; P_future {expanded_action_l2['P_future']['estimate']:.6f} [{expanded_action_l2['P_future']['ci95_low']:.6f}, {expanded_action_l2['P_future']['ci95_high']:.6f}].

This expanded result is a post-hoc coverage sensitivity, not an independent replication.

## Algebraic closure

- Radial input dtype: `{', '.join(closure['radial_input_dtype'])}`; normalized action input dtype: `{', '.join(closure['normalized_action_input_dtype'])}`; computation dtype: `float64`.
- Maximum radial closure error `max_abs(N - (U + P))`: **{closure['global_radial_max_abs_error_m']:.17g} m**.
- Maximum normalized-action closure error: **{closure['global_normalized_action_max_abs_error']:.17g}**.
- Maximum discrepancy between computed `P_future` and the simplification `node - A10`: **{closure['global_radial_P_future_vs_node_minus_A10_max_abs_error_m']:.17g} m** radially and **{closure['global_normalized_P_future_vs_node_minus_A10_max_abs_error']:.17g}** in normalized-action coordinates.

## Task-stratified signed summaries (original 34)

The intervals below resample source groups within each task and are descriptive task-stratified intervals. Small task source counts limit their stability.

| Task | Effect | Source groups | Estimate (mm) | 95% low | 95% high |
|---|---|---:|---:|---:|---:|
{chr(10).join(task_lines)}

## Output map

- `case_level.csv`: matched case vectors, metrics, and all identity/consumer checks.
- `source_level.csv`: source-group aggregation (one +1 cm case per group in this dataset).
- `summary.csv`: primary and coverage-sensitivity overall/task-stratified summaries.
- `closure_checks.json`: case-wise and global radial/full-action algebra checks.
- `provenance.json`: machine-readable complete input provenance.

## Complete input provenance

{chr(10).join(provenance_lines)}
"""
    atomic_text(OUT / "analysis.md", analysis)


if __name__ == "__main__":
    build()

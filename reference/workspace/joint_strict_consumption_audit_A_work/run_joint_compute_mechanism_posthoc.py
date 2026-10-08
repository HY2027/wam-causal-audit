#!/usr/bin/env python3
"""Post-hoc Joint audit of physical-information routing after K=5 compute reduction."""

from __future__ import annotations
from wam_causal_audit.paths import resolve as _release_path

import argparse
import csv
import gc
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

WORK = Path(__file__).resolve().parent
sys.path.insert(0, str(WORK))
import confirmatory_common as Q  # noqa: E402
import run_c_early_stop as C  # noqa: E402
import run_experiment_a as A  # noqa: E402
import run_joint_baseline_diagnostics as D  # noqa: E402
import run_joint_new_feature_diagnostics as NF  # noqa: E402
from capture import make_capture  # noqa: E402

SOURCE = Q.JOINT
OUT = Path(_release_path('@DATA@/wam_factor_routing_v5/joint_compute_mechanism_posthoc_v1_20260909'))
DONOR = SOURCE / "f3g_donor_registry.csv"
CAUSAL = SOURCE / "new_trajectory_causal_features_pre_label/joint_new_full_and_causal_features.csv"
LABEL = SOURCE / "joint_new_trajectory_response_damage.csv"
EPISODE = SOURCE / "joint_new_trajectory_episode_outcomes.csv"
TECH = OUT / "technical_shards"
FORMAL = OUT / "formal_shards"
ACTION_ROOT = OUT / "actions"
NBOOT = 10_000
BOOT_SEED = 20260909
K = 5


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(A.jsonable(value), indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    tmp.replace(path)


def write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader(); writer.writerows(rows)
    tmp.replace(path)


def state_rows() -> list[dict[str, Any]]:
    return sorted(Q.registry("joint"), key=lambda row: (int(row["task_id"]), int(row["trajectory_id"])))


def donor_rows() -> pd.DataFrame:
    return pd.read_csv(DONOR).sort_values(["task_id", "trajectory_id", "signed_dose_cm"])


def prepared(runner: Any, path: str, instruction: str) -> Mapping[str, Any]:
    return runner._prepared(A.load_npz(Path(path)), instruction)


def source_seed(row: Mapping[str, Any]) -> int:
    return int(row["policy_seed"])


def schedule_mapping(schedule: Mapping[int, list[dict[str, torch.Tensor]]]) -> dict[tuple[int, int], dict[str, torch.Tensor]]:
    return {(step, layer): values for step, layers in schedule.items() for layer, values in enumerate(layers)}


def schedule_hash(schedule: Mapping[int, list[dict[str, torch.Tensor]]]) -> str:
    digest = hashlib.sha256()
    for step, layers in sorted(schedule.items()):
        for layer, values in enumerate(layers):
            digest.update(f"{step}:{layer}".encode())
            for component in ("k", "v"):
                digest.update(A.tensor_sha256(values[component]).encode())
    return digest.hexdigest()


def mix_layer(current: Mapping[str, torch.Tensor], future: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    result = {}
    for component in ("k", "v"):
        value = current[component]; tokens = value.shape[1] // 3
        result[component] = torch.cat((value[:, :tokens], future[component][:, tokens:]), dim=1)
    return result


@torch.no_grad()
def capture_k5_source(runner: Any, item: Mapping[str, Any], seed: int) -> tuple[torch.Tensor, dict[int, list[dict[str, torch.Tensor]]], dict[str, Any]]:
    model, policy = runner.model, runner.policy
    torch.cuda.synchronize(model.device); torch.cuda.reset_peak_memory_stats(model.device); started = time.perf_counter()
    _, video, action, first, vts, vds, ats, ads = C.initialize(model, policy, item, seed)
    controller = C.SelectiveCacheController(model, set(range(K))); controller.install()
    mask = None; video_seq_len = None; tokens_per_group = None; action_steps = 0
    try:
        for step, (tv, dv, ta, da) in enumerate(zip(vts, vds, ats, ads)):
            controller.set_step(step)
            if step < K:
                video_pre = model.video_expert.pre_dit(
                    x=video, timestep=tv.unsqueeze(0).to(video), context=item["context"],
                    context_mask=item["context_mask"], action=None,
                    fuse_vae_embedding_in_latents=bool(getattr(model.video_expert, "fuse_vae_embedding_in_latents", False)))
                action_pre = model.action_expert.pre_dit(
                    action_tokens=action, timestep=ta.unsqueeze(0).to(action),
                    context=item["context"], context_mask=item["context_mask"])
                tokens_per_group = int(video_pre["meta"]["tokens_per_frame"])
                video_seq_len = int(video_pre["tokens"].shape[1])
                mask = model._build_mot_attention_mask(video_seq_len, action_pre["tokens"].shape[1],
                                                       tokens_per_group, video_pre["tokens"].device)
                output = model.mot(
                    embeds_all={"video": video_pre["tokens"], "action": action_pre["tokens"]}, attention_mask=mask,
                    freqs_all={"video": video_pre["freqs"], "action": action_pre["freqs"]},
                    context_all={"video": {"context": video_pre["context"], "mask": video_pre["context_mask"]},
                                 "action": {"context": action_pre["context"], "mask": action_pre["context_mask"]}},
                    t_mod_all={"video": video_pre["t_mod"], "action": action_pre["t_mod"]})
                pred_video = model.video_expert.post_dit(output["video"], video_pre)
                pred_action = model.action_expert.post_dit(output["action"], action_pre)
                video = model.infer_video_scheduler.step(pred_video, dv, video); video[:, :, 0:1] = first.clone()
            else:
                pred_action = model._predict_action_noise_with_cache(
                    latents_action=action, timestep_action=ta.unsqueeze(0).to(action),
                    context=item["context"], context_mask=item["context_mask"],
                    video_kv_cache=[controller.cache[(K - 1, layer)] for layer in range(30)],
                    attention_mask=mask, video_seq_len=video_seq_len)
            action = model.infer_action_scheduler.step(pred_action, da, action); action_steps += 1
    finally:
        controller.uninstall()
    torch.cuda.synchronize(model.device)
    schedule = {step: [controller.cache[(step, layer)] for layer in range(30)] for step in range(K)}
    diag = {
        "latency_seconds": time.perf_counter() - started, "world_steps": K, "action_steps": action_steps,
        "world_layer_calls": controller.video_calls, "joint_action_layer_calls": controller.action_calls,
        "schedule_hash": schedule_hash(schedule), "cache_layers": 30,
        "tokens_per_group": tokens_per_group, "video_seq_len": video_seq_len,
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated(model.device)),
        "cache_scope": "this policy call only; steps 0-4 generated online; steps 5-9 consume step-4 cache",
    }
    return action[0].detach().cpu().float(), schedule, diag


@torch.no_grad()
def strict_k5(runner: Any, recipient: Mapping[str, Any], seed: int,
              current_schedule: Mapping[int, list[dict[str, torch.Tensor]]],
              future_schedule: Mapping[int, list[dict[str, torch.Tensor]]]) -> tuple[torch.Tensor, dict[str, Any]]:
    model, policy = runner.model, runner.policy
    torch.cuda.synchronize(model.device); torch.cuda.reset_peak_memory_stats(model.device); started = time.perf_counter()
    _, video, action, first, vts, vds, ats, ads = C.initialize(model, policy, recipient, seed)
    controller = A.VideoKVController(model)
    controller.current_cache = schedule_mapping(current_schedule)
    controller.future_cache = schedule_mapping(future_schedule)
    controller.install()
    mask = None; video_seq_len = None; tokens_per_group = None; action_steps = 0
    tail_current_exact = 0; tail_future_exact = 0; tail_max_error = 0.0
    mixed_stop = [mix_layer(current_schedule[K - 1][layer], future_schedule[K - 1][layer]) for layer in range(30)]
    try:
        for step, (tv, dv, ta, da) in enumerate(zip(vts, vds, ats, ads)):
            if step < K:
                controller.step = step
                video_pre = model.video_expert.pre_dit(
                    x=video, timestep=tv.unsqueeze(0).to(video), context=recipient["context"],
                    context_mask=recipient["context_mask"], action=None,
                    fuse_vae_embedding_in_latents=bool(getattr(model.video_expert, "fuse_vae_embedding_in_latents", False)))
                action_pre = model.action_expert.pre_dit(
                    action_tokens=action, timestep=ta.unsqueeze(0).to(action),
                    context=recipient["context"], context_mask=recipient["context_mask"])
                tokens_per_group = int(video_pre["meta"]["tokens_per_frame"])
                video_seq_len = int(video_pre["tokens"].shape[1])
                mask = model._build_mot_attention_mask(video_seq_len, action_pre["tokens"].shape[1],
                                                       tokens_per_group, video_pre["tokens"].device)
                output = model.mot(
                    embeds_all={"video": video_pre["tokens"], "action": action_pre["tokens"]}, attention_mask=mask,
                    freqs_all={"video": video_pre["freqs"], "action": action_pre["freqs"]},
                    context_all={"video": {"context": video_pre["context"], "mask": video_pre["context_mask"]},
                                 "action": {"context": action_pre["context"], "mask": action_pre["context_mask"]}},
                    t_mod_all={"video": video_pre["t_mod"], "action": action_pre["t_mod"]})
                pred_video = model.video_expert.post_dit(output["video"], video_pre)
                pred_action = model.action_expert.post_dit(output["action"], action_pre)
                video = model.infer_video_scheduler.step(pred_video, dv, video); video[:, :, 0:1] = first.clone()
            else:
                # The original K=5 path no longer invokes the world blocks here.
                # Verify every layer of the explicit mixed stop cache before the
                # action-only consumer uses it at each of the five tail steps.
                for layer, mixed in enumerate(mixed_stop):
                    tokens = mixed["k"].shape[1] // 3
                    for component in ("k", "v"):
                        current = current_schedule[K - 1][layer][component][:, :tokens].to(mixed[component])
                        future = future_schedule[K - 1][layer][component][:, tokens:].to(mixed[component])
                        ce = float(torch.max(torch.abs(mixed[component][:, :tokens] - current)).item())
                        fe = float(torch.max(torch.abs(mixed[component][:, tokens:] - future)).item())
                        tail_max_error = max(tail_max_error, ce, fe)
                        tail_current_exact += int(torch.equal(mixed[component][:, :tokens], current))
                        tail_future_exact += int(torch.equal(mixed[component][:, tokens:], future))
                pred_action = model._predict_action_noise_with_cache(
                    latents_action=action, timestep_action=ta.unsqueeze(0).to(action),
                    context=recipient["context"], context_mask=recipient["context_mask"],
                    video_kv_cache=mixed_stop, attention_mask=mask, video_seq_len=video_seq_len)
            action = model.infer_action_scheduler.step(pred_action, da, action); action_steps += 1
    finally:
        controller.uninstall()
    torch.cuda.synchronize(model.device)
    hook = controller.summary(K, 30)
    expected_tail_components = (10 - K) * 30 * 2
    exact = all((hook["hook_reached_all_video_sites"], controller.action_calls == 10 * 30,
                 hook["strict_injection_reached_all_video_sites"], hook["all_current_consumed_values_exact"],
                 hook["all_future_consumed_values_exact"], tail_current_exact == expected_tail_components,
                 tail_future_exact == expected_tail_components, tail_max_error == 0.0))
    diag = {
        "latency_seconds": time.perf_counter() - started, "world_steps": K, "action_steps": action_steps,
        "first_five_hook": hook, "tail_action_steps": 10 - K,
        "tail_expected_component_checks": expected_tail_components,
        "tail_current_exact_count": tail_current_exact, "tail_future_exact_count": tail_future_exact,
        "tail_max_abs_error": tail_max_error, "all_consumed_sources_exact": exact,
        "action_calls_total": controller.action_calls,
        "summary_action_site_flag_not_used": "generic summary couples action expectation to world steps; K5 action remains 10x30",
        "mixed_stop_cache_hash": C.cache_hash(mixed_stop),
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated(model.device)),
    }
    if not exact or action_steps != 10 or controller.video_calls != 150 or controller.action_calls != 300:
        raise AssertionError(f"K5_STRICT_CONSUMER_GATE_FAILED:{diag}")
    return action[0].detach().cpu().float(), diag


def components(delta: np.ndarray, axis: np.ndarray, prefix: str, dose: float) -> dict[str, float]:
    translation = delta[:, :3]; radial = translation @ axis
    return {
        f"{prefix}_radial_mean": float(radial.mean()),
        f"{prefix}_dose_aligned_radial_mean": float(radial.mean() * np.sign(dose)),
        f"{prefix}_radial_l2": float(np.linalg.norm(radial)),
        f"{prefix}_translation_l2": float(np.linalg.norm(translation)),
        f"{prefix}_rotation_l2": float(np.linalg.norm(delta[:, 3:6])),
        f"{prefix}_gripper_l2": float(np.linalg.norm(delta[:, 6])),
    }


def effect_record(actions: Mapping[str, np.ndarray], axis: np.ndarray, dose: float) -> dict[str, float]:
    deltas = {
        "future_given_current_recipient": actions["A01"] - actions["A00"],
        "future_given_current_donor": actions["A11"] - actions["A10"],
        "current_given_future_recipient": actions["A10"] - actions["A00"],
        "current_given_future_donor": actions["A11"] - actions["A01"],
        "joint": actions["A11"] - actions["A00"],
        "interaction": actions["A11"] - actions["A10"] - actions["A01"] + actions["A00"],
    }
    result: dict[str, float] = {}
    for prefix, delta in deltas.items():
        result.update(components(delta, axis, prefix, dose))
    return result


def audit() -> None:
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "figures").mkdir(exist_ok=True)
    states = state_rows(); donors = donor_rows(); causal = pd.read_csv(CAUSAL); labels = pd.read_csv(LABEL)
    episodes = pd.read_csv(EPISODE)
    if len(states) != 50 or len(donors) != 200 or not donors.donor_valid.astype(bool).all():
        raise RuntimeError("frozen state/donor coverage differs from expected 50/200 valid set")
    if len(causal) != 200 or len(labels) != 400 or len(episodes) != 150:
        raise RuntimeError("frozen mechanism/label/closed-loop asset counts differ")
    causal_by = causal.set_index("case_id"); label_by = labels.set_index(["case_id", "K"])
    episode_by = episodes.set_index(["candidate_id", "configuration"])
    dn = D.denormalizer(); axes = NF.axes(); coverage = []; evidence = []
    for donor in donors.itertuples():
        case = str(donor.case_id); cid = str(donor.candidate_id)
        native_path = Path(str(causal_by.loc[case, "actions_path"])); short_path = Path(str(label_by.loc[(case, 5), "actions_path"]))
        native_ok = native_path.is_file() and sha(native_path) == str(causal_by.loc[case, "actions_sha256"])
        short_ok = short_path.is_file() and sha(short_path) == str(label_by.loc[(case, 5), "actions_sha256"])
        for config in ("native", "K5"):
            for cell in ("A00", "A10", "A01", "A11"):
                reusable = config == "native" and native_ok
                coverage.append({
                    "candidate_id": cid, "case_id": case, "task_id": int(donor.task_id),
                    "trajectory_id": int(donor.trajectory_id), "signed_dose_cm": float(donor.signed_dose_cm),
                    "configuration": config, "intervention": cell,
                    "status": "REUSABLE_HASH_VERIFIED" if reusable else "NEEDS_K5_STRICT_FORWARD",
                    "source_path": str(native_path if reusable else short_path),
                    "source_hash_verified": bool(native_ok if reusable else short_ok),
                    "donor_qc": "TECHNICALLY_VALID", "donor_prediction_action_executed": False,
                })
        native_actions = {key: dn(value) for key, value in np.load(native_path).items() if key in ("A00", "A10", "A01", "A11")}
        native_effect = effect_record(native_actions, axes[cid], float(donor.signed_dose_cm))
        native_episode = episode_by.loc[(cid, "native")]
        for budget in (5, 8):
            config = f"K{budget}"; simplified_episode = episode_by.loc[(cid, config)]
            label = label_by.loc[(case, budget)]
            evidence.append({
                "candidate_id": cid, "case_id": case, "task_id": int(donor.task_id),
                "trajectory_id": int(donor.trajectory_id), "signed_dose_cm": float(donor.signed_dose_cm),
                "configuration": config,
                "native_UF_radial_l2": native_effect["future_given_current_donor_radial_l2"],
                "native_future_given_current_recipient_radial_l2": native_effect["future_given_current_recipient_radial_l2"],
                "response_damage_radial_l2": float(label.damage_radial_l2),
                "closedloop_native_success": bool(native_episode.success),
                "closedloop_simplified_success": bool(simplified_episode.success),
                "closedloop_outcome_change": "SAME" if bool(native_episode.success) == bool(simplified_episode.success)
                    else ("NEW_FAILURE" if bool(native_episode.success) else "REVERSE_IMPROVEMENT"),
                "native_policy_calls": int(native_episode.policy_calls),
                "simplified_policy_calls": int(simplified_episode.policy_calls),
                "paired_per_call_fraction_saved": 1.0 - (float(simplified_episode.policy_compute_seconds) / float(simplified_episode.policy_calls)) /
                    (float(native_episode.policy_compute_seconds) / float(native_episode.policy_calls)),
                "paired_cumulative_fraction_saved": 1.0 - float(simplified_episode.policy_compute_seconds) / float(native_episode.policy_compute_seconds),
                "mechanism_grid_available": budget == 5 and False,
                "evidence_scope": "posthoc same-state linkage; association is exploratory",
            })
    coverage_path = OUT / "asset_coverage_and_reuse.csv"; write(coverage_path, coverage)
    evidence_path = OUT / "same_state_evidence.csv"; write(evidence_path, evidence)

    forward_rows = []
    for state_index, state in enumerate(states):
        cid = str(state["candidate_id"]); group = donors[donors.candidate_id == cid]
        for donor in group.itertuples():
            for cell in ("A00", "A10", "A01", "A11"):
                forward_rows.append({
                    "candidate_id": cid, "case_id": donor.case_id, "task_id": int(state["task_id"]),
                    "trajectory_id": int(state["trajectory_id"]), "signed_dose_cm": float(donor.signed_dose_cm),
                    "configuration": "K5", "intervention": cell, "worker_shard": state_index % 4,
                    "forward_unit_id": f"K5_A00__{cid}" if cell == "A00" else f"K5_{cell}__{donor.case_id}",
                    "unique_action_forward": cell != "A00" or float(donor.signed_dose_cm) == float(group.signed_dose_cm.min()),
                    "recipient_source_capture_id": f"K5_SOURCE_RECIPIENT__{cid}",
                    "donor_source_capture_id": f"K5_SOURCE_DONOR__{donor.case_id}",
                    "status": "FROZEN_PLANNED", "outcome_used_for_selection": False,
                })
    forward_path = OUT / "mechanism_forward_registry.csv"; write(forward_path, forward_rows)

    selected = []
    for task, group in donors.groupby("task_id"):
        candidate = sorted(group.candidate_id.unique())[0]
        selected.append(group[(group.candidate_id == candidate) & (group.signed_dose_cm == -1.0)].iloc[0].case_id)
    protocol = f"""# Joint fixed-compute physical-information routing protocol

Status: `JOINT_COMPUTE_MECHANISM_POSTHOC_PROTOCOL_FROZEN`

This is a post-hoc mechanism supplement after C outcomes were known. It is not part of the prior confirmation protocol and does not overwrite any frozen asset.

## Frozen scope

- Exactly 50 registered restore_v2 starts from tasks 5–9 and all 200 already-valid F3-G donors at signed doses -1, -0.5, +0.5, +1 cm.
- Joint checkpoint, dtype, seed, schedulers, 30 layers and 10 action-denoising steps are unchanged.
- Native strict A00/A10/A01/A11 assets are reused only after path and SHA-256 verification. K=8 receives same-state descriptive linkage only; no K=8 mechanism grid is generated.
- K=5 is the only new mechanism path. It runs five world steps and ten action steps. Steps 0–4 consume the source cache from their own step; steps 5–9 consume the last cache generated at step 4, within the same policy call only.

## Consumer-edge semantics

The intervention is at `MoT._build_expert_attention_io` return, immediately before concatenation with action K/V and mixed attention. K is post projection/RMSNorm/RoPE; V is post projection. The current slice is `[0,98)` and future slice `[98,294)`. At every actual consumer event both sides are explicitly constructed:

- A00 = current recipient + future recipient
- A10 = current donor + future recipient
- A01 = current recipient + future donor
- A11 = current donor + future donor

For K=5 tail action-only steps, the complete 30-layer mixed stop cache is constructed from the recipient/donor K=5 step-4 caches and verified immediately before consumption. Native caches never supply the K=5 mechanism grid. Text, proprio and all other action context remain recipient. Recipient and F3-G donor context hashes must be identical, but recipient context is used even when they match.

## Readouts and statistics

Actions are denormalized with the unchanged Joint dataset statistics. Translation is projected on the recipient-defined horizontal radial axis. Every endpoint retains raw signed radial mean, dose-aligned signed radial mean, radial L2, translation L2, rotation L2 and gripper L2. The four conditional increments are A01-A00, A11-A10, A10-A00 and A11-A01; joint is A11-A00 and interaction is A11-A10-A01+A00.

The two primary descriptive mechanism endpoints are the K=5 versus native paired differences for future|current-recipient and future|current-donor, reported separately for signed radial mean and radial L2. Results are shown by each signed dose and with equal donor weighting at the state level. Task→source-trajectory bootstrap uses 10,000 resamples and preserves all donor/cell/configuration pairing. No p-values or binary equivalence tests are used, so no multiple-testing adjustment is invoked. No ratio is interpreted when the joint effect is small.

Technical cases were selected before new effects as the lexicographically first registered state in each task with its -1 cm donor: `{selected}`. Four workers must each pass their assigned technical cases. Donor predictions are never executed in the simulator.

## Frozen execution count

- Reusable native mechanism results: 800 cell-results (200 donors x 4 cells).
- Missing K=5 mechanism results: 800 cell-results.
- Minimal unique new action forwards: 900 = 50 recipient source captures + 200 donor source captures + 50 shared A00 + 600 donor-specific A10/A01/A11.
- No closed-loop rollout is added.
"""
    protocol_path = OUT / "mechanism_protocol.md"; protocol_path.write_text(protocol)
    audit_manifest = {
        "status": "JOINT_COMPUTE_MECHANISM_ASSET_AUDIT_AND_PROTOCOL_FROZEN",
        "created_unix": time.time(), "states": 50, "valid_donors": 200,
        "asset_rows": 1600, "native_reusable_cell_results": 800,
        "k5_missing_cell_results": 800, "minimal_unique_new_forwards": 900,
        "technical_case_ids": selected, "protocol_sha256": sha(protocol_path),
        "coverage_sha256": sha(coverage_path), "same_state_evidence_sha256": sha(evidence_path),
        "forward_registry_sha256": sha(forward_path), "source_registry_sha256": sha(SOURCE / "joint_new_trajectory_registry.csv"),
        "donor_registry_sha256": sha(DONOR), "causal_assets_sha256": sha(CAUSAL),
        "damage_labels_sha256": sha(LABEL), "closedloop_outcomes_sha256": sha(EPISODE),
        "posthoc_after_C_results": True, "code_sha256": sha(Path(__file__)),
    }
    dump(OUT / "asset_audit_and_protocol_manifest.json", audit_manifest)
    print(json.dumps(audit_manifest))


def record_attempt1() -> None:
    original = json.loads((OUT / "asset_audit_and_protocol_manifest.json").read_text())
    incident = {
        "status": "JOINT_COMPUTE_MECHANISM_TECHNICAL_ATTEMPT1_BLOCKED_COUNTING_ASSERTION",
        "created_unix": time.time(), "formal_science_started": False, "workers_failed": [0, 1, 2, 3],
        "original_code_sha256": original["code_sha256"], "protocol_sha256": original["protocol_sha256"],
        "observed_common_values": {
            "world_video_calls": 150, "action_calls": 300, "injected_video_calls": 150,
            "current_component_exact": 300, "future_component_exact": 300,
            "tail_current_exact": 300, "tail_future_exact": 300, "max_source_error": 0.0,
        },
        "root_cause": "A.VideoKVController.summary(expected_steps=5) assumes action calls equal world calls; the K5 operator correctly keeps 10 action steps, so action calls are 10x30=300 while world calls are 5x30=150.",
        "scientific_interface_changed": False, "state_or_donor_changed": False,
        "K_or_schedule_changed": False, "tolerance_changed": False,
    }
    dump(OUT / "technical_gate_attempt1_blocked.json", incident)
    amendment = {
        "status": "JOINT_COMPUTE_MECHANISM_TECHNICAL_COUNTING_AMENDMENT_FROZEN",
        "created_unix": time.time(), "parent_protocol_sha256": original["protocol_sha256"],
        "parent_code_sha256": original["code_sha256"], "attempt1_sha256": sha(OUT / "technical_gate_attempt1_blocked.json"),
        "change": "For K5 strict execution, require video_calls=5x30 and action_calls=10x30 explicitly; do not use the generic coupled action-site flag.",
        "unchanged": ["states", "donors", "K", "consumer edge", "cache construction", "randomness", "readouts", "statistics", "tolerances"],
        "new_code_sha256": sha(Path(__file__)), "formal_science_started": False,
    }
    dump(OUT / "mechanism_protocol_technical_counting_amendment.json", amendment)
    print(json.dumps(amendment))


def record_attempt2() -> None:
    original = json.loads((OUT / "asset_audit_and_protocol_manifest.json").read_text())
    incident = {
        "status": "JOINT_COMPUTE_MECHANISM_TECHNICAL_ATTEMPT2_BLOCKED_REMAINING_ASSERTION",
        "created_unix": time.time(), "formal_science_started": False, "workers_failed": [0, 1, 2, 3],
        "parent_protocol_sha256": original["protocol_sha256"],
        "attempt1_sha256": sha(OUT / "technical_gate_attempt1_blocked.json"),
        "amendment1_sha256": sha(OUT / "mechanism_protocol_technical_counting_amendment.json"),
        "attempt2_code_sha256": "1f69fe10a9f83a609bd1efe12dea1472c2cbd97d6ee8f567e0087a243c084493",
        "observed_common_values": {
            "world_video_calls": 150, "action_calls": 300, "injected_video_calls": 150,
            "current_component_exact": 300, "future_component_exact": 300,
            "tail_current_exact": 300, "tail_future_exact": 300, "max_source_error": 0.0,
            "all_consumed_sources_exact": True,
        },
        "root_cause": "The internal exactness predicate was corrected, but the terminal assertion retained the old hard-coded action_calls==150 check. K5 correctly executes 10x30=300 action consumer calls.",
        "scientific_interface_changed": False, "state_or_donor_changed": False,
        "K_or_schedule_changed": False, "tolerance_changed": False,
    }
    dump(OUT / "technical_gate_attempt2_blocked.json", incident)
    amendment = {
        "status": "JOINT_COMPUTE_MECHANISM_TECHNICAL_COUNTING_AMENDMENT_V2_FROZEN",
        "created_unix": time.time(), "parent_protocol_sha256": original["protocol_sha256"],
        "attempt2_sha256": sha(OUT / "technical_gate_attempt2_blocked.json"),
        "change": "Correct only the remaining terminal K5 assertion to require video_calls=150 and action_calls=300.",
        "statistics_implementation": "Vectorize the same frozen 10000-resample task-then-trajectory bootstrap without changing hierarchy, seed, estimates, or quantiles.",
        "unchanged": ["states", "donors", "K", "consumer edge", "cache construction", "randomness", "readouts", "bootstrap resamples", "bootstrap hierarchy", "tolerances"],
        "new_code_sha256": sha(Path(__file__)), "formal_science_started": False,
    }
    dump(OUT / "mechanism_protocol_technical_counting_amendment_v2.json", amendment)
    print(json.dumps(amendment))


def check_row(case: str, task: int, worker: int, name: str, passed: bool, error: float = 0.0,
              detail: str = "") -> dict[str, Any]:
    return {"case_id": case, "task_id": task, "worker_shard": worker, "check": name,
            "pass": bool(passed), "max_abs_error": float(error), "detail": detail}


def technical_worker(gpu: int, shard: int) -> None:
    audit_manifest = json.loads((OUT / "asset_audit_and_protocol_manifest.json").read_text())
    selected = audit_manifest["technical_case_ids"]
    own = [case for index, case in enumerate(selected) if index % 4 == shard]
    states = {str(row["candidate_id"]): row for row in state_rows()}; donors = donor_rows().set_index("case_id")
    labels = pd.read_csv(LABEL).query("K == 5").set_index("case_id")
    runner = make_capture("joint", gpu); before = A.model_weight_hash(runner.model)
    rows = []; costs = []
    for case in own:
        donor = donors.loc[case]; state = states[str(donor.candidate_id)]; seed = source_seed(state)
        recipient = prepared(runner, str(state["observation_path"]), str(state["instruction"]))
        donor_item = prepared(runner, str(donor.donor_observation_path), str(state["instruction"]))
        old = np.load(Path(str(labels.loc[case, "actions_path"])))
        rec1, rec_schedule1, rec_diag1 = capture_k5_source(runner, recipient, seed)
        rec2, _, rec_diag2 = capture_k5_source(runner, recipient, seed)
        donor_action, donor_schedule, donor_diag = capture_k5_source(runner, donor_item, seed)
        a00, a00_diag = strict_k5(runner, recipient, seed, rec_schedule1, rec_schedule1)
        a11, a11_diag = strict_k5(runner, recipient, seed, donor_schedule, donor_schedule)
        rec3, rec_schedule3, _ = capture_k5_source(runner, recipient, seed)
        context_same = A.tensor_sha256(recipient["context"]) == A.tensor_sha256(donor_item["context"])
        context_mask_same = A.tensor_sha256(recipient["context_mask"]) == A.tensor_sha256(donor_item["context_mask"])
        comparisons = [
            ("K5_NATIVE_REPEAT", rec1, rec2),
            ("K5_NEW_ENTRY_REPRODUCES_FROZEN_RECIPIENT", rec1, torch.from_numpy(old["recipient_K5"])),
            ("K5_NEW_ENTRY_REPRODUCES_FROZEN_DONOR", donor_action, torch.from_numpy(old["donor_K5"])),
            ("K5_SAME_VALUE_A00", a00, rec1),
            ("K5_SAME_VALUE_A11_WITH_SAME_Z", a11, donor_action),
            ("K5_RECIPIENT_AFTER_DONOR_NO_CROSS_CALL_POLLUTION", rec3, rec1),
        ]
        for name, left, right in comparisons:
            rows.append(check_row(case, int(donor.task_id), shard, name, torch.equal(left, right), A.max_abs(left, right)))
        rows += [
            check_row(case, int(donor.task_id), shard, "K5_RECIPIENT_CACHE_REPEAT", schedule_hash(rec_schedule1) == schedule_hash(rec_schedule3), 0.0),
            check_row(case, int(donor.task_id), shard, "K5_A00_ALL_CONSUMED_SOURCES_EXACT", a00_diag["all_consumed_sources_exact"], max(a00_diag["tail_max_abs_error"], a00_diag["first_five_hook"]["max_current_abs_error"], a00_diag["first_five_hook"]["max_future_abs_error"])),
            check_row(case, int(donor.task_id), shard, "K5_A11_ALL_CONSUMED_SOURCES_EXACT", a11_diag["all_consumed_sources_exact"], max(a11_diag["tail_max_abs_error"], a11_diag["first_five_hook"]["max_current_abs_error"], a11_diag["first_five_hook"]["max_future_abs_error"])),
            check_row(case, int(donor.task_id), shard, "K5_WORLD_ACTION_COUNTS_5_10", all(x["world_steps"] == 5 and x["action_steps"] == 10 for x in (rec_diag1, rec_diag2, donor_diag, a00_diag, a11_diag))),
            check_row(case, int(donor.task_id), shard, "F3G_RECIPIENT_DONOR_Z_SAME_VALUE", context_same and context_mask_same),
            check_row(case, int(donor.task_id), shard, "DONOR_OBSERVATION_HASH_MATCH", sha(Path(str(donor.donor_observation_path))) == str(donor.donor_observation_sha256)),
            check_row(case, int(donor.task_id), shard, "DONOR_ACTION_NOT_EXECUTED", True, detail="offline model forward only"),
        ]
        costs.append({"stage": "TECHNICAL", "worker_shard": shard, "case_id": case,
                      "source_capture_seconds": rec_diag1["latency_seconds"] + rec_diag2["latency_seconds"] + donor_diag["latency_seconds"],
                      "strict_cell_seconds": a00_diag["latency_seconds"] + a11_diag["latency_seconds"],
                      "new_action_forwards": 5})
        del recipient, donor_item, rec_schedule1, rec_schedule3, donor_schedule
        gc.collect(); torch.cuda.empty_cache()
    after = A.model_weight_hash(runner.model)
    rows.append(check_row("GLOBAL", -1, shard, "MODEL_WEIGHT_HASH_UNCHANGED", before == after))
    write(TECH / f"shard_{shard:02d}_checks.csv", rows); write(TECH / f"shard_{shard:02d}_cost.csv", costs)
    dump(TECH / f"shard_{shard:02d}.json", {
        "status": "COMPLETE" if all(row["pass"] for row in rows) else "BLOCKED",
        "shard": shard, "gpu_visible_index": gpu, "cases": len(own), "checks": len(rows),
        "weight_hash_before": before, "weight_hash_after": after,
        "checks_sha256": sha(TECH / f"shard_{shard:02d}_checks.csv"), "code_sha256": sha(Path(__file__)),
    })
    print(json.dumps({"shard": shard, "cases": len(own), "pass": all(row["pass"] for row in rows)}))


def technical_finalize() -> None:
    manifests = [TECH / f"shard_{shard:02d}.json" for shard in range(4)]
    if not all(path.is_file() for path in manifests):
        raise RuntimeError("four worker technical manifests required")
    rows = pd.concat([pd.read_csv(TECH / f"shard_{shard:02d}_checks.csv") for shard in range(4)], ignore_index=True)
    native_manifest = json.loads((SOURCE / "new_trajectory_causal_features_pre_label/joint_new_causal_feature_manifest.json").read_text())
    reused = [
        {"case_id": "REUSED_NATIVE_GATE", "task_id": -1, "worker_shard": -1,
         "check": "NATIVE_STRICT_ALL_200_HOOKS_EXACT_REUSED", "pass": bool(native_manifest["all_hooks_exact"]),
         "max_abs_error": 0.0, "detail": f"source={sha(SOURCE / 'new_trajectory_causal_features_pre_label/joint_new_causal_feature_manifest.json')}"},
        {"case_id": "REUSED_NATIVE_GATE", "task_id": -1, "worker_shard": -1,
         "check": "NATIVE_WORLD_ACTION_COUNTS_10_10_REUSED", "pass": True, "max_abs_error": 0.0,
         "detail": "200 frozen rows all K10_world_steps=10,K10_action_steps=10"},
    ]
    causal = pd.read_csv(CAUSAL)
    reused[1]["pass"] = bool(((causal.K10_world_steps == 10) & (causal.K10_action_steps == 10)).all())
    all_rows = pd.concat([rows, pd.DataFrame(reused)], ignore_index=True)
    path = OUT / "mechanism_technical_checks.csv"; all_rows.to_csv(path, index=False)
    passed = bool(all_rows["pass"].astype(str).str.lower().eq("true").all())
    report = {"status": "JOINT_COMPUTE_MECHANISM_TECHNICAL_PASS" if passed else "JOINT_COMPUTE_MECHANISM_TECHNICAL_BLOCKED",
              "created_unix": time.time(), "pass": passed, "workers": 4,
              "technical_cases": int(rows[rows.case_id != "GLOBAL"].case_id.nunique()),
              "checks": int(len(all_rows)), "max_abs_error": float(all_rows.max_abs_error.fillna(0).max()),
              "checks_sha256": sha(path), "protocol_sha256": sha(OUT / "mechanism_protocol.md"),
              "code_sha256": sha(Path(__file__))}
    dump(OUT / "mechanism_technical_gate.json", report); print(json.dumps(report))


def formal_worker(gpu: int, shard: int) -> None:
    gate = json.loads((OUT / "mechanism_technical_gate.json").read_text())
    if not gate["pass"]:
        raise RuntimeError("mechanism technical gate did not pass")
    states = state_rows(); own = [row for index, row in enumerate(states) if index % 4 == shard]
    donors = donor_rows(); by = {str(key): group.to_dict("records") for key, group in donors.groupby("candidate_id")}
    labels = pd.read_csv(LABEL).query("K == 5").set_index("case_id")
    runner = make_capture("joint", gpu); before = A.model_weight_hash(runner.model)
    rows = []; costs = []
    for index, state in enumerate(own, 1):
        cid = str(state["candidate_id"]); seed = source_seed(state)
        recipient = prepared(runner, str(state["observation_path"]), str(state["instruction"]))
        rec_action, rec_schedule, rec_diag = capture_k5_source(runner, recipient, seed)
        first_case = str(by[cid][0]["case_id"]); old_rec = np.load(Path(str(labels.loc[first_case, "actions_path"])))
        if not torch.equal(rec_action, torch.from_numpy(old_rec["recipient_K5"])):
            raise AssertionError(f"K5 recipient source mismatch {cid}")
        a00, a00_diag = strict_k5(runner, recipient, seed, rec_schedule, rec_schedule)
        if not torch.equal(a00, rec_action):
            raise AssertionError(f"K5 A00 identity mismatch {cid}")
        costs.append({"stage": "FORMAL", "worker_shard": shard, "candidate_id": cid, "case_id": "SHARED_A00",
                      "source_capture_seconds": rec_diag["latency_seconds"], "strict_cell_seconds": a00_diag["latency_seconds"],
                      "new_action_forwards": 2})
        for donor in by[cid]:
            case = str(donor["case_id"]); donor_item = prepared(runner, str(donor["donor_observation_path"]), str(state["instruction"]))
            donor_action, donor_schedule, donor_diag = capture_k5_source(runner, donor_item, seed)
            old = np.load(Path(str(labels.loc[case, "actions_path"])))
            if not torch.equal(donor_action, torch.from_numpy(old["donor_K5"])):
                raise AssertionError(f"K5 donor source mismatch {case}")
            a10, d10 = strict_k5(runner, recipient, seed, donor_schedule, rec_schedule)
            a01, d01 = strict_k5(runner, recipient, seed, rec_schedule, donor_schedule)
            a11, d11 = strict_k5(runner, recipient, seed, donor_schedule, donor_schedule)
            case_dir = ACTION_ROOT / case; case_dir.mkdir(parents=True, exist_ok=True)
            action_path = case_dir / "K5_strict_four_cells.npz"
            np.savez_compressed(action_path, recipient_K5=rec_action.numpy(), donor_K5=donor_action.numpy(),
                                A00=a00.numpy(), A10=a10.numpy(), A01=a01.numpy(), A11=a11.numpy())
            rows.append({
                "candidate_id": cid, "case_id": case, "task_id": int(state["task_id"]),
                "trajectory_id": int(state["trajectory_id"]), "signed_dose_cm": float(donor["signed_dose_cm"]),
                "configuration": "K5", "action_path": str(action_path), "action_sha256": sha(action_path),
                "recipient_source_matches_frozen_K5": True, "donor_source_matches_frozen_K5": True,
                "A00_same_value_bit_exact": torch.equal(a00, rec_action),
                "A11_same_value_bit_exact": torch.equal(a11, donor_action),
                "strict_all_consumed_sources_exact": all(x["all_consumed_sources_exact"] for x in (a00_diag, d10, d01, d11)),
                "world_steps": 5, "action_steps": 10, "donor_prediction_action_executed": False,
            })
            costs.append({"stage": "FORMAL", "worker_shard": shard, "candidate_id": cid, "case_id": case,
                          "source_capture_seconds": donor_diag["latency_seconds"],
                          "strict_cell_seconds": d10["latency_seconds"] + d01["latency_seconds"] + d11["latency_seconds"],
                          "new_action_forwards": 4})
            del donor_item, donor_schedule
            gc.collect(); torch.cuda.empty_cache()
        print(json.dumps({"stage": "K5_MECHANISM", "shard": shard, "state": index, "total": len(own), "cases": len(rows)}), flush=True)
        del recipient, rec_schedule
        gc.collect(); torch.cuda.empty_cache()
    after = A.model_weight_hash(runner.model)
    write(FORMAL / f"shard_{shard:02d}.csv", rows); write(FORMAL / f"shard_{shard:02d}_cost.csv", costs)
    dump(FORMAL / f"shard_{shard:02d}.json", {
        "status": "COMPLETE", "shard": shard, "states": len(own), "cases": len(rows),
        "actual_new_action_forwards": int(sum(row["new_action_forwards"] for row in costs)),
        "all_sources_exact": all(row["strict_all_consumed_sources_exact"] for row in rows),
        "weight_hash_before": before, "weight_hash_after": after,
        "output_sha256": sha(FORMAL / f"shard_{shard:02d}.csv"), "code_sha256": sha(Path(__file__)),
    })


def task_trajectory_ci(frame: pd.DataFrame, value: str, seed_offset: int) -> tuple[float, float, float]:
    trajectory = frame.groupby(["task_id", "trajectory_id"])[value].mean().reset_index()
    tasks = sorted(trajectory.task_id.unique()); arrays = {t: trajectory.loc[trajectory.task_id == t, value].to_numpy(float) for t in tasks}
    point = float(trajectory[value].mean()); rng = np.random.default_rng(BOOT_SEED + seed_offset)
    lengths = {len(values) for values in arrays.values()}
    if len(lengths) == 1:
        per_task = lengths.pop(); matrix = np.stack([arrays[t] for t in tasks])
        selected_tasks = rng.integers(0, len(tasks), size=(NBOOT, len(tasks)))
        selected_trajectories = rng.integers(0, per_task, size=(NBOOT, len(tasks), per_task))
        draws = matrix[selected_tasks[:, :, None], selected_trajectories].mean(axis=(1, 2))
    else:
        draws = np.empty(NBOOT)
        for draw in range(NBOOT):
            selected = []
            for task in rng.choice(tasks, len(tasks), replace=True):
                values = arrays[task]; selected.append(rng.choice(values, len(values), replace=True))
            draws[draw] = np.concatenate(selected).mean()
    return point, float(np.quantile(draws, .025)), float(np.quantile(draws, .975))


def formal_finalize() -> None:
    manifests = [FORMAL / f"shard_{shard:02d}.json" for shard in range(4)]
    if not all(path.is_file() for path in manifests):
        raise RuntimeError("four formal worker manifests required")
    meta = [json.loads(path.read_text()) for path in manifests]
    if not all(row["status"] == "COMPLETE" and row["weight_hash_before"] == row["weight_hash_after"] for row in meta):
        raise RuntimeError("formal worker incomplete or weight hash changed")
    k5 = pd.concat([pd.read_csv(FORMAL / f"shard_{shard:02d}.csv") for shard in range(4)], ignore_index=True)
    if len(k5) != 200 or k5.candidate_id.nunique() != 50 or not k5.strict_all_consumed_sources_exact.all():
        raise RuntimeError("K5 formal mechanism coverage incomplete")
    causal = pd.read_csv(CAUSAL); donors = donor_rows().set_index("case_id"); dn = D.denormalizer(); axes = NF.axes()
    cell_rows = []; effect_rows = []
    sources = [("native", causal.set_index("case_id")), ("K5", k5.set_index("case_id"))]
    for config, table in sources:
        for case, record in table.iterrows():
            donor = donors.loc[case]; cid = str(donor.candidate_id); dose = float(donor.signed_dose_cm)
            path = Path(str(record.actions_path if config == "native" else record.action_path))
            archive = np.load(path); actions = {cell: dn(archive[cell]) for cell in ("A00", "A10", "A01", "A11")}
            for cell, action in actions.items():
                delta = action - actions["A00"]; metrics = components(delta, axes[cid], cell, dose)
                cell_rows.append({"candidate_id": cid, "case_id": case, "task_id": int(donor.task_id),
                                  "trajectory_id": int(donor.trajectory_id), "signed_dose_cm": dose,
                                  "configuration": config, "intervention": cell, **metrics,
                                  "action_path": str(path), "action_file_sha256": sha(path),
                                  "forward_source": "REUSED_FROZEN_NATIVE" if config == "native" else "NEW_POSTHOC_K5",
                                  "predicted_action_not_executed": True})
            effect_rows.append({"candidate_id": cid, "case_id": case, "task_id": int(donor.task_id),
                                "trajectory_id": int(donor.trajectory_id), "signed_dose_cm": dose,
                                "configuration": config, **effect_record(actions, axes[cid], dose),
                                "action_path": str(path), "predicted_action_not_executed": True})
    cells = pd.DataFrame(cell_rows).sort_values(["task_id", "trajectory_id", "case_id", "configuration", "intervention"])
    effects = pd.DataFrame(effect_rows).sort_values(["task_id", "trajectory_id", "case_id", "configuration"])
    cells_path = OUT / "native_k5_four_cell_results.csv"; effects_path = OUT / "native_k5_conditional_effects.csv"
    cells.to_csv(cells_path, index=False); effects.to_csv(effects_path, index=False)

    endpoints = ("future_given_current_recipient", "future_given_current_donor",
                 "current_given_future_recipient", "current_given_future_donor", "joint", "interaction")
    metrics = ("radial_mean", "dose_aligned_radial_mean", "radial_l2", "translation_l2", "rotation_l2", "gripper_l2")
    stats = []
    for endpoint in endpoints:
        for metric in metrics:
            column = f"{endpoint}_{metric}"
            for dose_scope in ("ALL_EQUAL_DONOR", -1.0, -0.5, 0.5, 1.0):
                part = effects if dose_scope == "ALL_EQUAL_DONOR" else effects[effects.signed_dose_cm == dose_scope]
                wide = part.pivot(index=["case_id", "candidate_id", "task_id", "trajectory_id", "signed_dose_cm"],
                                  columns="configuration", values=column).reset_index().dropna()
                for config in ("native", "K5"):
                    point, low, high = task_trajectory_ci(wide.rename(columns={config: "value"}), "value",
                                                         sum(map(ord, endpoint + metric + config + str(dose_scope))))
                    stats.append({"endpoint": endpoint, "metric": metric, "dose_scope": dose_scope,
                                  "comparison": config, "estimate": point, "ci_low": low, "ci_high": high,
                                  "states": int(wide.candidate_id.nunique()), "cases": int(len(wide)),
                                  "bootstrap_resamples": NBOOT, "bootstrap_hierarchy": "task then source trajectory"})
                wide["difference"] = wide.K5 - wide.native
                point, low, high = task_trajectory_ci(wide, "difference", sum(map(ord, endpoint + metric + str(dose_scope))))
                stats.append({"endpoint": endpoint, "metric": metric, "dose_scope": dose_scope,
                              "comparison": "K5_MINUS_NATIVE", "estimate": point, "ci_low": low, "ci_high": high,
                              "states": int(wide.candidate_id.nunique()), "cases": int(len(wide)),
                              "bootstrap_resamples": NBOOT, "bootstrap_hierarchy": "task then source trajectory"})
    stats_frame = pd.DataFrame(stats)
    stats_path = OUT / "native_k5_paired_statistics.csv"; stats_frame.to_csv(stats_path, index=False)

    costs = pd.concat([pd.read_csv(TECH / f"shard_{s:02d}_cost.csv") for s in range(4)] +
                      [pd.read_csv(FORMAL / f"shard_{s:02d}_cost.csv") for s in range(4)], ignore_index=True)
    costs_path = OUT / "diagnostic_compute_cost.csv"; costs.to_csv(costs_path, index=False)
    existing = pd.read_csv(OUT / "same_state_evidence.csv")
    state_evidence = existing.groupby(["candidate_id", "task_id", "trajectory_id", "configuration"]).agg(
        native_UF_radial_l2=("native_UF_radial_l2", "mean"),
        response_damage_radial_l2=("response_damage_radial_l2", "mean"),
        cumulative_fraction_saved=("paired_cumulative_fraction_saved", "first"),
        per_call_fraction_saved=("paired_per_call_fraction_saved", "first"),
        native_success=("closedloop_native_success", "first"), simplified_success=("closedloop_simplified_success", "first"),
        outcome_change=("closedloop_outcome_change", "first")).reset_index()
    state_mech = effects.groupby(["candidate_id", "task_id", "trajectory_id", "configuration"]).agg(
        future_Crec_radial_l2=("future_given_current_recipient_radial_l2", "mean"),
        future_Cdonor_radial_l2=("future_given_current_donor_radial_l2", "mean"),
        future_Crec_aligned_mean=("future_given_current_recipient_dose_aligned_radial_mean", "mean"),
        future_Cdonor_aligned_mean=("future_given_current_donor_dose_aligned_radial_mean", "mean")).reset_index()
    state_mech.to_csv(OUT / "same_state_mechanism_summary.csv", index=False)

    figures = OUT / "figures"; figures.mkdir(exist_ok=True)
    fig, axes_plot = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    for ax, config in zip(axes_plot, ("K5", "K8")):
        part = state_evidence[state_evidence.configuration == config]
        for task, group in part.groupby("task_id"):
            ax.scatter(group.native_UF_radial_l2, group.response_damage_radial_l2, label=f"task {task}", s=28, alpha=.8)
        changed = part[part.outcome_change != "SAME"]
        ax.scatter(changed.native_UF_radial_l2, changed.response_damage_radial_l2,
                   facecolors="none", edgecolors="black", linewidths=1.8, s=90)
        ax.set(xlabel="native UF radial L2", ylabel=f"{config} response-damage radial L2",
               title=f"{config}: post-hoc same-state association")
    axes_plot[0].legend(fontsize=7, ncol=2)
    fig.savefig(figures / "native_future_effect_vs_response_damage.png", dpi=180); plt.close(fig)

    fig, axes_plot = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    for ax, config in zip(axes_plot, ("K5", "K8")):
        part = state_evidence[state_evidence.configuration == config]
        for task, group in part.groupby("task_id"):
            ax.scatter(group.response_damage_radial_l2, group.cumulative_fraction_saved, label=f"task {task}", s=28, alpha=.8)
        changed = part[part.outcome_change != "SAME"]
        ax.scatter(changed.response_damage_radial_l2, changed.cumulative_fraction_saved,
                   facecolors="none", edgecolors="black", linewidths=1.8, s=90)
        ax.set(xlabel=f"{config} response-damage radial L2", ylabel="paired cumulative compute fraction saved",
               title=f"{config}: damage vs compute saving")
    axes_plot[0].legend(fontsize=7, ncol=2)
    fig.savefig(figures / "response_damage_vs_cumulative_saving.png", dpi=180); plt.close(fig)

    association = []
    for config, group in state_evidence.groupby("configuration"):
        for x, y in (("native_UF_radial_l2", "response_damage_radial_l2"),
                     ("response_damage_radial_l2", "cumulative_fraction_saved")):
            association.append({"configuration": config, "x": x, "y": y,
                                "pearson_r": float(group[x].corr(group[y], method="pearson")),
                                "spearman_rho": float(group[x].corr(group[y], method="spearman")),
                                "states": len(group), "interpretation": "posthoc descriptive; not mechanism or risk prediction"})
    pd.DataFrame(association).to_csv(OUT / "same_state_exploratory_associations.csv", index=False)

    formal_manifest = {
        "status": "JOINT_K5_MECHANISM_GRID_COMPLETE", "created_unix": time.time(),
        "states": 50, "donors": 200, "native_reused_cell_results": 800, "K5_cell_results": 800,
        "actual_formal_new_action_forwards": int(sum(row["actual_new_action_forwards"] for row in meta)),
        "technical_new_action_forwards": int(costs[costs.stage == "TECHNICAL"].new_action_forwards.sum()),
        "all_weights_unchanged": True, "all_sources_exact": True, "donor_action_executed": False,
        "four_cell_results_sha256": sha(cells_path), "conditional_effects_sha256": sha(effects_path),
        "paired_statistics_sha256": sha(stats_path), "compute_cost_sha256": sha(costs_path),
        "protocol_sha256": sha(OUT / "mechanism_protocol.md"), "code_sha256": sha(Path(__file__)),
    }
    dump(OUT / "mechanism_formal_manifest.json", formal_manifest)
    print(json.dumps(formal_manifest))


def report() -> None:
    manifest = json.loads((OUT / "mechanism_formal_manifest.json").read_text())
    stats = pd.read_csv(OUT / "native_k5_paired_statistics.csv")
    effects = pd.read_csv(OUT / "native_k5_conditional_effects.csv")
    evidence = pd.read_csv(OUT / "same_state_evidence.csv")
    associations = pd.read_csv(OUT / "same_state_exploratory_associations.csv")
    episodes = pd.read_csv(EPISODE)
    cost = pd.read_csv(OUT / "diagnostic_compute_cost.csv")

    def stat(endpoint: str, metric: str, comparison: str):
        return stats[(stats.endpoint == endpoint) & (stats.metric == metric) &
                     (stats.dose_scope.astype(str) == "ALL_EQUAL_DONOR") & (stats.comparison == comparison)].iloc[0]

    fcr_n = stat("future_given_current_recipient", "radial_l2", "native")
    fcr_k = stat("future_given_current_recipient", "radial_l2", "K5")
    fcr_d = stat("future_given_current_recipient", "radial_l2", "K5_MINUS_NATIVE")
    fcd_n = stat("future_given_current_donor", "radial_l2", "native")
    fcd_k = stat("future_given_current_donor", "radial_l2", "K5")
    fcd_d = stat("future_given_current_donor", "radial_l2", "K5_MINUS_NATIVE")
    aligned_k1 = stat("future_given_current_recipient", "dose_aligned_radial_mean", "K5")
    aligned_k2 = stat("future_given_current_donor", "dose_aligned_radial_mean", "K5")
    k5_ep = episodes[episodes.configuration == "K5"]; native_ep = episodes[episodes.configuration == "native"]
    k5_saving = float(evidence[evidence.configuration == "K5"].drop_duplicates("candidate_id").paired_cumulative_fraction_saved.median())
    assoc_k5 = associations[(associations.configuration == "K5") & (associations.x == "native_UF_radial_l2")].iloc[0]
    task_table = effects.groupby(["task_id", "configuration"]).agg(
        future_Crec_radial_l2=("future_given_current_recipient_radial_l2", "mean"),
        future_Cdonor_radial_l2=("future_given_current_donor_radial_l2", "mean"),
        joint_radial_l2=("joint_radial_l2", "mean"), interaction_radial_l2=("interaction_radial_l2", "mean")).reset_index()
    task_table_path = OUT / "native_k5_task_summary.csv"; task_table.to_csv(task_table_path, index=False)

    report_path = OUT / "report_joint_compute_mechanism.md"
    lines = [
        "# Joint-WAM 固定省算后的物理信息通路审计",
        "",
        "状态：`JOINT_COMPUTE_MECHANISM_POSTHOC_COMPLETE`",
        "",
        "本轮是 C 结果已知后的事后机制补充，不属于原确认性协议，也不改变原报告。donor 场景只用于离线生成缓存和预测动作，预测动作从未在 donor 模拟器中执行。",
        "",
        "## 1. A/B 机制证据与 C 是否在同一批状态对齐？",
        "",
        "是。原 50 个 restore_v2 起点、200 个有效 F3-G donor、native 严格四格、K=5/K=8 局部响应损伤和 150 条闭环 outcomes 已按 candidate/trajectory/case 精确连接。native 的 800 个四格 cell-result 全部按哈希复用；K=5 补齐 800 个 cell-result，实际正式新增 900 次动作前向。没有筛除任何起点或 donor。",
        "",
        f"同状态描述性关联中，native UF 径向 L2 与 K=5 响应损伤的 Pearson r={assoc_k5.pearson_r:.3f}、Spearman ρ={assoc_k5.spearman_rho:.3f}。这是探索性关联，不是机制保留判据或风险预测性能。",
        "",
        "## 2. K=5 下 future 条件效应",
        "",
        f"- future | current recipient：native 径向 L2 {fcr_n.estimate:.6f} [{fcr_n.ci_low:.6f}, {fcr_n.ci_high:.6f}]；K=5 {fcr_k.estimate:.6f} [{fcr_k.ci_low:.6f}, {fcr_k.ci_high:.6f}]；配对差 {fcr_d.estimate:+.6f} [{fcr_d.ci_low:+.6f}, {fcr_d.ci_high:+.6f}]。",
        f"- future | current donor：native 径向 L2 {fcd_n.estimate:.6f} [{fcd_n.ci_low:.6f}, {fcd_n.ci_high:.6f}]；K=5 {fcd_k.estimate:.6f} [{fcd_k.ci_low:.6f}, {fcd_k.ci_high:.6f}]；配对差 {fcd_d.estimate:+.6f} [{fcd_d.ci_low:+.6f}, {fcd_d.ci_high:+.6f}]。",
        f"- K=5 的 dose-aligned signed radial mean：current-recipient 条件 {aligned_k1.estimate:+.6f} [{aligned_k1.ci_low:+.6f}, {aligned_k1.ci_high:+.6f}]；current-donor 条件 {aligned_k2.estimate:+.6f} [{aligned_k2.ci_low:+.6f}, {aligned_k2.ci_high:+.6f}]。",
        "",
        "这些结果直接说明 K=5 自身计算路径产生的 future K/V 仍能在动作消费者处传递登记的 F3-G 目标位置变化。它不等于 future 对全部动作必需，也没有建立 K=5 与 native 的机制等效。各 signed dose、current 条件、任务及其他动作通道的原始结果均保存在统计表中。",
        "",
        "## 3. 哪些效应改变、哪些尚无法区分？",
        "",
        "K=5 与 native 的 future 条件效应、current 条件效应、联合效应和交互均已逐状态配对。置信区间描述的是本批任务→轨迹重采样不确定性；没有预注册等效容限，因此 CI 跨零只能表示未分辨配对变化，不能证明相同或等效。联合效应较小时没有计算来源保留百分比。",
        "",
        "## 4. 与闭环结果共同支持什么？",
        "",
        f"同一 50 个 recipient 起点中，native {int(native_ep.success.sum())}/50、K=5 {int(k5_ep.success.sum())}/50 成功；K=5 配对累计策略计算节省中位数 {100*k5_saving:.1f}%。因此可同时陈述：在这些登记起点上，K=5 路径仍传递 F3-G 目标位置信息、局部响应发生可测变化、recipient 闭环全部完成，并获得真实计算节省。",
        "",
        "不能由此声称整个闭环始终使用相同机制、donor 目标变化已经闭环验证、future 对所有动作必需、或省算机制无损。K=8 的唯一失败只链接原有视频与终止记录，本轮未对 K=8 补机制网格或寻找失败阈值。",
        "",
        "## 5. C 的科学定位",
        "",
        "C 仍应保留为效率应用实验。此次补充为 C 增加了同状态、严格消费边的事后机制解释：省算后 future 物理信息通路仍存在，但改变了部分局部响应。由于机制协议在 C 结果揭示后冻结，它不能追认为原 C 的确认性机制贡献。",
        "",
        "## 技术与成本",
        "",
        f"四个 worker 技术门通过，最大动作误差为 0；正式 900 次新前向，技术门 {int(cost[cost.stage=='TECHNICAL'].new_action_forwards.sum())} 次前向。模型权重哈希前后一致，世界/动作计数为 K=5 的 5/10；native 10/10 复用原已验证资产。",
        "",
        "## 主要文件",
        "",
        f"- 资产覆盖：`asset_coverage_and_reuse.csv`（SHA-256 `{sha(OUT / 'asset_coverage_and_reuse.csv')}`）",
        f"- 四格结果：`native_k5_four_cell_results.csv`（SHA-256 `{sha(OUT / 'native_k5_four_cell_results.csv')}`）",
        f"- 条件效应：`native_k5_conditional_effects.csv`（SHA-256 `{sha(OUT / 'native_k5_conditional_effects.csv')}`）",
        f"- 配对统计：`native_k5_paired_statistics.csv`（SHA-256 `{sha(OUT / 'native_k5_paired_statistics.csv')}`）",
        f"- 计算成本：`diagnostic_compute_cost.csv`（SHA-256 `{sha(OUT / 'diagnostic_compute_cost.csv')}`）",
        "",
        "图中所有动作均为预测动作、未执行；闭环信息来自既有 recipient 分支。",
    ]
    report_path.write_text("\n".join(lines) + "\n")
    final = {
        "status": "JOINT_COMPUTE_MECHANISM_POSTHOC_COMPLETE", "created_unix": time.time(),
        "states": 50, "donors": 200, "native_cell_results": 800, "K5_cell_results": 800,
        "formal_new_action_forwards": manifest["actual_formal_new_action_forwards"],
        "technical_gate_sha256": sha(OUT / "mechanism_technical_gate.json"),
        "protocol_sha256": sha(OUT / "mechanism_protocol.md"), "report_sha256": sha(report_path),
        "task_summary_sha256": sha(task_table_path), "posthoc_after_C_results": True,
        "historical_results_modified": False, "donor_action_executed": False,
        "no_K8_mechanism_forward": True, "no_equivalence_claim": True, "code_sha256": sha(Path(__file__)),
    }
    dump(OUT / "mechanism_final_manifest.json", final); print(json.dumps(final))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["audit", "record-attempt1", "record-attempt2", "technical-worker", "technical-finalize", "formal-worker", "formal-finalize", "report"])
    parser.add_argument("--gpu", type=int, default=0); parser.add_argument("--shard", type=int, default=0)
    args = parser.parse_args()
    if args.mode == "audit": audit()
    elif args.mode == "record-attempt1": record_attempt1()
    elif args.mode == "record-attempt2": record_attempt2()
    elif args.mode == "technical-worker": technical_worker(args.gpu, args.shard)
    elif args.mode == "technical-finalize": technical_finalize()
    elif args.mode == "formal-worker": formal_worker(args.gpu, args.shard)
    elif args.mode == "formal-finalize": formal_finalize()
    else: report()


if __name__ == "__main__":
    main()

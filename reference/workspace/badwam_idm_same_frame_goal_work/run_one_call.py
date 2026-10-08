#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from common import (
    ALT_INSTRUCTION,
    DONOR_TASK,
    TARGETS,
    action_from_latent,
    action_noise,
    body_name,
    create_policy,
    finite,
    fgl,
    generate_video_latent,
    hash_text,
    make_env,
    normalized_to_env,
    patch_future,
    per_dimension_change,
    prepare,
    seed_for,
    tensor_sha256,
    transfer_metrics,
    write_json,
)
from imagination_intervention import parameter_signature, tensor_summary


PRIMARY = ("clean", "future_only", "context_only", "both", "shuffled_future")
TEMPORAL = ("future_group_1", "future_group_2", "future_swap_content")


def safe_torch_save(path: Path, payload: Any) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def vector_metrics(lhs: torch.Tensor, rhs: torch.Tensor) -> dict[str, float]:
    a = lhs.detach().double().reshape(-1)
    b = rhs.detach().double().reshape(-1)
    delta = b - a
    an = float(torch.linalg.vector_norm(a))
    bn = float(torch.linalg.vector_norm(b))
    dn = float(torch.linalg.vector_norm(delta))
    dot = float(torch.dot(a, b))
    return {
        "source_l2": an,
        "goal_l2": bn,
        "difference_l2": dn,
        "difference_over_source_l2": dn / (an + 1e-12),
        "cosine": 0.0 if an == 0.0 or bn == 0.0 else dot / (an * bn),
    }


def cache_difference(
    source: list[dict[str, torch.Tensor]],
    goal: list[dict[str, torch.Tensor]],
    tokens_per_group: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for layer, (source_layer, goal_layer) in enumerate(zip(source, goal)):
        for kind in ("k", "v"):
            for group in range(3):
                sl = slice(group * tokens_per_group, (group + 1) * tokens_per_group)
                rows.append(
                    {
                        "layer": layer,
                        "component": kind,
                        "temporal_group": group,
                        **vector_metrics(source_layer[kind][:, sl], goal_layer[kind][:, sl]),
                    }
                )
    return rows


def latent_difference(source: torch.Tensor, goal: torch.Tensor) -> list[dict[str, Any]]:
    return [
        {"temporal_group": group, **vector_metrics(source[:, :, group], goal[:, :, group])}
        for group in range(int(source.shape[2]))
    ]


def run_action_chunk(task_id: int, state_index: int, action: torch.Tensor, processor: Any) -> dict[str, Any]:
    env, _, obs = make_env(task_id, state_index)
    source_body = body_name(env, TARGETS[task_id])
    alt_body = body_name(env, TARGETS[DONOR_TASK[task_id]])
    sim = fgl.get_sim(env)
    start_eef = fgl.get_eef_pos(env, obs)
    source_pos = fgl.body_pos(sim, source_body)
    alt_pos = fgl.body_pos(sim, alt_body)
    source_direction = source_pos - start_eef
    alt_direction = alt_pos - start_eef
    source_direction /= np.linalg.norm(source_direction) + 1e-12
    alt_direction /= np.linalg.norm(alt_direction) + 1e-12
    source_distance_before = float(np.linalg.norm(start_eef - source_pos))
    alt_distance_before = float(np.linalg.norm(start_eef - alt_pos))
    processed = normalized_to_env(action, processor)
    done = False
    steps = 0
    try:
        for row in processed:
            obs, _, done, _ = env.step(row.tolist())
            obs = dict(obs)
            steps += 1
            if done:
                break
        end_eef = fgl.get_eef_pos(env, obs)
        sim = fgl.get_sim(env)
        source_pos_after = fgl.body_pos(sim, source_body)
        alt_pos_after = fgl.body_pos(sim, alt_body)
    finally:
        env.close()
    eef_delta = end_eef - start_eef
    source_distance_after = float(np.linalg.norm(end_eef - source_pos_after))
    alt_distance_after = float(np.linalg.norm(end_eef - alt_pos_after))
    return {
        "executed_action_steps": steps,
        "done": bool(done),
        "eef_start": start_eef,
        "eef_end": end_eef,
        "eef_delta": eef_delta,
        "eef_delta_l2": float(np.linalg.norm(eef_delta)),
        "eef_projection_toward_source": float(np.dot(eef_delta, source_direction)),
        "eef_projection_toward_alternate": float(np.dot(eef_delta, alt_direction)),
        "source_target_distance_before": source_distance_before,
        "source_target_distance_after": source_distance_after,
        "source_target_distance_change": source_distance_after - source_distance_before,
        "source_target_progress": source_distance_before - source_distance_after,
        "alternate_target_distance_before": alt_distance_before,
        "alternate_target_distance_after": alt_distance_after,
        "alternate_target_distance_change": alt_distance_after - alt_distance_before,
        "alternate_target_progress": alt_distance_before - alt_distance_after,
    }


def run_pair(policy: Any, root: Path, task_id: int, pair_id: int, state_index: int) -> dict[str, Any]:
    result_path = root / f"results/per_pair/task_{task_id}/pair_{pair_id:02d}.json"
    if result_path.exists():
        return json.loads(result_path.read_text())

    env, task, obs = make_env(task_id, state_index)
    try:
        source_prepared = prepare(policy, obs, task.language)
        goal_instruction = ALT_INSTRUCTION[task_id]
        goal_prepared = prepare(policy, obs, goal_instruction)
    finally:
        env.close()

    # Same-state controls: only language differs between the two prepared inputs.
    if not torch.equal(source_prepared["image"], goal_prepared["image"]):
        raise AssertionError("same-state image tensor mismatch")
    if not torch.equal(source_prepared["proprio"], goal_prepared["proprio"]):
        raise AssertionError("same-state proprio tensor mismatch")

    seed = seed_for(task_id, pair_id)
    signature_before = parameter_signature(policy.model)
    source_video = generate_video_latent(policy, source_prepared, seed=seed)
    goal_video = generate_video_latent(policy, goal_prepared, seed=seed)
    if not torch.equal(source_video["initial_noise"], goal_video["initial_noise"]):
        raise AssertionError("video initial noise mismatch")
    if not torch.equal(source_video["first_frame_latent"], goal_video["first_frame_latent"]):
        raise AssertionError("encoded current RGB mismatch")
    if not torch.equal(source_video["latent"][:, :, 0], goal_video["latent"][:, :, 0]):
        raise AssertionError("current temporal latent group mismatch")

    source_latent = source_video["latent"]
    goal_latent = goal_video["latent"]
    source_noise = action_noise(policy, seed)
    repeated_noise = action_noise(policy, seed)
    if not torch.equal(source_noise, repeated_noise):
        raise AssertionError("action noise replay mismatch")

    source_result = action_from_latent(
        policy, source_prepared, source_latent, source_noise, capture_cache=True
    )
    goal_result = action_from_latent(
        policy, goal_prepared, goal_latent, repeated_noise, capture_cache=True
    )
    source_action = source_result["action"]
    goal_action = goal_result["action"]
    tokens_per_group = int(source_result["tokens_per_temporal_group"])

    identity_latent, identity_debug = patch_future(source_latent, source_latent, "clean", seed)
    identity_result = action_from_latent(
        policy, source_prepared, identity_latent, action_noise(policy, seed), capture_cache=False
    )
    if not torch.equal(source_action, identity_result["action"]):
        raise AssertionError("STOP: identity intervention is not bit exact")

    actions: dict[str, torch.Tensor] = {"clean": source_action, "both": goal_action}
    debug: dict[str, Any] = {"identity": identity_debug}
    condition_specs = {
        "future_only": (source_prepared, "future_only"),
        "context_only": (goal_prepared, "context_only"),
        "shuffled_future": (source_prepared, "shuffled_future"),
        "future_group_1": (source_prepared, "future_group_1"),
        "future_group_2": (source_prepared, "future_group_2"),
        "future_swap_content": (source_prepared, "future_swap_content"),
    }
    for condition, (prepared, patch_condition) in condition_specs.items():
        donor = goal_latent if patch_condition.startswith("future_group") or patch_condition == "future_only" else source_latent
        patched, patch_debug = patch_future(source_latent, donor, patch_condition, seed)
        if not torch.equal(patched[:, :, 0], source_latent[:, :, 0]):
            raise AssertionError(f"STOP: current latent changed under {condition}")
        if not finite(patched):
            raise AssertionError(f"STOP: non-finite latent under {condition}")
        action_result = action_from_latent(
            policy, prepared, patched, action_noise(policy, seed), capture_cache=False
        )
        if not finite(action_result["action"]):
            raise AssertionError(f"STOP: non-finite action under {condition}")
        actions[condition] = action_result["action"]
        debug[condition] = patch_debug

    # BOTH is the independently generated alternate-goal reference, not a copied
    # source action. Recompute once to make its deterministic replay explicit.
    both_replay = action_from_latent(
        policy, goal_prepared, goal_latent, action_noise(policy, seed), capture_cache=False
    )["action"]
    if not torch.equal(goal_action, both_replay):
        raise AssertionError("STOP: alternate-goal action replay is not bit exact")

    metrics: dict[str, Any] = {}
    spatial: dict[str, Any] = {}
    for condition in PRIMARY:
        metrics[condition] = {
            **transfer_metrics(source_action, goal_action, actions[condition]),
            **per_dimension_change(source_action, actions[condition]),
        }
        spatial[condition] = run_action_chunk(task_id, state_index, actions[condition], policy.processor)
    temporal_metrics = {
        condition: {
            **transfer_metrics(source_action, goal_action, actions[condition]),
            **per_dimension_change(source_action, actions[condition]),
        }
        for condition in TEMPORAL
    }

    additive_prediction = (
        source_action
        + (actions["future_only"] - source_action)
        + (actions["context_only"] - source_action)
    )
    additive_residual = goal_action - additive_prediction
    additive_scale = float(torch.linalg.vector_norm((goal_action - source_action).double()))
    additive = {
        "prediction_l2_from_both": float(torch.linalg.vector_norm(additive_residual.double())),
        "prediction_error_over_donor_delta": float(torch.linalg.vector_norm(additive_residual.double()))
        / (additive_scale + 1e-12),
        "future_delta_l2": float(torch.linalg.vector_norm((actions["future_only"] - source_action).double())),
        "context_delta_l2": float(torch.linalg.vector_norm((actions["context_only"] - source_action).double())),
        "both_delta_l2": additive_scale,
        "future_context_delta_cosine": vector_metrics(
            actions["future_only"] - source_action,
            actions["context_only"] - source_action,
        )["cosine"],
    }

    representation_dir = root / f"representations/task_{task_id}/pair_{pair_id:02d}"
    middle_layer = len(source_result["cache"]) // 2
    middle_start = tokens_per_group
    middle_payload = {
        "layer": middle_layer,
        "tokens_per_temporal_group": tokens_per_group,
        "source_future_k": source_result["cache"][middle_layer]["k"][:, middle_start:].clone(),
        "source_future_v": source_result["cache"][middle_layer]["v"][:, middle_start:].clone(),
        "goal_future_k": goal_result["cache"][middle_layer]["k"][:, middle_start:].clone(),
        "goal_future_v": goal_result["cache"][middle_layer]["v"][:, middle_start:].clone(),
    }
    safe_torch_save(
        representation_dir / "generated_future_latents.pt",
        {
            "source": source_latent.detach().cpu().to(torch.float16),
            "alternate_goal": goal_latent.detach().cpu().to(torch.float16),
            "source_summary": tensor_summary(source_latent),
            "alternate_goal_summary": tensor_summary(goal_latent),
        },
    )
    safe_torch_save(representation_dir / "middle_layer_future_kv.pt", middle_payload)

    actions_path = root / f"actions/task_{task_id}/pair_{pair_id:02d}.npz"
    if actions_path.exists():
        raise FileExistsError(f"Refusing to overwrite {actions_path}")
    actions_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(actions_path, **{name: value.numpy() for name, value in actions.items()})

    signature_after = parameter_signature(policy.model)
    if signature_before != signature_after:
        raise AssertionError("STOP: pretrained parameter signature changed")
    result = {
        "task": task_id,
        "pair_id": pair_id,
        "source_state_index": state_index,
        "source_instruction": task.language,
        "alternate_instruction": goal_instruction,
        "source_target": TARGETS[task_id],
        "alternate_target": TARGETS[DONOR_TASK[task_id]],
        "seed": seed,
        "same_state_controls": {
            "rgb_bit_exact": True,
            "proprio_bit_exact": True,
            "video_initial_noise_bit_exact": True,
            "action_initial_noise_bit_exact": True,
            "first_frame_latent_bit_exact": True,
            "current_temporal_group_bit_exact": True,
            "identity_action_bit_exact": True,
            "both_action_replay_bit_exact": True,
            "pretrained_parameter_signature_unchanged": True,
            "rgb_sha256": tensor_sha256(source_prepared["image"]),
            "proprio_sha256": tensor_sha256(source_prepared["proprio"]),
            "source_instruction_sha256": hash_text(task.language),
            "alternate_instruction_sha256": hash_text(goal_instruction),
        },
        "latent_difference_by_temporal_group": latent_difference(source_latent, goal_latent),
        "goal_conditioned_kv_difference_by_layer": cache_difference(
            source_result["cache"], goal_result["cache"], tokens_per_group
        ),
        "metrics": metrics,
        "spatial_metrics": spatial,
        "temporal_metrics": temporal_metrics,
        "causal_decomposition": additive,
        "debug": debug,
        "artifacts": {
            "actions": str(actions_path),
            "generated_future_latents": str(representation_dir / "generated_future_latents.pt"),
            "middle_layer_future_kv": str(representation_dir / "middle_layer_future_kv.pt"),
        },
    }
    write_json(result_path, result)
    print(json.dumps({"task": task_id, "pair": pair_id, "state": state_index, "status": "ok"}), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--screening-root", type=Path, required=True)
    parser.add_argument("--task", type=int, required=True)
    parser.add_argument("--gpu-id", type=int, default=0)
    args = parser.parse_args()
    selection_path = args.screening_root / f"task_{args.task}/selected_states.json"
    selection = json.loads(selection_path.read_text())
    if not selection.get("complete") or len(selection["selected_states"]) != 10:
        raise RuntimeError(f"Incomplete clean screening: {selection_path}")
    policy = create_policy(args.gpu_id)
    for pair_id, state_index in enumerate(selection["selected_states"]):
        run_pair(policy, args.root.resolve(), args.task, pair_id, int(state_index))


if __name__ == "__main__":
    main()

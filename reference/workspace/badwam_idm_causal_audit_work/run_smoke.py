#!/usr/bin/env python3
from __future__ import annotations
from wam_causal_audit.paths import resolve as _release_path

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import imageio.v2 as imageio
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2] if "experiments" in Path(__file__).parts else Path(_release_path('@DATA@/BadWAM'))
for extra in (
    REPO,
    REPO / "experiments/libero",
    REPO / "experiments/first_grasp_lock",
    Path(_release_path('@WORKSPACE@/LIBERO')),
    Path(_release_path('@WORKSPACE@/evaluation/d2_ghost_pilot')),
):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from libero.libero import benchmark, get_libero_path  # noqa: E402
from libero.libero.envs import OffScreenRenderEnv  # noqa: E402

import run_first_grasp_lock as fgl  # noqa: E402
from imagination_intervention import (  # noqa: E402
    ImaginationIntervention,
    load_representation,
    parameter_signature,
    save_representation,
    tensor_sha256,
    tensor_summary,
)


SOURCE_TASK = 0
SOURCE_STATE = 0
SOURCE_TARGET = "alphabet_soup_1"
DONOR_TARGET = "cream_cheese_1"
DONOR_INSTRUCTION = "pick up the cream cheese and place it in the basket"
SEED = 42


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_observation(task_id: int, state_index: int) -> tuple[Any, Any, dict[str, Any]]:
    suite = benchmark.get_benchmark_dict()["libero_object"]()
    task = suite.get_task(task_id)
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=bddl,
        camera_heights=fgl.LIBERO_ENV_RESOLUTION,
        camera_widths=fgl.LIBERO_ENV_RESOLUTION,
    )
    env.seed(fgl.ENVIRONMENT_SEED)
    env.reset()
    obs = dict(env.set_init_state(suite.get_task_init_states(task_id)[state_index]))
    for _ in range(30):
        obs, _, _, _ = env.step(fgl.get_libero_dummy_action())
        obs = dict(obs)
    return env, task, obs


def prepare(policy: fgl.BadWAMPolicy, obs: Mapping[str, Any], instruction: str) -> dict[str, Any]:
    image, proprio, raw_images = fgl._obs_to_model_input(
        dict(obs),
        cfg=policy.cfg,
        processor=policy.processor,
        width=policy.input_w,
        height=policy.input_h,
        device=str(policy.model.device),
        dtype=policy.model.torch_dtype,
    )
    prompt = fgl.DEFAULT_PROMPT.format(task=instruction)
    context, context_mask = policy.model.encode_prompt(prompt)
    context, context_mask = policy.model._append_proprio_to_context(
        context=context,
        context_mask=context_mask,
        proprio=proprio.to(policy.model.device, policy.model.torch_dtype),
    )
    return {
        "instruction": instruction,
        "prompt": prompt,
        "image": image,
        "proprio": proprio,
        "raw_images": raw_images,
        "context": context,
        "context_mask": context_mask,
    }


@torch.no_grad()
def generate_video_latent(
    policy: fgl.BadWAMPolicy,
    prepared: Mapping[str, Any],
    *,
    seed: int,
) -> dict[str, Any]:
    model = policy.model
    image = prepared["image"].to(model.device, model.torch_dtype)
    _, _, height, width = image.shape
    latent_t = (policy.num_video_frames - 1) // model.vae.temporal_downsample_factor + 1
    generator = torch.Generator(device=str(policy.cfg.EVALUATION.rand_device)).manual_seed(seed)
    initial_noise = torch.randn(
        (
            1,
            model.vae.model.z_dim,
            latent_t,
            height // model.vae.upsampling_factor,
            width // model.vae.upsampling_factor,
        ),
        generator=generator,
        device=str(policy.cfg.EVALUATION.rand_device),
        dtype=torch.float32,
    ).to(model.device, model.torch_dtype)
    first_frame = model._encode_input_image_latents_tensor(
        input_image=image, tiled=bool(policy.cfg.EVALUATION.tiled)
    )
    latents = initial_noise.clone()
    latents[:, :, 0:1] = first_frame.clone()
    timesteps, deltas = model.infer_video_scheduler.build_inference_schedule(
        policy.num_inference_steps,
        model.device,
        latents.dtype,
        shift_override=None,
    )
    fuse = bool(getattr(model.video_expert, "fuse_vae_embedding_in_latents", False))
    for step_t, delta in zip(timesteps, deltas):
        prediction = model.video_expert(
            x=latents,
            timestep=step_t.unsqueeze(0).to(latents),
            context=prepared["context"],
            context_mask=prepared["context_mask"],
            action=None,
            fuse_vae_embedding_in_latents=fuse,
        )
        latents = model.infer_video_scheduler.step(prediction, delta, latents)
        latents[:, :, 0:1] = first_frame.clone()
    return {
        "latent": latents,
        "initial_noise": initial_noise,
        "first_frame_latent": first_frame,
        "timesteps": timesteps,
        "deltas": deltas,
    }


def clone_cache_cpu(cache: Sequence[Mapping[str, torch.Tensor]]) -> list[dict[str, torch.Tensor]]:
    return [
        {kind: row[kind].detach().to(device="cpu", dtype=torch.float16) for kind in ("k", "v")}
        for row in cache
    ]


def cache_trace(cache: Sequence[Mapping[str, torch.Tensor]]) -> list[dict[str, Any]]:
    return [
        {
            "layer": layer,
            "k": tensor_summary(row["k"]),
            "v": tensor_summary(row["v"]),
        }
        for layer, row in enumerate(cache)
    ]


@torch.no_grad()
def action_from_latent(
    policy: fgl.BadWAMPolicy,
    prepared: Mapping[str, Any],
    future_latent: torch.Tensor,
    action_noise: torch.Tensor,
    *,
    capture_cache: bool = True,
) -> dict[str, Any]:
    model = policy.model
    timestep_video = torch.zeros((1,), dtype=future_latent.dtype, device=model.device)
    fuse = bool(getattr(model.video_expert, "fuse_vae_embedding_in_latents", False))
    pre = model.video_expert.pre_dit(
        x=future_latent,
        timestep=timestep_video,
        context=prepared["context"],
        context_mask=prepared["context_mask"],
        action=None,
        fuse_vae_embedding_in_latents=fuse,
    )
    video_seq_len = int(pre["tokens"].shape[1])
    tokens_per_group = int(pre["meta"]["tokens_per_frame"])
    attention_mask = model._build_mot_attention_mask(
        video_seq_len=video_seq_len,
        action_seq_len=action_noise.shape[1],
        video_tokens_per_frame=tokens_per_group,
        device=pre["tokens"].device,
    )
    cache = model.mot.prefill_video_cache(
        video_tokens=pre["tokens"],
        video_freqs=pre["freqs"],
        video_t_mod=pre["t_mod"],
        video_context_payload={"context": pre["context"], "mask": pre["context_mask"]},
        video_attention_mask=attention_mask[:video_seq_len, :video_seq_len],
    )
    latents_action = action_noise.clone()
    timesteps, deltas = model.infer_action_scheduler.build_inference_schedule(
        policy.num_inference_steps,
        model.device,
        latents_action.dtype,
        shift_override=None,
    )
    for step_t, delta in zip(timesteps, deltas):
        prediction = model._predict_action_noise_with_cache(
            latents_action=latents_action,
            timestep_action=step_t.unsqueeze(0).to(latents_action),
            context=prepared["context"],
            context_mask=prepared["context_mask"],
            video_kv_cache=cache,
            attention_mask=attention_mask,
            video_seq_len=video_seq_len,
        )
        latents_action = model.infer_action_scheduler.step(prediction, delta, latents_action)
    result = {
        "action": latents_action[0].detach().cpu().float(),
        "video_pre_tokens": tensor_summary(pre["tokens"]),
        "video_seq_len": video_seq_len,
        "tokens_per_temporal_group": tokens_per_group,
        "grid_size": list(pre["meta"]["grid_size"]),
        "attention_mask_shape": list(attention_mask.shape),
        "action_timesteps": timesteps.detach().cpu().float(),
        "action_deltas": deltas.detach().cpu().float(),
    }
    if capture_cache:
        result["cache"] = clone_cache_cpu(cache)
        result["cache_trace"] = cache_trace(cache)
    return result


def action_noise(policy: fgl.BadWAMPolicy, seed: int) -> torch.Tensor:
    generator = torch.Generator(device=str(policy.cfg.EVALUATION.rand_device)).manual_seed(seed)
    return torch.randn(
        (1, policy.action_horizon, policy.model.action_expert.action_dim),
        generator=generator,
        device=str(policy.cfg.EVALUATION.rand_device),
        dtype=torch.float32,
    ).to(policy.model.device, policy.model.torch_dtype)


def transfer_metrics(source: torch.Tensor, donor: torch.Tensor, patched: torch.Tensor) -> dict[str, float]:
    def compute(a: torch.Tensor, b: torch.Tensor, p: torch.Tensor) -> tuple[float, float, float]:
        donor_delta = (b - a).reshape(-1).double()
        patch_delta = (p - a).reshape(-1).double()
        donor_norm = float(torch.linalg.vector_norm(donor_delta))
        patch_norm = float(torch.linalg.vector_norm(patch_delta))
        dot = float(torch.dot(patch_delta, donor_delta))
        transfer = dot / (donor_norm * donor_norm + 1e-12)
        cosine = 0.0 if donor_norm == 0.0 or patch_norm == 0.0 else dot / (donor_norm * patch_norm)
        return patch_norm, transfer, cosine

    first_l2, first_transfer, first_cosine = compute(source[0], donor[0], patched[0])
    full_l2, full_transfer, full_cosine = compute(source, donor, patched)
    return {
        "first_step_action_l2": first_l2,
        "first_step_donor_transfer": first_transfer,
        "first_step_directional_cosine": first_cosine,
        "full_chunk_action_l2": full_l2,
        "full_chunk_donor_transfer": full_transfer,
        "full_chunk_directional_cosine": full_cosine,
    }


def assert_current_cache_unchanged(
    source: Sequence[Mapping[str, torch.Tensor]],
    patched: Sequence[Mapping[str, torch.Tensor]],
    tokens_per_group: int,
) -> None:
    for layer, (lhs, rhs) in enumerate(zip(source, patched)):
        for kind in ("k", "v"):
            if not torch.equal(lhs[kind][:, :tokens_per_group], rhs[kind][:, :tokens_per_group]):
                raise AssertionError(f"Current-observation K/V changed at layer {layer}/{kind}")


def save_decoded_video(policy: fgl.BadWAMPolicy, latent: torch.Tensor, path: Path) -> None:
    frames = policy.model._decode_latents(latent, tiled=bool(policy.cfg.EVALUATION.tiled))
    writer = imageio.get_writer(path, fps=8, codec="libx264", quality=8)
    try:
        for frame in frames:
            writer.append_data(np.asarray(frame.convert("RGB"), dtype=np.uint8))
    finally:
        writer.close()


def architecture_report(trace: Mapping[str, Any]) -> str:
    shape = trace["representations"]["final_video_latent"]["shape"]
    kv_shape = trace["representations"]["video_kv_layer_0_k"]["shape"]
    return f"""# BadWAM IDM-WAM architecture audit

## Verified released LIBERO inference path

1. RGB observations are assembled in `experiments/libero/eval_libero_single.py:_obs_to_model_input:191-248`; `FastWAMIDM.infer_joint` validates and moves the `[1,3,H,W]` tensor at `src/fastwam/models/wan22/fastwam_idm.py:297-348`.
2. The instruction is formatted at `experiments/first_grasp_lock/run_first_grasp_lock.py:400-409`, encoded at `src/fastwam/models/wan22/fastwam.py:201-217`, and consumed in IDM at `fastwam_idm.py:351-378`.
3. Proprio is constructed/normalized at `eval_libero_single.py:174-188,246-263`, validated at `fastwam_idm.py:314-325`, and appended as one language-context token at `fastwam.py:219-240`.
4. The imagination module is stage-1 video diffusion in `fastwam_idm.py:327-399`.
5. Future construction starts from random video latent noise, clamps the current-frame latent, and denoises the full latent at `fastwam_idm.py:331-399`.
6. The final video latent `{shape}` is frozen at `fastwam_idm.py:397-404`; `video_expert.pre_dit` turns it into temporally ordered tokens at `wan_video_dit.py:509-620`; `MoT.prefill_video_cache` creates 30 layers of K/V at `mot.py:257-341` (layer-0 K shape `{kv_shape}`).
7. Optional pixel decoding occurs only in the return expression at `fastwam_idm.py:449-451`, implemented by `fastwam.py:_decode_latents:267-275`.
8. Action denoising receives source context plus the frozen video K/V at `fastwam_idm.py:430-446` and `fastwam.py:_predict_action_noise_with_cache:695-723`.
9. Action queries concatenate video K/V with action K/V at `mot.py:343-445`; action tokens are projected to noise predictions by `action_dit.py:226-302`, then the action scheduler updates the action latent at `fastwam_idm.py:436-447`.
10. The final 32x7 action chunk is returned at `fastwam_idm.py:449-452`.

## Architectural answer

The release contains a genuine staged imagination-to-action interface, but it is not an exclusive bottleneck. The action expert reads future-side video K/V derived from the generated video latent, while also directly cross-attending the same language-plus-proprio context and retaining its own action state/noise. The accurate graph is:

`RGB + instruction + proprio + video noise -> generated video latent -> video hidden states/K/V -> action attention`

plus the parallel path

`instruction + proprio + action noise/state -> action expert`.

Decoded RGB future pixels are not consumed by action generation. They are computed only after the action denoising loop for the returned visualization.
"""


def candidate_report(trace: Mapping[str, Any]) -> str:
    latent = trace["representations"]["final_video_latent"]
    kv = trace["representations"]["video_kv_layer_0_k"]
    video_tokens = trace["representations"]["video_pre_tokens"]
    action_shape = trace["representations"]["action_output"]["shape"]
    return f"""# Candidate imagination interventions

| Candidate | Location | Shape/dtype | Future-specific | Contents and action use | Patch feasibility |
|---|---|---|---|---|---|
| Decoded future pixels | `fastwam.py:_decode_latents:267-275` | 9 RGB frames, uint8 | Yes, except returned current frame | Visualization only; action never consumes pixels | Patchable but causally downstream of action, therefore invalid |
| Final denoised video latent | `fastwam_idm.py:397-405` | `{latent['shape']}`, `{latent['dtype']}` | Temporal groups 1-2 are future-specific; group 0 is current observation | Generated from current RGB, instruction, proprio and video noise; indirectly consumed through video prefill/K/V | **Chosen IMAGINATION_REP**; replace `[:, :, 1:]` without weight changes |
| Video hidden states | `mot.py:298-340` | per layer `{video_tokens['shape']}`, `{video_tokens['dtype']}` | Contains current and future token groups | Includes language/proprio through per-layer cross-attention; generates next-layer K/V | Feasible with a deeper hook, but later and less minimal |
| Future-side K/V | `mot.py:301-341` | layer-0 K `{kv['shape']}`, `{kv['dtype']}`, 30 layers | Token groups 1-2 are future-side | Includes generated video, source language/proprio and layer history; directly read by action queries | Supported by utility for selected layers/components/time groups |
| Shared world/action hidden states | none persistent in released IDM stage 2 | N/A | N/A | Video is prefilled separately; only action hidden state is recurrent across action layers | Not a valid future representation |
| Action hidden states | `mot.py:390-445` | action sequence `{action_shape}` before 1024-D embedding, bf16 hidden state afterward | No | Mix action state, direct text/proprio cross-attention, and reads from video K/V | Downstream action representation; not IMAGINATION_REP |

The selected representation is the final stage-1 video-latent future slice. It is the earliest practical, temporally identifiable tensor produced by the completed imagination pathway and causally upstream of action. It is broader than a pure task plan: it also contains predicted visual state conditioned on the current observation, language and proprio.

The modular utility `imagination_intervention.py` saves/reloads this tensor, replaces selected future temporal groups, and can alternatively replace selected K/V layers, temporal groups and K/V components. It does not modify model parameters or monkey-patch the released model.
"""


def temporal_report(trace: Mapping[str, Any]) -> str:
    temporal = trace["temporal_structure"]
    return f"""# IDM-WAM temporal structure

- Pixel-video horizon: {temporal['pixel_frames']} frames: current frame plus {temporal['pixel_future_frames']} predicted frames.
- VAE temporal downsampling factor: {temporal['vae_temporal_downsample_factor']}.
- Latent temporal groups: {temporal['latent_temporal_groups']} total; group 0 is the clamped current observation and groups 1-2 are future groups.
- Spatial tokens per temporal group: {temporal['tokens_per_temporal_group']}.
- Total video tokens: {temporal['total_video_tokens']}.
- Token order is frame-major `(f,h,w)` (`wan_video_dit.py:600-606`). 3D RoPE includes a temporal coordinate (`wan_video_dit.py:602-606`).
- The first-frame-causal video mask prevents current tokens from reading future groups (`wan_video_dit.py:501-505`). Action queries can read all video groups (`fastwam_joint.py:29-49`).
- Temporal identities therefore survive into the K/V positions consumed by action. There is no hard one-to-one mapping between an action step and a future group; every action query can attend all current/future video tokens.

The two future latent groups are temporally identifiable but should be interpreted as temporally compressed video groups, not exact one-token-per-decoded-frame states. With only two future groups, an early/late smoke is possible; there is no separate middle group.
"""


def smoke_report(metrics: Sequence[Mapping[str, Any]], passed: bool) -> str:
    rows = {row["condition"]: row for row in metrics}
    identity = rows["identity"]
    semantic = rows.get("semantic_donor")
    shuffled = rows.get("shuffled")
    if not passed or semantic is None or shuffled is None:
        return f"""# IDM-WAM causal-interface smoke report

Identity intervention failed. Maximum action L2 was {identity['full_chunk_action_l2']}. Per protocol, semantic donor and shuffled experiments were not run. Inspect `logs/identity_failure.json`.
"""
    return f"""# IDM-WAM causal-interface smoke report

## Scope

One same-state LIBERO-Object case was used: identical RGB and proprio, source instruction targets `{SOURCE_TARGET}`, and the counterfactual donor instruction targets `{DONOR_TARGET}` in the same scene. Video noise, action noise, seeds and denoising schedules are shared. Only the final generated future latent groups are replaced; the source current-observation latent, source instruction/proprio context and all current-observation K/V remain bit-exact.

## Results

- Identity: full action L2 `{identity['full_chunk_action_l2']:.8g}`; bit-exact `{bool(identity['bit_exact'])}`.
- Semantic donor, full chunk: L2 `{semantic['full_chunk_action_l2']:.4f}`, transfer `{semantic['full_chunk_donor_transfer']:.4f}`, cosine `{semantic['full_chunk_directional_cosine']:.4f}`.
- Semantic donor, first step: L2 `{semantic['first_step_action_l2']:.4f}`, transfer `{semantic['first_step_donor_transfer']:.4f}`, cosine `{semantic['first_step_directional_cosine']:.4f}`.
- Shuffled, full chunk: L2 `{shuffled['full_chunk_action_l2']:.4f}`, transfer `{shuffled['full_chunk_donor_transfer']:.4f}`, cosine `{shuffled['full_chunk_directional_cosine']:.4f}`.
- Shuffled, first step: L2 `{shuffled['first_step_action_l2']:.4f}`, transfer `{shuffled['first_step_donor_transfer']:.4f}`, cosine `{shuffled['first_step_directional_cosine']:.4f}`.

This is a single-case interface validation, not evidence of a population-level effect. A semantic donor is considered promising only if its donor-direction transfer/cosine is materially stronger than shuffled corruption; action L2 alone is not used as evidence.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=REPO / "runs/badwam_idm_causal_audit")
    parser.add_argument("--gpu-id", type=int, default=0)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing experiment directory: {output}")
    for child in ("actions", "representations", "logs"):
        (output / child).mkdir(parents=True, exist_ok=True)

    policy = fgl.BadWAMPolicy("idm", args.gpu_id)
    model = policy.model
    checkpoint = policy.checkpoint.resolve()
    checkpoint_hash_before = file_sha256(checkpoint)
    params_before = parameter_signature(model)
    env, task, obs = make_observation(SOURCE_TASK, SOURCE_STATE)
    try:
        source = prepare(policy, obs, task.language)
        donor = prepare(policy, obs, DONOR_INSTRUCTION)
    finally:
        env.close()

    # Strict same-state controls.
    assert torch.equal(source["image"], donor["image"])
    assert torch.equal(source["proprio"], donor["proprio"])
    source_input_hashes = {
        "rgb": tensor_sha256(source["image"]),
        "proprio": tensor_sha256(source["proprio"]),
        "context": tensor_sha256(source["context"]),
        "context_mask": tensor_sha256(source["context_mask"]),
        "prompt_sha256": hashlib.sha256(source["prompt"].encode()).hexdigest(),
    }
    donor_input_hashes = {
        "rgb": tensor_sha256(donor["image"]),
        "proprio": tensor_sha256(donor["proprio"]),
        "context": tensor_sha256(donor["context"]),
        "context_mask": tensor_sha256(donor["context_mask"]),
        "prompt_sha256": hashlib.sha256(donor["prompt"].encode()).hexdigest(),
    }

    source_video = generate_video_latent(policy, source, seed=SEED)
    donor_video = generate_video_latent(policy, donor, seed=SEED)
    assert torch.equal(source_video["initial_noise"], donor_video["initial_noise"])
    assert torch.equal(source_video["first_frame_latent"], donor_video["first_frame_latent"])
    shared_action_noise = action_noise(policy, SEED)

    source_summary = save_representation(
        output / "representations/source_final_video_latent.pt", source_video["latent"]
    )
    donor_summary = save_representation(
        output / "representations/donor_final_video_latent.pt", donor_video["latent"]
    )
    reloaded_source, reloaded_summary = load_representation(
        output / "representations/source_final_video_latent.pt"
    )
    reloaded_source = reloaded_source.to(model.device, model.torch_dtype)
    if not torch.equal(reloaded_source, source_video["latent"]):
        raise AssertionError("Reloaded source representation is not bit-exact")

    clean = action_from_latent(policy, source, source_video["latent"], shared_action_noise)
    donor_reference = action_from_latent(policy, donor, donor_video["latent"], shared_action_noise)

    identity_hook = ImaginationIntervention(mode="identity", temporal_groups=(1, 2))
    identity_latent = identity_hook.patch_latent(reloaded_source, reloaded_source)
    identity = action_from_latent(policy, source, identity_latent, shared_action_noise)
    identity_equal = torch.equal(identity["action"], clean["action"])
    identity_max_abs = float((identity["action"] - clean["action"]).abs().max())
    identity_metrics = {
        "condition": "identity",
        **transfer_metrics(clean["action"], donor_reference["action"], identity["action"]),
        "bit_exact": identity_equal,
        "max_abs_vs_clean": identity_max_abs,
    }
    metrics: list[dict[str, Any]] = [identity_metrics]
    debug_rows: list[dict[str, Any]] = []

    def record_debug(condition: str, hook: ImaginationIntervention, result: Mapping[str, Any]) -> None:
        debug_rows.append(
            {
                "condition": condition,
                "source_rgb_sha256": source_input_hashes["rgb"],
                "source_proprio_sha256": source_input_hashes["proprio"],
                "source_context_sha256": source_input_hashes["context"],
                "action_noise_sha256": tensor_sha256(shared_action_noise),
                "video_schedule_sha256": tensor_sha256(source_video["timesteps"]),
                "action_schedule_sha256": tensor_sha256(result["action_timesteps"]),
                "source_rep_sha256": source_summary["sha256"],
                "donor_rep_sha256": donor_summary["sha256"],
                "injected_rep_sha256": hook.debug["injected"]["sha256"],
                "selected_temporal_groups": json.dumps(hook.debug["selected_temporal_groups"]),
                "current_latent_group_bit_exact": hook.debug["current_group_preserved_bit_exact"],
                "current_kv_all_layers_bit_exact": True,
                "parameters_unchanged": parameter_signature(model) == params_before,
            }
        )

    assert_current_cache_unchanged(
        clean["cache"], identity["cache"], clean["tokens_per_temporal_group"]
    )
    record_debug("identity", identity_hook, identity)
    np.savez_compressed(output / "actions/clean.npz", action=clean["action"].numpy())
    np.savez_compressed(output / "actions/donor_reference.npz", action=donor_reference["action"].numpy())
    np.savez_compressed(output / "actions/identity.npz", action=identity["action"].numpy())
    write_json(output / "logs/identity_intervention.json", identity_hook.debug)

    if not identity_equal:
        write_json(
            output / "logs/identity_failure.json",
            {"bit_exact": False, "max_abs": identity_max_abs, "tolerance": 0.0},
        )
    else:
        experiments = [
            ("semantic_donor", ImaginationIntervention(mode="semantic_donor", temporal_groups=(1, 2))),
            ("shuffled", ImaginationIntervention(mode="shuffled", temporal_groups=(1, 2), random_seed=1042)),
            ("semantic_donor_early_future", ImaginationIntervention(mode="semantic_donor", temporal_groups=(1,))),
            ("semantic_donor_late_future", ImaginationIntervention(mode="semantic_donor", temporal_groups=(2,))),
        ]
        for condition, hook in experiments:
            patched_latent = hook.patch_latent(source_video["latent"], donor_video["latent"])
            result = action_from_latent(policy, source, patched_latent, shared_action_noise)
            assert_current_cache_unchanged(
                clean["cache"], result["cache"], clean["tokens_per_temporal_group"]
            )
            row = {
                "condition": condition,
                **transfer_metrics(clean["action"], donor_reference["action"], result["action"]),
                "bit_exact": False,
                "max_abs_vs_clean": float((result["action"] - clean["action"]).abs().max()),
            }
            metrics.append(row)
            record_debug(condition, hook, result)
            np.savez_compressed(output / f"actions/{condition}.npz", action=result["action"].numpy())
            write_json(output / f"logs/{condition}_intervention.json", hook.debug)

    if parameter_signature(model) != params_before:
        raise AssertionError("Pretrained parameters changed during intervention")
    checkpoint_hash_after = file_sha256(checkpoint)
    if checkpoint_hash_after != checkpoint_hash_before:
        raise AssertionError("Checkpoint bytes changed during intervention")

    write_csv(output / "smoke_metrics.csv", metrics)
    write_csv(output / "intervention_debug.csv", debug_rows)
    temporal_groups = int(source_video["latent"].shape[2])
    temporal_structure = {
        "pixel_frames": int(policy.num_video_frames),
        "pixel_future_frames": int(policy.num_video_frames - 1),
        "vae_temporal_downsample_factor": int(model.vae.temporal_downsample_factor),
        "latent_temporal_groups": temporal_groups,
        "future_latent_groups": list(range(1, temporal_groups)),
        "latent_group_semantics": {
            "0": "clamped current-observation latent",
            "1": "early temporally compressed future group",
            "2": "late temporally compressed future group",
        },
        "tokens_per_temporal_group": int(clean["tokens_per_temporal_group"]),
        "total_video_tokens": int(clean["video_seq_len"]),
        "grid_size_f_h_w": clean["grid_size"],
        "token_order": "frame-major flattening of (f,h,w)",
        "temporal_positional_encoding": "3D RoPE with explicit temporal coordinate",
        "temporal_identity_survives_to_action": True,
        "action_to_time_mapping": "all action queries may attend all video temporal groups; no one-to-one hard map",
    }
    write_json(output / "temporal_structure.json", temporal_structure)
    tensor_trace = {
        "model": {
            **policy.metadata(),
            "checkpoint_sha256_before": checkpoint_hash_before,
            "checkpoint_sha256_after": checkpoint_hash_after,
            "parameters_unchanged": parameter_signature(model) == params_before,
        },
        "case": {
            "suite": "libero_object",
            "task_index": SOURCE_TASK,
            "state_index": SOURCE_STATE,
            "source_instruction": task.language,
            "donor_instruction": DONOR_INSTRUCTION,
            "source_target": SOURCE_TARGET,
            "donor_target": DONOR_TARGET,
            "same_rgb": source_input_hashes["rgb"] == donor_input_hashes["rgb"],
            "same_proprio": source_input_hashes["proprio"] == donor_input_hashes["proprio"],
        },
        "source_inputs": source_input_hashes,
        "donor_inputs": donor_input_hashes,
        "shared_randomness": {
            "seed": SEED,
            "video_initial_noise": tensor_summary(source_video["initial_noise"]),
            "action_initial_noise": tensor_summary(shared_action_noise),
            "video_timesteps": source_video["timesteps"],
            "video_deltas": source_video["deltas"],
            "action_timesteps": clean["action_timesteps"],
            "action_deltas": clean["action_deltas"],
        },
        "representations": {
            "final_video_latent": source_summary,
            "donor_final_video_latent": donor_summary,
            "reloaded_source": reloaded_summary,
            "video_pre_tokens": clean["video_pre_tokens"],
            "video_kv_layer_0_k": clean["cache_trace"][0]["k"],
            "video_kv_layer_0_v": clean["cache_trace"][0]["v"],
            "action_output": tensor_summary(clean["action"]),
        },
        "video_kv_cache_by_layer": clean["cache_trace"],
        "temporal_structure": temporal_structure,
        "identity": {"bit_exact": identity_equal, "max_abs": identity_max_abs},
    }
    write_json(output / "tensor_trace.json", tensor_trace)
    (output / "architecture_audit.md").write_text(architecture_report(tensor_trace), encoding="utf-8")
    (output / "candidate_interventions.md").write_text(candidate_report(tensor_trace), encoding="utf-8")
    (output / "smoke_test_report.md").write_text(smoke_report(metrics, identity_equal), encoding="utf-8")

    # Decoding is deliberately performed after all actions to demonstrate that
    # pixels are downstream outputs, not action inputs.
    save_decoded_video(policy, source_video["latent"], output / "representations/source_decoded_future.mp4")
    save_decoded_video(policy, donor_video["latent"], output / "representations/donor_decoded_future.mp4")

    audit = {
        "complete": bool(identity_equal and len(metrics) == 5),
        "identity_bit_exact": identity_equal,
        "required_files": {},
        "zero_byte_files": [],
        "checkpoint_unchanged": checkpoint_hash_before == checkpoint_hash_after,
        "parameters_unchanged": parameter_signature(model) == params_before,
    }
    required = [
        "architecture_audit.md",
        "candidate_interventions.md",
        "tensor_trace.json",
        "temporal_structure.json",
        "smoke_metrics.csv",
        "intervention_debug.csv",
        "smoke_test_report.md",
    ]
    audit["required_files"] = {name: (output / name).is_file() for name in required}
    audit["zero_byte_files"] = [str(path) for path in output.rglob("*") if path.is_file() and path.stat().st_size == 0]
    audit["complete"] = bool(
        audit["complete"]
        and all(audit["required_files"].values())
        and not audit["zero_byte_files"]
        and audit["checkpoint_unchanged"]
        and audit["parameters_unchanged"]
    )
    write_json(output / "logs/final_audit.json", audit)
    print(json.dumps(jsonable({"output": output, "audit": audit, "metrics": metrics}), sort_keys=True), flush=True)
    if not identity_equal:
        raise SystemExit("STOP: identity intervention was not bit-exact")


if __name__ == "__main__":
    main()

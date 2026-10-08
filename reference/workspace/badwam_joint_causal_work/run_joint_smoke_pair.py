#!/usr/bin/env python3
from __future__ import annotations
from wam_causal_audit.paths import resolve as _release_path

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import imageio.v2 as imageio
import numpy as np
import torch

REPO = Path(_release_path('@DATA@/BadWAM'))
AUDIT_WORK = Path(_release_path('@WORKSPACE@/badwam_idm_causal_audit_work'))
IDM_WORK = Path(_release_path('@WORKSPACE@/badwam_idm_same_frame_goal_work'))
for extra in (
    REPO,
    REPO / "src",
    REPO / "experiments/libero",
    REPO / "experiments/first_grasp_lock",
    AUDIT_WORK,
    IDM_WORK,
    Path(_release_path('@WORKSPACE@/LIBERO')),
    Path(_release_path('@WORKSPACE@/evaluation/d2_ghost_pilot')),
):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import run_first_grasp_lock as fgl  # noqa: E402
from common import ALT_INSTRUCTION, TARGETS, make_env, seed_for  # noqa: E402
from imagination_intervention import (  # noqa: E402
    parameter_signature,
    save_representation,
    tensor_sha256,
    tensor_summary,
)
from run_smoke import file_sha256, prepare  # noqa: E402


ROOT = REPO / "runs/badwam_joint_causal"
LAYER_GROUPS = {
    "early": tuple(range(0, 8)),
    "middle": tuple(range(8, 22)),
    "late": tuple(range(22, 30)),
    "all": tuple(range(30)),
}


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
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.write_text(json.dumps(jsonable(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def hash_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def transfer_metrics(source: torch.Tensor, donor: torch.Tensor, patched: torch.Tensor, prefix: str) -> dict[str, float]:
    a = source.reshape(-1).double()
    b = donor.reshape(-1).double()
    p = patched.reshape(-1).double()
    donor_delta = b - a
    patch_delta = p - a
    donor_norm = float(torch.linalg.vector_norm(donor_delta))
    patch_norm = float(torch.linalg.vector_norm(patch_delta))
    dot = float(torch.dot(patch_delta, donor_delta))
    return {
        f"{prefix}_l2": patch_norm,
        f"{prefix}_donor_l2": donor_norm,
        f"{prefix}_transfer": dot / (donor_norm * donor_norm + 1e-12),
        f"{prefix}_cosine": 0.0 if donor_norm == 0.0 or patch_norm == 0.0 else dot / (donor_norm * patch_norm),
    }


class QKVController:
    """Capture or replace one expert's Q/K/V at the real mixed-attention boundary."""

    def __init__(self, model: Any) -> None:
        self.model = model
        self.original = model.mot._build_expert_attention_io
        self.block_map = {id(block): ("video", i) for i, block in enumerate(model.video_expert.blocks)}
        self.block_map.update({id(block): ("action", i) for i, block in enumerate(model.action_expert.blocks)})
        self.step = -1
        self.capture = False
        self.cache: dict[str, dict[tuple[int, int], dict[str, torch.Tensor]]] = {"video": {}, "action": {}}
        self.patch_cache: dict[tuple[int, int], dict[str, torch.Tensor]] | None = None
        self.patch_modality = "video"
        self.patch_layers: set[int] = set()
        self.patch_temporal_groups: tuple[int, ...] = (0, 1, 2)
        self.patch_components = "kv"
        self.patch_mode = "semantic"
        self.random_seed = 0
        self.tokens_per_group = 98
        self.events: list[dict[str, Any]] = []

    def install(self) -> None:
        def wrapped(expert: Any, block: Any, *args: Any, **kwargs: Any):
            result = self.original(expert, block, *args, **kwargs)
            modality, layer = self.block_map[id(block)]
            if self.capture:
                self.cache[modality][(self.step, layer)] = {
                    "k": result[1].detach().clone(),
                    "v": result[2].detach().clone(),
                }
            if (
                self.patch_cache is not None
                and modality == self.patch_modality
                and layer in self.patch_layers
            ):
                row = self.patch_cache[(self.step, layer)]
                changed = list(result)
                kinds = ("k", "v") if self.patch_components == "kv" else (self.patch_components,)
                selected_tokens = torch.cat(
                    [
                        torch.arange(group * self.tokens_per_group, (group + 1) * self.tokens_per_group)
                        for group in self.patch_temporal_groups
                    ]
                ).to(result[1].device)
                for kind in kinds:
                    slot = 1 if kind == "k" else 2
                    replacement = row[kind].index_select(1, selected_tokens).clone()
                    if self.patch_mode == "shuffled":
                        pieces = []
                        offset = 0
                        for group in self.patch_temporal_groups:
                            count = self.tokens_per_group
                            generator = torch.Generator(device="cpu").manual_seed(
                                self.random_seed + self.step * 100_003 + layer * 101 + group * 7 + slot
                            )
                            permutation = torch.randperm(count, generator=generator).to(replacement.device)
                            pieces.append(replacement[:, offset : offset + count].index_select(1, permutation))
                            offset += count
                        replacement = torch.cat(pieces, dim=1)
                    injected = changed[slot].clone()
                    injected[:, selected_tokens] = replacement
                    changed[slot] = injected
                if len(self.events) < 8:
                    self.events.append(
                        {
                            "step": self.step,
                            "layer": layer,
                            "modality": modality,
                            "components": self.patch_components,
                            "temporal_groups": list(self.patch_temporal_groups),
                            "mode": self.patch_mode,
                        }
                    )
                return tuple(changed)
            return result

        self.model.mot._build_expert_attention_io = wrapped

    def uninstall(self) -> None:
        self.model.mot._build_expert_attention_io = self.original


@torch.no_grad()
def controlled_infer(
    policy: fgl.BadWAMPolicy,
    image: torch.Tensor,
    video_context: torch.Tensor,
    video_context_mask: torch.Tensor,
    action_context: torch.Tensor,
    action_context_mask: torch.Tensor,
    *,
    seed: int,
    controller: QKVController | None,
) -> dict[str, Any]:
    """Official Joint-WAM loop with independently supplied video/action context payloads."""
    model = policy.model
    image = image.to(model.device, model.torch_dtype)
    _, _, height, width = image.shape
    latent_t = (policy.num_video_frames - 1) // model.vae.temporal_downsample_factor + 1
    rand_device = str(policy.cfg.EVALUATION.rand_device)
    video_generator = torch.Generator(device=rand_device).manual_seed(seed)
    action_generator = torch.Generator(device=rand_device).manual_seed(seed)
    initial_video_noise = torch.randn(
        (1, model.vae.model.z_dim, latent_t, height // model.vae.upsampling_factor, width // model.vae.upsampling_factor),
        generator=video_generator,
        device=rand_device,
        dtype=torch.float32,
    ).to(model.device, model.torch_dtype)
    initial_action_noise = torch.randn(
        (1, policy.action_horizon, model.action_expert.action_dim),
        generator=action_generator,
        device=rand_device,
        dtype=torch.float32,
    ).to(model.device, model.torch_dtype)
    first_frame = model._encode_input_image_latents_tensor(
        input_image=image, tiled=bool(policy.cfg.EVALUATION.tiled)
    )
    latents_video = initial_video_noise.clone()
    latents_action = initial_action_noise.clone()
    latents_video[:, :, 0:1] = first_frame.clone()
    video_timesteps, video_deltas = model.infer_video_scheduler.build_inference_schedule(
        policy.num_inference_steps, model.device, latents_video.dtype, shift_override=None
    )
    action_timesteps, action_deltas = model.infer_action_scheduler.build_inference_schedule(
        policy.num_inference_steps, model.device, latents_action.dtype, shift_override=None
    )
    fuse = bool(getattr(model.video_expert, "fuse_vae_embedding_in_latents", False))
    for step, (tv, dv, ta, da) in enumerate(zip(video_timesteps, video_deltas, action_timesteps, action_deltas)):
        if controller is not None:
            controller.step = step
        video_pre = model.video_expert.pre_dit(
            x=latents_video,
            timestep=tv.unsqueeze(0).to(latents_video),
            context=video_context,
            context_mask=video_context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse,
        )
        action_pre = model.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=ta.unsqueeze(0).to(latents_action),
            context=action_context,
            context_mask=action_context_mask,
        )
        tokens_per_group = int(video_pre["meta"]["tokens_per_frame"])
        if controller is not None:
            controller.tokens_per_group = tokens_per_group
        mask = model._build_mot_attention_mask(
            video_seq_len=video_pre["tokens"].shape[1],
            action_seq_len=action_pre["tokens"].shape[1],
            video_tokens_per_frame=tokens_per_group,
            device=video_pre["tokens"].device,
        )
        output = model.mot(
            embeds_all={"video": video_pre["tokens"], "action": action_pre["tokens"]},
            attention_mask=mask,
            freqs_all={"video": video_pre["freqs"], "action": action_pre["freqs"]},
            context_all={
                "video": {"context": video_pre["context"], "mask": video_pre["context_mask"]},
                "action": {"context": action_pre["context"], "mask": action_pre["context_mask"]},
            },
            t_mod_all={"video": video_pre["t_mod"], "action": action_pre["t_mod"]},
        )
        pred_video = model.video_expert.post_dit(output["video"], video_pre)
        pred_action = model.action_expert.post_dit(output["action"], action_pre)
        latents_video = model.infer_video_scheduler.step(pred_video, dv, latents_video)
        latents_action = model.infer_action_scheduler.step(pred_action, da, latents_action)
        latents_video[:, :, 0:1] = first_frame.clone()
    return {
        "video_latent": latents_video.detach(),
        "action": latents_action[0].detach().cpu().float(),
        "initial_video_noise": initial_video_noise.detach(),
        "initial_action_noise": initial_action_noise.detach(),
        "first_frame_latent": first_frame.detach(),
        "video_timesteps": video_timesteps.detach(),
        "video_deltas": video_deltas.detach(),
        "action_timesteps": action_timesteps.detach(),
        "action_deltas": action_deltas.detach(),
        "tokens_per_group": tokens_per_group,
    }


def cache_manifest(cache: Mapping[str, Mapping[tuple[int, int], Mapping[str, torch.Tensor]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for modality in ("video", "action"):
        for (step, layer), values in sorted(cache[modality].items()):
            for kind in ("k", "v"):
                summary = tensor_summary(values[kind])
                rows.append({"modality": modality, "step": step, "layer": layer, "component": kind, **summary})
    return rows


def representation_difference_rows(
    source: Mapping[str, Mapping[tuple[int, int], Mapping[str, torch.Tensor]]],
    donor: Mapping[str, Mapping[tuple[int, int], Mapping[str, torch.Tensor]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for modality in ("video", "action"):
        for key in sorted(source[modality]):
            step, layer = key
            for kind in ("k", "v"):
                a = source[modality][key][kind].double()
                b = donor[modality][key][kind].double()
                delta = b - a
                a_norm = float(torch.linalg.vector_norm(a))
                b_norm = float(torch.linalg.vector_norm(b))
                delta_norm = float(torch.linalg.vector_norm(delta))
                dot = float(torch.sum(a * b))
                rows.append(
                    {
                        "modality": modality,
                        "step": step,
                        "layer": layer,
                        "component": kind,
                        "source_l2": a_norm,
                        "donor_l2": b_norm,
                        "difference_l2": delta_norm,
                        "relative_difference": delta_norm / (a_norm + 1e-12),
                        "source_donor_cosine": dot / (a_norm * b_norm + 1e-12),
                        "bit_exact": torch.equal(source[modality][key][kind], donor[modality][key][kind]),
                    }
                )
    return rows


def save_cache(path: Path, cache: Mapping[str, Mapping[tuple[int, int], Mapping[str, torch.Tensor]]]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        modality: {
            f"step_{step:02d}_layer_{layer:02d}": {
                kind: tensor.detach().cpu() for kind, tensor in values.items()
            }
            for (step, layer), values in sorted(rows.items())
        }
        for modality, rows in cache.items()
    }
    torch.save(payload, path)


def save_video(policy: fgl.BadWAMPolicy, latent: torch.Tensor, path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = policy.model._decode_latents(latent, tiled=bool(policy.cfg.EVALUATION.tiled))
    writer = imageio.get_writer(path, fps=8, codec="libx264", quality=8)
    try:
        for frame in frames:
            writer.append_data(np.asarray(frame.convert("RGB"), dtype=np.uint8))
    finally:
        writer.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=int, required=True)
    parser.add_argument("--state", type=int, default=0)
    parser.add_argument("--pair-id", type=int, default=0)
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument("--verify-official", action="store_true")
    parser.add_argument("--phase", choices=("smoke", "one_call"), default="smoke")
    parser.add_argument("--save-full-qkv", action="store_true")
    args = parser.parse_args()
    pair_dir = ROOT / args.phase / f"task_{args.task}_state_{args.state}"
    if pair_dir.exists() and any(path.is_file() for path in pair_dir.rglob("*")):
        raise FileExistsError(f"Refusing to overwrite existing pair directory: {pair_dir}")
    for child in ("actions", "future_latents", "representations", "videos", "logs"):
        (pair_dir / child).mkdir(parents=True, exist_ok=True)

    started = time.time()
    policy = fgl.BadWAMPolicy("joint", args.gpu_id)
    model = policy.model
    checkpoint = policy.checkpoint.resolve()
    checkpoint_hash = file_sha256(checkpoint)
    params_before = parameter_signature(model)
    env, task, obs = make_env(args.task, args.state)
    try:
        source = prepare(policy, obs, task.language)
        alternate_instruction = ALT_INSTRUCTION[args.task]
        donor = prepare(policy, obs, alternate_instruction)
    finally:
        env.close()
    assert torch.equal(source["image"], donor["image"])
    assert torch.equal(source["proprio"], donor["proprio"])
    seed = seed_for(args.task, args.pair_id)
    source_hashes = {
        "rgb": tensor_sha256(source["image"]),
        "proprio": tensor_sha256(source["proprio"]),
        "context": tensor_sha256(source["context"]),
        "context_mask": tensor_sha256(source["context_mask"]),
        "instruction": hash_text(task.language),
    }
    donor_hashes = {
        "rgb": tensor_sha256(donor["image"]),
        "proprio": tensor_sha256(donor["proprio"]),
        "context": tensor_sha256(donor["context"]),
        "context_mask": tensor_sha256(donor["context_mask"]),
        "instruction": hash_text(alternate_instruction),
    }

    controller = QKVController(model)
    controller.install()
    try:
        controller.capture = True
        clean_a = controlled_infer(
            policy, source["image"], source["context"], source["context_mask"],
            source["context"], source["context_mask"], seed=seed, controller=controller
        )
        source_cache = controller.cache
        controller.cache = {"video": {}, "action": {}}
        clean_b = controlled_infer(
            policy, donor["image"], donor["context"], donor["context_mask"],
            donor["context"], donor["context_mask"], seed=seed, controller=controller
        )
        donor_cache = controller.cache
        controller.capture = False
        controller.cache = {"video": {}, "action": {}}

        assert torch.equal(clean_a["initial_video_noise"], clean_b["initial_video_noise"])
        assert torch.equal(clean_a["initial_action_noise"], clean_b["initial_action_noise"])
        assert torch.equal(clean_a["first_frame_latent"], clean_b["first_frame_latent"])

        conditions: dict[str, dict[str, Any]] = {"clean_a": clean_a, "clean_b": clean_b}
        settings = [
            ("identity", source_cache["video"], "semantic", LAYER_GROUPS["all"], source, source),
            ("world_all", donor_cache["video"], "semantic", LAYER_GROUPS["all"], source, source),
            ("shuffled_world", donor_cache["video"], "shuffled", LAYER_GROUPS["all"], source, source),
            ("world_early", donor_cache["video"], "semantic", LAYER_GROUPS["early"], source, source),
            ("world_middle", donor_cache["video"], "semantic", LAYER_GROUPS["middle"], source, source),
            ("world_late", donor_cache["video"], "semantic", LAYER_GROUPS["late"], source, source),
        ]
        debug_rows: list[dict[str, Any]] = []
        for condition, patch_cache, mode, layers, video_input, action_input in settings:
            controller.patch_cache = patch_cache
            controller.patch_modality = "video"
            controller.patch_layers = set(layers)
            controller.patch_temporal_groups = (0, 1, 2)
            controller.patch_components = "kv"
            controller.patch_mode = mode
            controller.random_seed = seed + 991
            controller.events = []
            conditions[condition] = controlled_infer(
                policy, source["image"], video_input["context"], video_input["context_mask"],
                action_input["context"], action_input["context_mask"], seed=seed, controller=controller
            )
            debug_rows.append(
                {
                    "condition": condition,
                    "interface": "per-layer video K/V before mixed attention",
                    "mode": mode,
                    "layers": json.dumps(list(layers)),
                    "temporal_groups": "[0, 1, 2]",
                    "components": "kv",
                    "patch_event_count": len(layers) * policy.num_inference_steps,
                    "source_rgb_sha256": source_hashes["rgb"],
                    "source_proprio_sha256": source_hashes["proprio"],
                    "source_action_instruction_sha256": source_hashes["instruction"],
                    "video_noise_sha256": tensor_sha256(clean_a["initial_video_noise"]),
                    "action_noise_sha256": tensor_sha256(clean_a["initial_action_noise"]),
                    "video_schedule_sha256": tensor_sha256(clean_a["video_timesteps"]),
                    "action_schedule_sha256": tensor_sha256(clean_a["action_timesteps"]),
                    "parameters_unchanged": parameter_signature(model) == params_before,
                }
            )
        controller.patch_cache = None
        # The mask makes this a genuine separable route: video remains source/A,
        # while only the action expert receives the alternate direct context.
        conditions["context_only"] = controlled_infer(
            policy, source["image"], source["context"], source["context_mask"],
            donor["context"], donor["context_mask"], seed=seed, controller=controller
        )
    finally:
        controller.uninstall()

    identity_action = torch.equal(conditions["identity"]["action"], clean_a["action"])
    identity_future = torch.equal(conditions["identity"]["video_latent"], clean_a["video_latent"])
    if not identity_action or not identity_future:
        write_json(pair_dir / "logs/identity_failure.json", {
            "action_bit_exact": identity_action,
            "future_bit_exact": identity_future,
            "action_max_abs": float((conditions["identity"]["action"] - clean_a["action"]).abs().max()),
            "future_max_abs": float((conditions["identity"]["video_latent"] - clean_a["video_latent"]).abs().max()),
        })
        raise SystemExit("STOP: identity replacement was not bit-exact")

    official_parity: dict[str, Any] = {"tested": False}
    if args.verify_official:
        official = model.infer_joint(
            prompt=None, context=source["context"], context_mask=source["context_mask"],
            input_image=source["image"], proprio=None,
            num_video_frames=policy.num_video_frames, action_horizon=policy.action_horizon,
            num_inference_steps=policy.num_inference_steps, seed=seed,
            rand_device=str(policy.cfg.EVALUATION.rand_device), tiled=bool(policy.cfg.EVALUATION.tiled),
            test_action_with_infer_action=False,
        )
        official_parity = {
            "tested": True,
            "action_bit_exact": torch.equal(official["action"], clean_a["action"]),
            "action_max_abs": float((official["action"] - clean_a["action"]).abs().max()),
        }
        if not official_parity["action_bit_exact"]:
            raise AssertionError(f"Controlled loop differs from official loop: {official_parity}")

    metrics: list[dict[str, Any]] = []
    future_a = clean_a["video_latent"][:, :, 1:]
    future_b = clean_b["video_latent"][:, :, 1:]
    for name, result in conditions.items():
        action_metrics = transfer_metrics(clean_a["action"], clean_b["action"], result["action"], "action")
        future_metrics = transfer_metrics(future_a, future_b, result["video_latent"][:, :, 1:], "future")
        row = {
            "task": args.task,
            "state": args.state,
            "pair_id": args.pair_id,
            "condition": name,
            **action_metrics,
            **future_metrics,
            "dissociation_action_minus_future": action_metrics["action_transfer"] - future_metrics["future_transfer"],
            "action_bit_exact_vs_a": torch.equal(result["action"], clean_a["action"]),
            "future_bit_exact_vs_a": torch.equal(result["video_latent"], clean_a["video_latent"]),
        }
        metrics.append(row)
        np.savez_compressed(pair_dir / "actions" / f"{name}.npz", action=result["action"].numpy())
        save_representation(pair_dir / "future_latents" / f"{name}.pt", result["video_latent"])

    source_manifest = cache_manifest(source_cache)
    donor_manifest = cache_manifest(donor_cache)
    write_csv(pair_dir / "representations/source_qkv_manifest.csv", source_manifest)
    write_csv(pair_dir / "representations/donor_qkv_manifest.csv", donor_manifest)
    write_csv(
        pair_dir / "representations/goal_conditioned_qkv_difference.csv",
        representation_difference_rows(source_cache, donor_cache),
    )
    if args.phase == "smoke" or args.save_full_qkv:
        save_cache(pair_dir / "representations/source_qkv.pt", source_cache)
        save_cache(pair_dir / "representations/donor_qkv.pt", donor_cache)
    write_csv(pair_dir / "metrics.csv", metrics)
    write_csv(pair_dir / "intervention_debug.csv", debug_rows)

    for name, result in conditions.items():
        save_video(policy, result["video_latent"], pair_dir / "videos" / f"{name}_predicted_future.mp4")

    if parameter_signature(model) != params_before:
        raise AssertionError("Pretrained parameter objects or versions changed")
    if file_sha256(checkpoint) != checkpoint_hash:
        raise AssertionError("Checkpoint bytes changed")
    write_json(pair_dir / "logs/run.json", {
        "suite": "libero_object",
        "task": args.task,
        "state": args.state,
        "pair_id": args.pair_id,
        "phase": args.phase,
        "seed": seed,
        "source_instruction": task.language,
        "alternate_instruction": alternate_instruction,
        "source_target": TARGETS[args.task],
        "same_rgb": source_hashes["rgb"] == donor_hashes["rgb"],
        "same_proprio": source_hashes["proprio"] == donor_hashes["proprio"],
        "source_hashes": source_hashes,
        "donor_hashes": donor_hashes,
        "shared_noise": {
            "video": tensor_summary(clean_a["initial_video_noise"]),
            "action": tensor_summary(clean_a["initial_action_noise"]),
        },
        "schedules": {
            "video_timesteps": tensor_summary(clean_a["video_timesteps"]),
            "video_deltas": tensor_summary(clean_a["video_deltas"]),
            "action_timesteps": tensor_summary(clean_a["action_timesteps"]),
            "action_deltas": tensor_summary(clean_a["action_deltas"]),
        },
        "identity": {"action_bit_exact": identity_action, "future_bit_exact": identity_future},
        "official_parity": official_parity,
        "checkpoint": {"path": checkpoint, "sha256": checkpoint_hash},
        "parameters_unchanged": True,
        "elapsed_seconds": time.time() - started,
    })
    print(json.dumps(jsonable({"pair_dir": pair_dir, "metrics": metrics, "official_parity": official_parity})), flush=True)


if __name__ == "__main__":
    main()

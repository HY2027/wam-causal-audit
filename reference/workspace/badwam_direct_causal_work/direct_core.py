from __future__ import annotations
from wam_causal_audit.paths import resolve as _release_path

import hashlib
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

REPO = Path(_release_path('@DATA@/BadWAM'))
IDM_AUDIT = Path(_release_path('@WORKSPACE@/badwam_idm_causal_audit_work'))
IDM_WORK = Path(_release_path('@WORKSPACE@/badwam_idm_same_frame_goal_work'))
for extra in (
    REPO, REPO / "src", REPO / "experiments/libero", REPO / "experiments/first_grasp_lock",
    IDM_AUDIT, IDM_WORK, Path(_release_path('@WORKSPACE@/LIBERO')),
    Path(_release_path('@WORKSPACE@/evaluation/d2_ghost_pilot')),
):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import run_first_grasp_lock as fgl  # noqa: E402
from imagination_intervention import tensor_sha256, tensor_summary  # noqa: E402
from run_smoke import prepare  # noqa: E402


LAYER_GROUPS = {
    "early": tuple(range(0, 8)),
    "middle": tuple(range(8, 22)),
    "late": tuple(range(22, 30)),
    "all": tuple(range(30)),
}


def hash_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def action_noise(policy: Any, seed: int) -> torch.Tensor:
    generator = torch.Generator(device=str(policy.cfg.EVALUATION.rand_device)).manual_seed(seed)
    return torch.randn(
        (1, policy.action_horizon, policy.model.action_expert.action_dim),
        generator=generator,
        device=str(policy.cfg.EVALUATION.rand_device),
        dtype=torch.float32,
    ).to(policy.model.device, policy.model.torch_dtype)


@torch.no_grad()
def build_current_cache(policy: Any, prepared: Mapping[str, Any]) -> dict[str, Any]:
    model = policy.model
    image = prepared["image"].to(model.device, model.torch_dtype)
    first_frame = model._encode_input_image_latents_tensor(
        input_image=image, tiled=bool(policy.cfg.EVALUATION.tiled)
    )
    timestep = torch.zeros((1,), dtype=first_frame.dtype, device=model.device)
    pre = model.video_expert.pre_dit(
        x=first_frame,
        timestep=timestep,
        context=prepared["context"],
        context_mask=prepared["context_mask"],
        action=None,
        fuse_vae_embedding_in_latents=bool(getattr(model.video_expert, "fuse_vae_embedding_in_latents", False)),
    )
    video_len = int(pre["tokens"].shape[1])
    tokens_per_group = int(pre["meta"]["tokens_per_frame"])
    mask = model._build_mot_attention_mask(
        video_seq_len=video_len,
        action_seq_len=policy.action_horizon,
        video_tokens_per_frame=tokens_per_group,
        device=pre["tokens"].device,
    )
    cache = model.mot.prefill_video_cache(
        video_tokens=pre["tokens"],
        video_freqs=pre["freqs"],
        video_t_mod=pre["t_mod"],
        video_context_payload={"context": pre["context"], "mask": pre["context_mask"]},
        video_attention_mask=mask[:video_len, :video_len],
    )
    return {
        "first_frame_latent": first_frame.detach(),
        "video_pre_tokens": pre["tokens"].detach(),
        "cache": cache,
        "attention_mask": mask,
        "video_seq_len": video_len,
        "tokens_per_group": tokens_per_group,
        "grid_size": tuple(int(value) for value in pre["meta"]["grid_size"]),
    }


def patch_cache(
    source: Sequence[Mapping[str, torch.Tensor]],
    donor: Sequence[Mapping[str, torch.Tensor]],
    *,
    layers: Sequence[int],
    mode: str,
    components: str = "kv",
    seed: int = 0,
) -> tuple[list[dict[str, torch.Tensor]], list[dict[str, Any]]]:
    if mode not in {"identity", "semantic", "shuffled"}:
        raise ValueError(mode)
    if components not in {"k", "v", "kv"}:
        raise ValueError(components)
    selected = set(int(layer) for layer in layers)
    result = [{kind: value.detach().clone() for kind, value in row.items()} for row in source]
    debug = []
    for layer in sorted(selected):
        for kind in (("k", "v") if components == "kv" else (components,)):
            replacement = donor[layer][kind].detach().clone()
            permutation = None
            if mode == "shuffled":
                generator = torch.Generator(device="cpu").manual_seed(seed + layer * 101 + (0 if kind == "k" else 1))
                permutation = torch.randperm(replacement.shape[1], generator=generator).to(replacement.device)
                replacement = replacement.index_select(1, permutation)
            result[layer][kind] = replacement
            debug.append({
                "layer": layer, "component": kind, "mode": mode,
                "source_sha256": tensor_sha256(source[layer][kind]),
                "donor_sha256": tensor_sha256(donor[layer][kind]),
                "injected_sha256": tensor_sha256(replacement),
                "permutation_sha256": None if permutation is None else tensor_sha256(permutation),
            })
    return result, debug


@torch.no_grad()
def action_from_cache(
    policy: Any,
    action_prepared: Mapping[str, Any],
    current: Mapping[str, Any],
    cache: Sequence[Mapping[str, torch.Tensor]],
    *,
    seed: int,
) -> dict[str, Any]:
    model = policy.model
    initial_noise = action_noise(policy, seed)
    latents = initial_noise.clone()
    timesteps, deltas = model.infer_action_scheduler.build_inference_schedule(
        policy.num_inference_steps, model.device, latents.dtype, shift_override=None
    )
    for timestep, delta in zip(timesteps, deltas):
        prediction = model._predict_action_noise_with_cache(
            latents_action=latents,
            timestep_action=timestep.unsqueeze(0).to(latents),
            context=action_prepared["context"],
            context_mask=action_prepared["context_mask"],
            video_kv_cache=list(cache),
            attention_mask=current["attention_mask"],
            video_seq_len=int(current["video_seq_len"]),
        )
        latents = model.infer_action_scheduler.step(prediction, delta, latents)
    return {
        "action": latents[0].detach().cpu().float(),
        "initial_action_noise": initial_noise.detach(),
        "timesteps": timesteps.detach(),
        "deltas": deltas.detach(),
    }


def transfer_metrics(source: torch.Tensor, donor: torch.Tensor, patched: torch.Tensor) -> dict[str, float]:
    def one(a: torch.Tensor, b: torch.Tensor, p: torch.Tensor) -> tuple[float, float, float]:
        donor_delta = (b - a).reshape(-1).double()
        patch_delta = (p - a).reshape(-1).double()
        donor_norm = float(torch.linalg.vector_norm(donor_delta))
        patch_norm = float(torch.linalg.vector_norm(patch_delta))
        dot = float(torch.dot(patch_delta, donor_delta))
        return patch_norm, dot / (donor_norm * donor_norm + 1e-12), 0.0 if donor_norm == 0 or patch_norm == 0 else dot / (donor_norm * patch_norm)

    full_l2, full_transfer, full_cosine = one(source, donor, patched)
    first_l2, first_transfer, first_cosine = one(source[0], donor[0], patched[0])
    eef_l2, eef_transfer, eef_cosine = one(source[0, :3], donor[0, :3], patched[0, :3])
    return {
        "action_l2": full_l2, "transfer": full_transfer, "cosine": full_cosine,
        "first_step_action_l2": first_l2, "first_step_transfer": first_transfer,
        "first_step_cosine": first_cosine, "eef_action_l2": eef_l2,
        "eef_transfer": eef_transfer, "eef_cosine": eef_cosine,
    }


def cache_difference(source: Sequence[Mapping[str, torch.Tensor]], donor: Sequence[Mapping[str, torch.Tensor]]) -> list[dict[str, Any]]:
    rows = []
    for layer, (a_row, b_row) in enumerate(zip(source, donor)):
        for kind in ("k", "v"):
            a = a_row[kind].double()
            b = b_row[kind].double()
            delta = b - a
            a_norm = float(torch.linalg.vector_norm(a))
            b_norm = float(torch.linalg.vector_norm(b))
            rows.append({
                "layer": layer, "component": kind,
                "source": tensor_summary(a_row[kind]), "donor": tensor_summary(b_row[kind]),
                "difference_l2": float(torch.linalg.vector_norm(delta)),
                "relative_difference": float(torch.linalg.vector_norm(delta)) / (a_norm + 1e-12),
                "source_donor_cosine": float(torch.sum(a * b)) / (a_norm * b_norm + 1e-12),
                "bit_exact": torch.equal(a_row[kind], b_row[kind]),
            })
    return rows


def cache_cpu(cache: Sequence[Mapping[str, torch.Tensor]]) -> list[dict[str, torch.Tensor]]:
    return [{kind: value.detach().cpu().to(torch.float16) for kind, value in row.items()} for row in cache]


def finite(value: torch.Tensor) -> bool:
    return bool(torch.isfinite(value).all())


def normalized_to_env(action: torch.Tensor, processor: Any) -> np.ndarray:
    processed = fgl._denormalize_action(action, processor)[0]
    processed[..., -1] = processed[..., -1] * 2 - 1
    processed = fgl.invert_gripper_action(processed)
    processed[..., -1] = np.sign(processed[..., -1])
    return np.asarray(processed, dtype=np.float32)

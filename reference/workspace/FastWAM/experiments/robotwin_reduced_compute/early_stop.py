from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping

import torch

from common import EXPECTED_LAYERS, EXPECTED_STEPS, tensor_sha


def cache_sha(cache: list[dict[str, torch.Tensor]]) -> str:
    digest = hashlib.sha256()
    for layer, row in enumerate(cache):
        digest.update(str(layer).encode())
        for component in ("k", "v"):
            digest.update(tensor_sha(row[component]).encode())
    return digest.hexdigest()


@dataclass
class RunOutput:
    action: torch.Tensor
    normalized_action_hash: str
    world_steps: int
    action_steps: int
    world_layer_calls: int
    action_layer_calls: int
    initial_world_noise_hash: str
    initial_action_noise_hash: str
    cache_events: list[dict[str, Any]]
    captured: dict[tuple[int, int, str], torch.Tensor]
    stop_cache_hash: str
    latency_seconds: float
    peak_memory_bytes: int


class JointCacheController:
    """Audited post-RoPE K/post-projection V capture/strict assignment.

    The controller operates only on video-expert events.  For K=5 there are
    exactly 5*30 video events; later action-only steps consume the complete
    cache produced by event group world step 5 of this same call.
    """

    def __init__(
        self,
        model: Any,
        expected_world_steps: int,
        current_bank: Mapping[tuple[int, int, str], torch.Tensor] | None = None,
        future_bank: Mapping[tuple[int, int, str], torch.Tensor] | None = None,
        current_label: str = "NATURAL_CURRENT",
        future_label: str = "NATURAL_FUTURE",
    ):
        self.model = model
        self.expected_world_steps = expected_world_steps
        self.current_bank = current_bank
        self.future_bank = future_bank
        self.current_label = current_label
        self.future_label = future_label
        self.original_build = model.mot._build_expert_attention_io
        self.original_mix = model.mot._mixed_attention
        self.block_map = {id(block): ("video", i) for i, block in enumerate(model.video_expert.blocks)}
        self.block_map.update({id(block): ("action", i) for i, block in enumerate(model.action_expert.blocks)})
        self.video_calls = 0
        self.action_calls = 0
        self.mix_calls = 0
        self.captured: dict[tuple[int, int, str], torch.Tensor] = {}
        self.events: list[dict[str, Any]] = []
        self.pending: dict[str, Any] | None = None
        self.cleanup = False

    def install(self) -> None:
        outer = self

        def build(expert: Any, block: Any, *args: Any, **kwargs: Any):
            output = list(outer.original_build(expert, block, *args, **kwargs))
            modality, layer = outer.block_map[id(block)]
            if modality == "action":
                outer.action_calls += 1
                return tuple(output)
            step, expected_layer = divmod(outer.video_calls, EXPECTED_LAYERS)
            if step >= outer.expected_world_steps or layer != expected_layer:
                raise RuntimeError(f"VIDEO_EVENT_ORDER:{step}:{expected_layer}:{layer}")
            outer.video_calls += 1
            length = int(output[1].shape[1])
            if length % 3:
                raise RuntimeError(f"VIDEO_TEMPORAL_LAYOUT:{length}")
            group = length // 3
            expected: dict[str, str] = {}
            for component, slot in (("k", 1), ("v", 2)):
                key = (step, layer, component)
                natural = output[slot]
                if outer.current_bank is None and outer.future_bank is None:
                    written = natural
                else:
                    if outer.current_bank is None or outer.future_bank is None:
                        raise RuntimeError("STRICT_REQUIRES_BOTH_BANKS")
                    current = outer.current_bank[key].to(natural)[:, :group]
                    future = outer.future_bank[key].to(natural)[:, group:]
                    written = torch.cat((current, future), dim=1)
                    output[slot] = written
                outer.captured[key] = written.detach().cpu().clone()
                expected[component] = tensor_sha(written)
                outer.events.append({
                    "world_step_zero_based": step,
                    "world_step_one_based": step + 1,
                    "layer": layer,
                    "component": component,
                    "location": "_build_expert_attention_io return; post-RoPE K/post-projection V; before _mixed_attention",
                    "natural_hash": tensor_sha(natural),
                    "written_hash": expected[component],
                    "current_source": outer.current_label,
                    "future_source": outer.future_label,
                    "tokens_per_temporal_group": group,
                })
            outer.pending = {"length": length, "hashes": expected, "step": step, "layer": layer}
            return tuple(output)

        def mixed(*args: Any, **kwargs: Any):
            if args:
                raise RuntimeError("MIXED_ATTENTION_POSITIONAL_SIGNATURE")
            if outer.pending is None:
                # K<10 later steps enter MoT.forward_action_with_video_cache.
                # Their mixed attention contains action queries plus explicitly
                # supplied cached video K/V, but no video-expert build event.
                # Cache provenance for these reads is logged by infer_budget;
                # the original mixed-attention implementation must run here.
                return outer.original_mix(**kwargs)
            pending = outer.pending
            for component in ("k", "v"):
                value = kwargs[component + "_cat"][:, : pending["length"]]
                if tensor_sha(value) != pending["hashes"][component]:
                    raise RuntimeError(f"WRITTEN_VALUE_NOT_CONSUMED:{pending['step']}:{pending['layer']}:{component}")
            outer.mix_calls += 1
            outer.pending = None
            return outer.original_mix(**kwargs)

        self.model.mot._build_expert_attention_io = build
        self.model.mot._mixed_attention = mixed

    def uninstall(self) -> None:
        self.model.mot._build_expert_attention_io = self.original_build
        self.model.mot._mixed_attention = self.original_mix
        self.cleanup = (
            self.model.mot._build_expert_attention_io == self.original_build
            and self.model.mot._mixed_attention == self.original_mix
        )

    def validate(self) -> None:
        if self.pending is not None:
            raise RuntimeError("UNCONSUMED_FINAL_EVENT")
        if self.video_calls != self.expected_world_steps * EXPECTED_LAYERS:
            raise RuntimeError(f"WORLD_LAYER_COUNT:{self.video_calls}")
        # Action expert runs in both coupled and cache-backed action-only
        # forwards, hence always ten complete layer stacks.
        if self.action_calls != EXPECTED_STEPS * EXPECTED_LAYERS:
            raise RuntimeError(f"JOINT_ACTION_LAYER_COUNT:{self.action_calls}")
        if self.mix_calls != self.expected_world_steps * EXPECTED_LAYERS:
            raise RuntimeError(f"MIX_COUNT:{self.mix_calls}")
        if not self.cleanup:
            raise RuntimeError("HOOK_CLEANUP_FAILED")

    def cache_at(self, step: int) -> list[dict[str, torch.Tensor]]:
        return [
            {component: self.captured[(step, layer, component)] for component in ("k", "v")}
            for layer in range(EXPECTED_LAYERS)
        ]


def _initialize(model: Any, image: torch.Tensor, action_horizon: int, num_video_frames: int, seed: int):
    _, _, height, width = image.shape
    latent_t = (num_video_frames - 1) // model.vae.temporal_downsample_factor + 1
    world_rng = torch.Generator(device="cpu").manual_seed(seed)
    action_rng = torch.Generator(device="cpu").manual_seed(seed)
    world_noise = torch.randn(
        (1, model.vae.model.z_dim, latent_t, height // model.vae.upsampling_factor, width // model.vae.upsampling_factor),
        generator=world_rng,
        device="cpu",
        dtype=torch.float32,
    )
    action_noise = torch.randn(
        (1, action_horizon, model.action_expert.action_dim),
        generator=action_rng,
        device="cpu",
        dtype=torch.float32,
    )
    video = world_noise.to(model.device, model.torch_dtype)
    action = action_noise.to(model.device, model.torch_dtype)
    first = model._encode_input_image_latents_tensor(image.to(model.device, model.torch_dtype), tiled=False)
    video[:, :, 0:1] = first.clone()
    return video, action, first, tensor_sha(world_noise), tensor_sha(action_noise)


@torch.no_grad()
def infer_budget(
    model: Any,
    image: torch.Tensor,
    proprio: torch.Tensor,
    prompt: str,
    seed: int,
    budget: int,
    current_bank: Mapping[tuple[int, int, str], torch.Tensor] | None = None,
    future_bank: Mapping[tuple[int, int, str], torch.Tensor] | None = None,
    current_label: str = "NATURAL_CURRENT",
    future_label: str = "NATURAL_FUTURE",
    policy_call_id: str = "UNREGISTERED_CALL",
) -> RunOutput:
    if budget not in (5, 10):
        raise ValueError(f"BUDGET_NOT_REGISTERED:{budget}")
    model.eval()
    # Deployment-comparable policy/model boundary: preprocessing has already
    # produced image/proprio, while prompt/proprio context construction, image
    # latent encoding, model computation and cache handling are all included.
    torch.cuda.reset_peak_memory_stats(model.device)
    torch.cuda.synchronize(model.device)
    start = __import__("time").perf_counter()
    context, context_mask = model.encode_prompt(prompt)
    context, context_mask = model._append_proprio_to_context(
        context=context,
        context_mask=context_mask,
        proprio=proprio.to(model.device, model.torch_dtype),
    )
    video, action, first, world_noise_hash, action_noise_hash = _initialize(model, image, 32, 9, seed)
    vts, vds = model.infer_video_scheduler.build_inference_schedule(10, model.device, video.dtype, shift_override=None)
    ats, ads = model.infer_action_scheduler.build_inference_schedule(10, model.device, action.dtype, shift_override=None)
    controller = JointCacheController(
        model,
        expected_world_steps=budget,
        current_bank=current_bank,
        future_bank=future_bank,
        current_label=current_label,
        future_label=future_label,
    )
    controller.install()
    attention_mask = None
    video_seq_len = None
    action_steps = 0
    late_events: list[dict[str, Any]] = []
    try:
        for step, (tv, dv, ta, da) in enumerate(zip(vts, vds, ats, ads)):
            if step < budget:
                video_pre = model.video_expert.pre_dit(
                    x=video,
                    timestep=tv.unsqueeze(0).to(video),
                    context=context,
                    context_mask=context_mask,
                    action=None,
                    fuse_vae_embedding_in_latents=bool(getattr(model.video_expert, "fuse_vae_embedding_in_latents", False)),
                )
                action_pre = model.action_expert.pre_dit(
                    action_tokens=action,
                    timestep=ta.unsqueeze(0).to(action),
                    context=context,
                    context_mask=context_mask,
                )
                video_seq_len = int(video_pre["tokens"].shape[1])
                attention_mask = model._build_mot_attention_mask(
                    video_seq_len,
                    int(action_pre["tokens"].shape[1]),
                    int(video_pre["meta"]["tokens_per_frame"]),
                    video_pre["tokens"].device,
                )
                output = model.mot(
                    embeds_all={"video": video_pre["tokens"], "action": action_pre["tokens"]},
                    attention_mask=attention_mask,
                    freqs_all={"video": video_pre["freqs"], "action": action_pre["freqs"]},
                    context_all={
                        "video": {"context": video_pre["context"], "mask": video_pre["context_mask"]},
                        "action": {"context": action_pre["context"], "mask": action_pre["context_mask"]},
                    },
                    t_mod_all={"video": video_pre["t_mod"], "action": action_pre["t_mod"]},
                )
                pred_video = model.video_expert.post_dit(output["video"], video_pre)
                pred_action = model.action_expert.post_dit(output["action"], action_pre)
                video = model.infer_video_scheduler.step(pred_video, dv, video)
                video[:, :, 0:1] = first.clone()
            else:
                if attention_mask is None or video_seq_len is None:
                    raise RuntimeError("STOP_CACHE_CONTEXT_MISSING")
                cache = controller.cache_at(budget - 1)
                pred_action = model._predict_action_noise_with_cache(
                    latents_action=action,
                    timestep_action=ta.unsqueeze(0).to(action),
                    context=context,
                    context_mask=context_mask,
                    video_kv_cache=[{k: v.to(model.device) for k, v in row.items()} for row in cache],
                    attention_mask=attention_mask,
                    video_seq_len=video_seq_len,
                )
                for layer, row in enumerate(cache):
                    for component in ("k", "v"):
                        late_events.append({
                            "policy_call_id": policy_call_id,
                            "action_step_zero_based": step,
                            "action_step_one_based": step + 1,
                            "world_step_that_created_cache_one_based": budget,
                            "layer": layer,
                            "component": component,
                            "tensor_hash": tensor_sha(row[component]),
                            "same_call": True,
                        })
            action = model.infer_action_scheduler.step(pred_action, da, action)
            action_steps += 1
    finally:
        controller.uninstall()
    torch.cuda.synchronize(model.device)
    elapsed = __import__("time").perf_counter() - start
    controller.validate()
    stop_cache = controller.cache_at(budget - 1)
    events = [dict(event, policy_call_id=policy_call_id) for event in controller.events] + late_events
    return RunOutput(
        action=action[0].detach().cpu().float(),
        normalized_action_hash=tensor_sha(action[0]),
        world_steps=budget,
        action_steps=action_steps,
        world_layer_calls=controller.video_calls,
        action_layer_calls=controller.action_calls,
        initial_world_noise_hash=world_noise_hash,
        initial_action_noise_hash=action_noise_hash,
        cache_events=events,
        captured=controller.captured,
        stop_cache_hash=cache_sha(stop_cache),
        latency_seconds=elapsed,
        peak_memory_bytes=int(torch.cuda.max_memory_allocated(model.device)),
    )

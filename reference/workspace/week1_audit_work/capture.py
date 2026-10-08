from __future__ import annotations
from wam_causal_audit.paths import resolve as _release_path

import hashlib
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from protocol import mean_channel, pool_hidden, pool_latent, pool_qkv_rows, tensor_sha256


BADWAM = Path(_release_path('@DATA@/BadWAM'))
IMAGEWAM = Path(_release_path('@WORKSPACE@/ImageWAM'))
for path in (
    Path(__file__).resolve().parent,
    Path(_release_path('@WORKSPACE@/badwam_direct_causal_work')),
    Path(_release_path('@WORKSPACE@/badwam_joint_causal_work')),
    Path(_release_path('@WORKSPACE@/badwam_idm_same_frame_goal_work')),
    Path(_release_path('@WORKSPACE@/badwam_idm_causal_audit_work')),
    Path(_release_path('@WORKSPACE@/cross_wam_language_rep_2x2_work')),
    BADWAM,
    BADWAM / "src",
    BADWAM / "experiments/libero",
    BADWAM / "experiments/first_grasp_lock",
    Path(_release_path('@WORKSPACE@/LIBERO')),
    Path(_release_path('@WORKSPACE@/evaluation/d2_ghost_pilot')),
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _feature_hash(features: Mapping[str, np.ndarray | None]) -> str:
    digest = hashlib.sha256()
    for name in sorted(features):
        digest.update(name.encode())
        value = features[name]
        digest.update(b"LOCUS_NOT_PRESENT" if value is None else tensor_sha256(value).encode())
    return digest.hexdigest()


class BadAttentionTrace:
    """Capture exact attention-path tensors without changing model outputs."""

    def __init__(self, model: Any, *, keep_full: bool = False) -> None:
        self.model = model
        self.keep_full = keep_full
        self.original = model.mot._build_expert_attention_io
        self.num_heads = int(model.video_expert.num_heads)
        self.block_map = {id(block): ("video", index) for index, block in enumerate(model.video_expert.blocks)}
        self.block_map.update({id(block): ("action", index) for index, block in enumerate(model.action_expert.blocks)})
        self.occurrence: dict[tuple[str, int], int] = defaultdict(int)
        self.rows: dict[tuple[str, int, int], dict[str, torch.Tensor]] = {}
        # Small temporal-slice pools used by the frozen 51-locus registry.
        # These preserve head-channel features without retaining full video
        # tensors for every denoising step.
        self.temporal_rows: dict[tuple[str, int, int], dict[str, dict[str, torch.Tensor]]] = {}
        self.hashes: dict[tuple[str, int, int, str], str] = {}

    def install(self) -> None:
        def wrapped(expert: Any, block: Any, *args: Any, **kwargs: Any):
            result = self.original(expert, block, *args, **kwargs)
            modality, layer = self.block_map[id(block)]
            occurrence = self.occurrence[(modality, layer)]
            self.occurrence[(modality, layer)] += 1
            values = {"q": result[0], "k": result[1], "v": result[2], "hidden": result[3]}
            key = (modality, occurrence, layer)
            self.rows[key] = {}
            self.temporal_rows[key] = {}
            for name, value in values.items():
                if self.keep_full:
                    stored = value.detach().to("cpu")
                elif name in {"q", "k", "v"} and value.ndim == 3 and value.shape[-1] % self.num_heads == 0:
                    shaped = value.detach().float().reshape(value.shape[0], value.shape[1], self.num_heads, value.shape[-1] // self.num_heads)
                    stored = shaped.mean(dim=(0, 1, 2)).cpu()
                else:
                    stored = mean_channel(value, channel_axis=-1)
                self.rows[key][name] = stored
                if (
                    modality == "video" and name in {"k", "v"} and value.ndim == 3
                    and value.shape[1] % 3 == 0 and value.shape[-1] % self.num_heads == 0
                ):
                    tokens_per_group = value.shape[1] // 3
                    self.temporal_rows[key][name] = {}
                    for temporal_name, start, stop in (
                        ("current", 0, tokens_per_group),
                        ("future", tokens_per_group, 3 * tokens_per_group),
                    ):
                        selected = value[:, start:stop].detach().float().reshape(
                            value.shape[0], stop - start, self.num_heads, value.shape[-1] // self.num_heads
                        )
                        self.temporal_rows[key][name][temporal_name] = selected.mean(dim=(0, 1, 2)).cpu()
            for name, value in values.items():
                self.hashes[(modality, occurrence, layer, name)] = tensor_sha256(value)
            return result

        self.model.mot._build_expert_attention_io = wrapped

    def uninstall(self) -> None:
        self.model.mot._build_expert_attention_io = self.original

    def records(self, modality: str, layers: set[int] | None = None) -> list[dict[str, torch.Tensor]]:
        output = []
        for (kind, occurrence, layer), row in sorted(self.rows.items(), key=lambda item: (item[0][1], item[0][2])):
            if kind == modality and (layers is None or layer in layers):
                output.append(row)
        return output

    def mapping_kv_hash(self, modality: str) -> str:
        """Match the validated adapter's cache_sequence_sha byte protocol.

        The historical hash concatenates per-tensor SHA strings in
        step/layer insertion order and does not include labels.
        """
        digest = hashlib.sha256()
        keys = sorted((occurrence, layer) for kind, occurrence, layer in self.rows if kind == modality)
        for occurrence, layer in keys:
            for component in ("k", "v"):
                digest.update(self.hashes[(modality, occurrence, layer, component)].encode())
        return digest.hexdigest()


def _sequence_cache_hash(cache: list[dict[str, torch.Tensor]]) -> str:
    """Match counterfactual_empty_location_work.cache_sequence_sha."""
    digest = hashlib.sha256()
    for row in cache:
        for component in ("k", "v"):
            digest.update(tensor_sha256(row[component]).encode())
    return digest.hexdigest()


class BadWAMCapture:
    def __init__(self, model_name: str, gpu_id: int) -> None:
        import run_first_grasp_lock as fgl
        from common import create_policy
        self.name = model_name
        self.policy = create_policy(gpu_id) if model_name == "idm" else fgl.BadWAMPolicy(model_name, gpu_id)
        self.model = self.policy.model
        self.processor = self.policy.processor
        self.num_heads = int(self.model.video_expert.num_heads)

    def normalized_to_env(self, action: torch.Tensor) -> np.ndarray:
        from common import normalized_to_env
        return normalized_to_env(action, self.processor)

    def _prepared(self, observation: Mapping[str, Any], instruction: str):
        from direct_core import prepare
        return prepare(self.policy, observation, instruction)

    def capture(self, observation: Mapping[str, Any], instruction: str, seed: int, *, keep_full: bool = False) -> dict[str, Any]:
        if self.name == "direct":
            return self._direct(observation, instruction, seed, keep_full=keep_full)
        if self.name == "joint":
            return self._joint(observation, instruction, seed, keep_full=keep_full)
        return self._idm(observation, instruction, seed, keep_full=keep_full)

    def _direct(self, observation, instruction, seed, *, keep_full):
        from direct_core import action_from_cache, build_current_cache
        prepared = self._prepared(observation, instruction)
        trace = BadAttentionTrace(self.model, keep_full=keep_full); trace.install()
        try:
            current = build_current_cache(self.policy, prepared)
            result = action_from_cache(self.policy, prepared, current, current["cache"], seed=seed)
        finally:
            trace.uninstall()
        cache = current["cache"]
        if keep_full:
            features = None
        else:
            features = {
                "L1": pool_qkv_rows(cache[0:10], num_heads=self.num_heads),
                "L2": pool_qkv_rows(cache[10:20], num_heads=self.num_heads),
                "L3": pool_qkv_rows(cache[20:30], num_heads=self.num_heads),
                "L4": pool_qkv_rows(cache[11:15], num_heads=self.num_heads),
                "L5": pool_hidden([row["hidden"] for row in trace.records("action")]),
                "L6": None,
            }
        return {
            "action": result["action"], "features": features,
            "feature_bundle_sha256": None if features is None else _feature_hash(features),
            "representation_sha256": _sequence_cache_hash(cache), "trace": trace,
            "current": current,
            "context": prepared["context"].detach().cpu(), "context_mask": prepared["context_mask"].detach().cpu(),
            "proprio": prepared["proprio"].detach().cpu(), "locus_not_present": ["L6"],
        }

    def _joint(self, observation, instruction, seed, *, keep_full):
        from run_joint_smoke_pair import controlled_infer
        prepared = self._prepared(observation, instruction)
        trace = BadAttentionTrace(self.model, keep_full=keep_full); trace.install()
        try:
            result = controlled_infer(
                self.policy, prepared["image"], prepared["context"], prepared["context_mask"],
                prepared["context"], prepared["context_mask"], seed=seed, controller=None,
            )
        finally:
            trace.uninstall()
        video = trace.records("video")
        action = trace.records("action")
        if keep_full:
            features = None
        else:
            features = {
                "L1": pool_qkv_rows([row for key, row in trace.rows.items() if key[0] == "video" and key[2] in set(range(0, 10))], num_heads=self.num_heads),
                "L2": pool_qkv_rows([row for key, row in trace.rows.items() if key[0] == "video" and key[2] in set(range(10, 20))], num_heads=self.num_heads),
                "L3": pool_qkv_rows([row for key, row in trace.rows.items() if key[0] == "video" and key[2] in set(range(20, 30))], num_heads=self.num_heads),
                "L4": pool_qkv_rows(action, components=("q", "k", "v"), num_heads=self.num_heads),
                "L5": pool_hidden([row["hidden"] for row in action]),
                "L6": None,
            }
        return {
            "action": result["action"], "features": features,
            "feature_bundle_sha256": None if features is None else _feature_hash(features),
            "representation_sha256": trace.mapping_kv_hash("video"), "trace": trace,
            "joint_result": result,
            "context": prepared["context"].detach().cpu(), "context_mask": prepared["context_mask"].detach().cpu(),
            "proprio": prepared["proprio"].detach().cpu(), "locus_not_present": ["L6"],
        }

    def _idm(self, observation, instruction, seed, *, keep_full):
        from common import action_from_latent, action_noise, generate_video_latent
        prepared = self._prepared(observation, instruction)
        video = generate_video_latent(self.policy, prepared, seed=seed)
        trace = BadAttentionTrace(self.model, keep_full=keep_full); trace.install()
        try:
            result = action_from_latent(self.policy, prepared, video["latent"], action_noise(self.policy, seed), capture_cache=True)
        finally:
            trace.uninstall()
        cache = result["cache"]
        if keep_full:
            features = None
        else:
            features = {
                "L1": pool_latent(video["latent"], (0,)),
                "L2": pool_latent(video["latent"], (1,)),
                "L3": pool_latent(video["latent"], (2,)),
                "L4": pool_latent(video["latent"], (1, 2)),
                "L5": pool_qkv_rows(cache, num_heads=self.num_heads),
                "L6": mean_channel(prepared["context"], channel_axis=-1).numpy().astype(np.float32),
                "L7": None,
            }
        return {
            "action": result["action"], "features": features,
            "feature_bundle_sha256": None if features is None else _feature_hash(features),
            "representation_sha256": tensor_sha256(video["latent"][:, :, 1:]), "trace": trace,
            "generated_latent": video["latent"].detach().cpu(),
            "video_cache": cache,
            "context": prepared["context"].detach().cpu(), "context_mask": prepared["context_mask"].detach().cpu(),
            "proprio": prepared["proprio"].detach().cpu(), "locus_not_present": ["L7"],
        }


class ImageActionTrace:
    def __init__(self, model: Any, *, keep_full: bool = False) -> None:
        self.model = model; self.keep_full = keep_full
        self.rows: dict[tuple[int, int], dict[str, torch.Tensor]] = {}
        self.inputs: list[torch.Tensor] = []
        self._original_blocks: list[tuple[Any, Any]] = []
        self._original_pre = model.action_expert.pre_dit
        self.num_heads = int(getattr(model.action_expert, "num_heads", 24))
        self.step = -1

    def install(self) -> None:
        def pre_wrapped(*args, **kwargs):
            output = self._original_pre(*args, **kwargs)
            self.step += 1
            value = output["tokens"]
            self.inputs.append(value.detach().to("cpu") if self.keep_full else mean_channel(value, channel_axis=-1))
            return output
        self.model.action_expert.pre_dit = pre_wrapped
        blocks = list(self.model.action_expert.double_blocks) + list(self.model.action_expert.single_blocks)
        for layer, block in enumerate(blocks):
            original = block.prepare_qkv
            self._original_blocks.append((block, original))
            def wrapped(*args, __original=original, __layer=layer, **kwargs):
                output = __original(*args, **kwargs)
                self.rows[(self.step, __layer)] = {}
                for key, value in output.items():
                    if key not in {"q", "k", "v", "residual_x"} or not torch.is_tensor(value): continue
                    if self.keep_full:
                        stored = value.detach().to("cpu")
                    elif key in {"q", "k", "v"} and value.ndim == 3 and value.shape[-1] % self.num_heads == 0:
                        shaped = value.detach().float().reshape(value.shape[0], value.shape[1], self.num_heads, value.shape[-1] // self.num_heads)
                        stored = shaped.mean(dim=(0, 1, 2)).cpu()
                    else:
                        stored = mean_channel(value, channel_axis=-1)
                    self.rows[(self.step, __layer)][key] = stored
                return output
            block.prepare_qkv = wrapped

    def uninstall(self) -> None:
        self.model.action_expert.pre_dit = self._original_pre
        for block, original in self._original_blocks:
            block.prepare_qkv = original


class ImageWAMCapture:
    def __init__(self, gpu_id: int) -> None:
        for name in ("run_smoke", "run_one_call", "world_rep", "controlled_prefill", "run_libero_object_causal"):
            sys.modules.pop(name, None)
        for name in [key for key in sys.modules if key == "experiments" or key.startswith("experiments.")]:
            sys.modules.pop(name, None)
        from omegaconf import OmegaConf
        for resolver in ("eval", "max", "split"):
            if OmegaConf.has_resolver(resolver): OmegaConf.clear_resolver(resolver)
        for path in reversed((
            IMAGEWAM, IMAGEWAM / "src", IMAGEWAM / "experiments/libero", IMAGEWAM / "scripts/first_grasp_lock",
            IMAGEWAM / "scripts/causal_world_rep", IMAGEWAM / "scripts/same_frame_goal_kv",
            Path(_release_path('@WORKSPACE@/cross_wam_language_rep_2x2_work')),
        )):
            if str(path) in sys.path: sys.path.remove(str(path))
            sys.path.insert(0, str(path))
        from k2_imagewam_adapter import ImageWAMAdapter
        self.adapter = ImageWAMAdapter(gpu_id); self.model = self.adapter.model
        self.processor = self.adapter.processor; self.cfg = self.adapter.cfg
        self.num_heads = int(getattr(self.model.action_expert, "num_heads", 24))

    def normalized_to_env(self, action: torch.Tensor) -> np.ndarray:
        return self.adapter.normalized_to_env(action)

    def capture(self, observation: Mapping[str, Any], instruction: str, seed: int, *, keep_full: bool = False) -> dict[str, Any]:
        from imagewam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
        from run_libero_object_causal import model_input
        from world_rep import WorldRepIntervention
        from controlled_prefill import cache_sha256
        image, proprio = model_input(self.model, self.processor, self.cfg, observation)
        control = WorldRepIntervention(mode="capture")
        action_trace = ImageActionTrace(self.model, keep_full=keep_full); action_trace.install()
        try:
            output = self.model.infer_action(
                prompt=DEFAULT_PROMPT.format(task=instruction), input_image=image.to("cuda"), action_horizon=16,
                proprio=proprio, num_inference_steps=10, sigma_shift=None, seed=int(seed), rand_device="cpu",
                tiled=False, world_rep_intervention=control,
            )
        finally:
            action_trace.uninstall()
        cache = control.captured
        if cache is None: raise AssertionError("ImageWAM cache capture missing")
        layers = cache["double"] + cache["single"]
        start, stop = int(cache["txt_len"]), int(cache["txt_len"]) + int(cache["img_len"])
        image_rows = [{kind: row[kind][:, start:stop] for kind in ("k", "v")} for row in layers]
        prefix_rows = [{kind: row[kind][:, :start] for kind in ("k", "v")} for row in layers]
        all_rows = [{kind: row[kind] for kind in ("k", "v")} for row in layers]
        qkv_rows = list(action_trace.rows.values())
        if keep_full:
            features = None
        else:
            groups = (range(0, 6), range(6, 19), range(19, 25))
            features = {
                "L1": pool_qkv_rows([image_rows[index] for index in groups[0]], num_heads=self.num_heads),
                "L2": pool_qkv_rows([image_rows[index] for index in groups[1]], num_heads=self.num_heads),
                "L3": pool_qkv_rows([image_rows[index] for index in groups[2]], num_heads=self.num_heads),
                "L4": pool_qkv_rows(prefix_rows, num_heads=self.num_heads),
                "L5": pool_hidden(action_trace.inputs),
                "L6": pool_qkv_rows(all_rows, num_heads=self.num_heads),
            }
        return {
            "action": output["action"].detach().cpu().float(), "features": features,
            "feature_bundle_sha256": None if features is None else _feature_hash(features),
            "representation_sha256": cache_sha256(cache), "cache": cache, "action_trace": action_trace,
            "source_inputs": control.source_inputs, "proprio": proprio.detach().cpu(), "locus_not_present": [],
        }


def make_capture(model: str, gpu_id: int):
    # Step 0.2 frozen authority guard. CUDA_VISIBLE_DEVICES maps the selected
    # physical GPU to visible index 0 in all current experiment launchers.
    from hardware_authority import assert_hardware_authority
    assert_hardware_authority(model, cuda_index=0)
    return ImageWAMCapture(gpu_id) if model == "imagewam" else BadWAMCapture(model, gpu_id)


def tensor_difference(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    a = left.detach().float(); b = right.detach().float()
    if a.shape != b.shape: raise AssertionError(f"Tensor shape mismatch {tuple(a.shape)} != {tuple(b.shape)}")
    delta = b - a
    return {
        "shape": list(a.shape), "dtype_a": str(left.dtype), "dtype_b": str(right.dtype),
        "source_sha256": tensor_sha256(left), "donor_sha256": tensor_sha256(right),
        "max_absolute_difference": float(delta.abs().max()) if delta.numel() else 0.0,
        "relative_frobenius_difference": float(torch.linalg.vector_norm(delta) / (torch.linalg.vector_norm(a) + 1e-12)),
        "bit_exact": bool(torch.equal(left, right)),
    }

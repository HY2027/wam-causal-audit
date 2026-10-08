from __future__ import annotations
from wam_causal_audit.paths import resolve as _release_path

import hashlib
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch


WORK = Path(__file__).resolve().parent
PATHS = (
    WORK,
    Path(_release_path('@WORKSPACE@/week1_audit_work')),
    Path(_release_path('@WORKSPACE@/badwam_direct_causal_work')),
    Path(_release_path('@WORKSPACE@/badwam_joint_causal_work')),
    Path(_release_path('@WORKSPACE@/badwam_idm_same_frame_goal_work')),
    Path(_release_path('@WORKSPACE@/badwam_idm_causal_audit_work')),
)
for path in PATHS:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from protocol import tensor_sha256  # noqa: E402
from registry_features import model_loci  # noqa: E402


COMPONENT_SLOT = {"q": 0, "k": 1, "v": 2, "hidden": 3}


def clone_rows(rows: Sequence[Mapping[str, torch.Tensor]]) -> list[dict[str, torch.Tensor]]:
    return [{kind: value.detach().clone() for kind, value in row.items()} for row in rows]


def tensor_collection_hash(values: Iterable[torch.Tensor]) -> str:
    digest = hashlib.sha256()
    count = 0
    for value in values:
        digest.update(tensor_sha256(value).encode()); count += 1
    digest.update(str(count).encode())
    return digest.hexdigest()


def selected_loci(model: str, locus_ids: Iterable[str]) -> list[dict[str, Any]]:
    wanted = set(locus_ids)
    rows = [row for row in model_loci(model) if row["locus_id"] in wanted]
    if {row["locus_id"] for row in rows} != wanted:
        raise AssertionError(f"Unknown loci for {model}: {wanted - {row['locus_id'] for row in rows}}")
    return rows


class BadPatchTrace:
    """Capture/replace BadWAM mixed-attention IO at frozen registry loci."""

    def __init__(self, model: Any, model_name: str, *, capture: bool = False,
                 donor: "BadPatchTrace | None" = None, locus_ids: Sequence[str] = ()) -> None:
        self.model = model; self.model_name = model_name; self.capture = capture; self.donor = donor
        self.loci = selected_loci(model_name, locus_ids)
        self.original = model.mot._build_expert_attention_io
        self.block_map = {id(block): ("video", i) for i, block in enumerate(model.video_expert.blocks)}
        self.block_map.update({id(block): ("action", i) for i, block in enumerate(model.action_expert.blocks)})
        self.occurrence: dict[tuple[str, int], int] = defaultdict(int)
        self.rows: dict[tuple[str, int, int], dict[str, torch.Tensor]] = {}
        self.injected_rows: dict[tuple[str, int, int], dict[str, torch.Tensor]] = {}
        self.latents: dict[int, torch.Tensor] = {}
        self.injected_latents: dict[int, torch.Tensor] = {}
        self.events = 0

    def _rules(self, modality: str, layer: int, component: str) -> list[dict[str, Any]]:
        output = []
        for row in self.loci:
            family = row["family"]
            layer_range = row.get("layer_range")
            layer_ok = layer_range is None or int(layer_range[0]) <= layer <= int(layer_range[1])
            if not layer_ok:
                continue
            if modality == "video" and family == "VIDEO_WORLD_KV" and component in {"k", "v"}:
                output.append(row)
            elif modality == "action" and family == "ACTION_QKV" and component in {"q", "k", "v"}:
                output.append(row)
            elif modality == "action" and family == "ACTION_HIDDEN" and component == "hidden":
                output.append(row)
        return output

    def install(self) -> None:
        def wrapped(expert: Any, block: Any, *args: Any, **kwargs: Any):
            result = list(self.original(expert, block, *args, **kwargs))
            modality, layer = self.block_map[id(block)]
            occurrence = self.occurrence[(modality, layer)]
            self.occurrence[(modality, layer)] += 1
            key = (modality, occurrence, layer)
            if self.capture:
                self.rows[key] = {name: result[slot].detach().cpu().clone() for name, slot in COMPONENT_SLOT.items()}
            if self.donor is not None:
                for component, slot in COMPONENT_SLOT.items():
                    rules = self._rules(modality, layer, component)
                    if not rules:
                        continue
                    replacement = self.donor.rows[key][component].to(device=result[slot].device, dtype=result[slot].dtype)
                    changed = result[slot].clone()
                    for rule in rules:
                        temporal = rule.get("temporal_selection")
                        if modality == "video" and temporal in {"current", "future"}:
                            if changed.ndim != 3 or changed.shape[1] % 3:
                                raise AssertionError(f"Video temporal shape mismatch {tuple(changed.shape)}")
                            tpg = changed.shape[1] // 3
                            start, stop = (0, tpg) if temporal == "current" else (tpg, 3 * tpg)
                            changed[:, start:stop] = replacement[:, start:stop]
                        else:
                            changed = replacement.clone()
                        self.events += 1
                    result[slot] = changed
            if self.capture or self.donor is not None:
                self.injected_rows[key] = {
                    name: result[slot].detach().cpu().clone()
                    for name, slot in COMPONENT_SLOT.items()
                }
            return tuple(result)
        self.model.mot._build_expert_attention_io = wrapped

    def uninstall(self) -> None:
        self.model.mot._build_expert_attention_io = self.original

    def patch_joint_latent(self, value: torch.Tensor, step: int) -> torch.Tensor:
        if self.capture:
            self.latents[step] = value.detach().cpu().clone()
        if self.donor is None:
            if self.capture:
                self.injected_latents[step] = value.detach().cpu().clone()
            return value
        result = value
        for row in self.loci:
            if row["family"] != "VIDEO_LATENT":
                continue
            temporal = row["temporal_selection"]
            replacement = self.donor.latents[step].to(value)
            result = result.clone()
            if temporal == "current":
                result[:, :, 0:1] = replacement[:, :, 0:1]
            elif temporal == "future":
                result[:, :, 1:] = replacement[:, :, 1:]
            else:
                raise AssertionError(temporal)
            self.events += 1
        if self.capture or self.donor is not None:
            self.injected_latents[step] = result.detach().cpu().clone()
        return result

    def _locus_hash(
        self,
        locus_id: str,
        rows: Mapping[tuple[str, int, int], Mapping[str, torch.Tensor]],
        latents: Mapping[int, torch.Tensor],
    ) -> str:
        meta = selected_loci(self.model_name, (locus_id,))[0]
        values = []
        if meta["family"] == "VIDEO_LATENT":
            temporal = meta["temporal_selection"]
            for step in sorted(latents):
                value = latents[step]
                values.append(value[:, :, 0:1] if temporal == "current" else value[:, :, 1:])
        else:
            for (modality, _occ, layer), row in sorted(rows.items()):
                lr = meta.get("layer_range")
                if lr is not None and not int(lr[0]) <= layer <= int(lr[1]):
                    continue
                if meta["family"] == "VIDEO_WORLD_KV" and modality == "video":
                    tpg = row["k"].shape[1] // 3
                    start, stop = (0, tpg) if meta["temporal_selection"] == "current" else (tpg, 3*tpg)
                    values.extend((row["k"][:, start:stop], row["v"][:, start:stop]))
                elif meta["family"] == "ACTION_QKV" and modality == "action":
                    values.extend((row["q"], row["k"], row["v"]))
                elif meta["family"] == "ACTION_HIDDEN" and modality == "action":
                    values.append(row["hidden"])
        return tensor_collection_hash(values)

    def locus_hash(self, locus_id: str) -> str:
        return self._locus_hash(locus_id, self.rows, self.latents)

    def injected_locus_hash(self, locus_id: str) -> str:
        return self._locus_hash(locus_id, self.injected_rows, self.injected_latents)


class ImagePatchTrace:
    def __init__(self, model: Any, *, capture: bool = False, donor: "ImagePatchTrace | None" = None,
                 locus_ids: Sequence[str] = ()) -> None:
        self.model = model; self.capture = capture; self.donor = donor
        self.loci = selected_loci("imagewam", locus_ids)
        self.original_pre = model.action_expert.pre_dit
        self.original_blocks: list[tuple[Any, Any]] = []
        self.step = -1; self.rows = {}; self.inputs = {}; self.events = 0

    def install(self) -> None:
        def pre_wrapped(*args: Any, **kwargs: Any):
            output = self.original_pre(*args, **kwargs); self.step += 1
            value = output["tokens"]
            if self.capture:
                self.inputs[self.step] = value.detach().cpu().clone()
            if self.donor is not None and any(row["family"] == "ACTION_HIDDEN" for row in self.loci):
                output = dict(output); output["tokens"] = self.donor.inputs[self.step].to(value).clone(); self.events += 1
            return output
        self.model.action_expert.pre_dit = pre_wrapped
        blocks = list(self.model.action_expert.double_blocks) + list(self.model.action_expert.single_blocks)
        for layer, block in enumerate(blocks):
            original = block.prepare_qkv; self.original_blocks.append((block, original))
            def wrapped(*args: Any, __original=original, __layer=layer, **kwargs: Any):
                output = dict(__original(*args, **kwargs)); key = (self.step, __layer)
                if self.capture:
                    self.rows[key] = {name: value.detach().cpu().clone() for name, value in output.items()
                                      if name in {"q", "k", "v", "residual_x"} and torch.is_tensor(value)}
                if self.donor is not None:
                    if any(row["family"] == "ACTION_QKV" for row in self.loci):
                        for component in ("q", "k", "v"):
                            output[component] = self.donor.rows[key][component].to(output[component]).clone(); self.events += 1
                    if any(row["family"] == "ACTION_HIDDEN" for row in self.loci) and "residual_x" in output:
                        output["residual_x"] = self.donor.rows[key]["residual_x"].to(output["residual_x"]).clone(); self.events += 1
                return output
            block.prepare_qkv = wrapped

    def uninstall(self) -> None:
        self.model.action_expert.pre_dit = self.original_pre
        for block, original in self.original_blocks:
            block.prepare_qkv = original

    def locus_hash(self, locus_id: str) -> str:
        meta = selected_loci("imagewam", (locus_id,))[0]
        values = []
        if meta["family"] == "ACTION_QKV":
            for row in self.rows.values(): values.extend(row[x] for x in ("q", "k", "v"))
        elif meta["family"] == "ACTION_HIDDEN":
            values.extend(self.inputs.values())
            values.extend(row["residual_x"] for row in self.rows.values() if "residual_x" in row)
        return tensor_collection_hash(values)


@torch.no_grad()
def joint_infer(policy: Any, video_prepared: Mapping[str, Any], action_prepared: Mapping[str, Any], seed: int,
                trace: BadPatchTrace) -> torch.Tensor:
    model = policy.model; image = video_prepared["image"].to(model.device, model.torch_dtype)
    _, _, height, width = image.shape
    latent_t = (policy.num_video_frames - 1) // model.vae.temporal_downsample_factor + 1
    rand_device = str(policy.cfg.EVALUATION.rand_device)
    vg = torch.Generator(device=rand_device).manual_seed(seed); ag = torch.Generator(device=rand_device).manual_seed(seed)
    video = torch.randn((1, model.vae.model.z_dim, latent_t, height // model.vae.upsampling_factor,
                         width // model.vae.upsampling_factor), generator=vg, device=rand_device,
                        dtype=torch.float32).to(model.device, model.torch_dtype)
    action = torch.randn((1, policy.action_horizon, model.action_expert.action_dim), generator=ag,
                         device=rand_device, dtype=torch.float32).to(model.device, model.torch_dtype)
    first = model._encode_input_image_latents_tensor(input_image=image, tiled=bool(policy.cfg.EVALUATION.tiled))
    video[:, :, 0:1] = first
    vt, vd = model.infer_video_scheduler.build_inference_schedule(policy.num_inference_steps, model.device, video.dtype, shift_override=None)
    at, ad = model.infer_action_scheduler.build_inference_schedule(policy.num_inference_steps, model.device, action.dtype, shift_override=None)
    fuse = bool(getattr(model.video_expert, "fuse_vae_embedding_in_latents", False))
    trace.install()
    try:
        for step, (tv, dv, ta, da) in enumerate(zip(vt, vd, at, ad)):
            video = trace.patch_joint_latent(video, step)
            vp = model.video_expert.pre_dit(x=video, timestep=tv.unsqueeze(0).to(video),
                context=video_prepared["context"], context_mask=video_prepared["context_mask"], action=None,
                fuse_vae_embedding_in_latents=fuse)
            ap = model.action_expert.pre_dit(action_tokens=action, timestep=ta.unsqueeze(0).to(action),
                context=action_prepared["context"], context_mask=action_prepared["context_mask"])
            tpg = int(vp["meta"]["tokens_per_frame"])
            mask = model._build_mot_attention_mask(video_seq_len=vp["tokens"].shape[1], action_seq_len=ap["tokens"].shape[1],
                video_tokens_per_frame=tpg, device=video.device)
            output = model.mot(embeds_all={"video": vp["tokens"], "action": ap["tokens"]}, attention_mask=mask,
                freqs_all={"video": vp["freqs"], "action": ap["freqs"]},
                context_all={"video": {"context": vp["context"], "mask": vp["context_mask"]},
                             "action": {"context": ap["context"], "mask": ap["context_mask"]}},
                t_mod_all={"video": vp["t_mod"], "action": ap["t_mod"]})
            video = model.infer_video_scheduler.step(model.video_expert.post_dit(output["video"], vp), dv, video)
            action = model.infer_action_scheduler.step(model.action_expert.post_dit(output["action"], ap), da, action)
            video[:, :, 0:1] = first
    finally:
        trace.uninstall()
    return action[0].detach().cpu().float()


def patch_static_cache(base: Sequence[Mapping[str, torch.Tensor]], donor: Sequence[Mapping[str, torch.Tensor]],
                       loci: Sequence[dict[str, Any]], tokens_per_group: int | None = None) -> list[dict[str, torch.Tensor]]:
    result = clone_rows(base)
    for meta in loci:
        if meta["family"] not in {"IMAGE_KV", "VIDEO_WORLD_KV"}:
            continue
        start_layer, stop_layer = [int(value) for value in meta["layer_range"]]
        for layer in range(start_layer, stop_layer + 1):
            for component in ("k", "v"):
                source = donor[layer][component].to(result[layer][component])
                temporal = meta.get("temporal_selection")
                if temporal in {"current", "future"}:
                    if tokens_per_group is None:
                        raise AssertionError("tokens_per_group required")
                    start, stop = (0, tokens_per_group) if temporal == "current" else (tokens_per_group, 3*tokens_per_group)
                    result[layer][component][:, start:stop] = source[:, start:stop]
                else:
                    result[layer][component] = source.clone()
    return result


def static_cache_locus_hash(cache: Sequence[Mapping[str, torch.Tensor]], meta: Mapping[str, Any],
                            tokens_per_group: int | None = None) -> str:
    values = []
    start, stop = [int(value) for value in meta["layer_range"]]
    for layer in range(start, stop + 1):
        for component in ("k", "v"):
            value = cache[layer][component]
            temporal = meta.get("temporal_selection")
            if temporal in {"current", "future"}:
                if tokens_per_group is None: raise AssertionError("tokens_per_group required")
                lo, hi = (0, tokens_per_group) if temporal == "current" else (tokens_per_group, 3*tokens_per_group)
                value = value[:, lo:hi]
            values.append(value)
    return tensor_collection_hash(values)


def patch_latent(base: torch.Tensor, donor: torch.Tensor, loci: Sequence[dict[str, Any]]) -> torch.Tensor:
    result = base.clone()
    for meta in loci:
        if meta["family"] != "VIDEO_LATENT": continue
        if meta["temporal_selection"] == "current": result[:, :, 0:1] = donor[:, :, 0:1]
        elif meta["temporal_selection"] == "future": result[:, :, 1:] = donor[:, :, 1:]
        else: raise AssertionError(meta["temporal_selection"])
    return result


@torch.no_grad()
def idm_infer(runner: Any, prepared: Mapping[str, Any], latent: torch.Tensor, noise: torch.Tensor,
              trace: BadPatchTrace, cache_donor: Sequence[Mapping[str, torch.Tensor]] | None = None,
              locus_ids: Sequence[str] = ()) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]], int]:
    model = runner.model; timestep_video = torch.zeros((1,), dtype=latent.dtype, device=model.device)
    fuse = bool(getattr(model.video_expert, "fuse_vae_embedding_in_latents", False))
    pre = model.video_expert.pre_dit(x=latent, timestep=timestep_video, context=prepared["context"],
        context_mask=prepared["context_mask"], action=None, fuse_vae_embedding_in_latents=fuse)
    video_len = int(pre["tokens"].shape[1]); tpg = int(pre["meta"]["tokens_per_frame"])
    mask = model._build_mot_attention_mask(video_seq_len=video_len, action_seq_len=noise.shape[1],
        video_tokens_per_frame=tpg, device=latent.device)
    cache = model.mot.prefill_video_cache(video_tokens=pre["tokens"], video_freqs=pre["freqs"],
        video_t_mod=pre["t_mod"], video_context_payload={"context": pre["context"], "mask": pre["context_mask"]},
        video_attention_mask=mask[:video_len, :video_len])
    loci = selected_loci("idm", locus_ids)
    if cache_donor is not None:
        cache = patch_static_cache(cache, cache_donor, loci, tpg)
    action = noise.clone(); timesteps, deltas = model.infer_action_scheduler.build_inference_schedule(
        runner.policy.num_inference_steps, model.device, action.dtype, shift_override=None)
    trace.install()
    try:
        for timestep, delta in zip(timesteps, deltas):
            prediction = model._predict_action_noise_with_cache(latents_action=action,
                timestep_action=timestep.unsqueeze(0).to(action), context=prepared["context"],
                context_mask=prepared["context_mask"], video_kv_cache=cache, attention_mask=mask, video_seq_len=video_len)
            action = model.infer_action_scheduler.step(prediction, delta, action)
    finally:
        trace.uninstall()
    cache_cpu = [{kind: row[kind].detach().cpu().clone() for kind in ("k", "v")} for row in cache]
    return action[0].detach().cpu().float(), cache_cpu, tpg


def clone_image_cache(cache: Mapping[str, Any]) -> dict[str, Any]:
    return {"double": clone_rows(cache["double"]), "single": clone_rows(cache["single"]),
            "txt_len": int(cache["txt_len"]), "img_len": int(cache["img_len"])}


def image_cache_rows(cache: Mapping[str, Any]) -> list[dict[str, torch.Tensor]]:
    return list(cache["double"]) + list(cache["single"])


def patch_image_cache(base: Mapping[str, Any], donor: Mapping[str, Any], loci: Sequence[dict[str, Any]]) -> dict[str, Any]:
    result = clone_image_cache(base); result_rows = image_cache_rows(result); donor_rows = image_cache_rows(donor)
    prefix = int(result["txt_len"]); image_stop = prefix + int(result["img_len"])
    for meta in loci:
        if meta["family"] == "IMAGE_KV":
            start, stop = [int(value) for value in meta["layer_range"]]
            for layer in range(start, stop + 1):
                for component in ("k", "v"):
                    result_rows[layer][component][:, prefix:image_stop] = donor_rows[layer][component][:, prefix:image_stop].to(result_rows[layer][component])
        elif meta["family"] == "PREFILL_PREFIX_KV":
            for layer in range(len(result_rows)):
                for component in ("k", "v"):
                    result_rows[layer][component][:, :prefix] = donor_rows[layer][component][:, :prefix].to(result_rows[layer][component])
    return result


def image_cache_locus_hash(cache: Mapping[str, Any], meta: Mapping[str, Any]) -> str:
    rows = image_cache_rows(cache); prefix = int(cache["txt_len"]); image_stop = prefix + int(cache["img_len"])
    values = []
    if meta["family"] == "IMAGE_KV":
        start, stop = [int(value) for value in meta["layer_range"]]
        indexes = range(start, stop + 1); bounds = (prefix, image_stop)
    elif meta["family"] == "PREFILL_PREFIX_KV":
        indexes = range(len(rows)); bounds = (0, prefix)
    else:
        return tensor_collection_hash(())
    for layer in indexes:
        values.extend(rows[layer][component][:, bounds[0]:bounds[1]] for component in ("k", "v"))
    return tensor_collection_hash(values)


def closing_drive(runner: Any, action: torch.Tensor) -> tuple[float, np.ndarray]:
    env = np.asarray(runner.normalized_to_env(action), dtype=np.float32)
    return float(np.maximum(-env[:, 6].astype(np.float64), 0.0).sum()), env


def action_comparison(action: torch.Tensor, clean: torch.Tensor, action_env: np.ndarray, clean_env: np.ndarray,
                      target_direction: np.ndarray) -> dict[str, Any]:
    array = action.detach().cpu().float().numpy(); clean_array = clean.detach().cpu().float().numpy()
    delta = array.astype(np.float64) - clean_array.astype(np.float64)
    motion = np.asarray(action_env, dtype=np.float64)[:, :3]
    clean_motion = np.asarray(clean_env, dtype=np.float64)[:, :3]
    def cosine(a, b):
        a=np.asarray(a).reshape(-1); b=np.asarray(b).reshape(-1); den=float(np.linalg.norm(a)*np.linalg.norm(b))
        return float(np.dot(a,b)/den) if den>1e-12 else float("nan")
    tiled = np.repeat(np.asarray(target_direction, dtype=np.float64)[None], len(motion), axis=0)
    return {
        "action_l2_to_clean": float(np.linalg.norm(delta)),
        "action_max_abs_to_clean": float(np.max(np.abs(delta))),
        "elementwise_exact_to_clean": bool(np.array_equal(array, clean_array)),
        "MotionCos_to_clean": cosine(motion, clean_motion),
        "TargetCos": cosine(motion, tiled),
        "action_sha256": tensor_sha256(action),
    }

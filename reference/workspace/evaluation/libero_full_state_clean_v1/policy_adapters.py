from __future__ import annotations
from wam_causal_audit.paths import resolve as _release_path

import math
import os
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


PI05_ROOT = Path(_release_path('@DATA@/openpi_pi05/openpi'))
LINGBOT_ROOT = Path(_release_path('@DATA@/lingbot-va'))
VLA_JEPA_ROOT = Path(_release_path('@DATA@/VLA-JEPA/source'))
VLA_JEPA_CHECKPOINT = Path(_release_path('@DATA@/VLA-JEPA/assets/vla-jepa/LIBERO/checkpoints/VLA-JEPA-LIBERO.pt'))
FASTWAM_ROOT = Path(_release_path('@WORKSPACE@/FastWAM'))
FASTWAM_CHECKPOINT = FASTWAM_ROOT / "checkpoints/fastwam_release/libero_uncond_2cam224.pt"


def disable_localhost_proxy() -> None:
    for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "all_proxy"):
        os.environ.pop(key, None)
    entries = [x.strip() for x in (os.environ.get("NO_PROXY") or "").split(",") if x.strip()]
    for entry in ("127.0.0.1", "localhost", "::1"):
        if entry not in entries:
            entries.append(entry)
    os.environ["NO_PROXY"] = ",".join(entries)
    os.environ["no_proxy"] = os.environ["NO_PROXY"]


def _add_paths(paths: Sequence[Path]) -> None:
    for path in paths:
        if path.exists() and str(path.resolve()) not in sys.path:
            sys.path.insert(0, str(path.resolve()))


def quat_xyzw_to_axis_angle(quat: Sequence[float]) -> np.ndarray:
    value = np.asarray(quat, dtype=np.float64).reshape(-1)[:4].copy()
    value[3] = np.clip(value[3], -1.0, 1.0)
    denominator = math.sqrt(max(0.0, 1.0 - value[3] * value[3]))
    if math.isclose(denominator, 0.0):
        return np.zeros(3, dtype=np.float32)
    return (value[:3] * 2.0 * math.acos(float(value[3])) / denominator).astype(np.float32)


@dataclass
class PolicyOutput:
    action_chunk: np.ndarray
    executable_actions: np.ndarray
    executed_action_indices: list[int]
    model_input_metadata: dict[str, Any]
    latents_video: np.ndarray | None = None
    latent_alignment: dict[str, Any] | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    opaque: Any = None


class PolicyAdapter:
    model_name: str
    render_resolution: int
    num_steps_wait: int
    max_policy_steps: int
    dummy_action: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0)

    def reset(self, task_description: str, trace_meta: Mapping[str, Any]) -> None:
        raise NotImplementedError

    def infer(self, raw_obs: Mapping[str, Any], call_idx: int, env_step: int, first_call: bool) -> PolicyOutput:
        raise NotImplementedError

    def after_chunk(self, output: PolicyOutput, keyframe_observations: Sequence[Mapping[str, np.ndarray]], done: bool) -> None:
        del output, keyframe_observations, done

    def model_metadata(self) -> Mapping[str, Any]:
        raise NotImplementedError

    def policy_protocol(self) -> Mapping[str, Any]:
        raise NotImplementedError


class Pi05Adapter(PolicyAdapter):
    model_name = "pi05"
    render_resolution = 256
    num_steps_wait = 10
    max_policy_steps = 520
    replan_steps = 5

    def __init__(self, host: str, port: int) -> None:
        _add_paths((PI05_ROOT, PI05_ROOT / "packages/openpi-client/src", PI05_ROOT / "third_party/libero"))
        disable_localhost_proxy()
        from openpi_client import image_tools
        from openpi_client import websocket_client_policy

        self.image_tools = image_tools
        self.client = websocket_client_policy.WebsocketClientPolicy(host=host, port=port)
        try:
            self.server_metadata = dict(self.client.get_server_metadata())
        except Exception:
            self.server_metadata = {}

    def infer(self, raw_obs: Mapping[str, Any], call_idx: int, env_step: int, first_call: bool) -> PolicyOutput:
        del call_idx, env_step, first_call
        agent = np.ascontiguousarray(np.asarray(raw_obs["agentview_image"])[::-1, ::-1])
        wrist = np.ascontiguousarray(np.asarray(raw_obs["robot0_eye_in_hand_image"])[::-1, ::-1])
        agent = self.image_tools.convert_to_uint8(self.image_tools.resize_with_pad(agent, 224, 224))
        wrist = self.image_tools.convert_to_uint8(self.image_tools.resize_with_pad(wrist, 224, 224))
        state = np.concatenate([
            np.asarray(raw_obs["robot0_eef_pos"], dtype=np.float32).reshape(3),
            quat_xyzw_to_axis_angle(raw_obs["robot0_eef_quat"]),
            np.asarray(raw_obs["robot0_gripper_qpos"], dtype=np.float32).reshape(-1)[:2],
        ])
        response = dict(self.client.infer({
            "observation/image": agent,
            "observation/wrist_image": wrist,
            "observation/state": state,
            "prompt": self.task_description,
        }))
        actions = np.asarray(response["actions"], dtype=np.float32)
        if actions.ndim == 1:
            actions = actions.reshape(1, -1)
        if actions.shape[0] < self.replan_steps or actions.shape[1] < 7:
            raise RuntimeError(f"pi0.5 invalid action chunk shape: {actions.shape}")
        return PolicyOutput(
            action_chunk=actions.copy(),
            executable_actions=actions[: self.replan_steps, :7].copy(),
            executed_action_indices=list(range(self.replan_steps)),
            model_input_metadata={
                "camera_inputs": ["agentview_image", "robot0_eye_in_hand_image"],
                "raw_preprocess": "rotate_180_then_resize_with_pad_224_uint8",
                "state": "eef_pos[3]+eef_quat_axis_angle[3]+gripper_qpos[2]",
            },
            extra={"server_response_keys": sorted(response.keys())},
        )

    def reset(self, task_description: str, trace_meta: Mapping[str, Any]) -> None:
        self.task_description = str(task_description)
        self.policy_rng_seed = int(trace_meta["seed"])
        self.client.infer({"__temporal_control__": {"operation": "reset", "seed": self.policy_rng_seed}})

    def model_metadata(self) -> Mapping[str, Any]:
        return {
            "checkpoint": _release_path('@DATA@/openpi_pi05/cache/openpi-assets/checkpoints/pi05_libero'),
            "server_metadata": self.server_metadata,
            "runtime_policy_rng_seed": getattr(self, "policy_rng_seed", None),
        }

    def policy_protocol(self) -> Mapping[str, Any]:
        return {
            "official_entrypoint": str(PI05_ROOT / "examples/libero/main.py"),
            "full_chunk_returned": True,
            "executed_prefix_steps": self.replan_steps,
            "server_rng_reset_each_episode": "trace_meta.seed",
        }


class VLAJEPAAdapter(PolicyAdapter):
    model_name = "vla_jepa"
    render_resolution = 256
    num_steps_wait = 10
    max_policy_steps = 520

    def __init__(self, host: str, port: int) -> None:
        _add_paths((VLA_JEPA_ROOT,))
        disable_localhost_proxy()
        from examples.LIBERO.model2libero_interface import M1Inference

        self.policy = M1Inference(
            policy_ckpt_path=VLA_JEPA_CHECKPOINT,
            host=host,
            port=port,
            image_size=[224, 224],
        )
        self.chunk_size = int(self.policy.action_chunk_size)

    def reset(self, task_description: str, trace_meta: Mapping[str, Any]) -> None:
        self.task_description = str(task_description)
        self.policy_rng_seed = int(trace_meta["seed"])
        self.policy.reset(task_description=self.task_description)
        # M1Inference.reset() clears the local ensemble/history but hard-codes
        # server seed 0.  Re-issue only the server reset with the paired clean
        # seed after the local state is clear.
        self.policy.client.reset(self.task_description, seed=self.policy_rng_seed)

    def infer(self, raw_obs: Mapping[str, Any], call_idx: int, env_step: int, first_call: bool) -> PolicyOutput:
        del env_step, first_call
        agent = np.ascontiguousarray(np.asarray(raw_obs["agentview_image"])[::-1, ::-1])
        wrist = np.ascontiguousarray(np.asarray(raw_obs["robot0_eye_in_hand_image"])[::-1, ::-1])
        state = np.concatenate([
            np.asarray(raw_obs["robot0_eef_pos"], dtype=np.float32).reshape(3),
            quat_xyzw_to_axis_angle(raw_obs["robot0_eef_quat"]),
            np.asarray(raw_obs["robot0_gripper_qpos"], dtype=np.float32).reshape(-1)[:2],
        ])[None]
        response = self.policy.step(
            images=[agent, wrist],
            task_description=self.task_description,
            state=state,
            step=int(call_idx) * self.chunk_size,
        )
        if not response["new_policy_call"]:
            raise RuntimeError("VLA-JEPA adapter expected a fresh policy call")
        raw_chunk = np.asarray(response["raw_action_chunk"], dtype=np.float32)
        executable = raw_chunk[:, :7].copy()
        executable[:, 6] = 1.0 - 2.0 * (executable[:, 6] > 0.5)
        return PolicyOutput(
            action_chunk=raw_chunk,
            executable_actions=executable,
            executed_action_indices=list(range(raw_chunk.shape[0])),
            model_input_metadata={
                "camera_inputs": ["agentview_image", "robot0_eye_in_hand_image"],
                "raw_preprocess": "rotate_180_then_cv2_resize_224",
                "state": "eef_pos[3]+eef_quat_axis_angle[3]+gripper_qpos[2]",
                "policy_step_for_call": int(call_idx) * self.chunk_size,
            },
            extra={
                "server_rng_before": response.get("server_rng_before"),
                "server_rng_after": response.get("server_rng_after"),
                "gripper_conversion": "open_probability > 0.5 maps to LIBERO -1; otherwise +1",
            },
        )

    def model_metadata(self) -> Mapping[str, Any]:
        return {
            "checkpoint": str(VLA_JEPA_CHECKPOINT),
            "runtime_policy_rng_seed": getattr(self, "policy_rng_seed", None),
        }

    def policy_protocol(self) -> Mapping[str, Any]:
        return {
            "official_entrypoint": str(VLA_JEPA_ROOT / "examples/LIBERO/eval_libero.py"),
            "chunk_size": self.chunk_size,
            "executed_steps": self.chunk_size,
            "server_rng_reset_each_episode": "trace_meta.seed",
        }


def _lingbot_obs(raw_obs: Mapping[str, Any]) -> dict[str, np.ndarray]:
    return {
        "observation.images.agentview_rgb": np.ascontiguousarray(np.asarray(raw_obs["agentview_image"])[::-1]),
        "observation.images.eye_in_hand_rgb": np.ascontiguousarray(np.asarray(raw_obs["robot0_eye_in_hand_image"])[::-1]),
    }


class LingBotAdapter(PolicyAdapter):
    model_name = "lingbot_va"
    render_resolution = 128
    num_steps_wait = 5
    max_policy_steps = 795
    dummy_action = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    def __init__(self, host: str, port: int) -> None:
        client_root = LINGBOT_ROOT / "wan_va/utils/Simple_Remote_Infer"
        _add_paths((LINGBOT_ROOT, client_root))
        disable_localhost_proxy()
        from deploy.websocket_client_policy import WebsocketClientPolicy

        self._client_class = WebsocketClientPolicy
        self._host = str(host)
        self._port = int(port)
        self.client = self._client_class(host=self._host, port=self._port)

    def _reconnect(self) -> None:
        self.client = self._client_class(host=self._host, port=self._port)

    def reset(self, task_description: str, trace_meta: Mapping[str, Any]) -> None:
        self.task_description = str(task_description)
        self.policy_rng_seed = int(trace_meta["seed"])
        payload = dict(reset=True, prompt=self.task_description, trace_meta=dict(trace_meta))
        def reset_and_seed() -> None:
            self.client.infer(payload)
            self.client.infer({
                "temporal_control": {
                    "operation": "rng_reset",
                    "seed": self.policy_rng_seed,
                }
            })

        try:
            reset_and_seed()
        except Exception:
            # A websocket handler deliberately closes the connection after an
            # internal model error.  A new episode must not inherit that dead
            # client and turn every subsequent branch into a placeholder.
            self._reconnect()
            reset_and_seed()

    def temporal_snapshot(self, label: str) -> None:
        """Save the server-side KV and streaming-VAE state for a probe."""
        self.client.infer({"temporal_control": {"operation": "snapshot", "label": str(label)}})

    def temporal_restore(self, label: str) -> None:
        """Restore the server-side KV/VAE state saved by ``temporal_snapshot``."""
        self.client.infer({"temporal_control": {"operation": "restore", "label": str(label)}})

    def temporal_drop_snapshot(self, label: str) -> None:
        self.client.infer({"temporal_control": {"operation": "drop_snapshot", "label": str(label)}})

    def rng_snapshot(self) -> Mapping[str, Any]:
        response = self.client.infer({"temporal_control": {"operation": "rng_snapshot"}})
        return dict(response["temporal"])

    def rng_restore(self, state: Mapping[str, Any]) -> None:
        self.client.infer({"temporal_control": {"operation": "rng_restore", "state": dict(state)}})

    def infer(self, raw_obs: Mapping[str, Any], call_idx: int, env_step: int, first_call: bool) -> PolicyOutput:
        current_obs = _lingbot_obs(raw_obs)
        response = dict(self.client.infer(dict(obs=current_obs, prompt=self.task_description, return_latent=True)))
        action = np.asarray(response["action"], dtype=np.float32)
        if action.ndim != 3 or action.shape[0] < 7 or action.shape[2] % 4 != 0:
            raise RuntimeError(f"LingBot invalid action shape: {action.shape}")
        action_per_frame = int(action.shape[2] // 4)
        start_frame = 1 if first_call else 0
        flat_actions = []
        flat_indices = []
        for frame_idx in range(start_frame, action.shape[1]):
            for token_idx in range(action.shape[2]):
                flat_actions.append(np.asarray(action[:, frame_idx, token_idx], dtype=np.float32).reshape(-1)[:7])
                flat_indices.append(frame_idx * action.shape[2] + token_idx)
        executable = np.asarray(flat_actions, dtype=np.float32)
        predicted_frame_env_steps = [int(env_step)] if first_call else []
        for frame_idx in range(start_frame, action.shape[1]):
            predicted_frame_env_steps.append(int(env_step + (frame_idx - start_frame + 1) * action.shape[2]))
        return PolicyOutput(
            action_chunk=action,
            executable_actions=executable,
            executed_action_indices=flat_indices,
            model_input_metadata={
                "camera_inputs": ["agentview_image", "robot0_eye_in_hand_image"],
                "raw_preprocess": "vertical_flip_only_128",
                "proprioception": "none",
                "persistent_server_cache": True,
            },
            latents_video=np.asarray(response["video_latent"]) if "video_latent" in response else None,
            latent_alignment={
                "representation": "LingBot returned video_latent",
                "frame_to_planned_env_step": predicted_frame_env_steps,
                "action_per_predicted_frame": action_per_frame,
                "first_call_skips_frame_zero_actions": bool(first_call),
            },
            extra={
                "raw_action_shape": list(action.shape),
                "action_per_frame": action_per_frame,
                "first_call_skipped_frame_zero": bool(first_call),
                "response_keys": sorted(response.keys()),
                "imagination_mask_audit": response.get("imagination_mask_audit"),
                "imagined_kv_audit": response.get("imagined_kv_audit"),
            },
            opaque=action,
        )

    def after_chunk(self, output: PolicyOutput, keyframe_observations: Sequence[Mapping[str, np.ndarray]], done: bool) -> None:
        if done or not keyframe_observations:
            return
        self.client.infer(dict(
            obs=list(keyframe_observations),
            compute_kv_cache=True,
            imagine=False,
            state=output.opaque,
        ))

    def model_metadata(self) -> Mapping[str, Any]:
        return {
            "checkpoint": _release_path('@DATA@/checkpoints/lingbot-va-posttrain-libero-long'),
            "runtime_policy_rng_seed": getattr(self, "policy_rng_seed", None),
        }

    def policy_protocol(self) -> Mapping[str, Any]:
        return {
            "official_entrypoint": str(LINGBOT_ROOT / "evaluation/libero/client.py"),
            "persistent_kv_cache": True,
            "first_call_skips_frame_zero_actions": True,
            "cache_update_after_each_chunk": True,
            "server_rng_reset_each_episode": "trace_meta.seed",
        }


class FastWAMAdapter(PolicyAdapter):
    model_name = "fastwam"
    render_resolution = 256
    num_steps_wait = 30
    max_policy_steps = 700

    def __init__(self) -> None:
        _add_paths((FASTWAM_ROOT, FASTWAM_ROOT / "experiments/libero"))
        import torch
        from hydra import compose, initialize_config_dir
        from hydra.utils import instantiate
        from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor
        from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
        from experiments.libero.eval_libero_single import (
            _load_model_checkpoint,
            _mixed_precision_to_model_dtype,
            _resolve_dataset_stats_path,
            _resolve_eval_device,
        )

        self.torch = torch
        with initialize_config_dir(config_dir=str(FASTWAM_ROOT / "configs"), version_base="1.3"):
            self.cfg = compose(
                config_name="sim_libero.yaml",
                overrides=[
                    f"ckpt={FASTWAM_CHECKPOINT}",
                    "EVALUATION.task_suite_name=libero_10",
                    "gpu_id=0",
                ],
            )
        device = _resolve_eval_device(self.cfg)
        dtype = _mixed_precision_to_model_dtype(self.cfg.get("mixed_precision", "bf16"))
        self.model = instantiate(self.cfg.model, model_dtype=dtype, device=device)
        _load_model_checkpoint(self.model, str(FASTWAM_CHECKPOINT))
        self.model = self.model.to(device).eval()
        stats = load_dataset_stats_from_json(str(_resolve_dataset_stats_path(self.cfg)))
        self.processor: FastWAMProcessor = instantiate(self.cfg.data.train.processor).eval()
        self.processor.set_normalizer_from_stats(stats)
        video_size = self.cfg.data.train.video_size
        self.input_h, self.input_w = int(video_size[0]), int(video_size[1])
        configured_horizon = self.cfg.EVALUATION.get("action_horizon")
        self.action_horizon = int(configured_horizon) if configured_horizon is not None else int(self.cfg.data.train.num_frames) - 1
        self.replan_steps = int(self.cfg.EVALUATION.replan_steps)
        self.video_frame_count = (int(self.cfg.data.train.num_frames) - 1) // int(self.cfg.data.train.action_video_freq_ratio) + 1
        self.action_video_freq_ratio = int(self.cfg.data.train.action_video_freq_ratio)

    def reset(self, task_description: str, trace_meta: Mapping[str, Any]) -> None:
        self.task_description = str(task_description)
        self.policy_rng_seed = int(trace_meta["seed"])
        random.seed(self.policy_rng_seed)
        np.random.seed(self.policy_rng_seed)
        self.torch.manual_seed(self.policy_rng_seed)
        if self.torch.cuda.is_available():
            self.torch.cuda.manual_seed_all(self.policy_rng_seed)

    def infer(self, raw_obs: Mapping[str, Any], call_idx: int, env_step: int, first_call: bool) -> PolicyOutput:
        del call_idx, first_call
        from experiments.libero.eval_libero_single import _predict_action_chunk

        action, _imgs, _predicted_frames, raw_prediction = _predict_action_chunk(
            obs=dict(raw_obs),
            task_description=self.task_description,
            model=self.model,
            processor=self.processor,
            cfg=self.cfg,
            action_horizon=self.action_horizon,
            input_w=self.input_w,
            input_h=self.input_h,
            model_device=str(self.model.device),
            force_joint_latents=True,
            return_raw_prediction=True,
        )
        action = np.asarray(action, dtype=np.float32)
        latents = raw_prediction.get("video_latents")
        if hasattr(latents, "detach"):
            latents = latents.detach().cpu().numpy()
        video_frame_steps = [int(env_step + index * self.action_video_freq_ratio) for index in range(self.video_frame_count)]
        return PolicyOutput(
            action_chunk=action,
            executable_actions=action[: self.replan_steps, :7].copy(),
            executed_action_indices=list(range(self.replan_steps)),
            model_input_metadata={
                "camera_inputs": ["agentview_image", "robot0_eye_in_hand_image"],
                "raw_preprocess": "FastWAMProcessor two-camera center-crop/resize and stitched 224x448 input",
                "proprioception": "checkpoint FastWAM LIBERO state processor",
                "action_only_rollout_contract": True,
                "joint_video_sampling_is_observational_sidecar": True,
            },
            latents_video=None if latents is None else np.asarray(latents),
            latent_alignment={
                "representation": "FastWAM final denoised VAE video_latents",
                "planned_decoded_video_frame_to_env_step": video_frame_steps,
                "action_video_frequency_ratio": self.action_video_freq_ratio,
                "executed_prefix_env_step_limit": int(env_step + self.replan_steps),
                "latent_temporal_compression_factor": int(self.model.vae.temporal_downsample_factor),
                "note": "future frames beyond executed prefix are model plans and are not reached before replanning",
            },
            extra={"raw_prediction_keys": sorted(raw_prediction.keys()), "latent_sidecar_included_in_latency": True},
        )

    def model_metadata(self) -> Mapping[str, Any]:
        return {
            "checkpoint": str(FASTWAM_CHECKPOINT),
            "dtype": str(self.model.torch_dtype),
            "config_seed": int(self.cfg.seed),
            "runtime_policy_rng_seed": getattr(self, "policy_rng_seed", None),
        }

    def policy_protocol(self) -> Mapping[str, Any]:
        return {
            "official_entrypoint": str(FASTWAM_ROOT / "experiments/libero/eval_libero_single.py"),
            "full_action_horizon": self.action_horizon,
            "executed_prefix_steps": self.replan_steps,
            "num_steps_wait": self.num_steps_wait,
            "runtime_rng_reset_each_episode": "trace_meta.seed",
            "latent_export": "infer_joint(decode_video=False) returns action-only contract plus video_latents sidecar",
        }


def make_adapter(model: str, host: str, port: int) -> PolicyAdapter:
    key = str(model).lower().replace("-", "_").replace(".", "")
    if key in {"pi05", "pi0_5"}:
        return Pi05Adapter(host, port)
    if key in {"lingbot", "lingbot_va"}:
        return LingBotAdapter(host, port)
    if key in {"vla_jepa", "vlajepa"}:
        return VLAJEPAAdapter(host, port)
    if key == "fastwam":
        return FastWAMAdapter()
    raise ValueError(f"unknown model: {model}")

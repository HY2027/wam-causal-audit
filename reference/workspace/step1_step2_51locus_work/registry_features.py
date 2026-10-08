from __future__ import annotations
from wam_causal_audit.paths import resolve as _release_path

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


WEEK1 = Path(_release_path('@WORKSPACE@/week1_audit_work'))
if str(WEEK1) not in sys.path:
    sys.path.insert(0, str(WEEK1))

from protocol import mean_channel, pool_hidden, pool_latent, pool_qkv_rows, tensor_sha256  # noqa: E402


REGISTRY_PATH = Path(_release_path('@WORKSPACE@/step0_tensor_inventory_work/step0_1_v2/locus_registry_v2.json'))


def registry() -> dict[str, Any]:
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def model_loci(model: str) -> list[dict[str, Any]]:
    return [row for row in registry()["loci"] if row["model"] == model]


def context_feature(context: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
    continuous = mean_channel(context, channel_axis=-1).numpy().astype(np.float32, copy=False).reshape(-1)
    mask_stats = np.asarray([
        float(mask.detach().float().mean()),
        float(mask.detach().float().std(unbiased=False)),
    ], dtype=np.float32)
    return np.concatenate((continuous, mask_stats))


def _layer_rows(rows: Mapping[tuple[str, int, int], dict[str, torch.Tensor]], modality: str, start: int, stop: int):
    return [row for (kind, _occurrence, layer), row in sorted(rows.items(), key=lambda item: (item[0][1], item[0][2]))
            if kind == modality and start <= layer <= stop]


def _temporal_rows(trace: Any, start: int, stop: int, temporal: str):
    output = []
    for (kind, _occurrence, layer), row in sorted(trace.temporal_rows.items(), key=lambda item: (item[0][1], item[0][2])):
        if kind != "video" or not start <= layer <= stop:
            continue
        if not all(component in row and temporal in row[component] for component in ("k", "v")):
            raise AssertionError(f"Missing temporal pool at layer {layer}/{temporal}")
        output.append({component: row[component][temporal] for component in ("k", "v")})
    return output


def features_from_capture(model: str, capture: Mapping[str, Any], num_heads: int) -> dict[str, np.ndarray]:
    features: dict[str, np.ndarray] = {}
    if model == "direct":
        cache = capture["current"]["cache"]
        for index, (start, stop) in enumerate(((0, 4), (5, 9), (10, 14), (15, 19), (20, 24), (25, 29)), 1):
            features[f"DIRECT_L{index:02d}"] = pool_qkv_rows(cache[start:stop + 1], num_heads=num_heads)
        action_rows = _layer_rows(capture["trace"].rows, "action", 0, 29)
        features["DIRECT_L07"] = pool_qkv_rows(action_rows, components=("q", "k", "v"), num_heads=num_heads)
        features["DIRECT_L08"] = pool_hidden([row["hidden"] for row in action_rows])
        features["DIRECT_L09"] = context_feature(capture["context"], capture["context_mask"])

    elif model == "joint":
        trace = capture["trace"]
        locus_index = 1
        for start, stop in ((0, 4), (5, 9), (10, 14), (15, 19), (20, 24), (25, 29)):
            for temporal in ("current", "future"):
                features[f"JOINT_L{locus_index:02d}"] = pool_qkv_rows(
                    _temporal_rows(trace, start, stop, temporal), num_heads=num_heads
                )
                locus_index += 1
        latent = capture["joint_result"]["video_latent"].detach().cpu()
        features["JOINT_L13"] = pool_latent(latent, (0,))
        features["JOINT_L14"] = pool_latent(latent, (1, 2))
        action_rows = _layer_rows(trace.rows, "action", 0, 29)
        features["JOINT_L15"] = pool_qkv_rows(action_rows, components=("q", "k", "v"), num_heads=num_heads)
        features["JOINT_L16"] = pool_hidden([row["hidden"] for row in action_rows])
        features["JOINT_L17"] = context_feature(capture["context"], capture["context_mask"])

    elif model == "idm":
        cache = capture["video_cache"]
        tokens_per_group = int(cache[0]["k"].shape[1] // 3)
        locus_index = 1
        for start, stop in ((0, 4), (5, 9), (10, 14), (15, 19), (20, 24), (25, 29)):
            for temporal in ("current", "future"):
                lo, hi = (0, tokens_per_group) if temporal == "current" else (tokens_per_group, 3 * tokens_per_group)
                rows = [{kind: cache[layer][kind][:, lo:hi] for kind in ("k", "v")} for layer in range(start, stop + 1)]
                features[f"IDM_L{locus_index:02d}"] = pool_qkv_rows(rows, num_heads=num_heads)
                locus_index += 1
        latent = capture["generated_latent"]
        features["IDM_L13"] = pool_latent(latent, (0,))
        features["IDM_L14"] = pool_latent(latent, (1, 2))
        action_rows = _layer_rows(capture["trace"].rows, "action", 0, 29)
        features["IDM_L15"] = pool_qkv_rows(action_rows, components=("q", "k", "v"), num_heads=num_heads)
        features["IDM_L16"] = pool_hidden([row["hidden"] for row in action_rows])
        features["IDM_L17"] = context_feature(capture["context"], capture["context_mask"])

    elif model == "imagewam":
        cache = capture["cache"]
        rows = list(cache["double"]) + list(cache["single"])
        prefix_stop = int(cache["txt_len"])
        image_stop = prefix_stop + int(cache["img_len"])
        for index, (start, stop) in enumerate(((0, 4), (5, 9), (10, 14), (15, 19), (20, 24)), 1):
            selected = [{kind: rows[layer][kind][:, prefix_stop:image_stop] for kind in ("k", "v")}
                        for layer in range(start, stop + 1)]
            features[f"IMAGEWAM_L{index:02d}"] = pool_qkv_rows(selected, num_heads=num_heads)
        prefix = [{kind: row[kind][:, :prefix_stop] for kind in ("k", "v")} for row in rows]
        features["IMAGEWAM_L06"] = pool_qkv_rows(prefix, num_heads=num_heads)
        action_rows = [row for _key, row in sorted(capture["action_trace"].rows.items())]
        features["IMAGEWAM_L07"] = pool_qkv_rows(action_rows, components=("q", "k", "v"), num_heads=num_heads)
        residuals = list(capture["action_trace"].inputs) + [row["residual_x"] for row in action_rows if "residual_x" in row]
        features["IMAGEWAM_L08"] = pool_hidden(residuals)
    else:
        raise ValueError(model)

    expected = [row["locus_id"] for row in model_loci(model)]
    if sorted(features) != sorted(expected):
        raise AssertionError(f"51-locus feature mismatch for {model}: got={sorted(features)} expected={sorted(expected)}")
    if not all(np.all(np.isfinite(value)) for value in features.values()):
        raise AssertionError(f"Nonfinite registry feature for {model}")
    return features


def feature_bundle_sha256(features: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for locus in sorted(features):
        digest.update(locus.encode())
        digest.update(tensor_sha256(features[locus]).encode())
    return digest.hexdigest()

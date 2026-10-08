from __future__ import annotations
from wam_causal_audit.paths import resolve as _release_path

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch


WORK = Path(__file__).resolve().parent
ROOT = Path(os.environ.get(
    "WEEK1_AUDIT_ROOT",
    _release_path('@WORKSPACE@/runs/provenance_encoding_authority_audit_week1'),
))
RADIAL_ROOT = Path(_release_path('@WORKSPACE@/runs/radial_phase_validity_dose_sweep'))
PHASE_ROOT = Path(_release_path('@WORKSPACE@/runs/embodiment_phase_gating_decomposition'))
STATE_ROOT = Path(_release_path('@WORKSPACE@/runs/state_conditioned_rep_replacement'))

MODELS = ("direct", "joint", "idm", "imagewam")
TASKS = (0, 1, 2, 3)
DOSES_CM = (0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0)
NEGATIVE_DOSES_CM = (-0.5, -1.0, -2.0)
STATE_PAIR_TASKS = (0, 1, 2, 3, 4)
STATE_PAIR_IDS = tuple(range(10))
SEED_RANDOM_LOCUS = 20260823
BOOTSTRAP_SEED = 20260823
BOOTSTRAP_REPLICATES = 10_000
STATE_VARYING_THRESHOLD = 1e-4

LOCUS_SPECS: dict[str, tuple[tuple[str, str], ...]] = {
    "direct": (
        ("L1", "image_kv_early_layers_1_10"),
        ("L2", "image_kv_middle_layers_11_20"),
        ("L3", "image_kv_late_layers_21_30"),
        ("L4", "image_kv_gated_global_layers_12_15"),
        ("L5", "action_stream_hidden_states"),
        ("L6", "non_image_prefix_kv"),
    ),
    "joint": (
        ("L1", "video_world_kv_early_layers_1_10"),
        ("L2", "video_world_kv_middle_layers_11_20"),
        ("L3", "video_world_kv_late_layers_21_30"),
        ("L4", "action_qkv"),
        ("L5", "action_stream_hidden_states"),
        ("L6", "non_video_prefix_kv"),
    ),
    "idm": (
        ("L1", "current_latent_group"),
        ("L2", "future_temporal_group_1"),
        ("L3", "future_temporal_group_2"),
        ("L4", "future_groups_pooled"),
        ("L5", "kv_video_read_by_action_decoder"),
        ("L6", "action_side_context"),
        ("L7", "decoded_future_pixels"),
    ),
    "imagewam": (
        ("L1", "image_token_kv_early_group"),
        ("L2", "image_token_kv_middle_group"),
        ("L3", "image_token_kv_late_group"),
        ("L4", "non_image_prefix_kv"),
        ("L5", "action_expert_input"),
        ("L6", "full_state_varying_cache_concatenated"),
    ),
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
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(jsonable(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows and fieldnames is None:
        raise ValueError(f"No rows and no schema for {path}")
    names = list(fieldnames or rows[0].keys())
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def tensor_sha256(value: torch.Tensor | np.ndarray) -> str:
    tensor = torch.as_tensor(value).detach().to("cpu").contiguous()
    return hashlib.sha256(memoryview(tensor.reshape(-1).view(torch.uint8).numpy())).hexdigest()


def observation_sha256(observation: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in sorted(observation):
        array = np.ascontiguousarray(observation[key])
        digest.update(key.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(memoryview(array.view(np.uint8)))
    return digest.hexdigest()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as bundle:
        return {key: bundle[key].copy() for key in bundle.files}


def selected_clean_call(model: str, task: int, state: int) -> dict[str, Any]:
    catalog = json.loads((RADIAL_ROOT / "source_state_catalog.json").read_text(encoding="utf-8"))
    matches = [row for row in catalog["sources"] if row["model"] == model and int(row["task_id"]) == task and int(row["source_state_id"]) == state]
    if len(matches) != 1: raise AssertionError(f"Source catalog mismatch: {model}/task{task}/state{state}")
    clean = Path(matches[0]["clean_dir"])
    result = json.loads((clean / "result.json").read_text(encoding="utf-8"))
    with (clean / "actions.csv").open(newline="", encoding="utf-8") as stream:
        actions = list(csv.DictReader(stream))
    calls = json.loads((clean / "policy_calls.json").read_text(encoding="utf-8"))
    attempt = int(result["first_grasp_attempt_step"])
    attempt_rows = [row for row in actions if int(row["step"]) == attempt]
    if len(attempt_rows) != 1 or attempt_rows[0]["closure_attempt"] != "True":
        raise AssertionError("First-grasp lineage mismatch")
    call_id = int(attempt_rows[0]["policy_call"])
    call = calls[call_id]
    return {
        "instruction": result["instruction"], "target_object": result["target_object"],
        "seed": int(call["seed"]), "selected_policy_call": call_id,
        "old_clean_action_sha256": call["action_sha256"], "old_source_rep_sha256": call["source_rep_sha256"],
        "clean_dir": str(clean),
    }


def radial_source_ids(model: str, task: int, smoke: bool = False) -> list[int]:
    paths = sorted((RADIAL_ROOT / "geometry" / model / f"task_{task}").glob("state_*/source_observation.npz"))
    ids = [int(path.parent.name.split("_")[-1]) for path in paths]
    return ids[:2] if smoke else ids


def dose_tag(dose_cm: float) -> str:
    return str(float(dose_cm)).replace("-", "neg_").replace(".", "p")


def radial_observation_path(model: str, task: int, state: int, dose_cm: float) -> Path:
    base = RADIAL_ROOT / "geometry" / model / f"task_{task}" / f"state_{state:02d}"
    if float(dose_cm) == 0.0:
        return base / "source_observation.npz"
    return base / f"donor_observation__dose_{dose_tag(dose_cm)}cm.npz"


def radial_geometry_path(model: str, task: int, state: int, dose_cm: float) -> Path:
    return RADIAL_ROOT / "geometry" / model / f"task_{task}" / f"state_{state:02d}" / f"geometry__dose_{dose_tag(dose_cm)}cm.json"


def radial_result_path(model: str, task: int, state: int, dose_cm: float) -> Path:
    return RADIAL_ROOT / "cases" / model / f"task_{task}" / f"state_{state:02d}" / f"dose_{dose_tag(dose_cm)}cm" / "result.json"


def mean_channel(value: torch.Tensor, *, channel_axis: int = -1) -> torch.Tensor:
    """Mean every axis except the declared channel axis and return float32 CPU."""
    x = value.detach().float()
    axis = channel_axis if channel_axis >= 0 else x.ndim + channel_axis
    if axis < 0 or axis >= x.ndim:
        raise ValueError(f"Invalid channel axis {channel_axis} for {tuple(x.shape)}")
    reduce = tuple(index for index in range(x.ndim) if index != axis)
    return x.mean(dim=reduce).cpu() if reduce else x.cpu()


def pool_qkv_rows(
    rows: Iterable[Mapping[str, torch.Tensor]],
    *,
    components: Sequence[str] = ("k", "v"),
    num_heads: int | None = None,
) -> np.ndarray:
    """Pool tokens, heads, layers and denoise steps; concatenate Q/K/V channels.

    Q/K/V are stored as [B,S,H*Dh].  We restore heads when num_heads is
    known, mean over B/S/H and preserve Dh.  If heads are unavailable, the
    flattened H*Dh dimension is treated as the channel dimension.
    """
    materialized = list(rows)
    if not materialized:
        raise ValueError("No tensors to pool")
    pooled_components: list[torch.Tensor] = []
    for component in components:
        vectors: list[torch.Tensor] = []
        for row in materialized:
            x = row[component].detach().float()
            if num_heads is not None and x.ndim == 3 and x.shape[-1] % num_heads == 0:
                x = x.reshape(x.shape[0], x.shape[1], num_heads, x.shape[-1] // num_heads)
                vectors.append(x.mean(dim=(0, 1, 2)).cpu())
            else:
                vectors.append(mean_channel(x, channel_axis=-1))
        pooled_components.append(torch.stack(vectors).mean(dim=0))
    return torch.cat(pooled_components).numpy().astype(np.float32, copy=False)


def pool_hidden(values: Iterable[torch.Tensor]) -> np.ndarray:
    vectors = [mean_channel(value, channel_axis=-1) for value in values]
    if not vectors:
        raise ValueError("No hidden tensors to pool")
    return torch.stack(vectors).mean(dim=0).numpy().astype(np.float32, copy=False)


def pool_latent(value: torch.Tensor, temporal_groups: Sequence[int] | None = None) -> np.ndarray:
    """Pool [B,C,T,H,W] while preserving latent channel C."""
    x = value.detach().float()
    if x.ndim != 5:
        raise ValueError(f"Expected [B,C,T,H,W], got {tuple(x.shape)}")
    if temporal_groups is not None:
        x = x[:, :, list(temporal_groups)]
    return x.mean(dim=(0, 2, 3, 4)).cpu().numpy().astype(np.float32, copy=False)


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator > 1e-12 else float("nan")


def paired_bootstrap(values: Sequence[float], *, seed: int = BOOTSTRAP_SEED, n_boot: int = BOOTSTRAP_REPLICATES) -> tuple[float, float, float]:
    data = np.asarray(values, dtype=np.float64)
    data = data[np.isfinite(data)]
    if not len(data):
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=np.float64)
    for index in range(n_boot):
        means[index] = np.mean(data[rng.integers(0, len(data), size=len(data))])
    return float(np.mean(data)), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))

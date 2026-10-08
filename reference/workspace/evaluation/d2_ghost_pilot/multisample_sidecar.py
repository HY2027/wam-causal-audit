from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class MultiSampleDecision:
    selected: bool
    reason: str


class MultiSampleCallScheduler:
    """Select occluded, post-occlusion, and clean-baseline policy calls."""

    def __init__(self, *, clean_calls: int = 3, post_calls: int = 5) -> None:
        self.clean_calls = int(clean_calls)
        self.post_calls = int(post_calls)
        self._clean_selected = 0
        self._saw_occlusion = False
        self._post_selected = 0

    def decide(self, *, occlusion_active: bool, occlusion_has_triggered: bool) -> MultiSampleDecision:
        if occlusion_active:
            self._saw_occlusion = True
            return MultiSampleDecision(True, "occlusion_window")
        if self._saw_occlusion and self._post_selected < self.post_calls:
            self._post_selected += 1
            return MultiSampleDecision(True, "post_occlusion")
        if not occlusion_has_triggered and self._clean_selected < self.clean_calls:
            self._clean_selected += 1
            return MultiSampleDecision(True, "clean_baseline")
        return MultiSampleDecision(False, "single_sample")


def action_variance(action_chunks: Sequence[np.ndarray]) -> dict[str, float]:
    stack = np.stack([np.asarray(item, dtype=np.float64) for item in action_chunks], axis=0)
    per_element = np.var(stack, axis=0, ddof=0)
    return {
        "mean_element_variance": float(np.mean(per_element)),
        "max_element_variance": float(np.max(per_element)),
        "mean_pairwise_l2": float(
            np.mean(
                [np.linalg.norm(stack[i] - stack[j]) for i in range(len(stack)) for j in range(i + 1, len(stack))]
            )
        ),
    }


def save_multisample_call(
    root: str | Path,
    *,
    call_idx: int,
    env_step: int,
    reason: str,
    noise_seeds: Sequence[int],
    action_chunks: Sequence[np.ndarray],
    latent_paths: Sequence[str | Path | None] | None = None,
    primary_reference: np.ndarray | None = None,
) -> dict[str, Any]:
    root_path = Path(root)
    call_dir = root_path / f"call_{int(call_idx):04d}"
    call_dir.mkdir(parents=True, exist_ok=True)
    actions = np.stack([np.asarray(item, dtype=np.float32) for item in action_chunks], axis=0)
    np.savez_compressed(call_dir / "action_chunks.npz", action_chunks=actions, noise_seeds=np.asarray(noise_seeds))
    exact = None
    max_abs = None
    if primary_reference is not None:
        reference = np.asarray(primary_reference, dtype=np.float32)
        exact = bool(np.array_equal(actions[0], reference))
        max_abs = float(np.max(np.abs(actions[0].astype(np.float64) - reference.astype(np.float64))))
    payload: dict[str, Any] = {
        "policy_call_idx": int(call_idx),
        "env_step": int(env_step),
        "selection_reason": str(reason),
        "noise_seeds": [int(seed) for seed in noise_seeds],
        "action_chunks_path": str(call_dir / "action_chunks.npz"),
        "action_shape": list(actions.shape),
        "variance": action_variance(list(actions)),
        "executed_sample_index": 0,
        "executed_noise_seed": int(noise_seeds[0]),
        "primary_reference_exact": exact,
        "primary_reference_max_abs": max_abs,
        "latent_paths": [None if path is None else str(path) for path in (latent_paths or [])],
    }
    (call_dir / "metadata.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def multisample_event_summary(records: Sequence[Mapping[str, Any]], *, enabled: bool, sample_count: int) -> dict[str, Any]:
    return {
        "enabled": bool(enabled),
        "sample_count": int(sample_count),
        "executed_sample_index": 0,
        "records": [dict(item) for item in records],
        "validation": {
            "all_primary_reference_exact": bool(records) and all(
                item.get("primary_reference_exact") is True for item in records
            ),
            "recorded_calls": int(len(records)),
        },
    }

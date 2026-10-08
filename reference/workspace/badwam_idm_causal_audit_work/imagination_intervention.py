from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(str(tuple(tensor.shape)).encode())
    digest.update(memoryview(tensor.reshape(-1).view(torch.uint8).numpy()))
    return digest.hexdigest()


def tensor_summary(value: torch.Tensor) -> dict[str, Any]:
    tensor = value.detach().float().cpu()
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "device": str(value.device),
        "sha256": tensor_sha256(value),
        "mean": float(tensor.mean()),
        "std": float(tensor.std()),
        "l2": float(torch.linalg.vector_norm(tensor)),
        "minimum": float(tensor.min()),
        "maximum": float(tensor.max()),
    }


def parameter_signature(module: torch.nn.Module) -> tuple[tuple[Any, ...], ...]:
    return tuple(
        (
            name,
            id(parameter),
            int(parameter.data_ptr()),
            tuple(parameter.shape),
            str(parameter.dtype),
            str(parameter.device),
            int(parameter._version),
        )
        for name, parameter in module.named_parameters()
    )


def save_representation(path: Path, value: torch.Tensor) -> dict[str, Any]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "tensor": value.detach().to(device="cpu"),
        "summary": tensor_summary(value),
    }
    torch.save(payload, path)
    return payload["summary"]


def load_representation(path: Path) -> tuple[torch.Tensor, dict[str, Any]]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not torch.is_tensor(payload.get("tensor")):
        raise ValueError(f"Invalid saved imagination representation: {path}")
    observed = tensor_summary(payload["tensor"])
    expected = payload.get("summary", {})
    if observed.get("sha256") != expected.get("sha256"):
        raise AssertionError(f"Saved representation checksum mismatch: {path}")
    return payload["tensor"], observed


def _resolve_indices(length: int, selected: Iterable[int] | None, *, exclude_zero: bool = False) -> tuple[int, ...]:
    if selected is None:
        result = tuple(range(1 if exclude_zero else 0, length))
    else:
        result = tuple(sorted(set(int(index) for index in selected)))
    if not result:
        raise ValueError("Intervention selection is empty")
    if min(result) < 0 or max(result) >= length:
        raise IndexError(f"Selection {result} is outside [0, {length})")
    if exclude_zero and 0 in result:
        raise AssertionError("Current-observation temporal group 0 cannot be selected as future")
    return result


@dataclass
class ImaginationIntervention:
    """Parameter-free IDM interface intervention for final video latents or video K/V."""

    mode: str
    temporal_groups: Iterable[int] | None = None
    layers: Iterable[int] | None = None
    components: str = "kv"
    random_seed: int = 0

    def __post_init__(self) -> None:
        if self.mode not in {"identity", "semantic_donor", "shuffled"}:
            raise ValueError(f"Unsupported intervention mode: {self.mode}")
        if self.components not in {"k", "v", "kv"}:
            raise ValueError("components must be k, v, or kv")
        self.debug: dict[str, Any] = {
            "mode": self.mode,
            "components": self.components,
            "random_seed": int(self.random_seed),
        }

    def patch_latent(self, source: torch.Tensor, donor: torch.Tensor) -> torch.Tensor:
        if source.ndim != 5:
            raise ValueError(f"Expected video latent [B,C,T,H,W], got {tuple(source.shape)}")
        if source.shape != donor.shape or source.dtype != donor.dtype:
            raise AssertionError("Source/donor latent shape or dtype mismatch")
        selected = _resolve_indices(int(source.shape[2]), self.temporal_groups, exclude_zero=True)
        patched = source.clone()
        replacement = donor[:, :, selected].clone()
        permutation = None
        if self.mode == "shuffled":
            # Shuffle complete future spatiotemporal sites while preserving every
            # channel vector and therefore the exact empirical marginal values.
            flat = replacement.permute(0, 2, 3, 4, 1).reshape(-1, replacement.shape[1])
            generator = torch.Generator(device="cpu").manual_seed(int(self.random_seed))
            permutation = torch.randperm(flat.shape[0], generator=generator)
            flat = flat.index_select(0, permutation.to(flat.device))
            replacement = flat.reshape(
                replacement.shape[0], len(selected), replacement.shape[3], replacement.shape[4], replacement.shape[1]
            ).permute(0, 4, 1, 2, 3).contiguous()
        patched[:, :, selected] = replacement
        if not torch.equal(patched[:, :, 0], source[:, :, 0]):
            raise AssertionError("Current-observation latent group changed")
        self.debug.update(
            {
                "intervention_point": "final stage-1 video latent before video_expert.pre_dit",
                "selected_temporal_groups": list(selected),
                "source": tensor_summary(source),
                "donor": tensor_summary(donor),
                "replacement": tensor_summary(replacement),
                "injected": tensor_summary(patched),
                "current_group_preserved_bit_exact": True,
                "permutation_sha256": None
                if permutation is None
                else tensor_sha256(permutation),
            }
        )
        return patched

    def patch_cache(
        self,
        source: Sequence[Mapping[str, torch.Tensor]],
        donor: Sequence[Mapping[str, torch.Tensor]],
        *,
        tokens_per_temporal_group: int,
    ) -> list[dict[str, torch.Tensor]]:
        if len(source) != len(donor):
            raise AssertionError("Source/donor cache layer count mismatch")
        selected_layers = _resolve_indices(len(source), self.layers)
        sequence_length = int(source[0]["k"].shape[1])
        if sequence_length % int(tokens_per_temporal_group) != 0:
            raise AssertionError("K/V sequence cannot be divided into temporal groups")
        temporal_count = sequence_length // int(tokens_per_temporal_group)
        selected_times = _resolve_indices(temporal_count, self.temporal_groups, exclude_zero=True)
        selected_tokens = torch.cat(
            [
                torch.arange(
                    time_index * tokens_per_temporal_group,
                    (time_index + 1) * tokens_per_temporal_group,
                )
                for time_index in selected_times
            ]
        )
        kinds = ("k", "v") if self.components == "kv" else (self.components,)
        result = [{kind: value.detach().clone() for kind, value in layer.items()} for layer in source]
        injected = []
        for layer_index in selected_layers:
            for kind in kinds:
                source_tensor = source[layer_index][kind]
                donor_tensor = donor[layer_index][kind]
                if source_tensor.shape != donor_tensor.shape or source_tensor.dtype != donor_tensor.dtype:
                    raise AssertionError(f"Cache mismatch at layer {layer_index}/{kind}")
                replacement = donor_tensor.index_select(1, selected_tokens.to(donor_tensor.device)).clone()
                if self.mode == "shuffled":
                    generator = torch.Generator(device="cpu").manual_seed(
                        int(self.random_seed) + layer_index * 17 + (0 if kind == "k" else 1)
                    )
                    permutation = torch.randperm(replacement.shape[1], generator=generator)
                    replacement = replacement.index_select(1, permutation.to(replacement.device))
                result[layer_index][kind][:, selected_tokens.to(source_tensor.device)] = replacement
                injected.append(
                    {
                        "layer": layer_index,
                        "component": kind,
                        "source_sha256": tensor_sha256(source_tensor),
                        "donor_sha256": tensor_sha256(donor_tensor),
                        "injected_sha256": tensor_sha256(result[layer_index][kind]),
                    }
                )
        self.debug.update(
            {
                "intervention_point": "layer-wise video K/V consumed by action queries",
                "selected_layers": list(selected_layers),
                "selected_temporal_groups": list(selected_times),
                "tokens_per_temporal_group": int(tokens_per_temporal_group),
                "injected_tensors": injected,
            }
        )
        return result

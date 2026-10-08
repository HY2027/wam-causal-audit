"""Frozen hardware-class authority guard for mechanistic experiment captures.

This module intentionally has no bypass.  A model capture must run on its
frozen authority class so that bit-/element-exact comparisons are meaningful.
"""

from __future__ import annotations

from typing import Final


FROZEN_HARDWARE_AUTHORITY: Final[dict[str, str]] = {
    "direct": "RTX4090",
    "joint": "RTX4090",
    "idm": "BLACKWELL",
    "imagewam": "RTX4090",
}


def classify_hardware_name(device_name: str) -> str:
    normalized = device_name.upper()
    if "RTX 4090" in normalized:
        return "RTX4090"
    if "BLACKWELL" in normalized or "RTX PRO 6000" in normalized:
        return "BLACKWELL"
    return "UNSUPPORTED"


def assert_hardware_authority(model: str, cuda_index: int = 0) -> str:
    """Return detected class, or abort before model construction on mismatch."""
    normalized_model = model.lower()
    if normalized_model not in FROZEN_HARDWARE_AUTHORITY:
        raise RuntimeError(
            f"HARDWARE AUTHORITY ABORT: unknown model {model!r}; "
            f"known models={sorted(FROZEN_HARDWARE_AUTHORITY)}"
        )

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("HARDWARE AUTHORITY ABORT: CUDA is not available")
    device_name = torch.cuda.get_device_name(cuda_index)
    detected = classify_hardware_name(device_name)
    expected = FROZEN_HARDWARE_AUTHORITY[normalized_model]
    if detected != expected:
        raise RuntimeError(
            "HARDWARE AUTHORITY ABORT: "
            f"model={normalized_model}, expected_class={expected}, "
            f"detected_class={detected}, device={device_name!r}. "
            "Bit-/element-exact comparisons across hardware classes are invalid."
        )
    return detected

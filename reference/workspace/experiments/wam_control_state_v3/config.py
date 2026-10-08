from __future__ import annotations
from wam_causal_audit.paths import resolve as _release_path

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


CODE_ROOT = Path(__file__).resolve().parent
RESULT_ROOT = Path(_release_path('@DATA@/wam_control_state_v3'))
RADIAL_ROOT = Path(_release_path('@WORKSPACE@/runs/radial_phase_validity_dose_sweep'))
OLD_ARTIFACT_ROOT = Path(
    _release_path('@DATA@/experiments/mechanism_guided_stress/artifacts')
)
MODELS = ("direct", "joint", "idm", "imagewam")
HARDWARE_AUTHORITY = {
    "direct": "RTX4090",
    "joint": "RTX4090",
    "idm": "BLACKWELL",
    "imagewam": "RTX4090",
}
CAUSAL_CUTS = {
    "direct": [f"DIRECT_L{index:02d}" for index in range(1, 7)],
    "joint": [f"JOINT_L{index:02d}" for index in range(1, 13)],
    "idm": ["IDM_L14"],
    "imagewam": [f"IMAGEWAM_L{index:02d}" for index in range(1, 7)],
}
DECODE_TARGETS = {
    "direct": "DIRECT_L06",
    "joint": "JOINT_L04",
    "idm": "IDM_L11",
    "imagewam": "IMAGEWAM_L04",
}
GROUP0_TASK = 0
GROUP0_STATE = 0
GROUP0_DOSE_CM = 2.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Mapping[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)

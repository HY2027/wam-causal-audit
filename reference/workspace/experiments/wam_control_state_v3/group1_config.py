from __future__ import annotations
from wam_causal_audit.paths import resolve as _release_path

import hashlib
import json
from pathlib import Path
from typing import Any

from config import CAUSAL_CUTS, DECODE_TARGETS, HARDWARE_AUTHORITY, MODELS, RESULT_ROOT


GROUP1_ROOT = RESULT_ROOT / "group1_factor_phase"
OLD_CLEAN_ROOT = Path(
    _release_path('@WORKSPACE@/runs/counterfactual_empty_location_closed_loop/rollouts/full')
)
REGISTRY_PATH = Path(
    _release_path('@WORKSPACE@/step0_tensor_inventory_work/step0_1_v2/locus_registry_v2.json')
)
AUTHORITY_PATH = Path(
    _release_path('@WORKSPACE@/step1_step2_51locus_work/step2_causal_per_locus.csv')
)

TASKS = tuple(range(5))
STATES = tuple(range(10))
PHASES = ("PREGRASP", "TRANSPORT", "PREPLACE")
GOAL_OBJECT = "basket_1"

# Frozen after the simulator-only smoke.  These thresholds are inherited from
# the already validated LIBERO geometry stack unless explicitly noted.
ATTACHMENT_CONSECUTIVE_STEPS = 3
GRIPPER_OPEN_APERTURE_M = 0.055
CLOSE_COMMAND_THRESHOLD = 0.5
OPEN_COMMAND_THRESHOLD = -0.5
TRANSPORT_OBJECT_GOAL_MIN_M = 0.20
EEF_OBJECT_RELATIVE_POSITION_TOLERANCE_M = 2e-4
POSE_POSITION_TOLERANCE_M = 2e-4
POSE_ORIENTATION_TOLERANCE_RAD = 0.003490658503988659  # 0.2 degrees
DOSE_TOLERANCE_M = 2e-4
COLLISION_PENETRATION_TOLERANCE_M = 1e-5

FACTOR_SPECS: dict[str, dict[str, Any]] = {
    "F1_ROBOT_RADIAL_PROGRESS": {
        "phases": ["PREGRASP"],
        "primary_signed_doses": [-4, -2, -1, 1, 2, 4],
        "dose_unit": "cm",
    },
    "F2_OBJECT_ROBOT_GEOMETRY": {
        "phases": ["PREGRASP"],
        "primary_signed_doses": [-4, -2, -1, 1, 2, 4],
        "stress_signed_doses": [-8, 8],
        "dose_unit": "cm",
        "relation_controls": ["OBJECT_ONLY", "OBJECT_AND_GOAL_RIGID_SHIFT"],
    },
    "F3_OBJECT_GOAL_RADIAL_PROGRESS": {
        "phases": ["TRANSPORT", "PREPLACE"],
        "primary_signed_doses": [-4, -2, -1, 1, 2, 4],
        "dose_unit": "cm",
    },
    "F4_TANGENTIAL_DISPLACEMENT": {
        "phases": ["PREGRASP"],
        "primary_signed_doses": [-4, -2, -1, 1, 2, 4],
        "dose_unit": "cm_euclidean_chord",
    },
    "F5_ORIENTATION_ONLY": {
        "phases": ["PREGRASP"],
        "primary_signed_doses": [-30, -20, -10, 10, 20, 30],
        "dose_unit": "degree",
    },
}

RUN_CONDITIONS = (
    "RECIPIENT_NATIVE",
    "DONOR_NATIVE",
    "FULL_CAUSAL_CUT_SWAP",
    "DECODE_LOCUS_SWAP",
    "NONCAUSAL_MATCHED_SWAP",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: Any) -> str:
    import numpy as np

    array = np.ascontiguousarray(np.asarray(value))
    return hashlib.sha256(memoryview(array.view(np.uint8))).hexdigest()


def observation_sha256(obs: dict[str, Any]) -> str:
    import numpy as np

    digest = hashlib.sha256()
    for key in sorted(obs):
        value = np.asarray(obs[key])
        if value.dtype.hasobject:
            continue
        value = np.ascontiguousarray(value)
        digest.update(key.encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(memoryview(value.view(np.uint8)))
    return digest.hexdigest()


def jsonable(value: Any) -> Any:
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
    except ImportError:
        pass
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(jsonable(value), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def clean_dir(model: str, task: int, state: int) -> Path:
    return OLD_CLEAN_ROOT / model / f"task_{task}" / f"state_{state:02d}" / "clean"


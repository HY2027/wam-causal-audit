from __future__ import annotations
from wam_causal_audit.paths import resolve as _release_path

import csv
import hashlib
import importlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

REPO = Path(_release_path('@WORKSPACE@/FastWAM'))
RESULTS = REPO / "results/robotwin_reduced_compute"
LEGACY_RUN = Path(
    _release_path('@DATA@/wam_factor_routing_v5/experiments/robotwin_wam_physical_mechanism_20260921T063246Z')
)
IDENTITY_RUN = Path(
    _release_path('@DATA@/wam_factor_routing_v5/experiments/robotwin_interface_proprio_continuation_20260923T042601Z')
)
WORKER_DIR = Path(
    _release_path('@DATA@/wam_factor_routing_v5/experiments/robotwin_identity_amendment_coverage_20260921T103100Z/runtime')
)
CONFIG = Path(
    _release_path('@DATA@/wam_factor_routing_v5/experiments/robotwin_finite_technical_20260920T181651Z/joint_config.json')
)
TASKS = (
    "adjust_bottle",
    "beat_block_hammer",
    "hanging_mug",
    "move_can_pot",
    "move_pillbottle_pad",
)
EXPECTED_STEPS = 10
EXPECTED_LAYERS = 30
ACTION_SHAPE = (32, 14)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def tensor_sha(tensor: Any) -> str:
    import torch

    return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(value), sort_keys=True, ensure_ascii=False) + "\n")


def atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def legacy_worker():
    if str(WORKER_DIR) not in sys.path:
        sys.path.insert(0, str(WORKER_DIR))
    return importlib.import_module("robotwin_identity_amendment_worker")


def load_source_manifest() -> list[dict[str, Any]]:
    payload = json.loads((IDENTITY_RUN / "source_manifest.json").read_text())
    rows = sorted(payload["sources"], key=lambda r: (TASKS.index(r["task_id"]), r["source_id"]))
    return rows


def original34() -> list[dict[str, Any]]:
    rows = [r for r in load_source_manifest() if r["analysis_set"] == "ORIGINAL34_RESULT_DRIVEN_FOLLOWUP"]
    if len(rows) != 34:
        raise RuntimeError(f"EXPECTED_ORIGINAL34_GOT_{len(rows)}")
    return rows


def expanded42() -> list[dict[str, Any]]:
    rows = load_source_manifest()
    if len(rows) != 42:
        raise RuntimeError(f"EXPECTED_EXPANDED42_GOT_{len(rows)}")
    return rows


def geometry(row: Mapping[str, Any]) -> dict[str, Any]:
    return legacy_worker().read_geometry(row)


def reset_random(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_backend(device: str = "cuda"):
    os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = str(REPO / "checkpoints")
    os.environ["DIFFSYNTH_SKIP_DOWNLOAD"] = "true"
    if str(REPO / "src") not in sys.path:
        sys.path.insert(0, str(REPO / "src"))
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json

    cfg = OmegaConf.create(json.loads(CONFIG.read_text()))
    reset_random(20260916)
    model = instantiate(cfg.model, model_dtype=torch.bfloat16, device=device).eval()
    checkpoint = Path(cfg.ckpt)
    payload = torch.load(checkpoint, weights_only=True, map_location="cpu", mmap=True)
    model.mot.load_state_dict(payload["mot"], strict=True)
    if model.proprio_encoder is not None:
        model.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
    del payload
    model = model.to(device).eval()
    processor = instantiate(cfg.data.train.processor).eval()
    stats = Path(cfg.EVALUATION.dataset_stats_path)
    processor.set_normalizer_from_stats(load_dataset_stats_from_json(str(stats)))
    return model, processor, cfg, checkpoint, stats


def prepare_input(source: Mapping[str, Any], geom: Mapping[str, Any], label: str, processor: Any):
    root = geom["root"]
    item = geom["inputs"][label]
    path = Path(item.get("observation", root / label / "obs.pkl"))
    image, proprio, raw = legacy_worker().preprocess(path, processor)
    return image, proprio, raw, path


def denormalize(processor: Any, action: Any) -> np.ndarray:
    return legacy_worker().denormalize(processor, action)


def readout(geom: Mapping[str, Any], label: str, joint_targets: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    worker = legacy_worker()
    eef = np.asarray([worker.fk_target(geom["contract"], row) for row in joint_targets])
    radial = eef @ geom["u"]
    context_eef = worker.recipient_eef(geom["root"], label, geom["contract"]["side"])
    radial_relative = (eef - context_eef[None, :]) @ geom["u"]
    return eef, radial, radial_relative


def model_identity(checkpoint: Path, stats: Path) -> dict[str, Any]:
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha(checkpoint),
        "config": str(CONFIG),
        "config_sha256": sha(CONFIG),
        "stats": str(stats),
        "stats_sha256": sha(stats),
    }


def gpu_identity() -> dict[str, Any]:
    import subprocess
    query_error = None
    try:
        query = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.STDOUT,
        )
        lines = [line.strip() for line in query.splitlines()]
    except Exception as exc:
        # Some containerized CUDA launchers expose the selected device to
        # PyTorch but deny NVML enumeration inside the child process.  The
        # supervisor performs the authoritative pre-launch nvidia-smi check;
        # retain this as an explicit missing in-worker audit field.
        lines = []
        query_error = repr(exc)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    result = {"CUDA_VISIBLE_DEVICES": visible, "nvidia_smi": lines, "nvidia_smi_error": query_error}
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            result["torch_visible_device"] = {
                "index": 0,
                "name": props.name,
                "total_memory_bytes": int(props.total_memory),
            }
    except Exception as exc:
        result["torch_device_error"] = repr(exc)
    return result


def timed_cuda(device: Any):
    import torch

    torch.cuda.synchronize(device)
    return time.perf_counter()

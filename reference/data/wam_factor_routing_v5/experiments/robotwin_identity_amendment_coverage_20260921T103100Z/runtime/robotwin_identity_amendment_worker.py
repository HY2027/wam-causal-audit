"""Post-hoc RoboTwin identity-amendment coverage-sensitivity worker.

This program never creates a simulator and never executes a predicted action.  It
consumes frozen ``obs.pkl`` files made by ``robotwin_full_geometry_worker.py``.

Expected source manifest (JSON)::

  {"sources": [{"source_id": "...", "task_id": "...",
                "role": "POSTHOC_IDENTITY_AMENDMENT_COVERAGE_SENSITIVITY",
                "geometry_dir": "/...", "instruction": "...",
                "policy_seed": 20260916}]}

The run registry may override ``config_root`` and ``call_timeout_seconds``.  By
default the already-audited RoboTwin Direct/Joint configs are loaded from the
finite technical acceptance run.  Calls are charged before preprocessing and
are never retried by this worker.  This worker does not create or certify a
strong-independence claim; it requires a separately frozen identity manifest.
"""
from __future__ import annotations
from wam_causal_audit.paths import resolve as _release_path

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import pickle
import random
import signal
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import runtime as rt  # noqa: E402

DEFAULT_CONFIG_ROOT = Path(
    _release_path('@DATA@/wam_factor_routing_v5/experiments/robotwin_finite_technical_20260920T181651Z')
)
EXPECTED_STEPS = 10
EXPECTED_LAYERS = 30
EXPECTED_COMPONENT_EVENTS = EXPECTED_STEPS * EXPECTED_LAYERS * 2
ACTION_SHAPE = (32, 14)
PHASE = "POSTHOC_IDENTITY_AMENDMENT_COVERAGE_SENSITIVITY"
EXPECTED_SOURCES = 8
DIRECT_ARMS = (
    "capture_Z1", "capture_Dplus", "native_Z2", "same_value_Z1", "interface_Dplus_on_Z1",
)
JOINT_ARMS = (
    "capture_Z1", "capture_Dplus", "native_Z2",
    "A00_Crecipient_Frecipient", "A10_Cdonor_Frecipient",
    "A01_Crecipient_Fdonor", "A11_Cdonor_Fdonor",
    "NODE_CURRENT_PROPAGATION_ALLOWED",
)


def tensor_hash(value: Any) -> str:
    import torch

    x = value.detach().cpu().contiguous().view(torch.uint8).numpy()
    return hashlib.sha256(x.tobytes()).hexdigest()


def model_weight_hash(model: Any) -> str:
    """Expensive but direct immutable-weight evidence, once before/after a run."""
    h = hashlib.sha256()
    modules = [("mot", model.mot)]
    if getattr(model, "proprio_encoder", None) is not None:
        modules.append(("proprio", model.proprio_encoder))
    for prefix, module in modules:
        for name, value in sorted(module.state_dict().items()):
            h.update((prefix + ":" + name + ":" + str(tuple(value.shape)) + ":" + str(value.dtype)).encode())
            h.update(value.detach().cpu().contiguous().view(__import__("torch").uint8).numpy().tobytes())
    return h.hexdigest()


def reset_random(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@contextlib.contextmanager
def alarm_timeout(seconds: int):
    """Best-effort Python alarm; the campaign supervisor must also police CUDA."""
    previous = signal.getsignal(signal.SIGALRM)

    def expired(*_: Any) -> None:
        raise TimeoutError(f"POLICY_CALL_TIMEOUT_{seconds}s")

    signal.signal(signal.SIGALRM, expired)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def load_sources(path: Path) -> list[dict[str, Any]]:
    payload = rt.read(path)
    rows = payload["sources"] if isinstance(payload, dict) else payload
    required = {"source_id", "task_id", "role", "geometry_dir", "instruction", "policy_seed"}
    for row in rows:
        missing = sorted(required - set(row))
        if missing:
            raise ValueError(f"SOURCE_MANIFEST_FIELDS:{row.get('source_id')}:{missing}")
    ids = [str(row["source_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("DUPLICATE_SOURCE_ID")
    return sorted(rows, key=lambda row: (str(row["task_id"]), str(row["source_id"])))


def reserve_policy_call(run: Path, row: Mapping[str, Any], maximum: int) -> None:
    """Atomically charge the shared campaign ledger before preprocessing."""
    lock_path = run / "ledger.lock"
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        history = rt.lines(run / "ledger.jsonl")
        if sum(item.get("event") == "CALL_STARTED" for item in history) >= maximum:
            raise RuntimeError("CAMPAIGN_POLICY_CALL_LIMIT")
        if any(item.get("event") == "CALL_STARTED" and item.get("call_id") == row["call_id"] for item in history):
            raise RuntimeError(f"NO_RETRY_ALREADY_STARTED:{row['call_id']}")
        rt.append(run / "ledger.jsonl", dict(row))


def read_geometry(row: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(row["geometry_dir"])
    result = rt.read(root / "result.json")
    development_legacy = row.get("role") == "development" and result.get("status") == "GEOMETRY_MEASUREMENTS_FINISHED"
    if not development_legacy and (result.get("status") != "COMPLETE" or not result.get("source_valid")):
        raise RuntimeError(f"GEOMETRY_SOURCE_NOT_COMPLETE:{root}")
    registered_initial = row.get("initial_state_sha256")
    if registered_initial and result.get("initial_state_hash") != registered_initial:
        raise RuntimeError(f"INITIAL_STATE_IDENTITY_DRIFT:{root}")
    input_registry_path = Path(row.get("input_registry_path", root / "input_registry.json"))
    inputs = {}
    for original in rt.read(input_registry_path):
        item = dict(original)
        label = item.get("label", item.get("name"))
        if not label:
            raise RuntimeError(f"GEOMETRY_INPUT_WITHOUT_LABEL:{input_registry_path}")
        # The frozen development run predates the campaign schema.  Its
        # post-audit registry uses model_eligible; confirmation geometry uses
        # valid.  This is a schema mapping, not a re-judgement of validity.
        if "valid" not in item:
            item["valid"] = bool(item.get("model_eligible"))
        item["label"] = label
        inputs[label] = item
    # This amendment has no Dminus arm.  Do not even synthesize an invalid
    # placeholder: its absence is part of the frozen call contract.
    for label in ("Z1", "Z2", "Dplus"):
        if label not in inputs:
            inputs[label] = {"label": label, "valid": False, "reason": "INPUT_NOT_REGISTERED"}
    if not inputs["Z1"].get("valid") or not inputs["Z2"].get("valid") or not inputs["Dplus"].get("valid"):
        raise RuntimeError("PRIMARY_GEOMETRY_INPUT_MISSING")
    contract_path = root / "fk_contract.json"
    if not contract_path.exists() and development_legacy:
        contract_path = root / "target_fk_contract.json"
    contract = rt.read(contract_path)
    if not contract.get("side") and row.get("side_override"):
        contract["side"] = row["side_override"]
    if contract.get("side") not in {"left", "right"}:
        raise RuntimeError(f"FK_CONTRACT_SIDE_MISSING:{contract_path}")
    u = np.asarray(contract["u"], dtype=np.float64)
    if u.shape != (3,) or not np.isfinite(u).all() or abs(np.linalg.norm(u) - 1.0) > 1e-9:
        raise RuntimeError("INVALID_FROZEN_RADIAL_AXIS")
    return {"root": root, "inputs": inputs, "contract": contract, "u": u, "result": result}


def fk_target(contract: Mapping[str, Any], action: np.ndarray) -> np.ndarray:
    """Named-joint FK copied from the audited RoboTwin readout contract."""
    q = dict(contract["qpos_by_name"])
    q.update(zip(contract["left_names"], np.asarray(action[:6], dtype=np.float64)))
    q.update(zip(contract["right_names"], np.asarray(action[7:13], dtype=np.float64)))
    transform = np.asarray(contract["root"], dtype=np.float64)
    for joint in contract["chain"]:
        motion = np.eye(4, dtype=np.float64)
        value = float(q.get(joint["name"], 0.0))
        kind = str(joint["type"])
        if int(joint["dof"]):
            if "revolute" in kind:
                motion[1:3, 1:3] = [[np.cos(value), -np.sin(value)], [np.sin(value), np.cos(value)]]
            elif "prismatic" in kind:
                motion[0, 3] = value
            else:
                raise RuntimeError(f"UNSUPPORTED_FK_JOINT:{kind}")
        transform = (
            transform
            @ np.asarray(joint["parent_pose"], dtype=np.float64)
            @ motion
            @ np.linalg.inv(np.asarray(joint["child_pose"], dtype=np.float64))
        )
    transform = transform @ np.asarray(contract["joint_pose_in_child"], dtype=np.float64)
    rotation = (
        transform[:3, :3]
        @ np.asarray(contract["global_matrix"], dtype=np.float64)
        @ np.asarray(contract["delta_matrix"], dtype=np.float64)
    )
    return transform[:3, 3] + rotation @ np.asarray([contract["bias"], 0.0, 0.0])


def recipient_eef(root: Path, label: str, side: str) -> np.ndarray:
    path = root / label / "eef.pkl"
    if path.exists():
        with path.open("rb") as stream:
            values = pickle.load(stream)
        return np.asarray(values[side], dtype=np.float64)[:3]
    # Frozen contact-v2 development assets used a prior field layout.  The
    # value is the directly recorded simulator EEF at the same saved input.
    legacy = root / label / "after_assignment_and_render" / "simulator_eef.pkl"
    if legacy.exists():
        with legacy.open("rb") as stream:
            values = pickle.load(stream)
        return np.asarray(values[0 if side == "left" else 1], dtype=np.float64)[:3]
    raise FileNotFoundError(f"EEF_RECORD_MISSING:{path}:{legacy}")


class DirectController:
    """Capture/replace the full action-facing video cache at each actual read."""

    def __init__(
        self,
        model: Any,
        mode: str,
        bank: Mapping[tuple[int, int, str], Any] | None = None,
        bank_label: str | None = None,
    ):
        self.model = model
        self.mode = mode
        self.bank = bank
        self.bank_label = bank_label
        self.original = model.mot.forward_action_with_video_cache
        self.events: list[dict[str, Any]] = []
        self.captured: dict[tuple[int, int, str], Any] = {}
        self.step = 0
        self.cleanup = False

    def install(self) -> None:
        outer = self

        def wrapped(*args: Any, **kwargs: Any):
            if args:
                raise RuntimeError("DIRECT_CACHE_SIGNATURE_POSITIONAL")
            if outer.step >= EXPECTED_STEPS:
                raise RuntimeError("DIRECT_EXTRA_ACTION_STEP")
            rows = list(kwargs["video_kv_cache"])
            if len(rows) != EXPECTED_LAYERS:
                raise RuntimeError(f"DIRECT_LAYER_COUNT:{len(rows)}")
            consumed: set[tuple[int, int, str]] = set()
            wrapped_rows = []
            for layer, source_row in enumerate(rows):
                row = dict(source_row)
                for component in ("k", "v"):
                    key = (outer.step, layer, component)
                    natural = row[component]
                    if outer.mode == "capture":
                        outer.captured[key] = natural.detach().cpu().clone()
                        written = natural
                        source = outer.bank_label or "CURRENT_CALL"
                    elif outer.mode == "replace":
                        if outer.bank is None or key not in outer.bank:
                            raise RuntimeError(f"DIRECT_BANK_MISSING:{key}")
                        written = outer.bank[key].to(device=natural.device, dtype=natural.dtype).clone()
                        if written.shape != natural.shape:
                            raise RuntimeError(f"DIRECT_BANK_LAYOUT:{key}")
                        row[component] = written
                        source = outer.bank_label or "FROZEN_BANK"
                    else:
                        raise RuntimeError("DIRECT_CONTROLLER_MODE")
                    outer.events.append({
                        "step": outer.step,
                        "layer": layer,
                        "component": component,
                        "natural_hash": tensor_hash(natural),
                        "written_hash": tensor_hash(written),
                        "source": source,
                    })

                class ReadRow(dict):
                    def __init__(self, values: Mapping[str, Any], layer_index: int):
                        super().__init__(values)
                        self.layer_index = layer_index

                    def __getitem__(self, component: str):
                        value = super().__getitem__(component)
                        if component in ("k", "v"):
                            key = (outer.step, self.layer_index, component)
                            consumed.add(key)
                            expected = next(x["written_hash"] for x in outer.events if (x["step"], x["layer"], x["component"]) == key)
                            observed = tensor_hash(value)
                            if observed != expected:
                                raise RuntimeError(f"DIRECT_WRITTEN_VALUE_NOT_CONSUMED:{key}")
                        return value

                wrapped_rows.append(ReadRow(row, layer))
            kwargs = dict(kwargs, video_kv_cache=wrapped_rows)
            result = outer.original(**kwargs)
            expected = {(outer.step, layer, component) for layer in range(EXPECTED_LAYERS) for component in ("k", "v")}
            if consumed != expected:
                raise RuntimeError(f"DIRECT_CONSUMER_COVERAGE:{len(consumed)}")
            outer.step += 1
            return result

        self.model.mot.forward_action_with_video_cache = wrapped

    def uninstall(self) -> None:
        self.model.mot.forward_action_with_video_cache = self.original
        self.cleanup = self.model.mot.forward_action_with_video_cache == self.original

    def validate(self) -> None:
        if self.step != EXPECTED_STEPS or len(self.events) != EXPECTED_COMPONENT_EVENTS:
            raise RuntimeError(f"DIRECT_EVENT_COVERAGE:{self.step}:{len(self.events)}")
        if self.mode == "capture" and len(self.captured) != EXPECTED_COMPONENT_EVENTS:
            raise RuntimeError("DIRECT_CAPTURE_COVERAGE")


class JointController:
    """Post-RoPE/post-projection C/F capture and strict consumer-edge write."""

    def __init__(
        self,
        model: Any,
        mode: str,
        current_bank: Mapping[tuple[int, int, str], Any] | None = None,
        future_bank: Mapping[tuple[int, int, str], Any] | None = None,
        current_label: str | None = None,
        future_label: str | None = None,
    ):
        self.model = model
        self.mode = mode
        self.current_bank = current_bank
        self.future_bank = future_bank
        self.current_label = current_label
        self.future_label = future_label
        self.original_build = model.mot._build_expert_attention_io
        self.original_mix = model.mot._mixed_attention
        self.block_map = {id(block): ("video", i) for i, block in enumerate(model.video_expert.blocks)}
        self.block_map.update({id(block): ("action", i) for i, block in enumerate(model.action_expert.blocks)})
        self.video_calls = 0
        self.action_calls = 0
        self.mix_calls = 0
        self.events: list[dict[str, Any]] = []
        self.captured: dict[tuple[int, int, str], Any] = {}
        self.pending: dict[str, Any] | None = None
        self.tokens_per_group: int | None = None
        self.cleanup = False

    def install(self) -> None:
        outer = self

        def build(expert: Any, block: Any, *args: Any, **kwargs: Any):
            output = list(outer.original_build(expert, block, *args, **kwargs))
            modality, layer = outer.block_map[id(block)]
            if modality == "action":
                outer.action_calls += 1
                return tuple(output)
            step, expected_layer = divmod(outer.video_calls, EXPECTED_LAYERS)
            if step >= EXPECTED_STEPS or expected_layer != layer:
                raise RuntimeError(f"JOINT_VIDEO_EVENT_ORDER:{step}:{expected_layer}:{layer}")
            outer.video_calls += 1
            token_count = int(output[1].shape[1])
            if token_count % 3:
                raise RuntimeError(f"JOINT_TEMPORAL_LAYOUT:{token_count}")
            per_group = token_count // 3
            outer.tokens_per_group = per_group if outer.tokens_per_group is None else outer.tokens_per_group
            if outer.tokens_per_group != per_group:
                raise RuntimeError("JOINT_TEMPORAL_LAYOUT_DRIFT")
            expected_hashes: dict[str, str] = {}
            for component, slot in (("k", 1), ("v", 2)):
                key = (step, layer, component)
                natural = output[slot]
                if outer.mode == "capture":
                    outer.captured[key] = natural.detach().cpu().clone()
                    written = natural
                    source_current = outer.current_label or "CURRENT_CALL"
                    source_future = outer.future_label or outer.current_label or "CURRENT_CALL"
                elif outer.mode == "strict":
                    if outer.current_bank is None or outer.future_bank is None:
                        raise RuntimeError("JOINT_STRICT_REQUIRES_BOTH_BANKS")
                    current = outer.current_bank[key].to(device=natural.device, dtype=natural.dtype)[:, :per_group]
                    future = outer.future_bank[key].to(device=natural.device, dtype=natural.dtype)[:, per_group:]
                    written = __import__("torch").cat((current, future), dim=1)
                    source_current = outer.current_label or "FROZEN_CURRENT_BANK"
                    source_future = outer.future_label or "FROZEN_FUTURE_BANK"
                    output[slot] = written
                elif outer.mode == "node_current":
                    if outer.current_bank is None:
                        raise RuntimeError("JOINT_NODE_REQUIRES_CURRENT_BANK")
                    current = outer.current_bank[key].to(device=natural.device, dtype=natural.dtype)[:, :per_group]
                    written = natural.clone()
                    written[:, :per_group] = current
                    output[slot] = written
                    source_current = outer.current_label or "FROZEN_CURRENT_BANK"
                    source_future = "PROPAGATION_ALLOWED_DYNAMIC_FUTURE"
                else:
                    raise RuntimeError("JOINT_CONTROLLER_MODE")
                expected_hashes[component] = tensor_hash(written)
                outer.events.append({
                    "step": step,
                    "layer": layer,
                    "component": component,
                    "location": "_build_expert_attention_io return, post-RoPE K/post-projection V, before mixed attention",
                    "natural_hash": tensor_hash(natural),
                    "written_hash": expected_hashes[component],
                    "current_hash": tensor_hash(written[:, :per_group]),
                    "future_hash": tensor_hash(written[:, per_group:]),
                    "current_source": source_current,
                    "future_source": source_future,
                    "tokens_per_group": per_group,
                })
            outer.pending = {"step": step, "layer": layer, "length": token_count, "hashes": expected_hashes}
            return tuple(output)

        def mixed(*args: Any, **kwargs: Any):
            if outer.pending is None:
                raise RuntimeError("JOINT_MIX_BEFORE_VIDEO_BUILD")
            if args:
                raise RuntimeError("JOINT_MIX_POSITIONAL_SIGNATURE")
            pending = outer.pending
            for component in ("k", "v"):
                key = component + "_cat"
                value = kwargs[key][:, : pending["length"]]
                if tensor_hash(value) != pending["hashes"][component]:
                    raise RuntimeError(
                        f"JOINT_WRITTEN_VALUE_NOT_CONSUMED:{pending['step']}:{pending['layer']}:{component}"
                    )
            outer.mix_calls += 1
            outer.pending = None
            return outer.original_mix(**kwargs)

        self.model.mot._build_expert_attention_io = build
        self.model.mot._mixed_attention = mixed

    def uninstall(self) -> None:
        self.model.mot._build_expert_attention_io = self.original_build
        self.model.mot._mixed_attention = self.original_mix
        self.cleanup = (
            self.model.mot._build_expert_attention_io == self.original_build
            and self.model.mot._mixed_attention == self.original_mix
        )

    def validate(self) -> None:
        if self.pending is not None:
            raise RuntimeError("JOINT_UNCONSUMED_FINAL_EVENT")
        if self.video_calls != EXPECTED_STEPS * EXPECTED_LAYERS:
            raise RuntimeError(f"JOINT_VIDEO_COVERAGE:{self.video_calls}")
        if self.action_calls != EXPECTED_STEPS * EXPECTED_LAYERS:
            raise RuntimeError(f"JOINT_ACTION_COVERAGE:{self.action_calls}")
        if self.mix_calls != EXPECTED_STEPS * EXPECTED_LAYERS:
            raise RuntimeError(f"JOINT_MIX_COVERAGE:{self.mix_calls}")
        if len(self.events) != EXPECTED_COMPONENT_EVENTS:
            raise RuntimeError(f"JOINT_COMPONENT_COVERAGE:{len(self.events)}")
        if self.mode == "capture" and len(self.captured) != EXPECTED_COMPONENT_EVENTS:
            raise RuntimeError("JOINT_CAPTURE_COVERAGE")


def preprocess(obs_path: Path, processor: Any):
    import torch
    from PIL import Image

    with obs_path.open("rb") as stream:
        obs = pickle.load(stream)
    cameras = obs["observation"]

    def resize(name: str, size: tuple[int, int]) -> np.ndarray:
        return np.asarray(Image.fromarray(cameras[name]["rgb"]).resize(size, Image.Resampling.BILINEAR))

    rgb = np.concatenate(
        [
            resize("head_camera", (320, 256)),
            np.concatenate([resize("left_camera", (160, 128)), resize("right_camera", (160, 128))], axis=1),
        ],
        axis=0,
    )
    image = torch.from_numpy(rgb.copy()).permute(2, 0, 1).unsqueeze(0).to("cuda", torch.bfloat16) * (2 / 255.0) - 1
    state = torch.as_tensor(np.asarray(obs["joint_action"]["vector"], dtype=np.float32)).unsqueeze(0)
    transformed = processor.action_state_transform({"state": {"default": state}})
    proprio = processor.normalizer.forward(transformed)["state"]["default"]
    return image, proprio, obs


def denormalize(processor: Any, action: Any) -> np.ndarray:
    key = processor.action_meta[0]["key"] if hasattr(processor, "action_meta") else processor.shape_meta["action"][0]["key"]
    return (
        processor.normalizer.normalizers["action"][key]
        .backward(action.detach().cpu().float().unsqueeze(0))
        .squeeze(0)
        .numpy()
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--kind", choices=("direct", "joint"), required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--phase", choices=(PHASE,), required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    run.mkdir(parents=True, exist_ok=True)
    registry = rt.read(run / "registry.json")
    output = run / f"{args.kind}_{args.phase}"
    output.mkdir(parents=True, exist_ok=True)
    if (output / "result.json").exists():
        raise FileExistsError(f"REFUSE_OVERWRITE_COMPLETED_BRANCH:{output}")
    expected_worker = registry.get("model_worker_sha256")
    if expected_worker and rt.sha(Path(__file__)) != expected_worker:
        raise RuntimeError("MODEL_WORKER_CODE_DRIFT")
    if registry.get("registry_status") != "IDENTITY_MANIFEST_FROZEN":
        raise RuntimeError("BLOCKED_PENDING_FROZEN_IDENTITY_MANIFEST")
    if registry.get("scientific_identity") != PHASE:
        raise RuntimeError("SCIENTIFIC_IDENTITY_MISMATCH")
    if registry.get("strong_independence_claim") is not False:
        raise RuntimeError("STRONG_INDEPENDENCE_CLAIM_MUST_BE_FALSE")
    if rt.sha(args.sources) != registry.get("source_manifest_sha256"):
        raise RuntimeError("SOURCE_MANIFEST_IDENTITY_DRIFT")
    source_payload = rt.read(args.sources)
    if source_payload.get("status") != "IDENTITY_MANIFEST_FROZEN":
        raise RuntimeError("SOURCE_MANIFEST_NOT_FROZEN")
    if source_payload.get("scientific_identity") != PHASE:
        raise RuntimeError("SOURCE_MANIFEST_PHASE_MISMATCH")
    if source_payload.get("strong_independence_claim") is not False:
        raise RuntimeError("SOURCE_MANIFEST_OVERCLAIMS_INDEPENDENCE")
    source_rows = [row for row in load_sources(args.sources) if row["role"] == PHASE]
    if len(source_rows) != EXPECTED_SOURCES:
        raise RuntimeError(f"EXPECTED_8_FROZEN_SOURCES_GOT_{len(source_rows)}")
    for row in source_rows:
        initial_hash = str(row.get("initial_state_sha256", ""))
        source_group = str(row.get("source_group_id", ""))
        if len(initial_hash) != 64 or not source_group:
            raise RuntimeError(f"POSTHOC_SOURCE_IDENTITY_INCOMPLETE:{row['source_id']}")
        if row.get("identity_status") != "REGISTERED_FOR_POSTHOC_COVERAGE_SENSITIVITY":
            raise RuntimeError(f"POSTHOC_IDENTITY_STATUS:{row['source_id']}")
        if row.get("strong_independence_claim") is not False:
            raise RuntimeError(f"POSTHOC_SOURCE_OVERCLAIMS_INDEPENDENCE:{row['source_id']}")
        if not row.get("ancestry_status") or not row.get("prior_exposure_status"):
            raise RuntimeError(f"POSTHOC_IDENTITY_METADATA_MISSING:{row['source_id']}")
    timeout_seconds = int(registry.get("call_timeout_seconds", 600))
    max_calls = int(registry.get("maximum_policy_call_attempts", 0))
    if max_calls != 120:
        raise RuntimeError(f"POSTHOC_CALL_BUDGET_DRIFT:{max_calls}")
    config_root = Path(registry.get("config_root", DEFAULT_CONFIG_ROOT))
    config_path = Path(registry.get(f"{args.kind}_config", config_root / f"{args.kind}_config.json"))
    if not config_path.is_file():
        raise FileNotFoundError(config_path)

    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(InterruptedError("SUPERVISOR_STOP")))
    load_started = time.monotonic()
    rt.append(run / "ledger.jsonl", {"event": "MODEL_LOAD_STARTED", "model": args.kind, "phase": args.phase, "time": time.time()})
    try:
        os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = _release_path('@WORKSPACE@/FastWAM/checkpoints')
        os.environ["DIFFSYNTH_SKIP_DOWNLOAD"] = "true"
        if _release_path('@WORKSPACE@/FastWAM/src') not in sys.path:
            sys.path.insert(0, _release_path('@WORKSPACE@/FastWAM/src'))
        import torch
        from hydra.utils import instantiate
        from omegaconf import OmegaConf
        from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
        from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json

        cfg = OmegaConf.create(rt.read(config_path))
        cap_mib = float(os.environ.get("CAMPAIGN_TASK_MEMORY_CAP_MIB", "0"))
        if cap_mib <= 0:
            raise RuntimeError("SUPERVISOR_MEMORY_CAP_MISSING")
        total_mib = torch.cuda.get_device_properties(0).total_memory / (1024.0 * 1024.0)
        memory_fraction = min(0.95, cap_mib / total_mib)
        if memory_fraction <= 0:
            raise RuntimeError("SUPERVISOR_MEMORY_CAP_NONPOSITIVE")
        torch.cuda.set_per_process_memory_fraction(memory_fraction, device=0)
        rt.save(
            output / "allocator_cap.json",
            {
                "task_cap_mib": cap_mib,
                "device_total_mib": total_mib,
                "per_process_fraction": memory_fraction,
                "set_before_model_construction": True,
                "whole_device_limit_fraction": 0.95,
            },
        )
        torch.set_num_threads(int(registry.get("torch_cpu_threads", 16)))
        load_seed = int(registry.get("model_load_seed", 20260916))
        reset_random(load_seed)
        model = instantiate(cfg.model, model_dtype=torch.bfloat16, device="cuda").eval()
        checkpoint_path = Path(cfg.ckpt)
        checkpoint_hash = rt.sha(checkpoint_path)
        payload = torch.load(checkpoint_path, weights_only=True, map_location="cpu", mmap=True)
        model.mot.load_state_dict(payload["mot"], strict=True)
        if model.proprio_encoder is not None:
            model.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
        del payload
        processor = instantiate(cfg.data.train.processor).eval()
        stats_path = Path(cfg.EVALUATION.dataset_stats_path)
        processor.set_normalizer_from_stats(load_dataset_stats_from_json(stats_path))
        before_weights = model_weight_hash(model)
        load_status = "PASS"
    except BaseException as error:
        rt.append(
            run / "ledger.jsonl",
            {
                "event": "MODEL_LOAD_FINISHED",
                "model": args.kind,
                "phase": args.phase,
                "status": "FAILED",
                "seconds": time.monotonic() - load_started,
                "error": repr(error),
            },
        )
        raise
    rt.append(
        run / "ledger.jsonl",
        {
            "event": "MODEL_LOAD_FINISHED",
            "model": args.kind,
            "phase": args.phase,
            "status": load_status,
            "seconds": time.monotonic() - load_started,
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_hash,
            "stats": str(stats_path),
            "stats_sha256": rt.sha(stats_path),
            "config": str(config_path),
            "config_sha256": rt.sha(config_path),
        },
    )

    call_summaries: list[dict[str, Any]] = []
    source_summaries: list[dict[str, Any]] = []

    def call(
        source: Mapping[str, Any],
        geometry: Mapping[str, Any],
        label: str,
        context_label: str,
        controller: DirectController | JointController | None,
    ) -> tuple[Any, dict[str, Any]] | None:
        call_id = f"{args.kind}__{args.phase}__{source['source_id']}__{label}"
        item = geometry["inputs"][context_label]
        obs_path = Path(item.get("observation", geometry["root"] / context_label / "obs.pkl"))
        started_row = {
            "event": "CALL_STARTED",
            "call_id": call_id,
            "model": args.kind,
            "phase": args.phase,
            "source_id": source["source_id"],
            "task_id": source["task_id"],
            "arm": label,
            "context_label": context_label,
            "time": time.time(),
            "policy_call_attempts_delta": 1,
        }
        reserve_policy_call(run, started_row, max_calls)
        rt.save(output / "active_call.json", started_row)
        call_started = time.monotonic()
        status, error = "FAILED", None
        hook_cleanup: dict[str, Any] = {"installed": controller is not None, "success": controller is None}
        event_path = output / "events" / str(source["source_id"]) / f"{label}.json"
        event_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            seed = int(source["policy_seed"])
            reset_random(seed)
            image, proprio, _ = preprocess(obs_path, processor)
            input_identity = {
                "observation": str(obs_path),
                "observation_sha256": rt.sha(obs_path),
                "image_sha256": tensor_hash(image),
                "proprio_sha256": tensor_hash(proprio),
                "instruction_sha256": hashlib.sha256(str(source["instruction"]).encode()).hexdigest(),
                "seed": seed,
            }
            kwargs = {
                "prompt": DEFAULT_PROMPT.format(task=source["instruction"]),
                "input_image": image,
                "proprio": proprio,
                "action_horizon": 32,
                "num_inference_steps": 10,
                "seed": seed,
                "rand_device": "cpu",
                "text_cfg_scale": 1.0,
                "negative_prompt": "",
                "sigma_shift": None,
                "tiled": False,
            }
            if args.kind == "joint":
                kwargs["num_video_frames"] = 9
            if controller is not None:
                controller.install()
            try:
                with alarm_timeout(timeout_seconds), torch.inference_mode():
                    result = model.infer_action(**kwargs)
                    action = result["action"].detach().cpu()
                    torch.cuda.synchronize()
            finally:
                if controller is not None:
                    controller.uninstall()
                    hook_cleanup = {"installed": True, "success": controller.cleanup}
            if controller is not None:
                controller.validate()
            if tuple(action.shape) != ACTION_SHAPE or not bool(torch.isfinite(action).all()):
                raise RuntimeError(f"ACTION_INVALID:{tuple(action.shape)}")
            joint_targets = denormalize(processor, action)
            eef_world = np.asarray([fk_target(geometry["contract"], row) for row in joint_targets])
            radial_world = eef_world @ geometry["u"]
            current_eef = recipient_eef(geometry["root"], context_label, geometry["contract"]["side"])
            radial_relative = (eef_world - current_eef[None, :]) @ geometry["u"]
            artifact = output / "actions" / str(source["source_id"]) / f"{label}.npz"
            atomic_npz(
                artifact,
                normalized=action.float().numpy(),
                joint_targets=joint_targets,
                eef_target_world=eef_world,
                radial_world=radial_world,
                radial_relative_to_context_eef=radial_relative,
                u=geometry["u"],
                context_eef=current_eef,
            )
            if controller is not None:
                rt.save(event_path, controller.events)
            summary = {
                "call_id": call_id,
                "label": label,
                "context_label": context_label,
                "status": "PASS",
                "input": input_identity,
                "action_artifact": str(artifact),
                "action_artifact_sha256": rt.sha(artifact),
                "normalized_action_sha256": tensor_hash(action),
                "consumer_component_events": 0 if controller is None else len(controller.events),
                "event_log": None if controller is None else str(event_path),
                "event_log_sha256": None if controller is None else rt.sha(event_path),
                "hook_cleanup": hook_cleanup,
                "predicted_action_executed": False,
            }
            call_summaries.append(summary)
            status = "PASS"
            return action, summary
        except BaseException as exc:
            error = repr(exc)
            call_summaries.append(
                {
                    "call_id": call_id,
                    "label": label,
                    "context_label": context_label,
                    "status": "FAILED",
                    "error": error,
                    "hook_cleanup": hook_cleanup,
                    "predicted_action_executed": False,
                }
            )
            return None
        finally:
            if controller is not None and controller.events:
                # Preserve partial coverage if a hook or numerical check fails.
                rt.save(event_path, controller.events)
            rt.save(output / "call_summaries.json", call_summaries)
            rt.append(
                run / "ledger.jsonl",
                {
                    **started_row,
                    "event": "CALL_FINISHED",
                    "policy_call_attempts_delta": 0,
                    "status": status,
                    "error": error,
                    "seconds": time.monotonic() - call_started,
                    "hook_cleanup": hook_cleanup,
                },
            )
            rt.save(output / "active_call.json", {"event": "IDLE"})

    for source in source_rows:
        source_id = str(source["source_id"])
        geometry = read_geometry(source)
        per_source: dict[str, Any] = {
            "source_id": source_id,
            "task_id": source["task_id"],
            "role": source["role"],
            "geometry_dir": str(geometry["root"]),
            "geometry_result_sha256": rt.sha(geometry["root"] / "result.json"),
            "calls": [],
            "status": "STARTED",
        }
        try:
            def assert_recipient_identity(measured: tuple[Any, dict[str, Any]], reference: tuple[Any, dict[str, Any]]) -> None:
                keys = ("observation_sha256", "image_sha256", "proprio_sha256", "instruction_sha256", "seed")
                left = {key: measured[1]["input"][key] for key in keys}
                right = {key: reference[1]["input"][key] for key in keys}
                if left != right:
                    raise RuntimeError(f"NON_TARGET_RECIPIENT_INPUT_DRIFT:{left}:{right}")

            if args.kind == "direct":
                z1_controller = DirectController(model, "capture", bank_label="Z1_CURRENT_CALL")
                z1 = call(source, geometry, "capture_Z1", "Z1", z1_controller)
                if z1 is None:
                    raise RuntimeError("DEPENDENCY_CAPTURE_Z1")
                donor_controller = DirectController(model, "capture", bank_label="DPLUS_CURRENT_CALL")
                donor = call(source, geometry, "capture_Dplus", "Dplus", donor_controller)
                if donor is None:
                    raise RuntimeError("DEPENDENCY_CAPTURE_DPLUS")
                call(source, geometry, "native_Z2", "Z2", None)
                same_controller = DirectController(model, "replace", z1_controller.captured, "Z1_FROZEN_BANK")
                same = call(source, geometry, "same_value_Z1", "Z1", same_controller)
                swap_controller = DirectController(model, "replace", donor_controller.captured, "DPLUS_FROZEN_BANK")
                swap = call(source, geometry, "interface_Dplus_on_Z1", "Z1", swap_controller)
                if same is None or swap is None:
                    raise RuntimeError("DEPENDENCY_DIRECT_CROSS_SOURCE")
                assert_recipient_identity(same, z1)
                assert_recipient_identity(swap, z1)
                same_exact = bool(__import__("torch").equal(same[0], z1[0]))
                same_error = float((same[0] - z1[0]).abs().max())
                if not same_exact:
                    raise RuntimeError(f"DIRECT_SAME_VALUE_NOT_EXACT:{same_error}")
                per_source.update(
                    direct_same_value_bit_exact=same_exact,
                    direct_same_value_max_abs=same_error,
                    direct_consumer_events=EXPECTED_COMPONENT_EVENTS,
                    direct_other_conditions="recipient instruction and proprio fixed for interface swap",
                )
            else:
                z1_controller = JointController(model, "capture", current_label="Z1_CURRENT_CALL", future_label="Z1_CURRENT_CALL")
                z1 = call(source, geometry, "capture_Z1", "Z1", z1_controller)
                if z1 is None:
                    raise RuntimeError("DEPENDENCY_CAPTURE_Z1")
                donor_controller = JointController(model, "capture", current_label="DPLUS_CURRENT_CALL", future_label="DPLUS_CURRENT_CALL")
                donor = call(source, geometry, "capture_Dplus", "Dplus", donor_controller)
                if donor is None:
                    raise RuntimeError("DEPENDENCY_CAPTURE_DPLUS")
                call(source, geometry, "native_Z2", "Z2", None)
                arms = {
                    "A00_Crecipient_Frecipient": (z1_controller.captured, z1_controller.captured, "Z1", "Z1"),
                    "A10_Cdonor_Frecipient": (donor_controller.captured, z1_controller.captured, "Dplus", "Z1"),
                    "A01_Crecipient_Fdonor": (z1_controller.captured, donor_controller.captured, "Z1", "Dplus"),
                    "A11_Cdonor_Fdonor": (donor_controller.captured, donor_controller.captured, "Dplus", "Dplus"),
                }
                arm_actions: dict[str, Any] = {}
                for label, (current_bank, future_bank, current_label, future_label) in arms.items():
                    controller = JointController(
                        model,
                        "strict",
                        current_bank,
                        future_bank,
                        current_label=current_label,
                        future_label=future_label,
                    )
                    measured = call(source, geometry, label, "Z1", controller)
                    if measured is None:
                        raise RuntimeError(f"DEPENDENCY_JOINT_STRICT:{label}")
                    assert_recipient_identity(measured, z1)
                    arm_actions[label] = measured[0]
                node_controller = JointController(
                    model,
                    "node_current",
                    donor_controller.captured,
                    None,
                    current_label="Dplus",
                    future_label="PROPAGATION_ALLOWED_DYNAMIC_FUTURE",
                )
                node = call(source, geometry, "NODE_CURRENT_PROPAGATION_ALLOWED", "Z1", node_controller)
                if node is None:
                    raise RuntimeError("DEPENDENCY_JOINT_NODE")
                assert_recipient_identity(node, z1)
                same_exact = bool(__import__("torch").equal(arm_actions["A00_Crecipient_Frecipient"], z1[0]))
                same_error = float((arm_actions["A00_Crecipient_Frecipient"] - z1[0]).abs().max())
                if not same_exact:
                    raise RuntimeError(f"JOINT_A00_NOT_EXACT:{same_error}")
                per_source.update(
                    joint_A00_bit_exact=same_exact,
                    joint_A00_max_abs=same_error,
                    joint_tokens_per_temporal_group=z1_controller.tokens_per_group,
                    joint_consumption_scope="30 layers x 10 denoising steps, post-RoPE K/post-projection V",
                    joint_other_conditions="recipient image path, instruction, proprio, action noise and scheduler fixed for A00/A10/A01/A11",
                    node_interpretation="current replaced at every upstream node; future not fixed and may carry propagated changes",
                )
            per_source["status"] = "PASS"
        except BaseException as exc:
            per_source["status"] = "FAILED"
            per_source["error"] = repr(exc)
            # This is a fixed post-hoc coverage sensitivity set.  A local
            # failure is retained; independent runnable rows continue.
            source_summaries.append(per_source)
            rt.save(output / "source_summaries.json", source_summaries)
            rt.save(output / "call_summaries.json", call_summaries)
            continue
        source_summaries.append(per_source)
        rt.save(output / "source_summaries.json", source_summaries)
        rt.save(output / "call_summaries.json", call_summaries)
        import torch

        torch.cuda.empty_cache()

    after_weights = model_weight_hash(model)
    weights_unchanged = before_weights == after_weights
    rt.save(
        output / "weights.json",
        {"before": before_weights, "after": after_weights, "unchanged": weights_unchanged},
    )
    if not weights_unchanged:
        raise RuntimeError("MODEL_WEIGHT_MUTATION")
    passed = sum(row["status"] == "PASS" for row in source_summaries)
    failed = len(source_summaries) - passed
    rt.save(
        output / "result.json",
        {
            "status": "COMPLETE" if failed == 0 else "COMPLETE_WITH_SOURCE_FAILURES",
            "model": args.kind,
            "phase": args.phase,
            "source_count": len(source_summaries),
            "passed_sources": passed,
            "failed_sources": failed,
            "policy_call_attempts": len(call_summaries),
            "policy_call_pass": sum(row["status"] == "PASS" for row in call_summaries),
            "policy_call_failed": sum(row["status"] == "FAILED" for row in call_summaries),
            "predicted_actions_executed": 0,
            "simulator_initializations": 0,
            "physics_steps": 0,
            "checkpoint_sha256": checkpoint_hash,
            "config_sha256": rt.sha(config_path),
            "stats_sha256": rt.sha(stats_path),
            "worker_sha256": rt.sha(Path(__file__)),
            "weights_unchanged": True,
            "scientific_identity": PHASE,
            "strong_independence_claim": False,
            "fixed_arms": list(DIRECT_ARMS if args.kind == "direct" else JOINT_ARMS),
            "dminus_calls": 0,
        },
    )


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        # The destination may not be known if argument parsing itself failed.
        try:
            arguments = sys.argv
            run_index = arguments.index("--run") + 1
            kind_index = arguments.index("--kind") + 1
            phase_index = arguments.index("--phase") + 1
            destination = Path(arguments[run_index]) / f"{arguments[kind_index]}_{arguments[phase_index]}"
            destination.mkdir(parents=True, exist_ok=True)
            rt.save(destination / "failure.json", {"error": repr(exc), "traceback": traceback.format_exc()})
        except Exception:
            pass
        raise

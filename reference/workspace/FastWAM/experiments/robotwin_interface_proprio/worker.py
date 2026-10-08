#!/usr/bin/env python3
"""RoboTwin interface x proprio static-factor worker.

This worker creates no simulator, advances no physics, and executes no predicted
action.  Endpoint-native interface banks are captured in-memory, then every
registered consumer read is overwritten from the requested bank while the
complete decoder proprio tensor is selected independently.
"""
from __future__ import annotations
from wam_causal_audit.paths import resolve as _release_path

import argparse
import contextlib
import hashlib
import importlib.util
import json
import os
import random
import signal
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping

import numpy as np

BASE_RUN = Path(_release_path('@DATA@/wam_factor_routing_v5/experiments/robotwin_identity_amendment_coverage_20260921T103100Z'))
BASE_WORKER = BASE_RUN / "runtime/robotwin_identity_amendment_worker.py"
PHASE = "POSTHOC_INTERFACE_X_PROPRIO_FACTOR"
ACTION_SHAPE = (32, 14)


def load_base():
    spec = importlib.util.spec_from_file_location("robotwin_prior_worker", BASE_WORKER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


b = load_base()
rt = b.rt


def reset_random(seed: int) -> None:
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def sha_bytes(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()


class ProprioTrace:
    """Trace the full normalized proprio through encoder and action consumer."""

    def __init__(self, model: Any, label: str, expected: Any):
        self.model = model
        self.label = label
        self.expected = expected.detach().to(dtype=model.torch_dtype).cpu().clone()
        self.encoder = model.proprio_encoder
        self.encoder_forward = self.encoder.forward
        self.action_pre = model.action_expert.pre_dit
        self.encoder_events: list[dict[str, Any]] = []
        self.action_events: list[dict[str, Any]] = []
        self.cleanup = False

    def install(self) -> None:
        outer = self

        def encoder_forward(value, *args, **kwargs):
            observed = value.detach().cpu()
            if observed.ndim == 3 and observed.shape[1] == 1:
                observed_compare = observed[:, 0]
            else:
                observed_compare = observed
            if not __import__("torch").equal(observed_compare, outer.expected):
                raise RuntimeError("PROPRIO_ENCODER_INPUT_SOURCE_MISMATCH")
            output = outer.encoder_forward(value, *args, **kwargs)
            outer.encoder_events.append({
                "source": outer.label,
                "input_shape": list(value.shape),
                "input_dtype": str(value.dtype),
                "input_hash": b.tensor_hash(value),
                "encoded_shape": list(output.shape),
                "encoded_hash": b.tensor_hash(output),
            })
            return output

        def action_pre(*args, **kwargs):
            context = kwargs.get("context")
            mask = kwargs.get("context_mask")
            if context is None or mask is None:
                raise RuntimeError("ACTION_PRE_DIT_CONTEXT_NOT_OBSERVED")
            outer.action_events.append({
                "source": outer.label,
                "context_shape": list(context.shape),
                "context_hash": b.tensor_hash(context),
                "last_context_token_hash": b.tensor_hash(context[:, -1:]),
                "last_mask_all_true": bool(mask[:, -1:].all()),
            })
            return outer.action_pre(*args, **kwargs)

        self.encoder.forward = encoder_forward
        self.model.action_expert.pre_dit = action_pre

    def uninstall(self) -> None:
        self.encoder.forward = self.encoder_forward
        self.model.action_expert.pre_dit = self.action_pre
        self.cleanup = self.encoder.forward == self.encoder_forward and self.model.action_expert.pre_dit == self.action_pre

    def validate(self) -> None:
        if len(self.encoder_events) != 1:
            raise RuntimeError(f"PROPRIO_ENCODER_EVENT_COUNT:{len(self.encoder_events)}")
        if len(self.action_events) != 10:
            raise RuntimeError(f"PROPRIO_ACTION_CONSUMER_EVENT_COUNT:{len(self.action_events)}")
        if not all(row["last_mask_all_true"] for row in self.action_events):
            raise RuntimeError("PROPRIO_CONTEXT_MASKED")


@contextlib.contextmanager
def installed(controller: Any, proprio_trace: ProprioTrace):
    controller.install()
    proprio_trace.install()
    try:
        yield
    finally:
        proprio_trace.uninstall()
        controller.uninstall()


def load_sources(path: Path) -> list[dict[str, Any]]:
    payload = rt.read(path)
    rows = payload["sources"]
    return sorted(rows, key=lambda x: (str(x["task_id"]), str(x["source_id"])))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--kind", required=True, choices=("direct", "joint"))
    parser.add_argument("--sources", required=True, type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    registry = rt.read(run / "registry.json")
    sources = load_sources(args.sources)
    output = run / args.kind
    output.mkdir(parents=True, exist_ok=True)

    if rt.sha(Path(__file__)) != registry["worker_sha256"]:
        raise RuntimeError("WORKER_HASH_DRIFT")
    if rt.sha(args.sources) != registry["source_manifest_sha256"]:
        raise RuntimeError("SOURCE_MANIFEST_HASH_DRIFT")
    if len(sources) != 42:
        raise RuntimeError(f"SOURCE_COUNT:{len(sources)}")

    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(InterruptedError("SUPERVISOR_STOP")))
    os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = _release_path('@WORKSPACE@/FastWAM/checkpoints')
    os.environ["DIFFSYNTH_SKIP_DOWNLOAD"] = "true"
    sys.path.insert(0, _release_path('@WORKSPACE@/FastWAM/src'))
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
    from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json

    config_path = Path(registry[f"{args.kind}_config"])
    cfg = OmegaConf.create(rt.read(config_path))
    cap_mib = float(os.environ["CAMPAIGN_TASK_MEMORY_CAP_MIB"])
    total_mib = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
    torch.cuda.set_per_process_memory_fraction(min(0.95, cap_mib / total_mib), 0)
    reset_random(int(registry["model_load_seed"]))
    load_start = time.monotonic()
    model = instantiate(cfg.model, model_dtype=torch.bfloat16, device="cuda").eval()
    checkpoint = Path(cfg.ckpt)
    payload = torch.load(checkpoint, weights_only=True, map_location="cpu", mmap=True)
    model.mot.load_state_dict(payload["mot"], strict=True)
    if model.proprio_encoder is None:
        raise RuntimeError("PROPRIO_ENCODER_DISABLED")
    model.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
    del payload
    processor = instantiate(cfg.data.train.processor).eval()
    stats_path = Path(cfg.EVALUATION.dataset_stats_path)
    processor.set_normalizer_from_stats(load_dataset_stats_from_json(stats_path))
    weight_before = b.model_weight_hash(model)
    rt.append(run / "ledger.jsonl", {"event": "MODEL_LOADED", "model": args.kind, "seconds": time.monotonic()-load_start, "time": time.time()})

    call_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    max_calls = int(registry["maximum_policy_call_attempts"])

    def call(source: Mapping[str, Any], geometry: Mapping[str, Any], arm: str, image_label: str,
             proprio_label: str, controller: Any):
        call_id = f"{args.kind}__{source['source_id']}__{arm}"
        started = {"event": "CALL_STARTED", "call_id": call_id, "model": args.kind,
                   "source_id": source["source_id"], "task_id": source["task_id"], "arm": arm,
                   "interface_source": arm.split("__I",1)[1].split("__",1)[0] if "__I" in arm else image_label,
                   "proprio_source": proprio_label, "time": time.time(), "policy_call_attempts_delta": 1}
        b.reserve_policy_call(run, started, max_calls)
        rt.save(output / "active_call.json", started)
        started_mono = time.monotonic()
        status, error = "FAILED", None
        event_path = output / "events" / source["source_id"] / f"{arm}.json"
        event_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            image_obs = Path(geometry["inputs"][image_label].get("observation", geometry["root"] / image_label / "obs.pkl"))
            proprio_obs = Path(geometry["inputs"][proprio_label].get("observation", geometry["root"] / proprio_label / "obs.pkl"))
            image, image_proprio, _ = b.preprocess(image_obs, processor)
            _, selected_proprio, raw = b.preprocess(proprio_obs, processor)
            seed = int(source["policy_seed"])
            reset_random(seed)
            trace = ProprioTrace(model, proprio_label, selected_proprio)
            kwargs = {"prompt": DEFAULT_PROMPT.format(task=source["instruction"]), "input_image": image,
                      "proprio": selected_proprio, "action_horizon": 32, "num_inference_steps": 10,
                      "seed": seed, "rand_device": "cpu", "text_cfg_scale": 1.0,
                      "negative_prompt": "", "sigma_shift": None, "tiled": False}
            if args.kind == "joint":
                kwargs["num_video_frames"] = 9
            with installed(controller, trace), b.alarm_timeout(int(registry["call_timeout_seconds"])), torch.inference_mode():
                action = model.infer_action(**kwargs)["action"].detach().cpu()
                torch.cuda.synchronize()
            controller.validate(); trace.validate()
            if not controller.cleanup or not trace.cleanup:
                raise RuntimeError("HOOK_CLEANUP_FAILED")
            if tuple(action.shape) != ACTION_SHAPE or not bool(torch.isfinite(action).all()):
                raise RuntimeError("ACTION_INVALID")
            joint_targets = b.denormalize(processor, action)
            eef_world = np.asarray([b.fk_target(geometry["contract"], row) for row in joint_targets])
            radial_world = eef_world @ geometry["u"]
            context_eef = b.recipient_eef(geometry["root"], "Z1", geometry["contract"]["side"])
            artifact = output / "actions" / source["source_id"] / f"{arm}.npz"
            atomic_npz(artifact, normalized=action.float().numpy(), joint_targets=joint_targets,
                       eef_target_world=eef_world, radial_world=radial_world,
                       radial_relative_to_recipient_eef=(eef_world-context_eef[None,:]) @ geometry["u"],
                       u=geometry["u"], context_eef=context_eef,
                       raw_proprio=np.asarray(raw["joint_action"]["vector"], dtype=np.float32),
                       normalized_proprio=selected_proprio.float().numpy())
            event_payload = {"interface_events": controller.events, "proprio_encoder_events": trace.encoder_events,
                             "proprio_action_consumer_events": trace.action_events}
            rt.save(event_path, event_payload)
            row = {**started, "event": "CALL_FINISHED", "status": "PASS", "seconds": time.monotonic()-started_mono,
                   "image_observation": str(image_obs), "image_observation_sha256": rt.sha(image_obs),
                   "proprio_observation": str(proprio_obs), "proprio_observation_sha256": rt.sha(proprio_obs),
                   "selected_proprio_sha256": b.tensor_hash(selected_proprio),
                   "image_native_proprio_sha256": b.tensor_hash(image_proprio),
                   "raw_proprio_sha256": sha_bytes(np.asarray(raw["joint_action"]["vector"], dtype=np.float32)),
                   "action_artifact": str(artifact), "action_artifact_sha256": rt.sha(artifact),
                   "events": str(event_path), "events_sha256": rt.sha(event_path),
                   "interface_event_count": len(controller.events), "proprio_encoder_event_count": len(trace.encoder_events),
                   "proprio_action_consumer_event_count": len(trace.action_events),
                   "interface_hook_cleanup": controller.cleanup, "proprio_hook_cleanup": trace.cleanup,
                   "predicted_action_executed": False}
            status = "PASS"; call_rows.append(row); rt.append(run / "ledger.jsonl", row)
            return action, controller.captured if getattr(controller, "mode", None) == "capture" else None, row
        except BaseException as exc:
            error = repr(exc)
            row = {**started, "event": "CALL_FINISHED", "status": "FAILED", "error": error,
                   "seconds": time.monotonic()-started_mono, "predicted_action_executed": False}
            call_rows.append(row); rt.append(run / "ledger.jsonl", row)
            return None
        finally:
            rt.save(output / "call_summaries.json", call_rows)
            rt.save(output / "active_call.json", {"event": "IDLE"})

    for source in sources:
        summary = {"source_id": source["source_id"], "task_id": source["task_id"], "status": "STARTED"}
        try:
            geometry = b.read_geometry(source)
            if args.kind == "direct":
                zc = b.DirectController(model, "capture", bank_label="Z1")
                dc = b.DirectController(model, "capture", bank_label="Dplus")
            else:
                zc = b.JointController(model, "capture", current_label="Z1", future_label="Z1")
                dc = b.JointController(model, "capture", current_label="Dplus", future_label="Dplus")
            zcap = call(source, geometry, "capture_Z1", "Z1", "Z1", zc)
            dcap = call(source, geometry, "capture_Dplus", "Dplus", "Dplus", dc)
            if zcap is None or dcap is None:
                raise RuntimeError("CAPTURE_DEPENDENCY_FAILED")
            zb, db = zcap[1], dcap[1]
            actions = {}
            for cell, ibank, ilabel, plabel in (("Y00",zb,"Z1","Z1"),("Y10",db,"Dplus","Z1"),
                                                ("Y01",zb,"Z1","Dplus"),("Y11",db,"Dplus","Dplus")):
                if args.kind == "direct":
                    controller = b.DirectController(model, "replace", ibank, f"{ilabel}_FROZEN_INTERFACE")
                else:
                    controller = b.JointController(model, "strict", ibank, ibank,
                                                   current_label=ilabel, future_label=ilabel)
                measured = call(source, geometry, f"{cell}__I{ilabel}__P{plabel}", "Z1", plabel, controller)
                if measured is None:
                    raise RuntimeError(f"CELL_FAILED:{cell}")
                actions[cell] = measured[0]
            y00_error = float((actions["Y00"] - zcap[0]).abs().max())
            historical_root = Path(source["historical_action_root"]) / f"{args.kind}_{source['historical_phase']}"
            y10_hist = (historical_root / "actions" / source["historical_source_id"] /
                        ("interface_Dplus_on_Z1.npz" if args.kind == "direct" else "A11_Cdonor_Fdonor.npz"))
            historical_error = None
            if y10_hist.exists():
                historical_error = float(np.max(np.abs(np.load(y10_hist)["normalized"] - actions["Y10"].numpy())))
            if y00_error != 0.0:
                raise RuntimeError(f"Y00_NOT_EXACT_NATIVE:{y00_error}")
            summary.update(status="PASS", y00_max_abs_vs_capture_Z1=y00_error,
                           y10_historical_artifact=str(y10_hist) if y10_hist.exists() else None,
                           y10_historical_normalized_max_abs=historical_error)
        except BaseException as exc:
            summary.update(status="FAILED", error=repr(exc), traceback=traceback.format_exc())
        source_rows.append(summary); rt.save(output / "source_summaries.json", source_rows)
        torch.cuda.empty_cache()

    weight_after = b.model_weight_hash(model)
    rt.save(output / "weights.json", {"before": weight_before, "after": weight_after, "unchanged": weight_before == weight_after})
    if weight_before != weight_after:
        raise RuntimeError("MODEL_WEIGHT_MUTATION")
    rt.save(output / "result.json", {"status": "COMPLETE" if all(x["status"]=="PASS" for x in source_rows) else "COMPLETE_WITH_FAILURES",
                                      "model": args.kind, "sources": len(source_rows),
                                      "passed_sources": sum(x["status"]=="PASS" for x in source_rows),
                                      "policy_call_attempts": len(call_rows), "physics_steps": 0,
                                      "simulator_initializations": 0, "predicted_actions_executed": 0,
                                      "worker_sha256": rt.sha(Path(__file__)), "weights_unchanged": True})


if __name__ == "__main__":
    main()

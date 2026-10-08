#!/usr/bin/env python3
"""Joint-WAM C: online world-branch early stopping with per-layer K/V reuse."""
from __future__ import annotations
from wam_causal_audit.paths import resolve as _release_path

import argparse
import csv
import gc
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

WORK = Path(__file__).resolve().parent
sys.path.insert(0, str(WORK))
import run_experiment_a as A  # noqa: E402
from capture import make_capture  # noqa: E402


BASE = Path(_release_path('@DATA@/wam_factor_routing_v5/joint_experiment_B_distance_v1'))
BROOT = BASE / "B_forward_development_validation_v1"
SYNC = BASE / "B_SYNC_V1"
CROOT = BASE / "C_world_early_stop_v1"
PROTOCOL = "joint-C-world-early-stop-v1"
KS = (10, 8, 5)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(A.jsonable(value), indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    tmp.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader(); writer.writerows(rows)
    tmp.replace(path)


def tensor_hash(tensor: torch.Tensor) -> str:
    return A.tensor_sha256(tensor.detach())


def cache_hash(cache: list[dict[str, torch.Tensor]]) -> str:
    h = hashlib.sha256()
    for layer, values in enumerate(cache):
        h.update(str(layer).encode())
        for name in ("k", "v"):
            h.update(tensor_hash(values[name]).encode())
    return h.hexdigest()


def split_cache(cache: list[dict[str, torch.Tensor]], which: str) -> str:
    h = hashlib.sha256()
    for values in cache:
        tpg = values["k"].shape[1] // 3
        sl = slice(0, tpg) if which == "current" else slice(tpg, None)
        for name in ("k", "v"):
            h.update(tensor_hash(values[name][:, sl]).encode())
    return h.hexdigest()


def load_registry() -> tuple[list[dict], dict[str, dict]]:
    states = [r for r in csv.DictReader((BASE / "B_split_registry.csv").open()) if r["selection_status"].startswith("SELECTED")]
    state_by_id = {r["candidate_id"]: r for r in states}
    donors: list[dict] = []
    for path in sorted((SYNC / "f3g_synced_shards").glob("shard_*_of_04.json")):
        donors.extend(r for r in json.loads(path.read_text()) if r["donor_valid"])
    return states, {r["case_id"]: {**r, "state": state_by_id[r["case_id"].split("__F3G_SYNC_")[0]]} for r in donors}


def seed(state: Mapping[str, Any]) -> int:
    return 820000 + int(state["task_id"]) * 1000 + int(state["trajectory_id"]) * 20 + int(state["policy_call"])


def prepared(runner, path: Path | str, instruction: str):
    return runner._prepared(A.load_npz(Path(path)), instruction)


class SelectiveCacheController(A.VideoKVController):
    def __init__(self, model: Any, capture_steps: set[int]):
        super().__init__(model)
        self.capture_steps = capture_steps

    def set_step(self, step: int) -> None:
        self.step = step
        self.capture = step in self.capture_steps

    def cache_at(self, step: int) -> list[dict[str, torch.Tensor]]:
        return [self.cache[(step, layer)] for layer in range(len(self.model.video_expert.blocks))]


def initialize(model, policy, item: Mapping[str, Any], seed_value: int):
    image = item["image"].to(model.device, model.torch_dtype)
    _, _, height, width = image.shape
    latent_t = (policy.num_video_frames - 1) // model.vae.temporal_downsample_factor + 1
    rand_device = str(policy.cfg.EVALUATION.rand_device)
    video_generator = torch.Generator(device=rand_device).manual_seed(seed_value)
    action_generator = torch.Generator(device=rand_device).manual_seed(seed_value)
    video = torch.randn((1, model.vae.model.z_dim, latent_t, height // model.vae.upsampling_factor,
                         width // model.vae.upsampling_factor), generator=video_generator,
                        device=rand_device, dtype=torch.float32).to(model.device, model.torch_dtype)
    action = torch.randn((1, policy.action_horizon, model.action_expert.action_dim), generator=action_generator,
                         device=rand_device, dtype=torch.float32).to(model.device, model.torch_dtype)
    first = model._encode_input_image_latents_tensor(image, tiled=bool(policy.cfg.EVALUATION.tiled))
    video[:, :, 0:1] = first.clone()
    vts, vds = model.infer_video_scheduler.build_inference_schedule(policy.num_inference_steps, model.device, video.dtype, shift_override=None)
    ats, ads = model.infer_action_scheduler.build_inference_schedule(policy.num_inference_steps, model.device, action.dtype, shift_override=None)
    return image, video, action, first, vts, vds, ats, ads


@torch.no_grad()
def early_stop_infer(runner, item: Mapping[str, Any], seed_value: int, k: int,
                     supplied_stop_cache: list[dict[str, torch.Tensor]] | None = None,
                     capture_all: bool = False) -> tuple[torch.Tensor, dict, dict[int, list[dict[str, torch.Tensor]]]]:
    if k not in KS:
        raise ValueError(k)
    model, policy = runner.model, runner.policy
    torch.cuda.synchronize(model.device)
    torch.cuda.reset_peak_memory_stats(model.device)
    start = time.perf_counter()
    _, video, action, first, vts, vds, ats, ads = initialize(model, policy, item, seed_value)
    capture_steps = set(range(10)) if capture_all else {k - 1}
    ctl = SelectiveCacheController(model, capture_steps)
    ctl.install()
    last_cache = None
    mask = None
    video_seq_len = None
    tpg = None
    try:
        for step, (tv, dv, ta, da) in enumerate(zip(vts, vds, ats, ads)):
            ctl.set_step(step)
            if step < k:
                video_pre = model.video_expert.pre_dit(
                    x=video, timestep=tv.unsqueeze(0).to(video), context=item["context"],
                    context_mask=item["context_mask"], action=None,
                    fuse_vae_embedding_in_latents=bool(getattr(model.video_expert, "fuse_vae_embedding_in_latents", False)))
                action_pre = model.action_expert.pre_dit(
                    action_tokens=action, timestep=ta.unsqueeze(0).to(action),
                    context=item["context"], context_mask=item["context_mask"])
                tpg = int(video_pre["meta"]["tokens_per_frame"])
                video_seq_len = int(video_pre["tokens"].shape[1])
                mask = model._build_mot_attention_mask(video_seq_len, action_pre["tokens"].shape[1], tpg, video_pre["tokens"].device)
                out = model.mot(
                    embeds_all={"video": video_pre["tokens"], "action": action_pre["tokens"]}, attention_mask=mask,
                    freqs_all={"video": video_pre["freqs"], "action": action_pre["freqs"]},
                    context_all={"video": {"context": video_pre["context"], "mask": video_pre["context_mask"]},
                                 "action": {"context": action_pre["context"], "mask": action_pre["context_mask"]}},
                    t_mod_all={"video": video_pre["t_mod"], "action": action_pre["t_mod"]})
                pred_video = model.video_expert.post_dit(out["video"], video_pre)
                pred_action = model.action_expert.post_dit(out["action"], action_pre)
                video = model.infer_video_scheduler.step(pred_video, dv, video)
                video[:, :, 0:1] = first.clone()
                if step == k - 1:
                    last_cache = ctl.cache_at(step)
            else:
                use_cache = supplied_stop_cache if supplied_stop_cache is not None else last_cache
                if use_cache is None or mask is None or video_seq_len is None:
                    raise AssertionError("missing stop cache")
                pred_action = model._predict_action_noise_with_cache(
                    latents_action=action, timestep_action=ta.unsqueeze(0).to(action),
                    context=item["context"], context_mask=item["context_mask"],
                    video_kv_cache=use_cache, attention_mask=mask, video_seq_len=video_seq_len)
            action = model.infer_action_scheduler.step(pred_action, da, action)
    finally:
        ctl.uninstall()
    torch.cuda.synchronize(model.device)
    elapsed = time.perf_counter() - start
    schedule = {step: ctl.cache_at(step) for step in capture_steps}
    stop_cache = supplied_stop_cache if supplied_stop_cache is not None else schedule[k - 1]
    diag = {
        "K": k, "world_branch_steps": k, "action_branch_steps": 10,
        "world_layer_calls": ctl.video_calls, "action_layer_calls": ctl.action_calls,
        "expected_world_layer_calls": k * len(model.video_expert.blocks),
        "expected_action_layer_calls": 10 * len(model.action_expert.blocks),
        "stop_cache_hash": cache_hash(stop_cache),
        "cache_source_forward_step_zero_based": k - 1,
        "cache_source_forward_step_one_based": k,
        "cache_is_post_rope_k_post_projection_v": True,
        "cache_layers": len(stop_cache), "video_seq_len": video_seq_len,
        "tokens_per_temporal_group": tpg, "latency_seconds": elapsed,
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated(model.device)),
        "next_policy_call_cache_reuse": False,
    }
    return action[0].detach().cpu().float(), diag, schedule


@torch.no_grad()
def full_cache_action_replay(runner, item: Mapping[str, Any], seed_value: int,
                             schedule: dict[int, list[dict[str, torch.Tensor]]],
                             video_seq_len: int, tokens_per_group: int) -> tuple[torch.Tensor, dict]:
    model, policy = runner.model, runner.policy
    torch.cuda.synchronize(model.device); torch.cuda.reset_peak_memory_stats(model.device); start = time.perf_counter()
    _, _, action, _, _, _, ats, ads = initialize(model, policy, item, seed_value)
    action_steps = 0
    for step, (ta, da) in enumerate(zip(ats, ads)):
        action_pre = model.action_expert.pre_dit(action_tokens=action, timestep=ta.unsqueeze(0).to(action),
                                                  context=item["context"], context_mask=item["context_mask"])
        mask = model._build_mot_attention_mask(video_seq_len, action_pre["tokens"].shape[1], tokens_per_group, action_pre["tokens"].device)
        pred = model._predict_action_noise_with_cache(action, ta.unsqueeze(0).to(action), item["context"], item["context_mask"],
                                                       schedule[step], mask, video_seq_len)
        action = model.infer_action_scheduler.step(pred, da, action); action_steps += 1
    torch.cuda.synchronize(model.device)
    return action[0].detach().cpu().float(), {"world_branch_steps": 0, "action_branch_steps": action_steps,
                                               "latency_seconds": time.perf_counter() - start,
                                               "peak_memory_bytes": int(torch.cuda.max_memory_allocated(model.device))}


def standalone_copy_seconds(model, cache: list[dict[str, torch.Tensor]]) -> float:
    torch.cuda.synchronize(model.device); start = time.perf_counter()
    copied = [{"k": x["k"].clone(), "v": x["v"].clone()} for x in cache]
    torch.cuda.synchronize(model.device); elapsed = time.perf_counter() - start
    del copied
    return elapsed


def native_timed(runner, item, seed_value):
    torch.cuda.reset_peak_memory_stats(runner.model.device)
    action, run = A.infer(runner, item, item, seed_value, None)
    return action, {"latency_seconds": run["model_inference_seconds"],
                    "peak_memory_bytes": int(torch.cuda.max_memory_allocated(runner.model.device)),
                    "world_branch_steps": 10, "action_branch_steps": 10}


def technical_inputs(states: list[dict], donors: dict[str, dict]) -> list[dict]:
    frozen = json.loads((BROOT / "smoke_selection.json").read_text())
    state_by_id = {x["candidate_id"]: x for x in states}
    inputs = []
    for row in frozen:
        state = state_by_id[row["state_id"]]; donor = donors[row["case_id"]]
        inputs.append({"input_id": "recipient::" + row["state_id"], "role": "recipient", "state": state,
                       "path": str(SYNC / "synced_recipients" / row["state_id"] / "recipient_policy_observation.npz")})
        inputs.append({"input_id": "donor::" + row["case_id"], "role": "donor", "state": state,
                       "path": donor["donor_observation_path"]})
    return inputs


def formal_inputs(states: list[dict], donors: dict[str, dict]) -> list[dict]:
    items = []
    for state in states:
        cid = state["candidate_id"]
        items.append({"input_id": "recipient::" + cid, "role": "recipient", "state": state,
                      "path": str(SYNC / "synced_recipients" / cid / "recipient_policy_observation.npz"),
                      "b_action": str(BROOT / "states" / cid / "recipient_actions.npz"), "b_key": "natural"})
    for case_id, donor in donors.items():
        state = donor["state"]; cid = state["candidate_id"]
        items.append({"input_id": "donor::" + case_id, "role": "donor", "state": state,
                      "path": donor["donor_observation_path"],
                      "b_action": str(BROOT / "states" / cid / "donors" / case_id / "actions.npz"), "b_key": "donor_natural"})
    return items


def technical_one(runner, record: dict) -> tuple[list[dict], list[dict]]:
    state = record["state"]; s = seed(state); item = prepared(runner, record["path"], state["instruction"])
    before = A.model_weight_hash(runner.model)
    native1, native_diag = native_timed(runner, item, s)
    native2, _ = native_timed(runner, item, s)
    k10, k10diag, schedule = early_stop_infer(runner, item, s, 10, capture_all=True)
    replay, replaydiag = full_cache_action_replay(runner, item, s, schedule, k10diag["video_seq_len"], k10diag["tokens_per_temporal_group"])
    rows = []
    def check(name, a, b):
        rows.append({"input_id": record["input_id"], "role": record["role"], "task_id": state["task_id"],
                     "trajectory_id": state["trajectory_id"], "check": name, "bit_exact": torch.equal(a, b),
                     "max_abs_error": A.max_abs(a, b), "pass": torch.equal(a, b)})
    check("native_repeat", native1, native2)
    check("K10_new_path_vs_native", native1, k10)
    check("full_step_cache_action_only_replay", native1, replay)
    latency = [{"input_id": record["input_id"], "role": record["role"], "configuration": "native_full", **native_diag},
               {"input_id": record["input_id"], "role": record["role"], "configuration": "K10_new_path", **k10diag},
               {"input_id": record["input_id"], "role": record["role"], "configuration": "full_cache_technical_replay", **replaydiag}]
    for k in (8, 5):
        action, diag, own = early_stop_infer(runner, item, s, k)
        replay_action, replay_diag, replay_schedule = early_stop_infer(runner, item, s, k, supplied_stop_cache=own[k - 1])
        check(f"K{k}_online_vs_own_cache_replay", action, replay_action)
        rows.append({"input_id": record["input_id"], "role": record["role"], "task_id": state["task_id"],
                     "trajectory_id": state["trajectory_id"], "check": f"K{k}_online_replay_cache_hash",
                     "bit_exact": diag["stop_cache_hash"] == replay_diag["stop_cache_hash"], "max_abs_error": 0.0,
                     "pass": diag["stop_cache_hash"] == replay_diag["stop_cache_hash"]})
        rows.append({"input_id": record["input_id"], "role": record["role"], "task_id": state["task_id"],
                     "trajectory_id": state["trajectory_id"], "check": f"K{k}_execution_counts",
                     "bit_exact": True, "max_abs_error": 0.0,
                     "pass": diag["world_branch_steps"] == k and diag["action_branch_steps"] == 10 and
                             diag["world_layer_calls"] == k * 30 and diag["action_layer_calls"] == 300})
        latency.append({"input_id": record["input_id"], "role": record["role"], "configuration": f"K{k}", **diag,
                        "standalone_cache_copy_seconds": standalone_copy_seconds(runner.model, own[k - 1])})
        del own, replay_schedule, action, replay_action
    current_hashes = [split_cache(schedule[x], "current") for x in range(10)]
    future_hashes = [split_cache(schedule[x], "future") for x in range(10)]
    rows.append({"input_id": record["input_id"], "role": record["role"], "task_id": state["task_id"],
                 "trajectory_id": state["trajectory_id"], "check": "current_cache_step_variation_recorded",
                 "bit_exact": len(set(current_hashes)) == 1, "max_abs_error": np.nan, "pass": True,
                 "unique_step_hashes": len(set(current_hashes)), "interpretation": "diagnostic_not_gate"})
    rows.append({"input_id": record["input_id"], "role": record["role"], "task_id": state["task_id"],
                 "trajectory_id": state["trajectory_id"], "check": "future_cache_step_variation_recorded",
                 "bit_exact": len(set(future_hashes)) == 1, "max_abs_error": np.nan, "pass": True,
                 "unique_step_hashes": len(set(future_hashes)), "interpretation": "diagnostic_not_gate"})
    rows.append({"input_id": record["input_id"], "role": record["role"], "task_id": state["task_id"],
                 "trajectory_id": state["trajectory_id"], "check": "weight_hash_unchanged",
                 "bit_exact": before == A.model_weight_hash(runner.model), "max_abs_error": 0.0,
                 "pass": before == A.model_weight_hash(runner.model)})
    del schedule, item; gc.collect(); torch.cuda.empty_cache()
    return rows, latency


def formal_one(runner, record: dict, ordinal: int) -> tuple[dict, list[dict]]:
    state = record["state"]; s = seed(state); item = prepared(runner, record["path"], state["instruction"])
    b = np.load(record["b_action"])[record["b_key"]].astype(np.float32)
    order = (8, 5) if ordinal % 2 == 0 else (5, 8)
    values = {"K10_normalized": b}; latencies = []
    diags = {}
    for k in order:
        action, diag, _ = early_stop_infer(runner, item, s, k)
        values[f"K{k}_normalized"] = action.numpy(); diags[k] = diag
        latencies.append({"input_id": record["input_id"], "role": record["role"], "task_id": state["task_id"],
                          "trajectory_id": state["trajectory_id"], "policy_call": state["policy_call"],
                          "split": state["split"], "configuration": f"K{k}", **diag})
    out = CROOT / "inputs" / record["input_id"].replace("::", "__")
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "actions.npz", **values)
    meta = {"input_id": record["input_id"], "role": record["role"], "task_id": int(state["task_id"]),
            "trajectory_id": int(state["trajectory_id"]), "policy_call": int(float(state["policy_call"])),
            "split": state["split"], "path": record["path"], "seed": s,
            "K10_source": "reused frozen B natural action after technical K10 exact gate",
            "K8_stop_cache_hash": diags[8]["stop_cache_hash"], "K5_stop_cache_hash": diags[5]["stop_cache_hash"],
            "environment_action_executed": False, "completed": True}
    atomic_json(out / "manifest.json", meta)
    del item; gc.collect(); torch.cuda.empty_cache()
    return meta, latencies


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["technical", "formal"], required=True)
    parser.add_argument("--gpu", type=int, required=True); parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--shards", type=int, default=4); parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args(); CROOT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    states, donors = load_registry(); runner = make_capture("joint", args.gpu)
    weight_before = A.model_weight_hash(runner.model)
    if args.mode == "technical":
        items = technical_inputs(states, donors)
        own = [x for i, x in enumerate(items) if i % args.shards == args.shard]
        checks, latency = [], []
        for i, item in enumerate(own):
            c, l = technical_one(runner, item); checks.extend(c); latency.extend(l)
            print(json.dumps({"mode": "technical", "shard": args.shard, "done": i + 1, "total": len(own), "input": item["input_id"]}), flush=True)
        write_csv(CROOT / "technical_shards" / f"shard_{args.shard:02d}_checks.csv", checks)
        write_csv(CROOT / "technical_shards" / f"shard_{args.shard:02d}_latency.csv", latency)
        atomic_json(CROOT / "technical_shards" / f"shard_{args.shard:02d}.json", {
            "protocol": PROTOCOL, "shard": args.shard, "gpu_visible_index": args.gpu, "inputs": len(own),
            "pass": all(bool(x["pass"]) for x in checks), "weight_hash_before": weight_before,
            "weight_hash_after": A.model_weight_hash(runner.model), "metadata": A.run_metadata(runner, args.gpu)})
    else:
        gate = json.loads((CROOT / "C_technical_gate.json").read_text())
        if not gate["pass"]:
            raise RuntimeError("C technical gate did not pass")
        items = formal_inputs(states, donors)
        own = [x for i, x in enumerate(items) if i % args.shards == args.shard]
        manifests, latency = [], []
        for i, item in enumerate(own):
            m, l = formal_one(runner, item, i); manifests.append(m); latency.extend(l)
            if (i + 1) % 10 == 0 or i + 1 == len(own):
                print(json.dumps({"mode": "formal", "shard": args.shard, "done": i + 1, "total": len(own)}), flush=True)
        write_csv(CROOT / "formal_shards" / f"shard_{args.shard:02d}_inputs.csv", manifests)
        write_csv(CROOT / "formal_shards" / f"shard_{args.shard:02d}_latency.csv", latency)
        atomic_json(CROOT / "formal_shards" / f"shard_{args.shard:02d}.json", {
            "protocol": PROTOCOL, "shard": args.shard, "inputs": len(own), "K10_reused": True,
            "weight_hash_before": weight_before, "weight_hash_after": A.model_weight_hash(runner.model),
            "no_final_test": True, "no_environment_action": True, "metadata": A.run_metadata(runner, args.gpu)})


if __name__ == "__main__":
    main()

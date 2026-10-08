'''Independent budget-parametric copy of the registered K5 strict consumer path.'''
from __future__ import annotations
import time
from typing import Any, Mapping
import torch
from run_joint_compute_mechanism_posthoc import A, C, schedule_mapping, schedule_hash, mix_layer
K = 8

@torch.no_grad()
def capture_k5_source(runner: Any, item: Mapping[str, Any], seed: int) -> tuple[torch.Tensor, dict[int, list[dict[str, torch.Tensor]]], dict[str, Any]]:
    model, policy = runner.model, runner.policy
    torch.cuda.synchronize(model.device); torch.cuda.reset_peak_memory_stats(model.device); started = time.perf_counter()
    _, video, action, first, vts, vds, ats, ads = C.initialize(model, policy, item, seed)
    controller = C.SelectiveCacheController(model, set(range(K))); controller.install()
    mask = None; video_seq_len = None; tokens_per_group = None; action_steps = 0
    try:
        for step, (tv, dv, ta, da) in enumerate(zip(vts, vds, ats, ads)):
            controller.set_step(step)
            if step < K:
                video_pre = model.video_expert.pre_dit(
                    x=video, timestep=tv.unsqueeze(0).to(video), context=item["context"],
                    context_mask=item["context_mask"], action=None,
                    fuse_vae_embedding_in_latents=bool(getattr(model.video_expert, "fuse_vae_embedding_in_latents", False)))
                action_pre = model.action_expert.pre_dit(
                    action_tokens=action, timestep=ta.unsqueeze(0).to(action),
                    context=item["context"], context_mask=item["context_mask"])
                tokens_per_group = int(video_pre["meta"]["tokens_per_frame"])
                video_seq_len = int(video_pre["tokens"].shape[1])
                mask = model._build_mot_attention_mask(video_seq_len, action_pre["tokens"].shape[1],
                                                       tokens_per_group, video_pre["tokens"].device)
                output = model.mot(
                    embeds_all={"video": video_pre["tokens"], "action": action_pre["tokens"]}, attention_mask=mask,
                    freqs_all={"video": video_pre["freqs"], "action": action_pre["freqs"]},
                    context_all={"video": {"context": video_pre["context"], "mask": video_pre["context_mask"]},
                                 "action": {"context": action_pre["context"], "mask": action_pre["context_mask"]}},
                    t_mod_all={"video": video_pre["t_mod"], "action": action_pre["t_mod"]})
                pred_video = model.video_expert.post_dit(output["video"], video_pre)
                pred_action = model.action_expert.post_dit(output["action"], action_pre)
                video = model.infer_video_scheduler.step(pred_video, dv, video); video[:, :, 0:1] = first.clone()
            else:
                pred_action = model._predict_action_noise_with_cache(
                    latents_action=action, timestep_action=ta.unsqueeze(0).to(action),
                    context=item["context"], context_mask=item["context_mask"],
                    video_kv_cache=[controller.cache[(K - 1, layer)] for layer in range(30)],
                    attention_mask=mask, video_seq_len=video_seq_len)
            action = model.infer_action_scheduler.step(pred_action, da, action); action_steps += 1
    finally:
        controller.uninstall()
    torch.cuda.synchronize(model.device)
    schedule = {step: [controller.cache[(step, layer)] for layer in range(30)] for step in range(K)}
    diag = {
        "latency_seconds": time.perf_counter() - started, "world_steps": K, "action_steps": action_steps,
        "world_layer_calls": controller.video_calls, "joint_action_layer_calls": controller.action_calls,
        "schedule_hash": schedule_hash(schedule), "cache_layers": 30,
        "tokens_per_group": tokens_per_group, "video_seq_len": video_seq_len,
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated(model.device)),
        "cache_scope": f"this policy call only; steps 0-{K-1} generated online; steps {K}-9 consume step-{K-1} cache",
    }
    return action[0].detach().cpu().float(), schedule, diag


@torch.no_grad()
def strict_k5(runner: Any, recipient: Mapping[str, Any], seed: int,
              current_schedule: Mapping[int, list[dict[str, torch.Tensor]]],
              future_schedule: Mapping[int, list[dict[str, torch.Tensor]]]) -> tuple[torch.Tensor, dict[str, Any]]:
    model, policy = runner.model, runner.policy
    torch.cuda.synchronize(model.device); torch.cuda.reset_peak_memory_stats(model.device); started = time.perf_counter()
    _, video, action, first, vts, vds, ats, ads = C.initialize(model, policy, recipient, seed)
    controller = A.VideoKVController(model)
    controller.current_cache = schedule_mapping(current_schedule)
    controller.future_cache = schedule_mapping(future_schedule)
    controller.install()
    mask = None; video_seq_len = None; tokens_per_group = None; action_steps = 0
    tail_current_exact = 0; tail_future_exact = 0; tail_max_error = 0.0
    mixed_stop = [mix_layer(current_schedule[K - 1][layer], future_schedule[K - 1][layer]) for layer in range(30)]
    try:
        for step, (tv, dv, ta, da) in enumerate(zip(vts, vds, ats, ads)):
            if step < K:
                controller.step = step
                video_pre = model.video_expert.pre_dit(
                    x=video, timestep=tv.unsqueeze(0).to(video), context=recipient["context"],
                    context_mask=recipient["context_mask"], action=None,
                    fuse_vae_embedding_in_latents=bool(getattr(model.video_expert, "fuse_vae_embedding_in_latents", False)))
                action_pre = model.action_expert.pre_dit(
                    action_tokens=action, timestep=ta.unsqueeze(0).to(action),
                    context=recipient["context"], context_mask=recipient["context_mask"])
                tokens_per_group = int(video_pre["meta"]["tokens_per_frame"])
                video_seq_len = int(video_pre["tokens"].shape[1])
                mask = model._build_mot_attention_mask(video_seq_len, action_pre["tokens"].shape[1],
                                                       tokens_per_group, video_pre["tokens"].device)
                output = model.mot(
                    embeds_all={"video": video_pre["tokens"], "action": action_pre["tokens"]}, attention_mask=mask,
                    freqs_all={"video": video_pre["freqs"], "action": action_pre["freqs"]},
                    context_all={"video": {"context": video_pre["context"], "mask": video_pre["context_mask"]},
                                 "action": {"context": action_pre["context"], "mask": action_pre["context_mask"]}},
                    t_mod_all={"video": video_pre["t_mod"], "action": action_pre["t_mod"]})
                pred_video = model.video_expert.post_dit(output["video"], video_pre)
                pred_action = model.action_expert.post_dit(output["action"], action_pre)
                video = model.infer_video_scheduler.step(pred_video, dv, video); video[:, :, 0:1] = first.clone()
            else:
                # The original K=5 path no longer invokes the world blocks here.
                # Verify every layer of the explicit mixed stop cache before the
                # action-only consumer uses it at each of the five tail steps.
                for layer, mixed in enumerate(mixed_stop):
                    tokens = mixed["k"].shape[1] // 3
                    for component in ("k", "v"):
                        current = current_schedule[K - 1][layer][component][:, :tokens].to(mixed[component])
                        future = future_schedule[K - 1][layer][component][:, tokens:].to(mixed[component])
                        ce = float(torch.max(torch.abs(mixed[component][:, :tokens] - current)).item())
                        fe = float(torch.max(torch.abs(mixed[component][:, tokens:] - future)).item())
                        tail_max_error = max(tail_max_error, ce, fe)
                        tail_current_exact += int(torch.equal(mixed[component][:, :tokens], current))
                        tail_future_exact += int(torch.equal(mixed[component][:, tokens:], future))
                pred_action = model._predict_action_noise_with_cache(
                    latents_action=action, timestep_action=ta.unsqueeze(0).to(action),
                    context=recipient["context"], context_mask=recipient["context_mask"],
                    video_kv_cache=mixed_stop, attention_mask=mask, video_seq_len=video_seq_len)
            action = model.infer_action_scheduler.step(pred_action, da, action); action_steps += 1
    finally:
        controller.uninstall()
    torch.cuda.synchronize(model.device)
    hook = controller.summary(K, 30)
    expected_tail_components = (10 - K) * 30 * 2
    exact = all((hook["hook_reached_all_video_sites"], controller.action_calls == 10 * 30,
                 hook["strict_injection_reached_all_video_sites"], hook["all_current_consumed_values_exact"],
                 hook["all_future_consumed_values_exact"], tail_current_exact == expected_tail_components,
                 tail_future_exact == expected_tail_components, tail_max_error == 0.0))
    diag = {
        "latency_seconds": time.perf_counter() - started, "world_steps": K, "action_steps": action_steps,
        "online_world_hook": hook, "tail_action_steps": 10 - K,
        "tail_expected_component_checks": expected_tail_components,
        "tail_current_exact_count": tail_current_exact, "tail_future_exact_count": tail_future_exact,
        "tail_max_abs_error": tail_max_error, "all_consumed_sources_exact": exact,
        "action_calls_total": controller.action_calls,
        "summary_action_site_flag_not_used": "generic summary couples action expectation to world steps; K5 action remains 10x30",
        "mixed_stop_cache_hash": C.cache_hash(mixed_stop),
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated(model.device)),
    }
    if not exact or action_steps != 10 or controller.video_calls != 30 * K or controller.action_calls != 300:
        raise AssertionError(f"BUDGET_STRICT_CONSUMER_GATE_FAILED:{diag}")
    return action[0].detach().cpu().float(), diag

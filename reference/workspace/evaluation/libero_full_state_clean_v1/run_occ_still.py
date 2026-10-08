#!/usr/bin/env python3
"""New-schema stationary-world background-fill occ_still experiment.

The trigger is read from the collision-conditioned DDI direction screen, so
each model/task/seed starts masking at precisely the paired DDI t_inject step.
Only the policy observation is changed; the simulator state and target object
remain untouched and every environment step is written with full MuJoCo state.

``--trigger-mode online_threshold`` checks the *actual rollout* after every
environment action and triggers at the first EEF--target 3-D distance below
the requested threshold.  In the corrected occ_still protocol, the target
free joint is re-committed only inside the occlusion window and is released at
``t1 + 1``.  It must not be confused with the older ``clean_fixed`` convenience
mode, which selects a step from a saved clean trace and is useful only for DDI
phase matching.
"""
from __future__ import annotations
from wam_causal_audit.paths import resolve as _release_path

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
for path in (ROOT, Path(_release_path('@WORKSPACE@/LIBERO')), Path(_release_path('@WORKSPACE@/evaluation/d2_ghost_pilot'))):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from full_state_writer import FullStateRolloutWriter
from policy_adapters import LingBotAdapter, PolicyAdapter, PolicyOutput, make_adapter
from run_clean import ENVIRONMENT_SEED, _env_step, _lingbot_keyframe
from task_specs import TASK_SPECS
from perm_occlusion_scheduler import OcclusionMaskScheduler
from ddi_experiment import _choose_trigger
from translation_injection_scheduler import TranslationInjectionScheduler
from multisample_sidecar import MultiSampleCallScheduler

CLEAN_ROOT = Path(_release_path('@WORKSPACE@/results/libero_full_state_clean_v1'))
SCREEN_ROOT = Path(_release_path('@WORKSPACE@/results/libero_ddi_cc_v1/screen'))
DEFAULT_OUT = Path(_release_path('@WORKSPACE@/results/libero_full_state_occ_still_v1'))
TASKS = (0, 2, 3, 5, 9)
# Exact old-protocol wall-clock spans requested by the user: nominally three
# observations at each model's native replan cadence.
OCCLUSION_STEPS = {'pi05': 15, 'fastwam': 30, 'lingbot_va': 48, 'vla_jepa': 21}


class FixedOcclusionAnchor:
    """Minimal scheduler interface consumed by OcclusionMaskScheduler."""
    def __init__(self, t0_env_step: int) -> None:
        self.t0_env_step = int(t0_env_step)


def resolve_target_free_joint_name(env: Any, target_entity: str, target_body_name: str) -> str:
    """Find the target's free joint without hard-coding five task-specific names."""
    sim = getattr(getattr(env, 'env', env), 'sim', getattr(env, 'sim', None))
    model = sim.model
    body_names = {
        int(idx): str(model.body_id2name(idx) or '')
        for idx in range(int(model.nbody))
    }
    target_body_id = next((idx for idx, name in body_names.items() if name == target_body_name), None)
    candidates: list[tuple[int, str]] = []
    for joint_id in range(int(model.njnt)):
        name = str(model.joint_id2name(joint_id) or '')
        body_id = int(model.jnt_bodyid[joint_id])
        is_free = int(model.jnt_type[joint_id]) == 0  # mujoco.mjtJoint.mjJNT_FREE
        if not is_free:
            continue
        exact_body = target_body_id is not None and body_id == target_body_id
        related = target_entity.lower() in f'{name} {body_names.get(body_id, "")}'.lower()
        if exact_body or related:
            candidates.append((0 if exact_body else 1, name))
    if not candidates:
        raise KeyError(f'cannot resolve free joint for {target_entity!r} / {target_body_name!r}')
    return sorted(candidates)[0][1]


def apply_legacy_primary_rng(adapter: PolicyAdapter, decision_reason: str | None) -> str | None:
    """Reproduce the old executed sample (seed 0) at selected policy calls.

    Legacy π0.5 and FastWAM occ runners evaluated N=8 samples but executed
    sample zero, restoring the server/RNG state to just after that primary
    call.  Resetting immediately before a single primary inference gives the
    identical executed sample and the same post-primary RNG state without
    performing seven expensive, non-executed forwards.  VLA-JEPA has no old
    occ runner; it retains its episode-reset deployment RNG protocol.
    """
    if decision_reason is None:
        return None
    if adapter.model_name == 'pi05':
        adapter.client.infer({'__temporal_control__': {'operation': 'reset', 'seed': 0}})
        return 'server_rng_reset_seed0_primary_sample'
    if adapter.model_name == 'fastwam':
        adapter.torch.manual_seed(0)
        if adapter.torch.cuda.is_available():
            adapter.torch.cuda.manual_seed_all(0)
        return 'torch_rng_reset_seed0_primary_sample'
    return 'episode_rng_only_no_historical_vla_occ_runner'


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    os.replace(temporary, path)


def seed_pool(model: str, task_id: int) -> list[int]:
    summary = json.loads((CLEAN_ROOT/'clean_summary.json').read_text())
    pool = [int(value) for value in summary['combinations'][f'{model}/task{task_id:02d}']['success_seed_pool']]
    if len(pool) != 10:
        raise RuntimeError(f'expected ten clean-success seeds for {model}/task{task_id:02d}, got {pool}')
    return pool


def trigger(model: str, task_id: int, seed: int, *, target_distance_cm: float | None, band_halfwidth_cm: float) -> dict[str, Any]:
    """Return either the historical DDI-aligned trigger or a clean-distance trigger.

    ``target_distance_cm`` is intentionally measured in the same EEF-to-target
    three-dimensional metric used for the DDI screen.  This makes a 25-cm
    occlusion a separately labelled protocol, rather than silently changing
    the existing 8--15-cm DDI-aligned condition.
    """
    if target_distance_cm is not None:
        clean_dir = CLEAN_ROOT/'clean'/model/f'task{task_id:02d}'/f'init_{seed:03d}'
        steps = pd.read_parquet(clean_dir/'steps.parquet')
        target_m = float(target_distance_cm) / 100.0
        halfwidth_m = float(band_halfwidth_cm) / 100.0
        _, result = _choose_trigger(
            steps, task_id,
            distance_band_m=(target_m-halfwidth_m, target_m+halfwidth_m),
            target_distance_m=target_m,
        )
        result['trigger_source'] = 'clean_approach_eef_target_distance'
        result['requested_distance_cm'] = float(target_distance_cm)
        result['requested_band_halfwidth_cm'] = float(band_halfwidth_cm)
        result['source_clean_rollout'] = str(clean_dir)
        result['selection'] = (
            f"approach-phase env step closest to {target_distance_cm:.1f}cm "
            f"within [{target_distance_cm-band_halfwidth_cm:.1f},{target_distance_cm+band_halfwidth_cm:.1f}]cm"
            if not result['distance_band_exception'] else
            f"nearest available approach env step to {target_distance_cm:.1f}cm; requested band unavailable"
        )
        return result
    path = SCREEN_ROOT/model/f'task{task_id:02d}'/f'init_{seed:03d}'/'direction_screen.json'
    payload = json.loads(path.read_text())
    if bool(payload.get('infeasible')):
        raise RuntimeError(f'DDI trigger is infeasible: {path}')
    result = dict(payload['trigger'])
    result['source_screen_path'] = str(path)
    result['trigger_source'] = 'ddi_direction_screen'
    return result


def resolve_target_body_name(env: Any, target_entity: str) -> str:
    """Resolve the concrete MuJoCo body name used by segmentation fallback.

    LIBERO task specs name entities (for example ``black_book_1``), whereas
    their free-joint body is normally suffixed ``_main``.  Segmentation can
    match either spelling, but the projection fallback needs an exact body.
    """
    sim = getattr(getattr(env, 'env', env), 'sim', getattr(env, 'sim', None))
    model = sim.model
    names: list[str] = []
    for body_id in range(int(model.nbody)):
        name = model.body_id2name(body_id)
        if name:
            names.append(str(name))
    for candidate in (f'{target_entity}_main', str(target_entity)):
        if candidate in names:
            return candidate
    related = [name for name in names if str(target_entity).lower() in name.lower()]
    if related:
        return related[0]
    raise KeyError(f'cannot resolve MuJoCo body for target entity {target_entity!r}')


def run_episode(*, adapter: PolicyAdapter, suite: Any, task_id: int, seed: int, out_root: Path, physical_gpu: int, overwrite: bool,
                trigger_mode: str, target_distance_cm: float | None, band_halfwidth_cm: float,
                legacy_primary_rng: bool, target_hold_scope: str) -> dict[str, Any]:
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    task = suite.get_task(task_id)
    spec = TASK_SPECS[task_id]
    online_threshold = trigger_mode == 'online_threshold'
    if online_threshold and target_distance_cm is None:
        raise ValueError('online_threshold requires --trigger-distance-cm')
    event = None if online_threshold else trigger(
        adapter.model_name, task_id, seed,
        target_distance_cm=target_distance_cm, band_halfwidth_cm=band_halfwidth_cm,
    )
    t0 = None if event is None else int(event['env_step'])
    span_steps = int(OCCLUSION_STEPS[adapter.model_name])
    t1 = None if t0 is None else t0 + span_steps - 1
    bddl = Path(get_libero_path('bddl_files')) / task.problem_folder / task.bddl_file
    rollout_dir = out_root/'occ_still'/adapter.model_name/f'task{task_id:02d}'/f'init_{seed:03d}'
    result_path = rollout_dir/'result.json'
    if result_path.exists() and not overwrite:
        return json.loads(result_path.read_text())
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=adapter.render_resolution, camera_widths=adapter.render_resolution)
    env.seed(ENVIRONMENT_SEED)
    writer = FullStateRolloutWriter(
        rollout_dir, model=adapter.model_name, task_id=task_id, init_state_index=seed,
        task_description=task.language, task_spec=spec, bddl_file=bddl, environment_seed=ENVIRONMENT_SEED,
        physical_gpu=physical_gpu, image_width=adapter.render_resolution, image_height=adapter.render_resolution,
        condition=(
            'occ_still_online_threshold_' + f'{target_distance_cm:g}cm'
            if online_threshold else
            ('occ_still' if target_distance_cm is None else f'occ_still_{target_distance_cm:g}cm')
        ),
    )
    writer.prepare(overwrite=overwrite or (rollout_dir/'steps.parquet').exists())
    target_body_name = resolve_target_body_name(env, spec.manipulated_objects[0])
    target_joint_name = None
    if online_threshold:
        target_joint_name = resolve_target_free_joint_name(env, spec.manipulated_objects[0], target_body_name)
        anchor: Any = TranslationInjectionScheduler(
            target_object_name=spec.manipulated_objects[0], target_body_name=target_body_name,
            target_joint_name=target_joint_name, trigger_radius_m=float(target_distance_cm) / 100.0,
            delta_xy_m=(0.0, 0.0), n_slide_steps=1,
            direction_rule='stationary_target_trigger_only', axis='PERM', trigger_only=True,
        )
    else:
        anchor = FixedOcclusionAnchor(int(t0))
    masker = OcclusionMaskScheduler(
        translation_scheduler=anchor, target_object_name=spec.manipulated_objects[0],
        target_body_name=target_body_name, occlusion_steps=span_steps,
        sanity_dir=rollout_dir/'occlusion_sanity', sanity_max_frames=8,
    )
    success, termination, error = False, 'not_started', None
    calls, policy_steps = 0, 0
    legacy_selector = MultiSampleCallScheduler(clean_calls=3, post_calls=5) if (online_threshold and legacy_primary_rng) else None
    started = time.time()
    try:
        env.reset(); raw_obs = dict(env.set_init_state(suite.get_task_init_states(task_id)[seed]))
        writer.record_state(env_step=_env_step(env), env=env, raw_obs=raw_obs, action_from_previous_step=None, reward=None, done=False, phase='pre_occlusion')
        for _ in range(adapter.num_steps_wait):
            raw_obs, reward, done, _ = env.step(list(adapter.dummy_action)); raw_obs = dict(raw_obs)
            writer.record_state(env_step=_env_step(env), env=env, raw_obs=raw_obs, action_from_previous_step=adapter.dummy_action, reward=reward, done=bool(done), phase='pre_occlusion')
            if done:
                success, termination = True, 'success_during_settle'; break
        if not success:
            adapter.reset(task.language, {
                'experiment':'libero_full_state_occ_still_v1',
                'condition': writer.condition,
                'task_id':task_id,'seed':seed,'trigger':event,
            })
        while not success and policy_steps < adapter.max_policy_steps:
            call_step = _env_step(env)
            policy_obs = masker.apply_to_raw_obs(env, raw_obs, env_step=call_step, policy_call_idx=calls, role='policy_observation')
            decision_reason = None
            if legacy_selector is not None:
                decision = legacy_selector.decide(
                    occlusion_active=masker.is_active_step(call_step),
                    occlusion_has_triggered=anchor.t0_env_step is not None,
                )
                if decision.selected:
                    decision_reason = decision.reason
                    apply_legacy_primary_rng(adapter, decision_reason)
            started_infer = time.perf_counter()
            output = adapter.infer(policy_obs, call_idx=calls, env_step=call_step, first_call=(calls == 0))
            latency_ms = (time.perf_counter()-started_infer)*1000
            writer.record_policy_call(
                call_idx=calls, env_step=call_step, env=env, raw_obs=policy_obs, action_chunk=output.action_chunk,
                infer_latency_ms=latency_ms, model_input_metadata=output.model_input_metadata,
                executed_action_indices=output.executed_action_indices, latents_video=output.latents_video,
                latent_alignment=output.latent_alignment,
                extra={'occ_still_active': masker.is_active_step(call_step),'occlusion_span':masker.occlusion_span,
                       'trigger':event,'trigger_mode':trigger_mode,
                       'legacy_primary_rng_selection':decision_reason,
                       'observation_layer_only':True,
                       'target_hold_scope':target_hold_scope if online_threshold else 'none',
                       'target_hold_active':bool(online_threshold and anchor.t0_env_step is not None and masker.is_active_step(call_step))},
            )
            keyframes: list[Mapping[str,np.ndarray]] = []
            complete = True
            for offset, action in enumerate(output.executable_actions):
                if policy_steps >= adapter.max_policy_steps:
                    complete=False; break
                raw_obs, reward, done, _ = env.step(np.asarray(action,dtype=np.float32).reshape(-1)[:7].tolist()); raw_obs=dict(raw_obs)
                policy_steps += 1; step = _env_step(env)
                should_update_online_anchor = (
                    online_threshold and (
                        anchor.t0_env_step is None
                        or target_hold_scope == 'after_trigger_forever'
                        or step <= int(anchor.t0_env_step) + span_steps - 1
                    )
                )
                if should_update_online_anchor:
                    anchor.maybe_apply_after_step(env, raw_obs, low_level_step=step)
                    if anchor.invalid:
                        termination = f'trigger_invalid:{anchor.invalid_reason}'
                        complete = False
                current_t0 = anchor.t0_env_step
                current_t1 = None if current_t0 is None else int(current_t0) + span_steps - 1
                phase = 'occluded' if masker.is_active_step(step) else (
                    'post_occlusion' if current_t1 is not None and step > current_t1 else 'pre_occlusion'
                )
                writer.record_state(env_step=step, env=env, raw_obs=raw_obs, action_from_previous_step=action, reward=reward, done=bool(done), phase=phase)
                if isinstance(adapter, LingBotAdapter):
                    frame_obs = masker.apply_to_raw_obs(env, raw_obs, env_step=step, policy_call_idx=calls, role='lingbot_cache_keyframe')
                    keyframe = _lingbot_keyframe(output, offset, frame_obs)
                    if keyframe is not None: keyframes.append(keyframe)
                if done:
                    success, termination, complete = True, 'success', False; break
                if online_threshold and anchor.invalid:
                    break
            if complete:
                adapter.after_chunk(output, keyframes, done=False)
            calls += 1
        if not success and termination == 'not_started':
            termination = 'no_trigger' if online_threshold and anchor.t0_env_step is None else 'timeout'
    except Exception as exc:
        termination=f'exception:{type(exc).__name__}'
        error=''.join(traceback.format_exception(type(exc),exc,exc.__traceback__))
        print(error, file=sys.stderr, flush=True)
    finally:
        if online_threshold:
            record = anchor.trigger_record
            event = {
                'trigger_source': 'online_actual_rollout_eef_target_distance',
                'requested_distance_cm': float(target_distance_cm),
                'target_joint_name': target_joint_name,
                'target_body_name': target_body_name,
                'target_hold_scope': target_hold_scope,
                'stationary_target_hold_window_only': target_hold_scope == 'occlusion_window',
                'state': anchor.state,
                'invalid_reason': anchor.invalid_reason,
                'env_step': None if record is None else int(record.low_level_step),
                'distance_m': None if record is None else float(record.distance_m),
                'eef_pos': None if record is None else list(record.eef_pos),
                'target_center': None if record is None else list(record.target_center),
                'target_displacement_m': None if record is None else float(record.target_displacement_m),
                'target_gripper_contact': None if record is None else bool(record.target_gripper_contact),
                'legacy_primary_rng': bool(legacy_primary_rng),
            }
            t0 = anchor.t0_env_step
            t1 = None if t0 is None else int(t0) + span_steps - 1
            event['target_hold_release_env_step'] = None if t1 is None else int(t1) + 1
        if writer._rows:
            writer.finish(success=success, termination=termination, model_metadata=adapter.model_metadata(), policy_protocol=adapter.policy_protocol(), extra_meta={
                'occ_still': {'protocol':('online_threshold_window_hold_stationary_world_background_fill_two_camera' if online_threshold and target_hold_scope == 'occlusion_window' else ('online_threshold_permanent_hold_diagnostic' if online_threshold else 'stationary_world_background_fill_two_camera')),
                              'trigger':event,'trigger_mode':trigger_mode,
                              'occlusion_span':None if t0 is None else [t0,t1],
                              'occlusion_env_steps':span_steps,'occlusion_policy_observation_count':sum(1 for item in masker.records if item['role']=='policy_observation'),
                              'target_entity':spec.manipulated_objects[0],'target_body_name':target_body_name,
                              'target_joint_name':target_joint_name,
                              'mask_records':masker.records,'mask_failures':masker.failures,'target_physics_modified':False},
                'policy_steps':policy_steps,'wall_time_s':time.time()-started,'exception_traceback':error,
            })
        try: env.close()
        except Exception: pass
    result={'schema_version':'libero_full_state_occ_still_v1_result','model':adapter.model_name,'task_id':task_id,'init_state_index':seed,
            'success':success,'termination':termination,'policy_steps':policy_steps,'policy_calls':calls,'trigger':event,
            'trigger_mode':trigger_mode,'occlusion_span':None if t0 is None else [t0,t1],
            'occlusion_steps':span_steps,'occluded_policy_observations':sum(1 for item in masker.records if item['role']=='policy_observation'),
            'rollout_dir':str(rollout_dir),'wall_time_s':time.time()-started,'exception_traceback':error}
    write_json(result_path,result); print(json.dumps(result,sort_keys=True),flush=True); return result


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',required=True,choices=['pi05','fastwam','lingbot_va','vla_jepa'])
    parser.add_argument('--tasks',default='0,2,3,5,9'); parser.add_argument('--host',default='127.0.0.1'); parser.add_argument('--port',type=int,default=8000)
    parser.add_argument('--gpu',required=True,type=int); parser.add_argument('--output-root',type=Path,default=DEFAULT_OUT)
    parser.add_argument('--only-init-index',type=int); parser.add_argument('--overwrite',action='store_true')
    parser.add_argument('--trigger-mode',choices=['ddi_aligned','clean_fixed','online_threshold'],default='ddi_aligned',
                        help='ddi_aligned: paired screen step; clean_fixed: clean trace step; online_threshold: actual rollout first crossing')
    parser.add_argument('--trigger-distance-cm',type=float,default=None,
                        help='distance for clean_fixed or online_threshold; omit only for ddi_aligned')
    parser.add_argument('--trigger-band-halfwidth-cm',type=float,default=1.0,
                        help='half-width of the clean-distance trigger band (default: 1cm)')
    parser.add_argument('--target-hold-scope',choices=['occlusion_window','after_trigger_forever'],default='occlusion_window',
                        help='online trigger: hold target only through t1 (correct protocol), or retain the legacy-bug diagnostic')
    parser.add_argument('--legacy-primary-rng',action='store_true',
                        help='reproduce old π0.5/FastWAM seed-0 primary-action selection; off by default for paired clean/occ rollouts')
    args=parser.parse_args(); os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu)
    if args.trigger_mode == 'clean_fixed' and args.trigger_distance_cm is None:
        raise ValueError('clean_fixed requires --trigger-distance-cm')
    if args.trigger_mode == 'online_threshold' and args.trigger_distance_cm is None:
        raise ValueError('online_threshold requires --trigger-distance-cm')
    if args.trigger_mode == 'ddi_aligned' and args.trigger_distance_cm is not None:
        raise ValueError('ddi_aligned does not accept --trigger-distance-cm; use clean_fixed or online_threshold')
    if args.gpu==1 and args.model!='lingbot_va': raise ValueError('GPU1 remains reserved for LingBot-VA')
    from libero.libero import benchmark
    suite=benchmark.get_benchmark_dict()['libero_10'](); adapter=make_adapter(args.model,args.host,args.port)
    for task_id in [int(value) for value in args.tasks.split(',')]:
        if task_id not in TASKS: raise ValueError(f'unsupported task {task_id}')
        seeds=seed_pool(adapter.model_name,task_id)
        if args.only_init_index is not None: seeds=[args.only_init_index]
        for seed in seeds:
            run_episode(
                adapter=adapter,suite=suite,task_id=task_id,seed=seed,out_root=args.output_root,
                physical_gpu=args.gpu,overwrite=args.overwrite,
                trigger_mode=args.trigger_mode, target_distance_cm=args.trigger_distance_cm,
                band_halfwidth_cm=args.trigger_band_halfwidth_cm,
                legacy_primary_rng=bool(args.legacy_primary_rng),
                target_hold_scope=args.target_hold_scope,
            )

if __name__=='__main__': main()

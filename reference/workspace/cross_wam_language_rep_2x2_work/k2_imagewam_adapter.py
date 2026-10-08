from __future__ import annotations
from wam_causal_audit.paths import resolve as _release_path

import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

IW=Path(_release_path('@WORKSPACE@/ImageWAM'))
for p in (IW,IW/'src',IW/'experiments/libero',IW/'scripts/first_grasp_lock',IW/'scripts/causal_world_rep',IW/'scripts/same_frame_goal_kv',Path(_release_path('@WORKSPACE@/LIBERO')),Path(_release_path('@WORKSPACE@/evaluation/d2_ghost_pilot'))):
    if str(p) not in sys.path: sys.path.insert(0,str(p))

from libero.libero import benchmark
from imagewam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from run_imagewam_first_grasp_lock import body_name, prepare_model
from run_libero_object_causal import DONOR_TASK, SCENE_OBJECTS, TARGETS, make_env, model_input, normalized_to_env
from d2_ghost_scheduler import body_pos, get_eef_pos, get_sim, target_gripper_contact
from controlled_prefill import cache_sha256, prefill_flux2_cache
from run_one_call import ALT_INSTRUCTION, action_infer
from world_rep import WorldRepIntervention, layer_groups, parameter_signature, world_rep_pair_stats


class FGLShim:
    get_sim=staticmethod(get_sim); get_eef_pos=staticmethod(get_eef_pos); body_pos=staticmethod(body_pos); target_gripper_contact=staticmethod(target_gripper_contact)


class ImageWAMAdapter:
    model_name='imagewam'
    fgl=FGLShim
    def __init__(self,gpu_id:int):
        os.environ['MUJOCO_GL']='egl'; os.environ['PYOPENGL_PLATFORM']='egl'
        args=SimpleNamespace(ckpt=str(IW/'checkpoints/imagewam_release/libero/flux2_klein_4b/model.pt'),dataset_stats=str(IW/'checkpoints/imagewam_release/libero/flux2_klein_4b/dataset_stats.json'),
            flux2_src=str(IW/'third_party/flux2'),flux2_model=str(IW/'checkpoints/flux2/FLUX.2-klein-base-4B/flux-2-klein-base-4b.safetensors'),
            ae_model=str(IW/'checkpoints/flux2/FLUX.2-dev/ae.safetensors'),qwen3_model=str(IW/'checkpoints/qwen/Qwen3-4B'),gpu=gpu_id)
        self.cfg,self.model,self.processor=prepare_model(args); self.suite=benchmark.get_benchmark_dict()['libero_object'](); self.replan_steps=12
    def make_env(self,task:int,state:int): return make_env(self.suite,task,state)
    def body_name(self,env:Any,entity:str): return body_name(env,entity)
    def scene_objects(self,task:int): return tuple(SCENE_OBJECTS[task])
    def source_target(self,task:int): return TARGETS[task]
    def alternate_target(self,task:int): return TARGETS[DONOR_TASK[task]]
    def alternate_instruction(self,task:int): return ALT_INSTRUCTION[task]
    def normalized_to_env(self,action:torch.Tensor)->np.ndarray: return normalized_to_env(action,self.processor)
    def parameter_signature(self): return parameter_signature(self.model)
    @torch.no_grad()
    def call(self,obs,source_instruction,alternate_instruction,seed,cell,active):
        image,proprio=model_input(self.model,self.processor,self.cfg,obs)
        source_prompt=DEFAULT_PROMPT.format(task=source_instruction); goal_prompt=DEFAULT_PROMPT.format(task=alternate_instruction)
        source_rep,source_meta=prefill_flux2_cache(self.model,prompt=source_prompt,input_image=image,proprio=proprio)
        goal_rep,goal_meta=prefill_flux2_cache(self.model,prompt=goal_prompt,input_image=image,proprio=proprio)
        if source_meta['input_image_sha256']!=goal_meta['input_image_sha256'] or source_meta['proprio_sha256']!=goal_meta['proprio_sha256']: raise AssertionError('ImageWAM same-state mismatch')
        source_capture=WorldRepIntervention(mode='capture')
        source_action=action_infer(self.model,source_instruction,image,proprio,seed,source_capture)
        goal_capture=WorldRepIntervention(mode='capture')
        goal_action=action_infer(self.model,alternate_instruction,image,proprio,seed,goal_capture)
        if cache_sha256(source_capture.captured)!=cache_sha256(source_rep) or cache_sha256(goal_capture.captured)!=cache_sha256(goal_rep): raise AssertionError('ImageWAM prefill/action cache mismatch')
        sh=cache_sha256(source_rep); gh=cache_sha256(goal_rep)
        if active and cell=='AB':
            control=WorldRepIntervention(mode='donor',donor=goal_rep,layers=layer_groups(25)['all'],token_scope='image',expected_source_inputs=source_capture.source_inputs)
            executed=action_infer(self.model,source_instruction,image,proprio,seed,control); ih=gh
        elif active and cell=='BA':
            control=WorldRepIntervention(mode='donor',donor=source_rep,layers=layer_groups(25)['all'],token_scope='image',expected_source_inputs=goal_capture.source_inputs)
            executed=action_infer(self.model,alternate_instruction,image,proprio,seed,control); ih=sh
        elif active and cell=='BB': control=None; executed=goal_action; ih=gh
        else: control=None; executed=source_action; ih=sh
        return {'source_action':source_action,'goal_action':goal_action,'executed_action':executed,'proprio_cpu':proprio.detach().cpu().float().numpy(),'gripper_state':float(proprio.detach().cpu().float().reshape(-1)[-1]),
            'debug':{'interface':'25-layer image-token K/V only','factorial_cell':cell,'action_language':('L_B' if active and cell in {'BA','BB'} else 'L_A'),'representation':('Z_B' if active and cell in {'AB','BB'} else 'Z_A'),'rgb_sha256':source_meta['input_image_sha256'],'proprio_sha256':source_meta['proprio_sha256'],
                'action_noise_sha256':source_capture.source_inputs['initial_action_latents']['sha256'],
                'source_rep_sha256':sh,'goal_rep_sha256':gh,'injected_rep_sha256':ih,'release_source_rep_bit_exact':bool(active or ih==sh),'text_proprio_prefix_untouched':True,
                'injected_tensor_count':0 if control is None else len(control.debug.get('injected_tensors',[]))},
            'release_representation':None if active else {'interface':'image-token K/V','source_sha256':sh,'goal_sha256':gh,'clean_post_release_sha256':ih,'source_goal_cosine':world_rep_pair_stats(source_rep,goal_rep,token_scope='image')['source_donor_cosine'],'clean_source_cosine':1.0,'clean_goal_cosine':world_rep_pair_stats(source_rep,goal_rep,token_scope='image')['source_donor_cosine'],
                'clean_equals_source':ih==sh,'clean_equals_goal':ih==gh}}

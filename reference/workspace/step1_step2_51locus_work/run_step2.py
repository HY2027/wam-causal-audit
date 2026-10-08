#!/usr/bin/env python3
from __future__ import annotations
from wam_causal_audit.paths import resolve as _release_path

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


WORK = Path(__file__).resolve().parent
WEEK1 = Path(_release_path('@WORKSPACE@/week1_audit_work'))
for path in (WORK, WEEK1):
    if str(path) not in sys.path: sys.path.insert(0, str(path))

from capture import make_capture  # noqa: E402
from protocol import (load_npz, observation_sha256, radial_geometry_path, radial_observation_path,
                      radial_result_path, selected_clean_call, tensor_sha256, write_json)  # noqa: E402
from registry_features import model_loci  # noqa: E402
from registry_interventions import (BadPatchTrace, ImagePatchTrace, action_comparison, closing_drive,
    clone_image_cache, idm_infer, image_cache_locus_hash, joint_infer, patch_image_cache, patch_latent,
    patch_static_cache, selected_loci, static_cache_locus_hash, tensor_collection_hash)  # noqa: E402


OUT = WORK / "step2_cases"
DENOMINATOR_FLOOR = 1e-9
FAULT_DOSE_CM = 2.0


def valid_cases(model: str) -> list[dict[str, Any]]:
    output = []
    for task in range(4):
        for state in range(30):
            path = radial_geometry_path(model, task, state, FAULT_DOSE_CM)
            if not path.is_file(): continue
            geometry = json.loads(path.read_text(encoding="utf-8"))
            if geometry["donor_valid"]:
                output.append({"model": model, "task_id": task, "source_state_id": state,
                               "case_id": f"{model}__task_{task}__state_{state:02d}",
                               "geometry": geometry, "geometry_artifact": str(path)})
    return sorted(output, key=lambda row: (row["source_state_id"], row["task_id"]))


def max_error(left: torch.Tensor, right: torch.Tensor) -> tuple[float, float, bool]:
    delta = left.detach().cpu().float() - right.detach().cpu().float()
    return float(delta.abs().max()), float(torch.linalg.vector_norm(delta)), bool(torch.equal(left, right))


class DirectEngine:
    def __init__(self, runner, source_obs, donor_obs, instruction, seed):
        from direct_core import action_from_cache, build_current_cache
        self.runner=runner; self.seed=seed; self.action_from_cache=action_from_cache
        self.ps=runner._prepared(source_obs,instruction); self.pd=runner._prepared(donor_obs,instruction)
        self.cs=build_current_cache(runner.policy,self.ps); self.cd=build_current_cache(runner.policy,self.pd)
        self.clean_trace=BadPatchTrace(runner.model,"direct",capture=True)
        self.clean=self._raw(self.cs["cache"],self.clean_trace)
        self.fault_trace=BadPatchTrace(runner.model,"direct",capture=True)
        self.fault=self._raw(self.cd["cache"],self.fault_trace)
    def _raw(self,cache,trace):
        trace.install()
        try:return self.action_from_cache(self.runner.policy,self.ps,self.cs,cache,seed=self.seed)["action"]
        finally:trace.uninstall()
    def infer(self,base,donor,locus_ids):
        metas=selected_loci("direct",locus_ids); base_cache=self.cs["cache"] if base=="clean" else self.cd["cache"]
        donor_cache=self.cs["cache"] if donor=="clean" else self.cd["cache"]
        cache=patch_static_cache(base_cache,donor_cache,metas)
        donor_trace=self.clean_trace if donor=="clean" else self.fault_trace
        trace=BadPatchTrace(self.runner.model,"direct",donor=donor_trace,locus_ids=locus_ids)
        return self._raw(cache,trace)
    def locus_hash(self,condition,locus_id):
        meta=selected_loci("direct",(locus_id,))[0]
        if meta["family"]=="IMAGE_KV":
            cache=self.cs["cache"] if condition=="clean" else self.cd["cache"]
            return static_cache_locus_hash(cache,meta)
        if meta["family"] in {"ACTION_QKV","ACTION_HIDDEN"}:
            return (self.clean_trace if condition=="clean" else self.fault_trace).locus_hash(locus_id)
        return tensor_collection_hash((self.ps["context"],self.ps["context_mask"]))


class JointEngine:
    def __init__(self,runner,source_obs,donor_obs,instruction,seed):
        self.runner=runner; self.seed=seed; self.ps=runner._prepared(source_obs,instruction); self.pd=runner._prepared(donor_obs,instruction)
        self.clean_trace=BadPatchTrace(runner.model,"joint",capture=True); self.clean=joint_infer(runner.policy,self.ps,self.ps,seed,self.clean_trace)
        self.fault_trace=BadPatchTrace(runner.model,"joint",capture=True); self.fault=joint_infer(runner.policy,self.pd,self.ps,seed,self.fault_trace)
    def infer(self,base,donor,locus_ids):
        video=self.ps if base=="clean" else self.pd; donor_trace=self.clean_trace if donor=="clean" else self.fault_trace
        trace=BadPatchTrace(self.runner.model,"joint",donor=donor_trace,locus_ids=locus_ids)
        return joint_infer(self.runner.policy,video,self.ps,self.seed,trace)
    def locus_hash(self,condition,locus_id):
        meta=selected_loci("joint",(locus_id,))[0]
        if meta["family"]=="ACTION_CONTEXT": return tensor_collection_hash((self.ps["context"],self.ps["context_mask"]))
        return (self.clean_trace if condition=="clean" else self.fault_trace).locus_hash(locus_id)


class IDMEngine:
    def __init__(self,runner,source_obs,donor_obs,instruction,seed):
        from common import action_noise, generate_video_latent
        self.runner=runner; self.seed=seed; self.ps=runner._prepared(source_obs,instruction); self.pd=runner._prepared(donor_obs,instruction)
        self.source_latent=generate_video_latent(runner.policy,self.ps,seed=seed)["latent"]
        donor_latent=generate_video_latent(runner.policy,self.pd,seed=seed)["latent"]
        self.fault_latent=self.source_latent.clone(); self.fault_latent[:,:,1:]=donor_latent[:,:,1:]
        self.noise=action_noise(runner.policy,seed)
        self.clean_trace=BadPatchTrace(runner.model,"idm",capture=True)
        self.clean,self.clean_cache,self.tpg=idm_infer(runner,self.ps,self.source_latent,self.noise,self.clean_trace)
        self.fault_trace=BadPatchTrace(runner.model,"idm",capture=True)
        self.fault,self.fault_cache,tpg=idm_infer(runner,self.ps,self.fault_latent,self.noise,self.fault_trace)
        if tpg!=self.tpg: raise AssertionError("IDM token-group drift")
    def infer(self,base,donor,locus_ids):
        metas=selected_loci("idm",locus_ids)
        base_latent=self.source_latent if base=="clean" else self.fault_latent
        donor_latent=self.source_latent if donor=="clean" else self.fault_latent
        latent=patch_latent(base_latent,donor_latent,metas)
        donor_cache=self.clean_cache if donor=="clean" else self.fault_cache
        donor_trace=self.clean_trace if donor=="clean" else self.fault_trace
        trace=BadPatchTrace(self.runner.model,"idm",donor=donor_trace,locus_ids=locus_ids)
        action,_cache,_=idm_infer(self.runner,self.ps,latent,self.noise,trace,cache_donor=donor_cache,locus_ids=locus_ids)
        return action
    def locus_hash(self,condition,locus_id):
        meta=selected_loci("idm",(locus_id,))[0]
        if meta["family"]=="VIDEO_WORLD_KV":
            return static_cache_locus_hash(self.clean_cache if condition=="clean" else self.fault_cache,meta,self.tpg)
        if meta["family"]=="VIDEO_LATENT":
            latent=self.source_latent if condition=="clean" else self.fault_latent
            value=latent[:,:,0:1] if meta["temporal_selection"]=="current" else latent[:,:,1:]
            return tensor_collection_hash((value,))
        if meta["family"] in {"ACTION_QKV","ACTION_HIDDEN"}:
            return (self.clean_trace if condition=="clean" else self.fault_trace).locus_hash(locus_id)
        return tensor_collection_hash((self.ps["context"],self.ps["context_mask"]))


class ImageEngine:
    def __init__(self,runner,source_obs,donor_obs,instruction,seed):
        from run_one_call import action_infer
        from run_libero_object_causal import model_input
        from world_rep import WorldRepIntervention,layer_groups
        self.runner=runner; self.seed=seed; self.instruction=instruction; self.action_infer=action_infer
        self.WorldRepIntervention=WorldRepIntervention; self.layers=layer_groups(25)["all"]
        self.si,self.sp=model_input(runner.model,runner.processor,runner.cfg,source_obs)
        di,dp=model_input(runner.model,runner.processor,runner.cfg,donor_obs)
        source_ctl=WorldRepIntervention(mode="capture"); self.clean_trace=ImagePatchTrace(runner.model,capture=True); self.clean_trace.install()
        try:self.clean=action_infer(runner.model,instruction,self.si,self.sp,seed,source_ctl)
        finally:self.clean_trace.uninstall()
        donor_ctl=WorldRepIntervention(mode="capture"); action_infer(runner.model,instruction,di,dp,seed,donor_ctl)
        self.source_cache=source_ctl.captured; self.donor_cache=donor_ctl.captured; self.source_inputs=source_ctl.source_inputs
        if self.source_cache is None or self.donor_cache is None: raise AssertionError("Image cache missing")
        self.fault_trace=ImagePatchTrace(runner.model,capture=True)
        self.fault=self._infer_cache(self.donor_cache,self.fault_trace,"donor")
    def _infer_cache(self,cache,trace,mode):
        ctl=self.WorldRepIntervention(mode=mode,donor=cache,layers=self.layers,token_scope="all",expected_source_inputs=self.source_inputs)
        trace.install()
        try:return self.action_infer(self.runner.model,self.instruction,self.si,self.sp,self.seed,ctl)
        finally:trace.uninstall()
    @staticmethod
    def _cache_bit_exact(left, right):
        if int(left["txt_len"]) != int(right["txt_len"]) or int(left["img_len"]) != int(right["img_len"]):
            return False
        left_rows = list(left["double"]) + list(left["single"])
        right_rows = list(right["double"]) + list(right["single"])
        return (len(left_rows) == len(right_rows)
                and all(torch.equal(a[k], b[k]) for a, b in zip(left_rows, right_rows)
                        for k in ("k", "v")))
    def infer(self,base,donor,locus_ids):
        metas=selected_loci("imagewam",locus_ids); base_cache=self.source_cache if base=="clean" else self.donor_cache
        donor_cache=self.source_cache if donor=="clean" else self.donor_cache
        cache=patch_image_cache(base_cache,donor_cache,metas)
        donor_trace=self.clean_trace if donor=="clean" else self.fault_trace
        trace=ImagePatchTrace(self.runner.model,donor=donor_trace,locus_ids=locus_ids)
        # Select identity from the actual cache content, not only from the
        # semantic base label.  In a complete repair, replacing all image and
        # prefix K/V turns the fault-base cache bit-exactly back into the cache
        # naturally produced by the current source observation.  Labelling
        # that valid endpoint as a donor injection trips the anti-reuse guard.
        # Non-identical donor injections retain the guard unchanged.
        mode="identity" if self._cache_bit_exact(cache,self.source_cache) else "donor"
        return self._infer_cache(cache,trace,mode)
    def locus_hash(self,condition,locus_id):
        meta=selected_loci("imagewam",(locus_id,))[0]
        if meta["family"] in {"IMAGE_KV","PREFILL_PREFIX_KV"}:
            return image_cache_locus_hash(self.source_cache if condition=="clean" else self.donor_cache,meta)
        return (self.clean_trace if condition=="clean" else self.fault_trace).locus_hash(locus_id)


ENGINES={"direct":DirectEngine,"joint":JointEngine,"idm":IDMEngine,"imagewam":ImageEngine}


def execute_case(runner,case,locus_ids,overwrite=False):
    model=case["model"]; task=case["task_id"]; state=case["source_state_id"]
    out=OUT/model/f"task_{task}"/f"state_{state:02d}"; result_path=out/"result.json"
    if result_path.is_file() and not overwrite:
        existing=json.loads(result_path.read_text(encoding="utf-8"))
        if existing.get("status") == "COMPLETED" and existing["locus_ids"]==list(locus_ids):
            return existing
    selection=selected_clean_call(model,task,state)
    source_path=radial_observation_path(model,task,state,0.0); donor_path=radial_observation_path(model,task,state,FAULT_DOSE_CM)
    source=load_npz(source_path); donor=load_npz(donor_path); started=time.time()
    engine=ENGINES[model](runner,source,donor,selection["instruction"],selection["seed"])
    clean=engine.clean; fault=engine.fault; clean_drive,clean_env=closing_drive(runner,clean); fault_drive,fault_env=closing_drive(runner,fault)
    block_c_root = (WORK / "part0_imagewam_block_c_rtx4090" / "block_c" if model == "imagewam"
                    else Path(_release_path('@WORKSPACE@/runs/provenance_encoding_authority_audit_week1/block_c')))
    authority_case = block_c_root / "cases" / model / f"task_{task}" / f"state_{state:02d}" / "result.json"
    authority = json.loads(authority_case.read_text(encoding="utf-8"))
    expected_clean = authority["conditions"]["clean"]["action_sha256"]
    expected_fault = authority["conditions"]["fault"]["action_sha256"]
    if tensor_sha256(clean) != expected_clean or tensor_sha256(fault) != expected_fault:
        raise AssertionError(
            f"Validated baseline mismatch {case['case_id']}: "
            f"clean {tensor_sha256(clean)} != {expected_clean}; fault {tensor_sha256(fault)} != {expected_fault}"
        )
    denominator=clean_drive-fault_drive
    geometry=case["geometry"]; target_direction=np.asarray(geometry["source_target_position_m"],dtype=np.float64)-np.asarray(geometry["source_eef_position_m"],dtype=np.float64)
    rows=[]
    for index,locus_id in enumerate(locus_ids):
        repair=engine.infer("fault","clean",(locus_id,)); corrupt=engine.infer("clean","fault",(locus_id,))
        identity_clean=engine.infer("clean","clean",(locus_id,)); identity_fault=engine.infer("fault","fault",(locus_id,))
        clean_max,clean_l2,clean_exact=max_error(identity_clean,clean); fault_max,fault_l2,fault_exact=max_error(identity_fault,fault)
        if not clean_exact or not fault_exact:
            failure={"model":model,"case_id":case["case_id"],"locus_id":locus_id,"clean_identity_max_abs":clean_max,
                     "fault_identity_max_abs":fault_max,"status":"HALT_IDENTITY_FAILURE"}
            write_json(out/"identity_failure.json",failure); raise AssertionError(f"Identity failure {failure}")
        repair_drive,repair_env=closing_drive(runner,repair); corrupt_drive,corrupt_env=closing_drive(runner,corrupt)
        informative=abs(denominator)>=DENOMINATOR_FLOOR
        row={
            "case_id":case["case_id"],"model":model,"task_id":task,"source_state_id":state,"locus_id":locus_id,
            "clean_closing_drive":clean_drive,"fault_closing_drive":fault_drive,"denominator":denominator,
            "denominator_floor":DENOMINATOR_FLOOR,"denominator_informative":informative,
            "repair_closing_drive":repair_drive,"corrupt_closing_drive":corrupt_drive,
            "repair_numerator":repair_drive-fault_drive,"corrupt_numerator":clean_drive-corrupt_drive,
            "RepairRatio":(repair_drive-fault_drive)/denominator if informative else None,
            "CorruptRatio":(clean_drive-corrupt_drive)/denominator if informative else None,
            **{f"repair_{k}":v for k,v in action_comparison(repair,clean,repair_env,clean_env,target_direction).items()},
            **{f"corrupt_{k}":v for k,v in action_comparison(corrupt,clean,corrupt_env,clean_env,target_direction).items()},
            "clean_identity_bit_exact":clean_exact,"fault_identity_bit_exact":fault_exact,
            "clean_identity_max_abs":clean_max,"fault_identity_max_abs":fault_max,
            "clean_identity_l2":clean_l2,"fault_identity_l2":fault_l2,
            "clean_action_sha256":tensor_sha256(clean),"fault_action_sha256":tensor_sha256(fault),
            "clean_locus_sha256":engine.locus_hash("clean",locus_id),"fault_locus_sha256":engine.locus_hash("fault",locus_id),
            "paired_case_id":case["case_id"],"generated_actions_executed":False,
        }
        rows.append(row)
        partial={"status":"IN_PROGRESS","model":model,"case_id":case["case_id"],"locus_ids":list(locus_ids),"rows":rows,
                 "clean_action_sha256":tensor_sha256(clean),"fault_action_sha256":tensor_sha256(fault)}
        write_json(result_path,partial)
        print(json.dumps({"model":model,"case_id":case["case_id"],"locus":locus_id,"done":index+1,"total":len(locus_ids)}),flush=True)
    interface=[row["locus_id"] for row in model_loci(model) if row["in_validated_causal_interface"]=="yes"]
    full_corrupt=engine.infer("clean","fault",interface); full_drive,_=closing_drive(runner,full_corrupt)
    full_ratio=(clean_drive-full_drive)/denominator if abs(denominator)>=DENOMINATOR_FLOOR else None
    result={
        "status":"COMPLETED","model":model,"case_id":case["case_id"],"task_id":task,"source_state_id":state,
        "locus_ids":list(locus_ids),"rows":rows,"validated_interface_partition":interface,
        "full_interface_corrupt_closing_drive":full_drive,"full_interface_CorruptRatio":full_ratio,
        "sum_single_interface_CorruptRatio":sum(float(row["CorruptRatio"]) for row in rows if row["locus_id"] in interface and row["CorruptRatio"] is not None) if full_ratio is not None else None,
        "source_observation_sha256":observation_sha256(source),"donor_observation_sha256":observation_sha256(donor),
        "source_observation_artifact":str(source_path),"donor_observation_artifact":str(donor_path),
        "geometry_artifact":case["geometry_artifact"],"validated_fault_reference_artifact":str(authority_case),
        "hardware_device":torch.cuda.get_device_name(0),
        "runtime_seconds":time.time()-started,"generated_actions_executed":False,"model_weights_modified":False,
    }
    write_json(result_path,result); return result


def run(args):
    all_ids=[row["locus_id"] for row in model_loci(args.model)]
    locus_ids=all_ids if args.loci is None else [value for value in args.loci.split(",") if value]
    if not set(locus_ids)<=set(all_ids): raise ValueError("Locus not in frozen registry")
    cases=valid_cases(args.model)[:args.max_cases]
    selected_path=WORK/f"step2_selected_cases_{args.model}_{args.max_cases}.json"
    write_json(selected_path,{"model":args.model,"selection_rule":"sort by (source_state_id, task_id), first N valid d_fault=2cm cases",
        "N":args.max_cases,"case_ids":[row["case_id"] for row in cases],"locus_ids":locus_ids,"outcome_based_filtering":False})
    runner=make_capture(args.model,0); results=[]
    for index,case in enumerate(cases):
        results.append(execute_case(runner,case,locus_ids,args.overwrite)); gc.collect(); torch.cuda.empty_cache()
        print(json.dumps({"model":args.model,"case_done":index+1,"case_total":len(cases)}),flush=True)
    write_json(WORK/f"step2_complete_{args.model}_{args.max_cases}.json",{
        "model":args.model,"case_count":len(results),"locus_count":len(locus_ids),
        "all_identity_bit_exact":all(all(row["clean_identity_bit_exact"] and row["fault_identity_bit_exact"] for row in result["rows"]) for result in results),
        "repair_corrupt_case_ids_paired":all(all(row["case_id"]==row["paired_case_id"] for row in result["rows"]) for result in results),
        "hardware_device":torch.cuda.get_device_name(0),"generated_actions_executed":False})


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--model",choices=("direct","joint","idm","imagewam"),required=True)
    parser.add_argument("--gpu-id",type=int,required=True); parser.add_argument("--max-cases",type=int,default=50)
    parser.add_argument("--loci"); parser.add_argument("--overwrite",action="store_true"); args=parser.parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES",str(args.gpu_id)); os.environ.setdefault("MUJOCO_GL","egl"); os.environ.setdefault("PYOPENGL_PLATFORM","egl")
    run(args)


if __name__=="__main__":main()

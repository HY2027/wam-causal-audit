#!/usr/bin/env python3
"""Bounded new native-source acquisition; separate phase2 validation namespace."""
from wam_causal_audit.paths import resolve as _release_path
import argparse,csv,hashlib,json,random,re,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
from phase2_local_future import sha,load,dump,table
WORK=Path(__file__).resolve().parent
ROOT=Path(_release_path('@DATA@/wam_factor_routing_v5'))
OUT=ROOT/'phase2_native_independent_validation_v1_20260910'
PREVIOUS=ROOT/'phase2_local_future_restoration_v1_20260910'
DOSES=(-4.,-2.,-1.,1.,2.,4.)

def freeze():
    sys.path.insert(0,_release_path('@WORKSPACE@/LIBERO'))
    from libero.libero import benchmark,get_libero_path
    if (OUT/'acquisition_freeze_manifest.json').exists():raise RuntimeError('Already frozen')
    OUT.mkdir(exist_ok=True)
    suite=benchmark.get_benchmark_dict()['libero_object']();used={0:set(range(30)),1:set(range(30))};audit=[]
    paths=subprocess.run(['rg','--files',str(ROOT),_release_path('@WORKSPACE@/runs'),'-g','*registry*.csv'],capture_output=True,text=True).stdout.splitlines()
    for path in paths:
        if str(OUT) in path:continue
        with Path(path).open() as f:
            reader=csv.DictReader(f);fields=set(reader.fieldnames or [])
            key=next((k for k in ['init_state_id','initial_state_id','initial_state_index'] if k in fields),None)
            if key and 'task_id' in fields:
                for r in reader:
                    try:t=int(float(r['task_id']));i=int(float(r[key]))
                    except (ValueError,TypeError):continue
                    if t in used:used[t].add(i);audit.append(dict(task_id=t,init_state_id=i,source=path,source_hash=sha(path),mapping_field=key))
    catalog=Path(_release_path('@WORKSPACE@/runs/radial_phase_validity_dose_sweep/source_state_catalog.json'))
    for t in used:
        audit.extend(dict(task_id=t,init_state_id=i,source=str(catalog),source_hash=sha(catalog),mapping_field='source_state_id -> native make_env init-state index, audited implementation') for i in range(30))
    table(OUT/'trajectory_overlap_audit.csv',audit)
    candidates=[];initial=[]
    for t in [0,1]:
        task=suite.get_task(t);p=Path(get_libero_path('init_states'))/task.problem_folder/task.init_states_file;a=suite.get_task_init_states(t)
        hashes=[hashlib.sha256(np.ascontiguousarray(np.asarray(x)).tobytes()).hexdigest() for x in a];usedhash={hashes[i] for i in used[t] if i<len(a)};chosenhash=set()
        for i,h in enumerate(hashes):
            reason='PREVIOUS_SOURCE' if i in used[t] else 'DUPLICATE_INITIAL_STATE' if h in usedhash or h in chosenhash else 'ELIGIBLE'
            row=dict(task_id=t,init_state_id=i,source_id=f'libero_object_task{t}_init{i:02d}',initial_state_hash=h,source_file=str(p),source_file_sha256=sha(p),status=reason)
            initial.append(row)
            if reason=='ELIGIBLE' and len([c for c in candidates if c['task_id']==t])<20:candidates.append(row);chosenhash.add(h)
    table(OUT/'initial_state_provenance.csv',initial);dump(OUT/'acquisition_candidates.json',candidates)
    protocol='''# Phase2 new native-source acquisition (frozen before execution)
This subprotocol collects new trajectories within already studied tasks, not a new scientific hypothesis. Prior four starts and blocked report remain immutable and excluded.
Tasks0/1: use ascending eligible benchmark initial-state indices after exclusions by stable initial-state ID and per-state content hash; duplicate initial states cannot be independent. Scan prior project registries with explicit init mappings plus audited old native sources0–29. At most20 attempts/task and stop at10 qualifying starts/task. No success/effect selection; no extension of400 environment control steps per attempt. Environment seed42; native policy seed720000+task*10000+init*100+policy_call, same earlier Joint native rule. Initial settle uses existing common.make_env; action replan10 and all10 world/action denoise steps. Weight/dtype/checkpoint unchanged, RTX4090 only.
PREGRASP: the saved policy-call boundary preceding the first actually executed native closing command>0.5 while aperture>=0.055; boundary gripper open and target not attached. Save every transient call snapshot until this first event, retain exactly that boundary. This is the old event rule, not a distance or effect threshold. Failed qualifications retained, no fallback state search. At first qualifying collection terminate as ACQUISITION_COMPLETE, not success/failure. The first qualified source per task is fixed dynamic technical subset: continue only enough to record30 subsequent executed controls from that boundary, within400 budget, then compare identical recorded controls from restored snapshot. If budget/termination prevents30, mark unavailable, no substitute.
New-input synchronization: at every native policy-call boundary preserve uninterrupted protected state/warmstart; one sim.forward and force_update observables without time advance; restore continuous qacc_warmstart afterward. Record old-live versus synchronized observation hash. This is the previously accepted B_SYNC/restore_v2 execution-v3 style, explicitly distinct from old A live inputs. New policy uses this synchronized observation consistently. Freeze selected new inputs only after new static and dynamic acceptance. No silent image substitution or stale/new cache mixing.
Snapshot: MuJoCo flat/time/qpos/qvel/act/ctrl/mocap/forces/warmstart/plugin if present; explicit gripper.current_action; OSC goal_pos/goal_ori/ori_ref/relative_ori/new_update; env timestep/cur_time/done; numpy/Python RNG state; action queue at fresh policy boundary(empty) and replan offset0; input observations and actual action history. Missing needed execution fields fail, no qpos-based gripper guess.
Restore uses existing confirmatory_common.restore with explicit fields and warmstart restored after forward. Static state/input identity on all selected sources. Fixed first source/task dynamic: compare all recorded controls including qpos/qvel/gripper, EEF/object pose and grasp on each step, exact original standard; no tolerance expansion. A substantive failure stops affected task.
Formal source registry is hash-frozen before donor/model interventions. F1 original generate_group1_donors.construct and its unchanged QC; only its state-loader is bound to the accepted new snapshot, with an explicit adapter and source-observation identity check. Doses±1,±2,±4cm. All invalid donors retained; no replacement. Original P legacy capture, W25–29,V0–4, R0/B/D/Pall/PW/PV/native10/10. No donor action execution.
Analysis: full32-position denormalized radial vector in recipient EEF→object axis. eB/eW/eV RMS toPall; primary eB-eW and eV-eW. Per-source donor equal weight, then equal source within each fixed task, equal tasks.10000 fixed-task trajectory paired bootstrap seed20260910. Report percentile95%CI and complete values; no significance/p-values so no uncorrected discovery claim. If p-values later supplied the two-primary family requires Holm, not independent single tests. No new equivalence or denominator threshold. Positive/negative doses and tasks separately. Secondary raw uP/uW/uV vectors/RMS and Pall-D errors. Nearzero directional ratios omitted rather than new threshold. Old technical data never pooled.
Stopping: all bounded attempts and accepted valid cases, or substantive technical failure per task; never expand K/tasks/layers or risk models. No donor closedloop. Failed task remains unvalidated, unaffected task continues.
'''
    (OUT/'native_source_acquisition_protocol.md').write_text(protocol)
    files=[Path(__file__),WORK/'phase2_local_future.py',WORK/'run_experiment_a.py',WORK/'confirmatory_common.py',Path(_release_path('@WORKSPACE@/experiments/wam_control_state_v3/generate_group1_donors.py')),Path(_release_path('@WORKSPACE@/experiments/wam_control_state_v3/group1_config.py')),PREVIOUS/'phase2_freeze_manifest.json',OUT/'native_source_acquisition_protocol.md',OUT/'initial_state_provenance.csv',OUT/'acquisition_candidates.json',OUT/'trajectory_overlap_audit.csv']
    dump(OUT/'acquisition_freeze_manifest.json',dict(status='ACQUISITION_AND_ANALYSIS_FROZEN',created_unix=time.time(),files={str(p):sha(p) for p in files},candidate_count=len(candidates),max_attempts_per_task=20,target_per_task=10,previous_blocked_manifest=sha(PREVIOUS/'phase2_completion_manifest.json')))
    print('FROZEN',len(candidates),flush=True)

def modules():
    import run_experiment_a as A
    import confirmatory_common as Q
    sys.path.insert(0,_release_path('@WORKSPACE@/experiments/wam_control_state_v3'))
    import generate_group1_donors as G
    return A,Q,G

def verify():
    f=load(OUT/'acquisition_freeze_manifest.json')
    for p,h in f['files'].items():assert sha(p)==h,p

def acquire(task,gpu):
    verify();import torch
    A,Q,G=modules();root=OUT/f'acquisition/task{task}';root.mkdir(parents=True,exist_ok=True)
    attempts=[];registry=[];checks=[];costs=[];diffs=[]
    def save():
        table(root/'native_acquisition_attempts.csv',attempts,['source_id','task_id','init_state_id','status','control_steps','qualified','error'])
        table(root/'restore_v2_acceptance_checks.csv',checks,['source_id','check','passed','max_abs_error','control_step'])
        table(root/'trajectory_registry.csv',registry)
        table(root/'compute_cost.csv',costs)
        table(root/'sync_input_differences.csv',diffs)
    def check(s,name,ok,error=0,step=-1):
        checks.append(dict(source_id=s,check=name,passed=bool(ok),max_abs_error=float(error),control_step=step));save()
        if not ok:raise RuntimeError('RESTORE_GATE_FAILED:'+name)
    def snapshot(env):
        sim=env.env.sim;c=env.env.robots[0].controller;s=Q.protected(sim)
        s['gripper_current_action']=np.asarray(env.env.robots[0].gripper.current_action).copy()
        for name in ['goal_ori','goal_pos','ori_ref','relative_ori']:
            if getattr(c,name,None) is not None:s['controller_'+name]=np.asarray(getattr(c,name)).copy()
        s.update(controller_new_update=np.asarray([bool(c.new_update)]),env_timestep=np.asarray([env.env.timestep]),env_cur_time=np.asarray([env.env.cur_time]),env_done=np.asarray([env.env.done]))
        state=np.random.get_state();s.update(numpy_rng_keys=state[1],numpy_rng_pos=np.asarray([state[2]]),numpy_rng_has_gauss=np.asarray([state[3]]),numpy_rng_cached_gaussian=np.asarray([state[4]]),python_rng_internal=np.asarray(random.getstate()[1],dtype=np.uint64),pending_actions=np.empty((0,7)),chunk_offset=np.asarray([0]))
        return s
    def sync(env,obs,sid):
        sim=env.env.sim;before=Q.protected(sim);old=Q.observation_hash(obs);sim.forward();new=dict(env.env._get_observations(force_update=True));sim.data.qacc_warmstart[:]=before['qacc_warmstart']
        if not np.array_equal(before['flat'],np.asarray(sim.get_state().flatten())):raise RuntimeError('SYNC_CHANGED_INTEGRATION_STATE')
        diffs.append(dict(source_id=sid,old_live_hash=old,synchronized_hash=Q.observation_hash(new),same=old==Q.observation_hash(new),time_unchanged=True))
        return new
    def restore_rng(s):
        np.random.set_state(('MT19937',s['numpy_rng_keys'],int(s['numpy_rng_pos'][0]),int(s['numpy_rng_has_gauss'][0]),float(s['numpy_rng_cached_gaussian'][0])))
        random.setstate((3,tuple(int(x) for x in s['python_rng_internal']),None))
    torch.set_num_threads(16);start=time.time();runner=A.make_capture('joint',gpu);costs.append(dict(source_id='LOAD',stage='model_load',seconds=time.time()-start));weight=A.model_weight_hash(runner.model)
    assert runner.policy.num_inference_steps==10 and runner.policy.replan_steps==10
    for candidate in [r for r in load(OUT/'acquisition_candidates.json') if r['task_id']==task]:
        if len(registry)>=10:break
        sid=candidate['source_id'];i=candidate['init_state_id'];directory=root/'sources'/sid;directory.mkdir(parents=True,exist_ok=True);env=None;rest=None;executed=[];calls=[];selected=None;boundary=None;first_attempt=False;status='NO_QUALIFYING_PREGRASP';err='';step=0
        try:
            env,taskobj,obs=Q.OLD.make_env(task,i);sim=env.env.sim;target=Q.OLD.TARGETS[task];call=0;queue=[]
            for step in range(400):
                if not queue:
                    obs=sync(env,obs,sid);snap=snapshot(env)
                    boundary=dict(snapshot=snap,observation={k:np.asarray(v).copy() for k,v in obs.items()},step=step,call=call,open=float(Q.OLD.aperture(obs))>=.055,attached=G.attachment_flag(env,target))
                    seed=720000+task*10000+i*100+call;t=time.time();prepared=runner._prepared(obs,taskobj.language);action,run=A.infer(runner,prepared,prepared,seed);costs.append(dict(source_id=sid,stage='native_policy',seconds=time.time()-t))
                    chunk=A.binary_env_action(action,runner.processor);queue=[r.astype(float) for r in chunk[:10]]
                    calls.append(dict(policy_call=call,step=step,seed=seed,action_hash=A.tensor_sha256(action),observation_hash=Q.observation_hash(obs)))
                    boundary['seed']=seed;boundary['native_action']=action.numpy();call+=1
                action=queue.pop(0);attempt=bool(action[6]>.5 and Q.OLD.aperture(obs)>=.055)
                if attempt and not first_attempt:
                    first_attempt=True
                    if boundary['open'] and not boundary['attached']:selected=boundary
                    else:status='FIRST_ATTEMPT_BOUNDARY_INELIGIBLE'
                obs,reward,done,_=env.step(action.tolist());obs=dict(obs);executed.append(action.copy())
                if selected is not None:
                    # Fixed dynamic subset: first qualified source in this task only.
                    need=30 if len(registry)==0 else 1
                    if len(executed)-selected['step']>=need:status='ACQUISITION_COMPLETE';break
                if first_attempt and selected is None:break
                if reward>0 or done:status='NATIVE_TERMINATED_BEFORE_COLLECTION_COMPLETE';break
            np.savez_compressed(directory/'executed_actions.npz',actions=np.asarray(executed));dump(directory/'policy_calls.json',calls)
            if selected is not None:
                st=selected['snapshot'];op=directory/'recipient_observation.npz';sp=directory/'recipient_state.npz';np.savez_compressed(op,**selected['observation']);np.savez_compressed(sp,**st);np.savez_compressed(directory/'native_boundary_action.npz',action=selected['native_action'])
                obj=selected['observation'];cid=f'P2NEW__task{task}__init{i:02d}__PREGRASP';row=dict(candidate_id=cid,base_state_id=cid,task_id=task,init_state_id=i,source_state_id=i,trajectory_id=sid,model='joint',phase='PREGRASP',policy_seed=selected['seed'],policy_call=selected['call'],environment_step=selected['step'],instruction=taskobj.language,target_identity=target,goal_identity='basket_1',state_path=str(sp),observation_path=str(op),recipient_state_path=str(sp),recipient_observation_path=str(op),state_file_sha256=sha(sp),observation_file_sha256=sha(op),flat_state_sha256=Q.raw_hash(st['flat']),observation_content_sha256=Q.observation_hash(obj),initial_state_hash=candidate['initial_state_hash'],initial_state_file=candidate['source_file'],initial_state_file_sha256=candidate['source_file_sha256'],source_actions_path=str(directory/'executed_actions.npz'),source_actions_sha256=sha(directory/'executed_actions.npz'))
                rest,robs,rst=Q.restore(row);restore_rng(st)
                check(sid,'static_observation',Q.observation_hash(robs)==Q.observation_hash(obj))
                check(sid,'static_flat_state',np.array_equal(np.asarray(rest.env.sim.get_state().flatten()),st['flat']))
                check(sid,'gripper_current_action',np.array_equal(rest.env.robots[0].gripper.current_action,st['gripper_current_action']))
                pos=np.asarray(rest.env.sim.data.get_body_xpos(G.body_name(rest,target))).copy();eef=np.asarray(rest.env.sim.data.site_xpos[rest.env.robots[0].eef_site_id]).copy();goal=np.asarray(rest.env.sim.data.get_body_xpos(G.body_name(rest,'basket_1'))).copy();row.update(object_position_m=pos.tolist(),eef_position_m=eef.tolist(),goal_position_m=goal.tolist())
                # Reconstruct uninterrupted reference from beginning using saved actual actions;
                # no model sampling enters the comparison.
                if len(registry)==0:
                    reference,_,refobs=Q.OLD.make_env(task,i)
                    try:
                        for a0 in executed[:selected['step']]:refobs,*_=reference.step(a0.tolist())
                        refobs=sync(reference,dict(refobs),sid)
                        check(sid,'continuous_boundary_flat',np.array_equal(np.asarray(reference.env.sim.get_state().flatten()),st['flat']),np.max(np.abs(np.asarray(reference.env.sim.get_state().flatten())-st['flat'])))
                        future=executed[selected['step']:selected['step']+30];check(sid,'dynamic_30_controls_available',len(future)==30)
                        for n,a0 in enumerate(future):
                            reference.step(a0.tolist());rest.step(a0.tolist());x=np.asarray(reference.env.sim.get_state().flatten());y=np.asarray(rest.env.sim.get_state().flatten())
                            check(sid,'dynamic_flat_bit_exact',np.array_equal(x,y),np.max(np.abs(x-y)),n)
                            check(sid,'dynamic_gripper_exact',np.array_equal(reference.env.robots[0].gripper.current_action,rest.env.robots[0].gripper.current_action),step=n)
                    finally:reference.close()
                registry.append(row);dump(directory/'selected_state.json',row);status='ACQUISITION_COMPLETE'
            attempts.append(dict(source_id=sid,task_id=task,init_state_id=i,status=status,control_steps=len(executed),qualified=selected is not None,error=''));save()
        except Exception:
            err=traceback.format_exc();attempts.append(dict(source_id=sid,task_id=task,init_state_id=i,status='TECHNICAL_FAILURE_TASK_STOPPED',control_steps=len(executed),qualified=False,error=err));save();dump(root/'failure.json',dict(source_id=sid,error=err));break
        finally:
            if rest is not None:rest.close()
            if env is not None:env.close()
    assert weight==A.model_weight_hash(runner.model)
    dump(root/'completion.json',dict(task_id=task,qualified=len(registry),attempts=len(attempts),status='TASK_BLOCKED' if (root/'failure.json').exists() else 'BOUNDED_ACQUISITION_COMPLETE',weight_hash=weight))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['freeze','acquire']);p.add_argument('--task',type=int);p.add_argument('--gpu',type=int,default=0);args=p.parse_args()
    freeze() if args.stage=='freeze' else acquire(args.task,args.gpu)

"""S1_WRAPPER_FIX_V3: reuse V2 provenance, all-recipient identity gate, missing node only."""
import argparse, csv, gc, hashlib, json, os, time, traceback
from pathlib import Path
import numpy as np
import s1_followup as H
import s1_hash_adapter_v2 as V
import s1_wrapper_v3 as W

ROOT=H.S1/'v3'
CK='7cd0af01f4a51838510265685fa533836c6c9c29e0c37e3f7235d14ac676c928'
def frozen():return H.load(H.OUT/'protocol_freeze_s1.json')
def setup():
    assert not ROOT.exists(),'V3 must not overwrite an existing attempt'
    previous=[H.info(p) for p in H.OUT.rglob('*') if p.is_file()]
    previous += [H.info(H.WORK/n) for n in ['s1_followup.py','s1_statistics.py','s1_hash_adapter_v2.py','s1_wrapper_v3.py','s1_run_v3.py','s1_finalize_v3.py']]
    f=frozen();assert H.load(V.DATA/'baseline_gate.json')['status']=='PASS'
    assert H.load(V.DATA/'01_hash_adapter_v2_audit.json')['B_HASH_REPRODUCTION']=='1052/1052'
    assert H.sha(H.CHECKPOINT)==CK
    unchanged=all(H.sha(i['absolute_path'])==i['sha256'] for i in f['inputs']);assert unchanged
    unit=W.unit_tests()
    H.dump(ROOT/'01_wrapper_unit_test.json',unit);H.dump(ROOT/'wrapper_v3_unit_test.json',unit)
    (ROOT/'wrapper_v3_unit_test.md').write_text('# No-WAM callback tests\n\nWRAPPER_UNIT_TEST = PASS\n\n'+ '\n'.join(f'- {k}: {v}' for k,v in unit['checks'].items())+'\n\nIncludes forced callback and observer exceptions, double/nested-install rejection, repeated cycles, and the actual new controller against CPU tensor/event stand-ins for all 300 video and 300 action events. No WAM weights, policy forward or simulator.\n')
    first=f['cases'][0]
    H.dump(ROOT/'00_v3_protocol.json',dict(version='S1_WRAPPER_FIX_V3',previous_attempt_status='S1_BLOCKED_INTEGRITY_V2',previous_block_reason='CALLBACK_LIFECYCLE_WRAPPER_BUG',previous_attempt_1_status='S1_BLOCKED_INTEGRITY',previous_attempt_1_forwards=0,previous_attempt_2_started=661,previous_attempt_2_completed=660,previous_scientific_node_cases=0,registry_sha256=H.sha(H.OUT/'protocol_freeze_s1.json'),checkpoint_sha256=CK,registry_counts=dict(states=132,trajectories=47,tasks=5,valid_donors=526,invalid_donors=2),smoke_recipient=first['state']['candidate_id'],spotcheck_donor=first['donor']['case_id'],selection='first valid entry in unchanged frozen registry ordering',gpu=3,dtype='bf16',threads=16,world_steps=10,action_steps=10,baseline_reuse='All 659 V2 exact comparisons reused; no whole-cohort baseline-only pass',cache_cost='132 recipient cache captures for same-value gate; 526 donor cache captures for node cell, including one spotcheck capture reused. One separate no-hook recipient spotcheck. No full cohort cache persistence required.',planned_policy_calls=1317,scientific_rules='Original F1/S1 32-position radial signed projection, per-case normalization, >1e-12 squared denominator, signed-dose means, original 10000 task→trajectory paired resampling; V2 actual original global min/max normalization, primary translation proven unchanged',labels='RC<=.20 & RF>=.80: REPLICATES; RC>=.60: FACTOR_DEPENDENT; otherwise PARTIAL',no_strict_grid_rerun=True,no_simulator_steps=True,unit_test=unit['WRAPPER_UNIT_TEST'],code=[H.info(H.WORK/n) for n in ['s1_run_v3.py','s1_wrapper_v3.py','s1_finalize_v3.py','s1_statistics.py','s1_hash_adapter_v2.py']],previous_immutable_files=previous))
    print('WRAPPER_UNIT_TEST = PASS; setup frozen; first recipient',first['state']['candidate_id'])

def run():
    import torch
    import run_experiment_a as A
    from capture import make_capture
    protocol=H.load(ROOT/'00_v3_protocol.json');assert H.load(ROOT/'01_wrapper_unit_test.json')['WRAPPER_UNIT_TEST']=='PASS'
    for i in protocol['code']:assert H.sha(i['absolute_path'])==i['sha256']
    assert not (ROOT/'progress.json').exists(),'Fresh process only; no implicit resume'
    f=frozen();states=list(dict.fromkeys(c['state']['candidate_id'] for c in f['cases']));by={s:[c for c in f['cases'] if c['state']['candidate_id']==s] for s in states}
    torch.set_num_threads(16);start=time.time();calls=[];started=0;stage='SMOKE';same=[];nodes=[];weight=None;stagepass={}
    def progress(status='RUNNING',**kw):H.dump(ROOT/'progress.json',dict(status=status,stage=stage,policy_calls_started=started,policy_calls_completed=len(calls),new_model_forward_passes=started,same_value_pass_count=len(same),scientific_node_cases=len(nodes),elapsed_seconds=time.time()-start,**kw))
    def infer(kind,cid,fn,*args):
        nonlocal started
        started+=1;progress(current_case=cid,current_kind=kind);t=time.time();out=fn(*args)
        calls.append(dict(index=started,kind=kind,case_id=cid,seconds=time.time()-t));H.csvwrite(ROOT/'policy_call_cost.csv',calls)
        return out
    def prep(c,side):
        d=c['donor'];obs=V.obsload(d[side+'_observation_path']);assert V.historical_hash(obs)==d[side+'_input_hash']
        return runner._prepared(obs,c['state']['instruction'])
    def action_check(a,b):
        aa=np.asarray(a);bb=np.asarray(b)
        result=dict(elementwise_exact=bool(np.array_equal(aa,bb)),max_abs_difference=float(np.max(np.abs(aa.astype(float)-bb.astype(float)))),historical_action_hash=hashlib.sha256(np.ascontiguousarray(bb).tobytes()).hexdigest(),new_action_hash=hashlib.sha256(np.ascontiguousarray(aa).tobytes()).hexdigest())
        assert result['elementwise_exact'],result
        return result
    def node(rec,seed,cache):
        ctl=W.CurrentNode(runner.model,cache)
        # A.infer guarantees uninstall in finally; guard restores exact saved callback identity.
        a,m=A.infer(runner,rec,rec,seed,ctl);summary=ctl.summary();m['node_integrity']=summary
        assert a.shape==(32,7)
        return a,m
    try:
        progress('LOADING');runner=make_capture('joint',3);meta=A.run_metadata(runner,3);weight=meta['loaded_weight_hash'];old=f['historical_metadata'][0]
        for k in ['gpu_name','dtype','loaded_weight_hash','git_head','action_timesteps','action_deltas','video_timesteps','video_deltas','attention_implementation','actual_action_denoising_steps_N','actual_video_denoising_steps_N']:assert meta[k]==old[k],k
        assert H.sha(H.CHECKPOINT)==CK
        H.dump(ROOT/'02_fresh_process_manifest.json',dict(pid=os.getpid(),metadata=meta,checkpoint_sha256=CK,model_weight_hash_after_load=weight,previous_baseline_provenance=H.info(V.DATA/'baseline_gate.json'),previous_baseline_csv=H.info(V.DATA/'02_baseline_reproduction.csv'),fresh_process=True))
        for sid in states:
            c=by[sid][0];r=c['state'];rec=prep(c,'recipient');seed=c['seed'];ra,rt,rm=infer('recipient_cache',sid,A.capture_source,runner,rec,seed)
            with np.load(c['old_actions']) as z:base=z['recipient_natural'].copy()
            capturecheck=action_check(ra.numpy(),base)
            ident,im=infer('same_value_current',sid,node,rec,seed,rt.cache);check=action_check(ident.numpy(),base)
            for k in ['initial_video_noise_hash','initial_action_noise_hash']:assert rm[k]==im[k]
            p=ROOT/'same_value'/sid;p.mkdir(parents=True)
            np.savez_compressed(p/'actions.npz',recipient=ra.numpy(),identity=ident.numpy())
            H.dump(p/'events.json',A.jsonable(dict(capture=capturecheck,identity=check,capture_run=rm,identity_run=im)))
            obs=V.obsload(c['donor']['recipient_observation_path'])
            same.append(dict(recipient_id=sid,task=r['task_id'],state_hash=H.sha(Path(c['donor']['recipient_observation_path']).parent/'recipient_state.npz'),historical_input_hash=c['donor']['recipient_input_hash'],structured_auxiliary_hash=V.structured_hash(obs),model_weight_hash=weight,hook_event_count=im['node_integrity']['observed_current_write_events'],hook_ordering_digest=im['node_integrity']['ordering_digest'],**check))
            H.csvwrite(ROOT/'04_same_value_all_recipients.csv',same)
            if sid==states[0]:
                after=A.model_weight_hash(runner.model);assert after==weight
                H.dump(ROOT/'03_same_value_smoke.json',dict(status='PASS',recipient=sid,identity=check,integrity=A.jsonable(im['node_integrity']),weight_before=weight,weight_after=after,passed=True))
                stagepass['smoke']='PASS';stage='SAME_VALUE'
            del rt;gc.collect();torch.cuda.empty_cache();print('SAME_VALUE',len(same),sid,flush=True)
        assert len(same)==132;stagepass['same_value']='132/132';stage='SPOTCHECK'
        c=f['cases'][0];rec=prep(c,'recipient');don=prep(c,'donor');seed=c['seed'];sid=c['state']['candidate_id'];cid=c['donor']['case_id']
        ra,rm=infer('native_recipient_spotcheck',sid,A.infer,runner,rec,rec,seed,None)
        da,dt,dm=infer('donor_cache_spotcheck',cid,A.capture_source,runner,don,seed)
        with np.load(c['old_actions']) as z:
            spot=[dict(kind='recipient',case_id=sid,**action_check(ra.numpy(),z['recipient_natural'])),dict(kind='donor',case_id=cid,**action_check(da.numpy(),z['donor_natural']))]
        H.csvwrite(ROOT/'05_baseline_spotcheck.csv',spot);stagepass['spotcheck']='PASS';stage='NODE_CELL'
        for i,c in enumerate(f['cases']):
            r=c['state'];d=c['donor'];sid=r['candidate_id'];cid=d['case_id'];seed=c['seed'];rec=prep(c,'recipient')
            if i:
                don=prep(c,'donor');da,dt,dm=infer('donor_cache',cid,A.capture_source,runner,don,seed)
            with np.load(c['old_actions']) as z:base=z['recipient_natural'].copy();donorbase=z['donor_natural'].copy()
            donorcheck=action_check(da.numpy(),donorbase)
            na,nm=infer('current_node',cid,node,rec,seed,dt.cache)
            for k in ['initial_video_noise_hash','initial_action_noise_hash']:assert nm[k]==dm[k]
            env=A.continuous_env_action(na,runner.processor);assert np.array_equal(env.reshape(32,7),V.denorm_actual(na.numpy()))
            p=ROOT/'node_cells'/cid;p.mkdir(parents=True)
            np.savez_compressed(p/'node_actions.npz',node_current=na.numpy(),recipient=base,donor=da.numpy(),node_env_continuous=env)
            integrity=nm['node_integrity'];assert integrity['donor_current_tensor_hash']==integrity['written_current_tensor_hash']
            H.dump(p/'result.json',A.jsonable(dict(case_id=cid,seed=seed,node_run=nm,donor_run=dm,donor_baseline_check=donorcheck,action_hash=A.tensor_sha256(na),no_future_clamp=True)))
            vec=lambda text:np.fromstring(text.strip('[]'),sep=' ')
            axis=vec(r['goal_position_m'])-vec(r['object_position_m']);axis[2]=0;axis/=np.linalg.norm(axis)
            response=(env.reshape(32,7)[:,:3].astype(float)-V.denorm_actual(base)[:,:3].astype(float))@axis
            nodes.append(dict(candidate_id=sid,task=r['task_id'],source_trajectory=r['trajectory_id'],recipient_id=sid,donor_id=cid,dose=d['signed_dose_cm'],sign=int(np.sign(d['signed_dose_cm'])),state_hash=H.sha(Path(d['recipient_observation_path']).parent/'recipient_state.npz'),donor_state_hash=H.sha(Path(d['donor_observation_path']).parent/'donor_state.npz'),recipient_input_hash=d['recipient_input_hash'],donor_input_hash=d['donor_input_hash'],hardware='RTX 4090 GPU3',checkpoint_hash=CK,model_weight_hash=weight,hook_ordering_digest=integrity['ordering_digest'],expected_current_write_count=300,observed_current_write_count=integrity['observed_current_write_events'],donor_current_tensor_hash=integrity['donor_current_tensor_hash'],written_current_tensor_hash=integrity['written_current_tensor_hash'],action_hash=A.tensor_sha256(na),denormalized_action=json.dumps(env.tolist()),radial_response_vector=json.dumps(response.tolist()),future_clamped=False))
            H.csvwrite(ROOT/'06_node_cell_per_case.csv',nodes)
            del dt;gc.collect();torch.cuda.empty_cache();print('NODE',len(nodes),cid,flush=True)
        assert len(nodes)==526
        after=A.model_weight_hash(runner.model);assert after==weight;assert H.sha(H.CHECKPOINT)==CK
        assert all(H.sha(i['absolute_path'])==i['sha256'] for i in f['inputs'])
        assert all(H.sha(i['absolute_path'])==i['sha256'] for i in protocol['previous_immutable_files'])
        H.dump(ROOT/'07_node_cell_integrity.json',dict(status='PASS',completed_cases=len(nodes),expected_cases=526,weight_before=weight,weight_after=after,checkpoint_unchanged=True,previous_attempts_unchanged=True,all_current_writes_exact=True,no_future_clamp=True,all_order_digests_identical=len(set(x['hook_ordering_digest'] for x in nodes))==1,all_132_identity_pass=True,baseline_spotcheck='PASS'))
        progress('COMPLETE',model_weight_hash_before=weight,model_weight_hash_after=after,gate_status=stagepass)
    except Exception as e:
        status={'SMOKE':'S1_BLOCKED_INTEGRITY_V3_SMOKE','SAME_VALUE':'S1_BLOCKED_INTEGRITY_V3_SAME_VALUE','SPOTCHECK':'S1_BLOCKED_DETERMINISM_V3','NODE_CELL':'S1_BLOCKED_INTEGRITY_V3_NODE_CELL'}[stage]
        progress(status,error=repr(e),traceback=traceback.format_exc(),gate_status=stagepass,model_weight_hash_before=weight)
        raise

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['setup','run']);args=ap.parse_args()
    if args.mode=='setup':setup()
    else:run()

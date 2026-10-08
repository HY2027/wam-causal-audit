"""New-source phase2 donor construction and frozen six-cell validation."""
import argparse,json,time,traceback
from pathlib import Path
import numpy as np
from phase2_native_validation import OUT,WORK,DOSES,verify
from phase2_local_future import sha,load,dump,table

def freeze():
    verify();rows=[]
    for task in [0,1]:
        base=OUT/f'acquisition/task{task}';done=load(base/'completion.json')
        if done['status']!='BOUNDED_ACQUISITION_COMPLETE':continue
        rows.extend(load(p) for p in sorted((base/'sources').glob('*/selected_state.json')))
    if not rows:raise RuntimeError('NO_ACCEPTED_INDEPENDENT_SOURCE')
    path=OUT/'formal_freeze_manifest.json'
    if path.exists():raise RuntimeError('already frozen')
    dump(OUT/'formal_sources.json',rows);table(OUT/'trajectory_registry.csv',rows)
    dump(path,dict(created_unix=time.time(),identity='NEW_INDEPENDENT_TRAJECTORY_VALIDATION_OF_EXISTING_FINDING',source_n=len(rows),source_registry_sha256=sha(OUT/'formal_sources.json'),code_sha256=sha(__file__),statistics_code_sha256=sha(WORK/'phase2_new_statistics.py'),acquisition_manifest_sha256=sha(OUT/'acquisition_freeze_manifest.json'),analysis='fixed-task trajectory bootstrap10000 seed20260910; paired eB-eW/eV-eW; no p-values; no equivalence criterion',state_inputs={r['state_path']:sha(r['state_path']) for r in rows},observation_inputs={r['observation_path']:sha(r['observation_path']) for r in rows}))

def run(task,gpu):
    import torch
    import phase2_native_launcher as L
    L.prepare_imports()
    import run_experiment_a as A
    import confirmatory_common as Q
    import generate_group1_donors as G
    f=load(OUT/'formal_freeze_manifest.json');assert f['code_sha256']==sha(__file__);assert f['source_registry_sha256']==sha(OUT/'formal_sources.json')
    for section in ['state_inputs','observation_inputs']:
        for p,h in f[section].items():assert sha(p)==h,p
    root=OUT/f'formal/task{task}';root.mkdir(parents=True,exist_ok=True);ds=[];checks=[];effects=[];events=[];cost=[]
    class PController(A.QKVController):
        def install(self):
            super().install();previous=self.model.mot._build_expert_attention_io;self.P={};self.counts={'video':0,'action':0}
            def wrapped(expert,block,*args,**kwargs):
                result=previous(expert,block,*args,**kwargs);mod,layer=self.block_map[id(block)];self.counts[mod]+=1
                if mod=='video':
                    key=(self.step,layer)
                    if key in self.P:raise RuntimeError('duplicate consumer event')
                    self.P[key]={k:result[i].detach().clone() for k,i in [('k',1),('v',2)]}
                return result
            self.model.mot._build_expert_attention_io=wrapped
    def timed(case,stage,fn):
        s=time.time();v=fn();cost.append(dict(case_id=case,stage=stage,seconds=time.time()-s));table(root/'compute_cost.csv',cost);return v
    def check(case,name,ok,error=0):
        checks.append(dict(case_id=case,check=name,passed=bool(ok),max_abs_error=float(error)));table(root/'intervention_integrity_checks.csv',checks)
        if not ok:raise RuntimeError('FORMAL_INTEGRITY_FAILED:'+name)
    torch.set_num_threads(16);runner=timed('LOAD','model_load',lambda:A.make_capture('joint',gpu));weight=A.model_weight_hash(runner.model)
    try:
        for row in [r for r in load(OUT/'formal_sources.json') if r['task_id']==task]:
            sid=row['candidate_id'];directory=root/'states'/sid;directory.mkdir(parents=True,exist_ok=True)
            env,obs,_=Q.restore(row);env.close();prepared=runner._prepared(obs,row['instruction']);seed=row['policy_seed'];base,rt,_=timed(sid,'recipient_capture',lambda:A.capture_source(runner,prepared,seed))
            with np.load(Path(row['state_path']).parent/'native_boundary_action.npz') as z:check(sid,'native_boundary_action_exact',np.array_equal(base.numpy(),z['action']),A.max_abs(base.numpy(),z['action']))
            R0,_=timed(sid,'R0_identity',lambda:A.strict_infer(runner,prepared,seed,rt.cache,rt.cache));check(sid,'same_value_exact',torch.equal(R0,base),A.max_abs(R0,base))
            axis=np.asarray(row['object_position_m'])-np.asarray(row['eef_position_m']);axis/=np.linalg.norm(axis)
            for dose in DOSES:
                case=sid+f'__F1__dose{dose:+g}';env=None
                try:
                    env,source_obs,_=Q.restore(row)
                    # Only input loading is adapted; original IK/physical QC is unmodified.
                    old_loader=G.load_phase_state
                    def accepted_loader(target_env,target_row):
                        assert target_env is env and target_row is row
                        check(case,'donor_source_input_identity',Q.observation_hash(source_obs)==row['observation_content_sha256'])
                        return target_env.env.sim,source_obs
                    G.load_phase_state=accepted_loader
                    try:dobs,qc,dstate=timed(case,'donor_geometry_render',lambda:G.construct(env,row,'F1_ROBOT_RADIAL_PROGRESS',dose,'STANDARD'))
                    finally:G.load_phase_state=old_loader
                finally:
                    if env is not None:env.close()
                qcpath=directory/f'dose{dose:+g}_qc.json';A.atomic_json(qcpath,qc);ds.append(dict(case_id=case,task_id=task,trajectory_id=row['trajectory_id'],signed_dose=dose,valid=qc['donor_valid'],exclusions=json.dumps(qc['invalid_reasons']),qc_path=str(qcpath),qc_sha256=sha(qcpath)));table(root/'donor_registry.csv',ds)
                if not qc['donor_valid']:continue
                A.atomic_npz(directory/f'dose{dose:+g}_donor_observation.npz',**dobs);A.atomic_npz(directory/f'dose{dose:+g}_donor_state.npz',**dstate)
                dp=runner._prepared(dobs,row['instruction']);da,dt,_=timed(case,'donor_capture',lambda:A.capture_source(runner,dp,seed))
                def propagate():
                    p=PController(runner.model);p.patch_cache=dt.cache;p.patch_layers=set(range(30));p.patch_temporal_groups=(0,);pa,run=A.infer(runner,prepared,prepared,seed,p);return pa,p
                pa,p=timed(case,'P_generation',propagate);check(case,'P_event_coverage',p.counts=={'video':300,'action':300} and set(p.P)==set(rt.cache))
                values={'R0':R0.numpy(),'native_recipient':base.numpy(),'native_donor':da.numpy(),'legacy_current':pa.numpy()}
                caches={'B':rt.cache,'D':dt.cache,'Pall':p.P,'PW':{key:(p.P[key] if key[1]>=25 else rt.cache[key]) for key in rt.cache},'PV':{key:(p.P[key] if key[1]<5 else rt.cache[key]) for key in rt.cache}}
                for config,cache in caches.items():
                    act,run=timed(case,config,lambda:A.strict_infer(runner,prepared,seed,dt.cache,cache));values[config]=act.numpy()
                    check(case,config+'_all_consumers_exact',run['hook']['all_current_consumed_values_exact'] and run['hook']['all_future_consumed_values_exact'] and run['hook']['video_calls']==300 and run['hook']['action_calls']==300)
                check(case,'Pall_reproduces_P_generating_action',np.array_equal(values['Pall'],values['legacy_current']),A.max_abs(values['Pall'],values['legacy_current']))
                for (step,layer),v in p.P.items():
                    for c in ['k','v']:
                        tpg=v[c].shape[1]//3;events.append(dict(case_id=case,step=step,layer=layer,component=c,current_donor_hash=A.tensor_sha256(dt.cache[(step,layer)][c][:,:tpg]),P_future_hash=A.tensor_sha256(v[c][:,tpg:]),recipient_future_hash=A.tensor_sha256(rt.cache[(step,layer)][c][:,tpg:]),natural_donor_future_hash=A.tensor_sha256(dt.cache[(step,layer)][c][:,tpg:]),P_same_value=torch.equal(v[c][:,tpg:],rt.cache[(step,layer)][c][:,tpg:]),W=layer>=25,V=layer<5))
                envvalues={k:A.continuous_env_action(torch.from_numpy(v),runner.processor) for k,v in values.items()};rad={k:v[:,:3]@axis for k,v in envvalues.items()};errs={k:float(np.sqrt(np.mean((rad[k]-rad['Pall'])**2))) for k in ['B','PW','PV']}
                effects.append(dict(case_id=case,task_id=task,trajectory_id=row['trajectory_id'],signed_dose=dose,eB=errs['B'],eW=errs['PW'],eV=errs['PV'],delta_restore=errs['B']-errs['PW'],delta_location=errs['PV']-errs['PW'],Pall_D_RMS=float(np.sqrt(np.mean((rad['Pall']-rad['D'])**2))),uP_RMS=float(np.sqrt(np.mean((rad['Pall']-rad['B'])**2))),uW_RMS=float(np.sqrt(np.mean((rad['PW']-rad['B'])**2))),uV_RMS=float(np.sqrt(np.mean((rad['PV']-rad['B'])**2)))))
                timed(case,'action_write',lambda:A.atomic_npz(directory/f'dose{dose:+g}_actions.npz',**values,**{k+'__env':v for k,v in envvalues.items()},**{k+'__radial':v for k,v in rad.items()}))
                table(root/'restoration_effects.csv',effects);table(root/'consumer_hashes.csv',events);dump(root/'progress.json',dict(completed_valid_donors=len(effects),attempted_donors=len(ds),source=sid));del dt,p,caches;torch.cuda.empty_cache()
            del rt;torch.cuda.empty_cache()
        check('ALL','weight_unchanged',weight==A.model_weight_hash(runner.model));dump(root/'completion.json',dict(status='FORMAL_TASK_COMPLETE',task_id=task,valid_donors=len(effects),intended_donors=len(ds)))
    except Exception:dump(root/'failure.json',dict(status='FORMAL_TASK_BLOCKED',traceback=traceback.format_exc()));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['freeze','run']);p.add_argument('--task',type=int);p.add_argument('--gpu',type=int,default=0);a=p.parse_args();freeze() if a.stage=='freeze' else run(a.task,a.gpu)

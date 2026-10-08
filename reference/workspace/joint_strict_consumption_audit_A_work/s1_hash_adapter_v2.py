"""S1_HASH_ADAPTER_V2. Historical files are read-only. No simulator creation."""
import argparse, ast, csv, gc, hashlib, json, os, sys, time, traceback
from pathlib import Path
import numpy as np
import s1_followup as H

PREVIOUS=H.OUT
OUT=PREVIOUS/'S1_HASH_ADAPTER_V2'
DATA=OUT/'s1_f3g_attribution'
BHASH_SOURCE=H.WORK/'run_b_sync_f3g_matrix.py'
def historical_hash(obs):
 # Execute ONLY the original pure h() AST and original recipient-input expression.
 tree=ast.parse(BHASH_SOURCE.read_text())
 hnode=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='h')
 one=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='one')
 expr=None
 for n in ast.walk(one):
  if isinstance(n,ast.Dict):
   for k,v in zip(n.keys,n.values):
    if isinstance(k,ast.Constant) and k.value=='recipient_input_hash':expr=v
 assert expr is not None
 scope={'np':np,'hashlib':hashlib,'rec':obs}
 exec(compile(ast.Module(body=[hnode],type_ignores=[]),str(BHASH_SOURCE),'exec'),scope)
 return eval(compile(ast.Expression(body=expr),str(BHASH_SOURCE),'eval'),scope)
def structured_hash(obs):
 h=hashlib.sha256()
 for k in sorted(obs):
  a=np.ascontiguousarray(obs[k]);h.update(k.encode());h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(memoryview(a.view(np.uint8)))
 return h.hexdigest()
def obsload(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def config():return H.load(OUT/'protocol_freeze_s1.json')
def denorm_actual(x):
 # Actual frozen LIBERO processor uses global min/max (libero_2cam.yaml:51).
 # The three translation channels are identical to the prior offline adapter.
 import torch
 d=H.load(H.STATS)['action']['default'];lo=torch.tensor(d['global_min'],dtype=torch.float32);hi=torch.tensor(d['global_max'],dtype=torch.float32)
 span=hi-lo;ignore=span<1e-4;span[ignore]=2.;scale=2./span;offset=-1.-scale*lo;offset[ignore]=-lo[ignore]
 y=((torch.as_tensor(np.array(x),dtype=torch.float32)-offset)/scale).numpy().reshape(32,7);y[:,-1]=-(y[:,-1]*2-1)
 return y
def audit():
 assert not OUT.exists(),'Never overwrite V2'
 f=H.load(PREVIOUS/'protocol_freeze_s1.json')
 failed=H.load(H.S1/'execution_progress.json');assert failed['status']=='S1_BLOCKED_INTEGRITY' and failed['new_model_forward_passes']==0
 prior=[H.info(p) for p in PREVIOUS.rglob('*') if p.is_file()]
 prior += [H.info(H.WORK/n) for n in ['s1_followup.py','s1_statistics.py','s1_hash_diagnosis.py']]
 unchanged=[dict(path=i['absolute_path'],match=H.sha(i['absolute_path'])==i['sha256']) for i in f['inputs']]
 rows=[];strict=[]
 for c in f['cases']:
  for side in ['recipient','donor']:
   d=c['donor'];p=d[side+'_observation_path'];obs=obsload(p);old=d[side+'_input_hash'];b=historical_hash(obs)
   rows.append(dict(case_id=d['case_id'],side=side,path=p,archived_historical_b_input_hash=old,historical_b_input_hash=b,structured_a_input_hash=structured_hash(obs),historical_equality=b==old))
  with np.load(c['old_actions']) as z:
   strict.append(dict(case_id=c['donor']['case_id'],keys=z.files,all_required=all(k in z for k in ['A00','A10','A01','A11','recipient_natural','donor_natural']),A00_exact=bool(np.array_equal(z['A00'],z['recipient_natural']))))
 counts=dict(states=len(f['states']),trajectories=len(set((r['task_id'],r['trajectory_id']) for r in f['states'])),tasks=len(set(r['task_id'] for r in f['states'])),valid_donors=len(f['cases']),invalid_donors=len(f['invalid_cases']))
 expected=dict(states=132,trajectories=47,tasks=5,valid_donors=526,invalid_donors=2)
 passed=counts==expected and len(rows)==1052 and all(r['historical_equality'] for r in rows) and all(i['match'] for i in unchanged) and all(r['all_required'] and r['A00_exact'] for r in strict)
 H.csvwrite(DATA/'historical_b_hash_unit_tests.csv',rows)
 H.dump(OUT/'previous_attempt_immutable_manifest.json',prior)
 H.dump(OUT/'protocol_freeze_s1.json',f)
 H.dump(DATA/'01_hash_adapter_v2_audit.json',dict(INPUT_HASH_GATE='PASS' if passed else 'FAIL',B_HASH_REPRODUCTION=f"{sum(r['historical_equality'] for r in rows)}/1052",counts=counts,unchanged_files=unchanged,strict_cells=strict,PREVIOUS_ATTEMPT_STATUS='S1_BLOCKED_INTEGRITY',PREVIOUS_ATTEMPT_NEW_FORWARD_PASSES=0,CORRECTION_TYPE='HASH_ADAPTER_ONLY',source=H.info(BHASH_SOURCE),code=H.info(__file__),native_baseline_reproduction='PENDING',new_model_forward_passes=0))
 (DATA/'01_hash_adapter_v2_audit.md').write_text('''# S1 hash adapter V2 — input audit

Previous attempt remains S1_BLOCKED_INTEGRITY / NOT_EVALUATED, zero forwards. This is not replication failure, factor dependence, model mismatch or input corruption.

The historical B function is extracted verbatim from the installed run_b_sync_f3g_matrix.py AST: h() and the recipient_input_hash expression in one(). No simulator module or one() is executed. Keys are sorted lexicographically; only values isinstance(np.ndarray) are retained; each is reshape(-1) in NumPy default C order; concatenate promotes to the common NumPy dtype (float64 in these inputs); ascontiguousarray and uint8 view supply native-byte-order bytes to hashlib.sha256(memoryview(...)). No key/dtype/shape prefix enters that historical hash. Archived arrays are all non-scalar arrays, so save/load did not change the inclusion set. No byte swapping is introduced.

A structured hash separately prefixes each key, dtype and shape then its bytes. It is recorded but never compared to B. Both fields are explicit in historical_b_hash_unit_tests.csv. Required historical equality is B recomputed == B archived.

'''+f'INPUT_HASH_GATE = {"PASS" if passed else "FAIL"}; B_HASH_REPRODUCTION = {sum(r["historical_equality"] for r in rows)}/1052. Counts: {counts}.\n'+'''
All scientific quantities, normalization, denominator and resampling follow the preceding frozen S1 files. The current authorization specifies otherwise→S1_PARTIAL, resolving the previously unspecified intermediate label. No numeric threshold is added. Baseline-only phase MUST finish across all 132 recipients and 526 donors before any intervention. Baseline mismatch branches to a new complete matched set on the same GPU/registry; historical strict actions are never paired in that branch. Unresolved native repeatability stops. Same-value and intervention integrity failures stop before statistics.

Cache production later may repeat native inference because full-cohort caches cannot all reside in memory; these calls count explicitly. All 30 layers × 10 denoising steps retained. No S2/S3.
''')
 H.dump(OUT/'v2_execution_freeze.json',dict(code=H.info(__file__),analysis_code=H.info(H.WORK/'s1_v2_finalize.py'),prior_protocol_sha256=H.sha(PREVIOUS/'protocol_freeze_s1.json'),v2_protocol_sha256=H.sha(OUT/'protocol_freeze_s1.json'),hash_adapter_source=H.info(BHASH_SOURCE),gpu=3,threads=16,baseline_order='all recipient/donor native baselines, no intervention; then branch and node',ordering='frozen registry order',labels='R_C<=.20 & R_F>=.80: REPLICATES; R_C>=.60: FACTOR_DEPENDENT; otherwise PARTIAL',previous_attempt_status='S1_BLOCKED_INTEGRITY'))
 print('INPUT_HASH_GATE',passed,counts,flush=True)
 if not passed:raise RuntimeError('S1_BLOCKED_B_HASH_REPRODUCTION')

def execute():
 import torch
 import run_experiment_a as A
 from capture import make_capture
 f=config();vf=H.load(OUT/'v2_execution_freeze_verified.json')
 assert H.load(DATA/'01_hash_adapter_v2_audit.json')['INPUT_HASH_GATE']=='PASS'
 assert H.sha(__file__)==vf['code']['sha256']
 assert H.sha(BHASH_SOURCE)==vf['hash_adapter_source']['sha256']
 assert not (DATA/'execution_progress.json').exists(),'No implicit resume/overwrite'
 torch.set_num_threads(16);count=0;calls=[];checks=[];stage='BASELINE';start=time.time()
 def progress(status='RUNNING',**kw):H.dump(DATA/'execution_progress.json',dict(status=status,stage=stage,new_model_forward_passes=count,elapsed_seconds=time.time()-start,**kw))
 def forward(name,case,fn,*args):
  nonlocal count
  count+=1;progress(current_call=name,current_case=case);t=time.time();value=fn(*args)
  calls.append(dict(call=count,stage=stage,condition=name,case_id=case,seconds=time.time()-t));H.csvwrite(DATA/'forward_calls.csv',calls)
  return value
 def compare(name,case,new,old,fatal=True):
  a=np.asarray(new);b=np.asarray(old);ok=bool(np.array_equal(a,b));err=float(np.max(np.abs(a.astype(float)-b.astype(float))))
  checks.append(dict(stage=stage,case_id=case,condition=name,elementwise_equal=ok,max_abs_difference=err,historical_action_hash=hashlib.sha256(np.ascontiguousarray(b).tobytes()).hexdigest(),reproduced_action_hash=hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest(),gpu=3))
  H.csvwrite(DATA/('02_baseline_reproduction.csv' if stage=='BASELINE' else 'intervention_integrity.csv'),[c for c in checks if c['stage']==stage])
  if fatal and not ok:raise RuntimeError(f'{name}: {case}: max_abs={err}')
  return ok
 def prepare(r,side,d):
  p=d[side+'_observation_path'];o=obsload(p)
  assert historical_hash(o)==d[side+'_input_hash']
  return runner._prepared(o,r['instruction'])
 class Node(A.QKVController):
  def __init__(self,model,cache):
   super().__init__(model);self.patch_cache=cache;self.patch_layers=set(range(30));self.patch_temporal_groups=(0,);self.all_events=[];self.ac=0
  def install(self):
   original=self.original;before={}
   def observe(expert,block,*args,**kw):
    out=original(expert,block,*args,**kw);before['out']=out;return out
   self.original=observe;super().install();patched=self.model.mot._build_expert_attention_io
   def verify(expert,block,*args,**kw):
    out=patched(expert,block,*args,**kw);raw=before['out'];mod,layer=self.block_map[id(block)]
    if mod=='action':self.ac+=1;assert all(a is b for a,b in zip(raw,out))
    else:
     key=(self.step,layer);target=self.patch_cache[key];assert out[1].shape[1]==294
     assert all(torch.equal(out[s][:,:98],target[k][:,:98]) and torch.equal(out[s][:,98:],raw[s][:,98:]) for k,s in [('k',1),('v',2)])
     self.all_events.append(dict(step=self.step,layer=layer,current_exact=True,future_manually_modified=False))
    return out
   self.model.mot._build_expert_attention_io=verify;self.original=original
 def node(rec,seed,cache):
  ctl=Node(runner.model,cache);a,m=A.infer(runner,rec,rec,seed,ctl)
  assert [(x['step'],x['layer']) for x in ctl.all_events]==[(s,l) for s in range(10) for l in range(30)] and ctl.ac==300
  m['node_events']=ctl.all_events;m['action_events']=ctl.ac;m['descendants_recomputed']='Original MoT video/action path every layer and every denoising step; no future cache writes'
  return a,m
 try:
  progress('LOADING');runner=make_capture('joint',3);meta=A.run_metadata(runner,3);ref=f['historical_metadata'][0]
  H.dump(DATA/'hardware_metadata.json',meta)
  for k in ['gpu_name','dtype','loaded_weight_hash','git_head','action_timesteps','action_deltas','video_timesteps','video_deltas','attention_implementation','actual_action_denoising_steps_N','actual_video_denoising_steps_N']:assert meta[k]==ref[k],k
  ck=next(i['sha256'] for i in f['inputs'] if i['absolute_path']==str(H.CHECKPOINT));assert H.sha(H.CHECKPOINT)==ck
  states=list(dict.fromkeys(c['state']['candidate_id'] for c in f['cases']));by={s:[c for c in f['cases'] if c['state']['candidate_id']==s] for s in states}
  # Entire-cohort baseline phase. No causal controllers.
  for sid in states:
   ds=by[sid];r=ds[0]['state'];seed=ds[0]['seed'];rec=prepare(r,'recipient',ds[0]['donor'])
   ra,rm=forward('native_recipient',sid,A.infer,runner,rec,rec,seed,None)
   dest=DATA/'baselines'/sid;dest.mkdir(parents=True)
   np.savez_compressed(dest/'recipient.npz',action=ra.numpy());H.dump(dest/'recipient.json',A.jsonable(rm))
   with np.load(ds[0]['old_actions']) as old:compare('recipient',sid,ra.numpy(),old['recipient_natural'],False)
   if sid==states[0]:
    rep,_=forward('native_repeat',sid,A.infer,runner,rec,rec,seed,None);compare('native_repeat',sid,rep.numpy(),ra.numpy())
   for c in ds:
    d=c['donor'];cid=d['case_id'];don=prepare(r,'donor',d);da,dm=forward('native_donor',cid,A.infer,runner,don,don,seed,None)
    with np.load(c['old_actions']) as old:compare('donor',cid,da.numpy(),old['donor_natural'],False)
    for key in ['initial_video_noise_hash','initial_action_noise_hash']:assert rm[key]==dm[key],key
    np.savez_compressed(dest/(cid+'.npz'),action=da.numpy());H.dump(dest/(cid+'.json'),A.jsonable(dm))
   print('BASELINE',sid,len(calls),flush=True);gc.collect();torch.cuda.empty_cache()
  assert A.model_weight_hash(runner.model)==meta['loaded_weight_hash']
  matched=all(c['elementwise_equal'] for c in checks)
  branch='S1_NEEDS_NODE_CELL' if matched else 'S1_RERUN_MATCHED_COHORT_REQUIRED'
  H.dump(DATA/'baseline_gate.json',dict(status='PASS' if matched else 'HISTORICAL_MISMATCH_NEW_MATCHED_REQUIRED',S1_EXECUTION_STATUS=branch,weight_before=meta['loaded_weight_hash'],weight_after=A.model_weight_hash(runner.model),checks=len(checks),mismatches=sum(not c['elementwise_equal'] for c in checks),forwards=count))
  (DATA/'02_baseline_reproduction.md').write_text(f'# Whole-cohort baseline gate\n\n{branch}; {sum(c["elementwise_equal"] for c in checks)}/{len(checks)} exact. All baselines were generated before any causal intervention. Weight before/after unchanged. GPU and action hashes, maximum errors: CSV.\n')
  stage='NODE';node_rows=[]
  for sid in states:
   ds=by[sid];r=ds[0]['state'];seed=ds[0]['seed'];rec=prepare(r,'recipient',ds[0]['donor'])
   ra,rt,rm=forward('recipient_cache',sid,A.capture_source,runner,rec,seed)
   with np.load(DATA/'baselines'/sid/'recipient.npz') as z:compare('recipient_repeat_cache',sid,ra.numpy(),z['action'])
   identity,im=forward('same_value_current',sid,node,rec,seed,rt.cache);compare('same_value_current',sid,identity.numpy(),ra.numpy())
   if not matched:a00,m00=forward('new_strict_A00',sid,A.strict_infer,runner,rec,seed,rt.cache,rt.cache);compare('A00',sid,a00.numpy(),ra.numpy())
   for c in ds:
    d=c['donor'];cid=d['case_id'];don=prepare(r,'donor',d);da,dt,dm=forward('donor_cache',cid,A.capture_source,runner,don,seed)
    with np.load(DATA/'baselines'/sid/(cid+'.npz')) as z:compare('donor_repeat_cache',cid,da.numpy(),z['action'])
    if not matched:
     a10,m10=forward('new_strict_A10',cid,A.strict_infer,runner,rec,seed,dt.cache,rt.cache)
     a01,m01=forward('new_strict_A01',cid,A.strict_infer,runner,rec,seed,rt.cache,dt.cache)
     a11,m11=forward('new_strict_A11',cid,A.strict_infer,runner,rec,seed,dt.cache,dt.cache)
     mp=DATA/'matched_strict'/cid;mp.mkdir(parents=True)
     np.savez_compressed(mp/'actions.npz',recipient_natural=ra.numpy(),donor_natural=da.numpy(),A00=a00.numpy(),A10=a10.numpy(),A01=a01.numpy(),A11=a11.numpy())
     H.dump(mp/'checks.json',A.jsonable([m00,m10,m01,m11]))
    na,nm=forward('current_node',cid,node,rec,seed,dt.cache)
    for key in ['initial_video_noise_hash','initial_action_noise_hash']:assert nm[key]==rm[key]==dm[key]
    dest=DATA/'node_cells'/cid;dest.mkdir(parents=True)
    env=A.continuous_env_action(na,runner.processor)
    assert np.array_equal(env.reshape(32,7),denorm_actual(na.numpy())),'Frozen processor denormalization mismatch'
    np.savez_compressed(dest/'node_actions.npz',node_current=na.numpy(),recipient=ra.numpy(),donor=da.numpy(),node_env_continuous=env)
    H.dump(dest/'result.json',A.jsonable(dict(case_id=cid,seed=seed,node_run=nm,recipient_run=rm,donor_run=dm,action_hash=A.tensor_sha256(na),context_same_value=torch.equal(rec['context'],don['context']),context_mask_same_value=torch.equal(rec['context_mask'],don['context_mask']),no_simulator_steps=True)))
    node_rows.append(dict(case_id=cid,state_id=sid,task=r['task_id'],donor_id=cid,seed=seed,node_action_hash=A.tensor_sha256(na),events=len(nm['node_events']),future_clamped=False,path=str(dest/'node_actions.npz')))
    H.csvwrite(DATA/'03_node_cell_per_case.csv',node_rows);del dt;gc.collect();torch.cuda.empty_cache()
   del rt;gc.collect();torch.cuda.empty_cache();print('NODE',sid,len(node_rows),flush=True)
  assert A.model_weight_hash(runner.model)==meta['loaded_weight_hash'];assert H.sha(H.CHECKPOINT)==ck
  assert all(H.sha(i['absolute_path'])==i['sha256'] for i in f['inputs'])
  assert all(H.sha(i['absolute_path'])==i['sha256'] for i in H.load(OUT/'previous_attempt_immutable_manifest.json'))
  progress('COMPLETE',S1_EXECUTION_STATUS=branch,completed_node_cases=len(node_rows),historical_files_unchanged=True,weight_hash_after=meta['loaded_weight_hash'])
 except Exception as e:
  status='S1_BLOCKED_DETERMINISM' if stage=='BASELINE' else 'S1_BLOCKED_INTEGRITY_V2'
  progress(status,error=repr(e),traceback=traceback.format_exc());raise

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['audit','execute']);a=p.parse_args()
 if a.mode=='audit':audit()
 else:execute()

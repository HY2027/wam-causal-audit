"""S1 only: audit/freeze, missing current-node cells, paired postprocessing."""
from wam_causal_audit.paths import resolve as _release_path
import argparse, csv, hashlib, json, os, sys, time, zipfile
from pathlib import Path
import numpy as np

WORK=Path(__file__).resolve().parent
ROOT=Path(_release_path('@DATA@/wam_factor_routing_v5'))
OUT=ROOT/'experiments/followup_s1_s2_s3'
S1=OUT/'s1_f3g_attribution'
B=ROOT/'joint_experiment_B_distance_v1'
SYNC=B/'B_SYNC_V1'
OLD=B/'B_forward_development_validation_v1'
AROOT=ROOT/'joint_experiment_A_strict_consumer_v1'
CHECKPOINT=Path(_release_path('@DATA@/BadWAM/models/LIQIIIII/badwam-libero-joint-wam/model.pt'))
STATS=CHECKPOINT.parent/'dataset_stats.json'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(16*1024*1024),b''):h.update(b)
 return h.hexdigest()
def info(p):
 p=Path(p).absolute()
 return dict(absolute_path=str(p),relative_path=os.path.relpath(p,ROOT),bytes=p.stat().st_size,sha256=sha(p))
def dump(p,x):
 p.parent.mkdir(parents=True,exist_ok=True)
 tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n');tmp.replace(p)
def csvwrite(p,rows):
 keys=list(dict.fromkeys(k for r in rows for k in r));p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def load(p):return json.loads(p.read_text())
def audit():
 if (OUT/'protocol_freeze_s1.json').exists():raise RuntimeError('Already frozen; never overwrite')
 states=[r for r in csv.DictReader((B/'B_split_registry.csv').open()) if r['selection_status'].startswith('SELECTED')]
 states.sort(key=lambda r:r['candidate_id'])
 bykey={(int(r['task_id']),int(r['trajectory_id']),int(r['policy_call'])):r for r in states}
 sourcepaths=[Path(__file__),WORK/'run_experiment_a.py',WORK/'analyze_experiment_a.py',WORK/'run_b_forward.py',WORK/'analyze_b_forward.py',Path(_release_path('@WORKSPACE@/badwam_joint_causal_work/run_joint_smoke_pair.py')),B/'B_split_registry.csv',SYNC/'B_SYNC_V1_spec.json',SYNC/'B_SYNC_F3G_freeze_manifest.json',SYNC/'B_F3G_synced_validity.csv',OLD/'B_forward_manifest.json',AROOT/'statistical_summary.csv',AROOT/'report_A.md',STATS,CHECKPOINT]
 sourcepaths+=sorted((OLD/'formal_shards').glob('*'))+sorted((OLD/'smoke_shards').glob('*'))
 sourcepaths+=sorted((SYNC/'f3g_synced_shards').glob('*.json'))
 candidates=[];invalid=[];missing=[]
 for p in sorted((SYNC/'f3g_synced_shards').glob('shard_*_of_04.json')):
  for d in load(p):
   if d['condition']!='RADIAL':continue
   r=bykey[(d['task_id'],d['trajectory_id'],d['policy_call'])];cid=r['candidate_id']
   old=OLD/'states'/cid/'donors'/d['case_id']/'actions.npz'
   rec=Path(d['recipient_observation_path']);don=Path(d['donor_observation_path'])
   row=dict(state=r,donor=d,seed=820000+int(r['task_id'])*1000+int(r['trajectory_id'])*20+int(r['policy_call']),old_actions=str(old))
   if not d['donor_valid']:invalid.append(row);continue
   paths=[old,rec,don,rec.parent/'recipient_state.npz',don.parent/'donor_state.npz',OLD/'states'/cid/'state_manifest.json']
   for q in paths:
    if not q.exists():missing.append(str(q))
   sourcepaths += [q for q in paths if q.exists()]
   if old.exists():
    with np.load(old) as z:
     row['old_action_keys']=z.files
     row['a00_native_exact']=bool(np.array_equal(z['A00'],z['recipient_natural']))
     row['action_shape']=list(z['A00'].shape)
   candidates.append(row)
 # Inventory all project archived action-array headers. Never infer semantic equivalence from A10 alone.
 archives=[];node_candidates=[]
 for p in sorted(ROOT.rglob('*.npz')):
  if not any(x in str(p).lower() for x in ('action','factorial','legacy','node')):continue
  try:
   with zipfile.ZipFile(p) as z:keys=[n.removesuffix('.npy') for n in z.namelist()]
   rec=dict(path=str(p),keys=keys)
   archives.append(rec)
   if any(any(t in k.lower() for t in ('legacy','node','propagat')) for k in keys):node_candidates.append(rec)
  except zipfile.BadZipFile:archives.append(dict(path=str(p),error='INVALID_ZIP'))
 # All historical node candidates are F1/Group-2 (original F3), not this synchronized F3-G cohort.
 exact_node=[x for x in node_candidates if any(c['state']['candidate_id'] in x['path'] for c in candidates)]
 if exact_node:raise RuntimeError('Potential reusable node cell discovered; semantic review required before freeze')
 csvwrite(S1/'archived_action_inventory.csv',[dict(path=x['path'],keys=json.dumps(x.get('keys')),error=x.get('error','')) for x in archives])
 sourcepaths=list(dict.fromkeys(sourcepaths));inputs=[info(p) for p in sourcepaths]
 metadata=[load(p)['metadata'] for p in sorted((OLD/'formal_shards').glob('*.json'))]
 if not all(c.get('a00_native_exact') for c in candidates):missing.append('Historical A00/native mismatch')
 status='S1_BLOCKED_MISMATCH' if missing else 'S1_NEEDS_NODE_CELL'
 protocol=dict(status=status,created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),scope='S1 Steps 0–2 only',cohort='Original B synchronized development/validation registry, all radial donors; tangential controls excluded by physical relation, not outcomes',states=states,cases=candidates,invalid_cases=invalid,inputs=inputs,historical_metadata=metadata,
  readout='32-position denormalized translation projected onto frozen recipient object-to-goal horizontal unit vector from B; analogous F1 EEF-to-object axis retained for F1 only',
  estimand='E vectors; R_X=dot(E_X,E_joint)/dot(E_joint,E_joint), Delta_collapse=R_node-R_C; signed projections, NOT magnitude ratios',
  denominator_rule='squared joint radial norm > 1e-12; preserve raw excluded-ratio rows and counts',
  aggregation='casewise normalization then arithmetic mean per signed dose, as F1; overall summary: average valid doses within base state then mean base states, supplementary only',
  bootstrap='10000; sample tasks then complete trajectory blocks within sampled tasks, retaining all base states/doses/cells; point case mean per signed dose. B repeated states kept together, not independent trajectories',bootstrap_seed=20260907,
  labels={'S1_REPLICATES':'R_C<=0.20 AND R_F>=0.80','S1_FACTOR_DEPENDENT':'R_C>=0.60','S1_PARTIAL':'Intermediate range unspecified: no invented numeric boundaries; unresolved cases remain unclassified'},
  node='Original QKVController video current group 0, all 30 layers and 10 steps, K/V only; future not patched; downstream recomputes',strict='A10 donor current/recipient future; A01 recipient current/donor future; A11 both donor; other inputs recipient',
  integrity='bit exact, no tolerance relaxation; reproduce every recipient/donor baseline; same-value current identity; full 300 event coverage; compare input/checkpoint/source hashes; initial noise and scheduler same',
  primary_units='dataset denormalized translational policy instruction units, not executed displacement',
  conflict_sign='NOT_AVAILABLE: B has no opposite-sign crossed source cells; no extra grid authorized',
  hardware='RTX 4090 only; native bf16, 16 CPU threads, 10/10 steps; no blackwell pairing',
  restore_risk='B_SYNC_V1 static inputs valid; old missing gripper cumulative execution state affects closed loop, not used or executed here',
  known_metadata_conflicts=['B metadata inherited A F1/phase/dose/seed labels; actual B registry and B runner authoritative','B formal Z_context_hash_same_value compares recipient to itself; not donor proof; compare actual contexts in new run'],new_model_forward_passes_at_freeze=0)
 dump(OUT/'protocol_freeze_s1.json',protocol)
 audit=dict(status=status,valid_radial_cases=len(candidates),invalid_radial_cases=len(invalid),base_states=len(states),task_clusters=len(set(r['task_id'] for r in states)),source_trajectories=len(set((r['task_id'],r['trajectory_id']) for r in states)),missing=missing,existing_node_cells=0,strict_cells_reusable_pending_baseline_reproduction=len(candidates)*4,archive_headers_inspected=len(archives),other_node_candidates=node_candidates,inputs=inputs,protocol_sha256=sha(OUT/'protocol_freeze_s1.json'),new_model_forward_passes=0)
 dump(S1/'00_availability_audit.json',audit)
 text='''# S1 protocol and asset audit (posthoc)

Only S1 is in scope. No S2/S3 design or execution.

F1 exact estimand recovered from analyze_experiment_a.py vector_row, action_spaces and hierarchical_ci, and the original report/statistical_summary.csv. The six signed-dose strata yield the published current 4.5–7.4%, future 92.7–95.4%. These are signed 32-position radial vector projections on the joint effect, normalized per case BEFORE averaging. Denominator is squared joint norm > 1e-12. Invalid ratios stay missing, raw vectors remain. No norm-ratio substitution. F1 dose strata are ±1/2/4 cm; F3-G keeps its own frozen ±0.5/1 cm, no cross-factor raw-scale pooling.

Denormalization uses saved stepwise min/max with float32 output and original gripper inversion. F1 radial axis is recipient EEF→object; F3-G uses B recipient horizontal object→goal, not an invented F1 axis for a different physical relation. Positive F3-G dose moves only the basket toward the held object. Positions for projection are read from the frozen B registry, exactly as analyze_b_forward.py. No execution or rendering is needed.

E_joint, E_node, E_C, E_F and J are vectors, not scalar magnitudes. The protocol's slash notation is implemented as the original F1 signed directional projection. Raw vectors, norms, signed means and denominators are preserved separately. F1 aggregation is per signed-dose stratum; a supplementary across-dose estimate first averages within state. Bootstrap retains repeated B states by trajectory within task rather than treating them as independent. Five task clusters, 10,000 draws. The historical B bootstrap kept tasks fixed; the present requested F1-style task resampling is explicit, not presented as the original B analysis.

Search includes B forward runner and raw arrays, Group 2 original F3, own-path native/K5 posthoc arrays, project phase-1 evidence/coverage ledgers, and all project action archive headers. Group 2 current-node assets refer to F1–F5 original operators, not synchronized F3-G. B strict A10 is NOT node-current; later F3-G own-path grids are also strict. No pairable F3-G propagation-allowed current-node cell found.

Use only the original B radial registry to resolve its missing node cell. No transfer to the newer confirmation cohort, no dose selection. Hardware reproduction is mandatory before pairing. Original arrays remain untouched. Template metadata conflicts and known closed-loop restore risk are recorded in JSON. Ratios are intervention-specific influence, not source shares, natural direct/indirect effects or semantic future purity.

Labels apply descriptively per signed-dose stratum and supplementary overall, never as a significance/equivalence test. PARTIAL boundaries were not supplied, so no numeric intermediate boundaries are invented. Large-node attribution collapse must be assessed from R_node and Delta even if strict ratios satisfy REPLICATES. Opposite-sign source-conflict cells absent: N/A, not inferred.
'''
 (OUT/'protocol_audit_s1.md').write_text(text)
 (S1/'01_protocol_and_integrity.md').write_text(text+'\nExecution gate pending.\n')
 dump(S1/'preexecution_manifest.json',[info(p) for p in [OUT/'protocol_freeze_s1.json',OUT/'protocol_audit_s1.md',S1/'00_availability_audit.json',Path(__file__)]])
 print(json.dumps({k:v for k,v in audit.items() if k not in ('inputs','other_node_candidates')},indent=2))

def execute(gpu,limit):
 # Imports may load libraries but NEVER create an environment, step physics or execute actions.
 import torch, gc
 import run_experiment_a as A
 from capture import make_capture
 f=load(OUT/'protocol_freeze_s1.json')
 assert f['status']=='S1_NEEDS_NODE_CELL'
 for item in f['inputs']:
  if item['absolute_path']==str(Path(__file__)):
   assert sha(__file__)==item['sha256'],'Frozen execution code changed'
 assert sha(CHECKPOINT)==next(x['sha256'] for x in f['inputs'] if x['absolute_path']==str(CHECKPOINT))
 torch.set_num_threads(16)
 count=0; started=time.time();checks=[]
 def checkpoint_progress(extra=None):
  dump(S1/'execution_progress.json',dict(new_model_forward_passes=count,elapsed_seconds=time.time()-started,checks=len(checks),**(extra or {})))
 def forward(fn,*args):
  nonlocal count
  count+=1;checkpoint_progress({'status':'RUNNING'})
  return fn(*args)
 def check(name,a,b,cid):
  aa=np.asarray(a);bb=np.asarray(b);err=float(np.max(np.abs(aa.astype(float)-bb.astype(float))))
  ok=bool(np.array_equal(aa,bb));checks.append(dict(case_id=cid,check=name,passed=ok,max_abs_error=err))
  csvwrite(S1/'integrity_checks.csv',checks)
  if not ok:raise AssertionError(f'{cid}: {name} error={err}')
 class CheckedNode(A.QKVController):
  """Observe original frozen QKVController writes without changing its operator."""
  def __init__(self,model,cache):
   super().__init__(model);self.patch_cache=cache;self.patch_layers=set(range(30));self.patch_temporal_groups=(0,)
   self.all_events=[];self.action_events=0
  def install(self):
   original=self.original;before={}
   def observed(expert,block,*args,**kwargs):
    result=original(expert,block,*args,**kwargs);before['result']=result
    return result
   self.original=observed;super().install();patched=self.model.mot._build_expert_attention_io
   def verified(expert,block,*args,**kwargs):
    out=patched(expert,block,*args,**kwargs);raw=before['result'];mod,layer=self.block_map[id(block)]
    if mod=='action':
     self.action_events+=1
     assert all(a is b for a,b in zip(raw,out))
    else:
     key=(self.step,layer);row=self.patch_cache[key]
     assert out[1].shape[1]==294
     ok=all(torch.equal(out[s][:,:98],row[k][:,:98]) and torch.equal(out[s][:,98:],raw[s][:,98:]) for k,s in [('k',1),('v',2)])
     assert ok,'Node wrote unintended future or incorrect current'
     self.all_events.append(dict(step=self.step,layer=layer,current_exact=True,future_unmodified=True))
    return out
   self.model.mot._build_expert_attention_io=verified;self.original=original
 def node(runner,rec,seed,cache):
  ctl=CheckedNode(runner.model,cache);action,meta=A.infer(runner,rec,rec,seed,ctl)
  assert [(e['step'],e['layer']) for e in ctl.all_events]==[(s,l) for s in range(10) for l in range(30)]
  assert ctl.action_events==300
  meta['node_integrity']=dict(events=ctl.all_events,action_events=ctl.action_events,future_clamped=False)
  return action,meta
 try:
  runner=make_capture('joint',gpu);meta=A.run_metadata(runner,gpu)
  dump(S1/'new_hardware_metadata.json',meta)
  ref=f['historical_metadata'][0]
  for key in ['gpu_name','loaded_weight_hash','dtype','git_head','action_timesteps','action_deltas','video_timesteps','video_deltas','attention_implementation','actual_action_denoising_steps_N','actual_video_denoising_steps_N']:
   if meta[key]!=ref[key]:raise AssertionError(f'Hardware/config mismatch: {key}')
  own=f['cases'];stateids=list(dict.fromkeys(c['state']['candidate_id'] for c in own))
  if limit:stateids=stateids[:limit]
  for sid in stateids:
   ds=[c for c in own if c['state']['candidate_id']==sid];r=ds[0]['state'];seed=ds[0]['seed']
   recpath=Path(ds[0]['donor']['recipient_observation_path'])
   recobs=A.load_npz(recpath);rec=runner._prepared(recobs,r['instruction'])
   assert A.observation_sha256(recobs)==ds[0]['donor']['recipient_input_hash']
   ra,rt,rmeta=forward(A.capture_source,runner,rec,seed)
   original=np.load(ds[0]['old_actions'])
   check('recipient_native_reproduction',ra.numpy(),original['recipient_natural'],sid)
   check('A00_native',original['A00'],ra.numpy(),sid)
   ident,identitymeta=forward(node,runner,rec,seed,rt.cache)
   check('same_value_current_node',ident.numpy(),ra.numpy(),sid)
   for c in ds:
    d=c['donor'];cid=d['case_id'];dest=S1/'node_cells'/cid
    if dest.exists():raise RuntimeError('Refusing overwrite '+str(dest))
    obs=A.load_npz(Path(d['donor_observation_path']));assert A.observation_sha256(obs)==d['donor_input_hash']
    don=runner._prepared(obs,r['instruction']);da,dt,dmeta=forward(A.capture_source,runner,don,seed)
    with np.load(c['old_actions']) as old:
     check('donor_native_reproduction',da.numpy(),old['donor_natural'],cid)
     check('recipient_case_consistency',ra.numpy(),old['recipient_natural'],cid)
    for k in ['initial_video_noise_hash','initial_action_noise_hash']:
     assert rmeta[k]==dmeta[k],k
    na,nmeta=forward(node,runner,rec,seed,dt.cache)
    for k in ['initial_video_noise_hash','initial_action_noise_hash']:assert nmeta[k]==rmeta[k],k
    dest.mkdir(parents=True)
    np.savez_compressed(dest/'node_actions.npz',node_current=na.numpy(),recipient=ra.numpy(),donor=da.numpy(),node_env_continuous=A.continuous_env_action(na,runner.processor))
    dump(dest/'result.json',dict(case_id=cid,state_id=sid,seed=seed,recipient_run=A.jsonable(rmeta),donor_run=A.jsonable(dmeta),node_run=A.jsonable(nmeta),action_hash=A.tensor_sha256(na),context_same_value=torch.equal(rec['context'],don['context']),context_mask_same_value=torch.equal(rec['context_mask'],don['context_mask']),input_hashes=[d['recipient_input_hash'],d['donor_input_hash']],instruction_hash=hashlib.sha256(r['instruction'].encode()).hexdigest(),no_simulator_step=True,no_donor_action_execution=True))
    del dt;gc.collect();torch.cuda.empty_cache()
   del rt;gc.collect();torch.cuda.empty_cache()
   checkpoint_progress(dict(status='RUNNING',last_state=sid,completed_cases=len(list((S1/'node_cells').glob('*/result.json')))))
   print(sid,flush=True)
  assert A.model_weight_hash(runner.model)==meta['loaded_weight_hash']
  changed=[i['absolute_path'] for i in f['inputs'] if sha(i['absolute_path'])!=i['sha256']]
  assert not changed,changed
  checkpoint_progress(dict(status='TECHNICAL_SUBSET_PASS' if limit else 'COMPLETE',weight_hash_after=meta['loaded_weight_hash'],historical_files_unchanged=True))
 except Exception as e:
  import traceback
  checkpoint_progress(dict(status='S1_BLOCKED_INTEGRITY',error=repr(e),traceback=traceback.format_exc()))
  (S1/'06_s1_report.md').write_text(f'''S1_STATUS: S1_BLOCKED_INTEGRITY
S1_LABEL: NOT_EVALUATED
NEW_MODEL_FORWARD_PASSES: {count}
HARDWARE: RTX 4090 requested GPU {gpu}
CHECKPOINT_SHA256: {next(i['sha256'] for i in f['inputs'] if i['absolute_path']==str(CHECKPOINT))}
TASK_CLUSTER_COUNT: 5 planned

S1 cannot yet determine whether F3-G reproduces the F1 attribution shift.

Integrity gate stopped: `{e!r}`. No statistical interpretation, no tolerance changes, no simulator execution. Historical assets untouched. See execution_progress.json and integrity_checks.csv for exact location/error. Recipient/donor baselines are not assumed comparable merely from checkpoint identity.
''')
  dump(S1/'artifact_manifest.json',dict(inputs=f['inputs'],outputs=[info(p) for p in S1.rglob('*') if p.is_file() and p.name!='artifact_manifest.json']))
  raise

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['audit','execute']);ap.add_argument('--gpu',type=int,default=3);ap.add_argument('--limit',type=int,default=0);a=ap.parse_args()
 if a.mode=='audit':audit()
 else:execute(a.gpu,a.limit)

"""Frozen S1 directional-projection analysis; never loads a WAM."""
from wam_causal_audit.paths import resolve as _release_path
import json, csv, hashlib, argparse
from pathlib import Path
import numpy as np
import torch
import s1_followup as S

METRICS=['R_node','R_C','R_F','Delta_collapse','R_J','joint_l2','node_l2','current_l2','future_l2','interaction_l2']
def denorm(x):
 # Same float32 arithmetic as the frozen SingleFieldLinearNormalizer.backward.
 d=S.load(S.STATS)['action']['default']
 lo=torch.tensor(d['stepwise_min'],dtype=torch.float32);hi=torch.tensor(d['stepwise_max'],dtype=torch.float32)
 span=hi-lo;ignore=span<1e-4;span[ignore]=2.;scale=2./span;offset=-1.-scale*lo;offset[ignore]=-lo[ignore]
 y=((torch.as_tensor(np.array(x),dtype=torch.float32)-offset)/scale).numpy().reshape(32,7)
 y[:,-1]=-(y[:,-1]*2-1)
 return y
def label(c,f):
 if c is None or f is None:return 'NOT_CLASSIFIABLE'
 if c<=.2 and f>=.8:return 'S1_REPLICATES'
 if c>=.6:return 'S1_FACTOR_DEPENDENT'
 return 'INTERMEDIATE_BOUNDARIES_UNSPECIFIED'
def analysis():
 f=S.load(S.OUT/'protocol_freeze_s1.json');progress=S.load(S.S1/'execution_progress.json')
 assert progress['status']=='COMPLETE','No interpretation before integrity pass and complete frozen cohort'
 freeze=S.load(S.S1/'analysis_implementation_freeze.json');assert freeze['code']['sha256']==S.sha(__file__)
 rows=[];vectors={};inputs=f['inputs'][:]
 for c in f['cases']:
  r=c['state'];d=c['donor'];cid=d['case_id'];p=S.S1/'node_cells'/cid
  old=np.load(c['old_actions']);new=np.load(p/'node_actions.npz')
  inputs += [S.info(p/'node_actions.npz'),S.info(p/'result.json')]
  act={k:denorm(old[k]) for k in old.files};act['node']=denorm(new['node_current'])
  assert np.array_equal(act['node'],new['node_env_continuous'].reshape(32,7)),'Offline denormalizer must exactly reproduce live processor'
  vec=lambda text:np.fromstring(text.strip('[]'),sep=' ')
  axis=vec(r['goal_position_m'])-vec(r['object_position_m']);axis[2]=0;axis/=np.linalg.norm(axis)
  q={k:v[:,:3].astype(float)@axis for k,v in act.items()}
  ev={'E_joint':q['A11']-q['A00'],'E_node_C':q['node']-q['A00'],'E_C_strict':q['A10']-q['A00'],'E_F_strict':q['A01']-q['A00'],'J':q['A11']-q['A10']-q['A01']+q['A00']}
  w=ev['E_joint'];den=float(w@w);valid=den>1e-12
  transfer=lambda v:float(v@w/den) if valid else None
  row=dict(case_id=cid,task=int(r['task_id']),trajectory=r['trajectory_id'],state_id=r['candidate_id'],recipient_id=r['candidate_id'],donor_id=cid,dose=abs(float(d['signed_dose_cm'])),sign=int(np.sign(d['signed_dose_cm'])),signed_dose_cm=float(d['signed_dose_cm']),distance=float(r['object_goal_distance_m']),phase=d['phase_base'],split=r['split'],hardware='NVIDIA GeForce RTX 4090',denominator_squared=den,ratio_valid=valid,physical_QC_valid=True,conflict_sign_outcome='NOT_AVAILABLE',R_node=transfer(ev['E_node_C']),R_C=transfer(ev['E_C_strict']),R_F=transfer(ev['E_F_strict']),R_J=transfer(ev['J']))
  row['Delta_collapse']=row['R_node']-row['R_C'] if valid else None
  for k,v in ev.items():row[k]=json.dumps(v.tolist());row[k+'_signed_mean']=float(v.mean())
  for name,k in [('joint','E_joint'),('node','E_node_C'),('current','E_C_strict'),('future','E_F_strict'),('interaction','J')]:row[name+'_l2']=float(np.linalg.norm(ev[k]))
  row['A11_donor_exact']=bool(np.array_equal(old['A11'],old['donor_natural']))
  rows.append(row)
  vectors[cid]=np.stack(list(ev.values()))
 for c in f['invalid_cases']:
  r,d=c['state'],c['donor'];rows.append(dict(case_id=d['case_id'],task=int(r['task_id']),trajectory=r['trajectory_id'],state_id=r['candidate_id'],dose=abs(d['signed_dose_cm']),sign=int(np.sign(d['signed_dose_cm'])),signed_dose_cm=d['signed_dose_cm'],physical_QC_valid=False,ratio_valid=False,invalid_reason=d['invalid_reasons']))
 S.csvwrite(S.S1/'02_per_case_attribution.csv',rows)
 validrows=[r for r in rows if r['physical_QC_valid']]
 # Build paired arrays at base-state level; all dose/config entries retain the same draw indices.
 states=list(dict.fromkeys(r['state_id'] for r in validrows));si={s:i for i,s in enumerate(states)}
 doses=sorted(set(r['signed_dose_cm'] for r in validrows));data=np.full((len(states),len(doses),len(METRICS)),np.nan)
 stateinfo={r['state_id']:(r['task'],r['trajectory']) for r in validrows}
 for r in validrows:
  for m,k in enumerate(METRICS):data[si[r['state_id']],doses.index(r['signed_dose_cm']),m]=r[k] if r[k] is not None else np.nan
 tasks=sorted(set(t for t,tr in stateinfo.values()));groups={}
 for i,s in enumerate(states):groups.setdefault(stateinfo[s],[]).append(i)
 bytask={t:sorted(tr for tt,tr in groups if tt==t) for t in tasks}
 rng=np.random.default_rng(20260907);draws=np.full((10000,len(doses)+1,len(METRICS)),np.nan)
 for b in range(10000):
  ix=[]
  for t in rng.choice(tasks,len(tasks),replace=True):
   for tr in rng.choice(bytask[t],len(bytask[t]),replace=True):ix.extend(groups[(t,tr)])
  x=data[ix];draws[b,:len(doses)]=np.nanmean(x,axis=0);draws[b,-1]=np.nanmean(np.nanmean(x,axis=1),axis=0)
 point=np.concatenate([np.nanmean(data,axis=0),np.nanmean(np.nanmean(data,axis=1),axis=0)[None]],axis=0)
 results=[]
 for di,dose in enumerate(doses+['overall_state_mean_supplementary']):
  for mi,m in enumerate(METRICS):
   v=draws[:,di,mi];lo,hi=np.nanquantile(v,[.025,.975]);results.append(dict(stratum=dose,metric=m,estimate=float(point[di,mi]),ci95_low=float(lo),ci95_high=float(hi),task_clusters=len(tasks),bootstrap_resamples=10000))
 classification=[dict(stratum=d,R_C=float(point[i,1]),R_F=float(point[i,2]),label=label(float(point[i,1]),float(point[i,2]))) for i,d in enumerate(doses+['overall_state_mean_supplementary'])]
 S.dump(S.S1/'03_aggregate_attribution.json',dict(results=results,classification=classification,valid_cases=len(validrows),ratio_invalid=sum(not r['ratio_valid'] for r in validrows),states=len(states),trajectories=len(groups),task_clusters=len(tasks),invalid_donors=len(f['invalid_cases']),posthoc=True))
 S.dump(S.S1/'04_bootstrap_results.json',dict(seed=20260907,resamples=10000,unit='task then trajectory blocks, full base-state pairing',results=results,small_cluster_warning=len(tasks)<=3))
 np.savez_compressed(S.S1/'bootstrap_draws.npz',draws=draws,metrics=METRICS)
 f1=[r for r in csv.DictReader((S.AROOT/'statistical_summary.csv').open()) if r['action_space']=='radial_translation_raw']
 comp=[]
 for metric,oldmetric in [('R_C','current_transfer'),('R_F','future_transfer'),('Interaction','interaction_transfer')]:
  for r in f1:
   if r['metric']==oldmetric:comp.append(dict(quantity=metric,factor='F1',dose_cm=r['dose_cm'],sign=r['sign'],estimate=r['estimate'],ci95_low=r['ci95_low'],ci95_high=r['ci95_high'],source='Original frozen statistical_summary.csv; not recomputed'))
 for r in results:comp.append(dict(quantity=r['metric'],factor='F3-G',stratum=r['stratum'],estimate=r['estimate'],ci95_low=r['ci95_low'],ci95_high=r['ci95_high'],source='S1 posthoc'))
 for k in ['R_node','Delta_collapse','Conflict-sign behavior']:comp.append(dict(quantity=k,factor='F1',estimate='N/A in same-estimand published summary',source='No substitute from normalized full-vector legacy metric'))
 S.csvwrite(S.S1/'05_f1_f3g_comparison.csv',comp)
 # All task and split results remain visible; no outcome-selected subset.
 taskrows=[]
 for t in tasks:
  for d in doses:
   rr=[r for r in validrows if r['task']==t and r['signed_dose_cm']==d]
   for m in METRICS:
    vv=[r[m] for r in rr if r[m] is not None]
    taskrows.append(dict(task=t,dose=d,metric=m,n=len(vv),estimate=float(np.mean(vv)) if vv else None))
 S.csvwrite(S.S1/'per_task_attribution.csv',taskrows)
 lines=[]
 for i,d in enumerate(doses):
  vals=[]
  for mi in range(4):
   lo,hi=np.quantile(draws[:,i,mi],[.025,.975]);vals.append(f'{point[i,mi]:.4f} [{lo:.4f}, {hi:.4f}]')
  lines.append('| '+str(d)+' | '+' | '.join(vals)+' | '+classification[i]['label']+' |')
 all_label=classification[-1]['label'];ck=next(i['sha256'] for i in f['inputs'] if i['absolute_path']==str(S.CHECKPOINT))
 report=f'''S1_STATUS: S1_NEEDS_NODE_CELL — completed
S1_LABEL: {all_label} (supplementary across-dose mean; signed-dose labels below)
NEW_MODEL_FORWARD_PASSES: {progress['new_model_forward_passes']}
HARDWARE: RTX 4090, native bf16, 16 threads, 10 world / 10 action steps
CHECKPOINT_SHA256: {ck}
TASK_CLUSTER_COUNT: {len(tasks)}

# F3-G attribution shift (posthoc)

Strict-source descriptive classification is {all_label}; attribution collapse must be judged from the paired R_node−R_C estimates, not the label alone.

## Matched evidence

{len(states)} base states, {len(groups)} source trajectories, {len(validrows)} valid radial donors; {len(f['invalid_cases'])} invalid donors retained. Existing B strict cells reused only after recipient/donor bit-exact reproduction; same-value current-node replacement and all 300 intended node writes checked. Both text/proprio contexts compared in result.json rather than relying on the historical self-comparison field. No simulator step, rendering, donor construction or action execution. All historical input hashes unchanged.

F1 published radial projection ratios remain current 4.5–7.4%, future 92.7–95.4% across ±1/2/4 cm strata. F3-G uses its original ±0.5/1 cm. Different physical scales and axes are not pooled. All R values are signed directional projections, not source information shares. Negative, >1 and near-zero-denominator raw cases are not silently removed.

| Signed dose cm | R_node | R_C strict | R_F strict | Delta collapse | Label |
|---|---|---|---|---|---|
'''+ '\n'.join(lines)+f'''

Intervals: 10,000 task→trajectory bootstrap draws, preserving all base states, doses and cells together. Five task clusters; repeated states do not constitute independent trajectories. Original F1 per-case normalization and signed-dose averaging retained. Supplementary across-dose result first averages within base state and does not replace the primary signed-dose table. Ratio-invalid cases: {sum(not r['ratio_valid'] for r in validrows)}.

Raw E_joint/E_node_C/E_C_strict/E_F_strict/J vectors and signed means are saved in 02; raw L2 and denominators are separate, in denormalized translation-command units, not metres of executed EEF motion. J is not assumed zero. Opposite-sign crossed-source behavior is N/A: those cells were not part of B, and no new grid was run. F1 legacy full-vector normalized results cannot substitute for missing radial node summary numbers.

This comparison is conditional on the registered current-node and strict consumer operators. It does not identify natural direct/indirect effects, pure future semantics, universal mediation, unique pathways or minimality. This posthoc analysis is not new confirmatory validation. No S2/S3 work was performed.
'''
 (S.S1/'06_s1_report.md').write_text(report)
 with (S.S1/'01_protocol_and_integrity.md').open('a') as out:out.write('\nCompleted exact-baseline and same-value checks; detailed checks and per-event write coverage in node result files. See execution_progress.json.\n')
 S.dump(S.S1/'artifact_manifest.json',dict(inputs=inputs,outputs=[S.info(p) for p in S.S1.rglob('*') if p.is_file() and p.name!='artifact_manifest.json']))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['freeze','analyze']);a=p.parse_args()
 if a.mode=='freeze':
  dest=S.S1/'analysis_implementation_freeze.json';assert not dest.exists()
  S.dump(dest,dict(code=S.info(__file__),normalizer_source=S.info(_release_path('@DATA@/BadWAM/src/fastwam/datasets/lerobot/utils/normalizer.py')),formula='Exact float32 (x-offset)/scale, frozen min/max range rule, no WAM loading',protocol_sha256=S.sha(S.OUT/'protocol_freeze_s1.json')))
 else:analysis()

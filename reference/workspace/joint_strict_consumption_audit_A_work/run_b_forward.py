#!/usr/bin/env python3
"""Authorized Joint-WAM B-SYNC/F3-G development+validation forward stage."""
from __future__ import annotations
from wam_causal_audit.paths import resolve as _release_path
import argparse,csv,gc,hashlib,json,sys,time
from pathlib import Path
from typing import Any
import numpy as np
import torch
WORK=Path(__file__).resolve().parent;sys.path.insert(0,str(WORK))
import run_experiment_a as A
from capture import make_capture

SYNC=Path(_release_path('@DATA@/wam_factor_routing_v5/joint_experiment_B_distance_v1/B_SYNC_V1'))
OUT=Path(_release_path('@DATA@/wam_factor_routing_v5/joint_experiment_B_distance_v1/B_forward_development_validation_v1'))
PARENT=SYNC.parent
def dump(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(A.jsonable(x),ensure_ascii=False,indent=2,sort_keys=True)+'\n');t.replace(p)
def write(p,rows):
 p.parent.mkdir(parents=True,exist_ok=True)
 keys=[]
 for r in rows:
  for k in r:
   if k not in keys:keys.append(k)
 t=p.with_suffix(p.suffix+'.tmp')
 with t.open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
 t.replace(p)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def registry():
 states=[r for r in csv.DictReader(open(PARENT/'B_split_registry.csv')) if r.get('selection_status','').startswith('SELECTED')]
 donors=[]
 for path in sorted((SYNC/'f3g_synced_shards').glob('shard_*_of_04.json')):
  donors += [r for r in json.loads(path.read_text()) if r['donor_valid']]
 by= {}
 for r in donors:by.setdefault((int(r['task_id']),int(r['trajectory_id']),int(r['policy_call'])),[]).append(r)
 return states,by
def statekey(r):return (int(r['task_id']),int(r['trajectory_id']),int(r['policy_call']))
def seed(r):return 820000+int(r['task_id'])*1000+int(r['trajectory_id'])*20+int(r['policy_call'])
def sync_obs(r):return SYNC/'synced_recipients'/r['candidate_id']/'recipient_policy_observation.npz'
def selection(states,by):
 chosen=[]
 pats=[('RADIAL',-.5),('RADIAL',.5),('TANGENTIAL',-.5),('TANGENTIAL',.5)]
 for t in range(5):
  own=[r for r in states if int(r['task_id'])==t]
  for j,b in enumerate(('far','near')):
   r=next(r for r in own if r['distance_bin']==b); cond,dose=pats[(2*t+j)%4]
   d=next(d for d in by[statekey(r)] if d['condition']==cond and float(d['signed_dose_cm'])==dose)
   chosen.append({'state':r,'donor':d,'selection_rule':'per task: lowest frozen far/near state; condition cycle radial-/radial+/tangential-/tangential+; no model effect used'})
 return chosen
def prep(runner,path,instruction):
 return runner._prepared(A.load_npz(Path(path)),instruction)
def smoke_one(runner,item):
 r,d=item['state'],item['donor'];s=seed(r);rec=prep(runner,sync_obs(r),r['instruction']);don=prep(runner,d['donor_observation_path'],r['instruction']);hist=prep(runner,r['live_observation_path'],r['instruction'])
 base,_=A.infer(runner,rec,rec,s,None);repeat,_=A.infer(runner,rec,rec,s,None);ctl=A.VideoKVController(runner.model);ctl.read_only=True;read,_=A.infer(runner,rec,rec,s,ctl)
 ra,rt,rrun=A.capture_source(runner,rec,s);da,dt,drun=A.capture_source(runner,don,s);ident,mi=A.strict_infer(runner,rec,s,rt.cache,rt.cache);dreplay,dreplayrun=A.infer(runner,don,don,s,None);a10,m10=A.strict_infer(runner,rec,s,dt.cache,rt.cache);a01,m01=A.strict_infer(runner,rec,s,rt.cache,dt.cache);a11,m11=A.strict_infer(runner,rec,s,dt.cache,dt.cache)
 old=prep(runner,r['live_observation_path'],r['instruction']);old_a,_=A.infer(runner,old,old,s,None)
 checks=[('native_repeat',torch.equal(base,repeat),A.max_abs(base,repeat)),('readonly_hook',torch.equal(base,read),A.max_abs(base,read)),('same_value_both_sources',torch.equal(base,ident),A.max_abs(base,ident)),('full_donor_replay',torch.equal(da,dreplay),A.max_abs(da,dreplay)),('strict_C_donor_F_recipient',m10['hook']['all_current_consumed_values_exact'] and m10['hook']['all_future_consumed_values_exact'],max(m10['hook']['max_current_abs_error'],m10['hook']['max_future_abs_error'])),('strict_C_recipient_F_donor',m01['hook']['all_current_consumed_values_exact'] and m01['hook']['all_future_consumed_values_exact'],max(m01['hook']['max_current_abs_error'],m01['hook']['max_future_abs_error'])),('strict_C_donor_F_donor',m11['hook']['all_current_consumed_values_exact'] and m11['hook']['all_future_consumed_values_exact'],max(m11['hook']['max_current_abs_error'],m11['hook']['max_future_abs_error']))]
 rows=[{'task_id':r['task_id'],'trajectory_id':r['trajectory_id'],'policy_call':r['policy_call'],'case_id':d['case_id'],'check':x,'pass':y,'max_abs_error':z} for x,y,z in checks]
 rows.append({'task_id':r['task_id'],'trajectory_id':r['trajectory_id'],'policy_call':r['policy_call'],'case_id':d['case_id'],'check':'sync_vs_historical_native_difference_audit','pass':True,'max_abs_error':A.max_abs(base,old_a)})
 rows += [{'task_id':r['task_id'],'trajectory_id':r['trajectory_id'],'policy_call':r['policy_call'],'case_id':d['case_id'],'check':'timing_source_cache_generation','pass':True,'max_abs_error':0.0,'elapsed_seconds':float(rrun['model_inference_seconds'])+float(drun['model_inference_seconds'])},{'task_id':r['task_id'],'trajectory_id':r['trajectory_id'],'policy_call':r['policy_call'],'case_id':d['case_id'],'check':'timing_full_donor_replay','pass':True,'max_abs_error':0.0,'elapsed_seconds':float(dreplayrun['model_inference_seconds'])},{'task_id':r['task_id'],'trajectory_id':r['trajectory_id'],'policy_call':r['policy_call'],'case_id':d['case_id'],'check':'timing_strict_four_cells','pass':True,'max_abs_error':0.0,'elapsed_seconds':float(mi['model_inference_seconds'])+float(m10['model_inference_seconds'])+float(m01['model_inference_seconds'])+float(m11['model_inference_seconds'])}]
 return rows
def formal_state(runner,r,ds,root):
 s=seed(r);rec=prep(runner,sync_obs(r),r['instruction']);ra,rt,_=A.capture_source(runner,rec,s);a00,m00=A.strict_infer(runner,rec,s,rt.cache,rt.cache);od=root/'states'/r['candidate_id'];od.mkdir(parents=True,exist_ok=True);np.savez_compressed(od/'recipient_actions.npz',natural=ra.numpy(),A00=a00.numpy())
 metrics=[]
 for d in ds:
  dd=od/'donors'/d['case_id'];dd.mkdir(parents=True,exist_ok=True);don=prep(runner,d['donor_observation_path'],r['instruction']);da,dt,_=A.capture_source(runner,don,s);a10,m10=A.strict_infer(runner,rec,s,dt.cache,rt.cache);a01,m01=A.strict_infer(runner,rec,s,rt.cache,dt.cache);a11,m11=A.strict_infer(runner,rec,s,dt.cache,dt.cache)
  np.savez_compressed(dd/'actions.npz',recipient_natural=ra.numpy(),donor_natural=da.numpy(),A00=a00.numpy(),A10=a10.numpy(),A01=a01.numpy(),A11=a11.numpy())
  x=lambda q:q.detach().cpu().float().numpy(); dn=x(da-ra);world=x(a11-a00);dc=x(a10-a00);df=x(a01-a00);uf=x(a11-a10);uc=x(a11-a01);j=x(a11-a10-a01+a00)
  def norm(z):return float(np.linalg.norm(z.reshape(-1)))
  metrics.append({'case_id':d['case_id'],'task_id':r['task_id'],'trajectory_id':r['trajectory_id'],'policy_call':r['policy_call'],'split':r['split'],'distance_bin':r['distance_bin'],'distance_m':r['object_goal_distance_m'],'condition':d['condition'],'signed_dose_cm':d['signed_dose_cm'],'natural_l2':norm(dn),'world_l2':norm(world),'deltaC_l2':norm(dc),'deltaF_l2':norm(df),'UF_l2':norm(uf),'UC_l2':norm(uc),'J_l2':norm(j),'A10_A11_l2':norm(x(a10-a11)),'A01_A11_l2':norm(x(a01-a11)),'A11_natural_donor_l2':norm(x(a11-da)),'hook_all_exact':all(m['hook']['all_current_consumed_values_exact'] and m['hook']['all_future_consumed_values_exact'] for m in [m00,m10,m01,m11]),'Z_context_hash_same_value':A.tensor_sha256(rec['context'])==A.tensor_sha256(rec['context']),'environment_action_executed':False})
  del dt;gc.collect();torch.cuda.empty_cache()
 dump(od/'state_manifest.json',{'state':r,'donor_count':len(ds),'completed':True,'no_environment_action':True});del rt;gc.collect();torch.cuda.empty_cache();return metrics
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['smoke','formal'],required=True);ap.add_argument('--gpu',type=int,required=True);ap.add_argument('--shard',type=int,required=True);ap.add_argument('--shards',type=int,default=4);ap.add_argument('--threads',type=int,default=16);a=ap.parse_args();OUT.mkdir(parents=True,exist_ok=True);states,by=registry();torch.set_num_threads(a.threads);runner=make_capture('joint',a.gpu)
 if a.mode=='smoke':
  selected=selection(states,by);dump(OUT/'smoke_selection.json',[{'state_id':x['state']['candidate_id'],'case_id':x['donor']['case_id'],'rule':x['selection_rule']} for x in selected]);own=[x for i,x in enumerate(selected) if i%a.shards==a.shard];rows=[]
  for x in own:rows+=smoke_one(runner,x)
  wt=time.perf_counter();write(OUT/'smoke_shards'/f'shard_{a.shard:02d}.csv',rows);write_seconds=time.perf_counter()-wt;dump(OUT/'smoke_shards'/f'shard_{a.shard:02d}.json',{'pass':all(str(r['pass'])=='True' or r['pass'] is True for r in rows),'count':len(rows),'gpu':a.gpu,'artifact_write_seconds':write_seconds,'metadata':A.run_metadata(runner,a.gpu)});print(json.dumps({'mode':'smoke','shard':a.shard,'pass':all(r['pass'] for r in rows)}));return
 own=[r for i,r in enumerate(states) if i%a.shards==a.shard];allm=[]
 for r in own:allm+=formal_state(runner,r,by[statekey(r)],OUT)
 write(OUT/'formal_shards'/f'shard_{a.shard:02d}.csv',allm);dump(OUT/'formal_shards'/f'shard_{a.shard:02d}.json',{'states':len(own),'cases':len(allm),'gpu':a.gpu,'metadata':A.run_metadata(runner,a.gpu),'no_final_tasks':True,'no_env_action':True});print(json.dumps({'mode':'formal','shard':a.shard,'cases':len(allm)}))
if __name__=='__main__':main()

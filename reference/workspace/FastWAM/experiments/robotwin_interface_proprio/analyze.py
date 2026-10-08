#!/usr/bin/env python3
"""Paired task-equal analysis for RoboTwin interface x proprio outputs."""
from __future__ import annotations
import argparse,csv,hashlib,json,os
from pathlib import Path
import numpy as np

SEED=20260922; DRAWS=10000
CELLS={'Y00':'Y00__IZ1__PZ1','Y10':'Y10__IDplus__PZ1','Y01':'Y01__IZ1__PDplus','Y11':'Y11__IDplus__PDplus'}
def save(p,x):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n');os.replace(t,p)
def load_radial(p):return np.load(p)['radial_world'].astype(np.float64)
def mean(v):return float(np.mean(v))
def rms(v):return float(np.sqrt(np.mean(np.square(v))))
def task_equal(rows,key):
 return float(np.mean([np.mean([r[key] for r in rows if r['task_id']==t]) for t in sorted({r['task_id'] for r in rows})]))
def boot(rows,keys):
 rng=np.random.default_rng(SEED);tasks=sorted({r['task_id'] for r in rows});out={k:[] for k in keys}
 groups={t:[r for r in rows if r['task_id']==t] for t in tasks}
 for _ in range(DRAWS):
  draw=[]
  for t in tasks:
   g=groups[t];draw += [g[i] for i in rng.integers(0,len(g),len(g))]
  for k in keys:out[k].append(task_equal(draw,k))
 return {k:{'estimate':task_equal(rows,k),'ci95':[float(np.quantile(out[k],.025)),float(np.quantile(out[k],.975))]} for k in keys}
def main():
 a=argparse.ArgumentParser();a.add_argument('--run',required=True,type=Path);x=a.parse_args();run=x.run.resolve();manifest=json.loads((run/'source_manifest.json').read_text())['sources']
 rows=[];channel=[];missing=[]
 for model in ('direct','joint'):
  for s in manifest:
   root=run/model/'actions'/s['source_id']
   paths={k:root/f'{v}.npz' for k,v in CELLS.items()};paths['D']=root/'capture_Dplus.npz';paths['Z']=root/'capture_Z1.npz'
   if not all(p.exists() for p in paths.values()):missing.append({'model':model,'source_id':s['source_id'],'missing':[str(p) for p in paths.values() if not p.exists()]});continue
   y={k:load_radial(p) for k,p in paths.items()};effects={'P_I0':y['Y01']-y['Y00'],'P_I1':y['Y11']-y['Y10'],'I_P0':y['Y10']-y['Y00'],'I_P1':y['Y11']-y['Y01'],'X':y['Y11']-y['Y10']-y['Y01']+y['Y00']}
   row={'model':model,'source_id':s['source_id'],'task_id':s['task_id'],'analysis_set':s['analysis_set'],'source_group_id':s['source_group_id'],
        'E10':rms(y['Y10']-y['D']),'E11':rms(y['Y11']-y['D'])}
   row['G_P_given_I']=row['E10']-row['E11'];row['zero_control_rms']=rms(y['Y00']-y['Z'])
   for k,v in effects.items():row[k+'_signed']=mean(v);row[k+'_rms']=rms(v)
   rows.append(row)
   z={k:np.load(p) for k,p in paths.items()}
   for idx in range(14):
    channel.append({'model':model,'source_id':s['source_id'],'task_id':s['task_id'],'channel':idx,
                    'P_I0_mean':mean(z['Y01']['joint_targets'][:,idx]-z['Y00']['joint_targets'][:,idx]),
                    'P_I1_mean':mean(z['Y11']['joint_targets'][:,idx]-z['Y10']['joint_targets'][:,idx]),
                    'X_mean':mean(z['Y11']['joint_targets'][:,idx]-z['Y10']['joint_targets'][:,idx]-z['Y01']['joint_targets'][:,idx]+z['Y00']['joint_targets'][:,idx])})
 out=run/'analysis';out.mkdir(exist_ok=True)
 for name,data in [('per_source_metrics.csv',rows),('per_channel_metrics.csv',channel)]:
  with (out/name).open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(data[0]) if data else ['status']);w.writeheader();w.writerows(data)
 keys=['E10','E11','G_P_given_I','zero_control_rms']+[f'{k}_{q}' for k in ('P_I0','P_I1','I_P0','I_P1','X') for q in ('signed','rms')]
 summary={}
 for model in ('direct','joint'):
  for setname,predicate in [('original34',lambda r:r['analysis_set']=='ORIGINAL34_RESULT_DRIVEN_FOLLOWUP'),('expanded42',lambda r:True)]:
   subset=[r for r in rows if r['model']==model and predicate(r)]
   summary[f'{model}_{setname}']={'n':len(subset),'tasks':sorted({r['task_id'] for r in subset}),'metrics':boot(subset,keys) if subset else {}}
 save(out/'bootstrap_summary.json',summary);save(out/'missing.json',missing)
 # Per-task, LOSO, and LOTO sensitivity use the same estimand and no source filtering.
 sensitivities=[]
 for model in ('direct','joint'):
  subset=[r for r in rows if r['model']==model]
  for task in sorted({r['task_id'] for r in subset}):
   g=[r for r in subset if r['task_id']==task];sensitivities.append({'model':model,'kind':'PER_TASK','omitted':'','task_id':task,'n':len(g),'G_P_given_I':mean([r['G_P_given_I'] for r in g])})
  for task in sorted({r['task_id'] for r in subset}):
   g=[r for r in subset if r['task_id']!=task];sensitivities.append({'model':model,'kind':'LOTO','omitted':task,'task_id':'','n':len(g),'G_P_given_I':task_equal(g,'G_P_given_I')})
  for r0 in subset:
   g=[r for r in subset if r['source_id']!=r0['source_id']];sensitivities.append({'model':model,'kind':'LOSO','omitted':r0['source_id'],'task_id':'','n':len(g),'G_P_given_I':task_equal(g,'G_P_given_I')})
 with (out/'sensitivity.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=sensitivities[0].keys());w.writeheader();w.writerows(sensitivities)
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 figdir=out/'figures';figdir.mkdir(exist_ok=True)
 colors={'direct':'#3B6FB6','joint':'#D07632'}
 fig,ax=plt.subplots(figsize=(5.2,4.2))
 for model in ('direct','joint'):
  g=[r for r in rows if r['model']==model];ax.scatter([r['E10']*1000 for r in g],[r['E11']*1000 for r in g],s=20,alpha=.7,label=model.capitalize(),c=colors[model])
 lim=max(ax.get_xlim()[1],ax.get_ylim()[1]);ax.plot([0,lim],[0,lim],'k--',lw=1);ax.set(xlabel='Interface only residual E10 (mm)',ylabel='Interface + proprio residual E11 (mm)');ax.legend();fig.tight_layout();fig.savefig(figdir/'paired_residuals.pdf');fig.savefig(figdir/'paired_residuals.png',dpi=240);plt.close(fig)
 fig,axs=plt.subplots(1,2,figsize=(9,4),sharey=False)
 for ai,model in enumerate(('direct','joint')):
  g=[r for r in rows if r['model']==model];vals=[[r[k+'_signed']*1000 for r in g] for k in ('P_I0','P_I1','I_P0','I_P1','X')];axs[ai].boxplot(vals,showfliers=False);axs[ai].axhline(0,color='k',lw=.8);axs[ai].set_xticklabels(['P|I0','P|I1','I|P0','I|P1','X'],rotation=25);axs[ai].set_title(model.capitalize());axs[ai].set_ylabel('Signed radial effect (mm)')
 fig.tight_layout();fig.savefig(figdir/'four_cell_effects.pdf');fig.savefig(figdir/'four_cell_effects.png',dpi=240);plt.close(fig)
 fig,ax=plt.subplots(figsize=(8,4.5));pt=[r for r in sensitivities if r['kind']=='PER_TASK'];labs=sorted({r['task_id'] for r in pt});xx=np.arange(len(labs));
 for j,m in enumerate(('direct','joint')):ax.scatter(xx+(j-.5)*.18,[next(r['G_P_given_I'] for r in pt if r['model']==m and r['task_id']==t)*1000 for t in labs],label=m.capitalize(),c=colors[m])
 ax.axhline(0,color='k',lw=.8);ax.set_xticks(xx,labs,rotation=25,ha='right');ax.set_ylabel('G P|I (mm)');ax.legend();fig.tight_layout();fig.savefig(figdir/'per_task_and_sensitivity.pdf');fig.savefig(figdir/'per_task_and_sensitivity.png',dpi=240);plt.close(fig)
 save(out/'analysis_manifest.json',{'status':'COMPLETE' if not missing else 'PARTIAL','seed':SEED,'draws':DRAWS,'n_rows':len(rows),'missing':len(missing),'scientific_identity':'POSTHOC_INTERFACE_X_PROPRIO_FACTOR'})
 print(json.dumps({'rows':len(rows),'missing':len(missing)}))
if __name__=='__main__':main()

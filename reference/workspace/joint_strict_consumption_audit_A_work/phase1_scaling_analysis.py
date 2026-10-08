#!/usr/bin/env python3
"""Offline-only posthoc scalar analysis. No project runtime/model imports."""
from wam_causal_audit.paths import resolve as _release_path
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(_release_path('@DATA@/wam_factor_routing_v5'))
SRC = ROOT / 'joint_compute_mechanism_posthoc_v1_20260909'
NEW = ROOT / 'joint_idm_compute_confirmatory_v1_20260909/joint_new_trajectory_confirmatory'
OUT = ROOT / 'phase1_evidence_mechanism_posthoc_v1_20260909'
STATS = Path(_release_path('@DATA@/BadWAM/models/LIQIIIII/badwam-libero-joint-wam/dataset_stats.json'))
B = 10000
SEED = 20260909

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def dump(p, d):
    Path(p).write_text(json.dumps(d, ensure_ascii=False, indent=2, allow_nan=False)+'\n')

def freeze():
    OUT.mkdir(exist_ok=True); (OUT/'figures').mkdir(exist_ok=True)
    target = OUT/'joint_scaling_freeze_manifest.json'
    if target.exists():
        raise RuntimeError('already frozen; do not overwrite')
    reg = pd.read_csv(NEW/'joint_new_trajectory_registry.csv').sort_values(['task_id','trajectory_id'])
    assert len(reg)==50 and not reg.duplicated(['task_id','trajectory_id']).any()
    fold=reg[['candidate_id','task_id','trajectory_id']].copy()
    fold['fold']=fold.groupby('task_id').cumcount()%5
    fold.to_csv(OUT/'joint_scaling_fold_registry.csv',index=False)
    protocol='''# Joint common-scaling posthoc analysis protocol

This explanation model is proposed after all C and strict-four-cell results were known. The 50 new-confirmation trajectories are now reused posthoc; cross-fitting does not restore blind confirmation status. Native and K5 use their own source schedules, as audited by the previous 71-check gate. No forward, donor construction, simulator, or rollout is executed here.

## Inputs and coordinates

Exactly tasks 5–9, 10 source trajectories per task, 50 registered starts and all 200 valid F3-G donors (-1,-0.5,+0.5,+1 cm). No selection by response or success. A00=(C0,F0), A10=(C1,F0), A01=(C0,F1), A11=(C1,F1); other conditions remain recipient. Native uses 10 world/10 action steps; K5 uses its own first five world caches, holding step-5 cache for action steps 6–10. Thirty layers are controlled throughout. F3 is never pooled with F3-G.

Use the existing denormalizer: a=(a_normalized+1)/2*(stepwise_max-stepwise_min)+stepwise_min; gripper=-(2*g-1). All arrays have 32 predicted action-chunk positions, not 32 denoising steps. Radial axis is recipient axis = -actual_goal_translation/(signed_dose_cm/100), normalized, matching the previous feature code; confirm all four doses produce the same axis. Positive dose moves goal toward object, opposite this object-to-goal axis. Radial values are dataset-denormalized translation commands, not measured executed displacements. Controller scaling is not silently treated as meters. Rotation and gripper remain separate channels.

## Fit and weighting

D10=A10-A00, D01=A01-A00, D11=A11-A00 projected at every chunk position. Only these three are fitting observations. For trajectory i, let x_i collect all four donors, three contrasts and 32 positions for native; y_i is the corresponding K5 vector. Minimize mean_i mean_j (y_ij-lambda*x_ij)^2, with one shared scalar through the origin: lambda=sum_i mean_j x_ij*y_ij / sum_i mean_j x_ij^2. No intercept, clipping, state-specific fit, parameter selection, or extra regressor. Each source trajectory has equal weight; all donors/contrasts/positions within it have equal weight. No missing case is replaced. Hash, shape, finite-value, axis or coverage failures block only this analysis and are reported.

Five fixed folds: sort trajectory IDs within each task, assign rank modulo 5. All donor/configuration/cell values from a trajectory stay in its fold. Fit on the other four folds; evaluate on the excluded fold. Full-data lambda is descriptive. Independent untouched matching four-cell data are not established, so no independent-test claim is made.

## Endpoints and uncertainty

Primary: held-fold radial-vector RMSE before (lambda=1) and after cross-fitted scaling; paired MSE change and fraction 1-SSE_scaled/SSE_unscaled over the full set. Fraction is omitted if the aggregate denominator is exactly zero; no per-case retained-percentage or cosine is computed because a suitable frozen near-zero validity domain was not established. The scalar is trained only on D10,D01,D11. Derived contrasts [D01, D11-D10, D10, D11-D01, D11, D11-D10-D01] are reported separately, not treated as extra independent fitting observations.

Report signed residual mean, dose-aligned signed mean, vector RMSE and raw trajectory values. Describe residuals by task, each frozen signed dose, and pre-fixed chunk quarters [0:8),[8:16),[16:24),[24:32). Save each time-position residual as well. Report original condition increments and native/K5 paired changes, including L2; repeated old figures are provenance checks, not new discoveries. Apply the same radial-fitted scalar to other translation axes, rotation and gripper solely as separate-unit secondary descriptions.

Bootstrap 10000 replicates with seed 20260909: sample five tasks with replacement, then 10 trajectories within each sampled task. All repeated values travel with that trajectory. Refit full-data and all held-fold scalars inside every bootstrap draw, preserving the original folds and preventing copies of a held-out trajectory entering training. Zero training denominators yield NA, with counts reported. Percentile 95% CIs are pointwise descriptive, not simultaneous significance tests; no p-values, equivalence decision, or multiple-testing discovery claims. Keep trajectory contributions; do not count donors, cells, quarters or positions as independent trajectories.

Natural donor-recipient and A11-A00 are compared directly using available arrays. They are called equal only if bit-exact; otherwise errors are separately reported. Hypotheses H_scale and H_residual are posthoc descriptive models. Nonzero future-vector magnitude supports sensitivity at this interface, not preservation of future dynamics reasoning or donor closed-loop redirection.
'''
    (OUT/'joint_scaling_analysis_protocol.md').write_text(protocol)
    inputs=[SRC/'native_k5_four_cell_results.csv',SRC/'mechanism_protocol.md',SRC/'mechanism_technical_gate.json',NEW/'joint_new_trajectory_registry.csv',NEW/'f3g_donor_registry.csv',STATS]
    dump(target,{'status':'POSTHOC_SCALING_FROZEN_BEFORE_COMPUTATION','created_unix':time.time(),'protocol_sha256':sha(OUT/'joint_scaling_analysis_protocol.md'),'folds_sha256':sha(OUT/'joint_scaling_fold_registry.csv'),'code_sha256':sha(__file__),'inputs':{str(p):sha(p) for p in inputs},'no_model_imports':True})
    print(json.dumps({'status':'FROZEN','path':str(target),'sha256':sha(target)}),flush=True)

def quant(a):
    a=np.asarray(a); a=a[np.isfinite(a)]
    return (float(np.quantile(a,.025)),float(np.quantile(a,.975)),len(a)) if len(a) else (None,None,0)

def run():
    manifest=json.loads((OUT/'joint_scaling_freeze_manifest.json').read_text())
    assert manifest['code_sha256']==sha(__file__)
    for p,h in manifest['inputs'].items(): assert sha(p)==h,p
    folds=pd.read_csv(OUT/'joint_scaling_fold_registry.csv')
    assert sha(OUT/'joint_scaling_fold_registry.csv')==manifest['folds_sha256']
    donors=pd.read_csv(NEW/'f3g_donor_registry.csv').set_index('case_id')
    tab=pd.read_csv(SRC/'native_k5_four_cell_results.csv')
    st=json.loads(STATS.read_text())['action']['default'];lo=np.asarray(st['stepwise_min']);hi=np.asarray(st['stepwise_max'])
    def dn(a):
        v=(np.asarray(a,dtype=float)+1)*.5*(hi-lo)+lo
        v[...,6]=-(2*v[...,6]-1);return v
    ns=50; nd=4;nt=32
    values=np.empty((ns,nd,2,4,nt,7)); radial_axes=[]; provenance=[]
    signs=np.array([-1.,-.5,.5,1.]); closure=[]
    natural_paths=pd.read_csv(NEW/'joint_new_trajectory_response_damage.csv').query('K == 5').set_index('case_id')
    for i,row in enumerate(folds.itertuples()):
        sub=tab[tab.candidate_id==row.candidate_id]; axes=[]
        assert len(sub)==32
        for j,dose in enumerate(signs):
            ds=sub[sub.signed_dose_cm==dose];case=ds.case_id.iloc[0];dr=donors.loc[case]
            axis=-np.array(json.loads(dr.actual_goal_translation_m))/(dose/100);axis/=np.linalg.norm(axis);axes.append(axis)
            for k,config in enumerate(['native','K5']):
                cr=ds[ds.configuration==config];p=Path(cr.action_path.iloc[0]);h=sha(p)
                assert (cr.action_file_sha256==h).all()
                with np.load(p,allow_pickle=False) as a:
                    for c,cell in enumerate(['A00','A10','A01','A11']):
                        assert a[cell].shape==(nt,7) and np.isfinite(a[cell]).all()
                        values[i,j,k,c]=dn(a[cell])
                    if k==0 and 'donor_K10' in a:
                        nat=dn(a['donor_K10'])-dn(a['recipient_K10'])
                    elif k==1:
                        npth=Path(natural_paths.loc[case,'actions_path']);assert sha(npth)==natural_paths.loc[case,'actions_sha256']
                        with np.load(npth,allow_pickle=False) as na:nat=dn(na['donor_K5'])-dn(na['recipient_K5'])
                    else: nat=None
                if nat is not None:
                    delta=values[i,j,k,3]-values[i,j,k,0];err=delta-nat
                    closure.append({'candidate_id':row.candidate_id,'case_id':case,'configuration':config,'bit_exact_effect_closure':np.array_equal(delta,nat),'radial_error_l2':np.linalg.norm(err[:,:3]@axis),'translation_error_l2':np.linalg.norm(err[:,:3]),'rotation_error_l2':np.linalg.norm(err[:,3:6]),'gripper_error_l2':np.linalg.norm(err[:,6])})
                provenance.append({'case_id':case,'configuration':config,'path':str(p),'sha256':h})
        assert np.max(np.abs(np.stack(axes)-axes[0]))<1e-10
        radial_axes.append(axes[0])
    pd.DataFrame(provenance).to_csv(OUT/'joint_scaling_input_action_hashes.csv',index=False)
    pd.DataFrame(closure).to_csv(OUT/'joint_world_vs_natural_closure.csv',index=False)
    delta=values[:,:,:,1:]-values[:,:,:,0:1]
    axis=np.stack(radial_axes); radial=np.einsum('idcptk,ik->idcpt',delta[...,:3],axis)
    x=radial[:,:,0];y=radial[:,:,1]
    xx=(x*x).mean((1,2,3));xy=(x*y).mean((1,2,3));yy=(y*y).mean((1,2,3))
    lam=xy.sum()/xx.sum();fid=folds.fold.to_numpy();ls=np.zeros(ns);params=[]
    rng=np.random.default_rng(SEED);tasks=folds.task_id.to_numpy();unique=np.unique(tasks)
    task_idx=rng.integers(0,5,size=(B,5));traj_idx=rng.integers(0,10,size=(B,5,10))
    task_rows=np.stack([np.flatnonzero(tasks==t) for t in unique])
    inds=task_rows[task_idx[:,:,None],traj_idx];weights=np.zeros((B,ns))
    np.add.at(weights,(np.repeat(np.arange(B),50),inds.reshape(-1)),1)
    bl=(weights@xy)/(weights@xx);bls=np.empty((B,ns))
    a,b,n=quant(bl);params.append({'scope':'FULL_DESCRIPTIVE','lambda':lam,'ci_low':a,'ci_high':b,'bootstrap_valid':n,'train_trajectories':50})
    for f in range(5):
        tr=fid!=f;te=~tr;l=xy[tr].sum()/xx[tr].sum();ls[te]=l
        den=weights[:,tr]@xx[tr];num=weights[:,tr]@xy[tr]
        boot=np.divide(num,den,out=np.full(B,np.nan),where=den>0);bls[:,te]=boot[:,None]
        a,b,n=quant(boot);params.append({'scope':f'HELD_FOLD_{f}','lambda':l,'ci_low':a,'ci_high':b,'bootstrap_valid':n,'train_trajectories':int(tr.sum())})
    pd.DataFrame(params).to_csv(OUT/'joint_scaling_parameters.csv',index=False)
    # Preserve the six derived effects without inflating the fitting basis.
    transform=np.array([[0,1,0],[-1,0,1],[1,0,0],[0,-1,1],[0,0,1],[-1,-1,1.]])
    names=['future_Crec','future_Cdonor','current_Frec','current_Fdonor','joint','interaction']
    ex=np.einsum('ep,idpt->idet',transform,x);ey=np.einsum('ep,idpt->idet',transform,y)
    res=ey-ls[:,None,None,None]*ex
    rows=[];traces=[];stats=[]
    def summarize(a,z,label,mask=None):
        aa=(a*a).mean(axis=tuple(range(1,a.ndim)));az=(a*z).mean(axis=tuple(range(1,a.ndim)));zz=(z*z).mean(axis=tuple(range(1,a.ndim)))
        am=a.mean(axis=tuple(range(1,a.ndim)));zm=z.mean(axis=tuple(range(1,a.ndim)))
        mask=np.ones(ns,dtype=bool) if mask is None else mask
        w=weights*mask[None,:];den=w.sum(1);valid=den>0
        scaled=zz-2*ls*az+ls**2*aa;raw=zz-2*az+aa
        bst=zz[None,:]-2*bls*az[None,:]+bls**2*aa[None,:]
        with np.errstate(divide='ignore',invalid='ignore'):
            bm=(w*bst).sum(1)/den;br=(w*raw).sum(1)/den
            signed=(w*(zm[None,:]-bls*am[None,:])).sum(1)/den
            explained=1-bm/br
        point_raw=float(raw[mask].mean());point_scaled=float(scaled[mask].mean())
        for metric,point,boot in [('rmse_unscaled',np.sqrt(max(0,point_raw)),np.sqrt(np.maximum(0,br))),('rmse_scaled_crossfit',np.sqrt(max(0,point_scaled)),np.sqrt(np.maximum(0,bm))),('mse_scaled_minus_unscaled',point_scaled-point_raw,bm-br),('signed_residual',float((zm-ls*am)[mask].mean()),signed)]:
            low,high,n=quant(boot[valid]);stats.append({**label,'metric':metric,'estimate':point,'ci_low':low,'ci_high':high,'bootstrap_valid':n,'trajectory_n':int(mask.sum()),'unit':'denormalized translation command' if 'mse_' not in metric else 'squared denormalized translation command'})
        if label.get('endpoint')=='FIT_BASIS' and label.get('scope')=='ALL' and point_raw>0:
            low,high,n=quant(explained[valid]);stats.append({**label,'metric':'fraction_unscaled_SSE_removed','estimate':1-point_scaled/point_raw,'ci_low':low,'ci_high':high,'bootstrap_valid':n,'trajectory_n':50,'unit':'aggregate SSE fraction; not source authority share'})
    summarize(x,y,{'endpoint':'FIT_BASIS','scope':'ALL','stratum':'ALL'})
    for e,name in enumerate(names):
        a=ex[:,:,e];z=ey[:,:,e]
        summarize(a,z,{'endpoint':name,'scope':'ALL','stratum':'ALL'})
        for t in unique:summarize(a,z,{'endpoint':name,'scope':'TASK','stratum':int(t)},tasks==t)
        for j,d in enumerate(signs):summarize(a[:,j],z[:,j],{'endpoint':name,'scope':'DOSE','stratum':float(d)})
        for q in range(4):summarize(a[:,:,8*q:8*(q+1)],z[:,:,8*q:8*(q+1)],{'endpoint':name,'scope':'CHUNK_QUARTER','stratum':q+1})
        for i,rr in enumerate(folds.itertuples()):
            for j,d in enumerate(signs):
                nr=a[i,j];kr=z[i,j];re=res[i,j,e]
                rows.append({'candidate_id':rr.candidate_id,'task_id':rr.task_id,'trajectory_id':rr.trajectory_id,'fold':rr.fold,'signed_dose_cm':d,'endpoint':name,'lambda_crossfit':ls[i],'native_mean':nr.mean(),'K5_mean':kr.mean(),'native_l2':np.linalg.norm(nr),'K5_l2':np.linalg.norm(kr),'scaled_native_l2':np.linalg.norm(ls[i]*nr),'raw_difference_rmse':np.sqrt(np.mean((kr-nr)**2)),'residual_rmse':np.sqrt(np.mean(re**2)),'signed_residual':re.mean(),'dose_aligned_signed_residual':re.mean()*np.sign(d)})
                for k in range(nt):traces.append({'candidate_id':rr.candidate_id,'task_id':rr.task_id,'trajectory_id':rr.trajectory_id,'signed_dose_cm':d,'endpoint':name,'chunk_index':k,'native':nr[k],'K5':kr[k],'scaled_native':ls[i]*nr[k],'residual':re[k]})
    pd.DataFrame(stats).to_csv(OUT/'joint_scaling_residual_statistics.csv',index=False)
    df=pd.DataFrame(rows);df.to_csv(OUT/'joint_scaling_trajectory_residuals.csv',index=False)
    trace=pd.DataFrame(traces);trace.to_csv(OUT/'joint_scaling_chunk_residuals.csv',index=False)
    secondary=[]
    for label,inds2 in [('translation_xyz',slice(0,3)),('rotation_xyz',slice(3,6)),('gripper',slice(6,7))]:
        sx=delta[:,:,0,...,inds2];sy=delta[:,:,1,...,inds2]
        for i,rr in enumerate(folds.itertuples()):secondary.append({'candidate_id':rr.candidate_id,'channel':label,'unscaled_rmse':np.sqrt(np.mean((sy[i]-sx[i])**2)),'scaled_rmse':np.sqrt(np.mean((sy[i]-ls[i]*sx[i])**2)),'interpretation':'secondary; radial-fitted scalar; channels not pooled'})
    pd.DataFrame(secondary).to_csv(OUT/'joint_scaling_secondary_channels.csv',index=False)
    fig,axs=plt.subplots(2,3,figsize=(13,7),constrained_layout=True)
    for e,ax in enumerate(axs.flat):
        p=trace[trace.endpoint==names[e]].groupby('chunk_index')[['native','K5','scaled_native','residual']].mean()
        for col in p:ax.plot(p.index,p[col],label=col)
        ax.axhline(0,color='grey',lw=.5);ax.set_title(names[e]);ax.set_xlabel('predicted chunk index');ax.set_ylabel('radial command')
    axs[0,0].legend(fontsize=8);fig.suptitle('Posthoc; predicted actions, not executed; equal trajectory/dose weights')
    fig.savefig(OUT/'figures/joint_radial_scaling_residuals.png',dpi=180);plt.close(fig)
    fig,axs=plt.subplots(2,3,figsize=(13,7),constrained_layout=True)
    for e,ax in enumerate(axs.flat):
        p=df[df.endpoint==names[e]].groupby(['task_id','trajectory_id'])[['native_l2','K5_l2','scaled_native_l2']].mean()
        for t in unique:
            v=p.loc[t];ax.scatter(v.native_l2,v.K5_l2-v.native_l2,s=15,label=f'task {t}')
        ax.axhline(0,color='grey',lw=.5);ax.set_title(names[e]);ax.set_xlabel('native radial L2');ax.set_ylabel('K5 - native radial L2')
    axs[0,0].legend(fontsize=7);fig.suptitle('All 50 trajectories; old paired differences reproduced descriptively')
    fig.savefig(OUT/'figures/joint_condition_trajectory_distribution.png',dpi=180);plt.close(fig)
    summary={'status':'JOINT_SCALING_POSTHOC_COMPLETE','lambda_full':float(lam),'lambda_folds':ls.reshape(-1).tolist(),'inputs_verified':len(provenance),'states':50,'donors':200,'bootstrap':B,'bootstrap_lambda_degenerate':int((~np.isfinite(bl)).sum()),'primary':[s for s in stats if s['endpoint']=='FIT_BASIS'],'protocol_sha256':manifest['protocol_sha256'],'code_sha256':sha(__file__),'model_forward_calls':0,'simulator_calls':0,'created_unix':time.time()}
    dump(OUT/'joint_scaling_result_manifest.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k!='lambda_folds'}),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['freeze','run']);args=parser.parse_args()
    freeze() if args.mode=='freeze' else run()

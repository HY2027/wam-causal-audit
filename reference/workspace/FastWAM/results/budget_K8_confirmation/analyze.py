"""CPU-only, pre-specified matched-budget analysis; never imports model runtime."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
from campaign import OUT,NEW,SRC,GAIN,MODEL,WORK,load,dump,sha,rows,table
def ci(x):
    x=np.asarray(x);x=x[np.isfinite(x)];return [float(np.quantile(x,.025)),float(np.quantile(x,.975))] if len(x) else [None,None]
def weights(folds):
    rng=np.random.default_rng(20260909);ti=rng.integers(0,5,size=(10000,5));si=rng.integers(0,10,size=(10000,5,10));tasks=folds.task_id.to_numpy()
    groups=np.stack([np.flatnonzero(tasks==t) for t in sorted(np.unique(tasks))]);inds=groups[ti[:,:,None],si];w=np.zeros((10000,50));np.add.at(w,(np.repeat(np.arange(10000),50),inds.reshape(-1)),1);return w
def gain(x,y,folds,w):
    xx=(x*x).mean((1,2,3));xy=(x*y).mean((1,2,3));yy=(y*y).mean((1,2,3));fid=folds.fold.to_numpy();ls=np.zeros(50);bls=np.empty((10000,50));params=[]
    def record(metric,point,boot):
        low,high=ci(boot);return dict(metric=metric,estimate=float(point),ci_low=low,ci_high=high,bootstrap_valid=int(np.isfinite(boot).sum()))
    params.append(record('lambda_full_descriptive',xy.sum()/xx.sum(),(w@xy)/(w@xx)))
    for f in range(5):
        tr=fid!=f;te=~tr;l=xy[tr].sum()/xx[tr].sum();ls[te]=l;den=w[:,tr]@xx[tr];num=w[:,tr]@xy[tr];boot=np.divide(num,den,out=np.full(10000,np.nan),where=den>0);bls[:,te]=boot[:,None];params.append(record(f'lambda_held_fold_{f}',l,boot))
    raw=yy-2*xy+xx;scaled=yy-2*ls*xy+ls**2*xx;bs=yy[None,:]-2*bls*xy[None,:]+bls**2*xx[None,:]
    br=(w*raw).sum(1)/50;bm=(w*bs).sum(1)/50
    params.extend([record('rmse_unscaled',np.sqrt(max(0,raw.mean())),np.sqrt(np.maximum(0,br))),record('rmse_scaled_crossfit',np.sqrt(max(0,scaled.mean())),np.sqrt(np.maximum(0,bm))),record('Q',1-scaled.mean()/raw.mean(),1-bm/br)])
    for name,index in [('fit_basis',None),('joint',2)]:
        a=x if index is None else x[:,:,index];b=y if index is None else y[:,:,index]
        nx=np.linalg.norm(a,axis=-1).mean(tuple(range(1,a.ndim-1)));ny=np.linalg.norm(b,axis=-1).mean(tuple(range(1,b.ndim-1)));diff=ny-nx
        params.append(record(f'{name}_mean_norm_change',diff.mean(),w@diff/50))
    return params,ls
def data(ks):
    cases=load(OUT/'cases.json');folds=pd.read_csv(GAIN/'joint_scaling_fold_registry.csv');assert folds.candidate_id.tolist()==list(dict.fromkeys(c['source']['candidate_id'] for c in cases))
    st=load(MODEL/'dataset_stats.json')['action']['default'];lo=np.array(st['stepwise_min']);hi=np.array(st['stepwise_max'])
    def denorm(a):
        v=(np.asarray(a,dtype=float)+1)*.5*(hi-lo)+lo;v[...,6]=-(2*v[...,6]-1);return v
    val=np.empty((50,4,len(ks),4,32,7));natural=np.empty((50,4,len(ks),32));axes=[];closure=[]
    for i in range(50):
        aa=[]
        for j in range(4):
            c=cases[i*4+j];d=c['donor'];axis=-np.array(json.loads(d['actual_goal_translation_m']))/(float(d['signed_dose_cm'])/100);axis/=np.linalg.norm(axis);aa.append(axis)
            for ki,k in enumerate(ks):
                p=OUT/'actions'/f"{d['case_id']}.npz" if k==8 else Path(c['native_path' if k==10 else 'K5_path'])
                with np.load(p) as z:
                    for a,arm in enumerate(['A00','A10','A01','A11']):
                        assert z[arm].shape==(32,7) and np.isfinite(z[arm]).all();val[i,j,ki,a]=denorm(z[arm])
                    n=denorm(z[f'donor_K{k}'])-denorm(z[f'recipient_K{k}']);natural[i,j,ki]=n[:,:3]@axis
                    er=(val[i,j,ki,3]-val[i,j,ki,0])-n;closure.append(dict(case_id=d['case_id'],K=k,exact=bool(np.array_equal(er,np.zeros_like(er))),max_abs_error=float(np.abs(er).max())))
        assert np.max(np.abs(np.stack(aa)-aa[0]))<1e-10;axes.append(aa[0])
    delta=val[:,:,:,1:]-val[:,:,:,0:1];radial=np.einsum('idcptk,ik->idcpt',delta[...,:3],np.stack(axes))
    return folds,cases,radial,natural,closure
def old_check(params):
    ps=pd.read_csv(GAIN/'joint_scaling_parameters.csv');ss=pd.read_csv(GAIN/'joint_scaling_residual_statistics.csv');out=[]
    for r in params:
        m=r['metric']
        if m.startswith('lambda'):
            scope='FULL_DESCRIPTIVE' if m=='lambda_full_descriptive' else 'HELD_FOLD_'+m.split('_')[-1];rr=ps[ps.scope==scope].iloc[0];old={'estimate':rr['lambda'],'ci_low':rr.ci_low,'ci_high':rr.ci_high}
        elif m in ['rmse_unscaled','rmse_scaled_crossfit','Q']:
            name='fraction_unscaled_SSE_removed' if m=='Q' else m;rr=ss[(ss.endpoint=='FIT_BASIS')&(ss.scope=='ALL')&(ss.metric==name)].iloc[0];old=rr
        else:continue
        for field in ['estimate','ci_low','ci_high']:out.append(dict(metric=m,field=field,old=float(old[field]),recomputed=r[field],absolute_error=abs(float(old[field])-r[field])))
    table(OUT/'K5_statistics_reproduction.csv',out);assert max(x['absolute_error'] for x in out)<1e-12,'HISTORICAL_STATISTICS_NOT_REPRODUCED'
    return max(x['absolute_error'] for x in out)
def run(precheck=False):
    ks=[10,5] if precheck else [10,8,5];folds,cases,r,nat,closure=data(ks);w=weights(folds);par,ls=gain(r[:,:,0],r[:,:,-1],folds,w);maxerr=old_check(par);table(OUT/'gain_K5_vs_K10_reproduced.csv',par)
    if precheck:
        dump(OUT/'offline_gate.json',dict(status='PASS',cases=200,trajectories=50,folds=5,bootstrap=10000,seed=20260909,maximum_statistics_absolute_error=maxerr));print('K5_GAIN_EXACT_REPRODUCTION',maxerr);return
    assert load(OUT/'run_status.json')['status']=='COMPLETE'
    par8,ls8=gain(r[:,:,0],r[:,:,1],folds,w);table(OUT/'gain_K8_vs_K10.csv',par8);dump(OUT/'natural_vs_A11_checks.json',closure)
    transform=np.array([[0,1,0],[-1,0,1],[1,0,0],[0,-1,1],[0,0,1],[-1,-1,1.]])
    effects=np.einsum('ep,idcpt->idcet',transform,r);names=['future_Crec','future_Cdonor','current_Frec','current_Fdonor','joint','interaction'];case_rows=[];trajectory=[];summary=[]
    timing=pd.read_csv(OUT/'timing_calls.csv');assert len(timing)==750
    for ki,k in enumerate(ks):
        for i in range(50):
            for j in range(4):
                c=cases[i*4+j];d=c['donor'];nr=nat[i,j,ki]
                row=dict(candidate_id=d['candidate_id'],case_id=d['case_id'],task_id=int(d['task_id']),trajectory_id=int(d['trajectory_id']),signed_dose_cm=float(d['signed_dose_cm']),K=k,natural_radial_vector=json.dumps(nr.tolist()),natural_radial_signed_mean=float(nr.mean()),natural_radial_l2=float(np.linalg.norm(nr)),natural_rmse_vs_K10=float(np.sqrt(np.mean((nr-nat[i,j,0])**2))),fit_basis_rmse_vs_K10=float(np.sqrt(np.mean((r[i,j,ki]-r[i,j,0])**2))))
                for q,label in enumerate(['delta10','delta01','delta11']):row[label+'_radial_vector']=json.dumps(r[i,j,ki,q].tolist())
                for e,name in enumerate(names):
                    a=effects[i,j,ki,e];row[name+'_signed_mean']=float(a.mean());row[name+'_l2']=float(np.linalg.norm(a));row[name+'_rms']=float(np.sqrt(np.mean(a*a)))
                case_rows.append(row)
            t=timing[(timing.candidate_id==folds.candidate_id.iloc[i])&(timing.K==k)];assert len(t)==5
            rr=dict(candidate_id=folds.candidate_id.iloc[i],task_id=int(folds.task_id.iloc[i]),trajectory_id=int(folds.trajectory_id.iloc[i]),fold=int(folds.fold.iloc[i]),K=k,donors=4,static_endpoint_calls=5,cumulative_policy_seconds=float(t.policy_seconds.sum()),median_policy_seconds=float(t.policy_seconds.median()),inclusive_static_seconds=float(t.inclusive_seconds.sum()),natural_mse_vs_K10=float(np.mean((nat[i,:,ki]-nat[i,:,0])**2)),fit_basis_mse_vs_K10=float(np.mean((r[i,:,ki]-r[i,:,0])**2)))
            for e,name in enumerate(names):rr[name+'_l2']=float(np.linalg.norm(effects[i,:,ki,e],axis=-1).mean());rr[name+'_signed_mean']=float(effects[i,:,ki,e].mean())
            trajectory.append(rr)
        tr=pd.DataFrame([z for z in trajectory if z['K']==k]);tt=timing[timing.K==k]
        def add(metric,value,boot,unit,scope='ALL'):
            low,high=ci(boot);summary.append(dict(K=k,scope=scope,metric=metric,estimate=float(value),ci_low=low,ci_high=high,unit=unit,trajectories=50 if scope=='ALL' else 10,donors=200 if scope=='ALL' else 40))
        for m in ['natural_mse_vs_K10','fit_basis_mse_vs_K10']:
            v=tr[m].to_numpy();add(m.replace('mse','rmse'),np.sqrt(v.mean()),np.sqrt(w@v/50),'denormalized translation command')
        for name in names:
            for suffix in ['l2','signed_mean']:
                v=tr[name+'_'+suffix].to_numpy();add(name+'_'+suffix,v.mean(),w@v/50,'denormalized translation command')
                for task in range(5,10):
                    mask=folds.task_id.to_numpy()==task;den=w[:,mask].sum(1);boot=np.divide(w[:,mask]@v[mask],den,out=np.full(10000,np.nan),where=den>0);add(name+'_'+suffix,v[mask].mean(),boot,'denormalized translation command',f'TASK_{task}')
        v=tr.cumulative_policy_seconds.to_numpy();add('cumulative_static_policy_seconds',v.sum(),w@v,'s for 250 fixed endpoint calls')
        for metric,value in [('per_call_median_seconds',tt.policy_seconds.median()),('per_call_p95_seconds',tt.policy_seconds.quantile(.95)),('inclusive_endpoint_seconds',tt.inclusive_seconds.sum())]:add(metric,value,[],'s')
    table(OUT/'case_level.csv',case_rows);table(OUT/'trajectory_summary.csv',trajectory);table(OUT/'budget_summary.csv',summary)
    np.savez_compressed(OUT/'matched_radial_vectors.npz',budgets=np.array(ks),delta_radial=r,natural_radial=nat,conditional_radial=effects,bootstrap_weights=w,lambda_K8=ls8,lambda_K5=ls)
    plot(pd.DataFrame(summary));report(par8,par,summary,maxerr)
def plot(s):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    sys.path.insert(0,str(OUT.parents[1]/'scripts'))
    from main_figure_layout_v2 import configure_layout_v2
    from main_figure_common import MODEL_COLORS
    configure_layout_v2();fig,axs=plt.subplots(1,3,figsize=(9,2.65),layout='constrained')
    panels=[('per_call_median_seconds','(a) Static policy computation','Median seconds / call'),('natural_rmse_vs_K10','(b) Natural-response discrepancy','Radial vector RMSE'),('future_Cdonor_l2','(c) Future | current donor','Mean radial vector L2')]
    for ax,(metric,title,label) in zip(axs,panels):
        z=s[(s.scope=='ALL')&(s.metric==metric)].set_index('K').loc[[10,8,5]];y=z.estimate.to_numpy();ax.plot(range(3),y,'o-',color=MODEL_COLORS['Joint'],lw=1.4,ms=4)
        if z.ci_low.notna().all():ax.errorbar(range(3),y,yerr=np.stack([y-z.ci_low.to_numpy(),z.ci_high.to_numpy()-y]),fmt='none',ecolor=MODEL_COLORS['Joint'],capsize=3)
        ax.set_xticks(range(3),['K=10','K=8','K=5']);ax.set_title(title,fontweight='bold');ax.set_xlabel('World denoising budget');ax.set_ylabel(label);ax.grid(axis='y',alpha=.15)
    fig.savefig(OUT/'budget_trend.pdf');fig.savefig(OUT/'budget_trend.png',dpi=240);plt.close(fig)
def report(p8,p5,summary,maxerr):
    def get(p,m):return next(r for r in p if r['metric']==m)
    s=pd.DataFrame(summary);s=s[s.scope=='ALL'];lines=['# Matched K=8 response extension','', 'Status: COMPLETE. Post-hoc budget extension on the same previously studied 50 confirmation trajectories and 200 valid F3-G donors. Tasks 5–9; four signed doses per trajectory. No new state, donor construction, simulation, predicted-action execution, or closed-loop rollout.', '', '## Reproduction and provenance', '', 'Five pre-registered technical cases (one per task, original −1 cm donors) reproduced K=10 and K=5 endpoint captures and strict four cells exactly before K=8. All subsequent own-cache timing actions were checked against matched arrays. Historical files remain unchanged.', f'Original K5 gain estimates and intervals reproduced with maximum absolute difference {maxerr:.17g}.', '', '## Separate gain comparisons', '', '| Budget | Unscaled RMSE | Cross-fitted RMSE | Q [95% CI] | Descriptive lambda [95% CI] |','|---|---:|---:|---|---|']
    for k,p in [(8,p8),(5,p5)]:
        a=get(p,'rmse_unscaled');b=get(p,'rmse_scaled_crossfit');q=get(p,'Q');l=get(p,'lambda_full_descriptive');lines.append(f"| {k} | {a['estimate']:.9g} | {b['estimate']:.9g} | {q['estimate']:.5f} [{q['ci_low']:.5f}, {q['ci_high']:.5f}] | {l['estimate']:.6f} [{l['ci_low']:.6f}, {l['ci_high']:.6f}] |")
    lines+=['', 'Fit basis: D10, D01 and D11, projected separately at all 32 predicted chunk positions. One scalar shared across sources/doses/contrasts/positions, no intercept or clipping. The five original trajectory folds are retained. K8 and K5 are fitted separately. Q is the aggregate fraction of unscaled squared vector discrepancy removed, not an information share or mechanism-preservation measure. Norm changes and individual folds are saved in the gain tables.', '', '## Descriptive three-budget trend', '', '| K | Natural-response RMSE vs native | Future conditional L2, current donor | Static median / p95 seconds | Cumulative policy seconds, 250 calls |','|---|---:|---:|---:|---:|']
    for k in [10,8,5]:
        z=s[s.K==k].set_index('metric').estimate
        lines.append(f"| {k} | {z['natural_rmse_vs_K10']:.9g} | {z['future_Cdonor_l2']:.9g} | {z['per_call_median_seconds']:.5f} / {z['per_call_p95_seconds']:.5f} | {z['cumulative_static_policy_seconds']:.3f} |")
    z=s[s.metric=='natural_rmse_vs_K10'].set_index('K').estimate
    lines+=['', 'Response discrepancy increases as world-branch computation is reduced across the tested budgets.' if z[10]<=z[8]<=z[5] else 'The natural-response discrepancy is non-monotonic across these budgets; K8 does not lie between native and K5.', '', 'The future increment is reported for both current-recipient (A01−A00) and current-donor (A11−A10) backgrounds in budget_summary.csv; the displayed norm is the latter. No threshold, breakpoint, or nonlinear response model was fitted.', '', '## Units, uncertainty and limits', '', 'Radial units are the original stepwise-denormalized translation-command coordinate, not measured robot displacement or millimetres. Radial axis and dose sign match the frozen F3-G analysis. L2 norms refer to a 32-position predicted vector; positions, donors and cells are not independent trajectories. A11−A00 was checked directly against natural donor−recipient arrays.', '', 'Intervals reuse the original 10,000 task→trajectory bootstrap draws (seed 20260909), jointly carrying all configurations, doses and cells. Gains are refitted in each draw; intervals are pointwise descriptive percentile intervals. Five task clusters and 50 source trajectories. Reuse of this studied cohort does not make the extension a new blind confirmation.', '', 'Timing compares the same 250 unique static inputs with deterministic counterbalanced budget order. Native uses the unmodified inference entry; K8/K5 use their own newly generated stop cache and all ten action steps. Policy timers exclude input preparation; preparation and inclusive outer elapsed times are separately saved. Strict diagnostics and schedule-hash capture costs are excluded from production timing. Cumulative static compute is not cumulative closed-loop compute or robot completion time. Shared-GPU occupancy can affect latency; 1-second resource samples are not a continuous-time peak guarantee.', '', 'No claims of a critical computation threshold, equivalence, safe deployment, or preserved mechanism. No K8 closed-loop campaign was launched.', '', '## Deliverables and cost', '', 'See forward_ledger.jsonl, consumption_checks.json, numerical_checks.json, timing_calls.csv, run_status.json and cleanup.json for every attempt, consumption check, output comparison, model-load cost, and cleanup. All 1,710 planned policy calls include 60 reproduction, 900 strict K8, and 750 production timing calls. File identities are sealed in provenance.json only after outputs stop changing.']
    (OUT/'analysis.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--precheck',action='store_true');args=p.parse_args();run(args.precheck)

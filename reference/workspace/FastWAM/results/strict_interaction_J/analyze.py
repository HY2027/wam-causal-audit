"""CPU-only post-hoc strict interaction audit. No model/simulator imports."""
from wam_causal_audit.paths import resolve as _release_path
from pathlib import Path
import csv, json, hashlib
from collections import defaultdict
import numpy as np

OUT=Path(__file__).resolve().parent
ROOT=Path(_release_path('@DATA@/wam_factor_routing_v5'))
A=ROOT/'joint_experiment_A_strict_consumer_v1'
G=Path(_release_path('@DATA@/wam_control_state_v3/group1_factor_phase'))
B=ROOT/'joint_experiment_B_distance_v1/B_forward_development_validation_v1'
S=ROOT/'experiments/followup_s1_s2_s3'
V=S/'s1_f3g_attribution/v3'
WORK=Path(_release_path('@WORKSPACE@/joint_strict_consumption_audit_A_work'))
INPUTS={}; CASES=[]; TENSORS={}; CHECKS=[]; EXCLUDED=[]

def track(p, expected=None):
    p=Path(p); key=str(p)
    if key not in INPUTS:
        INPUTS[key]=dict(path=key,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    if expected:
        assert INPUTS[key]['sha256']==expected, ('HASH_MISMATCH',key)
        INPUTS[key]['frozen_hash_verified']=True
    return p
def js(p):return json.loads(track(p).read_text())
def rows(p):return list(csv.DictReader(track(p).open()))
def archive(p):
    with np.load(track(p),allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def write(name, data):
    p=OUT/name
    if p.suffix=='.json':p.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    else:
        keys=list(dict.fromkeys(k for r in data for k in r))
        with p.open('w') as f:
            w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(data)
def append(meta, cells, norm, axis, evidence):
    assert set(cells)==set(['A00','A10','A01','A11'])
    assert all(x.shape==(32,7) and np.isfinite(x).all() for x in cells.values())
    q={k:x.astype(np.float64) for k,x in cells.items()}
    d={k:q['A'+k]-q['A00'] for k in ['10','01','11']}
    d['J']=d['11']-d['10']-d['01']
    n={k:x.astype(np.float64) for k,x in norm.items()}
    jn=(n['A11']-n['A00'])-(n['A10']-n['A00'])-(n['A01']-n['A00'])
    r=dict(meta,input_dtype=str(cells['A00'].dtype),analysis_dtype='float64',action_shape='32x7',radial_unit=json.dumps(axis.tolist()),identity_status=evidence)
    for k,v in d.items():
        rad=v[:,:3]@axis
        r[k+'_signed_mean']=float(rad.mean());r[k+'_radial_rms']=float(np.sqrt(np.mean(rad**2)))
        r[k+'_radial_vector']=json.dumps(rad.tolist())
        TENSORS[meta['case_id']+'__'+k+'__env_continuous']=v
    r['J_normalized_full_action_l2']=float(np.linalg.norm(jn))
    r['J_full_action_mixed_units_l2_descriptive_only']=float(np.linalg.norm(d['J']))
    r['J_translation_l2']=float(np.linalg.norm(d['J'][:,:3]))
    r['J_rotation_l2']=float(np.linalg.norm(d['J'][:,3:6]))
    r['J_gripper_l2']=float(np.linalg.norm(d['J'][:,6]))
    TENSORS[meta['case_id']+'__J__normalized']=jn
    closure=float(np.max(np.abs(d['11']-(d['10']+d['01']+d['J']))))
    wrong=float(np.max(np.abs(d['11']-(d['10']-d['01']-d['J']))))
    CHECKS.append(dict(case_id=meta['case_id'],correct_closure_max_abs=closure,user_step5_minus_expression_max_abs=wrong,input_dtype=r['input_dtype'],arithmetic_dtype='float64'))
    CASES.append(r)

def f1():
    validation=js(A/'analysis_validation.json')
    for name,h in validation['outputs'].items():
        if name in ['consumer_registry.json','factorial_metrics.csv']:track(A/name,h)
    js(A/'consumer_registry.json'); js(A/'run_manifest.json')
    phase=[json.loads(l) for l in track(G/'phase_registry.jsonl').read_text().splitlines()]
    phase={(int(r['task_id']),int(r['source_state_id'])):r for r in phase if r.get('model')=='joint' and r.get('phase')=='PREGRASP'}
    donors=[json.loads(l) for l in track(G/'donor_bank/joint/counterfactual_qc.jsonl').read_text().splitlines()]
    donors={(int(r['task_id']),int(r['source_state_id']),float(r['signed_dose'])):r for r in donors if r.get('factor')=='F1_ROBOT_RADIAL_PROGRESS' and r.get('phase')=='PREGRASP' and r.get('relation_control')=='STANDARD'}
    old={(int(r['task_id']),int(r['source_state_id']),float(r['signed_dose_cm'])):r for r in rows(A/'factorial_metrics.csv') if r['action_space']=='radial_translation_raw'}
    for idx in rows(A/'all_result_index.csv'):
        t,s=int(idx['task_id']),int(idx['source_state_id']);p=phase[t,s]
        cp=track(idx['case_complete_path'],idx['case_complete_sha256']);m=js(cp)
        track(idx['actions_path'],idx['actions_sha256']);z=archive(idx['actions_path'])
        assert m['seed']==p['policy_seed'] and m['base_state_id']==p['base_state_id']
        cfg={r['action_key']:r for r in rows(cp.parent/'configurations.csv')}
        caches=rows(cp.parent/'source_cache_hashes.csv');rows(cp.parent/'replay_checks.csv')
        assert np.array_equal(z['STRICT__C0__F0__normalized'],z['NATURAL__0__normalized'])
        axis=np.array(p['object_position_m'])-np.array(p['eef_position_m']);axis/=np.linalg.norm(axis)
        for dose in [-4.,-2.,-1.,1.,2.,4.]:
            lab=('p' if dose>0 else 'm')+str(int(abs(dose)))
            keys={'A00':'STRICT__C0__F0','A10':f'STRICT__C{lab}__F0','A01':f'STRICT__C0__F{lab}','A11':f'STRICT__C{lab}__F{lab}'}
            donor=donors.get((t,s,dose));cid=donor['case_id'] if donor else f'task{t}_state{s}_dose{dose}'
            if not all(k+'__env_continuous' in z for k in keys.values()):
                EXCLUDED.append(dict(factor='F1',case_id=cid,reason='STRICT_GRID_NOT_SAVED',donor_valid=donor.get('donor_valid') if donor else None,detail='Frozen A executes a magnitude grid only when both signs are valid; no new eligibility filter.'));continue
            assert donor and donor['donor_valid']
            for cell,k in keys.items():
                r=cfg[k];assert r['executed']=='True' and r['hook_all_exact']=='True'
                expected=(dose if cell in ['A10','A11'] else 0,dose if cell in ['A01','A11'] else 0)
                assert (float(r['C_source_cm']),float(r['F_source_cm']))==expected
            append(dict(factor='F1',case_id=cid,task=t,source_group=f'task{t}_state{s:02d}',state_id=p['base_state_id'],phase='PREGRASP',signed_dose_cm=dose,seed=m['seed'],split='historical_A',array_path=idx['actions_path']),{k:z[v+'__env_continuous'] for k,v in keys.items()},{k:z[v+'__normalized'] for k,v in keys.items()},axis,'HASH_VERIFIED; configurations hook_all_exact; strict runner assertion; trajectory ancestry not certified')
            assert np.isclose(CASES[-1]['J_radial_rms']*np.sqrt(32),float(old[t,s,dose]['interaction_l2']),rtol=1e-12,atol=1e-14)

def f3():
    f=js(S/'protocol_freeze_s1.json');expected={x['absolute_path']:x['sha256'] for x in f['inputs']}
    stats_path=Path(_release_path('@DATA@/BadWAM/models/LIQIIIII/badwam-libero-joint-wam/dataset_stats.json'))
    track(stats_path,expected[str(stats_path)]);st=js(stats_path)['action']['default']
    lo=np.asarray(st['global_min'],dtype=np.float32);hi=np.asarray(st['global_max'],dtype=np.float32)
    span=hi-lo;mask=span<1e-4;span[mask]=2.;scale=np.float32(2)/span;offset=np.float32(-1)-scale*lo;offset[mask]=-lo[mask]
    def denorm(x):
        y=((x.astype(np.float32)-offset)/scale).reshape(32,7);y[:,-1]=-(y[:,-1]*2-1);return y
    metrics={r['case_id']:r for r in rows(B/'B_factorial_metrics.csv')}
    historical={r['case_id']:r for r in rows(V/'02_per_case_attribution.csv') if r.get('physical_QC_valid')=='True'}
    js(B/'B_forward_manifest.json');rows(B/'B_forward_replay_checks.csv')
    for c in f['cases']:
        r,d=c['state'],c['donor'];path=Path(c['old_actions']);track(path,expected[str(path)])
        z=archive(path);m=js(path.parent.parent.parent/'state_manifest.json');metric=metrics[d['case_id']]
        assert m['completed'] and m['state']==r
        assert metric['hook_all_exact']=='True' and metric['condition']=='RADIAL'
        assert int(metric['task_id'])==int(r['task_id'])==d['task_id']
        assert int(metric['trajectory_id'])==int(r['trajectory_id'])==d['trajectory_id']
        assert float(metric['signed_dose_cm'])==d['signed_dose_cm'] and d['donor_valid']
        assert c['seed']==820000+int(r['task_id'])*1000+int(r['trajectory_id'])*20+int(r['policy_call'])
        assert np.array_equal(z['A00'],z['recipient_natural'])
        vec=lambda x:np.fromstring(x.strip('[]'),sep=' ')
        axis=vec(r['goal_position_m'])-vec(r['object_position_m']);axis[2]=0;axis/=np.linalg.norm(axis)
        keys=['A00','A10','A01','A11'];cells={k:denorm(z[k]) for k in keys}
        append(dict(factor='F3-G',case_id=d['case_id'],task=int(r['task_id']),source_group=f"task{r['task_id']}_traj{r['trajectory_id']}",state_id=r['candidate_id'],phase=d['phase_base'],signed_dose_cm=d['signed_dose_cm'],seed=c['seed'],split=r['split'],array_path=str(path)),cells,{k:z[k] for k in keys},axis,'HASH_VERIFIED; per-case hook_all_exact; recipient context in strict runner; no saved per-event context hash proof')
        # Independent historical radial-vector agreement; numerical reordering only.
        h=np.array(json.loads(historical[d['case_id']]['J']))
        assert np.allclose(h,np.array(json.loads(CASES[-1]['J_radial_vector'])),rtol=0,atol=2e-15)
    for c in f['invalid_cases']:EXCLUDED.append(dict(factor='F3-G',case_id=c['donor']['case_id'],reason='FROZEN_DONOR_QC_INVALID',detail=c['donor']['invalid_reasons']))

METRICS=[k+'_'+m for k in ['J','10','01','11'] for m in ['signed_mean','radial_rms']]
def stats(factor):
    rr=[r for r in CASES if r['factor']==factor];states=list(dict.fromkeys(r['state_id'] for r in rr));doses=sorted(set(r['signed_dose_cm'] for r in rr));si={s:i for i,s in enumerate(states)}
    data=np.full((len(states),len(doses),len(METRICS)),np.nan);meta={r['state_id']:r for r in rr}
    for r in rr:data[si[r['state_id']],doses.index(r['signed_dose_cm'])]=[r[k] for k in METRICS]
    groups=defaultdict(list)
    for s,i in si.items():groups[(meta[s]['task'],meta[s]['source_group'])].append(i)
    tasks=sorted(set(t for t,g in groups));bytask={t:sorted(g for tt,g in groups if tt==t) for t in tasks}
    samples=[]
    # F1 original signed-dose procedure uses available states in that stratum.
    # F3 original draws whole trajectory blocks for all doses, preserving repeated states.
    if factor=='F3-G':
        rng=np.random.default_rng(20260907);weights=np.zeros((10000,len(states)))
        for b in range(10000):
            for t in rng.choice(tasks,len(tasks),replace=True):
                for g in rng.choice(bytask[t],len(bytask[t]),replace=True):weights[b,groups[t,g]]+=1
    for di,dose in list(enumerate(doses))+[(None,'aggregate_state_mean')]:
        x=np.nanmean(data,axis=1) if di is None else data[:,di]
        valid=np.isfinite(x[:,0]);xx=x[valid];ids=np.flatnonzero(valid)
        if factor=='F1':
            rng=np.random.default_rng(20260907);w=np.zeros((10000,len(states)));bt={t:[i for i in ids if meta[states[i]]['task']==t] for t in tasks};ts=[t for t in tasks if bt[t]]
            for b in range(10000):
                for t in rng.choice(ts,len(ts),replace=True):np.add.at(w[b],rng.choice(bt[t],len(bt[t]),replace=True),1)
            ww=w[:,valid]
        else:ww=weights[:,valid]
        draw=(ww@xx)/ww.sum(axis=1)[:,None];point=xx.mean(axis=0);low,high=np.quantile(draw,[.025,.975],axis=0)
        for j,k in enumerate(METRICS):samples.append(dict(factor=factor,stratum=dose,metric=k,estimate=point[j],ci95_low=low[j],ci95_high=high[j],n_states=int(valid.sum()),n_source_groups=len(set(meta[states[i]]['source_group'] for i in ids)),task_clusters=len(set(meta[states[i]]['task'] for i in ids)),bootstrap_resamples=10000,bootstrap_seed=20260907,unit='de-normalized translation command; not executed metres',interpretation='The strict interaction is not resolved from zero under this readout.' if k=='J_signed_mean' and low[j]<=0<=high[j] else ''))
    return samples

def main():
    assert not (OUT/'case_level.csv').exists(),'Refuse overwrite: choose a new version directory.'
    for name in ['analyze_experiment_a.py','run_experiment_a.py','run_b_forward.py','s1_statistics.py','s1_hash_adapter_v2.py','s1_finalize_v3.py']:track(WORK/name)
    f1();f3();write('case_level.csv',CASES);write('excluded_cases.csv',EXCLUDED)
    np.savez_compressed(OUT/'case_vectors.npz',**TENSORS)
    source=[]
    for key in sorted(set((r['factor'],r['task'],r['source_group']) for r in CASES)):
        rr=[r for r in CASES if (r['factor'],r['task'],r['source_group'])==key]
        for dose in sorted(set(r['signed_dose_cm'] for r in rr))+['aggregate_state_mean']:
            ss=[r for r in rr if r['signed_dose_cm']==dose] if isinstance(dose,float) else rr
            bystate=defaultdict(list)
            for r in ss:bystate[r['state_id']].append([r[k] for k in METRICS])
            vals=np.mean([np.mean(v,axis=0) for v in bystate.values()],axis=0)
            source.append(dict(factor=key[0],task=key[1],source_group=key[2],stratum=dose,n_states=len(bystate),n_cases=len(ss),**dict(zip(METRICS,vals))))
    write('source_level.csv',source)
    summary=stats('F1')+stats('F3-G');write('dose_summary.csv',summary)
    task=[]
    for factor in ['F1','F3-G']:
        for t in sorted(set(r['task'] for r in CASES if r['factor']==factor)):
            rr=[r for r in CASES if r['factor']==factor and r['task']==t]
            for dose in sorted(set(r['signed_dose_cm'] for r in rr))+['aggregate_state_mean']:
                ss=[r for r in rr if r['signed_dose_cm']==dose] if isinstance(dose,float) else rr
                state=defaultdict(list)
                for r in ss:state[r['state_id']].append([r[k] for k in METRICS])
                val=np.mean([np.mean(v,axis=0) for v in state.values()],axis=0)
                task.append(dict(factor=factor,task=t,stratum=dose,n_cases=len(ss),n_states=len(state),n_source_groups=len(set(r['source_group'] for r in ss)),**dict(zip(METRICS,val))))
    write('task_summary.csv',task)
    closure=dict(definition='J=delta11-delta10-delta01',correct_identity='delta11=delta10+delta01+J',user_step5='minus signs are algebraically inconsistent with J definition',input_dtype='float32',arithmetic_dtype='float64',max_abs_error=max(r['correct_closure_max_abs'] for r in CHECKS),max_minus_expression_error=max(r['user_step5_minus_expression_max_abs'] for r in CHECKS),cases=CHECKS)
    write('closure_checks.json',closure)
    lines=['# Strict Joint-WAM interaction J — post-hoc saved-array analysis','',
    'No model forward pass, simulator step, rollout, rendering, or donor generation was executed. Historical files were not modified.',
    '', '## Definition and closure', 'J = (A11−A00)−(A10−A00)−(A01−A00). The correct closure is δ11=δ10+δ01+J. The requested step-5 minus-sign expression is not this identity and is separately audited.',
    f"Maximum correct full-tensor closure error: {closure['max_abs_error']:.17e}; input float32, subtraction/aggregation float64. Maximum error of the supplied minus-sign expression: {closure['max_minus_expression_error']:.17e}.",
    '', '## Identity and coverage',
    'F1 uses hash-verified per-state action archives, frozen phase/donor registries, executed configuration mappings and hook-exact flags. Strict inference fixes recipient context and injects post-RoPE K / projected V at all 30×10 consumption sites. A00 equals saved recipient exactly. A00 is the saved native baseline rather than an additional formal inference; its same-value equivalence is covered by historical A0 checks.',
    'F3-G reuses hash-verified original B arrays and the S1 registry; A00 equals recipient exactly. Per-case hook_all_exact and the strict runner confirm assigned sources. The historical Z_context_hash_same_value field compares the context to itself and is NOT independent runtime proof. Recipient-context consistency is supported by the actual call wiring; raw per-event context hashes are not saved. No missing log is presented as fresh dynamic verification.',
    'F1 uses the recipient EEF→object unit vector (all three dimensions); F3-G uses the recipient horizontal object→goal unit vector. Dose sign is retained, never folded into the axis. Readouts are de-normalized translation-command units, not executed displacement or physical force. F3-G normalization follows the verified global-min/max adapter, not the obsolete stepwise adapter. Historical radial J vectors/L2 were independently checked against newly derived values.',
    'Full J tensors and δ tensors are in case_vectors.npz. Normalized full-action L2 is dimensionless; raw full-action L2 is explicitly mixed-unit descriptive only, never a physical magnitude. Translation, rotation and gripper norms are separate.',
    '', '## Sampling and intervals',
    'F1: original task→registered-state resampling, 10,000 draws, seed 20260907. Within each dose only frozen complete grids enter; the historical runner omitted a magnitude grid if either sign failed QC. Aggregate first averages doses within state. Source_state_id is the original grouping key, not new certification of independent source trajectories. Earlier demo ancestry is not established here; the F1 intervals are conditional on that registered-state grouping.',
    'F3-G: original task→trajectory blocks, retaining every repeated state/dose/cell together; 10,000 draws, seed 20260907. Point estimates preserve original state weighting (not equal weighting of trajectories with unequal state counts). Across-dose supplementary estimates average within state first. No donor or chunk position is an independent sample. F3-G fit/validation identity stays in case_level.csv; this pooled post-hoc readout is not a new holdout test.',
    'The new raw-unit metrics use shared draws, rather than the old F1 metric-dependent seed offsets. Historical normalized-ratio confidence intervals are not overwritten or relabeled. Task summaries are descriptive; intervals in dose_summary.csv are pointwise 95%, not simultaneous. Signed mean and RMS answer different questions; RMS intervals above zero do not establish a consistent signed direction.',
    '', '## Results', '| Factor | Signed dose (cm) | Cases | State groups | J signed mean [95% CI] | Mean case radial RMS J |', '|---|---:|---:|---:|---|---:|']
    for r in summary:
        if r['metric']!='J_signed_mean':continue
        match=next(s for s in summary if s['factor']==r['factor'] and s['stratum']==r['stratum'] and s['metric']=='J_radial_rms')
        cases=[c for c in CASES if c['factor']==r['factor'] and (r['stratum']=='aggregate_state_mean' or c['signed_dose_cm']==r['stratum'])]
        lines.append(f"| {r['factor']} | {r['stratum']} | {len(cases)} | {r['n_source_groups']} | {r['estimate']:.8g} [{r['ci95_low']:.8g}, {r['ci95_high']:.8g}] | {match['estimate']:.8g} |")
        if r['interpretation']:lines.append('') if False else None
    for factor in ['F1','F3-G']:
        rr=[r for r in CASES if r['factor']==factor];lines += ['',f"{factor}: {len(rr)} complete cases, {len(set(r['state_id'] for r in rr))} states, {len(set(r['source_group'] for r in rr))} registered source groups, {len(set(r['task'] for r in rr))} tasks."]
        agg={r['metric']:r for r in summary if r['factor']==factor and r['stratum']=='aggregate_state_mean'}
        lines.append('Aggregate mean case radial RMS: '+', '.join(f"{k} = {agg[k+'_radial_rms']['estimate']:.8g}" for k in ['J','10','01','11'])+'. These are descriptive magnitude comparisons, not percentages or shares.')
        for r in summary:
            if r['factor']==factor and r['interpretation']:lines.append(f"{r['stratum']}: {r['interpretation']}")
    lines+=['','J measures background dependence/non-additivity of these registered interventions. An unresolved signed mean does not establish independent routes or exact additivity; nonzero case-wise interaction can cancel in the signed aggregate. This is not a source share, natural mediation interaction, or information percentage. No ratio or new significance threshold was introduced.','', '## Provenance','Every input path, byte count and SHA-256 is enumerated in provenance.json. Exclusions and missing grid cells remain in excluded_cases.csv. No historical result was rewritten.']
    (OUT/'analysis.md').write_text('\n'.join(lines)+'\n')
    track(__file__)
    outputs=[dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='provenance.json']
    write('provenance.json',dict(status='COMPLETE_POSTHOC_SAVED_ARRAYS',model_forwards=0,simulator_steps=0,input_files=list(INPUTS.values()),outputs=outputs,missing_evidence=['F1 source-state to independent demo ancestry not certified','B formal per-event context hashes not persisted; self-comparison field excluded as proof'],statistics_seed=20260907,bootstrap_resamples=10000))
    print(json.dumps(dict(cases=len(CASES),closure=closure['max_abs_error'],inputs=len(INPUTS),output=str(OUT))))

if __name__=='__main__':main()

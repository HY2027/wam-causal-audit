"""Matched static K8 extension. Historical assets are read-only; no simulator."""
from wam_causal_audit.paths import resolve as _release_path
import argparse,csv,json,hashlib,time,sys,os,gc,signal,threading,subprocess,traceback
from pathlib import Path
import numpy as np
OUT=Path(__file__).resolve().parent
WORK=Path(_release_path('@WORKSPACE@/joint_strict_consumption_audit_A_work'))
ROOT=Path(_release_path('@DATA@/wam_factor_routing_v5'))
NEW=ROOT/'joint_idm_compute_confirmatory_v1_20260909/joint_new_trajectory_confirmatory'
SRC=ROOT/'joint_compute_mechanism_posthoc_v1_20260909'
GAIN=ROOT/'phase1_evidence_mechanism_posthoc_v1_20260909'
MODEL=Path(_release_path('@DATA@/BadWAM/models/LIQIIIII/badwam-libero-joint-wam'))
GPU=0
UUID='GPU-d9c58e7c-772c-5fe1-b808-d201e93118dd'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(16*1024*1024),b''):h.update(b)
    return h.hexdigest()
def load(p):return json.loads(Path(p).read_text())
def dump(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,allow_nan=False,default=lambda v:v.tolist())+'\n');tmp.replace(p)
def rows(p):return list(csv.DictReader(Path(p).open()))
def table(p,rr):
    p=Path(p);keys=list(dict.fromkeys(k for r in rr for k in r))
    with p.open('w') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rr)
def log(x):
    with (OUT/'forward_ledger.jsonl').open('a') as f:f.write(json.dumps(dict(time=time.time(),**x))+'\n');f.flush()
def occupancy():
    s=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.used','--format=csv,noheader,nounits'],text=True,timeout=10)
    r=[x.strip() for x in next(l for l in s.splitlines() if l.split(',')[0].strip()==str(GPU)).split(',')]
    assert r[1]==UUID and 'RTX 4090' in r[2]
    return dict(timestamp=time.time(),gpu=GPU,uuid=r[1],name=r[2],total_mib=float(r[3]),used_mib=float(r[4]),all_gpu_occupancy=s)
def prepare():
    assert not (OUT/'protocol.json').exists(),'Preserve frozen protocol'
    inputs={}
    def reg(p,h=None):
        p=Path(p)
        if str(p) not in inputs:inputs[str(p)]=dict(sha256=sha(p),bytes=p.stat().st_size)
        if h is not None:assert inputs[str(p)]['sha256']==h,str(p)
    sources=sorted(rows(NEW/'joint_new_trajectory_registry_restore_v2_execution_v3.csv'),key=lambda r:(int(r['task_id']),int(r['trajectory_id'])))
    donors=sorted(rows(NEW/'f3g_donor_registry.csv'),key=lambda r:(int(r['task_id']),int(r['trajectory_id']),float(r['signed_dose_cm'])))
    cells=rows(SRC/'native_k5_four_cell_results.csv');assert len(sources)==50 and len(donors)==200 and len(cells)==1600
    assert len({(r['task_id'],r['trajectory_id']) for r in sources})==50
    paths={};cases=[]
    for r in cells:
        reg(r['action_path'],r['action_file_sha256']);paths[(r['case_id'],r['configuration'])]=r['action_path']
    for s in sources:
        reg(s['observation_path'],s['observation_file_sha256']);reg(s['state_path'],s['state_file_sha256'])
        ds=[d for d in donors if d['candidate_id']==s['candidate_id']];assert len(ds)==4
        assert [float(d['signed_dose_cm']) for d in ds]==[-1,-.5,.5,1]
        for d in ds:
            assert d['donor_valid']=='True' and d['recipient_observation_path']==s['observation_path']
            reg(d['donor_observation_path'],d['donor_observation_sha256']);reg(d['donor_state_path'],d['donor_state_sha256'])
            for k in ['native','K5']:
                subset=[r for r in cells if r['case_id']==d['case_id'] and r['configuration']==k]
                assert sorted(r['intervention'] for r in subset)==['A00','A01','A10','A11']
            cases.append(dict(source=s,donor=d,native_path=paths[d['case_id'],'native'],K5_path=paths[d['case_id'],'K5']))
    for p in [NEW/'joint_new_trajectory_registry.csv',NEW/'joint_new_trajectory_registry_restore_v2_execution_v3.csv',NEW/'f3g_donor_registry.csv',SRC/'native_k5_four_cell_results.csv',SRC/'mechanism_protocol.md',SRC/'mechanism_technical_gate.json',SRC/'asset_audit_and_protocol_manifest.json',GAIN/'joint_scaling_analysis_protocol.md',GAIN/'joint_scaling_fold_registry.csv',GAIN/'joint_scaling_freeze_manifest.json',GAIN/'joint_scaling_parameters.csv',GAIN/'joint_scaling_residual_statistics.csv',WORK/'phase1_scaling_analysis.py',WORK/'run_joint_compute_mechanism_posthoc.py',WORK/'run_c_early_stop.py',WORK/'run_experiment_a.py',MODEL/'dataset_stats.json']:reg(p)
    reg(MODEL/'model.pt','7cd0af01f4a51838510265685fa533836c6c9c29e0c37e3f7235d14ac676c928')
    gate=load(SRC/'asset_audit_and_protocol_manifest.json')['technical_case_ids'];assert len(gate)==5
    plan=[]
    def add(stage,cid,k,arm):plan.append(dict(index=len(plan)+1,stage=stage,case_id=cid,K=k,arm=arm))
    for cid in gate:
        for k in [10,5]:
            for a in ['recipient_capture','donor_capture','A00','A10','A01','A11']:add('gate',cid,k,a)
    for s in sources:
        sid=s['candidate_id'];add('formal',sid,8,'recipient_capture');add('formal',sid,8,'A00')
        for c in [c for c in cases if c['source']['candidate_id']==sid]:
            for a in ['donor_capture','A10','A01','A11']:add('formal',c['donor']['case_id'],8,a)
    endpoint=0
    for s in sources:
        for cid in [s['candidate_id']]+[c['donor']['case_id'] for c in cases if c['source']['candidate_id']==s['candidate_id']]:
            ks=[10,8,5];order=ks[endpoint%3:]+ks[:endpoint%3]
            for k in order:add('timing',cid,k,'native_own_call_cache')
            endpoint+=1
    assert len(plan)==1710
    table(OUT/'forward_plan.csv',plan);dump(OUT/'cases.json',cases);dump(OUT/'input_manifest.json',inputs)
    protocol=dict(status='FROZEN_BEFORE_NEW_K8_OUTPUTS',identity='posthoc matched-budget extension of the previously studied confirmation cohort',created_unix=time.time(),cohort=dict(trajectories=50,donors=200,tasks=list(range(5,10)),phase='TRANSPORT',factor='F3-G',doses_cm=[-1,-.5,.5,1],registry=str(NEW/'joint_new_trajectory_registry_restore_v2_execution_v3.csv')),budgets=[10,8,5],action_steps=10,gate_case_ids=gate,gate_standard='normalized action arrays exact numeric equality; historical zero-error standard; own-path four cells and endpoint native references',gate_calls=60,formal_K8_calls=900,timing_calls=750,total_calls=1710,cache='each endpoint capture freshly generates own K-layer schedules; strict experiment consumes named recipient/donor caches; production timing regenerates own stop cache every call; no production cross-call reuse',context='recipient text/proprio in strict cells; assert endpoint context equality for F3-G',cells={'A00':['recipient','recipient'],'A10':['donor','recipient'],'A01':['recipient','donor'],'A11':['donor','donor']},gain='exact historical D10/D01/D11 radial-vector fit, shared zero-intercept lambda, five saved trajectory folds, no clipping, separate K8 and K5 fits',bootstrap=dict(replicates=10000,seed=20260909,procedure='historical task->trajectory resampling; refit cross-fold gains in every draw; pointwise percentile CI'),aggregation='equal source, donor, baseline contrast and predicted position; 32 positions are not independent trajectories',primary_future_increment='A11-A10 (current donor); also report A01-A00 (current recipient)',timing='250 unique endpoints per budget; deterministic cyclic budget order by stable endpoint index; production uninstrumented native vs own-stop-cache shortened entry; preparation excluded from denoising timer but separate inclusive outer time recorded; no closed-loop cumulative time claim',resource=dict(gpu=GPU,uuid=UUID,whole_card_fraction=.95,extra_margin_mib=0,minimum_available_mib=30000,sampling_seconds=1,call_timeout_seconds=600,no_external_process_modification=True),no_simulation=True,no_new_closed_loop=True,new_K8_required=True)
    protocol['code_hashes']={p.name:sha(p) for p in [Path(__file__),OUT/'budget_backend.py',OUT/'analyze.py']}
    protocol['plan_sha256']=sha(OUT/'forward_plan.csv');protocol['inputs_sha256']=sha(OUT/'input_manifest.json');dump(OUT/'protocol.json',protocol)
    dump(OUT/'reproduction_gate.json',dict(status='NOT_RUN',new_K8_authorized_by_gate=False))
    print(json.dumps(dict(cases=200,sources=50,calls=1710,gate=gate)),flush=True)

def run():
    import fcntl
    lock=(OUT/'worker.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    p=load(OUT/'protocol.json');assert not (OUT/'forward_ledger.jsonl').exists(),'No implicit restart'
    amendment=load(OUT/'preexecution_code_amendment.json')
    assert amendment['protocol_sha256']==sha(OUT/'protocol.json')
    for file,h in amendment['code_hashes'].items():assert sha(OUT/file)==h,file
    assert sha(OUT/'forward_plan.csv')==p['plan_sha256']
    for file,r in load(OUT/'input_manifest.json').items():assert sha(file)==r['sha256'],file
    o=occupancy();dump(OUT/'resource_admission.json',o);print(json.dumps(o),flush=True)
    if .95*o['total_mib']-o['used_mib']<30000:dump(OUT/'run_status.json',dict(status='RESOURCE_BUSY',attempts=0));return
    stopped=threading.Event();runner=None;original=None;attempts=0;completed=0;begin=time.time();evidence=[];checks=[]
    def abort(*args):raise RuntimeError('RESOURCE_ABORT_OR_CALL_TIMEOUT')
    signal.signal(signal.SIGTERM,abort);signal.signal(signal.SIGALRM,abort)
    def monitor():
        while not stopped.wait(1):
            try:
                o=occupancy()
                with (OUT/'resource_samples.jsonl').open('a') as f:f.write(json.dumps(o)+'\n')
                if o['used_mib']>.95*o['total_mib']:raise RuntimeError('WHOLE_CARD_95_PERCENT')
            except Exception as e:
                dump(OUT/'resource_abort.json',dict(error=str(e),timestamp=time.time()));os.kill(os.getpid(),signal.SIGTERM);return
    threading.Thread(target=monitor,daemon=True).start()
    try:
        sys.path.insert(0,str(WORK));import run_experiment_a as A;import run_c_early_stop as C;import budget_backend as B;import torch
        torch.set_num_threads(16);log(dict(event='LOAD_STARTED'));runner=A.make_capture('joint',GPU);log(dict(event='LOAD_COMPLETED',seconds=time.time()-begin))
        original=runner.model.mot._build_expert_attention_io;weight=A.model_weight_hash(runner.model)
        import importlib.metadata as md
        dump(OUT/'runtime_identity.json',dict(interpreter=sys.executable,torch=torch.__version__,cuda=torch.version.cuda,transformers=md.version('transformers'),numpy=np.__version__,device=UUID,weight_hash=weight,model_dtype=str(next(runner.model.parameters()).dtype),world_scheduler=repr(runner.model.infer_video_scheduler),action_scheduler=repr(runner.model.infer_action_scheduler)))
        plan=rows(OUT/'forward_plan.csv')
        def call(stage,cid,k,arm,fn):
            nonlocal attempts,completed
            expected=plan[attempts];assert (stage,cid,k,arm)==(expected['stage'],expected['case_id'],int(expected['K']),expected['arm'])
            attempts+=1;t=time.time();log(dict(event='STARTED',index=attempts,stage=stage,case_id=cid,K=k,arm=arm));signal.alarm(600)
            try:
                v=fn();assert np.isfinite(v[0].numpy()).all();assert runner.model.mot._build_expert_attention_io==original
                completed+=1;log(dict(event='COMPLETED',index=attempts,stage=stage,case_id=cid,K=k,arm=arm,seconds=time.time()-t,hook_cleaned=True));return v
            except BaseException as e:log(dict(event='FAILED',index=attempts,stage=stage,case_id=cid,K=k,arm=arm,seconds=time.time()-t,error=str(e)));raise
            finally:signal.alarm(0)
        def prep(path,ins):return runner._prepared(A.load_npz(Path(path)),ins)
        def capture(item,seed,k):
            if k==10:
                a,ctl,meta=A.capture_source(runner,item,seed);schedule={t:[ctl.cache[t,l] for l in range(30)] for t in range(10)};return a,schedule,meta
            B.K=k;return B.capture_k5_source(runner,item,seed)
        def strict(item,seed,k,cs,fs):
            if k==10:return A.strict_infer(runner,item,seed,B.schedule_mapping(cs),B.schedule_mapping(fs))
            B.K=k;return B.strict_k5(runner,item,seed,cs,fs)
        def compare(a,b,label):
            x=a.numpy() if hasattr(a,'numpy') else a;y=np.asarray(b);ok=np.array_equal(x,y)
            r=dict(label=label,exact=bool(ok),max_abs_error=float(np.max(np.abs(x.astype(float)-y.astype(float)))),dtype=str(x.dtype),shape=list(x.shape));checks.append(r)
            dump(OUT/'numerical_checks.json',checks);assert ok,label
        cases=load(OUT/'cases.json');byid={c['donor']['case_id']:c for c in cases}
        for cid in p['gate_case_ids']:
            c=byid[cid];s=c['source'];seed=int(s['policy_seed']);rp=prep(s['observation_path'],s['instruction']);dp=prep(c['donor']['donor_observation_path'],s['instruction'])
            assert torch.equal(rp['context'],dp['context']) and torch.equal(rp['context_mask'],dp['context_mask'])
            for k in [10,5]:
                z=dict(np.load(c['native_path' if k==10 else 'K5_path']));ra,rs,rm=call('gate',cid,k,'recipient_capture',lambda:capture(rp,seed,k));da,ds,dm=call('gate',cid,k,'donor_capture',lambda:capture(dp,seed,k))
                compare(ra,z[f'recipient_K{k}'],f'gate/{cid}/K{k}/recipient');compare(da,z[f'donor_K{k}'],f'gate/{cid}/K{k}/donor')
                save={'recipient':ra.numpy(),'donor':da.numpy()}
                for arm,cs,fs in [('A00',rs,rs),('A10',ds,rs),('A01',rs,ds),('A11',ds,ds)]:
                    act,meta=call('gate',cid,k,arm,lambda:strict(rp,seed,k,cs,fs));compare(act,z[arm],f'gate/{cid}/K{k}/{arm}');save[arm]=act.numpy();evidence.append(dict(stage='gate',case_id=cid,K=k,arm=arm,meta=meta))
                (OUT/'gate_actions').mkdir(exist_ok=True);np.savez_compressed(OUT/'gate_actions'/f'{cid}__K{k}.npz',**save)
                del rs,ds,cs,fs;gc.collect();torch.cuda.empty_cache()
        dump(OUT/'reproduction_gate.json',dict(status='PASS_EXACT',cases=5,policy_calls=60,array_comparisons=len(checks),maximum_absolute_error=max(x['max_abs_error'] for x in checks),new_K8_authorized_by_gate=True,check_file='numerical_checks.json'))
        dump(OUT/'consumption_checks.json',evidence);print('REPRODUCTION_GATE_PASS',flush=True)
        sources=list({c['source']['candidate_id']:c['source'] for c in cases}.values());(OUT/'actions').mkdir(exist_ok=True)
        for s in sources:
            sid=s['candidate_id'];seed=int(s['policy_seed']);rp=prep(s['observation_path'],s['instruction'])
            ra,rs,rm=call('formal',sid,8,'recipient_capture',lambda:capture(rp,seed,8));a00,m00=call('formal',sid,8,'A00',lambda:strict(rp,seed,8,rs,rs));compare(a00,ra.numpy(),f'K8/{sid}/same_value_A00')
            evidence.extend([dict(stage='formal',case_id=sid,K=8,arm='recipient_capture',meta=rm),dict(stage='formal',case_id=sid,K=8,arm='A00',meta=m00)])
            for c in [c for c in cases if c['source']['candidate_id']==sid]:
                cid=c['donor']['case_id'];dp=prep(c['donor']['donor_observation_path'],s['instruction']);assert torch.equal(rp['context'],dp['context']) and torch.equal(rp['context_mask'],dp['context_mask'])
                da,ds,dm=call('formal',cid,8,'donor_capture',lambda:capture(dp,seed,8));save={'recipient_K8':ra.numpy(),'donor_K8':da.numpy(),'A00':a00.numpy()}
                evidence.append(dict(stage='formal',case_id=cid,K=8,arm='donor_capture',meta=dm,recipient_context_exact=True,seed=seed))
                for arm,cs,fs in [('A10',ds,rs),('A01',rs,ds),('A11',ds,ds)]:
                    act,meta=call('formal',cid,8,arm,lambda:strict(rp,seed,8,cs,fs));save[arm]=act.numpy();evidence.append(dict(stage='formal',case_id=cid,K=8,arm=arm,current_source='donor' if arm in ['A10','A11'] else 'recipient',future_source='donor' if arm in ['A01','A11'] else 'recipient',seed=seed,meta=meta))
                compare(save['A11'],save['donor_K8'],f'K8/{cid}/natural_A11');np.savez_compressed(OUT/'actions'/f'{cid}.npz',**save)
                dump(OUT/'consumption_checks.json',evidence);del ds,cs,fs,dp;gc.collect();torch.cuda.empty_cache()
            del rs,rp;gc.collect();torch.cuda.empty_cache();print(f'FORMAL {sid} calls={completed}',flush=True)
        timings=[];endpoint=0
        for s in sources:
            sid=s['candidate_id'];seed=int(s['policy_seed']);sc=[c for c in cases if c['source']['candidate_id']==sid]
            endpoints=[(sid,s['observation_path'],sc[0],'recipient')]+[(c['donor']['case_id'],c['donor']['donor_observation_path'],c,'donor') for c in sc]
            for cid,path,c,role in endpoints:
                ks=[10,8,5];order=ks[endpoint%3:]+ks[:endpoint%3];endpoint+=1
                for k in order:
                    t=time.perf_counter();item=prep(path,s['instruction']);torch.cuda.synchronize(runner.model.device);prep_s=time.perf_counter()-t
                    fn=(lambda:C.native_timed(runner,item,seed)) if k==10 else (lambda:C.early_stop_infer(runner,item,seed,k))
                    t=time.perf_counter();v=call('timing',cid,k,'native_own_call_cache',fn);torch.cuda.synchronize(runner.model.device);outer=time.perf_counter()-t
                    action,diag=v[:2];reference=OUT/'actions'/f"{c['donor']['case_id']}.npz" if k==8 else Path(c['native_path' if k==10 else 'K5_path'])
                    with np.load(reference) as z:compare(action,z[f'{role}_K{k}'],f'timing/{cid}/K{k}/reproduction')
                    timings.append(dict(candidate_id=sid,case_id=cid,task_id=int(s['task_id']),trajectory_id=int(s['trajectory_id']),endpoint=role,K=k,preparation_seconds=prep_s,policy_seconds=diag['latency_seconds'],outer_inference_seconds=outer,inclusive_seconds=prep_s+outer,world_steps=diag['world_branch_steps'],action_steps=diag['action_branch_steps'],peak_memory_bytes=diag['peak_memory_bytes'],reference_path=str(reference),action_sha256=hashlib.sha256(action.numpy().tobytes()).hexdigest()))
                    table(OUT/'timing_calls.csv',timings);del v,item;gc.collect();torch.cuda.empty_cache()
            print(f'TIMING {sid} calls={completed}',flush=True)
        assert attempts==completed==1710;assert A.model_weight_hash(runner.model)==weight
        dump(OUT/'run_status.json',dict(status='COMPLETE',attempts=attempts,completed=completed,seconds=time.time()-begin,weights_unchanged=True,simulator_steps=0,closed_loop_branches=0))
    except BaseException:
        dump(OUT/'run_status.json',dict(status='BLOCKED_TECHNICAL',attempts=attempts,completed=completed,seconds=time.time()-begin,traceback=traceback.format_exc()));print(traceback.format_exc(),flush=True)
        if not load(OUT/'reproduction_gate.json').get('new_K8_authorized_by_gate'):dump(OUT/'reproduction_gate.json',dict(status='FAIL_OR_BLOCKED',new_K8_authorized_by_gate=False,details='run_status.json',checks='numerical_checks.json'))
    finally:
        stopped.set();dump(OUT/'cleanup.json',dict(hook_restored=(runner.model.mot._build_expert_attention_io==original) if runner is not None and original is not None else None))
        if runner is not None:del runner;gc.collect();torch.cuda.empty_cache()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run']);a=p.parse_args();prepare() if a.mode=='prepare' else run()

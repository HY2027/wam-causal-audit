"""Bounded STATIC_INPUT_TECHNICAL_VALIDATION supervisor and worker."""
from wam_causal_audit.paths import resolve as _release_path
import argparse,csv,fcntl,hashlib,importlib.util,json,os,secrets,signal,subprocess,sys,time,traceback
from contextlib import contextmanager
from pathlib import Path
import psutil

STAGE='STATIC_INPUT_TECHNICAL_VALIDATION'
ARMS=['native_repeat_1','native_repeat_2','capture_readonly','N-REPLAY','D','D-SV-L','D-SV-E','R-L','R-E','R-ALL']
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,x):
    p=Path(p); tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2)+'\n');os.replace(tmp,p)
def append(p,x):
    with Path(p).open('a') as f:f.write(json.dumps(x)+'\n');f.flush();os.fsync(f.fileno())
def lines(p):return [json.loads(x) for x in Path(p).read_text().splitlines()] if Path(p).exists() else []
def gpu_query():
    def query(args):
        r=subprocess.run(['nvidia-smi',*args,'--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10,check=True)
        return list(csv.reader(r.stdout.splitlines(),skipinitialspace=True))
    devices={r[1]:{'index':int(r[0]),'uuid':r[1],'name':r[2],'total_mib':float(r[3]),'used_mib':float(r[4]),'free_mib':float(r[5])} for r in query(['--query-gpu=index,uuid,name,memory.total,memory.used,memory.free'])}
    processes=[{'uuid':r[0],'pid':int(r[1]),'used_mib':float(r[2])} for r in query(['--query-compute-apps=gpu_uuid,pid,used_memory'])]
    return devices,processes
def resource_reason(elapsed,total_elapsed,call_elapsed,used,total,limits):
    if elapsed>=limits['wall_seconds_per_model'] or total_elapsed>=limits['total_model_wall_seconds']:return 'WALL_LIMIT'
    if call_elapsed is not None and call_elapsed>=limits['call_seconds']:return 'CALL_LIMIT'
    if used>total*limits['gpu_memory_fraction']:return 'MEMORY_LIMIT'
    return None
def blocking_processes(processes,uuid,registry,own_pids=()):
    allowed=set(registry.get('occupancy_amendment',{}).get('allowed_external_pids_by_uuid',{}).get(uuid,[]))
    return [x for x in processes if x['uuid']==uuid and x['pid'] not in allowed and x['pid'] not in own_pids]
def remaining_task_mib(device,fraction):
    return max(0.,min(device['free_mib'],fraction*device['total_mib']-device['used_mib']))
def resource_admission_detail(device,apps,registry):
    fraction=registry['occupancy_amendment']['whole_device_fraction']
    allowed=registry['occupancy_amendment']['allowed_external_pids_by_uuid'].get(device['uuid'],[])
    blocked=blocking_processes(apps,device['uuid'],registry)
    return {'total_mib':device['total_mib'],'whole_used_mib':device['used_mib'],
        'whole_limit_mib':fraction*device['total_mib'],'reserved_fraction_mib':(1-fraction)*device['total_mib'],
        'additional_registered_margin_mib':None,'additional_margin_status':'NOT_SEPARATELY_REGISTERED',
        'nominal_task_headroom_mib':remaining_task_mib(device,fraction),'admitted_task_mib':0 if blocked else remaining_task_mib(device,fraction),
        'allowed_external_pids':allowed,'process_matching':[dict(x,allowlist_match=x['pid'] in allowed) for x in apps if x['uuid']==device['uuid']],
        'rejection_condition':'UNAUTHORIZED_COEXISTING_PROCESS' if blocked else 'NO_SHARED_MEMORY_HEADROOM' if remaining_task_mib(device,fraction)<=0 else None,
        'pid_identity_scope':'registered PID and GPU UUID; historical executable/start-time identity not registered'}
def reserve(run,model,arm,endpoint):
    r=read(run/'registry.json');campaign=Path(r['campaign_ledger'])
    with campaign.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        entries=lines(campaign)
        starts=[x for x in entries if x['event']=='CALL_STARTED']
        if len(starts)>=80 or sum(x['model']==model for x in starts)>=20:raise RuntimeError('CUMULATIVE_CALL_LIMIT')
        if any(x['model']==model and x['arm']==arm and x['endpoint']==endpoint for x in starts):raise RuntimeError('NO_RETRY_ALREADY_STARTED')
        row={'event':'CALL_STARTED','run':str(run),'model':model,'arm':arm,'endpoint':endpoint,'time':time.time(),'pid':os.getpid()}
        append(campaign,row);append(run/'actual_calls.jsonl',row)
        save(run/model/'active_call.json',row)
        return row
def authorize(run,model,token):
    r=read(run/'registry.json')
    if r['stage']!=STAGE or hashlib.sha256(token.encode()).hexdigest()!=r['token_sha256']:raise PermissionError('AUTHORIZATION_SCOPE_TOKEN')
    if sha(run/'static_input_technical_allowlist.json')!=r['allowlist_sha256']:raise PermissionError('ALLOWLIST_DRIFT')
    if sha(__file__)!=r['runtime_sha256']:raise PermissionError('RUNTIME_CODE_DRIFT')
    if os.environ.get('T1_REGISTRY_SHA256')!=sha(run/'registry.json'):raise PermissionError('REGISTRY_IDENTITY')
    if model not in r['devices']:raise PermissionError('MODEL_SCOPE')
    if any(r['other_stages'].values()):raise PermissionError('OTHER_STAGE_ENABLED')
    return r
@contextmanager
def restore_methods(model):
    """Catch normal/error/signal exits; SIGKILL is explicitly unverifiable."""
    baseline={name:getattr(model.mot,name) for name in ('_mixed_attention','_build_expert_attention_io','forward_action_with_video_cache') if hasattr(model.mot,name)}
    state={'attempted':False,'success':False}
    try:yield state
    finally:
        state['attempted']=True
        for name,value in baseline.items():setattr(model.mot,name,value)
        state['success']=all(getattr(model.mot,n)==v for n,v in baseline.items())

def worker(run,model,token):
    r=authorize(run,model,token); dest=run/model;dest.mkdir(exist_ok=True)
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(InterruptedError('RESOURCE_STOP_SIGNAL')))
    cleanup={'attempted':False,'success':None,'reason':'NO_MODEL_OR_HOOK_INSTALLED'}
    try:
        # Parent rehashes all checkpoint/dependency files before launch; child validates receipt.
        receipt=read(dest/'preload_identity.json')
        if receipt['registry_sha256']!=sha(run/'registry.json'):raise RuntimeError('PRELOAD_RECEIPT')
        devices,apps=gpu_query();dev=devices[r['devices'][model]['uuid']]
        if dev['name']!=r['devices'][model]['name']:raise RuntimeError('HARDWARE_DRIFT')
        if blocking_processes(apps,dev['uuid'],r,{os.getpid()}):raise RuntimeError('RESOURCE_BUSY')
        import torch
        # Install allocator cap before model construction. Whole-device NVML monitor
        # separately covers CUDA context/custom kernels and other processes' growth.
        current=gpu_query()[0][dev['uuid']]
        available=min(receipt['task_memory_cap_mib'],remaining_task_mib(current,r['occupancy_amendment']['whole_device_fraction']))
        if available<=0:raise RuntimeError('NO_SHARED_MEMORY_HEADROOM')
        torch.cuda.set_per_process_memory_fraction(available/current['total_mib'],device=0)
        save(dest/'allocator_cap.json',{'task_cap_mib':available,'whole_device_fraction':.95,'device':current,'set_before_model_load':True})
        backend_file=Path(r['backend_file'])
        if sha(backend_file)!=r['backend_sha256']:raise RuntimeError('BACKEND_DRIFT')
        spec=importlib.util.spec_from_file_location('t1_parent_binding',backend_file);b=importlib.util.module_from_spec(spec);sys.modules[spec.name]=b;spec.loader.exec_module(b)
        # New-run scoped gate only; immutable T0 source is never edited.
        b.runtime_gate=lambda:authorize(run,model,token)
        class AuditedResolver(b.Resolver):
            def resolve(self,event,current,start,stop):
                if self.arm=='R-CTRL':raise PermissionError('CONTENT_CONTROL_NOT_AUTHORIZED')
                own=self.banks['native'].get(self.ids['native'],event).to(device=current.device)
                if own.shape!=current.shape or own.dtype!=current.dtype:raise RuntimeError('BACKGROUND_LAYOUT')
                out=super().resolve(event,own,start,stop)
                delta=out[:,start:stop].float()-own[:,start:stop].float()
                self.records[-1].update(shape=list(out.shape),dtype=str(out.dtype),device=str(out.device),
                    attempted_write=True,changed_from_native=not bool(torch.equal(out[:,start:stop],own[:,start:stop])),
                    delta_from_native_l2=float(torch.linalg.vector_norm(delta)),
                    non_target_exact=bool(torch.equal(out[:,:start],own[:,:start]) and torch.equal(out[:,stop:],own[:,stop:])))
                return out
        b.Resolver=AuditedResolver
        original_joint=b.Backend._joint
        def joint_with_consumption(self,prepared,seed,identity,resolver):
            mixed=self.runner.model.mot._mixed_attention;consumed=[]
            def actual(*args,**kwargs):
                k=kwargs.get('k_cat',args[1] if len(args)>1 else None)
                v=kwargs.get('v_cat',args[2] if len(args)>2 else None)
                if k is None or v is None:raise RuntimeError('JOINT_MIXED_SIGNATURE')
                index=len(consumed)//2;step,layer=divmod(index,30)
                if step>=10:raise RuntimeError('EXTRA_JOINT_MIXED_EVENT')
                for component,value in (('k',k),('v',v)):
                    consumed.append({'event':[step,layer,component],'full_video_hash':b.tensor_hash(value[:,:294]),
                        'target_hash':b.tensor_hash(value[:,98:294]),'status':'ACTUAL_MIXED_ATTENTION_INPUT'})
                return mixed(*args,**kwargs)
            with b.patched(self.runner.model.mot,'_mixed_attention',actual):
                action,snapshot,writes=original_joint(self,prepared,seed,identity,resolver)
            if len(consumed)!=600:raise RuntimeError('MISSING_JOINT_CONSUMPTION')
            write_map={(x['step'],x['layer'],x['component']):x for x in writes}
            for row in consumed:
                key=tuple(row['event'])
                expected=snapshot.hashes[key] if resolver is None else write_map[key]['written_hash']
                observed=row['full_video_hash'] if resolver is None else row['target_hash']
                if expected!=observed:raise RuntimeError('JOINT_WRITTEN_VALUE_NOT_CONSUMED')
            return action,snapshot,{'writes':writes,'consumption':consumed}
        b.Backend._joint=joint_with_consumption
        loading=time.monotonic();backend=b.Backend.load(model,0)
        save(dest/'load.json',{'seconds':time.monotonic()-loading,'pid':os.getpid()})
        import torch,numpy as np
        allow=[x for x in read(run/'static_input_technical_allowlist.json')['endpoints'] if x['model']==model]
        observations={};ids={};snapshots={};native={};disrupted={};checks=[]
        for item in allow:
            ep=item['endpoint'];p=item['input']['path']
            if sha(p)!=item['input']['sha256']:raise RuntimeError('INPUT_DRIFT')
            with np.load(p,allow_pickle=False) as z:observations[ep]={k:z[k].copy() for k in z.files}
            ids[ep]=b.Identity(model,item['case_id'],item['source_group'],ep,'native_capture_'+str(ep),r['configuration_hashes'][model],item['input']['sha256'])
        original_gate=b.runtime_gate
        for arm in ARMS:
            for ep in (0,1):
                item=next(x for x in allow if x['endpoint']==ep)
                # identity checked before charging; inference/preprocessing failures do count.
                if sha(item['input']['path'])!=item['input']['sha256']:raise RuntimeError('INPUT_DRIFT')
                started=reserve(run,model,arm,ep);begin=time.monotonic();status='FAILED';events=None;state={}
                counts={}
                def profiler(frame,event,arg):
                    if event=='call' and frame.f_code.co_filename.startswith((_release_path('@DATA@/BadWAM/src/'),_release_path('@WORKSPACE@/ImageWAM/src/'))):
                        name=frame.f_code.co_name
                        if name in ('forward','pre_dit','_predict_action_noise_with_cache','_predict_noise','_predict_video_noise'):
                            key=frame.f_code.co_filename+':'+name;counts[key]=counts.get(key,0)+1
                try:
                    with restore_methods(backend.runner.model) as state:
                        sys.setprofile(profiler)
                        policy=getattr(backend.runner,'policy',None)
                        if policy is not None and hasattr(policy,'reset'):policy.reset()
                        prepared=backend.prepare(observations[ep],item['instruction'])
                        if arm.startswith('native_repeat'):
                            action=backend.native(prepared,item['seed'])
                            if arm=='native_repeat_1':native[ep]=action.detach().cpu().clone()
                            reference=native[ep]
                        elif arm=='capture_readonly':
                            action,snapshot,events=backend.infer(prepared,item['seed'],ids[ep]);snapshots[ep]=snapshot
                            torch.save(snapshot._events,dest/f'capture_endpoint{ep}.pt');reference=native[ep]
                        else:
                            early=range(5,8) if model=='imagewam' else range(5)
                            late=range(22,25) if model=='imagewam' else range(25,30)
                            identity=b.Identity(model,ids[ep].pair_id,ids[ep].source_id,ep,arm+'_'+str(ep),ids[ep].config_hash,ids[ep].input_hash)
                            resolver=b.Resolver(identity,arm,{'native':snapshots[ep],'anchor':snapshots[0]},
                                {'native':ids[ep],'anchor':ids[0]},early,late,range(25 if model=='imagewam' else 30))
                            action,_,events=backend.infer(prepared,item['seed'],identity,resolver)
                            if arm=='D':disrupted[ep]=action.detach().cpu().clone()
                            reference=disrupted[ep] if arm.startswith('D-SV') else native[ep] if arm in ('N-REPLAY','R-ALL') else None
                        a=action.detach().cpu();torch.save(a,dest/f'{arm}_endpoint{ep}.pt')
                        if not bool(torch.isfinite(a).all()):raise RuntimeError('NONFINITE_OUTPUT')
                        exact=None if reference is None else bool(torch.equal(a,reference))
                        error=None if reference is None else float((a-reference).abs().max())
                        if events is not None:save(dest/f'{arm}_endpoint{ep}_events.json',events)
                        checks.append({'arm':arm,'endpoint':ep,'bit_exact':exact,'max_abs':error,'action_hash':b.tensor_hash(a)})
                        if exact is False:raise RuntimeError('FROZEN_BIT_EXACT_CHECK_FAILED')
                        status='COMPLETED'
                finally:
                    sys.setprofile(None)
                    cleanup=state
                    row={**started,'event':'CALL_FINISHED','status':status,'duration_seconds':time.monotonic()-begin,'cleanup':state,'internal_python_call_counts':counts}
                    append(run/'actual_calls.jsonl',row);append(Path(r['campaign_ledger']),row)
                    save(dest/'active_call.json',{'event':'IDLE'});save(dest/'comparisons.json',checks);save(dest/'hook_cleanup.json',state)
                if not state.get('success'):raise RuntimeError('HOOK_CLEANUP_FAILED')
        save(dest/'result.json',{'status':'COMPLETED_REGISTERED_CALLS','checks':checks,'scientific_status':'NOT_ASSESSED'})
    except BaseException as exc:
        save(dest/'failure.json',{'exception':repr(exc),'traceback':traceback.format_exc(),'cleanup':cleanup})
        raise
    finally:save(dest/'worker_exit.json',{'time':time.time(),'pid':os.getpid(),'hook_cleanup':cleanup})

def stop_owned(process,identities):
    """PID creation-time ownership; never signal an unrelated reused PID."""
    live=[]
    for pid,created in identities.items():
        try:
            p=psutil.Process(pid)
            if p.create_time()==created:live.append(p);p.terminate()
        except psutil.NoSuchProcess:pass
    _,alive=psutil.wait_procs(live,timeout=5)
    for p in alive:
        try:
            if p.create_time()==identities[p.pid]:p.kill()
        except psutil.NoSuchProcess:pass
    return {'terminated':[p.pid for p in live],'force_killed':[p.pid for p in alive],
            'cleanup':'UNCONFIRMED_FOR_FORCE_KILL' if alive else 'CHECK_WORKER_EXIT_RECORD'}

def supervise(run,token):
    r=read(run/'registry.json');campaign=Path(r['campaign_ledger'])
    # One supervisor for the whole authorization lineage, including continuations.
    with campaign.with_suffix('.supervisor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        history=lines(campaign)
        if sum(x['event']=='MODEL_STARTED' for x in history)!=sum(x['event']=='MODEL_FINISHED' for x in history):
            raise RuntimeError('UNRECONCILED_PRIOR_MODEL_WALL_BUDGET_NO_RESET')
        for model in r['order']:
            authorize(run,model,token)
            dest=run/model;dest.mkdir(exist_ok=True)
            previous=lines(campaign)
            prior_attempts=sum(x['model']==model and x['event']=='MODEL_STARTED' for x in previous)
            retry=r.get('explicit_load_retry',{}).get(model)
            retry_ok=bool(retry and prior_attempts==retry['expected_prior_attempts'] and not any(x['model']==model and x['event']=='CALL_STARTED' for x in previous))
            if prior_attempts and not retry_ok:
                save(dest/'admission.json',{'status':'BLOCKED_NO_AUTOMATIC_RETRY'});continue
            devices,apps=gpu_query();expected=r['devices'][model];dev=devices.get(expected['uuid'])
            if not dev or dev['name']!=expected['name']:
                save(dest/'admission.json',{'status':'DEVICE_IDENTITY_MISMATCH'});continue
            occupied=blocking_processes(apps,dev['uuid'],r)
            save(dest/'resource_startup.json',{'device':dev,'processes':[x for x in apps if x['uuid']==dev['uuid']],'blocking_processes':occupied,'time':time.time(),'admission_detail':resource_admission_detail(dev,apps,r)})
            if occupied:
                save(dest/'admission.json',{'status':'RESOURCE_BUSY','processes':occupied});continue
            if remaining_task_mib(dev,r['occupancy_amendment']['whole_device_fraction'])<=0:
                save(dest/'admission.json',{'status':'RESOURCE_BUSY','reason':'NO_SHARED_MEMORY_HEADROOM'});continue
            # No process is launched before monitoring's device/process query succeeds.
            try:
                for row in r['preload_files'][model]:
                    if sha(row['path'])!=row['sha256']:raise RuntimeError('PRELOAD_FILE_IDENTITY:'+row['path'])
            except Exception as exc:
                save(dest/'admission.json',{'status':'PRELOAD_IDENTITY_BLOCKED','reason':repr(exc)});continue
            # Recheck once after lengthy hashes; this is launch race prevention, not waiting.
            devices,apps=gpu_query();dev=devices[dev['uuid']]
            if blocking_processes(apps,dev['uuid'],r):save(dest/'admission.json',{'status':'RESOURCE_BUSY'});continue
            task_cap=remaining_task_mib(dev,r['occupancy_amendment']['whole_device_fraction'])
            if task_cap<=0:save(dest/'admission.json',{'status':'RESOURCE_BUSY','reason':'NO_SHARED_MEMORY_HEADROOM'});continue
            save(dest/'preload_identity.json',{'registry_sha256':sha(run/'registry.json'),'verified_files':r['preload_files'][model],'task_memory_cap_mib':task_cap})
            elapsed_prior=sum(x['seconds'] for x in previous if x['event']=='MODEL_FINISHED')
            model_elapsed_prior=sum(x['seconds'] for x in previous if x['event']=='MODEL_FINISHED' and x['model']==model)
            if model_elapsed_prior>=r['limits']['wall_seconds_per_model']:save(dest/'admission.json',{'status':'MODEL_WALL_LIMIT'});continue
            if elapsed_prior>=r['limits']['total_model_wall_seconds']:save(dest/'admission.json',{'status':'TOTAL_WALL_LIMIT'});continue
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=dev['uuid'],T1_REGISTRY_SHA256=sha(run/'registry.json'),T1_TOKEN=token)
            begin=time.monotonic();append(campaign,{'event':'MODEL_STARTED','model':model,'run':str(run),'time':time.time()})
            owned={};reason=None;process=None
            try:
                with (dest/'worker.log').open('w') as log:
                    process=subprocess.Popen([r.get('interpreters',{}).get(model,sys.executable),__file__,'worker','--run',str(run),'--model',model],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    root=psutil.Process(process.pid);owned[root.pid]=root.create_time()
                    while process.poll() is None:
                        try:
                            for child in root.children(recursive=True):owned[child.pid]=child.create_time()
                            devices,apps=gpu_query()
                            if dev['uuid'] not in devices:raise RuntimeError('DEVICE_DISAPPEARED')
                            valid_owned=set()
                            for pid,created in owned.items():
                                try:
                                    if psutil.Process(pid).create_time()==created:valid_owned.add(pid)
                                except psutil.NoSuchProcess:pass
                            used={u:sum(x['used_mib'] for x in apps if x['uuid']==u and x['pid'] in valid_owned) for u in devices}
                            active=read(dest/'active_call.json') if (dest/'active_call.json').exists() else {}
                            call_elapsed=time.time()-active['time'] if active.get('event')=='CALL_STARTED' else None
                            elapsed=time.monotonic()-begin
                            reason=resource_reason(model_elapsed_prior+elapsed,elapsed_prior+elapsed,call_elapsed,used[dev['uuid']],dev['total_mib'],r['limits'])
                            if used[dev['uuid']]>task_cap:reason='REMAINING_SPACE_TASK_MEMORY_LIMIT'
                            if devices[dev['uuid']]['used_mib']>devices[dev['uuid']]['total_mib']*r['occupancy_amendment']['whole_device_fraction']:
                                reason='WHOLE_DEVICE_MEMORY_LIMIT'
                            if any(v>0 for u,v in used.items() if u!=dev['uuid']):reason='UNREGISTERED_GPU_ALLOCATION'
                            append(run/'resource_samples.jsonl',{'model':model,'time':time.time(),'owned':owned,'used_mib_by_uuid':used,'device_memory':devices[dev['uuid']],'task_cap_mib':task_cap,'elapsed':elapsed,'call_elapsed':call_elapsed})
                            if reason:break
                        except BaseException as exc:reason='RESOURCE_LIMIT_ENFORCEMENT_UNAVAILABLE:'+repr(exc);break
                        time.sleep(.5)
                    if reason:
                        stopped=stop_owned(process,owned);save(dest/'stop_event.json',{'reason':reason,**stopped})
                    process.wait()
                    # Also remove lingering task descendants after normal root exit.
                    alive={p:t for p,t in owned.items() if p!=process.pid and psutil.pid_exists(p)}
                    if alive:save(dest/'descendant_cleanup.json',stop_owned(process,alive))
                    save(dest/'admission.json',{'status':'STOPPED' if reason or process.returncode else 'FINISHED','reason':reason,'returncode':process.returncode})
            finally:
                if process is not None and process.poll() is None:stop_owned(process,owned)
                append(campaign,{'event':'MODEL_FINISHED','model':model,'run':str(run),'seconds':time.monotonic()-begin,'reason':reason})

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['supervise','worker']);p.add_argument('--run',type=Path,required=True);p.add_argument('--model');a=p.parse_args()
    token=os.environ.get('T1_TOKEN','')
    if a.mode=='worker':worker(a.run,a.model,token)
    else:supervise(a.run,token)
if __name__=='__main__':main()

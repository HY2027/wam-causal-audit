#!/usr/bin/env python3
"""Phase2: explicit legacy propagation capture and bounded technical acceptance.
No simulation or policy-action execution in this entry point.
"""
from wam_causal_audit.paths import resolve as _release_path
import argparse,csv,hashlib,json,time,traceback
from pathlib import Path
import numpy as np

ROOT=Path(_release_path('@DATA@/wam_factor_routing_v5'))
OUT=ROOT/'phase2_local_future_restoration_v1_20260910'
OLD=ROOT/'joint_experiment_A_strict_consumer_v1'
G1=Path(_release_path('@DATA@/wam_control_state_v3/group1_factor_phase'))
WORK=Path(__file__).resolve().parent
REQUEST=Path(_release_path('@WORKSPACE@/external_protocols/restoration_original_request.txt'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text())
def dump(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)))
def table(p,rows,fields=None):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    fields=fields or list(dict.fromkeys(k for r in rows for k in r))
    with p.open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def freeze():
    if (OUT/'phase2_freeze_manifest.json').exists():raise RuntimeError('already frozen')
    OUT.mkdir(parents=True,exist_ok=True)
    selection=load(OLD/'a0/a0_selection.json')
    phases=[json.loads(l) for l in (G1/'phase_registry.jsonl').read_text().splitlines()]
    donors=[json.loads(l) for l in (G1/'donor_bank/joint/counterfactual_qc.jsonl').read_text().splitlines()]
    states=[];ds=[]
    for s in selection['states']:
        p=next(p for p in phases if p['base_state_id']==s['base_state_id'])
        states.append(p)
        ds.extend(d for d in donors if d['task_id']==p['task_id'] and d['source_state_id']==p['source_state_id'] and d['factor']=='F1_ROBOT_RADIAL_PROGRESS' and d['phase']=='PREGRASP' and d['relation_control']=='STANDARD' and float(d['signed_dose']) in (-4,-2,-1,1,2,4))
    dump(OUT/'technical_selection.json',states);dump(OUT/'technical_donors.json',ds)
    reg=load(_release_path('@WORKSPACE@/step0_tensor_inventory_work/step0_1_v2/locus_registry_v2.json'))
    relevant=[r for r in reg['loci'] if r['locus_id'] in ['JOINT_L02','JOINT_L12']]
    dump(OUT/'registered_layer_mapping.json',relevant)
    configs={'R0':('recipient','recipient',[]),'B':('donor','recipient',[]),'D':('donor','natural_donor',list(range(30))),'Pall':('donor','P',list(range(30))),'PW':('donor','P',list(range(25,30))),'PV':('donor','P',list(range(5)))}
    table(OUT/'intervention_manifest.csv',[dict(configuration=k,current=v[0],selected_future=v[1],selected_layers=json.dumps(v[2]),unselected_future='recipient',action_steps=10,world_steps=10) for k,v in configs.items()])
    table(OUT/'consumer_event_mapping.csv',[dict(denoising_step=s,layer=l,component=c,current_group=0,future_groups='1,2',K_post_RoPE=True,V_post_projection=True,W=25<=l<=29,V=l<5,location='MoT._build_expert_attention_io:return before mixed attention') for s in range(10) for l in range(30) for c in ['k','v']])
    protocol=REQUEST.read_text()+'''\n\n# 实现冻结补充（2026-09-10，结果前）
技术样本原A0四ID原样：task0 state00/01，task1 state02/03。第一阶段未列具体ID，本次解析原A0已冻结登记，不声称第一阶段已经冻结这四个具体ID。剂量沿用A的±1、±2、±4cm；无效donor保留，不要求双符号共同有效才纳入单个有效donor。技术样本仅事后开发，不作独立验证。
P生成调用原QKVController current-only补丁（所有30层、10步、group0，K/V），在其返回之后捕获实际future；不改变其Q或后续计算。P采集与严格六格分离为不同policy calls，随机性相同。逐事件缓存完整性断言；P不等同自然donor。
正式目标仍task0/1各10新来源，稳定来源ID排序、原冻结PREGRASP规则；先完成跨全项目来源映射，无可靠映射则独立验证阻塞，绝不默认零重叠。旧静态技术样本只复用原输入，不把它们宣称为restore_v2动态验收。
读出沿用A全32个chunk位置，反归一化后xyz投影到recipient EEF→object单位向量；平移、旋转和夹爪不混单位。eB/eW/eV为径向RMS，两个主要配对差为eB-eW、eV-eW。不采用恢复率阈值或等效检验。方向比率不作主要终点，仅在原A EPS有效条件下描述。
按trajectory内有效donor等权，任务内trajectory等权，两任务等权。10000次固定任务内trajectory配对bootstrap，seed20260910，保持全部配置/dose配对；点估计与95% percentile CI，不发布显著性或p值，不作任务总体泛化推断。如后续另报告两主要检验显著性必须Holm。技术样本与新验证绝不混池。
技术通过前不允许正式前向。每worker独立一个旧起点，无重复case写入。任何确定性失败停止受影响worker并保存完整诊断；不得改容限。GPU authority RTX4090/native dtype，模型不训练。
'''
    (OUT/'phase2_protocol.md').write_text(protocol)
    (OUT/'propagation_source_audit.md').write_text('''# P的计算依赖与限定
原A legacy_infer → QKVController（每层、每步替换group0 K/V）→ 原混合注意力及后续世界/动作重算 → 下一层/步future K/V → 消费前捕获P。Q保持自然计算，current缓存来自自然donor，RGB及video/action context均recipient。
本轮在原补丁返回后采集P；旧QKVController.capture在补丁前采集，但只改变current时同事件future不变。仍选择返回后捕获以直接对应消费值。K已RoPE，V已投影，严格重放不得再次编码。
六格另行调用：current每消费事件明确为指定来源，future逐事件从recipient、自然donor或P选取。PW/PV未选层即使世界支路重新计算也不能改变实际消费的future。P与D不同不是技术失败。
这里的K/V也参与混合注意力世界更新；本试验限定为所登记节点干预产生的future之消费阻断/恢复，不称为所有current→action路径的完整中介分解。
''')
    files=[Path(__file__),WORK/'run_experiment_a.py',Path(_release_path('@WORKSPACE@/badwam_joint_causal_work/run_joint_smoke_pair.py')),REQUEST,OLD/'run_manifest.json',OLD/'consumer_registry.json',OLD/'a0/a0_selection.json',OUT/'phase2_protocol.md',OUT/'technical_selection.json',OUT/'technical_donors.json',OUT/'consumer_event_mapping.csv',OUT/'intervention_manifest.csv']
    dump(OUT/'phase2_freeze_manifest.json',{'status':'TECHNICAL_PROTOCOL_FROZEN_FORMAL_PROVENANCE_PENDING','created_unix':time.time(),'files':{str(p):sha(p) for p in files},'tasks':[0,1],'W':list(range(25,30)),'V':list(range(5)),'world_action_steps':[10,10],'new_trajectory_results_accessed':False})
    print('FROZEN',sha(OUT/'phase2_freeze_manifest.json'))

def technical(args):
    import torch
    import run_experiment_a as a
    frozen=load(OUT/'phase2_freeze_manifest.json')
    for p,h in frozen['files'].items():assert sha(p)==h,p
    state=load(OUT/'technical_selection.json')[args.worker]
    selected=[d for d in load(OUT/'technical_donors.json') if d['task_id']==state['task_id'] and d['source_state_id']==state['source_state_id']]
    out=OUT/'technical'/f'worker{args.worker}';out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():raise RuntimeError('refuse completed-worker overwrite')
    checks=[];cost=[];events=[];metrics=[];registry=[]
    def check(name,ok,error=0,case='recipient'):
        checks.append(dict(worker=args.worker,base_state_id=state['base_state_id'],case_id=case,check=name,passed=bool(ok),max_abs_error=float(error)))
        table(out/'technical_acceptance_checks.csv',checks)
        if not ok:raise RuntimeError('TECHNICAL_GATE_FAILED:'+name)
    class Propagation(a.QKVController):
        def install(self):
            super().install();patched=self.model.mot._build_expert_attention_io;self.propagated={};self.counts={'video':0,'action':0}
            def wrapped(expert,block,*aa,**kw):
                result=patched(expert,block,*aa,**kw);mod,l=self.block_map[id(block)];self.counts[mod]+=1
                if mod=='video':
                    key=(self.step,l)
                    if key in self.propagated:raise RuntimeError('duplicate P event')
                    self.propagated[key]={c:result[j].detach().clone() for c,j in [('k',1),('v',2)]}
                return result
            self.model.mot._build_expert_attention_io=wrapped
    def timed(label,fn):
        start=time.time();value=fn();cost.append(dict(worker=args.worker,stage=label,seconds=time.time()-start));table(out/'diagnostic_compute_cost.csv',cost);return value
    def hashes(cache):return {(s,l,c):a.tensor_sha256(v[c]) for (s,l),v in cache.items() for c in ['k','v']}
    try:
        torch.set_num_threads(16)
        runner=timed('load_model',lambda:a.make_capture('joint',args.gpu))
        check('RTX4090_authority','RTX 4090' in torch.cuda.get_device_name(runner.model.device))
        before=a.model_weight_hash(runner.model);dump(out/'run_metadata.json',a.run_metadata(runner,args.gpu))
        check('native_10_steps_30_layers',runner.policy.num_inference_steps==10 and len(runner.model.video_expert.blocks)==30)
        for kind in ['observation','state']:
            check('recipient_'+kind+'_file_hash',sha(state['recipient_'+kind+'_path'])==state['recipient_'+kind+'_file_sha256'])
        obs=a.load_npz(Path(state['recipient_observation_path']));rec=runner._prepared(obs,state['instruction']);seed=int(state['policy_seed'])
        base,rr=timed('native_no_hook',lambda:a.infer(runner,rec,rec,seed))
        repeat,_=timed('native_repeat',lambda:a.infer(runner,rec,rec,seed));check('native_repeat',torch.equal(base,repeat),a.max_abs(base,repeat))
        cr,rt,_=timed('recipient_capture',lambda:a.capture_source(runner,rec,seed));check('capture_read_only',torch.equal(base,cr),a.max_abs(base,cr))
        R0,_=timed('R0_same_value',lambda:a.strict_infer(runner,rec,seed,rt.cache,rt.cache));check('same_value',torch.equal(base,R0),a.max_abs(base,R0))
        axis=np.array(state['object_position_m'])-np.array(state['eef_position_m']);axis/=np.linalg.norm(axis)
        oldfile=OLD/'cases'/f"task_{state['task_id']}"/f"state_{state['source_state_id']:02d}"/'actions.npz'
        # Resolve actual per-state path from A helper, never guess a compatible action.
        oldfile=a.state_output_dir(OLD,state)/'actions.npz'
        old=np.load(oldfile,allow_pickle=False) if oldfile.exists() else None
        for d in selected:
            case=d['case_id'];dose=float(d['signed_dose']);registry.append(dict(case_id=case,signed_dose_cm=dose,valid=d.get('donor_valid'),reason=d.get('invalid_reason',''),source='frozen_A_donor'))
            table(out/'donor_registry.csv',registry)
            if not d.get('donor_valid'):continue
            dobs=a.load_npz(Path(d['donor_observation_path']));dp=runner._prepared(dobs,state['instruction'])
            da,dt,_=timed('donor_capture',lambda:a.capture_source(runner,dp,seed))
            legacy=timed('legacy_current_no_capture',lambda:a.legacy_infer(runner,rec,seed,dt.cache,'current'))
            def propagation():
                p=Propagation(runner.model);p.patch_cache=dt.cache;p.patch_modality='video';p.patch_layers=set(range(30));p.patch_temporal_groups=(0,)
                act,run=a.infer(runner,rec,rec,seed,p);return act,p
            pa,p=timed('P_capture',propagation);check('P_capture_matches_legacy',torch.equal(pa,legacy),a.max_abs(pa,legacy),case)
            pa2,p2=timed('P_repeat',propagation);check('P_repeat_action',torch.equal(pa,pa2),a.max_abs(pa,pa2),case)
            check('P_repeat_all_cache_hashes',hashes(p.propagated)==hashes(p2.propagated),case=case)
            check('P_300_video_300_action_events',p.counts=={'video':300,'action':300} and set(p.propagated)==set(rt.cache),case=case)
            del p2
            arrays={'R0':R0.numpy(),'native_recipient':base.numpy(),'native_donor':da.numpy(),'legacy_current':legacy.numpy()}
            sources={'B':rt.cache,'D':dt.cache,'Pall':p.propagated,'PW':{k:(p.propagated[k] if k[1]>=25 else rt.cache[k]) for k in rt.cache},'PV':{k:(p.propagated[k] if k[1]<5 else rt.cache[k]) for k in rt.cache}}
            for config,cache in sources.items():
                act,run=timed(config,lambda:a.strict_infer(runner,rec,seed,dt.cache,cache));arrays[config]=act.numpy()
                check(config+'_strict_consumer_all_events',all(run['hook'][k] for k in ['hook_reached_all_video_sites','hook_reached_all_action_sites','all_current_consumed_values_exact','all_future_consumed_values_exact']),case=case)
            # No assertion Pall==D or recovery effectiveness: neither is a technical gate.
            label=('p' if dose>0 else 'm')+f'{abs(dose):g}'
            for config,key in [('R0','NATURAL__0__normalized'),('B',f'STRICT__C{label}__F0__normalized'),('D',f'STRICT__C{label}__F{label}__normalized')]:
                if old is not None and key in old:check(config+'_historical_bit_exact',np.array_equal(arrays[config],old[key]),a.max_abs(arrays[config],old[key]),case)
                else:checks.append(dict(worker=args.worker,case_id=case,check=config+'_historical_asset',passed='UNAVAILABLE',max_abs_error='',base_state_id=state['base_state_id']))
            for (s,l),value in p.propagated.items():
                for c in ['k','v']:
                    tpg=value[c].shape[1]//3
                    events.append(dict(case_id=case,step=s,layer=l,component=c,P_future_hash=a.tensor_sha256(value[c][:,tpg:]),recipient_future_hash=a.tensor_sha256(rt.cache[(s,l)][c][:,tpg:]),donor_future_hash=a.tensor_sha256(dt.cache[(s,l)][c][:,tpg:]),P_recipient_same_value=torch.equal(value[c][:,tpg:],rt.cache[(s,l)][c][:,tpg:]),W=l>=25,V=l<5))
            radial={k:a.continuous_env_action(torch.from_numpy(v),runner.processor)[:,:3]@axis for k,v in arrays.items()}
            e={k:float(np.sqrt(np.mean((radial[k]-radial['Pall'])**2))) for k in ['B','PW','PV']}
            metrics.append(dict(case_id=case,task_id=state['task_id'],source_state_id=state['source_state_id'],signed_dose_cm=dose,eB=e['B'],eW=e['PW'],eV=e['PV'],delta_restore=e['B']-e['PW'],delta_location=e['PV']-e['PW'],Pall_D_radial_RMS=float(np.sqrt(np.mean((radial['Pall']-radial['D'])**2))),identity='POSTHOC_TECHNICAL_ONLY'))
            start=time.time();a.atomic_npz(out/'actions'/f'{case}.npz',**arrays,**{k+'__radial':v for k,v in radial.items()});cost.append(dict(worker=args.worker,stage='action_write',seconds=time.time()-start))
            table(out/'restoration_effects.csv',metrics);table(out/'cache_event_hashes.csv',events);table(out/'technical_acceptance_checks.csv',checks)
            dump(out/'progress.json',dict(completed_donors=len(metrics),intended_donors=len(selected),last_case=case))
            del dt,p,sources;torch.cuda.empty_cache()
        check('weights_unchanged',before==a.model_weight_hash(runner.model))
        dump(out/'complete.json',dict(status='TECHNICAL_PASS',states=1,valid_donors=len(metrics),worker=args.worker,model_weights_sha256=before,formal_authorized=False,formal_reason='independent provenance and restore_v2 registry gate still required'))
    except Exception:
        dump(out/'failure.json',dict(status='TECHNICAL_BLOCKED',traceback=traceback.format_exc(),worker=args.worker));raise

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['freeze','technical']);ap.add_argument('--worker',type=int,default=0);ap.add_argument('--gpu',type=int,default=0);args=ap.parse_args()
    freeze() if args.stage=='freeze' else technical(args)

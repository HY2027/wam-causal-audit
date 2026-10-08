"""Frozen fixed-task paired bootstrap and independent-source report."""
import json,time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from phase2_native_validation import OUT
from phase2_local_future import sha,load,dump,table

def analyze():
    f=load(OUT/'formal_freeze_manifest.json');assert f['statistics_code_sha256']==sha(__file__)
    tasks=sorted(set(r['task_id'] for r in load(OUT/'formal_sources.json')))
    for t in tasks:assert load(OUT/f'formal/task{t}/completion.json')['status']=='FORMAL_TASK_COMPLETE'
    concat=lambda filename,kind='formal':pd.concat([pd.read_csv(OUT/f'{kind}/task{t}'/filename) for t in tasks],ignore_index=True)
    effects=concat('restoration_effects.csv');donors=concat('donor_registry.csv');checks=concat('intervention_integrity_checks.csv')
    attempts=concat('native_acquisition_attempts.csv','acquisition');restore=concat('restore_v2_acceptance_checks.csv','acquisition')
    costs=pd.concat([concat('compute_cost.csv').assign(category='offline_diagnostic'),concat('compute_cost.csv','acquisition').assign(category='native_source_acquisition')],ignore_index=True)
    for name,data in [('restoration_effects.csv',effects),('donor_registry.csv',donors),('intervention_integrity_checks.csv',checks),('native_acquisition_attempts.csv',attempts),('restore_v2_acceptance_checks.csv',restore),('compute_cost.csv',costs)]:data.to_csv(OUT/name,index=False)
    assert checks.passed.all() and restore.passed.all()
    metrics=['eB','eW','eV','delta_restore','delta_location','Pall_D_RMS','uP_RMS','uW_RMS','uV_RMS'];statistics=[]
    # Replicates keep each source's doses and configurations paired; tasks fixed.
    def summarize(frame,scope,label):
        s=frame.groupby(['task_id','trajectory_id'])[metrics].mean();rng=np.random.default_rng(20260910);boots=[];means=[]
        for t in sorted(s.index.get_level_values(0).unique()):
            v=s.loc[t].to_numpy();idx=rng.integers(0,len(v),size=(10000,len(v)));boots.append(v[idx].mean(1));means.append(v.mean(0))
        b=np.mean(boots,axis=0);point=np.mean(means,axis=0)
        for k,metric in enumerate(metrics):statistics.append(dict(scope=scope,stratum=label,metric=metric,estimate=point[k],ci_low=np.quantile(b[:,k],.025),ci_high=np.quantile(b[:,k],.975),trajectory_n=len(s),task_n=len(means),bootstrap_unit='trajectory within fixed task',p_value='NOT_REPORTED',interpretation='pointwise paired interval; no equivalence claim'))
    summarize(effects,'OVERALL','ALL')
    for t in tasks:summarize(effects[effects.task_id==t],'TASK',str(t))
    for sign in [-1,1]:summarize(effects[np.sign(effects.signed_dose)==sign],'SIGN',str(sign))
    for dose in sorted(effects.signed_dose.unique()):summarize(effects[effects.signed_dose==dose],'DOSE',str(dose))
    table(OUT/'paired_statistics.csv',statistics)
    vectors=[];raw=[]
    for p in sorted((OUT/'formal').glob('task*/states/*/*_actions.npz')):
        case=p.parent.name+'__F1__'+p.name.replace('_actions.npz','')
        with np.load(p,allow_pickle=False) as z:
            uP=z['Pall__radial']-z['B__radial'];uW=z['PW__radial']-z['B__radial'];uV=z['PV__radial']-z['B__radial']
            for j in range(len(uP)):vectors.append(dict(case_id=case,chunk_index=j,uP=uP[j],uW=uW[j],uV=uV[j],remaining_W=uW[j]-uP[j],remaining_V=uV[j]-uP[j],Pall_minus_D=z['Pall__radial'][j]-z['D__radial'][j]))
            raw.append(dict(case_id=case,path=str(p),sha256=sha(p),uP_L2=np.linalg.norm(uP),uW_L2=np.linalg.norm(uW),uV_L2=np.linalg.norm(uV),uW_dot_uP=np.dot(uW,uP),uV_dot_uP=np.dot(uV,uP),normalized_projection='N/A: no new small-response threshold; raw vectors and inner products retained'))
    table(OUT/'signed_response_vectors.csv',vectors);table(OUT/'raw_action_manifest_and_secondary.csv',raw)
    (OUT/'figures').mkdir(exist_ok=True)
    state=effects.groupby(['task_id','trajectory_id'])[metrics].mean().reset_index();fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for _,r in state.iterrows():
        axes[0].plot([0,1,2],[r.eB,r.eW,r.eV],c=f'C{int(r.task_id)}',alpha=.5,marker='.')
        axes[1].scatter(r.delta_restore,r.delta_location,c=f'C{int(r.task_id)}')
    axes[0].set_xticks([0,1,2],['B','PW','PV']);axes[0].set_ylabel('radial RMS error to Pall (translation commands)');axes[1].axhline(0,c='gray');axes[1].axvline(0,c='gray');axes[1].set_xlabel('eB - eW');axes[1].set_ylabel('eV - eW');fig.suptitle('New native trajectories; predicted diagnostic actions NOT executed')
    fig.savefig(OUT/'figures/independent_restoration.png',dpi=180);plt.close(fig)
    s=pd.DataFrame(statistics);overall=s[s.scope=='OVERALL'].set_index('metric');r=overall.loc['delta_restore'];l=overall.loc['delta_location'];eb=overall.loc['eB'];ew=overall.loc['eW'];ev=overall.loc['eV'];pdiff=overall.loc['Pall_D_RMS']
    report=f'''# 第二阶段：新原生来源的独立轨迹验证

本轮属于既有发现驱动的新干预验证，不是全项目从未接触的新科学假说。旧四点技术数据、原空registry和阻塞报告完全保留且未进入本轮统计。

## 来源与恢复
实际尝试{len(attempts)}次，冻结来源{len(load(OUT/'formal_sources.json'))}条，来自固定task0/1的未使用benchmark初始状态。每任务上限20尝试、目标10合格，按来源ID排序，不按后续成功或恢复效果选择。采集终止ACQUISITION_COMPLETE不是任务成功标签。完整初始状态hash和旧来源映射见initial_state_provenance.csv、trajectory_overlap_audit.csv。
所有登记新起点的静态输入、flat状态、gripper.current_action验收通过；预先固定的每任务首个合格来源完成30步实际动作重放，连续与restore_v2的flat状态和夹爪状态逐步bit-exact。同步后输入与旧live观测的差异记录在各task sync_input_differences.csv；原始状态无时间推进，连续warmstart在forward后恢复。没有拿旧A静态门替代新恢复验证。

## 六配置和有效覆盖
共{len(donors)}个预定donor，其中{int(donors.valid.sum())}个有效，{int((~donors.valid).sum())}个几何无效未替换。全部有效案例均进入统计，未删除小效应、负方向或恢复不佳者。无有效donor的起点若存在，仍保留来源登记并作为诊断缺失；有效统计来源数{int(r.trajectory_n)}。
P来自旧current-only节点干预真实生成的future K/V，非自然donor替代。native世界/动作10/10；30层×10步严格固定两来源，PW仅25–29层使用P，PV仅0–4层；未选future每事件固定recipient。Pall独立消费复现P生成动作、R0同值和原生边界动作核验均通过；记录逐事件hash和same-value标记。未执行任何donor场景预测动作。

## 主要结果
每来源内先平均有效donor，再任务内来源等权、两个固定任务等权。10000次固定任务内trajectory配对bootstrap，保留dose/六格配对；95%区间为pointwise，不报告显著性p值，没有多重检验发现或等效声明。

| 指标 | 估计 | 95%配对区间 |
| --- | ---: | --- |
| eB | {eb.estimate:.6g} | [{eb.ci_low:.6g}, {eb.ci_high:.6g}] |
| eW | {ew.estimate:.6g} | [{ew.ci_low:.6g}, {ew.ci_high:.6g}] |
| eV | {ev.estimate:.6g} | [{ev.ci_low:.6g}, {ev.ci_high:.6g}] |
| Δrestore=eB−eW | {r.estimate:.6g} | [{r.ci_low:.6g}, {r.ci_high:.6g}] |
| Δlocation=eV−eW | {l.estimate:.6g} | [{l.ci_low:.6g}, {l.ci_high:.6g}] |
| Pall−D 径向RMS | {pdiff.estimate:.6g} | [{pdiff.ci_low:.6g}, {pdiff.ci_high:.6g}] |

单位为原A数据集反归一化平移动作指令；全32个预测chunk位置，不是执行物理时间。逐任务、正负号、每剂量结果见paired_statistics.csv；逐来源和原始有符号向量均保存。归一化方向比率未因本轮结果增设分母阈值，输出原向量、L2和点积作为次要描述。

## 解释与论文范围
Δrestore与Δlocation及剩余eW分别回答末五层恢复误差是否降低、相对指定首五层对照是否更好，以及尚未恢复的部分；不把正值直接命名为完全恢复。若其配对区间支持正方向，可增加限定句：“在已登记的两个任务的新原生轨迹上，对该current节点干预产生future变化进行严格阻断后，预定末五层恢复降低径向向量误差，且优于预定首五层对照。”若区间不支持，则应收缩为旧技术点观察尚未在本轮精度下确立，不能借此换位置。
无论结果，均不证明末五层全局最优/最小、完全充分、恢复未来动力学推理、所有current→action路径完整中介，或donor闭环任务改善。两个固定任务的不确定性不能当任务总体泛化。Pall与D分别保存，不假定同一参照。旧四点只用于既有现象的背景比较，不参与置信区间。

## 成本与停止
compute_cost.csv将原生来源采集与离线诊断分列；原生策略执行仅用于取得起点和固定技术重放，不称为闭环机制验证。真实执行历史、policy-call、快照、观测和缓存hash保留。完成后停止，未扩展模型、task、W/V、K、预算策略或闭环。
'''
    (OUT/'report_phase2_independent_validation.md').write_text(report)
    dump(OUT/'independent_validation_completion_manifest.json',dict(status='PHASE2_INDEPENDENT_VALIDATION_COMPLETE',source_n=int(r.trajectory_n),valid_donors=int(donors.valid.sum()),attempts=len(attempts),finished_unix=time.time(),analysis_code_sha256=sha(__file__),outputs={str(p):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='independent_validation_completion_manifest.json'}))

if __name__=='__main__':analyze()

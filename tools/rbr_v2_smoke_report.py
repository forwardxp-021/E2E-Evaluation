#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Package actual smoke-run evidence, independently verify train-only moments."""
import os
import sys
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import csv
import hashlib
import json
import platform
import subprocess
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/rbr_v2_smoke_v1'
DOC=ROOT/'docs/stageR/rbr_v2_smoke_v1'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def write(path,obj):Path(path).write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')


def normalizer_check(manifest):
    from tools.rbr_v2_canonical83 import ANGLES,PRESENCE
    scaler=json.loads((OUT/'normalizer.json').read_text());result={}
    for label,width,key,maskkey in [('input',83,'x','channel_valid'),('semantic',13,'semantic','semantic_valid')]:
        n=np.zeros(width,np.int64);s=np.zeros(width,np.float64);sq=np.zeros(width,np.float64)
        for row in manifest['rows']:
            if row['split']!='train':continue
            with np.load(OUT/row['path']) as data:
                a=data[key].astype(np.float64).reshape(-1,width);mask=data[maskkey].reshape(a.shape)
                n+=mask.sum(0);s+=np.where(mask,a,0).sum(0);sq+=np.where(mask,a*a,0).sum(0)
        mean=s/np.maximum(n,1);std=np.sqrt(np.maximum(sq/np.maximum(n,1)-mean*mean,0));std[std<=1e-6]=1
        if label=='input':
            mean[ANGLES]=0;std[ANGLES]=np.pi;mean[PRESENCE]=0;std[PRESENCE]=1
        state=scaler[label]
        np.testing.assert_array_equal(n,state['count'])
        np.testing.assert_allclose(mean,state['mean'],atol=1e-8,rtol=1e-7)
        np.testing.assert_allclose(std,state['std'],atol=1e-8,rtol=1e-7)
        result[label]=dict(status='PASS',valid_counts=n.tolist(),
            max_mean_difference=float(np.abs(mean-np.array(state['mean'])).max()),
            max_std_difference=float(np.abs(std-np.array(state['std'])).max()))
    result.update(fit_train_windows=2048,diagnostic_windows_fitted=0,
        independent_algorithm='double precision sum / sum-of-squares; original fitter uses streaming centered moments')
    write(OUT/'normalizer_verification.json',result)
    return result


def main():
    DOC.mkdir(parents=True,exist_ok=True)
    data=json.loads((OUT/'data_manifest.json').read_text());progress=json.loads((OUT/'run/progress.json').read_text())
    proc=json.loads((OUT/'training_process.json').read_text());epochs=json.loads((OUT/'run/epochs.json').read_text())
    steps=json.loads((OUT/'run/steps.json').read_text());reload=json.loads((OUT/'export/manifest.json').read_text())
    change=json.loads((OUT/'run/parameter_changes.json').read_text());probe=json.loads((OUT/'runtime_probe.json').read_text())
    norm=normalizer_check(data);rejected=json.loads((OUT/'materialization_rejections.json').read_text())
    stats=dict(selected_windows=len(data['rows']),considered_identity_prefix=len(data['rows'])+len(rejected),
        rejected_invalid=len(rejected),prefix_rejection_rate=len(rejected)/(len(data['rows'])+len(rejected)),
        prior_raw_quality_rejections=sum(r['stage']=='prior_complete_raw_physical_quality_check' for r in rejected),
        new_materialization_invalid=sum(r['stage']=='new_real_canonical_materialization' for r in rejected),
        valid_ego_frames=dict(min=min(r['valid_ego_frames'] for r in data['rows']),
                              max=max(r['valid_ego_frames'] for r in data['rows']),
                              mean=float(np.mean([r['valid_ego_frames'] for r in data['rows']]))),
        valid_channel_cells=sum(r['valid_channel_cells'] for r in data['rows']),
        present_neighbor_frames=sum(r['present_neighbor_frames'] for r in data['rows']),
        source_risk='RAW_SEGMENT_DISJOINTNESS_NOT_ESTABLISHED')
    write(OUT/'materialization_statistics.json',stats)
    checks=dict(real_data_counts=data['counts']=={'train':2048,'diagnostic':256},
        observed_group_disjoint=data['observed_groups']['intersection']==0,
        actual_32_optimizer_steps=len(steps)==32 and [s['step'] for s in steps]==list(range(1,33)) and progress['optimizer_steps']==32,
        complete_two_epochs=progress['complete_epochs']==2 and [e['epoch'] for e in epochs]==[1,2],
        both_shared_loss_gradients=all(min(s['repr_to_shared_grad_l2'],s['sem_to_shared_grad_l2'])>0 for s in steps),
        all_parameter_groups_updated=all(min(s['parameter_delta_l2'].values())>0 for s in steps),
        finite_training=bool(np.isfinite([[s[k] for k in ['L_repr','L_sem','total','repr_to_shared_grad_l2','sem_to_shared_grad_l2']] for s in steps]).all()),
        parameters_changed=change['total_l2']>0 and change['initial_sha256']!=change['final_sha256'],
        process_budget=proc['returncode']==0 and proc['wall_seconds']<=1800,
        fresh_process_reload=reload['status']=='PASS' and reload['new_process_pid']!=reload['training_process_pid'],
        actual_z64=reload['shape']==[256,64] and reload['finite'],
        smoke_only=not data['training_certified'] and not reload['primary_encoder_eligible'] and reload['checkpoint_label']=='SMOKE_ONLY_NOT_PRIMARY_CHECKPOINT')
    if not all(checks.values()):raise ValueError('Smoke acceptance failed: '+str(checks))
    # Verify all immutable bindings and the real bytes, not just declared JSON statuses.
    for row in data['rows']:
        if sha(OUT/row['path'])!=row['sha256']:raise ValueError('Real data artifact changed')
    for p,h in reload['source_sha256'].items():
        if sha(ROOT/p)!=h:raise ValueError('Training source changed')
    for p,k in [(OUT/'run/epoch2_smoke.pt','checkpoint_sha256'),(OUT/'normalizer.json','normalizer_sha256'),
                (OUT/'data_manifest.json','data_manifest_sha256'),(OUT/'export/diagnostic_z64.npy','z64_sha256')]:
        if sha(p)!=reload[k]:raise ValueError('Reload binding mismatch: '+str(p))
    before=json.loads((OUT/'protected_before.json').read_text());changed=[]
    for p,h in before.items():
        if p=='QUICK_REFERENCE.md':continue
        if sha(ROOT/p)!=h:changed.append(p)
    if changed:raise ValueError('Historical/user assets changed: '+str(changed))
    section=OUT/'quick_section.md'
    if section.exists():
        q=(ROOT/'QUICK_REFERENCE.md').read_bytes();suffix=section.read_bytes()
        if not q.endswith(suffix) or hashlib.sha256(q[:-len(suffix)]).hexdigest()!=before['QUICK_REFERENCE.md']:
            raise ValueError('Pre-existing user Quick content changed')
    protected=dict(checked=len(before)-1,changed=changed,
        protected_csv_sha256=sha(ROOT/'outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/behavior_events_v2/behavior_event_metrics_v2.csv'),
        previous_training_certified=json.loads((ROOT/'outputs/rbr_v2_uonly_v1/data_manifest.json').read_text())['training_certified'])
    if protected['protected_csv_sha256']!='e8deb93312e82183b6c2c0db30fd18cbf9c32d32d566038419a5be65b389d9d8':raise ValueError('Protected CSV hash')
    write(OUT/'protected_after.json',protected)
    # Compact tabular records while preserving complete JSON step details.
    with (OUT/'training_steps.csv').open('w',newline='') as fh:
        fields=['epoch','step','windows','L_repr','L_sem','total','repr_to_shared_grad_l2','sem_to_shared_grad_l2',
                'step_seconds','data_seconds','device_phase_seconds','cpu_seconds','windows_per_second']
        w=csv.DictWriter(fh,fieldnames=fields,lineterminator='\n');w.writeheader()
        for s in steps:w.writerow({k:s[k] for k in fields})
    with (OUT/'selected_samples.csv').open('w',newline='') as fh:
        fields=['split','scenario_id','actor_id','start','length','dependency_group','rank','identity_sha256',
                'valid_ego_frames','valid_channel_cells','present_neighbor_frames','raw_file','raw_record_offset',
                'raw_payload_sha256','path','sha256']
        w=csv.DictWriter(fh,fieldnames=fields,lineterminator='\n');w.writeheader()
        for r in data['rows']:w.writerow({k:r[k] for k in fields})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,4.5))
    for key,label,color in [('L_repr','Train temporal reconstruction','#0072b2'),('L_sem','Train semantic','#009e73'),('total','Train total','#d55e00')]:
        ax.plot([1,2],[e[key] for e in epochs],marker='o',label=label,color=color)
    ax.plot([1,2],[e['diagnostic']['total'] for e in epochs],'--o',color='#666666',label='Engineering diagnostic total (no selection)')
    ax.set_xticks([1,2]);ax.set_xlabel('Completed epoch');ax.set_ylabel('Masked SmoothL1 loss');ax.grid(alpha=.25)
    ax.set_title('Real Waymo RBR-V2 smoke — seed3407, last epoch only');ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(DOC/'Smoke_Loss_Curve.png',dpi=160);plt.close(fig)
    sum_step=sum(s['step_seconds'] for s in steps);mean_step=sum_step/32
    full=json.loads((ROOT/'outputs/rbr_v2_uonly_v1/quality_audit.json').read_text())['counts']['eligible_train']
    full_steps=(full+127)//128
    estimate=dict(conditional_candidate_train_windows=full,source_certified=False,steps_per_epoch=full_steps,
        measured_mean_step_seconds=mean_step,measured_windows_per_second=4096/sum_step,
        projected_train_seconds_per_epoch=full_steps*mean_step,
        projected_30_epoch_train_hours=30*full_steps*mean_step/3600,
        projected_materialization_hours=full*data['elapsed_seconds']/2304/3600,
        assumptions=['same hardware/runtime/batch128/source density and IO behavior',
                     'same per-step three-backprop gradient audit; no unmeasured speedup factor',
                     'training estimate excludes future full U_val/checkpoint overhead and early stopping',
                     'current candidate counts may change after upstream source certification',
                     'linear extrapolation is a resource estimate, NOT completed training or authorization'])
    write(OUT/'conditional_full_training_estimate.json',estimate)
    footprint=max(s['resources'].get('mps_driver_bytes',s['resources']['rss_peak_bytes']) for s in steps)
    loss_rows='\n'.join('| %d | %.6f | %.6f | %.6f | %.6f | %.2f |'%(e['epoch'],e['L_repr'],e['L_sem'],e['total'],e['diagnostic']['total'],e['train_seconds']) for e in epochs)
    report=f'''# RBR-V2-SMOKE：真实Waymo小规模训练总报告

**最终结论：`RBR_V2_SMOKE_TRAINING_PASS`。真实2048窗口、2完整epoch、32次optimizer.step、双Head共享编码器梯度、checkpoint保存、新进程重载、256×64真实表示导出均通过。**

工程状态永久为`RBR_V2_ENGINEERING_SMOKE_ONLY`；checkpoint永久标记`SMOKE_ONLY_NOT_PRIMARY_CHECKPOINT`。来源不确定性`RAW_SEGMENT_DISJOINTNESS_NOT_ESTABLISHED`保留，旧`training_certified=false`未改。本模型不能登记为Primary encoder，诊断loss不作为U_val或泛化证据。

## 实际数据和抽样

使用上一轮既有source record索引与观察依赖图，没有重新扫描51份TFRecord，也没有读取旧错误83维数组。抽样前冻结protocol、完整可复算身份排序及处理代码hash。仅取candidate train；排除任何连接到candidate test或legacy test的观察依赖组。再按组固定哈希90%/10%分为smoke train/diagnostic，在各侧按窗口身份hash排名，跳过预定物理规则无效的窗口并记录原因，不以loss或模型输出替换。

实际物化训练2048、工程诊断256，观察依赖组分别{data['observed_groups']['train']}和{data['observed_groups']['diagnostic']}，交集0。固定排序前缀共检查{stats['considered_identity_prefix']}个身份，保留{len(rejected)}个物理输入排除记录，前缀排除率{stats['prefix_rejection_rate']:.4%}。
其中先验真实原始质量预检排除{stats['prior_raw_quality_rejections']}个、新物化不合格{stats['new_materialization_invalid']}个，不把预检跳过项伪称为已重新物化。没有访问正式U_test数值样本。
选中样本全部重新读取真实Waymo原始payload并验证SHA，通过既有`from_waymo → canonical83_v1`生产接口生成80×83、channel-valid、presence、邻车身份、时间与13项目标，未改Canonical83定义。
有效ego帧数范围{stats['valid_ego_frames']['min']}～{stats['valid_ego_frames']['max']}/80，均值{stats['valid_ego_frames']['mean']:.3f}；保留{stats['present_neighbor_frames']:,}个邻车存在的slot-frame，不是仅用自车或全零邻车占位。

完整样本身份见`selected_samples.csv`，完整mask和时钟在每个NPZ中，数据manifest逐文件绑定SHA。2048训练窗口有效字段拟合normalizer；256诊断窗口从不参与拟合。另用独立sum/sum-of-squares算法逐字段复核训练统计和有效计数，输入与13目标均通过。缺失值、TTC/THW sentinel由mask排除。观察图分组互斥不能证明上游原始采集segment独立。

## 真实训练而非单元测试

直接复用原GRU128共享z64及Semantic13/Generic Temporal两个Head，没有架构搜索或新监督。seed3407、batch128、epoch2、AdamW lr0.001/weight_decay0.0001、clip1、两loss权重1:1均冻结。全部2048训练行每epoch固定seed打乱一次，16个完整batch；最后完整epoch2是唯一Smoke交付checkpoint，无best-epoch搜索、无early stopping。诊断仅做数值记录。

每个真实step分别以两个loss对共享GRU+projection计算autograd梯度，再对总loss执行backward、clip与一次optimizer.step。32步两种共享梯度均非零且有限，GRU/projection及两Head每步均有真实参数变化；初末参数hash不同，总参数变化L2为{change['total_l2']:.8f}。完整分组梯度、逐参数变化、NaN/Inf检查和每batch身份hash见`run/steps.json`及`run/parameter_changes.json`。

L_repr到共享encoder梯度L2范围：{min(s['repr_to_shared_grad_l2'] for s in steps):.8f}～{max(s['repr_to_shared_grad_l2'] for s in steps):.8f}；L_sem对应范围：{min(s['sem_to_shared_grad_l2'] for s in steps):.8f}～{max(s['sem_to_shared_grad_l2'] for s in steps):.8f}。这些是实际训练batch的梯度，不是沿用合成fixture的结果。

| Epoch | L_repr | L_sem | Train total | 工程诊断total | 训练秒数 |
|---|---:|---:|---:|---:|---:|
{loss_rows}

![真实训练loss曲线](Smoke_Loss_Curve.png)

这些loss只证明工程链路能学习并数值运行，不证明行为表示科学有效、泛化、未知行为发现或优于H30。

## 设备、资源与时间

实际训练设备：**{probe['selected_device'].upper()}**；PyTorch {probe['torch']}，NumPy {probe['numpy']}，独立Python环境。原nuPlan环境不覆盖。先进行零optimizer更新的设备兼容性检查，再冻结设备；实际run只启动一次，独立父进程实施1800秒硬墙钟上限，没有通过新建账本重获预算。

真实训练进程墙钟 **{proc['wall_seconds']:.2f}秒 / 1800秒**。32个step累计{sum_step:.2f}秒、平均{mean_step:.4f}秒/step，实测{4096/sum_step:.2f}窗口/秒；每epoch训练、诊断和每step数据准备、设备阶段、CPU秒数均保存。记录的最大设备driver/进程内存指标为{footprint:,}字节（MPS driver分配量不是GPU利用率）。CPU进程时间和MPS实际张量设备/分配内存有实测，未采样GPU百分比，不虚构利用率。

真实数据物化墙钟{data['elapsed_seconds']:.2f}秒，4个CPU worker；这是独立的数据准备阶段，不冒充训练墙钟。原始安装直连下载缓慢后改用代理，完整安装日志保留；不是第二次训练。依赖精确版本见`runtime_requirements.txt`。

## checkpoint重载与真实z64

epoch2 checkpoint：`run/epoch2_smoke.pt`，SHA256 `{reload['checkpoint_sha256']}`。

训练normalizer SHA256 `{reload['normalizer_sha256']}`。数据manifest SHA256 `{reload['data_manifest_sha256']}`。

训练进程退出后，在新进程重新加载checkpoint和normalizer，对事前固定的全部256个真实诊断窗口以eval模式、不遮挡输入导出`export/diagnostic_z64.npy`，shape=[256,64]、全部有限。新进程与训练进程内参考的最大绝对差为{reload['max_abs_reference_difference']:.9g}；新进程内重复eval最大差为{reload['max_abs_repeat_difference']:.9g}，均满足事前atol1e-6/rtol1e-5。z只来自共享编码器，不拼接语义真值或预测。前8行CSV只作表示输出示例，不挑选检测效果。

实际z64 SHA256 `{reload['z64_sha256']}`。源码和配置hash在checkpoint、execution freeze和最终Manifest间交叉绑定。

## 有条件的全量资源估算

当前未认证候选train中有{full:,}个自车有效窗口，batch128需{full_steps} step/epoch。若硬件、运行时、IO、邻车密度与本次一致，并保留本次每步三次反传的梯度审计成本，则线性外推训练约{estimate['projected_train_seconds_per_epoch']/60:.2f}分钟/epoch，30epoch约{estimate['projected_30_epoch_train_hours']:.2f}小时；不含未来完整U_val及额外checkpoint开销，也未假定early stopping或未实测加速。按本次4-worker数据物化吞吐，候选train重建约{estimate['projected_materialization_hours']:.2f}小时。

这只是基于当前小样本吞吐的条件预算，缓存、批量装配、场景复杂度、硬件热状态都可能改变结果。正式来源认证后样本数量可能变化，必须重新计预算。**没有执行全量训练，没有授予全量模型资格，也没有由此解除科学来源阻断。**

## 验证、历史保护与边界

采样/测试源排除边界测试与源码编译、临时依赖检查通过；PASS的主要依据是实际32步更新和跨进程重载，而不是单元测试。{protected['checked']:,}个既有文件逐字节保护校验无变化；旧training_certified仍为false。受保护CSV SHA256仍为`{protected['protected_csv_sha256']}`。用户原有设计文档和Quick未提交内容保留，Quick只追加本轮命令。

本轮RBR/H30比较、BDD/MMD、Primary V/C、新nuPlan仿真、全量正式训练及3408/3409均为0。B1、C2、旧P2、F2、SDC-v1结果及账本不回写。唯一实际训练是本次工程Smoke，不能当作科学效用结果。

下一阶段应基于本次实测吞吐与数据风险，决定正式U来源/划分主张、全量物化和预算；无需重做已经打通的Canonical83/GRU双Head基础链路。本轮完成后停止。
'''
    (DOC/'Final_Report_CN_v1.md').write_text(report)
    write(DOC/'Acceptance_v1.json',dict(status='RBR_V2_SMOKE_TRAINING_PASS',checks=checks,
        state='RBR_V2_ENGINEERING_SMOKE_ONLY',checkpoint_label='SMOKE_ONLY_NOT_PRIMARY_CHECKPOINT',
        training_certified=False,formal_training_runs=0,engineering_training_runs=1,optimizer_steps=32))
    sources=[*ROOT.glob('tools/rbr_v2_smoke_*.py'),ROOT/'tests/test_rbr_v2_smoke.py',ROOT/'configs/rbr_v2_smoke_v1/protocol.json']
    artifacts=[p for p in OUT.rglob('*') if p.is_file() and 'runtime' not in p.parts
               and p.name not in ['candidate_order.json','report_packaging.log']]
    write(DOC/'Manifest_v1.json',dict(status='RBR_V2_SMOKE_TRAINING_PASS',base_commit='f9b496cf629ab9cf3d938701f9bd9b9ac48e3985',
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in sources},
        artifact_sha256={str(p.relative_to(ROOT)):sha(p) for p in artifacts},
        documents_sha256={str(p.relative_to(ROOT)):sha(p) for p in DOC.rglob('*') if p.is_file() and p.name!='Manifest_v1.json'},
        local_rebuildable_order_sha256=sha(OUT/'candidate_order.json'),protected=protected,
        platform=platform.platform(),self_hash='Final Git commit/tree binds manifest; self excluded'))
    print('RBR_V2_SMOKE_TRAINING_PASS; real updates/reload verified, history preserved')


if __name__=='__main__':main()

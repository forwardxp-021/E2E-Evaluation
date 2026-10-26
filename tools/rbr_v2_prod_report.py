#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""One Chinese production report; model verification is performed by the new runtime exporter."""
import os
import sys
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import csv
import json
import math
import subprocess
from pathlib import Path
import numpy as np
from tools.rbr_v2_prod_runtime import Corpus,sha,write,ROOT,OUT,CFG
from tools.rbr_v2_canonical83 import ANGLES,PRESENCE

DOC=ROOT/'docs/stageR/rbr_v2_prod_v1'


def verify_normalizer(data):
    scaler=json.loads((OUT/'normalizer.json').read_text());proof={}
    for label,width,key,mkey in [('input',83,'x','channel_valid'),('semantic',13,'semantic','semantic_valid')]:
        n=np.zeros(width,np.int64);s=np.zeros(width);ss=np.zeros(width)
        for shard in data.maps:
            for offset in range(0,len(shard[key]),256):
                a=shard[key][offset:offset+256].astype(np.float64).reshape(-1,width)
                mask=shard[mkey][offset:offset+256].reshape(a.shape)
                n+=mask.sum(0);s+=np.where(mask,a,0).sum(0);ss+=np.where(mask,a*a,0).sum(0)
        mean=s/np.maximum(n,1);std=np.sqrt(np.maximum(ss/np.maximum(n,1)-mean*mean,0));std[std<=1e-6]=1
        if label=='input':mean[ANGLES]=0;std[ANGLES]=np.pi;mean[PRESENCE]=0;std[PRESENCE]=1
        np.testing.assert_array_equal(n,scaler[label]['count'])
        np.testing.assert_allclose(mean,scaler[label]['mean'],atol=1e-7,rtol=1e-7)
        np.testing.assert_allclose(std,scaler[label]['std'],atol=1e-7,rtol=1e-7)
        proof[label]=dict(status='PASS',max_mean_difference=float(np.abs(mean-scaler[label]['mean']).max()),
            max_std_difference=float(np.abs(std-scaler[label]['std']).max()),counts=n.tolist())
    proof.update(fit_rows=data.n,fit_role='TRAIN_ONLY',independent_validation_used=False,
        data_manifest_sha256=data.hash,normalizer_sha256=sha(OUT/'normalizer.json'))
    write(OUT/'normalizer_independent_verification.json',proof)


def main():
    DOC.mkdir(parents=True,exist_ok=True);data=Corpus(verify=True);manifest=data.manifest
    plan=json.loads((OUT/'production_plan.json').read_text());f=json.loads((OUT/'execution_freeze.json').read_text())
    history=json.loads((OUT/'run/epochs.json').read_text());progress=json.loads((OUT/'run/progress.json').read_text())
    attempts=json.loads((OUT/'run/attempts.json').read_text());identity=json.loads((OUT/'run/identity.json').read_text())
    exported=json.loads((OUT/'export/reload_verification.json').read_text())
    expected={':'.join([r['scenario_id'],r['actor_id'],r['start'],r['length']])
              for r in csv.DictReader((OUT/'planned_candidates.csv').open()) if r['canonical_eligible']=='True'}
    if len(set(data.identities))!=data.n or set(data.identities)!=expected:
        raise ValueError('Actual full corpus does not equal all preregistered eligible identities')
    if progress['state']!='TRAINING_COMPLETED_FIXED_EPOCH30' or len(history)!=30:raise ValueError('Incomplete fixed30 training')
    if [e['epoch'] for e in history]!=list(range(1,31)) or any(e['rows']!=data.n for e in history):raise ValueError('Epoch denominator/order mismatch')
    if progress['optimizer_steps']!=f['expected_committed_steps']:raise ValueError('Optimizer denominator mismatch')
    for e in history:
        grad=e['shared_gradient_audit']
        if min(grad.values())<=0 or not np.isfinite(list(grad.values())).all():raise ValueError('Shared/head gradient participation failed')
        if e['diagnostic']['used_for_selection'] or e['diagnostic']['role']!='IN_SAMPLE_NUMERICAL_ONLY':raise ValueError('Forbidden validation selection')
    if not exported.get('independent_process_repeat_pass') or not exported['finite'] or exported['shape']!=[256,64]:
        raise ValueError('Independent process z64 interface verification incomplete')
    if exported['pid']==attempts[-1]['pid'] or exported['independent_previous_pid']==attempts[-1]['pid']:
        raise ValueError('Reload must be independent of actual training process')
    total_device=sum(a['wall_seconds'] for a in attempts)
    if total_device>86400:raise ValueError('Authorized device budget exceeded')
    for path,key in [(OUT/'run/epoch30.pt','checkpoint_sha256'),(OUT/'normalizer.json','normalizer_sha256'),
        (OUT/'data_manifest.json','data_manifest_sha256'),(OUT/'export/z64.npy','z64_sha256')]:
        if sha(path)!=exported[key]:raise ValueError('Model/interface binding changed')
    for path,h in f['source_sha256'].items():
        if sha(ROOT/path)!=h:raise ValueError('Frozen scientific source changed')
    saved=OUT/'normalizer_independent_verification.json'
    verified=json.loads(saved.read_text()) if saved.exists() else {}
    if verified.get('data_manifest_sha256')!=data.hash or verified.get('normalizer_sha256')!=sha(OUT/'normalizer.json'):
        verify_normalizer(data)
    before=json.loads((OUT/'protected_before.json').read_text());changed=[]
    for p,h in before.items():
        if p=='QUICK_REFERENCE.md':continue
        if sha(ROOT/p)!=h:changed.append(p)
    if changed:raise ValueError('Historical/user asset changed: '+str(changed))
    section=OUT/'quick_section.md'
    if section.exists():
        import hashlib
        q=(ROOT/'QUICK_REFERENCE.md').read_bytes();suffix=section.read_bytes()
        if not q.endswith(suffix) or hashlib.sha256(q[:-len(suffix)]).hexdigest()!=before['QUICK_REFERENCE.md']:
            raise ValueError('User Quick prefix changed')
    protected=dict(checked=len(before)-1,changed=changed,
        protected_csv_sha256=sha(ROOT/'outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/behavior_events_v2/behavior_event_metrics_v2.csv'))
    if protected['protected_csv_sha256']!='e8deb93312e82183b6c2c0db30fd18cbf9c32d32d566038419a5be65b389d9d8':raise ValueError('Protected CSV identity')
    write(OUT/'protected_after.json',protected)
    with (OUT/'epochs.csv').open('w',newline='') as ff:
        fields=['epoch','rows','optimizer_steps','L_repr','L_sem','total','train_seconds','diagnostic_seconds']
        w=csv.DictWriter(ff,fieldnames=fields,lineterminator='\n');w.writeheader()
        for e in history:w.writerow({k:e[k] for k in fields})
    committed_checkpoints={p.name:sha(p) for p in sorted((OUT/'run').glob('epoch*.pt'))}
    if len(committed_checkpoints)!=30:raise ValueError('Every epoch checkpoint must be retained')
    write(OUT/'epoch_checkpoint_inventory.json',committed_checkpoints)
    prod_seconds=json.loads((OUT/'production_progress.json').read_text())['this_process_wall_seconds']
    training_seconds=sum(e['train_seconds'] for e in history)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,4.8));epochs=[e['epoch'] for e in history]
    for key,label in [('L_repr','Train temporal'),('L_sem','Train semantic'),('total','Train total')]:
        ax.plot(epochs,[e[key] for e in history],label=label)
    ax.plot(epochs,[e['diagnostic']['total'] for e in history],'--',label='In-sample diagnostic (no selection)',color='gray')
    ax.set_xlabel('Completed epoch');ax.set_ylabel('Masked SmoothL1 loss');ax.set_title('RBR-V2 U-only production — fixed epoch30, seed3407')
    ax.grid(alpha=.25);ax.legend();fig.tight_layout();fig.savefig(DOC/'Training_Loss_Curve.png',dpi=160);plt.close(fig)
    first,last=history[0],history[-1]
    report=f'''# RBR-V2-PROD：Waymo完整生产与首个正式RBR-64候选

**结果A：`RBR_V2_U_ONLY_TRAINED_CANDIDATE`。附加限制：`U_SOURCE_COMPLETE_INDEPENDENCE_NOT_ESTABLISHED`。**

实际完整语料、随机初始化seed3407、固定30个完整epoch、epoch30唯一交付checkpoint及独立进程z64接口验收均完成。本模型是受限科学主张下的U-only训练候选，不是独立验证成功或检测优势结果。

## 来源、角色与实际分母

本地51份`uncompressed_scenario_training_training.tfrecord-xxxxx-of-01000`属于训练分片命名与Scenario格式一致的子集，**不是全部Waymo训练集**。既有完整文件SHA与每个使用record的payload SHA重新核验。原始下载release收据未在本地找到，不能虚构具体发布版本或上游采集身份链。未发现可认证的独立官方验证来源，因此在第一个正式optimizer step之前冻结`FIXED_EPOCH_30_NO_U_VAL_CHECKPOINT_SELECTION`，不在看到loss后切换路线。

[Waymo官方Motion说明](https://waymo.com/intl/it/open/data/motion/)说明该格式来自重叠的9秒窗口，91个10Hz状态，并且坐标原点可随scene变化；Scenario互斥不能替代原始采集片段互斥。原始segment、跨Scenario对象/绝对时间映射缺失持续披露，旧BLOCKED记录、旧training_certified=false和Smoke模型工程身份均未改写。

既有候选总数{plan['denominator']:,}；candidate_train原有{plan['candidate_train_before_test_group_isolation']:,}。
按原观察依赖图隔离一切与candidate/legacy test相连的组后，共{manifest['candidates']:,}个候选；真实物理输入不合格{manifest['excluded']:,}个，保留排除原因，合格率{manifest['rows']/manifest['candidates']:.4%}。
最终正式训练分母为 **{data.n:,}个80×83窗口**，观察依赖组{manifest['observed_train_groups']:,}个。窗口数不是完全独立来源数；未按loss、检测或安全结果删样本，也未随机缩小最终训练集合。

全部合格窗口均用于训练。固定256个训练窗口仅作**IN_SAMPLE_NUMERICAL_ONLY**诊断及接口导出，包含在训练分母中；它们不是U_val、泛化测试或新的独立验证集。没有使用U_test、nuPlan B/V/C、SDC、H30或BDD结果选择模型。完整逐行来源、依赖组、raw文件/record偏移与payload SHA、缓存坐标见`actual_training_manifest.csv`。

## Canonical83生产与优化

保留原Canonical83物理公式、槽位选择、mask、80位置10Hz时钟及Semantic13定义，原Stage5/6/7历史数组未作正式输入。按Scenario解码和重采样一次，共享各车辆/时刻的车道投影；批处理只筛选投影候选，最终投影使用原标量内核，阈值边界回到原实现。相同Scenario的多个自车窗口复用这些确定性结果。存储仅作无损布局变化：邻车ID由空字符串/数字字符串映射为-1/int64，不进入网络作为数值特征。

51个源文件分别成为可恢复的认证分片，mmap数组存储x、channel_valid、ego_valid、Semantic13与有效位、slot ID及时间。每个分片独立SHA提交；未完成attempt保留，重启只核验并复用已认证分片，不重复解码全部已完成源文件。不会在每epoch重新解码TFRecord，也不合并全部大数组进内存。

开发固定真实样例与边界测试通过；**每个正式分片的固定窗口**均逐位匹配原处理器，并且重复生产逐位一致。全部{len(manifest['shards'])}个分片完成。生产墙钟约{prod_seconds:.2f}秒，实测{data.n/prod_seconds:.2f}窗口/秒；6个worker累计{manifest['worker_seconds']:.2f}秒（含各worker处理及校验），与墙钟不同。

标准化仅使用全部正式训练窗口的有效字段及目标拟合，角度/presence规则继承原合同，无效TTC/THW不参与拟合。另用独立sum/sum-of-squares重算全语料统计，计数、mean/std均通过。诊断没有额外拟合参数。normalizer SHA：`{exported['normalizer_sha256']}`。

## 事前冻结与正式训练

预训练冻结提交：`{identity['frozen_code_commit']}`。该提交包含最终实际样本Manifest、数据/依赖规则、完整模型与训练实现、train-only normalizer、配置及执行冻结；大缓存通过SHA定位而不要求全部入Git。first optimizer step前，执行器强制检查这些冻结文件已提交且源码/hash未改变。

原`GRU128 → shared z64 → Semantic13 + Generic Temporal Head`复用，Canonical83+有效位+物理时间内部167维。seed3407、batch128、AdamW lr0.001、weight_decay0.0001、双loss 1:1、clip1均固定，无scheduler、early stopping、架构或seed搜索。正式模型从随机初始化开始，没有加载Smoke参数。保留随机初始状态及hash便于核验。

遮挡生成优化严格复现原15%两帧span首次访问规则；CPU预取与按shard聚合gather保持样本顺序。真实生产batch的遮挡/标准化逐位匹配原函数；额外梯度审计与单次总loss backward的梯度等价性在更新前通过。正式训练每epoch首批检查两loss对共享encoder的实际梯度，其余正常批次只进行一次总loss backward，不改变真正优化梯度。

实际完成 **30 epoch × {data.n:,}窗口，{progress['optimizer_steps']:,}次有效optimizer.step**。每epoch均保存checkpoint、optimizer/RNG/数据顺序与资源状态；固定选择epoch30，诊断最小值从未选择模型。首epoch训练总loss={first['total']:.8f}，末epoch={last['total']:.8f}；这些是数值学习记录，不是科学检测效用结论。

![真实30epoch训练曲线](Training_Loss_Curve.png)

唯一正式训练身份，实际执行attempt={len(attempts)}。累计训练设备计费墙钟 **{total_device:.2f}秒 / 86400秒**；实际epoch训练阶段累计{training_seconds:.2f}秒，约{30*data.n/training_seconds:.2f}窗口/秒。每步/每epoch的loss、gradient状态、吞吐、CPU时间、MPS分配内存、实际checkpoint见原始日志和`epochs.csv`，不伪造GPU利用率百分比。24h硬上限与持续资源外推均受控，未因资源缩小样本或改训练标准。

恢复只允许同一身份从完整model+optimizer+RNG+当前epoch顺序/next offset/累计loss恢复；额外attempt和消耗入账，数值故障或预算停止不能换目录重新开始。监督器丢失时只允许保守计费上界恢复，不能因此获得新预算。实际恢复情况由`run/attempts.json`列明，不能把预注册恢复能力冒充额外真实复现实验。

## 模型与接口验收

唯一交付模型：`run/epoch30.pt`，SHA256 `{exported['checkpoint_sha256']}`。完整optimizer各参数step均为预期值{f['expected_committed_steps']}，GRU/projection及两个Head相对随机初态的参数均发生真实变化；30个epoch都有两loss向共享encoder的非零有限梯度证据。

独立新进程读取完整正式checkpoint、normalizer和冻结真实输入，导出`export/z64.npy`，shape=[256,64]、全部有限。与训练进程内参考的最大绝对差{exported['reference_max_abs_difference']:.9g}，满足atol1e-6/rtol1e-5；第二个独立重载进程也通过重复一致性检查。z64 SHA `{exported['z64_sha256']}`。表示来自共享z64，不拼接语义真值或Head预测。

模型侧已经具备后续公平RBR/H30比较所需的固定64维接口，但后续仍须按既有科学合同冻结H30、detector开发/标定、独立评价身份及Primary规则，并另行授权执行。本轮没有自动执行比较、BDD/MMD、Primary V/C或新仿真。

## 科学主张与保护

可以支持：在披露来源残余风险、观察依赖图控制和固定U-only训练规则下，已生产全部指定合法语料，并得到可追溯、可重载的首个正式RBR-V2训练候选。

不能支持：完整上游来源独立、独立U验证成功、未见场景泛化、未知行为发现、低阶干扰消除、优于H30或任何BDD/Primary成功。loss降低不能替代这些独立科学检验。

保护校验{protected['checked']:,}个既有文件无改动，受保护CSV保持SHA `{protected['protected_csv_sha256']}`。原B1/C2/旧P2/SDC-v1、Smoke checkpoint、旧BLOCKED记录、Primary统计合同及用户未提交内容保留。只新增本轮正式产物和Quick命令章节。完成后停止，不启动下游实验。
'''
    (DOC/'Final_Report_CN_v1.md').write_text(report)
    write(DOC/'Acceptance_v1.json',dict(status='RBR_V2_U_ONLY_TRAINED_CANDIDATE',
        additional_status='U_SOURCE_COMPLETE_INDEPENDENCE_NOT_ESTABLISHED',rows=data.n,epochs=30,
        committed_optimizer_steps=progress['optimizer_steps'],selection=f['selection'],
        training_attempts=len(attempts),device_seconds=total_device,independent_U_validation_certified=False,
        model_interface_pass=True,downstream_execution=0))
    keep=[OUT/'production_plan.json',OUT/'data_manifest.json',OUT/'execution_freeze.json',OUT/'normalizer.json',
          OUT/'actual_training_manifest.csv',OUT/'planned_candidates.csv',OUT/'gradient_equivalence.json',OUT/'epoch_checkpoint_inventory.json',
          OUT/'normalizer_independent_verification.json',OUT/'epochs.csv',OUT/'protected_before.json',OUT/'protected_after.json']
    keep += [p for p in (OUT/'export').rglob('*') if p.is_file()]
    keep += [p for p in (OUT/'run').iterdir() if p.is_file() and (p.suffix in ['.json','.jsonl','.log','.npy'] or p.name in ['epoch30.pt','random_initial_state.pt'])]
    keep += [p for p in OUT.iterdir() if p.is_file() and p.suffix in ['.log','.json','.md'] and p.name not in ['jobs.json','report_packaging.log']]
    unique={str(p.relative_to(ROOT)):p for p in keep}
    sources=list(ROOT.glob('tools/rbr_v2_prod_*.py'))+list(ROOT.glob('tests/test_rbr_v2_prod*.py'))+[CFG]
    write(DOC/'Manifest_v1.json',dict(status='RBR_V2_U_ONLY_TRAINED_CANDIDATE',source_risk='U_SOURCE_COMPLETE_INDEPENDENCE_NOT_ESTABLISHED',
        base_commit='14cd8d670afc2038d1f14afcc5d3185871fe5624',
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in sources},
        artifact_sha256={k:sha(p) for k,p in unique.items()},
        document_sha256={str(p.relative_to(ROOT)):sha(p) for p in DOC.rglob('*') if p.is_file() and p.name!='Manifest_v1.json'},
        local_large_cache='cache/<shard_id> arrays and all intermediate epoch checkpoints retained locally; hashes in data/checkpoint manifests',
        protected=protected,self_hash='Excluded; final Git commit/tree binds manifest'))
    print('RBR_V2_U_ONLY_TRAINED_CANDIDATE; disclosed source risk; no downstream execution')


if __name__=='__main__':main()

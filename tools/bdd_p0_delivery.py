#!/usr/bin/env python3
"""BDD-P0 source audit, independent reload and final report; no new scientific pipeline selection."""
import os
import sys
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse,csv,json,platform,subprocess
from pathlib import Path
from collections import Counter
import numpy as np
from tools.bdd_p0_data import ROOT,OUT,PLAN,read,write,check
from tools.rbr_v2_d1_interface import sha
DOC=ROOT/'docs/stageR/bdd_p0_v1'


def audit():
    p=check();m=read(OUT/'dataset_manifest.json');rows=[r for r in m['rows'] if r['status']=='COMPLETE'];source=read(OUT/'identity_roster.json');conflicts=[];shared_vehicles={}
    for key in ['session_id','dependency_block','source_sha256','raw_db_header_token']:
        if len({r[key] for r in rows})!=len(rows):raise ValueError('Source alias '+key)
    for i,a in enumerate(rows):
        for b in rows[i+1:]:
            if a['role']==b['role']:continue
            va=a['session_id'].split('_veh-')[-1];vb=b['session_id'].split('_veh-')[-1]
            if va==vb and max(a['raw_time_span_us'][0],b['raw_time_span_us'][0])<=min(a['raw_time_span_us'][1],b['raw_time_span_us'][1]):conflicts.append(dict(kind='SAME_VEHICLE_RAW_INTERVAL',ids=[a['session_id'],b['session_id']]))
            if a['map_name']==b['map_name'] and max(a['origin_us'],b['origin_us'])<=min(a['end_us'],b['end_us']):conflicts.append(dict(kind='SAME_MAP_SELECTED_TIME',ids=[a['session_id'],b['session_id']]))
    if conflicts:raise ValueError('Source independence requires review; do not reallocate after outcomes '+repr(conflicts))
    for role in sorted({r['role'] for r in rows}):shared_vehicles[role]=sorted({r['session_id'].split('_veh-')[-1] for r in rows if r['role']==role})
    source_rows=[dict(session_id=r['session_id'],dependency_block=r['dependency_block'],role=r['role'],source_path=r['source_path'],source_sha256=r['source_sha256'],certified_archive_receipt=r['archive_receipt'],header_token=r['raw_db_header_token'],raw_time_span_us=r['raw_time_span_us'],selected_time_us=[r['origin_us'],r['end_us']],task=r['task'],map_name=r['map_name'],input_path=r['input_path'],input_sha256=r['input_sha256'],raw_SHA_verified_during_extraction=True) for r in rows]
    for r in source_rows:
        receipt=ROOT/'outputs/rbr_v2_d2_v1/downloads'/(Path(r['source_path']).name+'.json')
        if receipt.exists():
            original=read(receipt)
            if original['sha256']!=r['source_sha256']:raise ValueError('Official raw receipt mismatch')
            r['official_source_receipt']=str(receipt.relative_to(ROOT));r['official_source_receipt_sha256']=sha(receipt)
            r['official_origin']={k:original.get(k) for k in ['archive','member','etag','official_split','crc32','license']}
        else:r['official_origin']='HISTORICAL_LOCAL_RAW_SOURCE_BOUND_BY_OLD_EXECUTION_METADATA; no new acquisition'
    maps={str(f):sha(f) for f in (ROOT.parent/'nuplan/dataset/maps').rglob('*.gpkg')};mapindex=ROOT.parent/'nuplan/dataset/maps/nuplan-maps-v1.0.json';maps[str(mapindex)]=sha(mapindex)
    write(OUT/'Source_Manifest.json',dict(source_rows=source_rows,task_definition=p['tasks'],source_counts=dict(existing_downloaded_SESSION=source['source_session_denominator'],identity_role_available=source['identity_available'],after_dependency_guard_SESSION=source['after_guard_SESSION'],fresh_dependency_blocks=source['fresh_blocks'],exposed_development_blocks=source['development_blocks']),cross_role_overlap_conflicts=conflicts,vehicles_by_role=shared_vehicles,source_independence_scope='Unique raw LOG hashes/tokens, nonoverlapping time intervals and one window per vehicle-date block. Persistent vehicle/driver/environment correlations across days are not certified absent.',map_sha256=maps,license_file='outputs/rbr_v2_d2_v1/official_LICENSE.txt',license_sha256=sha(ROOT/'outputs/rbr_v2_d2_v1/official_LICENSE.txt'),new_raw_download_bytes=0,protected_V_C_read=False))
    # Derive MC precision only; these are NOT scientific binomial confidence intervals.
    s=read(OUT/'statistics_results.json');summaries=[]
    for design,res in s['results'].items():
        threshold_design='ACTIVITY_Q' if design=='ACTIVITY_STRESS' else 'FULL_TASK_Q' if design=='FULL_TASK_STRESS' else design
        alarms={}
        for name in ['RBR64','H30']:
            a=np.load(OUT/(design+'_'+name+'_eval.npy'));cut=s['thresholds'][threshold_design][name];alarms[name]=a>cut
            summaries.append(dict(design=design,representation=name,draws=len(a),alarms=int(alarms[name].sum()),alarm_fraction=float(alarms[name].mean()),conditional_MC_SE=float(alarms[name].std(ddof=1)/np.sqrt(len(a))),frozen_threshold=cut,negative_MMD_fraction=float((a<0).mean()),source_CI='NOT_ESTABLISHED'))
        delta=alarms['RBR64'].astype(float)-alarms['H30'];res['conditional_MC_SE_delta']=float(delta.std(ddof=1)/np.sqrt(len(delta)))
    with (OUT/'Actual_Experiment_Summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summaries[0]),lineterminator='\n');w.writeheader();w.writerows(summaries)
    write(OUT/'point_precision.json',dict(results=s['results'],scientific_interval='NOT_ESTABLISHED; conditional MC precision does not count new independent sources'))
    import importlib.metadata
    write(OUT/'Runtime_And_Engineering_Record.json',dict(Python=sys.version,platform=platform.platform(),dependencies={n:importlib.metadata.version(n) for n in ['numpy','scipy','shapely','geopandas']},nuplan_devkit_HEAD=subprocess.check_output(['git','-C',str(ROOT.parent/'nuplan-devkit'),'rev-parse','HEAD']).decode().strip(),engineering_events=[dict(log='extraction_attempt01_json_int64.log',cause='First feature NPZ completed but receipt JSON could not serialize NumPy int64; metadata converted to Python int; initial NPZ bitwise parity verified, no window/source/feature change'),dict(log='diagnostics_attempt01_syntax.log',cause='Diagnostic packaging syntax fixed; no detector statistic or preprocessing changed')],new_vehicle_propagation=0,new_training_steps=0,new_download_bytes=0,Primary_V_C=0))
    print('P0 direct identity/time/source audit and MC summary complete')


def reload_verify():
    check();import torch
    from tools.rbr_v2_model import RBRV2,normalize
    r=read(OUT/'representations.json')['rows'][:16];normalizer=read(ROOT/'outputs/rbr_v2_prod_v1/normalizer.json')['input'];a=[np.load(ROOT/v['input_path']) for v in r]
    x=np.stack([normalize(v['x'],v['channel_valid'],normalizer) for v in a]);mask=np.stack([v['channel_valid'] for v in a]);ev=np.stack([v['ego_valid'] for v in a])
    model=RBRV2();state=torch.load(ROOT/'outputs/rbr_v2_prod_v1/run/epoch30.pt',map_location='cpu',weights_only=False);model.load_state_dict(state['model']);device='mps' if torch.backends.mps.is_available() else 'cpu';model.to(device);model.eval()
    with torch.no_grad():z=model(torch.as_tensor(x,device=device),torch.as_tensor(mask,device=device),torch.as_tensor(ev,device=device))[0].cpu().numpy()
    expected=np.load(OUT/'RBR64.npy')[:16];error=float(np.max(np.abs(z-expected)))
    if error>1e-5:raise ValueError('Independent reload outside fixed1e-5 numerical tolerance')
    write(OUT/'encoder_reload_verification.json',dict(new_process=True,rows=16,shape=list(z.shape),max_abs_error=error,bitwise_equal=bool(np.array_equal(z,expected)),tolerance=1e-5,device=device,torch=torch.__version__,checkpoint_sha256=sha(ROOT/'outputs/rbr_v2_prod_v1/run/epoch30.pt'),new_optimizer_steps=0))
    print('Independent real-window frozen encoder reload PASS',error)


def figures():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    s=read(OUT/'statistics_results.json');fig,ax=plt.subplots(figsize=(9,4.5));designs=['RAW_IID','ACTIVITY_Q','ACTIVITY_STRESS'];x=np.arange(3)
    for name,off in [('RBR64',-.18),('H30',.18)]:ax.bar(x+off,[s['results'][d]['alarm_fraction'][name] for d in designs],.36,label=name)
    ax.axhline(.05,color='red',ls='--',label='Nominal alpha0.05');ax.set_xticks(x);ax.set_xticklabels(['Raw real-log A/A','Activity-matched A/A','Artificial activity mix shift']);ax.set(ylabel='Finite-pool alarm fraction',ylim=(0,.4),title='Real human logs: external calibration and held-out diagnostics')
    ax.legend(fontsize=8);ax.grid(alpha=.2);fig.text(.5,.01,'10,000 dependent release draws per point; source-domain FPR/CI not certified.',ha='center',fontsize=8);fig.tight_layout(rect=(0,.04,1,1));fig.savefig(DOC/'P0_Actual_AA_And_Composition.png',dpi=170);plt.close(fig)
    from tools.bdd_p0_data import digest
    rows=read(OUT/'dataset_manifest.json')['rows'];fig,ax=plt.subplots(figsize=(9,4));chosen=[]
    for task in ['LOW','LANE','JUNCTION']:
        r=min([v for v in rows if v['status']=='COMPLETE' and v['role']=='D_P0_FIT_EXPOSED' and v['task']==task],key=lambda v:digest('P0_FIXED_DEVELOPMENT_CURVES',v['session_id']));a=np.load(ROOT/r['input_path']);ax.plot(a['time_us']/1e6,a['x'][:,5],label=task+' '+r['session_id']);chosen.append(dict(task=task,session_id=r['session_id'],rule='Lowest identity hash within development task; no detector outcome selection'))
    ax.set(xlabel='Physical time from raw window start (s)',ylabel='Actual logged center speed (m/s)',title='Actual human-log windows (no simulated vehicle propagation)');ax.legend(fontsize=7);ax.grid(alpha=.2);fig.tight_layout();fig.savefig(DOC/'P0_Real_Human_Window_Examples.png',dpi=170);plt.close(fig);write(OUT/'curve_choices.json',chosen)


def report():
    check();figures();ds=read(OUT/'dataset_manifest.json');source=read(OUT/'Source_Manifest.json');stat=read(OUT/'statistics_results.json');num=read(OUT/'numerical_diagnostics.json');rep=read(OUT/'representations.json');fit=read(OUT/'detector_fit.json');precision=read(OUT/'point_precision.json');reload=read(OUT/'encoder_reload_verification.json')
    rows=[r for r in ds['rows'] if r['status']=='COMPLETE'];summary='\n'.join(f"|{d}|{v['alarm_fraction']['RBR64']:.4%}|{v['alarm_fraction']['H30']:.4%}|{v['delta_raw_alarm']:.4f}|" for d,v in stat['results'].items())
    contexts='\n'.join(f"|{role}|{v['SESSION']}|{v['tasks'].get('LOW',0)}|{v['tasks'].get('LANE',0)}|{v['tasks'].get('JUNCTION',0)}|{v['tasks'].get('OTHER',0)}|" for role,v in num['role_contexts'].items())
    text=f'''# BDD-P0：正常驾驶日志的多场景BDD实际研究入口

**BDD_P0_REAL_DATA_RESEARCH_ENTRY_ESTABLISHED_WITH_LIMITATIONS**

本轮实际生成190个真实nuPlan窗口、190×64冻结RBR表示和190×30完整H30，完成外部校准、留出随机分割A/A和人为活动构成压力测试。**数据足以继续开发任务条件化BDD，但不足以认证来源总体FPR/区间覆盖、完整拓扑条件化或真实E2E策略版本比较。** 不是RBR获胜，也不是D6恢复PASS。SDC保持正控制，旧q*及所有历史SAP/结果不变。

## 1. 真实数据与科学角色

已有D2第一来源1253个SESSION未采用SDC适格结果重新筛选。按旧D、未来cal/V/C及保护身份排除后可用1091；再排除这些身份同vehicle日期±1日的依赖保护范围，剩398个SESSION、133个vehicle-date采集块。本轮每块固定身份hash选1个SESSION，按地图内hash顺序交替分配68 CAL与65 HELDOUT，**在表示/报警计算前冻结**。已有E3仅用于正常原日志预处理拟合，63个来源按同依赖单位选出57块，绝不称新确认样本。

需要披露的选择偏差：保留资源中含很多原SDC静态适格源，因此“尚未被保留的来源框”会与SDC适用性相关；本轮没有按其PASS/FAIL选窗口，却也不能宣称当前语料代表全部nuPlan。旧253源被保守屏蔽，不声称已穷尽本机全部正常驾驶容量。实际认证的窗口容量是190，不是1091个全量可比窗口。

|角色|SESSION/采集块|t0 LOW|t0 LANE|t0 JUNCTION|OTHER|
|---|---:|---:|---:|---:|---:|
{contexts}

新校准/留出共133块覆盖新加坡、波士顿、拉斯维加斯；开发拟合还包含匹兹堡。每源只取1个80帧、10Hz、7.9s窗口；时钟候选按身份hash选择，允许仅因预声明物理输入无效继续下一候选，不读scenario_tag或检测效果。最终190/190源均物化，未删除不利样本。

源身份审查：SESSION、vehicle-date、raw SHA和raw header token均不重复；不同角色同vehicle的整个原始LOG时间区间无重叠，同地图所选时间区间也无重叠。不同日期仍可能共享vehicle/driver风格、环境和采集协议，**不宣称已经证明完全车队IID**。留出是P0事前未参与拟合的开发留出，不是原Primary V/C；本轮用于方法研究后全部登记E3，不能以后洗成同方法独立确认。

## 2. 新正常日志处理入口

SQL仅读取已有数据库中的ego_pose、lidar_pc、lidar_box、track/category，地图来自现有官方GPKG。对压缩本地缓存，在本轮私有scratch解压并核对原SHA，处理后只移除私有scratch；原数据库/压缩归档不改、不重新下载。保留原许可文件及来源索引hash。

原DB自车速度是body约定，车辆检测速度为world；自车中心位置/速度按Pacifica杠杆臂和记录angular-rate转换，随后复用冻结Canonical83的物理重采样、邻车槽位和mask。没有推导新“理想轨迹”，没有车辆状态传播。每个窗口都用官方EgoState中心/速度属性作独立对照；位置有限差分的声明坐标速度RMSE中位约{num['source_velocity_oracle']['body_rotated_world_RMSE_median']:.3f}m/s，错误不旋转约{num['source_velocity_oracle']['wrong_unrotated_RMSE_median']:.3f}m/s，仅作物理诊断而不据此选好窗口。

真实20Hz左右的不规则时钟重采为10Hz，跨越>150ms缺口不插值；ego80帧须有效，邻车缺失按mask保留，不几何填邻车。H30从同一车辆、同一80时刻的rear-axle记录按冻结函数计算，RBR使用约定center输入；两者信息集不同（RBR含邻车），不能把所有变化直接归于纯自车时序。

唯一epoch30 SHA=`5b0e5738c0b67d32865545764bc4bcd12152a507f62bcb0f323233a3bb8deed3`；Waymo normalizer不变。实际MPS导出190×64，有限、中心化秩{rep['RBR_centered_rank']}、熵有效秩{rep['RBR_entropy_rank']:.2f}；重复eval一致，独立进程重载16个实际窗口最大差{reload['max_abs_error']:.3g}。未训练/选checkpoint，未用Semantic Head替代z64。`U_SOURCE_COMPLETE_INDEPENDENCE_NOT_ESTABLISHED`保留。

## 3. 任务、协议与可评估区域

任务事前固定：LOW=t0速度<2m/s；运动中若初始rear点在连接段则JUNCTION，否则在车道则LANE，其余OTHER。它是窗口起点标签，**LOW不代表整个窗口静止、JUNCTION也不代表完整转弯事件**。初速是人类行为的一部分，此处仅作描述性条件，不能无条件视为真实E2E版本对比中的外生变量。

预注册并全部保留三个层次：RAW_IID不调任务混合；ACTIVITY_Q每侧10 LOW＋10 MOVING；FULL_TASK_Q每侧5 LOW＋5 JUNCTION＋10 LANE。全设计都用40个不同SESSION/采集块，外部CAL与HELDOUT互斥，两表示使用同一个身份draw及预算。

活动条件有共同支持：CAL 28 LOW/40 MOVING，HELDOUT 23/42。完整三任务设计需每池至少10 JUNCTION，但CAL只有9，故**FULL_TASK_Q及其压力实验NOT_ASSESSABLE**。不从留出挪1个、不换窗口、不缩m、不修改配额。OTHER按原规则会在粗活动/RAW分析保留，但不属于完整三任务条件域；本次实际没有OTHER。

这里没有救活D6的四单元总体：任务定义、来源框及数据形态均不同，未要求SDC曲率、安全或双制动形态。

## 4. 已实际完成的实验

检测预处理为**新开发候选**：仅57个已暴露正常原日志窗口拟合两表示各自population mean/SD，原SD≤1e−12常量规则及median-positive-distance带宽数学规则不变。一次拟合、零核/feature/seed搜索；独立CAL产生1999个AA分数、取1900阶，严格大于才报警；HELDOUT每设计10000次释放。源码、角色、fit与数组SHA已在任何P0检测结果之前冻结。固定任务配额下沿用原无偏两样本形式，不宣称它仍是任意边缘总体MMD²的无偏估计；校准必须复制相同抽样设计。

|设计|RBR64报警比例|H30报警比例|原始报警差|
|---|---:|---:|---:|
{summary}

RAW_IID是真实日志同一来源框的随机分割负控制，reference与target使用不同采集块；它不是同一轨迹复制，不是D4配对方向交换。条件分布总体以共同日志框为假设，有限CAL/HELDOUT仍存在任务/来源比例波动。

ACTIVITY_STRESS在同一HELDOUT池中，reference 10 LOW/10 MOVING，target 5 LOW/15 MOVING，任务内仍随机取原日志；**只人为改变集合构成，没有编辑轨迹、没有指定任务内行为变化**。使用事前匹配reference活动配额的AA阈值。它与ACTIVITY_Q的3.94%/5.60%对照，展示在这个有限池中恢复共同活动构成可降低本次构成变化造成的报警，但这是重采样设计结果，不能宣称已实现一般场景纠偏。原始边缘分布确实随任务混合改变，压力报警不自动是边缘零假设下的数学误报。

RBR较高的RAW A/A报警或压力报警不代表更好；前者可能是校准误差，后者不是目标策略行为效用。压力中的+.0723不是Primary ΔBDD，不作RBR优势解释。所有负MMD²保留，逐draw原统计量与固定阈值均交付；`Actual_Experiment_Summary.csv`给出计数和条件MC SE。

## 5. 数值、残余混杂与统计限制

正常原日志H30非恒定最小scale约{min(v for i,v in enumerate(fit['pipelines']['H30']['scale']) if i not in fit['pipelines']['H30']['constant_columns']):.5f}，不再由SDC确定性制动尾巴的10⁻¹²级尺度主导。H30常量列0-based27/29保留，scale1，不删列。RBR/H30在CAL与HELDOUT的非对角kernel下溢比例均0；H30的SI/g/kmh/s物理单位对照核差最大约1.67e−16。20个事前规定的开发来源bootstrap拟合全部报告、不替换原fit；它们仍显示有限样本拟合波动，不能当预处理已独立认证。

原冻结Waymo normalizer下有效输入绝对值99分位约{num['normalized_valid_absolute_quantiles'][2]:.2f}、最大约{num['normalized_valid_absolute_quantiles'][3]:.2f}，未因域偏移删除异常窗口或clip参数。map、任务、初速、mask与邻车presence分布均报告；活动控制不能消除运动组中的道路拓扑差异：CAL MOVING内JUNCTION9/40，HELDOUT19/42。不同车、驾驶员、时段与传感可见性仍可能影响表示。因此**有限池点值接近5%不等于已获得来源总体FPR控制**。

200×500来源级重采样诊断：RAW有66/200次唯一来源支持不足，ACTIVITY_Q有200/200次不足；所有失败保留，不用成功子集构造区间。source-domain CI、95%覆盖、正式功效均NOT_ESTABLISHED。10000个重叠释放只是条件MC精度，不能当10000个独立道路发布试验；P0也没有满足原SAP的正式N与推断门。

本次CAL与HELDOUT既共享来源框又在实样本任务构成不同，条件化仅在已有支持域操作。已有数据提供方法开发和数值比较入口，尚不足以独立证明消除混杂或泛化到新城市。后续任何算法依据本轮结果改变后，都需要新独立开发/验证来源，不得重新命名这65个留出块。

## 6. 五类证据与真正策略版本缺口

|证据|本轮状态|可支持的结论|
|---|---|---|
|真实人类驾驶日志|190个实际窗口|输入/表示/检测工程链路和来源约束内开发诊断|
|人为场景构成变化|活动比例压力测试实际运行|检测器对样本构成的响应；不是策略发布|
|合成行为干预|0|本轮没有此类行为证据|
|SDC闭环受控实验|本轮0；历史正控制保留|不作为P0人类轨迹，也不扩展旧SDC主张|
|真实E2E策略版本比较|**NOT_ESTABLISHED**|现有日志没有两套冻结策略版本身份及其输出|

不可替代的缺口是**真实策略版本数据**：明确策略/软件hash、采集与测量协议、任务来源/分配、各版本的实际行为输出，以及可辩护的同条件A/A与版本A/B。可来自合法部署随机/分期设计或前瞻性两策略受控运行；只有人类轨迹与人为混合，无法推断另一策略的反事实，更不能借当前RBR（行为表示模型）冒充自动驾驶策略。

## 7. 最终判断、成果及边界

**可以继续开发任务条件化BDD：真实数据、正常驾驶处理器、冻结表示、公平统计接口、负控制和构成压力已实际运行。** 完整拓扑控制和源域推断仍被容量阻断，当前不具备Primary/正式V/C条件。任务分层＋MMD本身是既有技术，数据入口与问题诊断有研究价值，但不自动构成已认证的新方法或博士创新。

SDC继续作正控制；当前真实日志的粗条件实验提供了与其独立的数据形态。后续研究应检验残余场景混杂、采样/标定可迁移性和有效不确定性，允许RBR/H30同样敏感或H30更好。不得用本轮报警差挑选有利任务、换encoder或削弱H30。

## 8. 执行与复现

原始数据物化最终完整过程墙钟253.14秒，冻结模型导出约{rep['wall_seconds']:.2f}秒，统计及来源支持诊断约{stat['wall_seconds']:.2f}秒；不含写代码/审计/打包时间。压缩输入缓存、完整来源Manifest、所有分数数组、fit和角色曝光均保留。元数据JSON类型修复后首个NPZ逐字节一致；一次辅助诊断语法错误日志保留，未改变统计规则或重新挑源。

运行/验证入口见QUICK_REFERENCE。新窗口全部来自已有原数据库，无网络下载；只使用私有解压scratch。历史受保护CSV、模型、H30定义、D3/D4、D5/D6 SAP与停止账本不改。P0进行了冻结encoder推理及开发性MMD，**车辆传播、optimizer更新、Primary/V/C均0**。完成提交推送后停止，不自动新增数据或科学实验。
'''
    (DOC/'Final_Report_CN_v1.md').write_text(text)
    write(DOC/'Status_v1.json',dict(status='BDD_P0_REAL_DATA_RESEARCH_ENTRY_ESTABLISHED_WITH_LIMITATIONS',actual_real_windows=len(rows),fresh_calibration_blocks=68,fresh_heldout_blocks=65,exposed_fit_blocks=57,real_raw_AA_executed=True,activity_control_and_composition_stress_executed=True,full_task_control='NOT_ASSESSABLE_CAL_JUNCTION9_LT10',source_FPR_CI_power='NOT_ESTABLISHED',real_E2E_policy_version='NOT_ESTABLISHED',new_vehicle_propagation=0,new_optimizer_steps=0,new_download_bytes=0,Primary_V_C=0))
    print('BDD-P0 Chinese report and real evidence figures complete')


def manifest():
    check();target=DOC/'Manifest_v1.json';paths=list(OUT.rglob('*'))+list(DOC.rglob('*'))+list((ROOT/'configs/bdd_p0_v1').rglob('*'))+list((ROOT/'tools').glob('bdd_p0_*.py'))+list((ROOT/'tests').glob('test_bdd_p0_*.py'));files={}
    for p in paths:
        if not p.is_file() or p==target or '__pycache__' in p.parts or 'work_cache' in p.parts or p.name=='user_Quick_before.txt' or p.suffix=='.writing' or p.name.startswith('manifest_packaging'):continue
        files[str(p.relative_to(ROOT))]=dict(sha256=sha(p),bytes=p.stat().st_size)
    write(target,dict(status='BDD_P0_REAL_DATA_RESEARCH_ENTRY_ESTABLISHED_WITH_LIMITATIONS',files=files,protected=read(PLAN)['protected'],source_commit=read(PLAN)['start_commit'],raw_sources='outputs/bdd_p0_v1/Source_Manifest.json',new_vehicle_propagation=0,new_optimizer_steps=0,new_download_bytes=0,Primary_V_C=0))
    print('P0 provenance bound',len(files),'owned files')


def verify():
    check();m=read(DOC/'Manifest_v1.json')
    for p,h in m['files'].items():
        if sha(ROOT/p)!=h['sha256']:raise ValueError('P0 artifact changed '+p)
    for f,h in read(OUT/'pre_statistics_freeze.json')['source_hashes'].items():
        if sha(ROOT/f)!=h:raise ValueError('Frozen statistical pipeline changed '+f)
    print('BDD-P0 verification PASS:',len(m['files']),'own hashes;190 real inputs; frozen model/history preserved; no Primary/V/C')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['audit','reload','report','manifest','verify']);a=p.parse_args();{'audit':audit,'reload':reload_verify,'report':report,'manifest':manifest,'verify':verify}[a.mode]()

#!/usr/bin/env python3
"""Package actual U-only engineering evidence; no training or downstream evaluation."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import csv
import hashlib
import json
import platform
import subprocess
from pathlib import Path
import numpy as np
import torch
import google.protobuf
from tools.rbr_v2_canonical83 import NAMES, from_waymo, from_nuplan
from tools.rbr_v2_u_data import sha, write

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/rbr_v2_uonly_v1'
DOC=ROOT/'docs/stageR/rbr_v2_uonly_v1'


def fixture_oracle():
    """Closed-form polynomials independent of production feature/derivative helpers."""
    t=np.arange(80)*.1; a=np.zeros((80,83)); mask=np.ones((80,83),bool)
    a[:,:8]=np.stack([8*t+.25*t*t,np.zeros(80),8+.5*t,np.zeros(80),np.zeros(80),
                      8+.5*t,np.r_[0,np.full(79,.5)],np.zeros(80)],1)
    mask[0,6:8]=False
    for j in range(5):
        k=8+15*j
        if j==0:
            gap=20-2*t-.05*t*t; lat=np.full(80,-.5); rv=-2-.1*t
            speed=6+.4*t; accel=np.r_[0,np.full(79,.4)]
            heading=.1+.02*t; yaw=np.r_[0,np.full(79,.02)]
        else:
            y,v,x0=[(10,7,-4),(-10,9,-4),(15,7.5,4),(-15,8.5,4)][j-1]
            gap=y+(v-8)*t; lat=np.full(80,-x0); rv=np.full(80,v-8)
            speed=v+.5*t; accel=np.r_[0,np.full(79,.5)]; heading=np.zeros(80); yaw=np.zeros(80)
        dist=np.sqrt(gap*gap+lat*lat); closing=-gap*rv/dist
        ttc=np.full(80,999.); tc=closing>1e-3; ttc[tc]=np.minimum(dist[tc]/closing[tc],999)
        thw=np.full(80,999.); hw=gap>0; thw[hw]=np.minimum(gap[hw]/(8+.5*t[hw]),999)
        a[:,k:k+15]=np.stack([np.ones(80),gap,lat,rv,np.zeros(80),dist,gap,lat,closing,
                              ttc,thw,speed,accel,heading,yaw],1)
        mask[:,k+9]=tc; mask[:,k+10]=hw; mask[0,[k+12,k+14]]=False
    return a,mask


def evidence():
    from tests.test_rbr_v2_uonly import five_slot_fixture,nuplan_fixture,nuplan_map_fixture
    s=five_slot_fixture(); w=from_waymo(s,'1'); es,obs=nuplan_fixture(s)
    n=from_nuplan(es,obs,nuplan_map_fixture(),'body'); truth,mask=fixture_oracle()
    np.testing.assert_array_equal(w['channel_valid'],mask)
    np.testing.assert_array_equal(n['channel_valid'],mask)
    np.testing.assert_allclose(w['x'],truth,atol=4e-5,rtol=2e-6)
    np.testing.assert_allclose(n['x'],truth,atol=4e-5,rtol=2e-6)
    rows=[]
    ego_units=['m','m','m/s','m/s','rad','m/s','m/s^2','rad/s']
    nb_units=['bool','m','m','m/s','m/s','m','m','m','m/s','s','s','m/s','m/s^2','rad','rad/s']
    formulas=['R(h0)(p-p0).x','R(h0)(p-p0).y','R(h0)v_world.x','R(h0)v_world.y',
              'wrap(h_world-h0)','norm(v_world)','backward_diff(speed)/0.1','wrap(backward_diff(h_world))/0.1']
    nb=['known lane slot occupied','R(h_ego)(p_n-p_e).x','R(h_ego)(p_n-p_e).y',
        'R(h_ego)(v_n-v_e).x','R(h_ego)(v_n-v_e).y','norm(p_n-p_e)',
        'duplicate rel_x','duplicate rel_y','-dot(delta_p,delta_v)/max(distance,1e-6)',
        'min(distance/closing,999); valid closing>1e-3 & distance>1e-6',
        'min(rel_x/ego_forward_speed,999); valid rel_x>0 & ego_forward_speed>1e-3',
        'norm(v_n_world)','backward_diff(speed_n)/0.1 same identity only',
        'wrap(h_n_world-h_e_world)','wrap(backward_diff(h_n_world))/0.1 same identity only']
    with (DOC/'Canonical83_Fields_And_Parity.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['index','field','unit','definition','valid_frames','waymo_oracle_max_abs',
            'nuplan_oracle_max_abs','waymo_nuplan_max_abs','status'])
        writer.writeheader()
        for i,name in enumerate(NAMES):
            j=(i-8)%15
            row=dict(index=i,field=name,unit=ego_units[i] if i<8 else nb_units[j],
                definition=formulas[i] if i<8 else nb[j],valid_frames=int(mask[:,i].sum()),
                waymo_oracle_max_abs=float(np.abs(w['x'][:,i]-truth[:,i]).max()),
                nuplan_oracle_max_abs=float(np.abs(n['x'][:,i]-truth[:,i]).max()),
                waymo_nuplan_max_abs=float(np.abs(w['x'][:,i]-n['x'][:,i]).max()),status='PASS_SYNTHETIC_PHYSICAL_ORACLE')
            rows.append(row); writer.writerow(row)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,1,figsize=(9,8),sharex=True)
    for ax,col,label in zip(axes,[5,6,16],['Ego speed (m/s)','Ego acceleration (m/s²)','Front radial closing (m/s)']):
        tt=np.arange(80)*.1; good=mask[:,col]
        ax.plot(tt[good],truth[good,col],color='black',lw=2,label='Independent physics oracle')
        ax.plot(tt[good],w['x'][good,col],'--',color='#0072b2',label='Waymo interface')
        ax.plot(tt[good],n['x'][good,col],':',color='#d55e00',label='nuPlan official objects')
        ax.set_ylabel(label); ax.grid(alpha=.25)
        if col==6:
            ax.set_ylim(.45,.55); ax.set_yticks([.45,.5,.55]); ax.ticklabel_format(axis='y',useOffset=False)
    axes[0].legend(fontsize=8); axes[-1].set_xlabel('Physical time (s)')
    fig.suptitle('Canonical83 physical validation — synthetic fixture, zero training')
    fig.tight_layout(); fig.savefig(DOC/'Canonical83_Physical_Validation.png',dpi=160); plt.close(fig)
    return rows


def main():
    if (OUT/'seed3407').exists():
        raise ValueError('Unexpected training namespace; this blocked package must have zero training')
    parity=evidence(); data=json.loads((OUT/'data_manifest.json').read_text())
    quality=json.loads((OUT/'quality_audit.json').read_text()); split=json.loads((OUT/'split_audit.json').read_text())
    smoke=json.loads((OUT/'canonical_smoke_v1/manifest.json').read_text())
    before=json.loads((OUT/'protected_before.json').read_text()); changed=[]
    quick=(ROOT/'QUICK_REFERENCE.md').read_bytes(); section=(OUT/'quick_section.md').read_bytes()
    if not quick.endswith(section) or hashlib.sha256(quick[:-len(section)]).hexdigest()!=before['QUICK_REFERENCE.md']:
        raise ValueError('Pre-existing user Quick content changed')
    # QUICK_REFERENCE is the only authorized append to a pre-existing user-edited file.
    for p,h in before.items():
        if p=='QUICK_REFERENCE.md': continue
        if sha(ROOT/p)!=h: changed.append(p)
    if changed: raise ValueError('Protected content changed: '+str(changed))
    pcsv='outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/behavior_events_v2/behavior_event_metrics_v2.csv'
    protected=dict(checked=len(before)-1,changed=changed,quick_user_prefix_unchanged=True,protected_csv_sha256=sha(ROOT/pcsv))
    if protected['protected_csv_sha256']!='e8deb93312e82183b6c2c0db30fd18cbf9c32d32d566038419a5be65b389d9d8':
        raise ValueError('Protected CSV changed')
    write(OUT/'protected_after.json',protected)
    runtime=dict(python=sys.version,platform=platform.platform(),numpy=np.__version__,torch=torch.__version__,
        protobuf=google.protobuf.__version__,cuda=torch.cuda.is_available(),nuplan_commit=subprocess.check_output(
            ['git','-C',str(ROOT.parent/'nuplan-devkit'),'rev-parse','HEAD'],text=True).strip(),
        protobuf_install='Isolated --target outputs/rbr_v2_uonly_v1/runtime_packages; frozen nuplan environment unmodified')
    from google.protobuf.internal import api_implementation
    from importlib.metadata import version
    runtime['dependency_versions']={p:version(p) for p in ['numpy','torch','protobuf','pandas',
        'scipy','matplotlib','shapely','geopandas','pyquaternion']}
    import inspect
    from tools.rbr_v2_u_data import census
    runtime['protobuf_implementation']=api_implementation.Type()
    runtime['census_function_source_sha256']=hashlib.sha256(inspect.getsource(census).encode()).hexdigest()
    runtime['census_provenance']='Census function unchanged since execution; later additions quality/verify are separately included in final source hash'
    runtime['official_nuplan_sources_sha256']={p:sha(ROOT.parent/'nuplan-devkit'/p) for p in [
        'nuplan/common/actor_state/ego_state.py','nuplan/common/actor_state/dynamic_car_state.py',
        'nuplan/common/actor_state/agent.py','nuplan/common/maps/nuplan_map/lane.py',
        'nuplan/common/maps/abstract_map_objects.py','nuplan/common/maps/nuplan_map/polyline_map_object.py',
        'nuplan/planning/simulation/controller/motion_model/kinematic_bicycle.py','requirements_torch.txt']}
    write(OUT/'runtime.json',runtime)
    gates=[dict(gate='Canonical83 physical oracle and all-slot layout',status='PASS',evidence='Canonical83_Fields_And_Parity.csv; tests_final_v3.log'),
           dict(gate='Real raw Waymo processor smoke',status='PASS',evidence='16 frozen identities, 1 explicitly input-invalid, 0 replacements'),
           dict(gate='Observed dependency grouping before split',status='PASS_LIMITED',evidence='candidate_group_cross_split=0; NOT a complete source certificate'),
           dict(gate='Raw source independence',status='BLOCKED',evidence='RAW_SEGMENT_DISJOINTNESS_NOT_ESTABLISHED'),
           dict(gate='Full certified canonical U corpus and train-only scaler',status='BLOCKED',evidence='Source prerequisite fails; 16 examples are engineering artifacts only'),
           dict(gate='Shared z64 and two-head gradients',status='PASS_ZERO_STEP',evidence='tests_final_v3.log; no optimizer steps'),
           dict(gate='Sole seed3407 executable protocol and rejection firewall',status='PASS_ZERO_STEP_ONLY',evidence='seed3407.json; entry_gate.log; training not production-certified'),
           dict(gate='First formal U-only training',status='NOT_STARTED_BLOCKED',evidence='No run namespace/checkpoint/optimizer step')]
    write(DOC/'Training_Entry_Status.json',dict(overall='RESULT_B_BLOCKED_U_SOURCE_INDEPENDENCE',gates=gates,
        training_runs=0,optimizer_steps=0,device_training_seconds=0,checkpoint=None,normalizer=None))
    c=data['counts']; q=quality['counts']
    report=f'''# RBR-V2 U-only 首次工程交付总报告

**结论：结果B，`BLOCKED_U_SOURCE_INDEPENDENCE`。输入修复、来源依赖审计、候选重划分、共享z64及双Head实现已完成；正式训练没有启动。没有checkpoint、最佳epoch、训练loss曲线或模型效果结论。**

基准提交：`d2bbda12aa78e1fb15353d617f72be36cf5bacee`；分支`20260825_stageR_new`。本轮条件训练授权继承S1R已批准U-only边界及S3-P0单候选，不重新选择研究方向。S3-P0历史文件不回写，旧“未授权”记录保留其历史时点含义。

## 做成了什么

1. 新建`canonical83_v1`及Waymo/nuPlan实际对象适配器，旧Stage5/6/7完全保留。83个字段逐项定义、单位、有效掩码及实测误差见[字段与物理验证CSV](Canonical83_Fields_And_Parity.csv)。不再把[5,80,15]直接reshape成[80,75]。新接口直接按[time,slot,field]写入。
2. 全部51个原始TFRecord重新读取并核对既有SHA256；扫描{c['scenarios']:,}个Scenario、{c['vehicle_tracks']:,}条vehicle轨迹。原始record身份、偏移、payload SHA、时间、可用字段重新索引。
3. 对全部车辆的连续8帧移动片段逐帧错位匹配，共{c['moving_overlap_snippets']:,}个片段（精确和保守量化各一份检索键）。先组依赖，再固定哈希划分；没有用下游结果筛选。
4. 新候选窗口索引保留全部168,700个旧U身份，划分变为train {c['candidate_train']:,} / val {c['candidate_val']:,} / test {c['candidate_test']:,}。这是**未认证候选索引**，不是已经独立有效的新训练集。依赖组内不删除重复观测、不选择好样本；重复仍可能在同组同split内存在，训练权重与独立性不能混为一谈。
5. 全部候选自车重新做物理有效性审查：{q['eligible']:,}个满足至少64/80真实有效帧及至少2个相邻差分，{q['excluded']:,}个不满足，排除率{quality['exclusion_rate']:.4%}。有效候选train {q['eligible_train']:,} / val {q['eligible_val']:,} / test {q['eligible_test']:,}。这不是完整邻车上下文生产验收，更不是来源认证。
6. 固定哈希选取16个真实原始窗口，通过新生产适配器重新计算全部83维及mask/身份；{smoke['excluded_invalid']}个保留为input-invalid，没有更换它凑通过率。仅这16个真实canonical示例已物化；**没有宣称168,700窗口全部重建完成**。未在来源阻断下把候选升级成生产训练语料。
7. 单候选模型、双loss、train-only流式normalizer、冻结配置、U_val选择规则、训练拒绝入口及z64导出实现完成。两个Head分别反向传播到GRU的零优化步测试通过。正式训练和导出尚未获数据入口认证。

## 83维的物理合同与验证

80个固定10Hz时间位置，原始时间先保留float64/整数微秒，再映射0到7.9秒；不得把epoch秒转float32后差分。中心位置、完整二维速度、世界heading均有显式来源。自车位置与速度投影到首个有效自车heading坐标系；邻车相对位置与相对速度投影到当前自车heading。

closing为`-dot(p_n-p_e,v_n-v_e)/distance`，不再把自车速度减两次。同速8m/s、前方20m的样例得到closing=0、TTC无效、THW=2.5s。TTC是中心距离/径向接近速度的代理，不宣称车体碰撞预测；THW只在正纵向间距且自车正向速度成立时有效。无效TTC/THW保留999及mask=false，不能参与normalizer统计。

加速度取速度模长的后向物理差分，yaw-rate取**绝对世界heading**的包角差分；初帧、缺帧或邻车身份切换对应导数无效。没有平滑或跨缺口伪造状态。重采样只允许两个相邻有效源端点且间隔不超过0.15秒，不外推、不跨无效帧。空槽与地图/投影不可用分别通过presence和channel mask表示，禁止静默几何fallback。槽位仍调用现有生产lane-aware selector，Waymo侧邻接遵守proto分段拓扑。nuPlan使用官方Lane/Polyline接口转换整lane邻接；官方LaneConnector不能确定侧邻接时，四个侧槽的presence和全部连续字段均mask=false，不伪称没有侧车。两源拓扑粒度的差别由版本化地图适配与mask披露；新模型不把slot当永久车辆身份。

官方运动模型的EgoState速度在车体坐标使用；新nuPlan适配器获取车体中心完整速度并旋转到世界坐标，再与世界坐标交通对象速度比较。入口要求显式提供有证据的body/world声明；来源坐标未知时拒绝，绝不靠speed×heading猜测横向速度。当前证据支持官方模型对象适配，**不认证所有历史数据库包装器的坐标声明**。真实跨数据集不会具有相同驾驶轨迹；数值parity指同一规范运动经两套实际接口的对照，不是宣称真实Waymo与nuPlan样本逐值相等。

独立解析多项式oracle不调用生产旋转、差分或closing公式；Waymo typed protobuf与官方nuPlan EgoState/Agent/DetectionsTracks两入口分别核验。五个槽均有不同运动，验证80×83排列；另测侧向速度、角度环绕、缺帧、身份切换、同速跟车和大时间戳。容差abs 4e-5、rel 2e-6用于已声明的float32源heading/velocity量化，不用于修改科学测量阈值。

![物理输入核验（合成样例、零训练）](Canonical83_Physical_Validation.png)

## 为什么不能启动训练

观察到的依赖图为{data['observed_groups']:,}组，其中{data['multi_scenario_groups']:,}组含多个Scenario，最大{data['largest_group']}个Scenario。精确片段产生{data['union_edges'].get('E',0):,}条实际合并边，量化片段再产生{data['union_edges'].get('Q',0):,}条保守合并边；它们是并查集合并次数，不是全部重复对的数量。量化合并可能过于保守，不能把每条边都宣称为同一原始采集片段的确定证明。

其中{split['groups_spanning_legacy_splits']:,}个观察依赖组跨旧split，涉及{split['scenarios_in_these_groups']:,}个Scenario、{split['windows_in_these_groups']:,}个候选窗口。这表示这些窗口处在风险依赖组，不是认定其中每个窗口均为精确重复。新候选组跨split=0；S3-P0已确认的两条精确重复ego现处于同一test组。新旧split变化{split['windows_changed_candidate_split']:,}行，完全由冻结组哈希决定，不保留旧test来挑有利结果。

但全部{c['scenarios']:,}个原始Scenario时间起点都是0，schema及当前51份文件未提供原始采集segment、跨Scenario全局对象身份及绝对采集时间映射，camera/lidar附加帧均为0。TFRecord是容器，不能作为独立驾驶来源单位。Scenario ID互斥、文件互斥、精确/近似去重均不能补出这些缺失事实。

残余未知包括：同片段不同时间/不同车辆而没有重合片段、少于8帧或静止重叠、量化边界附近近似重叠，以及Scenario坐标原点变化使世界坐标匹配失效。没有声称穷尽模糊/平移/旋转等价的物理重叠，也没有仅凭相似的标准化匀速动作合并无关车辆。因此必须保留`RAW_SEGMENT_DISJOINTNESS_NOT_ESTABLISHED`，新候选数据`training_certified=false`。这不是可通过改mask、再跑几个测试或降低门槛解决的工程故障。

## 模型和训练合同的完成范围

固定`canonical83 + 83有效位 + t/7.9 → 单层GRU128 → 128-ReLU-64共享z → Semantic Head(64→64→13) + Temporal Head([z,t]65→128→83)`。无效ego位置不更新隐状态，不压缩物理时间。z接口[N,64]，不拼接语义真值或预测，不强制残差相减，不宣称已获得语义独立信息。

语义13目标来自U的速度、速度变化、加速度、jerk、绝对yaw-rate、总转角和中心路径长度，具体计算见`semantic13`和S3-P0定义；全部依据物理量重算，不使用旧Raw33错误导数或ego13距离监督。两个loss均通过共享z对编码器回传。通用目标是15%有效连续通道的2帧span遮挡重建，presence不作为重建目标；没有人工距离、triplet、机制标签或样本对挖掘。该监督提供已知语义约束，不构成新行为发现能力的自我证明。

固定AdamW lr0.001、weight_decay0.0001、batch128、clip1；唯一seed3407、最多30epoch、patience5、U_val总loss严格改善>1e-6、并列取早期；验证遮挡由数据hash/行身份/固定seed2026102301决定。U_test不用于训练或选择。配置继承S3-P0，取消其未经本轮授权的3408/3409，唯一run目录及24h设备预算不可通过换输出目录重置。

只有U_train可拟合输入与目标normalizer，无效通道排除；角度除π、presence不变、无效归零。由于完整来源认证失败，**本轮没有拟合正式normalizer、没有训练、没有选择checkpoint、没有输出真实U表示**。trainer/exporter是已完成的条件实现与零训练验证；尚不能称作经真实训练验收的生产模型。未来正式语料、normalizer、源映射及代码/config必须共同hash冻结后才能调用。

## 科学防火墙与保护

未读取TSB/SDC成功率、H30比较或BDD结果用于模型决策；没有访问Primary保留C。nuPlan仅用于官方源码和非科学合成对象测试。B1历史PASS、C2 FAIL、F2 Development PASS、旧P2 FAIL、SDC-v1历史结果和原账本均保留。S1R U/B/V/C、H30、Primary统计合同不改；语义模型将来仍须接受强人工基线的独立比较，loss下降不替代检测效用验证。

保护校验覆盖{protected['checked']:,}个既有文件（用户Quick文档只追加本轮命令章节）；受保护CSV SHA256保持`{protected['protected_csv_sha256']}`。原用户设计文档改动、Quick既有127行、未跟踪文件和nuplan requirements改动保留。

## 测试、资源与复现

10项单元/物理/梯度/防火墙测试通过；真实原始输入16个固定窗口全部产出（15满足输入门槛、1如实不合格），最终源码再次逐数组复现这16份产物，不覆盖原字节；完整原始身份与轨迹扫描完成。入口拒绝是预期的正确行为，不是失败后另换seed。新增源码全部通过实际训练运行时Python3.9编译；仓库全量tools通过系统Python3.13编译。全量历史tools在Python3.9编译发现旧`build_unified_bdd_posttraining_report.py:550`使用新版f-string语法，已如实保留失败日志，未改动此无关受保护脚本。临时依赖检查通过，日志随hash绑定。

全来源与重叠census实测墙钟{data['elapsed_seconds']:.2f}秒；全候选质量扫描{quality['elapsed_seconds']:.2f}秒；真实16窗口物化22.89秒、最大RSS245,694,464字节（见系统资源日志）。训练设备消耗严格为**0秒/24小时**，optimizer steps=0。未计量的编辑与编译耗时不伪称为精确总CPU。大型排序索引保存在本地并由hash绑定，不提交约2.8GB可再生中间文件；完整名单、依赖边、组、质量CSV及原始记录索引进入Git。

第一次raw解析因既有环境protobuf C++接口冲突停止；以独立target安装protobuf4.25.8修复，没有修改冻结nuplan环境。全槽fixture曾错误假设会发生换道/超越的运动仍保持固定槽位，修正解析fixture后通过，生产槽位规则未改；全部开发日志保留。这些是工程尝试，不是训练attempt，也未消耗科学样本。

## 唯一推荐的后续路径

获取带来源证据的`Scenario → 原始采集segment/session → 绝对时间/对象命名空间`映射。源持有者若不能提供，应改用能证明这些信息的新U来源；这属于外部数据选择，不能由本地去重算法伪造。随后在**新数据版本**中按真实来源合并现有依赖图、先分组再划分、完整物化canonical83、只在train拟合normalizer、重做零训练绑定，才能执行已固定seed3407候选。不得只把当前manifest的false改成true。当前没有需要Owner重新选择架构或主统计规则的事项；阻断是缺少可认证的来源事实。

本轮停止在来源阻断，已交付可复用输入/模型/审计实现。没有自动启动仿真、训练、下游比较或Primary。
'''
    (DOC/'Final_Report_CN_v1.md').write_text(report)
    sources=[*ROOT.glob('tools/rbr_v2_*.py'),ROOT/'tests/test_rbr_v2_uonly.py',ROOT/'configs/rbr_v2_uonly_v1/seed3407.json']
    artifact_paths=[p for p in OUT.rglob('*') if p.is_file() and 'runtime_packages' not in p.parts
        and p.name not in ('world_overlap_anchors.tsv','world_overlap_anchors.sorted.tsv','package.log')]
    local_intermediates={str(p.relative_to(ROOT)):dict(bytes=p.stat().st_size,sha256=sha(p))
                         for p in [OUT/'world_overlap_anchors.tsv',OUT/'world_overlap_anchors.sorted.tsv']}
    manifest=dict(status='RESULT_B_BLOCKED_U_SOURCE_INDEPENDENCE',base_git='d2bbda12aa78e1fb15353d617f72be36cf5bacee',
        training_runs=0,training_device_seconds=0,optimizer_steps=0,checkpoint=None,normalizer=None,
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in sources},
        artifact_sha256={str(p.relative_to(ROOT)):sha(p) for p in artifact_paths},
        document_sha256={str(p.relative_to(ROOT)):sha(p) for p in DOC.rglob('*') if p.is_file() and p.name!='Manifest_v1.json'},
        reproducible_local_intermediates=local_intermediates,protected=protected,runtime=runtime,
        official_sources=[{'url':'https://raw.githubusercontent.com/waymo-research/waymo-open-dataset/master/src/waymo_open_dataset/protos/map.proto',
                           'local':'evidence/map.proto'},
                          {'url':'https://raw.githubusercontent.com/waymo-research/waymo-open-dataset/master/src/waymo_open_dataset/protos/camera_tokens.proto',
                           'local':'evidence/camera_tokens.proto'}],
        self_hash='Excluded; final Git commit/tree binds this manifest')
    write(DOC/'Manifest_v1.json',manifest)
    print('Packaged RESULT_B_BLOCKED_U_SOURCE_INDEPENDENCE; protected files unchanged')


if __name__=='__main__': main()

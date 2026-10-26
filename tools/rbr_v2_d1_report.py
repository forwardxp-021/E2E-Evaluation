#!/usr/bin/env python3
"""Assemble one Chinese D1 report and traceable capacity/role evidence, no experiment launch."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import csv
import hashlib
import json
import subprocess
from pathlib import Path
from collections import Counter
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/rbr_v2_d1_v1';DOC=ROOT/'docs/stageR/rbr_v2_d1_v1';SDC=ROOT/'docs/stageR/e2e_scoped_direct_coast_v1'


def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def csvwrite(p,rows):
 with Path(p).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)


def main():
 DOC.mkdir(exist_ok=True)
 prior=read(SDC/'DEVELOPMENT_Execution_Binding_v1.json');bound={}
 for name,h in prior['data_sha256'].items():
  p=Path(name) if Path(name).is_absolute() else ROOT/name
  if sha(p)!=h:raise ValueError('Original D raw source/map binding changed '+name)
  bound[name]=h
 external={}
 for rel in ['nuplan/common/actor_state/dynamic_car_state.py','nuplan/planning/simulation/controller/motion_model/kinematic_bicycle.py','nuplan/planning/simulation/simulation_log.py','nuplan/common/maps/nuplan_map/lane.py','nuplan/common/maps/nuplan_map/lane_connector.py']:
  p=ROOT.parent/'nuplan-devkit'/rel;external[str(p)]=sha(p)
 import platform,importlib.metadata
 write(OUT/'Source_And_Runtime_Binding.json',dict(verified_original_D_source_map_sha256=bound,official_source_sha256=external,
       metadata_runtime=dict(python=sys.version,numpy=np.__version__,platform=platform.platform(),dependencies={n:importlib.metadata.version(n) for n in ['shapely','geopandas','torch','numpy']}),
       scientific_runner_called=False))
 cfg=read(ROOT/'configs/rbr_v2_d1_v1/contract.json');inp=read(OUT/'input_manifest.json');diag=read(OUT/'representation_diagnostics.json');inv=read(OUT/'local_source_inventory.json');resolve=read(OUT/'unfiltered_source_resolution.json')
 roles={r['session_id']:r for r in csv.DictReader((ROOT/'docs/stageR/s2r_role_aware_preflight/S2R_V_Eligibility_Census_v1.csv').open())}
 static={r['session_id']:r for r in csv.DictReader((ROOT/'docs/stageR/s2r_engineering_closure/S2R_Static_Eligibility_Census_v1.csv').open())}
 latest={sid:r['exposure_class'] for sid,r in roles.items()};events=[]
 def add(sid,label):
  if sid not in latest:raise ValueError('Exposure lacks original SESSION provenance')
  if int(label[1])>int(latest[sid][1]) or latest[sid].startswith('E5'):latest[sid]=label
 for p in ['docs/stageR/b1_resume_execution_r2/B1_R2_Exposure_Ledger_Update_v1.csv','docs/stageR/tsb_direct_coast_p2_q1/P2_Q1_Exposure_Update_v1.csv','docs/stageR/e2e_scoped_direct_coast_v1/Exposure_Ledger_v1.csv']:
  for e in csv.DictReader((ROOT/p).open()):add(e['session_id'],'E2_BENCHMARK_ENGINEERING')
 assessed={};evidence={}
 for phase,rev in [('DEVELOPMENT',4),('QUALIFICATION',2)]:
  p=SDC/f'evidence/{phase}_Applicability_v{rev}.json'
  for r in read(p)['rows']:
   if r['status']!='POLICY_EXCLUDED':assessed[r['session_id']]=r;evidence[r['session_id']]=str(p.relative_to(ROOT))
 additional=read(OUT/'additional_static_results.json')
 for r in additional['rows']:
  sid=r['session_id']
  if sid in assessed:raise ValueError('Duplicated prior static inspection')
  assessed[sid]=r;evidence[sid]='outputs/rbr_v2_d1_v1/additional_static_results.json';add(sid,'E2_BENCHMARK_ENGINEERING')
  events.append(dict(session_id=sid,event='SAME_RULE_METADATA_APPLICABILITY_AUDIT',role='B_METADATA',previous_class=roles[sid]['exposure_class'],new_class=latest[sid],future_V_current_claim='STATIC_ELIGIBILITY_ONLY',future_C='NO_E5_RELABELLING',run_id='',source=evidence[sid]))
 d_audit=read(OUT/'D_E3_Static_Results.json');d_extra={r['session_id'] for r in d_audit['rows'] if r['status']=='ELIGIBLE_D_ONLY'};d_rows={r['session_id']:r for r in d_audit['rows']}
 for sid in d_rows:
  events.append(dict(session_id=sid,event='D_ONLY_E3_STATIC_SOURCE_AUDIT_NO_V_RESTORATION',role='D_METADATA',previous_class=latest[sid],new_class=latest[sid],future_V_current_claim='EXCLUDED_E3',future_C='EXCLUDED_E3',run_id='',source='outputs/rbr_v2_d1_v1/D_E3_Static_Results.json'))
 d_ids=sorted({r['session_id'] for r in cfg['runs']})
 for sid in d_ids:
  previous=latest[sid];add(sid,'E3_PRIMARY_METHOD_DEVELOPMENT')
  events.append(dict(session_id=sid,event='D_INTERFACE_REPRESENTATION_AND_BASELINE_SCALER_FIT',role='D_INTERFACE_FIT_ONLY',previous_class=previous,new_class=latest[sid],future_V_current_claim='EXCLUDED',future_C='EXCLUDED',run_id=';'.join(r['run_id'] for r in cfg['runs'] if r['session_id']==sid),source='configs/rbr_v2_d1_v1/contract.json'))
 for r in inv['new_source_rows']:
  events.append(dict(session_id=r['session_id'],event='UNFILTERED_LOG_HEADER_TAGCOUNT_SOURCE_HASH_ONLY',role='B_METADATA',previous_class=roles.get(r['session_id'],{}).get('exposure_class','OUTSIDE_APPROVED_FRAME'),new_class=latest.get(r['session_id'],'UNRESOLVED_NO_TAGGED_ANCHORS'),future_V_current_claim='NO_NEW_ELIGIBLE_SESSION',future_C='NO_NEW_C_RESERVATION',run_id='',source='outputs/rbr_v2_d1_v1/unfiltered_source_resolution.json'))
 csvwrite(OUT/'Exposure_Update.csv',events)
 eligible={sid for sid,r in assessed.items() if r['status']=='ELIGIBLE'};v=eligible-set(d_ids);C={sid for sid,r in roles.items() if r['exposure_class'].startswith('E5')}
 if len(eligible)!=52 or len(v)!=49 or len(assessed)!=187 or len(d_extra)!=10:raise ValueError('Capacity denominator mismatch')
 q={r['session_id'] for r in read(SDC/'QUALIFICATION_Frozen_Roster_v2.json')['roster']}
 if set(d_ids)&q or set(d_ids)&C:raise ValueError('D source conflicts with qualification/C')
 rows=[]
 for sid,r in roles.items():
  a=assessed.get(sid);base=static[sid];status=a['status'] if a else 'NOT_ASSESSED_POLICY_OR_SOURCE_PRESELECTION'
  rows.append(dict(session_id=sid,original_class=r['exposure_class'],current_class=latest[sid],original_representative_log=base['representative_log_id'],SDC_static_status=d_rows[sid]['status'] if sid in d_rows else status,SDC_static_evidence='outputs/rbr_v2_d1_v1/D_E3_Static_Results.json' if sid in d_rows else evidence.get(sid,''),D_bound=sid in d_ids,D_only_additional_static_candidate=sid in d_extra,V_claim_B_static_candidate=sid in v,V_full_primary_certified=False,C_protected=sid in C,C_SDC_eligibility='NOT_ACCESSED' if sid in C else 'NOT_C',existing_SDC_qualification_pair=sid in q,reason='D_INTERFACE_E3' if sid in d_ids else ';'.join(a.get('reasons',[])) if a else ';'.join(next(x for x in read(SDC/'Pool_Split_v1.json')['rows'] if x['session_id']==sid)['policy_reasons'])))
 for sid in inv['new_SESSION']:
  rows.append(dict(session_id=sid,original_class='OUTSIDE_248_ELIGIBLE_FRAME',current_class='NO_TAGGED_ANCHOR_SOURCE_INELIGIBLE',original_representative_log='',SDC_static_status='NO_TAGGED_ANCHORS',SDC_static_evidence='outputs/rbr_v2_d1_v1/unfiltered_source_resolution.json',D_bound=False,D_only_additional_static_candidate=False,V_claim_B_static_candidate=False,V_full_primary_certified=False,C_protected=False,C_SDC_eligibility='NO_RESERVATION',existing_SDC_qualification_pair=False,reason='ALL_LOCAL_CHUNKS_HAVE_ZERO_SCENARIO_TAG_ANCHORS; no automatic new frame'))
 csvwrite(OUT/'SESSION_Capacity_Audit.csv',rows)
 summary=dict(status='RBR_H30_D_STAGE_BLOCKED_WITH_ACTIONABLE_CAUSE',raw_local_physical_DB_files=inv['physical_DB_files'],raw_unique_chunks=inv['unique_filename_chunks'],raw_SESSION=253,original_eligible_source_SESSION=248,
              same_rule_V_B_static_inspected_SESSION=187,D_only_E3_static_inspected_SESSION=31,D_only_E3_static_eligible=10,D_total_static_potential_including_V_shared=62,same_rule_static_eligible=52,additional_inspected=29,additional_eligible=9,D_interface_fit_complete=3,D_final_calibration_bound=0,D_power_coverage_validation_completed=False,
              V_claim_B_static_candidates=49,V_already_SDC_benchmark_qualified=20,V_other_static_candidates_need_SDC_qualification=29,V_final_roster_or_outcome_access=0,C_protected_raw=5,C_previous_broad_static=4,C_SDC_certified=0,C_SDC_actual_capacity='UNKNOWN_NOT_ACCESSED',
              operative_exposure_counts={**dict(Counter(latest.values())),'E4_PRIMARY_OUTCOME_DRIVEN_ADAPTATION':0},unpaired_minimum_planning={'cal':80,'AA':80,'AB':160,'total':320,'shortfall_eligible_SESSION':261,'minimum_additional_V_eligible':191,'remaining_calibration_shortfall_after_ten_D_only':70},paired_minimum_planning={'cal':40,'AA':40,'AB':80,'total':160,'shortfall_eligible_SESSION':101,'minimum_additional_V_eligible':71,'remaining_calibration_shortfall_after_ten_D_only':30},
              bare_unpaired_distinct_draw_role_floor=120,bare_paired_distinct_draw_role_floor=60,bare_floors_are_not_power_approved=True,
              capacity_claim='V/B source frame unchanged; legacy E3 assessed separately for D-only. Deficits assume all10 D-only candidates become final-cal eligible; no power/sample-size certification',
              source_interval='NOT_ESTABLISHED',formal_FPR_gate='NOT_EVALUATED',RBR_vs_H30_power='NOT_ESTABLISHED',new_scientific_runs=0)
 write(OUT/'Capacity_Summary.json',summary)
 # Domain shift diagnostics from frozen U statistics, never U-vs-D MMD or encoder tuning.
 norm=read(ROOT/cfg['normalizer']);n=read(ROOT/'outputs/rbr_v2_prod_v1/data_manifest.json')['rows'];u_counts=np.array(norm['input']['count']);nu_mask=np.array(diag['channel_valid_fraction'])
 domain=dict(Waymo_train_valid_channel_fraction=float(u_counts.sum()/(n*80*83)),nuPlan_development_valid_channel_fraction=diag['channel_mask_valid_fraction'],Waymo_channel_valid_fraction=(u_counts/(n*80)).tolist(),nuPlan_channel_valid_fraction=nu_mask.tolist(),
             nuPlan_neighbor_present_cells=[r['neighbor_present_cells'] for r in inp['rows']],nuPlan_presence_known_cells=[r['presence_known_cells'] for r in inp['rows']],
             ego_position_reference='Waymo center; nuPlan center with rigid-body center velocity; frozen H retains rear axle summary',
             slots=['front','left_front','left_rear','right_front','right_rear'],units='m, m/s, m/s^2, rad, rad/s, seconds; canonical derivative0.1s; time integer us',
             missing_policy='Mask unknown lane/connector side topology and identity-discontinuous derivatives; invalid normalized channels zero; do not invent missing neighbors',
             map_difference='Waymo index-range adjacency versus nuPlan whole-lane adjacency; connector side adjacency unknown and masked',
             same_index_replay='DISCLOSED_OFFLINE_SAME_INDEX_REPLAY_NOT_STRICT_EVENT_TIME_CAUSALITY',no_normalizer_refit_for_nuplan_encoder=True)
 write(OUT/'Input_Domain_And_Mask_Audit.json',domain)
 # Every original tracked byte is protected; Quick is allowed only an appended command section.
 before=read(OUT/'protected_before.json');changed=[]
 for p,h in before.items():
  if p=='QUICK_REFERENCE.md':continue
  if sha(ROOT/p)!=h:changed.append(p)
 if changed:raise ValueError('Historical/user file modified '+str(changed))
 suffix=OUT/'quick_section.md'
 if suffix.exists():
  quick=(ROOT/'QUICK_REFERENCE.md').read_bytes();section=suffix.read_bytes()
  if not quick.endswith(section) or hashlib.sha256(quick[:-len(section)]).hexdigest()!=before['QUICK_REFERENCE.md']:raise ValueError('User Quick prefix changed')
 protected=dict(checked=len(before)-1,changed=changed,checkpoint_sha256=sha(ROOT/cfg['checkpoint']),protected_csv_sha256=sha(ROOT/'outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/behavior_events_v2/behavior_event_metrics_v2.csv'))
 if protected['checkpoint_sha256']!=cfg['checkpoint_sha256'] or protected['protected_csv_sha256']!='e8deb93312e82183b6c2c0db30fd18cbf9c32d32d566038419a5be65b389d9d8':raise ValueError('Protected model/CSV identity failure')
 write(OUT/'protected_after.json',protected)
 synth=read(OUT/'synthetic_AA_diagnostic.json')
 report=f'''# RBR-V2-D1：冻结模型、公平接口与BDD容量闭环

**`RBR_H30_D_STAGE_BLOCKED_WITH_ACTIONABLE_CAUSE`**。接口完成；原BDD主实验的独立SESSION容量、真实D-only误报/功效/区间覆盖验证仍不足。附加`U_SOURCE_COMPLETE_INDEPENDENCE_NOT_ESTABLISHED`永久保留。

## 三个直接答案

1. **RBR64与H30已有真实同条件接口。** 全部既定3个SDC开发SESSION、6个历史arm，同一序列的80个真实EgoState、同index观测、规范10Hz时刻、车辆/LOG/SESSION绑定后，使用原Canonical83、正式Waymo normalizer与唯一epoch30 encoder输出[6,64]；原`handcrafted_h`完整输出[6,30]。没有读20对SDC资格样本的表示或检测结果，没有挑样本。
2. **现有冻结适用框不能支持原Primary。** 补查29个历史SESSION新增9个适格；V/B原框187个政策/原预选允许检查的SESSION中52个静态适格。隔离本轮3个D后，剩49个Claim-B静态候选，其中已有SDC基准资格的20个、仅静态适格尚需SDC执行的29个。49不是49个已通过Primary科学资格或已冻结V名单。最小非配对规划80+80+160=320，缺至少261个适格SESSION；配对40+40+80=160，缺至少101个。已将旧E3新查出的10个D-only静态候选作为潜在校准资源计入，尚未科学执行；两者都是规划包络，不是已证实功效样本量，缺口还未包含新增D功效验证和C资源。
3. **真实D仅完成接口、三SESSION baseline预处理拟合、输入/数值/重复性检查。** 新增10个E3静态D候选没有本SDC版本的既存两臂数据，本轮不启动新rollout；3个独立SESSION不能构造每侧m20的真实A/A，也不能认证误报、功效或源域区间。没有真实RBR/H30 alarm概率、检测差值或Primary结论。合成A/A只验证统计代码。

## 生效合同、固定模型与防火墙

S1文件虽然保留Draft名称，其已批准内容结合S1R Owner Approval Record解释。S1R已生效，替换绝对历史暴露禁用及Q自动变D；保留U-only encoder、E3/E4排除、SESSION聚类、强H30、校准/评价隔离及paired/unpaired主张边界。S2R旧214静态候选使用较宽3.61m/s及native-route条件，不能直接代替SDC的≥5m/s、低曲率、完整车体走廊与离线交通条件。

唯一checkpoint：`outputs/rbr_v2_prod_v1/run/epoch30.pt`，SHA256 `{cfg['checkpoint_sha256']}`。normalizer SHA `{cfg['normalizer_sha256']}`。网络、seed3407、epoch30、双loss、训练参数及H源码无改动。不加载Smoke模型，不创建optimizer，不用D/SDC/BDD选择encoder。

先写`configs/rbr_v2_d1_v1/contract.json`锁定全部6个原SDC开发arm，再提取真实输入。仅这3个SESSION明确分配D_INTERFACE_FIT_ONLY；因使用其表示及baseline数值拟合检测scaler/带宽，本轮追加E3，禁止作为该管线独立V/C证据。不是将所有B/Q自动转换D。未打开C保护原始源、正式V/C结果或未来评价样本。资格20对只使用既有名单元数据计数，未使用其曲线/机制/安全/检测值选规则。旧B1、C2、旧P2、SDC、Smoke和正式训练历史不变。

## 同轨迹提取、物理输入与域偏移

只反序列化已有官方simulation log，不调用runner、tracker或车辆传播。原execution_manifest全部产物SHA核验；80个EgoState与冻结current-ego trace逐点比较rear axle位置、heading、速度及时间，并与canonical clock绑定。H与RBR使用完全相同帧/车辆/来源，表示定义的参考点保持各自冻结合同：RBR采用车体中心，H的历史位置/路径摘要采用rear axle，不能为数值相同而改写H。nuPlan官方模型的body-frame中心速度旋转到world后，再转入初始自车坐标；有独立三角公式和原物理oracle验证。

83维仍为8+5×15。坐标m、速度m/s、加速度m/s²、heading rad、yaw rate rad/s、TTC/THW秒；80帧相差100000µs。相对速度、closing speed、时间导数、角度环绕、身份连续性与无效值mask沿用Canonical83。Waymo的分段邻接和nuPlan whole-lane邻接不能被伪称完全一致；连接器侧邻接未知时mask，不用几何猜测补齐。

Waymo完整U_train有效channel占{domain['Waymo_train_valid_channel_fraction']:.4%}；本轮nuPlan6个arm占{domain['nuPlan_development_valid_channel_fraction']:.4%}。各arm邻车presence帧数为{domain['nuPlan_neighbor_present_cells']}；缺失包含真实无邻车、lane/connector不可观测和连续导数边界，不能全解释为传感器故障。SDC交通走廊筛选和地图拓扑造成明显输入域差异，mask与80帧身份全部保存。不重新fit encoder normalizer、删异常值、插值救样本或修改网络。

归一化有效输入|x|的p99={diag['normalized_valid_abs_p99']:.6f}，最大={diag['normalized_valid_abs_max']:.6f}，|x|>5占{diag['normalized_fraction_abs_gt5']:.6%}，>10占{diag['normalized_fraction_abs_gt10']:.6%}。这是诊断，无新阈值筛选。

z64全有限、同输入eval逐位一致；64列方差范围[{diag['variance_min']:.8f},{diag['variance_max']:.8f}]，未见常量塌缩。6行中心化数值秩{diag['centered_numerical_rank']}（数学上限5），熵有效秩{diag['effective_rank']:.6f}；3对B/T的L2差{diag['pair_L2_distances']}。它们说明本组输出不相同，不能证明完整64维有效利用、跨场景泛化、足够检测功效或优于H。早期秩诊断曾使用float32均值导致虚假第6秩；修正为float64中心化，仅改诊断，没有改模型/数据，原日志保留。

H30的13个原ego13、1个mean_abs_accel、8个signed acceleration分箱、3个ACF、2个制动mass摘要、2个validity和1个cadence全部保留。强H对时序可能敏感，不能删列。encoder的U normalizer与BDD的D_fit baseline scaler是两层不同处理。

## 原SAP数值接口：实现与尚未认证部分

`rbr_v2_d1_statistics.py`实现无偏二次MMD²，保留负值；各表示在同3个D_fit baseline SESSION分别拟合population mean/SD，SD≤1e-12列scale1；相同median-positive-squared-distance规则，sigma=sqrt(median/2)，全常数管线明确degenerate而非epsilon救援。工程scaler不冒充正式校准。

固定alpha=.05、m20、pi=.50、ΔBDD、δ*=.10与原SAP不变。非配对每次40个不同原始SESSION，20 reference/20 target且target恰好10 treatment；两表示共享身份和随机mask。1999校准抽样seed2026091101、阈值第1900顺序统计量、严格>报警、tie不报警。独立AA/AB角色池不与D_fit/彼此重叠。

提供D-only条件统计与源cluster bootstrap：SESSION全部arms共同携带；bootstrap使用原始ID权重并在release内无放回分配，重复抽样副本不能变成不同SESSION。外层2000、内层2000、97.5%各管线FPR上限和2.5/97.5%差值区间；不足唯一支持直接阻断，不删除失败replicate。缺少真实D覆盖率验证时接口拒绝产生源域CI。该实现仅通过确定性单元检查，**没有取得真实覆盖验证证书**。Primary成功/失败及FPR门仍按原SAP，当前不可执行。

配对交换primitive保留whole-SESSION依赖，但不激活fallback、不借非配对阈值；仍需预注册成对null/独立pair AA、功效与覆盖验证。它测试controlled assignment/swap null，不是operational release FPR。基线复制成左右相同数组不构成有方差的独立A/A。反复随机抽同一小池也不能增加独立样本。

8项统计测试：负MMD²、标签交换/顺序不变（1e-14浮点容差）、常数/近常数尺度、SESSION重复及支持不足拒绝、共享20/20与10个treatment身份、成对交换、D角色隔离和阈值/tie、原SAP成功/失败/FPR判定。7项既有输入/来源测试（含独立物理oracle）复核两真实处理接口、五槽、world yaw、同速closing、刚体中心速度、缺失和物理时间规则。首个严格逐位标签交换断言曾因约2e-16求和末位差失败，改为明确数学容差；原失败日志保留，没有改统计量。

确定性合成高斯A/A：D_fit80、cal80、AA80、AB-null160（全为合成身份，AB也未植入漂移），1999校准/10000评价draw。AA报警率：64维{ synth['AA']['alarm_probability']['RBR64']:.4f}、30维{ synth['AA']['alarm_probability']['H30']:.4f}。这是合成张量维度的代码检查，**不是实际RBR/H30误报或性能**，也没有认证source-domain CI、功效或真实FPR gate。

## SESSION容量闭环与覆盖率

本地3370个物理DB路径只有1621个不同chunk文件名、253个SESSION前缀；副本、scenario token、同SESSION其它chunk均不新增独立单位。原S1R适格source frame覆盖1564 chunks、248 SESSION。对差异57个chunk核对只读log header、tag数及完整DB SHA：45个无tagged anchor，12个在原绑定黑名单。额外5个SESSION所有本地chunk的tagged anchor总数均0；它们已在旧unfiltered source universe中，不是新可用来源。故现行规则下本地未扫描目录没有新增适格SESSION；更换anchor类型或代表LOG属于新的前瞻性抽样框，不能暗中算作原合同容量。

旧资格池138→37只是该冻结资格子池覆盖26.81%。本轮把原S1R允许复用、此前仅因基准独立资格排除的29个历史SESSION也按同一原代表LOG、16-anchor哈希顺序、官方唯一可运行身份、初速/路线/走廊/离线交通规则审查，9适格、20无适格anchor；没有runner或新闭环结果参与选择。与旧开发池20→6、资格池138→37合计：187检查→52适格，覆盖52/187=27.81%；相对248为20.97%。其余61个保留原政策/预选原因（E3、C保护、初速或静态源条件），不是科学rollout FAIL。

另对旧31个E3作事前固定的D-only静态审查：复用原S2R hash/top256选择首个宽域代表，保留3.61m/s再经SDC≥5m/s代表预选及原16-anchor物理规则，不把它们插回原V框。10个静态D候选、21个未适格；31个保持E3，尚无本SDC版本的两臂轨迹，不能据此开展真实D A/A或称其已通过科学资格。D_plan、拒绝记录及SHA完整保留。

3个D隔离后：**D接口已执行3；额外D-only静态候选10；另49个V候选可改绑D但必须相应从V扣除；D正式最终校准0；V Claim-B静态49；C保护raw5、旧宽域静态4、本SDC域资格未知且未访问（已认证0）**。D/V/C不是可以重叠相加的永久标签，未来分配final-cal/AA/AB需互斥；本轮没有冻结最终V/C身份。当前E3合计{summary['operative_exposure_counts'].get('E3_PRIMARY_METHOD_DEVELOPMENT')}，E4保持0，不把E2/E3洗成E5。

即使只看构造一次release所需的原始ID下限，非配对三个互斥cal/AA/AB池至少40+40+40=120，配对至少20+20+20=60；计入10个D-only后共59个可分配静态候选也不足；这不是批准把规划容量降到120或60。原SAP规划小档非配对320、配对160，更不能靠20对B基准或重复Monte Carlo补足。新49候选也不能无条件安全/科学资格PASS。功效≥.80、δ*.10/设计替代.20、source CI覆盖、两管线FPR上界≤.075均没有实际D证据，因此不得Ready。

旧B1/SDC的机制、匹配、安全暴露按S1R可作为E2用于冻结后的Claim-B V（需同干预合同、角色认证和完整分母）；不能声称Claim-A未见SESSION。旧B1曲线不是SDC控制，不能直接填充SDC treatment；可复用其SESSION做前瞻性新执行，但本轮只做静态审查。用于D_fit、核/校准开发、功效或RBR/H30观察而影响方法的SESSION需E3隔离；结果驱动改变样本/方法则E4。20个SDC资格SESSION本轮保留，未拿其检测结果开发接口或筛样本。

## 最小可执行下一步与预算边界

先解决**独立来源容量**，不重训encoder，不改H30/m/pi/FPR/SAP，不在已知成功样本上继续挑选。按现行适用框，配对规划下，在10个D-only均能满足最终校准合同的乐观条件下，还需至少101个适格SESSION（其中至少71个必须可用于独立V，另30个补校准）；非配对至少261个（至少191个V、另70个校准）。这只是final-cal+评价规划缺口下界，尚需独立D功效/覆盖验证资源与C资源。若扫描全SESSION其它时刻/非tagged anchors，必须先前瞻性冻结新的抽样框并披露覆盖，不能因本轮表示或任何未来检测结果调整；不能冒充当前规则已足够。

可审查的后续**D-only算法验证预算提案（未执行、无自动授权）**：在来源足够后，按最小paired规划包络为算法验证准备互斥40/40/80共160个D SESSION，两臂各一次共320个首次arm，技术重试0、每arm墙钟≤1200秒、总CPU建议≤10800秒。之前SDC40arm实测子进程CPU约400秒，线性参考约3200秒/320arm，仅作条件估算；不能保证新来源成本或以估算替代计量。D-only这160个一旦用于功效/覆盖开发，就不能再当该方法独立V/C证据，也不是已批准转成paired Primary；后续final-cal/V另行分配。优先先获取/核对来源，容量不足时不启动该预算。

若无法获得足够独立日志，保留Level1基准构建与可复现表示模型贡献，现有Primary保持BLOCKED；不能降低统计要求制造论文正结论。C的低暴露小样本若不足，须如实报告稳健性证据不足，不把同一SESSION重复抽样作为扩容。

## 文件、运行与验收

新增：`rbr_v2_d1_interface.py`（extract/encode）、`rbr_v2_d1_capacity.py`（inventory/scan）、`rbr_v2_d1_d_capacity.py`（旧E3的D-only静态容量）、`rbr_v2_d1_statistics.py`（SAP primitives）、`rbr_v2_d1_synthetic.py`（纯合成A/A）、`rbr_v2_d1_report.py`（只读汇总）、统计测试与唯一配置。真实输入npz、同身份RBR64/H30数组、完整mask/时钟/来源、scaler、容量CSV/JSON及增量Exposure_Update均在`outputs/rbr_v2_d1_v1/`；原大文件路径及SHA可定位。

运行命令、期望行为和通过标准见QUICK_REFERENCE新增D1章节。已运行真实历史接口提取/正式模型推理、全目录source metadata核查、29个SESSION静态扩展、31个旧E3 D-only静态审查、合成A/A及15项测试；编译与no_tmp检查日志随Manifest绑定。静态查询产生的官方map NaN cast警告保留，未隐藏错误或放宽路线条件。

完整保护核验{protected['checked']}个既有tracked文件无变更（Quick只追加命令，用户前缀完整）。受保护CSV SHA保持`{protected['protected_csv_sha256']}`。模型、normalizer、H和Canonical83源码逐SHA复核。实际新增rollout、训练、checkpoint选择、真实RBR/H30检测比较、Primary/V/C均0。只读静态检查与固定模型工程推理不计为科学仿真。对31个E3的D-only原始查询也由zero-run guard验证，没有恢复任何V身份。

**本轮完成并停止：工程接口已交付；实际阻断为独立适格SESSION容量与真实D统计验证，不是模型训练链路。**
'''
 (DOC/'Final_Report_CN_v1.md').write_text(report)
 write(DOC/'Status_v1.json',dict(status=summary['status'],interface='PASS_REAL_SIX_ARM',capacity='BLOCKED',D_real_AA_power_coverage='BLOCKED_INSUFFICIENT_DATA',model_checkpoint_sha256=cfg['checkpoint_sha256'],source_risk=cfg['source_risk'],new_scientific_simulation=0,training=0,primary=0,V=0,C=0))
 # Hash governing protocols without altering their historical status.
 governing=[]
 for folder in ['s1','s1r_owner_amendment','s2r_engineering_closure','s2r_role_aware_preflight']:
  governing.extend(p for p in (ROOT/'docs/stageR'/folder).rglob('*') if p.is_file())
 governing.extend(ROOT/p for p in ['tools/rbr_v2_canonical83.py','tools/rbr_v2_model.py','tools/s1_protocol_schema.py','tools/stage6l_prepare_context_representation_ablation.py','tools/r1_closed_loop_benchmark_v2_1.py','tools/sdc_v1_selection.py','docs/stageR/e2e_scoped_direct_coast_v1/Frozen_Qualification_Protocol_v1.json','docs/stageR/e2e_scoped_direct_coast_v1/Pool_Split_v1.json','docs/stageR/e2e_scoped_direct_coast_v1/Exposure_Ledger_v1.csv','docs/stageR/e2e_scoped_direct_coast_v1/Manifest_v1.json','docs/stageR/r1/r1_fresh_smoke_source_universe_v0.1.json','docs/stageR/r1/r1_technical_smoke_v1_permanent_blacklist_v1.json','docs/stageR/b1_resume_execution_r2/B1_R2_Exposure_Ledger_Update_v1.csv','docs/stageR/tsb_direct_coast_p2_q1/P2_Q1_Exposure_Update_v1.csv',cfg['checkpoint'],cfg['normalizer']])
 sources=list(ROOT.glob('tools/rbr_v2_d1*.py'))+list(ROOT.glob('tests/test_rbr_v2_d1*.py'))+[ROOT/'configs/rbr_v2_d1_v1/contract.json']
 artifacts=[p for p in OUT.rglob('*') if p.is_file() and p.name not in ['report_packaging.log']]
 docfiles=[p for p in DOC.rglob('*') if p.is_file() and p.name!='Manifest_v1.json']
 write(DOC/'Manifest_v1.json',dict(status=summary['status'],base_commit='f8d308a5f358284317e878318dd50754f26d1c29',
       source_sha256={str(p.relative_to(ROOT)):sha(p) for p in sources},governing_sha256={str(p.relative_to(ROOT)):sha(p) for p in governing},
       artifact_sha256={str(p.relative_to(ROOT)):sha(p) for p in artifacts},document_sha256={str(p.relative_to(ROOT)):sha(p) for p in docfiles},
       raw_source_hashes='input_manifest.json, local_source_inventory.json and additional_static_results.json bind each accessed raw LOG/map metadata source; protected C raw not opened',protected=protected,self_hash='Bound by final Git commit; excludes itself'))
 print(summary['status']);print(json.dumps(summary,indent=2))

if __name__=='__main__':main()

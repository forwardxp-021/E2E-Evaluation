#!/usr/bin/env python3
"""Chinese D3 final report, full denominator, recorded curves and provenance; no model/runner invocation."""
import os
import sys
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse,csv,json,math,subprocess
from pathlib import Path
import numpy as np
from tools.rbr_v2_d2_download import ROOT,read,write,sha
OUT=ROOT/'outputs/rbr_v2_d3_v1';DOC=ROOT/'docs/stageR/rbr_v2_d3_v1';SCIENCE=ROOT/'outputs/stageR/rbr_v2_d3'


def csvwrite(path,rows):
 with path.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)


def plots():
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 old=read(OUT/'reference_lqr_diagnosis.json');new=read(OUT/'patched_reference_v2_diagnosis.json');origin=np.array([old['reference_states'][0]['x'],old['reference_states'][0]['y']])
 fig,ax=plt.subplots(2,2,figsize=(12,8));curve_rows=[]
 for d,label,color in [(old,'Original planned reference','firebrick'),(new,'Arc-bound planned reference','green')]:
  xy=np.array([[r['x'],r['y']] for r in d['reference_states'][:4]])-origin;ax[0,0].plot(xy[:,0],xy[:,1],'o-',color=color,label=label)
  for i,(x,y) in enumerate(xy):ax[0,0].annotate(str(i),(x,y),fontsize=8)
 ax[0,0].set(title='Deterministic reference replay: first future step',xlabel='Relative world X (m)',ylabel='Relative world Y (m)');ax[0,0].legend(fontsize=8);ax[0,0].grid(alpha=.2)
 oldroot=ROOT/'outputs/stageR/rbr_v2_d2_D/RBR-D2-D-01-TREATMENT';oldtrace=[json.loads(l)['current_ego'] for l in (oldroot/'trace/realized_current_ego.jsonl').read_text().splitlines()]
 ax[0,1].plot(np.arange(len(oldtrace))*.1,[r['rear_axle']['heading'] for r in oldtrace],'o-',color='firebrick',label='Old D2 incomplete actual state')
 for arm,color in [('BASELINE','tab:blue'),('TREATMENT','tab:orange')]:
  root=SCIENCE/'SCIENCE'/('D3-D-01-'+arm);states=[json.loads(l)['current_ego'] for l in (root/'trace/realized_current_ego.jsonl').read_text().splitlines()];controls=[json.loads(l) for l in (root/'telemetry/actual_lqr_controller_telemetry.jsonl').read_text().splitlines()];review=read(root/'arm_science_review.json')
  t=np.arange(80)*.1;accel=np.r_[controls[0]['model_acceleration_before'],[r['model_realized_acceleration'] for r in controls]];u=np.r_[[r['longitudinal_requested_acceleration'] for r in controls],np.nan]
  ax[0,1].plot(t[:12],[r['rear_axle']['heading'] for r in states[:12]],'.-',color=color,label='D3 '+arm)
  ax[1,0].plot(t,accel,color=color,label=arm+' actual model a');ax[1,0].step(t,u,where='post',ls=':',color=color,label=arm+' request')
  ax[1,1].plot(t,[r['speed_mps'] for r in states],color=color,label=arm)
  for i,s in enumerate(states):curve_rows.append(dict(run_id='D3-D-01-'+arm,frame=i,episode_s=float(t[i]),model_state_accel_mps2=float(accel[i]),frozen_median3_speed_accel_proxy_mps2=review['mechanism']['acceleration_from_median3_speed'][i],executed_longitudinal_request_mps2=float(u[i]) if i<79 else '',speed_mps=s['speed_mps'],heading_rad=s['rear_axle']['heading'],rear_x=s['rear_axle']['x'],rear_y=s['rear_axle']['y']))
 ax[0,1].set(title='Recorded heading before first brake',xlabel='Episode time (s)',ylabel='Heading (rad)');ax[0,1].legend(fontsize=8);ax[0,1].grid(alpha=.2)
 ax[1,0].axvspan(2.1,3.9,color='green',alpha=.09,label='Treatment zero-request interval');ax[1,0].set(title='Actual official model acceleration and final request',xlabel='Episode time (s)',ylabel='Acceleration (m/s2)');ax[1,0].grid(alpha=.2)
 ax[1,1].set(title='Actual speed: matched endpoint, different timing',xlabel='Episode time (s)',ylabel='Speed (m/s)');ax[1,1].legend(fontsize=8);ax[1,1].grid(alpha=.2)
 handles,labels=ax[1,0].get_legend_handles_labels();fig.legend(handles,labels,ncol=3,fontsize=7.5,loc='lower center',bbox_to_anchor=(.5,.01))
 fig.tight_layout(rect=(0,.08,1,1));fig.savefig(DOC/'D3_Recorded_Root_And_Braking_Curves_v1.png',dpi=170);plt.close(fig);csvwrite(DOC/'D3_Representative_Actual_Curves_v1.csv',curve_rows)


def protection():
 before=read(OUT/'protected_before.json');own=(OUT/'quick_section.md').read_text();q=(ROOT/'QUICK_REFERENCE.md').read_text();assert q.endswith(own)
 bad=[]
 for name,h in before.items():
  actual=__import__('hashlib').sha256(q[:-len(own)].encode()).hexdigest() if name=='QUICK_REFERENCE.md' else sha(ROOT/name)
  if actual!=h:bad.append(name)
 d2=read(ROOT/'outputs/rbr_v2_d2_v1/Provenance_File_Index_v1.json');bad2=[p for p,v in d2['files'].items() if sha(ROOT/p)!=v['sha256']]
 native=read(ROOT/'docs/stageR/tsb_direct_coast_f2/F2_Execution_Source_Binding_v1.json')['source_config_sha256'];badnative=[p for p,h in native.items() if sha(ROOT/p)!=h]
 result=dict(initial_tracked_assets=len(before),changed_initial_assets=bad,D2_bound_files=len(d2['files']),changed_D2_files=bad2,official_F2_native_source_configs=len(native),changed_official_files=badnative,
       protected_CSV_sha256=sha(ROOT/'outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/behavior_events_v2/behavior_event_metrics_v2.csv'),model_sha256=sha(ROOT/'outputs/rbr_v2_prod_v1/run/epoch30.pt'),
       unchanged_download_budget_sha256=sha(ROOT/'outputs/rbr_v2_d2_v1/download_budget.json'),new_downloads_bytes=0,new_training_steps=0,Primary_V_C_runs=0,user_Quick_prefix_unchanged='QUICK_REFERENCE.md' not in bad)
 write(OUT/'protected_after.json',result)
 if bad or bad2 or badnative:raise ValueError('Protected asset mutation detected')
 return result


def report():
 v=read(OUT/'SCIENCE_analysis_view.json');eng=read(OUT/'ENGINEERING_ledger.json');c=read(DOC/'SCIENCE_Protocol_v1.json');stat=read(OUT/'D_statistics_results.json') if (OUT/'D_statistics_results.json').exists() else None
 export=read(OUT/'interface/representation_diagnostics.json') if (OUT/'interface/representation_diagnostics.json').exists() else None
 rows=[]
 for s in c['arms']:
  root=SCIENCE/'SCIENCE'/s['run_id'];complete=(root/'completion.json').exists();a=read(root/'arm_science_review.json') if complete else {};ph=a.get('mechanism',{}).get('brake_phases',[])
  rows.append(dict(run_id=s['run_id'],session_id=s['session_id'],arm=s['arm'],actual_complete=complete,scientific_pass=a.get('arm_pass','NOT_ESTABLISHED'),mechanism_pass=a.get('mechanism',{}).get('proposal_diagnostic_pass','NOT_ESTABLISHED'),official_safety_pass=a.get('official_safety_pass','NOT_ESTABLISHED'),
       braking_phases=len(ph),first_peak_mps2=ph[0]['peak_decel_mps2'] if ph else '',first_dose_mps=ph[0]['dose_mps'] if ph else '',second_peak_mps2=ph[1]['peak_decel_mps2'] if len(ph)>1 else '',second_dose_mps=ph[1]['dose_mps'] if len(ph)>1 else '',
       actual_frames=80 if complete else '',actual_transitions=79 if complete else '',original_D2_exposure_disclosed=True,recovered_monitor_event=s['run_id']=='D3-D-13-TREATMENT'))
 csvwrite(DOC/'D3_All80_Actual_Arm_Results_v1.csv',rows)
 pairrows=[]
 for p in v['pairs']:
  fm=p.get('F_match',{});delta=fm.get('absolute_delta',{})
  pairrows.append(dict(pair_id=p['pair_id'],session_id=p['session_id'],pair_pass=p['pair_pass'],F_match_pass=fm.get('pass','NOT_EVALUATED'),balance_pass=p.get('retained_P1_balance_review',{}).get('pass','NOT_EVALUATED'),temporal_pass=p['temporal']['pass_gate'],max_actual_speed_difference_mps=p['temporal']['max_abs_speed_difference_mps'],
     delta_mean_speed_mps=delta.get('mean_speed',''),delta_end_minus_start_speed_mps=delta.get('end_minus_start_speed',''),delta_path_length_m=delta.get('path_length',''),delta_mean_abs_accel_mps2=delta.get('mean_abs_accel','')))
 csvwrite(DOC/'D3_All40_Actual_Pair_Results_v1.csv',pairrows)
 proto=read(ROOT/'configs/rbr_v2_d3_v1/interface_contract.json');exposure=[]
 for r in proto['runs']:
  if r['arm']=='BASELINE':exposure.append(dict(session_id=r['session_id'],detector_role=r['new_role'],current_exposure='E3_DETECTOR_DEVELOPMENT',same_claim_future_V_C_excluded=True,independent_confirmation_claim=False))
 csvwrite(DOC/'D3_Detector_Exposure_Update_v1.csv',exposure)
 if v['total_actual_complete_arms']>=2:plots()
 protected=protection();legacy=read(OUT/'ENGINEERING_gate.json');capacity=read(ROOT/'docs/stageR/rbr_v2_d2_v1/Capacity_And_Resource_Summary_v1.json')
 aa=stat['outcomes']['AA_CONTROLLED_SWAP']['alarm_probability'] if stat else {};ab=stat['outcomes']['AB_PAIRED_PI50']['alarm_probability'] if stat else {};delta=stat['outcomes']['AB_PAIRED_PI50']['delta_BDD'] if stat else None
 audit=read(OUT/'paired_calibration_transfer_audit.json') if (OUT/'paired_calibration_transfer_audit.json').exists() else None
 sat=bool(stat and stat['saturation']['H30']['AB_alarm_at_least_099']);both=bool(sat and stat['saturation']['RBR64']['AB_alarm_at_least_099'])
 interpretation=('两者在这个冻结D有限池中都接近饱和；它支持二者都能捕捉该受控时序变化，不支持RBR在H30之外还有实用增量。' if both else 'H30在该D有限池中接近饱和，RBR相对它的最大报警率增量空间很小。' if sat else '当前D有限池结果不能描述为两者均已饱和；应按实际报警率和校准限制判断，不预设RBR优势。') if stat else '没有足够完整独立来源执行原m20诊断；仅保留真实描述性输出。'
 verdict='RBR_V2_D3_REAL_D_EVIDENCE_COMPLETE_STATISTICAL_FEASIBILITY_BLOCKED' if stat else 'RBR_V2_D3_DATA_OR_STATISTICAL_SUPPORT_BLOCKED'
 arm_cpu=v['observed_measured_CPU_seconds']/max(1,v['total_first_attempts']);medium_def=max(0,240-capacity['total_mutually_disjoint_potential_final_cal_V_sources']);yield2=capacity['conditional_cost']['observed_second_frame_incremental_yield'];bytes2=capacity['conditional_cost']['second_frame_mean_compressed_bytes_per_LOG'];extra_logs=math.ceil(medium_def/yield2)
 text=f'''# RBR-V2-D3：路线兼容修复与第一次真实D阶段比较

**{verdict}**

本轮解决了新增nuPlan来源上的已证实路线弧长绑定缺陷，完成12个工程arm及{v['total_actual_complete_arms']}个真实D科学arm、{v['total_actual_complete_pairs']}个完整D配对，其中{v['science_pairs_passed']}对通过冻结资格。主要剩余阻断是RBR在held-out pair-swap AA上的报警率为100%，当前标定无法跨D来源迁移，不能在有效误报控制条件下比较检测增益。{interpretation} Primary、V/C、训练和新增下载均未执行。

## 1. 实际根因

原D2失效场景仍在lane51400多边形内，但其rear axle已超过这条车道中心线端点。单车道API投影被截断为42.918042633m，而实际被采样的原生拼接线对应位置为44.346439927m，坐标绑定少了1.428397294m。参考第0点强制设为真实自车后，第1点反而退后约0.339m，其heading接近π。官方LQR从参考拟合到约3.2836m⁻¹的异常初段曲率，因此输出8.321856872rad/s。零运行重放精确复现该请求；初始横向误差、heading误差和转向角均为0，故不是凭一个大数猜测控制器错误。

官方API按单段中心线投影并截断的行为正确；错误在本仓库把它的单段弧长误用为拼接参考线上的当前位置。

官方LQR输出与运动模型输入都是转向角速度，模型按dt积分并滤波；没有发现角度/角速度接口混用。旧D2的0.7秒实际偏转和路线关联丢失保持原事实，未产生官方安全metric，不能追溯称为官方碰撞或越界。

## 2. 修复方法与科学身份

技术版本`SDC_D3_ENDPOINT_ARC_BINDING_COMPAT_V1`只在当前车道端点截断导致弧长坐标不一致时，把坐标绑定到**同一条原生拼接参考线**。不更换冻结道路/车道连接，不更换LQR、增益、车辆模型、纵向命令、时钟、测量或安全阈值。原坐标正确的情形返回原浮点值；无参数网格搜索。

修复后首请求约−0.000587565rad/s，参考首步恢复前进。独立直线/旋转平移物理oracle通过。历史D20的3200次参考构造中3196次逐位一致；4次存在较小同类端点截断，另外选定对应两对历史D与一个旧D_fit配对做完整工程重放。6个工程配对全部通过原机制、匹配、平衡、安全，所有对应旧/新纵向请求、模型加速度、80帧速度逐位一致，实测坐标差最大{max(x['max_abs_XY_coordinate_difference_m'] for x in legacy['targeted_physical_replays']):.9f}m。不同技术版本的已知差异已披露；不是声称旧/新所有轨迹逐位相等。

据此归类为A工程坐标兼容修复，而非新的制动科学干预。`ENGINEERING_gate.json`在正式D执行前冻结兼容依据；旧D20只按明确D角色用于开发性paired-swap校准。**旧SDC-v1的20/20不被自动当作D3新资格证明**，本轮新D资格来自实际新执行与完整分母。该兼容证书不授权Primary，也不消除源域统计不确定性。

## 3. 真实执行、资格与故障账本

工程：12/12 arm、6/6配对通过，CPU实测{eng['cpu_seconds']+eng['driver_cpu_seconds']:.3f}s。科学：原D40原顺序，{v['total_first_attempts']}/80个首次attempt，{v['total_actual_complete_arms']}/80个完整arm，{v['science_pairs_passed']}/{v['total_actual_complete_pairs']}个完整配对通过，科学重试0，失败SESSION替换0。

第13对Treatment主进程返回0且已经生成完整80帧/79控制、官方指标、serializer、completion和全部匹配SHA；旧监护器因退出时辅助进程而报CPU不确定。暂停账本原样保留。只读核验既有产物、确认原进程组已消失，并把从故障到核验的整个时间段按硬件CPU全占用上界计费；未重跑该arm。只继续原名单剩余54个首次身份，另立链接账本，不复活旧账本、不重置预算。新监护器有界清理辅助进程，每arm按完整墙钟×核心数保守计入预算，实际CPU另外记录。

`SCIENCE_analysis_view.json`只是原始账本、后续账本和已有产物恢复证明的只读聚合，**不是新执行账本**。原始TECHNICAL_FAIL监护记录仍可见，不篡改为从未发生。两次D2失败和stopped账本完全不变。

逐arm及逐pair CSV保留所有80个事前身份、实际状态、机制峰值/剂量、官方安全、四项F_match、额外平衡和时序分离结果。B/T同源、同初态、80帧10Hz、79次官方模型传播；中间18次请求严格0。未把近零实际加速度当作真实零踏板/零轮端扭矩，滤波尾巴保留。原same-index raw观察映射及其离线时间语义限制不变，不宣称在线部署能力。

图`D3_Recorded_Root_And_Braking_Curves_v1.png`固定展示D01：左上是零运行参考构造，右上是旧D2与新D3实际初段heading，下面是实际模型加速度/最终请求及真实速度；理想参考未替代真实轨迹。曲线CSV末帧没有控制推进，空请求不伪填为已执行0。

## 4. RBR64与完整H30：真实同条件输入

在读取检测结果前已冻结全部D身份、物理版本和统计代码。仅使用正式epoch30，SHA`5b0e5738c0b67d32865545764bc4bcd12152a507f62bcb0f323233a3bb8deed3`；Waymo normalizer、Canonical83、H30全部30维和D1既定scaler/带宽规则不变。`U_SOURCE_COMPLETE_INDEPENDENCE_NOT_ESTABLISHED`继续披露。

从实际官方simulation log读取相同车辆、SESSION、80帧EgoState及同index观测，逐点对照真实trace/时钟，真实生成Canonical83和mask；H30从相同80帧轨迹计算。没有用语义真值代替z64，没有从不完整序列补帧，没有按表现换样。实际导出{export['RBR64_shape'] if export else 'NOT_ESTABLISHED'}与{export['H30_shape'] if export else 'NOT_ESTABLISHED'}；有限性和重复eval一致性、每维方差、有效秩、mask及每对差异存于`representation_diagnostics.json`。原始L2跨表示单位不同，不能用其大小排名检测效用。

z64的64列均有非零方差，中心化数值秩64、熵有效秩约10.33，重复eval逐位一致；当前没有常量塌缩证据，但这不证明有效检测或额外信息独立性。

## 5. 真正D-only统计结果

{'已完成原m20、π=.50、α=.05的paired controlled-swap开发性诊断。三组各20个独立SESSION：旧D20校准、新D01–20独立swap-AA、新D21–40配对AB，彼此及D_fit3互斥。' if stat else '原m20的三组独立来源支持不完整，未执行检测率比较。'}

| 指标 | RBR64 | H30 |
|---|---:|---:|
| 独立D pair-swap AA报警比例 | {aa.get('RBR64','NOT_ESTABLISHED')} | {aa.get('H30','NOT_ESTABLISHED')} |
| D paired-AB报警比例 | {ab.get('RBR64','NOT_ESTABLISHED')} | {ab.get('H30','NOT_ESTABLISHED')} |

ΔBDD条件有限池点估计：**{delta if delta is not None else 'NOT_ESTABLISHED'}**。{'条件Monte Carlo SE='+str(stat['outcomes']['AB_PAIRED_PI50']['conditional_MC_SE'])+'；实际AB draws='+str(stat['outcomes']['AB_PAIRED_PI50']['draws'])+'。' if stat else ''} 无偏MMD²形式保留负值；两边使用相同源身份、mask、数学带宽规则及抽样预算，1999校准取第1900顺序统计量，严格大于阈值才报警，tie不报警。全部原始统计量、alarm数组、阈值和kernel数值诊断保留。

**最重要的统计失败：RBR held-out swap-AA报警为1.0000，而AB为0。当前RBR管线没有在独立D池上实现可信的名义误报控制。** 因此−0.2462只能报告为冻结管线的D有限池原始报警差，不能把它解释成已通过同误报 operating point资格后的H30/RBR优劣结论。H30的0.0212也不自动构成总体FPR资格PASS。

数值核查没有发现重排或缓存公式错误：将6组各前20个实际抽样逐个用原始标准化向量和原MMD函数重算，与缓存矩阵分数最大差为{audit['direct_vs_cached_max_abs_error'] if audit else 'NOT_ESTABLISHED'}。没有重新拟合scaler、换带宽、截断负值或在AA/AB上重标定。

另外对事前选定的6个工程兼容重放进行冻结encoder一致性探针，不新增SESSION或仿真，不替换校准向量：两对受影响旧cal数据在旧/修复版本之间，RBR kernel相似度均≥0.999999996，H30均≥0.9999829；旧D_fit案例也仅作诊断且不重拟合。该结果不支持把此次大幅校准迁移异常归因于那4条旧cal轨迹的微小修复差异；仍不声称唯一总体原因。探针不进入原60源/120arm统计分母。

对同一组20个baseline向量，冻结的两样本无偏形式满足`MMD²(X,X)=2/m×(平均非对角kernel−1)`，通常为负；这里X/X共享来源，只用于代数诊断，不能冒充合法独立A/A。在RBR的cal/AA/AB池，baseline非对角kernel均值分别约0.412966/0.561279/0.339223，对应共享来源基线分数−0.058703/−0.043872/−0.066078。原阈值为−0.057149，而实际AA分数全部更大、AB全部更小。**跨来源背景kernel几何与小校准池造成的迁移风险已有直接数值证据**；不能把这个诊断当作已证明唯一总体原因，也不能据此自行修改SAP的paired统计量或null合同。`D3_Frozen_Calibration_Transfer_v1.png`展示完整冻结分数分布。

在冻结缩放/RBF下，同SESSION B/T的RBR kernel平均相似度约0.98，而H30为数值0，说明两者对本干预的成对响应不同；这是各自核空间内的D描述，不能把它等同于release报警率或跨空间比较原始MMD强弱。H30的AB报警率只有0.2462，**没有H30已达到检测饱和、挤掉全部增量空间的证据**。

**这些AA比例是受控pair-swap pseudo-null，不是operational baseline release FPR。** 20个独立源被反复释放不等于10000个独立科学样本；条件MC误差不等于SESSION层面区间。每组仅20源与m20相等，cluster bootstrap会丢失唯一源支持；源域95%CI、独立总体FPR资格、功效及区间覆盖均`NOT_ESTABLISHED`。没有用条件点估计宣布Primary成功或失败。

{interpretation} {'相对H30在本D有限池/运行点的报警率最大可能增量为'+str(1-ab['H30'])+'；这只是代数上界，不是总体效应的置信上界。增加日志可提高精度和外推检验能力，却不能保证创造原本不存在的增量效应。不得因此重训RBR、删H30特征或修改SDC以制造优势。' if stat else ''}

## 6. 容量与后续成本：仍不能Primary Ready

D2的1253新增真实SESSION、162静态适格保持。完整D40仍全部E3，不能因通过而回流独立V/C；旧D20校准、D_fit3与未来资源继续隔离。当前候选final-cal/V为151个V静态候选＋10个**最终校准适用性尚待认证**的旧E3候选＝乐观161。最低配对160只是计数；若这10个不能满足最终校准独立性，剩151、最低配对仍缺9。原非配对320缺159。C原5个及新20个目录保留身份未访问其科学结果/原始轨迹，不宣称C科学容量已认证。

本轮实际D配对通过率{v['science_pairs_passed']}/{v['total_actual_complete_pairs']}仅描述已冻结D来源，不能证明全部静态候选或第二LOG来源、C均同样通过；工程观察过的来源不能冒充未经观察的确认样本。原最小40/40/80配对包络的bootstrap仍有6/2000次唯一支持不足；不是改小m的理由。

如果仅为得到更稳健的source-bootstrap支持，60/60/120规划包络需240源，相对乐观161至少再增{medium_def}个适格SESSION；这仍不是功效证书。按D2已观察到的第二框净增率{yield2:.2%}、平均压缩LOG {bytes2/1e6:.2f}MB粗算，约需检查{extra_logs}个额外完整LOG、约{extra_logs*bytes2/1e9:.2f}GB下载；非配对缺159对应约928个LOG、42.93GB。均为有条件成本场景，固定框非IID，未来增率和资格率无保证；只能在新的授权预算下进行。本轮新增下载0，未超过原50GB账本。

按本轮实测每科学arm约{arm_cpu:.2f} CPU秒作条件外推，最低配对160源两臂约320arm需{320*arm_cpu/3600:.2f} CPU小时，非配对320源两臂640arm约{640*arm_cpu/3600:.2f} CPU小时；不含来源下载、资格失败、D_fit、C、预检、表示和统计开销，不是运行保证或授权。{'若优先问题是RBR对已饱和H30的实用增量，则不建议仅为追求该增量而直接追加大量下载；应先接受本D证据的科学边界。' if sat else '是否追加下载应由真实增量空间、校准风险及验证价值共同判断，不能仅为了跨过160个计数门槛。'}

本轮不建议仅因161≥160便追加paired确认实验。应先用已冻结D证据关闭“跨来源校准是否成立”这一统计问题；当前不是普通工程故障，改变paired统计量/null/阈值合同属于受保护科学协议修约，本轮未实施。若坚持原非配对开发路径，单次release至少需要40个不同SESSION；三个互斥的cal/AA/AB开发池按最低每池40计共120个D来源，相对本轮60个真实完整D来源至少还缺60个，且这仍不足以认证source-bootstrap覆盖和功效。这些新增D不能从已保护final-cal/V/C临时转入；上文79/159缺口也未包含它们和C。增加日志可能改善来源代表性、精度和标定稳定性，但不能保证RBR产生正增量。

## 7. 实际资源、复现及保护

科学实测可观测CPU {v['observed_measured_CPU_seconds']:.3f}s；含退出辅助进程不确定期的保守科学CPU预算计费{v['conservative_scientific_CPU_budget_charge_seconds']:.3f}s＜14400s，单arm均受1200s墙钟限制。工程CPU {eng['cpu_seconds']+eng['driver_cpu_seconds']:.3f}s；科学driver执行墙钟累计{v['science_driver_wall_seconds']:.3f}s，不包括停顿调查时间。进程实际CPU与保守上界分别公开，没有把未知部分伪报成精确实耗。额外非科学fixture不调用车辆模型。

真实120arm输入提取墙钟124.95s（CPU user122.51s＋sys3.60s），冻结模型MPS导出墙钟1.60s，原D统计墙钟0.88s；进程资源原文保存。推理环境只用本机已有requests等五个纯Python包建立独立覆盖层，未联网安装、未改nuPlan或旧推理环境，overlay版本与每文件SHA另存。模型、normalizer和所有实际输入均绑定输出，不以loss或检测效果选模型。

冻结前14项弧长/统计测试及独立辅助进程fixture通过；真实工程12arm和D完整记录才是闭环依据，单元测试不代替科学结果。所有源码编译、临时依赖检查、模型/数据SHA、各阶段提交/协议和原始预算账本在Manifest绑定。正式D代码在首个模型step前提交推送，基础设施续接另有事前绑定，只改进程监督与只读账本聚合。

原{protected['initial_tracked_assets']}个已跟踪资产及D2的{protected['D2_bound_files']}个绑定文件、官方F2的{protected['official_F2_native_source_configs']}个源码/配置保持SHA；用户原Quick前缀和已有未提交文件保留。受保护CSV SHA`{protected['protected_CSV_sha256']}`。B1历史PASS、C2/P2 FAIL、旧SDC-v1 20/20、D2两次失败及旧停止账本全部不变。

本轮完成后停止；没有RBR训练/新seed、H30或kernel搜索、Primary、V/C执行或额外下载。取得的是首次真实、边界明确的D-only检测证据，不是论文主实验优势证明。
'''
 (DOC/'Final_Report_CN_v1.md').write_text(text,encoding='utf-8');write(DOC/'Status_v1.json',dict(status=verdict,engineering_pairs_passed=6,new_scientific_complete_pairs=v['total_actual_complete_pairs'],new_scientific_pairs_passed=v['science_pairs_passed'],real_D_statistics=bool(stat),Primary_ready=False,
    calibration_transfer_status='RBR_HELD_OUT_SWAP_AA_DIAGNOSTIC_FAILURE',held_out_swap_AA_alarm=aa,paired_AB_alarm=ab,conditional_raw_delta_BDD=delta,valid_FPR_controlled_superiority='NOT_ESTABLISHED',H30_saturation_observed=sat,
    source_CI_power_coverage='NOT_ESTABLISHED',new_downloads=0,new_training=0,V_C=0))
 print(verdict,flush=True)


def manifest():
 paths=list(OUT.rglob('*'))+list(DOC.rglob('*'))+list(SCIENCE.rglob('*'))+list((ROOT/'configs/rbr_v2_d3_v1').rglob('*'))+list((ROOT/'tools').glob('rbr_v2_d3_*.py'))+list((ROOT/'tests').glob('test_rbr_v2_d3_*.py'))+[ROOT/'tests/fixtures/d3_supervision_fixture.py']
 index=OUT/'Provenance_File_Index_v1.json';target=DOC/'Manifest_v1.json';files={}
 for p in sorted(set(paths)):
  if not p.is_file() or p in [index,target] or '__pycache__' in p.parts or p.suffix in ['.lock','.writing','.pyc'] or p.name.startswith('manifest_packaging'):continue
  files[str(p.relative_to(ROOT))]=dict(sha256=sha(p),bytes=p.stat().st_size)
 write(index,dict(files=files,policy='Real raw model histories/metrics retained locally and bound; no retrial or synthesized scientific samples; physical/control/statistical version histories remain in Git'))
 write(target,dict(status=read(DOC/'Status_v1.json')['status'],provenance_index=str(index.relative_to(ROOT)),provenance_index_sha256=sha(index),bound_files=len(files),
    checkpoint_sha256=sha(ROOT/'outputs/rbr_v2_prod_v1/run/epoch30.pt'),normalizer_sha256=sha(ROOT/'outputs/rbr_v2_prod_v1/normalizer.json'),
    D2_manifest_sha256=sha(ROOT/'docs/stageR/rbr_v2_d2_v1/Manifest_v1.json'),engineering_gate_sha256=sha(OUT/'ENGINEERING_gate.json'),science_analysis_view_sha256=sha(OUT/'SCIENCE_analysis_view.json'),
    original_D3_stopped_ledger_sha256=sha(OUT/'SCIENCE_ledger.json'),continuation_ledger_sha256=sha(OUT/'SCIENCE_continuation_ledger.json'),source_independence_risk='U_SOURCE_COMPLETE_INDEPENDENCE_NOT_ESTABLISHED',
    stage_commits=subprocess.check_output(['git','log','-4','--format=%H'],cwd=ROOT).decode().splitlines(),Primary_V_C=0,training=0,downloads=0))
 print('D3 Manifest bound files:',len(files),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['report','manifest','protected']);a=p.parse_args();{'report':report,'manifest':manifest,'protected':protection}[a.mode]()

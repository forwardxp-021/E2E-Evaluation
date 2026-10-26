#!/usr/bin/env python3
"""Close the bounded D2 source census and retained failed D attempts; never launches science or inference."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse,csv,datetime,hashlib,json,shutil,subprocess,time
from collections import Counter,defaultdict
from pathlib import Path
from tools.rbr_v2_d2_download import ROOT,OUT,LOCAL,read,write,sha
from tools.rbr_v2_d2_round2 import combined

DOC=ROOT/'docs/stageR/rbr_v2_d2_v1'


def csvfile(path,rows,columns=None):
 path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=columns or list(rows[0]));w.writeheader();w.writerows(rows)


def protected():
 before=read(OUT/'protected_before.json');changed=[];missing=[]
 own=(OUT/'quick_section.md').read_text();quick=(ROOT/'QUICK_REFERENCE.md').read_text()
 if not quick.endswith(own):raise ValueError('Own Quick section not an exact suffix; cannot verify user prefix')
 for name,h in before.items():
  path=ROOT/name
  if not path.exists():missing.append(name)
  elif name=='QUICK_REFERENCE.md':
   if hashlib.sha256(quick[:-len(own)].encode()).hexdigest()!=h:changed.append(name)
  elif sha(path)!=h:changed.append(name)
 result=dict(initial_tracked_assets=len(before),changed_from_task_start=changed,missing_from_task_start=missing,
             user_Quick_prefix_unchanged='QUICK_REFERENCE.md' not in changed,
             protected_CSV_sha256=sha(ROOT/'outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/behavior_events_v2/behavior_event_metrics_v2.csv'),
             expected_protected_CSV_sha256='e8deb93312e82183b6c2c0db30fd18cbf9c32d32d566038419a5be65b389d9d8')
 write(OUT/'protected_after.json',result)
 if changed or missing or result['protected_CSV_sha256']!=result['expected_protected_CSV_sha256']:raise ValueError('Historical/user asset mutation detected; inspect protected_after.json')
 return result


def monitor():
 """Measure still-live source processes. Prior terminated setup CPU is explicitly unmeasured."""
 import psutil
 observed={};started=time.monotonic()
 while True:
  for p in psutil.process_iter(['pid','cmdline','create_time','cpu_times','memory_info']):
   try:
    argv=p.info['cmdline'] or []
    if not any(Path(x).name in ['rbr_v2_d2_fast_acquire.py','rbr_v2_d2_round2.py'] for x in argv):continue
    key=str((p.pid,p.info['create_time']));c=p.info['cpu_times'];m=p.info['memory_info'];old=observed.get(key,{})
    observed[key]=dict(pid=p.pid,command=argv,cumulative_CPU_seconds=max(old.get('cumulative_CPU_seconds',0),c.user+c.system),observed_peak_RSS_bytes=max(old.get('observed_peak_RSS_bytes',0),m.rss))
   except (psutil.NoSuchProcess,psutil.AccessDenied):pass
  write(OUT/'source_process_resource_measurement.json',dict(processes=observed,measured_source_CPU_seconds=sum(v['cumulative_CPU_seconds'] for v in observed.values()),
        coverage='Still-live first-source segmented process and second-source process; earlier terminated catalog/setup/single-transfer processes NOT fully metered. This is NOT total task CPU.',monitor_wall_seconds=time.monotonic()-started))
  if (OUT/'acquisition_status.json').exists() and (OUT/'round2_status.json').exists():break
  time.sleep(5)
 print('Source dependency processes reached terminal statuses; monitor closed',flush=True)


def main():
 if not (OUT/'acquisition_status.json').exists() or not (OUT/'round2_status.json').exists():raise RuntimeError('Source scan still active; no final capacity claim')
 if (DOC/'Final_Report_CN_v1.md').exists():raise FileExistsError('Preserve already closed report; no silent revision')
 from tools.rbr_v2_d2_roles import allocate
 allocate();DOC.mkdir(exist_ok=True,parents=True)
 b=read(OUT/'D_execution_binding.json');cfg=read(ROOT/'configs/rbr_v2_d2_v1/interface_contract.json');plan=read(OUT/'acquisition_plan.json');plan2=read(OUT/'round2_source_plan.json')
 first=[read(p) for p in sorted((OUT/'static').glob('*.json'))];second=[read(p) for p in sorted((OUT/'round2/static').glob('*.json'))];allnew=combined()
 eligible={s:r for s,r in allnew.items() if r['status']=='ELIGIBLE'};prior={r['session_id']:r for r in first};second_increment=sum(r['status']=='ELIGIBLE' and prior.get(r['session_id'],{}).get('status')!='ELIGIBLE' for r in second)
 receipts=[read(p) for p in sorted((OUT/'downloads').glob('*.json'))];new_sessions={r['session_id'] for r in receipts};raw_by_name={r['filename']:r for r in receipts};source_rows=[];headers=defaultdict(set)
 for frame,records in [('FIRST_COST_MINIMUM',first),('SECOND_COMPLETENESS',second)]:
  for r in records:
   p=Path(r['log']['source_file']);receipt=raw_by_name[p.name];headers[r['session_id']].add(r['log']['database_log_token']);cache=OUT/'cache_transforms'/(p.name+'.json')
   source_rows.append(dict(session_id=r['session_id'],source_frame=frame,static_status=r['status'],static_reasons=';'.join(r['reasons']),official_split=r['source_split'],log_id=r['log']['log_id'],database_log_token=r['log']['database_log_token'],
       source_path=str(p),source_sha256=r['download_sha256'],source_bytes=receipt['uncompressed_bytes'],compressed_download_bytes=receipt['compressed_bytes'],archive=receipt['archive'],member=receipt['member'],official_ETag=receipt['etag'],
       ZIP_CRC_verified=receipt['crc_verified'],lossless_cache_path=read(cache)['archive_path'] if cache.exists() else '',lossless_cache_sha256=read(cache)['archive_sha256'] if cache.exists() else '',
       metadata_sha256=r['log']['sha256'],static_evidence=str((OUT/('static' if frame=='FIRST_COST_MINIMUM' else 'round2/static')/(r['session_id']+'.json')).relative_to(ROOT)),scanner_sha256=r['scanner_sha256'],new_scientific_runs_in_static=r['new_scientific_runs']))
 if new_sessions&set(plan['old_SESSION_excluded']) or new_sessions&set(plan['new_C_METADATA_ONLY']):raise ValueError('Source aliases old or protected C SESSION')
 tokens=defaultdict(set)
 for sid,ts in headers.items():
  for token in ts:tokens[token].add(sid)
 if any(len(s)>1 for s in tokens.values()):raise ValueError('Identical raw database log token under distinct SESSION')
 hashes=defaultdict(set)
 for r in receipts:hashes[r['sha256']].add(r['session_id'])
 if any(len(s)>1 for s in hashes.values()):raise ValueError('Exact raw DB duplicated across independent SESSION')
 csvfile(DOC/'Official_Source_And_Static_Audit_v1.csv',source_rows)
 ledgers=[read(OUT/p) for p in ['science_ledger.json','science_continuation_ledger.json']];attempts=[a for d in ledgers for a in d['attempts']]
 if not all(d['stopped'] for d in ledgers) or len({a['run_id'] for a in attempts})!=len(attempts):raise ValueError('Live ledger or duplicate scientific attempt')
 attempts_by_id={a['run_id']:a for a in attempts};armrows=[]
 for s in b['arms']:
  a=attempts_by_id.get(s['run_id']);armrows.append(dict(run_id=s['run_id'],session_id=s['session_id'],arm=s['arm'],status=a['status'] if a else 'NOT_RUN_AFTER_INFRASTRUCTURE_STOP',wall_seconds=a['wall_seconds'] if a else 0,cpu_seconds=a['cpu_seconds'] if a else 0,
      mechanism='NOT_ESTABLISHED_INCOMPLETE_RUN',F_match='NOT_EVALUATED_INCOMPLETE_PAIR',official_safety='NOT_PRODUCED_INCOMPLETE_RUN',technical_retry=0))
 csvfile(DOC/'D_All80_Arm_Denominator_v1.csv',armrows)
 Dnew={r['session_id'] for r in b['arms']};Dold={r['session_id'] for r in cfg['runs'] if r['new_role']=='D_SWAP_CAL_EXISTING_B_REUSE'};roles=read(OUT/'candidate_role_partition_final.json')
 exposures=[]
 for sid in sorted(new_sessions|Dold|set(roles['D_fit'])|set(plan['new_C_METADATA_ONLY'])|set(roles['original_C_protected'])):
  if sid in Dnew:role='D_DEVELOPMENT_BOUND';exposure='E3_ROLE_RESERVED_NO_DETECTOR_RESULTS_READ'
  elif sid in Dold:role='D_SWAP_CAL_EXISTING_B_REUSE_BOUND';exposure='HISTORICAL_E2_WITH_EXPLICIT_D_ASSIGNMENT_NO_NEW_DETECTOR_RESULTS_READ'
  elif sid in roles['D_fit']:role='D_FIT_EXISTING';exposure='E3_EXISTING_D1'
  elif sid in plan['new_C_METADATA_ONLY']:role='C_INDEX_RESERVED_ONLY';exposure='INDEX_ONLY_NO_RAW_ACCESS'
  elif sid in roles['original_C_protected']:role='C_ORIGINAL_PROTECTED';exposure='UNCHANGED_NO_RAW_ACCESS'
  else:role='METADATA_ONLY_STATIC_CANDIDATE_OR_EXCLUDED';exposure='E0_METADATA_ONLY'
  n=sum(a['run_id'] in {s['run_id'] for s in b['arms'] if s['session_id']==sid} for a in attempts)
  exposures.append(dict(session_id=sid,role=role,exposure=exposure,new_scientific_attempts=n,new_detector_outputs_read=False,future_same_claim_V_C_excluded=role.startswith('D_'),new_raw_source_access=sid in new_sessions))
 csvfile(DOC/'SESSION_Roles_And_Exposure_v1.csv',exposures)
 budget=read(OUT/'download_budget.json');end1=read(OUT/'acquisition_status.json');end2=read(OUT/'round2_status.json');resource=read(OUT/'source_process_resource_measurement.json') if (OUT/'source_process_resource_measurement.json').exists() else None
 if budget['reserved_bytes'] or budget['received_bytes']>budget['limit_bytes']:raise ValueError('Unresolved network reservation or download budget violation')
 oldfit=set(roles['D_fit']);nominal=len(roles['V_AA_candidates'])+len(roles['V_AB_candidates'])+len(roles['final_calibration_candidates'])+len(roles['unallocated_candidates'])
 # Count all available candidate sources, not merely the prefix allocated to one insufficient envelope.
 available=len(eligible)-len(Dnew)+29+10;deficit_unpaired=max(0,320-available);deficit_paired=max(0,160-available)
 scientific_cpu=sum(d['cpu_seconds']+d.get('driver_cpu_seconds',0) for d in ledgers);scientific_wall=sum(d.get('scientific_wall_seconds',d.get('wall_seconds',0)) for d in ledgers)
 invariant=protected()
 first_rate=sum(r['status']=='ELIGIBLE' for r in first)/len(first);second_rate=sum(r['status']=='ELIGIBLE' for r in second)/len(second) if second else 0;combined_rate=len(eligible)/len(allnew)
 new_rate=second_increment/len(second) if second else 0
 # Descriptive cost scenarios only: neither random-source yield nor future eligibility is guaranteed.
 source_mean_bytes=sum(r['compressed_bytes'] for r in receipts)/len(receipts)
 cost=dict(observed_first_frame_yield=first_rate,observed_second_frame_yield=second_rate,observed_second_frame_incremental_yield=new_rate,observed_combined_yield=combined_rate,
      conditional_additional_new_SESSION_for_paired_at_observed_combined_yield=math_ceil(deficit_paired/combined_rate),conditional_additional_new_SESSION_for_unpaired_at_observed_combined_yield=math_ceil(deficit_unpaired/combined_rate),
      mean_compressed_bytes_per_downloaded_LOG=source_mean_bytes,warning='Deterministic unequal-duration source frames; no IID rate or confidence interval claimed. Additional source/byte scenarios are not a guaranteed acquisition budget or approved extension.')
 write(DOC/'Capacity_And_Resource_Summary_v1.json',dict(new_raw_SESSION=len(new_sessions),new_raw_LOGs=len(receipts),official_splits=dict(Counter(r['official_split'] for r in receipts)),new_static_eligible_SESSION=len(eligible),first_frame=dict(processed=len(first),eligible=sum(r['status']=='ELIGIBLE' for r in first),stop=end1['stop_reason']),
       second_frame=dict(processed=len(second),eligible=sum(r['status']=='ELIGIBLE' for r in second),new_eligible_SESSION_increment=second_increment,stop=end2['stop_reason']),
       total_mutually_disjoint_potential_final_cal_V_sources=available,unpaired320_deficit=deficit_unpaired,paired160_deficit=deficit_paired,D_new_bound_SESSION=40,D_existing_swap_cal_bound_SESSION=20,D_fit_existing_SESSION=len(oldfit),
       C_original_protected=len(roles['original_C_protected']),C_new_index_only=len(plan['new_C_METADATA_ONLY']),download_bytes=budget['received_bytes'],download_hard_limit_bytes=budget['limit_bytes'],local_raw_DB_bytes=sum(r['uncompressed_bytes'] for r in receipts),
       scientific_first_attempts=len(attempts),technical_failed_attempts=sum(a['status']=='TECHNICAL_FAIL' for a in attempts),complete_pairs=0,technical_retries=0,scientific_CPU_seconds=scientific_cpu,scientific_wall_seconds=scientific_wall,
       source_process_resource_measurement=resource,conditional_cost=cost,source_coverage='SESSION/LOG header and exact raw hash auditable; no claim that distinct SESSION imply all environmental/route factors statistically independent',
       real_AA_FPR='NOT_ESTABLISHED',AB_detection_rates='NOT_ESTABLISHED',DeltaBDD='NOT_ESTABLISHED',H30_saturation='NOT_ESTABLISHED',power='NOT_ESTABLISHED',CI_coverage='NOT_ESTABLISHED'))
 route=read(OUT/'route_diagnostic/diagnostic.json');r0=route['rows'][0];r7=route['rows'][-1]
 text=f'''# RBR-V2-D2：官方nuPlan来源扩展与BDD可行性

**BDD_D2_CAPACITY_OR_SCIENTIFIC_FEASIBILITY_BLOCKED**

本轮完成了真实官方数据的增量获取、原SDC-v1静态筛查、SESSION去重、角色容量审计及一次失败的D执行入口验证；没有完成可用的D检测实验。不能报告RBR优于H30，也不能判断H30是否饱和。`U_SOURCE_COMPLETE_INDEPENDENCE_NOT_ESTABLISHED`保持。

## 直接答案

1. 新增真实官方train/val SQLite **{len(receipts)}个LOG chunk、{len(new_sessions)}个不同SESSION**，这些SESSION不在原本地253集合。不是将chunk数作为独立样本数。另有20个新C只保留官方目录身份，未下载原始数据，未计入已核验容量；原C五个保持保护。
2. 原物理规则下新增**{len(eligible)}个静态适格SESSION**。首来源框{len(first)}个SESSION中{sum(r['status']=='ELIGIBLE' for r in first)}个适格（{first_rate:.2%}）；第二来源框检查{len(second)}个同SESSION的另一个完整性较好的LOG，其中{sum(r['status']=='ELIGIBLE' for r in second)}个适格、净新增{second_increment}个。合并按SESSION仅计一次。**静态适格不等于闭环科学资格。**
3. 扣除预先固定的新D40、旧SDC20显式D校准分配，结合原剩余29个V静态候选与10个潜在D校准来源，最多**{available}个互斥的final-cal/V静态候选**。非配对最低320尚缺{deficit_unpaired}；配对最低160尚缺{deficit_paired}。这些是乐观规划数，10个旧E3的最终校准适用性仍待合同绑定，候选也尚未完成新闭环资格，不能等同于已具备的科学样本。
4. 冻结40个D SESSION、80个arm，零运行预检80/80；真实启动**{len(attempts)}个首次arm，技术失败{sum(a['status']=='TECHNICAL_FAIL' for a in attempts)}个，完整对0，重试0，余78个未启动**。没有失败替换、补样或成功子集。
5. 真实A/A误报、两表示检测率、ΔBDD、H30饱和、源域功效与区间覆盖全部**NOT_ESTABLISHED**。没有运行统计或模型推理来填补缺失科学数据。

## 官方来源、许可与增量方法

[AWS官方开放数据注册页](https://registry.opendata.aws/motional-nuplan/)公开列出匿名S3；仅从`motional-nuplan/public/nuplan-v1.1/`的10个train/val ZIP读取目录和SQLite member，未读取test ZIP或传感器blob内容。保存官方LICENSE原文、ETag、member偏移、CRC及本地完整原始SHA256；SHA为下载后计算并绑定的值，不冒充官方公布的签名SHA。许可包含CC BY-NC-SA4.0及Motional附加条款，限本非商业博士研究使用。[nuPlan项目](https://www.nuplan.org/)为数据说明来源。

原本地253个SESSION整体排除。新官方目录给出1273个候选前缀；按固定身份hash预留20个新C，剩1253个进入事前来源框。第一框按最小数据库成本、文件名时长≥15秒固定一个LOG，原S2R代表筛选后最多16个SDC anchor。短chunk常缺官方runnable场景支持，这是实际失败原因之一。第二框在任何D runner之前固定，对所有具备额外≥120秒LOG的1229个SESSION一致采用另一个最低成本LOG，包含首框成功者；不是只对失败者寻找成功替补。120秒是来源完整性裕量，未改变≥5m/s、车体走廊、路线曲率、初始位置、原始交通占用等科学谓词。两个来源框及全部排除记录均保留，合并采用固定round1→round2首个静态适格LOG。

来源选择是有条件的固定成本/完整性框，不是nuPlan全部时段的随机代表。旧37/138覆盖率既不能作为新数据适用率，也不能作为整个nuPlan的容量上限。此次实测合并适格率{combined_rate:.2%}，不外推为任意来源成功概率；离线未来交通检查仍只支持离线受控基准，不能宣称在线避障能力。

16MiB分段Range、并发4段、断点分片与ETag/Content-Range校验降低整块失败重传；ZIP CRC、长度、SQLite header、完整原始SHA与数据库LOG/SESSION header同时闭合。已认证缓存不重复下载。为保留磁盘余量，仅本轮新增且静态排除的DB进行zlib无损缓存，独立完整解压SHA验证后才移除自己的解压副本；原路径、mtime、raw SHA、压缩缓存SHA和恢复入口全部留存。未处理任何旧数据或C。原四幅官方地图直接复用，版本及SHA保持。

实际网络负载**{budget['received_bytes']/1e9:.6f} GB**，上限50.000000 GB；原始SQLite解压总量{sum(r['uncompressed_bytes'] for r in receipts)/1e9:.3f} GB。第一框终止原因`{end1['stop_reason']}`，第二框`{end2['stop_reason']}`；无自动扩容。逐LOG来源表与原始收据定位到本地可恢复缓存。确定性顺序/预算截断有选择限制，不把未扫描来源当作不适格。

## D执行真实失败及停止原因

全部身份、科学源码、配置、数据SHA、80个零运行预检先提交6015bf852431a90ec81cd3cbe125375510521a87并推送，之后才启动第一个arm。第一Baseline因**我编写的执行封装漏传`spec`与批准合同路径**，在生命周期函数进入前失败，没有runner或状态传播。这是工程遗漏，不是场景科学FAIL。已补充调用签名回归测试，原失败永久保留，没有重试。

修正封装只把原同一函数调用接上三个必需实参，并提交077d62478482631138806dd20dcd75da6340de51。原40SESSION/80身份、SDC控制器、路线、模型和门槛未改；链接原账本、携带已耗CPU，只允许原名单尚未启动的79个首次arm。第一个剩余身份Treatment实际执行7次控制、8个状态，在**{r7['episode_seconds']:.1f}秒、第{r7['iteration']}帧**抛出`NATIVE_ROUTE_FAIL: current native edge unavailable`。这发生在1.1秒第一脚制动之前，7次纵向请求均为0，不能由此宣称双制动或F_match科学FAIL。

只读地图重定位确认末帧原路线LANE/CONNECTOR成员数为{r7['current_point_route_lane_count']}；rear axle从({r0['x']:.3f},{r0['y']:.3f})到({r7['x']:.3f},{r7['y']:.3f})，heading从{r0['heading_rad']:.6f}到{r7['heading_rad']:.6f}rad。首个横向请求为{r0['executed_steering_rate_request_rad_s']:.6f}rad/s（转向角速度请求，不是纵向加速度）。这是实测横向偏转与路线关联丢失证据；尚不能唯一归因于LQR、参考连接几何、初始状态或地图。没有完整安全parquet，**不能伪称官方已判碰撞或驶出可行驶区**。诊断图与事件表只基于已生成状态和地图，不传播车辆模型。

零运行预检检验装配、时钟、回调与对象绑定，没有证明运行中的路线稳定；静态车体走廊也不是实际横向轨迹安全保证。第二项限制不能通过恢复同一失败身份解决；修复还可能改变冻结SDC路线/横向科学实现，因此继续科学执行已停止。两个账本均`stopped=true`，原结果不改。完整80个身份表中缺失明确保留，未运行78arm不被记作科学FAIL。

## 容量、角色与统计边界

新D40采用首次来源框完整hash前缀中的首40适格SESSION，在看到任何D运行结果前固定。原SDC20基准样本依S1R明确分配D-swap校准，当前管线不再将其计算为独立V；不是所有B自动变D。已有3个D1 baseline维持原D_fit scaler，两表示使用原同一数学带宽规则。E3/E4排除、E0–E5、U-only encoder和Primary C保护不变。新D40虽仅首SESSION启动，整个冻结D角色均保留隔离，不因失败重新回流V。

`candidate_role_partition_final.json`按事前固定的元数据hash规则给出互斥D_fit、D_new、D_cal、final-cal候选、V_AA、V_AB、C身份；这是候选容量，不是激活的Primary名单。20个新C仍只有目录身份，不宣称静态资格或闭环通过。不同scenario、同SESSION其它LOG、重复随机释放与bootstrap副本从不增加独立源数。

原SAP保留ΔBDD、α=.05、m20、π=.50、δ*=.10、SESSION聚类和非配对优先。源码支持负值保留的无偏MMD²、相同median正距离带宽、严格报警阈值、相同身份/预算及paired依赖；本轮仅运行确定性代码检查，不产生真实性能。拟议D controlled-swap null也不能冒充operational A/A FPR。源域功效/覆盖没有认证，不能凭320/160最低数或重复释放宣称功效达到要求。

纯抽样支持检查显示40/40/80配对最小包络的2000个cluster bootstrap中6次唯一源不足m20；60/60/120包络与非配对80/80/160的该项检查未出现不足。这只是唯一来源占用检查，不是新增样本、功效模拟、区间覆盖或修改SAP。所有规划包络与缺口保留在角色审计中，不因本轮D失败更改分组规则。

当前至少有两个独立阻断：**真实独立静态候选及其闭环资格/角色容量尚未认证**，以及**冻结SDC在新增来源存在运行时路线限制，D完整统计证据缺失**。不能判断方法本身一定无检测价值，也不能断言H30已饱和；目前首先缺合法可用的实际实验数据和统计认证。

## 成本、测试与保护

两次科学首次attempt含执行driver累计CPU **{scientific_cpu:.6f}秒**，科学driver wall累计{scientific_wall:.6f}秒，低于4小时CPU；单arm均未超20分钟。0次技术重试，0次新增训练/模型选择、0次真实MMD/BDD比较、0次V/C/Primary。源扫描过程CPU/RSS监测单独保存，早期已退出的目录/准备进程未完整计量，因此没有把它伪报为整个任务CPU精确总量。源扫描wall分段与各LOG下载、静态扫描耗时均可核查；下载含并行重叠，不能直接相加作wall。

通过的测试包括原D1统计8项、D2传输/身份/paired核缓存/无损缓存/分段SQLite/调用签名测试、既有7项物理oracle和真实适配器测试；精确命令与实际日志在Manifest绑定。所有tools编译及禁止临时依赖检查另行保存。它们证明已测工程性质，不能代替真实D科学PASS。

最终保护检查：初始{invariant['initial_tracked_assets']}个已跟踪资产相对任务开始无变化（仅允许自有Quick尾段）；用户原Quick前缀、Stage5修改及所有既有用户文件保留。受保护CSV SHA为`{invariant['protected_CSV_sha256']}`。正式epoch30 SHA仍为`5b0e5738c0b67d32865545764bc4bcd12152a507f62bcb0f323233a3bb8deed3`；normalizer、Canonical83、H30和原科学控制/SAP保持。旧B1 PASS、C2 FAIL、P2 FAIL、SDC-v1 20/20和所有历史执行账本均未重判。

## 最小后续工作

停止本轮科学执行；不重试原两次身份、不换样本、不自动开始新协议。需要先对新增源的路线/横向稳定性做独立的、事前定义的工程修复与验证，明确是否改变SDC科学实现，再决定新的D证据合同。数据方面按实际来源框和本轮排除率评估额外独立SESSION/完整LOG成本，不能只增加token。条件式成本估算见资源JSON：只是有前提的粗算，不是承诺能以同一适用率补足容量，更不是自动授权超过50GB。

本轮最终产物是可复核的官方新增来源、静态容量与清楚的D阻断；没有得到RBR/H30优势、饱和、功效或Primary可行性结论。
'''
 (DOC/'Final_Report_CN_v1.md').write_text(text,encoding='utf-8')
 write(DOC/'Status_v1.json',dict(status='BDD_D2_CAPACITY_OR_SCIENTIFIC_FEASIBILITY_BLOCKED',checkpoint_unchanged=True,source_independence_risk='U_SOURCE_COMPLETE_INDEPENDENCE_NOT_ESTABLISHED',D_complete_pairs=0,real_detector_statistics_executed=False,Primary_V_C_executed=False))
 print('Closed D2 report:',len(new_sessions),'new SESSION;',len(eligible),'static eligible;',len(attempts),'failed first arms;',available,'potential final-cal/V sources',flush=True)


def math_ceil(x):
 import math
 return int(math.ceil(x))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['monitor','finalize','protected']);a=p.parse_args()
 if a.mode=='monitor':monitor()
 elif a.mode=='protected':protected()
 else:main()

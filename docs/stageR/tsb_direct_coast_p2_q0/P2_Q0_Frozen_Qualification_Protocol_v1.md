# TSB-DirectCoast P2-Q0 — 冻结资格验证协议 v1

Scientific Owner接受TSB_DIRECT_COAST_F2_DEVELOPMENT_PASS，并在2026-10-25批准F2为独立资格验证唯一候选。本轮仅准备20 SESSION/40arm，科学执行NOT_AUTHORIZED。早期S1R草案标题中的PROPOSED由后续S1R_Owner_Approval_Record_v1正式批准覆盖；本协议依当前Owner指令与已批准S1R角色/暴露合同，不恢复旧绝对未暴露容量限制。

## 唯一科学候选：逐值保持F2

两臂均用原ScriptedLongitudinalRequestTrackerV1。Baseline[2100000,3900000)微秒u=−1.45 m/s²；Treatment[1100000,2100000)及[3900000,4900000)u=−1.305；其余时间u=0。固定命令积分两臂−2.61 m/s、连续区间重心3.0秒。无正补偿，无SESSION参数调整，不创建F3。

直接复用原F2 factory/planner/adapter/recorder/lifecycle/analyzer；官方KinematicBicycleModel(tau=.2)、横向LQR、native-route与10Hz规范整数时钟保持，80状态/79transition。源码及资格JSON以F2_Owner_Protocol_v1.json、F2_Execution_Source_Binding_v1.json的逐文件SHA为准；本轮不修改任何F2科学实现。P2只新建零运行身份装配入口，旧B1/C2/F2凭据不可授权P2。

机制算子：median3真实speed→物理时间gradient edge2；a≤−.6连续支持候选，支持≥.4s、峰≥.8m/s²、剂量≥.35m/s，domain[1.1,5.5]，Baseline恰好1段/Treatment恰好2段、分离≥.7s。Treatment[2.1,3.9)18个最终请求精确0，同packet到官方model、状态持续推进、filter关系成立。正速度，低于1m/s不得持续≥.5s，reference floor拒绝。近零a/漂移/正恢复保留诊断角色，不恢复为旧资格门槛。能力仍ABSTRACT_COAST_PROXY_ONLY，不宣称现实零踏板/轮端扭矩。

四F定义和差值caliper：raw speed80点均值.10m/s；raw末−首变化.15m/s；rear-axle XY欧氏步长和.80m；raw speed physical gradient(edge2)绝对值80点均值.02m/s²。六位舍入及epsilon1e−12不变。额外全窗负a剂量差≤.15m/s、每个T峰相对B峰差≤.20m/s²不变。两臂arm科学、安全及技术完整才进行正式pair判断；失败不以事后描述值替代资格。

官方安全取实际nuPlan产物的at-fault collision计数0及drivable_area=true；缺失属于基础设施不完整，不能填0。必须完成runner/recorder/metric/serializer/manifest/hash完整生命周期及实际观察审计。

## 名单与独立性

选择规则在查阅完整候选池前独立提交冻结。以全部248SESSION静态普查中已有代表为抽样框，元数据适格、初速≥5m/s、S1R允许且来源/身份/暴露可确认，固定SHA256(salt|SESSION)排序取前20；不新搜有利scenario。严格排除旧B1全部20SESSION、Coast DEV01/02/03（包含F2及C2科学DEV01），E3/E4/UNKNOWN排除；全部5个E5及其raw LOG来源保守保护给C，不占用它们。

名单先于40arm预检冻结，20个SESSION各1对。若容量不足，P2_Q0_CAPACITY_BLOCKED；如原名单中metadata/生产预检有阻断，保留20名及排名、失败原因，不能事后找替补。不用科学试跑、机制/F_match/安全或检测结果筛选。独立单位是SESSION；不同token不创造独立SESSION，不宣称独立driver/地点或所有交通关联已排除。

本次为B Benchmark资格验证，不是V/C或Claim-A未见SESSION泛化。零运行检查只追加metadata访问事件，保留既有E1/E2历史；未来启动科学资格后按实际暴露登记E2，不重命名成E5。原DEV01成功仅为已暴露B-development案例；本次不执行P2资格结果到D/V/C的自动转移。

## 一次性零运行预检

对固定20对的40arm逐一验证唯一SESSION/log/scenario、实际初态与两臂一致、canonical clock/raw same-index映射、初始有限native-route查询、F2 factory/tracker/官方model与recorder/callback/serializer/正式安全指标装配、fresh对象/输出namespace、数据与源码/配置/协议SHA。runner/model/control/metrics/finalizer入口统一guard禁止，history/recorder必须0。

检查失败后不得修F2科学代码并继续：仅保存基础设施缺陷，交Owner裁定；本轮原样报告BLOCKED。初始有限route查询通过不证明未来完整episode路由必定成功，科学机制、匹配和安全尚未观察。

保留同index观察限制：各SESSION原raw timestamp不插值/换帧，允许的index-cell偏差须逐SESSION记录并核查，不能一律套用DEV01偏差。DISCLOSED_OFFLINE_SAME_INDEX_REPLAY_NOT_STRICT_EVENT_TIME_CAUSALITY；安全阈值附近的标签可能敏感，不宣称严格在线因果或量产E2E。

## 待独立Owner终审的未来执行预算与停止规则

20SESSION/40首次arm。顺序按冻结pair排名，每对Baseline一次，通过技术/科学/实际观察后Treatment一次；fresh独立OS进程及全对象重置。每arm wall≤1200秒，累计CPU建议≤57600秒(16小时)。最大42累计attempt仅为待批天花板，绝不等于额外2次技术retry授权；当前所有技术重试NOT_AUTHORIZED。

首次真实科学失败立即停止，保留全部已启动attempt及结果，不调命令/阈值、不换SESSION、不继续筛成功率。技术故障立即停止，单独请求Owner分类及是否修复/重试裁定；无自动恢复。未运行成员继续保留原身份，不删除分母。

只有原20对全部完成，且每对机制/匹配/安全/完整性全部满足事前Owner批准条件，才允许P2_QUALIFICATION_PASS。任何失败/缺失/未运行不能用成功子集代替20/20；20/20不等于总体通过概率100%，不是Primary功效证明。

P2执行须在第一次runner前由Owner另行批准身份、完整来源/源码绑定、上述预算/成功和停止规则；本轮Authorization永远PROPOSED_ONLY，Q0工具没有科学运行路径。旧C2/F2停止账本不重置，本轮不创建可claim的P2预算账本。

## 防火墙

RBR训练/推理、H30检测、MMD/BDD Primary、V/C均0；不能用P2选择encoder架构、loss、seed、checkpoint或改变H、Primary统计参数、未来Primary样本/成功规则。U-only、SESSION推断、已批准E0–E5及E3/E4排除、Primary freeze边界不变。旧B1 20/20 PASS、C2 FAIL、F1报告与F2结果永久保护。

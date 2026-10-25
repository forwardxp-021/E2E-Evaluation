# TSB_DIRECT_COAST_F2 — 冻结执行规格 v1

Scientific Owner在2026-10-25明确不接受关闭全部Coast，并授权这个独立Development版本；此决定覆盖F1路线一建议，不改变F1归因、旧C2 FAIL或旧B1 PASS。Owner结构化指令摘要和准确数值在F2_Owner_Protocol_v1.json；该JSON是本次消息的结构化记录，不冒称逐字原文。

两臂共同使用ScriptedLongitudinalRequestTrackerV1。整数微秒half-open区间：Baseline[2100000,3900000)请求−1.45；Treatment[1100000,2100000)及[3900000,4900000)请求−1.305；其余所有时刻0。连续命令冲量各−2.61 m/s，区间中点加权时间重心3秒。数值冻结，不按SESSION调整、不补偿正请求。

不调用纵向LQR、停车控制器或base.track_trajectory。保留官方LQR的位姿/曲率读取及横向solver，按实际请求构造原公式的横向速度预测；不改官方源码、不写实际ego速度或加速度，不改KinematicBicycleModel。官方tau=.2、dt=.1；零请求后残余a自然衰减，能力仍ABSTRACT_COAST_PROXY_ONLY，不声称真实踏板/轮端扭矩滑行。

原native-route planner只作为路线几何/参考生成器复用；当前参考a、最终u、官方model实际a分别记录。初始日志/SESSION/token、precontext与route维持原DEV01；新run IDs只为F2-DEV-01-BASELINE/TREATMENT，角色为已暴露B-development，不是P2独立资格。不得执行DEV02/03。

机制继承D1/D2：median3真实速度→物理时间gradient edge2；a≤−.6形成支持候选，支持≥.4秒、峰≥.8 m/s²、剂量≥.35 m/s，domain[1.1,5.5]，Baseline恰好1段、Treatment恰好2段且分离≥.7秒。正速度、低于1 m/s不得持续≥.5秒、reference floor拒绝。Treatment[2.1,3.9)18请求精确0，同packet到model、持续传播及filter equation有效；原近零/漂移/正恢复只作诊断。安全来自正式at-fault collision计数0和drivable_area=true。

两臂技术/科学及实际观察审计完成后才调用正式pair analyzer。四F不变：raw速度80点均值差≤.10、末−首变化差≤.15、rear-axle XY步长和差≤.80、raw速度physical gradient edge2绝对值均值差≤.02；六位舍入，epsilon1e−12。保留已批准额外负a剂量差≤.15 m/s及每个T峰相对B峰差≤.20 m/s²，不偷删条件。任一arm未过则停止，不正式执行pair qualification，只允许描述性差值。

规范10Hz时钟、80状态/79transition、原raw观察same-index/token不变。DEV01偏差−330～+323us，DISCLOSED_OFFLINE_SAME_INDEX_REPLAY_NOT_STRICT_EVENT_TIME_CAUSALITY；raw观察可略晚于模拟时刻，安全边界敏感性限制保留，不宣称严格在线因果或量产E2E评价。

执行顺序：固定初始状态隔离模型验证一次→生产零运行预检→源码commit/逐文件hash与新授权冻结→独立OS进程Baseline一次→技术/机制/安全/实际观察审计PASS→fresh进程Treatment一次→符合条件才联合pair判断。任何接口/model预检、技术或科学失败停止，不调参/不重试/不新建F3。

新账本独立于旧C2，最大2首次attempt、累计scientific CPU≤3600秒、每arm wall≤1200秒，领取发生于进程启动之前。旧C2 stopped=true、累计2attempt/15.017957642CPU秒不重用不重置。隔离合成state的158次model更新属于本次明确允许的实现预检，不是SESSION科学attempt；单独记录，不伪称全轮model calls=0。

RBR训练/推理、H30/BDD/MMD比较、Primary/V/C、P2、DEV02/03均未授权。训练/Primary/H30/U-only规则、S1/S1R原合同、B1 20/20 PASS以及C2科学FAIL均只读保护。

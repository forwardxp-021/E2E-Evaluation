# P1R Scientific Owner决策备忘录 v1

**主状态：`P1R_BLOCKED_PHYSICAL_CLOCK_CONTRACT`；并行状态：`P1R_BLOCKED_OWNER_SCIENTIFIC_AMENDMENT`。** 本轮工程集成已形成可审查代码和证据；当前不能冻结启动首对。以下三个事项均有具体对象，不要求重批既有S1R、H30、C2参数或Primary统计合同。

## D1：新代理机制合同（待批准）

请批准或明确拒绝 `P1R_Scientific_Amendment_Proposal_v1.json` 中的新分层资格规格：零请求、同一命令到model、状态推进、官方滤波响应、保留双制动/间隔/低速/匹配/安全阈值。将旧近零core、固定漂移和正恢复阈值及依赖它们的稳定性结论改为完整保留的诊断，强制响应门槛改为可观察命令及模型方程异常。

依据：官方状态方程及真实模型隔离测试；不依赖任何真实C2、H或RBR结果。所有阈值的继承/技术数值预算及不可验证物理语义已写入机制提案。批准仅覆盖 `ABSTRACT_COAST_PROXY_ONLY`，不允许改称真实零踏板/零扭矩滑行。

## D2：保留F caliper，改变人工压力测试的准入角色（待批准）

请批准或明确拒绝：±.01m/s人工扰动继续保留为独立鲁棒性诊断，不再作为未经证实的生产误差上界；正式模型内判断继续使用原四项caliper（尤其mean_abs_accel=.02）、raw physical-time gradient、原舍入和比较规则。

依据：40条冻结trace/3200float无损序列化、不按.01量化、官方model/writer不注入速度测量噪声、模型内离散数值预算远小于.02。明确不包含真实传感器误差或完整runner可重复性证明。如Owner要求在±.01未知独立误差下仍鲁棒，当前合同依旧阻断；不能放宽到.10。

## D3：精确模拟时钟与日志观察时刻的对应（新增实际接口阻断，待决）

三个既定候选相对t0+i×100000微秒的最大偏差为330、1038、402微秒，均有跨2.1/3.9秒的控制步。原adapter拒绝跨界，新planner/recorder也要求精确时钟，原P1允许的1微秒cadence误差无法容纳。不能因偏差小就隐式忽略，更不能从54个候选里另找更容易通过的SESSION。

**唯一推荐**：前瞻批准一个仅用于新Coast栈的显式规范10Hz模拟时钟方案：t0保持官方初始EgoState整数时间；模拟状态时钟=t0+i×100000；日志观察仍按既定scenario索引读取，并同时保存原始日志时间、规范模拟时间及二者偏差，明确这种观察映射属于仿真近似。C2幅值/时长、80状态/79transition、官方LQR/model和F导数/caliper不变。

这只是待批准设计，**本轮未创建该时钟控制器、未改官方时间源码、未对日志或状态舍入/改写、未给观察映射自定容差**。Owner需先确认该模拟时钟/日志观察映射是否符合研究构念；若接受，下一项只能是这个有限接口修复及零运行时钟核验，保持三对名单不变。也可拒绝规范时钟；若保留原日志物理时刻，则需明确新的边界步处理和采样合同，不能让当前版本直接运行。

## 已批准且不重开的问题

旧B1维持20/20 PASS、40arms、41attempts；低阶因素消除/clean residual仍NOT_ESTABLISHED。C2唯一候选及参数不变。P1共3对/6首次arm/8总attempt、10800CPU秒和每arm1200秒不扩大；旧2轮实现不重置。S1R U-only encoder selection、H30、SESSION cluster推断、E3/E4排除及原Primary operating point不变。Coast未批准替换Primary。

v2.4本地存在但未Git跟踪，保留 `ROADMAP_V2_4_REPOSITORY_SYNC_PENDING`，原稿还停在B1前17对/路线风险快照；本轮不覆盖它，也不以旧叙述推翻正式B1闭合或Owner合同。

## 下一阶段及停止点

先完成Owner对D1–D3的书面审阅；若采用推荐D3，只授权最小时钟接口修复与同一名单零运行验证，更新source/config/contract/preflight绑定后，再申请**第一对**Development科学执行授权。不得直接将待批JSON状态改成AUTHORIZED来绕过当前clock BLOCKED。若Owner拒绝某项，保留相应阻断并停止。

本轮到此停止：科学仿真、正式Coast attempts、RBR训练/推理、H30评价、MMD/BDD、Primary/V/C均0；新科学执行、Coast P2、RBR训练及Primary均NOT_AUTHORIZED。不启动额外研究或重复概念审查。

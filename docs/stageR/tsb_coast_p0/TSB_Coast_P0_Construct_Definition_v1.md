# TSB-Coast P0 — 行为构念定义 v1

状态：`TSB_COAST_P0_KINEMATICALLY_FEASIBLE_CLOSED_LOOP_UNVERIFIED`。日期：2026-10-23（Asia/Shanghai）。

## 权限与历史边界

本轮Owner指令正式选择 `OWNER_CONSTRUCT_DECISION=ROUTE_B_SELECTED; ROUTE_B=TSB_COAST`，只授权定义、解析运动学、源码审查和协议设计。`TSB_COAST_DESIGN=AUTHORIZED`；`TSB_COAST_SCIENTIFIC_SIMULATION=NOT_AUTHORIZED; PRIMARY_PROTOCOL_AMENDMENT=NOT_YET_APPROVED; RBR_TRAINING=NOT_AUTHORIZED`。本包数值全部为**待Owner审查的提案**，不是已冻结执行配置。

历史B1永久保持 `B1_TSB_BENCHMARK_QUALIFICATION_PASS`：20/20、40个完整arm、41次尝试、1次技术retry、0科学失败；其真实构念是 `BRAKE → POSITIVE_ACCELERATION_RECOVERY → BRAKE`。旧参数、源码、roster、结果及S1/S1R合同一律保留。旧B1不是TSB-Coast资格证据；旧TSB-SR的“路线待决”是当时历史状态，本次新指令前瞻地解决方向，不回写旧报告。

`LOW_ORDER_NUISANCE_ELIMINATED=NOT_ESTABLISHED; TSB_CLEAN_RESIDUAL_TASK=NOT_ESTABLISHED`。

## 正式名称和操作性定义

中文：**制动—近似滑行—再次制动的双阶段纵向减速行为**。短名：TSB-Coast。版本命名建议：`TSB_COAST_CONSTRUCT_v1`，不得沿用旧 `R-TSB/OPTION_A` 身份而让历史adapter误分派。

在同一连续、前向行驶的有效测量窗口内，存在两个具有幅度、持续时间和降速冲量的实际负加速度阶段；二者之间存在足够长的实际近零切向加速度区间。速度可以缓慢下降，也可以在测量容差内微幅上升；**不以速度恢复为目标，不要求恒速，不允许持续明显正加速度**。Baseline为同剂量、同时间重心的单次连续制动，而不是把Treatment减速变大后与旧Baseline硬比。

必须区分以下五层，最终pair资格是各层有效性的交集：

| 层 | 所表达的内容 | 不能替代 |
|---|---|---|
| INTENDED_INTERVENTION | 非正reference加速度/非增reference速度；两脉冲与中间平台 | 不代表实际曲线已成形 |
| REALIZED_BEHAVIOR | 官方current-ego轨迹中实际分段速度和加速度 | 不代表测量可靠、匹配或安全 |
| MECHANISM_QUALIFICATION | 独立新analyzer按预冻结规则确认双制动/滑行 | 不代表F_match或安全 |
| F_MATCH | 整体剂量与运动量的配对差异受限 | 不代表两次制动、也不消除所有低阶因素 |
| OFFICIAL_SAFETY | 官方collision/drivable指标及完整工件 | 不代表行为构念或所有真实安全风险 |

## 意图、踏板与物理适用性

`BRAKE_RELEASE_INTENT`只指设计参考从负加速度回到平台。`REALIZED_NEAR_ZERO_LONGITUDINAL_ACCELERATION`必须由真实闭环轨迹测得。现有接口不记录真实驾驶员踏板状态，LQR输出是加速度和转向率请求；不能声称验证了“驾驶员抬脚”“松开踏板”“未踩油门”或驾驶心理。

近零加速度也不能唯一识别被动滑行：坡度、滚阻、风阻、驱动补偿都可能参与。此基准是**加速度行为学代理**，不是自由滑行力学模型或自然行为频率验证。参考a≤0排除人为速度恢复设计，但实际控制器仍可能产生小幅正命令；必须公开记录并按实际曲线排除明显恢复。真实车若要验证踏板机制，需要额外纵向力/坡度/踏板观测和独立协议。

拟沿用nuPlan 80帧/10Hz，真实首尾间隔7.9s，不以8.0s替代积分时间；nonreactive其他交通参与者、固定route与LQR/motion model限制外推。不能用该参数化干预冒充真正E2E软件版本行为，不能由单一已知构念宣称一般未知行为发现。

来源：本轮Owner指令（副本与SHA见Manifest）；`../b1_resume_execution_r2/B1_R2_Final_Qualification_Report_v1.md`；`../tsb_scientific_semantics_review/TSB_SR_Scientific_Construct_Review_v1.md`；实际入口见Implementation Design。

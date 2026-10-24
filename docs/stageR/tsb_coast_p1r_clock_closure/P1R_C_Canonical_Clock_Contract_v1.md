# P1R-C 规范模拟时钟合同 v1

基线提交 `8c6a3ad0bd4ec701b9f05476dbc12ee1fb76067d`；2026-10-24。D1/D2正式批准及D3限定工程授权见 `P1R_C_Owner_Approved_Scientific_Amendment_v1.json`；指令原文见evidence。代码/config实际SHA256见 `P1R_C_Execution_Source_Binding_v1.json`，本文件及所有证据hash见Manifest。

## 唯一模拟时间权威

新 `CoastCanonicalTimeControllerV1` 继承官方StepSimulationTimeController的结束/重置语义，独立实现时间查询。原始index0的整数微秒时间为t0，任何模拟iteration i的权威时间为 `t_sim(i)=t0+i×100000`。原始scenario、原R1 Primary80、官方LQR/TwoStageController/KinematicBicycleModel源码均不修改。

新Hydra组 `simulation_time_controller=tsb_coast_canonical_clock_v1` 在新factory构造时正式实例化该控制器。没有先构造旧clock再临时替换；resolved config、Simulation及其SimulationSetup持有同一clock对象。controller、observer、callback的setup均关联同一原scenario。

81个可查询iteration索引0..80。按官方end规则实际planner/history索引为0..79，controller/model transition为0→1至78→79，恰好80实现状态、79次推进。index80仅用于终点判断及元数据支持，不是第81帧实现状态。80帧首末时域为7.9秒，不以80×.1宣称8秒。

## 控制、参考与观测

- Treatment滑行严格为相对t0的 `[2100000,3900000)` 微秒，18个control step；两个边界恰落于index21、39。参考制动窗口、−1.305/−1.45幅值和C2全部参数不改。
- TwoStageController从传入的current/next iteration计算dt；官方model收到100000微秒dt并返回下一时刻EgoState。planner及记录器校验该实际时间，不自行用墙钟建立另一个episode。
- planner未来轨迹以权威当前EgoState时间递增100000微秒生成；前1.1秒两臂整个未来参考保持common Baseline。预测轨迹的未来点不是日志观察，不能混为读取未来SESSION状态。
- 原 `CoastTransitionRecorderV1` 的命令对象身份、model传播、79条完整性及物理锚点检查全部保留。新factory在安装前显式验证新clock精确类型/权威身份，安装后绑定同一clock；不将新类伪装成旧R1类名，不改旧记录器源码。
- 观察仍按原index读取；每index原日志timestamp和lidar token原样保存在mapping证据。日志中的EgoState、actor、traffic light时间戳不改写；没有插值、换帧或重采样。

## 映射验收规则与依据

`SAME_INDEX_UNIQUE_NEAREST_CELL_v1`：日志时刻严格递增、token唯一、初始t0完全相同；对每个i，令δ_i=t_log(i)−t_sim(i)，要求 `2×|δ_i| < min(100000, 相邻原日志间隔)`。两端仅取可用相邻间隔；原始timestamp必须是整数，拒绝隐式舍入。

这一规则在数学上保证规范时刻仍处于同一index的唯一最近帧单元，未越到相邻帧；**不执行nearest-neighbor重新选帧**，原index始终不变。50毫秒是10Hz半周期派生的歧义上界，另受相邻实际间隔的半值约束；它不是依当前330/1038/402微秒选出的1或2毫秒门槛，也不是被批准的传感器误差、安全容差或因果延迟豁免。三对的实际偏差及更窄数值完整披露，不能把该粗上界当成物理精度证明。

正δ意味着同index日志时标略晚于模拟时刻。这是显式的离线index回放近似，**不是严格event-time causal replay**。合法性限于此受控抽象基准的已披露映射；不能扩展到在线感知/端到端控制因果性结论。首对新授权须明确承认这一披露，见PROPOSED_ONLY草案。

## 不变项与停止规则

D1采用 `ABSTRACT_COAST_PROXY_ONLY`，不证明真实零踏板/零扭矩。D2四项F、raw梯度、六位舍入及比较规则均不变；F的物理时间来自新实际模拟状态，不能拿日志clock或参考状态替代。官方collision/drivable规则、H30、U-only、S1R及Primary统计合同保持。

任意时钟身份不符、日志排序/token重复、映射越出本index单元、锚点改变、缺帧、非整数timestamp、旧授权或未批准执行凭证均fail closed。真实闭环双制动/F/安全尚未运行，不能预先承诺通过。新科学执行、P2、RBR和Primary/V/C均未授权。

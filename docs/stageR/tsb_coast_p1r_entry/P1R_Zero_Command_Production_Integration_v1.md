# P1R 零纵向请求生产集成 v1

2026-10-24；基线提交 `a6447c285b1c349cbb6d1b31330c88ce87c2ceb2`；分支 `20260825_stageR_new`。指令原文见 `evidence/Owner_Instruction_v1.txt`；源码、配置、依赖、数据及输出SHA256统一见 `P1R_Manifest_v1.json` 和 `evidence/Source_Binding_v1.json`。

## 主结论

`P1R_BLOCKED_PHYSICAL_CLOCK_CONTRACT`，并行保留 `P1R_BLOCKED_OWNER_SCIENTIFIC_AMENDMENT`。**不声明READY，不恢复旧P1授权。**

候选适配器已经接入独立的真实官方factory，六个臂均可构建未初始化的官方runner，并绑定正式controller、model、callback、metric、serializer和新记录器。合成状态上的真实官方模型验证通过。**三对真实候选的日志时钟与精确100000微秒时钟不一致，跨越指定滑行边界；生产入口必须拒绝启动。正式C2行为仍为 `CLOSED_LOOP_BEHAVIOR_NOT_ESTABLISHED`。**

## 新栈及固定C2

- `tools/tsb_coast_p1r_planner.py::CoastC2PlannerV1` 独立生成C2参考，复用既有官方planner接口/地图查询/被动current-ego writer，不调用旧TSB generator或旧phase机制。public dispatch每次仅写一行真实current-ego；拒绝重复、缺步及不符物理锚点的调用。
- `tools/tsb_coast_p1r_production.py::build_coast_arm` 复用官方Hydra/`build_simulations`，预建新planner，并使用新配置 `configs/r1_official_technical_smoke_hydra/planner/tsb_coast_c2_p1r.yaml`。禁止复用输出命名空间；每arm独立对象。
- Treatment：官方TwoStageController → 已冻结 `ZeroLongitudinalRequestTrackerV1` → 官方KinematicBicycleModel；inner tracker仍为精确类型官方LQRTracker。Baseline保持原版未包装LQR。
- C2参考固定：Treatment −1.305@[1.1,2.1)、0@[2.1,3.9)、−1.305@[3.9,4.9)；Baseline −1.45@[2.1,3.9)。两臂参考积分均−2.61m/s。没有改幅值、时长、gain、车辆动力学或选择C1/C3。
- t<1.1时，两臂整个未来参考均用Baseline，以免LQR预瞄提前泄露Treatment；t≥1.1才切换参考。参考计划不是真实实现曲线，也不保证实际制动剂量相等。
- 滑行模式只由显式episode整数微秒锚点及 `[2100000,3900000)` 决定；不使用墙钟或未来参考索引。原适配器对跨边界控制步继续fail closed，未放宽。

## 时间接口阻断

官方 `R1Primary80ScientificTimeControllerV1` 仅限定81个迭代上限，`get_iteration`继承StepSimulationTimeController，实际时间取 `scenario.get_time_point(index)`。它没有将日志时间变成精确10Hz。新planner/记录器要求t0+i×100000；新模型方程审计也使用该冻结步长。源日志实际偏差及滑行跨边界证据在 `evidence/Production_Clock_Preflight_v1.json`。

| pair | 相对精确时钟最大偏差（微秒） | 实际步长min/max（微秒） | 2.1/3.9秒跨步 |
|---|---:|---:|---|
| DEV-01 | 330 | 99944 / 100075 | 两处均有 |
| DEV-02 | 1038 | 99923 / 100087 | 两处均有 |
| DEV-03 | 402 | 99947 / 100048 | 两处均有 |

不能把“frame21”自动当成2.1秒，也不能舍入日志时间、扩容1微秒测量容差或让适配器容忍跨界来宣告通过。`require_exact_episode_clock`会在runner之前拒绝；新授权验证还要求完整预检PASS，因此当前版本在预算claim之前即拒绝授权激活。

本轮只提出时钟合同决策，未实施时间重映射、插值、额外substep或更换SESSION。见Owner备忘录D3；这是可定位的接口/采样合同问题，不是车辆模型缺少零请求接口。

## 证据层级及历史保护

`Production_Preflight_v1.json`与`Production_Preflight_Final_v1.json`中的PASS仅表示对象/回调构建，不包含随后发现的日志时钟兼容性。以**包含实际时间戳的 `Production_Clock_Preflight_v1.json` 的BLOCKED为完整入口结论**。这些零运行对象均已丢弃，不可在guard解除后继续运行。审计中多次构建用于验证本轮确定性接口修复，没有产生新候选、科学attempt或预算重置。

旧B1仍20/20 PASS、40完成臂、41累计attempt、1技术retry、0科学失败。`LOW_ORDER_NUISANCE_ELIMINATED=NOT_ESTABLISHED`；`TSB_CLEAN_RESIDUAL_TASK=NOT_ESTABLISHED`。旧P0/P1/C0/S1/S1R、旧adapter和官方源码均只读保护。Coast未获批为Primary替代干预。

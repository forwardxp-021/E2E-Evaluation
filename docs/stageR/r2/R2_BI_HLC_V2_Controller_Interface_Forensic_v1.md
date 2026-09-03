# R2-BI HLC V2 Controller-Interface Forensic v1

## 结论

`R2_BH_V2_CONTROLLER_INTERFACE_DIAGNOSIS = SUPPORTED`。本审计只读解析既有 R2-BH telemetry，并运行合成几何计算；scientific simulation 为 0。

## 冻结控制接口

nuPlan `TwoStageController` 将 planner trajectory 交给 `LQRTracker`。LQR 在当前 iteration 从 trajectory state0 计算 lateral/heading error；R2-BH 强制 state0 与 current ego 完全相同，因此这两个误差为零。冻结配置采用 0.1 s 离散、10 step horizon，即 1.0 s lookahead；reference velocity/curvature 从完整 pose trajectory 拟合，而不是读取 V2 的 additive residual 字段。

R2-BH 共复核 3840 条 planner telemetry，state0 pose identity 为 3840/3840。R2-BH 当时没有记录 controller return value，因此 direct historical steering command 明确为 `NOT_AVAILABLE`；本轮不会把推导量冒充历史实测量。

## V2 结构失败

V2 在 base xy 上添加 lateral residual 后，独立叠加 heading residual，没有从最终 xy 重算 tangent/curvature。deadline 后又把 weight[0] 固定为 1、weight[1:] 设为 0；只要 realized residual 未归零，state0→state1 就会出现横向跳跃。straight、curved、left、right 四组合成 corridor 均重现该问题，4/4 支持 Owner diagnosis。

因此必须区分：algebraic residual term、实际 planned target-frame offset、运动学可实现 reference、LQR steering command 与 realized closed-loop offset。`state1 additive residual = 0` 不能作为 target capture 成功证据。

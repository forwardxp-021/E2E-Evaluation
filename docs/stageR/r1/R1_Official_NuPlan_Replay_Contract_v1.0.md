# R1 官方 nuPlan Replay 合同 v1.0

## 状态

`BOUND_RUNTIME_DETERMINISM_VERIFIED`。在 V3 authorization 所绑定的 planner、canonical metric parser、4-scenario roster、DB/map、seed、generator 与 simulation config 下，background replay 为 `VERIFIED_ON_BOUND_RUNTIME`，official replay 为 `READY_FOR_TECHNICAL_SMOKE_REVIEW`。

## 验证条件

- 8/8 新 V3 official runs 成功，4/4 A/B pairs 完成。
- 每个 pair 的 15 类比较均为 exact canonical equality；collision/drivable 使用冻结 canonical semantic payload，不是 Parquet container SHA。
- metric parser preflight 已零预算通过，且每次 run 都完成 fail-closed canonicalization。
- 第九次 pre-run claim 已在 simulation 之前拒绝。

## 边界

本合同只适用于上述 bound runtime，不证明 treatment effect、F_match、BDD、representation、probe 或 RBR。它只让 48-call smoke 进入 scientific-owner review 的技术就绪状态；不授权选择 roster、执行 smoke 或训练。

完整 machine-readable binding 见同名 JSON。

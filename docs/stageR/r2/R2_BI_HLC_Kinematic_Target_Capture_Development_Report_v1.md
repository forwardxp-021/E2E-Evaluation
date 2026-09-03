# R2-BI HLC 运动学 Target-Capture 架构开发报告 v1

## 结论

`R2_BI_DEVELOPMENT_NOT_CONVERGED`。V3 在 simulation 前完成了 25/25 mandatory zero-run entry gates，随后按授权启动 Round 0。第一个 baseline 完成 Primary80；第一个 treatment 在绝对时间 1.1 秒（首次允许 arm divergence）被冻结运动学可行性门 fail-closed。剩余 14 个 run 未启动，Round 1 未启动，也未冻结 selected HLC V3 或 complete G_R2 candidate。

## 失败证据

离线以同一 frozen map、route、current ego 与 Round 0 参数重建失败 planner call，得到 `max_abs_lateral_acceleration_mps2=7.391761`，超过冻结上限 `6.0`。同一 reference 的 curvature 为 `0.051156`、yaw-rate 为 `0.614922`、state0→state1 距离为 `1.214314 m`、state0 tangent mismatch 为 `0.000200 rad`、future XY-heading mismatch 为 `0`。因此失败是 controller-visible morphology/capture 合成在真实速度下违反横向加速度门，不是 XY-heading 不一致、硬跳或基础设施故障。

## 运行处置

- 工程 `runner.run` 实际调用 2 次：baseline 成功 1 次，treatment 架构失败 1 次。
- baseline artifacts 为 trace/planner/controller `80/80/79`；treatment 在失败前为 `12/11/11`。
- 已观测 controller actual 与 exact frozen shadow 为 baseline `79/79`、treatment `11/11` 方向一致，命令差为 0；treatment 数据仅覆盖分化前，不能外推为分化后的 transfer 结论。
- 此失败不是技术基础设施故障，禁止 fresh-ID 技术重跑。单一 identity 不被描述为“跨 identity 系统性失败”；但 frozen V3 contract 要求直接 fail closed，固定 cohort Round 0 无法完成，也就没有合法 aggregate numerical update 依据，因此 Round 1 不获授权。

## 防火墙

R2-BH 的 8 个 identities 已冻结为 history-only；新 DEV-KIN 8 个 identities 与 historical/R1/R2-A/R2-B/R2-BH 重叠为 0，全部永久 engineering-only。未修改 scientific mechanism、endpoint、F_match 或 safety threshold；未按 scenario/log 适配；未使用 R2-BH raw 做 V3 数值调参。

科学仿真为 0，TSB 仿真为 0；R2-C、confirmatory smoke 与 RBR 均未启动。Raw outputs 不提交 Git，仅以 SHA provenance 固化。

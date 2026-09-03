# R2-B Controller-Aware Generator 开发报告 v1

## 结论

R2-B 的最终状态为 **DEVELOPMENT_NOT_CONVERGED**。TSB 在第 0 轮达到全部 8/8 DEV-CAL 开发标准；HLC 在冻结的 4 轮上限内未达到联合标准，因此没有冻结完整 G_R2 候选，也没有进入 R2-C。

## 数据防火墙

- DEV-CAL：HLC 8 个、TSB 8 个；与 R1 official、R2-A、既有黑名单的重叠均为 0。
- R2-A surrogate 仅用于第 0 轮初始化；R2-A identity 未重跑。
- 数值反馈只来自冻结的 R2-B DEV-CAL identities；没有 scenario token/log ID 查表适配。
- 科学阈值、F_match、endpoint、engineering 与 official safety 定义均未修改。

## 架构

HLC 将 `DESIRED_REALIZED_MORPHOLOGY` 与 `PRECOMPENSATED_PLANNER_MORPHOLOGY` 分离，显式参数化 advance、hold、retreat、recommit、lag 与 settling。TSB 显式参数化 first-brake、release、second-brake 的幅值和有效时长，并在 absolute-time repeated replanning 中补偿 phase shortening、boundary migration、lookahead mixing 与 release carryover；不是简单常数增益求逆。

## 校准结果

- HLC：4 轮；最终 mechanism 6/8、F_match 8/8、endpoint 0/8、engineering 8/8、safety 8/8。8/8 treatment 均实现至少一次 retreat，延迟裕量均为正；两个 pair 的 monotonic 差值未越过冻结 -0.10 gate。endpoint 失败由 treatment terminal offset gate 主导。
- TSB：1 轮；measurement OK 8/8、baseline one-phase 8/8、treatment two-phase 8/8、完整 mechanism 8/8、F_match 8/8、safety 8/8。
- 实际 DEV 工程运行：80。HLC 第 0 轮 16 个已完成产物只做了后处理恢复，恢复未增加 runner.run。

## 治理处置

未生成 `r2_b_selected_generator_parameters_v1.0.json`，因为完整候选不满足冻结的 HLC+TSB 联合开发标准。严格停止在 4 轮上限；不增加第 5 轮、不换 identity、不降低 gate、不选择 R2-C identities、不启动 confirmatory smoke 或 RBR。

## 产物溯源

- round summary SHA256：`4d91cf3ce6093d5ee405119bc4b7d624b481a6c6b9f8cef8274e2b64f7559c8c`
- data firewall SHA256：`a56403ee65bf1138b9c0d74567b9efb19ac6bd366005d9c4d85ecc40c5ad1488`
- raw DEV aggregate tree SHA256：`4689058d3e442a52156af9d26f04b352efdb94e87feebb85df3d4bdecf2794c8`

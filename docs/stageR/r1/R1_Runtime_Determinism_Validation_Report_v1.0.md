# R1 Runtime Determinism Validation V2 报告 v1.0（fail-closed）

## 结论

`R1_RUNTIME_DETERMINISM_VALIDATION_V2 = FAIL_CLOSED_TECHNICAL_OUTPUT_DISCOVERY_FAILURE`。
`BACKGROUND_REPLAY_DETERMINISM = NOT_VERIFIED`，`OFFICIAL_REPLAY = NOT_READY`。

本报告只记录 bound-runtime reproducibility 的执行完整性，不解释 HLC/TSB 的科学 outcome，未执行
treatment、48-call smoke、roster re-selection、development roster 或 RBR training。

## 冻结输入与零预算预检

- V2 授权：`AUTHORIZED_ONCE`；只允许新增 8 个 `OFFICIAL_CLOSED_LOOP_RUN`，原四个 outcome-blind roster
  scenario 各使用 `V2_RUN_A/B` 一次。V1 的历史失败不计入 V2 cap，也不构成 V2 evidence。
- roster：R-HLC=2、R-TSB=2、token=4、log=4，永久隔离标签均保留；无 selector rerun、无 scenario replacement。
- final interface preflight：`PASS_NO_OFFICIAL_RUN_BUDGET_CONSUMED`；修复后的 planner 为 nuPlan
  `AbstractPlanner` 子类，并完成 HLC 与 TSB 各一次内存单步调用。preflight 的 claim/start/budget 均为 0。
- 绑定：`MASTER_SEED=2026082701`；planner、V2 executor、DB/map、HLC/TSB generator 与 simulation config SHA
  均由 V2 authorization 固定。

## 实际 V2 执行

第一笔且唯一一笔 pre-run claim 是
`R-HLC__25944935eadb52f1__V2_RUN_A`。账本记录为 `V2_CLAIMED_BEFORE_SIMULATION`，实际运行号 1。
nuPlan runner 报告 1/1 simulation 成功，planner trace 为连续 149 steps，绑定文件也存在；命令返回码为 0。

但 V2 执行器的 metric discovery 仅枚举 `*.json`，而官方 nuPlan 这次输出的 collision 与
drivable-area 指标是 Parquet 文件。因此执行器记录
`COLLISION_METRIC_UNAVAILABLE` 与 `OFFROAD_DRIVABLE_METRIC_UNAVAILABLE`，并将该 run 判为
`TECHNICAL_FAILURE`。该 fail-closed 判定触发后，未启动 V2_RUN_B 或其余三行 roster：实际为 1/8，
没有 A/B pair，也没有第九次 pre-run cap 检查。

## 冻结判定

| 项目 | 状态 | 依据 |
|---|---|---|
| V2 官方运行数 | `1/8，EARLY_STOP` | 第一条运行后立即停止；未重跑。 |
| 15 类 exact comparison | `NOT_EVALUABLE` | 缺少 V2_RUN_B；comparison CSV 仅保留表头。 |
| background replay determinism | `NOT_VERIFIED` | 0/4 可比较 A/B pair。 |
| official replay | `NOT_READY` | V2 被 metric-discovery technical failure 收束。 |
| collision/drivable official artifacts | `PRESENT_AS_PARQUET_BUT_NOT_DISCOVERED_BY_FROZEN_V2_EXECUTOR` | 原始输出中存在相应 Parquet；执行器没有纳入哈希比较。 |
| 48-call core budget executor | `READY_BY_FAIL_CLOSED_PREFLIGHT` | 独立的 pre-call cap 机制已就绪；不等同于烟雾授权。 |
| official closed-loop execution path | `NOT_READY_AFTER_V2_TECHNICAL_FAILURE` | 与上项明确分离；不得据此启动 48-call。 |

## Protocol deviation 与科学诊断

本次不是 mechanism、threshold、target、roster 或 primary scientific metric 的修改，故
`SCIENTIFIC_PROTOCOL_DEVIATION = NO`。问题是 V2 executor 对官方 output 文件格式的发现遗漏，记录为
`EXECUTION_OUTPUT_DISCOVERY_FAILURE`；它改变了执行完成性，因而必须保留 `NOT_VERIFIED/NOT_READY`，
不能把已生成的 trace 或 Parquet 文件事后当作完整 pair 证据。

修正后的 Wave1 scientific diagnosis 是：本次只证明 repaired planner 可完成单个 bound runtime simulation
并产生 trace；没有证明 A/B replay determinism，也没有形成 HLC、TSB 或任何 outcome 的科学结论。

## 后续边界

不允许在本授权下修复 metric discovery 后重跑、继续余下 7 次、替换 scenario、调整配置或启动 48-call。
若未来需要恢复 V2/replacement validation，必须由 scientific owner 另行授权新的绑定执行；该授权不得由本
报告自动推导。

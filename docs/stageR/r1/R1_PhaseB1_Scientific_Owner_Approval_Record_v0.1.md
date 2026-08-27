# R1 Phase B1 科学负责人批准记录 v0.1

状态：`APPROVED_AS_RECORDED`。本记录只授权 R1 Phase B1.1 的合同冻结与一次性 runtime determinism validation，不授权 fresh 48-call smoke、development roster、treatment rollout、RBR 或任何 representation/BDD/probe 读取。

- HLC generator 选择 `HLC_GEN_V2_OPTION_B`，状态 `APPROVED_FOR_PARAMETER_FREEZE`。
- endpoint Primary validity 选择 `OPTION_ENDPOINT_RESOLUTION_BASED`：offset `≤0.25m`、heading `≤0.05rad`、lateral velocity `≤0.25m/s`、pair route-progress delta `≤1.5m`。strict 方案仅 sensitivity audit，不能排除 Primary-valid pair。
- `OFFICIAL_NUPLAN_DB = READY`；runtime 仍为 `NOT_READY_PENDING_REPLAY_DETERMINISM`。
- 批准 `MASTER_SEED=2026082701` 与既有版本/SHA/seed propagation binding；background determinism 尚未验证。
- 一次性授权 4 scenarios × 2 repetitions = 最多 8 个 `OFFICIAL_CLOSED_LOOP_RUN`，仅 baseline；新 compliant 48-call smoke 仍 `NOT_AUTHORIZED`。

冻结 mechanism thresholds、R0 历史结果与旧 12 smoke identity 均不得修改或重用。

# R1 B2.7-R1 / B2.8-R1 快速操作说明

## 1. 命令

本轮枚举已完成，**不得再次运行**下列命令；它属于已消耗的一次性授权：

```bash
PYTHONWARNINGS=ignore /Users/liuqing/miniconda3/envs/nuplan/bin/python3.9 \
  tools/r1_b2_7_freeze_official_smoke_roster_v2_1.py
```

核验已冻结结果可运行：

```bash
PYTHONWARNINGS=ignore /Users/liuqing/miniconda3/envs/nuplan/bin/python3.9 -m pytest -q \
  tests/test_r1_b2_7_r1_lazy_selector.py \
  tests/test_r1_b2_7_freeze_roster.py
```

B2.8 只读、零额度完整性预检可运行：

```bash
python tools/r1_b2_8_official_smoke_integrity_preflight.py
```

该命令当前预期以退出码 `2` 停止：它不启动仿真、不消费 48-run budget；只有获得新的 Scientific Owner 授权后，才允许修复已记录的执行集成缺口。

B2.8-R1 修复与零运行重新预检已完成，**不得再次执行**；其输出是新的待审 execution SHA：

```bash
/Users/liuqing/miniconda3/envs/nuplan/bin/python3.9 -m pytest -q \
  tests/test_r1_b2_8_r1_execution_wiring.py
```

## 2. 期望行为

冻结工具先对 1,624 个 DB 做全局 token 去重与 source count closure，再按固定 SHA rank 逐个执行原冻结 eligibility，直到 HLC/TSB 各 12 个身份被数学确定。它生成 roster、48-run schedule、leakage/dedup/eligibility 审计和 24/24 zero-run preflight。

它不会启动 `run_simulation.py`、official simulation、planner rollout、embedding、BDD 或 RBR。

B2.8 预检会核验授权 SHA、24 个固定身份、48 个固定 run、24/24 DB/map/route 预检记录、保护指标 SHA 与绑定运行时路径。它还会核验冻结 V2.1 链能否产出 `REALIZED_CURRENT_EGO` 的 80 帧主测量输入；任何失败均以 `STOP_PRE_SIMULATION` 结束。

B2.8-R1 使用版本化 V2.2 planner：仅在 planner 调用入口被动记录当前已实现 ego state，再直接委托 V2.1 生成轨迹。Hydra 通过精确 run ID 从 immutable schedule 与 roster 唯一构造参数，缺失、歧义或 arm 不匹配均在 simulator start 前失败。

## 3. 通过标准

- `5,405,672 - 19,097 = 5,386,575`。
- 全局 log 数为 1,621，重复 identity conflict 为 0。
- HLC 12、TSB 12，24 个 token 和 24 个 log 均唯一。
- schedule 有 48 个唯一 run ID、24 个 pair ID，且第 49 个预算 claim 被拒绝。
- zero-run roster preflight 为 24/24 PASS。
- leakage audit 为 `PASS_NO_OUTCOME_INPUTS`。
- `OFFICIAL_SMOKE_AUTHORIZED=false` 且 `NEW_RUN_BUDGET=0`。
- B2.8 当前为 `STOP_PRE_SIMULATION`，actual official runs 与 consumed budget 均为 `0/48`。
- v1.1 预检仅保留两项 fail-closed 缺口：realized-current-ego trace writer 未绑定，以及 Hydra 的 per-run frozen roster-row 参数未绑定。
- B2.8-R1 的 48/48 Hydra composition、80-row trace 与 fail-closed 测试均通过，`PRE_RUN_INTEGRITY=PASS_COMPLETE_EXECUTION_PATH_ZERO_RUN`。
- 该通过不构成 run 授权：`OFFICIAL_SMOKE_AUTHORIZED=false`、`NEW_EXECUTION_RUN_BUDGET=0`、RBR 仍未授权。

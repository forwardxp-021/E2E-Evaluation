# R1 B2.7-R1 快速操作说明

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

## 2. 期望行为

冻结工具先对 1,624 个 DB 做全局 token 去重与 source count closure，再按固定 SHA rank 逐个执行原冻结 eligibility，直到 HLC/TSB 各 12 个身份被数学确定。它生成 roster、48-run schedule、leakage/dedup/eligibility 审计和 24/24 zero-run preflight。

它不会启动 `run_simulation.py`、official simulation、planner rollout、embedding、BDD 或 RBR。

## 3. 通过标准

- `5,405,672 - 19,097 = 5,386,575`。
- 全局 log 数为 1,621，重复 identity conflict 为 0。
- HLC 12、TSB 12，24 个 token 和 24 个 log 均唯一。
- schedule 有 48 个唯一 run ID、24 个 pair ID，且第 49 个预算 claim 被拒绝。
- zero-run roster preflight 为 24/24 PASS。
- leakage audit 为 `PASS_NO_OUTCOME_INPUTS`。
- `OFFICIAL_SMOKE_AUTHORIZED=false` 且 `NEW_RUN_BUDGET=0`。

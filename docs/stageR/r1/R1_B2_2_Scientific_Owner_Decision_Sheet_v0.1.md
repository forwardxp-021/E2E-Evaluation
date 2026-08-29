# R1 B2.2 Scientific Owner 决策单 v0.1

## 当前冻结结论

- 当前状态：`BENCHMARK_FAMILY_NOT_READY`。
- 本页全部选项均为 prospective；未获 owner 明确批准前，不形成新冻结、不允许新 rollout。
- 禁止 outcome-driven threshold tuning；任何新版本都必须在执行前绑定实现 SHA、测量源和适用性规则。

## 待 owner 决策的候选修正

| 选项 | prospective 内容 | 本次是否实施 | owner 决策 |
|---|---|---:|---|
| A | 按冻结合同从 official observation 和 map query 构造 lane-aware slots、稳定 track ID、gap/relative speed/THW、traffic density、hazard multi-hot 与真实 missingness | 否 | 待批准 |
| B | TSB 改为 route-aligned longitudinal trajectory，同时保留 Option-A acceleration profile；明确连续重规划锚点 | 否 | 待批准 |
| C | HLC 在冻结前声明 map geometry applicability 与原生 8 秒 reference coverage；不得按结果拟合阈值 | 否 | 待批准 |
| D | planner 首状态以 current ego 形成连续 anchor，并版本化位置/heading/speed/timestamp 语义 | 否 | 待批准 |
| E | 在执行前冻结 primary measurement source：planned-first 或 realized-ego；若双报告，必须指定主次与冲突解释 | 否 | 待批准 |

## 建议的审批顺序

先审 A、D、E，再判断是否需要 B/C。当前证据不支持直接修改冻结生成器参数；先消除上下文和重规划实现缺陷，才能隔离 generator-specific failure。

## 执行授权

- 新 planner rollout：`NOT_AUTHORIZED`
- D2/D4：`NOT_AUTHORIZED`
- RBR A/B/C：`NOT_AUTHORIZED`

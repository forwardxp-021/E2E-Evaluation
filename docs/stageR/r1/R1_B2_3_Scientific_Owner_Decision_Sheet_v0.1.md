# R1 B2.3 Scientific Owner 决策单 v0.1

当前状态：`NEW_ROLLOUT_NOT_AUTHORIZED`。下一轮 owner 只需决定以下五项：

| 项 | 待决定内容 | 当前证据/状态 |
|---|---|---|
| A | 是否批准 TSB initial-speed floor = 2.0 m/s 并进入 final freeze | 解析推导与 0.001 m/s synthetic grid 一致；`PROPOSED_REQUIRES_OWNER_APPROVAL` |
| B | HLC dynamic-clearance 数值合同 | 8 s common-envelope 结构已提出；所有新增 buffer/footprint fallback/interpolation-gap 数值仍为 TBD |
| C | HLC map applicability 是否仍需要新 numeric value | 当前 exact topology、NO_EXTRAPOLATION 与既有 engineering limits 无新增值；若需要必须单独批准 |
| D | prospective implementations 是否与 owner A–E 决策一致 | context v2、realized primary、current-ego anchor、route/native geometry 已实现 synthetic/read-only 版本，待 owner review/freeze |
| E | 是否授权全新 identity 的下一轮 official smoke | 当前 `NOT_AUTHORIZED`；只有 A–D 与 selector preconditions 全部冻结后才可考虑 |

本决策单不授权 enumeration、roster selection、rollout、D2/D4 或 RBR。

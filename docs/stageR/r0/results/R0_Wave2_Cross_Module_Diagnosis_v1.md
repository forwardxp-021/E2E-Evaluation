# R0 Wave 2 跨模块科学诊断 v1

证据等级：`DEVELOPMENT_DIAGNOSTIC_EVIDENCE`。本诊断不修改 Protocol v1.0、D0/D1 gate、历史 Stage7L BDD 的 primary 结论，亦不授权任何 RBR 训练。

## 现有证据的正确拼接

- D0（Wave 1.1）：`D0_POOLING_EFFECT = MIXED`，`D0_MASK_PADDING_SENSITIVITY = MIXED`；embedding geometry sensitivity 有支持，但 semantic retention 不可概括为 information loss。
- D1：`KNOWN_SEMANTIC_INFORMATION_PRESENT = SUPPORTED` 维持不变。Wave 2 的 Waymo→nuPlan direct frozen-probe transfer 已执行，但没有冻结的跨域数值支持门，故 `D1_CROSS_DOMAIN_SEMANTIC_TRANSFER = INCONCLUSIVE`。
- D2：唯一可执行的是使用原生邻居缺失 sentinel 的诊断视图；ego 零化/仅上下文不具备合法缺失语义，完整上下文 shuffle 分层不能构建。因此 response、pairing 为 `NOT_EVALUABLE`，context、shortcut、ablation OOD 为 `INCONCLUSIVE`；绝不由自然数据 shuffle 或零化结果声称因果耦合。
- D3：三个 formal hypothesis 继续为 `INCONCLUSIVE`，没有重跑或改变其 primary 结果。

## 修正后的 Wave 1–2 scientific diagnosis

`CASE_C_TEMPORAL_CONTRIBUTION_MIXED_NOT_GENERALIZED` 继续有效：当前可支持的是 temporal/pooling geometry sensitivity，而非普遍的 temporal information-loss 结论。D1 显示 Waymo 冻结表示中存在可读 CORE semantic information；Wave 2 没有提供足以把这种可读性升级为跨域 semantic transfer 支持的冻结门证据。D2 也没有提供可归因的 ego/context causality 或 context shortcut 支持。

因此，正确的压缩结论是：`KNOWN_SEMANTIC_INFORMATION_PRESENT_SUPPORTED; CROSS_DOMAIN_TRANSFER_INCONCLUSIVE; GEN1_CONTEXT_RESPONSE_ATTRIBUTION_UNRESOLVED; D3_FORMAL_HYPOTHESES_INCONCLUSIVE`。

Wave 1 的 D0 primary-metric omission 已按接受的 completeness correction 关闭，未新增 evidence downgrade；这不是对 Protocol 的修改。

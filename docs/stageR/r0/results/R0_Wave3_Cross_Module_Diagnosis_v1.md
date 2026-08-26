# R0 Wave 3 跨模块科学诊断 v1

证据等级：`DEVELOPMENT_DIAGNOSTIC_EVIDENCE`。

- D0：pooling 与 mask/padding 均为 `MIXED`；可支持 geometry sensitivity，不支持普遍 information loss。
- D1：`KNOWN_SEMANTIC_INFORMATION_PRESENT = SUPPORTED`；cross-domain transfer 仍为 `INCONCLUSIVE`。
- D2：response/pairing `NOT_EVALUABLE`，其余正式状态保持 Wave2 的 `INCONCLUSIVE`；未解决的 D2 contract 不支持 RBR-C。
- D3：formal hypotheses 继续 `INCONCLUSIVE`。
- D4（Wave3）：R-HLC、R-TSB、R-IP 的 descriptor、mechanism 与 outcome-blind feasibility 三类 formal hypothesis 均为 `NOT_EVALUABLE`。这是冻结 implementation/context capacity limitation，不是 outcome-driven negative finding。

修正后的科学诊断：`KNOWN_SEMANTIC_INFORMATION_PRESENT_SUPPORTED; TEMPORAL_GEOMETRY_SENSITIVITY_MIXED; CROSS_DOMAIN_TRANSFER_INCONCLUSIVE; GEN1_CONTEXT_RESPONSE_ATTRIBUTION_UNRESOLVED; D3_INCONCLUSIVE; D4_RESIDUAL_BENCHMARK_NOT_EVALUABLE_WITH_EXISTING_ASSETS`。

`CASE_C_TEMPORAL_CONTRIBUTION_MIXED_NOT_GENERALIZED` 保持，不因本 Wave 3 改写。

# R2-BJ-A2 Scientific Owner 准备度请求 v0.1

## 结论

`REQUEST_WITHHELD`。A2 fail-closed 主状态为 `JOINT_SUPPORT_EXTRACTION_INCOMPLETE`；阻断类别为 `JOINT_SUPPORT_EXTRACTION_INCOMPLETE, CURVATURE_REPRESENTATION_UNRESOLVED`。

## 联合支持结果

仓库中 57 个唯一、已提交且 outcome-blind 选出的 HLC opportunity 被逐一检查，其中 47 个完成 V2.3 长窗口 joint-record reconstruction，10 个未完成。现有冻结 eligibility 管线只持久化 rank-stopping cohort，没有持久化全 source universe 的全部 eligibility-pass population，因此不能声称联合适用域提取 100% 完整。

已完整形成的 47 条 joint record 内部 provenance closure 为 100%；但对 57 条已提交记录的 extraction completion 仅为 82.46%，全 eligible population 的完成率不可计算。

在已物化 joint support 内，native-only 不可行 population 为 0/47，generated increment 不可行为 0/47，composite 不可行为 0/47，recommit 后 terminal settling 不可行为 0/47。这些记录不触发自动 identity exclusion。

## 曲率质量处置

主 joint support 的 source/target 曲率分类为：{'LOCALIZED_POINTWISE_SPIKE': 20, 'MIXED_CURVATURE_REPRESENTATION_UNRESOLVED': 7, 'RAW_ROBUST_CONCORDANT_SUSTAINED': 67}。其中仍有 7 条 reference side 的 raw/robust 关系不能按预注册规则归入“局部尖峰”或“持续曲率”，因此保留 `CURVATURE_REPRESENTATION_UNRESOLVED` 阻断。

历史 `0.082281 1/m` 已按 B2.1 原公式复现：它位于 target reference 末端的超短 segment 支持点；A2 turning-angle raw 与固定窗口 robust 均约为 `0.001 1/m`，且两条历史记录均缺少完整 7.9 秒 target reference coverage。因此该值被判为 terminal discretization/gradient artifact，仅留在 adversarial appendix，不作为实际 speed-curvature joint support。

## 治理

V4 参数和冻结阈值均未修改；BJ-A Cartesian envelope 仅作为 adversarial appendix 保留。未选择 roster，未申请 BJ-B execution。`runner.run=0`，engineering/scientific/TSB simulation 均为 0；R2-C、confirmatory smoke、RBR 均未启动。

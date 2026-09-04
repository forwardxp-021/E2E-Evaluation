# R2-BJ-A3 Scientific Owner 准备度请求 v0.1

## 结论

`JOINT_SUPPORT_EXTRACTION_INCOMPLETE`。

请求暂缓；保持 fail-closed，不进入 BJ-B。

## 前瞻性适用域闭环

- 固定 hash-ranked audit frame：256/256 全部完成，无提前停止。
- 同一 A3 predicate 通过：17/256（6.64%）。
- 通过者不是 BJ-B roster，本阶段没有选择或运行任何 identity。
- 47 条 A2 完整历史记录在修正速度包络下通过：46/47。
- 10 条历史 extraction failure 已统一重放并获得技术处置：10/10；未加入结果型 blacklist。
- 通过者 provenance、reference geometry、速度与组件审计 closure：100.0%。

## 速度与曲率

主审计速度严格使用 `max(official initial, pre-treatment 0–1.0 s max)`；裕量速度严格使用 `v_audit + max(0.5, 0.05*v_audit)`。anchor timestamp 仅保留作 provenance。

raw 与 robust 曲率均保留；所有通过者均有预注册的明确处置，无未定义 catch-all。历史 `0.082281 1/m` 保留在 adversarial appendix，并标记为 terminal short-segment gradient artifact，不进入实际 joint support。

## 治理

阻断类别：JOINT_SUPPORT_EXTRACTION_INCOMPLETE, V4_GENERATED_INCREMENT_INFEASIBLE。V4、科学/运动学阈值和结果防火墙未改变。`runner.run=0`，engineering/scientific/TSB simulation 均为 0；BJ-B roster、R2-C、confirmatory smoke、RBR 均未启动。

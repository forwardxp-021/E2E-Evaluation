# R0 Wave 2：RBR 设计含义 v1

本文件只给出未来设计约束；`RBR_A/B/C_TRAINING_AUTHORIZATION` 仍为 `NOT_AUTHORIZED`，本波不训练模型。

1. RBR-C 应原生表示 ego、context、neighbor source 的可用性，并把 missingness/slot validity 作为明确输入合同。共享 83D GRU 的 raw-zero 不能代替 source absence。
2. 训练与评估前应预先定义 full、ego-only、context-only、neighbor-ablated 和保持分层的 shuffle；每个视图都需要训练内覆盖或可证明的 OOD 控制。
3. 未来 context-shuffle 必须在冻结的 scenario family、direction、initial speed、traffic density、neighbor availability 和 event phase strata 内执行；应在数据入库时持久化所有预处理分层与独立单位键。
4. 应保存 R0 reference embedding/input bank 的可审计边界、PCA/nearest-neighbor OOD contract 与条件级聚合规则，避免在结果出现后决定何时称 OOD dominated。
5. 保持九个 CORE D1 targets、Waymo-only frozen probe 与 nuPlan direct-transfer 的区分；跨域支持门必须在未来执行前冻结，不能从本次描述性结果调参。
6. RBR-C 的 shortcut 控制应包含 log/map/route/source identity 的 group-aware contrast，并要求场景 identity 高预测能力不能单独作为 shortcut 结论。

这些是 execution/design constraints，而非关于模型优劣、因果 interaction 或训练授权的结论。

# R1 B2.9-A Runtime Repair Decision Matrix v1

本文件只做技术决策分析，不选择、不实现 repair，不授权 simulation。

| 维度 | OPTION A Active-reference-aware sampling | OPTION B Route-continuous native reference | OPTION C Decouple planner horizon |
|---|---|---|---|
| scientific semantic change | 对严格 zero-weight 项跳过采样在数学上不改变已定义 XY；但不能解决 active target 同步耗尽 | 有：reference domain 从单 lane baseline 扩展到 successor/route-continuous geometry | HLC progress 与 Primary 80-frame evaluation 不变，但 realized closed-loop path 可能改变 |
| runtime semantic change | 低；改变 evaluation order/访问集合，需要新 planner 版本与 exact parity | 中；每次 replan 使用扩展后的 native reference，需要新 builder/planner 版本 | 高；LQR 全轨迹速度/曲率拟合输入缩短，控制量可能改变 |
| map semantic change | 无 | 高；必须定义 source/target successor pairing、route occurrence、connector、adjacency continuity、方向、拼接与 ambiguity fail-closed | 无 |
| expected contamination risk | 低，但 repair 不充分 | 中；若规则 pre-frozen 且 outcome-blind 可控 | 中高；直接改变 controller 对未来轨迹的拟合输入 |
| need new selector? | 否 | 是：需要新的 route-continuous applicability audit；不得沿用旧 one-shot eligibility | 否；但需要全 roster 的 controller-parity preflight |
| need new roster? | 否 | 是：新版本 roster freeze；可优先 re-qualify 原 23 个未运行 identity，但不能假定全部通过 | identity roster 可不变，但 execution manifest/planner version 必须更新 |
| need new planner version? | 是 | 是 | 是 |
| can preserve 23 unrun identities? | 技术上可以，但 target exhaustion 仍阻断 | 条件性可以：只有新 map rule 下全部重新通过才可保留 | 可以作为 identity；必须使用统一新 planner 重建 execution binding |
| can consumed identity ever be reused? | 只能在新版本 repair protocol 下，不能作为当前 retry | 只能在新版本 repair protocol、new run ID/manifest/authorization 下 | 同左 |
| 对本次 iteration 33 的充分性 | 不充分：source 可跳过，但 target 仍越界 | 可能充分，需 12/12 rolling coverage + topology ambiguity 验证 | 可能充分，但改变 controller behavior，需重新冻结 comparability |

## OPTION A 评估

严格跳过数学权重为 0 的 reference 是 implementation-level correction，并可保持相应输出的代数语义；但本次 iteration 33 的 target 权重为 1 且 margin 为 `-0.5231 m`。因此 OPTION A 单独实施会立即在 target sampling 处再次失败，不能作为本故障的完整修复。

## OPTION B 评估

它保留每次 7.9 秒 planner trajectory、现有 LQR 消费结构、HLC progress 和 speed schedule，是三者中最能避免 controller-horizon contamination 的方案。但它不是纯 bug fix：必须版本化定义 native successor/route extension，并对 source/target 的 successor 对齐、roadblock/connector、branch ambiguity、方向、拼接连续性、adjacency 与 route occurrence 建立新规则。需要新 selector applicability contract 与新 roster freeze；不得根据本次 outcome 选择 successor。

## OPTION C 评估

当前 LQR 的显式 lookahead 为 1.0 秒，curvature profile 到 0.9 秒；11 states / 1.0 秒是避免显式 lookahead clamp 的最小规则候选。但现有 tracker 会对 planner 的整个输出执行全局正则化拟合，缩短到 1.0 秒会改变速度/曲率估计，进而可能改变 acceleration/steering。故它不能被认定为 scientific-neutral，必须新 planner/controller contract、行为差异审计和全 12 identity parity/contamination 评估。

## 建议（不构成选择或授权）

建议优先由 Scientific Owner 审议 **OPTION B 的版本化 route-continuous repair**，原因是它能保持当前 7.9 秒 trajectory 与 controller 消费结构；OPTION A 可作为独立的 zero-weight 防御性修正候选，但不能单独解决故障；OPTION C 仅在接受 controller behavior 改变并重新冻结 comparability 时考虑。

治理建议：新 selector required = `YES`；新 roster version required = `YES`。23 个未运行 identity 仅可在新规则下 outcome-blind re-qualification 后保留。consumed identity 永久保留 Attempt 1 exclusion，但可由 Owner 在新 protocol version 中批准一次显式 `VERSIONED_TECHNICAL_REPAIR_RERUN`；当前禁止 retry。

`SIMULATION=NOT_AUTHORIZED`；`RBR_A/B/C=NOT_AUTHORIZED`。

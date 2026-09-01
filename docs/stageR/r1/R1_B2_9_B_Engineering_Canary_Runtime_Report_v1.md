# R1 B2.9-B 工程 Canary 运行报告 v1

## 证据边界

本报告只记录 `NON_SCIENTIFIC_ENGINEERING_ONLY` 的技术运行行为，不是 official smoke、科学证据或 benchmark 结果。三个身份已写入永久科学排除账本；不得用结果调阈值、机制、F_match、安全定义或未来身份选择。

Canary identities：`b1be12bca092597a`、`25944935eadb52f1`、`ef3172a208cc5dd7`，每个均执行 baseline/treatment。

A01 为配置环境绑定缺失，6 次均在 runner 构造前停止；A02 证明 6/6 Primary 0...79 完整，但在 secondary 区间遇到官方目标车道拓扑终点；A03 保持严格 fail-closed 地图合同，以 80-call 工程 canary time-controller 正常完成全部 runner 与回调。所有尝试均使用新 run ID 和新输出根。

## 运行结果

实际 canary runs `12`，rerun `12`，技术完成 `6`，Primary 0...79 完成 `12`，历史累计 native coverage failure `6`，最终 native coverage failure `0`，最终其他技术 failure `0`，metric/callback 完成 `6`。历史累计 pre-start/wiring failure 与 post-primary coverage failure 均保留在 ledger。

最终状态：`ROUTE_CONTINUOUS_ENGINEERING_CANARY_PASS`。Attempt 1 身份只在先冻结为 `SCIENTIFIC_EVIDENCE_EXCLUDED=true` 后作为 canary replay；其余当前科学身份仿真：`false`。科学 roster 修改：`false`；threshold 修改：`false`。OFFICIAL_SMOKE_AUTHORIZED=false，RBR_A/B/C=NOT_AUTHORIZED。

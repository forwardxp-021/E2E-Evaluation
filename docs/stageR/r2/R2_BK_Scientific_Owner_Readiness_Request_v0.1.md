# R2-BK Scientific Owner Readiness Request

## 已完成冻结

1. HLC V4 canary 已作为有效的 negative engineering result 关闭；HLC V5 与剩余 14 runs 均未授权。
2. B1 canary identity 已 outcome-exposed 永久隔离；其余 7 个 B0 identities 继续冻结未运行。A5 剩余 26 条未自动改作 confirmatory。
3. family-scope bifurcation 已冻结：完整 `G_R2` claim 不存在，HLC 为 development nonconvergence，TSB 仅是待 fresh validation 的独立 family candidate，禁止 cross-family pooling。
4. TSB candidate 与参数、generator、8/8 DEV-CAL 结果及 39/33 项上层 manifest 绑定闭合。冻结最小 release-fraction margin 为 `0.588132`，最小 second-peak-ratio margin 为 `0.700240`。

## Fail-closed 容量结论

当前冻结 source-universe 文件只物化了 `5,338,021` 个 token 的集合 SHA 和 `1,564` 个 log 名单；历史 B2.7 明确记录 TSB total eligible count 为 `NOT_EXHAUSTIVELY_COUNTED_BY_DESIGN`。现有资产没有逐 token 的 TSB eligibility-pass population。

按永久历史/工程 exclusion 与 A5 角色保留做完只读集合审计后，结构性 log 上界为 `1,425`，但它不是 TSB eligible capacity，不能用来证明任何样本量可行。要得到真正的全量 eligible-capacity census，必须对冻结 source universe 做一次性、只读的 predicate materialization；这与本轮 `RESCAN_SOURCE_UNIVERSE=FALSE` 的授权边界冲突。因此本轮不使用局部扫描推断总体，不选择 roster、reserve 或 schedule。

## 请求 Owner 决策

请先决定是否另行授权一次性、只读、SHA 绑定的 frozen-source TSB eligibility materialization。容量闭合后，再从以下尚未选择的档位决定 TSB R2-C 样本量：

| 档位 | pairs | runs | 治理权衡 |
|---|---:|---:|---|
| MINIMAL | 12 | 24 | 预算最低，跨 identity 稳定性覆盖最弱 |
| RECOMMENDED | 20 | 40 | 覆盖与预算折中 |
| ROBUST | 32 | 64 | 覆盖更强、预算最高 |

上述档位只是候选，不是已预注册样本量。当前状态保持 `TSB_R2C_SAMPLE_SIZE_REQUIRES_OWNER_DECISION`；在容量和样本量均由 Owner 闭合前，不得生成真实 roster、reserve 顺序或 run schedule。

## 当前权限

`RUNNER_RUN=0`，`OFFLINE_RECOVERY_ANALYZER_INVOCATIONS=0`，`ROSTER_SELECTION=FALSE`，`R2_C_STARTED=FALSE`，`CONFIRMATORY_SMOKE_STARTED=FALSE`，`RBR_STARTED=FALSE`。

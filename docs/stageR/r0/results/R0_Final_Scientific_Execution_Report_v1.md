# R0 最终科学执行报告 v1

## 正式状态

|项目|正式状态|
|---|---|
|`R0_V1_SCIENTIFIC_EXECUTION`|`COMPLETE_WITH_NOT_EVALUABLE_COMPONENTS`|
|`R0_D4`|`NOT_EVALUABLE_WITH_EXISTING_HISTORICAL_ASSETS`|
|`RBR_FORMAL_TRAINING`|`NOT_AUTHORIZED`|
|`RBR_DEVELOPMENT_AUTHORIZATION`|`BLOCKED_PENDING_RESIDUAL_BENCHMARK_ENABLEMENT`|

R0 v1 的冻结协议执行至此正式结束。冻结 tag
`r0-v1.0-protocol-freeze` 对应的 commit 为
`319757c7f72efb55c80c780e4d0f17e5341b19ec`；全部 19 个绑定资产的 SHA256
在 Wave3 后复核一致。

## 保留的科学结论边界

- D0 Wave1.1：`MIXED`。只支持 pooling geometry sensitivity；不把该结果升级为
  temporal information loss 支持。Case C 保持 `MIXED_NOT_GENERALIZED`。
- D1：`KNOWN_SEMANTIC_INFORMATION_PRESENT = SUPPORTED`；跨域 transfer 仍为
  `INCONCLUSIVE`。
- D2：pairing/response 仍含 `NOT_EVALUABLE` 与 unresolved 项。
- D3：formal hypotheses 仍为 `INCONCLUSIVE`。
- D4：9 个 formal hypothesis 均为 `NOT_EVALUABLE`；R-HLC、R-TSB、R-IP 均为
  `INSUFFICIENT_FOR_RBR_DEVELOPMENT`。

## D4 的正确解释

D4 的不可评估来自既有历史资产没有逐项绑定的 exact pre-treatment
`Context_match` anchor，且冻结定义中的 family-specific mechanism 只有语义说明、
没有可执行的算法和阈值。它是
`BENCHMARK_ENABLEMENT / ASSET-CONTRACT LIMITATION`，不是 representation 的负面结果，
也不能改写为 failure 或 `SUPPORTED`。

因此，继续挖掘同一批历史资产不会改变 D4 的正式状态。后续工作仅可在独立的 R1
prospective benchmark enablement 协议下，先定义并获批 context、mechanism 和
controlled generator 合同；该工作不回溯修改 R0 Protocol v1.0 或历史 D4 结果。

## 训练授权

`r0_training_authorization_manifest_v1.0.json` 未修改：RBR-A、RBR-B、RBR-C 均为
`NOT_AUTHORIZED`。R1 Phase A 只是 prospective 设计，不能形成训练授权或 rollout 授权。

## 本报告的限制

本报告不新增实验、仿真、planner rollout、benchmark rollout、representation 评估或
RBR 训练。没有读取 embedding、BDD、probe 或 detection outcome 来形成上述结论。

# R0 Raw33 Provenance Addendum v0.1

## 1. 决策

```text
CURRENT_FILE_PROVENANCE_VERIFIED
HISTORICAL_LEDGER_ENTRY_NOT_AVAILABLE
```

本 addendum 非破坏性补记当前本地事实，不修改 Generation-1 历史 ledger，也不把 2026-08-25 发现时计算的 SHA 冒充为历史冻结 SHA。

机器清单：`docs/stageR/r0/manifests/r0_raw33_provenance_addendum_v0.1.csv`。

## 2. 当前文件事实

- 逻辑资产：`interaction_feat_style_raw.npy`；
- 实际文件数：36；
- 总行数：168700；
- 单行 shape：`[33]`；
- dtype：`float32`；
- 当前总 manifest：`outputs/stage6r_dynamic_full51_semantic_strict_v1/stage6r_dynamic_full51_manifest.json`；
- manifest SHA256：`c67391c26f70734ddecd38b95bcb81b744494383cf62bbfdeb4e024b1fe8305f`；
- manifest 声明 36 shards、168700 rows，split 为 train 135046、val 16870、historical-test 16784；
- 本地发现时间：`2026-08-25T07:27:04.593531+00:00`；
- 每个当前路径、SHA、part、shard、行数和 shape 见机器清单。

36 个当前文件 SHA 与本地只读核验结果一致，机器清单的 row count 合计也与 manifest 的 168700 一致。因此当前文件级 provenance 缺口已由 addendum 解决。

## 3. Builder 与 feature code

| 对象 | 路径 | SHA256 | Git 追溯 |
|---|---|---|---|
| Dynamic-v2 builder | `tools/build_waymo_dynamic_interaction_dataset_v2.py` | `1c0f8d77caf0b48a37fe47c673a4f9b293902fcbf1a58ada159f4572c90d1b79` | introducing commit `fa37948ce909ef83930fb34ef65342b912af93cb` |
| 33D feature implementation | `tools/interaction_context_features.py` | `ccc6c149f9fa4d9ce7ac541c300415c7c4cc0b43dcb0cee827141fc865ef7293` | 同一 commit 中的文件内容可复现该 SHA |

限制：历史 part manifest 记录 schema/artifact SHA，但没有把 builder/code SHA 作为生成时 artifact binding 写入。因此上述代码 SHA 的准确含义是：

```text
CURRENT_AND_GIT_TRACEABLE_NOT_ARTIFACT_BOUND_AT_GENERATION
```

不得将其写成“历史生成任务已经冻结并登记的 builder SHA”。

## 4. 历史 ledger 缺项

历史 ledger：`outputs/stage6r_dynamic_full51_semantic_strict_v1/stage6r_full51_sha256_ledger.json`，当前 SHA256 为 `e39e22354b11679eb66a43be2c658760ad627ac8b843452565225cecf8007d9e`。

该 ledger 登记了其他训练资产，但未登记 36 个 `interaction_feat_style_raw.npy`。本阶段没有修改此文件。每一机器清单行均固定：

```text
historical_ledger_status = HISTORICAL_LEDGER_ENTRY_NOT_AVAILABLE
historical_sha_claimed = false
```

这表示当前资产可按当前 SHA 用于 R0 development/provenance audit，但不能声称这些 SHA 在 Generation-1 当时已被 authoritative ledger 冻结。

## 5. 治理影响

- raw33 “文件存在但当前 SHA 未盘点”的问题已解决；
- raw33 “历史 ledger 当时是否登记”的事实仍为否，且不会被事后改写；
- R0 v1 freeze 可引用本 addendum 作为非破坏性 current-file provenance；
- 任何未来使用必须同时引用本 addendum、当前 manifest 和原历史 ledger 缺项状态；
- 本 addendum 不恢复 untouched holdout 身份，不授权 RBR 训练。

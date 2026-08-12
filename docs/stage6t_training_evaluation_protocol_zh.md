# Stage 6T A/B/C训练与盲测协议冻结

关联Issue：[GitHub #262](https://github.com/forwardxp-021/E2E-Evaluation/issues/262)。

## 1. 本阶段范围与结论

Stage6T只冻结训练和后续盲测协议，不启动训练、不写checkpoint、不读取Waymo test，也不运行或
读取nuPlan embedding、BDD/MMD和Stage6S-v2 confirmation outcome。

当前状态为：

`FROZEN_READY_FOR_ABC_TRAINER_IMPLEMENTATION_NOT_TRAINING`

这表示Dynamic Builder v2数据、Stage6O-v2训练准备度、Stage6S-v2 confirmation roster与A/B/C
实验设计已经通过训练前审计；下一项授权动作仅是实现并review统一trainer。它不表示A/B/C已经训练，
也不表示已经允许读取Waymo test或进入nuPlan正式盲测。

## 2. 为什么保留A/B/C三类candidate

三个candidate共享：

- 同一Dynamic Builder v2 full51数据与scenario级train/val/test split；
- 83D输入与64D输出接口；
- 相同optimizer、batch size、最大epoch、早停规则、三组seed和计算预算；
- Waymo train-only训练、Waymo val-only选epoch；
- 禁止用Waymo test、nuPlan、BDD/MMD、planner name或dose返工训练。

具体差异为：

| Candidate | 数据 | Encoder | sampling / dropout | objective | 可归因问题 |
|---|---|---|---|---|---|
| A | Dynamic v2 | 旧single-GRU 83→64 | 旧uniform、无dropout | 旧Stage5D objective | 修复训练数据本身是否已经改善old64 |
| B | Dynamic v2 | single-GRU 83→64，训练时划分long16/context48 | 新纵向采样、mask-aware dropout | clean longitudinal recovery v2 | 不改encoder拓扑，目标/采样能改善多少 |
| C | Dynamic v2 | 参数量匹配的ego16 + context48双分支 | 与B完全相同且共享随机draw | 与B完全相同 | 双分支架构相对B是否有额外价值 |

Candidate A与历史old64的差异除了数据，还包含Stage6T统一训练预算、seed和运行时，所以没有额外A0
时只能称为“dynamic-data-dominant comparison”，不能写成严格的纯数据因果效应。若论文以后需要严格
拆分数据贡献，应再训练可选A0：old builder数据 + 旧架构/objective + Stage6T共同seed/预算。

B与C的采样、dropout、objective、loss routing、seed和预算必须相同，且dropout使用相同随机draw；
两者唯一设计差异是encoder拓扑。B不是陪跑模型：若B通过全部门禁而C没有证明增量interaction信息，
应优先B，不能因为C更复杂就自动选择C。

## 3. 新发现的33D监督标准化问题

冻结审计发现，Dynamic v2由六个独立part构建，每个part的`feature_schema.json`相同，但
`interaction_feature_standardization.json`有六个不同SHA和六个局部train_count。这意味着各part的
`interaction_feat_style.npy`使用了各自局部train统计，不能把36个shard中的该数组直接混合训练，
否则相同33D目标在不同part中不在同一个坐标尺度。

Stage6T采用fail-closed修正，不覆盖或改写任何冻结shard：

1. A/B/C都只读取`interaction_feat_style_raw.npy`作为33D supervision的权威来源；
2. 使用全体Dynamic v2 train split的135046行一次性计算global population mean/std；
3. 同一组train统计原样应用到train/val/test；
4. Stage6T trainer禁止读取part-local的`interaction_feat_style.npy`；
5. 全局统计与每个训练输入文件SHA写入Stage6T freeze产物。

这不是根据模型结果调整方法，因为发现和修正规则都发生在第一个新checkpoint之前。

## 4. 模型和训练规则

### 4.1 架构

- A：GRU(83,128) + Linear(128,128) + ReLU + Linear(128,64)，encoder/projection参数量106560。
- B：与A同一single-GRU结构，训练loss routing把`z[0:16]`视作longitudinal view、`z[16:64]`
  视作context view；导出仍为一个64D embedding。
- C：ego 8D分支GRU hidden48输出16D；full context 83D分支GRU hidden120输出48D；拼接为64D。
  encoder/projection参数量105616，为B的0.9911倍。context branch保留完整ego与neighbor信息，不能称为
  neighbor-only branch。

参数量比较不含训练时auxiliary head；统一trainer必须让B/C使用同构、对应输出维度一致的训练head，
并在checkpoint中记录完整trainable parameter count。

### 4.2 优化预算

- Adam，learning rate=0.001，weight decay=0.0001；
- batch size=128，最多30 epochs，patience=5，min delta=0.0001；
- gradient clip norm=1.0，不使用mixed precision；
- seed=3407/3408/3409，primary seed预先固定为3407；
- 每seed最大31680 optimizer steps，即`ceil(135046/128) × 30`；
- Mac上使用MPS，若MPS不可用才使用CPU；每100 steps写JSONL heartbeat并显示epoch/validation进度。

### 4.3 checkpoint选择

每个candidate×seed独立用其冻结的Waymo val objective选择最早达到最低val loss的epoch。9个checkpoint
全部完成并锁定之前，禁止读取Waymo test。不能用test或nuPlan在三组seed中挑“最好seed”；后续nuPlan
primary始终使用预先固定的3407。

## 5. 四类成绩单和盲测顺序

固定顺序如下：

1. 仅用Waymo train/val锁定A/B/C全部9个checkpoint；
2. 一次性在相同Dynamic v2 Waymo test上评估old64和9个新checkpoint；
3. primary seed复用Stage6J/K既有rollout，做paired dose/task分析；
4. primary seed复用Stage6P的800 pair、489 log、2400 split，独立做A/A calibration与unpaired A/B；
5. 先运行Stage6S-v2的80-pair confirmation rollout与trajectory mechanism gate，不读取embedding；
6. 只有mechanism通过，才运行锁定的interaction embedding分析。

old64的历史Waymo test是旧builder分布，只能作历史参考；它必须在同一Dynamic v2 test行上重新评估一次，
才能作为A/B/C的非劣性参照。test结果只允许解释，不允许返工训练。

## 6. Candidate C成功标准

Candidate C必须同时满足：

- 三个seed完整训练并在test前锁定；
- Waymo primary seed通过纵向提升及following/lateral/behavior/retrieval非劣性，至少2/3 seed通过非劣性；
- Stage6J/K四个非零dose overall门禁、task×dose coverage、最小可检出剂量和剂量趋势门禁通过；
- Stage6P n=400独立A/A标定后FPR≤7.5%，context-balanced整体A/B检出率≥80%，两个方向各≥75%，
  raw方法检出率≥75%；
- Stage6S-v2先通过trajectory mechanism，之后C本身检出interaction差异；
- C full-context相对C neighbor-zero的增量使用“各representation独立null标准化Z_BDD之差”，并要求
  10000次log-cluster bootstrap 95%下界>0。

禁止跨representation比较raw MMD²，因此不能直接用`raw MMD²(C full) - raw MMD²(C zero)`作为
架构增量证据。C不要求击败ego13；ego13只用于检查纵向敏感性上限，不是最终模型。

## 7. 当前数据、盲态和环境审计结果

- Dynamic v2：36 shards、168700 rows、24548 scenario；train/val/test=135046/16870/16784；
- shape failure=0、raw33 nonfinite=0、scenario跨split重叠=0；
- Stage6R原始SHA ledger的36个shard训练资产全部匹配，mismatch=0；
- Stage6O-v1仍为`FROZEN_BLOCKED_WAYMO_COVERAGE_NOT_TRAINING`，SHA未变；
- Stage6O-v2全部数据门禁为true；
- Stage6S-v2仍为`CONFIRMATION_ROSTER_FROZEN_NOT_RUN`，80 pair，与development log/token重叠为0；
- A/B/C输出目录均不存在或为空，实际checkpoint数为0；
- `waymo_dev`为Python3.10.20、torch2.5.1，MPS built/available均为true。

## 8. 冻结产物

- 配置：`configs/stage6t_training_evaluation_protocol.json`；
- 冻结manifest：`outputs/stage6t_training_evaluation_protocol_freeze_v1/stage6t_training_evaluation_protocol_freeze_manifest.json`；
- A/B/C矩阵：`outputs/stage6t_training_evaluation_protocol_freeze_v1/stage6t_candidate_difference_matrix.csv`；
- 全局33D统计：`outputs/stage6t_training_evaluation_protocol_freeze_v1/stage6t_global_interaction_target_standardization.json`；
- 训练输入SHA：`outputs/stage6t_training_evaluation_protocol_freeze_v1/stage6t_training_input_sha256.json`；
- 中文运行报告：`outputs/stage6t_training_evaluation_protocol_freeze_v1/stage6t_training_evaluation_protocol_report_zh.md`。

冻结config SHA-256为`0d402d5ff845f5ab490f09295a60d043ef5c84e9ec0b865898285b03d9636570`，
包含输入与全局统计的content fingerprint为
`0f33a7939030397488dd53179fa7821e3b0c9a81721f01235a380ee058f5b1da`。

## 9. 下一步边界

下一步是按冻结协议实现统一A/B/C trainer和最小synthetic smoke，验证：

- A精确复现旧objective semantics，但使用全局raw33标准化；
- B/C共享相同batch、pair、dropout随机draw和loss实现；
- C/B参数量、输出shape与梯度都满足冻结约束；
- checkpoint metadata写入protocol fingerprint，已存在seed目录时fail closed；
- trainer没有Waymo test、nuPlan、BDD/MMD入口。

trainer实现与review通过后，仍需单独授权才能启动9个正式训练任务。

# Stage 6O：纵向敏感 64D Behavior Embedding 训练前冻结协议

## 1. 当前结论

Stage 6O v1 已完成训练协议和现有 Waymo 数据的只读冻结审计，但当前状态为：

`FROZEN_BLOCKED_WAYMO_COVERAGE_NOT_TRAINING`

这表示：

- 模型方向、数据规则、损失权重、随机种子、训练预算和验收门槛已经在新模型结果出现前冻结；
- 当前数据的完整性、shape、finite、split 和防泄漏检查全部通过；
- 当前 Waymo 数据没有 intermittent-following 窗口，未通过预冻结覆盖门槛；
- 因此不启动训练，不写入新 checkpoint，也不覆盖 Stage5D-balanced-v2 基线。

阻塞不是程序错误。它是训练前审计识别出的真实数据覆盖缺口，不能通过事后降低门槛消除。

## 2. 为什么要建立 Stage 6O

Stage 6L 的同场景配对消融显示：

- 当前完整 64D 的 median overall `Z_BDD=7.539`，task×dose Holm 通过 `7/12`；
- 同 checkpoint 邻车置零 64D 为 `11.066`、`11/12`；
- 显式 ego 运动学 13D 为 `21.082`、`12/12`。

Stage 6M 在 400 场景/版本时，context-balanced 非配对检出率为 `66.5%`，相对 raw 的
`63.0%` 只提高 3.5 个百分点，且 McNemar `p=0.2478`。因此当前首要瓶颈是纵向
representation sensitivity，而不是单纯的场景组成不平衡。

Stage 6O 的目标不是把 raw BDD 数字人为放大，而是训练一个在 Waymo 域内不退化、同时对
nuPlan 中经运动学确认的纵向风格剂量具有更稳定 null-standardized sensitivity 的新模型。

## 3. 权威输入与只读边界

### 3.1 Waymo 数据

- 数据集：`waymo_5neighbor_context_laneaware_clean_v1_full51_merged`
- shard manifest SHA-256：
  `12e37bdf5317690639d0a0ac01c7f72b65578658623f2149394e54bb5223451f`
- feature schema SHA-256：
  `b080287935ef0494e5a51685931361d78b2db76cd17b14fddfd42bfbf22902b5`
- 特征标准化只使用 train split，train count 为 `131998`。

### 3.2 基线模型

- family：`context_gru_stage5d_balanced_v2`
- checkpoint SHA-256：
  `909022f5df03a3f01c2149da6c9b44c613e955a4d816e8ec4d5862f39f8bf0cc`
- 基线始终只读；Stage 6O 即使通过，也只能新增候选，不能删除基线。

### 3.3 nuPlan 冻结证据

- Stage 6L repaired summary：
  `d93cf0b902015b857d1a6fbe3e8385d54d5a9f7d51389b5cb7e5d9bf5ff0136d`
- Stage 6M summary：
  `5a440e85d803c135836a0a6a88580b9ba6c38e3f7f93f1efd338bf442f036b09`
- repaired realized dose summary：
  `aa6999b7c03460063bf5b80ba2aa3d4712ff48452c6f2b38e20a35894a3adb1d`

## 4. Waymo 数据规模与质量审计

审计逐 shard 使用 mmap/顺序读取，没有合并 `context_traj.npy` 等大数组。

| 项目 | 结果 |
|---|---:|
| shard 数 | 35 |
| 总窗口数 | 164,871 |
| train | 131,998（80.061%） |
| validation | 16,481（9.996%） |
| test | 16,392（9.942%） |
| unique scenario | 24,426 |
| unique scenario-agent | 164,871 |
| scenario 跨 split 重叠 | 0 |
| scenario-agent 跨 split 重叠 | 0 |
| 重复 scenario-agent-start | 0 |
| context/feature 非有限值 | 0 |
| good lane context rate | 98.999% |
| lane assignment success | 100% |
| fallback assignment | 0% |

所有场景必须继续使用以下固定分割：

```text
h = int(md5(scenario_id)[:8], 16) / 0xffffffff
train: h < 0.8
val:   0.8 <= h < 0.9
test:  h >= 0.9
```

同一 scenario 的所有 agent 和窗口必须属于同一个 split。任何 scenario 或
scenario-agent 跨 split 都直接 fail closed。

## 5. 纵向覆盖审计

### 5.1 速度覆盖

train split：

| 速度档 | 定义 | 窗口数 |
|---|---|---:|
| low | `[0,5)` m/s | 52,098 |
| medium | `[5,15)` m/s | 66,676 |
| high | `>=15` m/s | 13,224 |

三档都超过每档 5,000 条的门槛。

### 5.2 运动状态覆盖

train split：

- stop/go proxy（min speed `<1` 且 max speed `>=5`）：39,949；
- low-speed variable：49,534；
- steady-speed（窗口 speed std `<=0.5`）：10,242；
- dynamic-speed：121,756。

stop/go 和 steady-speed 均超过 5,000 条门槛。

### 5.3 跟车覆盖与当前阻塞

train split：

| 跟车档 | 窗口数 |
|---|---:|
| free-flow，front valid ratio = 0 | 96,649 |
| intermittent-following，`0 < ratio < 0.5` | 0 |
| sustained-following，ratio `>=0.5` | 35,349 |

进一步检查发现所有非零 front valid ratio 的最小值都是 `0.8`。这与当前构建器的
neighbor valid 筛选逻辑一致：现有数据主要形成“整窗无前车”和“至少 80% 帧有前车”两类，
没有前车进入、离开、短时遮挡或边界跟车窗口。

因此，虽然当前数据总量大且基础质量通过，但不能宣称已覆盖完整的纵向 interaction
边界。Stage 6O v1 保持阻塞，不允许直接在这份数据上开始新训练。

### 5.4 旧纵向特征噪声

现有 33D raw supervision 中：

- RMS acceleration train median 约 `2.72 m/s²`；
- RMS jerk train median 约 `42.82 m/s³`；
- RMS jerk q90 约 `100.80 m/s³`。

这些量级表明未平滑差分对测量噪声非常敏感。新模型不得直接以这些 raw tail 作为唯一纵向
监督。协议冻结为：

1. 从 ego speed channel 5 计算纵向目标；
2. 使用 5 帧居中 median filter，边界复制；
3. 以 `dt=0.1s` 计算 acceleration 和 jerk；
4. 每个目标仅用 train q01/q99 winsorize；
5. 再用 train median/IQR 标准化；
6. train 的统计量原样应用到 val/test/nuPlan。

## 6. train/validation/test 与防泄漏规则

冻结规则如下：

1. split unit 是 `scenario_id`，不是窗口或 agent；
2. 标准化、winsorization、采样权重和 pair distance 阈值只用 train 拟合；
3. hard-negative 和 near-boundary pair 只能在同一 split 内构建；
4. validation 只用于 early stopping 和冻结的模型选择目标；
5. test 只在所有结构、权重、seed 和 checkpoint 选择规则固定后使用一次；
6. nuPlan 不得参与 Waymo 训练、采样、loss 权重、选 epoch 或选 seed；
7. planner 名称、scenario token、dose、BDD、MMD 和未来评估结果不得进入训练输入或采样。

## 7. 样本与困难负样本策略

基础 window sampler 使用：

```text
stratum = speed_bin × front_regime × lateral_nuisance_bin
weight = inverse_sqrt(train stratum frequency)
weight = clip(weight, 0.25, 4.0)
```

每个 epoch 固定采样 131,998 个窗口，有放回抽样。ranking pair 组成固定为：

- 50% context-matched hard negatives；
- 25% near-boundary pairs；
- 25% uniform pairs。

hard negative 必须属于相同 nuisance stratum，纵向目标距离不低于 train q75、nuisance
distance 不高于 train q25。near-boundary 的纵向距离位于 train q40–q60。tie 使用
`stable_hash(scenario_id,target_agent_id,start,pair_seed)`，不得人工挑选。

在 intermittent-following 数据补齐前，这套 pair pool 不完整，因此不生成正式训练 pair。

## 8. 冻结模型结构

接口保持：

```text
input:  [B,T,83]
output: [B,64]
```

内部 64D 固定为：

```text
z64 = concat(z_ego_longitudinal_16d, z_context_fusion_48d)
```

- ego branch 读取 0:8 自车通道；
- context/fusion branch 保留全部 83D 和 5-slot mask；
- 不删除邻车上下文；
- slot dropout 概率 0.15；
- 全邻车 dropout 概率 0.05；
- ego dropout 为 0；
- dropout 只用于训练鲁棒性，不改变测试输入。

该结构不是把 embedding 简化为速度分数。16D 子空间保证纵向信息有明确容量，48D 仍保留
跟车、横向、gap 和 interaction 表征。

## 9. 损失权重

| 目标 | 权重 |
|---|---:|
| global style soft contrastive | 1.00 |
| ego longitudinal auxiliary Huber | 1.00 |
| ego longitudinal metric alignment | 0.50 |
| ego longitudinal pair ranking | 0.50 |
| following interaction auxiliary | 0.75 |
| lateral dynamics auxiliary | 0.75 |
| lateral gap auxiliary | 0.50 |
| behavior proxy auxiliary | 0.25 |
| neighbor dropout consistency | 0.25 |

ranking margin 固定为 0.2。raw BDD 和任何 nuPlan 指标的训练权重固定为 0。

## 10. 随机种子与训练预算

- seeds：`3407 / 3408 / 3409`；
- primary seed：`3407`；
- pair seed：`93407`；
- optimizer：Adam；
- learning rate：`1e-3`；
- weight decay：`1e-4`；
- batch size：128；
- max epochs：30；
- early stopping patience：5；
- min delta：`1e-4`；
- gradient clip norm：1.0；
- 每 seed 最大 optimizer step：31,000；
- 每 seed 最大 wall-clock：24 小时；
- 设备策略：MPS 可用则 MPS，否则 CPU；
- 每 100 step 写入 JSONL heartbeat，并保留 epoch 进度条。

任何预算变化都必须创建新协议版本，不能根据首轮 nuPlan 结果延长训练。

## 11. checkpoint 命名与证据链

family 固定为：

```text
context_gru_stage6o_ego16_context48_v1
```

目录模板：

```text
outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/
  context_gru_stage6o_ego16_context48_v1/
    seed_3407/best_model.pt
    seed_3408/best_model.pt
    seed_3409/best_model.pt
```

checkpoint 必须记录 protocol、freeze/config/data hash、seed、epoch、64D/83D/16D 维度、
git commit 和 validation metrics。已有 seed 目录默认禁止覆盖。

## 12. Waymo 域内验收门槛

候选相对冻结基线必须满足：

| 指标 | 基线 | 候选门槛 |
|---|---:|---:|
| longitudinal mean Spearman | 0.15542 | >= 0.18542 |
| following mean Spearman | 0.49631 | >= 0.47631 |
| lateral mean Spearman | 0.25660 | >= 0.23660 |
| behavior proxy mean Spearman | 0.31685 | >= 0.29685 |
| retrieval hit@5 | 0.50756 | >= 0.48756 |
| mean neighbor feature distance | 1.92786 | <= 2.02425 |

primary seed 必须全部通过，且 3 个 seed 中至少 2 个通过全部非劣性门槛。所有差异报告配对
bootstrap 95% CI；纵向提升的 CI 下界必须大于 0。

## 13. nuPlan 验收门槛

只有 Waymo 候选锁定后才能运行 nuPlan。

### 13.1 配对剂量与任务覆盖

- 四个剂量 overall Holm 全部通过；
- task×dose 至少 `10/12` 通过；
- 最小 overall 检出剂量不高于 25%；
- median overall `Z_BDD >= 11.0`；
- nominal dose 与 `Z_BDD` 的 Spearman `>=0.8`；
- realized speed/acceleration/jerk 保持单调。

这里使用各自 null 标准化后的结论，不能直接比较跨模型 raw MMD²。

### 13.2 异场景非配对发布检验，n=400/版本

- primary method：context-balanced；
- A/A false-positive rate `<=7.5%`；
- A/B detection `>=80%`；
- 两个 A/B 方向分别 `>=75%`；
- raw method A/B detection `>=75%`；
- 每个方法必须单独做 A/A calibration。

## 14. 基线替换标准

默认决策永远是：

`KEEP_STAGE5D_BALANCED_V2_BASELINE`

只有以下条件全部成立，才可以人工批准：

`PROMOTE_STAGE6O_CANDIDATE_WITH_BASELINE_RETAINED`

1. 数据完整性和防泄漏门槛全部通过；
2. Waymo 域内门槛全部通过；
3. nuPlan paired dose/task 门槛全部通过；
4. nuPlan unpaired release 门槛全部通过；
5. seed 稳定性通过；
6. 人工复核证据链和论文表述。

部分通过只能标记为 `RETAIN_AS_RESEARCH_ABLATION_ONLY`。结果出现后不得修改本协议门槛。

## 15. 解除当前阻塞所需的数据工作

在训练前必须建立单独版本的 Waymo context dataset，至少做到：

1. 扩展当前 full51 之外的 Waymo scenario 文件，而不是只复制高度重叠窗口；
2. neighbor valid threshold 与 ego valid threshold 分离；
3. 支持在窗口中途出现/离开的前车，不能只依赖窗口起点候选；
4. 对 temporal slot reassignment 或 track-continuity 规则做显式审计；
5. intermittent-following train 窗口达到至少 5,000；
6. low/medium/high、free/intermittent/sustained、steady/dynamic、lateral/no-lateral 各层重新审计；
7. 新数据使用新 manifest、split audit 和 SHA-256，不覆盖当前 full51 数据；
8. 新数据通过 Stage 6O freeze 后，才允许实现并运行训练器。

## 16. 当前权威产物

- 配置：`configs/stage6o_longitudinal_representation_training_protocol.json`
- 冻结工具：`tools/stage6o_freeze_longitudinal_training_protocol.py`
- 冻结目录：`outputs/stage6o_longitudinal_training_protocol_freeze_v1/`
- 数据审计：`stage6o_waymo_data_audit.json`
- 冻结 manifest：`stage6o_training_protocol_freeze_manifest.json`
- 中文冻结报告：`stage6o_training_protocol_report_zh.md`
- GitHub Issue：[#256](https://github.com/forwardxp-021/E2E-Evaluation/issues/256)

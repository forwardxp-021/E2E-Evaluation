# 00_plans — E2E 行为风格评估论文路线与进展

> 更新时间：2026-05-19  
> 当前分支：`20260514_interaction_design`  
> 项目定位：博士论文 / 论文实验工程  
> 当前主线：从 ego-only behavior embedding 升级到 lane-aware 5-neighbor interaction-aware behavior embedding，并进入 Stage 5C evaluation。

---

## 1. 一句话总结当前路线

本项目要建立一套 **trajectory-level behavior evaluation benchmark**，用于评价自动驾驶 E2E / planning policy 的驾驶行为风格，而不是只依赖 ADE/FDE、collision rate、规则触发率等传统指标。

核心思想是：

> 把一段驾驶行为编码成 behavior embedding，再用 embedding distance、retrieval、style-distance correlation、BDD / style drift 等指标，评价不同模型版本或不同 policy 的驾驶风格差异。

截至目前，项目路线已经从最初的 synthetic policy validation，推进到 public human trajectory validation，再升级到 interaction-aware context embedding：

```text
Stage 1-3: synthetic controlled policy validation
  ↓
Stage 4: public human trajectory / ego-only behavior embedding
  ↓
Stage 5A: lane-aware 5-neighbor context dataset construction
  ↓
Stage 5B: Flatten Context GRU interaction-aware embedding training
  ↓
Stage 5C: context-aware embedding evaluation（当前下一步）
```

当前最重要结论：

```text
Stage 5A 数据构建完成；
Stage 5B context-aware embedding 已训练完成并导出；
现在正式进入 Stage 5C evaluation。
```

---

## 2. 研究背景与最初动机

最初的问题来自自动驾驶端到端模型开发中的真实观察：

- E2E 决策 / 规划模型越来越像“驾驶员”；
- 模型版本之间会出现明显驾驶风格变化；
- 有的版本更激进，有的版本更保守，有的版本更关注舒适性；
- 传统评价指标很难描述“风格变化”；
- 现实中评价人类司机时，并不会要求司机持续解释内部决策逻辑，而是看其实际驾驶行为。

因此，本研究的核心想法是：

> 对 E2E 自动驾驶系统，应该增加一套类似评价人类驾驶员的行为风格评价体系。

这个体系重点不是解释神经网络内部，而是评价它输出的 trajectory-level behavior。

---

## 3. 当前论文定位

建议论文定位为：

> **A trajectory-level behavior evaluation benchmark for interaction-aware autonomous driving behavior representation.**

也就是：

- 输入：trajectory window / policy rollout / human trajectory；
- 输出：behavior embedding / style feature / report card；
- 指标：retrieval、style-distance correlation、style drift、BDD、comfort / aggressiveness / interaction fingerprint；
- 适用对象：synthetic policy、rule-based policy、learning-based planner、E2E planner，只要能输出轨迹即可。

重要边界：

- 不做 sensor rendering；
- 不做 perception stack；
- 不要求实车私有数据作为第一阶段前提；
- 不声称等价于完整自动驾驶闭环仿真；
- 当前主要关注 planning / behavior trajectory。

论文中可以使用这样的表述：

> The benchmark is model-agnostic and accepts any trajectory-level rollout or trajectory window, including learned E2E planners, rule-based policies, synthetic policies, and public human trajectory data.

---

## 4. 核心研究问题

### Q1. Behavior embedding 是否能区分不同驾驶风格？

需要证明：

- 不同行为在 embedding 空间中有可分性；
- embedding 不是随机的；
- retrieval 能找回相似行为样本；
- embedding distance 与 style feature delta 有稳定相关性。

### Q2. Ego-only embedding 是否足够？

Stage 4 的经验表明：

- ego trajectory 可以学到 speed / jerk / yaw / comfort 等一部分风格；
- 但跟车、让行、压迫感、变道交互等行为不能只靠 ego trajectory 完整表达；
- 周围车辆，尤其是前车和左右相邻车道前后车，是驾驶风格的重要上下文。

因此进入 Stage 5：加入 lane-aware 5-neighbor context。

### Q3. 加入 5-neighbor context 后，embedding 是否更 interaction-aware？

这是 Stage 5C 当前最核心的问题。

需要证明 Stage 5B context-aware embedding 在以下方面优于或至少补充 Stage 4 ego-only：

- THW / front distance；
- relative speed；
- front slot occupied；
- left/right adjacent slot occupied；
- lateral interaction proxy；
- comfort / jerk / yaw / curvature 不塌缩。

### Q4. 如何避免 synthetic generator artifact 风险？

早期 synthetic policy validation 的最大审稿风险是：

> embedding 可能只是在识别 synthetic generator artifact。

因此 Stage 4 / Stage 5 改为使用 Waymo public human trajectories，并用 weak supervision / pseudo style / interaction features 验证真实轨迹上的行为结构。

---

## 5. 阶段总览

| 阶段 | 目标 | 当前状态 |
|---|---|---|
| Stage 1 | PR2 interpretability demo | 已完成 |
| Stage 2 | population-level synthetic policy evaluation | 已完成 |
| Stage 3 | generator ablation + local fine sweep | 已完成，形成 recommended_lateral_stable_v2 |
| Stage 4 | public human trajectory ego-only validation | 已完成主要链路，结论推动 Stage 5 |
| Stage 5A | lane-aware 5-neighbor context dataset | 已完成 full51 构建与 merge |
| Stage 5B | Flatten Context GRU context-aware embedding | 已完成训练与 embedding 导出 |
| Stage 5C | context-aware embedding evaluation | 当前下一步 |

---

## 6. Stage 1-3 简要回顾：synthetic controlled validation

Stage 1-3 的目标是先在可控 synthetic policy 环境中验证 behavior embedding 的基本有效性。

### 6.1 三类 synthetic policy

| policy_id | policy_name | 含义 |
|---|---|---|
| p0 | conservative | 保守型 policy |
| p1 | aggressive | 激进型 policy |
| p2 | lateral_stable | 横向稳定 / 舒适型 policy |

### 6.2 核心概念

#### source / source window

```text
scenario_id + start + window_len + front_id
```

代表同一段场景上下文。

#### within-source

同一 source 下比较 p0 / p1 / p2，控制场景变量，让差异主要来自 policy。

#### p2_farthest_rate

```text
p2_farthest = true if d(p0,p2) > d(p0,p1) and d(p1,p2) > d(p0,p1)
```

#### p2 separation margin

```text
p2_separation_margin = min(d(p0,p2), d(p1,p2)) - d(p0,p1)
```

解释：

- margin > 0：p2 比 p0-p1 之间还远，说明 p2 有强独立性；
- margin < 0：p2 仍更接近 p0 或 p1，独立性不完全。

### 6.3 Stage 1-2 结论

Stage 1-2 已经证明：

- embedding 能区分 synthetic policy；
- centroid classification 明显高于 chance；
- global retrieval 能找回同 policy / 相似行为样本；
- within-source aligned evaluation 对控制场景变量很有价值。

但同时发现：

> 原始 lateral_stable 不足以成为完全独立第三类。

### 6.4 Stage 3 ablation 结论

Stage 3 通过 broad ablation 和 local fine sweep，最终得到推荐配置：

```text
recommended_lateral_stable_v2
```

关键参数：

```text
heading_smooth_alpha = 0.75
yaw_rate_clip = 0.008
thw_target = 1.70
jerk_limit = 0.200
a_max = 1.275
a_min = -2.52
```

它相比 baseline_current 和 full_strong_lateral_stable：

- 提升 p2_farthest_rate；
- 改善 mean_p2_separation_margin；
- 提升 centroid_accuracy_p2；
- 提升 retrieval same-policy fraction；
- 降低 p2_rms_jerk；
- 降低 p2_rms_yaw_rate_proxy；
- 降低 curvature proxy；
- 基本保持 THW。

谨慎结论：

```text
p2 independence is improved but incomplete.
```

原因：

```text
mean_p2_separation_margin 仍然为负。
```

---

## 7. Stage 4：public human trajectory ego-only validation

Stage 4 的目标是从 synthetic policy 转向公开真实人类轨迹，降低 generator artifact 风险。

### 7.1 Stage 4 初始目标

最初定义为：

```text
输入统一格式 human trajectory arrays
↓
计算 / 加载 style features
↓
构造 pseudo style labels
↓
评估 embedding / baselines
↓
输出 validation report
```

统一格式包括：

```text
traj.npy
front.npy
meta.npy
split.npy
feat_style.npy
feat_style_raw.npy
feature_names_style.json
```

### 7.2 关于 data1 的误解与澄清

当时曾经误以为 `data1` 是真实公开 human trajectory 数据。

后来澄清：

```text
data1 不是新的公开真实 human trajectory 数据；
它主要是之前 ablation / synthetic pipeline 里基于 Waymo 提炼出来的数据；
Stage 4 软件版本一开始更多是统一格式和验证协议，而不是已经完成真正 public human validation。
```

这个澄清推动我们重新构建真正的 Waymo public human trajectory 数据。

### 7.3 Stage 4B：Waymo human trajectory dataset 构建

工具：

```text
tools/build_waymo_human_trajectory_dataset.py
```

最小 smoke 输出：

```text
n_windows_kept = 36
feature_dim = 16
```

small 输出：

```text
n_windows_kept = 260
split_counts = train 194 / val 27 / test 39
front_found_rate ≈ 0.9538
```

随后扩大到 full51。

### 7.4 Stage 4 full51 数据结果

Waymo human public full51 数据构建结果：

```text
out_dir = outputs/waymo_human_v1_full51
n_files_processed = 51
n_scenarios_processed = 24872
n_agents_considered = 1127346
n_windows_total = 1127346
n_windows_kept = 168191
n_front_found = 161682
front_found_rate ≈ 0.9613
split_counts:
  train = 134637
  val   = 16823
  test  = 16731
```

feature_names 共 16 个：

```text
mean_speed
std_speed
rms_accel
rms_jerk
rms_yaw_rate_proxy
rms_curvature_proxy
mean_thw
min_thw
mean_front_distance
min_front_distance
mean_rel_speed
std_rel_speed
max_abs_accel
max_abs_jerk
heading_change_total
valid_ratio
```

### 7.5 Stage 4C：pseudo style labels

pseudo label 规则使用 percentile 模式，target_quantile=0.25。

full51 pseudo label 结果：

```text
n_total = 168191
n_labeled = 75421
n_unlabeled = 92770
label_counts:
  conservative_like = 34416
  aggressive_like   = 33662
  lateral_stable_like = 7343
  unlabeled = 92770
```

这些 label 是 weak labels，不是 ground truth。

注意：

```text
pseudo labels 是由 style features 构造的；
因此 classification / retrieval 指标需要与 baseline、strict retrieval、feature correlation 一起解释，避免 feature leakage 审稿质疑。
```

### 7.6 Stage 4D-G：ego-only embedding 训练与改进

Stage 4 输入：

```text
traj.npy: [N, 80, 4]
feat_style.npy: [N, 16]
```

模型：

```text
ego trajectory
↓
GRU
↓
64-D embedding
```

训练目标逐步演化：

| 版本 | 思路 | 说明 |
|---|---|---|
| Stage 4D | ego-only baseline | soft contrastive learning |
| Stage 4E | jerk / comfort feature weighting | 重点增强 comfort / jerk |
| Stage 4F | comfort-aware auxiliary regression | 显式预测 comfort targets |
| Stage 4G | comfort metric alignment | 对齐 embedding distance 与 comfort feature distance |
| Stage 4H | shuffled sanity check | 验证提升不是随机 target 造成 |
| Stage 4I | comparison / reporting | 汇总训练与评估结果 |

过程中遇到并修复了：

- traj.npy 中存在 NaN；
- export embedding 时 normalize_local 遇到 non-finite；
- sanitizer / trajectory_preprocessing 需要统一；
- evaluate_aux_predictions.py 初期没有正确纳入主流程；
- paper table 一度误用 4D 而不是 4E/4F/4G 的结果；
- README / QUICK_REFERENCE 需要持续同步。

### 7.7 Stage 4 的核心结论

Stage 4 证明：

```text
ego-only human behavior embedding 可以在真实 Waymo human trajectory 上稳定训练；
comfort / jerk / yaw / curvature 等弱监督信号可以塑造 embedding；
metric alignment 比单纯 soft contrastive 更适合提升 jerk / comfort distance correlation。
```

但 Stage 4 也暴露出局限：

```text
ego-only trajectory 不能完整表达跟车、邻车压迫、让行、变道交互等 interaction style。
```

因此进入 Stage 5。

---

## 8. 从 Stage 4 到 Stage 5 的关键思想转变

### 8.1 旧 rel_kinematics 与新 Stage 5 输入的区别

之前曾讨论 `train_embedding.py` 中的 rel_kinematics 思路。它强调从 ego 与周围对象的相对运动中提取信息。

Stage 5 的核心变化是：

```text
不再只把 interaction 信息压缩成 handcrafted weak features；
而是把 lane-aware 5-neighbor relative trajectory / relative kinematics 作为模型输入。
```

### 8.2 模型输入与弱监督 feature 的区别

需要严格区分：

#### 模型输入

```text
ego trajectory + 5-neighbor ego-centric relative trajectory / kinematics
```

模型需要从时序上下文中学习驾驶行为表示。

#### 弱监督 feature

```text
speed / accel / jerk / yaw_rate / curvature / THW / front_distance / rel_speed / slot occupancy 等统计特征
```

这些用于指导 embedding 空间，但不是模型唯一输入。

换句话说：

```text
relative trajectory 是原始时序输入；
style features 是训练信号 / evaluation target。
```

### 8.3 为什么不是输入 neighbor absolute trajectory？

不直接输入全局 absolute neighbor trajectory，原因是：

- 全局坐标原点无意义；
- road heading / map orientation 会引入不必要 variation；
- ego-centric representation 更符合驾驶决策；
- relative kinematics 更能表达“对我有什么影响”。

因此 Stage 5 输入采用：

```text
neighbor absolute state
↓
ego-centric transform
↓
relative trajectory / relative kinematics
```

---

## 9. Stage 5 输入设计：lane-aware 5-neighbor context

### 9.1 5 个关键邻车 slot

最终确定 5 个邻车：

```text
front
left_front
left_rear
right_front
right_rear
```

理由：

- front：跟车、THW、TTC、压迫感、舒适性核心；
- left_front / left_rear：左变道可行性、让行、被压迫感；
- right_front / right_rear：右变道可行性、让行、被压迫感。

### 9.2 为什么必须 lane-aware assignment

曾经尝试 geometric fallback，但最终认为：

> neighbor car 定义如果不干净，会直接污染后面的 interaction embedding。

因此 Stage 5A 选择一开始就做 lane-aware assignment。

### 9.3 最终 slot eligibility 规则

#### front

```text
candidate lane == ego current lane
0 < delta_s <= 120m
abs(candidate_l) <= 2.0m
heading_diff <= 45°
当前帧有效
允许静止车
```

#### left_front / right_front

```text
candidate lane == left/right adjacent lane
0 < delta_s <= 80m
abs(candidate_l) <= 2.0m
heading_diff <= 45°
```

#### left_rear / right_rear

```text
candidate lane == left/right adjacent lane
-120m <= delta_s < 0
abs(candidate_l) <= 2.0m
heading_diff <= 45°
```

选择规则：

```text
1. abs(delta_s) 最小
2. projection_distance 最小
3. abs(candidate_l) 最小
4. heading_diff 最小
```

### 9.4 静止前车处理

最终结论：

```text
静止前车不直接排除。
```

原因：

- 红灯排队；
- 拥堵；
- 前车停车；
- 静止前车对跟车风格、刹车风格、舒适性有强影响。

但必须标记：

```text
neighbor_is_static
static_front_count
static_front_ratio
```

### 9.5 路口与 lane_context_quality

不是因为“路口”就删除，而是根据 lane context 质量判断：

```text
good
ambiguous_intersection
fallback
bad
```

重要修正：

```text
empty slot 是正常交通稀疏现象，不等于 ambiguous_intersection。
```

最开始出现过一个 bug：只要任一 slot empty，就把样本标成 ambiguous_intersection，导致 ambiguous rate 接近 99%。后来修正为：

```text
lane_context_quality 衡量 lane/map 语义可靠性；
slot coverage / empty slot ratio 单独统计。
```

---

## 10. Stage 5A：lane-aware 5-neighbor context dataset 构建

工具：

```text
tools/build_waymo_5neighbor_context_dataset.py
```

### 10.1 输出文件

每个 shard 包含：

```text
ego_seq.npy
neighbor_seq.npy
context_traj.npy
context_mask.npy
context_mask_window.npy
neighbor_slot_ids.npy
meta.npy
split.npy
interaction_feat_style_raw.npy
interaction_feat_style.npy
lane_assignment_debug.csv
shard_summary.json
```

合并目录包含：

```text
shard_manifest.json
build_summary.json
merged_build_summary.json
neighbor_context_summary.json
interaction_feature_standardization.json
build_report.md
```

### 10.2 Stage 5A 过程中修复的重要工程问题

Stage 5A 遇到并修复了大量问题：

```text
1. trajectory NaN / Inf 清洗问题；
2. project_point_to_lane 卡住，需要空间预筛 / top-k candidate；
3. tools import path 问题；
4. timing 初始化位置错误；
5. clean filtering 后 row alignment mismatch；
6. lane_assignment_debug.csv 字段不统一；
7. assignment_method_counts_by_slot 总数不等于 n_windows_kept；
8. lane_context_quality 把 empty slot 误判为 ambiguous；
9. full51 非 streaming 版本 OOM；
10. Codex 引入 /tmp/old.py 临时依赖；
11. SlotAssignResult 接口不一致；
12. streaming 版本缺 inner progress；
13. streaming summary 硬编码 split_counts = {}；
14. merge 脚本 shard path 拼接错误；
15. assignment_method_counts_by_slot merge 后全 0；
16. full51 需要 4 进程 file_start/file_end 分片并行。
```

最终工程原则：

```text
不把 full51 所有 scenario 一次性读入内存；
不拼接大型 npy；
使用 streaming + sharded output；
用 manifest + global standardization 合并；
大型数据读取优先 mmap。
```

### 10.3 full51 并行分片构建

由于单进程 CPU/GPU 利用率不高，最终采用 4 个外部进程按 TFRecord file range 分片并行：

```text
part_00_13
part_13_26
part_26_39
part_39_51
```

不在代码内部做 multiprocessing，原因是：

- TensorFlow Dataset 多进程安全复杂；
- shard 写入锁复杂；
- counter 聚合复杂；
- Codex 当时不稳定，大改风险高。

### 10.4 Stage 5A final merged 结果

最终合并目录：

```text
outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged
```

最终数据规模：

```text
n_windows_kept = 164871
n_shards = 35
split_counts:
  train = 131998
  val   = 16481
  test  = 16392
```

质量指标：

```text
nonfinite_output_detected = 0
lane_assignment_success_rate = 1.0
fallback_assignment_rate = 0.0
good_lane_context_rate = 0.9899861103529426
ambiguous_intersection_rate = 0.010013889647057397
```

slot occupied window ratio：

```text
front       = 0.2670390790
left_front  = 0.1412619563
left_rear   = 0.1514092836
right_front = 0.1580144477
right_rear  = 0.1589545766
```

slot valid frame ratio：

```text
front       = 0.2640142141
left_front  = 0.1396579750
left_rear   = 0.1501427631
right_front = 0.1562976509
right_rear  = 0.1577124691
```

empty slot ratio：

```text
front       = 0.7329609210
left_front  = 0.8587380437
left_rear   = 0.8485907164
right_front = 0.8419855523
right_rear  = 0.8410454234
```

解释：

```text
lane-aware 严格定义后，neighbor slot 是稀疏的；
但这种稀疏性是正常交通现象，不是数据错误；
训练和评估必须使用 context_mask / context_mask_window。
```

### 10.5 global interaction feature standardization

合并后重新计算全局标准化，使用所有 shard 的 train split：

```text
train_count = 131998
feature_dim = 33
```

重要原则：

```text
不能使用每个 part 自己的局部 mean/std；
必须使用 full51 train split 全局 statistics。
```

---

## 11. Stage 5B：Flatten Context GRU context-aware embedding

Stage 5B 是第一版 interaction-aware behavior embedding。

### 11.1 输入

```text
context_traj.npy: [N, 80, 83]
context_mask.npy: [N, 80, 5]
context_mask_window.npy: [N, 5]
interaction_feat_style.npy: [N, 33]
```

其中：

```text
context_dim = 83
feature_dim = 33
```

### 11.2 模型

```text
context_seq [B, 80, 83]
↓
GRU hidden_dim=128
↓
MLP projection
↓
64-D embedding
```

脚本：

```text
tools/context_shard_dataset.py
tools/train_context_behavior_embedding.py
tools/export_context_row_embeddings.py
```

### 11.3 训练目标

Stage 5B v1 使用：

```text
soft contrastive loss
+ metric alignment loss
```

其中 interaction_feat_style.npy 作为 weak supervision。

### 11.4 训练性能问题与解决

训练时观察到：

```text
batch_size 128 / 256 会被 Killed；
batch_size 64 可以跑；
GPU 算力利用率低；
CPU / 系统内存压力高。
```

结论：

```text
这不是 GPU 没用上，而是 dataloader / CPU / RAM bottleneck。
```

修复：

```text
np.load(..., mmap_mode="r")
cache_shards = 1
num_workers = 2
pin_memory = true
persistent_workers = true
```

### 11.5 正式训练配置

```text
batch_size = 64
epochs = 20
lr = 1e-3
temperature = 0.1
feature_temperature = 1.0
metric_alignment = true
metric_loss_weight = 0.1
metric_loss_type = huber
metric_targets = all
hidden_dim = 128
embedding_dim = 64
num_layers = 1
```

### 11.6 Stage 5B 训练结果

输出目录：

```text
outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/context_gru_stage5b_v1
```

结果：

```text
total_train_samples = 131998
total_val_samples = 16481
context_dim = 83
feature_dim = 33
embedding_dim = 64
best_val_loss = 3.8376607806942546
best_epoch = 19
final_train_loss = 3.840105953356572
final_val_loss = 3.8383810137683967
warnings = []
```

判断：

```text
训练稳定；
train / val loss 都下降；
train / val 接近；
无明显过拟合；
best epoch 在 19；
Stage 5B v1 训练成功。
```

### 11.7 Stage 5B embedding 导出

导出目录：

```text
outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/context_gru_stage5b_v1_embeddings
```

embedding manifest：

```text
outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/context_gru_stage5b_v1_embeddings/embedding_manifest.json
```

结果：

```text
embedding_dim = 64
total_rows = 164871
split = all
embedding_shards = 35
nonfinite_embedding_detected = 0
row_alignment = Each embedding shard follows source shard row order
```

结论：

```text
Stage 5B v1 embedding export completed.
```

---

## 12. Stage 5C：strict-schema context embedding evaluation

Stage 5C 已完成，且采用严格 schema 对齐流程完成复核。

### 12.1 Stage 5C 的两轮修正

- 初始 Stage 5C-v1 属于预评估版本，当时存在 fallback feature index，结论只可作为方向性参考。
- Stage 5C-1 修复为 strict feature schema，统一按 `feature_schema.json` 强约束映射，不再允许 fallback。
- Stage 5C-2 在 strict schema 基础上增加 category-wise evaluation，使各行为组表现可解释、可横向比较。

### 12.2 strict schema 核验结论

- `feature_schema_loaded = true`
- `strict_feature_schema = true`
- `paper_grade_valid = true`
- `no fallback feature index`
- `row_alignment_checks.aligned = true`

### 12.3 Stage 5B 基线在 strict schema 下的评估结果

- `hit@5 = 0.490300`
- `longitudinal_comfort = 0.150833`
- `following_interaction = 0.302917`
- `lateral_lane_dynamics = 0.266777`
- `behavior_proxy = 0.190567`

### 12.4 解释

- Stage 5B 结果是有意义的，不是随机波动。
- 相比 `random/context_l2` 有稳定优势。
- 在横向动态（lateral dynamics）上表现较强。
- 在跟驰/前车距离相关行为上相对偏弱。
- 因此直接推动 Stage 5D：在训练目标层面做 group-weighted 的多目标平衡优化。

---

## 13. Stage 5D：group-weighted multi-objective training

Stage 5D 不改数据集，核心只改训练目标。

### 13.1 原理

总损失：

```text
Total loss =
  style loss
  + weighted auxiliary regression losses
  + weighted group metric alignment losses
```

行为组：

- longitudinal_comfort
- following_interaction
- lateral_lane_dynamics
- lateral_gap_interaction
- behavior_proxy

训练权重逻辑：

- following 权重过低：THW/front distance/relative speed 学不强。
- following 权重过高：embedding 会被 following 主导，横向动态被挤压。
- lateral 权重过低：yaw/heading/lane-change 结构会弱化。
- 目标是“平衡行为表示”，而不是只把单一类别刷到最高。

### 13.2 Stage 5D-v1：following enhancement

结果：

- `hit@5 = 0.507992`
- `longitudinal_comfort = 0.151584`
- `following_interaction = 0.582954`
- `lateral_lane_dynamics = 0.204637`
- `behavior_proxy = 0.355707`

解释：

- following 显著增强。
- behavior_proxy 同步提升。
- lateral dynamics 出现过度校正下滑。
- 因此 v1 是重要 ablation，不是最终推荐模型。

### 13.3 Stage 5D-balanced-v2：current recommended model

结果：

- `hit@1 = 0.213092`
- `hit@5 = 0.526232`
- `mean_same_label_fraction_at_5 = 0.189776`
- `longitudinal_comfort = 0.171751`
- `following_interaction = 0.501998`
- `lateral_lane_dynamics = 0.245608`
- `behavior_proxy = 0.322344`

解释：

- 当前最优综合权衡（best current trade-off）。
- 全局 retrieval 指标提升。
- 修复 Stage 5B 的 following 弱项。
- 恢复了大部分 lateral dynamics 能力。
- 在 following_interaction 与 behavior_proxy 上明显胜出。
- longitudinal_comfort 与 lateral_lane_dynamics 与强基线接近或近似持平。
- 但仍不能宣称全局超越 `raw_feature/pca_feature` retrieval baselines。

---

## 14. Stage 5E：final comparison report

Stage 5E 在同一 strict-schema evaluation protocol 下，对 Stage 5B、Stage 5D-v1、Stage 5D-balanced-v2 做最终对比。

输出目录：

```text
outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/stage5_final_comparison
```

关键文件：

- `final_stage5_recommendation.md`
- `final_stage5_model_comparison.csv`
- `final_stage5_category_comparison.csv`
- `final_stage5_retrieval_comparison.csv`
- `final_stage5_learned_win_summary.csv`
- `final_stage5_comparison_plot.png`

对比表：

| Model | hit@5 | longitudinal | following | lateral | behavior_proxy | Interpretation |
|---|---:|---:|---:|---:|---:|---|
| Stage 5B baseline | 0.490300 | 0.150833 | 0.302917 | 0.266777 | 0.190567 | strong lateral, weak following |
| Stage 5D-v1 | 0.507992 | 0.151584 | 0.582954 | 0.204637 | 0.355707 | following over-correction |
| Stage 5D-balanced-v2 | 0.526232 | 0.171751 | 0.501998 | 0.245608 | 0.322344 | best current trade-off |

learned-win 特征统计：

- Stage 5B 同时胜过 raw/pca 的特征数：8。
- Stage 5D-v1 同时胜过 raw/pca 的特征数：10。
- Stage 5D-balanced-v2 同时胜过 raw/pca 的特征数：17。

最终推荐：

```text
Stage 5D-balanced-v2
```

研究解释：

- Stage 5D 是受控多目标权衡，不是随机调参。
- Stage 5B 暴露 following 弱项。
- Stage 5D-v1 证明 following 可被显著增强。
- Stage 5D-balanced-v2 将 following 增强与横向结构恢复到更平衡状态，成为当前推荐模型。

---

## 15. Stage 5F：paper-level experiment consolidation

Stage 5F 是当前下一阶段，且 **不是训练阶段**，而是论文级实验整合阶段。

计划输出目录：

```text
outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/stage5_paper_package
```

计划文件：

- `README.md`
- `stage5_paper_experiment_summary.md`
- `stage5_paper_tables.md`
- `stage5_method_section_draft.md`
- `stage5_results_section_draft.md`
- `stage5_limitations_and_next_steps.md`
- `final_stage5_comparison_plot.png`

目的：

- 汇总 Stage 5A-E。
- 冻结当前推荐模型。
- 形成 paper-ready 方法章节草稿。
- 形成 paper-ready 结果章节草稿。
- 明确当前限制与后续工作。
- 将 Stage 5 embedding 与后续 BDD / E2E style comparison 对接。

---

## 16. Stage 5G：optional Slot Encoder + Attention Pooling ablation

当前主线模型是 Flatten Context GRU，优势是简单、稳定，并且已经得到推荐模型 Stage 5D-balanced-v2。

Slot Encoder + Attention Pooling 延后至 Stage 5G（可选消融）。

动机：

- 保留显式 slot 结构。
- 分别编码 ego/front/left_front/left_rear/right_front/right_rear。
- 通过 attention pooling 学习关键交互槽位。
- 提高解释性与交互敏感度。
- 若论文需要更强架构贡献，可作为体系化架构消融补充。

重要边界：

- Stage 5G 是可选项。
- 不应阻塞 Stage 5F。
- 未经实证提升，不应替代 Stage 5D-balanced-v2 主线推荐地位。

---

## 17. Stage 6：E2E model style comparison / BDD report card

Stage 6 将 Stage 5D-balanced-v2 embedding 用于两版 E2E 模型（或两份 policy rollout 数据）之间的风格差异比较。

输入：

- Model A trajectory logs
- Model B trajectory logs
- trained Stage 5D-balanced-v2 encoder
- `feature_schema.json`
- scenario metadata（如可用）

BDD：Behavioral Distribution Distance

```text
BDD(A, B) = distance between embedding distributions Z_A and Z_B
```

可选距离：

- MMD
- Wasserstein
- Fréchet distance
- energy distance

BDD 回答“差多少”；为回答“差在哪”，Stage 6 输出：

- `category_delta.csv`
- `scenario_slice_delta.csv`
- `top_drift_cases.csv`
- `style_report_card.md/pdf`
- `style_radar.png`
- `embedding_umap.png`
- `case_gallery.html`

面向不同受众的表达形式：

1. Leadership：E2E Driving Style Report Card
2. Academic reviewers：Behavior Distribution Shift Evaluation
3. Engineering colleagues：Style Drift Debug Dashboard

Stage 6 需要真实 E2E 模型轨迹日志或成对 policy rollout 数据。

---

## 18. Synthetic Policy / BDD framework reminder

Stage 1-3 synthetic policy validation 是受控验证阶段。

synthetic policies 是可控行为变体：

- conservative
- aggressive
- lateral_stable
- comfort
- following_safe
- assertive
- yielding

目的：

这些可控策略提供“已知行为差异”，用于验证 behavior embedding 与 BDD 是否能正确检出风格偏移。

BDD：Behavioral Distribution Distance，是两份 policy / model 版本 embedding 分布之间的距离。

连接关系：

- Stage 5 提供更强的 interaction-aware 公共人类轨迹编码器。
- Stage 6 将用该编码器计算真实 E2E 版本间的 BDD。

---

## 19. 当前不能夸大的结论

当前不能声称：

- learned embedding 在全局上全面超越 raw/pca feature retrieval baselines；
- Stage 5D-balanced-v2 可替代全部 handcrafted metrics；
- 本 benchmark 等价于 closed-loop autonomous driving simulation；
- Waymo public human trajectory 验证等价于私有 E2E 实车验证。

当前可以声称：

- Stage 5D-balanced-v2 是目前 Stage 5 learned representation 中最优版本；
- 其在 learned retrieval 全局指标上优于 Stage 5B 与 Stage 5D-v1；
- 在关键行为类别上取得胜出或近似持平；
- 为 BDD / E2E model style comparison 提供了可落地基础。

---

## 20. 当前项目状态总览

| Module | Current status | Next step |
|---|---|---|
| synthetic policy generator | completed | can support BDD controlled validation |
| Stage 4 ego-only validation | completed main chain | paper background / baseline reference |
| Stage 5A context dataset | completed | do not modify builder |
| Stage 5B baseline | completed | baseline result |
| Stage 5C strict-schema evaluation | completed | use final eval outputs |
| Stage 5D training improvements | completed | Stage 5D-balanced-v2 recommended |
| Stage 5E final comparison | completed | use final comparison report |
| Stage 5F paper package | current next task | generate paper-ready files |
| Stage 5G slot encoder | optional future ablation | defer |
| Stage 6 E2E style report | future application | design later |

Priority：

- P0：repair `00_plans.md` without deleting old history
- P1：Stage 5F paper package
- P2：Stage 6 report card design
- P3：optional Stage 5G architecture ablation

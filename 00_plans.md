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

## 12. 当前下一步：Stage 5C evaluation

当前不应继续：

```text
不要继续优化 GPU 利用率；
不要继续改 Stage 5A builder；
不要重新训练 Stage 5B；
不要直接拿旧 Stage 4G embedding 与 Stage 5B 乱比。
```

当前应该进入：

```text
Stage 5C：Evaluate context-aware embedding
```

### 12.1 Stage 5C 核心问题

> Stage 5B context-aware embedding 是否真的学到了 interaction-aware behavior structure？

### 12.2 公平性问题

注意：

```text
旧 Stage 4G ego-only embedding 可能不是同一 row set；
旧 Stage 4 full51 是 168191 rows；
Stage 5A clean lane-aware merged dataset 是 164871 rows。
```

因此 Stage 5C 初版不要直接和旧 Stage 4G embedding 比较，除非确认 row alignment。

更公平路线：

#### 方案 A：先做 Stage 5 dataset 内部评估

比较：

```text
learned_context_embedding
raw_feature
pca_feature
context_l2
random
```

#### 方案 B：后续补 same-row ego-only baseline

在相同 164871 rows 上训练 ego-only baseline：

```text
ego_seq / context_traj ego part
↓
ego-only GRU
↓
64-D embedding
```

然后再公平比较：

```text
ego-only same-row baseline
vs
context-aware Stage 5B
```

### 12.3 Stage 5C 推荐工具

新增：

```text
tools/evaluate_context_embedding.py
```

输入：

```text
--embedding_manifest outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/context_gru_stage5b_v1_embeddings/embedding_manifest.json
--source_shard_manifest outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/shard_manifest.json
--out_dir outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/context_gru_stage5b_v1_eval
--eval_split test
--max_eval_samples 20000
```

### 12.4 Stage 5C 首轮比较对象

```text
learned_context_embedding
raw_feature
pca_feature
context_l2
random
```

### 12.5 Stage 5C 指标

#### A. Retrieval / kNN consistency

例如：

```text
hit@1
mean_same_fraction
neighbor feature distance
```

#### B. Style-distance correlation

计算 embedding distance 与 feature delta 的 Spearman correlation。

重点 feature：

```text
mean_speed
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
```

#### C. Interaction-specific sensitivity

重点验证：

```text
mean_thw
min_thw
front_distance
relative_speed
front slot occupied
left/right slot occupied
```

这是 Stage 5B 相比 Stage 4 的论文价值所在。

#### D. Visualization

抽样：

```text
max_eval_samples = 20000
```

图：

```text
embedding PCA / UMAP
colored by:
  mean_thw
  min_thw
  front_distance
  rms_jerk
  yaw_rate / curvature
  slot occupancy
```

### 12.6 Stage 5C 输出文件

建议输出：

```text
evaluation_summary.json
retrieval_metrics.csv
style_distance_correlation.csv
feature_delta_correlation_bar.png
retrieval_bar.png
pca_embedding.png
evaluation_report.md
```

其中 `evaluation_report.md` 必须中文。

### 12.7 Stage 5C 初版通过标准

```text
1. 能读取 sharded embedding_manifest。
2. 能读取 sharded source data。
3. row alignment 正确。
4. 支持 eval_split=test。
5. 支持 max_eval_samples=20000。
6. 不拼接全量巨大数组。
7. learned_context_embedding / raw_feature / pca_feature / context_l2 / random 都有结果。
8. 输出 evaluation_summary.json。
9. 输出 evaluation_report.md，且为中文。
10. 所有图表和 CSV 都能生成。
11. no NaN/Inf in loaded evaluation arrays。
```

---

## 13. Agent / Codex 工作规范补充

由于前期 Codex 多次出现：

```text
忘记更新 QUICK_REFERENCE.md
引入 /tmp/old.py
没有跑 py_compile
随意大改 builder
summary 硬编码空字段
路径拼接错误
接口字段不存在
```

已决定在仓库根目录新增：

```text
AGENTS.md
```

用途：

```text
给 Codex / AI coding agent 的长期工作规范。
```

每次给 Codex 的 prompt 第一行建议写：

```text
请先阅读仓库根目录 AGENTS.md，并严格遵守其中规则。
```

关键规范：

- 小步修改；
- 不大改已稳定模块；
- 每次修改必须更新 QUICK_REFERENCE.md；
- QUICK_REFERENCE.md 必须中文写清楚：命令、期望行为、通过标准；
- 每次 Python 修改必须 py_compile；
- 必须运行 check_no_tmp_dependencies.py；
- 不许依赖 `/tmp/old.py`；
- 不许用 exec 动态加载源码；
- sharded dataset 不许默认拼接大 npy；
- 当前只做 Stage 5C evaluation。

---

## 14. 当前可写入论文的结论

### 14.1 Synthetic controlled validation

> The learned behavior embedding is policy-discriminative and retrieval-capable under controlled synthetic rollout settings.

### 14.2 Within-source aligned evaluation

> Within-source comparison controls scene variation and allows policy-induced behavior differences to be measured directly.

### 14.3 Lateral-stable 机制结论

> Lateral-stable behavior requires joint lateral and longitudinal shaping. Stronger yaw-rate clipping and stricter jerk limitation improve p2 recognizability, retrieval consistency, yaw-rate stability, and longitudinal comfort.

### 14.4 Public human trajectory validation 结论

> Ego-only behavior embedding can be trained on public Waymo human trajectories with weak style supervision, but interaction-heavy style dimensions require explicit context.

### 14.5 Interaction-aware embedding 结论

当前可以写：

> We construct a lane-aware 5-neighbor context dataset and train a context-aware behavior embedding with a Flatten Context GRU. The training is stable and produces finite row-aligned sharded embeddings over 164,871 trajectory windows.

但还不能写：

```text
context-aware embedding 已经证明优于 ego-only。
```

因为 Stage 5C evaluation 还未完成。

---

## 15. 当前不能夸大的结论

不能写：

```text
lateral_stable 已经完全成为独立第三类驾驶风格。
```

应该写：

```text
lateral_stable 的独立性显著增强，但仍未完全成立。
```

不能写：

```text
embedding 已经全面证明适用于真实人类驾驶风格。
```

应该写：

```text
Stage 4/5 已经把验证推进到公开 Waymo human trajectories，但最终有效性需要 Stage 5C evaluation 支撑。
```

不能写：

```text
该 benchmark 等价于完整自动驾驶闭环仿真。
```

应该写：

```text
当前 benchmark 是 trajectory-level behavior evaluation，不包含 sensor rendering / perception stack。
```

不能写：

```text
Stage 5B 一定优于 Stage 4G。
```

应该写：

```text
Stage 5B 已完成训练和 embedding 导出，是否优于 ego-only 需要 Stage 5C 评估和 same-row baseline。
```

---

## 16. 当前项目状态总览

| 模块 | 当前状态 | 下一步 |
|---|---|---|
| synthetic policy generator | 可用 | 使用 recommended_lateral_stable_v2 |
| PR2 interpretability demo | 完成 | 保持，不再过度美化 |
| population evaluator | 完成 | 后续可作为 evaluation template |
| broad ablation | 完成 | 已得机制结论 |
| local fine sweep | 完成 | 已得 v2 配置 |
| Stage 4 human ego-only validation | 完成主要链路 | 作为 ego-only 经验基础 |
| Stage 5A context dataset | 完成 | 不再改 builder |
| Stage 5B context GRU training | 完成 | 不再重新训练，除非 evaluation 指出问题 |
| Stage 5B embedding export | 完成 | 用于 Stage 5C |
| Stage 5C evaluation | 当前下一步 | 新增 evaluate_context_embedding.py |
| same-row ego-only baseline | 未开始 | Stage 5C 后续公平对比 |
| paper outline | 需要更新 | 加入 Stage 4/5 结果 |
| QUICK_REFERENCE.md | 需要持续更新 | 每次新增工具都必须更新 |
| AGENTS.md | 建议创建 | 约束 Codex 行为 |

---

## 17. 当前最优先任务

### P0：Stage 5C 初版 evaluation

新增：

```text
tools/evaluate_context_embedding.py
```

完成：

```text
learned_context_embedding vs raw_feature vs pca_feature vs context_l2 vs random
```

输出：

```text
evaluation_summary.json
retrieval_metrics.csv
style_distance_correlation.csv
feature_delta_correlation_bar.png
retrieval_bar.png
pca_embedding.png
evaluation_report.md
```

### P1：same-row ego-only baseline

如果 Stage 5C 初版结果正常，下一步做 same-row ego-only baseline。

目的：

```text
公平比较 ego-only vs context-aware。
```

### P2：Stage 5 report card

围绕论文表达，生成：

```text
Driving Style Report Card
Interaction-aware Behavior Report
BDD / style drift demo
```

### P3：paper_outline.md 更新

把论文结构升级为：

```text
1. Introduction
2. Related Work
3. Behavior Embedding Framework
4. Controlled Synthetic Validation
5. Public Human Trajectory Validation
6. Interaction-aware Context Embedding
7. Evaluation and Ablation
8. Limitations
9. Conclusion
```

---

## 18. 最后结论

截至目前，项目已经完成了：

```text
synthetic policy validation
↓
public human ego-only validation
↓
lane-aware 5-neighbor context dataset construction
↓
Flatten Context GRU context-aware embedding training
↓
row-aligned sharded embedding export
```

当前最重要成果：

> 已经构建了 164,871 条 Waymo lane-aware 5-neighbor context trajectory windows，并训练出 64-D context-aware behavior embedding。

当前最重要限制：

> 还没有完成 Stage 5C evaluation，因此还不能声称 context-aware embedding 优于 ego-only 或 raw feature baseline。

下一步正式进入：

```text
Stage 5C：context-aware embedding evaluation
```

目标是回答：

> 5-neighbor interaction context 是否真正提升了 behavior embedding 对跟车、相对速度、THW、邻车交互和舒适性的表达能力？

# Stage 6L–6N：Context 消融、异场景平衡与模型重训练决策中文主报告

## 1. 结论摘要

本阶段回答了两个问题：当前 Waymo 训练的 64D behavior embedding 对 nuPlan 纯纵向
风格差异的检出是否依赖邻车/lane context，以及在真实异场景软件版本比较中，场景构成
平衡是否能显著提高单次发布检出率。

结论如下。

1. 当前完整 64D embedding 仍能在 25/50/75/100% 四档纯纵向处置上通过预冻结 overall
   随机化检验，因此 Stage 6J 的窄主张继续成立。
2. 但它没有比 ego-only control 增加纵向 style sensitivity。完整 64D 的 median overall
   `Z_BDD=7.539`、task-dose Holm 通过 `7/12`；同 checkpoint 邻车置零后为
   `11.066`、`11/12`；显式 ego 运动学 13D 为 `21.082`、`12/12`。
3. 因而当前 paired BDD 显著性不主要依赖 neighbor/lane context。对纯纵向差异，邻车
   context 更像在稀释或归一化 ego 信号，而不是增加检出能力。这一结论不能外推为
   “interaction 对所有驾驶风格都无用”。
4. fallback 与 pair distance 的正关联在邻车置零表示中仍存在，说明该关联至少部分反映
   场景难度、ego 轨迹或共同测量条件，而不能全部解释为邻车通道造成的 BDD 放大。
5. 800-pair 异 log/异场景 release emulation 中，n=400 时 raw、task-conditioned、
   context-balanced、task+context-balanced 的 A/B 检出率分别为 `63.0%`、`65.0%`、
   `66.5%`、`64.5%`。context-balanced 相对 raw 仅增加 `3.5` 个百分点，同一 release
   split 的配对 McNemar exact `p=0.2478`，不支持稳定提升。
6. 当前主要瓶颈更接近 representation sensitivity；scenario heterogeneity 是次要瓶颈，
   statistical calibration 是必须保留的运行条件，context construction 是重要测量限制，
   但不是本次受控 paired 信号成立的必要来源。
7. Go/No-Go 结论为：`GO_PREPARE_SEPARATELY_VERSIONED_TRAINING_PROTOCOL`。这只授权准备
   独立协议，不授权覆盖或立即重训冻结的 Stage5D-balanced-v2 checkpoint。

## 2. 数据完整性修正

在 Stage 6L 首次 representation 消融中发现，原
`outputs/stage6k_longitudinal_dose_context_v1/dose50` 和 `dose75` 的
`neighbor_seq.npy` 有效邻车覆盖为零。两个目录的 366/366 行均记录
`msgpack_timestep_mismatch samples=0`。原因是当时 context build 的运行环境没有同时加入
nuPlan devkit 和 tuPlan Garage 的 Python 路径，`SimulationLog` 反序列化失败；旧 slot
sanity 又把“所有 slot 全局零覆盖”当成低覆盖跳过项，而不是致命错误。

因此：

- 原 Stage 6K dose50/75 的完整 context 64D embedding 实际等价于邻车置零输入；
- 原 Stage 6K dose50/75 BDD 数值和基于它们的旧 dose curve 不再作为完整 context 的权威证据；
- 旧 Stage 6K 的 ego 速度、加速度和 jerk 结果仍有效，因为它们来自 rollout ego 轨迹；
- 旧 dose50/75 的 THW、gap 和前车有效率不再引用；
- `outputs/stage6l_context_representation_ablation_results_v1/` 已标记为 superseded，不删除。

修复版 context 写入新目录
`outputs/stage6k_longitudinal_dose_context_v2_runtime_repaired/`，未覆盖旧证据。三档的
邻车槽位帧覆盖率为 `17.14% / 17.44% / 17.37%`，均满足：

- `[366,150,83]` context 与 `[366,5,150,15]` neighbor shape；
- finite；
- scenario/planner/token 严格对齐；
- `validation.pass=true`；
- `required_nonzero_neighbor_coverage_pass=true`；
- 无 `samples=0` mismatch。

构建器和 Stage 6L freeze 均新增 fail-closed 非零邻车覆盖门禁。

## 3. Priority 1：Context-quality representation 消融

### 3.1 冻结设计

所有表示使用相同 183 个场景、156 个 log、25/50/75/100% 非零剂量、相同 Waymo
checkpoint 和 100000 次 paired/randomized null。不同表示各自重新冻结 bandwidth 和 null，
所以不能跨表示比较 raw MMD²，只比较各自 null 标准化后的 `BDD/q95`、`Z_BDD`、Holm
结论与最小检出剂量。

四个表示为：

- A：当前完整 context 的 learned 64D；
- B：相同 checkpoint、保留 ego 0:8 通道并把 neighbor 8:83 置零的输入消融；它不是
  重新训练的 learned ego-only 模型；
- C：mask-aware 显式 ego 运动学 13D；
- D：13D ego 与 33D interaction/trajectory summary 拼接后的 46D 手工表示。

C/D 的 median/IQR scaler 只用 dose100 conservative planner 的 183 行拟合，没有根据
BDD 结果、planner 名或剂量调参。

### 3.2 实现行为剂量

修复版 context 下的 overall A-B 差异如下。THW、gap 和 front-valid 是描述性指标，不是
运动学门禁；THW 中包含长时距截断值，因此不把其绝对差解释为真实跟车秒数的无偏估计。

| 名义剂量 | Δ平均速度 m/s | ΔRMS加速度 m/s² | ΔRMS jerk m/s³ | Δ平均THW s | Δ平均前车距离 m | Δ前车有效率 |
|---:|---:|---:|---:|---:|---:|---:|
| 25% | 0.2546 | 0.0361 | 0.0624 | -7.3998 | -0.1848 | -0.00709 |
| 50% | 0.4464 | 0.0770 | 0.0980 | -16.5166 | -1.3344 | -0.00317 |
| 75% | 0.6372 | 0.1279 | 0.1491 | -10.9336 | -1.2489 | -0.00463 |
| 100% | 0.9147 | 0.1816 | 0.2279 | -15.3846 | -1.6016 | 0.00858 |

速度、RMS 加速度和 RMS jerk 随名义剂量严格递增；THW/gap 不呈单调剂量响应。因此
处置首先是明确的纵向 ego style dose，不能把交互指标也表述为线性剂量。

### 3.3 Overall BDD

| 表示 | 剂量 | MMD² | null q95 | BDD/q95 | Z_BDD | raw p | 四档Holm p |
|---|---:|---:|---:|---:|---:|---:|---:|
| 完整64D | 25% | 0.001156 | 0.000898 | 1.288 | 3.647 | 0.004070 | 0.004070 |
| 完整64D | 50% | 0.002025 | 0.001223 | 1.655 | 5.885 | 0.000140 | 0.000280 |
| 完整64D | 75% | 0.003598 | 0.001590 | 2.263 | 9.193 | 0.000010 | 0.000040 |
| 完整64D | 100% | 0.005001 | 0.002096 | 2.386 | 9.228 | 0.000010 | 0.000040 |
| 邻车置零64D | 25% | 0.000547 | 0.000274 | 2.000 | 7.364 | 0.000010 | 0.000040 |
| 邻车置零64D | 50% | 0.001600 | 0.000654 | 2.445 | 9.601 | 0.000010 | 0.000040 |
| 邻车置零64D | 75% | 0.003322 | 0.001052 | 3.158 | 12.532 | 0.000010 | 0.000040 |
| 邻车置零64D | 100% | 0.006625 | 0.001507 | 4.397 | 17.900 | 0.000010 | 0.000040 |
| ego运动学13D | 25% | 0.002256 | 0.000802 | 2.812 | 11.484 | 0.000010 | 0.000040 |
| ego运动学13D | 50% | 0.006936 | 0.001720 | 4.033 | 16.581 | 0.000010 | 0.000040 |
| ego运动学13D | 75% | 0.014466 | 0.002334 | 6.198 | 25.582 | 0.000010 | 0.000040 |
| ego运动学13D | 100% | 0.027995 | 0.003130 | 8.945 | 35.371 | 0.000010 | 0.000040 |
| 手工交互+轨迹46D | 25% | 0.001013 | 0.002187 | 0.463 | -0.420 | 0.590504 | 0.590504 |
| 手工交互+轨迹46D | 50% | 0.004301 | 0.003033 | 1.418 | 3.811 | 0.008730 | 0.017460 |
| 手工交互+轨迹46D | 75% | 0.006859 | 0.003184 | 2.154 | 6.957 | 0.000330 | 0.000990 |
| 手工交互+轨迹46D | 100% | 0.009200 | 0.003873 | 2.375 | 7.665 | 0.000120 | 0.000480 |

### 3.4 Task 稳定性与最小检出剂量

| 表示 | task×dose Holm通过数/12 | median overall Z_BDD | 最小overall检出剂量 |
|---|---:|---:|---:|
| 完整64D | 7/12 | 7.539 | 25% |
| 邻车置零64D | 11/12 | 11.066 | 25% |
| ego运动学13D | 12/12 | 21.082 | 25% |
| 手工交互+轨迹46D | 2/12 | 5.384 | 50% |

25% 最小 overall 检出剂量在 A/B/C 中稳定，在 D 中不稳定。完整 64D 的 25% overall
结果成立，但不能写成“25%在所有典型纵向 task 都可靠检出”。

### 3.5 Context-quality association

完整 64D 的 fallback 与 pair L2 的 task-adjusted rank association 在四档均为正且 CI 不跨
零，估计约 `0.430/0.386/0.372/0.325`。absolute BDD contribution 的关联在前三档为正，
100% CI 跨零。

邻车置零后，fallback 与 pair L2 在四档仍为正且 CI 不跨零，估计约
`0.213/0.242/0.226/0.172`；absolute contribution 只在 25/50% 明确为正。这个结果说明：

- 完整 context 的质量与 embedding distance 确实共同变化，必须披露；
- 置零邻车后 pair-distance 关联减弱，但没有按预冻结规则减少至少两个“正关联剂量”；
- association 使用 rollout 后 fallback，只是描述性共同变化，不能据此删场景、重加权或作因果调整；
- `context-v2` 的机械决策是“准备独立版本协议”，但当前证据不支持“paired BDD 是由错误
  邻车 context 伪造”的说法。

## 4. Priority 2：Context-balanced unpaired BDD

### 4.1 四种 release-level 方法

Stage 6M 复用 Stage 6H 已冻结的 800 pairs、489 logs、2400 个 release splits 和固定
bandwidth，不重跑仿真或 embedding。四种方法在结果聚合前冻结：

1. raw marginal overall BDD；
2. 五个互斥冻结 task 的 raw scope MMD²，按完整 800-pair pool task prevalence 加权；
3. overall map_name×scenario_type exact common-support 标准化 BDD；
4. task 内 context 标准化后再按同一冻结 task prevalence 聚合。

每种方法和每个 `n=200/250/300/400` 都使用自己的独立 A/A calibration 95% empirical
threshold。matching 只使用 pre-treatment 的 map、scenario type 和 task；不使用 planner
outcome、fallback、realized braking/lane change、embedding distance 或 BDD。

### 4.2 A/A 与 A/B 可靠性

| 每版本场景数 | 方法 | A/A FPR | A/B detection |
|---:|---|---:|---:|
| 200 | raw / task / context / task+context | 7.5% / 6.0% / 8.0% / 4.0% | 31.5% / 27.0% / 30.0% / 15.6% |
| 250 | raw / task / context / task+context | 4.5% / 5.5% / 6.5% / 6.5% | 30.0% / 29.0% / 28.5% / 26.5% |
| 300 | raw / task / context / task+context | 2.0% / 0.5% / 3.5% / 0.5% | 26.0% / 22.0% / 41.5% / 20.5% |
| 400 | raw / task / context / task+context | 4.5% / 5.5% / 5.0% / 6.0% | 63.0% / 65.0% / 66.5% / 64.5% |

n=400 的 Wilson 95% CI 分别为：

- raw A/B：`[56.1%,69.4%]`；
- task-conditioned A/B：`[58.2%,71.3%]`；
- context-balanced A/B：`[59.7%,72.7%]`；
- task+context A/B：`[57.7%,70.8%]`。

四个区间高度重叠。context-balanced 相对 raw 的同 split 检出增量在 n=400 为 `+3.5pp`，
candidate-only/reference-only alerts 为 `17/10`，McNemar exact `p=0.2478`。n=300 曾出现
较大提升，但跨样本量不稳定，不能把单档结果解释为普遍增益。

context-balanced 的 n=400 median common support 约 `98.8%/99.0%`，最差均为 `96.0%`；
median ESS ratio 约 `0.990/0.991`，最差约 `0.958/0.949`；最大权重比最差约
`3.57/3.58`，均通过冻结门禁。28,800 条重建审计中只有 2 个 task scope-trial 不可比；
其余加权后 map/scenario-type 最大类别比例差为 `1.22×10^-15`，即数值零。

因此，失败不是“平衡没有真正实现”，而是已测量 scenario composition 平衡后，表示本身和
有限样本统计功效仍限制检出。

## 5. 当前瓶颈判断

| 候选瓶颈 | 证据 | 判断 |
|---|---|---|
| Context construction | fallback 与 distance 相关，旧 dose50/75 曾发生零邻车覆盖；但修复后 paired overall 仍显著，neighbor-zero 更敏感 | 重要测量限制，不是 paired 主信号的必要来源 |
| Scenario heterogeneity | n=400 context balance 由63.0%升至66.5%，n=300有较大增益但不稳定 | 次要且真实存在，不能单独解决可靠性 |
| Statistical calibration | 每方法/样本量独立 A/A 后 FPR 大致维持5%，阈值随设计变化 | 必须保留的运行条件，不是可删除的“损失” |
| Representation sensitivity | ego13D明显强于完整64D；neighbor-zero优于完整64D；平衡后仍约33.5%假阴性 | 当前首要瓶颈 |

## 6. Priority 3：新 checkpoint Go/No-Go

预冻结规则中以下两项触发：

- ego13D median Z_BDD 至少为完整64D的1.5倍，且多通过至少2个 task-dose Holm 单元；
- neighbor-zero 保留至少80%的完整64D median Z_BDD、task cell不多损失1项，同时质量关联
  触发 context-v2 protocol 准备条件。

所以结论为：

`GO_PREPARE_SEPARATELY_VERSIONED_TRAINING_PROTOCOL`

不是：

`START_OR_OVERWRITE_STAGE5D_BALANCED_V2_NOW`

新协议由 Issue #255 管理，至少包含：

1. 扩大 Waymo 纵向覆盖，重点增加低剂量、低速/高速、跟车、stop/go、free-flow 与 hard
   negative/near-boundary 样本；先做样本分布审计，不只增加总行数。
2. 加入 longitudinal contrastive/ranking objective，使已知强弱风格对在 embedding 中保持
   有序 margin；planner 名称不得作为 encoder 输入。
3. 加入 speed、acceleration、jerk、THW、gap auxiliary heads，并预冻结 loss 权重；auxiliary
   目标用于保证纵向可辨识性，不把最终 embedding 简化为单一速度分数。
4. 加入 neighbor/context dropout 和有效性 mask，让 ego style signal 不因偶发 context 质量
   下降而被淹没；context-quality-aware 训练规则必须在读取 nuPlan BDD 前冻结。
5. 旧 checkpoint SHA-256
   `909022f5df03a3f01c2149da6c9b44c613e955a4d816e8ec4d5862f39f8bf0cc`
   始终保留为 baseline，新模型使用独立目录、config、seed、版本号和证据链。
6. 新 checkpoint 必须重新验证 Waymo validation、Stage6J/6K paired dose curve、Stage6M
   unpaired A/A calibration、A/B detection、FPR/FNR tradeoff。只提高 paired BDD 不足以通过。

## 7. 论文 claim 与 limitation

可以写：

> 在固定 Waymo checkpoint、183 个 nuPlan same-scenario pairs 和预冻结 paired-null 设计下，
> 当前模型能够检出经运动学确认的典型纯纵向 planner 风格差异；修复 context 后，25% 名义
> 剂量仍为 overall 最小检出档。异 log/异场景 release emulation 中，经过匹配 A/A 标定的
> context-balanced 方法在400场景/版本获得66.5%检出率和5.0%误报率，证明公开数据上的
> 异场景版本风格检测可行，但可靠性尚不足以覆盖每次发布。

还可以写：

> Representation 消融表明，当前64D interaction-aware embedding 对纯纵向差异的
> null-standardized sensitivity 低于邻车置零和显式 ego 运动学 baseline；因此下一步改进
> 应优先增强纵向 representation objective，而不是以扩大 raw BDD 为目标调 lane pipeline。

不能写：

- 当前64D 比 ego-only 更擅长检出纵向风格；
- 0.001、0.005或Waymo验证集约0.5可以作为跨数据集通用BDD阈值；
- context balancing 已显著或稳定地把单次发布可靠性提高到80%；
- 25%在所有 task 都可靠检出；
- 当前结果已完成真实整车厂版本验证；
- BDD显著等于安全性提升、planner优越性或因果效应；
- fallback association 证明 lane assignment 导致全部 embedding difference。

## 8. 权威产物

- Stage 6L v2：`outputs/stage6l_context_representation_ablation_results_v2_runtime_repaired/`
- 修复 context：`outputs/stage6k_longitudinal_dose_context_v2_runtime_repaired/`
- 修复 embedding：`outputs/stage6k_longitudinal_dose_embeddings_v2_runtime_repaired/`
- 修复 realized dose：`outputs/stage6k_realized_longitudinal_dose_curve_v2_runtime_repaired/`
- Stage 6M freeze：`outputs/stage6m_context_balanced_unpaired_bdd_freeze_v1/`
- Stage 6M results：`outputs/stage6m_context_balanced_unpaired_bdd_results_v1/`
- Stage 6L dose-Z 图：
  `outputs/stage6l_context_representation_ablation_results_v2_runtime_repaired/stage6l_representation_dose_z_bdd.png`
- realized dose 图：
  `outputs/stage6k_realized_longitudinal_dose_curve_v2_runtime_repaired/stage6k_realized_dose_curve.png`
- 四方法可靠性图：
  `outputs/stage6m_context_balanced_unpaired_bdd_results_v1/stage6m_four_method_reliability.png`
- A/A 与 A/B 分布图：
  `outputs/stage6m_context_balanced_unpaired_bdd_results_v1/stage6m_aa_ab_distributions_n400.png`
- context-quality association 图：
  `outputs/stage6l_context_representation_ablation_results_v2_runtime_repaired/stage6l_context_quality_association.png`

本报告使用的 Issue 为 #253（Stage6L）、#254（Stage6M）和 #255（Stage6N retraining
protocol）。

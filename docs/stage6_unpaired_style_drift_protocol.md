# Stage 6A 非配对实路风格漂移评估协议（Unpaired-First）

## 1. 工程背景
Stage 6 的目标是服务真实 E2E 模型版本迭代：
- A 组：上一版模型实路日志；
- B 组：当前版模型实路日志。

在每次模型发布后，工程侧关心“行为风格是否漂移、漂移幅度多大、主要漂移在什么类型行为上”。这不是学术上的同场景 A/B 对照，而是实路异源日志比较。

## 2. 为什么 Stage 6 必须 unpaired-first
真实数据采集条件天然不一致：
- 城市、道路等级、路线分布不同；
- 时段、天气、交通拥堵水平不同；
- 测试司机与任务编排不同。

因此，部署模式不能依赖“逐样本配对（paired）”的假设。Stage 6A 以非配对分布比较为主，paired 仅保留为补充验证模式。

## 3. 为什么 paired 仅是验证模式
paired 在以下场景有价值：
- 仿真同场景 replay 的 sanity-check；
- planner/replay 回放对照；
- 受控实验的可解释归因。

但 paired 不是主部署入口，因为它不能代表实路异源日志比较的主体问题。

## 4. 非配对核心挑战：场景混杂（confounding）
非配对比较中，BDD 上升可能来自两类原因：
1) 模型行为风格确实变化；
2) A/B 场景分布（ODD）不同。

Stage 6A 不回避该问题，而是通过切片与解释层减少误判风险。

## 5. Stage 6A 总体流程
`logs -> window slicing -> scene/proxy tagging -> embedding -> BDD -> category/feature/slice explanation -> top drift cases -> report card`

### 5.1 输入
- A/B 索引（来自 split 构建工具）；
- 特征矩阵与 schema；
- 行对齐 embedding（优先）或 context+encoder（回退模式）。

### 5.2 主指标
- 在交互行为 embedding 空间计算 BDD-MMD（含 bootstrap CI 与 permutation p-value）。

### 5.3 解释层
- category delta（按 YAML 分组与方向定义）；
- feature delta（逐特征差异与统计显著性）；
- scenario/proxy slices（速度/THW/交互密度等切片）；
- top drift cases（可追踪样本级解释字段）。

## 6. BDD 测什么
BDD（MMD²）衡量的是 **A/B 在 embedding 空间的分布漂移幅度**。

## 7. BDD 不测什么
BDD 本身不直接给出：
- 漂移方向（更保守/更激进/更舒适）；
- 安全认证结论；
- 因果归因（是模型变化还是场景变化）。

## 8. 为什么 embedding 是主度量空间
Stage 5 学到的 interaction-aware embedding 作为主度量空间，原因：
- 保留时序轨迹结构；
- 保留 ego-neighbor 交互关系；
- 保留多特征联合分布（非单维均值）；
- 支持样本检索、原型分析、case mining。

## 9. 为什么 category/feature 是解释层
category/feature 的职责是“在检测到漂移后解释方向”，而非替代主分布比较。

即：先由 BDD 回答“是否漂移”，再由 category/feature/slice/case 回答“漂移到哪里”。

## 10. 为什么简单特征均值不足
- 均值会掩盖多模态分布；
- 同均值可对应不同分布形状；
- ego-only 指标无法覆盖交互动态；
- 难以做代表性 case 检索；
- 无法提供统一行为空间。

## 11. Stage 6A 验证实验
1. `negative_control_random`
   - 同一 test 池随机切 A/B，预期漂移较小。
2. `pseudo_style_aggressive_vs_conservative`
   - 用 proxy 构造伪风格两端，预期漂移较大。
3. `scene_confounding_control`
   - 构造低速高密 vs 高速低密，验证场景混杂可抬升 BDD。

## 12. Scenario/Proxy 切片
默认/推荐切片：
- `speed_bin`
- `thw_bin`
- `interaction_density_bin`
- `front_valid_bin`（若可用）

切片能降低混杂，但不能完全消除 unpaired 因果歧义。

## 13. Report Card 输出
- executive summary
- BDD summary
- `category_delta.csv`
- `feature_delta.csv`
- `scenario_slice_delta.csv`
- `top_drift_cases.csv`
- plots（category/feature/bdd/pca）
- `style_report_card.md`

## 14. 局限与评审风险
- weak labels / proxy labels 精度有限；
- unpaired 模式存在因果歧义；
- BDD 量纲需负/正对照标定；
- pseudo split 可能向解释特征“泄漏”；
- 无视频/元数据时 case 解释仅 proxy 级别。

## 15. Stage 6B/6C/6D 路线图
- Stage 6B：更强 scenario matching 与 baseline（重加权、matching、分层抽样）。
- Stage 6C：报告卡工程化与 case gallery（支持质检闭环）。
- Stage 6D：跨数据域验证（Argoverse / nuPlan / 公司实路日志）。


## 12. Issue #116 实施约束（full51）

- Stage 5 full51 数据默认是 **sharded dataset + manifest**，不是根目录扁平 npy。
- Stage 6A 构建 split 时，推荐使用 `--shard_manifest .../shard_manifest.json`。
- Stage 6A compare 时，推荐使用 `--source_shard_manifest` + `--embedding_manifest`（Stage5D-balanced-v2 导出的 `embedding_manifest.json`）。
- 扁平 `--feature_path/--split_path/--embedding_path` 模式仅作为 legacy fallback，不能作为 full51 主流程。

## 13. 2026-05-21：Stage 6A 统计与解释修正（negative control 复核）

- BDD 主度量使用标准 multi-kernel RBF MMD²：
  - `MMD² = mean(Kxx) + mean(Kyy) - 2*mean(Kxy)`；
  - 带宽来自 A/B 合并采样后的 pairwise distance 中位数，并使用 `[0.25, 0.5, 1, 2, 4]` 多尺度；
  - 中位数无效时回退到 `bandwidth=1.0` 并记录 warning。
- BDD 不再输出占位统计量：
  - `bdd_bootstrap_samples.csv` 输出 bootstrap 样本；
  - `bdd_permutation_samples.csv` 输出 permutation 样本；
  - `bdd_summary.json` 写入真实 `ci95_low/ci95_high/p_value`。
- category/feature 层的 `p_value` 改为双侧 permutation 计算，禁止默认写死 `1.0`。
- top drift cases 恢复解释字段：
  - `dominant_category` 基于 robust deviation（median/IQR）；
  - `top_changed_features`、`feature_values` 输出样本级特征变化；
  - `slice_tags` 使用 speed/thw/interaction density 代理分箱（可用即填）。
- scenario slicing 改为鲁棒三分位切片：
  - 代理特征支持 alias 扩展（含 `min_thw/thw_min/neighbor_valid_count`）；
  - 分位退化、单箱塌缩、样本不足会 warning 并跳过，不再伪造“全样本单切片”。

## Stage 6C pointer: dynamic interaction exposure and event-specific style diagnosis

Stage 6C adds a new diagnosis layer after Stage 6A/6B. It does not rewrite the Stage 6A unpaired BDD protocol and does not remove Stage 6B static ODD or coarse behavior-event outputs. The new layer separates:

1. **Static Map ODD**: road geometry / HD-map context such as map complexity, lane count, curvature, crosswalk, and stop sign;
2. **Dynamic Interaction Exposure**: traffic interaction conditions such as following pressure, cut-in exposure, overtake opportunity, dense traffic, front/side/gap pressure, and yielding conflict;
3. **Behavior Outcome / Style**: what the driver or model did, such as ego lane change, hard braking, late braking, hesitation, assertive interaction, stop-go, and lateral instability.

`exposure_*` bins can be considered for future dynamic matching/control. `outcome_*` bins should primarily be used for reporting and localization because they may directly encode behavior style. Embedding-based BDD remains the unified behavior distribution measurement layer, while event-specific handcrafted metrics provide semantic diagnosis of the detected drift.

Implementation details and commands are documented in `docs/stage6c_dynamic_interaction_event_design.md` and `QUICK_REFERENCE.md`.

## 16. Stage 6C behavior-event taxonomy v2 更新

Stage 6C v2 将 behavior-event bin 明确定义为 **task slice / comparable driving context**，用于在相同驾驶任务内计算 task-conditioned BDD。主评价对象是 embedding distribution difference within task，而不是 outcome bins。

新增设计文档：`docs/stage6c_behavior_event_taxonomy_v2.md`。

新增构建脚本：`tools/stage6c_build_behavior_events_v2.py`。

v2 primary task slices：

- following / car-following；
- lane change；
- overtake / passing；
- cut-in response；
- hesitation / aborted maneuver；
- yield conflict / interaction assertiveness。

后续 `negative_control_random`、`pseudo_agg_vs_cons`、`scene_confounding_control` 的 Stage 6C 报告应优先写 task-conditioned BDD 结论；THW、gap、decel、jerk、sharpness、yielding/assertiveness 等 handcrafted metrics 只作为漂移方向解释层。

## 17. Stage 6C v2 smoke 修正：物理信号质检与 strength-filtered BDD

Stage 6C v2 保持 task-conditioned behavior-event BDD 架构不变。本轮只修正 smoke test 暴露的具体工程问题：derivative finite-difference 噪声、过宽松 hesitation detector、过宽松 lead-brake proxy、queue strong/proxy 敏感性分析，以及 observed BDD 与 bootstrap CI 的估计器配置记录。

- `tools/stage6c_build_behavior_events_v2.py` 默认对 speed、accel、yaw_rate、lateral velocity 做 5 帧平滑，并对 accel/decel/jerk/yaw_rate/lateral_accel/curvature 使用物理裁剪；正式 `behavior_event_metrics_v2.csv` 使用 smoothed/clipped metrics。
- raw finite-difference 诊断不会被隐藏：`behavior_event_schema_v2.json` 记录 `raw_metric_diagnostics`、`clipped_metric_diagnostics`、`metric_quality_warnings`，并在 raw/final 指标超出物理范围时写 warning。
- hesitation 必须有 lane-change/lateral/heading maneuver context，并使用平滑后的 yaw/lateral velocity sign changes；目标是避免 positive_ratio 接近 1 的退化 detector。
- lead-brake response 优先使用 front_speed 的持续减速度作为 strong detector；fallback 才使用 sustained closing-rate derivative proxy，并继续通过 detector strength column 记录。
- `tools/stage6c_task_conditioned_bdd_report.py` 增加 `--detector_strength_filter {all,strong,strong_or_proxy}`，可对 queue 等 proxy 占比较高的 task 做 strong-only sensitivity check。
- `task_bdd_summary.csv` 增加 `bootstrap_mean`、`bootstrap_std`、`observed_in_bootstrap_ci`、`mmd_estimator_config`，用于确认 observed BDD 和 bootstrap CI 使用一致的 max-sample policy。

正式分析前必须先检查 QUICK_REFERENCE.md 中的 Stage 6C v2 smoothing / clipping validation checklist。

## Stage 6C v2 quality tightening note (2026-06-08)

本轮不改变 Stage 6C v2 的 task-conditioned BDD 设计，也不移除 task-conditioned BDD；只收紧 behavior-event 构建质量控制：

- TTC/THW 在进入 metrics 前清理 `>=999`、`<=0` 和超过有效上限（默认 30s）的值，避免 999 sentinel 进入报告或 diagnostic scores。
- `queue_distance_when_start_decel` 明确视为距离 metric，不再被 physical warning 误判为 deceleration metric。
- lane-change lateral speed 默认按 5.0m/s 裁剪，heading-change total 默认按 8.0rad 封顶；schema 会同时记录 raw vs clipped diagnostics。
- `task_lane_change` 必须有足够 lateral displacement；yaw-rate / heading-change 不可单独触发。若 positive_ratio 仍大于 0.40，输出 `lane_change_detector_broad`。
- `task_hesitation` 必须满足 maneuver context 且至少两个 evidence components，默认 sign-change 阈值提高到 8；若 positive_ratio 仍大于 0.40，输出 `hesitation_detector_broad`。
- 当前解释优先级：`following` 与 `yield_conflict` 是最可靠 strong detectors；`cutin`、`overtake` 以及相当一部分 `lead_brake` / `queue` 仍是 proxy-based；`lane_change` 与 `hesitation` 只有在收紧后 positive_ratio 不 broad 时才建议作为稳定结论。

## 18. Matched simulation 的 scenario-conditioned paired BDD

Stage6 unpaired-first 协议继续用于异源实路日志，不能被 paired BDD 替代。但
nuPlan same-scenario simulation 是明确的 matched experiment；对这类数据只使用
pooled label permutation 会丢弃实验设计并让 cross-scenario heterogeneity 主导
零假设。

Stage7 M6 因此增加两类补充统计：

1. original embedding 上的 within-scenario pair label swap MMD；
2. pair-midpoint residual embedding 上的 within-scenario pair label swap MMD。

二者的 estimand 是“控制 scenario 后的 planner effect”。原 Stage6 marginal
BDD 仍必须保留并报告，它回答“不使用 pairing 时的总体边际分布差异”。original
space 与 residual space 的 MMD² 因数据变换和 bandwidth 不同，不得直接比较数值
大小。

工具：`tools/stage7_m6_scenario_conditioned_bdd.py`。

完整设计与防泄漏要求见
`docs/stage7_cross_domain_style_sensitive_training_protocol.md`。

### 18.1 M6.1 冻结说明

Matched simulation 的 primary estimator 冻结为：原始64维 embedding、
single-RBF biased V-statistic MMD²、精确 pooled positive off-diagonal median
bandwidth、固定 bandwidth、within-pair label swap、100000 permutations 和
plus-one p-value。Pair-midpoint residual BDD 是 secondary，不能替代原空间
primary，也不能按 MMD² 数值与 primary 排名。

当前45对只用于方法开发。锁定确认必须使用新的 log/scenario-disjoint pairs、
独立冻结 selection config，并保持 planner treatment 参数不变，再复用冻结
估计器。M6.1 同时要求 complete-pair/token/planner/row/horizon
审计、Tier A 与 Tier A+B 敏感性分析，以及 fallback/ambiguous-rate 与 embedding
pair distance 的相关性检查。

这些约束不改变真实部署日志的 Stage6 unpaired-first 主协议：两套软件运行十天而
无法获得相同场景时，应先按预处理、静态 ODD 和动态 interaction exposure 做匹配/
重加权与 cluster-aware resampling，再报告 task-conditioned BDD；task frequency
shift 与 within-task behavior shift 必须分开。驾驶结果本身定义的 outcome bins
主要用于解释，不能作为无条件 matching covariates。

### 18.2 M6.2 的 task timing 约束

M6.2 确认性 task slices 只能来自 planner rollout 前已知的 nuPlan
`scenario_type`。这与 Stage6 实路日志原则一致：matching/stratification 使用
pre-treatment ODD 和 interaction exposure；由软件行为产生的 lane change、
hesitation、hard braking 等 outcome 只能做结果解释或明确标记的敏感性分析。

当前45对中五个冻结 task 各只有8–9对，低于12对运行下限；只有 high-motion
dynamics 的 learned-embedding paired BDD 在开发集上通过 Holm correction。不能
把该开发集结果外推为所有任务均有明显差异，也不能用总体 paired BDD 替代
task-specific coverage。

### 18.3 M6.3 功效配额与 Stage6 部署边界

M6.3 的每任务60个完整 pairs（20%损耗后75个 gross pairs）只适用于 nuPlan
same-scenario paired locked confirmation。它来自0.75 pilot-effect 假设下五任务
Holm family 的 simultaneous power 规划，不得迁移为 Stage6 十天异源实路日志的
逐样本配对要求。

Stage6 仍以 unpaired、exposure-matched/reweighted、cluster-aware inference 为
主。实际部署的样本量必须按 log/day/route cluster、任务暴露率、有效样本量和目标
最小可检测 BDD 另行规划；软件间 task-frequency shift 与 within-task behavior
shift 分开报告。M6.3 可借鉴的是“解盲前冻结任务、效应假设、功效目标和停止规则”，
不是其具体60/75数值。

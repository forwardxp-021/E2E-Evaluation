# Stage 6A 非配对实路风格漂移评估协议（Unpaired-First）

## Stage 6S-v2：interaction benchmark development与confirmation freeze（2026-08-12）

Issue #261在Stage6S-v1 limitation之后重新回到扩大nuPlan inventory，完全以pre-treatment front
exposure、initial gap、ego speed和closing pressure筛选。15,779个候选中301个eligible；24个
development pair的48条official rollout全部成功。planner只改变headway 0.8/2.2秒与minimum gap
0.5/2.5米，speed schedule、accel/decel能力和lateral配置相同。

Development realized mechanism通过：短减长的median mean-speed差为+0.259 m/s、RMS accel差为
+0.225 m/s²；front gap差为-4.284 m（91.7%方向一致），finite THW差为-2.660 s（100%方向一致）。
closing/following acceleration response未单独通过，因此不得将其写成已建立机制。THW冻结为pair内
median再跨pair median，只保留`0 < THW < 20 s`有限值并排除999/sentinel/cap。

机制通过后，从未参与development的日志中outcome-blind冻结80-pair/15-log confirmation roster。
development log overlap、development scenario overlap和Stage6S-v1 token overlap均为0。roster冻结后
不可根据old64、ego13或new64表现修改。当前未运行confirmation rollout、embedding/BDD、checkpoint
训练或正式模型评估。完整中文证据见
`docs/stage6s_v2_interaction_benchmark_confirmation_report_zh.md`。

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

## 19. Stage 6D：异源实路软件版本 BDD（Issue #242）

Stage 6D 面向整车厂真实发布流程：软件版本 A 和 B 在不同城市、路线、日期和交通
条件下分别路试，通常无法获得同场景配对。它不把两批日志的边际 BDD 直接解释为
软件风格差异，而是同时报告两个不同 estimand：

1. **raw observed-mixture BDD**：两批实际采集分布的总体差异，包含软件行为变化和
   ODD / exposure composition shift；
2. **common-support standardized BDD**：只在两版本都有数据的 pre-treatment 分层
   单元内，将两组重加权到相同的 equal-group pooled reference distribution 后的
   embedding MMD²，更接近“在可比较场景构成下的软件版本行为差异”。

二者都属于观察性描述指标。标准化只能控制已记录并正确建模的 covariates，不能消除
未观测混杂，因此不能解释为因果效应、安全认证或版本优劣。

### 19.1 冻结设计与防泄漏

运行前必须提供设计 JSON，并显式冻结：

- `group_column`、A/B 标签、行 ID 和 cluster ID；
- categorical exact cells，例如城市、道路等级、天气或时段；
- continuous pooled quantile bins，例如限速、交通密度和 pre-treatment exposure；
- pre-treatment task slices；
- support、ESS、最大权重和最少 cluster 数门槛；
- `post_treatment_columns` 排除清单。

所有 matching covariates 和 task slices 都必须声明 `timing=pre_treatment`。由待比较
软件产生的急刹、迟疑、变道结果、舒适性或 planner outcome 不得进入 matching；这些
字段只能作为结果解释层。工具发现 timing 不合规或字段重叠时 fail closed。

### 19.2 共同支持与标准化权重

对每个共同分层单元 `c`，先计算两组内部频率 `p_A(c)` 和 `p_B(c)`，再定义共同目标：

`q(c) = 0.5 * [p_A(c) + p_B(c)]`。

组 `g` 中该单元的样本权重与 `q(c) / p_g(c)` 成比例，并在组内归一化。只保留 A/B
都出现的单元。工具必须输出每组共同支持比例、有效样本量 `ESS=1/sum(w_i^2)`、ESS
比例、最大权重相对均匀权重的倍数、cluster 数和 covariate balance。任一冻结门槛失败
时状态为 `NOT_COMPARABLE_INSUFFICIENT_COMMON_SUPPORT`，不得把 BDD 当成可比较结论。

### 19.3 BDD、任务分解与不确定性

- raw 和 standardized 使用同一 scope 内冻结 bandwidth 的 single-RBF biased MMD²；
- overall BDD 与各 pre-treatment task-conditioned BDD 分开报告；
- task frequency shift 单独报告，不能与 within-task behavior shift 混为一谈；
- bootstrap 以 log / route / day / vehicle 等独立采集 cluster 为重采样单位，并在每次
  重采样内重新构造共同支持和权重；
- 当前 95% 区间为 observed MMD² 加减 `1.96 * cluster-bootstrap SE`，下界截断为 0；
- 当前里程碑不输出伪精确的 universal p-value。上线告警阈值必须用独立的同版本 A/A
  历史窗口校准，而不能把任意绝对 MMD² cutoff 当作通用标准。

### 19.4 工具与产物

工具：`tools/stage6d_unpaired_version_bdd.py`。

主要产物：

- `common_support_cells.csv`；
- `standardization_row_weights.csv`；
- `covariate_balance.csv`；
- `task_frequency_shift.csv`；
- `overall_bdd_summary.csv` 和 `task_bdd_summary.csv`；
- `cluster_bootstrap_mmd_samples.csv`；
- `stage6d_unpaired_version_summary.json`；
- `stage6d_reproducibility_provenance.json`；
- `stage6d_unpaired_version_report.md`。

nuPlan 310-pair confirmation embedding 的接口冒烟结果为 310/310 行、20/20 cells 全部
处于共同支持，A/B ESS ratio 均为 1.0，raw 与 standardized overall MMD² 均为
`0.0044865829`。这是预期的 paired/balanced 输入自洽检查，只证明工具接口和权重层
工作正常，不构成异源实路软件版本的有效性证据。生产验证下一步必须使用真实 A/A
跨路线、跨日期窗口和已知 A/B 版本对照来完成误报基线与检测功效标定。

## 20. Stage 6E：公开数据 A/A 标定与不配对发布模拟（Issue #243）

在公司真实路试数据暂不可用时，Stage 6E 使用现有 nuPlan 两套冻结 planner 的 paired
rollout 构造 **public field-release emulation**。原始 pair 只用于输入完整性审计和 paired
oracle；每次伪发布都以 log 为不可拆分 cluster，将 A/B 分到完全不同的 logs 和 scenario
tokens，再使用 Stage 6D 的共同支持标准化计算 BDD。

### 20.1 冻结实验族

- `AA_CALIBRATION_ASSERTIVE` / `AA_CALIBRATION_CONSERVATIVE`：每套 planner 各100次，
  合并后的 standardized BDD 95% empirical quantile 作为 scope-specific threshold；
- `AA_EVALUATION_*`：使用独立 seed stream 各100次，估计 held-out pseudo-release
  false-positive rate；
- `AB_ASSERTIVE_TO_CONSERVATIVE` 与反方向：各100次，估计已知 planner style change
  在主动丢弃 pairing 后的 detection rate；
- overall 是 primary emulation scope；五个 pre-treatment tasks 是没有 multiplicity
  control 的 diagnostic scopes。

所有600次 trial 均使用257个独立 logs 的近似平衡二分；A/B log overlap 和 scenario
token overlap 必须为0。固定 bandwidth 在读取任何 release labels 前由完整输入按 scope
冻结。共同支持不通过的 trial 标记为 `NOT_COMPARABLE`，不进入阈值或 rate 分母。

### 20.2 权威结果

总体 A/A threshold 为 standardized MMD²=`0.00994295`。独立 A/A evaluation 为
7/200 false positives，即3.5%（Wilson 95% CI `[1.7%, 7.0%]`）；A/B 为70/200
detections，即35.0%（`[28.7%, 41.8%]`）。两区间分离，A/B rate 是 A/A 的10倍，
说明版本信号在不配对条件下仍可检测；但35%远低于可靠单次发布报警所需的高敏感性，
所以结论冻结为 `AB_SEPARATED_FROM_AA_BUT_SINGLE_RELEASE_SENSITIVITY_LIMITED`。

任务诊断中 lane-change 最强：A/A=6.0%，A/B=53.5%；following=16.0%、high-motion=
12.5%、dense/vulnerable=19.0%。stop/go 的 A/B=2.0%，没有超过 A/A=3.6%，不能声称
该任务在当前样本量下具有版本检测能力。任务结果没有多重性控制，只能用于定位和后续
扩样设计，不能替代 overall primary。

### 20.3 论文和工程边界

该结果支持“公开闭环基准中的异地、异路线软件发布模拟存在可检测版本信号”，不支持
“已在公司真实版本上验证”或“任意一次发布都能稳定识别”。重复 trials 虽使用独立 seed
streams，但仍复用同一有限场景池，不能视为200次独立实路发布。公司数据可用后必须重新
做 A/A 标定；当前 absolute threshold 不得直接迁移到量产。

工具：`tools/stage6e_calibrate_unpaired_release.py`。权威输出：
`outputs/stage6e_nuplan_release_emulation_v1/`。

## 21. Stage 6F：不配对 BDD 实证功效曲线（Issue #244）

Stage 6F 回答 Stage 6E 暂未解决的问题：35% detection 是不是仅由样本量不足造成，以及
现有公开场景池能否达到面向单次发布的80% detection target。工具
`tools/stage6f_unpaired_power_curve.py` 在每版本40、60、80、100、125、150个目标场景
上分别重跑完整的 A/A calibration、A/A evaluation 和双方向 A/B evaluation。

### 21.1 每个样本量必须独立标定

每个样本量包含600个 pseudo-release trials：两套 planner 各100个 A/A calibration、
各100个 A/A evaluation，以及两个 A/B 方向各100个。所有 trials 仍以完整 log 为
不可拆分 cluster，A/B log/token overlap 为0。每个样本量使用自己的 A/A 95% empirical
threshold，禁止把 n=150 的 threshold 复用到 n=40。由于一个 log 可包含1–2个场景，
实际 n_A/n_B 允许相对目标±1并写入 audit。

冻结的 sufficiency gate 同时要求：

- A/B detection Wilson 95% CI 下界不低于80%；
- A/A false-positive Wilson 95% CI 上界不高于5%。

overall 是 primary；task curves 没有 multiplicity control，只是 diagnostic。
六个样本量的 overall threshold 均有200个有效 calibration trials。n=40 时 following、
stop/go、dense/vulnerable 的有效 task calibration trials 分别只有32、49、44，低于50
门槛；这些 n=40 task estimates 必须标记为 insufficient，不能解释。

### 21.2 权威总体曲线

| 每版本目标场景 | A/A threshold | A/A false positive | A/B detection |
| ---: | ---: | ---: | ---: |
| 40 | 0.041835 | 5.0% | 7.0% |
| 60 | 0.026537 | 5.5% | 10.5% |
| 80 | 0.020156 | 2.5% | 12.0% |
| 100 | 0.016624 | 1.5% | 11.5% |
| 125 | 0.012839 | 4.5% | 17.0% |
| 150 | 0.009948 | 7.0% | 35.0% |

有限 Monte Carlo trials 和同一场景池复用使曲线不保证逐点单调，例如 n=80 到 n=100
存在轻微回落；不得事后平滑后把平滑曲线当观测证据。n=150 的 A/B detection Wilson
95% CI 为 `[28.7%,41.8%]`，远低于80% target；A/A 为7.0%，CI `[4.2%,11.4%]`，
其上界也没有通过5%门。因此权威状态为
`TARGET_NOT_REACHED_WITH_AVAILABLE_PUBLIC_LOGS`。

n=150 任务诊断中 lane-change detection=40.5% 仍最强；following=15.5%、
stop/go=5.0%、high-motion=7.0%、dense/vulnerable=17.0%。这些结果不支持用单一任务
替代总体 primary，也不支持当前 stop/go/high-motion 具有稳定版本检测能力。

### 21.3 禁止超范围外推

当前310个唯一场景在 log-disjoint A/B 下最多支持约155个场景/版本。Stage 6F 不拟合
logistic/linear 外推，也不报告“达到80%精确需要多少场景”。若扩展到200/250/300/400
场景/版本，至少需要400/500/600/800个唯一场景池；相对当前310个场景分别至少新增
90/190/290/490个，并继续满足完整 log、pre-treatment ODD 和任务覆盖要求。这些只是
下一轮实证档位所需的集合规模，不是达到80% detection 的功效保证。

权威输出：`outputs/stage6f_nuplan_power_curve_v1/`。

Stage6 仍以 unpaired、exposure-matched/reweighted、cluster-aware inference 为
主。实际部署的样本量必须按 log/day/route cluster、任务暴露率、有效样本量和目标
最小可检测 BDD 另行规划；软件间 task-frequency shift 与 within-task behavior
shift 分开报告。M6.3 可借鉴的是“解盲前冻结任务、效应假设、功效目标和停止规则”，
不是其具体60/75数值。

## 22. Stage 6G：公开 nuPlan 发布池扩展冻结（Issue #245）

Stage 6F 表明310个唯一场景最多只能实证到每版本约150个样本，且该档位 A/B
detection 仍只有35%。Stage 6G 因此先扩展公开闭环池，再运行双 planner rollout；本阶段
只增加可供后续不配对发布模拟使用的唯一场景，不读取或重算 embedding、BDD、effect
size，也不依据 planner outcome 改变候选顺序或停止。

### 22.1 冻结任务定义和公开库存边界

五类 task 继续沿用 M6.2 在 rollout 前冻结的 nuPlan `scenario_type` 映射。排除现有310个
成功 token 后，原定义的 lane-change 虽有50个候选，但只有11个通过官方 scene-position
门槛。该限制不能通过加入转弯等不同语义标签来隐藏。因此新增490个主场景的解盲前配额
冻结为：

| task | 现有成功 | Stage 6G 新增主集 | 合并目标 |
| --- | ---: | ---: | ---: |
| following_interaction | 60 | 122 | 182 |
| lane_change | 60 | 11 | 71 |
| stop_go_control | 67 | 115 | 182 |
| high_motion_dynamics | 60 | 122 | 182 |
| dense_or_vulnerable_interaction | 63 | 120 | 183 |
| **合计** | **310** | **490** | **800** |

该表中的800是“主集全部技术成功时”的池规模目标，不是已经获得的成功数，也不是达到
80% detection 的功效保证。lane-change 的71个上限应在论文中作为公开数据覆盖限制报告。

### 22.2 outcome-blind 选择和 cluster 约束

- 排除现有310个 scenario token，但不排除其完整 log；后续 A/B 伪发布继续以整个 log 为
  不可拆分 assignment / resampling cluster，因此同一 log 内多个窗口不会跨版本泄漏；
- 候选按 task、log、token 和冻结 salt 的 SHA-256 排序；容量最小的 task 先占用 log cap，
  这是 pre-treatment 库存规则，不使用 planner 结果；
- 现有池、Stage 6G 主集和预备集合计每 log 最多3个场景；后续统计仍必须按 log 重采样，
  不得把800个窗口当作800个独立 cluster；
- 每个选中 token 在 rollout 前检查 DB 存在、token 定位和官方 scene position；数字型
  Hydra token 使用显式带引号的 `actual_nuplan_token`；
- 主集490个全部按冻结顺序尝试。100个预备场景仅覆盖主集记录的同任务技术失败；没有
  reserve-eligible failure 时 runner fail closed。

### 22.3 当前已验证状态

权威 freeze 目录为 `outputs/stage6g_expanded_release_pool_freeze_v1/`。冻结结果为 READY：
490个主场景、100个预备场景全部通过 scene-position 预检；主集与现有池 token overlap=0，
主集与预备集 overlap=0，三者合计最大每 log=3。真实 smoke 已完成第1个 lane-change
场景的两套 planner，严格 log/token alignment 和完整 pair audit 均通过；其余489个在
全量 runner 完成前仍是 pending。

工具：`tools/stage6g_freeze_expanded_release_pool.py`、
`tools/stage6g_run_expanded_release_pool.py`。配置：
`configs/stage6g_expanded_release_pool.json`。

### 22.4 Stage 6G 最终执行结果

新增主集490/490全部成功，技术失败、pending和running均为0；平均端到端耗时
33.15秒/场景，累计场景耗时约4.51小时。成功计数严格等于冻结配额：following=122、
lane-change=11、stop/go=115、high-motion=122、dense/vulnerable=120。100个技术预备场景
均未使用。Issue #245 因此按完成关闭；context、embedding和扩展功效曲线转入Issue #246。

## 23. Stage 6H：800-pair embedding 池与扩展实证功效曲线（Issue #246）

Stage 6H 不改变 Stage 6G 的场景选择。它把新增490个 official rollout 转换成与原310个
确认场景完全相同的 Stage5D 83D context，并用同一个 Waymo-trained encoder 生成64D
embedding；随后只在通过模型、schema、pair和行对齐审计后合并为800 pairs / 1600 rows。

### 23.1 表征一致性门槛

- 新增490场景必须逐一重跑 Stage7C audit，两planner、严格log/token alignment、非伪
  rollout、tensor和official msgpack均通过；
- context继续使用`lane_aware_with_geometric_fallback`，并启用与原310相同的projection和
  strict-filter diagnostics；Mac运行必须显式提供本地tuPlan Garage `PYTHONPATH`；
- checkpoint固定为Waymo训练的
  `outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/context_gru_stage5d_balanced_v2/best_model.pt`；
- 合并前必须比较checkpoint SHA-256、Stage5D channel/slot schema、metadata列顺序、64D
  embedding shape和finite状态；旧/新scenario token overlap必须为0；
- 合并后重新建立`global_row=0..1599`和`scenario_index=0..799`。每个token必须恰有两行、
  两个冻结planner，并在pair内保持log、map和scenario_type不变。

### 23.2 400/版本的完整log分配

扩展曲线的档位冻结为200、250、300和400场景/版本，每档独立运行200次A/A calibration、
200次held-out A/A evaluation和200次双方向A/B evaluation。最大档位会用完800个场景，
原Stage 6F“先把log数量切成两半、再取prefix”的策略无法保证两边同时得到400个场景。

Stage 6H 因此新增`sequential_full_log_pool_v1`：在完整随机log序列中先构造A的目标log集，
再从所有剩余log构造B；两边仍是完整log、互不重叠，并在24个预冻结候选中按support
composition和样本量误差选择最低分方案。旧Stage 6F配置默认继续使用
`half_log_pool_prefix_v1`，历史行为不变。

### 23.3 统计解释边界

每个样本量继续单独冻结A/A 95% empirical threshold。充分性要求A/B detection Wilson
95%下界不低于80%，且A/A false-positive Wilson 95%上界不高于5%。overall是primary；
task曲线没有multiplicity control，只用于诊断。即使400/版本通过门槛，也只支持当前有限
公开nuPlan池中的重复release emulation；不能替代公司真实A/A重标定，也不能把800个场景
解释为800个独立路试cluster。

### 23.4 Stage 6H 最终执行结果

新增490个rollout的Stage7C复审为490/490通过，生成的context为`[980,150,83]`、embedding
为`[980,64]`，均无non-finite值。它们与原310场景使用相同的83D Stage5D schema和同一
Waymo checkpoint（SHA-256
`909022f5df03a3f01c2149da6c9b44c613e955a4d816e8ec4d5862f39f8bf0cc`）。合并结果为
800个完整pair、1600行、489个log cluster；旧/新token overlap为0，冻结任务计数为
182/71/182/182/183。

扩展曲线的2400次release split全部得到精确目标样本量，log和scenario overlap均为0。
overall结果如下：

| 场景/版本 | A/A FPR（Wilson 95% CI） | A/B detection（Wilson 95% CI） |
| ---: | ---: | ---: |
| 200 | 8.0%（5.0%–12.6%） | 30.0%（24.1%–36.7%） |
| 250 | 6.5%（3.8%–10.8%） | 28.5%（22.7%–35.1%） |
| 300 | 3.5%（1.7%–7.0%） | 41.5%（34.9%–48.4%） |
| 400 | 5.0%（2.7%–9.0%） | 66.5%（59.7%–72.7%） |

因此状态为`TARGET_NOT_REACHED_WITH_AVAILABLE_PUBLIC_LOGS`：在公开库存允许的最大实证
档位400/版本，A/B detection点估计仍低于80%，Wilson下界也只有59.7%；A/A FPR点估计
为5%，但Wilson上界为9.0%，同样未通过冻结置信门槛。200到250的非单调波动保留原样，
因为每档分别标定threshold且都来自同一个有限公开池，不能事后平滑或据此外推400以上。

400/版本的task诊断检出率为：following 15.5%、lane-change 46.0%、stop/go 13.0%、
high-motion 63.0%、dense/vulnerable 11.5%。这些结果说明整体检出主要由high-motion和
lane-change贡献，但task曲线没有multiplicity control，不能作为独立确认性结论。权威输出
为`outputs/stage6h_nuplan_power_curve_800_v1/`。

## 24. Stage 6I：冻结可靠性分解与论文主张审计（Issue #247）

Stage 6I 不新增场景、不重算embedding或BDD，也不改变Stage 6H的threshold。它只读取
Stage 6H已冻结的summary和CSV，将“已发现版本信号”与“单次发布可靠性不足”拆成可以
直接审计的论文结论。工具为`tools/stage6i_build_reliability_evidence.py`，权威输出为
`outputs/stage6i_reliability_evidence_v1/`。

### 24.1 fail-closed输入边界

- embedding pool必须仍是800 pairs / 1600 rows / 489 log clusters，pair完整且finite；
- 必须存在200/250/300/400四档、6个冻结scope、6个experiment set、2400个release
  splits和14400个scope-level trial rows；
- 所有split必须精确达到目标n，log overlap和scenario overlap均为0；
- sufficiency target固定为detection Wilson下界≥80%、FPR Wilson上界≤5%；
- 工具不得读取rollout/context/embedding数组，不重新标定threshold、不平滑曲线，也不
  估计400场景/版本以上所需样本量。

### 24.2 可靠性与方向结果

四个观测档位的A/B detection Wilson下界均高于对应A/A FPR Wilson上界，区间分离margin
依次为11.5、11.9、27.8和50.7个百分点。因此“已知planner风格差异在异log、异场景
release emulation中仍可被检测”获得公开数据范围内的支持。

但这与高可靠报警不是同一个结论。400场景/版本时false-negative rate仍为33.5%
（Wilson 95% 27.3%–40.3%），冻结sufficiency gate没有通过。两个A/B方向的检出率为
62%和71%，绝对差9个百分点；由于没有预冻结direction-equivalence gate，该差异只作
诊断，不声称方向完全对称。

### 24.3 论文主张边界

Stage 6I冻结以下主张状态：

| 主张 | 状态 |
| --- | --- |
| Waymo-trained embedding在nuPlan已知planner对比中保留风格信号 | `SUPPORTED_WITHIN_PUBLIC_BENCHMARK` |
| 不同log/场景的两个伪发布仍可检测已知版本风格差异 | `SUPPORTED_WITHIN_PUBLIC_RELEASE_EMULATION` |
| 单次发布达到80%检出且5%误报置信门槛 | `NOT_SUPPORTED` |
| absolute BDD threshold可跨样本量、ODD、车队和公司通用 | `NOT_SUPPORTED` |
| 已在整车厂真实软件版本路试上验证 | `NOT_EVALUATED` |
| 每个task都构成独立确认性版本证据 | `NOT_SUPPORTED_AS_CONFIRMATORY` |

因此论文可以把本工作定位为“跨数据集学习表征 + 异场景版本差异的可检测性方法与公开
闭环验证”，但必须把33.5%假阴性、有限公开池复用、未观测混杂和公司A/A重标定需求作为
核心限制。不能写成量产发布判定系统已经达到工程可靠性。

### 24.4 Stage 6I中文报告、术语和分任务BDD补充（Issue #248）

Stage 6I报告必须使用中文，并统一以下术语：

- BDD（Behavior Distribution Difference）是要评价的行为分布差异；
- MMD（Maximum Mean Discrepancy）是当前使用的核两样本统计方法；
- 报告的BDD数值是MMD²；
- 本项目没有定义MDD；旧讨论中若出现MDD，按MMD笔误处理。

新增`stage6i_task_definitions.csv`、`stage6i_task_scenario_classification.csv`、
`stage6i_task_bdd_magnitudes.csv`和`stage6i_planner_treatment_audit.csv`。冻结800-pair
池的五类数量分别为182、71、182、182和183，每个scenario token恰好归入一个仿真前
`scenario_type` task。

同场景paired-oracle的五个task MMD²分别为0.02478050、0.02878431、0.00523033、
0.01445332和0.01379180。400场景/版本的异log/异场景release emulation中，相应A/B
检出率为15.5%、46.0%、13.0%、63.0%和11.5%；对应A/A FPR为1.5%、3.5%、6.0%、
7.5%和7.0%。配对和非配对结果使用不同estimand与bandwidth，不允许直接比较MMD²
绝对大小。

`lane_change`的定义只是原始nuPlan
`scenario_type in {changing_lane_to_left, changing_lane_to_right}`，语义状态固定为
`SCENARIO_TYPE_SLICE_NOT_CONFIRMED_EGO_LANE_CHANGE`。现有分析没有用lane ID或车道
拓扑证明PDM控制自车实际完成变道，旧`lane_change_count_proxy`也会被弯道或局部坐标
漂移触发。此外assertive与conservative的`lateral_offsets`分别为`[-1.5,1.5]`和
`[-0.5,0.5]`，因此当前planner对比是纵向+横向混合处置，不能解释为纯纵向风格实验。

针对论文的纵向目标，下一确认性实验必须保持两版横向参数完全相同，仅改变headway、
min-gap、speed fraction、accel/decel等纵向参数；先以同场景配对设计验证敏感性，再进入
异log/异场景release emulation。扩大Waymo训练集只有在定向增加跟车、启停和强加减速
样本、保持log-disjoint划分并增加speed/accel/jerk/THW/gap辅助约束时才构成有针对性的
改进；单纯增加同分布普通巡航片段不构成充分方案。

## 25. Stage 6J：纯纵向PDM同场景配对确认（Issue #249）

### 25.1 研究问题与处置冻结

Stage 6J用于回答比Stage 6I更窄、也更符合论文目标的问题：Waymo训练的embedding能否
检测nuPlan闭环仿真中人为设置的典型纯纵向风格差异。新planner为：

- `pdm_closed_assertive_longitudinal_v1`；
- `pdm_closed_conservative_longitudinal_v1`。

两者`lateral_offsets`均固定为`[-0.5,0.5]`，只允许以下纵向参数不同：
speed-limit fraction、fallback target velocity、min gap、headway、accel max和decel max。
冻结工具会比较Stage7C中的实际profile参数；任何横向差异、未知参数差异或应不同的纵向
参数意外相同都会fail closed。

### 25.2 场景estimand

Stage 6J复用M6.5已锁定confirmation ledger，但只读取scenario token、log、scenario
type和DB路径，不读取embedding、BDD或planner outcome数组。主分析包含：

- following interaction：60 pairs；
- stop/go control：67 pairs；
- longitudinal high-motion：56 pairs，只含`high_magnitude_speed`和
  `medium_magnitude_speed`。

总计183个same-scenario pairs、366条rollout、156个独立log。主分析明确排除lane-change、
dense/vulnerable和`high_lateral_acceleration`，避免再次把横向或场景标签信号混入纯
纵向主张。场景冻结状态为`FROZEN_BEFORE_PURE_LONGITUDINAL_ROLLOUTS`。

### 25.3 分阶段判定顺序

冻结的判定顺序为：

1. 先验证realized longitudinal kinematic contrast，包括speed、accel、jerk、THW和gap；
2. 再计算same-scenario paired overall BDD；
3. 再报告following、stop/go和longitudinal high-motion分任务BDD；
4. 使用log cluster robustness评估有限独立log影响；
5. 只有paired gate通过后，才进入异log/异场景release emulation。

如果realized kinematic gate失败，应调整PDM纵向处置，而不是归因于embedding；如果
kinematic gate通过但paired BDD弱，才构成扩大或重训Waymo纵向表征的直接依据；如果
paired BDD强而unpaired检出弱，主要问题应归因于场景混杂、独立log数量和统计聚合。

### 25.4 Mac真实smoke结果

首个冻结following场景`6b5a9da8c0b353b9`已完成两个official nuPlan rollouts：

```text
official success: 2 / 2
trajectory rows: 298
tensor shape: (1, 2, 149, 8)
same-log alignment: PASS
strict-token alignment: PASS
pseudo rollout: false
elapsed: about 33 seconds
```

两个输出planner metadata均记录`style_scope=pure_longitudinal_closed_loop_planner`，且
Hydra overrides中的`lateral_offsets=[-0.5,0.5]`逐字一致。该smoke只验证环境、配置和
轨迹导出，不提供BDD结论。全量366条rollout必须先具备可审计的批处理、进度、失败分类和
断点续跑，再通过显式execute确认启动。

### 25.5 可断点续跑全量执行（Issue #250）

`tools/stage6j_run_pure_longitudinal_rollouts.py`实现逐场景隔离的183场景/366 rollout
批处理。启动前必须复核freeze manifest状态、locked CSV SHA-256、planner parameter
fingerprints、连续collection order、唯一token、DB/log对应关系、Stage7C工具和
nuPlan/tuPlan commits。

工具默认dry-run；真实执行同时要求`--execute`和精确的
`--confirm_locked_scenarios_sha256`。每个场景创建独立attempt，不覆盖旧输出；
`--resume`重新审计已成功场景的2/2 official commands、trajectory、same-log、
strict-token和tensor完整性后跳过。失败场景不会自动重试，只有人工复核后显式
`--retry_failed`才生成新attempt。

批处理持续原子更新`batch_state.json`和`batch_scenario_status.csv`，并追加
`batch_events.jsonl`。全量运行使用Mac`caffeinate`，不读取embedding或BDD，也不允许
根据中途effect size提前停止。2026-08-10正式启动时前2个场景均成功、0失败，实测约
39秒/场景，初始剩余时间估计约2小时。

全量最终结果为183/183场景成功、366/366 official rollout、0失败、0 pending。collection
order 110和111首次运行时，16位纯数字或科学计数法样式token被Hydra解释为数值而不是
字符串；Stage 6J runner现仅对这类token增加Hydra字符串编码，并保留原始token用于严格
alignment审计。两场景经人工复核后使用`--retry_failed`分别在attempt 002重跑成功。

### 25.6 rollout统一视图与运动学门禁输入（Issue #251）

`tools/stage6j_prepare_pure_longitudinal_view.py`在不读取embedding、BDD或effect size的
前提下，将183个隔离Stage7C输出合并为Stage5D context builder可直接读取的统一视图。
合并前重新校验freeze/batch SHA-256、planner parameter fingerprints、183行冻结顺序和
任务构成；对每个场景重新审计2/2 official success、same-log、strict-token、tensor shape
与两个msgpack。任何一项失败都会fail closed，不能跳过场景或改变冻结样本。

2026-08-10实际复核结果为183/183通过、0失败、366条rollout、156个独立log，统一ego
张量shape为`(183,2,150,8)`，有效trajectory rows为54612。official rollout目录使用
symlink引用原始隔离输出，不复制或修改原始仿真文件。随后以
`lane_aware_with_geometric_fallback`启动5邻车Stage5D上下文构建，并生成projection和
strict-filter诊断。运动学门禁必须在该上下文完整通过后计算；门禁结论形成前仍不得读取
embedding或BDD。

运动学门禁配置`configs/stage6j_kinematic_gate.json`已在读取本批context结果前冻结，
SHA-256为`140ea34f505537e99d5e6726a884015304cda4ed1034eb1e66eb5f47284077f9`，并记录在
Issue #251。主门禁使用same-scenario的A-B配对差，并按`log_name`做10000次cluster
bootstrap：平均速度差95% CI下界必须不低于0.5 m/s，且RMS加速度差95% CI下界必须
不低于0.1 m/s²；两个指标必须同时通过。jerk、yaw-rate、THW、front distance和front
exposure为支持性诊断，不参与主门禁。这样可以先证明PDM处置确实实现了典型纵向差异，
避免在处置本身过弱时错误归因于Waymo embedding。

### 25.7 运动学门禁与纯纵向paired BDD结果

Stage5D context实际构建183/183场景，耗时14分37秒；输出366行、83维context，
`validation.pass=true`、map query成功、lane info count=2541、全局geometric fallback
约9.66%，且context无非有限值。运动学主门禁两个指标均通过：

- 平均速度A-B=0.9147 m/s，log-cluster bootstrap 95% CI为[0.7578,1.0784]；
- RMS加速度A-B=0.1816 m/s²，95% CI为[0.1456,0.2175]。

因此可以排除“仿真没有实现足够强的纵向处置”这一前置失败。支持性结果还显示总体RMS
jerk增加0.2279 m/s³；平均front distance减少1.6016 m，其cluster CI为
[-3.0192,-0.1463]。THW仅136个pair有限且CI很宽，按冻结规则不作为主门禁或核心结论。

BDD配置`configs/stage6j_paired_bdd_analysis.json`也在读取本批BDD数值前冻结，SHA-256为
`4a0e330e7148ac10dda61c146f1d81d7b447548ff2b654825ade063fb12ed613`。使用与既有
Stage5/6完全相同的Waymo checkpoint（SHA-256
`909022f5df03a3f01c2149da6c9b44c613e955a4d816e8ec4d5862f39f8bf0cc`）、原始64D
embedding、single-RBF biased MMD²、pooled exact median bandwidth和100000次同场景
pair内label swap。结果为：

| scope | pair | BDD / MMD² | raw p | Holm p | 结论 |
|---|---:|---:|---:|---:|---|
| overall primary | 183 | 0.00500090 | 0.0000099999 | 不适用 | reject |
| following | 60 | 0.01706723 | 0.00064999 | 0.00129999 | reject |
| stop/go | 67 | 0.00537483 | 0.03300967 | 0.03300967 | reject |
| longitudinal high-motion | 56 | 0.01358617 | 0.0000099999 | 0.0000299997 | reject |

这直接支持论文的窄主张：Waymo训练的模型能够在nuPlan闭环仿真中检出人为设置、已由
运动学门禁确认的典型纯纵向风格差异。MMD²绝对值约0.005不能单独解释为“太小”；它
受embedding尺度和kernel bandwidth影响，本实验的可检出性由预冻结的paired
randomization null判定。结果仍不支持异log/异场景单次release的高可靠检出率、通用
BDD阈值或真实整车厂验证；183场景也来自先前技术成功ledger，不能包装为全新独立场景
confirmation。

## 26. Stage 6K：纯纵向处置强度—检出能力剂量曲线（Issue #252）

### 26.1 研究问题与必要性

Stage 6J证明了100%端点可以检出，但单个raw BDD数值不能回答“多大的风格差异才可检出”。
Stage 6K因此冻结0%、25%、50%、75%、100%五档，估计在相同183个场景、相同Waymo
checkpoint和相同kernel下，运动学处置强度与BDD证据怎样随剂量变化。该实验的目标是给出
本协议内的最小可检出剂量和标定曲线，不建立跨Waymo/nuPlan、跨checkpoint或跨kernel的
通用raw BDD阈值。

### 26.2 处置与样本冻结

0%固定为`pdm_closed_conservative_longitudinal_v1`，100%固定为
`pdm_closed_assertive_longitudinal_v1`。25%、50%、75%对六个纵向IDM参数逐项线性插值：
speed-limit fraction、fallback target velocity、minimum gap、headway、最大加速度和
最大减速度。五档均固定`lateral_offsets=[-0.5,0.5]`，因而不引入横向处置。

0%同profile下限和100%端点复用Stage 6J；新增仿真只运行25%、50%、75%。场景固定为
Stage 6J的183个同场景pair，三档各183个，共549个场景×剂量任务和1098条official
rollout。冻结清单SHA-256为
`4bbfa3adb23c5e3e090c3d5a66f636cb9400d059257c987709dda55056980b26`。
冻结步骤不读取新增剂量的embedding、BDD或effect size，也不允许结果出来后换场景、删剂量
或显著后提前停止。

### 26.3 预冻结判定规则

每档首先验证实现的speed、acceleration、jerk、THW和gap差异，再使用相同64维embedding、
single-RBF biased MMD²、pooled exact median bandwidth与100000次pair内label swap计算
paired BDD。每档必须报告raw MMD²、null q95、`BDD/null q95`、null标准化`Z_BDD`和
paired p值；task结果使用Holm校正。

“最小可检出剂量”定义为同时通过冻结运动学处置门禁且paired BDD `p<0.05`的最小非零
剂量。剂量—运动学和剂量—BDD的Spearman有序趋势作为标定诊断。raw BDD不与Waymo验证中
约0.5的retrieval/相关性指标混用，也不设跨协议通用绝对阈值。

### 26.4 启动前验证

冻结审计实际得到183个场景、156个log、task计数60/56/67，三个中间profile的参数插值与
横向一致性全部通过。runner dry-run复核549个任务和1098条rollout。25%、50%、75%各1个
真实smoke均成功，耗时34.197、33.341、33.429秒；每个smoke均为2/2 official success，
same-log和strict-token alignment通过。三场景平均33.656秒，据此全量串行初始估计约
5小时8分钟；实际ETA以`batch_state.json`滚动均值为准。

### 26.5 全量完成与解盲前补充

全量最终为549/549 `SUCCEEDED`、0 failed、0 pending，共1098/1098条official rollout。
三档各183个相同场景，156个独立log；唯一重试任务order 391在attempt 002成功，最终仍为
严格2/2 official、same-log和strict-token alignment通过。

在读取25/50/75%新增embedding或BDD之前，另行冻结
`configs/stage6k_preanalysis_addendum.json`，manifest状态为
`FROZEN_BEFORE_NEW_DOSE_EMBEDDING_OR_BDD_READ`。该补充不覆盖原rollout freeze，也不改变
场景、planner或输出。新增冻结规则为：

1. 25/50/75/100%四个overall检验构成一个Holm family；
2. 4剂量×3个pre-treatment task共12项构成一个Holm family；
3. 最小可检出名义剂量必须同时通过实现运动学门禁与overall Holm p<0.05；
4. 同log整体label flip仅为supplementary cluster sensitivity；
5. lane fallback/ambiguity是post-treatment描述性变量，禁止用于删样本、重加权或替代primary。

### 26.6 实现运动学剂量

三档统一view均为183场景、366条rollout、54612行轨迹；三档context均为366行、83维，
validation PASS、无非有限值且场景—planner严格对齐。实现运动学结果为：

| 名义剂量 | Δ平均速度 m/s | 单侧95%下界 | ΔRMS加速度 m/s² | 单侧95%下界 | 门禁 |
|---:|---:|---:|---:|---:|---|
| 25% | 0.2546 | 0.2109 | 0.0361 | 0.0233 | PASS |
| 50% | 0.4464 | 0.3517 | 0.0770 | 0.0572 | PASS |
| 75% | 0.6372 | 0.5262 | 0.1279 | 0.1034 | PASS |
| 100% | 0.9147 | 0.7873 | 0.1816 | 0.1508 | PASS |

速度、RMS acceleration与RMS jerk对四档的描述性Spearman rho均为1.0。由此可以排除
“中间参数档没有在rollout中形成纵向行为差异”的解释；但名义参数插值仍不能称为线性真实
车辆风格强度，THW/front-distance在部分档位有限pair不足，只作支持性描述。

### 26.7 BDD剂量—响应结果

三档新增embedding均使用与Stage 6J相同的64D Waymo checkpoint，checkpoint SHA-256为
`909022f5df03a3f01c2149da6c9b44c613e955a4d816e8ec4d5862f39f8bf0cc`；无padding、无
schema变化、无非有限值。总体结果为：

| 名义剂量 | BDD/MMD² | paired-null q95 | BDD/q95 | Z_BDD | raw p | 四档Holm p |
|---:|---:|---:|---:|---:|---:|---:|
| 25% | 0.00115612 | 0.00089633 | 1.290 | 3.649 | 0.00428996 | 0.00428996 |
| 50% | 0.00159972 | 0.00065468 | 2.444 | 9.594 | 0.0000099999 | 0.0000399996 |
| 75% | 0.00332234 | 0.00105009 | 3.164 | 12.548 | 0.0000099999 | 0.0000399996 |
| 100% | 0.00500090 | 0.00209751 | 2.384 | 9.192 | 0.0000099999 | 0.0000399996 |

四档均同时通过运动学门禁和overall Holm检验，所以本冻结协议内的最小可检出名义剂量是
25%。这不是说0.001156是通用BDD阈值；它只表示在本checkpoint、183个pair、本bandwidth和
本零分布下，observed BDD高于预冻结随机化基线。raw BDD随剂量单调增加，但Z_BDD在75%
达到最高，说明不同剂量的null尺度也会变化，进一步证明不能只看裸BDD。

同log整体翻转敏感性在四档均通过四项Holm：25% Holm p=0.00488995，其余三档均为
0.0000399996，因此primary结果不依赖把同一log内的多个场景当作完全独立label flip。

12项task×dose Holm显示明显异质性：25%只有`longitudinal_high_motion`检出；following在
75%和100%检出；stop/go在50%和75%检出，但100%在全12项校正后未检出。因此不能写成
“25%在所有纵向任务都可靠检出”，task结果只能作为次要诊断。

### 26.8 Lane-quality敏感性与最终解释

post-treatment敏感性使用全部732个dose-pair，不删样本。max pair fallback rate与embedding
pair L2距离存在中等正关联；task-adjusted rank相关在25/50/75/100%分别约0.430、0.242、
0.226、0.325，log-cluster 95% CI均不跨0。ambiguity关联较不稳定，25% task-adjusted和
100%两种分析的CI跨0。

这项结果不推翻预冻结primary，但构成必须披露的测量限制：context assignment质量与模型距离
共同变化，当前实验不能证明全部embedding信号都来自纯粹、可解释的人类纵向风格因子。论文
可以支持的主张是：

> 在固定checkpoint、同场景受控nuPlan闭环对照中，Waymo-only训练的embedding能够检出经
> 运动学确认的典型纯纵向planner风格差异；本协议内overall最小可检出名义处置为25%。

仍不能支持通用BDD阈值、所有task在25%均检出、异场景release单次高可靠性或真实OEM验证。
下一项最有价值的稳健性工作不是覆盖当前checkpoint，而是在保留冻结baseline的前提下增加
context-quality ablation：例如独立的ego-only/kinematic baseline、改进lane projection后重建
context、以及对新checkpoint进行同一剂量曲线复验。

## 27. Stage 6K context 修正与 Stage 6L representation 消融（Issues #253）

原26.6–26.8中的ego速度、加速度、jerk及100%端点仍有效；但原dose50/75 context因
`SimulationLog`反序列化运行时路径缺失而出现全零neighbor coverage。因此原dose50/75
完整context BDD、THW/gap和基于它们的lane-quality数值由修复版v2取代，旧目录保留为
历史证据，不覆盖。

修复版三档neighbor slot-frame coverage为17.14%/17.44%/17.37%。构建器新增
`--require_nonzero_neighbor_coverage`，Stage6L freeze也独立检查`neighbor_seq[...,0]`
非零；任何全零neighbor数据均fail closed。

Stage6L在结果读取前冻结A完整learned64、B同checkpoint邻车置零、C显式ego13D、D手工
交互+轨迹46D。权威结果显示task-dose Holm通过7/12、11/12、12/12、2/12；median
overall Z_BDD为7.539、11.066、21.082、5.384。A/B/C的最小overall检出剂量仍为25%，D为
50%。这支持受控纯纵向差异可检出，但不支持“当前interaction context增加了纵向敏感性”。

fallback与pair L2的关联在B中仍存在，故质量关联不是邻车通道特有的因果证据。context-v2
只能另建版本作为测量稳健性协议，不能为了让BDD变大而修改lane参数。完整结果见
`docs/stage6n_context_balanced_retraining_decision.md`。

## 28. Stage 6M context-balanced unpaired release reliability（Issue #254）

四种release-level estimand在聚合前冻结：raw marginal、固定task prevalence加权的
task-conditioned、map×scenario-type common-support context-balanced、task内平衡后再固定
权重聚合。每种方法和200/250/300/400样本量各自使用独立A/A calibration阈值。

n=400的A/B detection依次为63.0%/65.0%/66.5%/64.5%，A/A FPR依次为4.5%/5.5%/5.0%/
6.0%。context-balanced相对raw为+3.5pp，配对McNemar exact p=0.2478，不支持稳定提升。
28800行平衡审计中2个task scope-trial不可比，其余map/scenario-type最大加权比例差为
1.22e-15。说明平衡实现正确，但已测量scenario composition不是约33.5%假阴性的主要解释。

真实OEM流程仍必须是对应measurement system下的A/A calibration + pre-treatment task/ODD
审计 + unpaired A/B BDD + FPR/FNR可靠性评估。禁止使用通用raw BDD阈值，禁止用rollout后
行为或fallback做确认性matching。

## 29. Stage 6N checkpoint Go/No-Go（Issue #255）

预冻结规则触发`GO_PREPARE_SEPARATELY_VERSIONED_TRAINING_PROTOCOL`：ego13D明显强于完整
64D，neighbor-zero保持更高敏感性，且context balancing没有解决unpaired可靠性瓶颈。
旧Stage5D-balanced-v2保持冻结；GO只允许准备扩大Waymo纵向coverage、contrastive/ranking、
纵向auxiliary objectives与context dropout/quality mask的独立协议，不授权立即覆盖训练。

任何新checkpoint都必须重新完成Waymo validation、Stage6J/6K paired dose curve、Stage6M
unpaired A/A calibration、A/B detection和FPR/FNR tradeoff。planner name不得进入encoder。

## 30. Stage 6O 纵向敏感 64D 训练前冻结（Issue #256）

Stage 6O 在任何新训练结果出现前冻结 `ego16 + context/fusion48 = 64D` 的双分支方向、
hard-negative/near-boundary 采样、全部 loss 权重、3 个随机种子、训练预算、checkpoint 命名、
Waymo 域内非劣性、nuPlan paired dose/task 和 n=400 unpaired release 门槛。raw BDD、nuPlan
BDD/MMD、planner name 和 dose 均禁止进入训练、采样、选 epoch 或选 seed。

逐 shard 真实审计确认现有 Waymo full51 数据为35 shards、164871 windows、24426 scenarios；
train/val/test=`131998/16481/16392`，scenario 与 scenario-agent 跨 split 重叠均为0，全部训练
数组 shape/finite 和基线/证据 SHA-256 均通过。速度低/中/高为52098/66676/13224，stop/go
proxy为39949，steady-speed为10242，基础纵向覆盖充分。

但 front valid ratio 只有0或至少0.8：train free-flow=96649、sustained-following=35349，
intermittent-following=0，未达到预冻结的5000条门槛。因此正式状态为
`FROZEN_BLOCKED_WAYMO_COVERAGE_NOT_TRAINING`，不授权训练。后续Stage6Q raw audit已经确认
full51原始scenario内存在大量动态前车窗口，当前应先新建支持逐帧front identity与mask的
builder版本，而不是先扩展full51；旧Stage6O v1继续保持blocked，禁止事后降低门槛。

旧33D supervision的RMS acceleration median约2.72 m/s²、RMS jerk median/q90约
42.82/100.80 m/s³，显示差分噪声不可忽略。新协议固定从ego speed经5帧median filter重新
计算纵向目标，仅用train q01/q99 winsorize和train median/IQR标准化。完整协议见
`docs/stage6o_longitudinal_representation_training_protocol.md`。

## 31. Stage 6P Representation × Unpaired Release（Issue #257）

Stage6P原样复用800 pair、489 log和2400个release split，在每个representation×样本量内独立
A/A校准。full64、ego13、handcrafted46、neighbor-zero diagnostic在n=400的A/A FPR分别为
4.5%/1.5%/3.5%/1.0%，A/B detection为63.5%/100%/100%/100%。

ego13相对full64在n=400高36.5个百分点；相同200个A/B release中ego13-only=73、
full64-only=0，McNemar exact p=2.12e-22。n=200/250/300也均为ego13 100%检出，FPR为
2.0%/4.0%/3.5%。因此ego13的配对纵向优势在真正unpaired release下仍成立。

不同representation使用各自bandwidth与A/A q95，禁止跨representation比较raw MMD²。
ego13更可靠不等于context无用，只说明当前full64没有充分保留这组受控纵向运动学信号。

## 32. Stage 6Q Waymo raw interaction coverage audit（Issue #258）

对生成full51的51个原始TFRecord、24872个scenario逐条审计。3m主规则下182837个raw合格窗口
包含lead entry=31555、lead exit=44349、intermittent<0.8=54829、identity switch=29103、
free→closing→following=3465、following→free=43153。2m/4m敏感性下intermittent仍为
53448/51109，均远高于冻结门槛5000。

根因已确定：正式builder只在参考帧调用一次`assign_stage5d_slots`，固定front track再接受整窗
`min_valid_ratio=0.8` sanitize，结构上只能稳定输出空槽或>=0.8持续槽。故Stage6O的0不是原始
Waymo缺失，而是builder/window语义过滤。下一步必须先版本化修builder、重建数据并重新执行
Stage6O；当前不扩大Waymo、不训练Interaction-aware v2。完整中文报告见
`docs/stage6p_stage6q_representation_unpaired_and_raw_audit.md`。

## 33. Stage 6R Dynamic Interaction Builder v2（Issue #259）

旧builder的五个semantic slot均为窗口参考帧静态分配。v2改为逐帧semantic assignment，显式保存
valid、track-id、identity-switch和derivative-valid时间序列；identity切换处禁止跨agent差分，
ego与neighbor validity门槛分离。首次pilot因丢失Waymo lane-neighbor局部index范围而在视觉检查
中发现横穿误配，已标记SUPERSEDED。修复版保留双方局部index区间，支持同时有效的多邻接关系，
强制`lane_aware_only`且禁止几何fallback。

修复版3-file pilot通过自动、原始TFRecord拓扑重建和独立视觉三层门禁；20例每slot 4例，0 topology
失败、0 track重建不一致、0几何fallback。full51随后重建51个TFRecord、24872个scenario、168700
窗口、36 shard，train/val/test为135046/16870/16784，scenario跨split重叠为0。

Stage6O-v2状态为`FROZEN_READY_FOR_INTERACTION_AWARE_V2_PREPARATION`。train lead entry/exit/
intermittent/front switch为47335/48074/63415/8294，两种transition为15175/50236；五槽switch rate
为1.29%–2.64%，finite/shape/跨identity导数违规均为0。新纵向监督按median5、train q01/q99和
train median/IQR生成；同窗口RMS口径下accel median为1.48 m/s²，jerk median/q90为15.51/
28.47 m/s³，明显优于旧2.72和42.82/100.80。旧Stage6O v1保持原SHA且永久BLOCKED。

## 34. Stage 6S interaction-dominant nuPlan benchmark（Issue #260）

24个same-scenario pair、48条official PDM rollout全部成功。两个planner保持desired speed、accel/
decel与lateral参数一致，只改变minimum gap与time headway；机制分析保持embedding/BDD盲态。
平均速度差0.232 m/s、RMS accel差0.108 m/s²满足“小变化”门禁，但front-gap差-1.208 m未达到
-2.0 m门槛，只有THW一个预冻结interaction指标通过，正式状态为
`PDM_INTERACTION_BENCHMARK_LIMITATION`。这说明当前PDM/scenario pool没有构成确认级benchmark，
不是Waymo模型失败，也不能外推成nuPlan永远无法构造。不得在本批结果后调planner并当作同一确认实验。

因此当前结论是：Waymo数据侧允许进入Interaction-aware v2训练准备；完整实验侧仍需先接受并披露
PDM limitation，或另行预注册新的planner/场景生成方案。当前未训练新checkpoint、未扩大Waymo、
未覆盖Stage5D-balanced-v2。

## 35. Stage 6S-v2 interaction development与confirmation冻结（Issue #261）

Stage6S-v2回到扩大Pittsburgh库存，仅用pre-treatment持续front exposure、初始gap、closing/following
pressure和ego有效速度筛选场景。24个development pair中，短headway减长headway的median
`Δ mean speed=+0.259 m/s`、`Δ RMS accel=+0.225 m/s²`，而`Δ front gap=-4.284 m`、
`Δ finite THW=-2.660 s`，分别有91.7%和100% pair方向一致，满足“小ego差异+至少两项interaction
mechanism通过”的预冻结门禁。THW仅使用有限`0 < THW < 20 s`，排除999/sentinel/cap。

机制通过后冻结80-pair、15-log confirmation roster；它与development的log/token重叠为0，与
Stage6S-v1 token重叠也为0。筛选未读取confirmation planner outcome、embedding或BDD/MMD，状态保持
`CONFIRMATION_ROSTER_FROZEN_NOT_RUN`。因此benchmark侧已具备训练后独立确认条件，但未授权训练或
confirmation rollout。

## 36. Stage 6T A/B/C训练与评估协议冻结（Issue #262）

Stage6T在任何新checkpoint前冻结A/B/C。A为Dynamic v2 + 旧single-GRU/objective的数据修复主导对照；
B保持single-GRU拓扑但加入clean longitudinal supervision、ranking/sampling和mask-aware dropout；
C与B的数据、采样、dropout、objective、loss routing、seed和预算完全相同，仅改为参数量匹配的
ego16+context48双分支。A/B/C均保留83D输入和64D输出；ego13只作参考，不训练ego-only最终模型。

严格归因限制为：没有使用旧builder数据和Stage6T共同seed/预算重训的A0时，old64→A只能称为
dynamic-data-dominant comparison，不能称为纯数据版本因果效应；B→C才是冻结的encoder topology
增量比较。C不自动优先，若B通过全部门禁而C没有full-context相对neighbor-zero增量，应优先B。

冻结过程新发现六个Dynamic v2 part的33D标准化不同。因此Stage6T禁止从part-local
`interaction_feat_style.npy`训练，统一读取raw33并用全体train 135046行拟合一次global mean/std，
且不覆盖冻结shard。36个shard原SHA全部匹配，168700行shape/finite通过，scenario跨split重叠为0。

训练只能用Waymo train，epoch只能用Waymo val选择；A/B/C×3 seed的9个checkpoint全部锁定后才允许
一次性读取同一Dynamic v2 Waymo test。之后按固定顺序运行Stage6J/K、Stage6P和Stage6S-v2；test或
nuPlan结果均不得返工训练。Stage6S-v2必须先过trajectory mechanism gate，再读取interaction embedding。
跨representation raw MMD²仍禁止；C context增量使用各自null标准化Z_BDD差及log-cluster bootstrap。

当前状态为`FROZEN_READY_FOR_ABC_TRAINER_IMPLEMENTATION_NOT_TRAINING`，只允许下一步实现并review统一
trainer。训练、checkpoint写入、Waymo test、nuPlan评估和confirmation rollout均未授权，当前0/9
checkpoint。完整协议见`docs/stage6t_training_evaluation_protocol_zh.md`。

## 37. Stage 6U Unified A/B/C Trainer实现冻结（Issue #263）

Stage6U以单一trainer实现A/B/C配置切换。A/B/C输入83D、输出64D，encoder参数量分别为106560、
106560、105616。A沿用legacy objective；B/C共用clean longitudinal supervision、ranking、sampling、
dropout、seed与预算，C只替换为ego16+context48双分支。

B/C公平性由candidate-independent epoch plan代码级保证。同seed的sampling weights、sample indices、batch
offsets、ranking positive/negative、pair type、slot/all-neighbor dropout mask、augmentation seed、optimizer
schedule和budget逐项SHA一致；synthetic和Waymo subset均通过，并有篡改检测单测。全量135046行计划生成
B/C fingerprint也一致。

Trainer dataset API只接受Dynamic v2 train/val，只载入raw33并使用Stage6T冻结global train mean/std；
test、part-local标准化数组、Stage6J/K/P、nuPlan、BDD/MMD和Stage6S-v2 confirmation均禁止。Synthetic与
Waymo train/val subset的A/B/C forward/backward、64D、finite loss/gradient均通过。Resume恢复epoch、
batch cursor、optimizer、constant scheduler、Python/NumPy/Torch RNG与plan ledger，连续与恢复路径的loss
序列和最终model state完全一致。

正式loop已包含最多30 epoch/31680 step、Waymo val早停、每100 step heartbeat、best/last checkpoint和
绑定SHA的resume。但formal模式必须由独立授权manifest绑定最终implementation freeze SHA；当前状态虽为
`FROZEN_READY_FOR_ABC_FORMAL_TRAINING`，仍有`formal_training_authorized=false`且checkpoint=0/9。

## 38. Stage 6U正式训练前复核、重新冻结与授权

2026-08-12用户授权A/B/C×3 seeds正式训练后，启动前只读复核确认旧implementation freeze SHA
`6d1032b47f7dfaf4329a83db63105bedbeabf5a88ecbbc309ca77714a4d938fb`与其记录文件一致，但formal实现尚有
进度条未实际启用、epoch边界plan ledger错误校验、checkpoint锁定元数据不完整、best-val选择错误复用
early-stopping min-delta四项问题。为避免用已知缺陷生成正式checkpoint，旧freeze标记为superseded，不启动
任何训练。

修复只涉及训练工程语义和审计链，不改变Stage6T冻结科研协议。Train/val分别使用tqdm；每100 optimizer
steps原子更新带epoch累计量与plan ledger的`resume_model.pt`；epoch边界checkpoint的plan ledger为空，恢复
下一epoch时不再误比较；最低Waymo val objective负责best checkpoint，`min_delta=1e-4`只负责patience。
Checkpoint同时绑定candidate/seed/trainer/config/Stage6T/implementation freeze/authorization/Dynamic v2
content signature/package IDs/环境/resume history。

重新运行Stage6T/6U tests、synthetic和真实Dynamic v2 train/val smoke、普通resume与formal epoch-boundary
resume后，才允许生成新的implementation freeze。独立formal authorization随后固定A→B→C、每个candidate
按3407→3408→3409、精确输出目录、单MPS串行、日志/checkpoint规则和全部盲测禁止项。9/9完成后只生成
checkpoint ledger并停止，不读取Waymo test、Stage6J/K/P、nuPlan、BDD/MMD或Stage6S-v2 confirmation。
MPS smoke估计9任务串行建议22–27小时，首个正式epoch后必须用实测更新。完整中文报告见
`docs/stage6u_unified_abc_trainer_implementation_zh.md`。

## 39. Stage 6U checkpoint锁定与Stage 6V一次性盲测

Stage6U最终9/9任务完成，primary seed固定3407，checkpoint ledger状态为
`LOCKED_9_OF_9_READY_FOR_BLIND_EVALUATION_UNLOCK`。Stage6V在任何test/nuPlan结果出现前冻结一次性授权，绑定
Stage6T、Stage6U implementation/formal authorization、checkpoint ledger、9个best checkpoint以及
Stage6S-v2 roster；任何评估结果均不得触发训练或协议变更。

Waymo Dynamic-v2 test上A/B/C primary longitudinal delta为-0.0232/+0.0248/+0.0159，综合非劣性均通过，
但完整Waymo门禁均未通过。Stage6J/K paired中A/B/C分别为4/4 overall+7/12 task、3/4+2/12、3/4+2/12，
三者均未通过冻结门禁；ego13以4/4+12/12通过。

Stage6P n=400 context-balanced detection从old64 66.5%提升到A/B/C的90.5%/100%/99.5%，FPR为
3.0%/5.0%/6.5%，三者均通过unpaired门禁且B/C跨seed稳定。这是新64D纵向signal recovery的主要正结果，
但不能单独覆盖Waymo/paired负结果。

Stage6S-v2的80-pair roster有61对official rollout成功；19对原token两次均被nuPlan官方`valid_scenes`
scene-rank边界规则排除。该规则在pre-treatment inventory建榜时遗漏。冻结后不得替换场景或使用complete-case
子集重新定义confirmation，因此机制门禁未评估、embedding/BDD未读取、C full-context相对neighbor-zero增量
不可判定。

按预冻结规则，A/B/C均不满足最终论文主模型联合门禁。论文可以报告unpaired release检出的强、跨seed提升；
必须同时把Waymo primary/paired失败与confirmation roster执行失败写为限制或负结果。完整中文报告见
`docs/stage6v_one_time_blind_evaluation_report_zh.md`。

## 40. Stage 6W-A paired/unpaired机制与Stage 6S-v3 prospective确认（Issue #266）

### 40.1 Stage6W-A同池控制

分析只读取冻结old64/A/B/C/ego13、Stage6P 800 pairs、489 logs、2400 release splits和既有nuPlan
rollout，不训练checkpoint、不重跑Stage6P simulation。每个n=400 A/B release分别在release-A与release-B的
scenario support上构造same-support paired contrast，所以pairing、scenario pool与样本量不再混杂。
每个representation使用自身bandwidth和null标准化；禁止跨representation比较raw MMD²。

同池n=400下old64/A/B/C/ego13的paired median Z为13.502/27.535/28.295/25.368/101.139，均不弱于
对应unpaired结果。B/C在同池paired下明显强于old64，证明Stage6J/K中B/C较弱不是paired统计量固有问题，也不能
由183 vs 400样本量直接解释；真正差异来自Stage6J/K窄纵向dose/task与Stage6P广义planner contrast、场景池和
estimand的交互。

B/C的release-direction resultant length为0.925/0.927，old64为0.815；planner signal energy fraction为
3.97%/3.80%，old64为1.62%。log heterogeneity仍约66%–67%，但release aggregation会平均局部异质性并保留
一致shift。context-balanced口径下，B/C标准化signal为old64的2.586×/2.643×，null noise为0.856×/0.927×；
signal占log-Z增益85.9%/92.8%。因此unpaired接近100%主要由signal增强和方向一致性提高驱动，null variance下降
只作次要贡献。raw-marginal口径下B/C null noise没有下降，仍得到相同主结论。

### 40.2 Stage6S-v3 prospective roster repair

Stage6S-v2失败记录SHA
`e092ee198d412c0fcc830649ae7b22031d09a4284197131b9d0f2733c61faea8`保持不变，禁止用61个成功场景
做post-hoc confirmation。v3只新增nuPlan官方scene边界`row_num >= 3 AND row_num < scene_count - 1`，其余
short/long planner、THW、front-gap、mechanism gates、bootstrap和representation endpoint均复用v2冻结设计。

旧v2 roster的官方查询返回61/80，与实际61成功/19失败逐token完全一致。新筛选依次排除Stage6S-v1 token、
v2 development token/log和v2全部80个confirmation token，162个候选中120个通过官方查询；冻结80个token、
11个log，80/80场景、160条planner rollout全部成功。development log/token与v1/v2 confirmation token重叠均为0。
排除development logs后没有任何candidate位于v2 confirmation之外的log，所以与v2做log-disjoint在当前库存不可行；
该限制在冻结前记录，selection保持outcome-blind，最终不确定性按log cluster bootstrap处理。

### 40.3 mechanism与representation结果

机制确认使用完整80对。短减长的median `Δ mean speed=+0.289 m/s`、`Δ RMS accel=+0.150 m/s²`均低于
冻结上限；`Δ front gap=-4.202 m`、`Δ finite THW=-2.670 s`，方向一致率93.75%/100%。closing与
following accel median差均为+0.085 m/s²、方向一致率88.75%。四项interaction checks和全部control gates均通过。
THW仍只取有限`0 < THW < 20 s`，排除sentinel/cap。

机制通过后才加载representation。old64/A/B/C/ego13/C-neighbor-zero的null-standardized Z分别为
27.976/26.454/30.603/28.955/35.905/36.807，各自均显著；这不是跨表示raw MMD比较。预冻结主端点
C-full减C-neighbor-zero的`ΔZ=-7.852`，log-cluster bootstrap 95% CI=`[-33.393, 29.219]`，故
`incremental_interaction_information_pass=false`。不能把两者都显著误写成C context有增量价值。

### 40.4 冻结决策

Stage6V联合结论仍为`NO_ABC_CANDIDATE_QUALIFIES_UNDER_PRE_FROZEN_RULE`。当前论文可以按混合结果收口：
B/C显著提高unpaired release detection；同池机制解释排除了pairing本身；interaction benchmark机制稳定；但C没有
证明context相对neighbor-zero的增量，Waymo primary与Stage6J/K完整门禁仍未通过。B可作为更简单的release-level
工程候选讨论，不能改写成通过联合门禁的最终主模型。

本阶段不授权训练v3。若论文必须坚持interaction-aware主模型这一更强主张，C增量失败提供了明确研究理由；但任何
v3设计必须只用Waymo train/val完成，并在训练前另行扩展和冻结真正未使用、100% runnable的confirmation。当前
120个runnable candidate中80个已用于v3，只余40个，不足既有60-pair最低规模，不能复用本次confirmation或降低门槛。

## 41. 统一BDD Evaluation Matrix与Style Report Card冻结

Stage6及后续BDD报告统一服从`docs/unified_bdd_evaluation_matrix_style_report_card_zh.md`。Behavior Drift Profile、
BDD Statistic和Representation Evaluation必须分离；Stage6J/K、Stage6P等表示能力结果不能直接替代行为报告。
所有BDD行必须显式记录Reference、Target、固定behavior dimension、task、paired/unpaired、representation、null及
semantic delta。固定13维taxonomy和Stage5/6/7 task mapping分别由
`configs/unified_bdd_reporting_schema_v1.json`与`configs/unified_bdd_stage_task_mapping_v1.csv`冻结。

Stage6J/K、Stage6P、Stage6S-v3和Stage6W现有数值未修改，只按新schema重新解释。缺失的free-flow、lane-keeping、
lateral-gap acceptance及exact merge/yield/cut-in证据标为N/A/evidence gap，不启动补实验。状态为
`UNIFIED_BDD_REPORTING_SCHEMA_FROZEN`。

## 42. 训练后比较试验的统一BDD输出

冻结A/B/C后的比较结果通过`tools/build_unified_bdd_posttraining_report.py`映射为固定表A/表B，输出目录为
`outputs/unified_bdd_posttraining_report_v1/`。工具只读取Stage6J/K、Stage6P、Stage6S-v3、Stage6W与Stage7
已冻结的CSV/JSON，明确禁止训练、仿真、embedding读取和BDD/MMD重算；写入的manifest保存每个输入与输出的SHA256。

表A固定为13行，主行为报告固定使用old64以避免把表示选择混入Target→Reference解释；A/B/C/ego13的训练后比较
只放在表B scorecard，使用paired coverage、n=400 unpaired detection/FPR、seed稳定性和Stage6W signal/noise
归因。报告状态为`FROZEN_UNIFIED_BDD_POSTTRAINING_REPORT_COMPLETE`，不会改变Stage6V的联合决策。

## 43. Standardized Fixed-Dimension BDD Matrix（冻结资产的统一考试卷）

`unified_bdd_reporting_schema_v1`只冻结报告字段；其后新增的
`configs/standardized_fixed_dimension_bdd_protocol_v1.json`进一步冻结同一张
`behavior dimension × representation`考试卷。它严格区分三类reference：

1. **Behavior Reference**：Reference planner/version/release与Target必须显式出现，全部语义量为Target−Reference；
2. **Null Reference**：paired为对应representation自己的within-pair label-swap/randomization null，unpaired为该representation自己的独立A/A calibration，必须保留null q95；
3. **Representation Baseline**：old64只是历史能力baseline，A/B/C/ego13只能以检测能力、ratio或各自null标准化Z解释，禁止以raw MMD²横向排名。

固定13维保留Overall、纵向、横向和interaction全部维度；没有冻结样本的free-flow、lane-keeping和lateral-gap仍为N/A/evidence gap。每一格保留raw MMD²、null q95、ratio、Z_BDD、raw/Holm p、pass、N scenario/log、semantic delta及CI、direction和evidence status。一个parent task BDD支持多个semantic子维度时，子行共享`parent_bdd_result_id`，不作为重复独立检验。

Stage6J/K完全沿用原确认性183-pair、四dose和四scope结果；同一following dose100为60个场景、52个log，old64/A/B/C/ego13分别输出完整BDD/null/Z/p。仅有speed/accel语义时，`LON.CAR_FOLLOWING`方向统一为
`TARGET_MORE_ACTIVE_FOLLOWING`，不再错误写成`CLOSER`。Stage6S-v3沿用80-pair/11-log已通过机制的short−long contrast；front gap/finite THW、closing与following-pressure三条语义行共享同一个BDD parent，C-neighbor-zero只保留为diagnostic。

为补齐横向与变道的同一工况representation矩阵，Stage7只使用既有310对
`pdm_closed_conservative_v1 → pdm_closed_assertive_v1` official rollout、冻结的pre-treatment task membership以及A/B/C primary seed 3407。对old64/A/B/C/ego13重新导出embedding与同一pair-swap null的BDD，严格标记为
`POST_HOC_STANDARDIZED_DESCRIPTIVE_EVALUATION`；它不得替代Stage6V预注册端点，不能据此触发训练或协议改变。lane-change是changing-lane scenario slice，不能自动解释为ego已经执行变道。

当前完整结果位于`outputs/standardized_fixed_dimension_bdd_matrix_v1/`，状态为
`STANDARDIZED_FIXED_DIMENSION_BDD_MATRIX_COMPLETE`。主矩阵的纵向与跟车最强within-null敏感性均仍为ego13；Stage6S-v3
interaction同样以ego13的Z最高，但这不构成raw MMD²或通用representation排名。C相对C-neighbor-zero的既有interaction增量门禁仍为false；Stage6V最终联合结论不变。

## 44. Final Standardized BDD Reporting System冻结

最终报告协议`standardized_fixed_dimension_bdd_protocol_v2_final_render_only`只读取第43节冻结输出的CSV/JSON与
Stage6P冻结decision表。源文件SHA不匹配时立即失败；不会读取checkpoint、embedding或rollout，不执行BDD/null重算，
不改变场景、planner、Stage6V结论或任何统计值。

报告固定分为两页：第一页用`Primary Representation = B`回答Target相对Behavior Reference的行为变化，并逐行绑定
Behavior Reference、Target、paired/unpaired mode及B自己的Null Reference；第二页才评价old64/A/B/C/ego13。
B是测量工具，不是被测planner/version。old64永久称为Representation Baseline，不能定义行为方向。

Stage6S-v3的closing、front-gap/THW与longitudinal following三条语义维度共享一个parent task-level BDD，主矩阵
统一显示`†`，机器审计按每个representation保留同一`parent_bdd_result_id`并把三行计为一个独立检验。原
`Best capability`字段废止，替换为`该Treatment下最高标准化检测敏感度`；它不是通用representation排名。

ego13的高within-null敏感度仅适用于当前大量直接作用于ego运动学的controlled treatments，不能推导为neighbor/context
无价值。B继续只定位为当前最简单的learned release-level工程候选；A/B/C的Waymo、paired、interaction与联合门禁
事实均保持不变。

最终输出状态：`FINAL_STANDARDIZED_BDD_REPORTING_SYSTEM_FROZEN`。到此停止扩展BDD报告体系。

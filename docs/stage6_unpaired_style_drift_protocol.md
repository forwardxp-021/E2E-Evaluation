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

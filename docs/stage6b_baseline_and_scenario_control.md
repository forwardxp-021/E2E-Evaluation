# Stage 6B：Baseline Comparison 与 Scenario-Controlled Drift Analysis

## 1. 为什么 Stage 6A 后还需要 Stage 6B
Stage 6A 已证明 BDD 能区分随机对照与伪风格漂移，但也显示 scene/proxy shift 会造成高漂移值。Stage 6B 用于回答：
- learned embedding/BDD 相比简单手工统计有什么增益；
- 在 unpaired A/B 中如何减少 scenario/ODD 混杂。

## 2. 为什么 feature mean baseline 不够
单变量均值差异会丢失：
- 多变量联合结构；
- 非线性边界；
- 时序与交互动态。
因此仅做 mean delta 容易欠检或误检。

## 3. 为什么增加 feature MMD 与 PCA-feature MMD
- **feature MMD** 在原始特征空间比较分布整体形状；
- **PCA-feature MMD** 在降噪后子空间比较主变化模式；
- 两者作为强基线，可和 embedding BDD 同台校准。

## 4. learned embedding BDD 的持续价值
- 编码时序-交互行为表示；
- 可支持 top-drift case retrieval；
- 提供统一 behavior space，便于横向实验对比。

## 5. 为什么需要 scenario balancing
unpaired log 对比通常存在 ODD 构成差异。若 A/B 在 proxy scene 分布不一致，raw BDD 可能混合“风格差异 + 场景差异”。按 proxy bin 做平衡能降低这种偏差。

## 6. raw / sliced / balanced 的区别
- **raw BDD**：直接比较原始 A/B；
- **scenario-sliced BDD**：分 bin 分别比较；
- **scenario-balanced BDD**：按 bin 下采样对齐后再比较整体。

## 7. 三类实验解释
- **negative_control_random**：同分布 sanity check，应无显著漂移；
- **pseudo_agg_vs_cons**：已知伪风格漂移，应显著；
- **scene_confounding**：proxy 场景漂移，应作为 confounding warning 解读。

## 8. 当前限制
- scene proxies 仍有限；
- speed proxy 可能在当前 schema 缺失或不稳定；
- 真正 ODD matching 需要 Stage 6C/6D 引入更丰富元数据。

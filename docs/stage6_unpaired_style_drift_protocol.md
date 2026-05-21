# Stage 6A 非配对实路风格漂移评估协议

## 1) 为什么 Stage 6 以 non-paired 为主
真实部署常见 A/B 日志来自不同城市、时间和路线，无法保证逐样本配对，因此主评估模式必须是非配对分布比较。

## 2) 为什么 paired 仅是验证模式
paired 可做 sanity-check（例如同场景同交通流），但不代表真实采集条件，不应作为部署主结论依据。

## 3) BDD 在测什么、不测什么
BDD-MMD 衡量 A/B 在行为嵌入空间的**总体分布差异幅度**，不直接给“更激进/更保守”方向语义。

## 4) 为什么以嵌入空间为主
Stage 5D-balanced-v2 已学习到 interaction-aware 行为结构。直接在该空间做分布比较，比单特征均值更稳健。

## 5) 为什么还要 category/feature 层
类别层和特征层用于解释方向：例如跟驰保守性、纵向舒适性、横向稳定性等。

## 6) 为什么简单均值不够
单特征均值会忽略特征耦合与多模态分布。BDD 可先判断“是否漂移”，再用解释层定位“漂移到哪里”。

## 7) 为什么要做 scenario/proxy slice
非配对比较会受场景分布混杂影响（速度段、交互密度等）。切片后可区分“风格差异”与“场景差异”。

## 8) Stage 6A 三个验证实验运行方式

### negative_control_random
- 目标：同 test 集随机切分，预期漂移较小或可解释为采样噪声。

### pseudo_style_aggressive_vs_conservative
- 目标：按特征分位构造“保守样本 vs 激进样本”，预期漂移更大。

### scene_confounding_control
- 目标：故意构造低速高交互 vs 高速低交互，验证场景混杂会抬高原始 BDD。

示例命令见 `QUICK_REFERENCE.md` 新增 Stage 6A 段落。

# E2E-Evaluation

## 1. 项目目标
本项目面向 **trajectory-level closed-loop evaluation benchmark**，用于评估端到端规划/决策策略（E2E planning / decision policy）的行为质量。

核心边界：
- 不做 sensor rendering；
- 不引入 perception stack；
- 直接基于轨迹 rollout 做行为级评估。

研究重点：
- behavior embedding 学习与表示质量；
- policy separation（策略可分性）；
- style retrieval（风格检索一致性）；
- style-distance correlation（风格距离与物理量关联）。

## 2. 核心思想
- 输入：trajectory rollout（可来自 synthetic policy、learned planner 或 human trajectory）。
- 输出：behavior embedding、style metrics、evaluation report。
- 评估框架与具体模型解耦（model-agnostic）：重点评估行为表现，不绑定某一训练范式。

## 3. 当前阶段总览

| 阶段 | 状态 | 简要说明 |
|---|---|---|
| Stage 1/2 synthetic rollout | 完成 | p0/p1/p2 policy separation |
| Stage 3 ablation/local sweep | 完成 | lateral_stable mechanism analysis |
| Stage 4A scaffold | 完成 | data1 synthetic scaffold |
| Stage 4B Waymo human extraction | 完成 | full51 = 168191 windows |
| Stage 4C baseline validation | 完成 | baseline-only full51 |
| Stage 4D row-level learned embedding | 完成 | learned evaluated on full51 |
| Stage 4E jerk/comfort-aware embedding | 进行中 | training done; export/eval/compare next |

## 4. 主要数据目录
- `data1/`：synthetic rollout scaffold 数据，用于流程联调，不等同于公开人类验证集。
- `outputs/waymo_human_v1_full51/`：`human_public` 验证数据目录（full51）。
- 任何 `embeddings*.npy` 都必须与 `traj.npy` 严格 row-aligned（行对齐）。
- Stage 4D 与 Stage 4E 输出必须分路径维护，禁止互相覆盖。

## 5. 当前主要结论
Stage 4D（row-level learned embedding）已完成的主要结论：
- learned embedding 已在 human_public full51 上完成 row-level 对齐与评估；
- learned 显著优于 random；
- learned 在分类指标上表现较强；
- raw_feature / pca_feature 在检索上仍更强（pseudo label 来自特征空间，存在同源优势）；
- learned 在 lateral/curvature 风格表达上优于 trajectory_l2；
- learned 对 jerk/comfort 的敏感性仍偏弱，构成 Stage 4E 的主要动机。

## 6. 当前限制
- pseudo labels 属于 weak labels，不是 ground truth；
- feature-derived 标签机制天然有利于 raw_feature / pca_feature；
- learned 尚未在所有指标上全面超越全部 baseline；
- 仍需补充 qualitative retrieval 与 paper-ready 分析。

## 7. 如何运行
详细命令见 [`QUICK_REFERENCE.md`](./QUICK_REFERENCE.md)。

## 8. 文档说明
- `README.md`：项目总览与研究状态（叙述性文档）。
- `QUICK_REFERENCE.md`：命令、期望行为、通过标准（操作性文档）。
- `06_experiment_4_waymo_human_validation.md`：Stage 4 详细实验记录与决策追踪。

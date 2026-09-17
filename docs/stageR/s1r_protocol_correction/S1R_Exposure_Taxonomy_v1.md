# S1R Exposure Taxonomy v1 — PROPOSED ONLY

状态：`PROPOSED_ONLY`。`OWNER_APPROVAL_REQUIRED = TRUE`。本文件不修改冻结 S1。

## 核心原则

隔离规则应对应当前 claim 的偏差路径：某 session 的 outcome 只有在被用于选择、修改或优化当前 Primary comparison 的 RBR、H、BDD statistic、calibration、operating point、cohort inclusion 或 sample-size decision 时，才构成 confirmatory contamination。历史使用仍需完整披露，但“用过”不自动等于“未来 controlled evaluation 不可用”。

| 类别 | 定义 | Controlled V | Confirmatory C |
|---|---|---|---|
| `E0_METADATA_ONLY` | 仅身份、schema、route 或其他 pre-outcome metadata | 可在 amendment 与 eligibility 完成后进入 | 可候选，但既有 reservation 必须先解决 |
| `E1_UNRELATED_HISTORICAL_USE` | 普通 Stage6/7/7L、old64/ego13 等与当前 Primary 方法选择无直接绑定的历史 outcome | 可透明标记后进入 | 不属于 E5 untouched |
| `E2_BENCHMARK_ENGINEERING` | HLC/TSB/controller/mechanism/F_match/safety/applicability 工程 | Claim B 可进入；Claim A unseen-generalization 不可当 unseen | 不属于 E5 untouched |
| `E3_PRIMARY_METHOD_DEVELOPMENT` | 直接用于当前 H、BDD readout/kernel/calibration/operating point/eligibility method 开发 | 不得用于同一冻结方法的无条件 confirmatory claim；只可作为开发/敏感性证据 | 排除 |
| `E4_PRIMARY_OUTCOME_DRIVEN_ADAPTATION` | 看过 Primary-like RBR-vs-H outcome 后改变方法、cohort、阈值、样本量或分析 | 排除对应 confirmatory claim | 严格排除 |
| `E5_UNTOUCHED_CONFIRMATORY` | 未参与 E1–E4 outcome use，且无冲突 reservation 的相对 untouched session | 可进入 | 首选 C 候选 |
| `UNKNOWN` | 身份或 decision-use provenance 不足 | 不默认可用 | 不可进入 |

E5 是 confirmatory role 标签，不表示“风险高于 E4”。逐会话 CSV 为了给出互斥统计，将无已知历史 exposure/reservation 的会话归入 E5；同时保留完整 evidence list。多重 exposure 使用 E4 > E3 > E2 > E1 > E0 的最高风险，E5 仅在没有 E1–E4 时成立。

## 当前证据分层

- E0：1
- E1：176
- E2：66
- E3：0（当前绑定账本未发现直接证据）
- E4：0（当前 Primary 从未授权；未发现 outcome-driven adaptation）
- E5：5
- UNKNOWN：0 个身份分类；但 E1 的“与当前方法完全无 decision-use 关系”只有中等置信度，因为旧账本没有专门记录该因果用途。

这些计数是 exposure-policy reclassification，不是 applicability、route、reset 或生产执行资格认证。

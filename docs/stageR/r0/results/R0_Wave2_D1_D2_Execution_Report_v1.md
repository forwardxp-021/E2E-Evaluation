# R0 Wave 2：D1 跨域迁移与 D2 上下文/响应审计执行报告 v1

证据等级：`DEVELOPMENT_DIAGNOSTIC_EVIDENCE`。本报告是 R0 development diagnostic，不是确认性结论；未执行训练、仿真或新 planner rollout。

## 执行基线与偏差闭环

- 本地执行基线：`677fac9b3b34bcf00079d4634026d7d90b69522a`；Wave 1.1 的本地等价提交为 `677fac9b3b34bcf00079d4634026d7d90b69522a`。
- 远端 `b5bc0b16a4fe5abd819a347bac6ee4b1ea365fbe` 为操作者声明、与本地等内容的仓库接口同步提交；当前本地 Git 对象库不含该对象，因此此处不把它冒充为本地 Git 可验证对象。
- 历史 D0 偏差原始记录保留不删；科学责任人处置已写为 `ACCEPTED_COMPLETENESS_CORRECTION; DEVIATION_STATUS=CLOSED; ADDITIONAL_EVIDENCE_DOWNGRADE=NO`，关闭时间 `2026-08-26T09:15:31Z`。

## D1：目标对齐与冻结跨域 probe

- 五个剂量、九个 CORE target 均进行了定义、单位、mask、有限支持、类别支持、槽位语义及 log 聚类合同审计；raw33 的四个 CORE target 与冻结聚合函数逐元素重算一致。
- Waymo→nuPlan 只使用 Waymo Dynamic-v2 validation 的 historical `last` embedding 重建 probe：同一 5-fold scenario GroupKFold、ridge/logistic grid、随机种子和预处理；没有在 nuPlan 上 refit、选择超参数或选择 target。
- 正式 `D1_CROSS_DOMAIN_SEMANTIC_TRANSFER = INCONCLUSIVE`，原因是冻结表没有跨域数值通过门。剂量间结果也不构成事后门槛；对每个 representation 和 target 的完整 CI、MAE/NRMSE、Spearman 与 calibration slope 均已保留在结果表。
- 直接迁移的 pattern 是分化的：多数 ego 连续时序 target 的 R² 中位数为负（例如 dose100 的 mean-speed/accel/yaw/heading 依次为 -1.4598/-41.2489/-63.0634/-33.1613），而 raw33 interaction target 的直接读出保持较高正 R²。下表给出十个 representation 的 target 级中位数；lane-change 为 BA，其余为 R²。

| 剂量 | lane-change BA | front-distance R² | rel-speed R² | front-pressure R² |
| --- | ---: | ---: | ---: | ---: |
| dose0 | 0.9474 | 0.9240 | 0.7640 | 0.8529 |
| dose100 | 0.9545 | 0.9387 | 0.7733 | 0.8724 |

lane-change 类别不平衡已在每个格的 class-support 与 log-cluster CI 中显式记录；没有替代或新增 target。

## D2：消融、OOD、shuffle 与 shortcut

- `NEIGHBOR_ABLATED` 是唯一可执行的 Gen-1 诊断视图：邻居的 valid=0 原生编码是全零，故保留 mask、槽位数量与输入形状。所有此类结果标签为 `DIAGNOSTIC_NOT_HISTORICAL` / `ABLATION_SENSITIVITY_ONLY`。
- `EGO_ABLATED` 和 `CONTEXT_ONLY` 为 `NOT_APPLICABLE_TO_ARCHITECTURE`：共享 83D GRU 没有 ego 缺失通道，raw zero 可代表物理值，不能当作 absence。
- `CONTEXT_SHUFFLE` 为 `NOT_EVALUABLE_FROZEN_STRATA_UNAVAILABLE`：缺少冻结的 `event_phase_bin` 与 `traffic_density_tertile` 来源/anchor，故不得启动或随意合并 strata。
- 对合法邻居消融，最低 representation×dose 中位 cosine 为 `0.8478`；四指标逐行 OOD >=2/4 的最高比例为 `0.0000`。interaction frozen-probe R² 在 dose0/dose100 的十个 representation 中位变化分别为：front-distance -1.9186/-2.0236、rel-speed -1.4389/-1.5250、front-pressure -1.1301/-1.1496。这是输入敏感性，不是因果依赖或 information attribution。
- q99 规则被逐行保留，但冻结合同没有 condition-level 聚合门，因此 `D2_ABLATION_OOD_RISK = INCONCLUSIVE`，不把描述性比例升级为 `OOD_DOMINATED`。
- shortcut 审计完成 `80` 个可计算 representation×dose×proxy 描述性格；地图/位置单类、路由/道路字段 unknown、scenario identity 一行一类均不强行建模。不存在冻结数值 shortcut 门，因此 `D2_SHORTCUT_RISK = INCONCLUSIVE`。

## 冻结逐假设结果

| 假设 | 结果 | 原因 |
| --- | --- | --- |
| D1_KNOWN_SEMANTIC_INFORMATION_PRESENT | SUPPORTED | 保留 Wave 1 的 frozen CORE-target 结果；本波未改写。 |
| D1_CROSS_DOMAIN_SEMANTIC_TRANSFER | INCONCLUSIVE | 已完成冻结 direct transfer，但无冻结跨域数值通过门。 |
| D2_RESPONSE_SENSITIVITY | NOT_EVALUABLE | ego absence 不能在 Gen-1 输入语义中合法定义。 |
| D2_CONTEXT_SENSITIVITY | INCONCLUSIVE | 仅有邻居消融敏感性；不得作因果归因。 |
| D2_PAIRING_SENSITIVITY | NOT_EVALUABLE | 完整冻结 shuffle strata 不可构造。 |
| D2_SHORTCUT_RISK | INCONCLUSIVE | 仅有低容量 group-aware 描述性关联，且无冻结数值门。 |
| D2_ABLATION_OOD_RISK | INCONCLUSIVE | 有逐行 q99 记录，无冻结条件级聚合规则。 |

历史 Stage6S BDD 没有被重跑或用于改变任何 primary conclusion；它仅保留为既有次级 development diagnosis。

所有数值见 `r0_cross_domain_probe_metrics.csv`、`r0_context_ablation_metrics.csv`、`r0_context_ablation_ood_metrics.csv`、`r0_context_ablation_probe_metrics.csv`、`r0_context_shuffle_metrics.csv` 和 `r0_context_leakage_probe_metrics.csv`。

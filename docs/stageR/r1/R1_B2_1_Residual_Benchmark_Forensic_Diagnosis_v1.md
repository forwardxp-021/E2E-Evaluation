# R1 B2.1 残差基准失败模式法证诊断 v1

## 结论

本次仅复核既有 B2.1 证据：48/48 run、24/24 pair、0 技术失败；roster SHA-256 为 `0617e79b9f51d8b2ae8ac76b110e1dbcfaa77dad200a73b405eb2d6a54675e52`，selector salt SHA-256 为 `617331678ef4573be11b5408a1dde2c910c8614177541dd51c650c08bc24baf9`，均与冻结值一致。没有新增 rollout、没有修改生成器或门禁、没有读取 representation/BDD/probe/RBR。

`R1_RESIDUAL_BENCHMARK_ENABLEMENT = BENCHMARK_FAMILY_NOT_READY` 保持不变。但 B2.1 的失败不应自动归因为冻结生成器参数：证据优先支持上下文 canonicalization 和 TSB 重规划锚点两个实现缺陷，并支持将 planned-first 与 realized measurement source 分开报告。

## 根因排序

1. **上下文实现：SUPPORTED_AS_IMPLEMENTATION_DEFECT。** raw official observation 在冻结十帧中实际含动态对象，但 adapter 将五个邻车槽、HLC target-front/target-rear 或 TSB front 全部强制为 ABSENT，并将 TSB hazard 固定为 `NONE_OBSERVED`。B2.1 的 pair hash identity 只支持 `RAW_PRE_CONTEXT_IDENTITY`，不支持 `FROZEN_CANONICAL_CONTEXT_SEMANTIC_CONFORMANCE`。
2. **planner/replan 空间连续性：SUPPORTED_AS_IMPLEMENTATION_DEFECT（TSB），MIXED（HLC）。** TSB `_build_tsb` 每次调用以冻结初始 x/y/heading 建轨迹，并令局部 `distance[0]=0`、`speed[0]=initial_speed`；24 个 TSB run 的逐调用首状态连续性均非全程 exact。TSB position-error max 范围为 2.597673–185.884969 m。
3. **measurement source：SUPPORTED。** B2.1 primary 使用第一次 planner output；既有 trace 同时允许合法构造 80 帧 realized ego sequence。planned/realized 机制状态计数为 `{"R-HLC": {"PLANNED_FIRST_OUTPUT": {"HLC_MECHANISM_PAIR_FAIL": 5, "HLC_MECHANISM_PAIR_PASS": 7}, "REALIZED_EGO_SEQUENCE": {"HLC_MECHANISM_PAIR_FAIL": 11, "HLC_MECHANISM_PAIR_PASS": 1}}, "R-TSB": {"PLANNED_FIRST_OUTPUT": {"TSB_MECHANISM_PAIR_FAIL": 7, "TSB_MECHANISM_PAIR_PASS": 5}, "REALIZED_EGO_SEQUENCE": {"TSB_MECHANISM_PAIR_FAIL": 12}}}`。两者不是可互换的测量源。
4. **scenario/map applicability：MIXED。** safety attribution 为 `{"R-HLC": {"BASELINE_ONLY_UNSAFE": 2, "BOTH_ARMS_SAFE": 3, "BOTH_ARMS_UNSAFE": 7}, "R-TSB": {"BOTH_ARMS_SAFE": 5, "BOTH_ARMS_UNSAFE": 7}}`；both-arm unsafe 更支持场景/基线适用性限制，treatment-only unsafe 才支持处置诱发风险。HLC 有 5/12 个 secondary heading delta 接近 2π，和 reference geometry/heading unwrap 诊断相关，不能据此新拟合阈值。
5. **冻结生成器参数：INCONCLUSIVE。** 在先处理上述实现与测量源问题前，现有证据不足以把残差 smoke 失败归因于生成器参数；本阶段不得改参数。

## Gate contingency 与安全归因

逐 pair 的 mechanism×engineering、mechanism×safety、engineering×safety 和三重交集见 `r1_b2_1_gate_contingency_audit_v1.csv`。HLC mechanism/engineering/safety 分别通过 7/6/3 pair，mechanism×engineering 为 6，但 mechanism×safety、engineering×safety、三重交集均为 0。TSB mechanism/safety 分别通过 5/5，交集为 0。安全归因严格区分 collision 与 drivable-area failure，并保留两臂重叠集合；24 pair 中 treatment-only unsafe 为 0。

## 冻结上下文与时间锚点

时间抽取 `history[-11:-1]` 在全部 48 个 run 中选中了预期的十个相邻 history frame，但其物理时间存在官方 lidar 微小抖动，并非合同字面要求的 exact 0.1 秒网格；adapter 随后直接重标为 0.0–0.9，而未记录版本化 resampling。因此 temporal anchor 为 `TEMPORAL_ANCHOR_IMPLEMENTATION_NONCONFORMANCE`。生成器的名义 0.0–1.0 秒仍为 common prefix，首次允许分歧是 1.1 秒。上下文语义另为 `CONTEXT_CANONICALIZATION_IMPLEMENTATION_NONCONFORMANCE`，两者不可合并判断。

## Planned 与 realized

所有 run 至少有 149 次调用，前 80 个 current-ego 样本是连续 simulator iteration。计算保留原 state，不插值、不外推，并按冻结 iteration-index 网格 0.0–7.9 秒评估；物理 timestamp 对名义网格的最大偏差单独列出。`r1_b2_1_plan_vs_realized_audit_v1.csv` 同时给出 HLC/TSB mechanism 和 Fmatch。该分析只作 development diagnosis，不覆盖 B2.1 历史 primary。

HLC planned-first 为 7/12 pass，realized 为 1/12 pass；TSB planned-first 为 5/12 pass，realized 为 0/12 pass。该方向一致支持 realized retention 较弱，但不授权把历史 primary 改写为 realized primary。

## HLC 法证

几何表报告 source/target 分离、tangent heading delta、曲率、方向一致性、自交/反转和原生 reference 覆盖：12/12 direction consistency 为 1.0，未检出自交，原生 8 秒 reference coverage 为 7/12；HLC 逐调用首状态 position-error max 范围为 0.661081–4.863528 m。机制表分别报告 planned/realized 的 retreat count、commit latency、monotonic fraction 及失败类别；分类计数为 `{"('PLANNED_FIRST_OUTPUT', 'GEOMETRY_PROJECTION')": 5, "('PLANNED_FIRST_OUTPUT', 'MECHANISM_RETAINED')": 1, "('PLANNED_FIRST_OUTPUT', 'REPLAN_DISCONTINUITY')": 6, "('REALIZED_EGO_SEQUENCE', 'GEOMETRY_PROJECTION')": 5, "('REALIZED_EGO_SEQUENCE', 'MECHANISM_RETAINED')": 1, "('REALIZED_EGO_SEQUENCE', 'MONOTONIC_GATE')": 3, "('REALIZED_EGO_SEQUENCE', 'RETREAT_NOT_RETAINED')": 3}`。约 2π 的 secondary cases优先标为 `GEOMETRY_PROJECTION` 诊断；不新增阈值。

## TSB 法证

planned/realized 均按冻结 Option-A calculator 重算 phase count、release fraction、second peak ratio 与 low-speed/endstop。planned 的 7 个失败均为 `LOW_SPEED_ENDSTOP`；realized 的 12 个失败分为 6 个 `PHASE_MERGE` 与 6 个 `LOW_SPEED_ENDSTOP`，完整分类计数为 `{"('PLANNED_FIRST_OUTPUT', 'LOW_SPEED_ENDSTOP')": 7, "('PLANNED_FIRST_OUTPUT', 'MECHANISM_RETAINED')": 5, "('REALIZED_EGO_SEQUENCE', 'LOW_SPEED_ENDSTOP')": 6, "('REALIZED_EGO_SEQUENCE', 'PHASE_MERGE')": 6}`。单独标记 `TSB_REPLAN_ANCHOR_IMPLEMENTATION_DEFECT` 和 `STRAIGHT_LINE_ROUTE_REALIZATION_LIMITATION` 的空间/安全关联；不修改 profile。

## 协议与授权

冻结合同没有变化，历史 v1.1 的 `SCIENTIFIC_PROTOCOL_DEVIATION` 原记录保持不动。本审计新增版本化记录 `IMPLEMENTATION_NONCONFORMANCE_AFFECTING_SCIENTIFIC_GATE`：这是实现合规修正，不伪装成合同修改。它不把 NOT_READY 翻为 READY，只修正失败原因的科学解释。

- `R1_RESIDUAL_BENCHMARK_ENABLEMENT = BENCHMARK_FAMILY_NOT_READY`
- `RBR_A = NOT_AUTHORIZED`
- `RBR_B = NOT_AUTHORIZED`
- `RBR_C = NOT_AUTHORIZED`
- `NEW_ROLLOUT = NOT_AUTHORIZED`

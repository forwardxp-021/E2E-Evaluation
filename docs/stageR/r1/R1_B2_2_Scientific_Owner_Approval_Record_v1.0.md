# R1 B2.2 Scientific Owner 批准记录 v1.0

状态：`APPROVED_AS_RECORDED_FOR_PROSPECTIVE_IMPLEMENTATION`。本批准只授权 R1 Phase B2.3 的合同与实现修正、合成测试和旧 trace 只读诊断；不授权任何新 planner rollout、真实 roster 选择、representation/BDD/probe/RBR 读取或 RBR training。

## 已批准决定

- A：`CORRECT_FROZEN_CONTEXT_CANONICALIZATION`。
- A2：`CORRECT_COMMON_PREFIX_TEMPORAL_ANCHOR`。
- B：`TSB_ROUTE_ALIGNED_LONGITUDINAL_REALIZATION`。
- C：`HLC_MAP_GEOMETRY_APPLICABILITY_CONTRACT`。
- D：HLC 与 TSB 均采用 `CURRENT_EGO_CONTINUOUS_REPLAN_ANCHOR`。
- E：Primary source 为 `REALIZED_CLOSED_LOOP_EGO`；Secondary source 为 `INITIAL_PLANNED_TRAJECTORY`，仅作 generator-intent diagnostic。
- F：`MECHANISM_APPLICABILITY_ELIGIBILITY = APPROVED_IN_PRINCIPLE`；TSB 数值 speed floor 必须先完成解析推导，再由 owner 决定是否冻结。

## 保持不变

`HLC_GEN_V2_OPTION_B` 与 `TSB_GEN_V2_OPTION_A` 参数、两类 mechanism thresholds、F_match calipers 均不变。历史 B2.1 artifacts 不覆盖，历史 `tools/r1_context_mechanism_core.py` 不修改。

## 授权边界

本记录不批准 fresh identity smoke。任何新 smoke 必须等待 context v2、temporal anchor、route realization、current-ego anchor、measurement source、HLC applicability 与 TSB applicability 全部冻结，并由 owner 再次明确授权。

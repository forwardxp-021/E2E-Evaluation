# R1 Phase B0 科学负责人决策单 v0.1

状态：`REQUIRES_SCIENTIFIC_OWNER_APPROVAL`。本文件不构成 protocol、generator 或 smoke 授权。

## A. HLC mechanism × F_match

审计结论：`MARGINALLY_FEASIBLE`。

纯合成 witness 证明交集非空；但可行性依赖窄车道/高速度包络与 baseline/treatment heading-total 的共同设计，不能外推为正式 roster 上普遍可行。

待审批：是否接受该分类，并允许进入 versioned generator 设计阶段。

## B. heading_change_abs_total 的结构性问题

结论：`STRUCTURAL_MECHANISM_OVERLAP_CONFIRMED`，但未达到“冻结合同全局无交集”的 `STRUCTURALLY_CONFLICTED`。

retreat 必然制造负横向速度/heading，recommit 再制造正 heading；因此 absolute heading total 会直接测到 primary mechanism morphology，而不是纯 nuisance matching。

待审批：是否认定该重叠需要 R1/D4 scientific amendment review。

## C. HLC amendment options

以下均为讨论稿，全部 `REQUIRES_SCIENTIFIC_OWNER_APPROVAL`，不得在本阶段 freeze：

1. 保留 heading 为 Primary F_match；版本化重设计 HLC generator，使 baseline/treatment heading-total 匹配且 retreat 更平缓。冻结 threshold 当前不变。
2. 将 `heading_change_abs_total` 从 Primary F_match 重分类为 mechanism-proximal / semantic audit feature；需正式 amendment，不能直接删除。
3. 经 owner 审阅后，用真正低阶的 lateral endpoint descriptor（例如 terminal lateral offset/velocity 的预声明组合）替代其 nuisance 角色；需新版本、先验 caliper 与独立验证。

待审批：A/B/C 中选择一项，或维持现状并接受 marginal feasibility 风险。

## D. HLC implementation bug

结论：`NO_IMPLEMENTATION_DEFINITION_BUG_CONFIRMED`。未复现 derivative discontinuity、phase stitching、heading/curvature、unit/frame 或 terminal matching 错误。

待审批：确认不以旧 diagnostic 0/6 为由启动 bug-fix；若改 profile，按 versioned scientific amendment 管理。

## E. TSB mechanism × F_match

结论：`JOINTLY_FEASIBLE`。三个纯合成 V2 witness 均同时通过冻结 mechanism/F_match。

待审批：是否接受可行性结论。

## F. TSB 当前 failure 根因

结论：`NO_IMPLEMENTATION_BUG_CONFIRMED / GENERATOR_PROFILE_REDESIGN_REQUIRED`。当前 release 的有效强度/时长不足，经 median3 与 timestamp-aware gradient 后，mild 的 phase 被合并，nominal/strong 的 release fraction 不足；未发现 timestamp、merge 或 integration 实现错误。

待审批：是否将根因记录为 profile insufficiency，而非 code bug。

## G. TSB Gen-V2 options

- `TSB_GEN_V2_OPTION_A`：-0.9×0.5s，+0.4×0.7s，-0.9×0.5s；合成 release fraction 0.333333，second ratio 1.0。
- `TSB_GEN_V2_OPTION_B`：-1.0×0.6s，+0.6×0.7s，-0.9×0.6s；合成 release fraction 0.533333，second ratio 0.9。
- `TSB_GEN_V2_OPTION_C`：-1.0×0.7s，+0.8×0.6s，-1.0×0.5s；合成 release fraction 0.5，second ratio 1.0；mean-abs-accel delta 0.113044，接近 0.11777666 caliper，裕量最小。

三项均为 `PROPOSED_NOT_FROZEN`。安全结论仅限合成运动学，不包含 collision/off-road/background replay。

待审批：选择一个候选、要求进一步无 outcome 合成设计，或全部拒绝。若选择，须冻结参数 JSON/SHA 后才能进入后续授权评估。

## H. official nuPlan runtime

总状态：`NOT_READY`。nuPlan 1.2.2、map、history/planner/traffic-light/route/metric 接口存在；scenario DB 不可用，fresh token/log 未绑定，deterministic replay seed 为 `VERSION_AMBIGUOUS`，original background replay 未验证。

待审批之外的工程阻断：补齐非空官方 DB 与 versioned seed/replay contract；本阶段不允许以 core-only 单测替代。

## I. 新 48-call compliant smoke 授权

建议：`DO_NOT_AUTHORIZE_YET`。

已满足：执行器 48-call preflight、baseline reuse、构造前 hard cap；old12 永久 blacklist；future selector 设计完成。

未满足：HLC scientific amendment/generator 决策、TSB V2 参数 freeze、official DB、fresh identity、deterministic replay seed、owner 最终授权。

待审批：上述条件关闭后，是否另行签署一次性 48-call smoke authorization。当前决策单本身不授权执行。

## 保持不变的正式状态

- `R1_CONTEXT_MECHANISM_CONTRACT = UNCHANGED_FROZEN`
- `R1_TECHNICAL_SMOKE_V1 = NONCOMPLIANT_EXECUTION_DIAGNOSTIC_ONLY`
- `R1_DEVELOPMENT_ROSTER = NOT_READY`
- `RBR_A/B/C = NOT_AUTHORIZED`

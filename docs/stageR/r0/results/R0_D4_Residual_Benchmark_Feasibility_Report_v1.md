# R0 D4 残余基准可行性报告 v1

## 结论

本 Wave 3 已完成冻结合同、既有 DEVELOPMENT 资产及 selection-leakage 的只读审计。三个 residual family 均**不能合法构造** descriptor-balanced / context-controlled / mechanism-confirmed residual benchmark：不是因为某种 representation、BDD 或 probe 结果，而是因为冻结的 exact pre-treatment context anchor 与可执行的 family-specific mechanism rule 在历史资产中均不可用。

所有证据等级均为 `DEVELOPMENT_DIAGNOSTIC_EVIDENCE`；没有训练、没有新 planner rollout、没有读取 representation/BDD/probe outcome，也没有修改冻结合同。

## 冻结核验

- tag commit：`319757c7f72efb55c80c780e4d0f17e5341b19ec`。
- freeze content commit：`5bd5c7ac58c284d4c938919cacf2eefb969a5c44`。
- 绑定的 19 个冻结 artifact SHA256 均匹配。

## Family 结果

|hypothesis_id|formal_hypothesis_result|development_feasibility_status|
|---|---|---|
|D4_DESCRIPTOR_EQUIVALENCE_R_HLC|NOT_EVALUABLE|INSUFFICIENT_FOR_RBR_DEVELOPMENT|
|D4_MECHANISM_DIFFERENCE_R_HLC|NOT_EVALUABLE|INSUFFICIENT_FOR_RBR_DEVELOPMENT|
|D4_OUTCOME_BLIND_FEASIBILITY_R_HLC|NOT_EVALUABLE|INSUFFICIENT_FOR_RBR_DEVELOPMENT|
|D4_DESCRIPTOR_EQUIVALENCE_R_TSB|NOT_EVALUABLE|INSUFFICIENT_FOR_RBR_DEVELOPMENT|
|D4_MECHANISM_DIFFERENCE_R_TSB|NOT_EVALUABLE|INSUFFICIENT_FOR_RBR_DEVELOPMENT|
|D4_OUTCOME_BLIND_FEASIBILITY_R_TSB|NOT_EVALUABLE|INSUFFICIENT_FOR_RBR_DEVELOPMENT|
|D4_DESCRIPTOR_EQUIVALENCE_R_IP|NOT_EVALUABLE|INSUFFICIENT_FOR_RBR_DEVELOPMENT|
|D4_MECHANISM_DIFFERENCE_R_IP|NOT_EVALUABLE|INSUFFICIENT_FOR_RBR_DEVELOPMENT|
|D4_OUTCOME_BLIND_FEASIBILITY_R_IP|NOT_EVALUABLE|INSUFFICIENT_FOR_RBR_DEVELOPMENT|

## 候选与匹配规模

|residual_family|pre_context_candidate_independent_units|matched_pairs_or_sets|mechanism_qualified_pairs|development_feasibility_status|
|---|---|---|---|---|
|R-HLC|80|0|0|INSUFFICIENT_FOR_RBR_DEVELOPMENT|
|R-TSB|183|0|0|INSUFFICIENT_FOR_RBR_DEVELOPMENT|
|R-IP|80|0|0|INSUFFICIENT_FOR_RBR_DEVELOPMENT|

`pre_context_candidate_independent_units` 仅是来源中的事前候选量；因 Context_match 不可评估，`matching_candidate_count=0`，没有进行任何 pair/set 选择。

## Mechanism derivation 审计

|residual_family|mechanism_variable|availability|
|---|---|---|
|R-HLC|mechanism.commit_latency_s|NOT_EVALUABLE_MECHANISM_VARIABLE|
|R-HLC|mechanism.hesitation_retreat_count|NOT_EVALUABLE_MECHANISM_VARIABLE|
|R-HLC|mechanism.monotonic_transition_fraction|NOT_EVALUABLE_MECHANISM_VARIABLE|
|R-TSB|mechanism.brake_phase_count|NOT_EVALUABLE_MECHANISM_VARIABLE|
|R-TSB|mechanism.interstage_release_fraction|NOT_EVALUABLE_MECHANISM_VARIABLE|
|R-TSB|mechanism.second_brake_peak_ratio|NOT_EVALUABLE_MECHANISM_VARIABLE|
|R-IP|mechanism.gap_acceptance_latency_s|NOT_EVALUABLE_MECHANISM_VARIABLE|
|R-IP|mechanism.minimum_accepted_rear_gap_m|NOT_EVALUABLE_MECHANISM_VARIABLE|
|R-IP|mechanism.yield_response_onset_s|NOT_EVALUABLE_MECHANISM_VARIABLE|

冻结变量只有语义性说明，未包含可执行阈值、anchor 或算法，且 target definition 明确写为 `REQUIRED_BEFORE_D4_EXECUTION`。因此这些变量全部标为 `NOT_EVALUABLE_MECHANISM_VARIABLE`；历史 raw33 和历史机制表不会被改名或当作替代指标。

## Development fallback

`D4_DEVELOPMENT_BALANCE_FALLBACK_V1` 的 caliper 未被重调，也未被用于不完整 context 的近似匹配。它保持 `NOT_FORMAL_PHYSICAL_EQUIVALENCE` 与 `NOT_R4_CONFIRMATORY_EQUIVALENCE`。

## Handcrafted challenge

没有 matched residual set，因此没有执行 ego13、extended handcrafted、DTW 或 raw mechanism 的组间可分性分析；不产生 `HANDCRAFTED_FEATURES_CANNOT_DETECT` 类主张。

## Selection leakage

Frozen F_match 与 M_behavior 在每个 family 的 Primary 角色零交集。Wave3 未执行 pair selection；工具未读取 embedding、BDD、probe 或 RBR outcome，故无 outcome-guided selection leakage。

## RBR 含义

三个 family 都是 `INSUFFICIENT_FOR_RBR_DEVELOPMENT`，不足两个 family 的最低要求。RBR-A/B 不具备 candidate-specific authorization review 条件；RBR-C 还保留 Wave2 D2 unresolved 状态。现有 training authorization manifest 不变，RBR 训练仍为 `NOT_AUTHORIZED`。

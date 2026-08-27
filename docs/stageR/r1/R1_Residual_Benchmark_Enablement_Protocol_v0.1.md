# R1 残余基准启用协议 v0.1

## 状态与范围

- 状态：`DRAFT_FOR_SCIENTIFIC_OWNER_REVIEW`。
- 本文是 prospective R1 Phase A 设计稿，不是冻结文件，也不修改任何 R0 v1.0
  protocol、SAP、decision table、D4 role、fallback 或历史结果。
- 本阶段只设计 R0 final closure 与 R1 residual benchmark enablement 的合同；不运行
  planner、rollout、benchmark、representation 或 RBR training。

## R0 依赖与结论边界

R0 tag `r0-v1.0-protocol-freeze`（`319757c7…`）保持有效。R0 D4 的
`NOT_EVALUABLE_WITH_EXISTING_HISTORICAL_ASSETS` 结论不改动：历史资产缺少 exact
pre-treatment anchor 与已实现的 mechanism contract。这是 asset-contract limitation，
不是 representation negative outcome。

R1 只能为未来独立 controlled benchmark 提供前瞻性、待审批的输入合同；它不将
`D4_DEVELOPMENT_BALANCE_FALLBACK_V1` 变为 formal physical equivalence，也不授予
RBR 训练授权。

## Family scope

|family|R1 Phase A 地位|理由|
|---|---|---|
|R-HLC|`PRIMARY`|直接检验 pure-lateral temporal morphology。|
|R-TSB|`PRIMARY`|提供独立的 longitudinal temporal morphology；与 HLC 一起满足最初至少两个 family 的主线目标。|
|R-IP|`SECONDARY_CONDITIONAL_NOT_REQUIRED_FOR_INITIAL_ENABLEMENT`|保留在协议中，但受 D2 attribution unresolved 与更复杂的 interaction anchor 约束；不作为初始 enablement 硬门。|

## 统一 pre-treatment 时间合同（提案）

设每个 prospective scenario 的 deterministic generator reference clock 为 `t=0`。

1. `t_anchor` 是两个 treatment arms 都仍使用相同冻结输入状态、相同 route、相同
   history 与相同 generator prefix 的离散时刻；它在 roster freeze 前写入 manifest。
2. `t_diverge` 是首次 condition-specific control command 不同的时刻，必须满足
   `t_diverge >= t_anchor + 0.1 s`。任何 `t < t_diverge` 的 arm 间轨迹、历史输入、
   map/route 绑定必须一致。
3. 主提案 `T_PRE_CONTEXT = [t_anchor - 1.0 s, t_anchor)`；在 `dt=0.1 s` 下要求
   10 个有效且时间戳误差不超过 `1e-6 s` 的历史帧。只可读取 original replay/
   history buffer 的这些帧，不读取 closed-loop response。
4. 若 runtime 无法提供完整 1.0 s 的 official history，唯一可提交给 owner 的替代是
   **保持 1.0 s 长度不变**，在 simulation 开始时加入 condition-identical 的 1.0 s
   warm-up，再把其末端标为 `t_anchor`。本稿不批准缩短窗口或用首帧替代。

每个 variable 的 source、单位、坐标、slot、map query、精度和缺失规则见
`r1_context_anchor_definition_proposal_v0.1.csv`。它们对 R1 prospective rollout
可以 exact implement；对既有 R0 historical assets 均不可追溯获得 exact parity，
因此不回填历史 D4。

R-TSB 的 `planned_stop_or_hazard_class` 只能使用 anchor 时可查询的 frozen route、
traffic-light state、stop-control map object 和 observed lead state；不得用事后生成的
`scenario_type` 标签。如果 runtime 不能暴露该 pre-treatment source，待 owner 审批的
替代仅是 `SCENARIO_ELIGIBILITY_ALTERNATIVE_TSB_LEAD_FOLLOWING_ONLY`：在 roster 前要求
10 个 history frames 都有合法 current-lane lead，并把无可观察 route/hazard source 的
scenario 标为不符合该候选 eligibility。该替代不是对 R0 role 的自行修改，不能在未经审批
时启动。

## 受控 generator 的设计原则

- 采用 rule-based / deterministic external planner chain，不引入新的 E2E planner。
- scenario、route、initial-state fingerprint、map references、background replay 和
  pre-treatment history 在 arm 间全部冻结；treatment 只在 `t_diverge` 后作用。
- R-HLC 候选实现以现有 `PureLateralExecutionPlanner` 的 frozen source/target
  centerline、quintic blend 与 canonical longitudinal generator 为结构起点。
- R-TSB 候选实现以相同 external-planner 接口增加 deterministic
  `PiecewiseLongitudinalProfileGenerator`；lateral route 与 background chain 固定。
- 非有限状态、map/route mismatch、initial-state fingerprint mismatch、不可安全插值
  都必须在 rollout 前作为 technical preflight failure 记录；不得静默 fallback。

`R1_Controlled_Treatment_Generator_Proposal_v0.1.md` 记录两种 treatment 的参数、
安全边界、nuisance 风险和运行量级；其中没有已经批准或已经运行的 generator。

## Whole-roster prospective 设计

```
pre-treatment eligibility（仅 t < t_anchor）
  -> roster freeze（scenario/log/anchor/route/context manifest）
  -> rollout（本 Phase 不执行）
  -> whole-roster mechanism gate
  -> representation evaluation
```

- 独立单位：`scenario_token`，每个 token 包含其全体预冻结 arm。
- cluster：`log_id/log_name`；报告和未来抽样按 log cluster。
- exclusion：只允许在 rollout 前因 source/context/map/route/anchor 合同不满足而排除，
  必须记录原因与数量。
- technical runtime failure：不替换；保留在完整 roster 分母并按预冻结类别报告。
- context/representation input missing：只报告事实缺失；不得用其他 arm 或 outcome 替代。
- treatment-mechanism weak、off-road、collision、incompletion 或 safety failure 不是
  technical deletion 原因；都是 whole-roster outcome 数据。
- intention-to-evaluate：每个已冻结 scenario 全部进入 arm-level accounting；不因
  mechanism 强弱、BDD、embedding 或 probe 结果删改 roster。

## R0 frozen F_match 的保留可行性

R-HLC 必须保留 `mean_speed`、`end_minus_start_speed`、
`heading_change_abs_total`、`path_length`；R-TSB 必须保留 `mean_speed`、
`end_minus_start_speed`、`mean_abs_accel`、`path_length`。每个 family 的所有 frozen
caliper 同时适用，且仍仅为 development fallback：不构成 R4 equivalence。

- HLC 的主要 generator 风险是 retreat/recommit 可能改变 `path_length` 和
  `heading_change_abs_total`；canonical longitudinal progress 应保持完全相同。
- TSB 的主要 generator 风险是 release 会直接增加 `mean_abs_accel`，同时影响 terminal
  speed 与 path length；parameter solver 必须在 outcome-blind trajectory-only preflight
  中约束这些 descriptor。
- 不得用 representation/BDD/probe outcome 调整 parameter、caliper、roster 或保留规则。

## 开发规模提案

所有规模均为 prospective planning，不是 confirmatory power calculation。`reserve` 必须在
rollout 前纳入 roster；不是以后替换 scenario 的权限。

|family|档位|最少 unique logs|目标可评估 scenarios|预冻结技术 reserve|总 roster scenarios|端点 paired rollouts|
|---|---:|---:|---:|---:|---:|---:|
|R-HLC|MINIMAL|12|24|6|30|60|
|R-HLC|RECOMMENDED|20|40|8|48|96|
|R-HLC|ROBUST|30|60|12|72|144|
|R-TSB|MINIMAL|12|24|6|30|60|
|R-TSB|RECOMMENDED|20|48|10|58|116|
|R-TSB|ROBUST|32|80|16|96|192|

参考既有 rules-only 轨迹链的技术记录，单 rollout 量级约 17.23 秒；该数字仅用于预算，
不来自 representation 或 BDD outcome。初始建议是两个 family 均采用
`RECOMMENDED`，但仍需要 scientific owner approval。

历史 raw metadata 仅用于规模量级：既有 HLC source 有 80 个 scenario/79 个 log，TSB
source 有 183 个 scenario/156 个 log。它们不构成 R1 roster，也没有按机制强弱或任何
representation/detection 结果被选择。reserve 反映 technical readiness 与未知 control
success 的前瞻性缓冲；每个 reserve item 同样在 roster freeze 后进入 whole-roster accounting。

## R1 Phase A 审批与训练状态

所有 anchor、context、mechanism option、generator、sample scale 和 R-IP activation
决定均为 `REQUIRES_SCIENTIFIC_OWNER_APPROVAL`。本稿没有 `FROZEN`、`APPROVED` 或
training authorization 状态。

`r0_training_authorization_manifest_v1.0.json` 不作修改；RBR-A、RBR-B、RBR-C 均保持
`NOT_AUTHORIZED`。

## Phase A raw-evidence 可复核命令

```bash
waymo_dev/bin/python tools/stageR_r1_phaseA_raw_evidence_audit.py
```

该命令只读取预先列明的 `ego_seq.npy` 与可用的 `ego_seq_mask.npy`，对每个 source 的全体
有效帧计算 measurement-scale quantile，并首次生成
`docs/stageR/r1/r1_phasea_raw_trajectory_evidence_v0.1.json`。它拒绝覆盖已有 evidence，
不会读取 embedding、BDD、probe、checkpoint、RBR、detection 或 planner rollout output。
通过标准是每个 declared source 的数组 shape/mask shape 合同成立，且 JSON 记录四个 source、
`dt=0.1 s` 与 `READ_ONLY_TREATMENT_INDEPENDENT_DEVELOPMENT_RAW_EVIDENCE` 状态。

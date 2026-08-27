# R1 Phase B0 Generator–Contract Compatibility 报告 v0.1

## 范围与治理结论

- 远端基线：`4407cdae8323ddbba839afb1ab1d9bd880b6e6f9`；本阶段开始时本地与远端 tree 一致。
- R0 freeze：`r0-v1.0-protocol-freeze`，19/19 binding 核验通过。
- scientific-owner approval SHA-256：`27aac073d2323aadd8d1a89b96d959fcdcb41e7b913d53e5d8acbc59b6dbc12c`。
- context contract SHA-256：`82e4d9dd8bd3a63e1e8dcfb4504d378df510b5d20aebd471e5dfaac3071b7365`。
- technical smoke scope correction SHA-256：`8e636060f8fcd4250e31d6b9c260c640daa21e58a4ed82b6ac64b2571c13c162`。
- readiness v0.2 SHA-256：`3b66504419a592dfa7e5b4977bcc170325886c5e01b79fd15fb8080c6a4f2e28`。

旧 12 个 smoke identity 已写入 fail-closed 永久黑名单，并同时带有 `TECHNICAL_SMOKE_ONLY`、`NONCOMPLIANT_EXECUTION_DIAGNOSTIC_ONLY`、`EXCLUDED_FROM_R1_DEVELOPMENT_ROSTER`、`EXCLUDED_FROM_FUTURE_R4_CONFIRMATION`。本审计未读取旧 smoke 数值作为参数搜索目标。

## HLC：物理包络与兼容性

冻结 retreat 最少为 `0.08 p`。以 2.7–4.2 m 物理 lane-width 包络换算，横向位移为 0.216–0.336 m，相对单调完成同一横向端点至少增加 0.432–0.672 m 的横向总变差；`dp/dt<=-0.10/s` 对应至少 0.27–0.42 m/s 的阈值横向速度。使用 treatment-independent DEVELOPMENT raw 的速度 q01–q99（4.992095–13.292885 m/s），负 heading excursion 近似为 0.020309–0.083935 rad；进入并离开负 heading 的额外绝对 heading 事件贡献下包络约为 0.040618–0.167871 rad。相对 baseline 的欧氏 path-length 差没有不依赖 baseline duration/longitudinal compensation 的严格正下界；合成 witness 给出实际差值。

冻结 heading caliper 为 0.0492160141 rad。因此：在大部分低速/宽车道包络内，heading 约束会直接压缩 retreat/recommit；在窄车道、高速边角，其理论下包络仍低于 caliper。`heading_change_abs_total` 不是与机制独立的低阶 nuisance feature，而是存在明确的 `STRUCTURAL_MECHANISM_OVERLAP`。

纯合成平行车道 witness（lane width 2.7 m、speed 13.292885 m/s，不对应任何真实 scenario）同时得到：

- `HLC_MECHANISM_PAIR_PASS`：commit latency delta 2.5 s，monotonic delta -0.121401；
- `F_MATCH_PASS`：mean-speed delta 0.034558 m/s、terminal-speed-change delta 0、path-length delta 0.224129 m、heading-total delta 0.001366 rad。

这证明冻结合同的交集非空，但 witness 依赖 favorable physical corner 以及 baseline/treatment heading-total 的共同设计，裕量不具有普遍性。最终分类：

`HLC_MECHANISM_FMATCH_COMPATIBILITY = MARGINALLY_FEASIBLE`

不是 `STRUCTURALLY_CONFLICTED`，但 heading 的结构性机制重叠已确认，建议进入科学修订审议，不能自动移除该 feature。

## HLC 实现审计

当前 progress state machine 使用 quintic phase joins；每个 join 的位置、一阶导、二阶导端点连续。合成平行车道 fixture 验证：baseline/treatment 在 `t<1.1s` 共用前缀，所有 treatment 在 4.5 s 内到达 `p=1`，heading/yaw/curvature 从同一 SI-unit `xy,time` 链生成且有限。

结论：`NO_IMPLEMENTATION_DEFINITION_BUG_CONFIRMED`。旧 diagnostic 的 0/6 运动学完整性更符合“当前 profile 的时长/幅值相对安全界限和 heading caliper 裕量不足”，不能据此声明 phase stitching、单位、heading reconstruction 或 curvature 公式错误。不得实施 versioned code fix；若改变 profile/mechanism，需科学负责人批准新版本。

## TSB：可行性与 V2 草案

纯合成分段加速度搜索固定 baseline 为一次 `-1.0 m/s² × 0.95s` 制动，只以冻结 mechanism/F_match 和运动学约束判定，不使用旧 smoke outcome。以下三个 `PROPOSED_NOT_FROZEN` witness 均为 2 个 brake phase，且同时通过冻结机制与 F_match：

|option|first brake|release|second brake|release fraction|second peak ratio|F_match 最大相对紧张项|
|---|---|---|---|---:|---:|---|
|A|-0.9×0.5s|+0.4×0.7s|-0.9×0.5s|0.333333|1.0|mean-speed 0.290869 / caliper 0.708204|
|B|-1.0×0.6s|+0.6×0.7s|-0.9×0.6s|0.533333|0.9|mean-abs-accel 0.095652 / caliper 0.117777|
|C|-1.0×0.7s|+0.8×0.6s|-1.0×0.5s|0.5|1.0|mean-abs-accel 0.113044 / caliper 0.117777|

`TSB_MECHANISM_FMATCH_COMPATIBILITY = JOINTLY_FEASIBLE`

当前实现审计未发现 release duration、timestamp、integration、phase merge 或 low-speed endstop 的定义性错误。median3 与 gradient 会削弱/平移短 release 的有效窗口：当前 mild 合成 profile 的两个 phase 会合并，nominal/strong 虽保留两 phase，但 release fraction 不足。因此结论为：

`NO_IMPLEMENTATION_BUG_CONFIRMED`

`GENERATOR_PROFILE_REDESIGN_REQUIRED`

## 48-call fail-closed 验证

执行器现在先生成纯 ID schedule：每 family 为 6 scenarios ×（1 baseline + 3 treatments）=24，两 family 合计 48。`CoreConstructionBudget.claim()` 在每次 trajectory-core construction 前调用；计数已为 48 时，第 49 次 claim 在构造前抛错，计数保持 48。9 个纯合成单测全部通过；没有运行执行器主流程或任何真实 scenario。

## 状态与建议

- `R1_CONTEXT_MECHANISM_CONTRACT = UNCHANGED_FROZEN`
- `R1_TECHNICAL_SMOKE_V1 = NONCOMPLIANT_EXECUTION_DIAGNOSTIC_ONLY`
- `R1_DEVELOPMENT_ROSTER = NOT_READY`
- `RBR_A/B/C = NOT_AUTHORIZED`
- official nuPlan runtime：`NOT_READY`，主要阻断为 scenario DB、fresh identity 与 deterministic replay seed contract。

建议科学负责人审议 HLC heading feature 的角色及 generator 版本、审批一个 TSB V2 草案，但当前不建议授权新 compliant smoke；运行时阻断关闭、合同/候选 SHA freeze 与 owner approval 完成后方可重新评估授权。

## 复现与通过标准

```bash
waymo_dev/bin/python tools/r1_phaseb0_compatibility_audit.py --output /new/path/r1_phaseb0_compatibility_results_v0.1.json
waymo_dev/bin/python -m unittest tests.test_r1_phaseb0_compatibility -v
```

工具拒绝覆盖已有结果。通过标准：HLC witness 同时通过 frozen mechanism/F_match，三个 TSB V2 witness 均通过，48-call schedule 精确且第 49 次在构造前被拒绝；任何真实 rollout 均不得发生。

# R1 B2.5 零 Rollout 官方执行集成冻结报告 v1

## 结论

B2.5 已在不枚举 candidate、不生成 roster、不启动 simulation 的前提下完成未来 V2 官方执行链的集成冻结。future path 从 roster-row schema、官方地图只读 bridge、Stage5D authoritative slot assignment、native route builder、current-ego generator construction、realized-first evaluator、official safety canonicalizer 到 ledger/budget 均有唯一版本化绑定；启动入口保持硬阻断。

## 关键核验

- Replay observation horizon：只有全局 official timestamp stream 覆盖 iterations 0...79、首尾完整且内部 gap ≤ 已冻结 0.25 s 时才 complete；complete+empty tracks 为 `DYNAMIC_CLEAR_NO_ACTORS`，否则 `NOT_ELIGIBLE`。
- Clearance：公共包络覆盖两臂各自 XY tangent heading 下的完整 Pacifica oriented footprint；仍只用 3.0 m/0.5 m buffer。
- Stage5D：adapter 直接调用 `assign_stage5d_slots(..., assignment_mode="lane_aware_only")`，slot identity exact parity，geometric fallback 未使用。
- 官方地图：bridge 绑定 nuPlan `AbstractMap` 的 lane/lane-connector、baseline path、native adjacency、stop-line 查询；歧义 fail closed。
- current-ego：HLC 与 TSB 的 state1 都由同一 builder、同一输入精确重建并按 canonical JSON machine representation 比对；不引入新距离或 heading threshold。
- 路线：`build_native_route_reference_v1_1` 实际调用 repeated occurrence cursor；A-B-A-C 的第二个 A 能沿 native successor 构建至 C。
- HLC endpoint：使用 actual `(t79-t78)`、final realized XY tangent、target native geometry 和 native route projection；0.25/0.05/0.25/1.5 gates 不变，+1 ms jitter 仍可评估。
- V2 evaluator：Primary 唯一顺序为 realized ego → timestamp-aware mechanism → prospective F_match → endpoint → engineering → official safety；planned trajectory 仅为 `SECONDARY_GENERATOR_INTENT_ONLY`。
- Ego footprint：绑定 official runtime Pacifica 5.176 m × 2.297 m；generic fallback 禁止。

## 授权边界

selector v0.6 状态为 `READY_FOR_SCIENTIFIC_OWNER_ENUMERATION_AUTHORIZATION`，但 `actual_candidates_enumerated=0`、`actual_roster_selected=false`、`enumeration_authorized=false`、`new_rollout_authorized=false`。下一轮仍需 owner 对 enumeration/24-identity roster 与后续 48-run smoke 作出明确授权。RBR 全部未授权。

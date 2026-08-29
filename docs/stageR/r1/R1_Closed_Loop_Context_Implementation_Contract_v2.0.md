# R1 Closed-loop Context Implementation Contract v2.0

状态：`PROSPECTIVE_IMPLEMENTATION_PENDING_FINAL_FREEZE`。

## 时间锚点

`t_anchor=1.0 s`、`t_diverge=1.1 s`。Primary `T_PRE_CONTEXT=[0.0,1.0)` 必须来自 condition-identical warmup 的 actual official closed-loop iterations 0–9，物理 timestamp 必须 exact dt=0.1 s。禁止再次使用 first planner call 的 `history[-11:-1]`，禁止把 physical -1..0 s history 重新标为 0..0.9；不满足时为 `NOT_EVALUABLE_TEMPORAL_GRID`，不得静默 resample。

## Canonical adapter

`tools/r1_closed_loop_context_adapter_v2.py` 从 official observation、official map query、traffic-light、route 数据真实构造 traffic density、五个 lane-aware neighbor slots、稳定 track IDs、map/lane validity 与 missingness。HLC 另构造 target-lane front/rear、arc gaps 和 intended direction；TSB 另构造 current-lane front、gap、lead relative speed、THW、hazard multi-hot。

只有真实查询无对象时才允许 ABSENT；只有 route signal、stop control、slow lead 均不存在时才允许 `NONE_OBSERVED`。最终 payload 送入历史冻结 canonicalizer 的语义函数，但不修改历史 core。

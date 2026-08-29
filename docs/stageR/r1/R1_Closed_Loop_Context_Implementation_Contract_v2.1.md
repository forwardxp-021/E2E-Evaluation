# R1 Closed-loop Context Implementation Contract v2.1

状态：`FROZEN_PROSPECTIVE_CONTEXT_V2_1`。v2.0 保留为历史 amendment 实现。

`PRE_CONTEXT=iterations 0...9`，`ANCHOR_FRAME=iteration 10`，`t_diverge=iteration 11`（nominal 1.1 s）。canonical ordering coordinate 为 `SIMULATION_ITERATION_INDEX`；nominal labels 为 0.0...0.9，同时强制保存 actual `time_us`。只要求 actual timestamps 有限、严格递增；不要求 exact 100000 us，禁止插值、外推或物理时间重标。

initial speed median、lane offset median、traffic density、neighbor pattern、front/target gaps、relative speed、THW 和 stable track IDs 均来自 PRE_CONTEXT。road class、route-relevant signal 和 static stop-control relation 仅来自 ANCHOR_FRAME。

TSB hazard priority 保持不变：route red/yellow > static stop > observed slow lead > none。route signal 与 stop-control 仅看 anchor；slow lead 必须由 pre-context front slot 满足 `>=8/10` 且同一 stable ID 后才能成立。

slot assignment 严格等价 Stage5D lane-aware lane-ID、projected arc、slot order 与 tie-break 语义；geometric fallback 禁止。velocity missing/nonfinite 立即 fail closed。HLC density 是 source+target 及其 immediate native adjacent corridor 内、冻结 50 m projected route distance；TSB 是 current+immediate adjacent corridor 内同一 50 m 定义。逐字段矩阵见 `r1_closed_loop_context_v2_1_conformance_matrix_v1.0.csv`。

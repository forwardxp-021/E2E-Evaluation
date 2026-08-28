# R1 Runtime Determinism Validation V3 授权 v1.0

scientific owner 已一次性授权 R1 Phase B1.3 的 V3：最多新增 8 个 `OFFICIAL_CLOSED_LOOP_RUN`，固定为
原 frozen outcome-blind 4-scenario roster 的每行 `V3_RUN_A` 与 `V3_RUN_B`。

V3 唯一允许 HLC decisive monotonic baseline 与 TSB single continuous braking baseline。其变化只限于对既有
official Parquet metrics 的 canonical binding；不改变 mechanism、F_match、generator、threshold、roster 或
scientific outcome 定义。

V1 的首条技术失败和 V2 的 Parquet-output-discovery 失败均永久保留为历史技术执行，不计入 V3 cap，也不构成
V3 evidence。V3 只有在 canonical metric parser 的零预算 preflight 已通过后才能发出第一条 pre-run claim。

完整 commit/tree、planner、V3 executor、metric parser/contract、roster、DB/map、seed、generator 与 simulation
config SHA binding 见同名 JSON。48-call smoke、development roster、treatment 和 RBR 仍未获授权。

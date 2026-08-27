# R1 HLC Generator V2 合同 v1.0

状态：`FROZEN_GENERATOR_PARAMETERS_EXECUTION_LIMITED_TO_BASELINE_RUNTIME_VALIDATION`。

冻结 `HLC_GEN_V2_OPTION_B`：advance `p:0→0.38 / 1.4s`；hold `0.6s`；retreat `0.16 progress / 1.0s`；recommit `2.4s`；总时长 `5.4s`；quintic C2 joins。

baseline 保持 `DECISIVE_MONOTONIC_LANE_CHANGE`，treatment 保持 `HESITANT_RETREAT_RECOMMIT`。冻结 HLC mechanism 不变；Primary F_match 继续为 mean speed、end-minus-start speed、path length；`heading_change_abs_total` 保持 secondary mechanism-proximal audit，不恢复为 Primary。

本阶段只可在 runtime determinism validation 中运行 HLC baseline；treatment execution 不获授权。

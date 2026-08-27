# R1 Phase B1 科学负责人决策单 v0.1

状态：`REQUIRES_SCIENTIFIC_OWNER_DECISIONS`。本决策单不自动授权 smoke。

## A. HLC V2 option

请在 `HLC_GEN_V2_OPTION_A/B/C` 中选择一个，或全部拒绝。A 最短但 yaw-rate 裕量仅 `0.000978 rad/s`；B 为中等时长/裕量；C 裕量最大但总时长最长。三者当前均为 `PROPOSED_NOT_FROZEN`。

## B. HLC endpoint validity 数值方案

请选择 `OPTION_ENDPOINT_STRICT`、`OPTION_ENDPOINT_RESOLUTION_BASED`，或要求新的 treatment-independent 方案。结构性 identity/continuity/safety checks 已就绪，但新数值 tolerance 尚未冻结。

## C. DB / runtime binding

请确认是否接受：DB layer `READY`（1,624 个可读且 map-compatible DB）与 official runtime overall `NOT_READY` 分开表述。DB 可用并不等于 background replay 已验证。

## D. replay contract

请确认 `MASTER_SEED=2026082701`、版本/SHA/map/config 绑定及稳定排序规则是否可接受；同时接受当前 background determinism 仍为 `VERSION_AMBIGUOUS`，不能伪报 READY。

## E. 一次性 fresh 48-call smoke

当前建议：`DO_NOT_AUTHORIZE_YET`。HLC option/endpoint 决策和 background replay determinism 未关闭。即使 owner 接受本轮 DB/runtime binding，也必须另行签署明确的一次性 smoke authorization；不得由本决策单隐含授权。

保持：`R1_DEVELOPMENT_ROSTER = NOT_READY`，`RBR_A/B/C = NOT_AUTHORIZED`。

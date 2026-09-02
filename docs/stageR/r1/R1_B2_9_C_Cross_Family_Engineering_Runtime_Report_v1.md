# R1 B2.9-C 跨 Family 工程运行报告 v1

## 证据边界

本轮仅使用永久科学排除身份执行非科学工程 canary，不是 official smoke，也不产生科学 roster。任何 evaluator PASS/FAIL 都未用于调阈值、换身份或修改机制/F_match。

HLC identities：`b1be12bca092597a`, `25944935eadb52f1`, `ef3172a208cc5dd7`。TSB identities：`b486f9cf33a85455`, `3edcce9e7e19573f`, `ff152a4cf9c4503b`。

## 运行结果

fresh actual runs `12`，reruns `0`；HLC technical complete `6/6`，TSB `6/6`；exact 80-row traces `12/12`，secondary planner calls `0`；metric/callback `12/12`，safety adapter structural complete `12/12`。pair dispatcher HLC `3/3`、TSB `3/3`。

实际 realized timestamp duration 为 `[7.899697, 7.899699999999999, 7.899998999999999, 7.900097, 7.900214999999999, 7.900682]` s；metric runner termination window 为 `[7.999663, 7.999745, 8.000017, 8.000084, 8.000188, 8.000665]` s。因此安全语义仅为 `OFFICIAL_SAFETY_WITHIN_FROZEN_R1_PRIMARY_RUNTIME_WINDOW`，不声称与历史 full-scenario metric exact parity。

当前 scientific identities 仿真：`false`。OFFICIAL_SMOKE_AUTHORIZED=`false`；RBR_A/B/C=`NOT_AUTHORIZED`。

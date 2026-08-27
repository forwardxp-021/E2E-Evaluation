# R1 HLC Generator Endpoint Validity 合同 v1.0

状态：`FROZEN_PRIMARY_VALIDITY_CONTRACT`。

Primary endpoint limits：target-center offset `≤0.25m`、heading error `≤0.05rad`、lateral velocity `≤0.25m/s`、baseline/treatment route-progress delta `≤1.5m`。identity 仍要求相同 source lane、intended target lane、方向，以及两 arm 都完成 target-lane transition；phase continuity、curvature、lateral acceleration 和 yaw-rate 继续审计。

`OPTION_ENDPOINT_STRICT`（`0.15m/0.03rad/0.15m/s/1.0m`）仅为 secondary sensitivity audit，不能以 strict 失败排除 Primary-valid pair。

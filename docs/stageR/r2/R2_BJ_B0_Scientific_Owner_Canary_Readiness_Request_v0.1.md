# R2-BJ-B0 Scientific Owner Canary Readiness Request v0.1

## 请求事项

请 Scientific Owner 审阅 B0 冻结包，并决定是否在下一阶段对冻结 roster 的 selection rank 1 授权一次 engineering canary：

- identity：`cc1abd3989065d8d` / `2021.10.01.16.53.37_veh-44_01126_01602`
- pair：`R2BJB0-HLC-01`
- intended runs：2
- 顺序：baseline 后 treatment
- replacement：禁止
- parameter update：禁止

## Readiness 证据

- B0 component SHA binding manifest：`35a1282328b461f0b1edbbd39a4284870382ad52a83bd2975d9a91bc0ece1cf9`。
- A5 applicable pool：34；选择算法独立复放通过；冻结 roster：8；未选择池：26。
- 精确 selection-rank tuple：`(1,2,3,4,5,6,7,9)`；方向、地图和速度带配额全部精确满足。
- roster token/log 唯一：8/8、8/8；历史和永久 exclusion overlap：0。
- 8 个 pre-outcome pair bindings 与 16-run intended schedule 已冻结。
- 16/16 Hydra composition、exact scenario resolution、V4 planner、Primary80 controller 和 SimulationRunner construction 均通过。
- 8/8 pairs 的 `t<1.1 s` baseline/treatment trajectory construction exact equal。
- 在线 architecture/infrastructure failure taxonomy、stop-current/stop-remaining 行为和授权/SHA/预算/schedule 的 simulator-start 前门已实现并完成 mutation test。
- V4 generator、V4 `_states`、global parameter space 和所有阈值未改变。

## 当前授权状态

本文件不是运行授权。当前强制状态保持：

```text
BJ_B_ENGINEERING_SIMULATION_AUTHORIZED = FALSE
CANARY_AUTHORIZED = FALSE
NEW_RUN_BUDGET = 0
RUNNER_RUN = 0
R2_C_STARTED = FALSE
CONFIRMATORY_SMOKE_STARTED = FALSE
RBR_STARTED = FALSE
```

若 Owner 决定授权，下一阶段必须逐字节绑定 B0 component manifest、schedule 和 pair-binding SHA，并仅给予这 2 个 intended runs 的一次性预算。B0 本身不会越过授权门。

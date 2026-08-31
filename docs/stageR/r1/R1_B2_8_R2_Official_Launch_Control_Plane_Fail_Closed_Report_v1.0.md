# R1 B2.8-R2 Official Launch Control-Plane Fail-Closed Report v1.0

## 结论

官方 nuPlan 1.2.2 exact resolution 为 40/48；4 个冻结 identity 为 0 match，对应 8 个 arm/run。
按冻结规则，0 match 必须 FAIL_CLOSED，且不得 replacement。因此没有继续 full Hydra composition、SimulationRunner construction 或任何仿真。

## 失败身份

- 2021.08.27.14.14.40_veh-45_01790_02016 / a6e0468e028357de（R-HLC）：0 official match。
- 2021.09.28.13.24.06_veh-44_02759_02879 / 0198af1831f65977（R-TSB）：0 official match。
- 2021.09.28.19.55.30_veh-44_01744_01819 / cf56ddebd44f5372（R-TSB）：0 official match。
- 2021.09.10.15.00.33_veh-45_01265_01432 / 0f67192c7dd45664（R-TSB）：0 official match。

## 保持的控制面状态

- simulation_started = false
- actual official runs = 0
- consumed budget = 0
- OFFICIAL_SMOKE_AUTHORIZED = false
- RBR_A/B/C = NOT_AUTHORIZED

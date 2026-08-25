# Stage7L-E E2机器结果冻结报告

> 状态：`STAGE7L_E_MACHINE_RESULTS_FROZEN_READY_FOR_E3_REPORTING`

本报告只冻结E2机器结果；没有修改模型、checkpoint、planner、roster、task、kernel、阈值或Stage6V结论。完整论文叙事、统一BDD矩阵与Style Report Card更新留给E3。

## 1. 冻结范围与输入

- Stage7L-D六项解锁条件全部通过。
- 仅复用400条冻结official rollout；nuPlan planner未重新运行。
- 五档输入均为`[80,150,83]`、float32、finite，scenario order一致。
- old64、A3407、B3407、C3407均成功生成`[80,64]`；ego13成功生成`[80,13]`。
- `LAT.LANE_CHANGE`四档均80对；`LAT.DYNAMICS`四档均38对。

## 2. 预注册Primary

| Rep | Contrast | Task | N | raw MMD² | null q95 | BDD/q95 | Z_BDD | plus-one p | 结论 |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| B3407 | dose100−dose0 | LAT.LANE_CHANGE | 80 | 0.001075041 | 0.002466807 | 0.436× | -0.065 | 0.411906 | FAIL |

冻结解释：planner-level pure-lateral treatment confirmation成功，但Candidate B未通过预先指定的prospective paired BDD Primary endpoint。不得据此更换checkpoint、task、kernel或重新训练。

## 3. dose100同一Treatment五representation对照

raw MMD²只在各representation内部解释，禁止跨representation排序。

| Representation | BDD/q95 | Z_BDD | raw p | Multiplicity |
|---|---:|---:|---:|---|
| old64 | 0.580× | -0.646 | 0.719943 | Secondary（Holm p=1.000000） |
| A_seed3407 | 0.460× | -0.021 | 0.378166 | Secondary（Holm p=1.000000） |
| B_seed3407 | 0.436× | -0.065 | 0.411906 | Primary（不进入Holm） |
| C_seed3407 | 0.545× | 0.373 | 0.262247 | Secondary（Holm p=1.000000） |
| ego13 | 13.087× | 40.201 | 9.9999e-06 | Secondary（Holm p=0.000390） |

ego13在该kinematic-heavy treatment下具有最高within-null标准化敏感度；这不表示ego13是全局最佳或最完整的behavior representation。

## 4. 四档Z_BDD曲线

### LAT.LANE_CHANGE

| Representation | dose25 | dose50 | dose75 | dose100 |
|---|---:|---:|---:|---:|
| old64 | -0.069 | -0.721 | -0.786 | -0.646 |
| A_seed3407 | -1.298 | -0.671 | 0.420 | -0.021 |
| B_seed3407 | -0.914 | -0.818 | 0.433 | -0.065 |
| C_seed3407 | -1.434 | -0.693 | 0.615 | 0.373 |
| ego13 | 39.387 | 39.336 | 39.881 | 40.201 |

### LAT.DYNAMICS

| Representation | dose25 | dose50 | dose75 | dose100 |
|---|---:|---:|---:|---:|
| old64 | -1.010 | -1.869 | -0.445 | 0.048 |
| A_seed3407 | 1.029 | -0.562 | 1.797 | 2.000 |
| B_seed3407 | 1.919 | -1.094 | 1.574 | 1.514 |
| C_seed3407 | 1.623 | -1.107 | 1.700 | 2.219 |
| ego13 | 19.662 | 19.611 | 19.788 | 20.269 |

## 5. Multiplicity与边界

- 固定理论矩阵40格；唯一Primary移除后，secondary Holm家族39格。
- Holm通过8格，均为ego13的四档×两个task。
- `NOT_COMPUTABLE`：0格；`LOW_N_SECONDARY_DIAGNOSTIC`：20格。
- `LAT.DYNAMICS`仍是38场景的pre-treatment high-motion `MIXED_PROXY`，不是pure lateral dynamics ground truth。
- 未跨representation比较raw MMD²；未改变Stage6V资格结论；未重跑planner；未修改训练或checkpoint。

## 6. E2冻结结论

`STAGE7L_E_MACHINE_RESULTS_FROZEN_READY_FOR_E3_REPORTING`

下一步仅允许E3报告整合与论文表达，不允许任何rescue experiment或模型返工。

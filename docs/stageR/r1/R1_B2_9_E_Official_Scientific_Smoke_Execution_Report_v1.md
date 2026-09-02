# R1 B2.9-E Official Scientific Smoke Execution Report v1

## 执行结论

新的 B2.9-E versioned package 已按一次性重新授权完整执行。48/48 run technical complete，24/24 pair 由冻结 binding、safety adapter、dispatcher 与 Evaluator V2.1 自动评估。没有重试、replacement、threshold change 或 outcome-driven intervention。

本次没有 technical infrastructure failure。科学 gate failure 均按合同记录后继续，不能被解释为执行失败，也不能触发调参。

## 授权与冻结绑定

- 远端执行基线 commit：`628787063901d420f5367d95941bd8b4ad9ded29`
- owner authorization SHA256：`4467e1106ffd3d9bc98491e751b79e6760da289d5a862b2b25677ccd84b6e7e1`
- final manifest SHA256：`99dfafaa719c5b2ea454b46f637132e5d6e9c755c972bf5a15ec37442057c006`
- roster SHA256：`efe8e9d680ca0bcacb367bc9b616610ca78c260195e53b8f025a7bd1d92c23e6`
- schedule v3.1 SHA256：`99f44095c27319b746921376d2549a00186303298b5266ff45dd008a98c08455`
- pair binding v2.1 SHA256：`a606a87b01cd1fdd340070fca7e77170b6e0782aafa1e7c19ab6c91228cc9fa6`
- protected CSV SHA256：`e8deb93312e82183b6c2c0db30fd18cbf9c32d32d566038419a5be65b389d9d8`

## 技术执行闭包

- claimed attempts：48；`run_runners` executions：48；technical complete：48/48。
- Primary trace：48/48 精确 80 行，iteration 0...79，来源 `REALIZED_CURRENT_EGO`；secondary planner calls：0。
- metric lifecycle：48/48；runner report：48/48；official safety adapter：48/48；pair dispatcher：24/24。
- retry：0；replacement：0；scientific identities unchanged：true。

## 严格科学结果

| Family | Context | Applicability | Mechanism | F_match | Endpoint | Engineering | Safety | FAMILY_SMOKE_READY |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| R-HLC | 12/12 | N/A | 0/12 | 12/12 | 6/12 | 12/12 | 11/12 | FAIL |
| R-TSB | 12/12 | 0/12 | 0/12 | 12/12 | N/A | N/A | 11/12 | FAIL |

严格规则要求每族 12/12 pair 的全部 required gates 均通过。因此 HLC 与 TSB 均为 `FAIL`，不得采用平均、majority vote 或容忍单个失败。

## HLC 语义分离

Planner reference 保持 `ROUTE_CONTINUOUS_V2_3`；measurement reference 保持 `FROZEN_NATIVE_SOURCE_TARGET_MEASUREMENT_CONTRACT`。未用 planner corridor 替换 measurement references。

## Governance

B2.9-D 两个旧 attempts 与旧输出保持只读，未补 callback/parquet/evaluator，也未用于本次 scientific pair。B2.9-E authorization 已消费。无论本轮科学结果如何，`RBR_A/B/C = NOT_AUTHORIZED_PENDING_SCIENTIFIC_OWNER_RESULT_REVIEW`。

# Stage7L-C2：Task-Population Consistency Amendment

## 1. 时点与唯一范围

本修订在Stage7L-D启动前、任何confirmation rollout/result产生前完成。修订时未读取rollout、embedding、BDD或MMD，也未训练模型。C2只闭合task population与secondary cell computability；Stage7L-C/C1的roster、treatment、gates、failure policy、checkpoint、Primary统计规则均未改变。

C2是Stage7L-D前最后一次允许的protocol consistency amendment。除明确代码不可执行、SHA资产损坏或protocol逻辑上致命不可运行外，Stage7L-D开始后不得再修改scientific/statistical protocol，尤其不得根据显著性或结果表现修订。

## 2. LAT.LANE_CHANGE最终定义

`LAT.LANE_CHANGE`定义为完整冻结Stage7L prospective lane-change-opportunity confirmation roster，即全部80个scenario。其mask是roster membership恒真，不使用`changing_lane_to_left/right`或expert最终行为二次筛选。

这80个scenario已通过native adjacent lane、route compatibility、map-based lane-change opportunity、静态eligibility、15 s dynamic clearance和official runnability，足以prospectively定义controlled lane-change execution opportunity population。

task population不得使用expert是否换道、rollout completion、treatment success、yaw、lateral displacement、collision、embedding、BDD或MMD。每个contrast的analysis population仍为该task population中全部完整dose0+doseX且representation input可合法构造的pair，不replacement、不按outcome删样。

## 3. Primary与矩阵对应格identity

唯一Primary保持`B_seed3407 × dose100_vs_dose0 × LAT.LANE_CHANGE`。Primary与理论40格矩阵中同名格共享完全相同的roster population、完整pair规则、representation、contrast、task、bandwidth、paired null和100,000 swaps。

其规范化cell definition SHA256固定为：

```text
283d4c0abd55990f99c5a1c4080c61667e395de7e4e2156cdae0a34c50427b1b
```

protocol在Primary与secondary excluded-cell两个位置引用同一SHA，validator必须验证两者等于现场规范化定义。因此理论40格只精确排除一次Primary，得到39个secondary tests。

## 4. LAT.DYNAMICS预处理mask

`LAT.DYNAMICS`是独立secondary mixed proxy。mask只读取冻结Pool B中的`official_scenario_types_json`，只要命中以下任一官方pre-treatment标签即为true：

- `high_lateral_acceleration`
- `high_magnitude_speed`
- `medium_magnitude_speed`

规则版本为`stage7l_c2_pretreatment_task_masks_v1`，生成工具为`tools/stage7l_generate_pretreatment_task_masks.py`。禁止使用Stage7L rollout、treatment response、处理后lateral acceleration/yaw、embedding、BDD或MMD。该维度必须标记`MIXED_PROXY`，不得描述为pure-lateral evidence。

在当前冻结资产上重放得到：`LAT.LANE_CHANGE=80/80`、`LAT.DYNAMICS=38/80`；有序token mask SHA分别为`99063af8910597cdf81d82086c13bd13fb489e3f74172f4b2e816a03e62ea0d1`和`2840d14d5d4083d7ed890363f0f91ee7d4b9e90aa176836b1f9627741508be9b`，二者不同。Stage7L-E必须报告mask SHA；若实现复用同一parent BDD，必须`shared_parent_bdd=true`并禁止重复计数。

## 5. 39-test family与不可计算规则

理论矩阵继续为5 representations×4 dose contrasts×2 task views=40格。排除一次唯一Primary后，secondary family永久为单一39-test Holm family。

若task population为空、无完整dose0+doseX pair或representation input无法合法构造，cell不得删除，固定：

```text
status = NOT_COMPUTABLE_PRE_FROZEN_TASK_POPULATION
raw_p_for_multiplicity = 1.0
```

不得缩小family、替换scenario或修改task mask。若cell可合法计算但N较小，则正常计算、报告真实`N_pair(task,doseX)`并标记`LOW_N_SECONDARY_DIAGNOSTIC`；不新增事后minimum-N门槛。

未来必须报告两个task各四个dose的`N_pair(task,doseX)`。Primary仍要求`N_pair(LAT.LANE_CHANGE,dose100)≥76`，raw/plus-one paired randomization p<0.05，且不进入Holm。

## 6. 不变性与状态

roster仍为80 scenarios、15 left、65 right、79 logs，SHA256仍为`90ec9b427636cefc59e6d7ace2507ac8364747e2a38964124be08fdc2a10acf9`。dose、trigger、planner、eligibility、dynamic clearance、buffer、mechanism/nuisance/safety gates、failure policy、minimum complete=76、Primary minimum pair=76、checkpoint、seed、paired null、100,000 swaps与Primary p规则均未改变。

最终状态：

```text
STAGE7L_C2_TASK_POPULATION_CONSISTENCY_AMENDMENT_FROZEN
STAGE7L_C1_PROTOCOL_CONSISTENCY_AMENDMENT_FROZEN
STAGE7L_C_PROSPECTIVE_PROTOCOL_FROZEN
STAGE7L_C_CONFIRMATION_ROSTER_FROZEN
STAGE7L_D_ONE_TIME_CONFIRMATION_AUTHORIZED
STAGE7L_D_NOT_STARTED
```

本修订不启动Stage7L-D。

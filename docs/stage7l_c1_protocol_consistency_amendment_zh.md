# Stage7L-C1：Protocol Consistency Amendment

## 1. Amendment时点与边界

本修订在Stage7L-D启动前、任何confirmation rollout/result产生前完成。修订时`stage7l_d_started=false`，未读取embedding、BDD或MMD，也未训练模型。

Stage7L-C的80场景roster保持不变：15 left、65 right、79 unique logs；roster SHA256仍为`90ec9b427636cefc59e6d7ace2507ac8364747e2a38964124be08fdc2a10acf9`。dose、trigger、eligibility、dynamic clearance、buffer、mechanism/nuisance/safety gate、checkpoint、seed、paired null和Primary endpoint科学定义均未改变。

原protocol SHA256：`ae4c1a3ea639d12c9d5f257d87b07e3442e4b22f11c199e40d14f8dab407d125`。

修订后protocol SHA256：`55eb0fe0cd606cd9607521439e163886bb9028c13a8c609762f12f71e65ef94f`。

## 2. Amendment A：76–79 complete case

设计人口永久为`N_design=80`，用于protocol identity、execution reporting、safety denominator、missing/runtime accounting和no-replacement audit。official completed少于76/80时，保持原状态`STAGE7L_D_CONFIRMATION_EXECUTION_INSUFFICIENT`，Stage7L-E不解锁。

对于每个`doseX vs dose0`，BDD analysis population定义为冻结80场景中同时具备完整dose0、完整doseX且required representation input存在并可合法构造的全部pair，记为`N_pair(doseX)`。不得replacement，也不得按collision、off-road、lane-change incomplete、BDD或embedding质量删样。treatment outcome failure只要representation input仍存在就必须进入BDD；只有input事实缺失或无法合法构造才是non-analyzable。

Primary要求`N_pair(dose100)≥76`。少于76时状态为`STAGE7L_E_PRIMARY_BDD_INSUFFICIENT_COMPLETE_PAIRS`，不得声明Primary成功。

未来结果必须同时报告：`N_design=80`、`N_complete_all_five_doses`、`N_pair(dose25/50/75/100)`，以及infrastructure/runtime、treatment outcome、invalid/incomplete、other pre-frozen category四类missing reason。official success/completion/responsible collision/off-road仍按原冻结80场景规则计算，不能改用BDD pair数。

## 3. Amendment B：39-test secondary family

唯一Primary保持`B seed3407 × dose100-vs-dose0 × LAT.LANE_CHANGE`，使用未校正raw/plus-one p，表中固定标记`PRIMARY_NOT_PART_OF_SECONDARY_HOLM_FAMILY`。

完整矩阵为5个representation（old64、A3407、B3407、C3407、ego13）×4个dose contrast×2个task view=40格。排除唯一Primary格后，冻结为单一39-test Holm secondary family。B必须保留dose25/50/75/100完整dose curve；其Primary格不再进入Holm，其他B格均进入。

现有Stage7实现把`LAT.LANE_CHANGE`映射为`lane_change` scope（changing_lane_to_left/right），把`LAT.DYNAMICS`映射为`high_motion_dynamics` scope（high_lateral_acceleration/high/medium_magnitude_speed）。两者由不同pre-treatment official type集合生成mask；后者是mixed proxy，不能解释成pure-lateral因果证据。Stage7L-E必须保存两个mask的SHA；只有mask确实不同才可作为两个test。若复用同一parent BDD，则必须标记`† shared parent BDD`并不得重复计为独立检验。

raw MMD²仍禁止跨representation排序；跨representation只比较BDD/q95、Z_BDD、detection、minimum detectable dose和task coverage。

## 4. Semantic CI展示规则

duration、RMS lateral acceleration、peak yaw和nuisance summary的95% CI固定使用log-cluster percentile bootstrap：cluster=`log_name`、10,000 replicates、seed=`620272`。该规则只用于CI/uncertainty reporting，不改变已冻结的paired median direction、directional consistency或nuisance门槛，也不改变Primary paired randomization。

## 5. 不变性与最终状态

自动验证必须确认roster SHA、80/15/65、development scenario/log overlap=0、dose/trigger/eligibility/checkpoint/Primary科学定义均未变；minimum complete=76、Primary minimum pair=76、secondary family=39、Primary排除于Holm、Stage7L-D未启动。

冻结状态：

```text
STAGE7L_C1_PROTOCOL_CONSISTENCY_AMENDMENT_FROZEN
STAGE7L_C_PROSPECTIVE_PROTOCOL_FROZEN
STAGE7L_C_CONFIRMATION_ROSTER_FROZEN
STAGE7L_D_ONE_TIME_CONFIRMATION_AUTHORIZED
STAGE7L_D = NOT_STARTED
```

本修订不授权自动启动Stage7L-D。

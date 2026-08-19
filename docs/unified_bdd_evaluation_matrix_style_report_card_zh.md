# 统一 BDD Evaluation Matrix 与 Style Report Card 规范

> Schema：`unified_bdd_reporting_schema_v1`
> 状态：`UNIFIED_BDD_REPORTING_SCHEMA_FROZEN`
> 冻结日期：2026-08-14
> 适用范围：此后全部 BDD、MMD²、task-conditioned drift 和 representation capability 报告

本规范只统一已有和未来结果的表达方式，不修改任何冻结实验、统计值、门槛或模型结论。本阶段没有训练、仿真、embedding 导出或 BDD 重算。

机器可读定义：

- [`configs/unified_bdd_reporting_schema_v1.json`](../configs/unified_bdd_reporting_schema_v1.json)
- [`configs/unified_bdd_stage_task_mapping_v1.csv`](../configs/unified_bdd_stage_task_mapping_v1.csv)

## 1. 三个概念必须分离

### 1.1 Behavior Drift Profile

回答：

> Target 相对 Reference 到底哪些驾驶行为发生了变化，变化方向是什么？

输出主体是固定 Behavior Drift Matrix。BDD 负责说明某一 task/slice 上的表示分布是否变化，semantic delta 负责说明变化方向。没有 semantic delta 时，只能写“存在分布差异，方向未判定”。

### 1.2 BDD Statistic

回答：

> 在固定 Reference、Target、task、representation 和 null/calibration 下，两组行为分布差异有多显著？

统一记法：

`BDD(Target | Reference, task, representation, evaluation_mode)`

BDD 在本项目中通常由冻结 kernel 下的 MMD²、null q95、Z_BDD 和随机化 p-value 表达。它不自带“激进/保守”“好/坏”“安全/危险”方向。

### 1.3 Representation Evaluation

回答：

> old64、A、B、C、ego13 等 representation，谁更可靠地检测一个已知 behavior treatment？

允许比较 detection rate、A/A FPR、detection−FPR、minimum detectable dose、task coverage、各表示自身 null 下的 normalized Z 和 seed stability。禁止跨 representation 比较 raw MMD²。

Stage6J/K、Stage6P 的门禁和检出率主要属于 Representation Evaluation；它们不能直接替代 Behavior Drift Profile。

## 2. Reference、Target 与符号规范

### 2.1 强制字段

任何 BDD 行必须显式给出：

1. `reference_id`：基准 planner/version/release 的不可歧义 ID；
2. `reference_role`：为什么它是 Reference，例如 frozen baseline、long-headway、conservative、release-N；
3. `target_id`：被比较 planner/version/release 的不可歧义 ID；
4. `target_role`：例如 candidate、short-headway、assertive、release-N+1；
5. `task_id` 与完整 scenario slice 定义；
6. `evaluation_mode`：paired 或 unpaired；
7. `representation_id` 和 checkpoint/version；
8. null/calibration 方法。

禁止只写“B 模型 Z_BDD=10”。合格写法示例：

`BDD(pdm_closed_assertive_longitudinal_v1 | pdm_closed_conservative_longitudinal_v1, following_interaction, old64, paired)`

### 2.2 唯一差值方向

全部 semantic delta 固定为：

`Δsemantic = Target − Reference`

当前既有结果的统一重述为：

| 既有实验 | Reference | Target | 统一 delta |
|---|---|---|---|
| Stage7 assertive/conservative | `pdm_closed_conservative_v1` | `pdm_closed_assertive_v1` | assertive − conservative |
| Stage6J/K纯纵向 | `pdm_closed_conservative_longitudinal_v1` | `pdm_closed_assertive_longitudinal_v1`或相应dose | dose/ assertive − conservative |
| Stage6S-v3 interaction | `pdm_closed_interaction_long_headway_v2` | `pdm_closed_interaction_short_headway_v2` | short − long |
| Stage6P双方向release | 每个trial显式指定 | 每个trial显式指定 | Target release − Reference release；聚合检出率不提供语义方向 |

### 2.3 BDD的对称性与方向

MMD² 对 A/B 标签交换通常对称，因此 Reference/Target 顺序不会使 raw BDD 自动变号。Reference/Target 仍必须固定，因为：

- semantic delta 有符号；
- 业务解释必须知道谁相对谁变化；
- unpaired monitoring 需要报告两个方向的稳定性；
- 同一结果不能在文字中随意反转“更快/更慢”。

## 3. 冻结 Behavior Drift Matrix taxonomy

以下 13 个叶子维度及其层级长期固定。未来允许增加 semantic metric，不允许临时改维度名称、把 proxy 改称 exact，或因某次实验没有数据就删除该行。

| ID | 一级维度 | 固定行为维度 | 中文含义 | 首选semantic指标 |
|---|---|---|---|---|
| `OVR.ALL` | Overall | overall behavior drift | 总体行为漂移 | 多指标bundle；无单一方向 |
| `LON.FREE_FLOW_SPEED` | Longitudinal | free-flow speed | 无持续可信前车时的速度选择 | free-flow mean/median speed |
| `LON.ACCEL_DECEL` | Longitudinal | acceleration/deceleration | 加速、减速及纵向动态强度 | positive accel、peak decel、RMS accel、mean speed |
| `LON.CAR_FOLLOWING` | Longitudinal | car-following | 跟车工况下的ego纵向行为 | following speed、accel、gap、finite THW |
| `LON.CLOSING_RESPONSE` | Longitudinal | closing response | 相对逼近前车时的响应 | closing accel、reaction delay、peak decel、TTC |
| `LON.COMFORT` | Longitudinal | longitudinal comfort | 纵向平顺性/激励强度 | RMS jerk、peak jerk、accel variability |
| `LAT.LANE_KEEPING` | Lateral | lane keeping | 车道保持与横向偏差 | lateral offset、heading error、boundary margin |
| `LAT.LANE_CHANGE` | Lateral | lane change | 变道频次、时长与形态 | event rate、duration、peak lateral accel/yaw rate |
| `LAT.DYNAMICS` | Lateral | lateral dynamics | 横向动态强度和平顺性 | RMS lateral accel、yaw rate、curvature |
| `INT.FRONT_GAP_THW` | Interaction | front-gap / THW interaction | 与前车距离和车头时距 | median front gap、finite THW |
| `INT.LONG_FOLLOWING` | Interaction | longitudinal following interaction | following pressure下的条件响应 | following-pressure accel、relative closing、delay |
| `INT.LATERAL_GAP` | Interaction | lateral gap acceptance / lateral interaction | 变道或横向交互的间隙接受 | accepted front/rear gap、gap acceptance score |
| `INT.MERGE_YIELD_CUTIN` | Interaction | merge / yield / cut-in | 汇入、让行与切入响应 | yield rate、cut-in delay、min gap、conflict accel |

### 3.1 维度行不可省略

每份完整 Style Report Card 必须保留13行。无可用样本或结果时填写 N/A，并给出固定 reason code。这样业务用户能区分“没有变化”“没有检出”和“根本没有证据”。

### 3.2 一个BDD支持多个语义子维度时

例如 Stage6S-v3 在一个 following interaction roster 上只计算一次 full-embedding BDD，但同时有 front-gap/THW、closing response 和 following-pressure accel 三组 semantic metrics。允许在表A展开为三个维度行，但必须：

- 使用同一个 `parent_bdd_result_id`；
- 明确“shared task-level BDD”；
- 不把三行计作三次独立检验；
- 每行只用本维度允许的 semantic delta 解释方向。

## 4. Stage/task 到统一维度的固定mapping

完整逐行映射见 `configs/unified_bdd_stage_task_mapping_v1.csv`。核心映射如下：

| 既有task/group | 主维度 | 证据强度 | 固定解释边界 |
|---|---|---|---|
| Stage7 `overall` | `OVR.ALL` | exact overall | planner-conditioned总体分布差异 |
| Stage7 `following_interaction` | `INT.LONG_FOLLOWING` | task-slice proxy | 同时可作为`LON.CAR_FOLLOWING`共享BDD；需task-specific semantic delta才能定方向 |
| Stage7 `lane_change` | `LAT.LANE_CHANGE` | task-slice proxy | scenario type不证明ego在rollout中实际完成变道 |
| Stage7 `stop_go_control` | `LON.ACCEL_DECEL` | task-slice proxy | comfort只能作为secondary proxy |
| Stage7 `high_motion_dynamics` | `LAT.DYNAMICS` | mixed proxy | 混合high lateral acceleration与speed magnitude，不能称为纯横向BDD |
| Stage7 `dense_or_vulnerable_interaction` | `INT.MERGE_YIELD_CUTIN` | insufficient | 只能写 broad dense/vulnerable drift，不能写具体merge/yield/cut-in已验证 |
| Stage6J/K `overall` | `LON.ACCEL_DECEL` | treatment-aligned proxy | planner差异为纯纵向，但场景包含多种task |
| Stage6J/K `following_interaction` | `LON.CAR_FOLLOWING` | task-slice proxy | 可报告跟车BDD和following slice semantic delta |
| Stage6J/K `longitudinal_high_motion` | `LON.ACCEL_DECEL` | task-slice proxy | 已排除high-lateral场景 |
| Stage6J/K `stop_go_control` | `LON.ACCEL_DECEL` | task-slice proxy | comfort为共享BDD的secondary semantic维度 |
| Stage6P context-balanced overall | `OVR.ALL` | representation evaluation only | 聚合A/B detection不能回答具体哪个行为维度变了 |
| Stage6W signal/noise decomposition | `OVR.ALL` | representation evaluation only | 解释检测能力，不解释Target行为方向 |
| Stage6S-v3 following confirmation | `INT.LONG_FOLLOWING` | exact treatment/task | 可共享到`INT.FRONT_GAP_THW`和`LON.CLOSING_RESPONSE` |

### 4.1 Stage6C v2旧taxonomy

Stage6C v2 的 following 与 yield-conflict 是相对可靠的detector；lead-brake、queue、cut-in、overtake、部分lane-change/hesitation仍含proxy。迁移时必须保留原 `strong/proxy/weak_proxy/unknown` strength，不能只保留统一维度名。

- `task_following` → `INT.LONG_FOLLOWING`，secondary为`LON.CAR_FOLLOWING`；
- `task_lead_brake_response` → `LON.CLOSING_RESPONSE`，当前为closing-derivative proxy；
- `task_queue_approach` → `LON.ACCEL_DECEL`，secondary为`LON.COMFORT`；
- `task_lane_change` → `LAT.LANE_CHANGE`，secondary为`LAT.DYNAMICS`与`INT.LATERAL_GAP`；
- `task_cutin_response` → `INT.MERGE_YIELD_CUTIN`，但当前不足以形成exact claim；
- `task_yield_conflict` → `INT.MERGE_YIELD_CUTIN` proxy；
- `task_hesitation` → `LAT.DYNAMICS` proxy。

### 4.2 Stage5 feature groups

Stage5的following/longitudinal/lateral/behavior_proxy是训练和representation validation group，不是Reference→Target BDD行为报告。它们只进入Representation Scorecard，不可填入表A并声称某种驾驶行为发生漂移。

## 5. 每一行的固定字段

完整机器字段共48项，见schema JSON。报告不得少于下列核心字段：

| 字段组 | 必填内容 |
|---|---|
| Identity | schema version、report ID、result ID、parent BDD result ID |
| Behavior | dimension ID、中文/英文维度名、mapping strength、value status |
| Contrast | Reference ID/role、Target ID/role、Reference→Target label |
| Slice | task ID、scenario slice定义、pre/post-treatment选择时点 |
| Mode | paired/unpaired、pairing unit或cluster unit |
| Sample | N reference、N target、N pair、N scenario、N log；不适用项填N/A |
| Representation | representation ID、checkpoint/version SHA |
| Statistic | statistic name、raw MMD²、kernel、bandwidth、null/calibration、repetitions、null q95、Z_BDD |
| Inference | raw p、corrected p、multiplicity family、alarm threshold、detection/alarm |
| Semantics | metric、Target−Reference delta、unit、95% CI、direction |
| Interpretation | 一句业务结论、proxy/quality限制、provenance路径和SHA |

### 5.1 paired字段规范

paired报告必须提供：

- 同一场景的Reference与Target rollout；
- `n_pairs`、`n_scenarios`、`n_logs`；
- pair key，一般为scenario token；
- pair内标签交换或其他冻结null；
- task是否pre-treatment选择；
- log-cluster uncertainty是否用于semantic delta。

paired的`n_reference=n_target=n_pairs`；若存在缺失，必须报告complete pairs和缺失原因，不能仅删除失败行。

### 5.2 unpaired字段规范

unpaired报告必须提供：

- Reference release与Target release各自的scenario/log数量；
- log/scenario overlap规则；
- A/A calibration来源和holdout方式；
- 每个representation自己的阈值；
- A/A FPR、A/B detection、detection−FPR；
- 两个A/B方向；
- 重复pseudo-release次数和cluster unit。

聚合 detection rate 属于表B。若要进入表A，必须输出一个具体、可追溯的Reference release与Target release BDD行，并另有同slice semantic delta；不能用200次trial的告警率代替行为方向。

## 6. BDD与semantic delta绑定规则

### 6.1 基本规则

1. BDD显著、semantic delta显著：可写“该维度存在漂移，Target表现为……”。
2. BDD显著、semantic delta缺失：只写“该task slice存在表示分布差异，方向为N/A”。
3. BDD不显著、semantic delta显著：写“某已知指标变化，但当前representation BDD未检出总体分布差异”。
4. BDD和semantic均不显著：写“当前样本与协议下未获得变化证据”，不能写“完全相同”。
5. semantic metrics方向冲突：Direction=`MIXED`，逐项列出，不强行贴“激进/保守”。

### 6.2 固定方向词汇

方向必须是可观测描述，例如：

- `Δspeed > 0`：Target更快；
- `Δfront gap < 0`：Target保持更短前车间距；
- `Δfinite THW < 0`：Target保持更短车头时距；
- `ΔRMS accel > 0`：Target纵向动态强度更高；
- `ΔRMS jerk > 0`：Target纵向jerk更高/更不平顺；
- `Δlane-change duration < 0`：Target变道时长更短；
- `Δlateral accel > 0`：Target横向动态强度更高；
- `Δaccepted gap < 0`：Target接受更小横向间隙。

“更激进”“更保守”“更舒适”“更安全”只有在预先冻结的复合定义及全部所需指标同时满足时才允许作为secondary label；默认不使用。

## 7. 表A：BDD Behavior Profile / Style Report Card

### 7.1 业务总表固定格式

| Behavior dimension | Reference→Target | Task/slice | Mode | N scenario/log | Representation | Z_BDD | 显著性 | Semantic Δ | Direction | Conclusion |
|---|---|---|---|---|---|---:|---|---|---|---|
| 固定taxonomy行 | 明确版本ID | 明确task | paired/unpaired | N/N | 明确representation | 数值或N/A | raw/corrected p | Target−Reference及CI | 固定方向词 | 一句话边界化结论 |

正式机器表还必须保留raw MMD²、null q95、bandwidth、calibration、checkpoint SHA和provenance；业务表可以折叠这些审计字段，但不能丢失Reference/Target和方向依据。

### 7.2 使用当前冻结结果填充的示例Report Card

以下示例把已有结果重新表达为统一schema，没有重算统计量。raw MMD²只用于单行审计，不用于跨representation排序。`Legacy N/A`表示历史冻结结果未保存null q95/Z，不能事后用其他scope替代。

| Dimension | Reference→Target | Slice / mode | N scenario/log | Rep. | raw MMD² / null q95 / Z | p / corrected p | Semantic Δ Target−Reference (95% CI) | Direction | 统一结论 |
|---|---|---|---|---|---|---|---|---|---|
| `OVR.ALL` 总体 | conservative v1 → assertive v1 | 5-task locked confirmation / paired | 310/257 | old64 | 0.004469 / Legacy N/A / Legacy N/A | 9.9999e-6 / N/A | mean speed +1.281 m/s [1.122,1.446]；RMS accel +0.235 m/s² [0.206,0.263] | 更快且纵向动态更高；总体方向仍为mixed | 总体behavior distribution显著漂移；不代表安全或planner优劣 |
| `LON.FREE_FLOW_SPEED` 自由流速度 | N/A | fixed dimension | N/A | N/A | N/A | N/A | N/A | N/A | `EVIDENCE_GAP_BDD_NOT_COMPUTED`：没有冻结的纯free-flow task BDD |
| `LON.ACCEL_DECEL` 纵向加减速 | conservative-longitudinal → assertive-longitudinal | pure longitudinal dose100 / paired | 183/156 | old64 | 0.005001 / 0.002096 / 9.228 | 9.9999e-6 / 3.99996e-5 | speed +0.915 m/s [0.758,1.078]；RMS accel +0.182 m/s² [0.146,0.217] | 更快、纵向动态更高 | 纯纵向处置产生显著总体纵向漂移 |
| `LON.CAR_FOLLOWING` 跟车 | conservative-longitudinal → assertive-longitudinal | following_interaction dose100 / paired | 60/52 | old64 | 0.017067 / 0.009227 / 5.634 | 6.8999e-4 / 0.006900 | following speed +0.917 m/s [0.625,1.246]；RMS accel +0.234 m/s² [0.187,0.280] | 跟车slice中更快、纵向动态更高；gap方向不稳定 | 跟车BDD显著，方向由speed/accel支持，不宣称稳定更短THW |
| `LON.CLOSING_RESPONSE` 逼近响应 | long-headway v2 → short-headway v2 | following confirmation / paired | 80/11 | old64 | 0.063202 / 0.009246 / 27.976 | 9.9999e-6 / N/A | closing accel +0.085 m/s² [0.022,0.450] | closing时维持更多加速度 | shared task-level BDD；closing response方向有semantic支持 |
| `LON.COMFORT` 纵向平顺性 | conservative-longitudinal → assertive-longitudinal | pure longitudinal dose100 / paired | 183/156 | old64 | shared parent: 0.005001 / 0.002096 / 9.228 | shared parent | RMS jerk +0.228 m/s³ [0.142,0.319] | Target纵向jerk更高 | 使用与LON.ACCEL_DECEL相同parent BDD，不是独立comfort BDD检验 |
| `LAT.LANE_KEEPING` 车道保持 | N/A | fixed dimension | N/A | N/A | N/A | N/A | N/A | N/A | `EVIDENCE_GAP_BDD_NOT_COMPUTED` |
| `LAT.LANE_CHANGE` 变道 | conservative v1 → assertive v1 | lane_change scenario slice / paired | 60/60 | old64 | 0.028784 / Legacy N/A / Legacy N/A | 8.9999e-5 / 3.59996e-4 | task-specific lane-change semantic delta未冻结 | N/A | 变道场景slice BDD显著，但无法判定变道频次、时长或sharpness方向 |
| `LAT.DYNAMICS` 横向动态 | conservative v1 → assertive v1 | high_motion_dynamics / paired | 60/59 | old64 | 0.014453 / Legacy N/A / Legacy N/A | 1.39999e-4 / 4.19996e-4 | task-specific lateral delta未冻结 | N/A | `MIXED_PROXY`：BDD显著，但slice混合高横向加速度和速度幅值，不能称纯横向BDD |
| `INT.FRONT_GAP_THW` 前车间距/THW | long-headway v2 → short-headway v2 | following confirmation / paired | 80/11 | old64 | shared parent: 0.063202 / 0.009246 / 27.976 | 9.9999e-6 / N/A | median gap −4.202 m [−5.791,−1.181]；finite THW −2.670 s [−3.687,−2.275] | Target间距和THW更短 | interaction mechanism及BDD均有证据；与下一行共享parent BDD |
| `INT.LONG_FOLLOWING` 纵向跟车交互 | long-headway v2 → short-headway v2 | following confirmation / paired | 80/11 | old64 | 0.063202 / 0.009246 / 27.976 | 9.9999e-6 / N/A | following-pressure accel +0.085 m/s² [0.022,0.450] | pressure下维持更多加速度 | 该interaction treatment存在显著behavior drift |
| `INT.LATERAL_GAP` 横向间隙接受 | N/A | fixed dimension | N/A | N/A | N/A | N/A | N/A | N/A | `EVIDENCE_GAP_SEMANTIC_DELTA_NOT_COMPUTED`：lane-change slice没有冻结gap-acceptance delta |
| `INT.MERGE_YIELD_CUTIN` 汇入/让行/切入 | conservative v1 → assertive v1 | dense_or_vulnerable proxy / paired | 63/57 | old64 | 0.013792 / Legacy N/A / Legacy N/A | 0.001290 / 0.002580 | specific merge/yield/cut-in delta未冻结 | N/A | `PROXY_ONLY_NOT_CONFIRMATORY`：只能报告broad interaction drift，具体维度仍为evidence gap |

这个示例恢复了用户可读的总体、纵向、跟车、变道、横向和interaction BDD，同时明确指出哪些方向是有semantic证据的，哪些只是task-slice proxy。

## 8. 表B：BDD Evaluator / Representation Scorecard

### 8.1 固定格式

| Representation | Longitudinal | Following | Lane change | Interaction | Unpaired release | FPR | Capability conclusion |
|---|---|---|---|---|---|---|---|
| ID/checkpoint | min dose + task coverage | task coverage / normalized Z | detection/coverage | normalized Z + increment test | detection rate | A/A holdout FPR | 使用范围与限制 |

表B只评价检测器。它不回答Target更快、更近还是更平顺。

### 8.2 当前old64/A/B/C/ego13示例Scorecard

| Representation | Pure-longitudinal paired | Following dose cells | Lane change | Interaction confirmation | n=400 context-balanced release | A/A FPR | detection−FPR | Seed stability / 结论 |
|---|---|---|---|---|---:|---:|---:|---|
| old64 | 4/4 overall；7/12 task×dose；min dose 0.25 | 2/4 | Holm pass于旧310-pair的60场景；无公平cross-rep矩阵 | Z=27.976，detected | 66.5% | 5.0% | 61.5 pp | 历史baseline；release能力不足 |
| A | 4/4 overall；7/12；min dose 0.25 | 4/4 | N/A：未按相同locked lane-change协议评估 | Z=26.454，detected | 90.5% | 3.0% | 87.5 pp | 三seed detection 90.5%–97.0%；数据修复候选 |
| B | 3/4 overall；2/12；min dose 0.50 | 1/4 | N/A | Z=30.603，detected | 100.0% | 5.0% | 95.0 pp | 三seed均100%；最强且最简单release-level learned候选，但paired未通过 |
| C | 3/4 overall；2/12；min dose 0.50 | 1/4 | N/A | Z=28.955，detected；full−neighbor-zero ΔZ=−7.852，CI [−33.393,29.219] | 99.5% | 6.5% | 93.0 pp | 三seed均99.5%；未证明context增量 |
| ego13 | 4/4 overall；12/12；min dose 0.25 | 4/4 | N/A | Z=35.905，detected | 100.0% | 2.0% | 98.0 pp | controlled-longitudinal诊断参考；不是完整context style模型 |

禁止根据此表中的raw MMD²评价representation；因此表B默认不显示raw MMD²。normalized Z必须标明“各自null下”，不能被解释为统一物理距离。

## 9. N/A与evidence-gap规则

### 9.1 固定reason codes

| Code | 含义 | 示例 |
|---|---|---|
| `N/A_NO_ELIGIBLE_SAMPLES` | 固定维度存在，但该报告没有达到样本门槛 | 无有效merge场景 |
| `N/A_NOT_APPLICABLE_TO_SLICE` | 维度与当前task结构上不适用 | pure free-flow报告中的lane-change |
| `EVIDENCE_GAP_BDD_NOT_COMPUTED` | 有潜在数据或semantic metric，但没有冻结BDD | 当前lane keeping |
| `EVIDENCE_GAP_SEMANTIC_DELTA_NOT_COMPUTED` | BDD存在，但没有同task的semantic delta | Stage7 lane-change direction |
| `EVIDENCE_GAP_REPRESENTATION_NOT_EVALUATED` | 某representation没有运行该冻结协议 | A/B/C/ego13 lane-change公平比较 |
| `LEGACY_FIELD_NOT_ARCHIVED` | 历史冻结结果未保存新schema字段 | Stage7 task null q95/Z |
| `PROXY_ONLY_NOT_CONFIRMATORY` | 只有proxy，不能形成exact dimension claim | dense/vulnerable→merge/yield/cut-in |

### 9.2 禁止的填充方式

- 不得用0代替N/A；
- 不得用overall semantic delta填充task-specific方向；
- 不得从另一representation复制null q95或Z；
- 不得为补齐旧表而重新读取blind embedding并计算新主指标；
- 不得把“未评估”写成“未检出”；
- 不得把“未检出”写成“没有差异”。

## 10. paired与unpaired不能合并为单一BDD分数

同一representation必须分别保留：

- paired task sensitivity；
- unpaired release detection；
- A/A FPR；
- sample size/dose曲线；
- interaction increment diagnostics。

Stage6W已经证明历史paired/unpaired分离主要来自treatment、task、pool和estimand，不是配对统计本身。统一schema的目的不是把二者平均，而是让读者清楚每个结论回答哪个问题。

## 11. 报告质量门禁

最终报告必须能直接回答：

1. 谁是Reference，谁是Target？
2. 哪个固定行为维度发生变化？
3. 变化有多显著，null是什么？
4. semantic delta显示的方向是什么？
5. 哪些task/slice差异最大？
6. 哪个representation对该已知treatment最可靠？
7. 结论来自paired还是unpaired？

新实验中任一问题无法回答时，状态为`SCHEMA_INVALID_INCOMPLETE_REPORT`。历史结果允许标记`LEGACY_PARTIAL_SCHEMA`，但所有缺失字段必须有reason code。

### 11.1 自动/人工检查顺序

1. 先检查Reference/Target、task、mode和N；
2. 再检查representation、kernel、null/calibration与multiplicity；
3. 再检查BDD显著性；
4. 最后读取semantic delta解释方向；
5. 单独读取表B评价representation；
6. 检查proxy、quality warning和evidence gap；
7. 生成业务表A和研究表B。

## 12. 对既有冻结结论的影响

本次只重组表达，不改变任何结论：

- Stage6J/K仍表明ego13的controlled longitudinal sensitivity最强；
- Stage6P仍表明A/B/C显著改善release-level detection；
- Stage6W仍表明B/C提升主要来自signal增强；
- Stage6S-v3仍表明interaction mechanism通过，但C没有证明context增量；
- Stage6V联合决策仍为`NO_ABC_CANDIDATE_QUALIFIES_UNDER_PRE_FROZEN_RULE`；
- 论文实验仍冻结，不启动新模型或新仿真。

## 13. 冻结结论

自本规范冻结后，所有新BDD报告必须：

- 使用13维固定taxonomy；
- 显式写Reference、Target、task、mode、representation和null；
- 并列输出BDD与semantic delta；
- 分开生成Behavior Profile表A和Representation Scorecard表B；
- 对缺失维度使用N/A/evidence-gap reason code；
- 禁止跨representation比较raw MMD²。

`UNIFIED_BDD_REPORTING_SCHEMA_FROZEN`

## 14. 最终报告体系冻结（v2）

v1的13维taxonomy、字段、mapping与历史输出保持不变；最终控制层升级为
`configs/unified_bdd_reporting_schema_v2.json`。v2不重新定义统计量，只冻结以下最后一次表达清理。

### 14.1 两页固定结构

第一页固定为`Behavior Drift / Style Report Card`，顶部必须独立显示：

- **Behavior Reference**；
- **Target**；
- **Evaluation mode**；
- **Primary Representation**；
- **Null Reference**。

当前最终报告的`Primary Representation = B`。B只负责测量Behavior Reference→Target漂移，不是被测planner/version。
第一页主体不得进行representation优劣排名。第二页固定为`Representation Qualification Matrix`，分别报告
old64/A/B/C/ego13的固定treatment标准化敏感度、Stage6P n=400 detection/FPR、paired/unpaired/Waymo/
interaction/Stage6V联合门禁和适用边界。

### 14.2 三类Reference

以后永久只使用：

1. **Behavior Reference**：与Target共同定义变化对象和Target−Reference方向；
2. **Null Reference**：paired randomization q95或unpaired A/A calibration q95，`BDD/null-q95=1.0×`为统计背景线；
3. **Representation Baseline**：old64历史能力baseline，不定义行为方向。

禁止把三者混写为模糊的“reference BDD”。

### 14.3 shared-parent BDD

`LON.CLOSING_RESPONSE`、`INT.FRONT_GAP_THW`与`INT.LONG_FOLLOWING`共享同一Stage6S-v3 task-level BDD。
主矩阵所有相关单元格必须追加`†`，并固定使用表下注释：

> † These semantic dimensions share the same parent task-level BDD and are not independent BDD tests.

机器长表和最终审计表必须按representation保留相同`parent_bdd_result_id`。三条semantic row只计一次独立BDD检验，
不得写成三次独立发现。

### 14.4 标准化敏感度列与ego13边界

原`Best capability`永久改名为`Highest standardized sensitivity on this treatment`，中文为
`该Treatment下最高标准化检测敏感度`。它只表示特定已知treatment下相对各representation自身null的标准化敏感度，
不表示完整、通用或全局最优representation。

ego13在多个controlled treatments中具有最高within-null标准化敏感度，但这些treatment大量直接作用于ego运动学。
因此不能称ego13为通用style representation，不能据此宣称neighbor/context无价值。learned64的主要强正证据仍包括
production-style unpaired release monitoring；representation能力必须按deployment/evaluation task解释。

### 14.5 不变内容

- 13维与全部统计值不变；
- 跟车保持60 scenario / 52 log；
- Stage7变道保持固定60场景及`POST_HOC_STANDARDIZED_DESCRIPTIVE_EVALUATION`身份；
- Stage6S-v3保持80 pair / 11 log；
- free-flow speed、lane keeping、lateral gap interaction保持N/A；
- Stage6V联合结论保持不变；不新增训练、仿真、checkpoint、场景或post-hoc主指标。

最终权威输出为
`outputs/final_standardized_bdd_style_report_card_v1/final_standardized_bdd_style_report_card_zh.md`，状态：

`FINAL_STANDARDIZED_BDD_REPORTING_SYSTEM_FROZEN`

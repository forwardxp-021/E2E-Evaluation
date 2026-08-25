# 新训练模型后对比试验：统一 BDD Style Report Card

> Schema：`unified_bdd_reporting_schema_v1`
> 状态：`FROZEN_UNIFIED_BDD_POSTTRAINING_REPORT_COMPLETE`
> 生成方式：只读取已冻结结果；不训练、不仿真、不读取embedding、不重算BDD/MMD。
> 全部semantic delta为 **Target − Reference**。

## 1. 读表前必须区分的三件事

- **表A 行为漂移画像**回答Target相对Reference的行为变化和方向。本次采用old64作为历史主行为报告的固定representation；这不表示old64是最佳检测器。
- **BDD统计量**只在同一行的Reference、Target、task、representation、null下有效。禁止跨representation比较raw MMD²。
- **表B 表示能力评分卡**回答old64/A/B/C/ego13检测已知处置的可靠性，不能反过来当作Target的风格方向报告。

## 2. 表A：BDD Behavior Profile / Style Report Card

| Behavior dimension | Reference→Target | Task / mode | N scenario/log | Rep. | Z_BDD | corrected p / raw p | Semantic Δ (Target−Reference) | Direction | Conclusion |
|---|---|---|---:|---|---:|---|---|---|---|
| `OVR.ALL` 总体行为漂移 | pdm_closed_conservative_v1 → pdm_closed_assertive_v1 | Stage7 locked 5-task confirmation / paired | 310/257 | old64 | N/A | N/A | mean speed +1.281 m/s; RMS accel +0.235 m/s² | TARGET_HIGHER_SPEED_AND_LONGITUDINAL_EXCITATION; overall=MIXED | 总体行为分布显著漂移；不表示安全性、优劣或单一风格标签。 |
| `LON.FREE_FLOW_SPEED` 自由流速度 | N/A → N/A | fixed taxonomy row / N/A | N/A/N/A | N/A | N/A | N/A | N/A | N/A | 没有冻结的纯自由流BDD及绑定的自由流semantic delta。 |
| `LON.ACCEL_DECEL` 纵向加速/减速 | pdm_closed_longitudinal_conservative_v2 → pdm_closed_longitudinal_assertive_v2 | Stage6J/K pure-longitudinal dose100 overall / paired | 183/156 | old64 | 9.228485941359448 | 3.999960000399996e-05 | mean speed +0.915 m/s; RMS accel +0.182 m/s² | TARGET_HIGHER_LONGITUDINAL_EXCITATION | 纯纵向处置产生显著总体纵向行为漂移。 |
| `LON.CAR_FOLLOWING` 跟车行为 | pdm_closed_longitudinal_conservative_v2 → pdm_closed_longitudinal_assertive_v2 | Stage6J/K following_interaction dose100 / paired | 60/52 | old64 | 5.63432382209787 | 0.006899931000689993 | following mean speed +0.917 m/s; RMS accel +0.234 m/s² | TARGET_CLOSER_OR_MORE_ACTIVE_FOLLOWING (speed/accel only) | 跟车slice BDD显著；前车间距/THW方向不稳定，不能把本行解释为稳定更近。 |
| `LON.CLOSING_RESPONSE` 逼近前车响应 | pdm_closed_interaction_long_headway_v2 → pdm_closed_interaction_short_headway_v2 | Stage6S-v3 following interaction confirmation / paired | 80/11 | old64 | 27.976397758226604 | N/A | delta_mean_accel_during_closing_mps2 +0.085 m/s² | TARGET_HIGHER_CLOSING_ACCELERATION | 逼近阶段维持更多加速度。 |
| `INT.LONG_FOLLOWING` 纵向跟车交互响应 | pdm_closed_interaction_long_headway_v2 → pdm_closed_interaction_short_headway_v2 | Stage6S-v3 following interaction confirmation / paired | 80/11 | old64 | 27.976397758226604 | N/A | delta_mean_accel_during_following_pressure_mps2 +0.085 m/s² | TARGET_HIGHER_FOLLOWING_PRESSURE_ACCELERATION | 跟车压力阶段维持更多加速度。 |
| `LON.COMFORT` 纵向平顺性 | pdm_closed_longitudinal_conservative_v2 → pdm_closed_longitudinal_assertive_v2 | Stage6J/K pure-longitudinal dose100 overall (shared parent) / paired | 183/156 | old64 | 9.228485941359448 | 3.999960000399996e-05 | RMS jerk +0.228 m/s³ | TARGET_HIGHER_LONGITUDINAL_JERK | 与LON.ACCEL_DECEL共享parent BDD；不是独立平顺性BDD检验。 |
| `LAT.LANE_KEEPING` 车道保持 | N/A → N/A | fixed taxonomy row / N/A | N/A/N/A | N/A | N/A | N/A | N/A | N/A | 现有冻结库存没有精确lane-keeping BDD和语义增量。 |
| `LAT.LANE_CHANGE` 变道行为 | pdm_closed_conservative_v1 → pdm_closed_assertive_v1 | Stage7 lane_change / paired | 60/N/A (legacy field not archived) | old64 | N/A | 0.0003599964000359 | N/A | N/A | 变道场景slice BDD显著，但无冻结task级方向。 |
| `LAT.DYNAMICS` 横向动态 | pdm_closed_conservative_v1 → pdm_closed_assertive_v1 | Stage7 high_motion_dynamics / paired | 60/N/A (legacy field not archived) | old64 | N/A | 0.0004199958000419 | N/A | N/A | 混合高运动slice显著，不能称纯横向BDD。 |
| `INT.MERGE_YIELD_CUTIN` 汇入/让行/切入响应 | pdm_closed_conservative_v1 → pdm_closed_assertive_v1 | Stage7 dense_or_vulnerable_interaction / paired | 63/N/A (legacy field not archived) | old64 | N/A | 0.0025799742002579 | N/A | N/A | 仅broad dense/vulnerable interaction proxy，不能确认具体汇入/让行/切入。 |
| `INT.FRONT_GAP_THW` 前车间距/车头时距交互 | pdm_closed_interaction_long_headway_v2 → pdm_closed_interaction_short_headway_v2 | Stage6S-v3 following interaction confirmation / paired | 80/11 | old64 | 27.976397758226604 | N/A | median front gap -4.202 m; finite THW -2.670 s | TARGET_SHORTER_FRONT_GAP_AND_FINITE_THW | interaction mechanism与BDD均有冻结证据；THW排除sentinel/cap。 |
| `INT.LATERAL_GAP` 横向间隙接受/横向交互 | N/A → N/A | fixed taxonomy row / N/A | N/A/N/A | N/A | N/A | N/A | N/A | N/A | 没有冻结的横向gap acceptance BDD及同slice semantic delta。 |


说明：`N/A`和`EVIDENCE_GAP_*`表示尚无冻结证据，**不是没有行为差异**。Stage7历史任务行未归档null q95/Z，不能事后由其他representation或其他task补填。

## 3. 表B：BDD Evaluator / Representation Scorecard

| Representation | Pure-longitudinal paired | Following paired | Interaction confirmation | n=400 detection | A/A FPR | detection−FPR | Capability conclusion |
|---|---|---|---|---:|---:|---:|---|
| old64 | overall 4/4; task×dose 7/12; MDD=0.25; median Z=7.539 | Holm pass 2/4 dose cells | Z=27.976; detected=True | 66.5% | 5.0% | 61.5% | 历史baseline；release-level unpaired detection不足。 |
| A | overall 4/4; task×dose 7/12; MDD=0.25; median Z=8.630 | Holm pass 4/4 dose cells | Z=26.454; detected=True | 90.5% | 3.0% | 87.5% | 动态数据修复候选；release detection改善，但整体门禁不通过。 |
| B | overall 3/4; task×dose 2/12; MDD=0.5; median Z=6.015 | Holm pass 1/4 dose cells | Z=30.603; detected=True | 100.0% | 5.0% | 95.0% | 当前最简单、最强release-level learned工程候选；不是universal/final validated representation。 |
| C | overall 3/4; task×dose 2/12; MDD=0.5; median Z=5.189 | Holm pass 1/4 dose cells | Z=28.955; detected=True; full−neighbor-zero ΔZ=-7.852, CI=[-33.393, 29.219], pass=False | 99.5% | 6.5% | 93.0% | release-level signal强，但未证明full-context相对neighbor-zero的增量interaction信息。 |
| ego13 | overall 4/4; task×dose 12/12; MDD=0.25; median Z=21.115 | Holm pass 4/4 dose cells | Z=35.905; detected=True | 100.0% | 2.0% | 98.0% | controlled-longitudinal诊断参考；不是完整context style模型。 |


固定解释：Stage6P是context-balanced、log-disjoint unpaired release监测，n=400；A/A calibration独立完成。Stage6J/K与Stage6S-v3是paired条件。B/C的release-level提升主要由标准化signal增强驱动，不能用“raw MMD²更大”解释。

## 4. 新训练模型后比较：直接结论

1. **release-level longitudinal drift**：old64的n=400 detection为66.5%，A/B/C为90.5%/100.0%/99.5%，对应FPR为3.0%/5.0%/6.5%。B是当前最简单且最强的learned release-level工程候选。
2. **controlled paired longitudinal**：ego13仍最强（4/4 overall、12/12 task×dose）；A保持old64级别，B/C为3/4 overall、2/12 task×dose。因此A/B/C不是全局、最终验证representation。
3. **interaction**：Stage6S-v3的轨迹机制门禁通过；但是C full-context相对C neighbor-zero的ΔZ为−7.852，log-cluster 95% CI为[−33.393, 29.219]，没有证明增量interaction信息。
4. **最终模型判断不变**：`NO_ABC_CANDIDATE_QUALIFIES_UNDER_PRE_FROZEN_RULE`。本报告只统一表达，绝不改变冻结结论。

## 5. 固定证据缺口

| Dimension | Behavior profile | BDD | Semantic direction | Representation comparison | Reason |
|---|---|---|---|---|---|
| `OVR.ALL` 总体行为漂移 | AVAILABLE | AVAILABLE | AVAILABLE | EVIDENCE_GAP_REPRESENTATION_NOT_EVALUATED | — |
| `LON.FREE_FLOW_SPEED` 自由流速度 | EVIDENCE_GAP_BDD_NOT_COMPUTED | EVIDENCE_GAP_BDD_NOT_COMPUTED | EVIDENCE_GAP_SEMANTIC_DELTA_NOT_COMPUTED | EVIDENCE_GAP_REPRESENTATION_NOT_EVALUATED | EVIDENCE_GAP_BDD_NOT_COMPUTED |
| `LON.ACCEL_DECEL` 纵向加速/减速 | AVAILABLE | AVAILABLE | AVAILABLE | AVAILABLE_PRIMARY_ONLY | — |
| `LON.CAR_FOLLOWING` 跟车行为 | AVAILABLE | AVAILABLE | AVAILABLE | AVAILABLE_PRIMARY_ONLY | — |
| `LON.CLOSING_RESPONSE` 逼近前车响应 | AVAILABLE | AVAILABLE | AVAILABLE | EVIDENCE_GAP_REPRESENTATION_NOT_EVALUATED | — |
| `LON.COMFORT` 纵向平顺性 | PROXY_ONLY_NOT_CONFIRMATORY | AVAILABLE | AVAILABLE | EVIDENCE_GAP_REPRESENTATION_NOT_EVALUATED | PROXY_ONLY_NOT_CONFIRMATORY |
| `LAT.LANE_KEEPING` 车道保持 | EVIDENCE_GAP_BDD_NOT_COMPUTED | EVIDENCE_GAP_BDD_NOT_COMPUTED | EVIDENCE_GAP_SEMANTIC_DELTA_NOT_COMPUTED | EVIDENCE_GAP_REPRESENTATION_NOT_EVALUATED | EVIDENCE_GAP_BDD_NOT_COMPUTED |
| `LAT.LANE_CHANGE` 变道行为 | AVAILABLE | AVAILABLE | EVIDENCE_GAP_SEMANTIC_DELTA_NOT_COMPUTED | EVIDENCE_GAP_REPRESENTATION_NOT_EVALUATED | EVIDENCE_GAP_SEMANTIC_DELTA_NOT_COMPUTED |
| `LAT.DYNAMICS` 横向动态 | PROXY_ONLY_NOT_CONFIRMATORY | AVAILABLE | EVIDENCE_GAP_SEMANTIC_DELTA_NOT_COMPUTED | EVIDENCE_GAP_REPRESENTATION_NOT_EVALUATED | EVIDENCE_GAP_SEMANTIC_DELTA_NOT_COMPUTED |
| `INT.FRONT_GAP_THW` 前车间距/车头时距交互 | AVAILABLE | AVAILABLE | AVAILABLE | AVAILABLE_PRIMARY_ONLY | — |
| `INT.LONG_FOLLOWING` 纵向跟车交互响应 | AVAILABLE | AVAILABLE | AVAILABLE | AVAILABLE_PRIMARY_ONLY | — |
| `INT.LATERAL_GAP` 横向间隙接受/横向交互 | EVIDENCE_GAP_SEMANTIC_DELTA_NOT_COMPUTED | EVIDENCE_GAP_BDD_NOT_COMPUTED | EVIDENCE_GAP_SEMANTIC_DELTA_NOT_COMPUTED | EVIDENCE_GAP_REPRESENTATION_NOT_EVALUATED | EVIDENCE_GAP_SEMANTIC_DELTA_NOT_COMPUTED |
| `INT.MERGE_YIELD_CUTIN` 汇入/让行/切入响应 | PROXY_ONLY_NOT_CONFIRMATORY | AVAILABLE | EVIDENCE_GAP_SEMANTIC_DELTA_NOT_COMPUTED | EVIDENCE_GAP_REPRESENTATION_NOT_EVALUATED | PROXY_ONLY_NOT_CONFIRMATORY |


完整机器可读表见：

- `behavior_drift_profile.csv`（表A，含raw MMD²、null、p、semantic和provenance字段）
- `representation_scorecard.csv`（表B，不用raw MMD²跨表示排序）
- `evidence_gap_matrix.csv`（13维完整coverage）

`FROZEN_UNIFIED_BDD_POSTTRAINING_REPORT_COMPLETE`

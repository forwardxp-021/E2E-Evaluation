# Stage7L-C：前瞻性 Protocol 与 80 场景 Confirmation Roster 冻结

> Stage7L-C1 consistency amendment：本文件于Stage7L-D启动前、任何confirmation结果产生前补充了76–79 complete case的paired analysis population，以及B的secondary dose/task family归属。原protocol SHA256为`ae4c1a3ea639d12c9d5f257d87b07e3442e4b22f11c199e40d14f8dab407d125`。roster、dose、eligibility、mechanism/nuisance/safety gate、checkpoint、paired null和Primary endpoint科学定义均未改变。

## 1. Scientific question

本协议检验：在80个全新、完全预处理筛选、动态交通净空的相同场景上，Sharp lateral execution（dose100）相对于 Gentle lateral execution（dose0）是否产生可验证的**纯横向执行处置**；只有运动学、安全和纵向 nuisance 门禁通过后，才允许读取 Primary Representation B 的 paired BDD。

这是 official nuPlan closed-loop、固定 replay background 下的受控验证，不是实际道路版本释放、reactive traffic、换道决策策略或安全优越性验证。

## 2. Treatment definition

冻结五档 lateral transition length：`60.0 / 58.5 / 57.0 / 55.5 / 54.0 m`，对应`dose0/25/50/75/100`。主对照是`dose100 − dose0`；Behavior Reference 是 Gentle Lateral Execution（dose0），Target 是 Sharp Lateral Execution（dose100）。

固定：trigger `s_route=12.0 m`、planner horizon `0.4 s`、采样`0.1 s`、场景 horizon `15 s`、target speed `5.0 m/s`、accel limit `1.0 m/s²`、background `closed_loop_nonreactive_agents`。dose axis 仅为 lateral transition length；不允许在后续改变。

## 3. Pure-lateral causal boundary

处置只作用于横向轨迹生成通道。canonical longitudinal route progress、初始状态、scenario、source/target lane、direction、trigger、background 和所有纵向控制参数均固定。realized closed-loop 纵向量可能存在很小的数值或间接差异，因此以预冻结 nuisance gate 检验，而不把“代码共享进度”误写为“纵向绝对相同”。

## 4. Eligibility

候选必须是 Pittsburgh、native source lane 与 native adjacent target lane、route compatible、official runnable、初始 target-lane object gap≥15 m，且 source/target reference 能覆盖 canonical progress through 15.4 s；同时保留 Stage7L-B 的 map/geometry exclusions。

动态门禁固定使用 Stage7L-B2：15 s、0.1 s grid、原始`lidar_box` replay track、最大插值间隔0.25 s且不外推；trigger 前 source-only、transition family 中 source→target common envelope、transition+settling 后 target-only；ego footprint `5.0×2.0 m`、纵向/横向 buffer `3.0/0.5 m`。该规则对所有dose共同，不读取 rollout 或结果。

## 5. Sample size

冻结`N=80 scenarios × 5 doses = 400 official rollouts`，不扩展至100。Pool B共152个预处理候选，80是先前设计的正式规模，同时保留未使用候选用于未来独立研究，而非本确认集运行失败后的替换。

## 6. Roster selection

仅以 Pool B eligibility、direction、log、map/route geometry、source/target curvature、lane width、initial speed、reference remaining distance、replay traffic density 和 roadblock 使用确定性选择。冻结 seed 为`620271`。

选择器按左/右分层执行 seeded farthest-point geometry coverage，并在可行时优先新 log。方向配额固定为`15 left + 65 right`。Pool B 的19个left仅来自14个log，因此15-left必然出现一次预先记录的 log 重用；right 则优先不与已选left log重用。selection trace 保存全部152个候选的rank、stratum、选中状态和原因，可由 Pool B+config+seed 重放。

## 7. Mechanism endpoints

主语义机制：`lane_change_duration_s`。核心物理机制：`rms_lateral_accel_mps2`、`peak_yaw_rate_radps`。次要指标：peak lateral acceleration、RMS yaw rate、RMS lateral jerk、settling time 和 final target center offset。

所有 semantic delta 为`Target − Behavior Reference = dose100 − dose0`。预期方向分别为：duration `<0`、RMS lateral acceleration `>0`、peak yaw rate `>0`。

机制delta与nuisance summary的95% CI只作不确定性展示，固定使用以`log_name`为cluster的percentile bootstrap、10,000次重采样、seed=`620272`。该CI不参与mechanism/nuisance gate；原有paired median方向和directional consistency门槛保持不变。

## 8. Mechanism pass criteria

在80个冻结场景的主对照上：

- duration 的 paired median delta 必须`<0`，方向一致率≥70%；
- RMS lateral acceleration 的 paired median delta 必须`>0`，方向一致率≥80%；
- peak yaw rate 的 paired median delta 必须`>0`，方向一致率≥80%。

这些门槛来自 Stage7L-B 安全dose development 的非贴边结果。次要 dose curve 为 dose25/50/75/100 各自相对dose0；不要求每档严格显著或严格单调，只报告总体横向激励趋势。

## 9. Nuisance gates

正式主对照为dose100−dose0。`paired median`和`p90`均必须不超过：mean speed `0.02 m/s`、RMS longitudinal acceleration `0.05 m/s²`、RMS longitudinal jerk `0.10 m/s³`、route progress `0.25 m`。max仅作诊断，避免单点异常决定整个门禁。

## 10. Safety gates

在全部80个冻结场景中，official success与lane-change completion均须≥95%，off-road与responsible collision均须≤5%。报告collision、route failure、invalid 和 incomplete；不得以成功换道、低横向误差或任何后处理结果重新定义BDD population。

## 11. Failure policy

Infrastructure/runtime failure（loader、DB、Hydra、official scene construction）不允许替换token。若official completed少于`76/80`，状态为`STAGE7L_D_CONFIRMATION_EXECUTION_INSUFFICIENT`并停止。

Treatment/trajectory outcome failure（collision、off-road、invalid、incomplete、route failure）保留为结果，绝不删除。即使完成数为76–79，仍保留完整80场景冻结population并按预注册缺失/失败策略报告。

必须区分两个人口：`N_design=80`始终用于protocol identity、execution/safety denominator、missing/runtime failure和no-replacement审计；某个`doseX vs dose0`的BDD analysis population则是冻结80场景中同时具有完整dose0、完整doseX且representation input存在并可合法构造的全部pair，记为`N_pair(doseX)`。不得replacement，不得依据collision、off-road、lane-change incomplete、BDD或embedding表现删样。只要representation input技术上存在，treatment outcome failure仍进入BDD；只有input事实缺失或无法合法构造才是non-analyzable。

结果必须同时报告`N_design=80`、`N_complete_all_five_doses`、`N_pair(dose25/50/75/100)`，以及`infrastructure/runtime`、`treatment outcome`、`invalid/incomplete`、`other pre-frozen category`四类missing reason。official success、completion、responsible collision和off-road仍以原冻结80场景规则为分母，不能改用BDD pair数。

## 12. Representation lock

Primary learned representation固定为`B, seed=3407`；historical baseline为`old64`；ego13为诊断运动学参考；A与C为次要representation。old64、A/B/C primary checkpoint及ego13定义/scaler实现的路径和SHA均写入配置与盲测授权，禁止换seed、换epoch、换checkpoint。

ego13即使在本横向kinematic treatment中更敏感，也仅说明其在该处置上的within-null sensitivity高；不能解释为全局最佳representation或neighbor/context无价值。

## 13. Primary BDD

只有 Stage7L-D mechanism/nuisance/safety gates全数通过且official completed≥76/80时，才解锁 Stage7L-E。Primary endpoint的科学定义保持：`B seed3407; dose100 vs dose0; paired; LAT.LANE_CHANGE`。其design population是全部80个冻结场景；analysis population是其中全部完整dose100-vs-dose0 pair，不替换、不按outcome删除，且必须`N_pair(dose100)≥76`。若少于76，状态固定为`STAGE7L_E_PRIMARY_BDD_INSUFFICIENT_COMPLETE_PAIRS`，不得声明Primary成功。

统计采用 same-scenario pair-label-swap、100,000 swaps、plus-one p-value、representation-specific RBF median-heuristic bandwidth和representation-specific paired-null q95。输出 raw MMD²、null mean/SD/q95、BDD/q95、Z_BDD及plus-one p。主成功标准为预先指定、未经校正的`p<0.05`；只有一个primary endpoint，因此不作Holm。

## 14. Secondary BDD

次要矩阵包含old64、A、B、C、ego13全部五个representation；对照为dose25/50/75/100各自相对dose0；task为`LAT.LANE_CHANGE`和`LAT.DYNAMICS`。B必须输出完整dose curve；唯一Primary格除外，其余B格均进入secondary family。禁止跨representation排序raw MMD²；仅比较各自null标准化的BDD/q95、Z_BDD、detection、minimum detectable dose与覆盖度。

现有Stage7实现将两项定义为不同的pre-treatment task scope：`LAT.LANE_CHANGE`对应official scenario types `changing_lane_to_left/right`；`LAT.DYNAMICS`对应`high_lateral_acceleration/high_magnitude_speed/medium_magnitude_speed`，后者仍须标为mixed proxy，不能写成pure-lateral因果证据。Stage7L-E只有在两者确实形成不同的pre-treatment mask并记录不同mask SHA时，才按两个独立test计数；若实际复用同一个parent BDD，必须标记`† shared parent BDD`且不得重复计入独立检验。

## 15. Multiplicity

唯一primary endpoint使用未校正、预先指定的raw/plus-one p，并固定标记`PRIMARY_NOT_PART_OF_SECONDARY_HOLM_FAMILY`。理论矩阵为`5 representations × 4 dose contrasts × 2 independently computed task views = 40`；排除`B × dose100 × LAT.LANE_CHANGE`这一Primary格后，冻结为单一39-test secondary Holm family。Primary不得再次进入Holm，也不得用adjusted p改变Primary结论。若一个task-level BDD映射多个semantic row，必须标记`† shared parent BDD`，不得当成多次独立检验。

## 16. Reporting

业务页为 Behavior Drift / Style Report Card：明确 Behavior Reference、Target、Null Reference、Primary Representation、BDD/q95、Z、显著性、semantic delta及方向。第二页为 Representation Qualification Matrix：说明old64/A/B/C/ego13在固定treatment下的null-standardized sensitivity。

本确认集主要更新`LAT.LANE_CHANGE`与`LAT.DYNAMICS`；`INT.LATERAL_GAP`默认N/A，不能因表格完整性强行填充。

## 17. Claim boundary

若成功，唯一允许的结论是：在official nuPlan closed-loop、fixed replay background和fresh pre-treatment dynamic-clean lane-change opportunities上，prospectively frozen pure-lateral execution treatment产生预期横向运动学变化，并可由BDD检测。不得扩展到fleet、reactive traffic、换道决策、safety superiority、universal representation 或任意ODD。

## 18. No-replacement/no-retuning 与 D/E unlock

冻结后不得换scenario/log/比例、dose、trigger、eligibility、buffer、mechanism primary、nuisance threshold、checkpoint、seed、null、primary endpoint或原failure policy；C1只把原有minimum-complete规则对应的analysis population闭合，不因不显著、B不如ego13或次要dose失败而重训、调kernel或再次确认。

Stage7L-C只授权 Stage7L-D 的400条planner-level rollout与机制/安全审计。只有脚本判定D通过既定门禁才输出`STAGE7L_E_REPRESENTATION_EVALUATION_UNLOCKED`；否则输出`STAGE7L_E_REPRESENTATION_EVALUATION_NOT_UNLOCKED`并停止。Stage7L-C本身不运行任何rollout。

# Stage 6R/6S 阶段结果（中文）

## Stage 6R pilot结论

首次Dynamic Interaction Builder v2 pilot的自动门禁与原始TFRecord重建曾通过，但随后对20-case overview做真实视觉检查时发现，部分`left_front`轨迹近乎垂直横穿ego轨迹。根因是旧lane解析只保存neighbor lane id，丢弃Waymo proto的`self_start/end_index`与`neighbor_start/end_index`局部邻接范围。原“pilot通过”及其full51授权已经显式标记为`SUPERSEDED`，首轮full51分段构建已经中止，不得续跑或finalize。

- old builder同源场景：10268窗口；五个slot的`neighbor_slot_ids`均为`[N,5]`整窗固定，front identity switch结构上不可表达。
- v2：10488窗口、1512 scenario。front lead entry/exit/intermittent/identity-switch分别为3895/4014/5289/865；旧builder相应为9/147/0/0。
- v2五slot逐帧覆盖率：front 32.58%、left-front 19.08%、left-rear 20.96%、right-front 19.96%、right-rear 22.23%。
- 五slot identity switch rate为1.95%–3.20%；跨identity accel/yaw-rate derivative违规为0。
- 局部ego坐标方向正确率为91.11%–99.92%。该指标在弯道/交叉口只作诊断；20个固定seed case回到原始TFRecord后，使用ego lane、candidate lane、slot lane与delta-s复核，每slot 4例全部通过，共499 lane-aware帧与21 geometric-fallback帧，0 topology失败、0重建track-id不一致。
- 新longitudinal raw `|q99|`约为speed 21.95 m/s、accel 6.25 m/s²、jerk 77.53 m/s³；winsorized+median/IQR normalized `|q999|`为2.02/4.87/4.58，明显低于pilot冻结上界25。

首轮统计说明逐帧分配能够恢复entry、exit、intermittent、identity switch与transition，但不能据此证明五槽语义可靠。当前已增加局部邻接区间过滤，并把门禁拆为自动统计、原始TFRecord拓扑重建、独立视觉语义检查三层；必须完成修复版3-file pilot后才能重新决定是否重建full51。旧full51与Stage6O v1未修改。

修复版`semantic_strict_multirelation` pilot现已三层通过：强制`lane_aware_only`且几何fallback为0；同一位置存在多条合法邻接关系时全部纳入候选择优。10488窗口中front entry/exit/intermittent/front-switch为3720/3845/4964/681，五槽switch rate为1.26%–2.65%，跨identity导数违规为0。20例固定seed原始TFRecord重建0不一致、0 topology失败；逐图视觉检查未再发现垂直横穿误配。该修复版只授权独立目录重建full51，仍不授权训练。

## Stage 6R full51与Stage6O-v2结论

修复版随后在独立目录完成51/51个原始TFRecord重建：共24872个scenario、168700个窗口、36个shard，train/val/test为135046/16870/16784，scenario跨split重叠为0。所有part均记录`lane_aware_only`、关闭几何邻道推断且动态汇总校验通过；旧full51和Stage6O v1均未覆盖。

Stage6O-v2正式状态为`FROZEN_READY_FOR_INTERACTION_AWARE_V2_PREPARATION`，全部预冻结门禁通过：

- train lead entry/exit/intermittent/front identity switch为47335/48074/63415/8294；free-flow→closing→following与following→free-flow为15175/50236。旧Stage6O v1的intermittent=0已被明确证明是builder结构性过滤，而不是Waymo原始数据缺失。
- front、left-front、left-rear、right-front、right-rear的帧覆盖率为28.73%/17.78%/16.77%/17.65%/17.19%，窗口占用率为65.64%/42.61%/42.01%/42.07%/42.58%。五槽switch rate为1.29%/2.09%/2.48%/2.12%/2.64%，低于20%冻结上界。
- finite、shape和跨identity导数违规均为0；旧Stage6O v1 SHA256仍为`4175054bbcf38d604ff0bab5bda77233a066c475c5e19335b0d219f00f1d164e`。
- 新纵向逐帧raw绝对值q99为speed 21.64 m/s、accel 6.20 m/s²、jerk 76.30 m/s³，normalized最大绝对值4.74。使用相同窗口RMS口径，accel median由旧2.72降至1.48 m/s²；jerk median/q90由旧42.82/100.80降至15.51/28.47 m/s³。这说明差分噪声和长尾明显减弱；原始监督仍单独保存，不能把平滑后的较小数值解释为真实瞬态全部消失。

五槽语义的结论应限定为“在严格Waymo局部lane-neighbor关系、20例分层视觉抽查和全量结构门禁下可靠”，而不是声称168700个窗口逐帧都经过人工真值标注。语义正确性优先于track连续性，缺少可信拓扑时宁可留空；这是当前可靠性边界。

## Stage 6S结论

24个same-scenario pair、48条official PDM rollout全部成功，0失败；统一轨迹视图和逐帧lane-aware context均通过结构审计。front覆盖46.53%，23/24 pair具有双planner有效front；map-name解析率100%，fallback frame rate 11.34%。

预冻结机制门禁结果为`PDM_INTERACTION_BENCHMARK_LIMITATION`：

- short-headway minus long-headway平均速度差0.232 m/s，满足绝对值≤1.0 m/s；
- RMS加速度差0.108 m/s²，满足绝对值≤0.75 m/s²；
- front gap差−1.208 m，未达到≤−2.0 m；
- closing期加速度响应差0.073 m/s²；预冻结配置没有给它设置数值阈值，因此只作诊断，不能计入确认性门禁；
- 原始mean THW差−106.85 s命中方向门禁，但THW含999 s cap/sentinel，绝对量级不宜物理解读。排除cap后的pair-median诊断仍方向为负，但这是结果后稳健性说明，不替代冻结门禁。

因此当前PDM配置实现了“ego整体速度/加速度变化较小”，但没有同时实现至少两项清晰interaction-response机制差异。按冻结规则不继续按结果调planner，也不读取embedding/BDD为planner调参。该结果是planner benchmark limitation，不是Waymo模型失败证据。

分层诊断解释了限制来源：4个`following_lane_with_slow_lead`的双planner front覆盖约70%，front-gap差−1.52 m，closing响应差0.143 m/s²；20个`near_long_vehicle`的front覆盖只有约40%–44%，front-gap差−1.14 m，closing响应差0.055 m/s²且closing暴露仅约16%–19%。也就是说优先类型池中真正slow-lead场景只有4个，其余20个`near_long_vehicle`多数没有形成持续、强处置的跟驰暴露。该分层是结果后根因诊断，不授权从当前结果中重新挑选“效果好”的场景作确认性结论。

## 当前训练授权状态

Waymo数据侧已经具备“准备Interaction-aware v2训练”的条件，但本阶段没有启动checkpoint训练，也没有扩大Waymo。整体论文实验侧仍不建议立即启动正式训练：Stage6S表明当前冻结的PDM/scenario pool未构造出确认级interaction-dominant benchmark。下一步应先决定把该结果作为planner limitation接受，还是另行预注册新的planner/场景生成方案；不能用本批结果后调参再冒充确认性benchmark。Stage6O v1永久保持BLOCKED，Stage5D-balanced-v2 checkpoint不得覆盖。

## 对五个研究问题的直接回答

1. **是否恢复动态interaction信息：是。** intermittent从旧正式数据的0恢复到train 63415，并同时恢复entry、exit、identity switch与两类状态转换；数值远超预冻结5000门槛。
2. **五个semantic slots是否可靠：在当前审计边界内是。** 五槽均逐帧分配、0几何fallback、20例每槽4例视觉通过、全量switch rate低且跨identity导数违规为0；但这不是全量人工标注精度证明。
3. **纵向supervision物理质量是否改善：明显改善。** 同口径窗口RMS accel与jerk的中位数及jerk q90显著下降，finite/shape全通过，并完成train-only winsorize与median/IQR；仍应在后续训练中监控平滑是否削弱真实急剧响应。
4. **nuPlan能否构造目标benchmark：本次PDM冻结配置没有做到确认级。** 平均速度差较小，但front-gap未通过，只有一个预冻结interaction指标通过；应记录为当前planner/scenario-pool limitation，而不是模型失败或nuPlan普遍不可能。
5. **是否具备启动训练条件：数据侧已具备，完整实验侧尚未。** Stage6O-v2允许进入训练准备，但正式训练应等新的预注册benchmark方案或对PDM limitation的明确研究决策，并需用户另行授权。

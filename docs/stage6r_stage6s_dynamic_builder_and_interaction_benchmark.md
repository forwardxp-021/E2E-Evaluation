# Stage 6R/6S：Dynamic Interaction Builder v2 与 interaction-dominant benchmark

## 研究边界

本阶段不训练新 checkpoint、不扩大 Waymo source、不读取 embedding/BDD 调参，也不覆盖旧 full51、Stage6O v1 或 Stage5D-balanced-v2 checkpoint。论文定义保持：`behavior style = ego response conditioned on traffic / interaction context`。

## Stage 6R：Waymo Dynamic Interaction Builder v2

Issue：#259。

旧 `build_waymo_5neighbor_context_dataset.py` 在每个80帧窗口只调用一次 `assign_stage5d_slots`，因此 front、left_front、left_rear、right_front、right_rear 五个slot均按参考帧静态绑定track；邻车还复用ego的`min_valid_ratio=0.8`整窗过滤。该实现无法表达真实的front identity switch，并结构性压低entry、exit与intermittent窗口。

v2由 `tools/build_waymo_dynamic_interaction_dataset_v2.py` 实现：

- 每一帧重新进行五slot semantic assignment；semantic correctness优先于track continuity。
- 保留Waymo lane neighbor relation的`self_start/end_index`与`neighbor_start/end_index`，只在局部有效区间承认相邻车道。
- 强制`lane_aware_only`；禁止几何fallback与几何相邻车道猜测，缺少可信拓扑时slot为空。
- 输出 `slot_track_id_timeline.npy`、`slot_valid_mask.npy`、`slot_identity_switch_mask.npy`、`slot_derivative_valid_mask.npy`。
- identity切换时将neighbor accel/yaw-rate置零并把derivative-valid置false，禁止跨agent差分。
- ego仍使用独立的0.8有效率门槛；neighbor只要求当前帧有效，不再要求整窗>=0.8。
- 新纵向监督严格按 `speed -> median5 -> accel/jerk -> train q01/q99 winsorize -> train median/IQR` 生成；旧33D监督和原full51保持不变。
- full51允许按TFRecord范围并行构建，但最终必须由 `stage6r_finalize_dynamic_full51.py` 使用全体train split重新统一归一化，不能使用part局部统计作为正式训练统计。

Pilot先使用TFRecord 00000–00002。自动审计比较旧builder同源scenario与v2的slot coverage、front有效率、动态事件、switch rate、finite/shape、跨identity导数和纵向target范围；固定seed抽取20个典型case。门禁明确分为自动统计、原始TFRecord拓扑重建、实际查看overview的独立视觉语义检查三层。首次pre-fix pilot因视觉检查发现横穿车道被当作侧向slot而失效，首轮full51授权撤销。只有修复版三层门禁全部通过，才允许重建full51。

full51完成后由 `stage6o_v2_freeze_training_readiness.py` 建立独立Stage6O-v2。旧Stage6O v1的SHA256必须保持不变；intermittent train计数仍使用预冻结门槛5000，不允许事后降低。

实际full51已完成：51个TFRecord、24872个scenario、168700个窗口、36个shard；scenario跨split重叠为0。Stage6O-v2全部门禁通过，train intermittent为63415，五槽switch rate为1.29%–2.64%，finite/shape/跨identity导数违规均为0。新纵向监督的窗口RMS accel median为1.48 m/s²，RMS jerk median/q90为15.51/28.47 m/s³；旧口径对应为2.72和42.82/100.80，物理长尾明显收敛。完整数字与解释边界见`docs/stage6r_stage6s_results_zh.md`。

## Stage 6S：interaction-dominant nuPlan benchmark

Issue：#260。

冻结24个same-scenario pair，优先使用`following_lane_with_slow_lead`、`following_lane_with_lead`和`near_long_vehicle`。两个PDM planner共享desired-speed schedule、fallback speed、accel/decel和lateral offsets，只允许以下参数不同：

- short-headway：minimum gap 0.5 m，headway 0.8 s；
- long-headway：minimum gap 2.5 m，headway 2.2 s。

先运行official closed-loop rollout，再构造逐帧lane-aware front context，只看realized trajectory/mechanism：mean speed、RMS accel、THW、front gap、closing与closing期acceleration response。整个过程保持embedding/BDD盲态。如果预冻结门禁失败，结论记录为PDM planner limitation，不按结果重新挑场景或调参。

## 训练授权规则

Stage6R full51与Stage6O-v2已完成，Waymo数据侧只代表“具备准备Interaction-aware v2训练的条件”，不代表本阶段已经授权训练。Stage6S仍为`PDM_INTERACTION_BENCHMARK_LIMITATION`，所以整体实验侧应先预注册新的benchmark方案或明确接受该限制。正式训练需用户另行确认，且不得覆盖Stage5D-balanced-v2 checkpoint。

## Stage6S-v2 补充（Issue #261，2026-08-12）

Stage6S-v1的24场景结论保留为历史limitation，不覆盖。Stage6S-v2改从扩大nuPlan inventory中仅用
pre-treatment信息筛选，24-pair development已在mean-speed/RMS-accel小差异约束下建立front-gap与
finite-THW两项稳定机制，并冻结独立的80-pair、15-log confirmation roster。新roster与development
log/token和Stage6S-v1 token均无重叠，尚未启动rollout或任何embedding/BDD分析。权威结果与边界见
`docs/stage6s_v2_interaction_benchmark_confirmation_report_zh.md`。

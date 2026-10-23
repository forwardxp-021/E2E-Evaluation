# TSB-Coast P0 — nuPlan最小版本化实现设计 v1

状态：`SOURCE_REVIEW_FEASIBLE_IMPLEMENTATION_PENDING`。本轮没有实现新planner/generator/analyzer/executor，更没有构建或运行runner。下列新文件名均是**拟议文件，不声称已存在**。

## 生产链审计与最小边界

| 组件 | 已核查实际入口 | 拟议处理 |
|---|---|---|
| 旧TSB generator | `tools/r2_b_controller_aware_generator_v1.py::tsb_controller_aware_acceleration` | 保持旧+1.4不动；新建`tsb_coast_generator_v1.py`和独立family/参数schema，reference非正/速度非增断言 |
| controller-aware planner | `tools/r2_b_controller_aware_planner_v1.py::_tsb_states, _tsb_required_distance, R2BControllerAwarePlannerV1.compute_planner_trajectory` | 新建`tsb_coast_planner_v1.py`，复用已审计route/状态构造原语，禁止继承旧generator隐式分派；支持新Baseline开始时刻，绝不修改旧类 |
| native route | `tools/b1_native_route_infrastructure.py::build_native_route_reference_v1_2, production_route_precheck_v1_2` | 原样复用；按新reference实际有限7.9s距离需求预检，不恢复旧全路由infinity检查，不外推/拼直线/强行fallback |
| 官方runner/stack | `tools/s2r_production_executor.py::build_fresh_arm`及`tools/b1_tsb_qualification_executor.py::complete_production_arm` | 生命周期原理复用；旧build_fresh_arm硬绑旧PARAMETERS/PLANNER/ANALYZER，不能假传新参数后复用旧授权。新建薄binding，独立allowlist/stage/hash和output namespace，runner本体不fork |
| passive recorder | `tools/r2_bj_b0_2_passive_actual_lqr_recorder.py::PassiveActualLQRRecorderV1` | 原样记录79个实际controller transitions，validate_installability/actual-shadow一致性保留；不改变track_trajectory返回值 |
| measurement analyzer | `tools/r1_context_mechanism_core.py::calculate_tsb_option_a, qualify_tsb_pair`及`tools/r1_b2_8_r3_2_post_run_evaluator_dispatcher.py` | 旧函数永久冻结；新建`tsb_coast_mechanism_v1.py`独立schema/dispatcher，不调用旧恢复fraction门槛 |
| F_match | `tools/r1_closed_loop_benchmark_v2_1.py::trajectory_descriptors_timestamp_aware` | 可复用测量公式；新合同参数及附加gate独立版本，不能覆盖`TSB_PRIMARY_F_MATCH_CALIPERS` |
| 官方安全 | `tools/r1_b2_8_r3_1_official_safety_adapter.py::adapt_official_safety`及canonicalizer | 原样复用：官方parquet每arm at-fault collision=0且drivable=true；不自制collision近似替代官方值 |

controller源码：`../nuplan-devkit/nuplan/planning/simulation/controller/tracker/lqr.py`；实际默认config：`../nuplan-devkit/nuplan/planning/script/config/simulation/ego_controller/tracker/lqr_tracker.yaml`。h=.1×10=1s，jerk_penalty=1e−4用于pose到速度拟合，不是新行为资格阈值。本轮hash绑定外部源码和配置，未修改依赖；未来必须核对**resolved runtime config**等于批准版本，默认yaml本身不是未来执行证明。

## 不重新引入主动恢复的实现约束

1. 参考加速度逐点≤0，reference speed全未来序列非增，禁止正overshoot、先加速再减速的逆滤波、以负制动面积后置正补偿。可用非负对称核平滑非正a；先固定kernel/边界和I/M积分再运行。
2. P1前最多两个零运行实现版本，只修数值边界/接口，C2的总剂量、对称性、coast语义和测量阈值不变；不引入新的科学候选。实际每.1s步若边界跨越，按段重叠精确积分而非float条件漏一个step，特别是C1的1.75s边界（本轮仅解析对照）。
3. 保留旧架构的共同precontext原则：absolute<1.1s时双方**整条未来轨迹**使用新Baseline参考，不只是首点相同，以防预瞄泄漏；1.1s后才允许差异。共同未来Baseline可能已预减速，需按固定规则报告，不能把这当新的T第一脚。current-ego第0点位置/航向/速度/整数时间戳精确一致。
4. controller-aware预补偿若必需，仅允许前瞻固定的非正制动边界/单调参考数值实现，不允许持续正恢复、按SESSION调参或换LQR gain。超出C2固定范围需停下新Owner决策，不能在本P1预算中隐式搜索。
5. 不通过直接覆盖actual a、关闭安全引擎、换控制器或修改motion model“保真”；若现有LQR无法形成滑行，报告平台限制。

## 生命周期、授权与可复现性

每arm独立对象/进程/随机状态和新namespace（例如`TSB_COAST_DEV_v1`、`TSB_COAST_Q_v1`，目前未创建科学output）；同SESSION配对身份及共同precontext绑定。执行授权必须精确绑定generator版本、参数合同、mechanism analyzer版本、qualification protocol和execution authorization五项，以及code/config/runtime/roster/precontext/route hashes。

保留六步：RUNNER_COMPLETE→RECORDER_COMPLETE→OFFICIAL_METRICS_COMPLETE→SERIALIZER_COMPLETE→MANIFEST_COMPLETE→ARTIFACT_HASHES_VALIDATED。官方`save_runner_reports`后必须调用`multi_main_callback.on_run_simulation_end()`，由官方MetricFileCallback最终落盘；16份官方metric parquet和1份非空simulation log作为当前栈预期，未来执行前按同源resolved config核对。passive recorder不能代替官方安全，callback存在不能代替工件完整。

Technical-complete与scientific-pass分开；arm完整须80真实帧/80planner记录/79controller transitions及全部SHA。pair完整须双方完整、身份/precontext/route合同一致，再独立判断测量、机制、F_match、安全。原B1授权的stage、run IDs和41attempt总限额**绝不可移用**。

## 闭环风险及先决测试

预瞄混合、pose拟合速度过冲、逐次真实状态重规划、运动模型滞后、.2m/s floor、route有限长度、低速和窗口尾部均是未验证风险。纯函数运动学通过不能验证任何一个闭环风险。P1前最少做monotonic-reference/精确边界/共同future-precontext/新旧dispatcher隔离/测量fixture/生命周期拒绝错误授权等零运行测试；本轮只实现解析部分，不能声称新执行栈ready。

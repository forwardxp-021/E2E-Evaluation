# B1-RP Callback Root Cause v1

旧断言位于 `tools/b1_resume_execution.py::preflight`，检查 `len(arm.callbacks) > 0`。`tools/s2r_production_executor.py::build_official_tsb_arm` 将 `build_simulation_callbacks` 的返回值赋给 `FreshArm.callbacks`；这是共享 callback 列表，不是 simulation 的完整回调图，也不是官方 MetricFileCallback 的所有者。

实际官方实现（源码 SHA 由旧授权依赖绑定、此次再次核对）：

1. `../nuplan-devkit/nuplan/planning/script/builders/simulation_callback_builder.py:60` 有意跳过带状态的 MetricCallback 和 SimulationLogCallback。当前正式配置的共享列表长度因此合法为0。
2. `../nuplan-devkit/nuplan/planning/script/builders/simulation_builder.py:106` 为每个 scenario 创建 MetricCallback；`:113` 创建 SimulationLogCallback；`:126` 将 `MultiCallback(callbacks + stateful_callbacks)` 装入真正的 `Simulation`。这些对象在工厂构建时已绑定，尚未进入 runner 初始化。
3. 真实权威所有者是 `arm.runner.simulation.callback`，其身份必须等于 `arm.simulation.callback`；其两个成员实际是官方 MetricCallback 和 SimulationLogCallback。六份 live-object evidence 均确认这一结构，不能只凭类型名称或历史 parquet 推断。
4. `../nuplan-devkit/nuplan/planning/script/builders/main_callback_builder.py::build_main_multi_callback` 从正式 cfg.main_callback 另外构建 MultiMainCallback。`set_up_common_builder` 已调用其 on_run_simulation_start；`arm.common_builder.multi_main_callback` 持有 MetricFileCallback，负责最终文件整合。
5. MetricCallback 持有 MetricsEngine，其 `_main_save_path` 指向本诊断 namespace 的 raw/metrics；MetricFileCallback 的 `_scenario_metric_paths` 和 `_metric_file_output_path` 必须与该路径、common.output_dir 和 resolved cfg 一致。六项真实对象均通过逐项路径检查。
6. 正式配置实际装配4个main callbacks：TimeCallback、MetricFileCallback、MetricAggregatorCallback、MetricSummaryCallback。所有正常项均有且仅有一个官方 MetricFileCallback，实例方法绑定和主链分派有效。
7. B1-IR 修复的是 runner 后缺失 save_runner_reports 与 main on_run_simulation_end 调用；它与 B1-R 后来新增的错误共享列表断言属于不同层。此次确认 IR 的正式 executor/lifecycle 源码逐字节未变，且其真实主回调所有者仍在。旧预检错误不自动证明生命周期正确；实际对象、路径、源码调用序列和故障阻断才提供此次静态就绪证据。

新检查没有无条件删除旧断言，而是使用 `tools/b1_production_preflight.py::validate_production_callback_binding` 同时检查真实simulation/主回调两个所有者、官方具体类型、方法绑定、engine配置、输出路径、serializer、recorder和未初始化状态。缺失或损坏仍 fail closed。

工程验证还明确区分15个实际 metric builder与16类预期统计产物：官方 DrivableAreaComplianceStatistics 会额外输出 corners_in_drivable_area。检查使用实际 resolved配置的builder名称/具体类型，而不把产物名称误当独立builder。该区别由官方 drivable_area_compliance.py 和 metric_engine.py 源码确认，未调用指标计算。首次开发检查的这一映射假设在源码冻结前修正；它没有消耗预算或触发科学入口。
